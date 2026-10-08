"""
train_semantic_anchor_model.py
================================
Consolidated Semantic-Anchor Emotion Classifier: shared encoder + semantic
anchor branch (Person 1), classifier head + cross-interaction fusion + LACR
+ loss (Person 2), and the training/eval pipeline (Person 3).

Built from the repo's three merged notebooks:
    - Encoder_and_Semantic_Anchors (2).ipynb
    - Classifier_Head_Fusion_and_Loss (2).ipynb
    - training 1.ipynb   (the fuller of the two training notebooks: AMP,
      gradient accumulation, warmup/decay LR schedule, early stopping,
      per-class threshold calibration, held-out masking)

Architecture (matches both whiteboards: LACR runs AFTER fusion, on z):

    text -> SharedEncoder -> h --------------------------+
                                                          |
    28 anchor texts -> SharedEncoder (offline, no grad)  |
                     -> AnchorBank.A [28, 768]            |
                                                          v
    Path 1 (parametric): X = W_c h + b_c          [B, 28]
    Path 2 (interaction): Y_cos = cos(h, A) -> CrossInteractionLayer
                           -> SharedInteractionMLP -> Y    [B, 28]
    Fusion: z = X * seen_mask + Y                          [B, 28]
    LACR:   z_final = LayerNorm(z + Residual(z @ A_corr^T)) [B, 28]

Run:
    python train_semantic_anchor_model.py                # full training run
    python train_semantic_anchor_model.py --smoke-test    # tiny fast sanity check
"""

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers import logging as hf_logging

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
hf_logging.set_verbosity_error()


# =============================================================================
# Labels, definitions, small shared helpers
# =============================================================================

GOEMOTIONS_LABELS: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
NUM_CLASSES: int = len(GOEMOTIONS_LABELS)  # 28
NEUTRAL_IDX: int = GOEMOTIONS_LABELS.index("neutral")

GOEMOTIONS_DEFINITIONS: Dict[str, str] = {
    'admiration': (
        "feeling profound respect, wonder, and high esteem toward a person for exceptional talent, skill, courage, "
        "character, or achievement; typically expressed as praise or awe, such as 'incredible work', 'you are amazing', 'what a legend'."
    ),
    'amusement': (
        "finding something funny and enjoying it: laughter, chuckling, playful wit, jokes, sarcasm taken lightly, "
        "and lighthearted entertainment; typically expressed as 'haha', 'lol', 'that is hilarious', or a witty remark."
    ),
    'anger': (
        "intense rage, fury, hostility, and outrage directed at a person or group that has wronged, insulted, "
        "or provoked the speaker; strong, heated, often aggressive wording such as 'I hate you', 'how dare they', 'this is infuriating'."
    ),
    'annoyance': (
        "mild to moderate irritation, impatience, and exasperation caused by small, repeated, or petty disturbances, "
        "inconveniences, or other people's behavior; low-intensity grumbling such as 'ugh, not again', 'this is so irritating', 'stop doing that'."
    ),
    'approval': (
        "expressing agreement, endorsement, and positive judgment of an idea, decision, opinion, or action; "
        "saying that something is right, fair, or well done, such as 'that makes sense', 'good call', 'I agree', 'fair enough', 'that is the right choice'."
    ),
    'caring': (
        "showing warmth, concern, and kindness toward someone's well-being: offering help, comfort, support, or protection, "
        "and checking on others; typically expressed as 'take care of yourself', 'I am here for you', 'hope you feel better'."
    ),
    'confusion': (
        "not understanding something that is unclear, contradictory, or illogical: puzzlement, being lost, and needing clarification; "
        "typically expressed as 'I don't get it', 'what does that mean?', 'wait, what?', 'this makes no sense to me'."
    ),
    'curiosity': (
        "an eager wish to learn more: asking questions, wanting to explore, investigate, or find out hidden or missing information; "
        "typically expressed as 'I wonder why', 'how does that work?', 'tell me more', 'I would like to know'."
    ),
    'desire': (
        "a strong wish, craving, or longing to have, do, or become something specific; wanting an object, an experience, "
        "an outcome, or a person; typically expressed as 'I really want', 'I wish I could', 'I need this', 'I long for'."
    ),
    'disappointment': (
        "feeling let down and discouraged because an expectation, hope, or promise was not met and the outcome turned out worse than hoped; "
        "a quiet dejection about something that fell short, such as 'I expected more', 'what a letdown', 'that was underwhelming', 'I hoped it would be better'."
    ),
    'disapproval': (
        "a negative judgment that something is wrong, unacceptable, or a bad idea: objecting to or condemning an action, decision, opinion, or standard, "
        "without necessarily being angry; typically expressed as 'that is not okay', 'you should not do that', 'I don't agree', 'that is a terrible idea'."
    ),
    'disgust': (
        "strong revulsion and distaste toward something repulsive, filthy, rotten, vile, or morally sickening; "
        "a physical or moral urge to recoil, typically expressed as 'that is gross', 'ew', 'disgusting', 'it makes me sick'."
    ),
    'embarrassment': (
        "feeling self-conscious, awkward, or ashamed in front of others after a blunder, slip, or exposure: blushing, wanting to hide, "
        "cringing at a public mistake; typically expressed as 'how awkward', 'I want to disappear', 'that was so embarrassing', 'I feel so silly'."
    ),
    'excitement': (
        "high-energy enthusiasm and eager anticipation about something thrilling that is happening or about to happen; "
        "can't-wait feelings, thrill, and exhilaration, typically expressed as 'I can't wait', 'so pumped', 'this is going to be awesome!', with exclamation marks."
    ),
    'fear': (
        "feeling frightened or threatened by danger, harm, or something terrifying: terror, panic, and the urge to flee or protect oneself; "
        "typically expressed as 'I am scared', 'this is terrifying', 'I'm afraid something will happen', 'run!'."
    ),
    'gratitude': (
        "feeling and expressing thankfulness and appreciation toward someone for a favor, gift, help, kindness, or support; "
        "typically expressed as 'thank you so much', 'I really appreciate it', 'thanks for your help', 'I am grateful for everything'."
    ),
    'grief': (
        "deep, heavy mourning and heartbreak after the death of a loved one or another devastating, irreversible loss; "
        "talk of someone passing away, funerals, missing the deceased, such as 'I lost my mother', 'rest in peace', 'she is gone and I can't cope'."
    ),
    'joy': (
        "a bright feeling of happiness, delight, and contentment about one's own good situation or good news in the moment; "
        "cheerful, glowing wellbeing, typically expressed as 'I am so happy', 'this made my day', 'what a wonderful feeling', 'I'm thrilled and smiling'."
    ),
    'love': (
        "deep affection, devotion, and tenderness toward a person, partner, family member, pet, or beloved thing; "
        "romantic or heartfelt attachment, typically expressed as 'I love you', 'you mean everything to me', 'my heart belongs to you', 'adore'."
    ),
    'nervousness': (
        "anxious, jittery tension and worry about an upcoming uncertain event such as an exam, interview, performance, or result; "
        "butterflies in the stomach and stage fright, typically expressed as 'I'm so nervous', 'what if it goes wrong', 'my hands are shaking'."
    ),
    'optimism': (
        "a hopeful, confident outlook that things will turn out well in the future, looking on the bright side despite difficulties; "
        "typically expressed as 'it will get better', 'I'm sure it will work out', 'things are looking up', 'better days are ahead'."
    ),
    'pride': (
        "satisfaction and dignity felt about one's own accomplishment, or about the success of one's child, family, team, or country; "
        "typically expressed as 'I did it', 'so proud of my son', 'I worked hard for this and it paid off', 'proud to be part of this'."
    ),
    'realization': (
        "the moment of suddenly understanding or noticing something that was not clear before: an insight, discovery, or the dawning of a fact, "
        "mistake, or truth; typically expressed as 'oh, now I get it', 'I just realized', 'it hit me that', 'so that's why', 'I never noticed before'."
    ),
    'relief': (
        "the feeling of tension and worry draining away because something bad did not happen or a stressful situation has finally ended well; "
        "typically expressed as 'what a relief', 'thank goodness', 'I can finally breathe', 'phew, it's over', 'glad that's behind me'."
    ),
    'remorse': (
        "guilt, regret, and sorrow about one's own wrongdoing: feeling bad for having hurt or failed someone and wishing to undo or apologize for it; "
        "typically expressed as 'I'm so sorry', 'I feel terrible about what I did', 'it's my fault', 'I should never have said that'."
    ),
    'sadness': (
        "feeling sorrowful, down, lonely, or hopeless because of hardship, rejection, or a painful situation; low mood, crying, and melancholy, "
        "typically expressed as 'I feel so sad', 'this hurts', 'I'm depressed', 'it breaks my heart', 'nothing seems to help'."
    ),
    'surprise': (
        "astonishment at something sudden, unexpected, or hard to believe: being startled or caught off guard, whether pleasantly or not; "
        "typically expressed as 'wow', 'I did not see that coming', 'no way!', 'what?!', 'I can't believe it'."
    ),
    'neutral': (
        "plain, factual, matter-of-fact language that states information, gives a description, asks a routine question, or reports an event "
        "without any emotional attitude, opinion, or feeling; objective statements such as 'the meeting is at 3 pm', 'here is the link', 'it is located downtown'."
    ),
}
assert set(GOEMOTIONS_LABELS) == set(GOEMOTIONS_DEFINITIONS), "labels and definitions must match 1-to-1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def resolve_held_out(names: Sequence[str]) -> List[int]:
    """Map held-out class names to column indices (validates names, rejects 'all classes')."""
    unknown = [n for n in names if n not in GOEMOTIONS_LABELS]
    assert not unknown, f"unknown held-out classes: {unknown}"
    idx = sorted({GOEMOTIONS_LABELS.index(n) for n in names})
    assert len(idx) < NUM_CLASSES, "at least one class must remain seen"
    return idx


def make_seen_mask(num_classes: int, held_out_classes: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Float mask [C]: 1.0 for seen classes, 0.0 for held-out (zero-shot) classes."""
    mask = torch.ones(num_classes)
    if held_out_classes:
        mask[torch.as_tensor(list(held_out_classes), dtype=torch.long)] = 0.0
    return mask


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    model_name: str = "roberta-base"
    max_len: int = 64
    seed: int = 42
    held_out_classes: List[str] = field(default_factory=list)
    # ---- adjacency (A_corr) ----
    self_loop_weight: float = 1.0
    min_edge_prob: float = 0.0
    # ---- classifier head / fusion / LACR ----
    mlp_hidden: int = 512
    dropout: float = 0.3
    lacr_proj_mode: str = "dense"
    lacr_transpose_adj: bool = True
    lacr_ln_bias_init: float = -2.0
    # ---- loss ----
    loss_type: str = "ldam_asl"
    gamma_pos: float = 0.0
    gamma_neg: float = 2.0
    prob_margin: float = 0.05
    ldam_max_margin: float = 0.3
    loss_reduction: str = "batch_mean"
    logit_clip: float = 8.0
    logit_penalty: float = 0.005
    # ---- distribution-balanced loss (loss_type == "db") ----
    db_neg_scale: float = 2.0
    db_logit_bias_scale: float = 0.05
    db_map_alpha: float = 10.0
    db_map_beta: float = 0.2
    db_map_delta: float = 0.1


@dataclass
class TrainingSettings:
    epochs: int = 7
    batch_size: int = 8
    eval_batch_size: int = 32
    grad_accum_steps: int = 4
    encoder_lr: float = 2e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    patience: int = 3
    min_delta: float = 0.0
    max_len: int = 64
    seed: int = 42
    use_amp: bool = True
    bounded_loss: bool = False


# =============================================================================
# Person 1 -- Shared encoder + semantic anchor branch
# =============================================================================

def build_anchor_texts(label_names: Sequence[str] = GOEMOTIONS_LABELS,
                        definitions: Dict[str, str] = GOEMOTIONS_DEFINITIONS) -> List[str]:
    texts = []
    for name in label_names:
        assert name in definitions and definitions[name].strip(), f"missing definition for {name!r}"
        texts.append(f"{name}: {definitions[name].strip()}")
    assert len(texts) == NUM_CLASSES
    return texts


class SharedEncoder(nn.Module):
    """RoBERTa backbone + attention-mask-weighted mean pooling: (input_ids, attention_mask) -> h [B, D]."""

    def __init__(self, model_name: str = "roberta-base"):
        super().__init__()
        try:
            self.backbone = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        except TypeError:
            self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(self.backbone.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).type_as(tokens)
        return (tokens * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)


class AnchorBank(nn.Module):
    """Fixed semantic anchors A in R^{C x D}: encoded ONCE offline, mean-pooled, L2-normalized, non-trainable."""

    def __init__(self, anchors: torch.Tensor):
        super().__init__()
        assert anchors.dim() == 2, f"anchors must be [C, D], got {tuple(anchors.shape)}"
        self.register_buffer("A", F.normalize(anchors.detach().float(), p=2, dim=-1))

    @staticmethod
    @torch.no_grad()
    def _encode(encoder: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        was_training = encoder.training
        encoder.eval()
        try:
            raw = encoder(input_ids, attention_mask)
        finally:
            encoder.train(was_training)
        return F.normalize(raw.detach().float(), p=2, dim=-1)

    @classmethod
    def build(cls, encoder: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> "AnchorBank":
        bank = cls(cls._encode(encoder, input_ids, attention_mask))
        bank.register_buffer("_ids", input_ids.detach().clone(), persistent=False)
        bank.register_buffer("_mask", attention_mask.detach().clone(), persistent=False)
        return bank

    @classmethod
    def from_texts(cls, encoder: nn.Module, tokenizer, texts: Sequence[str], max_length: int = 64) -> "AnchorBank":
        enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        dev = next(encoder.parameters()).device
        return cls.build(encoder, enc["input_ids"].to(dev), enc["attention_mask"].to(dev))

    @torch.no_grad()
    def refresh_(self, encoder: nn.Module) -> None:
        self.A.copy_(self._encode(encoder, self._ids, self._mask).to(self.A.device))

    def cosine(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(h, p=2, dim=-1) @ self.A.t()


def build_adjacency_matrix(train_labels: torch.Tensor, num_classes: int = NUM_CLASSES,
                            held_out_classes: Optional[Sequence[int]] = None,
                            self_loop_weight: float = 1.0, min_edge_prob: float = 0.0) -> torch.Tensor:
    """Row-normalised label co-occurrence graph from TRAIN labels only; held-out rows/cols zeroed."""
    assert train_labels.dim() == 2 and train_labels.size(1) == num_classes
    held = list(held_out_classes or [])
    Y = train_labels.float()
    co = Y.t() @ Y
    counts = Y.sum(dim=0).clamp(min=1e-9)
    P = co / counts.unsqueeze(1)
    P.fill_diagonal_(0.0)
    if min_edge_prob > 0:
        P = torch.where(P >= min_edge_prob, P, torch.zeros_like(P))
    A = P + self_loop_weight * torch.eye(num_classes)
    if held:
        idx = torch.as_tensor(held, dtype=torch.long)
        A[idx, :] = 0.0
        A[:, idx] = 0.0
    A = A / A.sum(dim=1, keepdim=True).clamp(min=1e-9)
    seen = make_seen_mask(num_classes, held).bool()
    assert torch.allclose(A[seen].sum(dim=1), torch.ones(int(seen.sum())), atol=1e-5), "seen rows must sum to 1"
    if held:
        assert A[~seen].abs().sum() == 0 and A[:, ~seen].abs().sum() == 0, "held-out rows/cols must be zero"
    return A


# =============================================================================
# Person 2 -- Classifier head + cross-interaction fusion + LACR + loss
# =============================================================================

class CrossInteractionLayer(nn.Module):
    """Vectorised u_i = [ h || A_i || (h * A_i) || Y_cos,i ]  ->  [B, C, 3D + 1]   (2305 for D = 768)."""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size

    @property
    def out_dim(self) -> int:
        return 3 * self.hidden_size + 1

    def forward(self, h: torch.Tensor, A: torch.Tensor, y_cos: torch.Tensor) -> torch.Tensor:
        B, D = h.shape
        C = A.size(0)
        assert D == self.hidden_size and A.shape == (C, D), f"h {tuple(h.shape)} vs A {tuple(A.shape)}"
        assert y_cos.shape == (B, C), f"Y_cos must be [{B}, {C}], got {tuple(y_cos.shape)}"
        h_e = h.unsqueeze(1).expand(B, C, D)
        A_e = A.unsqueeze(0).expand(B, C, D)
        return torch.cat([h_e, A_e, h_e * A_e, y_cos.unsqueeze(-1)], dim=-1)


class SharedInteractionMLP(nn.Module):
    """One MLP shared by all classes: u_i (3D+1) -> scalar logit.  [B, C, 3D+1] -> [B, C]."""

    def __init__(self, in_dim: int = 2305, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u).squeeze(-1)


class InteractionBranch(nn.Module):
    """Path 2: (h, A) -> (Y [B, C], Y_cos [B, C])."""

    def __init__(self, hidden_size: int = 768, mlp_hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.cross = CrossInteractionLayer(hidden_size)
        self.mlp = SharedInteractionMLP(self.cross.out_dim, mlp_hidden, dropout)

    def forward(self, h: torch.Tensor, A: torch.Tensor, y_cos: Optional[torch.Tensor] = None):
        if y_cos is None:
            y_cos = F.normalize(h, p=2, dim=-1) @ F.normalize(A, p=2, dim=-1).t()
        return self.mlp(self.cross(h, A, y_cos)), y_cos


class ParametricHead(nn.Module):
    """Path 1: X = W_c h + b_c  ->  [B, C]."""

    def __init__(self, hidden_size: int = 768, num_classes: int = 28, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(h))


class HybridFusion(nn.Module):
    """z = X * mask + Y : seen classes use X + Y, held-out (zero-shot) classes use Y only."""

    def __init__(self, num_classes: int = 28, held_out_classes: Optional[Sequence[int]] = None):
        super().__init__()
        self.register_buffer("seen_mask", make_seen_mask(num_classes, held_out_classes))

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        assert X.shape == Y.shape and X.size(1) == self.seen_mask.numel()
        return X * self.seen_mask + Y


class LACRModule(nn.Module):
    """Label-Aware Correlation Residual, applied strictly AFTER fusion, on logits z [B, C]."""

    def __init__(self, adjacency: torch.Tensor, held_out_classes: Optional[Sequence[int]] = None,
                 proj_mode: str = "dense", transpose_adj: bool = True, zero_init: bool = True,
                 ln_bias_init: float = 0.0, eps: float = 1e-5):
        super().__init__()
        C = adjacency.size(0)
        assert adjacency.shape == (C, C) and proj_mode in ("dense", "shared")
        seen = make_seen_mask(C, held_out_classes)
        held = seen == 0
        A = adjacency.detach().float().clone()
        assert A[held].abs().sum() == 0 and A[:, held].abs().sum() == 0, "A_corr leaks held-out classes"
        assert torch.allclose(A[~held].sum(dim=1), torch.ones(int((~held).sum())), atol=1e-4), "seen rows must sum to 1"
        self.proj_mode, self.transpose_adj = proj_mode, transpose_adj
        self.register_buffer("adjacency", A)
        self.register_buffer("seen_mask", seen)
        self.proj = nn.Linear(C, C) if proj_mode == "dense" else nn.Linear(1, 1)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(C, eps=eps)
        nn.init.constant_(self.norm.bias, ln_bias_init)

    def correlation(self, z: torch.Tensor) -> torch.Tensor:
        A = self.adjacency.t() if self.transpose_adj else self.adjacency
        return z @ A

    def residual(self, z: torch.Tensor) -> torch.Tensor:
        corr = self.correlation(z)
        if self.proj_mode == "dense":
            W = self.proj.weight * self.seen_mask.unsqueeze(1) * self.seen_mask.unsqueeze(0)
            res = F.linear(corr, W, self.proj.bias * self.seen_mask)
        else:
            res = self.proj(corr.unsqueeze(-1)).squeeze(-1)
        return res * self.seen_mask

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 2 and z.size(1) == self.adjacency.size(0), f"LACR expects fused logits [B, C], got {tuple(z.shape)}"
        return self.norm(z + self.residual(z))


class SemanticAnchorEmotionModel(nn.Module):
    """input_ids/attention_mask -> h -> (Path 1: X) + (Path 2: Y_cos -> u -> Y) -> z -> LACR -> z_final [B, 28]."""

    def __init__(self, encoder: nn.Module, anchor_bank: nn.Module, adjacency: torch.Tensor,
                 num_classes: int = NUM_CLASSES, held_out_classes: Optional[Sequence[int]] = None,
                 mlp_hidden: int = 512, dropout: float = 0.1,
                 lacr_kwargs: Optional[dict] = None):
        super().__init__()
        hidden = encoder.hidden_size
        assert anchor_bank.A.shape == (num_classes, hidden), f"anchors {tuple(anchor_bank.A.shape)} vs D={hidden}"
        self.encoder = encoder
        self.anchor_bank = anchor_bank
        self.interaction = InteractionBranch(hidden, mlp_hidden, dropout)
        self.head = ParametricHead(hidden, num_classes, dropout)
        self.fusion = HybridFusion(num_classes, held_out_classes)
        self.lacr = LACRModule(adjacency, held_out_classes, **(lacr_kwargs or {}))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(input_ids, attention_mask)
        A = self.anchor_bank.A
        Y, Y_cos = self.interaction(h, A)
        X = self.head(h)
        z = self.fusion(X, Y)
        return {"h": h, "Y_cos": Y_cos, "X": X, "Y": Y, "z": z, "z_final": self.lacr(z)}


SemanticAnchorClassifier = SemanticAnchorEmotionModel


class MaskedLDAMASLLoss(nn.Module):
    """LDAM margins + Asymmetric Loss focusing, with zero-shot (held-out) column masking."""

    def __init__(self, class_counts: Sequence[float], seen_mask: torch.Tensor, loss_type: str = "ldam_asl",
                 ldam_max_margin: float = 0.5, gamma_pos: float = 0.0, gamma_neg: float = 4.0,
                 prob_margin: float = 0.05, reduction: str = "batch_mean", eps: float = 1e-8):
        super().__init__()
        assert loss_type in ("ldam_asl", "ldam", "asl", "bce"), loss_type
        assert reduction in ("batch_mean", "mean"), reduction
        assert gamma_pos >= 0 and (gamma_neg == 0 or gamma_neg >= 1), "gamma_neg must be 0 or >= 1"
        assert 0.0 <= prob_margin < 1.0
        seen = seen_mask.float()
        counts = torch.as_tensor(np.asarray(class_counts), dtype=torch.float32).clamp(min=1.0)
        inv_quarter = counts.pow(-0.25)
        cst = ldam_max_margin / inv_quarter[seen.bool()].max()
        margins = cst * inv_quarter * seen
        use_ldam = loss_type in ("ldam_asl", "ldam")
        use_asl = loss_type in ("ldam_asl", "asl")
        self.register_buffer("margins", margins if use_ldam else torch.zeros_like(margins))
        self.register_buffer("seen_mask", seen)
        self.gamma_pos = gamma_pos if use_asl else 0.0
        self.gamma_neg = gamma_neg if use_asl else 0.0
        self.prob_margin = prob_margin if use_asl else 0.0
        self.reduction, self.eps = reduction, eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        z = logits.float()
        y = targets.float()
        assert z.shape == y.shape and z.size(1) == self.seen_mask.numel(), (tuple(z.shape), tuple(y.shape))
        z_adj = z - self.margins.unsqueeze(0) * y
        p = torch.sigmoid(z_adj)
        loss_pos = y * F.logsigmoid(z_adj)
        if self.gamma_pos > 0:
            loss_pos = loss_pos * (1.0 - p).pow(self.gamma_pos)
        if self.prob_margin > 0:
            p_m = (p - self.prob_margin).clamp(min=0.0)
            log_neg = torch.log((1.0 - p_m).clamp(min=self.eps))
        else:
            p_m = p
            log_neg = F.logsigmoid(-z_adj)
        loss_neg = (1.0 - y) * log_neg
        if self.gamma_neg > 0:
            loss_neg = loss_neg * p_m.clamp(min=self.eps).pow(self.gamma_neg)
        loss = -(loss_pos + loss_neg) * self.seen_mask.unsqueeze(0)
        if self.reduction == "batch_mean":
            return loss.sum() / z.size(0)
        return loss.sum() / (z.size(0) * self.seen_mask.sum().clamp(min=1.0))


class BoundedMaskedLDAMASLLoss(MaskedLDAMASLLoss):
    """Same as MaskedLDAMASLLoss, plus a soft logit bound (tanh) and an L2 logit penalty."""

    def __init__(self, *args, logit_clip: float = 8.0, logit_penalty: float = 0.005, **kwargs):
        super().__init__(*args, **kwargs)
        self.logit_clip, self.logit_penalty = float(logit_clip), float(logit_penalty)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        z = logits.float()
        z_b = self.logit_clip * torch.tanh(z / self.logit_clip) if self.logit_clip > 0 else z
        loss = super().forward(z_b, targets)
        if self.logit_penalty > 0:
            pen = (z.pow(2) * self.seen_mask.unsqueeze(0)).sum() / (z.size(0) * self.seen_mask.sum().clamp(min=1.0))
            loss = loss + self.logit_penalty * pen
        return loss


class DistributionBalancedLoss(nn.Module):
    """
    Distribution-Balanced Loss (Wu et al., "Distribution-Balanced Loss for
    Multi-Label Classification in Long-Tailed Datasets", ECCV 2020), adapted
    to this project's multi-label sigmoid logits with zero-shot (held-out)
    column masking.

    Two components counter GoEmotions' long-tailed, multi-label co-occurrence
    structure (a single sentence carrying several rare labels at once, which
    plain per-class resampling/weighting does not account for):

      1. Rebalanced weighting -- corrects the implicit over-sampling bias a
         sample would get under naive class-balanced resampling when it
         carries multiple rare labels at once. For sample k with target row
         y_k and per-class positive counts n_i:
             repeat_rate_k = sum_i y_k_i / n_i
             r_{k,i}       = (1 / n_i) / repeat_rate_k
             weight_{k,i}  = sigmoid(alpha * (r_{k,i} - beta)) + delta
         (alpha/beta/delta squash the raw ratio into a bounded, smooth
         per-(sample, class) weight on the loss term.)

      2. Negative-Tolerant Regularization (NTR) -- a class-specific logit
         bias derived from each class's prior frequency (same intuition as
         the focal-loss "prior probability" bias-init trick), plus a scale
         applied only to negative logits, so the flood of easy negatives
         from rare classes stops drowning out their few positives:
             bias_i   = logit_bias_scale * log(n_i / (N - n_i))
             z'_{k,i} = z_{k,i} + bias_i                      if y_{k,i} = 1
                      = neg_scale * (z_{k,i} + bias_i)         if y_{k,i} = 0

    Final loss: BCE-with-logits on z', scaled by the rebalanced weight,
    held-out columns zeroed, reduced batch_mean (sum over classes, mean over
    batch) to match MaskedLDAMASLLoss's convention.
    """

    def __init__(self, class_counts: Sequence[float], seen_mask: torch.Tensor, train_num: int,
                 neg_scale: float = 2.0, logit_bias_scale: float = 0.05,
                 map_alpha: float = 10.0, map_beta: float = 0.2, map_delta: float = 0.1,
                 reduction: str = "batch_mean", eps: float = 1e-8):
        super().__init__()
        assert reduction in ("batch_mean", "mean"), reduction
        assert train_num > 0
        seen = seen_mask.float()
        counts = torch.as_tensor(np.asarray(class_counts), dtype=torch.float32).clamp(min=1.0)
        freq_inv = 1.0 / counts
        prior = (counts / float(train_num)).clamp(min=eps, max=1.0 - eps)
        bias = logit_bias_scale * torch.log(prior / (1.0 - prior))
        self.register_buffer("freq_inv", freq_inv)
        self.register_buffer("bias", bias * seen)   # zero bias for held-out classes: no leakage
        self.register_buffer("seen_mask", seen)
        self.neg_scale = float(neg_scale)
        self.map_alpha, self.map_beta, self.map_delta = float(map_alpha), float(map_beta), float(map_delta)
        self.reduction, self.eps = reduction, eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        z = logits.float()
        y = targets.float()
        assert z.shape == y.shape and z.size(1) == self.seen_mask.numel(), (tuple(z.shape), tuple(y.shape))

        repeat_rate = (y * self.freq_inv.unsqueeze(0)).sum(dim=1, keepdim=True).clamp(min=self.eps)
        ratio = self.freq_inv.unsqueeze(0) / repeat_rate
        weight = torch.sigmoid(self.map_alpha * (ratio - self.map_beta)) + self.map_delta

        z_biased = z + self.bias.unsqueeze(0)
        z_reg = torch.where(y > 0, z_biased, self.neg_scale * z_biased)

        bce = F.binary_cross_entropy_with_logits(z_reg, y, reduction="none")
        loss = bce * weight * self.seen_mask.unsqueeze(0)
        if self.reduction == "batch_mean":
            return loss.sum() / z.size(0)
        return loss.sum() / (z.size(0) * self.seen_mask.sum().clamp(min=1.0))


def build_loss_fn(cfg: Config, class_counts: Sequence[float], seen_mask: torch.Tensor,
                   train_num: Optional[int] = None) -> nn.Module:
    if cfg.loss_type in ("ldam_asl", "ldam", "asl", "bce"):
        return MaskedLDAMASLLoss(class_counts, seen_mask, cfg.loss_type, cfg.ldam_max_margin, cfg.gamma_pos,
                                  cfg.gamma_neg, cfg.prob_margin, cfg.loss_reduction)
    if cfg.loss_type == "db":
        assert train_num is not None, "train_num is required to build DistributionBalancedLoss"
        return DistributionBalancedLoss(class_counts, seen_mask, train_num, cfg.db_neg_scale,
                                         cfg.db_logit_bias_scale, cfg.db_map_alpha, cfg.db_map_beta,
                                         cfg.db_map_delta, cfg.loss_reduction)
    raise ValueError(f"unknown loss_type {cfg.loss_type!r}; expected ldam_asl | ldam | asl | bce | db")


# =============================================================================
# Person 3 -- Data, training loop, evaluation
# =============================================================================

def multi_hot(label_lists, num_labels: int) -> np.ndarray:
    targets = np.zeros((len(label_lists), num_labels), dtype=np.float32)
    for row, labels in enumerate(label_lists):
        targets[row, labels] = 1.0
    return targets


def autocast_context(use_mixed_precision: bool):
    return torch.autocast("cuda", dtype=torch.float16) if use_mixed_precision else nullcontext()


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def calculate_metrics(targets, probabilities, threshold=0.5) -> Dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int32)
    result = {"subset_accuracy": float(accuracy_score(targets, predictions))}
    for average in ("micro", "macro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average=average, zero_division=0)
        result.update({f"precision_{average}": float(precision),
                        f"recall_{average}": float(recall), f"f1_{average}": float(f1)})
    return result


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_mixed_precision):
    model.eval()
    total_loss, total_examples = 0.0, 0
    all_probabilities, all_targets = [], []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast_context(use_mixed_precision):
            logits = model(batch["input_ids"], batch["attention_mask"])["z_final"]
        loss = loss_fn(logits.float(), batch["labels"])
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation loss/logits.")
        batch_size = len(batch["labels"])
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_probabilities.append(logits.float().sigmoid().cpu().numpy())
        all_targets.append(batch["labels"].cpu().numpy())
    if total_examples == 0:
        raise ValueError("Evaluation loader is empty.")
    return total_loss / total_examples, np.concatenate(all_probabilities), np.concatenate(all_targets)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn, epoch, training, device, use_mixed_precision):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_examples, examples_since_update, skipped_updates = 0.0, 0, 0, 0
    progress = tqdm(loader, desc=f"Epoch {epoch}/{training.epochs}")
    for batch_index, batch in enumerate(progress):
        batch = to_device(batch, device)
        batch_size = len(batch["labels"])
        with autocast_context(use_mixed_precision):
            logits = model(batch["input_ids"], batch["attention_mask"])["z_final"]
        loss = loss_fn(logits.float(), batch["labels"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at batch {batch_index}.")
        scaler.scale(loss * batch_size).backward()
        examples_since_update += batch_size
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        ready_to_update = (batch_index + 1) % training.grad_accum_steps == 0 or batch_index + 1 == len(loader)
        if ready_to_update:
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(examples_since_update)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.max_grad_norm,
                                            error_if_nonfinite=not use_mixed_precision)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            else:
                skipped_updates += 1
            optimizer.zero_grad(set_to_none=True)
            examples_since_update = 0
        progress.set_postfix(loss=f"{total_loss / total_examples:.4f}")
    return total_loss / total_examples, skipped_updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                         help="tiny fast run (small data slice, 1 epoch) to sanity-check the pipeline")
    parser.add_argument("--epochs", type=int, default=None, help="override TrainingSettings.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="override TrainingSettings.batch_size")
    parser.add_argument("--loss-type", type=str, default=None, choices=["ldam_asl", "ldam", "asl", "bce", "db"],
                         help="override Config.loss_type")
    args = parser.parse_args()

    cfg = Config()
    training = TrainingSettings()
    if args.loss_type is not None:
        cfg.loss_type = args.loss_type
    if args.smoke_test:
        training.epochs = 1
        training.batch_size = 4
        training.grad_accum_steps = 1
        training.eval_batch_size = 4
        training.patience = 1
    if args.epochs is not None:
        training.epochs = args.epochs
    if args.batch_size is not None:
        training.batch_size = args.batch_size

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    banner("CONFIGURATION")
    print(f"[ENV] device={device} | model={cfg.model_name} | seed={cfg.seed}")
    print(f"[CFG] held-out={cfg.held_out_classes or 'none (fully supervised)'} | loss={cfg.loss_type}")
    print(f"[TRAIN] epochs={training.epochs} batch_size={training.batch_size} "
          f"grad_accum={training.grad_accum_steps} smoke_test={args.smoke_test}")

    banner("DATA")
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    label_names = dataset["train"].features["labels"].feature.names
    assert label_names == GOEMOTIONS_LABELS, "Dataset and model label order must match."
    num_labels = len(label_names)

    if args.smoke_test:
        for split in ("train", "validation", "test"):
            dataset[split] = dataset[split].select(range(min(64, len(dataset[split]))))

    train_df = dataset["train"].to_pandas()
    train_labels = multi_hot(dataset["train"]["labels"], num_labels)
    class_counts = train_labels.sum(axis=0)
    if not args.smoke_test:
        assert (class_counts > 0).all(), "every class needs at least one positive training example"
    print({split: len(dataset[split]) for split in dataset})

    banner("CELL: OFFLINE SEMANTIC ANCHORS")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    held_out = resolve_held_out(cfg.held_out_classes)
    encoder = SharedEncoder(cfg.model_name).to(device)
    anchor_texts = build_anchor_texts()
    anchor_bank = AnchorBank.from_texts(encoder, tokenizer, anchor_texts, max_length=cfg.max_len).to(device)
    print(f"[ANCHORS] A {tuple(anchor_bank.A.shape)} | requires_grad={anchor_bank.A.requires_grad}")

    banner("A_corr: LABEL CO-OCCURRENCE ADJACENCY (TRAIN ONLY)")
    adjacency = build_adjacency_matrix(torch.as_tensor(train_labels), NUM_CLASSES, held_out,
                                        cfg.self_loop_weight, cfg.min_edge_prob)
    print(f"[ADJ] A_corr {tuple(adjacency.shape)}")

    banner("MODEL")
    model = SemanticAnchorClassifier(
        encoder, anchor_bank, adjacency, NUM_CLASSES, held_out, cfg.mlp_hidden, cfg.dropout,
        lacr_kwargs={"proj_mode": cfg.lacr_proj_mode, "transpose_adj": cfg.lacr_transpose_adj,
                     "ln_bias_init": cfg.lacr_ln_bias_init}).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] trainable parameters: {n_params / 1e6:.1f}M | held-out: "
          f"{[GOEMOTIONS_LABELS[i] for i in held_out] or 'none (fully supervised)'}")

    probe = tokenizer(train_df["text"].head(4).tolist(), padding=True, truncation=True,
                       max_length=cfg.max_len, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        probe_out = model(probe["input_ids"].to(device), probe["attention_mask"].to(device))
    model.train()
    assert probe_out["z_final"].shape == (probe["input_ids"].shape[0], NUM_CLASSES)
    assert torch.isfinite(probe_out["z_final"]).all(), "non-finite logits"
    print(f"[VERIFY] forward pass OK | z_final {tuple(probe_out['z_final'].shape)}")

    banner("DATA LOADERS")
    assert not cfg.held_out_classes, "This training setup supervises all labels."
    assert torch.all(model.fusion.seen_mask == 1), "Rebuild the model without held-out labels."

    use_mixed_precision = training.use_amp and device.type == "cuda"
    output_dir = Path("goemotions_runs") / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    print("Device:", device)
    print("Effective batch size:", training.batch_size * training.grad_accum_steps)
    print("Results folder:", output_dir.resolve())

    def prepare_batch(batch):
        encoded = tokenizer(batch["text"], truncation=True, max_length=training.max_len)
        encoded["labels"] = multi_hot(batch["labels"], num_labels).tolist()
        return encoded

    tokenized = {
        split: dataset[split].map(prepare_batch, batched=True, remove_columns=dataset[split].column_names)
        for split in ("train", "validation", "test")
    }

    def collate_batch(examples):
        features = [{k: example[k] for k in ("input_ids", "attention_mask")} for example in examples]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor([example["labels"] for example in examples], dtype=torch.float32)
        return batch

    train_loader = DataLoader(tokenized["train"], batch_size=training.batch_size,
                               shuffle=True, generator=torch.Generator().manual_seed(training.seed),
                               collate_fn=collate_batch, pin_memory=device.type == "cuda")
    val_loader = DataLoader(tokenized["validation"], batch_size=training.eval_batch_size,
                             shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(tokenized["test"], batch_size=training.eval_batch_size,
                              shuffle=False, collate_fn=collate_batch)

    banner("LOSS + OPTIMIZER")
    seen_mask = model.fusion.seen_mask.detach().cpu()
    train_num = int(train_labels.shape[0])
    if training.bounded_loss and cfg.loss_type in ("ldam_asl", "ldam", "asl", "bce"):
        loss_fn = BoundedMaskedLDAMASLLoss(
            class_counts, seen_mask, loss_type=cfg.loss_type,
            ldam_max_margin=cfg.ldam_max_margin, gamma_pos=cfg.gamma_pos,
            gamma_neg=cfg.gamma_neg, prob_margin=cfg.prob_margin,
            reduction=cfg.loss_reduction, logit_clip=cfg.logit_clip,
            logit_penalty=cfg.logit_penalty).to(device)
    else:
        loss_fn = build_loss_fn(cfg, class_counts, seen_mask, train_num).to(device)
    print("Loss:", cfg.loss_type, "| bounded:", training.bounded_loss and cfg.loss_type in ("ldam_asl", "ldam", "asl", "bce"))

    tokenizer.save_pretrained(output_dir / "tokenizer")
    (output_dir / "settings.json").write_text(json.dumps({
        "training": asdict(training), "classifier": asdict(cfg),
        "encoder_config": model.encoder.backbone.config.to_dict(),
        "label_names": label_names, "class_counts": class_counts.tolist(),
        "dataset": "google-research-datasets/go_emotions", "dataset_config": "simplified",
        "anchor_policy": "use the existing fixed anchor bank without refreshing",
        "probabilities": "sigmoid(raw z_final)", "smoke_test": args.smoke_test,
    }, indent=2))

    parameter_groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        learning_rate = training.encoder_lr if name.startswith("encoder.") else training.head_lr
        weight_decay = 0.0 if parameter.ndim <= 1 or name.endswith("bias") else training.weight_decay
        parameter_groups.setdefault((learning_rate, weight_decay), []).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": parameters, "lr": learning_rate, "weight_decay": weight_decay}
        for (learning_rate, weight_decay), parameters in parameter_groups.items()])
    updates_per_epoch = math.ceil(len(train_loader) / training.grad_accum_steps)
    total_updates = updates_per_epoch * training.epochs
    warmup_updates = int(total_updates * training.warmup_ratio)

    def learning_rate_schedule(step):
        if step < warmup_updates:
            return float(step + 1) / max(1, warmup_updates)
        return max(0.0, (total_updates - step) / max(1, total_updates - warmup_updates))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)

    banner("TRAINING")
    best_score, best_epoch, epochs_without_improvement = -1.0, 0, 0
    history = []
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, training.epochs + 1):
        train_loss, skipped_updates = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, loss_fn, epoch, training, device, use_mixed_precision)
        val_loss, val_probabilities, val_targets = evaluate(model, val_loader, loss_fn, device, use_mixed_precision)
        val_metrics = calculate_metrics(val_targets, val_probabilities, 0.5)
        result_row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                      "skipped_amp_updates": skipped_updates,
                      **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(result_row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        score = val_metrics["f1_macro"]
        print(f"Epoch {epoch}: train loss={train_loss:.4f}, val loss={val_loss:.4f}, "
              f"val Macro-F1={score:.4f}, Micro-F1={val_metrics['f1_micro']:.4f}")
        if score > best_score + training.min_delta:
            best_score, best_epoch, epochs_without_improvement = score, epoch, 0
            saved_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            temporary_path = output_dir / "best_model.tmp"
            torch.save(saved_weights, temporary_path)
            temporary_path.replace(checkpoint_path)
            del saved_weights
            (output_dir / "best_validation.json").write_text(json.dumps(
                {"epoch": best_epoch, "threshold": 0.5, **val_metrics}, indent=2))
            print("Saved new best model.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training.patience:
                print(f"Early stopping: {training.patience} epochs without improvement.")
                break
    print(f"Best epoch: {best_epoch}; validation Macro-F1 @ 0.5: {best_score:.4f}")

    banner("THRESHOLD CALIBRATION")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.to(device)
    val_loss, val_probabilities, val_targets = evaluate(model, val_loader, loss_fn, device, use_mixed_precision)
    threshold_results = []
    for threshold in np.linspace(0.05, 0.95, 19):
        threshold_results.append({"threshold": float(threshold),
                                   **calculate_metrics(val_targets, val_probabilities, float(threshold))})
    best_result = max(threshold_results, key=lambda r: (r["f1_macro"], -abs(r["threshold"] - 0.5)))
    best_threshold = best_result["threshold"]
    pd.DataFrame(threshold_results).to_csv(output_dir / "validation_thresholds.csv", index=False)
    (output_dir / "inference_config.json").write_text(json.dumps({
        "threshold": best_threshold, "label_names": label_names,
        "best_epoch": best_epoch, "validation_metrics": best_result,
        "probabilities": "sigmoid(raw z_final)", "comparison": ">="}, indent=2))
    print("Validation-selected threshold:", best_threshold)

    banner("FINAL TEST EVALUATION")
    test_loss, test_probabilities, test_targets = evaluate(model, test_loader, loss_fn, device, use_mixed_precision)
    test_metrics = calculate_metrics(test_targets, test_probabilities, best_threshold)
    report = {"best_epoch": best_epoch, "threshold": best_threshold, "test_loss": test_loss, **test_metrics}
    (output_dir / "test_metrics.json").write_text(json.dumps(report, indent=2))
    test_predictions = (test_probabilities >= best_threshold).astype(np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(
        test_targets, test_predictions, average=None, zero_division=0)
    per_label = pd.DataFrame({"label": label_names, "precision": precision, "recall": recall,
                               "f1": f1, "support": support.astype(int)})
    per_label.to_csv(output_dir / "test_per_label.csv", index=False)
    print(json.dumps(report, indent=2))
    print("Files saved in:", output_dir.resolve())


if __name__ == "__main__":
    main()
