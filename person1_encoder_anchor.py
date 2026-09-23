"""
person1_encoder_anchor.py
==========================
Owner: Person 1 -- Shared Encoder + Semantic Anchor Extraction & LAC Refinement

Merged from `person1_encoder_anchor_branch.py` and
`person1_encoder_anchor (2).py`, reconciled against Person 1's task list in
TASKS.md:

    - SharedEncoder: RoBERTa-base backbone, one instance reused for both
      sentence encoding and anchor-text encoding (no separate copy).
    - AnchorBranch: builds the 28 semantic anchor embeddings (name +
      definition + LLM-generated exemplar sentences), refreshed from the
      *current* encoder weights every epoch (TASKS.md: "No frozen/cached
      anchors ... recomputed at minimum each epoch").
    - build_adjacency_matrix: empirical label co-occurrence graph
      `A: [28, 28]`, with a `held_out_classes` parameter so the zero-shot
      experiment (Person 3) never leaks held-out co-occurrence statistics.
    - LACModule: the graph-attention LAC layer TASKS.md assigns to Person 1,
      run over the 28 anchor nodes using `A` as structure, with a residual
      connection back to the raw anchor embedding:
          E_lac = E_raw + GAT(E_raw, A)
    - AnchorBranch.similarity: cosine similarity between sentence
      representations `h` and the LAC-refined anchor embeddings `E_lac`,
      producing `Y` -- the "Vector space" step in TASKS.md.

Variable/class naming follows `person1_encoder_anchor (2).py` (mean-pooled
`SharedEncoder`, cached `anchor_embeddings` buffer refreshed via
`recompute_anchors`, CLIP-style `logit_scale` temperature, `GOEMOTIONS_*`
constants, etc). The GAT/LAC step itself did not exist in that file -- it
was factored out into `LACModule` below, adapted from the graph-attention
logic in `person1_encoder_anchor_branch.py` but applied to the raw anchor
embeddings `E_raw` (per TASKS.md) rather than to post-fusion logits.

Person 1's output interface for Person 2's Cross-Interaction Layer is a
dict with `anchors: [28, 768]` (now `E_lac`, not raw `E_raw`) and
`cos_sim: [B, 28]` -- the `A_i` and `Y_cos` terms needed to build
`u_i = [h || A_i || (h * A_i) || Y_cos_i]`.

References
----------
[1] Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions", 2020.
    arXiv:2005.00547
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


GOEMOTIONS_LABELS: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

# Short human-authored definitions used to seed the anchor text for each
# label. In production these are concatenated with LLM-generated exemplar
# sentences (see `AnchorBranch.default_anchor_texts`).
GOEMOTIONS_DEFINITIONS: Dict[str, str] = {
    "admiration": "a feeling of respect and warm approval for someone or something impressive.",
    "amusement": "the state of finding something funny or entertaining.",
    "anger": "a strong feeling of displeasure or hostility.",
    "annoyance": "a mild feeling of irritation.",
    "approval": "a positive endorsement or agreement with something.",
    "caring": "displaying kindness and concern for others.",
    "confusion": "a lack of understanding or a feeling of being disoriented.",
    "curiosity": "a strong desire to know or learn something.",
    "desire": "a strong wish to have or do something.",
    "disappointment": "sadness caused by the non-fulfillment of hopes.",
    "disapproval": "an unfavorable or negative opinion about something.",
    "disgust": "a feeling of revulsion or strong disapproval.",
    "embarrassment": "a feeling of self-consciousness or shame.",
    "excitement": "a feeling of great enthusiasm and eagerness.",
    "fear": "an unpleasant emotion caused by threat of danger or harm.",
    "gratitude": "the quality of being thankful.",
    "grief": "deep sorrow, especially caused by loss.",
    "joy": "a feeling of great pleasure and happiness.",
    "love": "an intense feeling of deep affection.",
    "nervousness": "a feeling of anxiety or apprehension.",
    "optimism": "hopefulness and confidence about the future.",
    "pride": "a feeling of deep satisfaction from one's own achievements.",
    "realization": "becoming fully aware of something as a fact.",
    "relief": "a feeling of reassurance following release from anxiety.",
    "remorse": "deep regret for a wrong committed.",
    "sadness": "the feeling of sorrow or unhappiness.",
    "surprise": "a feeling of mild astonishment or shock.",
    "neutral": "an absence of strong emotion; emotionally flat.",
}


class SharedEncoder(nn.Module):
    """RoBERTa-base encoder with masked mean pooling.

    The exact same nn.Module instance must be used to encode both the
    input sentences and the anchor texts, ensuring both representation
    sets live in the same embedding space (no encoder drift between
    branches).
    """

    def __init__(self, model_name: str = "roberta-base", hidden_size: int = 768):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, L] token ids.
            attention_mask: [B, L] 1 for real tokens, 0 for padding.
        Returns:
            h: [B, 768] masked mean-pooled contextual representation.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # [B, L, 768]

        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)  # [B, L, 1]
        summed = torch.sum(token_embeddings * mask, dim=1)             # [B, 768]
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)                # [B, 1]
        h = summed / counts
        return h


def build_adjacency_matrix(
    train_labels: torch.Tensor,
    num_classes: int = 28,
    held_out_classes: Optional[List[int]] = None,
    self_loop_weight: float = 1.0,
) -> torch.Tensor:
    """
    Builds a row-normalized empirical label co-occurrence graph.

    A[i, j] approximates P(L_j | L_i) = count(L_i & L_j) / count(L_i),
    i.e. the conditional probability that label j is also present
    given that label i is present, estimated purely from TRAIN labels.

    This is the structural graph `A` consumed by `LACModule` below to
    refine the raw anchor embeddings (TASKS.md: "LAC module: graph
    attention layer(s) over the 28 anchor nodes using A as structure").

    Zero-Shot leakage protection: any row/column corresponding to a
    held-out class is fully zeroed both before AND after self-loops are
    injected, guaranteeing no co-occurrence statistic computed on a
    held-out label -- nor even a self-loop identity signal -- can leak
    into the graph structure any downstream module conditions on.

    Args:
        train_labels: [N, num_classes] multi-hot binary label matrix,
            TRAIN SPLIT ONLY.
        num_classes: number of label classes (28 for GoEmotions).
        held_out_classes: label indices to treat as unseen (zero-shot).
        self_loop_weight: weight added to the diagonal before row-norm.

    Returns:
        A: [num_classes, num_classes] row-normalized adjacency matrix.
    """
    assert train_labels.dim() == 2 and train_labels.size(1) == num_classes
    held_out_classes = held_out_classes or []

    Y = train_labels.float()
    # Co-occurrence counts: C[i, j] = sum_n Y[n, i] * Y[n, j]
    co_occurrence = Y.t() @ Y  # [C, C]
    label_counts = Y.sum(dim=0).clamp(min=1e-9)  # [C]

    # P(L_j | L_i) = C[i, j] / count(L_i)  -> divide row i by count(L_i)
    A = co_occurrence / label_counts.unsqueeze(1)

    def _zero_held_out(mat: torch.Tensor) -> torch.Tensor:
        if held_out_classes:
            idx = torch.tensor(held_out_classes, dtype=torch.long, device=mat.device)
            mat[idx, :] = 0.0
            mat[:, idx] = 0.0
        return mat

    A = _zero_held_out(A)

    # Remove trivial self co-occurrence (always 1.0) and inject a
    # controlled, learnable-strength self-loop instead.
    A.fill_diagonal_(0.0)
    A = A + self_loop_weight * torch.eye(num_classes, device=A.device)

    # Re-zero after adding I so held-out classes get NO self-loop signal
    # either -- they must be fully isolated / unreachable in the graph.
    A = _zero_held_out(A)

    # Row-normalize (guard div-by-zero for isolated / held-out rows).
    row_sums = A.sum(dim=1, keepdim=True).clamp(min=1e-9)
    A = A / row_sums
    return A


class LACModule(nn.Module):
    """Label-Aware Correlation (LAC) graph-attention layer.

    Refines the 28 raw anchor embeddings using the label co-occurrence
    graph `A` from `build_adjacency_matrix`, per TASKS.md's Person 1 spec:

        E_lac = E_raw + GAT(E_raw, A)

    A standard single-head GAT layer: pairwise attention scores are
    computed over every anchor pair from their projected features, masked
    to the co-occurrence edges in `A` (`A[i, j] > 0`), softmax-normalized
    per row, then used to mix neighboring anchor features. The result is
    projected back to `hidden_size` and added back to `E_raw` as a
    residual, so an anchor with no informative neighbors (e.g. an
    isolated / held-out node) falls back to its own raw embedding.
    """

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size

        # projects raw anchor embeddings into attention-feature space
        self.node_proj = nn.Linear(hidden_size, hidden_size)

        # standard GAT-style attention scoring function over concatenated
        # (node_i, node_j) feature pairs
        self.attn_a = nn.Linear(2 * hidden_size, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # projects attended node features back into embedding space before
        # the residual add
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, E_raw: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            E_raw: [num_classes, hidden_size] raw pooled anchor embeddings.
            A:     [num_classes, num_classes] co-occurrence adjacency
                   matrix from `build_adjacency_matrix` (edges = `A > 0`).
        Returns:
            E_lac: [num_classes, hidden_size] LAC-refined anchor
                   embeddings, `E_raw + GAT(E_raw, A)`.
        """
        C = E_raw.size(0)
        node_feats = self.node_proj(E_raw)  # [C, hidden_size]

        feats_i = node_feats.unsqueeze(1).expand(C, C, -1)  # [C, C, hidden_size]
        feats_j = node_feats.unsqueeze(0).expand(C, C, -1)  # [C, C, hidden_size]
        e = self.leaky_relu(self.attn_a(torch.cat([feats_i, feats_j], dim=-1))).squeeze(-1)  # [C, C]

        # mask to co-occurrence edges only, then softmax per row
        edge_mask = A > 0
        e_masked = e.masked_fill(~edge_mask, float("-inf"))
        alpha = torch.softmax(e_masked, dim=-1)  # [C, C]

        attended = alpha @ node_feats             # [C, hidden_size]
        gat_out = self.out_proj(attended)          # [C, hidden_size]

        E_lac = E_raw + gat_out                     # residual, per TASKS.md
        return E_lac


class AnchorBranch(nn.Module):
    """Semantic anchor extraction and LAC refinement for zero-shot emotion
    recognition.

    Builds anchor embeddings from (name + definition + exemplar sentences)
    via the SHARED encoder, refines them with `LACModule` using the label
    co-occurrence graph, caches the refined result in a module buffer to
    avoid re-encoding + re-refining all 28 anchors on every forward pass,
    and exposes a temperature-scaled cosine-similarity scorer against
    sentence representations `h`.

    Person 1's contract with Person 2 is the structured output of
    `forward()`: `{"anchors": [28, 768], "cos_sim": [B, 28]}`, where
    `anchors` is now `E_lac` (LAC-refined), not the raw pooled embedding.
    """

    def __init__(self, shared_encoder: SharedEncoder, tokenizer: AutoTokenizer,
                 num_classes: int = 28, hidden_size: int = 768,
                 max_anchor_len: int = 128,
                 logit_scale_init: float = float(np.log(1 / 0.07)),
                 logit_scale_max: float = float(np.log(100.0))):
        super().__init__()
        self.shared_encoder = shared_encoder
        self.tokenizer = tokenizer
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        # 128 (default) comfortably fits "name + definition + a few LLM
        # exemplar sentences" without premature truncation; bump to 256
        # if exemplar sets are long.
        self.max_anchor_len = max_anchor_len

        # LAC graph-attention refinement layer (TASKS.md: E_lac = E_raw +
        # GAT(E_raw, A)), applied to the raw anchor embeddings before they
        # are cached / used for similarity.
        self.lac_module = LACModule(hidden_size=hidden_size)

        # Learnable temperature / logit scale, CLIP-style: logits are
        # cosine similarities (bounded in [-1, 1]) multiplied by
        # exp(logit_scale). This widens the effective logit range so
        # BCEWithLogitsLoss (and LDAM/ASL/DB-Loss downstream) receive
        # well-scaled, stably-gradiented inputs instead of raw cosine
        # values that saturate the sigmoid too slowly. Initialized to
        # log(1/0.07) per CLIP; clamped at exp(logit_scale_max) = 100
        # to prevent runaway scaling during training.
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)
        self.logit_scale_max = logit_scale_max

        # Cached anchor buffer (holds E_lac, not raw E_raw). `persistent=
        # False`: anchors are recomputed from text at the start of every
        # epoch (see `recompute_anchors`) -- this satisfies TASKS.md's
        # "no frozen/cached anchors ... recomputed at minimum each epoch"
        # while avoiding the VRAM cost of re-running the encoder + GAT on
        # every batch. Registering as a buffer (rather than a plain
        # attribute) ensures it correctly follows `.to(device)` /
        # `.cuda()` calls on the parent module.
        self.register_buffer(
            "anchor_embeddings",
            torch.zeros(num_classes, hidden_size),
            persistent=False,
        )
        self._anchors_initialized = False

    @staticmethod
    def default_anchor_texts(
        exemplar_sentences: Optional[Dict[str, Sequence[str]]] = None,
        label_names: Sequence[str] = GOEMOTIONS_LABELS,
        definitions: Dict[str, str] = GOEMOTIONS_DEFINITIONS,
    ) -> List[str]:
        """Builds the "name + definition + exemplar sentences" anchor
        text for every class, in label-index order.

        `exemplar_sentences` is an optional dict of label -> list of
        LLM-generated exemplar sentences; if omitted only name +
        definition are used (still a valid, if weaker, anchor).
        """
        exemplar_sentences = exemplar_sentences or {}
        texts: List[str] = []
        for label in label_names:
            parts = [label, definitions.get(label, "")]
            examples = exemplar_sentences.get(label, [])
            if examples:
                parts.append(" ".join(examples))
            texts.append(". ".join(p for p in parts if p).strip())
        return texts

    def _infer_device(self) -> torch.device:
        """Infers the module's current device from its own parameters
        (e.g. `self.logit_scale`), so callers do not have to thread a
        `device` argument through every call.
        """
        return next(self.parameters()).device

    def encode_anchors(self, anchor_texts: List[str], A: torch.Tensor,
                        device: Optional[torch.device] = None) -> torch.Tensor:
        """Tokenizes, encodes, and LAC-refines anchor texts with the
        shared encoder and `LACModule`.

        This is a DIFFERENTIABLE pass through the backbone and GAT layer
        -- gradients flow back into `shared_encoder` and `lac_module`
        when called inside a `torch.enable_grad()` context. For the
        memory-efficient, gradient-free per-epoch cache refresh, use
        `recompute_anchors` instead.

        Args:
            anchor_texts: length-`num_classes` list of anchor strings.
            A: [num_classes, num_classes] co-occurrence adjacency matrix
                from `build_adjacency_matrix`, consumed by `lac_module`.
            device: target device. If omitted, inferred automatically
                from this module's own parameters.
        Returns:
            E_lac: [num_classes, hidden_size], LAC-refined, L2-normalized.
        """
        device = device if device is not None else self._infer_device()
        assert len(anchor_texts) == self.num_classes
        enc = self.tokenizer(
            anchor_texts, padding=True, truncation=True,
            max_length=self.max_anchor_len, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        E_raw = self.shared_encoder(input_ids, attention_mask)  # [28, 768]
        E_lac = self.lac_module(E_raw, A.to(device))              # [28, 768]
        E_lac = F.normalize(E_lac, p=2, dim=-1)
        return E_lac

    @torch.no_grad()
    def recompute_anchors(self, anchor_texts: List[str], A: torch.Tensor,
                           device: Optional[torch.device] = None) -> torch.Tensor:
        """Memory-efficient anchor recomputation strategy.

        Refreshes the cached `anchor_embeddings` buffer by re-encoding all
        28 anchor texts through the (possibly-updated) shared encoder and
        re-running LAC refinement over the (possibly-updated) `lac_module`.
        Intended to be called ONCE AT THE START OF EACH TRAINING EPOCH
        (and once before evaluation), rather than on every batch.

        Why this matters:
            - Both the shared encoder and `lac_module` are trained
              jointly with sentences, so anchor embeddings computed at
              epoch 0 will semantically drift away from the encoder's
              current representation space as training progresses.
              Periodic recomputation keeps anchors aligned with the
              encoder without paying the cost on every single batch.
            - Running the encoder + GAT over 28 anchor texts on every
              batch (as a differentiable op) would needlessly duplicate
              compute and retain 28-anchors' worth of activation memory
              in the autograd graph for every batch in the epoch, which
              -- multiplied across batches -- is a common source of
              avoidable VRAM OOM. Wrapping this call in `torch.no_grad()`
              and caching the DETACHED result means the rest of the
              epoch's forward passes only pay the cost of a buffer
              lookup, not a fresh backbone + GAT pass.

        Args:
            anchor_texts: length-`num_classes` list of anchor strings.
            A: [num_classes, num_classes] co-occurrence adjacency matrix
                from `build_adjacency_matrix`.
            device: target device. If omitted, inferred automatically
                from this module's own parameters.
        Returns:
            The refreshed `anchor_embeddings` buffer, [28, hidden_size].
        """
        device = device if device is not None else self._infer_device()
        E_lac = self.encode_anchors(anchor_texts, A, device)  # runs under @torch.no_grad()
        self.anchor_embeddings = E_lac.to(device)
        self._anchors_initialized = True
        return self.anchor_embeddings

    def similarity(self, h: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled cosine similarity between sentence reps
        and LAC-refined anchor reps ("Vector space" in TASKS.md),
        returned as usable logits `Y`.

        Raw cosine similarity is bounded to [-1, 1], which is far too
        narrow a range for stable `BCEWithLogitsLoss` (and the
        downstream LDAM / ASL / DB-Loss margin math) -- gradients
        through a sigmoid over such a small range saturate slowly and
        the model struggles to produce confident predictions. Scaling
        by a learnable, CLIP-style temperature (`exp(logit_scale)`,
        clamped for numerical stability) lets the model learn how
        sharply to separate its cosine scores.

        Args:
            h:       [B, 768] sentence representations (need not be
                     pre-normalized).
            anchors: [28, 768] LAC-refined anchor representations
                     (`E_lac`; need not be pre-normalized, re-normalized
                     defensively here).
        Returns:
            Y: [B, 28] temperature-scaled similarity logits.
        """
        h_norm = F.normalize(h, p=2, dim=-1)
        E_norm = F.normalize(anchors, p=2, dim=-1)
        raw_cos = h_norm @ E_norm.t()  # [B, 28], in [-1, 1]

        scale = self.logit_scale.clamp(max=self.logit_scale_max).exp()
        Y = raw_cos * scale
        return Y

    def forward(self, h: torch.Tensor, anchor_texts: Optional[List[str]] = None,
                A: Optional[torch.Tensor] = None,
                device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        Computes Person 1's full output contract for Person 2's
        Cross-Interaction Layer: the (cached, LAC-refined, L2-normalized)
        anchor matrix and the temperature-scaled cosine similarities.

        Args:
            h: [B, 768] sentence representations from `SharedEncoder`.
            anchor_texts: optional length-28 list of anchor strings.
                - If `None` (the default / expected steady-state path):
                  uses the cached `anchor_embeddings` buffer populated
                  by the most recent `recompute_anchors` call -- no
                  backbone/GAT pass over the anchors happens here,
                  keeping every batch's forward pass cheap and
                  memory-stable.
                - If provided: encodes + LAC-refines anchors fresh,
                  DIFFERENTIABLY, via `encode_anchors` (does NOT touch
                  the cache; requires `A` to also be provided). Use this
                  only when a differentiable anchor path is explicitly
                  required (e.g. anchor-embedding ablation studies);
                  ordinary training/eval should rely on the cached
                  buffer refreshed once per epoch via
                  `recompute_anchors`.
            A: [28, 28] co-occurrence adjacency matrix, required when
                `anchor_texts` is provided.
            device: target device. If omitted, inferred automatically
                from this module's own parameters.
        Returns:
            {
              "anchors":  [28, 768]  LAC-refined, L2-normalized anchor
                                     embeddings (`E_lac`; `A_i` in
                                     Person 2's interaction vector u_i),
              "cos_sim":  [B, 28]    temperature-scaled cosine
                                     similarities (`Y`; `Y_cos` in
                                     Person 2's interaction vector u_i),
            }
        """
        device = device if device is not None else self._infer_device()

        if anchor_texts is not None:
            assert A is not None, "A (adjacency matrix) is required when anchor_texts is provided"
            anchors = self.encode_anchors(anchor_texts, A, device)
        else:
            if not self._anchors_initialized:
                raise RuntimeError(
                    "AnchorBranch.forward() called with no anchor_texts before "
                    "any call to recompute_anchors(); the cached anchor buffer "
                    "is still all-zero. Call recompute_anchors(anchor_texts, A) "
                    "once (e.g. at the start of epoch 0) before training/eval."
                )
            anchors = self.anchor_embeddings.to(device)

        cos_sim = self.similarity(h, anchors)
        return {"anchors": anchors, "cos_sim": cos_sim}

    @staticmethod
    def predict(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Multi-label inference utility: sigmoid + fixed threshold.

        Args:
            logits: [B, num_classes] raw (pre-sigmoid) logits -- e.g.
                the final fused/LACR-refined logits produced downstream
                by Person 2, or `cos_sim` on its own for a quick check.
            threshold: probability threshold applied uniformly across
                all classes. For per-class calibrated thresholds, use
                `person3_training_eval_pipeline.calibrate_thresholds`
                and threshold externally instead of this helper.
        Returns:
            preds: [B, num_classes] multi-hot binary tensor (0.0/1.0),
                same dtype/device as the sigmoid probabilities.
        """
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        return preds


class Person1InterfaceReference:
    """
    Shows how the pieces above wire together end-to-end, matching
    TASKS.md's Person 1 deliverable (`build_anchors` -> `E_lac`,
    `similarity(h, E_lac)` -> `Y`):

        shared_encoder = SharedEncoder()                       # built once
        anchor_branch  = AnchorBranch(shared_encoder, shared_encoder.tokenizer)
        # (tokenizer is typically owned by whoever tokenizes sentences too;
        # SharedEncoder itself does not hold one -- pass the same
        # AutoTokenizer used for sentence tokenization.)

        A = build_adjacency_matrix(train_labels, held_out_classes=[...])  # [28, 28]

        # once per epoch (E_lac recomputed from current encoder + LAC weights):
        anchor_texts = AnchorBranch.default_anchor_texts(exemplar_sentences)
        anchor_branch.recompute_anchors(anchor_texts, A)

        # every forward pass:
        h = shared_encoder(sentence_ids, sentence_mask)   # [B, 768]
        out = anchor_branch(h)                             # cached E_lac path
        E_lac, Y = out["anchors"], out["cos_sim"]           # [28, 768], [B, 28]

        # handed off to Person 2:
        X = person2.classifier_head(h)
        z = X + person2.fusion_weight(Y)
    """
    pass
