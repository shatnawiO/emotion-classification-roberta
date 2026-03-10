# Emotion Classification from Text using RoBERTa

A multi-label emotion classification model fine-tuned on the **GoEmotions** dataset using **RoBERTa-base**. The model classifies Reddit comments into 28 fine-grained emotion categories such as joy, anger, sadness, fear, curiosity, and more.

---

## Test Set Results

| Metric | Score |
|---|---|
| Accuracy | 0.454 |
| Micro F1 | 0.5788 |
| Macro F1 | 0.5284 |
| Micro Precision | 0.5875 |
| Micro Recall | 0.5704 |
| Optimal Threshold | 0.7 |

> GoEmotions is a highly imbalanced multi-label benchmark with 28 emotion classes. Achieving Micro F1 > 0.57 on this dataset is competitive for a fine-tuned RoBERTa-base without data augmentation.

---

## Dataset

| Property | Details |
|---|---|
| Name | [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) |
| Source | Google Research |
| Size | ~58,000 Reddit comments |
| Labels | 28 emotion categories |
| Task | Multi-label classification |

---

## Model & Training

| Property | Details |
|---|---|
| Base Model | `roberta-base` (HuggingFace) |
| Framework | PyTorch + HuggingFace Trainer |
| Loss Function | `BCEWithLogitsLoss` with class weights |
| Learning Rate | 2e-5 |
| Batch Size | 32 |
| Epochs | 7 |
| Max Sequence Length | 128 tokens |
| Seed | 42 |

---

## Pipeline

### 1. Data Loading
Load GoEmotions directly from HuggingFace `datasets`.

### 2. Class Weight Calculation
Computed per-label positive weights using square root scaling to handle severe label imbalance (e.g. "grief" has only 77 samples vs "neutral" with 14,219). Weights were clipped to a maximum of 15x to avoid over-correction.

### 3. Label Encoding
Converted sparse label indices into **multi-hot float32 vectors** of size 28 for multi-label compatibility with `BCEWithLogitsLoss`.

### 4. Tokenization
Tokenized all splits (train/validation/test) using the RoBERTa tokenizer with `max_length=128` and `padding="max_length"`.

### 5. Custom Weighted Trainer
Extended HuggingFace `Trainer` with a custom `WeightedTrainer` class that applies the per-class positive weights during loss computation.

### 6. Threshold Tuning
During evaluation, tested thresholds from 0.3 to 0.7 and selected the best threshold based on Macro F1. Optimal threshold found: **0.7**.

### 7. Evaluation
Evaluated on validation and test sets using:
- `accuracy_score`
- `precision_recall_fscore_support` (micro & macro)
- Per-emotion accuracy analysis across all 28 classes

---

## Requirements

```
torch
transformers
datasets
scikit-learn
numpy
tqdm
```

Install with:

```bash
pip install torch transformers datasets scikit-learn numpy tqdm
```

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/shatnawiO/emotion-classification-roberta.git
cd emotion-classification-roberta
```

2. Install dependencies:
```bash
pip install torch transformers datasets scikit-learn numpy tqdm
```

3. Open the notebook in VS Code or Jupyter:
```bash
jupyter notebook finetune.ipynb
```

4. Run all cells from top to bottom. Training takes ~7 epochs and requires a GPU for reasonable speed.

---

## Per-Emotion Accuracy (Training Set)

The model achieves high per-label accuracy on most emotions, with the hardest being `neutral` (0.869) due to class overlap.

| Emotion | Accuracy |
|---|---|
| admiration | 0.957 |
| amusement | 0.981 |
| anger | 0.972 |
| neutral | 0.869 |
| grief | 0.998 |
| embarrassment | 0.995 |

*(Full results available in the notebook output)*

---

## Author

**Mohammad Shatnawi**  
AI Student — Jordan University of Science and Technology  
GitHub: [@shatnawiO](https://github.com/shatnawiO)  
LinkedIn: [mohamed-shatnawi](https://linkedin.com/in/mohamed-shatnawi-70408b303)
