# Tabular LLMs, Generative Transformers & Foundation Model Recipes

Generative causal language models, any-order transformers, and full-context tabular foundation models provide strong orthogonal signal to GBDTs and MLPs—frequently capturing the dominant share of meta-stacker weights in synthetic and high-signal tabular benchmarks.

---

## 1. Recipe 1: Target-First Tabular LLM ("The Storyteller" — Generative Replica + Bayes Log-Sum-Exp)

**The Storyteller** flips discriminative classification $P(y \mid \mathbf{x})$ into conditional generative density estimation $P(\mathbf{x} \mid y)$, inverting via Bayes' rule. This is particularly effective on LLM-synthesized tables (`GReaT` / autoregressive tabular synthesizers).

### Step 1: Text Serialization & Original-Data Pretraining
Format every row deterministically (`ints` as `'36'`, `floats` as `'0.0'`, `NaN` as `'None'`, binary target as `'True'`/`'False'`), joined by `", "` using a byte-pair tokenizer such as **GPT-2 BPE** (preserves multi-digit numerical tokens better than digit-by-digit tokenizers on standard LLM-generated tables):

```python
import numpy as np


def format_row_tabular_text(row_dict: dict, cols_order: list[str]) -> str:
    """Serialize a tabular row into comma-separated 'col is val' text."""
    parts = []
    for c in cols_order:
        v = row_dict[c]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            s = "None"
        elif isinstance(v, (float, np.floating)):
            s = str(float(v))
        else:
            s = str(v)
        parts.append(f"{c} is {s}")
    return ", ".join(parts)
```

When an uncorrupted source/reference dataset exists, pretrain a causal LM replica **exclusively on original reference rows** (never touching competition validation/test labels) for a full cosine schedule (e.g., `60 epochs`, matching full replica convergence rather than short `8-epoch` warmups):

```python
from be_great import GReaT

# Pretrain generative replica solely on reference/original data
replica = GReaT(llm="gpt2", batch_size=16, epochs=60)
replica.fit(original_df)
# Reuse saved weights to initialize per-fold fine-tuning (gpt2, gpt2-medium, gpt2-large, gpt-neo-125M)
```

### Step 2: Fold Classifier Fine-Tuning
For each CV fold, initialize from the pretrained replica checkpoint and fine-tune on the fold's training rows with **the target placed first** (or at a uniformly sampled random column position for the **Target-Anywhere** variant) and all feature columns in a fresh random permutation every step. Optimize causal LM cross-entropy across all tokens (`AdamW`, `lr=5e-5` linear decay to `0`, `bf16` on Ampere/Blackwell/Hopper GPUs; use `fp16` + `GradScaler` on Turing GPUs).

### Step 3: Multi-Order Bayes Readout (`logsumexp`)
Score each validation/test row under both target prefixes across `n_orders` (typically `16–32`) random feature permutations and pool via `logsumexp`:

```python
import math
import numpy as np
import torch
from scipy.special import logsumexp


def score_storyteller_row(
    model,
    tok,
    feat_dict: dict,
    feat_cols: list[str],
    prior_pos: float,
    target_col: str = "target",
    pos_val: str = "True",
    neg_val: str = "False",
    n_orders: int = 30,
    rng=None,
) -> float:
    """Compute Bayes log-odds(y=1 | x) pooled across random column permutations."""
    if rng is None:
        rng = np.random.default_rng(42)
    eos = tok.eos_token
    device = next(model.parameters()).device

    def log_p(text: str, n_prefix: int) -> float:
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            lp = model(ids).logits[0, :-1].log_softmax(-1)
            tok_lp = lp.gather(-1, ids[0, 1:, None])[:, 0]
        return tok_lp[n_prefix - 1:].sum().item()

    pos_prefix = f"{target_col} is {pos_val},"
    neg_prefix = f"{target_col} is {neg_val},"
    n_pos = len(tok(pos_prefix).input_ids)
    n_neg = len(tok(neg_prefix).input_ids)

    ll_pos, ll_neg = [], []
    for _ in range(n_orders):
        perm = rng.permutation(feat_cols).tolist()
        body = " " + format_row_tabular_text(feat_dict, perm) + eos
        ll_pos.append(log_p(pos_prefix + body, n_pos))
        ll_neg.append(log_p(neg_prefix + body, n_neg))

    log_prior_ratio = math.log(prior_pos / (1.0 - prior_pos))
    return log_prior_ratio + float(logsumexp(ll_pos) - logsumexp(ll_neg))
```

- **Target-Anywhere Variant**: Place the target at a random column position during training (~20–24 epochs on compact backbones) and score by averaging per-order log-likelihood ratios across permutations.
- **Avoid Hybrid BCE Collapse**: Adding a discriminative BCE loss directly inside generative Storyteller fine-tuning shifts weights away from pure $P(\mathbf{x} \mid y)$ density estimation and often hurts both standalone accuracy and ensemble orthogonality compared to pure generative cross-entropy (`GenPure`).

---

## 2. Recipe 2: Target-Last Tabular LLM ("The Detective" — Single-Pass Causal Classifier)

When fast 1-pass inference is required or the data does not follow a GReaT-style synthesis process, **The Detective** trains a causal LM on randomly permuted feature prompts with the target placed **last**:

```text
<col_1> is <val_1>, <col_2> is <val_2>, ..., <col_M> is <val_M>, <target_col> is <label_val>
```

- **Training**: Shuffle feature order every step so the model learns an order-invariant tabular prompt. Compute cross-entropy loss across all tokens or strictly on the final label token.
- **1-Pass Inference**: Pass the prompt truncated right after `"<target_col> is"` and extract normalized softmax probabilities over positive vs. negative label tokens:

```python
import torch


def predict_detective(model, tok, prompt_without_label: str, pos_tok: str = " True", neg_tok: str = " False") -> float:
    """Single-pass next-token conditional classification probability."""
    ids = tok(prompt_without_label, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    with torch.no_grad():
        probs = model(ids).logits[0, -1].softmax(-1)
    t_id = tok(pos_tok, add_special_tokens=False).input_ids[0]
    f_id = tok(neg_tok, add_special_tokens=False).input_ids[0]
    return float((probs[t_id] / (probs[t_id] + probs[f_id])).item())
```

*(Note: Keep Storyteller and Detective predictions together in the meta-stacker—bidirectional factorization orders produce decorrelated prediction errors.)*

---

## 3. Recipe 3: From-Scratch Any-Order & Bagged Generative Transformers (`AO-ARM` / `GenPure`)

When pretrained text LLMs underperform due to specialized numeric vocabularies or non-English schemas, train a compact **Any-Order Autoregressive Transformer** from scratch directly over discrete table tokens:

1. **Compact Column-Value Vocabulary**: Map each column to discrete value tokens (`1 token per column`). Collapse singleton continuous/integer values occurring fewer than `3` times across `train + test` into coarse domain or $\log_2$ quantile bins so value embeddings do not memorize noise.
2. **Order-Free Causal Transformer**:
   - Use a Pre-LayerNorm causal Transformer (e.g., 6 layers, `d_model=256`, 8 heads) combining learned `col_embed + val_embed` with **zero fixed positional order**.
   - **Semi-Supervised Unlabeled Feature Likelihood (`weight = 0.5`)**: Pretrain label-free on `train + test (+ orig)` rows, or include unlabeled feature-only causal likelihood from validation/test predictors (`weight = 0.5`) during training (strictly zero validation/test labels).
   - **No Early Stopping**: Run full cosine LR decay schedules (e.g., `36 epochs`); patience-based early stopping under-trains autoregressive feature densities.
3. **Bayes Inference + Calibration + Bagging**:
   - Compute $\log P(\mathbf{x}, y=1) - \log P(\mathbf{x}, y=0)$ pooled over `8–32` random column orders via `logsumexp`.
   - Fit Platt calibration on an inner holdout per fold and bag `8–16` random seeds before passing logits to a regularized linear stacker.

---

## 4. Recipe 4: Self-Supervised Order-Free Feature-Token Transformers (Masked & RTD Encoders)

For discriminative set transformers over tabular columns:
1. **Permutation-Invariant Tokenization**: Encode each column as `col_embed + val_embed + periodic_num_embed + auxiliary_stats` inside a bidirectional Transformer (e.g., QK-norm Pre-LN Transformer or `ModernBERT`-style backbone, 4–8 layers) with **all position IDs fixed to `0`** plus `1–2` `[CLS]` pooling tokens.
2. **Self-Supervised Pretraining on Unlabeled `Train + Test (+ Orig)`**:
   - Optimize **Masked-Token Reconstruction** jointly with **Hard-Negative Replaced-Token Detection (RTD)**—corrupting tokens with plausible nearby values from the same column's empirical distribution rather than uniform random draws.
   - Train weights **from scratch via RTD** rather than text-pretrained NLP weights whenever numeric/tabular relationships dominate.
3. **Supervised Fine-Tuning**: Fine-tune 5-fold for `2–4` epochs with low learning rate (`AdamW` or `SAM`, optionally pairing standard softmax attention with differential/gated attention variants for ensemble diversity).

---

## 5. Recipe 5: Synthetic Labelled "Replica Worlds" Source Pretraining

When a competition dataset is synthesized from an uncorrupted seed dataset:
1. Train a noised generative replica strictly on the **original dataset** (zero competition `train.csv` or `test.csv` rows), calibrating label-flip noise and discrete $\pm 1$ jitter to approximate the target competition's noise level.
2. Sample 1M+ synthetic labelled rows (**Replica Worlds**).
3. **Pretrain** field-token transformers or BatchEnsemble MLPs on Replica Worlds before fine-tuning across competition CV folds.

---

## 6. Recipe 6: Zero-Shot Full-Context Tabular Foundation Model (TFM) Zoo

Never stop at a single foundation model—different TFM pretraining priors yield orthogonal error modes on tabular manifolds:

| Model Family | Universal Operational Practices |
| :--- | :--- |
| **`TabPFN-3.5`** | Explicitly cast high-cardinality integer/decimal physical keys as **categorical** when they encode discrete generator IDs; scale to full training folds via `fit_mode='fit_with_cache'` and `ignore_pretraining_limits=True`. |
| **`Kumo-Tabular` (large / small)** | Evaluate multiple capacity tiers for non-tree inductive bias in linear logit stacks. |
| **`TabICLv2`** | Scales strongly when calibrated with original-source rows appended to context + larger context windows (`14k+` rows). |
| **`TabFM` / `EXAONE-Tabular` / `LimiX-2` / `Causilo` / `Mitra-v2`** | Diverse pretraining corpora and structural tokenization priors; pool as raw zero-shot predictors inside Level-2 stacking. |

> [!CAUTION]
> **OOF Leakage Guardrail for In-Context TFMs**: Never pass out-of-fold predictions of other models as input features to full-context attention models (`TabPFN`, `TabICL`, etc.). Because in-context self-attention attends across support rows whose labels are visible, target-derived columns leak validation labels inside attention blocks. Feed only raw predictors and label-free features.
