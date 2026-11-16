# Tabular LLMs, Generative Transformers & Foundation Model Recipes

Modern tabular competitions (especially GReaT/LLM-synthesized Playground Series datasets and high-signal tabular benchmarks) have seen **generative causal language models, any-order transformers, and full-context foundation models** outperform standalone GBDTs and dominate stacking weights (~70%–80% of final meta-learner output in S6E10).

Refer to [`Handbook/tabular/playground-s6e10-airline-satisfaction.md`](../../../../Handbook/tabular/playground-s6e10-airline-satisfaction.md) and [`Handbook/tabular/playground-s6e9-will-buy-ev.md`](../../../../Handbook/tabular/playground-s6e9-will-buy-ev.md) for full benchmark evidence and ablations.

---

## 1. Recipe 1: Target-First Tabular LLM ("The Storyteller" — GReaT Replica + Bayes Log-Sum-Exp)

Introduced by Chris Deotte (`@cdeotte`) in S6E9 (`0.946485` OOF) and S6E10 (`0.962425` OOF single model / `0.962452` Target-Anywhere), **The Storyteller** flips classification from $P(y \mid \mathbf{x})$ to generative density estimation $P(\mathbf{x} \mid y)$, inverting via Bayes' rule. It directly mirrors `GReaT` (`be-great`) data generation.

### Step 1: Text Serialization & Replica Pretraining (Original Data Only)
Format every row identically to how `GReaT` writes training text (`ints` as `'36'`, `floats` as `'0.0'`, `NaN` as `'None'`, target as `'True'`/`'False'`), joined by `", "` using **GPT-2's BPE tokenizer** (digit-by-digit tokenizers such as Qwen performed worse on Playground GReaT data):

```python
def format_row_great(row_dict: dict, cols_order: list[str]) -> str:
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

Pretrain a **GReaT replica exclusively on the original dataset** (zero competition labels) for **60 epochs** (a 60-epoch replica beat an 8-epoch replica by `+0.0012` final AUC in S6E10; starting from plain GPT-2 is much worse):

```python
from be_great import GReaT

# Pretrain replica solely on original_df (e.g. 129,880 rows)
replica = GReaT(llm="gpt2", batch_size=16, epochs=60)
replica.fit(original_df)
# Save state_dict for fold classifiers (gpt2, gpt2-medium, gpt2-large, EleutherAI/gpt-neo-125M)
```

### Step 2: Fold Classifier Fine-Tuning
For each CV fold, initialize from the 60-epoch replica checkpoint and fine-tune on the fold's training rows with **the target placed first** (or at a random position among all columns for the **Target-Anywhere** variant) and all feature columns in a fresh random order every step, computing causal LM cross-entropy across all tokens (`8 epochs`, `AdamW`, `lr=5e-5` linear decay to `0`, `batch_size=32`, `bf16` on A100/L4; use `fp16` + `GradScaler` on T4 GPUs where `bf16` emulates slowly).

### Step 3: Multi-Order Bayes Readout (`logsumexp`)
Score each validation/test row twice (with prefix `"target is True,"` and `"target is False,"`) across `K=30` random feature permutations and pool via `logsumexp`:

```python
import math
import torch
import numpy as np
from scipy.special import logsumexp

def score_storyteller_row(model, tok, feat_dict: dict, cols: list[str], prior_pos: float, n_orders: int = 30, rng=None) -> float:
    """Compute Bayes log-odds(y=1 | x) averaged over n_orders random column permutations."""
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

    n_true = len(tok("satisfaction is True,").input_ids)
    n_false = len(tok("satisfaction is False,").input_ids)

    ll_true, ll_false = [], []
    for _ in range(n_orders):
        perm = rng.permutation(cols).tolist()
        body = " " + format_row_great(feat_dict, perm) + eos
        ll_true.append(log_p("satisfaction is True," + body, n_true))
        ll_false.append(log_p("satisfaction is False," + body, n_false))

    log_prior_ratio = math.log(prior_pos / (1.0 - prior_pos))
    return log_prior_ratio + float(logsumexp(ll_true) - logsumexp(ll_false))
```

- **Target-Anywhere Variant**: Place the target at a random column position during training (`24 epochs` on GPT-2 small reached `0.962452` OOF in S6E10) and score by averaging per-order log-likelihood ratios over 30 random orders.
- **Warning on Hybrid BCE**: Adding a discriminative BCE loss term inside the generative fine-tune (`GenHybrid`) shifts the model away from pure $P(\mathbf{x} \mid y)$ density and typically *lowers* both standalone OOF and stack diversity compared to pure generative training (`GenPure`).

---

## 2. Recipe 2: Target-Last Tabular LLM ("The Detective" — Single-Pass Causal Classifier)

When data is *not* generated by an autoregressive LM (or when fast 1-pass inference is required), **The Detective** trains a causal LM on shuffled feature prompts with the target placed **last**:

```text
Gender is Female, Customer Type is Loyal Customer, ..., Arrival Delay in Minutes is 0.0, satisfaction is True
```

- **Training**: Shuffle feature order every step (treating the row as an order-invariant set). Compute loss on all tokens or only the label token.
- **1-Pass Inference**: Feed text ending right after `"<target> is"` and extract softmax probabilities of the positive vs. negative label tokens:

```python
def predict_detective(model, tok, prompt_without_label: str, pos_tok=" True", neg_tok=" False") -> float:
    ids = tok(prompt_without_label, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    with torch.no_grad():
        probs = model(ids).logits[0, -1].softmax(-1)
    t_id = tok(pos_tok, add_special_tokens=False).input_ids[0]
    f_id = tok(neg_tok, add_special_tokens=False).input_ids[0]
    return float((probs[t_id] / (probs[t_id] + probs[f_id])).item())
```

*(Note: Keep Storyteller and Detective models together in the stack—they read rows in opposite directions and yield complementary errors.)*

---

## 3. Recipe 3: From-Scratch Any-Order / Bagged Generative Transformers (`AO-ARM` & `GenPure`)

If pretrained GPT-2 replicas are unavailable or columns have specialized numeric structure, build an **Any-Order Autoregressive Transformer** (`AO-ARM`, 4th Place S6E10, `0.96217` CV) or **Bagged Generative Transformer** (`GenPure`, 7th Place S6E10, `0.96119` OOF) from scratch over column/value tokens:

1. **Token Vocabulary**: Map each column to `C` discrete tokens (`1 token per column`). Group rare continuous/integer values appearing `< 3` times in train+test into coarse domain/$\log_2$ bins so embeddings never overfit singletons.
2. **Order-Free Causal Training**:
   - Architecture: 6-layer, width-256, 8-head causal Transformer (`Pre-LayerNorm`) with learned column + value embeddings and **no fixed positional order**.
   - **Semi-Supervised Feature Likelihood (`weight = 0.5`)**: Pretrain label-free on `train + test + original` rows (or feed unlabeled validation/test feature-only likelihood at `weight=0.5` during training, zero labels seen).
   - **No Early Stopping**: Run full cosine decay schedules (**36 epochs**; turning off `patience=4` early stopping prevented `-0.0009` to `-0.0016` under-training in `GenPure`).
3. **Bayes Inference + Bagging**:
   - Score $\log P(\mathbf{x}, y=1) - \log P(\mathbf{x}, y=0)$ averaged (`log-mean-exp` / `logsumexp`) across **8–32 random column permutations**.
   - Calibrate per fold with Platt scaling on a 10% inner holdout and **bag multiple seeds (`8–16` seeds)** directly into the $L_2$ logistic meta-stacker.

---

## 4. Recipe 4: Self-Supervised & Order-Free Feature-Token Transformers (`Q-FT v2` & RTD Encoders)

For discriminative tabular transformers (4th, 9th, and 11th Place S6E10):
1. **Order-Free Set Encoder**: Feed each column as `col_embed + value_embed + periodic_num_embed + side_features` into a bidirectional Transformer (`ModernBERT-base` or custom QK-norm Transformer `d=256..1024`, 4–8 layers) with **all position IDs fixed to `0`** (so rows act as permutation-invariant sets) plus 1–2 `[CLS]` / summary tokens.
2. **Self-Supervised Pretraining on `Train + Test + Original` (No Labels)**:
   - Pretrain using **Masked-Token Reconstruction** and **Replaced-Token Detection (RTD)** — especially **Near-Value / Hard-Negative RTD** (corrupting a feature token with a plausible nearby value from the column's distribution) for `3–2,000` epochs depending on model size.
   - *Important*: Train from scratch with RTD rather than initializing with NLP text weights (text-pretrained `ModernBERT` scored `0.96019` vs `0.96061` from scratch in S6E10).
3. **Supervised Fine-Tuning**: Fine-tune 5-fold for `2–4` epochs with low lr (`AdamW` or `SAM` optimizer, differential/gated attention variants for diversity).

---

## 5. Recipe 5: Synthetic Labelled "Replica Worlds" Source Pretraining

When the competition dataset is synthetic and an original seed dataset exists (`data.csv`):
1. Train a noised GReaT replica solely on `original_df` (never touching competition `train.csv` or `test.csv`), applying calibrated label-flip noise and $\pm 1$ discrete jitter to match the competition's noise rate.
2. Sample 1M+ synthetic labelled rows (`Replica Worlds`, e.g., `M32_1M` and `MB32_1M`).
3. **Pretrain** field transformers (`m19_source_aug`, `0.961409` OOF) and BatchEnsemble MLPs (`field_be`, `pair_be`) on the Replica Worlds before fine-tuning on the 5 competition folds.

---

## 6. Recipe 6: Zero-Shot Full-Context Tabular Foundation Model (TFM) Zoo

Never stop at a single TFM. On raw columns (without feeding other models' OOF predictions, which leaks targets via cross-attention!), evaluate the diverse TFM zoo on the shared 5 folds:

| Foundation Model | S6E10 OOF AUC (Raw) | Key Operational Tips |
| :--- | :---: | :--- |
| **`TabPFN-3.5`** | **0.961145** | Declare high-cardinality physical IDs (e.g., `Flight Distance`) as **categorical**; scale context rows via `fit_mode='fit_with_cache'`. Often gets the **highest non-GPT weight** in stacks. |
| **`Kumo-Tabular` (large / small)** | 0.960036 / 0.958484 | Strong zero-shot inductive bias orthogonal to trees. |
| **`TabFM 1.0`** | 0.958871 | Pretrained tabular foundation transformer. |
| **`EXAONE-Tabular`** | 0.958557 | Distinct pretraining corpus & tokenization. |
| **`LimiX-2`** | 0.958474 | Complementary foundation tabular model. |
| **`Causilo`** | 0.958275 | Foundation model focused on structural relationships. |
| **`TabICLv2`** | 0.958088 | Scales well with original rows appended + fine-tuning on larger context (`14k` rows). |
| **`Mitra-v2`** | 0.957075 | Additional zero-shot diversity for meta-stacking. |
| *9 TFMs Stacked Together* | **0.96120** | Outperforms tuned XGBoost (`0.96149` tuned, `0.95985` raw) in blend diversity (`+1.4e-4` over 219 non-TFM models). |
