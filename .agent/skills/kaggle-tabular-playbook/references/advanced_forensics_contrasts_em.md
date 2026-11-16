# Advanced Tabular Forensics, Contrast Carriers & Semi-Supervised EM

This cookbook provides universal, competition-agnostic recipes and drop-in Python patterns for **marginal residual extraction**, **fold-safe semi-supervised density estimation**, **multi-resolution target encoding regularization**, **generative replica forensics**, **label-free logit alignment**, and **class-weighted multiclass stacking**.

---

## 1. Contrast Carriers (`real - zero` Residual Chains & Cross-Family Contrasts)

When adding a subtle engineered feature block to a mature 30+ model ensemble, standard retrained models correlate at $\rho > 0.99$ with existing models, drowning the marginal signal in meta-learner collinearity. **Contrast Carriers** isolate the pure directional delta induced by the new feature group.

### A. Same-Seed Feature-Ablation Contrast Chain
Train the exact same deterministic model architecture and random seeds twice per fold:
- `real`: candidate feature block active.
- `zero`: candidate feature block zeroed out (identical row order, seeds, and hyperparameters).

Feed the delta direction into a residual chain clipped safely in logit space:

```python
import numpy as np
from scipy.special import logit


def build_ablation_contrast_chain(
    p_base: np.ndarray,
    p_real: np.ndarray,
    p_zero: np.ndarray,
    alpha: float = 2.0,
    eps: float = 1e-6,
    clip_val: float = 30.0,
) -> np.ndarray:
    """Isolate pure marginal feature-block lift in logit space."""
    z_base = np.clip(logit(np.clip(p_base, eps, 1.0 - eps)), -clip_val, clip_val)
    z_real = np.clip(logit(np.clip(p_real, eps, 1.0 - eps)), -clip_val, clip_val)
    z_zero = np.clip(logit(np.clip(p_zero, eps, 1.0 - eps)), -clip_val, clip_val)
    return np.clip(z_base + alpha * (z_real - z_zero), -clip_val, clip_val)
```

### B. Probit Interaction & Cross-Family Contrasts
- **Two-Block Non-Additive Interaction Contrast**: Train four matched-seed runs ($\varnothing$, $A$, $B$, $A+B$) and extract $\Delta_{\text{int}} = -(z_{A+B}) + z_A + z_B - z_{\varnothing}$ (optionally in probit space $\Phi^{-1}(p)$).
- **Cross-Family Contrasts**: Expose amplified structural differences between orthogonal model families directly to the linear stacker:
  $$z_{\text{contrast}} = a \cdot z(\text{Family}_1) - b \cdot z(\text{Family}_2)$$
  *(Note: Even when standalone OOF AUC of a contrast carrier is lower than the best tree model, regularized logistic/Ridge stackers exploit its orthogonal residual vector cleanly.)*

---

## 2. Fold-Safe Semi-Supervised Exact-Value EM & Context-Shape Residuals

In tabular datasets where exact discrete numerical values repeat across synthetic `train` and `test` rows (e.g., sampled from finite discrete balances, salaries, or dosages), each unique value $v$ carries a latent target rate $\theta_v$. Instead of estimating $\theta_v$ from noisy sparse labeled counts alone, pool labeled fold-train counts with **unlabeled `val + test` context posteriors** inside an EM loop:

```python
import numpy as np
import pandas as pd
from scipy.special import expit, logit


def fit_fold_exact_value_em(
    train_vals: np.ndarray,
    y_train: np.ndarray,
    ctx_logit_train: np.ndarray,
    unlab_vals: np.ndarray,
    ctx_logit_unlab: np.ndarray,
    prior: float,
    prior_strength: float = 12.0,
    w_unlab: float = 1.0,
    n_iter: int = 8,
) -> dict:
    """
    Estimate fold-safe semi-supervised class rates theta_v per exact repeated value v.
    IMPORTANT: 'ctx_logit_*' must come from a value-blind teacher trained ONLY on fold train rows;
    'unlab_vals' (outer-val + test) must NEVER expose validation labels.
    """
    base_logit = logit(np.clip(prior, 1e-6, 1.0 - 1e-6))

    # Labeled sufficient statistics by exact value
    df_tr = pd.DataFrame({"v": train_vals, "y": y_train, "ctx": ctx_logit_train})
    grp_tr = df_tr.groupby("v")["y"].agg(["sum", "count"])
    all_vals = np.union1d(grp_tr.index.values, np.unique(unlab_vals))
    pos_tr = grp_tr["sum"].reindex(all_vals, fill_value=0.0).to_numpy(dtype=np.float64)
    tot_tr = grp_tr["count"].reindex(all_vals, fill_value=0.0).to_numpy(dtype=np.float64)

    val_to_idx = {v: i for i, v in enumerate(all_vals)}
    unlab_idx = np.array([val_to_idx[v] for v in unlab_vals], dtype=np.int64)

    # Initialize theta_v with Beta prior shrinkage
    theta = (pos_tr + prior_strength * prior) / (tot_tr + prior_strength)
    theta_logit_offset = logit(np.clip(theta, 1e-6, 1.0 - 1e-6)) - base_logit

    for _ in range(n_iter):
        # E-step on unlabeled (val + test) rows: combine value-blind context LLR + current theta_v
        post_unlab = expit(ctx_logit_unlab + theta_logit_offset[unlab_idx])
        pos_unlab = np.bincount(unlab_idx, weights=post_unlab, minlength=len(all_vals))
        tot_unlab = np.bincount(unlab_idx, minlength=len(all_vals)).astype(np.float64)

        # M-step with unlabeled evidence weight w_unlab (sweep e.g. 0.25, 1.0, 4.0)
        theta_new = (pos_tr + w_unlab * pos_unlab + prior_strength * prior) / (
            tot_tr + w_unlab * tot_unlab + prior_strength
        )
        theta = 0.5 * theta + 0.5 * theta_new  # Damped update for stability
        theta_logit_offset = logit(np.clip(theta, 1e-6, 1.0 - 1e-6)) - base_logit

    return {v: float(theta_logit_offset[i]) for i, v in enumerate(all_vals)}
```

- **Value-Blind Context Teacher**: Always train the auxiliary context GBDT *excluding* the target numeric column (and any direct proxy of it), and calibrate its mean prediction per coarse bucket (e.g., `val // 1000` or quantile bin) so global scale shifts do not distort `theta_v`.
- **Multi-Weight Mixture Features**: Extract `theta_logit_offset` across `w_unlab` $\in \{0.25, 1.0, 4.0\}$ together with observed Fisher information `tot_tr + w_unlab * tot_unlab`.
- **Context-Shape Residuals**: Measure how the empirical frequency of auxiliary binary/categorical indicators among rows sharing exact value $v$ deviates from the rate predicted solely by $\theta_v$ and the class conditionals.

---

## 3. Multi-Resolution TE Regularization in GBDTs (`noise` + `feature_weights`)

When feeding large libraries (100–500 columns) of multi-resolution target encodings and resolution differences into GBDTs, uniform `colsample_bytree` causes greedy trees to overfit high-cardinality smoothed encodings. Regularize via two strict mechanisms:

1. **Training-Only Gaussian Noise on Target-Derived Columns**: Inject Gaussian noise (`sd = 0.04`) strictly onto target-encoded columns during fit (never touch validation or test rows).
2. **Differential Per-Column Sampling Weights (`feature_weights`)**:

```python
import numpy as np
import xgboost as xgb


def add_train_only_gaussian_noise(X_tr_df, te_cols: list[str], sd: float = 0.04, seed: int = 42):
    """Add zero-mean Gaussian jitter strictly to target-derived columns on training rows."""
    rng = np.random.default_rng(seed)
    X_out = X_tr_df.copy()
    noise = rng.normal(0.0, sd, size=(len(X_out), len(te_cols))).astype(np.float32)
    X_out[te_cols] = X_out[te_cols].to_numpy(dtype=np.float32) + noise
    return X_out


def build_xgb_multires_dmatrix(X_df, y, col_tiers: dict[str, float]):
    """
    Build an XGBoost DMatrix assigning differential column sampling weights:
      - raw / structural features: 1.0
      - base single-scale TEs:     0.6
      - multi-resolution TEs:      0.2
      - resolution difference cols: 0.05
    """
    weights = [col_tiers.get(c, 1.0) for c in X_df.columns]
    return xgb.DMatrix(X_df, label=y, feature_weights=weights)
```

---

## 4. All-Target-Position Chain, Per-Field Surprisal & Acceptance Entropy

When scoring rows with frozen or per-fold causal tabular LLMs (`"col is val"` serialization):

1. **All-Target-Position Chain**: For a column permutation $o$ and target insertion index $p \in \{0, \dots, M\}$, all tokens *strictly before* index $p$ are shared between $y=1$ and $y=0$ and cancel identically:
   $$\text{LLR}(o, p) = \Big[\log P(y=1 \mid \mathbf{x}_{<p}) + \log P(\mathbf{x}_{>p} \mid y=1, \mathbf{x}_{<p})\Big] - \Big[\log P(y=0 \mid \mathbf{x}_{<p}) + \log P(\mathbf{x}_{>p} \mid y=0, \mathbf{x}_{<p})\Big]$$
   Pooling $\text{LLR}(o, p)$ across all $M+1$ label positions yields strong marginal lift over target-first scoring alone.
2. **Per-Field Pseudo-Likelihood (Surprisal)**: Hold each column $j$ at the end of a target-first sequence (`target is True, ..., col_j is val_j`) under frozen pretraining replicas to compute per-field likelihood and `True - False` log-ratio features for downstream GBDTs/MLPs.
3. **Generator Acceptance Entropy**: Under label-first prefixes (`True` vs `False`), compute the Shannon entropy of the generative LM's next-token distribution restricted strictly to valid initial tokens for field $j$.

---

## 5. Label-Free Per-Fold Test Logit Alignment & Frozen LLM Internal Heads

### A. Label-Free Test-Distribution Logit Alignment
Across CV folds, deep networks and generative classifiers can drift slightly in logit offset or temperature. Before averaging $K$-fold test predictions (or feeding fold OOFs to a stacker), align each fold's logits strictly using that fold's own unlabeled test mean and standard deviation:

```python
import numpy as np
from scipy.special import expit, logit


def align_fold_test_logits(
    oof_fold: np.ndarray,
    test_fold: np.ndarray,
    scale: float = 3.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Label-free per-fold standardization against fold test logit distribution."""
    z_test = logit(np.clip(test_fold, eps, 1.0 - eps))
    z_oof = logit(np.clip(oof_fold, eps, 1.0 - eps))
    mu = float(np.mean(z_test))
    sd = float(np.std(z_test) + 1e-12)
    return expit(scale * (z_oof - mu) / sd), expit(scale * (z_test - mu) / sd)
```

### B. Frozen Per-Fold LLM Hidden Projections + BatchEnsemble Skip Head
Extract frozen features from per-fold fine-tuned causal tabular LLMs across forward and reversed field orders:
- Restrict softmax over exact table vocabulary tokens (`val_tokens`), logging per-field total token log-probs, value-token entropies, and `True - False` deltas.
- Project mean and final hidden states with a fixed `32-dim` orthonormal/random projection.
- Train a lightweight **4-member BatchEnsemble MLP** (`width=384`, `SiLU`, `dropout=0.1`) with a direct linear skip connection from the raw Storyteller LLR.

---

## 6. Cross-Fitted Field Reconstruction, MLM Density (`mfill`/`qfill`) & Source-Donor Attention

For noisy synthetic tables or latent structural constraints:
1. **Cross-Fitted Field Reconstruction Residuals**: Train inner cross-fitted Autoencoders and GBDTs predicting each column $j$ from all remaining columns $\mathbf{x}_{\setminus j}$, adding $\hat{x}_j$ and reconstruction residuals $x_j - \hat{x}_j$ as features.
2. **MLM Posterior Expectations & Uncertainty (`mfill`, `qfill`, `cmp`)**: Pretrain a Masked-Language-Model Transformer over discretized tabular tokens to extract expected imputed token values (`mfill`), posterior quantile spread/skew (`qfill`), and full-row completion log-densities (`cmp`).
3. **Label-Conditional Masked-Field Transformers**: Include both `label` and `data_source` (`Synthetic` vs `Original`) as explicit discrete tokens. Pretrain with 25% feature masking and 50% known-label masking on outer-train (`outer-val` and `test` permanently masked), then continue per fold with BCE directly on the masked label-token logits.
4. **Cross-Fitted Pair-Lift / PMI & Source-Donor Attention**: Compute fold-safe Pointwise Mutual Information (PMI) and Unsupported-Pair base margins (`init_score`), or route synthetic rows to their top-$K$ nearest original-data donors via mask-routed cross-attention.

---

## 7. Class-Weighted GPU Multinomial Logit Stacker (Balanced Accuracy / Imbalanced Multiclass)

When stacking $M$ multiclass predictors across $C > 2$ imbalanced classes evaluated under **Balanced Accuracy** or macro metrics, standard unweighted log-loss underweights minority classes. Combine **1 Ordinal Pairwise + $C$ One-vs-Rest Pairwise Rankers** with a **GPU inverse-class-frequency weighted Multinomial Logistic stacker**:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fit_balanced_multiclass_logit_stacker(
    logits_oof: np.ndarray,
    y: np.ndarray,
    C_reg: float = 10.0,
    epochs: int = 400,
    lr: float = 0.05,
    device: str = "cuda",
) -> nn.Linear:
    """Fit an L2-regularized class-weighted multinomial logit stacker over (N, M, C) logits."""
    n_samples, n_models, n_classes = logits_oof.shape
    X = torch.tensor(logits_oof.reshape(n_samples, n_models * n_classes), dtype=torch.float32, device=device)
    t = torch.tensor(y, dtype=torch.long, device=device)

    counts = torch.bincount(t, minlength=n_classes).float().clamp(min=1.0)
    sample_w = (n_samples / (n_classes * counts))[t]

    linear = nn.Linear(n_models * n_classes, n_classes, device=device)
    opt = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=1.0 / (C_reg * n_samples))
    for _ in range(epochs):
        opt.zero_grad()
        ce = F.cross_entropy(linear(X), t, reduction="none")
        loss = (ce * sample_w).mean()
        loss.backward()
        opt.step()
    return linear
```
