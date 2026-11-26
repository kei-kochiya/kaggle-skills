# Logit Stacking, Rank Averaging, Synchronous Fold Augmentation & Hedging

Universal patterns for stacking correlated probability pools, bridging differing calibration scales, safely injecting external/original datasets, auditing covariate drift without in-sample leakage, and hedging two final submissions under Clark (1961).

---

## 1. Logit Stacking via $L_2$-Regularized Logistic Regression

Transforming probabilities $p \in (0, 1)$ into clipped log-odds $z = \text{clip}(\log(p / (1 - p)), -30, 30)$ linearizes extreme tails for regularized linear meta-learners.

### Key Rules for Rank-Based Metrics (ROC-AUC)
1. **`class_weight=None`**: Class balancing distorts log-odds intercepts and hurts rank resolution.
2. **Safe Clipping**: Clip probabilities to $[10^{-13}, 1 - 10^{-13}]$ and logits to $[-30, +30]$.
3. **Strict Solver Tolerance (`tol=1e-8`)**: Default `tol=1e-4` terminates prematurely on flat log-loss surfaces when stacking 50+ correlated logit columns.

```python
from typing import Optional
import numpy as np

try:
    from cuml.linear_model import LogisticRegression as GPULogisticRegression
    HAS_CUML = True
except ImportError:
    from sklearn.linear_model import LogisticRegression as CPULogisticRegression
    HAS_CUML = False


class LogitStacker:
    """L2-penalized Logistic Regression meta-learner in clipped logit space (cuML or sklearn)."""

    def __init__(self, C: float = 1.0, clip_val: float = 30.0, eps: float = 1e-13):
        self.C = C
        self.clip_val = clip_val
        self.eps = eps
        if HAS_CUML:
            self.model = GPULogisticRegression(C=self.C, penalty="l2", max_iter=2000, tol=1e-8)
        else:
            self.model = CPULogisticRegression(
                C=self.C,
                penalty="l2",
                class_weight=None,
                solver="lbfgs",
                tol=1e-8,
                max_iter=2000,
                random_state=42,
            )

    def _to_logit(self, p: np.ndarray) -> np.ndarray:
        p_safe = np.clip(p, self.eps, 1.0 - self.eps)
        return np.clip(np.log(p_safe / (1.0 - p_safe)), -self.clip_val, self.clip_val)

    def fit(self, oof_matrix: np.ndarray, y_true: np.ndarray):
        assert oof_matrix.ndim == 2, "oof_matrix must be 2D (n_samples, n_models)"
        Z = np.column_stack([self._to_logit(oof_matrix[:, i]) for i in range(oof_matrix.shape[1])])
        self.model.fit(Z, y_true)
        return self

    def predict_proba(self, test_matrix: np.ndarray) -> np.ndarray:
        assert test_matrix.ndim == 2, "test_matrix must be 2D (n_test, n_models)"
        Z_test = np.column_stack([self._to_logit(test_matrix[:, i]) for i in range(test_matrix.shape[1])])
        return self.model.predict_proba(Z_test)[:, 1]
```

---

## 2. Cross-Ensembler Percentile Rank Averaging

Different ensemblers output radically different probability spreads (e.g., calibrated logit stackers near base rate vs. tree/hill-climb averages compressed near `0.5`). Converting predictions to empirical uniform percentiles $r_i = \text{rank}(p_i) / N$ ensures balanced weighting under rank-based metrics:

```python
from scipy.stats import rankdata


def rank_average(*predictions: np.ndarray, weights: Optional[list[float]] = None) -> np.ndarray:
    """Weighted percentile rank average across multiple prediction arrays."""
    assert len(predictions) > 0, "Provide at least one prediction array"
    n_samples = len(predictions[0])

    if weights is None:
        weights = [1.0 / len(predictions)] * len(predictions)
    else:
        assert len(weights) == len(predictions), "weights length must match predictions length"
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

    blended = np.zeros(n_samples, dtype=np.float64)
    for p, w in zip(predictions, weights):
        assert len(p) == n_samples, "All prediction vectors must have identical length"
        blended += w * (rankdata(p) / n_samples)
    return blended
```

---

## 3. Synchronous Multi-Dataset Fold Concatenation Protocol

When augmenting competition training data with an external or original reference dataset, **always ablate** weighted row appending against zero-leak static lookup priors (`attach_external_lookup_prior` in [`feature_engineering.md`](./feature_engineering.md)) on host-only CV—row appending helps when label distributions align cleanly and can hurt when synthetic label noise diverges.

When row appending improves CV, split synchronously and evaluate **exclusively on unmodified host validation rows**:

```python
from typing import Generator, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def synchronous_stratified_folds(
    df_host: pd.DataFrame,
    target_host: pd.Series,
    df_external: pd.DataFrame,
    target_ext: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
    ext_sample_weight: float = 0.75,
) -> Generator[Tuple[int, pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame, pd.Series, np.ndarray], None, None]:
    """
    Leak-free synchronous CV combining host and external training folds while
    validating strictly on host validation splits.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    host_splits = list(skf.split(df_host, target_host))
    ext_splits = list(skf.split(df_external, target_ext))

    for fold, ((tr_idx, val_idx), (ex_tr_idx, _)) in enumerate(zip(host_splits, ext_splits), 1):
        X_tr_host, y_tr_host = df_host.iloc[tr_idx], target_host.iloc[tr_idx]
        X_val_host, y_val_host = df_host.iloc[val_idx], target_host.iloc[val_idx]
        X_tr_ext, y_tr_ext = df_external.iloc[ex_tr_idx], target_ext.iloc[ex_tr_idx]

        X_train_comb = pd.concat([X_tr_host, X_tr_ext], axis=0).reset_index(drop=True)
        y_train_comb = pd.concat([y_tr_host, y_tr_ext], axis=0).reset_index(drop=True)
        sample_weights = np.concatenate(
            [np.ones(len(X_tr_host), dtype="float32"), np.full(len(X_tr_ext), ext_sample_weight, dtype="float32")]
        )
        yield fold, X_train_comb, y_train_comb, sample_weights, X_val_host, y_val_host, val_idx
```

---

## 4. Out-Of-Fold Adversarial Feature Drift Audit & Dual-Stream Branching

Never score adversarial feature drift in-sample (decision trees overfit high-cardinality columns and report false drift). Always compute **out-of-fold** univariate adversarial AUC via `StratifiedKFold`:

```python
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score


def audit_feature_drift(
    train_df: pd.DataFrame, test_df: pd.DataFrame, candidate_cols: list[str], n_splits: int = 3
) -> pd.Series:
    """
    Compute out-of-fold univariate adversarial AUC per column (Train=0 vs Test=1).
    OOF AUC ~ 0.50: no shift; OOF AUC > 0.65: substantial covariate shift.
    """
    adv_df = pd.concat([train_df[candidate_cols], test_df[candidate_cols]], axis=0).reset_index(drop=True)
    adv_target = np.concatenate([np.zeros(len(train_df), dtype=int), np.ones(len(test_df), dtype=int)])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    drift_report = {}
    for col in candidate_cols:
        if adv_df[col].dtype == "object" or str(adv_df[col].dtype) == "category":
            X_col = pd.factorize(adv_df[col])[0].reshape(-1, 1)
        else:
            X_col = adv_df[[col]].to_numpy()

        oof = np.zeros(len(adv_target), dtype=np.float64)
        for tr_idx, va_idx in skf.split(X_col, adv_target):
            clf = LGBMClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1
            )
            clf.fit(X_col[tr_idx], adv_target[tr_idx])
            oof[va_idx] = clf.predict_proba(X_col[va_idx])[:, 1]
        drift_report[col] = roc_auc_score(adv_target, oof)

    return pd.Series(drift_report).sort_values(ascending=False)
```

**Dual-Stream Mitigation**: When a column shows high OOF adversarial drift ($\text{AUC} > 0.65$), train two parallel model streams—**Stream A (Full)** and **Stream B (Drifting Feature Dropped)**—and let nested CV stacking assign optimal weights.

---

## 5. Shake-Up Defense: The Clark (1961) Final Hedging Law

When selecting two final submissions, picking two models correlated at $\rho > 0.99$ provides near-zero protection against private shakeup. Under Clark's (1961) exact formula for $\mathbb{E}[\max(A, B)] - \mu_A$ with $A \sim \mathcal{N}(\mu_A, \sigma^2)$, $B \sim \mathcal{N}(\mu_B, \sigma^2)$, $\Delta = \mu_A - \mu_B \ge 0$, and $\theta = \sigma\sqrt{2(1 - \rho)}$:
$$\mathbb{E}[\max(A, B)] - \mu_A = \theta\,\phi\!\left(\frac{\Delta}{\theta}\right) + \Delta\,\Phi\!\left(\frac{\Delta}{\theta}\right) - \Delta$$
A valid hedge should satisfy both **$\rho \le 0.98$** (genuine decorrelation) and **$\Delta < 2.0\sigma$** (competitive CV proximity):

```python
from scipy.stats import norm, spearmanr


def clark_expected_max_gain(score_ceiling: float, score_hedge: float, rho: float, sigma: float) -> float:
    """Return expected lift (E[max(A, B)] - E[A]) in units of standard error sigma."""
    delta = score_ceiling - score_hedge
    if delta < 0:
        raise ValueError("score_ceiling must be >= score_hedge")
    theta = sigma * np.sqrt(2.0 * max(0.0, 1.0 - rho))
    if theta < 1e-12:
        return 0.0
    alpha = delta / theta
    gain = theta * norm.pdf(alpha) + delta * norm.cdf(alpha) - delta
    return float(gain / sigma)


def audit_submission_pair(sub_a_path: str, sub_b_path: str, score_a: float, score_b: float, sigma: float):
    """Audit whether submission B is a statistically meaningful hedge against submission A."""
    df_a, df_b = pd.read_csv(sub_a_path), pd.read_csv(sub_b_path)
    target_col = df_a.columns[1]
    rho = float(spearmanr(df_a[target_col], df_b[target_col]).statistic)
    delta_se = (score_a - score_b) / sigma
    gain_se = clark_expected_max_gain(score_a, score_b, rho, sigma)

    status = (
        "INERT (rho > 0.99)"
        if rho > 0.990
        else ("INERT (delta > 2.0 sigma)" if delta_se > 2.0 else "VALID HEDGE")
    )
    return {"rho": rho, "delta_sigma": delta_se, "expected_gain_sigma": gain_se, "status": status}
```
