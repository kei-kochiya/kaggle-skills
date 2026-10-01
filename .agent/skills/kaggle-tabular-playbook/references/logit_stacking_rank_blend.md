# Logit Stacking, Rank Averaging & Synchronous Fold Protocol

A battle-tested production reference for ensembling high-correlation model pools, bridging disparate probability calibration scales, and safely injecting external/original datasets without cross-fold data leakage. Distilled from **Playground Series S6E3** and **S6E5** (1st & 2nd place solutions).

---

## 1. Logit Stacking via $L_2$-Regularized Logistic Regression

### Why Logit Space?
When stacking probabilities $p \in (0, 1)$ from diverse tree models and neural networks, raw probabilities compress near 0 and 1, creating artificial non-linearities for linear meta-learners.

Transforming probabilities into unbounded log-odds linearizes the boundary:

$$z = \text{logit}(p) = \log\left(\frac{p}{1 - p}\right)$$

### Key Rules for Rank-Based Metrics (ROC-AUC)
1. **`class_weight=None`**: Applying class balancing (`class_weight='balanced'`) distorts predicted probabilities and shifts the intercept. For AUC, only monotonic rank matters. Balancing hurts rank resolution.
2. **Clipping**: Unbounded log-odds can explode to $\pm \infty$ on confident predictions ($p=0.0$ or $p=1.0$). Always clip input probabilities to $[10^{-13}, 1 - 10^{-13}]$ and output logits to $[-30, +30]$.
3. **Regularization ($C=1.0$)**: Light $L_2$ penalty prevents high-magnitude weights from overfitting correlated base models while letting complementary models decorrelate.

### Production Implementation (GPU & CPU)

```python
import numpy as np
import pandas as pd
from typing import Optional

try:
    # High-speed GPU implementation
    from cuml.linear_model import LogisticRegression as GPULogisticRegression
    HAS_CUML = True
except ImportError:
    from sklearn.linear_model import LogisticRegression as CPULogisticRegression
    HAS_CUML = False


class LogitStacker:
    """
    Fits an L2-penalized Logistic Regression meta-learner in logit space.
    Compatible with cuML (GPU) and scikit-learn (CPU).
    """
    def __init__(self, C: float = 1.0, clip_val: float = 30.0, eps: float = 1e-13):
        self.C = C
        self.clip_val = clip_val
        self.eps = eps
        
        if HAS_CUML:
            self.model = GPULogisticRegression(C=self.C, penalty='l2', max_iter=2000, tol=1e-8)
        else:
            # Critical: Default tol=1e-4 terminates prematurely on flat loss surfaces
            # when stacking 50+ correlated model prediction columns. tol=1e-8 is mandatory.
            self.model = CPULogisticRegression(
                C=self.C,
                penalty='l2',
                class_weight=None,
                solver='lbfgs',
                tol=1e-8,
                max_iter=2000,
                random_state=42
            )

    def _to_logit(self, p: np.ndarray) -> np.ndarray:
        p_safe = np.clip(p, self.eps, 1.0 - self.eps)
        z = np.log(p_safe / (1.0 - p_safe))
        return np.clip(z, -self.clip_val, self.clip_val)

    def fit(self, oof_matrix: np.ndarray, y_true: np.ndarray):
        """
        oof_matrix: np.ndarray of shape (n_samples, n_models) containing probability predictions
        y_true: ground truth binary labels
        """
        assert oof_matrix.ndim == 2, "oof_matrix must be 2D (samples, models)"
        Z = np.column_stack([self._to_logit(oof_matrix[:, i]) for i in range(oof_matrix.shape[1])])
        self.model.fit(Z, y_true)
        return self

    def predict_proba(self, test_matrix: np.ndarray) -> np.ndarray:
        """
        test_matrix: np.ndarray of shape (n_test, n_models) containing probability predictions
        Returns: 1D np.ndarray of positive class probabilities
        """
        assert test_matrix.ndim == 2, "test_matrix must be 2D (samples, models)"
        Z_test = np.column_stack([self._to_logit(test_matrix[:, i]) for i in range(test_matrix.shape[1])])
        return self.model.predict_proba(Z_test)[:, 1]
```

---

## 2. Cross-Ensembler Rank Averaging

### The Probability Spread Mismatch Problem
Different ensembling methods produce radically different probability distributions:
- **Logit Stacker**: Calibrated near the natural empirical prior (e.g. $p \approx 0.20$).
- **Hill-Climbing / Tree Blends**: Often uncalibrated, with predictions compressed around the center ($0.40 - 0.60$).

If you compute a naive arithmetic mean:

$$\text{Blend} = 0.5 \cdot p_{\text{LogitStack}} + 0.5 \cdot p_{\text{HillClimb}}$$

The model with larger variance dominates the ordering, degrading the AUC.

### The Rank Transformation Solution
Converting each model's predictions to empirical uniform percentiles ($[0, 1]$) guarantees equal weighting:

$$r_i = \frac{\text{rank}(p_i)}{N}$$

```python
from scipy.stats import rankdata
import numpy as np

def rank_average(*predictions: np.ndarray, weights: Optional[list[float]] = None) -> np.ndarray:
    """
    Computes a weighted percentile rank average across multiple prediction vectors.
    Strictly preserves AUC ranking without distortion from differing probability calibrations.
    
    Example:
        final_sub = rank_average(pred_logit_stack, pred_autogluon, weights=[0.5, 0.5])
    """
    assert len(predictions) > 0, "Must provide at least one prediction array"
    n_samples = len(predictions[0])
    
    if weights is None:
        weights = [1.0 / len(predictions)] * len(predictions)
    else:
        assert len(weights) == len(predictions), "weights length must match predictions length"
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        
    blended_rank = np.zeros(n_samples, dtype='float64')
    for p, w in zip(predictions, weights):
        assert len(p) == n_samples, "All prediction vectors must have identical length"
        # rankdata assigns 1 to N
        norm_rank = rankdata(p) / n_samples
        blended_rank += w * norm_rank
        
    return blended_rank
```

---

## 3. Synchronous Multi-Dataset Fold Concatenation Protocol

When augmenting synthetic competition data with an external/original dataset (e.g. `f1_strategy_dataset_v4.csv` in S6E5 or `telco-customer-churn` in S6E3), naive concatenation prior to splitting causes severe cross-fold leakage.

### The Ingestion Protocol
1. **Parallel Split**: Partition synthetic train into $K$ Stratified folds. Partition original data into $K$ Stratified folds using the identical random state.
2. **Inner Concatenation**: For each fold $k$, merge `synthetic_train[k]` and `original_train[k]`.
3. **Pure Synthetic Validation**: Evaluate **strictly** on `synthetic_val[k]`. Never evaluate CV on the original dataset!
4. **Sample Weighting**: Modulate the influence of original data using sample weights ($0.50 - 1.00$).

```python
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from typing import Generator, Tuple

def synchronous_stratified_folds(
    df_synthetic: pd.DataFrame,
    target_syn: pd.Series,
    df_original: pd.DataFrame,
    target_orig: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
    orig_sample_weight: float = 0.75
) -> Generator[Tuple[int, pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame, pd.Series, np.ndarray], None, None]:
    """
    Generates leak-free synchronous cross-validation folds combining synthetic and original data.
    
    Yields:
        (fold, X_train_comb, y_train_comb, sample_weights, X_val_syn, y_val_syn, val_idx)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    syn_splits = list(skf.split(df_synthetic, target_syn))
    orig_splits = list(skf.split(df_original, target_orig))
    
    for fold, ((tr_idx, val_idx), (or_tr_idx, or_val_idx)) in enumerate(zip(syn_splits, orig_splits), 1):
        # Synthetic slice
        X_tr_syn, y_tr_syn = df_synthetic.iloc[tr_idx], target_syn.iloc[tr_idx]
        X_val_syn, y_val_syn = df_synthetic.iloc[val_idx], target_syn.iloc[val_idx]
        
        # Aligned original slice
        X_tr_orig, y_tr_orig = df_original.iloc[or_tr_idx], target_orig.iloc[or_tr_idx]
        
        # Merge training sets
        X_train_comb = pd.concat([X_tr_syn, X_tr_orig], axis=0).reset_index(drop=True)
        y_train_comb = pd.concat([y_tr_syn, y_tr_orig], axis=0).reset_index(drop=True)
        
        # Sample weights
        weights_syn = np.ones(len(X_tr_syn), dtype='float32')
        weights_orig = np.full(len(X_tr_orig), orig_sample_weight, dtype='float32')
        sample_weights = np.concatenate([weights_syn, weights_orig])
        
        yield fold, X_train_comb, y_train_comb, sample_weights, X_val_syn, y_val_syn, val_idx
```

---

## 4. Adversarial Feature Drift Detection & Pruning Protocol

In synthetic competitions, individual categorical or numerical features may suffer severe generator distortion, causing models to overfit spurious correlations.

### Step 1: Run Per-Feature Adversarial AUC
Label training rows as $0$ and test rows as $1$. Train a 1-feature LightGBM model for every column:

```python
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

def audit_feature_drift(train_df: pd.DataFrame, test_df: pd.DataFrame, candidate_cols: list[str]):
    """
    Computes univariate adversarial AUC for each candidate column.
    AUC ~ 0.50 indicates identical distributions.
    AUC > 0.65 indicates severe covariate shift.
    """
    adv_df = pd.concat([train_df[candidate_cols], test_df[candidate_cols]], axis=0).reset_index(drop=True)
    adv_target = np.concatenate([np.zeros(len(train_df)), np.ones(len(test_df))])
    
    drift_report = {}
    for col in candidate_cols:
        clf = LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1)
        # Handle string categoricals
        X_col = pd.factorize(adv_df[col])[0].reshape(-1, 1) if adv_df[col].dtype == 'object' else adv_df[[col]].values
        
        clf.fit(X_col, adv_target)
        preds = clf.predict_proba(X_col)[:, 1]
        auc = roc_auc_score(adv_target, preds)
        drift_report[col] = auc
        
    return pd.Series(drift_report).sort_values(ascending=False)
```

### Step 2: Parallel Dual-Stream Branching
If a feature shows high adversarial drift ($\text{AUC} > 0.65$, e.g. `Driver` in S6E5):
1. **Stream A (Full Feature Set)**: Captures within-dataset driver patterns.
2. **Stream B (Drift Feature Dropped)**: Eliminates domain shift, providing robust, un-corrupted predictions.
3. **Ensemble Injection**: Both streams are fed into the Logit Stacker, allowing the meta-learner to assign optimal orthogonal weights.

---

## 5. Shake-Up Defense: The Clark (1961) Final Hedging Decision Rule

When selecting two final submissions on Kaggle, teams often pick their two highest public leaderboard scores. However, if these submissions originate from the same pipeline or share high Spearman correlation ($\rho > 0.99$), **the second submission provides zero statistical hedging value against private shake-up**.

Under Clark's (1961) closed-form expectation for the maximum of two correlated normal variables $A \sim \mathcal{N}(\mu_A, \sigma^2)$ and $B \sim \mathcal{N}(\mu_B, \sigma^2)$ with correlation $\rho$:

$$\mathbb{E}[\max(A, B)] - \mu_A = \sigma \sqrt{2(1 - \rho)} \cdot \phi\left(\frac{\Delta}{\sigma \sqrt{2(1 - \rho)}}\right) + \Delta \cdot \Phi\left(\frac{\Delta}{\sigma \sqrt{2(1 - \rho)}}\right) - \Delta$$

Where $\Delta = \mu_A - \mu_B \ge 0$, and $\phi, \Phi$ are the standard Gaussian PDF and CDF.

### The Two Laws of Hedging:
1. **The Correlation Ceiling ($\rho \le 0.98$):** If $\rho(A, B) > 0.99$, the expected gain $\mathbb{E}[\max] - \mu_A$ rounds to zero (less than $0.05 \sigma$). It is literally the same submission wearing a hat.
2. **The Proximity Ceiling ($\Delta < 2.0 \sigma$):** A candidate decorrelated at $\rho = 0.90$ that sits more than $2.0$ private standard errors below your ceiling will almost never exceed $A$ in any realization.

### Complete Python Hedging Calculator:

```python
import numpy as np
from scipy.stats import norm, spearmanr

def clark_expected_max_gain(score_ceiling: float, score_hedge: float, rho: float, sigma: float) -> float:
    """
    Computes E[max(A, B)] - E[A] under Clark (1961) formula.
    
    score_ceiling: Public or CV score of your best candidate (A)
    score_hedge: Public or CV score of your secondary candidate (B)
    rho: Spearman correlation between submission A and submission B predictions
    sigma: Estimated private leaderboard standard error (e.g. sqrt(AUC*(1-AUC)/N_test))
    
    Returns:
        Expected score lift above candidate A in standard error units (z-score)
    """
    delta = score_ceiling - score_hedge
    if delta < 0:
        raise ValueError("score_ceiling must be >= score_hedge")
        
    theta = sigma * np.sqrt(2.0 * (1.0 - rho))
    if theta < 1e-12:
        return 0.0
        
    alpha = delta / theta
    gain = theta * norm.pdf(alpha) + delta * norm.cdf(alpha) - delta
    return gain / sigma # Return in units of sigma


def audit_submission_pair(sub_a_path: str, sub_b_path: str, score_a: float, score_b: float, sigma: float):
    """
    Audits whether submission B is a valid statistical hedge against submission A.
    """
    df_a = pd.read_csv(sub_a_path)
    df_b = pd.read_csv(sub_b_path)
    target_col = df_a.columns[1]
    
    rho = spearmanr(df_a[target_col], df_b[target_col]).statistic
    delta_se = (score_a - score_b) / sigma
    gain_se = clark_expected_max_gain(score_a, score_b, rho, sigma)
    
    print("=" * 60)
    print("CLARK (1961) FINAL HEDGING SELECTION AUDIT")
    print("=" * 60)
    print(f"Candidate A (Ceiling): Score = {score_a:.5f}")
    print(f"Candidate B (Hedge)  : Score = {score_b:.5f} (Δ = {delta_se:.2f} σ)")
    print(f"Spearman Rank Correlation (ρ): {rho:.5f}")
    print(f"Expected Hedging Gain: +{gain_se:.4f} σ")
    
    if rho > 0.990:
        print("[DECISION: INERT HEDGE] ρ > 0.99. Ticket B provides NO protection against noise.")
    elif delta_se > 2.0:
        print("[DECISION: INERT HEDGE] Distance > 2.0 σ. Ticket B is too weak to win.")
    else:
        print("[DECISION: VALID HEDGE] Ticket B provides genuine statistical hedging!")
```
