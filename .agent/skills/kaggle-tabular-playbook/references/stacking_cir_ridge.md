# Centered Isotonic Calibration & Ridge Stacking

A reproducible guide for implementing the 1st-place ensembling strategy combining **Centered Isotonic Regression (CIR)** calibration with **Ridge Regression** meta-stacking.

---

## 1. Why Isotonic Calibration Before Stacking?

Standard model ensembles (GBDTs, Neural Nets, Linear Models) output predictions that often exhibit:
1. **Local probability/metric distortion** (e.g., underpredicting at tails, overpredicting in dense regions).
2. **Inter-model scale mismatch** (some models produce wider dynamic ranges than others).

By fitting an isotonic regression model onto each base model's predictions, we map predictions monotonically to match empirical target quantiles. `CenteredIsotonicRegression` preserves the overall center of mass, preventing mean drift.

---

## 2. End-to-End Pipeline Implementation

```python
import numpy as np
import pandas as pd
from cir_model import CenteredIsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error

def train_cir_ridge_ensemble(x_train_oofs, y_true, x_test_preds, alphas=[0.1, 1.0, 10.0, 100.0]):
    """
    x_train_oofs: np.ndarray of shape (n_samples, n_models)
    y_true: np.ndarray of shape (n_samples,)
    x_test_preds: np.ndarray of shape (n_test_samples, n_models)
    
    Returns:
        calibrated_oof_preds: np.ndarray
        calibrated_test_preds: np.ndarray
        ridge_model: fitted RidgeCV estimator
    """
    n_models = x_train_oofs.shape[1]
    calibrated_train = np.zeros_like(x_train_oofs)
    calibrated_test = np.zeros_like(x_test_preds)

    print(f"Applying Centered Isotonic Regression to {n_models} models...")
    for i in range(n_models):
        cir = CenteredIsotonicRegression()
        cir.fit(x_train_oofs[:, i], y_true)
        
        # Train transform
        pred_train = cir.transform(x_train_oofs[:, i])
        nan_mask = np.isnan(pred_train)
        if np.any(nan_mask):
            pred_train[nan_mask] = x_train_oofs[:, i][nan_mask]
        calibrated_train[:, i] = pred_train
        
        # Test transform
        pred_test = cir.transform(x_test_preds[:, i])
        nan_mask = np.isnan(pred_test)
        if np.any(nan_mask):
            pred_test[nan_mask] = x_test_preds[:, i][nan_mask]
        calibrated_test[:, i] = pred_test

    print("Fitting RidgeCV meta-regressor...")
    ridge = RidgeCV(alphas=alphas)
    ridge.fit(calibrated_train, y_true)
    print(f"Optimal Ridge alpha: {ridge.alpha_}")

    meta_oof = ridge.predict(calibrated_train)
    meta_test = ridge.predict(calibrated_test)

    pre_cir_rmse = root_mean_squared_error(y_true, meta_oof)
    print(f"Pre-PostProcessing OOF RMSE: {pre_cir_rmse:.6f}")

    # Post-stacking CIR
    print("Applying final Centered Isotonic Calibration...")
    post_cir = CenteredIsotonicRegression()
    post_cir.fit(meta_oof, y_true)
    
    final_oof = post_cir.transform(meta_oof)
    final_test = post_cir.transform(meta_test)

    final_rmse = root_mean_squared_error(y_true, final_oof)
    print(f"Final Calibrated Ensemble OOF RMSE: {final_rmse:.6f}")

    return final_oof, final_test, ridge
```

---

## 3. Logit-Space Stacking with Fold-Wise Ordinal Rank Calibration (Binary Classification)

For binary classification problems evaluated on **ROC-AUC** or **LogLoss**, ensembling tens or hundreds of model probability outputs requires converting to log-odds (logit space) and applying strong $L_2$ regularization.

Fold-wise probability alignment ensures model probabilities do not suffer from fold-to-fold calibration shifts:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def prob_to_logit(p, eps=1e-15, clip=30.0):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.clip(np.log(p / (1.0 - p)), -clip, clip)

def ordinal_common_fold_ranks(values, common_size):
    """Maps values into common quantile rank bins preserving monotonic order."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = values.size
    order = np.argsort(values, kind="mergesort")
    pos = np.arange(n, dtype=np.int64)
    ranks_sorted = np.clip((pos * common_size) // n, 0, common_size - 1).astype(np.int32)
    ranks = np.empty(n, dtype=np.int32)
    ranks[order] = ranks_sorted
    return ranks

def fold_calibrate_oof_probs(oof, folds, eps=1e-15):
    """Calibrates Out-of-Fold probabilities across cross-validation splits."""
    min_fold_size = int(min(len(va_idx) for _, va_idx in folds))
    ranks = np.empty(oof.shape[0], dtype=np.int32)
    for _, va_idx in folds:
        ranks[va_idx] = ordinal_common_fold_ranks(oof[va_idx], common_size=min_fold_size)
    rank_means = pd.DataFrame({"rank": ranks, "prob": oof}).groupby("rank", sort=True)["prob"].mean()
    calibrated = rank_means.iloc[ranks].to_numpy(dtype=np.float64)
    return np.clip(calibrated, eps, 1.0 - eps)

def train_logit_stacker(oof_matrix, y_true, test_matrix, folds, C=0.01):
    """
    Fits an L2-penalized Logistic Regression on calibrated logit features.
    Accepts oof_matrix: (N_samples, N_models) and test_matrix: (N_test, N_models).
    """
    N, M = oof_matrix.shape
    calibrated_oof = np.zeros_like(oof_matrix)
    
    # 1. Fold-wise calibration
    for m in range(M):
        calibrated_oof[:, m] = fold_calibrate_oof_probs(oof_matrix[:, m], folds)
        
    # 2. Map to logit space
    X_train_logit = prob_to_logit(calibrated_oof)
    X_test_logit  = prob_to_logit(test_matrix)
    
    # 3. Honest CV Evaluation
    meta_oof = np.zeros(N, dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(folds):
        clf = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=2000)
        clf.fit(X_train_logit[tr_idx], y_true[tr_idx])
        meta_oof[va_idx] = clf.predict_proba(X_train_logit[va_idx])[:, 1]
        
    print(f"Honest Meta-Learner CV AUC: {roc_auc_score(y_true, meta_oof):.6f}")
    
    # 4. Fit on full data for test predictions
    final_clf = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=2000)
    final_clf.fit(X_train_logit, y_true)
    final_test = final_clf.predict_proba(X_test_logit)[:, 1]
    
    return meta_oof, final_test, final_clf.coef_.reshape(-1)
```

---

## 4. Greedy Forward Hill-Climbing Model Selection

When training 50–200 models, greedy forward selection identifies the optimal non-redundant subset:

```python
def greedy_forward_selection(oof_matrix, y_true, model_names, max_models=150, min_gain=1e-5):
    """
    Greedily selects models that maximize ensemble ROC-AUC.
    """
    selected_idx = []
    current_best_score = 0.0
    N, M = oof_matrix.shape
    current_ensemble = np.zeros(N, dtype=np.float64)
    
    for step in range(max_models):
        best_candidate = None
        best_score = current_best_score
        
        for m in range(M):
            if m in selected_idx:
                continue
            trial_ensemble = (current_ensemble * step + oof_matrix[:, m]) / (step + 1)
            score = roc_auc_score(y_true, trial_ensemble)
            
            if score > best_score:
                best_score = score
                best_candidate = m
                
        if best_candidate is not None and (best_score - current_best_score) >= min_gain:
            selected_idx.append(best_candidate)
            current_best_score = best_score
            current_ensemble = (current_ensemble * step + oof_matrix[:, best_candidate]) / (step + 1)
            print(f"Step {step+1:3d}: Added {model_names[best_candidate]:25s} | AUC: {best_score:.6f}")
        else:
            print("Converged: no candidate improves ensemble score beyond threshold.")
            break
            
    return [model_names[i] for i in selected_idx]
```

---

## 5. The Spearman Rank Correlation Selection Rule ($\rho \le 0.998$)

In late-stage competitions or when constructing mega-ensembles, testing every candidate model on cross-validation or live submissions wastes time and submissions. AUC and rank-based metrics depend strictly on relative ordering. 

### The Mathematical Screening Rule
Compute the Spearman rank correlation $\rho$ of each candidate model $M_k$ against the current best blend $E$:
$$\rho(M_k, E) = 1 - \frac{6 \sum_{i=1}^N (R_{k, i} - R_{E, i})^2}{N(N^2 - 1)}$$

```python
from scipy.stats import spearmanr, rankdata

def evaluate_candidate_diversity(candidate_preds, current_blend_preds, threshold=0.998):
    """
    Screens whether a candidate model provides sufficient structural diversity
    to move a saturated ensemble score.
    """
    r_cand = rankdata(candidate_preds)
    r_blend = rankdata(current_blend_preds)
    
    rho = spearmanr(r_cand, r_blend).statistic
    is_diverse = (rho <= threshold)
    
    print(f"Spearman Correlation vs. Current Blend: {rho:.5f}")
    if is_diverse:
        print(f"  -> ACCEPT: Diverse candidate ({rho:.5f} <= {threshold}). Eligible for weight sweep.")
    else:
        print(f"  -> REJECT: Redundant model ({rho:.5f} > {threshold}). Disagreements are insufficient to move leaderboard.")
    return is_diverse, rho
```

**Rule of Thumb**:
- $\rho > 0.998$: **Collinear / Redundant**. Adding this model at any weight will not shift the score beyond rounding error.
- $\rho \in [0.995, 0.998]$: **Diverse Candidate**. Worth sweeping weights between $5\%$ and $20\%$.
- $\rho < 0.995$: **Highly Orthogonal Anchor**. Prime candidate for substantial weight ($15\% - 30\%$).

---

## 6. Inductive Bias Pairing: Neural Networks vs. Decision Trees

A frequent failure mode in competitive tabular ML is assembling 10–15 tree-based models (LightGBM, CatBoost, XGBoost, HistGradientBoosting). Even with different hyperparameters and feature subsets, tree models share the same fundamental inductive bias: **orthogonal piecewise-constant hyperplanes with step-cut discontinuities**.

### The Neural Manifold Lift
Deep Tabular Neural Networks (e.g. PyTorch RealMLP, TabNet, ResNet, TabM) learn **continuous, smooth, curved decision boundaries**.
- A tabular neural network often scores **lower** as a standalone model (e.g., $0.94595$ vs. $0.94640$ for GBDT).
- However, because its errors occur in completely different regions of the feature space, its Spearman correlation against a saturated GBDT blend is typically **$\rho \approx 0.9960 - 0.9975$**.
- Adding $10\% - 20\%$ of a continuous neural manifold model into a saturated tree ensemble consistently provides the decisive $+0.00002$ to $+0.00005$ breakthrough.

```python
# Blending continuous neural manifold with discrete GBDT tree consensus
# r_trees: percent ranks of top GBDT ensemble
# r_realmlp: percent ranks of PyTorch RealMLP neural network
final_rank = 0.85 * r_trees + 0.15 * r_realmlp
```

---

## 7. The Candidate Pooling Fallacy & Plateau Center-Selection

Two critical empirical ensembling rules discovered in live Kaggle competitions:

### A. The "Candidate Pooling Fallacy"
When you have 4–5 diverse models that each correlate $\rho \approx 0.9965 - 0.9975$ with the current blend, a common instinct is to average them together into a "diversity sub-ensemble" and blend that pool into the main model.
- **Why this fails**: The disagreements of diverse models are primarily *with each other*. Averaging them together cancels their unique individual signals, causing the pool's correlation against the master blend to jump to **$\rho > 0.9990$** (higher than any single member!).
- **Actionable Rule**: Never pre-pool diverse models into an unweighted average. Evaluate and inject each diverse model individually, or optimize their joint weights simultaneously via constrained optimization / Ridge regression.

### B. The Plateau Center-Selection Rule
When sweeping candidate weights (e.g. testing $0\%, 10\%, 15\%, 20\%, 25\%$), public leaderboard scores often exhibit a flat plateau:
- $0\%$ RealMLP: $0.94643$
- $10\%$ RealMLP: $0.94644$ (Tie)
- **$15\%$ RealMLP**: **$0.94644$ (Center of Plateau — PICK THIS)**
- $20\%$ RealMLP: $0.94644$ (Tie)
- $25\%$ RealMLP: $0.94643$ (Decline)

> [!TIP]
> **Pick the middle of a plateau, not the edge**: If $10\%$, $15\%$, and $20\%$ all tie on the public leaderboard, reporting $10\%$ or $20\%$ is an artifact of discrete sampling. The center ($15\%$) sits furthest from the performance cliffs on both sides, providing maximum safety margin against distribution shifts and shakeout on the private leaderboard.

---

## 8. Dynamic Public Source Auditing & Timestamp Tracking

When participating in competitions where public notebooks contribute components:
1. **GBDT Histogram Non-Determinism**: Because GPU and multi-threaded CPU histogram binning in GBDT libraries are not bit-deterministic, upstream authors re-running notebooks create slight shifts in predictions ($10^{-5}$ scale).
2. **Timestamped Blend Measurements**: A blend weight measured against public sources is only valid for the specific version measured. If upstream sources improve, a candidate weight that yielded $+0.00000$ yesterday can yield $+0.00002$ today against the updated baseline. Always re-download upstream sources when conducting fine sweeps.

