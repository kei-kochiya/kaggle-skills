# Centered Isotonic Calibration, Ridge/Logit Stacking & Diversity Screening

Reusable ensembling recipes combining **nested out-of-fold Centered Isotonic Regression (CIR)**, **Ridge/Logit linear meta-stacking**, **greedy forward selection**, and **correlation-aware diversity screening**.

---

## 1. Leak-Free Nested CIR Calibration + Ridge Stacking (Regression)

Heterogeneous regression models (GBDTs, tabular NNs, linear models) frequently suffer from local tail distortion and inter-model scale mismatch. Applying `CenteredIsotonicRegression` aligns each base model monotonically to empirical target quantiles while preserving the global mean.

> [!IMPORTANT]
> Always fit per-model CIR, Ridge, and post-stack CIR **strictly inside CV folds** to measure honest OOF RMSE. Fitting CIR on full `x_train_oofs` and scoring in-sample creates severe optimistic leakage!

```python
import numpy as np
from cir_model import CenteredIsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold


def _safe_cir_fit_transform(train_col, y_train, eval_col):
    cir = CenteredIsotonicRegression()
    cir.fit(train_col, y_train)
    pred = cir.transform(eval_col)
    nan_mask = np.isnan(pred)
    if np.any(nan_mask):
        pred[nan_mask] = eval_col[nan_mask]
    return cir, pred


def train_nested_cir_ridge_ensemble(
    x_train_oofs: np.ndarray,
    y_true: np.ndarray,
    x_test_preds: np.ndarray,
    folds=None,
    alphas=(0.1, 1.0, 10.0, 100.0),
    apply_post_cir: bool = True,
):
    """
    Leak-free nested CV pipeline for CIR -> RidgeCV -> optional post-CIR.
    x_train_oofs: (n_samples, n_models), x_test_preds: (n_test, n_models).
    """
    n_samples, n_models = x_train_oofs.shape
    if folds is None:
        folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(x_train_oofs))

    meta_oof = np.zeros(n_samples, dtype=np.float64)
    final_oof = np.zeros(n_samples, dtype=np.float64)
    test_fold_preds = np.zeros((len(folds), len(x_test_preds)), dtype=np.float64)

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        cal_tr = np.zeros((len(tr_idx), n_models), dtype=np.float64)
        cal_va = np.zeros((len(va_idx), n_models), dtype=np.float64)
        cal_te = np.zeros((len(x_test_preds), n_models), dtype=np.float64)

        for m in range(n_models):
            cir, cal_tr[:, m] = _safe_cir_fit_transform(
                x_train_oofs[tr_idx, m], y_true[tr_idx], x_train_oofs[tr_idx, m]
            )
            va_p = cir.transform(x_train_oofs[va_idx, m])
            cal_va[:, m] = np.where(np.isnan(va_p), x_train_oofs[va_idx, m], va_p)
            te_p = cir.transform(x_test_preds[:, m])
            cal_te[:, m] = np.where(np.isnan(te_p), x_test_preds[:, m], te_p)

        ridge = RidgeCV(alphas=alphas)
        ridge.fit(cal_tr, y_true[tr_idx])

        tr_meta = ridge.predict(cal_tr)
        va_meta = ridge.predict(cal_va)
        te_meta = ridge.predict(cal_te)
        meta_oof[va_idx] = va_meta

        if apply_post_cir:
            post_cir = CenteredIsotonicRegression()
            post_cir.fit(tr_meta, y_true[tr_idx])
            va_post = post_cir.transform(va_meta)
            te_post = post_cir.transform(te_meta)
            final_oof[va_idx] = np.where(np.isnan(va_post), va_meta, va_post)
            test_fold_preds[fold_idx] = np.where(np.isnan(te_post), te_meta, te_post)
        else:
            final_oof[va_idx] = va_meta
            test_fold_preds[fold_idx] = te_meta

    print(f"Honest Nested OOF RMSE (Pre-Post-CIR):  {root_mean_squared_error(y_true, meta_oof):.6f}")
    print(f"Honest Nested OOF RMSE (Final):         {root_mean_squared_error(y_true, final_oof):.6f}")
    return final_oof, test_fold_preds.mean(axis=0)
```

---

## 2. Logit-Space Stacking with Fold-Wise Ordinal Calibration (Binary Classification)

For ROC-AUC or LogLoss stacking across many models, align fold-to-fold probability shifts, transform to clipped log-odds, and fit an $L_2$-penalized `LogisticRegression` with strict convergence tolerance (`tol=1e-8`):

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def prob_to_logit(p, eps=1e-15, clip=30.0):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.clip(np.log(p / (1.0 - p)), -clip, clip)


def ordinal_common_fold_ranks(values, common_size):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = values.size
    order = np.argsort(values, kind="mergesort")
    pos = np.arange(n, dtype=np.int64)
    ranks_sorted = np.clip((pos * common_size) // n, 0, common_size - 1).astype(np.int32)
    ranks = np.empty(n, dtype=np.int32)
    ranks[order] = ranks_sorted
    return ranks


def fold_calibrate_oof_probs(oof, folds, eps=1e-15):
    min_fold_size = int(min(len(va_idx) for _, va_idx in folds))
    ranks = np.empty(oof.shape[0], dtype=np.int32)
    for _, va_idx in folds:
        ranks[va_idx] = ordinal_common_fold_ranks(oof[va_idx], common_size=min_fold_size)
    rank_means = pd.DataFrame({"rank": ranks, "prob": oof}).groupby("rank", sort=True)["prob"].mean()
    calibrated = rank_means.iloc[ranks].to_numpy(dtype=np.float64)
    return np.clip(calibrated, eps, 1.0 - eps)


def train_logit_stacker(oof_matrix, y_true, test_matrix, folds, C=0.01):
    N, M = oof_matrix.shape
    calibrated_oof = np.column_stack(
        [fold_calibrate_oof_probs(oof_matrix[:, m], folds) for m in range(M)]
    )
    X_train_logit = prob_to_logit(calibrated_oof)
    X_test_logit = prob_to_logit(test_matrix)

    meta_oof = np.zeros(N, dtype=np.float64)
    test_preds = np.zeros(len(test_matrix), dtype=np.float64)
    for tr_idx, va_idx in folds:
        clf = LogisticRegression(
            penalty="l2", C=C, class_weight=None, solver="lbfgs", tol=1e-8, max_iter=2000
        )
        clf.fit(X_train_logit[tr_idx], y_true[tr_idx])
        meta_oof[va_idx] = clf.predict_proba(X_train_logit[va_idx])[:, 1]
        test_preds += clf.predict_proba(X_test_logit)[:, 1] / len(folds)

    print(f"Honest Meta-Learner CV AUC: {roc_auc_score(y_true, meta_oof):.6f}")
    return meta_oof, test_preds
```

---

## 3. Greedy Forward Hill-Climbing Selection

Use greedy forward selection (ideally verified inside nested CV to guard against selection optimism!) to prune redundant models from a large pool:

```python
def greedy_forward_selection(oof_matrix, y_true, model_names, max_models=150, min_gain=1e-5):
    selected_idx = []
    current_best_score = 0.0
    N, M = oof_matrix.shape
    current_ensemble = np.zeros(N, dtype=np.float64)

    for step in range(max_models):
        best_candidate, best_score = None, current_best_score
        for m in range(M):
            if m in selected_idx:
                continue
            trial = (current_ensemble * step + oof_matrix[:, m]) / (step + 1)
            score = roc_auc_score(y_true, trial)
            if score > best_score:
                best_score, best_candidate = score, m

        if best_candidate is not None and (best_score - current_best_score) >= min_gain:
            selected_idx.append(best_candidate)
            current_best_score = best_score
            current_ensemble = (current_ensemble * step + oof_matrix[:, best_candidate]) / (step + 1)
        else:
            break

    return [model_names[i] for i in selected_idx]
```

> **Selection-Overfitting Guard**: When stacking many correlated streams, always compare greedy subset selection against keeping **all streams in strongly regularized $L_2$ Logistic/Ridge Regression** inside nested CV—on high-correlation pools, aggressive SFS/hill-climbing can overfit OOF labels by `5–10u`.

---

## 4. Diversity Screening, Inductive Bias Pairing & Stability Rules

### 4.1 Spearman Correlation Screening
```python
from scipy.stats import rankdata, spearmanr


def evaluate_candidate_diversity(candidate_preds, current_blend_preds, threshold=0.998):
    rho = spearmanr(rankdata(candidate_preds), rankdata(current_blend_preds)).statistic
    return (rho <= threshold), float(rho)
```
- Prioritize candidates with genuinely different inductive biases: pairing smooth continuous neural manifolds (RealMLP, TabM, FT-Transformer, TFMs) with axis-aligned tree ensembles consistently delivers far more ensemble lift than adding another near-twin GBDT.

### 4.2 The Candidate Pooling Fallacy
Never pre-average several diverse candidates into one unweighted sub-pool before blending into the master stack: averaging diverse models cancels their individual disagreements and drives correlation toward `> 0.999`. Feed diverse streams individually into regularized Ridge/Logistic stacking or simplex optimization.

### 4.3 CV Plateau Center-Selection
When sweeping blend weights or regularization parameters across **nested CV**, scores often flatten across a contiguous range (e.g., `10%`, `15%`, and `20%` tie within noise). Always select the **center of the CV plateau** (`15%`) rather than an edge to maximize stability under test distribution shift. Never tune micro-weights directly on Public LB.

### 4.4 Reproducibility & Artifact Integrity
- Because GPU histogram binning in GBDTs is non-deterministic, average multiple seeds (`3–10`) for anchor models.
- Verify every saved OOF/test pair with exact shape assertions, NaN checks, and a shared fold-index hash (`sha256`) before stacking.
