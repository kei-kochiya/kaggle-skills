# Soft Pseudo-Labeling Distillation & Group-Wise Partitions

Universal recipes for distilling fold-isolated teacher ensembles into student models using continuous soft probabilities and generating decorrelated residuals via group-wise subgroup models.

---

## 1. Why Continuous Soft Pseudo-Labels Beat Hard Thresholding

Thresholding test predictions into hard `{0, 1}` pseudo-labels amplifies boundary label noise and injects steep miscalibrated gradients into binary cross-entropy. Training students on **continuous soft teacher probabilities** $p_j^{\text{teacher}} \in (0, 1)$ transfers calibrated epistemic uncertainty:
$$\mathcal{L}_{\text{student}} = \sum_{i \in \text{train}} \text{BCE}(y_i, f(\mathbf{x}_i)) + w_{\text{test}} \sum_{j \in \text{test}} \text{BCE}(p_j^{\text{teacher}}, f(\mathbf{x}_j))$$
where $w_{\text{test}}$ (typically tuned in `[0.5, 2.0]` on honest CV) controls distillation strength.

---

## 2. Fold-Isolated Teacher Distillation Implementation

> [!IMPORTANT]
> **Strict Fold Isolation**: The soft test labels used inside outer fold $k$ **MUST** come from a teacher trained strictly on outer training fold $k$ (excluding validation fold $k$!). Using full-dataset averaged test predictions leaks validation information across folds and produces overly optimistic CV.

```python
import lightgbm as lgb
import numpy as np
import pandas as pd


def train_soft_distillation_student(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    fold_teacher_test_preds: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    test_sample_weight: float = 1.0,
):
    """
    Distill fold-isolated soft teacher test predictions into a student GBDT.
    fold_teacher_test_preds: shape (len(folds), len(X_test)), where row k was
    predicted by a teacher that NEVER saw validation fold k.
    """
    assert fold_teacher_test_preds.shape == (len(folds), len(X_test))
    student_oof = np.zeros(len(y_train), dtype=np.float64)
    student_test = np.zeros(len(X_test), dtype=np.float64)

    for fold, (tr_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
        soft_y_test = fold_teacher_test_preds[fold]

        X_comb = pd.concat([X_tr, X_test], ignore_index=True)
        y_comb = np.concatenate([y_tr, soft_y_test])
        weights = np.concatenate(
            [np.ones(len(y_tr), dtype=np.float32), np.full(len(soft_y_test), test_sample_weight, dtype=np.float32)]
        )

        trn_data = lgb.Dataset(X_comb, label=y_comb, weight=weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)

        params = {
            "objective": "binary",  # LightGBM binary loss accepts continuous soft targets in (0, 1)
            "metric": "auc",
            "learning_rate": 0.03,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "verbose": -1,
            "random_state": 42 + fold,
        }
        model = lgb.train(
            params,
            trn_data,
            num_boost_round=3000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        student_oof[val_idx] = model.predict(X_val)
        student_test += model.predict(X_test) / len(folds)

    return student_oof, student_test
```

---

## 3. Group-Wise Subgroup Partitioning for Ensemble Diversity

Global GBDT models often correlate at $\rho > 0.985$. Training dedicated subgroup models on coarse categorical slices (e.g., high-volume segment or interaction strata) produces weaker standalone CV (fewer rows per slice) yet **decorrelated error residuals** that can add consistent lift inside a meta-stacker.

Always reuse the **exact global outer fold split** and initialize outputs from a global fallback predictor so rare or unseen slices never receive raw `0.0` defaults:

```python
def train_groupwise_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    group_col: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    global_oof_fallback: np.ndarray,
    global_test_fallback: np.ndarray,
    min_group_rows: int = 500,
):
    """
    Train isolated models per sufficiently populated subgroup using shared outer folds,
    falling back smoothly to a global model on rare/unseen groups.
    """
    oof_preds = global_oof_fallback.copy()
    test_preds = global_test_fallback.copy()

    for g in X_train[group_col].dropna().unique():
        tr_mask_all = (X_train[group_col] == g).to_numpy()
        te_mask = (X_test[group_col] == g).to_numpy()
        if tr_mask_all.sum() < min_group_rows:
            continue

        g_test_accum = np.zeros(te_mask.sum(), dtype=np.float64)
        for fold, (tr_idx, val_idx) in enumerate(folds):
            g_tr_idx = tr_idx[tr_mask_all[tr_idx]]
            g_val_idx = val_idx[tr_mask_all[val_idx]]
            if len(g_tr_idx) < 50 or len(g_val_idx) == 0 or len(np.unique(y_train[g_tr_idx])) < 2:
                continue

            model = lgb.LGBMClassifier(
                n_estimators=1500, learning_rate=0.03, random_state=42 + fold, verbose=-1
            )
            model.fit(
                X_train.iloc[g_tr_idx].drop(columns=[group_col]),
                y_train[g_tr_idx],
                eval_set=[(X_train.iloc[g_val_idx].drop(columns=[group_col]), y_train[g_val_idx])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            oof_preds[g_val_idx] = model.predict_proba(X_train.iloc[g_val_idx].drop(columns=[group_col]))[:, 1]
            if te_mask.any():
                g_test_accum += model.predict_proba(X_test.loc[te_mask].drop(columns=[group_col]))[:, 1] / len(folds)

        if te_mask.any():
            test_preds[te_mask] = g_test_accum

    return oof_preds, test_preds
```
