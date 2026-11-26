# GLM / Linear Residual Base Margin Offset (`base_margin` / `init_score`)

A universal recipe for training gradient-boosted decision trees (LightGBM, XGBoost, CatBoost) to learn strictly the non-linear interaction residuals on top of a regularized linear/GAM baseline.

---

## 1. Why Linear Base Margins Beat Plain GBDTs

Decision trees partition feature space via axis-aligned orthogonal step functions—ideal for localized non-linear interactions, but inefficient at approximating smooth, global additive linear surfaces (continuous splines, high-dimensional one-hot/BPE/TE linear baselines).

Decoupling the model into an additive linear margin plus tree residuals lets each model class focus on its inductive strength:
$$\eta_{\text{final}}(\mathbf{x}) = \eta_{\text{GLM}}(X_{\text{linear}}) + f_{\text{GBDT}}(X_{\text{tree}})$$
where $\eta_{\text{GLM}} = \text{logit}(p_{\text{GLM}})$ for binary classification (or $\hat{y}_{\text{Ridge}}$ for regression) initializes the booster via `init_score` (LightGBM) or `base_margin` (XGBoost / CatBoost `baseline`).

---

## 2. Leak-Free Nested Architecture

> [!CAUTION]
> **Zero Data Leakage**: The base margin $\eta_{\text{GLM}}$ for outer training fold $k$ **MUST** be generated via nested inner cross-validation strictly inside outer training fold $k$. Validation and test margins for outer fold $k$ are predicted by the linear model fitted on outer training fold $k$.

```text
Outer Fold k Training Rows
 ├── Inner 5-Fold CV: Fit L2 GLM on X_linear -> inner OOF logits η_train_k
 └── Fit L2 GLM on full Outer Training Fold k:
      ├── Predict Outer Validation Fold k -> η_val_k  (eval_set margin)
      └── Predict Test Set                -> η_test_k (test margin)
```

---

## 3. End-to-End Implementation

```python
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def safe_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return logit(np.clip(p, eps, 1.0 - eps))


def generate_nested_base_margins(
    X_linear_tr, y_tr, X_linear_val, X_linear_test, n_inner_folds: int = 5, C: float = 0.3
):
    """
    Generate strictly leak-free inner-OOF training logits plus outer-fold
    validation and test logits from a regularized linear model.
    """
    inner_skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
    margin_tr = np.zeros(len(y_tr), dtype=np.float64)

    for in_tr_idx, in_val_idx in inner_skf.split(X_linear_tr, y_tr):
        glm_in = LogisticRegression(C=C, penalty="l2", solver="lbfgs", tol=1e-8, max_iter=2000)
        glm_in.fit(X_linear_tr.iloc[in_tr_idx], y_tr[in_tr_idx])
        margin_tr[in_val_idx] = safe_logit(glm_in.predict_proba(X_linear_tr.iloc[in_val_idx])[:, 1])

    glm_full = LogisticRegression(C=C, penalty="l2", solver="lbfgs", tol=1e-8, max_iter=2000)
    glm_full.fit(X_linear_tr, y_tr)
    margin_val = safe_logit(glm_full.predict_proba(X_linear_val)[:, 1])
    margin_test = safe_logit(glm_full.predict_proba(X_linear_test)[:, 1])

    return margin_tr, margin_val, margin_test


def train_residual_lightgbm(
    X_tree_tr, y_tr, margin_tr, X_tree_val, y_val, margin_val, X_tree_test, margin_test
):
    """
    Train LightGBM on linear logit base margins via init_score.
    NOTE: Booster.predict(array, raw_score=True) returns ONLY the tree residual f(X);
    you MUST add margin_val / margin_test explicitly before applying expit()!
    """
    trn_data = lgb.Dataset(X_tree_tr, label=y_tr, init_score=margin_tr)
    val_data = lgb.Dataset(X_tree_val, label=y_val, init_score=margin_val, reference=trn_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "random_state": 42,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        trn_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    # Explicitly add base margins to LightGBM raw tree residuals
    logit_val = margin_val + model.predict(X_tree_val, raw_score=True)
    logit_test = margin_test + model.predict(X_tree_test, raw_score=True)
    return expit(logit_val), expit(logit_test)


def train_residual_xgboost(
    X_tree_tr, y_tr, margin_tr, X_tree_val, y_val, margin_val, X_tree_test, margin_test
):
    """
    Train XGBoost initialized with linear logit base margins via DMatrix.
    XGBoost DMatrix automatically incorporates base_margin during model.predict(dmat).
    """
    dtrain = xgb.DMatrix(X_tree_tr, label=y_tr, base_margin=margin_tr)
    dval = xgb.DMatrix(X_tree_val, label=y_val, base_margin=margin_val)
    dtest = xgb.DMatrix(X_tree_test, base_margin=margin_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "seed": 42,
        "tree_method": "hist",
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    return model.predict(dval), model.predict(dtest)
```
