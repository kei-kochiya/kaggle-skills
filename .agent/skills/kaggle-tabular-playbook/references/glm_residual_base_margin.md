# GLM / Linear Residual Base Margin Offset (`base_margin` / `init_score`)

A production, battle-tested recipe for training gradient-boosted decision trees (LightGBM, XGBoost, CatBoost) to learn strictly the non-linear interaction residuals of a regularized linear model. Distilled from **S6E9 (3rd, 8th, and 12th place solutions)** and **S6E10 (goodpjw2008 top-tier pipeline)**.

---

## 1. Why Linear Base Margins Beat Plain GBDTs

Decision trees partition feature space via axis-aligned orthogonal step functions. While exceptional at isolating localized non-linear interactions, they are inherently inefficient at fitting smooth, continuous, global additive linear trends (such as continuous target encodings, monotonic income penalties, or spline baselines).

Instead of forcing trees to spend depth and leaf cuts reconstructing a linear surface, we decouple the problem into two distinct mathematical stages:

$$\eta_{\text{final}}(\mathbf{x}) = \eta_{\text{GLM}}(X_{\text{linear}}) + f_{\text{GBDT}}(X_{\text{tree}})$$

Where:
- $\eta_{\text{GLM}} = \text{logit}(p_{\text{GLM}}) = \mathbf{w}^T \mathbf{x} + b$ is the unconstrained log-odds prediction of an $L_2$-regularized linear model.
- $f_{\text{GBDT}}$ is a tree ensemble trained directly on the residual loss gradient with `base_margin` initialized to $\eta_{\text{GLM}}$.

### Empirical Benchmarks:
- **3rd Place (Team M & M):** GLM alone = `0.946326` $\to$ GLM + residual LightGBM = **`0.946502`** (+0.00018 across all 5 folds).
- **8th Place (yuurei):** Logistic alone = `0.946509` $\to$ Logistic + residual XGBoost = **`0.946617`** (+0.00011 across all 10 folds).

---

## 2. Leak-Free Nested Architecture

> [!CRITICAL]
> **Zero Data Leakage:** The base margin $\eta_{\text{GLM}}$ for training fold $k$ **MUST** be generated using nested cross-validation (e.g. an inner 5-fold split) strictly within the training portion of fold $k$. No training row must ever see its own ground truth through the linear model.

```
Outer Fold k Training Rows (80% of Data)
 ├── Inner Fold 1-4: Train GLM(C=0.3) on X_linear
 └── Inner Fold 5: Predict Out-Of-Fold Logit Margin η_inner
 ──> Concatenate inner OOFs to form full η_train_k (base_margin for GBDT)

Outer Fold k Validation Rows (20% of Data)
 ──> Predict with GLM trained on full 80% Outer Training Fold
 ──> η_val_k (eval_set base_margin for early stopping)

Test Rows (100% of Test Data)
 ──> Predict with GLM trained on full 80% Outer Training Fold
 ──> Average across outer folds to form η_test
```

---

## 3. End-to-End Implementation

```python
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

def safe_logit(p, eps=1e-7):
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return logit(p_clipped)

def generate_nested_base_margins(X_linear_train, y_train, X_linear_test, n_inner_folds=5, C=0.3):
    """
    Computes strictly leak-free base margins for training rows via inner CV,
    and out-of-fold linear predictions for test rows.
    """
    inner_skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
    inner_oof_logits = np.zeros(len(y_train), dtype=np.float64)
    test_logits_list = []
    
    for in_tr_idx, in_val_idx in inner_skf.split(X_linear_train, y_train):
        X_in_tr, y_in_tr = X_linear_train.iloc[in_tr_idx], y_train[in_tr_idx]
        X_in_val = X_linear_train.iloc[in_val_idx]
        
        # Fit regularized linear model (L2)
        glm = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
        glm.fit(X_in_tr, y_in_tr)
        
        # Inner validation logit prediction
        p_val = glm.predict_proba(X_in_val)[:, 1]
        inner_oof_logits[in_val_idx] = safe_logit(p_val)
        
        # Inner test logit prediction
        p_test = glm.predict_proba(X_linear_test)[:, 1]
        test_logits_list.append(safe_logit(p_test))
        
    avg_test_logits = np.mean(test_logits_list, axis=0)
    return inner_oof_logits, avg_test_logits


def train_residual_lightgbm(X_tree_tr, y_tr, margin_tr, X_tree_val, y_val, margin_val, X_tree_test, margin_test):
    """
    Trains LightGBM initialized with linear logit base margins via init_score.
    """
    # Create LightGBM datasets with init_score
    trn_data = lgb.Dataset(X_tree_tr, label=y_tr, init_score=margin_tr)
    val_data = lgb.Dataset(X_tree_val, label=y_val, init_score=margin_val, reference=trn_data)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.03,
        'num_leaves': 63,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.train(
        params,
        trn_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    
    # Predict residual and add base margin
    val_pred = safe_logit(model.predict(X_tree_val, raw_score=True)) # raw score adds to init_score
    # In LightGBM raw_score=True returns margin + f(X)
    raw_val = model.predict(X_tree_val, raw_score=True)
    prob_val = 1.0 / (1.0 + np.exp(-raw_val))
    
    raw_test = model.predict(X_tree_test, raw_score=True)
    prob_test = 1.0 / (1.0 + np.exp(-raw_test))
    
    return prob_val, prob_test


def train_residual_xgboost(X_tree_tr, y_tr, margin_tr, X_tree_val, y_val, margin_val, X_tree_test, margin_test):
    """
    Trains XGBoost initialized with linear logit base margins via DMatrix base_margin.
    """
    dtrain = xgb.DMatrix(X_tree_tr, label=y_tr, base_margin=margin_tr)
    dval = xgb.DMatrix(X_tree_val, label=y_val, base_margin=margin_val)
    dtest = xgb.DMatrix(X_tree_test, base_margin=margin_test)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.03,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'seed': 42,
        'tree_method': 'hist'
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    
    prob_val = model.predict(dval)
    prob_test = model.predict(dtest)
    
    return prob_val, prob_test
```
