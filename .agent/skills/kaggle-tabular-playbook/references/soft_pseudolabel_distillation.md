# Soft Pseudo-Labeling Distillation & Group-Wise Partitions

A production reference for distilling out-of-fold teacher ensembles into student models and generating decorrelated error residuals via group-wise subgroup models. Distilled from **S6E9 (3rd Place Team M & M and 8th Place yuurei)**.

---

## 1. Why Hard Pseudo-Labels Fail vs. Soft Distillation

Standard pseudo-labeling often attempts to threshold test set predictions into hard binary targets $\hat{y} \in \{0, 1\}$. 

In competitive tabular machine learning, **hard pseudo-labels consistently degrade cross-validation**:
- **Threshold Noise Amplification:** Forcing uncertain probabilities (e.g. $p=0.52$) into class 1 injects severe label noise directly into the loss function.
- **Gradient Shock:** Binary cross-entropy assigns massive gradients to misclassified confident points.

### The Soft Distillation Law:
Instead of thresholding, train student models on continuous probabilities $\hat{y} = p \in (0, 1)$ with sample weighting:

$$\mathcal{L}_{\text{student}} = \sum_{i \in \text{train}} \text{BCE}(y_i, f(\mathbf{x}_i)) + w_{\text{test}} \sum_{j \in \text{test}} \text{BCE}(p_j^{\text{teacher}}, f(\mathbf{x}_j)), \quad w_{\text{test}} = 2.0$$

#### Empirical Proof (3rd Place S6E9):
- Hard pseudo-labels ($0/1$) scored **lower** than training with no pseudo-labels at all.
- Soft continuous pseudo-labels scored **higher** across all folds.
- Permuting soft pseudo-labels collapsed performance completely, proving that the continuous probabilities transferred calibrated epistemic teacher uncertainty.

---

## 2. Out-Of-Fold Teacher Isolation Architecture

> [!IMPORTANT]
> The teacher predictions for test rows used in outer fold $k$ **MUST** be generated strictly by a teacher whose training set **excluded validation fold $k$**. This maintains strict out-of-fold validity and prevents artificial variance shrinkage.

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

def train_soft_distillation_student(X_train, y_train, X_test, teacher_test_preds, test_sample_weight=2.0, cv=5):
    """
    Distills continuous soft pseudo-labels from an apex teacher into a student model.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    student_oof = np.zeros(len(y_train))
    student_test = np.zeros(len(X_test))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr_orig, y_tr_orig = X_train.iloc[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
        
        # 1. Combine training fold with soft pseudo-labeled test rows
        X_combined = pd.concat([X_tr_orig, X_test], ignore_index=True)
        y_combined = np.concatenate([y_tr_orig, teacher_test_preds])
        
        # Sample weights: 1.0 for true training labels, 2.0 for soft teacher pseudo-labels
        weights = np.concatenate([
            np.ones(len(y_tr_orig), dtype=np.float32),
            np.full(len(teacher_test_preds), test_sample_weight, dtype=np.float32)
        ])
        
        trn_data = lgb.Dataset(X_combined, label=y_combined, weight=weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)
        
        params = {
            'objective': 'binary', # In LightGBM binary handles continuous targets p in (0, 1)
            'metric': 'auc',
            'learning_rate': 0.03,
            'num_leaves': 63,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'verbose': -1,
            'random_state': 42 + fold
        }
        
        model = lgb.train(
            params,
            trn_data,
            num_boost_round=3000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        
        student_oof[val_idx] = model.predict(X_val)
        student_test += model.predict(X_test) / cv
        
    return student_oof, student_test
```

---

## 3. Group-Wise Subgroup Partitioning for Ensembling Diversity

In mature stacks, global tree models (LightGBM, XGBoost, CatBoost) share high prediction correlation ($\rho > 0.985$). 

To introduce orthogonal variance without exotic architectures, train **group-wise models**:
1. Partition the dataset by high-level categorical slices (e.g. `City Type` $\times$ `Car Type`, `Age Group`).
2. Train an isolated model strictly on each subgroup slice.
3. For validation and test, route each row to its corresponding subgroup model.

### The Subgroup Paradox:
- A single subgroup model trained on $1/10$-th of the data will have **lower standalone CV** (e.g., `0.9449` vs `0.9462` global).
- However, when added to a meta-stack, it provides **consistent positive lift (+0.000016 across all 10 folds in 8th place)** because its localized errors are decorrelated from the global models' errors.

```python
def train_groupwise_ensemble(X_train, y_train, X_test, group_col, cv=5):
    """
    Trains dedicated models per subgroup and stitches predictions back.
    """
    groups = X_train[group_col].unique()
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    
    for g in groups:
        tr_mask = (X_train[group_col] == g)
        te_mask = (X_test[group_col] == g)
        
        if tr_mask.sum() < 100:
            continue
            
        X_tr_g = X_train[tr_mask].drop(columns=[group_col])
        y_tr_g = y_train[tr_mask]
        X_te_g = X_test[te_mask].drop(columns=[group_col])
        
        # Train standard GBDT on this subgroup
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        g_oof = np.zeros(len(X_tr_g))
        g_test = np.zeros(len(X_te_g))
        
        for tr_idx, val_idx in skf.split(X_tr_g, y_tr_g):
            model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, random_state=42, verbose=-1)
            model.fit(
                X_tr_g.iloc[tr_idx], y_tr_g.iloc[tr_idx],
                eval_set=[(X_tr_g.iloc[val_idx], y_tr_g.iloc[val_idx])],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            g_oof[val_idx] = model.predict_proba(X_tr_g.iloc[val_idx])[:, 1]
            if len(X_te_g) > 0:
                g_test += model.predict_proba(X_te_g)[:, 1] / cv
                
        oof_predictions[tr_mask] = g_oof
        test_predictions[te_mask] = g_test
        
    return oof_predictions, test_predictions
```
