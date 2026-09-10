---
name: kaggle-tabular-playbook
description: >-
  Standard operating procedure and battle-tested recipes for competitive tabular machine learning on Kaggle. Use when training models, engineering features, performing cross-validation, or building ensembles for tabular regression or classification challenges.
---

# Kaggle Tabular Competitions Playbook

This skill provides a structured, high-performance workflow for competing in Kaggle tabular challenges (Playground Series and Featured competitions).

## Table of Contents
1. [Core Philosophy & Phased Strategy](#1-core-philosophy--phased-strategy)
2. [Phase 1: Setup & Leak-Free Cross-Validation](#2-phase-1-setup--leak-free-cross-validation)
3. [Phase 2: Exploratory Data Analysis & Synthetic Formula Discovery](#3-phase-2-exploratory-data-analysis--synthetic-formula-discovery)
4. [Phase 3: Advanced Feature Engineering Recipes](#4-phase-3-advanced-feature-engineering-recipes)
5. [Phase 4: Multi-Model Zoo Execution](#5-phase-4-multi-model-zoo-execution)
6. [Phase 5: Centered Isotonic Calibration & Ridge Stacking](#6-phase-5-centered-isotonic-calibration--ridge-stacking)
7. [References & Deep Dives](#7-references--deep-dives)

---

## 1. Core Philosophy & Phased Strategy

Winning tabular competitions requires discipline in three pillars:
- **Never Overfit the Leaderboard**: Build a local CV scheme that matches the test split exactly. Trust CV over Public LB.
- **Extract Latent Data Generators**: Modern synthetic benchmarks (e.g. Playground Series) often contain underlying generative formulas or artifact discontinuities (rounding, modulo digits, periodic cycles).
- **Stack Heterogeneous Predictions with Regularized Linear Models**: Avoid complex non-linear meta-learners. Use Centered Isotonic Regression (CIR) calibration followed by Ridge regression.

---

## 2. Phase 1: Setup & Leak-Free Cross-Validation

Always initialize a strict, reproducible out-of-fold scheme before touching feature engineering:

```python
from sklearn.model_selection import KFold, StratifiedKFold

# Standard 5-fold KFold (Regression)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# If grouping is present (e.g. user_id, hospital_id):
# from sklearn.model_selection import GroupKFold
# kf = GroupKFold(n_splits=5)
```

> [!IMPORTANT]
> Any transformation that computes statistics from the target variable (Target Encoding) or relies on distribution aggregations (Frequency Encoding on combined sets) **MUST** be computed strictly inside each training fold and applied to validation/test folds to prevent leakage.

---

## 3. Phase 2: Exploratory Data Forensics (DS & DA Playbook)

Refer to [`references/eda_data_forensics.md`](./references/eda_data_forensics.md) for the complete 7-pillar forensic methodology:

1. **Adversarial Validation (Train vs. Test Shift)**:
   Train LightGBM to distinguish Train vs. Test. If $\text{ROC-AUC} > 0.65$, isolate and prune or neutralize the drifting features.
2. **Synthetic vs. Original Data Forensics**:
   Query original datasets via `cKDTree` to measure generator perturbation distance distributions and detect frequency oversampling ratios.
3. **Bivariate Target Profiling & Step-Function Discovery**:
   Discretize continuous variables into high-resolution quantile bins and plot empirical target rates with 95% Wilson confidence intervals to uncover non-linear thresholds.
4. **Mantissa, Rounding & Benford's Law Forensics**:
   Audit decimal fractions (`frac = x - floor(x)`) and leading-digit distributions to detect synthetic rounding artifacts.
5. **Domain Equation Residual Auditing**:
   Calculate residuals $e = y_{\text{domain}} - f(X)$ (e.g. $\text{TotalCharges} - \text{tenure} \times \text{MonthlyCharges}$). Correlate residuals with the target to isolate hidden business signals.
6. **Categorical Interaction Screening**:
   Screen pairwise and three-way combinations via Chi-square contingency table divergence to isolate the highest-signal cross-features.
7. **Synthetic Formula Discovery**:
   Fit an unregularized Linear Regression or OLS to recover additive linear formula offsets across discrete categories.

---

## 4. Phase 3: Advanced Feature Engineering Recipes

Refer to detailed guides in [`references/feature_engineering.md`](./references/feature_engineering.md):

1. **Synthetic Snap Matching & Perturbation Diff**:
   Map synthetic floats to nearest original dataset values via `np.searchsorted` to recover the true manifold coordinates and measure generator noise:
   $$x_{\text{snap}} = \text{argmin}_{v \in V_{\text{orig}}} |x - v|, \quad x_{\text{snap\_diff}} = x - x_{\text{snap}}$$
2. **Radix Continuous-Categorical Split Encoding**:
   Combine continuous snap features and categorical codes into a single integer so tree models perform joint splits:
   $$\text{radix} = \lfloor x_{\text{snap}} \cdot 100 \rfloor + \text{cat\_code} \cdot 100{,}000$$
3. **cKDTree Nearest-Neighbor Ground-Truth Prior**:
   Query a spatial `cKDTree` on standardized original dataset features to attach the nearest true label as a zero-leakage feature.
4. **Modulo Digit Decomposition & Mantissa Residuals**:
   Expose rounding artifacts to tree and neural models via base-10 mantissas and fractional residuals from common denominators ($1/2, 1/4, 1/5, 1/10$).
5. **Trigonometric / Periodic Features**:
   Continuous variables with cyclic patterns benefit from harmonic embeddings ($p \in \{12, 14, 20\}$).
6. **Multi-Aggregation OOF Target Encoding**:
   Compute `mean`, `std`, and `skew` with Empirical Bayes smoothing. See [`references/oof_target_encoding.md`](./references/oof_target_encoding.md).

---

## 5. Phase 4: Multi-Model Zoo Execution

Never rely on a single model family. Train diverse model classes on the exact same folds:

1. **Tree Ensembles**:
   - **LightGBM**: Leaf-wise growth, fast, handles count-ratio and multi-scale quantile bins.
   - **XGBoost**: Depth-wise splits, supports `XGBRanker` pairwise ranking objective (`rank:pairwise`).
   - **CatBoost**: Native Ordered Target Statistics (OTS) for high-order categorical interactions.
   - **YDF (Yggdrasil Decision Forests)**: Ultra-shallow stumps (`max_depth=2`) for smooth regularization.
   - **cuML Random Forest**: GPU-accelerated bagging tree model for orthogonal variance reduction.
2. **Deep Tabular Neural Networks**:
   - **RealMLP (`pytabkit`)**: State-of-the-art tabular MLP utilizing Piecewise Linear Representations (PLR), cosine log learning rate scheduling, and SiLU/Mish activations.
   - **TabM**: Bilinear multi-component interaction architecture ($k=32$ basis components).
   - **TabICL**: Zero-shot in-context tabular foundation Transformer requiring no gradient updates.
   - **GraphSAGE GNN**: KNN graph embeddings built on GPU via cuML KNN.
   - **FT-Transformer / TabTransformer / Trompt / SNN (SELU)**: Attention and self-normalizing representations.

Save both **Out-Of-Fold (OOF)** predictions on the training set and **Test** predictions for each model:
`models/model_name_oof.npy` (or `.csv`) and `models/model_name_test.npy`.

---

## 6. Phase 5: Ensembling & Meta-Stacking Playbook

Refer to [`references/stacking_cir_ridge.md`](./references/stacking_cir_ridge.md) for full implementations.

### Track A: Continuous Regression (RMSE / MAE)
1. **Monotonic Calibration via Centered Isotonic Regression (CIR)**:
   Calibrate every model's OOF and Test predictions individually before stacking.
2. **Ridge Regression Stacker**:
   Fit an $L_2$-regularized `RidgeCV(alphas=[1.0, 10.0, 100.0])` on calibrated OOF predictions.
3. **Secondary CIR Post-Processing**:
   Fit a final CIR model on the Ridge meta-predictions against ground truth.

### Track B: Binary Classification (ROC-AUC / LogLoss)
1. **Greedy Forward Hill Climbing**:
   Filter redundant models by greedily accumulating models that improve honest OOF AUC.
2. **Fold-Wise Ordinal Rank Probability Calibration**:
   Map OOF predictions within each fold to common quantile ranks and replace with empirical rank means.
3. **Logit-Space Transform**:
   $$z = \text{clip}\left(\ln \frac{p}{1 - p}, -30.0, 30.0\right)$$
4. **$L_2$-Regularized Logistic Regression Meta-Learner**:
   Fit a strongly regularized Logistic Regression (`C=0.01`, solver `lbfgs` / `qn`) on the logit features.

---

## 7. References & Deep Dives
- [Exploratory Data Forensics (DS & DA Playbook)](./references/eda_data_forensics.md)
- [Feature Engineering Toolkit](./references/feature_engineering.md)
- [Leak-Free Multi-Agg Target Encoding](./references/oof_target_encoding.md)
- [CIR Calibration, Logit Stacking & Hill Climbing](./references/stacking_cir_ridge.md)
- [Autonomous Multi-LLM Competitive Workflow](../../Handbook/workflows/llm-agentic-kaggle-workflow.md)

