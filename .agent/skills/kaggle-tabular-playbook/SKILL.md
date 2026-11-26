---
name: kaggle-tabular-playbook
description: >-
  Standard operating procedure and battle-tested recipes for competitive tabular machine learning on Kaggle. Use when training models, engineering features, performing cross-validation, or building ensembles for tabular regression or classification challenges.
---

# Kaggle Tabular Competitions Playbook

A structured, universal workflow for competitive tabular machine learning across both **real-world Featured competitions** and **synthetic-generator benchmarks (e.g., Playground Series)**.

## Table of Contents
1. [Core Philosophy & Phased Strategy](#1-core-philosophy--phased-strategy)
2. [Phase 1: Setup & Leak-Free Cross-Validation](#2-phase-1-setup--leak-free-cross-validation)
3. [Phase 2: Exploratory Data Forensics](#3-phase-2-exploratory-data-forensics)
4. [Phase 3: Feature Engineering (Universal Core + Optional Synthetic Track)](#4-phase-3-feature-engineering-universal-core--optional-synthetic-track)
5. [Phase 4: Multi-Model Zoo Execution](#5-phase-4-multi-model-zoo-execution)
6. [Phase 5: Ensembling & Meta-Stacking Playbook](#6-phase-5-ensembling--meta-stacking-playbook)
7. [References & Deep Dives](#7-references--deep-dives)

---

## 1. Core Philosophy & Phased Strategy

Winning tabular competitions requires discipline in three pillars:
- **Match Validation to Test Mechanics**: Lock a reproducible CV split matching grouping, time order, or stratification before engineering features. Trust nested CV over noisy Public LB deltas.
- **Forensics Before Feature Engineering**: Diagnose data regime first—covariate shift, missingness structure, exact-value signal, domain physics equations, and whether a synthetic generator is present—before selecting transforms.
- **Heterogeneous Diversity + Regularized Metric-Aligned Stacking**: Combine genuinely distinct model families (GBDTs, deep tabular NNs, linear/GAM, foundation models, generative/SSL transformers) with strongly regularized linear or metric-exact meta-learners rather than deep non-linear stackers.

---

## 2. Phase 1: Setup & Leak-Free Cross-Validation

Always lock a deterministic, shared fold assignment across all team members and model families:

```python
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

# Classification: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Regression:     KFold(n_splits=5, shuffle=True, random_state=42)
# Grouped/Time:   GroupKFold(n_splits=5) or Leave-One-Group-Out / purged temporal splits
```

> [!IMPORTANT]
> Any target-dependent transform (Target Encoding, base-margin GLMs, isotonic calibration, pseudo-label teachers, or meta-stackers) **MUST** be fitted strictly inside inner training folds (nested CV) so validation rows never leak into their own features or calibration curves.

---

## 3. Phase 2: Exploratory Data Forensics

Refer to [`references/eda_data_forensics.md`](./references/eda_data_forensics.md) and [`references/advanced_forensics_contrasts_em.md`](./references/advanced_forensics_contrasts_em.md):

1. **Adversarial Validation (Train vs. Test Shift)**:
   Run cross-validated LightGBM classifying Train vs. Test. If OOF $\text{ROC-AUC} > 0.65$, inspect per-feature adversarial AUC to isolate drifting columns (or missingness-driven drift) and test drift-pruned model streams.
2. **Exact-Value & Bivariate Target Profiling**:
   Discretize continuous columns into quantile bins with 95% Wilson CIs to spot step-functions and non-monotonic anomalies (e.g., `0` meaning bypassed/N/A). Run a **split-half exact-value residual correlation test** to verify whether repeated numeric values carry distinct latent signal.
3. **Domain Equation Residual Auditing**:
   Test physical/accounting relationships $e = y_{\text{derived}} - f(X)$ and inspect whether residuals correlate strongly with the target.
4. **Missingness & Reconstruction Forensics**:
   Check whether `NaN` patterns carry predictive signal (never blindly pre-impute for GBDTs!) and audit column-to-column cross-fitted reconstruction residuals ($x_j - \hat{x}_j$).
5. **Categorical Interaction Screening**:
   Screen pairwise and 3-way categorical combinations via Chi-square divergence or mutual information to prioritize high-signal crosses.
6. **Synthetic-Generator Detection (When Applicable)**:
   Check if an original seed dataset exists, test nearest-neighbor distance distributions (`cKDTree`), decimal mantissa / Benford anomalies, exact-value duplication rate, and additive linear formula fit to determine whether synthetic-generator tricks apply. *(Skip generator-artifact features when forensics show smooth continuous real-world data!)*

---

## 4. Phase 3: Feature Engineering (Universal Core + Optional Synthetic Track)

Refer to [`references/feature_engineering.md`](./references/feature_engineering.md) and [`references/oof_target_encoding.md`](./references/oof_target_encoding.md).

### A. Universal Tabular Core (Applies to All Datasets)
1. **Multi-Aggregation OOF Target Encoding**:
   Compute `mean`, `std`, and `skew` strictly out-of-fold with consistent Empirical Bayes smoothing on both train (`fit_transform`) and test (`transform`).
2. **Scale-Adaptive Multi-Granularity TE Pyramids**:
   Encode high-signal numeric columns across coarse-to-fine scales (exact value, magnitude-adaptive rounded clusters, quantile bins) inside nested folds.
3. **Multi-Scale Binning & Combinatorial Interactions**:
   Combine quantile (`qcut`), uniform (`cut`), and domain-meaningful bins with frequency-filtered pairwise categorical crosses.
4. **External / Original Data Transfer (Ablate Both Modes on CV)**:
   - **Mode 1 — Static Zero-Leak Lookup Priors (`orig_rate__*`)**: Map target statistics from external/original data as features. Often strictly superior when original and competition label-noise processes differ.
   - **Mode 2 — Synchronous Sample-Weighted Row Appending**: Concatenate external/original training folds with tuned sample weights (`w = 0.25–1.25`), evaluating CV **strictly on unmodified host validation rows**.
   - *Rule*: Never assume one mode dominates universally—ablate both on honest CV.
5. **Multi-Resolution TE Regularization (`noise` + `feature_weights`)**:
   When feeding dense multi-scale TE banks into GBDTs, inject mild zero-mean Gaussian noise strictly on training folds and pass differential `feature_weights` (higher on raw features, lower on derived TE diffs) to curb split overfitting.

### B. Optional: Synthetic-Generator / Playground Track *(Activate Only After Forensics Confirm)*
1. **Snap Matching & Perturbation Diff**: Map synthetic floats to nearest original seed values via `np.searchsorted` ($x_{\text{snap}}$ and $x - x_{\text{snap}}$).
2. **Radix Joint Split Encoding**: Pack quantized continuous + categorical code into one integer so trees split jointly.
3. **cKDTree Original-Seed Prior**: Query standardized original seed rows to attach nearest-neighbor historical label and distance.
4. **Modulo Digit / Mantissa Residuals & Data-Driven Harmonics**: Extract base-10 digits, fractional residuals against common denominators ($1/2, 1/4, 1/5, 1/10$), and periodic `sin`/`cos` bases only at frequencies confirmed by FFT/EDA.
5. **BPE Subword Token Encodings on Numeric Strings**: Tokenize integer strings with GPT-2 BPE when forensics confirm LLM/GReaT synthesis artifacts.
6. **Fold-Safe Semi-Supervised Exact-Value EM**: Pool outer-train counts with value-blind context-teacher weighted unlabeled rows when exact numeric keys repeat across `train + test`. See [`references/advanced_forensics_contrasts_em.md`](./references/advanced_forensics_contrasts_em.md).

---

## 5. Phase 4: Multi-Model Zoo Execution

Train heterogeneous model families on the **exact same saved fold indices**:

1. **Tree Ensembles**:
   - **LightGBM** (leaf-wise, GOSS, DART), **XGBoost** (depth-wise, `XGBRanker` pairwise ranking), **CatBoost** (native Ordered Target Statistics), **YDF** (shallow stumps), and **cuML / sklearn Random Forest** (bagging diversity).
2. **Deep Tabular Neural Networks**:
   - **RealMLP (`pytabkit`)** (Piecewise Linear Representations + SiLU/Mish), **TabM** (batch-ensemble multi-head MLP), **FT-Transformer**, **GraphSAGE / KNN-GNN**, **DAE / Masked Density Transformer**, **DeepFM**, **ModernNCA**, **ExcelFormer**, **Trompt**.
3. **Tabular Foundation Models, SSL & Generative Transformers**:
   - **Full-Context Zero-Shot / Fine-Tuned TFMs**: `TabPFN`, `TabICL`, `Kumo`, `TabFM`, `Mitra` (declare high-cardinality discrete IDs as categorical; never pass cross-validated OOFs into in-context attention!).
   - **Self-Supervised Feature-Token Transformers (Universal)**: Pretrain order-free column-token transformers via Masked-Token Reconstruction + Hard-Negative Replaced-Token Detection (RTD) on `train + test` before fold fine-tuning.
   - **Generative Bayes Transformers (`Storyteller` / `AO-ARM` / `GenPure`)**: Score $P(\mathbf{x} \mid y)$ across random column permutations via `logsumexp` Bayes LLR (especially dominant on GReaT/LLM-synthesized or discrete-table regimes). See [`references/tabular_llms_generative_transformers.md`](./references/tabular_llms_generative_transformers.md).
4. **Continuous Residual Margin Boosting (`init_score` / `base_margin`)**:
   - Fit nested OOF regularized GLM/GAM/LR log-odds and initialize GBDTs from those margins so trees model strictly non-linear interaction residuals. See [`references/glm_residual_base_margin.md`](./references/glm_residual_base_margin.md).
5. **Group-Wise Subgroup Models & Continuous Soft Pseudo-Labeling**:
   - Train isolated subgroup models on high-level categorical slices for decorrelated stack residuals, and prefer **continuous fold-isolated soft teacher probabilities** (never hard `{0, 1}` thresholds) when pseudo-labeling improves CV. See [`references/soft_pseudolabel_distillation.md`](./references/soft_pseudolabel_distillation.md).

Save paired OOF and Test arrays (`models/<name>_oof.npy`, `models/<name>_test.npy`) verified against a shared fold hash.

---

## 6. Phase 5: Ensembling & Meta-Stacking Playbook

Refer to [`references/stacking_cir_ridge.md`](./references/stacking_cir_ridge.md) and [`references/logit_stacking_rank_blend.md`](./references/logit_stacking_rank_blend.md).

### Track A: Continuous Regression (RMSE / MAE)
1. **Nested OOF Centered Isotonic Regression (CIR) + Ridge Stacking**: Fit per-model CIR inside CV folds, combine via $L_2$-regularized `RidgeCV`, and validate any post-stack CIR strictly out-of-fold.
2. **MAE-Specific Post-Processing**: Test median TTA/aggregation and simple OOF affine bias calibration (`y_cal = a·y + b`).

### Track B: Binary Classification (ROC-AUC / LogLoss)
1. **Diversity-First Screening**: Audit pairwise Spearman correlation $\rho$ against the current blend; prioritize orthogonal model families (e.g., neural manifolds + GBDTs + TFMs) over near-duplicate tree twins, and avoid pre-pooling diverse models into an unweighted average before stacking (**Candidate Pooling Fallacy**).
2. **Fold Alignment & Logit-Space Stacking**:
   - Align per-fold calibration drift via fold-wise ordinal rank calibration or **label-free per-fold test-distribution logit standardization** (`scale * (z - mu_test) / sd_test`).
   - Clip probabilities safely, map to logits $z = \text{clip}(\ln(p / (1-p)), -30, 30)$, and fit $L_2$-regularized Logistic Regression (`class_weight=None` for AUC, tight solver tolerance **`tol=1e-8`**, `max_iter=2000`) inside nested CV.
3. **Metric-Exact Rank & Geometric Blenders (ROC-AUC)**:
   - **Percentile Rank Averaging**: Use `rankdata(p) / len(p)` when combining meta-learners operating on different probability scales.
   - **AUC-Direct FFT Blending**: Optimize the smooth all-pairs WMW sigmoid loss in $O(n + N \log N)$ via [`references/auc_direct_fft_blend.md`](./references/auc_direct_fft_blend.md).
   - **Riemannian $\mathbb{S}^{N-1}$ Fréchet Barycenter & Zero-Tie `lexrank`**: Avoid Euclidean tail shrinkage on hypersphere probit vectors and break ties via an independent continuous predictor (`np.lexsort`). See [`references/riemannian_hypersphere_frechet_blend.md`](./references/riemannian_hypersphere_frechet_blend.md).
4. **CV Stability & Shakeup Hedging**:
   - **CV Plateau Center-Selection**: When nested-CV weight sweeps flatten across an interval, choose the center of the plateau rather than an edge.
   - **Feature-Ablation Contrast Carriers**: Feed matched-seed `real - zero` or cross-family contrasts into regularized linear stackers when raw models saturate at high correlation.
   - **Clark (1961) Final Hedging Law**: When picking two final submissions, verify that the hedge satisfies both sufficient decorrelation ($\rho \le 0.98$) and competitive CV proximity ($\Delta < 2.0\sigma$).

### Track C: Imbalanced Multiclass & Specialized Tabular Regimes
1. **Balanced-Accuracy Mathematical Duality (Never Double-Correct!)**: Class-weighted log-loss ($w_c = 1/(K\cdot\pi_c)$) and unweighted calibrated probabilities divided by priors (`argmax(p / π_c)`) share the exact Bayes rule. Combine weighted GBDTs + prior-divided NNs, **never apply both to the same model** (`-0.04` to `-0.06` drop!), and strip base PP before meta-stacking.
2. **Ordinal + OvR Pairwise Rankers & Hierarchical Binary Cascades**: Pair ordinal/OvR `XGBRanker` and 2-stage hierarchical cascades with a **Class-Weighted GPU Multinomial Logit Stacker**.
3. **Specialized Tabular Regimes**: See [`references/specialized_tabular_regimes.md`](./references/specialized_tabular_regimes.md) for **split-half exact-value diagnostics & `float32` vocab safety**, **target-free Masked Density Transformer distributional imputation**, **factorized two-task survival modeling (`P(event)` + `E[rank(-t)|event=1]`)**, **pairwise matchup symmetry & graph ratings**, **dirty external-data 5-step pipelines & dense→sparse chaining**, **heavy-tail `k*` validation**, and **extreme-imbalance (`<0.2%`) entity-relative outlier context & leak-proof 2-stage vision-OOF → GBDT stacking**.
4. **Recommender Systems & Tabular Learning-to-Rank**: See [`references/recsys_retrieval_and_ltr.md`](./references/recsys_retrieval_and_ltr.md) for **two-stage session RecSys (`Recall@K` multi-view co-visitation/neural/GNN retrieval → leak-free multi-objective GBDT/GNN reranking)** and **tabular query/session LTR (`HitRate@K`/`NDCG`, transductive count encoding, dual-level group/sub-entity ranks, strictly causal numeric/rank history, post-FE group capping, per-group normalization)**.

---

## 7. References & Deep Dives
- [Recommender Systems, Multi-Stage Retrieval & Learning-to-Rank](./references/recsys_retrieval_and_ltr.md)
- [Specialized Tabular Regimes (Balanced Acc, Imputation, Survival, Matchups, External Data, Extreme Imbalance & `k*`)](./references/specialized_tabular_regimes.md)
- [Exploratory Data Forensics (DS & DA Playbook)](./references/eda_data_forensics.md)
- [Feature Engineering Toolkit](./references/feature_engineering.md)
- [Leak-Free Multi-Agg Target Encoding](./references/oof_target_encoding.md)
- [GLM / Linear Residual Base Margin Boosting](./references/glm_residual_base_margin.md)
- [Soft Pseudo-Label Distillation & Group-Wise Partitions](./references/soft_pseudolabel_distillation.md)
- [CIR Calibration, Logit Stacking & Hill Climbing](./references/stacking_cir_ridge.md)
- [Logit Stacking, Rank Averaging & Synchronous Fold Protocol](./references/logit_stacking_rank_blend.md)
- [Riemannian Hypersphere Fréchet Ensembling & Boundary Rules](./references/riemannian_hypersphere_frechet_blend.md)
- [AUC-Direct FFT Ensembling & Full-Context TFM Guardrails](./references/auc_direct_fft_blend.md)
- [Tabular LLMs, Generative Transformers & Foundation Models](./references/tabular_llms_generative_transformers.md)
- [Advanced Forensics, Contrast Carriers & Semi-Supervised EM](./references/advanced_forensics_contrasts_em.md)
- [Autonomous Multi-LLM Competitive Workflow](../../../Handbook/workflows/llm-agentic-kaggle-workflow.md)
