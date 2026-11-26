# Specialized Tabular Regimes — Universal Recipes

Transferable, cross-competition tabular patterns distilled from gold-medal solutions across imbalanced multiclass (`Balanced Accuracy`), heavy-missingness tabular benchmarks, right-censored survival analysis, pairwise matchup forecasting, dirty external scientific datasets, and heavy-tail grouped regression.

---

## 1. Balanced Accuracy Mathematical Duality & Guardrails

Under multiclass **Balanced Accuracy** ($\frac{1}{K}\sum_{c=1}^K \text{Recall}_c$), minimizing **class-weighted log-loss** with inverse-frequency weights $w_c = \frac{1}{K \cdot \pi_c}$ shares the exact Bayes-optimal decision rule with dividing unweighted calibrated probabilities by class priors:
$$\hat{y}(x) = \arg\max_c \frac{\eta_c(x)}{\pi_c}$$

- **Two Equivalent, Clean Routes (NEVER Mix Both!)**:
  - **Route A (Weighted Training)**: Train with $w_c = 1/(K \cdot \pi_c)$ and decode with raw `argmax(p)`.
  - **Route B (Unweighted Training + Prior Division)**: Train unweighted well-calibrated models (especially tabular NNs/RealMLP/FTT, which often train more stably unweighted), blend on LogLoss (e.g., SLSQP per-class simplex weights), and apply zero-DOF `argmax(p / π_c)` at decode time.
  - **The Double-Correction Trap**: Training with `class_weight='balanced'` **AND** dividing by $\pi_c$ (or applying aggressive minority multipliers) double-corrects class priors and collapses Balanced Accuracy by `-0.04` to `-0.06`.
- **Strip Post-Processing Before Meta-Stacking**: Always save raw probability/logit OOFs from base models; never feed thresholded or multiplier-adjusted outputs into a Level-2 stacker.
- **Hierarchical Binary / Chain-Cascade Decomposition**: When two extreme classes have near-zero mutual confusion (e.g., `Low ↔ High` or `A ↔ C`), decompose $K$-class prediction into sequential binary tasks — Stage 1: `Class 0 vs Rest`, Stage 2: `Class 1 vs Class 2` trained strictly on non-Class-0 rows — and ensemble alongside flat multiclass models.
- **Tiny-Minority Shakeup Defense**: When the rarest class is `< 5%`, every misclassified minority row exerts $5\times$–$10\times$ leverage on Balanced Accuracy. Audit predicted minority class count drift against training priors (and prune unstable weak families like noisy GNNs/SVMs before final stacking).

---

## 2. Exact-Value Forensics, `float32` Bugs & Distributional Imputation

1. **Split-Half Non-Monotonic Exact-Value Diagnostic**:
   - Before treating a continuous/decimal column purely as smooth numeric, split `train` randomly in half, fit a smooth monotone curve (or spline), and compute the **split-half correlation $r$ of per-exact-value residuals**.
   - High split-half residual correlation ($r > 0.5$) proves genuine exact-value signal: unlock it via fold-local exact-value Target Encoding, high GBDT bin resolution (`max_bin=512..768`), and PLR/exact-lookup tabular NNs. Never screen exact-value TE on tiny subsamples (creates false negatives).
2. **Silent `float32` Vocabulary Lookup Corruption**:
   - If you build exact-value dictionaries/embeddings from `float64` keys and later cast feature arrays to `float32` for GPU training, exact float keys stop matching and silently map hundreds of thousands of rows to `UNK`! Always cast keys consistently (or format canonical fixed-precision strings/integers) *before* constructing lookup tables.
3. **Never Pre-Impute `NaN`s for GBDTs — Use Distributional Imputation as Extra Features**:
   - Replacing `NaN`s with medians/means before GBDTs destroys native missingness splits (can crash ROC-AUC by `-0.04+`).
   - Instead, when missingness is heavy (`5–20%+`), train a **target-free Masked Density Transformer** (predicting each masked cell's full distribution from observed columns across `train + test`, trained with random `p=0.25` multi-pass cell masking) and append **distributional repair posteriors + pair-lift / surprise residuals** alongside raw `NaN`-preserving columns!

---

## 3. Factorized Two-Task Censored Survival Modeling

For right-censored survival data `(event ∈ {0, 1}, time)` evaluated by **Concordance Index (C-index)** or **Group-Stratified C-index (`mean - std`)**:

1. **Factorize Event Occurrence and Conditional Time**:
   - Single-head Cox/KM transformations conflate censoring probability with event-time ordering. Decompose into two cleanly supervised tasks:
     - **Head 1 (`P(event = 1)`)**: Binary classifier on `event`.
     - **Head 2 (`E[rank(-t) | event = 1]`)**: Regressor trained **strictly on uncensored (`event == 1`) rows** predicting rank-normalized/probit/logit negative survival time.
     - Merge via expected-rank formula (or power-calibrated blend) on OOF.
2. **Differentiable Pairwise-Ranking NNs + Per-Fold Location Alignment**:
   - Pairwise ranking losses (`tanh`/`sigmoid` smooth C-index) are **shift-invariant** — each CV fold learns an arbitrary additive offset! Always align per-fold mean/median (`z - fold_median`) *before* concatenating OOFs or averaging test predictions.
3. **Variance Control Under `mean - std` Equity Metrics**:
   - High fold variance directly inflates the `-std` penalty across subgroups. Use **repeated multi-seed CV** (`4-fold × 20–50 seeds` or `10–20` folds without early stopping) and **never inject artificial post-hoc subgroup noise/shifts** (consistently overfits Public LB and fails Private LB).

---

## 4. Pairwise Matchup Symmetry, Asymmetric Flipping & Graph Ratings

For head-to-head outcome prediction (`Brier Score` / `LogLoss` / `RMSE` across seasons, tournaments, or unseen games):

- **Strict Antisymmetric vs. Asymmetric-Advantage Flip Augmentation + TTA**:
  - For symmetric neutral-site matchups, construct antisymmetric differences (`f_A - f_B`) and mirror `(B, A, 1 - y)` (or `-y` on `[-1, 1]`).
  - When structural **side/first-mover/home-field bias** exists ($u(A, B) \neq -u(B, A)$), swap `(A, B)` and negate `y` **while also inverting all side-advantage & balance covariates** (`side_adv → 1 - side_adv`, `balance → -balance` / inverted) and test an `is_augmented` flag, **separate original vs. flipped models**, **Stage-1 Flipped → Stage-2 Original**, or **`MultiRMSE([y, -y])`**, averaging `(pred_orig - pred_flip) / 2` at inference (always score CV strictly on unaugmented original rows!).
- **Strict Unseen-Group / Near-Duplicate Clustered CV**: Use Leave-One-Season-Out (LOSO) or `StratifiedGroupKFold`; when group IDs contain near-duplicate variants (e.g., similar rulesets/entities), cluster group text/embeddings (`TF-IDF + KMeans`) *before* assigning folds.
- **Global Graph Latent Ratings vs. Embedded Fast-Proxy Simulation**:
  - For seasonal schedules, compute global graph ratings (ridge/GLM team quality, Colley/SRS, margin-of-victory Elo, pace-normalized efficiency), then aggressively prune to the top `3–10` orthogonal differences (`LogisticRegression` + shallow `XGB(max_depth=2..4)` crush deep trees on small-N seasons).
  - When an exact domain simulator/interpreter is permitted at test time, run a **fast proxy simulation probe** per unseen group + train with **multi-run stochastic feature noise augmentation** (sampling across repeated simulation runs per epoch) to prevent overfitting single-run noise.
- **Calibration & Mean-Shrinkage Control**: Regress continuous score differentials when available → OOF Isotonic/Logistic calibration (gentle `[0.02, 0.98]` clipping under Brier); on bounded `[-1, 1]` RMSE targets where GBDTs shrink toward `0`, pair **`LightGBM DART` + `TabM` + `CatBoost`** with **nested OOF Isotonic / CIR** rather than unvalidated LB multipliers.

---

## 5. Dirty External-Data 5-Step Pipeline & Multi-Target Chaining

When supplementing small/sparse competition tables with noisy external databases (e.g., molecular, scientific, or crowdsourced benchmarks):

1. **Host-Only Validation**: Evaluate CV strictly on unmodified competition rows; never allow external rows into validation folds.
2. **Exact & Near-Duplicate Similarity Purge**: Remove any external row with exact key match or structural similarity `> 0.99` (e.g., Tanimoto/cosine) to validation/test folds to block silent leakage.
3. **Monotonic External Label Rescaling**: Fit an **Isotonic Regression** mapping external measurements onto the host dataset's scale/units (corrects systematic measurement/protocol offset).
4. **OOF MAE Error Filtering + Per-Dataset Weights**: Score external rows with a host-trained OOF ensemble, drop gross residual outliers, and down-weight external rows (`w = 0.25–0.75`) relative to host rows (`1.0`).
5. **Dense-to-Sparse Target Chaining & MAE Post-Processing**: When some targets are dense (`90%+` observed) and others are sparse (`< 15%`), predict dense targets first and feed leak-free OOF dense predictions as features into sparse-target models. Under **MAE**, use **median TTA** (e.g., randomized SMILES/order TTA) and test simple OOF **post-hoc linear bias calibration (`y_cal = a·y + b`)**.

---

## 6. Heavy-Tail `k*` Validation, Joint Real+Synth & Convex Gating

- **Leave-Largest-Contribution-Out (`k*`) Validation**: Under pooled RMSE where a few extreme groups contribute `30–50%` of total squared error, standard mean CV hides catastrophic tail failures. Report both overall CV and **CV after removing the top-$k$ error-contributing groups** (and vice-versa, inspect top-error groups explicitly) so single outliers don't hijack model selection.
- **Joint Real + Physics-Consistent Synthetic Training > 2-Stage Pretrain→FT**: Training jointly on mixed real + synthetic batches prevents catastrophic forgetting of synthetic physics priors during fine-tuning.
- **Replacing `LayerNorm` with `BatchNorm2dFp32(track_running_stats=False)` in 2D Alignment Backbones**: When adapting pretrained `ConvNeXt` backbones to 2D alignment lattices/heatmaps with batch-specific intensity drift, swap `LayerNorm` for `BatchNorm2d` in FP32 with `track_running_stats=False` while copying pretrained affine `(weight, bias)` parameters.
- **SoftMax Convex Gating + De-Shrinkage Gain**: Combine heterogeneous sequence/probabilistic models via a learned SoftMax convex gate (`sum w_m = 1`), and verify whether mild variance de-shrinkage (`1.02–1.03×` around fold mean) improves pooled RMSE on CV.
