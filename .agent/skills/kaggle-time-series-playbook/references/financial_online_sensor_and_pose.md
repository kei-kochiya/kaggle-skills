# Streaming Financial Online Learning, Wearable Sensors & Multi-Agent Pose — Universal Recipes

Transferable temporal modeling patterns distilled from gold-medal solutions across streaming financial forecasting (`Zero-Mean R²`, `Penalized Sharpe`, `Cross-Sectional Spearman Sharpe`), multimodal wearable IMU/ToF gesture recognition, and heterogeneous multi-agent pose action segmentation.

---

## 1. Streaming Financial Forecasting & Online Adaptation

1. **Cross-Sectional + Temporal Neural Factorizations**:
   - Row-wise GBDTs/MLPs hit a hard plateau on multi-asset panel series. First add **cross-sectional `(date, time)` market means & z-scores**, then upgrade to **Cross-Asset Self-Attention + Per-Symbol Temporal GRU (`TransformerLayer2d`)** with **strictly causal masking** across time (never allow bidirectional temporal attention — leaks rolling features into fake `R² ~ 0.30` CV that collapses live!).
   - **Multi-Horizon Auxiliary Targets**: Jointly predicting correlated return horizons (e.g., short/medium/long forward SMAs) or recursively forecasting underlying asset states before computing algebraic spreads consistently stabilizes noisy single-horizon targets.
2. **Daily Online Learning Protocol (When API Reveals Lagged Ground Truth)**:
   - Without online updates, financial models decay severely across multi-month live horizons. Validate decay explicitly using a **200-day-gap holdout fold** alongside expanding/purged CV (`CombinatorialPurgedGroupKFold`).
   - **Online Optimizer Hygiene**: **Never carry offline Adam momentum buffers into online deployment** and avoid stale multi-day replay buffers. Either **reinitialize `AdamW` fresh every day** (`lr ~ 3e-4, grad_clip = 1.0`) OR use a **fresh online-only Adam with short-memory betas (`betas=(0.8, 0.95)`, 2–5 day memory)**.
   - **Always update during unscored warmup replay (`is_scored == False`)** so weights arrive warm at the live scoring window; update online strictly on the primary target if auxiliary targets are noisier.
   - **Online GBDTs**: Native `init_model` warm-starting frequently fails under drift — replace old tree blocks by refitting **rolling subsampled time blocks** (e.g., 5 × 120-day blocks at `40–50%` row subsampling) or sliding `~800`-day windows, and test blending matured API `label_lags` directly.
3. **Low-SNR Financial Risk Engineering & Anti-Overfitting Discipline**:
   - Daily financial return OOS $R^2$ is $\sim 0.001$–$0.013$, and short `90–180`-day live Sharpe metrics carry extreme sampling variance (`36%` of random walks exceed Sharpe `1.0`).
   - **Risk & Allocation Geometry > Point MSE**:
     - Use **inverse-volatility weighting ($w_i \propto \sigma_i^{-1}$)** rather than inverting noisy sample covariance matrices.
     - Anchor position sizing directly to metric volatility cliffs (`clip(s / (cap · σ_hat), 0, max_w)` with realized+implied vol blending) and prefer conservative unlevered `{0, 1}` or smooth `tanh` sizing over aggressive `2×` leverage.
     - Under cross-sectional rank Sharpe (`mean(ρ_t) / std(ρ_t)`), compute the **covariance-shrunk Kelly rank ordering ($\Sigma_{\text{rank}}^{-1} \mu_{\text{rank}}$)** or optimize **composite `0.2·MSE + 0.8·(1 - Spearman)`** losses with short lookback windows (`≤ 4` days).
   - **Strict Financial Leakage Hygiene**: Never use `bfill` imputation (leaks future regimes; strictly causal `ffill` or drop early NaN eras), handle multi-exchange holidays & mid-horizon ticker delistings (`assert` zero `NaN`s in outputs), and prefer fixed-weight small ensembles over meta-stackers on low-SNR folds.

---

## 2. Wearable IMU & Multimodal Sensor Sequence Modeling

1. **Sensor Physics, Chirality & Inverted-Wearing Forensics**:
   - Audit **handedness/chirality** and **180° upside-down device wearing** immediately (Public test can be 100% canonical while Private contains mirrored/inverted subjects!). Canonicalize left→right coordinates (negate lateral axes, swap symmetric sensor pairs, mirror spatial grids) and build a **per-sequence orientation/flip detector** (do not assume an entire subject wears the device consistently!).
   - **Quaternion Continuity & Gravity Removal**: Raw quaternions jump discontinuously across `-q ≡ q`. Replace Euler angles with **6D rotation representation** (or dot-product continuity `dot(q_t, q_{t-1}) >= 0` / symmetric `f(q) + f(-q)`), rotate raw acceleration into world frame to remove gravity (`9.81 m/s²`), and train **dedicated `acc`-only fallback branches** for sequences with missing quaternions.
   - Under time-stretch augmentation, stretch **only linear acceleration by `1 / stretch_factor²`** and recompute gravity from Slerp-stretched quaternions to preserve Newtonian kinematics.
2. **Multi-Branch Sensor Fusion & 2D-Vision Matrix Diversity**:
   - Use **channel-independent / grouped multi-scale 1D-CNN (`k=[3,5,7]`) + SE + BiGRU/BiLSTM/Mamba** (avoid early dense cross-sensor mixing on small sequence counts) + **2D/3D spatial CNNs** for grid sensors + **phase-aware temporal attention pooling**.
   - **BatchNorm-First (No External Scaler)**: Skip per-sequence normalization (which destroys absolute gravity/temperature baselines) and let a leading `BatchNorm1d` normalize raw physical units.
   - **High-Diversity 2D Vision Stream**: Order sensor channels by physical proximity, resize `(T, F)` to 2D images, and train `ConvNeXt` / `EfficientNet` / `MaxViT` — consistently adds orthogonal error diversity over 1D models.
   - **Missing-Modality Routing**: Train separate full-sensor and core-IMU models (or modality-dropout gated networks), **blend IMU-only logits into Full-sensor predictions** for extra robustness, and skip modality hallucination / cross-fold OOF KD (leaks subject IDs!).
3. **Small-Subject CV & Online Group Assignment**:
   - Under `StratifiedGroupKFold(subject)` with `< 100` subjects, per-fold F1 early stopping overfits `±0.01` noise — train **fixed epochs with cosine decay + EMA (`0.99–0.9995`)** across multiple CV seeds, drop subject demographics (high memorization risk), and skip classical Butterworth/SavGol/Wavelet filters & spectrograms on sharp transient gestures.
   - When evaluation streams multiple sequences per subject/group under known cardinality or protocol structure, exploit cross-sequence context via a **Group-Level Transformer** or **Online Hungarian Assignment (`linear_sum_assignment`)**.

---

## 3. Multi-Agent Pose Action Segmentation Across Heterogeneous Sources

1. **Self-Supervised Motion Pretraining on Unlabeled Pose**:
   - When unlabeled pose trajectories vastly outnumber labeled clips (e.g., `90%` unlabeled), **self-supervised pretraining predicting future speed + direction** massively outperforms naive pseudo-labeling or semi-supervised hard thresholding.
2. **Heterogeneous Multi-Source Canonicalization & 3-State Partial Masking**:
   - Resample heterogeneous frame rates to one uniform FPS (passing `fps_scale` as a feature), normalize coordinates by physical scale (`pix_per_cm` or body length), map heterogeneous bodypart sets onto a **canonical 5–8 keypoint master skeleton**, and project into an **egocentric / agent-centric coordinate frame** (zero raw world coordinates!).
   - **Selective Interpolation Only**: Blindly interpolating missing keypoints across all sources creates teleportation spikes during tracker ID-switches; interpolate only sparse-tracking sources with explicit `is_interpolated` flags.
   - **3-State Partial Supervision Mask**: When each video annotates only a known whitelist of actions, strictly mask unlisted actions from loss (`1 / 0 / MASK` or `logits[invalid] = -inf` before softmax) and exclude train-only labs/datasets from validation scoring.
3. **Graph-Temporal Encoders, Cross-Family NN+GBDT Synergy & Fast Decoding**:
   - Use **GNN (`TransformerConv` / `ST-GCN`) + Egocentric Squeezeformer / RoPE Transformer + Pairwise RelationNetwork** (or `MacroSoftF1` annealing) instead of concatenating agents into flat conv channels.
   - **Strong Cross-Family Lift**: Blending temporal neural networks (`TCN`/`Squeezeformer`/`CNN-Transformer`) with per-(domain, action) GBDTs delivers huge additive gains (`+0.02+` F1 over either family alone; note horizontal flip helps 1D NNs but can hurt GBDTs, and reverse-time TTA hurts directed actions!).
   - **Sliding-Window Boundary Cropping & Margin Decoding**: Discard noisy edge frames (`~32` frames each side) during sliding-window overlap averaging, test **offset-rotated strided inference** (`stride=7` + linear interpolation for `7×` speedup + smoothing), and resolve simultaneous multi-action triggers via **per-(domain, action) margin tie-breaking (`argmax(p - thr)`, `p / thr`, z-score, or learned paired `p1 - p2` tournament)**.

---

## 4. Multi-Horizon Retail Demand Forecasting (`MAPE` & Structural Growth)

When forecasting daily/weekly multi-entity demand across a fixed future horizon (`30–90` days) under **MAPE** with calendar effects and non-stationary entity growth:

1. **Horizon-Matched Multi-Window Temporal CV & Simple Anchor**:
   - Never rely on a small random-split Public LB. Validate across **multiple non-overlapping horizon-matched rolling windows** (e.g., `5 × 61-day` blocks) plus **year-prior same-season splits** to align major moving holidays, anchored against a zero-leakage **previous-horizon weekday-average baseline**.
2. **Calendar Forensics & Structural-Shift Trimming**:
   - Audit missing dates (typically closed-entity days / regional Sundays) and patch missing regional holidays across the **merged `train + test` calendar** *before* computing `days_until_next_holiday`, `days_since_last_holiday`, and contiguous closed-day run lengths.
   - Because standard GBDTs cannot extrapolate unbounded trends, **trim stale pre-growth history** when entities undergo structural regime jumps (e.g., new launches or acquisitions) and test **`delta_days_from_start`** + **LLM text embeddings of holiday descriptions** alongside `1.3×` holiday sample weights.
3. **MAPE Objective Hygiene & Direct-Horizon Anti-Patterns**:
   - Ablate **`log(y)` target + `MAE (L1)` loss** (often far more stable against Public→Private shakeup under small-denominator variance) directly against **native `MAPE` loss** on multi-window horizon CV.
   - **Avoid**: Raw short-horizon target lags on long direct forecast horizons without test actuals (consistently degrade GBDTs), predicting train-only operational/weather columns as noisy proxies, and splitting out individual outlier entities into isolated models instead of one shared global calendar GBDT (or test **lags-as-extra-rows** multi-offset augmentation).

---

## 5. Irregular Multi-Band Heteroscedastic Time-Series Classification (Small-N `F1`)

When classifying rare events (`< 5%` positives, tiny positive counts `100–300`) from asynchronously sampled multi-channel curves `(t_i, val_i, err_i, band_i)` under discrete **Binary F1**:

1. **2D Gaussian Process Interpolation & Parametric Template $\chi^2$ Residuals**:
   - Fit a **2D separable GP kernel (`k_time * k_channel`, e.g., Matérn-3/2)** incorporating heteroscedastic measurement errors `err_i²` directly into the noise diagonal to synthesize dense cross-band grids, post-peak colors/ratios, and characteristic timescales.
   - Compute **goodness-of-fit $\chi^2$ residuals against canonical physical/parametric curve templates** (often more robust on faint/noisy events than noisy free-parameter fits) alongside baseline-crossing / negative-signal fraction, SNR duration width, and uncertainty-weighted shape moments.
2. **Covariate-Shift Weighting & External Pretraining for Deep Sequence Models**:
   - When test carries noisier auxiliary metadata than train (e.g., estimated vs. exact measurements), run **Adversarial Validation sample reweighting** to restore CV→LB correlation—and ablate whether keeping observer-frame features + raw noisy covariate beats propagating noisy covariate errors into transformed timescales!
   - From-scratch RNNs/Transformers collapse on tiny irregular positive counts (`~100–200`); unlock neural diversity via **pretraining on larger external multi-band corpora** (testing ReLU and no positional encoding on irregular grids) and blending with GBDTs via **disagreement/conflict-zone rules (`|P_DL - P_GBDT|`)**.
3. **Base-Rate-Aligned Small-N `F1` Thresholding & Anti-Patterns**:
   - Tune thresholds via **global OOF PR-curve or median per-fold OOF threshold**, verifying that the test predicted positive rate stays tightly anchored to the true class base rate.
   - **Avoid on Small-N Imbalanced `F1`**: `SMOTE` / `ADASYN` / synthetic oversampling, aggressive SHAP feature pruning, heavy Optuna HPO, naive test pseudo-labeling, and blind multi-model/multi-seed blending when each fold contains only `~20–30` positives (regularized single GBDTs frequently out-generalize complex stacks!).
