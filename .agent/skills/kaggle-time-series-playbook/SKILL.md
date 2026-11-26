---
name: kaggle-time-series-playbook
description: >-
  Universal competitive time-series and multi-agent spatiotemporal forecasting playbook for Kaggle. Use when setting up grouped/temporal CV, parameterizing displacement/cumsum targets, choosing GaussianNLL vs decayed TemporalHuber losses, building factorized spatiotemporal encoders (Depthwise Conv1d / GRU + Entity Transformer), exploiting geometric symmetries and early-frame shift augmentation, or ensembling trajectories.
---

# Kaggle Time Series & Spatiotemporal Forecasting Playbook

A universal competitive workflow for sequential and multi-agent spatiotemporal forecasting challenges. Competition-specific field geometry, IDs, and external seasonal datasets remain strictly in `Handbook/time_series/`.

---

## 1. Phase 1: Prospective & Grouped Temporal Validation

1. **Match Live / Future Evaluation Regimes**:
   - Never split windows from the same episode, game, sensor session, or contiguous date block across train and validation.
   - Use **repeated `GroupKFold(session/episode_id)`** (e.g., `3×5` splits) or a **chronological trailing-window holdout** mirroring the future test horizon.
2. **Audit Outliers Before Feature Engineering**:
   - Inspect sequence length extremes and missing anchor entities early; a handful of corrupted sequences can destabilize gradient norms across all folds.

---

## 2. Phase 2: Target Parameterization & Losses

Refer to [`references/spatiotemporal_forecasting_recipes.md`](./references/spatiotemporal_forecasting_recipes.md) for implementations.

1. **Predict Displacements or Step Deltas (`cumsum`) — Not Raw Positions or Polar Angles**:
   - Predict either **offset from last observed state** $(\Delta x_t, \Delta y_t)$ or **step-to-step velocity deltas integrated via `cumsum`**.
   - Avoid predicting raw absolute coordinates or discontinuous polar `(speed, angle)` directly (angle wrap-around and compounding error drift degrade accuracy).
   - Add **1st- and 2nd-order finite-difference smoothness/auxiliary losses** (velocity and acceleration) directly on reconstructed sequences.
2. **Heteroscedastic `GaussianNLLLoss` vs. Decayed `TemporalHuber` (And Blend Both!)**:
   - **`GaussianNLLLoss`**: Predict `(mean, softplus(var) + eps)` so the network automatically downweights irreducible late-horizon uncertainty instead of letting unpredictable tail cases corrupt gradients.
   - **Decayed `TemporalHuber`**: `Huber(delta)` weighted by $\exp(-\lambda \cdot t)$ over horizon step $t$ (ensure clean numerator-only mask normalization without double-squaring weights).
   - **Multi-Loss Ensembling**: Train parallel models under `GaussianNLLLoss` and `TemporalHuber` and average their trajectories.

---

## 3. Phase 3: Symmetries & Self-Supervised Trajectory Expansion

When sequence counts are limited (`10k–50k` episodes), physics-preserving augmentation outperforms heavy hand-crafted tabular features:

1. **Coordinate Canonicalization & Exact Geometric Symmetries**:
   - Canonicalize global attack/flow direction (`180°` alignment) to halve the state space.
   - Apply exact **chiral mirror flips** (negating reflected velocity/acceleration components and wrapping angles) during both training and `2×–4×` TTA.
   - For isotropic multi-agent systems, test **uniform $360^\circ$ group rotation** around the entity centroid (frequently outperforms small-angle Gaussian rotations by breaking axis-aligned priors).
2. **Early-Frame / Sliding-Window Origin Shift**:
   - Randomly shift the forecast origin earlier by $T \in [0, T_{\max}]$ historical steps, training the model to forecast `(T historical steps + original future horizon)` conditioned on $T$ (with $T=0$ at test time).
   - Complement with **self-supervised inter-frame & endpoint auxiliary losses**, random history head-cropping, and entity dropout/slot-order shuffling (to remove slot-index bias).

---

## 4. Phase 4: Factorized Spatiotemporal Neural Architectures

Lean raw/kinematic inputs fed into factorized neural encoders consistently beat bloated hand-crafted tabular feature banks and GBDTs on multi-agent dynamics:

1. **Factorize Time and Entity Axes**:
   - **Temporal Axis**: Use **Depthwise `Conv1d`** (channel-wise, avoiding regular `Conv1d` overfitting on modest datasets) or **RNN / GRU** encoders (which preserve local kinematic inductive bias better than pure temporal self-attention).
   - **Entity Interaction Axis**: Stack **wide, shallow Entity Transformer layers** (e.g., `SwiGLU` FFN, `2–3` layers, low/zero dropout on continuous coordinate regression) across masked entities.
   - **Decoders**: Pair with `Conv1d` / `TXP-CNN` temporal decoders, RoPE future-frame index queries, or coarse-to-fine residual decoders; consider single-target sparse attention pooling (**Entmax**) when focusing on one target entity at a time.
2. **The External Target / Global Context as a Graph/Attention Node**:
   - When a known reference point or shared global cue is relevant to *some* entities but irrelevant noise for others, **do not concatenate it onto every entity's feature vector**—inject it as a **separate fixed token/node** in entity attention so each entity dynamically attends to or ignores it.
3. **Training Stability**:
   - Always enable **EMA (`decay = 0.995–0.9995`)** to damp volatile sequence validation curves; test **Muon**, `RAdam`, or layer-wise `AdamW` optimizers.

---

## 5. Phase 5: Trajectory Ensembling & Universal Anti-Patterns

- **Ensembling**: Combine diverse temporal/interaction backbones (`Depthwise-Conv+Transformer`, `GRU+Transformer`, `ST-Transformer`, `GNN`), multiple losses (`GaussianNLL` + `TemporalHuber`), repeated grouped CV splits, and exact symmetry TTA. When ensembling directional trajectories that diverge in turn direction, test **polar-vector averaging** (averaging speed magnitudes and unit direction vectors separately to avoid velocity collapse).
- **Universal Anti-Patterns**:
  - **CVAE / multimodal sampling under unimodal RMSE**: Under squared error, the Bayes-optimal predictor is the conditional expectation $\mathbb{E}[y \mid x]$; sampling stochastic modes hurts RMSE.
  - **Heavy public tabular feature bloat**: Slows iteration and adds collinear noise compared to compact raw kinematics + relative vectors.
  - **Pretrain-only on historical seasons without joint fine-tuning**: Historical drift requires joint inclusion during final fine-tuning rather than static pretraining alone.
  - **Over-predicting rare extreme horizons**: Clamp or extrapolate cautiously beyond the densely supported training horizon.

---

## 6. Phase 6: Streaming Finance, Wearable Sensors & Multi-Agent Pose Segmentation

Refer to [`references/financial_online_sensor_and_pose.md`](./references/financial_online_sensor_and_pose.md) for concrete implementations:
1. **Streaming Financial Online Learning & Risk Engineering**: Pair **Cross-Asset Self-Attention + causal Temporal GRU** + multi-horizon auxiliary targets with **daily online updates** on lagged ground truth (fresh `AdamW` or short-memory `betas=(0.8,0.95)`, NEVER carry offline Adam momentum, always update on `is_scored=False` warmup; rolling block refits for GBDTs). Validate decay via **200-day-gap holdouts**; in ultra-low-SNR markets, prioritize **inverse-vol weighting, metric-anchored vol targeting, shrunk Kelly rank covariance ($\Sigma_{\text{rank}}^{-1}\mu_{\text{rank}}$)**, and strict causal `ffill` (never `bfill`).
2. **Wearable IMU & Multimodal Sensor Fusion**: Canonicalize handedness chirality, run a **per-sequence 180° upside-down detector**, replace discontinuous quaternions with **6D rotations** + world-frame gravity removal + `acc`-only missing-`rot` fallbacks. Combine **BatchNorm-first** channel-independent 1D-CNN/BiGRU + **2D Vision on resized `(T,F)` sensor matrices**, blend IMU-only into Full predictions, use **fixed-epoch Cosine+EMA** (beat F1 early stopping), and apply online Hungarian assignment when group protocol constraints exist.
3. **Multi-Agent Pose Action Segmentation**: Pretrain on unlabeled pose via **future-motion SSL** (beats naive PL), canonicalize heterogeneous sources to uniform FPS + physical scale + **egocentric master skeleton** (selective NaN interpolation only to avoid ID-switch spikes), enforce **3-state partial-supervision loss masking**, fuse **GNN/ST-GCN + Squeezeformer + GBDTs**, crop sliding-window boundaries, and break multi-action ties by margin/paired tournament.
4. **Multi-Horizon Retail Demand Forecasting (`MAPE`)**: Validate with **horizon-matched multi-window rolling + same-season CV** (never chase small random Public LBs), audit closed-day gaps & patch missing holidays over merged `train+test` calendars, trim stale pre-growth history under structural shifts, ablate **`log(y) + MAE`** stability vs. native `MAPE`, engineer holiday proximity/closure/LLM-description embeddings, and avoid raw short target lags on long direct horizons.
5. **Irregular Multi-Band Heteroscedastic Classification (Small-N `F1`)**: Use **2D `Time × Channel` GP interpolation** (with measurement error `err_i²` in the noise diagonal) + **parametric template $\chi^2$ residuals**, fix measurement/covariate shifts with **adversarial-validation sample reweighting**, unlock sequence NNs via **external-corpus pretraining** + disagreement-zone blending, anchor OOF PR-curve/median-fold `F1` thresholds to the true base rate, and avoid `SMOTE`, aggressive SHAP pruning, and over-tuned Optuna when positive counts are tiny (`< 300`).

---

## 7. References & Case Studies
- [Spatiotemporal Losses, Early-Frame Shift, Context-Node Attention & Polar Ensembling](./references/spatiotemporal_forecasting_recipes.md)
- [Streaming Financial Online Learning, Wearable Sensors, Multi-Agent Pose, Retail Demand & Irregular Multi-Band TS](./references/financial_online_sensor_and_pose.md)
- Case Studies: [`NFL BDB 2026`](../../../Handbook/time_series/nfl-big-data-bowl-2026.md) · [`Jane Street`](../../../Handbook/time_series/jane-street-realtime-market.md) · [`Hull Tactical`](../../../Handbook/time_series/hull-tactical-market-prediction.md) · [`CMI Sensor`](../../../Handbook/time_series/cmi-sensor-behavior.md) · [`MITSUI Commodity`](../../../Handbook/time_series/mitsui-commodity-prediction.md) · [`MABe Behavior`](../../../Handbook/time_series/mabe-mouse-behavior.md) · [`Rohlik Orders`](../../../Handbook/time_series/rohlik-orders-forecasting.md) · [`MALLORN Astronomy`](../../../Handbook/time_series/mallorn-astronomical-classification.md)
