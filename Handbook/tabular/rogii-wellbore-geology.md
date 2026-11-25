# ROGII — Wellbore Geology Prediction: Top 10 Gold Solutions, Code Deep-Dive & Post-Mortem

> **Competition**: [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) (May 5 – Aug 6, 2026 | 6,125 Teams | 161,975 Submissions | \$50,000 Prize Pool)  
> **Task**: Predict stratigraphic **True Vertical Thickness (`tvt`, ft)** every `1 ft` along the hidden evaluation suffix of horizontal oil/gas wells from known prefix `TVT_input`, wellbore path `(MD, X, Y, Z)`, horizontal Gamma Ray (`GR`), and vertical reference `typewell.csv`  
> **Evaluation Metric**: Pooled row-level **RMSE** (Public ~26% / ~52 wells; Private ~74% / ~148 wells, 1 private outlier well excluded)  
> **Dataset**: `773` Train wells (`3,783,989` scored rows; consolidating into `54` master typewell GR series) | `~200` Hidden Test wells (`<= 9h` GPU/CPU notebook)

---

## Official Top 10 Gold Leaderboard Standings & Solution Index

Public LB (~52 wells) suffered severe shakeup (Public #1–#3 fell -17 to -27 ranks), while physics-grounded **2D `TVT × MD` alignment U-Nets trained jointly on real + synthetic wells, AnchorCNN Dynamic Programming marginalization, and HMM/Particle-Filter + SoftMax convex gates** climbed +8 to +41 ranks into Private Top 10:

| Private Rank | Δ Rank | Team | CV (RMSE ↓) | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **+30** | **Ruby** (`w5833946`) | **4.627** | 5.980 *(#31 5.648)* | **5.639** | 6× **`ConvNeXt-Small` (`LayerNorm → BatchNormFp32`, `track_running_stats=False`)** 2D U-Nets (`345×400` grid) + joint real+Z-shift synth + CE/Huber/GR-penalty loss + **reliability-gated XY-neighbor vs PF routing** | [Writeup](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/1st-place-solution) · [GitHub](https://github.com/IAmAValidUsername/kaggle_ROGII_1st_place_solution_Ruby) · [Notebook](https://www.kaggle.com/code/w5833946/submit-reproduce) |
| 🥈 **#2** | **+41** | **Bilzard** (`tatamikenn`) | **5.140** | 6.146 *(#43 5.780)* | **5.802** | **`AnchorCNN`**: `EffNet-B0 + FPN` predicts conditional moves $P(\Delta\text{TVT}\mid\text{TVT})$ → **exact DP marginal expectation** + 54-master re-skinning synth (`2,048/ep`) + **MD-phase TTA8** + **Leave-Largest-Contribution-Out (`k*`) CV** | [Writeup](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/2nd-place-solution-anchorcnn-conditional-probab) · [Notebook](https://www.kaggle.com/code/tatamikenn/rogii-2nd-place-inference?scriptVersionId=340907651) |
| 🥉 **#3** | **+39** | **tereka & Takoi** | **5.2884** | 6.043 *(#42 5.769)* | **5.836** | Strictly fold-safe **`5×5` GroupKFold**: 3-family HMM + 500-particle PF + **Last-PS TCN/MaxViT1D (`GaussianNLL`)** + Delta NN + 1D SDF → **BiLSTM SoftMax Convex Gate** | [Writeup](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/3rd-place-solution) · [Notebook](https://www.kaggle.com/code/tereka/rogii-exp417-gnll-five-groupkfold-submit?scriptVersionId=340176212) |
| 🏅 **#4** | **+8** | **L (nust misis) & J & A & A** | **4.998** | 5.452 *(#12 5.269)* | **5.870** | Ridge blend (`0.644/0.327/0.029`): **`ConvNeXtV2-L` 3-mode MTP FPN** (500k+37k synth pretrain, V-flip + quarter re-anchor TTA; **`5.730` Priv solo!**) + 16-ch U-Nets/Squeezeformer + PF+LGBM | [Writeup](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/4th-place-solution) |
| 🏅 **#5** | **+22** | **daimaru** (`daimaruk`) | **4.87** | 5.593 *(#27 5.585)* | **5.940** | **Latent high-res `0.25 ft` geology synth** (`6.342` Priv zero-shot!) + 14-ch `256×(768+48)` canvas with **48-col appendix painting aux loss** (`maxvit_tiny`/`effnetv2_rw_s`/`hrnet_w18`) + 2-pass wide-canvas edge fallback | [Writeup](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/5th-place-solution) |
| 🏅 **#6** | **+14** | **k256.dev** (`k256net`) | **5.4577** | 5.626 *(#20 5.458)* | **5.984** | **91 decorrelated GPU Particle Filters** (200× speedup, Kitagawa ancestral smoothing) across 5 GR reps (`twGR`, `selfGR`, `nbrGR`, `nbr_gr5`, `self_graft`) + OOF dip prior + attention soft-bag + **de-shrink gain `1.03`** | [Writeup](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/6th-place-diverse-gr-representations-91-pfs) · [GitHub](https://github.com/kosakae256/kaggle_rogii) |
| 🏅 **#7** | **+15** | **roglike** | **5.2025** | 5.518 | **6.057** | Numba HMM (`5.676`) → learned UNet emissions (`5.301`) → **Stage-3 Refiner UNet pretrained on fabricated random-walk errors** + disagreement-gated hedges `σ/(σ+1)` | [Writeup](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/7th-place-solution-hmm-unet-agent-is-all-you) |
| 🏅 **#8** | **+15** | **富士山** | 5.71 / 6.31 | 5.521 | **6.180** | 20 CPU PF segment ranker (`5.709`) + ResNet-34 U-Net keeping **`synth_prob=0.5` joint continuation** (`6.6 → 5.06` Fold 0) + 70/30 Stratified + Spatial-Group fold blend | [Writeup](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/8th-place-pf-ranker) |
| 🏅 **#9** | -3 | **tremors** | — | 5.078 *(#6)* | **6.251** | 8-member zoo (`Conv2d+BiGRU`, `well/relief-unet`, drift HMM, 24-beam seg) + **entropy-weighted 10-view TTA** (`6.33 → 5.86`) + 7-band position-dependent NNLS + spread calibration | [Writeup](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/9th-place-solution) |
| 🏅 **#10** | +11 | **Can** (`canentay`) | **5.7189** | 5.620 *(#21 5.488)* | **6.269** | PF + Gaussian/Cauchy HMM + 2D U-Nets → LGBM candidate selector + dilated TCN corrector + **OOF WLS disagreement-gated blend** + heel-anchored gain `1.024` | [Writeup](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/10th-place-solution-a-compass-not-a-map) |

---

## 1. Domain Physics Forensics & Why Pure Tabular GBDTs Plateaued

1. **Fundamental Stratigraphic Identity (`TVT + Z = A(MD)`)**: Across `766/773` wells, `TVT(MD) + Z(MD) - formation_top` is constant to `0.07 ft` — meaning $\Delta\text{TVT} = \Delta z_{\text{layer}} - \Delta Z$, where known `Z(MD)` swings `160.6 ft` median vs. only `26.4 ft` for `TVT`, and expert-interpreted `dz_layer/dMD` is piecewise-linear with rare geological faults.
2. **Why Point Regressors / Per-Row GBDTs Failed (`#2`, `#6`)**: Periodic Eagle Ford Gamma Ray beds (`~15–18 ft` aliasing wavelength) induce a **multimodal posterior** ("upper vs. lower stratigraphic branch"). Row-wise GBDTs and direct scalar regressors collapse onto the unphysical midpoint between modes (`>7–13` RMSE) and cannot fix whole-segment branch jumps.
3. **Random-Walk Heavy Tail & Error Geometry**: `89%` of total SSE sits `>1,500 ft` past Prediction Start (PS); the worst **10 wells (`1.3%`) generate `22.7%` of SSE**, and the worst **39 wells (`5%`) generate `43.9%` of SSE** (#10).
4. **Self-Prefix & Sibling-Lateral GR Beat Smoothed Typewell GR**: Lateral GR measured before PS has higher vertical resolution than smoothed vertical `typewell.csv` (`discussion/698825`); 752 unique train typewell CSVs consolidate into **54 master series** (#2). Meanwhile PNG/PPTX plots use meters and mismatched windows — do not treat them as clean inputs.

---

## 2. Core Winning Architectures & Code Deep-Dive

### 2.1 1st Place Code Deep-Dive (`Ruby` — [`kaggle_ROGII_1st_place_solution_Ruby`](https://github.com/IAmAValidUsername/kaggle_ROGII_1st_place_solution_Ruby))
Inspecting `reference_results/0803_V2/seq_NN_pretrained_unet.py` and `seq_NN_train.py` directly reveals exact engineering choices behind the `5.639` Private win:
- **2D Alignment Lattice (`345 MD × 400 TVT`)**: Downsampled `×32` along MD (`1,024` visible + `10,000` target) × `0.5 ft` vertical TVT grid (`±100 ft` around last visible TVT), combining typewell GR, prefix-calibrated GR, per-32ft horizontal GR stats/quadratic fit/RMSE, `|tw_GR - hw_GR|`, `z_diff`, optional GPU Particle Filter heatmap, and weighted-least-squares XY-neighbor dip surface `ΔTVT = aΔX + bΔY - ΔZ`.
- **Critical Backbone Hack (`LayerNorm → BatchNorm2dFp32` on `timm/convnext_small.in12k_ft_in1k_384`)**:
  - Replaced every ConvNeXt `LayerNorm` / `ChannelLayerNorm` with `BatchNorm2dFp32` / `ChannelLastBatchNorm2d` using **`track_running_stats=False`**, `eps=1e-3`, computed in **FP32 inside BF16 autocast**, **copying pretrained LayerNorm `weight`/`bias` directly into the new BatchNorm affine parameters** (`_copy_norm_affine`)! Disabling running stats prevented train-vs-test statistic drift while preserving pretrained ConvNeXt weights.
  - Simple bilinear upsampling + `ResBlock2d(SiLU, BN)` outperformed learnable down/upsampling.
- **3-Term Physics-Aligned Loss (`seq_NN_train.py`)**: Exponentially smoothed column-wise Cross-Entropy (`masked_typewell_soft_cross_entropy`) + Huber loss on expected TVT path (`masked_offset_huber_loss`) + **`GR_penalty = mean(prob × grid_GR_gap)`** (`masked_gr_penalty_loss`).
- **Reliability-Gated Routing**: Because test wells are spatially more isolated (`683 ft` median NN vs `470 ft` in train), `~10%` of wells exceeding the 95th percentile of 5 neighborhood reliability stats routed strictly to default + PF ConvNeXt models (`0.40/0.20/0.40`), while reliable-neighbor wells routed predominantly to XY-neighbor ConvNeXt models (`0.28×3 + 0.07×2`) → **CV `4.627` / Private `5.639`**.

### 2.2 Three Orthogonal Winning Paradigms (#2, #3, #4, #5, #6, #7)
1. **AnchorCNN + Exact DP Marginalization (`#2 Bilzard`, `5.802` Priv)**: Predicted **conditional transitions $P(\Delta\text{TVT} \in \{0, \pm 2, \dots, \pm 20\} \mid \text{TVT}, \text{MD})$** via `EffNet-B0 + FPN` and computed exact marginal path expectations via Dynamic Programming — outperforming Viterbi mode-tracking, stochastic rollout rerankers, and GRPO RL! Combined with **MD-phase TTA8** (shifting the 32-ft grid origin by `{0, 8, ..., 56}` ft to defeat 18-ft GR aliasing: OOF `5.624 → 5.417`).
2. **Joint Real + Physics-Consistent Synthetic Training (`#1`, `#2`, `#4`, `#5`, `#8`)**: Unanimous consensus that **joint mixed real+synthetic training** (2,048 synthetic wells/epoch in #2; `synth_prob=0.5` continuation in #8 breaking Fold 0 from `6.6 → 5.06`; #5's latent `0.25 ft` geology generator reaching `6.342` Private **zero-shot**) beat 2-stage pretrain-then-finetune (#1).
3. **Multi-Trajectory Prediction (MTP) + Vertical-Flip & Quarter Re-Anchor TTA (`#4 James`, `5.730` Priv solo)**: `ConvNeXtV2-Large` FPN predicting 3 candidate trajectories + mode probabilities; physical vertical flip (`-0.13` CV) + cropping last 75% anchored at 25% prediction (`-0.18` CV).
4. **91 Decorrelated GPU Particle Filters + Convex SoftMax Gating + De-Shrink Gain (`#3`, `#6`, `#10`)**: Running 91 PFs across 5 GR representations (including **GR-similarity neighbors ~5,800 ft away**, not just XY-nearest!) + strict 5-fold OOF physical dip prior (#6) or fusing HMM + PF + `GaussianNLL` TCN/MaxViT1D + Delta NN via a **BiLSTM SoftMax Convex Gate** (#3) beat unconstrained sequence refiners on 5/5 folds; global **de-shrink gain `a ≈ 1.024–1.03`** undid soft-averaging amplitude shrinkage (#6, #10).
5. **Heavy-Tail Validation & Fabricated-Error Refiners (`#2`, `#7`, `#9`)**:
   - **Leave-Largest-Contribution-Out (`k*`) CV (#2)**: Sort wells by `|SSE_treat - SSE_base|` and verify how many top wells `k*` can be dropped before a CV win disappears (reject `<8`, accept `>52`).
   - **Fabricated-Error Refiner Pretraining (#7)**: Pretrain Stage-2 refiner UNet on `truth + low-pass random-walk noise` before real fine-tuning to prevent it from simply copying clean in-fold OOFs.
   - **Entropy-Weighted 10-View TTA (`softmax(-T·entropy)`, #9)**: Cut OOF `6.33 → 5.86`.

---

## 3. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Per-Row Tabular GBDTs / Unconstrained Point Regressors** | Plateaued far behind 2D lattice/PF models (`>7–13` RMSE) | #2, #6 | Averages multimodal periodic GR branches into unphysical midpoints |
| **Two-Stage Pretrain → Real-Only Finetune (vs. Joint Real+Synth)** | Worse CV/LB; plateaued at `6.6` vs `5.06` joint (#8) | #1, #8 | Only 773 real wells causes rapid overfitting once synthetic stream turns off |
| **Transformer Backbones vs. ConvNeXt / EffNet / ResNet 2D U-Nets** | Lost consistently to ConvNeXt (#1); `iTransformer`/`Mamba`/`BiGRU` failed (#6) | #1, #6 | 2D matching lattices need strong local spatial translation equivariance |
| **Viterbi Mode Decoding / RL & Learned Rollout Rerankers** | Lost to exact DP marginal expectation | #2 | Under squared-error RMSE, posterior expectation $\mathbb{E}[\text{TVT}]$ dominates MAP/mode paths |
| **Unconstrained Sequence Refiners / Training Refiner Only on Real OOFs** | Lost in 5/5 folds to convex SoftMax gate (#3); copied input (#7) | #3, #7 | Convex gating prevents catastrophic tail divergence; real OOFs have train-vs-val cleanliness gap |
| **Ungated XY-Neighbor Features / Hard Well-Level Shifts / Clipping to Typewell Range** | Boosted CV (`-0.18` to `-0.30`) while hurting isolated test wells; clipping `+0.29` worse (#4); constant shifts `+0.184` worse (#6) | #1, #4, #6, #8, #10 | Test wells are more isolated spatially (`683 ft` vs `470 ft` median); requires strict OOF + reliability gating |
