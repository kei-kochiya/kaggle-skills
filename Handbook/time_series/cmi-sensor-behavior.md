# CMI — Detect Behavior with Sensor Data — Top Gold Solutions & Post-Mortem

> **Competition**: [CMI - Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data) (May 29 – Sep 2, 2025 | 2,657 Teams | 75,777 Submissions | \$50,000 Prize Pool)  
> **Task**: Classify **18 wrist-worn gestures** (**8 BFRB target** vs **10 non-target**) from `IMU` (`acc` + quaternion `rot`), `5× Thermopiles`, & `5× 8×8 Time-of-Flight (ToF)` sensor streams (`693` cols), where **50% of test sequences are IMU-only** (`thm`/`tof` missing)  
> **Evaluation Metric**: $\frac{1}{2}\left(\text{Binary F1}_{\text{BFRB vs non-BFRB}} + \text{Macro F1}_{9\text{-class}}\right)$ (10 non-target gestures collapsed into `non_target`)  
> **Dataset**: `train.csv` (`~8,151` sequences across `~81` subjects, `1.12 GB`) | Hidden Test: `~3,500` sequences (`42%` Pub / `58%` Priv, served 1 sequence at a time in randomized order; `<= 9h` GPU/CPU notebook)

---

## Official Top Gold Leaderboard Standings & Solution Index

Three decisive axes separated Top-10 gold solutions and drove `+14` to `+16` Private shakeup climbs: **(1) sensor physics & anomaly forensics** (left-handed canonicalization, per-sequence **180° Z-rotated upside-down sensor detection**, and `acc`-only fallbacks for missing quaternions), **(2) multi-branch 1D-CNN/BiGRU/Mamba + 2D-Vision-on-sensor-matrix diversity** (blending IMU-only models into Full-sensor predictions), and **(3) online subject-protocol Hungarian assignment (`+0.014` to `+0.027` Private)**:

| Private Rank | Public Rank | Team | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **#1** | **`Devin \| Ogurtsov \| zyz`** | **0.904551** | **0.886193** | Left-hand canonicalization + world-frame gravity removal (`linear_acc`/`global_acc`) + channel-independent multi-scale 1D-CNN/BiGRU + 3D ToF Conv (`~0.83` solo) + **BatchNorm-first (no scaler)** + Full+IMU logit blend + online subject-protocol PP | [Writeup](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/cmi-1st-place-solution) · [GitHub](https://github.com/statist-bhfz/kaggle_cmi_1st_place_solution) · [NB](https://www.kaggle.com/code/yuanzhezhou/final-sub-cmi3) |
| 🥈 **#2** | #2 | **`daiwakun`** | 0.900320 | **0.878839** | **4-way missingness grid** (`rot` × `THM/ToF`), **6D rotation rep**, 102/144 composite targets, **phase-aware 3-pool attention**, online batch PL (`+0.004` Priv), & **online Hungarian assignment (`+0.020` Priv: `0.858 → 0.878`)** | [Writeup](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/2nd-place-solution) · [GitHub](https://github.com/Yamato-Arai/kaggle-cmi-detect-behavior-2nd-place-solution) |
| 🥉 **#3** | #4 | **`ln \| yu4u \| Theo \| minerppdy`** | 0.897932 | **0.878261** *(+1)* | True-north `-110°` quat fix + symmetric `rot(q)+rot(-q)` block; **2D Vision on resized sensor matrices (`EffNet-B5`, `MaxViT`, `ConvNeXt-V2`)** + `ModernBERT` + 3D CNN; **fixed-epoch Cosine+EMA** (avoiding F1 early-stop overfit) + Hungarian PP (`+0.014` Priv) | [Writeup](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/3rd-place-solution) |
| 🏅 **#4** | #5 | **`dott`** | 0.893481 | **0.876955** *(+1)* | Multi-branch 1D-CNN/BiGRU + 5-fold BERT + 102/144 composite targets + online Hungarian PP | [Writeup](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/4th-place-solution) · [NB](https://www.kaggle.com/code/dott1718/cmikn-ensemble3b-re5) |
| 🏅 **#5** | #3 | **`have fun`** (Rib~ & Ethan) | 0.898439 | **0.873902** | **Subject-Level Cross-Sequence Transformer** aggregating all accumulated sequences per subject (**`+0.02` CV** over sequence-only), blended via history-ramp weights + `0.2` IMU blend into Full | [Writeup](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/5th-solution) |
| 🏅 **#6** | #22 | **`Jack (Japan)`** (`rsakata`) | 0.875011 | **0.873899** *(+16)* | **Per-sequence 180° Z-rotation upside-down detector (`+0.013` Priv: `0.860 → 0.873`!)** + facing-angle-corrected handedness flip + 8-head multi-branch U-Net segmenter w/ `acc`-block fallback | [Writeup](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/6th-place-solution) · [NB](https://www.kaggle.com/code/rsakata/cmi3-6th-place-solution-partial) |
| 🏅 **#7** | #11 | **`ishikei`** | 0.882676 | **0.869938** *(+4)* | Pure 2D Vision (`ConvNeXt-Base` IMU + `EffNet-B5` Full) on `(512, 1024)` proximity-ordered sensor images + multi-scale behavior head + count-equalizing PP (`+0.005` Priv) | [Writeup](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/7th-place-solution) |
| 🏅 **#8** | #9 | **`My code wrote Junie`** (K. Yakovlev) | 0.883616 | **0.868994** *(+1)* | Unified model w/ `fusion_modality_dropout=0.2` + dual fusion streams + Dirichlet-pruned ensemble + online PP | [Writeup](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/8th-place-solution-and-competition-walkthrough) · [PP NB](https://www.kaggle.com/code/kyakovlev/cmi-8th-place-postprocessing) |
| 🏅 **#9** | #7 | **`Fnoa`** | 0.884853 | **0.868692** | 51 `(orient, gesture)` normal + inverted BCE targets + aux orientation head (`+0.01`) + subject-history PP (`+0.027` Priv: `0.841 → 0.868`) | [Writeup](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/9th-place-solution) |
| 🏅 **#10** | #24 | **`ohkawa3`** | 0.874796 | **0.867474** *(+14)* | **Per-sequence 180° Z-rotation fix (`+0.010` Priv: `0.857 → 0.867`!)** + physics-preserving `1/s²` linear-acc time stretch + 7-layer pooling-free 15-frame CNN + 3-seed 15-model CV | [Writeup](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/10th-place-solution) |

---

## 1. Universal Wearable Sensor & Multimodal Sequence Breakthroughs

1. **Sensor Physics, Handedness Canonicalization & Inverted-Wearing Detection (`#1`–`#19`)**:
   - **Chirality / Left-Right Mirroring**: Public test had **100% right-handed** subjects, while Private had left-handed subjects + `NaN` handedness (`discussion/603543`). Canonicalizing left-handed streams to right-handed coordinates (`-acc_x`, `-ang_vel_{y,z}`, swapping `thm/tof 3 ↔ 5`, mirroring 8×8 ToF grids, canceling facing yaw `#3, #6`) prevented silent Private collapse.
   - **Per-Sequence 180° Z-Rotation Detection (`+0.010` to `+0.013` Priv, `#6`, `#10`)**: Two train subjects (`SUBJ_019262`, `SUBJ_045235`) and ~30–40 Private test sequences wore the watch upside-down (even on a minority of a subject's own recordings!). Detecting & rotating per-sequence (`-acc_{x,y}`, quat rotation, swap `2↔4` & `3↔5`, `rot90(k=2)` ToF) drove the largest Private climbs (`+16` & `+14` ranks).
   - **Quaternion Sign Continuity & Gravity Removal**: Quaternions jump discontinuously across `-q ≡ q`; **6D rotation representation (`+0.002–0.003` CV over Euler RPY, `#2`, `#17`)**, dot-product sign alignment (`dot(q_t, q_{t-1}) >= 0`, `#10`, `#12`), or symmetric `f(q) + f(-q)` (`#3`) eliminated artifacts. Rotating `acc` by `rot` into world frame separates gravity (`9.81 m/s²`) from linear motion — and **dedicated `acc`-only models/branches** (`#2`, `#6`, `#13`, `#14` `+0.0015` Priv) saved sequences where `rot_*` was missing.
2. **Multi-Branch Sensor Encoders + 2D-Vision Diversity + Dual-Modality Routing**:
   - **Channel-Independent / Grouped 1D-CNN + SE + BiGRU/BiLSTM/Mamba (`#1`, `#11`, `#13`, `#19`)**: Avoid early dense feature-mixing layers on `~8k` sequences; combine multi-scale 1D-CNN (`k=[3,5,7]`), 2D/3D spatial CNNs on ToF grids, and **phase-aware temporal pooling (`#2`, `#6`)** so long rest/transition segments don't drown out short gestures.
   - **2D Vision Backbones on Resized `(T, F)` Sensor Images (`#3`, `#7`)**: Ordering sensor channels by physical proximity and resizing `(T, F)` to 2D images for `EfficientNet-B5` / `ConvNeXt-V2` / `MaxViT` delivered strong standalone accuracy (`0.869–0.897`) and top-tier ensemble diversity over 1D models.
   - **Blending IMU-Only Into Full-Sensor Predictions (`#1`, `#3`, `#4`, `#5`)**: Even when ToF/THM sensors are present, blending `0.2–0.3` IMU-only logits consistently improved Full-sensor generalization (immune to partial ToF dropouts).
3. **Small-Subject Grouped CV Discipline & Online Group Assignment (`#2`, `#3`, `#5`, `#9`)**:
   - Under `StratifiedGroupKFold(subject)` with only ~81 subjects, per-fold early stopping & `ReduceLROnPlateau` overfit `±0.01` F1 noise — use **fixed epochs + cosine decay + EMA (`0.99–0.9995`, `#3`)** and multi-seed averaging (`#10`).
   - Exploiting cross-sequence subject context via a **Subject-Level Transformer (`+0.02` CV, `#5`)** or **Online Hungarian Assignment (`linear_sum_assignment`)** under protocol slot capacities (`+0.014` to `+0.027` Priv, `#2`, `#3`, `#9`) unlocked massive final gains.

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **OOF Knowledge Distillation & Modality Hallucination (IMU → ToF/THM)** | OOF KD leaked CV to `0.88` while **hurting LB** (`#17`); hallucination gave `0` gain (`#13`) | #13, #17 | Cross-fold OOF teacher targets leak subject information; missing spatial depth cannot be reliably hallucinated from wrist IMU |
| **Classical Signal Filters (Butterworth/SavGol/Wavelet/Kalman) & Spectrograms** | Consistently degraded or matched raw waveforms (`#11`, `#13`, `#29`) | #11, #13, #29 | Short gesture transients lose sharp onset edges under smoothing/STFT windowing |
| **`StandardScaler` / Per-Sequence Normalization vs. BatchNorm-First** | BatchNorm-first (no external scaler) outperformed explicit scalers | #1, #9, #11, #19 | Per-sequence normalization erases absolute gravity orientation and resting temperature baselines |
| **Using Demographics (`age`, `sex`, `height`) & GBDT Stacking** | Demographics overfit 81 subjects (`#11`, `#17`, `#29`); LGBM stack gave `0` gain (`#19`) | #11, #17, #19, #29 | High subject-ID memorization risk; `handedness` also had `NaN`s in Private test |
| **Early Linear Cross-Sensor Mixing, ArcFace/Triplet, & Optuna OOF Weight Tuning** | Overfit 8k sequences / noisy F1 CV (`#11`, `#13`, `#19`) | #11, #13, #19 | Simple probability/rank averaging or Dirichlet-pruned weights generalized far better than over-tuned meta-weights |
