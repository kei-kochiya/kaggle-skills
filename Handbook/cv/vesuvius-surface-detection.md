# Vesuvius Challenge — Surface Detection: Top 10 Gold Solutions & Post-Mortem

> **Competition**: [Vesuvius Challenge - Surface Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection) (Nov 2025 – Feb 28, 2026 | 1,391 Teams | 30,824 Submissions | \$200,000 Prize Pool)  
> **Task**: 3D CT scan surface segmentation of carbonized Herculaneum papyrus sheets (`0` bg, `1` fg, `2` unlabeled/ignore) without artificial sheet mergers or spurious holes  
> **Evaluation Metric**: `0.30 × TopoScore (Betti 0/1/2) + 0.35 × SurfaceDice@2.0 + 0.35 × VOI_score (α=0.3, 26-conn)`  
> **Dataset**: `27.32 GB` 3D `.tif` volumes (mostly `320³`) | `~120` Hidden Test volumes (`20%` Public / `80%` Private; `<= 9h` notebook)

---

## Official Top 10 Gold Leaderboard Standings & Solution Index

`nnU-Net ResEnc` trained for long schedules (`1,000–8,000` epochs) dominated **9 of the Top 10**, while topological post-processing contributed **`+0.02` to `+0.03+`** Private LB across nearly every winner:

| Private Rank | Δ Rank | Team | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | 0 | **Vesuvius Team** (`Tony Li`, `OzanM.`, `Yiheng Wang`, `PaulG`) | **0.61338 (#1)** *(0.606 sel.)* | **0.62702** | 4-model multi-patch **`nnU-Net`** (`128³` 4000ep → FT `192³`/`256³` + scratch `192³` 4000ep, `bs=2`) + 5-step topology PP (**`0.596 → 0.627` Priv**: CC `<20k`, **2×2×2 watertight LUT**, height-map hole patch, `r=3` closing, fill holes) | [Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su) · [NB](https://www.kaggle.com/code/tonylica/nnunet-4-model-7-5-2-1-final-submit-so-long?scriptVersionId=300373304) |
| 🥈 **#2** | **+17** | **risk of overfitting** (`Duong Nguyen`, `Marius Heuser`) | 0.58849 (#19) | **0.62250** *(0.631 unsel.)* | Vanilla `nnUNet ResEnc` (`128³` + `160³`, 1000ep logit blend) → **Euler-number (`χ < 1`) PCA/SVD surface interpolation** + **recursive multi-threshold (`0.2→0.5→0.6→0.7`) sheet unmerging** (`0.631` global unsel.) | [Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/2nd-place-solution-vesuvius-challenge-a-postproc) · [0.631 NB](https://www.kaggle.com/code/mariusheuser/interference-with-global-interpolation?scriptVersionId=300668972) |
| 🥉 **#3** | **+11** | **W & A** (`wpl`, `Arunodhayan`) | 0.59226 (#14) | **0.62090** *(0.624 unsel.)* | `nnU-Net` M-size (`128³` train → **`192³` infer**), **2-stage `3d_lowres → cascade` (`8000+8000 ep`)**, `Dice+CE+clDice`, `RAdamScheduleFree`, `±15°` in-plane rot TTA + **3D Frangi ridge + Z-consistency pruning** | [Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/quick-preview-of-the-3rd-place) · [GitHub](https://github.com/arunodhayan/Vesuvius-Challenge---Surface-Detection) |
| 🏅 **#4** | -2 | **Starry** (`shtljw`) | 0.60901 (#2) | **0.61986** *(0.622 abl.)* | 7-stage `nnU-Net ResEnc` (`192³`, `bs=8`, `2000ep`) + Stage-1 projection PCA hole filling + Stage-2 **`20³` C++ `betti_matching` local PCA fill** + Betti-2 metric hack | [Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/4-th-place-solution) · [NB](https://www.kaggle.com/code/shtljw/vcsd-4th-place-solution) |
| 🏅 **#5** | -2 | **Dieter** (Christof Henkel) | 0.60805 (#3) | **0.61941** | **Non-nnU-Net Pure PyTorch**: 3D **`SEResNeXt152` + AttUNet** trained w/ **Gaussian-weighted SDF L1 + SDF Mass Dice** → **full-volume `320³` inference (no SWI!)** + **13-iter C++ H1 tunnel sphere-filling w/ bridge check (`+0.08` TopoScore)** | [Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/5th-place-solution) |
| 🏅 **#6** | +4 | **`#hui`** (`da da`, `artineon`) | 0.59488 (#10) | **0.61847** | `nnUNetTrainerMedialSurfaceRecall` (`128³`, 1200ep) + geodesic watershed + Betti-1 **Dijkstra loop unrolling** + RBF fill + scroll-density adaptive PP | [Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/6-th-place-solution) |
| 🏅 **#7** | +5 | **DECEM** | 0.59388 (#12) | **0.61812** | `nnUNetResEncM` (`CE+Dice+SkeletonRecall`) **blending organizer-preprocessed labels + raw labels** + **`FastMuon` optimizer** (`0.595` vs `0.594` AdamW / `0.590` SGD) + 3D Frangi + GPU **Coherence Enhancing Diffusion (CED)** | [Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/7st-place-solution-for-the-vesuvius-challenge) · [FastMuon](https://github.com/Decem-Y/FastMuon) |
| 🏅 **#8** | -2 | **`lingyundev`** | 0.59860 (#6) | **0.61697** | Multi-patch `nnUNetResEncL` (`224³`/`256³`/`288³`, `MedialSurfaceRecall`) + **2× Perona-Malik Gradient Anisotropic Diffusion** + direction-aware hysteresis (`T_low=0.50, T_high=0.90`) | [Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/8th-place-solution) · [GitHub](https://github.com/LingYunGit/SurfaceDetection_8th_Solution) |
| 🏅 **#9** | -4 | **`阿對對對對隊`** | 0.59927 (#5) | **0.61641** | `nnUNetTrainerMedialSurfaceRecall` (`192³`, `MUSGD`, GroupNorm32+GELU) + 2D Z-slice skeletonize + `r=1.5` EDT normalization + 5-loop gap fill | [Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/9th-place-solution) · [GitHub](https://github.com/TingYi028/Kaggle-vesuvius-9th) |
| 🏅 **#10** | -2 | **Vibes & Scrolls Trade-off** | 0.59571 (#8) | **0.61490** *(0.624 post)* | 5-stage `ResEnc-L + Primus-B` + **Diffeomorphic SVF + SDF-TopoFix**; post-deadline `7–10×` iterative median filter jumped Priv `0.615 → 0.624` | [Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/10th-place-solution) · [GitHub](https://github.com/Sersasj/vesuvius-challenge-10th-solution) |

---

## 1. Universal 3D Volumetric & Topological Segmentation Lessons

### 1.1 Modeling Paradigms: Long-Schedule `nnU-Net ResEnc` vs. Full-Volume `SEResNeXt152` SDF
- **Multi-Patch Long-Horizon `nnU-Net` (`#1`–`#4`, `#6`–`#9`)**: Training `nnU-Net ResEnc` on `fold=all` with small batch (`bs=2`) for `1,000–8,000` epochs across multi-scale patches (`128³ → 192³ → 256³`), asymmetric patch inference (`128³` train → `192³` infer, `#3`), or `SkeletonRecall` / `MedialSurfaceRecall` / `clDice` losses provided rock-solid baselines.
- **Full-Volume `320³` Inference & Signed Distance Field (`SDF`) Regression (`#5 Dieter`)**:
  - Inferring directly on full `320³` volumes (no sliding window!) **eliminated border/overlap seam artifacts AND ran `1.5×` cheaper than `160³` SWI at `0.5` overlap**!
  - Predicting clamped GPU SDF `sdf = clamp(bg_edt - fg_edt, -100, 5)` under `0.5 × Gaussian-Weighted SDF L1 (peak wt 9.0 at -5 vx) + 0.5 × SDF Mass Dice` and **averaging raw SDFs across models** outperformed converting to probabilities (`sigmoid(-sdf/T)`).
  - *FP16 Safety*: Training in `bfloat16` then clamping weights to `[-65504, 65504]` + NaN/Inf hooks enabled clean `float16` T4 inference.
- **Fast `Muon` Optimizer (`#7 DECEM`)**: `SingleDeviceMuonWithAuxAdam` (`1.25×` SGD speed) beat both SGD (`0.590`) and AdamW (`0.594`) at **`0.595`** on 3D U-Net segmentation.
- **Train–Test Label Pipeline Parity (`#7`)**: Ensembling a raw-label model with one trained on **organizer-preprocessed labels** hedged label-generation shift.

### 1.2 Topological Post-Processing Toolbox (`+0.02` to `+0.03+` Private LB!)
Because topology losses during training often degraded SurfaceDice (`#2`, `#5`), winners maximized geometric/surface precision in the network and repaired topology post-hoc:
1. **Connectivity-Aware Dust Removal (`#1`, `#5`)**: Drop small components (`< 20k–50k` voxels, 6-conn) **while protecting smaller fragments touching `≥ 3` volume boundaries** (`#5`), worth **`+0.018` Private LB** in `#1` (`0.596 → 0.614`).
2. **2×2×2 Watertight LUT (`#1`)**: Fast 256-entry lookup table plugging 1-voxel 6-connected holes (`0.614 → 0.622` Private!).
3. **Bridge-Safe Hole & Tunnel Filling (`#1`, `#2`, `#4`, `#5`, `#6`)**:
   - Detect holes via **Euler number `χ < 1`** (`#2`) or fast C++ **`betti_matching` H1 barcodes** (`#4`, `#5`, `#6`).
   - Patch holes via PCA/height-map interpolation or adaptive spheres (`#5`: `+0.08` TopoScore CV) **strictly checking that component count does NOT drop** (discard any patch that merges two adjacent sheets!).
4. **Multi-Threshold Recursive Sheet Unmerging (`#2`, `0.631` unsel. Priv)**: When an interpolated component shows poor Dice/coverage (indicating merged parallel sheets), recursively re-segment using higher probability thresholds (`0.2 → 0.5 → 0.6 → 0.7`) to split touching sheets before re-patching.
5. **Probability-Map Diffusion Smoothing (`#3`, `#7`, `#8`)**: 3D Frangi sheetness + GPU Coherence Enhancing Diffusion (CED, `#7`) or 2× Perona-Malik Gradient Anisotropic Diffusion (`#8`) + direction-aware hysteresis bridged thin breaks without thickening sheets.

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Enforcing Minimum Sheet Thickness / Naive Dilation** | Collapsed Private LB to **`0.454`** (`#10`) | #1, #10 | Bridges narrow inter-sheet gaps, catastrophically destroying `VOI_score` and `TopoScore` |
| **3D Transformer / Modern Hybrids (`SwinUNETR`, `MedNeXt`, `UNETR++`, `SegFormer3D`, `ConvNeXtV2`, `U-Mamba`)** | All underperformed `SEResNeXt152 + AttUNet` / `nnU-Net ResEnc` | #5 | High-res local 3D residual convolutions preserve thin multi-sheet boundaries better under 9h compute |
| **Pseudo-Labeling on Unlabeled / External Volumes** | Zero gain or degraded score | #3, #7 | Propagated sheet mergers and missing-surface artifacts |
| **Heavy Training Topology Losses (`Skeaw-BorT`, `MedialSurface`, PD-matching)** | Traded away SurfaceDice or too slow | #2, #5 | Better to maximize SurfaceDice/SDF in network and fix H1 holes via fast bridge-checked post-processing |
| **Chasing 20% Public LB Thresholds & Probability-vs-Logit Fusion** | Public favored lower `0.20–0.26` & prob fusion vs `0.35–0.40` & logit fusion on Private | #1, #2 | Small 20% Public volume subset (~24 volumes) noisy on Betti/VOI metrics |
