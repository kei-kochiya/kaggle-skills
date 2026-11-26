# CZII — CryoET Object Identification: Top-10 Solutions & Small-N 3D Detection Post-Mortem

> **Competition**: [CZII — CryoET Object Identification](https://www.kaggle.com/competitions/czii-cryo-et-object-identification) (Nov 2024 – Feb 2025, 931 Teams, \$75,000 Prize Pool; hosted by CZ Imaging Institute)  
> **Track**: 3D Biomedical Computer Vision, Extreme Small-N Detection & Recall-Heavy Localization  
> **Dataset & Compute**: **Only 7 labeled 3D training tomograms** (`~184×630×630` at `10.012 Å/voxel`; train has 4 reconstruction variants `denoised/wbp/ctfdeconvolved/isonetcorrected`, test has **only `denoised.zarr`** across `~500` volumes inside `<= 12h` on `2× T4`)  
> **Evaluation Metric**: Class-weighted micro-$F_\beta$ with **$\beta = 4$** (recall weighted **`16×`** higher than precision) at tolerance `0.5 × particle_radius` across 5 scored classes (`apo-ferritin`, `ribosome`, `virus-like-particle` weight `1`; `beta-galactosidase`, `thyroglobulin` weight `2`; `beta-amylase` unscored weight `0`)

---

## 1. Official Top-10 Leaderboard & Architectural Diversity

Public-to-Private shakeup among the Top 10 was remarkably mild (`0.7726–0.7876`), with **1st place (`Daddies`)** pulling away by `+0.00378` by fusing two genuinely orthogonal 3D paradigms (anchor-free stride-2 Object Detection + penultimate-layer "Partly U-Net" Segmentation) via **rank/quantile histogram matching**.

| Private Rank | Team | Private LB | Public LB (Δ) | Paradigm & Core Innovations | Writeup / Code |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **1st** | **Daddies** (`bloodaxe`, `christofhenkel`) | **0.78759** | 0.79461 (0) | **OD (`SegResNet`/`DynUNet` stride-2 Varifocal+OKS + TensorRT 2×T4)** fused with **"Partly U-Net" (`FlexibleUnet` penultimate `48³` single-pixel `256:1` weighted CE + `beta-amylase` aux class + Beta-MixUp)** via **per-class rank/quantile matching** | [OD](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/561440) · [Seg](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/writeups/daddies-1st-place-solution-segmentation-with-partl) · [GH-OD](https://github.com/BloodAxe/Kaggle-2024-CryoET) · [GH-Seg](https://github.com/ChristofHenkel/kaggle-cryoet-1st-place-segmentation) |
| 🥈 **2nd** | **LuoZiqian&Lion** | **0.78381** | 0.79391 (0) | 10 ultra-compact 3D seg CNNs (`873K–14.2M`: `UNet3D`, `DenseVNet`, `VoxResNet`, `VoxHRNet`, `SegResNet`); **`InstanceNorm3d + PReLU`** stability fix; `0.4–0.5×` shrunk spheres; low `0.15` prob thr + `cc3d` | [Writeup](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/writeups/luoziqian-lion-2nd-place-solution) · [GH](https://github.com/GWwangshuo/Kaggle-2024-CryoET-Pub) |
| 🥉 **3rd** | **ONCE UPON A MOON** | **0.78351** | 0.78327 (+3) | Simple `smp_3d` `Unet(resnet101)` (4/7 LOTO folds, 7-class CE, EMA); **trained on all 4 reconstruction types** (`denoised/wbp/ctf/isonet`) as domain aug; `64×128² → 64×256²` infer (`+0.001`) | [Writeup](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/writeups/once-upon-a-moon-3rd-place-solution) |
| 🏅 **4th** | **yu4u & tattaka** | **0.78306** | 0.78837 (-1) | **2.5D U-Nets** (`ConvNeXt-Nano` + depth pooling + **pixel shuffle** & `ResNetRS-50` + JPU + SESC) predicting 3D Gaussian heatmaps (`balanced pos/neg MSE`); **`+1.0 / -0.5` pixel-center offset**; `maxpool3d(7)` + TensorRT | [Writeup](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/writeups/yu4u-tattaka-4th-place-solution-source-codes-submi) · [GH-yu4u](https://github.com/yu4u/kaggle-czii-4th) · [GH-tattaka](https://github.com/tattaka/czii-cryo-et-object-identification-public) |
| 🏅 **5th** | **Youssef Ouertani** et al. | **0.78252** | 0.78019 (+5) | **Ultra-compact `1.44 MB` DeepFinder-style 3D CNN** (`28,32,36` ch) trained on **all 7 volumes × 4 seeds**; `(5,99)` pct norm; `log2(r)*0.8` masks; `128³` 6× TTA | [Writeup](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/writeups/youssef-ouertani-5th-place-solution) |
| 🏅 **6th** | **tomoon33** | **0.78022** | 0.78512 (-2) | 10 models (2.5D UNets + **`polnet` simulated-pretrained** 3D UNet/SegResNet); **`FocalTversky++` (`+0.02` CV/LB vs saturated Dice/Tversky)**; TensorRT `2×T4`, discard outer 8% borders | [Writeup](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/writeups/tomoon33-6th-place-solution) · [GH](https://github.com/uchiyama33/czii-6th-place) |
| 🏅 **7th** | **kobakos** | **0.77708** | 0.78351 (-2) | Sim-pretrained `ResNet50d`/`EffNetV2-M` UNet + `DeepLabV3+`; high-recall **weighted BCE (`pos_weight=20–96`)**, **logit-space averaging & negative logit thresholds** + 3D WBF; 3×3 LB linear probe | [Writeup](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/writeups/kobakos-7th-place-solution-3d-unet-with-gaussian-h) |
| 🏅 **8th** | **I Cryo Everyteim** | **0.77612** | 0.78079 (+1) | **4 MONAI 3D U-Net Model Soups** + 6 denoised synthetic tomos (`+0.01` LB) + **large `(160,384,384)` infer patches (`+0.01` LB vs `128³`)** + GPU `cucim/cupy` distance-transform watershed | [Writeup](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/writeups/i-cryo-everyteim-8th-place-solution-for-the-czii-c) · [GH](https://github.com/IAmPara0x/czii-8th-solution) |
| 🏅 **9th** | **Avengers** (`wadakoki`) | **0.77274** | 0.78245 (-1) | Custom **3D ConvNeXt (`2×2` stem, `3×3` DWConv)**; **shrunk mask radii (`r/2` small, `r/3` large: `+0.02–0.04` LB)**; `cc3d` + DBSCAN | [Writeup](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/writeups/avengers-9th-place-solution) |
| 🏅 **10th** | **Josef Slavicek** | **0.77263** | 0.77817 (+1) | 9 Optuna MONAI `3DUNet`s (8 sim-pretrained), logit averaging + **greedy KL-divergence template deconvolution** to split merged blobs | [Writeup](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/writeups/josef-slavicek-10th-place-solution) |

---

## 2. Surviving Extreme Small-N (`7` Labeled Tomograms)

1. **Leave-One-Tomogram-Out (LOTO) 7-Fold CV & Validation Smoothing**:
   - Virtually all gold medalists (#1-Seg, #2, #3, #8, #10) locked **7-fold LOTO CV** (1st-OD used 5-fold and applied D4 flips/rotations to validation volumes to smooth noisy $F_4$ threshold curves; #5 trained all-7-volume 4-seed full fits to jump `+5` on Private LB).
2. **Keep the Unscored Distractor Class as an Auxiliary Output (#1, #2, #3)**:
   - Even though `beta-amylase` had metric weight `0`, supervising it as a 7th class explicitly taught networks not to confuse it with high-weight (`2×`) `beta-galactosidase`.
3. **Multi-Reconstruction Training Augmentation (#3, #8)**:
   - Although test sets contain only `denoised.zarr`, training across all 4 provided reconstruction types (`denoised`, `wbp`, `ctfdeconvolved`, `isonetcorrected`) acted as physics-grounded denoising/contrast augmentation (`0.78327 → 0.78351` zero-shakeup bronze in #3!).
4. **Compact Architectures + `InstanceNorm3d`/`PReLU` Beat Heavy 3D Models**:
   - With only 7 volumes, compact `1M–14M` 3D CNNs (#2 `873K–14.2M`; #5 `1.44 MB` CNN) and pretrained **2.5D U-Nets** (#4, #6: ImageNet 2D encoder + **depth pooling**, preserving 2D filters!) dominated heavy 3D backbones and 3D Transformers. Switching `BatchNorm3d+ReLU` to **`InstanceNorm3d + PReLU`** eliminated convergence collapse across `VoxResNet`/`VoxHRNet`/`DenseVNet` (#2).
5. **External Simulated Pretraining — Quality-Sensitive**:
   - Helped strongly when shape/noise matched real tomograms (`polnet` in #6 lifted 3D UNet to 2.5D parity; 6 Gaussian-denoised synthetic tomos added `+0.01` LB in #8; sim-pretraining worked in #7, #10), whereas naive host simulated data failed to improve #1, #2, #3, #9.

---

## 3. Target Formulation, Cross-Paradigm Calibration & High-Recall $F_4$ Serving

### 3.1 Four Gold-Winning 3D Localization Formulations
- **Anchor-Free Stride-2 3D Point Detection (#1-OD `bloodaxe`)**: Predicts classmap + 3D center offset at `D/2, H/2, W/2` (`50%` faster than stride 1, only `~0.002` LB cost) trained with **Varifocal + Object Keypoint Similarity (`exp(-mse / 2r²)`)** point-IoU assignment. Offset regression naturally absorbs grid-origin shifts.
- **Penultimate-Layer "Partly U-Net" (#1-Seg `christofhenkel`)**: Truncated `FlexibleUnet` (`resnet34` / `effnet-b3`) at the stride-2 penultimate decoder map (`48³` from `96³` input) with **single-pixel point labels** (`256:1` positive-to-background weighted CE) + GPU **Beta-MixUp** (doubled training epochs without overfitting; `0.778` standalone seg ensemble in `<2h`).
- **2.5D Gaussian Heatmap + Pixel Shuffle (#4 `yu4u & tattaka`)**: `ConvNeXt-Nano` + depth pooling + **3D pixel shuffle from stride 4** (avoiding heavy full-res deconvs and checkerboard artifacts) with balanced pos/neg MSE and `+1.0 / -0.5` sub-voxel coordinate alignment.
- **Shrunk Sphere Masks (`0.33×–0.5×`) (#2, #3, #5, #8, #9)**: Shrinking target masks below the `0.5×r` evaluation radius prevented touching particles from merging into single blobs (`+0.02–0.04` LB in #9).

### 3.2 Blending Calibration & $F_4$ Post-Processing
- **Rank / Quantile Histogram Matching (#1 `Daddies`)**: OD classmaps and Seg probability maps sit on incompatible calibration scales. Sorting all voxel scores per class and replacing map $B$'s values with map $A$'s quantiles aligned both paradigms cleanly before classmap averaging and peak NMS.
- **Logit-Space Averaging vs. Saturated Dice/Tversky (#6, #7, #8, #10)**: Standard Dice/Tversky saturated outputs to binary `0/1` and hurt map averaging and $F_4$ thresholding; replacing with **`FocalTversky++` (`+0.02` CV/LB, #6)** or **weighted CE/BCE (`pos_weight=20–96`) + logit-space averaging (#7)** preserved smooth recall tails (`thr = 0.15` prob in #2; negative logits `-0.2` in #7).
- **Peak Splitting & Throughput (`~500` volumes `<12h`)**:
  - `Gaussian blur → maxpool3d → 3D point-IoU NMS / WBF` (#1, #4, #7) was orders of magnitude faster than `cc3d` or GPU watershed (`cucim`, #8) / KL template deconvolution (#10).
  - **Pre-compiled TensorRT on `2× T4`** (`200%` speedup vs `torch.jit`, #1, #4, #6), **Model Soups** (#7, #8), and **larger inference patches (`160×384×384`, `+0.01` LB over `128³`, #8)** with Gaussian/border-discard blending unlocked massive accuracy-per-hour gains.

---

## 4. Empirical Negative Results & What Failed

| Failed Hypothesis | Observed Result | Root Cause |
| :--- | :--- | :--- |
| **Two-Stage Crop-and-Refine Cascades (#2, #4, #5, #6)** | Improved CV (#4) or failed outright on LB | On 7 training tomograms, Stage-2 classifiers overfit Stage-1 crop distributions and hurt $F_4$ recall on unseen test tomograms. |
| **Heavy 3D Backbones & 3D Transformers (#1-OD, #2, #5, #8)** | Overfit badly vs. compact `1M–14M` CNNs | Severe parameter-to-volume mismatch on 7 training studies. |
| **Standard Dice / Tversky Loss Saturation (#1-Seg, #6)** | Degraded map ensembling & $F_4$ recall (`-0.02` vs `FocalTversky++`) | Binary `0/1` confidence saturation destroys continuous ranking needed for low-threshold ($\beta=4$) recall tuning. |
| **Strided 3D Conv Replacing Depth Pooling in 2.5D Encoders (#4)** | Degraded accuracy | Corrupted pretrained ImageNet 2D spatial filter representations. |
| **2D-Only U-Nets (#8)** | Only `0.636` LB (vs. `0.776+` 3D) | Lost Z-axis spherical context critical for distinguishing macromolecules from noise. |
| **MixUp/CutMix on Anchor-Free OD (#1-OD) & Some 3D UNets (#8)** | Helped CV, hurt LB (while essential for #1-Seg, #3, #7, #10) | Threshold-sensitive point regression heads reacted poorly to blended background intensity statistics. |
| **Multi-Scale Deep Supervision & Extra Box/Offset Heads (#1, #7)** | Minimal/zero gain over stride-2 penultimate head | Fixed physical particle radii make bounding-box regression redundant once centers are localized to `<0.5r`. |

---

## 5. Universal Takeaways Promoted / Reinforced

- **Rank/Quantile Histogram Matching for Cross-Paradigm Dense Map Ensembling**: When blending voxel/pixel score maps from incompatible heads (e.g., Varifocal OD + Weighted-CE Seg), map both tensors to matched per-class empirical quantiles before weighted averaging.
- **Penultimate Stride-2 Heads & Shrunk Target Masks for 3D Keypoint/Blob Detection**: Truncating U-Nets at stride-2 (`50%` faster, sharper center localization) or shrinking target sphere masks to `0.33×–0.5×` prevents blob merging and aligns cleanly with radius-tolerance metrics.
- **Avoid Probability-Saturating Losses Under Recall-Heavy $F_\beta$ ($\beta \ge 2$)**: Prefer weighted CE/BCE, balanced MSE, or `FocalTversky++` + **logit-space averaging** over raw Dice/Tversky so low-threshold recall calibration remains smooth.
- **Keep Unscored Distractor Classes & Multi-Reconstruction Variants in Training**: Supervising unscored look-alike classes reduces hard false positives, and training on multiple non-test reconstruction filters acts as free physics-grounded augmentation.
