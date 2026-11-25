# BYU — Locating Bacterial Flagellar Motors 2025 — Top Gold Solutions & Post-Mortem

> **Competition**: [BYU - Locating Bacterial Flagellar Motors 2025](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025) (Mar 5 – Jun 5, 2025 | 1,136 Teams | 45,278 Submissions | \$65,000 Prize Pool)  
> **Task**: 3D keypoint localization of bacterial flagellar motor centers in noisy cryo-electron tomography (cryo-ET) 3D tomograms  
> **Evaluation Metric**: **$F_2$ Score** (`5·TP / (5·TP + 4·FN + FP)`), matching a predicted `(z, y, x)` point within `<= 1000 Å` Euclidean distance of ground truth (`(-1, -1, -1)` if no motor)  
> **Dataset**: `73.87 GB` (`270,296` JPEG slices across `n = 648` official train tomograms having `0, 1, or multiple` motors) | `~900` Hidden Test tomograms (`28%` Public / `72%` Private, **strictly 0 or 1 motor**, `~50/50` pos/neg, **no `Voxel spacing` metadata at test time!**; `<= 12h` GPU/CPU notebook)

---

## Official Top Gold Leaderboard Standings & Solution Index

Pure **3D volumetric models** (coarse-stride 3D Gaussian/EDT blob regression or spatial classification) plus **raw external CryoET data curation & quantile/Double-Otsu $F_2$ thresholding** dominated the Private leaderboard and survived a severe 2D-YOLO shakeup (Public #2 and #7 dropped `-31` to `-40` places):

| Private Rank | Public Rank | Team | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **#1** | **`Bartley`** (`brendanartley`) | **0.88235** | **0.87656** | Kinetics-pretrained **3D `ResNet200` U-Net** (1 deconv block) on `(128,704,704)` → **8× downsampled 3D Gaussian heatmap**; **3-term SmoothBCE** (main + deep sup + **`4×4` max-pooled loss**); **Z-only SWI (`0.875` overlap, 4× faster)** w/ border downweighting; 8-seed sum + **quantile thresholding (`0.560` Priv / `0.565` Pub)** | [Writeup](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/writeups/bartley-1st-place-3d-u-net-quantile-thresholding) · [GitHub](https://github.com/brendanartley/BYU-competition) |
| 🥈 **#2** | #10 | **`MIC_DKFZ`** (`FabianIsensee` et al.) | 0.86734 | **0.87656** *(tied #1)* | **2,490 tomograms** (648 official + 1,287 raw re-downloaded CryoET + 555 extra across 104 datasets; **213 manually Napari-corrected**, jumping low-compute Priv **`0.83181 → 0.86253`**!); **single from-scratch 3D `nnU-Net ResEnc`** (no ensemble!) predicting **EDT `r=25` blobs** w/ **TopK-20% BCE**, `(128,256,256)` patches, 3,500 ep (875k steps, 8×A100), Blosc2 memmap, all-axis flip TTA + `thr=0.15` | [Writeup](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/writeups/mic-dkfz-2nd-place-solution-3d-nnu-net-blob-regres) · [GitHub](https://github.com/MIC-DKFZ/kaggle_BYU_Locating_Bacterial-Flagellar_Motors_2025_solution) |
| 🥉 **#3** | #9 | **`Mimi`** (`dangnh0611`) | 0.86734 | **0.86607** | Resampled to `16.0 Å` (inferred test mean `13.1 Å`), CVAT-cleaned **188 FN + 55 FP**; **16× downsampled physical Gaussian (`σ=200 Å`)**; 5-model ensemble (`3D ResNeXt50` `0.862` Priv, `3D DenseNet121` `0.857`, `X3D-M+FPN` `0.846`, `2D CoaT-Lite-M` `0.833`, `2D MaxViT-T` `0.825`) → NMS + **3D WBF (L2 dist)** + **55% quantile thresholding** | [Writeup](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/writeups/mimi-3rd-place-solution-3d-2d-unet-with-gaussian-h) · [GitHub](https://github.com/dangnh0611/kaggle_byu) |
| 🏅 **#4** | #5 | **`Daddies`** (`Dieter` & `bloodaxe`) | 0.87563 | **0.86529** | **Split-half val CV (`:W/2`, `W/2:`)** matching 50/50 test prior; Dieter: **decoder-free 3D `ResNet18` 76-class spatial classifier** (75 `stride-32` voxels + 1 no-motor class, `DenseCrossEntropy1D`, constrained `≤1` motor MixUp → **`0.875` Pub solo!**); bloodaxe: **3D-2D-3D `MaxxViT` detector** (FP16-mixed + TensorRT `>200%` speedup); blended via **Winner-Takes-All + Double Otsu** | [Writeup](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/writeups/daddies-4th-place-simple-resnet18-classification) · [Part 2](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/583228) |
| 🏅 **#6** | #4 | **`YUAN High-Tech`** | 0.88010 | **0.85804** | 4× 2.5D `[z-3, z, z+3]` `YOLOv12m` + Z-axis Hamming smoothing filter + 56th-percentile / 2nd-derivative knee auto-thresholding | [Discussion](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion) |
| 🏅 **#9** | #15 | **`Forcewithdonut`** | 0.86075 | **0.85288** | 2-stage coarse-to-fine `YOLOv8l/11l` (Standard + SAHI → rotated zoom-in Stage 2 + midpoint reasoning) | [Discussion](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion) |

---

## 1. Data Forensics & Label Curation Breakthroughs

1. **Massive Unlabeled Motor Noise (`>80%` Flagellum-Less FN) & Manual Relabeling (`+0.031` Priv)**:
   - Over `80%` of flagellum-less motors in raw cryo-ET tomograms were unlabeled in the official/portal annotations.
   - `#2 MIC_DKFZ` manually inspected and corrected **213 tomograms in Napari** (merging 648 official + 1,287 raw re-downloaded CryoET + 555 extra tomograms = `2,490` total across 104 datasets), jumping their low-compute baseline Private LB from **`0.83181 → 0.86253`** (`+0.0307`!).
   - `#3 Mimi` similarly cleaned **188 false negatives and 55 false positives** in CVAT.
2. **External Dataset Corruption Forensics**:
   - Bartley's widely used public `(128, 512, 512)` resized external dataset distorted physical aspect ratios and **erased `Voxel spacing` metadata** — top teams (`#2`, `#3`) re-downloaded raw CryoET portal volumes instead.
   - Signed `int8` vs `uint8` pixel wrapping corrupted naive CryoET portal loaders (`Bilzard` / community catch).
3. **Matching the Hidden Test Prior (`0 or 1` Motor, `~50/50`) in Local CV (`#4 Daddies`)**:
   - Official train had multi-motor tomograms while hidden test had strictly `0 or 1` motor with `~50/50` balance. `#4` created a **split-half validation scheme (`:W/2` and `W/2:`)** so local validation mirrored the `~50/50` single-or-zero motor test prior, and constrained MixUp so mixed crops never exceeded `1` motor.

---

## 2. Winning 3D Keypoint Detection Paradigms

1. **Coarse-Stride 3D Heatmap / EDT Blob Regression (`#1`, `#2`, `#3`)**:
   - Rather than reconstructing full-resolution segmentation masks (slow and fragile under tiny point labels), top solutions regressed **8× (`#1`) or 16× (`#3`) downsampled 3D Gaussian heatmaps**, or **EDT (`r=25`) spherical blobs (`#2`)** trained with **TopK-20% BCE** (EDT `r=25` beat Gaussian `r=15`: `0.8489` vs `0.8451`).
   - `#1 Bartley` added a **3-term SmoothBCE loss** combining main heatmap, deep supervision, and a **`4×4` max-pooled heatmap loss** that explicitly forgave small spatial offset errors within the `1000 Å` metric tolerance.
2. **Decoder-Free Spatial Classification (`#4 Dieter`, `0.875` Pub Solo!)**:
   - Replacing the entire 3D U-Net decoder with a **3D `ResNet18` 76-class classifier** (`75` stride-32 spatial voxels + `1` "no motor" class, trained with `DenseCrossEntropy1D`) directly aligned with the `0-or-1` motor test structure and achieved `0.875` Public LB from a single lightweight architecture.
3. **Fast Volumetric Inference Engineering (`#1`, `#2`, `#4`)**:
   - **Z-only sliding window (`0.875` Z-overlap, full XY crop `(128, 704, 704)`)** ran **4× faster** than 3D XYZ sliding windows while avoiding XY tile seams (`#1`), paired with border downweighting.
   - **Blosc2 memory-mapped loading** eliminated I/O bottlenecks across 875k steps (`#2`).
   - **FP16-mixed + TensorRT** delivered `>200%` speedup (`#4 bloodaxe`); critically, **BF16-trained weights broke when exported to FP16 TensorRT** (always train FP16-mixed or clamp weights prior to FP16 TRT export).
4. **Distribution-Aware Thresholding for $F_2$ (`#1`, `#3`, `#4`, `#6`)**:
   - Because $F_2$ weights recall `4×` heavier than precision and the hidden test prior was `~50%` positive, fixed probability thresholds drifted across ensembles/shifts.
   - Top teams unanimously switched to **test-batch quantile thresholding** (`55%–56.5%` positive rate, `#1`, `#3`, `#6`) or **Double Otsu** adaptive bimodal thresholding (`#4`) after 3D NMS / L2-distance **3D Weighted Boxes Fusion (WBF)** (`#3`).

---

## 3. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Pure 2D/2.5D YOLO Without 3D Volumetric Context** | Severe Private shakeup (Public #2/#7 dropped `-31` to `-40` ranks) | #1–#4, LB | Single/3-slice views confuse membrane/ribosome artifacts with 3D cylindrical motor complexes |
| **Using Resized Public External `(128,512,512)` Data Without Raw Voxel Spacing** | Distorted physical `1000 Å` scale & motor geometry | #2, #3 | Aspect distortion breaks fixed-radius kernels; re-download raw CryoET portal volumes |
| **MSE / L1 / Soft-Dice / Focal / Tversky Losses for Sparse 3D Keypoints** | Consistently inferior to TopK-20% BCE & 3-term SmoothBCE | #1, #2 | Extreme foreground sparsity causes gradient collapse or poorly calibrated peak probabilities |
| **`nnDetection` / `nnLandmark` & False-Positive Patch Oversampling** | Underperformed 3D `nnU-Net` EDT blob regression; oversampling degraded calibration | #2 | Over-representing hard negatives distorts the `F_2` score's precision-recall balance |
| **Stride-32 Heatmaps (for Regression) & Exporting BF16 Weights to FP16 TensorRT** | Stride-32 lost localization precision (`#3`); BF16→FP16 TRT silently degraded outputs (`#4`) | #3, #4 | Stride-32 is too coarse for continuous heatmap regression without classification framing; BF16 dynamic range overflows FP16 TensorRT kernels |
