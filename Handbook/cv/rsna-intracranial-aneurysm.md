# RSNA Intracranial Aneurysm Detection — Top Gold Solutions & Post-Mortem

> **Competition**: [RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection) (Jul 28 – Oct 15, 2025 | 1,147 Teams | 16,298 Submissions | \$50,000 Prize Pool)  
> **Task**: Multi-modality 3D intracranial aneurysm detection across **CTA, MRA, T1 post-contrast, and T2 MRI** — `Aneurysm Present` + **13 Circle-of-Willis anatomical locations**  
> **Evaluation Metric**: Weighted Mean Columnwise ROC-AUC: $\frac{1}{2}\left(\text{AUC}_{\text{AP}} + \frac{1}{13}\sum_{i=0}^{12}\text{AUC}_i\right)$  
> **Dataset**: `311.25 GB` (`1,002,760` DICOM/NIfTI files) with `train_localizers.csv` xy-points + partial 13-class NIfTI vessel segmentations | `~2,500` Hidden Test series served one-by-one (`~32%` Public / `~68%` Private; `<= 12h` 2×T4 notebook)

---

## Official Top Gold Leaderboard Standings & Solution Index

In contrast to RSNA Knee (where 2D/2.5D MIL won across multi-plane thick-slice studies), **anatomical 3D vessel-ROI extraction + 3D `nnU-Net` vessel-segmentation pretraining / multi-tasking** decisively swept the podium and generalized cleanly to Private LB (`1st`: `0.883 → 0.879`; while pure-2.5D 5th fell `0.893 → 0.847`):

| Private Rank | Public Rank | Team | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | #3 | **`tomoon33`** | 0.88285 | **0.87912** | **Coarse-to-Fine 3D `nnU-Net` Vessel Seg** (coarse `1mm` → fine balanced + recall `0.8×0.45×0.44mm`) → **vessel-seg pretrained 3D `nnU-Net`** (`128×256×256`, **`+0.108` CV!**) + **13-vessel masked pooling → Location-Aware Transformer** + aux `r=5` sphere head (`+0.026` CV) + 2× L/R TTA | [Writeup](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution) · [GitHub](https://github.com/uchiyama33/rsna2025_1st_place) · [NB](https://www.kaggle.com/code/tomoon33/rsna2025-submission-1st-place) |
| 🥈 **#2** | **#1** | **`BraveCoWCoW`** (TopCoW team) | **0.90035** | **0.86732** | **Fast 9-slice 2D tri-axial `nnU-Net` ROI** → **`224³` 3D Multi-Task `nnU-Net`** (curated **26-class** TopCoW/TopBrain vessel+aneurysm masks + cross-attn pooling + 4-class modality head; **seg decoder dropped at test!**) + **8× L/R-swap TTA** | [Writeup](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution) · [GitHub](https://github.com/PengchengShi1220/RSNA2025_Intracranial-Aneurysm-Detection) |
| 🥉 **#3** | #6 | **`BTYND`** | 0.86715 | **0.85065** | `YOLOv8` sag/cor MIP ROI (`90–120mm³`) → **11 stride-reduced 3D `ResNet-18`s** (enlarged `25³–48³` voxel feature maps, Top-N mean) + **`EffNetV2-S` image-based voxel-spacing predictor (`0.84 → 0.85` Priv!)** | [Writeup](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution) · [GitHub](https://github.com/tamotamo17/RSNA2025-3rd-place-solution) |
| 🏅 **#4** | #5 | **Harshit Sheoran** | 0.87155 | **0.84750** | 8-day speedrun: `DINOv3` 48-slice ROI regressor → **2.5D `(-2,0,+2)` `CoaT-Lite-M` + `MaxViT`** (`384px`) + **soft pseudo-distillation** on ~500k unlabeled slices from positive patients (`0.86 → 0.89` CV) + max agg | [Writeup](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution) · [NB](https://www.kaggle.com/code/harshitsheoran/rsna-aneurysm-detection-demo-submission) |
| 🏅 **#5** | #2 | **`more CV challenge pls`** (`HoangHuyen`) | 0.89306 | **0.84746** | `YOLOv5n` brain crop (`+0.03–0.05`) + manual `YOLOv11x` (`1280px`) → 2.5D `ViT-L` + `EVA-L` + multi-task `MiT-B4 FPN` w/ L/R-swap HFlip (`+0.01`) + OpenNeuro TOF-MRA PL | [Writeup](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution) · [GitHub](https://github.com/hoanghuyen797/RSNA-Intracranial-Aneurysm-Detection) |
| 🏅 **#6** | #9 | **Ian, Theo & Bartley** | 0.84545 | **0.84007** | 3D skull crop + train-only vessel seg for **hard-negative slice mining** → 2.5D `CoAtNet-2` + `MaxViT-B` (`rmlp` preserving L/R) + custom sequence aggregator (`0.87 → 0.88 → 0.895` CV) + adaptive frame subsampling | [Writeup](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/6th-place-solution) |
| 🏅 **#7** | #7 | **`MIC-DKFZ`** (`Fabian Isensee` et al.) | 0.84675 | **0.83173** | **Single 3D `nnU-Net ResEnc` (`96×160×128`, no ensemble, no TTA!)**: `[200,160,160]mm` superior-central crop → median `(0.70,0.47,0.47)mm` GPU resample → **14-ch EDT blob regression (`r=65`)** w/ TopK-20% BCE | [Writeup](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution) · [GitHub](https://github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution) |
| 🏅 **#9** | #12 | **Vibes & Genius Trade-Off** | 0.83884 | **0.82300** | 3-stream (`YOLOv11m` + `YOLO-EffNetV2-S` + 2D-to-3D `CenterNet` w/ aux 2D vessel seg) → **GBDT (`LGBM+XGB+CB`) 44-feat meta-classifier** capturing cross-location correlations | [Writeup](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/9th-place-solution) · [GitHub](https://github.com/tom99763/9th-place-solution-RSNA-IAD) |

---

## 1. Key Universal 3D Medical Detection Breakthroughs

1. **Anatomical Segmentation Pretraining + Region-Masked Pooling (`#1` Controlled Ablations)**:
   - On identical `96×192×192` 3D `nnU-Net` backbones: **removing vessel-segmentation pretraining crashed CV from `0.902 → 0.794` (`-0.108` AUC!)**; removing the auxiliary `r=5` aneurysm sphere detection head dropped CV `0.902 → 0.876`; removing the recall-focused `SkeletonRecall(w=3)` vessel model dropped CV `0.902 → 0.891`.
   - Pooling 3D decoder features **inside each of the 13 predicted anatomical vessel masks** and feeding those 13 tokens + encoder GAP into a **Location-Aware Transformer** (`#1`), or using **cross-attention pooling** over a 26-class multi-task encoder while **dropping the 3D seg decoder at test time** (`#2`), solved both fine localization and the 12h T4 budget.
2. **Stride-Reduced High-Resolution 3D Feature Maps (`#3 BTYND`)**:
   - Replacing selected stride-2 convolutions in 3D `ResNet-18` with stride-1 expanded final spatial grids from `8³ → 25³–48³` (single-fold LB jumped **`0.77 → 0.81`**) and decoded per-class Top-N spatial means cleanly without a heavy decoder.
3. **Resilience to Missing Test DICOM Metadata (`#2`, `#3`)**:
   - Hidden test DICOMs frequently stripped voxel-spacing or orientation tags. Training a fast **2.5D `EffNetV2-S` voxel-spacing regressor** (`MAE 0.015/0.020/0.071 mm`) jumped Private LB **`0.84 → 0.85`** (`#3`), while `#2` trained a T2 orientation classifier and removed silent `0.5` try-except fallbacks.
4. **Universal Per-Volume Z-Score & Label-Swapping L/R Mirroring (`#1`, `#2`, `#5`, `#6`, `#7`)**:
   - **Per-volume / global z-score normalization** worked uniformly across CTA, MRA, T1, and T2 without fragile HU windowing, and keeping native anisotropic median spacing (`0.70 × 0.47 × 0.47 mm`) dramatically beat `1.0³ mm` isotropic resampling (`#7`).
   - **Left/Right flipping MUST swap L/R location labels and masks** (boosted `#2` `0.857 → 0.862` Priv via 8× TTA, `#5` `+0.01`, `#6` `+0.01`).
5. **Runtime Engineering for 2,500 Series in 12 Hours**: Fast 9-slice 2D tri-axial ROI (`#2`), `YOLOv8` MIP ROI (`#3`), dropping the 3D seg decoder at test time (`#2`), GPU PyTorch resampling (`#7`), and adaptive `1/2`–`1/4` slice stride for long `>64–128` slice stacks covered by 3-slice 2.5D windows (`#6`).

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **End-to-End 3D Classifier Without Anatomical ROI / Vessel Pretraining** | Failed on 13 locations (`-0.108` CV without vessel pretrain, `#1`); `0.79` LB (`#6`) | #1, #6 | Tiny 2–5mm aneurysms vanish inside full uncropped head volumes |
| **`1.0 × 1.0 × 1.0 mm` Isotropic Resampling / Z-Resampling for 2.5D YOLO** | Substantially hurt 3D `nnU-Net` (`#7`); worsened YOLO vs raw triplets (`-0.02`, `#9`) | #7, #9 | Downsampling high-res in-plane `0.45mm` vessels blurs small aneurysms |
| **Naively Mirroring L/R Without Swapping L/R Labels & Masks** | Corrupts 10 lateralized anatomical classes | #1, #2, #5, #6 | Left vs Right ICA/MCA/ACA/PCom flip sides under horizontal mirror |
| **Hard-Filtering YOLO Detections With 3D Vessel Seg / Vessel-Crop 2D Classification** | Killed recall (`#9`); poor 2D accuracy (`#6`) | #6, #9 | Imperfect vessel masks drop real aneurysms; use vessel masks as soft pooling/hard-negative sampling instead |
| **Public-LB Chasing (`nnDetection` / 50-Series Val)** | `nnDetection` beat `nnU-Net` on Pub but lost on Priv (`#7`); `#5` dropped `0.893 → 0.847` | #5, #7 | 32% Public split differs in modality/institution mix from 68% Private |
