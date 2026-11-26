# HuBMAP — Hacking the Human Vasculature 🏆

> **Track:** Computer Vision — 2D Instance Segmentation (Research Code) · **Scale:** 1,021 teams · \$50,000 · Closed Aug 1, 2023  
> **Core Lesson:** In 2D instance segmentation with tiered annotation quality (expert vs. noisy/sparse + unlabeled domains), **WSI/spatial-grouped CV strictly on expert labels**, **2-stage noisy-pretrain → clean-finetune** (or balanced batch mixing), **Box-First WBF → high-res Mask R-CNN/U-Net decoding**, **high-resolution `1400–2048px` backbones + multi-WSI stain transfer**, and **pixel-voting mask fusion** defeated Public-LB mask-dilation traps (`#1` jumped **+51 ranks** from `0.317` Public to **`0.589` Private**!).

---

## 1. Competition DNA & Shakeup Anatomy

- **Task:** Segment `blood_vessel` instances on `512×512` PAS-stained kidney histology tiles (`7,038` files, `4.37 GB`), ignoring `glomerulus` and `unsure` (`glomerulus` test polygons and `tile_meta.csv` coordinates `(source_wsi, i, j)` provided at inference).
- **Metric:** **OpenImages Instance Segmentation AP at `IoU = 0.6`** on `blood_vessel` only (`0 {conf} {RLE}`).
- **Tiered Label & Split Structure (Root Cause of Massive Shakeup):**
  - **Dataset 1 (Expert-reviewed):** `422` train tiles (WSI 1–2) + **all test tiles** (~650 tiles: Public = WSI 3–4 [28%]; Private = **unseen WSI 5** [72%]). Distinguishes `blood_vessel` vs `unsure`.
  - **Dataset 2 (Sparse/non-expert):** `1,211` train tiles (WSI 1–4 — **overlaps Public WSIs 3–4, zero tiles from Private WSI 5!**), merges `sure + unsure`, rougher/missing polygons (`~4.5%` duplicate GT annotations).
  - **Dataset 3 (Unannotated):** Tiles from **9 extra WSIs** for semi-supervised PL & stain style transfer.

| Private # | Shake | Team | Private LB | Public LB (Rank) | Core Breakthroughs |
| :---: | :---: | :--- | :---: | :---: | :--- |
| 🥇 **1st** | **+51** | **tascj** | **0.589** | 0.317 | **Box-First WBF (`3× RTMDet-x` + `YOLOX-x` w/ aux mask head) → `1440px` Mask R-CNN** + fixed-LR EMA + `3 DS1:5 DS2` batches + `'i'`-coord CV |
| 🥈 **2nd** | **+13** | **QL** | **0.588** | 0.562 (#15) | `128px` neighbor-tile WSI context padding + `staintools` 9-WSI style transfer + DS2→DS1 Cascade Mask R-CNN (`Swin/CoaT/ConvNeXt`) + pixel-vote filter |
| 🥉 **3rd** | `-2` | **Nischay** | **0.583** | **0.619 (#1)** | 2-stage DS2→DS1 (`+4–6%` LB) + `1400–2048px` **`ViT-Adapter-L` (`0.589` Priv solo!)** + `CBNetV2` + `DetectoRS` (SGD > AdamW, `2–4×` mask loss) |
| 🏅 **4th** | **+16** | **[SDSRV.AI] GoN** | **0.573** | 0.554 (#20) | DS2→DS1 `Cascade/Mask R-CNN/HTC` + `YOLOv6m` (high recall `55.1` MAR) + **`score_bbox * mean(mask>0.5)`** calibration |
| 🏅 **5th** | **+8** | **zw** | **0.570** | 0.567 (#13) | DS1-only **`HTC-db-b` (`512–1536` multi-scale + SWA)** + **Weighted Cluster-NMS (DIoU)** |
| 🏅 **7th** | `-1` | **yu4u** | **0.562** | 0.575 (#6) | 5-fold DS1 → PL on DS2+DS3 + Swin `Mask R-CNN` w/ HTC RoI + **RPN & RoI head ensembling** |
| 🏅 **8th** | **+50** | **ria** | **0.560** | 0.391 | Single **`YOLOv8x-seg`** (`512→768`, `mask_ratio=1`, `mixup+copy_paste+mosaic`) with strict `mask_thr=0.5` |
| 🏅 **9th** | `-6` | **D.Imanishi** | **0.560** | 0.580 (#3) | DS1-corrected DS2 boxes + 5 YOLO detectors (`160` preds WBF) → cropped `EffNetB1/B2-UNet` |

---

## 2. Data Forensics, WSI Stitching & Semi-Supervised Recipes

1. **Exploiting Tiered Labels (`DS1` Clean vs `DS2` Noisy vs `DS3` Unlabeled):**
   - **2-Stage Pretrain → Clean Fine-Tune (`#2`, `#3`, `#4`, `#12`):** Pretrain `10–30` epochs on noisy `DS2` (higher LR, lighter aug), then fine-tune `15–25` epochs on clean `DS1` (heavy aug, higher res, low LR / SWA). Lifted `#3` by **`+2–3%` CV / `+4–6%` LB** and eliminated the dilation gap.
   - **Balanced Batch Mixing (`#1`, `#14`):** `#1` enforced `3 DS1 + 5 DS2` per batch-8 step so noisy tiles never overwhelmed expert boundaries.
   - **DS1-Guided Correction & Multi-Round PL (`#7`, `#9`, `#14`, `#19`):** `#9` replaced noisy `DS2` boxes with DS1-model predictions (`IoU > 0.4`); `#7` and `#14` pseudo-labeled `DS2 + DS3` (`+0.04` mask mAP50 on WSI 1–2); `#14` even ran **test-time PL** inside the 9h notebook (`+0.005` Pub & Priv).
2. **Neighbor-Tile WSI Context Padding (`#2`):** Used `(source_wsi, i, j)` in `tile_meta.csv` to stitch **`128px` of neighboring context** around every `512×512` tile (splitting WSIs into Left/Right halves so stitched borders never leaked across CV folds!), then center-cropped predictions at inference.
3. **Cross-WSI Stain Style Transfer (`#2`):** Used `staintools` to transfer training tiles into the stain distributions of the **9 unlabeled `DS3` WSIs** (`p=1.0` for same-WSI tiles, `p=0.5` otherwise) to generalize directly to unseen Private WSI 5.

---

## 3. 2D Instance Segmentation Architectures & Training Recipes

- **2D Instance Detectors Crushed Semantic Segmentation + Watershed:**
  - **Decoupled Detect → High-Res Mask Decode (`#1`, `#9`):** `#1` fused boxes from `3× RTMDet-x` (`768px`, adding an auxiliary mask head specifically to boost bbox mAP `0.424 → 0.432` and segm mAP60 `0.688`) + `YOLOX-x` via WBF, then ran **Mask R-CNN's mask head once at `1440×1440`** on fused boxes (`0.565+` Priv single fold w/o TTA!). `#9` fused 5 YOLO detectors via WBF into a cropped `EffNet-UNet`.
  - **High-Res 2-Stage & Transformer Detectors (`#2`, `#3`, `#4`, `#5`, `#7`):** **`ViT-Adapter-L`** (`1400–1600px`, `2×` mask loss weight → **`0.589` Priv solo!** in `#3`), `DetectoRS` (`2048px`, `4×` mask loss), `CBNetV2-Base`, `HTC-db-b` (`512–1536` multi-scale), `Cascade Mask R-CNN` (`Swin/CoaT/ConvNeXt`), paired with high-recall 1-stage `YOLOv6m`/`YOLOv8x-seg` (`#4`, `#8`).
- **Training Discipline:**
  - **Fixed-LR EMA (`#1`):** Highlighted by `tascj` as the #1 training trick (`torch._foreach_lerp_` across multiple EMA momentums).
  - **Heavy Geometric Aug:** Full `[-180°, 180°]` rotation with exact polygon→bbox recomputation (`+0.010` bbox mAP in `#1`); `mosaic + mixup + copy_paste` critical for YOLO-Seg (`#8`, `#14`), whereas heavy copy-paste hurt `HTC` (`#5`).

---

## 4. Instance-Level Ensembling & Post-Processing

1. **Box-First WBF + Pixel-Voting Mask Averaging:** Dominant ensemble protocol (`#1`, `#3`, `#4`, `#9`, `#12`, `#14`). `YYama` even inserted **pure object detectors (`DINO`, `DiffusionDet`, `ViTDet`) into WBF** strictly to improve box confidence calibration! `#5` replaced NMS with **Weighted Cluster-NMS (DIoU)**; `#7` ensembled at **RPN + RoI head level**.
2. **Confidence Calibration & Filtering:**
   - **Mask-Weighted Confidence (`#4`):** `score = score_bbox * mean(mask[mask > 0.5])`.
   - **Pixel & Instance Vote Quantile Pruning (`#2`):** Dropped masks below the `q=0.05` pixel-vote quantile or `q=0.075` `IoU=0.65` match-count quantile.
   - **Small-Area & Glomerulus Suppression:** Dropped tiny masks `< 64–100 px` (`#2`, `#4`, `#7`) and any mask overlapping test `glomerulus` polygons by `> 60%` (`#2`).

---

## 5. Validation Strategy & The Famous Dilation Shakeup

- **Gold-Standard CV:** Evaluate **strictly on expert `DS1` tiles**, grouped by **WSI** (Leave-One-WSI-Out) or **spatial coordinate split** (`'i'`-coord split in `#1`; Left/Right half in `#2`). Random tile splits leaked adjacent vessels and stain profiles (`0.72` fake `segm_mAP60` vs `0.68` true spatial CV in `#1`).
- **Verified Anti-Patterns (Public-LB Shakeup Traps):**
  1. **Blind Mask Dilation (`cv2.dilate`):** Huge fake Public LB gains on WSI 3–4 (`+0.08–0.11`) that **destroyed Private LB on unseen WSI 5**: `#2` (`0.588` no-dilate → `0.526` dilated), `#13` (`0.541 → 0.462`), `#9` (`0.560 → 0.549`). *(Only clean `erode → 1× dilate` morphological smoothing helped `#3` by `+0.005`.)*
  2. **Over-Low Mask Threshold (`0.2` vs `0.5`):** `#8` showed `thr=0.5` gave **`0.560` Priv / `0.391` Pub**, while `thr=0.2` flipped to **`0.369` Priv / `0.506` Pub**!
  3. **Mask Union / Addition in Ensembles:** Over-expanded polygons below `IoU=0.6`, crashing `YYama` from **9th Public (`0.572`) → 295th Private (`0.397`)** (pixel voting instead hit **`0.575` Private**!).
  4. **Duplicate GT Polygons (`~4.5%`) & Evaluating on `DS2`:** Deduplicate `IoU > 0.99` GT masks (`#4`) and never validate on sparsely annotated `DS2`.

---

## 6. Verified Solution Sources

- **1st Place (`tascj`):** `https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/writeups/tascj-1st-place-solution`
- **2nd Place (`QL`):** `https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/writeups/ql-2nd-place-solution`
- **3rd Place (`Nischay`):** `https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/writeups/nischay-3rd-place-solution-how-to-properly-utilise`
- **4th Place (`[SDSRV.AI] GoN`):** `https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/writeups/sdsrv-ai-gon-4th-place-solution-sdsrv-ai-gon`
- **5th Place (`zw`):** `https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/writeups/zw-5th-place-solution-higher-resolution-and-datase`
- **7th Place (`yu4u`):** `https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/writeups/yu4u-7th-place-solution`
- **8th Place (`ria`):** `https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/writeups/ria-8th-place-solution-single-model-with-heavy-aug`
- **9th Place (`D.Imanishi`):** `https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/writeups/d-imanishi-9th-place-solution`
- **Shakeup Post-Mortem (`YYama`, 9th Pub → 295th Priv):** `https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/writeups/yyama-public-9th-private-295th-achieved-private-0-575`
