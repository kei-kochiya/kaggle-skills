# TensorFlow — Help Protect the Great Barrier Reef 🏆

> **Track:** Computer Vision — 2D Small-Object Detection & Video Tracking (Research Code) · **Scale:** 2,025 teams · \$150,000 · Closed Feb 15, 2022  
> **Core Lesson:** In 2D small-object detection over sequential video under recall-heavy multi-IoU `F2`, **Leave-One-Video-Out CV (including 100% of empty background frames)**, **train-time `2×–3×` high-resolution / tiled training** (never inference-only extreme upscaling!), **2nd-stage ordinal-IoU box re-scoring classifiers**, **low-threshold pre-WBF fusion (`conf=0.01`)**, and **lightweight temporal attention/homography/Seq-NMS propagation** crushed Public-LB overfitters (`#1` surged **+120 ranks**, `#7` **+1,098 ranks**!).

---

## 1. Competition DNA & Shakeup Anatomy

- **Task:** Detect small underwater Crown-of-Thorns Starfish (`COTS`, `20–50 px`) in `1280×720` video frames (`23,501` train frames from just **3 videos** — `video_0/1/2`, with severe annotation sparsity: `25.5%`, `32%`, and `7.9%` annotated frames respectively; `~13,000` hidden test frames served sequentially via `iter_test()` with zero look-ahead, split ~25% Public / ~75% Private by **contiguous video clip**).
- **Metric:** **Mean Micro-`F2` across 11 IoU thresholds `0.30:0.05:0.80`**, matching predictions to ground truth in **descending confidence order** (weighting recall `4×` over precision and making box confidence ranking crucial!).

| Private # | Shake | Team | Private F2 | Public F2 (Rank) | Core Breakthroughs |
| :---: | :---: | :--- | :---: | :---: | :--- |
| 🥇 **1st** | **+120** | **QNS** | **0.760** | #121 | **"Trust CV" (`video_id` split)** + `3× YOLOv5@3648` + `3× YOLOv5` patch@1536 + **7-bin ordinal-IoU 2nd-stage classifier** (`0.716→0.73+`) + 3-frame attention boost (`→0.74+`) |
| 🥈 **2nd** | **+15** | **元宵快乐** | **0.737** | 0.648 sub (#17) | `video_id` 3-fold CV + `4× YOLOv5l6` scaled `1280→3200px` (`0.620→0.698` CV) + `RandomRotate90` (`+0.02`) + 1-frame attention boost (`+0.01`) |
| 🥉 **3rd** | `+1` | **Team Hydrogen** | **0.735** | 0.790 (#4) | Diverse 5-family ensemble (`YOLOv5l6`, `CenterNet HRNet`, `Faster R-CNN`, `FCOS`, `EffDet`) + **Background-Mix** + track-level max-confidence propagation (`+0.01`) |
| 🏅 **4th** | `-3` | **outrunner** | **0.728** | **0.806 (#1)** | Solo **CenterNet (`DeepLabV3+ + EffNetV2 B0–XL`)** at `1/8` heatmap scale |
| 🏅 **5th** | **+19** | **bestfitting** | **0.728** | 0.735 (#24) | **Poisson Blending + Real/Fake box classifier** + `YOLOv5/YOLOX/YOLOR/HRNet` + **SuperPoint+SuperGlue homography + DeepSort** |
| 🏅 **6th** | **+35** | **Recall Is All You Need** | **0.727** | 0.668 sub (#41) | `YOLOv5m6/s6` + **ultra-low `conf=0.01` pre-WBF (`+0.02`)** + Swin 2nd-stage classifier (`+0.006–0.010`) |
| 🏅 **7th** | **+1098** | **Three Man Works!** | **0.727** | >#1000 | Single customized **Cascade R-CNN (`CBNet + PAFPN + Double Head + IoU loss`)** |
| 🏅 **8th** | **+220** | **tk** | **0.726** | 0.611 (#228) | Strict `video_2` holdout CV + **`Cascade R-CNN (ConvNeXt-Base)`** (`2816px`) + `YOLOX-s` WBF |
| 🏅 **9th** | **+60** | **bilzard** | **0.724** | #69 | **Tiled `320×320 → 640px` training → full-frame `2560px` inference** + PL + **Online Seq-NMS (`+0.01` Priv)** |
| 🏅 **10th** | **+26** | **Where's COTS?** | **0.722** | 0.718 (#36) | **P2/4 stride-1 YOLOv5l6** (`1280px` input w/ `2560`-equiv grids!) + **ORB homography + Norfair** (`+0.01–0.02`) |

---

## 2. Validation Strategy ("Trust CV" vs. Video Leakage)

- **Why Random Frame Splits Collapsed:** Consecutive video frames are nearly identical; random splits leak adjacent frames and produce fake `0.95+` CV.
- **Leave-One-Video-Out (`video_id` 3-Fold) & Sequence GroupKFold:** `#1`, `#2`, `#6`, `#8`, `#10` validated strictly by `video_id` (or disjoint `sequence` clips in `#5`, `#11`, `#12`, `#15`), **always evaluating on 100% of frames including empty background frames** to accurately capture false positives.
- **Exact Metric Alignment (`#1`):** Team members initially got `0.62`, `0.66`, and `0.68` on the exact same OOF from subtle F2 script bugs — standardizing on one strict official F2 implementation was mandatory before comparing experiments.

---

## 3. Small-Object Architectures, Resolution Scaling & Synthetic Blending

1. **Train-Time High Resolution & Tiling (Biggest Single Driver):**
   - `#2` ablation on `YOLOv5l6` (`train = infer` res): `1280px → 0.620` | `1800px → 0.6685` | `2400px → 0.6865` | **`3200px → 0.6976` CV!**
   - **Tiled Training → Full-Frame HR Inference (`#1`, `#9`, `#12`):** Train on overlapping crops (`320×320 @ 640px` in `#9`; `512×320 @ 1536px` in `#1`; `4× 1280×720` crops of `2560×1440` on 16GB P100 in `#12`), then infer on full frame at `2560–3648px`. Single `YOLOv5l` in `#9` hit **`0.726` Private**!
   - **Stride-1 `P2/4` Head Trick (`#10`):** Replaced stride `2 → 1` in YOLOv5l6's 2nd conv (`obj` gain `8.0`) so native `1280×720` input achieved `2560`-equivalent anchor density without interpolation artifacts.
2. **Cross-Family Diversity:** `YOLOv5l6/m6`, `YOLOX`, `Cascade R-CNN (ConvNeXt-B / CBNet Double-Head)`, `CenterNet (DeepLabV3+ EffNetV2 / HRNet)`, `FCOS`, `YOLOR-P6`, `HRNetV2P-W18`.
3. **Training Augmentations & Label Cleaning:**
   - **Background Frames in Train:** Most top teams trained on `0%–10%` background frames (keeping gradients focused on positive boxes) while **validating on 100% of frames**.
   - **Poisson Blending + Real/Fake Box Filter (`#5 bestfitting`):** Eliminated naive copy-paste seam/background artifacts by Poisson-blending cropped objects onto background frames and filtering unrealistic placements with a Real-vs-Fake classifier.
   - **Background-Mix (`#3`):** Blended annotated frames with background-only frames (randomizing contrast/background without occluding boxes).
   - **`RandomRotate90` (`+0.02` CV & LB in `#2`)** and missing-box manual/PL relabeling (`+0.01` CV/Priv in `#9`, `#11`).

---

## 4. 2nd-Stage Box Re-Scoring, WBF & Temporal Video Tracking

1. **7-Bin Ordinal-IoU 2nd-Stage Box Classifier (`#1 QNS` — `+0.015+` CV!):**
   - Cropped every OOF box (`conf > 0.01`, `1.2× max(w, h)` square → `224–256px`) and trained a heavily regularized CNN (`dropout=0.7, drop_path=0.5`) on **7 binary ordinal targets (`IoU > 0.3, 0.4, ..., 0.9`)** via `BCE`, averaging all 7 probabilities as the calibrated box score (`0.716 → 0.727 → 0.73+` CV; `#6` also gained `+0.006–0.010` via a Swin crop classifier). Directly aligns confidence ordering with multi-IoU `F2`!
2. **Low-Threshold Pre-WBF Fusion (`conf = 0.01–0.05`):** Running detectors at `conf = 0.01–0.05` *before* WBF allows weak multi-model detections to reinforce each other (`+0.02` in `#6`, `+0.005` in `#11`), then applying the `F2` recall threshold (`~0.10–0.20`) post-WBF. `#11` gated Cascade R-CNN boxes (`IoU >= 0.3`) strictly to refine YOLO localization.
3. **Temporal Video Tracking & Camera-Motion Propagation (`+0.01` to `+0.02` CV & Priv):**
   - **Attention-Area Confidence Boost (`#1`, `#2`):** High-conf boxes in frame `t` boost overlapping (`IoU > 0.5`) low-conf boxes (`> 0.01`) in frames `t+1..t+3` via `score += conf * IoU` (`+0.01` CV & LB, beating noisier Kalman/SORT setups!).
   - **Track-Level Max-Confidence Propagation (`#3`):** Elevated all detections on a persistent Euclidean track to `max(track_conf)` (`+0.01` CV).
   - **Homography + Tracker + Optical-Flow Reset (`#5`, `#10`, `#11`, `#12`):** Since targets are static while the camera moves, **SuperPoint+SuperGlue** (`#5`) or **ORB** (`#10`) homography + `Norfair`/`DeepSort` added `+0.01–0.02` across CV/Pub/Priv; **resetting trackers on optical-flow cuts** (`#11`) added `+0.01` Priv; **20-frame decayed Online Seq-NMS** (`#9`, `#15`) added `+0.009–0.011` Priv.

---

## 5. Verified Anti-Patterns (Anatomy of the Famous Shakeup)

1. **Inference-Only Extreme Upscaling (`2×–2.5×` Over Train Res):** Inferring `#2`'s `2400px` model at `4800px` crashed CV **`0.6865 → 0.3925`** while artificially spiking Public LB (`0.60 → 0.730`); `#38` inferred a `3600px` model at `7200px` and dropped **Public #6 (`0.769`) → Private `0.641`**!
2. **Public-LB Bounding-Box Shrinking Overfit:** Public test annotations (a single contiguous ~3,250-frame clip) happened to be `~6%–12%` tighter than train & Private boxes. Shrinking box edges by `6%` jumped `atom1231` to **Public #2 (`0.788`)**, then crashed to **Private #376 (`0.641`)**! `#14` confirmed `0.875×` scaling gave `0.737 Pub → 0.673 Priv` vs `1.0×` giving `0.636 Pub → 0.712 Priv`.
3. **Chasing Public LB Over Video CV:** `#6`'s submission matrix nailed it — best CV `1.5×` sub scored `0.668 Pub → 0.727 Priv`, whereas best Public LB sub (`0.716 Pub`) dropped to `0.692 Priv`.

---

## 6. Verified Solution Sources

- **1st Place (`QNS`):** `https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/writeups/qns-trust-cv-1st-place-solution`
- **2nd Place (`元宵快乐`):** `https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/writeups/2nd-solution-yolov5`
- **3rd Place (`Team Hydrogen`):** `https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/writeups/team-hydrogen-3rd-place-solution-team-hydrogen`
- **4th Place (`outrunner`):** `https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/writeups/outrunner-4th-place-solution-centernet`
- **5th Place (`bestfitting`):** `https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/writeups/bestfitting-5th-place-solution-poisson-blending-de`
- **6th Place (`Recall Is All You Need`):** `https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/writeups/recall-is-all-you-need-6th-place-solution-yolov5m6`
- **7th Place (`Three Man Works!`):** `https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/writeups/three-man-works-7th-place-solution-cascade-rcnn-tr`
- **8th Place (`tk`):** `https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/writeups/tk-8th-place-solution`
- **9th Place (`bilzard`):** `https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/writeups/bilzard-9th-place-solution-tiled-training-seq-nms`
- **10th Place (`Where's COTS?`):** `https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/writeups/where-s-crown-of-thorns-starfish-10th-place-soluti`
