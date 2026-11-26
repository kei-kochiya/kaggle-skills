# Image Matching Challenge 2025 (CVPR): Top-10 Solutions & Multi-Scene 3D SfM Post-Mortem

> **Competition**: [Image Matching Challenge 2025](https://www.kaggle.com/competitions/image-matching-challenge-2025) (Apr – Jun 2025, 943 Teams, \$50,000 Prize Pool; hosted by CTU Prague / CVPR 2025)  
> **Track**: 3D Computer Vision, Multi-Scene Structure-from-Motion (SfM), Local/3D Foundation Matching & Graph Clustering  
> **Task & Runtime**: Unstructured flat image collections (`~1,300` hidden test images) mixing multiple visually similar non-overlapping scenes plus distractors/outliers — must cluster images into scenes (or `outliers`) AND recover 6-DoF camera extrinsics `(R, t)` inside `<= 9h` offline on `2× NVIDIA T4`  
> **Evaluation Metric**: Dataset-averaged **harmonic mean (F1)** of **Clustering Precision** and **mAA** (pose recall + accuracy across meter thresholds) under greedy scene-to-cluster association

---

## 1. Official Top-10 Leaderboard & Architectural Paradigms

**1st place (`ns64`, `56.00` Private)** crushed the field by **`+3.95` points** over #2 using a **MASt3R semi-dense + sparse keypoint hybrid** with zero pre-clustering (letting incremental `pycolmap` partition high-precision match graphs), while #2–#4, #6–#10 refined multi-detector/tiled `ALIKED`/`RDD`/`GIM` + `LightGlue` cascades.

| Private Rank | Team | Private Score | Public Score (Δ) | Core Pipeline & Key Innovations | Writeup / Code |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **1st** | **ns64** | **56.00** | 52.64 (0) | **MP-SfM-inspired MASt3R (`512px`) hybrid**: matched pre-extracted **ALIKED (`1280px`) + SuperPoint (`1600px`)** via MASt3R's local feature head + semi-dense matches; multi-retriever union (`ASMK+SPoC+DINOv2+ISC`); **zero pre-clustering (`+5.07` Priv)**; CUDA `curope` + SDPA | [Writeup](https://www.kaggle.com/competitions/image-matching-challenge-2025/writeups/ns64-1st-place-solution) · [GitHub](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025) |
| 🥈 **2nd** | **Nazarko99** | **52.05** | 52.44 (0) | 5-source cascade @ `1280px` (`GIM-SuperPoint` → `ALIKED+LG` + `SIFT+NN`) + **ROI target-object crops** re-matched by `ALIKED/SP+LG` + RANSAC `>=150` matches filter + default COLMAP | [Writeup](https://www.kaggle.com/competitions/image-matching-challenge-2025/writeups/nazarko99-2nd-place-solution) |
| 🥉 **3rd** | **Roni Heka** | **49.05** | 49.37 (0) | `DINOv2` → `ALIKED+LG @ 960px` (0/90/180/270° retry) → **5-image tiling (4 quadrants + 0.5× full = 25 tile pairs)** + `USAC_MAGSAC (50k iters)` + ascending-size cluster merge order | [Writeup](https://www.kaggle.com/competitions/image-matching-challenge-2025/writeups/roni-heka-3rd-place-solution) |
| 🏅 **4th** | **xtcpete** | **47.88** | 48.02 (0) | **RDD** (CVPR 2025 deformable transformer, `8192` kpts, multi-scale `1024+1280px` TTA) + `LightGlue` + **IMC+MegaDepth linear-transformer pair classifier** + `NetVLAD` | [Writeup](https://www.kaggle.com/competitions/image-matching-challenge-2025/writeups/xtcpete-4th-place-solution) · [GitHub](https://github.com/xtcpete/rdd) |
| 🏅 **5th** | **Sayan Paul** | **47.12** | 43.37 (+3) | Ultra-clean (`12` entries!): **`MASt3R-ASMK` retrieval (`47.12` vs `DINOv2 43.64`)** + `MASt3R` semi-dense (`512px`) + retryable subprocess `pycolmap` (`min_model_size=3`) | [Writeup](https://www.kaggle.com/competitions/image-matching-challenge-2025/writeups/sayan-paul-5th-place-solution-mast3r-is-all-you-ne) |
| 🏅 **6th** | **Never happen** | **46.61** | 46.98 (-1) | All-pair `ALIKED+LG` + adaptive top `k·log(n)/(n-1)%` edge pruning (`1024 → 2048px`) (beat `Louvain` pre-clustering: `46.61` vs `45.33` Priv) | [Writeup](https://www.kaggle.com/competitions/image-matching-challenge-2025/writeups/never-happen-6th-place-solution) |
| 🏅 **7th** | **KaQlN** | **45.58** | 42.09 (+6) | Connected components on inlier graph + **0/90/180/270° rotation correction (`38.67 → 45.58` Priv!)** + `1792px` crop `ALIKED` | [Writeup](https://www.kaggle.com/competitions/image-matching-challenge-2025/writeups/kaqln-7th-place-solution) |
| 🏅 **8th** | **yangyefd** | **45.36** | 45.07 (-2) | `GIM-LightGlue` + `CLIP (0.76)` retrieval + DBSCAN crop secondary matching + **cycle/loop consistency pruning (`>30px` dropped)** + top-1,500 match cap | [Writeup](https://www.kaggle.com/competitions/image-matching-challenge-2025/writeups/yangyefd-8th-place-solution) · [GitHub](https://github.com/yangyefd/IMC2025) |
| 🏅 **9th** | **ymg_aq** | **44.69** | 41.08 (+6) | `DINOv2` top-50 (switching to exhaustive if `N < 100`) + **quadratic `∑ n_i²` load balancing across `2× T4` (`-15%` runtime)** | [Writeup](https://www.kaggle.com/competitions/image-matching-challenge-2025/writeups/ymg-aq-9th-place-solution-dinov2-optimized-filteri) |
| 🏅 **10th** | **Team Sony Matching** | **44.51** | 39.65 (+18) | **KeyNet-AdaLAM** pair ranking + dynamic MST-like top-$k$ + `ALIKED+LG (10k kpts)` + DBSCAN crops; also benchmarked **`VGGT` + `LightGlue` refiner (`36.65 → 41.02` Priv)** | [Writeup](https://www.kaggle.com/competitions/image-matching-challenge-2025/writeups/team-sony-matching-10th-place-solution) |

---

## 2. Core 3D Vision, Retrieval & SfM Breakthroughs

### 2.1 Why 3D Foundation Matching (`MASt3R` Hybrid) Shattered Local-Only Ceilings (#1, #5)
- **3D Geometric Precision Against Visual Symmetry**: Unlike MegaDepth-trained 2D matchers that hallucinate matches across symmetrical walls, staircases (`stairs`), and repeating vineyard rows (`fbk_vineyard`), **MASt3R**'s 3D-pretraining yields drastically lower cross-scene false-positive match rates and native rotation/viewpoint invariance (eliminating rotation preprocessing in #1).
- **Semi-Dense + Sparse Keypoint Matching (#1 `ns64`)**: Matching pre-extracted high-resolution **`ALIKED` (`1280px`, `4k` kpts) + `SuperPoint` (`1600px`, `4k` kpts)** using MASt3R's local feature head alongside MASt3R semi-dense reciprocal NNs jumped scores from `53.79` (sparse-only) to **`55.31–56.00` Private**, combining MASt3R's 3D precision with sub-pixel keypoint accuracy.
- **Multi-Retriever Shortlisting (`#1, #5`)**: Combining **`MASt3R-ASMK (k=25) + MASt3R-SPoC (k=10) + DINOv2 (k=10) + ISC (k=10)`** (#1) matched heavy `ASMK(k=60)` at a fraction of the time. On #5's clean ablation, retrieval quality alone swung Private score by `+16.4` pts: `XFeat-agg (30.68)` ≪ `DINOv2 (43.64)` ≪ **`MASt3R-ASMK (47.12)`**.

### 2.2 Multi-Scale Cropping, Tiling, Cycle Consistency & Clustering
- **Skip Explicit Pre-Clustering When Matches Are Clean (#1, #5, #6)**: Letting incremental `pycolmap` (`min_model_size=3, max_num_models=25`) split sparse sub-models beat explicit pre-clustering: #1 dropped **`56.00 → 50.93` Private** (`fbk_vineyard 72.09 → 47.25`) when adding farthest-point pre-clustering; #6 beat `Louvain` (`46.61` vs `45.33` Priv).
- **Two-Pass ROI Crops & 5-Tile Matching (#2, #3, #8, #10)**: Extracting dense match bounding boxes (via ensemble/DBSCAN match clusters) and re-matching high-res crops (`1280–1792px`) or matching 5 tiles (`4 quadrants + 0.5× full = 25 tile pairs`, #3) rescued small overlapping regions on wide-baseline pairs (`36.98 → 41.7` in #8).
- **Cycle / Loop-Closure Pruning (#8)**: Dropping image pairs whose 3-view match cycle error exceeded `30px` cleaned false symmetry bridges cleanly before bundle adjustment.
- **Rotation Correction (`+6.91` Priv for 2D Matchers, #3, #4, #7)**: Testing `0°/90°/180°/270°` rotations lifted #7 (`ALIKED`) from `38.67 → 45.58` Private, while #1 (`MASt3R`) needed zero rotation correction.
- **2× T4 Systems Engineering (#1, #5, #9)**: Compiling CroCo/MASt3R `curope` (RoPE2D) with CUDA + PyTorch `SDPA` FlashAttention, wrapping crash-prone `pycolmap` calls in **retryable subprocesses** (#5), and partitioning datasets across `2× T4` by balancing **$\sum n_i^2$** (matching pair-graph quadratic complexity, **`-15%` runtime**, #9).

### 2.3 Where `VGGT` Shone — And Why It Didn't Win Yet (#1, #10)
- `VGGT`'s global multi-image attention excelled on brutal symmetry (`25.07` on `stairs` vs `15.89` for #1's MASt3R!) and sparse scenes (`amy_gardens`, `fbk_vineyard`).
- Pairing `VGGT` coarse `518px` poses with **`ALIKED+LightGlue` + `DetectorFreeSfM` `incremental_model_refiner`** lifted #10's VGGT branch from `36.65 → 41.02` Private (`ETs 17.54 → 66.67`!). However, `VGGT` assumed single-scene outlier-free inputs and hit VRAM / `9h` timeout walls on Kaggle `2× T4`.

---

## 3. Empirical Negative Results & What Failed

| Failed Hypothesis | Observed Result | Root Cause |
| :--- | :--- | :--- |
| **`GLOMAP` (Global SfM) vs. Incremental `pycolmap` (#1, #5)** | Consistently lower CV & LB | Global SfM is far more fragile to multi-scene false-match bridges and distractor outliers than incremental `pycolmap` multi-model splitting. |
| **Explicit Scene Pre-Clustering / `Louvain` (#1, #6)** | **`56.00 → 50.93` Priv** (#1); `46.61 → 45.33` (#6) | Hard pre-clustering fragmented valid weak-bridge scenes (`fbk_vineyard 72.09 → 47.25`) before 3D geometric verification could rescue them. |
| **Raw `VGGT` & `MASt3R-SfM` `fastNN` Direct Poses (#1, #7, #10)** | Timeouts (`TimeoutError`) or insufficient pose precision | `518px` input bottleneck and direct non-BA poses failed strict meter-scale mAA thresholds without high-res `pycolmap`/refiner bundle adjustment. |
| **Dense Detector-Free Matchers (`LoFTR`, `DKM`, `OmniGlue`, `MicKey`) (#6, #8, #9)** | Too slow or diverged on low-parallax / loop closures | Overly dense noisy graphs caused false coarse convergence and loop-closure pose drift vs. high-precision `MASt3R` or `ALIKED/RDD+LG`. |
| **Test-Time Self-Supervised `LightGlue` TTT (#8)** | Improved `stairs` matches but **dropped overall LB** | Unsupervised TTT on mixed multi-scene collections amplified cross-scene false matches between visually similar buildings. |
| **YOLO Person Masking & Generic Pair Classifiers (#4, #8, #9)** | `0` LB gain (even with `+2` train CV in #8) | YOLO accidentally masked **statues/sculptures (`ETs`)**; `Doppelgängers++` and generic CNN/LR pair classifiers failed to generalize across unseen test scene types. |

---

## 4. Universal Takeaways Promoted / Reinforced

- **Hybrid 3D-Foundation + High-Res Sparse Keypoint Matching**: Match high-resolution sparse keypoints (`ALIKED`/`SuperPoint`/`RDD`) using 3D-foundation local features (`MASt3R`) + incremental multi-model `pycolmap` bundle adjustment—combining 3D geometric false-positive rejection with sub-pixel pose precision.
- **Let Geometric Verification Split Clusters Rather Than Fragile Early Pre-Clustering**: Use multi-retriever shortlisting (`ASMK + DINOv2`) + cycle-consistency pruning and allow multi-model incremental SfM (`pycolmap` in retryable subprocesses) to separate scenes and reject outliers.
- **Quadratic ($\sum n_i^2$) Multi-GPU Sharding for Pairwise Pipelines**: When distributing pairwise graph/matching tasks across multiple GPUs, balance partitions by $\sum n_i^2$ (pair count) rather than $\sum n_i$ (item count).
