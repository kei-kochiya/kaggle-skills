---
name: kaggle-cv-playbook
description: >-
  Universal competitive computer vision workflow for image regression, classification, detection, segmentation, and visual forensics on Kaggle. Use when designing CV splits, choosing vision foundation backbones (DINOv3/ConvNeXt), preserving aspect ratios, handling domain shift with TTT/SSA, building dual-threshold empty-mask gates, or deciding between end-to-end DL and structured detection+matching pipelines.
---

# Kaggle Computer Vision Competitions Playbook

A universal, cross-competition runbook distilled from gold-medal CV solutions across image regression, instance/semantic segmentation, and visual forensics. Competition-specific constants and dataset quirks remain strictly in `Handbook/cv/`.

---

## 1. Phase 1: Problem Framing & Validation Under Domain Shift

Before writing a training loop, audit two failure modes that sink most CV entries:

1. **Test-Aligned Validation Beats Random Train Splits**:
   - Group folds by capture session, date, camera, patient, or site (`StratifiedGroupKFold`) whenever repeated environments exist.
   - If train and test come from distinct visual regimes (e.g., single-object lab crops vs. compound real-world scenes), **do not optimize hyperparams against a misaligned train split**. Build a clean, test-aligned validation proxy (supplemental real-world images, external curated holdouts, or embedding-distance `easy` vs. `hard` validation slices via 5-NN cosine/Euclidean distance to train).
2. **Audit Whether End-to-End DL Is Even the Right Formulation**:
   - For **dense classification / regression / direct segmentation**, foundation vision encoders dominate.
   - For **compound multi-region matching, correspondence, or copy-move forensics**, decompose into **Region Detection (`YOLO`/`SAM`) → Distractor Suppression (text/plots/borders) → Metric-Learning Retrieval (`SupCon`) → Keypoint Matching (`SIFT`/`RootSIFT`/`ALIKED`/`SuperPoint` + `LightGlue`) → Geometric Verification (`RANSAC`/`MAGSAC++`)** before trusting pure pixel segmentation. See [`references/domain_shift_ttt_and_gating.md`](./references/domain_shift_ttt_and_gating.md).

---

## 2. Phase 2: Vision Foundation Backbones & Spatial Geometry

Refer to [`references/dense_foundation_backbones.md`](./references/dense_foundation_backbones.md) for implementations.

1. **Self-Supervised Dense Backbones (`DINOv3`) First for Spatial/Texture Tasks**:
   - When predictions depend on fine-grained spatial structure, density, texture, or subtle localized cues (rather than coarse semantic captions), self-supervised ViTs (`DINOv3` Large/Huge+/7B) consistently outperform language-supervised (`CLIP`/`SigLIP`) and purely supervised CNN backbones.
   - Pair `DINOv3` with orthogonal families (`ConvNeXtV2`, `ViT-MAE`, `EVA02`) strictly for ensemble diversity.
2. **Preserve Native Aspect Ratio & Physical Scale**:
   - Never blindly warp `2:1` or wide images to squares. Prefer:
     - **Dual-stream square tiling** (shared encoder + cross-tile MHSA or additive head pooling),
     - **Native rectangular patch grids** with RoPE ViTs, or
     - **Image + additive-target halving** (`2×` training samples when targets sum linearly across sub-regions).
3. **Discard the CLS Token for Dense / Spatial-Sum Tasks**:
   - Reshape intermediate or final patch tokens `[B, N, C] → [B, C, H_p, W_p]`:
     - **Minimalist Density / Segmentation Head**: Single **`Conv2d(1×1)`** (avoid unstable `3×3` on small datasets) followed by `Softplus → Spatial Sum` (for areal/count regression) or bilinear upsampling (for segmentation).
     - **Multi-Scale Pyramid Heads**: Tap 3–4 intermediate transformer layers into a `DPTHead` or ViTDet `SimpleFeaturePyramid` when multi-scale context matters.
4. **Small-N Backbone Protection (Never Destroy Pretrained Geometry)**:
   - On small-to-medium datasets (`< 5k` images), aggressive full fine-tuning collapses generalization. Use:
     - **Two-stage freeze → unfreeze** (head-only warmup, then `0.1×` backbone LR or unfreeze only the top `20%–50%` of transformer blocks),
     - **Frozen backbone + QKV LoRA** on upper blocks (mandatory for 7B-scale backbones),
     - **8-bit quantization at inference** to fit multi-billion-parameter ViTs on 16GB competition GPUs.

---

## 3. Phase 3: Losses, Training Discipline & Augmentations

1. **Hybrid Regression + Quantile Interval Classification (UEPNet Pattern)**:
   - On skewed continuous regression targets, attach an auxiliary $K$-bin quantile/binary-search interval classification head (`SmoothL1Loss + 0.3 * CrossEntropyLoss`) alongside the regression head to stabilize gradients across wide dynamic ranges.
2. **Heteroscedastic Uncertainty Loss (`GaussianNLLLoss`)**:
   - Predict `(mean, softplus(var_logit) + eps)` when labels contain measurement noise or heavy outliers so the network automatically downweights unpredictable samples.
3. **Per-Target Checkpoint Early Stopping**:
   - In multi-target CV models, distinct targets frequently peak 10+ epochs apart. Track validation metrics **per target** and load per-target best checkpoints (or use SWA/EMA) instead of one global stop epoch.
4. **Label-Preserving Augmentations & Synthetic Data Hygiene**:
   - Use scale/occlusion regularizers (`RandomResizedCrop`, `RandomErasing`, `MixUp/CutMix`, strip/tile permutations that preserve total mass).
   - **Never reuse original labels on generative/diffusion-edited images**—edits silently alter target signal. Always pseudo-label edited images with a real-data teacher, or construct exact zero-target negative anchors.

---

## 4. Phase 4: Test-Time Adaptation, Gating & Post-Processing

Refer to [`references/domain_shift_ttt_and_gating.md`](./references/domain_shift_ttt_and_gating.md) for full patterns.

1. **Test-Time Online Training (TTT) + Clean-Train Refit**:
   - When hidden test sets are large and domain-shifted, run iterative hard pseudo-labeling (`train + pseudo_test` with SWA) followed by a **short clean fine-tune on original labeled train data** to purge pseudo-label drift before inference.
2. **Training-Free Feature Adaptation (SSA)**:
   - When runtime prohibits full TTT, apply **Significant Subspace Alignment (SSA)** to align test feature moments to precomputed source PCA subspaces via a lightweight adapter.
3. **Dual-Threshold Empty/Presence Gating (Segmentation / Detection)**:
   - Under metrics where a false-positive mask on an empty image converts `1.0 → 0.0` (e.g., `oF1`, Dice with empty images), separate **detection** from **masking**:
     - Gate 1 (Presence): Require `sum(prob > high_thr) >= min_area` (or `quantile(prob, 0.999) >= q_thr`), else predict empty.
     - Gate 2 (Masking): Only for images passing Gate 1, threshold at lower recall-optimal `mask_thr` and merge overlapping/fragmented components if excess instances are penalized.
4. **Soft Physical Identity Reconciliation**:
   - When targets obey exact linear sums ($T = A + B$), avoid rigid hard training constraints if component noise is high; train independent heads and reconcile softly at post-processing (`0.5 * direct + 0.5 * sum`).

---

## 5. Phase 5: 3D Volumetric, Medical, Tracking & Topology Patterns

Refer to [`references/volumetric_3d_medical_and_tracking.md`](./references/volumetric_3d_medical_and_tracking.md) for full recipes:
1. **2D/2.5D MIL vs. 3D Vessel-ROI `nnU-Net`**: Multi-plane thick slices → physical-mm FOV + anatomical mirroring + 2D/2.5D MIL (with test-time attention sharpening `τ > 1`); dense 3D scans → coarse-to-fine ROI + anatomical-seg pretrained 3D `nnU-Net` + region-masked pooling (drop seg decoder at test; always swap L/R labels & masks on L/R mirror; train an image-based voxel-spacing regressor if test DICOM tags are stripped).
2. **Site/Scanner-Clustered CV & Continuous VLM Soft Labels**: Cluster CV by hospital/scanner to stop shortcut leaks; distill continuous `[0, 1]` report+image soft probabilities (crushes discrete bins) while keeping gold benchmarks held out.
3. **3-State Sparse Masking, 3D+T Tracking & Topology PP**: Mask unknown voxels (`1/0/MASK`), use XY-only D4 aug on Z-anisotropic volumes, add 3-frame `(t-1,t,t+1)` context (never average raw 3D motion vectors across TTA), solve global tracking with **LP-first HiGHS MILP**, average raw SDFs under full-volume inference, and repair topology in PP (watertight LUT, bridge-checked Betti-1 hole filling, recursive sheet unmerging).
4. **Coarse-Stride & Extreme Small-N 3D Keypoints + Recall-Heavy $F_\beta$**: Regress `8×–16×` Gaussian/EDT blobs or truncate U-Nets at the **penultimate stride-2 map** with shrunk sphere masks (`0.33×–0.5×`); keep unscored distractor classes as auxiliary outputs and train across all available reconstruction variants; prefer compact `1M–14M` 3D CNNs (`InstanceNorm3d+PReLU`) or **2.5D depth-pooling U-Nets** over heavy 3D Transformers; align cross-paradigm OD+Seg score maps via **per-class rank/quantile histogram matching**; under $\beta \ge 2$, avoid saturating Dice/Tversky (use weighted CE/BCE, `FocalTversky++`, or TopK-20% BCE + **logit-space averaging**), infer on larger patches with pre-compiled TensorRT (`fp16` on T4, never emulated `bf16` or unclamped `bf16→fp16`), and test `Muon`.
5. **Multi-Scene 3D SfM & Correspondence Graphs**: Match high-resolution sparse keypoints (`ALIKED`/`SuperPoint`/`RDD`) via **3D-foundation local features (`MASt3R`)** + multi-retriever shortlisting (`ASMK + DINOv2`) + cycle-consistency pruning; let **incremental multi-model `pycolmap`** (in retryable subprocesses) separate scenes and outliers rather than fragile early pre-clustering or `GLOMAP`, and balance pairwise multi-GPU workloads by **$\sum n_i^2$**.

---

## 6. Universal CV Anti-Patterns

- **Blindly Enabling TTA**: Flip/crop TTA can hurt calibrated regression metrics ($R^2$, RMSE) under shift—always ablate TTA per task.
- **Unverified Auxiliary Metadata Losses**: Train-only metadata auxiliary heads often dilute visual capacity and degrade primary target generalization; verify strictly via grouped CV.
- **Hand-Cleaning Suspected Noisy Train Labels**: If the test set shares the same annotation noise process, removing "noisy" train rows inflates CV while hurting Private LB.
- **Single-Split Public LB Over-Selection**: When Public/Private domain shift exists, prefer multi-seed / multi-backbone SWA ensembles over the highest single Public-LB checkpoint.

---

## 7. References & Case Studies
- [Dense Foundation Backbones, Patch Heads & Training Discipline](./references/dense_foundation_backbones.md)
- [Domain Shift Validation, TTT/SSA, Dual-Threshold Gating & Matching Pipelines](./references/domain_shift_ttt_and_gating.md)
- [3D Volumetric, Medical Imaging, Tracking & Topology](./references/volumetric_3d_medical_and_tracking.md)
- Case Studies: [`CSIRO Biomass`](../../../Handbook/cv/csiro-image2biomass.md) · [`Recod.ai Forgery`](../../../Handbook/cv/recodai-scientific-image-forgery.md) · [`RSNA Knee`](../../../Handbook/cv/rsna-knee-abnormality.md) · [`Biohub Cell Tracking`](../../../Handbook/cv/biohub-cell-tracking.md) · [`Vesuvius Surface`](../../../Handbook/cv/vesuvius-surface-detection.md) · [`RSNA Aneurysm`](../../../Handbook/cv/rsna-intracranial-aneurysm.md) · [`BYU Flagellar Motors`](../../../Handbook/cv/byu-flagellar-motors.md) · [`CZII CryoET`](../../../Handbook/cv/czii-cryoet-object-identification.md) · [`IMC 2025`](../../../Handbook/cv/image-matching-challenge-2025.md)
