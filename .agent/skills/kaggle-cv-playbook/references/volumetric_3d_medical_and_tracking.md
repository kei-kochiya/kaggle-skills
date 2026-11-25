# 3D Volumetric, Medical Imaging, Tracking & Topology — Universal Recipes

Transferable 3D, volumetric, medical, and spatiotemporal CV patterns distilled from gold-medal solutions across multi-plane MRI classification, multi-modality 3D vessel localization, 3D+Time microscopy tracking, topological scroll surface segmentation, and 3D cryo-ET keypoint detection.

---

## 1. Architecture Decision Matrix: 2D/2.5D MIL vs. 3D Vessel-ROI `nnU-Net`

Choosing the wrong spatial dimensionality kills CV and runtime immediately:

| Volume Geometry | Recommended Paradigm | Proven Core Mechanics |
| :--- | :--- | :--- |
| **Multi-plane thick / irregular slices** (e.g., 3-plane 2D MRI stacks with varying thickness/views) | **Physical-mm FOV + 2D/2.5D MIL** (`ConvNeXt`, `CoAtNet-2`, `MaxViT-V2`, `DINOv3`) | Normalize to fixed physical FOV (`112–150 mm → 384 px`), canonicalize L/R anatomy via a fast side classifier, extract 2D/2.5D slice embeddings (`K=24` bags), aggregate via **gated MIL attention** with **test-time attention sharpening (`τ > 1`)**. |
| **Dense 3D volumetric scans** (CT, CTA, MRA, 3D microscopy, cryo-ET) | **Coarse-to-Fine 3D ROI + 3D `nnU-Net ResEnc`** (or stride-reduced 3D CNN) | Crop anatomical ROI first (fast 2D tri-axial / MIP YOLO / coarse 1mm 3D seg), preserve native anisotropic median spacing (never blindly downsample high-res in-plane axes to `1mm³` isotropic!), use **per-volume z-score normalization** across modalities. |

### High-Impact 3D Medical Patterns
1. **Anatomical Segmentation Pretraining + Region-Masked Pooling**:
   - Pretraining a 3D `nnU-Net` encoder on anatomical structure segmentation before target detection can deliver **`+0.10+` ROC-AUC** over training from scratch.
   - Pool 3D decoder features **inside each predicted anatomical region mask** → pass region tokens + global encoder GAP to a **Location-Aware Transformer** (or cross-attention pooling), adding an auxiliary local sphere/EDT detection head during training and **dropping the heavy 3D seg decoder at test time** to fit 12h T4 limits.
2. **Metadata-Missingness Resilience & Lateralized Mirroring**:
   - Hidden medical DICOMs frequently strip voxel-spacing or orientation tags. Train a lightweight **2.5D image-based voxel-spacing regressor** (`EffNetV2-S`) as a drop-in fallback when DICOM spacing is absent.
   - **L/R Mirror Swap Rule**: Whenever applying horizontal flipping (train or 8× TTA) on bilateral anatomy, you **MUST swap Left and Right class labels and segmentation masks** — naive mirroring silently corrupts lateralized classes.
3. **Site/Scanner-Clustered CV & Continuous Soft-Label VLM Distillation**:
   - Cluster CV folds by **hospital/scanner signature** (not patient ID alone) to prevent `+0.015–0.020` scanner-shortcut inflation.
   - When train includes radiology reports/unlabeled scans alongside tiny gold labels, distill **continuous `[0, 1]` soft probabilities** from report+image VLMs (dramatically outperforms discrete binned pseudo-labels) and keep the tiny gold test-like benchmark **100% held out** for unbiased validation.

---

## 2. 3-State Sparse Supervision, 3D+Time Tracking & Conservation MILP

For sparsely annotated 3D/4D microscopy or volumetric tracking (where `< 5%` of objects/frames are labeled):

1. **3-State Sparse Masking (`Positive / Verified Negative / Unknown-MASK`)**:
   - Never treat unlabeled space as negative background. Zero out gradients on unknown voxels via explicit `valid_mask` (3-tier normalized BCE, DoG exclusion zones, or global count-prior regularization) and apply **XY-only D4 dihedral augmentations** on Z-anisotropic volumes.
2. **3-Frame Temporal Context & Untransformed Motion TTA**:
   - Feeding `(t-1, t, t+1)` temporal triplets into a cached per-frame encoder rescues occluded/faint detections (`0.73 → 0.93`!).
   - **Motion-Vector TTA Trap**: Average scalar detection/segmentation logits across XY flips/rotations, **NEVER naively average raw 3D displacement vectors across TTA** (rotate/invert vectors back to canonical coordinates or keep untransformed motion head).
3. **Factored Division/Fork Heads & LP-First HiGHS Conservation MILP**:
   - Factor mitosis/branching into `kind` (none/continue/split) × `who` (parent-child candidate pairing) heads.
   - Solve global trajectory linking as a **flow-conservation MILP** with per-node division costs using **LP-first HiGHS** (delivers `38×–166×` speedup over SCIP with exact optimality), followed by drift-compensated local line-fit smoothing and short gap closing.

---

## 3. Full-Volume Inference, Raw SDF Averaging & Topological Post-Processing

Under topology-sensitive 3D surface metrics (`TopoScore / Betti`, `SurfaceDice`, `VOI`):

1. **Separate Geometric Accuracy (Network) From Topological Validity (Post-Processing)**:
   - Train long-schedule 3D `nnU-Net ResEnc` (`1k–8k` ep, multi-patch `128³–256³`) or **full-volume 3D encoders** (`320³` single pass — eliminates sliding-window boundary seams completely!) predicting **Signed Distance Fields (SDF)** / soft logits.
   - Ensemble in **raw SDF / logit space** before thresholding.
2. **Safe Topological Post-Processing Toolkit (`+0.02–0.08` Topo/Combined)**:
   - **2×2×2 Watertight LUT**: Local lookup-table pass preventing non-manifold edge/vertex pinches.
   - **Bridge-Checked Tunnel/Hole Filling (Betti-1 / Euler $\chi < 1$)**: Fill small holes/tunnels strictly when doing so does **not** merge separate disconnected sheets (verify component separation / non-bridging first!).
   - **Recursive Multi-Threshold Sheet Unmerging**: Detect merged sheets via component size/Euler anomaly and re-threshold locally at higher probability to split bridges cleanly.
   - **Anti-Pattern**: Never apply naive uniform morphological dilation/minimum-thickness inflation — it bridges neighboring layers and catastrophically collapses VOI/TopoScore!

---

## 4. Coarse-Stride 3D Keypoint Detection & $F_\beta$ Thresholding

For sparse 3D point/keypoint localization under recall-heavy $F_\beta$ (e.g., $F_2$) metrics:

1. **Coarse-Stride Blob Regression > Full-Res Masks or 2D Slice Detectors**:
   - Pure 2D/2.5D slice detectors (YOLO) suffer severe Private shakeup on noisy 3D volumes due to missing Z-context.
   - Regress **`8×`–`16×` downsampled 3D Gaussian or EDT (`r=25`) spherical blobs** (or use a **decoder-free stride-32 spatial classifier** when volumes contain strictly `0 or 1` target!).
   - Losses: **TopK-20% BCE** or **3-term SmoothBCE** (main + deep supervision + `4×4` max-pooled loss forgiving tolerance-radius offsets) decisively beat MSE/L1/Dice/Focal/Tversky under extreme foreground sparsity.
2. **Fast Volumetric Serving & Precision Traps**:
   - **Z-Only Sliding Window** (full XY crop with high Z-overlap + border downweighting) runs `4×` faster than XYZ tiling with zero XY seams.
   - **T4 GPU `bf16` & TensorRT Trap**: Emulated `bf16` on T4 (`sm_75`) is `3.5×–6×` slower than `fp16` — always train/infer `fp16` on T4, and **never export `bf16`-trained weights directly to `fp16` TensorRT** without range clamping (silent numerical overflow).
   - **Optimizer**: Test **`Muon` / `FastMuon`** alongside SGD/AdamW on 3D conv/transformer blocks.
3. **Prior-Matched CV & Distribution-Aware Thresholding**:
   - If test has a fixed `0-or-1` object prior (e.g. `50/50`) while train has multi-object volumes, create a **split-half validation split** matching the test prior.
   - Aggregate ensembles via **3D NMS + L2-distance 3D WBF**, then replace fragile fixed probability cutoffs with **test-batch quantile thresholding** or **Double-Otsu bimodal thresholding** aligned to $F_\beta$.
