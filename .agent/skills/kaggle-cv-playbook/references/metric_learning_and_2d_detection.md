# Open-Set Metric Learning, 2D Small-Object Detection & 2D Instance Segmentation — Universal Recipes

Transferable, cross-competition computer vision recipes covering open-set long-tail image retrieval (`MAP@K`), 2D small-object video detection (`F_beta` / multi-IoU), and 2D instance segmentation (`AP@IoU`) under tiered label noise.

---

## 1. Open-Set Long-Tail Metric Learning & Image Retrieval (`MAP@K`)

When identifying $10^4+$ long-tail individuals/entities alongside an open-set **`new_entity` / unseen class**:

1. **Multi-Crop Detector Diversity & High-Resolution CNNs**:
   - Invest early in iterative detector cleaning (hand-auditing low-confidence OOF boxes + `+15–20%` box padding) and **mix multiple crop views during training** (full-frame, whole-object, fine-part crops) so embeddings survive partial views at test time.
   - To fuse two views (`[Original | Crop]` or `[Body | Part]`), use **side-by-side spatial concatenation** into high-resolution CNNs (`EffNet-L2/V2`, `ConvNeXt-L/XL`, `ECA-NFNet-L2` at `768–1440px`)—**never stack views into 6-channel early inputs** and never blindly double classes with horizontal flip (use standard HFlip + sub-center metric heads).
   - For multi-object test images, match each detected box embedding against the gallery and pick the closest match above a cosine floor (else highest detector score).
2. **Sub-Center ArcFace + Dynamic Margins + Fixed `GeM` + `BNNeck`**:
   - Pair **`GeM (p=3, fixed)`** (trainable `p` frequently degrades convergence) + multi-stage feature concatenation + **`BNNeck`** (`BatchNorm` > `LayerNorm`) with **Sub-Center ArcFace (`k=2–3`) + Dynamic Margins ($m_i \propto n_i^{-1/4}$)** (optionally linearly ramping margin scale over the first 5 epochs and giving the metric head **`10×` higher LR** than the backbone).
   - Attach a **multi-task coarse-category/species head** to regularize rare singleton identities.
3. **Iterative Open-Set Pseudo-Labeling (Weights Only — NEVER the Gallery!)**:
   - Retraining from pretrained init across `2–5` rounds on clean train + high-confidence test pseudo-labels (filtering out uncertain / `new_entity` predictions) is the largest universal open-set retrieval lever.
   - **Critical Rule**: Use pseudo-labels strictly to **train stronger encoder weights**—**never insert pseudo-labeled test embeddings into the kNN inference gallery**.
4. **Retrieval Calibration, Open-Set Thresholding & Re-Ranking Reality Check**:
   - **kNN + Margin-Free FC Logit Blending**: Pure cosine kNN over-predicts frequent training classes; blend top-K max cosine similarity with margin-free classification logits to neutralize frequency bias.
   - **Group-Calibrated Open-Set Thresholds**: Tune **per-category relative open-set thresholds on CV** (shifted by a single global offset calibrated to the expected `new_entity` rate) rather than one flat global cutoff.
   - **Supervised Rerankers Beat Unsupervised `QE / DBA / k-Reciprocal`**: In open-set long-tail re-ID with many singletons and unseen queries, classical unsupervised re-ranking (`Query Expansion`, `DBA`, `k-reciprocal`, `GNN re-ranking`) repeatedly degrades precision. Instead, ensemble via **L2-normalized embedding concatenation** (retraining final encoders on 100% of train after tuning via open-set CV) and add **2nd-stage GBDT or Siamese pairwise local-feature rerankers** over top-K kNN candidates.

---

## 2. 2D Small-Object Detection & Sequential Video Tracking

When detecting tiny objects (`20–50 px`) across video streams under recall-heavy multi-IoU `F_beta`:

1. **Strict Video/Sequence Group CV (Including 100% Empty Frames)**:
   - Random frame splits leak near-identical consecutive frames (`0.95+` fake CV). Always validate with **Leave-One-Video-Out** or disjoint **sequence GroupKFold**, **always evaluating on 100% of frames including background-only frames** (even if training uses only `0%–10%` background frames to focus positive gradients).
   - Standardize one exact official `F_beta` / IoU-sweep script across all experiments before tuning thresholds.
2. **Train-Time High-Resolution / Tiled Scaling (NEVER Inference-Only Extreme Upscaling!)**:
   - Small objects require high internal feature density: either train at `2×–3×` native resolution, run **tiled overlapping training (`320–640px` crops) → full-frame high-res inference**, or convert the early stem stride `2 → 1` (`P2/4` feature grid) to boost anchor density without interpolation blur.
   - **Fatal Shakeup Trap**: Inferring a detector at `2×–2.5×` *higher* than its training resolution (or shrinking predicted boxes to chase a tighter Public LB clip) causes massive Private LB collapse!
3. **Seam-Free Small-Object Synthesis & Label Auditing**:
   - Prefer **Poisson blending + a Real-vs-Fake box filter** or **Background-Mix** (blending annotated frames with empty background frames) over naive seam-heavy copy-paste.
   - Inspect high-confidence OOF false positives across video sequences to recover systematically missed ground-truth annotations.
4. **2nd-Stage Ordinal-IoU Box Re-Scorer, Low-Threshold Pre-WBF & Lightweight Tracking**:
   - **Ordinal-IoU Box Classifier**: Whenever metrics match predictions in descending confidence order across multiple IoU thresholds (`0.30..0.80`), crop OOF boxes (`conf > 0.01`, `1.2×` padded square) and train a heavily regularized classifier (`dropout=0.7, drop_path=0.5`) on **ordinal binary IoU targets (`IoU > 0.3, 0.4, ..., 0.9`)**, averaging all bin probabilities as the calibrated box score.
   - **Pre-WBF Low-Confidence Retention**: Run heterogeneous detectors (`YOLOv5/YOLOX`, `Cascade R-CNN`, `CenterNet`) at very low `conf = 0.01–0.05` *before* **Weighted Boxes Fusion (WBF)** so weak overlapping detections reinforce each other, applying the recall-optimal `F_beta` threshold only *after* WBF.
   - **Temporal Video Propagation**: On sequential frames, prefer simple, low-DOF temporal boosts—**attention-area confidence boost** across `t+1..t+3` (`score += conf * IoU`), **track-level max-confidence propagation**, **Online Seq-NMS**, or **keypoint homography (`SuperPoint`/`ORB`) + tracker with optical-flow cut resets**—over fragile hand-tuned Kalman setups.

---

## 3. 2D Instance Segmentation Under Tiered Label Quality (`AP@IoU`)

When combining small expert-annotated sets (`DS1`), large noisy/sparse-labeled sets (`DS2`), and unlabeled domains (`DS3`):

1. **Spatial / Slide-Grouped CV Strictly on Expert Labels**:
   - Adjacent image tiles share objects and domain texture—random tile splits leak heavily. Always group CV by **whole slide / spatial coordinate block** and evaluate **strictly on expert-verified labels** (never validate on sparse/noisy labels!).
2. **Exploiting Tiered Labels & Domain Context**:
   - **2-Stage Noisy Pretrain → Clean Fine-Tune** (higher LR/light aug on noisy `DS2` → low LR/heavy aug/higher res/SWA on clean `DS1`) or **balanced batch mixing** (fixed clean:noisy ratio per minibatch) consistently outperforms naive concatenation.
   - Use spatial tile coordinates `(i, j)` to **pad `128px` of neighboring tile context** around every crop (splitting slides into disjoint spatial halves so padded borders never cross folds), and apply **cross-domain stain/style transfer** from unlabeled `DS3` domains during training.
   - Deduplicate near-identical GT polygons (`IoU > 0.99`) before training.
3. **Decoupled Detect → High-Res Mask Decode & Instance Ensembling**:
   - 2D instance detectors (`ViT-Adapter`, `DetectoRS`, `HTC`, `Cascade Mask R-CNN`, `RTMDet`, `YOLOv8-seg`) decisively beat pure semantic U-Net + watershed.
   - **Box-First WBF → High-Res Mask Head**: Fuse boxes across diverse 1-stage/2-stage detectors (adding an auxiliary mask head to 1-stage detectors boosts bbox mAP, and pure detectors can join WBF to calibrate box confidence!) and decode masks once at high resolution (`1440–2048px`) via a Mask R-CNN head or cropped U-Net. Stabilize with **fixed-LR EMA** and full `[-180°, 180°]` rotation with exact polygon→bbox recomputation.
   - **Mask Calibration & Pixel Voting (NEVER Mask Union!)**:
     - Score instances via `score = score_bbox * mean(mask[mask > 0.5])`, prune tiny area artifacts and known-exclusion-zone overlaps, and aggregate ensemble masks strictly by **pixel-level probability averaging / voting** (plus vote-quantile filtering)—**mask addition/union over-expands polygons below the IoU cutoff and causes catastrophic Private collapse**.
   - **The Public-LB Dilation & Low-Threshold Trap**: Blind morphological dilation (`cv2.dilate`) or ultra-low mask thresholds (`0.2` vs `0.5`) can massively inflate Public LB if Public splits share noisy/loose boundaries while devastating clean unseen Private domains—never tune mask dilation against Public LB!
