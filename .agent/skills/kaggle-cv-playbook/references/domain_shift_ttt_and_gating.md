# Domain Shift Validation, TTT/SSA, Dual-Threshold Gating & Matching Pipelines

Universal recipes for surviving severe train-to-test visual shift, asymmetric segmentation metrics, and compound visual correspondence tasks.

---

## 1. Embedding-Distance Easy/Hard Validation Diagnostic

When train and test domains shift unpredictably, complement `StratifiedGroupKFold` by auditing performance separately on in-distribution (`easy`) vs. out-of-distribution (`hard`) validation images:

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def split_easy_hard_val(train_emb: np.ndarray, val_emb: np.ndarray, k: int = 5):
    """Partition validation indices by mean k-NN distance to training embeddings."""
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(train_emb)
    dists, _ = nn.kneighbors(val_emb)
    mean_dist = dists.mean(axis=1)
    med = np.median(mean_dist)
    return mean_dist <= med, mean_dist > med  # easy_mask, hard_mask
```

- Track CV on both `easy` and `hard` slices. Models that gain strictly on `easy` while degrading `hard` carry high Private LB shakeup risk.

---

## 2. Two-Stage Test-Time Online Training (TTT) + Clean-Train Refit

Effective when the hidden test set is larger than `train` and sits in a shifted capture regime (lighting, site, sensor):

1. **Diverse Teacher Pseudo-Labels**: Predict hard/soft test targets using a diverse ensemble of Stage-1 models.
2. **Joint Pseudo-Train + SWA**: Train fresh student backbones on `full_train + pseudo_labeled_test`, averaging weights via **SWA** across the final epochs.
3. **Clean-Train Anchor Refit (Critical Guardrail)**: Fine-tune the SWA checkpoint for a few low-LR epochs strictly on the **original labeled training set** to eliminate systematic bias amplified by noisy pseudo-labels.
4. **Optional Round 2 / Blend**: Optionally regenerate cleaner pseudo-labels for a second backbone family (`Large → Base`), and blend Stage-1 with Stage-2 TTT (`0.2–0.4` Stage-1 + `0.6–0.8` TTT) to hedge against adaptation drift.

*Low-Compute Alternative — Significant Subspace Alignment (SSA)*: Precompute source feature PCA mean and top eigenvectors $(\mu_s, U_s)$ on training embeddings; at test time, adapt a lightweight linear projection between the frozen backbone and head so test feature subspace moments align to $(\mu_s, U_s)$ with zero backpropagation through the backbone.

---

## 3. Dual-Threshold Empty/Presence Gating (Asymmetric Segmentation Metrics)

Whenever an empty image scored correctly gives `1.0` and *any* false-positive mask drops the image score to `0.0`:

```python
import numpy as np

def dual_threshold_mask_gate(prob_map: np.ndarray, high_thr: float = 0.75,
                             min_high_pixels: int = 64, mask_thr: float = 0.35,
                             quantile_gate: tuple[float, float] | None = None) -> np.ndarray | None:
    """Separate empty-vs-present detection (high precision) from pixel delineation (high recall)."""
    if quantile_gate is not None:
        q, q_thr = quantile_gate
        if np.quantile(prob_map, q) < q_thr:
            return None
    if int((prob_map >= high_thr).sum()) < min_high_pixels:
        return None
    mask = (prob_map >= mask_thr).astype(np.uint8)
    return mask if mask.sum() > 0 else None
```

- Tune `(high_thr, min_high_pixels, mask_thr)` strictly on a test-aligned validation split by measuring exact net lift (`tp_gain - fp_loss`) over the constant-empty baseline.
- If the metric penalizes excess instances (e.g., `G / max(P, G)` in Hungarian `oF1`), merge nearby/overlapping connected components (`IoU >= 0.1` or `IoMin >= 0.3`) rather than emitting fragmented masks.

---

## 4. Structured Detection → Retrieval → Keypoint Verification Pipeline

For compound visual correspondence, re-identification, or copy-move forensics where end-to-end segmentation collapses under domain shift:

1. **Region Extraction & Distractor Masking**: Detect valid sub-regions (`YOLO` / prompted `SAM`) and explicitly mask out repetitive non-target distractors (text captions, scale bars, plot axes/legends) that trigger false matches.
2. **Fast Pairwise Shortlisting**: Use supervised contrastive embeddings (`SupCon` loss on `NextViT` / `ConvNeXt` / `DINOv3`) or aspect/area filters to shortlist top-$K$ candidate pairs in milliseconds.
3. **High-Recall Keypoint Extraction**:
   - On low-contrast textures (medical/microscopy/blots), upscale small regions (`2×–4×`) and lower SIFT `contrast_threshold` (`0.001` vs default `0.04`), or use learned `ALIKED` / `SuperPoint` + `LightGlue` (**disable LightGlue early stopping** on sparse matches).
   - Always test **mirror/flip hypotheses** if mirrored duplication is plausible.
4. **Geometric Verification & Mask Refinement**: Verify matches with Affine/Homography **RANSAC / MAGSAC++** (`min_inliers` & `min_inlier_ratio`, rejecting impossible scale/shear matrices), merge clusters with matching affine transforms via Union-Find, and refine masks via bidirectional reprojection error (`<= mean + 2σ`).

---

## 5. Multi-Scene 3D SfM, Foundation Matching & Quadratic GPU Balancing

When reconstructing camera poses `(R, t)` from unstructured flat collections mixing multiple visually similar scenes and distractors:

1. **Hybrid 3D-Foundation + High-Resolution Sparse Keypoints**: Pair 3D-foundation matchers (`MASt3R`) with high-resolution sparse keypoint detectors (`ALIKED`/`SuperPoint`/`RDD`) matched through the foundation local feature head—combining 3D geometric symmetry/rotation robustness with sub-pixel keypoint accuracy.
2. **Multi-Retriever Shortlisting & Cycle Consistency**: Combine orthogonal global retrievers (`ASMK + SPoC + DINOv2 + ISC`), re-match high-res ROI crops / 5-tile quadrants on wide-baseline pairs, and drop pairs failing 3-view **cycle/loop-closure consistency** (`>30 px` error).
3. **Let Incremental Multi-Model SfM Split Clusters**: When match precision is high, skip fragile early pre-clustering (`Louvain` / farthest-point clustering) and let incremental multi-model **`pycolmap`** (wrapped in **retryable subprocesses** for segfault/OOM resilience) separate scenes and reject outliers; avoid global `GLOMAP` or unsupervised test-time matcher TTT on mixed multi-scene collections (amplifies cross-scene false bridges).
4. **Quadratic Multi-GPU Workload Sharding**: Pairwise matching scales as $\mathcal{O}(n_i^2)$ per scene—partition datasets across multiple GPUs by balancing **$\sum n_i^2$** rather than image count $\sum n_i$ to eliminate straggler idle time.
