---
name: kaggle-scientific-optimization-playbook
description: >-
  Universal competitive playbook for scientific ML, physics-coupled inverse problems, hierarchical ontology prediction, cross-modal image-to-signal digitization, Best-of-K 3D structure portfolios, combinatorial geometric optimization, and autonomous multi-LLM engineering factories. Use when fusing known forward physics operators with deep learning, predicting on DAG hierarchies, digitizing 1D signals from images, allocating Best-of-K candidate slots, solving 2D/3D packing or NP-hard search with GPU/SIMD relaxation, or orchestrating multi-agent optimization workflows.
---

# Kaggle Scientific ML, Inverse Physics & Combinatorial Optimization Playbook

A strictly universal, cross-competition runbook distilled from gold-medal solutions across **inverse physics & exoplanet spectroscopy**, **protein function DAG prediction**, **3D biomolecular folding**, **cross-modal image-to-signal & text-to-SVG generation**, **2D geometric packing**, **exact program compression**, and **autonomous multi-LLM optimization factories**. Competition-specific formulas and constants stay strictly in `Handbook/`.

---

## 1. Inverse Physics & Forward-Operator Deep Learning

Refer to [`references/physics_inverse_and_cross_modal.md`](./references/physics_inverse_and_cross_modal.md) for implementations.

Whenever a known forward physical simulator $F: y \mapsto x$ exists (PDEs, wave propagation, optics, rendering):
1. **Port $F$ to `torch.compile` / Custom CUDA First (`100×` Speedup)**:
   - Audit exact precision (`float32` vs `float64`), 0-based vs 1-based grid indexing, boundary quirks, and exact symmetry axes (odd/even sensor grids break naive horizontal flipping!).
2. **Choose Among Five Proven Physics–DL Fusion Paradigms**:
   - **Iterative Forward-Simulated Pseudo-Labeling**: Predict $\hat{y}$ on test $\to$ run exact $x_{\text{sim}} = F(\hat{y}) \to$ retrain on exact physics pairs $(x_{\text{sim}}, \hat{y})$ (zero pseudo-label noise!).
   - **Regularized Physics Test-Time Optimization**: Refine a neural warm-start via $\min_y p(y) + \|F(y) - x\|^2$ using **explicit structural priors $p(y)$** (TV $p<1$, truncated GP SVD) — **never run unregularized FWI** (diverges into cycle-skipping local minima!).
   - **Per-Sample Network-Weight TTO**: Optimize network weights $\theta$ per test sample on $\|F(M_\theta(x)) - x\|_1$ (uses the neural architecture as an implicit prior).
   - **Gradient-Free Fixed-Point Residual Iteration**: $x_{k+1} = x_k - \lambda (F(M(x_k)) - x)$ when $F$ is non-differentiable or expensive to backprop.
   - **2-Stage Physics-Residual Correction Network**: Feed `[x, x - F(y_1), y_1]` into Stage 2 to predict $\delta y = y - y_1$.
3. **Non-Aligned Domain Architectures, GLL Uncertainty Calibration & L1/MAE Ensembling**:
   - When input and output lack pixel-to-pixel alignment, discard pixel-aligned UNet skips in favor of **side-by-side spatial canvases + pure ViT+RoPE encoder–decoders** + **`MuonWithAuxAdam`**.
   - Under **Gaussian Log-Likelihood (GLL)**, **never submit uncalibrated Bayesian/GP/ensemble $\sigma$**: enforce hard **`MIN_SIGMA` floors** + boundary/fit-residual fallback inflation, use **FP64** for ill-conditioned physics+polynomial solvers, separate mean ($\mu$) fitting from out-of-fold per-channel $\sigma$ calibration (combining epistemic spread + **cross-method disagreement `|pred_A - pred_B|`**), and test **Median-Thresholded Ensemble Loss** (`relu(loss - median)`) to prevent likelihood ensemble collapse.
   - Under **MAE**, ensemble via exact **median (`torch.quantile(stack, 0.5)`)** across models/checkpoints (never implement weighted median by repeating arrays, which collapses to unweighted median).

---

## 2. Cross-Modal Digitization & Differentiable Format-Constrained Generation

1. **Single-Step High-Res Rectification & Extreme Horizontal Resolution**:
   - Estimate rotation + indexed grid keypoints/lines at low resolution, then apply **one single `grid_sample` / `INTER_LANCZOS4` remap directly to the raw high-res image** (eliminates double-interpolation blur for `+2.5 dB` gains).
   - Maximize horizontal (temporal) width (`2×–4×` upscaling, or horizontal-only U-Net upsampling after `1/8` scale). If skipping rectification entirely, pair a 2-pass region detector with **embedding-based (`ArcFace`) OOD gating** to switch between OOD-robust and ID-specialist digitizers.
2. **Sub-Pixel Signal Decoding & Fourier Resampling**:
   - Replace integer `argmax` with **direct 1D regression**, **sparse 2-px fractional masks**, **column-wise JSD / local Gaussian `CoordLoss`**, **parabolic peak refinement**, or **temperature/top-K soft-argmax**.
   - Resample band-limited waveforms in the **Fourier domain (`scipy.signal.resample` / `resample_torch`)** instead of linear interpolation, and audit whether training ground truth was resampled from a cleaner native-rate source.
3. **Lag-Aligned Signal-Space Blending, Weighted Median & Gated Physics**:
   - **Never average unaligned 1D signals across diverse models** (acts as a low-pass peak-blurring filter!) — cross-correlation align within metric lag tolerance *before* blending, and prefer **Weighted Median** over Weighted Mean to suppress outlier spikes.
   - Fuse stacked simultaneous channels vertically in decoder skip connections, and apply physical conservation laws (e.g., Einthoven's Law) **only when residuals fall below a tight gate**.
4. **Differentiable Metric-Filter Surrogates, Format-Aware Fine-Tuning & Async `GPU0→CPU→GPU1` Pipelining**:
   - When an evaluation metric applies non-differentiable filters (JPEG, crop, blur, FFT) before VLM/CLIP scoring, build a **differentiable PyTorch mirror** for end-to-end test-time optimization (prefer straight-line polygons over unstable Bézier curves in `diffvg`), and backpropagate downstream format-conversion fidelity (`-LPIPS(raw, converted)`) via **`DRaFT-LV`**.
   - Overlap heavy CPU conversion/vectorization asynchronously between generator (`cuda:0`) and VLM reranker (`cuda:1`), and prefer simple **binary VLM yes/no logit scoring** over noisy multi-question synthetic VQA at test time.

---

## 3. Hierarchical DAG Ontologies & Best-of-K Structural Portfolios

1. **Weak/Electronic Labels as *Input Features*, Never as Ground-Truth Targets**:
   - On scientific benchmarks with abundant noisy automated annotations alongside sparse experimental labels, training directly on noisy labels degrades generalization. Encode weak annotations, evidence counts, and provenance codes as **input features** into a Stage-2 DAG GCN / Transformer / Learning-to-Rank stacker, paired with **frozen foundation embeddings + alignment kNN + curated literature embeddings** and **prospective temporal-release CV**.
   - Enforce DAG hierarchy via **conditional $P(\text{child} \mid \ge 1 \text{ parent})$ predictors + noisy-OR/product channels + bidirectional parent/child smoothing** or differentiable **Max Constraint Modules (`MCM`)**, with **dampened metric weights** (`1 + w` or capped `sqrt(w)`).
2. **Best-of-K Generation Portfolios (Never Mode-Average Distinct 3D Conformations!)**:
   - Under `Best-of-K` structural metrics (e.g., `Best-of-5 TM-score`), coordinate-space averaging across distinct folds collapses valid modes (average only within `< 2 Å` RMSD).
   - Allocate the $K$ slots explicitly across **orthogonal paradigms & cluster-disjoint templates**, inject soft biophysical priors via **minimal `LinearNoBias` residuals** into foundation Pairformers (avoid heavy new modules), train **multimodal / CUDA-accelerated pairwise template rerankers**, handle ultra-long sequences via **homomer deduplication + Kabsch SVD overlap chunking**, strip crystallization/buffer artifacts, and enforce **deterministic kernels** during ablations.

---

## 4. Combinatorial Geometric Optimization & Autonomous Multi-LLM Factories

Refer to [`references/combinatorial_packing_and_agent_factories.md`](./references/combinatorial_packing_and_agent_factories.md) for implementations.

1. **Combinatorial & Geometric Packing Playbook**:
   - **Fixed-Box Feasibility Relaxation + Discrete Shrink > Joint Container Minimization**: Fix container size $L - \delta$, minimize overlap/violation energy to zero, then step-shrink $L$ (reserve joint container optimization strictly for final `legalize`).
   - **Precomputed Collision LUTs (`30k` Relaxations/s)**: Replace inline polygon intersection with precomputed **CUDA 3D Minkowski separation LUTs** or **convex-part SAT LUTs** feeding batched GPU L-BFGS / `float32` SIMD PBD relaxation (`4e-7` safety gap), and **always legalize against the host's exact precision/geometry engine** (`Decimal`/`Shapely`).
   - **Asymptotic Bulk Cores, Symmetry & Cross-$N$ Transfer**: Solve the infinite periodic crystal first (check **non-perpendicular** lattices!), freeze or merge interior bulk cores into a single rigid hull, exploit exact `180°` symmetry, and transfer structures across $N \leftrightarrow N\pm 1$ via buffer-zone removal, corner transplants, and aspect-ratio perturbations.
2. **Autonomous Multi-LLM Optimization Factories, Exact Compression DP & Global Budget Packing**:
   - **Blank-Slate Architectural Rewrites (`10×` Gain) >> Local Pruning**: Force agents to price multiple paradigms from scratch in clean empty sandboxes when greedy refinement stalls, pair **vertical deep-thinking exploration** with **horizontal proof-verified idiom registries (`md5` + votes + `negative_knowledge.md`)**, and trigger targeted LLM hints via automated AST diagnostics.
   - **Optimize Strictly After Downstream Transforms (Compression/Escaping)**: Whenever final score depends on compressed or escaped byte length, evaluate every candidate *after* the full downstream transform (including "DRY" repetition when it shrinks entropy); solve discrete subproblems (Huffman lengths, escaped-literal LZ77) with **exact DP** and outer assignments (variable renaming) via liveness-graph Optuna → SA.
   - **Transactional Validation (`backup → validate → promote/restore`) + Fresh OOD Stress Tests**: LLM-claimed improvements fail hidden tests frequently without independent reference consensus and fresh stress generators.
   - **Multiple-Choice Knapsack (`MCKP`) Final Assembly**: Whenever bundling $N$ independent models/tasks under a hard global Kaggle runtime cap, retain Pareto `(score, runtime)` candidates per task and solve an **exact MCKP** to maximize total score inside the runtime ceiling.

---

## 5. References & Case Studies
- [Physics Inversion, Cross-Modal Digitization, DAG Stacking & Best-of-K Portfolios](./references/physics_inverse_and_cross_modal.md)
- [Combinatorial Packing, Precomputed Collision LUTs & Autonomous Multi-LLM Factories](./references/combinatorial_packing_and_agent_factories.md)
- Case Studies: [`CAFA 6`](../../../Handbook/multimodal/cafa-6-protein-function.md) · [`Stanford RNA 3D-2`](../../../Handbook/multimodal/stanford-rna-3d-folding-2.md) · [`PhysioNet ECG`](../../../Handbook/multimodal/physionet-ecg-digitization.md) · [`Yale FWI`](../../../Handbook/multimodal/yale-geophysical-waveform-inversion.md) · [`Ariel 2025`](../../../Handbook/multimodal/ariel-data-challenge-2025.md) · [`Drawing with LLMs`](../../../Handbook/multimodal/drawing-with-llms.md) · [`Santa 2025`](../../../Handbook/optimization/santa-2025-tree-packing.md) · [`Google Code Golf 2025`](../../../Handbook/optimization/google-code-golf-2025.md) · [`NeuroGolf 2026`](../../../Handbook/optimization/neurogolf-2026.md)
