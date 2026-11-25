# Physics Inversion, Cross-Modal Digitization, DAG Stacking & Best-of-K Portfolios

Concrete, reusable code patterns and checklists for scientific ML competitions.

---

## 1. Five Forward-Operator Inverse-Problem Loops

Given an accelerated forward physical simulator `F(y) -> x` (ported via `torch.compile` or custom CUDA) and a trained neural inverse model `M(x) -> y`:

```python
# Pattern A: Iterative Forward-Simulated Pseudo-Labeling (Zero Pseudo-Label Drift)
with torch.no_grad():
    y_hat = M(x_test)              # Predicted physical field on test manifold
    x_sim = F(y_hat)               # Exact forward simulation: (x_sim, y_hat) obeys true physics!
# Next epoch trains M on union of real (x_train, y_train) + exact simulated (x_sim, y_hat)

# Pattern B: Regularized Physics Test-Time Optimization (Warm-Start + Prior p(y))
y = M(x_obs).detach().clone().requires_grad_(True)
opt = torch.optim.LBFGS([y], lr=1.0, max_iter=50)
def closure():
    opt.zero_grad()
    data_loss = ((F(y) - x_obs) * time_weight).pow(2).mean()
    # Explicit structural prior p(y) is MANDATORY (unregularized FWI diverges!)
    prior = tv_p_norm(y, p=0.7)    # Sharp layered media (or truncated GP SVD for smooth media)
    loss = data_loss + lam * prior
    loss.backward()
    return loss
opt.step(closure)

# Pattern C: Gradient-Free Fixed-Point Residual Iteration (Zero Simulator Backprop!)
x_cur = x_obs.clone()
for _ in range(5):
    y_cur = M(x_cur)
    x_cur = x_cur - step_lambda * (F(y_cur) - x_obs)
y_final = M(x_cur)
```

- **2-Stage Residual Net**: Train Stage 2 on `[x_obs, x_obs - F(y_1), y_1]` to predict `delta_y = y_true - y_1`.
- **Non-Aligned Layout Rule**: When input dimensions `(sources, time, receivers)` do not align spatially with output `(depth, width)`, reshape inputs into a single side-by-side 2D canvas `(1, H', W')` for a pure **ViT+RoPE encoder–decoder** instead of pixel-aligned UNet skips.
- **Exact MAE Median Ensembling**: Always compute `torch.quantile(preds_stack, 0.5, dim=0)` (or true weighted quantile — never repeat elements `[a,a,a,b,b,b,c]`, which silently collapses to `median([a,b,c])`).

---

## 2. Cross-Modal Image-to-1D-Signal Digitization Checklist

1. **Single-Step High-Res Remapping**:
   - Predict rotation (`0/90/180/270°`) + indexed grid lines / keypoints on downsampled input, rescale predicted coordinates back to raw image pixel space, and call `F.grid_sample` / `cv2.remap(..., INTER_LANCZOS4)` **exactly once** on the raw high-res image (avoids double-interpolation blur).
2. **Sub-Pixel Extraction & Fourier Resampling**:
   - Prefer **direct 1D regression**, **sparse 2-px fractional masks**, **column-wise JSD / local Gaussian `CoordLoss`**, or **parabolic peak refinement** (`delta = (y_l - y_r) / (2*(y_l - 2*y_c + y_r))`) over integer `argmax`.
   - Resample digitized 1D signals to target `fs` via **Fourier resampling (`scipy.signal.resample`)**, never linear interpolation.
3. **Lag-Aligned Weighted-Median Signal Ensembling**:
   ```python
   # Align model m's waveform s_m to reference s_ref within metric lag tolerance ±max_lag
   lags = np.arange(-max_lag, max_lag + 1)
   best_lag = lags[np.argmax([np.dot(shift(s_m, d), s_ref) for d in lags])]
   s_m_aligned = shift(s_m, best_lag)
   s_m_aligned -= np.mean(s_m_aligned - s_ref)   # Remove vertical DC offset
   # Combine across aligned models via weighted median (rejects outlier spikes!)
   ```
4. **OOD-Gated No-Warp Fallback**: When skipping rectification, train an `ArcFace` image-type embedding gate (`cosine < 0.5`) to mask out ID-overfit specialists on degraded OOD photos.

---

## 3. Hierarchical DAG Stacking & Best-of-K Structural Portfolios

### DAG Multi-Label Stacking
- **Weak Annotations as Features**: Extract binary presence, distinct-evidence-code counts (`1..5+`), and qualifier flags from weak/electronic sources as **Stage-2 input channels** — keep training targets strictly experimental/clean.
- **Ontology-Conditional Training**: If an ontology is evaluated only on items with `≥ 1` positive label in that ontology, train its stacker strictly on items having `≥ 1` positive label to match $P(\text{term} \mid \text{ontology exists})$.
- **Bidirectional DAG Reconciliation**: Combine bottom-up True-Path max propagation with top-down conditional/noisy-OR channels inside a DAG GCN, and polish via `final(t) = mean(raw(t), max(children(t)), min(parents(t)))`.
- **Dampened Metric Loss Weights**: Replace raw uncapped information-accretion weights `w` with `1 + w` or `min(sqrt(w), 0.5)` to stabilize gradients on ultra-rare leaf nodes.

### Best-of-K 3D Conformation Portfolios
- **Slot Allocation**: Never average distinct 3D conformations in coordinate space (only average if Kabsch RMSD `< 2 Å`). Assign slots across distinct generative/template families + **cluster-disjoint templates** (`MMseqs 0.30` clusters) + **cross-pollinated hypotheses** (feed Model A's de novo output as an input template to Model B).
- **Minimal Soft Biophysical Residual**: Inject `[N, N]` secondary-structure / contact probabilities into pretrained Pairformer `z_init` via a single `nn.Linear(1, d_pair, bias=False)` — outperforms complex multi-block embedders or binned one-hots.
- **Long-Sequence Kabsch Chunking**: Deduplicate homomeric copies, chunk long chains (`448–512` tokens, `64–128` overlap), and stitch via **Kabsch SVD** + narrow linear (`±6`) or $C^2$-quintic smoothstep (`6t⁵ - 15t⁴ + 10t³`) overlap blending.
- **Ablation Hygiene**: Disable stochastic attention/MC dropout during local CV ablations so `±0.02–0.03` diffusion variance cannot mask real improvements.
