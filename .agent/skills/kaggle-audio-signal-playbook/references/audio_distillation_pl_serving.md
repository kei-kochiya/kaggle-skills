# Cosine Embedding Distillation, Capped Soft PL, Rank Blending & Shared-Mel OpenVINO

Reusable code patterns for multi-label audio/signal competitions under domain shift and strict CPU budgets.

---

## 1. Phase-1 Cosine Embedding Distillation Head

Cosine alignment converges faster and transfers cleaner geometry than raw MSE when distilling frozen audio foundation embeddings (`D_teacher`, e.g., 1536) into lightweight spectrogram CNNs:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineEmbeddingDistillLoss(nn.Module):
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_emb: torch.Tensor, teacher_emb: torch.Tensor) -> torch.Tensor:
        s = F.normalize(self.proj(student_emb), dim=-1)
        t = F.normalize(teacher_emb.detach(), dim=-1)
        return (1.0 - (s * t).sum(dim=-1)).mean()
```

---

## 2. Label-Sum-Capped Soft Pseudo-Label Injection

Prevents frequent target-domain background classes from hijacking gradients during Noisy Student / soundscape mixup:

```python
import numpy as np

def cap_soft_label_sum(probs: np.ndarray, max_sum: float = 0.75, zero_below: float = 0.05) -> np.ndarray:
    """Denoise tail probabilities and cap total label mass per target-domain chunk."""
    y = np.where(probs >= zero_below, probs, 0.0).astype(np.float32)
    row_sum = y.sum(axis=-1, keepdims=True)
    scale = np.where(row_sum > max_sum, max_sum / np.maximum(row_sum, 1e-8), 1.0)
    return y * scale
```

**Sampling Rule**: Sample unlabeled recordings by `p(site) ∝ 1 / sqrt(count(site))`, and apply labeled-target injection (`max_sum <= 0.5`) and unlabeled-PL injection (`max_sum <= 0.75`) to **disjoint** focal samples within each minibatch.

---

## 3. Per-Column Percentile Rank Blending & Adaptive Temporal Smoothing

```python
import numpy as np
from scipy.stats import rankdata

def columnwise_rank_blend(pred_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Rank-normalize each class column independently across all test chunks before blending."""
    n_rows, n_cols = pred_list[0].shape
    out = np.zeros((n_rows, n_cols), dtype=np.float64)
    w_sum = sum(weights)
    for preds, w in zip(pred_list, weights):
        ranked = np.empty_like(preds, dtype=np.float64)
        for c in range(n_cols):
            ranked[:, c] = rankdata(preds[:, c]) / n_rows
        out += (w / w_sum) * ranked
    return out

def recording_temporal_smooth(chunks: np.ndarray, file_max_alpha: float = 0.20,
                              neighbor_alpha: float = 0.20) -> np.ndarray:
    """chunks: [T_chunks, C] probabilities for a single continuous recording."""
    # 1. File-max anchoring
    fmax = chunks.max(axis=0, keepdims=True)
    out = (1.0 - file_max_alpha) * chunks + file_max_alpha * fmax
    # 2. Uncertainty-adaptive neighbor smoothing: smooth harder when local confidence is low
    if out.shape[0] >= 3:
        prev_c = np.vstack([out[:1], out[:-1]])
        next_c = np.vstack([out[1:], out[-1:]])
        nbr_mean = 0.5 * (prev_c + next_c)
        dyn_alpha = neighbor_alpha * (1.0 - out.max(axis=1, keepdims=True))
        out = (1.0 - dyn_alpha) * out + dyn_alpha * nbr_mean
    return out
```

---

## 4. Shared-Spectrogram Single-Graph CPU Export (OpenVINO IR)

When running multiple spectrogram CNNs under a tight CPU runtime cap, computing separate STFT/mel-spectrograms per model wastes 10%–20% of wall-clock time. Wrap all models sharing the same mel frontend into one `nn.Module` (`waveform [B, L] → shared LogMel → [m1, m2, ..., mk] → weighted average`) and export once to ONNX → **OpenVINO IR** (`ov.convert_model`), running independent frontends (e.g., native foundation ONNX vs. OpenVINO CNNs) across disjoint CPU thread lanes (`OMP_NUM_THREADS=2` × 2 workers on 4 vCPUs).
