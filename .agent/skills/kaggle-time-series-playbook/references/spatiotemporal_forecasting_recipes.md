# Spatiotemporal Forecasting Recipes

Battle-tested PyTorch loss, target-integration, context-node, and ensembling primitives for multi-agent and continuous horizon forecasting.

---

## 1. Heteroscedastic `GaussianNLLLoss` With Velocity & Acceleration Consistency

Automatically downweights high-uncertainty late horizons while enforcing smooth kinematics:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class KinematicGaussianNLLLoss(nn.Module):
    def __init__(self, vel_weight: float = 0.2, acc_weight: float = 0.1, eps: float = 1e-3):
        super().__init__()
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self.eps = eps

    def _masked_gnll(self, mean: torch.Tensor, raw_var: torch.Tensor,
                     target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        var = F.softplus(raw_var) + self.eps
        nll = 0.5 * (torch.log(var) + (target - mean) ** 2 / var)
        return (nll * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, pred_xy: torch.Tensor, pred_var: torch.Tensor,
                target_xy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Shapes: [B, P, T, 2], mask: [B, P, T, 1]
        loss = self._masked_gnll(pred_xy, pred_var, target_xy, mask)
        if pred_xy.shape[2] > 1 and self.vel_weight > 0:
            v_pred = pred_xy[:, :, 1:] - pred_xy[:, :, :-1]
            v_true = target_xy[:, :, 1:] - target_xy[:, :, :-1]
            v_mask = mask[:, :, 1:] * mask[:, :, :-1]
            v_var = pred_var[:, :, 1:] + pred_var[:, :, :-1]
            loss = loss + self.vel_weight * self._masked_gnll(v_pred, v_var, v_true, v_mask)
        if pred_xy.shape[2] > 2 and self.acc_weight > 0:
            a_pred = pred_xy[:, :, 2:] - 2 * pred_xy[:, :, 1:-1] + pred_xy[:, :, :-2]
            a_true = target_xy[:, :, 2:] - 2 * target_xy[:, :, 1:-1] + target_xy[:, :, :-2]
            a_mask = mask[:, :, 2:] * mask[:, :, 1:-1] * mask[:, :, :-2]
            loss = loss + self.acc_weight * (F.smooth_l1_loss(a_pred, a_true, reduction="none") * a_mask).sum() / a_mask.sum().clamp_min(1.0)
        return loss
```

---

## 2. Horizon-Decayed `TemporalHuber` Loss (Clean Numerator Normalization)

```python
class TemporalHuberLoss(nn.Module):
    def __init__(self, delta: float = 0.5, decay: float = 0.03):
        super().__init__()
        self.delta = delta
        self.decay = decay

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pred, target: [B, P, T, 2], mask: [B, P, T, 1]
        t_idx = torch.arange(pred.shape[2], device=pred.device, dtype=pred.dtype)
        time_w = torch.exp(-self.decay * t_idx).view(1, 1, -1, 1)
        huber = F.huber_loss(pred, target, delta=self.delta, reduction="none")
        w = time_w * mask
        return (huber * w).sum() / w.sum().clamp_min(1.0)
```

---

## 3. Cumulative Step Integration & External Context-Node Attention

- **Cumulative Displacement Head**: Predict frame-to-frame deltas `dxy` (`[B, P, T, 2]`) and reconstruct `xy = last_obs_xy.unsqueeze(2) + torch.cumsum(dxy, dim=2)`.
- **Context-Node Injection**: Append global/target reference coordinates (e.g., destination point, landmark, market anchor) as an extra entity token `P + 1` with its own role embedding before entity self-attention, rather than repeating its coordinates as numeric columns on every entity. Entities that depend on the landmark attend to it; unrelated entities learn zero attention weight.

---

## 4. Early-Frame Origin Shift Augmentation

During training, sample an integer shift `shift ~ Uniform(0, max_shift)` (`shift = 0` at validation/test):
1. Slice input history to end `shift` steps earlier.
2. Prepend the `shift` skipped historical frames onto the front of the target trajectory (with valid masks for all entities).
3. Pass `shift` (normalized) as a static conditioning scalar so the decoder knows how many pre-origin steps precede the true forecast horizon.

---

## 5. Polar-Vector Trajectory Ensembling

When averaging curved 2D trajectories from diverse models, naive Cartesian averaging shrinks step lengths when models disagree slightly on heading angle. Decompose each step displacement $(\Delta x_m, \Delta y_m)$ into speed $s_m = \|\Delta \mathbf{p}_m\|_2$ and unit direction $\mathbf{u}_m = \Delta \mathbf{p}_m / \max(s_m, \epsilon)$, then average magnitudes and normalized directions separately:

$$\bar{s} = \sum_m w_m s_m, \qquad \bar{\mathbf{u}} = \frac{\sum_m w_m \mathbf{u}_m}{\left\|\sum_m w_m \mathbf{u}_m\right\|_2 + \epsilon}, \qquad \overline{\Delta \mathbf{p}} = \bar{s}\,\bar{\mathbf{u}}$$
