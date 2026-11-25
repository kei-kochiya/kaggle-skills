# Dense Foundation Backbones, Patch Heads & Training Discipline

Universal architectural patterns proven across both dense image regression and segmentation competitions.

---

## 1. CLS-Free Patch-Token Head (`Conv2d(1×1)`)

When using `DINOv3` / `DINOv2` ViTs for spatial regression (counting, biomass, density) or direct segmentation, patch tokens preserve spatial locality far better than the global `[CLS]` token:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DensePatchHead(nn.Module):
    """Minimalist patch-token head for spatial-sum regression or binary/multi-class segmentation."""
    def __init__(self, embed_dim: int, num_targets: int, mode: str = "density_sum"):
        super().__init__()
        self.mode = mode
        # 1x1 conv avoids small-N instability / NaN divergence seen with 3x3 convs
        self.proj = nn.Conv2d(embed_dim, num_targets, kernel_size=1)

    def forward(self, patch_tokens: torch.Tensor, grid_hw: tuple[int, int], out_hw: tuple[int, int] | None = None):
        # patch_tokens: [B, H_p * W_p, C] (CLS and register tokens already stripped)
        b, n, c = patch_tokens.shape
        h_p, w_p = grid_hw
        feat = patch_tokens.transpose(1, 2).reshape(b, c, h_p, w_p)
        logits = self.proj(feat)  # [B, T, H_p, W_p]

        if self.mode == "density_sum":
            # Non-negative per-patch density summed over spatial grid
            return F.softplus(logits).sum(dim=(2, 3))  # [B, T]
        elif self.mode == "segmentation":
            if out_hw is not None:
                logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
            return logits  # [B, T, H, W]
        raise ValueError(f"Unknown mode: {self.mode}")
```

### Multi-Layer Pyramid Extension
When objects span multiple scales, extract intermediate transformer blocks (e.g., 4 evenly spaced layers such as `[4, 11, 17, 23]` on a 24-block ViT-Large, or `[23, 31, 39]` on ViT-7B) and fuse via a `DPTHead` or ViTDet `SimpleFeaturePyramid` (`[2×, 1×, 0.5×, 0.25×]` scale branches) rather than relying solely on the final layer.

---

## 2. Two-Stage Partial Unfreezing & Differential Learning Rates

Prevent catastrophic forgetting of self-supervised representations on small/medium datasets:

```python
def configure_backbone_groups(model, head_lr: float = 2e-4, backbone_lr_mult: float = 0.1,
                              unfreeze_fraction: float = 0.5, weight_decay: float = 1e-2):
    """Freeze lower transformer blocks and apply differential LR to unfrozen top blocks."""
    blocks = list(model.backbone.blocks)
    cut = int(len(blocks) * (1.0 - unfreeze_fraction))
    for p in model.backbone.parameters():
        p.requires_grad = False
    for blk in blocks[cut:]:
        for p in blk.parameters():
            p.requires_grad = True

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "backbone" in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "backbone" not in n]
    return [
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        {"params": backbone_params, "lr": head_lr * backbone_lr_mult, "weight_decay": weight_decay},
    ]
```

**Recommended Schedule**:
1. **Stage 1 (Epochs 1–5/10)**: Train head only (`backbone` frozen) so randomly initialized heads do not corrupt pretrained patch attention.
2. **Stage 2**: Unfreeze top `20%–50%` of blocks (or add QKV LoRA on giant backbones) with `0.1×` backbone LR (or `AdamWScheduleFree`) + SWA over final/top checkpoints.

---

## 3. Hybrid Regression + Interval Classification (UEPNet Loss)

Combining continuous regression with coarse quantile/interval classification stabilizes gradients on heavy-tailed targets:

```python
class HybridRegIntervalLoss(nn.Module):
    def __init__(self, borders_per_target: list[list[float]], weights: list[float], cls_weight: float = 0.3):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.borders = [torch.tensor(b, dtype=torch.float32) for b in borders_per_target]
        self.cls_weight = cls_weight

    def forward(self, pred_reg: torch.Tensor, pred_cls_list: list[torch.Tensor], target: torch.Tensor):
        # pred_reg: [B, T], pred_cls_list: T tensors of [B, K_bins], target: [B, T]
        reg_loss = (F.smooth_l1_loss(pred_reg, target, reduction="none") * self.weights).sum(dim=1).mean()
        cls_loss = 0.0
        for i, logits in enumerate(pred_cls_list):
            bins = torch.bucketize(target[:, i], self.borders[i].to(target.device))
            cls_loss = cls_loss + self.weights[i] * F.cross_entropy(logits, bins)
        return reg_loss + self.cls_weight * cls_loss
```

---

## 4. Per-Target Checkpoint Selection

In multi-target regression, easy/dominant targets frequently peak and begin overfitting while harder minority targets are still converging:
- After every epoch, compute validation error **per target column** (`val_loss_t`, `val_r2_t`).
- Maintain `best_ckpt[t]` independently for each target `t`, or export one checkpoint per convergence group (`early_peak` targets vs. `late_peak` targets) and assemble predictions target-wise at inference.
