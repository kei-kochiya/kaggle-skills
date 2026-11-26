# Neural Policy & Value Architectures for Competitive Games

Choose model structure from the game's observation geometry and measured bottlenecks of a compact baseline. Grid CNNs, pooled MLPs, graph networks, and entity Transformers each excel in different regimes.

**Scope:** All channel widths, depths, player counts, and head shapes below are configurable templates—size them to your compute and CPU serving budgets. In partially observable games, always isolate each player's private information even when batching multiple actors through shared weights.

---

## 1. Variable-Entity Transformer with Global Scratch Tokens

When environments contain variable numbers of heterogeneous objects (nodes, units, items, tiles), represent the world as a sequence of typed entity tokens plus learned control tokens:

```text
  [Typed Entity Input Stems]
    ├── Static/Map Entities:   (B, N_s, D_s) ──> MLP/Embed ──> (B, N_s, D)
    ├── Dynamic/Unit Entities: (B, N_d, D_d) ──> MLP/Embed ──> (B, N_d, D)
    └── Global State Features: (B, 1,   D_g) ──> Linear    ──> (B, 1,   D)

  [Learned Control & Workspace Tokens]
    ├── P× Player Summary Tokens: (B, P, D)  (Macro state + slot embedding)
    ├── P× Actor Plan Tokens:     (B, P, D)  (Query anchors for policy heads)
    ├── P× Critic Value Tokens:   (B, P, D)  (Anchors for value/win-prob heads)
    └── K× Global Scratch Tokens: (B, K, D)  (Unconstrained message-passing registers)
                                      |
                                      v
                [Pre-Norm Residual Transformer Trunk]
                                      |
                     +----------------+----------------+
                     v                                 v
           [Structured Actor Heads]          [Multi-Player Critic Head]
```

**Why Global Scratch Tokens Help**: Learned register tokens (`K = 4..8`) carry no direct supervision but attend freely to all entities across every layer, acting as a global working memory for long-range coordination and stabilizing deep residual gradients.

---

## 2. Single-Pass Multi-Player Trunk (Full-Observability Speedup)

In fully observable $P$-player games, instead of running $P$ separate perspective-rotated forward passes per environment step, project all entities once and attach per-player control tokens to compute policies and values for all $P$ players in **one forward pass** ($P\times$ rollout throughput). *(For partially observable games, run separate per-seat observations or apply strict per-player visibility masks so private state never leaks across seats.)*

```python
import torch
import torch.nn as nn

class SinglePassMultiPlayerTransformer(nn.Module):
    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        global_dim: int,
        player_dim: int,
        n_players: int = 2,
        n_scratch: int = 4,
        embed_dim: int = 256,
        depth: int = 8,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.static_proj = nn.Sequential(nn.Linear(static_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.dynamic_proj = nn.Sequential(nn.Linear(dynamic_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.global_proj = nn.Linear(global_dim, embed_dim)
        self.player_proj = nn.Linear(player_dim, embed_dim)

        self.player_slot_emb = nn.Parameter(torch.randn(1, n_players, embed_dim) * 0.02)
        self.plan_tokens = nn.Parameter(torch.randn(1, n_players, embed_dim) * 0.02)
        self.value_tokens = nn.Parameter(torch.randn(1, n_players, embed_dim) * 0.02)
        self.scratch_tokens = nn.Parameter(torch.randn(1, n_scratch, embed_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, static_feats, dynamic_feats, global_feats, player_feats, key_padding_mask=None):
        B = static_feats.shape[0]
        s_tok = self.static_proj(static_feats)
        d_tok = self.dynamic_proj(dynamic_feats)
        g_tok = self.global_proj(global_feats).unsqueeze(1)
        p_tok = self.player_proj(player_feats) + self.player_slot_emb
        ctrl = torch.cat(
            [
                p_tok,
                g_tok,
                self.scratch_tokens.expand(B, -1, -1),
                self.plan_tokens.expand(B, -1, -1),
                self.value_tokens.expand(B, -1, -1),
            ],
            dim=1,
        )
        seq = torch.cat([s_tok, d_tok, ctrl], dim=1)
        return self.norm(self.trunk(seq, src_key_padding_mask=key_padding_mask))
```

---

## 3. Factorized Structured Action Heads

When an entity chooses **(1) whether to act**, **(2) which variable target entity to interact with**, and **(3) a bounded continuous/integer quantity**, factorize the policy hierarchically:

1. **Binary Activation Gate**: $\pi_{\text{act}}(1 \mid i, p) = \sigma(\mathbf{W}_{\text{act}}[\mathbf{h}_i \,\|\, \mathbf{z}_p] + b)$.
2. **Pointer / Query–Key Target Selection**: Handles variable target counts cleanly without fixed output classes:
   $$\text{logits}_{\text{target}}(i \to j) = \frac{\mathbf{Q}_i \cdot \mathbf{K}_j}{\sqrt{d}} + \mathbf{M}_{\text{legal}}(i, j)$$
3. **Truncated Discretized Logistic Mixture Quantity Head**: Models multimodal integer/continuous allocations over `[min_qty, max_qty]` cleanly with far fewer parameters than large flat categorical heads:

```python
import torch.nn.functional as F

class LogisticMixtureQuantityHead(nn.Module):
    def __init__(self, embed_dim: int = 256, n_mixtures: int = 8):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_mixtures * 3),
        )

    def forward(self, source_hidden, target_hidden, min_qty, max_qty):
        params = self.mlp(torch.cat([source_hidden, target_hidden], dim=-1))
        pi_logits = params[..., : self.n_mixtures]
        means_norm = torch.sigmoid(params[..., self.n_mixtures : 2 * self.n_mixtures])
        log_scales = torch.clamp(params[..., 2 * self.n_mixtures :], min=-3.0, max=3.0)

        span = torch.clamp(max_qty - min_qty, min=1.0).unsqueeze(-1)
        means = min_qty.unsqueeze(-1) + means_norm * span
        return F.softmax(pi_logits, dim=-1), means, torch.exp(log_scales)
```

---

## 4. Multi-Player Zero-Sum & Exclusive-Winner Critic Heads

- **Exclusive-Winner Softmax Critic ($P$ Players)**: Pass each player's value token through a shared scalar MLP, mask eliminated players to `-1e9`, and apply `softmax` across players to guarantee $\sum_p \hat{p}_p = 1$.
- **Zero-Sum Paired Critic (2-Player With Score Differential)**: Combining relative score difference with learned phase-dependent weights $(a\tau + b)$ gives a calibrated zero-sum baseline $V_0 + V_1 = 0$:
  $$z_i = f_\theta(o_i) + (a\tau + b)\,\Delta m_i, \qquad V_i = \tanh\!\left(\frac{z_i - z_{1-i}}{2}\right)$$
  *(For games with draws, score-based objectives, or general-sum payoffs, output separate outcome/return heads matching the actual scoring rule rather than forcing an exclusive-winner softmax.)*

```python
class MultiPlayerWinProbCritic(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1))

    def forward(self, value_tokens, alive_mask):
        # value_tokens: (B, P, D), alive_mask: (B, P) bool
        logits = self.mlp(value_tokens).squeeze(-1).masked_fill(~alive_mask, -1e9)
        return F.softmax(logits, dim=-1)
```

---

## 5. Relational Edge-Attention

When pairwise relationships between entities (travel time, line-of-sight obstruction, graph distance, resource compatibility) dominate tactics, inject pairwise edge features $\mathbf{E} \in \mathbb{R}^{B \times N \times N \times D_e}$ directly as per-head attention biases. Compact edge-attention Transformers (`2M–5M` params) frequently match much larger plain Transformers under modest compute budgets:

```python
class RelationalEdgeAttention(nn.Module):
    def __init__(self, embed_dim: int = 256, n_heads: int = 8, edge_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.edge_proj = nn.Linear(edge_dim, n_heads)

    def forward(self, x, edges, valid_key_mask=None):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores + self.edge_proj(edges).permute(0, 3, 1, 2)

        if valid_key_mask is not None:
            scores = scores.masked_fill(~valid_key_mask.unsqueeze(1).unsqueeze(2), -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.out_proj(out)
```

---

## 6. 2D Rotary Position Embeddings (2D RoPE)

For entities embedded in continuous or discrete 2D space $(x_i, y_i)$, rotating separate channel partitions by $x$ and $y$ makes $\mathbf{q}_i \cdot \mathbf{k}_j$ depend natively on relative displacement $(\mathbf{x}_i - \mathbf{x}_j)$. When a board contains **separate non-contiguous coordinate frames** (e.g., two disconnected player maps plus non-spatial global tokens), apply 2D RoPE **only between tokens sharing the same physical coordinate frame**, leaving cross-frame and global channels unrotated:

```python
def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

def apply_2d_rope(q, k, pos_x, pos_y, theta_base: float = 1000.0):
    # q, k: (B, N, H, D) with D divisible by 4; pos_x, pos_y: (B, N)
    d_half = q.shape[-1] // 2
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, d_half, 2, device=q.device).float() / d_half))

    ang_x = torch.einsum("bn,d->bnd", pos_x.float(), inv_freq)
    ang_y = torch.einsum("bn,d->bnd", pos_y.float(), inv_freq)
    rot_x = torch.cat([ang_x, ang_x], dim=-1).unsqueeze(2)
    rot_y = torch.cat([ang_y, ang_y], dim=-1).unsqueeze(2)

    qx, qy = q[..., :d_half], q[..., d_half:]
    kx, ky = k[..., :d_half], k[..., d_half:]

    q_rot = torch.cat([qx * rot_x.cos() + rotate_half(qx) * rot_x.sin(), qy * rot_y.cos() + rotate_half(qy) * rot_y.sin()], dim=-1)
    k_rot = torch.cat([kx * rot_x.cos() + rotate_half(kx) * rot_x.sin(), ky * rot_y.cos() + rotate_half(ky) * rot_y.sin()], dim=-1)
    return q_rot, k_rot
```

---

## 7. Deterministic Forward-Horizon State Projections

Whenever parts of the environment evolve deterministically from currently visible state (e.g., in-transit arrivals, scheduled production, cooldown expiration, or predictable boundary movement), explicitly forward-simulate multi-horizon projections $H \in \{1, 2, 5, 10, \dots\}$ and append them to each entity's feature vector:
$$\hat{S}_i(t + H) = \text{SimulateVisibleDynamics}(S(t), H)$$
Baking known deterministic arrivals and resource projections into the observation eliminates temporal blindness, enables synchronized multi-entity timing without heavy recurrent memory, and strictly respects partial observability (never reading hidden private state or future RNG).
