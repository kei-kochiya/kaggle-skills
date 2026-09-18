# Neural Policy & Value Architectures for Competitive Games 🧠

In complex simulation environments (e.g. multi-planet space conquest, grid RTS games, economic bidding), fixed-size convolutional or feedforward networks struggle with **variable entity counts**, **spatial permutation invariance**, and **multi-agent game dynamics**.

This reference guide details the architecture of **Entity-Centric Transformers**, **Single-Pass Multi-Player Policy Heads**, and **Continuous-Discrete Hybrid Action Decoders** distilled from top-tier competitive RL solutions.

---

## 1. Unified Multi-Entity Attention Architecture

Rather than flattening board states into rigid grids or fixed-length vectors, represent the game world as a variable-length sequence of distinct entity tokens processed by a deep residual Transformer trunk.

```
                     ENTITY TOKENIZATION & TRUNK DATA FLOW
                     
 [Observation Tensors]
   ├── Static Planets:   (B, N_p, D_p) ──> Linear/GELU ──> (B, N_p, 768)
   ├── Orbiting Planets: (B, N_o, D_o) ──> Linear/GELU ──> (B, N_o, 768)
   ├── Active Fleets:    (B, N_f, D_f) ──> Linear/GELU ──> (B, N_f, 768)
   └── Moving Comets:    (B, N_c, D_c) ──> Linear/GELU ──> (B, N_c, 768)
 
 [Special Control Tokens (Learned Embeddings)]
   ├── 4× Player Summaries: (B, 4, 768)  (Macro state + learned slot embedding)
   ├── 1× Global Board:     (B, 1, 768)  (Game clock, phase, active player count)
   ├── 4× Actor Plan:       (B, 4, 768)  (Query anchors for action decoding)
   ├── 4× Critic Value:     (B, 4, 768)  (Anchors for win-probability estimation)
   └── 4× Global Scratch:   (B, 4, 768)  (Unconstrained cross-attention scratchpad)
                                 |
                                 v
          [Concatenate on Entity Axis: (B, Total_Tokens, 768)]
                                 |
                 +---------------+---------------+
                 |  Pre-Norm Residual Trunk      |
                 |  • 38 Transformer Blocks      |
                 |  • 16 Self-Attention Heads    |
                 |  • 1536-d MLP Hidden Layer    |
                 +---------------+---------------+
                                 |
                 +---------------+---------------+
                 |                               |
                 v                               v
    [Multi-Player Actor Heads]       [Multi-Player Critic Head]
    (Joint distribution over         (Softmax win probability
     all players & entities)          over all 4 players)
```

### 1.1 The Role of Global Scratch Tokens
Scratch tokens are learned parameter vectors $\mathbf{S} \in \mathbb{R}^{K \times D}$ that have no direct supervisory loss or explicit physical meaning. They append to the input sequence and attend freely to all board entities across all 38 transformer blocks:
- Act as a **shared global workspace / bottleneck register** (similar to Perceiver / Transformer memory slots).
- Enable distant planets and fleet clusters to exchange situational awareness without requiring direct pairwise quadratic attention across all entity tokens.
- Significantly stabilize gradient flow through deep residual trunks.

---

## 2. Single-Pass Multi-Player Inference

In an $N$-player game, conventional MARL runs $N$ independent forward passes per time step (one for each player's private perspective). The **Single-Pass Multi-Player Trunk** computes policy distributions and value estimates for all active players in a **single forward evaluation**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinglePassMultiPlayerTransformer(nn.Module):
    def __init__(self, embed_dim=768, depth=38, n_heads=16, mlp_ratio=2.0):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Entity input projection stems
        self.planet_proj = nn.Sequential(nn.Linear(107, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))
        self.fleet_proj = nn.Sequential(nn.Linear(79, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))
        self.global_proj = nn.Linear(17, embed_dim)
        
        # Learned control & workspace tokens
        self.player_tokens = nn.Parameter(torch.randn(4, embed_dim))
        self.plan_tokens = nn.Parameter(torch.randn(4, embed_dim))
        self.value_tokens = nn.Parameter(torch.randn(4, embed_dim))
        self.scratch_tokens = nn.Parameter(torch.randn(4, embed_dim))
        
        # Pre-norm transformer trunk
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_dim=int(embed_dim * mlp_ratio))
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, obs_batch):
        B = obs_batch["planet_features"].shape[0]
        
        # 1. Project raw entities
        p_tokens = self.planet_proj(obs_batch["planet_features"]) # (B, N_p, 768)
        f_tokens = self.fleet_proj(obs_batch["fleet_features"])   # (B, N_f, 768)
        g_token = self.global_proj(obs_batch["global_features"]).unsqueeze(1) # (B, 1, 768)
        
        # 2. Expand special tokens across batch
        player_toks = self.player_tokens.unsqueeze(0).expand(B, -1, -1) + obs_batch["player_features_proj"]
        plan_toks = self.plan_tokens.unsqueeze(0).expand(B, -1, -1)
        val_toks = self.value_tokens.unsqueeze(0).expand(B, -1, -1)
        scratch_toks = self.scratch_tokens.unsqueeze(0).expand(B, -1, -1)
        
        # 3. Concatenate unified sequence
        seq = torch.cat([p_tokens, f_tokens, player_toks, g_token, scratch_toks, plan_toks, val_toks], dim=1)
        
        # 4. Pass through deep transformer trunk
        for block in self.blocks:
            seq = block(seq)
        seq = self.norm(seq)
        
        return seq
```

---

## 3. Decoupled Multi-Head Policy Architecture

For each active player $p$ and each eligible source planet $i$, the action space decomposes into three structured sub-actions:

```
                           ACTION DECODING CASCADE
                           
  [Source Planet Token h_i]  +  [Player Plan Token z_p]
                   |                      |
                   +----------+-----------+
                              |
                              v
                 [Bernoulli Launch Head]
              P(launch | i, p) = σ(W_l [h_i || z_p])
                              |
             +----------------+----------------+
             | (Launch = 0)                    | (Launch = 1)
             v                                 v
        [No Action]             [Query-Key Target Selection]
                                Q_i = W_Q h_i,  K_j = W_K h_j
                                Target Logits = (Q_i · K_j^T) / √d
                                               |
                                               v
                                [Target Representation V_target]
                                               |
                                               v
                                [Discretized Logistic Mixture]
                                8 Components over [3, Max_Ships]
```

### 3.1 Bernoulli Launch Decision
Computes whether source planet $i$ owned by player $p$ initiates a fleet launch:
$$z_{\text{launch}} = \mathbf{W}_{\text{launch}} [\mathbf{h}_i \,\|\, \mathbf{z}_p] + b_{\text{launch}}$$
$$\pi_{\text{launch}}(1 \mid i, p) = \sigma(z_{\text{launch}})$$

### 3.2 Scaled Dot-Product Target Attention Head
Instead of predicting target IDs through a static classification layer (which breaks if the map has variable planets), target selection is formulated as **attention matching**:
$$\mathbf{Q}_i = \mathbf{W}_Q [\mathbf{h}_i \,\|\, \mathbf{z}_p], \quad \mathbf{K}_j = \mathbf{W}_K \mathbf{h}_j$$
$$\text{logits}_{\text{target}}(i \to j) = \frac{\mathbf{Q}_i \cdot \mathbf{K}_j}{\sqrt{d}} + \mathbf{M}_{\text{mask}}(i, j)$$
where $\mathbf{M}_{\text{mask}}$ masks out the source planet itself and invalid targets.

### 3.3 Truncated Discretized Logistic Mixture Head
Fleet sizing requires predicting an integer count $s \in [3, S_{\text{available}}]$. Naive discretization into 500 bins adds massive parameter overhead, while single Gaussian heads suffer from poor tail modeling. Use a **Truncated Discretized Logistic Mixture Model** with 8 components:

```python
class LogisticMixtureFleetSizingHead(nn.Module):
    def __init__(self, embed_dim=768, n_mixtures=8):
        super().__init__()
        self.n_mixtures = n_mixtures
        # Outputs: 8 mixture weights, 8 means (normalized in [0,1]), 8 log-scales
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_mixtures * 3)
        )

    def forward(self, source_hidden, target_value, max_ships):
        B = source_hidden.shape[0]
        combined = torch.cat([source_hidden, target_value], dim=-1)
        params = self.mlp(combined)
        
        pi_logits = params[:, :self.n_mixtures]
        means_norm = torch.sigmoid(params[:, self.n_mixtures:2*self.n_mixtures])
        log_scales = torch.clamp(params[:, 2*self.n_mixtures:], min=-3.0, max=3.0)
        
        # Scale means to valid ship range [3, max_ships]
        means = 3.0 + means_norm * (max_ships.unsqueeze(-1) - 3.0)
        scales = torch.exp(log_scales)
        
        return F.softmax(pi_logits, dim=-1), means, scales
```

---

## 4. Multi-Player Softmax Value Head

In multi-agent simulation games, scalar value heads ($V(s) \in \mathbb{R}$) fail to account for relative survival and non-zero-sum game dynamics. The critic evaluates the **joint probability distribution over which player wins**:

```python
class MultiPlayerWinProbCritic(nn.Module):
    def __init__(self, embed_dim=768, n_players=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, value_tokens, still_playing_mask):
        # value_tokens: (B, 4, 768)
        logits = self.mlp(value_tokens).squeeze(-1) # (B, 4)
        
        # Mask out eliminated players
        logits = logits.masked_fill(~still_playing_mask, -1e9)
        win_probs = F.softmax(logits, dim=-1) # (B, 4)
        
        return win_probs
```

### Advantage Computation for 2-Player & 4-Player Modes
- In **2-player zero-sum matches**, convert player $p$'s win probability $\hat{p}_p \in [0, 1]$ directly to a symmetric $[-1, 1]$ value target:
  $$V_p(s) = 2 \hat{p}_p - 1$$
- In **4-player matches**, use the win probability $\hat{p}_p$ as the baseline for Generalized Advantage Estimation (GAE):
  $$\delta_t^p = r_t^p + \gamma \hat{p}_{t+1}^p - \hat{p}_t^p$$
  $$\hat{A}_t^p = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^p$$
