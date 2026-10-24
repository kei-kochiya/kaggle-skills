# Kaggriculture Operational Implementation Guide: Blueprints, Runbooks & Guardrails

This runbook provides actionable, production-grade instructions for implementing, training, post-processing, and evaluating an agent for the advanced **Kaggriculture** environment.

Refer to the [Kaggriculture Handbook Deep Dive](../../../../Handbook/reinforcement-learning/kaggriculture.md) for full context, mathematical derivations, and competition post-mortems across all gold medal teams.

---

## 1. Decision Matrix: Choosing Your Architecture

Before spending compute, choose the archetype matching your team's timeline and compute budget:

| Goal / Constraint | Recommended Archetype | Reference Solutions | Compute Needed | Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **Gold medal with minimal compute & zero RL instability** | **Archetype 2: Pure BC on Executed Labels** | 4th Place (Farmcore), 7th Place (DECEM) | 5 to 10 GPU-hours | Low |
| **Clean separation between strategy and micro-robotics** | **Archetype 3: Macro BC + Day Compiler** | 26th Place Gold | 10 to 20 GPU-hours + C++ | Medium |
| **Flawless endgame liquidation & farm budgeting** | **Archetype 4: Hybrid Opening + C++/Rust Planner** | 28th Place Gold (吃白饭的大肥鱼) | 10 GPU-hours + Planner | Medium-High |
| **Emergent cooperative play & full policy ownership** | **Archetype 1: BC → Critic Pre-Warm → PPO** | 1st Place (MMPQ), 3rd Place (DSM) | 100+ GPU-hours / Cluster | High |

---

## 2. Replay Ingestion & The Executed-Action Protocol

### The Phantom Order Problem
In high-level replays, elite players frequently over-order (submitting impossible seed purchases or unit actions) and let the engine silently drop them. Training directly on `steps[t+1].action` poisons your imitation policy with phantom commands.

### The Ingestion Pipeline
1. **Download Replays**: Use Kaggle API to scrape daily top-episode JSON files:
   ```bash
   kaggle competitions episodes -c kaggriculture --top-only
   ```
2. **Execute-Label Verification**: Replay each downloaded episode through the exact engine. Record only the state modifications that the engine actually resolved:
   ```python
   def extract_executed_transitions(episode_json):
       """Replay step by step and extract only applied state changes."""
       # Map steps[t].observation -> state changes at t+1
       for t in range(719):
           obs_t = episode[t]['observation']
           next_obs = episode[t+1]['observation']
           
           # Derive true executed actions from differences between obs_t and next_obs
           executed_units = derive_applied_unit_actions(obs_t, next_obs)
           executed_trades = derive_applied_market_trades(obs_t, next_obs)
           
           yield {
               "tokens": encode_board_tokens(obs_t),
               "unit_labels": executed_units,
               "market_labels": executed_trades,
           }
   ```
3. **Parity Check**: Drop any episode where `encode → decode → engine` does not produce the identical final coin balance.
4. **Conditioning Embeddings**: Tag each training trajectory with:
   - `Team / Style ID`: One-hot vector of the target author (MMPQ, DSM, DECEM).
   - `Recency`: Submission timestamp (weight newer replays exponentially higher).
   - `Outcome`: 1 for win, 0 for loss.
   - *Inference prompt*: Always set `Style = Top-1`, `Recency = Max`, `Outcome = 1`.

---

## 3. Model Architecture Blueprints

### Blueprint A: 266-Token Entity Transformer (DSM / MMPQ Standard)
```text
Token Layout (Total = 266 Tokens):
├── [0..199]   Board Tiles (200 tokens): 10x10 grid for both farms (type, age, wetness, weeds)
├── [200..219] Own Units (20 tokens): Farmer + 19 workers (x, y, tool, cargo_type, cargo_qty)
├── [220..239] Opponent Units (20 tokens): Public unit positions and visible tools
├── [240..255] Shops (16 tokens): Shop item, multiplier, distance to farm
├── [256..264] Commodities (9 tokens): Global inventories, base prices, daily town demands
└── [265]      Global State (1 token): Day [0..29], Hour [0..23], Bank Balance, Overage
```

#### Tensor Processing
```python
import torch
import torch.nn as nn

class KaggricultureEncoder(nn.Module):
    def __init__(self, d_model=320, nhead=8, num_layers=6, dim_feedforward=1024):
        super().__init__()
        self.cat_embed = nn.Embedding(num_embeddings=2048, embedding_dim=d_model)
        self.num_proj = nn.Linear(in_features=64, out_features=d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            activation='gelu', norm_first=True, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, cat_ids, num_features):
        # Sum categorical embeddings and projected continuous features
        tokens = self.cat_embed(cat_ids) + self.num_proj(num_features)
        return self.transformer(tokens)  # (Batch, 266, d_model)
```

### Blueprint B: One-Pass Draft + Joint Corrector (Farmcore 4th Place)
To avoid slow autoregressive decoding during Kaggle CPU submission:
1. Encoder generates simultaneous candidate action logits for all 20 units and 10 market orders in a single forward pass.
2. A 6-layer Transformer Corrector takes all 31 draft decision tokens as input and attends across them:
   $$\text{Corrected Logits} = \text{Corrector}(\text{Draft Tokens})$$
3. Units adjust their planned harvest/movement to prevent collisions and shared seed over-consumption without serial autoregression.

### Blueprint C: Pure NumPy CPU Deployment (DECEM 7th Place)
- Export model weights to quantized/half-precision NumPy arrays (`.npz`).
- Implement forward pass in pure NumPy:
  - Linear layers: `np.dot(x, W) + b`
  - GELU: `0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))`
  - LayerNorm: `(x - mean) / np.sqrt(var + eps) * gamma + beta`
- Latency: **~0.095s to 0.110s per step** on 1 core of Kaggle submission CPU.

---

## 4. Inference-Time Guardrails & Post-Processing Engine

Layer these battle-tested rules on top of raw model outputs:

### 1. Land Purchase Trigger (`_land` Rule)
```python
def apply_land_guardrail(actions, obs, step):
    # Between hours 240 and 709, buy land if cash + immediate expected sales >= 4000
    if 240 <= step <= 709 and obs['num_plots'] < 4:
        expected_sales = estimate_turn_sales_revenue(obs)
        if obs['money'] + expected_sales >= 4000:
            actions['market_orders'].insert(0, {'type': 'BUY_LAND', 'qty': 1})
    return actions
```

### 2. Melon Double-Cropping Rule (`MELON16`)
```python
def apply_melon_replanting(actions, step):
    # Days 16-19 (steps 384-479): Override wheat/carrot with melons
    day = step // 24
    if 16 <= day <= 19:
        for unit in actions['unit_actions']:
            if unit['action'] in ['PLANT_WHEAT', 'PLANT_CARROT']:
                if has_melon_seeds_available():
                    unit['action'] = 'PLANT_MELON'
    return actions
```

### 3. Order-Slot Priority Front-Running
The official engine processes orders **slot by slot** between Player 0 and Player 1.
```python
def sort_market_orders_by_revenue_priority(market_orders):
    """Sort orders so highest expected revenue/margin trades occupy slots 0, 1, 2."""
    return sorted(market_orders, key=lambda order: order['expected_revenue'], reverse=True)[:10]
```

### 4. Hourly Opponent Sales Inference & Dynamic Programming Liquidation
```python
def infer_opponent_sales(current_market, prev_market, our_sales_last_turn, town_drain):
    """Calculate exact quantity of goods sold by opponent in the previous turn."""
    market_delta = current_market['supply'] - prev_market['supply']
    # Delta = our_sales + opp_sales - town_drain
    opp_sales = market_delta + town_drain - our_sales_last_turn
    return max(0, opp_sales)
```

### 5. Terminal Liquidation Window
At **Day 29, Hour 22** (turn 718):
- Cancel all planting, watering, and feeding chores.
- Units drop cargo directly into the shed or town stall.
- Issue 10 `SELL` orders for 100% of remaining inventory.
- Ending the game with unharvested crops or unsold goods in the barn is a fatal flaw.

---

## 5. Fast Simulation Engines & Validation Gating

### Engine Options
- **C++ `kagsim`**: PyBind11 bindings to the official game rules. Reaches **256 games/s** locally (12,000× speedup over `kaggle-environments`).
- **Rust Engine**: Multi-threaded simulator using Rayon. Reaches **29,800 games/s** for large-scale league round robins.

### Exact Parity Requirements
1. **MT19937 RNG**: Match Python's exact random sequence. Seed state must be bit-level identical.
2. **Dict Insertion Order**: Shed overflow resolution must delete the earliest inserted keys in the storage dictionary.

### The Paired-Seed Gate Protocol
```python
def run_paired_gate_evaluation(candidate_agent, champion_agent, num_seed_pairs=768):
    """
    Evaluate candidate vs champion across paired seeds:
    Match A: candidate is Seat 0, champion is Seat 1 on seed S.
    Match B: champion is Seat 0, candidate is Seat 1 on seed S.
    """
    cand_wins = 0
    total_games = num_seed_pairs * 2
    
    for seed in range(num_seed_pairs):
        # Match A
        res_a = play_game(candidate_agent, champion_agent, seed=seed)
        if res_a['winner'] == 0: cand_wins += 1
        
        # Match B (swapped seats)
        res_b = play_game(champion_agent, candidate_agent, seed=seed)
        if res_b['winner'] == 1: cand_wins += 1
        
    win_rate = cand_wins / total_games
    std_error = (win_rate * (1 - win_rate) / total_games) ** 0.5
    z_score = (win_rate - 0.50) / std_error
    
    # Gate Condition: Win rate >= 52% and z > 2.0
    passed = (win_rate >= 0.52) and (z_score > 2.0)
    return passed, win_rate, z_score
```

---

## 6. Training Stability Checklist for RL (Archetype 1)

If conducting PPO self-play:
- [ ] **Freeze-Actor Critic Pre-Warm**: Fit value network for 50–100 iterations on rollout games while freezing the BC actor.
- [ ] **Zero-Sum Paired Seats**: Reward is $+1$ for win, $-1$ for loss. Rollouts must collect both Seat 0 and Seat 1 of each seed.
- [ ] **Discounting**: Use $\gamma = 1.0$. Do not apply standard $\gamma = 0.99$ discounting (at turn 719, $0.99^{718} \approx 0.0007$, completely erasing endgame liquidation value).
- [ ] **KL Regularization**: Anchor updates with $\mathcal{L}_{\mathrm{KL}} = 0.2 \times \mathrm{KL}(\pi_{\mathrm{teacher}} \parallel \pi_{\mathrm{policy}})$.
- [ ] **Exclude Forced Moves**: Exclude deterministic rule-forced actions (`BUY_LAND`, terminal sell) from the actor loss gradient, but retain their state consequences for the critic value target.
