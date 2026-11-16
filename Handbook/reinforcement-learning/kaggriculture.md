# Kaggriculture Master Post-Mortem: Gold Medal Solutions & The Four Winning Archetypes

The **Kaggriculture** simulation competition (720 turns, 30 game days, 2-player zero-sum shared market economy) challenged participants to control a complex farming enterprise: coordinate unit movements, crop watering and fertilizing, livestock care, storage capacity, and competitive market timing.

Across more than 10,200 competing teams, the gold medal bracket converged on a striking theoretical lesson: **pure reinforcement learning from scratch completely collapsed**, while **Behavioral Cloning (BC), executed-state imitation, hierarchical day compilers, and hybrid neural-combinatorial planners dominated the leaderboard**.

This handbook synthesizes the post-competition writeups and code releases from the confirmed top medalists (out of 10,148 competing teams):
- **1st Place**: Team M & M & P & Q (`msd0110`, `morim3`, `Piiiiiiiii`, `qistripute` — [Writeup](https://www.kaggle.com/competitions/kaggriculture/writeups/1st-place-solution-our-journey-to-becoming-profe): 10.2M Transformer with within-farm 2D RoPE, 50d opponent inventory estimator, LNS+DP heuristic solver, alternating BC $\leftrightarrow$ self-play PPO over 8.29M–9.04M games, and PPO-masked action corrections)
- **2nd Place**: Victor Mercklé (`vmerckle` — [Writeup](https://www.kaggle.com/competitions/kaggriculture/writeups/trading-clutch-no-b200) & [GitHub Repo](https://github.com/vmerckle/kaggriculture): 37M Encoder-Decoder HourFormer trained in **30 RTX 5090 GPU-hours (~$10)** via 22k-game BC + 1-min MMPQ fine-tune + 2h10m hard-example PPO + Wheat Trading Clutch algorithm & 2 override rules)
- **3rd Place**: Team DSM (`denden12`, `shimishige`, `masspeaks` — gen70 10.4M Transformer, C++/Rust engines, multi-stage post-processing)
- **4th Place**: Team Farmcore (Iaroslav Pudovkin — pure BC, executed-action training, one-pass transformer + corrector, zero RL/search)
- **6th Place**: Split-game BC/PPO (BC opening Days 1–11 + PPO mid/endgame Days 12–30, dual CPU/GPU leagues)
- **7th Place**: DECEM (Switched from heuristics in final 9 days; 8.2M entity Transformer with plain NumPy inference in 0.1s/turn)
- **19th Place**: Solo Gold (Multi-teacher distillation cloning 1st, 3rd, and 7th place teams)
- **26th Place**: Macro BC + Day Compiler (Hierarchical dawn `DayIntent` neural policy + C++ deterministic micro router & DP seller)
- **28th Place**: Team 吃白饭的大肥鱼 (Hybrid 10-day BC neural opening + 20-day Rust tile-project portfolio planner)

---

## 1. The Four Winning Solution Archetypes

The top 30 leaderboard segregated into four distinct, battle-tested architectural archetypes:

```mermaid
flowchart TD
    subgraph Archetype1["Archetype 1: BC → Critic Pre-Warm → PPO (1st, 3rd, 10th)"]
        A1_BC["Broad Replay BC"] --> A1_Val["Value Network Pre-Warm"]
        A1_Val --> A1_PPO["Paired Self-Play PPO + KL Anchor"]
        A1_PPO --> A1_Post["Multi-Stage Rule Post-Processing"]
    end

    subgraph Archetype2["Archetype 2: Pure BC on Executed State (4th, 7th, 19th, 27th)"]
        A2_Data["Kaggle API Replay Scraper"] --> A2_Clean["Filter to Executed State Changes (Drop Phantom Orders)"]
        A2_Clean --> A2_Cond["Conditioning: (Team, Recency, Win)"]
        A2_Cond --> A2_Net["Entity Transformer + Joint Corrector / NumPy"]
        A2_Net --> A2_Soup["Checkpoint Weight Soups"]
    end

    subgraph Archetype3["Archetype 3: Hierarchical Macro BC + Day Compiler (24th/26th)"]
        A3_Dawn["Dawn Neural Network: DayIntent"] --> A3_Comp["Deterministic C++ Day Compiler"]
        A3_Comp --> A3_DP["Hourly DP Seller vs Inferred Opponent Sales"]
    end

    subgraph Archetype4["Archetype 4: Hybrid Opening BC + Combinatorial Planner (28th)"]
        A4_BC["Neural BC Policy (Days 1–10)"] --> A4_Hand["History Replay Handover"]
        A4_Hand --> A4_Rust["C++/Rust Tile-Portfolio Planner (Days 11–30)"]
    end
```

| Dimension | Archetype 1: BC + PPO + Rules | Archetype 2: Pure Executed-Label BC | Archetype 3: Macro BC + Day Compiler | Archetype 4: Hybrid Opening + Planner |
| :--- | :--- | :--- | :--- | :--- |
| **Pioneering Teams** | 1st (MMPQ), 3rd (DSM), 10th | 4th (Farmcore), 7th (DECEM), 19th, 27th | 24th / 26th Gold | 28th Gold (吃白饭的大肥鱼) |
| **Compute Budget** | High (Multi-GPU / Cluster, 360M+ steps) | Low to Modest (5.5 A100-hrs to single Colab/5070) | Modest (Built in 5 days with LLM agent) | Modest (Trained on PPU, tested on CPU) |
| **Role of RL** | Large-scale self-play fine-tuning | Zero RL (RL found 10x less sample-efficient) | Zero RL | Zero RL |
| **Inference Mechanism** | Neural network + confidence repair / overrides | Neural network + joint corrector or 2 minimal rules | Dawn neural inference + deterministic C++ micro compiler | Days 1–10 Neural; Days 11–30 Combinatorial optimization |
| **Key Advantage** | Discovers emergent multi-agent coordination | Maximum training stability; zero RL reward noise | Eliminates mechanical pathing & collision failures | Exact mathematical crop portfolio & shed liquidation |

---

## 2. Core Breakthroughs & Competitive Lessons

### Breakthrough 1: Executed-State Labeling vs Player Orders (The Farmcore Discovery)
In complex game environments, top players frequently **over-order** (e.g. requesting multiple competing seed purchases or unit actions) and rely on the game engine to silently discard or reject invalid orders.
- **The Pathology**: Naive Behavioral Cloning trained directly on `steps[t+1].action` forces the neural policy to memorize illegal or phantom orders, teaching it to issue empty commands during inference.
- **The Fix (4th Place Farmcore)**: Decode the trajectory and train strictly on the **actual state changes applied by the official engine**. Retraining the exact same network on executed labels + a single feeding rule caused win rate against their previous best agent to surge from **3/24 to 23/24**.
- **Rule of Thumb**: Drop any replay where `encode → decode → engine` does not match the official game outcome to the exact coin.

### Breakthrough 2: Hierarchical Decomposition (Macro Intent at Dawn + Micro Day Compiler)
Managing 20 units across 720 individual turns creates a colossal action space ($31^{720}$) fraught with pathing deadlocks and coordination breakdowns.
- **26th Place Solution**: Decoupled strategic direction from labor execution.
  - **Dawn Neural Head (`DayIntent`)**: Evaluates board state once every 24 hours (at hour 0). Outputs only high-level macro targets: crops to buy/plant, plots of land to purchase, and group care/harvest quotas.
  - **Deterministic C++ Day Compiler**: Solves all micro logistics—worker route assignment, job dispatch, watering cadence, cargo hauling, shed space management, and market slot insertion.

### Breakthrough 3: Opponent Sales Inference & Dynamic Programming Liquidation
The opponent's private inventory and current turn actions are concealed, but market parameters and aggregate prices are public:
$$\Delta \text{Market Inventory} = \text{Our Sales} + \text{Opponent Sales} - \text{Town/Shop Consumption}$$
- By inverting this formula every hour, teams accurately computed:
$$\text{Opponent Sales}_t = \Delta \text{Market Inventory}_t + \text{Consumption}_t - \text{Our Sales}_t$$
- **Hourly DP Seller**: A dynamic programming routine planned sales across remaining hours, optimizing instantaneous price impact against the opponent's inferred production schedule and front-running their trades.

### Breakthrough 4: Market Slot Priority Mechanics
The official engine resolves simultaneous trades by matching orders **slot by slot in the order array**.
- If Player A and Player B both sell tomatoes at hour $t$, Player A's Order 0 is processed before Player B's Order 1.
- Listing higher-revenue goods in earlier order slots yielded **+\$4 to +\$10 per unit sold**, totaling **\$400–\$600 extra cash per match** in top-bracket games.

### Breakthrough 5: The Handover Protocol (Hybrid Opening + Endgame Planner)
- **28th Place (吃白饭的大肥鱼)** realized that Days 1–10 require flexible spatial intuition to adapt to random shop placements and weed configurations. Days 11–30 require strict combinatorial math to maximize revenue before turn 719.
- At hour 0 of Day 11, the agent transfers control from the neural network to a Rust/C++ planner.
- **State Handover**: The planner replays the entire 10-day history through its internal trackers to reconstitute tile ages, soil moisture, and worker queues before solving the tile project portfolio.

### Breakthrough 6: Conditioning on Team, Recency, and Win Outcome
- **7th Place (DECEM)** and **27th Place (monsaraida)** conditioned their entity Transformers on:
  1. `Team / Style ID`: One-hot vector of elite target teams (MMPQ, DSM, DECEM).
  2. `Recency`: Timestamp / submission date of the game (exponentially weighting newer replays as meta evolved).
  3. `Outcome`: Binary indicator of match win/loss.
- At inference time, agents were conditioned with: `Style = Team 1`, `Recency = Latest`, and `Outcome = Win`.

---

## 3. Deep-Dive: Solution Architectures

### A. Team MMPQ (1st Place) — Alternating BC & Self-Play PPO, 2D Within-Farm RoPE, Opponent Inventory Estimator, LNS+DP Solver & RL Action Corrections

Team MMPQ (`msd0110`, `morim3`, `Piiiiiiiii`, `qistripute`) won 1st place out of 10,148 teams using two ~10.2M-parameter Transformers (`Final A`: 8.29M games / 11.922B player steps; `Final B`: 9.04M games) built in Rust + distributed JAX (`30× A30 + 18× A100`, ~25–26 full games/s including PPO updates).

#### 1. Model Architecture & 6 Token Types (`~10.2M` Params)
- **Backbone**: Pre-LayerNorm Entity Transformer with **12 layers, width 256, 8 heads (`head_dim=32`), FFN 1024, GELU**, balancing cluster rollout throughput with Kaggle CPU submission limits.
- **6 Token Types**:
  - `Global` (34 dims): time, bank balances, unlocked land, shops, total inventory, free warehouse capacity, worker count.
  - `Cell` (34 dims $\times$ 200 cells across both farms): farm identity, coords, terrain, crop/livestock state, Unit occupancy.
  - `Unit` (22 dims): farm identity, coords, farmer/worker type, carried inventory, warehouse access.
  - `Product` (19 dims $\times$ 12 products): identity, warehouse inventory, seed inventory, market inventory, price, town demand.
  - `Opponent Inventory Estimate` (50 dims): estimated total/carried quantities, uncertainties, unresolved spending, price-floor sales ambiguity.
  - `Market Slot` (10 dims $\times$ 10 slots): slot order index.
  - Explicit aggregate features (total warehouse inventory, free capacity, total carried across Units and per Unit) were added so the 12-layer encoder did not need to learn summation across many tokens.
- **Within-Farm 2D RoPE**: Applied 2D Rotary Position Embeddings (`base=100`, 8 horizontal + 8 vertical + 16 unrotated dims per head) **strictly between Cell and Unit tokens on the same farm**, because coordinate differences across two separate farms have no physical meaning (tokens across farms still attend via standard unrotated projections).
- **Single-Pass Parallel Action Heads**: Avoided slow autoregression by emitting all logits in one forward pass:
  - *Unit Head (500 candidates/Unit)*: 480 `PICKUP`/`PLACE` (`2 ops × 12 products × 20 quantities`), 5 crop-specific `PLANT`, 15 move/water/other.
  - *Market Head (1,903 candidates $\times$ 10 slots)*: 900 `SELL` (factorized into 9-product distribution $\times$ product-specific `1–100` quantity distribution), 1,000 purchase (`5 seeds + wheat + fertilizer + 3 livestock × 100 quantities`), `no-order`, `HIRE`, `BUY_LAND`.
- **Zero-Sum Paired Value Head**:
  $$z_i = f_\theta(o_i) + (a\tau + b)\frac{m_i - m_{1-i}}{20{,}000}, \qquad V_i = \frac{2\exp(z_i)}{\exp(z_0) + \exp(z_1)} - 1 = \tanh\!\left(\frac{z_i - z_{1-i}}{2}\right)$$
  where $V_0 + V_1 = 0$, $m_i$ is bank balance, $\tau \in [0, 1]$ is elapsed game fraction, and $a, b$ are trainable scalars.

#### 2. 50-Dimensional Opponent Hidden Inventory Estimator
Because opponent warehouse and carried inventories are hidden, MMPQ built a Rust state observer tracking 12 products by combining: (i) visible board updates (separating overnight growth from Unit harvesting/feeding/planting/fertilizing), (ii) market supply changes minus town consumption and own trades, and (iii) unexplained opponent bank-balance deltas matched against fixed-price seed/livestock combinations (adding minimum feasible quantities across valid combinations and emitting 24 uncertainty dims + 2 ambiguity dims). Evaluated across 197,725 transitions (275 game-player pairs):
- **Total inventory**: **0.702 units MAE** overall (**1.926** when nonzero), **86.80% exact match**.
- **Carried inventory**: **0.700 units MAE** overall (**2.799** when nonzero), **88.07% exact match**.

#### 3. Alternating BC $\leftrightarrow$ Self-Play PPO Cycle
1. **Diverse Teacher Initialization**: Initialized with 30 epochs of BC across **34 diverse public submissions** (game-level split). Deliberately selected the 34-teacher pool even though it scored `10–90` immediately post-BC vs `82–18` for a narrow teacher pool, to preserve strategy diversity for PPO. Froze the encoder/policy and pretrained the value head before PPO ($\gamma=1.0, \text{GAE }\lambda=0.97$ — necessary since at horizon $719$, $0.97^{718} \approx 3\times 10^{-10}$, making intermediate critic estimates critical).
2. **Pure Terminal PPO**: Optimized strictly terminal outcome ($+1$ win, $0$ draw, $-1$ loss; **zero intermediate rewards**), Huber value loss ($\times 2$), PPO clip $\epsilon=0.2$, global cross-GPU advantage normalization, and forward KL penalty $D_{\text{KL}}(\rho \parallel \pi_\theta)$ to a frozen reference $\rho$ promoted when a candidate won $\ge 70/100$ games.
3. **Targeted Recovery BC on Plateau**: Whenever PPO plateaued or forgot hard multi-step sequences (e.g., buying/farming the **4th land plot** or switching to **tomatoes** under pizza/greengrocer spikes), they added targeted replays (e.g., 849 public 4th-plot games + 300 solver games) for **1–3 BC epochs** with entropy + KL regularization:
   $$\mathcal{L}_{\mathrm{BC}} = \mathrm{CE}_U + \mathrm{CE}_M + 0.05(D_U + D_M) - 0.10 H_U - 0.10 H_M$$
   keeping normalized entropy $H(p)/\log K \approx 0.12$ (`0.1271` Unit, `0.1168` Market). Even though the short BC checkpoint lost `0–100` to the pre-BC policy, resuming PPO (with lr warmup $0 \to 5\times 10^{-5}$ over 240 Adam steps) surpassed the old ceiling while retaining the new strategy.

#### 4. LNS + Per-Cell DP Heuristic Solver
- Precomputed per-cell future action sequences via independent cell Dynamic Programming $V(d, s) = \max_a [r(a) + \gamma V(d+1, s')]$ against projected market prices, then ran **Large Neighborhood Search (LNS)** with simulated annealing over daily worker routes, hiring, and land expansion.
- Complete plans were scored by farm-wide simulated net cash-flow difference through Day 29 (explicitly modeling price depression from bulk melon sales). Reached #1 Public LB (~3,150) in mid-September, served as a BC teacher for price-spike adaptations, and ran on Day 29 in `Final A`.
- *Note*: A hierarchical dawn-macro NN + heuristic executor failed to match full-turn models due to slow CPU collection and mismatch with worker execution/sale timing.

#### 5. The 8 Action Corrections (A–H) & Incorporating Rules Into PPO Rollouts
Near the deadline, replay inspection uncovered warehouse overflow losses, unsold terminal goods, and parallel Unit conflicts. MMPQ designed 8 rules:
- **A**: Sell inventory that would overflow warehouse at night; **B**: Sell warehouse inventory on final 2 turns + force adjacent loaded Units to `DROP` on turn 719 (cancel purchases); **C**: Resolve duplicate Unit targets on same cell; **D**: Remove planting actions exceeding remaining seeds; **E/F/G**: Mask ineffective fertilize/livestock/out-of-bounds actions; **H**: Switch to LNS+DP solver on Day 29 (helped earlier checkpoints in `Final A`, hurt after 1 more day of NN training).

| Submission | How Rules Were Applied | Record vs `Victor A (56710008)` | Record vs `Victor B (56722220)` |
| :--- | :--- | :---: | :---: |
| **`Final A (56701470)`** | Inference-time corrections A–G + Day 29 solver (H) | **81.0%** (`136–32`) | **41.9%** (`223–309` — would have placed 2nd!) |
| **`Final B (56711476)`** | **Rules E/F/G masked before sampling; C/D masked sequentially (farmer $\to$ workers); overridden A/B excluded from PPO log-probs during final RL** | **90.7%** (`39–4`) | **61.1%** (`386–246` — secured **1st Place**) |

- **Net2Net Expansion & Replay-Hiding Strategy**: Expanded Transformers `6 -> 12 -> 24` layers (`10M -> 20M -> 30M`) by zero-initializing output projections of new attention/FFN blocks ($F(x)=0$) and copying Adam moment estimates; deliberately submitted weaker decoy agents after testing a #1 model to prevent rivals from scraping active top-2 submission replays.

---

### B. Victor Mercklé (`vmerckle`, 2nd Place) — 37M HourFormer in 30 GPU-Hours (~$10), Hard-Example PPO & The Wheat Trading Clutch

Victor Mercklé (`vmerckle`) finished 2nd with **~1.5k lines of plain Python/PyTorch** ([`github.com/vmerckle/kaggriculture`](https://github.com/vmerckle/kaggriculture)), training a 37M-parameter model for only **30 RTX 5090 GPU-hours (~$10)** and boosting head-to-head win rate against MMPQ from **9% to 39%** on the final day (including `58.1%` / `309–223` against MMPQ's `Final A`).

#### 1. 37M-Parameter Encoder-Decoder HourFormer Architecture
- **19M Encoder** (run 719 times/game): Reads **217 tokens** per hour (1 summary token, 200 tiles of both farms, 16 worker slots) + a learned **Team Conditioning Embedding** identifying which of 23 cloned teams to imitate.
- **17M Autoregressive Decoder**: Emits the hour's 26 actions one by one: **10 market orders first**, then **16 worker commands** each encoded as `(operation, item, quantity)`.

#### 2. The 30 RTX 5090 GPU-Hour Simplified Recipe
1. **Behavioral Cloning (25 GPU-hours / 4h wall-clock on 8 GPUs)**:
   - Dataset: **22,000 scraped games from the top 23 leaderboard teams** (up to 15h before deadline).
   - `110,000` steps of `1,800` game-hours each, `AdamW` ($\text{lr}: 10^{-3} \to 10^{-4}$ over 300 warmup steps + cosine decay, `weight_decay=0.1`, `dropout=0.1`), with the team ID token masked out in `10%` of game-hours to force generalization.
2. **Top-Team Fine-Tuning (1 minute)**: `600` steps strictly on 1st-place **MMPQ** replays at `lr=5e-5`.
3. **Hard-Example PPO Fine-Tuning (2h 10m)**:
   - Started from fine-tune step 600 playing against older fine-tune checkpoints: `15` iterations of `688` games (`temperature=0.5`), each followed by `22` optimizer steps of `44 × 512` game-hours (`lr=2e-5`, 20 warmup steps, `GAE λ=1.0`, target KL `0.05`).
   - **25% Lost-Game Replay**: `25%` of every iteration's rollout games replayed seeds lost in earlier iterations (each lost game replayed at most twice).

#### 3. The "Wheat Trading Clutch" (+$700 Cash) & Two Deterministic Rules (+$600)
- **Suppressing Cyclic Micro-Trades in RL, Unleashing at Submission**: In self-play RL, hourly wheat front-running creates toxic rock-paper-scissors cycles ($A > B > C > A$) that waste GPU updates chasing `$10` timing edges instead of growing tomatoes. Victor **actively prevented wheat trading during all RL training**, then deployed an external deterministic **Wheat Trading Clutch** algorithm *only on his final submission*, adding **+$700** and jumping win rate vs MMPQ from `9%` to `39%`.
- **Two Post-Forward Rules (+$600)**:
  1. Sell any end-of-day overflow inventory before expiry (`+$350` to `+$800`).
  2. Endgame recall: force workers to walk to the shed and sell held inventory before game end (`+$30` to `+$180`).
- **Last-$x$-Hour Mini-Game Tree Search Debugging**: Built isolated mini-games starting from realistic states with only $x=1$ to $x \le 5$ hours remaining, where exact tree search revealed optimal endgame liquidation and immediately exposed subtle RL/execution bugs.

---

### C. Team DSM (3rd Place) — End-to-End Transformer + PPO + Post-Processing

#### 1. Input Tokenization (266 Tokens)
```text
Total Tokens: 266
├── 200 Board Tiles (10x10 farm grid for both players: crop type, growth age, wetness, weed status)
├── 20 Own Units (Farmer + up to 19 workers: position, held cargo, tool status)
├── 20 Opponent Units (Visible public positions and actions)
├── 16 Shops (Type, current commodity, price multipliers)
├── 9 Item / Commodity Summaries (Global supply, base prices, town demand)
└── 1 Global Token (Current day [0-29], hour [0-23], cash balance, remaining overage)
```
- Numeric continuous variables pass through a `Linear` layer; categorical features pass through categorical embeddings. Embeddings and projected continuous vectors are summed. No sequence positional embeddings are used.

#### 2. Network Backbone & Heads
- **Encoder**: 6-layer Transformer, $d=320$, 8 attention heads, Feed-Forward dimension 1024, Pre-LayerNorm.
- **Unit Head**: Concatenates unit token with the underlying tile token; outputs logits over 44 candidate actions masked by legal board state. A conditional 24-class head predicts movement/action quantities.
- **Market Head**: 2-layer autoregressive cross-attention decoder attending to all 266 encoder tokens. Outputs up to 10 market orders sequentially until `STOP` (23 order categories, 132 quantity buckets).
- **Parameters**: 10,441,973. Inference latency: ~0.112s on Kaggle CPU.

#### 3. Loss Formulation
$$\mathcal{L}_{\mathrm{BC}}(s) = -\sum_{u} \Big[\log \pi(c_u^* \mid s) + 0.5 \log \pi(q_u^* \mid s, c_u^*)\Big] - \sum_{j=1}^{L} \Big[w_j \log \pi(m_j^* \mid s, m_{<j}^*) + 0.5 \log \pi(r_j^* \mid s, m_{\le j}^*)\Big]$$
- Crucial nuance: $w_j = 20$ for `BUY_LAND` actions to counter extreme class imbalance (a typical match contains only 1–3 land purchases across 720 turns).

#### 4. Inference-Time Post-Processing Engine
DSM layered strict rule interventions over the raw model predictions:
1. `_land` rule: Between steps 240 and 709, if owning $<4$ plots and $(\text{Cash} + \text{Expected Sales}) \ge 4000$, inject `BUY_LAND`.
2. `MELON16` & Double-Cropping: On Day 16, buy melon seeds up to 3 units. On Days 16–19, replace `PLANT WHEAT` / `PLANT CARROT` with `PLANT MELON` (+456 coins/game gain on official engine).
3. `_pace` liquidation: From step 710 onward, sell remaining warehouse inventory divided by steps left. At hour 23, liquidate all barn contents before daily reset.

---

### D. Team Farmcore (4th Place) — One-Pass Transformer + Joint Corrector

Farmcore achieved 4th place using **zero self-play reinforcement learning and zero tree search**, relying purely on an elegant one-pass architecture and clean data engineering:

```mermaid
flowchart LR
    Obs["Board State (266 Entities)"] --> Enc["Transformer Encoder (8M Params)"]
    Enc --> Draft["Joint Turn Draft (All 31 Decisions)"]
    Draft --> Corr["6-Layer Cross-Decision Corrector"]
    Corr --> Action["Valid Coordinated Action"]
```

- **One-Pass Draft + Corrector**: Instead of autoregressive token generation (which is slow and introduces order bias), the encoder generates a simultaneous soft draft for all 20 units and 10 market orders. A 6-layer Transformer Corrector cross-attends across all draft decisions, allowing units to adjust to each other's planned locations and resource requirements in parallel.
- **Pretrain → Fine-Tune → Average (Model Soup)**:
  1. Pretrain on 8,583 replays from 67 top submissions.
  2. Fine-tune narrowly on a single top-1 team (563–766 replays).
  3. Average model weights across the top fine-tuned checkpoints (`Checkpoint Soup`).

---

### E. 26th Place Gold — Macro BC + Day Compiler

```mermaid
sequenceDiagram
    participant E as Game Engine
    participant N as Dawn Neural Policy
    participant C as C++ Day Compiler
    participant S as DP Hourly Seller

    Note over E,N: Hour 0 (Dawn)
    E->>N: Dawn Observation (Day t)
    N->>C: DayIntent (Macro Goals, Herd Quotas, Plot Purchases)
    Note over C: Compile Worker Routes, Tile Tasks, Cargo Schedules
    loop Every Hour (0 to 23)
        C->>S: Current Inventory & Worker Status
        S->>E: Execute Hourly Unit Actions & Priority Market Orders
        E-->>S: Hourly Trade & Inventory Feedback
    end
```

- **Output Interface (`DayIntent`)**:
  - `new_cow = 2`, `buy_land = 1`, `plant_tomato = 14`.
  - The network outputs high-level intentions. The deterministic C++ compiler handles all A* pathing, obstacle avoidance, tool equipping, watering cadence, and cargo return trips.
- **Benefits**: Completely eliminated suicidal worker behaviors, seed jamming, and pathing deadlocks that plagued unconstrained end-to-end neural policies.

---

### F. 28th Place Gold (吃白饭的大肥鱼) — Hybrid 10-Day BC + 20-Day Rust Planner

- **The Problem**: Pure heuristics struggled with opening versatility (shop lottery variations, initial quadrant layout). Pure neural policies struggled with endgame cash maximization and shed liquidation.
- **The Protocol**:
  - **Days 1–10**: Pure BC neural policy controls expansion, opening plot selection, and early animal acquisition.
  - **Day 11 Dawn Handover**: Rust combinatorial planner evaluates the farm.
  - **Tile Project Optimization**: Evaluates project portfolios $P$ by solving:
    $$J(P) = B(F(P)) - L(W(P)) - C(P)$$
    where $B(F(P))$ is product flow market revenue, $L(W(P))$ is labor wage cost, and $C(P)$ is capital expenditure.
  - **Terminal Liquidation Window**: Strict liquidation trigger at **Day 29, Hour 22** (turn 718). All remaining crops are converted to cash; no item is left in unit backpacks or storage.

---

## 4. Evaluation Engineering: C++ & Rust Simulation Engines

Top teams abandoned the official Python interpreter (`kaggle-environments`) for local training and league validation:

| Engine | Technology | Throughput | Use Case | Fidelity Requirements |
| :--- | :--- | :--- | :--- | :--- |
| **Official Python** | `kaggle-environments` | ~0.02 games/s | Final verification & ground truth | Native Python baseline |
| **DSM C++ (`kagsim`)** | C++17 / PyBind11 | 256 games/s (12,000× speedup) | PPO rollouts & FastEval | CPython MT19937 RNG parity |
| **DSM Rust Engine** | Rust / Rayon | 29,800 games/s (8 threads) | Round-robin league tournaments | MT19937 + dict insertion order parity |
| **6th Place GPU Engine** | LibTorch / CUDA | Batches of 128 parallel games | PPO rollouts on GPU cluster | Parallel tensor mechanics |

### Critical Parity Traps
1. **RNG Parity**: The official interpreter uses Python's standard `random.Random` (Mersenne Twister MT19937). Fast engines had to implement exact CPython-compatible MT19937 state generation.
2. **Dict Insertion Order**: When the shed overflows (capacity 100), items are preserved according to Python dict insertion order. Fast engines that used hash maps without insertion preservation diverged from official match outcomes.
3. **Paired-Seed Evaluation Gate**:
   - Never evaluate on random unpaired games.
   - Run candidate models on **768 to 1,536 paired games** (swapping Seat 0 and Seat 1 on identical seeds).
   - Gate criterion: Candidate must achieve win rate $\ge 0.52$ ($z > 2.0$) against incumbent champion across paired games.

---

## 5. What Failed: Anti-Patterns & Negative Results

The following intuitive ideas were tested extensively by gold medalists and decisively failed:

| Failed Hypothesis | Experimental Result | Why It Failed |
| :--- | :--- | :--- |
| **From-Scratch PPO (No BC)** | Complete training collapse; win rate 0.00 | Horizon is 720 steps with 31 decisions/step. Sparse terminal win/loss cannot bridge credit assignment to early actions. |
| **Optimizing on Held-Out BC Loss** | Lower BC loss produced zero gain or weaker agents | Cross-entropy loss measures token matching on average moves (dominated by `PASS` / movement), not strategic game-winning inflection points. |
| **Scaling Model Size to 27M–66M** | 27.8M model showed identical win rate (+604 ± 1400 coins) | 10M parameters was sufficient to saturate policy capacity; bottleneck was data diversity and game coordination. |
| **Mixed Multi-Teacher Embeddings** | Weaker than fine-tuning on a single elite teacher | Blending incompatible high-level play styles (e.g. melon rushing vs multi-animal farming) created indecisive hybrid policies. |
| **Action Sampling at Play Time** | Degraded performance across all benchmarks | Temperature sampling introduced fatal action errors; deterministic greedy argmax inference won consistently. |
| **Feeding Unit Previous Action** | Clone learned to copy itself, losing -\$25k | Created pathological loops where units repeatedly executed obsolete commands. |

---

## 6. Comprehensive Summary Table of Winning Approaches

| Rank | Team / Author | Model Family | Key Innovation | Compute / Framework |
| :---: | :--- | :--- | :--- | :--- |
| 🥇 **1st** | **M & M & P & Q** (`msd0110`, `morim3`, `Piiiiiiiii`, `qistripute`) | 10.2M Entity Transformer | Within-farm 2D RoPE; 50d opponent inventory estimator; LNS+DP heuristic solver; alternating BC $\leftrightarrow$ PPO + PPO-masked rules | 30× A30 + 18× A100 / Rust + JAX |
| 🥈 **2nd** | **Victor Mercklé (`vmerckle`)** | 37M Enc-Dec HourFormer | 22k-game BC + 1-min MMPQ fine-tune + 25% lost-game PPO; suppressed cyclic wheat trades in RL, added Wheat Trading Clutch at submission | **30× RTX 5090 GPU-hours (~$10)** / PyTorch |
| 🥉 **3rd** | **DSM** (`denden12`, `shimishige`, `masspeaks`) | 10.4M Transformer (gen70) | Replay BC on MMPQ; value pre-warm PPO; Rust engine (29.8k g/s); post-processing rules | JAX / C++ `kagsim` / Rust |
| 🏅 **4th** | Farmcore | 8M Transformer | Pure BC on executed state changes; 1-pass draft + 6-layer joint corrector; checkpoint soup | 5.5 A100-hours / PyTorch |
| 🏅 **6th** | 6th Place | Split Transformer | Split-game policy: BC Days 1–11 + PPO Days 12–30; dual CPU/GPU evaluation leagues | Dual CPU/GPU League |
| 🏅 **7th** | DECEM | 8.2M Entity Transformer | Switched to BC in final 9 days; conditioned on (Team, Version, Win); pure NumPy inference (0.1s) | Single GPU + NumPy CPU |
| 🏅 **10th** | 10th Place | Transformer | Consumer hardware proof: BC + PPO trained end-to-end on a single RTX 5090 | 1× RTX 5090 |
| 🏅 **19th** | Solo Gold | Entity Transformer | Multi-teacher distillation cloning 1st, 3rd, and 7th place teams with outcome conditioning | Single GPU |
| 🏅 **26th** | Macro BC | Dawn Transformer + C++ | Hierarchical dawn `DayIntent` + C++ Day Compiler + hourly DP seller vs inferred opponent sales | 5-day agent build / PyTorch |
| 🏅 **28th** | 吃白饭的大肥鱼 | Hybrid BC + Rust | 10-day BC neural opening + 20-day Rust tile-project portfolio planner; Day 29 H22 liquidation | PPU / Rust |
