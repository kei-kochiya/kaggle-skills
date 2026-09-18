# Orbit Wars: Scaling Reinforcement Learning to the Stars 🚀

**Competition**: [Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)  
**1st Place Solution**: *Scaling Reinforcement Learning to the Stars* by Isaiah Pressman ([Writeup](https://www.kaggle.com/competitions/orbit-wars/writeups/1st-place-solution-scaling-reinforcement-learnin))  
**Open-Source Repository**: [`IsaiahPressman/kaggle-orbit-wars`](https://github.com/IsaiahPressman/kaggle-orbit-wars)  
**Track**: Competitive Multi-Agent Reinforcement Learning & Continuous Simulation  
**Core Methodology**: 200M-Parameter Entity Transformer, 15 Billion Steps Pure Self-Play PPO, Rust Engine Acceleration, NF4/NF5 Quantization, and Multi-Tier CPU Fallback Serving  
**Hardware Cluster**: 4 Nodes × 8× NVIDIA B200 GPUs (32× B200 total), 8,192 Parallel Environments (~2,400 B200-hours)

---

### Official Top-10 Winning Scoreboard & Methodology Matrix

| Rank / Competitor | Framework & Simulation Engine | Model Architecture & Size | Training Scale & Compute | Core Winning Innovations & Paradigms | Final Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 🥇 **1st Place (`IsaiahPressman`)** | **Rust Vector Engine**<br>(PyO3 + Rayon) | **200M Transformer**<br>(38 blocks, 16 heads) | **15 Billion Steps**<br>(32× B200 GPUs,<br>~2,400 B200-hrs) | • **Bitter Lesson validation**: pure self-play PPO at scale<br>• **Single-pass multi-player forward pass**: all agents predicted in 1 pass<br>• **120k+ steps/sec Rust simulator** with pinned zero-copy buffers<br>• **Sub-100MiB NF4-LSQ quantization** + dynamic int8 CPU serving<br>• **Latency circuit breaker**: 5M model fallback when bank < 1.0s | 🥇 **1st Place Gold**<br>Decisive orbital macro-dominance across 2p & 4p |
| 🥈 **2nd Place (`simjeg`)** | **Custom Engine**<br>PyTorch RL | Compact Entity Neural Policy | ~2 Billion Steps<br>(Multi-GPU cluster) | • **Progressive Curriculum**: Heuristic bots (Top 50) $\to$ Behavioral Cloning on top replays (Top 10) $\to$ RL fine-tuning (Top 5) $\to$ RL from scratch<br>• **Anti-Stall Reward Shaping**: $+1.0$ for winning $<500$ turns, $+0.5$ for winning $\ge 500$ turns | 🥈 **2nd Place Silver**<br>Elite tactical precision |
| 🥉 **3rd Place (`Felix M. Neumann`)**<br>*"Ab in den Orbit"* | **JAX Vector Engine**<br>(`jax.jit` / `jax.vmap`) | Transformer Policy | ~3 Billion Steps<br>(GPU cluster) | • High-throughput JAX self-play PPO pipeline<br>• Transformer-based entity attention for orbital fleet routing | 🥉 **3rd Place Bronze**<br>High-speed parallel RL |
| 🏅 **5th Place (`TonyK`)** | **Distributed IMPALA Engine** | Deep Actor-Critic Policy | ~2.5 Billion Steps<br>(Distributed actors) | • **Behavioral Cloning initialization** on top human/bot tournament replays<br>• **Asynchronous IMPALA** with V-trace off-policy corrections<br>• **Delayed moving teacher (Polyak anchor)** for policy regularization<br>• **Frozen historical opponent pool** to eliminate cyclic forgetting | 🏅 **5th Place Top 5**<br>Robust diverse counter-strategies |
| 🏅 **6th Place (`flg`)** | **C++ Native Engine** | **2.5M Transformer**<br>with **Edge-Attention** | ~2 Billion Steps<br>(Local cluster) | • **Custom Relational Edge-Attention**: injects pairwise planet travel times, transit curves, and ownership change horizons directly into attention logits<br>• **2-Step Lookahead Rollout Search**: evaluates short rollouts during 2p inference | 🏅 **6th Place Top 10**<br>Hybrid RL + Search |
| 🏅 **7th Place (`Audun Ljone Henriksen & Eirik Torp`)** | Structured Experimentation Harness | Hybrid Neural / Tactical Policy | Systematic ablation ladder | • **Component-Isolated Experimentation**: rigorous decomposition of tactical heuristics, search controllers, and RL modules<br>• **Local Elo Ladder**: high-confidence tracking preventing leaderboard overfitting | 🏅 **7th Place Top 10**<br>Disciplined validation |
| 🏅 **8th Place (`Billy Bradley`)**<br>*"Ender for <$200"* | **JAX Vector Engine** | Frugal Transformer<br>with **2D RoPE** | **<$200 Compute Budget**<br>(Single GPU) | • **2D Rotary Position Embeddings (2D RoPE)** for planar orbital coordinates $(r, \theta)$<br>• **Autoregressive micro-steps** constructing multi-source launches<br>• **Precomputed reachability & ETA candidate filtering** | 🏅 **8th Place Top 10**<br>Most compute-efficient RL |
| 🏅 **9th Place (`Boey`)**<br>*"End-to-End JAX PPO"* | **100% Pure JAX Pipeline**<br>(`jax.jit`, `jax.vmap`) | Entity Transformer Policy | ~4 Billion Steps<br>(TPU v4 / GPU) | • **Zero CPU-GPU transfer overhead**: entire environment, observation generation, action masking, rollout, and PPO loss fused into JIT GPU kernels<br>• **"Planet Future" Projections**: turn-by-turn simulated garrison forecasts | 🏅 **9th Place Top 10**<br>Engineered feature projections |
| 🏅 **10th Place (`Xiangyu Liu`)** | **Decoupled Strategic / Physics Engine** | **44-Planet Compact Transformer** | ~1.5 Billion Steps<br>(JAX + C++) | • **Decoupled Architecture**: `MapCache` (geometry) + `StrategicEnv` (combat/economy) + JAX PPO + C++ analytical intercept solver<br>• Compact 44-entity observation space with folded fleet dynamics | 🏅 **10th Place Top 10**<br>Open-source clean systems design |

---

## 1. Executive Summary & The Winning Paradigms

Kaggle's **Orbit Wars** challenged competitors to build autonomous agents capable of conquering a 2D continuous solar system governed by gravitational mechanics. Ships launch from planets, enter elliptical orbital transfers around a central star, intercept moving comets, and engage in planetary combat across both **2-player** and **4-player** modes.

The winning solution by Isaiah Pressman represents a historic milestone in competitive Kaggle simulation benchmarks by validating **Sutton's Bitter Lesson**: *general methods that leverage massive computation and expressive model capacity decisively outperform domain-specific human heuristics.*

```
                          OWL SYSTEM HIGH-LEVEL ARCHITECTURE
                          
 +--------------------------------------------------------------------------------+
 |                         RUST SIMULATION ACCELERATOR                            |
 |  • 8,192 Parallel Environments via Rayon Threads                               |
 |  • AABB Collision Broad-Phase Filtering (120,000+ env steps/sec)               |
 |  • Preallocated Pinned Memory Buffers (Direct Zero-Copy CPU->GPU Handoff)      |
 |  • Automated Replay Parity Testing against Official Kaggle Tournament JSONs    |
 +---------------------------------------+----------------------------------------+
                                         |
                                         v
 +--------------------------------------------------------------------------------+
 |                  200M-PARAMETER ENTITY TRANSFORMER (38 BLOCKS)                 |
 |                                                                                |
 |  [Planets MLP]  [Comets MLP]  [Fleets MLP]  [17 Control/Scratch Tokens]        |
 |       |              |             |                  |                        |
 |       +--------------+-------------+------------------+                        |
 |                                    |                                           |
 |                       Shared 768-d Embedding Space                             |
 |                                    |                                           |
 |              38 Residual Pre-Norm Transformer Blocks (16 Heads)                |
 |                                    |                                           |
 |            +-----------------------+-----------------------+                   |
 |            |                                               |                   |
 |    [Multi-Player Actor Head]                       [Global Critic Head]        |
 |    • Bernoulli Launch Decision                     • Softmax Win Probability   |
 |    • Query-Key Target Selection (Q·K^T / √d)         over active players       |
 |    • 8-Component Logistic Mixture Fleet Sizing                                 |
 +---------------------------------------+----------------------------------------+
                                         |
                                         v
 +--------------------------------------------------------------------------------+
 |                    KAGGLE SUBMISSION RUNTIME SERVING STACK                     |
 |  • Compressed Checkpoint: 4-bit NormalFloat (NF4-LSQ) Group 128 (90.7MiB)      |
 |  • Startup Streaming Dequantizer: Reconstructs weights tensor-by-tensor        |
 |  • Serving Latency Guardrail: Dynamic int8 CPU Linear inference                |
 |  • Fail-Safe Circuit Breaker: Auto-switch to 5M fallback model when bank < 1s  |
 +--------------------------------------------------------------------------------+
```

### 1.1 The Bitter Lesson in Competitive Machine Learning
Rather than spending months engineering hand-crafted gravitational trajectory heuristics, collision avoidance tables, or manual threat scoring rules, the author committed to three radical constraints:
1. **Fully Agentic Software Development**: Using OpenAI Codex to scaffold, implement, and maintain the codebase with documentation-driven human review.
2. **Minimal Inductive Biases**: Keeping raw entity observations and requiring the network to internalize continuous orbital mechanics and collision boundaries directly from self-play.
3. **Model Scaling**: Pushing the network architecture to **200 million parameters** and training for **15 billion environment steps** across 32× B200 GPUs.

### 1.2 The Single-Pass Multi-Player Forward Pass
Standard multi-agent reinforcement learning architectures evaluate each player's perspective in an independent forward pass ($N$ passes per board state in an $N$-player game). The 1st place solution unified the representation:
- All board entities (planets, comets, fleets) are projected into a shared attention trunk.
- Each player slot is allocated specialized player summary and actor plan tokens.
- **A single forward pass computes action distributions and critic win probabilities for all active players simultaneously**, yielding a **2× to 4× throughput multiplier** during rollout generation and PPO gradient updates.

---

## 2. Celestial Orbital Mechanics & Environment Dynamics

### 2.1 The Orbit Wars Universe
The environment models a continuous 2D planar solar system:
- **Central Star (Sun)**: Located at $(0, 0)$ with radius $R_{sun}$. Any fleet colliding with the sun is instantly vaporized.
- **Planets**: Rotate in circular orbits around the sun at fixed orbital radii $r_p$ and angular velocities $\omega_p = \sqrt{\frac{G M}{r_p^3}}$. Planets produce ships at rate $P_p$ per turn and possess gravitational capture radii.
- **Comets**: Follow eccentric linear or parabolic flyby paths through the system, carrying neutral ship caches that can be intercepted or colonized.
- **Fleets**: Launched from planets with initial velocity $v_{launch}$ along a heading angle $\theta$, subject to gravitational attraction and drag.

```
                      ORBITAL GEOMETRY & TRAJECTORY MECHANICS
                      
                                     .( Orbit 2 ).
                                 . '               ' .
                              .                         .
                            '     [Planet B]              '
                           '          *---> (Target)       '
                          '                                 '
                         '            /\                     '
                        '             | (Transfer Arc)        '
                       '              |                        '
                       '          +-------+                    '
                      '           |  SUN  |                     '
                      '           | (0,0) |                     '
                       '          +-------+                    '
                       '                                       '
                        '     [Planet A]                      '
                         '    (Source)                       '
                          '       *                         '
                           '                               '
                            '                             '
                              .                         .
                                 . '               ' .
                                     '( Orbit 1 )'
```

### 2.2 Action Space Evolution: From Raw Angles to Discrete Targets
The action space design underwent a critical evolution during the campaign:

#### The Continuous Failure Mode (Raw Angle + Fleet Size)
Initially, the policy directly predicted continuous launch actions per owned planet:
$$\mathbf{a} = (\text{launch} \in \{0, 1\}, \, \theta \in [-\pi, \pi], \, s \in [3, S_{available}])$$
**Result**: The model struggled to achieve competitive Elo. Learning precise Keplerian orbital transfer angles while avoiding solar destruction required thousands of hours of exploration just to discover valid transfer arcs.

#### The Discrete-Target Breakthrough
The author refactored the action space to high-level tactical intent:
$$\mathbf{a} = (\text{launch} \in \{0, 1\}, \, \text{target\_id} \in \{1, \dots, N_{planets}\}, \, s \in [3, S_{available}])$$
- The policy chooses **which planet to attack or reinforce**.
- A deterministic, high-speed analytical solver embedded in the environment computes the optimal launch angle $\theta^*$ that intercepts the moving target while clearing the sun's collision boundary.
- If no collision-free arc exists, the launch is rejected as a no-op.
This abstraction reduced sample complexity by orders of magnitude, freeing the 200M transformer to master macro-strategy, multi-front timing, and fleet allocation.

---

## 3. The Simulation Engine: Rust Rewrite & Parity Verification

The official Kaggle environment implemented in pure Python ran at ~120 steps per second per core. Training a 200M parameter model for 15 billion steps on this Python baseline would have required over **38,000 CPU-years**.

### 3.1 Rust Architecture (`src/rules_engine/`)
The author rewrote the entire simulation engine in Rust, binding it to Python via **PyO3** and orchestrating thousands of parallel environments using **Rayon**:

```rust
// Core vectorized environment step structure
pub struct VectorEnv {
    n_envs: usize,
    states: Vec<State>,
    cached_action_slots: Vec<ActionSlots>,
    pinned_obs_buffer: PinnedBuffer,
    thread_pool: rayon::ThreadPool,
}

impl VectorEnv {
    pub fn step_fused(&mut self, actions: &[ActionBatch]) -> StepResults {
        self.thread_pool.install(|| {
            self.states.par_iter_mut()
                .zip(&self.cached_action_slots)
                .zip(actions)
                .map(|((state, slots), action)| {
                    state.decode_and_step(slots, action)
                })
                .collect()
        })
    }
}
```

### 3.2 High-Throughput Optimizations
1. **Fused Step-Obs-Reset Loop**: Vectorized `step()` calls decode actions, advance physical state, calculate terminal rewards, auto-reset finished episodes, and write observation tensors into memory in a single Rayon parallel pass, eliminating redundant thread dispatch overhead.
2. **Zero-Copy Pinned Tensor Transfers**: State observations are written directly into preallocated, page-locked (pinned) CPU memory buffers. PyTorch tensors wrap these pointers with zero copy, streaming them asynchronously to GPU via CUDA streams:
   $$\text{Throughput}: 120,000+\text{ env steps/sec across 2048 parallel workers}.$$
3. **AABB Broad-Phase Collision Filtering**: Fleet-sun and fleet-planet collision tests avoid costly exact quadratic distance checks by first evaluating an axis-aligned bounding box expanded by radius $\epsilon$:
   $$\text{if } |x_{fleet} - x_{planet}| > R + \epsilon \lor |y_{fleet} - y_{planet}| > R + \epsilon \implies \text{Skip exact distance}.$$

### 3.3 Replay Parity Testing Framework
To guarantee that the Rust simulation matched Kaggle's official Python environment bit-for-bit, the author constructed an automated regression suite:
- Downloaded official Kaggle tournament match replays (e.g. Episode `75930761` for 2-player, `75926553` for 4-player).
- Fed the exact historical action sequences into both the Python reference and Rust engines.
- Enforced a hard gate: **100% state parity across every transition step** (planet ship counts, fleet coordinates, combat survival, and terminal reward assignment).

---

## 4. Scaled Transformer Policy & Value Architecture

The policy is a **38-block pre-norm residual Transformer** with 16 attention heads, 768-dimensional embedding space, and 1536 hidden MLP dimensions (~200 million parameters total).

```
                     TOKEN EMBEDDING & SEQUENCE COMPOSITION
                     
   Entity Inputs                        Control & Workspace Tokens
+-------------------------+          +-------------------------------+
| Planets: (N, 107) -> 768 |          | 4× Player Summaries  (768-d)  |
| Comets:  (C, 330) -> 768 |          | 1× Global Board Token (768-d) |
| Fleets:  (F, 79)  -> 768 |          | 4× Actor Plan Tokens  (768-d) |
+------------+------------+          | 4× Critic Value Tokens (768-d)|
             |                       | 4× Shared Scratch Tokens (768)|
             |                       +---------------+---------------+
             |                                       |
             +-------------------+-------------------+
                                 |
                                 v
         [Unified Sequence: (Batch, Total_Tokens, 768)]
                                 |
           +---------------------v---------------------+
           |   38-Block Pre-Norm Transformer Trunk     |
           |   • 16 Multi-Head Self-Attention Blocks   |
           |   • 1536-d MLP with GELU Activations      |
           |   • Residual Connections + LayerNorm      |
           +---------------------+---------------------+
                                 |
                 +---------------+---------------+
                 |                               |
                 v                               v
    [Per-Player Actor Decoders]       [Multi-Player Critic Head]
    • Bernoulli Launch Logit          • MLP over 4 Value Tokens
    • Query-Key Target Dot-Product    • Softmax Win Probabilities
    • 8-Mixture Logistic Fleet Size     (p_1, p_2, p_3, p_4)
```

### 4.1 Token Schema & Sequence Structure
Each observation frame is mapped into a variable-length token sequence:
1. **Entity Tokens**:
   - **Static Planets** ($107 \to 768$ MLP stem): Radial distance, orbital velocity, current ship count, owner one-hot, production rate.
   - **Orbiting Planets** ($107 \to 768$ MLP stem): Current Keplerian coordinates $(x, y, v_x, v_y)$, phase angle, target incoming fleet tally.
   - **Comets** ($330 \to 768$ MLP stem): Piecewise linear path segments, entry/exit horizon times, carry ship reserves.
   - **Fleets** ($79 \to 768$ MLP stem): Current $(x, y)$, velocity vector, ship mass, owning player ID, target planet destination.
2. **17 Specialized Control Tokens**:
   - **Player Summary Tokens ($4 \times 768$-d)**: Encodes per-player macro state (total fleet count, total planet production, territory percentage). Added to learned player embeddings.
   - **Global Summary Token ($1 \times 768$-d)**: Encodes global game clock, total alive player count, and remaining turn budget.
   - **Actor Plan Tokens ($4 \times 768$-d)**: Learned embeddings serving as routing anchors for actor policy heads.
   - **Critic Value Tokens ($4 \times 768$-d)**: Learned representations pooled to evaluate per-player terminal outcomes.
   - **Board Scratch Tokens ($4 \times 768$-d)**: Pure unconstrained learned tokens serving as an attention workspace for cross-entity message passing.

### 4.2 Actor Head Formulation
For each active player $p$ and eligible launch origin planet $i$:
1. **Launch Decision (Bernoulli Head)**:
   The entity representation $\mathbf{h}_i$ is concatenated with player plan token $\mathbf{z}_p$ and projected to a scalar logit:
   $$P(\text{launch} \mid i, p) = \sigma(\mathbf{W}_{\text{launch}} [\mathbf{h}_i \,\|\, \mathbf{z}_p] + b_{\text{launch}})$$
2. **Target Selection (Query-Key Matching)**:
   If the planet launches, it computes a query $\mathbf{Q}_i = \mathbf{W}_Q \mathbf{h}_i$. Every candidate destination planet $j$ computes key $\mathbf{K}_j = \mathbf{W}_K \mathbf{h}_j$. The categorical target distribution is:
   $$P(\text{target} = j \mid i, p) = \frac{\exp\left(\frac{\mathbf{Q}_i \cdot \mathbf{K}_j}{\sqrt{d}}\right)}{\sum_{k \in \mathcal{V}_{valid}} \exp\left(\frac{\mathbf{Q}_i \cdot \mathbf{K}_k}{\sqrt{d}}\right)}$$
3. **Fleet Sizing (Truncated Discretized Logistic Mixture)**:
   The selected target's value vector $\mathbf{V}_j$ is added into the source stream. The combined representation feeds into an 8-component discretized logistic mixture model over the valid launch range $[3, S_{available}]$:
   $$P(s \mid i, j, p) = \sum_{m=1}^{8} \pi_m \left[ \sigma\left(\frac{s + 0.5 - \mu_m}{\sigma_m}\right) - \sigma\left(\frac{s - 0.5 - \mu_m}{\sigma_m}\right) \right]$$

### 4.3 Multi-Player Critic Head
Instead of estimating arbitrary state values, the critic evaluates game-theoretic win probabilities. The 4 critic value tokens $\mathbf{v}_1, \dots, \mathbf{v}_4$ are passed through an MLP to generate scalar logits for all players, normalized via a masked softmax:
$$\hat{\mathbf{p}}_{win} = \text{Softmax}\left(\text{MLP}(\mathbf{v}_1), \dots, \text{MLP}(\mathbf{v}_4)\right)$$
For 2-player zero-sum matches, the scalar advantage is computed from the probability delta:
$$V(s) = 2 \cdot \hat{p}_{win} - 1 \in [-1, 1]$$

---

## 5. Reinforcement Learning Pipeline & Distributed Scaling

### 5.1 Proximal Policy Optimization (PPO) Setup
Training was conducted with clipped PPO across massive rollout buffers:
- **Rollout Length**: $T = 64$ steps per environment.
- **Parallel Environments**: 8,192 concurrent games across 4 nodes.
- **Batch Size**: $8,192 \times 64 = 524,288$ environment steps per rollout iteration.
- **Advantage Estimation**: Generalized Advantage Estimation (GAE-λ) with $\lambda = 0.95$, $\gamma = 1.0$.
- **Surrogate Objective**:
  $$\mathcal{L}_{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$
  with clipping parameter $\epsilon = 0.20$.

### 5.2 Teacher Distillation & Stabilization
To prevent policy collapse and catastrophic forgetting during multi-billion-step self-play, training was stabilized using a **Last-Best Teacher Distillation Loss**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PPO}} + c_{\text{val}} \mathcal{L}_{\text{critic}} + c_{\text{ent}} \mathcal{H}(\pi) + \alpha_{\text{KL}} D_{\text{KL}}(\pi_{\text{teacher}} \,\|\, \pi_{\theta}) + \alpha_{\text{CE}} \mathcal{L}_{\text{CE}}(V_{\text{teacher}}, V_{\theta})$$

#### Checkpoint Promotion Protocol
1. The current training policy checkpoints periodically every 20 million steps.
2. The candidate plays an evaluation gauntlet of 2,048 benchmark games against the current `checkpoint_last_best.pt` (with seats randomly permuted).
3. **Hard Promotion Gate**: The candidate replaces the teacher **only if it achieves $\ge 70\%$ win rate**.
4. If promoted, the new checkpoint becomes the reference anchor for policy KL and value distillation.

### 5.3 Distributed Training Infrastructure
- **Compute Cluster**: 4 nodes × 8× NVIDIA B200 GPUs (32 GPUs total).
- **Communication Backend**: PyTorch DistributedDataParallel (DDP) with NCCL.
- **Optimization**: Muon optimizer for large 2D transformer weight matrices combined with AdamW for 1D embeddings and normalization layers; bfloat16 mixed-precision autocast.
- **Throughput**: ~6.3 million environment steps per GPU-hour (~110,000 unmasked tokens per GPU-second).
- **Total Training Budget**: 15 billion steps completed in ~2,400 B200 GPU-hours.

---

## 6. Submission Engineering Under Hard Constraints

Kaggle simulation rules enforce two non-negotiable constraints:
1. **Submission Artifact Size Limit**: $\le 100\text{ MiB}$ total compressed archive.
2. **Turn Latency Limit**: $1.0\text{ second}$ per turn, plus a shared $60.0\text{ second}$ overage bank for the entire game, running on a single slow x86 CPU core.

A standard fp32 checkpoint for a 200M parameter model is $\sim 800\text{ MiB}$ and takes over $2.5\text{ seconds}$ per inference pass on a single CPU core. Deploying this model required a three-tier compression and inference stack:

```
                  SUBMISSION COMPRESSION & SERVING PIPELINE
                  
 [200M Model (800MiB fp32)]
              |
              v
 [4-bit NormalFloat (NF4-LSQ) Group 128 Compression]
  • 2D weight matrices -> 4-bit NF codebook indices (0..15)
  • Per-group of 128 weights -> fp16 scale with least-squares refinement
  • Checkpoint size shrinks from 800MiB to 90.7MiB!
              |
              v
 [Kaggle submission.tar.gz (<100MiB Cap)]
              |
              | (Agent Startup on Kaggle CPU)
              v
 [Streaming Dequantization Loader]
  • Streams tensors one-by-one directly into CPU memory
  • Avoids materializing full 800MiB state-dict at once
              |
              v
 [Dynamic int8 CPU Inference]
  • Linear layers dynamically quantized to int8
  • Latency drops from 2500ms to 450ms per step
              |
              v
 [Turn Execution & Safety Cascade]
  • Step time < 1000ms: Normal 200M play
  • Remaining Bank Time < 1.0s: Immediate Fallback to 5M Checkpoint
```

### 6.1 Checkpoint Compression: Grouped NormalFloat (NF4-LSQ)
To compress 200M parameters into $<100\text{ MiB}$, the author implemented custom **4-bit NormalFloat (NF4)** codebook quantization with **Group Size 128** and **Least-Squares (LSQ) Scale Refinement**:
- Weights in each linear layer are divided into blocks of 128 values.
- Each block is normalized by an fp16 scale factor:
  $$s^* = \arg\min_s \sum_{i=1}^{128} \left( w_i - s \cdot \mathcal{Q}_{\text{NF4}}(w_i / s) \right)^2$$
- Each weight is mapped to the nearest entry in the 16-element zero-mean Gaussian quantile codebook.
- **Results**: Checkpoint size dropped from **800 MiB (fp32) to 90.7 MiB (NF4-LSQ)**.
- In head-to-head self-play, the 90.7MiB quantized model retained **~40% win rate against the unquantized fp32 teacher** (compared to <10% for naive uniform 4-bit quantization).

### 6.2 Serving Latency Optimization: Dynamic int8 CPU Execution
On the Kaggle evaluation CPU, fp32 transformer evaluation took 2.5s per turn. The author applied **PyTorch dynamic int8 quantization** (`torch.ao.quantization.quantize_dynamic`) to all non-output `nn.Linear` layers upon startup:
- Quantizes weights to signed int8 and dynamically quantizes activations during runtime matmuls.
- Reduces inference latency from **2,500ms down to 420–550ms per turn**.
- Final actor and critic heads are kept in fp32 to prevent probability distribution distortion.

### 6.3 Observation Compaction & Fleet Pruning
In complex 4-player games with hundreds of active fleets, attention cost scales quadratically with entity count:
$$\mathcal{O}((N_{planets} + N_{comets} + N_{fleets})^2)$$
To maintain latency within 1.0s:
- The observation encoder discards minor fleets below a threshold $\tau_{fleet} < 3$ ships.
- **Safety Invariant**: If a player has lost all planets and relies solely on tiny fleets, the encoder always preserves that player's single largest fleet, preventing premature elimination classification.

### 6.4 Fail-Safe Fallback Cascade
In approximately 8% of 4-player matches, the agent was assigned to severely degraded or throttled Kaggle CPU nodes where turn times exceeded 1.0s, depleting the 60s overage bank.
- **Circuit Breaker**: The agent tracks `observation.remainingOverageTime`.
- **The Handoff**: If overage time drops below **1.0 second**, the agent dynamically swaps its model pointer to a packaged **5M-parameter lightweight transformer**.
- **Result**: The 5M fallback executed in $<40\text{ ms}$ per turn. According to the critic, matches reaching late turns were already strategically decided, and the 5M model achieved a **100% win conversion rate** from winning positions.

---

## 7. Critical Post-Mortem Insights & Anti-Patterns

### 7.1 The Stalling Bug ($\gamma = 1.0$)
- **The Problem**: In 4-player games, setting the discount factor $\gamma = 1.0$ ensured that terminal win probabilities were theoretically exact. However, without discounting, the agent had zero incentive to win *now* rather than 300 turns later. Once the agent secured an overwhelming ship lead, it would enter a perpetual stall pattern—circling planets and hoarding fleets without attacking.
- **Training Impact**: Billions of training environment steps were wasted simulating already-decided games.
- **The Fix**: Future competitive RL setups must incorporate early termination, surrender conditions, or a mild time penalty ($r_t = -0.001$ per step) to maximize state exploration density.

### 7.2 The Matchmaking Inversion & The Need for League Play
- **The Trap**: During the active competition, the public leaderboard matchmaking placed top bots overwhelmingly into **2-player games** (~90% 2p). The author optimized training throughput by setting `two_player_weight = 0.90` and omitted league play against past checkpoints to maximize self-play throughput.
- **The Disaster**: After the final submission deadline, Kaggle inverted tournament matchmaking, weighting final private evaluation heavily toward **4-player games**.
- **The Game-Theoretic Lesson**: Unlike 2-player zero-sum games where self-play converges to minimax optimality, 4-player games exhibit severe **non-transitive strategic cycles** (Rock-Paper-Scissors dynamics and kingmaker coalitions). Pure self-play caused the agent to overfit to its own policy trajectory. Incorporating **AlphaStar-style League Play** (evaluating against a diverse pool of historical and exploit-seeking checkpoints) is mandatory for multi-agent games.

### 7.3 The Action Masking Paradox
- **The Assumption**: Conventional RL practice suggests applying action masks to forbid illegal or suicidal actions (e.g. launching fleets straight into the burning sun).
- **The Finding**: Training with an action mask that prohibited sun collisions produced an agent with **substantially worse final Elo**.
- **The Rationale**: Forcing the model to suffer the consequences of bad launches compelled the 200M transformer to internally simulate the gravitational physics and celestial collision geometry of the solar system. The author removed the mask during training, re-introducing it only for a brief final fine-tuning phase and runtime serving.

### 7.4 Fully Agentic Development Reflections
- Software development was driven by LLM coding agents (OpenAI Codex).
- **What Worked**: Codex accelerated environment scaffolding, PyO3 bindings, and PPO boilerplate from months to weeks.
- **What Failed**: Coding agents frequently suggested overly complex domain adaptations (cross-attention tricks, discrete bins) that underperformed compared to scaling raw model capacity. Human guidance remained vital for high-level research direction, hypothesis testing, and rigorous doc review.

---

## 8. The Top-10 Solution Breakdown & Architectural Paradigms

While the 1st place solution demonstrated the supremacy of pure compute and large-scale model expressivity, the remaining prize-winning solutions (2nd through 10th place) uncovered vital complementary engineering paradigms—from progressive imitation curricula and frugal JAX optimizations to relational edge-attention and test-time rollout search.

```
                      THE 4 COMPETITIVE PARADIGMS OF ORBIT WARS
                      
 Paradigm A: Brute Scale & Self-Play
 🥇 1st Place (IsaiahPressman): 200M Transformer, 15B steps, pure self-play PPO, Rust accelerator
                                      ▲
                                      │
 ┌────────────────────────────────────┼────────────────────────────────────┐
 │                                    │                                    │
 ▼                                    ▼                                    ▼
 Paradigm B: Progressive Curricula    Paradigm C: Pure JAX Pipelines       Paradigm D: Inductive Biases & Search
 🥈 2nd (simjeg): Heuristics->BC->RL  🥉 3rd (Felix): JAX PPO              🏅 6th (flg): Edge-Attention + 2-step lookahead
 🏅 5th (TonyK): IMPALA + Replay BC   🏅 8th (Bradley): <$200 JAX + 2D RoPE 🏅 9th (Boey): "Planet Future" garrison forecasts
                                      🏅 9th (Boey): 100% JIT zero-copy    🏅 10th (Liu): Strategic/Physics hierarchy
                                      🏅 10th (Liu): JAX + C++ Solver
```

---

### 8.1 2nd Place Solution (`simjeg`): Progressive Imitation-to-RL Curriculum & Anti-Stall Rewards

**Author**: `simjeg` | **Final Rank**: 🥈 2nd Place Silver  
**Core Innovation**: Progressive 4-stage training curriculum and step-conditioned reward decay.

#### The Progressive Development Ladder
Rather than training a massive network from scratch on Day 1, `simjeg` built a step-by-step competitive progression:
1. **Stage 1 (Heuristic & Search Bots)**: Built a hand-crafted rule-based agent to establish strong baseline game understanding, breaking into the Top 50.
2. **Stage 2 (Behavioral Cloning / Imitation Learning)**: Collected tournament replays from top-performing leaderboard bots and trained a compact neural policy via supervised behavioral cloning (cross-entropy loss over historical expert actions). This leapfrogged the agent directly into the Top 10 without spending millions of steps exploring random actions.
3. **Stage 3 (Reinforcement Learning Fine-Tuning)**: Initialized an actor-critic model with the imitation weights and fine-tuned using self-play PPO, reaching the Top 5.
4. **Stage 4 (From-Scratch RL Realignment)**: Having identified optimal hyperparameters, reward scalings, and network structures during fine-tuning, the final submission was trained from scratch with RL to avoid imitating human sub-optimal heuristics.

#### Step-Conditioned Anti-Stall Reward Shaping
To solve the $\gamma=1.0$ stalling crisis (where winning bots refuse to finish games), `simjeg` engineered an elegant piecewise step-decay reward:
$$R_{\text{terminal}} = \begin{cases} +1.0 & \text{if Player Wins in } t < 500 \text{ steps} \\ +0.5 & \text{if Player Wins in } t \ge 500 \text{ steps} \\ -1.0 & \text{if Player Loses} \end{cases}$$
*Impact*: The model learned that delaying victory halved its reward. Stalling behavior was eliminated overnight, dramatically increasing training state diversity and finishing tournament matches with ruthless efficiency.

---

### 8.2 3rd Place Solution (`Felix M. Neumann`, "Ab in den Orbit")

**Author**: Felix M. Neumann | **Final Rank**: 🥉 3rd Place Bronze  
**Core Innovation**: High-throughput self-play PPO and Transformer policy accelerated natively in JAX.

#### JAX-Native Vectorized Self-Play
Neumann leveraged **JAX** (`jax.jit`, `jax.vmap`) to maintain high training throughput. By avoiding Python-GPU transfer barriers, the self-play PPO agent scaled across multi-GPU hardware, training an expressive Transformer policy that excelled at multi-planet coordination and synchronized fleet arrivals.

---

### 8.3 5th Place Solution (`TonyK`): Asynchronous IMPALA + Frozen Historical Opponents

**Author**: `TonyK` | **Final Rank**: 🏅 5th Place  
**Core Innovation**: Distributed Asynchronous IMPALA with V-trace, Replay Behavioral Cloning initialization, and frozen historical opponent pools.

#### The IMPALA Distributed Architecture
While 1st place utilized synchronous on-policy PPO, `TonyK` implemented **IMPALA (Importance Weighted Actor-Learner Architecture)**:
- Decoupled parallel actors running continuous simulation on CPU from a centralized GPU learner.
- Corrected for policy lag between actor rollouts and learner gradients using **V-trace importance sampling weights**:
  $$v_s = V(x_s) + \sum_{t=s}^{s+k-1} \gamma^{t-s} \left( \prod_{i=s}^{t-1} c_i \right) \delta_t V$$
  where $c_i = \min\left(\bar{c}, \frac{\pi(a_i \mid x_i)}{\mu(a_i \mid x_i)}\right)$ and $\delta_t V = \rho_t \left( r_t + \gamma V(x_{t+1}) - V(x_t) \right)$.

#### Delayed Moving Teacher & Historical Opponent Pools
To prevent cyclic forgetting in multi-agent games:
1. **Delayed Moving Teacher (Polyak Average)**: Maintained an exponential moving average (EMA) of network weights to serve as a stable distillation anchor ($\theta_{\text{teacher}} \leftarrow \tau \theta_{\text{teacher}} + (1 - \tau) \theta$).
2. **Frozen Historical Opponent Pool**: Actors did not solely play the current policy. 25% of match seats were assigned to historical policy checkpoints frozen at earlier epochs. This maintained strategic pressure against rush tactics and prevented the agent from developing blind spots to discarded strategies.

---

### 8.4 6th Place Solution (`flg`): Relational Edge-Attention & 2-Step Lookahead Search

**Author**: `flg` | **Final Rank**: 🏅 6th Place  
**Core Innovation**: Custom Edge-Attention Transformer (2.5M params) and test-time 2-step rollout lookahead search.

#### Custom Relational Edge-Attention
Standard self-attention computes query-key affinity solely from node embeddings: $\frac{\mathbf{q}_i \mathbf{k}_j^T}{\sqrt{d}}$. In Orbit Wars, the tactical relationship between two planets depends heavily on **pairwise physical transit geometry**.  
`flg` developed an **Edge-Attention Transformer** where explicit edge features $\mathbf{e}_{i,j}$ bias the attention logits directly:
$$\mathbf{A}_{i,j} = \text{Softmax}\left(\frac{\mathbf{q}_i \mathbf{k}_j^T}{\sqrt{d}} + \mathbf{W}_e \mathbf{e}_{i,j} + b_e\right)$$
where $\mathbf{e}_{i,j}$ encodes:
- Gravitational transfer flight duration $\Delta t_{i \to j}$.
- Solar obstacle proximity along the transfer arc.
- Estimated defensive garrison at impact time.
*Impact*: The model required only **2.5 million parameters** to achieve grandmaster-level tactical awareness, drastically outperforming standard transformers of equivalent parameter size.

#### 2-Step Lookahead Rollout Search
During runtime inference in 2-player matches, `flg` augmented the neural policy with a **2-step Monte Carlo lookahead search**:
1. Sample top-$K$ candidate launch actions from the neural policy $\pi_\theta(a \mid s)$.
2. Simulate the forward state $s_{t+1}$ using a fast C++ forward simulator.
3. Evaluate the successor states using the learned critic $V(s_{t+1})$.
4. Execute the action that maximizes the expected 2-step value return.

---

### 8.5 7th Place Solution (`Audun Ljone Henriksen & Eirik Torp`): Component-Isolated Experimentation

**Authors**: Audun Ljone Henriksen & Eirik Torp | **Final Rank**: 🏅 7th Place  
**Core Innovation**: Rigorous component isolation and internal Elo validation ladders.

#### Systematic Decomposition
The authors addressed the notorious difficulty of debugging RL policies by isolating sub-systems:
- Built a modular testbench separating **macro-economic targeting** from **low-level collision-avoidance solvers**.
- Maintained a continuous internal Elo rating system playing thousands of validation matches against a fixed suite of diverse benchmark bots (heuristic aggressors, defensive expanders, previous top models).
- Enforced a rule: *No neural network or architectural change was submitted to Kaggle unless it proved a statistically significant Elo gain on the local benchmark ladder.*

---

### 8.6 8th Place Solution (`Billy Bradley`, "Ender for <$200"): Frugal JAX RL & 2D RoPE

**Author**: Billy Bradley | **Final Rank**: 🏅 8th Place  
**Core Innovation**: Training an elite agent on a single GPU for under $200 using JAX, 2D Rotary Position Embeddings (2D RoPE), and autoregressive micro-steps.

#### 2D Rotary Position Embeddings (2D RoPE) for Orbital Space
Standard transformers use learnable 1D position embeddings, which fail to capture continuous 2D planar distances and rotations. Bradley applied **2D Rotary Position Embeddings (2D RoPE)** to celestial coordinates $(r, \theta)$ or $(x, y)$:
- Given coordinates $\mathbf{x}_i = (x_i, y_i)$, the query and key vectors are rotated in the complex plane across partitioned channel pairs:
  $$\mathbf{q}_i^{(m)} = \mathbf{R}_{\Theta, x_i}^{(m)} \mathbf{q}_i^{(m)}, \quad \mathbf{k}_j^{(m)} = \mathbf{R}_{\Theta, x_j}^{(m)} \mathbf{k}_j^{(m)}$$
- The resulting dot-product $\mathbf{q}_i \cdot \mathbf{k}_j$ depends naturally on the **relative Euclidean displacement** $(\mathbf{x}_i - \mathbf{x}_j)$ and relative orbital angle, providing the exact inductive bias required for Keplerian space mechanics without rigid grids.

#### Autoregressive Micro-Steps
Instead of predicting a single massive joint action vector for all 44 planets at once, Bradley's agent generated moves sequentially across autoregressive micro-steps:
$$\pi(\mathbf{a}) = \prod_{k=1}^{M} \pi(a_k \mid a_{<k}, s)$$
This allowed early planet launches to dynamically condition later planet targeting decisions within the same turn.

---

### 8.7 9th Place Solution (`Boey`, "End-to-End JAX PPO"): Pure JAX JIT & "Planet Future" Forecasts

**Author**: Boey | **Final Rank**: 🏅 9th Place  
**Core Innovation**: 100% pure JAX JIT-compiled pipeline with zero CPU-GPU transfer overhead and forward-simulated "Planet Future" state projections.

#### The Zero-Overhead JAX JIT Pipeline
While 1st place used Rust on CPU with async CUDA streams, Boey implemented the **entire training loop in JAX**:
- Simulation environment physics, observation tensor construction, action masking, rollout trajectory collection, and PPO loss backprop were compiled into a **single unified GPU/TPU kernel** via `jax.jit` and vectorized across parallel games with `jax.vmap`.
- **Zero Python or PCIe bottleneck**: Simulation and optimization resided continuously on the GPU accelerator, achieving multi-thousand steps/sec on modest single-node hardware.

#### "Planet Future" Trajectory Projections
Recognizing that planets have predictable circular orbits and fleet arrival times are deterministic, Boey engineered the **"Planet Future" tensor**:
- For each planet $i$ and future lookahead step $k \in \{1, 2, 5, 10, 20\}$, the environment forward-simulated the expected garrison:
  $$\hat{G}_i(t + k) = G_i(t) + k \cdot P_i + \sum_{\text{friendly fleets}} S_f - \sum_{\text{hostile fleets}} S_h$$
- Passing this explicit future horizon directly into the entity transformer allowed the policy to make proactive defensive reinforcements 15 turns before hostile fleets arrived.

---

### 8.8 10th Place Solution (`Xiangyu Liu`): Hierarchical Decoupled Architecture

**Author**: Xiangyu Liu | **Final Rank**: 🏅 10th Place  
**Core Innovation**: Clean architectural decoupling between geometric caching, state simulation, and neural policy.

#### The 3-Tier Systems Architecture
Liu decoupled the Orbit Wars problem into three independent modules:
1. **`MapCache` (Static Geometry)**: Precomputed static orbital trajectories, sun hazard arcs, and feasible Keplerian transfer windows.
2. **`StrategicEnv` (Macro Simulation)**: A lightweight JAX simulation handling only discrete state changes—planet ownership, ship production, and fleet collision combat.
3. **`JAX PPO Agent` + C++ Intercept Solver**: The neural network operates purely on a compacted 44-planet observation space (folding fleet metrics into planet arrival buckets). At test time, a specialized C++ solver translates the network's high-level discrete planet decisions into continuous intercept launch angles.

---

## 9. Comprehensive Takeaways: The Competitive RL Hierarchy

Synthesizing all prize-winning solutions (1st through 10th place) reveals an unambiguous roadmap for winning competitive simulation challenges:

1. **Compute vs. Inductive Bias Tradeoff**:
   - **If you have massive compute (1st Place)**: Follow Sutton's Bitter Lesson. Scale to a 200M Transformer, train for 15B steps on Rust/Rayon vector envs, and let the model internalize physics unassisted.
   - **If you have a modest compute budget (6th, 8th, 9th Place)**: Use domain inductive biases! Add **2D RoPE** (Bradley), **Edge-Attention** (flg), and **Planet Future Projections** (Boey) to achieve Top-10 performance on a 2.5M model for under $200.
2. **Simulation Acceleration is Mandatory**:
   - Every top-10 team rewrote the simulation environment: **Rust (1st Place)**, **JAX JIT (3rd, 8th, 9th, 10th Place)**, or **C++ (6th Place)**. Standard Python baselines are dead on arrival.
3. **Imitation Warm-Starts Accelerate Early Elo**:
   - If initial exploration in continuous physics is too sparse, use **Behavioral Cloning on tournament replays** (2nd Place `simjeg`, 5th Place `TonyK`) to seed tactical competency before launching PPO/IMPALA.
4. **Solve Multiplayer Game Dynamics**:
   - Never rely on pure self-play in $N \ge 3$ player games. Deploy **frozen historical opponent pools** (`TonyK`) and **AlphaStar league matchmaking** to eliminate non-transitive Rock-Paper-Scissors cycles.
5. **Mitigate the Stalling Bug Early**:
   - Always implement **step-decay rewards** (`simjeg`: $+1.0$ early vs $+0.5$ late) or surrender thresholds to prevent winning agents from stalling for hundreds of turns.
6. **Deploy Turn-Time Circuit Breakers**:
   - Package a micro-fallback model (1st Place: 5M model, $<30\text{ms}$) triggered by an overage bank threshold ($<1.0\text{s}$) to guarantee zero disqualifications on throttled Kaggle CPU workers.
