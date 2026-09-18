# Orbit Wars: Scaling Reinforcement Learning to the Stars 🚀

**Competition**: [Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)  
**1st Place Solution**: *Scaling Reinforcement Learning to the Stars* by Isaiah Pressman ([Writeup](https://www.kaggle.com/competitions/orbit-wars/writeups/1st-place-solution-scaling-reinforcement-learnin))  
**Open-Source Repository**: [`IsaiahPressman/kaggle-orbit-wars`](https://github.com/IsaiahPressman/kaggle-orbit-wars)  
**Track**: Competitive Multi-Agent Reinforcement Learning & Continuous Simulation  
**Core Methodology**: 200M-Parameter Entity Transformer, 15 Billion Steps Pure Self-Play PPO, Rust Engine Acceleration, NF4/NF5 Quantization, and Multi-Tier CPU Fallback Serving  
**Hardware Cluster**: 4 Nodes × 8× NVIDIA B200 GPUs (32× B200 total), 8,192 Parallel Environments (~2,400 B200-hours)

---

### Official Winning Scoreboard & Technical Paradigm Comparison

| Rank / Competitor | Core Architecture & Engine | Training Scale | Key Innovations & Strategic Paradigms | Final Outcome |
| :--- | :--- | :--- | :--- | :--- |
| 🥇 **1st Place (`IsaiahPressman`)** | **200M Parameter Transformer**<br>High-speed **Rust Simulator** | **15 Billion Steps**<br>(~2,400 B200-hours,<br>4× 8×B200 cluster) | • **Bitter Lesson validation**: pure self-play RL over human heuristics<br>• **Single-pass multi-player inference**: all agents predicted in 1 forward pass<br>• **Rust environment rewrite**: 120,000+ steps/sec with zero-copy pinned buffers<br>• **Sub-100MiB NF4 compression**: 4-bit NormalFloat group-128 LSQ quantization<br>• **Runtime CPU fallback cascade**: dynamic int8 + 5M fallback when overage < 1s | 🥇 **1st Place Gold**<br>Dominant tactical orbital dominance across 2p and 4p games |
| 🥈 **2nd Place (`@re-writeup`)** | **Heuristic-to-PPO Transition**<br>PufferLib Vectorization | ~1–2 Billion Steps<br>(8× RTX 4090s) | • PufferLib accelerated rollout harness<br>• Imitation learning warm-start from heuristic bot trajectories<br>• Hybrid rule-based emergency overriding for sun collisions | 🥈 **2nd Place Silver** |
| 🏅 **9th Place (`@jax-team`)** | **JAX Custom Simulator**<br>Reachability Tensor Policy | ~3 Billion Steps<br>(TPU v4 cluster) | • Fully vectorized simulator written in JAX with end-to-end GPU stepping<br>• 3D "Reachability Tensor" tracking fleet arrival horizons across planets<br>• Redundant state encoding with "Calendar" histogram summary tokens | 🏅 **9th Place Top 10** |

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

## 8. Summary of Reusable Architectural Patterns

1. **Sutton's Bitter Lesson**: When computational resources permit, scale expressivity and training throughput rather than hand-tuning domain features.
2. **Unified Multi-Agent Forward Pass**: In $N$-player environments, output all agents' action distributions from a single shared transformer sequence.
3. **Discrete Intent + Analytical Physics**: Combine discrete strategic neural selection (`source`, `target`) with deterministic continuous solvers for physical execution.
4. **Grouped NormalFloat Quantization**: Use NF4/NF5 codebooks with LSQ block scale refinement to compress large networks into constrained deployment packages.
5. **Runtime Fallback Cascades**: Pair heavy champion models with micro-architectures managed by runtime latency circuit breakers.
