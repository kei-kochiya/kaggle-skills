---
name: kaggle-rl-simulation
description: Use when building, training, optimizing, or deploying reinforcement learning agents, neural policies, and simulation environments for competitive games or simulation challenges on Kaggle.
---

# Competitive Reinforcement Learning & Simulation Playbook 🚀

## Overview

Competitive game and simulation challenges on Kaggle (e.g. Orbit Wars, Lux AI, Kore, Halite, Google Football) present a fundamentally different paradigm from tabular or static vision/NLP tracks. Success is governed by **Sutton's Bitter Lesson**: *expressive neural architectures and extreme simulation throughput scale far higher than hand-crafted human heuristics*.

This skill operationalizes the complete engineering lifecycle to develop, train, stabilize, and deploy gold-medal reinforcement learning agents under strict submission constraints.

```dot
digraph rl_pipeline {
    rankdir=TD;
    node [shape=box, style="rounded,filled", fillcolor="#f8f9fa", color="#343a40", fontname="Helvetica"];
    
    "1. Simulation Architecture" -> "Vectorizable Tensor Logic?" [shape=diamond, fillcolor="#e9ecef"];
    "Vectorizable Tensor Logic?" -> "Pure JAX GPU Pipeline (Zero Host-Device Transfer)" [label="Yes (Dense Tensor)"];
    "Vectorizable Tensor Logic?" -> "Rust / C++ Engine (PyO3/Rayon + Pinned Zero-Copy)" [label="No (Irregular Graph/Branching)"];
    "Pure JAX GPU Pipeline (Zero Host-Device Transfer)" -> "2. Policy Initialization";
    "Rust / C++ Engine (PyO3/Rayon + Pinned Zero-Copy)" -> "2. Policy Initialization";
    
    "2. Policy Initialization" -> "Complex Action Space / Sparse Rewards?" [shape=diamond, fillcolor="#e9ecef"];
    "Complex Action Space / Sparse Rewards?" -> "Bootstrap: Behavioral Cloning on Heuristics / Replays" [label="Yes"];
    "Complex Action Space / Sparse Rewards?" -> "Direct RL from Scratch" [label="No"];
    "Bootstrap: Behavioral Cloning on Heuristics / Replays" -> "3. Neural Architecture Design";
    "Direct RL from Scratch" -> "3. Neural Architecture Design";
    
    "3. Neural Architecture Design" -> "Entity Transformer + Relational Edge-Attention & 2D RoPE";
    "Entity Transformer + Relational Edge-Attention & 2D RoPE" -> "4. Stabilized RL Optimization";
    
    "4. Stabilized RL Optimization" -> "Multiplayer / Non-Transitive Dynamics?" [shape=diamond, fillcolor="#e9ecef"];
    "Multiplayer / Non-Transitive Dynamics?" -> "AlphaStar League / Frozen Historical Opponent Pool" [label="Yes"];
    "Multiplayer / Non-Transitive Dynamics?" -> "Teacher Distillation + Step-Conditioned Anti-Stall Rewards" [label="No (2-Player)"];
    "AlphaStar League / Frozen Historical Opponent Pool" -> "5. Submission Deployment Stack";
    "Teacher Distillation + Step-Conditioned Anti-Stall Rewards" -> "5. Submission Deployment Stack";
    
    "5. Submission Deployment Stack" -> "Sub-100MiB NF4-LSQ Quantization + Dynamic int8 + Test-Time Lookahead Search";
}
```

---

## The 5-Phase Competitive RL Protocol

### Phase 1: Environment Diagnostics & Acceleration
*Before writing neural network code, benchmark the raw simulation throughput.*
1. **Benchmark Baseline Throughput**: If the official Python environment runs below $5,000\text{ steps/sec}$, full self-play will fail to converge within reasonable compute budgets.
2. **Select Acceleration Backend**:
   - **Pure JAX on GPU (`jax.jit`, `jax.vmap`)**: When dynamics can be vectorized into fixed-dimension tensors. Eliminates CPU-to-GPU memory transfer bottlenecks entirely, reaching up to $500,000+\text{ sps}$ on a single GPU (used by 8th place *Bradley* & 9th place *Boey*).
   - **Compiled Rust (PyO3 + Rayon) or C++**: When game rules require dynamic allocations, variable-length event queues, or irregular graph traversals. Use Rayon thread pools with preallocated pinned CPU memory buffers wrapped directly by PyTorch tensors (used by 1st place *Pressman*, 6th place *flg*, 10th place *Liu*).
3. **Replay Parity Gate**: Validate that the compiled engine reproduces official tournament match JSON replays bit-for-bit across every turn before launching training runs.
> **Detailed Guide:** See [`references/simulator_acceleration.md`](references/simulator_acceleration.md) for Rust Rayon bindings, JAX vectorization patterns, and parity regression harnesses.

---

### Phase 2: Observation & Action Space Structuring
*Structure representations to maximize learnability and geometric fidelity.*
1. **Entity-Based Observations**: Represent game boards as sets of distinct entity tokens (planets, units, obstacles) rather than rigid spatial grids.
2. **Continuous 2D Rotary Position Embeddings (2D RoPE)**:
   - For environments with continuous coordinates $(x, y)$, project 2D coordinates into orthogonal frequency bands. Preserves continuous relative distance and angle in self-attention without artificial grid discretization (8th place *Bradley*).
3. **Relational Edge Tensors & Edge-Attention**:
   - Pairwise features (flight times, Euclidean distances, collision risks) should modulate self-attention logits directly: $\text{Attention}(Q, K, E) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}} + W_e E\right) V$. This provides an inductive bias for physics and topology without deep graph convolutions (6th place *flg*).
4. **Deterministic Future Forecasting Tensors ("Planet Future")**:
   - Compute exact future arrivals of in-flight fleets over the next $K$ turns analytically and feed this tensor as input features to prevent the policy from having to count fleets across temporal sequences (9th place *Boey*).
5. **The Discrete Intent Factorization**:
   - Let the neural policy predict high-level discrete tactical intent (`source_entity`, `target_entity`), and embed a deterministic analytical solver in the environment to compute the exact physical trajectories.
6. **Continuous Bounded Sizing**: For continuous allocations (e.g. ship count or bid percentage), use a **Truncated Discretized Logistic Mixture Head** over the valid dynamic range $[S_{\min}, S_{\max}]$.

---

### Phase 3: Neural Policy Architecture
*Deploy scalable, permutation-invariant multi-agent models.*
1. **Unified Multi-Entity Transformer**:
   - Independent MLP stems project distinct entity types into a shared embedding space ($D=256 - 768$).
   - Pre-norm residual Transformer trunk (12 to 38 blocks, 8 to 16 attention heads).
2. **Specialized Control & Workspace Tokens**:
   - **Player Summary Tokens**: Encodes macro economy, alive status, and player identity.
   - **Global Summary Token**: Encodes turn clock, step counter, and game phase.
   - **Board Scratch Tokens**: Learned embedding vectors that serve as a shared global workspace for inter-entity attention without supervisory loss.
3. **Single-Pass Multi-Player Evaluation**:
   - Compute actions and win probabilities for all active players in **one single forward evaluation**, cutting training and rollout compute by $2\times - 4\times$.
4. **Multi-Player Softmax Value Critic**:
   - Predict the joint categorical win probability distribution over active players ($\sum \hat{p}_i = 1$).
> **Detailed Guide:** See [`references/model_architectures.md`](references/model_architectures.md) for PyTorch implementations of entity transformers, relational edge-attention, 2D RoPE, and mixture heads.

---

### Phase 4: Distributed PPO & Training Stabilization
*Scale policy optimization without catastrophic forgetting, circular meta-drift, or stalling.*
1. **Curriculum Warm-Starting (Behavioral Cloning)**:
   - In sparse-reward environments, bootstrap cold-start policies with Behavioral Cloning (BC) on heuristic bots or top player replays before transitioning to pure PPO (2nd place *simjeg*, 5th place *TonyK*).
2. **Distributed Scaled PPO**:
   - Vectorize across $2,048 - 8,192$ parallel environments with $T=64$ rollouts.
   - Single-epoch PPO updates to prevent overfitting to recent trajectories.
   - **Muon + AdamW Optimizers**: Use Muon for 2D attention/linear matrices ($\ge 25M$ params) and AdamW for 1D embeddings/biases.
3. **Teacher Distillation Anchor**:
   - Anchor policy updates against a historical `checkpoint_last_best.pt` using Policy KL-divergence and Critic Cross-Entropy loss terms.
   - **The 70% Promotion Gate**: Replace the teacher checkpoint **only** when a candidate achieves $\ge 70\%$ win rate across 2,048 evaluation games.
4. **Anti-Stall Reward Shaping**:
   - Avoid undiscounted $\gamma=1.0$ without step penalties.
   - **Step-Conditioned Terminal Bonus**: Reward quick wins higher than delayed stalemates ($+1.0$ for win $<500$ steps, $+0.5$ for win $\ge 500$ steps) to eradicate defensive unit hoarding (2nd place *simjeg*).
5. **Game-Theoretic Dynamics (2-Player vs Multiplayer)**:
   - **2-Player**: Pure self-play with teacher distillation converges to robust minimax policies.
   - **Multiplayer (3+ Players)**: Pure self-play risks non-transitive Rock-Paper-Scissors cycles. Maintain a **Frozen Historical Opponent Pool** (e.g. 50% current self-play, 35% frozen historical checkpoints, 15% heuristic/exploiters) to maintain policy diversity (5th place *TonyK*).
> **Detailed Guide:** See [`references/rl_training_stability.md`](references/rl_training_stability.md) for PPO hyperparameter tables, BC warm-start pipelines, anti-stall reward curves, and league matchmaking algorithms.

---

### Phase 5: Submission Packaging, Quantization & Latency Guardrails
*Navigate strict Kaggle sandbox constraints ($\le 100\text{ MiB}$ file size, $1.0\text{s}$ turn latency on slow CPU).*
1. **Checkpoint Compression**:
   - Quantize 2D linear weights using **4-bit NormalFloat (NF4/NF5)** with group size 128 and Least-Squares (LSQ) scale refinement.
   - Compresses 200M parameter models ($800\text{ MiB}$) down to **$90.7\text{ MiB}$**, fitting comfortably inside the 100MiB archive cap while retaining ~40% win rate against unquantized fp32.
2. **Streaming Dequantizer**:
   - Load and dequantize tensors one-by-one at container startup directly into model buffers, avoiding full state-dict memory spikes.
3. **Dynamic int8 CPU Inference**:
   - Convert non-output linear layers to signed int8 via `torch.ao.quantization.quantize_dynamic`, cutting inference latency from $2,500\text{ ms}$ to $<450\text{ ms}$.
4. **Test-Time Rollout Lookahead Search**:
   - Combine neural policy priors with a shallow $1\text{--}2$ turn lookahead rollout using the compiled simulator core to verify tactical survival and prune obvious blunders (6th place *flg*).
5. **Hierarchical Decoupling**:
   - Separate high-level strategic target selection (evaluated every $N$ turns by neural policy) from low-level local tactical execution handled turn-by-turn by an analytical solver (10th place *Liu*).
6. **Circuit-Breaker Fallback Cascade**:
   - Monitor `observation.remainingOverageTime`.
   - If overage time drops below $1.0\text{ second}$, immediately trigger an automatic handoff to a packaged, lightweight $5\text{M}$ parameter model ($<30\text{ ms}$ inference).
> **Detailed Guide:** See [`references/submission_quantization_serving.md`](references/submission_quantization_serving.md) for complete NF4-LSQ encoders, test-time rollout searchers, streaming loaders, and dual-model fallback wrappers.

---

## Quick Reference & Hyperparameter Cheat Sheet

| Problem Domain | Recommended Solution | Key Parameters / Code Reference |
| :--- | :--- | :--- |
| **Simulator Throughput (CPU)** | Rust (PyO3 + Rayon) / C++ | `par_iter_mut()`, preallocated pinned CPU buffers (`pin_memory=True`) |
| **Simulator Throughput (GPU)** | Pure JAX JIT Environment | `jax.vmap`, `jax.lax.scan`, zero host-to-device memory copying ($500\text{k+ sps}$) |
| **Variable Entity Boards** | Entity-Centric Transformer | Shared 768-d embedding, 17 special tokens (Player, Global, Scratch) |
| **Spatial Coordinates** | Continuous 2D RoPE | Orthogonal rotation frequencies along $(x, y)$ axes |
| **Pairwise Physics Relations** | Relational Edge-Attention | Attention bias $Q K^T / \sqrt{d} + W_e E$ injecting distances/flight times |
| **Future Planning Signals** | Planet Future Forecasting | Analytical future-arrival garrison tensor projected over $K$ horizons |
| **Action Generation** | Discrete Intent + Solver | $Q \cdot K^T / \sqrt{d}$ target selection + analytical physical intercept |
| **Continuous Fleet Sizing** | Truncated Logistic Mixture | 8 components over $[S_{\min}, S_{\max}]$, normalized sigmoid means |
| **Sparse-Reward Cold Start** | Behavioral Cloning Bootstrap | Supervised pretraining on heuristics/replays before RL fine-tuning |
| **Optimization Stability** | Last-Best Teacher Distillation | $\alpha_{\text{KL}} D_{\text{KL}}(\pi_{\text{teacher}} \|\, \pi) + \alpha_{\text{CE}} \mathcal{L}_{\text{CE}}$, promote at $\ge 70\%$ win rate |
| **Anti-Stall Terminal Reward** | Step-Conditioned Bonus | $+1.0$ if steps $< 500$, $+0.5$ if steps $\ge 500$ |
| **Multi-Player Dynamics** | Frozen Opponent League | 50% self-play, 35% historic checkpoints, 15% exploiters |
| **100MiB Submission Cap** | Grouped NF4-LSQ Codebook | Group size 128, fp16 scales, least-squares scale fitting ($90.7\text{ MiB}$) |
| **Tactical Blunder Pruning** | Test-Time Lookahead Search | 1-2 turn shallow forward simulation evaluating survivability |
| **CPU Latency Budget** | Dynamic int8 + Fallback | `quantize_dynamic(model, {nn.Linear}, qint8)` + 5M model fallback if bank $<1.0\text{s}$ |

---

## Common Mistakes & Anti-Patterns

| Anti-Pattern | Why It Fails | Battle-Tested Fix |
| :--- | :--- | :--- |
| **Cold-Start RL Exploration Trap** | Pure random exploration in complex multi-agent environments rarely encounters winning terminal states. | Warm-start policy with **Behavioral Cloning (BC)** on strong heuristic bots or top match replays before RL. |
| **Premature Action Masking** | Masking illegal/suicidal actions early prevents the neural net from internalizing physical game boundaries. | Train **unmasked** during exploration so the network learns physics; introduce masks only for final fine-tuning and test serving. |
| **Undiscounted Stalling ($\gamma=1.0$)** | With no temporal penalty, winning bots hoard units and stall for hundreds of turns rather than finishing matches. | Set $\gamma = 0.99$, add step-conditioned terminal rewards ($+1.0$ if $<500$ steps, $+0.5$ if $\ge 500$), or deduct small per-step penalties. |
| **Pure Self-Play in Multiplayer ($N \ge 3$)** | Multi-agent environments have non-transitive dynamics; pure self-play leads to circular overfitting (Rock-Paper-Scissors). | Maintain a **Frozen Historical Opponent Pool** to evaluate against past generations and prevent meta-drift. |
| **Single-Player Forward Evaluation** | Running $N$ separate forward passes per state wastes $2\times - 4\times$ rollout compute and GPU VRAM. | Unified Single-Pass Transformer: Concatenate all players' tokens into one sequence and predict all policies simultaneously. |
| **Uniform INT4 Quantization** | Naive uniform quantization destroys attention weight distributions, causing catastrophic policy collapse. | Use **NormalFloat 4 (NF4)** codebook quantization with group size 128 and Least-Squares (LSQ) scale refinement. |
| **Full State-Dict Deserialization** | Dequantizing the entire 800MiB model at once on Kaggle CPU causes out-of-memory container crashes. | Use **streaming dequantization**, reconstructing one parameter tensor at a time directly into preallocated model storage. |
| **Lookahead-Blind Tactical Blunders** | Large transformers can occasionally miss immediate 1-ply tactical traps or suicide flights. | Add a **shallow test-time rollout search** (1–2 turns) using the fast simulation engine to filter suicidal candidate moves. |
| **Unprotected CPU Latency** | Relying solely on a heavy model risks disqualification if Kaggle assigns a throttled CPU core. | Implement a **circuit-breaker fallback cascade**: auto-switch to a sub-5M model when bank overage drops below $1.0\text{ second}$. |

---

## Complete Competition Case Studies

- **Orbit Wars (1st–10th Place Post-Mortem)**: [Deep Dive Post-Mortem](../../../Handbook/reinforcement-learning/orbit-wars.md) — Comprehensive comparative autopsy covering the 200M Transformer, Rust engine acceleration, pure JAX JIT pipelines, NF4-LSQ quantization, Relational Edge-Attention, 2D RoPE, anti-stall reward curves, and test-time lookahead search.
