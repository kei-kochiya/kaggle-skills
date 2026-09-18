# Design Spec: Orbit Wars Top-10 Solution Distillation & Reusable RL Skills

- **Date**: 2026-09-18
- **Status**: Approved
- **Scope**: Architectural

---

## 1. Overview & Objectives

Following the distillation of the 1st Place Orbit Wars solution, this specification synthesizes key insights, architectural patterns, and algorithmic breakthroughs from **all prize-winning top-10 solutions (2nd through 10th place)** into the repository:
1. **Handbook Masterclass Expansion (`Handbook/reinforcement-learning/orbit-wars.md`)**:
   - Integrate an authoritative Top-10 comparative scoreboard and competitive taxonomy.
   - Detail solutions across four distinct paradigms:
     - **Paradigm A: Extreme Model Scaling & Self-Play** (1st Place `IsaiahPressman` 200M Transformer).
     - **Paradigm B: Behavioral Cloning Warm-Start to RL Curriculums** (2nd Place `simjeg`, 5th Place `TonyK`).
     - **Paradigm C: End-to-End JAX JIT-Compiled Simulation Pipelines** (9th Place `Boey`, 8th Place `Billy Bradley`, 10th Place `Xiangyu Liu`).
     - **Paradigm D: Domain Inductive Biases & Hybrid Lookahead Search** (6th Place `flg`, 8th Place `Billy Bradley`, 9th Place `Boey`).
2. **AI Agent Skills Enrichment (`.agent/skills/kaggle-rl-simulation/`)**:
   - Codify actionable, production-grade recipes across all 4 reference cookbooks and update `SKILL.md`.

---

## 2. Component Enhancements

### A. Handbook Deep Dive: `Handbook/reinforcement-learning/orbit-wars.md`

1. **Top-10 Comparative Scoreboard**:
   - Ranks 1 to 10 with competitor handles, framework/engine (Rust, JAX, C++, PyTorch), model parameters, training scale/compute budget, and core breakthroughs.
2. **The 4 Architectural Paradigms of Competitive RL**:
   - Comparative analysis contrasting Bitter Lesson brute scaling vs. Frugal JAX end-to-end efficiency vs. Imitation curricula vs. Lookahead search.
3. **Deep Solution Breakdowns (2nd to 10th Place)**:
   - **2nd Place (`simjeg`)**: Heuristic $\to$ Behavioral Cloning $\to$ RL fine-tuning $\to$ RL from scratch. Step-conditioned anti-stall reward shaping ($+1.0$ before 500 steps, $+0.5$ after).
   - **3rd Place (`Felix M. Neumann`, "Ab in den Orbit")**: Transformer policy with JAX PPO self-play.
   - **5th Place (`TonyK`)**: Asynchronous distributed IMPALA, BC replay initialization, delayed moving teacher (Polyak anchor), frozen historical opponent pool.
   - **6th Place (`flg`)**: Custom Edge-Attention Transformer (2.5M params), injecting pairwise planet travel times and transit curves into attention weights, 2-step rollout lookahead search during 2p inference, custom C++ engine.
   - **7th Place (`Audun Ljone Henriksen & Eirik Torp`)**: Structured component isolation, tactical heuristics vs RL, local Elo validation ladders.
   - **8th Place (`Billy Bradley`, "Ender for <$200")**: Frugal RL on a budget, autoregressive micro-step action decoding, 2D Rotary Position Embeddings (2D RoPE) for continuous orbital coordinates, ETA reachability filtering.
   - **9th Place (`Boey`)**: 100% pure JAX JIT pipeline (`jax.jit`, `jax.vmap`) eliminating CPU-GPU transfer overhead completely, "Planet Future" forward-simulated trajectory projections.
   - **10th Place (`Xiangyu Liu`)**: Decoupled systems hierarchy (Precomputed `MapCache` + `StrategicEnv` + JAX PPO + C++ solver), 44 compacted planet entities with folded fleet dynamics.

---

### B. Agent Skill Reference Cookbooks (`.agent/skills/kaggle-rl-simulation/references/`)

1. **`references/simulator_acceleration.md`**:
   - Add **End-to-End JAX JIT-Compiled Simulation Pipelines** (`jax.jit`, `jax.vmap`).
   - Eliminate Python-CPU-GPU transfer overhead by keeping environment stepping, observation packing, and PPO loss optimization inside a single fused GPU kernel.
   - Compare Rust (PyO3/Rayon) vs. JAX JIT architectures (throughput, memory, ease of experimentation).

2. **`references/model_architectures.md`**:
   - Add **Relational Edge-Attention**: Injecting pairwise planet-to-planet relational edges (transit time, gravitational hazard, ownership change horizons) into attention logit matrices:
     $$\text{Attention}(Q, K, E) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + \mathbf{W}_E E\right)$$
   - Add **2D Rotary Position Embeddings (2D RoPE)** for 2D continuous coordinates $(r, \theta)$ or $(x, y)$, preserving spatial translation and rotational relativity without rigid Cartesian grids.
   - Add **"Planet Future" Forward Trajectory Projections**: Precomputing and encoding multi-step lookahead garrison forecasts into observation tensors.

3. **`references/rl_training_stability.md`**:
   - Add **Behavioral Cloning Warm-Start & Imitation-to-RL Curricula**: Pretraining policies on high-Elo tournament replays before switching to PPO/IMPALA.
   - Add **Step-Conditioned Anti-Stall Reward Shaping**: Resolving the $\gamma=1.0$ stalling dilemma via temporal milestone decay (e.g. $+1.0$ for fast wins, $+0.5$ for late wins).
   - Add **Frozen Historical Opponent League Pools**: Managing an evolving pool of frozen historical checkpoints to eliminate non-transitive strategic cycles.

4. **`references/submission_quantization_serving.md`**:
   - Add **Test-Time Rollout Lookahead Search**: Using lightweight neural policies to evaluate 1-2 step simulated rollouts during tournament inference.
   - Add **Strategic / Physics Hierarchical Decoupling**: Separating strategic macro-selection (`StrategicEnv` / neural net) from deterministic continuous trajectory solvers (`MapCache` / C++ solver).

5. **`SKILL.md`**:
   - Update operational phases, decision flowcharts, quick-reference table, and common mistakes with the new techniques.

---

## 3. Verification & Validation Plan

- Verify all technical descriptions match the public writeups and implementations.
- Verify all relative markdown links resolve cleanly.
- Maintain clean git status and commit history.
