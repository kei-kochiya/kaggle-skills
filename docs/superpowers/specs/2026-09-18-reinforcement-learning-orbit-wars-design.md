# Design Spec: Reinforcement Learning & Simulation Track (Orbit Wars)

- **Date**: 2026-09-18
- **Status**: Approved
- **Scope**: Architectural

---

## 1. Overview & Motivation

This specification formalizes the addition of the **Reinforcement Learning (RL) & Simulation Track** to the repository. The track captures competitive machine learning methodologies from game simulations (multi-agent strategy, continuous orbital mechanics, turn-based/real-time decision making) distilled from the 1st Place Orbit Wars solution (a 200-million parameter Transformer trained for 15 billion steps).

Following the repository's dual-layer philosophy:
1. **Human Knowledge Base (`Handbook/`)**: A comprehensive engineering post-mortem of the Orbit Wars competition and 1st place solution.
2. **AI Agent Skills (`.agent/skills/`)**: A modular, discoverable Antigravity skill (`kaggle-rl-simulation`) complete with concrete reference guides enabling autonomous agents to execute end-to-end competitive RL campaigns.

---

## 2. Directory Architecture & Taxonomy

```text
kaggle-skills/
├── README.md                                         # Update: RL track catalog & skill index
├── Handbook/
│   ├── README.md                                     # Update: Add RL competition index table & taxonomy
│   └── reinforcement-learning/                       # New Track Directory
│       └── orbit-wars.md                             # 1st Place Orbit Wars Deep Dive Post-Mortem
└── .agent/skills/
    └── kaggle-rl-simulation/                         # New Agent Skill Directory
        ├── SKILL.md                                  # Core runbook, decision trees & operational flow
        └── references/
            ├── simulator_acceleration.md             # High-throughput Rust/JAX environments & parity
            ├── model_architectures.md                # Multi-entity transformer, scratch tokens & heads
            ├── rl_training_stability.md              # PPO, teacher distillation, self-play vs league play
            └── submission_quantization_serving.md    # NF4/NF5 quantization, int8 CPU serving & fallbacks
```

---

## 3. Component Details

### A. Handbook Deep Dive: `Handbook/reinforcement-learning/orbit-wars.md`

A structured technical autopsy following the Handbook schema:
1. **Executive Summary & Competition DNA**:
   - 2-player and 4-player orbital space conquest in 2D continuous space.
   - Core breakthrough: Validation of Sutton's Bitter Lesson — scaling a 200M parameter Transformer for 15 billion self-play steps over hand-engineered heuristic agents.
   - Summary of key metrics, hardware scale (32× B200 GPUs, 2400 GPU-hours), and deployment innovations.
2. **Environment Dynamics & Action Formulation**:
   - Celestial orbital mechanics, gravitational pull, sun collision hazards, moving comets, and fleet combat.
   - Action space progression: Why continuous launch angle prediction failed to reach competitive play and how transitioning to `(source_planet, target_planet)` discrete-target pairs with an embedded analytical angle solver transformed policy learnability.
3. **The Simulation Engine (Rust Rewrite & Parity Verification)**:
   - High-throughput Rust environment using PyO3 bindings and Rayon multi-threading achieving 120,000+ steps/sec.
   - Zero-copy buffer recycling and preallocated pinned memory.
   - Collision optimization: conservative AABB broad-phase filtering before exact contact geometry.
   - Replay parity test framework: automated regression against Kaggle tournament JSON episode replays.
4. **Policy & Value Model Architecture**:
   - Multi-entity projection: Independent MLPs projecting planets, comets, and fleets into a shared 768-d embedding space.
   - 17 Specialized Control Tokens: 4 Player Summaries, 1 Global Board state, 4 Actor Plan tokens, 4 Critic Value tokens, and 4 learned Scratch tokens (global workspace attention).
   - Transformer Trunk: 38-block pre-norm residual Transformer (16 heads, 1536 MLP hidden width, GELU activations).
   - Single-Pass Multi-Player Inference: Predicting actions and win probabilities for all players in a single forward pass, saving 2×–4× compute.
   - Actor & Critic Heads: Bernoulli launch probability, scaled dot-product target selection ($\frac{Q \cdot K^T}{\sqrt{d}}$), truncated discretized logistic mixture fleet sizing (8 components), and multi-player softmax win-probability critic.
5. **RL Training Pipeline & Distributed Scaling**:
   - Scaled PPO recipe: GAE-λ advantage estimation, clipped surrogate objective, entropy bonus, normalized advantages.
   - Teacher distillation: Policy KL and cross-entropy value loss against the historical "last-best" checkpoint.
   - Checkpoint promotion: Replacing the teacher only when the candidate achieves >70% head-to-head win rate across mixed 2p/4p matches.
   - Training scale: 4 nodes × 8× B200 GPUs, 8,192 parallel environments, 64-step rollouts (~6.3M steps/GPU-hour).
6. **Submission Engineering Under Tight Constraints**:
   - 100MiB File Cap: 4-bit NormalFloat (NF4/NF5) group quantization with group size 128, fp16 group scales, and least-squares (LSQ) scale refinement (90.7MiB package maintaining ~40% win rate against unquantized fp32).
   - CPU Turn Latency Budget (1s turn + 60s bank): Runtime dynamic int8 CPU quantization of `nn.Linear` layers.
   - Observation entity compaction & minor fleet filtering.
   - Dynamic Fallback: Automatic switch to a 5M parameter model if remaining overage time drops below 1s (100% win-conversion in decided games).
7. **Post-Mortem Lessons & Anti-Patterns**:
   - The Stalling Problem ($\gamma = 1.0$): Why lack of discount factor led to lead-stalling during training and the need for early truncation/surrender mechanisms.
   - Inverted Matchmaking & Non-Transitive Dynamics: Shifting from 2-player to 4-player dominance and why league play with diverse checkpoint pools is essential to eliminate strategic cycles.
   - The Action Masking Paradox: Why masking suicidal launches degraded early policy training by preventing the model from internally learning orbital physics.
   - Fully Agentic Development: Using Codex and LLM agents with documentation-driven human review.

---

### B. Reusable Agent Skill: `.agent/skills/kaggle-rl-simulation/`

#### 1. Main Entrypoint: `SKILL.md`
- **Frontmatter**:
  - `name`: `kaggle-rl-simulation`
  - `description`: *"Use when developing, training, optimizing, or deploying reinforcement learning agents and multi-agent simulation policies for competitive games or simulation challenges on Kaggle. Covers high-throughput environment rewrites, transformer policy architectures, stabilized self-play PPO, and strict CPU/memory submission deployment."*
- **Operational Workflow**:
  - Phase 1: Environment Diagnostics & Throughput Profiling.
  - Phase 2: Observation & Action Space Structuring.
  - Phase 3: Neural Policy Architecture & Multi-Agent Representation.
  - Phase 4: Distributed PPO & Training Stabilization.
  - Phase 5: Submission Packaging, Quantization & Latency Guardrails.
- **Decision Trees**:
  - Continuous Angle vs Discrete Target Selection.
  - Self-Play vs League Play selection based on player count.
  - Checkpoint Quantization selection (fp32 vs fp16 vs NF5/NF4 vs int8).

#### 2. Modular Reference Guides (`references/`):
- `simulator_acceleration.md`:
  - Rust + PyO3 / Rayon architecture patterns for 100,000+ steps/sec vector environments.
  - Zero-copy tensor buffers, pinned memory, and contiguous layout alignment.
  - Spatial indexing and broad-phase AABB collision filtering.
  - Automated replay parity test harnesses validating bit-for-bit transitions against Kaggle replay JSONs.
- `model_architectures.md`:
  - Entity-centric tokenization for variable-size boards.
  - Global scratch tokens as an attention workspace.
  - Joint multi-player forward prediction (computing all agent policies in 1 pass).
  - Truncated discretized logistic mixture heads for continuous/bounded actions.
- `rl_training_stability.md`:
  - Production PPO hyperparameter configurations (AdamW/Muon, GAE-λ, clipping).
  - Teacher distillation against historical checkpoints (>70% win-rate promotion).
  - Multi-agent game theory: 2-player zero-sum vs N-player non-transitive dynamics.
  - Reward formulation, the $\gamma=1.0$ dilemma, and truncation incentives.
- `submission_quantization_serving.md`:
  - Grouped NormalFloat (NF4/NF5) quantization with least-squares scale fitting.
  - Dynamic int8 runtime CPU quantization with PyTorch.
  - Observation compaction and fleet entity pruning.
  - Two-tier fallback routing cascades for overage time protection.

---

### C. Repository Index Updates

1. `Handbook/README.md`:
   - Add new section: `### 🚀 Reinforcement Learning & Simulation Competitions` with Orbit Wars index row.
   - Update taxonomy to include `reinforcement-learning/`.
   - Add `kaggle-rl-simulation` to the Agent Skills catalog.
2. `README.md`:
   - Add Orbit Wars to the Competition Index.
   - Add `kaggle-rl-simulation` and its reference cookbooks to the Agent Skills Architecture table.
   - Update repository tree structure diagram.

---

## 4. Verification Plan

1. **Schema & SDO Compliance**:
   - Verify `SKILL.md` frontmatter adheres to the Agentskills specification (max 1024 chars, starts with "Use when...", third person, no workflow summary in description).
   - Ensure all markdown links use valid relative paths.
2. **Technical Correctness**:
   - Verify all architectural formulas ($\frac{Q \cdot K^T}{\sqrt{d}}$, logistic mixture components, PPO GAE, NF4 LSQ formulas) are mathematically accurate.
   - Validate code snippets against the cloned `IsaiahPressman/kaggle-orbit-wars` repository implementations.
3. **Repository Consistency**:
   - Check that all tables, links, and indexes in `Handbook/README.md` and `README.md` resolve cleanly.
