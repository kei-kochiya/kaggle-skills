---
name: kaggle-rl-simulation
description: Use for Kaggle game and simulation competition kickoff, heuristic/search/neural agent development, replay imitation, RL training, simulator validation and submission evaluation.
---

# Simulation Competition Workflow

Choose an approach using the game contract, available demonstrations, measured outcomes, hardware, and deadline. A heuristic, bounded search planner, behavioral cloning (BC) policy, reinforcement learning (RL) agent, or hybrid can be optimal depending on episode structure. Preserve a working baseline while testing improvements.

## Routing

- For a new competition or a request to transfer past lessons, read the [simulation starter and kickoff prompt](../../../Handbook/reinforcement-learning/simulation-competition-starter.md) to establish first deliverables and identify which assumptions must be re-derived.
- Load detailed references below only when the current stage needs them. All dimensions, batch sizes, horizons, and serving limits must be derived from the active competition's contract and local profiling.

## 1. Establish a Runnable Foundation

1. **Game Contract**: Inspect official rules, source/configuration, scoring, and submission constraints. Record environment versions and source identities. Verify player count, visible vs. private observations, randomness/RNG semantics, action execution order, illegal-action handling, and terminal/round-reset behavior.
2. **Baseline & Replay**: Run an official starter or simple heuristic through a complete game. Save its source, replay, terminal result, and per-turn latency. A read-only research request does not imply starting a training campaign.
3. **Repeatable Evaluator**: Build an evaluation harness with distinct opponent families and separate development vs. final held-out scenario panels. Use paired-seed seat swaps for two-player games and balanced seat rotations for multiplayer games.
4. **Profile Bottlenecks**: Measure simulation, observation encoding, model inference, host-to-device transfer, and gradient update costs separately. Check host RAM, GPU VRAM, and deployment CPU limits before scaling.
5. **Early Packaging Check**: Package a simple baseline locally early enough to verify entry points, dependencies, archive size, and startup/turn latency. Local success does not guarantee identical CPU throttling in the official runtime.

## 2. Choose the Next Experiment

| Measured Evidence | Candidate Direction |
| :--- | :--- |
| Strong rules or public replays already exist | Improve the heuristic directly or train a small BC policy on valid demonstrations |
| Micro-logistics or pathing causes deadlocks/collisions | Hierarchical Macro-BC (periodic high-level intent) + deterministic micro compiler |
| Opening has high branching entropy, endgame is mathematical | Split-horizon hybrid: neural BC opening + combinatorial/search endgame planner |
| Short-horizon forward simulation is fast and deterministic | Bounded test-time lookahead search; validate simulator parity and latency |
| Good teachers exist, but rewards are sparse & actions complex | Executed-label BC → full-game validation → bounded RL fine-tuning if helpful |
| Simple actions and informative rewards | Direct RL probe with a compact model |
| Compact policy saturates on spatial/relational reasoning | Compare grid CNN, pooled MLP, graph/edge-attention, or entity Transformer |
| Simulation or transfer dominates wall time | Optimize the measured bottleneck (Rust/C++/JAX); enforce replay parity gates |

## 3. Demonstration & Actor-Input Invariants

- **Explicit Provenance & Alignment**: Record teacher identity, episode ID, player seat, and version. Derive observation-to-action alignment strictly from the active environment's replay schema.
- **Episode-Level Splits**: Keep entire episodes (all seats and turns) inside one train or validation split. Deduplicate repeated episode/seat rows.
- **Executed-State Labels Over Raw Orders**: In environments where the engine silently drops or repairs invalid/conflicting commands, train BC on the **executed state transitions** actually applied by the engine rather than raw submitted orders. Verify `encode → decode → engine` parity on every retained replay.
- **Conditioning Over Hard Filtering**: Instead of discarding good-but-not-top trajectories, condition the policy on `(Teacher/Style ID, Recency, Outcome)` with conditioning dropout during training, and prompt at inference with `(Top Style, Latest Recency, Win)`.
- **Strict Visibility & Relative Perspective**: Feed the actor only observations available at inference. In batched multi-player forward passes, keep private information isolated per seat. Encode **relative turn-order / active-perspective indicators** rather than raw physical seat indices whenever turn order can rotate across rounds.
- **Sequential Resource Reservation**: When multiple units/slots compete for shared resources within one turn, reserve resources in the engine's exact execution order (or use a cross-decision corrector).

## 4. Learning, RL Stability & Promotion

- **Full-Game BC Evaluation**: Never rely solely on held-out cross-entropy loss (often dominated by frequent `PASS`/idle actions). Evaluate BC policies in complete unseen games and track rare high-leverage decisions separately.
- **Critic Pre-Warm & Horizon Alignment**: When transitioning from BC to actor-critic RL, test freezing the actor while pre-warming the value head first. Align reward, discount $\gamma$, and GAE $\lambda$ with the true horizon and termination rules; distinguish true episode end from rollout truncation.
- **Exact PPO Support Consistency**: Store sampled action IDs, old log-probabilities, conditional masks, temperatures, and forced-action flags during rollouts. Recompute log-probabilities under identical masks before updating (verify ratio $\approx 1.0$ on step 0). Exclude post-sampling deterministic overrides from actor log-prob gradients while preserving their state transitions for the critic.
- **External & Paired-Seed Promotion**: Never trust self-play win rates alone—even two-player zero-sum games can cycle, and homogeneous self-play misses aggressive out-of-distribution heuristics. Gate promotions via paired-seed seat-swapped matches against both the incumbent champion and an independent opponent panel.

## 5. Evaluate & Retain Evidence

- Track the competition's primary outcome metric alongside operational diagnostics: illegal/dropped action rate, resource waste, terminal state efficiency, latency percentiles, and overage usage.
- Require exact discrete-rule parity (including RNG state and tie-breaking/container ordering) and explicit float tolerances before trusting any accelerated custom simulator.
- Benchmark the exact packaged submission under CPU/RAM/time caps. Apply quantization, observation compaction, pure-NumPy serving, or latency circuit-breakers only when measured limits demand them, and re-verify win rate afterward.

## Detailed References & Case Studies

- [BC, Replay Engineering & Promotion Gating](references/bc_replay_and_gating.md): Executed-state replay verification, style/recency/outcome conditioning, one-pass draft+corrector, checkpoint soups, sequential conflict masking, hard-example replay, pure-NumPy serving, and paired-seed gating.
- [Simulator Acceleration & Parity](references/simulator_acceleration.md): Rust (PyO3/Rayon), C++, and pure-JAX vectorized environments, zero-copy pinned transfers, spatial broad-phase filtering, and automated replay parity harnesses.
- [Neural Policy & Value Architectures](references/model_architectures.md): Entity Transformers, global scratch tokens, single-pass multi-player inference, pointer-attention & discretized logistic-mixture heads, relational edge-attention, 2D RoPE, and deterministic forward-horizon projections.
- [RL Training Stability & League Play](references/rl_training_stability.md): Stabilized PPO, Polyak/KL teacher anchoring, anti-stall reward shaping, frozen historical pools, and adversarial league exploiters.
- [Quantization, CPU Serving & Fallback Cascades](references/submission_quantization_serving.md): Grouped NF4-LSQ compression, streaming dequantization, dynamic int8 CPU serving, entity compaction with safety invariants, overage circuit-breakers, lookahead search, and runtime rejection/round-reset guards.
- Case Studies: [`Kaggriculture`](../../../Handbook/reinforcement-learning/kaggriculture.md) · [`Orbit Wars`](../../../Handbook/reinforcement-learning/orbit-wars.md) · [`Maze Crawler`](../../../Handbook/reinforcement-learning/maze-crawler.md) · [`Pokémon TCG`](../../../Handbook/reinforcement-learning/pokemon-tcg-playground.md)
