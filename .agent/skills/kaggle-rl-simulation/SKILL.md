---
name: kaggle-rl-simulation
description: Use for Kaggle game and simulation competition kickoff, heuristic/search/neural agent development, replay imitation, RL training, simulator validation and submission evaluation.
---

# Simulation competition workflow

Choose an approach using the game, available teachers, measured outcomes, hardware and deadline. A heuristic, search planner, learned policy or hybrid can be appropriate. Preserve a working baseline while testing improvements.

## Routing

- For a new competition or a request to transfer past lessons, read the [simulation starter and kickoff prompt](../../../Handbook/reinforcement-learning/simulation-competition-starter.md). It explains the first deliverables and which assumptions must be re-derived.
- For advanced Kaggriculture reproduction or tape-to-learning work, read [references/kaggriculture.md](references/kaggriculture.md). Its source pins and controller semantics take precedence over unrelated examples.
- Read detailed references below only when the current stage needs them. Their numerical settings and scaling examples are game-specific.

## Establish a runnable foundation

1. Inspect official rules, source/configuration, scoring and submission constraints. Record environment versions and source identities. Determine player count, allowed observations, randomness, action execution order, illegal-action handling and terminal behavior.
2. Run an official starter or simple heuristic through a complete game. Save its source, replay, terminal result and latency. A read-only research request does not imply starting a training campaign.
3. Build a repeatable evaluator with active, distinct opponent families when applicable. Use development and final held-out scenario panels. Swap seats for two-player games and use appropriate assignments for multiplayer or single-player tasks.
4. Measure simulation, encoding, inference, transfers and updates separately. Check host RAM, GPU memory and the deployment CPU budget. Project wall time from measured throughput before scaling.
5. Package a simple baseline locally early enough to check entry points, dependencies and target limits. Local success does not prove acceptance or identical time accounting in the official runtime.

Choose useful work within the user's scope and compute budget. Do not replace a research task with unrequested long training, or assume a repository contains its authors' weights and datasets.

## Choose the next experiment

| Evidence | Candidate direction |
| --- | --- |
| Strong rules/tapes already work | Improve the heuristic or use valid demonstrations for BC |
| Short-horizon planning is affordable | Bounded search; validate simulator fidelity and serving cost |
| Good teachers, sparse reward, complex actions | Small BC policy, complete-game validation, then an RL probe if useful |
| Simple actions and informative reward | Consider direct RL with a small model |
| Small policy is limited by representation | Compare compact grid, pooled/entity or graph models according to the game |
| Simulation or transfer dominates cost | Optimize the measured bottleneck; custom engines need parity checking |

Entity Transformers, compiled Rust/JAX simulators, PPO, opponent leagues and quantization are options. Model size, optimizer, rollout length, discount and serving limits are not universal defaults. An original tape's schedule may break when learned choices change resources; use compatible plans or a reactive executor.

## Demonstration and actor-input invariants

- Identify the teacher, episode, player position and source version explicitly. Derive observation/action alignment from this game's replay format; Kaggriculture's offset is not a generic rule.
- Keep entire episodes/scenarios within one training or validation split. Deduplicate repeated game/seat rows and account for shared teacher lineage.
- Preserve proposed, repaired and executed actions when they differ. Define labels using the actual resolver and action vocabulary; filter impossible/unexecuted commands or represent their semantics explicitly.
- Train the actor from the observation it can receive at inference. Batched multi-player evaluation must keep each actor's private information isolated. Forecast features may use visible evidence and known deterministic dynamics, not hidden future randomness.
- Verify that learned choices affect emitted actions and that conditional decisions reserve shared resources in execution order. Legality alone does not establish strategic usefulness.

## Learning and improvement

Start with a feasible model and data subset. Inspect host expansion and batch memory, not just compressed dataset sizes. BC success requires complete held-out games and useful rare decisions as well as supervised loss. Student mistakes can move play outside the teacher's recorded states.

For actor-critic RL, decide whether a separate value-fitting stage is useful before full updates. Frozen-actor critic fitting is Kaggriculture's adopted procedure, not mandatory for every algorithm. Align rewards, discounting and returns with the actual score/horizon; preserve terminal outcomes and handle collection cutoffs correctly.

For PPO, retain sampled action IDs, old log probabilities, values, conditional masks, temperatures and any forced-factor flags. Before an update, recomputation should agree with collection and the ratio should be approximately one. Use the ratio/clipped objective; plain `-log_probability * advantage` is not PPO-Clip. Account explicitly for deterministic controllers instead of treating forced replacements as sampled choices.

Distinguish BC teachers from neural probability-distribution references used for KL regularization. An initial random model is not a competent teacher. Use a fixed reference or promotion scheme only when justified by the experiment; there is no universal promotion win rate.

Diagnose failures by opponent/scenario/phase. Improve a teacher or controller for those cases, collect targeted demonstrations when useful, and compare the resulting learner against the preserved evaluation panel. Teacher refinement, search, more data and architecture growth are separate experiments. Two-player games can also have non-transitive strategies; self-play alone does not establish broad strength.

## Evaluate and retain evidence

Track the competition's actual outcome metric plus relevant diagnostics: action failures, waste, terminal inventory, survival, latency or other game-specific indicators. Preserve seed/scenario clusters and player assignments when estimating uncertainty. One mirror match, own earnings against PASS, or a public rating snapshot is insufficient evidence of generalization.

Verify state, reward, terminal and observation-visibility parity before trusting a custom simulator. Require exact discrete-rule agreement and explicit floating-point tolerances; include boundary cases appropriate to the game.

Benchmark the exact packaged controller under target-like conditions. Apply compression, search budgets or fallback behavior when measured limits call for them, and retest outcome quality after those changes. Keep the strongest verified agent and record source/configuration, data, weights, commands, measured costs and results separately.

## Detailed references and case studies

- [Simulator acceleration](references/simulator_acceleration.md): Rust/JAX and parity patterns; use after profiling identifies a need.
- [Model architectures](references/model_architectures.md): entity/relational representations and action heads; verify visibility and serving cost before adopting.
- [Training stability](references/rl_training_stability.md): example PPO, teacher and league recipes; choose reward/horizon settings from the game.
- [Quantization and serving](references/submission_quantization_serving.md): example compression/search/fallback implementations; derive current submission limits first.
- [Kaggriculture](../../../Handbook/reinforcement-learning/kaggriculture.md): BC/PPO, targeted teachers, resource masks and distinct Final A/B controllers. The analyzed October 2, 2026 first-place claim was provisional.
- [Maze Crawler](../../../Handbook/reinforcement-learning/maze-crawler.md): heuristic targeting and self-play blind spots.
- [Orbit Wars](../../../Handbook/reinforcement-learning/orbit-wars.md): historical simulation scaling, neural architectures and serving examples.
