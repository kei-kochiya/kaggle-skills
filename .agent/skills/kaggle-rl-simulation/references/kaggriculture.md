# Kaggriculture: BC → critic fitting → self-play PPO

Use for the advanced `kaggriculture` simulation, its tape agents, replay imitation or the M & M & P & Q release. Do not transfer this schema to `kaggriculture_beginner`. Read this reference before applying the generic Orbit Wars scaling, discounting or quantization recipes to Kaggriculture.

## Establish the actual task

- Determine whether the user wants research, a tape repair, a compact learned controller or reproduction of the release. Implement the requested scope; a research request does not imply a long GPU training campaign.
- Treat the October 2, 2026 writeup as **provisional current first place**, not a confirmed final result. Recheck current status for time-sensitive claims.
- Inspect actual notebook source and linked training code. Tags and titles do not distinguish tape routing, heuristics, BC and PPO reliably. Identify exactly which learned choices influence execution.
- For a tape user, keep the existing archive as a baseline/teacher. Convert actual rollout states and teacher actions into demonstrations. A learned market intervention may break the tape's later schedule; use compatible plan variants or a reactive worker executor.

Primary sources: [discussion](https://www.kaggle.com/competitions/kaggriculture/discussion/745073), [pinned source release](https://github.com/msdsm/kaggriculture-solution/tree/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d), [handbook comparison](../../../../Handbook/reinforcement-learning/kaggriculture.md).

For an explanation request, use handbook sections 1–6: they define the learning terms, show a seed-conflict example and compare the final controllers. For implementation planning, sections 7–8 give the proposed milestones; the appendices preserve exact source versions, dimensions, settings and audit evidence. Distinguish the team's reported recipe from the smaller experiments proposed for a tape user.

## Pin mechanics and replay semantics

The inspected release pins `kaggle-environments==1.32.7` and `kagg-engine==0.3.24`. Preserve those versions for reproduction; verify current competition rules separately when adapting to another environment version. Record interpreter/config hashes and source/license provenance.

- 720 recorded states give 719 decisions per seat. One full two-seat rollout counts 1,438 player steps.
- Pair `steps[t][seat].observation` with `steps[t+1][seat].action`. The student's input uses that seat's observation; opponent private data and future randomness are unavailable.
- Unit actions execute before market orders. Seeds bought in the current market phase cannot enable planting already processed. An overcommitted crop's planting requests are all rejected by the pinned interpreter.
- Workers may share positions. Reserve conflicting tasks/resources, rather than impose an invented universal square-occupancy rule.
- Shed capacity is 100 non-seed items. Account for carried cargo and nightly deposits, preserve feed wheat and liquidate before the last useful action, 718.
- Shops unlock with replacement and consume per instance. Town-center consumption in 1.32.7 is once daily. Use counts and a versioned demand model.
- Sales process unit by unit with both players. Do not value a large sale as quantity multiplied by the pre-sale quote.

## Build demonstrations

Supply a JSONL index with explicit teacher seats; paths are relative to the index:

```json
{"path":"replays/game-1.json.gz","episode_id":1,"seat":0,"submission_id":11111111}
```

The release accepts full JSON or JSON.GZ replays. Require matching version, 720 states and complete provenance before cache generation. Deduplicate episode-seat pairs. Split by episode/seed before creating transitions; both seats of a game share the split. Ignore invalid/unexecuted unit/order labels and use actual executed SELL quantities.

The historical final mix was 849 public trajectories plus both seats of 300 heuristic games: 100 normal, 100 with ≥2 tomato-related shops, 100 with ≥3 copies of one shop. These overrides belong in teacher generation; final evaluation keeps normal randomness. Counts describe a historical recipe, not a mandatory minimum or bundled dataset.

Source commands, run from the solution repository root:

```bash
python scripts/index_replays.py data/teacher-seats.jsonl --output data/replays
python scripts/prepare_replays.py --manifest data/replays/11111111/manifest.json --cache data/public-cache --workers 4
```

The cache expands to FP16 `[719,264,124]`, about 44.9 MiB per teacher trajectory. The existing loader is eager: 1,449 trajectories occupy about 63.5 GiB of feature arrays before overhead. On a smaller host, use fewer trajectories or implement bounded streaming; compressed NPZ is not directly memory-mapped. Assess host RAM separately from GPU VRAM.

## Choose a feasible first policy

For a new RL implementation on modest hardware, start with a small learned economic controller plus deterministic, state-aware logistics. Verify that every sampled choice affects the actual action. Then consider typed board/unit/global/commodity tokens for a full-unit policy.

For release reproduction, the final model has 12 blocks, width 256, 8 attention heads, up to 264 typed tokens and 124 feature columns. Unit heads have 500 candidates; ten market slots each decode 1,903 choices including absolute SELL quantities. The rule-updated memory token estimates opponent inventories from public evidence; it is not privileged state.

The 6-block bootstrap is a smaller ancestor architecture. The 1-block width-32 `smoke` preset is for compatibility checks, not an established competitive policy. Net2Net can insert zero-output residual blocks to preserve a trained function when growing depth. Final submitted agents were 10.23M, not the parallel 20M models.

## Train in stages

1. BC on valid teacher actions, with held-out episodes and per-head diagnostics. Confirm complete closed-loop games, not only loss or average accuracy dominated by PASS/NONE.
2. Fresh critic fitting with actor/trunk frozen. Verify the actor identity is unchanged and use disjoint validation seeds.
3. Current-policy self-play from the fitted checkpoint. Teacher weights supply KL regularization; this differs from teacher-only opposition.
4. Diagnose failures by shop mix, opponent and phase. Improve the heuristic in those regimes, regenerate targeted demonstrations and repeat BC/PPO.

Inspected defaults: terminal win/draw/loss `+1/0/-1`, `gamma=1`, GAE lambda 0.97, PPO clip 0.2, one epoch, Huber value weight 2, teacher `KL(teacher || policy)` coefficient 0.2, unit/market entropy 0.0015 each, Adam epsilon `1e-5`, gradient clip 5, base LR `5e-5`. These are a starting reproduction recipe, not guaranteed optimal settings for a reduced controller.

The fixed season does not reward early termination. Do not import mandatory step penalties or `gamma<1` from combat games. At 0.99, outcomes 718 decisions away are weighted about 0.00074. Treat dense money shaping as an explicit, outcome-validated ablation.

The public 64-game PPO config has `inference_batch_size = 2 * games` and requires `games % segments_per_minibatch == 0`; shards preserve seat pairs. Reduce compatible dimensions together. Measure simulation, encoding, inference, transfer and optimizer throughput before increasing rollout size. A published cluster count does not predict local throughput.

## Audit PPO correctness

- Store sampled IDs, old log probabilities, values, support masks, forced-factor flags and terminal outcomes. Recomputed probabilities must match the rollout policy before any update, with ratio approximately one.
- Use `ratio = exp(new_log_prob - old_log_prob)` and a clipped PPO objective. An updater using only `-log_prob * advantage` is not PPO-Clip.
- Conditional masks and temperature must match between sampling and updates. Sum probabilities across active learned action factors; exclude padding and deterministic forced factors.
- Verify learned outputs affect emitted actions. Test action-to-execution behavior, terminal delta/return alignment and mask validity, including simultaneous seed demand and sequential stock reservation.
- Distinguish raw terminal bank, training win reward and ladder rating. Test against active opponents rather than a function named “random” that emits PASS.

The downloaded public trainer linked by [this PPO notebook](https://www.kaggle.com/code/alejandrofonda/kaggriculture-ppo-training) exhibited these audit failures in the October 2 snapshot: no PPO ratio/clipping, rollout/update support mismatch, duplicated SELL mask prefix, unused learned sell choice and a PASS opponent. The [handbook audit](../../../../Handbook/reinforcement-learning/kaggriculture.md#public-trainer-audit-identify-the-exact-snapshot) records the exact file hash and code locations. Inspect the current dependency before applying these findings to another version.

## Match the controller to the training

**Final A:** greedy neural proposals → confidence-ordered repair and inventory safeguards → final-day search when budget permits. Search binary failure or insufficient overage falls back to the neural controller.

**Final B:** extra PPO with farmer-then-worker resource reservations and matching conditional masks at inference. Forced sales/drop are excluded from actor loss while outcomes train the critic. Neural policy controls the whole game, with no final-day search. Enabling its config flag alone does not reproduce its trained weights.

An arbitrary action repair cannot be scored under an unchanged policy as though it were sampled. If a deterministic controller replaces a factor, represent it explicitly in rollout data and objective bookkeeping.

## Evaluate and package

- Freeze exact teacher/baseline versions. Use diverse tape, heuristic and frozen-policy opponents; account for shared tape lineage.
- Swap seats at identical seeds. Keep a final seed/opponent panel untouched. Report win/draw/loss, within-game margin, shop-stratum outcomes, terminal waste and CPU latency.
- A mirror episode is a compatibility test. Public replay panels are observational and matchmaking-correlated; use seed-pair/episode-level resampling when estimating local uncertainty.
- Check simulator parity at boundaries: shared prices, unit-order execution, purchases, overflow, daily refresh, animal production/care, final cash-out and hidden-state observation filtering.
- Benchmark the exact packaged controller on CPU. JAX GPU training uses Linux wheels; native Windows NVIDIA support is absent, WSL2 is experimental. Agent A needs a Linux x86-64 planner binary. The upstream pipe-based evaluator needs Linux or Windows adaptation.
- Keep source, data and weight identities separate. The release omits trained weights, historical replay sets and elastic scheduling. A smoke test does not reproduce strength or Kaggle time accounting.

Refer to the release's `docs/data.md`, `docs/training.md`, `docs/architecture.md`, `docs/training-lineage.md`, and `docs/operations.md` for each mode. Read the relevant document when implementing that stage rather than loading every reference by default.
