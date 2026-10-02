# Kaggriculture: moving from tapes to BC, self-play PPO and heuristic control

Research snapshot: **October 2, 2026**. The linked team's writeup calls its result **current first place**; final standings were not established. The official submission deadline was September 30 at 23:59 UTC, and the ladder continues to settle through approximately October 15. [Discussion](https://www.kaggle.com/competitions/kaggriculture/discussion/745073), [official timeline](https://www.kaggle.com/competitions/kaggriculture/overview/timeline).

## 1. Competition DNA

Advanced Kaggriculture is a two-player economic control problem: farms are separate, but product prices respond to both players' trades and town demand. The season has 720 recorded states and 719 action decisions per player. Each turn combines a farmer action, actions for hired workers and up to ten ordered market operations. The official terminal reward is banked money; the ladder uses win/draw/loss, rather than the size of the money margin. [Official evaluation](https://www.kaggle.com/competitions/kaggriculture/overview/evaluation), [environment source](https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/kaggriculture).

Known rules do not make the game fully observed: the opponent's shed, seeds and carried inventories are private. Shops unlock randomly with replacement; weeds are stochastic. The competition is distinct from `kaggriculture_beginner`, which removes important economic and logistics mechanics.

This study inspected the released solution at commit `84057a0fda4238ccdebc46f9bf5496c6c4b2e00d` and the official **kaggle-environments 1.32.7** wheel. The GitHub environment's moving `master` is a discovery source; use the pinned package for reproduction. No training, performance ablation or head-to-head benchmark was run here.

## 2. What differs from public notebooks

The comparison below is based on downloaded source, not titles, notebook tags or advertised scores. The selection covers six public examples, not every notebook in the competition. Notebook versions and linked datasets can change independently.

| Approach / inspected source | Decision mechanism | What is learned | Useful strength | Limitation / difference from the team |
| --- | --- | --- | --- | --- |
| [Official Getting Started](https://www.kaggle.com/code/bovard/kaggriculture-getting-started) | Melon-focused movement, watering, harvesting and threshold sales | None | Clear API and first runnable agent | Narrow baseline; not a complete worker/animal/economic controller |
| [AgroBoss](https://www.kaggle.com/code/songoku2005/kaggriculture) | Greedy job priorities and worker assignment from live farm state | None | Reactive logistics without neural inference | Economic priorities are hand-set; its prose describes full observability, but opponent private inventories are hidden |
| [Adaptive Farming Strategy](https://www.kaggle.com/code/tetsutani/adaptive-farming-strategy-for-kaggriculture) | Five full-season action streams selected from early shop information, with execution repairs | None | Coherent schedules plus weed, storage, timing and endgame guards | Selects and protects a finite plan library; it does not learn a general state-to-action policy |
| [Multi-Route Farming Agent](https://www.kaggle.com/code/flexonafft/kaggriculture-multi-route-farming-agent) | Extracted V43 source includes a tape chassis, shop-pair routing and multiple reactive economic/closure layers | None in the inspected notebook | Demonstrates how sophisticated tape agents become | Complex runtime repair can be strong without being RL; downloaded source includes upstream attribution and exact-byte packaging |
| [Public PPO training example](https://www.kaggle.com/code/alejandrofonda/kaggriculture-ppo-training) plus its [linked trainer dataset](https://www.kaggle.com/datasets/alejandrofonda/kaggriculture-ppo-model) | Small MLP economic choices attached to scripted farming | BC plus a policy-gradient updater in the inspected source | A manageable reduced-action learning idea | The visible updater lacks PPO ratio/clipping; sampled sell choice is ignored by order generation; evaluation's “random” path actually passes |
| [What 2600+ Farms Do Differently](https://www.kaggle.com/code/georgymamarin/kaggriculture-what-2600-farms-do-differently) | Replay analytics, strategy fingerprints, noise and lineage analysis | No agent trained | Helps choose opponents and find failure regimes | Observational, sampled data; rankings and correlations do not establish causal strength or private implementation |
| [M & M & P & Q release](https://github.com/msdsm/kaggriculture-solution) | State-conditioned shared Transformer, replay imitation, current-policy self-play, rules and final-day planning | Unit actions, market orders and value | Learns across farms, logistics and shared economics while retaining deterministic safeguards | Requires data, valid action semantics, accelerated simulation and substantial training; code alone contains no trained weights |

For a tape user, the conceptual change is **learning from `(actual state, teacher action)` pairs**. A tape maps phase and route to a scheduled action. BC maps the current observation to an action distribution, which PPO can subsequently improve. Runtime guards remain valuable in both designs.

### Public trainer audit: identify the exact snapshot

The linked dataset's `train_rl.py` was downloaded on October 2, 2026. Its SHA-256 was `696d00f8cee6b117e789b0b7b34592805bfda6a15c9597ecfc6a1dc6c409ebda`. Static inspection and an isolated mask-lookup reproduction found:

- `ppo_update` (lines 684–717) uses `-log_prob * advantage`, without stored old log probabilities, an importance ratio or clipping. This is a policy-gradient update, not PPO-Clip.
- `run_episode` samples masked logits with a variable temperature, while the updater scores raw logits. The rollout and update distributions differ.
- The sell-mask lookup adds `SELL_` to choices already named `SELL_MELON`, etc. For legal flags `NONE=True` and `SELL_MELON=True`, its five-choice support becomes `[True, False, False, False, False]`.
- `make_sell_orders` (lines 465–475) never reads its `sell_act` argument; fixing the mask alone would not give the learned sell head control over emitted sales.
- The nested `random_agent` (lines 647–648) returns PASS with empty worker/market orders. That comparison measures earnings against an inactive opponent.
- The final money change after the last agent callback is omitted from the trajectory's reward deltas.

These findings apply to those file bytes, not every notebook called PPO. The trainer was not imported or trained in this audit. Recheck the dependency before using a newer version; fixing these issues is a prerequisite to interpreting a learning curve or match result.

## 3. Reading the three diagrams

The [training-loop diagram](https://github.com/msdsm/kaggriculture-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/docs/images/overview.png) shows an iterative process: imitate public agents; improve through self-play; inspect weaknesses; improve the heuristic; generate targeted demonstrations; return to BC. Their weakness cases included tomato-heavy demand and heavily repeated shop types. The heuristic was both a teacher and an inference controller.

The [chronology](https://github.com/msdsm/kaggriculture-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/docs/images/training_detail.png) distinguishes teacher trajectories, unique games and PPO experience. Final A's adopted ancestry totals **8,290,767 self-play games / 11,922,122,946 player decisions**. A game counts `719 × 2 = 1,438` steps. Parallel larger-model forks share ancestors and must not be summed into this total. Some larger-branch counts include post-deadline work. Final B's additional step count was not recovered. [Lineage details](https://github.com/msdsm/kaggriculture-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/docs/training-lineage.md).

The [architecture diagram](https://github.com/msdsm/kaggriculture-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/docs/images/model_detail.png) separates the neural policy from controller logic. Both submitted models use **12 blocks and 10,225,070 parameters**. A 20M family was explored, but was not the final submitted network. More parameters were one experiment alongside more games and better control/data; there is no published controlled ablation establishing which component supplied which gain.

## 4. Representation and coupled actions

Up to 264 typed tokens represent both farms' 200 cells, both players' 40 unit slots, a global token, 12 commodity/animal tokens, a rule-updated opponent-inventory estimate and ten market slots. The final feature schema has 124 columns with type-specific projections. The trunk is width 256, eight attention heads, pre-LayerNorm blocks, GELU feed-forward layers and selective 2D RoPE. The memory estimate uses public information; it is neither private-state access nor a recurrent neural hidden state. [Architecture documentation](https://github.com/msdsm/kaggriculture-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/docs/architecture.md).

Each own unit has 500 action candidates; each market slot decodes 1,903 choices, including absolute SELL quantities. Heads avoid enumerating the entire Cartesian product, but actions still compete for resources. Two planting requests can individually look valid while jointly exceeding available seeds. Multiple orders can spend the same money or stock. Unit moves to the same square are permitted; the conflict of interest is duplicate work/resource claims, not a universal occupancy restriction.

The official 1.32.7 interpreter rejects **all planting requests for a crop** when their combined demand exceeds its seed stock. Units act before market orders, so buying seeds this turn does not create seeds for this turn's planting. Post-processing or conditional masks must reflect this order.

## 5. Training/data details that matter

The team's data preparation aligns `observation[t]` with recorded `action[t+1]`, filters invalid/unexecuted labels and learns SELL's executed absolute quantity. It explicitly names teacher seats, checks replay version/hash/length, deduplicates episode-seat rows and keeps both seats of one episode in the same holdout. The last BC stage combined 849 public teacher trajectories with 600 synthetic trajectories from both seats of 300 heuristic games. Standard, tomato-heavy and repeated-shop scenarios each supplied 100 games. Scenario overrides were for teacher generation, not normal evaluation. [Data preparation](https://github.com/msdsm/kaggriculture-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/docs/data.md).

After BC, the adopted warmup fitted only the critic while preserving actor/trunk weights. PPO collected games with the **current policy in both seats**; the teacher provided a KL reference rather than being the sole opponent. The published defaults use terminal ±1/0, `gamma=1`, GAE lambda 0.97, clip 0.2, one PPO epoch, Huber value loss, teacher KL 0.2 and Adam. The 64-game preset is a single-device example, not the team's historical cluster configuration. [Training/checkpoints](https://github.com/msdsm/kaggriculture-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/docs/training.md).

In a fixed-length season, indiscriminate anti-stall penalties imported from combat games can favor underinvestment. A 0.99 discount heavily attenuates distant terminal outcomes. Reward design should reflect relative terminal wealth and be verified against match outcomes, rather than copy a generic recipe.

## 6. Final A and B are different controllers

**A:** greedy neural proposals receive confidence-ordered repair, stock/seed safeguards and overflow liquidation while retaining feed wheat. At final-day dawn, day 29, it can switch to a C++ task/route/trade planner with a bounded search budget; it falls back to neural control when loading/search/budget checks fail.

**B:** additional rule-aware PPO uses sequential farmer-then-worker masks. Earlier choices reserve resources for later ones. Sampling and update-time log probabilities use the same conditional support. Forced sale/drop factors are excluded from policy loss, while their outcomes still contribute to value targets. B uses neural control throughout the season and no final-day search handoff. [Controllers](https://github.com/msdsm/kaggriculture-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/docs/architecture.md).

Switching a config flag does not reproduce B's extra training. Arbitrarily repairing sampled actions and then scoring their replacements as if the policy sampled them can invalidate the actor update. Keep proposed, selected, forced and executed actions distinguishable.

## 7. Mechanics to audit before judging a policy

| Mechanic | Why it changes learning or evaluation |
| --- | --- |
| Unit work → market → town consumption → decay/night | Fresh seed/worker purchases cannot serve unit work already processed; nightly cargo may arrive too late to sell |
| Non-seed shed capacity 100 | Carrying extra cargo does not bypass overnight overflow; lost goods can erase otherwise productive work |
| Seeds live separately from shed items | Do not apply non-seed capacity rules to seed counts |
| Shops sampled with replacement, max eight instances | Repeated shops concentrate demand; counts matter, not just presence/absence |
| Town center consumes once daily in 1.32.7 | Older summaries can use different demand schedules; a version mismatch changes the economy |
| Simultaneous unit-by-unit market pricing | Selling a large quantity at one displayed spot price overestimates proceeds; opponent timing and order position matter |
| Animal `max_held` caps uncollected product | It is not a lifetime yield cap; feeding/care/collection cadence matters |
| Last useful action is indexed 718 | Do not expect a nonexistent action 719 to liquidate stock; plan physical delivery and sales before termination |

These mechanics were checked against the pinned package's source and config. Treat prose in old notebooks as hypotheses when it disagrees with the active interpreter.

## 8. Practical transfer and limits

For limited compute, preserve a strong tape as teacher/baseline, begin with BC on a compact market controller plus reactive logistics, and expand to unit decisions after learning/execution bookkeeping works. Fit the critic before PPO, and turn failure regimes into new demonstrations. Benchmark simulation/encoding/inference/update separately; copying a large model cannot solve a throughput bottleneck.

The released cache has FP16 `[719,264,124]` features: about 44.9 MiB per teacher trajectory. Its eager loader expands the complete 1,449-trajectory recipe to about 63.5 GiB before overhead. A 32 GB host needs smaller data or a bounded/streaming loader. This is a calculation from source shapes, not a memory benchmark. [Loader](https://github.com/msdsm/kaggriculture-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/data/dataset.py).

Evaluation needs diverse active opponents, seat-swapped seeds, hidden development holdouts, full seasons, outcome and within-game margin, and phase/shop diagnostics. Raw earnings against PASS, one mirror match or a notebook's historical rating does not establish generalization. Replay analytics also have selection and serial-correlation limits. [Public replay report](https://www.kaggle.com/code/georgymamarin/kaggriculture-what-2600-farms-do-differently).

The source release omits historical weights, replay datasets and elastic cluster scheduling. Native Windows cannot train JAX on NVIDIA through the documented wheels; Linux/WSL2 is the relevant path for that stack, with WSL2 GPU support marked experimental. The planner submission binary must target Linux x86-64. [Release setup](https://github.com/msdsm/kaggriculture-solution/tree/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d), [JAX support matrix](https://docs.jax.dev/en/latest/installation.html).

## Reusable workflow

See the [Kaggriculture skill reference](../../.agent/skills/kaggle-rl-simulation/references/kaggriculture.md) for replay alignment, legal joint actions, PPO checks and staged reproduction. Recommendations here are proposed experiments; only attributed code behavior and the team's reported history are established by the inspected sources.
