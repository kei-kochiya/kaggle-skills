# Topic 750617: 10th place: BC + PPO on a Single RTX 5090

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/750617
- **Author**: 
- **Post Date**: 2026-10-19T07:20:13.277000
- **Votes**: 7

## Messages

### Message 1 by  (Votes: 7)

<p>Code, weights and reproduction instructions: <a href="https://github.com/KawattaTaido/kaggriculture-release" target="_blank">https://github.com/KawattaTaido/kaggriculture-release</a> (CC BY 4.0).</p>
<p>Thank you to the organizers for hosting this competition.</p>
<h2>1. Summary</h2>
<p>The agent is <strong>one policy network</strong> (21.8M parameters, JAX/Flax), with no search and no
hand-written planner. It is trained in two steps:</p>
<ol>
<li><strong>Behavior cloning (BC)</strong> on the most recent leaderboard games: 2,380 episodes from
the last four days before the final submission, using both seats.</li>
<li><strong>PPO</strong> (≈370M environment steps, ≈19 h on one RTX 5090). In half of the games the
agent plays itself. In the other half it plays one opponent from a small pool: two
past generations of itself, or a <strong>recording of a real LB agent</strong> replayed move by move.</li>
</ol>
<p>Three things mattered most:</p>
<ul>
<li><strong>The action representation follows the game.</strong> Each worker unit <em>points at a target
tile</em>, and the market head outputs <em>a quantity per item</em>.</li>
<li><strong>The strength of the BC starting point, driven by fresh LB data.</strong> A new day of LB
games was released every morning, and rebuilding the BC teachers from the latest days
made the model stronger almost every time. Imitating top agents was the cheapest large gain (Section 4).</li>
<li><strong>Model size.</strong> Making the network wider kept improving the results; the last two
widenings took it to 11.96M and then 21.83M parameters. The gain does not show in BC;
it shows only after PPO (Section 4).</li>
</ul>
<p>At a glance:</p>
<ul>
<li><strong>Most important inputs</strong>: the own-farm map with plant and animal states, market
prices, stocks and price impact, the money and stock of both players, and estimates of
the opponent's hidden barn and of when the opponent will deliver.</li>
<li><strong>Tools</strong>:<ul>
<li>JAX / Flax / Optax for the network and PPO; a JAX port of the environment runs all
PPO rollouts on the GPU.</li>
<li>Rust (PyO3) for the game rules, legal-action masks and observation features. The
same compiled extension is used in training and in the submission.</li>
<li><code>kaggle-environments</code> as the reference implementation.</li></ul></li>
<li><strong>Training time</strong> (one RTX 5090): ≈23 h from the release of the last day of LB data to the
final weights; ≈3.5 h for the data and BC, ≈19 h for PPO.</li>
<li><strong>Inference</strong>: ≈18 ms per turn locally, greedy (argmax) decoding. No ensemble.</li>
</ul>
<hr>
<h2>2. Features selection / engineering</h2>
<p>The input is not a feature table but the game state, encoded into three tensors
(specification v12). Every feature is computed in Rust; a JAX implementation is checked
against it bit by bit, and replays of LB games verify both.</p>
<table>
<thead>
<tr>
<th>Input</th>
<th>Shape</th>
<th>Content</th>
</tr>
</thead>
<tbody>
<tr>
<td>map</td>
<td>65 × 10 × 10</td>
<td>own farm: tile type, crop, plant state (water, fertilizer, turns to maturity, expected yield), animal state, unit occupancy, <strong>number of harvests left in the season</strong></td>
</tr>
<tr>
<td>global</td>
<td>312</td>
<td>time and phase one-hots, money, seeds, barn and carried stock, market prices / stocks / price impact, town demand, labor, land, recent history, expected supply, <strong>estimated opponent stock and delivery timing</strong>, stock in hand + barn</td>
</tr>
<tr>
<td>units</td>
<td>24 × 25 (+ mask)</td>
<td>per worker: position, carried items, distance to the barn, …</td>
</tr>
</tbody>
</table>
<p>Most important, judged by what changed results when added or removed:</p>
<ol>
<li><strong>Market state and price impact.</strong> Selling moves the price; both players share one
market, and the town buys only a fixed amount (≈206k per game in total over all
items).</li>
<li><strong>Opponent estimates.</strong> The opponent's barn is not observable, so its stock is
estimated from public information (purchases, harvests, sales), together with when it
will deliver. An auxiliary head that predicts the opponent's sales 1 and 24 turns ahead
(loss weight 0.05) trains the same representation.</li>
<li><strong>Plant state, including the harvests left in the season.</strong> Continuing crops
(strawberry, tomato) are worth more when several harvests remain.</li>
</ol>
<p>Feature selection and transformations:</p>
<ul>
<li>Features were added one hypothesis at a time and kept only if BC <em>and</em> PPO improved,
judged by paired comparisons on the same boards.</li>
<li>A map of the opponent's farm was removed (no effect). Two later specifications were not
adopted because they lost after PPO (Section 4).</li>
</ul>
<p><strong>External data</strong>: none apart from the LB replays published through the Kaggle API (the
competition's own episodes).</p>
<h2>3. Training methods</h2>
<p>No ensembling; one network.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F7165289%2F3ee28dc56624a1f1bdfd64dea415ddf9%2Ffig1_pipeline.png?generation=1792394221064026&amp;alt=media" alt="Training pipeline"></p>
<p><em>Figure 1. Training pipeline. LB replays feed both behavior cloning and the "tape" slot of
the PPO opponent pool. Every 210 PPO iterations the KL teacher is replaced by the current
weights, and the current weights join the opponent pool.</em></p>
<h3>Network (Figure 2)</h3>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F7165289%2F2ba5ce5f90a94b94620696dc53fdea9c%2Ffig2_network.png?generation=1792394204575262&amp;alt=media" alt="Network"></p>
<p><em>Figure 2. Network. The own-farm map is encoded by a ResNet and self-attention, the global
vector by an MLP, and the worker units attend to the board. Three heads produce unit
actions, market orders and value/auxiliary predictions. Every factor of every action is
masked by legal-action masks computed in Rust.</em></p>
<table>
<thead>
<tr>
<th>Part</th>
<th>Details</th>
</tr>
</thead>
<tbody>
<tr>
<td>Map encoder</td>
<td>ResNet, 6 blocks, 96 channels, GroupNorm; FiLM conditioning on the global vector; 2 self-attention layers over the 100 tiles</td>
</tr>
<tr>
<td>Global encoder</td>
<td>MLP, hidden 512 → g (1024)</td>
</tr>
<tr>
<td>Unit encoder</td>
<td>2 cross-attention layers from units to tiles, with a relative-position bias, then unit self-attention</td>
</tr>
<tr>
<td><strong>Unit head (pointer)</strong></td>
<td>per unit, autoregressive: operation, <strong>target tile</strong> (logit = query·tile key + learned function of distance / dx / dy / reservations of earlier units), argument, quantity. A unit on its target performs the operation; otherwise it moves one step along a fixed path. Targets are re-chosen every turn. A tile and operation already reserved by an earlier unit is masked, so units never duplicate work.</td>
</tr>
<tr>
<td><strong>Market head</strong></td>
<td>a turn's market action is a <em>quantity per item</em> (21 factors, 352 logits), decoded autoregressively in execution order by a 2-layer transformer over 14 item tokens. Budget and order-slot limits are tracked while decoding, so every sampled action is legal. The orders are then placed into the 10 order slots by a fixed rule: sells of items the opponent also holds come first, largest expected price drop first, because the environment processes the two players' slots alternately and earlier slots sell at higher prices.</td>
</tr>
<tr>
<td>Value / auxiliary</td>
<td>value head (final money difference), auxiliary heads (final money, win probability, opponent sales)</td>
</tr>
<tr>
<td>Size</td>
<td>21.83M parameters; compute in bfloat16</td>
</tr>
</tbody>
</table>
<p><strong>Legal-action masks</strong> are computed in Rust and applied to every factor. Besides the
rules, they remove actions that are provably wasteful:</p>
<ul>
<li>market orders that are certainly loss-making or cannot fill;</li>
<li>digging up a plant before harvest (continuing crops excepted);</li>
<li>fertilizer that does not raise the yield;</li>
<li>planting that cannot mature before the season ends;</li>
<li>care on the last day;</li>
<li>dropping items into a full barn.</li>
</ul>
<p>Each mask was checked by replaying LB games, so it does not remove moves that top agents
actually play.</p>
<h3>Behavior cloning</h3>
<ul>
<li><strong>Teachers</strong>: 2,380 LB games of 2026-09-26..29 in which both players had a rating of at
least 2,800, both seats (3.4M samples; 1,891 games for training, 489 held out).</li>
<li><strong>Labels</strong>: LB actions are projected onto the network's action representation (target
tile for units, quantity per item for the market). A sell and a buy of the same item in
the same turn are netted, and sells are budgeted against the barn <em>after</em> the unit
actions of the turn.</li>
<li><strong>Training</strong>: 60k steps, batch 512, Adam with lr 3e-4 and a cosine schedule, gradient
clip 1.0, auxiliary opponent-sales loss 0.05. Hold-out NLL 5.35 per sample (retrained model in the repository; the original BC checkpoint was lost).</li>
</ul>
<h3>PPO</h3>
<table>
<thead>
<tr>
<th>Setting</th>
<th>Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>Environment</td>
<td>JAX port, 64 games in parallel, whole 720-turn game in <code>lax.scan</code> on the GPU</td>
</tr>
<tr>
<td>Reward</td>
<td>final money difference / 30,000, terminal only; γ = λ = 1</td>
</tr>
<tr>
<td>Update</td>
<td>clip 0.2, 2 epochs, 16 minibatches, entropy 0.02, value 0.5, auxiliary 0.1, grad norm 0.5, early stop on KL (0.003)</td>
</tr>
<tr>
<td>Learning rate</td>
<td>stage 1: 5e-5 for 420 iterations; stage 2: <strong>1e-5</strong> for 3,570 iterations</td>
</tr>
<tr>
<td>Kickstarting</td>
<td>KL to a teacher policy, coefficient 1.0. The teacher starts as the BC model and is <strong>replaced by the current weights every 210 iterations</strong> ("segments").</td>
</tr>
<tr>
<td>Opponents</td>
<td>50% self-play; 50% one of a pool of 3: two past generations (pushed every 210 iterations) and one <strong>LB tape</strong></td>
</tr>
<tr>
<td>TV guard</td>
<td>after each update, if the 90th percentile over games of the change in action probabilities (total variation) at any turn exceeds 0.5, the update is shrunk to ½ or ¼</td>
</tr>
</tbody>
</table>
<p><strong>LB tape opponent.</strong> One pool slot replays a recorded LB agent. The recordings come from
800 games: 657 LB games of 2026-09-25..28 (both ratings ≥ 2,800) and 143 LB games that our earlier submissions lost. A
recording is open-loop, so two things are aligned with the original game:</p>
<ul>
<li><strong>Randomness.</strong> The tape seat's random events (weeds, the shop unlock order) follow the
recording.</li>
<li><strong>Market fills.</strong> For every order of the recorded player, the number of units that
filled on the LB is precomputed. The replay stops at that number, and missing money is
lent and subtracted at the end.</li>
</ul>
<p>Without this alignment, a difference of a few coins on day 0 makes the tape's first
purchases fail, and in about 30% of games the opponent collapses. Opponents are sampled
per team, and teams that are harder to beat are drawn more often.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F7165289%2F5596bdf5d645902108cbc45081892351%2Ffig3_curves.png?generation=1792394254023811&amp;alt=media" alt="Learning curves"></p>
<p><em>Figure 3. During PPO training, the win rate against the recorded LB agents (the tape
slot of the pool) rose from 0.53 to ≈0.88.</em></p>
<h2>4. Interesting findings</h2>
<p><strong>What mattered most</strong></p>
<ol>
<li><strong>Fresh BC teachers.</strong> A new day of LB games was released every morning, and rebuilding
the BC teachers from the latest four days made the model stronger almost every time.
This was the cheapest large gain. On one model
generation it raised the win rate against a strong public rule-based agent from 0.835
to 0.993.</li>
<li><strong>An action representation that matches the game</strong>: pointing at tiles instead of
choosing move directions, quantities per item instead of a fixed list of order slots,
and sell budgets counted on the real barn.</li>
<li><strong>Model size, which shows up in PPO, not in BC.</strong> Every widening of the network improved
the final result. Wider fully connected layers barely changed the BC hold-out NLL
(−0.04) and lost BC-vs-BC games. At the same PPO step count, however, they raised the
win rate against tapes from 0.454 to 0.589.</li>
<li><strong>Realistic recorded opponents.</strong> Self-play alone never meets some LB behaviors, such
as selling in the same turn as a delivery or replanting late. The tapes only helped
once randomness and fills were aligned (Section 3).</li>
<li><strong>A low learning rate for long PPO runs.</strong> At 5e-5 the run stopped improving after
about 40M steps. Restarting at 1e-5:<ul>
<li>lowered the clipped fraction from 0.26 to 0.09;</li>
<li>gained +355 money against held-out LB tapes in two seeds;</li>
<li>won head-to-head (0.56 and 0.69);</li>
<li>kept improving for the remaining 3,500 iterations (Figure 3).</li></ul></li>
<li><strong>Guarding against "opening flips".</strong> A single ordinary-sized update sometimes flipped
the near-deterministic opening purchases (for example buying 3 geese on turn 1). Each
flip sent the policy into another strategy. A KL measured on sampled actions cannot
see this (≈3e-5 on average). The per-turn TV guard removed all such spikes in a
verification run, at a cost of +9% time per iteration.</li>
</ol>
<p><strong>Relationships in the game</strong></p>
<ul>
<li><strong>Fixed demand.</strong> The town buys a fixed amount per game (≈206k in total), and the two
players' sales nearly saturate it, so taking the opponent's share is the shortest way
to win.</li>
<li><strong>Shared market.</strong> When our money rose by 1,000, the opponent's rose by 690–790, through
the shared market. Maximizing our own money and maximizing the win rate are therefore
different objectives. The PPO reward is the money <em>difference</em>.</li>
<li><strong>One loss-making action.</strong> A single legal but certainly loss-making action (for
example an animal that cannot pay back before the end) is enough to make PPO suppress
all spending, including planting. Masking provable losses was a prerequisite for
learning.</li>
</ul>
<p><strong>Tried and not adopted</strong> (each lost in paired evaluation):</p>
<ul>
<li>day-level planners and solvers for the workforce (timed out on Kaggle or lost
head-to-head);</li>
<li>local search over the policy's actions;</li>
<li>a calendar one-hot plus "next land" features, and opponent-unit cross-attention (both
better in BC, both worse after PPO);</li>
<li>a weaker kickstart (0.3);</li>
</ul>
<h2>5. Simple features and methods</h2>
<ul>
<li><strong>Simplest useful model</strong>: the BC model alone, with the same network and inputs and
no PPO. It already plays a complete, sensible economy because it imitates recent top
LB agents, and it needs no long training (≈3.5 h on an RTX 5090, data preparation included).</li>
<li><strong>Most important step</strong>: PPO starting from BC. Figure 3 shows the gain against recorded
LB agents during training. Most of the improvement comes in the first ≈2,000
iterations (≈10 h on an RTX 5090).</li>
<li><strong>Smaller input subsets</strong> were not evaluated systematically. The input is the encoded
game state rather than a hand-made feature table, so it does not reduce to a short list
of features.</li>
</ul>
<h2>6. Model execution time</h2>
<table>
<thead>
<tr>
<th></th>
<th>Time</th>
</tr>
</thead>
<tbody>
<tr>
<td>LB data download, BC dataset and BC training (60k steps)</td>
<td>≈3.5 h</td>
</tr>
<tr>
<td>PPO stage 1 (420 iterations)</td>
<td>≈2.2 h</td>
</tr>
<tr>
<td>PPO stage 2 (3,570 iterations)</td>
<td>≈17 h</td>
</tr>
<tr>
<td>Total</td>
<td>≈23 h on one RTX 5090 (≈17 s per PPO iteration)</td>
</tr>
<tr>
<td>Prediction</td>
<td>≈18 ms per turn locally; about 35 ms per turn and ≈35 s for the first turn (JIT compilation) on Kaggle's runtime, measured on an earlier submission. A game has 720 turns.</td>
</tr>
<tr>
<td>Simplified model (BC only)</td>
<td>training ≈3.5 h; prediction as above</td>
</tr>
</tbody>
</table>

---

