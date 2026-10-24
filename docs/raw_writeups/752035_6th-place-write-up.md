# Topic 752035: 6th place write-up

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/752035
- **Author**: 
- **Post Date**: 2026-10-23T14:32:09.427000
- **Votes**: 3

## Messages

### Message 1 by  (Votes: 3)

<h1>6th-place solution: replay-conditioned imitation learning and PPO</h1>
<p>Thanks to the organizers for a fun and surprisingly difficult game, and to everyone who shared agents, replays, and ideas during the competition.</p>
<p>Our final solution combined behavior cloning (BC) from strong public players with reinforcement learning. The central idea was simple: start from a policy that already knows how to farm, preserve its reliable opening, and use PPO to improve the rest of the game.</p>
<p>The model used atomic actions, a small transformer, and a reference action sequence from a replay as an additional input. In the submitted agent, the original BC policy played days 1–11, and the PPO policy took over from day 12. Both used greedy inference.</p>
<h2>Why this game was difficult</h2>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1648129%2F71476acc14711be054c125e15fa184ed%2Fgame-overview.png?generation=1792765603696425&amp;alt=media" alt="A Kaggriculture game, with the two farms and their shared market."></p>
<p>Kaggriculture is a farming game, where a successful player has to coordinate several actions at once: movement, planting, watering, animals, hired hands, inventory, and a shared market. Useful actions often pay off much later. Buying a seed is only the beginning of a chain that ends with harvesting, transporting, and selling the product. This co-dependency between actions spanning many turns and even days to produce a single cohesive strategy makes exploration very challenging.</p>
<p>Additionally, for reinforcement learning, that meant a long horizon—30 days of hourly decisions—with a large action space and a final outcome that depends on hundreds of earlier choices. Meanwhile, strong deterministic agents were improving quickly on the public leaderboard. Learning basic farming from scratch was an expensive way to try to keep up.</p>
<p>We therefore put substantial effort into two things before the final PPO stage: a strong imitation-learning starting point and a local evaluation system we could trust.</p>
<h2>Local validation</h2>
<p>We maintained two complementary leagues.</p>
<p>The CPU league contained around 200 automatically collected public agents and used the original game engine. We also used coding-agent workflows to port public agents to C++ for faster execution.</p>
<p>The GPU league contained around 20 of our local BC, PPO, and deterministic agents. It used a PyTorch-based GPU game engine, with batches of 128 games and 128 games per pairwise matchup. Some deterministic agents were ported to CUDA so they could participate efficiently.</p>
<h2>Behavior cloning</h2>
<p>Our BC work started by identifying useful teachers. We cloned the top 20 public agents, played them against one another and against our league, and studied the strongest three more closely.</p>
<p>Fine-tuning was progressive rather than a single pass over an undifferentiated replay dataset. We used time-based stages and similarity-based refinement, focusing on similar game states at roughly the same point in a game. Candidate models were evaluated against the latest GPU league.</p>
<p>The important modeling choice was replay conditioning.</p>
<h3>A reference tape as an input</h3>
<p>A reference tape is a recorded sequence of actions from a game. At turn <code>t</code>, the network receives the corresponding reference actions alongside the current observation and recent history. These actions are embedded into the decoder inputs.</p>
<p>The tape provides a useful starting suggestion: what did a strong player do at this point in a game? The network still sees the actual board, inventory, market, and units, and predicts its own actions. It is not forced to replay the tape, and it is not simply executing a script with occasional overrides.</p>
<p>This was particularly useful for a game with long, coordinated sequences of actions. Imitation supplied examples of those sequences without requiring PPO to discover all of them through random exploration.</p>
<p>We explored multi-tape BC variants as well, including a 64-tape version. However, the final PPO submission described used one fixed reference tape, shared by the opening and PPO policies.</p>
<h2>Policy architecture</h2>
<p>The policy had approximately <strong>2.2 million parameters</strong>. The opening and PPO models used the same architecture, with separate weights.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1648129%2F7e30700e4f11619c4d6bbe39ceb71435%2Farchitecture.png?generation=1792765651618106&amp;alt=media" alt="Architecture of the final replay-conditioned atomic-action policy."></p>
<p>The state representation combined public game information, our private inventory, and four turns of action history. Both farms, own and opponent units, products, shops, and inventory were represented as tokens. Board features passed through a small convolutional stem before joining the other tokens.</p>
<p>The resulting sequence contained 469 tokens of width 128. A four-layer, pre-norm transformer encoded them, using eight attention heads and a feed-forward width of 512.</p>
<p>There were two output branches:</p>
<ul>
<li><strong>Units:</strong> 32 query slots for the farmer and padded hired-hand slots. These were decoded in parallel, with inactive slots masked out.</li>
<li><strong>Market:</strong> up to 10 orders followed by a stop token. This branch was autoregressive, so later orders could condition on earlier orders. It also attended to state and unit representations.</li>
</ul>
<p>Both branches used two decoder blocks. Shared output heads predicted an operation, an argument, and a quantity, with 26, 13, and 115 classes respectively. Operation-specific masks restricted the applicable fields and choices.</p>
<p>These were atomic game commands, for example, moving north, planting a crop on the current tile, and buying seeds were separate decisions. The network had to learn how to combine them over time.</p>
<p>For PPO, we added a training-only critic. It received detached policy features and privileged inventory information from both players. The deployed actor did not receive the opponent's hidden inventory, and the critic was not part of submission inference.</p>
<h2>PPO: preserve the opening, improve the finish</h2>
<p>Starting from a strong BC policy changed the question from “can this agent learn to farm?” to “can it improve decisions beyond imitation?”</p>
<p>We initially trained only from day 18 onward, then expanded the learning window to day 12. The selected submission used the following split:</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1648129%2F2e381e0b72fa4f03998e6d5bf1189222%2Fppo-handoff.png?generation=1792765718812253&amp;alt=media" alt="The frozen BC opening, PPO training window, and two-model deployment."></p>
<p>We ran the frozen BC policy greedily across many seeds and cached the resulting game states and recent histories. PPO then collected complete trajectories from the handoff point to the end of the game. This avoided repeatedly simulating the opening during training and shortened the horizon the learner needed to optimize.</p>
<p>All weights of the late-game policy were trainable—not just the action heads. The opening model remained frozen, and dropout was disabled during PPO.</p>
<p>The deployment had to match this training setup exactly. We shipped both sets of weights, used the original model for days 1–11, and switched at the beginning of day 12, corresponding to step 264. The four-turn history continued across the switch; it was not reset.</p>
<p>This is a deliberate trade-off. PPO can improve the game from the handoff onward, but it cannot change the frozen opening or learn a completely different opening strategy.</p>
<h3>Opponents and reward</h3>
<p>The selected training recipe used 90% self-play and 10% play against a frozen incumbent. These proportions refer to retained learner trajectories: self-play supplies trajectories from both players, whereas games against the incumbent supply only the learner's trajectory.</p>
<p>The incumbent initially came from the BC model and was replaced with a snapshot of the current policy every 250 million player-steps. It played greedily. The learner sampled actions during collection.</p>
<p>We used only a terminal reward, combining the match outcome with a bounded final-bank margin:</p>
<pre><code>outcome = 1 if our_bank &gt; opponent_bank else -1 if our_bank &lt; opponent_bank else 0
reward = 0.5 * outcome + 0.5 * tanh((our_bank - opponent_bank) / 2000)
</code></pre>
<p>The outcome term rewards winning. The margin term distinguishes close results and saturates for large differences. There were no daily rewards or intermediate net-worth rewards in this final recipe.</p>
<h3>Training settings</h3>
<table>
<thead>
<tr>
<th>Setting</th>
<th>Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>Optimizer</td>
<td>AdamW</td>
</tr>
<tr>
<td>Learning rate</td>
<td><code>6e-6</code></td>
</tr>
<tr>
<td>PPO epochs per rollout</td>
<td>1</td>
</tr>
<tr>
<td>Discount / GAE lambda</td>
<td><code>1.0</code> / <code>1.0</code></td>
</tr>
<tr>
<td>PPO clipping</td>
<td><code>0.2</code></td>
</tr>
<tr>
<td>Value clipping / value-loss weight</td>
<td><code>0.2</code> / <code>0.5</code></td>
</tr>
<tr>
<td>Entropy weight</td>
<td><code>1e-6</code></td>
</tr>
<tr>
<td>Gradient-norm cap</td>
<td><code>1.0</code></td>
</tr>
<tr>
<td>Aggregate optimizer minibatch</td>
<td>8,192 player-turn samples</td>
</tr>
<tr>
<td>Aggregate rollout</td>
<td>About 5.2 million player-turn samples</td>
</tr>
<tr>
<td>Incumbent refresh</td>
<td>Every 250 million player-steps</td>
</tr>
</tbody>
</table>
<p>A player-step here means one player's decision for one game turn, not one individual unit command. One turn can contain actions for several units and multiple market orders.</p>
<h2>Making training practical</h2>
<p>The GPU game engine and batched policy inference were important parts of the solution. Neural inference alone was not enough to make training fast: game simulation, observation construction, action decoding, and moving rollout data through the learner all mattered.</p>
<p>The final implementation used mixed precision, CUDA graphs for collection, and compiled learner computation. Cached openings removed work that did not contribute PPO training samples. Distributed training split the aggregate rollout and optimizer batches across GPUs.</p>
<h2>What we would take away</h2>
<p>For us, the most useful combination was a strong imitation-learning starting point, a shorter PPO training horizon, and evaluation that matched the submitted agent.</p>
<ul>
<li><strong>BC supplied coordinated behavior.</strong> It gave PPO a policy that already knew useful farming sequences instead of asking it to discover everything from scratch.</li>
<li><strong>The opening/finish split made the problem more manageable.</strong> It reduced collection cost and protected a reliable opening, at the cost of restricting what PPO could improve.</li>
<li><strong>Throughput and learning quality were separate questions.</strong> More samples per second were valuable, but changes to batch geometry could change convergence too.</li>
<li><strong>The training chart was not the final test.</strong> Sampled play against a changing incumbent was useful feedback; greedy head-to-head evaluation was the checkpoint-selection criterion.</li>
</ul>
<p>The final result was not PPO learning the entire game from scratch. It was imitation learning providing a strong base, followed by PPO improving the part of the game where we allowed it to act.</p>

---

