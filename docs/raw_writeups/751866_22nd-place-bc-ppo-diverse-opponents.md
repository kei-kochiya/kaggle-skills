# Topic 751866: [22nd Place] Behavior Cloning and PPO with Diverse Opponents

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/751866
- **Author**: 
- **Post Date**: 2026-10-22T23:59:34.437000
- **Votes**: 0

## Messages

### Message 1 by  (Votes: 0)

<h2>Introduction</h2>
<p>I initialized the policy with behavior cloning (BC) from strong-player replays and then improved it with PPO. Simulating all 720 turns of every episode was expensive, however, and self-play against fixed opponents encouraged narrow strategies.</p>
<p>I therefore split the game into an opening model and a mid-to-late-game model, pre-generated states at the end of the opening, and trained against a mixture of opponent policies. This writeup describes the policy network, BC, PPO, and the inference path used in the final submission.</p>
<p>The implementation is available in the <a href="https://github.com/atsushi11o7/kaggriculture" target="_blank">GitHub repository</a>.</p>
<h2>TL;DR</h2>
<ul>
<li>An Opening model handles displayed days 1–11; a Day 11+ model handles days 12–30.</li>
<li>Observations become 128-dimensional tokens processed by a four-layer Transformer.</li>
<li>The Actor predicts 25 unit slots and 10 market slots in parallel. A sequential Executor resolves feasibility while updating game state.</li>
<li>The Critic shares the public-state encoder and additionally sees private state and macro features during training.</li>
<li>Day 12 starting states are pre-generated, so PPO rolls out only the remaining 456 turns.</li>
<li>PPO opponents include self, a fixed anchor, past checkpoints, and player-specialized BC models.</li>
<li>Candidates are promoted to best only after passing evaluations against both best and anchor.</li>
<li>A PPO checkpoint fine-tuned with strong-player replays ultimately outperformed continuous PPO training and finished 22nd, earning a gold medal.</li>
</ul>
<h2>1. Overall Design</h2>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F18310689%2F7191213b0ba2c810f1b171e8d554e487%2Ftraining_pipeline.png?generation=1792712671168593&amp;alt=media" alt=""></p>
<p>Internally, <code>obs["day"]</code> is zero-indexed. I distinguish raw days from displayed days where needed.</p>
<table>
<thead>
<tr>
<th>Model</th>
<th>Raw day</th>
<th>Displayed day</th>
<th>Turns</th>
</tr>
</thead>
<tbody>
<tr>
<td>Opening model</td>
<td>0–10</td>
<td>1–11</td>
<td>264</td>
</tr>
<tr>
<td>Day 11+ model</td>
<td>11–29</td>
<td>12–30</td>
<td>456</td>
</tr>
</tbody>
</table>
<p>This split was driven by compute constraints. I trained on one RTX 5070 Ti without rented compute. With a larger budget, I would likely have trained one model over the full game.</p>
<p>To reduce the search space, I selected strong players who consistently acquired all four plots by raw day 10 and reproduced that behavior with BC. The Day 11+ problem then became simpler: start with four usable plots and some accumulated money and resources, and earn more than the opponent. PPO only had to optimize this later phase.</p>
<p>Replay analysis suggested that opening actions depended more on available town shops and the player's resources than on the opponent's strategy. This was exploratory rather than a controlled statistical result, but relatively few opening branches appeared to change substantially in response to the opponent. The Opening model still receives the opponent's public state; only its data and training range are restricted.</p>
<p>Rather than rerun the Opening model for 264 turns in every PPO episode, I generated day 12 states for both players and stored them in a state bank. PPO sampled these states and simulated the remaining 456 turns. The submission used no bank: the Opening model handled raw days 0–10 and the Day 11+ model handled raw days 11–29.</p>
<h2>2. Policy / Value Network</h2>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F18310689%2F301c321c19be0fa165593f0e8c2d953b%2Fpolicy_network_architecture_portrait.png?generation=1792712692009653&amp;alt=media" alt=""></p>
<h3>2.1 State Representation</h3>
<p>Global information, player features, board objects, sheds, and market information are converted into a fixed-length token sequence. Sparse categorical features use separate embedding tables; embeddings belonging to one entity are summed into a 128-dimensional token.</p>
<p>A global token begins the sequence. Type and position embeddings are added, padded positions are removed with an attention mask, and a four-layer, four-head, pre-LayerNorm Transformer with GELU processes the sequence.</p>
<p>This preserves entity structure and its interactions instead of flattening the board into one dense vector. It also made identical preprocessing easier to reproduce in JAX and PyTorch.</p>
<h3>2.2 Parallel Actor and Sequential Executor</h3>
<p>The Actor processes 35 queries at once: 25 unit slots and 10 market slots. Query self-attention and cross-attention to state tokens are followed by logits for each slot's operation, target, quantity, and other arguments.</p>
<p>All slots are computed in one forward pass, so this is not an autoregressive decoder in which one choice changes later distributions. It preserves inference speed but cannot represent every dependency among sub-actions by itself.</p>
<p>The Executor treats Actor outputs as candidate intents and processes them in a fixed order while updating a shadow state. Shared inventory, cash, quantity limits, conflicts on the same target, and market STOP are resolved against the state produced by preceding intents.</p>
<p>This is less expressive than autoregressive decoding, but avoids running the Transformer once per slot. Most relevant dependencies arose because sequential execution changes the environment state, making this a practical balance between speed and complexity.</p>
<h3>2.3 Legal and Strategy Masks</h3>
<p>Deterministic legal masks remove mechanically impossible actions before sampling: for example PICKUP/DROP without an adjacent shed, PICKUP for absent inventory, or purchases exceeding available cash.</p>
<p>Separate strategy masks suppress investments that cannot produce value before the game ends. PLANT, BUY_SEED, BUY_ANIMAL, and BUY_LAND are removed when first production cannot finish in time. Immediately useful operations such as HIRE are not prohibited unconditionally. Legal masks represent game rules; strategy masks encode chosen heuristics.</p>
<p>Parallel selection can still create within-turn conflicts, so the Executor performs the final consistency checks.</p>
<h3>2.4 Asymmetric Critic</h3>
<p>The Critic shares the public-state Transformer encoder with the Actor. A dedicated encoder also processes both players' private state and combines it with game-level macro features before value prediction.</p>
<p>The Actor sees only information available at submission time. The Critic receives full training state to simplify value estimation. The private encoder, macro MLP, and value head are not exported. Each exported Opening and Day 11+ Actor has approximately 2.03 million parameters.</p>
<h3>2.5 Matching JAX and PyTorch</h3>
<p>The simulator and neural-network training core were implemented in JAX and batched on GPU. Replay analysis, BC cache construction, and data loading ran on CPU; the Python data loader was sometimes the bottleneck. Submission inference used converted PyTorch weights on CPU.</p>
<p>Tokenizer, Transformer, query decoder, masks, and Executor semantics were kept aligned across both implementations.</p>
<h2>3. Behavior Cloning</h2>
<h3>3.1 Fixed-Shape Targets from Replays</h3>
<p>Replay JSON files were converted on CPU into fixed-shape <code>State + Intent + episode history</code> samples. Episode-level cache shards were prefetched as needed, avoiding loading the full dataset into RAM and allowing reuse when rules and replay sets were unchanged.</p>
<p>Opening data was filtered in two stages. I first trained on several players who acquired all four plots by raw day 10, then fine-tuned on one strong player with a particularly consistent strategy. This reduced conflicting targets for similar states.</p>
<p>The initial broad BC run used about 5,700 games, batch size 128, ten epochs, and roughly 640,000 optimization steps. Separate Opening and Day 11+ models were then trained further on progressively narrower datasets.</p>
<h3>3.2 Not Teaching Invalid Actions as PASS</h3>
<p>Initially, invalid operations normalized to PASS or WAIT could become apparently correct labels. Because invalid operations increase late in a game, the model could learn a spurious association between late-game states and PASS.</p>
<p>PASS and WAIT were therefore retained only when explicitly present in the original action. A slot whose invalid action normalized to PASS or WAIT was excluded from policy loss, while other slots and value loss remained active.</p>
<h3>3.3 Joint Actor and Critic Training</h3>
<p>BC computed policy and value losses in one forward pass:</p>
<p>$$
L = L_{BC} + \lambda_v L_{value}
$$</p>
<p>Both losses updated the shared encoder. Policy heads learned from BC loss; the private encoder and value head learned from value loss. Completed-game outcomes were value targets, and incomplete games were excluded from value training.</p>
<h3>3.4 Player-Specific Fine-Tuning from PPO Checkpoints</h3>
<p>Mixing strong players unconditionally can create incompatible strategic targets for similar states. I initialized from intermediate PPO checkpoints, split the latest replays by player, and BC-fine-tuned several specialized models. These frozen models became PPO opponents, exposing the learner to consistent but diverse strategies.</p>
<h2>4. PPO</h2>
<h3>4.1 Opening State Bank</h3>
<p>Matches between Opening models generated complete game states at raw day 11, the start of displayed day 12. The bank contains both players' boards and assets plus market state. For fixed opponents, a manifest records which Opening model generated each state and which later-game model takes over, preventing mismatched pairings.</p>
<h3>4.2 Updates from Complete Games</h3>
<p>Each update collected two rounds of batch-size-32 rollouts, 64 games total. Games ran from day 12 through day 30, and GAE and PPO losses were computed from the collected transitions.</p>
<table>
<thead>
<tr>
<th>Setting</th>
<th>Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>rollout horizon</td>
<td>456</td>
</tr>
<tr>
<td>games / update</td>
<td>64</td>
</tr>
<tr>
<td>discount factor $\gamma$</td>
<td>0.999</td>
</tr>
<tr>
<td>GAE $\lambda$</td>
<td>1.0</td>
</tr>
<tr>
<td>PPO clip</td>
<td>0.2</td>
</tr>
<tr>
<td>learning rate</td>
<td>$1\times10^{-5}$</td>
</tr>
<tr>
<td>minibatch size</td>
<td>128</td>
</tr>
<tr>
<td>segment length</td>
<td>72</td>
</tr>
</tbody>
</table>
<p>Reward was primarily the terminal result, with no intermediate daily reward. I used $\gamma$ near one and GAE $\lambda=1.0$ to propagate the long-horizon signal.</p>
<h3>4.3 Opponent Diversity</h3>
<p>One rollout round sampled opponents from:</p>
<ul>
<li>self: the current learner</li>
<li>anchor: the fixed initial policy</li>
<li>checkpoint pool: previously promoted policies</li>
</ul>
<p>The probabilities were 25% self, 25% anchor, and 50% pool; before the pool was populated, its mass returned to anchor.</p>
<p>The other round used player-specialized BC opponents. The final run weighted a model close to the learner's strategy more heavily while retaining several others, including both four-plot and three-plot policies.</p>
<p>Early runs used only self and anchor. Later, models lightly BC-fine-tuned from the current PPO checkpoint on strong leaderboard-player data were added to broaden the training distribution.</p>
<h3>4.4 Anchor and Reference</h3>
<p>The two could point to the same checkpoint but served different roles:</p>
<ul>
<li><strong>anchor</strong>: fixed opponent and regression baseline</li>
<li><strong>reference</strong>: target of an L2 penalty preventing excessive Actor drift</li>
</ul>
<p>Anchor performance guarded against regression, while the reference penalty reduced abrupt collapse of the BC-initialized policy.</p>
<h3>4.5 Promotion to Best</h3>
<p>Every ten updates, the candidate was evaluated on an independent state bank from both player IDs: 64 games per side, 128 total. Promotion required:</p>
<ol>
<li>at least 55% wins against current best</li>
<li>at least 60% wins against anchor</li>
</ol>
<p>Promoted checkpoints replaced best and entered a pool capped at eight. Failed promotion did not stop learner updates, but best and the pool stayed unchanged.</p>
<h3>4.6 Training Scale</h3>
<p>The PPO sequence leading to the final policy covered 370 updates, 23,680 games, and about 10.8 million player-turns over roughly 22 hours.</p>
<p>Across all experiments, settings varied: 64–256 games per update and 456–720 turns per game. Including changes to opponents, starting states, and rewards, the total was about 1,200 updates, 100,000 games, and 50 million player-turns.</p>
<p>These totals combine different experiments rather than one policy trained continuously under one configuration. The final setup ran for only 370 updates. Given the long horizon and large action space, this was not enough to establish convergence. Much time was spent constructing suitable opponents and start-state distributions, so insufficient training under the final setup may have mattered as much as the PPO design.</p>
<h2>5. Deterministic Helper Layers</h2>
<h3>5.1 Land-Purchase Safeguard</h3>
<p>During the opening, if a purchase deadline had passed and enough operating cash would remain, a missing BUY_LAND order was inserted at the front of the market list. This was a narrow correction to model output, not a rule-based replacement for the opening. JAX state-bank generation and PyTorch submission inference used identical deadlines, cash requirements, and insertion positions.</p>
<h3>5.2 Endgame Handling</h3>
<p>Near the end, remaining actions determined when a unit had to return to the shed and DROP carried items. The final step submitted sellable inventory to market.</p>
<p>The layer did not replace all low-value movement with PASS. It acted only at the deadline beyond which return was impossible and at the final sale. Corrected slots were excluded from BC/PPO policy loss so the correction was not learned as the model's own choice.</p>
<h2>6. What Did Not Work and What I Learned</h2>
<h3>6.1 Fixed Opening Tapes</h3>
<p>Replaying recorded opening sequences was unreliable. BC was easier to implement and tune, and it behaved better, so the Opening model also used BC.</p>
<h3>6.2 Full-Game PPO</h3>
<p>Rolling out all 720 turns wasted compute on nearly deterministic opening actions and increased GPU memory pressure. Precomputing 264 opening turns improved speed and memory usage.</p>
<h3>6.3 Autoregressive Market Orders</h3>
<p>Generating orders one by one represented dependencies more directly but sharply reduced rollout throughput. Since PPO also needed many environment transitions, I returned to the parallel Actor and sequential Executor.</p>
<h3>6.4 Invalid Actions Collapsed into PASS</h3>
<p>Treating invalid demonstrations as PASS lowered loss while making the policy overproduce PASS. The original explicit-PASS status had to be preserved.</p>
<h3>6.5 Fixed PPO Opponents</h3>
<p>A single anchor or recent best encouraged specialization. Mixing self, anchor, historical checkpoints, and player-specialized BC models—and matching their starting states—broadened the distribution.</p>
<h3>6.6 Matching Training and Submission</h3>
<p>Differences in land purchasing, endgame handling, masks, or Executor behavior between JAX and PyTorch broke reproducibility despite good offline evaluation. The entire preprocessing and postprocessing path had to match, not only the weights.</p>
<h3>6.7 PPO Did Not Catch Up with Strong-Player BC</h3>
<p>I continued PPO until the deadline while repeatedly changing the opponent pool. In the final experiment, several candidates were BC-fine-tuned from PPO weights using newly available strong-player replays. The learner could not clearly beat one of them, so that model became the second final submission alongside the continuously trained PPO policy.</p>
<p>Other participants also reported strong results from BC on strong-player replays. Under my compute budget, opponent mixture, and training duration, PPO did not clearly surpass the last BC-fine-tuned model. Concentrating more compute on BC data selection and training may have been more effective.</p>
<h2>7. Final Submission</h2>
<p>I submitted the continuously updated PPO policy and the PPO-based policy fine-tuned on strong-player replays. The former remained around 40th–50th on the leaderboard; the latter finished 22nd and earned a gold medal.</p>
<p>Each submission used two PyTorch checkpoints. The Opening model handled raw days through day 10, and Day 11+ took over from raw day 11. Land-purchase and endgame corrections were applied when needed. The Critic was training-only; only Actor weights were exported, allowing CPU inference without private information.</p>
<h2>Conclusion</h2>
<p>In Kaggriculture, deciding what to learn, what to handle deterministically, and which match distribution to train from mattered more than merely enlarging the network.</p>
<p>The final design centered on three ideas:</p>
<ol>
<li>isolate the opening in its own model and state bank, concentrating PPO on the strategic phase</li>
<li>predict compound actions in parallel and resolve environment dependencies sequentially</li>
<li>train against self, historical checkpoints, fixed anchors, and BC models with different strategies</li>
</ol>
<p>Many choices were introduced incrementally under limited time and compute. Several experiments changed multiple components at once, and I could not run thorough ablations. I therefore cannot attribute precise gains to individual components. This is not a proven optimum, but the design reached through competition-time iteration.</p>
<p>The same ideas may transfer to other simulations with long horizons, compound discrete actions, and diverse opponents.</p>
<h2>Acknowledgments</h2>
<p>Thank you to Kaggle and the competition organizers, and to everyone who participated and played against my agents.</p>

---

