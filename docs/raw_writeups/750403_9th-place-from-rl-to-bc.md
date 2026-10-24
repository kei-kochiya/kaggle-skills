# Topic 750403: 9th: From Reinforcement Learning to a Final BC Agent

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/750403
- **Author**: 
- **Post Date**: 2026-10-18T06:26:34.050000
- **Votes**: 8

## Messages

### Message 1 by  (Votes: 8)

<p>First, I'd like to thank my teammates, <strong>tetsuya</strong> and <strong>guoqin gu</strong>. It was a pleasure taking part in this competition together. Thanks also to the hosts and everyone who shared strategies and experiences in the discussions.</p>
<h1>Overview</h1>
<p>Our final agent is an <strong>18.4M-parameter Transformer + CNN policy</strong> trained by behavior cloning on strong game replays. It directly predicts worker commands and market orders, using EMA weights and single-model argmax decoding.</p>
<table>
<thead>
<tr>
<th>Final agent</th>
<th>At a glance</th>
</tr>
</thead>
<tbody>
<tr>
<td>Policy</td>
<td>18.4M parameters · Transformer + farm CNN</td>
</tr>
<tr>
<td>Learning</td>
<td>Behavior cloning · EMA weights</td>
</tr>
<tr>
<td>Final replay mixture</td>
<td>70% MMPQ · 30% other sources</td>
</tr>
<tr>
<td>Inference</td>
<td>Single model · FP32 · argmax · CPU AOT</td>
</tr>
</tbody>
</table>
<p>Development began with RL and later shifted toward BC as the replay pipeline and model improved. The sections below describe the training setup, the development timeline and the final agent.</p>
<h1>Training pipeline</h1>
<p>The final submission uses behavior cloning with normalized replay targets, weighted source sampling and auxiliary objectives. A later PPO branch starts from a BC actor and continues training through interaction with opponents.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4578277%2Fa380982c9ad5e3395e77b874adafa616%2F01-training-pipeline.png?generation=1792304458543618&amp;alt=media" alt="Training and final submission"></p>
<p><em>Figure 1. The final BC submission pipeline and our later direct-action PPO branch. Our early RL exploration preceded this final setup.</em></p>
<p>Three design choices were central to the final agent:</p>
<ul>
<li><p><strong>Requested actions and executed actions can differ.</strong> We reconstructed execution effects before building BC targets.</p></li>
<li><p><strong>Decisions within a turn depend on one another.</strong> Workers and market orders are decoded in execution order, with later decisions seeing the resource changes caused by earlier ones.</p></li>
<li><p><strong>A single observation does not describe the market process.</strong> We added history statistics at multiple timescales, together with economic and forecasting objectives.</p></li>
</ul>
<h2>Behavior cloning: replay data and action labels</h2>
<p>Competition replays are converted into player-side trajectories containing observations, history and actions. The final fine-tuning corpus contains 4,996 trajectories from 38 submissions, or approximately 3.59 million action time steps.</p>
<p><strong>Replay requests can differ from executed actions.</strong> A worker may request more items than remain available, a market order may receive no fill, or a command may have no effect. Cloning these requests directly can leave the action targets inconsistent with subsequent state changes.</p>
<p>Our preprocessing reconstructs execution: ineffective worker commands become PASS, quantities are corrected, unfilled market orders become NOOP, and sales can use SELL ALL. BC imitates this normalized command sequence.</p>
<h2>Mixing replay sources</h2>
<p>BC starts with pretraining on high-scoring replays, followed by adjustments to replay sources and sampling weights. The final branch continues from an earlier BC checkpoint with a 70:30 mixture of MMPQ and other replays.</p>
<p>The corpus contains 1,165 MMPQ trajectories and 3,831 from other sources. The 70% sampling share deliberately emphasizes a relatively consistent group of strategies while retaining behavior from other sources.</p>
<h2>Training objectives</h2>
<p>Action imitation is combined with economic and forecasting objectives. Economic targets cover labor, land, cash, cash margin, stock and field service. Forecast targets describe subsequent market and economic changes in the replay, giving the shared encoder supervision beyond the next action.</p>
<p>The total loss is:</p>
<pre><code>L = 2 * worker NLL + market NLL
    + economic loss + 0.05 * forecast loss
</code></pre>
<p>Future replay frames are used to construct targets only. During play, the policy consumes current legal observations and the history maintained so far.</p>
<p>Compressed trajectory pools and rotating sampling keep random reads and memory use manageable.</p>
<h2>Reinforcement learning: BC initialization and PPO</h2>
<p>The later direct-action RL implementation initializes the actor from BC and uses PPO to continue learning against a mixture of opponents. This branch is distinct from the early RL experiments that motivated the initial approach.</p>
<p>A vectorized JAX simulator collects on-policy data, with 512 games per rollout wave across four GPUs. Early experiments mainly used self-play. The later opponent mixture has five approximately equal groups: self-play and four groups of frozen neural opponents.</p>
<p>The later reward emphasizes terminal win/loss and adds a cash-margin signal. With margin m and s(m)=tanh(m/100000):</p>
<pre><code>r = 0.25 * (s(m_next) - s(m_now))
    + terminal * (sign(m_next) + 0.25 * s(m_next))
</code></pre>
<p>PPO uses gamma 0.995, GAE lambda 0.95, ratio clipping to [0.8, 1.2] and normalized advantages. The actor learning rate is 1e-6; the independent critic uses 3e-4. Policy sampling and PPO likelihood evaluation both use temperature 0.5.</p>
<p>Two parts received particular attention. The first was <strong>critic initialization</strong>: after changing the reward, we held the actor fixed and fitted returns from complete games, holding out whole games for validation. The second was <strong>controlling policy movement</strong>: after each proposed update, we checked KL on both the update minibatch and a fixed reference minibatch from that rollout, reverting proposals that exceeded the limits.</p>
<h1>Training history</h1>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4578277%2Fb12e5348574f11e23200e55425afde2f%2F04-competition-timeline.png?generation=1792304533221100&amp;alt=media" alt="Competition timeline"></p>
<p><em>Figure 2. Four milestones in our competition. The final BC model came from a continuation run in the last two or three hours.</em></p>
<h2>RL exploration and scaling</h2>
<p>Early experiments established that RL was viable in this environment. Once the training pipeline was working, further development focused on improving the policy and scaling training.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4578277%2F9279f19009846c68ae559c1f421a1185%2F05-cartoon-rl-exploration.png?generation=1792304553142539&amp;alt=media" alt="Exploring reinforcement learning on the farm"></p>
<p><em>Learning through trial and error. A conceptual illustration of our early RL exploration.</em></p>
<p>Scaling produced substantial improvements, but issues in the JAX implementation and model design slowed iteration. Limited experience with these parts of the system contributed to longer debugging cycles and made scaling less effective than expected.</p>
<h2>Shift toward BC</h2>
<p>Strong BC agents from other teams prompted a renewed focus on imitation learning during the final week. The last three days of development concentrated on correcting weaknesses in the BC training pipeline, followed by changes to the policy model.</p>
<p>In local testing, the revised BC models substantially outperformed the RL models available at that stage. Further RL training was also tested, but improvements arrived more slowly, so the remaining development focused on BC.</p>
<p>A further BC continuation ran during the final two or three hours before the deadline. Local testing identified its step197129 EMA checkpoint as the strongest candidate available at that point, and it was selected for submission.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4578277%2F9625ee0eac04047b37530bde1396dae9%2F06-cartoon-final-bc.png?generation=1792304567545760&amp;alt=media" alt="One final late-night behavior-cloning run"></p>
<p><em>Learning from replays, with one last experiment near the deadline. A conceptual illustration of the final BC run.</em></p>
<h1>Architecture and inference</h1>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4578277%2F058cc7db907b61fa16651a37e6a44cce%2F02-policy-architecture.png?generation=1792304502912275&amp;alt=media" alt="Policy architecture"></p>
<p><em>Figure 3. Main policy pathways. History and spatial features condition the entity representation; the resource ledger connects worker and market decoding.</em></p>
<h2>Entity Transformer and farm CNN</h2>
<p>The state is represented as entity tokens: a global state, workers, tiles, shops, commodities and project features. Attention connects these entities, including relationships between workers and tiles, production and inventory, and the player's farm and the shared market.</p>
<p>The backbone has two interaction rounds. Each round processes worker and field tokens in separate branches before shared attention exchanges information. A residual 5×5 CNN additionally processes our own 10×10 farm. The opponent's public farm participates through entity features.</p>
<table>
<thead>
<tr>
<th>Component</th>
<th>Final configuration</th>
</tr>
</thead>
<tbody>
<tr>
<td>Parameters</td>
<td>18,425,190</td>
</tr>
<tr>
<td>Transformer width</td>
<td>288</td>
</tr>
<tr>
<td>Feed-forward width</td>
<td>224</td>
</tr>
<tr>
<td>Attention heads</td>
<td>4</td>
</tr>
<tr>
<td>Entity-branch and shared-attention rounds</td>
<td>2</td>
</tr>
<tr>
<td>Own-farm CNN</td>
<td>Two 5×5 convolutions, 63→224→960 channels, projected to width 288</td>
</tr>
</tbody>
</table>
<p>History is represented by rolling means and lagged differences, with windows of 6, 24, 72, 168, 360 and 719 turns. These statistics can be updated incrementally during play and cached during preprocessing, allowing BC to sample random time steps without unrolling an entire game.</p>
<h2>Sequential action decoding</h2>
<p>Worker actions are factorized into direction, local operation, argument and quantity. During autonomous play, commands are generated in worker execution order, updating the resource ledger after each one. Items taken by an earlier worker therefore change the inventory context available to the next worker.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4578277%2F042d22630cbdc94601d124f301d994f1%2F07-cartoon-sequential-actions.png?generation=1792304581111238&amp;alt=media" alt="Workers share a changing inventory"></p>
<p><em>Earlier actions change what remains available. A conceptual illustration of the resource ledger used in sequential decoding.</em></p>
<p>During BC training, causal prefixes are precomputed from demonstrated actions. Worker losses can then be evaluated in parallel while preserving within-turn dependencies.</p>
<p>The market decoder similarly generates up to ten ordered slots, conditioning each one on the prefix updated by earlier orders. Quantities support numeric values and SELL ALL. The final policy emits commands directly, without an additional state-dependent legality mask.</p>
<h2>CPU deployment</h2>
<p>Inference uses one model, FP32 weights and conditional-head argmax. To fit the competition's CPU runtime, we compiled the initial forward pass, subsequent forward pass and history update ahead of time for Broadwell CPUs.</p>
<p>We also fixed persistent inventory insertion order in the deployment adapter. This affects which items survive DROP when shed capacity is insufficient, and therefore changes later states. We maintain the order from consecutive observations so deployment follows the execution semantics used in training. The final package passed a complete-season raw-source execution check.</p>
<h1>Evaluation and checkpoint selection</h1>
<p>Local evaluation uses fixed historical BC agents as opponents. Each comparison contains 512 seeds, with both seats played on every seed, for 1,024 games per pairing. Deterministic argmax matches submission-time decoding.</p>
<p>The final step197129 EMA achieved the following results. Draws are not counted as wins, and cash margin is our terminal cash minus the opponent's.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4578277%2Fe36aedac60433fda91e2427739017d1a%2F03-offline-evaluation.png?generation=1792304518599267&amp;alt=media" alt="Offline evaluation"></p>
<p><em>Figure 4. Strict win rates and mean terminal cash margins against five historical BC opponents. Each pairing contains 1,024 games with both seats on every seed.</em></p>
<table>
<thead>
<tr>
<th>Opponent</th>
<th>W / L / D</th>
<th>Win rate</th>
<th>Mean cash margin</th>
</tr>
</thead>
<tbody>
<tr>
<td>Earlier BC 0.40</td>
<td>944 / 80 / 0</td>
<td>92.19%</td>
<td>+4,935</td>
</tr>
<tr>
<td>Split2 BC 0.45</td>
<td>900 / 124 / 0</td>
<td>87.89%</td>
<td>+3,770</td>
</tr>
<tr>
<td>Double BC 0.25</td>
<td>876 / 148 / 0</td>
<td>85.55%</td>
<td>+3,623</td>
</tr>
<tr>
<td>Stage14 BC 0.25</td>
<td>861 / 163 / 0</td>
<td>84.08%</td>
<td>+3,338</td>
</tr>
<tr>
<td>37.3M BC step41000</td>
<td>781 / 243 / 0</td>
<td>76.27%</td>
<td>+2,301</td>
</tr>
</tbody>
</table>
<p>These results are from our local JAX simulator against the fixed historical panel used for checkpoint comparison.</p>
<h2>Final checkpoint selection</h2>
<p>Our final checkpoint is the stage 3 EMA at step197129, saved when the mean of the last twenty raw-model total training losses reached approximately 0.50. Training later continued to step202752, but the later checkpoint did not improve every matchup. Across four common opponents, the two checkpoints won 3,581 and 3,579 games respectively, with different strengths by opponent.</p>
<p>Checkpoint selection considered complete-game performance alongside BC loss. The loss measures fit to demonstrations; local matches test behavior over a full season. These comparisons do not isolate the effects of model size, training data or optimization.</p>
<p>Submission remained uncertain: a comparable model had not yet been tested on the leaderboard, and an earlier submission had failed to generalize due to overfitting. This checkpoint was submitted once and became the team's highest-scoring submission.</p>
<h1>Discussion and future work</h1>
<p><strong>Iteration speed.</strong> Scaling improved the models, but implementation and architecture issues limited experiment throughput. Faster validation and more reliable training code would have made successive changes easier to assess.</p>
<p><strong>Data quality starts with execution semantics.</strong> In a sequential environment, action names and requested quantities do not fully describe a sample. Within-turn resource changes, partial fills and ineffective commands all belong in the data pipeline.</p>
<p><strong>Replay distribution and the BC-to-RL transition.</strong> Further work would compare teacher consistency, behavioral diversity and sampling weights, then examine critic pretraining, opponent selection and policy-update stability under matched training and evaluation conditions.</p>
<h1>Acknowledgements</h1>
<p>Thanks again to <strong>tetsuya</strong> and <strong>guoqin gu</strong> for sharing this competition journey. Thanks to the hosts for the environment and replay access, and to the competitors who shared strategies, code and discussions. Strong replays provided the foundation for our final BC agent, and the community's work helped us throughout the competition.</p>

---

### Message 2 by  (Votes: 1)

<p>Did you track PLANT batches rejected for exceeding a crop's seed stock during autonomous games? I'm curious how often that happened with the sequential ledger and no extra legality mask.</p>

---

### Message 3 by  (Votes: 0)

<p>Thanks for asking! We didn’t separately track the frequency of seed-oversubscription cancellations. The sequential ledger provided updated resource context to later workers, but the final no-mask policy did not enforce a hard seed budget. After generating all worker commands, we applied crop-wide cancellation when reconstructing the state for market decoding, without modifying the worker commands. So the ledger accounted for the consequences, but didn’t guarantee that oversubscription wouldn’t occur.</p>

---

### Message 4 by  (Votes: 1)

<p>Thanks for your sharing！</p>

---

