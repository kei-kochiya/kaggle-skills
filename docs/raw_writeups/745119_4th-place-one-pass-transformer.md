# Topic 745119: [4th Place]  No RL, no search: a one-forward-pass transformer that learned from what the models actually did

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/745119
- **Author**: 
- **Post Date**: 2026-10-02T10:31:05.261000
- **Votes**: 35

## Messages

### Message 1 by  (Votes: 35)

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F21974454%2F5ed8446fbd148245ccf03cf55923ae2d%2Fintro.gif?generation=1790936677594603&amp;alt=media" alt="Our competition, in one tractor ride"></p>
<p>UPD: Writeup was uploaded before final BT evaluation. So I update it and you can check a more current version here:
<a href="https://www.kaggle.com/competitions/kaggriculture/writeups/farmcore-a-one-pass-transformer-trained-on-what-t" target="_blank">https://www.kaggle.com/competitions/kaggriculture/writeups/farmcore-a-one-pass-transformer-trained-on-what-t</a></p>
<p>Summary. Our final agent is pure behavioral cloning, with no reinforcement learning and no test-time search. A transformer drafts the whole turn in a single forward pass, and a six-layer corrector then lets the units' and the market's decisions see each other and revise. Most of our performance came from three choices:</p>
<ol>
<li>Training on the state changes the engine actually committed rather than on raw player actions, because many strong players submit oversized market orders that the engine clips silently.</li>
<li>Pretraining on many top ladder submissions, fine-tuning on a single strong team, and averaging the weights of two such fine-tunes.</li>
<li>Selecting checkpoints only by money on identical seeds, because validation accuracy correlated negatively with how well a model played.</li>
</ol>
<p>Training ran on two A100 GPUs plus free cloud GPU notebooks.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F21974454%2F4f8a2b9e409a3a19119515decafa843f%2Ftimeline.png?generation=1790936715934687&amp;alt=media" alt="Rating of every team submission"></p>
<p>Metric conventions. A strong agent finishes a game with roughly $100k. The engine is deterministic, so we compare a candidate with a baseline by playing both on the same seeds (typically 48). "+$X" is the average difference in final money per game over those paired seeds, and "44/48" means the candidate finished ahead on 44 of the 48 seeds. <em>Margin</em> is our money minus the opponent's.</p>
<h2>Implementation differences</h2>
<p>Most high-ranking submissions rely on a common set of techniques: a bit-exact engine port, paired-seed evaluation, behavioral cloning followed by large-scale PPO self-play, and heuristic overrides. We used similar foundations, but our pipeline diverged in four places:</p>
<table>
<thead>
<tr>
<th>Area</th>
<th>Common approach</th>
<th>Our pipeline</th>
</tr>
</thead>
<tbody>
<tr>
<td>Imitation target</td>
<td>Recorded player actions</td>
<td>State changes the engine actually applied</td>
</tr>
<tr>
<td>Turn generation</td>
<td>Per-unit logits or autoregressive decoding</td>
<td>A draft of the whole turn, then a corrector over all decisions</td>
</tr>
<tr>
<td>After cloning</td>
<td>Large-scale PPO self-play</td>
<td>Broad pretraining, single-team fine-tuning, weight averaging</td>
</tr>
<tr>
<td>Checkpoint selection</td>
<td>Validation metrics or local win rate</td>
<td>Money on paired seeds</td>
</tr>
</tbody>
</table>
<h2>Training on what the engine executed, not on what was requested</h2>
<p>The engine executes market orders one unit at a time. When an order tries to sell from an empty shed or buy beyond the available money, it stops at the first unit that fails, and the rest of that order is silently dropped.</p>
<p>Many top players place oversized orders and let the engine trim them. When we replayed games of 15 top ladder players through our local engine, we found that only part of what they requested ever executed:</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F21974454%2F41e312e855a132abced03d16bdb43d88%2Fordered_vs_executed.png?generation=1790936747907134&amp;alt=media" alt="Requested vs executed market orders per teacher"></p>
<p>One player had only 30% of its requested sells filled and two others about 2%. Our first teacher never executed 87% of its requested goods purchases. Trained directly on recorded actions, a policy learns to emit these empty orders, which are then clipped at play time by the shed, the budget and the decoder. It is trained in one world and plays in another.</p>
<p>To fix this, we replay every game through the engine with a bit-exact money check at each step, keep only the trades that executed, and net out round trips (selling an item and buying it back in the same turn). All input features, such as our own recent sales, are computed from the same replayed record as the labels. With the cleaned labels, a clone that won 3 of 24 games against our previous best network won 12 of 24. With the feed guard below added, it won 23 of 24.</p>
<h3>Closing gaps in the corpus with rules from the game mechanics</h3>
<p>A clone eventually reaches states its teacher never visited. Our teacher always fed its animals its own wheat, so an empty shed with a hungry herd never appeared in its games. In that state the clone would stall, and animals left unfed for two days run away. We closed such gaps with deterministic rules that follow from the engine's mechanics:</p>
<ul>
<li>Feed guard: each turn, buy exactly enough wheat for the animals not yet fed today.</li>
<li>Night-shed sell: at night, everything the units carry is dumped into the shed and whatever does not fit is destroyed. On the last turn of each day we simulate that night and sell what would be lost (+$0.6k).</li>
<li>End sell: sell the whole shed on the final turn.</li>
</ul>
<p>Six other "common sense" rules, including building more coops, capping wheat purchases and spreading units over tiles, were neutral or negative. Even stopping units from requesting more than the shed holds lost money: the engine turns such a request into a harmless no-op, whereas the blocked unit simply wasted its turn. The engine's silent no-op is often cheaper than enforced legality.</p>
<h3>Finding spurious correlations by watching games</h3>
<p>A clone of one planner-style teacher had near-perfect validation accuracy (0.937 on unit targets) yet lost badly in play (−$86k against our reference). Watching individual games showed that on day 6 the teacher buys new land while the clone does not. By swapping the teacher's input feature groups into the clone's state one at a time, we traced the decision to a single feature: wheat in the shed per animal.</p>
<p>The network had learned "buy land when wheat per animal is low". On 30% of training rows we shifted the shed's wheat by a random amount, recomputed every feature where that stock is visible, and kept the loss only on the heads that legitimately depend on stock. That broke the correlation: +$45k for that teacher and a small gain for our main one.</p>
<h2>Staged training: pretraining, targeted fine-tuning, and weight soups</h2>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F21974454%2F70796cfc8621ce8e9117390b09725e44%2Fpipeline.png?generation=1790936764218995&amp;alt=media" alt="Training pipeline"></p>
<p>Training on a pooled corpus of many top players was worse than cloning a single team. A clone matched its own teacher on 88% of unit actions, whereas independent top teams agreed with each other on only 66–68% of decisions. Our explanation is that a pooled model takes neighbouring decisions from different players, so its plans stop fitting together. Cloning the top-ranked ladder player also underperformed: reactive players cloned cleanly, whereas long-horizon planners did not translate well into turn-by-turn imitation.</p>
<p>Separating the training into phases worked:</p>
<ul>
<li>Pretraining on the relabeled games of many top teams.</li>
<li>Fine-tuning on a single strong team. This gained +$9.2k and won 89 of 96 games against our previous model, clearly ahead of a network trained on that team's games from scratch.</li>
<li>Averaging the weights of two fine-tunes that start from the same pretrained model but use different submissions of that team. This soup gained +$3.3k and won 46/48 against its parent, and held up across a panel of other opponents.</li>
</ul>
<p>Other soups failed. Averaging two random seeds of the same recipe gave no measurable benefit, and averaging fine-tunes on different teams made play worse with each extra team. The two parents agree on roughly 80% of unit actions, and the remaining 20% are mostly near-ties. Because both parents start from the same point, averaging their weights behaves much like ensembling their logits, at no extra inference cost.</p>
<h2>Single-pass architecture with decision correction</h2>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F21974454%2F2945e13a2e3bd55d94c836cae5afefef%2Farchitecture.png?generation=1790936780547705&amp;alt=media" alt="Architecture"></p>
<p>We designed the turn this way from the start. A turn is a plan, and a plan is easiest to fix when all of it is visible at once. So the network first drafts every decision of the turn in one pass, and then a self-attention corrector lets those decisions see each other and revise.</p>
<p>The policy has 21.5M parameters, of which about 16M run at play time; the rest is a value tower used only in training. Inference runs in plain FP32 PyTorch on two CPU threads.</p>
<p>The input has one token per own tile (100), per unit (up to 24), per product, crop and animal, plus a global token. The opponent's farm passes through a separate encoder and a four-latent bottleneck, so that it cannot dominate attention over our own tokens.</p>
<p>The trunk has nine transformer blocks. Each attention head has a learned 19×19 relative-position table indexed by the spatial offset (dx, dy), shared across all trunk layers. Removing this bias lost in paired-seed games.</p>
<p>Draft heads give each unit a pointer to its target tile and an action. Market heads produce an open/closed gate per item and order type, a quantity, hiring and land purchases. Each head reads the state, but none of them sees what the others decided. Unit 3 doesn't know that unit 7 is heading for the same melon, and the sell head doesn't know how much the units will drop into the shed this turn.</p>
<p>Movement uses an exact shortest-path rule instead of learned steps. Every step of the teacher we measured was a shortest-path move toward the tile of the unit's next job. So the network predicts the target tile, and the rule takes the steps.</p>
<p>The corrector is where the decisions meet:</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F21974454%2Fadcadc4c07a72ae2ed69020eafb34350%2Fturn_in_one_pass.png?generation=1790936816791832&amp;alt=media" alt="One pass vs autoregressive"></p>
<p>Each unit's draft target distribution is applied to the tile keys to build a probability-weighted tile, the unit's intent. When two units aim at the same tile, their intents align, and attention sees the conflict as a dot product. Units also cross-attend to our own tiles to see what lies on a contested target. Market tokens receive the expected sells and buys and the shed as it will be at market time, after this turn's pickups and drop-offs. The global token receives the expected spend and revenue, computed explicitly rather than learned. Six transformer layers then run over these 42 decision tokens, and the same heads produce the final decisions.</p>
<p>The corrector reads the network's own soft draft, not ground truth, and the whole stack trains end to end with the loss only on the final decisions, so the draft is trained only through the corrector. An earlier variant fed the true plan back in. It learned to trust that plan, and lost in live play, where the plan is the network's own mistakes. Soft distributions also carry each unit's confidence, so a unit that is 99% sure can keep its target over one torn 51/49.</p>
<p>This design also turned out to be cheap. Generating the turn autoregressively would need roughly one pass per decision (15–25 per turn). We estimated that at 486–811 ms from a 32 ms single pass of an earlier, smaller model on a laptop, against a 1 s limit on a slower Kaggle CPU. Apart from the first turn, which loads the package, no ladder turn of ours took more than 1 s. And when we broke down the clone's errors, the kind autoregression would fix (the right action given to the wrong unit) accounted for only about a fifth of them.</p>
<p>The corrector is critical to performance: it changes the target of 25% of units and the market orders in 74% of turns. Playing the draft alone drops final money from ~$100k to ~$14k, and the agent loses all 48 paired games.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F21974454%2F85515640efe2764d6f261cbbdf597fd3%2Fcorrector_turn.png?generation=1790936830401979&amp;alt=media" alt="One turn before and after the corrector"></p>
<p>In this turn the corrector mostly removes waste. It can also change which jobs get done, as in this turn later in the season:</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F21974454%2F75e7d9c1c346ba4b6ea4df56a8681663%2Fcorrector_turn2.png?generation=1790936841819641&amp;alt=media" alt="The corrector re-routes workers"></p>
<p>Averaged over all turns, the same pattern shows up in the market gates:</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F21974454%2F859733bca5d5e5ed4e907cbf6094e413%2Fcorrector_gates.png?generation=1790936854801096&amp;alt=media" alt="Market gates: draft vs final"></p>
<p>Melon is the main exception: the corrector opens its sell gate exactly when a unit is carrying melons to the shed on that turn.</p>
<p>We also apply logit adjustment at play time, subtracting the training class priors with β = 1. On the same weights this gave +$3.2k and won 44/48 against unadjusted logits.</p>
<h2>Policy analysis and emergent behaviors</h2>
<p>Shuffling one input group across thousands of live states of the final network, and measuring the margin lost per game, shows what the policy relies on:</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F21974454%2Fafa919e9bff4f3018dffeeff00763754%2Finput_ablation.png?generation=1790936876690503&amp;alt=media" alt="Margin lost when an input group is shuffled"></p>
<p>What the units carry and what grows on our own tiles matter most. Shuffling the opponent's board costs only a few thousand dollars of margin, and the money value inside the global token is barely read.</p>
<p>The policy discovered the town's four-turn shop cycle on its own. Strawberries, milk and wool are sold almost only on shop turns (sell probability 0.4–0.7, against under 0.1 on other turns), and sales rise with price and with the stock in the shed:</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F21974454%2Fbe05a02eb4301da0e2da121199bc714c%2Fsell_shop_cycle.png?generation=1790936906344515&amp;alt=media" alt="Sell-gate response to one substituted input"></p>
<p>Each attention head took on its own spatial role, and because the relative-position table is shared across all nine layers, a head keeps that role in every layer:</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F21974454%2Fed1143173ada7a48676689b2cbb8b590%2Frelbias_heads.png?generation=1790936803889865&amp;alt=media" alt="Relative-position bias per head"></p>
<p>Individual heads specialize in the unit's own tile, its neighbours along x or y, the diagonals, and a ring around the unit.</p>
<p>The opponent representation collapsed almost entirely into one signal: the four opponent latent vectors have a cosine similarity of 0.993. The policy tracks the opponent mainly through the shared market rather than through the opponent's board. The main exception is livestock: buying cows and sheep also takes milk and wool demand away from the opponent, because that demand is finite and shared. Disabling the corrector for livestock purchases cost $29k of margin, of which $26k went to the opponent.</p>
<h2>Evaluation and training dynamics</h2>
<ul>
<li>Validation accuracy correlated negatively with money: a rank correlation of −0.86 across eight versions trained with the same protocol. Our best-ever validation came from a data leak, a legality mask computed at the destination tile rather than at the unit's current position, and that checkpoint went broke in live games. We select checkpoints strictly by paired-seed money.</li>
<li>Matching the teacher more closely did not necessarily make the policy stronger. Several changes improved imitation accuracy without moving money.</li>
<li>Policy strength was non-transitive. Some configurations gained +$8k against one baseline but lost against others. In the final week we evaluated every candidate against a panel of 5–7 opponents.</li>
<li>Train/play mismatches caused our most severe bugs, for example computing the shed before versus after the market resolves. They never raised an exception or showed up in validation, and were caught only by step-by-step comparison with the engine.</li>
</ul>
<h2>Reinforcement learning experiments</h2>
<p>We ran PPO on top of our imitation checkpoints, with the change in farm net worth as the reward. It gave small improvements over the starting checkpoints, as did our teammate's independent self-play runs of ~100k games each: a few thousand dollars per game. Collecting those rollouts took days of GPU time, whereas relabeling the data with executed trades gave far larger gains in an afternoon. With our compute, we spent the budget on the data instead, and both final submissions are pure imitation.</p>
<h2>Unsuccessful approaches</h2>
<ul>
<li>Training on mixed multi-teacher corpora and conditioning on teacher style embeddings: weaker than cloning a single consistent teacher.</li>
<li>Routing between expert models per decision head: gains against one opponent, losses against others.</li>
<li>Feeding each unit its previous action as input: the clone learned to copy itself, costing −$25k for one teacher.</li>
<li>Putting all board tiles into the corrector: worse, because attention spread away from the units.</li>
<li>Sampling actions instead of argmax: money within noise, fewer wins.</li>
<li>Iterative multi-pass draft refinement: no measurable improvement over a single corrector step.</li>
</ul>
<p>Thanks to the organizers for a genuinely interesting game, and to the teams at the top of the ladder: every one of our models learned by watching your games.</p>

---

