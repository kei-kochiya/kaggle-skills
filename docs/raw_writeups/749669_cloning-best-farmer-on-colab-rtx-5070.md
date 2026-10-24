# Topic 749669: Cloning the best farmer on Colab

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/749669
- **Author**: 
- **Post Date**: 2026-10-15T18:26:10.140000
- **Votes**: 9

## Messages

### Message 1 by  (Votes: 9)

<p>Thanks to Kaggle and the hosts for Kaggriculture. The game looks simple, but the shared market makes it hard. And thanks to the top teams, especially <strong>DSM</strong>, <strong>M &amp; M &amp; P &amp; Q</strong> and <strong>Victor @ Tufa Labs</strong>. Their ladder games were our training data, so this solution is partly theirs.</p>
<p><strong>TL;DR.</strong> For three weeks we built rule-based agents, planners and RL sale-timing policies, and they plateaued around 2,550-2,800. In the last four days we switched to <strong>closed-loop behavioural cloning (BC)</strong>: per-unit imitation of the top teams' ladder games, iterated on a single <strong>RTX 5070 (12 GB)</strong>. The CNN generations and the final fine-tunes ran on the 5070; the final ViT-style trunk was pre-trained on a Colab A100. The final agent is a hybrid ViT clone fine-tuned on DSM's tomato-always versions, with a small tomato top-up wrapper. Its final ladder score was <strong>2807.4</strong>.</p>
<h2><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F22592122%2Fe662f30d27235e0019e96928533cf1ac%2F01_pipeline.png?generation=1792087864388614&amp;alt=media" alt=""></h2>
<h2>1. Weeks 1-3: hand-built agents (subs 1-66)</h2>
<p>Briefly, because most of it was beaten later:</p>
<ul>
<li><strong>Rules plus overlays.</strong> We started from a public V37-lineage agent and layered on sale-timing, feed/fertilizer, opening guards and terminal planners.</li>
<li><strong>An exact C++ port of our agent.</strong> It passed two gates (zero action mismatches over 23 Python games) and runs <strong>817 games/s</strong> on 12 CPU threads, against about 0.01 games/s per core in Python. With it we trained a <strong>REINFORCE sale-horizon table over 4.3M games</strong>. That made sub 22/23, which reached 2801 on the mid-September ladder.</li>
<li><strong>Route planners and VRP crews.</strong> These included a closed-loop farm simulator with vehicle routing, route bandits and ridge-regression route tables.</li>
</ul>
<p>What we learned from those weeks:</p>
<ol>
<li><strong>The route (what you grow, and when you buy land and animals) is worth far more than the market layer.</strong> We tested eleven market-layer levers, and every one was refuted.</li>
<li><strong>Replaying tapes is biased.</strong> A recorded rival does not react to you, so a tape evaluation is only valid when the agent you substitute plays like the one the tape was recorded against.</li>
<li><strong>Mirror games are decided by who sells first.</strong> Any fixed sale horizon loses once it is copied. A reactive lead detector, and then the RL table, beat every fixed horizon.</li>
<li><strong>We could not hand-build what the top teams did.</strong> An exact cash audit showed DSM about <strong>+$8.9k per game</strong> ahead of us after controlling for shop luck. We tried tape executors, DSM-plan executors and a reactor on our planner, and none of them closed the gap.</li>
</ol>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F22592122%2F5404fcd3d6eee1a3208c3c99d8b5f6a5%2F06_ladder_history.png?generation=1792087904910593&amp;alt=media" alt=""></p>
<h2>2. The switch: per-unit closed-loop BC</h2>
<p>Earlier "clones" replayed open-loop tapes of recorded games, so execution kept breaking. We made the policy itself the clone: <strong>every turn, every unit (farmer and each hand) picks an action from a network that sees the current state.</strong></p>
<h3>Data</h3>
<ul>
<li>We pulled top-team ladder replays through the Kaggle API, each with its seed and both seats' actions.</li>
<li>A <strong>seat-game</strong> is one player's side of one game, so a game has two. We trained only on the seats played by the teams we were cloning. The opponent's farm is part of the observed state, but its actions are never training targets.</li>
<li>Every game is <strong>re-simulated exactly</strong> in our fastsim (Python, later the C++ bridge). A game is kept only if the replayed money matches the recorded money (1644/1644 seat-games for the first DSM set).</li>
<li>The data grew from <strong>1.6k DSM seat-games</strong> (v1) to <strong>10.3k seat-games from the top ~30 teams</strong> (data6), which is about 4.5M turns in the "latest agents" subset.</li>
</ul>
<h3>Model</h3>
<ul>
<li><strong>v1-v4 (CNN):</strong> a residual CNN over the 10×10 farm. Tile classes are embedded (ours and the opponent's), plus 12 numeric channels and a global vector: hour, day, cash, shed, seeds, prices, market inventory, shops, a team/style slot and carried goods.<ul>
<li><strong>Unit head:</strong> the trunk feature at the unit's tile, the pooled trunk, the globals, the unit index and its inventory → 47 action tokens.</li>
<li><strong>Market head:</strong> 21 order keys × 18 count buckets. Each key (e.g. <code>SELL MILK</code>, <code>BUY_SEED WHEAT</code>, <code>HIRE</code>) is its own 18-way classification over order sizes, where bucket 0 means "no order". We use classification rather than regression because order sizes are bursty, and most turns place no order at all.</li>
<li><strong>Loss:</strong> cross-entropy on the unit actions plus cross-entropy on the market buckets (averaged over the 21 keys), with label smoothing 0.02.</li></ul></li>
<li><strong>ViT trunk:</strong> a transformer over the 100 tile tokens.</li>
<li><strong>worker_hybrid (final trunk):</strong> two local convolutions, then attention over tile tokens plus explicit <strong>worker tokens</strong> (position, identity, inventory). Width 256, 6 layers.</li>
</ul>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F22592122%2F7a302f5fca4f307654f7deaf1588f5df%2F09_model_architecture.png?generation=1792087920890003&amp;alt=media" alt=""></p>
<ul>
<li><strong>Decoding is parallel, not autoregressive.</strong> The trunk runs once per turn, and each unit's head reads that shared trunk with its own position, index and inventory. All units decode independently in that one pass, so no unit sees what another unit chose this turn. Two units can therefore claim the same harvest or feed (see section 6 for why we left that alone).</li>
<li><strong>Inference:</strong> a plain numpy forward pass with fp16 weights base64-encoded into a single <code>main.py</code> (about 5.6 MB). Parity with PyTorch is about 1e-6. A turn takes 20 ms for the CNN and 73 ms for the ViT locally, against the 1 s actTimeout.</li>
</ul>
<h3>The decoder mattered more than the network</h3>
<p>The first trained build finished the season on <strong>$0</strong>. The network was fine; the wrapper around it was not:</p>
<ol>
<li><strong>Calibrated market thresholds.</strong> Argmax never fires rare orders such as <code>BUY_LAND</code> or <code>HIRE</code>. Each market key now fires when P(count &gt; 0) exceeds a per-key threshold, calibrated so the clone's held-out firing rate matches the teacher's.</li>
<li><strong>A sell-what-arrives clamp.</strong> DSM sells the same turn it delivers, so at observation time its shed shows 0. SELL counts must be clamped to shed stock <strong>plus goods PLACEd or DROPped this turn</strong>. Without this, the clone hoarded milk and wool.</li>
<li><strong>A legal-op mask, a PLANT cap at seeds in stock, and a last-day dump.</strong> The mask runs at inference only. Before the argmax, every op that is illegal for that unit on its tile this turn has its logit set to −1e9. Training sees the teacher's actions unmasked.</li>
</ol>
<p>With those fixes, the same weights went from $0.1k to $75.6k in DSM's recorded worlds after 2 epochs. A few epochs later the clone made <strong>more money than DSM itself</strong> in DSM's own hold-out worlds (v1 at epoch 9: $128k against DSM's $121k over 16 worlds):</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F22592122%2F42f5e2113345ae0634595d1f1e0ba082%2F08_decoder_fixes.png?generation=1792087936221207&amp;alt=media" alt="">
That benchmark keeps the opponent's recorded actions fixed, so it is a fidelity check, not proof that the clone was stronger than DSM (section 4 covers this bias). The closed-loop results are what convinced us. The first upload (v2f) went 24/26 on the ladder with a +$31k mean margin. Locally it beat our best hand-built agent <strong>24/24 at +$19.6k per game</strong>.</p>
<h2>3. Training: an RTX 5070, plus Colab for the big trunks</h2>
<p>All the CNN generations and the final fine-tunes were trained on one consumer GPU in a Windows desktop. The ViT and worker_hybrid sweeps were too slow on a 12 GB card, so we pre-trained those trunks on a Colab A100 and then fine-tuned and evaluated them on the 5070. The times below come from the training logs:</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F22592122%2F6c8c839f83aedcea21b1a11f5cedffb6%2F02_training_curves_5070.png?generation=1792087953451713&amp;alt=media" alt=""></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F22592122%2F0471eb07243676ecd2e528174c33ac80%2F03_train_time_5070.png?generation=1792087961104422&amp;alt=media" alt=""></p>
<table>
<thead>
<tr>
<th>model</th>
<th>data</th>
<th>epochs</th>
<th>wall-clock</th>
<th>where</th>
</tr>
</thead>
<tbody>
<tr>
<td>v1 CNN 128×6</td>
<td>1.1M DSM turns</td>
<td>16</td>
<td>36 min</td>
<td>RTX 5070</td>
</tr>
<tr>
<td>v2 (+carried goods)</td>
<td>1.2M turns</td>
<td>16</td>
<td>40 min</td>
<td>RTX 5070</td>
</tr>
<tr>
<td>v3 CNN 160×8</td>
<td>1.2M turns</td>
<td>20</td>
<td>73 min</td>
<td>RTX 5070</td>
</tr>
<tr>
<td>v4 top-20 teams</td>
<td>3.5M turns</td>
<td>12</td>
<td>117 min</td>
<td>RTX 5070</td>
</tr>
<tr>
<td>v4d (DSM fine-tune)</td>
<td>1.2M turns</td>
<td>6</td>
<td>21 min</td>
<td>RTX 5070</td>
</tr>
<tr>
<td>ViT / worker_hybrid sweeps</td>
<td>7-10k seat-games</td>
<td>12</td>
<td>hours</td>
<td>Colab A100</td>
</tr>
<tr>
<td><strong>dsmT (final fine-tune)</strong></td>
<td>219 DSM seat-games</td>
<td>3 (and 6)</td>
<td><strong>3.5 min</strong> (3 epochs)</td>
<td>RTX 5070</td>
</tr>
</tbody>
</table>
<p>What we learned about the hardware:</p>
<ul>
<li><strong>One GPU job at a time.</strong> Running two GPU-batched harnesses together left half the games with an agent that never acted (bank stuck at $3,000), including the control.</li>
<li><strong>The 12 GB card pages at ViT batch 1024</strong> once anything else holds memory, which made training about 25× slower. We used a smaller batch or Colab.</li>
<li><strong>Under Windows WDDM, each model call costs about 18 ms regardless of batch size</strong>, so on this machine batching across games matters more than model size.</li>
<li><strong>Pin BLAS to one thread</strong> in evaluation workers. OpenBLAS threading made our small conv matmuls 2.5× slower.</li>
</ul>
<h3>Architecture sweep</h3>
<p>On the same data, features, heads and 12 epochs, <strong>spatial attention beat temporal context by a wide margin.</strong> Nearly all the gain from history comes from the previous action alone.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F22592122%2F5771b74c4b96b8cb46a47a71ee3fdb84%2F04_arch_sweep.png?generation=1792087980847389&amp;alt=media" alt="">
The held-out loss did not always predict closed-loop strength. A DSM-only fine-tune of the ViT had <em>lower</em> loss (0.194) but lost about $4k per game. <strong>We judged every model by games, not by loss.</strong></p>
<h2>4. Evaluation: fast, paired and on the right worlds</h2>
<p>The bottleneck was always evaluation, not training.</p>
<ul>
<li><strong>GPU-batched gauntlet.</strong> Worker processes play games as threads, and one GPU forward per build serves every waiting game (full fp32, TF32 off, because TF32 broke parity). <strong>48 games take 101 s instead of 13 minutes</strong>, and 45/48 games match the numpy path's money exactly.</li>
<li><strong>Pinned ladder worlds.</strong> A replay's <code>info.seed</code> reproduces the game exactly, so we replay <em>our real ladder losses</em> with the rival's recorded actions pinned and swap in a candidate. This is how we found the tomato gap below.</li>
<li><strong>Why pinned worlds are still usable, despite the tape bias from section 1.</strong> The pinned rival is open-loop. Once our candidate changes the shared market, the rival keeps selling into prices it never saw. That makes absolute pinned scores unreliable. We only used pinned worlds to compare two candidates, both close to the agent the tapes were recorded against. Because both sides of that comparison inherit the same bias, the paired difference stays meaningful while the market changes only a little. We also watched for open-loop artifacts. One MMPQ fine-tune scored +$24k per game pinned, but 54 of its pinned rival tapes crashed once the market diverged, so we threw it out. Pinned worlds were a screen, and closed-loop head-to-head games were the gate.</li>
<li><strong>Teacher hold-out worlds.</strong> These are the teacher's own post-training ladder games. Both pinned and head-to-head wins can be artifacts (that MMPQ fine-tune also won 75% head-to-head, yet was not a faithful clone), so the teacher's own held-out worlds were our fidelity check.</li>
<li><strong>Count seeds, not games.</strong> Every head-to-head seed is played twice with the seats swapped. Shop unlocks and weed spawns come from the seed, and the game treats the two seats the same. So when both agents play the same way in both games, the seat-swapped game mirrors the first and each agent's money comes out identical. That happened in about 70% of seeds between our near-identical clones. With 24 games you have about 12 independent worlds, not 24.</li>
<li><strong>Gate rule.</strong> Before any upload, the candidate must win the incumbent's ladder losses and post a clear head-to-head win rate against the incumbent.</li>
</ul>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F22592122%2F04307676ea01127ba9036474ede9183c%2F05_incumbent_chain.png?generation=1792087993290831&amp;alt=media" alt=""></p>
<h2>5. The last gap: tomatoes</h2>
<p>We audited 86 ladder games against the ~2800 tier. Our clone matched or beat them on every product except one:</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F22592122%2F5495d6e4d3ca699681cd9b385b3b36e8%2F07_tomato_gap.png?generation=1792088006393681&amp;alt=media" alt="">
The top teams started tomatoes late (median opponent tomato plants: 0 by day 14, 8 by day 17, 11.5 by day 20) and sold them at about $113. Our clone came from DSM versions that did not grow tomato, so it never started.</p>
<ul>
<li><strong>R4 wrapper.</strong> If the opponent has at least 4 tomato plants between days 13 and 20, swap the clone's <code>PLANT WHEAT</code> to <code>PLANT TOMATO</code>, with a cap of 10 and a $1,500 cash guard. On 40 real tomato-opponent ladder worlds it scored <strong>+$2.2k per game paired, with 7 losses flipped to wins and 0 wins flipped to losses.</strong> Early tomato (from day 8) starved cash on days 8-10 and <em>lost</em> $6-9k per game.</li>
<li><strong>Wrapper tuning hit a local optimum.</strong> Other caps and triggers all landed within ±$230 of R4.</li>
<li><strong>dsmT, the final model.</strong> DSM's newest versions grow tomato in 100% of games, so we fine-tuned worker_hybrid on only those <strong>219 seat-games</strong> (lr/4, 3 epochs, <strong>3.5 min on the 5070</strong>). Validation unit loss went from 0.217 to 0.141. On its own, dsmT beat the wrapper agent (worker_hybrid + R4) by <strong>+$2.4k per game</strong> on 212 pinned ladder worlds, almost all of it in tomato worlds.</li>
<li><strong><code>_R4a</code>, the additive top-up.</strong> R4 rewrites the network's choices: it turns <code>PLANT WHEAT</code> into <code>PLANT TOMATO</code> and limits tomato seed stock. That would fight a model that already wants tomato. <code>_R4a</code> uses the same trigger (opponent has 4 or more tomato plants on days 13-20), but it never edits or limits the network's own tomato buys and plants. It only adds tomato when the network's own plan is still under the cap. We kept it as a safety net because dsmT sometimes starts too late. In one head-to-head game it planted its first tomato on day 18, with only 4-6 plants. dsmT with <code>_R4a</code> beat R4 61% head-to-head and flipped 6 of R4's 12 losses on R4's own ladder worlds.</li>
<li><strong>dsmT6e_R4a, the best final sub.</strong> It is the same recipe at 6 epochs. Its validation loss is <em>worse</em> (0.150), but it went 70% head-to-head against R4 and 24/40 in DSM's held-out seat (R4 21/40). It ended at <strong>2807.4</strong>.</li>
</ul>
<p>The lesson is that <strong>a crop strategy has to come from the plan (the teacher), not from a wrapper.</strong> As the main mechanism, the wrapper only patched the symptom: R4 alone hit a local optimum within ±$230. Switching the teacher was worth +$2.4k on top of that. In the final agent the wrapper is demoted to a small safety net on top of a policy that already grows tomato.</p>
<h2>6. What did not work</h2>
<table>
<thead>
<tr>
<th>idea</th>
<th>result</th>
</tr>
</thead>
<tbody>
<tr>
<td>Unit-coordination masks (dedupe units claiming the same fertilizer, harvest or feed)</td>
<td>−$42k to −$89k per game. The clone learned DSM's own collisions, and "fixing" them pushes it off-distribution.</td>
</tr>
<tr>
<td>Relabelling engine no-ops as PASS in training</td>
<td>62% fewer no-ops, no score change</td>
</tr>
<tr>
<td>Single-team fine-tunes (MMPQ, Victor) / playing in a non-DSM style slot</td>
<td>−$3k to −$4k per game, despite better loss on that team's games</td>
</tr>
<tr>
<td>Mixing all top-20 teams with a style slot (v4)</td>
<td>no better than DSM-only</td>
</tr>
<tr>
<td>Weight soup (0.5·base + 0.5·fine-tune)</td>
<td>loss in between, but 25-38% head-to-head</td>
</tr>
<tr>
<td>Logit ensemble of 3 fine-tunes</td>
<td>+$0.4-0.7k per game, but no extra wins, and 3× slower (321 ms per turn)</td>
</tr>
<tr>
<td>Head-only PPO on the BC policy (frozen trunk, exact legal-set log-probs)</td>
<td>tied the base (+$36 ± $596)</td>
</tr>
<tr>
<td>Sale horizon by rival class, land-buying wrapper, sheep/goose annexes, melon races</td>
<td>all flat or negative</td>
</tr>
</tbody>
</table>
<h2>7. Takeaways</h2>
<ol>
<li><strong>If move-level data from a stronger agent exists, clone it per unit and in closed loop first.</strong> We spent weeks hand-building executors for plans that a 40-minute BC run on a consumer GPU learned better.</li>
<li><strong>Audit the decoder before you judge the model.</strong> Threshold calibration and a same-turn sell clamp were the difference between $0 and beating the teacher.</li>
<li><strong>Lower validation loss did not reliably mean a stronger agent.</strong> Gate on games: paired, seed-counted and in the right worlds (your real losses and the teacher's hold-outs).</li>
<li><strong>On one GPU, spend compute on evaluation throughput.</strong> Batching the network across games gave an 8× faster gauntlet, which was worth more than a bigger model.</li>
<li><strong>Look at the losses one product at a time.</strong> One crop explained most of the gap to the 2800 tier.</li>
<li><strong>Don't "fix" the teacher's mistakes at play time.</strong> Units in our clone sometimes collide on the same task, as DSM's do. Masking those collisions cost $42k-$89k per game, because the clone was pushed into states it had never seen. If a behaviour has to change, change it in training.</li>
</ol>
<p>Thanks for reading. Happy to answer questions in the comments.</p>

---

### Message 2 by  (Votes: 0)

<p>Your sell-what-arrives fix prompted me to check DROP with a nearly full shed on 1.32.7: four carried carrots and two free slots leave two carrots discarded. Did shed overflow show up in your game audits, or did the teacher generally avoid it?</p>

---

### Message 3 by  (Votes: 0)

<p>Yes, overflow showed up in our own audits. The main cause was hour-23 buying orders landing right after the shed had been emptied. Trimming those buys and re-issuing them at hour 0 cut overflow to 3.2 units per game.</p>

---

### Message 4 by  (Votes: 0)

<p>I feel so happy that you analyse our records and understand our model more deeply than we do:)</p>

---

### Message 5 by  (Votes: 0)

<p>so cool, that all this is from home gpu! thanks for sharing this! </p>

---

