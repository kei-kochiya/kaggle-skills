# Topic 751681: [7th Place] My solution: only behavior cloning + two small rules

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/751681
- **Author**: 
- **Post Date**: 2026-10-22T13:28:42.483000
- **Votes**: 4

## Messages

### Message 1 by  (Votes: 4)

<p>Thanks to Kaggle and the hosts for a really fun competition, and congrats to the top teams!</p>
<p>I finished 7th with a final score of 5459.9. I spent the first seven weeks on a heuristic approach. This post is about the last nine days, after I switched to behavior cloning (BC).</p>
<p>Code: <a href="https://github.com/Decem-Y/kaggriculture-bc" target="_blank">Decem-Y/kaggriculture-bc</a>. It has the data building, all three training stages, the NumPy agent and the fork evaluation, and the README lists the commands for each step.</p>
<h2>The short version</h2>
<ul>
<li>An 8.2M-parameter entity Transformer, trained by BC on replays of the top agents and conditioned on team, version and game outcome. It runs in plain NumPy at about 0.1 s per turn.</li>
<li>Fresh data mattered more than anything else. The top teams kept shipping new versions, so the newest games get the most weight in training, and at play time the model is told to play like the newest version of the strongest team.</li>
<li>The last stage is a specialist fine-tune on that team's newest versions. It was the only way I found to learn a new opening; mixed in with everything else, it just got averaged out.</li>
<li>RL on top helped a little and I submitted it several times, but I stopped once I counted how few games it could actually flip. Both final submissions are BC plus two small rules that keep the engine from throwing my goods away.</li>
</ul>
<h2>Data</h2>
<p>Pretraining and the first fine-tune use Kaggle's public daily episode datasets. The specialist stage uses the public replays of the target team's newest versions.</p>
<p>I keep a game only if it ran all 720 steps on engine 1.32.7 and no agent ended in ERROR, TIMEOUT or INVALID. I also drop every game in which a team plays a byte-identical action stream that also shows up in another of its games. Those are fixed scripts that never look at the board, so there's nothing to learn from them. My own seats are never trained on.</p>
<p>The final pretraining set is 10,150 games from September 14 to 27. Every turn of both seats is a sample. Games are drawn with weight 0.5^(age in days / 3), counted back from the newest day in the set. Validation games are the ones with episode_id % 50 == 0, and % 4 == 0 for the specialist stage.</p>
<h2>Conditioning</h2>
<p>Each training sample is a (state, action) pair from one player's seat. Three embeddings get added to the global token:</p>
<ul>
<li>Team. A team with at least 50 trainable seats gets its own embedding (34 teams in the final model), and everyone else shares one <code>&lt;other&gt;</code> embedding.</li>
<li>Date bucket. One per calendar day, counted from August 17, 42 buckets in total, with later days put in the last one. In practice a bucket is one version of that team.</li>
<li>Won. 1 or 0 from that player's final result. Losing seats are trained too.</li>
</ul>
<p>At play time I just ask for the strongest team, its newest bucket and won = 1. Switching teachers is a one-line config change, which came in handy when the team worth copying changed in the last week.</p>
<h2>The model</h2>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F6491363%2F0a266817e956957db4ee67e7b4ea7925%2Farch_en.png?generation=1792675234592357&amp;alt=media" alt="Policy network"></p>
<p>It's one network with 8.2M parameters. Inputs are always written from my seat, so the model only ever sees "my farm" against "their farm".</p>
<p><strong>Input: 237 tokens per turn.</strong></p>
<ul>
<li>1 global token: money on both sides and the gap between them, hires, owned quadrants, unit counts, which shops are open, the day and the hour, plus the three conditioning embeddings.</li>
<li>12 item tokens, one for each of the 9 goods and 3 animals: price relative to its base price, how far the market inventory has drifted from where it started, how much the open shops consume every 4 steps, how many I have in the shed, in backpacks and as seeds, and, for crops and animals, how many tiles each farm has.</li>
<li>24 unit tokens: the farmer and up to 23 hands, each with a position embedding, a slot embedding and its backpack. Missing units get a "nowhere" position instead of a padding mask, and the model learns to ignore them.</li>
<li>200 tile tokens, 100 for my farm and 100 for the opponent's. Every tile field is a small integer (what's on it, age, yield, watered or fed today, days without water or food, fertilizer state, steps until it starts to rot, care state, how many units stand on it), so each field gets its own embedding table and the tile token is just their sum.</li>
</ul>
<p><strong>Encoder.</strong> 8 Transformer layers, d_model 256, 8 heads, FFN 1024, pre-LN, no dropout. Positions use 2-D axial RoPE (θ = 100): in each head, half of the rotation pairs turn with the tile's column and the other half with its row, so the attention score between two tiles depends on their offset (Δx, Δy). Both farms share one coordinate system, a unit uses the tile it's standing on, and the global and item tokens aren't rotated. Every output logit goes through a soft cap, 30·tanh(x/30), so a single odd sample can't blow up the loss.</p>
<p><strong>Unit decoder.</strong> Units are decoded in order, farmer first. The query for unit i is its encoder token plus the token of the tile under it. It cross-attends (2 layers, FFN 512) to a memory that holds one learned empty slot plus one entry per earlier unit: that unit's token and what it chose (task, target tile, raw action and item). A causal mask means unit i only sees the units before it. Each unit then makes four choices, each conditioned on the ones before it:</p>
<ol>
<li>A task, one of 14: the 18 unit actions minus the four moves.</li>
<li>A target tile, as a pointer over my 100 tiles, scored jointly with the task.</li>
<li>The raw action for this turn, one of 18 including the moves, conditioned on the task, the target tile and how far it is from the unit (Δx, Δy).</li>
<li>An item for PICKUP / PLANT / PLACE, and a quantity for PICKUP / PLACE.</li>
</ol>
<p>Only the raw action goes to the engine. The task and target are there so later units can see what's already been claimed, and so the raw action knows where it's going. That's what stops two units from walking over to water the same tile.</p>
<p><strong>Market heads.</strong> These don't need decoding and read straight off the encoder. Each good's item token gives a sell quantity, each crop's gives a seed purchase, wheat and fertilizer get a buy quantity, each animal gets how many to buy (0–8), and the global token decides hires (0–15) and whether to buy land. Quantities are classes over bins: every integer from 0 to 30, then 35, 40, 45, 50, 60, 70, 80, 90, 100 for sells and unit quantities, and purchases add 150, 200, 300, 500, 1000. A target between two bins goes to the closer one.</p>
<p><strong>Labels and loss.</strong> The task label is the first thing the unit does from now until the end of the day that isn't a move, and the target is the tile it was standing on when it did it. If it only walks for the rest of the day, the label is PASS at the tile where it stops. The raw action label is just what it did this turn. Market labels are quantities per item. Sells are capped at what's actually in the shed after this turn's unit actions, and animal purchases count the animals that actually arrived. Every head uses plain cross-entropy and the losses are summed, with the item and quantity heads at half weight and everything else at weight 1. The decoder is trained with teacher forcing, so each unit sees the teacher's choices for the units before it.</p>
<p><strong>At play time.</strong> The encoder runs once per turn and only the small decoder runs once per unit, so going autoregressive was cheap. For the submission I rewrote the forward pass in NumPy, so there's no torch in the package. A turn takes about 0.1 s on Kaggle's CPU, and the package is ~30 MB. Every choice is greedy, nothing is sampled.</p>
<p>Three decoding details mattered more than I expected:</p>
<ul>
<li>The raw action doesn't commit to one plan first. It's picked by marginalizing over the top-8 (task, target) plans, weighted by their probabilities. "Go east" is often right for several targets at once.</li>
<li>A legality mask replays the turn so far, unit by unit, so unit 5 doesn't try to pick up the wheat unit 2 just took.</li>
<li>Market quantities go through an expected-quantity accumulator. Each step adds the expected quantity under the model's distribution to a running total, the order is the larger of the argmax and the whole part of that total, and the ordered amount is subtracted. With a plain per-step argmax, a model that is unsure about timing says "0" every single step and never buys the thing.</li>
</ul>
<p>The first versions had a hand-written executor: the model picked a target and rules walked the unit there. Switching to raw actions from the decoder matched the teachers better and won a few more games, and I got to delete a lot of fragile code.</p>
<h2>Training</h2>
<p>Three stages, each starting from the previous one's checkpoint, all on one RTX 5090:</p>
<table>
<thead>
<tr>
<th></th>
<th>Pretrain</th>
<th>Fine-tune</th>
<th>Specialist</th>
</tr>
</thead>
<tbody>
<tr>
<td>Data</td>
<td>10,150 games, Sep 14–27, all teams</td>
<td>same set, Sep 27 only</td>
<td>645 games from the target team's 6 newest versions (482 train / 163 val)</td>
</tr>
<tr>
<td>Steps</td>
<td>30,000</td>
<td>4,000</td>
<td>1,000, early-stopped on val loss</td>
</tr>
<tr>
<td>Warmup</td>
<td>1,000</td>
<td>200</td>
<td>50</td>
</tr>
<tr>
<td>Peak LR, Muon / AdamW</td>
<td>3e-3 / 4e-4</td>
<td>7.5e-4 / 1e-4</td>
<td>7.5e-4 / 1e-4</td>
</tr>
<tr>
<td>EMA decay</td>
<td>0.9998</td>
<td>0.999</td>
<td>0.999</td>
</tr>
<tr>
<td>Wall time</td>
<td>51 min</td>
<td>7 min</td>
<td>3 min</td>
</tr>
</tbody>
</table>
<p>Shared settings:</p>
<ul>
<li>Batch of 512 (turn, seat) samples, bf16 autocast, linear warmup then cosine decay to 5% of the peak, gradient clipping at 1.0.</li>
<li>Muon handles the hidden 2-D weight matrices: momentum 0.95, 5 Newton–Schulz steps with the Polar Express coefficients, Q/K/V orthogonalized as separate blocks, no weight decay. Embeddings, norms, biases and output heads use AdamW with betas (0.9, 0.95) and weight decay 0.05.</li>
<li>A step is skipped when its gradient norm jumps far above the running norm.</li>
<li>Only the EMA weights are evaluated and deployed.</li>
</ul>
<p>In the specialist stage every seat of the target team goes into one date bucket, and the agent plays with that bucket. In the final run the best validation loss came at the last step.</p>
<p>My two final submissions use exactly this recipe. They differ only in the specialist data: one also includes a version the target team released on the last day. The team score takes the better of the two, and that was the one with the extra version.</p>
<h2>How to reproduce</h2>
<p>The README has every flag. The pipeline is:</p>
<pre><code>python scripts/build_census.py ...                      # replays -&gt; filtered census, caches, team vocabulary
python train.py --config configs/pretrain.yaml   --run pretrain
python train.py --config configs/finetune.yaml   --run finetune
python train.py --config configs/specialize.yaml --run specialize
python scripts/build_agent.py runs/specialize/best.pt agents/final --team "&lt;TARGET TEAM&gt;" --date 41 --topk 8
python scripts/package.py agents/final --out final.tar.gz --parity
</code></pre>
<p><code>build_census.py</code> runs twice, once on all teams for the first two stages and once on the target team's newest versions with the first vocabulary. <code>package.py</code> checks the tarball on the official engine.</p>
<h2>RL on top of BC</h2>
<p>I really wanted RL to be the main line and spent several days on it. The conditioning has a nice side effect here: ask the BC model to play as other teams and you get a whole league of opponents for free.</p>
<ul>
<li><strong>Evolution strategies (ES) in the conditioning space.</strong> Freeze the network and learn an offset to the team embedding, restricted to a 16-dim subspace (the top principal directions of the differences between team, date-bucket and won embeddings). It made +$1.4k per game against the clone league and about +$800 on real game records, and I submitted it. I think it works because nudging the "style" makes the whole network re-plan together, while changing one macro decision by hand breaks the coordination BC had learned.</li>
<li><strong>PPO on the same offset.</strong> About +$1k per game on fresh games. Also submitted.</li>
</ul>
<p>Going down to the action level went worse:</p>
<ul>
<li>Full-network PPO with a KL penalty toward the BC policy. Both players got about $500 richer, and margin and win rate didn't move. The critic wasn't the problem (0.99 explained variance). One decision is worth a few tens of dollars, and game-to-game noise is around $4k.</li>
<li>Per-unit credit assignment, where each unit only gets credit for its own work. Units learned to skip animal harvests and deliveries that didn't pay off in their own books. My supply dropped, the shared market price went up, and the opponent pocketed about $2.6k per game.</li>
<li>Residual policies on macro decisions (more animals, more hires, a different crop mix) never got above zero.</li>
<li>Using recorded opponents as the RL environment: the policy just learns exploits that only work because a recording can't react.</li>
<li>Tomato sell timing made +$3.3k per game in games where tomatoes were in play, but it didn't flip a single game, and the ladder only cares about wins.</li>
</ul>
<p>What made me stop was going through my losses against the three strongest teams. Most of them weren't close: the median margin was about $4.7k, and only 12–15% were within $2k. So even a reliable +$1k per game would turn maybe one loss in seven into a win. On top of that, those teams put out a new version almost every day and kept getting stronger. For me, keeping up with their latest version was simply worth more than another RL run.</p>
<h2>The two rules I kept</h2>
<p>Almost every rule I tried on top of BC was neutral or negative, and the same rule could flip sign with new weights. A "rescue watering" rule was +$1.6k on one model and −$2.8k on the next. The two I kept both stop the engine from destroying my stuff:</p>
<ul>
<li><strong>Overflow sell.</strong> At the end of each day the engine empties every backpack into the shed and throws away anything above the 100-item cap. On the last step of each day, the agent counts how much of the unload won't fit and sells that many of the same items from the shed first. About +$300 per game in margin.</li>
<li><strong>Last-day drop-off.</strong> Day 29 has no end-of-day unload, so anything still in a backpack at the end is lost. Units carrying goods head back and DROP when the remaining steps equal their distance to the shed. On step 718, the last step that takes actions, the agent puts the whole shed up for sale, including what was just dropped.</li>
</ul>
<p>On 61 held-out forks built from lost ladder games, the two together turned 6 losses into wins and no wins into losses. There's also the usual crash guard, which returns PASS for the turn if anything throws.</p>
<h2>Evaluation</h2>
<p>Honestly, this is where most of my time went.</p>
<ul>
<li><strong>Engine.</strong> The community C++ port of the engine (destbreso's kaggriculture-cppsim, engine core by nikital7, thank you both!), with a parity check against the official engine for every new opponent. Even with it, most of the wall-clock time went into my own Python.</li>
<li><strong>Paired games.</strong> Candidate and reference play the same seeds, shop draws and seats. I count strict flips (loss→win vs win→loss, ties counted separately) and run a sign test. I don't trust average money much, because plenty of changes only win bigger in games that were already won.</li>
<li><strong>Me vs. them.</strong> I split every gain into how much more I made and how much less the opponent made. A change that only makes the opponent earn less can rank completely differently by margin than by wins.</li>
<li><strong>Real-game forks.</strong> Take real ladder games, replay both sides from the record up to day 8, then let the candidate take over my side while the opponent keeps following its recorded actions. Replaying my own ladder games with my original agent reproduced 99 of 99 to the dollar, so the forks are faithful. Most model choices in the last week were made on them.</li>
</ul>
<h2>Final results</h2>
<p>Counting only the games played after the deadline, where both sides were final submissions, my record against the six teams above me was:</p>
<table>
<thead>
<tr>
<th>Opponent</th>
<th>Wins–losses</th>
</tr>
</thead>
<tbody>
<tr>
<td>1st</td>
<td>0–15</td>
</tr>
<tr>
<td>2nd</td>
<td>16–66</td>
</tr>
<tr>
<td>3rd</td>
<td>102–200</td>
</tr>
<tr>
<td>4th</td>
<td>101–142</td>
</tr>
<tr>
<td>5th</td>
<td>133–111</td>
</tr>
<tr>
<td>6th</td>
<td>145–134</td>
</tr>
</tbody>
</table>
<p>There were no draws. I was slightly ahead of 5th and 6th head-to-head and clearly behind the top three. That fits the method: a clone of one team's newest versions can get close to the team it copies, but it has no way to get past it, and the top teams kept improving until the deadline.</p>
<h2>Things that didn't work</h2>
<ul>
<li>Copying single macro moves from other teams (buy the 4th quadrant on day 10, add cows or sheep, always plant tomatoes on day 16). Each one hurt on its own.</li>
<li>A contrastive auxiliary loss with hard negatives from the same game. The model solved the contrastive task almost immediately, the BC loss didn't budge, and it lost to the baseline in closed loop.</li>
<li>Classifier-free-guidance style extrapolation of the conditioning: no effect, and inference got twice as slow.</li>
</ul>
<h2>Compute</h2>
<p>Everything ran locally: one RTX 5090 for training and an Intel Core Ultra 7 270K Plus (24 cores) for evaluation.</p>
<p>Thanks again to the hosts and to everyone who shared code and ideas on the forum.</p>
<p>Special thanks to M &amp; M &amp; P &amp; Q and DSM. Most of what my agent knows comes from your replays. DSM was my main teacher for most of those nine days, and M &amp; M &amp; P &amp; Q took over in the last few. My final submission is a specialist fine-tune on M &amp; M &amp; P &amp; Q's newest versions. You kept shipping new versions right up to the deadline, and trying to keep up with you was a lot of fun.</p>
<p>The code is MIT-licensed, and my winning submission is also released under CC-BY 4.0 as the competition rules require.</p>
<p>DECEM</p>

---

### Message 2 by  (Votes: 0)

<p>Your diagram looks great. How was it generated?</p>

---

### Message 3 by  (Votes: 0)

<p>Thank you for your compliment! </p>
<p>DeepSeek web version is all you need! 😂</p>

---

