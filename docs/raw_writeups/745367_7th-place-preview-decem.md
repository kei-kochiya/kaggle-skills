# Topic 745367: [7th Place (currently)] A preview of my solution: Only Behavior Cloning + two small rules (Published code)

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/745367
- **Author**: 
- **Post Date**: 2026-10-03T04:12:30.389000
- **Votes**: 58

## Messages

### Message 1 by  (Votes: 1)

<p>wow This is a excellent solution! Thanks for sharing.</p>

---

### Message 2 by  (Votes: 1)

<p>This is a great read, thank you for writing and sharing the recipe. </p>

---

### Message 3 by  (Votes: 58)

<p>Thanks to Kaggle and the hosts for a really fun competition, and congrats to the top teams!</p>
<p>The leaderboard is still moving, so "7th" is just where I am today. I spent the first seven weeks on a heuristic approach. This post is about the last nine days, after I switched to behavior cloning (BC).</p>
<h2>The short version</h2>
<ul>
<li>An 8.2M-parameter entity Transformer, trained by BC on replays of the top agents and conditioned on team, version and game outcome. It runs in plain NumPy at about 0.1 s per turn.</li>
<li>Fresh data mattered more than anything else. The top teams kept shipping new versions, so the newest games get the most weight in training, and at play time the model is told to play like the newest version of the strongest team.</li>
<li>The last stage is a specialist fine-tune on that team's newest versions. It was the only way I found to learn a new opening; mixed in with everything else, it just got averaged out.</li>
<li>RL on top helped a little and I submitted it several times, but I stopped once I counted how few games it could actually flip. Both final submissions are BC plus two small rules that keep the engine from throwing my goods away.</li>
</ul>
<h2>Data and conditioning</h2>
<p>Each training sample is a (state, action) pair from one player's seat. Three embeddings get added to the global token: the team, the date bucket (one per day, which in practice is one version of that team) and whether that player won. At play time I just ask for the strongest team, its newest bucket and won = 1. Switching teachers is a one-line config change, which came in handy when the team worth copying changed in the last week.</p>
<p>Samples are weighted by recency with a 3-day half-life.</p>
<h2>The model</h2>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F6491363%2F32adeee0b88c0babc697966b834b5d57%2Farch_en.png?generation=1791000700942076&amp;alt=media" alt="Policy network"></p>
<p>It's one network with 8.2M parameters. Inputs are always written from my seat, so the model only ever sees "my farm" against "their farm".</p>
<p><strong>Input: 237 tokens per turn.</strong></p>
<ul>
<li>1 global token: money on both sides and the gap between them, hires, owned quadrants, unit counts, which shops are open, the day and the hour, plus the three conditioning embeddings.</li>
<li>12 item tokens, one for each of the 9 goods and 3 animals: price relative to its base price, how far the market inventory has drifted from where it started, how much the open shops consume every 4 steps, how many I have in the shed, in backpacks and as seeds, and, for crops and animals, how many tiles each farm has.</li>
<li>24 unit tokens: the farmer and up to 23 hands, each with a position embedding, a slot embedding and its backpack. Missing units get a "nowhere" position instead of a padding mask, and the model learns to ignore them.</li>
<li>200 tile tokens, 100 for my farm and 100 for the opponent's. Every tile field is a small integer (what's on it, age, yield, watered or fed today, days without water or food, fertilizer state, steps until it starts to rot, care state, how many units stand on it), so each field gets its own embedding table and the tile token is just their sum.</li>
</ul>
<p><strong>Encoder.</strong> 8 Transformer layers, d_model 256, 8 heads, FFN 1024, pre-LN. Positions use 2-D axial RoPE: in each head, half of the rotation pairs turn with the tile's column and the other half with its row, so the attention score between two tiles depends on their offset (Δx, Δy). Both farms share one coordinate system, a unit uses the tile it's standing on, and the global and item tokens aren't rotated. Every output logit goes through a soft cap, 30·tanh(x/30), so a single odd sample can't blow up the loss.</p>
<p><strong>Unit decoder.</strong> Units are decoded in order, farmer first. The query for unit i is its encoder token plus the token of the tile under it. It cross-attends (2 layers, FFN 512) to a memory that holds one learned empty slot plus one entry per earlier unit: that unit's token and what it chose (task, target tile, raw action and item). A causal mask means unit i only sees the units before it. Each unit then makes four choices, each conditioned on the ones before it:</p>
<ol>
<li>A task, one of 14: the 18 unit actions minus the four moves.</li>
<li>A target tile, as a pointer over my 100 tiles, scored jointly with the task.</li>
<li>The raw action for this turn, one of 18 including the moves, conditioned on the task, the target tile and how far it is from the unit (Δx, Δy).</li>
<li>An item for PICKUP / PLANT / PLACE, and a quantity for PICKUP / PLACE.</li>
</ol>
<p>Only the raw action goes to the engine. The task and target are there so later units can see what's already been claimed, and so the raw action knows where it's going. That's what stops two units from walking over to water the same tile.</p>
<p><strong>Market heads.</strong> These don't need decoding and read straight off the encoder. Each good's item token gives a sell quantity, each crop's gives a seed purchase, wheat and fertilizer get a buy quantity, each animal gets how many to buy (0–8), and the global token decides hires (0–15) and whether to buy land. Quantities are predicted as classes over bins: every integer up to 30, then coarser steps up to 100 for sells and up to 1000 for purchases.</p>
<p><strong>Labels and loss.</strong> The task label is the first thing the unit does from now until the end of the day that isn't a move, and the target is the tile it was standing on when it did it. If it only walks for the rest of the day, the label is PASS at the tile where it stops. The raw action label is just what it did this turn. Market labels are quantities per item. Sells are capped at what's actually in the shed after this turn's unit actions, and animal purchases count the animals that actually arrived. Every head uses plain cross-entropy and the losses are summed, with the item and quantity heads at half weight. The decoder is trained with teacher forcing, so each unit sees the teacher's choices for the units before it.</p>
<p><strong>At play time.</strong> The encoder runs once per turn and only the small decoder runs once per unit, so going autoregressive was cheap. For the submission I rewrote the forward pass in NumPy, so there's no torch in the package. A turn takes about 0.1 s on Kaggle's CPU, and the package is ~30 MB.</p>
<p>Three decoding details mattered more than I expected:</p>
<ul>
<li>The raw action doesn't commit to one plan first. It's picked by marginalizing over the top-8 (task, target) plans, weighted by their probabilities. "Go east" is often right for several targets at once.</li>
<li>A legality mask replays the turn so far, unit by unit, so unit 5 doesn't try to pick up the wheat unit 2 just took.</li>
<li>Market quantities go through an expected-quantity accumulator. With a per-step argmax, a model that is unsure about timing says "0" every single step and never buys the thing.</li>
</ul>
<p>The first versions had a hand-written executor: the model picked a target and rules walked the unit there. Switching to raw actions from the decoder matched the teachers better and won a few more games, and I got to delete a lot of fragile code.</p>
<h2>Training recipe</h2>
<ol>
<li>Pretrain on ~10k games from the last two weeks, all seats, recency weighted. 30k steps at batch size 512 with the Muon optimizer, about 50 minutes on one RTX 5090.</li>
<li>Fine-tune on the latest day (4k steps).</li>
<li>Specialist: fine-tune only on the newest versions of one strong team (versions that play the same plan), early-stopped on validation loss after a few hundred to ~1k steps.</li>
<li>Evaluate EMA weights only.</li>
</ol>
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
<li><strong>Overflow sell.</strong> At the end of each day the engine empties every backpack into the shed and throws away anything above the 100-item cap. Right before the day ends, the agent counts how much of the unload won't fit and sells that many of the same items from the shed first. About +$300 per game in margin.</li>
<li><strong>Last-day drop-off.</strong> Day 29 has no end-of-day unload, so anything still in a backpack at the end is lost. Units carrying goods head back and DROP when the remaining steps equal their distance to the shed.</li>
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
<h2>Things that didn't work</h2>
<ul>
<li>Copying single macro moves from other teams (buy the 4th quadrant on day 10, add cows or sheep, always plant tomatoes on day 16). Each one hurt on its own.</li>
<li>A contrastive auxiliary loss with hard negatives from the same game. The model solved the contrastive task almost immediately, the BC loss didn't budge, and it lost to the baseline in closed loop.</li>
<li>Classifier-free-guidance style extrapolation of the conditioning: no effect, and inference got twice as slow.</li>
</ul>
<h2>Compute</h2>
<p>Everything ran locally: one RTX 5090 for training and an Intel Core Ultra 7 270K Plus (24 cores) for evaluation.</p>
<p>Thanks again to the hosts and to everyone who shared code and ideas on the forum. I also owe a lot to the top teams I cloned. Their new versions kept coming, and chasing them was a lot of fun.</p>
<p>Code is open source now: <a href="https://github.com/Decem-Y/kaggriculture-bc" target="_blank">Decem-Y/kaggriculture-bc</a></p>
<p>DECEM</p>

---

### Message 4 by  (Votes: 3)

<p>Code is open source now: <a href="https://github.com/Decem-Y/kaggriculture-bc" target="_blank">https://github.com/Decem-Y/kaggriculture-bc</a></p>

---

### Message 5 by  (Votes: 1)

<p>This is actually awesome and very informative for some of us. Congratulations DECEM!</p>

---

### Message 6 by  (Votes: 0)

<p>Thank you for your recognition!</p>

---

### Message 7 by  (Votes: 1)

<p>Great share and congratulations!!</p>

---

### Message 8 by  (Votes: 1)

<p>Great job. Congratulations!! 🎉</p>

---

### Message 9 by  (Votes: 0)

<p>Thank you!</p>

---

