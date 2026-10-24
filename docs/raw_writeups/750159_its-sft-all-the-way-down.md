# Topic 750159: It’s SFT all the way down

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/750159
- **Author**: 
- **Post Date**: 2026-10-17T02:57:57.963000
- **Votes**: 13

## Messages

### Message 1 by  (Votes: 13)

<p><em>5th place (gold) in <a href="https://www.kaggle.com/competitions/kaggriculture" target="_blank">Kaggriculture</a>, team 有辣条有权.</em></p>
<table>
<thead>
<tr>
<th>Looking for</th>
<th>Go to</th>
</tr>
</thead>
<tbody>
<tr>
<td>What we did, technique by technique, with links to the code</td>
<td><a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/blob/main/HOW-IT-WORKS.md" target="_blank">HOW-IT-WORKS.md</a></td>
</tr>
<tr>
<td>Every feature, and the solvers that compute them</td>
<td><a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/blob/main/FEATURES.md" target="_blank">FEATURES.md</a></td>
</tr>
<tr>
<td>The code that matters, flat and grouped: features, Model 1, Model 2</td>
<td><a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/tree/main/code" target="_blank">code/</a> (Model 1 <a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/blob/main/code/model1/model.py" target="_blank"><code>model.py</code></a>, Model 2 <a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/blob/main/code/model2/nf_model.py" target="_blank"><code>nf_model.py</code></a>)</td>
</tr>
<tr>
<td>Training on one replay, end to end</td>
<td><a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/tree/main/quickstart" target="_blank">quickstart/</a></td>
</tr>
<tr>
<td>How games were collected, filtered and sampled</td>
<td><a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/tree/main/data" target="_blank">data/</a></td>
</tr>
<tr>
<td>The submitted agents, and how candidates were compared</td>
<td><a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/tree/main/submissions" target="_blank">submissions/</a> · <a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/tree/main/evaluation" target="_blank">evaluation/</a></td>
</tr>
<tr>
<td>Rebuilding both submissions from scratch</td>
<td><a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/blob/main/REPRODUCE.md" target="_blank">REPRODUCE.md</a></td>
</tr>
</tbody>
</table>
<h2>Intro</h2>
<p>This covers my attempt at using SFT to solve Kaggriculture. It assumes the vocabulary in the <a href="https://www.kaggle.com/competitions/kaggriculture/overview" target="_blank">competition overview</a>.</p>
<p>I don't quote leaderboard scores. The Elo-style rating only means something relative to the field at the time: a bot of the same strength would score several hundred points lower later on, because the field as a whole got stronger as time went on, and the rate of improvement increased closer to the deadline, especially once a lot of people moved to using all four quadrants in the last three days.</p>
<p>I mostly document what worked. Plenty didn't, and I don't keep meticulous logs of what failed. Each failure still cost real time. For example, I spent two weeks factoring the game's micro decisions into macro decisions, down to a page of theorems on how to handle them (311 claims, 919 linked pairs). This is the approach I really wanted to take, but it proved too much for agents to keep track of at once, even after I simplified it for them. They still made very subtle mistakes. I imagine this method will work once agents get stronger.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2F95fe092034ec22b5743a6948a37d5c6f%2F01-figA-theorems.png?generation=1790888796158407&amp;alt=media" alt="The abandoned macro approach: 311 claims and their links, with random samples"></p>
<h2>Resources</h2>
<p>I enter each competition to win or learn as much as possible, within a budget. Here: two 4090s I own, a 3090 (that kept dying), a DGX Spark and a laptop 4060 for eval games, Colab G4s rented on 29 and 30 Sep, and Kaggle CPU notebooks for evals. I have two Claude and two GPT subscriptions. These resources are shared among all my projects, including personal ones.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2F21f1a2e4c919925eb42360c4a9d1dfc2%2F02-figB-resources.png?generation=1790889014377947&amp;alt=media" alt="GPU-hours by machine and training hours behind each final model"></p>
<p>If I had more compute, I would have used it.</p>
<p>Team name: <strong>有辣条有权</strong> ("you la tiao you quan"). La tiao is a famous cheap spicy snack, and the name is a play on <strong>有钱有权</strong> ("you qian you quan"): "To have money is to have power." Swap 钱 (qian, money) for 辣条 (la tiao) and you get the team name.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2F54762354ec96d0bb7073d1841ffdb709%2F03-latiao-photo.png?generation=1790889065062352&amp;alt=media" alt=""></p>
<h2>Models</h2>
<p>These were my two final submissions. The larger, longer-trained B-even model finished ahead of N-flow and set the team's placing.</p>
<h3>Abandoned</h3>
<p>The macro approach went first, so I won't discuss the architectures I considered there. Everything below came after I switched to micro decisions generated autoregressively. I had <a href="https://www.youtube.com/watch?v=SQ3fZ1sAqXI&amp;list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_" target="_blank">this lecture series</a> playing in the background, and it drove the switch. This game doesn't seem that different from writing a paragraph in a very specific style.</p>
<p>I train on the state the winner saw and the actions they sent. During play, the model generates worker commands, then orders, then STOP, one token at a time, with each token seeing the earlier choices from that turn.</p>
<p>Other failures:</p>
<ol>
<li><strong>Chain of thought:</strong> force an unsupervised label onto every tile before deciding.</li>
<li><strong>Recurrence in depth:</strong> rerun the middle blocks several times. Showed a lot of promise early, with some of our best losses, but needed too many ablations to get right. No time to tune how many times, or which layers, to recur, and it ended up doing poorly late with the config I chose. Something I'll try earlier next time.</li>
<li><strong>Depth over width:</strong> worse loss immediately.</li>
<li><strong>Massive parallel heads:</strong> 36 heads of width 12. Did very well and only started to do poorly later, which leads to Model 1.</li>
</ol>
<h3>Model 1: B-even</h3>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2Fc95ff4441c6f91668e4f434365e1b788%2F04-figG-arch-b.png?generation=1790889149090921&amp;alt=media" alt="Model 1 architecture: game state to tokens and features, encoder x10, decoder x8, heads"></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2Ff0758b56dbcca66501260dea12a101e1%2F05-m1-a-spec-sheet.png?generation=1790889172246342&amp;alt=media" alt="Model 1 spec sheet"></p>
<p>The 36-head model with its heads regrouped to 18 × 24. Shipped with a 4-update sprinkle on the five newest #1-team wins.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2F56122cdef94188d2216030051fe2c288%2F06-m1-e-signature.png?generation=1790889189208629&amp;alt=media" alt="Same width, half the heads"></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2F388b64d3e4c952e4c0c8b2f8cd389a9c%2F07-m1-b-parameter-map.png?generation=1790889206439751&amp;alt=media" alt="Where Model 1's 54.6M parameters sit"></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2F5caf8ae9d6871276ca9126abb48ae9de%2F08-m1-c-timeline.png?generation=1790889236903836&amp;alt=media" alt="Model 1 training timeline"></p>
<h3>Model 2: N flow</h3>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2F22dfd1ad42263d3b5dd5d9ce39b218cd%2F09-figH-arch-nflow.png?generation=1790889258947483&amp;alt=media" alt="Model 2 architecture: game state to a 320-node graph, relational blocks, three memories, decoder x2"></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2F0736806df36e9d5e62a40a782fc6e345%2F10-m2-a-spec-sheet.png?generation=1790889275819768&amp;alt=media" alt="Model 2 spec sheet"></p>
<p>Random init on one G4. Shipped with a 15-min wash: low-learning-rate SFT on top-3 wins.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2Fa8ec267d8c8c55a5cf8e11096e37dc50%2F11-m2-e-signature.png?generation=1790889294872592&amp;alt=media" alt="A graph that remembers"></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2Ff9e8d47f2a7b25dcf76850e350e30262%2F12-m2-b-parameter-map.png?generation=1790889315949475&amp;alt=media" alt="Where Model 2's 7.25M parameters sit"></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2F2fa5e87fbcbb2bcbee0a0e98ebaa18b0%2F13-m2-c-timeline.png?generation=1790889323570908&amp;alt=media" alt="Model 2 training timeline"></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2F5b8b344129759bd07cbf58e2d03eef4e%2F14-m2-d-evidence.png?generation=1790889331191666&amp;alt=media" alt="Model 2 evidence"></p>
<h2>Features</h2>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2F551c4730bbd71047874055a6168877d4%2F15-figI-feature-table.png?generation=1790889337323612&amp;alt=media" alt="Every feature family, coloured by which final model uses it"></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2F7484215df5981bb1c4c21710ae89fa88%2F16-figD-features.png?generation=1790889344644384&amp;alt=media" alt="Loss increase on earlier B/C checkpoints when each feature group is removed"></p>
<h2>Data</h2>
<p>How the SFT data was filtered and changed:</p>
<ul>
<li><strong>Winners only.</strong> Never imitate a losing action. In games we lost, train on the player who beat us.</li>
<li><strong>Weighted to fresh and strong.</strong> Hourly buckets by day and rank; the newest and strongest games get most of the draws.</li>
<li><strong>Earlier, our own ladder games too.</strong> Whatever beat our policy became training data.</li>
<li><strong>Last day: top 3 only.</strong> 30 Sep wins by the top three teams (404 source seats); the seats that beat #1 count 4x.</li>
<li><strong>Short top-ups.</strong> 15 minutes at low LR on the newest winners, exporting the current, unaveraged weights (RAW).</li>
<li><strong>Held out.</strong> Validation and sealed games were excluded early. Some validation games entered the late pool.</li>
</ul>
<p>We could only find 3 wins against the first-place solution at the time we trained, and two of them were from its own validation games!</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2F89e5795ecd196536f51c624ee01c59b5%2F17-fig4-data.png?generation=1790889348590016&amp;alt=media" alt="Winner-only training mixture and the last-day pool"></p>
<p>Replays came from Kaggle's daily episode dumps. The replay API was only for our own losses, and on the last two days for everything, at ~150-170 an hour. I also tried pulling replays by hand on the website: similar rate, but the API was less boring :D</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2F0e347106e0b10ec001c834a84757dd5f%2F18-fig5-recipe.png?generation=1790888977327942&amp;alt=media" alt="Recipe: replays, winner seats, SFT, wash, paired games, ship"></p>
<h2>Biggest lesson</h2>
<p>What I think was most useful was the workflow I refined during this competition, and will keep refining going forward.</p>
<p>Keep 2-3 ideas alive. Each has a fallback, a current best and a current experiment. Go crazy with the experiment until it beats the best, then try a new crazy experiment. Once you've experimented enough, the best becomes the new fallback. An experiment can itself be a set of ideas, so it's basically tree search over ideas.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2F9d2afb06b83dce2b99261b2279b7bb79%2F19-figF-workflow.png?generation=1790888999083652&amp;alt=media" alt="Workflow: ideas with fallback, best and experiment; this competition's idea tree"></p>
<p>Finally, we used a special version of the engine for matched games, so that shop rolls depend only on the initial seed.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2Fad615d14e60e83cc02f28fabdd38a9c7%2F20-fig7-paired.png?generation=1790888953677126&amp;alt=media" alt="Paired games on shared worlds"></p>
<h2>Code</h2>
<p>Everything needed to rebuild both final submissions from the Kaggle episodes is in this repo. The episodes themselves are not included; they come from Kaggle. See <a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/blob/main/REPRODUCE.md" target="_blank">REPRODUCE.md</a>. To jump from an idea in this write-up to the code that implements it, use <a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/blob/main/HOW-IT-WORKS.md" target="_blank">HOW-IT-WORKS.md</a>.</p>
<table>
<thead>
<tr>
<th>Folder</th>
<th>What</th>
</tr>
</thead>
<tbody>
<tr>
<td><a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/tree/main/code" target="_blank"><code>code/</code></a></td>
<td>start here to read: byte-identical copies of the ~50 files that matter, flat and grouped</td>
</tr>
<tr>
<td><a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/tree/main/quickstart" target="_blank"><code>quickstart/</code></a></td>
<td>start here to run: one replay → features → a few training updates for each model</td>
</tr>
<tr>
<td><a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/tree/main/submissions" target="_blank"><code>submissions/</code></a></td>
<td>the two final archives exactly as submitted, plus their unpacked inference code</td>
</tr>
<tr>
<td><a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/tree/main/data" target="_blank"><code>data/</code></a></td>
<td>episode download, replay processing, winner-seat selection, buckets and rosters</td>
</tr>
<tr>
<td><a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/tree/main/model1-b-even" target="_blank"><code>model1-b-even/</code></a></td>
<td>Model 1 training code and every stage from random init to the shipped checkpoint</td>
</tr>
<tr>
<td><a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/tree/main/model2-n-flow" target="_blank"><code>model2-n-flow/</code></a></td>
<td>Model 2 training code and every stage from random init to the shipped checkpoint</td>
</tr>
<tr>
<td><a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/tree/main/evaluation" target="_blank"><code>evaluation/</code></a></td>
<td>paired-game comparison on shared worlds, including the shop-fixed version, and the records behind both release decisions</td>
</tr>
<tr>
<td><a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/tree/main/figures" target="_blank"><code>figures/</code></a></td>
<td>scripts and data behind the figures in this write-up</td>
</tr>
<tr>
<td><a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/tree/main/tools" target="_blank"><code>tools/</code></a></td>
<td><code>list_features.py</code>, which generates <a href="https://github.com/definitelynotrussellkirk-bit/kaggriculture-sft/blob/main/FEATURES.md" target="_blank">FEATURES.md</a> from the code</td>
</tr>
</tbody>
</table>

---

### Message 2 by  (Votes: 0)

<p>how much time did you spend on this? great work btw</p>

---

### Message 3 by  (Votes: 1)

<p>Nice Work!</p>

---

