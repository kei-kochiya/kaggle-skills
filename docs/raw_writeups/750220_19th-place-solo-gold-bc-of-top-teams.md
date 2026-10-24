# Topic 750220: 19th place (solo gold): a behaviour clone of the 1st, 3rd and 7th place teams

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/750220
- **Author**: 
- **Post Date**: 2026-10-17T14:48:34.523000
- **Votes**: 5

## Messages

### Message 1 by  (Votes: 5)

<h2>Summary</h2>
<p>My final agent (MXFMHF12D) finished 19th of 10,148 teams with a final score of 5,210.3. It is a behaviour clone of the teams that finished 1st, 3rd and 7th: M &amp; M &amp; P &amp; Q, DSM and DECEM. Twelve small networks trained on their replays (and on older games in slots the agent does not play from) each score every move, the agent averages their probabilities, and three rules sell goods the engine would otherwise destroy. No RL went into it, and it does no search at play time.</p>
<p>Four things I measured that may be useful next to Farmcore's and DECEM's write-ups:</p>
<ol>
<li>Pooling three teams that play the same four-quadrant plan beat a clone of M &amp; M &amp; P &amp; Q alone against all seven opponents I tested.</li>
<li>Training from scratch and keeping the last epoch beat fine-tuning older networks and beat picking epochs by validation loss.</li>
<li>Averaging networks from families trained on different views of the data beat adding more networks of one family.</li>
<li>A local test that ranked my live files correctly still could not see where my clones lose.</li>
</ol>
<p><img src="https://www.kaggle.com/writeup-links/200797/images/original" alt="How the final agent was built: replays of the 1st, 3rd and 7th place teams, one record per teacher seat, twelve networks in three families, and their averaged probabilities plus three sell rules at play time"></p>
<h2>Data</h2>
<p>For each teacher I downloaded the replays of its current submissions through Kaggle's episode list endpoint, every 30 minutes on the last day. By the morning of 30 September I had 653 M &amp; M &amp; P &amp; Q games, 578 DSM games and 549 DECEM games, plus about 1,900 games from 12 older M &amp; M &amp; P &amp; Q submissions.</p>
<p>Each training record is one teacher seat in one game, seen from that seat. The labels are the orders the teacher requested, with market quantities clipped to 30.</p>
<p>One bug mattered. BUY_PRODUCT orders above 30 were not clipped, so any record that contained one was dropped. DSM and DECEM often buy wheat in bulk around day 10, so until a fix on the last day more than half of their games (316 of 589) never reached training. Farmcore relabelled to the trades the engine actually executed, and that is what I would add first.</p>
<h2>Model</h2>
<p>Each network is a small CNN with 898,498 parameters. The board runs through three residual blocks with the global features added to every tile, and each unit's 132 features join at the unit head. The unit head picks each unit's verb and quantity, and an autoregressive market head writes up to ten orders after seeing what the units will do to the shed. Every network trains from scratch with AdamW (batch 256, peak learning rate 1e-3) and a OneCycle schedule that anneals to zero, for 10 epochs at first and 16 later, and I keep the last epoch.</p>
<p>The trainer has three version slots. The teachers' newest games sit in the slot the agent plays from, and older games (Boey's, and older M &amp; M &amp; P &amp; Q submissions in some families) sit in the other two, as far as I can reconstruct from the trainer. The network learns from old games without being asked to play like them, the same idea as DECEM's date buckets.</p>
<p><strong>Decoding.</strong> Units take the argmax of the averaged distribution, with a legality mask so a later unit sees what earlier units took. Purchases need more care: plain argmax never makes a purchase the teacher makes with a 20 % chance per step. Error diffusion fixes that. An accumulator adds the expected purchases each step and buys when it reaches one. On my first clone, error diffusion and a later refinement were worth 15k to 20k coins per game; the final file uses plain error diffusion. DECEM describes a similar accumulator.</p>
<p><img src="https://www.kaggle.com/writeup-links/200805/images/original" alt="Inside one network: the board, the global features and the version slot feed a convolutional trunk with three residual blocks, and a unit head that also sees each unit's own features, a market head and a value head read from it; below, an example where argmax never buys and error diffusion buys three times in fifteen steps"></p>
<h2>What mattered</h2>
<p>All results are paired: a candidate and a control play the same towns against the same opponent, with a dithered opening, and I report the mean difference in final margin. ± is one standard error. A final bank is roughly 90k to 160k coins.</p>
<p><img src="https://www.kaggle.com/writeup-links/200794/images/original" alt="Three paired results: fresh training with the last epoch beat fine-tuning, pooling three teachers beat one against seven opponents, and my final files with three families scored above the one-family clone on the local score V1"></p>
<p><strong>Train from scratch, and keep the last epoch.</strong> Eight fresh networks beat my live clone of Boey (23rd) by 11.0k per game summed over three opponents. Fine-tuning older networks on the same data gave 0.9k, or 3.5k with the newest games weighted up. The same eight networks at their lowest-validation-loss epoch gave 7.4k. The agent plays argmax, so validation loss is the wrong yardstick.</p>
<p><strong>Pool three teachers that share a plan.</strong> DSM and DECEM play the same four-quadrant skeleton as M &amp; M &amp; P &amp; Q: land, hires, a large wheat ring and animals. Alone they had too few games for a good clone. Pooled with M &amp; M &amp; P &amp; Q, they beat the M&amp;M-only clone by 3.9k ± 0.8k summed over three standard opponents, and by 1.2k to 4.1k against four others. Farmcore found that pooling many top players was worse than cloning one team, and they explain it by plans that stop fitting together. My three teachers share one plan, which may be why pooling worked; I did not measure how often they agree.</p>
<p><strong>Families beat more networks.</strong> A family is a set of networks trained with one recipe on one view of the data. More networks within a family did nothing measurable, while a second family lifted my local score V1 (see Validation) from 0.705 to 0.784. My final file (Final A) averages three families, MXNF2 ×3, MXNM ×6 and MXHF ×3, and scored 0.879 on 700 towns, against 0.847 for my other final file (Final B) and 0.700 for a clone of the teachers' newest games. Part of the lead is selection: the same recipe with new seeds scored 0.835, and the two final files ended 24 points apart in the final standings.</p>
<p><strong>Three small rules.</strong> OVERFLOW sells what will not fit in the shed at midnight, ENDGAME stops investments that cannot pay back and sells everything at the end, and ENDRUSH sends units carrying goods to the shed before the last step. Leaving one out cost 1,722, 676 and 252 coins per game, summed over three opponents.</p>
<h2>Validation</h2>
<p><strong>Dither the opening.</strong> Days 0 to 2 have no shops and weeds start at step 48, so two deterministic programs play the same opening in every town, and a test measures one opening plus noise. Against a dithered opponent, a null change read −677 ± 1,290 on 50 towns instead of −11,389 ± 2,140 on 15. Seeing 1k at two standard errors then takes about 340 towns.</p>
<p><strong>V1</strong> = ½ × mean(win rate against a strong M&amp;M copy, win rate against the M&amp;M-only clone) + ½ × win rate against the pooled clone. Across five live files its two copy cells correlated 0.93 and 0.94 with the live ratings, and the final standings ordered my two final files the way V1 did.</p>
<p><strong>What V1 missed.</strong> Its references are my own clones, so they share my blind spots. In 18 replays against teams rated above 2,550, three of the six losses came from too few strawberries and too little milk, and in wins and losses alike my agent left 16 to 22 more animal-days unfed per game than M &amp; M &amp; P &amp; Q did against the same opponents. In M &amp; M &amp; P &amp; Q's own positions it picks their exact action 0.91 to 0.99 of the time, but in free play only 0.26 to 0.40 of positions match from day 6. The money is lost in states the teacher never visits.</p>
<h2>What did not work</h2>
<table>
<thead>
<tr>
<th>Idea</th>
<th>Result</th>
</tr>
</thead>
<tbody>
<tr>
<td>PPO from an early clone (6.0M steps)</td>
<td>its own sampled metric rose 3.0k; the deterministic agent I ship lost 23.3k</td>
</tr>
<tr>
<td>Fine-tuning a clone on its own best rollouts</td>
<td>−5.0k</td>
</tr>
<tr>
<td>DSM's recorded opening, then my clone</td>
<td>−22.5k against one opponent</td>
</tr>
<tr>
<td>Sending units to rescue drying plants</td>
<td>−1,756 against one opponent</td>
</tr>
</tbody>
</table>
<p>The PPO run was small, so it says nothing about PPO at the scale the winners used.</p>
<h2>Thanks</h2>
<p>Thanks to Kaggle for the game, and to M &amp; M &amp; P &amp; Q, DSM and DECEM, whose games my networks learned from. Thanks also to Boey, whose games taught my earlier clones and are still in the final networks' training data, and to the authors of the public notebooks I submitted or forked:</p>
<ul>
<li>Thomas Tschinkel's <a href="https://www.kaggle.com/code/thomastschinkel/kaggriculture-93-8-win-rate-public-state-router" target="_blank">Public State Router</a></li>
<li>WUZB's <a href="https://www.kaggle.com/code/wzhengbiao/kaggriculture-v15stack-submit" target="_blank">v15stack</a></li>
<li>Ahmed Berat Özer's <a href="https://www.kaggle.com/code/ahmedberatozer/kaggriculture-v57-funding-order-invariant" target="_blank">V57</a></li>
<li>prvsiyan's <a href="https://www.kaggle.com/code/prvsiyan/kaggriculture-frontier-the-moon-counts-melons" target="_blank">The Moon Counts Melons</a> and <a href="https://www.kaggle.com/code/prvsiyan/kaggriculture-frontier-the-soil-remembers-rain" target="_blank">The Soil Remembers Rain</a>, whose controller builds on haideptry's <a href="https://www.kaggle.com/code/haideptry/the-2965-master-hybrid-engine" target="_blank">The 2965 Master Hybrid Engine</a></li>
</ul>

---

