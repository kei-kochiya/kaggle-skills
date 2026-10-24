# Topic 749775: 26th Place Gold | Macro BC + Day Compiler

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/749775
- **Author**: 
- **Post Date**: 2026-10-16T00:01:31.627000
- **Votes**: 6

## Messages

### Message 1 by  (Votes: 6)

<h1>Macro BC + Day Compiler</h1>
<p><a href="https://github.com/PavelSavchenkov/kaggriculture/tree/main/final_handoff" target="_blank">Github</a></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3790528%2F1e2bc44d92271b367b36e6576eb8acd1%2Fsummary.png?generation=1791113611314562&amp;alt=media" alt="summary"></p>
<h2>TL;DR</h2>
<ul>
<li>Once per day, at dawn, a behaviour-cloned network outputs a <strong>DayIntent</strong>: only the macro decisions. What to plant and buy, whether to buy land, and for each group of identical crops or animals how many to water, fertilize, harvest, clear, feed, care for and collect.</li>
<li>A deterministic C++ <strong>day compiler</strong> does everything else: tiles, workers, routes, hires, purchases, when to bring goods back to the shed, funding checks, and an hourly selling plan against a learned forecast of the opponent's sales.</li>
<li>We built this in the last ~5 days of the competition with Claude Opus 5.5.</li>
</ul>
<hr>
<h2>1. The idea: separate long-term decisions from day planning</h2>
<p>We did our best to separate two kinds of decisions:</p>
<ul>
<li><strong>Long-term decisions</strong>: what the farm grows, how big the herd is, when to buy land, which crops to keep. A <strong>network</strong> makes them once per day, at dawn. It learns them from top players' games.</li>
<li><strong>Day planning</strong>: where to put new crops and animals, who walks where, how many workers to hire, order slots, when to sell the harvest. A deterministic C++ <strong>day compiler</strong> does this with exact search and dynamic programming.</li>
</ul>
<p>The test for where a choice goes: does it pay off today or on later days? A new cow gives its first milk at age 8 and a strawberry its first fruit at age 10, so how many of them to add is the network's call. A route, a tile or a sale pays off today, so the compiler works it out exactly.</p>
<hr>
<h2>2. The network: one DayIntent per dawn</h2>
<h3>2.1 The output interface</h3>
<p>The goal of the interface was to pull out <strong>every decision that shapes the macro strategy</strong> and leave every mechanical choice to the compiler. Macro decisions include what to grow, how many animals of each kind, land, which crops to keep or clear, and how much service each group gets.</p>
<p>Crops and animals that are identical except for their tile form a <strong>group</strong>. The network outputs counts per group, not per tile. The compiler chooses which members get which action. Groups are rebuilt from each dawn observation.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3790528%2F1c73f77d24643fbf00ca47cdb10f2c8c%2Fdayintent.png?generation=1791113565105976&amp;alt=media" alt="dayintent"></p>
<p>Three things made this interface work:</p>
<ul>
<li><strong>Outcomes, not actions.</strong> <code>new_cow = 2</code> means "two cows placed and alive tonight". Buying, carrying, placing and feeding them is the compiler's job. <code>retain / clear / abandon</code> says what the ongoing crop group should look like tonight, not how to get there.</li>
<li><strong>Fixed values.</strong> When an action is impossible or worthless, its field is fixed to 0 from the dawn state alone. Examples: harvesting an unripe crop, care whose bonus cannot be used before the game ends, anything new on day 29. The same mask is used in label building, training (no loss), decoding (never chosen) and compiling.</li>
<li><strong>Exact labels, no fallback.</strong> Each day of a top player is replayed in our exact engine and converted to its DayIntent by definition. The converter never guesses or clamps. A day it cannot represent stops dataset generation until the interface is fixed. On 1.45M expert days: 0 failures. Only two cases are skipped: two land quadrants bought in one day, or an animal thrown away.</li>
</ul>
<h3>2.2 Training: the non-trivial parts</h3>
<ul>
<li><strong>Data.</strong> Games of strong Kaggle submissions (final rating ≥ 2,700) plus earlier top-20 games: 29k episodes, 1.45M player-days. Later networks were fine-tuned on fresh top games up to Sep 28.</li>
<li><strong>Condition, don't filter.</strong> Every training row carries three conditioning inputs:<ul>
<li><em>Strength</em> = the submission's final rating minus a ladder reference for that date (90th percentile of top-episode ratings). Raw ratings drift by ~300 points over weeks; this makes old and new games comparable.</li>
<li><em>Recency</em> = the date of the game.</li>
<li><em>Style</em> = a one-hot of 9 top teams (M&amp;M, Majkel, DECEM, DSM, …).</li>
<li>Each input is dropped to "unknown" 10% of the time. At play time we ask for strength +150 and a recent date.</li>
<li>Filtering the data to the strongest players instead lost $11.3k per game. 13× more data with conditioning raised our local-LB win rate from 79% to 88%.</li></ul></li>
<li><strong>Opening style.</strong> Days 0–5 are decoded in DSM's style, later days with no style (all teams pooled). This won 82–38 against a Vadim-style opening, which in turn beat no style.</li>
<li><strong>Select by games, not by validation loss.</strong> Seeds of one recipe with the same loss won between 3 and 23 of 40 games against the reference network. We trained several seeds, played them, and kept the best.</li>
<li><strong>Final networks.</strong> Main network v17 (width 512, fine-tuned on fresh games) plus 4 older networks. On days 6+, their logits are averaged with the main network's, which is worth ~$1.8k per game. The main network alone decides days 0–5.</li>
</ul>
<hr>
<h2>3. The day compiler</h2>
<p>The compiler is deterministic C++ (~4,000 lines for its core). It runs at dawn, about 1–2 s and at worst 3.5 s on Kaggle's CPU, and outputs an hourly plan for every worker plus purchases and hires. The same input always gives the same game on any machine, so every game we play can be replayed exactly.</p>
<p>What it decides:</p>
<table>
<thead>
<tr>
<th>Area</th>
<th>What the compiler does</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Binding</strong></td>
<td>Which members of each group get each action. Members are interchangeable, so the router may swap them.</td>
</tr>
<tr>
<td><strong>Placement</strong></td>
<td>Tiles for new crops and animals are chosen inside the route search: a free tile, or one freed today by a harvest or a clear. Tiles near the shed are kept for animals, and each new entity pays a long-term cost for its distance to the shed.</td>
</tr>
<tr>
<td><strong>Routes</strong></td>
<td>The farmer plus up to 13 hires. Work is a set of stops (actions on one tile). Shed stops pick up feed wheat, fertilizer and animals, and drop off products. Routes are built by construction plus local search (swaps, tail exchanges).</td>
</tr>
<tr>
<td><strong>Crew size</strong></td>
<td>Hire-count search: from a lower bound up to the first complete crew, then down again, dissolving the least-loaded route. Cost = wages (1, 1, 2, 3, 5, …) + route time − deposit value + value of dropped work.</td>
</tr>
<tr>
<td><strong>Watering</strong></td>
<td>The network says how many crops in each group to water (alone or with fertilizer / harvest). The compiler picks which crops, which worker and what hour. It always waters new crops.</td>
</tr>
<tr>
<td><strong>Return-to-shed schedule</strong></td>
<td>Each unit that reaches the shed at hour h gets a value from the sale DP: when it can be sold and at what price. The router trades walking time against these values, so it searches over when each worker comes back. The values are averaged over scenarios of the opponent's sale timing.</td>
</tr>
<tr>
<td><strong>Purchases and funding</strong></td>
<td>Seeds, animals, wheat and fertilizer are bought just in time. The whole plan is played hour by hour in an engine copy with the real seller, under an expected and a stress forecast (the opponent sells all its visible goods at hour 2). Morning sales may fund same-day purchases.</td>
</tr>
<tr>
<td><strong>When money is short</strong></td>
<td>Unaffordable purchases move later. Then new crops and animals are trimmed one unit at a time. Then the fallback ladder: drop land → new entities → collection → survival. A cash floor keeps animals fed.</td>
</tr>
<tr>
<td><strong>Night</strong></td>
<td>The shed holds 100 items: cargo that cannot be stored tonight is planned away.</td>
</tr>
<tr>
<td><strong>Plan menu</strong></td>
<td>A few alternative plans (with earlier deliveries) are each played against the forecast. The best end-of-day value wins: our money + stock − the opponent's money.</td>
</tr>
<tr>
<td><strong>Market orders</strong></td>
<td>Sales are listed by revenue at stake: both players' orders fill slot by slot, so an earlier slot gets the better quote. Hires use the same 10 order slots, so a dawn sale takes a hire's slot when the crew still fits.</td>
</tr>
<tr>
<td><strong>Selling</strong></td>
<td>Every hour, see 3.1.</td>
</tr>
<tr>
<td><strong>Executor</strong></td>
<td>Follows the plan hour by hour and repairs it when an action fails or cash differs.</td>
</tr>
</tbody>
</table>
<h3>3.1 The hourly seller</h3>
<p>Per product, a dynamic program over the remaining hours of the day. It includes:</p>
<ul>
<li>the price impact of our own units;</li>
<li>town and shop drains;</li>
<li>the opponent's forecast sales, hour by hour;</li>
<li>same-turn quotes exactly as the engine computes them (both players' i-th units get the same quote);</li>
<li>the objective: our revenue minus the opponent's revenue, because a unit we sell first takes price from the opponent;</li>
<li>the value of units kept overnight: 0.95 × the expected price tomorrow morning;</li>
<li>16 opponent sale paths sampled from the forecast. This hour's lot is chosen by its mean value over them;</li>
<li>on a tie between selling now and later: sell now.</li>
</ul>
<h3>3.2 The opponent sales forecast</h3>
<p>This was our largest single component. In a 20-game head-to-head, our agent without the learned forecaster lost all 20 games to the same agent with it, by $4.2k per game on average. Removing only the intra-day head cost $2.3k per game.</p>
<ul>
<li><p><strong>The opponent's sales are not visible, so we infer them</strong> every hour: market inventory change + known town and shop drains − our actual fills. Using our requested quantities instead of actual fills turns our own failed orders into fake opponent sales.</p></li>
<li><p><strong>Model.</strong> For each game and opponent, a sequence of days. Per day and product, the inputs are:</p>
<ul>
<li>the opponent's recent sales and yesterday's 24 hourly sales;</li>
<li>its inferred shed stock and its visible output (ripe crops, products waiting on animals);</li>
<li>its herd and money.</li></ul>
<p>A causal transformer over days predicts the opponent's sales for each of the next 24 hours.</p></li>
<li><p><strong>Intra-day head</strong>: given what the opponent has sold so far today, re-forecast the remaining hours.</p></li>
<li><p><strong>Next-morning head</strong>: tomorrow's first 12 hours, used to value the stock we keep overnight.</p></li>
<li><p>3 seeds averaged at runtime.</p></li>
<li><p><strong>Timing matters more than totals.</strong> An oracle of the opponent's timing was worth +$8.3k per game, an oracle of its totals +$4.5k.</p></li>
<li><p>Chosen by games, not by held-out loss. In our second final pick, the forecaster is trained only on other teams' games.</p></li>
</ul>
<hr>
<h2>4. Validation</h2>
<p>Every comparison is paired: same world, same seed, same seat, both arms. We report:</p>
<ul>
<li>the margin (our money − the opponent's) and its standard error;</li>
<li>our money and the opponent's money separately;</li>
<li>the win rate.</li>
</ul>
<h3>Test beds</h3>
<ol>
<li><strong>Our live Kaggle games.</strong> Every Kaggle replay reproduces exactly in our C++ engine, so we can study any game turn by turn.</li>
<li><strong>Take-over continuations.</strong> Load a recorded live game at the dawn of day d and let the candidate take over our seat, while the opponent replays its recorded moves. Play 1 day, 5 days, or to the end. Before using it, the unchanged agent must reproduce the recorded game exactly (700 of 700 days did).</li>
<li><strong>Frozen leaderboard games.</strong> Put our agent into one seat of a recorded top-team game.</li>
<li><strong>Teacher days.</strong> Start from a top player's recorded dawn and give the compiler that player's own DayIntent. Then compare the compiled day with what the player did. This separates compiler problems from network problems.</li>
<li><strong>Exact local-leaderboard judge.</strong> Our team's local leaderboard, run with its own code. It reproduces the leaderboard's official games to the dollar.</li>
</ol>
<h3>Gates</h3>
<ul>
<li>Win rate plus the paired margin with its SE.</li>
<li>Sequential reading at 24 / 48 / 96 / 192 games. Reject when mean + 1.28 SE &lt; 0. Promote when mean − 2 SE &gt; 0.</li>
<li>One 40-game mirror set can produce a false winner, so we adopted a change only after ≥ 4 seed sets.</li>
<li>Beds with real opponent behaviour (live games, continuations, frozen top-team games) come first. Our own agents sell later than real teams, so beds made only of our own agents overrate early-selling changes.</li>
</ul>
<h3>Iterations came from concrete failures in games</h3>
<p>We did not tune against offline metrics. We traced lost games, found the exact cause, and fixed the compiler's model of the game:</p>
<table>
<thead>
<tr>
<th>What we saw in games</th>
<th>Cause</th>
<th>Fix</th>
<th>Effect</th>
</tr>
</thead>
<tbody>
<tr>
<td>The 3rd sheep was lost in every game</td>
<td>The day-0 budget trim dropped animals first</td>
<td>On day 0, trim crops before animals</td>
<td>+$6.6k per game</td>
</tr>
<tr>
<td>A short day lost all its new crops and animals</td>
<td>The trim was all-or-nothing</td>
<td>Trim one unit at a time</td>
<td>14 → 42 wins of 64</td>
</tr>
<tr>
<td>Animals escaped; one game lost by $186k</td>
<td>An unfunded day left no cash for feed wheat</td>
<td>Survival cash floor</td>
<td>that world: −$186k → −$7.9k</td>
</tr>
<tr>
<td>Day-0 cash left unspent, game lost by $15–20k</td>
<td>The opening plan silently dropped work</td>
<td>Keep trimming until a complete plan is funded</td>
<td>removed the cliff</td>
</tr>
<tr>
<td>The opponent got the better price in the same hour</td>
<td>Both players' orders fill slot by slot</td>
<td>List sales by revenue at stake</td>
<td>39 of 40 mirror games, +$11.6k</td>
</tr>
<tr>
<td>Dawn sales never happened on big hiring days</td>
<td>Hires used all 10 order slots</td>
<td>Reserve slots for dawn sales</td>
<td>+$0.36k on real games</td>
</tr>
<tr>
<td>The seller's expected prices were wrong</td>
<td>Our units were priced before the opponent's in the same turn</td>
<td>Price same-turn sales the way the engine does</td>
<td>+$1.6k to +$4.6k</td>
</tr>
<tr>
<td>Investments (e.g. the day-3 cow) came late</td>
<td>Goods were held for the evening, so there was no morning cash</td>
<td>Sell the morning's goods to fund same-day purchases</td>
<td>+$1.5k on real games</td>
</tr>
<tr>
<td>Products were lost at night with 4 land quadrants</td>
<td>Shed overflow</td>
<td>Plan night cargo against shed room</td>
<td>+$0.75k on real games</td>
</tr>
</tbody>
</table>
<p>Many of our gains were <strong>denial</strong>. Our fresh-data network, for example, won mostly because the opponent earned $4.6k less, while our own money changed little. So we always read our money and the opponent's money separately.</p>
<h2>Kaggriculture LocalLB: From Pull Request to Leaderboard</h2>
<p>LocalLB is a self-hosted leaderboard for comparing Kaggriculture agents. Submit an agent through a GitHub pull request, and a Docker worker handles validation, evaluation, and ranking updates automatically.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3790528%2F5504a3fc7d0a4dae9f2fd3a389df792d%2Fimagettt.png?generation=1791113715427755&amp;alt=media" alt="locallb"></p>
<ol>
<li><strong>Submit:</strong> Open a non-draft PR targeting <code>main</code>, adding exactly one new agent under <code>agents/&lt;agent_id&gt;/</code>.</li>
<li><strong>Validate:</strong> The worker checks submission rules and runs a smoke test. Passing submissions are automatically merged; ranking matches start <strong>after merge</strong>.</li>
<li><strong>Evaluate:</strong> The challenger plays the current top 10 agents using <strong>15 seeds in both seat positions</strong>, giving <strong>30 games per opponent</strong>.</li>
<li><strong>Update:</strong> Recorded results determine Elo ratings and the retained top 10. Published ratings are recomputed using only matches among those ten agents, giving each <strong>270 games</strong>.</li>
<li><strong>Publish:</strong> The worker commits and pushes match records, rankings, and the README leaderboard to <code>main</code>.</li>
</ol>
<p>The separate 30-agent leaderboard requires a manual round robin and is not triggered by PR submissions.</p>

---

