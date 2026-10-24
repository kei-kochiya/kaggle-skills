# Topic 751741: Humble Solution

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/751741
- **Author**: 
- **Post Date**: 2026-10-22T16:48:17.960000
- **Votes**: -2

## Messages

### Message 1 by  (Votes: -2)

<p><strong>TL;DR</strong></p>
<ul>
<li>Our final agent is a public agent (tetsutani's <em>Demand-Preserving Turn Sale Timing</em> notebook, Sept 27 version, Apache-2.0) plus a few re-tuned constants and a stack of small layers that mostly change the <strong>order</strong> of market orders, not the production plan. Near 2,300 about half of our opponents ran forks of that same agent family, so many games were near-identical "mirror matches" decided by a few hundred dollars. That is where our edge came from.</li>
<li>Our biggest gain came from <strong>fixing an evaluation mistake</strong>. For three weeks we tested our own agents only against opponents that never react. When we finally played them against the public agents they lost 0–20; switching to a public base took us from ≈1,160 to ≈2,290.</li>
<li>The part we are proudest of is the <strong>local evaluation harness</strong>: paired comparisons on the official engine, the seed as the unit of analysis, a rival panel weighted by a census of our real opponents, and pre-registered confirmations on fresh seeds.</li>
<li>The most interesting mechanism: both players' orders execute index by index, interleaved. That turns "buy X extra, resell X at the last index" into a <strong>near-free option</strong>: worth exactly \$0 if nobody trades in between, profitable when a mirror opponent buys in between. Late on, some opponents ran the same trade one index earlier than us; we shipped an in-game detector that switches our option off when it sees that.</li>
<li>Final: 755th of 10,148 teams, bronze medal (top 7.4%; final rating ≈1,961). Best single-submission rating during the submission window: ≈2,385 (Sept 28). Post-deadline ratings are not directly comparable (§8).</li>
</ul>
<hr>
<h2>1. Context: at ~2,300 the ladder was a mirror league</h2>
<p>Kaggriculture is a 2-player, 720-step farming game (30 days × 24 steps); the richer farm wins. By late September a large share of the ladder around 2,300 ran forks of one public agent family (tetsutani's notebook and its relatives). We call it <strong>the base family</strong> below. These agents follow precomputed <strong>routes</strong>: production-and-trading scripts chosen by which shops the seed unlocks. So two forks in the same world often issue the same orders on the same steps.</p>
<p>When we classified our real opponents from replays (see §3), <strong>≈50–52% were from the base family</strong>. Consequences:</p>
<ul>
<li>Many games are <strong>mirrors</strong>: same route, same world, nearly identical farms. In one sample of 40 real games, <strong>29 were decided by less than \$1,000</strong>, on final farms typically worth \$80k–150k.</li>
<li>What decides a mirror is <strong>who sells first, and at which price</strong>, not the production plan.</li>
<li>Win/loss is close to a threshold function of the money margin, so a layer that adds a few hundred dollars per game flips exactly the close ones.</li>
</ul>
<h2>2. The mistake that cost us three weeks</h2>
<p>Until Sept 23 we submitted our own agents: first a hand-written planner, then <strong>tapes</strong> (fixed action sequences mined from top players' public replays, plus small cash guards and shop-based branching). We evaluated them against a panel of other tapes. A tape replays blindly; it never reacts to you.</p>
<ul>
<li>The same tape-based agent won <strong>71.5% locally and ≈40% on the ladder</strong>.</li>
<li>Local improvements of +2 to +6 pp never showed up in the rating.</li>
<li>When we finally played our two active agents against public agents in full games where both sides react, they lost <strong>0–20 to each of 7 public agents</strong>, by \$12k–16k per game.</li>
</ul>
<p>Our harness had only ever compared <em>our own variants against each other</em>; it never measured absolute level against the population that actually plays. So we switched: the winner of that local tournament, submitted unchanged (arsgorynich's <em>Herd-Safe v3</em>, Apache-2.0), climbed to <strong>≈2,290</strong> within a day, versus ≈1,160 for our agents at the time (our best ever had been ≈1,325). After that we kept tracking the public frontier; on Sept 27 tetsutani's agent overtook it and became our final base. Everything after that is tuning and layers on top of it.</p>
<h2>3. How we evaluated (the harness)</h2>
<p>Our tournament script plays full games on the <strong>official engine</strong> (<code>kaggle_environments</code> 1.32.7), loading each agent file exactly as Kaggle does (last callable in the module).</p>
<table>
<thead>
<tr>
<th>Concept</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Seed</strong></td>
<td>Fixes the game's randomness: which town shops unlock (one every 3 days, up to 8, drawn with replacement), which decides which products are valuable (e.g. a yarn shop makes wool valuable), and where weeds spawn. Same seed + same two agents + fixed <code>PYTHONHASHSEED</code> → the exact same game.</td>
</tr>
<tr>
<td><strong>Seat</strong></td>
<td>Playing as player 0 or 1. We always play both.</td>
</tr>
<tr>
<td><strong>Cell</strong></td>
<td>(rival, seed, seat). Candidate and reference play <strong>the same cells</strong>, so each cell gives a paired difference with the world held constant.</td>
</tr>
</tbody>
</table>
<p>Notation below: <strong>pp</strong> = percentage points of win rate; brackets = 95% CI computed over seeds; a <strong>changed cell</strong> = a cell whose win/loss differs between candidate and reference.</p>
<p>Design choices that mattered:</p>
<ol>
<li><strong>Win/loss is the primary metric</strong>, because the rating only sees W/L/T. Money margin is a secondary, far more sensitive signal for mirror games.</li>
<li><strong>The seed is the unit of analysis, not the cell.</strong> Both seats of a seed are often literally the same game seen from the other side, so counting every cell as an independent game would be fake precision.</li>
<li><strong>A panel weighted by a census of the real population.</strong> We downloaded our ladder replays, fed each public agent the exact observations our real opponent saw, and counted identical actions to classify every opponent into a family. Weighted deltas estimate the effect on the population we actually face. Hard rule: reject a change if any single family gets clearly worse (95% CI entirely below −10 pp).</li>
<li><strong>Pre-registration and confirmation on fresh seeds.</strong> Variants, rivals, seeds, metric and pass threshold were written down before each confirmation run (sometimes after a small screening pilot). Why: in one pilot of 111 candidates, the winner scored <strong>87.5% and fell to 31%</strong> against a fresh set of opponents and seeds (winner's curse). Not every shipped layer cleared its adoption threshold: several shipped under a weaker pre-declared "promising" rule (point estimate ≥ 0, no family harmed, margin CI &gt; 0), so we also measured each combined agent directly against the previous one (§6).</li>
<li><strong>Controls that the instrument isn't lying:</strong><ul>
<li><strong>Action hashes</strong> per seat: if a layer must not change behavior in some situation, we check it byte for byte.</li>
<li><strong>Determinism:</strong> one whole confirmation was invalidated because a <code>sorted(set(...))</code> without a tie-break depended on Python's hash seed. Since then everything runs with <code>PYTHONHASHSEED=0</code>. (On Kaggle each agent process gets its own hash seed, which made some "identical" agents diverge.)</li>
<li><strong>Fidelity:</strong> real Kaggle replays reproduced exactly in the local engine (719 actions / 720 states).</li>
<li><strong>Does the change even fire?</strong> In a 24-variant parameter sweep (12 constants, two values each), <strong>18 variants were exactly inert</strong>: those constants are read at import time, so our end-of-module override came too late. We now check in order: does the code path run, does it change the emitted action, does the engine execute it, does it have the intended market effect, and only then do we measure the result.</li></ul></li>
</ol>
<h2>4. The layers</h2>
<p>Each step a player submits a list of up to 10 orders (indices 0–9): market buys and sells plus fixed-price orders such as hiring or buying seeds. The engine executes <strong>index 0 of both players, then index 1, and so on</strong>; within an index the two players' units alternate one at a time. So whatever you put at index 9 executes after everything your opponent put at indices 0–8. Every unit sold lowers the price of the next one and every unit bought raises it; curves are steep for milk, wool and strawberries (≈\$2 per unit) and almost flat for wheat and fertilizer (≈\$0.2 per unit). A buy is quoted at inventory − 1, so buying X and reselling X in the same step nets exactly \$0 if nobody trades in between.</p>
<p>None of this lockstep logic is new: the base already uses it to order its sales against a mirror (credited in the code to Seyit Kaan Gunes and shiiin9). Our additions, on top of the base:</p>
<p><strong>Re-tuned constants</strong> (each confirmed separately): a forecast top-k from 1 to 3 (18–4–10 head-to-head against the unmodified base); a sales look-ahead and a sell margin (together +8.4 pp [+4.3; +12.5]); and a wheat→carrot conversion margin of −22 (+5.3 pp [+1.2; +9.5]), the same value haodou092 had published in <em>Harvest Ledger</em>. That last one is a production change; everything else below changes <strong>order, not production</strong>:</p>
<table>
<thead>
<tr>
<th>Layer</th>
<th>What it does</th>
<th>Evidence</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Front-run defense</strong></td>
<td>In ≈11% of games the route makes one big wheat purchase at a known step. Two opponents bought that wheat one step ahead of us and sold it at index 9, right after our purchase had pushed the price up (−\$8k to −\$11k per game for us). We now buy one step earlier; if the opponent follows the same route, we also buy extra and resell it at index 9. A pre-sell variant sells one step early the wheat the route will sell next step.</td>
<td>+37.2 pp [+26.3; +48.2] in those games (51/51 changed cells in our favor); pre-sell +\$1,934 per game</td>
</tr>
<tr>
<td><strong>Mirror-aware sell order</strong></td>
<td>One extra call of the base's existing best-response sales reorder (shiiin9's "counter D") before our front-run layer</td>
<td>W/L +0.3 pp (CI includes 0); margin +\$80 [+51; +110] vs the base; census-weighted replication +7.3 pp [+1.4; +13.2] †</td>
</tr>
<tr>
<td><strong>Shepherd delivers first</strong></td>
<td>The base's sheep worker picked up fertilizer before delivering wool, so its wool reached the market 3 steps later than some other families'. Wool's price falls fast as units are sold, so it now delivers first.</td>
<td>+2.2 pp [−1.5; +5.8]; on seeds with a yarn shop +2.5 pp [−0.4; +5.5]; 13/13 changed cells in favor; 0/1,120 cells lose margin †. On the ladder: 6–0 in games where it built a 6-sheep flock</td>
</tr>
<tr>
<td><strong>Buys before fixed-price orders</strong></td>
<td>Wheat/fertilizer buys move ahead of hiring, seeds, animals and land (still behind our sales). Small on its own, but it puts our buy early in the list, which the option below relies on.</td>
<td>+\$13 to +\$19 margin vs every family (CI &gt; 0); 5/5 changed cells in favor †</td>
</tr>
<tr>
<td><strong>Resale option</strong> (buy side + sell-side twin)</td>
<td>Add X (≤ 90) to our routine wheat/fertilizer buy and resell X at index 9. If the opponent buys the same product in between, it pays the higher price our extra buy created, and its purchase pushes the price up further before our resale, so we sell X for more than we paid. If nobody trades in between, the round trip nets exactly \$0. The sell-side twin sells X extra from stock and buys it back at index 9.</td>
<td>+3.44 pp [+0.95; +5.92]; margin +\$396 per game vs the base family; 0/672 cells with negative margin</td>
</tr>
</tbody>
</table>
<p>† Shipped under the weaker "promising" rule (§3): the win-rate CI included 0, but no family was harmed and the margin CI was above 0.</p>
<p>The versions we refer to below:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>What it is</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Base</strong></td>
<td>tetsutani's public agent (Sept 27 version)</td>
</tr>
<tr>
<td><strong>A</strong></td>
<td>Base + re-tuned constants + front-run defense + mirror-aware sell order</td>
</tr>
<tr>
<td><strong>B</strong></td>
<td>A + shepherd-first + buys before fixed-price orders + resale option (both sides)</td>
</tr>
<tr>
<td><strong>B+guard</strong></td>
<td>B + the guard from §5</td>
</tr>
</tbody>
</table>
<h2>5. Opponents who raced the option, and the guard</h2>
<p>Among the last 33 ladder opponents of B, 8 ran the same trade <strong>one index earlier</strong>: buy X extra at index 0 and resell it at indices 1–8, before our index 9. They won that race and took <strong>\$383 to \$5,914 per game</strong> from us; we lost all 8 games. We cannot tell whether they learned it from our replays or found it independently. We found no public notebook containing it (we checked the 13 notebooks updated on Sept 28–29), so it was presumably private code.</p>
<p><strong>The guard.</strong> After each step in which our resale option was active, we re-simulate that market step exactly, assuming a "benign" rival: at most one order per product, at any index, with quantities that reproduce the inventory we observed. That gives the least money we could have ended the step with against any benign rival. If we actually ended with less (by more than \$150 once, or by more than \$20 on buy-side steps / \$60 on sell-side steps twice), the rival must have traded the same product in both directions around our orders, so we switch the option off for the rest of the game. When it does not fire, its actions are identical to B's.</p>
<table>
<thead>
<tr>
<th>Test</th>
<th>Result</th>
</tr>
</thead>
<tbody>
<tr>
<td>Pre-registered confirmation vs 4 synthetic, non-reactive copiers (two closely reproduce real opponents' patterns)</td>
<td><strong>+44.9 pp [+38.2; +51.2]</strong>; the guard fired in 768/768 cells</td>
</tr>
<tr>
<td>No false alarms: 7-agent panel, and a wider check vs 30 public agents + self-play</td>
<td>0/672 and 0/480 cells with any action difference</td>
</tr>
<tr>
<td>13 copier variants in total</td>
<td>B+guard beat B against every one</td>
</tr>
<tr>
<td>Games not used to build the guard: 191 real ladder games of A, re-run with B+guard against each opponent's recorded (non-reactive) actions</td>
<td>+\$142 per game; 1 loss → win, 0 wins → losses; 1 false alarm (cost \$73)</td>
</tr>
</tbody>
</table>
<p>We also tried a <strong>counter-attack</strong>: after detection, buy at index 0 and resell at index 1, ahead of the other side's resale. Three versions were refuted in two rounds by an adversarial review step (an AI agent tasked with breaking each design): first with a "staircase" pattern taken from a real opponent, then with a fixed-chunk pattern. Each time it lost to the simple guard (in one case 10 wins turned into losses). It is an arms race over indices; the bounded-downside option (switch off) was the robust one.</p>
<h2>6. Choosing the final pair, and did the harness transfer?</h2>
<p>Kaggle scores only your two most recent submissions. On the last day A and B were on the ladder, and B+guard had been confirmed locally.</p>
<p><strong>Did the harness transfer?</strong> It predicted B &gt; A by <strong>+7.55 pp [+3.80; +11.31]</strong> (29 of 384 cells changed outcome, all 29 in B's favor). On the ladder, over the same time window, B went <strong>23–15</strong> and A <strong>8–26</strong>. A logistic (Elo-style) model using each opponent's current rating put the gap at roughly <strong>+92 to +135 rating points</strong>.</p>
<p>The day before the deadline, before seeing new data, we wrote a decision rule: replace A only if B led it by ≥ 40 rating points (or clearly out-won it in games played after the rule was written), with a recent win rate of at least 45% and no excess of errored games, and with a fallback for a failed validation that could never push B out. The rule was met (+97 points). We then submitted B+guard, which had not played a single ladder game, and kept B. We accepted that because, unless the guard fires, its actions are identical to B's (0 differences in 1,152 local cells against up to 30 public agents). Swapping in the guarded file was an amendment made after its local confirmation; the rule itself was fixed before the new data.</p>
<h2>7. What did not work</h2>
<table>
<thead>
<tr>
<th>Idea</th>
<th>Result</th>
</tr>
</thead>
<tbody>
<tr>
<td>Selling leftover wool/milk/strawberries early (changes quantities sold)</td>
<td>−2.8 pp [−6.4; +0.8] despite +\$107 mean margin: extra units sold into close games</td>
</tr>
<tr>
<td>A tomato-instead-of-cow opening</td>
<td>−80 pp</td>
</tr>
<tr>
<td>Replacing our base with a full public package from another lineage (in our panel, at that time)</td>
<td>−64 pp</td>
</tr>
<tr>
<td>Letting the herd model count demand from shops not yet unlocked (it swapped geese for cows)</td>
<td>−84 pp</td>
</tr>
<tr>
<td>Buying first at index 0 (pushes our sales behind the mirror's)</td>
<td>−\$400 to −\$1,000 per game</td>
</tr>
</tbody>
</table>
<p>The base was already strong on production. Apart from one small production retune (the −22 margin above), the edge we found was in sale timing and market order.</p>
<h2>8. What we could not solve</h2>
<ul>
<li><strong>A private family with a cow opening</strong> (≈6% of opponents near the end): A and B, our two active agents at the time, both went 1–7 against it. It bought its second land plot ≈70 steps earlier than us and scaled production differently. That is a production-plan gap, and our panel could not contain it because the code was not public.</li>
<li><strong>The post-deadline drop.</strong> During the final evaluation (until Oct 14) our two final agents slid from ≈2,200 to ≈1,924–1,961. After the deadline every team's two final submissions keep playing each other and ratings across the ladder re-settle; we cannot tell how much of our drop is a stronger pool and how much a shifted scale. Either way, our harness measured improvements relative to what we could see, not the distance to what we could not.</li>
</ul>
<h2>9. Lessons</h2>
<ol>
<li>Measure absolute level against the best available alternative before tuning.</li>
<li>Your offline metric is a proxy: calibrate it against production, and refresh it when it saturates (by the end we were winning ≈95% of the public panel).</li>
<li>Most "improvements" are noise: pair your comparisons, pick the right unit of analysis, pre-register, confirm on fresh data.</li>
<li>Verify that a change actually fires before measuring its effect.</li>
<li>The best hypotheses came from forensic analysis of lost games, not brainstorming.</li>
<li>Success changes the environment: once an edge works, someone else runs it too. Prefer changes with bounded downside.</li>
</ol>
<h2>Credits</h2>
<p>Our final agent is the code published in tetsutani's <em>Demand-Preserving Turn Sale Timing</em> notebook (Sept 27 version, Apache-2.0) with our changes on top. That code carries attribution to thomastschinkel, yhay81, destbreso, aurax7, prvsiyan, Dmitrii Gluzdov, Ahmed Berat Ozer, Seyit Kaan Gunes (lockstep sale ordering against clones), shiiin9 (the best-response reorder against a copy of ourselves, which our sell-order layer re-applies), lucifer19, Moon and the kaggle-environments contributors. Along the way we also submitted, unmodified, arsgorynich's <em>Herd-Safe v3</em> (the agent that first took us to ≈2,290), thomastschinkel's <em>Metav4 v13</em> and Abhinav0370's <em>cha22</em> agent. Our wheat→carrot margin of −22 is the value haodou092 published in <em>Harvest Ledger</em>. All license notices are retained in our submission. Thanks to everyone who published agents and notebooks: this competition was a great example of open iteration.</p>
<p>github: <a href="https://github.com/rospeltof/kaggriculture" target="_blank">https://github.com/rospeltof/kaggriculture</a></p>

---

