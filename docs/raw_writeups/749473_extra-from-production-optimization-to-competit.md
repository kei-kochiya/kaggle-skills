# Topic 749473: From Production Optimization to Competitive Farming: My Kaggriculture Solution

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/749473
- **Author**: 
- **Post Date**: 2026-10-15T04:56:24.503000
- **Votes**: 2

## Messages

### Message 1 by  (Votes: 2)

<h3>At a glance</h3>
<p>My final agent combines a reliable production schedule extracted from game replays with two kinds of adaptation: an opponent-aware seller that decides when and how much to sell, and crop/livestock choices that preserve an already-tested execution schedule. Rules coordinate farm execution, while a small learned model estimates the opponent's future sales.</p>
<p>The central lesson was a change in the question I asked. I initially asked, <strong>“How can I produce more, or finish with more money?”</strong> Eventually, the more useful question became, <strong>“How does this decision change the money difference between my opponent and me?”</strong> In a shared market, those questions can have different answers.</p>
<p>The final submission, <strong>56707930</strong>, combines synchronized crop substitutions, adaptive livestock selection, and forecast-guided selling. Adding livestock selection improved mean cash difference by 1,770 across 18 recorded-opponent scenarios in the local confirmation test. The October 15 online snapshot reported a score of <strong>2226.6</strong> and <strong>535 wins in 925 completed public games (57.84%)</strong>. Sections 8 and 10 describe the local and online results separately.</p>
<h2>1. Why farming was also a competitive market problem</h2>
<p>Kaggriculture couples a farm-management problem to a shared-market problem. Workers must carry out feasible sequences of movement, planting, watering, feeding, harvesting, and delivery. Meanwhile, both players sell into the same market, where supply changes prices and town shops consume products. The game therefore rewards more than a large production total. Timing, product mix, working capital, and the opponent's supply all matter. The official implementation is the reference for action ordering, production, and market settlement. <a href="https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kaggriculture/kaggriculture.py" target="_blank">1</a></p>
<p>For analysis, I used terminal money difference:</p>
<pre><code>margin = our terminal cash - opponent terminal cash

change in margin = change in our cash - change in opponent cash
</code></pre>
<p>A move can increase our cash but increase the opponent's cash even more. Equally, two agents can produce the same goods and finish with different money because their selling decisions interact.</p>
<p>There are also two distinct notions of success:</p>
<ul>
<li><strong>Execution success:</strong> the intended production and delivery occur without unrecovered disruption.</li>
<li><strong>Competitive success:</strong> that production, sold against this opponent in this market, wins the game.</li>
</ul>
<p>Much of the project consisted of learning not to confuse the first with the second.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F31328756%2F909a0f4f3ccf8860f5508e8cefb7a5fb%2Ffig_shared_market.svg?generation=1792035136046328&amp;alt=media" alt=""></p>
<p><em>Figure 1. Each farm has its own production plan, but both players' sales affect prices in the shared market.</em></p>
<h2>2. The research path: what changed, and why</h2>
<p>The project used public high-scoring replays as demonstrations and sources of production plans. Kaggle's Daily Top Episodes discussion provided access to the replay collection used in the early imitation-learning and route-search experiments. Section 7 gives the training set used for the final market predictor. <a href="https://www.kaggle.com/competitions/kaggriculture/discussion/731215" target="_blank">2</a></p>
<table>
<thead>
<tr>
<th>Stage</th>
<th>Main idea</th>
<th>What it taught me</th>
</tr>
</thead>
<tbody>
<tr>
<td>High-throughput fixed schedules</td>
<td>Reuse strong planting, livestock, and worker schedules.</td>
<td>A reliable schedule is valuable, but it may pursue nearly the same business regardless of market conditions.</td>
</tr>
<tr>
<td>Daily-target imitation and general executors</td>
<td>Predict a desirable farm state, then ask an executor to reach it.</td>
<td>Accurate state prediction is not enough. Targets may be jointly infeasible, or the executor may miss the resource and timing dependencies needed to realize them.</td>
</tr>
<tr>
<td>Replay-derived decision forests</td>
<td>Join compatible daily farm states and choose among prerecorded production continuations.</td>
<td>A shared farm snapshot is not the entire execution state. Inventory, bags, worker identities, hire timing, and unresolved tasks must also remain compatible.</td>
</tr>
<tr>
<td>Node-selection RL and large route screening</td>
<td>Learn which branch to take; screen many routes for stability and self-play cash.</td>
<td>A large action space is not automatically a useful one. The selector cannot invent a winning production option, and a high self-play cash total does not establish strength against strong opponents.</td>
</tr>
<tr>
<td>Market-first competition, then constrained production choices</td>
<td>Freeze a strong executor, improve trading against opponents, and add production alternatives checked for execution compatibility.</td>
<td>This made changes easier to attribute and allowed useful adaptation without redesigning the whole farm.</td>
</tr>
</tbody>
</table>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F31328756%2F789a87a612a6b234b20e0b541019632c%2Ffig_research_path.svg?generation=1792035215877554&amp;alt=media" alt=""></p>
<p><em>Figure 2. Each research stage changed what I optimized: execution accuracy, route flexibility, competitive trading, and finally a small set of reliable production choices.</em></p>
<h3>2.1 Why good imitation metrics did not settle the execution problem</h3>
<p>An early daily-target experiment used 25,923 distinct games and about 1.56 million player-day examples, each describing one player's farm for one game day. Its offline tile-state accuracy could look strong while actual farming remained weak. Most tiles often remain unchanged, so aggregate accuracy can hide a few critical errors: one missing seed, one late watering action, or an impossible combination of otherwise plausible targets.</p>
<p>There was also a planning issue. Independent tile predictions could mix fragments of different strategies without preserving a coherent resource budget. Some early scheduling logic was too all-or-nothing: when a complete daily plan was infeasible, it could fail to execute useful partial work. More training did not, by itself, repair that interface.</p>
<p>The lesson was not that imitation learning cannot work. It was that <strong>the learned representation and the executor must agree about what can actually be executed</strong>.</p>
<h3>2.2 Why larger decision forests were not the final answer</h3>
<p>The forest approach tried to make production flexible without learning every worker action. Each daily segment inherited a detailed action schedule; compatible states allowed a switch to another episode's continuation. Procurement and hiring were reconstructed around the selected schedule.</p>
<p>This exposed genuinely useful engineering problems: replay action alignment, random weeds, just-in-time purchases, worker spawn positions, order limits, and delayed recovery after a cash shortage. Solving those problems improved the eventual executor.</p>
<p>However, the forest also exposed a strategic limitation. More branches could mean many nearly identical production profiles, not more meaningful choices. Branches could cluster early, before informative shops opened, or an apparently high-yield route could require an unaffordable opening. A rule or RL selector could then collapse to one route because the remaining choices were weak or insufficiently differentiated.</p>
<p>A particularly instructive intermediate tree had three stable routes. Across nine difficult recorded-opponent scenarios, all 27 route/scenario runs completed production, but none won. The rule selector chose the same route in all nine scenarios. Execution had become reliable, but the available production choices were still too weak.</p>
<p>These experiences led to a narrower design: preserve a strong production backbone, diagnose the market independently, and add small, economically meaningful freedoms whose execution consequences could be checked.</p>
<h2>3. The turning point: measure interaction, not just output</h2>
<h3>3.1 Freeze the farm to isolate the market</h3>
<p>The first market-competition experiment held production code fixed and compared three sellers on 48 scenarios, each defined by a random seed and an opponent. Each scenario was played twice, swapping player positions. Six opponent types included three with the same production and three older agents with different production. Each seller played 96 games; the comparison contained 288 games in total.</p>
<table>
<thead>
<tr>
<th>Seller</th>
<th>Mean own cash</th>
<th>Mean margin</th>
<th>Win rate, ties worth half</th>
</tr>
</thead>
<tbody>
<tr>
<td>Baseline rule-based seller</td>
<td>106,710</td>
<td>+3,148</td>
<td>62.5%</td>
</tr>
<tr>
<td>Opponent-aware revenue search</td>
<td>107,171</td>
<td>+6,388</td>
<td>89.6%</td>
</tr>
<tr>
<td>Revenue search with an opponent-income penalty</td>
<td>105,249</td>
<td>+5,772</td>
<td>83.3%</td>
</tr>
</tbody>
</table>
<p>The opponent-aware revenue search improved margin by approximately 3,241: our cash rose by about 462 while the opponent's fell by about 2,779. Production did not increase. In the same-production subset, win rate moved from 45.8% to 100%; against the three different-production agents, all three sellers had the same 79.2% win rate, despite differences in average margin.</p>
<p>The negative control mattered. Adding a stronger explicit penalty on the opponent's revenue did <strong>not</strong> produce the best seller. A short-horizon model can misprice the cost of hurting both players. The best local result came from anticipating the opponent while scoring our own revenue conservatively.</p>
<p>Thus, the shift toward competition was a shift in <strong>observations, experiments, and acceptance criteria</strong>, not simply the addition of a negative opponent-reward term.</p>
<h3>3.2 The same production can win or lose through trading</h3>
<p>In one difficult replay-based scenario, the environment preserved a recorded opponent's production while both agents traded at live market prices. Changing only our seller produced the following outcome. The opponent sold products immediately, and production quantities on both sides were unchanged.</p>
<table>
<thead>
<tr>
<th>Our seller</th>
<th>Our cash</th>
<th>Opponent net cash</th>
<th>Margin</th>
</tr>
</thead>
<tbody>
<tr>
<td>Opponent-aware revenue search</td>
<td>133,711</td>
<td>136,662</td>
<td>−2,951</td>
</tr>
<tr>
<td>Finer search with rule-based supply estimates</td>
<td>130,918</td>
<td>136,610</td>
<td>−5,692</td>
</tr>
<tr>
<td>Forecast-guided seller</td>
<td>135,893</td>
<td>134,518</td>
<td>+1,375</td>
</tr>
</tbody>
</table>
<p>The forecast-guided seller improved our cash by 2,182 and reduced opponent cash by 2,144: a 4,326 improvement in margin and a loss-to-win reversal.</p>
<p>On a separate confirmation test, the official engine ran both complete agents across 48 new seed/opponent scenarios, with 96 games per version. The forecast-guided seller improved mean margin over the opponent-aware revenue search by approximately 724; win rate rose from 91.67% to 93.75%. The paired-scenario interval for the margin gain was approximately +406 to +1,048. This test used six fixed local opponent types and fresh seeds.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F31328756%2F2dda2a6aa084be4a65d84ead72e0ba2e%2Ffig_cash_decomposition.svg?generation=1792035288741439&amp;alt=media" alt="Our cash gain and the reduction in rival cash add to the market-only improvement in margin."></p>
<p><em>Figure 3. In this example, replacing rule-based supply estimates with a learned forecast raised our cash by 2,182 and reduced rival cash by 2,144, producing a 4,326 margin gain.</em></p>
<h3>3.3 Why farming still had room to improve</h3>
<p>Market improvements did not erase every production disadvantage. Against nine strong recorded production schedules, each paired with two opponent selling policies and both player positions, the forecast-guided seller won only 2 of 36 games, with mean margin −15,593. The preceding seller won none.</p>
<p>This motivated the next question: <strong>can the farm change its product mix without losing the execution reliability that took so long to build?</strong></p>
<h2>4. Final architecture: reliable execution with bounded adaptation</h2>
<p>The final agent retained one tested production schedule extracted from a strong replay. It added a small set of crop and livestock alternatives whose complete work sequences had been checked for compatibility. The diagram below follows the full loop from observation to production decisions, worker execution, trading, and feedback.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F31328756%2Fddb581017600c8a7cbd9afc8c8fed55a%2Farchitecture_detailed.svg?generation=1792035345447262&amp;alt=media" alt="Final architecture connecting production choices, protected execution, public-information trading and the official game.">
<em>Figure 4. The complete control loop. Production decisions update the work plan; the executor prepares worker actions and required purchases. The seller uses market forecasts and the farm's resource budget to propose sales. The scheduler combines these orders, and the next game observation closes the loop.</em></p>
<table>
<thead>
<tr>
<th>Component</th>
<th>What remains fixed</th>
<th>What adapts</th>
</tr>
</thead>
<tbody>
<tr>
<td>Farm executor</td>
<td>Tested work structure and production dependencies</td>
<td>Local weed/cash recovery and feasible purchase/hire timing</td>
</tr>
<tr>
<td>Synchronized crop substitutions</td>
<td>Compatible worker schedules checked in advance</td>
<td>Selected wheat cycles may become carrot cycles</td>
</tr>
<tr>
<td>Adaptive livestock selection</td>
<td>Compatible placement, care, and transport structure</td>
<td>Selected planned new animals may change between sheep and cows</td>
</tr>
<tr>
<td>Forecast-guided seller</td>
<td>Production reserves, liquidity/capacity safeguards</td>
<td>Sale quantities and timing, using estimated opponent supply</td>
</tr>
</tbody>
</table>
<p>The architecture treats a production change as a <strong>small plan recompilation</strong>, not a text substitution. Changing what a worker plants or places also changes what must be bought, what will be harvested, what that worker carries, and what later reaches the shed.</p>
<h2>5. Farm execution: preserving the useful part of the replay</h2>
<h3>5.1 Align the data before copying it</h3>
<p>The action stored in replay frame <code>t + 1</code> produces the transition from observation <code>t</code> to observation <code>t + 1</code>. I used that alignment when extracting worker schedules and transaction labels.</p>
<p>The executor also distinguishes field-action time from market-settlement time. If a worker needs a seed or carried input for an action now, buying it later in that turn cannot be treated as if it were already available during the earlier action phase.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F31328756%2F3ff2634ff75facd98ea0ee200f008036%2Ffig_execution_timing.svg?generation=1792035409595398&amp;alt=media" alt="Replay frame alignment and the order of worker actions and market settlement."></p>
<p><em>Figure 5. Align the stored action with the transition it actually causes. Procurement cannot retroactively supply an earlier worker action.</em></p>
<h3>5.2 Reconstruct procurement and hiring around the action plan</h3>
<p>The source provides a useful work schedule, but its historical market orders are not blindly reused. The executor determines upcoming input requirements and compares them with actual inventory. It preserves production reserves, schedules hiring around the source work sequence, and attempts to avoid spending scarce opening cash on unnecessary early purchases.</p>
<p>The ten-order limit is a hard scheduling constraint. Necessary farm orders reserve their capacity first. Sales may occupy the remaining slots and can appear before purchases to fund them, but they must not displace reserved necessary orders. Optional pre-purchases can reduce the residual sale capacity further. If necessary orders already fill all slots, liquidity must be raised earlier; the seller cannot invent an eleventh order.</p>
<p>The market receives not only available goods but also upcoming cash and capacity obligations. For example, selling before a nightly deposit can prevent shed overflow. Goods reserved as feed or fertilizer are not treated as freely disposable output.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F31328756%2F23320792edca14f2a41cd047d24fbbfb%2Ffig_order_budget.svg?generation=1792035441098356&amp;alt=media" alt="Ten-order capacity example with seven necessary farm orders and three optional sales."></p>
<p><em>Figure 6. With seven required farm orders, three of the ten slots remain available for sales. Accepted sales can execute first to fund purchases.</em></p>
<h3>5.3 Recover locally rather than discard the whole schedule</h3>
<p>Random weeds can invalidate an otherwise correct prerecorded action. The recovery logic looks for a feasible extra clearing action in existing workers' schedules, including the worker that will use the affected tile; when necessary, it can use an extra worker. Position and worker-index bookkeeping matter because a helper must not silently change the meaning of the original commands.</p>
<p>Local resource shortages create deferred tasks. Recovery reschedules the affected work while preserving the rest of the plan, subject to the labor and resources still available.</p>
<p>New production substitutions are disabled when relevant recovery is unresolved or cash is below the selector's guard. In other words, flexibility is optional; restoring a disrupted farm takes priority.</p>
<h2>6. Production flexibility without rebuilding the executor</h2>
<h3>6.1 Crops: compatible wheat-to-carrot cycles</h3>
<p>The crop layer exposes 40 checked planting-to-harvest cycles over 15 planting days, beginning at day 10 under zero-based numbering. Each selected cycle must support carrot growth and harvesting within the original worker schedule.</p>
<p>The crop selector can replace up to two eligible cycles whose products reach the shed on the <strong>same turn</strong>. Synchronizing arrivals makes their combined supply easier to value. Before accepting a second replacement, it adjusts the market estimate for the first one's missing wheat and additional carrots.</p>
<p>The crop valuation blends current-price revenue and a projected supply/demand estimate, subtracts seed cost, and includes the opportunity cost of the original wheat. The guard requires cash of at least 2,000, no relevant pending recovery, and an estimated gain of at least 20. Wheat removed from production is not assumed to remain available as free animal feed.</p>
<p>Tomato and strawberry schedules remain fixed in this version.</p>
<h3>6.2 Livestock: choose the animal before buying it</h3>
<p>Cow and sheep plans share useful operations, but their maturation, production intervals, and accumulated care bonuses differ. Therefore, the final livestock layer first replays each candidate animal lifecycle with the official animal rules.</p>
<p>Of the 14 animal lifecycles in the source schedule, seven supported replacement. The opening schedule and more complex transport sequences stayed fixed. For every permitted replacement, the plan generator updates:</p>
<ul>
<li>animal pickup and placement instructions;</li>
<li>required purchases and shed resource changes;</li>
<li>expected harvest quantities and carried goods;</li>
<li>delivery quantities and daily reference tile states.</li>
</ul>
<p>Each game gets its own copy of the production schedule. Substitutions are decided before buying a new animal; if pickup occurs early the next day, the decision is brought forward so procurement buys the correct type.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F31328756%2F3f0a1b55b3d281bb00d3894947384612%2Ffig_recompilation.svg?generation=1792035481373704&amp;alt=media" alt="Animal substitution recompiles inputs, production, and the continuation state."></p>
<p><em>Figure 7. An executable substitution must update the complete dependency chain, not just the animal name.</em></p>
<h3>6.3 Value the whole portfolio, not each animal in isolation</h3>
<p>At an eligible decision point, the selector compares the current plan with each candidate alternative. Its estimate uses current prices and market inventory, already-open shops, visible rival livestock and their maturity, recent inferred rival supply, and our own remaining planned deliveries. Unknown future shops and the opponent's hidden warehouse are unavailable.</p>
<p>The score blends two estimates equally: a current-price opportunity estimate and a forward supply/demand projection. The forward projection includes the price impact of <strong>all already-committed milk and wool</strong>, so several candidate cows cannot each pretend to sell into an unaffected market. The animal purchase-cost difference is subtracted.</p>
<p>The selector is greedy, recomputing values after each accepted change. It requires estimated gain of at least 150, cash of at least 2,000, no unresolved recovery, and at most two animal changes per decision day.</p>
<pre><code># Livestock-selection pseudocode.
if eligible_slots_today and cash &gt;= 2000 and not unresolved_recovery:
    changed = 0
    while eligible_slots_today:
        baseline = portfolio_value(committed_plan, public_observation)
        slot, gain = best_remaining_alternative(
            committed_plan, public_observation, baseline,
            include_purchase_cost=True,
        )
        if gain &gt;= 150 and changed &lt; 2:
            recompile_plan_copy(slot)
            changed += 1
        eligible_slots_today.remove(slot)
</code></pre>
<p>Eligible decision days were 3, 4, 9, 11, and 12, but accepted substitutions in the formal tests concentrated on days 11–12 because of the opening cash requirements. Across those games, the selector replaced zero to three planned sheep with cows.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F31328756%2Fa8c913ac1f742fe2bf20f06dd795ff50%2Ffig_production_choices.svg?generation=1792035509918618&amp;alt=media" alt="Observed milk and wool production combinations and their frequencies."></p>
<p><em>Figure 8. Milk and wool output for zero to three sheep-to-cow substitutions. The counts cover all 84 candidate games, including both player positions.</em></p>
<h2>7. The market model: learn opponent supply, then search a small action space</h2>
<h3>7.1 What is learned</h3>
<p>The market predictor estimates opponent sale quantities over four future intervals: <code>[0, 1)</code>, <code>[1, 4)</code>, <code>[4, 12)</code>, and <code>[12, 24)</code> turns. A small gradient-boosted tree model with a Poisson loss predicts the quantity in each interval. The models share parameters across products and use economic features rather than product or replay identifiers.</p>
<p>The 16 inputs are compact and economically interpretable:</p>
<table>
<thead>
<tr>
<th>Group</th>
<th>Inputs</th>
</tr>
</thead>
<tbody>
<tr>
<td>Time, 3</td>
<td>Season progress, hour, time to next known shop-consumption tick</td>
</tr>
<tr>
<td>Market and demand, 4</td>
<td>Normalized market inventory, relative price, price impact of five extra units, known daily shop demand</td>
</tr>
<tr>
<td>Visible rival capacity, 4</td>
<td>Rival cash, relevant productive tiles, goods visibly held on those tiles, estimated three-day production</td>
</tr>
<tr>
<td>Supply history and uncertainty, 5</td>
<td>Inferred unsold backlog, smoothed supply, recent 24-turn supply, time since last positive supply, flow-observability flag</td>
</tr>
</tbody>
</table>
<p>The public-flow tracker subtracts our own realized market contribution and accounts for known shop consumption. Ambiguous flows are marked uncertain rather than represented as exact hidden inventory.</p>
<p>The training labels are <strong>executed sales</strong>, not merely requested orders. This distinction prevents an unaffordable, unavailable, or otherwise unexecuted order from becoming a fictitious transaction label.</p>
<p>Training used 200 games from August 15–September 8, with 99 later games from September 9–16 held out for validation. These produced 1,252,800 training rows and 620,136 validation rows; observations from the same game are correlated.</p>
<p>Each interval model used 80 boosting iterations, at most 15 leaves, and maximum depth four. The exported trees run in pure Python during competition play.</p>
<h3>7.2 What is searched</h3>
<p>For an eligible sale product, the seller compares no sale, approximately one-quarter, one-half, three-quarters, and all available stock. It also includes the quantity recommended by the baseline rule-based seller, which handles cash needs, inventory age, and storage capacity. Candidate waiting horizons are known shop-consumption opportunities within 24 turns.</p>
<p>For each candidate, it considers three opponent-volume scales—0.5, 1.0, and 1.5 times the prediction—and three settlement positions: our sale before, alongside, or after the opponent's. Both sets use weights 0.25, 0.50, and 0.25. Revenues are calculated with the official unit-level price mechanics under each hypothesized order sequence.</p>
<p>The scoring rule is:</p>
<pre><code>score(quantity, horizon)
  = expected own revenue now and at the candidate horizon
    - 0.20 × downside deviation across scenarios
    - 0.002 × horizon × retained stock × base product price
</code></pre>
<p>The final term discourages indefinite holding. Each candidate values the immediate sale and a later liquidation of retained stock using the estimated supply and shop demand.</p>
<pre><code># Simplified description of the deployed timing search.
for quantity in candidate_sale_quantities(stock, baseline_quantity):
    for horizon in known_consumption_ticks_within_24_turns():
        outcomes = []
        for volume_scale, volume_weight in volume_scenarios:
            for order_position, position_weight in ordering_scenarios:
                revenue = simulate_own_receipts(
                    quantity, horizon, predicted_rival_flow * volume_scale,
                    order_position, known_shop_demand, official_price_rules,
                )
                outcomes.append((revenue, volume_weight * position_weight))
        score = expected_value(outcomes) - downside_penalty(outcomes)
        score -= holding_penalty(quantity, horizon)
        keep_best_candidate(score, quantity, horizon)

quantity = max(best_quantity, overdue_sellable_quantity)
# Production reserves and admissible order slots still constrain issuance.
</code></pre>
<p>The final seller's explicit rival-income penalty is zero. It is opponent-aware because it models the opponent's supply and settlement interaction, not because it directly subtracts their profit in every local score.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F31328756%2F8b642a0a5c9306517cf9629f2a1fcf2e%2Ffig_market_decision.svg?generation=1792035554271754&amp;alt=media" alt="Public observations feed a volume predictor and a small scenario-based selling search."></p>
<p><em>Figure 9. The supply forecast feeds a search over sale quantities and timing. Nine scenarios combine three rival-volume assumptions with three settlement positions.</em></p>
<h3>7.3 Liquidity and execution constraints override speculation</h3>
<p>The forecast-guided search defers to the baseline selling rules when cash is below 10,000, for wheat and fertilizer, for very small stock, and in the terminal phase. Below 10,000, the stock-age limit is four turns; above it, the limit is 48 turns. This limits how long goods are held, while the 24-turn search horizon determines how far ahead each trading decision looks.</p>
<p>The seller also reserves cash for upcoming purchases and hires. Funding needs, shed capacity, and terminal clearance can require sales even when waiting looks attractive.</p>
<p>A proposed sell order is not a completed sale. Inventory age is updated from observed stock, so an omitted or blocked order does not magically reset the age of unsold goods. The scheduler admits only sale orders that fit beside protected farm orders; it then observes what actually happened and replans next turn.</p>
<p>Finally, goods still carried by workers are not equivalent to sellable shed inventory. The endgame requires timely delivery as well as liquidation. The archived final validation ended with both sheds and worker bags empty.</p>
<h2>8. Local evaluation</h2>
<h3>8.1 How I designed local validation</h3>
<p>The goal was to turn real high-scoring games into test questions: <strong>can my agent beat this opponent's demonstrated production capability in a live shared market?</strong> The broader design favored recent replays, diverse shop sequences and production paths, and sampling without replacement from a large collection. The final comparison used the subset described in Section 8.2.</p>
<p>Simply replaying an opponent's actions was unreliable: our trades change prices and cash availability, so its original purchases can fail and cause an artificial collapse. Earlier tests rejected failed opponent reproductions. The later <strong>recorded-production opponent</strong> instead follows a fixed production-and-cost schedule, called a production contract:</p>
<ol>
<li>Extract realized production, deliveries, resource consumption, purchases, and hiring from a real episode, respecting action alignment and actual fills.</li>
<li>Preserve its shop-opening schedule in an instrumented official-engine environment. Run our candidate normally; inject the opponent's recorded production transitions rather than requiring every worker action to replay successfully.</li>
<li>Let both sellers trade in the live shared market. Prices and revenues are recalculated, not copied. Charge the opponent's operating costs; funding shortfalls become debt, repaid from sales or deducted from terminal cash. Our candidate receives no such credit. Warehouse limits and surplus losses still apply.</li>
<li>Compare <strong>our cash minus the opponent's debt-adjusted cash</strong>, alongside win rate and production stability. Swap player positions and use the same shops for paired version comparisons, retaining candidate execution failures in the results.</li>
</ol>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F31328756%2F4ba04fd4fb24dc63ddbec32a520a6203%2Ffig_validation_design.svg?generation=1792035577536973&amp;alt=media" alt="A normal candidate and a recorded production contract trade in the same live market."></p>
<p><em>Figure 10. The candidate runs normally while the environment preserves the opponent's recorded production. Both trade in the live market; any funding advanced to the opponent is deducted from its cash balance.</em></p>
<p>The candidate sees only information available at the current turn. Some earlier tests used the same seller on both sides to isolate production strength. The final recorded-production tests used our forecast-guided seller against the baseline rule-based seller.</p>
<p>As a complementary check, <strong>native-agent games</strong> run two complete agents in the official engine under new seeds, without injected production or credit. The contract method preserves demonstrated capability but not a rival's full adaptive behavior. Its results, native results, self-play cash, and actual online results are therefore reported separately.</p>
<h3>8.2 Final production-layer comparison</h3>
<p>The final comparison isolated <strong>adaptive livestock selection</strong>: the crop-only predecessor already had synchronized crop substitutions and the same forecast-guided seller. From 75 recorded online opponents, 12 previously examined scenarios formed the development set. The confirmation set used 18 scenarios selected before simulation, evenly spaced through the remaining 63. The native-agent test used 12 new seed/opponent scenarios across six older local agents.</p>
<table>
<thead>
<tr>
<th>Test set</th>
<th>Scenarios</th>
<th>Games per version</th>
<th>Crop-only wins</th>
<th>Crop + livestock wins</th>
<th>Mean margin, old → new</th>
<th>Paired margin gain</th>
</tr>
</thead>
<tbody>
<tr>
<td>Recorded opponents: development</td>
<td>12</td>
<td>24</td>
<td>18/24</td>
<td>22/24</td>
<td>10,323 → 12,682</td>
<td>+2,359</td>
</tr>
<tr>
<td>Recorded opponents: confirmation</td>
<td>18</td>
<td>36</td>
<td>30/36</td>
<td>32/36</td>
<td>10,915 → 12,684</td>
<td>+1,770</td>
</tr>
<tr>
<td>Complete-agent games</td>
<td>12</td>
<td>24</td>
<td>24/24</td>
<td>24/24</td>
<td>8,990 → 9,160</td>
<td>+171</td>
</tr>
</tbody>
</table>
<p>In confirmation, our mean cash rose from 112,406 to 113,130. The opponent's mean cash fell by approximately 1,046, explaining why the margin increase exceeded our own cash increase.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F31328756%2F5510afa2bb48c7d726d657b1ddc1eedb%2Ffig_final_evaluation.svg?generation=1792035624850597&amp;alt=media" alt="Final paired margin gains with uncertainty intervals on three distinct panels."></p>
<p><em>Figure 11. Mean margin gains with 95% bootstrap intervals, resampling whole scenarios together with their two player positions. The confirmation interval (−144 to +3,828) and complete-agent interval (−1,317 to +1,980) both include zero; the evidence for this final incremental improvement remains uncertain.</em></p>
<p>There were 168 formal games in total: 84 candidate and 84 predecessor runs. Daily and whole-season production matched the selected plans; transaction-ledger checks reported no errors. Each paired comparison used the same shop-opening sequence.</p>
<p>Confirmation's worst margin remained −20,372, although its tenth-percentile margin improved from −8,757 to +1,341. In the complete-agent test, some margins became worse even though every game remained a win.</p>
<h3>8.3 Failed ideas remained part of the evidence</h3>
<p>Several negative results shaped the final design:</p>
<ul>
<li><strong>Explicitly penalizing opponent income:</strong> this improved over the baseline seller but underperformed the opponent-aware search that scored our own revenue.</li>
<li><strong>Finer search without better supply estimation:</strong> increasing search resolution reduced margin in screening. More detailed enumeration could amplify a poor forecast.</li>
<li><strong>Unconditional livestock replacement:</strong> forcing all candidate slots toward sheep caused early funding pressure and production failures in the separate mechanism panel. Forcing cows had opposite economic effects under different shop conditions. Neither became the release policy.</li>
<li><strong>More cash but less advantage:</strong> in one confirmation scenario, a livestock change increased our cash by 105 and opponent cash by 6,517. The farm worked, but the economic estimate missed an important interaction.</li>
<li><strong>Reliable execution with a weak product mix:</strong> some routes completed production consistently yet still lost to stronger opponents.</li>
</ul>
<p>The forced-substitution experiments were separate from the 168 games comparing the final candidate with its predecessor.</p>
<h2>9. What remains open on the farm side</h2>
<p>The final agent adapts within a largely preset work schedule. Most movement and long-lived production remain prescribed. Livestock choices concentrate in the middle of the season, and the initial investment structure remains largely fixed.</p>
<p>I see four substantive next steps rather than a larger collection of ad hoc switches:</p>
<ol>
<li><strong>Expand the library of checked production plans.</strong> Add alternatives only when the entire lifecycle is feasible: labor, maturation, feed, harvest, bags, transport, and the state needed for subsequent work. A reusable plan should expose its costs and delivery curve alongside its end-of-day farm state.</li>
<li><strong>Make the execution interface compositional.</strong> Define entry resources, occupied time and tiles, promised output, and recovery obligations. This could let a planner combine blocks without depending on two historical episodes having exactly the same farm snapshot.</li>
<li><strong>Evaluate product mix by competitive value.</strong> A product's value depends on our supply, the opponent's supply, shop demand, and the seller's response. More milk is not inherently better; neither is diversification for its own sake.</li>
<li><strong>Improve uncertainty and response modeling.</strong> The current market predictor estimates mean volume. Rare large sales and an opponent changing behavior after our trade are not well represented by three scaled mean scenarios. A better model should be accepted through full-game cash and margin, not prediction error alone.</li>
</ol>
<p>Further work on production and market forecasting should continue together: product mix determines what can be sold, while market conditions determine its value.</p>
<h3>Post-deadline perspectives from other solutions</h3>
<p>Two solutions published in October offer useful perspectives on this September 30 submission.</p>
<p>M &amp; M &amp; P &amp; Q describe a hybrid of behavior cloning, self-play PPO, and heuristics. Their published architecture highlights both confidence-ordered repair/final-day search and a separate sequential resource-masking approach. This supports a broader lesson: learned planning and execution constraints need to be designed together. It is a much more ambitious learned-control approach than my final rule-guided agent. <a href="https://www.kaggle.com/competitions/kaggriculture/discussion/745073" target="_blank">3</a> <a href="https://github.com/msdsm/kaggriculture-solution/blob/main/docs/architecture.md" target="_blank">4</a></p>
<p>DECEM publishes an entity-Transformer behavior-cloning approach, with recency-weighted training, recent-day fine-tuning, and team-specific specialization. It uses sequential unit decoding and two small goods-preservation rules. This shows a promising direction for worker-level imitation: consistent demonstrations, suitable state representations, and action-decoding constraints. Its evaluation also branches from recorded games to test alternative behavior. <a href="https://www.kaggle.com/competitions/kaggriculture/discussion/745367" target="_blank">5</a> <a href="https://github.com/Decem-Y/kaggriculture-bc" target="_blank">6</a></p>
<h2>10. Release verification and reproducibility</h2>
<h3>10.1 Actual online performance</h3>
<p>The September 30 release, <strong>submission 56707930</strong>, was queried on October 15, 2026 at 02:36 UTC. The returned completed public games span September 30–October 14 UTC.</p>
<table>
<thead>
<tr>
<th>Online metric</th>
<th>Observed result</th>
</tr>
</thead>
<tbody>
<tr>
<td>Reported score snapshot</td>
<td>2226.6</td>
</tr>
<tr>
<td>Completed public games</td>
<td>925</td>
</tr>
<tr>
<td>Wins / losses / ties</td>
<td>535 / 390 / 0</td>
</tr>
<tr>
<td>Win rate</td>
<td>57.84%</td>
</tr>
<tr>
<td>Mean terminal cash</td>
<td>104,604</td>
</tr>
<tr>
<td>Mean terminal cash difference</td>
<td>+890</td>
</tr>
</tbody>
</table>
<p>The API returned 927 episodes: these 925 games, one self-play validation, and one errored public episode. The latter two are excluded from the completed-game rate. The snapshot covers 470 opponent teams and 592 opponent submissions, with repeated matchups.</p>
<p>Online matchmaking exposes the agent to a different opponent mix and fully adaptive behavior, complementing the controlled local comparisons.</p>
<h3>10.2 Deployment and reproducibility</h3>
<p>The initial Kaggle self-play validation completed with both agents <code>DONE</code>, terminal cash <strong>76,801 / 76,801</strong>, and empty sheds and bags. Re-running the uploaded archive on those online observations reproduced all <strong>1,438 actions</strong>. Cold loads were approximately 9.6 seconds; subsequent maximum action times were approximately 0.19 seconds in that validation.</p>
<p>Local experiments used <code>kaggle-environments 1.32.7</code>. The experiment archive retains the source files, package and model hashes, verification results, and per-game online statistics.</p>
<h2>Conclusion</h2>
<p>The most useful change in this project was not a bigger network or a bigger decision tree. It was separating reliable execution from competitive value, and then reconnecting them through explicit, testable interfaces.</p>
<p>The final agent keeps a production schedule it can execute, makes a limited number of compatible production choices, and sells with an estimate of the opponent's supply. This produced measurable local gains while preserving tested production. It also left clear weaknesses: narrow farm flexibility, approximate opponent modeling, and limited independent evidence of the last incremental improvement.</p>
<p>My main takeaway is simple: <strong>a farming plan is not good merely because it produces a lot, and a trade is not good merely because it raises our balance. What matters is what both players can earn after the decision—and whether the farm can still execute the plan that follows.</strong></p>
<h2>References</h2>
<ol>
<li>Kaggle. <a href="https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kaggriculture/kaggriculture.py" target="_blank">Official Kaggriculture engine</a>. Public reference; local experiments used the archived engine identified in the evidence bundle.</li>
<li>Kaggle competition discussion. <a href="https://www.kaggle.com/competitions/kaggriculture/discussion/731215" target="_blank">Daily Top Episodes Dataset</a>.</li>
<li>M &amp; M &amp; P &amp; Q. <a href="https://www.kaggle.com/competitions/kaggriculture/discussion/745073" target="_blank">A preview of our solution: BC, Self-Play PPO, and Heuristics</a>. October 2, 2026; the rank in the original post title was provisional.</li>
<li>M &amp; M &amp; P &amp; Q. <a href="https://github.com/msdsm/kaggriculture-solution/blob/main/docs/architecture.md" target="_blank">Published architecture and inference design</a>.</li>
<li>DECEM. <a href="https://www.kaggle.com/competitions/kaggriculture/discussion/745367" target="_blank">A preview of my solution: Only Behavior Cloning + two small rules</a>. October 3, 2026; the rank in the original post title was provisional.</li>
<li>DECEM. <a href="https://github.com/Decem-Y/kaggriculture-bc" target="_blank">Entity-Transformer behavior-cloning repository</a>.</li>
</ol>
<p><em>External references reviewed October 15, 2026.</em></p>

---

