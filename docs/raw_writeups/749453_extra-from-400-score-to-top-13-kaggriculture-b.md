# Topic 749453: From 400 Score to Top 13: Kaggriculture Bronze Medal Solution

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/749453
- **Author**: 
- **Post Date**: 2026-10-15T00:18:07.210000
- **Votes**: -1

## Messages

### Message 1 by  (Votes: -1)

<p><img src="https://raw.githubusercontent.com/sarthak-here/kaggriculture/master/docs/images/rank512-bronze-banner.svg" alt="Kaggriculture Rank 512 Bronze"></p>
<p>I finished Kaggriculture at <strong>rank 512 and earned a Bronze Medal</strong>.</p>
<p>The final rank tells only part of the story. My earliest agents were near a rating of 400 and could barely operate a complete farm. After many iterations, replay investigations and local tournaments, one of my agents climbed into the <strong>top 13</strong> during the competition.</p>
<p><img src="https://raw.githubusercontent.com/sarthak-here/kaggriculture/master/docs/images/top13-leaderboard.png" alt="Kaggriculture leaderboard showing rank 13"></p>
<p><em>Figure 1. The agent reached rank 13 during the competition with a rating of 2810.0.</em></p>
<p>I could not preserve that position until the final evaluation, but progressing from an almost non-functional farmer to competing near the top was the most rewarding part of the competition.</p>
<p>My final submitted policy, <strong>Severe route current-rule cleanup</strong>, received a final rating of <strong>2089.3</strong>.</p>
<h2>Understanding the game</h2>
<p>Kaggriculture is a two-player farming simulation played over 30 days, with 24 turns per day. Each player must coordinate an entire production system: purchase seeds, animals and land; hire workers; plant and water crops; feed and care for animals; harvest products; transport everything back to the shed; and sell products before market demand disappears.</p>
<p>The player with the most money at the end wins. Produce left on the farm or inside the shed has no terminal value.</p>
<p>The most important complication is that both players use the same market. If both farms produce the same product, they compete for a limited amount of demand and push its price down. This meant that simply producing more was not enough. The agent had to produce the right items, deliver them on time and sell them before the opponent captured the available demand.</p>
<h2>Why I used a route-based policy</h2>
<p>I initially explored learned policies, including behaviour cloning from strong public matches. The offline model reached high action-prediction accuracy, but it performed poorly in complete games. A few incorrect movement or economic decisions moved the farm into states that were not represented in the training data. The errors then accumulated over hundreds of turns.</p>
<p>I therefore focused on a deterministic route-based policy. A route provided a coordinated long-term schedule for:</p>
<ul>
<li>Farm expansion</li>
<li>Crop placement</li>
<li>Animal purchases</li>
<li>Worker movement</li>
<li>Watering and harvesting</li>
<li>Feeding and animal care</li>
<li>Product transportation</li>
<li>Market activity</li>
</ul>
<p>This approach was easier to inspect and evaluate. When a match was lost, I could trace the first important difference and determine whether the cause was financing, production, labour, transportation or selling.</p>
<h2>Learning from games</h2>
<p>I downloaded competition replays and converted match information into structured data that could be compared across agents.</p>
<p>For every important loss, I examined:</p>
<ul>
<li>The sequence of unlocked shops</li>
<li>Purchases made by both players</li>
<li>Farm composition and worker count</li>
<li>Planted and harvested crops</li>
<li>Animal feed consumption</li>
<li>Shed inventory and carried products</li>
<li>Market orders and sale timing</li>
<li>Money over time</li>
<li>Unsold production at the end</li>
</ul>
<p>The useful question was not simply, “Why did the opponent earn more?” I looked for the first decision that created the eventual gap. Lower revenue could come from producing too little, but it could also be caused by an early cash shortage, a missed worker hire, late harvesting, slow transportation or selling after the opponent.</p>
<p><img src="https://raw.githubusercontent.com/sarthak-here/kaggriculture/master/docs/images/agent-flow.svg" alt="How the Kaggriculture agent worked"></p>
<p><em>Figure 2. The final policy combined a planned farming route with reactive economic and execution controls.</em></p>
<h2>Shop-conditioned strategies</h2>
<p>The town’s unlocked shops determined which products would receive demand. Shops could also appear more than once, making every world economically different.</p>
<p>My agents therefore used different farming routes depending on the visible shop sequence.</p>
<table>
<thead>
<tr>
<th>Visible condition</th>
<th>Strategic effect</th>
</tr>
</thead>
<tbody>
<tr>
<td>YARN store</td>
<td>Increased the potential value of wool and sheep-heavy routes</td>
</tr>
<tr>
<td>Bakery-related shops</td>
<td>Changed the value of wheat, feed and animal products</td>
</tr>
<tr>
<td>Milk-supporting shops</td>
<td>Made cattle-heavy farms more attractive</td>
</tr>
<tr>
<td>Duplicate shops</td>
<td>Concentrated demand and changed the best product mix</td>
</tr>
<tr>
<td>Weak demand worlds</td>
<td>Favoured a leaner farm that protected cash</td>
</tr>
</tbody>
</table>
<p>This was a major improvement over using one fixed plan in every match. The agent selected a route early in the game and then used reactive controllers to handle market conditions, capital constraints and unexpected execution states.</p>
<h2>The final farming strategy</h2>
<p>The final Severe route was reconstructed from one of the strongest farm structures found during loss analysis. Unlike a small market-order patch, it changed the complete production architecture.</p>
<p>Its typical terminal structure was:</p>
<ul>
<li>Three active quadrants</li>
<li>Approximately eight cows</li>
<li>Four geese</li>
<li>Around ten workers</li>
<li>No sheep retained in the final farm</li>
<li>A fixed 54-carrot planting programme</li>
<li>Approximately 173 carrots sold per game</li>
</ul>
<p>This was a relatively lean farm compared with several labour-heavy opponents.</p>
<p>The goal was not to maximize every type of production. It was to create a farm whose land, workers, feed requirements and market supply remained balanced. Earlier agents often created larger mixed farms that looked productive but consumed too much feed and labour. The Severe route reduced that burden while preserving cattle, fertilizer and high-value crop production.</p>
<h2>Protecting operating capital</h2>
<p>Several losses began with decisions that appeared small. Buying a few extra seeds could leave the farm without enough money to hire workers the next morning. Missing those workers then caused crops to go unwatered, animals to go unfed and harvests to arrive late.</p>
<p>The controller therefore protected money required for upcoming commitments:</p>
<ul>
<li>Worker hiring</li>
<li>Seeds</li>
<li>Animals</li>
<li>Feed</li>
<li>Land expansion</li>
<li>Route-specific purchases</li>
</ul>
<p>A purchase was not considered profitable merely because the crop or animal could eventually generate revenue. It also had to leave enough working capital for the rest of the farm to continue operating.</p>
<h2>Making the route executable</h2>
<p>A long-term schedule works only if workers can execute it. The controller had to coordinate workers with different positions, inventories and deadlines.</p>
<p>The route handled planting, watering, feeding, caring, harvesting and carrying products back to the shed as one connected process. A profitable crop had no value if it was harvested too late, left in a worker’s inventory or never sold.</p>
<p>Late in the match, the policy stopped treating every possible farm action as useful. New investments, fertilizer, care and harvesting had to contribute to a sale before the final market window.</p>
<h2>Reactive market control</h2>
<p>The market controller adjusted selling decisions using:</p>
<ul>
<li>Current inventory</li>
<li>Available town demand</li>
<li>Expected price impact</li>
<li>Remaining production time</li>
<li>Product priority</li>
<li>Opponent behaviour</li>
<li>The stage of the game</li>
</ul>
<p>Orders were prioritized by their expected impact rather than only their nominal revenue. This mattered because both players submitted market orders during the same phase. Selling a product one turn earlier could capture demand before the opponent filled the market.</p>
<p>Against opponents with very similar behaviour, the agent could detect repeated action patterns and alter selected sale timing to avoid losing identical market races.</p>
<h2>YARN and bakery handling</h2>
<p>YARN openings were among the most difficult match families. Some strong opponents used labour-heavy sheep farms with 11 or more sheep and as many as 13 workers. My Severe route generally remained closer to a lean cattle-and-geese structure with around ten workers.</p>
<p>I tested several possible repairs:</p>
<ul>
<li>Converting cows into sheep</li>
<li>Changing wool-sale timing</li>
<li>Increasing animal feed</li>
<li>Adding terminal harvesting</li>
<li>Replacing the complete route</li>
<li>Detecting late YARN openings</li>
</ul>
<p>Some changes improved the final cash margin but still failed to change the winner. This was an important distinction: explaining a loss did not mean that the proposed patch was better.</p>
<p>Bakery worlds required similar care. Wheat could be sold, used as animal feed or sacrificed for another crop. The agent used dedicated capital and route conditions to avoid expanding into a bakery strategy that it could not afford to execute.</p>
<h2>Execution safety</h2>
<p>A strong strategy was useless if the submitted Python file failed during a match. The final agent included execution guards that:</p>
<ul>
<li>Removed empty market orders</li>
<li>Removed invalid market PASS entries</li>
<li>Rejected malformed quantities</li>
<li>Removed non-positive orders</li>
<li>Limited the market queue to ten entries</li>
<li>Returned a safe fallback action if the main controller failed</li>
</ul>
<p>The final artifact completed full 720-step validation under the current game rules with no hard action errors.</p>
<h2>Testing improvements</h2>
<p>Every serious candidate was tested in both seat orders across fresh seeds. Seat pairing was necessary because the two players started in different positions. Even identical agents could behave differently depending on their seat.</p>
<p>I also tested against multiple opponent families. Beating one previous model was not enough because many strategies had highly favourable but narrow matchups.</p>
<p>The Severe route’s main validation result was:</p>
<table>
<thead>
<tr>
<th>Evaluation</th>
<th>Result</th>
</tr>
</thead>
<tbody>
<tr>
<td>Ten strong opponent families</td>
<td>136 wins, 64 losses</td>
</tr>
<tr>
<td>Overall win rate</td>
<td>68%</td>
</tr>
<tr>
<td>Failed games</td>
<td>0</td>
</tr>
<tr>
<td>Diagnostic top-player replay panel</td>
<td>27 wins, 3 losses</td>
</tr>
</tbody>
</table>
<p>The 27–3 replay result was useful for diagnosing coverage, but it was not treated as proof of a 90% live win rate because recorded opponents could not react to changes in the market.</p>
<p>In a separate six-agent current-rule round robin, Severe recorded <strong>38 wins and 12 losses</strong>. It defeated the historical pf_all agent in all ten games and beat the strongest broad internal agent 6–4 in their direct matchup.</p>
<p>These tests supported submission, but they did not guarantee leaderboard performance.</p>
<h2>What failed and what I learned</h2>
<p>A large part of the project consisted of rejecting ideas that sounded reasonable.</p>
<p>Failed or inconsistent approaches included:</p>
<ul>
<li>Training a behaviour-cloning policy only from replay actions</li>
<li>Buying additional land without a compatible worker schedule</li>
<li>Increasing the number of workers globally</li>
<li>Producing more crops without considering shared demand</li>
<li>Generic weed-clearing overrides</li>
<li>Unconditional feed top-ups</li>
<li>Terminal harvesting changes that disrupted other work</li>
<li>Selecting routes using replay reward alone</li>
<li>Majority-voting actions from several strong routes</li>
<li>Promoting a candidate because it improved one diagnostic metric</li>
<li>Overfitting to a single opponent family</li>
</ul>
<p>One particularly important lesson came from a lower-feed route. It appeared to fix the problem I had diagnosed, but direct testing showed that it lost badly to the route it was intended to replace.</p>
<p>After that, every replacement had to be tested directly against the incumbent. Fixing the suspected cause of a loss was not sufficient evidence.</p>
<p>Another lesson was that cash margin and match wins were not interchangeable. A candidate could earn a larger average margin while failing to improve the number of games won. I therefore promoted agents by paired-seat win rate, using margin only to understand the mechanism.</p>
<h2>From 400 to the top 13</h2>
<p>The earliest agents were close to a 400 rating and often failed for basic reasons: broken submission entry points, invalid market actions, poor planting, missed hiring and incomplete harvesting.</p>
<p>The next stage produced functional farms but still lost because they followed one fixed plan in every world.</p>
<p>Route portfolios, shop-conditioned strategies and better market handling produced the largest improvement. One of these agents climbed into the top 13 during the competition.</p>
<p>However, the leaderboard continued to evolve. Some locally dominant agents had narrow matchup advantages, and several strong changes did not generalize to new opponent families.</p>
<p>I eventually finished at rank 512 with a Bronze Medal. Falling from a temporary top-13 position was disappointing, but it also made the final result more meaningful. The project was not a smooth climb. It was a long sequence of improvements, regressions, failed submissions, corrected evaluations and rebuilt agents.</p>
<h2>Final lessons</h2>
<p>The most valuable lessons were not individual farming tricks. They were about experimental discipline:</p>
<ul>
<li>Test both seat orders.</li>
<li>Use fresh seeds for confirmation.</li>
<li>Evaluate against several independent opponent families.</li>
<li>Treat replays as diagnostic evidence, not live-match proof.</li>
<li>Measure wins and losses, not only cash margin.</li>
<li>Verify that a mechanism actually activates.</li>
<li>Test a replacement directly against the component it replaces.</li>
<li>Keep exact source files and hashes for serious submissions.</li>
<li>Never assume that more production creates more profit in a shared market.</li>
</ul>
<p>Kaggriculture became much more than a leaderboard attempt. It was an exercise in simulation, economic modelling, adversarial decision-making, software reliability and learning from failure.</p>
<p>I started with an agent around 400 rating, briefly reached the top 13 and finished with a Bronze Medal at rank 512. It was a difficult competition, but I genuinely enjoyed the process.</p>
<h2>Code</h2>
<p>The exact final main.py and the complete experiment history are available here:</p>
<ul>
<li><a href="https://www.kaggle.com/datasets/sarthaksharma14/kaggriculture-rank-512-bronze-agent" target="_blank">Final Kaggriculture agent</a></li>
<li><a href="https://github.com/sarthak-here/kaggriculture" target="_blank">GitHub repository</a></li>
</ul>
<p><strong>Final result: Rank 512 — Bronze Medal</strong></p>

---

