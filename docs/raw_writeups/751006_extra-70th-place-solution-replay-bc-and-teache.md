# Topic 751006: 70th Place Solution — Replay BC and Teacher Selection

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/751006
- **Author**: 
- **Post Date**: 2026-10-20T14:28:17.467000
- **Votes**: 1

## Messages

### Message 1 by  (Votes: 1)

<h1>70th place: Replay BC, teacher selection, and a tomato rule</h1>
<p>Our team, <strong>NSK_54</strong>, finished <strong>70th out of 10,148 teams</strong>. This write-up focuses on the replay-BC agent that set our final score: <strong>4883.6 private</strong>, with a public rating of <strong>2612.0</strong>. The private and public ratings use different scales.</p>
<p>Our final agent combined a <strong>6.54M-parameter recurrent entity Transformer</strong>, behavior cloning from public replays, and a small deterministic overlay. The biggest improvements came from choosing <strong>strong, current teachers with compatible strategies</strong>.</p>
<h2>The final agent</h2>
<p>We represented the observation as <strong>277 entity tokens</strong>: one global token, 12 item tokens, 200 farm tiles, 32 own-unit slots, and 32 opponent-unit slots. Only public opponent information was included.</p>
<p>The encoder used four Transformer blocks, width 256, eight attention heads, and FFN width 1024. A pooled representation updated a <strong>256-dimensional GRU</strong> once per turn. Two single-block autoregressive decoders generated:</p>
<ul>
<li>Worker commands, in engine order: farmer first, then farmhands.</li>
<li>Up to ten market orders, attending to both the board and the decoded worker commands.</li>
</ul>
<p>Each decision was factorized into <strong>operation → item → quantity</strong>, where applicable.</p>
<p>After every choice, we updated a <strong>shadow state</strong> tracking stock, seeds, changed tiles, cash, hires, and market supply. The next decision received legality masks and features from that updated state. This prevented reuse of stock already consumed by another worker and allowed a sale to fund a later purchase in the same turn.</p>
<p>The network chose routes, work allocation, investments, and sales. The shadow handled local validity and accounting; it could not prevent every strategic mistake, such as two workers walking toward the same distant tile.</p>
<p>Inference was greedy. In one paired test with identical weights, greedy decoding scored <strong>0.836</strong>, versus <strong>0.094</strong> with temperature-1 sampling. Small sampled errors accumulated into failed economies.</p>
<p>Throughout this post, local <strong>score = (wins + 0.5 × draws) / games</strong>.</p>
<h2>Replay preparation</h2>
<p>Our initial clean top-30 dataset contained <strong>4,290 player trajectories</strong>, or <strong>3,084,510 observation–action pairs</strong>, from 18 teams. Each trajectory contained one player's 719 decisions.</p>
<p>We re-simulated recorded games to reconstruct observations, aligned each observation with the following replay action entry, and converted commands into <strong>what the engine actually executed</strong>. Every converted trajectory had to reproduce both players' next observations when its labels were decoded and replayed.</p>
<p>Market positions required special care. Strong teachers often submitted standing SELL orders for goods they did not hold. Approximately 10% of elite teachers' submitted SELL orders executed nothing in one audit.</p>
<p>We represented those orders as positional <strong>SKIPs</strong>. Removing them would shift subsequent orders: the engine processes both players' market lists by index, so position affects execution timing.</p>
<p>We also tracked exact team identities and submission versions, filtered invalid games and incompatible engine versions, deduplicated episodes, and excluded evaluation seeds. Both seats of an episode stayed in the same data split. Ordinary teacher losses were retained.</p>
<h2>Choosing compatible teachers</h2>
<p>The first observation was <strong>identical across seeds</strong>. A greedy policy therefore made the same opening decision in every game. When we mixed teachers with incompatible openings, the student sometimes combined purchases from one strategy with labor allocation from another.</p>
<p>One failed opening spent <strong>2,893 of the initial 3,000 coins</strong>, then lost its animals by day three. Small changes near an opening logit tie could change every rollout at once.</p>
<p>We grouped teacher versions into families using their <strong>turn-0/turn-1 order signatures</strong>. An early mixed-family fine-tune lost all 32 games against our scripted reference, while a compatible-family fine-tune won <strong>107/128</strong>, scoring 0.836.</p>
<p>Compatibility alone was insufficient. A later comparison kept the architecture, initialization, budget, and learning-rate schedule fixed while changing teacher strength within the same opening family:</p>
<table>
<thead>
<tr>
<th>Teachers</th>
<th>Wins against the wheat-dump reference</th>
</tr>
</thead>
<tbody>
<tr>
<td>Current top-30, compatible family</td>
<td>0/32</td>
</tr>
<tr>
<td>Current top-eight, compatible family</td>
<td>60/64</td>
</tr>
</tbody>
</table>
<p>The top-eight teachers differed in investment pace, market behavior, and later execution, despite similar openings.</p>
<p>Our successful training chain progressively narrowed the demonstrations:</p>
<table>
<thead>
<tr>
<th>Stage</th>
<th>Data</th>
<th>Budget</th>
</tr>
</thead>
<tbody>
<tr>
<td>Broad pretraining</td>
<td>4,290 clean top-30 trajectories</td>
<td>Selected checkpoint after 14 epochs</td>
</tr>
<tr>
<td>Compatible-family refinement</td>
<td>Strong and elite teacher tiers</td>
<td>18k + 9k updates</td>
</tr>
<tr>
<td>Current top-eight refinement</td>
<td>877 compatible trajectories</td>
<td>6k updates</td>
</tr>
<tr>
<td>Final specialist fine-tune</td>
<td>Newest compatible DSM, DECEM, and yuto083 versions</td>
<td>12k updates</td>
</tr>
</tbody>
</table>
<p>The final dataset contained <strong>213 trajectories, 197 for training</strong>. Those 197 trajectories were the last fine-tuning set, following broad pretraining and intermediate refinement.</p>
<p>Adding older compatible data diluted the result. Several highly ranked off-family teachers also damaged the policy. For our model and budget, teacher compatibility, strength, and exact version mattered more than dataset size alone.</p>
<h3>Training details</h3>
<p>We used masked cross-entropy over worker and market operation/item/quantity factors, with teacher-forced prefixes. A nominal update contained <strong>eight trajectories × 64 turns = 512 rows</strong>. GRU state was carried between chunks and detached at their boundaries.</p>
<p>AdamW used weight decay 0.01. Peak pretraining learning rate was 3e-4; the final fine-tunes decayed from 3e-5 to 5e-6.</p>
<p>Two smaller details helped define the training recipe:</p>
<ul>
<li><strong>Repeated-history weighting:</strong> identical history prefixes received row weight <code>min(1, 25/m)</code>, where <code>m</code> was their multiplicity. This reduced repeated-opening dominance while preserving later unique behavior.</li>
<li><strong>Fixed loss denominators:</strong> normalizing each chunk separately had given the short 15-turn tail chunk about 4.3 times the per-row influence of a full chunk. We switched to fixed denominators.</li>
</ul>
<p>These changes entered alongside the teacher-tier pipeline, so we did not isolate their individual contributions.</p>
<h2>The wheat dump: more targeted data made things worse</h2>
<p>Eight of one BC version's first nine Kaggle losses came against opponents selling at least 20 wheat on turn zero. We reproduced the failure with a scripted opponent that submitted a large opening wheat sale.</p>
<p>The sale depressed wheat prices and squeezed our opening liquidity. Our policy reached zero cash, bought small amounts of feed, and fed animals that were not yet at risk. A cow unfed since day zero died at turn 48.</p>
<p>Our first hypothesis was missing coverage. We devoted <strong>30% of a fine-tuning mixture</strong> to teacher wins against these openings. It regressed:</p>
<table>
<thead>
<tr>
<th>Attempt</th>
<th>Result</th>
</tr>
</thead>
<tbody>
<tr>
<td>Targeted teacher wins against dump openers</td>
<td>Paired score change <strong>−0.286</strong>, 95% CI <strong>[−0.531, −0.040]</strong></td>
</tr>
<tr>
<td>Current top-eight compatible teachers, including newer openings</td>
<td><strong>64/64 wins</strong> on fresh seeds, mean margin about <strong>+20.1k coins</strong></td>
</tr>
</tbody>
</table>
<p>The targeted model's own bank actually rose, but its market behavior helped the opponent even more. In a shared market, higher own income is an incomplete metric.</p>
<p>In this experiment, extra teacher examples did not teach reliable recovery from the student's mistakes. Stronger current teachers, including a more robust opening, were more effective.</p>
<h2>Detours: macro executors and RL</h2>
<p>A macro-task policy produced our first major improvement. Instead of predicting individual movement steps, it predicted a task and destination, while a persistent executor handled routing and reservations. Wasted movement fell from <strong>35.1% to 14.0%</strong>, and the agent beat our handwritten baseline <strong>50–0</strong>.</p>
<p>Later, a higher-level planner became a bottleneck. Even when fed teachers' reconstructed investment schedules on 21 held-out games, it banked only about <strong>55k</strong>, against the teachers' <strong>106k</strong>. This diagnosed limitations of that executor's placement, labor, timing, and selling. We moved execution back into the network while retaining shadow-state accounting.</p>
<p>We also tried extensive PPO and self-play. A scratch run on the final-sized architecture reached <strong>47M environment steps over 91 hours</strong> without learning useful farming. One BC-initialized PPO branch improved its public rating from 1738.5 to 2322.0, but the local gain was insignificant and the ladder comparison included possible field drift.</p>
<p>The later BC chain overtook that branch. <strong>No RL updates were applied to the final agent described here.</strong></p>
<h2>The tomato overlay</h2>
<p>Our stronger BC agents still lost close games late. Across 94 losses from two versions, <strong>62% were not behind at day 20</strong>. On day 22, they averaged <strong>0.9 tomato plants</strong>, versus opponents' <strong>7.9</strong>.</p>
<p>A parallel LLM-selector experiment helped explore crop-timing changes. We distilled its useful tomato decision into deterministic code, matching <strong>140/140 tested LLM decisions</strong>.</p>
<p>The final rule replaced planned carrot-seed purchases with tomato seed on <strong>engine days 16–19</strong>, provided:</p>
<ul>
<li>Tomato price was at least 1.25 times base price and an unlocked shop bought tomatoes.</li>
<li>Plants plus held seeds stayed within eight per tomato-buying shop.</li>
<li>At least 300 coins remained after purchase.</li>
</ul>
<p>Tomatoes produce four times before decaying, making planting time important for the late-season shortage.</p>
<p>The overlay also directed carried goods toward the shed during the final 20 steps and added final-step sales. Against the plain base on <strong>96 fresh seeds, both seats</strong>, with five rotating opponents, the complete overlay gained:</p>
<table>
<thead>
<tr>
<th>Metric</th>
<th>Paired improvement, 95% CI</th>
</tr>
</thead>
<tbody>
<tr>
<td>Panel score</td>
<td><strong>+0.130 [0.072, 0.189]</strong></td>
</tr>
<tr>
<td>Final bank</td>
<td><strong>+1,456 [688, 2,224]</strong></td>
</tr>
</tbody>
</table>
<p>Its standalone leaderboard contribution remained unresolved: the plain control was retired before the private tournament. The evidence for this improvement is the paired local test.</p>
<h2>Evaluation and deployment</h2>
<p>We selected checkpoints through <strong>full greedy games</strong>, using matched seeds, seats, and opponents, separate development and confirmation bands, and confidence intervals over seeds. External opponents and same-recipe retrains helped expose overfitting and training noise.</p>
<p>Offline accuracy could improve while play remained poor. One stage raised turn-exact accuracy from <strong>0.085 to 0.248</strong> without winning a game against the reference.</p>
<p>Deployment used a <strong>NumPy policy in one file</strong>, approximately 24.5 MB zipped. Export checks reproduced <strong>2,876/2,876 action turns</strong> between PyTorch and NumPy. Quiet-host p99 inference latency was approximately <strong>0.28–0.29 seconds</strong>.</p>
<p>We trained on an RTX 3090 Ti and a machine with four GTX 1080 Ti cards. Broad pretraining took about 12 hours; later 6,000-update fine-tunes took about 50 minutes on the four-card machine.</p>
<p>The experiments we would prioritize earlier next time are teacher-version audits, executor tests with true teacher targets, and full-game comparisons of proposed failure fixes. Those diagnostics repeatedly changed where we spent our effort.</p>
<h2>Acknowledgements</h2>
<p>Thanks to the hosts and all teams sharing public replays, especially DSM, DECEM, and yuto083 for the final specialist demonstrations. Earlier stages also benefited from other public leaders. Our separate scripted-baseline line used Ahmed Berat Ozer's V43 agent, shared through sdy623's notebook under Apache-2.0.</p>

---

