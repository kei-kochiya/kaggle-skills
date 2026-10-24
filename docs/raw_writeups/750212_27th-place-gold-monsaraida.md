# Topic 750212: 27th Place (Gold) Solution: monsaraida

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/750212
- **Author**: 
- **Post Date**: 2026-10-17T09:38:50.007000
- **Votes**: 2

## Messages

### Message 1 by  (Votes: 2)

<p>Thank you to Kaggle and the organizers for running the competition and for providing the daily top-episode dataset and the API, and to all the participants who shared their insights in Discussions and public notebooks. Special thanks to team DSM, whose games served as my teacher.</p>
<h2>1. Overview</h2>
<p>I finished 27th out of 10,148 teams (gold medal, solo). The solution is imitation learning (behavior cloning) on games won by top teams. The model is a Transformer with about 2.7M parameters. It uses no reinforcement learning, no lookahead search, and no hand-written rules; actions are chosen from the model output alone.</p>
<p>The structure is the same two-stage training as in my <a href="https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/monsaraida-12th-place-solution" target="_blank">WSDM Cup 12th place solution</a>. Stage 1 (pre-training) uses games of the top 15 teams as the teacher, and Stage 2 (fine-tuning) uses only games of DSM, the team ranked 3rd on the LB. In both stages, only the winning side's actions are imitated. Below, I call DSM "the fine-tuning teacher".</p>
<p>The biggest gain came not from the model design but from how I chose the fine-tuning teacher. The rule-based agent I built early on plateaued, but while building it I had compared top games by what they planted and when, whether they bought land, and how they earned money. That comparison later guided my choice of teacher.</p>
<p>I competed solo, working with two AI coding agents (Codex and Claude Code); see Section 4.</p>
<h2>2. Solution</h2>
<h3>2.1 Stage 1: Pre-training</h3>
<p>Stage 1 imitates the winning side's actions in the 5,974 games won by the top 15 LB teams during the 10 days before the deadline.</p>
<p>The model is a 4-layer Transformer (Pre-LN, d_model 256, about 2.7M parameters). It reads the board as 292 tokens (field tiles, people, market items, shops, recent moves, and so on). The decisions within one turn (the action, item, and quantity for the farmer and for each hired worker, and the market orders) are chosen one at a time, each conditioned on the decisions already made. Combinations that are not valid actions (for example, choosing a non-crop item for a planting action) are masked out before choosing.</p>
<p>I trained for 60 epochs at a learning rate of 0.004, then lowered it to 0.001 and continued to epoch 70; this final low-learning-rate phase made the model stronger. I trained three models with different seeds.</p>
<h3>2.2 Stage 2: Fine-tuning</h3>
<p>To choose the fine-tuning teacher, I fine-tuned on each top team's games separately and compared the results. The pre-trained model could imitate this teacher's strategy, but not the strategies of the other top teams: the fine-tuning validation loss went down to about 0.11 with this teacher, but stayed at 0.56 with another top team.</p>
<p>This teacher re-submitted every few hours and often changed its crop strategy substantially. So I compared its submissions by what they planted and when, whether they bought land, and how they earned money, and selected three submissions with a consistent strategy. These three plant tomatoes in about 70% of games (mostly from day 16 to 18), never buy land, and have high win rates of 71–78% against the top 10 teams. Submissions with a different strategy (planting tomatoes from in-game day 9 and buying land in every game) were excluded.</p>
<p>In the 292 games won by these three submissions, I imitated only the winning side's (the fine-tuning teacher's) actions. Each of the three pre-trained models was fine-tuned with a learning rate of 3e-4, a batch size of 512, and 3200 steps. Short fine-tuning was not enough: 3200 steps beat 800 steps.</p>
<h3>2.3 Choosing the weights</h3>
<p>I averaged the weights from the end of fine-tuning (checkpoint averaging).</p>
<p>The fine-tuning validation loss was best at step 1504 and got worse afterwards. Yet the late weights, which the validation loss would call overfitted, were stronger. For each choice of weights, I built a 3-seed ensemble and played 128 games (64 game seeds × both seats) against the step-1504 ensemble:</p>
<ul>
<li>Last weights (step 3200): 85 wins, 43 losses (66.4%)</li>
<li>Average of the weights at steps 2800, 3000, and 3200: 75 wins, 53 losses (58.6%)</li>
</ul>
<p>Against the other submission I had at the time, only the late average had a winning record (78 wins, 50 losses), so I used it for the final submission.</p>
<p>I believe the late weights were stronger because they reproduced the teacher's strategy better. Tomatoes were planted in 69% of games by the teacher's three submissions, in 65.6% by the late average, and in 57.8% by step 1504.</p>
<p>The same thing happened in feature selection. A model with 43 additional features had a better validation loss in 15 of 16 settings, yet had a losing record over 19,200 head-to-head games. In imitation learning, the validation loss was not a reliable measure of how well the teacher's strategy was reproduced.</p>
<h3>2.4 Ensemble</h3>
<p>I averaged the action probabilities of three models that differ only in seed. The ensemble won 43–44 of 64 games (67–69%) against the strongest of the three single models. The final submission is this 3-model ensemble, with each model's late weights averaged.</p>
<h2>3. Choosing the fine-tuning teacher</h2>
<p>In imitation learning, it is tempting to think that more teacher games make a stronger model. In practice, mixing teachers with different strategies made the learned policy an average of those strategies, and it fell apart.</p>
<ul>
<li><strong>Changing the teacher team</strong>: With the same pre-trained model and the same settings, fine-tuning on another top team's games produced a model that could barely beat the fixed evaluation opponents.</li>
<li><strong>Mixing teachers</strong>: Using a mix of the top 5 teams as the teacher barely lowered the fine-tuning validation loss (0.396→0.379).</li>
<li><strong>Ensembling models with different teachers</strong>: No combination beat the ensemble of three models trained on the same teacher. What worked was combining models that share the same teacher and differ only in seed.</li>
</ul>
<h2>4. Development process</h2>
<p>The right side of the figure shows how the development team was organized.</p>
<ul>
<li><strong>Human–AI collaboration</strong>: Under me (Project Owner), there is one Project Manager. The Project Manager assigns work to Specialists, each responsible for an area such as data collection, training, evaluation, analysis of top games, reviewing decisions, and records and reports. Specialists launch temporary helpers (subagents, shown as Workers in the figure) as needed.<ul>
<li>The Project Manager and the Specialists run persistently, as threads in Codex and as sessions in Claude Code. Each of them switches between Codex and Claude Code when one is unavailable or depending on the strengths of the models.</li>
<li>Instructions, work procedures, and records are kept in one place, and Codex and Claude Code read the same files.</li>
<li>The six Specialists and their colors in the figure are examples.</li></ul></li>
<li><strong>Gemba walk (seeing the actual work)</strong>: Rather than leaving everything to the AI, I read each Specialist's work records directly. I repeated a cycle: see → notice → give the Project Manager detailed instructions → record the improved way of working as a rule. I never instructed Specialists directly; all instructions went through the Project Manager.</li>
<li><strong>Change of direction and result</strong>: While I left things to the AI, it kept refining hand-written rules, and at one point my rank fell to 706th. Five days before the deadline, I started reading the Specialists' records and giving detailed instructions, and in the process switched the approach to imitation learning from top teams. The final result was 27th place (gold medal).</li>
</ul>
<p>My takeaway from this competition: <strong>for now, combining human insight with AI's implementation power works better than full autonomy.</strong></p>
<h2>References</h2>
<ul>
<li><a href="https://www.kaggle.com/datasets/kaggle/kaggriculture-episodes-index" target="_blank">Official daily top episode dataset</a></li>
<li>Public notebooks and Discussions</li>
<li><a href="https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/monsaraida-12th-place-solution" target="_blank">WSDM Cup - Multilingual Chatbot Arena 12th Place Solution</a> (monsaraida, two-stage training)</li>
<li>Game screen in the figure: <a href="https://www.kaggle.com/competitions/kaggriculture" target="_blank">Kaggriculture</a> (Kaggle)</li>
</ul>

---

