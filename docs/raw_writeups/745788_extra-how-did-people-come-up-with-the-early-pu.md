# Topic 745788: How did people come up with the early public strategies in the first place?

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/745788
- **Author**: 
- **Post Date**: 2026-10-04T12:19:38.286000
- **Votes**: 7

## Messages

### Message 1 by  (Votes: 3)

<p>I was doing RL for about a month of a half. BC saved a lot of training time but didn't seem necessary for a good bot, as improvements were pretty correlated with training time: my fresh RL runs could catch up to my weaker BC runs in an hour or two. I think this describes a lot of the top bots that went the RL+self-play route as well.</p>
<p>I wasn't high on the leaderboard until the end, but that was much more because I finally did a big training run than because of BC. If you have big batch sizes, a competitive league, and a well tuned RL algorithm and model, it will learn quite optimal strategies and continue improving with training.</p>
<p><a href="https://github.com/CarsonBurke/kaggriculture" target="_blank">code and diagrams</a></p>

---

### Message 2 by  (Votes: 1)

<p>I found this part in your repo, it says you tried RL without BC and it plateaued (at a similar level to my attempt :D). What did you change in order for pure RL to be successful?
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F23060793%2F2dd06bb8338911f852bda21363feaeb4%2FScreenshot%202026-10-09%20184127.png?generation=1791546201245686&amp;alt=media" alt=""></p>

---

### Message 3 by  (Votes: 1)

<p>That was very early on, and excuse the slop, I probably should have cleaned up the repo more. I did not see plateau in elo gain after a combination of improvements:</p>
<ol>
<li>a better model: lejepa backbone, position-specified targets for movement (moveto(x, y) and pseudo-pathfind, instead of my previous move(direction)), Win/Draw/Loss reward/critic instead of dense or soft-terminal rewards. As well as better market actions allowing more specific trading quantities.</li>
<li>critic warmup on the random model, mostly just saves training time.</li>
<li>a hardness-focused self-play league with screening of historical opponents we are uncertain a about (due to having not played recently), and play against those we think will be the hardest to win against.</li>
<li>sufficient training time at a lower learning rate: after 30-45mins (more when BCing from the top elo games), the from-fresh model improves quite well and catches up to BC+RL.</li>
</ol>
<p>I found about similar entropy and strategy after awhile.</p>

---

### Message 4 by  (Votes: 7)

<p>Strong BC models only started showing up during the last couple of weeks of the competition. Since I had mostly been following public replays and existing solutions, I always wondered: how did people figure out those strategies before BC models were available?
This is the GitHub repository for my solution:
<a href="https://github.com/wangmeiwen/kaggriculture-replay-planner" target="_blank">GitHub repository</a></p>

---

### Message 5 by  (Votes: 5)

<p>I'm also eager to hear from the top teams also.</p>
<p>I will share an observation I had. In this competiton, we can first build a solo bot that plays without an opponent and maximizes its money. We could achieve $200k+ this way. Then add reinforcement learning to react/adapt/challenge opponents. </p>

---

### Message 6 by  (Votes: 0)

<p>I had the same question and wait here too !</p>

---

