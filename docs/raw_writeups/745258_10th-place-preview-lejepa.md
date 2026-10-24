# Topic 745258: [10th place (currently)] Highlights + Code and Diagrams + LeJEPA

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/745258
- **Author**: 
- **Post Date**: 2026-10-02T20:23:42.440000
- **Votes**: 34

## Messages

### Message 1 by  (Votes: 34)

<p>I was pretty excited to apply what I learned from the Orbit Wars competition here. I tried out a lot of ideas, and some of them worked quite well. <a href="https://github.com/CarsonBurke/kaggriculture" target="_blank">Code and Diagrams</a></p>
<p>I trained an ~947k parameter actor for about 10 hours on an RTX 5090. Behavioural Cloning on leaderboard data from Sept. 27/28 and then a PPO self play league.</p>
<p>Highlights:</p>
<ul>
<li>About 2 hours before submission I recognized it was buying 3/4 land plots every game, so I gave it an entropy bonus. It seemed to improve its movement and trading Δ, but did not fix its collapsed strategy in time.</li>
<li>I should have pretrained my critic on either the leaderboard data I cloned or my BC'd actor, and then it probably wouldn't have suffered from said strategy collapse.</li>
<li>An ~800k parameter <a href="https://arxiv.org/html/2511.08544v3" target="_blank">lejepa</a> backbone improved learning speed a lot, and made it easier to share parameters between the actor and critic without worrying about gradient conflict.</li>
<li>A good league is incredibly important. You want only the toughest opponents, but that is a dynamic class, so I did screening of old versions with high posterior σ.</li>
<li>Win/Draw/Loss CE critic seems to perform better than any soft reward based on bank-margin, or dense rewards on asset liquidity (which I figured might work as a proxy for final bank).</li>
</ul>

---

### Message 2 by  (Votes: 0)

<p>Congrats on the gold medal!</p>
<p>I am a little bit curious on how did you approach action space definition? Was moving (SOUTH, WEST, etc.) part of the learning to be done? Or more like task target (WATER (2,1))?</p>

---

### Message 3 by  (Votes: 0)

<p>Thank you.</p>
<p>Yes I had the "move cardinal directions" categorical at the start. I found better results using a categorical over all 99 tiles the unit could move to (masked for unpurchased and whatever) as the destination to inform which direction to move.</p>
<p>I'm not sure if an actual pointer would have been better. I tested it but never got around to evaluating it (some things just slip past when you do so many ablations, I guess 😅).</p>

---

### Message 4 by  (Votes: 0)

<p>So wait. MOVE was categorical over all tiles, but attached to a task? Like (MOVE -&gt;8,3 &amp; WATER, occupied till its done) or a task is a different action to choose? Like MOVE -&gt; 8,3 OR WATER as 2 different decisions?</p>

---

### Message 5 by  (Votes: 0)

<p>No. Please read the code for the nuances </p>

---

### Message 6 by  (Votes: 0)

<p>Thank you very much! Will read it carefully.</p>

---

### Message 7 by  (Votes: 0)

<p>It's impressive that you can achive this rank by using such low computing resource. And thanks for sharing your experience with LeJEPA, this paper is exactly what I've been looking for—I can't believe I missed it before!</p>

---

