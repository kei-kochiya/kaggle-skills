# Topic 750261: 32nd Place Solution - Opening Book & Heuristics

- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/750261
- **Author**: 
- **Post Date**: 2026-10-17T13:52:11.257000
- **Votes**: 4

## Messages

### Message 1 by  (Votes: 4)

<h1>Overview</h1>
<p>My solution used heuristics and search, without machine learning. It combined an opening book derived from top players' public replays with daily planning based mainly on simulated annealing and beam search. For the final submission, the opening book came from the public replays of <strong>M &amp; M &amp; P &amp; Q</strong>.</p>
<p>The <a href="https://www.kaggle.com/code/mtmrs1/32nd-place-solution-opening-book-heuristics" target="_blank">submission notebook</a> explains the solution in detail. The diagram below, created with Claude Opus 5.5 based on that notebook, summarizes the agent's architecture. Here, I will focus on how I approached the competition and developed the agent.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F30762370%2F9075ce319e1f55e4e654516cbf42d17a%2Fkaggriculture_overview.png?generation=1792241417036441&amp;alt=media" alt="Overview"></p>
<h1>From replay-based strategies to AI-driven heuristic development</h1>
<p>When I first looked into this competition, building an agent that could adapt to the game state seemed very difficult. Submitted agents' actions were also exposed through public replays. Some discussions suggested that there might be little room for improvement beyond copying strong fixed scenarios and making small modifications. Initially, I shared that view.</p>
<p>I started with <a href="https://www.kaggle.com/code/yhay81/three-day-shop-router" target="_blank">Three-Day Shop Router</a>, the strongest public notebook I had found at the time. I asked GPT-6 Pro to improve it, and in a single chat turn, it produced a version that substantially outperformed the original in local matches.</p>
<p>That experiment convinced me to pursue an AI-driven approach to improving fixed scenarios and their surrounding heuristics. However, I thought that simply asking the model to keep improving the agent would eventually stall, so I continued to steer the direction of development.</p>
<p>One important turning point came when the AI proposed rebuilding the plan for the final day. This made me wonder whether the same idea could be applied on earlier days as well. That led to the final architecture: an opening book followed by daily replanning.</p>
<p>For the opening book, I moved from public notebooks to collecting public replays that were both strong and easy to reuse as opening routines. I used DECEM and DSM replays for much of development. Near the end of the competition, I updated the book with M &amp; M &amp; P &amp; Q replays.</p>
<p>The approach worked better than I had initially expected. On September 23, one week before the submission deadline, my agent reached third place on the public leaderboard.</p>
<p>Other participants kept improving their agents, though, and I ended up finishing <strong>32nd with a silver medal</strong>.</p>
<h1>Notes on working with AI</h1>
<p>I mainly used GPT-6 Pro through the ChatGPT Pro 5x plan. Among the models I tried, it stood out for this kind of work. Toward the end of the competition, however, I felt that its performance had declined, and I saw other users reporting similar experiences. This was difficult to assess quantitatively, but it made it harder to make steady progress.</p>
<p>I also used Astra and GPT-5.6 Sol with xhigh reasoning, partly because of usage limits.</p>
<p>Codex and Work had very restrictive usage limits, while chat was unlimited except for Pro mode. I therefore set up GitHub Actions so ChatGPT could launch cloud instances with many CPU cores and run large batches of matches between agent versions directly from the chat interface.</p>
<p><em>I drafted this writeup in Japanese, my native language, then used GPT-6.1-Sol to translate it into English and polish the wording and formatting. I reviewed and edited the English version before publishing it.</em></p>

---

