# AI Mathematical Olympiad — Progress Prize 3 (AIMO3): Top Solutions & Post-Mortem

> **Competition**: [AI Mathematical Olympiad - Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3) (Nov 2025 – Apr 16, 2026 | 4,138 Teams | \$2,207,152 Prize Pool)  
> **Task**: Solve 50 Public + 50 Private Olympiad-level LaTeX math problems (non-negative integer answers in `[0, 99999]`)  
> **Evaluation Metric**: Private LB runs each notebook **twice** over 50 problems in randomized sequential order (`1.0` if both runs correct, `0.5` if 1/2 correct, `0.0` if neither; max `50.0`; both runs must finish without timeout/error)  
> **Hardware Constraint**: Single **1× NVIDIA H100 80GB** notebook (`<= 5 hours` / `18,000s`, internet disabled, no per-problem time cap)  
> **Overall Progress Prize (`>= 47/50` on both Public & Private)**: Unclaimed (top Private score `44.0/50`)

---

## Official Top Prize Winners & Verified Writeup Index

Every single top-ranked writeup converged on **pure inference-time systems, serving, budgeting, and aggregation engineering on pretrained `gpt-oss-120b`** (a ~117B total / ~12B active MoE model) with **zero SFT or RL fine-tuning**, building on community Tool-Integrated Reasoning (TIR) notebooks by Andreas Bisiadis, Parthenos, ZaynYu, and datasciencegrad:

| Private Rank | Team / Competitor | Private Score | Base Model & Training | Core Winning Paradigm | Writeup / Code |
| :---: | :--- | :---: | :--- | :--- | :--- |
| 🥇 **#1** | **Exalted Joseph** (`theexaltedone`) | **44.0** | `gpt-oss-120b` (Zero training) | OS page-cache weight preload + 8-bit KV-cache + prefix caching + persistent Jupyter TIR + adaptive deadline scheduler + entropy-weighted self-consistency | [Writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/1st-place-solution-for-the-aimo3-competition) · [Notebook](https://kaggle.com/code/theexaltedone/aimol) |
| 🥈 **#2** | **`varianceofx`** | **44.0** | `gpt-oss-120b` (Zero training) | **Uncatchable-leader early stopping** + **tail-256 median entropy** + **vote-dominant consensus formula** + strict `\boxed{}` parser + sharpened modular-reduction prompt (`8` parallel attempts, `16` kernels) | [Writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/2nd-place-solution-ai-mathematical-olympiad-prog) · [Notebook](https://www.kaggle.com/code/varianceofx/let-me-improve-this-cooking?scriptVersionId=306081269) · [HF](https://huggingface.co/varianceofx/aimo3-2nd-place-solver-adapted) |
| 🥉 **#3** | **`SKobayak`** (`skobayak`) | **44.0** | `gpt-oss-120b` (Zero training) | **1,120-trace offline error taxonomy** → "Avoid Frequent Errors" prompt (constraint checks, Python verification, complexity pruning before loops) + 8-run majority vote | [Writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/3rd-place-solution-for-the-aimo3-competition) · [Notebook](https://kaggle.com/code/skobayak/aimo3-prompt6-avoiding-frequent-errors) |
| 🏅 **#5** | **`TAMU-TACO`** (Peiran Li & Fangzhou Lin) | **43.5** | `gpt-oss-120b` (Zero training) | **Sequential stochastic resource allocation** optimized across **768 full 100-problem runs** (4 adversarial hard-problem orderings on 64× H100): locked `[300s, 900s]`, `attempts=8`, `early_stop=4` + `fp8_e4m3` vLLM | [Writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/aimo3-tamu-taco-solution) · [Notebook](https://kaggle.com/code/peirancs/gpt-oss-120b?scriptVersionId=292141254) |
| 🏅 **#7** | **`yemao ye`** (`yemaoye`) | **43.5** *(Pub `36–40`)* | `gpt-oss-120b` (Zero training) | Minimalist IMO prompt + simple `1 / mean_entropy` weighting over `81,920`-context FP8 vLLM; proved long prompts and complex entropy formulas hurt Public LB | [Writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/7th-place-solution-for-the-aimo3-competition) · [Notebook](https://kaggle.com/code/yemaoye/aimo-3-yemao) |
| 🏅 **#37** | **`esprit`** (`takuji`) | **43.0** | `gpt-oss-120b` (Zero training) | "Prompt to Shorten Thinking": anti-rechecking loop guardrails + exact Python `int`/`pow(a,b,m)`/`math.gcd` policy banning float/`numpy` for exact integer checks | [Writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/38th-place-solution-prompt-to-shorten-thinking) · [Notebook](https://kaggle.com/code/takuji/38th-place-prompt-to-shorten-thinking) |

*(Note: Official prize winners announced in `/discussion/714435` were #1 `Exalted Joseph`, #2 `varianceofx`, #3 `SKobayak`, #5 `TAMU-TACO`, and #7 `yemao ye`. Additional Special Prizes: **Hard Problem Prize** `\$30k` to #14 `Md. Hamid Hosen` [`43.5`, sole solver of `ACUTES` in both runs]; **Writeup Prizes** `\$15k` each to `Geremie Yeo & Chan Ka Vu` [Nemotron Cascade 2 FP8/NVFP4 + EAGLE-3 + hybrid-Mamba vLLM] and `natnitarach` [Model Capability Dominates]; **Math Corpus Prize** `\$30k` to `Yi-Chia Chen` [*CrystalMath*].)*

---

## 1. Competition Mechanics & Why Pure Inference Engineering Won

### 1.1 Dual-Run Private Evaluation & Sequential API
- **Stochasticity Penalty**: Running the entire 50-problem suite **twice** with independent random problem orderings (`1.0 / 0.5 / 0.0` scoring) sharply penalized brittle timeouts, ordering sensitivity, and high-variance sampling. A single crash or timeout on either run invalidated the entire submission.
- **5-Hour Budget Algebra**: Out of `18,000s` total (`5h`), cold-starting vLLM and loading a 120B MoE checkpoint consumed `~400–600s`, leaving `~17,400s` (`~348s` mean per problem). Because problem difficulty was wildly skewed (some solved in `<60s`, hard problems needing `600–900s`, and unseen hard problems could cluster early or late in the random stream), **online sequential time allocation** was a first-class competitive lever.
- **Why Fine-Tuning Lost**: Per host commentary (`/discussion/714435`) and ablation writeups (`ofidaner`'s behavioral regularization SFT study), `gpt-oss-120b` proved extremely fragile to SFT/RL under competition compute limits, regressing across ~10 ablation dimensions. Because top solutions clustered tightly at `43.0–44.0 / 50`, small gains in throughput, voting robustness, parsing hygiene, and timeout survival separated 1st place from 50th.

---

## 2. Systems & Serving Optimization on Single 80GB GPU

Running a 120B-parameter MoE model at `65,536–81,920` context with **8 simultaneous parallel tool-using rollouts** inside 80GB VRAM required tight systems discipline across 1st, 2nd, 5th, and 7th place:

| Engineering Lever | Exact Configuration Used by Top Solutions | Measured / Structural Benefit |
| :--- | :--- | :--- |
| **OS Page-Cache Weight Preloading** (1st) | Pre-read model shards sequentially into Linux page cache prior to vLLM launch | Eliminates fragmented random disk reads; cuts startup overhead from total 5h clock |
| **8-bit FP8 KV-Cache** (1st, 2nd, 5th, 7th) | `kv_cache_dtype="fp8_e4m3"`, `gpu_memory_utilization=0.96` | Doubles usable KV memory so all **8 parallel attempts** run in a single GPU wave (`68.6–74.2 GB` VRAM at 16k–32k tokens/path) without multi-wave queuing |
| **Prefix Caching + Async Scheduling** (1st, 5th) | `--enable-prefix-caching`, `--async-scheduling`, `batch_size=256` | Shares system + problem prompt KV blocks across all 8 parallel trajectories |
| **Pre-Warmed Persistent Jupyter Pool** (1st, 2nd, 5th, 7th) | **16 pre-started stateful kernels** for 8 workers (`2×` headroom) with `math, numpy, sympy, mpmath (64dp), itertools, collections` pre-imported | Zero cold-start import latency; preserves variables and intermediate algebraic objects across multi-turn TIR steps |
| **Prompt Token Hygiene** (2nd) | Shrunk tool preference hints from ~200 tokens to 1 concise line | Reclaimed ~200 tokens × `8 attempts × up to 128 turns × 50 problems` |
| **Streaming Early-Halt per Attempt** (5th) | Stop streaming a trajectory the instant a valid `\boxed{integer}` completes | Prevents wasteful post-answer chatter and frees KV cache immediately |

---

## 3. Sequential Time Budgeting & Early-Stopping Algorithms

### 3.1 Adversarial Ordering Calibration (5th Place: `TAMU-TACO`)
Rather than guessing timeouts by hand on 10 reference problems, `TAMU-TACO` framed AIMO3 as **sequential stochastic resource allocation** and evaluated hyperparameters across **768 full 100-problem runs** (12 runs per vector = **4 adversarial hard-problem placements** [`hard_front50`, `hard_middle50`, `hard_back50`, `hard_random`] × 3 repeats on 64× H100):
- **Phase 1 (46 rounds / 552 runs)**: Locked `base_problem_timeout = 300s` (stable window `288–306s`; never `<200s` since the fastest correct AIMO3 reference trace took `212s`) and `high_problem_timeout = 900s` (stable window `894–906s`).
- **Phase 2 (18 rounds / 216 runs)**: Locked **`attempts = 8`** and **`early_stop = 4`** (`>8` attempts risked OOM or forced slower two-wave execution on 80GB H100).

### 3.2 Mathematically Uncatchable-Leader Early Stopping (2nd Place: `varianceofx`)
Standard fixed-threshold early stopping (`stop at 4 votes`) wastes GPU time when an answer has already mathematically clinched a plurality (e.g., 3 votes vs. 1 vote with only 1 attempt remaining, or 4-0). `varianceofx` stopped sampling as soon as:
$$\text{votes}(\text{leader}) \ge \text{early\_stop} \quad \text{OR} \quad \text{votes}(\text{second}) + \text{remaining\_attempts} \le \text{votes}(\text{leader})$$
Reallocating those saved seconds directly to remaining hard problems improved end-of-run survival without sacrificing consensus accuracy.

### 3.3 Graceful Deadline Degradation (1st Place: `Exalted Joseph`)
As remaining wall-clock time approaches the 5-hour cap, the scheduler dynamically trims parallel attempt count and verification turns, guaranteeing every remaining problem produces a valid best-effort integer (with deterministic fallback `0` if parsing yields nothing in `[0, 99999]`) so neither private run ever triggers a catastrophic timeout.

---

## 4. Aggregation, Confidence Weighting & Parsing Hygiene

### 4.1 Strict Answer Parsing (2nd Place)
Public baseline notebooks included a regex fallback catching prose like `"the final answer is N"`. `varianceofx` discovered this misfired on intermediate scratchpad derivations (e.g., *"the final answer is 42 for the base case..."*). Fixing the extractor to:
1. Parse **strictly `\boxed{...}`** (removing the prose fallback completely),
2. Strip formatting commas and whitespace (`\boxed{42,000}` → `42000`),
3. Take **last-match-wins** and validate integer range `[0, 99999]`,
eliminated silent extraction corruption.

### 4.2 Majority Voting vs. Entropy-Weighted Consensus
Top solutions converged on three clean aggregation regimes—and warned explicitly against over-engineered entropy heuristics:

| Strategy | Formula / Rule | Evidence & Trade-Off |
| :--- | :--- | :--- |
| **Simple Inverse Mean-Entropy** (1st, 5th, 7th) | $w_i = 1 / \bar{H}_i$, where $H_t = -\sum_{k=1}^{5} p_{t,k}\log_2 p_{t,k}$ from top-5 vLLM logprobs; sum $w_i$ per answer | Outperformed complex 5-factor entropy formulas (7th place: `36–40` vs `36–39` Public LB); zero extra hyperparams |
| **Vote-Dominant + Tail-Windowed Median Entropy** (2nd) | Compute entropy strictly on the **last 256 tokens** (`logprob_tail=256`); take **median** entropy per candidate; score via:<br>$\text{score} = 2.0 \cdot \text{vote\_share} + 0.3 \cdot \text{conf} + 0.5 \cdot \mathbb{I}[\text{top\_votes} \land \text{conf} > 1]$ | Fixes two failure modes of pure `sum(1/H)`: (1) stops one ultra-low-entropy wrong fluke from beating a 5–1 majority, and (2) ignores noisy early-exploration entropy |
| **Pure Majority Vote** (3rd) | Unweighted plurality across 8 parallel runs with early stopping | Simplest baseline; tied for 1st–3rd at `44.0` when paired with strong error-avoidance prompting |

---

## 5. Prompt Engineering & Offline Failure-Mode Mining

- **Data-Driven Failure Taxonomy (`SKobayak`, 3rd Place)**: Logged **1,120 full traces** (`32 × 10` reference + `16 × 50` external hard problems) and analyzed errors with a stronger external LLM anchored on correct `gpt-oss-120b` traces so advice stayed within model capability. Distilled 3 concrete rules: (1) check every problem constraint against candidates, (2) derive mathematical reductions before Python verification, and (3) estimate loop complexity and prune before brute-forcing.
- **Exact Arithmetic & Anti-Loop Rules (`esprit`, 37th Place; `varianceofx`, 2nd Place)**:
  - Explicitly remind the model to check for modular reduction whenever a raw integer exceeds `99999`.
  - Prioritize Python exact arbitrary-precision `int`, `pow(a, b, m)`, `math.gcd`, `math.comb`, and explicit `sympy` assumptions (`integer=True, positive=True`); avoid `numpy` floating-point arithmetic for exact integer verification and restrict slow symbolic calls (`factorint`, `groebner`) to avoid timeouts.
- **Prompt Complexity Caution (`Exalted Joseph`, 1st; `yemao ye`, 7th; `natnitarach`, Writeup Prize)**: Overly verbose prompts consumed context and degraded Public LB in controlled 6-run tests (`35–39` structured 5-step vs `36–40` concise IMO prompt in 7th place). Base model capability dominated prompt engineering by ~4× (`natnitarach`).

---

## 6. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Outcome | Source | Root Cause |
| :--- | :--- | :--- | :--- |
| **Scaling Parallel Attempts `8 → 16` on Single H100** | Degraded precision & increased total errors; OOM / multi-wave risk | #2, #3, #5 | Exceeds single-wave 80GB KV-cache capacity (`68–74GB` at `N=8`) and triggers premature early completions / timeout starvation |
| **Naive SFT on `gpt-oss-120b`** | Regressed across ~10 ablation dimensions | `ofidaner`, Host Notes | Perturbs finely tuned MoE reasoning & tool-use distributions under limited competition compute/data |
| **Diverse Multi-Persona Voter Prompts** | Zero measurable gain over one crisp prompt across >20 controlled runs | #2, `natnitarach` | Persona diversity degrades weaker prompts more than it decorrelates errors on IMO-grade math |
| **Lowering Sampling Temperature `< 1.0`** | Hurt reasoning diversity & self-consistency | #2 | Collapses parallel attempts onto identical wrong paths |
| **Complex Multi-Factor Entropy Formulas** | Public LB `36–39` vs `36–40` for simple `1/entropy` | #7 | Amplifies token-entropy noise and introduces untunable hyperparameters |
| **Unadjusted Single-Split Evaluation on Small Benchmarks** | 0/16 modifications beat baseline after Benjamini-Hochberg correction (`N=199`) | `Hail Mary` | High stochastic decoding variance (`N=50` Public LB) creates pervasive false positives without repeated runs & multiple-testing control |
| **Prose Regex Answer Fallbacks (`"final answer is N"`)** | Corrupted extracted answers on base-case intermediate steps | #2 | Intermediate proof steps frequently match natural-language summary phrases; restrict strictly to `\boxed{}` |
| **Two-Step Forced Draft-Then-Finalize / Self-Correction** | Discarded after zero gains | #3 | Wastes limited token/time budget compared to independent parallel TIR trajectories |
