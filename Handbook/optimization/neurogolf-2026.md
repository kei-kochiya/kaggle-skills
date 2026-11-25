# The 2026 NeuroGolf Championship — Top Gold Solutions & Post-Mortem

> **Competition**: [The 2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026) (Apr 15 – Jul 15, 2026 | IJCAI-ECAI 2026 | 2,963 Teams | 492,086 Submissions | \$50,000 Prize Pool incl. \$10k Longest Leader)  
> **Task**: Solve all **400 ARC-AGI-1 tasks** (`train` + `test` + `arc-gen` + hidden private tests) using static-shape `[1, 10, 30, 30]` `float32` **ONNX** computation graphs (`output > 0` exact cell match; `Loop`/`Scan`/`NonZero`/`Unique`/`If`/`Compress` banned; max `1.44 MB`/model)  
> **Evaluation Metric (Post–May 4 Reset)**: $\sum_{i=1}^{400} \max(1, 25 - \ln(\max(1, \text{params}_i + \text{peak\_mem\_bytes}_i)))$ profiled in ORT `1.24.4` (max theoretical `10,000`; `input`/`output` excluded from memory; MACs removed May 4 → compute free up to Kaggle global runtime cap!)

---

## Official Top Gold Leaderboard Standings & Solution Index

Zero gradient-trained networks appeared near the top: **100% of gold solutions were exact algebraic ONNX circuits** synthesized by **autonomous multi-LLM factories** (combining vertical deep reasoning with horizontal cross-task idiom propagation, atomic validation harnesses, and **Multiple-Choice Knapsack (MCKP)** global runtime assembly):

| Rank | Team | Score | Margin vs #1 | Core Winning Paradigm | Writeup / Code |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **`Kaggle Agent`** | **8314.03** | — | **Vertical deep thinking** (80-min single-turn ChatGPT Pro: `33%` hit rate for `+0.5`/task above `7900`! + Codex + Claude Opus) + **horizontal task-clustering transfer** + **Self-Evolving Prompts** (4h mini-contests promoting winning context to manager writing `cookbook.md`) + **MCKP** runtime packing | [Writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/1st-place-kaggle-agent) · [Dashboard](https://daxiongshu.github.io/kaggle-agent-neurogolf-2026/) · [Prompts (`726883`)](https://www.kaggle.com/competitions/neurogolf-2026/discussion/726883) |
| 🥈 **#2** | **`neurogolf team`** | **8243.89** | `-70.14` | 5 independent pipelines via HF Space DB + MCP + Modal; **single-scalar (`params=1, mem=0 → score 25`) float32 subnormal underflow `Einsum`** (`task359`), static **`RNN(Relu)`** replacing banned `Scan` (`task138`), **homotopy factorization** (`16→12`), & 52-var `Einsum` splitting | [Writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/work-in-progress-2nd-place-neurogolf-team) · [GitHub](https://github.com/bogoconic1/neurogolf-2026-2nd-place-solution) |
| 🥉 **#3** | **`The Final Ensemble`** | **8221.27** | `-92.76` | Dual Codex CLI (`~70%` late gains) + ChatGPT UI (`~30%`) sharing `attempt_ledger` + **`negative_knowledge.md`**; forced agents to price **3 paradigms** before editing; consensus verification across top-10 Code Golf 2025 solvers + fresh ARC-GEN + MCKP | [Writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/3rd-place-solution-writeup) |
| 🏅 **#4** | **`M & H & M & N & M`** | **8142.02** | `-172.01` | Hash-keyed async loop + **live proof-voted Idiom System (`model_md5`)**; **224/400 single-node models** (`193 Einsum`), **ULP-collision 2-param `Einsum` (`score 24.307`, `89` `+2` proofs!)**, & **zero-cost negative-pad `LpPool` (`score 25` on 9 tasks!)** | [Writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/4th-place-solution-writeup) |
| 🏅 **#5** | **`claudex`** | **8118.97** | `-195.06` | Two sub-teams **swapping 200-task ranges every 1–2 days** to escape local optima + score-band operator libraries | [Writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/6th-place-solution-claudex) · [GitHub](https://github.com/PNTAN17/neurogolf) |
| 🏅 **#6** | **`Slow and Steady`** | **8067.63** | `-246.40` | Curated 57 expert ONNX golfing rules (`R1–R57`), single 5-input `Einsum` cutting `task110` **`22,871 → 110`**, 50k stress tests | [Writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/6th-place-solution-slow-and-steady) |
| 🏅 **#7** | **`kluges clueless`** | **8048.16** | `-265.87` | **330-worker LLM factory** (~$\$300$/mo flat-rate CDP) + **Git-as-blackboard** (`instructions/CURRENT.md` steered all 330 workers in 1 commit) + MAP-Elites/MCTS seeding; `ConvTranspose` role-swap + `NaN` masking (`30→4`); documented ORT contraction-order **`39s → 0.07s`** speedup | [Writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/8th-place-writeup-draft) · [GitHub](https://github.com/alexeyshmelev/autoresearch) |
| 🏅 **#8** | **`Pavel & CroDoc`** | **8038.62** | `-275.41` | Independent parallel attacks + periodic merge, 200-line `MEMORY.md`, "target gaslighting" (`"others hit +1.0"`), 202/400 1-node models + MCKP timeout rescue | [Writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/8th-place-gold-pavel-and-crodoc) · [GitHub](https://github.com/CroDoc/neurogolf-2026) |
| 🏅 **#9** | **`Timeout exceeded : (`** *(solo)* | **8028.25** | `-285.78` | **Transactional 3-file resumable Codex scheduler (`backup → validate → promote/restore`)**: 5 unattended days **`+59.77` across 271 tasks, 0 regressions!** 226-operand `Einsum` (`task001` cost `98`) + congruence certificates (`606→70`) | [Writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/9th-place-solution) |

---

## 1. Universal Multi-Agent Engineering & Exact Graph-Golf Breakthroughs

1. **Multi-Agent Competitive Engineering Principles (`#1`–`#9` — Directly Transferable Beyond NeuroGolf!)**:
   - **Blank-Slate Architectural Rewrites (`+0.5`/task) >> Incremental Pruning (`+0.05`/task) (`#1`, `#3`)**: Agents left to iterate locally get stuck in suboptimal basins; forcing agents to price 3 distinct paradigms before editing (`#3`) or swapping task ownership every 1–2 days (`#5`) broke plateaus.
   - **Vertical Deep Reasoning + Horizontal Idiom Propagation (`#1`, `#4`, `#7`, `#8`)**: Use high-thinking models for breakthrough discoveries, then broadcast proven recipes across the remaining tasks via a **proof-verified idiom registry (`model_md5` + votes, `#4`)**, **`negative_knowledge.md` (`#3`)**, **Self-Evolving Prompts / Manager `cookbook.md` (`#1`)**, or a **Git-blackboard instruction file (`#7`)**.
   - **Transactional Validation (`backup → validate → promote/restore`, `#9`) + Stress Verification (`#3`, `#5`, `#6`, `#7`)**: LLM-claimed gains failed hidden validation **~71% of the time** without fresh out-of-distribution stress generators (`#7` observed `~29%` realization rate). Consensus validation across independent reference solvers + 1,000–50,000 fresh synthetic tests (`#3, #5, #6`) and binary-search Kaggle bisection (`#4, #9, #10`) prevented hidden regressions.
   - **Global Budget Optimization via Multiple-Choice Knapsack (`MCKP`, `#1`, `#3`, `#4`, `#8`)**: When submitting an ensemble or multi-task bundle under a hard Kaggle runtime ceiling, store Pareto `(score, runtime)` candidates per task and solve an **exact MCKP** to maximize total score inside the runtime cap.
2. **Exact Algebraic Graph Compilation Tricks (ONNX / Static Tensor Graphs)**:
   - **Zero-Intermediate Direct `Einsum`**: Repeating `input` across dozens/hundreds of operands in a single `Einsum` writing directly to `output` achieved `memory = 0` (`224/400` 1-node models in `#4`; `task110` `22,871 → 110` in `#6`; `task001` cost `98` in `#9`).
   - **Float32 Subnormal Underflow & ULP-Collision Selectors (`#1`, `#2`, `#4`, `#7`)**: Calibrated tiny `float32` scalars push losing terms below `1.4e-45` (`+0.0`) while winners remain positive subnormals (**cost 1, score 25** on `task359` `#2` & `task129` `#7`), and adjacent signed `[c_hi, -c_lo]` ULP-collision pairs select exact class matches with **cost 2, score `24.307`** (`#4`).
   - **Zero-Cost Transport & `NaN` Masking (`#4`, `#7`, `#9`)**: Negative-pad `LpPool`/`Conv` (`score 25`, `cost 0` on 9 tasks `#4`), `ConvTranspose` role-swapping `input` as weights + `np.nan` trimming (`30 → 4`, `#7`), and static `RNN(Relu)` replacing banned `Loop`/`Scan` (`#2`).
   - **Contraction Order Is Part of Both Speed & Numerics (`#2`, `#4`, `#7`, `#9`)**: ORT contracts `Einsum` left-to-right — reordering operands sped up `task294` **`39s → 0.07s` (`550×`)** (`#7`), while also altering `float32` ULP rounding order!

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Gradient-Trained PyTorch Networks / Generic `onnxsim` / Pure Brute-Force Search** | Non-competitive vs compiled algebraic circuits; `onnxsim` negligible on golfed graphs (`#7`) | #1–#9 | Trained weights require thousands of parameters; exact algebraic/Einsum circuits need `0–100` params (use searchers only to seed novel graphs for LLMs!) |
| **Trusting Local ARC-GEN Validation Without Fresh OOD Stress Tests** | `~20%` point loss / only `~29%` LLM gain realization on hidden tests (`#7`) | #3, #5, #7 | LLMs silently overfit provided train/test/arc-gen edge cases |
| **Short `Conv` Bias, Dynamic `dim_param`, & Negative-Pad `MaxPool`/`AveragePool`** | Silent `0.0` bundled score / phantom `0` memory (`346` vs `4,542`) / grader hard-crash (`#2, #4, #7`) | #2, #4, #5, #7 | Always run static ONNX graph linters (`bias.len == W.shape[0]`, static `value_info`, safe op allowlist) |
| **Materializing Generator Ops (`ConstantOfShape`, `EyeLike`, `Range`)** | Inflated intermediate memory bytes (`#7`) | #7 | Materialized tensors count toward ORT peak memory; compute inline inside `Einsum` |
| **Greedy Local Pruning Without Blank-Slate Rewrites or MCKP Runtime Control** | Stalled at `+0.05`/task (`#1`); timed out on Kaggle (`#8` `8041.2 → 8038.6` fallback) | #1, #3, #8 | Architectural shifts deliver `10×` more gain than pruning; unbudgeted heavy `Einsum` graphs blow global CPU time |
