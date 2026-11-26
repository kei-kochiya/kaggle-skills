# Konwinski Prize: Top Solutions & Asymmetric-Abstention SWE Post-Mortem

> **Competition**: [Konwinski Prize](https://www.kaggle.com/competitions/konwinski-prize) (Dec 2024 – Jul 2025; Submission Freeze Mar 12, 2025; 472 Final Cleaned Teams / 617 Pre-Cleaning; \$1,225,000 Prize Pool — \$100k Base Top-5 Awarded)  
> **Track**: Agentic Software Engineering, Offline Code LLM Systems & Selective Abstention (Zero-Contamination SWE-bench)  
> **Compute & Runtime**: Offline `4× NVIDIA L4` (`96 GB` VRAM), internet disabled, open-weight models required; `24h` Private Forecasting budget (`9h` Public) & `30 min` hard cap per issue; **120 post-freeze GitHub issues** (`71` Public)  
> **Evaluation Metric**: Asymmetric penalty with mandatory solve floor:
> $$\text{score} = \begin{cases} -1 & \text{if } a = 0 \\ \dfrac{a - b - 10^{-4}\,c}{a + b + c} & \text{if } a \ge 1 \end{cases}$$
> where $a =$ correct patches, $b =$ wrong patches, $c =$ skipped (`SKIP`) issues

---

## 1. Official Final Leaderboard & Abstention Economics

Under `score = (a - b - 1e-4*c) / 120`, **one false positive (`-1/120`) erases one full correct solve (`+1/120`)** and costs **`10,000×`** more than skipping (`-1e-4/120`), while submitting zero solves crashes to `-1.000000`. Consequently, even the 1st-place winner abstained on **90.8% (`109/120`)** of post-freeze issues to protect a `9:2` precision ratio (nobody cleared the 30% or 90% threshold prize pools).

| Private Rank | Team | Private Score | Private `(C, W, S)` | Public Score / Rank | Core Architecture & Abstention Gate | Writeup / Code |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| 🥇 **1st** | **Eduardo Rocha de Andrade** (`arc144`) | **0.058242** | **(9, 2, 109)** | `~0.056–0.098` (#1 Pub) | **Qwen2.5-Coder-32B** + customized **Agentless 1.5**: **F2P repro tests FIRST** (2-stage test-file context, 5+5 retry → `0/10` SKIP) + `*SEARCH/REPLACE*` + **F2P & P2P baseline pytest gate** | [Writeup](https://www.kaggle.com/competitions/konwinski-prize/writeups/eduardo-rocha-de-andrade-1st-place-solution-write-) · [Notebook](https://www.kaggle.com/code/arc144/custom-agentless-fork?scriptVersionId=226880215) |
| 🥈 **2nd** | **Camaro** (`bamps53`) | **0.016576** | **(7, 5, 108)** | `0.028–0.070` (#2 Pub) | **Qwen2.5-Coder-32B** via vLLM (`130k` ctx) + JSON-schema `old_string`/`new_string` edits + 5-retry F2P repair loop + 3-part LLM verification gate | [Writeup](https://www.kaggle.com/competitions/konwinski-prize/writeups/camaro-public-2nd-place-solution) |
| 🥉 **3rd** | **akami** (`akiyama`, `sakami`, `kami`) | **0.016575** | **(6, 4, 110)** | — | **DeepSeek-R1-Distill-Llama-70B-AWQ** + **LoRA (`r=32`) difficulty classifier** trained on SWE-bench Verified (`P(easy) < 0.5 → SKIP`) + `unidiff` / `patch --dry-run` (zero pytest execution) | [Writeup](https://www.kaggle.com/competitions/konwinski-prize/writeups/3rd-place-solution) |
| 🏅 **4th** | **Genxxsky** | **0.016571** | **(4, 2, 114)** | `-0.000097` (`1,1,69`) | **Tree-sitter AST function/class slicing** + lean patch/verify + **dual-state pytest gate** (`test_before_patch` passes unpatched/fails patched; `test_after_patch` inverse) | [Writeup](https://www.kaggle.com/competitions/konwinski-prize/writeups/4th-place-solution) · [Notebook](https://www.kaggle.com/code/genxxsky/4th-place-solution-select-patch-verify-test) |
| 🏅 **5th** | **Kenj** (`ManhNguyen315`) | **0.008239** | **(4, 3, 113)** | — | **DeepSeek-R1-Distill-Qwen-14B** (zero pytest): **skip all non-traceback issues**, deterministic regex traceback localization + `*SEARCH/REPLACE*` → `git diff .` | [Writeup](https://www.kaggle.com/competitions/konwinski-prize/writeups/5th-place-solution) · [Notebook](https://www.kaggle.com/code/manhnguyen315/5th-place-solution) |
| 🏅 **8th** | **first lucky gold!** (`yuanzhe zhou`) | **0.008235** | **(2, 1, 117)** | — | `Select-Patch-Verify` tuned for ultra-high precision on narrow solved subset | [Writeup](https://www.kaggle.com/competitions/konwinski-prize/writeups/first-lucky-gold-8th-place-solution) |
| **15th** | **GraphNINJAs** (`isakatsuyoshi` et al.) | **-0.000096** | **(2, 2, 116)** | **`+0.056243`** (`4,0,67`, **#4 Pub**) | **DeepSeek-R1-Distill-Qwen-32B-AWQ** + edit-distance medoid selection + heuristic char/file filters tuned on 71 Public issues (classic shakeup cautionary tale) | [Writeup](https://www.kaggle.com/competitions/konwinski-prize/writeups/graphninjas-public-4th-place-private-15th-place-so) |

---

## 2. Core Architectural Breakthroughs Across Winning Solutions

### 2.1 Test-First Agentless Pipeline & Dual F2P/P2P Verification (#1 Eduardo, `9–2`, `3.5×` #2's Score)
Eduardo won 1st place by a decisive `+0.0417` margin (`a - b = +7` vs. `+2` for 2nd–4th) by flipping the standard Agentless order and treating **reproduction test verification as the primary abstention gate**:
1. **Stage 1 — Context-Aware F2P Reproduction Tests FIRST**:
   - Upgraded Agentless 1.5 (which generates reproduction tests blind) with **2-stage test-file context retrieval**: (i) select relevant unit-test files from precomputed repo AST skeletons (computed once/repo, saving `5–20s`/step), (ii) inspect requested test classes/methods + imports.
   - Sampled **5 diverse F2P scripts** with layered context & decoding (`1` full test context, `1` imports-only, `3` zero context; `1` greedy + `4` sampled) printing `"Issue reproduced"`, `"Issue resolved"`, or `"Other issues"` (discarding ambiguous errors).
   - **Abstention Gate 1**: If `0/5` reproduce, retry `5` more at higher temperature; if `0/10` reproduce, **SKIP immediately** (zero compute wasted on localization/patching!).
2. **Stage 2 — 3-Step Hierarchical Localization & `*SEARCH/REPLACE*` Patching**:
   - Greedy file localization → greedy class/method localization → `2×` sampled line-level edit locations → `4` patches per location (`1` greedy + `3` sampled = `8` candidates).
   - Emitted **Aider-style `*SEARCH/REPLACE*` blocks** converted via `git diff` rather than raw unified diffs (dramatically higher valid-apply rate on 32B models).
3. **Stage 3 — Dual Execution Verification (Abstention Gate 2)**:
   - `git apply --check` dry-run → retain only patches passing $\ge 1$ valid F2P reproduction test → run model-selected **existing repo P2P unit tests (`pytest -rA`)** in timeout-protected parallel subprocesses and submit **only if P2P pass/fail counts are equal or better than the unpatched baseline** (otherwise `SKIP`).
   - Total Private runtime: only **5–7 hours** (`12 min`/issue cap, `20h` global cutoff).

### 2.2 Alternative Abstention Paradigms (#2, #3, #4, #5)
- **Structured JSON-Schema Self-Repair (#2 Camaro)**: Served `Qwen2.5-Coder-32B` via vLLM at `130k` context with structured JSON (`file_path`, `old_string`, `new_string`), feeding match/test errors back for up to 5 retries, gated by a 3-part LLM verification check (`test valid`, `issue fixed`, `no regression`).
- **Learned LoRA Difficulty Classifier — Zero Pytest Execution (#3 `akami`)**: Loaded **DeepSeek-R1-Distill-Llama-70B-AWQ** once in VRAM and swapped a **rank-32 LoRA classifier** fine-tuned on SWE-bench Verified (`400` train / `100` val) predicting `easy (<15m) / medium / difficult`, skipping immediately whenever $P(\text{easy}) < 0.5$ — proving learned calibrated abstention transfers cleanly to unseen repos without sandbox test overhead!
- **Tree-Sitter AST Slicing + Dual-State Pytest Gate (#4 Genxxsky)**: Replaced fixed line-window context (`±12` lines) with **tree-sitter enclosing class/function extraction**, paired with dual-state tests (`test_before_patch` must pass→fail, `test_after_patch` must fail→pass).
- **Deterministic Traceback Localization (#5 Kenj)**: Exploited high-signal Python error tracebacks via regex file/line extraction + `*SEARCH/REPLACE*`, skipping all non-traceback issues without running pytest.
- **Consensus Edit-Distance Medoid Selection (#15 GraphNINJAs)**: Among valid patches (`len <= 1000`), picked the patch minimizing total pairwise Levenshtein edit distance to all other candidates (rescued earlier `LB = -1` crashes).

---

## 3. Empirical Negative Results & What Failed

| Tested Hypothesis | Observed Outcome | Root Cause |
| :--- | :--- | :--- |
| **Reasoning Models (`QwQ-32B`, `DeepSeek-R1`) & `Qwen2.5-72B` vs. `Qwen2.5-Coder-32B` (#1, #2, #15)** | Slower & lower precision than `Qwen2.5-Coder-32B` | Long `"but wait..."` CoT loops exhausted per-issue time caps without improving patch apply or test-pass accuracy. |
| **Open-Ended Agentic Loops (`OpenHands`, Multi-Turn Bash) (#1, #2)** | Underperformed structured **Agentless** pipelines | 14B–32B local models drifted in multi-turn tool loops and were far harder to verify/gate than deterministic 3-stage pipelines. |
| **Raw Unified `git diff` Output (#1, #4, #5)** | Frequent patch corruption & failed applies | 14B–32B models miscount diff hunk line offsets; `*SEARCH/REPLACE*` or JSON `(old_string, new_string)` → `git diff` was strictly superior (rule-based hunk line-count fixers failed randomly). |
| **Over-Broad Localization (`5 → 10` files, embeddings, 2nd pass) (#2)** | Higher `file_recall` but **worse final score** | Extra noisy context induced unnecessary/hallucinated edits that tripped regressions. |
| **Public-LB Heuristic Threshold Tuning (#15 GraphNINJAs)** | **Public #4 (`4, 0, 67`) → Private #15 (`2, 2, 116`)** | Tuning narrow char-length (`1802–3400`), file-count (`<= 388`), and odd-attempt caps (`challenges=5`) on 71 Public issues severely overfit Public LB. |
| **14B Models on Complex Contexts (#2)** | Context loss & frequent syntax failures | `32B` was the practical reliability floor for full-repo localization + multi-file test generation (except #5's narrow traceback-only regime). |
| **Extra Post-Patch Generated Filter Tests (#2) & Heavy Multi-Patch Exploration (#4)** | Over-rejected good patches or timed out (`-0.0084` Private in #4's `Choose` variant) | Post-hoc LLM tests were too brittle to control; lean, test-first pipelines dominated. |

---

## 4. Universal Takeaways Promoted / Reinforced

- **Execution-Verified & Learned Abstention Under Asymmetric Penalties**: Whenever false positives carry severe penalties (`+1 / -1 / 0`), gate submissions via **reproduction-first execution tests (F2P + baseline P2P regression)** or a **supervised calibrated difficulty/abstention classifier** trained on external benchmarks—never threshold raw text heuristics on small Public splits.
- **Structured `SEARCH/REPLACE` → `git diff` > Raw Unified Diffs**: For 14B–32B local code LLMs, emit exact search-and-replace blocks or JSON `(old_string, new_string)` edits and let `git diff` synthesize valid hunk headers.
- **Structured Agentless Pipelines + AST Slicing Beat Open-Ended Loops on 32B Local Models**: Precomputing repo AST/tree-sitter skeletons and running fixed localize → reproduce → patch → verify stages delivers higher precision and predictable latency than unconstrained multi-turn agent loops.
