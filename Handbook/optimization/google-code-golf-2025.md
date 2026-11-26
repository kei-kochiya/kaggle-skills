# NeurIPS 2025 — Google Code Golf Championship: Top-10 Solutions & Post-Mortem

> **Competition**: [NeurIPS 2025 — Google Code Golf Championship](https://www.kaggle.com/competitions/google-code-golf-2025) (Aug – Oct 2025, 1,142 Teams, \$100,000 Prize Pool; hosted by Google DeepMind)  
> **Track**: Program Synthesis, Exact Compression, Compiler/DEFLATE Optimization & Autonomous Code-Golfing Workflows  
> **Dataset & Metric**: `400` ARC-AGI v1 tasks (`train + test + arc-gen`, `1×1`–`30×30` grids); self-contained Python 3.11 standard-library `p(g)` files (no `numpy`) verified in isolated `python:3.11.13-slim` Docker containers across **all** test cases. Score per task = `max(1, 2500 - byte_length)` if 100% exact, else `0.001` (max theoretical score = `1,000,000`).

---

## 1. Official Top-10 Leaderboard & Human vs. LLM Gap

Google Code Golf 2025 revealed a striking **`~3,974`-point cliff** separating human code-golfing specialists (#1–#3, `961,784–962,070`, avg `<96` bytes/task!) from tightly clustered LLM-assisted and competitive-programming teams (#4–#10, `956,458–957,810`).

| Rank | Team | Final Score | Entries | Primary Paradigm & Standout Breakthroughs | Writeup / Code |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **1st** | **Code Golf International** | **962,070.000** | 384 | Human specialists + **escaped-literal LZ77 DP reencoder (`reencoder.py`)** + `re.sub` (`46/400` tasks) + slice/dimension recursion + `pysearch` | [Writeup](https://www.kaggle.com/competitions/google-code-golf-2025/writeups/cgi) · [GitHub](https://github.com/Seek64/NeurIPS-Code-Golf-2025) |
| 🥈 **2nd** | **jailctf merger** | **961,805.000** | 169 | Human specialists + **`charbrute` PyPy fuzzer** (`-100+` B, discovered pop-rotation) + `infgen` Huffman byte alignment + `var_genetic` GA (`2.91×` compression ratio) | [Writeup](https://www.kaggle.com/competitions/google-code-golf-2025/writeups/solution) · [GitHub](https://github.com/orgs/jailctf-merger-goog-golf) |
| 🥉 **3rd** | **ox jam!** | **961,784.000** | 330 | Human specialists + Rust `pysearch` expression synthesis + algebraic/regex golf | [Writeup](https://www.kaggle.com/competitions/google-code-golf-2025/writeups/3rd-place-submission-ox-jam) · [GitHub](https://github.com/xsot/google-code-golf-2025) |
| 🏅 **4th** | **FuunAgent** | **957,810.000** | 573 | **98% LLM (`gpt5-codex` parallel `N=4` sessions) + AST rule-based prompt generator** + empty-sandbox blank-slate restarts + post-`zopfli` scoring + human regexes | [Writeup](https://www.kaggle.com/competitions/google-code-golf-2025/writeups/parallel-sampling-rule-based-prompt-generation) |
| 🏅 **5th** | **HIMAGINE THE FUTURE.** | **957,568.000** | 51 | **Exact DP for optimal DEFLATE Huffman code lengths** + joint var-name/DEFLATE GA (`+~700` pts over `zlib`) + **MT19937 PRNG seed cracking** (`2025 + i`) | [Writeup](https://www.kaggle.com/competitions/google-code-golf-2025/writeups/himagine-the-future-5th-place-writeup) · [GitHub](https://github.com/key-moon/golf) |
| 🏅 **6th** | **Theoretical Syntax Golfers** | **957,527.000** | 38 | Zopfli `-8` (`+2,276` pts total) + **2-stage variable renaming (`~400` pts)**: liveness interference graph Optuna/QUBO (`3h`) → Simulated Annealing (`12h`) | [Writeup](https://www.kaggle.com/competitions/google-code-golf-2025/writeups/6th-place-solution-for-google-code-golf-championsh) · [GitHub](https://github.com/Yosshi999/ARCGolfVisualizer) |
| 🏅 **7th** | **LogicLynx** | **957,513.000** | 88 | Semi-automated `Codex` (`N=4`) + `AGENTS.md` idiom book + CRT/coprime loop flattening + random single-letter var search (`-2` to `-12B`/task) | [Writeup](https://www.kaggle.com/competitions/google-code-golf-2025/writeups/7th-place-write-up-logiclynx) · [GitHub](https://github.com/shunsuke-toba/kaggle-google-code-golf-2025/) |
| 🏅 **8th** | **import itertools** | **957,177.000** | 569 | Human CP/golf + **"DRY" (Do Repeat Yourself) compression coding** (`task238`: `251B → 223B` by expanding `281B → 338B` raw!) + Zopfli iteration sweeps | [Writeup](https://www.kaggle.com/competitions/google-code-golf-2025/writeups/8th-place-write-up-import-itertools) · [GitHub](https://github.com/Natanaelel/GoogleCodeGolf2025) |
| 🏅 **9th** | **Santa Golf Club 2500** | **957,151.000** | 290 | `pysearch` expression synthesis + systematic algorithmic simplification | [Writeup](https://www.kaggle.com/competitions/google-code-golf-2025/writeups/9th-place-write-up) · [GitHub](https://github.com/4eta/google-code-golf-2025) |
| 🏅 **10th** | **lv1.dev** | **956,458.000** | 39 | Compact Python + Zopfli compression pipeline | [Writeup](https://www.kaggle.com/competitions/google-code-golf-2025/writeups/solution-for-arc-golf-lv1-dev) · [GitHub](https://github.com/lv1-dev/google-code-golf-2025) |

*(Note: After an August 2025 metric overhaul patched `__eq__`/`exit(g)` exploits, `json.dumps` normalization accepted tuples, floats, and booleans; top-3 teams also mutually agreed not to invoke shell/Perl via `os.popen`.)*

---

## 2. Core Algorithmic & Compression Breakthroughs

### 2.1 Custom DEFLATE & Escaped-Literal Compression Engineering (Above ~200–240 Bytes)
Compression beat raw Python above `~200–240B` (`21` tasks in #1, `24` in #2; `lzma`/`bz2` lost below `~1,000B` due to header overhead). Top teams pushed well past stock `zlib` and `zopfli`:
1. **Minimal Latin-1 Raw-DEFLATE Stub**: PEP 263 `#coding:L1` maps `0x00–0xFF` 1:1 into raw string bytes; stripping the 2-byte zlib header + 4-byte Adler32 trailer (`[2:-4]`) and calling `zlib.decompress(bytes("...",'L1'),-9)` saved a net `2–3` bytes per compressed task.
2. **Exact Escaped-Literal LZ77 & Optimal Huffman DP (#1, #5)**:
   - **1st (`reencoder.py`)**: Fixed Zopfli/libdeflate Huffman trees and ran **Dynamic Programming over LZ77 parsing directly minimizing escaped Python string-literal byte length** (accounting for `\x00`, `\n`, `\r`, quote, and backslash escape penalties!).
   - **5th (`key-moon/golf`, `+~700` pts over `zlib`)**: Hill-climbed Zopfli bitstreams to eliminate escaped bytes AND derived an **exact DP algorithm for optimal Huffman code lengths** given fixed LZ77 + `cl-code` (shrinking e.g. `218B` Zopfli → `215B`).
   - **2nd (`infgen` Huffman Alignment)**: Inspected raw DEFLATE bitstreams with `infgen` and brute-forced constant padding characters to align literals cleanly with existing Huffman symbols.
3. **Automated Variable-Name Optimization (`~400`–`500+` Bytes Across Tasks)**:
   - **5th & 6th**: Built variable liveness interference graphs and ran **2-stage static Optuna/QUBO (`3h`) → Simulated Annealing (`12h`)** (#6: **`~400` pts** from renaming alone out of `+2,276` Zopfli pts) or a **joint variable-name + DEFLATE Genetic Algorithm** (#5: `+~100B`).
4. **"DRY" (Do Repeat Yourself) Compression Coding (#2, #8)**:
   - Opposite of normal code golf: duplicate code blocks, unroll loops, eliminate uppercase characters (which break RLE runs in DEFLATE's code-length table), prefer tabs over spaces (adjacent to `\n`), and prefer `'` over `"`.
   - *Concrete Ablation (#8 `task238`)*: Tight lambda golf (`281B` raw → `251B` compressed) lost by **`28` bytes** to repetitive unrolled code (`338B` raw → **`223B`** compressed!). #2 hit **`2.91×`** (`task044`) and **`2.46×`** (`task173`) compression ratios.

### 2.2 Python Golfing Idioms & Automated Expression/Character Search
- **Pop Rotation (`charbrute` discovery, #2)**: Rotating square grids 90° CCW without `zip(*g)` via `[[r.pop()for r in g]for*r,in g]` (`-6B` on `task015`).
- **Regex Grid Rewriting (`46/400` tasks in #1)**: Exploiting fixed-width `3n+1` vertical strides in `str(g)` (`(?<=5.{34})5(?=.{34}5)`), f-string tuple unpacking (`f"{*zip(*g[::-1]),}"`), conditional regex groups `(?(2)2|1)` (#5), and `eval()`-in-replacement slices (#1, #2).
- **Slice, Dimension & Coprime-CRT Recursion (#1, #7, #8)**: Eliminating loop counters via `g*2` slice bounds (`g[H*(2^k-1):]`), unified 2D/1D/0D recursion via `g*0!=0`, and collapsing nested loops into one `for k in range(...)` with coprime moduli `k%a, k%b` (`-10B` over prior public best on `task240`, #7).
- **Rust Expression Brute-Forcing (`lynn/pysearch`) & PyPy Character Fuzzing (`charbrute`)**: Exhaustively synthesized `<=14B` arithmetic/bitwise formulas (`109//c&12%c+7`, `b**c*x or c`) across #1, #2, #3, #8, #9; #2's PyPy3.11 hotspot fuzzer `charbrute` shaved **`100+` extra bytes** late in the contest.
- **PRNG Seed Cracking (#5)**: Leaked clean MT19937 bits from power-of-two `randint` tasks (`task043/127/142`), uncovering sequential **`2025 + i`** competition seeds per example index `i`.

### 2.3 LLM-Assisted Synthesis Workflows (#4 `FuunAgent`, #7 `LogicLynx`)
- **Parallel Sampling + AST-Driven Prompting (#4)**: Ran `~15` parallel `gpt5-codex` sessions (`N=4` samples/turn) with an **AST rule-based prompt generator** injecting targeted structural nudges (`no re → try regex`, `def → lambda`, `loops → recursion`, or inspect `ARCGEN` generator scripts).
- **Blank-Slate Local-Optima Escape (#4, #7)**: Showing prior code trapped LLMs in local minima; spawning **clean empty sandboxes** ("write from scratch") and clearing per-task experiment memos upon any byte drop unlocked major step-function reductions.
- **Compression-Aware Objective Scoring (#4)**: Scoring `N=4` LLM samples directly on **post-`zopfli` byte length** with *"minimal Kolmogorov complexity"* prompts produced top-1 task scores (`task324`).

---

## 3. Empirical Negative Results & What Failed

| Tested Hypothesis | Observed Outcome | Root Cause |
| :--- | :--- | :--- |
| **Pure LLM Golfing vs. Human Specialists (#1–#10)** | `~3,974`-pt cliff (`<=957,810` vs. `962,070`); `0` early `o4-mini` solutions survived (#5); #8 abandoned LLMs | LLMs struggled to invent radical structural rewrites (e.g., `re.sub` stride regexes from scratch) and got trapped locally when shown prior code. |
| **GA / Simulated Annealing Directly on DEFLATE Huffman Trees (#5)** | Underperformed exact DP | Discrete Huffman bit-length landscapes are non-smooth; exact DP solved optimal code lengths deterministically. |
| **Uncompressed-Shorter Rewrites Inside Compressed Tasks (#2, #8)** | Inflated post-`zopfli` byte count (`+28B` in `task238`) | Optimizing raw syntax increases symbol entropy and destroys LZ77 matches; compressed tasks must optimize directly for post-compression escaped literal size. |
| **`astbrute` & `charbrute` Chunk-Mover Mutations (#2)** | Abandoned (poor ROI) | AST-level search was high-complexity/low-yield compared to `pysearch` + targeted character-level fuzzing. |
| **Static QUBO/Optuna Variable Renaming Alone (#6)** | Left score on the table vs. SA | Needed 2-stage **static liveness-graph initialization → 12h Simulated Annealing** directly on exact compressed byte count. |
| **Permissive "Sacrifice Correctness for Bytes" Prompts (#4)** | Ineffective; removed | Hard 100%-exact verification gates (`0.001` score on any failure) meant approximate shortcuts wasted turns. |

---

## 4. Universal Takeaways Promoted / Reinforced

- **Optimize Directly for the True Downstream Cost (Escaped Compressed Bytes, Not Raw Source Length)**: When a deterministic post-transform (compression, quantization, escaping, or compiler pass) determines the final metric, score every search/LLM/DP candidate **after** the full downstream transform (including "DRY" repetition when it shrinks compressed entropy).
- **Exact DP Over Discrete Combinatorial Subproblems Beats Metaheuristics**: Replace noisy GA/SA on structured subproblems (like Huffman code-length or LZ77-with-escaping parsing) with exact Dynamic Programming, reserving GA/SA for outer non-separable assignments (like variable renaming).
- **Blank-Slate Restarts + AST-Triggered Hinting in LLM Code Optimization**: Avoid feeding entrenched code into iterative LLM loops; pair empty-sandbox rewrites with automated AST diagnostics and strict 100%-exact unit-test gates.
