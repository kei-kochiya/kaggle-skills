# ARC Prize 2025: Top Solutions, Paper Awards & Test-Time Reasoning Post-Mortem

> **Competition**: [ARC Prize 2025](https://www.kaggle.com/competitions/arc-prize-2025) (Mar – Nov 2025, 1,455 Teams, \$1,000,000 Prize Pool — \$125k Progress & Paper Prizes Awarded; \$700k `>=85%` Grand Prize Unclaimed)  
> **Track**: Abstract Reasoning, Test-Time Training (TTT), Diffusion/Recursive LLMs & Program Synthesis (ARC-AGI-2)  
> **Compute & Runtime**: Offline Kaggle Code Notebook (`<= 12h`), **`4× NVIDIA L4`** (`96 GB` VRAM), internet disabled, open-source required for prizes  
> **Dataset & Metric**: `240` unseen ARC-AGI-2 tasks (`120` Public / `120` Private; `1×1` to `30×30` 10-color grids; note some test tasks have only **1** demo pair!) scored by exact pixel-for-pixel & shape **`pass@2`** (`attempt_1` or `attempt_2`, `0–100%` scale)

---

## 1. Official Top-10 Leaderboard & Research Standings

ARC Prize 2025 established a decisive hierarchy under tight test-time compute (`12h` on `4× L4`): **amortizing synthesis into massive offline synthetic pretraining + fast single-task LoRA TTT** (#1 `NVARC`, `24.03%`) crushed raw 2024 baselines (`~4–6%`), while **2D masked diffusion recursion** (#2 `the ARChitects`, `16.53%`), **encoder-heavy AIRV+TTT self-ensembles** (#3 `MindsAI`, `12.64%`), and **7M-parameter recursive networks (`TRM`)** unlocked brand-new reasoning paradigms.

| Private Rank | Team | Private LB | Public LB (Rank) | Paradigm & Key Breakthroughs | Writeup / Code |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **1st** | **NVARC** (`ivan`, `CPMP`) | **24.03%** | **27.64%** (#1) | **103k consistent `gpt-oss-120b` synthetic Python puzzles** → `3.25M` corpus → **Qwen3-4B** (16-token vocab, `bfloat16` single-task LoRA `r=256` TTT + batched DFS + 8 fixed-aug rescoring) + 2h **TRM** (`\$0.20/task`; `29.72%` late sub) | [Writeup](https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc) · [GitHub](https://github.com/1ytic/NVARC) |
| 🥈 **2nd** | **the ARChitects** (`Franzen`, `Disselhoff`, `Hartmann`) | **16.53%** | **21.67%** (#2) | Pivoted past AR ceiling (`16.94%`) to **2D Masked Diffusion `LLaDA-8B`** (2D Golden Gate RoPE + stripped padding) + **Token Algebra soft-masking recursion** (`2×51` steps, linearly rescaled logits + `<mask>`) + separate `85%` shape model + Most-Visited `pass@2` | [Writeup](https://www.kaggle.com/competitions/arc-prize-2025/writeups/the-architects-solution-2025) · [Report](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/) · [GitHub](https://github.com/LambdaLabsML/ARC2025_Solution_by_the_ARChitects) |
| 🥉 **3rd** | **MindsAI & Tufa Labs** (`Cole`, `Smit`, `Pressman`, `Osman`, `Hodel`) | **12.64%** | **15.42%** (#3) | Pruned **660M CodeT5-Large** (24 enc / 16 dec) on `>100M` tasks + reversal augs (`+100%` 0-shot) + **full-FT TTT (`45k` ex) + `10,000`-aug AIRV board majority vote** + BPE dropout + Mixup/Combine/I-O Swap + 2-ckpt self-ensemble | [Writeup](https://www.kaggle.com/competitions/arc-prize-2025/writeups/mindsai-and-tufa-labs-arc-prize-2025-solution) · [GitHub](https://github.com/jcole75/arc_2025_mindsai) |
| 🏅 **4th** | **Lonnie** | **6.67%** | **4.17%** (#344) | 2024 ARChitects `Nemo-Mini-8B` LoRA `r=32` + seed variance case study (`3.33%–6.67%` ±4-task swing across 120-task exact-match splits) | [Writeup](https://www.kaggle.com/competitions/arc-prize-2025/writeups/arc-prize-2025-competition-writeup-5th-place) |
| 🏅 **5th** | **Guillermo Barbadillo (`ironbar`)** | **6.53%** | **11.94%** (#5) | Single-task TTT parallelized on 4 GPUs + **Search & Learn (Hindsight Experience Replay / HER)** lifting BARC `23.3% → 28.3%` on ARC-1 | [Writeup](https://www.kaggle.com/competitions/arc-prize-2025/writeups/exploring-the-combination-of-search-and-learn-for) · [GitHub](https://github.com/ironbar/arc25) |
| 🏅 **6th** | **ippeiogawa** | **6.25%** | **10.28%** (#7) | 2024 ARChitects single-task TTT (`24` epochs, `top_k=4` DFS, background-line removal) | [Writeup](https://www.kaggle.com/competitions/arc-prize-2025/writeups/lb10-00-with-small-change-to-2024architects) |
| 🏆 **Paper #1** | **Tiny Recursive Model (`TRM`)** (`A. Jolicoeur-Martineau`, \$50k) | **7.8% ARC-2** (`44.6%` ARC-1) | — | **Single 2-layer 7M network** updating 2 states `(y, z)` recursively (`n=6` latent + `1` answer × `T=3` cycles, `N_sup=16` deep supervision + single-pass BCE ACT halting + `EMA=0.999`) | [arXiv:2510.04871](https://arxiv.org/abs/2510.04871) · [GitHub](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) |

*(Other Official Paper Awards: **2nd Paper (\$20k) — SOAR** self-improving evolutionary program synthesis reaching `52%` ARC-1 without human DSLs; **3rd Paper (\$5k) — CompressARC**, a `76K`-param randomly initialized VAE solved per puzzle via MDL with zero pretraining reaching `~20–34%` ARC-1 / `~4%` ARC-2.)*

---

## 2. Deep-Dive Solution Architectures & Breakthroughs

### 2.1 1st Place — `NVARC`: Offline Synthetic Program Amortization + Compact 16-Token TTT (`24.03%` Priv)
Because test-time CoT and test-time Python synthesis were **`~100×` too slow** for the 12h 4×L4 budget, `NVARC` moved reasoning and code generation **offline into massive Synthetic Data Generation (SDG)**:
1. **4-Stage `gpt-oss-120b` + `NeMo-Skills` SDG Pipeline (`15k tok/s` on 8×H100)**:
   - Collected human natural-language task descriptions (`H-ARC` + `BARC` → `716` ARC-2 train summaries) → prompted LLM to combine summary pairs (**`266,593`** mixed summaries) → generated Python input generators + unit tests (**`126,901`** valid) → generated multiple candidate transformation programs and retained **strictly consistency-checked programs** (**`103,253` consistent synthetic puzzles**).
   - Combined with `MINI-ARC`, `ConceptARC`, `RE-ARC`, and `ARC-AGI-2` train into **`3,255,481` augmented pretraining samples**.
2. **Vocab-Shrunk `Qwen3-4B` Transduction + Single-Task LoRA TTT**:
   - **Shrunk vocabulary to just `16` tokens** (digits `0–9`, `\n`, chat role/control tokens) and full-FT pretrained on `4×8×H100` for `27h` (`NeMo-RL`/Megatron).
   - **Kaggle 4×L4 Inference**: Single-task LoRA TTT (**`r=256, alpha=32`**, pure **`bfloat16`** — no 4-bit quantization since 4-bit slowed inference — with Flash Attention 2 + Unsloth) + **custom batched DFS** + **8 fixed augmentations** (identical across all candidates of a task so log-probs are directly comparable; post-comp weighting DFS visit counts by **geometric mean** of log-probs improved selection further, reaching **`29.72%`** in a late run).
3. **2-Hour `TRM` Integration**: Pretrained a `7M` TRM on `4,073` puzzles (`1.04M` augs, `8×H100`) and ran 2h TTT (`128` augs, `2,000` epochs) on Kaggle (`7.5%` standalone LB in 2h; `10.0%` in `<4h` post-deadline). Scoring TRM's 10 candidates with Qwen3 lifted `Qwen3-2B` (`21.53% → 22.50%`), though it was flat on `Qwen3-4B` (`27.22% → 27.22%`) because Qwen3 failed to rank TRM's 2–3 unique solves above its own candidates.

### 2.2 2nd Place — `the ARChitects`: 2D Recursive Masked Diffusion & Soft-Masking Token Algebra (`16.53%` Priv)
After upgrading their 2024 AR pipeline (`Mistral-NeMo-8B` + single-task TTT + speculative constant-token guessing `4.7×` speedup + DFS KV-prefix caching `5.8×` speedup + `32`-aug PoE), AR hit a hard wall at **`16.94%` Public LB** because token-by-token decoding locks into early grid mistakes.
- **2D-Aware Masked Diffusion (`LLaDA-8B`)**: Padded grids to `32×32`, assigned **2D Golden Gate RoPE** position IDs (encoding horizontal, vertical, and diagonal geometry), then **stripped padding tokens** to save FLOPs while retaining exact 2D coordinates. Pretrained `r=512` LoRA (`175k` steps on `8×H100`, `39h`), followed by single-task `r=32` LoRA TTT (`128` steps, `BS=1`, `128` random augs) on Kaggle.
- **"Token Algebra" Soft-Masking Recursion**: Instead of hard discrete re-masking, fed **linearly rescaled continuous logits** (much stronger than softmax!) summed with `<mask>` embeddings (`<color_ANY> + <mask>`) back into the network for **102 iterative refinement steps** (`2 × 51` steps with a mid-run cold restart, fresh random augmentation every step!).
- **Decoupled Shape Model + Most-Visited Selection**: A dedicated `LLaDA-8B` shape predictor (`85% ± 2%` accuracy) predicted right/bottom delimiters; `pass@2` picked the **2 most-visited discrete grids** across the 102 recursive steps (`30.5%` known-shape eval × `85%` shape ≈ `26%` expected / `21.67%` Pub / `16.53%` Priv).

### 2.3 3rd Place — `MindsAI & Tufa Labs`: Pruned Encoder-Heavy `CodeT5` + TTT/AIRV Additivity (`12.64%` Priv)
- **Encoder Depth > Decoder Depth**: Progressively pruned `CodeT5-Large` decoder from `24 → 16` layers while keeping all `24` encoder layers (`770M → 660M`), trained across `>100M` examples with span corruption + **reversal augmentations** (breaking a 4-month plateau for a **`+100%` 0-shot jump**).
- **Clean Ablations of TTT + AIRV + Self-Ensembling (77M ablation model on ARC 1.5)**:
  - `Greedy`: `4.0%` → `Beam=4`: `10.1%` Top-2 → `AIRV only`: `20.5%` → `TTT only`: `21.2%` → **`AIRV + TTT`: `36.5%` (`+261%` over beam, near-exact additivity!)** → `2-ckpt Self-Ensemble`: `39.2%` → `3-ckpt SE`: `41.2%`.
  - **Compute-Matched Self-Ensembling**: Splitting compute across 2 checkpoints (`90K/30K × 2 SE`: `55.7%` Top-2) beat double compute on 1 model (`180K/60K`: `52.4%`) by **`+6.2%`**.
  - **Augmentation & BPE Dropout Ladder**: Adding `Mixup + Combine + Input/Output Swap` lifted Geo+Color from `35.0% → 37.2%` (`+6.3%` relative); **BPE Tokenizer Dropout** (`p=0.2` train / `p=0.05` AIRV) added `+2.0 pp` (`15.4% → 17.4%`), and **board-level majority voting** beat row-wise, pixel-wise, and logit averaging.

### 2.4 1st Paper Award — `TRM` (`7M` Params) & Single-Task vs. Multi-Task TTT
- **Tiny Recursive Model (`TRM`, `arXiv:2510.04871`)**: Proved a **single tiny 2-layer network (`7M`)** with **2 explicit states `(answer y, reasoning latent z)`**, `n=6` latent steps + `1` answer update repeated `T=3` cycles (`T-1` no-grad + 1 full BPTT), `N_sup=16` deep supervision, single-pass BCE ACT halting, and `EMA=0.999` beats `27M` HRM (`7.8%` vs `5.0%` ARC-2) and `4-layer` variants (`87.4%` vs `79.5%` Sudoku-Extreme — **less depth + more recursion prevents small-data overfitting!**).
- **Single-Task TTT >> Multi-Task TTT**: Confirmed independently across #1 `NVARC`, #2 `ARChitects`, #5 `ironbar`, and `jojobagel` (`Qwen2.5-0.5B`: **`8.33%` single-task TTT vs. `<=4%` multi-task TTT** on 60 tasks/GPU; also observed TTT double descent).

---

## 3. Empirical Negative Results & What Failed

| Tested Hypothesis | Observed Result | Root Cause |
| :--- | :--- | :--- |
| **Test-Time CoT & Test-Time Python Synthesis (#1, #5)** | `0` Private solves (#5 HER on ARC-2) / `~100×` too slow (#1) | 12h 4×L4 cap starves sample budgets on 7B–8B code models (`0.8%` base BARC on ARC-2); amortizing synthesis offline into pretraining won decisively. |
| **GRPO / RL Fine-Tuning & TTRL / DPO (#3, #5)** | Every GRPO run collapsed into token repetition loops (#5); TTRL/DPO non-additive with TTT+AIRV (#3) | Small-model RL lacked stability even with 128 gens/step, higher KL, and no 4-bit quantization; TTT+AIRV already absorbed refinement gains. |
| **Synthetic Atari/DSL Generators Without Diversity (#2, #5)** | Only `1/50–1/100` VLM-reward puzzles novel (#2); `32` hand-coded DSL generators solved `0` ARC tasks (#5) | Hand-coded/game generators lacked compositional novelty compared to human-summary-recombined Python SDG (`NVARC`). |
| **4-Bit Quantization & Deterministic `batch_invariant_ops` (#1)** | 4-bit slowed inference; `batch_invariant_ops` ran `~17%` slower | Under a hard 12h TTT+DFS budget, pure `bfloat16` + FlashAttn2 throughput beat precision/determinism trade-offs. |
| **Small-LLM Execution-Feedback Self-Refinement (#5)** | No gain over independent sampling; prompting 7B for diversity made it **copy prior failed code** | 7B/8B models anchor heavily on wrong in-context code; independent sampling + HER TTT worked better (`23.3% → 28.3%` ARC-1). |
| **TRM Architectural Traps (`arXiv:2510.04871`)** | MoE SwiGLU collapsed; TorchDEQ hurt; weight-tying dropped sharply; `4` layers (`79.5%`) lost to `2` layers (`87.4%`); removing EMA dropped `87.4% → 79.9%` | Tiny data regimes require tight parameter budgets, explicit 2-state `(y, z)` separation, and strong EMA smoothing. |
| **120-Task Exact-Match Seed Variance (#4 Lonnie)** | Changing only random seed swung `3.33% → 6.67%` (**#344 Pub → #4 Priv**) | With 120 binary exact-match tasks, `±4` tasks shifts rankings by hundreds of places among clustered baseline forks. |

---

## 4. Universal Takeaways Promoted / Reinforced

- **Amortize Expensive Search/Synthesis Into Consistency-Filtered Offline Pretraining**: When test-time budgets cap reasoning depth, synthesize massive consistency-verified program corpora offline with frontier models and distill into compact vocab-shrunk models (`<=4B`) for fast test-time adaptation.
- **Single-Task TTT + Compute-Matched Self-Ensembling + Board-Level Majority / Visit Voting**: Splitting fixed test-time compute across 2 checkpoints + single-task LoRA/full-FT TTT + geometric/color/I-O-swap AIRV voting consistently outperforms spending 2× steps on one checkpoint.
- **Continuous Iterative Refinement & Recursive Low-Depth Nets**: On 2D structured grid reasoning where AR locks into early mistakes, 2D-RoPE masked diffusion with continuous soft-masking recursion or 2-layer deep-supervision recursive networks (`TRM`) break AR plateaus.
