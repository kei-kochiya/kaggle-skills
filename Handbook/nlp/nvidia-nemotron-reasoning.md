# NVIDIA Nemotron Model Reasoning Challenge — Top Gold Solutions & Post-Mortem

> **Competition**: [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) (Mar 16 – Jun 15, 2026 | 4,185 Teams post-cleanup [4,354 raw] | 71,952 Submissions | \$106,388 Prize Pool incl. DGX Sparks)  
> **Task**: Post-train **`Nemotron-3-Nano-30B-A3B-BF16`** (52-layer Hybrid Mamba-2 + Transformer + MoE, 30B total / 3.5B active) via a **PEFT LoRA adapter ONLY** (`max_lora_rank = 32`; zero custom inference code/tools) across **7 inductive reasoning & cipher domains** (`bit_manipulation`, `equation_numeric`, `cryptarithm`, `text_cipher`, `numeral_system`, `unit_conversion`, `gravity`)  
> **Evaluation Metric**: Exact / relative-tolerance (`1e-2`) accuracy on `\boxed{...}` answers under fixed vLLM scoring (`max_tokens=7680`, `max_model_len=8192`, `temp=0.0`, `RTX Pro 6000`)  
> **Dataset**: `train.csv` (`9,500` labeled prompts, `3.07 MB`) | `~500` Hidden Test prompts (`250` Public / `250` Private — `1` problem = `0.004`)

---

## Official Top Gold Leaderboard Standings & Solution Index

**Programmatic synthetic "Trainable-Trace" SFT** completely crushed RL/GRPO/DPO across the leaderboard. Because evaluation ran a fixed vLLM loop (`temp=0.0`, `7,680` tokens) with only a `rank <= 32` LoRA adapter, winning boiled down to **separating weight-memorized lookup catalogs from token-efficient in-context verification/DFS**, **exact PEFT export without quantization/SVD loss**, and **holding out clean `train.csv` validation** against a noisy 250-sample Public LB plateau (`0.868–0.876`):

| Private Rank | Public Rank | Team | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **#1** | **`NullSira`** | **0.912** | **0.920** *(0.932 unsel.)* | **Weight-memorized signature catalogs + in-context DFS/verification** via 2-phase Unsloth BF16 LoRA SFT (`r=32, α=32`, Mamba+MoE+`lm_head`, `220k` @ `2e-4` + `104k` @ `1e-5`); **HEX bit-column compaction (`-27.8%` tokens)** + 5,238 bit-rule catalog & 4,205 cryptarithm signatures | [Writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/1st-place-solution) · [GitHub](https://github.com/xrwr/kaggle-nvidia-nemotron-model-reasoning-challenge-1st-place) · [NB](https://kaggle.com/code/refv00/nullsira-nemotron-final-submission) |
| 🥈 **#2** | #5 | **`vli`** | 0.884 | **0.908** | **100% synthetic training (`train.csv` 100% held-out val: `0.9074`)** + exact 6-family bit grammar (`0.9732` LoRA); **Megatron-Bridge exact PEFT** beat Tinker lossy SVD (`0.81 → 0.89` bit!); space-prefixed hex + `│` delimiter + trained `lm_head`; 57.6k ex / `159M` tok (`4×A100`, 11h) | [Writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-place-solution) · [GitHub](https://github.com/livctr/nvidia-nemotron) · [NB](https://kaggle.com/code/victorlifire/nemotron-tinker-eval) |
| 🥉 **#3** | #42 | **`YS-L`** | 0.872 | **0.900** *(+39)* | **Two-stage decoupled SFT**: Stage 1 cryptarithm signature memorization (`159M` tok) → Stage 2 CSP backtracking (`299M` tok + 3k drills) — **`+0.012` Priv LB** (`16/17` vs `12/17`) over direct training; HEX+tail (`-25%` tok) & **"perfect trace rate"** CV metric | [Writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/3rd-place-solution) · [GitHub](https://github.com/YS-L/nvidia-nemotron-reasoning-3rd-place-solution) · [NB](https://www.kaggle.com/code/liauys/nemotron-reasoning-3rd-place-lora-submission) |
| 🏅 **#4** | #3 | **`/make-no-mistakes`** | 0.892 | **0.900** | **"5 Mechanical Rules for Trainable Traces"** (AST/reference-integrity verified) achieving **>95% bit accuracy from only `2.5k` examples** on a \$120 budget; 5 bit templates (`99.4%` cov, `4,148` mean tok); `27.7k` total mix + untied MoE LoRA + Cut Cross-Entropy (11h) | [Writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/4th-place-solution) · [NB](https://www.kaggle.com/code/dipamc77/nemotron-training-final) |
| 🏅 **#5** | #4 | **`Domdolus Tolus`** | 0.888 | **0.896** *(0.900 iter15)* | Zero `train.csv` training (~2B synthetic tokens across 15 iters); **Phase-2 Pre/Post LoRA Averaging** (`<1%` LB jitter) + **Phase-3 Iterative Model Soup** (3 parallel synthetic branches averaged each round) | [Writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/domdolus-tolus-solution) · [NB](https://www.kaggle.com/code/darrenamadeusmartin/domdolus-tolus-nemotron) |
| 🏅 **#6** | #2 | **`Alehandreus & Yurnero`** | **0.900** | **0.896** | Per-domain Public LB probing (`\boxed{DUMMY}` adapters); 17 pure-hex bit templates w/ printed hex mul table; documented **cryptarithm vs bit-manip LoRA capacity interference** when doubling cryptarithm volume | [Writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-public-6th-private-place-solution) |
| 🏅 **#7** | #531 | **`Solved Bit Manipulation`** | — | **0.888** *(+524)* | 10-day sprint (0 cryptarithm!): `>96%` bit val via truth-table deduction + 64-row collision DFS, **single-bit tokenization**, & **dynamic loss masking** on dead-end backtracking branches | [Writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/7th-place-solution-solving-bitmanipulation-via-st) · [NB](https://www.kaggle.com/code/prateekagnihotri/synthetic-cot-bitflip-2lakh-backtracking) |
| 🏅 **#9** | #7 | **`Fate`** (`Chi-Han Tsai`) | 0.880 | **0.884** | **Zero synthetic problem expansion** — strictly curated `train.csv` folds + human-auditable CoT (`5×` Bit, `2×` Cipher/Eq, `400` Numeral, `packing=False`) | [Writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/9th-place-solution-human-auditable-cot) |
| 🏅 **#10** | #24 | **`kansai-kanto-kaggler`** | 0.872 *(sel. 0.860)* | **0.880** *(+14)* | Key-table bit reasoner (`98.75%` oracle); `lora_alpha/r = 4` (`r=32, α=128`); **low-logprob retraining** targeting fragile reasoning steps; trusted `0.904` CV over `0.860` Pub | [Writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/10th-place-solution) |

---

## 1. Universal LLM Reasoning Post-Training Breakthroughs

1. **The 5 Mechanical Laws of "Trainable Synthetic CoT Traces" (`#4`, `#1`, `#2`, `discussion/688461`)**:
   - Naive solver dumps fail to train LLMs efficiently. `#4` proved that mechanically enforcing 5 properties (verified by AST/reference-integrity tests) enabled **>95% accuracy from just `2,500` examples**:
     1. **No hidden compute**: Every emitted token must be locally derivable from visible context + a tiny fixed rule set.
     2. **Locality**: Perform each calculation within `~10–20` lines of its operands.
     3. **Causal consistency**: Trace branching must depend strictly on prior tokens — never "solve externally first, then decorate with plausible text".
     4. **Reference integrity**: Quoted strings/numbers must appear verbatim earlier in the trace.
     5. **Bit-serial / character-by-character execution + independent verification**: Parallel 8-bit ops collapsed to `~9.3%`, whereas serial step-by-step execution + round-trip consistency checks hit `98–100%`.
2. **Decoupling Weight-Memorized Lookup Catalogs from Sequential Execution (`#1`, `#3`)**:
   - Rather than forcing an LLM to discover combinatorial spaces from scratch inside `7,680` tokens, `#1` and `#3` precomputed canonical normalized lookup catalogs (e.g., `4,205` cryptarithm signatures / `5,238` bit-rule signatures), memorized them in LoRA weights, and trained the CoT to emit candidate recall + **sequential DFS backtracking + verification**.
   - **Controlled Ablation (`#3`)**: Training **Stage 1 (signature memorization) → Stage 2 (CSP backtracking execution + memory drills)** crushed joint single-stage training: `7/11` vs `0/11` at 12k samples, and **`0.900` vs `0.888` Private LB (`+0.012`)** on full 40k+ data.
3. **Tokenizer-Aware Token-Budget Compaction (`#1`, `#2`, `#3`, `#7`)**:
   - Nemotron tokenizes digits/bits 1-per-token, making raw binary tables blow past the `7,680`-token cutoff.
   - Compressing bit columns into **HEX (`-25%` to `-27.8%` tokens, `#1`, `#3`)**, **space-prefixing hex digits `A-F`** so BPE doesn't merge adjacent hex chars (`#2`), forcing **single-bit tokenization** when individual bit alignment matters (`#7`), and **always including `lm_head` (`output_layer`) in LoRA targets** when introducing custom formatting/delimiters (`#1`, `#2`, `#4`) were decisive.
4. **Hybrid Mamba-2 + MoE LoRA Engineering & Exact PEFT Export (`#1`, `#2`, `#4`, `#5`)**:
   - Targeting **Mamba (`in_proj, out_proj`) + MoE (`up/down/gate_proj`, untied experts) + Attention (`q/k/v/o`) + `lm_head`** at full `rank=32` in BF16 was the universal winning configuration (`bs=12–32`, `LR=2e-4` cosine cooldown, Cut Cross-Entropy `#4`).
   - Switching from **Tinker's lossy SVD-to-PEFT conversion** to **Megatron-Bridge exact PEFT export** jumped bit accuracy from **`0.81 → 0.89`** with identical data (`#2`).
   - **Iterative LoRA Model Soup (`#5`)**: Averaging LoRA weights before/after training rounds damped LB variance to `<1%`, and branching 3 parallel synthetic runs into an iterative soup stabilized generalization (though `#10` found naive post-hoc SVD/seed merging did not beat the best single run).
5. **100% Held-Out `train.csv` Validation & "Perfect Trace Rate" (`#2`, `#3`, `#5`, `#10`)**:
   - When tasks admit exact programmatic synthetic generation, holding out **100% of `train.csv` (`9,500` rows)** as an untouched validation suite (`#2`, `#5`) eliminates overfitting to the tiny `250`-problem Public LB (`1` problem = `0.004`).
   - SFT training loss (`0.0010–0.0023`) was completely uncorrelated with final accuracy (`#5`); top teams validated on **full end-to-end greedy rollout accuracy** plus **"perfect trace rate"** (`#3`) and targeted fragile steps via **low-logprob retraining** (`#10`).

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **RL / GRPO / DPO / ORPO / KTO Instead of Algorithmic Synthetic SFT** | Failed to improve deduction (`#5`); GRPO crawled at **2 tok/s** (`discussion/690161`) | #5, Community | `modeling_nemotron_h.py` KV-cache mismatch (`past_key_values` vs `cache_params`, 20× slowdown); bottleneck was algorithmic search & token budget, not preference alignment |
| **Unsloth 4-Bit Quantization + Sequence Packing on Mamba-2 SSM** | Failed to learn even after `100k` examples (`#5`) | #5 | 4-bit quantization + unmasked cross-example packing corrupts recurrent Mamba-2 SSM state dynamics |
| **Lossy SVD-to-PEFT Adapter Conversion (Tinker) / DoRA** | Dropped bit accuracy `0.89 → 0.81` (`#2`); DoRA rejected by vLLM (`#5`) | #2, #5 | SVD rank truncation degrades memorized exact lookup tables in MoE/Mamba layers; always train & export native `r=32` PEFT |
| **Short-CoT / Compressed Skips & Capacity-Unbalanced Domain Doubling** | Short-CoT (`250k` ex) failed (`#5`); doubling cryptarithm hurt bit CV (`#6`) | #5, #6 | Models cannot skip serial computation steps; `r=32` LoRA has finite memorization capacity causing cross-domain interference |
| **Focal Loss / Token Reweighting / Curriculum Batching / Dropping Failed Traces** | No gain or hurt CV (`#2`, `#10`) | #2, #10 | Standard unweighted CE (or targeted dynamic dead-end masking in DFS `#7`) is superior; dropping `rule_unknown` harms calibration |
