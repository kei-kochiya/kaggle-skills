# WSDM Cup — Multilingual Chatbot Arena — Top Gold Solutions & Post-Mortem

> **Competition**: [WSDM Cup - Multilingual Chatbot Arena](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena) (Nov 18, 2024 – Feb 3, 2025 train / Mar 10, 2025 live forecast | 889 Teams post-DQ [950 raw] | 1,550 Submissions | \$50,000 Prize Pool)  
> **Task**: Predict human preference (`winner ∈ {model_a, model_b}`, binary — no tie class) across head-to-head Chatbot Arena battles spanning 100+ languages  
> **Evaluation Metric**: **Categorization Accuracy** (pivoting from LogLoss in the earlier LMSYS competition)  
> **Dataset**: `train.parquet` (`~48k` battles, `113.98 MB`, w/ train-only `model_a`, `model_b`, `language`) | Hidden Live Forecasting Test: `~10k` train-phase (`<= 4.75h` 2×T4) → **`<= 25k` newly collected post-deadline battles (`<= 12h` 2×T4)**

---

## Official Top Gold Leaderboard Standings & Solution Index

In a live post-deadline forecasting evaluation where **54.5% of Gold (6/11)** survived Public→Private shakeup, winning recipes unified **70B–72B privileged-teacher / multi-stage pseudo-label distillation into 8B–14B students**, **same-step `(A,B)/(B,A)` symmetry + proportional middle truncation**, and **uncertainty-gated swap TTA / 3-tier cascades** running via either `tascj0` sequence unpadding or `vLLM` `A`/`B` logit scoring:

| Private Rank | Team | Private Acc | Core Winning Paradigm | Writeup / Code |
| :---: | :--- | :---: | :--- | :--- |
| 🥇 **#1** | **`whitefebruary`** | **0.716092** | **Privileged-info 5-fold `Qwen2.5-72B` teachers** (prepended train-only `model_a`/`model_b` IDs!) soft-labeling 236k aux/tie/50k-synth battles → distilled into full-FT `Qwen2.5-14B` (`CE+KL+Cosine`, same-step `(A,B)+(B,A)` symmetry, weight merge) + **`tascj0` unpadding** + **33% uncertain swap TTA** | [Writeup](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/whitefebruary-1st-place-solution) · [GitHub](https://github.com/maxreciprocate/kaggle-lmarena-1st-place) · [NB](https://www.kaggle.com/code/orangeaugust/lmsys2?scriptVersionId=220505454) |
| 🥈 **#2** | **`zhudong1949`** | **0.716047** | Initialized from `tascj0` LMSYS `gemma2-9b` & `ArmoRM-Llama3-8B` → 2-stage PL over **95k non-EN-heavy (`70–95%`) battles** (filtered `\|p_A-p_B\|≤0.03`) → **2-epoch EN→Multilingual curriculum**; `gemma(PAB)` + **`llama(PBA)` gated on `\|p_A-p_B\|<0.8` (`2.5:1`)** | [Writeup](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/zhudong1949-the-2nd-place-solution) · [GitHub](https://github.com/zhudongwork/wsdm-cub-2nd-place-solution.git) |
| 🥉 **#3** | **`PLaMo 1000000B`** | **0.714967** | **`CausalLM` `'A'`/`'B'` logits in vLLM (`70k–74k tok/min` w/ prefix caching)**; multi-teacher (`Qwen2.5-72B + Athene-V2`, **`T=5, w=0.9`** KL jumped CV `0.7051 → 0.7169`!) → `Qwen2.5-14B` + `Phi4` (`AutoRound seqlen=6144, iter=3000`) + **length-gated ensemble weights** (`Phi4` `0.5→0.1`) | [Writeup](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/plamo-1000000b-3rd-place-solution) · [GitHub](https://github.com/LIHANG-HONG/WSDM-Cup-3rd-place-solution) |
| 🏅 **#4** | **`HKUST-GZ DSA KiMi Lab`** | **0.714067** | **Stage-wise loss**: Stage-1 full-vocab CE w/ `(A,B)+(B,A)` in same batch → Stage-2 2-class `[A,B]` CE (`2ep`, `+0.002` CV across all 5 folds) distilled from 3×70B teachers; **vLLM `allowed_token_ids=[a,b]`** + **GPTQ 8-bit (`0.714` Priv) crushed AWQ 4-bit (`0.709` Priv)** | [Writeup](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/hkust-gz-dsa-kimi-lab-4th-place-solution) · [NB](https://www.kaggle.com/code/daihengwei/wsdm-vllm-gen-inference-pb714) |
| 🏅 **#5** | **`Just doing`** | **0.712627** | **Top zero-70B-teacher solution**: 2-stage `Qwen2.5-14B` self-distillation (`r=64, α=128`, `mid_cut + swap + soft_label` `0.6907 → 0.7068` Pub → `0.7126` Priv) + **number-count tie-breaker** when logit gap `<0.05` | [Writeup](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/just-doing-5th-place-detailed-solution) · [GitHub](https://github.com/lcy80366872/kaggle-project/tree/main/wsdm) |
| 🏅 **#6** | **`Oh Your Mercy Never Fails`** | **0.712582** | Pseudo-labeled **560k** battles (`>0.693` LB with zero comp data!); distilled `Qwen2.5-14B` (100% rows) + `Gemma2-9B` (uncertain rows only) | [Writeup](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/oh-your-mercy-never-fails-6th-place-solution) |
| 🏅 **#7** | **`Team Turing`** (R. Biswas, `trushk`) | **0.712222** | **3-tier uncertainty-gated vLLM cascade**: 100% `Gemma2-9B` (`max_len=4096`) → top **35%** uncertain `Qwen2.5-14B` → top **15%** uncertain `Qwen2.5-32B` / `Mistral-Small-24B` (GPTQ) | [Discussion](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567589) · [NB](https://www.kaggle.com/code/conjuring92/wsdm-7th-place-solution) |
| 🏅 **#12** | **`monsaraida`** | **0.710872** | `Qwen2.5-14B` QLoRA w/ **`snip <N> chars`** truncation marker, **aux model-ID head (`w=0.1`)**, alternating length-sorted 2×T4 inference + **40% uncertain swap TTA** | [Writeup](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/monsaraida-12th-place-solution) |

---

## 1. Universal Pairwise Reward Modeling, Distillation & Serving Breakthroughs

1. **Privileged-Information Teacher Distillation & Multi-Teacher Temperature Scaling (`#1`, `#3`, `#4`, `#12`)**:
   - **Privileged Train-Only Metadata (`#1`)**: Prepending train-only `model_a` / `model_b` identities into 5-fold `Qwen2.5-72B` teachers let teachers absorb arena Elo priors and transfer sharper soft targets over **236k auxiliary/tie/first-turn/synthetic battles** to deployment students that never see model names (also mirrored by `#12`'s auxiliary model-ID head `w=0.1`).
   - **Temperature-Scaled KL Distillation (`#3` Controlled Ablations)**: On `Qwen2.5-14B` Fold 0, baseline `0.7051` → `T=1, w=0.5` (`0.7062`) → `T=5, w=0.5` (`0.7103`) → `T=5, w=0.9` (`0.7135`) → **multi-teacher (`Qwen2.5-72B + Athene-V2`, `T=5, w=0.9`) `0.7169` (`+0.0118` CV!)**. Even without 70B models, 2-stage 14B self-distillation reached **#5 (`0.7126` Priv)**.
   - **Filtering Ambiguous Pseudo-Labels (`#2`)**: Dropping near-tie pseudo-labels (`|p_A - p_B| <= 0.03`) and excluding minimal-gap frontier model pairs (`nbroad` v3) prevented noisy distillation labels.
2. **Pairwise Order Symmetry, Middle Truncation & Curriculum (`#1`, `#2`, `#4`, `#5`, `#12`)**:
   - **Same-Step `(A,B) + (B,A)` Batching (`#1`, `#4`)**: Putting both orders in the same gradient step directly cancels position bias.
   - **Proportional Middle Truncation (`#1`, `#2`, `#5`, `#12`)**: Preserving prompt + head + tail of both responses beat left-truncation (`0.6938` vs `0.6907` LB, `#5`); inserting explicit **`snip <N> chars`** (`#12`) preserves length-bias cues without blowing `max_len`.
   - **Multilingual Surprise**: Fine-tuned LLMs achieved nearly uniform accuracy across languages without translation (`discussion/551674`), though **non-English-heavy synthetic PL (`50–95%`) + 2-epoch EN→Multilingual curriculum (`#2`)** pushed generalization cleanly on the live forecast split.
3. **vLLM Serving, Quantization & Uncertainty-Gated Cascades (`#1`–`#7`, `#12`)**:
   - **`allowed_token_ids=[a_id, b_id]` (`#4`)**: Mandatory in vLLM `SamplingParams(max_tokens=1, logprobs=10)` so `'A'`/`'B'` never drop out of top-K logprobs.
   - **Prefix Caching for Swap TTA (`#3`)**: Because `(A,B)` and `(B,A)` share the exact prompt prefix, vLLM prefix caching lifted throughput from `70k → 74k tok/min` on 2×T4.
   - **Uncertainty-Gated TTA & Multi-Tier Cascades (`#1`, `#2`, `#3`, `#6`, `#7`, `#12`)**: Spending Pass-2 TTA or heavier 14B–32B models exclusively on the uncertain band (`|p - 0.5|` smallest `15–40%`, or `#3`'s **length-gated ensemble weights** matching each backbone's context-length strength) maximized 12h T4 ROI.
   - **Quantization Precision (`#3`, `#4`)**: **GPTQ 8-bit (`0.713 CV / 0.714 Priv`)** decisively beat **AWQ 4-bit (`0.710 CV / 0.709 Priv`)** (`#4`); heavy **`AutoRound(seqlen=6144, iters=3000)`** preserved FP16 accuracy (`#3`).

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Chain-of-Thought (CoT) Critiques / `DeepSeek-R1-Distill` for Single-Token Scoring** | GPT-4o critiques worsened soft labels (`#1`); `R1-Distill-14B` dropped **`-0.003` LB** (`#5`); CoT timed out (`#3`) | #1, #3, #5, #12 | Single-token A/B classification bypasses CoT reasoning; generating critiques explodes 2×T4 latency |
| **Training `bf16 + FA2` (A100) → Inferring `fp16 + sdpa` (T4)** | Silent **`-0.001` to `-0.002`** accuracy drop (`0.7011 → 0.699`) | `hengck23`, `lhwcv` | Attention/rounding precision mismatch between Ampere BF16/FA2 and Turing FP16/SDPA |
| **AWQ 4-Bit vs GPTQ 8-Bit, Layer Pruning, & Naive LoRA Merging** | AWQ 4-bit lost `-0.005` Priv (`#4`); layer pruning & LoRA merging degraded accuracy (`#1`, `#6`) | #1, #4, #6 | Pairwise preference margins are razor-thin (`<0.05` logits); aggressive 4-bit/pruning flips close calls |
| **BT Loss, DPO Loss (`0.7 DPO + 0.3 BCE`), Random Few-Shot, & `max_len=8192` Train** | Underperformed standard CE/KL distillation (`#2`, `#5`) | #2, #5 | Direct binary CE + soft KL aligns best with Accuracy; overly long training context dilutes signal |
| **Translating Prompts or Adding Auxiliary Language-ID Head** | Distorted nuance & zero CV gain (`#12`, `discussion/551674`) | #6, #12, Community | Modern 9B–14B LLMs already encode multilingual semantics uniformly once fine-tuned |
