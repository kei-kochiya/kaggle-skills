# Jigsaw — Agile Community Rules Classification — Top Gold Solutions & Post-Mortem

> **Competition**: [Jigsaw - Agile Community Rules Classification](https://www.kaggle.com/competitions/jigsaw-agile-community-rules) (Jul 23 – Oct 23, 2025 | 2,445 Teams | 64,997 Submissions | \$100,000 Prize Pool)  
> **Task**: Few-shot rule-conditioned binary moderation predicting `rule_violation` for Reddit comments (`body`) against specified community rules (`rule`) with 2 positive & 2 negative exemplars per row  
> **Evaluation Metric**: **Rule-Averaged ROC-AUC** (unweighted mean of per-`rule` ROC-AUC across all test rules)  
> **Dataset**: `train.csv` (`2,029` rows covering **only 2 rules**: `No Advertising`, `No legal advice`) | Hidden `test.csv` (`~55,000` rows, `~29%` Public / `~71%` Private) covering **6 rules** (`2` train + `4` unseen rules, each row carrying `positive/negative_example_{1,2}`; `<= 12h` 2×T4 notebook)

---

## Official Top Gold Leaderboard Standings & Solution Index

Because 4 of the 6 test rules never appeared in `train.csv` — yet every test row carried labeled exemplars — **100% of Top-10 solutions performed online Test-Time Training (TTT)** during the 12-hour 2×T4 submission window, separating gold winners by **target-token loss masking, label conflict resolution, safety-pretrained backbones, online Deep Mutual Learning (DML), and multi-family rank blending**:

| Private Rank | Public Rank | Team | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **#1** | **`Guanshuo Xu`** | **0.93476** | **0.92930** | Unsloth QLoRA TTT (`1ep train + 2ep test`, dropped `subreddit`); **masked all wrapper tokens so ONLY `"Yes"/"No"` gets CE loss**; `forward()`-only expanded Yes/No log-odds + **per-rule rank norm (`+0.002`)** across 6 models (`Qwen3-14B/8B/4B`, `Qwen2.5-14B`, `Llama3.1-8B`, `Ettin-400M`) | [Writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/1st-place-solution) · [NB](https://www.kaggle.com/code/wowfattie/notebookc4e4a9206d) |
| 🥈 **#2** | #6 | **`^^`** | 0.93316 | **0.92923** *(+4)* | Clean 1-epoch Unsloth 4-bit SFT (`"Rule: ...\nComment: ..."`, dropped `subreddit`, majority-vote dedup) across 4 diverse LLMs (`Phi-4-14B`, `Qwen3-14B/8B`, `Qwen2.5-7B`) split over 2×T4 (`~11h`), next-token Yes/No logit diff | [Writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/2nd-place-solution) · [NB](https://www.kaggle.com/code/yaoqingjin/4-llms-ensemble) |
| 🥉 **#3** | #5 | **`Sergio Manuel Papadakis`** | 0.93335 | **0.92871** *(+2)* | **Pretrained LoRA → Online TTT → vLLM final-token embeddings → `LGBM + XGB + SVC`** across `Qwen3/2.5` (7B–14B) + Fast TTT stream (`BGE-base` Triplet diff-to-centroid, `DeBERTa-v3-small`, `Qwen3-Emb-0.6B` ArcFace) | [Writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/3rd-place-solution) · [Sub NB](https://www.kaggle.com/code/socom20/t73t74-ens-7b-9b-unsloth-qwen14bx4-qwen8bx1-qwen7) |
| 🏅 **#4** | #3 | **`Dream Team`** (R. Berdichevsky) | 0.93382 | **0.92860** | AWQ `Qwen3-4B/14B` `{"0","1"}` logits + **soft disagreement targets + Tsallis GCE (`q=0.955`)** + `{rule}`-first **prefix KV-cache reuse** + **2-stage most-uncertain (~8k) soft PL** (`0.8/0.2` blend) | [Writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/4th-place-solution) · [NB](https://www.kaggle.com/code/rusalmighty/jigsaw-4th-place-solution) |
| 🏅 **#5** | #2 | **`Flash ⚡`** | 0.93472 | **0.92855** | **7-family ensemble**: DDP `r=256` `Llama3.1-8B` (`0.9262` Priv) & `Qwen3-8B` (`0.9231`), Triplet+FAISS, 6× Bi-Encoders (`0.7 LR + 0.3 KNN`), `DeBERTa` + 8-paraphrase NLI, & **`Qwen2.5-32B` asymmetric RAG** (closest pos + farthest hard neg: **`+0.006`**) | [Writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/5th-place-solution-diverse-ensemble) · [NB](https://www.kaggle.com/code/kishanvavdara/5th-place-solution-diverse-ensemble) |
| 🏅 **#6** | #12 | **`ducnh279`** | 0.93249 | **0.92781** *(+6)* | **Online Deep Mutual Learning (DML)**: joint `CE + KL` TTT across 3 peers (`Qwen3-14B + Qwen3-8B + Qwen3Guard-Gen-4B`), boosting single 14B models **`+0.004–0.006`** (`0.921→0.925` Priv) | [Writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/6th-place-solution) · [NB](https://www.kaggle.com/code/ducnh279/6th-place-solution-deep-mutual-learning) |
| 🏅 **#7** | #9 | **`ktr`** | 0.93281 | **0.92760** *(+2)* | **Moderation-pretrained LLMs beat general chat LLMs**: **`ShieldGemma-9B`** (`0.927 Pub / 0.922 Priv`, single best!) > `Qwen3-14B`; `Qwen3-8B-Guard` beat vanilla `Qwen3-8B` (`+0.003`); patched vLLM FP16 block for Gemma-2 | [Writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/7th-place-solution) · [NB](https://kaggle.com/code/kcotton21/jigsaw-acrc-7th-place-private-0-92760) |
| 🏅 **#8** | #4 | **`Reddit comment moderators`** | 0.93370 | **0.92759** | `SequenceClassification` head QLoRA (`3× Qwen3-14B + Llama2-13B + BGE-base`) + `group_by_length` (`2×` train) + **3-bucket length-sorted dual-GPU inference (`4×` speedup)** | [Writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/8th-place-solution-qwen3-14b3-llama2-13b1-b) · [NB](https://www.kaggle.com/code/williamwu88/giga-ensemble-top-932) |
| 🏅 **#9** | #7 | **`lllleeeo + mango`** | 0.93315 | **0.92747** | `0.70` small Gen TTT (`Qwen3-4B`, `Llama3.2-3B`, `Phi-3.5-Mini`, `Qwen2.5-7B`) + `0.25` Encoders (`DeBERTa/E5/MPNet` w/ LLDR `+0.003`) + `0.05` BGE Triplet+UMAP | [Writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/9th-place-solution) |
| 🏅 **#12** | — | **`#12` (Unlabeled Corpus)** | — | — | Pretrained `DeBERTa-base` on **800k soft LLM pseudo-labels** filtered from historical Reddit corpus (`<0.5%` OOD) → **`0.912 → 0.924` LB (`+0.012`)** in ~1h | [Writeup](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/12th-place-solution) |

---

## 1. Universal Few-Shot LLM Classification & Online TTT Breakthroughs

1. **Data Forensics: Shortcut Dropping & Conflict-Aware Deduplication (`#1`, `#2`, `#4`, `#5`, `#7`, `#9`)**:
   - Dropping `subreddit` (which human annotators never saw) removed spurious shortcut bias and enabled cross-subreddit deduplication, yielding **`+0.007` LB (`#7`)** and faster training (`#9`).
   - Same `(body, rule)` pairs carried conflicting labels across test exemplars. Top teams resolved collisions via **majority vote + dropping ties (`#2`, `#7`)**, or collapsed duplicates into **soft violation ratios** trained with noise-robust **Tsallis Generalized Cross-Entropy (`q = 0.955`, `#4`)**, and down-weighted near-duplicates (`cosine > 0.95`, `#5`).
2. **Target-Token-Only Loss Masking & Single-Forward Log-Odds Scoring (`#1`, `#2`, `#4`)**:
   - Standard chat-template SFT wastes gradient updates on boilerplate tokens (`\n\n`, EOS, assistant wrappers), causing inconsistent convergence across LLM families. Masking all wrapper tokens so **only the single `"Yes"`/`"No"` (or `"1"`/`"0"`) target token receives CE loss** (`#1`) unified training across `Qwen3`, `Qwen2.5`, and `Llama3.1`.
   - At inference, running a **single `forward()` pass** (zero `generate()` loop) and computing log-odds over expanded lexical variants (`["Yes","YES","Y","yes","True"]` vs `["No","NO","N","no","False"]` + space-prefixed tokens, `#1`), combined with **length-sorted batching (`2×` train / `4×` infer speedup, `#1`, `#6`, `#8`)** and **`{rule}`-first prefix KV-caching (`#4`)**, fit 4–7 large models inside 12h on 2×T4.
3. **Safety/Domain-Pretrained Backbones & Online Deep Mutual Learning (`#6`, `#7`)**:
   - Domain-matched safety LLMs punched far above their parameter count: **`ShieldGemma-9B`** beat `Qwen3-14B` (`0.922` vs `0.919` Priv), and **`Qwen3-8B-Guard`** beat vanilla `Qwen3-8B` (`+0.003`, `#7`).
   - **Online Deep Mutual Learning (`DML`, `#6`)**: Jointly training 2–3 peer LLMs on test exemplars with `CE + KL` divergence overcame sharp single-teacher distillation and boosted individual 14B models by **`+0.004–0.006`** (`0.921 → 0.925` Priv).
4. **Metric-Aligned Ensembling: Per-Rule Rank Normalization & Cross-Paradigm Diversity (`#1`, `#3`, `#5`)**:
   - Because the metric averages ROC-AUC **per rule**, different models exhibit rule-specific logit calibration shifts. **Per-rule rank normalization to `[0, 1]`** immediately added **`+0.002` LB (`#1`)**.
   - Stacking **LLM final-token vLLM embeddings (`#3`)** + **Triplet/ArcFace bi-encoders (`BGE`, `E5`, `GTE`, `Qwen3-Emb`, `#3`, `#5`, `#11`)** + **NLI zero-shot encoders (`#5`)** + **asymmetric RAG prompting** (closest positive + *farthest* hard negative: **`+0.006`**, `#5`) provided genuine error decorrelation on top of generative QLoRA LLMs.

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Including `subreddit` or Hand-Crafted Rule-Specific Prompts** | Lost `-0.002` to `-0.007` LB | #1–#5, #7, #9 | Annotators ignored `subreddit`; rule-specific prompts broke cross-rule transfer and prevented deduplication |
| **Standard Confident Pseudo-Labeling & LLM-Generated Synthetic Comments** | Hurt LB (`#7`, `#8`, `#9`, `#17`) | #7, #8, #9, #17 | Synthetic text drifts from authentic Reddit style; confident PL reinforces confirmation bias (only `#4`'s *most-uncertain* soft PL & `#12`'s real historical Reddit corpus helped) |
| **Training Only on `train.csv` (Even With `Qwen3-32B`)** | Badly lagged online TTT | #1–#9 | 4 of 6 test rules are unseen in `train.csv`; online TTT on test exemplars is non-negotiable |
| **`bf16` on T4 GPUs & vLLM Quantized `SequenceClassification` w/ `TP=2`** | `bf16` was **6× slower** (`#12`); vLLM `TP=2` crashed | #12, Community | T4 (`sm_75`) lacks native BF16 tensor cores (always use `fp16` on T4) |
| **Raw Logit Averaging / Label Smoothing / Layer Freezing / `Muon` on QLoRA** | Underperformed per-rule rank blending & AdamW (`#9`, `#17`) | #9, #17 | Cross-family logit scales diverge across unseen rules |
