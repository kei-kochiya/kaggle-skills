# MAP — Charting Student Math Misunderstandings — Top Gold Solutions & Post-Mortem

> **Competition**: [MAP - Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings) (Jul 10 – Oct 15, 2025 | 1,857 Teams | 39,742 Submissions | \$55,000 Prize Pool)  
> **Task**: Predict top-3 `Category:Misconception` labels (`65` total classes → `37` after deterministic `True_/False_` correctness split) from a student's math multiple-choice answer (`MC_Answer`) and open-ended `StudentExplanation`  
> **Evaluation Metric**: **`MAP@3`** (1 ground-truth label per row → AP = `1.0`, `0.5`, or `1/3` for rank 1/2/3)  
> **Dataset**: `train.csv` (`36,696` rows, `35,960` deduplicated across **15 fixed `QuestionId`s**, `7.94 MB`) | Hidden Test: `~16,000` rows from the **exact same 15 questions** (`~50%` Pub / `~50%` Priv; `<= 9h` 2×T4 notebook)

---

## Official Top Gold Leaderboard Standings & Solution Index

Winning hinged on **(1) exploiting exact per-question candidate sets (`4–6` valid labels per `QuestionId`) and including all 4 MC `Choices` in the prompt**, **(2) denoising ambiguous `Neither` annotations via teacher soft-label distillation + multi-seed / full-data scaling up to 32B–72B**, and **(3) ultra-fast 2×T4 inference** (FlexAttention + W8A8 SmoothQuant + 2-layer GPU streaming, vLLM `task="embed"` + external linear head, or confidence-gated 72B cascading):

| Private Rank | Public Rank | Team | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | #14 | **`tascj`** *(1 sub!)* | 0.95204 | **0.94919** *(+13)* | **Prefix-Shared Suffix Classification + `FlexAttention`** (`prefix ++ suf_0..suf_K`); full-FT `Qwen3-32B` + `GLM-Z1-32B` via `offload_adam` (3 seeds, trusted **CE loss** over noisy MAP@3); **W8A8 `SmoothQuant(0.75)` + 2-layer GPU streaming** (65m on 2×T4!) | [Writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/1st-place-solution) · [GitHub](https://github.com/tascj/kaggle-map-charting-student-math-misunderstandings) · [NB](https://www.kaggle.com/code/tascj0/map-submit/notebook) |
| 🥈 **#2** | #2 | **`looking for remote job`** | 0.95373 | **0.94890** | **80k contrastive 4-way synthetic explanations** soft-labeled by 4 teachers (`Phi-4`, `Qwen3-32B`, `Mistral-12B`, `Qwen2.5-72B`) + leak-free OOF train soft labels; equal blend of four **37-cls** (DeepSeek expanded defs, `QwQ-32B` **`0.948` Priv solo!**) + four **65-cls** models | [Writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/map2025-private-and-public-2nd) |
| 🥉 **#3** | **#1** | **`monsaraida & Masaya`** | **0.95443** | **0.94866** | `Qwen3-14B` Multi-Task `SeqCls` + **R-Drop (`>+0.001`)** + AWP/EMA + `fp16`/no-pad (`4×` speedup) blended w/ `CausalLM` **per-question `A..E` options (`+0.002–0.003`, `3×` faster)** + MC `Choices` (`+0.003–0.004`) + **2-stage confidence routing** (`<0.95` → `Qwen2.5-72B` EMA `0.948` Priv) | [Writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/3rd-place-solution) |
| 🏅 **#4** | #13 | **`THLUO`** | 0.95211 | **0.94835** *(+9)* | **Best single model (`0.94873` Priv solo!)**: **`Qwen3-Reranker-8B`** grouped CE over 4–6 per-question candidates + **`Qwen2.5-Math-72B` distill** + **6 few-shot student explanations/candidate** (`+0.006` Priv trajectory), blended w/ `Qwen3-32B-AWQ` | [Writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/4th-place-solution-single-model-pb-0-948-lb-0-951) |
| 🏅 **#5** | #24 | **`Naoism`** | 0.95133 | **0.94830** *(+19)* | `StratifiedGroupKFold` by normalized explanation; 6-model multi-family blend (`Gemma2-9B`, `Qwen2.5-14B`, `Qwen3-8B`, `DeepSeekMath-7B`); **10% `False→True` answer-rewrite aug** + per-`QuestionId` mask | [Writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/5th-place-solution) |
| 🏅 **#6** | #22 | **`Manan Jhaveri`** | 0.95145 | **0.94828** *(+16)* | 37-cls `SeqCls` 7-model Qwen-semble (`Qwen3-Embedding-8B` **`0.949` LB solo**, `Qwen3-14B`, `Qwen2.5-14B`); **pseudo-labeled conflicting duplicates** + rule-based typo/number aug + rare-class GPT-4o synth | [Writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/mananjhaveri-map-6th-place-solution) · [GitHub](https://github.com/mananjhaveri/MAP-Charting-Student-Math-Misunderstandings-6th-place-solution/tree/main) |
| 🏅 **#7** | #10 | **`S T I T Y`** | 0.95247 | **0.94820** *(+3)* | 50k–200k 3-shot synthetic distill (`100%` OOF logits on synth, `50/50` on train); `Gemma2-9B` (`LS=0.1, T=1.5`) + `Qwen2.5-32B` + `DeepSeek-R1-32B` + `Qwen3-8B` (**existing vocab tokens beat new special tokens `+0.005` CV/LB!**) | [Writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/private-7th-public-10th-place-solution) |
| 🏅 **#8** | #18 | **`Takashi Someya`** | 0.95173 | **0.94818** *(+10)* | **50-model (`10×5f`)** single-token Unicode symbol (`■,□,▲`) `CausalLM` (`Qwen2.5/3` 7B–14B) + OOF-error-targeted `gpt-5/o3` synth | [Writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/8th-place-solution) |
| 🏅 **#9** | #28 | **`kanto-kaggler`** | 0.95109 | **0.94801** *(+19)* | Fixed `Q31778` + `Wrong_Fraction` casing; **10-model 32B vLLM** (`5f Qwen2.5-32B` + `5f DeepSeek-R1-32B`) + self-distillation (`+0.001` Priv) | [Writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/9th-place-solution) · [NB](https://kaggle.com/code/neilus/map-9th-place-solution) |
| 🏅 **#10** | #17 | **`Liwen Zheng`** | 0.95190 | **0.94801** *(+7)* | **15-model `SeqCls`** (`5f × Qwen3-Reranker-8B / Emb-8B / Qwen2.5-32B`), Focal+LS CE w/ class-weight warmup; **GPTQ-4bit `vLLM task="embed"` (`LAST`) + external FP16 `nn.Linear` head** | [Writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/10th-place-solution) · [NB](https://kaggle.com/code/zhengliwen/map-final-clean) |

---

## 1. Universal LLM Classification, Reranking & Serving Breakthroughs

1. **Candidate-Conditioned Formulation & Context Completeness (`#1`, `#2`, `#3`, `#4`, `#7`)**:
   - Including **all 4 MC `Choices` + `Correct Answer`** in the prompt delivered **`+0.003–0.004` CV/LB** (`#3`), because knowing the distractors makes the student's error immediately legible.
   - Restricting logits/options strictly to each question's **4–6 valid targets** (via per-question `A..E` letter mapping `+0.002–0.003` & `3×` faster `#3`, grouped candidate reranking `#4`, or prefix-shared suffix classification `#1`) beat unconstrained 65-class classification.
   - **Existing Vocab Tokens > New Special Tokens (`#7`, `#8`)**: Mapping labels to existing single-token characters (`A..Z` or Unicode symbols `■,□,▲`) beat adding untrained special tokens by **`+0.005` CV and `+0.005` LB (`#7`)**.
2. **Prefix-Shared Suffix Classification (`FlexAttention`) & Grouped Candidate Reranking (`#1`, `#4`)**:
   - `#1 tascj` packed `prefix ++ suffix_0 ++ ... ++ suffix_K` into a single forward pass with a custom `FlexAttention` mask (`causal & (same_suffix | is_prefix) & same_doc`), computing prefix tokens once while giving every candidate suffix rich semantic label interaction.
   - `#4 THLUO` packed all 4–6 candidate `(Yes/No)` prompts for a single `row_id` into one grouped `CrossEntropy` batch on **`Qwen3-Reranker-8B`** + `Qwen2.5-Math-72B` distillation + 6 few-shot student explanations per candidate → **`0.94873` Private LB solo** (beating every ensemble outside Top 2!).
3. **Soft-Label Denoising & Contrastive Synthetic Distillation (`#2`, `#4`, `#6`, `#7`, `#8`)**:
   - High annotation ambiguity (`Neither` vs `Misconception`) caused severe single-seed variance. Top teams denoised via **OOF teacher soft-label distillation (`100%` soft logits on synthetic data, `50/50` hard+soft on train, `#2`, `#7`)**, **pseudo-labeling conflicting duplicates (`#6`)**, **R-Drop (`>+0.001` CV, `#3`)**, and trusting **multi-seed cross-entropy loss** over noisy `MAP@3` (`#1`).
4. **32B–72B Serving Tricks Inside 2×T4 / 9h (`#1`, `#3`, `#10`, `#18`)**:
   - **W8A8 `SmoothQuant(α=0.75)` + 2-Layer GPU Streaming (`#1`)**: Unlocked `40+ TFLOPS` on T4 (2× FP16) and streamed 32B full-FT weights from `/kaggle/working` (avoiding `/tmp/`'s Copy-on-Write quota bug!) in **65 min** for 16k rows.
   - **Split `SequenceClassification` via `vLLM task="embed"` (`#10`)**: Quantize the merged backbone to GPTQ-4bit, run `vLLM task="embed"` (`pooling_type="LAST", normalize=False`), and multiply by the external FP16 `nn.Linear` head.
   - **Confidence-Gated Cascading / Pyramid Ensemble (`#3`, `#18`)**: Score 100% of test rows with fast 8B–32B models, and route only the uncertain tail (`top1 < 0.95` or bottom `10–50%`) to heavy 72B models (`3×` speedup!).

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Chain-of-Thought (CoT) Generation & RAG Top-N Similar Explanations** | CoT zero gain; RAG (`Qwen3-Emb-8B`) kept CV `0.9493` but **hurt Pub & Priv LB** (`#4`) | #4, #6 | Short explanations don't benefit from verbose CoT; dynamically retrieved neighbors overfit train-fold lexical quirks (static per-candidate few-shots generalized far better) |
| **Naive `StratifiedKFold` Without Deduplication / Trusting Single-Seed `MAP@3`** | Duplicate explanations leaked across folds (`#5`); high seed variance misled tuning (`#1`) | #1, #5 | Identical student text across rows leaks; optimize multi-seed CE loss under noisy labels |
| **2-Stage Hierarchical Pipelines & Merging 5-Fold LoRA Weights** | Hierarchical hurt (`#6`, `#10`); LoRA weight merging scored worse than single fold (`#9`) | #6, #9, #10 | Hard stage-1 cutoffs destroy `MAP@3` top-3 probability recovery; LoRA fold weights lie in misaligned basins |
| **Putting Large Streamed Weights in `/tmp/` or Using `bf16` on T4** | `/tmp/` CoW disk OOM crash (`#1`); `bf16` was `2×` slower (`#3`) & overflowed to NaN in vLLM (`#4`) | #1, #3, #4 | Always stage streamed models in `/kaggle/working` and run `fp16` or W8A8 INT8 on T4 |
| **Per-Question Dynamic Ensemble Weights & Heuristic `Q31778` Reversal** | Overfit CV / Public LB and **hurt Private LB** (`#3`, `#7`) | #3, #7 | Optimizing extra degrees of freedom on noisy splits overfits; simple equal/fixed logit blends generalized cleanly |
