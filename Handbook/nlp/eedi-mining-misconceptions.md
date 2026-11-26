# Eedi — Mining Misconceptions in Mathematics 🏆

> **Track:** NLP & LLM Retrieval / Reranking (Featured Code) · **Scale:** 1,446 teams · \$55,000 · Closed Dec 13, 2024  
> **Core Lesson:** In open-catalog LLM retrieval + reranking with unseen target labels under 2×T4 / 9h limits, **optimizing Stage-1 retrievers strictly for `Recall@K` (not `MAP@K`)**, **clustered synthetic generation + CoT wrong-reasoning distillation**, a **progressive `14B → 32B → 72B` pointwise-to-listwise funnel** (using `vLLM` prefix caching + `cpu_offload_gb`), **order-bias permutation TTA**, and **unseen-class-aware `GroupKFold(Construct/Subject)` CV** beat Public-LB-tuned unseen-prior multipliers by **+0.034 Private LB**.

---

## 1. Competition DNA & Leaderboard

- **Task:** Given math multiple-choice questions (`~1,869` train MCQs, `~685` hidden test MCQs, each with 3 incorrect distractors; `NaN` unannotated distractors ignored), rank the top **25 `MisconceptionId`s** from a catalog of **`2,587` misconceptions** (`>900` never appear in `train.csv`; LB probing showed an ~`1:3` seen:unseen misconception ratio in test!).
- **Metric (`MAP@25`):** Since each annotated distractor has at most 1 ground-truth misconception, `MAP@25` equals **Mean Reciprocal Rank truncated at 25** ($\frac{1}{\text{rank}}$ if $\text{rank} \le 25$, else $0$).
- **Constraints:** **2× NVIDIA T4 (30 GB total VRAM), 9-hour offline runtime**.

| Private # | Public # | Shakeup | Team | Private LB | Public LB | Core Breakthroughs |
| :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| 🥇 **1st** | 6 | **+5** | **MTH 101** (`conjuring92`) | **0.63881** | 0.64379 | **+0.03401 blowout!** 10.6k clustered Claude 3.5 synthetic MCQs + distilled CoT (`+0.019`) + 72B PL (`+0.044`) + `14B→32B→72B` funnel (zero LB-tuned PP) |
| 🥈 **2nd** | 8 | **+6** | **CQYR** | **0.60480** | 0.62947 | Offline 72B misconception expansion (`+2–4%`) + `QwQ-32B` retriever (`0.500` Priv) + `AutoRound` 4-bit `Qwen72B+Llama70B` sliding window + reverse TTA |
| 🥉 **3rd** | 1 | -2 | **Waseda Pochi** (`threerabbits`) | **0.60248** | **0.67081** | Random candidate-order shuffle TTA + test-probed `1:3` unseen-prior multiplier (`+0.036` Priv / `+0.068` Pub) |
| 🏅 **4th** | 3 | -1 | **takoi & charmq & kami** | **0.60044** | 0.65568 | `Qwen14B+32B` concat retriever (25 all + 15 unseen) + 100-shot 72B synth + train-seen-only negs (`+0.025`) + parameter-averaged LoRA rerankers |
| 🏅 **5th** | 4 | -1 | **Ebi & ktr** (`kcotton21`) | **0.59765** | 0.65374 | **52-way single-token (`A–Z, a–z`) listwise `Qwen2.5-32B`** over top-104 (`52×2`) + `GroupKFold(SubjectId)` + CoT KD (`+0.04` retriever CV) |
| 🏅 **6th** | 5 | -1 | **Nikita Babich** | **0.58998** | 0.65040 | 50-way listwise + excluding unseen misconceptions from train (`+0.02`) + KL self-distillation + top-2 permutation TTA |
| 🏅 **7th** | 2 | **-5** | **terekaerumasahmet** | **0.58693** | 0.66270 | 3-nature `Qwen32B` blend (binary + 40-choice + 9-choice shuffled TTA) + RRF voting; heavy unseen boosting shaked down `-5` |
| 🏅 **8th** | 13 | **+5** | **Yannan Chen** | **0.58160** | 0.61350 | 4-fold `Qwen14B` + single `Qwen32B` pointwise `"Yes"` with **`WiSE-FT` LoRA merge (`alpha=0.8`, `+0.02` LB)** |

---

## 2. Validation Strategy (Avoiding the Seen-Class Trap)

- **Why `GroupKFold(QuestionId)` Lies:** Leaves `~80%` seen and `≤ 20%` unseen misconceptions in validation (`#6`), whereas test is `~75%` unseen — rewarding overfitting to train misconception IDs (`0.65*0.8 + 0.30*0.2 = 0.58` CV while LB collapses).
- **Reliable CV Designs:**
  - **`GroupKFold(ConstructId)` (`#1`):** Balanced between optimistic `QuestionId` and pessimistic `SubjectId`; synthetic data placed strictly in fold 99 (train-only).
  - **`GroupKFold(SubjectId)` (`#5`):** Directly mirrors unseen-subject test generalization (`0.626 CV` vs `0.633 LB`).
  - **Dual Slice Monitoring (`#4`, `#6`):** Always track **unseen-misconception-only CV** alongside overall CV (or `50/50` seen/unseen weighted validation).

---

## 3. Stage 1 — LLM Bi-Encoder Retrieval & Synthetic Data

- **Dominant Backbones:** Fine-tuned `Qwen2.5-14B/32B` (last-token pooling > mean pooling), `QwQ-32B-Preview` (`0.500` Priv solo in `#2`!), `SFR-Embedding-2_R`, `Linq-Embed-Mistral`, `bge-en-icl`, `e5-mistral-7b`.
- **Contrastive Training Recipes (`MultipleNegativesRankingLoss`):**
  1. **Optimize `Recall@K`, NOT `MAP@25` (`#1`):** Standard `Qwen2.5-14B` (`temp=0.01`, LoRA `r=64`, `bs=128`, **strictly 1 demo per misconception per batch** to eliminate false in-batch negatives) hit **`Recall@32 = 0.937`**. Distilling reranker scores into the retriever boosted `MAP@25` (`0.486 → 0.528`) but **dropped `Recall@32` (`0.937 → 0.926`)**, hurting final funnel score — so `#1` kept the higher-recall checkpoint!
  2. **Dynamic Candidate Cutoff (`#1`, `#4`):** `#1` took top-32 plus up to 32 more within `top1_sim - 0.06` (`32–64` cands); `#4` pooled `25` global + `15` explicitly from unseen misconceptions (`Recall@25 = 0.926`).
- **Solving Unseen Misconceptions via Clustered Synthetic Data & Offline Expansion:**
  - **Clustered Contrastive Generation (`#1`):** Grouped neighbor misconceptions from validation co-occurrence and prompted `Claude 3.5 Sonnet` to generate **10.6k MCQs** (5/cluster alongside 4–5 reference MCQs so the generator captured fine distinctions), judged with `GPT-4o` and deduped (`gte-base` cosine `>=0.995` merge, `0.95–0.995` drop) → `4,791` misconceptions.
  - **Zero-Inference-Cost Misconception Expansion (`#2`):** Pre-generating rich LLM explanations of all 2,587 catalog misconception texts offline (`Qwen2.5-72B` / `Llama-3.1-70B`) lifted retriever scores **`+2–4%`** at zero test-time latency.
  - **Distilled Wrong-Reasoning CoT (`#1`, `#5`):** Fine-tuning `7B/14B/32B` reasoners to explain *why the student chose the distractor* (without seeing candidate misconceptions) boosted `#5`'s bi-encoder by **`+0.04 CV`** and `#1`'s 14B reranker by **`+0.019 Private LB`** (`0.596 → 0.615`).

---

## 4. Stage 2 — LLM Reranking (Pointwise, Listwise & Order Bias)

1. **Progressive Pointwise → Listwise Funnel (`#1` — Cleanest Ablation Chain):**
   - **Stage 1 (`Qwen2.5-14B` Pointwise, `32–64 → top 8`):** `logits_yes - logits_no` with grouped CE (`1 pos + N negs`, frozen `lm_head`). Private LB progression: Baseline `0.495` → +Few-Shot (0–2 train demos) **`0.531 (+0.036)`** → +72B Teacher Pseudo-Label Distillation on synthetic data **`0.575 (+0.044)`** → +Neg Ratio `16→24` & 2× synthetic **`0.596 (+0.021)`** → +CoT (50% dropout in train) **`0.615 (+0.019)`**.
   - **Stage 2 (`Qwen2.5-32B` Pointwise, `top 8 → top 5`):** **`0.625 (+0.010)`** Private LB.
   - **Stage 3 (`Qwen2.5-72B` Listwise, `top 5 → final 1–5`):** Jointly sees all top-5 candidates + 3 CoTs (`7B/14B/32B`) + 1 reference example per candidate → **`0.63881 (+0.013)`** Private LB. Final list concatenates `[1–5 from 72B, 6–8 from 32B, 9–25 from 14B]`.
2. **Single-Token Multi-Choice Listwise Reranking (`#2`, `#5`, `#6`, `#7`):**
   - **`52`-Way Single-Token Reranker (`#5`):** Mapped 52 candidates to single-token `A–Z, a–z` + `"NA"`, scoring top-104 in two `52`-candidate passes (`0.572 → 0.581` Priv single fold; 3-fold `0.597` Priv).
   - **Sliding-Window `72B + 70B` (`#2`):** `Qwen14B` reranked positions `8–17`, then `Qwen2.5-72B + Llama-3.3-70B` reranked `1–10` using `MultipleChoiceLogitsProcessor` (`+100` bias on valid choice tokens + logprobs).
   - **Train-Seen-Only Negatives (`#4`, `#6`):** Restricting training negatives to misconceptions that *also* appear as positives in train prevented the model from learning a shortcut to penalize unseen catalog IDs (**`+0.025` LB** in `#4`; **`+0.020`** in `#6`).
3. **Order-Bias Mitigation:** Listwise LLMs strongly bias toward early candidate positions. Fixed via **reverse-order TTA** (`#2`), **random shuffle TTA** (`#3`: `+0.012` Pub / `+0.002` Priv), **exhaustive top-2 permutation TTA** (`#6`), and **order-shuffled 9-choice training** (`#7`).

---

## 5. Serving 72B on 2×T4, Ensembling & Anti-Patterns

- **Fitting 72B/70B Models Inside 2×T4 (30 GB VRAM) / 9h:**
  - `vllm` with `enable_prefix_caching=True` (shared question+CoT prefix across 32–64 pointwise candidates), `tensor_parallel_size=2`, `gpu_memory_utilization=0.98`, `enforce_eager=True`, **`cpu_offload_gb=8`** (16 GB CPU RAM offload), `max_tokens=1`, and `MultipleChoiceLogitsProcessor`.
  - **Quantization:** Task-calibrated **AutoAWQ** (`#1` — generic calibration hurt top-1!), **GPTQ Int4** (`#4`, `#5`), or **`intel/auto-round` 4-bit** (`#2`, padding Qwen-72B `intermediate_size 29568→29696` for TP=2).
- **Ensembling:** Funnel cascade blending (`#1`, `#2`), **parameter-space LoRA averaging / `WiSE-FT` (`alpha=0.8`, `+0.02` LB in `#8`)**, and Reciprocal Rank voting (`1/i`, `#7`).
- **Verified Anti-Patterns:**
  1. **Over-Tuning Unseen-Class Multipliers on Small Public LB:** Public LB had only ~192 questions (28% of 685). Aggressive unseen-boosting (`#3`, `#7`) inflated Public LB (`0.671` / `0.663`) but suffered sharp Private drops (`0.602` / `0.587`), whereas `#1` & `#2` jumped `+5` and `+6` ranks by trusting CV and synthetic generalization.
  2. **Iterative Hard-Negative Mining:** Consistently hurt generalization (`#1`, `#5`, `#7`-neighbor) — likely because top-ranked "hard negatives" often contained false negatives / near-duplicate misconceptions.
  3. **`QwQ-32B-Preview` as Reranker:** Failed across `#1`, `#5`, `#7` (reasoning/CoT models hurt calibrated single-token reranking, even though `#2` got `0.500` using it as a *bi-encoder*!).
  4. **Multi-Character Choice Labels & AutoAWQ on Small `lm_head`:** Non-single-token labels (`#5`) and quantizing `lm_head` on small models (`#8`) degraded logprobs.

---

## 6. Verified Solution Sources

- **1st Place (`MTH 101`):** `https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution`
- **2nd Place (`CQYR`):** `https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/cqyr-2nd-place-solution`
- **3rd Place (`Waseda Pochi`):** `https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/waseda-pochi-3rd-place-solution-with-magic-boost`
- **4th Place (`takoi & charmq & kami`):** `https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/takoi-charmq-kami-4th-place-solution`
- **5th Place (`Ebi & ktr`):** `https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/ebi-ktr-5th-place-solution`
- **6th Place (`Nikita Babich`):** `https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/nikita-babich-6th-place-solution`
- **7th Place (`terekaerumasahmet`):** `https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/terekaerumasahmet-private-7th-public-2nd-place-sol`
- **8th Place (`Yannan Chen`):** `https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/yannan-chen-8th-place-solution`
