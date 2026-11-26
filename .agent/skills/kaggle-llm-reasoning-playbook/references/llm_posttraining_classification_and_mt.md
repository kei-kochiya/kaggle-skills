# LLM Post-Training, Classification, Reward Modeling, Extraction & MT — Universal Recipes

Transferable NLP/LLM patterns distilled from gold-medal solutions across inductive reasoning LoRA post-training, ultra-low-resource machine translation, online test-time classification, candidate reranking, long-document scientific extraction, and multilingual pairwise reward modeling.

---

## 1. Algorithmic Reasoning Post-Training: "Trainable-Trace" SFT

When an open LLM must solve structured algorithmic/inductive reasoning under a fixed token budget (and RL/GRPO/DPO stalls):

1. **The 5 Mechanical Rules of Trainable Synthetic CoT Traces**:
   - Enforce automatically via AST / reference-integrity tests before SFT:
     1. **No hidden compute**: Every emitted step is locally derivable from visible context + a tiny fixed rule set.
     2. **Locality**: Perform each operation within `~10–20` lines of its operands.
     3. **Causal consistency**: Branching depends strictly on prior emitted tokens (never solve externally first, then decorate).
     4. **Reference integrity**: Every quoted value appears verbatim earlier in the trace.
     5. **Serial execution + independent verification**: Execute bit/digit/char steps serially and include a round-trip consistency check.
2. **Decouple Weight-Memorized Lookups From Sequential Execution (2-Stage SFT)**:
   - Precompute canonical lookup catalogs, train **Stage 1 (catalog memorization)**, then **Stage 2 (sequential DFS/CSP backtracking + memory drills)** — consistently outperforms joint single-stage SFT.
3. **Tokenizer Compaction & Hybrid Mamba/MoE LoRA Hygiene**:
   - Compact verbose digit/bit tables (e.g., HEX encoding cuts `25–28%` tokens; space-prefix hex `A–F` to stop BPE merges) and **always include `lm_head` (`output_layer`) in LoRA targets** when introducing new trace delimiters.
   - Export **exact native BF16 PEFT** (avoid lossy SVD-to-PEFT conversion); **never combine 4-bit quantization + sequence packing on Mamba-2 SSM layers** (collapses recurrent state); verify `past_key_values` vs `cache_params` on hybrid architectures before running GRPO.
   - Ignore uninformative SFT training loss (`0.001–0.002`) — validate strictly via **end-to-end greedy rollout accuracy + "perfect trace rate"** on a 100% held-out validation split; stabilize noisy adapters via **Iterative LoRA Model Soup** when branches share initialization.

---

## 2. Single-Token LLM Classification, Online TTT & Candidate Reranking

For discriminative classification, moderation, or multi-choice reranking (`ROC-AUC`, `MAP@k`, `Accuracy`):

1. **Target-Token-Only CE Masking & Existing Vocab Tokens**:
   - Mask chat-template wrapper tokens (`\n\n`, EOS, assistant headers) out of CE loss so **only the single target token (`"Yes"/"No"` or `"A".."Z"`) gets gradient** — standardizes convergence across Qwen/Llama/Gemma/Phi.
   - Map class labels to **existing single-token vocab characters** (`A..Z` or single Unicode symbols `■,□,▲`) instead of adding untrained special tokens (`+0.005` CV/LB).
   - At inference, use a **single `forward()` pass** (no `generate()` loop) or `vllm.SamplingParams(max_tokens=1, logprobs=10, allowed_token_ids=[...])` (`allowed_token_ids` prevents valid labels from falling out of top-K logprobs!), plus **length-sorted batching (`2×–4×` speedup)** and **prefix KV-caching** (put shared rule/question *before* variable text!).
2. **Prefix-Shared Suffix Classification (`FlexAttention`) & Grouped Reranking**:
   - Pack `prefix ++ suffix_0 ++ ... ++ suffix_K` in one forward pass with a custom `FlexAttention` mask (`causal & (same_suffix | is_prefix) & same_doc`) so the prefix is encoded once while every candidate suffix interacts deeply with context.
   - For pointwise rerankers (`Qwen3-Reranker`), pack all candidate prompts of a single `row_id` into one grouped `CrossEntropy` batch and include all distractor choices + static per-candidate few-shot examples in context (beats dynamic top-N RAG retrieval, which can overfit train phrasing).
3. **Online Test-Time Training (TTT), Guard Backbones & Deep Mutual Learning**:
   - When unseen test tasks carry labeled exemplars, run online QLoRA TTT inside the submission runtime: drop annotator-invisible shortcut metadata (e.g., `subreddit`), resolve duplicate label conflicts via majority vote or **soft ratio + Tsallis GCE (`q ≈ 0.95`)**, and normalize ensemble outputs via **per-group rank normalization** when metrics average across groups.
   - Prefer **domain-aligned pretrained backbones** (e.g., `ShieldGemma-9B`, `Qwen3Guard` for moderation) over larger generic chat models; boost online TTT further via **Online Deep Mutual Learning (`CE + KL` across 2–3 peer LLMs)**.
   - Under noisy human labels, denoise via **OOF teacher soft-label distillation** (`100%` soft logits on synthetic data, `50/50` on train) + **R-Drop** + trusting **multi-seed CE loss** over noisy discrete `MAP@k`.

---

## 3. Pairwise Reward Modeling, Quantization & Uncertainty Cascades

1. **Privileged-Information Teacher Distillation & Same-Step Order Symmetry**:
   - Prepend **train-only privileged metadata** (e.g., model identities, reports) into 70B+ teacher ensembles to emit calibrated soft targets over unlabeled/tie/synthetic corpora (`T=5, w=0.9` KL loss; filter near-tie pseudo-labels `|p_A - p_B| <= 0.03`) for lean 8B–14B deployment students.
   - Put **both `(A, B)` and `(B, A)` in the same gradient step** to cancel position bias; use **proportional middle truncation** with an explicit **`snip <N> chars`** length cue (beats left-truncation).
2. **Fitting 32B–72B LLMs Inside 2×T4 / 9–12h Budgets**:
   - **GPTQ 8-bit > AWQ 4-bit** on razor-thin margin tasks (or heavy `AutoRound(seqlen=6144, iters=3000)`).
   - **W8A8 `SmoothQuant(α=0.75)` + 2-Layer GPU Streaming**: Unlocks `40+ TFLOPS` INT8 on T4 (`2×` FP16) and streams 32B weights layer-by-layer from **`/kaggle/working`** (**NEVER `/tmp/`**, which is Copy-on-Write with hidden disk caps!).
   - **Split `SequenceClassification` via `vLLM task="embed"`**: Quantize merged backbone to GPTQ-4bit, extract `pooling_type="LAST"` embeddings in vLLM, and multiply by an external FP16 `nn.Linear` head.
   - **Uncertainty-Gated Cascades & Swap TTA**: Score 100% of test with fast 8B–14B models, then route **only the uncertain band** (e.g., `top1 < 0.95` or `|p - 0.5|` smallest `15–40%`) through swapped `(B, A)` TTA (reusing prompt KV prefix!) or 32B–72B models. Watch for **`bf16 + FA2` (A100) → `fp16 + sdpa` (T4)** precision drift (`-0.001` to `-0.002`).
   - **Avoid CoT / Reasoning Models for Single-Token Classification**: Generating critiques blows T4 latency and frequently hurts calibrated single-token accuracy.

---

## 4. Ultra-Low-Resource Machine Translation & Scientific Extraction

1. **Byte-Level Seq2Seq (`ByT5`) + Data Rebuilding + Source-Grounded MBR**:
   - On diacritic-heavy, morphologically rich, or non-standard scripts where subword tokenizers fragment tokens, **`ByT5` (`base/large/xl`)** beats standalone decoder LLMs. Keep Stage-2 fine-tuning in **`FP32`** (BF16 blurs byte diacritics), set **`Adafactor(β₁=0.9)`** when `group_by_length=True` to kill gradient spikes, and stop checkpoints at **`eval_loss` minimum** (BLEU/chrF keeps rising via style overfitting).
   - Rebuild misaligned corpora via layout OCR + VLM **3-page sliding windows** (catches cross-page spill) + **OOF error mining** (flag worst OOF docs for re-extraction) + **embedding-RAG few-shot pseudo-labeling**, then train **2-stage `CPT (WSD + grad_clip=0.3 + AWP) → Clean SFT`** + **reverse back-translation**.
   - Serve via **CTranslate2 `int8_float32` (`~5×` speedup)** into **cross-model MBR** (never single-model multi-checkpoint MBR or sampling-only MBR!) or **Source-Grounded MBR** (adding verifiable source-invariant checks: numbers, entities, `<gap>` parity) / pairwise LLM reranker.
2. **Long-Document Scientific Extraction**:
   - Under entity+type tuple F1, a **type error on a TP costs 2× a retrieval error** (1 FP + 1 FN).
   - **Structured Metadata GBDT > 32B LLM**: Whenever structured external metadata (authors, dates, titles, counts) exists, an article-grouped CatBoost beats 32B LLMs in minutes; reserve constrained `A`/`B` LLMs for unstructured text snippets (abstract + mention window + Data Availability + table captions/headers or top-3 BM25 chunks).
   - **Annotator View Parity**: Audit whether annotators read PDF or XML (exclude XML-only online tables if annotators read PDFs!), stabilize tiny noisy encoders via **target-entity masking + `bs=256` + EMA (`0.9995`)**, and avoid hard-coding Public-LB repository heuristics that collapse on Private split shifts.

---

## 5. Synthesis Amortization, Asymmetric Code Abstention & Quantized Runtime Parity

1. **Offline Synthesis Amortization, Vocab Shrinking & Single-Task LoRA TTT**:
   - When test-time compute cannot host giant reasoning teachers directly, **amortize frontier reasoning offline** by synthesizing `10k–100k+` programmatically verified tasks (plus inverse/augmentation transforms) to pre-train a compact `1B–8B` student.
   - **Task-Alphabet Vocabulary Shrinking**: On closed-symbol structured tasks (grids, coordinates, finite alphabets), prune the `lm_head` and input embeddings down to the exact active token set (`~16–256` tokens) — slashes VRAM, unlocks high-throughput batched decoding, and stabilizes small-data gradients.
   - **Per-Task Online LoRA TTT + Deterministic Search**: Fine-tune a fresh lightweight LoRA per unseen test task using dihedral/permutation augmentations of its few-shot examples, then decode via **batched best-first / DFS log-probability search** (filtering/ranking candidates with exact program or structural verifiers) instead of unguided temperature sampling.
2. **Asymmetric-Penalty Code Repair & Learned Abstention**:
   - Whenever an evaluation metric penalizes a wrong submission far more severely than abstaining (e.g., `(+1, -1, 0)` pass/fail/skip scoring), **abstention calibration beats raw solve rate**.
   - **Verification-First Patching**: Structure agent pipelines around (a) localization → (b) fail-to-pass (F2P) reproduction test → (c) pass-to-pass (P2P) regression baseline → (d) structured **`*SEARCH/REPLACE*`** edits (far higher syntactic validity than raw unified `git diff` hunks).
   - **Calibrated Skip Gate**: Train a lightweight classifier (or LoRA difficulty/verification scorer) on out-of-fold patch execution traces, localization agreement, and unit-test coverage to submit **only** when expected utility `E[score] > 0`.
3. **Quantized Runtime Parity & Full-Trajectory Agentic Optimization**:
   - **Exact Target-Runtime Parity**: When evaluation executes against a quantized inference engine (e.g., GGUF / `llama.cpp` / INT4) or stateful tool harness, reproduce the **exact runtime build, quantization format, chat template, and tool-execution loop** locally — float16/BF16 surrogate scores frequently fail to transfer across quantization boundaries.
   - **Bridge BF16 Optimization → Quantized Targets via Surrogate Reranking**: Generate diverse candidate prompts/triggers with continuous or discrete gradient search (`GCG` / coordinate ascent) on BF16 weights, then **rerank/filter through a fast regression surrogate (e.g., Ridge) trained directly on quantized runtime responses** and verify top candidates end-to-end against live environment state.
   - **Full-Trajectory Loss & Probe Diversity**: Optimize jointly over both the **tool-invocation tokens** and **post-tool follow-up trajectory tokens** (`L_call + L_post_tool`); prefer **diverse single-hop probes** over brittle long multi-hop payload packing, which degrades sharply on unseen private templates and hidden system prompts.

---

## 6. Open-Catalog LLM Bi-Encoder Retrieval & Progressive Multi-Stage Reranking

When retrieving and ranking top-$K$ candidates from a `1k–10k+` text catalog where **a large fraction of test target labels are unseen during training**:

1. **Unseen-Target-Aware Validation**:
   - Grouping folds by fine-grained item/question ID leaves `~80%` seen targets in validation and rewards memorizing train labels. Group folds by a **higher semantic hierarchy** (topic/construct/subject) and track an **unseen-label-only validation slice** alongside overall `MAP@K` / `Recall@K`.
2. **Stage 1 — Optimize Bi-Encoders Strictly for `Recall@K` (Not `MAP@K`)**:
   - Fine-tune decoder-LLM embedders (`14B–32B`, last-token pooling > mean pooling) with `MultipleNegativesRankingLoss` (`temp ≈ 0.01`), enforcing **strictly 1 example per label per batch** to avoid false in-batch negatives.
   - **Recall Trap**: Distilling reranker rankings into a retriever can raise retriever `MAP@K` while *dropping* `Recall@32`—always select Stage-1 checkpoints by **downstream candidate recall** (`Recall@32–64`) and use dynamic similarity cutoffs (`top-32` + up to `32` within `top1_sim - 0.06`).
   - **Closing the Unseen-Catalog Gap Offline**:
     - **Zero-Latency Catalog Expansion**: Pre-generate rich LLM descriptions/examples for every catalog label offline before embedding (`+2–4%` recall at zero test-time latency).
     - **Clustered Contrastive Synthetic Data**: Cluster neighbor catalog entries by confusion/co-occurrence and generate synthetic queries referencing 4–5 sibling labels so the generator teaches fine boundaries, filtered by an LLM judge and cosine deduplication.
     - **Distilled Query-Reasoning CoT**: Fine-tune compact reasoners to explain the query's underlying intent *without* seeing candidates, appending that CoT to both retriever and reranker inputs (with 50% CoT dropout during training).
3. **Stage 2 — Progressive `14B → 32B → 72B` Pointwise-to-Listwise Funnel & Serving**:
   - **Train-Seen-Only Negative Sampling**: Restrict training negatives to catalog labels that *also* appear as positives in train; otherwise the reranker learns a shortcut to penalize unseen catalog entries (`+0.02` to `+0.025` gain!).
   - **Pointwise → Listwise Funnel**: Score `32–64` candidates pointwise (`logits_yes - logits_no` with grouped CE + `vLLM enable_prefix_caching=True`) using a `14B` model → refine top `8` with a `32B` model → jointly rank top `5` (or sliding `10–52`-way single-token `A–Z, a–z` windows with `MultipleChoiceLogitsProcessor`) using a `70B/72B` model.
   - **Order-Bias Mitigation**: Always mitigate listwise candidate-position bias via **random shuffling during training + reverse/permutation TTA at inference**.
   - **Serving 70B/72B on 2×T4 & Weight Merging**: Combine **task-calibrated AutoAWQ / `AutoRound` 4-bit** (generic calibration degrades top-1), `tensor_parallel_size=2`, **`cpu_offload_gb=8`**, `max_tokens=1`, and **parameter-space LoRA averaging / `WiSE-FT` (`alpha ≈ 0.8`)**.
   - **Anti-Patterns**: Iterative hard-negative mining often pulls in unlabeled near-duplicate false negatives and hurts generalization; avoid CoT-reasoning checkpoints as single-token rerankers, and avoid aggressively tuning unseen-class probability multipliers on tiny Public LBs.
