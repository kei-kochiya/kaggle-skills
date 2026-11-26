---
name: kaggle-llm-reasoning-playbook
description: >-
  Universal competitive LLM inference, test-time compute scaling, tool-integrated reasoning (TIR), serving optimization, and statistical validation playbook for Kaggle NLP, math, and reasoning code competitions. Use when serving large open LLMs under fixed GPU/time caps, designing Python tool sandboxes, allocating per-problem time budgets, aggregating parallel rollouts, or avoiding small-benchmark overfitting.
---

# Kaggle LLM & Reasoning Competitions Playbook

A universal systems-and-inference runbook for code competitions running open-weight LLMs under strict GPU memory and wall-clock limits. Competition-specific prompts and benchmark IDs remain strictly in `Handbook/nlp/`.

---

## 1. Core Principle: Model Capability & Inference Systems Beat Ad-Hoc Tweaks

In time-bounded open-LLM competitions:
- **Prefer inference-time systems, throughput, budgeting, and aggregation over risky SFT/RL** unless you have a large verified corpus and compute budget—naive SFT on strong MoE reasoning checkpoints frequently causes behavioral regression.
- **Treat serving efficiency as accuracy**: Higher tokens/sec and larger single-wave batch capacity directly buy more parallel reasoning attempts and deeper verification within the same wall-clock budget.

---

## 2. Phase 1: Single-GPU Serving & Tool-Sandbox Engineering

Refer to [`references/vllm_tir_budgeting_and_voting.md`](./references/vllm_tir_budgeting_and_voting.md) for concrete configurations.

1. **OS Page-Cache Weight Preloading**:
   - Sequentially read model shard files into the OS page cache before starting the inference engine to cut fragmented startup I/O out of the fixed competition clock.
2. **Single-Wave VRAM Sizing (FP8 KV-Cache + Prefix Caching)**:
   - Enable **8-bit FP8 KV-cache** (`kv_cache_dtype="fp8_e4m3"`), **prefix caching** (`--enable-prefix-caching`), and async scheduling (`--async-scheduling`, `gpu_memory_utilization=0.95–0.96`).
   - Size parallel attempt count $N$ to fit cleanly in **one simultaneous GPU wave** (e.g., `N=8` at 64k–81k context on 80GB VRAM). Pushing $N$ beyond single-wave VRAM capacity triggers multi-wave queuing or OOMs and typically degrades accuracy.
3. **Pre-Warmed Persistent Code Sandbox Pool (`2×` Workers)**:
   - For Tool-Integrated Reasoning (TIR), pre-start a pool of **persistent stateful Jupyter kernels** (`2 × N` headroom) with core libraries pre-imported (`math`, `sympy`, `numpy`, `mpmath`, `itertools`, `collections`).
   - Stateful kernels preserve intermediate variables across turns and eliminate per-call cold-start latency.
4. **Token & Streaming Hygiene**:
   - Keep system/tool instructions concise (hundreds of wasted prompt tokens multiply across `N attempts × turns × problems`).
   - **Stream-halt** each trajectory the instant a valid final answer delimiter completes, immediately releasing KV blocks for remaining workers.

---

## 3. Phase 2: Sequential Stochastic Resource Allocation & Early Stopping

When total wall-clock time $T_{\text{total}}$ is fixed across $M$ sequential problems of unknown difficulty order:

1. **Bounded Dynamic Per-Problem Budgeting**:
   - Maintain remaining budget $B_{\text{rem}} = (T_{\text{total}} - T_{\text{elapsed}} - T_{\text{reserve}}) / M_{\text{rem}}$, clipped to calibrated `[base_timeout, high_timeout]` bounds (`base_timeout` strictly above the observed fast-solution floor; `high_timeout` capped so one intractable problem cannot starve the queue).
2. **Mathematically Uncatchable-Leader Early Stopping**:
   - Halt remaining parallel attempts early as soon as either target consensus is reached **or** no runner-up can mathematically tie/overtake the leader:
     $$\text{votes}(\text{leader}) \ge \text{early\_stop} \quad \text{OR} \quad \text{votes}(\text{second}) + \text{remaining\_attempts} \le \text{votes}(\text{leader})$$
   - Reallocate saved seconds straight into the pool for harder downstream problems.
3. **Graceful Deadline Degradation**:
   - Never allow a notebook timeout (which invalidates the entire submission). As remaining time shrinks near the cap, automatically cut parallel attempt depth and verification turns, emitting the best-valid partial candidate or deterministic valid fallback.

---

## 4. Phase 3: Answer Extraction & Confidence-Weighted Aggregation

1. **Strict Delimiter Parsing (Zero Greedy Prose Fallbacks)**:
   - Extract strictly from the required output delimiter (e.g., `\boxed{...}` or explicit JSON/XML tag), normalizing whitespace/commas and taking **last-match-wins**.
   - **Never add loose natural-language regex fallbacks** (`"final answer is N"`)—they silently trigger on intermediate base-case or scratchpad sentences.
2. **Vote-Dominant + Tail-Windowed Confidence Aggregation**:
   - Pure unweighted majority voting ignores calibrated certainty, while pure `sum(1 / mean_entropy)` lets one ultra-confident wrong fluke overrule a 5–1 majority and is polluted by noisy early exploration tokens.
   - Prefer either:
     - **Vote-dominant + tail-windowed median entropy** (compute token entropy strictly over the **last $K$ commitment tokens**, take **median** across supporting attempts, and weight vote share highest), OR
     - **Simple inverse mean-entropy (`1 / mean_entropy`)**—avoid multi-factor hand-tuned entropy formulas with un-tunable coefficients.

---

## 5. Phase 4: Prompting & Statistical Validation Rigor

1. **Offline Error-Taxonomy Prompting (Anchored to Model Capability)**:
   - Log hundreds of local traces and categorize concrete failure modes (missed constraints, unpruned brute-force loops timing out Python, floating-point inaccuracies where exact integer `int`/`pow(a,b,m)` is required).
   - Keep prompts crisp and targeted; **avoid multi-persona voter prompts** (repeatedly shown neutral or harmful) and **avoid lowering temperature too aggressively** (`temp < 1.0` collapses parallel path diversity).
2. **Adversarial Ordering & Multiple-Testing Discipline**:
   - Small stochastic test sets (`N = 50–100`) have huge run-to-run variance (`±6–10%`).
   - Build a local **100–200+ problem difficulty-stratified benchmark** (`Basic / Harder / Hardest`), evaluate time-budget policies across **4 adversarial orderings** (`hard_front`, `hard_middle`, `hard_back`, `hard_random`) × multiple repeats, and apply **Benjamini-Hochberg multiple-testing correction** before accepting pipeline modifications.

---

## 6. Phase 5: Post-Training, Classification, Reward Modeling, Extraction & MT

Refer to [`references/llm_posttraining_classification_and_mt.md`](./references/llm_posttraining_classification_and_mt.md) for full cross-competition recipes:
1. **Algorithmic "Trainable-Trace" SFT**: When fixed-budget deduction is the bottleneck, enforce the 5 mechanical trace rules, decouple **Stage-1 lookup memorization → Stage-2 DFS/CSP execution**, compact tokens + train `lm_head`, export native BF16 PEFT (never 4-bit+packing on Mamba-2 SSM), and validate via end-to-end rollout accuracy + perfect trace rate.
2. **Single-Token Classification, Reranking & Online TTT**: Mask wrapper tokens so **only the target token** gets CE loss, map labels to **existing single-vocab tokens** (not new special tokens), use `FlexAttention` prefix-shared suffixes or grouped candidate rerankers, run online QLoRA TTT + **Online Deep Mutual Learning (`CE+KL`)** + **safety/domain-pretrained backbones** (`ShieldGemma`/`Qwen3Guard`) + per-group rank normalization.
3. **Pairwise Reward Modeling & T4 Serving Cascades**: Distill **privileged-info 70B+ teachers** (`T=5, w=0.9` KL, same-step `(A,B)+(B,A)` symmetry, proportional middle truncation + `snip <N> chars`) into 8B–14B students; prefer **GPTQ 8-bit > AWQ 4-bit**, **W8A8 `SmoothQuant` + 2-layer GPU streaming (from `/kaggle/working`, never `/tmp/`)**, `vLLM task="embed"` + external FP16 head, `allowed_token_ids`, and **uncertainty-gated swap TTA / 3-tier cascades**.
4. **Low-Resource MT & Scientific Extraction**: Use **byte-level `ByT5` (`FP32` Stage-2, `Adafactor β₁=0.9`)** + VLM 3-page PDF rebuilding + OOF error mining + 2-stage `CPT→SFT` + reverse back-translation + `eval_loss` stopping + CTranslate2 `int8_float32` + **cross-model / Source-Grounded MBR**. On scientific extraction, prefer **structured-metadata CatBoost over 32B LLMs** when metadata exists, enforce PDF-vs-XML annotator parity, and stabilize small noisy encoders with target-entity masking + `bs=256`/EMA.
5. **Offline Synthesis Amortization, Asymmetric Abstention & Quantized Runtime Parity**: Under tight test-time caps, amortize search/synthesis into **consistency-verified offline pretraining + vocab-shrunk single-task TTT + compute-matched 2-checkpoint self-ensembles** (or continuous soft-masking diffusion / 2-layer recursive networks); under asymmetric `+1 / -1 / 0` SWE/code penalties, gate `SKIP`s via **reproduction-first F2P+P2P execution tests** or a **calibrated difficulty classifier** and emit **`*SEARCH/REPLACE*` → `git diff`** instead of raw diffs; when optimizing prompts/tokens for quantized serving, calibrate BF16→GGUF drift with a ridge reranker, pin exact serving-engine minor versions, and profile **full multi-turn latency (`T_call + T_post_tool`)** via interleaved A/B micro-benchmarks.

---

## 7. References & Case Studies
- [vLLM Serving, Persistent TIR Pools, Time Budgeting & Consensus Voting](./references/vllm_tir_budgeting_and_voting.md)
- [LLM Post-Training, Classification, Reward Modeling, Extraction & MT](./references/llm_posttraining_classification_and_mt.md)
- Case Studies: [`AIMO3`](../../../Handbook/nlp/aimo-progress-prize-3.md) · [`Nemotron Reasoning`](../../../Handbook/nlp/nvidia-nemotron-reasoning.md) · [`ARC Prize 2025`](../../../Handbook/nlp/arc-prize-2025.md) · [`Konwinski Prize`](../../../Handbook/nlp/konwinski-prize.md) · [`AI Agent Security`](../../../Handbook/nlp/ai-agent-security-tool-attacks.md) · [`Deep Past Akkadian MT`](../../../Handbook/nlp/deep-past-akkadian-mt.md) · [`Jigsaw Agile Rules`](../../../Handbook/nlp/jigsaw-agile-community-rules.md) · [`MAP Math`](../../../Handbook/nlp/map-student-math-misunderstandings.md) · [`Make Data Count`](../../../Handbook/nlp/make-data-count-references.md) · [`WSDM Cup Arena`](../../../Handbook/nlp/wsdm-cup-multilingual-chatbot-arena.md)
