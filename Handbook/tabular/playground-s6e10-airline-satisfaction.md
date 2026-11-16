# Playground Series S6E10: Predicting Airline Passenger Satisfaction — Top 15 Solutions & Post-Mortem

> **Competition**: [Kaggle Playground Series Season 6 Episode 10](https://www.kaggle.com/competitions/playground-series-s6e10) (Oct 2026, 3,747 Teams)  
> **Target**: Binary `satisfaction` (`True` / `False`, Train Positive Rate: `44.3573%`)  
> **Evaluation Metric**: Area Under the ROC Curve ($\text{ROC-AUC}$)  
> **Dataset**: 699,635 Train Rows $\times$ 23 Cols | 299,844 Test Rows $\times$ 22 Cols | Original Seed: 129,880 Rows (`43.4463%` Positive)  
> **1st Place Code & Writeup**: [1st Place — TFMs and LLMs with Tabular Data](https://www.kaggle.com/competitions/playground-series-s6e10/writeups/1st-place-tfms-and-llms-with-tabular-data) & [GitHub 27-Model Repo (`s6e10_airline_satisfaction`)](https://github.com/cdeotte/kaggle-playground-solutions/tree/main/s6e10_airline_satisfaction)

---

## Official Top 15 Leaderboard Standings & Solution Index

The S6E10 race produced dramatic Private LB shakeups driven by blind public blending overfitting the 20% public split, while generative tabular transformers and self-supervised column-token transformers swept the top ranks:

| Private Rank | Team / Competitor | CV (OOF AUC) | Public LB | Private LB | Δ Rank | Core Winning Paradigm |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 **#1** | **Chris Deotte (`@cdeotte`)** | 0.96290 *(0.962836 repo)* | 0.96262 *(0.96247 repo)* | **0.96318** *(0.96314 repo)* | +6 | Target-First ("Storyteller") & Target-Anywhere GPT-2 on GReaT replicas + GReaT Replica Worlds + 9 TFMs + 245 public pairs |
| 🥈 **#2** | **Prior** (`@philippsinger`, `@theklem1234`, `@anuraggarg1209`) | — | — | **0.96308** | +7 | *(No writeup published)* |
| 🥉 **#3** | **Christoffer Thimsen** (`@lowasitgoes`) | — | — | **0.96306** | +8 | *(No writeup published)* |
| 🏅 **#4** | **Ole-Jakob** (`@olejakobrosenvold`) | **0.96279** | — | **0.96304** | +10 | Custom from-scratch **Any-Order Autoregressive Transformer (`AO-ARM`)** (21 streams, `0.96217` CV) + 189 own streams + Claude/Codex `/goal` loop |
| 🏅 **#5** | **Abdelrahman mohsen** | **0.96270** | 0.96242 | **0.96292** | +5 | Own GPU-trained models + public OOF stack; held discipline when Public LB stalled flat across final 3 submissions |
| 🏅 **#7** | **UIT - Loc & An** (Pham Van Loc & Dang Nguyen Thai An) | **0.962473** *(0.962287 Slot A)* | 0.96210 | **0.96263** | +396 | 16 seed-bagged 36-epoch **`GenPure`** causal generative transformers + 56 own streams + frozen 4-gate screen + `L2` logit stack |
| 🏅 **#9** | **Fritz Cremer** (`@fritzcremer`) | — | 0.96213 | **0.96259** | -1 | Amex-style **Self-Supervised Column-Token Transformer** (240–2,000 epochs Masked/RTD pretraining on train+test) + 4 TabPFN-3.5 pairs + GBDTs |
| 🏅 **#11** | **MILANFX** (Xin Feng) | **0.96241** | 0.96205 | **0.96254** | +418 | 17 **`Q-FT` Transformer** variants (ModernBERT, Gated/Differential Attention, Hard-Negative RTD, SAM) on 2× T4 + 300-model `L2` LogReg |

*(Other Top 15 finishers without published writeups: #6 `srikalyan` `0.96280`, #8 `Maher el Ouahabi` `0.96262`, #10 `Vivillon` `0.96254`, #12 `Tataky` `0.96251`, #13 `lucataco` `0.96248`, #14 `Svyatoslav Suglobov` `0.96245`, #15 `tine.sh agent` `0.96244`.)*

---

## 1. Dataset Forensics & Why Generative Models Dominated

### 1.1 Zero Covariate Shift & Synthetic Noise Geometry
- **Adversarial Validation**: 3-fold Stratified LightGBM classifying `train.csv` vs. `test.csv` achieves **$\text{ROC-AUC} = 0.5010 \pm 0.0034$**. Train and Test come from an exact random split with zero covariate drift; all public/private discrepancies stem from finite-sample variance (estimated public-diff noise $\sim 0.00002$ between correlated submissions) and LB-chasing optimism.
- **Label Noise Ceiling vs. Generator Signal**: On the original 129,880-row Airline Passenger Satisfaction survey, an untuned LightGBM reaches `0.9950` AUC. Because Kaggle's synthetic sampler flipped roughly **3.8% of labels** during temperature sampling, a purely white-noise flip would cap AUC at `0.9613`. Generative LLMs broke through `0.9624+` single-model and `0.96318` stacked because the generator's token probabilities encode exact conditional likelihoods $P(\mathbf{x} \mid y)$ across the synthetic rows.

### 1.2 Proof That Kaggle Used GReaT (Label-First GPT-2 Sampling)
Chris Deotte reverse-engineered the exact data-generation process behind S6E10 as **`be-great` (`GReaT`)**:
1. GPT-2 is fine-tuned on the original 129,880-row dataset serialized as `"column is value"` fragments joined by `", "` in random column order.
2. Synthetic rows are sampled token-by-token at `temperature=0.7, top_k=50` starting from a **label-first prompt** (`"satisfaction is True,"` or `"satisfaction is False,"`), which reproduces both the exact training positive rate (`44.36%`) and the empirical distortions relative to the original dataset.
3. **GPT-2 Tokenizer Fingerprints**: 3-digit `Flight Distance` integers are single GPT-2 tokens, while 4-digit distances split into two BPE tokens sharing prefixes.

### 1.3 Key Tabular Quirks & Original Data Handling
- **`Flight Distance` as a Physical Route ID**: All but 4 of the 699,635 training flight distances exist in the original survey data. Standard `max_bin=255` lumps distinct airport routes together; declaring `Flight Distance` as a categorical route ID (for TabPFN-3.5 / target encoding) or raising `max_bin` to `1024–4096` adds up to `+0.00075` AUC.
- **Zero-Rating Bypass Anomaly (Non-Monotonicity)**: Survey ratings span `1–5`, while `0` means *service bypassed / not applicable* (e.g., VIP concierge or pre-booked priority). Rows with $\ge 3$ zero-ratings have $>86.2\%$ satisfaction (vs. `44.2%` baseline), and `Inflight wifi service == 0` has `62.0%` satisfaction vs. `10.5%` for rating `1`. Enforcing monotonic constraints on the 11 service ratings in XGBoost **degraded OOF by `-0.0031` AUC** (verified by 7th place).
- **Original Data Concatenation vs. Lookup / Replica Pretraining**: Directly appending raw original rows into GBDTs without downweighting/flagging degraded 5-fold OOF (`0.958952 -> 0.958699`, `-0.000252` across 0/5 folds). In contrast, mapping original target rates as static lookup priors (`orig_rate__*`, `+0.000927` across 5/5 folds), training teacher models on original data, or pretraining generative GPT-2 replicas / synthetic worlds on original data unlocked clean leak-free gains.

---

## 2. 1st Place Deep Dive: Chris Deotte — Tabular LLMs, Replica Worlds & TFMs

Chris Deotte's [`s6e10_airline_satisfaction`](https://github.com/cdeotte/kaggle-playground-solutions/tree/main/s6e10_airline_satisfaction) repository releases **27 internal models** trained on `StratifiedKFold(5, shuffle=True, random_state=42)` that score **0.962836 CV / 0.96247 Public / 0.96314 Private** completely standalone (still winning 1st place without public OOFs). Combining 25 of those models with 245 public OOF/test pairs (`goodpjw2008`, `kratosyan`, et al.) yielded **0.96290 CV / 0.96262 Public / 0.96318 Private** (an alternate 332-pair stack reached `0.96299` CV / `0.96260` Public).

### 2.1 Two New Tabular LLMs (No Feature Tokenizer Needed)
Instead of converting columns into learned embeddings, each row is formatted identically to `GReaT` training text (`ints` as `'36'`, `floats` as `'0.0'`, `NaN` as `'None'`, target as `'True'`/`'False'`), tokenized with **GPT-2's BPE tokenizer** (digit-by-digit tokenizers such as Qwen performed worse):
```text
Gender is Female, Customer Type is Loyal Customer, Age is 36, ..., Arrival Delay in Minutes is 0.0
```

| Paradigm | Nickname | Formulated Distribution | Training & Scoring Mechanics | S6E10 OOF AUC |
| :--- | :--- | :--- | :--- | :---: |
| **Target Last** | **The Detective** | $P(y \mid \mathbf{x})$ | Discriminative causal LM reading 21 shuffled features then predicting next-token prob of `" True"` vs. `" False"` after `"satisfaction is"`. 1 forward pass per row. | `0.960961` (~0.961) |
| **Target First** | **The Storyteller** | $P(\mathbf{x} \mid y) \xrightarrow{\text{Bayes}} P(y \mid \mathbf{x})$ | Generative causal LM starting with `"satisfaction is True,"` / `"satisfaction is False,"` + 21 shuffled features + `<eos>`. Scored across **30 random column orders** via `logsumexp`. | **`0.962425`** (`0.96189` Public alone) |
| **Target Anywhere** | — | Joint $P(\mathbf{x}, y)$ | Target sits at a random position among all 22 fields during training; scored by averaging per-order log-odds over 30 orders (`24 epochs`). | **`0.962452`** (Highest single GPT-2) |

#### The 3-Step "Storyteller" Recipe (`gpt2_target_first`)
1. **GReaT Replica Pretraining (`models/gpt2_replica`)**: Pretrain GPT-2 from scratch as a GReaT replica *exclusively on the 129,880 original rows* (`be-great==0.0.3`, `GReaT(llm="gpt2", batch_size=16, epochs=60).fit(original_df)`). **Critically, 60 epochs beats an 8-epoch replica by `+0.0012` final AUC**, and starting from plain GPT-2 is far worse.
2. **Fold Fine-Tuning**: Continue training each fold's replica on competition training rows with target first and 21 features in fresh random column order every step, computing loss on all tokens (`8 epochs`, `AdamW`, `lr=5e-5` linear decay to 0, `batch_size=32`, `bf16`; ~600 rows/s, ~2h/fold on 1× A100; EMA decay `0.9999` for medium/large).
3. **Multi-Order Bayes Readout**: Evaluate $\log P(\mathbf{x} \mid \text{True}, o)$ and $\log P(\mathbf{x} \mid \text{False}, o)$ over 30 random feature orders $o$ and pool with `logsumexp` plus the training label log-odds prior ($\pi_1 \approx 0.4436$):
   $$\text{log-odds}(y=1 \mid \mathbf{x}) = \log \frac{\pi_1}{1 - \pi_1} + \operatorname{logsumexp}_{o=1}^{30} \log P(\mathbf{x} \mid y=\text{True}, o) - \operatorname{logsumexp}_{o=1}^{30} \log P(\mathbf{x} \mid y=\text{False}, o)$$

### 2.2 Replica Worlds, Hybrid BCE, & All 27 Open-Source Models
Chris also generated two ~1M-row synthetic labelled **"Replica Worlds"** (`M32_1M` and `MB32_1M`) by sampling from noised GReaT GPT-2 medium replicas trained solely on `data.csv` (with 7%–9% label flips and $\pm 1$ rating jitter at `temp=0.7, top_k=50`), then pretrained field transformers and BatchEnsemble MLPs on those worlds before fine-tuning on the 5 competition folds:

| Folder | Pair / Variant | Architecture / Method Summary | OOF AUC |
| :--- | :--- | :--- | :---: |
| `gpt2_label_anywhere` | `gpt2_small_anywhere_24ep` / `medium_8ep` | GPT-2 small (24ep) & medium (8ep), target at random pos, mean per-order LLR | **0.962452** / 0.962407 |
| `gpt2_target_first` | `gpt2_small` / `medium` / `large_target_first` | Target-first Storyteller GPT-2 small, medium, large (8 epochs, 30 orders) | **0.962425** / 0.962370 / 0.962360 |
| `gpt_hybrid` | `gpt_hybrid_cv6` | Continues `gpt2_small_target_first` for 1 epoch with gen loss $+ 0.1 \times$ BCE | 0.962378 |
| `gpt_neo_target_first` | `gpt_neo_target_first` | EleutherAI `gpt-neo-125M` Storyteller (60-epoch replica + target first) | 0.962159 |
| `m19_source_aug` | `m19_source_aug_bag2_aligned` | Masked-field transformer pretrained on Replica Worlds + source, fold fine-tuned | **0.961409** |
| `tabpfn` | `tabpfn35` | Full-context zero-shot **TabPFN-3.5** on raw columns | **0.961145** |
| `field_be` / `pair_be` | `field_be_transfer_aligned` / `pair_be` | BatchEnsemble MLPs (field view & pair view), source/world pretrained | 0.960886 / 0.960170 |
| `qft_v2` / `qft` | `qft_v2` / `qft` | Order-free `ModernBERT` + side features + RTD pretraining (`qft_v2`) vs `Qwen3-0.6B` LoRA (`qft`) | 0.960798 / 0.959795 |
| `m23_bce` | `m23_bce` | Label-conditional masked-field transformer + BCE continuation | 0.960796 |
| `gpt_fold_be` | `gpt_fold_be_bag2_aligned` | BatchEnsemble head on frozen GPT-2 fold-model features + world vocab stats | 0.960561 |
| Other 8 TFMs | `kumo_large` (`0.960036`), `tabfm` (`0.958871`), `exaone` (`0.958557`), `kumo_small` (`0.958484`), `limix2` (`0.958474`), `causilo` (`0.958275`), `tabicl2` (`0.958088`), `mitra2` (`0.957075`) | Full-context zero-shot Tabular Foundation Models on raw columns (9 TFMs stacked alone = `0.96120` CV) | 0.957075–0.960036 |
| Raw Baselines | `xgboost_raw` / `realmlp_raw` | 8-seed XGBoost & RealMLP baselines on raw features | 0.959852 / 0.959311 |
| **Stacker (`stack.py`)** | **27 Repo Pairs (`cuML L2 LogReg`, `C=1.0`)** | 5-fold CV stack over logits clipped `[-30, 30]`, averaged across 5 cuML refits | **0.962836** (`0.96314` Private) |

- **Impact in the 259-Pair Stack**: Stacking 245 public pairs alone yielded `0.96218` CV / `0.96186` Public LB. Adding the first target-first GPT-2 pair jumped CV `+4.9e-4` to `0.96266` (`0.96223` Public LB). Across all 259 pairs (`0.96290` CV / `0.96262` Public LB / `0.96318` Private LB), the **10 GPT pairs carried ~80% of the stack's output weight**, while a public route-categorical TabPFN-3.5 pair (`@goodpjw2008`) received the largest non-GPT weight.

---

## 3. 4th Place Deep Dive: Ole-Jakob (`0.96304` Private / `0.96279` CV)

Ole-Jakob finished 4th solo (+10 places on Private LB) using **zero public OOFs or submissions**, running 189 self-trained streams across 10 families on one local RTX 5090 + Colab via an autonomous **Claude Code + Codex `/goal`** harness (Claude logged 77 streams, Codex 71 streams, sharing a ledger and diversity gate against a fixed LR stack):

1. **From-Scratch Any-Order Autoregressive Transformer (`AO-ARM`)**:
   - Independently arrived at the exact generative Bayes concept as Chris Deotte's Storyteller, but built a **custom Transformer from scratch on learned column/value embeddings** (no pretrained GPT-2 weights).
   - Pretrained label-free on `train + test + original` rows with shuffled column orders, then trained on each fold under both label hypotheses (`satisfied` vs. `dissatisfied`).
   - At inference, averaged log-likelihood ratios over **32 random orders** and adjusted with the training prior; a single 6-fold AO-ARM stream scored **`0.96201` CV / `0.96258` Private** (top-10 alone!). Across the final stack, 21 AO-ARM streams (peak `0.96217` CV) contributed **~70% of the net standardized coefficients**.
2. **189-Stream Family Breakdown (`0.96279` CV Logistic Regression + Signed-Pooling Correction)**:
   - `AO-ARM transformers` (21 streams, `0.96217`), `XGBoost incl. NCE` (51, `0.96184`), `LightGBM` (8, `0.96181`), `Other neural models` (TabM, masked generative MLPs, set transformers, DeepFM, ResNets — 19, `0.96176`), `TabPFN` (14, `0.96154`), `RealMLP` (41, `0.96140`), `Other TFMs` (TabFM, Kumo, EXAONE, Causilo, TabICL, Mitra — 14, `0.96140`), `CatBoost` (16, `0.96117`), `AutoML` (AutoGluon + LightAutoML — 2, `0.95898`), `LR & EBM` (3, `0.95328`).
3. **Two Critical Mistakes Documented**:
   - *Stopping a DistilGPT-2 replica too early*: Parking DistilGPT-2 after 4 label-free epochs (`0.9595` OOF) missed Chris's 60-epoch recipe (`0.9624` OOF) — *"park the exact recipe that failed, not the whole idea."*
   - *Selection overfitting*: SFS/SBS and hill-climbing pushed CV to `0.962820` but lost `5–10u` on LB vs keeping all 189 streams in $L_2$ LR without member selection.

---

## 4. 5th, 7th, 9th & 11th Place Solutions Distilled

### 4.1 5th Place: Abdelrahman mohsen (`0.96292` Private / `0.96270` CV, +5 Ranks)
- Joined only in the last few days, combining self-trained GPU models with public OOF datasets (`@cdeotte`, `@goodpjw2008`, `@dariushafshar`, `@kagankoral`, `@koumeimaki`).
- Demonstrated strict CV trust when Public LB stalled completely across the final three submissions (`CV 0.96264 -> 0.96268 -> 0.96270`; `Public LB 0.96241 -> 0.96240 -> 0.96242`), while Private LB tracked CV upward monotonically (`0.96289 -> 0.96291 -> 0.96292`).

### 4.2 7th Place: Team UIT - Loc & An (`0.96263` Private / +396 Ranks from #318 Public)
Pham Van Loc and Dang Nguyen Thai An leaped 396 spots on Private LB with a **frozen, leak-audited 56-stream pipeline** built on one fixed split (`StratifiedKFold(5, shuffle=True, random_state=20261018)`):

1. **`GenPure` — Seed-Bagged Generative Causal Transformer (16 Long Seeds, `0.96042–0.96119` OOF)**:
   - Built a 6-layer, width-256, 8-head causal transformer from scratch over 21 column tokens (numeric values seen $<3$ times grouped into 250-wide `Flight Distance` bins or $\log_2$ bins).
   - Trained with a **random column order per batch**; held-out fold and test rows contributed feature-only likelihood at weight $0.5$ (zero labels). Scored via Bayes log-odds over 8 fixed orders using `log-mean-exp`, calibrated with per-fold Platt scaling on a 10% inner holdout.
   - **Two Crucial Ablations**: (a) Pure generative loss (`GenPure` `0.95960`) outperformed adding a discriminative BCE loss (`GenHybrid` `0.95907`); (b) **Turning early stopping OFF** and training a full **36 epochs** (while lr decayed to 0) lifted single-seed OOF by **`+0.0016`** (peaking at `0.96119`), and bagging 16 long seeds kept lifting stack CV — the final 4 seeds bumped Private LB from `0.96261` (#8) to **`0.96263` (#7)**.
2. **Dual-Slot Architecture & Frozen Pre-Registered Blend Screen**:
   - **Slot A (Own-Only Stack, 56 streams, `0.962287` CV / `0.96259` Private)**: M&M standardized logit stack (`clip [1e-6, 1-1e-6]`, fold-only standardization, `LogisticRegression(C=0.3, solver='lbfgs', tol=1e-10)`) with **zero stream selection** (no optimism bias).
   - **Slot B (Public-Assisted Blend, `0.962473` CV / `0.96263` Private)**: `0.55 * Goodpjw245 + 0.30 * SlotA + 0.15 * GenBagL`, chosen by nested CV from 68 candidates passing a pre-registered 4-gate screen (`>= +0.000020` pooled gain, $\ge 4/5$ folds up, worst fold $\ge -0.000020$, mean gain $\ge 2$ SE). Every stream required SHA-256 verification and $10^{-9}$ OOF audit.
   - **GPU `bf16` Trap on NVIDIA T4**: PyTorch silently emulates `bf16` slowly on T4 GPUs (~10 min/epoch); switching to `fp16` with `GradScaler` restored full speed.

### 4.3 9th Place: Fritz Cremer (`0.96259` Private / `0.96213` Public)
- Reused his Amex gold-medal recipe: a **Column-Token Transformer** (`width 256–1024`, 4–8 layers, QK-norm) **self-supervised pretrained for 240–2,000 epochs** on all `train + test` rows (without labels) using **Masked-Token Reconstruction + Replaced-Token Detection (RTD)**, followed by a brief 3–4 epoch 5-fold supervised fine-tune with small exponentially decaying learning rate.
- Single `d1024` fine-tuned model scored `0.96187` Private (`0.96127` Public); blending 13 transformer fine-tunes scored `0.96202` Private (`0.96167` Public). Stacked via nested CV with route-feature LightGBM, original-data model, kNN, MLP/DCN, and 4 public TabPFN-3.5 pairs (`0.96259` Private).

### 4.4 11th Place: MILANFX / Xin Feng (`0.96254` Private / +418 Ranks)
- After GBDTs, RealMLP (`0.96131`), TabM, and T4-constrained TFMs (`0.96068`) plateaued (`0.96213` OOF / `0.96235` Private), Xin Feng trained **17 `Q-FT` Transformer variants across 2× T4 GPUs** in the final 5 days, jumping the 300-model $L_2$ LogReg stack (`C=0.001`) to **`0.96241` OOF / `0.96254` Private**:
  - `ModernBERT-base + RTD + Train TE` (1 seed): `0.96172` OOF.
  - `Gated Attention + Near-value RTD + Train TE + SAM Optimizer` (1 seed): `0.96184` OOF.
  - `Differential Attention + RTD + Train TE` (3 seeds): `0.96188` OOF.
  - `Gated Attention + Hard-negative RTD + Train TE` (3 seeds): **`0.96201` OOF** (`0.96175` Public).
- **Simulated Public/Private Hill-Climbing Proof**: Splitting OOF randomly into 50k fake Public and 200k fake Private rows showed that running greedy hill-climbing against fake Public LB **degraded fake Private LB within 3 rounds on every single seed**.

---

## 5. S6E10 Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact Across Top 15 Teams | Root Cause |
| :--- | :--- | :--- |
| **Monotonic Constraints on 11 Service Ratings** | **`-0.0031` OOF AUC** (7th place XGBoost) | Rating `0` represents *bypassed VIP/concierge steps* (`62%` satisfied on Wi-Fi `0` vs `10.5%` on Wi-Fi `1`), breaking monotonicity. |
| **Blind Blending / Public Hill-Climbing** | Fake Private LB drops within **3 rounds** (11th place); **-200 to -1000 rank** shakeups | 20% Public split variance ($\sim 2\text{u}$) dominates micro-weights; SFS/SBS on stacked OOFs overfits (`-5u` to `-10u` in 4th place). |
| **Rating Sum/Mean/Std Aggregates in GBDTs** | **`-0.00022` OOF AUC** | Collinear aggregates dilute `colsample_bytree` away from primary splitters (`Online boarding`, `Inflight entertainment`, `Type of Travel`). |
| **Appending Unweighted Original Rows Directly** | **`-0.000252` OOF AUC** (0/5 folds up) | Original dataset lacks the ~3.8% synthetic label noise and generator distortions; use lookup priors, replica pretraining, or sample-weighted flags instead. |
| **Text-Pretrained ModernBERT Weights for Set Encoder** | **`0.96019` vs `0.96061`** from scratch (7th place) | NLP weight initialization hurts order-free tabular set encoders compared to tabular RTD pretraining from scratch. |
| **Discriminative BCE Loss Inside Generative Transformer (`GenHybrid`)** | **`0.95907` vs `0.95960`** `GenPure` (7th place); `0.962378` vs `0.962425` (1st place) | Adding discriminative BCE shifts the transformer away from learning pure conditional $P(\mathbf{x} \mid y)$ density, reducing diversity with GBDTs. |
| **Stopping Generative / Replica Pretraining Early** | **`0.9595` (4 ep) vs `0.9624` (60+8 ep)** (4th vs 1st); **`-0.0016`** (12 vs 36 ep, 7th) | Generative transformers need full cosine schedules without early stopping (`patience=4` cost `-0.0009` in 7th place). |
| **Forward Selection on Top of Public 245-Stack** | **`+0.0000003`** honest nested CV (7th place) | Earlier apparent gains came from selection optimism on the same OOF labels. |
