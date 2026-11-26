# OTTO — Multi-Objective Recommender System 🏆

> **Track:** Tabular & Recommender Systems (Featured) · **Scale:** 2,574 teams · \$30,000 · Closed Feb 1, 2023  
> **Core Lesson:** Two-stage session recommendation (`Recall-Maximizing Multi-Source Retrieval → Multi-Objective GBDT/GNN Reranking`) wins via **leak-free temporal/fold separation**, **multi-view time-decayed co-visitation + neural/graph retrievers**, **`CatBoost PairLogitPairwise` / `LGBM lambdarank` / Subgraph GNN** diversity, and **calibrated rank/probability blending**.

---

## 1. Competition DNA & Leaderboard

- **Task:** Given truncated e-commerce sessions (`train`: 4 weeks, `12,899,779` sessions, `1,855,603` unique `aid`s, `216.7M` events; `test`: Week 5, `1,671,803` sessions, `13.85M` events), predict up to **20 `aid`s** per session for each of three objectives: next `clicks` (1 ground-truth item), remaining-session `carts` (all carted items), and remaining-session `orders` (all ordered items).
- **Metric (Weighted Recall@20):**
  $$\text{Score} = 0.10 \cdot R_{\text{clicks}} + 0.30 \cdot R_{\text{carts}} + 0.60 \cdot R_{\text{orders}}, \quad R_{\text{type}} = \frac{\sum_i |\hat{Y}_{i,\text{type}} \cap Y_{i,\text{type}}|}{\sum_i \min(20, |Y_{i,\text{type}}|)}$$
- **Data Properties:** Event mix is `89.85%` clicks / `7.80%` carts / `2.35%` orders (density `0.0005%`). Every session has `>= 2` clicks, all test `aid`s appear in train, and carts/orders can occur without an in-window click (pre-existing carts/wishlists).

| Private Rank | Shakeup | Team | Private LB | Public LB (Rank) | Core Architecture |
| :---: | :---: | :--- | :---: | :---: | :--- |
| 🥇 **1st** | `—` | **mrkmakr** | **0.60503** | 0.60541 (#1) | Multi-hop co-visit + Next-Multi-Aid NN + 9× LGBMRanker (~1,200 cand pool, top 100 feats) |
| 🥈 **2nd** | `↑ 4` | **SOS3** (`ONODERA`, `Silogram`, `senkin13`, `30CrMnSiA`) | **0.60446** | 0.60406 (#6) | 20–30 GPU co-visit matrices + `CatBoost PairLogitPairwise` + LGBM/XGB + rank blend |
| 🥉 **3rd** | `—` | **G & B & D & T** (`Giba`, `Benny`, `Theo`, `C. Deotte`) | **0.60382** | 0.60437 (#3) | 20 co-visit rules (`0.590` alone!) + 5-fold Week-4 leak-free FE + XGB/CatBoost/LGBM rank blend |
| 🏅 **4th** | `—` | **CWKTJ** | **0.60381** | 0.60433 (#4) | Co-visit + W2V + BPR/ALS + GBDTs + hybrid score & index-score blending |
| 🏅 **5th** | `↓ 3` | **NikhilMishra** | **0.60375** | 0.60451 (#2) | Normalized asymmetric co-visit (Optuna-tuned) + single unified multi-objective LGBM (`6:3:1`) |
| 🏅 **6th** | `↓ 1` | **THLUO** | **0.60341** | 0.60412 (#5) | Single-model `LGBM binary` (`0.5915 CV → 0.60341` Priv) |
| 🏅 **7th** | `↑ 4` | **Jack & toshi_k** | **0.60313** | 0.60311 (#11) | Log-Naive-Bayes + LGBM (`0.60008`) blended with **8-layer Subgraph GNN (`0.59874` Priv)** |

---

## 2. Validation Strategy & Leakage Prevention

1. **Temporal Truncation Validation (Radek's Split):** Truncate **Week 4** (`1,801,251` sessions) with the host's exact cutoff distribution into input (`Val A`) and future targets (`Val B`), using Weeks 1–3 as history. Local CV sits `~0.010–0.013` below LB (3 weeks vs 4 weeks of history) but correlates nearly 1:1 (`0.5915 CV → 0.60335 LB` in #6; `0.5917 CV → 0.6028 LB` in #3).
2. **Preventing Stage-1 → Stage-2 Distribution Shift & Target Leakage:**
   - **5-Fold Session-Split Feature Separation (`BenediktSchifferer`, #3):** Truncating 100% of Week 4 starves co-visitation matrices of recent Week-4 dynamics (`0.584 LB`). Splitting Week 4 into **5 session folds**—truncating only 1 fold (20%) as targets while keeping Weeks 1–3 + the remaining **80% full Week 4** + test inputs for co-visitation/item statistics—jumped to `0.591 LB` (1 fold) and `0.598 LB` (all 5 folds).
   - **Strictly Disjoint Temporal Windows (`Jack`, #7):** Built 3 separate co-visitation tables for train (Week 3 targets), validation (Week 4 targets), and test (Week 5) so co-visitation counts never absorb target events.
   - **Multi-Seed Cutoff Augmentation:** Generating 5 random truncation cutoffs per session across 3 weeks (`#11`) added `+0.001 CV`; recovering sessions discarded by the default split script (`#3`) added `+0.0005–0.001` on carts/orders.
   - **Fast Iteration Proxy (`#1`):** Train on 5% of sessions and validate on 10% (~3h full cycle) before scaling to 100%.

---

## 3. Stage 1 — Candidate Generation & Memory Engineering

### Multi-View Co-Visitation & Rule Retrievers
- **20 Co-Visitation Matrices = `0.590 LB` Rules-Only (`Chris Deotte`, #3):** Scaling from 3 baseline matrices (`0.575 LB`) to 20 specialized matrices reached 49th place with zero ML:
  - Consecutive pair windows (`|Δk| == 1, <= 2, <= 3, <= 6`), forward-only `clicks/carts/orders → carts/orders`, exponential time decay `0.5^(|Δts|/3600)`, recency slices (last 3w / 2w / 1w / 2h), cold-start first-item pairs (`n == 0`), session-length splits (`< 6` vs `> 6`), and time-of-day splits.
- **Multi-Hop Beam Co-Visitation (`#1`):** Iterated co-visitation matrices across multiple hops (beam search). Ablation: dropping **visited `aid`s** lost `-0.00223 CV` (`0.58836 → 0.58613`); dropping **co-visitation** lost `-0.00522 CV` (`→ 0.58314`).
- **Normalized Asymmetric Distance Weighting (`#5`):** `w(aid_i, aid_j) = |j - i| / freq(aid_i)` with backward weight halved (`w/2`) for causality, tuning retriever mix weights via **Optuna**.
- **Log-Naive-Bayes Posterior (`#7`):** Recency-weighted PMI over the last 10 events: $\log P(a) + \sum_{i=1}^{10} \frac{11-i}{10} \log \frac{P((a_i, t_i), a)}{P(a_i, t_i) P(a)}$.
- **Inactivity Session Segmentation (`#9`, `#20`):** Splitting raw sessions on `> 900s` or `> 4h` gaps before co-occurrence counting sharpened transition matrices.

### Neural, Graph & Embedding Retrievers
- **Next-Multi-Aid MLP / Transformer (`#1`):** Shared `x_aid`/`y_aid` embedding table + cyclical hour/day time encodings + **target-type conditioning** + a dedicated variant trained strictly on **non-visited targets** (maximizing complementarity with revisit heuristics). Dropping the NN lost `-0.00530 CV` (`0.58836 → 0.58306`).
- **Subgraph GNN (`toshi_k`, #7):** Extracted per-session subgraphs from the global top-`P` (`P=20,30,40`) transition graph and ran an **8-layer GCN** (18 node feats, 14 edge feats, 3-head softmax/BCE). Single GNN hit **`0.59719` Private** (1,024 cands); 4-GNN ensemble reached **`0.59874` Private**!
- **SRGNN (`#12`):** Adding top-172 SRGNN candidates to history + co-visitation pushed overall recall ceiling from `0.675` to **`0.695`** (`orders 0.746`, `carts 0.581`, `clicks 0.734`).
- **Word2Vec / ProNE / BPR-MF / SKNN:** 50–100d `gensim` W2V + `cuML` cosine kNN (nearly universal), 1000-dim `ProNE` graph embeddings (`#2`, `#8`), PyTorch `BPR-MF` (`0.665` recall@200, `#20`), and Sequence-Time-Aware SKNN (`#10`).

### Recall Ceiling & Scale Engineering
- **Recall Scaling (`#15` → `#12`):** Weighted recall ceiling rose from `0.575` (`20` cands) → `0.624` (`50`) → `0.657` (`100`) → `0.685` (`150`, `#13`) → `0.695` (`~234`, `#12`); `#1` pooled up to `~1,200` candidates shared across objectives.
- **Systems Stack:** `RAPIDS cuDF/cuML` (`int32`/`float32` bitwidths — 20–30 co-visit matrices built in ~2 min on V100), **Polars** (`40×` faster joins than pandas, `#2`), `Numba`, `8–32` chunked parquets, **Treelite** (`2×` faster LGBM inference, `#2`/`#5`), `cuML ForestInference` (`#20`), and **CatBoost-GPU** (`30×` faster than LGBM-CPU, `#2`).

---

## 4. Stage 2 — Feature Engineering & Multi-Objective Reranking

- **Feature Budget:** Top teams engineered 150–744 features (`#1` pruned ~200 → **top 100** by gain; `#2` pruned 5,000 combos → **400–500**; `#7` used **344**).
- **Highest-Leverage Feature Families:**
  1. **Session × Item & Co-Visit Aggregations:** Scores/ranks/normalized counts from 20–59 co-visit views aggregated (`sum/max/min/mean/std/last`) against last item, last 10–40 items, and carts/orders only; **Sequence Difference (`|idx_a - idx_b|`)** (`#2`); revisit counts by type, elapsed `Δts` & step distance since last interaction; Stage-1 candidate score/rank.
  2. **Item (`aid`) Popularity & Conversion:** Multi-window counts/ranks (1–7d, 1–4w, decayed), conversion funnels (`click→cart`, `cart→order`, `click→order`), revisit ratios, hourly promo spikes (`#15`). Dropping `aid` features in `#1` cost **`-0.00351 CV`** (whereas dropping session-only stats cost only `-0.000185 CV`).
  3. **Embedding Similarities:** Cosine/dot stats (`last/max/min/mean/std`) from W2V, ProNE, ALS/BPR/MF between candidate and recent session items.
- **Reranking Objectives & Models:**
  - **`CatBoost PairLogitPairwise` Dominated `YetiRank` & `binary`:** Switching from LGBM `binary` to `CatBoost PairLogitPairwise` boosted `#2` (`senkin13`) by `+0.0007` orders / `+0.0020` carts / `+0.0012` clicks; `#3` (`Chris Deotte`) confirmed `+0.0006 LB` over XGB ranker while **`YetiRank` performed poorly**.
  - **Single Unified Multi-Objective Ranker (`#5`):** Training **one** LGBM grouped by `session` with graded relevance **`Orders=6, Carts=3, Clicks=1`** beat 3 separate models by **`+0.001` to `+0.002`**.
  - **Negative Downsampling:** `5%` clicks / `25%` carts / `40%` orders (`#1`); `5%`–`20%` random negative downsampling (`#5`, `#12`, `#15`, `#20`) after dropping zero-target sessions.

---

## 5. Ensembling, Blending & Anti-Patterns

- **Calibrated GBDT + GNN Fusion (`#7`):** Solved per-`session_type` shift `C_hat` aligning `sigmoid(LambdaRank_score + C_hat)` with GNN probabilities, blending `0.525 LGBM (0.60008) + 0.475 GNN (0.59874) → 0.60313` Private LB (**`+0.00305` true cross-paradigm jump!**).
- **Rank & Index-Score Blending:** `#2` and `#3` blended teammates via summed inverse ranks (`0.601–0.6029 → 0.60382–0.60446`); `#4` combined raw scores with `1.00..0.05` rank-step index weights; `#1` averaged raw scores of 9 LGBMRankers (`0.604 → 0.605`); `#11` used cross-objective blending (`0.7*orders + 0.2*carts + 0.1*clicks`, `+0.0004` orders CV).
- **Verified Anti-Patterns:**
  1. **Cross-Target OOF Stacking Leakage:** Feeding `clicks`/`carts` OOF probabilities into `orders` when filtered to positive sessions produced fake `+0.001 CV` gains that **degraded LB** (`#10`, `#13`, `#15`, `#20`).
  2. **Full-Catalog Dense Softmax over `1.85M` Items:** Hits VRAM/sparsity collapse (`#3`, `#7`); always use subgraph extraction or negative sampling.
  3. **2nd-Stage Meta-Stacking & Pseudo-Labeling:** Overfit CV with zero LB gain (`#8`, `#12`, `#13`).
  4. **Unfilled `<20` Cold Sessions:** `3,327` test sessions had `<20` candidates (`#12`) — always backfill via unconstrained full-history co-visitation + recent hot items.

---

## 6. Verified Solution Sources

- **1st Place (`mrkmakr`):** `https://www.kaggle.com/competitions/otto-recommender-system/writeups/mrkmakr-1st-place-solution`
- **2nd Place (`SOS3`):** `https://www.kaggle.com/competitions/otto-recommender-system/writeups/sos3-2nd-place-solution-onodera-part`
- **3rd Place (`G & B & D & T`):** `https://www.kaggle.com/competitions/otto-recommender-system/writeups/g-b-d-t-3rd-place-using-only-rules-achieves-lb-0-5`
- **4th Place (`CWKTJ`):** `https://www.kaggle.com/competitions/otto-recommender-system/writeups/cwktj-5th-place-yet-solution-carno-2u-jiahong-s-pa`
- **5th Place (`NikhilMishra`):** `https://www.kaggle.com/competitions/otto-recommender-system/writeups/nikhilmishra-6th-place-solution`
- **6th Place (`THLUO`):** `https://www.kaggle.com/competitions/otto-recommender-system/writeups/thluo-6th-place-solution-single-model-lb-0-603`
- **7th Place (`Jack & toshi_k`):** `https://www.kaggle.com/competitions/otto-recommender-system/writeups/jack-toshi-k-7th-place-solution`
