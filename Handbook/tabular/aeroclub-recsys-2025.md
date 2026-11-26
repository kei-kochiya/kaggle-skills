# FlightRank 2025: Aeroclub RecSys Cup 🏆

> **Track:** Tabular & Learning-to-Rank RecSys (Community Prediction) · **Scale:** 578 teams · \$10,000 · Closed Aug 17, 2025  
> **Core Lesson:** In real-world session/query learning-to-rank with unseen entities and temporal shift, **label-free transductive count encoding**, **strictly causal numeric/rank user-company history** (never historical categorical IDs), **full-group features computed *before* negative downsampling (`cap 50–64 rows`)**, and **multi-loss GBDT averaging** beat Neural/Transformer rankers and 41-model Optuna tuning with **zero Top-5 shakeup**.

---

## 1. Competition DNA & Leaderboard

- **Task:** Rank corporate flight search options per session (`ranker_id`: `105,539` train groups, `45,231` test groups, ~70/30 split across `229` days from May 17 to Dec 31, 2024; groups span from a few options up to `3,500+` on major routes). Exactly one row per `ranker_id` has `selected = 1`; output must be an exact tie-free permutation `1..N`.
- **Metric (`HitRate@3` on groups with `> 10` options):**
  $$\text{HitRate@3} = \frac{1}{|Q_{>10}|} \sum_{i \in Q_{>10}} \mathbb{1}(\text{rank}_i \le 3)$$
  *(Critical gotcha: groups with `≤ 10` rows remain in train/test for signal, but are excluded from scoring — unfiltered `HitRate@3` reads `~0.921` vs filtered `~0.508`!)*
- **Zero Top-5 Shakeup (Public 29% vs Private 71%):**

| Private Rank | Shift | Team | Private LB | Public LB (Rank) | Best Single Model PB | Core Breakthroughs |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| 🥇 **1st** | `—` | **gezi** | **0.54466** | 0.54062 (#1) | **0.54156** (late `0.54942`) | Transductive `train+test` count encoding + UTC/Haversine + dual-level `flight_hash` ranks + causal numeric history + 5-loss equal-weight blend (`0.55302` late) |
| 🥈 **2nd** | `—` | **Phaedrus** | **0.54175** | 0.53548 (#2) | 0.53807 | 300+ segment-tier & OOF behavioral feats + XGB `max_pairs=8` + per-group Softmax + 25-bracket Optuna (13 models) |
| 🥉 **3rd** | `—` | **AF** | **0.53718** | 0.53548 (#3) | 0.53265 | 50%/50% temporal history split (~200 feats) + CatBoost `QueryCrossEntropy`/`PairLogitPairwise` + `64`-row group cap |
| 🏅 **4th** | `—` | **mango789 & Ionut Visan** | **0.53672** | 0.53438 (#4) | 0.53552 | `flight_hash` duplicate-offer grouping + JSON display labels + RRF/confidence + PCA/Spearman uniqueness blend |
| 🏅 **5th** | `—` | **Taos David** | **0.53311** | 0.52970 (#5) | — | Insights Award winner |
| 🏅 **6th** | `+2` | **XS 330** | **0.53261** | 0.52529 (#8) | 0.52823 | Single LGBM `lambdarank` (`0.52823` PB) |
| 🏅 **8th** | `-2` | **Mikhail Golubchik** | **0.53079** | 0.52814 (#6) | 0.53056 | Causal numeric/rank `uid`/`companyID` history + `1%` train cat NaN masking + `50`-row post-FE cap |

---

## 2. EDA & Data Forensics

1. **Duplicate Physical Flights (`flight_hash`):** Raw JSON inspection (`4.66 GB` archive of `150,770` sessions) showed one physical itinerary appears multiple times in parquet under different promotional/tariff pricings, only one of which can be `selected = 1`. Grouping via `flight_hash` over departure/arrival timestamps + flight numbers unlocked both strong intra-flight features (`#1`, `#4`) and deduplication reranking (`#4`, `#10`).
2. **Severe `bySelf` Train/Test Shift:** `100%` of training rows had `bySelf == True`, whereas test contained `~200,000` rows with `bySelf == False` (travel agents booking on behalf of travelers).
3. **Unseen Entities & Principal-Agent Dynamics:** Only `~50%` of test `profileId` users appear in train, plus many unseen categorical levels (`#8` randomly masked `1%` of train categoricals to `-1` to regularize unseen categories). In corporate travel, employees earn personal frequent-flyer miles while companies pay the fare—creating systematic preference for loyalty carriers over cheapest/shortest options.
4. **Schema & Ordering Traps:** Cabin mapping was actually `1=economy, 2=comfort, 3=business, 4=first` (doc error fixed by host). Raw parquet row ordering carried non-deterministic supplier bias—positional features were unstable and harmful.

---

## 3. Feature Engineering Playbook

1. **Label-Free Transductive Count Encoding (`+0.01464` PB! — `#1`):** Replaced *all* categorical features with frequency counts computed over **`train + test` combined** (passing zero raw categorical indices to GBDTs). Ablation: removing `test` from count encoding dropped Valid `0.50475 → 0.47870` and Private LB **`0.54156 → 0.52692`**.
2. **UTC Duration, Cyclical Hour & Geo Efficiency (`#1`, `#3`):** Converted local times to UTC (`timezonefinder`) for exact durations, cyclical `sin/cos` hour encodings (hours, not minutes), Haversine airport distances (`airportsdata`), and `direct_price_per_km` / `price_per_minute`.
3. **Dual-Level Intra-Group Ranks & Segment Tiers (`#1`, `#2`, `#4`):**
   - Computed Polars `rank(method='average')` over **both** `ranker_id` **and** `(ranker_id, flight_hash)` across 9 core features (price, duration, lead time, duration ratio, segments, cabin, baggage count/weight, price/km).
   - **Segment Intelligence (`#2`, top importances):** `segment_tier = total_segments - min(total_segments).over('ranker_id') + 1`, `position_within_segment_tier` (price rank within same stop count), `price_premium_vs_min_segments`, `is_only_direct_option`, and normalized percentiles `(rank - 1) / (count - 1) * 100`.
4. **Label-Free Exposure Stats + Strictly Causal Selected-History Numeric Averages (`+0.00786` PB — `#1`, `#8`, `#11`):**
   - Label-free `mean/min/max/std/median/zscore/minmax/ratio` of the 9 rank columns per `profileId` and `companyID` over `train + test`.
   - **Causal Selected-History (`make_history_avg`):** Computed strictly on **earlier-in-time `selected = 1`** bookings per user/company over **numeric & rank attributes only** (price rank, duration rank, hour, cabin, baggage, price/km, fare rules). Raised `#1`'s single-model PB **`0.54156 → 0.54942`** (`#11` also gained `+0.01–0.02` via causal mean/median/IQR quantiles). Crucially, `#8` proved historical *categorical* preferences (past airline/aircraft IDs) overfit heavily—only numeric/rank history generalized.

---

## 4. Models, Objectives & CV Strategy

- **Temporal Holdout Crushed GroupKFold:** Seasonal ticket drift caused 10-fold GroupKFold to massively inflate CV (`0.641–0.662` GKF vs `0.538–0.542` PB in `#2`; `~0.60` GKF vs `0.49–0.51` LB across forums). `#1` (days `1–103` train / `104–166` valid / `167–229` test), `#4` (last 10%), `#8` (last month: `0.535 CV vs 0.531 PB`), and `#3` (50% historical reference / 50% train) all tracked LB reliably.
- **GBDT Ranking Losses Dominated Neural Rankers:**
  - **XGBoost (`#1` identical features):** `rank:ndcg` (`0.50475` valid, `0.54156` PB) > `rank:pairwise` (`0.50398`) > `rank:map` (`0.50362`) > `binary` (`0.49897`); `#2` swept `max_pairs ∈ {8..128}` and won single-model with `max_pairs=8` (`0.53807` PB).
  - **LightGBM:** `lambdarank` hit `0.52823` PB single (`#6`); `xendcg` performed **much worse** (`#1`).
  - **CatBoostRanker:** Core of `#3` (`QueryCrossEntropy(alpha=1.0)`, `PairLogitPairwise`, `YetiRankPairwise` → `0.53265` single / `0.53718` PB ensemble).
  - **Neural/Transformer Rankers Failed:** `#4` tested Wide&Deep, MLP (`xRankNet`), SENet, bilinear interactions, self-attention, and Transformer encoders — topping out at **`0.480–0.495`** vs XGBoost `0.53552`; `#10`'s `Bert2rec` also failed badly.
- **Post-FE Training-Time Negative Downsampling (`#3`, `#8`):** Capping training sessions at **`50–64` rows** (keeping `selected=1` + sampled negatives) sped up training and boosted ranking sharpness—**strictly *after* computing full-group features**.

---

## 5. Ensembling, Post-Processing & Anti-Patterns

- **Equal-Weight Multi-Loss Averaging (`+0.00569` PB — `#1`):** Averaging 5 loss-diverse models (`rank:ndcg`, `rank:pairwise`, `rank:map`, binary XGB, `lambdarank` LGBM) on identical features jumped `0.54156 → 0.54725` (`0.55302` late with causal history).
- **Group-Wise Score Normalization & Bracket Ensembling (`#2`, `#3`, `#10`):** Per-group Softmax (`#2`) or per-model Min-Max scaling (`#10`, beat median/rank averaging) aligned heterogeneous LTR scales before blending; `#2` tuned 13-model Optuna weights across 25 session-size brackets (`0.53807 → 0.54175`).
- **Deduplication Post-Processing (`#4`, `#10`):** Penalizing near-duplicate `flight_hash` entries gained `+0.001–0.002` on weaker baselines, but became unnecessary once `(ranker_id, flight_hash)` features were given directly to strong GBDTs (`#1`).
- **Verified Anti-Patterns:**
  1. **Optuna Blending Over Too Many Models (`#1`):** Tuning Optuna weights over 41 models improved local CV but **dropped Private LB** vs equal-weight top-5 (`HitRate@3` top-3 discontinuities overfit easily).
  2. **Non-Causal Target Stats & Historical Categorical Preferences:** Future `selected=1` leakage and past user-airline ID features caused heavy train overfitting (`#1`, `#8`).
  3. **Downsampling Groups Before FE:** Corrupts intra-group ranks/percentiles (`#8`).
  4. **Pseudo-Labeling, Group Sample Weights, `xendcg`, & Positional Row-Order Features:** Consistently failed across `#1`, `#2`, `#4`.

---

## 6. Verified Solution Sources

- **1st Place (`gezi`):** `https://www.kaggle.com/competitions/aeroclub-recsys-2025/writeups/1st-solution-using-count-based-features`
- **2nd Place (`Phaedrus`):** `https://www.kaggle.com/competitions/aeroclub-recsys-2025/writeups/flightrank-2025-2nd-place-solution`
- **3rd Place (`AF`):** `https://www.kaggle.com/competitions/aeroclub-recsys-2025/writeups/3rd-place-solution`
- **4th Place (`mango789 & Ionut Visan`):** `https://www.kaggle.com/competitions/aeroclub-recsys-2025/writeups/4th-solution-xgboost-model`
- **8th Place (`Mikhail Golubchik`):** `https://www.kaggle.com/competitions/aeroclub-recsys-2025/writeups/8-th-place-solution`
