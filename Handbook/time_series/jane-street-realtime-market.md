# Jane Street Real-Time Market Data Forecasting — Top Gold Solutions & Post-Mortem

> **Competition**: [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) (Oct 14, 2024 – Jan 13, 2025 train / Jul 13–14, 2025 live forecast | 2,781 Ranked Teams [3,757 raw] | 4,294 Submissions | \$120,000 Prize Pool)  
> **Task**: Real-time intraday return forecasting (`responder_6`) across `39` instruments (`symbol_id`) via a streaming API (`kaggle_evaluation.jane_street_inference_server`) that reveals the **prior day's ground-truth `responder_0..8` at `time_id == 0` of every new `date_id`**  
> **Evaluation Metric**: Sample-Weighted **Zero-Mean $R^2$** ($1 - \frac{\sum w_i (y_i - \hat{y}_i)^2}{\sum w_i y_i^2}$; predicting `0` gives `0.0`)  
> **Dataset**: `train.parquet` (`47.13M` rows, `12.3 GB`, `1,699` dates, `79` features, `9` responders) | Live Forecast Test: `~4.5M` unscored warmup rows + `~4.5M` live scored rows (`<= 9h` total, `<= 1 min` per `predict()` call)

---

## Official Top Gold Leaderboard Standings & Solution Index

In a grueling 6-month live forecast where static models decayed catastrophically (`0.0161 → 0.0011` across a 200-day gap) and Bronze churned `99.5%` on the first live rescore, **gold winners separated into a rock-solid 94–100% stable tier** by combining **cross-asset attention + temporal GRUs**, **multi-responder auxiliary targets**, and **disciplined daily online learning on lagged ground truth**:

| Private Rank | Team | Private (Live) $R^2$ | Reported Public LB | Core Winning Paradigm | Writeup / Code |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **`ms capital`** (`chenxin1991`) | **0.013890** | — | *(No public writeup; confirmed CatBoost `init_model` failed online)* | — |
| 🥈 **#2** | **`Patrick Yam`** (`wimwim`) | **0.013273** | **0.01303** | **17-seed `TransformerLayer2d`** (8 blocks: **cross-asset 8-head self-attention + temporal GRU + gated FFN**), multi-task all 9 responders (`w=[1,1,1,6,2,2,12,5,5]`), recency+968-step weighting; **online 3 steps/day `Adam(lr=1e-4, betas=(0.8, 0.95))`** (fresh online state, no replay!) + stacked `einsum` (2.5h total) | [Reconstructed Repo](https://github.com/cweill/jane-street-causal-forecasting) |
| 🥉 **#3** | **`shorturl.at/LKhAD`** (`yuanzhezhou`) | **0.013163** | — | *(No public writeup)* | — |
| 🏅 **#7** | **`leo`** (`calibrator`) | **0.010480** | — | Reported **daily optimizer reinitialization** outperformed persistent optimizer state by **`~+0.001`** | [Comment](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556541) |
| 🏅 **#8** | **`Evgeniia Grigoreva`** (`eivolkova`, sole Top-50 LB writeup!) | **0.010434** | **0.0112** (#6) | Trained on **`date_id >= 700`** + 16-feat cross-sectional/1000-step rolling stats (`+0.002` CV); 6× GRU (`3L-GRU` + `1L-GRU+Linear`) predicting **4 multi-horizon aux responders (`4d/8d/60d/120d`) → `Linear(4,1)`** (`+0.001`); **daily fresh `AdamW(lr=3e-4, clip=1.0)` on `responder_6` only** across all days incl. `is_scored=False` (`+0.0125` on 200d-gap CV!) | [Writeup](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/writeups/evgeniia-grigoreva-private-lb-8th-solution) · [GitHub](https://github.com/evgeniavolkova/kagglejanestreet) · [NB](https://www.kaggle.com/code/eivolkova/public-6th-place?scriptVersionId=217330222) |
| 🏅 **#9** | **`HAO LI`** | **0.010417** | `0.008+` (CB) | **Online block-refit CatBoost** (rolling 600 days → 5 × 120-day blocks subsampled 40–50%, bypassing broken `init_model`) + lag features + cross-symbol mixing before GRU (`0.0096 → 0.0106`) | [Thread](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556544) |
| 🏅 **#11** | **`Victor Shlepov`** | **0.009866** | — | Stateful `[1, T, 39, 79]` Gated-Fusion Encoder-Decoder + **short-memory Adam betas (2–5 day horizon)** (`+0.0030` OL gain); mapped progressive symbol emergence (`date_id 0..1063`) | [Posts](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556686) |
| 🏅 **#12** | **`Maaax`** (`M. Zawadzki`) | **0.009774** | `~0.0096` (#13) | Clean progression: Static GBDT (`0.0045`) → Online LGBM (`0.0058`) → MLP (`0.0064`) → **Cross-sectional `(date,time)` mean+zscore (`0.0089`, `+0.0025`!)** → diffs (`0.0092`) → 12 targets (`0.0094`) → OHE symbol (`0.0096`) | [Thread](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556701) |
| 🏅 **#15** | **`gold digger`** (`snehal`) | **0.009602** | (#17) | Symbol-attention + Time-attention Transformer + **2-tier OL**: daily/7d gradient updates (`+0.0015` CV) + **periodic 56-day refit every 8 days (`+0.0012` CV)** | [Thread](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556541) |

---

## 1. Universal Streaming Financial Time-Series Breakthroughs

1. **Daily Online Learning Protocol (The `+0.0065` to `+0.0125` Live Survival Engine)**:
   - On `#8`'s realistic **200-day-gap validation fold**, static GRU collapsed from `0.0161 → 0.0011`, whereas **1 daily gradient pass** over the newly arrived prior day's full sequence recovered it to **`0.0136–0.0162` (`+0.0125` $R^2$)**. Independent causal reproduction of `#2` confirmed `+0.00645` (`0.01433 → 0.02078`).
   - **Golden Rules for Online Neural Updates**:
     - **Never carry offline Adam momentum buffers into online deployment** (`#2`): either **reinitialize `AdamW` fresh every day** (`lr=3e-4, clip=1.0`, `#7`, `#8`) OR initialize a **separate online Adam with short-memory betas (`betas=(0.8, 0.95)`, 2–5 day memory, 3 steps/day, `#2`, `#11`)**.
     - **Always update on unscored warmup replay days (`is_scored == False`)** so weights track regime drift seamlessly into the live Private window (`#8`).
     - **No stale multi-day replay buffer** needed (`#2`, `#8`); for periodic broader adaptation, run a **2-tier update** (daily gradient step + periodic 56-day block refit every 8 days, `#15`).
     - **Online GBDTs (`#9`)**: When native `init_model` fails, replace old tree blocks by refitting rolling subsampled `120`-day blocks (`40–50%` rows).
2. **Breaking the Row-Wise Plateau: Cross-Sectional + Temporal Architectures (`#2`, `#8`, `#11`, `#12`, `#15`)**:
   - Row-wise GBDTs/MLPs hit a ceiling around `0.0045–0.0064` (`#12`). Computing **cross-sectional market means & z-scores per `(date_id, time_id)`** immediately jumped **`+0.0025`** (`0.0064 → 0.0089`), and **explicit Cross-Asset Self-Attention + Per-Symbol Temporal GRU (`TransformerLayer2d`, `#2`)** broke past `0.0104–0.0139`.
3. **Multi-Horizon Auxiliary Target Learning (`#2`, `#8`, `#12`, `#19`)**:
   - Reverse-engineering showed `responder_0..8` encoded shifted `4/20/120`-step SMAs across primary/secondary feeds (`#19`). Jointly predicting all 9 weighted responders (`w_6 = 12`, `#2`) or constructing intermediate `8`-step and `60`-step horizon sums combined via `nn.Linear(4, 1)` (`#8`) reliably added **`+0.001` CV/LB** — while updating online strictly on primary `responder_6` (`#8`) prevented noisy auxiliary drift.
4. **Regime-Aware Filtering, Gap-CV & Sub-Second Serving (`#2`, `#8`, `#11`)**:
   - Because early `date_id`s had missing symbols (`0..1063`) and shorter 849-step days (`< 677–700`), winners either **trained strictly on `date_id >= 700` (`#8`)** or applied **recency + full-day weighting (`(200+d)/(200+max_d) × 1.5`, `#2`)**.
   - Validating on a **200-day-gap holdout** (`#8`) and multi-vintage sliding 120-day blocks (`#9`) caught silent static decay before submission. Stacked `torch.einsum` inference ran 17 models in **2.5h** (`#2`); `#8` kept step latency at `0.06s` and daily online updates at `3.6s`.

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Static Models Without Online Updating (or Skipping `is_scored=False` Replay)** | Collapsed on 200d gap (`0.0161 → 0.0011`, `#8`); Bronze `99.5%` churn | #2, #8, #11, LB | Non-stationary financial alpha decays within weeks; continuous replay adaptation is mandatory |
| **Unmasked Bidirectional Temporal Attention Across `time_id`** | Fake `R² ~ 0.30` CV → collapsed in live streaming | #11 (`549746`) | Future intraday steps leak into rolling features when attention isn't strictly causal |
| **Reusing Offline Adam Momentum / Multi-Day Replay Buffer Online** | Degraded online adaptation (`#2`) | #2, #7, #8 | Stale offline curvature/momentum resists rapid market regime shifts |
| **CatBoost `init_model` Warm-Start & Intraday `t-20` Pseudo-Label Updates** | Both failed to beat fresh block refits / daily ground-truth updates | #1, #9, #19 | Tree structure locked by `init_model` can't re-split under drift; noisy intraday pseudo-targets poison gradients |
| **Hard Prediction Clipping & Unregularized Static Public Notebook Blends** | Clipping gave no gain (`#11`); static forks swung `>2,000` ranks | #11, LB | Zero-mean weighted $R^2$ rewards calibrated shrinkage, not post-hoc hard clipping |
