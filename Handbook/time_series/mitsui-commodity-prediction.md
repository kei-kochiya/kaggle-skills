# MITSUI&CO. Commodity Prediction Challenge — Top Solutions & Live-Forecast Post-Mortem

> **Competition**: [MITSUI&CO. Commodity Prediction Challenge](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge) (Jul 24 – Oct 6, 2025 train / Jan 16, 2026 live forecast | 1,126 Final Teams [1,711 raw] | 1,730 Submissions | \$100,000 Prize Pool)  
> **Task**: Forecast **424** daily multi-horizon (`lag ∈ {1, 2, 3, 4}`) log-return & pairwise cross-asset spread targets across **LME, JPX, US equities, and FX** via a 1-day-at-a-time streaming API with causally maturing `label_lags_1..4`  
> **Evaluation Metric**: **Sharpe Ratio of Daily Cross-Sectional Spearman Rank Correlations** — $\frac{\text{mean}_t(\rho_t)}{\text{std}_t(\rho_t;\, \text{ddof}=0)}$ (computed per date across non-null targets)  
> **Dataset**: `train.csv` (`1,917` trading days: `2018-01-02` to `2025-05-23`, `25.04 MB`, `target_pairs.csv` anchored on **6 hub commodities**) | Live Private Test: `~73–90` post-deadline trading days (`<= 9h` notebook; Public LB overlapped last 90 rows of `train.csv`)

---

## Official Private (Live) Leaderboard Standings & Solution Index

Because Public LB overlapped the last 90 rows of `train.csv` and a 500k Monte Carlo audit showed random daily predictions yield `std = 0.107` (range `[-0.49, +0.55]`), heavy `2.0`-CV backtest pipelines collapsed live while **simple short-window neural nets with composite ranking loss, recursive feature forecasting + lag blending, and covariance-shrunk Kelly rank ordering** claimed the top published spots:

| Private Rank | Team | Private Sharpe-Spearman | Core Winning / Documented Paradigm | Writeup / Code |
| :---: | :--- | :---: | :--- | :--- |
| 🥇 **#1** | **`anonemaus`** | **0.63834** | *(No full writeup; noted concatenating streamed test onto `train_df[1728..1826]` sorted by `date_id`)* | [Notes](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/discussion/687186) |
| 🏅 **#5** | **`ZLF`** (`Zhu Linfeng`) | **0.53231** | Simple mean of **4-day window RNN (`0.509` solo!)** + **single-day MLP** on raw features (`-1` fill, internal `LayerNorm`) trained with **composite MSE + ranking loss** (`~3–4 min` on T4; longer windows hurt!) | [Writeup](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/zlf-solution-of-the-mitsui-commodity-predict) |
| 🏅 **#6** | **`Poidenko Artem`** (`Artem777`) | **0.51444** | **Underlying-feature forecasting**: 50d Many-to-One LSTM (`424→256→424`) recursively forecasting raw prices 5 days ahead → algebraic spread formula + **Label Lag Correction (`α=0.7` blend w/ matured API `label_lags`)** (`<1 min` train) | [Writeup](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/artem777-in-mitsui-6th-Place) |
| 🏅 **#8** | **`honkai01031227`** | **0.49066** | Single **Tabular Transformer Encoder** jointly predicting 424 targets from `587` raw + `424` `lag-4` targets (causal `ffill`, `StandardScaler`, masked MSE, `AdamW` + cosine + `bf16`, `12m`) | [Writeup](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/transformer-based-solution) |
| 🏅 **#10** | **`miguel perez`** | **0.47974** | **Metric-Exact Regularized Kelly Rank Portfolio**: derived optimal static ordering $\Sigma_{\text{rank}}^{-1} \mu_{\text{rank}}$ and **shrunk the rank covariance matrix** toward mean cross-sectional ranks (validated on 90d windows, `~0.40` CV → `0.4797` Priv!) | [Writeup](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/discussion/668589) |
| 🏅 **#15** | **`Lonnie`** | **0.44582** | **Online retraining every 7 days** on matured `label_lags` buffer; Attention-DNN + Res-DNN + AutoEncoder with **`0.2·MSE + 0.8·(1 - Spearman)`**, `CombinatorialPurgedGroupKFold`, & `0.1/0.9` perf-recency blend | [Writeup](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/mitsui-and-co-commodity-prediction-challenge-15th) · [NB](https://www.kaggle.com/code/lonnieqin/mitsui-co-commodity-prediction-15th-place) |
| 🥈 **#16** | **`EVVSK`** | **0.44571** | Long-format (`×424` with `target_id`) DNN+RF+XGB+CB + recent lag blend + **1D conv `[0.25, 0.5, 0.25]` cross-target smoothing** | [Writeup](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/evvsk-mitsui-and-co-commodity-prediction-challenge) |
| 🥈 **#25–#26** | **`Halil Akça` / `\_\_kei\_\_`** | **0.4319 / 0.4279** | Market-grouped (`LME/JPX/US/FX`) LGBM+CB+Ridge + lag blending + EMA (`#25`); stability-selected rank stats + **UMAP+KMeans 2-cluster inversely-correlated target IDs** (`#26`) | [#25](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/25th-place-silver-mitsui-commodity-prediction) · [#26 GitHub](https://github.com/kono-k/Kaggle-MITSUICommodityPredictionChallenge) |
| 🥉 **#89** | **`SangSong`** | **0.387** | Pure static baseline: **`train_labels.rank(axis=1).mean()`** every day (proving cross-sectional baseline structure!) | [Writeup](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/89th-place-solution) |

---

## 1. Universal Multi-Target Financial & Cross-Sectional Rank Breakthroughs

1. **Exploiting Cross-Sectional Rank Sharpe Geometry (`#10`, `#89`, `#5`, `#15`)**:
   - Because the metric evaluates `mean(ρ_t) / std(ρ_t)` across 424 coupled spreads anchored on **6 hub commodities** (`Gold`, `Platinum`, `LME AH/CA/PB/ZS`), **consistent cross-sectional ordering** dominates noisy per-target magnitude prediction.
   - `#10` showed the metric-optimal static prediction is the **Kelly rank vector** $\Sigma_{\text{rank}}^{-1}\mu_{\text{rank}}$, and **shrinking the rank covariance matrix $\Sigma_{\text{rank}}$** toward the mean cross-sectional rank vector generalized cleanly (`~0.40` CV → `0.4797` Live) with zero overfitting.
   - For neural models, optimizing a **composite MSE + differentiable Spearman/ranking loss** (`0.2·MSE + 0.8·(1 - Spearman)`, `#5`, `#15`) and blending conservative sequential models (4-day RNN) with responsive single-day MLPs (`#5`: `0.509 → 0.532`) reduced daily $\text{std}(\rho_t)$ directly.
2. **Forecasting Underlying Assets vs. Noisy Spreads + Lag Blending (`#6`, `#8`, `#15`, `#25`)**:
   - Predicting noisy log-return differences directly suffers from double noise amplification. Recursively forecasting underlying asset states 5 days ahead and computing exact algebraic spreads from `target_pairs.csv` (`#6`), combined with **blending matured API `label_lags` (`α = 0.7`, `#6`)** or **periodic 7-day online retraining (`#15`)**, tracked regime drift without unstable daily over-fitting.
3. **Multi-Exchange Calendar & Delisting Resilience (`discussion/593820`, `606650`)**:
   - Global multi-exchange series (`LME`, `JPX`, `US`, `FX`) drop dual-holiday days (`2022-09-19` Queen's funeral) and inject exchange-specific `NaN`s on partial holidays — strictly use **causal forward-fill (`ffill`)** and **masked losses over finite targets (`#8`, `#37` `MultiRMSEWithMissingValues`)**.
   - Mid-horizon ticker delistings (`US_Stock_X`) turn target columns `NaN` at test time while still requiring finite submitted predictions (`assert` zero `NaN`s in output!).

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Chasing High Backtest Sharpe (`~2.0` CV) on Last 90–200 Train Days** | Collapsed to `0.14` or negative live; overlapped Public LB (`train[-90:]`) was pure leak | Community (`639895`) | Low-SNR `~90`-day Sharpe has massive sampling variance (random noise `std = 0.107`, range `[-0.49, +0.55]`) |
| **Long RNN Lookback Windows (`> 4` Days) & Over-Engineered Pct-Change Features** | Degraded CV & live generalization (`#5`, `#37`) | #5, #37 | Longer lookbacks inject regime noise faster than signal in cross-market commodity spreads |
| **Over-Training Deep Trees (`> 100` CatBoost Iters) or Unverified Rerun Pipelines** | Overfit noise (`#37`); many notebooks silently errored on final extended live rerun (`668489`) | #15, #37, LB | Always enforce strict `NaN`/delisted-ticker guards, `len(pred) == 1` assertions, and conservative runtime headroom |
| **Dropping `2020–2021` Volatility Regime When Estimating Covariance** | Hurt OOS rank-covariance shrinkage (`#10`) | #10 | Full multi-regime history (`2018–2025`) is needed to estimate stable cross-asset correlation structure |
