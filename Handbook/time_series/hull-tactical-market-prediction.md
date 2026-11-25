# Hull Tactical — Market Prediction — Top Solutions & Live-Market Post-Mortem

> **Competition**: [Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction) (Sep 16 – Dec 15, 2025 train / Jun 26, 2026 live forecast | 3,293 Final Teams [3,677 raw] | 5,612 Submissions | \$100,000 Prize Pool)  
> **Task**: Output a daily S&P 500 allocation weight $w_t \in [0, 2]$ ($0$ = cash/risk-free, $1$ = 100% S&P 500, $2$ = 2× leverage) via a 1-row-at-a-time streaming API revealing 1-day lagged returns  
> **Evaluation Metric**: **Volatility- & Return-Penalized Adjusted Sharpe** — $\frac{S}{\text{vol\_penalty} \cdot \text{return\_penalty}}$, where $\text{vol\_penalty} = 1 + \max(0, \sigma_{\text{strat}}/\sigma_{\text{mkt}} - 1.2)$ (**1.2× vol cliff**) and $\text{return\_penalty} = 1 + \text{return\_gap}^2 / 100$ (quadratic penalty below market excess return)  
> **Dataset**: `train.csv` (`9,021` trading days, `~94` anonymized `M/E/I/P/V/S/D` features; early pre-2000 NaNs) | Live Private Test: `~126–130` post-deadline trading days (`<= 9h` notebook; Public LB overlapped `train[8810:8989]`!)

---

## Official Private (Live) Leaderboard Standings & Solution Index

Because Public LB overlapped `train[8810:8989]` (spawning meaningless `2.6–17.5` leaked Public scores) and top live winners kept proprietary alpha private (**only #4 `YannFb` among Top 10** published an official writeup, alongside Silver/Bronze post-mortems `#61, #100, #163, #225`), this competition stands as a textbook study in **risk engineering over return forecasting, short-horizon Sharpe variance, and strict leakage hygiene**:

| Private Rank | Team | Private Adjusted Sharpe | Core Winning / Documented Paradigm | Writeup / Code |
| :---: | :--- | :---: | :--- | :--- |
| 🥇 **#1** | **`lunalu`** | **3.845** | *(Proprietary — no public writeup)* | — |
| 🥈 **#2** | **`FastScramblers`** | **3.842** | *(Proprietary — no public writeup)* | — |
| 🥉 **#3** | **`wynzaro`** | **3.533** | *(Proprietary — no public writeup)* | — |
| 🏅 **#4** | **`YannFb`** | **3.347** | **Zero ML, zero CV, zero HPO**: externally validated deterministic **short-term mean-reversion** (~16 trades/yr) + **inverse-volatility weighting (`w_i ∝ σ_i⁻¹`, skipping noisy covariance `Σ`)** + **long-window periodic volatility targeting** (`14.3%` vol vs `19.2%` B&H, Sharpe `0.66` vs `0.33` historically) | [Writeup](https://www.kaggle.com/competitions/hull-tactical-market-prediction/writeups/4th-place-technical-model-no-learning-short-te) |
| 🥈 **#61** | **`Nikitas Rafael Nikitas`** | **2.565** *(Pub `0.969`)* | **Single LightGBM** (4-fold `TimeSeriesSplit`, 14 base feats × 16 lags/rollings = `224` feats) tuned on **mean Spearman rank correlation** + **binary `{0, 1}` unlevered allocation** (beat multi-GBDT blend & continuous `[0, 2]` leverage!) | [Writeup](https://www.kaggle.com/competitions/hull-tactical-market-prediction/writeups/single-model-simple-decision-61st-place-silver) · [NB](https://www.kaggle.com/code/rafanikitas/hull-eda-training-pipeline) |
| 🥈 **#100** | **`Tingkai Kung`** | *(Silver)* | **800-row (~3yr) sliding online refit** on **de-meaned MAD-winsorized `market_forward_excess_returns`** (`0.30 ElasticNet + 0.35 XGB + 0.35 LGBM`, fixed weights beat Ridge meta-learner) + **1.2× vol-cliff sizer** (`clip(s / (1.2·σ_hat), 0, 2)` w/ GARCH-blended `V1` vol + `75/25` EWM smoothing) | [Writeup](https://www.kaggle.com/competitions/hull-tactical-market-prediction/writeups/100th-place-ensemble-prediction-naive-allocation) |
| 🥉 **#163** | **`Less` et al.** | **2.160** | Dropped early NaNs & held out `train[-180:]`; discovered **`bfill` imputation leaked future regimes (`-0.6` score!)**; smooth **`tanh` position sizing** beat sigmoid (`2.19` train → `2.16` live) | [Writeup](https://www.kaggle.com/competitions/hull-tactical-market-prediction/writeups/163th-place-solution-sharpe-ratio-2-16) |
| 🥉 **#225** | **`Shih Hua Chou`** | *(Bronze)* | Conservatively regularized single LGBM on `market_forward_excess_returns` + trailing-vol `tanh` regime dampener | [Writeup](https://www.kaggle.com/competitions/hull-tactical-market-prediction/writeups/225th-bronze-1st-try-simple-single-lgbm-and-vola) |
| — | **`AmbrosM` / `morodertobias`** | `0.520` *(leak-safe baseline)* | Forensics proving Public `>3.0` models had negative CV $R^2$, `D1 == D2` duplicate, `MOM*` absent, & `ElasticNetCV` leak-safe baseline vs `0.464` constant `w=1.0` | [EDA NB](https://www.kaggle.com/code/ambrosm/htmp-eda-which-makes-sense) · [Baseline NB](https://www.kaggle.com/code/morodertobias/hull-leak-safe-baseline) |

---

## 1. Universal Financial Forecasting & Allocation Breakthroughs

1. **Portfolio Risk Engineering Dominates Noisy Return Forecasting (`#4`, `#61`, `#100`, `#225`)**:
   - Daily equity index OOS $R^2$ sits around `0.001–0.013` (Hull's own *Micro Alphas* paper: `1.33%` OOS $R^2$, 35-year Sharpe `~0.55`).
   - Rather than chasing higher return-prediction MSE, winning portfolios structured the allocation map directly around the evaluation geometry:
     - **Inverse-volatility stabilization** ($w_i = \sigma_i^{-1} / \sum \sigma_j^{-1}$) without unstable sample covariance inversion (`#4`).
     - **Metric-anchored volatility targeting**: `clip(s / (1.2 · σ_hat), 0, 2)` blending realized 20d return variance with implied/level vol (`V1`, `#100`) or long-window periodic vol targeting (`#4`) to avoid the linear `1.2×` volatility cliff.
     - **Bounded / Unlevered Mapping (`{0, 1}` `#61` or smooth `tanh` `#163`, `#225`)**: Avoiding aggressive `[0, 2]` leverage prevented forecast noise from blowing up realized strategy variance.
2. **Target Hygiene & Rank-Aligned Objectives (`#61`, `#100`)**:
   - Predicting **`market_forward_excess_returns`** (5-year rolling de-meaned and MAD-winsorized) gave the single largest stability gain over raw `forward_returns` (`#100`).
   - Tuning hyperparameters for **mean fold Spearman rank correlation** (`#61`) or convex **mean + variance + CVaR** allocation surrogates (`#435` Sherlock Banks) beat raw RMSE tuning because relative timing signal matters more than outlier return magnitudes.
3. **Sliding-Window Online Refitting & Ultra-Conservative Regularization (`#61`, `#100`)**:
   - Discarding pre-2000 NaN-heavy regimes and continuously refitting a small fixed-weight ensemble (`ElasticNet + XGB + LGBM`) on a rolling **800-day (~3-year) window** via 1-day lagged API targets (`#100`), or using a single heavily regularized model (`#61`, `#225`), survived the 6-month live window where complex meta-learners collapsed.
4. **Recognizing Short-Horizon Evaluation Variance (`discussion/608135`, `608262`)**:
   - Monte Carlo audits proved that over `126–180` trading days, **`36.4%` of pure random allocations achieve Sharpe `>= 1.0`** and **`9.35%` achieve `>= 2.0`**. Never judge model iterations on a single 6-month holdout — validate across multi-decade expanding / walk-forward windows (`125d–252d` folds) and inspect OOS $R^2$ / Information Coefficient alongside Sharpe.

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Trusting Overlapped Public LB (`train[8810:8989]`) or Fitting Thresholds on Recent Windows** | Public `2.6` → **Private #2071** (`Andrey Chankin`); `1.765 → 0.453` (`ahsuna123`) | Community, `AmbrosM` | Public LB overlapped `train.csv` (oracle hit `17.396`); models with `>3.0` Public had **negative CV $R^2$** |
| **`bfill` Missing-Value Imputation on Financial Time Series** | Degraded live score by **`~0.6` (`#163`)** | #163 | Back-filling leaks future regime distributions into past timestamps (strictly use forward-fill or drop early NaN eras) |
| **Massive Feature Generation (6,800 Ratios) + Tree Importance Selection + 6-Bin Thresholding** | Inflated Public `1.07 → 2.60` → collapsed to **#2071 Private** | `discussion/663043` | Multiple-testing bias on low-SNR financial series memorizes noise regimes |
| **Meta-Learner Stacking (Ridge) & Heavy Multi-GBDT Blending on Small Folds** | Overfit ~135-row CV folds (`#100`); `LGBM+XGB+CB` scored worse than single LGBM (`#61`) | #61, #100 | When true signal SNR is tiny, extra meta-model / ensemble variance degrades out-of-sample Sharpe |
| **Dynamic `2×` Leverage Sizing & Training-Only Rolling Features Without Online Ring Buffer** | Amplified forecast errors (`#61`); broke streaming inference (`#100`) | #61, #100 | Streaming 1-row API requires an explicit stateful ring buffer; `2×` leverage triggers the `1.2×` volatility cliff |
