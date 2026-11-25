# CIBMTR — Equity in post-HCT Survival Predictions: Top Gold Solutions & Post-Mortem

> **Competition**: [CIBMTR - Equity in post-HCT Survival Predictions](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions) (Dec 2024 – Mar 6, 2025 | 3,325 Teams | 78,768 Submissions | \$50,000 Prize Pool)  
> **Task**: Right-censored post-transplant survival risk prediction (`efs` ∈ `{0, 1}`, `efs_time`) with demographic equity across 6 equal-sized `race_group` strata  
> **Evaluation Metric**: **Stratified Concordance Index**: $\text{Score} = \operatorname{mean}_g(\text{C-index}_g) - \operatorname{std}_g(\text{C-index}_g)$ across 6 racial groups  
> **Dataset**: `28,800` Train rows × `57` features (`35` categorical, `22` numeric — only `donor_age` & `age_at_hct` truly continuous) generated via `synthcity` **SurvivalGAN** | `~28.8k` Hidden Test rows (`25%` Public / `75%` Private; `<= 9h` code notebook)

---

## Official Top Gold Leaderboard Standings & Solution Index

The defining breakthrough shared by **all 6 Top-6 Gold Medalists** was abandoning naive 1D survival target heuristics (plain Kaplan-Meier / Nelson-Aalen) in favor of **Factorized Two-Task Survival Modeling** — separating $P(\text{efs})$ event classification from conditional $\mathbb{E}[\text{rank}(\text{efs\_time}) \mid \text{efs}=1]$ regression — combined with **multi-seed / high-fold CV** against heavy 25% Public LB variance:

| Private Rank | Δ Rank | Team / Competitor | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | 0 | **`minerppdy`** | **0.69806** (#1) | **0.70127** | **9 Classifiers (`P(efs=0)`) × 3 Regressors (`efs_time` rank w/ `efs` as input feature, `0.6:0.4` weights, set `efs=1` at test)** → 27 non-linear power merges + Optuna `[0.1, 1.0]` bounded blend; 10-fold SKF **no early stop** | [Writeup](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/writeups/minerppdy-1st-place-solution-2-targets-and-ensembl) · [Simplified NB (0.7003)](https://www.kaggle.com/code/minerppdy/1st-place-simplified-private-lb-0-700) |
| 🥈 **#2** | 0 | **`agalar`** (`Anil`, `Gunes`, `Ahmet`, `Tarik`, `Davut`) | **0.69726** (#2) | **0.70038** | Reverse-engineered SurvivalGAN `TimeRegressor`: `RealMLP/HistGBM/CB` (`efs`) × `XGB/HistGBM` (`efs_time` w/ `efs` input) + **differentiable `tanh` pairwise C-index NN** rank-blended | [Writeup](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/writeups/agalar-2nd-place-solution) · [Infer NB](https://www.kaggle.com/code/karakasatarik/2nd-place-solution-inference) |
| 🥉 **#3** | +1 | **`SurvivalHorror`** (`Churkin`, `Ukrainskiy` et al.) | 0.69692 *(#4 0.69712)* | **0.69937** | **4-fold × 20–100 seed CV**; race-stratified `[0,1]` vs `[1.345,1.355]` target trained w/ **BCE loss** + outlier denoising (`-1k` train rows) + **MLP-ODST + TabM solo (`0.69694` Priv!)** + GBDTs | [Writeup](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/writeups/survivalhorror-3rd-place-solution) · [Code](https://www.kaggle.com/code/churkinnikita/3rd-place-solution) · [Solo NN](https://www.kaggle.com/code/ukrainskiydv/3rd-place-solution-nn-solo-0-69694-private-lb) |
| 🏅 **#4** | -1 | **`Herra Huu`** | **0.69718** (#3) | **0.69928** *(0.69936 clean)* | **Exact Expected-Rank Formula** combining KM-weighted $P(\text{event})$ and **probit-transformed (`inv-normal CDF`)** conditional rank regression across `LGBM` (`0.69793` solo!), `CB`, `XGB`, `TabM` | [Writeup](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/writeups/herra-huu-4th-place-factorized-modelling-approach) · [Notebook](https://www.kaggle.com/code/herrahuu/4th-place-solution) |
| 🏅 **#5** | **+33** | **Robert Hatch** | 0.69496 (#38) | **0.69881** | 42-model blend (20 CB, 13 LGB, 4 XGB, 3 TabM, 1 NN, 1 ODST Pairwise): **A/B split (`efs_time < 13.326`)** w/ logit rank + **1,000-grid Nelson-Aalen extreme pair** + `expit(Cox)` raw blend | [Writeup](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/writeups/robert-hatch-5th-place-solution-full-write-up) · [Notebook](https://www.kaggle.com/code/roberthatch/cibmtr-5th-place-official-submission) |
| 🏅 **#6** | **+27** | **Aleksandr Timofeev** | 0.69501 (#33) | **0.69873** | **20-fold × 5-seed CV**; Step-1 GBDTs ($p(\text{efs})$, $\mathbb{E}[t\mid\text{efs}=1]$) + **TabM optimizing sigmoid-smoothed Stratified C-index** → **1-layer linear NN** on 2nd-order polynomials | [Writeup](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/writeups/aleksandr-timofeev-6th-place-solution) · [Notebook](https://www.kaggle.com/code/myprofileurl/6th-place-two-step-model) |
| 🏅 **#9** | **+18** | **`uedakeishi`** | 0.69522 (#27) | **0.69743** | Dual-branch rank blend: Pairwise-Rank NNs (`NA` + `KM`) + `efs=1 & t<24` Cox/Breslow/KM GBDTs multiplied by `scale_pos_weight=3` XGB $P(\text{efs}=1)$ | [Writeup](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/writeups/uedakeishi-9th-place-solution) |
| 🏅 **#11** | -2 | **Sercan \| Siddhant** | 0.69666 (#9) | **0.69607** | Pairwise-Rank NN + race-weighted GBDTs (`KM/NA/Cox` targets), cyclic year, Optuna ordinal encoding | [Writeup](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/writeups/sercan-siddhant-11th-place-solution) |

*(Shakeup benchmark: Public #6 fell to #41 Private; Public #10 fell to #854 Private; Private #10 jumped +782 spots!)*

---

## 1. Survival Data Forensics & Metric Structure

- **Right-Censored Bimodal Support**: Uncensored events (`efs = 1`) mostly cluster in `efs_time ∈ [0, 15]`, whereas censored survivors (`efs = 0`) cluster in `[15, 160]` (`EFS_SPLIT = 13.326` cleanly separates ~99% of events vs censored rows in #5).
- **Synthetic `SurvivalGAN` Architecture Leak (`#1`, `#2`)**: Because `synthcity` `SurvivalGAN` generates `efs_time` using a conditional `TimeRegressor(X, efs)` conditioned on the binary event indicator `efs`, **censored (`efs=0`) and uncensored (`efs=1`) rows share latent feature-to-time structure**. Passing `efs` as an input feature while training the `efs_time` regressor on *both* `efs=1` and `efs=0` rows (weighted `0.6 : 0.4` in #1), then setting **`efs = 1` at inference**, substantially improved `efs=1` C-index (`0.7702` XGB in #1) over training on `efs=1` rows alone!
- **Shallow Interactions & High Categorical Density**: Out of 57 features, 55 are discrete/categorical (only `donor_age` and `age_at_hct` are continuous, with a synthetic spike at `age_at_hct = 0.044` occurring 1,005 times). Optimal tree depth for XGBoost/LightGBM classifiers was **`max_depth = 2`** (#1), and heavy feature engineering consistently failed (#2, #3, #5).

---

## 2. Core Winning Survival & Ranking Formulations

### 2.1 Exact Factorized Risk Formulas Across Top 6
| Team | Event Component | Conditional Time Component | Exact Combination Formula |
| :--- | :--- | :--- | :--- |
| **#1 `minerppdy`** | 9 classifiers predicting $p_0 = P(\text{efs}=0)$ (`CB 0.7617`, `XGB`, `LGB`, `RankNN`, `TabM`, `RankTabM`, `GraphSAGE GNN`, `NN`, `RankGNN`) | 3 GBDT regressors predicting grouped `rank(efs_time)/N` (w/ `efs` input feature, `efs=1` at test) | `y_f = c * r^b`, `x_f = p_0^a`, `res = (1 - y_f)*x_f + y_f` (Optuna `(a,b,c)` per pair across 27 pairs) |
| **#2 `agalar`** | `RealMLP + HistGBM + CB` for $p_1 = P(\text{efs}=1)$ (zero FE) | `XGB + HistGBM` (`efs` input, `efs=1` at test) | $R = p_1 \cdot \sigma(-\hat{y}_{\text{reg}})$, rank-blended with **differentiable `tanh` pairwise C-index NN** |
| **#3 `SurvivalHorror`** | `MLP-ODST` + GBDTs + LR stacker | Race-stratified `rank(efs_time) ∈ [0,1]` (`efs=1`) & `[1.345, 1.355]` (`efs=0`), trained w/ **BCE loss** + `-1k` outlier drop | Probability-weighted conditional expectations (`GBMs + NNs` Bayesian-opt weights on 20-seed OOF) |
| **#4 `Herra Huu`** | KM-weighted $P(\text{efs}=1)$ | `efs=1` only: **probit `Φ⁻¹(rank(-t)/N)`** regression mapped back via $\Phi(\hat{y})$ | $P(e=0)\frac{s_{0,g}}{2} + P(e=1)\!\left(s_{0,g} + (1-s_{0,g})\mathbb{E}[\text{rank}\%\mid e=1]\right)$ |
| **#5 `Robert Hatch`** | `a = P(efs_time < 13.326)` | `b = logit(rank ∈ [0.03, 0.97])` on `efs=1 & t<13.326` (plus 1,000-grid NA targets & `expit(Cox)`) | $P = a \cdot b - (1 - a)\cdot 0.42456$ |
| **#6 `Timofeev`** | GBDT $p(\text{efs}=1)$ | GBDT $\mathbb{E}[t\mid\text{efs}=1]$ + **TabM on sigmoid-smoothed Stratified C-index** | **1-layer linear NN** on `[TabM, deg-2 poly(p, t)]` optimizing smoothed Stratified C-index |

### 2.2 Critical Neural & Ranking Engineering Details
1. **Asymmetric Family Strengths (`#3`, `#1`)**: Neural networks (**TabM**, **MLP-ODST**, **RealMLP**, **Pairwise-Rank NN**, **25-NN GraphSAGE GNN**) matched or beat GBDTs on event **classification** (`SurvivalHorror` solo NN hit **`0.69694` Private**, top-11 alone!), whereas **GBDTs (`XGB`/`CB`/`LGB`) dominated conditional time regression** (`#1` dropped NN regressors entirely).
2. **Pairwise-Ranking Per-Fold Offset Alignment (`#1`)**: Pairwise ranking losses (`RankNN`, `RankTabM`, `RankGNN`) and Cox models are shift-invariant, producing arbitrary per-fold prediction means/medians across CV folds. Always **normalize/align per-fold mean & median** before computing pooled OOF C-index or stacking!
3. **Anti-Shakeup CV & Bounded Meta-Weights (`#1`, `#3`, `#6`)**: Because `mean_g - std_g` over 6 races (~1,200 rows/race in 25% Public LB) is very noisy, winners used **4-fold × 20–100 split seeds** (#3), **10-fold SKF without early stopping** (#1), or **20-fold × 5 seeds** (#6), and constrained Optuna ensemble weights to **`[0.1, 1.0]`** (#1) or a **zero-nonlinearity 1-layer linear stacker** (#6).

---

## 3. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Outcome | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Per-Race Post-Processing / Adding Race-Specific Noise to Shrink `std_g`** | Inflated local CV strongly, **failed on both Public & Private LB** | #1 (`minerppdy`) | Directly gaming subgroup `std` overfits fold-specific sampling variance |
| **Heavy Feature Engineering / TE / Autoencoders / Synthetic GAN Aug / Feature Selection** | Neutral or hurt CV/LB; raw features + simple freq encoding won | #2, #3, #5 | 55/57 columns are already low-cardinality discrete clinical codes with shallow (`depth=2`) structure |
| **LightGBM Pairwise Ranker on Full Fold** | Crashes (`25,920 > 10,000` max query size) | #1 / Community | LightGBM `lambdarank` caps single-query group size at 10,000 rows |
| **Evaluating Unaligned Pairwise-Rank OOFs Across Folds** | Artificially lowers pooled OOF C-index | #1 (`minerppdy`) | Shift-invariant pairwise objectives drift in location/scale between folds |
| **1-Seed CV + Early Stopping + Public-LB Chasing** | Public #6 → #41, Public #10 → #854 Private | #3, LB Forensics | Public LB (~25%) sits near the 75th percentile of 4-fold variance on `mean - std` |
| **Outlier Row Dropping Applied to Classification** | Helped regression BCE, **hurt classification LogLoss** | #3 (`SurvivalHorror`) | Dropping hard uncensored cases biases event probability calibration |
