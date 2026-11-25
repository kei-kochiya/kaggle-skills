# March Machine Learning Mania 2026 — Top 10 Solutions & Post-Mortem

> **Competition**: [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) (Feb 19 – Apr 7, 2026 | 3,462 Teams | \$50,000 Prize Pool)  
> **Task**: Forecast pairwise win probabilities (`Pred = P(Team1 beats Team2)`, `Team1 < Team2`) across all **519,144** possible Men's + Women's NCAA D1 pairs; scored live on actual 2026 tournament matchups (~126 main-bracket games)  
> **Evaluation Metric**: **Brier Score** (MSE between predicted win probability and binary `{0, 1}` outcome) — 2026 restricted teams to **1 selected submission**  
> **Dataset**: `35` CSV files (`181 MB`): Men's compact results (`1985+`), Women's (`1998+`), detailed box scores (`2003+` M / `2010+` W), geography (`2010+`), Men's-only `MMasseyOrdinals` (`2003+`, `RankingDayNum <= 133`)

---

## Official Top 10 Private Leaderboard Standings & Solution Index

In a "chalky" (low-upset) 2026 tournament with only **1 submission slot**, ultra-compact graph-rating pipelines (**shallow `max_depth=2` XGBoost**, **regularized Logistic Regression**, **pure Elo**, and **point-differential regression → Isotonic calibration**) decisively beat deep tree/feature zoos, while external injury and market-odds integration drove the largest verified edge:

| Private Rank | Team / Competitor | Private LB (Brier ↓) | LOSO / Season CV | Core Winning Paradigm | Writeup / Code |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **Harrison Horan** (`harry`) | **0.109745** | 0.1590 cal *(0.1822 M / 0.1357 W)* | **Ultra-minimal 3–4 diff features** (`seed`, `opp_qlty`, custom `harry_Rating` w/ **Men's Rotowire+EvanMiya injury deductions** worth `~0.0041` Brier!) → shallow `XGBRegressor(max_depth=2)` + separate M/W `IsotonicRegression` + `>=97%` edge sharpening | [Writeup](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/march-machine-learning-mania-2026-1st-place-solut) · [GitHub](https://github.com/harrisonhoran/kaggle-march-mania-2026-1st-place) · [NB](https://kaggle.com/code/harrisonhoran/harry-march-mania-2026-notebook) |
| 🥈 **#2** | **Brendan Carlin** | **0.114989** | 0.1925 M / 0.1480 W | Official data only; unified M+W 35 diffs (Four Factors, MOV-Elo, Massey, SOS, RecentForm) → `0.40 * LogisticRegression(C=1)` + `0.60 * XGBClassifier(max_depth=4)` clipped **`[0.02, 0.98]`** | [Writeup](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/2nd-place-solution-for-the-march-machine-learning) · [GitHub](https://github.com/BrendanCarlin/march-mania-2026-2nd-place) |
| 🥉 **#3** | **Kevin E R MILLE** | **0.116037** | **0.12352** *(2015–25 LR)* | Separate M/W **pure `LogisticRegression`** (`C=100` M / `C=0.15` W) on pruned Barttorvik + GLM Quality + Colley + SRS + 28-system Massey → **4-tier Stage-2 market blend** (Vegas no-vig, ESPN BPI, Kalshi **Bradley-Terry**) | [Writeup](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/3rd-place-solution-march-machine-learning-mania) · [GitHub](https://github.com/kevin1000/march-mania-2026-3rd-place) |
| 🏅 **#4** | **Yi Zhang** (`Xroxa`) | **0.117265** | 5-fold `GroupKFold(Season)` | Official data only; unified M+W **symmetric augmentation (`(A,B)→1, (B,A)→0`)** on 20 possession-adjusted diffs + `elo_ratio` + `seed_product` → `0.65 XGB + 0.35 LGBM`, clipped `[0.025, 0.975]` | [Writeup](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/4th-place-solution-for-the-march-machine-learning) · [GitHub](https://github.com/Xroxa/4th-Place-Solution-for-the-March-Machine-Learning-Mania-2026-Competition) |
| 🏅 **#5** | **Shanru Lin** (`alphawave9`) | **0.117460** | Historical Brier | **Zero ML / Pure Elo** (`width=1200`, `K=125` M / `K=190` W, no box scores or seeds) + high-risk champion overrides on Michigan (M) & UCLA (W) | [Writeup](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/march-machine-learning-mania-2026-5nd-place-solut) · [NB](https://kaggle.com/code/alphawave9/march-2026-sub) |
| 🏅 **#6** | **`seddik SsD`** | **0.118119** | Temporal `< N → N` | `LR + XGB + LGBM` + Elo slope + **5%–10% seed-prior blending regularization** (`α·model + (1-α)·seed_pred`) | [Writeup](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/6th-place-solution-march-machine-learning-mania-20) · [GitHub](https://github.com/seddiksbnk-max/march-mania-2026-6th-place-solution) |
| 🏅 **#7** | **`abobus`** (`Baderko`, `Gorokhov`) | **0.118369** | 0.16272 LOSO | **Point-differential regression (`reg:squarederror`)** on 89 features w/ **40-min OT box-score normalization** + GLM quality + Elo → separate M/W **Isotonic calibration** (`[0.01, 0.99]`) | [Writeup](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/7th-place-solution) · [NB](https://kaggle.com/code/makarbaderko/abobus-solution-7th-place) |
| 🏅 **#8** | **Phạm Quang Hữu Phước** | **0.118450** | 0.1881 M / 0.1482 W | Separate M/W `0.5 XGB + 0.5 LGBM` MSE regressors, symmetric aug, exponential season decay `exp(-0.15·Δyr)`, 22 Sabermetric diffs (`Delta_Quality` top), clipped `[0.02, 0.98]` | [Writeup](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/8th-place-march-machine-learning-mania-2026) · [GitHub](https://github.com/huuphuoc-phamquang/8th-march_ML_mania) |
| 🏅 **#10** | **Cyrus** (`jamalsaeedi`) | **0.119254** | LOSO (2003+) | Joint M+W **PointDiff XGB regression** w/ OT-normalized stats + Gaussian GLM fixed effects + 6-seed Optuna joint tuning of XGB + feature selection + margin trim | [Writeup](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/march-mania-2026-10th-place-solution) |

---

## 1. Competition DNA & Pairwise Validation Discipline

- **Small-Sample Tournament Reality**: Only `~63–67` tournament games per gender occur each year (`~1,300` total games over 10 seasons). High-capacity GBDTs immediately memorize historical seasons unless aggressively regularized (`max_depth=2` in #1, `max_depth=4` in #2/#7) or replaced by linear models (#3: LR CV `0.124` vs XGB `0.157`).
- **Strict Season-Grouped CV**: Every medalist used **Leave-One-Season-Out (LOSO)**, `GroupKFold(Season)`, or **chronological `< N → N`** validation. Regular season ends at `DayNum = 132` (Selection Sunday) and final Massey rankings at `RankingDayNum = 133`; `raddar` (`/discussion/683033`) also recommended **dropping play-in games from training** (huge MSE outliers) and dropping teams with `<33%` win rate.
- **Men's vs. Women's Predictability Gap**: Women's brackets are far more seed-predictable (`76.2%` higher-seed win rate, CV Brier `~0.135–0.148`) than Men's (`68.9%`, CV Brier `~0.182–0.193`). Separate M/W pipelines allowed tighter regularization for Women (`C=0.15` vs `100` in #3) and Men's-only signals (Massey Ordinals, injury BPR).

---

## 2. Universal Pairwise Matchup & Probability Calibration Techniques

1. **Antisymmetric Difference Vectors + Symmetric Augmentation (`#1`–`#4`, `#8`)**:
   - Represent matchups strictly via antisymmetric differences (`feat(A) - feat(B)`) and symmetric products (`seed_A * seed_B`).
   - Duplicate every training game `(A, B) → 1` as `(B, A) → 0` (negating diffs, inverting ratios `1 / elo_ratio`) to enforce $P(A > B) + P(B > A) = 1$ and double effective sample size.
2. **Global Game-Graph Latent Ratings Beat Raw Box Scores**:
   - **Possession-Adjusted Efficiency & 40-Min OT Normalization**: Normalize box scores to 40 minutes (`#7`, `#10`) and per possession (`Poss = FGA - OR + TO + 0.475*FTA` or `0.44*FTA`) to eliminate pace and overtime distortion.
   - **Graph Ratings**: **Gaussian/MLE GLM team fixed-effect quality** (`#3`, `#7`, `#8`, `#10`, `raddar`), **Colley Matrix** `(2I + C)r = 1 + (w - l)/2` and iterative **SRS** (`#3`), **MOV-adjusted carryover Elo** (`K=20–32`, `home_adv=100`, `0.75` inter-season mean reversion to `1500` in `#2`–`#4`), and **all-system Massey composites** (`28` full-coverage systems beat a `15`-system subset by `-0.00246` Brier in `#3`).
3. **Continuous Margin Regression → Monotonic Calibration (`#7`, `#10`, `raddar`)**:
   - Regressing continuous `PointDiff = Score_1 - Score_2` (`reg:squarederror`) extracts far more information per game than binary `{0, 1}` targets, then converts OOF margins to probabilities via separate M/W **`IsotonicRegression`** (`#1`, `#7`).
4. **External Real-World Edge & Market Bradley-Terry Fusion (`#1`, `#3`)**:
   - **Roster Injury Adjustments (`#1`)**: Deducting `Rotowire status weight × EvanMiya Adj BPR` across 16 injured Men's players improved Men's Brier by `~0.0081` (`~0.0041` overall!), accounting for a huge portion of #1's `0.0052` margin over #2!
   - **Tiered Market Bradley-Terry Fusion (`#3`)**: Converting Vegas no-vig moneylines, ESPN BPI, and Kalshi championship futures into pairwise win probabilities via **Bradley-Terry** $s_A / (s_A + s_B)$ (`90%` market / `10%` LR in Men's R1; `15%–25%` futures BT in later rounds).
5. **Expected-Brier Tail Clipping vs. Contest-Meta Gambling**:
   - Under Brier score, clipping extreme probabilities to **`[0.02, 0.98]`** (#2, #8) or **`[0.025, 0.975]`** (#4) protects against catastrophic upset squared errors (`0.98` miss = `0.9604` vs `1.0` miss = `1.0`).
   - Conversely, `#1`'s `>=97% → 1.0` rounding (`+0.00011` actual gain) and `#5`'s Michigan/UCLA champion overrides are explicitly `-EV` variance gambles that only pay off in chalky single-slot years.

---

## 3. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Deep Trees (`max_depth >= 6`) / Complex Feature Bloat on ~1,300 Games** | Overfit CV/LB (`LR 0.124` vs `XGB 0.157` in #3; `depth=6/8` overfit in #2) | #1, #2, #3 | Tiny effective tournament sample size favors shallow trees (`depth 2–4`), backward pruning (`-0.007` Brier in #3), and linear models |
| **Rewarding Late-Season Recency Blindly** | Worsened CV in #1 and hurt LR in #3 | #1, #3 | Small late-season game samples add variance compared to full-season + early-poll stability |
| **Pre-2015 Data With Missing External Ratings / Raw Scoring Without OT & Pace Normalization** | Hurt CV/LB | #3, #4 | Systematic missingness/imputation across eras and pace/OT inflation distort efficiency metrics |
| **Unverified Post-Hoc Calibration / Manual Overrides** | Isotonic/Platt hurt in #3 & #8; manual overrides hurt log-loss in #6 | #3, #6, #8 | Well-regularized LR/MSE models are already near-calibrated; extra calibration layers add small-sample variance |
| **Skipping Symmetric Matchup Augmentation** | Introduced TeamID-ordering bias and halved training pairs | #4, #8 | `Team1 < Team2` ordering is arbitrary; exact antisymmetry must be enforced |
