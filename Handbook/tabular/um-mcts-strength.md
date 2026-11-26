# UM — Game-Playing Strength of MCTS Variants: Top-10 Solutions & Asymmetric-Matchup Tabular Post-Mortem

> **Competition**: [UM — Game-Playing Strength of MCTS Variants](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants) (Sep – Dec 2024, 1,608–1,620 Teams, \$50,000 Prize Pool; hosted by Maastricht University)  
> **Track**: Tabular Pairwise Matchup Regression, Unseen-Group Generalization & Embedded Domain Simulation  
> **Dataset & API**: `~233k` training rows (`1,373` unique `GameRulesetName`, `~590` non-trivial Ludii concept features + `LudRules`/`EnglishRules` + 72 compositional MCTS variants `agent1`/`agent2`); `~60k` test rows across **74 brand-new unseen games** served in **random 100-row batches** (`<= 9h` offline notebook, `10 min`/batch)  
> **Evaluation Metric**: **RMSE** on continuous `utility_agent1 = (wins - losses) / n_games ∈ [-1, 1]` (64 of 74 test games shared across 35% Public / 65% Private; 10 games Private-only)

---

## 1. Official Top-10 Leaderboard & Winning Paradigms

Because `64/74` unseen test games appeared in both Public and Private splits, top rankings were unusually stable. **1st place (`James Day`, `0.41780` Private)** won both Public and Private outright by **embedding the Ludii Java engine inside the submission notebook** to run fast 15-second starting-position MCTS evaluations (`+0.045 CV / +0.012 LB`), while #2–#10 perfected **asymmetric flip augmentation (inverting `AdvantageP1`/`Balance`)**, **nested two-stage stacking**, **LightGBM DART / TabM / DeepTables**, and **CIR or variance-scaling calibration**.

| Private Rank | Team | Private RMSE | Public RMSE (Δ) | Core Breakthroughs & Architecture | Writeup / Code |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **1st** | **James Day** | **0.41780** | **0.41369** (0) | **Live 15s Ludii v1.3.13 MCTS starting-position search (`+0.045 CV / +0.012 LB`)** + 14.3k extra GAVEL/LLM rulesets (`+0.004 CV`) + 10× search-noise aug + dropped 43 Ludii version-mismatch cols + **TabM (`0.352` CV) + LGBM DART (`+0.004`) + CatBoost + Isotonic/CIR (`+0.002`)** ("Trust CV" beat "Trust LB") | [Writeup](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/writeups/james-day-1st-place-solution) · [GitHub](https://github.com/jday96314/MCTS) |
| 🥈 **2nd** | **Richard_U** | **0.41996** | **0.41429** (0) | **TF-IDF+KMeans `LudRules` clustered GroupKFold (`1,373 → 1,270` clusters)** + flip aug (`AdvantageP1 → 1-AdvantageP1` + `is_augmented` dummy) + TTA `(orig - flip)/2` + 7 models (CatBoost/LGBM/MLP/AE-MLP) + OLS negative weights | [Writeup](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/writeups/richard-u-2nd-place-solution) · [Notebook](https://www.kaggle.com/code/fredrikupmark/um-gpsomctsv-2nd-place-solution-submission) |
| 🥉 **3rd** | **senkin13** | **0.42080** | **0.41455** (0) | **Two-stage flip-aug stacking** (Stage-1 trained strictly on flipped data → OOF features for Stage-2) + `EnglishRules` TF-IDF/SVD + Null Importance pruning + LGBM/CatBoost + **`1.12×` scaling (`+0.002` LB)** | [Writeup](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/writeups/senkin13-3rd-place-solution-two-stage-flip-augment) · [Notebook](https://www.kaggle.com/code/senkin13/um-final-sub-1) |
| 🏅 **4th** | **Manuel Campos** | **0.42192** | **0.41521** (+1) | Cascade merging of 11+ diverse pipelines (GBDTs, DeepTables NN, Matryoshka embeddings) with negative decorrelation weights + `1.25×` scaling clipped `[-0.98, 0.98]` | [Writeup](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/writeups/manuel-campos-4th-place-solution-wow-code-sharing) |
| 🏅 **5th** | **dümensemble** (`A. Ozturk`, `S. Yeşilöz`) | **0.42224** | **0.41621** (+3) | 10-fold GPU CatBoost (`depth=10`, flip+TTA: `0.4059 → 0.4030` CV, `adv_p1_adj = AdvP1*Completion + Drawishness/2`) blended with DeepTables (`DNN+FM+CIN+IPNN`) + stacked extra-trees LGBM/CatBoost | [Writeup](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/writeups/d-mensemble-5th-place-solution) |
| 🏅 **6th** | **Vadim Timakin** | **0.42283** | **0.41746** (+6) | Joint `(target, agent1)` `StratifiedGroupKFold` + **discovered inverting `Balance` alongside `AdvantageP1`** + leak-free **Nested `5×5` CatBoost OOF** → Stage-2 GBDTs & `Hardswish + MADGRAD` MLP + OpenFE (no group/freq) + `1.175×` clip | [Writeup](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/writeups/vadim-timakin-the-6th-place-solution-cv-0-39) · [GitHub](https://github.com/vadimtimakin/Kaggle-MCTS-6th-place-solution) |
| 🏅 **7th** | **gezi** | **0.42363** | **0.41694** (+2) | Flip+TTA (`+0.01` LB) + manual `*` & `/` crosses of top-20 numerics + PyTorch `CIN+FM` NN on **LGBM leaf features** + linear `a·x + b` PP | [Writeup](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/writeups/gezi-7th-solution-ensemble-of-tree-nn) |
| 🏅 **8th** | **Kohei** | **0.42374** | **0.41604** (-1) | Polars FE (`410s → 7s`) + single 10-fold CatBoost w/ flip aug + Ludii features + `1.20×` scaling | [Writeup](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/writeups/kohei-8th-place-solution-a-10-day-challenge) |
| 🏅 **9th** | **No Overfitting, Just Skills** | **0.42451** | **0.41579** (-3) | 8-fold GroupKFold; **separate original vs. flipped models (`+0.01 CV / +0.003 LB`)** & **`MultiRMSE([y, -y])`** + `2*AdvantageP1 - 1` blend | [Writeup](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/writeups/no-overfitting-just-skills-9th-place-solution-vari) |
| 🏅 **10th** | **DO THE HARLEM SHAKE** | **0.42458** | **0.41705** (0) | Flip aug (`0.427 → 0.422`) + **10-binning noisy `AdvantageP1`** (`→ 0.420`) + `1.25×` scaling (`→ 0.417` Pub / `0.42458` Priv) | [Writeup](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/writeups/do-the-harlem-shake-10th-place-solution) |

---

## 2. Core Tabular & Domain Engineering Breakthroughs

### 2.1 Embedded Fast-Simulator Proxy Features & Multi-Run Noise Augmentation (#1 James Day)
- **Fixing Random-Playout Bias**: Provided `AdvantageP1` came from *uniform-random* playouts, poorly reflecting first-mover advantage under smart MCTS search. Running the actual **Ludii v1.3.13 Java engine** inside the submission notebook for just **15 seconds of `UCB1Tuned-0.6-Random200-true` tree search on each unseen game's starting position** (`~250×` faster than full matches) delivered **`+0.045` CV / `+0.012` LB** (more than 5× any other single feature block!).
- **Stochastic Feature Noise Augmentation (`+0.002` CV/LB)**: Because MCTS evaluations are stochastic, James ran every training game's search `10×` (sampling from 9 non-held-out runs per epoch) and interpolated nondeterministic Ludii concepts `5×` so models stopped overfitting single-run evaluation jitter.
- **Version-Inconsistent Feature Purge (`+0.001–0.002` CV)**: Auditing deterministic features computed by Ludii v1.3.13 against `train.csv` caught **43 features with silent version mismatch**; dropping all 43 together lifted CV cleanly.
- **Synthetic Ruleset Generation (`+0.004` CV / `+0.001–0.002` LB)**: Generated `14,365` new rows across `484` synthetic rulesets (`391` GAVEL + `93` Llama-3.1-70B/Qwen-2.5-32B), down-weighted by per-model CV noise sensitivity (`CatBoost 0.25`, `TabM 0.5`, `LightGBM 1.0`).

### 2.2 Asymmetric Flip Augmentation, First-Mover Inversion & TTA (#2–#10)
Unlike symmetric sports matchups where $P(A, B) = 1 - P(B, A)$, board games carry strong **first-mover advantage**, so naive odd symmetry $u(A, B) = -u(B, A)$ fails. Winners unlocked **`+0.003–0.010` LB** via four refinements:
1. **Invert Asymmetric First-Mover Features**: Swap `(agent1, agent2)`, negate `utility_agent1 → -utility_agent1`, invert `AdvantageP1 → 1 - AdvantageP1` (#2, #3, #5, #9, #10), **AND invert `Balance`** (#6 Vadim Timakin discovered `Balance` was the #2 feature; inverting it closed the residual OOF error gap between original and flipped copies!).
2. **Track & Separate Flipped Semantics**: Add a binary `is_augmented` flag (#2), train **separate original vs. flipped models** (#9: `+0.01 CV / +0.003 LB` vs `+0.001 LB` merged), feed **Stage-1 Flipped OOFs into Stage-2 Original** (#3), or train CatBoost with **`MultiRMSE` on `[y, -y]`** (#9, `+0.002` CV).
3. **Symmetry TTA**: Predict `(pred(orig) - pred(flip)) / 2` at inference (`+0.001–0.003`), always scoring CV **strictly on unaugmented original rows**.

### 2.3 Validation, Model Diversity & De-Shrinkage Calibration
- **Near-Duplicate Entity-Cluster GroupKFold (#2)**: Clustering `LudRules` via `TF-IDF + KMeans` (`1,373 → 1,270` clusters) prevented slight variants of the same board game from leaking across folds.
- **Leak-Free Nested `5×5` Stage-2 Stacking (#6)**: Plain 5-fold OOF features leaked heavily when fed into Stage-2 GBDTs/NNs; computing Stage-1 OOFs via **nested `5×5` CatBoost** fixed leakage cleanly.
- **Model Family Winners**: **`TabM`** (`0.352` CV, #1's best single model!) + **`LightGBM DART`** (`+0.004 CV / +0.003 LB` over standard GBDT, #1) + deep regularized **`CatBoost`** (`depth=10`) + **`DeepTables` (`DNN+FM+CIN+IPNN`) / `Hardswish + MADGRAD` MLP** (#5, #6, #7).
- **Mean-Shrinkage Correction**: Under RMSE, GBDTs systematically shrink predictions toward `0`. While #3/#4/#6/#8/#10 used linear `1.12–1.25×` multipliers + `[-1, 1]` clipping (`+0.002–0.005` LB, aided by 64/74 shared Public/Private games), **#1 (`James Day`) achieved principled `+0.002` CV & LB gain via OOF Isotonic / Centered Isotonic Regression (CIR)** with zero LB fitting.

---

## 3. Empirical Negative Results & What Failed

| Failed Hypothesis | Observed Result | Root Cause |
| :--- | :--- | :--- |
| **Test-Time Pseudo-Labeling (#3, #6, #10)** | Degraded CV & LB | High baseline RMSE (`>0.40`) plus random 100-row API batching prevented reliable game-level pseudo-labels. |
| **Pretrained Transformer / LLM Embeddings on Rulesets (#1, #3, #6)** | Overfit or added zero gain | Only `1,373` unique training games + formal `LudRules` DSL syntax made text embeddings memorize training game identities. |
| **Multi-Task Decomposition (`wins/n - losses/n`) (#1, #6, #9)** | `+0.01` CV in #9 but **hurt LB (`0.421`)** | Decomposed heads over-relied on random-playout `AdvantageP1` (which reflects P1 win rate rather than smart-search win-minus-loss). |
| **Transitivity & Self-Play Synthetic Pairs (`A vs A`, `A>B & B>C`) (#6, #8, #9)** | Helped local CV (#9), failed LB/Private | Non-transitive rock-paper-scissors MCTS matchups broke assumed transitivity on unseen games. |
| **Group-Aggregated & Frequency OpenFE Features (#1, #6)** | Unstable on unseen test games | Game-level frequency/group encodings overfit the 1,373 seen games and shifted under denser test agent-pair sampling. |
| **`XGBoost` & `TabNet` (#1, #6, #9)** | Consistently trailed `CatBoost`, `LGBM DART`, and `TabM` | Oblivious trees (`CatBoost`), dropout-regularized trees (`DART`), and PLR neural ensembles (`TabM`) generalized much better across unseen game clusters. |
| **Standard Feature Importance / Forward-Backward Selection (#1, #6, #9)** | Failed to beat all-feature models (only version-mismatch purge #1 & Null Importance #3 helped) | Weakly correlated game-concept features carried aggregate tail signal across unseen rulesets. |

---

## 4. Universal Takeaways Promoted / Reinforced

- **Asymmetric Swap Augmentation & TTA**: When pairwise matchups have a structural side/order bias (first-mover, home-field, prompt A/B position), swap pair features and negate target **while also inverting side-advantage/balance covariates** (and test `is_augmented` flags, separate orig/flipped models, or `MultiRMSE([y, -y])`).
- **Text/Embedding-Clustered `GroupKFold` for Near-Duplicate Entities**: When group IDs include slight variants of the same underlying object/ruleset/molecule, cluster group descriptions (`TF-IDF + KMeans` or embedding similarity) *before* assigning `GroupKFold` splits.
- **Live Fast-Proxy Simulation + Multi-Run Noise Augmentation**: Whenever a deterministic domain simulator/interpreter is permitted at test time, running a fast proxy probe (e.g., 15s starting-position evaluation) + training with multi-run stochastic noise augmentation can unlock order-of-magnitude CV/LB breakthroughs over static metadata.
- **`LightGBM DART` + `TabM` + Nested OOF CIR Calibration**: Pair dropout-regularized `DART`, `TabM`, and `CatBoost` with OOF Isotonic/CIR calibration to counter tree mean-shrinkage under RMSE without risking LB-tuned variance multipliers.
