# Playground Series S6E4: Predicting Irrigation Need — Top Solutions & Post-Mortem

> **Competition**: [Kaggle Playground Series Season 6 Episode 4](https://www.kaggle.com/competitions/playground-series-s6e4) (Apr 1 – May 1, 2026 | 4,315 Teams | 44,334 Submissions)  
> **Target**: 3-Class Imbalanced `Irrigation_Need` (`Low` `58.7%`, `Medium` `37.9%`, **`High` `3.3%`** — `21,009` train rows; `~1,801` rows in 20% Public LB)  
> **Evaluation Metric**: **Balanced Accuracy** (`balanced_accuracy_score`)  
> **Dataset**: `630,000` Train Rows × `21` Cols | `270,000` Test Rows × `20` Cols | Original Seed: `10,000` Rows ([Irrigation Water Requirement Dataset](https://www.kaggle.com/datasets/miadul/irrigation-water-requirement-prediction-dataset/data))  
> **LB Split**: 20% Public (`54k` rows) / 80% Private (`216k` rows) — severe shakeup outside Top 5 due to 3.3% `High` class leverage (~17.6× per error vs `Low`)

---

## Official Top Standings & Solution Index

With `High` comprising only 3.3% of rows yet carrying **33.3% of Balanced Accuracy**, tiny Public LB fluctuations produced a massive shakeup outside ranks #2–#5 (Public #1, #7, #8, #10 all fell out of Top 49, while disciplined CV stackers jumped **+73 to +530 ranks**):

| Private Rank | Δ Rank | Team / Competitor | CV | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| 🥇 **#1** | +73 | **`cstdy`** (`kirill0212`) | **0.98155** | #74 | **0.98158** | **2-Stage Hierarchical Binary Decomposition** (`Low vs Rest`, then `Medium vs High` on predicted non-Low) + 61-OOF logit `LogisticRegressionCV` + greedy OOF bACC thresholds | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/1st-place-one-vs-rest-approach) · [Notebook](https://www.kaggle.com/code/kirill0212/ps6e4-ensemble-cv-0-98155) |
| 🥈 **#2** | +1 | **Chris Deotte** (`cdeotte`) | **0.98170** *(0.98175 unsel.)* | 0.98195 *(#3 0.98219)* | **0.98151** *(0.98172 unsel.)* | Claude Code + Codex CLI conversion of 150 S6E3 models (125+ L1) → **PyTorch GPU Class-Weighted Multinomial LogReg** (`argmax`, zero threshold PP) | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/2nd-place-claude-code-and-codex-gpu-logreg) · [Formula NB](https://www.kaggle.com/code/cdeotte/original-data-exact-formula) |
| 🥉 **#3** | +1 | **`r0tor team`** (`nikita7364777`) | — | #4 (0.98201) | **0.98147** | 203-model diverse zoo (TabM, XGB, LogReg, CB, LGB, HistGB, RF, RealMLP, Trompt, KerasMLP) → **single LightGBM stacker + conservative guarded override**; pruned YDF/GNN/SVM | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/error-diversity-matters-200-model-stacking-soluti) · [Code](https://www.kaggle.com/code/nikita7364777/test-200-models) |
| 🏅 **#4** | +1 | **`Optimistix`** | — | #5 (0.98200) | **0.98133** *(0.98148 unsel.)* | **"More Ensemblers Than Models"**: 11 L1 ensemblers → 4 L2 ensemblers (Hill Climb, Logit LR, DiffEvol, Top-K) → 3-way vote; 9-model unselected sub hit `0.98148` Priv | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/4th-place-more-ensemblers-than-models) |
| 🏅 **#5** | +1 | **`Topiast`** | **0.98178** | 0.98079 *(#6 0.98199)* | **0.98133** | 24/7 Claude/Codex queue: **agreement-threshold pruning (`0.9964`)** → 119 models across 16 families (incl. 1D-CNNs, DAE latents, rule features) → 5-seed LightGBM stacker | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/5th-place-solution-writeup-ai-for-large-scale-ex) |
| 🏅 **#12** | +457 | **Will** (`wguesdon`) | **0.98045** | 0.98081 | **0.98082** | **4×-Shuffled Ordered Target Encoding (`OTE`)** + digit extraction across 41 models (6 families) → LightGBM L2 stacker + **OOF log-bias correction** (`+0.0005–0.001`) | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/12th-place-solution-stacked-ensemble-with-ordered) · [Notebook](https://www.kaggle.com/code/wguesdon/ps6e4-30-model-ensemble-with-stacking) |
| 🏅 **#19** | +312 | **Elena Petrova** | — | — | **0.98071** | 29 models stacked via Optuna-tuned **`RidgeClassifier`** on probs + class labels + exact rule formula (no bias tuning) | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/19nd-place-ensemble-of-29-models) · [Notebook](https://www.kaggle.com/code/elenkapetrova/ps-s6-e4-metamodel-ridge) |
| 🏅 **#24** | +411 | **Shiv Satyam** | — | — | **0.98065** | 166 OOFs stacked via RealMLP + TabPFN; multi-strategy **`KBinsDiscretizer(80)`** (`uniform/quantile/kmeans`) boosted XGB (`+0.0010`) | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/24th-place-a-heavy-stacking-approach-with-166-oof) |
| 🏅 **#30** | +42 | **`chipssy`** (`shotafubasami`) | 0.98068 | 0.98059 | **0.98059** | **Conservative slot selection** (`0 < OOF - Pub ≲ 0.0001` & `High` drift `≤ ±100`) + proof that LR stacker requires all 3: logits + `class_weight='balanced'` + `multinomial` | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/ps-s6e4-retrospective-42-rank-jump-to-top-0-7) · [Notebook](https://www.kaggle.com/code/shotafubasami/ps6e4-42-rank-jump-conservative-slot) |

---

## 1. Data Forensics: Exact Additive Formula & Ordered Geometry

### 1.1 Exact Deterministic Rule on the 10,000-Row Original Dataset
Chris Deotte ([`original-data-exact-formula`](https://www.kaggle.com/code/cdeotte/original-data-exact-formula)) proved that **100% of original rows (`Balanced Accuracy = 1.0000`)** follow an exact integer scoring formula over **6 features** (9 binary conditions):

$$\text{HighScore} = 2\,\mathbb{I}[\text{Soil\_Moisture} < 25] + 2\,\mathbb{I}[\text{Rainfall\_mm} < 300] + \mathbb{I}[\text{Temperature\_C} > 30] + \mathbb{I}[\text{Wind\_Speed\_kmh} > 10]$$
$$\text{LowScore} = 2\,\mathbb{I}[\text{Stage} = \text{Harvest}] + 2\,\mathbb{I}[\text{Stage} = \text{Sowing}] + \mathbb{I}[\text{Mulching} = \text{Yes}]$$
$$\text{Score} = \text{HighScore} - \text{LowScore} \in [-3, 6] \implies \begin{cases} \text{Low}, & \text{Score} \le 0 \\ \text{Medium}, & 1 \le \text{Score} \le 3 \\ \text{High}, & \text{Score} \ge 4 \end{cases}$$

- **Structural Consequences**:
  1. **Zero `Low ↔ High` Confusion**: Because `Low` (`Score ≤ 0`) and `High` (`Score ≥ 4`) are separated by a 4-point margin across `Medium`, models almost never confuse `Low` and `High`—virtually all errors lie strictly on the `Low ↔ Medium` and `Medium ↔ High` boundaries.
  2. **Only 6 Core Signal Features**: `lofo-importance` (Ahmet Erdem) and optimal `xgb max_depth = 1` on original data confirmed minimal native interaction complexity in the original seed; the other 13 columns act largely as synthetic generator context/noise.
  3. **Original-to-Train Drift**: Adversarial validation between the 10k original dataset and 630k synthetic train exceeded **0.65 ROC-AUC**, and tuning the rule thresholds on synthetic train (`Rainfall_mm < 730.66` vs `300`) achieved `~0.94` alone (`~0.9745` when feeding rule logits into XGBoost).

---

## 2. Core Winning Techniques

### 2.1 2-Stage Hierarchical Binary Decomposition (1st Place — `cstdy`)
Directly exploiting the ordered zero-`Low↔High`-confusion geometry:
1. **Stage 1 (`Model_1`)**: Binary classifier trained on all rows predicting `Low` vs. `Rest (Medium + High)`.
2. **Stage 2 (`Model_2`)**: Binary classifier predicting `Medium` vs. `High`, **trained exclusively on rows predicted as non-Low by `Model_1`** (outperforming training on true `Medium/High` labels because Stage 2 sees Stage 1's actual boundary errors!).
3. **Exact Probability Reconstruction**:
   $$P(\text{Low}) = p_1, \qquad P(\text{Medium}) = (1 - p_1)(1 - p_2), \qquad P(\text{High}) = (1 - p_1)\,p_2$$
- Single XGBoost and RealMLP models reached **`0.9805` CV** under this decomposition alone. Applying the exact same 2-stage binary decomposition on **logits** of 61 OOF models via 5-fold `LogisticRegressionCV` + greedy OOF Balanced Accuracy threshold search won **1st Place (`0.98158` Private, `+73` ranks)**.

### 2.2 Class-Weighted PyTorch GPU Multinomial LogReg Stacker — Code Deep-Dive (2nd Place — Chris Deotte)
Why standard GPU/CPU linear stackers failed on Balanced Accuracy:
- `cuML LogisticRegression` fits **One-vs-Rest** binary models without class weights rather than true multinomial softmax.
- `chipssy` (#30) proved `LogisticRegression` stacking crashed to `0.97900` on raw probabilities unless **all three** conditions held simultaneously: **(1) logit inputs**, **(2) `class_weight='balanced'`**, and **(3) true `multinomial` loss**.
- **Code-Verified Implementation** ([`s6e4_irrigation_need/stack/stack.py`](https://github.com/cdeotte/kaggle-playground-solutions/tree/main/s6e4_irrigation_need)):
  - **100-Model Open-Source Repo Score**: **CV `0.98172` / Public `0.98161` / Private `0.98161`** (outscoring his own 229-model final sub `0.98151` and even #1's `0.98158`), formed by `argmax(P_A + 0.25 * P_B)` across **Stack A** (53 models) and **Stack B** (47 models including 9 ported public notebooks with **all base-model pseudo-labeling and threshold/bias tuning explicitly stripped**).
  - **Logit & Nested-OOF Handling**: Clips probabilities to `[1e-15, 1 - 1e-15]`, transforms to per-class logits `clip(log(p / (1 - p)), -30, 30)`, and averages 5×5 nested OOF tensors `(5, N, 3)` **in logit space** (`z.mean(axis=0)`).
  - **Exact PyTorch GPU Stacker**: Full-batch `nn.Linear(3 * n_pairs, 3)` trained for `1,000 epochs` with `Adam(lr=0.01)`, L2 `weight_decay = 1.0 / (C * len(tr))` (`C=10`) on `model.weight` **only** (`0.0` on `bias`), `CrossEntropyLoss(reduction='none') * balanced_sample_weights`, averaged across **5 seeds × 5 folds = 25 models** — requiring **zero post-processing** (`argmax` directly).
  - **Standout Single Base Models in Repo**: `realmlp_torch_nested_combo_te_b` (**`0.978822`** OOF, 8-member PyTorch RealMLP), `lgbm_three_block_union_nested_te_b` (**`0.978524`**), `blamerx_xgb` clean (`0.978298`), `xgb_union_snap_te_cs060_a` (`0.976888`), `lr_pair_te_logit3_formula_b` (**`0.975785`** — pure Logistic Regression on 171 pair TEs `[z, z², z³]` + rule score!), `deepfm_snapped_quantile_tokens_a` (`0.975683`).

### 2.3 Feature Engineering Highlights
- **4×-Shuffled Ordered Target Encoding (`OTE`) + Digit Extraction (12th Place, shared into 1st & 3rd)**: Concatenating 4 random fold permutations with sequential leave-one-out TE prevented target-encoding overfitting and boosted single GBDTs from `0.978 → 0.97942` CV.
- **Boundary-Proximity & Rule Features (5th, 19th)**: Distances to the 6 original-rule thresholds, derived rule scores/logits, and interactions with `Crop_Growth_Stage` & `Mulching_Used`.
- **Geophysics, Cyclic Embeddings & Multi-Strategy Discretization (3rd, 24th)**: Inverse conductivity (resistivity `ρ = 1 / σ`), `sin(2π * Rainfall_mm / 100)` & `sin(2π * Soil_Moisture / 4)` in `keras_mlp_v1`, plus `KBinsDiscretizer(n_bins=80)` across `uniform`, `quantile`, and `kmeans` (`+0.0010` XGB CV).

### 2.4 Error Diversity & Shakeup Defense on Imbalanced Balanced Accuracy
- **True Multi-Family Diversity (3rd, 5th, 12th)**: Mixing GBDTs (`XGB`, `LGBM`, `CatBoost`, `HistGB`) with `RealMLP`, `TabM`, `Trompt`, `KerasMLP`, `1D-CNN`, `TabTransformer`, `RandomForest`, `ExtraTrees`, `LogisticRegression`, `GaussianNB`, and `KNN` drove substantial stacker lift (`+0.0009` Public LB in 3rd place from adding NNs + RF + LogReg).
- **Agreement-Threshold Pruning (`0.9964`, 5th Place)** & **Conservative Guarded Overrides (3rd Place)**: Controlling `Medium → High` and `High → Medium` flips separately avoided the `4.4×` wrong-favored flip asymmetry on the 3.3% `High` class (`#30`).
- **Slot Selection Discipline (`#5`, `#30`)**: A 54-team public blend cluster at `0.98148–0.98152` collapsed `-0.0010` to `0.98049` Private. Selecting slots with `0 < (OOF - Pub) ≲ 0.0001` and `High` prediction count drift `≤ ±100` (and trusting `0.98178` CV despite `0.98079` Public LB in 5th place) completely neutralized shakeup.

---

## 3. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Outcome | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **YDF (Yggdrasil Decision Forests)** | Dropped LB `0.98201 → ~0.98130` (#3); negative (#24) | #3, #24 | Over-predicted majority `Medium` and suppressed rare `High` (`3.3%`) under Balanced Accuracy |
| **GraphSAGE KNN GNN & SVM/SVC** | Hurt LB (`-0.0001`, #12; dropped LB despite CV gain, #3) | #3, #12 | Low incremental stacker gain on noisy 6-feature geometry + >2h runtime for SVM |
| **NaN-Filled `OTE` Features Fed to `RealMLP`** | Crashed `RealMLP` CV `0.97802 → 0.96509` | #12 | First-row OTE `NaN` values filled with `0` created severe out-of-distribution spikes for neural inputs |
| **Hard Pseudo-Labeling (`99.9%` Agreement)** | Crashed CV to `0.95390` | #12 | Broke the 4× OTE shuffle structure and amplified rare-class boundary bias |
| **Greedy Forward Selection / Aggressive Threshold Tuning** | Overfit CV (`0.98081` CV → `0.97973` LB, #12); bias tuning hurt Priv (#19) | #12, #19, #30 | 3.3% `High` class makes greedy discrete optimization overfit validation fold noise |
| **Replacing Raw/TE Features With Rule Logits Only** | Dropped base bACC by `-0.033` | #30 | Synthetic generator drift means original rule alone misses generator perturbations |
| **Raw-Probability `LogisticRegression` Stacker** | Collapsed to `0.97900` | #2, #30 | Requires logit transform + `class_weight='balanced'` + true `multinomial` loss |
