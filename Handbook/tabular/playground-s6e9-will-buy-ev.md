# Kaggle Playground Series S6E9: Predicting Electric Vehicle Adoption (`Will_Buy_EV`)

> **Competition**: [Kaggle Playground Series Season 6 Episode 9](https://www.kaggle.com/competitions/playground-series-s6e9)  
> **Evaluation Metric**: Area Under the ROC Curve ($\text{ROC-AUC}$) on 286,571 test rows  
> **Dataset**: 668,665 Train Rows, 286,571 Test Rows (Synthesized from 10,000-row seed `itzzomkar/ev-adoption-behavior-and-range-anxiety`)  
> **Target**: Binary `Will_Buy_EV` ("Yes" = 1, "No" = 0, Positive Base Rate: `17.4645%`)  
> **1st Place Code & Writeup**: [1st Place — The AI Relay](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/1st-place-the-ai-relay) & [GitHub 60-Model Repo (`s6e9_ev_purchases`)](https://github.com/cdeotte/kaggle-playground-solutions/tree/main/s6e9_ev_purchases)

---

## 1. Competition Overview & Leaderboard Shakeup Forensics

Playground Series S6E9 concluded with a major shakeup between the 20% Public split (57,314 rows) and 80% Private split across 3,576 teams. Within the public Top 10, the Spearman correlation between public and private rank was just **0.236**, whereas strict nested CV ranked private submissions with a Spearman correlation of **0.991** (vs. **0.793** for Public LB).

| Public Rank | Team / Competitor | Public LB | Private LB | Private Rank | Key Paradigm |
| :---: | :--- | :---: | :---: | :---: | :--- |
| #3 | **Chris Deotte (`@cdeotte`)** | 0.94705 | **0.94602** | 🥇 **#1** (+2) | 3-leg Multi-Agent Relay, 60-model `cuML` LR stack (`0.94697` CV): Target-First GPT-2, Income EM, BPE LRs |
| #1 (fitted) | **Team Alicia (URAD)** | 0.94691* | **0.94588** | 🥈 **#2** (-1) | Full-context TabPFN-3.5 (428k–668k rows), `distilgpt2` LLR features, AUC-Direct FFT convolution ensembling |
| #9 | **M & M** (`Masaya Kawamata` & `mahog`) | 0.94692 | **0.94584** | 🥉 **#3** (+6) | GLM offset + Residual LightGBM (`0.946502`), Soft Pseudo-Label Distillation ($w=2$), DistilGPT-2 BPE tokens |
| #8 | **Prior** | 0.94694 | **0.94583** | 🏅 **#4** (+4) | — |
| — | **Will Guesdon & Aryan Kaisth** | 0.94687 | **0.94581** | 🏅 **#6** | Claude Code worker + Codex CLI auditor, 2-fold fast screen, SFT GPT-2 classifiers, 43-model $L_1$ SAGA stack |
| #6 | **Paul Bryan Elefante (`@heuljax`)** | 0.94697 | **0.94580** | 🏅 **#7** (-1) | GPT-2 BPE income tokenization + GPU $L_2$ Logistic Regression (`0.946400` OOF, $\rho \approx 0.989$ vs GBDTs), GAM margins |
| — | **yuurei** | 0.94674 | **0.94575** | 🏅 **#8** | 4-level Target Encoding Pyramids, Residual XGBoost on 622-col LR margins (`0.946617`), 11 subgroup models, `tol=1e-8` |
| — | **Xin Feng (`@milanfx`)** | 0.94684 | **0.94579** | 🏅 **#10** | Adversarial Claude + ChatGPT cross-examination (`0.94681` OOF) |
| #25 | **Ravi Ramakrishnan & Don Mani** | 0.94683 | **0.94574** | 🏅 **#12** (+13) | Three distinct feature views + GLM margin LightGBM + plain GBDT anchor |
| #2 | Gigrise | 0.94705 | 0.94564 | #101 (-99) | Overfit 20% Public split |
| #5 | Daniel Mineev | 0.94698 | 0.94565 | #79 (-74) | Overfit 20% Public split |
| #10 | CHEN Xiang | 0.94691 | 0.94551 | #213 (-203) | Overfit 20% Public split |

*\*Team Alicia's selected Final B scored `0.94691` Public; an unselected Public-LB-fitted probe reached `0.94945` Public (#1) but collapsed to `0.94313` Private (#1646).*

### Team Alicia's Overfitting Law
By measuring controlled Public-LB tilting experiments, Team Alicia established a tight linear penalty governing leaderboard fitting:
$$\Delta\text{Private} \approx +16.7\text{u} - 0.88 \times \Delta\text{Public}, \qquad r = -0.97$$
Every $1\text{u}$ ($10^{-5}$) of Public LB gained via LB-fitting cost nearly $0.9\text{u}$ on Private LB. Honest submissions maintained a near-constant Public-minus-Private offset of ~105u ($0.00105$).

---

## 2. Generator Forensics & Deterministic Invariants

1. **Autoregressive LM Synthesizer (GReaT / GPT-2)**:
   - Chris Deotte and Team M & M proved the dataset was synthesized by a GReaT-style GPT-2 generator fine-tuned on the 10,000-row original seed dataset and sampled with a label-first prompt.
   - Incomes from `$50,000` to `$99,999` ending in `< 200` dropped from `19.8%` (original) to `7.61%` (synthetic train); a local `distilgpt2` pilot reproduced `7.57%` ($r = 0.942$ histogram match).
   - **98% Income Memorization & Super-Binomial Variance**: 98% of synthetic rows copied exact original integer incomes (`$30,000`–`$223,000`), with empirical adoption variance $3.7\times$ larger than binomial noise due to partial record memorization (`City` $1.32\times$, `Charging_Stations` $1.30\times$, `Current_Car_Type` $1.17\times$).
2. **Simpson's Paradox in Charging Infrastructure**:
   - Aggregate `Charging_Stations_Near_Home` showed near-zero target correlation, but conditioning on `Home_Charging_Possible` inverted the sign (strongly positive when `No`; redundant when `Yes`).
3. **Four Deterministic Generator Boundary Invariants (3,733 Test Rows, 100% Train Precision)**:
   - `Annual_Income_USD >= 170537` $\to$ 100% positive (`3,248` test rows).
   - `31004 <= Annual_Income_USD <= 41970` $\to$ 100% negative (`461` test rows).
   - `Daily_Commute_km >= 83.0` $\to$ 100% negative (`18` test rows).
   - `Annual_Income_USD == 30000` & no subsidy & (`Environmental_Concerns_Score == 1` | `Daily_Commute_km > 70`) $\to$ 100% negative (`6` test rows).

*(Reusable post-processing code for Riemannian $\mathbb{S}^{N-1}$ Fréchet Barycenter ensembling, deterministic boundary overrides, and zero-tie `lexrank` sorting is codified in [`.agent/skills/kaggle-tabular-playbook/references/riemannian_hypersphere_frechet_blend.md`](../../.agent/skills/kaggle-tabular-playbook/references/riemannian_hypersphere_frechet_blend.md); AUC-direct FFT convolution ensembling is in [`auc_direct_fft_blend.md`](../../.agent/skills/kaggle-tabular-playbook/references/auc_direct_fft_blend.md).)*

---

## 3. 1st Place Solution Deep Dive — Chris Deotte (`@cdeotte`, `0.94602` Private)

### 3.1 The Three-Leg Multi-Agent Relay
Instead of one isolated run, Chris Deotte passed a shared codebase and structured markdown memory (`plan.md`, `progress.md`, `discoveries.md`, closed-ideas list, and S6E3–S6E8 cookbooks) across three generations of AI agents on one fixed 5-fold `StratifiedKFold(5, shuffle=True, random_state=42)` split:

| Leg | Agents | Timeline | CV | Public LB | Private LB | Key Contribution |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Leg 1** | `GPT-5.6 Sol` + `Fable 5` | Sep 2–3 (2 days) | 0.94656 | 0.94672 | 0.94568 | Ported S6E3–S6E8 cookbooks onto shared `folds.npy` |
| **Leg 2** | `GPT-6-Astra` + `Fable 5.1` | 2–3 days | 0.94683 | 0.94701 | 0.94593 | Uncovered original purchase recipe & core feature banks |
| **Leg 3** | `Opus 5.5` (19 helper subagents) | Sep 24–30 | **0.94697** | **0.94705** | **0.94602** | Ingested 1.8M LOC + 1,300 notes in 1 day; created Target-First GPT-2 classifier |

- **Score-Hiding Trick ($1 - p$)**: To probe Public LB without alerting rivals early in the month, Chris submitted inverted predictions `1 - p` (scoring `1 - 0.94701 = 0.05299`, which never updated the team's displayed leaderboard best).

### 3.2 Open-Source 60-Model Architecture (`s6e9_ev_purchases`)
All 60 models released in [`s6e9_ev_purchases`](https://github.com/cdeotte/kaggle-playground-solutions/tree/main/s6e9_ev_purchases) stack via `cuML LogisticRegression` ($L_2$) on OOF logits to reach **`0.94697` CV / `0.94602` Private**:

| Model Family (`models/` folder) | Standout Pairs & Techniques | Peak OOF AUC |
| :--- | :--- | :---: |
| **`combo_xgb` & `income_mixture_xgb`** | `combo_xgb_32seeds` (combined feature banks + EM blocks, 32 seeds); `xgb_full_context_contrasts` (16 seeds); `xgb_income_mixture` (exact-income class-mixture EM + commute/age EM) | **0.946574** |
| **`te_bank_lr`** | Elastic-net logistic regression (`te_bank_lr`) & RealMLP (`te_bank_realmlp`, `0.946442`) on nested target-encoding banks | **0.946499** |
| **`gpt2_target_first` / `chains` / `distillation`** | GReaT replicas (`e55..e70, d60`) fine-tuned 12 epochs label-first with Bayes LLR readout (`gpt2_target_first_12ep`: **`0.946485`**); XGBoost-distilled GPT-2 `" Yes"/" No"` head (`gpt2_native_kd4`: **`0.946447`**); all-target chains (`0.946107`); `gpt2_yes_no` (`0.945639`) | **0.946485** |
| **`lr_margin_xgb`** | `xgb_lr_margin_rte_art_bpe`: XGBoost with in-fold LR `base_margin` + residual encodings vs rule prob + digit/modulus artifacts + GPT-2 BPE income pieces | **0.946462** |
| **`response_features`** | `xgb_acceptance_entropy` (472 response-geometry cols + BPE-prefix encoding + 52 replica entropy cols); `xgb_response_geometry` (`0.946242`) | **0.946444** |
| **`nested_te_xgb` & `nested_te_realmlp_tabm`** | 203-channel nested TE library + periodic bases + 208 income response deltas + GPT-2 BPE income columns (`xgb_te_periodic_bpe`: `0.946442`; `realmlp_tabm_te_bpe`: `0.946250`) | **0.946442** |
| **`te_mlp_lr`** | Wide MLP (`mlp_pair_te_rule_wide_bpe`, `0.946181`) & `cuML` LR (`0.945957`) on pair encodings + rule offsets + BPE pieces | **0.946181** |
| **`neural_contrasts`** | 7 deliberate contrast models: `copula_gam_contrast` (`0.945777`), `copula_gpt2_contrast` (`0.944802`), `income_mixture_contrast` (`0.942705`) | **0.945777** |
| **`public_models`** | Ported onto shared folds: `@heuljax` XGB (`0.946190`) & LR (`0.946152`), `@najiama` LGBM (`0.945870`), `@kirill0212` LGBM (`0.945832`) | **0.946190** |

---

## 4. Other Top-12 Winning Solutions Distilled

1. **Team Alicia — 2nd Place (`0.94588` Private): Full-Context TabPFN-3.5 & AUC-Direct FFT Ensembling**
   - **Full-Context TabPFN-3.5 (`0.946485` pooled AUC)**: Scaled `tabpfn==9.0.0` (`tabpfn-v3.5-20260909.safetensors`, `fit_mode='fit_with_cache'`) to **428k–668k context rows** without subsampling, establishing a log-linear **Context Law of `+18.6u` per doubling of context rows**.
   - **Zero-Competition-Label Surrogate LLR**: Fine-tuned `distilgpt2` strictly on the 10k original seed rows and scored $\text{LLR} = \log p(\mathbf{x}\mid \text{Yes}) - \log p(\mathbf{x}\mid \text{No})$ across 4 column permutations (`+10.48u` stack gain).
   - **AUC-Direct Level-3 FFT Ensembling**: Minimized the smooth pairwise sigmoid ranking loss over all $2.6\times 10^{10}$ pairs in $O(n + N\log N)$ via `scipy.signal.fftconvolve` (`+1.62u` over log-loss stacking).

2. **Team M & M — 3rd Place (`0.94584` Private): GLM Offset, Soft Distillation & BPE Tokens**
   - **GLM Offset + Residual LightGBM (`0.946502` OOF)**: Passed logits from a 316-feature cross-fitted GLM (`0.946326`) as `init_score` to LightGBM, lifting all 5 folds by `+0.00018`.
   - **Continuous Soft Pseudo-Label Distillation**: Added all 286,571 test rows scored by validation-excluded teachers (`sample_weight=2`). Hard binary $\{0,1\}$ pseudo-labels *degraded* CV.
   - **GPT-2 BPE Income Tokenization**: Target-encoding BPE subword prefixes/suffixes of income strings (`50000` $\to$ `Ġ5` + `0000`) added `+0.000266` CV. Combined 115 columns via $L_2$ Logistic Regression (`C=0.3`).

3. **Will Guesdon & Aryan Kaisth — 6th Place (`0.94581` Private): Dual-Agent Auditor & SFT GPT-2**
   - **Cross-Family Worker/Auditor Loop**: `Claude Code` executed experiments while read-only `Codex CLI` enforced 8 strict anti-leakage checks. Switching from slow 5-fold exploration (`0.94644` plateau) to 2-fold fast screening $\to$ nested 5-fold confirmation lifted CV from `0.946365` to `0.946771` in 3.5 days.
   - **SFT GPT-2 Row Classifiers**: Serialized rows (65.7 tokens named / 29.7 tokens values) trained end-to-end with a linear binary head (`BCE`, `lr=5e-5`, L40S GPUs) added `+0.00004` Private LB (whereas frozen GPT-2 embeddings collapsed to `0.9040` AUC).
   - **43-Member $L_1$ SAGA Rank Stack & Public Hedge Failure Law**: `LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=3000)` on normalized ranks; **15/15 public blend hedges scored lower on Private LB than the honest base stack**.

4. **yuurei — 8th Place (`0.94575` Private): Multi-Granularity Pyramids & Solver Tolerance**
   - **4-Level Quantization TE Pyramid** (raw income, `$100`-rounded, `$1000`-rounded, floored commute; removing it dropped OOF by `0.00019`).
   - **Residual XGBoost (`0.946617` OOF)** initialized with 622-feature $L_2$ Logistic `base_margin` (`0.946509`) + **11 Subgroup-Partitioned XGBoost models** on city/car/age slices (`+0.000016` across all 10 folds).
   - **Stacking Convergence Fix**: Tightening `LogisticRegression` tolerance from default `tol=1e-4` to **`tol=1e-8`** (`max_iter=2000`) prevented premature stopping on 50+ collinear logit columns.

5. **Paul Bryan Elefante (`#7`, `0.94580`), Xin Feng (`#10`, `0.94579`) & Ravi/Don Mani (`#12`, `0.94574`)**
   - Paul Bryan Elefante pioneered GPT-2 BPE income tokenization + GPU $L_2$ Logistic Regression (`0.946400` OOF, $\rho \approx 0.989$ against trees) and GAM base margins (`0.946309`).
   - Ravi & Don Mani split features into **3 orthogonal views** (broad engineered, rolling window rates, donor stats) initialized with GLM logits and anchored with plain LightGBM/XGBoost (`20%` each).
   - Xin Feng paired Claude and ChatGPT to cross-examine 100+ interaction hypotheses, discovering that overriding the agent's false "dead-end" warnings on residual/token features jumped OOF from `0.94640` to `0.94681`.

---

## 5. Critical Tabular Traps & Anti-Patterns Codified

1. **The Neighbor-Reading Attention Leak (TabPFN / KNN Trap)**: Feeding cross-validated OOF predictions into in-context models (TabPFN) or KNN meta-learners creates an artificial `+162u` dev leak because attention reads context rows whose OOFs already reflect target labels.
2. **Nonlinear Meta-Learners Overfit**: Replacing linear/convex/geodesic stackers with monotone LightGBM, varying-coefficient models, or KNN smoothers lost `-2.7u` to `-52.0u` on private evaluation.
3. **Hard Binary Pseudo-Labels Amplify Noise**: Always use continuous soft teacher probabilities ($p \in (0, 1)$, excluding validation folds) with sample weighting rather than hard $\{0, 1\}$ thresholds.
4. **Frozen LLM Embeddings Collapse**: Passing frozen GPT-2 text embeddings into GBDTs dropped AUC to `0.9040`; LLMs must either be end-to-end supervised fine-tuned or scored generatively via Bayes log-likelihood ratio.
5. **Stacking Solver Premature Stop & Matrix Inversion Instability**: Default `tol=1e-4` stops early on 50+ correlated logit columns (use `tol=1e-8`), and analytical Lagrange multiplier matrix inversion explodes on singular correlation matrices (use regularized SAGA/L-BFGS/QN).
6. **Chasing Public Hedges & Candidate Pooling**: Averaging diverse models into one unweighted pool before blending drives mutual correlation $>0.999$, and blending public LB submissions injects `~12u–13u` optimism bias that underperforms honest stacks on Private LB.
