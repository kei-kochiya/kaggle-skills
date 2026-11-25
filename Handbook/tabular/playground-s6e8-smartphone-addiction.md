# Playground Series S6E8: Predicting Smartphone Addiction — Top Solutions, Code Deep-Dive & Post-Mortem

> **Competition**: [Kaggle Playground Series Season 6 Episode 8](https://www.kaggle.com/competitions/playground-series-s6e8) (Aug 1 – Sep 1, 2026 | 3,531 Teams | 33,982 Submissions)  
> **Target**: Binary `addicted_label` (`0`/`1`, Train Positive Rate `~70.94%`)  
> **Evaluation Metric**: Area Under the ROC Curve ($\text{ROC-AUC}$)  
> **Dataset**: `691,369` Train Rows × `14` Cols (`9` numeric + `3` categorical, heavy missingness `6%–19%`) | `296,302` Test Rows × `13` Cols | Equivalent Original Seed: `7,500` Rows ([Smartphone Usage & Addiction 7500](https://www.kaggle.com/datasets/jayjoshi37/smartphone-usage-and-addiction-prediction))  
> **1st Place Open-Source Code**: [`cdeotte/kaggle-playground-solutions/s6e8_smartphone_addiction`](https://github.com/cdeotte/kaggle-playground-solutions/tree/main/s6e8_smartphone_addiction) (102 models scoring **`0.97171` Private**, `+0.00048` over #2; single `realmlp_forensic_repair20` scores **`0.97145` Private**, beating #2 solo!)

---

## Official Top Standings & Solution Index

Chris Deotte swept **#1 Public (`0.97207`) and #1 Private (`0.97176`) by a massive `+0.00053` AUC margin** over 2nd place—driven by a target-free **4-model Masked Lattice Density Transformer** turning missing numeric cells into full conditional target-rate distributions (`+36.3u` single jump!):

| Private Rank | Public Rank | Δ Rank | Team / Competitor | CV (OOF AUC) | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| 🥇 **#1** | #1 | 0 | **Chris Deotte** (`cdeotte`) | **0.97101** *(102 repo)* · 0.97099 *(468 sub)* | **0.97207** *(0.97203 repo)* | **0.97176** *(0.97171 repo)* | **4-phase Distributed Multi-Agent Pipeline**: **Global 4-state Masked Lattice Density Transformer** distributional imputation + pair-lift residuals + repair posteriors → `realmlp_forensic_repair20` (`0.97071` CV / **`0.97145` Priv solo!**) + 102/468-model `cuML L2` LR logit stack | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/1st-place-distributed-intelligence-nvidia-infe) · [GitHub Repo](https://github.com/cdeotte/kaggle-playground-solutions/tree/main/s6e8_smartphone_addiction) |
| 🥈 **#2** | #3 | +1 | **MILANFX** (Xin Feng) | **0.97055** | 0.97149 | **0.97123** | **Screen-time budget reconstructions** (`fake_daily = social+work+game`, `fake_social`, `fake_work`, `fake_game`) + systematic `RAW_TE/Ratio/Digit/Round/Bins` across 200–400 own models + >200 public OOFs → LR blender | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/2nd-place-solution) |
| 🏅 **#7** | #6 | -1 | **Keanan** (`citerne`) | **0.970879** *(10-fold)* | 0.97139 | **0.97112** | Frozen 10-fold CV over **556 streams**: 9-col exact-value TE (`+0.00191` XGB) + **Rank-LogReg C-bag** (`C ∈ {0.01..3}`) + negative residual correctors + Gaussian-copula hedge | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/7th-place-many-models-one-simple-stack) |
| 🏅 **#14** | #13 | -1 | **Taemyung Heo** (`tmheo74`) | **0.97038** *(nested)* | 0.97135 | **0.97109** | 36 own + 278 shared OOFs: **Lookup-Transformer** (`0.96941`, **Muon** optimizer) + **3× cell-masking missingness aug (`p=0.25`)** + RealMLP `float32` vocab bug fix (`+0.00461`) + placebo-gated FE | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/14th-place-278-shared-oof-sets-36-own-models) |
| 🏅 **#25** | #18 | -7 | **Ravi Ramakrishnan** (`ravi20076`) | 15-fold SKF | 0.97133 | **0.97106** | >600-feature store + >400 candidates (`RealMLP` best solo, `TabM`, `RepleafGBM`, `CNN`, `AutoInt`, `YDF`) → Logistic Regression stacker | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/public-18-private-25-approach) · [L2 Stack NB](https://kaggle.com/code/ravi20076/playgrounds6e8-public-l2stack-v1) |
| 📉 **#153** | #27 | **-126** | **`yukiZ`** (`hideyukizushi`) | — | 0.97131 | — | Shakeup case study: hyper-reactive micro-session flags (`app_opens/hr`), dynamic soft labeling (`|y-oof| > 0.80`), & 2D quadrant rank-logit blending | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/lb-27thpb-153th-sol-why-low-screen-time-users) |

---

## 1. Data Forensics: Missingness Drift, Exact Grid Artifacts & Screen-Time Budget

1. **"Missingness Is the Entire Drift" (`Dariush Afshar`, Discussion 732427 & 733214)**:
   - Every single one of the 12 columns had statistically different missingness rates between train and test (`z` from `+11.1` to `-44.1`, e.g., `social_media_hours` `19.38%` train vs `16.00%` test; `app_opens_per_day` `11.67%` vs `8.68%`).
   - Adversarial validation (`987,671` rows): **`0.564146` raw**, **`0.565120` on `is_missing` flags only**, **`0.503339` after imputation**, and **`0.498486` (95% CI `[0.4962, 0.5006]`) on complete rows**! Missingness itself was target-independent (`n_missing` AUC = `0.50172`), proving **zero underlying covariate value drift** between train and test.
2. **Exact-Value Grid & First-Decimal Swing**:
   - Numerics sat on discrete repeated grid points with stable non-monotonic target rates (`daily_screen_time_hours` first decimal digit swung **8.5 percentage points** from `.0` [`0.6513`] to `.2` [`0.7365`], while the second decimal carried zero signal).
   - Keeping **both numeric values AND exact-value categorical copies** jumped LightGBM from `0.96276 → 0.96605` (`+0.00329`), whereas dropping numerics for all-categoricals crashed `-0.00417` (#14).
3. **4-Term Linear Screen-Time Budget Constraint**:
   - Observed complete rows satisfy `daily_screen_time_hours >= social_media_hours + gaming_hours + work_study_hours` — specifically concentrating at **`(social + gaming + work) / daily ∈ [0.85, 0.95)`** (evidence of a generator post-repair step!). Because axis-aligned trees struggle on 4-term oblique planes, reconstructing missing components (`fake_daily = social + work + game`, `fake_social = daily - work - game`, etc.) added **`+0.00084` to `+0.00095`** across all tree depths (#2, #14).
4. **Smeared Original Rule & Inert Categoricals**:
   - In the 7,500-row original seed, `daily > 8 OR social > 4 → 1` and `daily <= 6 AND social <= 4 → 0` reached `0.9888` AUC (`6 < daily <= 8` was `0.50` pure coin flip). S6E8's generator smeared the hard rule (`0.835` AUC alone) while injecting smooth signal into the middle band (`0.896` AUC). Meanwhile `gender`, `stress_level`, and `academic_work_impact` had near-random univariate AUCs (`0.5019–0.5119`).

---

## 2. 1st Place Code Deep-Dive: `s6e8_smartphone_addiction` (`0.97171` Priv Repo / `0.97145` Solo!)

Inspecting Chris Deotte's open-source [`s6e8_smartphone_addiction`](https://github.com/cdeotte/kaggle-playground-solutions/tree/main/s6e8_smartphone_addiction) codebase (`models/realmlp_repair/README.md` & `stack/stack.py`) reveals the exact 6-step cumulative chain that took stock PyTabKit `RealMLP_TD_Classifier` (`[768, 384, 192]`, PBLRD embeddings `(16, 8)`, Mish, dropout `0.05`, wd `0.017`, label smoothing `0.005`, 32 internal members, 9 epochs + full outer-train refit) to **solo Private #1 (`0.97145`)**:

| Chain Step | Pair Name | Feature Block Added (Cumulative Cols) | 5-Fold OOF AUC | Δ OOF |
| :---: | :--- | :--- | :---: | :---: |
| **1** | `realmlp_forensic_base` | Raw + exact-value tokens + 19 screen-budget geometry cols + 95 forensic fold cols (nested TEs, PMI, original LLRs/priors) (`135` cols) | `0.970055` | — |
| **2** | `realmlp_forensic_density_completion` | `+ 10` **per-fold Masked Density Transformer completion aggregates** (`145` cols) | `0.970106` | `+5.1u` |
| **3** | `realmlp_forensic_reconstruction` | `+ 71` cross-fitted GPU XGBoost (`900` rds, `d=8`) + Masked Transformer AE reconstruction channels (`216` cols) | `0.970214` | `+10.8u` |
| **4** | `realmlp_forensic_pairlift` | `+ 10` **exact pair-lift residuals** against log-odds-independence baseline (`226` cols) | `0.970299` | `+8.5u` |
| **5** | `realmlp_forensic_global_density` | **Global 4-state train+test Masked Density Mixture** (`g4`, `g5`, `g4orig8`, `g9unitarget`, 60–80 ep, per-field tempered $T \in \{1.0, 1.1, 1.2\}$) → rebuilt completion aggregates, `24` surprisal/entropy + `27` fills + coarse/derived lift banks + `39` generator-temp channels (**dropped all 71 XGB/AE reconstructions!** `259` cols) | **`0.970662`** | **`+36.3u`** |
| **6** | **`realmlp_forensic_repair20`** | `+ 20` **repair-posterior channels** (conditional mass & z-shift inside `(social+gaming+work)/daily ∈ [0.85, 0.95)` under the global mixture; `279` cols) | **`0.970706`** *(Priv **`0.97145`**)* | `+4.4u` |

### Why Steps 2, 4 & 5 Are Major Universal Tabular Breakthroughs
1. **Probabilistic Distributional Imputation via Target-Free Masked Lattice Density Transformer**: Instead of filling a missing cell with one scalar point estimate, mask the column and integrate its **full conditional distribution $P(x_j = v \mid \mathbf{x}_{-j})$** against a fold-safe exact-token target-rate table (`splitmix64(id) % 5` inner exclusion), yielding exact **expected completion logit, std, surprisal, and entropy**. Replacing point reconstructions (Step 3) with a 4-model tempered `train + test` density mixture (Step 5) delivered **`+36.3u` OOF in a single step**!
2. **Pair-Lift Residuals Against Log-Odds Independence**: For feature pairs $(a, b)$, compute the shrunk residual of the pair's empirical target odds minus the log-odds-independent product of its marginals inside inner `StratifiedKFold(5)`, isolating pure second-order synergy without marginal double-counting.
3. **Stacker (`stack/stack.py`)**: Auto-detects probabilities (`[0, 1]`) vs already-logit pairs, converts to logits clipped `[-30, 30]` (`float32`), and runs 5-repeat averaged **`cuML LogisticRegression(penalty='l2', C=1.0, max_iter=2000, tol=1e-7, solver='qn')`** (or standardized `sklearn lbfgs`, `tol=1e-8`) across 102 diverse pairs (`XGB/LGBM/CatBoost` density-margin + `RealMLP` + `TabM` + `Lookup-Transformer` + `DCNv2` + `YDF` + `cuML RF/SVC/RFF` + `NODE` + `AFN` + `RuleFit` + `EBM` + `TabPFN-v3` + 25 public notebooks) → **`0.971014` CV / `0.97171` Private**.

---

## 3. Additional Winning Highlights (#7, #14) & Engineering Traps

- **The `float32` Vocabulary Cast Bug (`+0.00461` AUC, #14 Taemyung Heo)**: Building an exact-value string/float lookup vocab from `float64` then querying after casting features to `float32` silently mapped **`800,896` validation cells to `UNK`** across 6 fractional columns! Casting *after* lookup dropped `UNK`s to `23` and jumped RealMLP 3-seed OOF **`0.96371 → 0.96832`**.
- **3× Independent Cell-Masking Missingness Augmentation (`p = 0.25`, #14 & `Talha Tursun`)**: Tripling training folds with 2 copies having independent random `p = 0.25` cell masking (beating empirical row-mask sampling!) added `+0.000047` across 5/5 folds to tree/NN ensembles and `+0.0014` on missing-heavy rows for NNs.
- **Lookup-Transformer + `Muon` Optimizer (`0.96941` OOF, #14)**: 4-layer width-128 Transformer combining exact-value lookup embeddings + rank-gauss PLR embeddings trained with `Muon` optimizer + OneCycle + EMA was #14's single strongest architecture.
- **Rank-LogReg C-Bag + Negative Residual Correctors (#7 Keanan)**: Averaging ranks across 6 cross-fitted Logistic Regressions (`C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3}`) over 556 streams + subtracting `25%` of a public-only stack and `10%` of a Rank-Gauss stack lifted 10-fold OOF `0.970820 → 0.970849`.

---

## 4. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Source | Root Cause |
| :--- | :--- | :---: | :--- |
| **Out-of-CV Target Encoding Leakage** | Inflated CV, collapsed LB | `E散人chloeprice`, `broccoli beef` | Cross-fitting TE before outer CV leaks $y_k$ into $X_{-k}$ |
| **`float64` Vocab Queried With `float32` Numerics** | **`-0.00461` OOF AUC** (`800k+` silent `UNK`s) | #14 (`tmheo74`) | IEEE-754 float32 rounding alters decimal keys relative to float64 vocab dicts |
| **Replacing Numerics With All-Categoricals** | **`-0.00417` OOF AUC** | #14 (`tmheo74`) | Discards ordinal geometry; always keep numeric + exact-value categorical copies side by side |
| **Tree/NN Meta-Blenders (`XGB`, `CatBoost`, `RealMLP`, `AutoGluon`, `Ridge`)** | Underperformed L2 Logit Logistic Regression | #2, #25 | Linear L2 Logistic Regression on logits generalizes vastly better across 100–500 collinear streams |
| **Splitting & Blending Separately by Missingness Count** | Hurt / no gain | #2 (`MILANFX`) | Missingness was target-independent (`AUC 0.5017`); splitting starves fold sample size |
| **Naive 2-Column Ratios Before Tuning Tree Capacity** | `-0.00046` once `num_leaves` raised `15 → 31` (`+0.00773`) | `WOWTIMWOW` | Under-parameterized trees create illusory ratio gains (`18×` smaller than proper capacity) |
| **Hard-Coding Original Threshold Rules / Pseudo-Labeling / DAE Latents** | Hard rule `0.835` AUC; PL hurt LGBM (Tilii); DAE `-0.00016` CB (#14) | #14, Tilii, EDA | Synthetic smearing invalidated hard cutoffs; DAE latents underperformed exact masked density completion |
| **Blindly Trusting High-LB Public "NN" Blend Notebooks** | Many gave NN `1e-6` weight and cloned existing CSVs; Public #27 fell to Private #153 (`-126`) | `Naji`, #153 | Audit every external OOF for leakage, weight integrity, and 5/5 fold nested OOF lift |
