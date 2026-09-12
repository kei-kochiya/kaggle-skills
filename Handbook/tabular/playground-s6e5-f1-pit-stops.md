# Playground Series S6E5: Predicting F1 Pit Stops

**Competition**: [Kaggle Playground Series - Season 6 Episode 5](https://www.kaggle.com/competitions/playground-series-s6e5)  
**1st Place Solution**: *1st Place: By the skin of my teeth* by `@optimistix`  
**2nd Place Solution**: *2nd Place - Autonomous Codex Yolo!* by Chris Deotte (`@cdeotte`, Senior Data Scientist at NVIDIA)  
**Track**: Tabular Binary Classification  
**Evaluation Metric**: Area Under the ROC Curve ($\text{ROC-AUC}$)  
**Official Reference Notebooks**: [EDA - Predicting F1 Pit Stops](https://www.kaggle.com/code/cdeotte/eda-predicting-f1-pit-stops) by `@cdeotte` & [PS|S6|E5: RealMLP · PyTabKit](https://www.kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytabkit) by `@yekenot`  
**Original Source Dataset**: [F1 Strategy Dataset | Pit Stop Prediction](https://www.kaggle.com/datasets/vanshjasuja16/f1-strategy-dataset-pit-stop-prediction) (`f1_strategy_dataset_v4.csv`, 101,371 rows)

---

### Official Winning Scoreboard

| Rank / Submission | Architecture & Setup | Cross-Validation (CV) | Public LB | Private LB | Outcome / Strategic Significance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 🥇 **1st Place (`@optimistix`)** | **50-50 Blend of AutoGluon + 186-OOF Logit Stack** (182 L1 + 4 L2 OOFs, Driver-dropped models, sample weights 0.5–1.0) | — | 0.95488 | **0.95503** | **Won by +0.00001 in literally the last minute!** Exploited adversarial Driver-drift mitigation & AutoGluon/Logit ensembling. |
| 🥈 **2nd Place (`@cdeotte`)** | **Autonomous Codex GPT-5.5 YOLO** (218 models across 37 architectures, cuML $L_2$ Logistic Regression on logits) | 0.9552+ | 0.95487 | **0.95502** | Missed 1st by 0.00001. An unselected morning submission scored **0.95506** (would have taken 1st). Autonomous agent executed end-to-end on 4× A100 GPUs. |
| 🏅 **5th Place Post-Deadline (`@703572`)** | **99-Model Logit Stack + Post-Deadline Rank Average** | 0.95536 | 0.9548+ | **0.95505** | Rank-averaging Hill-Climbing and Logit-Stacking (`rankdata(p)/N`) outscored raw probability averaging (0.95503). |
| 🏅 **8th Place (`@703539`)** | **L5 Multi-Split Ensemble** (Averaging 5-fold, 7-fold, and 10-fold L4 ensembles across 147 models) | 0.95503–0.95510 | 0.95462 | **0.95487** | Exploited split diversity (5-fold, 7-fold, 10-fold) paired with original-row vs. original-column feature policies. |
| **Top Single Model: RealMLP** (`realmlp2_exp147`) | PyTabKit RealMLP-TD Classifier (5 seeds, PLR embeddings, SiLU) | **0.954426** | 0.95382 | 0.95421 | Clear individual champion model across all architectures. |
| **Top Single Model: XGBoost** (`gpt1020_xgb`) | Deep bagging with Tweedie/hazard loss formulation | **0.953553** | 0.95294 | 0.95354 | Dominant tree-based baseline. |
| **Top Single Model: CatBoost** (`gpt1016_cat`) | Ordered boosting with Categorical Target Encoding | **0.953404** | 0.95105 | 0.95190 | Strongest on raw interaction categories. |
| **Top Single Model: TabM** (`tabm_exp089`) | Multi-head tabular neural net with batch ensembling | **0.953371** | 0.95304 | 0.95345 | High tree-competitive neural architecture. |
| **Top Single Model: LightGBM** (`lgbm_exp091`) | GOSS / adaptive learning rate / DART | **0.953023** | 0.95267 | 0.95290 | Fast backbone workhorse. |
| **Top Single Model: TabICLv2** (`pri589_tabicl`) | In-context tabular foundation model | **0.950827** | 0.95053 | 0.95085 | High ensemble decorrelation weight. |

---

## 1. Executive Summary: The Winning Paradigms

Playground Series S6E5 tasked competitors with predicting whether a Formula 1 driver would enter the pit lane on the very next lap (`PitNextLap`) using telemetry, race position, tyre degradation, and strategy indicators. The competition culminated in one of the tightest finishes in Kaggle history, decided by **0.00001 AUC**, while establishing two transformative competitive paradigms:

### 1.1 Autonomous Codex YOLO: Hands-Off Multi-GPU Agent Scaling
For the 2nd place solution, Chris Deotte deployed **OpenAI Codex (GPT-5.5)** in full autonomous execution mode (`--yolo`) on a 4× NVIDIA A100 GPU cluster.
- **Autonomous Feedback Loop**: Given access to a single baseline notebook and a tracking file (`local_leaderboard.md`), Codex autonomously designed hypotheses, authored code, ran GPU training jobs, recorded CV scores, inspected residuals, and branched further experiments without human intervention.
- **Scale and Taxonomy**: In mere hours, the agent generated **218 diverse, evaluated models** spanning **37 model classes** (from GBDTs and Tabular NNs to Graph Neural Networks, Factorization Machines, and Liquid Neural Networks), providing an unprecedented pool for meta-learning.

### 1.2 The Last-Minute 0.00001 Upset: Strategic Diversity & Driver Drift
The 1st place winner, `@optimistix`, clinched gold with a submission uploaded in the final minute:
- **Adversarial Driver-Drift Mitigation**: While adding the original FastF1 dataset provided a massive baseline lift, adversarial validation revealed that the `Driver` feature exhibited extreme domain shift between the original data and the synthetic test set. Models trained with `Driver` dropped decorrelated the ensemble and proved substantially more robust on the private leaderboard.
- **Dual Meta-Learner Blending**: Combining a heavy **AutoGluon** stack with an $L_2$-penalized **Logit Logistic Regression** stack (186 OOFs total) in a 50-50 consensus blend bridged local calibration artifacts and delivered the winning margin of victory.

```
                                  S6E5 WINNING FLOW ARCHITECTURE
                                  
+---------------------------+       +----------------------------+
|  Synthetic Data (439k)    |       |   Original FastF1 (101k)   |
|   Target Positive: 19.9%  |       |    Target Positive: 25.5%  |
+-------------+-------------+       +--------------+-------------+
              |                                    |
              +------------------+-----------------+
                                 |
              [Synchronous Fold-Aligned Split (5-Fold / 10-Fold)]
              [Sample Weights: 1.0 (Synthetic) vs 0.5-1.0 (Orig)]
                                 |
              +------------------+------------------+
              | Feature Engineering:                |
              | • Total Race Laps: Lap / RaceProg   |
              | • Tyre Degradation: TyreLife / Lap  |
              | • Discretization: 200 Quantile Bins |
              | • Interaction Target Encoding       |
              | • Driver-Dropped Model Stream       |
              +------------------+------------------+
                                 |
         +-----------------------+-----------------------+
         |                                               |
+--------v-----------------------+             +---------v-----------------------+
|  Tier 1: High-Capacity Models  |             |  Tier 2: Orthogonal Decorrelators |
|  • RealMLP (Pytabkit)  0.9544  |             |  • Deep FFM (AUC 0.918, ρ=0.852)|
|  • XGBoost (Tweedie)   0.9535  |             |  • GraphSAGE GNN (AUC 0.940)    |
|  • CatBoost (CTR TE)   0.9534  |             |  • Nyström RBF SVM (AUC 0.925)  |
|  • TabM (Multi-Head)   0.9533  |             |  • TabICLv2 Foundation (0.9508) |
|  • LightGBM (GOSS)     0.9530  |             |  • Liquid Neural Net (AUC 0.893)|
+----------------+---------------+             +-----------------+---------------+
                 |                                               |
                 +-----------------------+-----------------------+
                                         |
                               [Logit Transformation]
                               z = log(p / (1 - p)), clip [-30, 30]
                                         |
                         +---------------+---------------+
                         |                               |
                 +-------v-------+               +-------v-------+
                 | Logit Stacker |               | AutoGluon / HC|
                 | (cuML L2 Reg) |               |  Stacker OOFs |
                 +-------+-------+               +-------+-------+
                         |                               |
                         +---------------+---------------+
                                         |
                            [Rank-Average Consensus]
                         0.5 * (Rank_LS + Rank_AutoGluon)
                                         |
                                 🥇 1st Place LB
```

---

## 2. Dataset Architecture & Exploratory Data Analysis

### 2.1 Dataset Parameters

| Dataset Partition | Row Count | Column Count | Target Positive Count | Target Base Rate | Source / Generation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Competition Train** | 439,140 | 16 (inc. `id`, `target`) | 87,381 | **19.898%** | Synthetic (CTGAN/diffusion variant from original) |
| **Competition Test** | 188,165 | 15 (inc. `id`) | *Hidden* | *Estimated ~19.9%* | Synthetic holdout |
| **Original External Dataset** | 101,371 | 16 (with `Normalized_TyreLife`)| 25,829 | **25.480%** | Real FastF1 telemetry & race logs (v4) |

### 2.2 Feature Roster & Semantic Role

| Column | Raw Type | Physical / Strategic Meaning | Key Quirks & Distribution Dynamics |
| :--- | :--- | :--- | :--- |
| `Driver` | Categorical (String) | Driver identifier (e.g. `D109`, `D105`) | **Severe covariate shift**. Adversarial AUC ~0.70+ between original and synthetic test. |
| `Compound` | Categorical (String) | Tyre compound: `SOFT`, `MEDIUM`, `HARD`, `INTERMEDIATE`, `WET` | Pit hazard curves vary drastically by compound. Soft pits earliest; Hard stretches stints. |
| `Race` | Categorical (String) | Grand Prix circuit name (e.g. `Canadian Grand Prix`) | Dictates total race distance, degradation slope, and pit lane loss time. |
| `Year` | Numeric / Ordinal | Championship season ($2019 - 2024$) | Reflects changing aerodynamic and tyre regulations (e.g. 13-inch vs 18-inch wheels in 2022). |
| `PitStop` | Binary Flag ($0 / 1$) | Whether the driver pitted on the *current* lap | Consecutive pit stops are exceedingly rare unless wing damage occurs. |
| `LapNumber` | Numeric (Integer) | Current lap of the race ($1 - 78$) | Monotonically increases during each Grand Prix stint. |
| `Stint` | Numeric (Integer) | Stint counter for the car ($1, 2, 3, \dots$) | Stint 1 and 2 follow standard strategy; Stints 3+ indicate multi-stop or wet weather. |
| `TyreLife` | Numeric (Float) | Number of laps completed on the current set of tyres | Primary physical driver of degradation cliff. |
| `Position` | Numeric (Integer) | Current track position ($1 - 20$) | Drivers in traffic or defending track position alter pit windows to undercut. |
| `LapTime (s)` | Numeric (Float) | Elapsed lap time in seconds (e.g. $78.491$) | Correlates with track length and tyre drop-off. |
| `LapTime_Delta` | Numeric (Float) | $\text{LapTime}_i - \text{LapTime}_{i-1}$ | Spike indicates tyre degradation, traffic, yellow flags, or pit out/in laps. |
| `Cumulative_Degradation`| Numeric (Float) | Estimated total pace lost over the stint | Non-linear metric quantifying tyre drop-off. |
| `RaceProgress` | Numeric (Float) | Normalized race progression ($0.0 \to 1.0$) | In real racing: $\text{LapNumber} / \text{TotalLaps}$. Crucial anchor feature. |
| `Position_Change` | Numeric (Float) | Net position change from previous lap | Track battles; aggressive passing vs falling back. |
| `PitNextLap` | Binary ($0 / 1$) | **Target Variable**: Driver enters pits on next lap | Evaluation metric is ROC-AUC. |

### 2.3 Synthetic Generator Forensic Discoveries
1. **Broken Sequential Coherence**:
   In the real FastF1 dataset, each `(Driver, Race, Year)` is a contiguous physical time series. In the Kaggle synthetic dataset, rows were independently sampled and perturbed. The sequential identity $\text{LapTime\_Delta}_i = \text{LapTime}_i - \text{LapTime}_{i-1}$ and $\text{PitStop}_{i+1} = \text{PitNextLap}_i$ was broken for the majority of rows. Lag and autoregressive features suffered substantial signal loss compared to static physical interactions.
2. **Real Signal vs. Synthetic Artifacts**:
   Unlike Playground episodes S6E1 and S6E3, where modulo mantissas ($10^k \pmod{10}$) and synthetic snap features yielded massive gains, competitors verified that **digit decomposition and snap-to-grid features yielded no meaningful lift in S6E5**. The data retained genuine non-linear physical signals (tyre cliffs, stint degradation, circuit geometry) that required domain-grounded feature engineering rather than generator reverse-engineering.

---

## 3. The Feature Engineering Playbook

Feature engineering provided a decisive $+0.0015$ to $+0.0020$ AUC boost over raw feature baselines. The most effective transformations focused on **recovering latent physical constants** and **capturing multi-variable degradation interactions**.

```
+-----------------------------------------------------------------------------------------+
|                              S6E5 FEATURE ENGINEERING SUITE                             |
+-----------------------------------------------------------------------------------------+
| 1. Latent Total Laps      --> LapNumber / (RaceProgress + 1e-6) [Estimates Grand Prix Laps]|
| 2. Tyre Wear Ratio        --> TyreLife / LapNumber.clip(lower=1) [Stint vs Race Distance] |
| 3. Degradation Pace       --> Cumulative_Degradation / LapNumber [Degradation rate per lap]|
| 4. Thermal Pace Drag      --> LapTime * Cumulative_Degradation                            |
| 5. Normalized Degradation --> LapTime / (abs(Cumulative_Degradation) + 1e-6)              |
| 6. Quantile Discretization--> KBinsDiscretizer(200 bins) on RaceProgress                  |
| 7. Multi-Scale Categorical--> Floor + Factorize on all continuous columns                 |
| 8. Composite Bigrams      --> Race__Compound, Race__Year, Driver__Race                     |
| 9. Out-of-Fold TE         --> Nested 5-fold Target Encoding on composite interactions      |
| 10. Driver-Dropped Branch --> Parallel feature matrix omitting 'Driver' to counter drift   |
+-----------------------------------------------------------------------------------------+
```

### 3.1 Reverse-Engineering Latent Total Race Laps
Because `RaceProgress` was computed as $\frac{\text{LapNumber}}{\text{TotalLaps}}$, computing their ratio reconstructs the exact total laps of each circuit (e.g. 70 laps for Montreal, 53 for Monza, 71 for Spielberg):

$$\text{EstimatedTotalLaps} = \frac{\text{LapNumber}}{\text{RaceProgress} + 10^{-6}}$$

```python
import numpy as np
import pandas as pd

def add_race_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Reconstruct total scheduled race laps
    df['_LapNumber_/_RaceProgress'] = (df['LapNumber'] / (df['RaceProgress'] + 1e-6)).astype('float32')
    
    # 2. Tyre wear progression relative to race progress
    df['_TyreLife_/_LapNumber'] = (df['TyreLife'] / df['LapNumber'].clip(lower=1)).astype('float32')
    
    # 3. Degradation velocity (cumulative degradation per lap completed)
    df['_Degradation_Per_Lap'] = (df['Cumulative_Degradation'] / df['LapNumber'].clip(lower=1)).astype('float32')
    
    # 4. Pace drag: compound interaction between lap time and cumulative degradation
    df['_LapTime_*_Cumulative_Degradation'] = (df['LapTime (s)'] * df['Cumulative_Degradation']).astype('float32')
    df['_LapTime_*_Degradation_abs'] = (df['LapTime (s)'] * df['Cumulative_Degradation'].abs()).astype('float32')
    df['_LapTime_/_Degradation_abs'] = (df['LapTime (s)'] / (df['Cumulative_Degradation'].abs() + 1e-6)).astype('float32')
    
    return df
```

### 3.2 Quantile Discretization & Multi-Scale Binning
Tabular neural networks (RealMLP, TabM) and gradient boosting trees benefit heavily when continuous progress variables are discretized into high-resolution ordinal bins:
- `RaceProgress`: 200 quantile bins. In F1 strategy, pit stops occur in narrow windows (e.g. laps 18–24 and 42–48); 200 bins cleanly isolates the pit windows along the race timeline.
- `LapTime (s)`: 7 quantile bins, segregating out-laps, in-laps, traffic pace, clean air pace, and safety-car deltas.

```python
from sklearn.preprocessing import KBinsDiscretizer

def discretize_progress(train_df, test_df, orig_df=None):
    kb = KBinsDiscretizer(n_bins=200, encode='ordinal', strategy='quantile', subsample=None)
    
    train_df['RaceProgress_200_bin_'] = kb.fit_transform(train_df[['RaceProgress']]).ravel().astype(str)
    test_df['RaceProgress_200_bin_'] = kb.transform(test_df[['RaceProgress']]).ravel().astype(str)
    
    if orig_df is not None:
        orig_df['RaceProgress_200_bin_'] = kb.transform(orig_df[['RaceProgress']]).ravel().astype(str)
        
    return train_df, test_df, orig_df
```

### 3.3 Categorical Interactions & Nested Target Encoding
Strategy decisions are heavily conditioned on the specific pairing of track characteristics and tyre compounds:
- **`Race__Compound`**: Captures compound durability on specific track asphalt (e.g. Soft at Silverstone degrades twice as fast as Soft at Monaco).
- **`Race__Year`**: Encapsulates year-specific weather, safety cars, and tyre allocations.
- **Target Encoding Protocol**: Target encoding was strictly applied using out-of-fold estimation with smoothing to avoid leakage:

```python
from sklearn.preprocessing import TargetEncoder

def fit_transform_target_encoding(X_tr, y_tr, X_val, X_test, te_cols=['Race_Compound_', 'Race_Year_']):
    te = TargetEncoder(cv=5, smooth='auto', shuffle=True, random_state=42)
    
    tr_enc = te.fit_transform(X_tr[te_cols], y_tr)
    val_enc = te.transform(X_val[te_cols])
    tst_enc = te.transform(X_test[te_cols])
    
    for i, col in enumerate(te_cols):
        X_tr[f'_{col}TE'] = tr_enc[:, i].astype('float32')
        X_val[f'_{col}TE'] = val_enc[:, i].astype('float32')
        X_test[f'_{col}TE'] = tst_enc[:, i].astype('float32')
        
    return X_tr, X_val, X_test
```

### 3.4 The Driver-Dropped Adversarial Ablation
A core discovery of `@optimistix` (1st place) and confirmed by 11th place:
- **The Problem**: In synthetic data generation, the `Driver` distribution was heavily distorted. An adversarial classifier trained to distinguish train/original from test achieved an AUC $> 0.70$ when using `Driver`, but dropped to $\sim 0.50$ (random) when `Driver` was removed.
- **The Solution**: Build two parallel model pipelines:
  1. Standard pipeline with all features (including `Driver`).
  2. **Driver-dropped pipeline**: Completely exclude `Driver` from training.
- **Outcome**: The Driver-dropped models not only scored higher on private CV, but provided vital decorrelation when blended into the final meta-stacker.

---

## 4. Modeling & Architecture Zoo

The winning solutions established that single-model accuracy was dominated by **RealMLP**, but final leaderboard triumph required a massive, heterogeneous model zoo. Chris Deotte's 2nd place solution trained **218 models** spanning 37 model architectures:

### 4.1 The Big 6 Core Architecture Comparison

| Family | Model ID / Variant | OOF AUC | Public LB | Private LB | Optimal Architecture / Hyperparameter Secret |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RealMLP** | `realmlp2_exp147` (`pytabkit`) | **0.954426** | 0.95382 | 0.95421 | 3-layer MLP `[512, 256, 128]`, SiLU activation, Periodic Linear (PLR) embeddings (`plr_sigma=2.33`), dynamic bias init `neg-uniform-dynamic-2`, 5-seed average. |
| **XGBoost** | `gpt1020_xgb_orighazard` | **0.953553** | 0.95294 | 0.95354 | Deep tree structure (`max_depth=8`), `colsample_bytree=0.45`, `subsample=0.80`, Tweedie/hazard loss formulation, original data appended. |
| **CatBoost** | `gpt1016_cat_ctrte` | **0.953404** | 0.95105 | 0.95190 | `depth=7`, `learning_rate=0.035`, Ordered Target Statistics on `Race_Compound` and `Driver_Race`, `l2_leaf_reg=5.0`. |
| **TabM** | `tabm_exp089` | **0.953371** | 0.95304 | 0.95345 | Multi-head tabular neural network (32 batch ensemble heads), shared representation trunk, Mish activation, weight decay 0.01. |
| **LightGBM**| `lgbm_exp091` | **0.953023** | 0.95267 | 0.95290 | GOSS sampling (`top_rate=0.2, other_rate=0.1`), cosine annealing learning rate scheduler, max bin 512. |
| **TabICLv2**| `pri589_tabicl_v2` | **0.950827** | 0.95053 | 0.95085 | Tabular in-context foundation model fine-tuned on synthetic F1 telemetry embeddings. Zero feature engineering required. |

### 4.2 The Long Tail of Orthogonal Decorrelators

In the 5th place solution analysis of 4,851 model pairs:
- **Redundant Cluster**: The LightGBM family had a mean pairwise Spearman correlation of **$\rho \approx 0.968$** (several near-twins with $\rho > 0.998$). Adding a fourth LightGBM added virtually zero stacking value.
- **Diverse Decorrelator Cluster**: Models with lower individual AUC provided the largest marginal weight in the final meta-model:
  - **Deep FFM (Field-aware Factorization Machine)**: Single AUC $0.918$, but lowest mean pairwise correlation ($\rho = 0.852$). Provided the highest positive marginal contribution per slot in the meta-learner.
  - **GraphSAGE / GNN**: Formulated lap progressions as bipartite graphs between Drivers and Races. Single AUC $0.940$, decorrelation weight $+0.082$.
  - **Nyström Kernel Subsampled RBF SVM**: Kernelized non-linear boundaries. Single AUC $0.925$, correlation $\rho = 0.884$.
  - **Liquid Neural Networks (LNN) & Recurrent GRU**: Sequential modeling over lap numbers. Single AUC $0.893–0.941$, capturing temporal inertia that trees missed.

### 4.3 Master RealMLP PyTabKit Implementation
Below is the exact production configuration used by `@yekenot` and Chris Deotte for the top-scoring single model:

```python
import torch
from pytabkit.models import RealMLP_TD_Classifier

realmlp_params = {
    'random_state': 42,
    'verbosity': 1,
    'val_metric_name': '1-auc_ovr',

    # Optimization schedule
    'n_ens': 20,                          # 20 internal sub-models
    'n_epochs': 5,                         # Fast convergence
    'batch_size': 256,
    'use_early_stopping': False,
    'lr': 0.019,
    'wd': 0.01,                           # Weight decay
    'sq_mom': 0.99,
    'lr_sched': 'lin_cos_log_15',
    'first_layer_lr_factor': 0.25,

    # Architecture
    'hidden_sizes': [512, 256, 128],
    'act': 'silu',
    'p_drop': 0.05,
    'p_drop_sched': 'invsqrtp1e-3',

    # Embeddings & PLR (Periodic Linear Representation)
    'embedding_size': 6,
    'max_one_hot_cat_size': 18,
    'plr_hidden_1': 16,
    'plr_hidden_2': 8,
    'plr_act_name': 'gelu',
    'plr_lr_factor': 0.1151,
    'plr_sigma': 2.33,

    # Loss & Regularization
    'ls_eps': 0.01,                       # Label smoothing
    'ls_eps_sched': 'sqrt_cos',
    'add_front_scale': False,
    'bias_init_mode': 'neg-uniform-dynamic-2',
    'tfms': [
        'one_hot', 'median_center', 'robust_scale',
        'smooth_clip', 'embedding', 'l2_normalize'
    ],
}
```

---

## 5. Cross-Validation & Original Data Ingestion Protocol

### 5.1 Synchronous Fold Splitting
Adding the 101,371 rows from `f1_strategy_dataset_v4.csv` to the 439,140 synthetic rows was necessary to reach the $>0.954$ AUC tier. However, naive concatenation causes data leakage if rows from the same original source fold leak into validation.

The standard leak-free protocol requires **synchronous fold generation**:
1. Split competition data into $K$ Stratified folds.
2. Split original external data into $K$ Stratified folds with the identical seed.
3. For fold $k$: concatenate `train[k]` and `orig[k]` as the training set; evaluate **exclusively** on `synthetic_val[k]`.

```python
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

def run_synchronous_cv(X_syn, y_syn, X_orig, y_orig, n_splits=5, seed=42, orig_weight=0.75):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    syn_splits = list(skf.split(X_syn, y_syn))
    orig_splits = list(skf.split(X_orig, y_orig))
    
    oof_preds = np.zeros(len(X_syn))
    
    for fold, ((tr_idx, val_idx), (or_tr_idx, or_val_idx)) in enumerate(zip(syn_splits, orig_splits), 1):
        # Synthetic train & val
        X_tr_syn, y_tr_syn = X_syn.iloc[tr_idx], y_syn.iloc[tr_idx]
        X_val, y_val = X_syn.iloc[val_idx], y_syn.iloc[val_idx]
        
        # Original train
        X_tr_orig, y_tr_orig = X_orig.iloc[or_tr_idx], y_orig.iloc[or_tr_idx]
        
        # Combine training data
        X_train_combined = pd.concat([X_tr_syn, X_tr_orig], axis=0).reset_index(drop=True)
        y_train_combined = pd.concat([y_tr_syn, y_tr_orig], axis=0).reset_index(drop=True)
        
        # Sample weights: downweight original data slightly to match synthetic test distribution
        weights_syn = np.ones(len(X_tr_syn), dtype='float32')
        weights_orig = np.full(len(X_tr_orig), orig_weight, dtype='float32')
        sample_weights = np.concatenate([weights_syn, weights_orig])
        
        # Train model with sample_weights on combined data, evaluate solely on synthetic X_val
        yield fold, X_train_combined, y_train_combined, sample_weights, X_val, y_val, val_idx
```

### 5.2 The Pseudo-Labeling Leakage Trap
Several competitors fell into a catastrophic validation trap when attempting test-set pseudo-labeling:
- **The Mistake**: Taking high-confidence test predictions ($p > 0.90$ or $p < 0.05$) and concatenating them into the raw dataset *before* running `StratifiedKFold`.
- **The Failure**: Because easy test examples were partitioned across folds, they leaked into validation splits, creating an artificial local OOF AUC of **$0.9469 \to 0.9650$**, while the true test leaderboard score collapsed due to noise amplification.
- **The Rule**: Pseudo-labels must only be assigned to out-of-fold partitions during an inner cross-validation loop, or used via strict self-distillation student-teacher architectures.

---

## 6. Ensembling, Logit Stacking & Rank Averaging

### 6.1 Logit Stacking Formulation
Both the 1st place and 2nd place solutions avoided complex tree-based stacking in favor of **$L_2$-penalized Logistic Regression in Logit Space**.

#### Why Logits?
Raw model output probabilities $p_i \in (0, 1)$ compress at extreme values. Transforming probabilities into unbounded log-odds linearizes the decision boundary:

$$z_i = \text{logit}(p_i) = \log\left(\frac{p_i}{1 - p_i}\right), \quad \text{clipped to } [-30, +30]$$

#### Why `class_weight=None`?
For ROC-AUC, only the relative monotonic order of predictions matters. Applying balanced class weights shifts the intercept and non-linearly distorts probabilities, degrading rank-order resolution.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

class LogitStacker:
    def __init__(self, C: float = 1.0, clip_val: float = 30.0):
        self.C = C
        self.clip_val = clip_val
        self.meta_model = LogisticRegression(
            C=self.C,
            penalty='l2',
            class_weight=None,
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        
    def _to_logit(self, p: np.ndarray) -> np.ndarray:
        p_clipped = np.clip(p, 1e-13, 1.0 - 1e-13)
        logit = np.log(p_clipped / (1.0 - p_clipped))
        return np.clip(logit, -self.clip_val, self.clip_val)
        
    def fit(self, oof_matrix: np.ndarray, y_true: np.ndarray):
        """oof_matrix: shape (N, num_models) containing probability predictions"""
        Z = np.column_stack([self._to_logit(oof_matrix[:, i]) for i in range(oof_matrix.shape[1])])
        self.meta_model.fit(Z, y_true)
        return self
        
    def predict_proba(self, test_matrix: np.ndarray) -> np.ndarray:
        """test_matrix: shape (M, num_models) containing probability predictions"""
        Z_test = np.column_stack([self._to_logit(test_matrix[:, i]) for i in range(test_matrix.shape[1])])
        return self.meta_model.predict_proba(Z_test)[:, 1]
```

### 6.2 The Post-Competition Rank-Averaging Breakthrough
In the 5th place post-mortem (`@703572`), the competitor revealed a post-deadline discovery that outscored both 1st and 2nd place on the private leaderboard (**$0.95505$ private AUC**):
- **The Problem**: Blending a Logit Stacker (LS) with a Hill-Climbing (HC) or AutoGluon ensemble using plain probability averaging ($0.5 \cdot p_{\text{LS}} + 0.5 \cdot p_{\text{HC}}$) is sub-optimal because their probability spreads differ substantially. LS is calibrated near the baseline rate ($~0.20$), whereas HC predictions are compressed toward the median ($0.50$). The blend is dominated by whichever model has larger variance.
- **The Solution**: Convert both predictions to empirical percentile ranks via `scipy.stats.rankdata` before averaging:

```python
from scipy.stats import rankdata
import numpy as np

def rank_average_predictions(preds_a: np.ndarray, preds_b: np.ndarray, weight_a: float = 0.5) -> np.ndarray:
    """
    Ranks both prediction arrays onto [0, 1] uniform marginals before blending.
    Optimal for rank-based metrics (ROC-AUC) combining disparate model families.
    """
    rank_a = rankdata(preds_a) / len(preds_a)
    rank_b = rankdata(preds_b) / len(preds_b)
    
    return weight_a * rank_a + (1.0 - weight_a) * rank_b
```
- **Leaderboard Proof**:
  - Plain probability average: Private LB = **$0.95503$**
  - Rank-normalized average: Private LB = **$0.95505$** (decisive $+0.00002$ jump in a race decided by $0.00001$).

---

## 7. Key Takeaways & Anti-Patterns

### 7.1 What Worked (Golden Insights)
1. **RealMLP as the Single-Model Anchor**:
   PyTabKit RealMLP was undeniably the king of continuous feature representation for F1 pit-stop prediction, topping all single tree models by $+0.0009$ CV.
2. **Original Data Synchronous Augmentation**:
   Appending `f1_strategy_dataset_v4.csv` rows to training folds with sample weights between $0.5$ and $1.0$ produced an immediate, reproducible $+0.0010$ AUC boost.
3. **Adversarial Driver-Drift Pruning**:
   Omitting `Driver` from a subset of models created a resilient, drift-free model stream that prevented public LB split-overfitting.
4. **Weak-Model Tail Decorrelation**:
   A $0.918$-AUC Deep Factorization Machine contributed more orthogonal ranking power to the final logit stack than a fourth $0.953$ LightGBM.
5. **Logit Stacking + Rank Averaging**:
   Fitting $L_2$ Logistic Regression on logit-transformed OOFs, then rank-averaging with an AutoGluon/Hill-Climbing ensemble provided the ultimate winning margin.

### 7.2 What Failed or Overfit (Anti-Patterns)
1. **Synthetic Generator Digit & Mantissa Decomposition**:
   Unlike S6E1/S6E3, extracting decimal mantissas, modulo digits, or Benford residuals failed to improve CV. The data contained genuine physical telemetry signal rather than generator rounding artifacts.
2. **AutoFE Libraries (FeatureTools, AutoFeat)**:
   FeatureTools produced degraded CV scores, and AutoFeat timed out after 12 hours without discovering non-linear signal. Domain-specific physical formulas (`LapNumber / RaceProgress`) outperformed automated search.
3. **Naive Test Pseudo-Labeling**:
   Concatenating high-confidence test pseudo-labels prior to CV split created severe artificial leakage (inflating OOF AUC to $>0.965$ while failing on LB).
4. **High-Order Non-Linear Stacking**:
   Training deep GBDTs or neural networks on Level 2/Level 3 meta-predictions overfitted the validation folds. Simple $L_2$ Logistic Regression on logits with `class_weight=None` proved far superior.
