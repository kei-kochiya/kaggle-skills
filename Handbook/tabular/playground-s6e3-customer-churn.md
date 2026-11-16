# Playground Series S6E3: Customer Churn Prediction

**Competition**: [Kaggle Playground Series - Season 6 Episode 3](https://www.kaggle.com/competitions/playground-series-s6e3)  
**1st Place Solution**: *1st Place — GPT5.4, Gemini3.1, ClaudeOpus4.6 and KGMON Playbook for Tabular Data* by Chris Deotte (`@cdeotte`, Senior Data Scientist at NVIDIA)  
**Track**: Tabular Binary Classification  
**Evaluation Metric**: Area Under the ROC Curve ($\text{ROC-AUC}$)  
**Official Code Artifacts**: [GitHub 100-Model Open-Source Repo (`s6e3_customer_churn`)](https://github.com/cdeotte/kaggle-playground-solutions/tree/main/s6e3_customer_churn), [Kaggle 1st Place Solution Writeup](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm), [Kaggle 1st Place Solution Notebook](https://www.kaggle.com/code/cdeotte/1st-place-nvidia-cuml-logistic-regression) & [Kaggle OOF Dataset](https://www.kaggle.com/datasets/cdeotte/s6e3-oof-and-test-pred-v2)

### Official Winning Scoreboard
| Submission Type | Cross-Validation (CV / OOF) | Public LB | Private LB | Full-Fit Train AUC | Notes / Outcome |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Final Selected 4-Level Stack (154 Models)** | **0.919857** | 0.91773 | **0.91856** | **0.920002** | 🥇 **1st Place Gold Medal (137 L1 + 17 L2/L3 models via cuML Logistic Regression)** |
| **Open-Source 100-Model Repo Subset (`s6e3_customer_churn`)** | **0.91983** | 0.91765 | **0.91853** | — | Standalone 100 Level-1 models + `stack.py` (also scores **1st Place** alone) |
| Top Single Model (LightGBM `lgbm_union_dist_ngram_te_slow` / XGB `xgb_ngram_te_binned_keys_drift`) | **0.919164** / **0.919175** | — | — | — | Top single GBDTs in open-source release (30k rounds slow LGBM / n-gram TE XGB) |
| Top Public RealMLP (`yekenot_realmlp_pytabkit_silu`) | **0.919118** | — | — | — | Strongest single neural net; internal `realmlp_v10700` got highest stack coef ($+0.2085$) |
| Foundation Model (`TabICL`) | ~0.9185 (`0.915195` public port) | — | — | — | Zero-shot in-context learning ($+0.1923$ meta weight in final stack) |
| Top GBDT Level 3 Stacker (`cat_v10324_stk`) | ~0.9192 | — | — | — | CatBoost stacking Level 2 OOFs ($+0.1551$ meta weight) |
| Best Single Baseline XGBoost (`XGBRanker` / Raw) | ~0.9149–0.9182 | — | — | — | Starting baseline |

---

## 1. Executive Summary: The Winning Paradigm

The winning solution for Playground Series S6E3 demonstrated how **frontier generative AI agents** paired with **GPU-accelerated RAPIDS infrastructure** can redefine competitive machine learning at unprecedented scale:

1. **Autonomous Scale (600k Lines, 850 Models)**:
   An ensemble of frontier LLMs (**GPT-5.4**, **Gemini 3.1**, and **Claude Opus 4.6**) wrote over 600,000 lines of code, ran 50 automated exploratory data analysis scripts, and trained 850 candidate models on a 4× NVIDIA A100 GPU cluster.
2. **Exploiting Synthetic Generator Fingerprints**:
   Rather than treating the synthetic dataset (594k rows) as noisy data, the team treated the synthetic data generator itself as a source of structured signal. By discovering "snap features" (mapping synthetic floats to the nearest original 7,032-row IBM Telco values) and decomposing decimal mantissas, the models learned the generator's underlying manifold and perturbation magnitude.
3. **Unprecedented Model Diversity (25 DL Families + 5 Tree Engines)**:
   The ensemble combined 90 tree models across 5 libraries (XGBoost, LightGBM, CatBoost, YDF, cuML Random Forest) and 60 deep learning models spanning 25 distinct neural architectures (RealMLP, TabM, TabICL, FT-Transformer, GraphSAGE GNN, DAE, SNN, Liquid Neural Networks, Trompt, GANDALF).
4. **Greedy Forward Hill Climbing**:
   Out of 850 trained models, automated greedy forward selection filtered out 696 correlated models, isolating 154 mutually orthogonal predictors.
5. **4-Level Stacking with Fold-Wise Rank Calibration**:
   The final layer combined Level 2 base models and Level 3 meta-stackers using **NVIDIA cuML $L_2$-penalized Logistic Regression** in logit space after applying fold-wise ordinal rank probability calibration.

---

## 2. Dataset Architecture & Exploratory Data Analysis

### Data Specs
- **Training Set**: 594,194 rows $\times$ 19 columns
- **Test Set**: 254,655 rows $\times$ 18 columns
- **Target**: `Churn` (Binary: `0` = Retained, `1` = Churned; Positive rate: $22.521\%$)
- **Original Source Dataset**: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,032 clean rows)
- **Validation Scheme**: 5-Fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

### Feature Inventory
| Feature Name | Type | Description / Synthetic Dynamics |
| :--- | :--- | :--- |
| `MonthlyCharges` | Numerical (Float) | Billed monthly amount; continuous float perturbed from original IBM values |
| `TotalCharges` | Numerical (Float) | Accumulated lifetime charges; heavily correlated with $\text{tenure} \times \text{MonthlyCharges}$ |
| `tenure` | Numerical (Integer) | Months the customer has stayed with the company ($0 - 72$) |
| `Contract` | Categorical | `Month-to-month`, `One year`, `Two year` (strongest single predictor) |
| `InternetService` | Categorical | `DSL`, `Fiber optic`, `No` |
| `PaymentMethod` | Categorical | `Electronic check`, `Mailed check`, `Bank transfer`, `Credit card` |
| `OnlineSecurity`, `OnlineBackup` | Categorical | Add-on services: `Yes`, `No`, `No internet service` |
| `DeviceProtection`, `TechSupport` | Categorical | Add-on services: `Yes`, `No`, `No internet service` |
| `StreamingTV`, `StreamingMovies` | Categorical | Entertainment services: `Yes`, `No`, `No internet service` |
| `MultipleLines` | Categorical | `Yes`, `No`, `No phone service` |
| `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Categorical / Binary | Customer demographic variables |
| `PhoneService`, `PaperlessBilling` | Binary | Account and service flags |

---

## 3. The 12-Component Feature Engineering Suite

Feature engineering was the single greatest driver of competitive performance, accounting for over $+0.0016$ AUC improvement over raw baselines.

```
+-----------------------------------------------------------------------------------+
|                        12-COMPONENT FEATURE SUITE                                 |
+-----------------------------------------------------------------------------------+
| 1. Snap Matching           --> Nearest original IBM floats & perturbation diff   |
| 2. Digit Decomposition     --> Decimal mantissas (frac, d1, d2, fractions)        |
| 3. Nested 5x5 TE           --> 10 statistical aggregations across 16 categoricals  |
| 4. Billing Deviations      --> TotalCharges - tenure * MonthlyCharges             |
| 5. Multi-Scale Binning     --> Up to 5,000 quantile bins reconstructing manifold  |
| 6. Categorical N-Grams     --> Bigrams & trigrams of high-signal service pairs    |
| 7. Frequency & Ratios      --> Synthetic count / original IBM count oversample    |
| 8. Service Aggregations    --> Total active security, tech, and streaming add-ons|
| 9. cKDTree Neighbor Lookup --> Nearest original customer ground-truth churn label |
| 10. Radix Interaction      --> int(MC_snap * 100) + cat_code * 100_000            |
| 11. Synthetic Artifacts    --> Benford's Law deviation & character n-gram TF-IDF  |
| 12. Manifold Projections   --> 12-dim PCA / Random Projections & cyclical tenure  |
+-----------------------------------------------------------------------------------+
```

### 3.1 Snap Features (Universal Core — ~150 Models)
Because the 594k synthetic records were generated by adding Gaussian/diffusion perturbation noise to the 7,032 original IBM rows, every continuous synthetic float was mapped back to its nearest original archetype:

```python
import numpy as np
import pandas as pd

def compute_snap_features(train_df, test_df, orig_df, cols=["MonthlyCharges", "TotalCharges"]):
    for col in cols:
        orig_vals = np.sort(orig_df[col].dropna().unique())
        
        for df in [train_df, test_df]:
            vals = df[col].values
            idx = np.searchsorted(orig_vals, vals)
            idx = np.clip(idx, 0, len(orig_vals) - 1)
            left_idx = np.clip(idx - 1, 0, len(orig_vals) - 1)
            
            dist_right = np.abs(vals - orig_vals[idx])
            dist_left = np.abs(vals - orig_vals[left_idx])
            nearest = np.where(dist_left < dist_right, orig_vals[left_idx], orig_vals[idx])
            
            df[f"{col}_snap"] = nearest
            df[f"{col}_snap_diff"] = vals - nearest  # Generator perturbation noise
```
- **Interpretation**: `col_snap` recovers the true unperturbed customer billing tier; `col_snap_diff` measures how far the synthetic generator drifted that specific customer.

### 3.2 Digit and Decimal Mantissa Extraction (~60 Models)
CTGAN and TVAE generators fail to preserve human decimal conventions. Exact decimal digits were isolated into explicit integer features:

```python
for col in ["MonthlyCharges", "TotalCharges", "MonthlyCharges_snap", "TotalCharges_snap"]:
    x = df[col]
    frac = x - np.floor(x)
    df[f"{col}_frac"] = frac
    df[f"{col}_d1"] = np.floor(frac * 10).astype(int)              # 1st decimal digit
    df[f"{col}_d2"] = (np.floor(frac * 100) % 10).astype(int)       # 2nd decimal digit
    df[f"{col}_frac100"] = np.round(frac * 100).astype(int)         # 2-digit integer
    df[f"{col}_mod10"] = (np.floor(x) % 10).astype(int)
    df[f"{col}_mod100"] = (np.floor(x) % 100).astype(int)
    
    # Common fraction residuals
    df[f"{col}_res_half"] = np.abs(frac - 0.5)
    df[f"{col}_res_quarter"] = np.min([np.abs(frac - q) for q in [0.0, 0.25, 0.5, 0.75]], axis=0)
    df[f"{col}_is_round"] = (frac < 0.005).astype(int)
```

### 3.3 Leak-Free Nested Target Encoding (~90 Models)
To eliminate target leakage, all target encodings used an inner 5-fold CV split inside each outer training fold:
- **Encoded Columns**: All 16 raw categoricals, bigrams (`Contract__InternetService`), binned numerics, and anchor keys `(MC_snap, tenure)`.
- **10 Statistical Aggregations**: `mean`, `std`, `min`, `max`, `median`, and quantiles ($5\text{th}, 10\text{th}, 45\text{th}, 55\text{th}, 90\text{th}, 95\text{th}$).
- **Original Dataset Priors**: Churn rates computed directly from the 7,032 original rows were mapped as zero-leakage prior features.

### 3.4 Arithmetic Billing Deviations (~45 Models)
In a real telecom billing system, $\text{TotalCharges} \approx \text{tenure} \times \text{MonthlyCharges}$. Discrepancies signify plan migrations, discounts, or unpaid penalties:

$$\text{TC\_deviation} = \text{TotalCharges} - \text{tenure} \times \text{MonthlyCharges}$$
$$\text{TC\_snap\_exp\_dev} = \text{TC\_snap} - \text{tenure} \times \text{MC\_snap}$$
$$\text{TC\_per\_month} = \frac{\text{TotalCharges}}{\text{tenure} + 1}$$
$$\text{MC\_to\_TC\_ratio} = \frac{\text{MonthlyCharges}}{\text{TotalCharges} + 10^{-9}}$$

### 3.5 Radix Interaction Features (~15 Models)
Encodes a continuous snap value and a categorical feature as a single combined integer:

$$\text{radix} = \lfloor \text{MC\_snap} \times 100 \rfloor + \text{cat\_code} \times 100{,}000$$

- **Tree Split Advantage**: A single decision tree split on `radix` simultaneously partitions both the continuous charge boundary and the categorical contract/service type without consuming two tree depth levels.

### 3.6 Original IBM Prior & cKDTree Nearest Neighbor Lookup (~7 Models)
Using `scipy.spatial.cKDTree` on standardized `(MonthlyCharges, TotalCharges, tenure)` from the 7,032 original IBM rows, the agent retrieved the nearest real-world customer for each synthetic row and injected the ground-truth historical churn outcome ($0$ or $1$) as an anchor feature.

### 3.7 Synthetic Artifact Detection & Benford's Law
- **Benford's Law Likelihood**: Log-odds deviation of leading digits $d \in \{1, \dots, 9\}$ against $\log_{10}(1 + 1/d)$.
- **Character N-Gram TF-IDF**: Extracted 128 character n-gram TF-IDF components from string representations of `MonthlyCharges` (e.g. `"29.85"` $\to$ `"29"`, `"9."`, `".8"`, `"85"`), capturing repeating mantissa artifacts.
- **Drift Ratios**: $\ln\left(1 + \frac{\text{synthetic\_count}}{\text{original\_count}}\right)$ measuring generator oversampling bias.

---

## 4. The 154-Model Diversity Zoo

From 850 trained candidates, greedy hill-climbing selected **154 models** spanning 5 GBDT libraries and 25 neural network architecture families:

```
+-----------------------------------------------------------------------------------+
|                        154 ENSEMBLE BASE MODELS                                   |
+-----------------------------------------------------------------------------------+
|  TREE MODELS (90 Total)                   DEEP LEARNING MODELS (60 Total)         |
|  * 37 XGBoost (Standard, Anchor, Ranker)  * 9 RealMLP (pytabkit & from-scratch)   |
|  * 22 LightGBM (Leaf-wise, Count-ratio)   * 3 TabM (Bilinear Multi-Component)     |
|  * 22 CatBoost (Native OTS, Binned Cross) * 3 TabICL (In-Context Foundation Model)|
|  * 2 YDF (Ultra-shallow depth=2 stumps)   * 4 FT-Transformer                      |
|  * 2 cuML Random Forest (GPU Bagging)     * 5 TabTransformer                      |
|                                           * 4 GraphSAGE GNN (cuML KNN graph)      |
|  LEVEL 3 STACKERS (36 of the 154)         * 10 DAE & DAE-Augmented GBDTs          |
|  * xgb_v3614_stk, cat_v10324_stk          * 2 GANDALF (Gated GFLU)                |
|  * realmlp_v1701n2_stk, tabm_v9102_stk    * 2 SELU-AlphaDropout SNN               |
|  * logit_v124_stk, saint_v9909_stk        * 3 RFF Kernel Networks                 |
|                                           * 10 FFM / FM / DeepFM                  |
|                                           * 3 Liquid Neural Networks (LNN)        |
|                                           * 3 Trompt (Tabular Prompt Model)       |
+-----------------------------------------------------------------------------------+
```

### 4.1 Tree Model Breakdown (90 Models)
1. **XGBoost (37 models)**:
   - Evaluated standard depth-wise growth with `max_depth` from 4 to 8.
   - Tested `XGBRanker` with pairwise ranking objective (`rank:pairwise`), directly targeting AUC concordant pairs.
   - Self-supervised auxiliary predictions: models trained to predict each feature from all others, using residual errors as inputs.
2. **LightGBM (22 models)**:
   - Leaf-wise growth (`num_leaves=63` to `77`, `min_child_samples=56`).
   - Highly responsive to count-ratio features and high-cardinality multi-scale quantile bins.
3. **CatBoost (22 models)**:
   - Native Ordered Target Statistics (OTS) handled categorical combinations without manual TE leakage risk.
   - Explicit binned interaction cross-terms ($a_{\text{bin}} \times 9 + b_{\text{bin}}$) compensated for oblivious (symmetric) tree limitations.
4. **Yggdrasil Decision Forests (YDF — 2 models)**:
   - Google's YDF configured with `max_depth=2` ultra-shallow stumps.
   - Provided smooth, highly regularized piecewise-constant surfaces that stabilized neural network predictions.
5. **RAPIDS cuML Random Forest (2 models)**:
   - The only bagging architecture in the tree pool (averaging independently grown deep trees).
   - Generated fundamentally different variance-reduction properties compared to boosting.

### 4.2 Deep Learning Architecture Breakdown (60 Models)
1. **RealMLP (`pytabkit` & from-scratch — 9 models)**:
   - Piecewise-linear embeddings (PLR) for continuous features + SiLU activations + L2 normalization + internal 8-member ensemble.
   - **Highest single-model impact**: `realmlp_v10700` received the single highest positive weight ($+0.2085$) in the final Level 4 meta-learner.
2. **TabICL (In-Context Foundation Model — 3 models)**:
   - Pre-trained Transformer foundation model for tabular data.
   - Required **zero gradient updates**; performed inference via attention over in-context training examples.
   - **Second highest meta-weight**: `tabicl_v9500` received $+0.1923$ weight.
3. **TabM (Multiplicative Bilinear Interactions — 3 models)**:
   - Decomposes predictions across $k=32$ basis components with multiplicative (bilinear) interactions alongside additive paths. OOF AUC: $0.918788$.
4. **GraphSAGE GNN (4 models)**:
   - Built an 8-nearest-neighbor customer graph via GPU cuML KNN.
   - Aggregated neighborhood churn dynamics through 2 SAGEConv layers.
5. **Denoising Autoencoders (DAE — 10 models)**:
   - Trained on the 7,032 original IBM rows with Gaussian corruptions. Latent embeddings and reconstruction error vectors were fed into GBDTs.
6. **Liquid Neural Networks (LNN — 3 models)**:
   - Continuous-time ODE-inspired neuron dynamics with learnable time constants $\tau$, providing orthogonal representation structure.

### 4.3 Open-Source 100-Model Repository Breakdown (`cdeotte/kaggle-playground-solutions`)
Chris Deotte released 100 of the 137 Level-1 models in [`s6e3_customer_churn`](https://github.com/cdeotte/kaggle-playground-solutions/tree/main/s6e3_customer_churn) (covering every architecture family and the strongest variant of each class). Stacked alone with fold-calibrated logits (`cuML LogisticRegression(C=0.01)`), these 100 models achieve **0.91983 CV / 0.91765 Public LB / 0.91853 Private LB** (still 1st Place):

- **XGBoost (20 released models, peak `0.919175` OOF)**:
  - `xgb_ngram_te_binned_keys_drift` (**`0.919175`**): Nested n-gram target encoding + binned/rounded/digit-pair keys, counts, drift, and original IBM churn rates.
  - `xgb_bigram_trigram_te_stats` (`0.919015`) and `xgb_union_ab_lgbm_port_prebinned` (`0.918804`, pre-binned to 256 quantile bins with `max_bin=16000`).
  - Specialized variants: `xgb_native_cat_tabpfn_original` (`0.916507`, TabPFN fitted on original 7k IBM rows as a prior feature), `xgb_original_self_supervised` (`0.915474`, auxiliary models predicting each column from all others on original rows), and `xgb_pairwise_ranker_platt` (`0.914916`, `XGBRanker(objective='rank:pairwise')` + Platt scaling).
- **LightGBM (13 released models, peak `0.919164` OOF)**:
  - `lgbm_union_dist_ngram_te_slow` (**`0.919164`**): 466-column feature union (transforms, flags, crosses, original priors, distribution/digit/n-gram blocks) trained for **30,000 rounds at `lr=0.0025`**.
  - `lgbm_union_dist_ngram_te` (`0.919138`), `lgbm_union_dist_ngram_te_goss` (`0.918806`, GOSS 63 leaves), `lgbm_union_dist_ngram_te_dart` (`0.918742`, DART 31 leaves, depth 4), `lgbm_extra_trees_pairs_digits_two_stage_te` (`0.918361`), and `lgbm_decimal_ngram_svd_prototypes` (`0.914912`, TF-IDF/SVD over charge text n-grams + k-means prototypes).
- **CatBoost (14 released models, peak `0.919017` OOF)**:
  - `catboost_numcat_digits_orig_rows` (**`0.919017`**) vs `catboost_numcat_digits_orig_priors_drift` (`0.918957`) — symmetric tree architectures (`max_ctr_complexity=4`, 65 native categoricals, PMI + KD-tree density in `catboost_fold_density_knn_orig_priors` at `0.918762`, exact numeric triple cross in `catboost_triple_cross_ranked_orig_rows` at `0.918874`).
- **RealMLP & TabM (7 internal + 5 public ports, peak `0.919118` OOF)**:
  - `yekenot_realmlp_pytabkit_silu` (**`0.919118`**), `yunsuxiaozi_realmlp_pytorch` (`0.918943`), `blamerx_tabm` (`0.918788`), `realmlp_pair_te_logit3_kbins_priors` (`0.918292`, KBins + value codes + 105 pair TEs expanded as cubic logit powers $[z, z^2, z^3]$).
- **31 Neural Network Variants (`models/neural_nets`, up to `0.917661` OOF)**:
  - Spans `dae_knn_discriminator_token_te_xgb` (`0.917661`), `emb_mlp_engineered_original_priors_drift` (`0.917578`), `tabtransformer_pair_te_logit3_digits_original_rows` (`0.917488`, 561 pair TE logit powers), `graphsage_sampled_snapped_rare_flags_ohe_knn_graph` (`0.915789`), `deepfm_snapped_quantile_prior_tokens` (`0.915703`), `nam_original_density_bins` (`0.914941`), `tokenized_cnn_original_gmm_archetypes` (`0.913764`), `prototype_net_original_mahalanobis` (`0.913463`), `ifan_original_noise_kmeans_kpca` (`0.913336`), `snn_charge_char_ngrams_benford_orig_knn` (`0.912932`), and `gandalf_raw_columns` (`0.911962`).
- **Classic, KNN & Bayesian Survival Models (`models/classic` & `public_models`)**:
  - `logreg_ohe_crosses_hinges_group_dense` (`0.916847`, cuML LR on 4,362 one-hot bins/digits/crosses + 148 dense cols), `johnnyhyland_survival_xgb` (`0.916741`, PyMC Bayesian survival-curve features into XGBoost), `siukeitin_ydf` (`0.917976`), and `lr_pair_te_logit3` (`0.916014`, cubic logit powers of 171 pair TEs).

---

## 5. Top Model Weights in the Level 4 Meta-Learner

The Level 4 NVIDIA cuML Logistic Regression assigned the following top 25 absolute weights across the 154 base models:

| Rank | Model Name | Architecture Family | Hierarchy Level | Stacker Coef ($\beta$) | Role / Inductive Bias |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | `realmlp_v10700` | RealMLP (`pytabkit`) | Level 2 | **+0.2085** | Primary neural anchor (PLR embeddings) |
| **2** | `tabicl_v9500` | TabICL Foundation Model | Level 2 | **+0.1923** | In-context Transformer representations |
| **3** | `cat_v10324_stk` | CatBoost Stacker | Level 3 | **+0.1551** | Symmetric tree meta-stacking |
| **4** | `xgb_v3614_stk` | XGBoost Stacker | Level 3 | **+0.1459** | Depth-wise tree meta-stacking |
| **5** | `realmlp_v1701n2_stk`| RealMLP Stacker | Level 3 | **-0.1426** | Negative residual suppressor |
| **6** | `tabm_v9100` | TabM Multi-Component | Level 2 | **+0.1395** | Bilinear multiplicative interaction |
| **7** | `xgb_v5409_stk` | XGBoost Stacker | Level 3 | **+0.1277** | Multi-seed tree stacker |
| **8** | `lgbm_v3308` | LightGBM | Level 2 | **-0.1269** | Leaf-wise residual balance |
| **9** | `xgb_v2200` | XGBoost | Level 2 | **-0.1128** | Depth-wise anchor baseline |
| **10** | `logit_v124_stk` | Logistic Stacker | Level 3 | **+0.1122** | Linear probability meta-blend |
| **11** | `realmlp_v10900` | RealMLP | Level 2 | **+0.1046** | High-capacity PLR neural net |
| **12** | `cat_v10327` | CatBoost | Level 2 | **-0.1017** | Native OTS regularizer |
| **13** | `cat_v10325` | CatBoost | Level 2 | **+0.1010** | Bigram/trigram cross-term model |
| **14** | `cat_v10326` | CatBoost | Level 2 | **+0.0978** | Binned numeric interactions |
| **15** | `realmlp_v1700` | RealMLP | Level 2 | **+0.0974** | Medium-capacity PLR model |
| **16** | `rff_v9705` | RFF Kernel Network | Level 2 | **-0.0945** | Random Fourier Features RBF kernel |
| **17** | `cat_v10315` | CatBoost | Level 2 | **-0.0891** | Native OTS baseline |
| **18** | `xgb_v3600` | XGBoost | Level 2 | **+0.0861** | Decaying learning rate schedule |
| **19** | `nn_v315n3_stk` | MLP Stacker | Level 3 | **-0.0845** | Residual error correction |
| **20** | `lgbm_v9802` | LightGBM | Level 2 | **-0.0787** | High-cardinality count features |
| **21** | `xgb_v3200` | XGBoost | Level 2 | **+0.0778** | Modulo digit decomposition |
| **22** | `nn_v308` | Embedding MLP | Level 2 | **+0.0746** | Deep embedding representation |
| **23** | `xgb_v7123` | XGBoost | Level 2 | **-0.0743** | Radix interaction model |
| **24** | `logreg_v7014n2_stk`| Logistic Stacker | Level 3 | **+0.0701** | Level 2 linear aggregator |
| **25** | `tabnet_v5200` | TabNet | Level 2 | **+0.0000** | Sparse sequential attention |

---

## 6. The 4-Level Stacking Pipeline Implementation

Below is the exact production architecture of the Level 4 Stacker combining all 154 models:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from cuml.linear_model import LogisticRegression as cuLogisticRegression

# 1. Configuration
N_SPLITS = 5
RANDOM_STATE = 42
META_C = 0.01          # Strong L2 regularization
META_SOLVER = "qn"     # Quasi-Newton L-BFGS
EPS = 1e-15
LOGIT_CLIP = 30.0

def prob_to_logit(p, eps=1e-15, clip=30.0):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.clip(np.log(p / (1.0 - p)), -clip, clip)

# 2. Fold-Wise Ordinal Rank Probability Calibration
def ordinal_common_fold_ranks(values, common_size):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = values.size
    order = np.argsort(values, kind="mergesort")
    pos = np.arange(n, dtype=np.int64)
    ranks_sorted = np.clip((pos * common_size) // n, 0, common_size - 1).astype(np.int32)
    ranks = np.empty(n, dtype=np.int32)
    ranks[order] = ranks_sorted
    return ranks

def fold_calibrate_oof_probs(oof, folds, eps=1e-15):
    min_fold_size = int(min(len(va_idx) for _, va_idx in folds))
    ranks = np.empty(oof.shape[0], dtype=np.int32)
    for _, va_idx in folds:
        ranks[va_idx] = ordinal_common_fold_ranks(oof[va_idx], common_size=min_fold_size)
    rank_means = pd.DataFrame({"rank": ranks, "prob": oof}).groupby("rank", sort=True)["prob"].mean()
    calibrated = rank_means.iloc[ranks].to_numpy(dtype=np.float64)
    return np.clip(calibrated, eps, 1.0 - eps)

# 3. Execution Pipeline
# (Assuming oof_matrix: (594194, 154) and test_matrix: (254655, 154))
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
FOLDS = list(skf.split(np.zeros(len(y)), y))

# Calibrate each base model across folds
for m in range(154):
    oof_matrix[:, m] = fold_calibrate_oof_probs(oof_matrix[:, m], FOLDS)

# Transform to Logit Space
X_train_logit = prob_to_logit(oof_matrix, eps=EPS, clip=LOGIT_CLIP).astype(np.float32)
X_test_logit  = prob_to_logit(test_matrix, eps=EPS, clip=LOGIT_CLIP).astype(np.float32)

# 4. Fit Fold-Wise for Honest CV AUC
meta_oof = np.zeros(len(y), dtype=np.float64)
for fold, (tr_idx, va_idx) in enumerate(FOLDS):
    clf = cuLogisticRegression(penalty="l2", C=META_C, solver=META_SOLVER, max_iter=10000, tol=1e-4)
    clf.fit(X_train_logit[tr_idx], y[tr_idx])
    meta_oof[va_idx] = clf.predict_proba(X_train_logit[va_idx])[:, 1]

honest_cv_auc = roc_auc_score(y, meta_oof)
print(f"Honest Level 4 cuML Logistic Stack OOF AUC: {honest_cv_auc:.6f}")  # 0.919857

# 5. Full-Data 10x Fit for Final Test Submission
final_clf = cuLogisticRegression(penalty="l2", C=META_C, solver=META_SOLVER, max_iter=10000, tol=1e-4)
final_clf.fit(X_train_logit, y)
final_test_pred = final_clf.predict_proba(X_test_logit)[:, 1]
```

---

## 7. Key Takeaways & Anti-Patterns

### What Worked
- **Snap Matching**: Recovered original customer ground-truth coordinates from synthetic floating-point perturbations ($+0.0008$ AUC).
- **Extreme Architecture Diversity**: Combining TabICL (in-context Transformer), RealMLP, and GBDTs yielded massive ensembling gains ($+0.0012$ over best single model).
- **Fold-Wise Ordinal Calibration**: Eliminated probability drift between folds before logit conversion, protecting ranking integrity.
- **$L_2$ Logistic Regression Meta-Learner**: Prevented tree-based stacker overfitting on 154 correlated inputs.

### What Failed / Anti-Patterns
- **Nonlinear GBDT / Neural Stackers**: GBDT stackers (LightGBM/XGBoost on Level 2 OOFs) severely overfit the validation folds and collapsed on private test evaluation.
- **Uncalibrated Averaging of 850 Models**: Simple averaging of all 850 trained models degraded CV AUC below $0.917$ due to high correlation among weak variants.
- **Target Encoding Without Inner Nested Folds**: Target encoding computed across the full outer training split leaked validation labels, generating optimistically false CV scores.
