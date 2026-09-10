# Playground Series S6E2: Predicting Heart Disease

**Competition**: [Kaggle Playground Series - Season 6 Episode 2](https://www.kaggle.com/competitions/playground-series-s6e2)  
**1st Place Solution**: *1st Place Solution — Diversity, Selection, and Trusting the CV–LB Relation* by Masaya Kawamata (`@masayakawamata`)  
**Track**: Tabular Binary Classification  
**Evaluation Metric**: Area Under the ROC Curve ($\text{ROC-AUC}$)  

### Official Winning Scoreboard
| Submission Type | Cross-Validation (CV) | Public Leaderboard (LB) | Private Leaderboard (LB) | Outcome / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Final Selected Submission** | **0.9557801** | **0.95396** | **0.95535** | 🥇 **1st Place Gold Medal (Winning Choice)** |
| **Best Overall Submission** | 0.9557901 | 0.955394 | 0.95536 | Highest private score obtained |
| **Highest CV Reached** | 0.9558650 | 0.955393 | 0.95534 | **Rejected** due to split-overfitting |
| **Top Single Model (RealMLP)** | 0.9557390 | 0.95397 | — | Top single neural model (`pytabkit`) |

---

## 1. Executive Summary: The Winning Paradigm

The winning solution for Playground Series S6E2 was not built around a single "magic" model or an opaque brute-force stack. Instead, it demonstrated a masterclass in **controlled diversity**, **disciplined subset selection**, and **validation integrity**:

1. **"Trust the CV–LB Relation, Not Just Best CV"**:
   The central philosophical insight was identifying the **$0.95578$ inflection point**. Up to CV $\approx 0.95578$, CV and Leaderboard improvements correlated reliably. Beyond $0.95578$ (e.g. CV $0.955865$), CV gains began diverging from the leaderboard, indicating that models were memorizing fold-specific split boundaries rather than learning true cardiac risk factors. Selecting submissions within the **$0.95578 - 0.95580$** band secured the 1st place gold medal on private evaluation.
2. **Controlled Feature Representation Diversity**:
   Instead of endlessly tweaking hyperparameters, the author engineered 7 distinct representation layers (quantile/uniform binning, decimal digit decomposition, string categorization, frequency counts, genetic programming features, original dataset priors, and Denoising Variational Autoencoder latents).
3. **Disciplined Ensemble Selection via Optuna**:
   From a pool of $\approx 150$ Out-Of-Fold (OOF) prediction files, Optuna ran 2,500 trials to search for optimal subsets. Only about $10\%$ to $15\%$ of the models were consistently selected.
4. **Simple & Robust Stacking**:
   Selected OOF probabilities were combined using simple **Ridge Regression**. Nonlinear meta-models (such as GBDT or MLP stackers) proved excessively flexible and consistently overfit the cross-validation splits.
5. **Full-Data Retraining with Multi-Seed Averaging**:
   For tree models and RGF, final models were retrained on the full dataset across 20 distinct random seeds with iterations set to $1.25 \times$ the average best CV iteration.

---

## 2. Dataset Architecture & Exploratory Data Analysis

### Data Specs
- **Training Set**: 630,000 rows $\times$ 15 columns
- **Test Set**: 270,000 rows $\times$ 14 columns
- **Target**: `Heart Disease` (`Absence`: $55.17\%$, `Presence`: $44.83\%$)
- **Original Auxiliary Dataset**: [Heart Disease Prediction dataset](https://www.kaggle.com/datasets/heartdisease/Heart_Disease_Prediction) (`Heart_Disease_Prediction.csv`)

### Feature Inventory
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Age` | Numerical | Age in years ($29 - 77$) |
| `Sex` | Binary | `0` = Female, `1` = Male |
| `Chest pain type` | Categorical / Ordinal | Levels `1`, `2`, `3`, `4` |
| `BP` | Numerical | Resting blood pressure (mmHg) |
| `Cholesterol` | Numerical | Serum cholesterol (mg/dl) |
| `FBS over 120` | Binary | Fasting blood sugar > 120 mg/dl (`0`, `1`) |
| `EKG results` | Categorical | Resting electrocardiographic results (`0`, `1`, `2`) |
| `Max HR` | Numerical | Maximum heart rate achieved |
| `Exercise angina` | Binary | Exercise induced angina (`0`, `1`) |
| `ST depression` | Numerical (Float) | ST depression induced by exercise relative to rest |
| `Slope of ST` | Categorical | Slope of peak exercise ST segment (`1`, `2`, `3`) |
| `Number of vessels fluro` | Discrete | Major vessels ($0 - 3$) colored by fluoroscopy |
| `Thallium` | Categorical | Defect type (`3` = Normal, `6` = Fixed defect, `7` = Reversible defect) |

---

## 3. The 7-Layer Feature Engineering Playbook

Rather than searching for a single "optimal" feature set, the author generated multiple complementary views of the base features:

### 3.1 Multi-Scale & Domain-Specific Binning
Discretization alters tree split structures and gives neural networks categorical anchor points:
- **Quantile Binning (`pd.qcut`)**: Equal-frequency bins with $q \in \{4, 5, 10, 20\}$.
- **Uniform Width Binning (`pd.cut`)**: Equal-interval bins with $b \in \{5, 10\}$.
- **Domain-Specific Diagnostic Rounding**:
  - `Age / 5` (5-year age brackets)
  - `BP / 10` & `Max HR / 10` (clinical ten-unit intervals)
  - `Cholesterol / 50` (50 mg/dl metabolic cohorts)

```python
target_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
for col in target_cols:
    for q in [4, 5, 10, 20]:
        df[f"{col}_bin_q{q}"] = pd.qcut(df[col], q=q, labels=False, duplicates='drop')
    for b in [5, 10]:
        df[f"{col}_bin_cut{b}"] = pd.cut(df[col], bins=b, labels=False)
```

### 3.2 Decimal Digit-Level Decomposition
Extracting integer base-10 digits and fractional decimals exposed rounding boundaries and generative artifacts:

```python
for col in BASE:
    max_val = df[col].abs().max()
    max_digits = 1 if max_val == 0 else int(np.log10(max_val)) + 1
    for i in range(max_digits):
        df[f"{col}_digit_pos_{i+1}"] = (df[col] // (10**i)) % 10
    if df[col].dtype == 'float64':
        if (df[col] % 1 != 0).any():
            for i in range(1, 3):
                df[f"{col}_digit_dec_{i}"] = (df[col] * (10**i)).round().astype(int) % 10
```

### 3.3 String Categorical Copies (`ALL_CATS`)
Converting every continuous variable to string format (`f"{col}_cat"`). In neural models like RealMLP, these are mapped into learned categorical embeddings, establishing a high-dimensional nonlinear coordinate space.

### 3.4 Frequency Encoding
Adding the value count frequencies for all features. Frequency encoding captures the prevalence of rare combinations and complements target encoding in tree splits.

### 3.5 Genetic Programming Non-Linear Interactions (`gplearn`)
Using symbolic genetic programming (`SymbolicTransformer` via `gplearn`) to automatically synthesize nonlinear interaction features (e.g., ratios, differences, and compound formulas).

### 3.6 External Signals from the Original Dataset
Because Playground data is synthetic, priors extracted from the original [Heart Disease Prediction dataset](https://www.kaggle.com/datasets/heartdisease/Heart_Disease_Prediction) provided valuable regularization:
- Target mean & Bayesian smoothed target mean
- Weight of Evidence (WoE)
- Information Entropy per categorical group

### 3.7 Denoising Variational Autoencoder (DVAE)
A DVAE trained with Gaussian input noise learned compressed latent bottleneck representations of the clinical features. The latent dimensions served as orthogonal representations for downstream tree models.

---

## 4. Model Zoo Performance & Diversity Ranking

The author generated $\approx 150$ OOF prediction streams. Below are representative single-model performance figures for models frequently chosen by the Optuna ensemble search:

| Feature Set Representation | Model Family | 5-Fold OOF ROC-AUC | Key Architectural Nuance |
| :--- | :--- | :--- | :--- |
| `BASE + BIN + DIGIT + ALL_CATS` | **AutoGluon** | **0.955747** | Multi-layer stack with internal presets |
| `BASE + BIN + DIGIT + ALL_CATS` | **RealMLP** | **0.955739** | Piecewise Linear Embeddings (PLR) + Mish |
| `ORIG + TE + EMB` | **RealMLP** | **0.955726** | Original dataset target encoding + entity embeddings |
| `BASE + GP_FEAT + ALL_CATS` | **RealMLP** | **0.955720** | Genetic programming nonlinear features |
| `BASE + BIN + DIGIT + ALL_CATS` | **CatBoost** | **0.955686** | Ordered boosting with categorical splits |
| `BASE + TE` | **XGBoost** | **0.955663** | Greedy histogram splits on target encodings |
| `BASE + BIN + DIGIT + ALL_CATS + FREQ` | **LightGBM** | **0.955652** | Leaf-wise tree growth with frequency features |
| `BASE + ALL_CATS` | **XGBoost** | **0.955619** | Categorical string embeddings |
| `ORIG + ALL_CATS` | **XGBoost** | **0.955599** | Original data distribution statistics |
| `BASE` | **XGBoost** | **0.955575** | Clean baseline features |
| `DVAE + ALL_CATS` | **XGBoost** | **0.955426** | Denoising VAE latent space representations |
| `BASE + BIN + DIGIT + ALL_CATS` | **RGF** | **0.954980** | Regularized Greedy Forest ($L_2$ leaf penalty) |
| `BASE` (subsample $100\text{k} \times 5$) | **TabICL** | **0.954971** | In-Context Learning for Tabular Data |

> [!NOTE]
> **Ensemble Contribution $\ne$ Standalone CV**:
> Note that models with lower standalone CV scores (such as RGF at `0.95498` and TabICL at `0.95497`) were selected frequently during Optuna search. Their orthogonal inductive biases provided the ensemble with non-correlated errors.

---

## 5. Optuna Subset Selection & Ridge Ensembling

### The 2,500-Trial Optuna Search
With 150 candidate models, simple uniform averaging diluted strong models, while unconstrained linear regression suffered from extreme multicollinearity. The solution was:
1. Formulate a binary/continuous selection vector $\mathbf{w} \in [0, 1]^{150}$.
2. Run Optuna across **2,500 trials** to search for model subsets that maximize overall out-of-fold ROC-AUC.
3. Only $\approx 15 - 25$ models were consistently selected across trials.

### Stacking with Ridge Regression
Selected model predictions were blended using $L_2$-regularized **Ridge Regression**:
- **Why Ridge?** Unlike unregularized linear regression or logistic regression, Ridge handles near-collinear prediction vectors ($r > 0.98$) gracefully without exploding coefficients.
- **Why Not GBDT/MLP Meta-Models?** Nonlinear stackers memorized fold quirks and degraded out-of-sample test accuracy.

### Full-Data Retraining Protocol
For GBDT and RGF models selected for the final ensemble:
1. Retrain on $100\%$ of the training data (all 630,000 rows).
2. Set `n_estimators` $= 1.25 \times \text{mean}(\text{best\_iteration across 5 folds})$.
3. Train across **20 different random seeds** and average test predictions.

---

## 6. The Central Case Study: "Trust the CV–LB Relation"

```text
Cross-Validation ROC-AUC      Public Leaderboard (LB)       Generalization Reality
──────────────────────────────────────────────────────────────────────────────────
0.9520  →  0.9550             Linear Improvement            Safe: Under-fit models progressing
0.9550  →  0.95578            Strong Correlation            Optimal: Genuine generalization
0.95578 →  0.955865+          Degrades / Flatlines          Trap: Overfitting the CV splits!
```

### What is Split Overfitting?
When tuning dozens of models and using meta-optimizers (like Optuna) over fixed validation folds, the optimization process begins exploiting idiosyncrasies of those specific 5 folds. 

Even though early stopping within standard K-Fold is standard practice, the validation fold indirectly influences model checkpointing. When hundreds of models are combined, this subtle selection bias compounds into **split overfitting**.

The 1st-place winner carefully submitted ensembles stopped at various CV tiers. While CV reached $0.955865$, the author observed that Public LB stalled and degraded beyond $0.95578$. By consciously selecting the submission from the **$0.9557801$** band, the model held up on the private test set ($0.95535$) to win 1st place overall.

---

## 7. Key Takeaways & Anti-Patterns

### ✅ What Won the Gold Medal
- **Meaningful Diversity**: Combining tree-based models, tabular MLPs (RealMLP), in-context tabular learners (TabICL), and regularized forests (RGF).
- **Multiple Feature Representations**: Quantile cuts, uniform cuts, domain rounding, and decimal digit positions.
- **Disciplined Selection**: Pruning down from 150 models to $\approx 20$ using Optuna before stacking.
- **Simple Meta-Learners**: Ridge regression for linear stability on correlated probability vectors.
- **Full-Data Multi-Seed Retraining**: Averaging 20 seeds with $1.25\times$ iterations on the full dataset.

### ❌ What Failed / Degraded Performance
- **Pseudo-Labeling**: Both hard and soft pseudo-labeling failed to improve CV.
- **Knowledge Distillation**: Compressing the ensemble into a single student network lost ensemble diversity.
- **Overly Deep Trees**: GBDTs with depth $> 8$ quickly overfit the synthetic data generator.
- **Nonlinear Stacking**: Multi-layer perceptrons or LightGBM as meta-learners severely overfit the OOF predictions.
- **Uncurated Stacking**: Averaging all 150 models without Optuna selection yielded worse performance than top solo models.
