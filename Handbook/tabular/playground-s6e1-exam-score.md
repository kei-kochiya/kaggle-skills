# Playground Series S6E1: Predicting Student Test Scores

**Competition**: [Kaggle Playground Series - Season 6 Episode 1](https://www.kaggle.com/competitions/playground-series-s6e1)  
**1st Place Solution**: *(I've ran out of catchy phrases :V)* by Mahog  
**Track**: Tabular Regression  
**Evaluation Metric**: Root Mean Squared Error ($\text{RMSE}$)  
**Winning Scores**: 
- **Cross-Validation (CV)**: `8.56634`
- **Public / Private Leaderboard (LB)**: `8.57273`

---

## 1. Executive Summary & Core Breakthroughs

Playground Series S6E1 tasked participants with predicting a student's `exam_score` based on demographic and academic behavioral attributes. Like many Playground competitions, the dataset was synthetically generated from a baseline dataset ([`Exam_Score_Prediction.csv`](https://www.kaggle.com/datasets/exam-score-prediction-dataset)).

The competition was won not by raw brute-force deep learning alone, but by a masterclass in **tabular feature reverse-engineering**, **deep tabular neural networks (RealMLP)**, and **monotonic calibration before and after linear meta-stacking**:

1. **Synthetic Formula Reverse-Engineering**: Competitors detected that synthetic dataset generators (e.g. CTGAN or modified copulas) preserved linear relationships from underlying formulas with additive noise. Incorporating exact reconstructed formulas provided substantial CV gains.
2. **Harmonic / Trigonometric Cyclic Encodings**: Numerical study hours and attendance rates exhibited cyclic patterns across periods $p \in \{12, 14, 20\}$.
3. **Digit-Level Modulo Features**: Decomposing continuous variables into decimal digits ($10^k \pmod{10}$) exposed non-smooth thresholds and artifact boundaries to tree and neural models.
4. **Leak-Free Multi-Aggregation Target Encoding**: Nested out-of-fold target encoding computing `mean`, `std`, and `skew` with Empirical Bayes smoothing.
5. **RealMLP (`pytabkit`) Neural Networks**: Tabular neural networks using piecewise linear representations (PLR) matched and frequently outperformed tuned GBDTs on individual folds.
6. **Centered Isotonic Regression (CIR) + Ridge Stacking**: Instead of raw OOF stacking, fitting a non-parametric `CenteredIsotonicRegression` onto each base model's predictions prior to $L_2$-regularized `RidgeCV` meta-learning, followed by a final CIR post-processing pass.

---

## 2. Dataset Architecture & Exploratory Data Analysis

### Data Specs
- **Training Set**: 630,000 rows $\times$ 12 features
- **Test Set**: 270,000 rows $\times$ 11 features
- **Target**: `exam_score` (bounded continuous variable in $[0.0, 100.0]$)
- **Original Dataset**: Available as `Exam_Score_Prediction.csv` (used as an auxiliary training set).

### Feature Inventory
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `age` | Numerical | Age in years (discrete integers) |
| `gender` | Categorical | `female`, `male`, `other` |
| `course` | Categorical | `b.sc`, `diploma`, `bca`, etc. |
| `study_hours` | Numerical | Hours studied per week / day |
| `class_attendance` | Numerical | Percentage attendance $[0, 100]$ |
| `internet_access` | Categorical | `yes`, `no` |
| `sleep_hours` | Numerical | Average sleep duration |
| `sleep_quality` | Categorical | `poor`, `average`, `good` |
| `study_method` | Categorical | `self-study`, `coaching`, `group study`, `online videos`, `mixed` |
| `facility_rating` | Categorical | `low`, `medium`, `high` |
| `exam_difficulty` | Categorical | `easy`, `moderate`, `hard` |

---

## 3. The Feature Engineering Playbook

### 3.1 Reverse-Engineered Data Generator Formulas
Through linear regression, symbolic regression, and residual analysis of the original dataset, top competitors discovered the structural relationship driving `exam_score`:

```python
# Primary continuous linear formula
train['feature_formula'] = (
    5.9051154511950499 * train['study_hours'] +
    0.34540967058057986 * train['class_attendance'] +
    1.423461171860262 * train['sleep_hours'] + 
    4.7819
)

# Extended formula incorporating discrete categorical offsets
train['_feature_formula'] = (
    6.0 * train['study_hours'] + 
    0.35 * train['class_attendance'] + 
    1.5 * train['sleep_hours'] +
    5.0 * (train['sleep_quality'] == 'good') + 
    -5.0 * (train['sleep_quality'] == 'poor') +
    10.0 * (train['study_method'] == 'coaching') + 
    5.0 * (train['study_method'] == 'mixed') + 
    2.0 * (train['study_method'] == 'group study') + 
    1.0 * (train['study_method'] == 'online videos') +
    4.0 * (train['facility_rating'] == 'high') + 
    -4.0 * (train['facility_rating'] == 'low')
)
```

### 3.2 Trigonometric / Periodic Harmonic Features
Periodic sine and cosine encodings over periods $p \in \{12, 14, 20\}$ captured nonlinear periodicities in continuous variables:

```python
for p in [12, 14, 20]:
    train[f"_study_hours_sin_{p}"] = np.sin(2 * np.pi * train['study_hours'] / p).astype('float32')
    train[f"_study_hours_cos_{p}"] = np.cos(2 * np.pi * train['study_hours'] / p).astype('float32')
    train[f"_class_attendance_sin_{p}"] = np.sin(2 * np.pi * train['class_attendance'] / p).astype('float32')
```

### 3.3 Modulo Digit-Level Decomposition
Continuous values often have digit-level rounding artifacts introduced during data generation. Extracting individual digits ($10^k \pmod{10}$) directly exposed those boundaries:

```python
for c in ['study_hours']:
    for k in range(0, 3):
        n = f'{c}_d{k}'
        train[n] = ((train[c] * 10**k) % 10).fillna(-1).astype("int8")
        test[n] = ((test[c] * 10**k) % 10).fillna(-1).astype("int8")

for c in ['class_attendance']:
    for k in range(-1, 2):
        n = f'{c}_d{k}'
        train[n] = ((train[c] * 10**k) % 10).fillna(-1).astype("int8")
        test[n] = ((test[c] * 10**k) % 10).fillna(-1).astype("int8")
```

### 3.4 Multi-Aggregation OOF Target Encoding with Empirical Bayes Smoothing
Standard target encoding computes only category means and often leaks if not strictly computed out-of-fold. The winning implementation computed `mean`, `std`, and `skew`, applying Empirical Bayes smoothing:

$$\text{Smoothed Mean} = \frac{n \cdot \bar{y}_c + m \cdot \bar{y}_{\text{global}}}{n + m}, \quad m = \frac{\sigma^2_{\text{within}}}{\sigma^2_{\text{between}}}$$

```python
class TargetEncoder_(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_encode, aggs=['mean', 'std', 'skew'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original

    def fit_transform(self, X, y):
        self.fit(X, y)
        encoded_features = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            temp_df = X_train.copy()
            temp_df['target'] = y_train

            for col in self.cols_to_encode:
                for agg_func in self.aggs:
                    new_col = f'TE_{col}_{agg_func}'
                    fold_global = y_train.agg(agg_func)
                    mapping = temp_df.groupby(col)['target'].agg(agg_func)

                    if agg_func == 'mean':
                        counts = temp_df.groupby(col)['target'].count()
                        if self.smooth == 'auto':
                            var_between = mapping.var()
                            var_within = temp_df.groupby(col)['target'].var().mean()
                            m = (var_within / var_between) if var_between > 0 else 0
                        else:
                            m = self.smooth
                        smoothed = (counts * mapping + m * fold_global) / (counts + m)
                        encoded_features.loc[X_val.index, new_col] = X_val[col].map(smoothed).fillna(fold_global)
                    else:
                        encoded_features.loc[X_val.index, new_col] = X_val[col].map(mapping).fillna(fold_global)

        X_out = X.copy()
        for col in encoded_features.columns:
            X_out[col] = encoded_features[col]
        return X_out
```

---

## 4. Modeling & Architectures

### 4.1 RealMLP (`pytabkit`) Neural Network
The single best model class was **RealMLP** (`RealMLP_TD_Regressor`), a modern neural network architecture tailored for tabular data from the `pytabkit` library:
- **Piecewise Linear Representation (PLR)** for numerical features with embedding size 8.
- **Mish Activation Function**: Smooth non-monotonic gating.
- **Cosine Log Learning Rate Scheduler (`coslog4`)** with weight decay $0.0237$.
- **Label Smoothing Epsilon**: $0.0115$ preventing overconfident predictions near boundaries.
- **Ensemble Depth**: $n_{\text{ens}} = 8$ internal sub-models averaged per fold.
- **Single Model Result**: Single RealMLP model achieved **CV 8.58748 / LB 8.58006**.

### 4.2 GBDT Diversity Zoo
The solution ensembled over 190 distinct configurations across:
- **XGBoost**: Diverse tree depths (3 to 10), subsample ratios (0.6 to 0.9), and regularizations.
- **LightGBM & DART**: Leaf-wise growth, `max_depth` regularization, DART dropout rates.
- **CatBoost**: Categorical feature combinations with ordered boosting.
- **Other Tabular NNs**: TabM, Trompt, ResNet-Tabular, AutoInt, Gandalf, DeepTables, and xLearn Field-aware Factorization Machines (FFM).

---

## 5. The 1st-Place Ensembling & Post-Processing Pipeline

The standout algorithmic contribution was the **three-stage Stacking & Calibration Pipeline**:

```
[ 190 Base Models OOF ] 
          │
          ▼
[ Step 1: Centered Isotonic Regression (CIR) Calibration on each model ]
          │
          ▼
[ Step 2: RidgeCV Meta-Regression Stacking (alphas=[1.0]) ]
          │
          ▼
[ Step 3: Secondary CIR Monotonic Calibration on Stacked Predictions ]
          │
          ▼
[ Final Submission: CV 8.56634 / LB 8.57273 ]
```

### Why Centered Isotonic Regression (CIR)?
Standard isotonic regression enforces monotonicity ($y_i \le y_j$ when $\hat{y}_i \le \hat{y}_j$) but can shift prediction medians or cause boundary bias. `CenteredIsotonicRegression` preserves calibration while ensuring predictions remain centered on the target mean.

Applying CIR to every base model's predictions *before* feeding them into Ridge:
1. Aligns the non-linear scale of tree models and neural models.
2. Removes local calibration drift across different folds.
3. Linearizes the relationship between predictions and the ground truth, making the subsequent `Ridge` meta-model strictly optimal.

```python
from cir_model import CenteredIsotonicRegression
from sklearn.linear_model import RidgeCV

# 1. Monotonic calibration of all base models
for i in range(x_train.shape[1]):
    cir = CenteredIsotonicRegression()
    cir.fit(x_train[:, i], y_true)
    x_train[:, i] = cir.transform(x_train[:, i])
    x_test[:, i] = cir.transform(x_test[:, i])

# 2. Ridge Stacking
ridge = RidgeCV(alphas=[1.0])
ridge.fit(x_train, y_true)
y_pred_oof = ridge.predict(x_train)
y_pred_test = ridge.predict(x_test)

# 3. Post-Stacking CIR Calibration
cir_post = CenteredIsotonicRegression()
cir_post.fit(y_pred_oof, y_true)
final_test_preds = cir_post.transform(y_pred_test)
```

---

## 6. Key Takeaways & Anti-Patterns

### ✅ What Worked
- **Target Encoding with Higher-Order Moments**: Adding `std` and `skew` captured conditional variance that tree splits alone missed.
- **Synthetic Equation Reconstruction**: Always inspect Playground datasets with simple linear regression and symbolic regression first.
- **CIR Before Stacking**: Pre-calibrating OOF predictions gave an instant $\approx 0.005$ RMSE drop over raw predictions.
- **Model Diversity**: Combining RealMLP + GBDT + FFM gave far greater ensembling lift than stacking multiple variations of XGBoost alone.

### ❌ What Failed / Overfit
- **Unsmoothed Target Encoding**: Raw target encoding severely overfit the validation folds due to high-cardinality combinatorial interactions.
- **Non-Regularized Deep Stacking**: Training a Multi-Layer Perceptron (MLP) or XGBoost as a level-2 meta-learner overfit the 190 OOF features. Simple regularized `Ridge` was strictly superior.
- **Raw Clamping to $[0, 100]$**: Clamping directly without smooth calibration produced artifacts at the extremes. CIR naturally handled the boundary distribution.
