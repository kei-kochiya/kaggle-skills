# Tabular Feature Engineering Toolkit

Reusable feature engineering recipes extracted from gold-medal tabular solutions.

---

## 1. Reverse-Engineered Synthetic Formulas

When analyzing datasets from Kaggle Playground, check for continuous linear equations with categorical shift parameters:

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

def discover_formula(df_train, num_cols, cat_cols, target_col):
    """
    Fits Ridge to estimate underlying coefficients and displays continuous weightings.
    """
    X_num = df_train[num_cols].fillna(df_train[num_cols].median())
    y = df_train[target_col]
    
    model = Ridge(alpha=1.0).fit(X_num, y)
    print("Estimated Linear Weights:")
    for col, coef in zip(num_cols, model.coef_):
        print(f"  {col}: {coef:.6f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    return model
```

---

## 2. Harmonic / Periodic Trigonometric Transformations

Continuous variables showing cyclic periodicity (e.g., hours of day, days of year, attendance percentages) benefit from sine and cosine embeddings across multiple harmonic periods:

```python
def add_periodic_features(df, col, periods=[12, 14, 20]):
    """
    Appends sine and cosine harmonic waves for given period lengths.
    """
    for p in periods:
        df[f"{col}_sin_{p}"] = np.sin(2 * np.pi * df[col] / p).astype('float32')
        df[f"{col}_cos_{p}"] = np.cos(2 * np.pi * df[col] / p).astype('float32')
    return df
```

---

## 3. Modulo Digit-Level Decomposition

Extracting decimal digits helps tree-based models and neural networks detect boundary thresholds and rounding artifacts without requiring deep trees:

```python
def extract_digits(df, col, power_range=(0, 2)):
    """
    Extracts base-10 digits: (x * 10^k) % 10.
    """
    for k in range(power_range[0], power_range[1] + 1):
        feature_name = f"{col}_d{k}"
        df[feature_name] = ((df[col] * (10**k)) % 10).fillna(-1).astype('int8')
    return df
```

---

## 4. Combinatorial Interaction Strings

Combine categorical and discretely binned numerical variables into joint identifiers for high-order interaction capture:

```python
from itertools import combinations

def create_pairwise_interactions(train_df, test_df, columns, max_cardinality_ratio=0.5):
    """
    Generates factorized pairwise combination features with frequency filtering.
    """
    new_cols = []
    for c1, c2 in combinations(columns, 2):
        name = f"{c1}_{c2}"
        combined = train_df[c1].astype(str) + "_" + train_df[c2].astype(str)
        combined_test = test_df[c1].astype(str) + "_" + test_df[c2].astype(str)
        
        full_series = pd.concat([combined, combined_test], ignore_index=True)
        factorized, _ = full_series.factorize()
        
        # Check cardinality
        if pd.Series(factorized).nunique() <= len(full_series) * max_cardinality_ratio:
            train_df[name] = factorized[:len(train_df)]
            test_df[name] = factorized[len(train_df):]
            new_cols.append(name)
    return train_df, test_df, new_cols
```

---

## 5. Multi-Scale & Domain-Specific Binning

Generating multiple representations of continuous variables via different discretization schemes creates complementary signals for tree and neural architectures:

```python
def create_multiscale_bins(df, target_cols, quantiles=[4, 5, 10, 20], uniform_cuts=[5, 10]):
    """
    Generates quantile (qcut), uniform (cut), and domain-specific rounding bins.
    """
    bin_cols = []
    for col in target_cols:
        # 1. Quantile-based (equal frequency per bin)
        for q in quantiles:
            name = f"{col}_bin_q{q}"
            try:
                df[name] = pd.qcut(df[col], q=q, labels=False, duplicates='drop')
                bin_cols.append(name)
            except ValueError:
                pass
                
        # 2. Uniform interval cuts
        for b in uniform_cuts:
            name = f"{col}_bin_cut{b}"
            df[name] = pd.cut(df[col], bins=b, labels=False)
            bin_cols.append(name)
            
    for col in bin_cols:
        df[col] = df[col].fillna(-1).astype(int)
    return df, bin_cols
```

---

## 6. Synthetic Snap Matching & Perturbation Diff

When synthetic tabular data (e.g. CTGAN/TVAE) is generated from an original historical dataset, continuous floats are perturbed versions of real customer records. Map each float to its nearest original neighbor and measure the perturbation delta:

```python
def compute_snap_features(train_df, test_df, orig_df, cols):
    """
    Finds nearest original values via searchsorted and extracts perturbation noise.
    """
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
            df[f"{col}_snap_diff"] = vals - nearest
    return train_df, test_df
```

---

## 7. Radix Joint Continuous-Categorical Split Encoding

Allows decision trees to make a joint split along continuous and categorical boundaries simultaneously with a single tree split:

```python
def create_radix_features(df, snap_num_col, cat_col, num_multiplier=100, cat_offset=100000):
    """
    Encodes (continuous_snap, categorical) as a single composite integer.
    Example: int(MonthlyCharges_snap * 100) + cat_code * 100_000
    """
    cat_codes = df[cat_col].astype('category').cat.codes
    radix = (df[snap_num_col] * num_multiplier).round().astype(int) + (cat_codes * cat_offset)
    return radix.astype('int64')
```

---

## 8. cKDTree Nearest-Neighbor Ground-Truth Prior

Queries a spatial KDTree built on the standardized numerical features of the original dataset, attaching the true historical label as a zero-leakage prior feature:

```python
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

def attach_orig_kdtree_prior(train_df, test_df, orig_df, match_cols, target_col):
    """
    Attaches nearest real-world neighbor's target label to synthetic data.
    """
    scaler = StandardScaler()
    orig_scaled = scaler.fit_transform(orig_df[match_cols].fillna(0))
    tree = cKDTree(orig_scaled)
    
    orig_targets = orig_df[target_col].to_numpy()
    
    for df in [train_df, test_df]:
        df_scaled = scaler.transform(df[match_cols].fillna(0))
        dists, idxs = tree.query(df_scaled, k=1)
        df[f"{target_col}_orig_prior"] = orig_targets[idxs]
        df[f"{target_col}_orig_dist"] = dists
    return train_df, test_df
```

---

## 9. Benford's Law Likelihood & Generator Anomaly Detection

Measures deviation of leading digits against Benford's logarithmic distribution ($P(d) = \log_{10}(1 + 1/d)$) to detect synthetic artifacts:

```python
def extract_benford_deviation(df, num_col):
    """
    Extracts leading non-zero digit and computes deviation from Benford's Law.
    """
    benford_probs = {d: np.log10(1 + 1.0 / d) for d in range(1, 10)}
    
    str_vals = df[num_col].abs().astype(str).str.replace(r"^0+\.?", "", regex=True)
    leading_digit = str_vals.str[0].astype(float).fillna(1).clip(1, 9).astype(int)
    
    expected_prob = leading_digit.map(benford_probs)
    df[f"{num_col}_lead_digit"] = leading_digit
    df[f"{num_col}_benford_expected"] = expected_prob
    return df
```


