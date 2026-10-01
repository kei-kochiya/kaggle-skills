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

---

## 10. Multi-Granularity Target Encoding Pyramids

Never target-encode a continuous column at only a single scale. Build a multi-tier coarse-to-fine quantization pyramid inside nested out-of-fold splits:

```python
from sklearn.preprocessing import TargetEncoder

def build_multigranularity_te_pyramid(train_df, test_df, continuous_col, target_col, cv=5):
    """
    Builds a 4-tier quantization pyramid of target encodings:
    1. Raw continuous value (fine-grained density)
    2. Rounded to 100 (sub-cluster density)
    3. Rounded to 1,000 (macro-cluster density)
    4. Rounded to 10,000 or quantile bin (global monotonic prior)
    """
    K_train = pd.DataFrame(index=train_df.index)
    K_test = pd.DataFrame(index=test_df.index)
    
    vals_tr = train_df[continuous_col]
    vals_te = test_df[continuous_col]
    
    # 4-tier quantization pyramid
    K_train['fine'] = vals_tr.astype(str)
    K_test['fine'] = vals_te.astype(str)
    
    K_train['round_100'] = (vals_tr // 100 * 100).astype(str)
    K_test['round_100'] = (vals_te // 100 * 100).astype(str)
    
    K_train['round_1k'] = (vals_tr // 1000 * 1000).astype(str)
    K_test['round_1k'] = (vals_te // 1000 * 1000).astype(str)
    
    K_train['quant_bin'] = pd.qcut(vals_tr, q=20, labels=False, duplicates='drop').astype(str)
    # Map test using train quantile boundaries
    _, bins = pd.qcut(vals_tr, q=20, retbins=True, duplicates='drop')
    K_test['quant_bin'] = pd.cut(vals_te, bins=bins, labels=False, include_lowest=True).fillna(-1).astype(str)
    
    # Smooth fold-safe out-of-fold target encoding
    keys = ['fine', 'round_100', 'round_1k', 'quant_bin']
    te = TargetEncoder(target_type='binary', smooth='auto', cv=cv, shuffle=True, random_state=42)
    
    te_tr = te.fit_transform(K_train[keys], train_df[target_col])
    te_te = te.transform(K_test[keys])
    
    for i, k in enumerate(keys):
        train_df[f"{continuous_col}_te_{k}"] = te_tr[:, i]
        test_df[f"{continuous_col}_te_{k}"] = te_te[:, i]
        
    return train_df, test_df
```

---

## 11. Offline Original Data Feature Lookup Prior (Zero-Leak Transfer)

When competing on synthetic tabular datasets generated from an original reference dataset (e.g. S6E9, S6E10):
- **Stacking original rows directly into training sets HURTS GBDTs** (e.g. $-0.00025$ AUC in S6E10) because tree splitters overfit the clean historical distribution.
- **Using original data as an offline static target-mean lookup prior GAINS $+0.00093$ to $+0.00103$ AUC** across all folds with zero data leakage:

```python
def attach_original_lookup_prior(train_df, test_df, original_df, feature_cols, target_col):
    """
    Computes satisfaction/target rates from the ORIGINAL historical dataset
    and maps them to competition train and test frames.
    Completely zero leak: depends solely on the external original dataset.
    """
    lookup_tables = {c: original_df.groupby(c)[target_col].mean() for c in feature_cols}
    
    for c in feature_cols:
        train_df[f"orig_rate__{c}"] = train_df[c].map(lookup_tables[c]).astype('float64')
        test_df[f"orig_rate__{c}"] = test_df[c].map(lookup_tables[c]).astype('float64')
        
    return train_df, test_df
```

---

## 12. BPE Subword Token Group Encodings on Numerical Strings

Modern tabular synthesizers (e.g. GReaT, LLM-based generators) emit continuous numbers as text tokens using Byte-Pair Encoding (BPE) tokenizers (like GPT-2 or LLaMA):
- Numerical strings are chopped into discrete subword chunks: `50000` $\to$ `Ġ5` + `0000`, `75000` $\to$ `Ġ75` + `000`.
- Groups defined by token prefixes and token lengths capture synthesizer density spikes that decimal operations miss (+0.00027 CV lift in S6E9):

```python
import tiktoken
from sklearn.preprocessing import TargetEncoder

def add_bpe_token_encodings(train_df, test_df, num_col, target_col, cv=5):
    """
    Tokenizes integer strings with GPT-2 BPE and computes target encodings
    over token prefixes, suffixes, and token-length patterns.
    """
    enc = tiktoken.get_encoding("gpt2")
    
    for df in [train_df, test_df]:
        # Prefix leading space to match GPT-2 word-start convention
        tokens = [enc.encode(" " + str(int(v))) if pd.notna(v) else [] for v in df[num_col]]
        df[f"{num_col}_tok_len"] = [len(t) for t in tokens]
        df[f"{num_col}_first_tok"] = [t[0] if len(t) > 0 else -1 for t in tokens]
        df[f"{num_col}_last_tok"] = [t[-1] if len(t) > 0 else -1 for t in tokens]
        
    # Target-encode token categories out-of-fold
    tok_keys = [f"{num_col}_tok_len", f"{num_col}_first_tok", f"{num_col}_last_tok"]
    te = TargetEncoder(target_type='binary', smooth='auto', cv=cv, shuffle=True, random_state=42)
    
    te_tr = te.fit_transform(train_df[tok_keys].astype(str), train_df[target_col])
    te_te = te.transform(test_df[tok_keys].astype(str))
    
    for i, k in enumerate(tok_keys):
        train_df[f"{k}_te"] = te_tr[:, i]
        test_df[f"{k}_te"] = te_te[:, i]
        
    return train_df, test_df
```


