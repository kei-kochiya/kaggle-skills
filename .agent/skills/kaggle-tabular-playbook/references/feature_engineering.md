# Tabular Feature Engineering Toolkit

Reusable tabular feature engineering patterns organized into **Universal Core Recipes** (applicable to both real-world and synthetic tabular data) and **Optional Synthetic-Generator Recipes** (activate only after exploratory forensics confirm generator artifacts).

---

## Part I: Universal Core Feature Engineering

### 1. Combinatorial Pairwise Interactions

Combine categorical and discretely binned numerical variables into joint identifiers with frequency/cardinality caps:

```python
from itertools import combinations
import numpy as np
import pandas as pd


def create_pairwise_interactions(train_df, test_df, columns, max_cardinality_ratio=0.5):
    """Generate factorized pairwise combination features with cardinality guard."""
    new_cols = []
    for c1, c2 in combinations(columns, 2):
        name = f"{c1}__{c2}"
        combined_tr = train_df[c1].astype(str) + "_" + train_df[c2].astype(str)
        combined_te = test_df[c1].astype(str) + "_" + test_df[c2].astype(str)

        full_series = pd.concat([combined_tr, combined_te], ignore_index=True)
        factorized, uniques = full_series.factorize()

        if len(uniques) <= len(full_series) * max_cardinality_ratio:
            train_df[name] = factorized[: len(train_df)].astype("int32")
            test_df[name] = factorized[len(train_df) :].astype("int32")
            new_cols.append(name)
    return train_df, test_df, new_cols
```

---

### 2. Multi-Scale Discretization (Fit Bin Edges Strictly on Train)

Generating quantile and uniform interval representations exposes non-linear thresholds to both trees and embedding-based neural networks. Always fit bin edges on `train_df` and apply identically to `test_df`:

```python
def create_multiscale_bins(train_df, test_df, cols, quantiles=(5, 10, 20), uniform_cuts=(5, 10)):
    """Fit quantile and uniform bin edges on train_df and transform both train and test."""
    bin_cols = []
    for col in cols:
        for q in quantiles:
            name = f"{col}_bin_q{q}"
            try:
                _, edges = pd.qcut(train_df[col], q=q, retbins=True, duplicates="drop")
                edges[0], edges[-1] = -np.inf, np.inf
                train_df[name] = pd.cut(train_df[col], bins=edges, labels=False).fillna(-1).astype("int16")
                test_df[name] = pd.cut(test_df[col], bins=edges, labels=False).fillna(-1).astype("int16")
                bin_cols.append(name)
            except ValueError:
                pass

        for b in uniform_cuts:
            name = f"{col}_bin_cut{b}"
            _, edges = pd.cut(train_df[col], bins=b, retbins=True)
            edges[0], edges[-1] = -np.inf, np.inf
            train_df[name] = pd.cut(train_df[col], bins=edges, labels=False).fillna(-1).astype("int16")
            test_df[name] = pd.cut(test_df[col], bins=edges, labels=False).fillna(-1).astype("int16")
            bin_cols.append(name)

    return train_df, test_df, bin_cols
```

---

### 3. Scale-Adaptive Multi-Granularity Target Encoding Pyramids

Encode high-signal numeric columns across coarse-to-fine scales inside nested out-of-fold splits. Derive rounding steps dynamically from the column's interquartile range (`IQR`) so the pyramid works across arbitrary feature scales (whether `[0, 1]` ratios or `[10^3, 10^6]` monetary figures):

```python
from sklearn.preprocessing import TargetEncoder


def build_multigranularity_te_pyramid(
    train_df, test_df, continuous_col, target_col, target_type="binary", cv=5
):
    """
    Build a scale-adaptive 4-tier quantization pyramid of out-of-fold target encodings:
    1. Exact string value (fine-grained exact-value signal)
    2. Medium scale-adaptive rounding (0.1x base magnitude of IQR)
    3. Coarse scale-adaptive rounding (1.0x base magnitude of IQR)
    4. Train-fitted 20-quantile bin (global monotonic prior)
    """
    vals_tr = train_df[continuous_col]
    vals_te = test_df[continuous_col]

    iqr = float(vals_tr.quantile(0.75) - vals_tr.quantile(0.25))
    scale = iqr if iqr > 0 else float(vals_tr.std() or 1.0)
    coarse_step = 10.0 ** np.floor(np.log10(scale))
    fine_step = coarse_step / 10.0

    K_train = pd.DataFrame(index=train_df.index)
    K_test = pd.DataFrame(index=test_df.index)

    K_train["exact"] = vals_tr.astype(str)
    K_test["exact"] = vals_te.astype(str)

    K_train["fine_round"] = (np.floor(vals_tr / fine_step) * fine_step).round(6).astype(str)
    K_test["fine_round"] = (np.floor(vals_te / fine_step) * fine_step).round(6).astype(str)

    K_train["coarse_round"] = (np.floor(vals_tr / coarse_step) * coarse_step).round(6).astype(str)
    K_test["coarse_round"] = (np.floor(vals_te / coarse_step) * coarse_step).round(6).astype(str)

    _, bins = pd.qcut(vals_tr, q=20, retbins=True, duplicates="drop")
    bins[0], bins[-1] = -np.inf, np.inf
    K_train["quant_bin"] = pd.cut(vals_tr, bins=bins, labels=False).fillna(-1).astype(str)
    K_test["quant_bin"] = pd.cut(vals_te, bins=bins, labels=False).fillna(-1).astype(str)

    keys = ["exact", "fine_round", "coarse_round", "quant_bin"]
    te = TargetEncoder(target_type=target_type, smooth="auto", cv=cv, shuffle=True, random_state=42)
    te_tr = te.fit_transform(K_train[keys], train_df[target_col])
    te_te = te.transform(K_test[keys])

    for i, k in enumerate(keys):
        train_df[f"{continuous_col}_te_{k}"] = te_tr[:, i].astype("float32")
        test_df[f"{continuous_col}_te_{k}"] = te_te[:, i].astype("float32")

    return train_df, test_df
```

---

### 4. Periodic / Harmonic Transformations (Data-Driven Periods)

When domain cycles (hours, days, angles) or FFT spectrum forensics reveal genuine periodicity at periods `periods`, add sine and cosine projections rather than raw wrap-around integers:

```python
def add_periodic_features(df, col, periods):
    """Append sine and cosine harmonic features for domain- or FFT-confirmed periods."""
    for p in periods:
        df[f"{col}_sin_{p}"] = np.sin(2.0 * np.pi * df[col] / p).astype("float32")
        df[f"{col}_cos_{p}"] = np.cos(2.0 * np.pi * df[col] / p).astype("float32")
    return df
```

---

### 5. External / Original Dataset Transfer (Ablate Lookup Priors vs. Weighted Row Appending)

When an external or original seed dataset is available alongside the competition training set, **never assume one injection strategy always wins**:
- **Static Lookup Priors (`orig_rate__*`)**: Zero-leakage feature transfer. Strictly better when competition data carries synthetic label noise or generator distortions that cause GBDTs to overfit clean original rows.
- **Synchronous Sample-Weighted Row Appending**: Merging original rows into training folds (with sample weight `w ∈ [0.25, 1.25]` and host-only CV validation — see [`logit_stacking_rank_blend.md`](./logit_stacking_rank_blend.md)) often yields strong gains when original and competition target distributions align closely.
- **Best Practice**: Ablate lookup priors, weighted row appending, and both combined on honest host-only CV (and consider keeping one model stream of each for ensemble diversity!).

```python
def attach_external_lookup_prior(train_df, test_df, external_df, feature_cols, target_col):
    """
    Compute target-mean lookup tables strictly from an independent external/original
    dataset and map them onto competition train and test frames (zero CV leakage).
    """
    global_mean = float(external_df[target_col].mean())
    for c in feature_cols:
        lookup = external_df.groupby(c)[target_col].mean()
        train_df[f"ext_rate__{c}"] = train_df[c].map(lookup).fillna(global_mean).astype("float32")
        test_df[f"ext_rate__{c}"] = test_df[c].map(lookup).fillna(global_mean).astype("float32")
    return train_df, test_df
```

---

## Part II: Optional Synthetic-Generator / Playground Recipes

> [!NOTE]
> **Forensics Gate Required**: Apply the recipes below **only** when EDA confirms synthetic-generator artifacts (e.g., an original seed dataset, discrete perturbation noise, decimal rounding spikes, or LLM/BPE tokenization). On smooth continuous real-world or diffusion-generated tables, these features can add pure noise—always ablate on CV.

### 6. Linear Formula Discovery
Check whether an underlying additive linear formula governs the synthetic target:

```python
from sklearn.linear_model import Ridge


def discover_linear_formula(df_train, num_cols, target_col, alpha=1.0):
    """Fit Ridge on numeric features to inspect dominant additive linear coefficients."""
    X_num = df_train[num_cols].fillna(df_train[num_cols].median())
    model = Ridge(alpha=alpha).fit(X_num, df_train[target_col])
    return pd.Series(model.coef_, index=num_cols), float(model.intercept_)
```

### 7. Modulo Digit Decomposition & Benford Deviation

```python
def extract_digits(df, col, power_range=(0, 2)):
    """Extract base-10 digits (x * 10^k) % 10 when decimal rounding artifacts exist."""
    for k in range(power_range[0], power_range[1] + 1):
        df[f"{col}_d{k}"] = (np.floor(df[col].abs() * (10**k)) % 10).fillna(-1).astype("int8")
    return df


def extract_benford_deviation(df, num_col):
    """Extract leading non-zero digit and theoretical Benford probability."""
    benford_probs = {d: np.log10(1.0 + 1.0 / d) for d in range(1, 10)}
    str_vals = df[num_col].abs().astype(str).str.replace(r"^0+\.?", "", regex=True)
    lead = pd.to_numeric(str_vals.str[0], errors="coerce").fillna(1).clip(1, 9).astype("int8")
    df[f"{num_col}_lead_digit"] = lead
    df[f"{num_col}_benford_expected"] = lead.map(benford_probs).astype("float32")
    return df
```

### 8. Synthetic Snap Matching, Radix Encoding & cKDTree Seed Prior

```python
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler


def compute_snap_features(train_df, test_df, orig_df, cols):
    """Map perturbed synthetic floats to nearest original seed values and record delta."""
    for col in cols:
        orig_vals = np.sort(orig_df[col].dropna().unique())
        for df in (train_df, test_df):
            vals = df[col].to_numpy()
            idx = np.clip(np.searchsorted(orig_vals, vals), 0, len(orig_vals) - 1)
            left_idx = np.clip(idx - 1, 0, len(orig_vals) - 1)
            nearest = np.where(
                np.abs(vals - orig_vals[left_idx]) < np.abs(vals - orig_vals[idx]),
                orig_vals[left_idx],
                orig_vals[idx],
            )
            df[f"{col}_snap"] = nearest
            df[f"{col}_snap_diff"] = vals - nearest
    return train_df, test_df


def create_radix_feature(df, continuous_col, cat_col, num_multiplier=100, cat_offset=100000):
    """Pack a quantized continuous column and categorical code into a single integer key."""
    cat_codes = df[cat_col].astype("category").cat.codes.astype("int64")
    return (df[continuous_col] * num_multiplier).round().astype("int64") + cat_codes * cat_offset


def attach_orig_kdtree_prior(train_df, test_df, orig_df, match_cols, target_col):
    """Attach nearest original-seed neighbor target and Euclidean distance (zero CV leak)."""
    scaler = StandardScaler()
    orig_scaled = scaler.fit_transform(orig_df[match_cols].fillna(0))
    tree = cKDTree(orig_scaled)
    orig_targets = orig_df[target_col].to_numpy()

    for df in (train_df, test_df):
        dists, idxs = tree.query(scaler.transform(df[match_cols].fillna(0)), k=1)
        df[f"{target_col}_orig_nn_prior"] = orig_targets[idxs]
        df[f"{target_col}_orig_nn_dist"] = dists.astype("float32")
    return train_df, test_df
```

### 9. BPE Subword Token Encodings on Numeric Strings (LLM / GReaT Synthesizers)

```python
import tiktoken


def add_bpe_token_encodings(train_df, test_df, num_col, target_col, target_type="binary", cv=5):
    """Tokenize numeric strings with GPT-2 BPE and out-of-fold target-encode token boundaries."""
    enc = tiktoken.get_encoding("gpt2")
    for df in (train_df, test_df):
        tokens = [enc.encode(" " + str(int(v))) if pd.notna(v) else [] for v in df[num_col]]
        df[f"{num_col}_tok_len"] = [len(t) for t in tokens]
        df[f"{num_col}_first_tok"] = [t[0] if t else -1 for t in tokens]
        df[f"{num_col}_last_tok"] = [t[-1] if t else -1 for t in tokens]

    tok_keys = [f"{num_col}_tok_len", f"{num_col}_first_tok", f"{num_col}_last_tok"]
    te = TargetEncoder(target_type=target_type, smooth="auto", cv=cv, shuffle=True, random_state=42)
    te_tr = te.fit_transform(train_df[tok_keys].astype(str), train_df[target_col])
    te_te = te.transform(test_df[tok_keys].astype(str))

    for i, k in enumerate(tok_keys):
        train_df[f"{k}_te"] = te_tr[:, i].astype("float32")
        test_df[f"{k}_te"] = te_te[:, i].astype("float32")
    return train_df, test_df
```
