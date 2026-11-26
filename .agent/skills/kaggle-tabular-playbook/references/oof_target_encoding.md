# Out-Of-Fold Multi-Aggregation Target Encoding

Standard target encoding computes raw category means, leaking target labels if applied in-sample and overfitting rare categories if unsmoothed. This recipe provides a leak-free out-of-fold Target Encoder with higher-order moments (`mean`, `std`, `skew`) and **consistent Empirical Bayes smoothing across both `fit_transform()` (train OOF) and `transform()` (test)**.

---

## 1. Empirical Bayes Smoothing Formula

For category $c$ with count $n$ and sample mean $\bar{y}_c$, the smoothed target mean is:
$$\hat{y}_{\text{smooth}} = \frac{n \cdot \bar{y}_c + m \cdot \mu_{\text{global}}}{n + m}, \qquad m = \frac{\sigma^2_{\text{within}}}{\sigma^2_{\text{between}}}$$
where $\sigma^2_{\text{within}}$ is the average within-category target variance and $\sigma^2_{\text{between}}$ is the variance across category means.

---

## 2. Complete Python Implementation

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


class MultiAggTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Leak-free out-of-fold target encoder supporting multiple aggregations
    (e.g., mean, std, skew) with identical Empirical Bayes smoothing on both
    training OOF (fit_transform) and inference (transform).
    """

    def __init__(self, cols_to_encode, aggs=("mean", "std", "skew"), cv=5, smooth="auto", drop_original=False):
        self.cols_to_encode = list(cols_to_encode)
        self.aggs = tuple(aggs)
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def _compute_mapping(self, df_with_target: pd.DataFrame, col: str, agg_func: str, global_val: float):
        grouped = df_with_target.groupby(col, observed=False)["target"]
        mapping = grouped.agg(agg_func)
        if agg_func != "mean":
            return mapping

        counts = grouped.count()
        if self.smooth == "auto":
            var_between = float(mapping.var())
            var_within = float(grouped.var().mean())
            m = (var_within / var_between) if (np.isfinite(var_between) and var_between > 0 and np.isfinite(var_within)) else 0.0
        else:
            m = float(self.smooth)
        return (counts * mapping + m * global_val) / (counts + m)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        temp_df = X[self.cols_to_encode].copy()
        temp_df["target"] = np.asarray(y)
        y_series = temp_df["target"]

        self.global_stats_ = {agg: float(y_series.agg(agg)) for agg in self.aggs}
        self.mappings_ = {}
        for col in self.cols_to_encode:
            self.mappings_[col] = {
                agg: self._compute_mapping(temp_df, col, agg, self.global_stats_[agg])
                for agg in self.aggs
            }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.cols_to_encode:
            for agg in self.aggs:
                new_col = f"TE_{col}_{agg}"
                X_out[new_col] = X[col].map(self.mappings_[col][agg]).fillna(self.global_stats_[agg]).astype("float32")

        if self.drop_original:
            X_out = X_out.drop(columns=self.cols_to_encode)
        return X_out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        encoded = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        y_arr = np.asarray(y)

        for train_idx, val_idx in kf.split(X, y_arr):
            temp_tr = X.iloc[train_idx][self.cols_to_encode].copy()
            temp_tr["target"] = y_arr[train_idx]
            X_val = X.iloc[val_idx]

            for col in self.cols_to_encode:
                for agg in self.aggs:
                    new_col = f"TE_{col}_{agg}"
                    fold_global = float(temp_tr["target"].agg(agg))
                    mapping = self._compute_mapping(temp_tr, col, agg, fold_global)
                    encoded.loc[X_val.index, new_col] = X_val[col].map(mapping).fillna(fold_global).astype("float32")

        X_out = X.copy()
        for col in encoded.columns:
            X_out[col] = encoded[col]
        if self.drop_original:
            X_out = X_out.drop(columns=self.cols_to_encode)
        return X_out
```
