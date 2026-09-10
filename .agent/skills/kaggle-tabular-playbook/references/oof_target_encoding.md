# Out-Of-Fold Multi-Aggregation Target Encoding

Standard target encoding calculates category-level means and can lead to severe leakage if not strictly evaluated out-of-fold. This reference provides an industrial-grade, leak-free Target Encoder with higher-order moments (`mean`, `std`, `skew`) and Empirical Bayes smoothing.

---

## 1. Empirical Bayes Smoothing Formula

For categorical group $c$ with sample count $n$ and sample target mean $\bar{y}_c$, the smoothed target mean is:

$$\hat{y}_{\text{smooth}} = \frac{n \cdot \bar{y}_c + m \cdot \mu_{\text{global}}}{n + m}$$

Where the smoothing parameter $m$ is automatically determined via Empirical Bayes shrinkage:

$$m = \frac{\sigma^2_{\text{within}}}{\sigma^2_{\text{between}}}$$

- $\sigma^2_{\text{within}}$ is the average variance of the target within category groups.
- $\sigma^2_{\text{between}}$ is the variance between category group means.

---

## 2. Complete Python Implementation

```python
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

class TargetEncoder_(BaseEstimator, TransformerMixin):
    """
    Leak-free out-of-fold target encoder supporting multiple aggregation functions
    (mean, std, skew, median) with Empirical Bayes smoothing.
    """
    def __init__(self, cols_to_encode, aggs=['mean', 'std', 'skew'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit(self, X, y):
        temp_df = X.copy()
        temp_df['target'] = y

        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)

        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                mapping = temp_df.groupby(col)['target'].agg(agg_func)
                self.mappings_[col][agg_func] = mapping

        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col = f'TE_{col}_{agg_func}'
                mapping = self.mappings_[col][agg_func]
                X_transformed[new_col] = X[col].map(mapping).fillna(self.global_stats_[agg_func])

        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
        return X_transformed

    def fit_transform(self, X, y):
        self.fit(X, y)
        encoded_features = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train

            for col in self.cols_to_encode:
                for agg_func in self.aggs:
                    new_col = f'TE_{col}_{agg_func}'
                    fold_global = y_train.agg(agg_func)
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)

                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()
                        if self.smooth == 'auto':
                            var_between = mapping.var()
                            var_within = temp_df_train.groupby(col)['target'].var().mean()
                            m = (var_within / var_between) if var_between > 0 else 0
                        else:
                            m = self.smooth
                        smoothed_mapping = (counts * mapping + m * fold_global) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)

                    encoded_features.loc[X_val.index, new_col] = encoded_values.fillna(fold_global)

        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]

        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
        return X_transformed
```
