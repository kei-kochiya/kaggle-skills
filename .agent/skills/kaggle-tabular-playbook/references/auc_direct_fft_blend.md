# AUC-Direct Ensembling via FFT Convolution & Full-Context TFM Guardrails

> **Target Metric:** Area Under the ROC Curve (ROC-AUC)  
> **Core Concept:** Directly optimizing the smooth all-pairs Wilcoxon–Mann–Whitney (WMW) ranking surrogate across all $n_+ \times n_-$ positive–negative pairs in $O(n + N \log N)$ time using linear-interpolation histograms and 1D FFT convolution.

---

## 1. Mathematical Formulation

### 1.1 The Pairwise ROC-AUC Surrogate Loss
Log-loss and squared-error meta-learners penalize global calibration deviations, whereas ROC-AUC evaluates strictly the pairwise ordering concordance between positive and negative rows. Replacing the non-differentiable indicator $\mathbb{I}(s_i \le s_j)$ with a smooth sigmoid surrogate gives:
$$\min_{w \ge 0,\ \sum_{k=1}^K w_k = 1} L_{\text{AUC}}(w) = \frac{1}{n_+ n_-} \sum_{i \in \mathcal{Y}_+} \sum_{j \in \mathcal{Y}_-} \sigma\!\left(-\frac{s_i - s_j}{\tau}\right), \qquad s = Xw$$
where $X \in \mathbb{R}^{n \times K}$ contains out-of-fold rank-logits $x_{ik} = \text{logit}\!\left(\frac{\text{rank}_{ik} - 0.5}{n}\right)$ and temperature $\tau \approx 0.1$ balances gradient smoothness with tight AUC alignment.

### 1.2 Exact $O(n + N \log N)$ Evaluation via FFT Convolution
On datasets with $n \sim 10^5$–$10^6$ rows, $n_+ \times n_-$ reaches $10^{10}$–$10^{11}$ pairs—intractable explicitly and noisy under Monte Carlo subsampling. Discretizing $s$ onto a fine uniform grid of $N$ bins (e.g., $[-16, 16]$ with step $H = 0.0005$, $N = 64{,}001$) converts the all-pairs sum into a 1D cross-correlation between linear-interpolation histograms $\mathbf{h}_+$ and $\mathbf{h}_-$:
$$K(d) = \sigma\!\left(-\frac{d \cdot H}{\tau}\right), \qquad \mathbf{F} = \text{fftconvolve}(\mathbf{h}_-, K)[N - 1 : 2N - 1], \qquad L_{\text{AUC}}(w) = \mathbf{h}_+^T \mathbf{F}$$
Evaluating tens of billions of pairs takes under 20 ms per optimization step.

---

## 2. Production Python Implementation (Use Inside Nested CV!)

```python
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.special import expit
from scipy.stats import rankdata


class AUCDirectFFTBlender:
    """
    Direct pairwise ROC-AUC optimizer using linear-interpolation histograms
    and 1D FFT convolution. Evaluates all n_+ * n_- pairs in O(n + N log N).
    Fit strictly inside nested CV folds to verify honest OOF AUC gain.
    """

    def __init__(self, tau: float = 0.1, grid_step: float = 0.0005, grid_bound: float = 16.0):
        self.tau = tau
        self.H = grid_step
        self.LO = -grid_bound
        self.HI = grid_bound
        self.N = int(np.round((self.HI - self.LO) / self.H)) + 1

        lags = np.arange(-self.N + 1, self.N) * self.H
        self.kernel = expit(-lags / self.tau)
        self.weights_ = None

    def _histogram(self, s: np.ndarray) -> np.ndarray:
        q = np.clip((s - self.LO) / self.H, 0.0, self.N - 1 - 1e-9)
        lo = np.floor(q).astype(int)
        fr = q - lo
        h = np.bincount(lo, weights=1.0 - fr, minlength=self.N) + np.bincount(
            lo + 1, weights=fr, minlength=self.N
        )
        return h[: self.N] / len(s)

    def loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        s = X @ w
        h_pos = self._histogram(s[y == 1])
        h_neg = self._histogram(s[y == 0])
        F = fftconvolve(h_neg, self.kernel)[self.N - 1 : 2 * self.N - 1]
        return float(np.dot(h_pos, F))

    def fit(self, X: np.ndarray, y: np.ndarray, w0: np.ndarray | None = None) -> "AUCDirectFFTBlender":
        """Fit non-negative simplex weights on OOF rank-logits via SLSQP."""
        K = X.shape[1]
        if w0 is None:
            w0 = np.full(K, 1.0 / K, dtype=np.float64)
        else:
            w0 = np.clip(np.asarray(w0, dtype=np.float64), 1e-6, 1.0)
            w0 /= w0.sum()

        res = minimize(
            fun=self.loss,
            x0=w0,
            args=(X, y),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * K,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            options={"ftol": 1e-9, "maxiter": 300},
        )
        self.weights_ = res.x
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        assert self.weights_ is not None, "Call fit() before predict()."
        scores = X_test @ self.weights_
        return (rankdata(scores) - 0.5) / len(scores)
```

---

## 3. Full-Context Tabular Foundation Model (TFM) Guardrails

Modern in-context tabular foundation models (e.g., `TabPFN`, `TabICL`, `Kumo`, `TabFM`) scale far beyond small-data regimes via KV-cached full-context inference (`fit_mode="fit_with_cache"`, `ignore_pretraining_limits=True`), often improving log-linearly as context size doubles.

> [!CAUTION]
> **The Neighbor-Reading Attention Leak (TabPFN / KNN Trap)**:  
> **Never pass cross-validated base-model OOF predictions as input features to an in-context learner (TabPFN/TabICL) or KNN meta-learner!** Because cross-attention reads context rows whose OOF predictions were produced by models trained on the query fold's labels, it creates a massive illusory CV spike that collapses on unseen test data. Pass only raw/leak-free covariates (or strictly nested inner-OOF features) into in-context models.

For full recipes on **Full-Context TFMs**, **Target-First / Any-Order Generative Transformers (Bayes LLR)**, and **Self-Supervised Feature-Token Transformers (RTD)**, see [`tabular_llms_generative_transformers.md`](./tabular_llms_generative_transformers.md).
