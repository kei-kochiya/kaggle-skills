# AUC-Direct Level-3 Ensembling via FFT Convolution & Foundation Tabular Transformers

> **Origin:** Team Alicia (2nd Place Solution, Kaggle Playground Series S6E9)  
> **Target Metric:** Area Under the ROC Curve (ROC-AUC)  
> **Core Concept:** Directly optimizing a smooth pairwise ranking surrogate across all $n_+ \times n_-$ pairs (over 26 billion pairs) in $O(n + N \log N)$ time using histogram binning and Fast Fourier Transform (FFT) convolution, paired with full-context TabPFN foundation models and generator-surrogate log-likelihood ratio (LLR) features.

---

## 1. Mathematical Formulation

### 1.1 The Pairwise ROC-AUC Surrogate Loss
Standard ensembling methods for binary classification minimize surrogate log-loss (Cross-Entropy) or squared error ($L_2$ Ridge regression on probabilities or logits). However, in high-stakes ROC-AUC competitions, optimizing cross-entropy is suboptimal: log-loss penalizes probability calibration errors across all samples, whereas ROC-AUC strictly evaluates the pairwise ordering concordance between positive and negative classes.

The exact Wilcoxon-Mann-Whitney ranking loss is:
$$L_{\text{WMW}}(w) = \frac{1}{n_+ n_-} \sum_{i \in \mathcal{Y}_+} \sum_{j \in \mathcal{Y}_-} \mathbb{I}(s_i \le s_j), \quad \text{where } s = Xw$$

Because the indicator function $\mathbb{I}$ is non-differentiable with zero gradients everywhere, we replace it with a smooth sigmoid temperature surrogate:
$$\min_{w \ge 0,\ \sum_{k=1}^K w_k = 1} L_{\text{AUC}}(w) = \frac{1}{n_+ n_-} \sum_{i \in \mathcal{Y}_+} \sum_{j \in \mathcal{Y}_-} \sigma\left( - \frac{s_i - s_j}{\tau} \right)$$
where:
- $X \in \mathbb{R}^{n \times K}$ holds the out-of-fold rank-logits of $K$ candidate models: $x_{ik} = \text{logit}\left(\frac{\text{rank}_{ik} - 0.5}{n}\right)$.
- $w \in \mathbb{R}^K$ is the ensemble weight vector constrained to the probability simplex ($\sum_k w_k = 1, w_k \ge 0$).
- $\tau$ is the temperature parameter (empirically optimal at $\tau = 0.1$ in rank-logit space; as $\tau \to 0$, $1 - L_{\text{AUC}} \to \text{ROC-AUC}$).
- $s = Xw$ is the ensemble ensemble score. Shifts cancel out in pairwise differences $(s_i - s_j)$, so no intercept is required.

---

### 1.2 The $O(n + N \log N)$ FFT Convolution Breakthrough
In large tabular datasets (e.g. $n = 534{,}932$ with $17.46\%$ positives), the number of positive-negative pairs is:
$$n_+ \times n_- = 93{,}422 \times 441{,}510 \approx 4.12 \times 10^{10} \text{ pairs}$$

Evaluating all pairs directly or via Monte Carlo subsampling is either computationally intractable or noisy (sampling variance masks micro-gains of $+0.00002$ AUC).

**The Solution:**
1. Discretize the ensemble score domain $[-16, 16]$ into a high-resolution linear grid of $N = 64{,}001$ bins with step $H = 0.0005$:
   $$g_m = -16 + m \cdot H, \quad m \in \{0, 1, \dots, N-1\}$$
2. Construct normalized linear-interpolation histograms for positive predictions $\mathbf{h}_+$ and negative predictions $\mathbf{h}_-$:
   $$\mathbf{h}_+(m) = \frac{1}{n_+} \sum_{i \in \mathcal{Y}_+} \Lambda(s_i, g_m), \quad \mathbf{h}_-(m) = \frac{1}{n_-} \sum_{j \in \mathcal{Y}_-} \Lambda(s_j, g_m)$$
3. The pairwise loss corresponds exactly to a cross-correlation between the two histograms with the sigmoid kernel:
   $$K(d) = \sigma\left( - \frac{d \cdot H}{\tau} \right), \quad d \in \{-(N-1), \dots, N-1\}$$
4. By the Convolution Theorem, cross-correlation is evaluated in frequency space in $O(N \log N)$ time:
   $$\mathbf{F} = \text{fftconvolve}(\mathbf{h}_-, K)[N-1 : 2N-1]$$
   $$L_{\text{AUC}}(w) = \mathbf{h}_+^T \mathbf{F}$$

This computes the **exact, all-pairs continuous ranking loss** across billions of pairs in less than 20 milliseconds per optimization step!

---

## 2. Production Python Implementation

```python
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import expit, logit
from scipy.optimize import minimize
from scipy.stats import rankdata

class AUCDirectFFTBlender:
    """
    Direct pairwise ROC-AUC optimizer using histogram linear interpolation
    and 1D FFT convolution. Evaluates all n_+ * n_- pairs in O(n + N log N).
    """
    def __init__(self, tau: float = 0.1, grid_step: float = 0.0005, grid_bound: float = 16.0):
        self.tau = tau
        self.H = grid_step
        self.LO = -grid_bound
        self.HI = grid_bound
        self.N = int(np.round((self.HI - self.LO) / self.H)) + 1
        
        # Precompute the sigmoid kernel across all possible grid lags: sigma(-lag / tau)
        lags = np.arange(-self.N + 1, self.N) * self.H
        self.kernel = expit(-lags / self.tau)
        self.weights_ = None

    def _histogram(self, s: np.ndarray) -> np.ndarray:
        """Constructs linear-interpolation histogram normalized to sum to 1."""
        q = np.clip((s - self.LO) / self.H, 0.0, self.N - 1 - 1e-9)
        lo = np.floor(q).astype(int)
        fr = q - lo
        h = np.bincount(lo, weights=1.0 - fr, minlength=self.N) + \
            np.bincount(lo + 1, weights=fr, minlength=self.N)
        return h[:self.N] / len(s)

    def loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Computes exact all-pairs sigmoid ranking loss."""
        s = X @ w
        h_pos = self._histogram(s[y == 1])
        h_neg = self._histogram(s[y == 0])
        
        # FFT convolution of negative histogram with sigmoid kernel
        F = fftconvolve(h_neg, self.kernel)[self.N - 1 : 2 * self.N - 1]
        return float(np.dot(h_pos, F))

    def fit(self, X: np.ndarray, y: np.ndarray, w0: np.ndarray | None = None) -> "AUCDirectFFTBlender":
        """
        Fits optimal non-negative ensemble weights summing to 1 via SLSQP.
        X should be out-of-fold rank-logits: logit((rank - 0.5) / n).
        """
        K = X.shape[1]
        if w0 is None:
            w0 = np.ones(K, dtype=np.float64) / K
        else:
            w0 = np.array(w0, dtype=np.float64)
            w0 = np.clip(w0, 1e-6, 1.0)
            w0 /= w0.sum()

        bounds = [(0.0, 1.0)] * K
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(
            fun=self.loss,
            x0=w0,
            args=(X, y),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 300}
        )
        self.weights_ = res.x
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Computes ensemble rank predictions on test data."""
        assert self.weights_ is not None, "Model must be fitted before predict."
        scores = X_test @ self.weights_
        return (rankdata(scores) - 0.5) / len(scores)
```

---

## 3. TabPFN-3.5: Scaling Context to Hundreds of Thousands of Rows

In tabular foundation models, **TabPFN-3.5** (`tabpfn==9.0.0`, checkpoint `tabpfn-v3.5-20260909.safetensors`) demonstrated unprecedented zero-shot capability on massive tabular datasets:

### 3.1 The Context Scaling Law
TabPFN was historically applied to small datasets ($<10{,}000$ rows). However, with modern caching and inference modes, TabPFN can ingest the **entire competition training set as in-context memory**:
- `fit_mode='fit_with_cache'` with automatic KV-cache precision.
- `ignore_pretraining_limits=True`.
- **The Empirical Scaling Law:** Out-of-fold AUC scales log-linearly with context length:
  $$\text{AUC}(N_{\text{context}}) \approx \text{AUC}_0 + \beta \log_2(N_{\text{context}}), \quad \beta \approx +18.6 \times 10^{-5} \text{ per doubling}$$
- Testing on full $668{,}665$ context rows produced the single strongest standalone model of the entire competition ($0.946485$ pooled AUC).

### 3.2 Critical Data Leakage Warning for Tabular Transformers
> [!CAUTION]
> **Never feed cross-validated model OOF predictions to a neighbour-reading learner like TabPFN or KNN!**  
> If TabPFN receives candidate model OOFs as feature columns alongside raw covariates, its cross-attention mechanism attends to context rows whose OOF predictions were generated by models trained with the query fold's ground-truth labels. This creates an artificial **$+0.00162$ AUC leakage illusion** that completely collapses on unseen test data ($-0.00054$ true loss).

---

## 4. Generator-Surrogate Log-Likelihood Ratio (LLR) Features

When competing on synthetic tabular datasets generated from a small real-world seed dataset (e.g. 10,000 rows):

1. **The Language Model Surrogate**:
   If the competition dataset was generated by an LLM (e.g. CTGAN or GReaT) trained on an original dataset $\mathcal{D}_{\text{orig}}$, train a surrogate language model (e.g. `distilgpt2`) **strictly on $\mathcal{D}_{\text{orig}}$ with zero access to competition labels**.
2. **Text Representation**:
   Format each row as a serialized sentence with randomized column order:
   `"Annual_Income_USD is 75000, Daily_Commute_km is 25, Will_Buy_EV is Yes"`
3. **Log-Likelihood Ratio (LLR)**:
   Evaluate the log-likelihood of each competition row conditioned on positive vs. negative target:
   $$\text{LLR} = \log p(\mathbf{x} \mid y = \text{"Yes"}) - \log p(\mathbf{x} \mid y = \text{"No"})$$
   Average this score across $K = 4$ fixed column permutations to cancel token ordering variance.
4. **Usage in Pipeline**:
   Do not treat LLR merely as an isolated feature for decision trees. Use LLR as an informative prior input into continuous neural models (TabPFN, RealMLP) and linear margin initializers (`init_score`).
