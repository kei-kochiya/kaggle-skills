# Riemannian Hypersphere ($\mathbb{S}^{N-1}$) Fréchet Barycenter Ensembling & Boundary Rules

> **Target Metric:** Rank-based classification metrics (ROC-AUC, Gini, PR-AUC).  
> **Core Concept:** Fusing high-dimensional model prediction vectors along the intrinsic geodesic curve of the unit hypersphere $\mathbb{S}^{N-1}$ to prevent Euclidean norm collapse, combined with strictly verified deterministic boundary guards and zero-tie lexicographical sorting (`lexrank`).

---

## 1. Mathematical Motivation

### 1.1 The Euclidean Shrinkage Problem in Ensembling
When ensembling $K$ diverse prediction vectors $\mathbf{p}_1, \dots, \mathbf{p}_K$ mapped to probit-transformed ranks $z_i = \Phi^{-1}(r_i)$, standard Euclidean convex combinations cut through the interior of the hypersphere rather than staying on its surface:
$$\left\| \sum_{k=1}^K w_k \mathbf{z}_k \right\|_2 < \sum_{k=1}^K w_k \|\mathbf{z}_k\|_2$$
This causes **norm shrinkage**: blended scores compress toward the center ($z \approx 0$), muting extreme tail confidence where ROC-AUC concordance is most decisive.

### 1.2 The Weighted Fréchet Barycenter on $\mathbb{S}^{N-1}$
Normalize each probit-rank vector onto the unit hypersphere $\mathbf{u}_k = \mathbf{z}_k / \|\mathbf{z}_k\|_2 \in \mathbb{S}^{N-1}$, where geodesic distance is $d_{\mathbb{S}}(\mathbf{u}_a, \mathbf{u}_b) = \arccos(\mathbf{u}_a^T \mathbf{u}_b)$. The weighted **Fréchet barycenter** minimizes $\sum_{k=1}^K w_k\, d_{\mathbb{S}}(\mathbf{u}, \mathbf{u}_k)^2$ via fast iterative tangent-space updates (typically 3–5 iterations):
1. **Log Map**: $\mathbf{v}_k = \frac{\theta_k}{\sin\theta_k}(\mathbf{u}_k - \cos\theta_k\,\mathbf{u})$ with $\theta_k = \arccos(\mathbf{u}^T \mathbf{u}_k)$.
2. **Weighted Tangent Mean**: $\bar{\mathbf{v}} = \sum_{k=1}^K w_k \mathbf{v}_k$.
3. **Exp Map**: $\mathbf{u}_{\text{next}} = \cos(\|\bar{\mathbf{v}}\|_2)\mathbf{u} + \sin(\|\bar{\mathbf{v}}\|_2)\frac{\bar{\mathbf{v}}}{\|\bar{\mathbf{v}}\|_2}$.

---

## 2. Guarded Deterministic Boundary Invariant Discovery

Some datasets (particularly rule-generated or synthetic benchmarks) contain **strict boundary regions** where the true target rate is empirically $0.0$ or $1.0$ with zero exceptions.

### Verification Protocol Before Applying Any Hard Override
Never hardcode domain guesses without statistical proof. Audit candidate rules strictly inside CV:
1. Fit shallow, high-`min_samples_leaf` decision trees (depth 1–3) on each training fold to surface candidate pure leaves ($p = 0.0$ or $p = 1.0$).
2. Require **100% purity across every CV validation fold** (zero exceptions on held-out folds) AND a conservative minimum support count (e.g., $n \ge 50$–$200$, Wilson lower bound $> 0.99$).
3. Verify positive out-of-fold metric lift when pushing confirmed positive masks to $+\infty$ and confirmed negative masks to $-\infty$ before ranking. *(Skip overrides entirely if any validation exception exists—one false deterministic flip penalizes AUC heavily!)*

---

## 3. Zero-Tie Lexicographical Ranking (`lexrank`)

Under ROC-AUC, tied prediction values receive only $0.5$ concordance credit. Tree leaf saturation, probability clipping, or rounding can introduce thousands of silent ties. Break all potential ties deterministically using an independent, continuous secondary predictor (e.g., a smooth tabular neural network such as RealMLP or TabM):
$$\text{order} = \text{np.lexsort}((\text{secondary\_continuous}, \text{primary\_ensemble}))$$
$$\text{rank}[\text{order}] = 1, 2, \dots, N \implies p_{\text{unique}} = \frac{\text{rank} - 0.5}{N}$$
This guarantees strictly $N$ unique values in $(0, 1)$ while preserving the exact primary hierarchy.

---

## 4. Production Python Implementation

```python
import numpy as np
from scipy.stats import norm, rankdata


def rank01(x: np.ndarray) -> np.ndarray:
    """Map continuous array to uniform open interval (0, 1)."""
    n = len(x)
    return (rankdata(x) - 0.5) / n


def probit(x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Probit (inverse standard normal CDF) transform."""
    return norm.ppf(np.clip(x, eps, 1.0 - eps))


def frechet_barycenter_hypersphere(
    predictions_list: list[np.ndarray],
    weights: list[float] | None = None,
    max_iter: int = 15,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Compute the weighted Frechet barycenter (Karcher mean) of model prediction
    vectors on the Riemannian unit hypersphere S^{N-1}.
    """
    K = len(predictions_list)
    N = len(predictions_list[0])

    if weights is None:
        w = np.full(K, 1.0 / K, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()

    U = np.empty((K, N), dtype=np.float64)
    for k in range(K):
        z = probit(rank01(predictions_list[k]))
        norm_z = np.linalg.norm(z)
        if norm_z == 0.0:
            raise ValueError(f"Model {k} has zero variance in probit space.")
        U[k] = z / norm_z

    u = np.tensordot(w, U, axes=(0, 0))
    u /= np.linalg.norm(u)

    for _ in range(max_iter):
        V_bar = np.zeros(N, dtype=np.float64)
        for k in range(K):
            cos_theta = float(np.clip(np.dot(u, U[k]), -1.0, 1.0))
            theta = np.arccos(cos_theta)
            if abs(theta) < 1e-12:
                continue
            v_k = (theta / np.sin(theta)) * (U[k] - cos_theta * u)
            V_bar += w[k] * v_k

        norm_v = np.linalg.norm(V_bar)
        if norm_v < tol:
            break
        u = np.cos(norm_v) * u + np.sin(norm_v) * (V_bar / norm_v)
        u /= np.linalg.norm(u)

    return u


def apply_verified_boundary_masks(
    preds: np.ndarray,
    pos_mask: np.ndarray | None = None,
    neg_mask: np.ndarray | None = None,
    extreme_val: float = 1e9,
) -> np.ndarray:
    """Push CV-verified 100%-pure deterministic boundary rows to +/- extreme ranks."""
    out = np.asarray(preds, dtype=np.float64).copy()
    if pos_mask is not None:
        out[np.asarray(pos_mask, dtype=bool)] = extreme_val
    if neg_mask is not None:
        out[np.asarray(neg_mask, dtype=bool)] = -extreme_val
    return out


def lexrank(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """
    Zero-tie lexicographical rank transform.
    Primary predictions govern ordering; continuous secondary predictions break ties.
    """
    N = len(primary)
    order = np.lexsort((secondary, primary))
    ranks = np.empty(N, dtype=np.float64)
    ranks[order] = np.arange(1, N + 1, dtype=np.float64)
    return (ranks - 0.5) / N
```

---

## 5. End-to-End Pipeline Pattern

```python
# Tune weights strictly on OOF CV before applying to test predictions
barycenter_test = frechet_barycenter_hypersphere(
    predictions_list=[test_stack_a, test_stack_b, test_stack_c],
    weights=[0.50, 0.30, 0.20],
)

# Optional: apply ONLY rules verified with 0 errors across all CV folds
# barycenter_test = apply_verified_boundary_masks(barycenter_test, pos_mask=pos_m, neg_mask=neg_m)

# Break any remaining ties using an independent smooth continuous model (e.g., RealMLP)
final_preds = lexrank(primary=barycenter_test, secondary=test_continuous_nn)

assert len(np.unique(final_preds)) == len(final_preds), "Ties detected!"
assert (final_preds > 0.0).all() and (final_preds < 1.0).all(), "Predictions out of (0, 1)!"
```
