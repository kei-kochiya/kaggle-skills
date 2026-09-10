# Kaggle Playground Series S6E9: Predicting Electric Vehicle Adoption (Will_Buy_EV)

> **Competition**: Kaggle Playground Series Season 6 Episode 9  
> **Evaluation Metric**: ROC-AUC  
> **Dataset**: 668,665 Train Rows, 286,571 Test Rows (Synthetic from 10,000-row seed dataset)  
> **Target**: Binary `Will_Buy_EV` ("Yes" = 1, "No" = 0, Positive Base Rate: 17.4645%)  
> **Key Progression**: 0.94162 (5-Model Zoo) $\to$ 0.94639 (Top-50 Blend) $\to$ 0.94643 (Logit Consensus) $\to$ **0.94644** (Neural Manifold Decorrelation)

---

## 1. Executive Summary & Core Breakthroughs

Playground Series S6E9 challenged competitors to predict whether consumers will purchase an electric vehicle (`Will_Buy_EV`). While standard GBDT ensembles quickly plateaued around `0.94638 – 0.94643`, breaking into the top tier (`0.94644+`) required uncovering a fundamental ensembling dynamic:

1. **The Spearman Rank Correlation Screening Rule ($\rho \le 0.998$)**:
   Testing random public models on cross-validation or live submissions is inefficient. AUC depends strictly on rank ordering. Candidate models must exhibit a Spearman rank correlation $\rho \le 0.998$ against the current blend to provide meaningful structural signal.
2. **The Inductive Bias Pairing Breakthrough (Neural Manifolds vs. Tree Step-Cuts)**:
   A collection of 11 GBDTs (LightGBM, CatBoost, XGBoost) saturates because all decision trees produce piecewise-constant, axis-aligned step functions. A deep tabular neural network (**PyTorch RealMLP**), despite having a substantially lower standalone score ($0.94595$ vs $0.94643$), provides a decorrelated continuous error surface ($\rho \approx 0.9964$). Adding $15\%$ RealMLP to the top GBDT blend pushed the public leaderboard to **`0.94644`**.
3. **The "Candidate Pooling Fallacy"**:
   Averaging 5 diverse models (each $\rho \approx 0.9965$ with the blend) before ensembling causes their mutual disagreements to cancel out, driving the pool's correlation against the master blend to $> 0.9990$. Diverse models must be evaluated and blended individually.
4. **The Plateau Center-Selection Rule**:
   Sweeping RealMLP weights revealed a flat performance plateau from $10\%$ to $20\%$ ($0.94644$). Choosing the center of the plateau ($15\%$) provides maximum margin of safety against private test set distribution shifts.

---

## 2. Dataset Architecture & Exploratory Forensics

### The Seed Origin
- The synthetic dataset was generated from `EV_Adoption_and_Range_Anxiety_Dataset.csv` (10,000 rows, published by `itzzomkar`).
- **Target Distribution**: Exactly $17.4645\%$ positive in train. The synthetic generator strictly preserved this base rate.

### Forensic Discoveries
1. **Purchase Price Threshold Spikes**:
   Vehicle price features clustered around rounded thresholds ($\$30{,}000$, $\$50{,}000$). Residuals between consumer income and EV purchase price showed steep inflection points where adoption likelihood inverted.
2. **Range Anxiety & Daily Commute Interaction**:
   The ratio $\text{Commute\_Distance} / \text{Battery\_Range}$ acted as a step-function trigger. When daily commute exceeded $40\%$ of battery capacity, purchase probability collapsed non-linearly.
3. **Mantissa Discretization**:
   Synthetic generator floats exhibited distinct fractional mantissa patterns ($x - \lfloor x \rfloor$) that allowed tree models to distinguish synthetic interpolations from exact seed copies.

---

## 3. Modeling Zoo & Inductive Biases

### Component Model Profiles
| Architecture | Model Family | Inductive Bias / Specialty | Standalone Score | $\rho$ vs. Blend |
| :--- | :--- | :--- | :---: | :---: |
| **Pure LGBM V3** (`najiama`) | LightGBM (Leaf-wise) | Mantissa extraction, seed priors, quantile bins | 0.94637 | 0.99935 |
| **Stacked Multi-GBDT** (`kospintr`) | LGBM + CatBoost + XGB + HGBC | Multi-tree variance reduction | 0.94638 | 0.99798 |
| **Single XGBoost** (`mikhailnaumov`) | Depth-wise XGBoost | Exact split histograms, depth-constrained | 0.94635 | 0.99877 |
| **Focal Loss LightGBM** (`sergeyqt2024`) | LightGBM with asymmetric loss | Hard-example micro-sensor for borderline cases | 0.94533 | 0.99687 |
| **PyTorch RealMLP** (`yekenot`) | Deep Tabular Neural Network | Continuous manifold, PLR embeddings, smooth curves | 0.94595 | **0.99641** |

---

## 4. The 0.94644 Winning Blend Architecture

Megayak's audited blend pipeline systematically layers multi-source GBDTs and anchors them with continuous neural manifold predictions:

```python
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# Standard percentile rank transform into [0, 1]
rank01 = lambda s: rankdata(s) / len(s)

# Stage 1: Consensus Baseline
base = 0.50 * rank01(sub_nina) + 0.50 * rank01(sub_zoomzoom)

# Stage 2: Tree Diversity Integration (Kospintr Stack + Mikhail XGB)
E = 0.90 * base + 0.05 * rank01(sub_kospintr) + 0.05 * rank01(sub_mikhail)

# Stage 3: Feature-Engineered SOTA Tree Anchor (Najiama Pure LGBM)
F = 0.50 * rank01(E) + 0.50 * rank01(sub_najiama)

# Stage 4: Neural Manifold Decorrelation (PyTorch RealMLP Anchor)
# 15% RealMLP sits directly in the center of the [10%, 20%] plateau
final = 0.85 * rank01(F) + 0.15 * rank01(sub_realmlp)
```

---

## 5. Post-Processing & Calibration

For ROC-AUC competitions on Kaggle, strictly monotonic transformations preserve the metric identically. However, calibrating probabilities to the true positive prior ($17.4645\%$) ensures predictions are well-calibrated probabilities rather than arbitrary ranks:

$$\text{logit}(p_{\text{calib}}) = \text{logit}\left(\frac{\text{rank} - 0.5}{N}\right) + c^*$$
where $c^*$ is solved via 1D root-finding such that $\mathbb{E}[p_{\text{calib}}] = 0.174645$.

- **Exact Rank Preservation**: $\text{diff\_ranks} = 0$, $\Delta\text{AUC} = 0.0$.
- **Validation**: Meets both calibration boundaries ($0.170 \le \bar{p} \le 0.180$) and rank validity.

---

## 6. Key Takeaways & Competitive Rules

1. **The 0.998 Spearman Rule**: Never waste a submission or compute evaluating a model that correlates $> 0.998$ with your existing blend.
2. **Pair Trees with Neural Networks**: In tabular competitions, GBDT ensembles inevitably hit a glass ceiling. Always include at least one well-regularized deep tabular network (RealMLP, TabNet, or TabM) to cancel tree boundary artifacts.
3. **Never Pre-Pool Diverse Candidates**: Averaging diverse models together before blending cancels their individual disagreements. Add them individually.
4. **Choose the Plateau Center**: When multiple weights score identically on public LB, always pick the center of the plateau to minimize private test shakeout.
5. **Auditing Upstream Notebooks**: Public notebook re-runs can alter predictions at the 5th decimal place due to GBDT non-determinism. Blend weights have timestamps.
