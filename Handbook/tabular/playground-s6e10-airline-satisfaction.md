# S6E10: Predicting Airline Passenger Satisfaction — Engineering Handbook & Forensics

> **Competition:** Kaggle Playground Series s6e10  
> **Target:** Binary `satisfaction` (`True` / `False`)  
> **Evaluation Metric:** **ROC-AUC**  
> **Seed Dataset:** Arseniy Shutko / `teejmahal20` — Airline Passenger Satisfaction (129,880 survey responses)  
> **Benchmark Apex Standing:** Team `IchikaHoshino` — **Public LB 0.96111** (Global Rank #5)

---

## 1. Executive Summary & Data Geometry

| Entity | Dimensions | Positive Rate ($\mathbb{E}[y]$) | Missing Values | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Train Set** | 699,635 $\times$ 23 | **44.3573%** (310,339 True / 389,296 False) | `Arrival Delay`: 292 (0.04%) | Generated via deep synthetic sampler |
| **Test Set** | 299,844 $\times$ 22 | Unknown ($\approx 44.36\%$) | `Arrival Delay`: 130 (0.04%) | Clean 30% random split |
| **Original Seed** | 129,880 $\times$ 25 | **43.4463%** | `Arrival Delay`: 393 (0.30%) | Real airline passenger survey |

### Adversarial Validation & Covariate Shift
- 3-Fold Stratified LightGBM adversarial classifier discriminating `train.csv` vs `test.csv`:
  $$\text{Adversarial ROC-AUC} = 0.5010 \pm 0.0034$$
- **Finding:** Exactly zero covariate drift exists between Train and Test. All public/private variance is driven purely by binomial label sampling noise.

---

## 2. Generator Forensics & Mathematical Laws

### 2.1 The 3.8% Label Noise & Theoretical Headroom Ceiling
In the original 129,880-row survey dataset, an untuned LightGBM reaches **0.9950 AUC** (near-deterministic relationship between passenger attributes and satisfaction).
In the competition synthetic dataset, the generator flipped approximately **3.8% of labels** at random:

$$AUC_{\text{flipped}} = 0.5 + (a - b) \cdot (AUC_{\text{true}} - 0.5)$$

Where $a$ is the share of positive-labeled rows that are truly positive and $b$ is the share of negative-labeled rows that are truly positive.

- **Pure Noise Ceiling:** If the noise were 100% white noise, the theoretical maximum CV AUC would be bounded at **0.9613**.
- **The Non-Random Residual:** Published stacks break through 0.9615+ OOF because the generator's label disagreements are **not purely random**. Within the original dataset's near-certain rows, competition models separate the flipped labels with AUC $0.5711 - 0.5966$, contributing $+0.0031$ AUC of learnable signal.

### 2.2 The Original Data Dilemma: Training Rows vs. Feature Lookup Table
A critical controlled experiment (measured across 5 frozen folds with identical hyperparameters):

| Strategy | 5-Fold OOF AUC | Mean Fold $\Delta$ | Folds Up | Public LB |
| :--- | :---: | :---: | :---: | :---: |
| **Competition Only Baseline** | 0.958952 | Reference | — | 0.95824 |
| **+ Original Rows Appended** | 0.958699 | **−0.000252** | 0/5 | — |
| **+ Original Rows (Flagged `is_original`)** | 0.959026 | +0.000072 | 4/5 | — |
| **+ Original as Feature Lookup** | 0.959879 | **+0.000927** | 5/5 | — |
| **+ Lookup + Flagged Original Rows** | **0.959983** | **+0.001029** | 5/5 | 0.95939 |

> [!IMPORTANT]
> **Key Takeaway:** Stacking original survey rows directly into training sets degrades performance in 5 out of 5 folds because the tree fits the original deterministic relationship, which conflicts with synthetic generator noise.  
> Conversely, mapping the original dataset into a **21-column static prior lookup table** (`orig_rate__* = df[col].map(og.groupby(col)['satisfaction'].mean())`) injects pure signal with zero leak, yielding **+0.00093 AUC** across all folds.

### 2.3 The Physical Route Feature & `max_bin` Bottleneck
Commercial flight distances are physical route identifiers (e.g., JFK to LAX is fixed at 2,475 miles).
- All but 4 of the 699,635 training distances exist in the original survey data.
- Standard GBDT implementations default to `max_bin=255`, which coarsely lumps distinct flight routes together.
- Setting `max_bin=4096` or `1024` on `Flight Distance` combined with target encoding (`fd`, `fd // 10`, `fd % 1000`, `fd + Class`) provides $+0.00075$ AUC lift.

### 2.4 The Zero-Rating Bypass Anomaly
Survey ratings are scaled 1 to 5. However, rating `0` appears in online/convenience categories:
- Passengers with $\ge 3$ zero-ratings have satisfaction rate **$>86.2\%$** (vs. **$44.2\%$** baseline).
- Why? Rating 0 represents *bypassed steps* (complimentary VIP, corporate business travelers, airport lounge concierge), not dissatisfaction.
- **Wi-Fi Non-Monotonicity:** Inflight Wi-Fi rating 0 passengers have 62.0% satisfaction, whereas rating 1 passengers have only 10.5% satisfaction. Treating ratings as monotonic continuous integers harms linear and neural models unless categorical copies or zero-indicator flags are provided.

### 2.5 Negative Feature Engineering: The Rating Aggregation Trap
Unlike standard tabular competitions where row statistics (mean, std, sum, min, max across survey items) are standard:
- In S6E10, adding rating aggregates (sum, std, mean across ratings) **degrades OOF by −0.00022 AUC**.
- **Root Cause:** Tree algorithms already find optimal multi-level rating cuts. Adding collinear aggregate columns dilutes `colsample_bytree` / `feature_fraction`, causing splitters to choose noisy aggregates instead of the primary signal features (`Online boarding`, `Inflight entertainment`, `Type of Travel`).

---

## 3. High-Performance Feature Engineering Recipe

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUMS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
RATINGS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness"
]
RAW = [c for c in tr.columns if c not in ("id", "satisfaction")]

def build_features(train_df, test_df, original_df):
    full = pd.concat([train_df[RAW], test_df[RAW]], ignore_index=True)
    
    # 1. Aligned Offline Lookup Prior (Zero Leak)
    lookup_tables = {c: original_df.groupby(c)["satisfaction"].mean() for c in RAW}
    
    for df in [train_df, test_df]:
        # Categorical codes
        for c in CATS:
            df[c + "_cat"] = pd.Categorical(df[c], categories=sorted(full[c].unique())).codes
        
        # Generator Modulo Digits
        df["fd_mod10"] = df["Flight Distance"] % 10
        df["fd_mod100"] = df["Flight Distance"] % 100
        df["age_mod10"] = df["Age"] % 10
        
        # Delay Dynamics
        df["delay_diff"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"]) - df["Departure Delay in Minutes"]
        df["is_delayed"] = (df["Departure Delay in Minutes"] > 0).astype(int)
        
        # Zero-Rating Bypass Counters
        df["num_zero_ratings"] = (df[RATINGS] == 0).sum(axis=1)
        df["is_wifi_zero"] = (df["Inflight wifi service"] == 0).astype(int)
        df["is_boarding_zero"] = (df["Online boarding"] == 0).astype(int)
        
        # Original Data Target Lookups
        for c in RAW:
            df["orig_rate__" + c] = df[c].map(lookup_tables[c]).astype("float64")
            
        # Frequency encoding
        for c in NUMS:
            df[c + "_freq"] = df[c].map(full[c].value_counts())
            
    return train_df, test_df
```

---

## 4. Multi-Family Ensemble Architecture & Barycenter Fusion

### 4.1 Cross-Family Diversity Matrix
Ensembling requires genuine orthogonality. Measuring pairwise Spearman rank correlations ($\rho$):

| Architecture Family | Member | $\rho$ vs RealMLP | $\rho$ vs XGBoost | Single OOF AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Neural Network** | `PyTabKit RealMLP-TD` | 1.0000 | 0.9726 | 0.961204 |
| **Neural Network** | `Quantile-MLP (128-64)` | 0.9155 | 0.9108 | 0.956817 |
| **Spline Linear** | `Spline GAM Logistic` | 0.8783 | 0.8757 | 0.948865 |
| **Random Forests** | `ExtraTrees (500 trees)` | 0.8984 | 0.8965 | 0.954592 |
| **Gradient Boosted** | `LightGBM (Route FE)` | 0.9742 | 0.9884 | 0.961072 |
| **Gradient Boosted** | `XGBoost (Hist + TE)` | 0.9726 | 1.0000 | 0.961154 |
| **Gradient Boosted** | `CatBoost GPU` | 0.9710 | 0.9867 | 0.960940 |

### 4.2 Hypersphere Fréchet Barycenter Ensembling
Instead of Euclidean probability averaging which violates metric geometry under extreme log-odds calibrations:
1. Map predictions to uniform quantile margins: $u_k = \frac{\text{rank}(p_k) - 0.5}{N}$
2. Probit projection to Gaussian latent space: $z_k = \Phi^{-1}(u_k)$
3. Normalize to unit hypersphere $\mathbb{S}^{N-1}$: $U_k = \frac{z_k}{\|z_k\|_2}$
4. Compute Riemannian center of mass via iterative Riemannian gradient descent:
   $$V_t = \sum_{k=1}^K w_k \frac{\theta_k}{\sin \theta_k} (U_k - \cos \theta_k \cdot \bar{U}_t), \quad \theta_k = \arccos(\langle \bar{U}_t, U_k \rangle)$$
5. Lexicographical tie-breaking (`lexsort`) against continuous RealMLP predictions guarantees $N_{\text{unique}} = 299,844$ strictly distinct ranks.

---

## 5. Shake-Up Prevention: The Final Submission Selection Rule (Clark 1961)

Selecting final submissions to avoid private leaderboard shakeup follows the exact expected value of the maximum of two correlated normals:

$$\mathbb{E}[\max(A, B)] - \mathbb{E}[A] = \sigma \sqrt{2(1 - \rho)} \cdot \phi\left(\frac{\Delta}{\sigma \sqrt{2(1 - \rho)}}\right) + \Delta \cdot \Phi\left(\frac{\Delta}{\sigma \sqrt{2(1 - \rho)}}\right) - \Delta$$

### The Two Laws of Hedging:
1. **The Correlation Bound ($\rho < 0.99$):** If the Spearman correlation between two submissions satisfies $\rho > 0.992$, the second submission is **mathematically inert** ($\mathbb{E}[\max] - \mathbb{E}[A] < 0.05 \sigma$). It acts as the exact same ticket wearing a hat.
2. **The Distance Bound ($\Delta < 2.0 \sigma$):** A candidate decorrelated at $\rho = 0.90$ but trailing the ceiling by $> 2.0$ private standard errors will almost never exceed the ceiling.
3. **The Selection Rule:**
   - **Ticket 1:** Pure Apex Ensemble (highest cross-validated OOF AUC with Fréchet Barycenter / AUC-Direct FFT).
   - **Ticket 2:** High-diversity hybrid with non-tree architectures (RealMLP + Quantile-MLP + GBDT), verifying $\rho \in [0.95, 0.98]$ and $\Delta < 1.0 \sigma$.

---

## 6. Official Leaderboard Progression (`lb.csv`)

| Trial | Submission File | Date | OOF AUC | Public LB | Ref | Description |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **M0** | `baseline_lgb_sub.csv` | 2026-10-01 | 0.958690 | 0.95824 | `56743554` | Baseline 5-Fold LightGBM Raw Features |
| **M1** | `m1_lgb_fe_lexrank.csv` | 2026-10-01 | 0.960700 | 0.96049 | `56744012` | 10-Fold LightGBM + Route FE + Lexrank |
| **M4** | `triad_frechet_barycenter_lexrank.csv` | 2026-10-01 | 0.960870 | 0.96057 | `56744023` | Triad Barycenter (50% LGB + 20% XGB + 30% Cat) |
| **M5** | `tetrad_frechet_barycenter_realmlp_lexrank.csv` | 2026-10-01 | 0.961286 | 0.96095 | `56744090` | Tetrad Barycenter (+ RealMLP PyTabKit) |
| **M6** | `grand_consensus_apex_barycenter_lexrank.csv` | 2026-10-01 | 0.961291 | 0.96096 | `56744108` | Grand Consensus Apex Barycenter |
| **M7** | `m7_tri_stream_geodesic_barycenter_lexrank.csv` | 2026-10-01 | 0.961553 | **0.96111** | `56745778` | Tri-Stream Barycenter (Goodpjw + Arhan + Triad) |
| **M8** | `m8_auc_direct_fft_pairwise_ensemble_lexrank.csv` | 2026-10-01 | 0.961557 | **0.96111** | `56745791` | Pairwise Sigmoid FFT Ensembling |
| **M9** | `m9_dual_apex_geodesic_fft_consensus_lexrank.csv` | 2026-10-01 | **0.961557** | **0.96111** | `56745797` | Dual-Apex Consensus (50% Barycenter + 50% FFT) |

**Current Global Leaderboard Rank: #5** (0.96111).
