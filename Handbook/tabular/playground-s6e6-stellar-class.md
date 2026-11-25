# Playground Series S6E6: Predicting Stellar Class — Top Solutions & Post-Mortem

> **Competition**: [Kaggle Playground Series Season 6 Episode 6](https://www.kaggle.com/competitions/playground-series-s6e6) (Jun 1 – Jul 1, 2026 | 2,816 Teams | 33,508 Submissions)  
> **Target**: 3-Class Imbalanced `class` (`GALAXY` `65.38%`, `QSO` `20.29%`, `STAR` `14.33%`)  
> **Evaluation Metric**: **Balanced Accuracy** (`balanced_accuracy_score`)  
> **Dataset**: `577,347` Train Rows × `12` Cols (`alpha, delta, u, g, r, i, z, redshift, spectral_type, galaxy_population`) | `247,435` Test Rows × `11` Cols | Original Seed: `100,000` Rows ([SDSS17 Stellar Classification](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17))  
> **LB Split**: ~20% Public (`~49.5k` rows, $1\sigma \approx 0.00087$) / ~80% Private (`~198k` rows, $1\sigma \approx 0.00035$) — **catastrophic Public LB shakeup** (entire Public Top 10 dropped out; Private Top 10 jumped **+144 to +485 ranks**!)

---

## Official Top Standings & Solution Index

Public LB was overrun by high-variance "ridge-flip" / public-LB-probing blends at `0.9726–0.9729` that collapsed on the 80% Private split. Every single Private Top-10 winner climbed **+144 to +485 places** by trusting local 5-fold CV and proper class-prior calibration:

| Private Rank | Δ Rank | Team / Competitor | CV (OOF) | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **+342** | **`Optimistix`** | **0.970598** | 0.97172 *(#343)* | **0.97075** | **89-model MLP stacker** (trusted higher CV over higher-Pub LR-Logits `0.97179`) + pruned back from 100–150+ models when CV degraded; CatBoost/RealMLP/TabM/AutoML zoo | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/1st-place-mission-300-accomplished) |
| 🥉 **#3** | **+146** | **`nybbler`** | — | #149 | **0.97062** | **2D Sky-Plane Forensics**: **HEALPix** pixelization + TE & spatial kNN class fractions on `(alpha, delta)` + ~180 broad decorrelated models incl. **GMM Bayes Classifier** | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/same-light-different-space) · [Spatial kNN NB](https://www.kaggle.com/code/omadon/s6e6-spatial-knn-class-fraction-features) |
| 🏅 **#6** | **+144** | **Andreas Palmgren** | **0.970718** | 0.97130 *(#150)* | **0.97054** | Strict 5-fold OOF-gated Codex loop over **92 diverse models** (24 RealMLP, 19 XGB, 12 CB, 5 LGB, FM, FT-T, TabM, TabPFN, TabR) | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/6th-place-solution-trusting-the-oof-plateau) |
| 🏅 **#8** | **+330** | **Jerry** | — | #338 | **0.97044** | **Weighted Log-Loss Surrogate Proof** (`w_c = 1/(K·π_c)`): unified weighted GBMs/LR (**Route A**) + unweighted RealMLP/TabPFN divided by empirical priors (**Route B**) → GPU LR stacker | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/8th-place-solution) |
| 🏅 **#9** | **+390** | **Vaibhav Nakrani** | **0.970355** | 0.97121 *(#399)* | **0.97043** | L2 LR stack over **63 diverse base models** (Claude Code supervised loop); RealMLP solo `~0.9693` broke `0.964` GBDT ceiling | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/9th-place-solution) |
| 🏅 **#12** | **+395** | **`Alexdruso`** | **~0.9705** | 0.97108 *(#407)* | **0.97041** | Broke `0.9662` GBDT plateau via 3 axes: 240-feat quantile-TE XGB (`0.9671`), **from-scratch RealMLP w/ logit-adjusted loss** (`0.9690`), & **Chain-Cascade** CatBoost (`P(QSO)` then `P(STAR\|¬QSO)`) | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/12th-place) |
| 🏅 **#19** | **+282** | **Zihe Henry Zhang** (`caesium137`) | **0.97118** | #301 | **0.97036** *(est.)* | Verified correlation-pruned 44-model LR stack (`TabPFNv2` `0.96915` strongest own) + per-class log-offset (`+0.009` over argmax) + noise-floor gate (`Δ ≥ 0.0006`) | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/19th-place-my-public-skill-md-for-your-agents) · [GitHub](https://github.com/caezium/s6e6-genuine-oof-stacking) |
| 🏅 **#24** | **+438** | **`cstdy`** (`kirill0212`) | **0.97043** | #462 | **0.97033** | 32-model stack blending `cdeotte` GPU LR stacker + `philippsinger` **TabPFN-3 stacker** + OvR XGB/TabM (`0.96862`) | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/24th-place-solution) · [Stacker NB](https://kaggle.com/code/kirill0212/ps6e6-lr-tabpfn-3-stacker) |
| 🏅 **#25** | **+167** | **Chris Deotte** (`cdeotte`) | **0.97045** | 0.97094 *(#192)* | **0.97033** | 20-model starter + Codex upgrade (**fine-tuned TabICLv2** + **OvR CatBoost**) + PyTorch GPU Multinomial LR (`RealMLP v5 0.96904`, `CatBoost v3 0.96892`, `XGB v5 0.96770`) | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/25th-place-my-public-starter-notebook) · [GPU LR NB](https://www.kaggle.com/code/cdeotte/gpu-logistic-regression-stacker) |

---

## 1. Data Forensics: Why S6E6 Broke Classic Synthetic Playground Tricks

### 1.1 Exact Post-Synthesis Derived Categoricals (`broccoli beef`, Discussion 703535)
Two categorical columns were computed **post-synthesis** with **100% deterministic accuracy** across train and test (must be recomputed on the 100k original SDSS17 rows before appending):
- `spectral_type = pd.cut(r - g, [-np.inf, -1, -0.5, 0, np.inf], labels=['M', 'G/K', 'A/F', 'O/B'])`
- `galaxy_population = pd.cut(u - r, [-np.inf, 2.2, np.inf], labels=['Blue_Cloud', 'Red_Sequence'])`

### 1.2 The Continuous-Generator Regime Shift (No Snap / Digit Artifacts!)
Unlike S6E1–S6E4 where continuous features repeated original values with small additive noise:
- **100% Unique Floats**: All 8 numerical features (`alpha, delta, u, g, r, i, z, redshift`) were continuous unique floats (`test → train` nearest neighbor distance `0.034`, `test → SDSS17` `0.042`; adversarial validation train-vs-test AUC = **`0.4995`**).
- **Null Generator-Artifact Proof (`vedant pol`)**: Original-value snap TE gave only `64.7%` LOO accuracy (≈ majority base rate); decimal-digit mutual information was **`0.0005`** (vs `redshift` MI = `0.50`); 1-NN to original SDSS17 scored only `11%` on model disagreements.
- **Scrambled Radial Physics vs. Preserved 2D Sky Plane (`nybbler`, #3)**: The VAE/diffusion-like generator broke real-world redshift physics (injecting synthetic `STAR`s up to `z = 5.4`, destroying 3D cosmological `astropy` voxels), **yet preserved 2D celestial `(alpha, delta)` sky-plane clustering** — unlocking **HEALPix** pixel target encoding and **Spatial kNN class-fraction features** (`omadon`).

---

## 2. Mathematical Calibration for Balanced Accuracy

Plain multinomial cross-entropy argmax severely over-predicted `GALAXY` (`98%` recall) and under-predicted minority `STAR` (`94%` recall). Top teams solved this via two mathematically exact frameworks worth **`+0.0066` to `+0.0090` Balanced Accuracy**:

### 2.1 Convex Weighted Log-Loss Surrogate & Dual-Route Alignment (`Jerry`, #8)
Balanced Accuracy is non-differentiable and noisy as a CV selection metric; **class-weighted negative log-loss** with $w_c = \frac{1}{K \pi_c}$ is strictly proper/convex and yields the **exact population Bayes-optimal Balanced Accuracy rule** $\arg\max_c \frac{\eta_c(x)}{\pi_c}$:
- **Route A (Weighted Models — GBDTs, LR)**: Pass sample weights $w_i = 1 / \pi_{y_i}$ during training.
- **Route B (Unweighted Models — RealMLP, TabPFN)**: Train with standard unweighted cross-entropy, then **divide predicted OOF/test probabilities by training class priors `[0.653818, 0.202899, 0.143283]` and renormalize** (or train with **logit-adjusted loss** / `loss_prior_power=1.075` in RealMLP, #12 & #25).
- Aligning every base model onto the same prior-tilted probability scale before stacking eliminated cross-family scale distortion.

### 2.2 Meta-Stacking Under Heavy Public-LB Variance
- **MLP Stacker vs. GPU Multinomial LR vs. TabPFN-3**:
  - **1st Place (`Optimistix`)** proved that an **89-model MLP stacker** beat LR-Logits on CV (`0.970598` vs `0.970510`) despite *lower* Public LB (`0.97172` vs `0.97179`), winning **#1 Private (`0.97075`)**—and crucially **pruned back from 100–150+ models** when excess collinearity degraded CV.
  - **24th Place (`cstdy`)** blended `cdeotte`'s PyTorch GPU Multinomial LR stacker (`5 seeds × 5 folds`) with `philippsinger`'s **TabPFN-3 stacker** (`0.97043` OOF).
  - **Monte Carlo Hedging Proof (`Siddhesh Sathe`, Discussion 704512)**: Across 4,000 draws of 198k Private rows, selecting a weaker decorrelated 2nd submission gave **zero expected uplift** over selecting your two strongest CV stacks.

---

## 3. Feature Engineering & Model Architecture Breakthroughs

- **Continuous Manifold Dominance — RealMLP & TabPFN Over GBDTs**:
  - Default GBDTs plateaued near `0.964–0.966`, whereas **RealMLP** (`cdeotte` `v5`: **`0.969039`** CV / `0.96979` LB with `flat_ratio=0.2, dropout=0.044, weight_decay=0.0125, loss_prior_power=1.075, ema_decay=0.997875`; `Vaibhav` `~0.9693`; `Alexdruso` logit-adjusted `0.9690`; `yekenot` PyTorch RealMLP `0.97055` LB) and **`TabPFNv2`** (`0.96915`, #19) + **fine-tuned `TabICLv2`** (#25) jumped `+0.003–0.005` solo!
  - Strongest GBDTs required astronomical color indices (`u-g`, `g-r`, `r-i`, `i-z`, `u-r`, magnitude sums/ratios, redshift ratios) + high-cardinality GPU features (`CatBoost v3` **`0.96892`** with 241 features & `0.06` original-SDSS weight; `XGB v5` **`0.96770`** with 370+ features, `max_bin=960`, TE smoothing `16`, zero original rows appended).
- **Target Chain-Cascade & OvR Diversity**: Factoring `P(QSO)` and `P(STAR | ¬QSO)` via two binary CatBoosts (#12) plus One-vs-Rest XGB/TabM/CatBoost (#24, #25) and a **GMM Bayes Classifier** (#3) decorrelated residual errors on the critical `GALAXY ↔ STAR` (`redshift ≈ 0`) boundary.

---

## 4. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Outcome | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Chasing Public LB / "Ridge-Flip" Blends** | Entire Public Top 10 fell out of Private Top 10; Private Top 10 climbed **+144 to +485 ranks** | #1–#25 | 20% Public split noise ($1\sigma \approx 0.00087$) swamped true CV differences; gate edits by noise floor (`Δ ≥ 0.0006`, #19) |
| **Blindly Scaling Ensemble Past 100–150 Models** | Degraded CV vs. pruned 89-model stack | #1 | Collinear GBDT duplicates diluted signal in MLP/LR meta-learners |
| **Snap-to-Original / Decimal-Digit / 1-NN SDSS Lookup** | Snap LOO `64.7%` (base rate); digit MI `0.0005`; 1-NN `11%` on disagreements; exact `(alpha, delta)` lineage overfit | #9, #12, `vedant pol` | Continuous VAE/diffusion generator produced 100% unique floats without discrete copy artifacts |
| **3D Cosmological Voxels (`astropy` on `alpha, delta, z`)** | Failed completely | #3 | Synthetic generator scrambled radial redshift distances (creating `STAR` at `z = 5.4`) |
| **Cleanlab Label Pruning, DAE Latents & Pseudo-Labeling** | Hurt CV/LB; DAE blurred `STAR/GALAXY` boundary | #9, #12 | Cleanlab discards real boundary examples; synthetic noise isn't cleanable label error |
| **Hill Climbing in Logit Space / `cuML` LR Convergence** | Underperformed PyTorch GPU Multinomial LR | #8, #25 | PyTorch Adam multinomial softmax with balanced weights converged more reliably on high-dim stacked OOFs |
| **Flux / Luptitude / RBF / Nystroem Transforms** | Low error correlation (`0.48–0.54`) but solo `~0.94` (too weak to lift stack) | #9 | Excessive solo accuracy deficit negates diversity benefit |
