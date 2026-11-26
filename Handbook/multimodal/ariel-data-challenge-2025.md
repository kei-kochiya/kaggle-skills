# NeurIPS — Ariel Data Challenge 2025: Top-14 Solutions & Bayesian Uncertainty Post-Mortem

> **Competition**: [NeurIPS — Ariel Data Challenge 2025](https://www.kaggle.com/competitions/ariel-data-challenge-2025) (Jun – Sep 2025, 860–876 Teams, \$50,000 Prize Pool; hosted by University College London)  
> **Track**: Scientific ML, Exoplanet Transit Spectroscopy, Inverse Physics & Calibrated Uncertainty Estimation  
> **Data & Runtime**: `264.3 GB` raw sensor time-series from **FGS1** (`135,000×32×32`, `0.60–0.80 µm`, 1 bin) and **AIRS-CH0** (`11,250×32×356`, `1.95–3.90 µm`, 282 bins) with oscillating CDS readout, calibration frames (`dark/dead/flat/linear_corr/read`), noisy `star_info.csv`, and repeated visits; `~1,100` test planets in `<= 9h` offline notebook  
> **Evaluation Metric**: Normalized clipped **Gaussian Log-Likelihood (GLL)** scoring both predicted transit depth mean $\mu$ and calibrated uncertainty $\sigma$ across 283 bins, with **FGS1 weighted `57.846×`** each AIRS bin (`~17%` of total metric weight on 1 scalar!) and ideal $\sigma_{\text{AIRS}} = 10\text{ ppm}$, $\sigma_{\text{FGS1}} = 1\text{ ppm}$

---

## 1. Official Top-14 Leaderboard & Paradigm Comparison

Top-3 rankings held identically across Public and Private LB, led by **Jeroen Cottaar's `+0.03656` Private blowout** using zero neural networks (pure iterative linearized Bayesian inference + 12-parameter GLL post-calibration), followed by two radically different paradigms (#2 pure empirical slope/bootstrap Ridge; #3 zero-binning CNN + 36-ensemble Rational Quadratic NN).

| Private Rank | Team | Private GLL | Public GLL (Δ) | Paradigm & Standout Innovations | Writeup / Code |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **1st** | **Jeroen Cottaar** | **0.64952** | **0.63700** (0) | **Pure Iterative Linearized Bayesian Inference (zero NNs)**: dispersion-axis jitter/non-Poisson extraction (`+0.013`) + `batman` 11-param LD + 2-kernel GP + 5-PCA prior + 200 posterior samples + **12-param GLL calibration** | [Writeup](https://www.kaggle.com/competitions/ariel-data-challenge-2025/writeups/1st-place-solution-bayesian-inference-of-course) · [GitHub](https://github.com/jcottaar/ariel2) |
| 🥈 **2nd** | **BeeMax** | **0.61296** | **0.61014** (0) | **Zero physics simulator**: 2nd-deriv boundary detection + degree-opt poly detrending + **wall-steepness slope features (`+7.1` Priv!)** + percentile depth stats (`+4.0`) + **bootstrapped $\sigma$ (`+14.6` Pub / `+3.6` Priv)** + Ridge | [Writeup](https://www.kaggle.com/competitions/ariel-data-challenge-2025/writeups/2nd-place-solution) · [Notebook](https://www.kaggle.com/code/beemax/neurips-adc-25-2nd-place-submission) |
| 🥉 **3rd** | **beez** (`moonpole`) | **0.60335** | **0.59154** (0) | **Zero binning** + **$8\sigma$ outlier filter (`+0.05` CV)** + CNN + **36-ensemble 88-cluster Rational Quadratic (RQ) NNs (`+0.15` CV)** + **Median-Thresholded Ensemble Loss (`+0.03+` CV)** | [Writeup](https://www.kaggle.com/competitions/ariel-data-challenge-2025/writeups/3rd-place-solution) · [Notebook](https://www.kaggle.com/code/zayyyy/ariel-25-3rd-place-submission) |
| 🏅 **4th** | **Thomas Dueholm Hansen** | **0.59081** | **0.58777** (+2) | **Kalman Smoother** init + 10-stage FP64 differentiable LD simulator + conditional 7-PCA subspace blend (`dist < 0.0015`) + dual ReLU×Sigmoid $\sigma$ MLPs | [Writeup](https://www.kaggle.com/competitions/ariel-data-challenge-2025/writeups/4th-place-solution) · [GitHub](https://github.com/thomasdueholm/kaggle_ariel_2025) |
| 🏅 **5th** | **Alehandreus** | **0.58956** | **0.58192** (+3) | **PyTorch FP64 40-ring integral LD + Levenberg-Marquardt** (220 iters) + grazing-transit guard ($b > 0.75 \implies \sigma \times 7$) + **GBDT $(\Delta\mu, \sigma)$ refinement (`+0.029` Priv)** | [Writeup](https://www.kaggle.com/competitions/ariel-data-challenge-2025/writeups/5th-place-solution) · [GitHub](https://github.com/Alehandreus/ariel-2025) |
| 🏅 **6th** | **Vicens Gaitan** | **0.58138** | **0.57740** (+3) | CuPy GPU preprocessing (`<1.5h` P100) + hierarchical **`batman` + `iminuit` (MIGRAD)** + PCA/Savitzky-Golay blend + **multi-term empirical $\sigma$** (`err`, $|p_{\text{PCA}} - p_{\text{SG}}|$, mismatch) | [Writeup](https://www.kaggle.com/competitions/ariel-data-challenge-2025/writeups/6th-solution-writeup) · [Notebook](https://www.kaggle.com/code/vicensgaitan/6th-place-solution-pipeline) |
| 🏅 **7th** | **Horikita Saku & takaito** | **0.58081** | **0.57303** (+4) | GP physics baseline (`0.45`) → BiGRU/BiLSTM+CNN residual `wl_diff` + quantile $\sigma$ (`0.55`) → **GBDT scalar $\sigma$ multiplier (`+0.01`)** + `wl_diff` PL (`+0.004`) | [Writeup](https://www.kaggle.com/competitions/ariel-data-challenge-2025/writeups/7st-place-solution) |
| 🏅 **8th** | **siwooyong** | **0.57753** | **0.58805** (-3) | `scipy.least_squares(trf, jac_sparsity)` 2-stage deg-3→4 LD solver + train-mean fallback on bad fits + 4-fold `GBR` per-sample $\sigma$ | [Writeup](https://www.kaggle.com/competitions/ariel-data-challenge-2025/writeups/8th-place-solution) · [GitHub](https://github.com/siwooyong/NeurIPS-Ariel-Data-Challenge-2025) |
| 🏅 **9th** | **CD** (`CPMP`, `Dieter`) | **0.57088** | **0.56083** (+4) | Fixed **ADC offset sign & dark-frame `dt` integration scaling (`>+0.005`)** + `batman` & `gpytorch` → log-space PyTorch `GLLLoss` combiner | [Writeup](https://www.kaggle.com/competitions/ariel-data-challenge-2025/writeups/9th-solution) |
| **12th** | **sroger** | **0.55990** | **0.55786** (+4) | Pure E2E BERT-style Transformer outputting $(\mu, \log\sigma^2)$ under direct `GLLLoss` + 3-seed inverse-variance blend | [Writeup](https://www.kaggle.com/competitions/ariel-data-challenge-2025/writeups/12th-place-solution-e2e-nn) |
| **14th** | **Hello Aliens** | **0.52781** | **0.58811** (**-10**) | FP64 GPU grid posterior — **cautionary shakeup**: coarse inclination grid produced near-zero $\sigma$ on a partial transit without a `MIN_SIGMA` floor! | [Writeup](https://www.kaggle.com/competitions/ariel-data-challenge-2025/writeups/14th-private4th-public-solution-brute-force-post) |

---

## 2. Signal Calibration, Physics Forward Models & Neural Alternatives

### 2.1 Sensor Calibration Forensics (`>+0.005` to `+0.05` Gains)
- **Dark-Frame Integration-Time Scaling & ADC Offset (#9 `CPMP & Dieter`, `>+0.005`)**: Subtracting dark current scaled by actual elapsed integration time `dt` since sensor reset (`dt[1::2] += 0.3` FGS1, `+= 0.2` AIRS) and fixing the inverted ADC offset formula yielded an immediate clean jump.
- **Dispersion-Axis Extraction & Non-Poisson Noise (#1 Jeroen Cottaar, `+0.013`)**: Instead of naive spatial summing across the AIRS dispersion axis, explicitly modeling PCA-identified pointing/defocus jitter, invalid pixels, non-Poisson brightness-dependent noise, and constant background added `~+0.013` GLL.
- **Zero-Binning + $8\sigma$ Temporal Outlier Clipping (#3 `beez`, `+0.05` CV)**: Avoiding premature time binning and masking temporal cosmic-ray/glitch spikes (`|x - mean| > 8*std → NaN`) delivered `+0.05` CV alone.
- **FGS1 Aperture + Center-of-Mass Jitter Correction (#6, #9)**: Because FGS1 carries **`57.846×`** weight on a single bin, aperture photometry + CoM X/Y jitter decorrelation beat PSF photometry on FGS1 (while disabling PCA jitter on AIRS helped #6).

### 2.2 Three Winning Modeling Paradigms
1. **Iterative Linearized Bayesian Inference (#1 Jeroen Cottaar — `0.64952`)**:
   - Modeled the complete observation jointly: uncorrelated per-$(t,\lambda)$ Gaussian noise + unregularized star spectrum + 3rd-order poly drift + **11-param `batman` transit** (with MLE-learned parameter covariance from train, including 6 wavelength-linear quadratic LD params) + **transit depth prior $D(\lambda)$** combining a global mean, **sum of 2 squared-exponential GPs over 282 AIRS wavelengths**, **5 learned PCA spectral modes**, and a per-planet amplitude scalar.
   - Solved via coarse `(D, t0)` grid → `BFGS` on wavelength-averaged AIRS → `4–8` iterations of Laplace/Gauss-Newton linearization (`4` iters in 9h notebook cost only `0.001`) + **200 posterior samples** for $(\mu, \sigma)$, with a high-residual fallback chopping boundary frames.
2. **Differentiable FP64 Physics Simulators + GBDT/PCA Refinement (#4, #5, #6, #8)**:
   - Built custom FP64 differentiable limb-darkening transit integrators optimized via Levenberg-Marquardt (#5), `iminuit` MIGRAD (#6), `trf` with sparse Jacobians (#8), or 10-stage Kalman-initialized optimization (#4) — FP32 failed on degree-4 drift polynomials (#14)!
   - Regularized raw spectra using **low-rank PCA (`3–7` components)** (blended conditionally in #4 only when subspace distance `< 0.0015`) + **GBDT residual $(\Delta\mu, \sigma)$ correction (`+0.029` Priv in #5!)**.
3. **Pure Empirical / Neural Models (#2 `0.61296`, #3 `0.60335`, #12 `0.55990`)**:
   - **#2 `BeeMax`**: Proved a physics-free Ridge pipeline could take 2nd place by feeding **transit wall-steepness slopes** (implicit LD/impact-parameter proxy: removing them lost **`-7.1` Priv!**) + multi-percentile depth stats (`-4.0` Priv without!) + SNR-weighted AIRS + separate edge-transit Ridge.
   - **#3 `beez`**: Shared-weight CNN + **36-ensemble Rational Quadratic (RQ) cluster NNs (`+0.15` CV)** trained with a **Median-Thresholded Ensemble Loss (`+0.03+` CV)**: `w = stop_gradient(relu(losses - median(losses)))` updates **only ensemble members worse than the median**, preventing ensemble collapse and overfitting!

---

## 3. Uncertainty ($\sigma$) Calibration Under Gaussian Log-Likelihood

Under $\text{GLL} \propto -\log\sigma^2 - (y - \mu)^2 / \sigma^2$, predicting $\sigma$ accurately is as important as predicting $\mu$:
- **Catastrophic Overconfidence Trap (#14's `-10` Shakeup)**: A single planet with a collapsed $\sigma \to 0$ explodes $(y-\mu)^2/\sigma^2$ and wipes out hundreds of well-predicted planets. **Every stable top solution enforced strict `MIN_SIGMA` floors** and explicit outlier/grazing-transit inflation (e.g., #5 inflated $\sigma \times 7$ for grazing impact parameter $b > 0.75$; #8 fell back to train mean/std when fit error $>10^{-3}$ or transit hit boundary `T1 < 0.01 / T4 > 0.99`).
- **Decomposing Aleatoric + Epistemic + Model-Disagreement $\sigma$**:
  - **1st Place 12-Param Post-Hoc Calibration**: Calibrating $(\mu, \sigma)$ per sensor (`AIRS` and `FGS1` separately!) with offset + scale + spectral-variation amplitude moved #1 from **~20th place to 1st place** (`0.64952`)!
  - **2nd Place Bootstrap $\sigma$**: Combining resampled bootstrap prediction std with CV per-frequency std added **`+14.6` Pub / `+3.6` Priv** over static validation RMSE.
  - **6th Place Multi-Term Empirical $\sigma$**: Directly minimized GroupKFold GLL over fit Hessian error, SNR power-law, base $\sigma_0$, **inter-method disagreement $|p_{\text{PCA}} - p_{\text{SG}}|$**, and FGS–AIRS mismatch.
  - **7th Place Scalar Multiplier**: A GBDT predicting a single per-planet $\sigma$ multiplier (separate for FGS1/AIRS) added **`+0.01` LB** on top of quantile-regression NN uncertainties.

---

## 4. Empirical Negative Results & What Failed

| Failed Hypothesis | Observed Result | Root Cause |
| :--- | :--- | :--- |
| **Uncalibrated Raw Bayesian / GP Posterior $\sigma$ (#1, #7)** | Dropped #1 from **1st → ~20th** without 12-param fudging; raw GP $\sigma$ lost `~0.10` in #7 | Model misspecification (imperfect LD/jitter/drift assumptions) makes raw Bayesian posteriors systematically overconfident under GLL. |
| **Missing `MIN_SIGMA` Floor on Coarse Grid Posteriors (#14)** | **Public #4 (`0.58811`) → Private #14 (`0.52781`, `-10`)** | Coarse inclination grid collapsed posterior variance to `~0` on a partial-transit planet, blowing up squared z-score penalty. |
| **FP32 Optimization on Degree-4 Drift Polynomials (#14)** | Numerical instability / failure | High-condition-number polynomial/LD systems required **FP64** (`#4, #5, #14`). |
| **Jointly Predicting $\mu$ Correction Inside $\sigma$ MLP (#4)** | Overfit CV/LB | Separate mean estimation from out-of-fold $\sigma$ calibration to prevent the network trading mean accuracy against variance inflation. |
| **L1/L2, Dropout, Splicing Aug & Mahalanobis in RQ NN (#3)** | Unstable or degraded CV | Median-thresholded ensemble gradient masking regularized cleanly without distorting calibrated likelihood outputs. |
| **Polynomial Drift Degree > 3 Without Physical Bounds (#6, #7)** | Destabilized transit depth recovery | High-degree polynomials absorb real transit curvature unless bounded jointly with physical orbital geometry. |
| **PSF Photometry & AIRS PCA Jitter Removal (#6)** | Lower SNR than FGS1 aperture photometry / hurt AIRS | Pointing jitter smeared fixed PSF templates; AIRS additive foreground noise resisted linear PCA decorrelation. |

---

## 5. Universal Takeaways Promoted / Reinforced

- **Post-Hoc Uncertainty Calibration & Hard `MIN_SIGMA` Floors Are Mandatory Under GLL**: Never submit raw analytical/Bayesian or ensemble-std $\sigma$ without out-of-fold GLL scale calibration, hard lower clips, and geometry/fit-residual fallback inflation for boundary/OOD samples.
- **Separate Mean ($\mu$) Fitting from Out-of-Fold Uncertainty ($\sigma$) Calibration**: Combine epistemic spread (bootstrap / posterior / ensemble), aleatoric noise, and **cross-method disagreement (`|pred_A - pred_B|`)** as features for a dedicated per-channel $\sigma$ calibrator.
- **FP64 Precision for Ill-Conditioned Physics/Polynomial Solvers & Median-Thresholded Ensemble Loss**: Use `float64` for coupled drift+physics nonlinear least-squares (`iminuit`/`LM`/`trf`), and consider median-thresholded ensemble updates (`relu(loss - median)`) to prevent deep ensemble collapse under likelihood losses.
