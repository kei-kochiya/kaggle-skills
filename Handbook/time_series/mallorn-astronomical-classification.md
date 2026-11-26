# MALLORN Astronomical Classification Challenge 🏆

> **Track:** Time Series — Irregular Multi-Band Scientific Classification (Community Prediction) · **Scale:** 893 teams · \$1,800 · Closed Jan 31, 2026  
> **Core Lesson:** On **small-N (`148` positives / `4.86%`) irregular multi-band heteroscedastic light curves** with a **spectroscopic→photometric covariate shift**, **2D `Time × Wavelength` Gaussian Process interpolation**, **parametric physical template $\chi^2$ residuals** (`TDE`, `Bazin`, `SALT2`), **adversarial-validation sample reweighting**, **external-corpus pretraining for sequence NNs**, and **base-rate-aligned OOF F1 thresholding** beat from-scratch deep learning (`F1 ~ 0.12`), SMOTE, and aggressive feature pruning.

---

## 1. Competition DNA & Shakeup Anatomy

- **Task:** Classify rare **Tidal Disruption Events (`TDE`, `target=1`)** vs. 15 other extragalactic transients (`AGN` 59%, `SN Ia` 26%, other SNe/SLSN 15%) from simulated 6-band LSST light curves (`u, g, r, i, z, y`; `Time (MJD)`, baseline-subtracted `Flux`, `Flux_err`) + redshift `Z` & Milky Way dust extinction `EBV`.
- **Scale & Imbalance:** `3,043` train objects (**only `148` positives = `4.86%`**, `19.56:1` ratio) and `7,135` test objects (~50% Public / ~50% Private).
- **Metric:** Exact **Binary F1 Score** on `{0, 1}`. Universal `~0.06–0.08` Public→Private F1 drop (`0.74–0.76` Pub vs `0.66–0.68` Priv) caused sharp shakeups (`-26` to `-53` for Public-LB chasers; `+15` to `+17` for CV/adversarial-weighted teams).

| Private # | Shake | Team | Private F1 | Public F1 (Rank) | Core Breakthroughs |
| :---: | :---: | :--- | :---: | :---: | :--- |
| 🥇 **1st** | **+3** | **John Titor** | **0.6824** *(0.7039 unsub.)* | 0.7409 (#4) | 2D `Matérn-3/2` GP (`Flux_err` noise) + **`template_chisq_tde`** (#1 feat) + `negative_flux_fraction` (#2) + CatBoost+MLP+KNN |
| 🥈 **2nd** | `-1` | **Suxing's Astronomer** | **0.6762** | **0.7574 (#1)** | 5-stream GBDTs blended 50/50 with **PLAsTiCC-pretrained `Inception+CBAM+BiGRU+Transformer` (`OOF 0.66+`)** + **conflict-zone `\|P_DL-P_GBDT\|` PP** |
| 🥉 **3rd** | `-1` | **Sigrid Nissen** | **0.6758** *(0.6903 unchosen)* | 0.7525 (#2) | Single CatBoost on 237 Avocado + 2D-GP `extrabol` blackbody + `SALT2` + ESN feature + median per-fold threshold (`415` TDEs) |
| 🏅 **4th** | **+15** | **Alexy** | **0.6684** | 0.6986 (#19) | **Adversarial-validation sample reweighting (`2.0×` vs `0.17×`)** fixing negative CV→LB corr + 222 2D-GP/Bazin feats + single XGBoost |
| 🏅 **5th** | **+17** | **Arnav Salkade** | **0.6666** *(0.6804 unchosen)* | 0.6934 (#22) | `0.2 XGB + 0.8 CatBoost` (`reg*1.6`, seed bagging) + PR-curve threshold (`346` TDEs) |
| 🏅 **7th** | **+8** | **Alejandro Ayestaran** | **0.6610** | 0.7157 (#15) | Single LightGBM (`159` GP+Bazin feats; **`bazin_global_chi2`** #1 feat) + `392` TDEs threshold |

---

## 2. EDA & Data Forensics (Irregular Multi-Band Shift)

1. **Spectroscopic vs. Photometric Redshift Covariate Shift (`Z` / `Z_err`):**
   - Train `Z` (`0.01–4.92`) is **spectroscopic** (`Z_err` is 100% null); Test `Z` (`-0.02–4.93`) is **photometric** (`Z_err ∈ [0.02, 0.12]`).
   - `#4` used **Adversarial Validation reweighting** (`~2.0×` for test-like rows vs `~0.17×` for train-only rows), flipping a negative CV→LB relationship (`0.7003` OOF → `0.6618` Pub) into positive (`0.6688` OOF → `0.6986` Pub, **+15** Priv climb!).
   - **Rest-Frame Dilation Tradeoff:** While `#1` applied $t_{\text{rest}} = t_{\text{obs}} / (1+Z)$, `#3`, `#17`, and `#64` found observer-frame GP features + passing `Z` directly outperformed explicit K-correction/time-dilation—because noisy photometric test `Z` propagates error into rescaled timescales!
2. **Baseline-Subtracted Negative Flux & Dust Extinction:** Because fluxes are sky-baseline subtracted, stochastic multi-year AGN frequently go negative whereas smooth single-flare TDEs stay positive (`negative_flux_fraction` = #2 feature in `#1`). Raw fluxes require `fitzpatrick99` de-reddening via `EBV`.

---

## 3. Physics-Informed Feature Engineering & 2D GP Interpolation

1. **2D Gaussian Process (`Time × Wavelength`) Interpolation:**
   - Standard PLAsTiCC/Avocado kernel $k(t_1,t_2,\lambda_1,\lambda_2) = k_{\text{time}} \cdot k_{\text{wave}}$ (`Matérn-3/2`) incorporating heteroscedastic **`Flux_err`** into the noise diagonal bridges asynchronous bands and seasonal gaps (`#1`, `#3`, `#4`, `#7`, `#17`), yielding dense uniform grids, cross-band colors (`g-r`, `u-g` at peak/+15d/+20d/+40d/+100d), `gp2d_wave_scale`, `gp2d_log_likelihood`, and `extrabol` blackbody temperature/radius fits (`#3`).
   - *Reproducibility note (`#1`):* `sklearn` `GaussianProcessRegressor` exhibits subtle BLAS/LAPACK optimizer drift across CPUs.
2. **Parametric Template Goodness-of-Fit ($\chi^2$ Residuals):**
   - **`template_chisq_tde`** (`#1`'s #1 feature — $\chi^2$ against an ideal normalized $t^2$ rise / $t^{-5/3}$ TDE decay template, beating noisy free-parameter fits on faint events).
   - **`bazin_global_chi2`** (`#7`'s #1 feature by gain — separates smooth single-flare TDEs from stochastic AGN).
   - **`SALT2 redchisq`** (`#3` top-15 SHAP — rejects Type Ia supernovae).
3. **Thermodynamic Color Stability & SNR Shape Stats:** TDEs maintain hot blue temperatures (`~20k–40k K`) unlike cooling SNe → `Blue Energy Fraction` (`#1`), post-peak `g-r`/`u-g` stability, `time_width_s2n_5` (`#3`'s #1 SHAP), `duty_cycle`, `resampled_skew_{r,g,i}`, and `Flux`/`SNR` FFT power (`#2`).

---

## 4. Models, Small-N F1 Thresholding & Verified Anti-Patterns

- **GBDT Dominance vs. External-Pretrained Sequence NNs:**
  - From-scratch LSTM/Transformers collapsed to **`F1 ~ 0.12`** on only `148` irregular positives (`#4`).
  - `#2` broke the DL barrier by pretraining **`Inception + CBAM + BiGRU + Transformer`** (no positional encoding, ReLU > SiLU, 6 Flux + 6 SNR channels) on external **PLAsTiCC train + PL test** (`OOF 0.66+`), then blending 50/50 with 5-stream GBDTs via **conflict-zone post-processing** (`|P_DL - P_GBDT|` buckets `<0.3`, `0.3–0.6`, `>0.6` → Pub `0.73 → 0.75`).
- **Base-Rate-Aligned Binary F1 Thresholding:** With `scale_pos_weight ≈ 19.6`, winners calibrated thresholds via global OOF PR-curve (`#4`, `#5`, `#7`) or median per-fold threshold (`#3`), keeping test predicted positives (`346–415 / 7,135 = 4.85%–5.82%`) tightly aligned with the `4.86%` base rate.
- **Verified Anti-Patterns:**
  1. **SMOTE / ADASYN & Synthetic Redshift Augmentation:** Consistently hurt across `#1`, `#3`, `#4` (`-2% to -4%` LB), `#7`, and `#17`.
  2. **Aggressive Feature Pruning & Optuna HPO on Small-N (`148` Positives):** Pruning `159 → 80` SHAP features lost **`-0.094` LB** in `#7`; Optuna HPO gave `+0.02` OOF but **worse LB** in `#4`.
  3. **Test-Set Pseudo-Labeling & Blind Multi-Model/Multi-Seed Blending:** Failed in `#1`, `#2`, `#4` (`+0.03` OOF → `-0.04` LB), and `#7` (high fold variance on ~30 positives/fold made simple regularized single models `#3/#4/#7` outperform complex blends).
  4. **Auto-Selecting Top-2 Public LB Submissions:** Both `#3` (`0.6903` unchosen) and `#5` (`0.6804` unchosen) had higher-Private submissions sitting lower on noisy Public LB.

---

## 5. Verified Solution Sources

- **1st Place (`John Titor`):** `https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/writeups/mallorn-1st-place-solution`
- **2nd Place (`Suxing's Astronomer`):** `https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/writeups/mallorn-2nd-place-solution`
- **3rd Place (`Sigrid Nissen`):** `https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/writeups/3rd-place-write-up-catboost-with-threshold-tuning`
- **4th Place (`Alexy`):** `https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/writeups/4th-place-solution`
- **5th Place (`Arnav Salkade`):** `https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/writeups/2nd-place-private-lb-0-6804`
- **7th Place (`Alejandro Ayestaran`):** `https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/discussion/671068`
- **17th Place (`Univ. of Nova Gorica`):** `https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/writeups/mallorn-solution-university-of-nova-gorica`
