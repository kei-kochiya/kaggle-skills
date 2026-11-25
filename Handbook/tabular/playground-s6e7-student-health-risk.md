# Playground Series S6E7: Predicting Student Health Risk — Top Solutions & Post-Mortem

> **Competition**: [Kaggle Playground Series Season 6 Episode 7](https://www.kaggle.com/competitions/playground-series-s6e7) (Jul 1 – Aug 1, 2026 | 3,355 Teams | 34,190 Submissions)  
> **Target**: 3-Class Imbalanced `health_condition` (`at-risk` `85.87%`, `unhealthy` `8.36%`, `fit` `5.77%`)  
> **Evaluation Metric**: **Balanced Accuracy** (`balanced_accuracy_score`)  
> **Dataset**: `690,088` Train Rows × `15` Cols (`7` numeric + `6` categorical, `1%–12%` NaNs) | `295,753` Test Rows × `14` Cols | Original Seed: `50,000` Rows ([College Student Health Behavior Dataset](https://www.kaggle.com/datasets/ziya07/college-student-health-behavior-dataset))  
> **LB Split**: ~20% Public (`~59.2k` rows, only `~3.4k` `fit` rows) / ~80% Private (`~236.6k` rows) — **total Public Top-40 wipeout** (Spearman $\rho = -0.414$ inside top Public band; **0 of Public Top 40** survived into Private Top 40!)

---

## Official Top Standings & Solution Index

Every single team in the Private Top 10 jumped **+55 to +449 ranks**, while Public #1–#47 (`0.95307–0.95340`) collapsed to `~0.950X` on the 80% Private split:

| Private Rank | Δ Rank | Team / Competitor | CV (OOF BA) | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| 🥈 **#2** | **+358** | **`kava1`** | **0.950737** *(LogLoss 0.08631)* | ~#360 | **0.95085** | **18 calibrated natural-prior models** → **SLSQP per-class simplex blend minimizing OOF LogLoss** (`FT-Transformer-v2` 50–72% wt + `XGB OvR` 28–46%) → Nelder-Mead multipliers (`+0.0616` over argmax) | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/2nd-place-solution) · [LGBM Stack NB](https://www.kaggle.com/code/kava1/s6e7-overcoming-0-95-tuned-lightgbm-stacking) |
| 🏅 **#4** | **+410** | **`Ricky`** (`rickyds97`) | **0.950633** | 0.95094 *(#414)* | **0.95084** | **Single model family (zero stacking!)**: 7-fold × 4-member **FT-Transformer-v2** (`PLR-lite` 24 freqs, fixed 16 ep, **no early stopping**) + **39 fold-local exact-value TEs** + exact prior correction `argmax(p / π_c)` | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/4th-place-from-414-to-4-trusting-oof-when-the) · [FTT-v2 Base](https://www.kaggle.com/code/masayakawamata/s6e7-ft-transformer-v2-cv-0-95063) |
| 🏅 **#11** | **+388** | **Ravi Ramakrishnan** (`ravi20076`) | **0.95069** | 0.95097 *(#399)* | **0.95081** | 5-fold diverse zoo (XGB, LGBM, CB, RepLeaf, RealMLP, **FT-Transformer**, **FM**, **DANet**) + private PyTorch GPU LogReg stacker; native NaNs untouched | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/rank11-approach) |
| 🏅 **#29** | **+643** | **`Hawksartelier`** (`satyapb2002`) | **0.95067** | 0.95034 *(#672)* | **0.95070** | Clean `n_ens=1` FT-Transformer-v2 + exact-value TE + prior correction; **32 falsified levers** (proved `TabPFNv2` had `0.914` error corr → `0` stack weight) | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/29th-place-ft-transformer-exact-value-target-en) · [Notebook](https://www.kaggle.com/code/satyapb2002/29th-place-ftt-exact-value-target-encoding) |
| 🏅 **#36** | **+459** | **Baseer Shah** (`baseershah`) | **0.950868** | 0.95056 *(#495)* | **0.95067** | 58 base models (`36` GBDTs + `22` NNs; `RealMLP-v9` `0.95046` best solo) → 174 logits clipped `[-30,30]` → **25-seed GPU CatBoost stacker** (`l2_leaf_reg=12.26`, `class_weights='Balanced'`) | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/pss6e7-cv-blending) |
| 🏅 **#55** | — | **`Jerry`** (`jerry34`) | — | — | — | **Optimal Brain Surgery (OBS)** pruning on weighted log-loss over 200+ models: LR/MLP needed only `~25` diverse models vs. trees using nearly all | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/s6e7-solution) |
| 🏅 **#92** | **+284** | **Reda Mountassir & Ayman Arib** | 0.95059 | #376 | — | Mathematical proof of **non-monotonic quantized exact-value signal** via split-half residual correlation ($r=0.90$ on `sleep_duration`); GBDT pairwise error $\phi > 0.93$ | [Writeup](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/376th-92nd-the-signal-was-non-monotonic-and-on) · [Notebook](https://www.kaggle.com/code/aribaymane61/ps-s6e7-ft-transformer-plr-embeddings-0-95059) |

*(Private #1 `Shantanu Chopra` `0.95085` [+449], #3 `Biel Escola Rodrigo` `0.95084` [+354], and #5–#10 did not publish writeups. Additional public GitHub research trail: [`msusol/kaggle-playground-series-s6e7`](https://github.com/msusol/kaggle-playground-series-s6e7).)*

---

## 1. Data Forensics: Quantized Non-Monotonic Signal & Missingness Traps

### 1.1 Deterministic Depth-4 Original Rule vs. Synthetic Noise Ceiling
In the 50k original seed (`student_health_dataset_50k.csv`), **only 3 columns** (`sleep_duration`, `stress_level`, `physical_activity_level`) separate `health_condition` with **100% accuracy** (`broccoli beef`, `/discussion/717222`):
- `sleep_duration <= 5.99`: `stress_level == 'high' → unhealthy`, else `at-risk`.
- `sleep_duration > 5.99`: `stress_level == 'low' AND physical_activity_level == 'active' AND sleep_duration > 6.99 → fit`, else `at-risk`.
- Kaggle's synthetic generator injected feature/label noise plus **1%–12% missingness** (highest on `sleep_duration` and `stress_level`, which was uniform random and thus unimputable), pinning the theoretical noise ceiling around **`~0.9508–0.9509`**. Appending the 50k original dataset at weight `1.0` **hurt OOF BA by `-0.000473`**.

### 1.2 Split-Half Non-Monotonic Exact-Value Forensics (`#92`, `#4`)
The synthetic generator resampled numeric columns onto a **finite quantized support** (`sleep_duration` had only **702 distinct values** across 690k rows), and conditional target probabilities oscillated **non-monotonically** from value to value:

| Numeric Feature | Best Monotone Fit ($R^2$) | Split-Half Residual Corr ($r$) | Signal Geometry |
| :--- | :---: | :---: | :--- |
| `sleep_duration` | 0.73 | **0.90** | Strong non-monotonic exact-value quantization |
| `water_intake` | 0.02 | **0.72** | Pure non-monotonic exact-value signal |
| `heart_rate` | 0.15 | **0.59** | Strong non-monotonic exact-value signal |
| `bmi` / `exercise_duration` / `step_count` / `calorie_expenditure` | 0.86–0.98 | `-0.10` to `+0.03` | Smooth monotonic (no exact-value spikes) |

- **Why Default GBDTs Lost**: Standard `max_bin=255` merges adjacent quantized values, wiping out the exact-value spikes (all 4 GBDT libraries shared pairwise error correlation $\phi > 0.93$, misclassifying 38,543 identical rows simultaneously!). Raising `max_bin=768` in XGBoost (`+0.00046`), adding **fold-local nested exact-value Target Encoding** on all 13 columns (`XGB: 0.94890 → 0.94956`), or using **FT-Transformer with PLR-lite periodic embeddings** unlocked the full ceiling.
- **Missingness Forensics**: Chi-square tests showed `bmi` missingness correlated strongly with the target (`p = 2.9e-144`, Cramér's `V = 0.031`), while `sleep_duration` / `stress_level` missingness was MCAR (`p > 0.85`). **Pre-imputing NaNs with median/mode crashed GBDTs from `0.95002 → 0.90300` LB**; leaving NaNs native for trees was strictly mandatory.

---

## 2. Core Winning Recipes

### 2.1 LogLoss-Optimal Simplex Blending + Exact Prior Correction (`#2 kava1`, `#4 Ricky`)
1. **Train Calibrated Natural-Prior Models (No Resampling / No Class Weights)**: Keeping natural training priors yields well-calibrated posterior probabilities $\hat{p}(y=c \mid x)$ with clean log-loss gradients.
2. **Simplex-Constrained Per-Class LogLoss Blending (`#2 kava1`)**: Optimize an $M \times 3$ weight matrix via `SLSQP` minimizing **OOF LogLoss** subject to $\sum_m w_{m,c} = 1, w_{m,c} \ge 0$ (reducing OOF LogLoss `0.105495 → 0.086310`). Redundant models automatically dropped to `0` weight; **`FT-Transformer-v2` captured 50%–72% weight across all 3 classes** and **`XGB OvR` captured 28%–46% on classes 1 & 2**.
3. **Zero-DOF Theoretical Prior Correction vs. Nelder-Mead**:
   - **Exact Bayes Rule (`#4 Ricky`, `#29`)**: `argmax_c (p_c / π_c)` jumped OOF BA by **`+0.05876`** (`0.89187 → 0.95063`, shifting `6.14%` of test rows) with **zero tunable parameters**, surging **+410 places** to **#4 Private (`0.95084`)** using a **single model family (FT-Transformer-v2, 7 folds × 4 seeds, fixed 16 epochs, NO early stopping)**.
   - **Nelder-Mead Class Multipliers (`#2 kava1`)**: `[0.124044, 1.499826, 1.250181]` on full 690k OOF BA jumped `+0.06161` (`0.88913 → 0.95074`) to **#2 Private (`0.95085`)**—close to inverse priors `[1/0.8587, 1/0.0836, 1/0.0577]` up to scale.

---

## 3. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **The Double-Correction Trap (`class_weight='balanced'` + `p / prior`)** | Collapsed OOF BA by **`-0.039` to `-0.060`** (`0.950 → 0.9047`) | Community Consensus | Reweights minority classes twice; choose **either** balanced loss (`argmax`) **or** natural unweighted loss + prior division |
| **Median/Mode Pre-Imputation of NaNs for GBDTs** | Crashed LB **`0.95002 → 0.90300`** | `naiQie`, #11 | Destroys informative `bmi` missingness (`p = 2.9e-144`) and forces arbitrary splits on MCAR `sleep_duration` NaNs |
| **Screening Exact-Value TE on a 70k Subsample** | False negative **`-0.0017`** (vs `+0.0012` at full 690k) | #4 (`Ricky`) | High-cardinality exact-value counts become sparse under subsampling; always screen exact TE at full fold size |
| **Public LB Chasing / 29-Row Public Probing** | **0 of Public Top 40** survived into Private Top 40 ($\rho = -0.414$) | LB Forensics, `yw8837` | Only `~3.4k` `fit` and `~5.0k` `unhealthy` rows in 20% Public split caused `±0.001–0.002` noise |
| **`TabPFN v2` & `TabICL` Context Subsampling** | `TabPFN` had **`0.914` error corr** → `0` stack weight (#29); `TabICL` `0.94809` (#92) | #29, #92 | Context subsampling on 690k rows loses exact-value repetition frequencies |
| **Appending Original 50k Seed / Explicit Rule FE / Missing Flags / Pseudo-Labeling** | Original `-0.000473`; Rule FE `-0.00002` to `-0.00029`; Missing flags `-0.000175`; PL `+0.000025` noise | #4, #29, Discussions | Original has 0% NaNs (distribution mismatch); exact-value TE + PLR already absorb rule boundaries cleanly |
