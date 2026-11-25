# Biohub — Cell Tracking During Development: Top 7 Gold Solutions & Post-Mortem

> **Competition**: [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) (Jun 29 – Sep 30, 2026 | 3,947 Teams | 95,438 Submissions | \$60,000 Prize Pool)  
> **Task**: 3D+Time zebrafish embryo cell detection, temporal linking, and mitosis/division reconstruction from Zarr v3 `(T, Z, Y, X) ≈ (100, 64, 256, 256)` microscopy volumes (`4:1` Z:XY anisotropy: `1.625 × 0.406 × 0.406 µm`)  
> **Evaluation Metric**: `score = adjusted_edge_jaccard + 0.1 * division_jaccard` (max match distance `7.0 µm`, node over-prediction penalty against `estimated_number_of_nodes`)  
> **Dataset**: `87.6 GB` — `199` Train movies from **2 embryos** (`44b6` dense, `6bba` sparse; **only ~3% of cells annotated**, `151` total annotated divisions!) | Disjoint Test embryos: `fdad` (`60` videos, `29%` Public) & `ea36` (`106` videos, `71%` Private) | `<= 12h` notebook

---

## Official Top 7 Prize-Winning Standings & Solution Index

| Private Rank | Δ Rank | Team / Competitor | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **+2** | **Sergio Alvarez** (`sersasj`) | 0.97635 (#3) | **0.97759** | 3D MAE-pretrained **Cellpose-flow 3D U-Net** (aniso `(1,2,2)` pool) + **"Soon Net"** 3-frame temporal occupancy tracker (`515` verified divisions) + LightGlue-style Transformer linker w/ **factored `kind × who` 137-way fork head** + greedy log-odds decode | [Writeup](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/1st-place-solution) |
| 🥈 **#2** | **+26** | **Soheil Ayati** | 0.96842 (#28) | **0.97041** | Isotropic `1×4×4` (`64³`) 3-frame/5-frame Res-3D U-Net + synthetic faint-cell aug + `acq_00/acq_01` contrast stratification + LGBM dimness candidate specialists + **`acq_01`-only faint-track recovery** | [Writeup](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/2nd-place-solution) · [Notebook](https://www.kaggle.com/code/soheilayati/biohub-2nd-place-notebook) |
| 🥉 **#3** | -2 | **`yu4u`** (`ren4yu`) | **0.97898 (#1)** | **0.96714** | 6-stage TensorRT pipeline: **2.5D EffNet-B7/V2-L + 3D SegResNet** (DoG exclusion mask) → dense 3D flow → cell matcher → 5-vol flow-aligned division CNNs → **`P(eval)` × `P(correct)` calibrated HiGHS LP→MILP** → affine coordinate refinement | [Writeup](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/3rd-place-solution) |
| 🏅 **#4** | **+16** | **Barry** (`SongqiZhou`) | 0.96987 (#20) | **0.96234** *(0.964 unsel.)* | **Dual-encoder 3D U-Net** (`frame t` + `(t+1)-(t-1)`, 3-tier norm BCE + Cellpose flows, `+102` Ultrack PL `+0.011` Priv) + DoG density router + **SCIP ILP with per-node division cost `6 - 16·s` (`+0.035`!)** + CatBoost fork verifier | [Writeup](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/4th-place-solution) · [GitHub](https://github.com/SongqiZhou/biohub-cell-tracking-4th-place) |
| 🏅 **#5** | **+3** | **Tang** (`hirotetsu`) | 0.97167 (#8) | **0.95462** | `1×4×4` 3D U-Net + **ZebraHub-pretrained Transformer Linker** (`+0.016` Priv, `462` verified divisions, 8× XY TTA) → **4-stage progressive ILP (`ILP1–ILP4`)** + line-fit smoothing | [Writeup](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/5th-place-3d-u-net-transformer-linker-multi-s) · [Notebook](https://www.kaggle.com/code/hirotetsu/ctdd-11model-final-v2-ilp1-rematch-adj-off) |
| 🏅 **#6** | **+88** | **Cyrus** (`jamalsaeedi`) | 0.96185 (#94) | **0.95398** | **Unified joint detect-and-link** (`IsotropicLineageNet` + `MultiScaleLineageNet`, cached per-frame encoders, 3-state loss + `N_est` count prior + low-contrast haze aug) + 5M-cell offline noisy-student + **LP-first HiGHS ILP (`38–166×` faster than SCIP)** + deadline guard | [Writeup](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/6th-place-solution) · [GitHub](https://github.com/jamal-saeedi/biohub_cell_tracking_kaggle) |
| 🏅 **#7** | 0 | **`tatsutaka`** | 0.97195 (#7) | **0.95273** | Single **3D nnU-Net `ResEncM`** (`t-d, t, t+d` + interval `d`, 8-view XY logit TTA, **untransformed motion**) + 14-ch 3D CNN-BiGRU division net + 2-stage ILP + freeze-interval snapping | [Writeup](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/7th-place-solution) |

---

## 1. Universal 3D+Time Detection, Sparse Supervision & Tracking Breakthroughs

### 1.1 Learning From `3%` Sparse Annotations Without Negative Collapse
Because `~97%` of cells in training frames were unlabeled, naive binary segmentation treated real unlabeled cells as false negatives and collapsed completely. All 7 winners used explicit **3-state sparse masking**:
- **Positive (`<= 2–3 µm` Gaussian / Cellpose flow)**, **Verified Background** (below local threshold/median and `> 6 µm` from GT — or **DoG peak exclusion zones**, #3), and **Unknown (loss weight `0`)**, with positive & background loss normalized by separate voxel counts (#2, #3, #4).
- **Count-Prior Regularization (`#6`)**: Penalize predicted total mass against `estimated_number_of_nodes` so unknown regions don't over-fire.
- **Pipeline-Verified Pseudo-Labeling**: Filter pseudo-labels through a working tracker (`min track length > 3`, #1; `>=50%` overlap with base detector + external Ultrack windows for **`+0.011` Priv**, #4; 5M ILP-filtered cells with GT authoritative for `+0.008` val, #6).

### 1.2 Anisotropy (`4:1` Z:XY) & 3-Frame Temporal Context
- **Anisotropy Handling**: Never apply out-of-plane 3D/Z rotations or Z-flips (`#3`, `#6` — use lateral **XY D4 symmetry only**!). Standardize anisotropy via either early lateral-only `(1,2,2)` pooling (#1, #6), isotropic `1×4×4` pooling (`64³`, #2, #5), or 2.5D slice-stack encoders (`EffNet-B7/V2-L`) + 3D decoder (#3).
- **3-Frame Temporal Context Is Non-Negotiable (`#2`, `#4`, `#6`, `#7`)**: In `#4`, removing `(t+1) - (t-1)` from the dual-encoder 3D U-Net crashed cross-embryo detection **`0.93 → 0.73`**! Cache per-frame encoder features across adjacent windows (`1.4–1.9×` speedup, #3 & #6).
- **Critical Motion-TTA Anti-Pattern (`#7`)**: Average **heatmap logits** across 8-view XY TTA, but predict **3D motion vectors strictly on untransformed inputs** (`0.91931` untransformed vs `0.91619` vector-averaged vs `0.90218` voting!).

### 1.3 Mitosis/Division Decoupling & Global Conservation ILP (`+0.024` to `+0.035`!)
Divisions (`0.1 × div_jaccard`, only 151 GT instances) delivered the single largest leap across top solutions:
1. **Factorize `P(kind) × P(who | kind)` (`#1`)**: A flat 137-way softmax (`1` none + `16` single + `120` daughter pairs) drowned rare divisions under 120 noisy pair logits; factoring into *dividing vs single vs none* × *which child(ren)* solved false forks cleanly.
2. **Flow-Aligned Temporal Difference Windows (`#3`, `#7`)**: Warping `t-1` and `t+1` onto frame `t` via predicted 3D flow before classifying divisions decoupled chromatin splitting from bulk embryo drift.
3. **Human-Verified Mined Divisions (`#1`, `#4`, `#5`)**: Visually verifying top predicted forks expanded division supervision `3×–4×` (`151 → 515` in #1, `+409` forks in #4, `+462` in #5).
4. **LP-First HiGHS Conservation MILP (`#3`, `#6`)**: Enforcing flow conservation (`<=1` parent, `<=1` event, daughter refractory rule, drift-corrected edge costs, per-node division cost `6 - 16·s` worth **`+0.035`** in `#4`) via **HiGHS LP → MILP over fractional variables ran `38–166×` faster than SCIP** (#6) and scaled cleanly within 12h on 2× T4 with `fp16` (3.5× faster than emulated `bf16` on T4) + graded deadline guards.
5. **Universal Track Post-Processing**: Motion/drift-compensated `±2..4`-frame straight-line coordinate smoothing (`#1`, `#3`–`#7`), short `1–3`-frame gap bridging (`#1`, `#3`–`#5`, `#7`), and pruning tiny unattached `<3–9`-frame fragments (`#1`, `#3`, `#7`).

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Hard-Negative Mining on Sparse Annotations** | Degraded division precision | #4 | Hardest "negatives" in 3%-labeled data are real unannotated divisions! |
| **Synthetic Pretraining / Over-Strong Synthetic Detector** | Raised Public (`0.960 → 0.966`, #7; `+0.0013` CV, #2) but **dropped Private (`0.956 → 0.947`, #7; `0.957 → 0.950` Pub, #2)** | #2, #7 | Synthetic texture mismatch hurts unseen sparse test embryos |
| **Averaging 3D Motion Vectors Across TTA** | `0.9193 → 0.9162 / 0.9022` | #7 | Ambiguous symmetries corrupt vector averaging; use TTA only on scalar heatmaps |
| **Full `256×256` 3D U-Net / 5-Frame Context** | `4.4×` compute with zero or negative gain (`-0.006` Pub, #5) | #4, #5 | 4:1 Z:XY anisotropy means `64–128` XY already matches Z resolution |
| **Pruning Candidate Edges by Probability / Aggressive Track Pruning / DoG-Only Veto** | `0.900 → 0.867` (#6); `-0.011` Priv DoG gate (#4); dropped Pub (#2) | #2, #4, #6 | Starves global ILP of feasible assignments on dim frames |
| **Using Emulated `bf16` on NVIDIA T4** | `3.5×` slower than `fp16` | #6 | T4 hardware lacks native `bf16` tensor cores |
