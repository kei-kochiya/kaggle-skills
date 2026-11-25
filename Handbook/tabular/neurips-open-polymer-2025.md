# NeurIPS — Open Polymer Prediction 2025: Top 5 Gold Solutions & Post-Mortem

> **Competition**: [NeurIPS - Open Polymer Prediction 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025) (Jun 16 – Sep 16, 2025 | 2,240 Teams | 51,013 Submissions | \$50,000 Prize Pool)  
> **Task**: Multi-target sparse polymer property regression from monomer `SMILES` (`Tg`, `FFV`, `Tc`, `Density`, `Rg`) averaged from Molecular Dynamics (MD) simulations  
> **Evaluation Metric**: Range- and inverse-sqrt-count-weighted MAE (**`wMAE`**): $w_i = \frac{1}{r_i} \cdot \frac{K \sqrt{1/n_i}}{\sum_j \sqrt{1/n_j}}$  
> **Dataset**: Highly imbalanced target availability (merged train + public external: `Tg` `8,244`, `FFV` `7,030`, `Density` `1,247`, `Tc` `866`, `Rg` `614`) | Hidden Test: `~1,500` polymers (`~8%` Public / `~92%` Private; severe `Tg` shift on Private LB!)

---

## Official Top Standings & Solution Index

A major `Tg` (glass transition temperature) distribution shift between training/external sources and the 92% Private test set drove dramatic shakeups (`#2–#4` Public fell -15 to -45 spots, while `#5` surged **+986 ranks** via physics blending):

| Private Rank | Δ Rank | Team / Competitor | Public LB (wMAE ↓) | Private LB (wMAE ↓) | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | 0 | **James Day** (`jsday96`) | **0.05412** (#1) | **0.07536** *(0.089 raw)* | **5-step dirty external label pipeline** (Isotonic rescaling + MAE error filter + Tanimoto `>0.99` leak purge) + **1,116 local RadonPy/LAMMPS MD sims** + **RankUp pairwise pretraining** on 50k PI1M (`ModernBERT-base` + `CodeBERT`, **50× SMILES TTA median**) + `AutoGluon` + `Uni-Mol 2` + `Tg += 0.5644·std` | [Writeup](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/writeups/1st-place-solution) · [GitHub](https://github.com/jday96314/NeurIPS-polymer-prediction) · [NB](https://www.kaggle.com/code/jsday96/polymers-tabular-bert-unimol) |
| 🥈 **#2** | **+19** | **Ezra** (`ezberch`) | 0.05992 (#21) | **0.07722** | Black-box diverse ensemble + **`Tg + 40`** shift calibration (post-comp showed `(9/5)*Tg + 32` Fahrenheit shift scored `0.068` Priv!) | [Writeup](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/writeups/2nd-place-solution) · [NB](https://kaggle.com/code/ezberch/open-polymer-prediction-2025-2nd-place) |
| 🥉 **#3** | **+7** | **Ghy HUST CS** (`fresnellll`, `stickday`) | 0.05737 (#10) | **0.07820** | **6-layer Residual `GATv2Conv` + Top-50 F-reg Morgan bits** + custom `wMAE` loss + `A → A-A-A` chain extension + 2-stage 100%-data refit + **post-hoc linear calibration** (`-5.47%` wMAE) | [Writeup](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/writeups/3rd-place-solution) · [GitHub](https://github.com/fresnellll/kaggle-NeurIPS-polymer-prediction-solution) |
| 🏅 **#4** | **+15** | **Felipe Ferreira** | 0.05972 (#19) | **0.07934** | **LightGBM Quantile Regression (`α = 0.85` on `Tg` & `Rg`, `α = 0.50` on rest)** + radical `*` positions + 3D RDKit descriptors + targeted functional-group substitution aug | [Writeup](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/writeups/4th-place-solution-lightgbm-with-smiles-derived-fe) |
| 🏅 **#5** | **+986** | **Suleyman Bilgin & Serkan Paker** | #991 | **0.07996** | **Hierarchical 2-stage target chaining** (dense `FFV, Tc` → sparse `Tg, Density, Rg`) + **Fox–Flory physics blend** on `Tg` | [Writeup](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/writeups/a-hybrid-physicsmachine-learning-framework-for-pol) |

---

## 1. Core Universal Breakthroughs Across Top 5

### 1.1 The 5-Step Dirty External Label Pipeline (`#1 James Day`)
Because raw `train.csv` had severe target sparsity while external literature datasets (`PI1M`, `LAMALAB`, `RadonPy`, `ChemProps`, etc.) suffered measurement-protocol and unit drift, James Day engineered a strictly leak-free external reconciliation workflow:
1. **Strict Host-Only Validation + Tanimoto `> 0.99` Leak Purge**: Evaluate 5-fold CV strictly on 20% held-out *original host data*; canonicalize all SMILES and **drop any external monomer with Morgan/Tanimoto similarity `> 0.99` against a fold's validation molecules**.
2. **Isotonic Regression Rescaling**: Fit an isotonic regression mapping each external dataset's labels onto a strong host-only ensemble's scale, blending raw and rescaled labels via Optuna per dataset.
3. **MAE Error Filtering**: Discard external samples whose absolute error against the host-only teacher exceeds an Optuna-tuned multiple of host MAE.
4. **Per-Dataset Optuna Sample Weighting** + domain outlier pruning (`RadonPy Tc > 0.402`).
5. **Physics Simulation Feature Stacking**: Ran **1,116 local RadonPy + LAMMPS MD simulations** on PI1M monomers (~600 atoms/chain) and fed **41 XGBoost simulation-output predictors** as tabular features into `AutoGluon` (`+0.0005` CV wMAE).

### 1.2 RankUp Pairwise Pretraining & `50×` Random SMILES Median TTA (`#1`)
- **RankUp Pretraining (`+0.010` CV / `+0.004` LB across 5 foundation backbones)**: Pretrained `ModernBERT-base` and `CodeBERT` on **50,000 unlabeled PI1M polymers** pseudolabeled by a teacher ensemble (`BERT + Uni-Mol + AutoGluon + D-MPNN`) using a **multi-task pairwise ranking classification loss** (ignoring near-tied pairs) before per-target fine-tuning (`AdamW`, head `10×` backbone LR).
- **`50×` Random Non-Canonical SMILES TTA Median (`+0.010` LB alone!)**: Trained with `10×` randomized SMILES (`canonical=False, doRandom=True, isomericSmiles=True`) and aggregated `50×` test-time SMILES enumerations via **median** (Bayes-optimal for MAE!).
- **Single-Target Ensemble Diversity**: Combined `ModernBERT-base` (`0.089` Priv), `CodeBERT` (`0.090` Priv), `AutoGluon best` (2h/target on RDKit + Morgan/MACCS/AtomPair + NetworkX + 107 backbone/sidechain + 41 MD features, `0.091` Priv), and `Uni-Mol 2 84M` on 4/5 targets (`0.091` Priv, excluding `FFV`).

### 1.3 Graph Neural Nets, Linear Calibration & Hierarchical Chaining (`#3`, `#4`, `#5`)
- **Hybrid `GATv2` + Selected Morgan Bits (`#3`)**: Concatenating 6-layer Residual `GATv2Conv` (8 heads, `d=384`) with **Top-50 F-regression-selected Morgan bits** per target + **`A → A-A-A` oligomer chain extension** + Stage-1 epoch selection → Stage-2 100%-data refit.
- **Post-Hoc Linear Calibration (`#3`, `-5.47%` wMAE)**: Fitting `y = a·x + b` per target on OOF predictions corrected GNN regression-to-the-mean shrinkage (`0.080 → 0.078` Priv).
- **Hierarchical Target Chaining (`#5`, `+986` ranks)**: Predict well-populated targets (`FFV`, `Tc`) in Stage 1, pass OOF/test predictions into sparse Stage-2 models (`Tg`, `Density`, `Rg`), and anchor `Tg` with the physical **Fox–Flory equation**.
- **Quantile Loss Under Upward Shift (`#4`)**: Using LightGBM quantile `α = 0.85` for `Tg`/`Rg` vs `α = 0.50` for `Tc`/`Density`/`FFV` implicitly hedged the upward `Tg` test shift.

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Domain-Specific Chemistry BERTs & Larger Backbones** | `ChemBERTa` (`0.0634` CV), `polyBERT` (`0.592`), `ModernBERT-large` (`0.0587`), `DeBERTa-v3-large` all lost to **`ModernBERT-base` (`0.0584`) & `CodeBERT`** | #1 | Code/general tokenizers & base capacity generalize better on sparse randomized SMILES than small overfit chemistry pretrains |
| **Feeding RDKit Descriptors Directly Into `GATv2` / Large Batch Sizes** | RDKit hurt `GATv2` significantly; batch `128/256` worse than `64` | #3 | High-dim tabular descriptors overpower small-batch message passing; Top-50 F-selected Morgan bits worked cleanly |
| **Uni-Mol 2 on `FFV` / Kekulized & Explicit-H SMILES Augs / `D-MPNN` / `AutoGluon extreme`** | OOM (>130 atoms) & poor accuracy on `FFV`; hurt or failed | #1 | Standard randomized canonical-free isomeric SMILES (`doRandom=True`) outperformed chemically altered SMILES strings |
| **Manual Optuna `XGB+LGBM+TabM` vs. `AutoGluon best`** | ~2% worse at 20× compute cost | #1 | On small sparse tabular sets (600–8k rows), AutoGluon multi-layer stacking beats hand-tuned 3-tree blends |
| **Ignoring Target Distribution Shift on `Tg`** | Raw ensemble `0.089` Priv vs `0.075` with `+0.5644·std` (#1); `#2–#4` Public fell -15 to -45 spots | #1, #2, #4, #5 | Systematic test-set `Tg` offset dominated Private LB (`92%` split) |
