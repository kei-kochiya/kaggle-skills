# ISIC 2024 — Skin Cancer Detection with 3D-TBP 🏆

> **Track:** Multimodal (Vision + Real-World Extreme-Imbalance Tabular, Research Code) · **Scale:** 2,739 teams · \$80,000 · Closed Sep 7, 2024  
> **Core Lesson:** Under **0.098% extreme class imbalance** (`393 / 401,059` positives) and **`pAUC >= 80% TPR`**, **patient-relative "Ugly Duckling" context features** (z-scores, ranks, `LOF`/`kNN`/`HBOS` per patient), **multi-task small/mid vision backbones** (`EVA02`, `SwinV2`, `ConvNeXtV2`, `ViT-Tiny`, `TIP` SSL) stacked into **seed/shard-bagged GBDTs** via **identical `patient_id` folds + no early stopping + Gaussian OOF noise (`std=0.1`) + patient-relative image scores** crushed end-to-end multimodal NNs and yielded **+439 to +1,234 Private LB climbs**.

---

## 1. Competition DNA & Shakeup Anatomy

- **Task:** Detect malignant skin lesions from `401,059` cropped 3D Total Body Photography (3D-TBP) tiles (`~133px` mean square crops, `15×15 mm` FOV) + `40+` clinical/color/geometry tabular features across `1,042` patients (`7` hospitals in train; `~500k–525k` test lesions across `1,250–1,500` unseen patients, `30%–40%` from **2 unseen hospitals**!). Only **`393` train positives (`0.098%`)**.
- **Metric (`pAUC >= 80% TPR`, range `[0, 0.20]`):** Partial ROC-AUC above `80%` sensitivity (`roc_auc_score(1 - y, 1 - p, max_fpr=0.2)` rescaled into `[0, 0.20]`). Public LB was only `~28%` of test (very few positives → severe variance).

| Private # | Shake | Team | Private LB | Public LB | CV | Core Breakthroughs |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| 🥇 **1st** | **+30** | **Ilya Novoselskiy** | **0.17264** | 0.18611 (#31) | 0.18195 | Per-patient **`LOF` outlier score** + `EVA02/EdgeNeXt` + 3-cls past-ISIC + **`image_pred/patient_mean`** + **`N(0,0.1)` OOF noise** + 150 GBDTs rank-avg + 10-seed paired t-test |
| 🥈 **2nd** | **+23** | **Yakiniku** | **0.17243** | 0.18635 (#25) | — | 9 vision models (`SwinV2` w/ `iddx_full` TF-IDF cluster loss, **`TIP` SSL tabular-image pretrain**) + 54 GBDTs (varying 0–3 image OOFs) × 5 full-data seeds |
| 🥉 **3rd** | **+764** | **KS** | **0.17229** | — | 0.18310 | 4 tiny/nano vision models (2 standard + 2 **expanded indeterminate positives**) + tab-only LGBM anchor |
| 🏅 **4th** | **+1,137** | **DungNB** *(BiBanhBao)* | **0.17225** | 0.18229 | 0.18703 | 10-model vision ensemble (`0.17458` OOF: 5-cls multi-ISIC + 3 **pseudo-seg FPN/UNet** + 3 ISIC-24) → GBDT Gmean (`0.1788→0.1870` CV) + `0.2` tab-only blend (also won Top-15 Retrieval) |
| 🏅 **5th** | **+1,100** | **Kanna Hashimoto fr. 2** | **0.17210** | ~0.183 | 0.1839 | Dual-track **with vs. without `attribution`** routing for unseen hospitals + `0.5` indeterminate soft labels |
| 🏅 **6th** | **+1,151** | **xck** | **0.17192** | 0.158–0.165 | 0.168 | 9-class `iddx_3` CNN (`EB3@192`: `0.158` Pub → `0.173` Priv!) + top-70k GBDT candidate re-scoring |
| 🏅 **7th** | **+439** | **Former ZLP-DSs** | **0.17163** | — | — | **5→7 hospital-holdout CV** + multi-head `target + lesion_id` CNN + 3-tier GBDT sample weights |
| 🏅 **8th** | **+1,234** | **Ujjwal Pandey** | **0.17158** | 0.180 | 0.1790 | `EffNet-B1` (`WeightedRandomSampler 1:0.01`, no early stop) → CatBoost (`0.160→0.171` Priv) + `SAGE=0` feature pruning |
| 🏅 **10th** | **+1,174** | **tomo20180402** | **0.17146** | 0.181 | 0.183 | Missing-`attribution` injection during train + 20 rotated negative shards + **1,500-GBDT bag** |
| *24* | **-23** | *nah, I'd lose* | *0.17054* | **0.18926 (#1)** | — | Unshuffled SGKF + CatBoost `use_best_model=True` + Public-LB Top-K feature tuning |

---

## 2. EDA & The "Ugly Duckling" Patient-Relative Breakthrough

1. **Patient-Level & Site-Level Relative Features (Biggest Tabular Signal):**
   - Benign lesions on the same patient look similar; malignant lesions stand out as intra-patient outliers ("Ugly Duckling Sign").
   - Compute `(x - mean) / std`, `x - mean`, `x / max`, ranks/ECDFs, and counts grouped by `patient_id`, `(patient_id, anatom_site_general / tbp_lv_location)`, and `(patient_id, tbp_tile_type)`.
   - **Unsupervised Per-Patient Outlier Scores:** **Local Outlier Factor (`LOF`)** over top CatBoost features (`#1`: CV `0.18149 → 0.18185`), per-patient **kNN (`k=5`)** neighbor diffs (`#12`), and **HBOS (`pyod`)** pruned via **KS-statistic** (`#15`: `+0.001` LB).
2. **Unseen Hospitals (`attribution`, 30–40% of Test) & Train-Only Columns:**
   - Columns `lesion_id`, `iddx_full`, `iddx_1..5`, `mel_thick_mm`, `tbp_lv_dnn_lesion_confidence` exist **only in train** — great for **auxiliary training targets**, never as test inputs!
   - Handle unseen hospitals via **dual-track routing** (with vs. without `attribution`, `#5`), **injecting `NaN` `attribution` copies during training** and mapping unseen test hospitals to `NaN` (`#10`), or **5-hospital → 7-hospital holdout CV** (`#7`).

---

## 3. Multimodal Vision + Tabular Fusion (Why 2-Stage Stacking Won)

1. **Small/Mid Multi-Task Vision Backbones (`~133px` crops):**
   - Compact backbones (`eva02_small`, `edgenext_base`, `swinv2_small`, `convnextv2_nano/tiny`, `vit_tiny@224–384`, `EffNet-B1/B3`) matched or beat giant ViTs and fit 500k-image inference budgets.
   - **Auxiliary Targets Unlocked Sparse 393-Positive Gradients:**
     - `iddx_full` TF-IDF + k-means centroid distance loss (`0.1612` CV) & **TIP self-supervised tabular-image pretraining** (`0.1607` CV, `#2`).
     - **Expanded positives** (`target=1` OR `iddx_1=='Indeterminate'`, `#3`), `target + lesion_id` presence (`#7`), `9-class iddx_3` (`#6`), and **5-class multi-ISIC + pseudo-mask segmentation `FPN/UNet`** (`#4`: 10-CNN ensemble hit `0.17458` OOF!).
   - **Imbalance Sampling:** `1:1` per-batch balanced sampling with fresh negative subsamples each epoch (`#1`, `#3`, `#5`), `WeightedRandomSampler(1.0, 0.01)` (`#8`), or `20` rotated negative shards (`#10`).
2. **2-Stage OOF Vision → Patient-Relative GBDT Stacking Crushed End-to-End Multimodal NNs:**
   - End-to-end multimodal models (`#7`, `#8` `EffNet+Meta 0.154` Priv, `#12` `FTTransformer 0.165` Priv) cannot see cross-lesion patient distributions at once and are slow over 500k images.
   - Stacking standardized vision OOFs into GBDTs — **plus computing patient-relative image features (`image_pred / patient_mean(image_pred)`, `#1`)** — lifted `#4` **`0.1788 → 0.1870` CV** and `#8` **`0.160 → 0.171` Private**!
   - **Leak-Proof Stacking Protocol:** (a) Strictly **identical `patient_id` folds** across Stage-1 CNNs and Stage-2 GBDTs; (b) **NO early stopping on val fold** (fixed epochs + EMA/SWA, `#2/#4/#5/#8/#10`); (c) add **Gaussian noise (`std=0.1`) to OOF image features** during GBDT training (`#1`, `#12`); (d) blend `~20%` tabular-only GBDTs (`#3`, `#4`) as regularization.

---

## 4. Validation, Extreme-Imbalance Bagging & Anti-Patterns

- **Validation:** **Triple-Stratified GroupKFold** (`patient_id` + malignancy rate + lesion-count bin, `#2/#6/#12/#14/#15`), **5→7 hospital-holdout CV** (`#7`), and **10-seed paired t-test** (`#1`) to filter noise across only `393` positives.
- **Seed/Shard Bagging & Pruning:** `150`–`1,500` GBDTs bagged across negative shards & seeds (`#1`, `#10`), aggressive **permutation-importance / `SAGE=0` pruning** (`#13` dropped ~190 noisy features: `0.179 → 0.183` CV; `#8`), blended via rank averaging (`#1`), geometric mean (`#4`), or hill-climbing (`#9`).
- **Verified Anti-Patterns:**
  1. **Image-OOF Target Leakage (`0.19+` Fake CV):** Mismatched folds or early-stopped vision OOFs fed directly to GBDTs (`#5`, `#8`, `#13`, `#15`).
  2. **External Dermoscopy Domain Gap (`AUC = 0.99`):** `#2` showed a classifier separated ISIC 2018 dermoscopy from 3D-TBP at `AUC = 0.99`; naive mixing failed unless carefully multi-tasked or down-weighted (`0.01`) and stripped of `clinical: overview` images (`#1`, `#2`, `#4`, `#11`).
  3. **Hard Negative Mining, Focal Loss, pAUC Surrogate Loss, Hair-Removal Filters, Synthetic Diffusion Positives, & 2nd-Stage Meta-Stackers:** All failed or lost to standard BCE/WeightedSampler + simple voting (`#1`, `#3`, `#7`, `#9`).
  4. **Albumentations `Resize` Placement Bug (`#9`):** Resizing *before* blur/dropout instead of right before `Normalize` cost `-0.007` CNN CV (`0.148 → 0.155`).

---

## 5. Verified Solution Sources

- **1st Place (`Ilya Novoselskiy`):** `https://www.kaggle.com/competitions/isic-2024-challenge/writeups/ilya-novoselskiy-1st-place-solution`
- **2nd Place (`Yakiniku`):** `https://www.kaggle.com/competitions/isic-2024-challenge/writeups/yakiniku-2nd-place-solution`
- **3rd Place (`KS`):** `https://www.kaggle.com/competitions/isic-2024-challenge/writeups/ks-3rd-place-solution`
- **4th Place (`DungNB` / `BiBanhBao`):** `https://www.kaggle.com/competitions/isic-2024-challenge/writeups/bibanhbao-4th-place-solution`
- **5th Place (`Kanna Hashimoto friends 2`):** `https://www.kaggle.com/competitions/isic-2024-challenge/writeups/kanna-hashimoto-friends-2-5th-place-solution`
- **6th Place (`xck`):** `https://www.kaggle.com/competitions/isic-2024-challenge/writeups/xck-6th-palce-solution`
- **7th Place (`Former ZLP-DSs`):** `https://www.kaggle.com/competitions/isic-2024-challenge/writeups/former-zlp-dss-7th-place-solution-a-good-cv-is-all`
- **8th Place (`Ujjwal Pandey`):** `https://www.kaggle.com/competitions/isic-2024-challenge/writeups/ujjwal-pandey-8th-place-solution-trust-your-cv`
- **9th Place (`Sinan Calisir`):** `https://www.kaggle.com/competitions/isic-2024-challenge/writeups/sinan-calisir-9th-place-solution`
- **10th Place (`tomo20180402`):** `https://www.kaggle.com/competitions/isic-2024-challenge/writeups/tomo20180402-10th-place-solution`
