# RSNA Knee Abnormality Detection — Top 7 Gold Solutions & Post-Mortem

> **Competition**: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) (Jul 30 – Oct 23, 2026 | 6,299 Teams | 156,448 Submissions | \$77,000 Prize Pool + Efficiency Track)  
> **Task**: Multi-label detection of **12 knee MRI abnormalities** (`ACL`, `MCL`, `Medial/Lateral Meniscus`, `Medial/Lateral/PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`) across multi-plane DICOM series (`Sagittal/Coronal/Axial × Fluid/Fat-Sat`)  
> **Evaluation Metric**: Macro-averaged ROC-AUC across 12 targets (`<= 9h` 2×T4 notebook, internet disabled)  
> **Dataset**: `569.76 GB` (`819,640` files) — `4,407` Train studies (**only `58` studies "Gold58" have expert 12-label ground truth**; `4,349` have multilingual free-text `Report` only, **absent at test time!**) | `~1,300` Hidden Test studies (`30%` Public / `70%` Private)

---

## Official Top 7 Gold Leaderboard Standings & Solution Index

With only **58/4,407** expert-labeled training studies, the competition became a masterclass in **frontier VLM image+report continuous soft-label distillation (+0.013 to +0.027 AUC)**, **physical-FOV 2D/2.5D Attention MIL**, **anatomical ROI cropping**, and **hospital/scanner-shortcut debiasing**:

| Private Rank | Δ Rank | Team / Competitor | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **+3** | **Team Kangaroo🦘** (`tascj0`, `liushuzhi`/`sakaku`) | 0.969 (#4) | **0.960** | Frontier VLM agentic viewing (`GPT-6 Astra`/`Sol`/`Hulu`/`Qwen`) → continuous `Mix5` soft probs + **3D ROI boxes (`751` studies) & 111k 2D lesion boxes** → 50% `tascj0` 2D box-pooling (`ConvNeXt-T`/`MambaOut`/`MaxViT-V2` + `Qwen3-8B` report-emb aux) + 50% `liushuzhi` 2.5D anatomy-attention (`MaxViT-V2`/`CoAtNet`); **also #1 Efficiency Track (`0.954` Priv in 230s via TRT+C++)** | [Writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/1st-place-solution) · [GitHub](https://github.com/ShuzhiLiu/RSNA-2026-Knee-MRI-Abnormality-Detection-1st-place-Team-Kangaroo-) · [Infer NB](https://www.kaggle.com/code/tascj0/knee-final-20261022) |
| 🥈 **#2** | **+3** | **MURAD** (`zzzoll`, `azamat1ch`) | 0.969 (#5) | **0.960** | Native-language report read (`0.899`) → `GPT-6 Astra` all-slice read (`0.942` → **`0.956`** Gold58 after PF/Lat OA fixes) → 4 core **2.5D MIL** (`ConvNeXt-T` fastMRI-pretrained `0.957` solo + `EffNetV2-S`) + 2 specialists (`Lat`, `PF` + OAI MOAKS aux) → **fixed per-finding rank blend** | [Writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/2nd-place-gold-solution-vision-llm-soft-labels-an) · [GitHub](https://github.com/azamat1ch/rsna-knee-2nd-place) |
| 🥉 **#3** | **+6** | **DataTech Club** (`Zhengru Li` et al.) | 0.968 (#9) | **0.959** | `GPT-6 Astra` soft labels + joint **~9,440 OAI** knees (`2k/ep` 70:30, deweighted teacher + OAI prevalence logit offset) → **36 checkpoints across 4 families** (`BiT ResNetV2-50`, `ConvNeXt-S`, `DINOv3 ViT-B/16`, `CoAtNet`) w/ within-series depth attention + ordinal OA head → soft group mix | [Writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/3rd-place-solution-rsna-knee-abnormality-detectio) · [GitHub](https://github.com/lizhihong718-alt/rsna-knee-2026-3rd-place) |
| 🏅 **#4** | **+9** | **Ilya Kolb** (`ilyakolb`) | 0.967 (#13) | **0.958** *(0.959 rerun)* | **ChatGPT Pro** rendered PNG packet reader (`0.950` Gold58; lifted same `CoAtNet-2` `0.925 → 0.952` Priv!) → `CoAtNet-2` (`K=24` bag sampling, `336→384px`, 48-target report aux) + geometry crop specialists + `H6` (`ConvNeXt-S` SKM-TEA/OAI) | [Writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/4th-place-llm-image-labels-attention-mil) · [GitHub](https://github.com/ilyakolbafk/rsna-knee-abnormality-detection-4th-place) |
| 🏅 **#5** | -2 | **RB & Matheus & Tucker & Tony** | 0.969 (#3) | **0.958** | **20-hospital site-clustered 5-fold CV** (caught `+0.019` site leak!) + `Opus 5.5` + `gpt-6.1-sol` + **`nnU-Net` bone segmentation** → 4 regional crops (`+0.003` Priv) → OAI-pretrained `ConvNeXt-Base` (**`0.958` Priv solo!**) + `EffNetV2-M` + specialists | [Writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/5th-place-solution) · [Infer NB](https://www.kaggle.com/code/rashmibanthia/rsna-knee-5th-place-solution-inference/notebook) |
| 🏅 **#6** | -4 | **CentralyDS** (`centralyds`) | 0.969 (#2) | **0.958** | **Within-scanner×field-strength proxy AUC** (Spearman `0.94` vs LB!) + 5-keypoint 3D knee frame → **70mm (`224px`) 3-slice tiles** (`ConvNeXt-T`/`CAFormer`/`ReXNet` + 1 Transformer + MIL) + 18.5k OAI/fastMRI 2-round noisy-student + **attention sharpening (`τ=12,32`)** + octant TTA | [Writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/6th-place-solution-knee-frame-tiles-pseudo-label) · [GitHub](https://github.com/adolic/rsna-knee-solution) |
| 🏅 **#7** | -6 | **Barry** (`SongqiZhou`) | **0.970 (#1)** | **0.957** *(0.958 solo)* | **Zero external data!** 3-pass `GPT-6 Astra` + **per-scanner Synovitis logit alignment** + 3-family OOF teacher (`0.952` Gold58) → `ConvNeXt-S` (`0.958` Priv solo) w/ **13 fixed geometric ROI crops** + `Query2Label` + MIL + test attention sharpening (`×1.25–1.75`) | [Writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/7th-place-solution-with-no-external-training-data) · [GitHub](https://github.com/SongqiZhou/rsna-knee-abnormality-detection-7th-place) |

---

## 1. The Four Universal Medical-Imaging Breakthroughs

### 1.1 Label Distillation From Unstructured Reports + Images (`+0.013` to `+0.027` AUC)
On `Gold58`, radiologists marked **240 positives** while free-text reports only mentioned **154** (e.g., `PF OA`: 21 actual vs 2 reported; report readers missed silent incidental findings and reflected hospital-specific dictation habits—`Synovitis` was marked `92%` vs `6%` depending purely on whether a hospital wrote the word in reports, `#7`):
- **Image + Report (Untrusted Hint) Beats Report-Only Dramatically**: Reading MRI slices directly with the native-language report treated strictly as an *untrusted hint* lifted Gold58 label AUC from `0.899 → 0.942 → 0.956` (`#2`) and raised identical `CoAtNet-2` Private LB from **`0.925 → 0.952`** (`#4`).
- **Continuous Soft Probabilities Crush Discrete Bins (`#1`)**: Outputting continuous `[0, 1]` finding probabilities reached **`0.936–0.939`** Gold58 AUC vs **`0.8832`** for 6-bin discrete scales!
- **Per-Hospital Shortcut Debiasing (`#5`, `#6`, `#7`)**:
  - Grouping CV folds by **inferred hospital/scanner cluster** exposed `+0.019` artificial AUC inflation from multi-site reporting shortcuts (`#5`).
  - Evaluating **within-scanner×field-strength confidence-weighted proxy AUC** jumped CV–LB Spearman from `0.86 → 0.94` (`#6`).
  - Solving a per-scanner logit shift to match expected `Synovitis` rate from image-predicted `Effusion` removed hospital-dictation bias (`#7`).

### 1.2 Physical-FOV Standardization + 2D/2.5D MIL Crushed End-to-End 3D CNNs
Every single Top-7 team converged on **physical-millimeter normalization + canonical laterality mirroring + 2D / 3-slice 2.5D encoders**:
- **Physical FOV (`112–150 mm → 384px`)**: Resample slices using DICOM pixel spacing and patient coordinates (`ImagePositionPatient`), sorting slices along anatomical axes and clipping intensities (`0.5–99.5%` or `1–99.9%` for fluid-sensitive).
- **Anatomical Laterality Mirroring**: Mirror all knees to one canonical side using a **dedicated image/3D side classifier (`99.3%` acc, `#1`, `#4`, `#5`, `#6`)**—never trust noisy DICOM laterality tags!
- **Backbone + Pooling Winners**: `ConvNeXt` (`Tiny`/`Small`/`Base`), `CoAtNet-2`, `MaxViT-V2-B`, `BiT ResNetV2-50`, `DINOv3 ViT-B/16`, `EfficientNetV2-S/M`, and `MambaOut-B+` paired with **per-finding gated Attention MIL**, **3D-box spatial attention pooling** (`#1`), or **2-query `Query2Label` + MIL** (`#7`).
- **Critical MIL Bag-Size Rule (`#4`)**: Increasing training bag sampling from `K=12 → K=24` windows per study prevented focal lesions from being missed during training (`+0.003` Pub / `+0.002` Priv; `K=48` hurt).
- **Test-Time Attention Sharpening (`#6`, `#7`)**: Sharpening MIL attention temperature at inference (`τ=12, 32` in `#6`, `×1.25–1.75` in `#7`) boosted Private LB `+0.002` to `+0.003` by suppressing background slice dilution.

### 1.3 Anatomy-Guided ROI Cropping & Auxiliary Supervision (`+0.002–0.003` Priv)
- Zooming into joint compartments (`Medial`, `Lateral`, `PF`, `ACL/Menisci`) via **learned 3D ROI detectors** (`#1`), **`nnU-Net` bone segmentation** (`#5`, `+0.003` Priv), **5-keypoint 3D knee frames** (`#6`), **geometry-only intensity valleys** (`#4`, `0.955 → 0.958` Priv with zero external data!), or **13 fixed normalized geometric crops** (`#7`) consistently fixed weak findings (`PF OA`, `Lateral OA`, `Lateral Meniscus`, `Synovitis`).
- **Train-Only Multimodal Auxiliary Heads**: Distilling frozen text report embeddings (`Qwen3-Embedding-8B` cosine loss, weight `0.3`, `#1`), 48-target report checklists (`#4`), and 2D lesion boxes (`#1`) improved visual encoders without needing reports at test time!

### 1.4 External Pretraining & Ensembling Guardrails
- **Balanced External Mixing (`#3`)**: When mixing ~9,440 external OAI knees, sampling `2,000/epoch` at `70:30` RSNA:OAI with a train-only **domain prevalence logit offset** beat using all 9,440/epoch.
- **Fixed Per-Finding Rank Blending (`#2`, `#3`, `#7`)**: Hand-set equal/group rank or probability blends outperformed tuned weights (tuned weights lost `0.002–0.007` vs fixed in `#2`), and **hard runtime budget fallbacks** on 2×T4 prevented deadline-day I/O timeouts (`#2`, `#4`, `#5`).

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **3D CNNs / Heavy Cross-Plane Study Transformers** | Did not beat 2D/2.5D MIL + box pooling (`~0` or worse) | #1, #4, #7 | Variable slice thickness, missing series, and small study count favor 2D/2.5D slice encoders + MIL |
| **Exposing Gold58 Ground Truth to VLM Labelers / Over-Weighting Gold58 (`w=5`)** | Caused copying/overfit (`0.80` blind, `#2`); `-0.003/-0.004` LB (`#4`) | #2, #4 | Tiny 58-study gold set must stay strictly held-out for unbiased label/recipe validation |
| **6-Bin Discrete Labels vs. Continuous Soft Probabilities** | `0.8832` vs `0.936–0.939` Gold58 AUC | #1 | Continuous probabilities preserve ranking calibration critical for ROC-AUC |
| **Tuned Ensemble Weights / Hard Routing / Standard TTA** | Tuned weights lost `-0.002` to `-0.007` (`#2`); hard routing `<` soft mix (`#3`); TTA `~0` (`#4`) | #2, #3, #4 | Tiny Gold58 / noisy Public split causes meta-weight overfitting |
| **Random Study CV Across Multi-Site Medical Data** | Inflated report-label AUC by **`+0.019`** with zero Gold58 gain | #5, #6 | Models memorize hospital scanner/language correlation with reporting conventions |
| **Close-Up Crops Without Whole-Knee Context / Narrow Bone Crops** | Lost `-0.010` on Lateral Meniscus (`#2`); missed top `20%` of Effusion slices (`#5`) | #2, #5 | Always fuse regional crops alongside whole-joint context covering full fluid recesses |
