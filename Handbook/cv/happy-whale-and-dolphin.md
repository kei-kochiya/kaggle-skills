# Happywhale — Whale and Dolphin Identification 🏆

> **Track:** Computer Vision — Metric Learning & Open-Set Image Retrieval (Research) · **Scale:** 1,588 teams · \$25,000 · Closed Apr 19, 2022  
> **Core Lesson:** In extreme long-tail open-set re-identification (`>15,000` classes + `new_individual`), **high-resolution multi-crop detectors** (mixing `fullbody`, `backfin`, and spatially concatenated `[Original | Crop]` views), **Sub-center ArcFace (`k=2`) + Dynamic Margins (`m_i ∝ n_i^{-1/4}`)** with **`GeM (p=3, fixed) + BNNeck`**, **iterative open-set pseudo-labeling** (into *training weights only*, never the kNN gallery), **kNN + margin-free FC logit blending**, and **per-species open-set thresholds** crushed classic unsupervised re-ranking (`QE / DBA / k-reciprocal`).

---

## 1. Competition DNA & Leaderboard

- **Task:** Identify `>15,000` individual marine mammals across `26` cleaned species (`30` raw strings with typos from `28` research orgs) in `78,991` images (`~51k` train, `~28k` test, `62.06 GB`), predicting up to **5 `individual_id`s** per test image — including the open-set class **`new_individual`** (`~11%` in train CV vs `~16–16.5%` in test; no test species labels given).
- **Metric:** **`MAP@5`** across test queries (~24% Public / ~76% Private).

| Private Rank | Shake | Team | Private LB | Public LB (Rank) | Core Breakthroughs |
| :---: | :---: | :--- | :---: | :---: | :--- |
| 🥇 **1st** | `↑ 1` | **Preferred Dolphin** (`charmq`, `knshnb`) | **0.87579** | 0.89680 (#2) | 5-source random multi-crop + `1024–1440px` `EffNet-L2-NS` + Sub-center (`k=2`) Dynamic-Margin ArcFace (`10×` head LR) + `GeM(p=3)` + 2-rd PL + **`kNN + FC logit` blend** |
| 🥈 **2nd** | `↓ 1` | **Rist Whales** (`takuoko`, `ishikei`, `tascj`) | **0.87262** | **0.89758 (#1)** | 4× `tf_efficientnet_l2_ns` (`1024px` + `[Orig \| YOLOX]` `1536×768`) + **5 rounds from-scratch PL (`0.812 → 0.870` Priv)** + custom YOLOX re-crop |
| 🥉 **3rd** | `↑ 5` | **lafe** | **0.86974** | 0.88655 (#8) | 5k hand-labeled YOLOv5 + `6` body + `2` backfin (`CurricularFace`) + **`GeM + BNNeck + aug8p3` (`+2.26` CV)** + `AdaFace` + progressive PL (`+0.03–0.04`) |
| 🏅 **4th** | `↑ 1` | **All Data Are Ext** (`boliu0`, `haqishen`) | **0.86517** | 0.88804 (#5) | 9-rd iterative YOLOv5m (`0.939` IoU, `+20%` pad) + **`ConvNeXt-L/XL`** + `DOLG` + 3-rd PL + **per-species relative CV thresholds** (`0.882→0.888` Pub) |
| 🏅 **6th** | `↑ 1` | **YujiAriyasu** | **0.86190** | 0.88662 (#7) | Multi-view crops + `Swin-L@768` + `EffNet-L2` + `ConvNeXt` + `20k-d` L2-concat + per-species thresholds (`+0.5–1%`) |
| 🏅 **7th** | `↓ 1` | **DOLG Army** | **0.86161** | 0.88693 (#6) | `[Fullbody \| Backfin]` `896×1792` spatial concat + `DOLG` + `CurricularFace` + **2nd-stage XGB/LGBM reranker** on top-100 kNN |
| 🏅 **8th** | `↑ 2` | **NTechLab** | **0.86104** | 0.88451 (#10) | 4k 2-class (`body+fin`) detector + shared-space multi-crop (`80k+` rows) + `CosFace` + 2-rd PL (`0.799→0.852`) + whale/dolphin thresholds (`0.872→0.884`) |
| 🏅 **11th** | `—` | **tereka & Ahmet & yu4u** | **0.85303** | 0.88133 (#11) | 5.8k hand-labeled 6-fold YOLOv5x6@1280 + `32k-d` concat + **4 Siamese pairwise rescorers** (`0.837→0.853` Priv) |

---

## 2. Preprocessing, Multi-Crop Detection & High-Resolution Scaling

1. **Species Label Typo Cleaning:** Collapsed `30` raw species strings into `26` canonical labels (`globis`/`pilot_whale` → `short_finned_pilot_whale`, `kiler_whale` → `killer_whale`, `bottlenose_dolpin` → `bottlenose_dolphin`).
2. **Detector Quality & Multi-Crop Diversity (Biggest Early Gain):**
   - **Random Multi-Crop Mixing (`#1`):** Sampled across 5 bbox views during training (`fullbody 60%`, custom YOLOv5 `15%`, `backfin 15%`, `detic 5%`, `uncropped 5%`) to enforce invariance to fin-only shots.
   - **Iterative Detector Cleaning (`#2`, `#3`, `#4`, `#8`, `#11`):** Hand-labeling `4k–5.8k` images or running 9-round OOF low-confidence audits (`IoU = 0.939`, `+20%` box padding in `#4`) yielded massive gains — `#2` fine-tuning Round-5 `EffNet-L2-NS` on cleaner custom YOLOX crops jumped **`0.863 → 0.870` Private LB**!
   - **Spatial Side-by-Side Concatenation > Early Channel Fusion:** Horizontally/vertically concatenating `[Original | Fullbody]` (`1536×768` in `#2`) or `[Fullbody | Backfin]` (`896×1792` in `#7`) boosted accuracy, whereas **6-channel early stacking `[body+fin]` and 4-channel mask inputs failed** (`#1`). For finless species (e.g., belugas), `#3` and `#10` gated to `fullbody`-only embeddings.
   - **Multi-Box Selection at Test (`#3`):** When multiple whales appeared in a test image, compute cosine distance of every box embedding to the train gallery (`cos_dist > 0.5` → pick closest match, else highest detector confidence).
3. **Aggressive Resolution Scaling:** Preserve fine nicks, notches, and scars: **`1024×1024`** standard (`#1`–`#4`, `#7`), up to **`1200–1440px`** (`#1`).

---

## 3. Metric Learning Architectures, Losses & Open-Set Pseudo-Labeling

- **Backbones:** High-res CNNs crushed vanilla ViTs (`#4`, `#11` reported ViT/Swin failures; only `#6` succeeded with `Swin-L` scaled to `768px`):
  - **`tf_efficientnet_l2_ns`** (top single model in `#1`, `#2`, `#6`, `#8`; `#2`'s entire 4-model ensemble was `l2_ns` with gradient checkpointing).
  - **`ConvNeXt-L/XL`** (best family in `#4`; core to `#6`, `#8`, `#10`) and **`eca_nfnet_l2` @ `1024px`** (`#3`, `#4`).
- **Pooling & Necks (`+2.26` CV Ablation in `#3`):**
  - **Fixed `GeM (p=3)` + `BNNeck` + Multi-Stage Concat:** `#1` concatenated GeM-pooled last 2 feature maps with `BatchNorm` (beat `LayerNorm`; **trainable `p` failed**). `#3` ablated on `effb5_ns@512`: `aug8p3` (`Sharpen/ToGray/CLAHE`) `+0.45` → `BNNeck` `+0.63` → **`aug8p3 + BNNeck + GeM` `+2.26` CV** (`+2.19` known, `+2.94` new) → `+mixup + freeze_bn` lifted `efb6@768` fold-0 LB **`0.784 → 0.817`**.
- **Long-Tail Metric Losses:**
  - **Sub-Center ArcFace (`k=2`) + Dynamic Margins (`m_i ∝ n_i^{-1/4}`):** Handled extreme class imbalance (`1` to `>400` images/ID) and multiple viewing angles without doubling classes (`#1`, `#2`, `#4`). `#1` tuned dynamic margins with Optuna on `256px` `effb0` and used **`10×` higher LR on the ArcFace head**; `#10` linearly ramped margin scale `0.2 → 1.0` over epochs `1–5` and used embedding-level mixup (`+0.003–0.005` CV).
  - **`AdaFace` / `CurricularFace` / `CosFace`:** Competitive alternatives (`#3`, `#7`, `#8`), often paired with a **multi-task 26-species auxiliary head** (`#1` used sub-center ArcFace for species too!).
- **Iterative Open-Set Pseudo-Labeling (The #1 Universal Boost — Every Top-10 Team):**
  - Re-training from ImageNet init each round on clean train + high-confidence test PLs (excluding low-confidence `new_individual` queries):
    - `#2`: 5 rounds (`softmax(logits*s) >= 0.8`) drove Private LB **`0.812 → 0.849 → 0.860 → 0.862 → 0.870`**!
    - `#1`: 2 rounds lifted Private LB **`0.85959 → 0.87062 → 0.87579`** (`+0.0162`).
    - `#8`: 2 rounds (top `60% → 70%` confident) lifted a solo model **`0.799 → 0.852`** LB.
  - **Critical Rule (`#1`):** Use pseudo-labels strictly to **train better model weights** — **NEVER add pseudo-labeled test embeddings into the kNN search gallery** (hurt LB!).

---

## 4. Retrieval, Open-Set Thresholding & Re-Ranking Reality Check

1. **kNN + Margin-Free FC Logit Blending (`#1`):** Pure cosine kNN over-predicts high-frequency training classes under long-tail imbalance. Blending max top-500 kNN cosine (orig+HFlip vs orig+HFlip) with **margin-free classification FC logits** (`knn_ratio = 0.5` pre-PL, `0.8` post-PL) directly fixed frequency bias.
2. **Per-Species / Group-Calibrated Open-Set Thresholds (`+0.005–0.012` LB):**
   - Global cosine threshold (`#1` calibrated so `new_individual` is Rank-1 for `16.5%` of test; `#2` fixed `0.5`).
   - **Per-species relative thresholds** tuned on CV and shifted by a single global offset jumped `#4` **`0.882 → 0.888`**, `#6` **`+0.5–1.0%`**, and `#8` **`0.872 → 0.884`**.
3. **2nd-Stage Supervised Rerankers Worked (`#7`, `#11`):**
   - `#7` trained 5-fold XGB/LGBM over top-100 kNN candidates (species probs, top-3 neighbor distances, centroid distance, rank, distance sums).
   - `#11` trained **4 Siamese local-feature comparison networks** on top-20k candidate pairs (`embedding_sim + siamese_score`), lifting Private LB **`0.837 → 0.853` (`+0.016`)**.
4. **Ensembling & Open-Set CV:** L2-normalized embedding concatenation (`4k–32k-d` across diverse backbones/crops) + retraining final models on **100% of train** (once hyperparams & relative thresholds are tuned via open-set CV mirroring the `new_individual` ratio).

---

## 5. Verified Anti-Patterns ("What Failed")

1. **Unsupervised Retrieval PP (`QE / α-QE`, `DBA`, `k-Reciprocal`, `GNN Re-Ranking`) FAILED:** Independently tested and rejected by `#3`, `#4`, and `#8` — in open-set long-tail re-ID with many singletons and `new_individual` queries, unsupervised neighborhood expansion corrupts precision; **iterative PL + supervised rerankers/thresholds** was the winning path.
2. **Doubling Classes via Horizontal Flip:** Classic 2018 fluke-matching trick **failed** (`#1`, `#4`, `#11`) across lateral-body/fin species — use standard HFlip + `k=2` sub-center ArcFace instead.
3. **Inserting Pseudo-Labels Into the kNN Inference Gallery (`#1`):** Hurt score vs training-only PL.
4. **Trainable GeM `p`, Triplet/AdaCos/Focal-Alone, CutMix, SWA, Single-Species Models, & 6-Ch Early Multi-Crop Stacking:** Consistently hurt or added zero gain across `#1`, `#4`, `#8`.

---

## 6. Verified Solution Sources

- **1st Place (`Preferred Dolphin`):** `https://www.kaggle.com/competitions/happy-whale-and-dolphin/writeups/preferred-dolphin-1st-place-solution`
- **2nd Place (`Rist Whales`):** `https://www.kaggle.com/competitions/happy-whale-and-dolphin/writeups/rist-whales-2nd-place-solution`
- **3rd Place (`lafe`):** `https://www.kaggle.com/competitions/happy-whale-and-dolphin/writeups/lafe-3rd-solution-part` (+ `/discussion/319896`, `/discussion/320191`)
- **4th Place (`All Data Are Ext`):** `https://www.kaggle.com/competitions/happy-whale-and-dolphin/writeups/all-data-are-ext-4th-place-solution`
- **6th Place (`YujiAriyasu`):** `https://www.kaggle.com/competitions/happy-whale-and-dolphin/writeups/aillis-jp-yujiariyasu-6th-place-solution`
- **7th Place (`DOLG Army`):** `https://www.kaggle.com/competitions/happy-whale-and-dolphin/writeups/dolg-army-dolg-army-7th-place-solution-summary`
- **8th Place (`NTechLab`):** `https://www.kaggle.com/competitions/happy-whale-and-dolphin/writeups/ntechlab-8th-place-solution-my-part`
- **10th Place (`Yiemon773`):** `https://www.kaggle.com/competitions/happy-whale-and-dolphin/writeups/yiemon773-10th-place-solution`
- **11th Place (`tereka & Ahmet & yu4u`):** `https://www.kaggle.com/competitions/happy-whale-and-dolphin/writeups/tereka-ahmet-yu4u-11th-place-solution`
