# BirdCLEF+ 2026: Acoustic Species Identification in the Pantanal — Top 11 Gold Solutions & Post-Mortem

> **Competition**: [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) (Mar 11 – Jun 4, 2026 | 4,094 Teams | \$50,000 Prize Pool)  
> **Task**: Multi-Label Acoustic Wildlife Classification (`234` target classes across *Aves, Amphibia, Mammalia, Insecta* [25 sonotypes], *Reptilia*) in 5-second field soundscape windows (`32 kHz .ogg`)  
> **Evaluation Metric**: Macro-averaged ROC-AUC skipping classes with zero true positives in the test set  
> **Dataset**: `35,550` focal `train_audio/` clips (`97.9%` Aves; **28 classes have 0 focal clips**) | `10,658` 1-min `train_soundscapes/` (`66` expert-labeled = `739` unique 5s segments; `10,592` unlabeled) | Hidden Test: `~591` 1-min soundscapes (`34%` Public / `66%` Private)  
> **Runtime Constraint**: **CPU-Only Notebook `<= 90 minutes` (4 vCPUs)**; GPU submissions disabled (`<= 1 min` cap)

---

## Official Top 11 Gold Leaderboard Standings & Solution Index

BirdCLEF+ 2026 showcased **foundation-model embedding distillation (Google Perch v2 & AudioProtoPNet)**, **domain-shift-aware iterative soft pseudo-labeling on unlabeled soundscapes**, and **OpenVINO / ONNX CPU serving engineering** under a tight 4-vCPU / 90-minute runtime budget:

| Private Rank | Public Rank (LB) | Team / Competitor | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :--- | :--- |
| 🥇 **#1** | #1 (0.96720) | **Nikita Babych** (`nikitababich`) | **0.96574** | **2-phase Cosine Embedding Distillation** (`Perch v2` + `AudioProtoPNet`) → CE fine-tune + **label-sum-capped non-overlapping LSS & PL injectors** (2 iters) + **Genus SED & Amphibia/Insecta specialist** + `0.80/0.20` rank blend w/ native Perch v2 | [Writeup](https://www.kaggle.com/competitions/birdclef-2026/writeups/1st-place-solution-noisy-student-meets-distillati) · [Infer](https://www.kaggle.com/code/nikitababich/birdclef2026-1st-place-inference) |
| 🥈 **#2** | #12 (0.95995) | **`tennogh`** (`FlorentinGe`) | **0.96013** | Deliberately **zero-KD** diverse 7-model ensemble (~87m CPU): native Perch + 5-round USS PL (`softAUC + 0.25 BCE`) on 2025 XC-pretrained `EffNetV2-S` & `NFNet` (keeping Spearman $\rho \approx 0.39–0.54$ vs Perch) + `EffNet-B0` Insecta specialist (`+0.004`) | [Writeup](https://www.kaggle.com/competitions/birdclef-2026/writeups/2nd-place-diverse-ensemble-with-pseudo-labeling-a) · [GitHub](https://github.com/FlorentinGe/Birdclef2026-2ndplace) |
| 🥉 **#3** | #541 (—) | **`kapenon`** (+538!) | **0.95992** | **OpenVINO IR** 3-way diversified blend (`128` vs `256` mels, w/o PL + w/ PL `EffNetV2-S` + `seresnext26t` Perch KD) + **`1/sqrt(site_count)` USS sampling** + Bayesian-shrunk site/hour priors + file-confidence/rank scaling | [Writeup](https://www.kaggle.com/competitions/birdclef-2026/writeups/3rd-place-solution) · [Notebook](https://www.kaggle.com/code/kapenon/birdclef2026-3rd-place-submission) |
| 🏅 **#4** | #3 (0.96334) | **BirdCLEF+ 2026 Team** (Tony Li, Yiheng Wang, Starry) | **0.95902** | **2-lane × 2-vCPU parallel scheduler**: `0.40` Perch v2 ONNX rank-gating + `0.26` `EffNetV2-S` SED (10s/5s) + `0.34` 256-mel Perch-distilled `EffNetV2-S` & `seresnext26d` + genus/fat-tail temporal smoothing | [Writeup](https://www.kaggle.com/competitions/birdclef-2026/writeups/4-th-place-solution) |
| 🏅 **#5** | #26 (0.95715) | **Jiacheng Ma** (`jakkma`) | **0.95824** | Perch-distilled CNN zoo (`hgnetv2b0`, `effv2_s` 5s/10s, `effb3`) w/ **2-iter self-distillation + PL** + **multiplicative dB FilterAug** (`+0.01+`) + `0.6/0.4` blend w/ `ProtoSSM` + `Distilled SED` | [Writeup](https://www.kaggle.com/competitions/birdclef-2026/writeups/5th-place-solution-both-are-all-you-need) · [GitHub](https://github.com/jak-ma/BirdCLEF2026-5th-solution) |
| 🏅 **#6** | #9 (0.96061) | **Sinan Calisir** (`snnclsr`) | **0.95762** | Independent 2–3 round soft PL across `ECA-NFNet`, `SwinV2`, `EffNetV2` (`p ≥ 0.55` filter, `<0.05` zeroed) + **single shared-spectrogram OpenVINO graph** (40–45 min) + `0.5/0.5` rank-pct blend w/ Perch/ProtoSSM | [Writeup](https://www.kaggle.com/competitions/birdclef-2026/writeups/6th-place-solution) |
| 🏅 **#7** | #45 (0.95541) | **空飛ぶ宝石** (Joseph Zhou) | **0.95715** | Fixed 3 XC/iNat taxonomy name mismatches + manual Audacity relabeling (4k+ clips); `EffNetV2-S` (Perch KD) + `mambaout_tiny` & `eca_nfnet_l0` (AudioProtoPNet KD) + ProtoSSM | [Writeup](https://www.kaggle.com/competitions/birdclef-2026/writeups/2026-birdclef-7th-solution) |
| 🏅 **#8** | #73 (0.94751) | **`kazumax`** | **0.95675** | Cosine Perch KD (`w=10.0`) on `effv2_b3`, `regnety_016`, `maxxvitv2_nano` (`192×320` mel) + **2nd-stage 60s BiGRU/LSTM sequence stackers** (residual for non-sonotypes, direct for sonotypes) + GBDTs | [Writeup](https://www.kaggle.com/competitions/birdclef-2026/writeups/8th-place-solution) |
| 🏅 **#9** | #2 (0.96703) | **Yannan Chen** | **0.95661** | 10s/2-seg `EffNetV2-S` + **zero-init Class-Interaction MLP** trained purely on unmixed LSS (`0.939 → 0.952`) + extra XC/iNat (cap 200); proved raw 2-model blend (`0.958` Priv) beat heavy PL (`0.956` Priv) | [Writeup](https://www.kaggle.com/competitions/birdclef-2026/writeups/private-9th-public-2nd-solution) · [GitHub](https://github.com/yannan90/birdclef-2026-private-9th-public-2nd-solution) |
| 🏅 **#10** | #11 (0.95995) | **`coolz`** | **0.95627** | **End-to-end trainable PyTorch Perch v2** (`20s` audio, frozen 128-mel) + **RoPE + 4-layer SED Transformer** (dual att/max head), 5-fold OpenVINO + site/hour/month priors | [Writeup](https://www.kaggle.com/competitions/birdclef-2026/writeups/10th-solution-simple-model-as-always) · [Notebook](https://www.kaggle.com/code/cooolz/perchv2-sed) |
| 🏅 **#11** | — | **Salman Ahmed** | **0.95599** *(0.959 unselected)* | **Zero Perch**: high-res **480-mel** (`n_fft=4096, hop=1001`, 20s → 20×1s frames) EfficientNet family + explicit `nocall` class; unselected Stage-1 hit `0.959` Priv | [GitHub](https://github.com/sahmedcodes/birdclef26-weekend-11thplace-3rd) |

---

## 1. Competition DNA & Extreme Domain-Shift Anatomy

1. **Severe Focal-to-Soundscape Taxonomic Inversion**:
   - `train_audio/` (`35,550` single-species focal clips from Xeno-Canto & iNaturalist) was **97.9% birds (*Aves*)**.
   - In contrast, expert-labeled `train_soundscapes_labels.csv` (`66` 1-min recordings = `739` unique 5s segments, duplicated 2× in CSV = `1,478` rows) flipped to **Amphibia (4,174 mentions) & Insecta (1,136)** dominating **Aves (824)**.
2. **The "Invisible 28" Zero-Focal Classes**:
   - **28 of 234 target classes** (all **25 insect sonotypes** like `47158son16` + **3 amphibians**) had **ZERO clips in `train_audio/`**, existing *only* inside the 66 labeled soundscapes and the 10,592 unlabeled soundscapes (`USS`).
3. **Site Skew & Unseen-Site Generalization**:
   - `train_soundscapes/` was heavily concentrated in 4 sites: Site 01 (`2,341`), Site 02 (`2,505`), Site 13 (`1,873`), and Site 22 (`3,383`). Hidden `test_soundscapes/` (~591 files) was also skewed toward Site 22 (215 files), while remaining test sites tested unseen-location transfer.
4. **Hard 90-Minute 4-vCPU Budget**:
   - GPU inference was disabled (`<= 1 min`). Heavy vision transformers (ViT, ConvNeXt-Large, PaSST) and naive multi-model Perch ensembles timed out immediately unless distilled into compact CNNs and compiled via **OpenVINO IR / ONNX Runtime**.

---

## 2. Two Dominant Modeling Paradigms: Foundation Distillation vs. Pure-Diversity Ensembling

### 2.1 Paradigm A — Two-Phase Embedding Distillation from Audio Foundation Models (#1, #3, #4, #5, #7, #8)
Inspired by Tucker Arrants's `Distilled SED Baseline` (`bc2026-distilled-sed`), top teams bypassed Perch v2's slow CPU footprint by **distilling Google Perch v2's 1536-d embeddings** (and **AudioProtoPNet-5-BirdSet-XCL** for teacher diversity in #1 and #7) into fast 2D mel-spectrogram CNNs (`eca_nfnet_l0`, `regnety_032`, `tf_efficientnetv2_s`, `seresnext26t/d`, `hgnetv2b0`):
- **Cosine Loss > MSE Loss (1st & 8th Place)**: Nikita Babych (#1) verified that **Phase 1 Cosine Embedding Distillation** (11 epochs, `lr=5e-4`, mixup `p=0.5` over focal + soundscapes via a linear projection head) converged significantly better than MSE distillation, followed by **Phase 2 Supervised/PL Fine-Tuning** (8–15 epochs, `lr=2e-4`, Cross-Entropy loss). Re-distilling independently per model seed preserved ensemble diversity.
- **Native Perch v2 Rank Anchor (`0.20–0.40` weight)**: Even after distilling Perch into CNNs, keeping a **native Perch v2 head** (covering 203 overlapping classes via ONNX Runtime) or `Perch + ProtoSSM` at `20%–40%` **rank-blended weight** consistently gave a huge boost because native Perch never forgot rare focal species during soundscape pseudo-labeling.

### 2.2 Paradigm B — Deliberate Zero-KD Diversity (`tennogh`, #2; `Salman Ahmed`, #11)
- **2nd Place (`tennogh`)** deliberately **discarded Perch KD on core CNNs** after seeing KD raise inter-model correlation with native Perch! Instead, by training 2025 XC-pretrained `EffNetV2-S` and `NFNet` through 5 rounds of USS pseudo-labeling with `softAUC + 0.25 BCE` loss, `tennogh` kept OOF Spearman correlation vs. Perch/SED at an ultra-low **`0.39–0.54`**, jumping from `#12` Public to **`#2` Private (`0.96013`)** with near-equal blending weights across 7 models.
- **10th Place (`coolz`)** showed an orthogonal frontier: fine-tuning **end-to-end trainable PyTorch Perch v2** on 20s clips (frozen 128-mel frontend) topped by **RoPE + 4-layer SED Transformer** (`0.95627` Priv).

### 2.3 Taxonomic Specialists & Hierarchy Heads (#1, #2, #8)
- **Genus-Level SED Head (1st Place, `+0.001–0.002` at `0.96+` LB)**: Trained an `eca_nfnet_l0` to predict `max(label)` per **biological genus** and spread predictions back to member species—directly regularizing rare sibling species.
- **Amphibia + Insecta / Insecta-Only Specialists (#1 & #2)**: Because 97.9% of focal clips were birds, training a dedicated specialist on non-avian taxa (`EffNet-B3` supervised-only on 41,127 extra XC+iNat Amphibia/Insecta clips in #1; `EffNet-B0` Insecta-only weighted 50% on Insecta for `+0.004` LB in #2) lifted the hardest macro-AUC denominator classes cleanly.
- **2nd-Stage 60-Second Sequence Stackers (8th Place)**: Fed all 12 five-second chunk logits from a full 1-minute soundscape into **BiGRU/LSTM stackers** (residual connection for non-sonotypes, direct for continuous insect sonotypes) to capture full-minute acoustic context.

---

## 3. Solving Domain Shift: Labeled Soundscape Injection & Iterative Soft Pseudo-Labeling

Naive Noisy Student pseudo-labeling **collapsed on top of Perch-distilled models** (1st place) and **overfit Public LB over Private LB** when pushed too aggressively (9th place ablation: 2 raw models `0.958` Priv beat heavy PL `0.956` Priv!). Top winners solved this with strict mathematical guardrails:

| Guardrail / Technique | Exact Implementation | Team & Measured Effect |
| :--- | :--- | :--- |
| **Non-Overlapping Label-Sum-Capped LSS & PL Injectors** | Mix LSS chunks (`18.75%` of batch, **label sum capped/normalized at `0.5`** + 2 clean concats) and USS PL chunks (`75%` of batch, **label sum capped at `0.75`**) onto **non-overlapping focal samples**; mask non-LSS species on Site 22 (`+0.002`) | **#1 (`nikitababich`)**: `eca_nfnet_l0` 2-seed: `0.935` → Iter 1 `0.946` → **Iter 2 `0.950`** → Iter 3 `0.949` (2 rounds optimal) |
| **Site-Balanced USS Sampling** | Sample unlabeled soundscapes proportional to **`1 / sqrt(site_count)`** so Site 22 / 01 / 02 don't drown out rare habitats | **#3 (`kapenon`)**: **+538 Private LB climb** (`#541` Pub → `#3` Priv!) |
| **Confidence-Gated Soft PL (`p ≥ 0.55`, `< 0.05 → 0`)** | Keep USS segments only where any species `p ≥ 0.55` (calibrated on held-out Site 22 for precision `≥ 0.80`), zero tail noise `< 0.05`, run 2–3 independent PL rounds per backbone family before ensemble PL | **#6 (`snnclsr`)**: single model `0.900–0.915` → `0.945–0.950` |
| **Zero-Init Class-Interaction MLP on Pure LSS** | Keep primary CNN clean; train a **zero-initialized residual Class-Interaction MLP strictly on unmixed LSS** to learn real-world soundscape co-occurrence without corrupting focal features | **#9 (`Yannan Chen`)**: `0.939 → 0.952` Pub / `0.950` Priv |
| **Substitution vs. Mixup + `softAUC + 0.25 BCE`** | Switch from mixing USS into focal clips to **substituting USS at `50%–60%` prob** in Rounds 2–5 with power scaling (`1.0–1.2`) and direct **`softAUC + 0.25 BCE`** optimization | **#2 (`tennogh`)**: R1 `0.919` → R4 **`0.938` Pub / `0.935` Priv** |

---

## 4. Audio Frontends, Augmentations, Post-Processing & CPU Optimization

### 4.1 Mel-Spectrogram Frontends & Augmentations
- **Canonical 5-Second Frontends**: `32 kHz`, `n_fft=2048`, `hop=512` (or `417`/`313`), `n_mels=128` (2025 XC-pretrained backbones) or **`n_mels=256`** (Perch-distilled `seresnext26` & `v2s` in #3 & #4; `192×320` in #8; `480` mels in #11). Always match the exact mel parameters used by a pretrained audio checkpoint!
- **Universal Audio Augmentations**: Waveform **MixUp** (`p=0.5`, max-lambda `max(β, 1-β)` in #5), **SpecAugment** (mild freq/time masking), and **`RandomFiltering`** (STFT → 4-band piecewise-linear EQ `[-20, 0]` dB → iSTFT in #3, #4, #6; plus #5's multiplicative dB filter discovery worth `+0.01+`).

### 4.2 Rank Blending & Soundscape Temporal Post-Processing
- **Per-Column Percentile Rank Blending**: Mandatory when combining native Perch v2, Perch-distilled SED CNNs, ProtoSSM, and pure CNNs whose probability scales diverge sharply (#1, #4, #6).
- **Soundscape File-Max & Class-Conditional Temporal Smoothing**:
  - **File-max anchoring**: Pull chunk predictions toward the 1-minute recording max (`α=0.20` in #1; `× file_max^0.4` & `× top2_mean^0.4` in #3) because a species singing loudly in one 5s window is far more likely to appear softly in neighboring windows.
  - **Class-conditional temporal smoothing (#1)**: Broad flat kernel `[0.33, 0.33, 0.33]` for continuous vocalizers (**Amphibia/Insecta**) vs. sharper `[0.2, 0.6, 0.2]` for transient callers (**Aves/Mammalia**), plus **genus (`α=0.12–0.15`) & class (`α=0.05`) taxonomy smoothing** and **Delta/2.5s-shift TTA** (#1, #3, #5).
  - **Adaptive smoothing (#3)**: Smooth proportionally to uncertainty: `alpha = 0.20 * (1 - max_prob)`.
  - **Warning on Site/Hour Priors (#9 vs #1/#3/#4)**: Bayesian-shrunk site/hour priors (`w = N/(N+K)` in logit space, #3) helped mid-tier/mixed models, whereas #9 observed site/hour priors turned **net-negative above `0.95` LB** and #5 warned against applying identical post-processing blindly across all model families.

### 4.3 4-vCPU / 90-Minute Inference Engineering
1. **Shared-Spectrogram Single OpenVINO Graph (#6)**: Export raw waveform `(B, 160000)` → **one shared log-mel frontend** → parallel CNN/Swin branches → weighted output inside a single OpenVINO IR graph, cutting 7–8 minutes of redundant CPU mel computation.
2. **Dynamic 2-Lane × 2-vCPU Pipeline (#4)**: Run ONNX Runtime (`Perch v2`) and OpenVINO IR CNNs concurrently across 2 worker lanes (`2` threads each on 4 vCPUs) to saturate CPU without thread-contention thrashing.

---

## 5. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Outcome | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Standard Uncapped Noisy Student PL on Perch-Distilled Models** | Collapsed (#1) or inflated Public LB while hurting Private LB (`0.958 → 0.956`, #9); hurt Insecta specialist (#2) | #1, #2, #9 | Frequent LSS species hijack gradient mass without label-sum caps (`0.5/0.75`) and site-balanced USS sampling |
| **Hard Pseudo-Labels vs. Soft Probabilities** | Inferior to soft probability distillation | #6 | Discards multi-species co-occurrence uncertainty and sharpens false positives on noisy soundscapes |
| **Over-Weighting Best Public-LB Models / Ignoring Perch Correlation** | Hurt Private LB vs. near-equal weights (#2); Perch KD inflated inter-model correlation | #2, #9 | Public LB was only 34% of test data with heavy site skew; maximizing diversity (`ρ ~ 0.4–0.5`) beat greedy Public-LB weighting |
| **Massive Unfiltered External Data / Domain-Mismatched Frog Sets** | From-scratch **814k XC pretrain** lost to 2025 2nd checkpoint; **AnuraSet** dropped LB; extra target-species XC hurt #1 | #1, #4 | Acoustic hardware/habitat mismatch and noisy focal labels dilute soundscape calibration (cap external data ≤100–200/species or restrict to non-avian specialists) |
| **Heavy Backbones (`ViT`, `ConvNeXt`, `PaSST`) Under 90-Min CPU Cap** | ViT `< 0.90` (#2); ConvNeXt/PaSST exceeded 90-min CPU budget | #2, #6 | Lightweight CNNs (`EffNetV2-S`, `ECA-NFNet-L0`, `RegNetY`, `SEResNeXt26`) + OpenVINO deliver far higher accuracy-per-CPU-second |
| **Blindly Using >5s Clips or Background-Noise Augmentation** | >5s hurt #1 & #6; extra background-noise aug hurt #2 | #1, #2, #6 | Competition evaluates 5s windows; real unlabeled soundscapes already supply authentic multi-species background acoustics |
| **Unverified Taxonomy Name Joins Across External Databases** | Silently mislabeled 3 species between iNat/official taxonomy and Xeno-Canto | #7 | Always audit scientific/common name synonyms before merging external bioacoustic corpora |
