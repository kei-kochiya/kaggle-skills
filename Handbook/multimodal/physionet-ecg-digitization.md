# PhysioNet — Digitization of ECG Images — Top Gold Solutions & Post-Mortem

> **Competition**: [PhysioNet - Digitization of ECG Images](https://www.kaggle.com/competitions/physionet-ecg-image-digitization) (Oct 21, 2025 – Jan 22, 2026 | 1,424 Teams | 23,524 Submissions | \$50,000 Prize Pool)  
> **Task**: Reconstruct **12-lead continuous 1D mV ECG waveforms** (`Lead II` = 10s, 11 other leads = 2.5s; `fs ∈ {250..1025} Hz`) from color/B&W scans, stained/moldy/wrinkled prints, and moiré smartphone photos (`3×4 + 1R` layout)  
> **Evaluation Metric**: **Modified SNR (dB)** — per-record `±0.2s` lag + vertical-offset alignment, linear power SNR summed across all 12 leads, **averaged in linear SNR space across records**, then `10·log10` (making clean/medium `25→30 dB` gains dominate near-0 dB outliers!)  
> **Dataset**: `85.75 GB` (`9,775` files across 9 degradation variants `-0001..-0012` + 69 rotated images) | Hidden Test: `~1,000` images (`20%` Pub / `80%` Priv; `<= 9h` 2×T4 notebook)

---

## Official Top Gold Leaderboard Standings & Solution Index

Every top solution converged on **high-resolution single-step rectification (`4400–5600 px` wide, avoiding double-interpolation blur)**, **sub-pixel signal extraction + Fourier resampling (`scipy.signal.resample`)**, and **outlier-resistant, lag-aligned signal-space ensembling + gated Einthoven physics**:

| Private Rank | Public Rank | Team | Public dB | Private dB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **#1** | **`YUAN High-Tech`** | **23.52273** | **23.33162** | **Direct 1D regression (zero seg mask!)** on **single-step HR remapped `5056×1280` grayscale+coord crops (`+2.57 dB`!)**; 10-model backbone zoo (`ConvNeXtV2`, `EffNetV2`, `MambaOut`, `InceptionNeXt`) predicting `4×20k` → **Fourier resample (`+1.66 dB`!)** + gated Einthoven (`<0.01`) | [Writeup](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/1st-place-solution) |
| 🥈 **#2** | #2 | **`Takashi Someya`** | 23.39003 | **23.27649** | **Recovered original `500 Hz` PTB-XL GT (`+0.82 dB` LB: `21.67 → 22.49`!)**; **sparse 2-px fractional masks** on `5600 px` width + **2.5D cross-lead skip fusion** across stacked rows (`EffNet B6/B7/V2-L`, 6-model blend) + Fourier resample | [Writeup](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/2nd-place-solution) · [GitHub](https://github.com/someya-takashi/physionet-image-digitization/tree/main) |
| 🥉 **#3** | #4 | **`Tang`** | 22.84023 | **22.76388** *(+1)* | **2× HR rectification (`4400×1700`, `+7.00 dB` theoretical ceiling)** + sharp 1-px masks (`ConvNeXtV2 > HRNet > EffNetV2`) + **parabolic sub-pixel interpolation** + signal-space blend + `2:1:1` Einthoven + 4× flip TTA | [Writeup](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/3rd-place-solution) · [GitHub](https://github.com/tanghaozhe/physionet-ecg-image-digitization-3rd-place) |
| 🏅 **#4** | #7 | **`tomoon33`** | 22.71163 | **22.69875** *(+3)* | **FCMAE-pretrained `ConvNeXt-S`** indexed `55×43` grid lines → 4 vertical `800×800` strips → **horizontally-only upsampled U-Net + 1D Conv** (SNR+Mean loss) → **Weighted Median ensemble beat Mean by `+0.32 dB` Priv!** | [Writeup](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/4th-place-solution) · [GitHub](https://github.com/uchiyama33/physionet_4th_place) |
| 🏅 **#5** | #3 | **`Đăng`** (`dangnh0611`) | 22.93061 | **22.62929** | `MINIMA-RoMa` → 5-fold `ConvNeXt-S` **2,422-kp local `K=16` homography (`INTER_LANCZOS4`)** → **Dual-Encoder (`VGG19` + `CoaT`/`ConvNeXt-L`)** 4-ch UNet (+ blank template) w/ **Column-wise JSD** + **`±6.4 mV` TALL refiner** | [Writeup](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/5th-place-solution) · [GitHub](https://github.com/dangnh0611/kaggle_ecg_digitization) |
| 🏅 **#6** | #5 | **`AnnieGo`** | 22.82528 | **22.58585** | `@hengck23` Stage 0+1 + `SoftArgmax(T=0.5)` direct lead regression + GPU FFT `resample_torch` (`5120 > 2560`: `+0.84 dB` LB; hflip TTA `+0.16 dB`) | [Writeup](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/6th-place-solution) · [GitHub](https://github.com/GWwangshuo/Kaggle-2025-PhysioNet/tree/main) |
| 🏅 **#7** | #6 | **`Pixels2Pulses`** (Bartley & Harshit) | 22.80704 | **22.48823** | **Zero dewarping!** 2-pass `ConvNeXt` lead detector/13-cls seg (`+0.3 dB`) → **`Muon`-trained `MaxxViT-v2` (5-ch crossing-aware) + progressive `DINOv3-S`** → **`ArcFace` OOD gating (`+0.38 dB` Priv!)** | [Writeup](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/7th-place-solution) · [GitHub](https://github.com/brendanartley/PhysioNet-Competition) |
| 🏅 **#8** | #8 | **`Imanishi & James & Liu`** | 22.16512 | **22.03217** | 3 diverse pipelines (`single-step grid_sample + CoordLoss +1.0 dB`, `DINOv2 8792px + top-10 softmax`, `ResNet34 frozen-BN`) blended via **lag-aligned signal-space averaging (`+0.60 dB` over best single!)** | [Writeup](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/8th-place-solution) · [NB](https://www.kaggle.com/code/liuzhangzhen/ecg-liu-james-imanishi-ensemble-vfin?scriptVersionId=293262413) |
| 🏅 **#9** | #10 | **`YumeNeko`** | 22.04196 | **21.92226** *(+1)* | `ALIKED+LightGlue` homography + **`UHDM` moiré removal** on type `0006` screen photos + piecewise affine + `6×` W-expanded `ResNeSt-14d` gated expectation head | [Writeup](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/9th-place-solution) |
| 🏅 **#10** | #11 | **`Giba`** | 22.03714 | **21.71036** *(+1)* | `YOLOv12` homography → **indexed 54×42 grid-line multilabel UNet** (`~18 → ~21 dB` over raw intersection+Delaunay!) → 16 lead crops | [Writeup](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/top-10-solution-giba) |

---

## 1. Universal Cross-Modal Image-to-1D-Signal Breakthroughs

1. **Single-Step High-Res Rectification & Asymmetric Horizontal Scaling (`#1`, `#3`, `#4`, `#5`, `#8`, `#10`)**:
   - **Never warp twice**: Estimate rotation & grid keypoints/indexed lines (`55×43`, `#4`, `#5`, `#10`) at manageable resolution, then apply **one single `grid_sample` / `INTER_LANCZOS4` remap directly to the raw high-res image** (`+2.57 dB`, `#1`, `#8`).
   - **Horizontal Resolution Dominates SNR**: Doubling rectified width `2200 → 4400 px` raised theoretical mask-to-signal SNR ceiling by **`+7.00 dB`** (`#3`), `2560 → 5120` added **`+0.84 dB` LB** (`#6`), and `#4` saved VRAM by **upsampling horizontally only** after `1/8` U-Net scale.
   - **OOD-Gated No-Warp Alternative (`#7`)**: Without grid dewarping, pairing a 2-pass lead detector with an OOD-robust digitizer + ID-specialist (`DINOv3`) gated by **`ArcFace` embedding cosine similarity (`< 0.5`)** jumped Private LB **`+0.38 dB`**.
2. **Sub-Pixel Signal Extraction & Fourier Resampling (`#1`–`#6`, `#8`)**:
   - Integer `argmax` on blurry masks loses `> 1 dB`. Winners extracted sub-pixel precision via **direct 1D regression (`#1`, `#4`)**, **sparse 2-px fractional masks (`#2`)**, **column-wise JSD / `CoordLoss` local Gaussian CE (`+1.0 dB`, `#5`, `#8`)**, **parabolic peak refinement (`#3`)**, or **top-K / temperature soft-argmax (`#6`, `#8`)**.
   - **Fourier resampling (`scipy.signal.resample` / `resample_torch`)** crushed linear interpolation by **`+1.66 dB` (`#1`, `#2`, `#6`)**.
   - **Recovering Unresampled Original Ground Truth (`#2`)**: Matching resampled train targets (`250–1025 Hz`) back to native `500 Hz` PTB-XL waveforms eliminated resampling aliasing and jumped single-fold LB by **`+0.82 dB` (`21.67 → 22.49`)**.
3. **2.5D Cross-Lead Skip Fusion, Dynamic Range Guards & Lag-Aligned Robust Ensembling**:
   - **Cross-Lead Context (`#2`, `#4`)**: Fusing skip features vertically across simultaneous stacked rows lets occluded/damaged leads borrow phase & QRS timing from clean leads above/below.
   - **Aligned Signal-Space Blending & Weighted Median (`#3`, `#4`, `#8`)**: Blending in 1D signal space **after metric-style `±0.2s` cross-correlation alignment** beat naive averaging by **`+0.60 dB` (`#8`)**; **Weighted Median** beat Weighted Mean by **`+0.32 dB` Priv (`#4`)** by rejecting outlier spikes!
   - **Gated Physics Constraints (`#1`, `#3`, `#5`, `#8`)**: Applying **Einthoven's Law (`II = I + III`)** conditionally (`|II - I - III| < 0.01`, `#1`) + Lead II short/long overlap blending added clean `+0.05–0.06 dB`.

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Direct 1D Regression on Unrectified Images / Predicting Dense Inverse-Warp Maps** | Non-competitive vs grid-rectified crops (`#1`) | #1 | Perspective distortion & paper curl break fixed column-to-time correspondence |
| **Naive Signal Mean Ensembling Without Lag Alignment or Outlier Rejection** | Unaligned mean blurs peaks (`-0.60 dB`, `#8`); Mean lost **`-0.32 dB` Priv** to Weighted Median (`#4`) | #3, #4, #8 | Tiny `±1–3` sample horizontal shifts across models cancel sharp QRS peaks; single outlier spikes crush linear SNR |
| **Linear Interpolation Resampling & Thick Gaussian-Blurred Masks** | Linear lost **`-1.66 dB`** vs Fourier (`#1`); blurred masks hurt peak accuracy (`#2`, `#3`) | #1, #2, #3, #6 | Band-limited ECG signals preserve high-frequency QRS spikes under sinc/FFT resampling and crisp 1–2 px masks |
| **Tight Vertical Crop Ranges (`512×512`) Without Tall-Amplitude Fallback** | Clipped high-voltage QRS complexes at `±3.2 mV` (`#5`) | #5 | Extreme amplitude beats require a `±6.4 mV` refinement head (`TALL` model) |
| **Unconditional `aVR + aVL + aVF = 0` Projection Without Residual Gating** | Helped CV but hurt LB (`#8-James`) | #8 | Badly reconstructed leads corrupt clean leads unless gated by small residual (`< 0.01`, `#1`) |
