# Yale/UNC-CH — Geophysical Waveform Inversion — Top Gold Solutions & Post-Mortem

> **Competition**: [Yale/UNC-CH - Geophysical Waveform Inversion](https://www.kaggle.com/competitions/waveform-inversion) (Apr 8 – Jun 30, 2025 | 1,365 Teams | 22,838 Submissions | \$50,000 Prize Pool)  
> **Task**: Full Waveform Inversion (FWI) — map 5-source × 1000-step × 70-receiver seismic gathers `(5, 1000, 70)` to 2D subsurface velocity maps `(70, 70)` in `[1500, 4500]` m/s across **10 OpenFWI geological families** (`Flat/CurveVel`, `Flat/CurveFault`, `Style_A/B`)  
> **Evaluation Metric**: **MAE** across all 70 rows and odd columns (`x_1, x_3, ..., x_69`)  
> **Dataset**: `107.55 GB` comp subset / `670 GB` full OpenFWI (`470k` labeled pairs) | Test: `~65.8k` samples (`25%` Pub / `75%` Priv; offline CSV upload — **ZERO shakeup across the entire Top 28!**)

---

## Official Top Gold Leaderboard Standings & Solution Index

Recovering and accelerating the **exact 2D acoustic FDTD forward wave simulator $F: y \mapsto x$** (`torch.compile` / custom CUDA at `90–120 samples/s`, `100×` faster than CPU) transformed the competition into a showcase of **five physics-coupled inverse-problem paradigms**, while pure **ViT+RoPE** spatial layouts and `Muon` broke standard UNet bottlenecks:

| Private Rank | Public Rank | Team | Public MAE | Private MAE | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **#1** | **`Harshit Sheoran`** | **6.92837** | **6.92484** | **Side-by-side `(1, 350, 350)` / `896²` layout (`46.9 → 32.5` CV!)** + **pure `DINOv2` ViT+RoPE encoder–decoder**; 4.7M pretrain → **570-ep iterative forward-simulated pseudo-labeling (`F(y_hat) → x_sim`)** → **Top-4 hard-family specialist** (`CurveFault_B/CurveVel_B/Style_A/B`) | [Writeup](https://www.kaggle.com/competitions/waveform-inversion/writeups/harshit-sheoran-1st-place-solution) · [Arch NB](https://www.kaggle.com/code/harshitsheoran/yale-fwi-vit-architecture?scriptVersionId=248201360) |
| 🥈 **#2** | #2 | **`Jeroen Cottaar`** | 7.69468 | **7.65585** | **Hybrid DL Init + Classical Regularized FWI**: unmodified public `28.8` CAFormer → custom double-precision **CUDA FDTD + BFGS/Gauss-Newton** (`30ms fwd / 70ms bwd`) with **family-specific priors $p(y)$** (1D/2D TV, truncated 1,073-mode GP SVD for `Style_A`, decoupled `min(vel)`) → **`28.8 → 7.65`!** | [Writeup](https://www.kaggle.com/competitions/waveform-inversion/writeups/jeroen-cottaar-2nd-place-solution-with-code-refine) · [NB](https://www.kaggle.com/code/jeroencottaar/geophysical-waveform-inversion-2nd-place) |
| 🥉 **#3** | #3 | **`tascj`** *(1 sub!)* | 8.41094 | **8.43207** *(7.99 fixed)* | **6-ch flip-symmetric `float32` GPU FDTD** (`13 TB` blend synth) + split `EVA02-L` ViT+RoPE + pixel-shuffle + **`MuonWithAuxAdam` (`9.x → 7.x` CurveFault_A!)**; caught `y=69` simulator row bug (`8.43 → 7.99` post-fix) | [Writeup](https://www.kaggle.com/competitions/waveform-inversion/writeups/tascj-3rd-place-solution) · [GitHub](https://github.com/tascj/kaggle-waveform-inversion) |
| 🏅 **#4** | #4 | **`greySnow`** | 8.50167 | **8.52966** | `10M+` synth (`bfloat16` downsampled) + **100M MaxViT-style 28-block Conv+Transformer** (`MAE + Confidence Loss`) + exact center-resim flipped twin + Style specialist (`+2.0`) + mode-restoring discrete PP | [Writeup](https://www.kaggle.com/competitions/waveform-inversion/writeups/greysnow-4th-place-solution-for-the-gwi-competitio) · [GitHub](https://github.com/shlomoron/GWI-solution) |
| 🏅 **#5** | #5 | **`F A M A S`** | 8.58786 | **8.66046** | `15×` synth (`Convformer-144` beat `CAFormer` by `-1.0 MAE`) + **Differentiable FDTD test-time pixel backprop** (`linspace(0.1,10)` time weight + **$p<1$ TV sharpening** + stripe scalar opt: **`-14.0` `Style_A`, `-6.0` `Style_B`!**) | [Writeup](https://www.kaggle.com/competitions/waveform-inversion/writeups/f-a-m-a-s-team-famas-5th-place-solution) |
| 🏅 **#6** | #6 | **`hyd`** *(1 sub!)* | 8.89326 | **8.93172** | `10M` synth (incl. **20% DDIM-generated velocity maps**) → `ConvNeXtV2-B + CAFormer-B36` (`10.0` Priv) → **2-epoch test-time FDTD pseudo-label FT (`10.0 → 8.93`!)** | [Writeup](https://www.kaggle.com/competitions/waveform-inversion/writeups/hyd-6th-place-solution-summary) |
| 🏅 **#9** | #9 | **`Dieter`** *(1 sub!)* | 11.49003 | **11.47934** | Custom batched CUDA wave kernel (`100×` speedup for online aug) + `CAFormer-B36` / `ConvNeXtV2-H` + **median of last 10 checkpoints** per run | [Writeup](https://www.kaggle.com/competitions/waveform-inversion/writeups/dieter-9th-place-custom-cuda-kernel-for-wave-propa) |
| 🏅 **#10** | #10 | **`monnu + yu4u`** | 11.91336 | **11.93397** | **Per-sample network-weight FWI (`L1(x, F(M_θ(x)))`**, `17.5 → 13.7` Priv!) + 3 rounds FWM PL re-simulation (`27.2 → 17.2 → 12.5`) | [Writeup](https://www.kaggle.com/competitions/waveform-inversion/writeups/monnu-yu4u-10th-place-solution-monnu-part) |
| 🏅 **#14–#15** | #14–#15 | **`Ruby` / `Jun Koda`** | 12.73 / 12.83 | **12.69 / 12.81** | **Gradient-free fixed-point residual opt** $x_{k+1}=x_k-\lambda(F(M(x_k))-x)$ (`17.17 → 12.34` CV, `#14`) & **2-stage physics residual net** `[x, x-F(y1), y1] → δy` (`27.9 → 13.0` LB, `#15`) | [#14](https://www.kaggle.com/competitions/waveform-inversion/writeups/ruby-14th-place-solution) · [#15](https://www.kaggle.com/competitions/waveform-inversion/writeups/jun-koda-15th-place-two-stage-model) |

---

## 1. Universal Scientific Inverse-Problem & Physics-Coupled ML Breakthroughs

1. **Five Universal Ways to Fuse a Known Forward Operator $F: y \mapsto x$ With Deep Learning**:
   - Once you port the exact forward simulator $F$ to `torch.compile`/CUDA (`90–120 samples/s`, `#1, #2, #3, #9, #12`), you unlock five complementary inverse-problem levers:
     1. **Iterative Forward-Simulated Pseudo-Labeling (`#1`, `#6`, `#10`)**: Predict $\hat{y}$ on test inputs, compute exact $x_{\text{sim}} = F(\hat{y})$, and retrain on exact pairs $(x_{\text{sim}}, \hat{y})$ — zero noisy pseudo-label drift because $(x_{\text{sim}}, \hat{y})$ obeys exact physics (`#1` 570 weighted epochs → `6.92`; `#6` 2 epochs → `10.0 → 8.93`)!
     2. **Regularized Physics Test-Time Optimization (`#2`, `#5`)**: Minimize $p(y) + \|F(y) - x\|^2$ starting from a neural warm-start, where **explicit structural priors $p(y)$** (TV $p<1$ for sharp discontinuities, truncated GP SVD for smooth fields, decoupling non-differentiable ops like `min(vel)`) prevent divergence and slash MAE **`28.8 → 7.65`** (`#2`) or `-14.0` on `Style_A` (`#5`).
     3. **Per-Sample Network-Weight Physics TTO (`#10 yu4u`)**: Instead of optimizing raw pixels $y$, optimize network weights $\theta$ per test sample via $\|F(M_\theta(x)) - x\|_1$ (implicit neural prior → `17.5 → 13.7` Priv!).
     4. **Gradient-Free Fixed-Point Residual Iteration (`#14 Ruby`)**: $x_{k+1} = x_k - \lambda(F(M(x_k)) - x)$ cut MAE **`17.17 → 12.34` CV** in 5 forward-only iterations when $F$ is non-differentiable or slow to backprop!
     5. **2-Stage Physics-Residual Correction Network (`#15 Jun Koda`)**: Feed `[x, x - F(y_1), y_1]` into Stage 2 to predict $\delta y = y - y_1$ (`27.9 → 13.0` LB).
2. **Non-Aligned Cross-Domain Vision Architectures (`#1`, `#3`, `#4`)**:
   - When inputs `(sources, time, receivers)` and outputs `(depth, width)` do **not** align pixel-for-pixel, standard UNet skip connections bottleneck performance (`46.9 MAE`). Rearranging gathers into a **single side-by-side spatial canvas `(1, 350, 350)` + pure ViT+RoPE encoder–decoder (`#1`: `46.9 → 32.5`)** or **split intra-channel ViT+RoPE + pixel-shuffle + `MuonWithAuxAdam` (`#3`: `9.x → 7.x`)** unlocked global wavefield-to-velocity translation.
3. **Exact Physics Symmetry & Simulator Forensics (`#2`, `#3`, `#4`, `#5`, `#14`, `#20`)**:
   - **Asymmetric Center Source (`34 → 35`)**: Naive horizontal flipping shifted the center source by 1 px; fixing exact flip symmetry (6-ch sim `#3`, center-only resim `#4`, zero-ch padding `#14`: `-10 MAE`) removed bias.
   - **Exact Simulator Quirks**: Checking `float32` vs `float64` precision (`#3`), 0-based vs 1-based indexing (`2–3×` error drop), and boundary-row bugs (`y=69` worth `8.43 → 7.99` in `#3`) paid huge dividends.
   - **Velocity–Time Scaling Law (`v → αv ⇔ t → t/α`)**: Provided physics-exact synthetic augmentation + **10-alpha median TTA** (`#5, #14, #20`).
4. **MAE Ensembling & Hard-Subfamily Specialization (`#1`, `#4`, `#9`, `#20`)**:
   - Under **L1/MAE**, **exact median ensembling** (`torch.quantile(stack, 0.5)` across checkpoints `#9` or diverse runs `#20`) beats mean ensembling on outlier-prone families (warning: repeating arrays `[a,a,a,b,b,b,c]` for weighted median silently collapses to unweighted `median([a,b,c])`, `#20`).
   - Fine-tuning **hard-subfamily specialists** (`#1` Top-4 families, `#4` Style specialist `+2.0`) broke global training dilution.

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Unregularized Physics FWI ($\min \|F(y) - x\|^2$ Without Priors $p(y)$)** | Diverged or stalled at `~0.1` gain (`#2`, `#20`) | #2, #5, #20 | Ill-posed wave inversion falls into cycle-skipping local minima without TV/GP/neural priors |
| **Standard Pixel-Aligned UNet Resizing `(5, 350, 70)` & Naive 5-Ch HFlip** | Lagged side-by-side ViT+RoPE by `>14 MAE` (`#1`); naive HFlip broke source `34→35` symmetry | #1, #3, #4, #14 | Wave arrival time is not spatially aligned with subsurface depth; odd/even source grids break naive mirror |
| **Dropping Original Data for Pure Synthetic Training** | Degraded simpler benchmark families (`#5`) | #5 | Synthetic generators slightly shift boundary/flat-layer statistics |
| **3rd-Stage Residual UNet, MoE 10-Decoder Routing, & MAE/MIM Pretraining** | Zero LB gain (`#1`, `#15`, `#20`) | #1, #15, #20 | Residual signal saturates after Stage 2; multi-family shared backbones generalize better than MoE routing |
| **Weighted Median via Element Repetition / Selecting Best by Recon Error** | `[a,a,a,b,b,b,c]` collapses to `median([a,b,c])` (`#20`); recon-error argmin lost to `np.median` (`#6`) | #6, #20 | Use proper weighted quantile math; median suppresses local forward-model non-uniqueness |
