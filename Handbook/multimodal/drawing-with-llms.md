# Drawing with LLMs: Top-10 Solutions & Differentiable Vectorization Post-Mortem

> **Competition**: [Drawing with LLMs](https://www.kaggle.com/competitions/drawing-with-llms) (Feb – May 2025, 1,309–1,313 Teams, \$50,000 Prize Pool; Kaggle's first **Kaggle Packages** competition)  
> **Track**: Multimodal Text-to-SVG Generation, Reward-Aligned Diffusion, Differentiable Vector Graphics (`diffvg`) & VLM Proxy Optimization  
> **Contract & Constraints**: Offline Kaggle Package (`<= 9h` total on `2× T4` / `P100`, `<= 5 min`/prompt across `~500` short text prompts; strict **`<= 10,000` bytes** valid allowlisted SVG, no `<style>` or raster `<image>`, rendered via `cairosvg`)  
> **Evaluation Metric (`SVG Image Fidelity` v17)**: Weighted harmonic mean ($\beta = 2$, weighting **VQA `2×`** Aesthetic) between **PaliGemma 2 (`10b-mix-448`) TIFA VQA × Anti-OCR Penalty** (`>4` detected chars penalized exponentially *before* filtering) and **CLIP-ViT-L/14 Aesthetic Score** evaluated *after* `ImageProcessor` (`random crop/resize → JPEG → median blur → FFT low-pass → bilateral filter`)

---

## 1. Official Top-10 Leaderboard & The Clean vs. Metric-Asymmetry Split

Direct LLM SVG generation lagged far behind (even frontier `Claude-3.7-Sonnet` scored `0.6705` vs. `0.769–0.791` for top pipelines). The final leaderboard split between **metric-asymmetry exploits** (#1, #2 — taking advantage of OCR running *before* `ImageProcessor` while VQA ran *after*) and **genuine zero-exploit diffusion + differentiable SVG engineering** (#3 `CP GMs` at `0.76947`, #4 `Kevin Clark` DRaFT-LV at `0.74314`, #5 `tomirol` at `0.73168`, #8–#12).

| Private Rank | Team | Private LB | Public LB (Δ) | Paradigm & Core Innovations | Writeup / Code |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **1st** | **new team name** (`conor`, `kenyeung.tech`, `Filtered`) | **0.79058** | 0.79038 (0) | `SSD-1B + taesdxl + DeepCache` (`64` imgs/30s) + 6-color OpenCV SVG (6KB) + `<pattern>`-masked prompt text exploiting pre/post `ImageProcessor` asymmetry (`VQA ~0.88`) + GA AES artifacts + `"ZOK"` decoy | [Writeup](https://www.kaggle.com/competitions/drawing-with-llms/writeups/new-team-name-1st-place-solution) · [Notebook](https://www.kaggle.com/code/conormacamhlaoibh/new-team-name-1st-place-solution) |
| 🥈 **2nd** | **move on** (`Yurnero`, `A. Poyda`, `takuji`, et al.) | **0.77776** | 0.77187 (+1) | Pure non-drawing SVG text-masking (`<pattern>`/`translate`) + A100 exhaustive AES artifact search (`0.652 → 0.77776`) | [Writeup](https://www.kaggle.com/competitions/drawing-with-llms/writeups/move-on-2nd-place-solution) · [Notebook](https://www.kaggle.com/code/takuji/2nd-place-solution-ocr-attack) |
| 🥉 **3rd** | **CP GMs** (`c-number`, `daiwakun`) | **0.76947** | 0.77523 (-1) | **Top Zero-Exploit Solution (`VQA 0.81 / AES 0.64`)**: Nunchaku `FLUX.1-schnell` + `vtracer` + SigLIP + **200-iter `pydiffvg` straight-line polygons backpropagated through differentiable `ImageProcessorTorch`** | [Writeup](https://www.kaggle.com/competitions/drawing-with-llms/writeups/cp-gms-3rd-place-solution-vqa-aes-0-81-0-64-diffus) · [Notebook](https://www.kaggle.com/code/cnumber/fork-of-flux-baseline-75c113?scriptVersionId=242056669) |
| 🏅 **4th** | **Kevin Clark** | **0.74314** | 0.74108 (+2) | **SD3.5-Medium (no T5) + DRaFT-LV LoRA** trained on **5 rewards** (incl. `-LPIPS(img, SVG)`) (`+0.03` LB) + heuristic SVG + `diffvg` L1 refinement + OCR decoy | [Writeup](https://www.kaggle.com/competitions/drawing-with-llms/writeups/kevin-clark-4th-place-solution-sd3-5m-draft-diffvg) · [GitHub](https://github.com/clarkkev/svg-diffusion) |
| 🏅 **5th** | **tomirol** (`matheus`) | **0.73168** | 0.74255 (-1) | **3-process async pipeline** (`cuda:0` SDXL-Turbo → `cpu` VTracer → `cuda:1` PaliGemma2; `28` cands/`55s`) + universal `96×96` DiffVG AES patch (`~0.7` AES) | [Writeup](https://www.kaggle.com/competitions/drawing-with-llms/writeups/tomirol-5th-place-solution-vtracer-and-diffvg-opti) · [GitHub](https://github.com/matheuspf/draw) |
| 🏅 **8th** | **Psiilu** (`Psi`, `philippsinger`) | **0.71961** | 0.73005 (+3) | Fine-tuned SDXL distilled from 16 `Flux Schnell + SDXL` top-scoring runs + `VTracer` + `scour` + `"+"` decoy + PaliGemma binary proxy | [Writeup](https://www.kaggle.com/competitions/drawing-with-llms/writeups/psiilu-8th-place-solution) |
| 🏅 **9th** | **linrock** | **0.71473** | 0.71924 (+4) | SDXL Base + TinyVAE LoRA (`offset_noise=0.0357`, 24-color filtered Flux/Hyper-SDXL dataset) + DE frame/gradient + 3 binary VQA prompts | [Writeup](https://www.kaggle.com/competitions/drawing-with-llms/writeups/linrock-9th-place-sdxl-lora-fine-tune) |
| 🏅 **10th** | **BERA** (`B. Droste`, `Raki`) | **0.71389** | 0.73824 (-3) | `Flux.1-Dev + HyperSD (8 steps)` via Nunchaku + 4-bit BnB (`10` imgs/`20s`) + `vtracer` + `svgo` + `2px` colinearity pruning + `"M"` decoys | [Writeup](https://www.kaggle.com/competitions/drawing-with-llms/writeups/bera-10th-place-solution-flux-1-dev-svg-optimizati) |
| **12th** | **Rock** | **0.71292** | 0.72911 (0) | Online **Flow-GRPO RL** on `SD3.5M` (`4× H800`, 500 steps, exact harmonic reward: local VQA **`0.82 → 0.91`**, total `0.72 → 0.77`) | [Discussion](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581034) · [GitHub](https://github.com/yanyanhuang/12th-Place-Solution-for-Kaggle-Drawing-with-LLMs) |

---

## 2. Core Multimodal & Generative Engineering Breakthroughs

### 2.1 Reward-Aligned Diffusion Fine-Tuning (`DRaFT-LV`, `Flow-GRPO`, Distilled SFT)
- **DRaFT-LV Backpropagation Through SVG Conversion (#4 Kevin Clark)**: Fine-tuned a LoRA on **SD3.5-Medium (dropping T5 to save VRAM/time)** in `18h` on 1× A100 across 1,000 GPT-4o prompts by backpropagating **5 rewards**: `HPSv2 + PickScore + CLIP Aesthetic + binary VQA ("Does the image show {prompt}?") - LPIPS(raw_image, convert_to_svg(raw_image))`. Optimizing directly for SVG-friendly flat imagery boosted AES **`+~10%`**, VQA **`+~5%`**, and LB **`+~0.03`** without brittle prompt templates.
- **Online `Flow-GRPO` RL (#12 Rock)**: Trained `SD3.5M` (`512×512`, `16` rollouts/prompt, `KL=4e-4`, 500 steps on 4× H800) directly against the competition's exact $\beta=2$ harmonic mean reward, lifting local VQA from **`0.82 → 0.91`** (`0.72 → 0.77` total).
- **Distilled SFT on Vector-Style Images (#8 `Psiilu`, #9 `linrock`)**: Generating thousands of Flux/SDXL candidates, quantizing to 24 colors + median/Gaussian filtering, and SFT-distilling into fast SDXL LoRAs (`8` steps `DPMSolver++`) gave simple, high-throughput, vector-ready renders when RL/DPO failed to stabilize.
- **Ultra-Fast Diffusion Inference on T4**: Combining `SSD-1B + taesdxl + DeepCache` (`64` images/`30s`, #1) or **Nunchaku + HyperSD 8-step + 4-bit BnB** on `Flux.1-Dev/schnell` (`10` images/`20s`, #3, #10) opened massive candidate search budgets inside 5 min/prompt.

### 2.2 Vectorization, 10KB Byte-Budgeting & Differentiable `pydiffvg` Through the Defense Chain
- **Fast Vectorization & Binary-Searched Resolution**: Rust-based **`VTracer`** (`<2s` single-core) crushed Go `primitive` (`>30s` for 50 paths, #5). Top teams (#3, #10, #13, #15) ran **binary search over bitmap resolution / VTracer precision** right up to `10,000` bytes, combined with relative path conversion, `H`/`V` shorthand, `translate` elimination (`+~50px` extra resolution in #13), same-fill merging, `2px` colinearity pruning (#10), and `scour`/`svgo` minification.
- **Differentiable `ImageProcessorTorch` + Straight-Line `pydiffvg` (#3 `CP GMs`, `0.76947`)**:
  - Re-implemented the entire evaluator defense chain (`crop/resize → diff_jpeg → median → FFT low-pass → bilateral`) differentiably in PyTorch!
  - Ran `10`-iter `pydiffvg` pre-screening on top-2 SigLIP candidates, then **`200`-iter Adam + cosine warmup** optimizing **straight-line polygon vertices, fills, and strokes** (explicitly avoiding Bézier curves, which oscillated/diverged!) against `Aesthetic + SigLIP2 + MSE(ImageProcessorTorch(SVG), target)`, lifting **AES from `<0.5` to `0.64`** while holding **VQA at `0.81`** (`~63s`/prompt).
- **Universal Aesthetic Patch (#5 `tomirol`)**: Optimized a reusable `~3.5 KB` `96×96` SVG patch via `diffvg` over 150 training backgrounds (placed away from edges to survive random crop), pushing AES to `~0.7` while keeping `~6 KB` for VTracer.

### 2.3 Proxy Reranking, Async Multi-GPU Pipelining & Anti-OCR Forensics
- **Single Binary VQA Question > Complex Synthetic TIFA Q&A (#4, #5, #8, #9, #15)**: Querying PaliGemma 2 `10b-mix-448` with a single `Yes/No` probability (`"Does this image show {prompt}?"`) beat generating 4 synthetic Qwen-2.5 questions at test time (#15 gained `+23` ranks Private by splitting prompts into 3 short binary chunk questions).
- **3-Process Async GPU–CPU–GPU Pipeline (#5)**: Overlapped `cuda:0` (SDXL-Turbo) → `cpu` (VTracer) → `cuda:1` (PaliGemma 2) asynchronously to evaluate **`28` SVG candidates in `~55s`/prompt**.
- **Anti-OCR Hallucination Anchor & Benchmark Design Lesson**:
  - Benign 1–3 char corner decoys (`"ZOK"`, `"M"`, `"+"`, `"a"`) prevented PaliGemma OCR from hallucinating `>4` phantom characters in noisy textures.
  - **Evaluation Pipeline Vulnerability (#1, #2)**: Running OCR *before* `ImageProcessor` while running VQA *after* `ImageProcessor` created a physical filter-inversion loophole: low-opacity `<pattern>` overlays hid prompt text from pre-filter OCR while JPEG/blur/FFT low-pass filtering smoothed the pattern away to reveal the text to post-filter VQA. *Lesson for multimodal benchmark design*: never run defensive filtering between an anti-cheat detector and the downstream VLM scorer.

---

## 3. Empirical Negative Results & What Failed

| Tested Hypothesis | Observed Outcome | Root Cause |
| :--- | :--- | :--- |
| **Direct LLM SVG XML Generation (`Claude-3.7`, `Qwen2.5-14B`, `R1`)** | Capped at **`0.6705`** (`AES 0.451`) vs. `0.769+` diffusion | Autoregressive XML tokens cannot efficiently match visual composition, spatial layout, and aesthetic manifolds under 10KB. |
| **Optimizing Bézier Curves in `pydiffvg` (#2, #3)** | Diverged / oscillated badly | Higher-order Bézier control gradients are ill-conditioned; straight-line polygons converged cleanly (fine curves wash out under blur/JPEG anyway). |
| **Direct `diffvg` Gradients Through PaliGemma VQA (#5)** | Failed / OOM'd on 16GB T4 (`ReNO` only fit 3B-224) | 10B VLM backprop exceeded T4 VRAM and gave noisy gradients vs. fast `SigLIP2 + CLIP Aesthetic` differentiable proxies. |
| **RL / DPO Diffusion Fine-Tuning on Small Prompts (#3, #4, #8)** | Abandoned; SFT & DRaFT-LV strictly superior | Offline DPO/RL lacked sample efficiency on `~2k` prompts compared to differentiable reward backprop (`DRaFT-LV`, #4) or large-cluster online `Flow-GRPO` (`4× H800`, #12). |
| **Same Prompt Sent to CLIP & T5 Dual Encoders (#10)** | Degraded Flux generation quality | CLIP expects concise comma-separated keywords whereas T5 expects fluent natural-language sentences. |
| **Synthetic Qwen 4-Question TIFA Proxy at Inference (#4, #5, #9)** | Worse than simple `"Does this image show {prompt}?"` | Noisy LLM-generated questions added latency and variance compared to direct binary yes/no logit scoring. |
| **Basic Geometric Primitives in Diffusion (#4, #10, #18)** | Only `~10%` success on trapezoids/pentagons across 12+ models | Diffusion models struggle with exact polygon counts; hybrid LLM-switcher overhead lost to extra diffusion candidate reranking. |

---

## 4. Universal Takeaways Promoted / Reinforced

- **Differentiable Surrogates of Metric Preprocessing Chains**: Whenever an evaluation metric applies non-differentiable image/audio filters (JPEG, blur, crop, FFT) before scoring, building a differentiable PyTorch mirror (`ImageProcessorTorch`) unlocks end-to-end gradient optimization aligned with the true post-filter score.
- **Direct Reward Backprop (`DRaFT-LV`) + Loss-Matched Format Penalty (`-LPIPS(img, converted)`)**: Backpropagating downstream format-conversion fidelity directly into generative fine-tuning aligns raw generations with strict output budgets (SVG bytes, quantization, etc.).
- **Async Producer–Consumer `GPU0 → CPU → GPU1` Pipelining**: Overlap heavy CPU post-processing (vectorization, decoding, meshing) asynchronously between GPU generation (`cuda:0`) and GPU VLM reranking (`cuda:1`) to multiply test-time candidate search budgets.
- **Simple Binary VLM Logit Scoring > Synthetic Multi-Question VQA Chains**: For test-time text-image alignment reranking, calibrated binary `"Does this show {prompt}?"` (or chunked binary prompts) is faster and lower-variance than generating multi-step Q&A at inference.
