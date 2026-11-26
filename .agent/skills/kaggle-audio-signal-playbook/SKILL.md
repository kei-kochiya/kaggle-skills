---
name: kaggle-audio-signal-playbook
description: >-
  Universal competitive audio, bioacoustics, and 1D-to-spectrogram signal classification playbook for Kaggle. Use when building mel-spectrogram SED pipelines, distilling audio foundation models (Perch/AudioProtoPNet) into fast CNNs, executing domain-shift-safe iterative soft pseudo-labeling on unlabeled field recordings, rank-blending miscalibrated families, or compiling shared-spectrogram OpenVINO/ONNX CPU pipelines.
---

# Kaggle Audio & Signal Classification Playbook

A universal competitive workflow for multi-label audio, bioacoustics, and spectrogram-based signal competitions under domain shift and tight CPU inference limits. Competition-specific species lists, sites, and external corpus names remain strictly in `Handbook/multimodal/`.

---

## 1. Phase 1: Target-Domain Validation Under Capture Shift

Audio competitions routinely train on clean, isolated **focal recordings** while evaluating on noisy, overlapping, multi-source **long-form field recordings**:

1. **Validate Strictly on Target-Domain Recordings**:
   - Never rely solely on focal-clip CV to tune pseudo-labeling or post-processing.
   - Combine two complementary target-domain splits when labeled target recordings are scarce:
     - **Unseen-Site / Unseen-Session Holdout**: Hold out entire recording sites/devices to measure true spatial generalization.
     - **Greedy Rare-Class Coverage Split**: Maximize positive classes present in validation while keeping $\ge 1$ sample per class in train.
2. **Audit Zero-Shot / Zero-Focal Classes Early**:
   - Verify whether any target classes have zero samples in the primary training audio and exist *only* in target-domain recordings; isolate those classes for dedicated pseudo-labeling, sampling, or specialist heads.

---

## 2. Phase 2: Two-Pillar Modeling (Foundation Distillation + Zero-KD Diversity)

Refer to [`references/audio_distillation_pl_serving.md`](./references/audio_distillation_pl_serving.md) for implementations.

1. **Pillar 1 — Two-Phase Cosine Embedding Distillation into Fast SED CNNs**:
   - When a strong audio foundation model (e.g., `Perch`, `AudioProtoPNet`, `BEATs`) is too slow to ensemble directly under CPU limits, distill its embeddings into lightweight 2D spectrogram backbones (`ECA-NFNet-L0`, `EfficientNetV2-S`, `RegNetY`, `SEResNeXt26`, `HGNetV2`) topped with Sound Event Detection (**SED**) attention heads:
     - **Phase 1**: Train a linear projection head with **Cosine Embedding Loss** (outperforms MSE convergence) across focal + unlabeled field recordings; re-distill independently per backbone/seed to preserve diversity.
     - **Phase 2**: Fine-tune end-to-end with `CrossEntropy` / `BCE` / `Focal BCE` + iterative soft pseudo-labeling.
   - Distill from **multiple distinct foundation teachers** when available to decorrelate student representations.
2. **Pillar 2 — Deliberate Zero-KD Diversity + Native Foundation Anchor**:
   - Distillation raises correlation between students and the teacher. Always keep **non-distilled pretrained spectrogram CNNs/Swin models** trained with direct ranking/classification losses (`softAUC + 0.25 BCE`) so cross-family Spearman correlation stays low (`ρ ~ 0.4–0.6`).
   - Retain the **native foundation model** (or a lightweight probe/SSM on its embeddings) at `20%–40%` **rank-blended weight** so rare focal classes forgotten during pseudo-labeling remain protected.
3. **Hierarchical & Minority-Taxon Specialists**:
   - Train auxiliary **parent-hierarchy heads** (e.g., genus/group max-label SED spread back to child classes) and dedicated **minority-taxon specialists** when one macro group dominates training clips.

---

## 3. Phase 3: Domain-Shift-Safe Iterative Soft Pseudo-Labeling

Uncapped pseudo-labeling collapses distilled models and overfits Public LB. Enforce four guardrails:

1. **Non-Overlapping Batch Injectors with Label-Sum Capping**:
   - Mix labeled target recordings (`~15%–20%` of batch) and unlabeled pseudo-labeled recordings (`~20%–75%`) onto **non-overlapping focal samples**, and **cap/normalize each injected target's label sum** (e.g., `≤ 0.5` for labeled target chunks, `≤ 0.75` for pseudo-labeled chunks) so frequent background classes cannot hijack gradient mass.
2. **Inverse-Sqrt Site/Domain Balancing**:
   - Sample unlabeled recordings proportional to $1 / \sqrt{N_{\text{site}}}$ so dominant recording locations do not drown out minority habitats.
3. **Soft Probabilities + Confidence Gating + 2-Round Saturation Cap**:
   - Always use **soft probabilities** (never hard 0/1 thresholds) with precision-calibrated confidence gating (keep clips with $\max_c p_c \ge \tau_{\text{high}}$, zero tail noise $p_c < \tau_{\text{low}}$).
   - Monitor saturation tightly: **2–3 rounds** is usually optimal; audit whether raw clean models outperform over-iterated PL on unseen sites.
4. **Zero-Init Residual Co-Occurrence Adapter**:
   - As a low-risk alternative/complement to heavy PL, train a **zero-initialized residual class-interaction MLP strictly on clean unmixed target-domain labels** to capture real-world co-occurrence without corrupting backbone features.

---

## 4. Phase 4: Audio Frontends, Augmentations, Post-Processing & CPU Serving

1. **Coupled Mel-Spectrogram Frontends & Augmentations**:
   - Match the exact `(sr, n_fft, hop_length, n_mels, fmin, fmax, power)` of any pretrained audio checkpoint; diversify `n_mels` (`128` vs `192` vs `256`) across ensemble members.
   - Core augmentations: waveform **MixUp** (`p=0.5`), mild **SpecAugment**, and **`RandomFiltering`** (STFT → multi-band piecewise-linear EQ `[-20, 0]` dB → iSTFT) to simulate microphone/distance frequency roll-off.
2. **Per-Column Percentile Rank Blending & Temporal Context**:
   - Never average raw probabilities across foundation probes, distilled SED CNNs, and pure CNNs—always **percentile-rank normalize per class column** before blending.
   - **Recording-Level Anchoring & Class-Conditional Temporal Smoothing**:
     - Pull chunk scores gently toward the parent recording max (`file_max` / `top2_mean`), or feed all chunks of a full recording into a **2nd-stage BiGRU/LSTM sequence stacker**.
     - Match temporal smoothing width to event physics: wider kernels (`[0.33, 0.33, 0.33]`) for continuous/stationary sources vs. narrower kernels (`[0.2, 0.6, 0.2]`) for transient events, or uncertainty-adaptive blending `alpha = α_0 * (1 - max_prob)`.
     - **Caution on Site/Time Priors**: Shrink empirical priors with Bayesian smoothing `w = N / (N + K)` in logit space, and verify they still help once ensemble AUC exceeds `0.95` (priors often turn net-negative on ultra-strong models).
3. **Strict CPU Inference Engineering**:
   - Compile CNN/Swin branches into a **single shared-spectrogram OpenVINO IR graph** (raw waveform → 1 shared mel computation → parallel model branches → weighted output) to eliminate redundant CPU FFT/mel overhead.
   - Partition multi-vCPU budgets cleanly (e.g., **2 parallel lanes × 2 threads** running ONNX Runtime + OpenVINO concurrently) rather than letting one process thrash all cores.

---

## 5. Phase 5: Out-of-Domain Speech ASR (`Whisper` + CTC + KenLM)

When the task is sequence-to-sequence speech transcription (`WER`/`CER`) under read→spontaneous OOD shift (see Section 5 of [`references/audio_distillation_pl_serving.md`](./references/audio_distillation_pl_serving.md)):
1. **Iterative Transcript Cleaning & OOD Length Simulation**: Bootstrap clean-data filtering (`WER < 15%` for Whisper; `WER/MOS/chars-per-sec` for CTC), concatenate clips to match test utterance length, simulate `16k→8k→16k` telephony + reverb/music noise, and validate on a target-domain OOD set (not clean read speech).
2. **Custom Compact Whisper Tokenizers & Decoupled Punctuation**: Train a compact language-specific Whisper tokenizer (`~12k` vocab) on non-English scripts to unlock beam search inside tight GPU caps, use timestamp-advancing `>30s` decoding, and decouple speech recognition from a dedicated **Transformer token-classification punctuation restorer** (`-0.02` to `-0.035` WER).
3. **True CTC Ensembling & KenLM Hygiene**: Never average raw CTC logits directly (spike misalignment)—fuse via **frozen hidden-state concat → tiny Transformer+CTC head** or **multi-model N-best KenLM sentence rescoring**; prefer pruned **ARPA > `.bin`**, verify `</s>` EOS, flush logits to disk + call `decoder.cleanup()` against C++ RAM leaks, and avoid unconditional neural speech denoisers (distort phonemes).

---

## 6. Universal Audio Anti-Patterns

- **Heavy Vision Transformers (`ViT`, `ConvNeXt-Large`, `PaSST`) under tight CPU caps**: Poor accuracy-per-CPU-second compared to OpenVINO-compiled `EffNetV2-S`, `ECA-NFNet-L0`, `RegNetY`, `SEResNeXt26`.
- **Unfiltered massive external audio dumps**: Domain/microphone mismatch and unverified taxonomy synonyms silently corrupt calibration; audit label joins and cap per-class external volume.
- **Applying identical post-processing blindly across all model families**: Tune smoothing and calibration separately per model family on target-domain CV.
- **Unconditional neural speech enhancement/denoising (`DeepFilterNet`, `CleanUNet`)**: Frequently distorts human speech phonemes and degrades ASR WER unless gated conditionally.

---

## 7. References & Case Studies
- [Cosine Embedding Distillation, Capped Soft PL, Rank Blending, Shared-Mel OpenVINO & OOD Speech ASR](./references/audio_distillation_pl_serving.md)
- Case Studies: [`BirdCLEF+ 2026`](../../../Handbook/multimodal/birdclef-2026.md) · [`Bengali.AI Speech ASR`](../../../Handbook/multimodal/bengaliai-speech-asr.md)
