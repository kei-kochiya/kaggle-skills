# Bengali.AI Speech Recognition 🏆

> **Track:** Multimodal, Audio & Speech ASR (Research Code) · **Scale:** 744 teams · \$53,000 · Closed Oct 18, 2023  
> **Core Lesson:** Under extreme read-to-spontaneous OOD speech shift (`8.4` train words → `34.4` test words across 18 domains), **iterative transcript noise filtering (`WER < 15%`)**, **custom language-compact Whisper tokenizers** (`+0.040` WER blowout by #1!), **multi-clip concatenation & telephony/noise augmentation**, **decoupled Transformer punctuation restoration (`-0.035` WER)**, and **hidden-state Transformer CTC fusion / KenLM sentence rescoring** (fixing CTC spike misalignment!) crushed naive denoising and raw CTC logit averaging.

---

## 1. Competition DNA & OOD Shift

- **Task:** Transcribe `~20` hours (`~8,000` test MP3s, `32 kHz`) spanning **18 domains** (17 out-of-distribution spontaneous/noisy/dialect/YouTube/telemedicine domains + 1 in-distribution subset; mean **34.42 words/clip**) trained from `~1,200` hours of crowdsourced read speech (`MaCro`, `25.81 GB`; noisy algorithmic `train` split + manually verified `valid` split; mean **8.42–9.21 words/clip**).
- **Metric (`mean_wer`):** `jiwer.wer` (punctuation **preserved & evaluated**, `bnUnicodeNormalizer` applied) computed per domain and unweighted-averaged across domains. Public (~46%) vs Private (~54%) correlated tightly (`r = 0.991`), though Private ran `~0.06–0.08` higher WER on unseen hard domains.
- **Constraints:** `<= 9h` GPU (P100/T4, `~13 GB` CPU RAM), offline inference.

| Private # | Shake | Team | Private WER | Public WER | Core Breakthroughs |
| :---: | :---: | :--- | :---: | :---: | :--- |
| 🥇 **1st** | `—` | **chimege** | **0.372** | **0.312** | **-0.040 WER blowout!** 3-rd `WER<15%` cleaning + 420k GoogleTTS + YouTube PL + **custom 12k Bengali `whisper-medium` tokenizer (`beam=4–8`)** + 70k concat + **4× `MuRIL` punctuation (`-0.035` WER)** |
| 🥈 **2nd** | `—` | **qdv206** | **0.412** | 0.328 | Top-90% WER cleaning + `IndicWav2Vec` (unfrozen encoder, 3-cycle cosine) keeping only `.` & `-` + DNS/MUSAN/RoomSim aug + **3-fold `IndicBERTv2+LSTM` punctuation** |
| 🥉 **3rd** | `↑ 1` | **Camphora** | **0.418** | 0.345 | `valid`-bootstrapped `WER<0.75` filter + length-sorted `IndicWav2Vec` + **conditional Demucs (`+0.004`)** + **`XLM-R-Large+Base` punctuation (`-0.030+`)** |
| 🏅 **4th** | `↑ 2` | **hanx** | **0.422** | 0.351 | Top-70% filter + 3-stage **`wav2vec2-xls-r-1b`** + concat/3-split aug + 7-gram KenLM |
| 🏅 **5th** | `↓ 2` | **benedikt.d** | **0.429** | 0.344 | Dual-model `WER<0.5` filter + **frozen hidden-state concat → Transformer CTC fusion head (`-0.011` WER!)** + vocab-pruned 5-gram KenLM + `XLM-R-Large` punct |
| 🏅 **7th** | `↓ 2` | **ktr** | **0.436** | 0.351 | `chars/sec<=20` + `CER<0.6` filter + Fairseq→HF `IndicWav2Vec` + **3-model N-best KenLM sentence rescoring (`-0.006–0.008`)** |
| 🏅 **10th** | `↑ 4` | **DarioAr** | **0.449** | 0.380 | MOS/WER filter + **80% 2-clip concat** + **custom timestamp-advancing `whisper-medium` loop (`0.43→0.38` Pub!)** |

---

## 2. Data Forensics, Noise Filtering & OOD Audio Augmentation

1. **Aggressive Transcript Noise Filtering (Mandatory for Both Whisper & CTC):**
   - Crowdsourced `train` transcripts contained heavy errors; Whisper collapsed/hallucinated without filtering (`#1`, `#10`), and CTC overfit after 40k steps (`#5`).
   - **Bootstrapped Multi-Round Cleaning:** `#1` trained an initial `whisper-medium` on clean OpenSLR 37/53 + 420k GoogleTTS and ran 3 cleaning rounds keeping **`WER < 15%`** across MaCro/MadASR/Shrutilipi/Kathbath; `#3`/`#4`/`#5`/`#7`/`#10` filtered via `valid`-only bootstrap, `chars/sec <= 20`, NISQAv2 `MOS > 1.5`, and dual-teacher `WER < 0.5–0.75`.
2. **Bridging Short Read Speech → Long Noisy OOD Audio:**
   - **Multi-Clip Concatenation:** Because test clips averaged `4×` train length (`34.4` vs `8.4` words), `#1` (`70k` merged audios: `0.380 → 0.370` Pub), `#2`, `#4`, and `#10` (`80%` 2-clip concat + `20%` single clips to prevent hallucinations) trained directly on stitched utterances.
   - **Telephony & Background Simulation:** `16kHz → 8kHz → 16kHz` resampling (`#1`), `libsonic` speed/pitch, `TimeStretch(0.8–2.0)`, `RoomSimulator`, DNS Challenge 2020 & MUSAN music (`SNR 3–30 dB`), applied **heavily on read speech and lightly on spontaneous corpora** (`#2`).
   - **OOD Pseudo-Labeling (`#1`):** Pseudo-labeling unlabeled Bengali YouTube videos drove Public LB from `0.370 → 0.360 → 0.312`.
3. **Unicode & Word-Boundary Punctuation Forensics:** Applied `bnUnicodeNormalizer` across all corpora. `#2` removed all punctuation *except* `.` and `-` during ASR training — delegating `[।,?!]` to text restoration — because missing word-internal `.` or `-` causes `2×+` word-split/merge WER penalties.

---

## 3. ASR Architectures: Custom-Tokenizer Whisper vs. CTC Backbones

1. **Why `#1`'s `whisper-medium` Crushed Every CTC System (`0.372` vs `0.412` Priv):**
   - **Custom 12k Language-Compact Tokenizer (`#1`):** Default Whisper BPE shattered Bengali into excessive byte tokens, causing 9h timeouts at `beam > 1`. Training a compact 12k Bengali tokenizer unlocked `num_beams=4–8`, `max_length=260`, `chunk_length_s=20.1s` in `< 7h` on P100!
   - **Long-Audio (>30s) Timestamp-Advancing Decoding (`#10`):** Standard 30s truncation looped or dropped words; stepping forward by predicted end/mid timestamps (carrying up to 224 prior prompt tokens) dropped single `whisper-medium` Public WER **`0.43 → 0.38`**.
2. **Best CTC Recipes (`IndicWav2Vec` & `wav2vec2-xls-r-1b`):**
   - Keep feature encoder **unfrozen** (`#2`), sort dynamic batches by length (`#3` — end-padding hurt CTC!), convert clean Fairseq checkpoints to avoid prior-finetune overfitting (`#7`), and use low LR (`1e-5`) with long cosine schedules on `1B` models (`#4`, `#5`).

---

## 4. Decoding, Decoupled Punctuation & Solving CTC Misalignment

1. **Decoupled Transformer Punctuation Restoration (`-0.020` to `-0.035` WER!):**
   - Training dedicated `AutoModelForTokenClassification` models on normalized text corpora (`IndicCorp` + train) predicting `[।,?!-]` dropped `#1`'s WER **`0.360 → 0.325`** (`4× MuRIL` at layers 6/8/11/12), `#3` **`> 0.030`** (`XLM-R-Large+Base`, `PAD` loss weight `0.0`), `#2` (`3-fold IndicBERTv2 + LSTM`, 15% token masking + beam decode), and `#5` (`+0.020`, plus forcing terminal `।`/`?` `+0.002`).
2. **KenLM N-Gram Beam Search Engineering (`pyctcdecode`):**
   - **ARPA > Quantized `.bin` (`+0.007` WER, `#11`):** Quantizing `.bin` degraded unigram scores; pruned 4/5-gram ARPA fit inside `13 GB` RAM.
   - **Missing `</s>` EOS Bug (`-0.023` WER fix, `#22`)** and **HF `Wav2Vec2ProcessorWithLM` C++ RAM Leak (`#5`):** Save CTC logits to disk → unload GPU models → decode from disk → explicitly call `decoder.cleanup()`.
3. **True ASR Ensembling (Fixing CTC Frame Misalignment):**
   - Naive CTC logit averaging fails because separately trained CTC models emit character spikes at slightly shifted frames (`#5`).
   - **Fix A — Hidden-State Fusion + Transformer CTC Head (`#5`, `-0.011` WER):** Concatenate frozen final hidden states across diverse CTC models and train a lightweight Transformer+CTC head for just `7k` steps (`0.355 → 0.344`).
   - **Fix B — Multi-Model N-Best KenLM Sentence Rescoring (`#7`, `-0.006–0.008` WER):** Decode candidates from 3 diverse CTC models and pick the full hypothesis with highest 5-gram KenLM log-probability.

---

## 5. Validation Strategy & Verified Anti-Patterns

- **OOD Validation vs. In-Distribution Trap:** Read-speech `valid` (`0.298` WER) had a `+0.173` gap to Public LB (`0.471`); teams validated LM `alpha`/`beta`, punctuation, and checkpoints on the manually transcribed **17-domain `examples/` OOD set** (`#3`, `#7`, `#11`).
- **Verified Anti-Patterns:**
  1. **Naive Speech Enhancement/Denoising:** `DeepFilterNet`, `FAIR Denoiser`, `CleanUNet`, and `SpeechBrain WHAM` all distorted speech phonemes and **hurt WER** (`#4`, `#7`, `#11`, `#31`). Only **conditional Demucs** (accepting separated output *only* if predicted token count `>=` raw) gained `+0.004` (`#3`).
  2. **Naive CTC Logit Averaging (`#5`)** and **Training Whisper on Unfiltered Crowdsourced Labels (`#1`, `#10`)**.
  3. **Default Non-English Whisper Tokenizer + Naive 30s Chunking (`#1`, `#10`)** and **`wav2vec2-xls-r-300m` / `2b` or Cross-Language Assamese Mixing (`#4`, `#7`)**.

---

## 6. Verified Solution Sources

- **1st Place (`chimege`):** `https://www.kaggle.com/competitions/bengaliai-speech/writeups/chimege-1st-place-solution`
- **2nd Place (`qdv206`):** `https://www.kaggle.com/competitions/bengaliai-speech/writeups/qdv206-2nd-place-solution`
- **3rd Place (`Camphora`):** `https://www.kaggle.com/competitions/bengaliai-speech/writeups/camphora-3rd-place-solution`
- **4th Place (`hanx`):** `https://www.kaggle.com/competitions/bengaliai-speech/writeups/hanx-4th-place-solution`
- **5th Place (`benedikt.d`):** `https://www.kaggle.com/competitions/bengaliai-speech/writeups/benedikt-d-5th-place-solution-ensembling-works`
- **7th Place (`ktr`):** `https://www.kaggle.com/competitions/bengaliai-speech/writeups/ktr-8th-place-solution`
- **10th Place (`DarioAr`):** `https://www.kaggle.com/competitions/bengaliai-speech/writeups/darioar-11th-place-solution`
- **11th Place (`DT_shakeup`):** `https://www.kaggle.com/competitions/bengaliai-speech/writeups/dt-shakeup-11th-place-solution`
