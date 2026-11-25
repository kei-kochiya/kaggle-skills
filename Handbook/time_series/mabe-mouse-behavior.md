# MABe Challenge — Social Action Recognition in Mice — Top Gold Solutions & Post-Mortem

> **Competition**: [MABe Challenge - Social Action Recognition in Mice](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection) (Sep 18 – Dec 15, 2025 | 1,412 Teams | 30,127 Submissions | \$50,000 Prize Pool)  
> **Task**: Multi-agent temporal action segmentation (`[start_frame, stop_frame, agent_id, target_id, action]`, **37 classes**: 11 solo + 26 pair) from heterogeneous 2D top-down pose-tracking keypoints across **21 labs** (`10–120 FPS`, `4–18` bodyparts, `11.7–65 px/cm`)  
> **Evaluation Metric**: Lab- & video-averaged **Frame-Level F1 (`beta=1`)** masked strictly to each video's provided `behaviors_labeled` whitelist  
> **Dataset**: `train` (`8,789` videos, `2.84 GB`: **~90% unlabeled** `MABe22` + `848` labeled videos across 15 test labs + train-only `CalMS21`/`CRIM13`) | Hidden Test: `~200` videos across exact **15 test labs** (`32%` Pub / `68%` Priv; `<= 9h` GPU/CPU notebook)

---

## Official Top Gold Leaderboard Standings & Solution Index

First place (`anon`) pulled off a **`+0.044` Private blowout** via **large-scale JAX self-supervised future-motion pretraining on the ~90% unlabeled `MABe22` pose clips**. Across `#2–#15`, gold solutions converged on **uniform 30-FPS + egocentric canonicalization**, **3-state `behaviors_labeled` loss masking**, **GNN/ST-GCN/Squeezeformer/TCN + GBDT ensembles (`+0.025` cross-family lift!)**, and **per-(lab, action) threshold + tie-breaking post-processing**:

| Private Rank | Public Rank | Team | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **#1** | **`anon`** | **0.58469** | **0.57473** | **2-stage JAX pure NN**: (1) **Self-supervised future-motion (speed+dir) pretraining** across all pose data incl. unlabeled `MABe22`; (2) joint `(agent, target)` bodypart message-passing temporal model ensembled over multi-FPS & keypoint subsets | [Summary (`665549`)](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/discussion/665549) |
| 🥈 **#2** | #8 | **`tenten`** | 0.54295 | **0.53049** *(+6)* | Fixed `AdaptableSnail` 25→30fps label-scaling & ID-swap bug (**`+0.009` Priv: `0.521 → 0.530`!**); separate solo/pair `T=512` 1D-Conv/BiGRU/Transformer + **5 Squeezeformers** (`EMA 0.999`, boundary-32 crop SWI + Flip TTA) | [Writeup](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/writeups/2nd-place-solution) · [NB](https://kaggle.com/code/takatoyoshikawa/ensemble-mabe-tenten-submit) |
| 🥉 **#3** | #6 | **`Youri Matiounine`** | 0.54880 | **0.52901** *(+3)* | Multi-layer LSTM (`masked softmax` + lab emb, 5 common keypoints) blended w/ tuned per-(lab, action) XGBoost + `argmax(p / thr)` decoding | [Writeup](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/writeups/3rd-place-solution) · [NB](https://www.kaggle.com/code/ymatioun/mabe-blend-ym) |
| 🏅 **#4** | #11 | **`The Social Gnaw-twork`** | 0.53986 | **0.52758** *(+7)* | Per-(lab, action) XGB + `(B, F, M, M, T)` `InstanceNorm3d` dilated residual NN (`all-lab → per-lab FT`); **selective NaN interpolation ONLY on 4 sparse labs** (avoiding ID-switch corruption); `argmax(p - thr)` | [Writeup](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/writeups/4th-place-solution-xgb-nn-ensemble) · [NB](https://www.kaggle.com/code/aerdem4/mabe-final-ensemble-f9e8a0-d41dff) |
| 🏅 **#5** | #12 | **`dan4o`** | 0.53889 | **0.52633** *(+7)* | Unified **6-point egocentric master skeleton** (`pix_per_cm`, zero raw coords) → **`TransformerConv` GNN → 4× Egocentric Squeezeformer → RelationNetwork** (`Focal γ=2` + duration weighting; 23-model blend) | [Writeup](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/writeups/5th-place-gnn-egocentric-squeezeformer) · [GitHub](https://github.com/GosUxD/MABe-5th) |
| 🏅 **#7** | #2 | **`Chris Deotte`** | 0.56123 | **0.52311** | **3-state (`1/0/MASK`)** multi-scale dilated 1D-CNN (`w=64..512`) + Cross/Self-Mouse Attention (`10 NNs` `0.518` Priv) stacked w/ XGB (`0.508` Priv) → **`0.523` Priv (`+0.005`)** | [Writeup](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/writeups/7th-place-gold-cnn-transformer-with-invariant-fe) · [NB](https://www.kaggle.com/code/cdeotte/7th-place-gold-mouse-comp-cv-0-56-lb-0-56) |
| 🏅 **#9** | #9 | **`Pavel`** | 0.54047 | **0.51692** | 4-block dilated TCN (`0.490` Priv) + 3-GBDT ensemble (`0.492` Priv) → **nested hill-climb blend (`0.516` Priv, `+0.025`!)** + **offset-rotated `stride=7` inference (`7×` faster, `+0.003`)** + **paired-action tournament (`p1 - p2`)** | [Writeup](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/writeups/9th-place-gold) · [GitHub](https://github.com/PavelSavchenkov/kaggle-gold-9th-place-mabe) |
| 🏅 **#10** | #7 | **`Ebi`** | 0.54327 | **0.51579** | **4× ST-GCN + Keypoint Attn → RoPE Transformer → Cross-Pair Attn** (`4/5/7kp` ensemble), `-inf` masked **`CE → MacroSoftF1` annealing** | [Writeup](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/writeups/10th-place-solution-st-gcn-transformer) |
| 🏅 **#12** | #13 | **`SnapPyLemuR`** | 0.53124 | **0.50957** *(+1)* | 5-family tabular+NN blend (`XGB + CB + PyBoost + TabM + LGBM`) w/ *Real* vs *Maybe* negative pools & 2-step OOF Platt prior calibration | [Writeup](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/writeups/12th-place-solution) · [GitHub](https://github.com/T0chka/mabe_12th_place_solution) |

---

## 1. Universal Multi-Agent Pose & Temporal Action Segmentation Breakthroughs

1. **Self-Supervised Future-Motion Pretraining on Unlabeled Pose (`#1 anon`, `+0.044` Gap!)**:
   - With `~90%` of training videos unlabeled (`7,926` `MABe22` clips), `#1` avoided noisy pseudo-labeling (`#10` confirmed PL failed) and instead pretrained a large JAX pose network to predict **future velocity (speed + direction)** across all trajectories before supervised fine-tuning — creating an uncatchable `+0.044` Private margin!
2. **Heterogeneous Multi-Lab Canonicalization & 3-State Partial Supervision (`#2`–`#15`)**:
   - **Temporal + Spatial + Schema Alignment**: Resample heterogeneous `10–120 FPS` videos to uniform **`30 FPS`** (feeding `fps_scale` as a feature, `#2`, `#4`, `#10`), normalize by physical `pix_per_cm` or agent body length, map heterogeneous `4–18` keypoint schemas onto a unified **5–8 point master skeleton** (`nose←head`, `side←hip/lateral`), and project into an **egocentric / agent-centric coordinate frame** (`origin = ears/body_center`, `tail_base → -y`, zero raw coordinates, `#5`, `#7`, `#9`, `#12`, `#15`).
   - **3-State Partial Label Masking**: When each recording only annotates a whitelist subset (`behaviors_labeled`), strictly mask unlisted triplets from loss (`1 / 0 / MASK`, `#7`, `#4`) or set `logits[invalid] = -inf` before softmax (`#10`), and evaluate CV strictly on the **15 test-set labs** (keeping train-only `CalMS21`/`CRIM13` exclusively in train folds, `#2`, `#7`, `#10`).
   - **Data Forensics (`AdaptableSnail`, `#2`)**: Catching 25-FPS videos whose annotations were accidentally scaled to 30 FPS (`× 30/25`) and ID-swapped alone added **`+0.009` Private LB (`0.521 → 0.530`)**!
3. **Graph-Temporal Architectures + Massive Cross-Family NN/GBDT Synergy (`#5`, `#7`, `#9`, `#10`)**:
   - Modeling cage-wide social dynamics via **`TransformerConv` GNN + Egocentric Squeezeformer + RelationNetwork (`#5`)** or **ST-GCN + RoPE Transformer + Cross-Pair Attention (`#10`)** outperformed flat multi-mouse channel concatenation.
   - **Massive GBDT + Neural Complementarity (`#7`, `#9`)**: On `#9`'s controlled ablation, 3-GBDT ensemble (`0.492` Priv) + 6-TCN ensemble (`0.490` Priv) jumped to **`0.516` Priv (`+0.024–0.026`!)**; `#7` similarly gained `+0.005` Priv blending 10 multi-scale CNN-Transformers (`0.518`) with stacked XGB (`0.508`).
4. **Sliding-Window Boundary Hygiene, Strided Inference & Margin Tie-Breaking (`#2`–`#15`)**:
   - Discard `32` noisy edge frames on both ends of `T=512` sliding windows before overlap-averaging (`#2`); **offset-rotated `stride=7` inference + linear interpolation** ran **7× faster** while smoothing predictions (`+0.001–0.003`, `#9`).
   - Tune **per-(lab, action) thresholds** via nested CV (`#9`) or fold-averaged OOF (`#2`), breaking multi-action ties by margin (`argmax(p - thr)` `#4`, `argmax(p / thr)` `#3`, z-score `#15`, or **paired-action `p1 - p2` tournament** `#9` `+0.002–0.004` CV).

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Blindly Interpolating Missing Keypoints Across All Labs** | Hurt local CV (`#4`, `#5`) | #4, #5 | Interpolating across tracker ID-switches creates synthetic teleportation spikes (restrict interpolation strictly to sparse-tracking labs w/ `is_interpolated` flags) |
| **Flat 1D-CNN Concatenating Multiple Agents in Channels** | Much wider CV–LB gap than GNN + Egocentric Squeezeformer (`#5`) | #5 | Breaks permutation equivariance across agent/target pairs |
| **Horizontal Flip Augmentation on GBDTs & Reverse-Time TTA** | Flip gave `+0.02` PR-AUC on TCN but **`CV↑ LB↓` on GBDTs** (`#9`, `#15`); reverse-time hurt (`#5`) | #5, #9, #15 | Directed social actions (`approach` vs `chase`/`flee`) are not time-reversible; GBDTs overfit mirrored tabular stats |
| **Naive Test Pseudo-Labeling & Unsupervised Semi-Sup on `MABe22`** | Failed to improve CV/LB (`#10`) | #10 | Noisy unlabeled social videos require motion-prediction SSL (`#1`), not hard pseudo-labeling |
| **Segment Gap-Filling / Spike Removal & Per-Lab NN Fine-Tuning** | Zero/negative gain or overfit (`#2`, `#5`, `#15`) | #2, #5, #15 | Metric is frame-level F1 (fragmentation isn't penalized); unified multi-lab NNs w/ lab embeddings generalize better than tiny per-lab fine-tunes |
