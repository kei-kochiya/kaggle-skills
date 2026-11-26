# NeurIPS 2024 — Lux AI Season 3: Top-10 Solutions & Post-Mortem

> **Competition**: [NeurIPS 2024 — Lux AI Season 3](https://www.kaggle.com/competitions/lux-ai-season-3) (Dec 2024 – Mar 2025, 701 Teams, \$50,000 Prize Pool)  
> **Track**: Multi-Agent Reinforcement Learning, Imitation Learning & Partially Observable Simulation  
> **Environment**: Best-of-5 matches (`100` steps/match) on a `24×24` symmetric fog-of-war grid with **hidden randomized physics parameters** fixed per game across all 5 matches (`16` units/team, `100 MB` archive cap, CPU-only runtime with overtime pool)  
> **Rating Metric**: Gaussian skill estimate $\mathcal{N}(\mu, \sigma^2)$ initialized at $\mu_0 = 600$ (binary match outcome; score margin ignored)

---

## 1. Official Top-10 Leaderboard & Paradigm Matrix

Lux AI Season 3 split cleanly into two competing paradigms: **large-scale self-play RL** (IMPALA/V-trace, PPO+Teacher-KL, PureJaxRL) occupying #1, #2, #5, #6, #10, and **low-compute Behavioral Cloning / Imitation Learning** scraping top RL replays (#3, #4, #7, #8, #9 — including a 3rd-place bot trained on a single **RTX 3060 6GB laptop**!).

| Rank | Team | Final Elo | Paradigm | Core Backbone & Key Breakthroughs | Writeup / Code |
| :---: | :--- | :---: | :--- | :--- | :--- |
| 🥇 **1st** | **Flat Neurons** | **2831.4** | Multi-Agent RL (`IMPALA + V-trace + Teacher KL`) | 24-block ResNet + `ConvLSTM` + 4 spatial Transformer blocks; **auxiliary Enemy Future Prediction Head** (`24×24×3`); unmasked **blind saps**; `8× H100` (`>20B` total steps, `1.5B` final); anti-IL replay obfuscation | [Writeup](https://www.kaggle.com/competitions/lux-ai-season-3/writeups/flat-neurons-1st-place-approach-by-flat-neurons) |
| 🥈 **2nd** | **Frog Parade** | **2425.8** | RL (`PPO + Teacher-KL` + Rust) | **10.8k-LOC Rust engine (`110k` SPS)**; exhaustive energy-field reconstruction (`>99%` exact); 10M `SE-ResNet` + centralized "cheating" critic; **3-way symmetry TTA + stochastic test sampling** | [Writeup](https://www.kaggle.com/competitions/lux-ai-season-3/writeups/frog-parade-frog-parade-s-solution) · [GitHub](https://github.com/IsaiahPressman/kaggle-lux-2024) |
| 🥉 **3rd** | **aDg4b** | **2306.1** | Imitation Learning (Pure BC) | Dual `24×24` UNets (`Unit-UNet` + `SAP-UNet`) cloning Frog Parade & Flat Neurons on a **single RTX 3060 6GB laptop**; energy-sorted stacked-unit assignment; excluded post-decided matches | [Writeup](https://www.kaggle.com/competitions/lux-ai-season-3/writeups/adg4b-imitation-learning-3rd-place-solution) · [GitHub](https://github.com/w9PcJLyb/lux3-bot) |
| 🏅 **4th** | **YumeNeko** | **2152.5** | Imitation Learning (Pure BC) | Dual UNets + **sap-dropoff inference with dynamic Model Switching**; 8-fold `D4` TTA; iterative `1/k` probability subtraction for stacked units; 2-stage Frog Parade curriculum | [Writeup](https://www.kaggle.com/competitions/lux-ai-season-3/writeups/yumeneko-4th-place-solution-imitation-learning-app) · [GitHub](https://github.com/KASSII/Kaggle_LuxAI-s3) |
| 🏅 **5th** | **OneUpKiwi** | **2101.3** | End-to-End JAX RL (`JointPPO`) | `ConvNeXt` + 4-layer Transformer + **pure-JAX `xLSTM`** (`mLSTM + sLSTM`) + autoregressive `JointPPO` decoder (`~2M` actor + `~2M` centralized critic) | [Writeup](https://www.kaggle.com/competitions/lux-ai-season-3/writeups/oneupkiwi-kiwis-xlstm-agent-5th-place-solution) · [GitHub](https://github.com/Elias-Buerger/kaggle-lux-s3) |
| 🏅 **6th** | **EcoBangBang** | **2093.6** | RL (`V-trace + Teacher-KL`) | 16-layer `ResNet` (`13.3M` params), joint `(16, 230)` action space; **dropping UPGO loss** jumped Elo from `1400+ → 2093.6`; Bayesian vs. sliding-window linear point-tile solver | [Writeup](https://www.kaggle.com/competitions/lux-ai-season-3/writeups/ecobangbang-ecobangbang-s-approach-yet-another-rl-) · [GitHub](https://github.com/flynnwang/eco_bang_bang/) |
| 🏅 **7th** | **Vadasz** | **2048.5** | Imitation Learning (UNet → ONNX) | Dual UNets exported to **ONNX inside a C# inference agent**; exact relic deduction + predicted unseen enemy positions on known score nodes | [Writeup](https://www.kaggle.com/competitions/lux-ai-season-3/writeups/vadasz-vadasz-agent-7th-solution) · [GitHub](https://github.com/zoli1984/LuxAIS3) |
| 🏅 **8th** | **Gregor Lied** | **2035.6** | Imitation Learning (UNet + DiT) | Ensembled per-agent UNet heads with an **Agent-ID-conditioned Diffusion Transformer (DiT)** head; exact drift-speed deduction | [Writeup](https://www.kaggle.com/competitions/lux-ai-season-3/writeups/gregor-lied-8th-place-solution-imitation-learning) · [GitHub](https://github.com/gregorlied/lux-s3/tree/main) |
| 🏅 **9th** | **team-k** | **1975.9** | Imitation Learning + Optimization | UNet + **Bayesian hidden-parameter tracker** (posterior mean, std, & spatial point-tile `ENTROPY` channels) + **Min-Cost Flow** unit action assignment | [Writeup](https://www.kaggle.com/competitions/lux-ai-season-3/writeups/team-k-9th-place-solution) · [GitHub](https://github.com/kuto5046/kaggle-luxai-s3) |
| 🏅 **10th** | **Boey** | **1884.5** | End-to-End JAX RL (`PureJaxRL`) | `1.8M` ResNet + BPTT + **separate centralized critic (`+113 Elo` despite halving `80k → 40k` SPS)** + `PFSP` (75% self-play, 25% win-rate-weighted pool) on 1× RTX 4090 | [Writeup](https://www.kaggle.com/competitions/lux-ai-season-3/writeups/boey-10th-place-solution-boey-end-to-end-jax-rl) |

---

## 2. Game Contract & Persistent Hidden-Parameter Inference

### 2.1 Core Environment Dynamics
Each game runs **Best-of-5 matches** (first to 3 wins, `100` steps/match) on a diagonally symmetric `24×24` grid. Hidden physics parameters (`unit_move_cost ∈ [1..5]`, `unit_sensor_range ∈ [2..4]`, `nebula_tile_vision_reduction ∈ [0..3]`, `nebula_tile_energy_reduction ∈ {0,0,10,25}`, `unit_sap_cost ∈ [30..50]`, `unit_sap_range ∈ [3..7]`, `unit_sap_dropoff_factor ∈ {0.25, 0.5, 1.0}`, `unit_energy_void_factor`, `nebula_tile_drift_speed ∈ {±0.025, ±0.05}`, and energy-node drift parameters) are randomized **once per game** and carry across all 5 matches. Units earn 1 point/step per occupied hidden point tile inside `5×5` windows centered on symmetric Relic Nodes (stacking multiple friendly units on the same tile yields zero extra points).

### 2.2 Explicit Stateful Belief Tracking (Unanimous Across Top 10)
None of the top 10 relied purely on raw end-to-end pixel histories to deduce hidden parameters; every medalist paired neural policies with deterministic or Bayesian belief trackers persisted across all 5 matches:

1. **Diagonal Map Mirroring**: Every discovered asteroid, nebula, relic node, energy tile, and deduced point tile is immediately mirrored across the diagonal `(x, y) ↔ (23 - y, 23 - x)`.
2. **Exhaustive Energy-Field Reconstruction (`>99%` Exact, #2, #6, #9)**: Because the `24×24` energy field is generated deterministically from one symmetric pair of energy nodes, precomputing all candidate energy fields and filtering against just a handful of observed tiles reconstructs the **entire `24×24` hidden energy field upon first unit spawn** (resetting and re-matching whenever a drift tick occurs).
3. **Discrete Nebula Drift Deduction & Forward Terrain Projection (#2, #4, #6, #8, #9)**: Observing whether terrain shifts on step `7` vs. step `20` immediately narrows `nebula_tile_drift_speed ∈ {±0.025, ±0.05}`, enabling exact deterministic `next_tile_type` lookahead channels (`1–3` steps ahead + countdown timer).
4. **Hidden `5×5` Relic Point-Tile Deduction**:
   - *Exact Constraint Elimination (#2, #7)*: Comparing `score_delta` against friendly unit occupancy in `5×5` relic windows clears zero-gain tiles (`NotScoreMap`) and locks confirmed tiles whenever `score_delta == #occupied_unknowns`.
   - *Bayesian Belief + Spatial Entropy Channels (#4, #6, #9)*: Sliding-window binary linear equations converge fastest once all relics are discovered, while Bayesian posterior maps (`prior = 0.25–0.50`) + spatial `ENTROPY` channels (#9) remain robust to points scored from undiscovered relics.
5. **Auxiliary Enemy Future Prediction Head (#1 Flat Neurons)**: Rather than hand-coding enemy tracking, Flat Neurons trained a supervised `24×24×3` head predicting **next-tick enemy occupancy, adjacent-to-enemy probability, and enemy sensor coverage** via weighted BCE on partial ground truth, feeding those predictions recursively into both the next observation and the action head.
6. **Sap Dropoff Inference & Dynamic Model Switching (#4 YumeNeko)**: Inferred `unit_sap_dropoff_factor` from observed energy drops on adjacent enemies after sap strikes, switching mid-game from a dropoff-agnostic policy to a dropoff-conditioned policy once identified.

---

## 3. Neural Architectures, Symmetry & Multi-Unit Coordination

### 3.1 Perspective Canonicalization & Symmetry TTA
- **Player-1 Diagonal Flip**: Mirroring observations and actions along the main diagonal whenever playing as Player 1 places the home spawn permanently at `(0, 0)` (note: #3 `aDg4b` countered the induced Right/Down action-frequency imbalance with class-weighted cross-entropy).
- **Test-Time Augmentation & Stochastic Mixed Strategies**:
  - **1st (Flat Neurons)**: `2×` x/y flip TTA (dynamically disabled if overtime consumption exceeded `30s`).
  - **2nd (Frog Parade)**: `3×` symmetry TTA (2 diagonal flips + `180°` rotation) combined with **stochastic action sampling at inference** — crucial in partial-information games where deterministic greedy play becomes exploitable to blind sapping and dodging.
  - **4th (YumeNeko)**: Full `8×` D4 symmetry TTA (4 rotations × flip).

### 3.2 Spatial Backbones & Action-Head Formulations
- **1st (`Flat Neurons` — Hybrid ResNet + ConvLSTM + Spatial Transformer)**: Compressed `~1,100` tile features (`~100` continuous + `~1,000` binned one-hot + `3` previous enemy-prediction maps) `→ 500 → 128` ch → **24 Residual blocks** → **ConvLSTM (`24×24×128`)** → **4 spatial Transformer blocks**. For each of 16 units, cropped a **unit-centered `15×15` local patch** (with out-of-bounds mask) + unit scalars → Movement head (`16×6`) and Sap-Target conv head (`16×15×15` matching max `unit_sap_range = 7`).
- **2nd (`Frog Parade` — 10M SE-ResNet)**: Stacked 10 historical frames + broadcast global features into an 8-block `3×3` `SE-ResNet` (`d=256`), indexing alive unit `(x, y)` features for move/sap choice and using a shared `1×24×24` masked sap-target map.
- **5th (`OneUpKiwi` — ConvNeXt + Transformer + Pure-JAX `xLSTM`)**: Combined full-res `ConvNeXt` map features, 4-layer Transformer over map/unit/global tokens, recurrent `xLSTM` (`mLSTM + sLSTM`), and an **autoregressive `JointPPO` Transformer decoder** coordinating unit actions sequentially.
- **Top IL Dual-UNet Standard (#3, #4, #7, #8, #9)**: Two `24×24` UNets with global parameters broadcast into the bottleneck — a map-level `Unit-UNet` (`6×24×24`) and a unit-centered `SAP-UNet` (`1×24×24`). Because map-level UNets predict one action distribution per tile, teams coordinated **co-located stacked units** via:
  - *Energy-priority split* (#3 `aDg4b`): Assign predicted action to the higher-energy half of stacked units.
  - *Iterative `1/k` probability subtraction* (#4 `YumeNeko`): Subtract `1/k` from a chosen action's probability after assigning each unit on a `k`-unit tile.
  - *Min-Cost Flow assignment* (#9 `team-k`): Solve exact min-cost flow separately for SAP and MOVE units with an occupancy penalty discouraging accidental stacking.

---

## 4. Large-Scale RL vs. Low-Compute Imitation Learning

### 4.1 RL Scaling, Centralized Critics & Stability Levers
- **Centralized ("Cheating") Value Critics (`+113 Elo`, #1, #2, #5, #10)**: Giving the training-only critic full observability of **both teams' states and hidden maps** while keeping the actor strictly fog-of-war compliant dramatically slashed GAE variance. Boey (#10) measured **`+113 Elo` (`1771.5 → 1884.5`)** from switching to a centralized critic even though training speed dropped in half (`80k → 40k` SPS on 1× RTX 4090).
- **Simulator Speed Threshold**: Pure Python envs (~20–100 SPS) failed to progress past `~650 Elo` (#85, #118) because RL only took off after **`10M+` steps (`~20k` games)**. All top RL bots compiled native/JIT engines:
  - **Rust + PyO3 (`Frog Parade`, #2)**: `10.8k` lines of Rust achieved **`110,000` SPS** on 16-core CPU (`300M` training steps in 8 days on RTX 3090 + 2070S).
  - **Pure JAX `PureJaxRL` (`Boey` #10, `OneUpKiwi` #5)**: `40k–80k` SPS (`23B–58B` steps in 7–8 days on a **single RTX 4090**!).
  - **Cluster `IMPALA + V-trace` (`Flat Neurons`, #1)**: `8× H100` (4 train / 4 rollout), `torch.compile()` (`1.5×` speedup), dynamic return scaling to `[-5, +5]`, scheduled target-entropy decay, and **Teacher-KL + Teacher-Baseline anchoring** against a 15–20 checkpoint historical pool.
- **Anti-IL Countermeasure (#1 Flat Neurons)**: To prevent leaderboard IL bots from cloning their strongest policy, Flat Neurons wrapped submissions in an **85% weaker / 15% stronger** randomized selector and later injected logit noise / per-round switching — cutting IL cloning accuracy on Flat Neurons' SAP moves to `0.5577` (vs. `0.7640` on Frog Parade).

### 4.2 Imitation Learning Best Practices (#3, #4, #7, #8, #9)
- **Exclude Post-Decided Matches**: Never clone turns after a team has already clinched 3 of 5 match wins — many top bots intentionally go passive or conserve overtime once the game is decided!
- **Two-Stage Single-Donor Curriculum**: Pretrain on `8k–20k` strong replays, then fine-tune exclusively on `1k–1.5k` **latest winning matches of one elite donor** (mixing incompatible top-bot styles degraded win rate in #4 and #9).
- **Runtime Export**: #7 `Vadasz` exported UNets to **ONNX** executed inside a compiled **C#** runtime for ultra-low CPU latency.

---

## 5. Empirical Negative Results & What Failed

| Failed Hypothesis | Observed Result | Root Cause |
| :--- | :--- | :--- |
| **RL Fine-Tuning from IL UNet Weights (#3, #4)** | Complete policy collapse or zero gain over 2–3 weeks | Map-level IL UNet logits lack calibrated value baselines and joint exploration entropy; IL consistently plateaued `~100–150 Elo` below its RL donor. |
| **Over-Restrictive Sap Action Masking (#2)** | Capped peak Elo vs. #1 (`2425.8` vs. `2831.4`) | Masking sap targets strictly to known point tiles forbade **blind saps** into high-probability enemy fog-of-war transit corridors. |
| **UPGO Loss & Dense Shaped Rewards (#1, #6, #10)** | Stalled `<700` Elo for 50 days (shaped rewards) & `1400` Elo (UPGO); **dropping UPGO jumped to `2093.6`** | Dense exploration/damage shaping caused passive reward hacking; UPGO PG loss overwhelmed baseline gradients (`10×` magnitude). Sparse terminal match/point rewards won decisively. |
| **RoPE ViT & VDN Value Factorization in PPO (#2)** | Collapsed once Teacher-KL decayed (`high clip_fraction`) | Vision Transformers were far more brittle under on-policy PPO shifts than `SE-ResNet`/ConvNet trunks without heavy stabilization. |
| **LSTM / Conv3D Long-Term History in IL (#4)** | Significantly degraded imitation accuracy | Added temporal overfitting noise on 24×24 maps; keeping only 1 previous frame (`pre_self_unit_pos`) helped movement while zero history worked best for sap targeting. |
| **Unaccelerated Python RL Envs (#85, #118)** | Stalled at `600–650 Elo` (`20–100` SPS) | Throughput was `400×–4,000×` too slow to cross the `10M+` step self-play takeoff threshold. |
| **Mixing Multiple Top Teachers in IL (#4, #9)** | Worse than clean 2-stage fine-tuning on one donor | Averaging incompatible strategic styles (e.g., aggressive blind-sapping vs. conservative energy play) yielded indecisive actions. |

---

## 6. Universal Takeaways Promoted / Reinforced

- **Centralized ("Cheating") Asymmetric Critic in Partially Observable RL**: Feeding full global ground truth to the training-only critic while keeping the actor strictly fog-of-war compliant yields massive Elo gains even at `0.5×` rollout throughput.
- **Explicit Belief/Physics Inference + Spatial Entropy Channels**: When hidden environment parameters persist across multi-round matches, combine deterministic/Bayesian state trackers (with posterior uncertainty/entropy channels) with neural policies rather than forcing a raw pixel encoder to rediscover physics from scratch.
- **Stochastic Mixed-Strategy Inference & Unmasked Legal Actions**: In imperfect-information adversarial games, avoid over-restrictive heuristic action masks that outlaw bluffing/blind attacks, and test stochastic sampling + symmetry TTA vs. greedy argmax.
- **Exclude Decided-Match Garbage Time from BC Datasets**: Filter out dead/clinched rounds in Best-of-N games before behavioral cloning.
