# Kaggle Skills & Handbook 🏆

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![RAPIDS cuDF & cuML](https://img.shields.io/badge/RAPIDS-GPU%20Accelerated-76B900.svg)](https://rapids.ai/)
[![Kaggle Grandmaster Playbook](https://img.shields.io/badge/Playbook-KGMON%202026-gold.svg)](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)

A curated repository of competitive machine learning knowledge distilled from winning Kaggle solutions, post-mortems, and gold-medal architectures.

- **Human Knowledge Base ([`Handbook/`](Handbook/README.md))**: Deep-dive competition post-mortems covering data forensics, architectures, CV discipline, ensembling, and anti-patterns.
- **Universal AI Agent Skills ([`.agent/skills/`](.agent/skills/))**: Auto-discovered Antigravity runbooks (`SKILL.md` + `references/`) containing **strictly transferable, cross-competition techniques** (competition-specific hacks stay in `Handbook/`).
- **Full Detailed Catalog ([`DETAILS.md`](DETAILS.md))**: Per-competition breakthrough tables, full cookbook index, and complete repository tree.

---

## 📊 Coverage by Track

| Track | Entries | Core Scope | Catalog |
| :--- | :---: | :--- | :--- |
| **Tabular** | 15 | Forensics, FE, GBDTs, Tabular NNs, Tabular LLMs/TFMs, live simulation, stacking & calibration | [Details](DETAILS.md#tabular-competitions) |
| **RL & Simulation** | 5 + Starter | BC/PPO, asymmetric critics, executed-action labels, fast Rust/JAX engines, search & serving | [Details](DETAILS.md#reinforcement-learning--simulation-competitions) |
| **Computer Vision** | 9 | Image regression, detection, segmentation, 3D volumetric/medical, SfM & foundation backbones | [Details](DETAILS.md#computer-vision-competitions) |
| **NLP & LLMs** | 10 | Reasoning, post-training, TTT, code abstention, agent security, MT, reranking & reward modeling | [Details](DETAILS.md#nlp--llm-reasoning-competitions) |
| **Time Series** | 6 | Temporal CV, kinematics, streaming finance, wearable sensors & pose-action segmentation | [Details](DETAILS.md#time-series--forecasting-competitions) |
| **Multimodal & Audio** | 7 | Bioacoustics/SED, protein DAGs, RNA 3D, ECG, Bayesian/PDE physics & differentiable rendering | [Details](DETAILS.md#multimodal-audio--science-competitions) |
| **Optimization & Compression** | 3 | Combinatorial packing, GPU/SIMD relaxation, LNS, ONNX circuits, code golf & multi-LLM factories | [Details](DETAILS.md#optimization--compression-competitions) |
| **Autonomous Workflows** | 1 | Multi-LLM competitive engineering & guardrails | [Guide](Handbook/workflows/llm-agentic-kaggle-workflow.md) |

---

## 🛠️ Agent Skills

Coding agents automatically discover skills inside `.agent/skills/`. Every skill enforces a **universal-only bar**: techniques must generalize across competitions in their track, not overfit a single dataset.

| Skill | Scope | Entrypoint |
| :--- | :--- | :--- |
| **`kaggle-tabular-playbook`** | End-to-end tabular workflow: forensics, FE, OOF TE, GBDT/NN/LLM/TFM zoo, survival/matchups, calibration & stacking | [`SKILL.md`](.agent/skills/kaggle-tabular-playbook/SKILL.md) |
| **`kaggle-rl-simulation`** | Simulation kickoff, heuristic/search/BC/RL selection, parity, evaluation & submission serving | [`SKILL.md`](.agent/skills/kaggle-rl-simulation/SKILL.md) |
| **`kaggle-cv-playbook`** | Dense foundation backbones (`DINOv3`), TTT/SSA, 2D/2.5D MIL vs 3D vessel `nnU-Net`, 3D+T tracking, topology & 3D keypoints | [`SKILL.md`](.agent/skills/kaggle-cv-playbook/SKILL.md) |
| **`kaggle-llm-reasoning-playbook`** | FP8 vLLM/TIR serving, Trainable-Trace SFT, online TTT/DML, pairwise reward cascades, low-resource `ByT5` MT & extraction | [`SKILL.md`](.agent/skills/kaggle-llm-reasoning-playbook/SKILL.md) |
| **`kaggle-time-series-playbook`** | Temporal/gap CV, ST encoders, streaming financial online learning, risk targeting, wearable IMU fusion & pose segmentation | [`SKILL.md`](.agent/skills/kaggle-time-series-playbook/SKILL.md) |
| **`kaggle-audio-signal-playbook`** | Cosine embedding distillation, zero-KD diversity, label-sum-capped soft PL, rank blending & shared-mel OpenVINO CPU serving | [`SKILL.md`](.agent/skills/kaggle-audio-signal-playbook/SKILL.md) |
| **`kaggle-scientific-optimization-playbook`** | Forward-operator PDE inversion, ECG signal digitization, DAG stacking, Best-of-K 3D portfolios, GPU/SIMD packing & MCKP agent factories | [`SKILL.md`](.agent/skills/kaggle-scientific-optimization-playbook/SKILL.md) |
| **`kaggle-competition-distiller`** | SOP for distilling winning Kaggle writeups into `Handbook/` entries and strictly universal skills | [`SKILL.md`](.agent/skills/kaggle-competition-distiller/SKILL.md) |

See [`DETAILS.md`](DETAILS.md#skill-cookbooks--references) for the complete list of cookbook references inside each skill, and [`Handbook/README.md`](Handbook/README.md) for entry templates and track taxonomy.

---

## 📄 License
MIT License. Feel free to use, adapt, and build upon these skills and handbook entries for your own Kaggle competitions and AI agent systems.
