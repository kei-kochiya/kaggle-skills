# Kaggle Skills & Handbook 🏆

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![RAPIDS cuDF & cuML](https://img.shields.io/badge/RAPIDS-GPU%20Accelerated-76B900.svg)](https://rapids.ai/)
[![Kaggle Grandmaster Playbook](https://img.shields.io/badge/Playbook-KGMON%202026-gold.svg)](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)

A curated repository of competitive machine learning knowledge distilled directly from winning solutions, post-mortems, and gold-medal architectures.

This repository serves two interconnected purposes:
1. **Human Knowledge Base (`Handbook/`)**: Deep-dive post-mortems detailing data dynamics, feature discovery, model architectures, validation discipline, and post-processing tricks.
2. **AI Agent Skills (`.agent/skills/`)**: Modular, actionable runbooks formatted according to Antigravity's native skill specification (`SKILL.md`) so coding agents can automatically discover and execute elite competitive workflows.

---

## 📊 Competition Index by Track

### Tabular Competitions

| Competition | Target & Metric | Key Breakthroughs & Winning Techniques | 1st Place CV / LB | Handbook Link |
| :--- | :--- | :--- | :--- | :--- |
| **Playground Series S6E1: Student Test Scores** | Continuous `exam_score` (RMSE) | • Reverse-engineered synthetic data generator formula<br>• Modulo digit decomposition ($10^k \pmod{10}$)<br>• Harmonic periodic trigonometric features ($p=12, 14, 20$)<br>• Out-of-fold multi-agg Target Encoding with Empirical Bayes smoothing<br>• PyTabKit RealMLP tabular neural nets + GBDT zoo<br>• Centered Isotonic Regression (CIR) calibration + Ridge Stacking | CV: 8.56634<br>LB: 8.57273 | [Deep Dive](Handbook/tabular/playground-s6e1-exam-score.md) |
| **Playground Series S6E2: Predicting Heart Disease** | Binary `Heart Disease` (ROC-AUC) | • Multi-scale quantile (`qcut`), uniform (`cut`), and domain rounding binning<br>• Decimal digit decomposition ($10^i \pmod{10}$ and decimals)<br>• RealMLP (`pytabkit`) tabular neural net with PLR embeddings & Mish<br>• Disciplined ensemble selection via Optuna over ~150 OOF models<br>• Robust Ridge meta-learner on model probabilities<br>• Trusting the CV-LB relation ($0.95578$ threshold prevention of split-overfitting) | CV: 0.95578 – 0.95580<br>Single RealMLP: 0.95574<br>🥇 1st Place | [Deep Dive](Handbook/tabular/playground-s6e2-heart-disease.md) |
| **Playground Series S6E3: Predicting Customer Churn** | Binary `Churn` (ROC-AUC) | • Snap feature mapping to 7k original IBM dataset & perturbation noise<br>• Mantissa decimal extraction, fractional residuals & Benford's Law<br>• Radix continuous-categorical split encoding<br>• 154-model ensemble selected via GPU forward hill climbing from 850 candidates<br>• 25 DL architectures (RealMLP, TabM, TabICL in-context foundation model, GNN)<br>• 4-level stack with fold-wise rank probability calibration & cuML $L_2$ Logistic Regression | CV: 0.919857<br>Train: 0.920002<br>🥇 1st Place | [Deep Dive](Handbook/tabular/playground-s6e3-customer-churn.md) |
| **Playground Series S6E5: Predicting F1 Pit Stops** | Binary `PitNextLap` (ROC-AUC) | • Autonomous Codex GPT-5.5 YOLO scaling: 218 models across 37 architectures on 4× A100 GPUs<br>• Reconstructed total scheduled race laps (`LapNumber / RaceProgress`) and degradation pace<br>• RealMLP champion single model (CV 0.9544) + multi-head TabM & Tweedie XGBoost<br>• Adversarial Driver-drift mitigation via parallel Driver-dropped model stream<br>• Synchronous fold-aligned original data augmentation (sample weights 0.5–1.0)<br>• Consensus blending: AutoGluon + 186-OOF Logit Stacker with percentile rank averaging | Public: 0.95488<br>Private: **0.95503**<br>🥇 1st Place (+0.00001 win) | [Deep Dive](Handbook/tabular/playground-s6e5-f1-pit-stops.md) |
| **Playground Series S6E9: Predicting EV Adoption (Will_Buy_EV)** | Binary `Will_Buy_EV` (ROC-AUC) | • Spearman rank diversity screening threshold ($\rho \le 0.998$)<br>• Inductive bias pairing: PyTorch RealMLP continuous manifold decorrelation with GBDTs (+15% weight)<br>• Avoiding the candidate pooling fallacy (individual vs. pooled injection)<br>• Leaderboard weight sweep & plateau center-selection ($0.94644$)<br>• Zero-leakage prior calibration via monotonic 1D root-finding (17.4645%) | Baseline: 0.94162<br>Top-50: 0.94639<br>Top-Tier LB: **0.94644** | [Deep Dive](Handbook/tabular/playground-s6e9-will-buy-ev.md) |

---

## 🤖 Autonomous AI & Multi-LLM Workflows

Specialized architectural runbooks for orchestrating multi-LLM teams and autonomous coding agents to execute end-to-end competitive campaigns:

| Workflow Runbook | Agent Roles & Ecosystem | Core Methodology & Scope | Link |
| :--- | :--- | :--- | :--- |
| **Autonomous Multi-LLM Competitive Playbook** | GPT-5.4, Gemini 3.1, Claude Opus 4.6 (4× A100 GPU RAPIDS cluster) | • KGMON 7-Phase Playbook: Automated EDA, 850-model scaling, 600k lines of code<br>• Synthetic data generator reverse-engineering (snap & digit forensics)<br>• GPU forward hill-climbing model selection & 4-level deep stacking<br>• Programmatic guardrails against data leakage and numerical instability | [Guide](Handbook/workflows/llm-agentic-kaggle-workflow.md) |

---

## 🛠️ Agent Skills Architecture

Coding agents (e.g. Antigravity) automatically discover and activate skills located inside `.agent/skills/`:

| Skill Name | Purpose | Location |
| :--- | :--- | :--- |
| **`kaggle-tabular-playbook`** | Complete end-to-end tabular competition runbook (EDA, formula discovery, Snap features, Radix encoding, OOF Target Encoding, RealMLP, GBDTs, CIR calibration, Logit Stacking & Hill Climbing) | [`.agent/skills/kaggle-tabular-playbook/SKILL.md`](.agent/skills/kaggle-tabular-playbook/SKILL.md) |
| **`kaggle-competition-distiller`** | Reusable agent workflow to intake raw competition files/notebooks, reverse-engineer winning recipes, and produce Handbook entries + Agent Skills | [`.agent/skills/kaggle-competition-distiller/SKILL.md`](.agent/skills/kaggle-competition-distiller/SKILL.md) |

### Skill Cookbooks & References
- [Exploratory Data Forensics (DS & DA Playbook)](.agent/skills/kaggle-tabular-playbook/references/eda_data_forensics.md): Adversarial validation, generator reverse-engineering, Wilson CI bivariate profiling, mantissa dissection, domain residual audits, and Chi-square interaction screening.
- [Feature Engineering Toolkit](.agent/skills/kaggle-tabular-playbook/references/feature_engineering.md): Snap mapping, Radix interactions, modulo digit extraction, cKDTree priors, Benford's Law anomaly detection.
- [Leak-Free Multi-Agg Target Encoding](.agent/skills/kaggle-tabular-playbook/references/oof_target_encoding.md): Nested 5×5 fold-safe empirical Bayes target encoding across 10 statistical aggregations.
- [Ensembling, Calibration & Hill Climbing](.agent/skills/kaggle-tabular-playbook/references/stacking_cir_ridge.md): Centered Isotonic Regression (CIR), Ridge regression, fold-wise ordinal rank calibration, and greedy forward selection.
- [Logit Stacking & Rank Blending](.agent/skills/kaggle-tabular-playbook/references/logit_stacking_rank_blend.md): Logit transformation with regularized Logistic Regression, cross-ensembler percentile rank blending, synchronous multi-dataset fold protocol, and adversarial drift ablation.

---

## 📁 Repository Structure

```text
kaggle-skills/
├── README.md                                 # Master repository catalog (this file)
├── .gitignore                                # Ignores local large competition data
├── .agent/
│   └── skills/                               # Native Antigravity auto-discovered skills
│       ├── kaggle-tabular-playbook/
│       │   ├── SKILL.md
│       │   └── references/
│       │       ├── eda_data_forensics.md
│       │       ├── feature_engineering.md
│       │       ├── oof_target_encoding.md
│       │       ├── stacking_cir_ridge.md
│       │       └── logit_stacking_rank_blend.md
│       └── kaggle-competition-distiller/
│           └── SKILL.md
└── Handbook/
    ├── README.md                             # Master track index & taxonomy
    ├── workflows/                            # Autonomous multi-LLM engineering guides
    │   └── llm-agentic-kaggle-workflow.md
    └── tabular/                              # Tabular competition post-mortems
        ├── playground-s6e1-exam-score.md
        ├── playground-s6e2-heart-disease.md
        ├── playground-s6e3-customer-churn.md
        ├── playground-s6e5-f1-pit-stops.md
        └── playground-s6e9-will-buy-ev.md
```

*(Note: Raw multi-gigabyte competition files, datasets, and local checkpoints in `Competition/` are excluded via `.gitignore`.)*

---

## 📄 License
MIT License. Feel free to use, adapt, and build upon these skills and handbook entries for your own Kaggle competitions and AI agent systems!
