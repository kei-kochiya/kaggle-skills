# The Kaggle Handbook 🏆

Welcome to the **Kaggle Handbook**, a curated repository of competitive machine learning knowledge distilled directly from winning solutions, post-mortems, and gold-medal architectures.

This repository serves two interconnected purposes:
1. **Human Knowledge Base (`Handbook/`)**: Deep-dive post-mortems detailing data dynamics, feature discovery, model architectures, validation discipline, and post-processing tricks.
2. **AI Agent Skills (`.agent/skills/`)**: Modular, actionable runbooks formatted according to Antigravity's native skill specification (`SKILL.md`) so coding agents can automatically discover and execute elite competitive workflows.

---

## Competition Index by Track

### 📊 Tabular Competitions

| Competition | Target & Metric | Key Breakthroughs & Winning Techniques | 1st Place CV / LB | Link |
| :--- | :--- | :--- | :--- | :--- |
| **Playground Series S6E1: Predicting Student Test Scores** | Continuous `exam_score` (RMSE) | • Reverse-engineered synthetic data generator formula<br>• Modulo digit decomposition ($10^k \pmod{10}$)<br>• Harmonic periodic trigonometric features ($p=12, 14, 20$)<br>• Out-of-fold multi-agg Target Encoding with Empirical Bayes smoothing<br>• PyTabKit RealMLP tabular neural nets + GBDT zoo<br>• Centered Isotonic Regression (CIR) calibration + Ridge Stacking | CV: 8.56634<br>LB: 8.57273 | [Deep Dive](tabular/playground-s6e1-exam-score.md) |
| **Playground Series S6E2: Predicting Heart Disease** | Binary `Heart Disease` (ROC-AUC) | • Multi-scale quantile (`qcut`), uniform (`cut`), and domain rounding binning<br>• Decimal digit decomposition ($10^i \pmod{10}$ and decimals)<br>• RealMLP (`pytabkit`) tabular neural net with PLR embeddings & Mish<br>• Disciplined ensemble selection via Optuna over ~150 OOF models<br>• Robust Ridge meta-learner on model probabilities<br>• Trusting the CV-LB relation ($0.95578$ threshold prevention of split-overfitting) | CV: 0.95578 – 0.95580<br>Single RealMLP: 0.95574<br>🥇 1st Place | [Deep Dive](tabular/playground-s6e2-heart-disease.md) |
| **Playground Series S6E3: Predicting Customer Churn** | Binary `Churn` (ROC-AUC) | • Snap feature mapping to 7k original IBM dataset & perturbation noise<br>• Mantissa decimal extraction, fractional residuals & Benford's Law<br>• Radix continuous-categorical split encoding<br>• 154-model ensemble selected via GPU forward hill climbing from 850 candidates<br>• 25 DL architectures (RealMLP, TabM, TabICL in-context foundation model, GNN)<br>• 4-level stack with fold-wise rank probability calibration & cuML $L_2$ Logistic Regression | CV: 0.919857<br>Train: 0.920002<br>🥇 1st Place | [Deep Dive](tabular/playground-s6e3-customer-churn.md) |
| **Playground Series S6E5: Predicting F1 Pit Stops** | Binary `PitNextLap` (ROC-AUC) | • Autonomous Codex GPT-5.5 YOLO scaling: 218 models across 37 architectures on 4× A100 GPUs<br>• Reconstructed total scheduled race laps (`LapNumber / RaceProgress`) and degradation pace<br>• RealMLP champion single model (CV 0.9544) + multi-head TabM & Tweedie XGBoost<br>• Adversarial Driver-drift mitigation via parallel Driver-dropped model stream<br>• Synchronous fold-aligned original data augmentation (sample weights 0.5–1.0)<br>• Consensus blending: AutoGluon + 186-OOF Logit Stacker with percentile rank averaging | Public: 0.95488<br>Private: **0.95503**<br>🥇 1st Place (+0.00001 win) | [Deep Dive](tabular/playground-s6e5-f1-pit-stops.md) |
| **Playground Series S6E9: Predicting EV Adoption (Will_Buy_EV)** | Binary `Will_Buy_EV` (ROC-AUC) | • Spearman rank diversity screening threshold ($\rho \le 0.998$)<br>• Inductive bias pairing: PyTorch RealMLP continuous manifold decorrelation with GBDTs (+15% weight)<br>• Avoiding the candidate pooling fallacy (individual vs. pooled injection)<br>• Leaderboard weight sweep & plateau center-selection ($0.94644$)<br>• Zero-leakage prior calibration via monotonic 1D root-finding (17.4645%) | Baseline: 0.94162<br>Top-50: 0.94639<br>Top-Tier LB: **0.94644** | [Deep Dive](tabular/playground-s6e9-will-buy-ev.md) |

*(More competition writeups will be added as new competitions are unpacked into `Competition/`)*

---

## 🤖 Autonomous AI & Agentic Workflows for Kaggle

Specialized architectural runbooks for orchestrating multi-LLM teams and autonomous coding agents to execute end-to-end competitive campaigns:

| Workflow Runbook | Agent Roles & Ecosystem | Core Methodology & Scope | Link |
| :--- | :--- | :--- | :--- |
| **Autonomous Multi-LLM Competitive Playbook** | GPT-5.4, Gemini 3.1, Claude Opus 4.6 (4× A100 GPU RAPIDS cluster) | • KGMON 7-Phase Playbook: Automated EDA, 850-model scaling, 600k lines of code<br>• Synthetic data generator reverse-engineering (snap & digit forensics)<br>• GPU forward hill-climbing model selection & 4-level deep stacking<br>• Programmatic guardrails against data leakage and numerical instability | [Guide](workflows/llm-agentic-kaggle-workflow.md) |

---

## Taxonomy & Structure of Handbook Entries

Every competition entry in the Handbook follows a standardized, battle-tested schema designed for maximum transferability:

```text
Handbook/
├── README.md                          # Master track index & taxonomy (this file)
├── workflows/                         # Autonomous agentic engineering playbooks
│   └── llm-agentic-kaggle-workflow.md # Multi-LLM competitive engineering runbook
├── tabular/                           # Structured/tabular data challenges
│   ├── playground-s6e1-exam-score.md
│   ├── playground-s6e2-heart-disease.md
│   ├── playground-s6e3-customer-churn.md
│   ├── playground-s6e5-f1-pit-stops.md
│   └── playground-s6e9-will-buy-ev.md
├── cv/                                # Computer vision (Classification, Detection, Segmentation)
├── nlp/                               # Natural language processing, LLMs, and retrieval
├── time_series/                       # Forecasting, financial, and temporal series
└── multimodal/                        # Audio, video, graph, and cross-modal tasks
```

### Standard Entry Template
1. **Competition DNA**: Core problem formulation, evaluation metric, dataset characteristics, and leaderboard dynamics.
2. **EDA & Data Discoveries**: Distribution shifts, synthetic generator quirks, leakages, and anomalies.
3. **Feature Engineering Playbook**: Mathematical transformations, interaction formulas, encoding schemes, and domain-specific features.
4. **Model Architecture Zoo**: Tree-based models (LightGBM, XGBoost, CatBoost), tabular neural nets (RealMLP, TabM, Trompt), and custom architectures.
5. **Cross-Validation Strategy**: CV scheme setup, alignment with private test splits, and leak-free transformations.
6. **Ensembling & Post-Processing**: Blending, stacking (Ridge, Nelder-Mead, Hill Climbing), probability/monotonic calibration (CIR), and metric-specific threshold optimization.
7. **Key Takeaways & Anti-Patterns**: What worked, what failed or caused overfitting, and lessons for future competitions.

---

## Agent Skills Integration

Antigravity automatically discovers and activates skills located inside [`.agent/skills/`](../.agent/skills/). 

| Skill Name | Purpose | Location |
| :--- | :--- | :--- |
| **`kaggle-tabular-playbook`** | Complete end-to-end tabular competition runbook (EDA, formula discovery, OOF Target Encoding, RealMLP, GBDTs, CIR calibration, Logit Stacking, Rank Blending) | [`.agent/skills/kaggle-tabular-playbook/SKILL.md`](../.agent/skills/kaggle-tabular-playbook/SKILL.md) |
| **`kaggle-competition-distiller`** | Reusable agent workflow to intake raw competition files/notebooks, reverse-engineer winning recipes, and produce Handbook entries + Agent Skills | [`.agent/skills/kaggle-competition-distiller/SKILL.md`](../.agent/skills/kaggle-competition-distiller/SKILL.md) |
