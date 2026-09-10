---
name: kaggle-competition-distiller
description: >-
  Standard operating procedure to analyze, reverse-engineer, and distill winning Kaggle competition solutions and notebooks into structured handbook documentation and reusable agent skills. Use whenever adding a new competition to the repository or extracting competitive machine learning takeaways.
---

# Kaggle Competition Distiller Runbook

This skill outlines the exact workflow for an AI agent to ingest raw competition artifacts (data zips, notebooks, writeup discussions) and distill them into the `Handbook/` and `.agent/skills/`.

---

## 1. Input Ingestion Protocol

When a new folder is placed in `Competition/<CompetitionName>/`:
1. **Locate Artifacts**:
   - Look for dataset archives (`*.zip`), solution notebooks (`*.ipynb`), training scripts (`*.py`), and discussion writeup URLs.
2. **Inspect Notebook Structure**:
   - Extract code cells and executed outputs without modifying original files.
   - Specifically identify:
     - Target variable and evaluation metric.
     - Feature engineering steps (custom transformers, formulas, aggregations).
     - Cross-validation split strategy (K-Fold, Stratified, Group, Time-Series).
     - Model families (GBDTs, Neural Networks, Ensembles).
     - Out-of-fold and Test prediction generation.
     - Ensembling, stacking, blending weights, and post-processing calibration.

---

## 2. Technical Extraction Checklist

Verify each of the following dimensions during analysis:
- [ ] **Metric Nuances**: Is the metric symmetric (RMSE, LogLoss) or rank-based (AUC, Spearman), or custom?
- [ ] **Data Dynamics**: Is it synthetic (CTGAN, Copula) or real? Were there duplicate rows, label noise, or data leakage?
- [ ] **Secret Sauce Features**: Did top solutions reconstruct underlying formulas? Were there periodic transforms, digit splits, or domain ratios?
- [ ] **Model Zoo**: What was the single best model score? What model families provided the highest ensemble diversity?
- [ ] **Validation Reliability**: Did CV score correlate with the public and private leaderboards? What caused teams to shake down?

---

## 3. Output Generation

### A. Create Handbook Report
Generate `Handbook/<track>/<competition-slug>.md` using the standard schema:
1. **Executive Summary & Core Breakthroughs** (Scores, CV/LB, key ideas).
2. **Dataset Architecture & EDA** (Features, distributions, quirks).
3. **Feature Engineering Playbook** (Exact mathematical formulas and code).
4. **Modeling & Architectures** (GBDTs, Tabular NNs, hyperparameters).
5. **Ensembling & Post-Processing** (Stacking, calibration, thresholds).
6. **Key Takeaways & Anti-Patterns** (What worked vs what overfit).

### B. Update Master Index
Update [`Handbook/README.md`](../../Handbook/README.md) with a new row in the competition index table.

### C. Codify Reusable Skills
If a novel, generalizable technique was uncovered (e.g. CIR calibration, a custom loss function, or a specialized CV scheme), codify it as a reusable skill in `.agent/skills/` with working code references.
