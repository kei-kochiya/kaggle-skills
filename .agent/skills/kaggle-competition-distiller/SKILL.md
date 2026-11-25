---
name: kaggle-competition-distiller
description: >-
  Standard operating procedure to analyze, reverse-engineer, and distill winning Kaggle competition solutions and notebooks into structured handbook documentation and universal reusable agent skills. Use whenever adding a new competition to the repository or extracting competitive machine learning takeaways.
---

# Kaggle Competition Distiller Runbook

This skill outlines the exact workflow for an AI agent to ingest completed Kaggle competitions (writeup discussions, leaderboards, notebooks, repositories) and distill them into `Handbook/` and `.agent/skills/`.

---

## 1. Pre-Flight Deduplication & Scope Check

Before researching a competition:
1. **Verify It Does Not Already Exist**: Check `Handbook/<track>/`, `Handbook/README.md`, and `DETAILS.md` to confirm the competition slug has not already been ingested.
2. **Prefer Settled Competitions**: Prioritize competitions that closed **> 1 month ago** with finalized leaderboards and substantive top-medal writeups.
3. **Inspect Full Primary Sources**: Read the official Overview, Data, Metric, Leaderboard, and the **full text** of the top 4–8 winning writeups (not just titles or snippets), recording exact Private/Public scores, rank shakeups, CV schemes, architectures, and reported negative results.

---

## 2. Technical Extraction Checklist

Verify each dimension during analysis:
- [ ] **Metric & Contract Nuances**: Exact formula, asymmetry/penalties, runtime/GPU/CPU caps, rerun or dual-run mechanics.
- [ ] **Data Dynamics & Shift**: Real vs. synthetic, train-vs-test distribution shift, metadata availability at test time, corrupted/outlier samples.
- [ ] **Modeling Breakthroughs**: Dominant backbones/architectures, representation tricks, loss functions, and what created real ensemble diversity.
- [ ] **Validation Reliability**: Grouped/temporal/test-aligned CV setup, CV–LB correlation, and root causes of Public-to-Private shakeup.
- [ ] **Empirical Negative Results**: Explicitly log hypotheses tested and rejected by medalists to prevent repeating dead ends.

---

## 3. Output Generation & Universal Skill Promotion Bar

### A. Create Handbook Report (`Handbook/<track>/<competition-slug>.md`)
Capture **all** competition details—including competition-specific data quirks, exact numbers, team standings, URLs, and negative results—inside the appropriate track folder (`tabular/`, `reinforcement-learning/`, `cv/`, `nlp/`, `time_series/`, `multimodal/`).

### B. Update Catalog & Indexes
Keep the short landing page and detailed catalogs synchronized:
1. Update [`Handbook/README.md`](../../../Handbook/README.md) (competition table + directory tree).
2. Update [`DETAILS.md`](../../../DETAILS.md) (competition table + directory tree + skill cookbooks if skills changed).
3. Update [`README.md`](../../../README.md) (increment track count and list any newly created skill—keep `README.md` concise!).

### C. Codify Strictly Universal Skills (`.agent/skills/`)
Apply a **strict universal-only filter** before modifying or creating any skill:
- **Promote to `.agent/skills/` ONLY IF** the technique transfers reliably across multiple competitions in its domain (e.g., dense DINOv3 `1×1 Conv` patch heads, dual-threshold empty-mask gating, FP8 KV-cache + uncatchable-leader early stopping, `GaussianNLL`/`TemporalHuber` displacement forecasting, cosine embedding distillation + label-sum-capped soft PL).
- **Keep STRICTLY in `Handbook/`** any single-competition hack, regional/dataset constant, magic multiplier, specific play/sample ID, or one-off external dataset join.
