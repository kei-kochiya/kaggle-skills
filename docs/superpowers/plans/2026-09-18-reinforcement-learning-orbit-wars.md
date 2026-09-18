# Reinforcement Learning & Simulation (Orbit Wars) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a comprehensive Reinforcement Learning & Simulation competition track in the repository, featuring a complete 1st Place Orbit Wars solution post-mortem and an elite reusable agent skill (`kaggle-rl-simulation`) with 4 modular reference guides.

**Architecture:** Dual-layer architecture: (1) In-depth technical Handbook entry in `Handbook/reinforcement-learning/orbit-wars.md` detailing the 200M transformer, Rust simulator, PPO teacher distillation, NF4/NF5 quantization, and 4-player game dynamics; (2) High-leverage, discoverable agent skill in `.agent/skills/kaggle-rl-simulation/` with dedicated reference cookbooks for environment acceleration, policy architectures, training stability, and constrained CPU submission serving.

**Tech Stack:** PyTorch, Rust (PyO3, Rayon), PPO (Proximal Policy Optimization), NormalFloat (NF4/NF5) quantization, dynamic int8 CPU inference, Markdown.

**Spec:** `docs/superpowers/specs/2026-09-18-reinforcement-learning-orbit-wars-design.md`

## Global Constraints
- Strictly adhere to Antigravity's Agent Skills specification (`SKILL.md` frontmatter max 1024 chars, description starts with "Use when...", third person, no workflow summary in description).
- All mathematical formulas must be accurately formatted in KaTeX/LaTeX.
- All code references and algorithms must reflect real implementations validated against `IsaiahPressman/kaggle-orbit-wars`.
- Markdown links must be relative and resolvable within the repository.

---

### Task 1: Create Handbook Deep-Dive (`Handbook/reinforcement-learning/orbit-wars.md`)

**Files:**
- Create: `Handbook/reinforcement-learning/orbit-wars.md`

**Interfaces:**
- Produces: Complete 7-part engineering autopsy of the 1st place solution and Orbit Wars competition.

- [ ] **Step 1: Draft the Executive Summary, Competition DNA & Physics Formulation**
  - Section 1: Executive Summary & Core Breakthroughs (200M transformer, 15B steps, single forward pass for all players, Rust simulator 120k steps/sec, sub-100MiB NF4 packing, 1s overage fallback).
  - Section 2: Celestial Orbital Mechanics & Environment Dynamics (2p vs 4p continuous 2D space, planetary gravity wells, sun collisions, comets, fleets, and the discrete-target action formulation transition).

- [ ] **Step 2: Draft the Simulation Engine & Model Architecture Sections**
  - Section 3: High-Throughput Rust Simulator & Replay Parity (PyO3, Rayon vector envs, zero-copy pinned buffers, broad-phase AABB collision filtering, replay parity testing against Kaggle tournament JSON episodes).
  - Section 4: Transformer Policy & Value Architecture (Multi-entity projections to 768-d, 17 control/scratch tokens, 38-block transformer trunk, single-pass multi-player prediction, Bernoulli launch head, query-key target selection, truncated discretized logistic mixture fleet sizing, softmax win-probability critic).

- [ ] **Step 3: Draft Training Infrastructure, Submission Quantization & Post-Mortem Sections**
  - Section 5: Scaled PPO & Distributed Training (GAE-λ, teacher distillation against last-best checkpoint with >70% win-rate replacement threshold, 4 nodes × 8× B200 GPUs, 8,192 parallel envs, 2400 B200-hours).
  - Section 6: Submission Engineering Under Hard Constraints (100MiB limit: NF4 group-128 LSQ quantization; 1s + 60s CPU limit: dynamic int8 CPU inference, observation entity compaction/fleet truncation, and 1s overage fallback routing).
  - Section 7: Critical Post-Mortem Insights & Anti-Patterns (The stalling bug from $\gamma = 1.0$, matchmaking inversion and the necessity of league play for multi-agent games, the action masking paradox).

- [ ] **Step 4: Review and verify completeness of `Handbook/reinforcement-learning/orbit-wars.md`**
  - Verify formatting, formulas, links, and code blocks.

- [ ] **Step 5: Commit**
  - `git add Handbook/reinforcement-learning/orbit-wars.md`
  - `git commit -m "feat(handbook): add comprehensive Orbit Wars 1st place post-mortem"`

---

### Task 2: Create Modular Reference Guides in `.agent/skills/kaggle-rl-simulation/references/`

**Files:**
- Create: `.agent/skills/kaggle-rl-simulation/references/simulator_acceleration.md`
- Create: `.agent/skills/kaggle-rl-simulation/references/model_architectures.md`
- Create: `.agent/skills/kaggle-rl-simulation/references/rl_training_stability.md`
- Create: `.agent/skills/kaggle-rl-simulation/references/submission_quantization_serving.md`

**Interfaces:**
- Produces: 4 specialized technical cookbooks referenced by `SKILL.md`.

- [ ] **Step 1: Write `simulator_acceleration.md`**
  - Native Rust (PyO3 + Rayon) vector environment patterns.
  - Zero-copy tensor buffers, pinned memory, and cache-aligned layouts.
  - Spatial partitioning, collision broad-phase AABB, and swept checks.
  - Automated replay parity test harness comparing transitions against Kaggle replay JSONs.

- [ ] **Step 2: Write `model_architectures.md`**
  - Entity-centric tokenization for variable-size boards (planets, units, obstacles).
  - Shared global scratch tokens for attention workspace.
  - Single-pass multi-player joint inference (evaluating all players in one forward pass).
  - Truncated discretized logistic mixture heads for bounded continuous actions.

- [ ] **Step 3: Write `rl_training_stability.md`**
  - Scaled PPO recipes (hyperparameters, AdamW/Muon, GAE-λ, clipping, batch size ratios).
  - Teacher distillation loss against historical checkpoints (>70% win-rate promotion).
  - Multi-agent game dynamics: 2-player zero-sum vs N-player non-transitive dynamics & league play.
  - Reward engineering: the discount factor dilemma ($\gamma=1.0$), early truncation, and surrender penalties.

- [ ] **Step 4: Write `submission_quantization_serving.md`**
  - Sub-100MiB checkpoint compression: Grouped NormalFloat (NF4/NF5) with least-squares (LSQ) scale refinement.
  - Runtime dynamic int8 CPU quantization via PyTorch for fast CPU inference.
  - Observation compaction and minor entity pruning under latency pressure.
  - Latency guardrails and two-tier fallback routing cascades.

- [ ] **Step 5: Commit**
  - `git add .agent/skills/kaggle-rl-simulation/references/`
  - `git commit -m "feat(skills): add modular reference guides for kaggle-rl-simulation"`

---

### Task 3: Create Master Skill Runbook (`.agent/skills/kaggle-rl-simulation/SKILL.md`)

**Files:**
- Create: `.agent/skills/kaggle-rl-simulation/SKILL.md`

**Interfaces:**
- Consumes: Reference guides from Task 2.
- Produces: Discoverable, actionable agent skill following Antigravity Agent Skills specification.

- [ ] **Step 1: Write YAML frontmatter**
  - `name`: `kaggle-rl-simulation`
  - `description`: Strict third-person condition-only description (`Use when...`).
  - Keep under 500 characters.

- [ ] **Step 2: Write Overview & Core Philosophy**
  - Sutton's Bitter Lesson in competitive environments.
  - 5-phase operational lifecycle.

- [ ] **Step 3: Write Operational Phases & Decision Flowcharts**
  - Phase 1: Simulator Diagnostics & Vectorization Protocol.
  - Phase 2: Observation & Action Space Structuring.
  - Phase 3: Neural Policy Architecture & Multi-Agent Representation.
  - Phase 4: Distributed PPO & Training Stabilization.
  - Phase 5: Submission Packaging, Quantization & Latency Guardrails.
  - Include decision trees for continuous vs discrete action spaces, self-play vs league play, and quantization codecs.

- [ ] **Step 4: Write Quick Reference Table & Common Anti-Patterns**
  - Quick reference lookup table for operations and formulas.
  - Common pitfalls (action masking paradox, gamma=1.0 stalling, slow CPU timeout, non-transitive multi-agent collapse) and concrete fixes.

- [ ] **Step 5: Commit**
  - `git add .agent/skills/kaggle-rl-simulation/SKILL.md`
  - `git commit -m "feat(skills): add kaggle-rl-simulation skill runbook"`

---

### Task 4: Update Repository Master Catalogs (`Handbook/README.md` and root `README.md`)

**Files:**
- Modify: `Handbook/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: Handbook entry and skills from Tasks 1-3.
- Produces: Synchronized master indexes and taxonomy across the entire repository.

- [ ] **Step 1: Update `Handbook/README.md`**
  - Add `### 🚀 Reinforcement Learning & Simulation Competitions` table with Orbit Wars entry.
  - Update directory taxonomy to include `Handbook/reinforcement-learning/`.
  - Add `kaggle-rl-simulation` to the Agent Skills Integration table.

- [ ] **Step 2: Update root `README.md`**
  - Add Orbit Wars to the competition index table.
  - Add `kaggle-rl-simulation` and its reference cookbooks to the Agent Skills Architecture table.
  - Update the repository file tree structure diagram.

- [ ] **Step 3: Commit**
  - `git add Handbook/README.md README.md`
  - `git commit -m "docs: update master repository catalogs with rl track and orbit wars"`

---

### Task 5: Verification & Quality Audit

**Files:**
- Inspect all newly created and modified files.

- [ ] **Step 1: Validate SKILL.md Frontmatter & SDO**
  - Check YAML frontmatter formatting and length.
  - Ensure description starts with "Use when..." and has no workflow summary.

- [ ] **Step 2: Audit Markdown Links & Anchors**
  - Run verification script to check that all relative links between READMEs, Handbook entries, and Skill reference docs resolve cleanly.

- [ ] **Step 3: Final Commit & Cleanup**
  - Ensure clean git status with all changes committed.
