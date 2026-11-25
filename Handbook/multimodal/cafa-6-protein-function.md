# CAFA 6 Protein Function Prediction — Top Gold Solutions & Post-Mortem

> **Competition**: [CAFA 6 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction) (Oct 15, 2025 – Feb 2, 2026 sub / Jun 1, 2026 prospective curation | 2,172 Final Teams [2,259 raw] | 3,747 Submissions | \$50,000 Prize Pool)  
> **Task**: Multi-label prediction of **Gene Ontology (GO)** terms across **MFO, BPO, CCO** DAG subontologies from amino acid sequences  
> **Evaluation Metric**: Information-Accretion (`IA.tsv`) weighted micro-$F_{\max}$ averaged across **9 prospective cells**: `{No-Knowledge (NK), Limited-Knowledge (LK), Partial-Knowledge (PK)} × {MFO, BPO, CCO}` (`cafaeval -prop fill -max_terms 500`)  
> **Dataset**: `train_sequences.fasta` (`~80k` Swiss-Prot → `145k` merged w/ CAFA5, `199.3 MB`), `go-basic.obo`, `IA.tsv` | Test Superset: `224,309` proteins (`8,023` prospectively annotated `t0→t1`: `749` NK, `485` LK, `6,951` PK; offline TSV upload `<= 1,500` terms/protein)

---

## Official Top Gold Leaderboard Standings & Solution Index

In a prospective 4-month experimental curation benchmark with massive live Public→Private shakeup (`#3` jumped from `~300th` pre-close; `#4`'s `0.335` Pub solo model crushed his `0.404` Pub overfit ensemble), **true multimodal fusion** — combining **curated PubMed literature mining, frozen Protein LMs (`ESM2-3B`/`ProtT5`), homology (`BLAST/DIAMOND`), 3D structure (`Foldseek`), InterPro domains, and non-experimental GOA evidence features** inside **XGBoost LTR or GO-DAG GCN stackers** — dominated:

| Private Rank | Team | Final Prospective $F_{\max}$ | Core Winning Paradigm | Writeup / Code |
| :---: | :--- | :---: | :--- | :--- |
| 🥇 **#1** | **`GOAlpha`** | **0.52430** | **7-modality fusion → XGBoost Learning-to-Rank** (1,000 species-matched val proteins): **curated Swiss-Prot literature (`GOXML` + `GORetriever`)**, `BLAST-KNN`, `Net-KNN`, `FoldSeek-KNN`, `SVM-ESM2`, `LR-InterPro` (`118.7k-d` binary), + 21-d species prior | [Writeup](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/1st-place-solution-for-the-cafa-6-protein-function) · [GitHub](https://github.com/ZhuLab-Fudan/GOAlpha) |
| 🥈 **#2** | **`protein_team`** | **0.48731** | **2-Stage Stacker (11 base predictors → 90-ch GO-DAG GCN w/ SWA)**: GPU **Py-Boost** (`4.5k` out) + **LogReg** (`13.5k` out) + FFN (both **raw & conditional `P(child\|≥1 parent)`**), **5k-d PubMed TF-IDF (`~+0.04` LB, 70–80% of gain!)**, 512-d cross-ontology SVD, **38 binary GOA evidence features**, & bidirectional DAG smoothing | [Writeup](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/2nd-solution-of-cafa-6-py-boost-gcn-articles-da) · [GitHub](https://github.com/btbpanda/CAFA6-updates) |
| 🥉 **#3** | **`yuanllo153`** | **0.44640** *(~#300 Pub!)* | Merged CAFA5+CAFA6 (`145,382` proteins) → `Py-Boost + LogReg + NN` → **21-ch GO-DAG GCN** + 4-config TTA; verified `ESM-IF` 3D embeddings hurt overall CV (`-0.0028`) | [Writeup](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/what-happened) · [GitHub](https://github.com/yuanlao153/CAFA5-protein-function-prediction-2nd-place/tree/cafa6-adapt) |
| 🏅 **#4** | **`FrOzen777`** | **0.43883** | Standalone **Py-Boost** (`ESM2/3-3B` on clean `150k` experimental set, `0.335` Pub!) beat his `0.404` Pub ensemble; proved **150k clean experimental > 1.88M noisy TrEMBL** | [Writeup](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/4st-place-solution-for-the-cafa-6-protein-function) · [GitHub](https://github.com/daiwb-sz/CAFA6-4th-from-CAFA5-2th) |
| 🏅 **#5** | **`AyPy`** | **0.43468** | 3 supervised sets (`0.4 MCM-NN + 0.3 cuML LogReg + 0.3 Py-Boost`) + 3 training-free kNNs (`Foldseek`, `DIAMOND`, **STRING PPI BP-only**); **SPROF-GO Max Constraint Module (`MCLoss`)** + **`1 + IA` loss weights** + `MedCPT` PubMed + taxon-forbidden pruning | [Writeup](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/5th-place-solution) · [GitHub](https://github.com/aypyaypy/CAFA6-5th-Solution) |
| 🏅 **#6** | **`Stefan Stefanov`** | **0.43424** | **Temporal GOA CV (`release 226 → 229`)**; 4 multimodal NNs (`35×` dropout-modality TTA): `TermEvidenceEncoder` Conv1D + Transformer + `TaxonEmbeddingEncoder` + `ESM2-3B` + `ProstT5(seq+3Di)` + `PubMedBERT`, gated residual logits, **`sqrt(IA)` capped `0.5`** loss | [Writeup](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/6th-place-solution-for-the-cafa-6-protein-function) · [GitHub](https://github.com/stefanistefanov/CAFA6-Solution) |
| 🏅 **#11** | **`Antonina Dolgorukova`** | **0.42114** *(0.4252 post)* | Hierarchy-aware `ESM2-3B` MLP + `ProtT5` PyBoost + `BLAST-KNN` (pair identity/coverage stats lifted post-reveal `0.4211 → 0.4252`) + GOA evidence → **XGBoost LTR** | [Writeup](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/11th-place-solution) · [GitHub](https://github.com/T0chka/cafa6_11th_place_solution) |

---

## 1. Universal Scientific Multimodal & Hierarchical DAG Breakthroughs

1. **Weak/Electronic Annotations as *Input Features*, Never as Noisy Targets (`#2`, `#4`, `#5`, `#6`, `#11`)**:
   - Directly training on `550k–1.88M` electronic TrEMBL annotations degraded LB (`#4`). Instead, `#2` audited that `~30%` of electronic GOA annotations later become experimental (`>50%` with `≥3` distinct evidence codes) and fed **38 binary GOA evidence-code & count features** directly into their Stage-2 GCN stacker (while `#6` built a dedicated `TermEvidenceEncoder` with a learned residual logit gate and modality-dropout TTA!).
2. **Curated Literature + Homology + Frozen PLMs > Any Single Backbone (`#1`, `#2`, `#5`, `#6`)**:
   - Adding **curated Swiss-Prot PubMed literature** (`GOXML`/`GORetriever` `#1`; `5,000-d` TF-IDF `#2` worth **`~+0.04` LB**; `MedCPT` `#5`; `PubMedBERT` `#6`) delivered the single largest gap in CAFA 6 — restricted carefully to 1 of 11 predictors in `#2` to prevent over-reliance when test proteins lack publications.
   - **Frozen mean-pooled PLMs (`ESM2-3B`, `ProtT5-XL`, `ProstT5`)** beat fine-tuning PLMs (`#2`) and paired cleanly with **training-free alignment kNNs** (`BLAST`, `DIAMOND`, `Foldseek`, and **STRING PPI strictly for BP** `#5`), **InterPro domains (`#1`)**, and **coarser taxonomic lineage embeddings** (Phylum/Order `[UNK]` bucketing `#5` or lineage Transformer `#6`).
3. **Hierarchical DAG Architectures, Dampened Metric Weights & Ontology Conditioning (`#2`, `#5`, `#6`)**:
   - **Conditional Ontology Training (`#2`)**: Training ontology-`X` stackers **only on proteins having `≥1` term in `X`** estimates $P(\text{term} \mid \text{ontology } X \text{ exists})$, matching CAFA's zero-penalty rule on proteins outside `X`'s evaluation set, while injecting `512-d` cross-ontology SVD features for LK/PK proteins.
   - **Conditional Models + Multi-Channel DAG Stacker (`#2`, `#3`)**: Predicting $P(\text{child} \mid \ge 1 \text{ parent})$ alongside raw probabilities, computing strict parent-product & relaxed noisy-OR propagation channels, running a **GO-DAG GCN**, and finishing with **bidirectional smoothing** (`mean(raw, max(children), min(parents))`) reconciled parent–child contradictions.
   - **Differentiable Hierarchy & Dampened Weights**: SPROF-GO's **Max Constraint Module (`MCM-NN` + asymmetric `MCLoss`, `#5`)** enforces True-Path monotonicity end-to-end; **dampening `IA` loss weights** (`1 + IA` `#5` or `sqrt(IA)` capped at `0.5` `#6`) prevented rare-term gradient explosions where raw `IA` weighting failed (`#4`).
4. **Prospective Temporal Split Validation (`#6`) & Exact Sequence Grouping (`#5`, `#11`)**:
   - Validating on a **temporal GOA release delta (`release 226 → 229`, `#6`)** directly measured NK/LK/PK prospective shift, whereas chasing the static Public LB caused severe shake-downs (`#4`). Grouping CV by exact sequence prevented trivial BLAST/PLM memorization (`#5`, `#11`).

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Overfitting Live Public LB Instead of Temporal / Sequence-Grouped CV** | `#4`'s `0.404` Pub ensemble lost to his `0.335` Pub solo Py-Boost (`4th` Priv!) | #3, #4, #6 | Live Public LB was a small static UniProt holdout unrepresentative of prospective NK/LK/PK accumulation |
| **Training on `1.88M` Electronic TrEMBL Targets & Fine-Tuning PLMs** | Electronic targets hurt LB (`#4`); PLM fine-tuning gave no gain (`#2`) | #2, #4 | Electronic annotations have systematic bias; feed them as *input features*, keep targets strictly experimental |
| **Global 3D Structure Embeddings (`ESM-IF` / `esm3` Domain Features)** | Dropped overall CV `0.5733 → 0.5705` (`-0.0028`, `#3`); lagged `ESM2-3B` (`#4`) | #3, #4 | Structural fold-matching (`Foldseek-kNN` / `ProstT5-3Di`) captures remote homology far better than global ESM-IF pooling |
| **Raw Uncapped `IA` Loss Weighting & Applying PPI Networks to MF/CC** | Raw `IA` impaired TRGO (`#4`); PPI hurt MF/CC (`#5`) | #4, #5 | Extreme `IA` skew destabilizes gradients (use `1+IA` or capped `sqrt(IA)`); PPI reflects shared Biological Processes, not Molecular Function |
| **Leaving Known `t0` Annotations in PK Submissions or Exceeding Term Caps** | Wastes top-500 `cafaeval` slots without earning PK credit | #5, #11 | Partial-Knowledge evaluates *newly accumulated* terms (`t0 → t1`); drop known `t0` terms and prune taxon-impossible GOs |
