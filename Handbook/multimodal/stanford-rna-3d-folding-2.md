# Stanford RNA 3D Folding Part 2 — Top Gold Solutions & Post-Mortem

> **Competition**: [Stanford RNA 3D Folding Part 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2) (Jan 7 – Mar 25, 2026 sub / Mar 31, 2026 blind rerun | 1,867 Teams | 3,394 Submissions | \$75,000 Prize Pool + CASP17)  
> **Task**: Predict **5 candidate 3D conformations** (C1' atom `(x, y, z)` coordinates in Å) per nucleotide for monomer, multi-chain, protein/DNA/ligand-bound RNAs up to `~4,640–5,500 nt`  
> **Evaluation Metric**: Average **Best-of-5 `TM-score`** via US-align (stricter Part 2 residue-numbering alignment)  
> **Dataset**: `310.63 GB` (`5,744` train structures, `MSA/`, `PDB_RNA/`, `stoichiometry`, `all_sequences`, `ligand_SMILES`) | Public Val: `28` post-May-2025 targets (`<30%` MMseqs2 id; `best_template_oracle = 0.5549`) | Live Blind Private: harder template-free/complex cryo-EM targets (`oracle = 0.5033`; `<= 8h` P100/2×T4)

---

## Official Top Gold Leaderboard Standings & Solution Index

Because the hidden live rerun shifted dramatically toward harder, template-free and multi-chain complexes (`best_template_oracle` fell `0.5549 → 0.5033`), Public-LB TBM overfitters collapsed while teams investing in **5-family portfolio diversity (`TBM + Boltz2 + RNAPro + Protenix/BPP-Protenix + DRFold2`), learned multimodal/cross-attention template rerankers, and Kabsch long-RNA chunking** surged **`+228` to `+944` ranks**:

| Private Rank | Public Rank | Team | Public LB | Private LB | Core Winning Paradigm | Writeup / Code |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | #229 | **`team_cp`** | 0.42854 | **0.49669** *(+228)* | **Length-adaptive 5-model portfolio** (`<250 nt`: `2×Boltz2 + RNAPro + Protenix + DRFold2`; `250–999 nt`: `TBM + 2×Boltz2 + 2×RNAPro`; `≥1000 nt`: `3×TBM + 2×Protenix` w/ `512/128` overlap chunking) | [Writeup](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/writeups/1st-place-solution-five-model-ensemble) · [GitHub](https://github.com/yutarooo216/Stanford-RNA-3D-Folding-Part-2-1st-place) · [NB](https://www.kaggle.com/code/yutaroito/1st-place-solution-cleaned-kernel) |
| 🥈 **#2** | **#1** | **`AyPy`** | **0.50415** | **0.49211** | **`BPP-Protenix` architectural mod**: injected continuous **EternaFold Base-Pair Probabilities (`LinearNoBias(1→128)`)** directly into Pairformer `z_init` (**`+0.043` Pub / `+0.017` Priv**, `+0.06` even on pseudoknots!) → `[TBM, 2×RNAPro, 2×BPP-Protenix]` | [Writeup](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/writeups/2nd-place-solution) |
| 🥉 **#3** | #775 | **`Stefan Stefanov`** | — | **0.48488** *(+772)* | **LightGBM TM-score reranker** (198k pairs: RNA align + **PubMedBERT text + BLOSUM62 protein + DNA + Morgan ligand Tanimoto**!) + circular chain-level `448/96` Kabsch chunking (deduping homomers!) + `[2×LGBM-TBM, 2×Protenix, RNAPro]` | [Writeup](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/writeups/3rd-place-solution) · [NB](https://www.kaggle.com/code/stefanstefanov/srna3d2-rnapro-protenix-tbm) |
| 🏅 **#4** | #5 | **`Claude Code and me`** (`gapchenko`) | 0.46218 | **0.48402** *(+1)* | **CUDA US-align** labeled **1.42M pairs** → **Cross-Attention TM Reranker** (`mmseqs_0.30` cluster-disjoint slots!) + co-folding w/ **crystallization-buffer ligand blacklist** + `±6`-res Kabsch blend | [Writeup](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/writeups/4th-place-solution-ca-template-reranker) · [CUDA NB](https://www.kaggle.com/code/gapchenko/gpu-accelerated-usalign) |
| 🏅 **#5** | #6 | **`huyang111`** | 0.45947 | **0.48059** *(+1)* | **MSA depth `>= 700` cutoff + strict determinism** (avoiding `±0.03` diffusion variance) + 2×T4 greedy chunk bin-packing + **$C^2$-quintic smoothstep Kabsch stitching** | [Writeup](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/writeups/5th-place-solution-stanford-rna-3d-folding-part) |
| 🏅 **#6** | #658 | **`Shivam Shinde`** | — | **0.47923** *(+652)* | **Cross-pollination**: runs no-MSA/no-template **Protenix** to generate diverse structural hypotheses → feeds **both TBM & Protenix outputs as `.pt` templates into RNAPro** | [Writeup](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/writeups/6th-place-solution) · [GitHub](https://github.com/ShindeShivam/Stanford-RNA-3D-Folding) |
| 🏅 **#7** | #375 | **`Samus`** | — | **0.47827** *(+368)* | Quality-tiered TBM (`id≥80%`, `≥50%`, else `1`) + Protenix fill → slots 1–4 refined by RNAPro, slot 5 kept raw (`6.2h` on 1×P100); rejected Chai-1, RhoFold+, standalone DRFold2 | [Writeup](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/writeups/7th-place-solution) |
| 🏅 **#8** | #4 | **`G&G`** (`G. Balazs`) | 0.46828 | **0.47641** | **12-letter secondary-structure (`ViennaRNA`) Parasail TBM** + **Hungarian multi-chain matching** + native `768 nt` Protenix (dynamic Pairformer chunk `128`) + centroid-shrunk duplicate homomer copies | [Writeup](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/writeups/8th-place-solution-tbm-and-protenix) · [NB](https://www.kaggle.com/code/gabalz/stanford-rna-3d-folding-part-2-8th-place) |
| 🏅 **#9** | #8 | **`Folded`** (`honganzhu`) | 0.45225 | **0.47598** | Greedy max-diversity TBM (quality × TM diversity) + Protenix (`MSA`+`noMSA`) + Boltz2 (`<900 nt`) → RNAPro refinement | [Writeup](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/writeups/optimize-diversity-when-representation-is-limited) · [NB](https://www.kaggle.com/code/honganzhu/protenix-tbm) |

---

## 1. Universal 3D Structural Folding & Best-of-K Portfolio Breakthroughs

1. **Best-of-5 Portfolio Diversity > Single-Best Mode Averaging (`#1`–`#9`)**:
   - Under `Best-of-5 TM-score`, averaging distinct 3D structures in coordinate space collapses valid conformational modes and degrades score (`#4`, `#6`; `#8` only averaged models within `< 2 Å` Kabsch RMSD).
   - Winning portfolios allocated slots explicitly for **cross-family & structural-cluster diversity**: `#1` combined **5 distinct paradigms (`TBM`, `Boltz2`, `RNAPro`, `Protenix`, `DRFold2`)** via length-adaptive routing; `#4` enforced **`mmseqs_0.30` cluster-disjoint TBM slots**; `#6` cross-pollinated Protenix de-novo predictions as input templates into `RNAPro`.
2. **Soft Biophysical Prior Injection Into Foundation Pairformers (`#2 AyPy`, `+0.043` Pub / `+0.017` Priv)**:
   - Computing continuous Base Pair Probabilities (`[N, N]`) with **EternaFold** (`96.2%` contact precision at `BPP ≥ 0.7` vs ViennaRNA `81.4%`) and adding a minimal **`LinearNoBias(1 → 128)`** residual directly into `Protenix`'s initial pair trunk `z_init` crushed complex alternatives (`20`-bin one-hot `0.317`, 2-block recycling `BppEmbedder` `0.320` → simple linear `0.340` solo!) and even improved pseudoknotted RNAs by `~+0.06` TM-score.
3. **Multimodal & Learned Pairwise Template Reranking (`#3`, `#4`, `#8`)**:
   - Sequence-only alignment misses structurally conserved folds in multi-molecule complexes. `#3` (`+772` ranks!) trained a **LightGBM TM-score reranker** fusing RNA alignment with **PubMedBERT description embeddings, BLOSUM62 protein partner similarity, DNA alignment, and Morgan ligand Tanimoto similarity**; `#4` accelerated US-align in **CUDA** to label **1.42M pairs** for a **Cross-Attention TM Predictor**; `#8` encoded RNA secondary structure into a **12-letter alphabet** for Parasail + **Hungarian multi-chain matching**.
4. **Long-Sequence Chunking, Co-Folding Hygiene & Deterministic Ablation (`#1`, `#3`, `#4`, `#5`, `#8`)**:
   - Bypassed `16 GB` T4/P100 VRAM ceilings via **homomer deduplication** (`9MME` `8×580 = 4,640 nt` collapsed to one `676 nt` run, `#3`), **overlapping sliding chunks (`448–512 nt`, `64–128 nt` overlap)** stitched via **Kabsch SVD** + narrow `±6`-res linear (`#4`) or **$C^2$-quintic smoothstep** (`#5`) blending, and skipping heavy `Boltz2`/`RNAPro` above `900–1,000 nt`.
   - **Co-Folding & MSA Hygiene**: Blacklisting crystallization/buffer ligands (`GOL, EDO, PEG, SO4, PO4, ...`, `#4`), thresholding MSA inclusion at **depth `>= 700`** (`#5`, avoiding noisy shallow MSAs), and enforcing **exact deterministic kernels** (`#5`) prevented `±0.02–0.03` stochastic diffusion noise from corrupting ablations.

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Coordinate-Space Averaging Across Distinct Conformations** | Destroyed Best-of-5 TM-score (`#4`, `#6`) | #4, #6, #8 | Averaging alternative folds produces physically invalid intermediate geometry (only average candidates `<2 Å` RMSD) |
| **Over-Engineered BPP Encoders (20-Bin One-Hot / Recycling `BppEmbedder`)** | Dropped Protenix LB `0.340 → 0.317 / 0.320` (`#2`) | #2 | Pretrained Pairformer trunks absorb continuous linear residuals cleanly; heavy new modules destabilize fine-tuning |
| **MSA Consensus ViennaRNA SSI & Blindly Expanding Template Pools** | Slower and worse than single-seq SSI (`#8`); `+167k` Ribonanza templates gave zero gain (`#7`) | #7, #8 | Noisy/predicted templates dilute top-K reranking precision |
| **Standalone `Chai-1` / `RhoFold+` / Confidence-Reranking Huge Protenix Pools** | Weak/slow (`#7`); confidence reranking failed to beat diversity (`#11`) | #7, #11 | Internal diffusion confidence poorly calibrates across distinct structural basins |
| **Passing Crystallization Buffers as Co-Folded Ligands / >2 TBM Slots in RNAPro** | Buffer ligands distract binding pockets (`#4`); >2 TBM slots plateaued (`#6`) | #4, #6 | Always strip PDB crystallization artifacts (`GOL/PEG/SO4/...`) before co-folding |
