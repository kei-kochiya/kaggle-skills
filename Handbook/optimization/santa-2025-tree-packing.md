# Santa 2025 — Christmas Tree Packing Challenge — Top Gold Solutions & Post-Mortem

> **Competition**: [Santa 2025 - Christmas Tree Packing Challenge](https://www.kaggle.com/competitions/santa-2025) (Nov 17, 2025 – Jan 30, 2026 | 3,357 Teams | 103,202 Submissions | \$50,000 Prize Pool incl. \$10k Rudolph Prize)  
> **Task**: Pack $N = 1, 2, \dots, 200$ identical 15-vertex non-convex 2D Christmas tree polygons (`20,100` trees total) into minimum axis-aligned bounding squares with **zero interior overlap** (`Decimal('1e18')` + `Shapely v2.1.2`)  
> **Evaluation Metric**: $\sum_{n=1}^{200} \frac{s_n^2}{n}$ where $s_n = \max(x_{\max}-x_{\min}, y_{\max}-y_{\min})$ (area lower bound `49.125`; optimal infinite periodic 2-tree lattice proxy `~60.86`; greedy baseline `167.079`)  
> **Format**: Offline CSV upload (`id,x,y,deg` prefixed with `'s'`; identical Public & Private evaluation)

---

## Official Top Gold Leaderboard Standings & Solution Index

Breaking past the `~70.0` "glassy state" wall of naive SA/DE required uniting **(1) optimal non-perpendicular 2-tree periodic crystals (`0.30430` area/tree vs orthogonal `0.31071`) + bulk-core freezing / 180° symmetry**, **(2) fixed-box feasibility relaxation with precomputed CUDA Minkowski LUTs / AVX2 SAT / SIMD PBD**, and **(3) cross-$N$ transfer ($N \leftrightarrow N\pm 1, N\pm 2$) & corner/motif splicing**:

| Rank | Team | Final Score | Margin vs #1 | Core Winning Paradigm | Writeup / Code |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **`Jeroen Cottaar`** | **68.781235** | — | **Ring-topology Island GA** (`3,750` offspring → `111` diversity-filtered via CUDA Hungarian distance) + **batched GPU L-BFGS (`30,000` relaxations/s on RTX 5090)** querying a **precomputed CUDA 3D Minkowski separation LUT**; fixed-box feasibility shrink + 180° symmetry + perfect non-perp dimer | [Writeup](https://www.kaggle.com/competitions/santa-2025/writeups/1st-place-genetic-algorithm-and-gpu-relaxation) · [GitHub](https://github.com/jcottaar/packing) · [NB](https://www.kaggle.com/code/jeroencottaar/santa-2025-1st-place-solution) |
| 🥈 **#2** | **`KiRaRe`** (`nagiss`, `terry_u16`, `c-number`) | **68.800802** | `+0.0196` | **Rust/AVX2 4-step LNS** (`remove` BFS around disorder → `loose` SA → `place` MCMC+clique → `tighten` SA) alternating **symmetry-invariant Beam Search** & **ILS**; 3-tier SoA AVX2 AABB pruning + `Shapely` legalization | [Writeup](https://www.kaggle.com/competitions/santa-2025/writeups/2nd-place-simulated-annealing-iterated-local-sea) · [GitHub](https://github.com/terry-u16/santa2025) |
| 🥉 **#3** | **`🎄 Jingle bins 🎄`** | **68.813070** | `+0.0318` | Recursive rectangle combiner (`~70.0`) → C++ SA + tray-shaking push (`~69.5`) → **customized Rust `sparrow`** (direct square `max(W,H)`, warm-start, 2/4-way symmetry, center-weighted `~22–24°` orientation bias) + $N\pm 1$ sampling | [Writeup](https://www.kaggle.com/competitions/santa-2025/writeups/third-place-solution-a-customized-sparrow-algorit) |
| 🏅 **#4** | **`bowwowplusa`** | **68.852415** | `+0.0712` | `(w, h)` bucket tile DB + 8-angle `sparrow` (`22.5°/202.5°` core) + **automated corner-region transplant** across $N' \to N$ refined via SA | [Writeup](https://www.kaggle.com/competitions/santa-2025/writeups/4th-place-hybrid-packing-with-sparrow-manual-str) |
| 🏅 **#5** | **`shr`** (`saharan`) | **68.876417** | `+0.0952` | **Custom `float32` SIMD PBD / Nonlinear Gauss-Seidel physics "diffusion"** (`4e-7` gap margin) + **8–12 replica Parallel Tempering SA** over local `8–16` tree neighborhoods + $N\pm 1 / N\pm 2$ transfer | [Writeup](https://www.kaggle.com/competitions/santa-2025/writeups/santa-2025-5th-place-solution) · [GitHub](https://github.com/saharan/santa-2025) |
| 🏅 **#6** | **`sparrow`** (J. Gardeyn, T. Wauters) | **68.907046** | `+0.1258` | Authors of open-source Rust `sparrow`/`jagua-rs` (Guided Local Search weights on persistent collisions + stepped compression) powering `#3, #4, #8, #9, #10, #12, #13` | [GitHub](https://github.com/JeroenGar/sparrow) · [Post](https://www.kaggle.com/competitions/santa-2025/discussion/671593) |
| 🏅 **#9** | **`redblackbst`** | **69.073356** | `+0.2921` | **4-part convex decomposition + 4,096-bin SAT LUT** (`L1` severity > `L2/L∞`), Owen-Sobol + Poisson-disc seeding, & rotation-invariant Motif Mining inside `sparrow` | [Writeup](https://www.kaggle.com/competitions/santa-2025/writeups/exploiting-homogeneity-in-feasibility-driven-polyg) |
| 🏅 **#12–#13** | **`Chaotic Elves` / `Tree Musketeers`** | **69.131 / 69.142** | `+0.35–0.36` | Rigid merged concave-hull bulk core (beating `sparrow`'s `~50`-polygon cliff), **aspect-ratio perturbation (`1.0→0.9–1.1→1.0`)**, **buffer-zone $N\to N-1$ removal** (pushed #1's `68.780→68.765` post-comp!), & C++ Billiard bisection squeeze | [#12](https://www.kaggle.com/competitions/santa-2025/writeups/santa25-12th-place) · [#13](https://www.kaggle.com/competitions/santa-2025/writeups/13th-place-santa25-billiard-customized-sparrow) |

---

## 1. Universal Combinatorial & Geometric Optimization Breakthroughs

1. **Fixed-Box Feasibility Relaxation + Discrete Shrink > Joint Container Minimization (`#1`, `#2`, `#6`)**:
   - Minimizing bounding-box size $L$ simultaneously inside the main GA/SA cost causes premature jamming into high-energy local minima ("glass states").
   - Every top algorithm decouples them: **fix a slightly shrunk box $L - \delta$, minimize overlap/boundary violation to zero via continuous relaxation or LNS/GLS, then step-shrink $L$** (enabling joint box-size minimization ONLY during final `legalize`, `#1`).
2. **Ultra-Fast Precomputed Collision Kernels (`30,000` Relaxations/s, `#1`, `#2`, `#5`, `#9`)**:
   - For identical rigid shapes, replacing inline polygon-intersection checks with a **precomputed 3D Minkowski separation lookup table `(Δx, Δy, Δθ)`** interpolated in CUDA (`#1`) or a **4,096-bin convex-part SAT LUT** (`#9`) unlocked orders-of-magnitude faster continuous L-BFGS / PBD relaxation (`30k/s` on RTX 5090 `#1`, `2×` SIMD float32 throughput `#5`).
   - Always include an explicit **safety gap (`4e-7`, `#5`)** and run final verification/legalization against the **exact official precision engine (`Shapely v2.1.2` + `Decimal('1e18')`)**, because `f32`/`f64` diverge from `Decimal` on grazing touches.
3. **Asymptotic Bulk Lattices, Core Freezing, Symmetry & Cross-$N$ Transfer (`#1`–`#5`, `#12`, `#13`)**:
   - Solve the infinite periodic packing problem first (**non-perpendicular 2-tree dimer `8277/27200 = 0.304301`** beat orthogonal `0.310709`), align along optimal square angles (`~22–24° / 202–204°`, `#3`, `#4`), and **freeze or merge the rigid interior crystal into a single hull polygon (`#8, #12, #13`)** so discrete solvers focus strictly on boundary irregularities.
   - **Exact 180° rotational symmetry** halved degrees of freedom for even $N$ (`#1`, scaling random GA from $N\approx 40 \to 70$), whereas 90° symmetry was suboptimal (`#1`, `#5`).
   - **Cross-Instance Structural Transfer**: Propagating solutions across $N \leftrightarrow N\pm 1 / N\pm 2$ via **buffer-zone removal (`#12` improved #1's `68.780 → 68.765`!)**, **corner-region transplant (`#4`)**, or **aspect-ratio perturbation (`1.0 → 0.9–1.1 → 1.0`, `#12`)** escaped single-$N$ local basins.
4. **Diversity-Preserving Metaheuristics (`#1`, `#2`, `#5`)**:
   - Prevent premature convergence via **ring-topology islands + Hungarian-distance diversity filtering (`#1`)**, **symmetry-invariant bottleneck-distance beam deduplication (`#2`)**, or **8–12 replica Parallel Tempering (`#5`)** targeting local high-disorder neighborhoods.

---

## 2. Master Anti-Patterns & Empirical Negative Results

| Tested Hypothesis | Verified Impact | Teams Reporting | Root Cause |
| :--- | :--- | :---: | :--- |
| **Jointly Optimizing Box Size Inside Main GA / Single-Run Soft SA / Direct DE** | Failed vs fixed-box shrink (`#1`); soft SA stalled at `~70.0` (`#11`); DE stalled at `~71.x` (`#14`) | #1, #11, #14 | Simultaneous shrinkage jams polygons into irreversible glassy states |
| **Unmodified `sparrow` Above `~50` Free Polygons & Naive RL Packing** | `sparrow` degraded above `~50` objects (`#13`); RL gave "abysmal results" (`#9`) | #9, #13 | Single-item moves cannot reorganize dense clusters (merge bulk core blocks instead); RL lacks exact geometric gradients |
| **Inline Convex Overlap / Star-Hypercube Islands / Tournament Selection** | Slower & worse than CUDA Minkowski LUT + ring islands + diversity filtering (`#1`) | #1 | Inline checks throttle batch throughput; aggressive selection collapses genotype diversity |
| **Uniformly Scaling Centers $(1+\epsilon)(x_i, y_i)$ Without Geometry Shift** | **"Hooking in expansion" (`#13`)**: introduces NEW collisions from collision-free states! | #13 | Non-convex branches hook around each other under pure center dilation |
| **Trusting `f32`/`f64` Zero-Overlap Without Official `Decimal`/`Shapely` Legalization** | Silent submission rejections (`ParticipantVisibleError`) | #2, #5 | Grazing contacts flip inside/outside under `Decimal('1e18')` + `float(deg)` rounding |
