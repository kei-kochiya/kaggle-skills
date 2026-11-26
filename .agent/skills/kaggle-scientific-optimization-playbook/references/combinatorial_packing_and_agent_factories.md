# Combinatorial Packing, Collision LUTs & Autonomous Multi-LLM Factories

Concrete, transferable algorithms for NP-hard geometric optimization and autonomous multi-agent competitive engineering.

---

## 1. Fixed-Box Feasibility Search & Precomputed Collision LUTs

1. **Decouple Container Size From Feasibility Relaxation**:
   - Optimizing container side length $L$ simultaneously inside SA/GA/L-BFGS causes premature jamming into high-energy "glassy states".
   - Standard winning loop:
     1. Set fixed target box $L_{\text{target}} = L_{\text{best}} - \delta$.
     2. Minimize smooth violation energy $E = \alpha \sum_{i<j} \text{overlap}_{ij}^2 + \beta \sum_i \text{outside}_i^2$ via batched GPU L-BFGS, SIMD PBD, or Guided Local Search until $E = 0$.
     3. Step-shrink $L_{\text{target}}$ and repeat; allow $L$ as a free variable **only during final `legalize` polishing**.
2. **Precomputed Collision Acceleration**:
   - **Identical Rigid Polygons (GPU)**: Precompute exact pairwise separation distance via Minkowski differences into a **3D lookup table `(Δx, Δy, Δθ)`** queried with trilinear interpolation in CUDA — enables tens of thousands of batched L-BFGS relaxations/sec with smooth gradients.
   - **CPU/SIMD (Rust/C++)**: Use Structure-of-Arrays (SoA) 3-tier AABB pruning (`shape AABB → edge AABB → exact segment cross`) in AVX2, or decompose concave polygons into $M$ convex parts + a `4,096`-bin relative-angle SAT LUT (`L1` overlap severity).
   - **Precision Legalization Contract**: If running `float32` SIMD (`2×` throughput), enforce a safety gap (`eps_gap ≈ 4e-7`) and **always validate & legalize every final candidate through the competition's exact evaluation library** (`Decimal` / `Shapely` / `GEOS`) — grazing contacts routinely invert between `f32`/`f64` and arbitrary-precision graders.
3. **Asymptotic Bulk Cores, Symmetry & Cross-$N$ Transfer**:
   - Solve the infinite periodic crystal first (search **non-perpendicular** spanning vectors, not just orthogonal grids!).
   - Above `~50` free objects, freeze or merge the rigid interior lattice into one concave hull polygon so discrete local search focuses 100% on the irregular boundary.
   - Exploit **180° rotational symmetry** to halve DOFs on even $N$ (audit whether 90° symmetry over-constrains packing!), and propagate discoveries across $N \leftrightarrow N \pm 1$ via **buffer-zone removal**, **corner-region transplants**, and **aspect-ratio perturbations (`1.0 → 0.9–1.1 → 1.0`)**.
   - Avoid uniform center dilation $(1+\epsilon)(x_i, y_i)$ on non-convex shapes ("hooking in expansion" creates *new* collisions from valid layouts!).

---

## 2. Autonomous Multi-LLM Engineering Factories & MCKP Budgeting

Battle-tested patterns proven across multi-agent tabular & 400-task neural-circuit optimization campaigns:

1. **Force Blank-Slate Architectural Rewrites (`10×` ROI Over Local Pruning)**:
   - Greedy incremental agents plateau rapidly (`+0.05`/task vs `+0.50`/task for structural rewrites).
   - Require agents to explicitly price **2–3 fundamentally different paradigms** before editing code, and periodically swap task partitions across sub-teams/agents to escape local basins.
2. **Vertical Deep Thinking + Horizontal Proof-Verified Idiom Registries**:
   - Pair high-reasoning single-turn sessions (discovering breakthrough idioms) with fast parallel coding workers (propagating idioms across all remaining tasks).
   - Maintain three lightweight shared memory artifacts:
     - **`cookbook.md` / Voted Idiom Registry**: Every promoted trick stores a verifiable checksum/proof (`model_md5` or CV hash) plus `+1/+2/-` hit-rate tallies across tasks.
     - **`negative_knowledge.md`**: Explicitly logs dead ends so fresh agents don't burn compute repeating failed hypotheses.
     - **Single-Commit Blackboard (`CURRENT.md`)**: Steer parallel workers atomically via one version-controlled instruction file.
3. **Transactional Atomic Units & Independent Stress Verification**:
   - Structure every task/experiment as a self-contained transactional bundle (`generator.py`, `notes.md`, `artifact`) wrapped in **`backup → static_lint → stress_validate → promote_or_restore`** so unattended multi-day runs accumulate **zero regressions**.
   - **Never trust local provided samples alone** (LLM-claimed gains frequently fail hidden edge cases ~70% of the time!). Validate against independent reference solvers + thousands of fresh synthetic stress inputs + static graph/schema linters.
4. **Global Runtime Assembly via Multiple-Choice Knapsack (MCKP)**:
   - Whenever final scoring evaluates a bundle of $M$ tasks/models under a hard global Kaggle wall-clock budget $T_{\max}$, archive multiple Pareto-optimal `(score_{i,k}, runtime_{i,k})` candidates per task $i$ and solve the exact **MCKP**:
     $$\max \sum_{i=1}^M \sum_{k} s_{i,k} x_{i,k} \quad \text{s.t.} \quad \sum_k x_{i,k} = 1\;\forall i,\quad \sum_{i,k} t_{i,k} x_{i,k} \le T_{\max} - T_{\text{margin}}$$
   - Guarantees maximum leaderboard score with zero timeout disqualifications.
5. **Downstream-Cost-Aware Compression & Exact Subproblem DP**:
   - Never optimize raw code/artifact size when the grader scores **post-compression / escaped byte length**—score every LLM sample, fuzzer edit, and rewrite *after* final compression (preferring "DRY" repetition and uniform symbol tables when they shrink compressed entropy).
   - Replace noisy GA/SA on structured discrete subproblems (optimal Huffman code lengths, escaped-literal LZ77 parsing) with **exact Dynamic Programming**, reserving 2-stage **liveness-graph Optuna → Simulated Annealing** for non-separable outer variable naming, and trigger targeted LLM exploration via **AST rule prompts + empty-sandbox blank-slate restarts**.
