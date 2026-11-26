# Recommender Systems, Multi-Stage Retrieval & Learning-to-Rank — Universal Recipes

Transferable, cross-competition patterns for large-scale session recommenders (`Recall@K`, multi-objective candidate-generation + reranking) and tabular query/session Learning-to-Rank (`HitRate@K`, `NDCG`, `MAP@K`).

---

## 1. Two-Stage Session Recommender Architecture (`Retrieval → Reranking`)

When catalog cardinality is large ($10^5$–$10^7$ items), end-to-end dense softmax layers hit sparsity and VRAM collapse. Decompose into two decoupled stages:

1. **Leak-Free Temporal / Fold Separation Between Stage 1 and Stage 2**:
   - Never compute co-visitation counts, item statistics, or embeddings over the same future target window used to train the Stage-2 reranker—doing so creates massive target leakage and train-vs-test distribution shift.
   - **Two Clean Protocols**:
     - **Disjoint Temporal Windows**: Build separate Stage-1 statistics/matrices for train targets (period $T-2$), validation targets (period $T-1$), and test (period $T$).
     - **K-Fold Session-Split Feature Separation**: Split the recent training period into $K$ session folds; truncate only 1 fold ($1/K$) as reranker targets while keeping all prior history plus the remaining $(K-1)/K$ full sessions for Stage-1 co-visitation and item statistics.
   - Match local validation truncation strictly to the official test cutoff distribution, and test multi-seed random truncation cutoffs to multiply reranker training samples cleanly.
2. **Stage 1 — Multi-View Recall-Maximizing Candidate Generation**:
   - Optimize Stage 1 strictly for **Recall ceiling (`Recall@100–300`)**, not top-20 precision.
   - Combine orthogonal candidate generators:
     - **Multi-View Co-Visitation Matrices** (built in minutes on GPU via `cuDF` `int32`/`float32` or `Polars`): consecutive transitions (`|Δk| = 1, 2, 3, 6`), forward-only conversion transitions (`view → high-intent`), exponential time decay (`0.5^(Δt / τ)`), recency slices (hours/days/weeks), asymmetric frequency-normalized distance weighting (`w(a,b) = |j-i| / freq(a)` with halved backward weight), inactivity-gap sub-session splitting, and multi-hop beam expansion.
     - **Session Revisit History** + **Log-Naive-Bayes / PMI Posterior** over recent events.
     - **Embedding & Graph Retrievers**: `Word2Vec` / `ProNE` + `cuML` cosine kNN, matrix factorization (`BPR-MF` / `ALS`), next-multi-item neural nets (including variants trained *strictly on non-visited targets* to maximize complementarity with revisit heuristics), and **session-extracted Subgraph GNNs** (avoiding full-catalog softmax!).
   - **Cold/Short-Session Backfill**: Always verify how many test sessions receive `< K` candidates and backfill deterministically via unconstrained full-history co-visitation + recent trending items.
3. **Stage 2 — Feature Engineering, Post-FE Downsampling & Multi-Objective Reranking**:
   - **Highest-Leverage Features**: Multi-view co-visitation score/rank aggregations (`sum/max/min/mean/std/last`) against last item, last `10–40` events, and high-intent events; **sequence index distance (`|idx_a - idx_b|`)**; revisit counts & elapsed `Δt`; Stage-1 candidate rank/score; multi-window item popularity & funnel conversion ratios (`view→cart`, `cart→buy`); embedding cosine stats.
   - **Objective Selection**:
     - Test **`CatBoost PairLogitPairwise`**, **`XGBoost rank:pairwise` / `rank:ndcg`**, and **`LightGBM lambdarank` / `binary`** (note `CatBoost YetiRank` and `LightGBM xendcg` often lag pairwise/NDCG objectives).
     - When objectives share sessions with hierarchical intent weights (e.g., `click < cart < buy`), test a **single unified multi-objective ranker** grouped by session with metric-proportional graded relevance labels alongside per-objective models.
   - **Negative Downsampling**: Drop zero-target sessions and randomly downsample negatives (`5%–40%` depending on positive density) **strictly after** computing all session-level and candidate-level features.
4. **Calibrated Cross-Paradigm Blending & Anti-Patterns**:
   - When blending pairwise-rank GBDTs (unbounded logits) with neural/GNN probabilities in `(0, 1)`, solve a per-objective additive shift `c` minimizing mean difference between `sigmoid(rank_score + c)` and neural probabilities before weighted averaging, or use inverse-rank / index-score (`1.00..0.05`) blending.
   - **Cross-Target OOF Leakage Warning**: Feeding predicted Stage-2 probabilities of one objective (e.g., clicks/carts) as features into another (e.g., orders) frequently causes fake CV spikes when training rows are filtered to positive-target sessions and degrades test performance unless trained strictly on disjoint non-overlapping session subsets.

---

## 2. Tabular Session / Query Learning-to-Rank (`HitRate@K` / `NDCG`)

For structured query-option ranking tables (each `query_id` offers $N \in [2, 3000+]$ tabular candidates with one or few `selected = 1` rows):

1. **Exact Metric Filtering & Temporal Validation**:
   - Audit whether the competition metric filters queries by candidate count (e.g., scoring only groups with `> 10` options)—keep small groups during feature computation and training for signal, but mirror the exact filter in CV.
   - Under seasonal or inventory drift, **temporal holdout validation** (trailing period or multi-window chronological split) is far more reliable than random `GroupKFold(query_id)`, which leaks seasonal pricing/catalog regimes and inflates CV.
2. **High-Leverage Query-Ranking Feature Engineering**:
   - **Label-Free Transductive Count Encoding**: When unseen entities and high-cardinality categoricals shift between train and test, replacing or augmenting categoricals with **label-free frequency counts across `train + test` combined** (plus `1%` random train categorical NaN masking for unseen test levels) strongly improves generalization.
   - **Dual-Level Intra-Group & Duplicate-Entity Ranking**: Compute group-relative ranks (`rank(method='average')`), percentiles `(rank - 1) / (count - 1)`, z-scores, ratios to group min/mean, and tier premiums over **both** `query_id` **and** `(query_id, physical_item_hash)` whenever near-duplicate/multi-tariff variants of the same underlying item coexist in a session.
   - **Strictly Causal User/Entity History (Numeric/Rank Only!)**:
     - Aggregate past `selected = 1` rows per user/organization **strictly over earlier timestamps** (`t_hist < t_cur`) using **numeric and intra-group rank attributes** (e.g., historical average price rank, duration rank, tier preference).
     - **Never aggregate historical categorical IDs** (e.g., past brand/carrier/item IDs) without extreme regularization—they memorize training entities and overfit heavily, whereas historical numeric/rank preferences transfer cleanly to unseen sessions.
3. **Training-Time Group Capping, Normalization & Ensembling**:
   - **Post-FE Training Group Cap**: Capping huge training queries to `50–64` rows (keeping all `selected = 1` rows + uniformly sampled negatives) speeds up training and sharpens ranking splits—**always apply capping *after* computing full-group features**!
   - **Multi-Loss GBDT Averaging & Scale Alignment**:
     - On tabular LTR with heterogeneous numeric/missing features, GBDT rankers (`XGB rank:ndcg / rank:pairwise / rank:map`, `LGBM lambdarank`, `CatBoost QueryCrossEntropy / PairLogitPairwise`) consistently outperform pure neural/Transformer rankers.
     - Normalize predictions **per query group** via **per-group Softmax** or **per-model Min-Max** before ensembling.
     - Under discontinuous top-K hit metrics (`HitRate@3`), **equal-weight averaging across 4–6 diverse-loss GBDTs** frequently beats heavy Optuna weight tuning over dozens of models (which overfits top-K rank boundaries).
   - **Avoid Positional Row-Order Features & Group-Size Biased Pseudo-Labeling**: Raw supplier/API row ordering is frequently non-deterministic across splits, and naive pseudo-labeling biases predictions toward small-group sessions.
