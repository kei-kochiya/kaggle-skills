# Behavioral Cloning, Replay Engineering & Promotion Gating

Transferable patterns for turning competition replays and heuristics into strong neural policies, avoiding imitation/RL pitfalls, and gating model promotions reliably. Read the [simulation workflow](../SKILL.md) first to establish your baseline and game contract.

---

## 1. Paradigm Selection by Compute & Action Structure

| Regime / Constraint | Recommended Paradigm | Why It Works |
| :--- | :--- | :--- |
| Strong replays exist; low compute budget or noisy RL | **Pure BC on Executed Transitions** | High stability, fast iteration, zero RL credit-assignment collapse |
| Complex micro-logistics cause collisions or deadlocks | **Hierarchical Macro-BC + Deterministic Micro Compiler** | Neural net predicts periodic strategic intent; compiled router executes legal micro-steps |
| High-entropy opening + mathematical endgame | **Split-Horizon Hybrid (BC Opening + Combinatorial Endgame)** | Neural intuition navigates uncertain openings; exact solver optimizes terminal conversion |
| Large compute + fast verified simulator | **BC Warm-Start → Frozen-Actor Critic Pre-Warm → Self-Play/League RL** | Escapes imitation ceiling while avoiding from-scratch exploration collapse |

---

## 2. Replay Ingestion & Executed-State Labeling

### 2.1 The Phantom-Order Trap
In turn-based or simultaneous-action simulations, strong bots often over-order or issue conflicting commands knowing the engine will silently drop invalid orders. Training BC directly on raw `submitted_action` teaches the student to waste capacity on phantom commands.

### 2.2 Executed-Transition Verification Protocol
1. **Derive Applied Actions**: Step through each replay with the official (or parity-verified) engine and extract only the actions/state deltas the engine actually executed.
2. **Round-Trip Parity Gate**: Drop any episode where `encode(obs) → decode(labels) → step(engine)` fails to reproduce the recorded terminal score/state within tolerance.
3. **Episode-Level Splitting & Deduplication**: Keep all turns and all seats of an episode in the same train or validation fold; deduplicate repeated episode/seat records across scrapes.

```python
def extract_verified_transitions(episode, engine, encode_obs, derive_executed_action, verify_terminal):
    """Yield training pairs only from episodes passing full state-transition parity."""
    transitions = []
    engine.reset_from_episode(episode)

    for t in range(len(episode) - 1):
        obs_t = episode[t]["observation"]
        next_obs = episode[t + 1]["observation"]
        executed_action = derive_executed_action(obs_t, next_obs)
        engine.step(executed_action)
        transitions.append((encode_obs(obs_t), executed_action))

    if not verify_terminal(engine.state, episode[-1]["observation"]):
        return []  # Reject corrupted or misaligned replay
    return transitions
```

### 2.3 Multi-Teacher Conditioning & Checkpoint Soups
Instead of discarding sub-optimal replays when building a diverse BC corpus:
- **Conditioning Embeddings**: Tag each trajectory with `(Teacher/Style ID, Recency Bucket, Outcome)` and randomly drop the style embedding (`~10%`) during training to force generalization. At inference, prompt with `(Top Teacher, Latest Recency, Win)`.
- **Pretrain → Narrow Fine-Tune → Weight Soup**: Pretrain on a broad multi-teacher corpus for coverage, briefly fine-tune on the single strongest elite style (or solver trajectories), and average weights across top fine-tuned checkpoints (**Checkpoint Soup**) for variance reduction.
- **Rare-Action Loss Weighting**: Up-weight rare, game-deciding strategic actions (e.g., permanent expansions or terminal transitions occurring only 1–3 times per episode) so frequent `PASS`/movement turns do not dominate cross-entropy gradients.
- **Exclude Decided-Match Garbage Time**: In multi-round / Best-of-N games, filter out turns from already-clinched rounds (where strong bots often go passive or conserve clock) before training behavioral cloning.
- **Persistent Belief-State Tracking + Spatial Entropy Channels**: When hidden physics or map parameters persist across multiple rounds of a match, pair the neural policy with a deterministic/Bayesian state tracker (exact constraint elimination or posterior probability maps + **spatial entropy channels**) rather than forcing raw end-to-end pixels to rediscover hidden parameters from scratch.

---

## 3. Fast Multi-Unit Action Decoding & Pure-NumPy CPU Serving

### 3.1 One-Pass Draft + Joint Corrector
Autoregressive decoding across $K$ units/slots per turn is slow on single-core CPUs and introduces arbitrary ordering bias. A fast parallel alternative:
1. Run the entity encoder once to produce simultaneous **draft action logits** for all $K$ decision slots.
2. Feed the $K$ draft decision embeddings into a lightweight **Cross-Decision Corrector** (2–6 Transformer blocks attending across slots) to output coordinated final logits, resolving collisions and shared-resource conflicts in a single batched pass.

### 3.2 Pure-NumPy CPU Export (Zero Framework Overhead)
When a compact policy (`<= 10M` params) faces tight startup or per-turn CPU budgets, exporting weights to `.npz` and implementing the forward pass in pure NumPy (matrix multiplies, manual LayerNorm, exact GELU/SiLU) avoids PyTorch/JAX import overhead, reduces memory footprint, and yields predictable sub-100ms CPU step latency.

---

## 4. Transitioning from BC to Stabilized RL

1. **Frozen-Actor Critic Pre-Warm**: Before enabling policy gradients, freeze the BC encoder/actor and fit only the value head for 50–100 rollout iterations so initial GAE advantages are not dominated by random critic noise.
2. **Sequential Conflict Masking Inside Rollouts**: Compute parallel unit/slot logits in one forward pass, then sample sequentially in engine execution order while dynamically masking conflicting targets or depleted resources. Store the exact per-step conditional masks so PPO update recomputation matches collection bit-for-bit.
3. **Exclude Post-Sampling Overrides from Policy Loss**: If deterministic guardrails override sampled actions (e.g., emergency overflow or terminal liquidation rules), exclude the overridden decision slots from actor log-prob gradients while retaining the resulting transitions for critic targets.
4. **Targeted Recovery BC on RL Plateaus**: When self-play RL forgets rare multi-step strategies, run 1–3 epochs of regularized recovery BC (cross-entropy + KL to reference + entropy floor) on targeted demonstrations or solver games, re-warm the critic, and resume PPO with a short learning-rate warmup.
5. **Hard-Example Seed Replay**: Reserve a fraction (`~20–25%`) of each RL rollout batch to replay seeds lost in recent iterations (capped at e.g. 2 replays per seed to avoid memorization), concentrating gradient budget on policy blind spots.
6. **Suppress Cyclic Micro-Arbitrage During RL**: If a zero-sum micro-mechanic induces rock-paper-scissors timing cycles ($A > B > C > A$) that hijack RL updates away from macro strategy, restrict or disable that micro-mechanic during RL training and re-enable an analytical/deterministic module at inference time.
7. **Endgame Mini-Game Debugging**: Slice realistic late-game states with only $x \in [1, 5]$ turns remaining and compare the learned policy against exact short-horizon tree search to rapidly expose value-head, discount, or execution bugs.

---

## 5. Paired-Seed Promotion Gating

Unpaired random seeds have high variance from map/spawn asymmetry. Always evaluate candidates against the incumbent champion on **identical paired seeds with swapped player seats** (plus a separate frozen panel of diverse external opponents):

```python
import math

def run_paired_seed_gate(candidate, champion, play_fn, n_seed_pairs=512, min_wr=0.52, min_z=2.0):
    """
    Compare candidate vs. champion on paired seeds with swapped seats.
    Returns (passed, win_rate, z_score).
    """
    pair_scores = []  # Average score per seed pair in {0.0, 0.5, 1.0}
    for seed in range(n_seed_pairs):
        res_a = play_fn(candidate, champion, seed=seed)  # Candidate in Seat 0
        res_b = play_fn(champion, candidate, seed=seed)  # Candidate in Seat 1
        s_a = 1.0 if res_a == 0 else (0.5 if res_a == -1 else 0.0)
        s_b = 1.0 if res_b == 1 else (0.5 if res_b == -1 else 0.0)
        pair_scores.append(0.5 * (s_a + s_b))

    mean_wr = sum(pair_scores) / len(pair_scores)
    # Cluster-aware SE computed across paired seeds (accounts for seed correlation)
    var = sum((x - mean_wr) ** 2 for x in pair_scores) / max(1, len(pair_scores) - 1)
    se = math.sqrt(var / len(pair_scores))
    z = (mean_wr - 0.5) / se if se > 0 else (float("inf") if mean_wr > 0.5 else 0.0)

    return (mean_wr >= min_wr and z >= min_z), mean_wr, z
```

> **Note on Uncertainty**: Computing standard error over **paired seed means** (`pair_scores`) instead of treating both games of a pair as independent samples avoids understating variance when map seeds are highly symmetric or lopsided.
