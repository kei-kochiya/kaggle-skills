# RL Training Stability, Distributed Scaling & Multi-Agent Game Theory

Training deep neural policies via self-play RL can fail from **policy collapse**, **catastrophic forgetting**, **non-transitive strategy cycles (Rock-Paper-Scissors)**, **homogeneous self-play blind spots**, and **endgame stalling**.

**Scope:** Start from the [simulation workflow](../SKILL.md). Every hyperparameter, discount factor, promotion gate, and league mixture below is a configurable template—derive concrete settings from the active competition's horizon, scoring rule, and compute budget.

---

## 1. Stabilized PPO Rollout & Update Protocol

```text
  [N Parallel Vector Environments]
                |  (Rollout horizon T steps)
                v
  [Rollout Buffer: N × T transitions]
   • Store sampled action IDs, old log-probs, conditional masks, values, forced-action flags
                |
                v
  [Policy Update (1–2 PPO epochs per rollout)]
   • Assert ratio ≈ 1.0 on recomputed log-probs before step 0
   • Global cross-rank advantage normalization
   • Clipped surrogate + Huber/CE value loss + annealed entropy + KL teacher anchor
   • Max gradient norm = 1.0 (Muon for large 2D attention weights + AdamW for 1D/embeddings)
```

### Key Stability Lever Reference

| Lever | Typical Starting Range | Rationale |
| :--- | :--- | :--- |
| `ppo_epochs` | `1–2` | Prevents overfitting to stale on-policy rollouts in non-stationary multi-agent settings |
| `clip_eps` ($\epsilon$) | `0.15–0.20` | Bounds single-update policy step size |
| `gae_lambda` ($\lambda$) | `0.95–1.00` | Trade-off between critic bias and Monte Carlo variance |
| `gamma` ($\gamma$) | Horizon-dependent (`0.99`–`1.0`) | Fixed-horizon games with terminal scoring often need $\gamma = 1.0$; indefinite games may need mild discounting |
| `entropy_coef` | Annealed (`0.01 → 0.001`) | Maintains exploration early without adding noise to late-stage tactical precision |
| `teacher_kl_coef` | `0.05–0.20` | Anchors policy distribution to a verified reference checkpoint against drift |

---

## 2. Teacher Anchoring (KL Distillation & Polyak EMA)

Pure self-play frequently drifts away from previously mastered skills. Anchor updates with a reference teacher distribution:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PPO}} + c_{\text{val}} \mathcal{L}_{\text{critic}} - c_{\text{ent}} \mathcal{H}(\pi_\theta) + \alpha_{\text{KL}} D_{\text{KL}}(\pi_{\text{teacher}} \,\|\, \pi_\theta) + \alpha_{\text{V}} \mathcal{L}_{\text{V}}(V_{\text{teacher}}, V_\theta)$$

Two proven reference update schemes:
1. **Gated Discrete Promotion**: Periodically evaluate the candidate against `checkpoint_best` over paired-seed seat-swapped games (plus an external benchmark panel) and promote only after clearing a statistically significant win-rate threshold.
2. **Delayed Moving Polyak Teacher (EMA)**: Continuously track a smoothed weight anchor:
   $$\theta_{\text{teacher}} \leftarrow \tau \theta_{\text{teacher}} + (1 - \tau) \theta_{\text{student}}, \qquad \tau \in [0.995, 0.9995]$$
   which eliminates sudden target jumps during high-entropy updates.

---

## 3. Behavioral Cloning Warm-Start & Imitation-to-RL Curriculum

When rewards are sparse or action spaces are high-dimensional, starting RL from random weights often collapses completely. Use a staged curriculum:

1. **Stage 1 — Executed-State BC**: Train a supervised policy via cross-entropy on verified valid demonstrations (see [bc_replay_and_gating.md](bc_replay_and_gating.md)), confirming full-game strength on unseen seeds.
2. **Stage 2 — Frozen-Actor Critic Pre-Warm**: Freeze the BC policy and train only the critic until value predictions stabilize.
3. **Stage 3 — Anchored RL Fine-Tuning**: Unfreeze the actor with a conservative learning rate (`1e-5`–`5e-5`), linear warmup, and KL anchor to the BC reference.
4. **Stage 4 — Failure-Driven Recovery**: When RL plateaus or forgets rare skills, inject targeted recovery BC demonstrations, re-warm the critic, and resume RL.

```python
def behavioral_cloning_epoch(model, dataloader, optimizer, max_grad_norm: float = 1.0):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        action_dist = model.get_action_distribution(batch["obs"], mask=batch.get("mask"))
        loss = -action_dist.log_prob(batch["executed_action"]).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))
```

---

## 4. Reward Shaping, Discounting & Action Masks

### 4.1 Diagnosing and Fixing Endgame Stalling
In games that can finish early, setting $\gamma = 1.0$ with win-only terminal reward gives zero gradient between winning in 100 steps vs. stalling until the turn cap—wasting rollout compute on decided states.
- **Piecewise Step-Conditioned Terminal Reward** (only when early victory is possible and aligns with the rules):
  $$R_{\text{terminal}}(t) = \begin{cases} +1.0 & \text{if Win and } t < T_{\text{cutoff}} \\ +0.5 & \text{if Win and } t \ge T_{\text{cutoff}} \\ -1.0 & \text{if Loss} \end{cases}$$
  Avoids noisy per-step shaping penalties that can trigger reckless suicidal rushes while rewarding clean game conversion.
- **Fixed-Horizon Exception**: If every episode runs a fixed number of turns to a terminal score comparison, do **not** add an artificial early-finish bonus; keep $\gamma = 1.0$ (or horizon-matched discounting) so late-game conversion retains full credit.

### 4.2 Action Masking & PPO Support Parity
- Separate **illegal/impossible actions** (must always be masked) from **legal-but-risky actions** (masking vs. letting the agent learn physics/consequences is an empirical ablation).
- Whatever conditional support mask and sampling temperature were applied during rollout collection **must** be reproduced identically during PPO gradient updates.

---

## 5. Multi-Agent Non-Transitivity, Historical Pools & Adversarial Exploiters

Even two-player zero-sum games can cycle non-transitively, and $N \ge 3$-player games add kingmaker/coalition dynamics. Furthermore, **homogeneous self-play** (training only against recent snapshots of one lineage) creates severe **population blindness**: if all self-play agents converge to a polite macro equilibrium (e.g., long-horizon resource accumulation), the critic never learns to defend against aggressive all-in heuristic rushers or edge-case exploits.

### Defense: Heterogeneous League Matchmaking
Combine four opponent sources in training rollouts:
1. **Current Self-Play (`~50%`)**: Drives peak strategic refinement.
2. **Frozen Historical Checkpoints (`~25%`)**: Prevents catastrophic forgetting of older tactics.
3. **Rule-Based / Heuristic Sparring Bots (`~15%`)**: Grounds the policy and critic against deterministic rushers, greedy economies, and public baselines.
4. **Adversarial Exploiter Agents (`~10%`)**: Auxiliary agents trained specifically to beat the current main policy (or optimize an alternate aggressive win condition), exposing critic blind spots before tournament evaluation.

```python
import copy
import numpy as np

class HeterogeneousLeagueMatchmaker:
    def __init__(self, main_policy, heuristic_bots=None, exploiters=None, max_frozen=20):
        self.main_policy = main_policy
        self.heuristics = list(heuristic_bots or [])
        self.exploiters = list(exploiters or [])
        self.frozen = []
        self.max_frozen = max_frozen

    def sample_opponent(self, probs=(0.50, 0.25, 0.15, 0.10)):
        r = np.random.rand()
        p_self, p_hist, p_heur, _ = probs
        if r < p_self or (not self.frozen and not self.heuristics and not self.exploiters):
            return self.main_policy
        if r < p_self + p_hist and self.frozen:
            return self.frozen[np.random.randint(len(self.frozen))]
        if r < p_self + p_hist + p_heur and self.heuristics:
            return self.heuristics[np.random.randint(len(self.heuristics))]
        if self.exploiters:
            return self.exploiters[np.random.randint(len(self.exploiters))]
        return self.frozen[np.random.randint(len(self.frozen))] if self.frozen else self.main_policy

    def maybe_snapshot(self, step: int, interval: int, model):
        if step > 0 and step % interval == 0:
            if len(self.frozen) >= self.max_frozen:
                self.frozen.pop(1)  # Retain earliest anchor at index 0
            self.frozen.append(copy.deepcopy(model.state_dict()))
```
