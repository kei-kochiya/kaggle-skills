# RL Training Stability, Distributed Scaling & Multi-Agent Game Theory ⚖️

Training deep transformers with Reinforcement Learning across billions of environment steps is notoriously unstable. Without strict algorithmic guardrails, policies suffer from **policy collapse**, **catastrophic forgetting**, **non-transitive cycles (Rock-Paper-Scissors dynamics)**, and **infinite stall loops**.

This reference guide establishes battle-tested recipes for **Stabilized PPO**, **Teacher Distillation**, **Multi-Agent League Play**, and **Reward Shaping**.

**Scope:** Start with the [new-competition workflow](../SKILL.md). The numerical scaling, promotion and anti-stall examples below come from particular game settings; they are not universal defaults. For fixed-season Kaggriculture, use [the competition reference](kaggriculture.md): its released recipe uses `gamma=1`, terminal win/draw/loss, a fixed teacher reference in the public preset and frozen-actor critic fitting. Two-player self-play also needs external strength evaluation. Apply early-victory incentives only when a game can finish early and the changed reward is justified by measured outcomes.

---

## 1. Production PPO Configuration & Scaling Protocol

PPO is one candidate on-policy algorithm. Choose it using action structure, available infrastructure and the experiment budget; throughput and scaling must be measured. The [PPO paper](https://arxiv.org/abs/1707.06347) defines its objectives. The diagram below illustrates a large training configuration rather than a starting requirement.

```
                      DISTRIBUTED PPO ROLLOUT & UPDATE CYCLE
                      
 [8,192 Parallel Environments on CPU Hosts (Rayon Threads)]
                          |
                          v  (Step T=64 Rollout)
 [Rollout Buffer: 8,192 × 64 = 524,288 Env Steps]
                          |
                          v  (Asynchronous Direct CUDA Stream)
 [Distributed Data Parallel Across 32 GPUs (NCCL)]
  ├── Minibatch Size: 16,384 transitions per worker
  ├── PPO Epochs: 1–2 epochs per rollout (prevents policy drift)
  ├── Gradient Clipping: Max norm = 1.0
  └── Optimizers: Muon (2D weight matrices) + AdamW (1D vectors/biases)
                          |
                          v
 [Loss Computation: PPO-Clip + Teacher Distillation + Value CE + Entropy]
```

### 1.1 Hyperparameter Gold Standard

| Parameter | Recommended Value | Strategic Rationale |
| :--- | :--- | :--- |
| `rollout_length` ($T$) | $64$ steps | Balances trajectory horizon with GPU VRAM capacity. |
| `n_envs` | $2,048 - 8,192$ | Maximizes environment throughput and eliminates sample correlation. |
| `ppo_epochs` | $1$ | **Single-epoch PPO** avoids overfitting to recent rollout data in multi-agent games. |
| `clip_eps` ($\epsilon$) | $0.20$ | Standard clipping radius for surrogate objective. |
| `gae_lambda` ($\lambda$) | $0.95$ | Balances bias and variance in advantage estimation. |
| `gamma` ($\gamma$) | Game-specific; examples $0.99 - 0.995$ | Test discounting against the actual horizon/objective. Fixed-season Kaggriculture uses $1.0$ without an early-finish bonus. |
| `learning_rate` | $1 \times 10^{-4} \to 1 \times 10^{-5}$ | Linear warmup (10M steps) followed by cosine decay. |
| `optimizer` | **Muon + AdamW** | Muon for large 2D attention matrices ($\ge 25M$ params); AdamW for embeddings. |
| `entropy_coef` ($c_{\text{ent}}$) | $0.01 \to 0.001$ | Annealed gradually as policies stabilize. |
| `teacher_kl_coef` | $0.10$ | Distillation penalty against historical anchor checkpoint. |

---

## 2. Teacher Distillation & Checkpoint Promotion Protocol

In self-play RL, training against only the current policy version causes cyclic degeneration: the policy unlearns tactics it mastered days earlier. To prevent this, anchor the training policy against a **Teacher Checkpoint**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PPO}} + c_{\text{val}} \mathcal{L}_{\text{critic}} + c_{\text{ent}} \mathcal{H}(\pi) + \alpha_{\text{KL}} D_{\text{KL}}(\pi_{\text{teacher}} \,\|\, \pi_{\theta}) + \alpha_{\text{CE}} \mathcal{L}_{\text{CE}}(V_{\text{teacher}}, V_{\theta})$$

```
                       TEACHER PROMOTION GAUNTLET
                       
 [Current Training Policy π_θ]  vs  [Last-Best Teacher Checkpoint π_teacher]
                                |
                                v
               [2,048 Head-to-Head Evaluation Games]
               • 50% 2-player / 50% 4-player matches
               • Player seating randomly permuted across seats
                                |
                +---------------+---------------+
                |                               |
        Win Rate < 70%                   Win Rate >= 70%
                |                               |
                v                               v
      [Keep Current Teacher]          [PROMOTE NEW TEACHER]
      Continue training π_θ           • Replace checkpoint_last_best.pt
                                      • Update teacher KL & CE targets
```

### Checkpoint Promotion Rules
1. Every $20\text{ million}$ environment steps, save a numbered checkpoint.
2. Evaluate the checkpoint against `checkpoint_last_best.pt` over $2,048$ games with seats randomly shuffled.
3. **Example Promotion Gate**: A $70\%$ threshold is a particular recipe. Set a promotion criterion from independent opponents, seat/scenario design, sample uncertainty and cost. A lower measured gain may be useful with adequate evidence; one head-to-head threshold does not prove broad strength. A fixed reference is another option.

---

## 3. Multi-Agent Game Theory: Self-Play vs League Play

### 3.1 Player count and opponent diversity

Two-player zero-sum games can be non-transitive: rock-paper-scissors is a direct example. The minimax theorem is not a convergence guarantee for an arbitrary neural self-play/PPO implementation. Multiplayer rewards may be zero-sum or general-sum depending on their definition; inspect the game's actual payoff.

An opponent pool can expose blind spots in either setting. Multiplayer games may additionally have kingmaker situations where one losing player's choices change which leader wins. Use external evaluation and consider historical policies, heuristics or exploiters when current-policy self-play misses relevant strategies. [Game-theoretic multi-agent learning](https://arxiv.org/abs/1711.00832).

### 3.2 AlphaStar-Style League Play for Multiplayer Games
When evidence supports a broader training opponent distribution, consider an **Agent League**. The mixture below is an example; opponent selection is part of the experiment:

```
                           LEAGUE MATCHMAKING POOL
                           
          +----------------------------------------------------+
          |                     LEAGUE POOL                    |
          |                                                    |
          |  [Main Agent π_θ]                                  |
          |  • 50% games vs current self-play                  |
          |  • 35% games vs historical checkpoints             |
          |  • 15% games vs exploiters                         |
          |                                                    |
          |  [Historical Checkpoint Pool]                      |
          |  • Checkpoints saved at 100M, 500M, 1B, 5B steps   |
          |                                                    |
          |  [Exploiter Agents]                                |
          |  • Trained specifically to maximize win rate       |
          |    against the current Main Agent                  |
          +----------------------------------------------------+
```

---

## 4. Reward Engineering & Post-Mortem Pitfalls

### 4.1 The Discount Factor Dilemma & The Stalling Bug
- **Conditional Pitfall**: With win-only reward, no intermediate rewards and `gamma=1`, equally likely early and late victories have the same expected return. A scalar critic represents expected return, not automatically a win probability: `+1/0/-1` targets give expected win-minus-loss. A fixed season can correctly use `gamma=1`.
- **Symptom**: Once an agent gains a decisive lead (e.g. controls 80% of ships), it refuses to launch finishing attacks. It hoards units and circles passively until the turn limit, burning billions of compute steps on trivial states.
- **Battle-Tested Fixes**:
  1. **Mild Temporal Decay**: Set $\gamma = 0.99$ or add a per-turn cost:
     $$r_t = -0.001 \quad \text{for non-terminal turns}$$
  2. **Early-Termination Experiment**: Ending a rollout from a critic confidence threshold changes the game and can label a mistaken estimate as a win. Keep it outside parity claims and verify strength in full official games; a critic estimate alone is not a proven terminal result.

### 4.2 Action masks and strategic filters

Distinguish impossible actions, joint resource constraints and legal but risky choices. A game-specific experiment may allow risky choices during exploration; this does not establish a general unmasked-training protocol or a fixed point at which to enable masks.

For PPO, sampling and update-time probability recomputation must use matching conditional support and temperature. Sequential choices reserve resources in the game's execution order. Treat a deterministic repair as a separate controller component and retain proposed versus executed choices. Kaggriculture's rule-aware controller is an example where consistent masks are central to the adopted training.

---

## 5. Behavioral Cloning Warm-Start & Imitation-to-RL Curricula

As demonstrated by `simjeg` (2nd Place) and `TonyK` (5th Place), starting RL from randomly initialized policies in environments with sparse rewards or complex continuous physics can waste millions of compute steps wandering through uninformative states.

```
                      PROGRESSIVE IMITATION-TO-RL CURRICULUM
                      
 Stage 1: Tournament Replay Ingestion (Top-10 Human / Bot Games)
                            |
                            v
 Stage 2: Supervised Behavioral Cloning (BC)
  • Minimize Cross-Entropy Loss: L_BC = -log π_θ(a_expert | s)
  • Uses recorded valid decisions; verify complete held-out student games
                            |
                            v
 Stage 3: RL Policy Fine-Tuning (PPO / Asynchronous IMPALA)
  • Initialize actor-critic trunk from BC weights
  • Fine-tune with clipped surrogate objective and low learning rate (1e-5)
  • Adds self-play exploration beyond the expert replay distribution
                            |
                            v
 Stage 4: Failure-Driven Improvement
  • Improve teachers/controllers or collect targeted demonstrations
  • Compare another BC/RL cycle with the preserved best agent
```

```python
def behavioral_cloning_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        obs, expert_actions = batch["obs"], batch["action"]
        
        # Forward pass through actor policy
        action_dist = model.get_action_distribution(obs)
        
        # Supervised negative log-likelihood loss
        loss = -action_dist.log_prob(expert_actions).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
```

---

## 6. Step-Conditioned Anti-Stall Reward Shaping

The undiscounted $\gamma=1.0$ stalling crisis occurs because an agent that has a 95% win probability receives identical terminal reward whether it wins on turn 100 or turn 500.

`simjeg`'s battle-tested solution applies a **step-conditioned piecewise terminal reward**:

$$R_{\text{terminal}}(t) = \begin{cases} +1.0 & \text{if Player Wins and } t < T_{\text{cutoff}} \\ +0.5 & \text{if Player Wins and } t \ge T_{\text{cutoff}} \\ -1.0 & \text{if Player Loses} \end{cases}$$

Where $T_{\text{cutoff}} = 500$ (or roughly half the max game horizon).
*Why this works*:
1. Does not distort intermediate policy gradients with noisy step penalties ($r_t = -0.001$ can sometimes cause suicidal rushes).
2. Creates an unmistakable gradient favoring early, decisive planetary annihilation over passive hoarding.

---

## 7. Frozen Historical Opponent League Pools & Polyak Teachers

Current-policy self-play can leave blind spots to other strategies in both two-player and multiplayer games. A **Frozen Opponent Matchmaking Pool** is one option to test. The example below illustrates retaining historical opponents; set its mixture and size from actual evaluation needs.

```python
class LeagueMatchmaker:
    def __init__(self, main_policy, max_frozen=20):
        self.main_policy = main_policy
        self.frozen_checkpoints = [] # Pool of (step_count, model_weights)
        self.max_frozen = max_frozen

    def sample_match_opponents(self):
        # 50% chance: Self-play against current policy
        # 35% chance: Play against a random historical frozen checkpoint
        # 15% chance: Play against the earliest 'rush-bot' baseline
        r = np.random.rand()
        if r < 0.50 or len(self.frozen_checkpoints) == 0:
            return self.main_policy
        elif r < 0.85:
            idx = np.random.randint(0, len(self.frozen_checkpoints))
            return self.frozen_checkpoints[idx]
        else:
            return self.frozen_checkpoints[0] # Earliest anchor

    def maybe_freeze_checkpoint(self, step, model):
        if step % 50_000_000 == 0:
            if len(self.frozen_checkpoints) >= self.max_frozen:
                self.frozen_checkpoints.pop(1) # Keep earliest anchor, rotate middle
            self.frozen_checkpoints.append(copy.deepcopy(model.state_dict()))
```

### Delayed Moving Polyak Teacher
Instead of updating the teacher anchor in discrete jumps, update an Exponential Moving Average (EMA) teacher continuously:
$$\theta_{\text{teacher}} \leftarrow \tau \theta_{\text{teacher}} + (1 - \tau) \theta_{\text{student}}, \quad \tau = 0.999$$
This smooths out policy jitter and prevents advantage variance spikes during high-entropy exploration phases.

---

## 8. Preventing Homogeneous Self-Play Blind Spots: The Maze Crawler Lesson

A critical failure mode in competitive self-play is **Population Homogeneity Blindness**, demonstrated in the *Kaggle Maze Crawler* competition (3rd Place Daniel Bekker):

### The Phenomenon:
1. When an agent trains strictly against itself or its past checkpoints, the population converges to an internally consistent equilibrium.
2. In Maze Crawler, this resulted in an agent that perfected long-term energy farming and survival, because both players in self-play played politely.
3. However, the agent was completely blind to aggressive combat rushes: the 1st place solution (Maksim Savelev) banked sufficient energy, charged across the maze, dropped a 300-energy unit, and forced a lethal collision at turn 120. Because the RL agent had never faced an opponent whose objective was an immediate collision rush, its value critic assigned near-zero risk to approaching enemy bases.

### The Fix: Adversarial League Exploiters
To immunize the main policy against predatory rushes, allocate a partition of training rollouts to **dedicated exploiter agents**:

```python
class AdversarialLeagueTrainer:
    """
    Maintains an active league containing:
    1. Main Agent (optimizes global win condition)
    2. Exploiter Agents (explicitly trained to defeat the Main Agent using edge-case policies)
    3. Heuristic Sparring Bots (rule-based combat rushers, greedy miners)
    """
    def __init__(self, main_policy, exploiter_policies, heuristic_bots):
        self.main_policy = main_policy
        self.exploiters = exploiter_policies # e.g. CombatRusherPolicy
        self.heuristics = heuristic_bots     # e.g. ScoredBFSRushBot
        
    def get_opponent_distribution(self):
        # 50% Main Agent self-play (stabilizes global meta)
        # 25% Historical Checkpoints (prevents catastrophic forgetting)
        # 15% Heuristic Sparring Bots (grounds policy against rule-based exploits)
        # 10% Active Adversarial Exploiter (attacks policy blind spots)
        return {
            "self": 0.50,
            "historical": 0.25,
            "heuristic": 0.15,
            "exploiter": 0.10
        }
```
This forces the value function to learn that close proximity to an aggressive opponent without defensive unit deployment is an immediate failure state, closing the gap between self-play perfection and open-tournament robustness.

