# RL Training Stability, Distributed Scaling & Multi-Agent Game Theory ⚖️

Training deep transformers with Reinforcement Learning across billions of environment steps is notoriously unstable. Without strict algorithmic guardrails, policies suffer from **policy collapse**, **catastrophic forgetting**, **non-transitive cycles (Rock-Paper-Scissors dynamics)**, and **infinite stall loops**.

This reference guide establishes battle-tested recipes for **Stabilized PPO**, **Teacher Distillation**, **Multi-Agent League Play**, and **Reward Shaping**.

---

## 1. Production PPO Configuration & Scaling Protocol

Proximal Policy Optimization (PPO) is the preferred algorithm for competitive simulation because it scales linearly with distributed data parallelism without requiring the delicate actor-learner queue balancing of off-policy architectures (e.g. IMPALA / Ape-X).

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
| `gamma` ($\gamma$) | $0.99 - 0.995$ | **Avoid $\gamma=1.0$** to prevent non-terminating stall strategies. |
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
3. **The 70% Win-Rate Gate**: Do not promote on marginal improvements (e.g. 52% or 55%). A candidate replaces the teacher **only if it wins $\ge 70\%$ of games**. This filters out stochastic evaluation noise.

---

## 3. Multi-Agent Game Theory: Self-Play vs League Play

### 3.1 The 2-Player vs 4-Player Fundamental Divergence
- **2-Player Zero-Sum Games**:
  - Minimax theorem guarantees that self-play with fictitious play or regularized PPO converges toward a Nash Equilibrium. Intransitive cycles are limited.
- **4-Player Multiplayer Games**:
  - The game is non-zero-sum from an individual player's perspective.
  - **Severe Non-Transitivity**: Policy A beats Policy B, Policy B beats Policy C, but Policy C crushes Policy A (Rock-Paper-Scissors).
  - **Kingmaker Scenarios**: A losing player can unpredictably determine which of the remaining leaders wins.

### 3.2 AlphaStar-Style League Play for Multiplayer Games
When competing in 3+ player games, replace pure self-play with an **Agent League**:

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
- **The Pitfall**: Setting $\gamma = 1.0$ (undiscounted) makes value heads mathematically equal to terminal win probabilities. However, the agent has **zero incentive to end the game early**.
- **Symptom**: Once an agent gains a decisive lead (e.g. controls 80% of ships), it refuses to launch finishing attacks. It hoards units and circles passively until the turn limit, burning billions of compute steps on trivial states.
- **Battle-Tested Fixes**:
  1. **Mild Temporal Decay**: Set $\gamma = 0.99$ or add a per-turn cost:
     $$r_t = -0.001 \quad \text{for non-terminal turns}$$
  2. **Surrender / Fast-Forward Trigger**: If the critic estimates win probability $P_{\text{win}}(p) > 0.98$ continuously for 30 turns, declare game over and award victory immediately during training rollouts.

### 4.2 The Action Masking Paradox
- **The Paradox**: Masking illegal or suicidal actions (e.g. launching fleets directly into the sun) seems intuitive. However, empirical testing in Orbit Wars proved that **training with action masks produced an inferior policy**.
- **The Explanation**:
  - When the mask blocks bad moves, the neural network treats the environment as a magic black box where the sun does not exist.
  - Without the mask, the network experiences catastrophic ship vaporization whenever it ignores orbital physics, forcing the internal transformer representations to model celestial gravity, velocity vectors, and hazard boundaries.
- **Standard Protocol**:
  - **Phase 1 (Exploration & Physics Learning)**: Train **unmasked**. Let the model fail and internalize environment dynamics.
  - **Phase 2 (Competitive Fine-Tuning & Test Time)**: Enable the action mask during the final 5% of training and enforce it during test-time inference.

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
  • Reaches ~Top 10 Elo baseline in hours with zero simulator stepping
                            |
                            v
 Stage 3: RL Policy Fine-Tuning (PPO / Asynchronous IMPALA)
  • Initialize actor-critic trunk from BC weights
  • Fine-tune with clipped surrogate objective and low learning rate (1e-5)
  • Adds self-play exploration beyond the expert replay distribution
                            |
                            v
 Stage 4: From-Scratch RL Realignment (Optional Final Polish)
  • Train final submission from scratch once hyperparameters & architectures
    are proven, eliminating any suboptimal habits copied from expert replays
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

In multi-agent environments with $N \ge 3$ players, policies trained purely against current self-play rapidly develop blind spots to earlier strategies (e.g. early rush attacks). `TonyK` (5th Place) and AlphaStar solve this with a **Frozen Opponent Matchmaking Pool**:

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
