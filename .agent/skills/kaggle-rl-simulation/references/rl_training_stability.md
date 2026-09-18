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
