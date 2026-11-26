# Simulator Acceleration & Parity Verification Playbook

Simulation throughput is frequently the primary bottleneck in competitive RL and search. Profile simulation alongside encoding, inference, transfers, and gradient updates before investing in a custom engine.

**Scope:** Start from the [simulation workflow](../SKILL.md) to verify your baseline and measured bottleneck first. Require exact observation-visibility, discrete-rule, RNG, reward, and terminal parity (with explicit floating-point tolerances) before trusting any accelerated simulator.

---

## 1. The Acceleration Hierarchy

```text
  Level 4: Zero-Copy Async Transfers (Pinned Host Memory -> Dedicated CUDA Stream)
                     ^
  Level 3: Multithreaded / Device Vectorization (Rayon / OpenMP / jax.vmap)
           [Fused Step-Obs-Reset across N concurrent environments]
                     ^
  Level 2: Algorithmic & Spatial Pruning (AABB Broad-Phase, Integer IDs, Precomputed LUTs)
                     ^
  Level 1: Native Compiled Rewrite (Rust PyO3 / C++ pybind11 / Pure JAX)
           [Preallocated flat buffers, zero GC pauses, exact transition parity]
                     ^
  Baseline: Official Python Environment (~20 - 200 env steps/sec)
```

---

## 2. High-Performance Rust Vector Engine (PyO3 + Rayon)

### 2.1 Fused Step-Obs-Reset Loop
Stepping environments, encoding observations, and resetting terminal episodes in separate Python passes adds heavy FFI and thread-dispatch overhead. Fuse all steps into one parallel Rayon pass parameterized by `n_players`:

```rust
// src/vec_env.rs
use rayon::prelude::*;

pub struct VectorEnv {
    n_envs: usize,
    n_players: usize,
    states: Vec<GameState>,
    action_caches: Vec<ActionCache>,
}

impl VectorEnv {
    /// Fuses action decoding, state transition, reward write, auto-reset,
    /// and contiguous observation serialization in one parallel dispatch.
    pub fn step_fused(
        &mut self,
        actions: &[ActionBatch],
        obs_buffer: &mut [f32],
        rewards: &mut [f32],
        dones: &mut [bool],
    ) {
        let obs_stride = self.obs_stride();
        let n_players = self.n_players;

        self.states
            .par_iter_mut()
            .zip(&mut self.action_caches)
            .zip(actions)
            .zip(obs_buffer.par_chunks_exact_mut(obs_stride))
            .zip(rewards.par_chunks_exact_mut(n_players))
            .zip(dones.par_iter_mut())
            .for_each(|(((((state, cache), action), obs_slice), rew_slice), done)| {
                let decoded = state.decode_actions(cache, action);
                let transition = state.step(&decoded);

                *done = transition.is_terminal;
                rew_slice.copy_from_slice(&transition.rewards[..n_players]);

                if *done {
                    state.reset_next_seed();
                }
                state.write_observation_tensor(cache, obs_slice);
            });
    }
}
```

### 2.2 Preallocated Pinned Memory & Async CUDA Transfers
Allocate page-locked host buffers once and copy asynchronously on a dedicated CUDA stream to avoid per-step Python allocations:

```python
import torch

class FastVectorEnvWrapper:
    def __init__(self, native_vec_env, n_envs, n_players, obs_shape, device="cuda"):
        self.env = native_vec_env
        self.obs_host = torch.empty((n_envs, *obs_shape), dtype=torch.float32, pin_memory=True)
        self.rewards_host = torch.empty((n_envs, n_players), dtype=torch.float32, pin_memory=True)
        self.dones_host = torch.empty((n_envs,), dtype=torch.bool, pin_memory=True)

        self.obs_device = torch.empty_like(self.obs_host, device=device)
        self.rewards_device = torch.empty_like(self.rewards_host, device=device)
        self.dones_device = torch.empty_like(self.dones_host, device=device)
        self.transfer_stream = torch.cuda.Stream()

    def step(self, actions_cpu: torch.Tensor):
        self.env.step_fused(
            actions_cpu.numpy(),
            self.obs_host.numpy(),
            self.rewards_host.numpy(),
            self.dones_host.numpy(),
        )
        with torch.cuda.stream(self.transfer_stream):
            self.obs_device.copy_(self.obs_host, non_blocking=True)
            self.rewards_device.copy_(self.rewards_host, non_blocking=True)
            self.dones_device.copy_(self.dones_host, non_blocking=True)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        return self.obs_device, self.rewards_device, self.dones_device
```

---

## 3. Spatial Broad-Phase & Swept Collision Filtering

In continuous 2D/3D simulations with many moving entities, reject non-interacting pairs cheaply before exact distance or ray-intersection math:

```rust
#[inline(always)]
pub fn circle_overlap_broad_narrow(p1: (f32, f32), r1: f32, p2: (f32, f32), r2: f32) -> bool {
    let r_sum = r1 + r2;
    let dx = (p1.0 - p2.0).abs();
    let dy = (p1.1 - p2.1).abs();
    if dx > r_sum || dy > r_sum {
        return false; // Broad-phase AABB rejection
    }
    (dx * dx) + (dy * dy) <= (r_sum * r_sum) // Narrow-phase squared distance (no sqrt)
}

#[inline(always)]
pub fn segment_intersects_circle(
    seg_start: (f32, f32),
    seg_end: (f32, f32),
    center: (f32, f32),
    radius: f32,
) -> bool {
    let d = (seg_end.0 - seg_start.0, seg_end.1 - seg_start.1);
    let f = (seg_start.0 - center.0, seg_start.1 - center.1);
    let a = d.0 * d.0 + d.1 * d.1;
    if a == 0.0 {
        return (f.0 * f.0 + f.1 * f.1) <= radius * radius;
    }
    let b = 2.0 * (f.0 * d.0 + f.1 * d.1);
    let c = (f.0 * f.0 + f.1 * f.1) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return false;
    }
    let s = disc.sqrt();
    let t1 = (-b - s) / (2.0 * a);
    let t2 = (-b + s) / (2.0 * a);
    (0.0..=1.0).contains(&t1) || (0.0..=1.0).contains(&t2)
}
```

---

## 4. Automated Replay Parity Testing Framework

A fast simulator that subtly diverges from official rules (collision radius, order resolution, RNG sequence, or container tie-breaking) causes policies to exploit phantom physics that fail on the leaderboard.

### Critical Parity Checklist
1. **RNG Compatibility**: Match the official engine's exact PRNG algorithm and seed sequence (e.g., CPython's Mersenne Twister MT19937 if the host uses `random.Random`).
2. **Deterministic Tie-Breaking & Ordering**: Match Python `dict` insertion order and tie-breaking rules (use order-preserving containers like `IndexMap` in Rust rather than randomized `HashMap` whenever overflow or iteration order affects state).
3. **Observation-Visibility Parity**: Verify that private/fog-of-war fields are masked identically to the official API for every player seat.

```rust
// tests/replay_parity_test.rs
use std::fs::File;
use std::io::{BufRead, BufReader};
use serde_json::Value;

#[test]
fn test_official_replay_parity() {
    let file = File::open("tests/fixtures/official_replay.jsonl").expect("Missing replay fixture");
    let mut env = GameState::from_initial_snapshot("tests/fixtures/initial_state.json");

    for (step_idx, line) in BufReader::new(file).lines().enumerate() {
        let record: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let actions = parse_official_actions(&record["actions"]);
        let expected = &record["next_observation"];

        env.step(&actions);

        // 1. Exact discrete state equality (owners, counts, terminal flags)
        assert_eq!(env.discrete_state_digest(), parse_discrete_digest(expected), "Step {step_idx} discrete mismatch");

        // 2. Bounded floating-point equality (positions, continuous resources)
        for (entity_id, (got_val, exp_val)) in env.compare_continuous_fields(expected) {
            assert!(
                (got_val - exp_val).abs() < 1e-4,
                "Step {step_idx} entity {entity_id}: got {got_val}, expected {exp_val}"
            );
        }
    }
}
```

---

## 5. End-to-End Pure-JAX JIT Simulation Pipelines

When the game dynamics can be expressed with fixed-size dense arrays, writing the simulator natively in JAX allows fusing environment steps, observation construction, action masking, rollout collection (`jax.lax.scan`), and policy updates inside one `jax.jit` GPU/TPU kernel with **zero CPU–GPU transfer overhead**:

```python
import jax
import jax.numpy as jnp

@jax.jit
def env_step(state, action, rng_key):
    next_positions = advance_kinematics(state.positions, state.velocities, action)
    next_resources, next_owners = resolve_interactions(state, next_positions, action, rng_key)
    is_terminal = check_terminal(next_owners, state.step + 1)

    next_state = state.replace(
        positions=next_positions,
        resources=next_resources,
        owners=next_owners,
        step=state.step + 1,
    )
    return next_state, build_observation(next_state), compute_rewards(next_state), is_terminal

# Vectorize across thousands of parallel environments on a single accelerator
vec_step = jax.vmap(env_step, in_axes=(0, 0, 0))
```

---

## 6. Engine Architecture Trade-Offs

| Dimension | Rust (PyO3 + Rayon) | Pure JAX (`jit` + `vmap`) | C++ (`pybind11`) |
| :--- | :--- | :--- | :--- |
| **Best Fit** | Complex branching rules, dynamic entity counts, multi-node CPU+GPU clusters | Fixed-shape tensor physics, single-GPU/TPU high-throughput budgets | Tight integration with native C++ search/planners (MCTS, LNS, DP) |
| **Bottleneck Eliminated** | Python interpreter + GIL overhead; zero-copy pinned transfers | Eliminates PCIe host–device transfer completely | Python loop overhead + fast test-time tree search |
| **Engineering Risk** | Must enforce exact parity with official Python rules & PRNG | Dynamic control flow/variable entity counts require careful padding/masking | Manual memory safety & debugging complexity |
| **Submission Portability** | Compiles cleanly to `.so` shared library | Export policy weights to PyTorch/ONNX/NumPy for CPU serving | Compiles cleanly to `.so` shared library |

---

## 7. Benchmarking SOP

- Compile Rust in release mode (`opt-level = 3`, `lto = "fat"`, `codegen-units = 1`) and C++ with `-O3 -march=native`.
- Discard the first JAX call (JIT compilation time) when measuring steady-state throughput.
- Confirm multi-core scaling ($\ge 0.85\times$ linear across physical CPU cores) before allocating cluster runs.
- Gate every simulator optimization behind the automated official-replay parity suite.
