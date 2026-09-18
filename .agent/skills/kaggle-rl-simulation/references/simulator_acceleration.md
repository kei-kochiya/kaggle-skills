# Simulator Acceleration & Parity Verification Playbook 🏎️

In competitive reinforcement learning and simulation benchmarks on Kaggle, **environment throughput is the single greatest bottleneck to final Elo**. The difference between a standard Python environment (~100 steps/sec) and an optimized compiled engine (~120,000+ steps/sec) represents a **1,000× training speedup**, enabling billions of self-play steps within modest compute budgets.

This reference guide outlines the standard operating procedure for building, accelerating, and verifying high-throughput simulation environments in **Rust (PyO3/Rayon)** and **JAX**, backed by automated replay parity testing against official Kaggle episode logs.

---

## 1. Environment Architecture & The Acceleration Hierarchy

When analyzing an official Kaggle simulation environment (typically provided as a single-threaded Python class under `kaggle_environments.envs.*`), prioritize optimizations according to this hierarchy:

```
                      SIMULATION ACCELERATION HIERARCHY
                      
 Level 4: Zero-Copy CUDA Streams
 [Pinned Host Memory -> Asynchronous GPU Transfer without Python GIL]
                      ^
                      |
 Level 3: Multithreaded Vectorization (Rayon / C++ OpenMP / JAX pmap)
 [Fused Step-Obs-Reset Loop across thousands of concurrent environments]
                      ^
                      |
 Level 2: Spatial Indexing & Algorithmic Pruning
 [AABB Broad-Phase Filtering, Precomputed Geometry, Integer IDs]
                      ^
                      |
 Level 1: Native Compiled Rewrite (Rust PyO3 / C++ pybind11)
 [Bit-for-bit transition matching, memory preallocation, no GC pauses]
                      ^
                      |
 Baseline: Official Kaggle Python Engine (~50 - 200 env steps/sec)
```

---

## 2. High-Performance Rust Vector Engine (PyO3 + Rayon)

### 2.1 The Fused Step-Obs-Reset Protocol
In standard Python RL loops, stepping environments, computing observations, and resetting finished games involve three separate dispatches and heavy memory allocations. The battle-tested pattern fuses all three into a single parallel loop:

```rust
// src/rules_engine/vec_env.rs
use rayon::prelude::*;
use pyo3::prelude::*;

pub struct VectorEnv {
    n_envs: usize,
    states: Vec<State>,
    cached_action_slots: Vec<ActionSlots>,
    max_entities: usize,
}

impl VectorEnv {
    /// Fuses action decode, physics step, terminal reward computation, 
    /// auto-reset, and observation tensor serialisation into one parallel dispatch.
    pub fn step_fused(
        &mut self,
        actions: &[ActionBatch],
        obs_buffer: &mut [f32],
        rewards: &mut [f32],
        dones: &mut [bool],
    ) {
        let n_envs = self.n_envs;
        let obs_stride = self.obs_stride();

        self.states
            .par_iter_mut()
            .zip(&mut self.cached_action_slots)
            .zip(actions)
            .zip(obs_buffer.par_chunks_exact_mut(obs_stride))
            .zip(rewards.par_chunks_exact_mut(4)) // 4 players max
            .zip(dones.par_iter_mut())
            .for_each(|(((((state, slots), action), obs_slice), rew_slice), done)| {
                // 1. Decode actions against cached slot state
                let decoded_actions = state.decode_actions(slots, action);

                // 2. Advance physics & resolution
                let transition = state.step(&decoded_actions);

                // 3. Handle terminal states & rewards
                *done = transition.is_terminal;
                for p in 0..4 {
                    rew_slice[p] = transition.rewards[p];
                }

                // 4. Auto-reset if terminal
                if *done {
                    state.reset();
                }

                // 5. Write observation directly into contiguous memory
                state.write_observation_tensor(slots, obs_slice);
            });
    }
}
```

### 2.2 Preallocated Pinned Memory & Zero-Copy Transfers
Avoid creating Python objects or allocating fresh numpy arrays on every step. Allocate page-locked (pinned) CPU memory upfront in PyTorch and pass the raw pointers into Rust:

```python
# python/owl/vec_env_wrapper.py
import torch
import numpy as np

class FastVectorEnvWrapper:
    def __init__(self, rust_vec_env, n_envs, obs_shape, device="cuda"):
        self.env = rust_vec_env
        self.n_envs = n_envs
        self.device = device
        
        # Preallocate pinned memory buffers on host CPU
        self.obs_host = torch.empty((n_envs, *obs_shape), dtype=torch.float32, pin_memory=True)
        self.rewards_host = torch.empty((n_envs, 4), dtype=torch.float32, pin_memory=True)
        self.dones_host = torch.empty((n_envs,), dtype=torch.bool, pin_memory=True)
        
        # Preallocate device buffers on GPU
        self.obs_device = torch.empty_like(self.obs_host, device=device)
        self.rewards_device = torch.empty_like(self.rewards_host, device=device)
        self.dones_device = torch.empty_like(self.dones_host, device=device)
        
        # Dedicated non-default CUDA stream for async transfers
        self.transfer_stream = torch.cuda.Stream()

    def step(self, actions_tensor):
        # Pass raw memory pointers into Rust via PyO3
        self.env.step_fused(
            actions_tensor.numpy(),
            self.obs_host.numpy(),
            self.rewards_host.numpy(),
            self.dones_host.numpy()
        )
        
        # Non-blocking async transfer from pinned host to GPU
        with torch.cuda.stream(self.transfer_stream):
            self.obs_device.copy_(self.obs_host, non_blocking=True)
            self.rewards_device.copy_(self.rewards_host, non_blocking=True)
            self.dones_device.copy_(self.dones_host, non_blocking=True)
            
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        return self.obs_device, self.rewards_device, self.dones_device
```

---

## 3. Spatial Partitioning & Collision Broad-Phase Filtering

In games with multiple fleets, projectiles, and celestial obstacles, naive all-pairs collision testing scales as $\mathcal{O}(N^2)$ or $\mathcal{O}(N \times M)$ distance checks involving costly square roots and trigonometric calls.

### 3.1 Axis-Aligned Bounding Box (AABB) Broad Phase
Before calculating Euclidean distance $d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2} < R_1 + R_2$, run a conservative Manhattan bounding box check:

```rust
#[inline(always)]
pub fn test_circle_collision(
    p1: (f32, f32), r1: f32,
    p2: (f32, f32), r2: f32,
) -> bool {
    let r_sum = r1 + r2;
    let dx = (p1.0 - p2.0).abs();
    let dy = (p1.1 - p2.1).abs();

    // 1. Broad-Phase AABB Rejection (Zero sqrt, pure comparisons)
    if dx > r_sum || dy > r_sum {
        return false;
    }

    // 2. Narrow-Phase Squared Distance Check (Avoid sqrt)
    (dx * dx) + (dy * dy) <= (r_sum * r_sum)
}
```

### 3.2 Piecewise Linear Swept Collision Checks
For high-speed entities (e.g. moving comets or ships travelling along ballistic trajectories), point-in-circle tests cause tunneling. Solve the exact continuous segment-to-circle intersection:

```rust
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
    let b = 2.0 * (f.0 * d.0 + f.1 * d.1);
    let c = (f.0 * f.0 + f.1 * f.1) - radius * radius;

    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return false;
    }

    let sqrt_disc = discriminant.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);

    (t1 >= 0.0 && t1 <= 1.0) || (t2 >= 0.0 && t2 <= 1.0)
}
```

---

## 4. Replay Parity Testing Framework

A fast simulator is useless if it diverges from official game physics. A subtle discrepancy in collision tolerance, production timing, or floating-point rounding will cause an RL agent to exploit artifacts that do not exist on the Kaggle leaderboard.

### 4.1 Automated Parity Pipeline
Build an automated test suite that fetches official tournament match JSONs from Kaggle and replays them synchronously:

```
                     REPLAY PARITY REGRESSION HARNESS
                     
 [Official Kaggle Episode JSON (e.g. Episode 75930761)]
                        |
            +-----------+-----------+
            |                       |
            v                       v
 [Extract Action Steps]    [Extract Expected State Transitions]
            |                       |
            v                       |
   [Execute in Rust Engine]         |
            |                       |
            +-----------+-----------+
                        |
                        v
          [Assert Exact Numerical Parity]
          • Planet owners & ship counts
          • Fleet positions & trajectories
          • Combat survival counts
          • Terminal rank & rewards
```

### 4.2 Parity Test Implementation Example
```rust
// tests/rules_parity_test.rs
use std::fs::File;
use std::io::{BufRead, BufReader};
use serde_json::Value;

#[test]
fn test_kaggle_tournament_replay_parity() {
    let fixture_file = "tests/fixtures/replays/episode_75930761.jsonl";
    let file = File::open(fixture_file).expect("Fixture file must exist");
    let reader = BufReader::new(file);

    let mut rust_env = State::new_from_kaggle_seed(42);

    for (step_idx, line) in reader.lines().enumerate() {
        let record: Value = serde_json::from_str(&line.unwrap()).unwrap();
        
        let actions = parse_kaggle_actions(&record["actions"]);
        let expected_next_state = &record["next_observation"];

        // Advance Rust engine with tournament actions
        rust_env.step(&actions);

        // Verify strict numerical equivalence
        for planet in rust_env.planets() {
            let p_id = planet.id;
            let expected_ships = expected_next_state["planets"][p_id]["ships"].as_f64().unwrap();
            let expected_owner = expected_next_state["planets"][p_id]["owner"].as_i64().unwrap();

            assert_eq!(
                planet.owner, expected_owner,
                "Step {step_idx}: Planet {p_id} owner mismatch"
            );
            assert!(
                (planet.ships as f64 - expected_ships).abs() < 1e-4,
                "Step {step_idx}: Planet {p_id} ships mismatch: got {}, expected {}",
                planet.ships, expected_ships
            );
        }
    }
}
```

---

## 5. Acceleration Checklist & Benchmarking SOP

When implementing or optimizing a simulation engine:

- [ ] **Release Mode Compilation**: Always benchmark with `cargo build --release` (`opt-level = 3`, `lto = "fat"`, `codegen-units = 1`).
- [ ] **Single-Core Baseline vs Multi-Core Scaling**: Verify that throughput scales linearly with physical CPU cores ($\ge 0.85 \times \text{cores}$).
- [ ] **Warmup & Thermal Cooldown**: Let CPU temperature settle between timing runs to prevent throttling noise.
- [ ] **Parity Suite Gate**: Never merge simulator optimizations without running the full Kaggle replay test suite.
