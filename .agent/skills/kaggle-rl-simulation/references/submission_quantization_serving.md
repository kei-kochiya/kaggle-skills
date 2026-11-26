# Submission Quantization, CPU Serving & Runtime Guardrails

Competitive simulation runtimes typically enforce hard constraints on **compressed archive size**, **container RAM**, **per-turn CPU latency**, and **shared match overage clocks**, often on a single throttled x86 CPU core.

**Scope:** Always read the active competition's official runtime limits and benchmark the exact packaged artifact first. Compact models may not need quantization or fallback tiers; apply each optimization only when profiling proves it necessary, and re-verify win rate afterward.

---

## 1. Multi-Tier Serving Architecture

```text
  [Offline Packing]
   • Primary Model  ──> Grouped NF4-LSQ (4-bit weights + fp16 group scales)
   • Fallback Model ──> Compact fp16/int8 policy or fast deterministic heuristic
                                |
                                v (Container Startup)
  [Streaming Tensor-by-Tensor Dequantization]
   • Avoids peak-RAM OOM from materializing a full fp32 state_dict copy
                                |
                                v
  [Runtime Dynamic int8 Linear Inference (CPU)]
   • Quantizes nn.Linear layers to qint8; keeps output heads in fp32
                                |
                                v (Every Turn)
  [Runtime Safety Layer]
   • Round-boundary state reset (for multi-round / Best-of-N matches)
   • Observation compaction with survival invariants
   • Overage circuit-breaker (swap to fast fallback when clock < threshold)
   • Rejection / re-presentation detector + zero-forfeit try/except guard
```

---

## 2. Grouped NormalFloat Compression (NF4-LSQ) & Streaming Load

Uniform 4-bit integer quantization degrades deep attention weights sharply. Mapping weights to 16 standard-normal quantiles (**NF4**) per block of $G = 128$ weights with **Least-Squares (LSQ) scale refinement** achieves ~7–8× size compression while preserving policy quality:

```python
import numpy as np
import torch

NF4_CODEBOOK = torch.tensor([
    -1.0, -0.6961928, -0.52507305, -0.39491749,
    -0.28444138, -0.18477343, -0.09105004, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.33791524,
    0.44070983, 0.562617, 0.72295684, 1.0,
], dtype=torch.float32)

def quantize_nf4_lsq(weight_2d: torch.Tensor, group_size: int = 128):
    orig_shape = weight_2d.shape
    flat = weight_2d.detach().float().flatten()
    pad_len = (group_size - (flat.numel() % group_size)) % group_size
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len, dtype=flat.dtype, device=flat.device)])

    groups = flat.view(-1, group_size)
    scales = torch.clamp(groups.abs().amax(dim=1, keepdim=True), min=1e-8)

    cb = NF4_CODEBOOK.to(groups.device)
    codes = torch.argmin((groups / scales).unsqueeze(-1).sub(cb).abs(), dim=-1)
    q_vals = cb[codes]

    # Least-squares optimal per-group scale
    dot_wq = (groups * q_vals).sum(dim=1, keepdim=True)
    dot_qq = torch.clamp((q_vals * q_vals).sum(dim=1, keepdim=True), min=1e-8)
    scales_lsq = torch.clamp(dot_wq / dot_qq, min=1e-8).to(torch.float16)

    codes_np = codes.cpu().numpy().astype(np.uint8)
    packed = (codes_np[:, 0::2] << 4) | (codes_np[:, 1::2] & 0x0F)
    return {
        "packed_codes": packed.tobytes(),
        "scales_fp16": scales_lsq.cpu().numpy().tobytes(),
        "orig_shape": orig_shape,
        "pad_len": pad_len,
        "group_size": group_size,
    }

def decode_nf4_lsq(packed_codes: bytes, scales_fp16: bytes, orig_shape, pad_len: int, group_size: int = 128):
    packed = np.frombuffer(packed_codes, dtype=np.uint8).reshape(-1, group_size // 2)
    codes = np.empty((packed.shape[0], group_size), dtype=np.int64)
    codes[:, 0::2] = (packed >> 4) & 0x0F
    codes[:, 1::2] = packed & 0x0F
    scales = torch.from_numpy(np.frombuffer(scales_fp16, dtype=np.float16).astype(np.float32)).unsqueeze(1)
    vals = (NF4_CODEBOOK[torch.from_numpy(codes)] * scales).flatten()
    if pad_len > 0:
        vals = vals[:-pad_len]
    return vals.view(orig_shape)

def load_quantized_streaming(model: torch.nn.Module, qpack_path: str):
    """Dequantize tensor-by-tensor directly into model buffers to cap startup RAM."""
    qdata = torch.load(qpack_path, map_location="cpu")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in qdata:
                d = qdata[name]
                param.copy_(decode_nf4_lsq(d["packed_codes"], d["scales_fp16"], d["orig_shape"], d["pad_len"], d["group_size"]))
```

---

## 3. CPU Inference Speedup & Observation Compaction

### 3.1 Dynamic `int8` Linear Quantization (Keep Heads FP32)
Apply PyTorch dynamic `int8` quantization to trunk `nn.Linear` layers after loading (often a `3×–6×` CPU latency speedup), while keeping final policy/value output heads in `fp32` to preserve logit precision:

```python
import torch.ao.quantization as quantization

def optimize_trunk_for_cpu(model: torch.nn.Module):
    model.eval()
    return quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

### 3.2 Entity Compaction with Survival Invariants
Quadratic self-attention $\mathcal{O}(L^2)$ can spike CPU latency when minor transient entities proliferate. If you prune low-importance entities below a threshold, **always enforce a per-player survival invariant** so no active player is erroneously treated as eliminated:

```python
def compact_entities(entities, primary_owners, min_weight, weight_key="strength", owner_key="owner"):
    kept = [e for e in entities if e[weight_key] >= min_weight]
    active_owners = {e[owner_key] for e in entities}
    kept_owners = {e[owner_key] for e in kept} | set(primary_owners)

    # Safety invariant: preserve the strongest entity for any owner with no other presence
    for owner in active_owners - kept_owners:
        owner_entities = [e for e in entities if e[owner_key] == owner]
        if owner_entities:
            kept.append(max(owner_entities, key=lambda e: e[weight_key]))
    return kept
```

---

## 4. Latency Circuit-Breaker & Zero-Forfeit Runtime Guards

CPU throttling on shared evaluation hosts can exhaust overage banks mid-match, and any unhandled exception (or rejected action loop) causes immediate forfeit. Protect the submission entrypoint with four guards:

1. **Latency Circuit-Breaker**: Track `remainingOverageTime`; lazily load a fast compact fallback model (or rule heuristic) after Turn 1 and switch over permanently once remaining overage falls below a safety margin.
2. **Round-Boundary Reset**: In multi-round / Best-of-N episodes, detect `round` transitions and immediately reset per-round trackers, hidden-state estimates, and rejection caches.
3. **Rejection / Re-Presentation Guard**: If the environment re-presents the identical decision signature with unchanged event logs, the prior command was rejected—cycle to the next highest-ranked legal action instead of looping until timeout.
4. **Explicit UTF-8 & Safe Fallback Wrapper**: Always pass `encoding="utf-8"` when reading local JSON/config assets (prevents silent OS-default encoding errors inside `try/except`), log/alert exceptions loudly in local tests, and fall back to a verified legal action only in production runtime.

```python
class FallbackCascadeAgent:
    def __init__(self, load_primary, load_fallback, legal_fallback_fn, panic_overage_sec: float = 2.0):
        self.primary = load_primary()
        self.load_fallback = load_fallback
        self.legal_fallback_fn = legal_fallback_fn
        self.panic_overage_sec = panic_overage_sec
        self.fallback = None
        self.use_fallback = False
        self.last_round = None

    def __call__(self, obs, config):
        try:
            # 1. Multi-round state reset
            rd = obs.get("round")
            if rd is not None and rd != self.last_round:
                if self.last_round is not None:
                    self.primary.reset_round_state()
                self.last_round = rd

            # 2. Lazy fallback init after Turn 0 to keep startup instant
            if self.fallback is None and obs.get("step", 0) >= 1:
                self.fallback = self.load_fallback()

            # 3. Overage circuit-breaker
            if obs.get("remainingOverageTime", float("inf")) < self.panic_overage_sec and self.fallback is not None:
                self.use_fallback = True

            model = self.fallback if self.use_fallback else self.primary
            return model.act(obs, config)
        except Exception:
            # Never forfeit in production; disable catch-all in local unit tests!
            return self.legal_fallback_fn(obs)
```

---

## 5. Test-Time Lookahead Search & Hierarchical Decoupling

### 5.1 Bounded Rollout Lookahead
When a fast, parity-verified forward simulator is available at serving time, evaluate the policy's top-$K$ candidate actions via 1–2 step lookahead scored by the critic (catching short-horizon tactical blunders):

```python
def test_time_lookahead(obs, policy_net, value_net, simulator, top_k: int = 4):
    candidates = policy_net.sample_top_k_actions(obs, k=top_k)
    best_action, best_score = candidates[0], -float("inf")
    for action in candidates:
        next_obs, reward, done = simulator.step_copy(obs, action)
        score = float(reward) if done else float(value_net(next_obs))
        if score > best_score:
            best_score, best_action = score, action
    return best_action
```

### 5.2 Hierarchical Macro-Policy + Native Low-Level Solver
When action execution combines **strategic target/allocation choices** with **exact continuous physics or micro-routing constraints**:
- Let a compact neural policy (`2M–10M` params) output high-level **discrete strategic intents** (which targets to act on, relative resource budgets, periodic macro quotas).
- Delegate exact continuous trajectory intercepts, hazard avoidance, pathfinding, and slot-priority ordering to a deterministic native C++/Rust micro-solver (`<5ms` CPU).
- Result: orders-of-magnitude lower sample complexity, compact submission size, and deterministic collision/legality guarantees.
