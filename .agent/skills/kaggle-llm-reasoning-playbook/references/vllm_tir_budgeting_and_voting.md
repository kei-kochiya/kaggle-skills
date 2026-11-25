# vLLM Serving, Persistent TIR Pools, Time Budgeting & Consensus Voting

Reusable production patterns for single-GPU open-LLM reasoning competitions.

---

## 1. Startup Page-Cache Preload & vLLM Serving Baseline

```python
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def preload_weights_to_page_cache(model_dir: str, workers: int = 8) -> None:
    """Warm Linux OS page cache sequentially before engine init to cut cold-start stalls."""
    files = sorted(Path(model_dir).rglob("*.safetensors"))
    def _touch(path: Path):
        with open(path, "rb") as f:
            while f.read(16 * 1024 * 1024):
                pass
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_touch, files))
```

Key single-GPU serving flags (adjust `max-model-len` and `gpu-memory-utilization` to target VRAM):
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.96 \
  --enable-prefix-caching \
  --async-scheduling \
  --max-num-seqs 256
```

---

## 2. Dynamic Per-Problem Time Scheduler & Uncatchable-Leader Stop

```python
import time
from collections import Counter

class SequentialBudgetScheduler:
    def __init__(self, total_seconds: float, num_problems: int,
                 base_timeout: float, high_timeout: float, safety_reserve: float = 300.0):
        self.deadline = time.monotonic() + total_seconds - safety_reserve
        self.num_problems = num_problems
        self.base_timeout = base_timeout
        self.high_timeout = high_timeout

    def allocate(self, problems_remaining: int) -> float:
        rem_time = max(0.0, self.deadline - time.monotonic())
        fair_share = rem_time / max(1, problems_remaining)
        # Allow hard problems to borrow slack while protecting trailing items
        budget = min(self.high_timeout, max(self.base_timeout, fair_share * 1.5))
        return min(budget, max(15.0, rem_time - 15.0 * max(0, problems_remaining - 1)))

def can_stop_early(answers: list[int], total_attempts: int, early_stop_votes: int) -> bool:
    """Stop as soon as target consensus is reached OR leader is mathematically uncatchable."""
    if not answers:
        return False
    counts = Counter(answers).most_common(2)
    leader_votes = counts[0][1]
    second_votes = counts[1][1] if len(counts) > 1 else 0
    remaining = total_attempts - len(answers)
    return (leader_votes >= early_stop_votes) or (second_votes + remaining <= leader_votes)
```

---

## 3. Tail-Windowed Median Entropy + Vote-Dominant Consensus

Avoid letting one ultra-low-entropy wrong trajectory override a clear majority, and ignore noisy early-exploration tokens:

```python
import math
import statistics
from collections import defaultdict

def token_entropy_from_top_logprobs(top_logprobs: list[dict[str, float]], tail_k: int = 256) -> float:
    """Compute mean Shannon entropy (bits) over the final `tail_k` commitment tokens."""
    window = top_logprobs[-tail_k:] if len(top_logprobs) > tail_k else top_logprobs
    if not window:
        return 1.0
    entropies = []
    for step_lp in window:
        probs = [math.exp(lp) for lp in step_lp.values()]
        z = sum(probs)
        if z <= 0:
            continue
        h = -sum((p / z) * math.log2(max(p / z, 1e-12)) for p in probs)
        entropies.append(h)
    return float(sum(entropies) / max(1, len(entropies)))

def select_vote_dominant_answer(attempts: list[tuple[int, float]]) -> int:
    """attempts: list of (parsed_answer, tail_entropy)."""
    by_ans: dict[int, list[float]] = defaultdict(list)
    for ans, ent in attempts:
        by_ans[ans].append(ent)

    total_votes = len(attempts)
    max_votes = max(len(v) for v in by_ans.values())
    best_ans, best_score = None, -1e18
    for ans, ents in by_ans.items():
        votes = len(ents)
        vote_share = votes / total_votes
        med_ent = max(statistics.median(ents), 1e-9)
        conf = 1.0 / med_ent
        consensus_bonus = 1.0 if (votes == max_votes and conf > 1.0) else 0.0
        score = 2.0 * vote_share + 0.3 * conf + 0.5 * consensus_bonus
        if score > best_score:
            best_score, best_ans = score, ans
    return int(best_ans)
```

---

## 4. Adversarial Ordering & Multiple-Testing Protocol

Before promoting any timeout, sampling, or prompt tweak:
1. Build a stratified offline suite ($N \ge 100$) split into `Basic`, `Harder`, and `Hardest` buckets.
2. Evaluate dynamic schedulers across **4 orderings** (`hard_front`, `hard_middle`, `hard_back`, `hard_random`) × `>= 3` seeds to expose late-run starvation.
3. When screening $M$ candidate modifications on the same benchmark, compute paired significance tests and adjust p-values with **Benjamini-Hochberg FDR**—discard changes that do not clear multiple-testing control.
