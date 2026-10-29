"""`bench`: measure this Mac's training speed per preset (~20 s), then derive the plan for a chosen time.

The learner picks a time, not a size. The rule that turns measured speed into a plan is printed in full:
the pretraining share of the time times the measured tokens/second gives each preset's token budget, and we
pick the largest model that still sees at least 20 training tokens per non-embedding parameter (the
compute-optimal ratio of Hoffmann et al. 2022); below that, a bigger model would be undertrained for the time.
"""

import json
import re
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .events import Emitter, StopFlag, log
from .presets import PRESETS, Preset
from .train import RunConfig, Trainer

TIME_LABELS = {"30m": 30 * 60, "2h": 2 * 3600, "overnight": 8 * 3600}
PRETRAIN_SHARE = 0.85  # of the chosen time; instruction tuning gets SFT_SHARE, the rest is milestones and slack
SFT_SHARE = 0.07
TOKENS_PER_PARAM = 20
MAX_PRETRAIN_TOKENS = 2_000_000_000  # wiki-0000 holds about 2.4B tokens; keep a margin and a single pass
SFT_TOKENS_ESTIMATE = 3_800_000  # what `prepare --sft` produces (about 12,500 conversations)
SFT_EPOCHS = 2
BYTES_PER_TOKEN_MEMORY = 1.6e6  # f32 logits + their gradient + softmax over 100,352 entries, per token, with margin
MEMORY_SHARE = 0.45  # of the GPU's recommended working set


def parse_time(s: str) -> tuple[float, str]:
    s = s.strip().lower()
    if s in TIME_LABELS:
        return float(TIME_LABELS[s]), s
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(s|m|min|h)?", s)
    if not m:
        raise ValueError(f"time {s!r}: use 30m, 2h, overnight, or a number with s/m/h")
    v = float(m.group(1)) * {"s": 1, None: 1, "m": 60, "min": 60, "h": 3600}[m.group(2)]
    return v, s


def device_memory() -> tuple[str, float]:
    info = mx.device_info()
    return info.get("device_name", "unknown"), float(info.get("max_recommended_working_set_size", info.get("memory_size", 8e9)))


def micro_batch_for(preset: Preset, working_set_bytes: float) -> int:
    """Sequences per micro-batch that fit in memory: the largest divisor of the step's sequence count that fits."""
    seqs = preset.tokens_per_step // preset.seq_len
    params_bytes = preset.shape.total_params() * 4 * 4  # f32 weights, gradients, two Adam moments
    budget = working_set_bytes * MEMORY_SHARE - params_bytes
    fit = max(int(budget // (BYTES_PER_TOKEN_MEMORY * preset.seq_len)), 1)
    return max(d for d in range(1, seqs + 1) if seqs % d == 0 and d <= fit)


class RandomTokens:
    def __init__(self, vocab: int, seq: int, seed: int = 0):
        self.vocab, self.seq, self.rng = vocab, seq, np.random.default_rng(seed)

    def batch(self, start, count):
        x = self.rng.integers(0, 100_257, size=(count, self.seq + 1)).astype(np.int32)
        return x[:, :-1], x[:, 1:], np.ones((count, self.seq), np.float32)


def measure(preset: Preset, micro: int, seconds: float, em: Emitter) -> dict:
    cfg = RunConfig(phase="bench", shape=preset.shape.to_dict(), seq_len=preset.seq_len, tokens_per_step=preset.tokens_per_step,
                    micro_batch=micro, total_steps=1000, peak_lr=preset.peak_lr, warmup_steps=10, schedule="cosine")
    tr = Trainer(Path("."), cfg, RandomTokens(preset.shape.vocab_size, preset.seq_len), None, None, em, StopFlag())
    tr.init_fresh()
    mx.reset_peak_memory()
    for _ in range(2):  # compile + warm up
        tr.train_step()
    times = []
    t_end = time.perf_counter() + seconds
    while len(times) < 3 or (time.perf_counter() < t_end and len(times) < 200):
        t0 = time.perf_counter()
        tr.train_step()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    out = {
        "preset": preset.name,
        "label": preset.label,
        "non_embedding_params": preset.shape.non_embedding_params(),
        "total_params": preset.shape.total_params(),
        "tokens_per_step": preset.tokens_per_step,
        "micro_batch": micro,
        "seq_len": preset.seq_len,
        "step_s": round(med, 4),
        "tok_per_s": round(preset.tokens_per_step / med, 1),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
        "steps_measured": len(times),
    }
    del tr
    mx.clear_cache()
    return out


def run_bench(workdir: Path, em: Emitter, seconds_per_preset: float = 4.0) -> dict:
    name, ws = device_memory()
    results = []
    for p in PRESETS.values():
        micro = micro_batch_for(p, ws)
        em.emit("status", phase="bench", message=f"measuring {p.label}", detail={"preset": p.name, "micro_batch": micro})
        r = measure(p, micro, seconds_per_preset, em)
        log(f"{p.name}: {r['tok_per_s']:,.0f} tok/s (step {r['step_s'] * 1e3:.0f} ms, micro-batch {micro}, peak {r['peak_memory_gb']} GB)")
        results.append(r)
    out = {"device": name, "memory_gb": round(ws / 1e9, 1), "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "presets": results}
    Path(workdir).mkdir(parents=True, exist_ok=True)
    (Path(workdir) / "bench.json").write_text(json.dumps(out, indent=1))
    em.emit("bench", device=name, memory_gb=out["memory_gb"], presets=results)
    return out


def _fmt_tokens(n: float) -> str:
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"


def derive_plan(bench: dict, seconds: float, label: str = "") -> dict:
    """Pure function: measured speeds + chosen time -> preset, token budget, steps, and the derivation as text."""
    pre_s, sft_s = seconds * PRETRAIN_SHARE, seconds * SFT_SHARE
    lines = [
        f"Time picked: {label or seconds} = {seconds:,.0f} s.",
        f"Pretraining gets {PRETRAIN_SHARE:.0%} of it ({pre_s:,.0f} s), instruction tuning {SFT_SHARE:.0%} ({sft_s:,.0f} s), "
        f"samples and checkpoints the rest.",
        f"Rule: the largest model that still sees at least {TOKENS_PER_PARAM} training tokens per non-embedding parameter "
        f"(the compute-optimal ratio), at most {_fmt_tokens(MAX_PRETRAIN_TOKENS)} tokens (one pass over the prepared text).",
    ]
    cands = []
    for r in sorted(bench["presets"], key=lambda r: r["non_embedding_params"]):
        tokens = min(r["tok_per_s"] * pre_s, MAX_PRETRAIN_TOKENS)
        ratio = tokens / r["non_embedding_params"]
        ok = ratio >= TOKENS_PER_PARAM
        cands.append({"preset": r["preset"], "tok_per_s": r["tok_per_s"], "tokens": int(tokens), "tokens_per_param": round(ratio, 1), "fits": ok})
        lines.append(
            f"  {r['preset']}: {r['tok_per_s']:,.0f} tok/s x {pre_s:,.0f} s = {_fmt_tokens(tokens)} tokens / "
            f"{r['non_embedding_params'] / 1e6:.1f}M parameters = {ratio:,.1f} tokens per parameter -> {'fits' if ok else 'too big for this time'}"
        )
    fitting = [c for c in cands if c["fits"]]
    choice = fitting[-1] if fitting else cands[0]
    r = next(r for r in bench["presets"] if r["preset"] == choice["preset"])
    tps = r["tokens_per_step"]
    steps = max(int(choice["tokens"] // tps), 1)
    sft_steps = max(int(min(SFT_EPOCHS * SFT_TOKENS_ESTIMATE, r["tok_per_s"] * sft_s) // tps), 1)
    lines.append(f"Chosen: {choice['preset']}, {steps:,} steps of {tps:,} tokens = {_fmt_tokens(steps * tps)} tokens"
                 + ("" if fitting else " (even the smallest model is undertrained in this time; it is still worth watching)") + ".")
    lines.append(f"Instruction tuning: {sft_steps:,} steps (up to {SFT_EPOCHS} passes over the conversations).")
    return {
        "time_budget_s": float(seconds),
        "time_label": label or f"{seconds:.0f}s",
        "preset": choice["preset"],
        "tokens": int(steps * tps),
        "pretrain_steps": steps,
        "tokens_per_step": tps,
        "seq_len": r["seq_len"],
        "sft_steps": sft_steps,
        "derivation": lines,
        "candidates": cands,
    }


def load_bench(workdir: Path) -> dict | None:
    p = Path(workdir) / "bench.json"
    return json.loads(p.read_text()) if p.exists() else None
