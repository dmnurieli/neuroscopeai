"""`compare`: the app engine's logits (engine/examples/logits.rs) against MLX's on the same exported weights and
the same token ids, at every position the engine wrote (the prompt's last token, then greedy steps through its KV
cache). MLX runs the full prefix each time, so the cache path is checked too.

MLX computes the reference in f32 on the CPU from the exported (bf16-rounded) weights (MLX's GPU f32 kernels are
looser: ~5e-3 on logits). Bars, as the engine's own reference test against transformers: engine f32
(NS_DTYPE=f32) needs cosine >= 0.99999, max |diff| <= 1e-3 and the
same top-1 token; engine bf16 (the app's default, which rounds activations at every layer) needs cosine >= 0.999,
max |diff| <= 1.0 and the same top-1 token.
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .events import Emitter, log
from .export import load_export


def compare_with_engine(export_dir: Path, engine_json: Path, em: Emitter, tol: float | None = None) -> bool:
    info = json.loads(engine_json.read_text())
    v = info["vocab_size"]
    raw = np.fromfile(engine_json.with_name(info["logits_file"]), dtype="<f4")
    rows = raw.reshape(-1, v)
    ids = info["ids"]
    dtype = info["dtype"]
    mx.set_default_device(mx.cpu)  # MLX's GPU f32 kernels drift by ~5e-3 on logits; its CPU path matches transformers to ~1e-5
    model = load_export(export_dir)  # f32 compute from the exported (bf16-rounded) weights: the reference
    logits = np.array(model(mx.array([ids]))[0].astype(mx.float32))  # [len, vocab]
    ok_all = True
    # The engine's own reference test (engine/tests/reference_olmo.rs) holds itself to cosine >= 0.99999 in f32 and
    # >= 0.999 in bf16 against Hugging Face transformers; the same bars apply here, plus the same top-1 token.
    max_tol = tol if tol is not None else (1e-3 if dtype == "f32" else 1.0)
    min_cos = 0.99999 if dtype == "f32" else 0.999
    for k, pos in enumerate(info["positions"]):
        a, b = rows[k], logits[pos]
        diff = float(np.max(np.abs(a - b)))
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        top1 = int(np.argmax(a)) == int(np.argmax(b))
        ok = diff <= max_tol and top1 and cos >= min_cos
        ok_all &= ok
        log(f"position {pos}: max |diff| {diff:.2e}, cosine {cos:.6f}, top-1 {'same' if top1 else 'DIFFERENT'} -> {'ok' if ok else 'FAIL'}")
        em.emit("compare", prompt=f"{info['prompt']} @ position {pos}", max_abs_diff=diff, cosine=cos, top1_match=top1,
                dtype=f"engine {dtype} / mlx f32 (cpu)", ok=ok)
    return ok_all
