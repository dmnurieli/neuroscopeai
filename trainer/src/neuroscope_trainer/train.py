"""The training loop shared by `pretrain` and `sft`: mixed precision (f32 master weights, bf16 compute, f32 loss),
AdamW with OLMo 2's weight-decay rule, gradient clipping, micro-batch accumulation, milestones (validation loss,
text samples, saved weights), resumable state and a clean stop on SIGTERM."""

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from .events import Emitter, StopFlag, log
from .model import Olmo2, cast_view, init_weights
from .presets import EOS_ID, ModelShape
from .shards import TokenWindows

STATE_DIR = "state"
MILESTONE_DIR = "milestones"
FINAL_WEIGHTS = "final.safetensors"
RUN_FILE = "run.json"
CHECKPOINT_EVERY_S = 600.0

# Where samples are drawn, as fractions of the run: dense early (the model changes fastest there), sparse late.
MILESTONE_FRACTIONS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


@dataclass
class RunConfig:
    phase: str  # "pretrain" or "sft"
    shape: dict
    seq_len: int
    tokens_per_step: int
    micro_batch: int  # sequences per micro-batch
    total_steps: int
    peak_lr: float
    warmup_steps: int
    schedule: str  # "cosine" (to 10% of peak) or "linear" (to 0)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0
    prompts: list = field(default_factory=list)
    prompt_kind: str = "plain"  # "plain" or "chat"
    sample_tokens: int = 40
    sample_temperature: float = 0.8
    sample_top_k: int = 40
    eval_windows: int = 32
    parent: str | None = None  # weights to start from (sft)
    preset: str = ""

    @property
    def model_shape(self) -> ModelShape:
        return ModelShape(**self.shape)

    @property
    def seqs_per_step(self) -> int:
        return self.tokens_per_step // self.seq_len


def lr_schedule(cfg: RunConfig):
    warm = max(cfg.warmup_steps, 1)
    rest = max(cfg.total_steps - warm, 1)
    if cfg.schedule == "cosine":
        decay = optim.cosine_decay(cfg.peak_lr, rest, cfg.peak_lr * 0.1)
    else:
        decay = optim.linear_schedule(cfg.peak_lr, 0.0, rest)
    return optim.join_schedules([optim.linear_schedule(cfg.peak_lr / warm, cfg.peak_lr, warm), decay], [warm])


def make_optimizer(cfg: RunConfig):
    """AdamW (0.9, 0.95); weight decay on the matrices of the transformer layers only — not on embeddings or norm
    weights (as OLMo 2)."""
    sched = lr_schedule(cfg)
    decayed = optim.AdamW(learning_rate=sched, betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg.weight_decay)
    plain = optim.AdamW(learning_rate=sched, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    return optim.MultiOptimizer([decayed, plain], [lambda path, p: p.ndim == 2 and "embed_tokens" not in path])


def masked_ce_sum(model: Olmo2, x, y, m):
    logits = model(x).astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, y, reduction="none")
    return (ce * m).sum()


def save_safetensors(path: Path, arrays: dict, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.safetensors")
    mx.save_safetensors(str(tmp), arrays, metadata=metadata or {})
    os.replace(tmp, path)


def load_weights(model: Olmo2, path: Path) -> None:
    w = mx.load(str(path))
    model.update(tree_unflatten([(k, v.astype(mx.float32)) for k, v in w.items()]))


class Trainer:
    def __init__(self, run_dir: Path, cfg: RunConfig, train: TokenWindows, val: TokenWindows | None, tok, em: Emitter, stop: StopFlag):
        self.run_dir, self.cfg, self.train, self.val, self.tok, self.em, self.stop = Path(run_dir), cfg, train, val, tok, em, stop
        mx.random.seed(cfg.seed)
        self.model = Olmo2(cfg.model_shape)
        self.opt = make_optimizer(cfg)
        self.step = 0
        self.elapsed = 0.0  # training seconds accumulated across resumes
        self.milestones: list[dict] = []

    # ---- compiled steps ----
    def _build_steps(self):
        """Compiles the steps against the CURRENT model and optimizer state objects; call again after replacing them
        (a restored optimizer state is a new set of arrays the old compiled graph does not know)."""
        model, opt, cfg = self.model, self.opt, self.cfg

        def loss_fn(mdl, x, y, m, norm):
            orig = cast_view(mdl, mx.bfloat16)
            s = masked_ce_sum(mdl, x, y, m)
            mdl.update(orig)
            return s / norm

        vg = nn.value_and_grad(model, loss_fn)

        def fused(x, y, m, norm):
            loss, grads = vg(model, x, y, m, norm)
            grads, gn = optim.clip_grad_norm(grads, cfg.grad_clip)
            opt.update(model, grads)
            return loss, gn

        def grad_only(x, y, m, norm):
            return vg(model, x, y, m, norm)

        def apply(grads):
            grads, gn = optim.clip_grad_norm(grads, cfg.grad_clip)
            opt.update(model, grads)
            return gn

        state = [model.state, opt.state]
        self._fused = mx.compile(fused, inputs=state, outputs=state)
        self._grad = mx.compile(grad_only, inputs=state, outputs=state)
        self._apply = mx.compile(apply, inputs=state, outputs=state)

        def eval_fn(x, y, m):
            orig = cast_view(model, mx.bfloat16)
            s = masked_ce_sum(model, x, y, m)
            model.update(orig)
            return s

        self._eval = eval_fn

    def train_step(self) -> tuple[float, float]:
        cfg = self.cfg
        n = cfg.seqs_per_step
        x, y, m = self.train.batch(self.step * n, n)
        norm = mx.array(max(float(m.sum()), 1.0))
        mb = cfg.micro_batch
        if mb >= n:
            loss, gn = self._fused(mx.array(x), mx.array(y), mx.array(m), norm)
            mx.eval(loss, gn, self.model.state, self.opt.state)
            return loss.item(), gn.item()
        acc, total = None, 0.0
        for i in range(0, n, mb):
            loss, g = self._grad(mx.array(x[i : i + mb]), mx.array(y[i : i + mb]), mx.array(m[i : i + mb]), norm)
            acc = g if acc is None else tree_map(mx.add, acc, g)
            mx.eval(loss, acc)
            total += loss.item()
        gn = self._apply(acc)
        mx.eval(gn, self.model.state, self.opt.state)
        return total, gn.item()

    # ---- evaluation and samples ----
    def val_loss(self) -> float | None:
        if self.val is None:
            return None
        x, y, m = self.val.fixed_windows(self.cfg.eval_windows)
        tot, cnt = 0.0, 0.0
        mb = max(self.cfg.micro_batch, 1)
        for i in range(0, len(x), mb):
            s = self._eval(mx.array(x[i : i + mb]), mx.array(y[i : i + mb]), mx.array(m[i : i + mb]))
            tot += s.item()
            cnt += float(m[i : i + mb].sum())
        return tot / max(cnt, 1.0)

    def sample(self, prompt_ids: list[int], seed: int) -> list[int]:
        cfg = self.cfg
        key = mx.random.key(seed)
        ids = list(prompt_ids)
        out = []
        for _ in range(cfg.sample_tokens):
            logits = self.model(mx.array([ids[-cfg.seq_len :]]))[0, -1].astype(mx.float32)
            if cfg.sample_temperature <= 0:
                nxt = int(mx.argmax(logits).item())
            else:
                k = min(cfg.sample_top_k, logits.shape[-1])
                kth = mx.sort(logits)[-k]
                logits = mx.where(logits < kth, -mx.inf, logits)
                key, sub = mx.random.split(key)
                nxt = int(mx.random.categorical(logits / cfg.sample_temperature, key=sub).item())
            if nxt == EOS_ID:
                break
            ids.append(nxt)
            out.append(nxt)
        return out

    def prompt_ids(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    # ---- milestones ----
    def milestone_steps(self) -> list[int]:
        T = self.cfg.total_steps
        return sorted({min(T, round(f * T)) for f in MILESTONE_FRACTIONS})

    def tokens_at(self, step: int) -> int:
        return step * self.cfg.tokens_per_step

    def milestone(self) -> None:
        cfg, step, tokens = self.cfg, self.step, self.tokens_at(self.step)
        rec = {"step": step, "tokens": tokens, "samples": []}
        vl = self.val_loss()
        if vl is not None:
            rec["val_loss"] = round(vl, 4)
            self.em.emit("eval", phase=cfg.phase, step=step, tokens=tokens, val_loss=round(vl, 4))
        for i, p in enumerate(cfg.prompts):
            out = self.sample(self.prompt_ids(p["text"]), seed=cfg.seed * 1000 + i)
            text = self.tok.decode(out, skip_special_tokens=False)
            rec["samples"].append({"prompt_id": i, "prompt": p["shown"], "text": text})
            self.em.emit("sample", phase=cfg.phase, step=step, tokens=tokens, prompt_id=i, prompt=p["shown"], text=text,
                         kind=cfg.prompt_kind)
        path = self.run_dir / MILESTONE_DIR / f"step-{step:07d}.safetensors"
        weights = {k: v.astype(mx.bfloat16) for k, v in tree_flatten(self.model.parameters())}
        save_safetensors(path, weights, {"format": "mlx", "step": str(step), "tokens": str(tokens)})
        rec["weights"] = str(path.relative_to(self.run_dir))
        self.milestones = [m for m in self.milestones if m["step"] != step] + [rec]
        (self.run_dir / "milestones.json").write_text(json.dumps(self.milestones, indent=1, ensure_ascii=False))
        self.em.emit("checkpoint", phase=cfg.phase, step=step, tokens=tokens, path=str(path), kind="milestone")

    # ---- state ----
    def save_state(self) -> None:
        d = self.run_dir / STATE_DIR
        save_safetensors(d / "model.safetensors", dict(tree_flatten(self.model.parameters())))
        save_safetensors(d / "optimizer.safetensors", dict(tree_flatten(self.opt.state)))
        tmp = d / "state.json.part"
        tmp.write_text(json.dumps({"step": self.step, "tokens": self.tokens_at(self.step), "elapsed_s": self.elapsed}))
        os.replace(tmp, d / "state.json")
        self.em.emit("checkpoint", phase=self.cfg.phase, step=self.step, tokens=self.tokens_at(self.step), path=str(d), kind="resume")

    def try_resume(self) -> bool:
        d = self.run_dir / STATE_DIR
        if not (d / "state.json").exists():
            return False
        st = json.loads((d / "state.json").read_text())
        load_weights(self.model, d / "model.safetensors")
        self.opt.init(self.model.trainable_parameters())
        self.opt.state = tree_unflatten(list(mx.load(str(d / "optimizer.safetensors")).items()))
        self.step, self.elapsed = int(st["step"]), float(st["elapsed_s"])
        mf = self.run_dir / "milestones.json"
        self.milestones = json.loads(mf.read_text()) if mf.exists() else []
        mx.eval(self.model.parameters(), self.opt.state)
        self._build_steps()
        log(f"resumed at step {self.step} ({self.tokens_at(self.step):,} tokens)")
        return True

    def init_fresh(self) -> None:
        if self.cfg.parent:
            load_weights(self.model, Path(self.cfg.parent))
        else:
            init_weights(self.model, self.cfg.seed)
        self.opt.init(self.model.trainable_parameters())
        mx.eval(self.model.parameters(), self.opt.state)
        self._build_steps()

    # ---- the loop ----
    def run(self) -> str:
        cfg = self.cfg
        if not self.try_resume():
            self.init_fresh()
        T = cfg.total_steps
        marks = [s for s in self.milestone_steps() if s >= self.step and not any(m["step"] == s for m in self.milestones)]
        if marks and marks[0] == self.step:
            self.milestone()
            marks.pop(0)
        last_ckpt = time.time()
        window: list[tuple[float, float, int]] = []  # (time, loss, tokens) since the last progress event
        last_emit = 0.0
        rate = 0.0
        while self.step < T:
            if self.stop:
                self.save_state()
                return "stopped"
            t0 = time.perf_counter()
            loss, gn = self.train_step()
            dt = time.perf_counter() - t0
            if not math.isfinite(loss):
                self.em.emit("error", code="nan_loss", message=f"loss became {loss} at step {self.step}")
                return "failed"
            self.step += 1
            self.elapsed += dt
            window.append((dt, loss, cfg.tokens_per_step))
            inst = cfg.tokens_per_step / dt
            rate = inst if rate == 0.0 or self.step < 4 else 0.9 * rate + 0.1 * inst
            now = time.time()
            if now - last_emit >= 1.0 or self.step == T or self.step == 1:
                last_emit = now
                lr = float(self.opt.optimizers[0].learning_rate.item()) if hasattr(self.opt, "optimizers") else 0.0
                remaining = (T - self.step) * cfg.tokens_per_step
                self.em.emit("progress", phase=cfg.phase, step=self.step, total_steps=T, tokens=self.tokens_at(self.step),
                             total_tokens=self.tokens_at(T), loss=round(float(np.mean([w[1] for w in window])), 4),
                             lr=float(f"{lr:.6g}"), tok_per_s=round(rate, 1), elapsed_s=round(self.elapsed, 1),
                             eta_s=round(remaining / rate if rate > 0 else 0.0, 1), grad_norm=round(gn, 4))
                window.clear()
            if marks and self.step == marks[0]:
                marks.pop(0)
                self.milestone()
                self.save_state()
                last_ckpt = time.time()
            elif time.time() - last_ckpt > CHECKPOINT_EVERY_S:
                self.save_state()
                last_ckpt = time.time()
        save_safetensors(self.run_dir / FINAL_WEIGHTS, dict(tree_flatten(self.model.parameters())))
        return "completed"


def write_run(run_dir: Path, cfg: RunConfig, extra: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_FILE).write_text(json.dumps({"config": asdict(cfg), **extra}, indent=1, ensure_ascii=False))


def read_run(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / RUN_FILE).read_text())
