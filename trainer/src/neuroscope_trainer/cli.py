"""Command line: `neuroscope-train <bench|prepare|pretrain|sft|export|compare> ...` (or `python -m neuroscope_trainer`).

Every subcommand writes protocol events (docs/trainer-protocol.md) to stdout and logs to stderr, starts with a
`hello` event and ends with a `done` event. Exit codes: 0 completed, 3 stopped by a signal, 1 failed, 2 bad
arguments.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from . import PROTOCOL_VERSION, __version__, bench, chat, prepare, presets
from .events import Emitter, StopFlag, log
from .shards import TokenWindows, read_manifest
from .train import FINAL_WEIGHTS, RunConfig, Trainer, read_run, write_run

DEFAULT_WORKDIR = Path.home() / "Library" / "Application Support" / "NeuroScope" / "training"

PRETRAIN_PROMPTS = [
    "The capital of France is",
    "Albert Einstein was a",
    "In 1969, the first",
    "The river flows",
    "An airplane is",
]
SFT_PROMPTS = [
    "What is the capital of France?",
    "Who was Albert Einstein?",
    "Write one sentence about the ocean.",
    "How do airplanes fly?",
    "Give me three words that rhyme with cat.",
]


class Fail(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def workdir_of(a) -> Path:
    return Path(a.workdir or os.environ.get("NEUROSCOPE_TRAIN_DIR") or DEFAULT_WORKDIR).expanduser()


def run_dir(wd: Path, name: str) -> Path:
    return wd / "runs" / name


def plan_for(wd: Path, time_arg: str) -> dict:
    seconds, label = bench.parse_time(time_arg)
    b = bench.load_bench(wd)
    if b is None:
        raise Fail("bad_args", "no bench.json yet: run `bench` first (it measures this Mac in about 20 seconds)")
    return bench.derive_plan(b, seconds, label)


def emit_plan(em: Emitter, plan: dict) -> None:
    for line in plan["derivation"]:
        log(line)
    em.emit("plan", **plan)


# ---- subcommands ----
def cmd_bench(a, em: Emitter, stop: StopFlag) -> str:
    wd = workdir_of(a)
    if a.plan_only:
        if not a.time:
            raise Fail("bad_args", "--plan-only needs --time")
    else:
        bench.run_bench(wd, em, seconds_per_preset=a.seconds)
    for t in a.time or []:
        emit_plan(em, plan_for(wd, t))
    return "completed"


def cmd_prepare(a, em: Emitter, stop: StopFlag) -> str:
    wd = workdir_of(a)
    if not a.sft_only:
        if a.tokens:
            target = int(float(a.tokens))
        elif a.time:
            plan = plan_for(wd, a.time)
            emit_plan(em, plan)
            target = plan["tokens"]
        else:
            raise Fail("bad_args", "prepare needs --tokens N or --time T (or --sft-only)")
        em.emit("status", phase="prepare", message="downloading and tokenizing Wikipedia", detail={"target_tokens": target})
        man = prepare.prepare_wiki(wd, target, em, stop)
        if stop and not man.get("complete"):
            return "stopped"
    if a.sft or a.sft_only:
        em.emit("status", phase="prepare", message="preparing the instruction-tuning conversations")
        prepare.prepare_sft(wd, em, stop)
    return "completed"


def _check_same(rd: Path, cfg: RunConfig) -> None:
    if (rd / "run.json").exists():
        old = read_run(rd)["config"]
        new = json.loads(json.dumps(cfg.__dict__))
        diff = [k for k in new if old.get(k) != new[k]]
        if diff:
            raise Fail("bad_args", f"{rd.name} already exists with different settings ({', '.join(diff)}); pick another --run name")


def cmd_pretrain(a, em: Emitter, stop: StopFlag) -> str:
    wd = workdir_of(a)
    if a.time:
        plan = plan_for(wd, a.time)
        emit_plan(em, plan)
        preset, steps = presets.get(plan["preset"]), plan["pretrain_steps"]
    elif a.preset:
        preset = presets.get(a.preset)
        if not (a.tokens or a.steps):
            raise Fail("bad_args", "--preset needs --tokens or --steps")
        steps = int(a.steps or int(float(a.tokens)) // preset.tokens_per_step)
        plan = None
    else:
        raise Fail("bad_args", "pretrain needs --time T, or --preset P with --tokens/--steps")
    man = read_manifest(prepare.wiki_dir(wd))
    if not man or not man["train"] or not man["val"]:
        raise Fail("no_data", "no prepared text: run `prepare` first")
    train = TokenWindows(prepare.wiki_dir(wd), man["train"], preset.seq_len, seed=a.seed)
    val = TokenWindows(prepare.wiki_dir(wd), man["val"], preset.seq_len)
    need = steps * preset.tokens_per_step
    if train.num_tokens < need:
        log(f"note: {train.num_tokens:,} tokens prepared, {need:,} planned: some text is read twice (run `prepare` for more)")
    _, ws = bench.device_memory()
    cfg = RunConfig(
        phase="pretrain", preset=preset.name, shape=preset.shape.to_dict(), seq_len=preset.seq_len,
        tokens_per_step=preset.tokens_per_step, micro_batch=a.micro_batch or bench.micro_batch_for(preset, ws),
        total_steps=steps, peak_lr=a.lr or preset.peak_lr, warmup_steps=max(min(2000, steps // 20), 10), schedule="cosine",
        seed=a.seed, prompts=[{"text": p, "shown": p} for p in PRETRAIN_PROMPTS], prompt_kind="plain")
    rd = run_dir(wd, a.run)
    _check_same(rd, cfg)
    write_run(rd, cfg, {"plan": plan, "data": {k: man[k] for k in ("source", "tokenizer_sha256")}, "data_tokens": train.num_tokens})
    em.tee(rd / "events.jsonl")
    tok = prepare.load_tokenizer(wd)
    return Trainer(rd, cfg, train, val, tok, em, stop).run()


def cmd_sft(a, em: Emitter, stop: StopFlag) -> str:
    wd = workdir_of(a)
    base = run_dir(wd, a.run)
    if not (base / FINAL_WEIGHTS).exists():
        raise Fail("no_checkpoint", f"{a.run} has not finished pretraining (no {FINAL_WEIGHTS})")
    bcfg = read_run(base)["config"]
    preset = presets.get(bcfg["preset"])
    man = read_manifest(prepare.sft_dir(wd))
    if not man or not man.get("complete"):
        em.emit("status", phase="prepare", message="preparing the instruction-tuning conversations")
        man = prepare.prepare_sft(wd, em, stop)
    train = TokenWindows(prepare.sft_dir(wd), man["train"], preset.seq_len, seed=a.seed)
    val = TokenWindows(prepare.sft_dir(wd), man["val"], preset.seq_len)
    passes = max(int(bench.SFT_EPOCHS * train.num_tokens // preset.tokens_per_step), 1)  # two passes over the conversations
    if a.steps:
        steps = a.steps
    elif a.time:
        steps = min(plan_for(wd, a.time)["sft_steps"], passes)
    else:
        steps = passes
    _, ws = bench.device_memory()
    cfg = RunConfig(
        phase="sft", preset=preset.name, shape=bcfg["shape"], seq_len=preset.seq_len, tokens_per_step=preset.tokens_per_step,
        micro_batch=a.micro_batch or bench.micro_batch_for(preset, ws), total_steps=steps,
        peak_lr=a.lr or bcfg["peak_lr"] * 0.1, warmup_steps=max(steps // 30, 1), schedule="linear", weight_decay=0.0,
        seed=a.seed, prompts=[{"text": chat.prompt_text(q), "shown": q} for q in SFT_PROMPTS], prompt_kind="chat",
        parent=str(base / FINAL_WEIGHTS), eval_windows=16)
    rd = run_dir(wd, a.out_run or f"{a.run}-sft")
    _check_same(rd, cfg)
    write_run(rd, cfg, {"base_run": a.run, "data": {k: man[k] for k in ("source", "filters", "examples", "by_source")},
                        "data_tokens": train.num_tokens})
    em.tee(rd / "events.jsonl")
    tok = prepare.load_tokenizer(wd)
    return Trainer(rd, cfg, train, val, tok, em, stop).run()


def cmd_export(a, em: Emitter, stop: StopFlag) -> str:
    from .export import write_export
    from .model import Olmo2
    from .train import MILESTONE_DIR, load_weights

    wd = workdir_of(a)
    rd = run_dir(wd, a.run)
    info = read_run(rd)
    cfg = info["config"]
    weights = rd / FINAL_WEIGHTS if a.step is None else rd / MILESTONE_DIR / f"step-{a.step:07d}.safetensors"
    if not weights.exists():
        raise Fail("no_checkpoint", f"{weights} does not exist")
    model = Olmo2(presets.ModelShape(**cfg["shape"]))
    load_weights(model, weights)
    is_sft = cfg["phase"] == "sft"
    base = read_run(run_dir(wd, info["base_run"])) if is_sft else info
    bcfg = base["config"]
    ms = json.loads((rd / "milestones.json").read_text()) if (rd / "milestones.json").exists() else []
    final_ms = next((m for m in ms if m["step"] == (cfg["total_steps"] if a.step is None else a.step)), None)
    training = {
        "trainer": f"neuroscope-trainer {__version__}",
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "preset": bcfg["preset"],
        "shape": cfg["shape"],
        "seq_len": cfg["seq_len"],
        # a milestone of a pretraining run carries only the tokens seen up to it
        "pretrain_steps": bcfg["total_steps"] if (is_sft or a.step is None) else a.step,
        "pretrain_tokens": (bcfg["total_steps"] if (is_sft or a.step is None) else a.step) * bcfg["tokens_per_step"],
        "pretrain_data": base["data"],
        "weights_from": str(weights.relative_to(wd)),
        "val_loss": final_ms.get("val_loss") if final_ms else None,
    }
    if is_sft:
        sft_steps = cfg["total_steps"] if a.step is None else a.step
        training["sft"] = {"steps": sft_steps, "tokens": sft_steps * cfg["tokens_per_step"], "data": info["data"]}
    out = Path(a.out).expanduser() if a.out else wd / "exports" / (a.run if a.step is None else f"{a.run}-step{a.step}")
    files = write_export(out, model, prepare.ensure_tokenizer(wd), chat=is_sft, training=training, dtype=a.dtype)
    em.emit("export", path=str(out), files=files, params=model.shape.total_params())
    return "completed"


def cmd_compare(a, em: Emitter, stop: StopFlag) -> str:
    from .compare import compare_with_engine

    ok = compare_with_engine(Path(a.export).expanduser(), Path(a.engine).expanduser(), em, tol=a.tol)
    if not ok:
        raise Fail("internal", "engine logits differ from MLX beyond the tolerance")
    return "completed"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="neuroscope-train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", help=f"where data, runs and exports live (default {DEFAULT_WORKDIR}, or $NEUROSCOPE_TRAIN_DIR)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="measure this Mac (about 20 s) and derive plans for chosen times")
    b.add_argument("--time", action="append", help="30m, 2h, overnight, or e.g. 45m; repeatable")
    b.add_argument("--seconds", type=float, default=4.0, help="measuring time per preset")
    b.add_argument("--plan-only", action="store_true", help="derive plans from the saved bench.json without measuring")

    p = sub.add_parser("prepare", help="stream and tokenize Wikipedia (and the instruction-tuning set)")
    p.add_argument("--tokens", help="training tokens to prepare, e.g. 50e6")
    p.add_argument("--time", help="prepare what the plan for this time needs")
    p.add_argument("--sft", action="store_true", help="also prepare the instruction-tuning conversations")
    p.add_argument("--sft-only", action="store_true")

    t = sub.add_parser("pretrain", help="train a model from scratch on the prepared text")
    t.add_argument("--run", default="my-model")
    t.add_argument("--time")
    t.add_argument("--preset", help=", ".join(presets.PRESETS))
    t.add_argument("--tokens")
    t.add_argument("--steps", type=int)
    t.add_argument("--lr", type=float)
    t.add_argument("--micro-batch", type=int)
    t.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("sft", help="instruction-tune a finished pretraining run")
    s.add_argument("--run", default="my-model", help="the pretraining run to start from")
    s.add_argument("--out-run", help="name of the tuned run (default <run>-sft)")
    s.add_argument("--time", help="use the plan's instruction-tuning steps for this time")
    s.add_argument("--steps", type=int)
    s.add_argument("--lr", type=float)
    s.add_argument("--micro-batch", type=int)
    s.add_argument("--seed", type=int, default=0)

    e = sub.add_parser("export", help="write a run as a Hugging Face OLMo 2 folder the app can open")
    e.add_argument("--run", default="my-model-sft")
    e.add_argument("--step", type=int, help="export a milestone instead of the final weights")
    e.add_argument("--out")
    e.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")

    c = sub.add_parser("compare", help="check the app engine's logits against MLX on an exported model")
    c.add_argument("--export", required=True)
    c.add_argument("--engine", required=True, help="JSON written by `cargo run --example logits` in engine/")
    c.add_argument("--tol", type=float, default=None, help="max |logit difference| allowed (default by dtype)")
    return ap


COMMANDS = {"bench": cmd_bench, "prepare": cmd_prepare, "pretrain": cmd_pretrain, "sft": cmd_sft, "export": cmd_export, "compare": cmd_compare}


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    em = Emitter(a.cmd)
    stop = StopFlag().install()
    em.emit("hello", version=__version__, protocol=PROTOCOL_VERSION, pid=os.getpid(), args={k: v for k, v in vars(a).items() if v is not None})
    status, code = "failed", 1
    try:
        status = COMMANDS[a.cmd](a, em, stop)
        code = {"completed": 0, "stopped": 3}.get(status, 1)
    except Fail as f:
        em.emit("error", code=f.code, message=str(f))
        code = 2 if f.code == "bad_args" else 1
    except prepare.DiskFull as e:
        em.emit("error", code="disk_full", message=str(e))
    except KeyboardInterrupt:
        status, code = "stopped", 3
    except Exception as e:  # noqa: BLE001 - the app needs an event, not a traceback
        from .net import NetworkError

        kind = "network" if isinstance(e, NetworkError) else ("out_of_memory" if "memory" in str(e).lower() else "internal")
        em.emit("error", code=kind, message=f"{type(e).__name__}: {e}", detail=traceback.format_exc()[-4000:])
    em.emit("done", status=status if code in (0, 3) else "failed")
    em.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
