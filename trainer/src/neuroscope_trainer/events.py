"""The JSON-lines progress protocol (docs/trainer-protocol.md is the contract).

One JSON object per line on stdout, flushed per line. Human-readable logs go to stderr. Every event carries
`v` (protocol version), `event` (its type), `t` (unix seconds) and `cmd` (the subcommand). Field names and types
per event are fixed in `SCHEMA`; `emit` refuses an event that does not match, so the contract cannot drift
silently.
"""

import json
import math
import signal
import sys
import time
from typing import IO, Any

from . import PROTOCOL_VERSION

NUM = (int, float)
# event -> {field: type or tuple of types}; optional fields are listed in OPTIONAL.
SCHEMA: dict[str, dict[str, Any]] = {
    "hello": {"version": str, "protocol": int, "pid": int, "args": dict},
    "status": {"phase": str, "message": str},
    "bench": {"device": str, "memory_gb": NUM, "presets": list},
    "plan": {"time_budget_s": NUM, "preset": str, "tokens": int, "sft_steps": int, "derivation": list, "candidates": list},
    "prepare": {"split": str, "tokens": int, "target_tokens": int, "docs": int, "bytes_downloaded": int, "tok_per_s": NUM, "eta_s": NUM},
    "progress": {"phase": str, "step": int, "total_steps": int, "tokens": int, "total_tokens": int, "loss": NUM, "lr": NUM, "tok_per_s": NUM, "elapsed_s": NUM, "eta_s": NUM},
    "eval": {"phase": str, "step": int, "tokens": int, "val_loss": NUM},
    "sample": {"phase": str, "step": int, "tokens": int, "prompt_id": int, "prompt": str, "text": str},
    "checkpoint": {"phase": str, "step": int, "tokens": int, "path": str, "kind": str},
    "export": {"path": str, "files": list, "params": int},
    "compare": {"prompt": str, "max_abs_diff": NUM, "cosine": NUM, "top1_match": bool, "dtype": str, "ok": bool},
    "done": {"status": str},
    "error": {"code": str, "message": str},
}
OPTIONAL: dict[str, dict[str, Any]] = {
    "plan": {"time_label": str, "seq_len": int, "tokens_per_step": int, "pretrain_steps": int},
    "progress": {"grad_norm": NUM},
    "done": {"summary": dict},
    "error": {"detail": str},
    "sample": {"kind": str},
    "status": {"detail": dict},
}

# Error codes the app can branch on.
ERROR_CODES = {"disk_full", "network", "no_data", "no_checkpoint", "bad_args", "nan_loss", "out_of_memory", "internal"}
DONE_STATUSES = {"completed", "stopped", "failed"}


class ProtocolError(ValueError):
    pass


def validate(ev: dict) -> None:
    kind = ev.get("event")
    if kind not in SCHEMA:
        raise ProtocolError(f"unknown event {kind!r}")
    req, opt = SCHEMA[kind], OPTIONAL.get(kind, {})
    for k, typ in req.items():
        if k not in ev:
            raise ProtocolError(f"{kind}: missing field {k!r}")
        if not _is(ev[k], typ):
            raise ProtocolError(f"{kind}.{k}: {type(ev[k]).__name__} is not {typ}")
    for k, v in ev.items():
        if k in ("v", "event", "t", "cmd") or k in req:
            continue
        if k not in opt:
            raise ProtocolError(f"{kind}: unexpected field {k!r}")
        if not _is(v, opt[k]):
            raise ProtocolError(f"{kind}.{k}: {type(v).__name__} is not {opt[k]}")
    if kind == "done" and ev["status"] not in DONE_STATUSES:
        raise ProtocolError(f"done.status {ev['status']!r}")
    if kind == "error" and ev["code"] not in ERROR_CODES:
        raise ProtocolError(f"error.code {ev['code']!r}")
    for k, v in ev.items():
        if isinstance(v, float) and not math.isfinite(v):
            raise ProtocolError(f"{kind}.{k} is not finite")


def _is(v, typ) -> bool:
    if isinstance(v, bool) and typ is not bool and (typ is int or (isinstance(typ, tuple) and bool not in typ)):
        return False  # bools are ints in Python; not in the protocol
    return isinstance(v, typ)


class Emitter:
    """Writes protocol events to `out` (stdout by default) and mirrors a copy to an optional log file."""

    def __init__(self, cmd: str, out: IO[str] | None = None):
        self.cmd = cmd
        self.out = out if out is not None else sys.stdout
        self.files: list[IO[str]] = []

    def tee(self, path) -> None:
        self.files.append(open(path, "a", encoding="utf-8"))

    def emit(self, event: str, **fields) -> dict:
        ev = {"v": PROTOCOL_VERSION, "event": event, "t": round(time.time(), 3), "cmd": self.cmd, **fields}
        validate(ev)
        line = json.dumps(ev, ensure_ascii=False, separators=(",", ":"))
        for f in (self.out, *self.files):
            f.write(line + "\n")
            f.flush()
        return ev

    def close(self) -> None:
        for f in self.files:
            f.close()
        self.files = []


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class StopFlag:
    """Set by SIGTERM or SIGINT; long loops check it between steps and stop cleanly."""

    def __init__(self):
        self.stop = False
        self._prev = {}

    def install(self) -> "StopFlag":
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._prev[sig] = signal.signal(sig, self._handle)
        return self

    def _handle(self, signum, frame):
        if self.stop:  # second signal: give up on a clean stop
            raise KeyboardInterrupt
        self.stop = True
        log(f"signal {signum}: stopping after the current step")

    def __bool__(self) -> bool:
        return self.stop
