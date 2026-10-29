"""Token shards on disk and the deterministic batch order the trainers read them in.

A split is a folder with `manifest.json` and shard files of little-endian uint32 token ids (OLMo 2's vocabulary
needs 17 bits, so uint16 will not do). Shards always end at a document boundary; each document is its tokens
followed by <|endoftext|>. An optional `.mask` companion file (uint8, same length) marks the tokens a loss is taken
on (instruction tuning trains on the assistant's words only).

Training reads windows of `seq_len + 1` tokens, stride `seq_len`, never crossing a shard boundary. The window
order is a seeded permutation, re-drawn per pass over the data, so batch `k` is a pure function of (seed, k):
resuming a run reads exactly the batches it would have read.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MANIFEST = "manifest.json"
TOKEN_DTYPE = np.dtype("<u4")


def read_manifest(folder: Path) -> dict | None:
    p = Path(folder) / MANIFEST
    if not p.exists():
        return None
    return json.loads(p.read_text())


def write_manifest(folder: Path, manifest: dict) -> None:
    p = Path(folder) / MANIFEST
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, p)


@dataclass
class ShardWriter:
    """Accumulates whole documents and writes a shard each time `shard_tokens` is reached.

    Shards are written to a temporary name and renamed, and the manifest is rewritten after every shard, so a
    killed process loses at most the documents of the shard in progress. `docs_end` in each shard record is the
    number of source documents consumed up to the end of that shard: the resume point.
    """

    folder: Path
    prefix: str
    shard_tokens: int
    with_mask: bool = False
    shards: list = field(default_factory=list)
    _buf: list = field(default_factory=list)
    _mbuf: list = field(default_factory=list)
    _n: int = 0

    def __post_init__(self):
        self.folder = Path(self.folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    @property
    def tokens_written(self) -> int:
        return sum(s["tokens"] for s in self.shards)

    @property
    def tokens_pending(self) -> int:
        return self._n

    def add(self, ids, docs_end: int, mask=None) -> dict | None:
        """Adds one document (ids already end with <|endoftext|>). Returns the shard record if one was written."""
        a = np.asarray(ids, dtype=TOKEN_DTYPE)
        self._buf.append(a)
        if self.with_mask:
            m = np.asarray(mask, dtype=np.uint8)
            assert m.shape == a.shape, "mask must match ids"
            self._mbuf.append(m)
        self._n += a.size
        if self._n >= self.shard_tokens:
            return self.flush(docs_end)
        return None

    def flush(self, docs_end: int) -> dict | None:
        if self._n == 0:
            return None
        name = f"{self.prefix}-{len(self.shards):05d}.u32"
        data = np.concatenate(self._buf)
        _atomic_write(self.folder / name, data.tobytes())
        rec = {"file": name, "tokens": int(data.size), "docs_end": int(docs_end)}
        if self.with_mask:
            _atomic_write(self.folder / (name + ".mask"), np.concatenate(self._mbuf).tobytes())
            rec["mask"] = name + ".mask"
        self.shards.append(rec)
        self._buf, self._mbuf, self._n = [], [], 0
        return rec


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


class TokenWindows:
    """Fixed-length training windows over the shards of one split."""

    def __init__(self, folder: Path, shard_records: list, seq_len: int, seed: int = 0):
        self.folder = Path(folder)
        self.seq_len = seq_len
        self.seed = seed
        self.tokens = []
        self.masks = []
        counts = []
        for rec in shard_records:
            t = np.memmap(self.folder / rec["file"], dtype=TOKEN_DTYPE, mode="r")
            assert t.size == rec["tokens"], f"{rec['file']}: {t.size} tokens on disk, manifest says {rec['tokens']}"
            self.tokens.append(t)
            self.masks.append(np.memmap(self.folder / rec["mask"], dtype=np.uint8, mode="r") if "mask" in rec else None)
            counts.append(max((t.size - 1) // seq_len, 0))
        self.counts = np.array(counts, dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.counts)])
        self.num_windows = int(self.starts[-1])
        if self.num_windows == 0:
            raise ValueError(f"{folder}: not enough tokens for one window of {seq_len + 1}")
        self._perm_epoch = -1
        self._perm = None

    @property
    def num_tokens(self) -> int:
        return int(sum(t.size for t in self.tokens))

    def _order(self, epoch: int) -> np.ndarray:
        if epoch != self._perm_epoch:
            self._perm = np.random.default_rng([self.seed, epoch]).permutation(self.num_windows)
            self._perm_epoch = epoch
        return self._perm

    def window_ids(self, start: int, count: int) -> list[int]:
        """Global window indices for draws start .. start+count-1 (a draw index counts across passes)."""
        out = []
        for i in range(start, start + count):
            epoch, j = divmod(i, self.num_windows)
            out.append(int(self._order(epoch)[j]))
        return out

    def window(self, w: int):
        s = int(np.searchsorted(self.starts, w, side="right") - 1)
        off = (w - int(self.starts[s])) * self.seq_len
        toks = np.asarray(self.tokens[s][off : off + self.seq_len + 1], dtype=np.int32)
        m = self.masks[s]
        mask = None if m is None else np.asarray(m[off + 1 : off + self.seq_len + 1], dtype=np.float32)
        return toks, mask

    def batch(self, start: int, count: int):
        """(inputs [count, seq], targets [count, seq], loss mask [count, seq] float32) for draws start..start+count-1."""
        xs, ms = [], []
        for w in self.window_ids(start, count):
            t, m = self.window(w)
            xs.append(t)
            ms.append(np.ones(self.seq_len, np.float32) if m is None else m)
        x = np.stack(xs)
        return x[:, :-1], x[:, 1:], np.stack(ms)

    def fixed_windows(self, count: int):
        """The same `count` evenly spaced windows every time (for validation loss)."""
        count = min(count, self.num_windows)
        idx = np.linspace(0, self.num_windows - 1, count).astype(np.int64)
        xs, ms = [], []
        for w in idx:
            t, m = self.window(int(w))
            xs.append(t)
            ms.append(np.ones(self.seq_len, np.float32) if m is None else m)
        x = np.stack(xs)
        return x[:, :-1], x[:, 1:], np.stack(ms)
