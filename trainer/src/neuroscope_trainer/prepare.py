"""`prepare`: fetch the tokenizer, stream Wikipedia from OLMo-mix into token shards, and build the small
instruction-tuning set. Resumable: rerunning continues from the last finished shard."""

import json
import shutil
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from . import chat, sources
from .events import Emitter, StopFlag, log
from .net import GzipLines, HttpRangeFile, download, sha256_file
from .presets import EOS_ID
from .shards import ShardWriter, read_manifest, write_manifest

BYTES_PER_TOKEN = 4
SHARD_TOKENS = 16 * 1024 * 1024  # 64 MB per shard
VAL_TOKENS = 1 << 20  # the first ~1M tokens are held out for validation loss
DISK_RESERVE = 2 << 30  # always leave 2 GB free
ENCODE_BATCH = 256


class DiskFull(RuntimeError):
    pass


def tokenizer_dir(workdir: Path) -> Path:
    return Path(workdir) / "tokenizer"


def ensure_tokenizer(workdir: Path) -> Path:
    """tokenizer.json (base model), tokenizer_config.json + special_tokens_map.json (Instruct: the chat template)."""
    d = tokenizer_dir(workdir)
    files = {
        "tokenizer.json": sources.model_file_url(sources.TOKENIZER_REPO, sources.TOKENIZER_REVISION, "tokenizer.json"),
        "tokenizer_config.json": sources.model_file_url(sources.INSTRUCT_REPO, sources.INSTRUCT_REVISION, "tokenizer_config.json"),
        "special_tokens_map.json": sources.model_file_url(sources.INSTRUCT_REPO, sources.INSTRUCT_REVISION, "special_tokens_map.json"),
    }
    for name, url in files.items():
        if not (d / name).exists():
            log(f"downloading {name}")
            download(url, d / name)
    return d


def load_tokenizer(workdir: Path) -> Tokenizer:
    tok = Tokenizer.from_file(str(ensure_tokenizer(workdir) / "tokenizer.json"))
    assert tok.token_to_id("<|endoftext|>") == EOS_ID, "unexpected tokenizer: <|endoftext|> is not 100257"
    return tok


def check_disk(folder: Path, need_bytes: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free
    if free - need_bytes < DISK_RESERVE:
        raise DiskFull(
            f"needs {need_bytes / 1e9:.1f} GB plus a {DISK_RESERVE / 1e9:.0f} GB reserve; {free / 1e9:.1f} GB free at {folder}"
        )


def wiki_dir(workdir: Path) -> Path:
    return Path(workdir) / "data" / "wiki"


def sft_dir(workdir: Path) -> Path:
    return Path(workdir) / "data" / "sft"


def prepare_wiki(workdir: Path, target_tokens: int, em: Emitter, stop: StopFlag) -> dict:
    """Streams wiki-0000.json.gz and writes ~`target_tokens` training tokens (whole documents) plus the
    validation split. Returns the manifest."""
    tok = load_tokenizer(workdir)
    tok_sha = sha256_file(tokenizer_dir(workdir) / "tokenizer.json")
    out = wiki_dir(workdir)
    man = read_manifest(out)
    if man and man.get("tokenizer_sha256") != tok_sha:
        raise RuntimeError(f"{out} was tokenized with another tokenizer; move it away to start over")
    if man is None:
        man = {
            "format": 1,
            "source": {"repo": sources.WIKI_REPO, "revision": sources.WIKI_REVISION, "file": sources.WIKI_FILE},
            "tokenizer_sha256": tok_sha,
            "eos_id": EOS_ID,
            "val": [],
            "train": [],
            "docs_consumed": 0,
        }
    have = sum(s["tokens"] for s in man["train"])
    man["target_tokens"] = int(target_tokens)
    if have >= target_tokens and man["val"]:
        man["complete"] = True
        write_manifest(out, man)
        em.emit("prepare", split="wiki", tokens=have, target_tokens=int(target_tokens), docs=man["docs_consumed"],
                bytes_downloaded=0, tok_per_s=0.0, eta_s=0.0)
        return man
    man["complete"] = False
    check_disk(out, int((target_tokens - have + (0 if man["val"] else VAL_TOKENS)) * BYTES_PER_TOKEN * 1.05))

    val_w = ShardWriter(out, "val", VAL_TOKENS, shards=list(man["val"]))
    train_w = ShardWriter(out, "train", SHARD_TOKENS, shards=list(man["train"]))
    skip = man["docs_consumed"]
    log(f"streaming {sources.WIKI_URL}" + (f" (skipping {skip:,} documents already tokenized)" if skip else ""))
    stream = GzipLines(sources.WIKI_URL)
    t0, last_emit, doc_i = time.time(), 0.0, 0
    batch: list[tuple[str, int]] = []

    def save():
        man["val"], man["train"] = val_w.shards, train_w.shards
        last = (train_w.shards or val_w.shards or [{"docs_end": 0}])[-1]
        man["docs_consumed"] = last["docs_end"]
        write_manifest(out, man)

    def done_tokens():
        return train_w.tokens_written + train_w.tokens_pending

    def report(force=False):
        nonlocal last_emit
        now = time.time()
        if not force and now - last_emit < 2.0:
            return
        last_emit = now
        n = done_tokens()
        rate = max(n - have, 0) / max(now - t0, 1e-6)
        eta = (target_tokens - n) / rate if rate > 0 else 0.0
        em.emit("prepare", split="wiki", tokens=int(n), target_tokens=int(target_tokens), docs=int(doc_i),
                bytes_downloaded=int(stream.bytes_read), tok_per_s=round(rate, 1), eta_s=round(max(eta, 0.0), 1))

    def flush_batch():
        texts = [t for t, _ in batch]
        for enc, (_, end) in zip(tok.encode_batch_fast(texts, add_special_tokens=False), batch):
            ids = enc.ids + [EOS_ID]
            if not val_w.shards:
                val_w.add(ids, docs_end=end)
                if val_w.shards:
                    save()
            elif train_w.add(ids, docs_end=end):
                save()
        batch.clear()

    for line in stream:
        doc_i += 1  # every line of the file counts as one document, empty or not, so resume offsets are exact
        if doc_i <= skip or not line.strip():
            continue
        text = json.loads(line).get("text") or ""
        if not text.strip():
            continue
        batch.append((text, doc_i))
        if len(batch) >= ENCODE_BATCH:
            flush_batch()
            report()
            if done_tokens() >= target_tokens or stop:
                break
    if batch:
        flush_batch()
    if done_tokens() >= target_tokens or not stop:
        # target reached (or the file ended): keep the partial shard
        train_w.flush(doc_i)
        save()
    man["complete"] = done_tokens() >= target_tokens
    write_manifest(out, man)
    report(force=True)
    return man


# ---- instruction tuning set ----
# Shard 0 of the Tulu 3 mixture holds its general English chat sources (row groups of 1,000 rows, contiguous by
# source). We read a fixed set of row groups over HTTP ranges, not the whole file.
SFT_FILE = sources.TULU_FILES[0]
# oasst1 (human-written, Apache 2.0) whole, and every 3rd FLAN v2 group (Apache 2.0). Left out on purpose: No Robots
# (non-commercial license), WildChat (replies written by another company's model, under that company's terms), and
# Tulu's hard-coded identity answers ("I am Tulu, made by Ai2": not this model's identity).
SFT_ROW_GROUPS = list(range(0, 7)) + list(range(7, 97, 3))
SFT_SOURCES = {"ai2-adapt-dev/oasst1_converted", "ai2-adapt-dev/flan_v2_converted"}
SFT_MAX_TOKENS = 1024  # the training context
FLAN_PER_OASST = 2
SFT_VAL_EXAMPLES = 200


def _english(text: str) -> bool:
    if not text:
        return False
    ascii_letters = sum(c.isascii() for c in text)
    return ascii_letters / len(text) >= 0.97


def prepare_sft(workdir: Path, em: Emitter, stop: StopFlag, max_examples: int = 15000) -> dict:
    import pyarrow.parquet as pq

    tok = load_tokenizer(workdir)
    out = sft_dir(workdir)
    man = read_manifest(out)
    if man and man.get("complete"):
        em.emit("prepare", split="sft", tokens=sum(s["tokens"] for s in man["train"]), target_tokens=0,
                docs=man["examples"], bytes_downloaded=0, tok_per_s=0.0, eta_s=0.0)
        return man
    check_disk(out, 200 << 20)
    url = sources.model_file_url("datasets/" + sources.TULU_REPO, sources.TULU_REVISION, SFT_FILE)
    f = HttpRangeFile(url)
    pf = pq.ParquetFile(f)
    rows = []
    t0 = time.time()
    for k, rg in enumerate(SFT_ROW_GROUPS):
        if stop:
            raise KeyboardInterrupt
        t = pf.read_row_group(rg, columns=["id", "messages", "source"]).to_pylist()
        rows.extend(r for r in t if r["source"] in SFT_SOURCES)
        em.emit("prepare", split="sft", tokens=0, target_tokens=0, docs=len(rows), bytes_downloaded=int(f.bytes_read),
                tok_per_s=0.0, eta_s=round((time.time() - t0) / (k + 1) * (len(SFT_ROW_GROUPS) - k - 1), 1))

    kept = []
    for r in rows:
        msgs = [m for m in r["messages"] if m["role"] in ("user", "assistant", "system")]
        if not msgs or msgs[-1]["role"] != "assistant" or not all(_english(m["content"]) for m in msgs):
            continue
        ids, mask = chat.render_training_example(tok, msgs)
        if len(ids) > SFT_MAX_TOKENS:
            continue
        kept.append((r["id"], r["source"], ids, mask))
    # Balance: FLAN's short task answers outnumber the human-written oasst1 conversations; keep at most
    # FLAN_PER_OASST of them per oasst1 conversation so the model learns to converse, not only to label.
    rng = np.random.default_rng(0)
    oasst = [k for k in kept if k[1] == "ai2-adapt-dev/oasst1_converted"]
    flan = [k for k in kept if k[1] != "ai2-adapt-dev/oasst1_converted"]
    flan = [flan[i] for i in rng.permutation(len(flan))[: FLAN_PER_OASST * len(oasst)]]
    kept = oasst + flan
    order = rng.permutation(len(kept))[:max_examples]
    kept = [kept[i] for i in order]
    val, train = kept[:SFT_VAL_EXAMPLES], kept[SFT_VAL_EXAMPLES:]
    val_w = ShardWriter(out, "val", 1 << 62, with_mask=True)
    train_w = ShardWriter(out, "train", 1 << 62, with_mask=True)
    for i, (_, _, ids, mask) in enumerate(val):
        val_w.add(ids, i + 1, mask)
    for i, (_, _, ids, mask) in enumerate(train):
        train_w.add(ids, i + 1, mask)
    val_w.flush(len(val))
    train_w.flush(len(train))
    by_source: dict[str, int] = {}
    for _, s, _, _ in kept:
        by_source[s] = by_source.get(s, 0) + 1
    man = {
        "format": 1,
        "source": {"repo": sources.TULU_REPO, "revision": sources.TULU_REVISION, "file": SFT_FILE, "row_groups": SFT_ROW_GROUPS},
        "filters": {"english_ascii_ratio": 0.97, "max_tokens": SFT_MAX_TOKENS, "sources": sorted(SFT_SOURCES), "flan_per_oasst": FLAN_PER_OASST},
        "examples": len(kept),
        "by_source": by_source,
        "val": val_w.shards,
        "train": train_w.shards,
        "complete": True,
    }
    write_manifest(out, man)
    em.emit("prepare", split="sft", tokens=train_w.tokens_written, target_tokens=train_w.tokens_written, docs=len(kept),
            bytes_downloaded=int(f.bytes_read), tok_per_s=0.0, eta_s=0.0)
    return man
