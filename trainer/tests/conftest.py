import io
import json

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

from neuroscope_trainer.events import Emitter, StopFlag
from neuroscope_trainer.presets import TEST_SHAPE
from neuroscope_trainer.shards import ShardWriter, TokenWindows
from neuroscope_trainer.train import RunConfig

CORPUS = [
    "The river flows to the sea. The sea is blue.",
    "An airplane is a machine that flies. Airplanes fly over the sea.",
    "The capital of France is Paris. Paris is a city.",
    "<|user|>\nWhat is blue?\n<|assistant|>\nThe sea is blue.",
] * 20


@pytest.fixture(scope="session")
def tiny_tokenizer() -> Tokenizer:
    """A byte-level BPE with OLMo 2's special token, small enough for the test model's 512-entry vocabulary."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=400, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(CORPUS, tr)
    return tok


@pytest.fixture
def emitter():
    buf = io.StringIO()
    em = Emitter("test", out=buf)
    em.buffer = buf
    return em


def events_of(em) -> list[dict]:
    return [json.loads(line) for line in em.buffer.getvalue().splitlines()]


def write_corpus(folder, tok, docs=200, shard_tokens=2000, with_mask=False, seed=0):
    eos = tok.token_to_id("<|endoftext|>")
    w = ShardWriter(folder, "train", shard_tokens, with_mask=with_mask)
    rng = np.random.default_rng(seed)
    for i in range(docs):
        ids = tok.encode(CORPUS[rng.integers(len(CORPUS))], add_special_tokens=False).ids + [eos]
        w.add(ids, i + 1, mask=[1] * len(ids) if with_mask else None)
    w.flush(docs)
    return w.shards


def tiny_run_config(steps=6, **kw) -> RunConfig:
    base = dict(phase="pretrain", shape=TEST_SHAPE.to_dict(), seq_len=32, tokens_per_step=4 * 32, micro_batch=4,
                total_steps=steps, peak_lr=3e-3, warmup_steps=2, schedule="cosine", seed=0,
                prompts=[{"text": "The river", "shown": "The river"}], sample_tokens=5, eval_windows=4)
    base.update(kw)
    return RunConfig(**base)


def windows(folder, shards, seq=32, seed=0):
    return TokenWindows(folder, shards, seq, seed=seed)


__all__ = ["events_of", "write_corpus", "tiny_run_config", "windows", "StopFlag"]
