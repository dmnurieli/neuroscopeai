import numpy as np

from neuroscope_trainer.shards import ShardWriter, TokenWindows, read_manifest, write_manifest


def make(tmp_path, docs, shard_tokens, with_mask=False):
    w = ShardWriter(tmp_path, "train", shard_tokens, with_mask=with_mask)
    for i, d in enumerate(docs):
        w.add(d, i + 1, mask=[i % 2] * len(d) if with_mask else None)
    w.flush(len(docs))
    return w.shards


def test_shards_hold_whole_documents_in_order(tmp_path):
    docs = [list(range(k * 10, k * 10 + 7)) + [99] for k in range(30)]
    shards = make(tmp_path, docs, shard_tokens=50)
    assert sum(s["tokens"] for s in shards) == 30 * 8
    joined = np.concatenate([np.fromfile(tmp_path / s["file"], dtype="<u4") for s in shards])
    assert joined.tolist() == [t for d in docs for t in d]
    for s in shards:  # every shard ends on a document's end token
        assert np.fromfile(tmp_path / s["file"], dtype="<u4")[-1] == 99
    assert shards[-1]["docs_end"] == 30
    assert [s["docs_end"] for s in shards] == sorted(s["docs_end"] for s in shards)
    assert not list(tmp_path.glob("*.part"))


def test_ids_above_uint16_survive(tmp_path):
    shards = make(tmp_path, [[100_257, 70_000, 5, 100_351]], shard_tokens=10)
    assert np.fromfile(tmp_path / shards[0]["file"], dtype="<u4").tolist() == [100_257, 70_000, 5, 100_351]


def test_windows_are_consecutive_tokens_and_never_cross_shards(tmp_path):
    docs = [[k] * 9 + [999] for k in range(40)]  # 10 tokens each
    shards = make(tmp_path, docs, shard_tokens=100)  # 4 shards of 100
    tw = TokenWindows(tmp_path, shards, seq_len=16)
    assert tw.counts.tolist() == [6, 6, 6, 6]  # (100 - 1) // 16
    for w in range(tw.num_windows):
        toks, _ = tw.window(w)
        s = int(np.searchsorted(tw.starts, w, side="right") - 1)
        shard = np.fromfile(tmp_path / shards[s]["file"], dtype="<u4")
        off = (w - tw.starts[s]) * 16
        assert toks.tolist() == shard[off : off + 17].tolist()
    x, y, m = tw.batch(0, 5)
    assert x.shape == (5, 16) and y.shape == (5, 16) and m.shape == (5, 16)
    assert (x[:, 1:] == y[:, :-1]).all()
    assert m.min() == 1.0


def test_batch_order_is_a_pure_function_of_seed_and_index(tmp_path):
    shards = make(tmp_path, [list(range(i, i + 50)) for i in range(20)], shard_tokens=200)
    a = TokenWindows(tmp_path, shards, 16, seed=7)
    b = TokenWindows(tmp_path, shards, 16, seed=7)
    whole = a.batch(0, 12)[0]
    # a resumed reader starting at draw 8 sees exactly draws 8..11
    assert (b.batch(8, 4)[0] == whole[8:12]).all()
    # one pass visits every window once, then a new order begins
    n = a.num_windows
    first = a.window_ids(0, n)
    assert sorted(first) == list(range(n))
    assert a.window_ids(n, n) != first
    assert TokenWindows(tmp_path, shards, 16, seed=8).window_ids(0, n) != first


def test_mask_is_aligned_with_targets(tmp_path):
    docs = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    shards = make(tmp_path, docs, shard_tokens=100, with_mask=True)  # masks: 0,0,0,0,0, 1,1,1,1,1, 0,...
    tw = TokenWindows(tmp_path, shards, seq_len=14)
    x, y, m = tw.batch(0, 1)
    assert y[0].tolist() == list(range(2, 16))
    # target token t is masked like the document it belongs to
    assert m[0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


def test_manifest_round_trip(tmp_path):
    write_manifest(tmp_path, {"a": 1, "train": []})
    assert read_manifest(tmp_path) == {"a": 1, "train": []}
    assert read_manifest(tmp_path / "nope") is None
