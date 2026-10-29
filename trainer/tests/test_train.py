import json

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

from neuroscope_trainer.bench import derive_plan, micro_batch_for, parse_time
from neuroscope_trainer.events import StopFlag, validate
from neuroscope_trainer.presets import PRESETS
from neuroscope_trainer.train import FINAL_WEIGHTS, Trainer

from conftest import events_of, tiny_run_config, windows, write_corpus


def flat(model):
    return {k: np.array(v) for k, v in tree_flatten(model.parameters())}


def test_loss_falls_and_every_event_is_valid(tmp_path, tiny_tokenizer, emitter):
    shards = write_corpus(tmp_path / "data", tiny_tokenizer)
    tr = Trainer(tmp_path / "run", tiny_run_config(steps=40, peak_lr=1e-2), windows(tmp_path / "data", shards),
                 windows(tmp_path / "data", shards), tiny_tokenizer, emitter, StopFlag())
    assert tr.run() == "completed"
    evs = events_of(emitter)
    for e in evs:
        validate(e)
    losses = [e["loss"] for e in evs if e["event"] == "progress"]
    vals = [e["val_loss"] for e in evs if e["event"] == "eval"]
    assert losses[-1] < losses[0] - 1.0
    assert vals[-1] < vals[0] - 1.0
    samples = [e for e in evs if e["event"] == "sample"]
    assert {e["step"] for e in samples} == set(tr.milestone_steps())
    assert (tmp_path / "run" / FINAL_WEIGHTS).exists()
    ms = json.loads((tmp_path / "run" / "milestones.json").read_text())
    assert [m["step"] for m in ms] == tr.milestone_steps()


def test_stop_then_resume_restores_exact_state(tmp_path, tiny_tokenizer, emitter):
    """Training on the GPU is not bit-reproducible (the embedding gradient's scatter-add order varies), so resume is
    checked where it can be exact: the restored weights, optimizer moments, step counters and data position."""
    shards = write_corpus(tmp_path / "data", tiny_tokenizer)
    data = lambda: windows(tmp_path / "data", shards, seed=5)  # noqa: E731
    cfg = tiny_run_config(steps=8, micro_batch=2)  # also exercises micro-batch accumulation

    stop = StopFlag()
    first = Trainer(tmp_path / "b", cfg, data(), None, tiny_tokenizer, emitter, stop)
    orig = first.train_step

    def step_then_stop():
        out = orig()
        if first.step == 4:  # this call becomes step 5; the loop stops before step 6
            stop.stop = True
        return out

    first.train_step = step_then_stop
    assert first.run() == "stopped"
    assert json.loads((tmp_path / "b" / "state" / "state.json").read_text())["step"] == 5
    saved_w, saved_o = flat(first.model), {k: np.array(v) for k, v in tree_flatten(first.opt.state)}

    second = Trainer(tmp_path / "b", cfg, data(), None, tiny_tokenizer, emitter, StopFlag())
    assert second.try_resume()
    assert second.step == 5
    got_w, got_o = flat(second.model), {k: np.array(v) for k, v in tree_flatten(second.opt.state)}
    assert saved_w.keys() == got_w.keys() and saved_o.keys() == got_o.keys()
    for k in saved_w:
        np.testing.assert_array_equal(saved_w[k], got_w[k], err_msg=k)
    for k in saved_o:
        np.testing.assert_array_equal(saved_o[k], got_o[k], err_msg=k)
    assert int(got_o["states.0.step"]) == 5
    n = cfg.seqs_per_step  # the next batch is the one an uninterrupted run reads at step 5
    np.testing.assert_array_equal(second.train.batch(5 * n, n)[0], data().batch(0, 6 * n)[0][5 * n :])

    assert second.run() == "completed"  # finishes the remaining steps, milestones included
    steps = [e["step"] for e in events_of(emitter) if e["event"] == "progress"]
    assert steps[-1] == 8
    ms = json.loads((tmp_path / "b" / "milestones.json").read_text())
    assert [m["step"] for m in ms] == second.milestone_steps()
    assert (tmp_path / "b" / FINAL_WEIGHTS).exists()


def test_accumulated_gradients_equal_one_big_batch(tmp_path, tiny_tokenizer, emitter):
    shards = write_corpus(tmp_path / "data", tiny_tokenizer)
    tr = Trainer(tmp_path / "r", tiny_run_config(steps=2), windows(tmp_path / "data", shards), None, tiny_tokenizer, emitter, StopFlag())
    tr.init_fresh()
    x, y, m = (mx.array(t) for t in tr.train.batch(0, 4))
    norm = mx.array(float(m.sum()))
    loss_all, g_all = tr._grad(x, y, m, norm)
    acc, loss_sum = None, 0.0
    for i in range(4):
        loss, g = tr._grad(x[i : i + 1], y[i : i + 1], m[i : i + 1], norm)
        acc = g if acc is None else tree_map(mx.add, acc, g)
        loss_sum += loss.item()
    assert abs(loss_all.item() - loss_sum) < 1e-3
    for (k, a), (_, b) in zip(tree_flatten(g_all), tree_flatten(acc)):
        a, b = np.array(a), np.array(b)
        assert np.abs(a - b).max() <= 2e-2 * np.abs(a).max() + 1e-7, k  # bf16 compute


def test_parse_time():
    assert parse_time("30m") == (1800.0, "30m")
    assert parse_time("overnight")[0] == 8 * 3600
    assert parse_time("1.5h")[0] == 5400
    assert parse_time("90")[0] == 90


def test_plan_picks_largest_model_with_enough_tokens():
    bench = {"presets": [
        {"preset": p.name, "tok_per_s": tps, "non_embedding_params": p.shape.non_embedding_params(),
         "tokens_per_step": p.tokens_per_step, "seq_len": p.seq_len}
        for p, tps in zip(PRESETS.values(), (120_000, 84_000, 52_000))
    ]}
    assert derive_plan(bench, 1800)["preset"] == "tiny"
    assert derive_plan(bench, 7200)["preset"] == "small"
    assert derive_plan(bench, 8 * 3600)["preset"] == "medium"
    tiny = derive_plan(bench, 60)  # nothing fits: the smallest, flagged
    assert tiny["preset"] == "tiny" and not any(c["fits"] for c in tiny["candidates"])
    p = derive_plan(bench, 7200)
    assert p["tokens"] == p["pretrain_steps"] * p["tokens_per_step"]
    assert any("20 training tokens per non-embedding parameter" in line for line in p["derivation"])
    validate({"v": 1, "event": "plan", "t": 0.0, "cmd": "x", **p})


def test_micro_batch_fits_memory():
    med = PRESETS["medium"]
    assert micro_batch_for(med, 80e9) == 16
    small_mac = micro_batch_for(med, 5e9)
    assert 1 <= small_mac < 16 and 16 % small_mac == 0
