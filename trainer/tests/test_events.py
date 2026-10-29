import io
import json

import pytest

from neuroscope_trainer.events import SCHEMA, Emitter, ProtocolError, validate


def ev(event, **kw):
    return {"v": 1, "event": event, "t": 1.0, "cmd": "x", **kw}


GOOD_PROGRESS = dict(phase="pretrain", step=3, total_steps=10, tokens=300, total_tokens=1000, loss=7.5, lr=1e-3,
                     tok_per_s=1000.0, elapsed_s=1.0, eta_s=7.0)


def test_valid_events_pass():
    validate(ev("progress", **GOOD_PROGRESS))
    validate(ev("progress", **GOOD_PROGRESS, grad_norm=0.5))
    validate(ev("done", status="completed"))
    validate(ev("error", code="disk_full", message="no room"))
    validate(ev("sample", phase="sft", step=0, tokens=0, prompt_id=1, prompt="Hi", text="Hello", kind="chat"))


@pytest.mark.parametrize("bad", [
    ev("nope"),
    ev("progress", **{k: v for k, v in GOOD_PROGRESS.items() if k != "loss"}),
    ev("progress", **{**GOOD_PROGRESS, "step": "3"}),
    ev("progress", **{**GOOD_PROGRESS, "step": True}),
    ev("progress", **{**GOOD_PROGRESS, "loss": float("nan")}),
    ev("progress", **GOOD_PROGRESS, surprise=1),
    ev("done", status="finished"),
    ev("error", code="oops", message="m"),
])
def test_invalid_events_are_refused(bad):
    with pytest.raises(ProtocolError):
        validate(bad)


def test_emitter_writes_one_json_object_per_line(tmp_path):
    buf = io.StringIO()
    em = Emitter("pretrain", out=buf)
    em.tee(tmp_path / "events.jsonl")
    em.emit("status", phase="pretrain", message="starting")
    em.emit("done", status="stopped")
    em.close()
    lines = buf.getvalue().splitlines()
    assert lines == (tmp_path / "events.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    assert first["v"] == 1 and first["cmd"] == "pretrain" and first["event"] == "status"
    with pytest.raises(ProtocolError):
        em.emit("status", phase="x")  # missing message: nothing written
    assert len(buf.getvalue().splitlines()) == 2


def test_protocol_doc_lists_every_event():
    from pathlib import Path

    doc = (Path(__file__).resolve().parents[2] / "docs" / "trainer-protocol.md").read_text()
    for name, fields in SCHEMA.items():
        assert f"`{name}`" in doc, f"docs/trainer-protocol.md does not describe the {name} event"
        for f in fields:
            assert f"`{f}`" in doc, f"docs/trainer-protocol.md does not describe {name}.{f}"
