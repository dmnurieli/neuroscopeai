import json
import shutil

import mlx.core as mx
import numpy as np
import pytest

from neuroscope_trainer.export import expected_hf_names, hf_config, hf_state_dict, load_export, write_export
from neuroscope_trainer.model import Olmo2, init_weights
from neuroscope_trainer.presets import PRESETS, TEST_SHAPE, ModelShape


def test_names_match_olmo2_for_every_preset():
    for p in PRESETS.values():
        m = Olmo2(p.shape)
        sd = hf_state_dict(m)
        assert set(sd) == set(expected_hf_names(p.shape))
        assert "lm_head.weight" not in sd  # tied
        assert sum(v.size for v in sd.values()) == p.shape.total_params()


def test_names_spot_check():
    names = expected_hf_names(TEST_SHAPE)
    for n in [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.self_attn.k_norm.weight",
        "model.layers.1.mlp.down_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.post_feedforward_layernorm.weight",
    ]:
        assert n in names
    assert not any("input_layernorm" in n for n in names)  # OLMo 2 has no pre-norms
    assert names["model.layers.0.self_attn.q_norm.weight"] == (TEST_SHAPE.hidden_size,)  # full projection, not per head
    assert names["model.layers.0.mlp.down_proj.weight"] == (TEST_SHAPE.hidden_size, TEST_SHAPE.intermediate_size)


def test_untied_shape_exports_lm_head():
    s = ModelShape(64, 1, 2, 128, vocab_size=512, tie_word_embeddings=False)
    sd = hf_state_dict(Olmo2(s))
    assert sd["lm_head.weight"].shape == (512, 64)


def test_config_is_olmo2():
    c = hf_config(PRESETS["medium"].shape, "bfloat16")
    assert c["model_type"] == "olmo2" and c["architectures"] == ["Olmo2ForCausalLM"]
    assert c["tie_word_embeddings"] is True and c["vocab_size"] == 100_352
    assert c["num_key_value_heads"] == c["num_attention_heads"] == 8
    assert c["hidden_size"] // c["num_attention_heads"] == 64
    assert c["attention_bias"] is False and c["hidden_act"] == "silu" and c["rope_scaling"] is None


@pytest.fixture
def tok_dir(tmp_path, tiny_tokenizer):
    d = tmp_path / "tok"
    d.mkdir()
    tiny_tokenizer.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ x }}", "eos_token": "<|endoftext|>"}))
    (d / "special_tokens_map.json").write_text("{}")
    return d


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_export_round_trip(tmp_path, tok_dir, dtype):
    m = Olmo2(TEST_SHAPE)
    init_weights(m, seed=3)
    out = tmp_path / "out"
    files = write_export(out, m, tok_dir, chat=False, training={"preset": "test", "pretrain_tokens": 10}, dtype=dtype)
    assert {"config.json", "model.safetensors", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "training.json", "ATTRIBUTION.md"} <= set(files)
    assert "chat_template" not in json.loads((out / "tokenizer_config.json").read_text())
    att = (out / "ATTRIBUTION.md").read_text()
    assert "CC BY-SA" in att and "ODC-By" in att and "open legal question" in att
    w = mx.load(str(out / "model.safetensors"))
    assert all(v.dtype == (mx.float32 if dtype == "float32" else mx.bfloat16) for v in w.values())
    back = load_export(out)
    ids = mx.array([[1, 5, 9, 2, 7]])
    if dtype == "bfloat16":
        m.update(back.parameters())  # compare like for like: same rounded weights
    a, b = np.array(m(ids)), np.array(back(ids))
    assert np.abs(a - b).max() < 1e-5


def test_sft_export_keeps_chat_template(tmp_path, tok_dir):
    m = Olmo2(TEST_SHAPE)
    sft = {"steps": 3, "tokens": 100, "data": {"source": {"repo": "r", "file": "f", "row_groups": [0, 1, 2, 5], "revision": "x"},
                                              "examples": 12, "filters": {"max_tokens": 768}}}
    write_export(tmp_path / "o", m, tok_dir, chat=True, training={"preset": "t", "pretrain_tokens": 1, "sft": sft})
    assert json.loads((tmp_path / "o" / "tokenizer_config.json").read_text())["chat_template"] == "{{ x }}"
    assert "0-2, 5" in (tmp_path / "o" / "ATTRIBUTION.md").read_text()
