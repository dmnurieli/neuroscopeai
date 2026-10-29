"""`export`: write a trained model in Hugging Face's OLMo 2 layout — the layout the app's engine and
`transformers` both open.

    config.json             model_type olmo2, Olmo2ForCausalLM, tie_word_embeddings true
    model.safetensors       HF weight names (bf16 by default); tied, so no lm_head.weight
    generation_config.json
    tokenizer.json, tokenizer_config.json, special_tokens_map.json
                            tokenizer_config.json carries OLMo 2's chat template only after instruction tuning, so the
                            engine renders prompts as chat for an SFT model and as plain text for a pretrained one
    training.json           what was trained, on what, for how long
    ATTRIBUTION.md          data and license notes that travel with the weights
"""

import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .model import Olmo2
from .presets import EOS_ID, PAD_ID, ModelShape
from .train import save_safetensors


def expected_hf_names(shape: ModelShape) -> dict[str, tuple[int, ...]]:
    """Every tensor Olmo2ForCausalLM (and engine/src/model.rs) reads, with its shape, written out independently
    of the MLX module tree so a renamed MLX attribute cannot pass silently."""
    d, ff, v = shape.hidden_size, shape.intermediate_size, shape.vocab_size
    qd = shape.num_heads * shape.head_dim
    names = {"model.embed_tokens.weight": (v, d), "model.norm.weight": (d,)}
    if not shape.tie_word_embeddings:
        names["lm_head.weight"] = (v, d)
    for i in range(shape.num_layers):
        p = f"model.layers.{i}"
        names.update({
            f"{p}.self_attn.q_proj.weight": (qd, d),
            f"{p}.self_attn.k_proj.weight": (qd, d),
            f"{p}.self_attn.v_proj.weight": (qd, d),
            f"{p}.self_attn.o_proj.weight": (d, qd),
            f"{p}.self_attn.q_norm.weight": (qd,),
            f"{p}.self_attn.k_norm.weight": (qd,),
            f"{p}.mlp.gate_proj.weight": (ff, d),
            f"{p}.mlp.up_proj.weight": (ff, d),
            f"{p}.mlp.down_proj.weight": (d, ff),
            f"{p}.post_attention_layernorm.weight": (d,),
            f"{p}.post_feedforward_layernorm.weight": (d,),
        })
    return names


def hf_name(mlx_name: str) -> str:
    """The MLX module tree mirrors Olmo2ForCausalLM, so a parameter's path is its HF name."""
    return mlx_name


def hf_state_dict(model: Olmo2, dtype=mx.bfloat16) -> dict[str, mx.array]:
    out = {hf_name(k): v.astype(dtype) for k, v in tree_flatten(model.parameters())}
    want = expected_hf_names(model.shape)
    missing, extra = set(want) - set(out), set(out) - set(want)
    if missing or extra:
        raise ValueError(f"export names do not match OLMo 2: missing {sorted(missing)[:5]}, unexpected {sorted(extra)[:5]}")
    for k, s in want.items():
        if tuple(out[k].shape) != s:
            raise ValueError(f"{k}: shape {tuple(out[k].shape)}, expected {s}")
    return out


def hf_config(shape: ModelShape, dtype_name: str) -> dict:
    return {
        "architectures": ["Olmo2ForCausalLM"],
        "model_type": "olmo2",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "eos_token_id": EOS_ID,
        "pad_token_id": PAD_ID,
        "hidden_act": "silu",
        "hidden_size": shape.hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": shape.intermediate_size,
        "max_position_embeddings": shape.max_position_embeddings,
        "num_attention_heads": shape.num_heads,
        "num_hidden_layers": shape.num_layers,
        "num_key_value_heads": shape.num_heads,
        "rms_norm_eps": shape.rms_norm_eps,
        "rope_scaling": None,
        "rope_theta": shape.rope_theta,
        "tie_word_embeddings": shape.tie_word_embeddings,
        "torch_dtype": dtype_name,
        "use_cache": True,
        "vocab_size": shape.vocab_size,
    }


ATTRIBUTION = """# Attribution

This model was trained from scratch on this computer with the NeuroScope trainer. It is an OLMo 2 style
decoder ({params:,} parameters: {preset} preset, {tokens:,} pretraining tokens{sft_note}).

## What it learned from

- **Pretraining text: the "wiki" slice of OLMo-mix-1124** by the Allen Institute for AI
  (`{wiki_repo}`, file `{wiki_file}`, revision `{wiki_rev}`): English Wikipedia and Wikibooks
  articles. OLMo-mix-1124 is released under the Open Data Commons Attribution License (ODC-By) v1.0. The
  articles themselves are written by Wikipedia's and Wikibooks' contributors and licensed under the Creative
  Commons Attribution-ShareAlike license (CC BY-SA).
{sft_block}
- **Tokenizer and chat template: OLMo 2** (`{tok_repo}`, `{instruct_repo}`), Apache License 2.0.

## About sharing these weights

Whether a model trained on CC BY-SA text counts as an adaptation of that text, and so must itself be shared
alike, is an open legal question. If you share this model, the cautious course is to credit the sources above,
keep this file with the weights, and offer the weights under compatible, open terms. This note is not legal
advice.

The model can repeat or garble facts from its training text. It is a teaching model, not a source of truth.
"""

SFT_BLOCK = """- **Instruction tuning: a filtered subset of the Tulu 3 SFT mixture** by the Allen Institute for AI
  (`{tulu_repo}`, file `{tulu_file}`, row groups {row_groups}, revision `{tulu_rev}`): {examples:,} English
  conversations from two of its sources, OpenAssistant oasst1 (Apache 2.0, written by volunteers) and the FLAN v2
  collection (Apache 2.0; its tasks come from many datasets with their own terms). The mixture is released under
  ODC-By v1.0 and each source keeps its own license. Conversations that were not English, or longer than
  {max_tokens} tokens, were left out.
"""


def write_export(out: Path, model: Olmo2, tokenizer_dir: Path, *, chat: bool, training: dict, dtype: str = "bfloat16") -> list[str]:
    from . import sources

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    mdtype = {"bfloat16": mx.bfloat16, "float32": mx.float32}[dtype]
    save_safetensors(out / "model.safetensors", hf_state_dict(model, mdtype), {"format": "pt"})
    (out / "config.json").write_text(json.dumps(hf_config(model.shape, dtype), indent=2) + "\n")
    (out / "generation_config.json").write_text(
        json.dumps({"_from_model_config": True, "eos_token_id": EOS_ID, "pad_token_id": PAD_ID}, indent=2) + "\n")
    shutil.copy(tokenizer_dir / "tokenizer.json", out / "tokenizer.json")
    shutil.copy(tokenizer_dir / "special_tokens_map.json", out / "special_tokens_map.json")
    tc = json.loads((tokenizer_dir / "tokenizer_config.json").read_text())
    if not chat:
        tc.pop("chat_template", None)
    (out / "tokenizer_config.json").write_text(json.dumps(tc, indent=2, ensure_ascii=False) + "\n")
    (out / "training.json").write_text(json.dumps(training, indent=1, ensure_ascii=False) + "\n")

    sft = training.get("sft")
    sft_block = ""
    if sft:
        src = sft["data"]["source"]
        sft_block = SFT_BLOCK.format(tulu_repo=src["repo"], tulu_file=src["file"], row_groups=_ranges(src["row_groups"]),
                                     tulu_rev=src["revision"], examples=sft["data"]["examples"],
                                     max_tokens=sft["data"]["filters"]["max_tokens"])
    (out / "ATTRIBUTION.md").write_text(ATTRIBUTION.format(
        params=model.shape.total_params(), preset=training.get("preset", "?"), tokens=training.get("pretrain_tokens", 0),
        sft_note=f", then {sft['tokens']:,} instruction-tuning tokens" if sft else "",
        wiki_repo=sources.WIKI_REPO, wiki_file=sources.WIKI_FILE, wiki_rev=sources.WIKI_REVISION,
        sft_block=sft_block, tok_repo=sources.TOKENIZER_REPO, instruct_repo=sources.INSTRUCT_REPO))
    return sorted(p.name for p in out.iterdir())


def _ranges(xs: list[int]) -> str:
    xs = sorted(xs)
    out, start, prev = [], xs[0], xs[0]
    for x in xs[1:] + [None]:
        if x is not None and x == prev + 1:
            prev = x
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        if x is not None:
            start = prev = x
    return ", ".join(out)


def load_export(folder: Path) -> Olmo2:
    """Reads an exported folder back into MLX (f32), checking names and shapes on the way."""
    cfg = json.loads((Path(folder) / "config.json").read_text())
    shape = ModelShape(
        hidden_size=cfg["hidden_size"], num_layers=cfg["num_hidden_layers"], num_heads=cfg["num_attention_heads"],
        intermediate_size=cfg["intermediate_size"], vocab_size=cfg["vocab_size"], tie_word_embeddings=cfg["tie_word_embeddings"],
        rope_theta=cfg["rope_theta"], rms_norm_eps=cfg["rms_norm_eps"], max_position_embeddings=cfg["max_position_embeddings"])
    w = mx.load(str(Path(folder) / "model.safetensors"))
    want = expected_hf_names(shape)
    if set(w) != set(want):
        raise ValueError(f"{folder}: weight names differ from OLMo 2's: {sorted(set(w) ^ set(want))[:5]}")
    m = Olmo2(shape)
    from mlx.utils import tree_unflatten

    m.update(tree_unflatten([(k, v.astype(mx.float32)) for k, v in w.items()]))
    mx.eval(m.parameters())
    return m
