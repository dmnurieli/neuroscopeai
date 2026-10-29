"""HF transformers reference for the engine's OLMo 2 path (CPU, float32, eager attention).

Usage: ref_olmo.py [model_dir] [prefix]   (defaults: the Instruct folder, "olmo")
Any OLMo 2 checkpoint works (base / early-training via model.safetensors.index.json, SFT / DPO via
pytorch_model.bin, RLVR1 / Instruct via model.safetensors). Checkpoints without a chat template (base,
early-training) are prompted with the plain text, as the engine's ChatFormat::Plain does.

Writes, beside this file (prefix "olmo" shown):
  olmo_fixtures.json  prompts, chat-template token ids, top-10 logits, greedy generations with per-step
                      logit margins, the edit specs, last-position MLP activations / head norms samples
  olmo_logits.bin     float32 little-endian, vocab_size floats per record, in the order of
                      fixtures["logit_records"] (full last-token logits; gitignored, ~1.2 MB per record)

Run:  scripts/reference/.venv/bin/python scripts/reference/ref_olmo.py [model_dir]
(the venv needs torch + transformers: uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python
 torch transformers safetensors numpy tokenizers jinja2). Then:
  cd engine && cargo test --release --test reference_olmo -- --ignored --nocapture --test-threads=1
"""
import json, os, struct, sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Library/Application Support/NeuroScope/models/OLMo-2-0425-1B-Instruct")
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "olmo"
PROMPTS = [
    "What is the capital of France?",
    "Describe how a bicycle works.",
    "Explain in one sentence why the sky is blue.",
]
GEN_TOKENS = 32
ACT_LAYERS = [0, 7, 15]          # layers whose last-position MLP activations are recorded (prompt 0)
NEURON_EDIT_LAYERS = [2, 7, 12]  # neuron ablation: the 16 strongest |activation| neurons of these layers
NEURON_EDIT_K = 16
HEAD_EDITS = [(3, 5), (9, 0), (14, 11)]  # (layer, head) ablated together


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    cfg = model.config
    L, nh = cfg.num_hidden_layers, cfg.num_attention_heads
    hd = cfg.hidden_size // nh
    layers = model.model.layers

    # ---- capture hooks: MLP activations (down_proj input) and per-head norms (o_proj input) ----
    acts, heads = {}, {}

    def cap_act(l):
        def f(_m, inp):
            acts[l] = inp[0][0].detach().clone()  # [S, N]
        return f

    def cap_head(l):
        def f(_m, inp):
            heads[l] = inp[0][0].detach().clone()  # [S, nh*hd]
        return f

    hooks = [layers[l].mlp.down_proj.register_forward_pre_hook(cap_act(l)) for l in range(L)]
    hooks += [layers[l].self_attn.o_proj.register_forward_pre_hook(cap_head(l)) for l in range(L)]

    records, rec_names = [], []
    out = {"model_dir": MODEL.replace(os.path.expanduser("~"), "~", 1), "vocab_size": cfg.vocab_size, "prompts": []}
    with torch.no_grad():
        for pi, prompt in enumerate(PROMPTS):
            if tok.chat_template:
                ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                              tokenize=True, return_dict=False)
                if hasattr(ids, "input_ids"):
                    ids = ids["input_ids"]
                rendered = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                   add_generation_prompt=True, tokenize=False)
            else:
                ids = tok(prompt)["input_ids"]
                rendered = prompt
            ids = [int(i) for i in ids]
            x = torch.tensor([ids])
            logits = model(x).logits[0, -1].float().numpy()
            records.append(logits)
            rec_names.append(f"prompt{pi}")
            top = np.argsort(-logits)[:10]
            entry = {"prompt": prompt, "rendered": rendered, "ids": ids,
                     "top10_ids": top.tolist(), "top10_logits": logits[top].tolist()}
            if pi == 0:
                entry["activations"] = {str(l): acts[l][-1].tolist() for l in ACT_LAYERS}
                entry["head_norms"] = [
                    [float(v) for v in heads[l][-1].reshape(nh, hd).norm(dim=-1)] for l in range(L)]
                last_acts = {l: acts[l][-1].clone() for l in range(L)}
            # greedy generation through HF's KV cache
            g = model.generate(x, max_new_tokens=GEN_TOKENS, do_sample=False, eos_token_id=cfg.eos_token_id,
                               pad_token_id=cfg.pad_token_id, return_dict_in_generate=True, output_logits=True,
                               use_cache=True)
            gen = g.sequences[0, len(ids):].tolist()
            margins = []
            for step in g.logits:
                v = torch.topk(step[0].float(), 2).values
                margins.append(float(v[0] - v[1]))
            entry["greedy"] = gen
            entry["greedy_margins"] = margins
            entry["greedy_text"] = tok.decode(gen)
            out["prompts"].append(entry)
            print(f"[{pi}] {len(ids)} prompt tokens; top1 {int(top[0])} {tok.decode([int(top[0])])!r}; "
                  f"greedy: {tok.decode(gen)!r}")
        for h in hooks:
            h.remove()

        ids0 = torch.tensor([out["prompts"][0]["ids"]])

        # ---- neuron ablation: zero the chosen SwiGLU units (the down_proj input) at every position ----
        neuron_edits = []
        for l in NEURON_EDIT_LAYERS:
            idx = torch.topk(last_acts[l].abs(), NEURON_EDIT_K).indices.tolist()
            neuron_edits += [{"layer": l, "index": int(i), "mode": "ablate", "value": 0.0} for i in idx]
        by_layer = {}
        for e in neuron_edits:
            by_layer.setdefault(e["layer"], []).append(e["index"])

        def zero_units(cols):
            def f(_m, inp):
                h = inp[0].clone()
                h[..., cols] = 0.0
                return (h,)
            return f

        hs = [layers[l].mlp.down_proj.register_forward_pre_hook(zero_units(c)) for l, c in by_layer.items()]
        abl = model(ids0).logits[0, -1].float().numpy()
        for h in hs:
            h.remove()
        records.append(abl)
        rec_names.append("prompt0_neuron_ablation")

        # ---- head ablation: zero the head's slice of the o_proj input ----
        def zero_head(hh):
            def f(_m, inp):
                a = inp[0].clone()
                a[..., hh * hd:(hh + 1) * hd] = 0.0
                return (a,)
            return f

        hs = [layers[l].self_attn.o_proj.register_forward_pre_hook(zero_head(hh)) for l, hh in HEAD_EDITS]
        habl = model(ids0).logits[0, -1].float().numpy()
        for h in hs:
            h.remove()
        records.append(habl)
        rec_names.append("prompt0_head_ablation")

    base = records[0]
    for name, r in [("neuron", abl), ("head", habl)]:
        d = r - base
        print(f"{name} ablation: |delta| {np.linalg.norm(d):.3f} (|logits| {np.linalg.norm(base):.3f}); "
              f"top1 {int(np.argmax(base))} -> {int(np.argmax(r))}")
    out["neuron_edits"] = neuron_edits
    out["head_edits"] = [{"layer": l, "head": hh, "mode": "ablate", "value": 0.0} for l, hh in HEAD_EDITS]
    out["logit_records"] = rec_names
    out["transformers"] = __import__("transformers").__version__
    out["torch"] = torch.__version__
    with open(os.path.join(HERE, f"{PREFIX}_logits.bin"), "wb") as f:
        for r in records:
            assert r.shape == (cfg.vocab_size,)
            f.write(r.astype("<f4").tobytes())
    with open(os.path.join(HERE, f"{PREFIX}_fixtures.json"), "w") as f:
        json.dump(out, f)
    print(f"wrote {PREFIX}_fixtures.json + {PREFIX}_logits.bin")


main()
