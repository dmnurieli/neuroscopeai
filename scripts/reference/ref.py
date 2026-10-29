"""Independent float32 numpy Qwen3 forward (HF modeling_qwen3 semantics) -> fixtures.json.
Run: .venv/bin/python ref.py   (reads the model dir, writes fixtures.json beside this file)."""
import json, os, struct, sys
import numpy as np

MODEL = os.path.expanduser("~/Library/Application Support/NeuroScope/models/Qwen3-0.6B")
PROMPT = "The capital of France is"
HERE = os.path.dirname(os.path.abspath(__file__))

def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        buf = np.memmap(path, dtype=np.uint8, mode="r", offset=base)
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        s, e = v["data_offsets"]
        raw = np.asarray(buf[s:e])
        if v["dtype"] == "BF16":
            a = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif v["dtype"] == "F32":
            a = raw.view(np.float32)
        elif v["dtype"] == "F16":
            a = raw.view(np.float16).astype(np.float32)
        else:
            raise ValueError(v["dtype"])
        out[k] = a.reshape(v["shape"])
    return out

def rms(x, w, eps):
    return w * (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps))

def silu(x):
    return x / (1.0 + np.exp(-x))

def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], -1)

def main():
    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    from tokenizers import Tokenizer
    ids = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json")).encode(PROMPT).ids
    W = load_safetensors(os.path.join(MODEL, "model.safetensors"))
    L, nh, nkv, hd = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]
    S = len(ids)
    x = W["model.embed_tokens.weight"][ids]  # [S, H]
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    fr = np.outer(np.arange(S), inv)
    emb = np.concatenate([fr, fr], -1)
    cos, sin = np.cos(emb).astype(np.float32)[:, None, :], np.sin(emb).astype(np.float32)[:, None, :]
    mask = np.triu(np.full((S, S), -np.inf, dtype=np.float32), 1)
    acts = {}
    for l in range(L):
        p = f"model.layers.{l}."
        h = rms(x, W[p + "input_layernorm.weight"], eps)
        q = (h @ W[p + "self_attn.q_proj.weight"].T).reshape(S, nh, hd)
        k = (h @ W[p + "self_attn.k_proj.weight"].T).reshape(S, nkv, hd)
        v = (h @ W[p + "self_attn.v_proj.weight"].T).reshape(S, nkv, hd)
        q = rms(q, W[p + "self_attn.q_norm.weight"], eps)
        k = rms(k, W[p + "self_attn.k_norm.weight"], eps)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        k = np.repeat(k, nh // nkv, axis=1)
        v = np.repeat(v, nh // nkv, axis=1)
        sc = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(hd) + mask
        sc = np.exp(sc - sc.max(-1, keepdims=True))
        sc /= sc.sum(-1, keepdims=True)
        o = np.einsum("hqk,khd->qhd", sc, v).reshape(S, nh * hd)
        x = x + o @ W[p + "self_attn.o_proj.weight"].T
        h = rms(x, W[p + "post_attention_layernorm.weight"], eps)
        a = silu(h @ W[p + "mlp.gate_proj.weight"].T) * (h @ W[p + "mlp.up_proj.weight"].T)
        if l in (0, 13, 27):
            acts[str(l)] = a[-1].tolist()
        x = x + a @ W[p + "mlp.down_proj.weight"].T
    x = rms(x, W["model.norm.weight"], eps)
    lm = W.get("lm_head.weight", W["model.embed_tokens.weight"])
    logits = x[-1] @ lm.T
    top = np.argsort(-logits)[:10]
    fx = {"prompt": PROMPT, "ids": ids, "top10_ids": top.tolist(),
          "top10_logits": logits[top].tolist(), "activations": acts}
    json.dump(fx, open(os.path.join(HERE, "fixtures.json"), "w"))
    tk = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
    print("ids", ids)
    for i, t in zip(top, logits[top]):
        print(int(i), repr(tk.decode([int(i)])), round(float(t), 4))

main()
