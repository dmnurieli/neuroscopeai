"""OLMo 2 decoder in MLX, with parameter names identical to Hugging Face's Olmo2ForCausalLM.

Per layer (post-norm on each sublayer's OUTPUT, no pre-norms):
    x = x + post_attention_layernorm(o_proj(attn(x)))
    x = x + post_feedforward_layernorm(down_proj(silu(gate_proj(x)) * up_proj(x)))
q_norm / k_norm are RMSNorms over the whole projection (heads * head_dim), applied before RoPE (rotate-half,
i.e. `traditional=False`). With tied embeddings the unembedding is `model.embed_tokens.weight`.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from .presets import ModelShape


class Attention(nn.Module):
    def __init__(self, s: ModelShape):
        super().__init__()
        d, h = s.hidden_size, s.num_heads
        self.n_heads, self.head_dim, self.theta = h, s.head_dim, s.rope_theta
        self.q_proj = nn.Linear(d, h * s.head_dim, bias=False)
        self.k_proj = nn.Linear(d, h * s.head_dim, bias=False)
        self.v_proj = nn.Linear(d, h * s.head_dim, bias=False)
        self.o_proj = nn.Linear(h * s.head_dim, d, bias=False)
        self.q_norm = nn.RMSNorm(h * s.head_dim, eps=s.rms_norm_eps)
        self.k_norm = nn.RMSNorm(h * s.head_dim, eps=s.rms_norm_eps)

    def __call__(self, x):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x))
        k = self.k_norm(self.k_proj(x))
        v = self.v_proj(x)
        q, k, v = (t.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3) for t in (q, k, v))
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.theta, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.theta, scale=1.0, offset=0)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask="causal")
        return self.o_proj(a.transpose(0, 2, 1, 3).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, s: ModelShape):
        super().__init__()
        self.gate_proj = nn.Linear(s.hidden_size, s.intermediate_size, bias=False)
        self.up_proj = nn.Linear(s.hidden_size, s.intermediate_size, bias=False)
        self.down_proj = nn.Linear(s.intermediate_size, s.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, s: ModelShape):
        super().__init__()
        self.self_attn = Attention(s)
        self.mlp = MLP(s)
        self.post_attention_layernorm = nn.RMSNorm(s.hidden_size, eps=s.rms_norm_eps)
        self.post_feedforward_layernorm = nn.RMSNorm(s.hidden_size, eps=s.rms_norm_eps)

    def __call__(self, x):
        x = x + self.post_attention_layernorm(self.self_attn(x))
        return x + self.post_feedforward_layernorm(self.mlp(x))


class Decoder(nn.Module):
    def __init__(self, s: ModelShape):
        super().__init__()
        self.embed_tokens = nn.Embedding(s.vocab_size, s.hidden_size)
        self.layers = [Layer(s) for _ in range(s.num_layers)]
        self.norm = nn.RMSNorm(s.hidden_size, eps=s.rms_norm_eps)


class Olmo2(nn.Module):
    def __init__(self, shape: ModelShape):
        super().__init__()
        self.shape = shape
        self.model = Decoder(shape)
        if not shape.tie_word_embeddings:
            self.lm_head = nn.Linear(shape.hidden_size, shape.vocab_size, bias=False)

    def __call__(self, ids):
        x = self.model.embed_tokens(ids)
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        if self.shape.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(x)
        return self.lm_head(x)


def init_weights(model: Olmo2, seed: int, std: float = 0.02) -> None:
    """OLMo 2's initialization: every matrix ~ N(0, 0.02) truncated at 3 std; norm weights 1."""
    key = mx.random.key(seed)
    flat = tree_flatten(model.parameters())
    keys = mx.random.split(key, len(flat))
    out = []
    for (name, p), k in zip(flat, keys):
        if p.ndim == 2:
            p = mx.random.truncated_normal(-3.0, 3.0, p.shape, key=k) * std
        out.append((name, p))
    model.update(tree_unflatten(out))


def cast_view(model: Olmo2, dtype):
    """Swaps a copy of the parameters in `dtype` into the model; returns the originals for restoring."""
    orig = model.parameters()
    model.update(tree_map(lambda p: p.astype(dtype), orig))
    return orig


def param_count(model: Olmo2) -> int:
    return sum(v.size for _, v in tree_flatten(model.parameters()))
