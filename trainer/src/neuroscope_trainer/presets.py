"""Model size presets. Every preset is an OLMo 2 decoder with tied embeddings, head_dim 64 and the OLMo 2
tokenizer's 100,352-entry vocabulary, so the export opens in the app's engine unchanged."""

from dataclasses import asdict, dataclass

VOCAB_SIZE = 100_352  # OLMo 2 (embedding rows; the tokenizer itself uses ids < 100,278)
EOS_ID = 100_257  # <|endoftext|>
PAD_ID = 100_277  # <|pad|>


@dataclass(frozen=True)
class ModelShape:
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    vocab_size: int = VOCAB_SIZE
    tie_word_embeddings: bool = True
    rope_theta: float = 500_000.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 4096

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def non_embedding_params(self) -> int:
        d, ff = self.hidden_size, self.intermediate_size
        # q,k,v,o; gate, up, down; q_norm, k_norm, post_attention_layernorm, post_feedforward_layernorm
        per_layer = 4 * d * d + 3 * d * ff + 4 * d
        return self.num_layers * per_layer + d  # + final norm

    def embedding_params(self) -> int:
        return self.vocab_size * self.hidden_size * (1 if self.tie_word_embeddings else 2)

    def total_params(self) -> int:
        return self.non_embedding_params() + self.embedding_params()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Preset:
    name: str
    label: str  # what the learner reads
    shape: ModelShape
    peak_lr: float
    tokens_per_step: int  # effective batch (micro-batches are accumulated up to this)
    seq_len: int = 1024


PRESETS: dict[str, Preset] = {
    p.name: p
    for p in (
        Preset("tiny", "Tiny", ModelShape(256, 6, 4, 768), peak_lr=3e-3, tokens_per_step=8 * 1024),
        Preset("small", "Small", ModelShape(384, 8, 6, 1024), peak_lr=2e-3, tokens_per_step=16 * 1024),
        Preset("medium", "Medium", ModelShape(512, 16, 8, 1376), peak_lr=1e-3, tokens_per_step=16 * 1024),
    )
}
# Names used by the speed benchmark that chose these sizes.
ALIASES = {"T50t": "small", "N50t": "medium"}

# For unit tests only: a model small enough to train on any machine in a second.
TEST_SHAPE = ModelShape(64, 2, 2, 128, vocab_size=512, max_position_embeddings=256)


def get(name: str) -> Preset:
    name = ALIASES.get(name, name)
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r} (have: {', '.join(PRESETS)})")
    return PRESETS[name]
