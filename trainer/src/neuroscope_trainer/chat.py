"""OLMo 2's chat template (the Instruct model's tokenizer_config.json), rendered by hand so training text and the
app's engine (engine/src/tokenize.rs, ChatFormat::Olmo2) agree byte for byte:

    <|endoftext|><|user|>\\n{user}\\n<|assistant|>\\n{reply}<|endoftext|>

with `\\n` between turns of a longer conversation and `<|system|>\\n{system}\\n` when there is a system message.
`<|user|>` and `<|assistant|>` are ordinary text (BPE pieces), not added tokens; `<|endoftext|>` is the added token
100257. The whole conversation is encoded as one string (so BPE sees what it will see at inference), and the loss
mask is set from character offsets: 1 on the assistant's words and the `<|endoftext|>` that ends each reply.
"""

from tokenizers import Tokenizer

BOS = EOS = "<|endoftext|>"


def render(messages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Returns the text and the [start, end) character spans the loss is taken on."""
    text, spans = BOS, []
    for i, m in enumerate(messages):
        role, content = m["role"], m["content"]
        last = i == len(messages) - 1
        if role == "system":
            text += "<|system|>\n" + content + "\n"
        elif role == "user":
            text += "<|user|>\n" + content + "\n"
        elif role == "assistant":
            text += "<|assistant|>\n"
            start = len(text)
            text += content + EOS
            spans.append((start, len(text)))
            if not last:
                text += "\n"
        else:
            raise ValueError(f"unknown role {role!r}")
    return text, spans


def prompt_text(user: str) -> str:
    """One user turn with the generation prompt (what the app's engine renders for an Instruct model)."""
    return f"{BOS}<|user|>\n{user}\n<|assistant|>\n"


def render_training_example(tok: Tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    text, spans = render(messages)
    enc = tok.encode(text, add_special_tokens=False)
    mask = []
    for a, b in enc.offsets:
        mask.append(int(any(s <= a and b <= e and b > a for s, e in spans)))
    return list(enc.ids), mask
