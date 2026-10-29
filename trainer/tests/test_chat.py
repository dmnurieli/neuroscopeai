from neuroscope_trainer.chat import prompt_text, render, render_training_example


def test_template_matches_olmo2_instruct():
    text, spans = render([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}])
    assert text == "<|endoftext|><|user|>\nHi\n<|assistant|>\nHello!<|endoftext|>"
    assert [text[a:b] for a, b in spans] == ["Hello!<|endoftext|>"]


def test_multi_turn_and_system():
    msgs = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "A?"},
        {"role": "assistant", "content": "B."},
        {"role": "user", "content": "C?"},
        {"role": "assistant", "content": "D."},
    ]
    text, spans = render(msgs)
    assert text == ("<|endoftext|><|system|>\nBe brief.\n<|user|>\nA?\n<|assistant|>\nB.<|endoftext|>\n"
                    "<|user|>\nC?\n<|assistant|>\nD.<|endoftext|>")
    assert [text[a:b] for a, b in spans] == ["B.<|endoftext|>", "D.<|endoftext|>"]


def test_prompt_is_what_the_engine_renders():
    # engine/src/tokenize.rs, ChatFormat::Olmo2
    assert prompt_text("How do airplanes fly?") == "<|endoftext|><|user|>\nHow do airplanes fly?\n<|assistant|>\n"


def test_loss_mask_covers_only_the_reply(tiny_tokenizer):
    msgs = [{"role": "user", "content": "What is blue?"}, {"role": "assistant", "content": "The sea is blue."}]
    ids, mask = render_training_example(tiny_tokenizer, msgs)
    assert len(ids) == len(mask)
    eos = tiny_tokenizer.token_to_id("<|endoftext|>")
    assert ids[0] == eos and ids[-1] == eos and mask[0] == 0 and mask[-1] == 1
    reply = tiny_tokenizer.decode([i for i, m in zip(ids, mask) if m], skip_special_tokens=True)
    assert reply == "The sea is blue."
    prompt = tiny_tokenizer.decode([i for i, m in zip(ids, mask) if not m], skip_special_tokens=False)
    assert "What is blue?" in prompt and "sea" not in prompt
