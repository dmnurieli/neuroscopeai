//! stage-record: runs the graduation lesson's fixed prompts on ONE OLMo 2 1B stage and appends one JSON line per
//! prompt to an output file. Greedy, bf16 on Metal (the app's settings), fixed max tokens per prompt.
//!
//!   stage-record --model DIR --stage ID --label TEXT --repo allenai/… --branch main --commit SHA
//!                --set snapshot|base|chat --tokens N --recorded-at ISO --out FILE.jsonl
//!
//! `--tokens` is how many word pieces of training the stage had seen (0 for post-trained stages; informational).

use neuroscope_engine::generate::generate;
use neuroscope_engine::tokenize::ChatFormat;
use neuroscope_engine::*;
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

/// (id, prompt, max new tokens). Plain-text prompts are continued like a document; chat prompts go through
/// the stage's own chat template (the engine picks it from tokenizer_config.json).
const BLANK: (&str, &str, usize) = ("blank", "The pilot grabbed the controls and landed the airplane safely on the", 24);
const CAPITAL: (&str, &str, usize) = ("capital", "The capital of France is", 24);
const CAKE_PLAIN: (&str, &str, usize) = ("cake", "How do I bake a chocolate cake?", 120);
const CAKE_CHAT: (&str, &str, usize) = ("cake", "How do I bake a chocolate cake?", 320);
const MATH: (&str, &str, usize) =
    ("math", "A flight leaves at 3:40 pm and the trip takes 2 hours and 35 minutes. What time does it land?", 200);
const PREMISE: (&str, &str, usize) =
    ("premise", "Why is the Great Wall of China the only man-made thing you can see from the Moon?", 200);

fn die(m: &str) -> ! {
    eprintln!("error: {m}");
    std::process::exit(2)
}

fn main() -> anyhow::Result<()> {
    let mut model = None;
    let (mut stage, mut label, mut repo, mut branch, mut commit, mut set, mut at, mut out) =
        (String::new(), String::new(), String::new(), String::new(), String::new(), String::new(), String::new(), None);
    let mut tokens: u64 = 0;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().unwrap_or_else(|| die(&format!("{a} needs a value")));
        match a.as_str() {
            "--model" => model = Some(PathBuf::from(v())),
            "--stage" => stage = v(),
            "--label" => label = v(),
            "--repo" => repo = v(),
            "--branch" => branch = v(),
            "--commit" => commit = v(),
            "--set" => set = v(),
            "--tokens" => tokens = v().parse().unwrap_or_else(|_| die("bad --tokens")),
            "--recorded-at" => at = v(),
            "--out" => out = Some(PathBuf::from(v())),
            other => die(&format!("unknown argument {other}")),
        }
    }
    let model = model.unwrap_or_else(|| die("--model is required"));
    let out = out.unwrap_or_else(|| die("--out is required"));
    if stage.is_empty() || repo.is_empty() || commit.is_empty() || at.is_empty() {
        die("--stage, --repo, --commit and --recorded-at are required");
    }
    let prompts: Vec<(&str, &str, usize)> = match set.as_str() {
        "snapshot" => vec![BLANK, CAPITAL, CAKE_PLAIN],
        "base" => vec![BLANK, CAPITAL, CAKE_PLAIN, MATH, PREMISE],
        "chat" => vec![CAKE_CHAT, MATH, PREMISE, BLANK],
        _ => die("--set must be snapshot, base or chat"),
    };

    let mut engine = Engine::open(&model)?;
    let format = match engine.tok.format {
        ChatFormat::Plain => "plain",
        ChatFormat::Olmo2 => "olmo2-chat",
        ChatFormat::Qwen3 => "qwen3-chat",
    };
    let device = format!("{} {:?}", if engine.model.device().is_metal() { "Metal" } else { "CPU" }, engine.model.dtype()).to_lowercase();
    let geo = engine.geometry;
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&out)?;
    let cancel = AtomicBool::new(false);

    for (id, prompt, max) in prompts {
        let req = Request {
            prompt: prompt.to_string(),
            thinking: false,
            max_tokens: max,
            logit_lens: false,
            top_k: 8,
            pace_ms: 0,
            sampling: Sampling::Greedy,
            edits: EditSet::default(),
        };
        let mut text = String::new();
        let mut first_top: Vec<(u32, f32)> = Vec::new();
        let mut prompt_tokens = 0usize;
        let summary = generate(&mut engine, &req, &cancel, &mut |f: &Frame| match f.stream {
            StreamKind::Prompt => {
                prompt_tokens += 1;
                if !f.top.is_empty() {
                    first_top = f.top.clone();
                }
            }
            StreamKind::Answer | StreamKind::Thinking => text.push_str(&f.text),
            StreamKind::Special => {}
        })?;
        let top: Vec<_> = first_top
            .iter()
            .map(|(t, p)| json!({"token": engine.tok.token_text(*t), "prob": p}))
            .collect();
        let line = json!({
            "stage": stage, "label": label, "repo": repo, "branch": branch, "commit": commit,
            "trained_tokens": tokens, "prompt_id": id, "prompt": prompt, "format": format,
            "output": text, "max_tokens": max, "tokens_generated": summary.tokens_generated,
            "stop": format!("{:?}", summary.stop).to_lowercase(), "prompt_tokens": prompt_tokens,
            "tok_per_s": summary.decode_tok_per_s, "prefill_ms": summary.prefill_ms,
            "next_top": top, "sampling": "greedy", "device": device, "recorded_at": at,
            "vocab_slots": geo.vocab_size, "vocab_entries": engine.tok.inner.get_vocab_size(true),
        });
        writeln!(file, "{line}")?;
        eprintln!("[{stage}/{id}] {:.1} tok/s, {} tokens, stop {:?}\n{}\n", summary.decode_tok_per_s,
                  summary.tokens_generated, summary.stop, text);
    }
    Ok(())
}
