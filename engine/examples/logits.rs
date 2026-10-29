//! Dumps the engine's next-token logits for a prompt and a few greedy steps after it, so another implementation
//! (the MLX trainer's `compare` subcommand) can check them position by position.
//!
//!   cargo run --release --example logits -- --model DIR [--steps 4] --out out.json "prompt"
//!
//! The prompt is rendered the way the app renders it (plain text for a pretrained model, OLMo 2's chat template
//! when tokenizer_config.json has one). Writes out.json (token ids, positions, dtype) and out.bin: f32
//! little-endian logits, one row of vocab_size per position: the prompt's last token, then each greedy step
//! (decoded through the KV cache, as in generation). NS_DTYPE=f32 runs the engine in full precision.

use neuroscope_engine::Engine;
use neuroscope_engine::edits::DeviceEdits;
use std::io::Write;
use std::path::PathBuf;

fn die(m: &str) -> ! {
    eprintln!("error: {m}");
    std::process::exit(2)
}

fn main() -> anyhow::Result<()> {
    let mut model: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut steps = 4usize;
    let mut prompt: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |name: &str| args.next().unwrap_or_else(|| die(&format!("{name} needs a value")));
        match a.as_str() {
            "--model" => model = Some(PathBuf::from(val("--model"))),
            "--out" => out = Some(PathBuf::from(val("--out"))),
            "--steps" => steps = val("--steps").parse().unwrap_or_else(|_| die("bad --steps")),
            _ => prompt.push(a),
        }
    }
    let usage = "usage: logits --model DIR --out OUT.json [--steps N] \"prompt\"";
    let model = model.unwrap_or_else(|| die(usage));
    let out = out.unwrap_or_else(|| die(usage));
    let prompt = prompt.join(" ");
    if prompt.is_empty() {
        die(usage);
    }

    let mut engine = Engine::open(&model)?;
    let ids = engine.tok.encode_chat(&prompt, false)?;
    let edits = DeviceEdits::none(engine.geometry.num_layers);
    let (mut logits, _) = engine.model.forward(&ids, 0, &edits, false)?;
    let mut rows = vec![logits.clone()];
    let mut all = ids.clone();
    let mut positions = vec![ids.len() - 1];
    for _ in 0..steps {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        let pos = all.len();
        all.push(next);
        let (lg, _) = engine.model.forward(&[next], pos, &edits, false)?;
        logits = lg;
        rows.push(logits.clone());
        positions.push(pos);
    }
    let bin = out.with_extension("bin");
    let mut f = std::fs::File::create(&bin)?;
    for r in &rows {
        for v in r {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    let dtype = std::env::var("NS_DTYPE").unwrap_or_else(|_| "bf16".into());
    let info = serde_json::json!({
        "model": model.display().to_string(),
        "prompt": prompt,
        "rendered": engine.tok.render(&prompt, false),
        "ids": all,
        "prompt_len": ids.len(),
        "positions": positions,
        "vocab_size": engine.geometry.vocab_size,
        "dtype": dtype,
        "logits_file": bin.file_name().unwrap().to_string_lossy(),
        "greedy_text": engine.tok.inner.decode(&all[ids.len()..], false).map_err(anyhow::Error::msg)?,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&info)?)?;
    println!("{} positions x {} logits -> {}", rows.len(), engine.geometry.vocab_size, bin.display());
    Ok(())
}
