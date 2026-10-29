use neuroscope_engine::generate::generate;
use neuroscope_engine::*;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

fn parse_f(s: &str) -> f32 {
    s.parse().unwrap_or_else(|_| die(&format!("bad number: {s}")))
}
fn parse_u(s: &str) -> usize {
    s.parse().unwrap_or_else(|_| die(&format!("bad integer: {s}")))
}
fn die(m: &str) -> ! {
    eprintln!("error: {m}");
    std::process::exit(2)
}

fn main() -> anyhow::Result<()> {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut model = PathBuf::from(home).join("Library/Application Support/NeuroScope/models/OLMo-2-0425-1B-Instruct");
    let mut req = Request {
        prompt: String::new(),
        thinking: true,
        max_tokens: 512,
        logit_lens: true,
        top_k: 8,
        pace_ms: 0,
        sampling: Sampling::Greedy,
        edits: EditSet::default(),
    };
    let mut prompt: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |name: &str| args.next().unwrap_or_else(|| die(&format!("{name} needs a value")));
        match a.as_str() {
            "--model" => model = PathBuf::from(val("--model")),
            "--no-think" => req.thinking = false,
            "--no-lens" => req.logit_lens = false,
            "--max" => req.max_tokens = parse_u(&val("--max")),
            "--sample" => {
                let seed = val("--sample").parse().unwrap_or_else(|_| die("bad seed"));
                req.sampling = Sampling::Sample { temperature: 0.6, top_p: 0.95, top_k: 20, seed };
            }
            "--ablate" | "--scale" | "--pin" => {
                let v = val(&a);
                let p: Vec<&str> = v.split(':').collect();
                let (mode, need) = match a.as_str() {
                    "--ablate" => (EditMode::Ablate, 2),
                    "--scale" => (EditMode::Scale, 3),
                    _ => (EditMode::Pin, 3),
                };
                if p.len() != need {
                    die(&format!("{a} expects {}", if need == 2 { "L:I" } else { "L:I:V" }));
                }
                let value = if need == 3 { parse_f(p[2]) } else { 0.0 };
                req.edits.neurons.push(NeuronEdit { layer: parse_u(p[0]), index: parse_u(p[1]), mode, value });
            }
            _ => prompt.push(a),
        }
    }
    req.prompt = prompt.join(" ");
    if req.prompt.is_empty() {
        die("usage: generate [--model DIR] [--no-think] [--no-lens] [--max N] [--sample SEED] [--ablate L:I] [--scale L:I:V] [--pin L:I:V] \"prompt\"");
    }

    let mut engine = Engine::open(&model)?;
    let geo = engine.geometry;
    let cancel = AtomicBool::new(false);
    let mut last: Option<Frame> = None;
    let mut out = std::io::stdout();
    let summary = generate(&mut engine, &req, &cancel, &mut |f: &Frame| {
        match f.stream {
            StreamKind::Thinking => print!("\x1b[2m{}\x1b[0m", f.text),
            StreamKind::Answer => print!("\x1b[1m{}\x1b[0m", f.text),
            _ => {}
        }
        let _ = out.flush();
        last = Some(f.clone());
    })?;
    println!();
    println!("\nprefill {:.1} ms, {:.1} tok/s, {} tokens, stop {:?}", summary.prefill_ms, summary.decode_tok_per_s, summary.tokens_generated, summary.stop);

    if let Some(f) = last {
        let mut idx: Vec<usize> = (0..f.neurons.len()).collect();
        idx.sort_by(|&a, &b| f.neurons[b].abs().total_cmp(&f.neurons[a].abs()));
        println!("\ntop 10 neurons at last token (layer:index value):");
        for &i in idx.iter().take(10) {
            println!("  L{}:{}  {:.4}", i / geo.num_neurons, i % geo.num_neurons, f.neurons[i]);
        }
        println!("\nlogit lens of last token (layer: token prob):");
        for (l, (id, p)) in f.lens.iter().enumerate() {
            println!("  L{l:>2}: {:?} {:.3}", engine.tok.token_text(*id), p);
        }
    }
    Ok(())
}
