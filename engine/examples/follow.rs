//! One desk's whole story demo.
//! usage: follow [--model DIR] [--prompt "…"] [--layer L] [--top N] [--hops K]
//! Generates a short greedy answer (thinking off), picks the answer's first word position and the busiest
//! desk on floor L there (default 11), and prints every number of ns_follow plus the greedy path.

use neuroscope_engine::generate::generate;
use neuroscope_engine::*;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

fn die(m: &str) -> ! {
    eprintln!("error: {m}");
    std::process::exit(2)
}

fn main() -> anyhow::Result<()> {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut model = PathBuf::from(home).join("Library/Application Support/NeuroScope/models/OLMo-2-0425-1B-Instruct");
    let mut prompt = "Answer in French, in one short sentence: Why is the sky blue?".to_string();
    let (mut layer, mut top, mut hops) = (11usize, 8usize, 5usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |n: &str| args.next().unwrap_or_else(|| die(&format!("{n} needs a value")));
        match a.as_str() {
            "--model" => model = PathBuf::from(val("--model")),
            "--prompt" => prompt = val("--prompt"),
            "--layer" => layer = val("--layer").parse().unwrap_or_else(|_| die("bad --layer")),
            "--top" => top = val("--top").parse().unwrap_or_else(|_| die("bad --top")),
            "--hops" => hops = val("--hops").parse().unwrap_or_else(|_| die("bad --hops")),
            _ => die("usage: follow [--model DIR] [--prompt P] [--layer L] [--top N] [--hops K]"),
        }
    }
    let mut engine = Engine::open(&model)?;
    let g = engine.geometry;
    if layer >= g.num_layers {
        die("--layer out of range");
    }

    let req = Request {
        prompt: prompt.clone(),
        thinking: false,
        max_tokens: 40,
        logit_lens: false,
        top_k: 1,
        pace_ms: 0,
        sampling: Sampling::Greedy,
        edits: EditSet::default(),
    };
    let mut ids: Vec<u32> = Vec::new();
    let mut answer = String::new();
    let mut first: Option<(usize, Vec<f32>)> = None;
    let cancel = AtomicBool::new(false);
    generate(&mut engine, &req, &cancel, &mut |f: &Frame| {
        ids.push(f.token_id);
        if f.stream == StreamKind::Answer {
            answer.push_str(&f.text);
            if first.is_none() && !f.text.trim().is_empty() {
                first = Some((f.index, f.neurons.clone()));
            }
        }
    })?;
    let (pos, neurons) = first.unwrap_or_else(|| die("no answer word generated"));
    let row = &neurons[layer * g.num_neurons..(layer + 1) * g.num_neurons];
    let busiest = (0..g.num_neurons).max_by(|&a, &b| row[a].abs().total_cmp(&row[b].abs())).unwrap();
    println!("prompt: {prompt}\nanswer: {:?}  ({} tokens total)", answer.trim(), ids.len());
    println!(
        "following L{layer}:{busiest} at position {pos} = {:?} (frame activation {:+.4})",
        engine.tok.token_text(ids[pos]),
        row[busiest]
    );

    let freq = FollowRequest { token_ids: ids.clone(), position: pos, edits: EditSet::default(), layer, index: busiest, top, hops };
    let mut res = None;
    for run in 0..4 {
        let t0 = std::time::Instant::now();
        let r = follow::follow(&mut engine, &freq)?;
        println!("follow run {run}: {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0);
        res = Some(r);
    }
    let r = res.unwrap();
    let json = serde_json::to_string(&r)?;
    println!("FollowResult JSON: {} bytes", json.len());
    let tt = |id: u32| engine.tok.token_text(id);

    println!("\n1. reads the memo:   |x| = {:.3}   rms(x) = {:.4}", r.memo_norm, r.rms);
    println!("2. asks (gate):      W_gate·x̃ = {:+.4}", r.gate);
    println!("3. switch:           σ = {:.4}   silu = gate·σ = {:+.4}", r.sigma, r.silu);
    println!("4. sizes (up):       W_up·x̃ = {:+.4}   h_raw = silu·up = {:+.4}   h = {:+.4}", r.up, r.h_raw, r.h);
    println!("   wakes on (static): {}", r.wakes_on.iter().map(|w| format!("{:?} {:.3}", w.token, w.score)).collect::<Vec<_>>().join(", "));
    println!("   asked by (explains {:.1}% of |gate|):", 100.0 * r.asked_by_explained);
    for e in &r.asked_by {
        println!("     L{:>2}:{:<5} {:+.4}", e.from_layer, e.from_index, e.weight);
    }
    println!("5. writes:           |W_down[:, j]| = {:.4}   |h|·|col| = {:.4}", r.column_norm, r.write_norm);
    println!("6. heard by:");
    for e in &r.heard_by {
        println!("     → L{:>2}:{:<5} {:+.4}", e.to_layer, e.to_index, e.weight);
    }
    println!("   path:");
    for e in &r.path {
        println!("     L{:>2}:{:<5} → L{:>2}:{:<5} {:+.4}", e.from_layer, e.from_index, e.to_layer, e.to_index, e.weight);
    }
    println!("7. next word {:?} (id {}): direct effect {:+.4} on its logit", tt(r.next_token), r.next_token, r.next_effect);
    println!("   effect on the top candidates:");
    for w in &r.word_effects {
        println!("     {:<16} {:+.4}", format!("{:?}", w.token), w.score);
    }
    Ok(())
}
