//! Who-triggered-whom demo.
//! usage: trace [--model DIR] [--prompt "…"] [--top N] [--steps K]
//! Generates a short greedy answer (thinking off), then traces the position of the answer's first word
//! with focus = that position's busiest last-floor desk. Prints timings, explained per floor, the 3
//! strongest arrows for the first K steps and the last step, and the focus arrows.

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
    let mut prompt = "What is the capital of France? One word.".to_string();
    let (mut top, mut steps) = (12usize, 4usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |n: &str| args.next().unwrap_or_else(|| die(&format!("{n} needs a value")));
        match a.as_str() {
            "--model" => model = PathBuf::from(val("--model")),
            "--prompt" => prompt = val("--prompt"),
            "--top" => top = val("--top").parse().unwrap_or_else(|_| die("bad --top")),
            "--steps" => steps = val("--steps").parse().unwrap_or_else(|_| die("bad --steps")),
            _ => die("usage: trace [--model DIR] [--prompt P] [--top N] [--steps K]"),
        }
    }
    let mut engine = Engine::open(&model)?;
    let g = engine.geometry;

    let req = Request {
        prompt: prompt.clone(),
        thinking: false,
        max_tokens: 12,
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
        assert_eq!(f.index, ids.len(), "frames out of order");
        ids.push(f.token_id);
        if f.stream == StreamKind::Answer {
            answer.push_str(&f.text);
            if first.is_none() && !f.text.trim().is_empty() {
                first = Some((f.index, f.neurons.clone()));
            }
        }
    })?;
    let (pos, neurons) = first.unwrap_or_else(|| die("no answer word generated"));
    let last = &neurons[(g.num_layers - 1) * g.num_neurons..];
    let busiest = (0..g.num_neurons).max_by(|&a, &b| last[a].abs().total_cmp(&last[b].abs())).unwrap();
    println!("prompt: {prompt}\nanswer: {:?}  ({} tokens total)", answer.trim(), ids.len());
    println!(
        "tracing position {pos} = {:?}; focus = L{}:{busiest} (a = {:+.3})",
        engine.tok.token_text(ids[pos]),
        g.num_layers - 1,
        last[busiest]
    );

    let treq = TraceRequest {
        token_ids: ids.clone(),
        position: pos,
        edits: EditSet::default(),
        top_sources: top,
        top_targets: top,
        focus: Some((g.num_layers - 1, busiest)),
        focus_top: 24,
    };
    let mut res = None;
    for run in 0..3 {
        let t0 = std::time::Instant::now();
        let r = trace::trace(&mut engine, &treq)?;
        println!(
            "trace run {run}: total {:.1} ms (forward_ms {:.1}, contributions {:.1} ms)",
            t0.elapsed().as_secs_f64() * 1000.0,
            r.forward_ms,
            t0.elapsed().as_secs_f64() * 1000.0 - r.forward_ms
        );
        res = Some(r);
    }
    let res = res.unwrap();
    // the JSON the FFI returns
    let json = serde_json::to_string(&res)?;
    println!("TraceResult JSON: {} bytes, {} edges, {} focus edges", json.len(), res.edges.len(), res.focus_edges.len());

    println!("\nexplained per target floor (share of shown targets' |gate| the shown arrows carry):");
    for (l, e) in res.explained.iter().enumerate() {
        println!("  L{l:>2}  {e:.3}  {}", "#".repeat((e * 50.0).round() as usize));
    }

    let per_step = top * top;
    let show: Vec<usize> = (0..steps.min(g.num_layers - 1)).chain([g.num_layers - 2]).collect();
    println!("\n3 strongest arrows per step:");
    for s in show {
        let step = &res.edges[s * per_step..(s + 1) * per_step];
        let txt: Vec<String> = step
            .iter()
            .take(3)
            .map(|e| format!("L{}:{} → L{}:{} {:+.3}", e.from_layer, e.from_index, e.to_layer, e.to_index, e.weight))
            .collect();
        println!("  step {s:>2}→{:<2}  {}", s + 1, txt.join("   "));
    }

    println!("\ntop focus arrows into L{}:{busiest}:", g.num_layers - 1);
    for e in res.focus_edges.iter().take(12) {
        println!("  L{:>2}:{:<5} {:+.4}", e.from_layer, e.from_index, e.weight);
    }
    Ok(())
}
