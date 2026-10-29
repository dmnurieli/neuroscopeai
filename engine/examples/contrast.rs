//! Difference-of-means contrast + abliteration/steering demo.
//! usage: contrast A.txt B.txt [--model DIR] [--think] [--max N] [--site S] [--alpha X]
//!                 [--ablate-on "prompt"] [--steer-on "prompt"]
//! A.txt / B.txt: one prompt per line (A has the concept, B is matched without it).
//! Prints the separation per site, the best site, the top-20 neurons by |mean diff|, then:
//!   * an A-style prompt, baseline vs the best direction ablated at EVERY site (abliteration);
//!   * a B-style prompt, baseline vs +alpha·d̂ added at the best site (steering).

use neuroscope_engine::generate::generate;
use neuroscope_engine::*;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

fn die(m: &str) -> ! {
    eprintln!("error: {m}");
    std::process::exit(2)
}

fn lines(path: &str) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("{path}: {e}")))
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

fn answer(engine: &mut Engine, prompt: &str, thinking: bool, max: usize, edits: EditSet) -> anyhow::Result<String> {
    let req = Request {
        prompt: prompt.to_string(),
        thinking,
        max_tokens: max,
        logit_lens: false,
        top_k: 1,
        pace_ms: 0,
        sampling: Sampling::Greedy,
        edits,
    };
    let mut out = String::new();
    let cancel = AtomicBool::new(false);
    let sum = generate(engine, &req, &cancel, &mut |f: &Frame| {
        if matches!(f.stream, StreamKind::Answer | StreamKind::Thinking) {
            out.push_str(&f.text);
        }
    })?;
    Ok(format!("{}\n   ({} tokens, {:.0} tok/s)", out.trim(), sum.tokens_generated, sum.decode_tok_per_s))
}

fn main() -> anyhow::Result<()> {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut model = PathBuf::from(home).join("Library/Application Support/NeuroScope/models/OLMo-2-0425-1B-Instruct");
    let (mut thinking, mut max, mut site, mut alpha) = (false, 60usize, None::<usize>, None::<f32>);
    let (mut ablate_on, mut steer_on) = (None::<String>, None::<String>);
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |n: &str| args.next().unwrap_or_else(|| die(&format!("{n} needs a value")));
        match a.as_str() {
            "--model" => model = PathBuf::from(val("--model")),
            "--think" => thinking = true,
            "--max" => max = val("--max").parse().unwrap_or_else(|_| die("bad --max")),
            "--site" => site = Some(val("--site").parse().unwrap_or_else(|_| die("bad --site"))),
            "--alpha" => alpha = Some(val("--alpha").parse().unwrap_or_else(|_| die("bad --alpha"))),
            "--ablate-on" => ablate_on = Some(val("--ablate-on")),
            "--steer-on" => steer_on = Some(val("--steer-on")),
            _ => files.push(a),
        }
    }
    if files.len() != 2 {
        die("usage: contrast A.txt B.txt [--model DIR] [--think] [--max N] [--site S] [--alpha X] [--ablate-on P] [--steer-on P]");
    }
    let (a, b) = (lines(&files[0]), lines(&files[1]));
    let mut engine = Engine::open(&model)?;
    let g = engine.geometry;

    let t0 = std::time::Instant::now();
    let req = ContrastRequest { a: a.clone(), b: b.clone(), thinking, edits: Default::default() };
    let res = contrast::contrast(&mut engine, &req)?;
    println!("contrast: {} A vs {} B prompts in {:.0} ms", res.count_a, res.count_b, t0.elapsed().as_secs_f64() * 1000.0);

    println!("\nsite  separation(d)  |mean diff|   (site 0 = embedding, s = after layer s-1)");
    for (s, (d, n)) in res.separation.iter().zip(&res.norms).enumerate() {
        let bar = "#".repeat((d.abs().min(40.0)) as usize);
        println!("{s:>4}  {d:>12.2}  {n:>11.2}  {bar}");
    }
    let best = site.unwrap_or_else(|| {
        (1..res.separation.len()).max_by(|&x, &y| res.separation[x].total_cmp(&res.separation[y])).unwrap_or(1)
    });
    println!("\nbest site: {best} (after layer {}), d = {:.2}, |diff| = {:.2}", best as i64 - 1, res.separation[best], res.norms[best]);

    let mut idx: Vec<usize> = (0..res.neuron_diff.len()).collect();
    idx.sort_by(|&x, &y| res.neuron_diff[y].abs().total_cmp(&res.neuron_diff[x].abs()));
    println!("\ntop 20 neurons by |mean A − mean B| (layer:index diff):");
    for &i in idx.iter().take(20) {
        println!("  L{}:{}  {:+.4}", i / g.num_neurons, i % g.num_neurons, res.neuron_diff[i]);
    }

    let dir = res.directions[best].clone();
    let alpha = alpha.unwrap_or(res.norms[best]);
    let ablate = EditSet {
        directions: vec![DirectionEdit { label: "contrast".into(), vector: dir.clone(), sites: vec![], mode: DirectionMode::Ablate, value: 0.0 }],
        ..Default::default()
    };
    let steer = EditSet {
        directions: vec![DirectionEdit { label: "contrast".into(), vector: dir, sites: vec![best], mode: DirectionMode::Add, value: alpha }],
        ..Default::default()
    };

    let pa = ablate_on.unwrap_or_else(|| a[0].clone());
    println!("\n=== abliteration (direction from site {best} removed at all {} sites) ===\nprompt: {pa}", g.num_layers + 1);
    println!("-- baseline --\n{}", answer(&mut engine, &pa, thinking, max, EditSet::default())?);
    println!("-- ablated --\n{}", answer(&mut engine, &pa, thinking, max, ablate)?);

    let pb = steer_on.unwrap_or_else(|| b[0].clone());
    println!("\n=== steering (+{alpha:.2}·d̂ at site {best}) ===\nprompt: {pb}");
    println!("-- baseline --\n{}", answer(&mut engine, &pb, thinking, max, EditSet::default())?);
    println!("-- steered --\n{}", answer(&mut engine, &pb, thinking, max, steer)?);
    Ok(())
}
