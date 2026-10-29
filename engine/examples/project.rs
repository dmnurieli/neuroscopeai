//! "French ink per floor": contrast 6 French-vs-English instruction pairs, take the direction at site 15,
//! greedily generate an answer to a French prompt, then run project (like ns_project) on the whole
//! prompt+answer without edits and with the direction ablated at every site.
//! usage: project [--model DIR] [--site S] [--max N]

use neuroscope_engine::generate::generate;
use neuroscope_engine::project::project;
use neuroscope_engine::*;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut model = PathBuf::from(home).join("Library/Application Support/NeuroScope/models/OLMo-2-0425-1B-Instruct");
    let (mut site, mut max) = (15usize, 40usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let v = args.next().unwrap_or_default();
        match a.as_str() {
            "--model" => model = PathBuf::from(v),
            "--site" => site = v.parse()?,
            "--max" => max = v.parse()?,
            _ => anyhow::bail!("usage: project [--model DIR] [--site S] [--max N]"),
        }
    }
    let qs = [
        "What is the capital of Italy?",
        "How do plants make food?",
        "Why do we have seasons?",
        "What causes rain?",
        "How does a magnet work?",
        "Why is the ocean salty?",
    ];
    let a: Vec<String> = qs.iter().map(|q| format!("Answer in French, in one short sentence: {q}")).collect();
    let b: Vec<String> = qs.iter().map(|q| format!("Answer in English, in one short sentence: {q}")).collect();
    let mut engine = Engine::open(&model)?;
    let g = engine.geometry;

    let con = contrast::contrast(&mut engine, &ContrastRequest { a, b, thinking: false, edits: Default::default() })?;
    let dir = con.directions[site].clone();
    println!("direction: contrast site {site}, separation d = {:.2}, |diff| = {:.2}", con.separation[site], con.norms[site]);

    let prompt = "Answer in French, in one short sentence: Why is the sky blue?";
    let n_prompt = engine.tok.encode_chat(prompt, false)?.len();
    let req = Request {
        prompt: prompt.into(),
        thinking: false,
        max_tokens: max,
        logit_lens: false,
        top_k: 1,
        pace_ms: 0,
        sampling: Sampling::Greedy,
        edits: Default::default(),
    };
    let (mut ids, mut text) = (Vec::new(), String::new());
    generate(&mut engine, &req, &AtomicBool::new(false), &mut |f: &Frame| {
        ids.push(f.token_id);
        if f.index >= n_prompt && f.stream == StreamKind::Answer {
            text.push_str(&f.text);
        }
    })?;
    println!("prompt tokens {n_prompt}, answer tokens {}: {:?}", ids.len() - n_prompt, text.trim());

    let ablate = EditSet {
        directions: vec![DirectionEdit { label: "french".into(), vector: dir.clone(), sites: vec![], mode: DirectionMode::Ablate, value: 0.0 }],
        ..Default::default()
    };
    let mk = |edits: EditSet, sample: bool| ProjectRequest {
        token_ids: ids.clone(),
        edits,
        direction: dir.clone(),
        sample_site: sample.then_some(site),
        sample_position: sample.then_some(n_prompt),
        sample_dims: 6,
    };
    let sites = g.num_layers + 1;
    // warm-up (shader compile), then timed runs
    project(&mut engine, &mk(EditSet::default(), false))?;
    let t0 = Instant::now();
    let plain = project(&mut engine, &mk(EditSet::default(), true))?;
    let ms_plain = t0.elapsed().as_secs_f64() * 1000.0;
    let t0 = Instant::now();
    let abl = project(&mut engine, &mk(ablate, true))?;
    let ms_abl = t0.elapsed().as_secs_f64() * 1000.0;

    let na = ids.len() - n_prompt;
    let mean = |which: &[f32], s: usize| -> f32 {
        (n_prompt..ids.len()).map(|p| which[p * sites + s]).sum::<f32>() / na as f32
    };
    println!("\nmean x·d̂ over the {na} answer positions (site 0 = embedding, s = after layer s-1)");
    println!("site   no edits(before)   ablated(before)   ablated(after)");
    for s in 0..sites {
        println!(
            "{s:>4}  {:>16.3}  {:>16.3}  {:>14.3}",
            mean(&plain.before, s),
            mean(&abl.before, s),
            mean(&abl.after, s)
        );
    }
    let worst = abl.after.iter().fold(0f32, |m, v| m.max(v.abs()));
    println!("\nmax |after| over all positions/sites with ablation: {worst:.4}");
    for (name, r) in [("no edits", &plain), ("ablated", &abl)] {
        let s = r.sample.as_ref().unwrap();
        println!("\nworked sample ({name}) site {} position {} (first answer token {:?}):", s.site, s.position, engine.tok.token_text(ids[s.position]));
        println!("  x·d̂ = {:.4}   after = {:.4}   |x| = {:.3}  |x_new| = {:.3}  |d| = {:.3}", s.dot, s.dot_after, s.x_norm, s.x_new_norm, s.d_norm);
        println!("  x     = {:.4?}\n  d̂     = {:.4?}\n  x_new = {:.4?}", s.x, s.d_hat, s.x_new);
    }
    println!("\ntiming: project {} tokens: no edits {ms_plain:.0} ms, ablated {ms_abl:.0} ms", ids.len());
    Ok(())
}
