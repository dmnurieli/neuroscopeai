//! Measures every number the "Inside a real AI" lessons state, on the model the app ships (OLMo 2 1B Instruct).
//! Mirrors what the app does (thinking off, greedy, the app's default request: logit lens on, top_k 8).
//!
//! usage: lesson_numbers [--model DIR] <command> [args]
//!   answer                      the lesson question's reply, token by token; first reply word; decode speed
//!   follow L I                  one desk's whole story at the first reply word (lesson 3)
//!   french [LO HI]              French-vs-English contrast: norms/separation per floor, top French desks on
//!                               floors LO..=HI with what they promote and what wakes them (lesson 3 candidates)
//!   derive [EXP]                the app's behavioural derivation for a Remove experiment (default remove-french),
//!                               then the chosen edit on held-out questions (lesson 4)
//!   ink SITE [Q]                the French direction from SITE: |d|, d-hat, ink per floor on the normal and erased run
//!   memo                        the memo's length entering every floor at the first reply word
//!   tokens TEXT...              how the chat template splits TEXT into word-pieces
//!   trace                       slow-motion arrows at the first reply word: explained share per floor
//!   add [EXP]                   the app's Add derivation for the ready-made Add experiments + the test question
//!   desks2 [LO HI]              French desks by activation at French words (NS_WB=0: input side only)
//! Model default: ~/Library/Application Support/NeuroScope/models/OLMo-2-0425-1B-Instruct.
//! Build + run: cd engine && cargo run --release --example lesson_numbers -- <command>

use neuroscope_engine::generate::generate;
use neuroscope_engine::*;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

const SKY: &str = "Answer in French, in one short sentence: Why is the sky blue?";
const QUESTIONS: [&str; 14] = [
    "What is the capital of Japan?", "How do plants make their food?", "Why do we have seasons?", "What is a rainbow?",
    "How does a bicycle stay upright?", "What do bees do all day?", "Why is the ocean salty?", "How do airplanes fly?",
    "What is the moon made of?", "Why do cats purr?", "How does a refrigerator keep food cold?", "What causes thunder?",
    "How do fish breathe underwater?", "What is gravity?",
];
const HELD_OUT: [&str; 2] = ["Why is the sky blue?", "What do cats eat?"];
/// Extra questions never used by the derivation, to check the chosen edit generalizes.
const FRESH: [&str; 14] = [
    "How do volcanoes erupt?", "Why do birds migrate?", "What is electricity?", "How do bees make honey?",
    "Why does ice float?", "What makes a desert dry?", "Why do leaves change color in autumn?", "How do stars shine?",
    "Why do we sleep?", "What is a cloud?", "How do magnets work?", "Why is snow white?", "What do dogs eat?",
    "How does rain form?",
];

fn die(m: &str) -> ! {
    eprintln!("error: {m}");
    std::process::exit(2)
}

struct Gen {
    ids: Vec<u32>,
    streams: Vec<StreamKind>,
    texts: Vec<String>,
    answer: String,
    neurons_at: Vec<Vec<f32>>,
    tok_per_s: f64,
}

fn run(e: &mut Engine, prompt: &str, max: usize, lens: bool, top_k: usize, edits: &EditSet, keep_neurons: bool) -> Gen {
    let req = Request { prompt: prompt.into(), thinking: false, max_tokens: max, logit_lens: lens, top_k, pace_ms: 0,
        sampling: Sampling::Greedy, edits: edits.clone() };
    let mut g = Gen { ids: vec![], streams: vec![], texts: vec![], answer: String::new(), neurons_at: vec![], tok_per_s: 0.0 };
    let s = generate(e, &req, &AtomicBool::new(false), &mut |f: &Frame| {
        g.ids.push(f.token_id);
        g.streams.push(f.stream);
        g.texts.push(f.text.clone());
        if f.stream == StreamKind::Answer { g.answer.push_str(&f.text) }
        if keep_neurons { g.neurons_at.push(f.neurons.clone()) }
    })
    .unwrap_or_else(|err| die(&format!("generate: {err}")));
    g.tok_per_s = s.decode_tok_per_s;
    g
}

fn answer(e: &mut Engine, q: &str, edits: &EditSet) -> String {
    run(e, q, 40, false, 1, edits, false).answer
}

/// The app's slow-motion / desk-story word: first answer token with visible text.
fn first_word(g: &Gen) -> usize {
    (0..g.ids.len())
        .find(|&i| g.streams[i] == StreamKind::Answer && !g.texts[i].trim().is_empty())
        .unwrap_or_else(|| die("no reply word"))
}

// ---- TextSim (app/Sources/NeuroScope/Model/TextSim.swift, line for line) ----
fn words(s: &str) -> Vec<String> {
    s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(String::from).collect()
}
fn similarity(a: &str, b: &str) -> f32 {
    use std::collections::HashSet;
    let (wa, wb) = (words(a), words(b));
    if wa.is_empty() && wb.is_empty() { return 1.0 }
    let sa: HashSet<_> = wa.iter().collect();
    let sb: HashSet<_> = wb.iter().collect();
    let jac = sa.intersection(&sb).count() as f32 / sa.union(&sb).count().max(1) as f32;
    let len = wa.len().min(wb.len()) as f32 / wa.len().max(wb.len()).max(1) as f32;
    0.5 * jac + 0.5 * len
}
fn is_broken(out: &str, question: &str) -> bool {
    use std::collections::HashSet;
    let w = words(out);
    if w.len() < 2 { return true }
    let q: HashSet<_> = words(question).into_iter().collect();
    let sw: HashSet<_> = w.iter().cloned().collect();
    let echo = sw.intersection(&q).count() as f32 / sw.len().max(1) as f32;
    if echo > 0.8 { return true }
    if w.len() >= 8 && (sw.len() as f32 / w.len() as f32) < 0.35 { return true }
    false
}

fn french_contrast(e: &mut Engine, concept: &str, opposite: &str) -> ContrastResult {
    let a = QUESTIONS.iter().map(|q| format!("{concept}{q}")).collect();
    let b = QUESTIONS.iter().map(|q| format!("{opposite}{q}")).collect();
    contrast::contrast(e, &ContrastRequest { a, b, thinking: false, edits: EditSet::default() })
        .unwrap_or_else(|err| die(&format!("contrast: {err}")))
}

fn ablate_all(c: &ContrastResult, site: usize) -> EditSet {
    EditSet { directions: vec![DirectionEdit { label: "remove".into(), vector: c.directions[site].clone(), sites: vec![],
        mode: DirectionMode::Ablate, value: 0.0 }], ..Default::default() }
}

fn short(s: &str) -> String {
    s.trim().replace('\n', " ").chars().take(110).collect()
}

fn main() {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut model = PathBuf::from(home).join("Library/Application Support/NeuroScope/models/OLMo-2-0425-1B-Instruct");
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--model") {
        model = PathBuf::from(args.get(1).cloned().unwrap_or_else(|| die("--model needs a value")));
        args.drain(..2);
    }
    let cmd = args.first().cloned().unwrap_or_else(|| die("command: answer | follow L I | french | derive | ink SITE"));
    let mut e = Engine::open(&model).unwrap_or_else(|err| die(&format!("open: {err}")));
    let g = e.geometry;
    println!("model {:?}: {} floors x {} desks = {} desks, hidden {}, heads {}", e.arch(), g.num_layers, g.num_neurons,
             g.num_layers * g.num_neurons, g.hidden_size, g.num_heads);
    let num = |i: usize, d: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);

    match cmd.as_str() {
        "answer" => {
            let _ = run(&mut e, SKY, 8, true, 8, &EditSet::default(), false); // warm-up
            for lens in [true, false] {
                let r = run(&mut e, SKY, 64, lens, 8, &EditSet::default(), false);
                println!("logit lens {lens}: {:.1} tok/s = {:.2} ms per word, {:.3} ms per floor", r.tok_per_s,
                         1000.0 / r.tok_per_s, 1000.0 / r.tok_per_s / g.num_layers as f64);
                if lens {
                    println!("answer: {:?}", r.answer.trim());
                    let fw = first_word(&r);
                    println!("first reply word: position {fw} = {:?} (next: {:?})", r.texts[fw], r.texts.get(fw + 1));
                    for i in 0..r.ids.len() {
                        if r.streams[i] != StreamKind::Prompt || i + 12 >= r.ids.len() {
                            println!("  {i:>3} {:?} {:?}", r.streams[i], r.texts[i]);
                        }
                    }
                    let prompt: Vec<String> = (0..r.ids.len()).filter(|&i| r.streams[i] == StreamKind::Prompt)
                        .map(|i| format!("{:?}", r.texts[i])).collect();
                    println!("prompt pieces: {}", prompt.join(" "));
                }
            }
        }
        "follow" => {
            let (layer, index) = (num(1, 0), num(2, 0));
            let r0 = run(&mut e, SKY, 64, true, 8, &EditSet::default(), false);
            let pos = first_word(&r0);
            let req = FollowRequest { token_ids: r0.ids.clone(), position: pos, edits: EditSet::default(), layer, index, top: 8, hops: 5 };
            let r = follow::follow(&mut e, &req).unwrap_or_else(|err| die(&format!("follow: {err}")));
            let tt = |id: u32| e.tok.token_text(id);
            println!("L{layer}:{index} at {pos} {:?}", r0.texts[pos]);
            println!("memo |x| {:.3}  rms {:.4}  gate {:+.4}  sigma {:.4}  silu {:+.4}  up {:+.4}  h_raw {:+.4}  h {:+.4}",
                     r.memo_norm, r.rms, r.gate, r.sigma, r.silu, r.up, r.h_raw, r.h);
            println!("wakes on: {}", r.wakes_on.iter().map(|w| format!("{:?} {:.3}", w.token, w.score)).collect::<Vec<_>>().join(", "));
            println!("asked by ({:.1}%): {}", 100.0 * r.asked_by_explained,
                     r.asked_by.iter().map(|x| format!("L{}:{} {:+.3}", x.from_layer, x.from_index, x.weight)).collect::<Vec<_>>().join(", "));
            println!("column |c| {:.4}  write |h c| {:.4}", r.column_norm, r.write_norm);
            println!("heard by: {}", r.heard_by.iter().map(|x| format!("L{}:{} {:+.3}", x.to_layer, x.to_index, x.weight)).collect::<Vec<_>>().join(", "));
            println!("path: {}", r.path.iter().map(|x| format!("L{}:{}→L{}:{} {:+.3}", x.from_layer, x.from_index, x.to_layer, x.to_index, x.weight)).collect::<Vec<_>>().join(", "));
            println!("next word {:?}: {:+.4}", tt(r.next_token), r.next_effect);
            println!("word effects: {}", r.word_effects.iter().map(|w| format!("{:?} {:+.3}", w.token, w.score)).collect::<Vec<_>>().join(", "));
        }
        "french" => {
            let (lo, hi) = (num(1, g.num_layers / 2), num(2, g.num_layers * 85 / 100));
            let c = french_contrast(&mut e, "Answer in French, in one short sentence: ", "Answer in English, in one short sentence: ");
            for s in 0..c.norms.len() {
                println!("site {s:>2} (after floor {:>2}): norm {:>8.3}  separation {:>7.3}", s as i64 - 1, c.norms[s], c.separation[s]);
            }
            let r0 = run(&mut e, SKY, 64, false, 1, &EditSet::default(), true);
            let pos = first_word(&r0);
            println!("sky answer: {:?}; first word {pos} {:?}", short(&r0.answer), r0.texts[pos]);
            let n = g.num_neurons;
            for l in lo..=hi {
                let row = &c.neuron_diff[l * n..(l + 1) * n];
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
                println!("== floor {l}: top French desks (mean diff over the 14 pairs)");
                for &i in idx.iter().take(4) {
                    let info = attribution::neuron_info(&e.model, &e.tok, l, i, 8).unwrap();
                    let fr = FollowRequest { token_ids: r0.ids.clone(), position: pos, edits: EditSet::default(), layer: l, index: i, top: 8, hops: 5 };
                    let f = follow::follow(&mut e, &fr).unwrap();
                    println!("  #{i:<5} diff {:+.3}  act@word {:+.3}  gate {:+.3}  promotes [{}]  wakes [{}]", row[i],
                             r0.neurons_at[pos][l * n + i], f.gate,
                             info.promotes.iter().map(|t| format!("{:?}", t.token)).collect::<Vec<_>>().join(" "),
                             f.wakes_on.iter().map(|w| format!("{:?}", w.token)).collect::<Vec<_>>().join(" "));
                }
            }
        }
        "desks" => {
            // Desks that fire on French/France/Paris across languages and stay quiet on other countries.
            let (lo, hi) = (num(1, g.num_layers / 2), num(2, g.num_layers * 85 / 100));
            let fr = ["I love French food.", "She moved to France last year.", "Paris is lovely in spring.",
                      "Il parle français.", "法国是一个美丽的国家。", "我去过巴黎。", "Ich spreche Französisch.",
                      "La France est grande.", "The French Revolution began in 1789."];
            let ctrl = ["I love German food.", "She moved to Spain last year.", "Berlin is lovely in spring.",
                        "Er spricht Deutsch.", "德国是一个美丽的国家。", "我去过东京。", "Ich spreche Italienisch.",
                        "Japan is large.", "The Russian Revolution began in 1917."];
            let n = g.num_neurons;
            let peak = |e: &mut Engine, s: &str| -> Vec<f32> {
                let r = run(e, s, 1, false, 1, &EditSet::default(), true);
                let mut m = vec![0f32; g.num_layers * n];
                for (i, fr) in r.neurons_at.iter().enumerate() {
                    if r.streams[i] != StreamKind::Prompt || i < 5 { continue } // skip the chat scaffold
                    for k in 0..m.len() { m[k] = m[k].max(fr[k]) }
                }
                m
            };
            let frs: Vec<Vec<f32>> = fr.iter().map(|s| peak(&mut e, s)).collect();
            let cts: Vec<Vec<f32>> = ctrl.iter().map(|s| peak(&mut e, s)).collect();
            let r0 = run(&mut e, SKY, 64, false, 1, &EditSet::default(), true);
            let pos = first_word(&r0);
            for l in lo..=hi {
                let mut sc: Vec<(usize, f32, f32, f32)> = (l * n..(l + 1) * n).map(|k| {
                    let fmin = frs.iter().map(|v| v[k]).fold(f32::INFINITY, f32::min);
                    let fmean = frs.iter().map(|v| v[k]).sum::<f32>() / frs.len() as f32;
                    let cmax = cts.iter().map(|v| v[k]).fold(f32::NEG_INFINITY, f32::max);
                    (k - l * n, fmin - cmax, fmean, cmax)
                }).collect();
                sc.sort_by(|a, b| b.1.total_cmp(&a.1));
                println!("== floor {l}");
                for &(i, s, fmean, cmax) in sc.iter().take(5) {
                    let info = attribution::neuron_info(&e.model, &e.tok, l, i, 8).unwrap();
                    let fq = FollowRequest { token_ids: r0.ids.clone(), position: pos, edits: EditSet::default(), layer: l, index: i, top: 8, hops: 5 };
                    let f = follow::follow(&mut e, &fq).unwrap();
                    println!("  #{i:<5} margin {s:+.3} fr-mean {fmean:+.3} ctrl-max {cmax:+.3} | at {:?}: gate {:+.3} h {:+.3} | promotes [{}] | wakes [{}]",
                             r0.texts[pos], f.gate, f.h,
                             info.promotes.iter().map(|t| format!("{:?}", t.token)).collect::<Vec<_>>().join(" "),
                             f.wakes_on.iter().map(|w| format!("{:?}", w.token)).collect::<Vec<_>>().join(" "));
                }
            }
        }
        "desks2" => {
            // A: activation at the French word itself vs the same sentence with another country's word.
            // B: mean activation over French replies minus English replies to the same questions.
            let (lo, hi) = (num(1, g.num_layers / 2), num(2, g.num_layers * 85 / 100));
            let pairs = [("I love French food.", "French", "I love German food.", "German"),
                         ("She moved to France last year.", "France", "She moved to Spain last year.", "Spain"),
                         ("Paris is lovely in spring.", "Paris", "Berlin is lovely in spring.", "Berlin"),
                         ("我去过法国。", "法国", "我去过德国。", "德国"),
                         ("我去过巴黎。", "巴黎", "我去过东京。", "东京"),
                         ("Ich spreche Französisch.", "Französisch", "Ich spreche Italienisch.", "Italienisch"),
                         ("The French Revolution began in 1789.", "French", "The Russian Revolution began in 1917.", "Russian"),
                         ("We drove to Lyon and Marseille.", "Marseille", "We drove to Munich and Hamburg.", "Hamburg")];
            let n = g.num_neurons;
            let at_word = |e: &mut Engine, s: &str, w: &str| -> Vec<f32> {
                let r = run(e, s, 1, false, 1, &EditSet::default(), true);
                // the marker's last token: where the full prompt and the prompt cut after the marker stop agreeing
                let cut = &s[..s.find(w).unwrap() + w.len()];
                let full = e.tok.encode_chat(s, false).unwrap();
                let part = e.tok.encode_chat(cut, false).unwrap();
                let common = full.iter().zip(&part).take_while(|(x, y)| x == y).count();
                let k = common - 1;
                r.neurons_at[k].clone()
            };
            let mut a = vec![0f32; g.num_layers * n];
            let mut a_min = vec![f32::INFINITY; g.num_layers * n];
            for (fs, fw, cs, cw) in pairs {
                let f = at_word(&mut e, fs, fw);
                let c = at_word(&mut e, cs, cw);
                for k in 0..a.len() { let d = f[k].abs() - c[k].abs(); a[k] += d / pairs.len() as f32; a_min[k] = a_min[k].min(d) }
            }
            let qs = ["Why is the sky blue?", "What do cats eat?", "How do volcanoes erupt?", "Why do birds migrate?"];
            let mut b = vec![0f32; g.num_layers * n];
            for q in qs {
                for (sign, pre) in [(1f32, "Answer in French, in one short sentence: "), (-1f32, "Answer in English, in one short sentence: ")] {
                    let r = run(&mut e, &format!("{pre}{q}"), 24, false, 1, &EditSet::default(), true);
                    let idx: Vec<usize> = (0..r.ids.len()).filter(|&i| r.streams[i] == StreamKind::Answer).collect();
                    for &i in &idx { for k in 0..b.len() { b[k] += sign * r.neurons_at[i][k] / (idx.len() * qs.len()) as f32 } }
                }
            }
            let r0 = run(&mut e, SKY, 64, false, 1, &EditSet::default(), true);
            let pos = first_word(&r0);
            for l in lo..=hi {
                let sl = l * n..(l + 1) * n;
                let sd = |v: &[f32]| { let m = v.iter().sum::<f32>() / v.len() as f32; (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32).sqrt().max(1e-6) };
                let (sa, sb) = (sd(&a[sl.clone()]), sd(&b[sl.clone()]));
                let mut idx: Vec<usize> = (0..n).collect();
                let wb: f32 = std::env::var("NS_WB").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
                let comb = |i: usize| a[l * n + i] / sa + wb * b[l * n + i] / sb;
                idx.sort_by(|&x, &y| comb(y).total_cmp(&comb(x)));
                println!("== floor {l} (sd A {sa:.3}, sd B {sb:.3})");
                for &i in idx.iter().take(6) {
                    let k = l * n + i;
                    let info = attribution::neuron_info(&e.model, &e.tok, l, i, 8).unwrap();
                    let fq = FollowRequest { token_ids: r0.ids.clone(), position: pos, edits: EditSet::default(), layer: l, index: i, top: 10, hops: 5 };
                    let f = follow::follow(&mut e, &fq).unwrap();
                    println!("  #{i:<5} A {:+.3} (min {:+.3}) B {:+.3} | at {:?}: gate {:+.3} h {:+.3} | promotes [{}] | wakes [{}]",
                             a[k], a_min[k], b[k], r0.texts[pos], f.gate, f.h,
                             info.promotes.iter().map(|t| format!("{:?}", t.token)).collect::<Vec<_>>().join(" "),
                             f.wakes_on.iter().map(|w| format!("{:?}", w.token)).collect::<Vec<_>>().join(" "));
                }
            }
        }
        "derive" => {
            let exp = args.get(1).cloned().unwrap_or_else(|| "remove-french".into());
            let (cp, op): (&str, &str) = match exp.as_str() {
                "remove-french" => ("Answer in French, in one short sentence: ", "Answer in English, in one short sentence: "),
                "remove-brevity" => ("Answer in at most five words: ", "Answer in detail, in several sentences: "),
                // derive custom "Answer in Spanish: " "Answer in English: " (the app's Make-your-own prefixes)
                "custom" => (args.get(2).map(String::as_str).unwrap_or_else(|| die("custom CP OP")),
                             args.get(3).map(String::as_str).unwrap_or_else(|| die("custom CP OP"))),
                _ => die("experiments: remove-french | remove-brevity | custom CP OP"),
            };
            let c = french_contrast(&mut e, cp, op);
            let probes: Vec<String> = HELD_OUT.iter().map(|q| format!("{cp}{q}")).collect();
            let targets: Vec<String> = HELD_OUT.iter().map(|q| format!("{op}{q}")).collect();
            let probe_ref: Vec<String> = probes.iter().map(|p| answer(&mut e, p, &EditSet::default())).collect();
            let target_ref: Vec<String> = targets.iter().map(|p| answer(&mut e, p, &EditSet::default())).collect();
            let plain_q = HELD_OUT[0];
            let plain_ref = answer(&mut e, plain_q, &EditSet::default());
            for (p, r) in probes.iter().zip(&probe_ref) { println!("probe  {p:<70} -> {}", short(r)) }
            for (p, r) in targets.iter().zip(&target_ref) { println!("target {p:<70} -> {}", short(r)) }
            println!("plain  {plain_q:<70} -> {}", short(&plain_ref));
            let l = g.num_layers;
            let cap = ((0.70 * l as f32).floor() as usize).clamp(1, l);
            let mut best = (1usize, f32::NEG_INFINITY);
            let (slo, shi) = std::env::var("NS_SITES").ok().and_then(|v| { let (a, b) = v.split_once('-')?; Some((a.parse().ok()?, b.parse().ok()?)) }).unwrap_or(((cap / 2).max(1), cap));
            for site in slo..=shi {
                let ed = ablate_all(&c, site);
                let mut total = 0f32;
                let mut sample = String::new();
                let mut rejected = false;
                for (i, p) in probes.iter().enumerate() {
                    let out = answer(&mut e, p, &ed);
                    if i == 0 { sample = out.clone() }
                    if is_broken(&out, p) { rejected = true; break }
                    total += similarity(&out, &target_ref[i]) - similarity(&out, &probe_ref[i]);
                }
                let mut plain = String::new();
                if !rejected {
                    plain = answer(&mut e, plain_q, &ed);
                    if is_broken(&plain, plain_q) || similarity(&plain, &plain_ref) < 0.25 { rejected = true }
                }
                let score = if rejected { -1.0 } else { total / probes.len() as f32 };
                if score > best.1 { best = (site, score) }
                println!("site {site:>2} (floor {:>2}) score {score:+.3}  {}  | plain: {}", site - 1, short(&sample), short(&plain));
            }
            println!("CHOSEN site {} (direction from floor {}), score {:+.3}{}", best.0, best.0 - 1, best.1,
                     if best.1 > 0.15 { "" } else { "  (weak: the app shows its heads-up note)" });
            let ed = ablate_all(&c, best.0);
            println!("== held-out check (never in the lists), before -> after");
            for q in FRESH.iter().chain(HELD_OUT.iter()) {
                let p = format!("{cp}{q}");
                println!("{p}\n   before: {}\n   after:  {}", short(&answer(&mut e, &p, &EditSet::default())), short(&answer(&mut e, &p, &ed)));
            }
            for q in FRESH.iter().take(4) {
                println!("plain {q}\n   before: {}\n   after:  {}", short(&answer(&mut e, q, &EditSet::default())), short(&answer(&mut e, q, &ed)));
            }
        }
        "ink" => {
            // ink SITE [QUESTION]: the French direction from SITE; ink per floor on the normal and the erased run
            let site = num(1, 1);
            let q = args.get(2).cloned().unwrap_or_else(|| SKY.into());
            let c = french_contrast(&mut e, "Answer in French, in one short sentence: ", "Answer in English, in one short sentence: ");
            println!("|d| at site {site} = {:.3}; 1.0/1.5/2.0 x |d| = {:.2} / {:.2} / {:.2}", c.norms[site], c.norms[site],
                     1.5 * c.norms[site], 2.0 * c.norms[site]);
            let dn = c.norms[site].max(1e-9);
            println!("d-hat first 6: {:?}", c.directions[site].iter().take(6).map(|v| (v / dn * 1000.0).round() / 1000.0).collect::<Vec<_>>());
            for (label, ed) in [("normal", EditSet::default()), ("erased everywhere", ablate_all(&c, site))] {
                let r0 = run(&mut e, &q, 512, false, 1, &ed, false);
                println!("  ({} tokens, reply {})", r0.ids.len(), r0.streams.iter().filter(|s| **s == StreamKind::Answer).count());
                let pos = first_word(&r0);
                let reply: Vec<usize> = (0..r0.ids.len()).filter(|&i| r0.streams[i] == StreamKind::Answer).collect();
                let req = ProjectRequest { token_ids: r0.ids.clone(), edits: ed.clone(), direction: c.directions[site].clone(),
                    sample_site: Some(site), sample_position: Some(pos), sample_dims: 6 };
                let p = project::project(&mut e, &req).unwrap_or_else(|err| die(&format!("project: {err}")));
                println!("== {label} run of {q:?}: {:?}", short(&r0.answer));
                for s in 0..p.sites {
                    let m = |v: &[f32]| reply.iter().map(|&k| v[k * p.sites + s]).sum::<f32>() / reply.len().max(1) as f32;
                    println!("  site {s:>2} (after floor {:>2}): mean ink over reply arriving {:>8.3} leaving {:>8.3} | first word arriving {:>8.3}",
                             s as i64 - 1, m(&p.before), m(&p.after), p.before[pos * p.sites + s]);
                }
                if let Some(sm) = p.sample {
                    println!("  sample site {} word {:?}: |x| {:.3} -> {:.3}  ink {:.3} -> {:.3}", sm.site, r0.texts[pos], sm.x_norm, sm.x_new_norm, sm.dot, sm.dot_after);
                    println!("  x first 6 {:?}\n  d-hat first 6 {:?}\n  x_new first 6 {:?}", sm.x, sm.d_hat, sm.x_new);
                }
            }
        }
        "stats" => {
            // the app's per-floor brightness normaliser (99.5th percentile of |activation|), with and without token 0
            let r0 = run(&mut e, SKY, 512, false, 1, &EditSet::default(), true);
            let n = g.num_neurons;
            let pct = |v: &mut Vec<f32>| { v.sort_by(|a, b| a.total_cmp(b)); v[((v.len() - 1) as f64 * 0.995) as usize] };
            for l in 0..g.num_layers {
                let mut all = vec![]; let mut rest = vec![]; let mut t0 = vec![];
                for (t, fr) in r0.neurons_at.iter().enumerate() {
                    for j in 0..n { let a = fr[l * n + j].abs(); all.push(a); if t > 0 { rest.push(a) } else { t0.push(a) } }
                }
                let mx0 = t0.iter().cloned().fold(0f32, f32::max);
                let p = { let mut c = all.clone(); pct(&mut c) };
                let frac = |k: f32| all.iter().filter(|&&a| a > k * p).count() as f32 / all.len() as f32 * 100.0;
                println!("          share above 0.08p {:.1}%  0.3p {:.1}%  0.5p {:.2}%", frac(0.08), frac(0.3), frac(0.5));
                println!("floor {l:>2}: p99.5 all {:.3}  without token 0 {:.3}  token-0 max {:.1}  token-0 p99.5 {:.3}", pct(&mut all), pct(&mut rest), mx0, pct(&mut t0));
            }
        }
        "memo" => {
            // |x| of the memo entering every floor's desks at the first reply word (ns_follow memo_norm)
            let r0 = run(&mut e, SKY, 64, false, 1, &EditSet::default(), false);
            let pos = first_word(&r0);
            for l in 0..g.num_layers {
                let f = follow::follow(&mut e, &FollowRequest { token_ids: r0.ids.clone(), position: pos, edits: EditSet::default(),
                    layer: l, index: 0, top: 1, hops: 0 }).unwrap();
                println!("floor {l:>2} at {:?}: memo |x| {:.2}", r0.texts[pos], f.memo_norm);
            }
        }
        "tokens" => {
            for w in &args[1..] {
                let ids = e.tok.encode_chat(w, false).unwrap();
                println!("{w:?}: {}", ids.iter().map(|&i| format!("{:?}", e.tok.token_text(i))).collect::<Vec<_>>().join(" "));
            }
        }
        "trace" => {
            let r0 = run(&mut e, SKY, 64, false, 1, &EditSet::default(), false);
            let pos = first_word(&r0);
            let t = trace::trace(&mut e, &TraceRequest { token_ids: r0.ids.clone(), position: pos, edits: EditSet::default(),
                top_sources: 12, top_targets: 12, focus: None, focus_top: 24 }).unwrap_or_else(|err| die(&format!("trace: {err}")));
            println!("trace at {pos} {:?}: forward {:.1} ms", r0.texts[pos], t.forward_ms);
            for (l, x) in t.explained.iter().enumerate() { println!("  target floor {l:>2}: explained {:.1}%", 100.0 * x) }
        }
        "add" => {
            // The app's Add derivation for the ready-made Add experiments, then the chosen edit on the test question.
            let exps = [("add-french", "Answer in French, in one short sentence: ", "Answer in English, in one short sentence: ", "How do volcanoes erupt?"),
                        ("add-pirate", "Answer like a pirate, in one short sentence: ", "Answer plainly, in one short sentence: ", "How do volcanoes erupt?"),
                        ("add-brevity", "Answer in at most five words: ", "Answer in detail, in several sentences: ", "How does the internet work?"),
                        ("add-enthusiasm", "Answer with lots of excitement, in one short sentence: ", "Answer calmly, in one short sentence: ", "Why do leaves change color in autumn?")];
            let only = args.get(1).cloned();
            for (id, cp, op, test) in exps {
                if only.as_deref().is_some_and(|o| o != id) { continue }
                let c = french_contrast(&mut e, cp, op);
                let probes: Vec<String> = HELD_OUT.iter().map(|q| q.to_string()).collect();
                let target_ref: Vec<String> = HELD_OUT.iter().map(|q| answer(&mut e, &format!("{cp}{q}"), &EditSet::default())).collect();
                let probe_ref: Vec<String> = probes.iter().map(|p| answer(&mut e, p, &EditSet::default())).collect();
                let l = g.num_layers as f32;
                let (lo, hi) = std::env::var("NS_SITES").ok().and_then(|v| { let (a, b) = v.split_once('-')?; Some((a.parse().ok()?, b.parse().ok()?)) })
                    .unwrap_or((((0.55 * l).round() as usize).max(1), ((0.90 * l).round() as usize).min(g.num_layers)));
                let base: f32 = std::env::var("NS_MULT").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5);
                let mut score = |e: &mut Engine, site: usize, v: f32| -> (f32, String) {
                    let ed = EditSet { directions: vec![DirectionEdit { label: id.into(), vector: c.directions[site].clone(), sites: vec![site],
                        mode: DirectionMode::Add, value: v }], ..Default::default() };
                    let mut total = 0f32;
                    let mut sample = String::new();
                    for (i, p) in probes.iter().enumerate() {
                        let out = answer(e, p, &ed);
                        if i == 0 { sample = out.clone() }
                        if is_broken(&out, p) { return (-1.0, sample) }
                        total += similarity(&out, &target_ref[i]) - similarity(&out, &probe_ref[i]);
                    }
                    (total / probes.len() as f32, sample)
                };
                let mut best = (lo, 0f32, f32::NEG_INFINITY);
                println!("== {id}");
                for site in lo..=hi {
                    let v = base * c.norms[site];
                    let (sc, sample) = score(&mut e, site, v);
                    println!("  site {site:>2} (floor {:>2}) value {v:>7.2} score {sc:+.3}  {}", site - 1, short(&sample));
                    if sc > best.2 { best = (site, v, sc) }
                }
                for m in [base * 2.0 / 3.0, base * 4.0 / 3.0] {
                    let v = m * c.norms[best.0];
                    let (sc, sample) = score(&mut e, best.0, v);
                    println!("  site {:>2} x{m} value {v:>7.2} score {sc:+.3}  {}", best.0, short(&sample));
                    if sc > best.2 { best = (best.0, v, sc) }
                }
                let ed = EditSet { directions: vec![DirectionEdit { label: id.into(), vector: c.directions[best.0].clone(), sites: vec![best.0],
                    mode: DirectionMode::Add, value: best.1 }], ..Default::default() };
                println!("  CHOSEN site {} value {:.2} score {:+.3}{}", best.0, best.1, best.2, if best.2 > 0.15 { "" } else { " (weak)" });
                println!("  test {test}\n    before: {}\n    after:  {}", short(&answer(&mut e, test, &EditSet::default())), short(&answer(&mut e, test, &ed)));
            }
        }
        _ => die("unknown command"),
    }
}
