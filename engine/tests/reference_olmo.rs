//! OLMo 2 path vs HF transformers (scripts/reference/ref_olmo.py -> olmo_fixtures.json + olmo_logits.bin),
//! on Metal bf16 (what the app runs) and CPU f32 (tight tolerances). Needs the weights in
//! ~/Library/Application Support/NeuroScope/models/OLMo-2-0425-1B-Instruct (override: NS_OLMO_DIR; another
//! stage's fixtures: `ref_olmo.py <dir> <prefix>` then NS_OLMO_DIR=<dir> NS_OLMO_FIXTURES=<prefix>).
//! Run:
//!   scripts/reference/.venv/bin/python scripts/reference/ref_olmo.py
//!   cd engine && cargo test --release --test reference_olmo -- --ignored --nocapture --test-threads=1

use neuroscope_engine::edits::DeviceEdits;
use neuroscope_engine::types::{EditSet, HeadEdit, NeuronEdit, Request, Sampling, StreamKind};
use neuroscope_engine::{Engine, generate};
use candle_core::Device;
use std::path::PathBuf;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt())
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn olmo_dir() -> PathBuf {
    std::env::var_os("NS_OLMO_DIR").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("HOME").unwrap())
            .join("Library/Application Support/NeuroScope/models/OLMo-2-0425-1B-Instruct")
    })
}

struct Fixtures {
    fx: serde_json::Value,
    /// name -> full last-token logits
    logits: std::collections::HashMap<String, Vec<f32>>,
}

fn fixtures() -> Fixtures {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../scripts/reference");
    let prefix = std::env::var("NS_OLMO_FIXTURES").unwrap_or_else(|_| "olmo".into());
    let fx: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(here.join(format!("{prefix}_fixtures.json"))).expect("run scripts/reference/ref_olmo.py first"),
    )
    .unwrap();
    let raw = std::fs::read(here.join(format!("{prefix}_logits.bin"))).expect("run scripts/reference/ref_olmo.py first");
    let v = fx["vocab_size"].as_u64().unwrap() as usize;
    let names: Vec<String> = fx["logit_records"].as_array().unwrap().iter().map(|n| n.as_str().unwrap().to_string()).collect();
    assert_eq!(raw.len(), names.len() * v * 4, "olmo_logits.bin size");
    let mut logits = std::collections::HashMap::new();
    for (k, name) in names.into_iter().enumerate() {
        let rec: Vec<f32> = raw[k * v * 4..(k + 1) * v * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        logits.insert(name, rec);
    }
    Fixtures { fx, logits }
}

fn u32s(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}
fn f32s(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

/// Tolerances per device/dtype.
struct Tol {
    logits: f64,
    /// cosine of (edited − unedited) logits, ours vs reference, for the 48-neuron ablation
    delta: f64,
    /// the same for the 3-head ablation. bf16 rounding is not common-mode between an edited and an
    /// unedited run, so small-effect edits sit in a noise floor (measured 2026-10-28, prompt 0: single-head
    /// ablations give Δ-cosine 0.10–0.98 in candle Metal bf16 and −0.20–0.999 in HF transformers bf16,
    /// while candle Metal f32 and CPU f32 give 1.00000) — hence a looser bf16 bound for heads.
    head_delta: f64,
    act: f64,
    exact_greedy: bool,
}

fn prefill(eng: &mut Engine, ids: &[u32], edits: &EditSet) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let g = eng.geometry;
    let de = if edits.is_empty() {
        DeviceEdits::none(g.num_layers)
    } else {
        DeviceEdits::build(edits, &g, eng.model.dtype(), eng.model.device()).unwrap()
    };
    eng.model.clear_cache();
    let (logits, cap) = eng.model.forward(ids, 0, &de, false).unwrap();
    eng.model.clear_cache();
    let last = ids.len() - 1;
    let n = cap.neurons[last * g.neurons_per_frame()..(last + 1) * g.neurons_per_frame()].to_vec();
    let h = cap.heads[last * g.heads_per_frame()..(last + 1) * g.heads_per_frame()].to_vec();
    (logits, n, h)
}

fn greedy(eng: &mut Engine, prompt: &str, max_tokens: usize, logit_lens: bool) -> (Vec<u32>, f64) {
    let req = Request {
        prompt: prompt.to_string(),
        thinking: false,
        max_tokens,
        logit_lens,
        top_k: 8,
        pace_ms: 0,
        sampling: Sampling::Greedy,
        edits: EditSet::default(),
    };
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let mut out = Vec::new();
    let s = generate::generate(eng, &req, &cancel, &mut |f| {
        if f.stream != StreamKind::Prompt {
            out.push(f.token_id);
        }
    })
    .unwrap();
    (out, s.decode_tok_per_s)
}

fn run(device: Device, dtype: candle_core::DType, label: &str, tol: Tol) {
    let fxs = fixtures();
    let fx = &fxs.fx;
    let t0 = std::time::Instant::now();
    let mut eng = Engine::open_with(&olmo_dir(), &device, dtype).unwrap();
    println!("[{label}] loaded in {:.1}s, geometry {:?}", t0.elapsed().as_secs_f64(), eng.geometry);
    let g = eng.geometry;
    assert_eq!((g.num_layers, g.num_neurons, g.num_heads, g.num_kv_heads, g.head_dim, g.hidden_size, g.vocab_size), (16, 8192, 16, 16, 128, 2048, 100352));

    let prompts = fx["prompts"].as_array().unwrap();
    let mut base0 = Vec::new();
    for (pi, p) in prompts.iter().enumerate() {
        let prompt = p["prompt"].as_str().unwrap();
        let ids = u32s(&p["ids"]);
        // (0) the engine renders + tokenizes the chat template exactly as HF apply_chat_template
        assert_eq!(eng.tok.encode_chat(prompt, false).unwrap(), ids, "prompt {pi}: chat-template ids differ from HF");
        assert_eq!(eng.tok.encode_chat(prompt, true).unwrap(), ids, "thinking flag must not change OLMo's template");

        // (a) prefill logits
        let (logits, neurons, heads) = prefill(&mut eng, &ids, &EditSet::default());
        let r = &fxs.logits[&format!("prompt{pi}")];
        let c = cosine(&logits, r);
        let (t_ours, t_ref) = (argmax(&logits), argmax(r));
        let top10 = u32s(&p["top10_ids"]);
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let overlap = top10.iter().filter(|&&i| order[..10].contains(&(i as usize))).count();
        let maxd = logits.iter().zip(r).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("[{label}] prompt {pi}: top1 ours {t_ours} ref {t_ref}; cosine {c:.6}; top10 overlap {overlap}/10; max |Δlogit| {maxd:.4}");
        assert_eq!(t_ours, t_ref, "prompt {pi}: top-1 mismatch");
        assert!(c >= tol.logits, "prompt {pi}: logit cosine {c} < {}", tol.logits);

        if pi == 0 {
            base0 = logits.clone();
            // captured MLP activations (silu(gate)*up = HF down_proj input) and per-head norms (o_proj input)
            for (l, v) in p["activations"].as_object().unwrap() {
                let l: usize = l.parse().unwrap();
                let c = cosine(&neurons[l * g.num_neurons..(l + 1) * g.num_neurons], &f32s(v));
                println!("[{label}]   layer {l:2} activation cosine {c:.6}");
                assert!(c >= tol.act, "layer {l} activation cosine {c}");
            }
            let hn: Vec<f32> = p["head_norms"].as_array().unwrap().iter().flat_map(|row| f32s(row)).collect();
            let c = cosine(&heads, &hn);
            let rel = heads.iter().zip(&hn).map(|(a, b)| ((a - b) / b.abs().max(1e-3)).abs()).fold(0f32, f32::max);
            println!("[{label}]   head norms (16x16) cosine {c:.6}, max rel err {rel:.4}");
            assert!(c >= tol.act, "head norm cosine {c}");
        }

        // (b) cached greedy generation (engine's generate(): prefill + KV-cache decode)
        let want = u32s(&p["greedy"]);
        let margins = f32s(&p["greedy_margins"]);
        let (got, tps) = greedy(&mut eng, prompt, 32, false);
        let first_diff = got.iter().zip(&want).position(|(a, b)| a != b);
        match first_diff {
            None => {
                assert_eq!(got.len(), want.len(), "prompt {pi}: lengths differ (ours {:?} ref {:?})", got, want);
                println!("[{label}] prompt {pi}: greedy {} tokens identical to HF ({tps:.1} tok/s)", got.len());
            }
            Some(k) => {
                println!(
                    "[{label}] prompt {pi}: greedy diverges at generated token {k} (ours {} ref {}); HF top1-top2 margin there {:.3}; \
                     identical before. ours: {:?}",
                    got[k], want[k], margins[k], eng.tok.inner.decode(&got, false).unwrap()
                );
                assert!(!tol.exact_greedy, "prompt {pi}: f32 greedy must match exactly");
                assert!(margins[k] < 0.5, "prompt {pi}: diverged at a confident step (margin {})", margins[k]);
            }
        }
    }

    // (c) edits on prompt 0: neurons, then heads — the engine's change must be the reference's change
    let ids0 = u32s(&prompts[0]["ids"]);
    let ref_base = &fxs.logits["prompt0"];
    let neuron_edits: Vec<NeuronEdit> = serde_json::from_value(fx["neuron_edits"].clone()).unwrap();
    let head_edits: Vec<HeadEdit> = serde_json::from_value(fx["head_edits"].clone()).unwrap();
    for (name, set, rec) in [
        ("neuron ablation", EditSet { neurons: neuron_edits, ..Default::default() }, "prompt0_neuron_ablation"),
        ("head ablation", EditSet { heads: head_edits, ..Default::default() }, "prompt0_head_ablation"),
    ] {
        let (ours, neurons, _) = prefill(&mut eng, &ids0, &set);
        let r = &fxs.logits[rec];
        let c = cosine(&ours, r);
        let cd = cosine(&sub(&ours, &base0), &sub(r, ref_base));
        let nd = sub(&ours, &base0).iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        println!("[{label}] {name} ({} units): edited-logit cosine {c:.6}; Δ-logit cosine {cd:.6}; |Δ| ours {nd:.2}", set.neurons.len() + set.heads.len());
        assert!(c >= tol.logits, "{name}: edited logit cosine {c}");
        let want = if set.heads.is_empty() { tol.delta } else { tol.head_delta };
        assert!(cd >= want, "{name}: delta cosine {cd} < {want}");
        for e in &set.neurons {
            assert_eq!(neurons[e.layer * g.num_neurons + e.index], 0.0, "ablated neuron captured non-zero");
        }
    }
}

/// What the app runs.
#[test]
#[ignore]
fn olmo_matches_transformers_metal_bf16() {
    let tol = Tol { logits: 0.999, delta: 0.999, head_delta: 0.9, act: 0.999, exact_greedy: false };
    run(Device::new_metal(0).unwrap(), candle_core::DType::BF16, "metal bf16", tol);
}

/// Same kernels in f32 (NS_DTYPE=f32): must be exact.
#[test]
#[ignore]
fn olmo_matches_transformers_metal_f32() {
    let tol = Tol { logits: 0.99999, delta: 0.9999, head_delta: 0.9999, act: 0.9999, exact_greedy: true };
    run(Device::new_metal(0).unwrap(), candle_core::DType::F32, "metal f32", tol);
}

#[test]
#[ignore]
fn olmo_matches_transformers_cpu_f32() {
    let tol = Tol { logits: 0.99999, delta: 0.9999, head_delta: 0.9999, act: 0.9999, exact_greedy: true };
    run(Device::Cpu, candle_core::DType::F32, "cpu f32", tol);
}

/// Decode speed on Metal, OLMo 2 1B vs Qwen3-0.6B, same prompt, 64 greedy tokens (EOS-free prompt),
/// with and without the logit lens (the app's default is on).
#[test]
#[ignore]
fn decode_speed_olmo_vs_qwen() {
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let qwen = home.join("Library/Application Support/NeuroScope/models/Qwen3-0.6B");
    let prompt = "Write a long story about a lighthouse keeper.";
    let metal = Device::new_metal(0).unwrap();
    for (label, dir, dt) in [
        ("OLMo-2-0425-1B-Instruct bf16", olmo_dir(), candle_core::DType::BF16),
        ("Qwen3-0.6B bf16", qwen.clone(), candle_core::DType::BF16),
        ("OLMo-2-0425-1B-Instruct f32", olmo_dir(), candle_core::DType::F32),
        ("Qwen3-0.6B f32", qwen, candle_core::DType::F32),
    ] {
        let mut eng = Engine::open_with(&dir, &metal, dt).unwrap();
        let _ = greedy(&mut eng, prompt, 8, false); // warm-up (Metal shader compile)
        for lens in [false, true] {
            let mut best = 0f64;
            let mut n = 0;
            for _ in 0..3 {
                let (toks, tps) = greedy(&mut eng, prompt, 64, lens);
                n = toks.len();
                best = best.max(tps);
            }
            println!("{label:30} logit_lens={lens:5}: {best:6.1} tok/s (best of 3, {n} tokens)");
        }
    }
}

/// Every analysis entry point on the real OLMo weights (Metal bf16): runs, has OLMo's shapes, and is
/// self-consistent where an exact identity exists.
#[test]
#[ignore]
fn olmo_analyses_run() {
    use neuroscope_engine::types::*;
    use neuroscope_engine::{attribution, contrast, follow, project, trace};
    let mut eng = Engine::open(&olmo_dir()).unwrap();
    let g = eng.geometry;
    let (ids, _) = greedy(&mut eng, "What is the capital of France?", 8, true);
    let mut all = eng.tok.encode_chat("What is the capital of France?", false).unwrap();
    all.extend(&ids);
    let pos = all.len() - 2;

    let tr = trace::trace(&mut eng, &TraceRequest { token_ids: all.clone(), position: pos, edits: EditSet::default(), top_sources: 12, top_targets: 12, focus: Some((10, 123)), focus_top: 24 }).unwrap();
    assert_eq!(tr.edges.len(), (g.num_layers - 1) * 144);
    assert_eq!(tr.explained.len(), 16);
    assert!(tr.explained.iter().all(|e| (0.0..=1.0).contains(e)));
    assert_eq!(tr.focus_edges.len(), 24);
    println!("trace: explained {:?}; strongest edge {:?}; {:.1} ms", tr.explained.iter().map(|e| (e * 100.0).round() / 100.0).collect::<Vec<_>>(), tr.edges[0], tr.forward_ms);

    let fo = follow::follow(&mut eng, &FollowRequest { token_ids: all.clone(), position: pos, edits: EditSet::default(), layer: 7, index: 4321, top: 8, hops: 5 }).unwrap();
    assert_eq!(fo.rms, 1.0, "OLMo 2 MLPs read the raw residual");
    assert!((fo.write_norm - fo.h.abs() * fo.column_norm).abs() <= 1e-4 * (1.0 + fo.write_norm));
    assert_eq!(fo.next_token, all[pos + 1]);
    assert!(fo.path.iter().all(|e| e.to_layer < 16));
    println!("follow (7, 4321): gate {:.3} h {:.4} write_norm {:.4}; wakes_on {:?}; heard_by[0] {:?}; next_effect {:.4}",
        fo.gate, fo.h, fo.write_norm, fo.wakes_on.iter().take(4).map(|w| w.token.clone()).collect::<Vec<_>>(), fo.heard_by.first(), fo.next_effect);

    let ni = attribution::neuron_info(&eng.model, &eng.tok, 15, 8191, 5).unwrap();
    println!("neuron_info (15, 8191): promotes {:?} out_norm {:.3}", ni.promotes.iter().map(|t| t.token.clone()).collect::<Vec<_>>(), ni.out_norm);
    assert_eq!(ni.promotes.len(), 5);

    let c = contrast::contrast(&mut eng, &ContrastRequest { a: vec!["I love this".into(), "What a wonderful day".into()], b: vec!["I hate this".into(), "What an awful day".into()], thinking: false, edits: EditSet::default() }).unwrap();
    assert_eq!((c.directions.len(), c.directions[0].len(), c.neuron_diff.len()), (17, 2048, 16 * 8192));
    let best = (0..17).max_by(|&a, &b| c.separation[a].total_cmp(&c.separation[b])).unwrap();
    println!("contrast: best site {best} separation {:.2}", c.separation[best]);

    let d = c.directions[best].clone();
    let p = project::project(&mut eng, &ProjectRequest { token_ids: all.clone(), edits: EditSet { directions: vec![DirectionEdit { label: "sentiment".into(), vector: d.clone(), sites: vec![], mode: DirectionMode::Ablate, value: 0.0 }], ..Default::default() }, direction: d, sample_site: Some(best), sample_position: Some(3), sample_dims: 6 }).unwrap();
    assert_eq!((p.sites, p.positions), (17, all.len()));
    let max_after = p.after.iter().fold(0f32, |m, v| m.max(v.abs()));
    let max_before = p.before.iter().fold(0f32, |m, v| m.max(v.abs()));
    println!("project: max |x·d̂| before {max_before:.2} after ablation {max_after:.4}");
    assert!(max_after <= 0.02 * max_before.max(1.0), "ablated direction must be gone after each site's edit");
}

/// The C ABI the Swift app calls (ns_engine_open → geometry → ns_generate with edits → close) on OLMo.
#[test]
#[ignore]
fn olmo_through_c_abi() {
    use neuroscope_engine::ffi::*;
    use std::ffi::{CStr, CString, c_void};
    struct Sink {
        frames: usize,
        streams: Vec<i32>,
        ok_sizes: bool,
        g: (u32, u32),
    }
    unsafe extern "C" fn on_frame(user: *mut c_void, f: *const NSTokenFrame) {
        let s = unsafe { &mut *(user as *mut Sink) };
        let f = unsafe { &*f };
        s.frames += 1;
        s.streams.push(f.stream);
        let n = unsafe { std::slice::from_raw_parts(f.neurons, (s.g.0 * s.g.1) as usize) };
        // lens pointers are set on generated frames (prompt frames carry the lens only at the last position)
        s.ok_sizes &= n.iter().all(|v| v.is_finite()) && !f.heads.is_null() && (f.stream == 0 || !f.lens_ids.is_null());
    }
    unsafe {
        let dir = CString::new(olmo_dir().to_str().unwrap()).unwrap();
        let mut err: *mut std::ffi::c_char = std::ptr::null_mut();
        let e = ns_engine_open(dir.as_ptr(), &mut err);
        assert!(!e.is_null(), "open failed: {:?}", if err.is_null() { None } else { Some(CStr::from_ptr(err)) });
        let g = ns_engine_geometry(e);
        assert_eq!((g.num_layers, g.num_neurons, g.num_heads, g.hidden_size, g.vocab_size), (16, 8192, 16, 2048, 100352));
        let req = CString::new(r#"{"prompt":"What is the capital of France?","thinking":true,"max_tokens":16,"logit_lens":true,"top_k":8,
            "edits":{"neurons":[{"layer":3,"index":120,"mode":"ablate"}],"heads":[{"layer":5,"head":2,"mode":"scale","value":0.5}]}}"#).unwrap();
        let mut sink = Sink { frames: 0, streams: Vec::new(), ok_sizes: true, g: (g.num_layers, g.num_neurons) };
        let out = ns_generate(e, 1, req.as_ptr(), Some(on_frame), &mut sink as *mut Sink as *mut c_void);
        let summary = CStr::from_ptr(out).to_str().unwrap().to_string();
        ns_string_free(out);
        let t = ns_token_text(e, 100257);
        assert_eq!(CStr::from_ptr(t).to_str().unwrap(), "<|endoftext|>");
        ns_string_free(t);
        let ni = ns_neuron_info(e, 15, 8191, 3);
        let ni_s = CStr::from_ptr(ni).to_str().unwrap().to_string();
        ns_string_free(ni);
        ns_engine_close(e);
        println!("C ABI summary {summary}; {} frames; streams {:?}; neuron_info {ni_s}", sink.frames, sink.streams);
        assert!(summary.contains("\"ok\":true") && summary.contains("\"stop\":\"eos\""), "{summary}");
        assert!(sink.ok_sizes);
        assert!(!sink.streams.contains(&1), "OLMo must never report thinking tokens");
        assert_eq!(*sink.streams.last().unwrap(), 3, "eos is a special token");
        assert!(ni_s.contains("\"promotes\""));
    }
}
