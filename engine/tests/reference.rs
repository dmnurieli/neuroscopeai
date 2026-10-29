//! Cross-check against the independent numpy forward (scripts/reference/ref.py -> fixtures.json).
//! Run: cargo test --release --test reference -- --ignored --nocapture

use neuroscope_engine::edits::DeviceEdits;
use neuroscope_engine::Engine;
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

#[test]
#[ignore]
fn matches_numpy_reference() {
    let fx_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../scripts/reference/fixtures.json");
    let fx: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fx_path).expect("run scripts/reference/ref.py first"))
            .unwrap();
    let ids: Vec<u32> = fx["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let top_ids: Vec<usize> = fx["top10_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let top_logits: Vec<f32> = fx["top10_logits"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();

    let home = std::env::var("HOME").unwrap();
    let dir = PathBuf::from(home).join("Library/Application Support/NeuroScope/models/Qwen3-0.6B");
    let mut eng = Engine::open(&dir).unwrap();
    let g = eng.geometry;
    let (logits, cap) = eng
        .model
        .forward(&ids, 0, &DeviceEdits::none(g.num_layers), false)
        .unwrap();

    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let ours5 = &order[..5];
    let overlap = top_ids[..5].iter().filter(|i| ours5.contains(i)).count();
    let max_diff = top_ids.iter().zip(&top_logits).map(|(&i, &r)| (logits[i] - r).abs()).fold(0f32, f32::max);
    println!("top1 ours={} ref={}  top5 overlap={}  max logit abs diff (ref top10)={:.4}", order[0], top_ids[0], overlap, max_diff);

    let last = ids.len() - 1;
    let mut min_cos = 1f64;
    for l in [0usize, 13, 27] {
        let r: Vec<f32> = fx["activations"][l.to_string()].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let base = (last * g.num_layers + l) * g.num_neurons;
        let c = cosine(&cap.neurons[base..base + g.num_neurons], &r);
        println!("layer {l:2} activation cosine = {c:.6}");
        min_cos = min_cos.min(c);
        assert!(c > 0.99, "layer {l} cosine {c}");
    }
    assert_eq!(order[0], top_ids[0], "top-1 mismatch");
    assert!(overlap >= 4, "top-5 overlap {overlap}");
    let _ = min_cos;
}
