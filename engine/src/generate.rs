//! One run: encode → prefill (emit a Frame per prompt token) → decode loop (emit a Frame per token).

use crate::engine::Engine;
use crate::types::{Frame, Request, Summary};
use std::sync::atomic::AtomicBool;

/// Order of frames: every prompt position first (stream Prompt, logprob NaN, `top` filled only on the
/// last prompt position, `lens` only on the last prompt position), then each generated token.
/// A generated token's frame carries the activations of the forward pass that PROCESSED it (i.e. the
/// pass that consumes it as input and predicts the next token), its `top` is that pass's next-token
/// prediction, and `logprob` is the log-prob it had when it was chosen. The final generated token
/// (eos or the max_tokens-th) still gets one more forward pass so its frame is complete.
/// Stops on the tokenizer's stop ids (Qwen3 <|im_end|> / <|endoftext|>, OLMo 2 <|endoftext|>), max_tokens, or cancel.
pub fn generate(
    engine: &mut Engine,
    req: &Request,
    cancel: &AtomicBool,
    on_frame: &mut dyn FnMut(&Frame),
) -> anyhow::Result<Summary> {
    use crate::edits::DeviceEdits;
    use crate::streams::StreamTracker;
    use crate::tokenize::StreamDecoder;
    use crate::types::{StopReason, StreamKind};
    use crate::{lens, sampler};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    cancel.store(false, Ordering::SeqCst);
    let g = engine.geometry;
    let (np, nh) = (g.neurons_per_frame(), g.heads_per_frame());
    let temp = match &req.sampling {
        crate::types::Sampling::Sample { temperature, .. } => *temperature,
        _ => 1.0,
    };
    let edits = DeviceEdits::build(&req.edits, &g, engine.model.dtype(), engine.model.device())?;
    engine.model.clear_cache();
    let ids = engine.tok.encode_chat(&req.prompt, req.thinking)?;
    anyhow::ensure!(!ids.is_empty(), "empty prompt");
    let n = ids.len();

    let t0 = Instant::now();
    let (mut logits, cap) = engine.model.forward(&ids, 0, &edits, req.logit_lens)?;
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
    for (p, &id) in ids.iter().enumerate() {
        let last = p + 1 == n;
        let (top, lens_v) = if last {
            let top = sampler::top_k_probs(&logits, temp, req.top_k);
            let lv = match (&cap.residuals, req.logit_lens) {
                (Some(r), true) => match &cap.lens {
                    Some(l) => l.clone(),
                    None => lens::logit_lens(&engine.model, r)?,
                },
                _ => Vec::new(),
            };
            (top, lv)
        } else {
            (Vec::new(), Vec::new())
        };
        let frame = Frame {
            index: p,
            token_id: id,
            stream: StreamKind::Prompt,
            text: engine.tok.token_text(id),
            neurons: cap.neurons[p * np..(p + 1) * np].to_vec(),
            heads: cap.heads[p * nh..(p + 1) * nh].to_vec(),
            top,
            lens: lens_v,
            logprob: f32::NAN,
        };
        on_frame(&frame);
    }
    drop(cap);

    let mut sampler = sampler::Sampler::new(&req.sampling);
    let mut tracker = StreamTracker::for_tok(&engine.tok, req.thinking);
    let mut decoder = StreamDecoder::new();
    let mut stop = StopReason::MaxTokens;
    let mut generated = 0usize;
    let mut pos = n;
    // NS_PROFILE=1: per-step wall time of the decode loop, printed to stderr at the end.
    let profile = std::env::var_os("NS_PROFILE").is_some();
    let mut prof = [0f64; 6]; // sample+logprob, forward, lens, top_k, text, on_frame
    let mut lap = |slot: usize, t: &mut Instant| {
        if profile {
            let now = Instant::now();
            prof[slot] += (now - *t).as_secs_f64() * 1000.0;
            *t = now;
        }
    };
    // The (max, Σexp) of the current logits at temperature 1, from the top_k pass that produced the
    // previous frame's `top` (only reusable when that pass ran at temperature 1).
    let mut stats: Option<(f32, f64)> = None;
    let d0 = Instant::now();
    while generated < req.max_tokens {
        let mut t = Instant::now();
        let next = sampler.sample(&logits);
        let lp = match stats {
            Some((mx, sum)) => sampler::log_prob_from(&logits, next, mx, sum),
            None => sampler::log_prob(&logits, next),
        };
        lap(0, &mut t);
        if cancel.load(Ordering::SeqCst) {
            stop = StopReason::Cancelled;
            break;
        }
        let (new_logits, mut cap) = engine.model.forward(&[next], pos, &edits, req.logit_lens)?;
        logits = new_logits;
        lap(1, &mut t);
        let lens_v = match (&cap.residuals, req.logit_lens) {
            (Some(r), true) => match cap.lens.take() {
                Some(l) => l,
                None => lens::logit_lens(&engine.model, r)?,
            },
            _ => Vec::new(),
        };
        lap(2, &mut t);
        let (top, mx, sum) = sampler::top_k_probs_stats(&logits, temp, req.top_k);
        stats = if temp == 1.0 { Some((mx, sum)) } else { None };
        lap(3, &mut t);
        let text = decoder.push(&engine.tok, next);
        lap(4, &mut t);
        let frame = Frame {
            index: pos,
            token_id: next,
            stream: tracker.classify(next),
            text,
            neurons: cap.neurons[..np].to_vec(),
            heads: cap.heads[..nh].to_vec(),
            top,
            lens: lens_v,
            logprob: lp,
        };
        on_frame(&frame);
        lap(5, &mut t);
        pos += 1;
        generated += 1;
        if engine.tok.is_stop(next) {
            stop = StopReason::Eos;
            break;
        }
        if req.pace_ms > 0 {
            std::thread::sleep(Duration::from_millis(req.pace_ms));
        }
    }
    let secs = d0.elapsed().as_secs_f64();
    if profile && generated > 0 {
        let g = generated as f64;
        eprintln!(
            "[profile] ms/token: sample+logprob {:.3}  forward {:.3}  lens {:.3}  top_k {:.3}  text {:.3}  on_frame {:.3}  (total {:.3})",
            prof[0] / g, prof[1] / g, prof[2] / g, prof[3] / g, prof[4] / g, prof[5] / g, secs * 1000.0 / g
        );
    }
    Ok(Summary {
        ok: true,
        tokens_generated: generated,
        stop,
        prefill_ms,
        decode_tok_per_s: if secs > 0.0 { generated as f64 / secs } else { 0.0 },
    })
}
