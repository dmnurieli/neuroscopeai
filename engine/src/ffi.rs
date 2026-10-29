//! C ABI — implements include/neuroscope.h exactly. Every entry point is wrapped in catch_unwind;
//! a panic never crosses into Swift (it becomes an error JSON / NULL).

#![allow(clippy::missing_safety_doc)]

use crate::engine::Engine;
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use crate::tokenize::Tok;
use crate::types::{Frame, Request};
use std::ffi::c_void;

#[repr(C)]
pub struct NSGeometry {
    pub num_layers: u32,
    pub num_neurons: u32,
    pub num_heads: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
}

#[repr(C)]
pub struct NSTokenFrame {
    pub run_id: u32,
    pub index: u32,
    pub token_id: u32,
    pub stream: i32,
    pub text: *const c_char,
    pub neurons: *const f32,
    pub heads: *const f32,
    pub top_ids: *const u32,
    pub top_probs: *const f32,
    pub top_k: u32,
    pub lens_ids: *const u32,
    pub lens_probs: *const f32,
    pub logprob: f32,
}

pub type NSFrameCallback = Option<unsafe extern "C" fn(user: *mut c_void, frame: *const NSTokenFrame)>;

/// Opaque handle behind `NSEngine*`.
pub struct NSEngine {
    pub engine: std::sync::Mutex<Engine>,
    pub cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    tok: Tok,
}

fn cstr(s: &str) -> *mut c_char {
    CString::new(s.replace('\0', "")).unwrap().into_raw()
}

fn err_json(msg: &str) -> *mut c_char {
    cstr(&serde_json::json!({"ok": false, "error": msg}).to_string())
}

fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        format!("panic: {s}")
    } else if let Some(s) = p.downcast_ref::<String>() {
        format!("panic: {s}")
    } else {
        "panic".to_string()
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ns_engine_open(model_dir: *const c_char, err_out: *mut *mut c_char) -> *mut NSEngine {
    let r = catch_unwind(AssertUnwindSafe(|| -> Result<*mut NSEngine, String> {
        if model_dir.is_null() {
            return Err("null model_dir".into());
        }
        let path = unsafe { CStr::from_ptr(model_dir) }.to_str().map_err(|e| format!("model_dir not UTF-8: {e}"))?;
        let eng = Engine::open(std::path::Path::new(path)).map_err(|e| format!("{e:#}"))?;
        let cancel = eng.cancel.clone();
        let tok = eng.tok.clone();
        Ok(Box::into_raw(Box::new(NSEngine { engine: std::sync::Mutex::new(eng), cancel, tok })))
    }));
    let res = match r {
        Ok(x) => x,
        Err(p) => Err(panic_msg(p)),
    };
    match res {
        Ok(p) => p,
        Err(m) => {
            if !err_out.is_null() {
                unsafe { *err_out = cstr(&m) };
            }
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ns_engine_close(engine: *mut NSEngine) {
    if engine.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(engine) })));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ns_engine_geometry(engine: *const NSEngine) -> NSGeometry {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let e = unsafe { &*engine };
        let g = match e.engine.lock() {
            Ok(g) => g.geometry,
            Err(p) => p.into_inner().geometry,
        };
        NSGeometry {
            num_layers: g.num_layers as u32,
            num_neurons: g.num_neurons as u32,
            num_heads: g.num_heads as u32,
            hidden_size: g.hidden_size as u32,
            vocab_size: g.vocab_size as u32,
        }
    }));
    r.unwrap_or(NSGeometry { num_layers: 0, num_neurons: 0, num_heads: 0, hidden_size: 0, vocab_size: 0 })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ns_generate(
    engine: *mut NSEngine,
    run_id: u32,
    request_json: *const c_char,
    on_frame: NSFrameCallback,
    user: *mut c_void,
) -> *mut c_char {
    let r = catch_unwind(AssertUnwindSafe(|| -> Result<String, String> {
        if engine.is_null() || request_json.is_null() {
            return Err("null argument".into());
        }
        let e = unsafe { &*engine };
        let s = unsafe { CStr::from_ptr(request_json) }.to_str().map_err(|e| format!("request not UTF-8: {e}"))?;
        let req: Request = serde_json::from_str(s).map_err(|e| format!("bad request: {e}"))?;
        let mut guard = match e.engine.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let cancel = e.cancel.clone();
        let mut cb = |f: &Frame| {
            let Some(cbf) = on_frame else { return };
            let text = CString::new(f.text.replace('\0', "")).unwrap_or_default();
            let top_ids: Vec<u32> = f.top.iter().map(|t| t.0).collect();
            let top_probs: Vec<f32> = f.top.iter().map(|t| t.1).collect();
            let lens_ids: Vec<u32> = f.lens.iter().map(|t| t.0).collect();
            let lens_probs: Vec<f32> = f.lens.iter().map(|t| t.1).collect();
            let has_lens = !f.lens.is_empty();
            let frame = NSTokenFrame {
                run_id,
                index: f.index as u32,
                token_id: f.token_id,
                stream: f.stream as i32,
                text: text.as_ptr(),
                neurons: f.neurons.as_ptr(),
                heads: f.heads.as_ptr(),
                top_ids: top_ids.as_ptr(),
                top_probs: top_probs.as_ptr(),
                top_k: top_ids.len() as u32,
                lens_ids: if has_lens { lens_ids.as_ptr() } else { std::ptr::null() },
                lens_probs: if has_lens { lens_probs.as_ptr() } else { std::ptr::null() },
                logprob: f.logprob,
            };
            unsafe { cbf(user, &frame) };
        };
        let summary = crate::generate::generate(&mut guard, &req, &cancel, &mut cb).map_err(|e| format!("{e:#}"))?;
        serde_json::to_string(&summary).map_err(|e| e.to_string())
    }));
    match r {
        Ok(Ok(s)) => cstr(&s),
        Ok(Err(m)) => err_json(&m),
        Err(p) => err_json(&panic_msg(p)),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ns_cancel(engine: *mut NSEngine) {
    if engine.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        unsafe { &*engine }.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
    }));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ns_neuron_info(engine: *mut NSEngine, layer: u32, index: u32, top_k: u32) -> *mut c_char {
    let r = catch_unwind(AssertUnwindSafe(|| -> String {
        if engine.is_null() {
            return serde_json::json!({"error": "null engine"}).to_string();
        }
        let e = unsafe { &*engine };
        let guard = match e.engine.try_lock() {
            Ok(g) => g,
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
            Err(_) => return serde_json::json!({"error": "busy"}).to_string(),
        };
        match crate::attribution::neuron_info(&guard.model, &guard.tok, layer as usize, index as usize, top_k as usize) {
            Ok(info) => serde_json::to_string(&info).unwrap_or_else(|e| serde_json::json!({"error": e.to_string()}).to_string()),
            Err(err) => serde_json::json!({"error": format!("{err:#}")}).to_string(),
        }
    }));
    match r {
        Ok(s) => cstr(&s),
        Err(p) => cstr(&serde_json::json!({"error": panic_msg(p)}).to_string()),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ns_contrast(engine: *mut NSEngine, request_json: *const c_char) -> *mut c_char {
    let r = catch_unwind(AssertUnwindSafe(|| -> String {
        let err = |m: String| serde_json::json!({"error": m}).to_string();
        if engine.is_null() || request_json.is_null() {
            return err("null argument".into());
        }
        let e = unsafe { &*engine };
        let s = match unsafe { CStr::from_ptr(request_json) }.to_str() {
            Ok(s) => s,
            Err(x) => return err(format!("request not UTF-8: {x}")),
        };
        let req: crate::types::ContrastRequest = match serde_json::from_str(s) {
            Ok(r) => r,
            Err(x) => return err(format!("bad request: {x}")),
        };
        let mut guard = match e.engine.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        match crate::contrast::contrast(&mut guard, &req) {
            Ok(res) => serde_json::to_string(&res).unwrap_or_else(|x| err(x.to_string())),
            Err(x) => err(format!("{x:#}")),
        }
    }));
    match r {
        Ok(s) => cstr(&s),
        Err(p) => cstr(&serde_json::json!({"error": panic_msg(p)}).to_string()),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ns_trace(engine: *mut NSEngine, request_json: *const c_char) -> *mut c_char {
    let r = catch_unwind(AssertUnwindSafe(|| -> String {
        let err = |m: String| serde_json::json!({"error": m}).to_string();
        if engine.is_null() || request_json.is_null() {
            return err("null argument".into());
        }
        let e = unsafe { &*engine };
        let s = match unsafe { CStr::from_ptr(request_json) }.to_str() {
            Ok(s) => s,
            Err(x) => return err(format!("request not UTF-8: {x}")),
        };
        let req: crate::types::TraceRequest = match serde_json::from_str(s) {
            Ok(r) => r,
            Err(x) => return err(format!("bad request: {x}")),
        };
        let mut guard = match e.engine.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        match crate::trace::trace(&mut guard, &req) {
            Ok(res) => serde_json::to_string(&res).unwrap_or_else(|x| err(x.to_string())),
            Err(x) => err(format!("{x:#}")),
        }
    }));
    match r {
        Ok(s) => cstr(&s),
        Err(p) => cstr(&serde_json::json!({"error": panic_msg(p)}).to_string()),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ns_project(engine: *mut NSEngine, request_json: *const c_char) -> *mut c_char {
    let r = catch_unwind(AssertUnwindSafe(|| -> String {
        let err = |m: String| serde_json::json!({"error": m}).to_string();
        if engine.is_null() || request_json.is_null() {
            return err("null argument".into());
        }
        let e = unsafe { &*engine };
        let s = match unsafe { CStr::from_ptr(request_json) }.to_str() {
            Ok(s) => s,
            Err(x) => return err(format!("request not UTF-8: {x}")),
        };
        let req: crate::types::ProjectRequest = match serde_json::from_str(s) {
            Ok(r) => r,
            Err(x) => return err(format!("bad request: {x}")),
        };
        let mut guard = match e.engine.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        match crate::project::project(&mut guard, &req) {
            Ok(res) => serde_json::to_string(&res).unwrap_or_else(|x| err(x.to_string())),
            Err(x) => err(format!("{x:#}")),
        }
    }));
    match r {
        Ok(s) => cstr(&s),
        Err(p) => cstr(&serde_json::json!({"error": panic_msg(p)}).to_string()),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ns_follow(engine: *mut NSEngine, request_json: *const c_char) -> *mut c_char {
    let r = catch_unwind(AssertUnwindSafe(|| -> String {
        let err = |m: String| serde_json::json!({"error": m}).to_string();
        if engine.is_null() || request_json.is_null() {
            return err("null argument".into());
        }
        let e = unsafe { &*engine };
        let s = match unsafe { CStr::from_ptr(request_json) }.to_str() {
            Ok(s) => s,
            Err(x) => return err(format!("request not UTF-8: {x}")),
        };
        let req: crate::types::FollowRequest = match serde_json::from_str(s) {
            Ok(r) => r,
            Err(x) => return err(format!("bad request: {x}")),
        };
        let mut guard = match e.engine.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        match crate::follow::follow(&mut guard, &req) {
            Ok(res) => serde_json::to_string(&res).unwrap_or_else(|x| err(x.to_string())),
            Err(x) => err(format!("{x:#}")),
        }
    }));
    match r {
        Ok(s) => cstr(&s),
        Err(p) => cstr(&serde_json::json!({"error": panic_msg(p)}).to_string()),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ns_token_text(engine: *mut NSEngine, token_id: u32) -> *mut c_char {
    let r = catch_unwind(AssertUnwindSafe(|| -> *mut c_char {
        if engine.is_null() {
            return std::ptr::null_mut();
        }
        cstr(&unsafe { &*engine }.tok.token_text(token_id))
    }));
    r.unwrap_or(std::ptr::null_mut())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ns_string_free(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { CString::from_raw(s) })));
}
