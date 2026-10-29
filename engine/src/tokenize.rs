//! tokenizer.json wrapper, the chat templates (Qwen3, OLMo 2), and incremental UTF-8-safe decoding.

use std::path::Path;

use crate::config::Arch;
use crate::types::special;

/// How a prompt is rendered before encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormat {
    /// "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" (+ empty think block when thinking is off).
    Qwen3,
    /// OLMo 2 SFT/DPO/RLVR/Instruct tokenizer_config.json chat_template for one user turn with
    /// add_generation_prompt: "<|endoftext|><|user|>\n{prompt}\n<|assistant|>\n". `<|user|>` and
    /// `<|assistant|>` are NOT added tokens in this tokenizer: they encode as ordinary BPE pieces.
    Olmo2,
    /// No chat template (OLMo 2 base and early-training checkpoints): the prompt text as is, no BOS
    /// (what `tokenizer(prompt)` gives in HF for these repos — the tokenizer has no post-processor).
    Plain,
}

#[derive(Clone)]
pub struct Tok {
    pub inner: tokenizers::Tokenizer,
    pub format: ChatFormat,
    /// Generation stops on any of these ids.
    pub stop_ids: Vec<u32>,
    /// Control tokens a generated frame reports as `StreamKind::Special`.
    pub special_ids: Vec<u32>,
    /// (`<think>`, `</think>`) when the model has a thinking mode (Qwen3).
    pub think_ids: Option<(u32, u32)>,
}

impl Tok {
    /// Loads tokenizer.json for the architecture named by the directory's config.json.
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        let cfg = crate::config::ModelConfig::load(model_dir)?;
        Self::load_for(model_dir, cfg.arch())
    }

    /// Loads tokenizer.json and checks the special ids it relies on. Qwen3: the ids in types::special
    /// must match the added tokens. OLMo 2: `<|endoftext|>` must be an added token; the chat format is
    /// Olmo2 when tokenizer_config.json carries a chat_template, else Plain.
    pub fn load_for(model_dir: &Path, arch: Arch) -> anyhow::Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("loading {}: {e}", path.display()))?;
        match arch {
            Arch::Qwen3 => Self::qwen3(inner),
            Arch::Olmo2 => {
                let eot = inner
                    .token_to_id("<|endoftext|>")
                    .ok_or_else(|| anyhow::anyhow!("OLMo 2 tokenizer has no <|endoftext|>"))?;
                let has_template = std::fs::read_to_string(model_dir.join("tokenizer_config.json"))
                    .ok()
                    .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
                    .map(|v| v.get("chat_template").is_some_and(|t| t.is_string()))
                    .unwrap_or(false);
                let mut special_ids = vec![eot];
                special_ids.extend(inner.token_to_id("<|pad|>"));
                Ok(Self {
                    inner,
                    format: if has_template { ChatFormat::Olmo2 } else { ChatFormat::Plain },
                    stop_ids: vec![eot],
                    special_ids,
                    think_ids: None,
                })
            }
        }
    }

    fn qwen3(inner: tokenizers::Tokenizer) -> anyhow::Result<Self> {
        let expected = [
            (special::ENDOFTEXT, "<|endoftext|>"),
            (special::IM_START, "<|im_start|>"),
            (special::IM_END, "<|im_end|>"),
            (special::THINK, "<think>"),
            (special::THINK_END, "</think>"),
        ];
        for (id, text) in expected {
            let got = inner.id_to_token(id);
            if got.as_deref() != Some(text) {
                anyhow::bail!("special token id {id}: expected {text:?}, tokenizer has {got:?}");
            }
        }
        Ok(Self {
            inner,
            format: ChatFormat::Qwen3,
            stop_ids: vec![special::IM_END, special::ENDOFTEXT],
            special_ids: vec![special::IM_END, special::ENDOFTEXT, special::IM_START],
            think_ids: Some((special::THINK, special::THINK_END)),
        })
    }

    /// The model's prompt rendering for one user turn. `thinking` only matters for Qwen3.
    pub fn render(&self, prompt: &str, thinking: bool) -> String {
        match self.format {
            ChatFormat::Qwen3 => Self::render_chat(prompt, thinking),
            ChatFormat::Olmo2 => format!("<|endoftext|><|user|>\n{prompt}\n<|assistant|>\n"),
            ChatFormat::Plain => prompt.to_string(),
        }
    }

    pub fn is_stop(&self, id: u32) -> bool {
        self.stop_ids.contains(&id)
    }

    /// Qwen3 template for one user turn:
    /// "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    /// and when thinking is disabled, followed by "<think>\n\n</think>\n\n".
    pub fn render_chat(prompt: &str, thinking: bool) -> String {
        let mut s = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
        if !thinking {
            s.push_str("<think>\n\n</think>\n\n");
        }
        s
    }

    /// render then encode WITHOUT adding extra special tokens (the template already has them).
    pub fn encode_chat(&self, prompt: &str, thinking: bool) -> anyhow::Result<Vec<u32>> {
        let text = self.render(prompt, thinking);
        let enc = self
            .inner
            .encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Display text of one token; special tokens shown literally ("<think>").
    pub fn token_text(&self, id: u32) -> String {
        self.inner.decode(&[id], false).unwrap_or_default()
    }
}

/// Emits the text each new token adds, holding back bytes until they form complete UTF-8.
pub struct StreamDecoder {
    ids: Vec<u32>,
    prefix_offset: usize,
    read_offset: usize,
}

impl Default for StreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self { ids: Vec::new(), prefix_offset: 0, read_offset: 0 }
    }

    /// Returns the newly completed text ("" if this token only started a multi-byte char).
    pub fn push(&mut self, tok: &Tok, id: u32) -> String {
        self.ids.push(id);
        let dec = |ids: &[u32]| tok.inner.decode(ids, false).unwrap_or_default();
        let prefix = dec(&self.ids[self.prefix_offset..self.read_offset]);
        let full = dec(&self.ids[self.prefix_offset..]);
        if full.len() > prefix.len() && !full.ends_with('\u{FFFD}') && full.is_char_boundary(prefix.len()) {
            let out = full[prefix.len()..].to_string();
            self.prefix_offset = self.read_offset;
            self.read_offset = self.ids.len();
            out
        } else {
            String::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dir() -> Option<PathBuf> {
        let p = PathBuf::from(std::env::var("HOME").ok()?)
            .join("Library/Application Support/NeuroScope/models/Qwen3-0.6B");
        p.join("tokenizer.json").exists().then_some(p)
    }

    #[test]
    fn template_and_specials() {
        let Some(d) = dir() else { return };
        let t = Tok::load(&d).unwrap();
        let ids = t.encode_chat("hi", false).unwrap();
        assert_eq!(ids[0], 151644);
        let ids = t.encode_chat("hi", true).unwrap();
        assert_eq!(ids[0], 151644);
        assert_eq!(t.token_text(151667), "<think>");
    }

    fn olmo_dir() -> Option<PathBuf> {
        let p = PathBuf::from(std::env::var("HOME").ok()?)
            .join("Library/Application Support/NeuroScope/models/OLMo-2-0425-1B-Instruct");
        p.join("tokenizer.json").exists().then_some(p)
    }

    #[test]
    fn olmo_template_and_stream() {
        let Some(d) = olmo_dir() else { return };
        let t = Tok::load(&d).unwrap();
        assert_eq!(t.format, ChatFormat::Olmo2);
        assert_eq!(t.stop_ids, vec![100257]);
        assert!(t.think_ids.is_none());
        let ids = t.encode_chat("hi", true).unwrap();
        assert_eq!(ids[0], 100257, "template starts with <|endoftext|> (the bos_token)");
        assert_eq!(t.inner.decode(&ids, false).unwrap(), "<|endoftext|><|user|>\nhi\n<|assistant|>\n");
        let s = "Hello, 世界 🌍 — it's 3.14!";
        let ids = t.inner.encode(s, false).unwrap().get_ids().to_vec();
        let mut sd = StreamDecoder::new();
        let mut out = String::new();
        for id in &ids {
            let p = sd.push(&t, *id);
            assert!(!p.contains('\u{FFFD}'));
            out.push_str(&p);
        }
        assert_eq!(out, s);
    }

    #[test]
    fn stream_matches_full() {
        let Some(d) = dir() else { return };
        let t = Tok::load(&d).unwrap();
        let s = "Hello, 世界 🌍";
        let ids = t.inner.encode(s, false).unwrap().get_ids().to_vec();
        let mut sd = StreamDecoder::new();
        let mut out = String::new();
        for id in &ids {
            let p = sd.push(&t, *id);
            assert!(!p.contains('\u{FFFD}'));
            out.push_str(&p);
        }
        assert_eq!(out, t.inner.decode(&ids, false).unwrap());
        assert_eq!(out, s);
    }
}
