//! Splits generated tokens into thinking / answer / special streams.

use crate::types::{StreamKind, special};

pub struct StreamTracker {
    thinking_enabled: bool,
    in_think: bool,
    seen_any: bool,
    /// (`<think>`, `</think>`); None for models without a thinking mode (OLMo 2): everything is Answer.
    think: Option<(u32, u32)>,
    specials: Vec<u32>,
}

impl StreamTracker {
    /// Qwen3 ids (types::special).
    pub fn new(thinking_enabled: bool) -> Self {
        Self::with_ids(
            thinking_enabled,
            Some((special::THINK, special::THINK_END)),
            vec![special::IM_END, special::ENDOFTEXT, special::IM_START],
        )
    }

    /// The tokenizer's own think / control ids (see `Tok`).
    pub fn for_tok(tok: &crate::tokenize::Tok, thinking_enabled: bool) -> Self {
        Self::with_ids(thinking_enabled && tok.think_ids.is_some(), tok.think_ids, tok.special_ids.clone())
    }

    fn with_ids(thinking_enabled: bool, think: Option<(u32, u32)>, specials: Vec<u32>) -> Self {
        Self { thinking_enabled, in_think: false, seen_any: false, think, specials }
    }

    /// Classifies the next generated token. With thinking enabled, Qwen3 opens with `<think>`;
    /// everything until `</think>` is Thinking, everything after is Answer.
    pub fn classify(&mut self, token_id: u32) -> StreamKind {
        let first = !self.seen_any;
        self.seen_any = true;
        if let Some((open, close)) = self.think {
            if token_id == open {
                self.in_think = self.thinking_enabled || first;
                return StreamKind::Special;
            }
            if token_id == close {
                self.in_think = false;
                return StreamKind::Special;
            }
        }
        if self.specials.contains(&token_id) {
            StreamKind::Special
        } else if self.in_think {
            StreamKind::Thinking
        } else {
            StreamKind::Answer
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn think_then_answer() {
        let mut t = StreamTracker::new(true);
        assert_eq!(t.classify(special::THINK), StreamKind::Special);
        assert_eq!(t.classify(100), StreamKind::Thinking);
        assert_eq!(t.classify(special::THINK_END), StreamKind::Special);
        assert_eq!(t.classify(200), StreamKind::Answer);
        assert_eq!(t.classify(special::IM_END), StreamKind::Special);
    }

    #[test]
    fn olmo_style_has_no_thinking() {
        let mut t = StreamTracker::with_ids(true, None, vec![100257]);
        assert_eq!(t.classify(special::THINK), StreamKind::Answer);
        assert_eq!(t.classify(100), StreamKind::Answer);
        assert_eq!(t.classify(100257), StreamKind::Special);
    }

    #[test]
    fn no_think_is_answer() {
        let mut t = StreamTracker::new(false);
        assert_eq!(t.classify(100), StreamKind::Answer);
    }
}
