//! Shared prompt-format builders used by chat renderers.

use crate::ChatTurn;

/// Render a multi-turn chat conversation in ChatML format.
pub(super) fn render_chatml(sys: &str, history: &[ChatTurn]) -> String {
    let mut out = String::new();
    out.push_str("<|im_start|>system\n");
    out.push_str(sys);
    out.push_str("<|im_end|>\n");
    for turn in history {
        out.push_str("<|im_start|>");
        out.push_str(turn.role.as_str());
        out.push('\n');
        out.push_str(&turn.text);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Render a multi-turn Qwen 2.5 chat conversation in ChatML format.
pub(super) fn render_qwen25_chatml(sys: &str, history: &[ChatTurn]) -> String {
    let mut out = String::new();
    out.push_str("<|im_start|>system\n");
    out.push_str(sys);
    out.push_str("<|im_end|>\n");
    for turn in history {
        out.push_str("<|im_start|>");
        out.push_str(turn.role.as_str());
        out.push('\n');
        out.push_str(&turn.text);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant");
    out
}
