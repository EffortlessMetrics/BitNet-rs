//! ChatML-family multi-turn rendering helpers.

use crate::{ChatTurn, render_chatml, render_qwen25_chatml};

pub(crate) fn with_default_system(
    system: Option<&str>,
    default_system: &'static str,
    history: &[ChatTurn],
) -> String {
    render_chatml(system.unwrap_or(default_system), history)
}

pub(crate) fn qwen25_with_default_system(
    system: Option<&str>,
    default_system: &'static str,
    history: &[ChatTurn],
) -> String {
    render_qwen25_chatml(system.unwrap_or(default_system), history)
}
