//! Llama 3-family header/eot chat rendering.

use anyhow::Result;
use std::fmt::Write as _;

use crate::ChatTurn;

pub(crate) fn render_optional_system(system: Option<&str>, history: &[ChatTurn]) -> Result<String> {
    let mut out = String::new();
    out.push_str("<|begin_of_text|>");

    if let Some(sys) = system {
        write!(out, "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>", sys)?;
    }

    append_history_and_assistant_marker(&mut out, history)?;
    Ok(out)
}

pub(crate) fn render_with_default_system(
    system: Option<&str>,
    default_system: &'static str,
    history: &[ChatTurn],
) -> Result<String> {
    let mut out = String::new();
    out.push_str("<|begin_of_text|>");

    write!(
        out,
        "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
        system.unwrap_or(default_system)
    )?;

    append_history_and_assistant_marker(&mut out, history)?;
    Ok(out)
}

fn append_history_and_assistant_marker(out: &mut String, history: &[ChatTurn]) -> Result<()> {
    for turn in history {
        write!(
            out,
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            turn.role.as_str(),
            turn.text
        )?;
    }

    write!(out, "<|start_header_id|>assistant<|end_header_id|>\n\n")?;
    Ok(())
}
