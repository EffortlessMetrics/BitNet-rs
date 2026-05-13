//! Gemma-style start/end-of-turn chat rendering.

use anyhow::Result;
use std::fmt::Write as _;

use crate::{ChatRole, ChatTurn};

pub(crate) fn render(system: Option<&str>, history: &[ChatTurn]) -> Result<String> {
    let mut out = String::new();
    let mut system_prepended = false;

    for turn in history {
        let role = match turn.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "model",
            ChatRole::System => continue,
        };
        writeln!(out, "<start_of_turn>{}", role)?;
        if role == "user" && !system_prepended {
            if let Some(sys) = system {
                writeln!(out, "{}\n", sys)?;
            }
            system_prepended = true;
        }
        writeln!(out, "{}<end_of_turn>", turn.text)?;
    }

    if !system_prepended && let Some(sys) = system {
        writeln!(out, "<start_of_turn>user\n{}<end_of_turn>", sys)?;
    }

    writeln!(out, "<start_of_turn>model")?;
    Ok(out)
}
