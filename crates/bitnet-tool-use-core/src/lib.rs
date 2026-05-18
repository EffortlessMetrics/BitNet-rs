//! Reusable tool-calling contracts and parsing/detection helpers.
//!
//! This crate intentionally contains only pure data contracts and lightweight
//! parsing/format detection logic so it can be shared by higher-level crates.

mod contracts;
mod detection;
mod parsing;

pub use contracts::{ToolCall, ToolDefinition, ToolParameter, ToolResult};
pub use detection::detect_tool_format;
pub use parsing::{ToolUseFormat, parse_tool_call};
