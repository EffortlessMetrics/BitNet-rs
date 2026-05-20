//! Serializable contracts for defining, invoking, and returning tool calls.

use serde::{Deserialize, Serialize};

/// Describes a single parameter accepted by a tool.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    pub param_type: String,
    pub description: String,
    pub required: bool,
    pub default_value: Option<String>,
}

/// A tool that can be offered to the model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
}

/// A structured tool invocation produced (or consumed) by the model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    /// JSON-encoded arguments string.
    pub arguments: String,
}

/// The result returned after executing a tool call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
    pub is_error: bool,
}
