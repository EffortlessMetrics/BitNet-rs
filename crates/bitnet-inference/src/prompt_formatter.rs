//! Prompt template formatting for different model architectures.
//!
//! Format user/system/assistant messages into model-specific prompt
//! templates (instruct, chat, raw, custom).

/// Role in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// A single message in a conversation.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }
}

/// Prompt template format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateFormat {
    /// ChatML format: <|im_start|>role\ncontent<|im_end|>
    ChatMl,
    /// LLaMA-3 instruct: <|begin_of_text|><|start_header_id|>...
    Llama3,
    /// Phi instruct: <|system|>\n...<|end|>\n<|user|>...
    Phi,
    /// Simple instruct: ### System:\n...\n### User:\n...
    SimpleInstruct,
    /// Raw: just concatenate content.
    Raw,
}

impl TemplateFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ChatMl => "chatml",
            Self::Llama3 => "llama3",
            Self::Phi => "phi",
            Self::SimpleInstruct => "simple_instruct",
            Self::Raw => "raw",
        }
    }
}

/// Format messages using the specified template.
pub fn format_prompt(messages: &[Message], template: &TemplateFormat) -> String {
    match template {
        TemplateFormat::ChatMl => format_chatml(messages),
        TemplateFormat::Llama3 => format_llama3(messages),
        TemplateFormat::Phi => format_phi(messages),
        TemplateFormat::SimpleInstruct => format_simple(messages),
        TemplateFormat::Raw => format_raw(messages),
    }
}

fn format_chatml(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role.as_str(), msg.content));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

fn format_llama3(messages: &[Message]) -> String {
    let mut out = String::from("<|begin_of_text|>");
    for msg in messages {
        out.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            msg.role.as_str(),
            msg.content
        ));
    }
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    out
}

fn format_phi(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&format!("<|{}|>\n{}<|end|>\n", msg.role.as_str(), msg.content));
    }
    out.push_str("<|assistant|>\n");
    out
}

fn format_simple(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        let label = match msg.role {
            Role::System => "### System",
            Role::User => "### User",
            Role::Assistant => "### Assistant",
        };
        out.push_str(&format!("{label}:\n{}\n\n", msg.content));
    }
    out.push_str("### Assistant:\n");
    out
}

fn format_raw(messages: &[Message]) -> String {
    messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n")
}

/// Auto-detect the best template format for a model name.
pub fn detect_template(model_name: &str) -> TemplateFormat {
    let lower = model_name.to_lowercase();
    if lower.contains("phi") {
        TemplateFormat::Phi
    } else if lower.contains("llama-3") || lower.contains("llama3") {
        TemplateFormat::Llama3
    } else if lower.contains("qwen") || lower.contains("chatml") {
        TemplateFormat::ChatMl
    } else if lower.contains("bitnet") {
        TemplateFormat::ChatMl
    } else {
        TemplateFormat::SimpleInstruct
    }
}

/// Count tokens approximately (word-based estimate).
pub fn estimate_token_count(text: &str) -> usize {
    // Rough estimate: ~4 chars per token
    (text.len() + 3) / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chatml_format() {
        let msgs = vec![Message::user("Hello")];
        let result = format_prompt(&msgs, &TemplateFormat::ChatMl);
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("Hello"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_llama3_format() {
        let msgs = vec![Message::system("You are helpful"), Message::user("Hi")];
        let result = format_prompt(&msgs, &TemplateFormat::Llama3);
        assert!(result.starts_with("<|begin_of_text|>"));
        assert!(result.contains("<|start_header_id|>system"));
        assert!(result.contains("<|eot_id|>"));
    }

    #[test]
    fn test_phi_format() {
        let msgs = vec![Message::user("Hello")];
        let result = format_prompt(&msgs, &TemplateFormat::Phi);
        assert!(result.contains("<|user|>"));
        assert!(result.contains("<|end|>"));
        assert!(result.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_simple_instruct() {
        let msgs = vec![Message::user("Hello")];
        let result = format_prompt(&msgs, &TemplateFormat::SimpleInstruct);
        assert!(result.contains("### User:"));
        assert!(result.ends_with("### Assistant:\n"));
    }

    #[test]
    fn test_raw_format() {
        let msgs = vec![Message::user("Hello"), Message::assistant("Hi")];
        let result = format_prompt(&msgs, &TemplateFormat::Raw);
        assert_eq!(result, "Hello\nHi");
    }

    #[test]
    fn test_detect_phi() {
        assert_eq!(detect_template("microsoft/phi-4"), TemplateFormat::Phi);
    }

    #[test]
    fn test_detect_llama3() {
        assert_eq!(detect_template("meta-llama/llama-3-8b"), TemplateFormat::Llama3);
    }

    #[test]
    fn test_detect_qwen() {
        assert_eq!(detect_template("Qwen/Qwen2-7B"), TemplateFormat::ChatMl);
    }

    #[test]
    fn test_detect_bitnet() {
        assert_eq!(detect_template("microsoft/bitnet-b1.58"), TemplateFormat::ChatMl);
    }

    #[test]
    fn test_detect_unknown() {
        assert_eq!(detect_template("some-model"), TemplateFormat::SimpleInstruct);
    }

    #[test]
    fn test_estimate_tokens() {
        assert!(estimate_token_count("Hello world") > 0);
        assert!(estimate_token_count("This is a longer sentence for testing") > 5);
    }

    #[test]
    fn test_message_constructors() {
        let sys = Message::system("test");
        assert_eq!(sys.role, Role::System);
        let usr = Message::user("test");
        assert_eq!(usr.role, Role::User);
        let ast = Message::assistant("test");
        assert_eq!(ast.role, Role::Assistant);
    }
}
