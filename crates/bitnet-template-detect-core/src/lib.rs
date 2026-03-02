//! Prompt-template auto-detection primitives shared across crates.

/// Return true only when we can be confident the model expects LLaMA-3 chat formatting.
#[must_use]
pub fn looks_like_llama3_chat(
    tokenizer_name: Option<&str>,
    chat_template_jinja: Option<&str>,
) -> bool {
    let name_hit = tokenizer_name
        .map(|s| s.to_ascii_lowercase())
        .map(|n| n.contains("llama") && n.contains('3'))
        .unwrap_or(false);

    let tmpl_hit = chat_template_jinja
        .map(|j| j.contains("<|start_header_id|>") && j.contains("<|eot_id|>"))
        .unwrap_or(false);

    name_hit || tmpl_hit
}

#[cfg(test)]
mod tests {
    use super::looks_like_llama3_chat;

    #[test]
    fn detects_llama3_from_tokenizer_name() {
        assert!(looks_like_llama3_chat(Some("meta-llama-3.1"), None));
    }

    #[test]
    fn detects_llama3_from_chat_template_tokens() {
        assert!(looks_like_llama3_chat(
            None,
            Some("<|start_header_id|>user<|end_header_id|>x<|eot_id|>")
        ));
    }

    #[test]
    fn ignores_non_llama_templates() {
        assert!(!looks_like_llama3_chat(Some("mistral"), Some("{{ prompt }}")));
    }
}
