//! Chat template auto-detection helpers.

/// Return true only when we can be confident the model expects LLaMA-3 chat formatting.
#[must_use]
pub fn looks_like_llama3_chat(
    tokenizer_name: Option<&str>,
    chat_template_jinja: Option<&str>,
) -> bool {
    let name_hit = tokenizer_name
        .map(|s| s.to_ascii_lowercase())
        .map(|n| n.contains("llama") && n.contains("3"))
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
    fn detects_from_tokenizer_name_case_insensitive() {
        assert!(looks_like_llama3_chat(Some("Meta-LLAMA-3.1-Instruct"), None));
    }

    #[test]
    fn detects_from_template_markers() {
        assert!(looks_like_llama3_chat(
            None,
            Some("{% for m in messages %}<|start_header_id|>user<|eot_id|>{% endfor %}")
        ));
    }

    #[test]
    fn ignores_non_llama3_signals() {
        assert!(!looks_like_llama3_chat(Some("Mistral-7B-Instruct"), Some("<s>{{ message }}</s>")));
    }
}
