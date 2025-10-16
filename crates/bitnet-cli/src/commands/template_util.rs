//! Template selection utilities (shared by chat + tests)

/// Return true only when we can be confident the model expects LLaMA-3 chat formatting.
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
