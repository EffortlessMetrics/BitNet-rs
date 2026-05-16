//! Prompt authority audit helpers.
//!
//! This module owns prompt-template comparison, tokenizer metadata extraction,
//! and reference-parity JSON assembly for the `prompt-authority-audit` command.

pub(crate) fn prompt_audit_reference_parity_json(
    reference_source: Option<String>,
    reference_rendered_prompt: Option<String>,
    reference_prompt_ids: &[u32],
    bitnet_rendered_prompt: &str,
    bitnet_prompt_ids: Option<&[u32]>,
) -> serde_json::Value {
    let rendered_prompt_available = reference_rendered_prompt.is_some();
    let prompt_token_ids_available = !reference_prompt_ids.is_empty();
    let rendered_prompt_match =
        reference_rendered_prompt.as_deref().map(|reference| reference == bitnet_rendered_prompt);
    let prompt_token_ids_match = if prompt_token_ids_available {
        bitnet_prompt_ids.map(|bitnet_ids| reference_prompt_ids == bitnet_ids)
    } else {
        None
    };
    let first_rendered_prompt_mismatch = reference_rendered_prompt
        .as_deref()
        .and_then(|reference| first_string_mismatch(reference, bitnet_rendered_prompt));
    let first_prompt_token_id_mismatch = if prompt_token_ids_available {
        bitnet_prompt_ids
            .and_then(|bitnet_ids| first_token_mismatch(reference_prompt_ids, bitnet_ids))
    } else {
        None
    };
    let passed = rendered_prompt_match == Some(true) && prompt_token_ids_match == Some(true);

    serde_json::json!({
        "available": rendered_prompt_available || prompt_token_ids_available,
        "source": reference_source.unwrap_or_else(|| "unspecified_external_reference".to_string()),
        "compared_against": "metadata_authority",
        "reference_rendered_prompt_available": rendered_prompt_available,
        "reference_prompt_token_ids_available": prompt_token_ids_available,
        "rendered_prompt_match": rendered_prompt_match,
        "prompt_token_ids_match": prompt_token_ids_match,
        "first_rendered_prompt_mismatch_index": first_rendered_prompt_mismatch,
        "first_prompt_token_id_mismatch_index": first_prompt_token_id_mismatch,
        "passed": passed,
    })
}

#[derive(Debug)]
pub(crate) struct TokenizerJsonPromptMetadata {
    pub(crate) family: Option<String>,
    pub(crate) chat_template: Option<String>,
}

pub(crate) fn read_resolved_tokenizer_json_prompt_metadata(
    source: bitnet_tokenizers::auto::TokenizerSource,
    path: Option<&std::path::Path>,
) -> Option<TokenizerJsonPromptMetadata> {
    match source {
        bitnet_tokenizers::auto::TokenizerSource::Explicit
        | bitnet_tokenizers::auto::TokenizerSource::Sibling => {
            path.and_then(read_tokenizer_json_prompt_metadata)
        }
        bitnet_tokenizers::auto::TokenizerSource::GgufMetadata
        | bitnet_tokenizers::auto::TokenizerSource::CompatibilityFallback => None,
    }
}

pub(crate) fn read_tokenizer_json_prompt_metadata(
    path: &std::path::Path,
) -> Option<TokenizerJsonPromptMetadata> {
    if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
        return None;
    }
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    let family = value
        .get("tokenizer_class")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            value
                .get("model")
                .and_then(|model| model.get("type"))
                .and_then(serde_json::Value::as_str)
        })
        .map(str::to_string);
    let chat_template =
        value.get("chat_template").and_then(serde_json::Value::as_str).map(str::to_string);
    Some(TokenizerJsonPromptMetadata { family, chat_template })
}

pub(crate) fn prompt_audit_current_default_template(
    model_path: &std::path::Path,
    tokenizer_path: Option<&std::path::Path>,
    tokenizer: &dyn bitnet_tokenizers::Tokenizer,
) -> bitnet_inference::TemplateType {
    let path_template =
        bitnet_inference::TemplateType::detect_from_paths(Some(model_path), tokenizer_path);
    if matches!(path_template, bitnet_inference::TemplateType::BitnetCppAnswer) {
        bitnet_inference::TemplateType::BitnetCppAnswer
    } else if tokenizer.token_to_id("<|eot_id|>").is_some() {
        bitnet_inference::TemplateType::Llama3Chat
    } else {
        bitnet_inference::TemplateType::Instruct
    }
}

pub(crate) fn prompt_audit_variant_json(
    label: &str,
    template_type: bitnet_inference::TemplateType,
    template_source: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    tokenizer: &dyn bitnet_tokenizers::Tokenizer,
) -> (serde_json::Value, Option<Vec<u32>>, String) {
    let rendered_prompt = template_type.apply(prompt, system_prompt);
    let add_bos = template_type.should_add_bos();
    let parse_special = template_type.parse_special();
    let encoded = tokenizer.encode(&rendered_prompt, add_bos, parse_special);
    let (ids, error) = match encoded {
        Ok(ids) => (Some(ids), None),
        Err(error) => (None, Some(error.to_string())),
    };
    let entry = serde_json::json!({
        "mode": label,
        "prompt_policy": {
            "template_type": template_type.to_string(),
            "template_source": template_source,
            "rendered_prompt": rendered_prompt,
            "rendered_sha256": crate::compute_sha256_bytes(rendered_prompt.as_bytes()),
            "add_bos": add_bos,
            "add_eos": false,
            "parse_special": parse_special,
            "add_generation_prompt": !matches!(template_type, bitnet_inference::TemplateType::Raw),
        },
        "tokens": {
            "prompt_token_ids": ids.clone().unwrap_or_default(),
            "prompt_token_count": ids.as_ref().map(Vec::len),
            "encode_error": error,
        }
    });
    (entry, ids, rendered_prompt)
}

pub(crate) fn prompt_audit_classification(
    current_rendered: &str,
    metadata_rendered: &str,
    current_ids: &Option<Vec<u32>>,
    metadata_ids: &Option<Vec<u32>>,
) -> (&'static str, Vec<String>, Option<usize>) {
    let mut notes = Vec::new();
    if current_rendered != metadata_rendered {
        notes.push("current_default_rendered_prompt_differs_from_metadata_authority".to_string());
        return ("prompt", notes, first_string_mismatch(current_rendered, metadata_rendered));
    }
    match (current_ids, metadata_ids) {
        (Some(left), Some(right)) if left != right => {
            notes.push(
                "current_default_prompt_token_ids_differ_from_metadata_authority".to_string(),
            );
            ("token_ids", notes, first_token_mismatch(left, right))
        }
        (Some(_), Some(_)) => {
            notes.push("current_default_and_metadata_authority_prompt_tokens_match".to_string());
            notes.push(
                "model_inference_not_run; first-token logits remain unclassified".to_string(),
            );
            ("unknown", notes, None)
        }
        _ => {
            notes.push("one_or_more_prompt_variants_failed_to_encode".to_string());
            ("token_ids", notes, None)
        }
    }
}

pub(crate) fn first_token_mismatch(left: &[u32], right: &[u32]) -> Option<usize> {
    let shared = left.len().min(right.len());
    (0..shared).find(|idx| left[*idx] != right[*idx]).or(if left.len() != right.len() {
        Some(shared)
    } else {
        None
    })
}

pub(crate) fn first_string_mismatch(left: &str, right: &str) -> Option<usize> {
    let mut left_chars = left.chars();
    let mut right_chars = right.chars();
    let mut index = 0;
    loop {
        match (left_chars.next(), right_chars.next()) {
            (Some(left), Some(right)) if left == right => index += 1,
            (Some(_), Some(_)) | (Some(_), None) | (None, Some(_)) => return Some(index),
            (None, None) => return None,
        }
    }
}
