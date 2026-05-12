use anyhow::{Context, Result};
use bitnet_inference::TemplateType;
use bitnet_tokenizers::Tokenizer;
use std::path::Path;

/// Parses a user-selected prompt template while preserving `auto` for later
/// tokenizer-aware detection.
pub(crate) fn parse_prompt_template(prompt_template: &str) -> Result<TemplateType> {
    if prompt_template == "auto" {
        return Ok(TemplateType::Instruct);
    }

    prompt_template.parse().with_context(|| {
        format!(
            "Invalid prompt template '{}'. Supported: raw, instruct, llama3-chat, bitnetcpp-answer",
            prompt_template
        )
    })
}

/// Resolves `auto` prompt templates after both model and tokenizer are known.
pub(crate) fn resolve_prompt_template(
    prompt_template: &str,
    parsed_template: TemplateType,
    model_path: &Path,
    tokenizer_path: Option<&Path>,
    tokenizer: &dyn Tokenizer,
) -> TemplateType {
    if prompt_template != "auto" {
        return parsed_template;
    }

    let path_template = TemplateType::detect_from_paths(Some(model_path), tokenizer_path);
    if matches!(path_template, TemplateType::BitnetCppAnswer) {
        tracing::debug!("Auto-detected bitnetcpp-answer template (model path matches BitNet)");
        TemplateType::BitnetCppAnswer
    } else if tokenizer.token_to_id("<|eot_id|>").is_some() {
        tracing::debug!("Auto-detected llama3-chat template (tokenizer has <|eot_id|>)");
        TemplateType::Llama3Chat
    } else {
        tracing::debug!("Auto-detected instruct template (fallback)");
        TemplateType::Instruct
    }
}

pub(crate) fn merge_stop_sequences(
    manual_stops: &[String],
    template_type: TemplateType,
) -> Vec<String> {
    let mut all_stop_sequences = manual_stops.to_vec();
    for template_stop in template_type.default_stop_sequences() {
        if !all_stop_sequences.contains(&template_stop) {
            all_stop_sequences.push(template_stop);
        }
    }
    all_stop_sequences
}

pub(crate) fn merge_stop_token_ids(
    manual_stop_ids: &[u32],
    template_type: TemplateType,
    tokenizer: &dyn Tokenizer,
) -> Vec<u32> {
    let mut all_stop_ids = manual_stop_ids.to_vec();
    for template_id in template_type.resolve_stop_token_ids(tokenizer) {
        if !all_stop_ids.contains(&template_id) {
            all_stop_ids.push(template_id);
        }
    }
    all_stop_ids
}

pub(crate) fn bos_policy(explicit_bos: bool, template_type: TemplateType) -> bool {
    explicit_bos || template_type.should_add_bos()
}
