use anyhow::{Context, Result};
use bitnet_inference::TemplateType;
use bitnet_tokenizers::Tokenizer;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
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

#[derive(Clone, Debug, Default)]
pub(crate) struct PromptGenerationParams {
    pub max_new_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub greedy: Option<bool>,
    pub deterministic: Option<bool>,
    pub threads: Option<usize>,
    pub qwen_no_think: Option<bool>,
    pub fixed_token_count: Option<bool>,
    pub stream: Option<bool>,
}

pub(crate) struct PromptGenerationIdentityInput<'a> {
    pub template_family: &'a str,
    pub template_source: &'a str,
    pub tokenizer_source: Option<&'a str>,
    pub tokenizer_authority: Option<&'a str>,
    pub tokenizer_sha256: Option<&'a str>,
    pub tokenizer_strict: Option<bool>,
    pub manual_stop_sequences: &'a [String],
    pub stop_sequences: &'a [String],
    pub manual_stop_token_ids: &'a [u32],
    pub stop_token_ids: &'a [u32],
    pub stop_string_window: Option<usize>,
    pub stop_policy: &'a str,
    pub generation_params: PromptGenerationParams,
}

pub(crate) fn prompt_generation_identity(
    input: PromptGenerationIdentityInput<'_>,
) -> serde_json::Value {
    let template = json!({
        "family": input.template_family,
        "source": input.template_source,
    });
    let tokenizer = json!({
        "source": input.tokenizer_source,
        "authority": input.tokenizer_authority,
        "sha256": input.tokenizer_sha256,
        "strict": input.tokenizer_strict,
    });
    let stop_criteria = json!({
        "policy": input.stop_policy,
        "manual_stop_sequences": input.manual_stop_sequences,
        "stop_sequences": input.stop_sequences,
        "manual_stop_token_ids": input.manual_stop_token_ids,
        "stop_token_ids": input.stop_token_ids,
        "stop_string_window": input.stop_string_window,
    });
    let generation_parameters = json!({
        "max_new_tokens": input.generation_params.max_new_tokens,
        "temperature": input.generation_params.temperature,
        "top_k": input.generation_params.top_k,
        "top_p": input.generation_params.top_p,
        "repetition_penalty": input.generation_params.repetition_penalty,
        "seed": input.generation_params.seed,
        "greedy": input.generation_params.greedy,
        "deterministic": input.generation_params.deterministic,
        "threads": input.generation_params.threads,
        "qwen_no_think": input.generation_params.qwen_no_think,
        "fixed_token_count": input.generation_params.fixed_token_count,
        "stream": input.generation_params.stream,
    });

    let mut identity = json!({
        "schema_version": "1.0.0",
        "template_family": input.template_family,
        "template_sha256": sha256_json(&template),
        "template": template,
        "tokenizer_authority": tokenizer,
        "stop_criteria": stop_criteria,
        "stop_criteria_sha256": sha256_json(&stop_criteria),
        "stop_sequences_sha256": sha256_json(input.stop_sequences),
        "stop_token_ids_sha256": sha256_json(input.stop_token_ids),
        "generation_parameters": generation_parameters,
        "generation_params_sha256": sha256_json(&generation_parameters),
    });
    let identity_sha256 = sha256_json(&identity);
    if let Some(object) = identity.as_object_mut() {
        object.insert("identity_sha256".to_string(), Value::String(identity_sha256));
    }
    identity
}

fn sha256_json<T: serde::Serialize + ?Sized>(value: &T) -> String {
    match serde_json::to_vec(value) {
        Ok(bytes) => sha256_hex(&bytes),
        Err(error) => sha256_hex(error.to_string().as_bytes()),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
