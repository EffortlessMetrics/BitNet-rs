use anyhow::{Context, Result, bail};
use bitnet_prompt_templates::TemplateType;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

const CRITICAL_NOT_CLAIMS: &[&str] = &[
    "selected_attention_residency",
    "resident_kv_decode",
    "attention_scores_residency",
    "softmax_residency",
    "attention_value_mix_residency",
    "full_support_op_residency",
    "full_device_residency",
    "completion",
];

#[derive(Debug, Deserialize)]
struct ProfileTable {
    profile: Vec<BenchProfile>,
}

#[derive(Debug, Deserialize)]
struct BenchProfile {
    id: String,
    #[serde(default)]
    prompt_tokens: Option<usize>,
    #[serde(default)]
    decode_tokens: Option<usize>,
    #[serde(default)]
    max_new_tokens: Option<usize>,
}

pub fn profile_cli_plan(
    model_contract: &Path,
    profiles: &Path,
    profile_id: &str,
    backend: &str,
    device_slug: &str,
    kernel_route: &str,
    output: Option<&Path>,
    format: &str,
) -> Result<()> {
    let contract = read_yaml(model_contract)?;
    let profile_table = read_profiles(profiles)?;
    let profile = profile_table
        .profile
        .iter()
        .find(|profile| profile.id == profile_id)
        .with_context(|| format!("profile {profile_id} not found in {}", profiles.display()))?;
    let target_prompt_tokens = profile.prompt_tokens.with_context(|| {
        format!("profile {profile_id} does not define prompt_tokens for CLI plan synthesis")
    })?;
    let max_new_tokens = profile.decode_tokens.or(profile.max_new_tokens).unwrap_or(64);
    let model_path = str_at(&contract, "/local_path").context("contract missing /local_path")?;
    let tokenizer_path =
        str_at(&contract, "/tokenizer/path").context("contract missing /tokenizer/path")?;
    let template_name = str_at(&contract, "/chat_template/name").unwrap_or("llama3-chat");
    let template = template_name
        .parse::<TemplateType>()
        .with_context(|| format!("parsing chat template {template_name}"))?;
    let tokenizer = bitnet_tokenizers::load_tokenizer(Path::new(tokenizer_path))
        .with_context(|| format!("loading tokenizer {}", tokenizer_path))?;

    let user_prompt =
        synthesize_profile_prompt(target_prompt_tokens, template, tokenizer.as_ref())?;
    let formatted_prompt = template.apply(&user_prompt, None);
    let add_bos = template.should_add_bos();
    let parse_special = template.parse_special();
    let token_ids = tokenizer
        .encode(&formatted_prompt, add_bos, parse_special)
        .with_context(|| "tokenizing synthesized profile prompt")?;
    if token_ids.len() != target_prompt_tokens {
        bail!(
            "profile prompt synthesis produced {} tokens, expected {}",
            token_ids.len(),
            target_prompt_tokens
        );
    }

    let cli_stage_output = "target/llm-experience/profile-cli-stage.json";
    let cli_command = build_cli_command(
        backend,
        model_path,
        tokenizer_path,
        &user_prompt,
        max_new_tokens,
        template_name,
        cli_stage_output,
        model_contract,
        kernel_route,
    );

    let mut not_claims = vec![
        "profile_cli_plan_proves_quality",
        "profile_cli_plan_promotes_benchmark_claim",
        "profile_cli_plan_promotes_residency",
    ];
    not_claims.extend_from_slice(CRITICAL_NOT_CLAIMS);

    let plan = json!({
        "diagnostic": "llm_experience_profile_cli_plan",
        "producer": "cargo xtask llm-experience profile-cli-plan",
        "diagnostic_only": true,
        "claimable": false,
        "model_contract": model_contract.display().to_string(),
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "backend": backend,
        "device_slug": device_slug,
        "kernel_route": {
            "route_id": kernel_route,
            "diagnostic_only": true,
            "claimable": false
        },
        "profile": {
            "id": profile.id.as_str(),
            "target_prompt_tokens": target_prompt_tokens,
            "max_new_tokens": max_new_tokens,
        },
        "prompt_identity": {
            "prompt_template": template_name,
            "add_bos": add_bos,
            "parse_special": parse_special,
            "rendered_prompt_sha256": sha256_text(&formatted_prompt),
            "prompt_token_ids_sha256": sha256_token_ids(&token_ids)?,
            "prompt_token_count": token_ids.len(),
        },
        "prompt": user_prompt,
        "cli_command": cli_command,
        "not_claims": not_claims,
    });

    if let Some(output) = output {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(output, serde_json::to_vec_pretty(&plan)?)
            .with_context(|| format!("writing {}", output.display()))?;
    }
    emit_value(&plan, format)
}

fn build_cli_command(
    backend: &str,
    model_path: &str,
    tokenizer_path: &str,
    user_prompt: &str,
    max_new_tokens: usize,
    template_name: &str,
    cli_stage_output: &str,
    model_contract: &Path,
    kernel_route: &str,
) -> Vec<String> {
    vec![
        "cargo".to_string(),
        "run".to_string(),
        "--locked".to_string(),
        "-p".to_string(),
        "bitnet-cli".to_string(),
        "--no-default-features".to_string(),
        "--features".to_string(),
        "cpu".to_string(),
        "--".to_string(),
        "--device".to_string(),
        backend.to_string(),
        "run".to_string(),
        "--model".to_string(),
        model_path.to_string(),
        "--tokenizer".to_string(),
        tokenizer_path.to_string(),
        "--prompt".to_string(),
        user_prompt.to_string(),
        "--max-new-tokens".to_string(),
        max_new_tokens.to_string(),
        "--temperature".to_string(),
        "0.0".to_string(),
        "--greedy".to_string(),
        "--deterministic".to_string(),
        "--strict-tokenizer".to_string(),
        "--strict-loader".to_string(),
        "--prompt-template".to_string(),
        template_name.to_string(),
        "--json-out".to_string(),
        cli_stage_output.to_string(),
        "--proof-model-contract".to_string(),
        model_contract.display().to_string(),
        "--proof-kernel-route".to_string(),
        kernel_route.to_string(),
    ]
}

fn synthesize_profile_prompt(
    target_prompt_tokens: usize,
    template: TemplateType,
    tokenizer: &(dyn bitnet_tokenizers::Tokenizer + Send + Sync),
) -> Result<String> {
    let mut prompt = "Answer this real local model check question in a concise paragraph. Explain why receipt-backed model contracts, route identity, quality gates, and explicit not-claims make a local LLM benchmark trustworthy.".to_string();
    let mut current = count_template_tokens(&prompt, template, tokenizer)?;
    if current > target_prompt_tokens {
        bail!(
            "base profile prompt has {current} tokens, target profile has {target_prompt_tokens}"
        );
    }
    let fillers = [
        " Include the model contract.",
        " Include the tokenizer hash.",
        " Include the prompt token hash.",
        " Include the A770 route.",
        " Include fallback status.",
        " Include quality evidence.",
        " Include load timing.",
        " Include TTFT.",
        " Include input speed.",
        " Include output speed.",
        " Include RSS.",
        " Include VRAM.",
        " Include transfer bytes.",
        " Include kernel counts.",
        " Include history.",
        " Include not-claims.",
        " State the claim boundary.",
        " Keep selected attention deferred.",
        " Keep resident KV unclaimed.",
        " Keep full residency unclaimed.",
        " receipt",
        " route",
        " token",
        " model",
        " proof",
        " quality",
        " benchmark",
        " history",
        " resource",
        " fallback",
        ".",
        " a",
        " the",
        " and",
    ];

    let mut cursor = 0usize;
    while current < target_prompt_tokens {
        let mut chosen: Option<(&str, usize, usize)> = None;
        for offset in 0..fillers.len() {
            let index = (cursor + offset) % fillers.len();
            let filler = fillers[index];
            let candidate = format!("{prompt}{filler}");
            let count = count_template_tokens(&candidate, template, tokenizer)?;
            if count > current && count <= target_prompt_tokens {
                chosen = Some((filler, count, index));
                break;
            }
        }
        if let Some((filler, count, index)) = chosen {
            prompt.push_str(filler);
            current = count;
            cursor = index + 1;
        } else {
            bail!(
                "could not synthesize exact {target_prompt_tokens}-token prompt; stopped at {current}"
            );
        }
    }

    Ok(prompt)
}

fn count_template_tokens(
    user_prompt: &str,
    template: TemplateType,
    tokenizer: &(dyn bitnet_tokenizers::Tokenizer + Send + Sync),
) -> Result<usize> {
    let formatted = template.apply(user_prompt, None);
    Ok(tokenizer
        .encode(&formatted, template.should_add_bos(), template.parse_special())
        .with_context(|| "tokenizing profile prompt candidate")?
        .len())
}

fn read_yaml(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_yaml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn read_profiles(path: &Path) -> Result<ProfileTable> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn str_at<'a>(value: &'a Value, pointer: &str) -> Option<&'a str> {
    value.pointer(pointer).and_then(Value::as_str)
}

fn sha256_text(value: &str) -> String {
    sha256_bytes(value.as_bytes())
}

fn sha256_token_ids(tokens: &[u32]) -> Result<String> {
    Ok(sha256_bytes(&serde_json::to_vec(tokens)?))
}

fn sha256_bytes(value: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value);
    format!("{:x}", hasher.finalize())
}

fn emit_value(value: &Value, format: &str) -> Result<()> {
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(value)?),
        "human" => {
            println!("diagnostic: {}", str_at(value, "/diagnostic").unwrap_or("llm_experience"));
            if let Some(claimable) = value.pointer("/claimable").and_then(Value::as_bool) {
                println!("claimable: {claimable}");
            }
            if let Some(profile) = str_at(value, "/profile/id") {
                println!("profile: {profile}");
            }
            if let Some(count) = value.pointer("/prompt_identity/prompt_token_count") {
                println!("prompt_token_count: {count}");
            }
            println!("not_claims: {}", serde_json::to_string(&value["not_claims"])?);
        }
        other => bail!("unsupported llm-experience output format: {other}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_command_binds_proof_identity() {
        let command = build_cli_command(
            "intel-arc-a770-opencl",
            "models/model.gguf",
            "models/tokenizer.json",
            "prompt",
            64,
            "llama3-chat",
            "target/llm-experience/profile-cli-stage.json",
            Path::new("docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml"),
            "a770.bitnet.i2s.qk256",
        );
        assert!(command.windows(2).any(|args| args == ["--device", "intel-arc-a770-opencl"]));
        assert!(command.iter().any(|arg| arg == "--proof-model-contract"));
        assert!(command.iter().any(|arg| arg == "--proof-kernel-route"));
        assert!(command.iter().any(|arg| arg == "a770.bitnet.i2s.qk256"));
    }

    #[test]
    fn synthesizes_exact_target_when_base_matches() {
        let tokenizer = CountingTokenizer;
        let base = "Answer this real local model check question in a concise paragraph. Explain why receipt-backed model contracts, route identity, quality gates, and explicit not-claims make a local LLM benchmark trustworthy.";
        let target =
            count_template_tokens(base, TemplateType::Raw, &tokenizer).expect("base token count");
        let prompt = synthesize_profile_prompt(target, TemplateType::Raw, &tokenizer)
            .expect("synthesize prompt");
        let count =
            count_template_tokens(&prompt, TemplateType::Raw, &tokenizer).expect("prompt count");
        assert_eq!(count, target);
    }

    struct CountingTokenizer;

    impl bitnet_tokenizers::Tokenizer for CountingTokenizer {
        fn encode(
            &self,
            text: &str,
            add_bos: bool,
            _add_special: bool,
        ) -> bitnet_common::Result<Vec<u32>> {
            let mut tokens = Vec::new();
            if add_bos {
                tokens.push(1);
            }
            tokens.extend((0..text.split_whitespace().count()).map(|index| index as u32 + 2));
            Ok(tokens)
        }

        fn decode(&self, tokens: &[u32]) -> bitnet_common::Result<String> {
            Ok(tokens.iter().map(u32::to_string).collect::<Vec<_>>().join(" "))
        }

        fn vocab_size(&self) -> usize {
            1024
        }

        fn token_to_piece(&self, token: u32) -> Option<String> {
            Some(token.to_string())
        }
    }
}
