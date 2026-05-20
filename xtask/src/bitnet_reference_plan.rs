use anyhow::{Context, Result, bail};
use bitnet_prompt_templates::TemplateType;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const CRITICAL_NOT_CLAIMS: &[&str] = &[
    "selected_attention_residency",
    "resident_kv_decode",
    "attention_scores_residency",
    "softmax_residency",
    "attention_value_mix_residency",
    "full_support_op_residency",
    "full_device_residency",
    "completion",
    "reference_execution_proven",
    "rust_reference_parity_proven",
    "a770_semantic_quality_proven",
];

struct PlanTokenizer {
    tokenizer: std::sync::Arc<dyn bitnet_tokenizers::Tokenizer + Send + Sync>,
    source: String,
    path: Option<String>,
    contract_path: Option<String>,
    rust_cli_tokenizer_arg: Option<String>,
}

#[derive(Debug)]
pub struct ReferencePlanArgs<'a> {
    pub model_contract: &'a Path,
    pub model: Option<&'a Path>,
    pub tokenizer: Option<&'a Path>,
    pub prompt_template: &'a str,
    pub system_prompt: Option<&'a str>,
    pub prompt: &'a str,
    pub max_new_tokens: usize,
    pub reference_exe: Option<&'a Path>,
    pub cpp_root: Option<&'a Path>,
    pub output: Option<&'a Path>,
    pub format: &'a str,
}

pub fn maybe_dispatch_from_env() -> Result<bool> {
    let args = std::env::args().collect::<Vec<_>>();
    maybe_dispatch(&args)
}

fn maybe_dispatch(args: &[String]) -> Result<bool> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-plan") {
        return Ok(false);
    }
    if args[2..].iter().any(|arg| arg == "-h" || arg == "--help") {
        print_help();
        return Ok(true);
    }

    let mut model_contract = PathBuf::from("docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml");
    let mut model: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut prompt_template = "llama3-chat".to_string();
    let mut system_prompt: Option<String> = None;
    let mut prompt = "What is 2+2?".to_string();
    let mut max_new_tokens = 16usize;
    let mut reference_exe: Option<PathBuf> = None;
    let mut cpp_root: Option<PathBuf> = None;
    let mut output = PathBuf::from("target/a770-diagnostic/bitnet-reference-plan.json");
    let mut format = "human".to_string();

    let mut i = 2usize;
    while i < args.len() {
        let key = args[i].as_str();
        i += 1;
        let mut value = || -> Result<String> {
            let value = args.get(i).with_context(|| format!("{key} requires a value"))?.clone();
            i += 1;
            Ok(value)
        };
        match key {
            "--model-contract" => model_contract = PathBuf::from(value()?),
            "--model" => model = Some(PathBuf::from(value()?)),
            "--tokenizer" => tokenizer = Some(PathBuf::from(value()?)),
            "--prompt-template" => prompt_template = value()?,
            "--system-prompt" => system_prompt = Some(value()?),
            "--prompt" => prompt = value()?,
            "--max-new-tokens" => {
                let raw = value()?;
                max_new_tokens =
                    raw.parse().with_context(|| format!("parsing --max-new-tokens {raw}"))?;
            }
            "--reference-exe" => reference_exe = Some(PathBuf::from(value()?)),
            "--cpp-root" => cpp_root = Some(PathBuf::from(value()?)),
            "--output" => output = PathBuf::from(value()?),
            "--format" => format = value()?,
            other => bail!("unknown bitnet-reference-plan option {other}"),
        }
    }

    run(ReferencePlanArgs {
        model_contract: &model_contract,
        model: model.as_deref(),
        tokenizer: tokenizer.as_deref(),
        prompt_template: &prompt_template,
        system_prompt: system_prompt.as_deref(),
        prompt: &prompt,
        max_new_tokens,
        reference_exe: reference_exe.as_deref(),
        cpp_root: cpp_root.as_deref(),
        output: Some(&output),
        format: &format,
    })?;
    Ok(true)
}

fn print_help() {
    println!(
        "Emit a target-local BitNet C++ reference-readiness plan\n\nUsage: xtask.exe bitnet-reference-plan [OPTIONS]\n\nOptions:\n      --model-contract <PATH>      Model contract YAML file [default: docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml]\n      --model <PATH>               Override model path\n      --tokenizer <PATH>           Override tokenizer path\n      --prompt-template <NAME>     Prompt template [default: llama3-chat]\n      --system-prompt <TEXT>       Optional system prompt\n      --prompt <TEXT>              User prompt [default: What is 2+2?]\n      --max-new-tokens <N>         Max new tokens [default: 16]\n      --reference-exe <PATH>       Explicit C++ reference executable path\n      --cpp-root <PATH>            C++ reference checkout/build root\n      --output <PATH>              Output plan JSON [default: target/a770-diagnostic/bitnet-reference-plan.json]\n      --format <human|json>        Output format [default: human]\n  -h, --help                       Print help"
    );
}

pub fn run(args: ReferencePlanArgs<'_>) -> Result<()> {
    let report = build_report(&args)?;
    if let Some(output) = args.output {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(output, serde_json::to_vec_pretty(&report)?)
            .with_context(|| format!("writing {}", output.display()))?;
    }
    emit_report(&report, args.format)
}

fn build_report(args: &ReferencePlanArgs<'_>) -> Result<Value> {
    let contract = read_yaml(args.model_contract)?;
    let model_path = args
        .model
        .map(path_to_string)
        .or_else(|| str_at(&contract, "/local_path").map(ToOwned::to_owned))
        .context("model path missing; pass --model or set /local_path in the model contract")?;
    let plan_tokenizer = resolve_plan_tokenizer(&model_path, args.tokenizer, &contract)?;
    let template = args
        .prompt_template
        .parse::<TemplateType>()
        .with_context(|| format!("parsing prompt template {}", args.prompt_template))?;
    let rendered_prompt = template.apply(args.prompt, args.system_prompt);
    let add_bos = template.should_add_bos();
    let parse_special = template.parse_special();
    let token_ids = plan_tokenizer
        .tokenizer
        .encode(&rendered_prompt, add_bos, parse_special)
        .with_context(|| "tokenizing rendered prompt with contract tokenizer")?;

    let candidates = reference_candidates(args.reference_exe, args.cpp_root);
    let selected = candidates.iter().find(|candidate| candidate.exists);
    let reference_ready = selected.is_some();
    let mut blocked_reasons = Vec::new();
    if !reference_ready {
        blocked_reasons.push("reference_executable_missing");
    }
    if !Path::new(&model_path).exists() {
        blocked_reasons.push("model_file_missing");
    }
    if let Some(path) = plan_tokenizer.path.as_deref()
        && !Path::new(path).exists()
    {
        blocked_reasons.push("tokenizer_file_missing");
    }

    let reference_argv = selected.map(|candidate| {
        vec![
            candidate.path.clone(),
            "-m".to_string(),
            model_path.clone(),
            "-p".to_string(),
            rendered_prompt.clone(),
            "-n".to_string(),
            args.max_new_tokens.to_string(),
            "--temp".to_string(),
            "0".to_string(),
            "--seed".to_string(),
            "0".to_string(),
        ]
    });

    Ok(json!({
        "schema_version": 1,
        "diagnostic": "bitnet_reference_plan",
        "producer": "cargo xtask bitnet-reference-plan",
        "created_at": chrono::Utc::now().to_rfc3339(),
        "diagnostic_only": true,
        "promotion_allowed": false,
        "claim_allowed": false,
        "classification": "diagnostic_only",
        "model": {
            "contract": path_to_string(args.model_contract),
            "model_id": str_at(&contract, "/model_id").unwrap_or("unknown-model"),
            "model_path": model_path,
            "model_exists": Path::new(&model_path).exists(),
            "tokenizer_path": plan_tokenizer.path.as_deref(),
            "tokenizer_source": plan_tokenizer.source.as_str(),
            "tokenizer_exists": plan_tokenizer
                .path
                .as_deref()
                .is_some_and(|path| Path::new(path).exists()),
            "contract_tokenizer_path": plan_tokenizer.contract_path.as_deref(),
            "contract_tokenizer_exists": plan_tokenizer
                .contract_path
                .as_deref()
                .is_some_and(|path| Path::new(path).exists()),
            "weights_sha256": str_at(&contract, "/sha256").unwrap_or(""),
            "tokenizer_sha256": str_at(&contract, "/tokenizer/sha256").unwrap_or(""),
        },
        "prompt_identity": {
            "prompt_template": args.prompt_template,
            "tokenizer_source": plan_tokenizer.source.as_str(),
            "system_prompt_present": args.system_prompt.is_some(),
            "rendered_prompt_sha256": sha256_text(&rendered_prompt),
            "prompt_token_ids_sha256": sha256_token_ids(&token_ids)?,
            "prompt_token_count": token_ids.len(),
            "add_bos": add_bos,
            "parse_special": parse_special,
            "max_new_tokens": args.max_new_tokens,
        },
        "reference": {
            "backend": "bitnet.cpp_or_llama.cpp_cli",
            "ready": reference_ready,
            "selected_executable": selected.map(|candidate| candidate.path.as_str()),
            "candidate_executables": candidates.iter().map(|candidate| {
                json!({
                    "path": candidate.path,
                    "source": candidate.source,
                    "exists": candidate.exists,
                })
            }).collect::<Vec<_>>(),
            "command_argv": reference_argv,
            "command_policy": "uses_rendered_prompt_text_for_template_parity; token parity still must be verified against reference output",
            "setup_command_pwsh": "cargo run --locked -p xtask -- setup-cpp-auto --emit=pwsh",
        },
        "rust_commands": {
            "cpu_argv": rust_cli_argv(
                "cpu",
                &model_path,
                plan_tokenizer.rust_cli_tokenizer_arg.as_deref(),
                args.prompt_template,
                args.system_prompt,
                args.prompt,
                args.max_new_tokens,
                "target/a770-diagnostic/reference-plan-cpu.json"
            ),
            "a770_argv": rust_cli_argv(
                "intel-arc-a770-opencl",
                &model_path,
                plan_tokenizer.rust_cli_tokenizer_arg.as_deref(),
                args.prompt_template,
                args.system_prompt,
                args.prompt,
                args.max_new_tokens,
                "target/a770-diagnostic/reference-plan-a770.json"
            ),
        },
        "decision": {
            "reference_required_before_math_change": true,
            "next_when_reference_ready": "run reference command, compare token ids/top-k logits with Rust CPU and strict A770 receipts",
            "current_blocked_reasons": blocked_reasons,
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    }))
}

fn resolve_plan_tokenizer(
    model_path: &str,
    tokenizer_override: Option<&Path>,
    contract: &Value,
) -> Result<PlanTokenizer> {
    let contract_path = str_at(contract, "/tokenizer/path").map(ToOwned::to_owned);
    if let Some(path) = tokenizer_override {
        let path_string = path_to_string(path);
        let tokenizer = bitnet_tokenizers::load_tokenizer(path)
            .with_context(|| format!("loading explicit tokenizer {path_string}"))?;
        return Ok(PlanTokenizer {
            tokenizer,
            source: "explicit".to_string(),
            path: Some(path_string.clone()),
            contract_path,
            rust_cli_tokenizer_arg: Some(path_string),
        });
    }

    let model = Path::new(model_path);
    if model.extension().and_then(|ext| ext.to_str()) == Some("gguf")
        && let Ok(resolution) = bitnet_tokenizers::auto::resolve_tokenizer(model, None, true)
    {
        let source = resolution.source.as_str().to_string();
        let path = resolution.path.as_deref().map(path_to_string);
        let rust_cli_tokenizer_arg =
            if resolution.source == bitnet_tokenizers::auto::TokenizerSource::GgufMetadata {
                None
            } else {
                path.clone()
            };
        return Ok(PlanTokenizer {
            tokenizer: resolution.tokenizer,
            source,
            path,
            contract_path,
            rust_cli_tokenizer_arg,
        });
    }

    let tokenizer_path = contract_path.clone().context(
        "tokenizer path missing; pass --tokenizer or set /tokenizer/path in the model contract",
    )?;
    let tokenizer = bitnet_tokenizers::load_tokenizer(Path::new(&tokenizer_path))
        .with_context(|| format!("loading contract tokenizer {tokenizer_path}"))?;
    Ok(PlanTokenizer {
        tokenizer,
        source: "contract_tokenizer".to_string(),
        path: Some(tokenizer_path.clone()),
        contract_path,
        rust_cli_tokenizer_arg: Some(tokenizer_path),
    })
}

#[derive(Debug)]
struct Candidate {
    path: String,
    source: String,
    exists: bool,
}

fn reference_candidates(reference_exe: Option<&Path>, cpp_root: Option<&Path>) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    if let Some(path) = reference_exe {
        push_candidate(&mut candidates, path.to_path_buf(), "explicit --reference-exe");
    }
    for env in ["BITNET_REFERENCE_EXE", "BITNET_CPP_EXE", "LLAMA_CPP_EXE"] {
        if let Ok(value) = std::env::var(env) {
            push_candidate(&mut candidates, PathBuf::from(value), env);
        }
    }

    let mut roots = Vec::new();
    if let Some(root) = cpp_root {
        roots.push((root.to_path_buf(), "explicit --cpp-root".to_string()));
    }
    for env in ["BITNET_CPP_DIR", "LLAMA_CPP_DIR"] {
        if let Ok(value) = std::env::var(env) {
            roots.push((PathBuf::from(value), env.to_string()));
        }
    }
    roots.push((PathBuf::from("target/external/BitNet"), "target/external/BitNet".to_string()));
    if let Some(home) = dirs::home_dir() {
        roots.push((home.join(".cache/bitnet_cpp"), "$HOME/.cache/bitnet_cpp".to_string()));
    }

    for (root, source) in roots {
        for subdir in ["build/bin", "build", "bin", ""] {
            for name in executable_names() {
                push_candidate(&mut candidates, root.join(subdir).join(name), &source);
            }
        }
    }

    let mut seen = HashSet::new();
    candidates.retain(|candidate| seen.insert(candidate.path.clone()));
    candidates
}

fn push_candidate(candidates: &mut Vec<Candidate>, path: PathBuf, source: &str) {
    let exists = path.is_file();
    candidates.push(Candidate { path: path_to_string(&path), source: source.to_string(), exists });
}

fn executable_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["llama-cli.exe", "main.exe", "bitnet-cli.exe", "bitnet-lut.exe"]
    } else {
        &["llama-cli", "main", "bitnet-cli", "bitnet-lut"]
    }
}

fn rust_cli_argv(
    device: &str,
    model: &str,
    tokenizer: Option<&str>,
    prompt_template: &str,
    system_prompt: Option<&str>,
    prompt: &str,
    max_new_tokens: usize,
    json_out: &str,
) -> Vec<String> {
    let features =
        if device.contains("a770") || device.contains("opencl") { "opencl" } else { "cpu" };
    let mut argv = vec![
        "cargo".to_string(),
        "run".to_string(),
        "--locked".to_string(),
        "-p".to_string(),
        "bitnet-cli".to_string(),
        "--no-default-features".to_string(),
        "--features".to_string(),
        features.to_string(),
        "--".to_string(),
        "run".to_string(),
        "--device".to_string(),
        device.to_string(),
        "--model".to_string(),
        model.to_string(),
        "--model-format".to_string(),
        "gguf".to_string(),
        "--prompt-template".to_string(),
        prompt_template.to_string(),
    ];
    if let Some(tokenizer) = tokenizer {
        argv.push("--tokenizer".to_string());
        argv.push(tokenizer.to_string());
    }
    if let Some(system_prompt) = system_prompt {
        argv.push("--system-prompt".to_string());
        argv.push(system_prompt.to_string());
    }
    argv.extend([
        "--prompt".to_string(),
        prompt.to_string(),
        "--max-new-tokens".to_string(),
        max_new_tokens.to_string(),
        "--greedy".to_string(),
        "--deterministic".to_string(),
        "--strict-loader".to_string(),
        "--strict-tokenizer".to_string(),
        "--strict-backend".to_string(),
        "--no-warnings".to_string(),
        "--json-out".to_string(),
        json_out.to_string(),
    ]);
    argv
}

fn read_yaml(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_yaml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn str_at<'a>(value: &'a Value, pointer: &str) -> Option<&'a str> {
    value.pointer(pointer).and_then(Value::as_str)
}

fn path_to_string(path: &Path) -> String {
    path.display().to_string()
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

fn emit_report(value: &Value, format: &str) -> Result<()> {
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(value)?),
        "human" => {
            println!(
                "diagnostic: {}",
                str_at(value, "/diagnostic").unwrap_or("bitnet_reference_plan")
            );
            println!(
                "classification: {}",
                str_at(value, "/classification").unwrap_or("diagnostic_only")
            );
            println!(
                "reference_ready: {}",
                value.pointer("/reference/ready").and_then(Value::as_bool).unwrap_or(false)
            );
            if let Some(reasons) = value.pointer("/decision/current_blocked_reasons") {
                println!("blocked_reasons: {}", serde_json::to_string(reasons)?);
            }
            println!("not_claims: {}", serde_json::to_string(&value["not_claims"])?);
        }
        other => bail!("unsupported bitnet-reference-plan output format: {other}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_cli_argv_keeps_a770_and_cpu_feature_routes_separate() {
        let cpu = rust_cli_argv("cpu", "m.gguf", Some("tok.json"), "raw", None, "x", 1, "cpu.json");
        let a770 = rust_cli_argv(
            "intel-arc-a770-opencl",
            "m.gguf",
            Some("tok.json"),
            "raw",
            None,
            "x",
            1,
            "a770.json",
        );
        assert!(cpu.windows(2).any(|args| args == ["--features", "cpu"]));
        assert!(a770.windows(2).any(|args| args == ["--features", "opencl"]));
        assert!(a770.windows(2).any(|args| args == ["--device", "intel-arc-a770-opencl"]));
    }

    #[test]
    fn rust_cli_argv_can_use_embedded_gguf_tokenizer() {
        let argv = rust_cli_argv(
            "cpu",
            "m.gguf",
            None,
            "llama3-chat",
            None,
            "What is 2+2?",
            1,
            "cpu.json",
        );

        assert!(!argv.iter().any(|arg| arg == "--tokenizer"));
        assert!(argv.iter().any(|arg| arg == "--strict-tokenizer"));
    }

    #[test]
    fn reference_candidates_include_explicit_path() {
        let path = Path::new("target/reference/llama-cli.exe");
        let candidates = reference_candidates(Some(path), None);
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.path.ends_with("target/reference/llama-cli.exe"))
        );
    }
}
