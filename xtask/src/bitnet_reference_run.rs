use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

const CRITICAL_NOT_CLAIMS: &[&str] = &[
    "selected_attention_residency",
    "resident_kv_decode",
    "attention_scores_residency",
    "softmax_residency",
    "attention_value_mix_residency",
    "full_support_op_residency",
    "full_device_residency",
    "completion",
    "reference_generated_token_ids",
    "reference_top_logits",
    "rust_reference_parity_proven",
    "a770_semantic_quality_proven",
];

#[derive(Debug)]
struct ReferenceRunArgs {
    plan: PathBuf,
    output: Option<PathBuf>,
    format: String,
}

#[derive(Debug)]
struct CommandCapture {
    status_code: Option<i32>,
    success: bool,
    stdout: String,
    stderr: String,
    elapsed_ms: f64,
}

pub fn maybe_dispatch_from_env() -> Result<bool> {
    let args = std::env::args().collect::<Vec<_>>();
    maybe_dispatch(&args)
}

fn maybe_dispatch(args: &[String]) -> Result<bool> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-run") {
        return Ok(false);
    }
    if args[2..].iter().any(|arg| arg == "-h" || arg == "--help") {
        print_help();
        return Ok(true);
    }
    let opts = parse_args(args)?;
    let report = run_reference(&opts)?;
    if let Some(output) = &opts.output {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(output, serde_json::to_vec_pretty(&report)?)
            .with_context(|| format!("writing {}", output.display()))?;
    }
    emit_report(&report, &opts.format)?;
    Ok(true)
}

fn print_help() {
    println!(
        "Run the BitNet C++ reference command from a plan and emit a diagnostic receipt\n\nUsage: xtask.exe bitnet-reference-run [OPTIONS]\n\nOptions:\n      --plan <PATH>       Reference plan JSON [default: target/a770-diagnostic/bitnet-reference-plan.json]\n      --output <PATH>     Output reference run JSON [default: target/a770-diagnostic/bitnet-reference-run.json]\n      --format <FORMAT>   Output format: human or json [default: human]\n  -h, --help              Print help"
    );
}

fn parse_args(args: &[String]) -> Result<ReferenceRunArgs> {
    let mut plan = PathBuf::from("target/a770-diagnostic/bitnet-reference-plan.json");
    let mut output = Some(PathBuf::from("target/a770-diagnostic/bitnet-reference-run.json"));
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
            "--plan" => plan = PathBuf::from(value()?),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--format" => format = value()?,
            other => bail!("unknown bitnet-reference-run option {other}"),
        }
    }
    Ok(ReferenceRunArgs { plan, output, format })
}

fn run_reference(args: &ReferenceRunArgs) -> Result<Value> {
    let plan = read_json(&args.plan)?;
    let argv = reference_argv(&plan)?;
    let capture = run_command(&argv)?;
    Ok(build_receipt(&args.plan, &plan, &argv, &capture))
}

fn reference_argv(plan: &Value) -> Result<Vec<String>> {
    let argv = plan
        .pointer("/reference/command_argv")
        .and_then(Value::as_array)
        .context("plan missing /reference/command_argv")?
        .iter()
        .map(|item| {
            item.as_str().map(ToOwned::to_owned).context("command_argv item is not a string")
        })
        .collect::<Result<Vec<_>>>()?;
    if argv.is_empty() {
        bail!("plan reference command_argv is empty");
    }
    if !argv.iter().any(|arg| arg == "--no-display-prompt") {
        bail!(
            "reference command must include --no-display-prompt before it can produce a clean text receipt"
        );
    }
    Ok(argv)
}

fn run_command(argv: &[String]) -> Result<CommandCapture> {
    let executable = argv.first().context("empty reference command")?;
    let start = Instant::now();
    let output = Command::new(executable)
        .args(&argv[1..])
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("running reference executable {executable}"))?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(CommandCapture {
        status_code: output.status.code(),
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        elapsed_ms,
    })
}

fn build_receipt(
    plan_path: &Path,
    plan: &Value,
    argv: &[String],
    capture: &CommandCapture,
) -> Value {
    let generated_text = generated_text_from_stdout(&capture.stdout);
    let reference_prompt_tokens = reference_prompt_token_report(plan, &capture.stderr);
    let stderr_contains_double_bos_warning =
        capture.stderr.contains("final prompt starts with 2 BOS tokens");
    let stderr_contains_auto_bos_override =
        capture.stderr.contains("tokenizer.ggml.add_bos_token") && capture.stderr.contains("false");
    let mut blocked_reasons = Vec::new();
    if !capture.success {
        blocked_reasons.push("reference_command_failed".to_string());
    }
    if generated_text.is_empty() {
        blocked_reasons.push("reference_generated_text_missing".to_string());
    }
    if stderr_contains_double_bos_warning {
        blocked_reasons.push("reference_double_bos_warning_present".to_string());
    }
    let reference_prompt_tokens_available =
        bool_at(&reference_prompt_tokens, "/available").unwrap_or(false);
    let reference_prompt_token_count_mismatch =
        bool_at(&reference_prompt_tokens, "/token_count_matches_plan") == Some(false);
    let reference_prompt_token_hash_mismatch =
        bool_at(&reference_prompt_tokens, "/token_ids_sha256_matches_plan") == Some(false);
    if !reference_prompt_tokens_available {
        blocked_reasons.push("reference_prompt_token_ids_missing".to_string());
    }
    if reference_prompt_token_count_mismatch {
        blocked_reasons.push("reference_prompt_token_count_mismatch".to_string());
    }
    if reference_prompt_token_hash_mismatch {
        blocked_reasons.push("reference_prompt_token_ids_sha256_mismatch".to_string());
    }
    let generated_text_present = !generated_text.is_empty();
    let next_when_ready = if !reference_prompt_tokens_available
        || reference_prompt_token_count_mismatch
        || reference_prompt_token_hash_mismatch
    {
        "resolve reference prompt tokenization mismatch before token/logit parity comparison"
    } else {
        "compare reference text against Rust CPU and strict A770 receipts; use a deeper reference hook before token/logit parity claims"
    };

    json!({
        "schema_version": 1,
        "receipt_type": "bitnet_reference_run",
        "diagnostic": "bitnet_reference_run",
        "producer": "cargo xtask bitnet-reference-run",
        "created_at": chrono::Utc::now().to_rfc3339(),
        "diagnostic_only": true,
        "promotion_allowed": false,
        "claim_allowed": false,
        "classification": "diagnostic_only",
        "plan": {
            "path": plan_path.display().to_string(),
            "diagnostic": str_at(plan, "/diagnostic").unwrap_or(""),
            "prompt_identity": plan.pointer("/prompt_identity").cloned().unwrap_or(Value::Null),
            "model": plan.pointer("/model").cloned().unwrap_or(Value::Null),
            "reference_text_tokenization": plan
                .pointer("/rust_commands/reference_text_tokenization")
                .cloned()
                .unwrap_or(Value::Null),
            "command_policy": str_at(plan, "/reference/command_policy").unwrap_or(""),
        },
        "reference_text_tokenization": plan
            .pointer("/rust_commands/reference_text_tokenization")
            .cloned()
            .unwrap_or(Value::Null),
        "reference_prompt_tokenization": reference_prompt_tokens,
        "reference": {
            "backend": str_at(plan, "/reference/backend").unwrap_or("bitnet.cpp_or_llama.cpp_cli"),
            "selected_executable": str_at(plan, "/reference/selected_executable").unwrap_or(""),
            "command_argv": argv,
            "command_argv_sha256": sha256_json(argv),
            "stderr_contains_auto_bos_override": stderr_contains_auto_bos_override,
            "stderr_contains_double_bos_warning": stderr_contains_double_bos_warning,
        },
        "execution": {
            "success": capture.success,
            "exit_code": capture.status_code,
            "elapsed_ms": capture.elapsed_ms,
            "stdout_bytes": capture.stdout.len(),
            "stderr_bytes": capture.stderr.len(),
            "stdout_sha256": sha256_text(&capture.stdout),
            "stderr_sha256": sha256_text(&capture.stderr),
            "stderr_tail": stderr_tail(&capture.stderr, 24),
        },
        "text": generated_text.clone(),
        "generated_text": generated_text,
        "signals": {
            "generated_text_present": generated_text_present,
            "actual_generated_token_ids_present": false,
            "top_logits_present": false,
            "policy": "reference CLI stdout is captured as generated text only; generated token ids and top logits are not inferred from text",
        },
        "decision": {
            "reference_execution_ready": blocked_reasons.is_empty(),
            "current_blocked_reasons": blocked_reasons,
            "next_when_ready": next_when_ready,
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    })
}

fn generated_text_from_stdout(stdout: &str) -> String {
    stdout.trim().to_string()
}

fn stderr_tail(stderr: &str, max_lines: usize) -> Vec<String> {
    let lines = stderr.lines().map(ToOwned::to_owned).collect::<Vec<_>>();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].to_vec()
}

#[derive(Debug, PartialEq, Eq)]
struct ReferencePromptTokens {
    token_count: Option<usize>,
    token_ids: Vec<u32>,
}

fn reference_prompt_token_report(plan: &Value, stderr: &str) -> Value {
    let parsed = reference_prompt_tokens_from_stderr(stderr);
    let available = !parsed.token_ids.is_empty();
    let token_ids_sha256 = if available { sha256_token_ids(&parsed.token_ids).ok() } else { None };
    let plan_count = u64_at(plan, "/prompt_identity/prompt_token_count");
    let plan_hash = str_at(plan, "/prompt_identity/prompt_token_ids_sha256");
    let token_count_matches_plan = parsed
        .token_count
        .zip(plan_count.map(|count| count as usize))
        .map(|(actual, planned)| actual == planned);
    let token_ids_sha256_matches_plan =
        token_ids_sha256.as_deref().zip(plan_hash).map(|(actual, planned)| actual == planned);

    json!({
        "diagnostic_only": true,
        "claimable": false,
        "available": available,
        "source": "reference_cli_verbose_prompt_stderr",
        "token_count": parsed.token_count,
        "token_ids": parsed.token_ids,
        "token_ids_sha256": token_ids_sha256,
        "plan_prompt_token_count": plan_count,
        "plan_prompt_token_ids_sha256": plan_hash,
        "token_count_matches_plan": token_count_matches_plan,
        "token_ids_sha256_matches_plan": token_ids_sha256_matches_plan,
        "policy": "actual reference prompt tokenization is parsed only from llama-cli --verbose-prompt stderr; it does not prove generated token ids or top logits",
        "not_claims": [
            "reference_prompt_tokenization_proves_reference_generated_token_ids",
            "reference_prompt_tokenization_proves_reference_top_logits",
            "reference_prompt_tokenization_promotes_reference_parity",
            "reference_prompt_tokenization_promotes_a770_semantic_quality"
        ],
    })
}

fn reference_prompt_tokens_from_stderr(stderr: &str) -> ReferencePromptTokens {
    let mut token_count = None;
    let mut token_ids = Vec::new();
    let mut in_prompt_tokens = false;

    for line in stderr.lines() {
        let trimmed = line.trim_start();
        if let Some(raw) = trimmed.strip_prefix("main: number of tokens in prompt = ") {
            token_count = raw.trim().parse::<usize>().ok();
            in_prompt_tokens = true;
            continue;
        }
        if !in_prompt_tokens {
            continue;
        }
        if trimmed.starts_with("sampler seed:")
            || trimmed.starts_with("sampler params:")
            || trimmed.starts_with("sampler chain:")
            || trimmed.starts_with("generate:")
            || trimmed.starts_with("llama_perf_")
        {
            break;
        }
        if let Some((raw_id, _piece)) = trimmed.split_once("->") {
            if let Ok(id) = raw_id.trim().parse::<u32>() {
                token_ids.push(id);
            }
        }
    }

    ReferencePromptTokens { token_count, token_ids }
}

fn read_json(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn str_at<'a>(value: &'a Value, pointer: &str) -> Option<&'a str> {
    value.pointer(pointer).and_then(Value::as_str)
}

fn bool_at(value: &Value, pointer: &str) -> Option<bool> {
    value.pointer(pointer).and_then(Value::as_bool)
}

fn u64_at(value: &Value, pointer: &str) -> Option<u64> {
    value.pointer(pointer).and_then(Value::as_u64)
}

fn sha256_json<T: serde::Serialize + ?Sized>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    sha256_bytes(&bytes)
}

fn sha256_token_ids(tokens: &[u32]) -> Result<String> {
    Ok(sha256_bytes(&serde_json::to_vec(tokens)?))
}

fn sha256_text(value: &str) -> String {
    sha256_bytes(value.as_bytes())
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
            println!("diagnostic: bitnet_reference_run");
            println!(
                "classification: {}",
                value
                    .pointer("/classification")
                    .and_then(Value::as_str)
                    .unwrap_or("diagnostic_only")
            );
            println!(
                "reference_execution_ready: {}",
                value
                    .pointer("/decision/reference_execution_ready")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            );
            if let Some(text) = value.pointer("/generated_text").and_then(Value::as_str) {
                println!("generated_text: {text}");
            }
            if let Some(reasons) = value.pointer("/decision/current_blocked_reasons") {
                println!("blocked_reasons: {}", serde_json::to_string(reasons)?);
            }
            println!("not_claims: {}", serde_json::to_string(&value["not_claims"])?);
        }
        other => bail!("unsupported bitnet-reference-run output format: {other}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_text_trims_reference_stdout() {
        assert_eq!(
            generated_text_from_stdout("\n2+2 equals 4. [end of text]\r\n\n"),
            "2+2 equals 4. [end of text]"
        );
    }

    #[test]
    fn reference_receipt_is_text_only_and_non_promoting() {
        let prompt_ids = vec![128000, 128006, 882, 128007];
        let prompt_hash = sha256_token_ids(&prompt_ids).unwrap();
        let plan = json!({
            "diagnostic": "bitnet_reference_plan",
            "prompt_identity": {
                "prompt_token_count": prompt_ids.len(),
                "prompt_token_ids_sha256": prompt_hash,
            },
            "model": {"model_id": "test"},
            "rust_commands": {
                "reference_text_tokenization": {
                    "diagnostic_only": true,
                    "selected_logit_probe_ids": [17, 10]
                }
            },
            "reference": {
                "backend": "bitnet.cpp_or_llama.cpp_cli",
                "selected_executable": "llama-cli",
                "command_policy": "test"
            }
        });
        let capture = CommandCapture {
            status_code: Some(0),
            success: true,
            stdout: "2+2 equals 4. [end of text]\n".to_string(),
            stderr: "validate_override: Using metadata override ( bool) 'tokenizer.ggml.add_bos_token' = false\n\
main: number of tokens in prompt = 4\n\
128000 -> '<|begin_of_text|>'\n\
128006 -> '<|start_header_id|>'\n\
   882 -> 'user'\n\
128007 -> '<|end_header_id|>'\n\
sampler seed: 0\n"
                .to_string(),
            elapsed_ms: 10.0,
        };
        let receipt =
            build_receipt(Path::new("plan.json"), &plan, &["llama-cli".to_string()], &capture);
        assert_eq!(receipt["claim_allowed"], false);
        assert_eq!(receipt["decision"]["reference_execution_ready"], true);
        assert_eq!(receipt["reference_prompt_tokenization"]["available"], true);
        assert_eq!(receipt["reference_prompt_tokenization"]["token_ids"], json!(prompt_ids));
        assert_eq!(receipt["reference_prompt_tokenization"]["token_count_matches_plan"], true);
        assert_eq!(receipt["reference_prompt_tokenization"]["token_ids_sha256_matches_plan"], true);
        assert_eq!(receipt["signals"]["actual_generated_token_ids_present"], false);
        assert_eq!(receipt["signals"]["top_logits_present"], false);
        assert_eq!(
            receipt["reference_text_tokenization"]["selected_logit_probe_ids"],
            json!([17, 10])
        );
        assert_eq!(
            receipt["plan"]["reference_text_tokenization"]["selected_logit_probe_ids"],
            json!([17, 10])
        );
        let not_claims = receipt["not_claims"].as_array().unwrap();
        assert!(not_claims.contains(&json!("reference_generated_token_ids")));
        assert!(not_claims.contains(&json!("reference_top_logits")));
        assert!(not_claims.contains(&json!("rust_reference_parity_proven")));
    }

    #[test]
    fn reference_prompt_tokenization_mismatch_blocks_ready_state() {
        let planned_ids = vec![1, 2];
        let plan = json!({
            "prompt_identity": {
                "prompt_token_count": planned_ids.len(),
                "prompt_token_ids_sha256": sha256_token_ids(&planned_ids).unwrap(),
            },
            "reference": {
                "backend": "bitnet.cpp_or_llama.cpp_cli",
                "selected_executable": "llama-cli",
                "command_policy": "test"
            }
        });
        let capture = CommandCapture {
            status_code: Some(0),
            success: true,
            stdout: "2\n".to_string(),
            stderr: "main: number of tokens in prompt = 3\n\
1 -> 'a'\n\
2 -> 'b'\n\
3 -> 'c'\n\
sampler seed: 0\n"
                .to_string(),
            elapsed_ms: 10.0,
        };

        let receipt =
            build_receipt(Path::new("plan.json"), &plan, &["llama-cli".to_string()], &capture);

        assert_eq!(receipt["decision"]["reference_execution_ready"], false);
        assert_eq!(receipt["reference_prompt_tokenization"]["token_count_matches_plan"], false);
        assert_eq!(
            receipt["reference_prompt_tokenization"]["token_ids_sha256_matches_plan"],
            false
        );
        let reasons = receipt["decision"]["current_blocked_reasons"].as_array().unwrap();
        assert!(reasons.contains(&json!("reference_prompt_token_count_mismatch")));
        assert!(reasons.contains(&json!("reference_prompt_token_ids_sha256_mismatch")));
    }

    #[test]
    fn parses_reference_verbose_prompt_tokens_from_stderr() {
        let parsed = reference_prompt_tokens_from_stderr(
            "noise\n\
main: number of tokens in prompt = 4\n\
128000 -> '<|begin_of_text|>'\n\
   882 -> 'user'\n\
   198 -> '<newline>'\n\
128009 -> '<|eot_id|>'\n\
sampler seed: 0\n",
        );

        assert_eq!(parsed.token_count, Some(4));
        assert_eq!(parsed.token_ids, vec![128000, 882, 198, 128009]);
    }

    #[test]
    fn reference_argv_requires_prompt_echo_suppression() {
        let plan = json!({
            "reference": {
                "command_argv": ["llama-cli", "-m", "model.gguf"]
            }
        });
        let error = reference_argv(&plan).unwrap_err().to_string();
        assert!(error.contains("--no-display-prompt"));
    }
}
