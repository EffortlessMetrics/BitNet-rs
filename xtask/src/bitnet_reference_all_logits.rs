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
    "reference_raw_logits",
    "rust_reference_parity_proven",
    "a770_semantic_quality_proven",
];

#[derive(Debug)]
struct ReferenceAllLogitsArgs {
    plan: PathBuf,
    output: Option<PathBuf>,
    logits_output: PathBuf,
    context_size: u32,
    batch_size: u32,
    threads: u32,
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

#[derive(Debug, Default, PartialEq, Eq)]
struct PerplexityLogitsHeader {
    magic_valid: bool,
    n_ctx: Option<i32>,
    n_vocab: Option<i32>,
    n_chunk: Option<i32>,
}

pub fn maybe_dispatch_from_env() -> Result<bool> {
    let args = std::env::args().collect::<Vec<_>>();
    maybe_dispatch(&args)
}

fn maybe_dispatch(args: &[String]) -> Result<bool> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-all-logits") {
        return Ok(false);
    }
    if args[2..].iter().any(|arg| arg == "-h" || arg == "--help") {
        print_help();
        return Ok(true);
    }
    let opts = parse_args(args)?;
    let report = run_reference_all_logits(&opts)?;
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
        "Attempt stock BitNet/llama.cpp all-logits capture and emit a diagnostic availability receipt\n\nUsage: xtask.exe bitnet-reference-all-logits [OPTIONS]\n\nOptions:\n      --plan <PATH>            Reference plan JSON [default: target/a770-diagnostic/bitnet-reference-plan.json]\n      --output <PATH>          Output receipt JSON [default: target/a770-diagnostic/bitnet-reference-all-logits.json]\n      --logits-output <PATH>   Stock tool logits file [default: target/a770-diagnostic/reference-all-logits-stock.bin]\n      --ctx-size <N>           Perplexity context size [default: 128]\n      --batch-size <N>         Perplexity batch size [default: 32]\n      --threads <N>            Perplexity thread count [default: 4]\n      --format <FORMAT>        Output format: human or json [default: human]\n  -h, --help                   Print help"
    );
}

fn parse_args(args: &[String]) -> Result<ReferenceAllLogitsArgs> {
    let mut plan = PathBuf::from("target/a770-diagnostic/bitnet-reference-plan.json");
    let mut output = Some(PathBuf::from("target/a770-diagnostic/bitnet-reference-all-logits.json"));
    let mut logits_output = PathBuf::from("target/a770-diagnostic/reference-all-logits-stock.bin");
    let mut context_size = 128u32;
    let mut batch_size = 32u32;
    let mut threads = 4u32;
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
            "--logits-output" => logits_output = PathBuf::from(value()?),
            "--ctx-size" => context_size = value()?.parse().context("parsing --ctx-size")?,
            "--batch-size" => batch_size = value()?.parse().context("parsing --batch-size")?,
            "--threads" => threads = value()?.parse().context("parsing --threads")?,
            "--format" => format = value()?,
            other => bail!("unknown bitnet-reference-all-logits option {other}"),
        }
    }
    Ok(ReferenceAllLogitsArgs {
        plan,
        output,
        logits_output,
        context_size,
        batch_size,
        threads,
        format,
    })
}

fn run_reference_all_logits(args: &ReferenceAllLogitsArgs) -> Result<Value> {
    let plan = read_json(&args.plan)?;
    let executable = perplexity_executable(&plan);
    let prompt = reference_prompt(&plan)?;
    let model = str_at(&plan, "/model/model_path").context("plan missing /model/model_path")?;
    if let Some(parent) = args.logits_output.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let _ = fs::remove_file(&args.logits_output);
    let argv = perplexity_argv(
        &executable,
        model,
        &prompt,
        &args.logits_output,
        args.context_size,
        args.batch_size,
        args.threads,
    );
    let capture = if Path::new(&executable).exists() { Some(run_command(&argv)?) } else { None };
    Ok(build_receipt(&args.plan, &plan, &argv, &args.logits_output, capture.as_ref()))
}

fn perplexity_executable(plan: &Value) -> String {
    if let Some(selected) = str_at(plan, "/reference/selected_executable") {
        let path = Path::new(selected);
        if let Some(parent) = path.parent() {
            return parent.join(exe_name("llama-perplexity")).display().to_string();
        }
    }
    Path::new("target/external/BitNet-reference/build/bin")
        .join(exe_name("llama-perplexity"))
        .display()
        .to_string()
}

fn exe_name(stem: &str) -> String {
    if cfg!(windows) { format!("{stem}.exe") } else { stem.to_string() }
}

fn reference_prompt(plan: &Value) -> Result<String> {
    let argv = plan
        .pointer("/reference/command_argv")
        .and_then(Value::as_array)
        .context("plan missing /reference/command_argv")?;
    for window in argv.windows(2) {
        if window[0].as_str() == Some("-p") || window[0].as_str() == Some("--prompt") {
            return window[1]
                .as_str()
                .map(ToOwned::to_owned)
                .context("reference prompt argument is not a string");
        }
    }
    bail!("reference command_argv missing -p/--prompt")
}

fn perplexity_argv(
    executable: &str,
    model: &str,
    prompt: &str,
    logits_output: &Path,
    context_size: u32,
    batch_size: u32,
    threads: u32,
) -> Vec<String> {
    vec![
        executable.to_string(),
        "-m".to_string(),
        model.to_string(),
        "--override-kv".to_string(),
        "tokenizer.ggml.add_bos_token=bool:false".to_string(),
        "-p".to_string(),
        prompt.to_string(),
        "--ctx-size".to_string(),
        context_size.to_string(),
        "--batch-size".to_string(),
        batch_size.to_string(),
        "--threads".to_string(),
        threads.to_string(),
        "--chunks".to_string(),
        "1".to_string(),
        "--all-logits".to_string(),
        "--save-all-logits".to_string(),
        logits_output.display().to_string(),
    ]
}

fn run_command(argv: &[String]) -> Result<CommandCapture> {
    let executable = argv.first().context("empty reference all-logits command")?;
    let start = Instant::now();
    let output = Command::new(executable)
        .args(&argv[1..])
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("running reference all-logits executable {executable}"))?;
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
    logits_output: &Path,
    capture: Option<&CommandCapture>,
) -> Value {
    let logits_file = logits_file_report(logits_output);
    let stderr = capture.map(|capture| capture.stderr.as_str()).unwrap_or("");
    let insufficient_tokens =
        stderr.contains("you need at least") && stderr.contains("tokens to evaluate perplexity");
    let data_file_token_count = parse_data_file_token_count(stderr);
    let required_token_count = parse_required_token_count(stderr);
    let header_complete = bool_at(&logits_file, "/header/header_complete").unwrap_or(false);
    let mut blocked_reasons = Vec::new();
    if capture.is_none() {
        blocked_reasons.push("stock_llama_perplexity_executable_missing".to_string());
    }
    if capture.is_some_and(|capture| !capture.success) {
        blocked_reasons.push("stock_llama_perplexity_command_failed".to_string());
    }
    if insufficient_tokens {
        blocked_reasons.push("stock_perplexity_requires_long_context_corpus".to_string());
    }
    if !header_complete {
        blocked_reasons.push("stock_all_logits_file_incomplete".to_string());
    }
    blocked_reasons.push(
        "stock_all_logits_file_is_perplexity_logprob_artifact_not_raw_first_token_logits"
            .to_string(),
    );
    blocked_reasons.sort_unstable();
    blocked_reasons.dedup();

    json!({
        "schema_version": 1,
        "receipt_type": "bitnet_reference_all_logits_attempt",
        "diagnostic": "bitnet_reference_all_logits_attempt",
        "producer": "cargo xtask bitnet-reference-all-logits",
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
            "reference_top_logit_capability": plan.pointer("/reference/top_logit_capability").cloned().unwrap_or(Value::Null),
        },
        "stock_tool_attempt": {
            "attempted_tool": "llama-perplexity",
            "command_argv": argv,
            "command_argv_sha256": sha256_json(argv),
            "success": capture.map(|capture| capture.success),
            "exit_code": capture.and_then(|capture| capture.status_code),
            "elapsed_ms": capture.map(|capture| capture.elapsed_ms),
            "stdout_bytes": capture.map(|capture| capture.stdout.len()),
            "stderr_bytes": capture.map(|capture| capture.stderr.len()),
            "stdout_sha256": capture.map(|capture| sha256_text(&capture.stdout)),
            "stderr_sha256": capture.map(|capture| sha256_text(&capture.stderr)),
            "stderr_tail": capture.map(|capture| stderr_tail(&capture.stderr, 24)),
            "insufficient_tokens_for_perplexity": insufficient_tokens,
            "required_token_count": required_token_count,
            "data_file_token_count": data_file_token_count,
            "policy": "stock llama-perplexity --save-all-logits writes a perplexity log-probability artifact, not a first-token raw-logit receipt",
        },
        "logits_file": logits_file,
        "target_probe_tokens": [
            {"token_id": 17, "token_piece": "2"},
            {"token_id": 58428, "token_piece": ".ps"}
        ],
        "decision": {
            "reference_raw_logits_available": false,
            "reference_first_token_logits_available": false,
            "reference_top_logits_available": false,
            "stock_all_logits_artifact_available": header_complete,
            "current_blocked_reasons": blocked_reasons,
            "next_action": "open minimal reference-instrumentation lane to emit first-token raw logits for token 17 and token 58428 under the matched prompt identity",
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    })
}

fn logits_file_report(path: &Path) -> Value {
    let bytes = fs::read(path).unwrap_or_default();
    let header = parse_perplexity_logits_header(&bytes);
    json!({
        "path": path.display().to_string(),
        "exists": path.exists(),
        "bytes": bytes.len(),
        "sha256": (!bytes.is_empty()).then(|| sha256_bytes(&bytes)),
        "format": "llama_perplexity_save_all_logits",
        "format_policy": "header is detectable, but this stock file stores compressed perplexity log probabilities rather than first-token raw logits",
        "header": {
            "magic_valid": header.magic_valid,
            "n_ctx": header.n_ctx,
            "n_vocab": header.n_vocab,
            "n_chunk": header.n_chunk,
            "header_complete": header.magic_valid && header.n_ctx.is_some() && header.n_vocab.is_some() && header.n_chunk.is_some(),
        }
    })
}

fn parse_perplexity_logits_header(bytes: &[u8]) -> PerplexityLogitsHeader {
    let magic_valid = bytes.get(0..8) == Some(b"_logits_");
    PerplexityLogitsHeader {
        magic_valid,
        n_ctx: read_i32_le(bytes, 8),
        n_vocab: read_i32_le(bytes, 12),
        n_chunk: read_i32_le(bytes, 16),
    }
}

fn read_i32_le(bytes: &[u8], offset: usize) -> Option<i32> {
    let raw = bytes.get(offset..offset + 4)?;
    Some(i32::from_le_bytes(raw.try_into().ok()?))
}

fn parse_required_token_count(stderr: &str) -> Option<u64> {
    parse_between(stderr, "you need at least ", " tokens to evaluate perplexity")
}

fn parse_data_file_token_count(stderr: &str) -> Option<u64> {
    parse_between(stderr, "tokenizes to only ", " tokens")
}

fn parse_between(text: &str, prefix: &str, suffix: &str) -> Option<u64> {
    let start = text.find(prefix)? + prefix.len();
    let rest = &text[start..];
    let end = rest.find(suffix)?;
    rest[..end].trim().parse().ok()
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

fn stderr_tail(stderr: &str, max_lines: usize) -> Vec<String> {
    let lines = stderr.lines().map(ToOwned::to_owned).collect::<Vec<_>>();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].to_vec()
}

fn sha256_json<T: serde::Serialize + ?Sized>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    sha256_bytes(&bytes)
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
            println!("diagnostic: bitnet_reference_all_logits_attempt");
            println!(
                "reference_raw_logits_available: {}",
                value
                    .pointer("/decision/reference_raw_logits_available")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            );
            if let Some(reasons) = value.pointer("/decision/current_blocked_reasons") {
                println!("blocked_reasons: {}", serde_json::to_string(reasons)?);
            }
            println!("not_claims: {}", serde_json::to_string(&value["not_claims"])?);
        }
        other => bail!("unsupported bitnet-reference-all-logits output format: {other}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_perplexity_executable_from_reference_cli_sibling() {
        let plan = json!({
            "reference": {
                "selected_executable": "target/external/BitNet-reference/build/bin/llama-cli.exe"
            }
        });
        let exe = perplexity_executable(&plan);
        let path = Path::new(&exe);
        let expected_name = exe_name("llama-perplexity");

        assert_eq!(path.file_name().and_then(|name| name.to_str()), Some(expected_name.as_str()));
        assert!(
            path.parent().is_some_and(
                |parent| parent.ends_with("target/external/BitNet-reference/build/bin")
            )
        );
    }

    #[test]
    fn extracts_rendered_prompt_from_reference_argv() {
        let plan = json!({
            "reference": {
                "command_argv": ["llama-cli", "-m", "model.gguf", "-p", "rendered prompt"]
            }
        });

        assert_eq!(reference_prompt(&plan).unwrap(), "rendered prompt");
    }

    #[test]
    fn parses_stock_perplexity_logits_header() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"_logits_");
        bytes.extend_from_slice(&128i32.to_le_bytes());
        bytes.extend_from_slice(&128256i32.to_le_bytes());
        bytes.extend_from_slice(&1i32.to_le_bytes());

        let header = parse_perplexity_logits_header(&bytes);

        assert_eq!(
            header,
            PerplexityLogitsHeader {
                magic_valid: true,
                n_ctx: Some(128),
                n_vocab: Some(128256),
                n_chunk: Some(1),
            }
        );
    }

    #[test]
    fn incomplete_stock_perplexity_file_does_not_promote_raw_logits() {
        let dir = tempfile::tempdir().unwrap();
        let plan_path = dir.path().join("plan.json");
        let logits_path = dir.path().join("logits.bin");
        fs::write(&logits_path, [b"_logits_".as_slice(), &128i32.to_le_bytes()].concat()).unwrap();
        let plan = json!({
            "diagnostic": "bitnet_reference_plan",
            "model": {"model_path": "model.gguf"},
            "prompt_identity": {"prompt_token_count": 18},
            "reference": {"top_logit_capability": {"diagnostic_only": true}}
        });
        let capture = CommandCapture {
            status_code: Some(0),
            success: true,
            stdout: String::new(),
            stderr: "perplexity: you need at least 256 tokens to evaluate perplexity with a context of 128\nperplexity: the data file you provided tokenizes to only 62 tokens\n".to_string(),
            elapsed_ms: 12.0,
        };

        let receipt = build_receipt(
            &plan_path,
            &plan,
            &["llama-perplexity".to_string()],
            &logits_path,
            Some(&capture),
        );

        assert_eq!(receipt["diagnostic_only"], true);
        assert_eq!(receipt["claim_allowed"], false);
        assert_eq!(receipt["decision"]["reference_raw_logits_available"], false);
        assert_eq!(receipt["decision"]["reference_first_token_logits_available"], false);
        assert_eq!(receipt["logits_file"]["header"]["magic_valid"], true);
        assert_eq!(receipt["logits_file"]["header"]["header_complete"], false);
        assert_eq!(receipt["stock_tool_attempt"]["required_token_count"], json!(256));
        assert_eq!(receipt["stock_tool_attempt"]["data_file_token_count"], json!(62));
        let blockers = receipt["decision"]["current_blocked_reasons"].as_array().unwrap();
        assert!(blockers.contains(&json!("stock_perplexity_requires_long_context_corpus")));
        assert!(blockers.contains(&json!("stock_all_logits_file_incomplete")));
        assert!(blockers.contains(&json!(
            "stock_all_logits_file_is_perplexity_logprob_artifact_not_raw_first_token_logits"
        )));
        let not_claims = receipt["not_claims"].as_array().unwrap();
        assert!(not_claims.contains(&json!("reference_raw_logits")));
        assert!(not_claims.contains(&json!("a770_semantic_quality_proven")));
    }
}
