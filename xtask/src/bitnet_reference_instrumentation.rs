use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const DEFAULT_CPP_ROOT: &str = "target/external/BitNet-reference/3rdparty/llama.cpp";
const DEFAULT_PATCH: &str = "ci/reference-instrumentation/bitnet-rs-first-token-logits-main.patch";
const DEFAULT_OUTPUT: &str = "target/a770-diagnostic/bitnet-reference-instrumentation-plan.json";

const REQUIRED_METADATA: &[&str] =
    &["Upstream-Issue", "Reason", "Status", "Created", "Review-By", "Author"];

const REQUIRED_ANCHORS: &[&str] = &[
    "BITNET_RS_REFERENCE_FIRST_TOKEN_LOGITS",
    "llama_get_logits_ith(ctx, -1)",
    "common_sampler_sample(smpl, ctx, -1)",
    "const llama_token probe_tokens[] = {17, 58428};",
    "probe_logits",
];

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
struct ReferenceInstrumentationArgs {
    cpp_root: PathBuf,
    patch: PathBuf,
    output: Option<PathBuf>,
    format: String,
}

#[derive(Debug)]
struct CommandCapture {
    status_code: Option<i32>,
    success: bool,
    stdout: String,
    stderr: String,
}

pub fn maybe_dispatch_from_env() -> Result<bool> {
    let args = std::env::args().collect::<Vec<_>>();
    maybe_dispatch(&args)
}

fn maybe_dispatch(args: &[String]) -> Result<bool> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-instrumentation-plan") {
        return Ok(false);
    }
    if args[2..].iter().any(|arg| arg == "-h" || arg == "--help") {
        print_help();
        return Ok(true);
    }
    let opts = parse_args(args)?;
    let report = build_plan(&opts)?;
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
        "Verify the target-local BitNet reference first-token logit instrumentation patch\n\nUsage: xtask.exe bitnet-reference-instrumentation-plan [OPTIONS]\n\nOptions:\n      --cpp-root <PATH>  llama.cpp checkout root [default: target/external/BitNet-reference/3rdparty/llama.cpp]\n      --patch <PATH>     Instrumentation patch [default: ci/reference-instrumentation/bitnet-rs-first-token-logits-main.patch]\n      --output <PATH>    Output JSON receipt [default: target/a770-diagnostic/bitnet-reference-instrumentation-plan.json]\n      --format <FORMAT>  Output format: human or json [default: human]\n  -h, --help             Print help"
    );
}

fn parse_args(args: &[String]) -> Result<ReferenceInstrumentationArgs> {
    let mut cpp_root = PathBuf::from(DEFAULT_CPP_ROOT);
    let mut patch = PathBuf::from(DEFAULT_PATCH);
    let mut output = Some(PathBuf::from(DEFAULT_OUTPUT));
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
            "--cpp-root" => cpp_root = PathBuf::from(value()?),
            "--patch" => patch = PathBuf::from(value()?),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--format" => format = value()?,
            other => bail!("unknown bitnet-reference-instrumentation-plan option {other}"),
        }
    }
    Ok(ReferenceInstrumentationArgs { cpp_root, patch, output, format })
}

fn build_plan(args: &ReferenceInstrumentationArgs) -> Result<Value> {
    let cpp_root = normalize_path(&args.cpp_root)?;
    let patch = normalize_path(&args.patch)?;
    let patch_text = fs::read_to_string(&patch).unwrap_or_default();
    let metadata = patch_metadata(&patch_text);
    let missing_metadata = missing_metadata(&metadata);
    let missing_anchors = missing_patch_anchors(&patch_text);
    let source_target = cpp_root.join("examples/main/main.cpp");
    let git_status = if cpp_root.is_dir() {
        Some(run_git(&cpp_root, &["status", "--porcelain"])?)
    } else {
        None
    };
    let apply_check = if cpp_root.is_dir() && patch.is_file() {
        Some(run_git(&cpp_root, &["apply", "--check", &path_to_string(&patch)])?)
    } else {
        None
    };
    let git_clean = git_status.as_ref().is_some_and(|capture| {
        capture.success && capture.stdout.trim().is_empty() && capture.stderr.trim().is_empty()
    });
    let patch_applies = apply_check.as_ref().is_some_and(|capture| capture.success);
    let ready = cpp_root.is_dir()
        && patch.is_file()
        && source_target.is_file()
        && git_clean
        && patch_applies
        && missing_metadata.is_empty()
        && missing_anchors.is_empty();
    let mut blocked_reasons = Vec::new();
    if !cpp_root.is_dir() {
        blocked_reasons.push("reference_llama_cpp_root_missing".to_string());
    }
    if !source_target.is_file() {
        blocked_reasons.push("reference_llama_main_source_missing".to_string());
    }
    if !patch.is_file() {
        blocked_reasons.push("reference_instrumentation_patch_missing".to_string());
    }
    if !git_clean {
        blocked_reasons.push("reference_llama_cpp_worktree_not_clean".to_string());
    }
    if !patch_applies {
        blocked_reasons.push("reference_instrumentation_patch_apply_check_failed".to_string());
    }
    if !missing_metadata.is_empty() {
        blocked_reasons.push("reference_instrumentation_patch_metadata_incomplete".to_string());
    }
    if !missing_anchors.is_empty() {
        blocked_reasons.push("reference_instrumentation_patch_anchor_missing".to_string());
    }
    blocked_reasons.sort_unstable();
    blocked_reasons.dedup();

    Ok(json!({
        "schema_version": 1,
        "receipt_type": "bitnet_reference_instrumentation_plan",
        "diagnostic": "bitnet_reference_instrumentation_plan",
        "producer": "cargo xtask bitnet-reference-instrumentation-plan",
        "created_at": chrono::Utc::now().to_rfc3339(),
        "diagnostic_only": true,
        "promotion_allowed": false,
        "claim_allowed": false,
        "classification": "diagnostic_only",
        "cpp_root": {
            "path": path_to_string(&cpp_root),
            "exists": cpp_root.is_dir(),
            "main_source": path_to_string(&source_target),
            "main_source_exists": source_target.is_file(),
            "git_status_clean": git_clean,
            "git_status_stdout": git_status.as_ref().map(|capture| capture.stdout.trim().to_string()),
            "git_status_stderr": git_status.as_ref().map(|capture| capture.stderr.trim().to_string()),
        },
        "patch": {
            "path": path_to_string(&patch),
            "exists": patch.is_file(),
            "sha256": patch.is_file().then(|| sha256_bytes(&fs::read(&patch).unwrap_or_default())),
            "metadata": metadata,
            "missing_metadata": missing_metadata,
            "required_anchors": REQUIRED_ANCHORS,
            "missing_anchors": missing_anchors,
            "apply_check_success": patch_applies,
            "apply_check_exit_code": apply_check.as_ref().and_then(|capture| capture.status_code),
            "apply_check_stdout": apply_check.as_ref().map(|capture| capture.stdout.trim().to_string()),
            "apply_check_stderr": apply_check.as_ref().map(|capture| capture.stderr.trim().to_string()),
            "default_applied": false,
            "policy": "target-local diagnostic instrumentation patch; not stored under patches/ and not applied by default fetch scripts",
        },
        "instrumentation": {
            "environment_variable": "BITNET_RS_REFERENCE_FIRST_TOKEN_LOGITS",
            "receipt_type_when_applied": "bitnet_reference_first_token_logits",
            "probe_token_ids": [17, 58428],
            "top_k": 16,
            "captures": [
                "prompt_token_count",
                "n_vocab",
                "probe token logits and probabilities",
                "top logits for first generated token"
            ],
            "not_claim": "instrumentation output is reference-side diagnostic evidence only until compared against Rust CPU and strict A770 receipts",
        },
        "operator_commands": {
            "apply_patch": format!("git -C {} apply {}", path_to_string(&cpp_root), path_to_string(&patch)),
            "rebuild_reference": "cmake --build target/external/BitNet-reference/build --config Release --target llama-cli",
            "run_with_receipt_env": "set BITNET_RS_REFERENCE_FIRST_TOKEN_LOGITS=target/a770-diagnostic/reference-first-token-logits.json before running bitnet-reference-run",
        },
        "decision": {
            "instrumentation_ready_to_apply": ready,
            "reference_raw_logits_available": false,
            "current_blocked_reasons": blocked_reasons,
            "next_action": "apply instrumentation patch in the external reference worktree, rebuild llama-cli, run matched reference prompt with BITNET_RS_REFERENCE_FIRST_TOKEN_LOGITS, then compare token 17 and 58428 against Rust CPU and strict A770 logits",
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    }))
}

fn normalize_path(path: &Path) -> Result<PathBuf> {
    let path =
        if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir()?.join(path) };
    Ok(path)
}

fn patch_metadata(text: &str) -> Value {
    let mut map = Map::new();
    for line in text.lines().take_while(|line| line.trim_start().starts_with('#')) {
        let line = line.trim_start().trim_start_matches('#').trim();
        if let Some((key, value)) = line.split_once(':') {
            map.insert(key.trim().to_string(), Value::String(value.trim().to_string()));
        }
    }
    Value::Object(map)
}

fn missing_metadata(metadata: &Value) -> Vec<String> {
    REQUIRED_METADATA
        .iter()
        .filter(|key| metadata.pointer(&format!("/{key}")).and_then(Value::as_str).is_none())
        .map(|key| (*key).to_string())
        .collect()
}

fn missing_patch_anchors(text: &str) -> Vec<String> {
    REQUIRED_ANCHORS
        .iter()
        .filter(|anchor| !text.contains(**anchor))
        .map(|anchor| (*anchor).to_string())
        .collect()
}

fn run_git(cwd: &Path, args: &[&str]) -> Result<CommandCapture> {
    let output = Command::new("git")
        .current_dir(cwd)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("running git {} in {}", args.join(" "), cwd.display()))?;
    Ok(CommandCapture {
        status_code: output.status.code(),
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

fn path_to_string(path: &Path) -> String {
    path.display().to_string()
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
            println!("diagnostic: bitnet_reference_instrumentation_plan");
            println!(
                "instrumentation_ready_to_apply: {}",
                value
                    .pointer("/decision/instrumentation_ready_to_apply")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            );
            if let Some(reasons) = value.pointer("/decision/current_blocked_reasons") {
                println!("blocked_reasons: {}", serde_json::to_string(reasons)?);
            }
            println!("not_claims: {}", serde_json::to_string(&value["not_claims"])?);
        }
        other => bail!("unsupported bitnet-reference-instrumentation-plan output format: {other}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_metadata_reads_header_fields() {
        let metadata = patch_metadata(
            "# Upstream-Issue: not-applicable\n# Reason: test\n\n--- a/file\n+++ b/file\n",
        );

        assert_eq!(metadata["Upstream-Issue"], json!("not-applicable"));
        assert_eq!(metadata["Reason"], json!("test"));
    }

    #[test]
    fn missing_metadata_reports_required_fields() {
        let metadata = json!({
            "Upstream-Issue": "not-applicable",
            "Reason": "test"
        });

        let missing = missing_metadata(&metadata);

        assert!(missing.contains(&"Status".to_string()));
        assert!(missing.contains(&"Created".to_string()));
        assert!(!missing.contains(&"Reason".to_string()));
    }

    #[test]
    fn anchor_check_requires_probe_tokens_and_env_name() {
        let missing = missing_patch_anchors(
            "BITNET_RS_REFERENCE_FIRST_TOKEN_LOGITS const llama_token probe_tokens[] = {17, 58428};",
        );

        assert!(!missing.contains(&"BITNET_RS_REFERENCE_FIRST_TOKEN_LOGITS".to_string()));
        assert!(!missing.contains(&"const llama_token probe_tokens[] = {17, 58428};".to_string()));
        assert!(missing.contains(&"llama_get_logits_ith(ctx, -1)".to_string()));
    }
}
