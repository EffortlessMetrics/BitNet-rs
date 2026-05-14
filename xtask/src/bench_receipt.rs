use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::process::Command;

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

const REQUIRED_POINTERS: &[&str] = &[
    "/schema_version",
    "/receipt_type",
    "/run_id",
    "/repo/commit",
    "/repo/tree",
    "/repo/dirty",
    "/repo/cargo_lock_hash",
    "/repo/rustc",
    "/repo/features",
    "/device/device_slug",
    "/device/device_instance_hash",
    "/model/contract",
    "/model/model_id",
    "/model/weights_sha256",
    "/model/tokenizer_sha256",
    "/model/chat_template_hash",
    "/backend/selected_backend",
    "/backend/backend_family",
    "/backend/fallback_used",
    "/kernel_route/route_verified",
    "/kernel_route/route_claimable",
    "/kernel_route/device_slug",
    "/kernel_route/selected_backend",
    "/kernel_route/kernel_variants",
    "/benchmark_profile/id",
    "/benchmark_profile/profile_hash",
    "/quality_gate/required",
    "/quality_gate/quality_passed",
    "/quality_gate/quality_receipt",
    "/claim_gate/classification",
    "/measurements",
    "/not_claims",
];

#[derive(Debug, Serialize)]
struct BenchReceiptVerifyReport {
    diagnostic: &'static str,
    producer: &'static str,
    receipt_path: String,
    passed: bool,
    claimable: bool,
    classification: String,
    quality_required: bool,
    quality_passed: bool,
    repo_dirty: bool,
    fallback_used: bool,
    route_verified: bool,
    route_claimable: bool,
    model_contract_matched: Option<bool>,
    resource_envelope_complete: Option<bool>,
    run_id: Option<String>,
    benchmark_profile: Option<String>,
    selected_backend: Option<String>,
    failures: Vec<String>,
    blocked_reasons: Vec<String>,
    not_claims: Vec<String>,
}

pub fn verify_receipt(receipt_path: &Path, format: &str, require_claimable: bool) -> Result<()> {
    let report = build_verify_report(receipt_path, require_claimable)?;
    emit_report(&report, format)?;
    if !report.passed {
        bail!("bench receipt verification failed: {}", report.failures.join(", "));
    }
    Ok(())
}

pub fn from_cli_stage(
    plan_path: &Path,
    cli_stage_receipt: &Path,
    model_contract: &Path,
    quality_receipt: &Path,
    quality_passed: bool,
    output: &Path,
    format: &str,
) -> Result<()> {
    let plan = read_json(plan_path)?;
    let cli_stage = read_json(cli_stage_receipt)?;
    let contract = read_yaml(model_contract)?;
    let receipt = build_from_cli_stage_receipt(
        plan_path,
        &plan,
        cli_stage_receipt,
        &cli_stage,
        model_contract,
        &contract,
        quality_receipt,
        quality_passed,
    )?;

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(output, serde_json::to_vec_pretty(&receipt)?)
        .with_context(|| format!("writing {}", output.display()))?;
    emit_json_or_human(&receipt, format)?;
    Ok(())
}

fn build_from_cli_stage_receipt(
    plan_path: &Path,
    plan: &Value,
    cli_stage_path: &Path,
    cli_stage: &Value,
    model_contract_path: &Path,
    contract: &Value,
    quality_receipt: &Path,
    quality_passed: bool,
) -> Result<Value> {
    let identity = cli_stage_identity(plan, cli_stage, model_contract_path);
    let repo = repo_identity()?;
    let planned_backend = str_at(plan, "/backend").unwrap_or("unknown-backend");
    let actual_selected_backend = str_at(cli_stage, "/proof_summary/selected_backend")
        .or_else(|| str_at(cli_stage, "/proof_summary/execution_backend"))
        .unwrap_or("unknown-backend");
    let actual_execution_backend =
        str_at(cli_stage, "/proof_summary/execution_backend").unwrap_or(actual_selected_backend);
    let fallback_used = bool_at(cli_stage, "/proof_summary/fallback_used").unwrap_or(true);
    let route_declared = bool_at(&identity, "/route_declared").unwrap_or(false);
    let execution_backend_matched =
        bool_at(&identity, "/execution_backend_matched").unwrap_or(false);
    let route_claimable = bool_at(&identity, "/route_claimable").unwrap_or(false);
    let route_verified = route_declared && execution_backend_matched && !fallback_used;
    let model_contract_matched = bool_at(&identity, "/model_matched").unwrap_or(false);
    let profile_identity_matched = bool_at(&identity, "/profile_matched").unwrap_or(false);
    let resource_envelope_complete = false;

    let mut blocked_reasons = Vec::new();
    if !quality_passed {
        blocked_reasons.push("quality_not_passed");
    }
    if bool_at(&repo, "/dirty").unwrap_or(true) {
        blocked_reasons.push("repo_dirty");
    }
    if fallback_used {
        blocked_reasons.push("fallback_used");
    }
    if !profile_identity_matched {
        blocked_reasons.push("profile_identity_mismatch");
    }
    if !model_contract_matched {
        blocked_reasons.push("model_contract_not_matched");
    }
    if !route_declared {
        blocked_reasons.push("route_not_declared");
    }
    if !execution_backend_matched {
        blocked_reasons.push("execution_backend_mismatch");
        if planned_backend.contains("a770") {
            blocked_reasons.push("execution_backend_not_a770");
        }
    }
    if !route_verified {
        blocked_reasons.push("route_not_verified");
    }
    if !route_claimable {
        blocked_reasons.push("route_not_claimable");
    }
    if !resource_envelope_complete {
        blocked_reasons.push("resource_envelope_incomplete");
    }
    blocked_reasons.sort_unstable();
    blocked_reasons.dedup();

    let benchmark_claim_allowed = quality_passed
        && !bool_at(&repo, "/dirty").unwrap_or(true)
        && !fallback_used
        && route_verified
        && route_claimable
        && resource_envelope_complete
        && profile_identity_matched
        && model_contract_matched;

    let profile_id = str_at(plan, "/profile/id").unwrap_or("unknown-profile");
    let run_id = format!(
        "{}_{}_{}",
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ"),
        str_at(plan, "/device_slug").unwrap_or("unknown-device"),
        profile_id
    );
    let prompt_tokens = number_at(cli_stage, "/tokens/prompt")
        .or_else(|| number_at(plan, "/profile/target_prompt_tokens"));
    let generated_tokens = number_at(cli_stage, "/tokens/generated");
    let tokens_per_second = number_at(cli_stage, "/throughput/tokens_per_second");
    let prefill_ms = number_at(cli_stage, "/ttft/first_decode_ms")
        .or_else(|| number_at(cli_stage, "/latency/decode_first_ms"));

    Ok(serde_json::json!({
        "schema_version": 1,
        "receipt_type": "bench_run",
        "run_id": run_id,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "source_receipts": {
            "profile_cli_plan": plan_path.display().to_string(),
            "cli_stage_receipt": cli_stage_path.display().to_string()
        },
        "repo": repo,
        "device": {
            "device_slug": str_at(plan, "/device_slug").unwrap_or("unknown-device"),
            "device_instance_hash": "sha256:local-unverified"
        },
        "model": {
            "contract": model_contract_path.display().to_string(),
            "model_id": str_at(contract, "/model_id").unwrap_or("unknown-model"),
            "weights_sha256": str_at(contract, "/sha256").unwrap_or(""),
            "tokenizer_sha256": str_at(contract, "/tokenizer/sha256").unwrap_or(""),
            "chat_template_hash": sha256_text(str_at(contract, "/chat_template/name").unwrap_or(""))
        },
        "backend": {
            "requested_backend": str_at(cli_stage, "/proof_summary/requested_backend").unwrap_or(planned_backend),
            "planned_backend": planned_backend,
            "selected_backend": actual_selected_backend,
            "execution_backend": actual_execution_backend,
            "backend_family": backend_family(actual_selected_backend),
            "fallback_used": fallback_used
        },
        "kernel_route": {
            "route_verified": route_verified,
            "route_claimable": route_claimable,
            "route_declared": route_declared,
            "device_slug": str_at(plan, "/device_slug").unwrap_or("unknown-device"),
            "selected_backend": planned_backend,
            "declared_route_id": str_at(cli_stage, "/proof_summary/kernel_route/route_id"),
            "kernel_variants": [],
            "not_claims": CRITICAL_NOT_CLAIMS
        },
        "benchmark_profile": {
            "id": profile_id,
            "profile_hash": sha256_text(&serde_json::to_string(&plan["profile"])?),
            "profile_identity": identity
        },
        "cli_stage_identity_validation": identity,
        "quality_gate": {
            "required": true,
            "quality_passed": quality_passed,
            "quality_receipt": quality_receipt.display().to_string()
        },
        "claim_gate": {
            "benchmark_claim_allowed": benchmark_claim_allowed,
            "classification": if benchmark_claim_allowed { "performance_proven" } else { "diagnostic_only" },
            "model_contract_matched": model_contract_matched,
            "resource_envelope_complete": resource_envelope_complete,
            "blocked_reasons": blocked_reasons
        },
        "measurements": {
            "summary": {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "prefill_ms": prefill_ms,
                "steady_state_output_tok_s": tokens_per_second,
                "end_to_end_output_tok_s": tokens_per_second,
                "peak_rss_bytes": number_at(cli_stage, "/resource_envelope/peak_rss_bytes"),
                "peak_vram_bytes": Value::Null,
                "host_device_transfer_bytes": Value::Null,
                "kernel_invocations": Value::Null
            }
        },
        "not_claims": CRITICAL_NOT_CLAIMS
    }))
}

fn build_verify_report(
    receipt_path: &Path,
    require_claimable: bool,
) -> Result<BenchReceiptVerifyReport> {
    let raw = fs::read_to_string(receipt_path)
        .with_context(|| format!("reading {}", receipt_path.display()))?;
    let value: Value = serde_json::from_str(&raw)
        .with_context(|| format!("parsing {}", receipt_path.display()))?;

    let mut failures = Vec::new();
    let mut blocked_reasons = Vec::new();
    for pointer in REQUIRED_POINTERS {
        if value.pointer(pointer).is_none_or(Value::is_null) {
            failures.push(format!("missing required field {pointer}"));
        }
    }

    if str_at(&value, "/receipt_type") != Some("bench_run") {
        failures.push("receipt_type must be bench_run".to_string());
    }

    let quality_required = bool_at(&value, "/quality_gate/required").unwrap_or(false);
    let quality_passed = bool_at(&value, "/quality_gate/quality_passed").unwrap_or(false);
    let repo_dirty = bool_at(&value, "/repo/dirty").unwrap_or(true);
    let fallback_used = bool_at(&value, "/backend/fallback_used").unwrap_or(true);
    let route_verified = bool_at(&value, "/kernel_route/route_verified").unwrap_or(false);
    let route_claimable = bool_at(&value, "/kernel_route/route_claimable").unwrap_or(false);
    let model_contract_matched = bool_at(&value, "/claim_gate/model_contract_matched");
    let resource_envelope_complete = bool_at(&value, "/claim_gate/resource_envelope_complete");
    let claimable = bool_at(&value, "/claim_gate/benchmark_claim_allowed")
        .or_else(|| bool_at(&value, "/claim_gate/claim_allowed"))
        .unwrap_or(false);
    let classification =
        str_at(&value, "/claim_gate/classification").unwrap_or("diagnostic_only").to_string();

    let not_claims = array_strings(&value, "/not_claims");
    for not_claim in CRITICAL_NOT_CLAIMS {
        if !not_claims.iter().any(|value| value == not_claim) {
            failures.push(format!("missing critical not-claim {not_claim}"));
        }
    }

    if quality_required && str_at(&value, "/quality_gate/quality_receipt").unwrap_or("").is_empty()
    {
        failures.push("quality is required but quality_receipt is empty".to_string());
    }
    if claimable && !quality_passed {
        failures.push("benchmark claim allowed while quality_passed=false".to_string());
    }
    if claimable && repo_dirty {
        failures.push("benchmark claim allowed from dirty repo".to_string());
    }
    if claimable && fallback_used {
        failures.push("benchmark claim allowed while fallback_used=true".to_string());
    }
    if claimable && !route_verified {
        failures.push("benchmark claim allowed while route_verified=false".to_string());
    }
    if claimable && !route_claimable {
        failures.push("benchmark claim allowed while route_claimable=false".to_string());
    }
    if claimable && model_contract_matched == Some(false) {
        failures.push("benchmark claim allowed while model_contract_matched=false".to_string());
    }
    if claimable && resource_envelope_complete == Some(false) {
        failures.push("benchmark claim allowed while resource_envelope_complete=false".to_string());
    }
    if claimable && classification == "diagnostic_only" {
        failures.push("benchmark claim allowed with diagnostic_only classification".to_string());
    }

    if !quality_passed {
        blocked_reasons.push("quality_not_passed".to_string());
    }
    if repo_dirty {
        blocked_reasons.push("repo_dirty".to_string());
    }
    if fallback_used {
        blocked_reasons.push("fallback_used".to_string());
    }
    if !route_verified {
        blocked_reasons.push("route_not_verified".to_string());
    }
    if !route_claimable {
        blocked_reasons.push("route_not_claimable".to_string());
    }
    if model_contract_matched == Some(false) {
        blocked_reasons.push("model_contract_not_matched".to_string());
    }
    if resource_envelope_complete == Some(false) {
        blocked_reasons.push("resource_envelope_incomplete".to_string());
    }
    if require_claimable && !claimable {
        failures.push("receipt is not claimable but --require-claimable was set".to_string());
    }

    Ok(BenchReceiptVerifyReport {
        diagnostic: "bench_receipt_verify",
        producer: "cargo xtask bench verify-receipt",
        receipt_path: receipt_path.display().to_string(),
        passed: failures.is_empty(),
        claimable,
        classification,
        quality_required,
        quality_passed,
        repo_dirty,
        fallback_used,
        route_verified,
        route_claimable,
        model_contract_matched,
        resource_envelope_complete,
        run_id: str_at(&value, "/run_id").map(ToOwned::to_owned),
        benchmark_profile: str_at(&value, "/benchmark_profile/id").map(ToOwned::to_owned),
        selected_backend: str_at(&value, "/backend/selected_backend").map(ToOwned::to_owned),
        failures,
        blocked_reasons,
        not_claims,
    })
}

fn emit_report(report: &BenchReceiptVerifyReport, format: &str) -> Result<()> {
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(report)?),
        "human" => {
            println!("bench receipt verify: passed={}", report.passed);
            println!("claimable: {}", report.claimable);
            println!("classification: {}", report.classification);
            if !report.blocked_reasons.is_empty() {
                println!("blocked_reasons: {}", report.blocked_reasons.join(", "));
            }
            if !report.failures.is_empty() {
                println!("failures: {}", report.failures.join(", "));
            }
            println!("not_claims: {}", report.not_claims.join(", "));
        }
        other => bail!("unsupported bench receipt output format: {other}"),
    }
    Ok(())
}

fn bool_at(value: &Value, pointer: &str) -> Option<bool> {
    value.pointer(pointer).and_then(Value::as_bool)
}

fn str_at<'a>(value: &'a Value, pointer: &str) -> Option<&'a str> {
    value.pointer(pointer).and_then(Value::as_str)
}

fn array_strings(value: &Value, pointer: &str) -> Vec<String> {
    value
        .pointer(pointer)
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_str).map(ToOwned::to_owned).collect())
        .unwrap_or_default()
}

fn cli_stage_identity(plan: &Value, cli_stage: &Value, model_contract_path: &Path) -> Value {
    let deterministic_temperature = Value::from(0.0);
    let prompt_profile_checks = [
        (
            "same_prompt_token_count",
            plan.pointer("/prompt_identity/prompt_token_count"),
            cli_stage.pointer("/prompt_identity/prompt_token_count"),
        ),
        (
            "same_rendered_prompt_hash",
            plan.pointer("/prompt_identity/rendered_prompt_sha256"),
            cli_stage.pointer("/prompt_identity/rendered_prompt_sha256"),
        ),
        (
            "same_token_ids_hash",
            plan.pointer("/prompt_identity/prompt_token_ids_sha256"),
            cli_stage.pointer("/prompt_identity/prompt_token_ids_sha256"),
        ),
        (
            "same_generated_token_count",
            plan.pointer("/profile/max_new_tokens"),
            cli_stage.pointer("/tokens/generated"),
        ),
        (
            "same_sampling_temperature",
            Some(&deterministic_temperature),
            cli_stage.pointer("/gen_policy/temperature"),
        ),
    ];

    let planned_backend = plan.pointer("/backend");
    let requested_backend = cli_stage.pointer("/proof_summary/requested_backend");
    let selected_backend = cli_stage.pointer("/proof_summary/selected_backend");
    let execution_backend = cli_stage.pointer("/proof_summary/execution_backend");
    let planned_route = plan.pointer("/kernel_route/route_id");
    let declared_route = cli_stage.pointer("/proof_summary/kernel_route/route_id");
    let expected_contract_text = model_contract_path.display().to_string();
    let expected_contract = Value::String(expected_contract_text.clone());
    let declared_contract = cli_stage.pointer("/proof_summary/model_contract");

    let mut object = serde_json::Map::new();
    let mut profile_failures = Vec::new();
    let mut route_failures = Vec::new();
    let mut model_failures = Vec::new();

    for (name, expected, actual) in prompt_profile_checks {
        let passed = expected.is_some() && expected == actual;
        object.insert(name.to_string(), Value::Bool(passed));
        if !passed {
            profile_failures.push(field_failure(name, expected, actual));
        }
    }

    let requested_backend_matched =
        planned_backend.is_some() && requested_backend.is_some() && planned_backend == requested_backend;
    object.insert("requested_backend_matched".to_string(), Value::Bool(requested_backend_matched));
    if !requested_backend_matched {
        route_failures.push(field_failure("requested_backend_matched", planned_backend, requested_backend));
    }

    let selected_backend_matched =
        planned_backend.is_some() && selected_backend.is_some() && planned_backend == selected_backend;
    object.insert("selected_backend_matched".to_string(), Value::Bool(selected_backend_matched));
    if !selected_backend_matched {
        route_failures.push(field_failure("selected_backend_matched", planned_backend, selected_backend));
    }

    let execution_backend_matched =
        planned_backend.is_some() && execution_backend.is_some() && planned_backend == execution_backend;
    object.insert(
        "execution_backend_matched".to_string(),
        Value::Bool(execution_backend_matched),
    );
    if !execution_backend_matched {
        route_failures.push(field_failure("execution_backend_matched", planned_backend, execution_backend));
    }

    let route_declared = bool_at(cli_stage, "/proof_summary/route_declared").unwrap_or(false)
        && declared_route.and_then(Value::as_str).is_some_and(|route| !route.is_empty());
    object.insert("route_declared".to_string(), Value::Bool(route_declared));
    if !route_declared {
        route_failures.push(field_failure("route_declared", Some(&Value::Bool(true)), Some(&Value::Bool(false))));
    }

    let same_kernel_route =
        planned_route.is_some() && declared_route.is_some() && planned_route == declared_route;
    object.insert("same_kernel_route".to_string(), Value::Bool(same_kernel_route));
    if !same_kernel_route {
        route_failures.push(field_failure("same_kernel_route", planned_route, declared_route));
    }

    let route_claimable = bool_at(cli_stage, "/proof_summary/kernel_route/claimable").unwrap_or(false);
    object.insert("route_claimable".to_string(), Value::Bool(route_claimable));
    if !route_claimable {
        route_failures.push(field_failure(
            "route_claimable",
            Some(&Value::Bool(true)),
            Some(&Value::Bool(false)),
        ));
    }

    let model_matched = declared_contract
        .and_then(Value::as_str)
        .is_some_and(|actual| same_path_string(&expected_contract_text, actual));
    object.insert("model_matched".to_string(), Value::Bool(model_matched));
    object.insert("same_model_contract".to_string(), Value::Bool(model_matched));
    if !model_matched {
        model_failures.push(field_failure(
            "model_matched",
            Some(&expected_contract),
            declared_contract,
        ));
    }

    let fallback_false = !bool_at(cli_stage, "/proof_summary/fallback_used").unwrap_or(true);
    object.insert("fallback_false".to_string(), Value::Bool(fallback_false));
    if !fallback_false {
        route_failures.push(field_failure(
            "fallback_false",
            Some(&Value::Bool(true)),
            Some(&Value::Bool(false)),
        ));
    }

    let profile_matched = profile_failures.is_empty();
    let route_identity_matched = requested_backend_matched && same_kernel_route && route_declared;
    let matched = profile_matched
        && model_matched
        && route_identity_matched
        && execution_backend_matched
        && fallback_false
        && route_claimable;

    object.insert("profile_matched".to_string(), Value::Bool(profile_matched));
    object.insert("route_identity_matched".to_string(), Value::Bool(route_identity_matched));
    object.insert("matched".to_string(), Value::Bool(matched));
    object.insert("profile_failures".to_string(), Value::Array(profile_failures));
    object.insert("route_failures".to_string(), Value::Array(route_failures));
    object.insert("model_failures".to_string(), Value::Array(model_failures));
    let failures = object
        .get("profile_failures")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .chain(object.get("route_failures").and_then(Value::as_array).into_iter().flatten())
        .chain(object.get("model_failures").and_then(Value::as_array).into_iter().flatten())
        .cloned()
        .collect();
    object.insert("failures".to_string(), Value::Array(failures));
    Value::Object(object)
}

fn field_failure(name: &str, expected: Option<&Value>, actual: Option<&Value>) -> Value {
    serde_json::json!({
        "field": name,
        "expected": expected.cloned().unwrap_or(Value::Null),
        "actual": actual.cloned().unwrap_or(Value::Null),
    })
}

fn same_path_string(expected: &str, actual: &str) -> bool {
    normalize_path_string(expected) == normalize_path_string(actual)
}

fn normalize_path_string(value: &str) -> String {
    value.replace('\\', "/")
}

fn repo_identity() -> Result<Value> {
    Ok(serde_json::json!({
        "commit": git_output(&["rev-parse", "HEAD"]).unwrap_or_else(|_| "unknown".to_string()),
        "tree": git_output(&["rev-parse", "HEAD^{tree}"]).unwrap_or_else(|_| "unknown".to_string()),
        "dirty": repo_dirty(),
        "cargo_lock_hash": format!("sha256:{}", sha256_file(Path::new("Cargo.lock")).unwrap_or_else(|_| "unknown".to_string())),
        "rustc": command_output("rustc", &["--version"]).unwrap_or_else(|_| "unknown".to_string()),
        "features": ["cpu"],
    }))
}

fn repo_dirty() -> bool {
    Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=all"])
        .output()
        .map(|output| !output.status.success() || !output.stdout.is_empty())
        .unwrap_or(true)
}

fn git_output(args: &[&str]) -> Result<String> {
    command_output("git", args)
}

fn command_output(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("running {program} {}", args.join(" ")))?;
    if !output.status.success() {
        bail!("{program} {} failed", args.join(" "));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn read_json(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn read_yaml(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_yaml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn emit_json_or_human(value: &Value, format: &str) -> Result<()> {
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(value)?),
        "human" => {
            println!("bench from-cli-stage: {}", str_at(value, "/run_id").unwrap_or("unknown"));
            println!(
                "benchmark_claim_allowed: {}",
                bool_at(value, "/claim_gate/benchmark_claim_allowed").unwrap_or(false)
            );
            println!(
                "classification: {}",
                str_at(value, "/claim_gate/classification").unwrap_or("diagnostic_only")
            );
            println!(
                "blocked_reasons: {}",
                value.pointer("/claim_gate/blocked_reasons").unwrap_or(&Value::Null)
            );
        }
        other => bail!("unsupported bench output format: {other}"),
    }
    Ok(())
}

fn number_at(value: &Value, pointer: &str) -> Option<Value> {
    value.pointer(pointer).filter(|value| value.is_number()).cloned()
}

fn backend_family(selected_backend: &str) -> &'static str {
    if selected_backend == "cpu" {
        "cpu"
    } else if selected_backend.contains("a770") || selected_backend.contains("opencl") {
        "intel-opencl"
    } else {
        "unknown"
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(sha256_bytes(&bytes))
}

fn sha256_text(value: &str) -> String {
    sha256_bytes(value.as_bytes())
}

fn sha256_bytes(value: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt(claim_allowed: bool, quality_passed: bool, dirty: bool) -> Value {
        serde_json::json!({
            "schema_version": 1,
            "receipt_type": "bench_run",
            "run_id": "run-1",
            "repo": {
                "commit": "abc",
                "tree": "def",
                "dirty": dirty,
                "cargo_lock_hash": "sha256:lock",
                "rustc": "rustc 1.95.0",
                "features": ["cpu"]
            },
            "device": {
                "device_slug": "amd-5700x-intel-a770",
                "device_instance_hash": "sha256:device"
            },
            "model": {
                "contract": "docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml",
                "model_id": "microsoft/bitnet-b1.58-2B-4T-gguf",
                "weights_sha256": "sha256:weights",
                "tokenizer_sha256": "sha256:tokenizer",
                "chat_template_hash": "sha256:template"
            },
            "backend": {
                "selected_backend": "intel-arc-a770-opencl",
                "backend_family": "intel-opencl",
                "fallback_used": false
            },
            "kernel_route": {
                "route_verified": true,
                "route_claimable": claim_allowed,
                "device_slug": "amd-5700x-intel-a770",
                "selected_backend": "intel-arc-a770-opencl",
                "kernel_variants": [
                    { "op": "qk256_i2s_gemv", "kernel_variant_id": "a770_opencl_qk256_i2s_route_pending_claim_receipts" }
                ]
            },
            "benchmark_profile": {
                "id": "prefill_512_decode_64",
                "profile_hash": "sha256:profile"
            },
            "quality_gate": {
                "required": true,
                "quality_passed": quality_passed,
                "quality_receipt": "quality.json"
            },
            "claim_gate": {
                "benchmark_claim_allowed": claim_allowed,
                "classification": if claim_allowed { "performance_proven" } else { "diagnostic_only" },
                "model_contract_matched": true,
                "resource_envelope_complete": true
            },
            "measurements": { "summary": {} },
            "not_claims": [
                "selected_attention_residency",
                "resident_kv_decode",
                "attention_scores_residency",
                "softmax_residency",
                "attention_value_mix_residency",
                "full_support_op_residency",
                "full_device_residency",
                "completion"
            ]
        })
    }

    fn write_receipt(value: &Value) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("receipt.json");
        fs::write(&path, serde_json::to_string_pretty(value).unwrap()).unwrap();
        (dir, path)
    }

    fn plan_and_cli_stage(execution_backend: &str) -> (Value, Value) {
        let plan = serde_json::json!({
            "backend": "intel-arc-a770-opencl",
            "device_slug": "amd-5700x-intel-a770",
            "kernel_route": {
                "route_id": "a770.bitnet.i2s.qk256"
            },
            "profile": {
                "id": "prefill_512_decode_64",
                "target_prompt_tokens": 512,
                "max_new_tokens": 64
            },
            "prompt_identity": {
                "prompt_token_count": 512,
                "rendered_prompt_sha256": "rendered",
                "prompt_token_ids_sha256": "tokens"
            }
        });
        let cli = serde_json::json!({
            "prompt_identity": {
                "prompt_token_count": 512,
                "rendered_prompt_sha256": "rendered",
                "prompt_token_ids_sha256": "tokens"
            },
            "tokens": {
                "prompt": 512,
                "generated": 64
            },
            "gen_policy": {
                "temperature": 0.0
            },
            "proof_summary": {
                "requested_backend": "intel-arc-a770-opencl",
                "selected_backend": execution_backend,
                "execution_backend": execution_backend,
                "fallback_used": false,
                "model_contract": "docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml",
                "route_declared": true,
                "kernel_route": {
                    "route_id": "a770.bitnet.i2s.qk256",
                    "claimable": false
                }
            },
            "throughput": {
                "tokens_per_second": 12.5
            },
            "ttft": {
                "first_decode_ms": 100.0
            }
        });
        (plan, cli)
    }

    fn contract() -> Value {
        serde_json::json!({
            "model_id": "microsoft/bitnet-b1.58-2B-4T-gguf",
            "sha256": "sha256:weights",
            "tokenizer": {
                "sha256": "sha256:tokenizer"
            },
            "chat_template": {
                "name": "llama3-chat"
            }
        })
    }

    #[test]
    fn diagnostic_receipt_can_verify_without_claim() {
        let (_dir, path) = write_receipt(&receipt(false, false, true));
        let report = build_verify_report(&path, false).unwrap();
        assert!(report.passed);
        assert!(!report.claimable);
        assert!(report.blocked_reasons.iter().any(|reason| reason == "quality_not_passed"));
    }

    #[test]
    fn rejects_quality_failed_claim() {
        let (_dir, path) = write_receipt(&receipt(true, false, false));
        let report = build_verify_report(&path, false).unwrap();
        assert!(!report.passed);
        assert!(report.failures.iter().any(|failure| failure.contains("quality_passed=false")));
    }

    #[test]
    fn rejects_dirty_claim() {
        let (_dir, path) = write_receipt(&receipt(true, true, true));
        let report = build_verify_report(&path, false).unwrap();
        assert!(!report.passed);
        assert!(report.failures.iter().any(|failure| failure.contains("dirty repo")));
    }

    #[test]
    fn cli_stage_identity_rejects_cpu_execution_with_declared_a770_route() {
        let (plan, cli) = plan_and_cli_stage("cpu");
        let identity =
            cli_stage_identity(&plan, &cli, Path::new("docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml"));

        assert_eq!(identity["profile_matched"], true);
        assert_eq!(identity["model_matched"], true);
        assert_eq!(identity["route_declared"], true);
        assert_eq!(identity["same_kernel_route"], true);
        assert_eq!(identity["requested_backend_matched"], true);
        assert_eq!(identity["execution_backend_matched"], false);
        assert_eq!(identity["route_claimable"], false);
        assert_eq!(identity["matched"], false);
    }

    #[test]
    fn from_cli_stage_blocks_cpu_as_a770_evidence() {
        let (plan, cli) = plan_and_cli_stage("cpu");
        let receipt = build_from_cli_stage_receipt(
            Path::new("plan.json"),
            &plan,
            Path::new("cli.json"),
            &cli,
            Path::new("docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml"),
            &contract(),
            Path::new("quality.json"),
            true,
        )
        .unwrap();

        assert_eq!(receipt["backend"]["selected_backend"], "cpu");
        assert_eq!(receipt["backend"]["execution_backend"], "cpu");
        assert_eq!(receipt["kernel_route"]["route_declared"], true);
        assert_eq!(receipt["kernel_route"]["route_verified"], false);
        assert_eq!(receipt["claim_gate"]["benchmark_claim_allowed"], false);
        let blocked = receipt["claim_gate"]["blocked_reasons"].as_array().unwrap();
        assert!(blocked.iter().any(|reason| reason == "execution_backend_not_a770"));
        assert!(blocked.iter().any(|reason| reason == "route_not_claimable"));
        assert!(blocked.iter().any(|reason| reason == "resource_envelope_incomplete"));
    }
}
