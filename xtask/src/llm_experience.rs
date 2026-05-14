use anyhow::{Context, Result, bail};
use bitnet_prompt_templates::TemplateType;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

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

const REQUIRED_EXPERIENCE_POINTERS: &[&str] = &[
    "/schema_version",
    "/receipt_type",
    "/run_id",
    "/producer",
    "/repo",
    "/model",
    "/device",
    "/kernel_route",
    "/backend",
    "/benchmark_profile",
    "/quality/passed",
    "/measurement_supplements/cli_stage_receipt/present",
    "/load/complete",
    "/ttft/complete",
    "/input_speed/complete",
    "/output_speed/complete",
    "/resource_envelope/complete",
    "/claim_gate/claim_allowed",
    "/claim_gate/classification",
    "/not_claims",
];

#[derive(Debug, Clone)]
struct Section {
    complete: bool,
    data: Map<String, Value>,
    missing: Vec<String>,
}

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

pub fn maybe_dispatch_from_env() -> Result<bool> {
    let args = std::env::args().collect::<Vec<_>>();
    maybe_dispatch(&args)
}

fn maybe_dispatch(args: &[String]) -> Result<bool> {
    if args.get(1).map(String::as_str) != Some("llm-experience") {
        return Ok(false);
    }
    match args.get(2).map(String::as_str) {
        Some("run") => {
            if args[3..].iter().any(|arg| arg == "-h" || arg == "--help") {
                print_run_help();
                return Ok(true);
            }
            let opts = parse_run_args(&args[3..])?;
            run(
                &opts.bench_receipt,
                opts.cli_stage_receipt.as_deref(),
                Some(&opts.output),
                &opts.format,
            )?;
            Ok(true)
        }
        Some("verify") => {
            if args[3..].iter().any(|arg| arg == "-h" || arg == "--help") {
                print_verify_help();
                return Ok(true);
            }
            let opts = parse_verify_args(&args[3..])?;
            verify(&opts.receipt, &opts.format, opts.require_claimable)?;
            Ok(true)
        }
        Some("publish") => {
            if args[3..].iter().any(|arg| arg == "-h" || arg == "--help") {
                print_publish_help();
                return Ok(true);
            }
            let opts = parse_publish_args(&args[3..])?;
            publish(&opts.receipt, &opts.history_root, &opts.format)?;
            Ok(true)
        }
        Some("compare") => {
            if args[3..].iter().any(|arg| arg == "-h" || arg == "--help") {
                print_compare_help();
                return Ok(true);
            }
            let opts = parse_compare_args(&args[3..])?;
            compare(
                &opts.history_root,
                &opts.device,
                &opts.profile,
                opts.require_same_route,
                opts.require_claim_ready,
                &opts.format,
            )?;
            Ok(true)
        }
        Some("docs") => {
            if args[3..].iter().any(|arg| arg == "-h" || arg == "--help") {
                print_docs_help();
                return Ok(true);
            }
            let opts = parse_docs_args(&args[3..])?;
            docs(&opts.history_root, &opts.output, opts.check, &opts.format)?;
            Ok(true)
        }
        Some("profile-cli-plan") | Some("-h") | Some("--help") | None => Ok(false),
        Some(other) => bail!(
            "unknown llm-experience subcommand {other}; supported manual subcommands: run, verify, publish, compare, docs"
        ),
    }
}

#[derive(Debug)]
struct RunArgs {
    bench_receipt: PathBuf,
    cli_stage_receipt: Option<PathBuf>,
    output: PathBuf,
    format: String,
}

#[derive(Debug)]
struct VerifyArgs {
    receipt: PathBuf,
    require_claimable: bool,
    format: String,
}

#[derive(Debug)]
struct PublishArgs {
    receipt: PathBuf,
    history_root: PathBuf,
    format: String,
}

#[derive(Debug)]
struct CompareArgs {
    history_root: PathBuf,
    device: String,
    profile: String,
    require_same_route: bool,
    require_claim_ready: bool,
    format: String,
}

#[derive(Debug)]
struct DocsArgs {
    history_root: PathBuf,
    output: PathBuf,
    check: bool,
    format: String,
}

fn parse_run_args(args: &[String]) -> Result<RunArgs> {
    let mut opts = RunArgs {
        bench_receipt: PathBuf::from("target/bench-runs/profile-cli-stage.json"),
        cli_stage_receipt: None,
        output: PathBuf::from("target/llm-experience/a770-bitnet-profile-stage.json"),
        format: "human".to_string(),
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--bench-receipt" => {
                index += 1;
                opts.bench_receipt = PathBuf::from(required_value(args, index, "--bench-receipt")?);
            }
            "--cli-stage-receipt" => {
                index += 1;
                opts.cli_stage_receipt =
                    Some(PathBuf::from(required_value(args, index, "--cli-stage-receipt")?));
            }
            "--output" => {
                index += 1;
                opts.output = PathBuf::from(required_value(args, index, "--output")?);
            }
            "--format" => {
                index += 1;
                opts.format = required_value(args, index, "--format")?.to_string();
            }
            other => bail!("unknown llm-experience run option {other}"),
        }
        index += 1;
    }
    Ok(opts)
}

fn parse_verify_args(args: &[String]) -> Result<VerifyArgs> {
    let mut opts = VerifyArgs {
        receipt: PathBuf::from("target/llm-experience/a770-bitnet-profile-stage.json"),
        require_claimable: false,
        format: "human".to_string(),
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--receipt" => {
                index += 1;
                opts.receipt = PathBuf::from(required_value(args, index, "--receipt")?);
            }
            "--require-claimable" => {
                opts.require_claimable = true;
            }
            "--format" => {
                index += 1;
                opts.format = required_value(args, index, "--format")?.to_string();
            }
            other => bail!("unknown llm-experience verify option {other}"),
        }
        index += 1;
    }
    Ok(opts)
}

fn parse_publish_args(args: &[String]) -> Result<PublishArgs> {
    let mut opts = PublishArgs {
        receipt: PathBuf::from("target/llm-experience/a770-bitnet-profile-stage.json"),
        history_root: PathBuf::from("target/llm-experience-history"),
        format: "human".to_string(),
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--receipt" => {
                index += 1;
                opts.receipt = PathBuf::from(required_value(args, index, "--receipt")?);
            }
            "--history-root" => {
                index += 1;
                opts.history_root = PathBuf::from(required_value(args, index, "--history-root")?);
            }
            "--format" => {
                index += 1;
                opts.format = required_value(args, index, "--format")?.to_string();
            }
            other => bail!("unknown llm-experience publish option {other}"),
        }
        index += 1;
    }
    Ok(opts)
}

fn parse_compare_args(args: &[String]) -> Result<CompareArgs> {
    let mut opts = CompareArgs {
        history_root: PathBuf::from("target/llm-experience-history"),
        device: "amd-5700x-intel-a770".to_string(),
        profile: "prefill_512_decode_64".to_string(),
        require_same_route: false,
        require_claim_ready: false,
        format: "human".to_string(),
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--history-root" => {
                index += 1;
                opts.history_root = PathBuf::from(required_value(args, index, "--history-root")?);
            }
            "--device" => {
                index += 1;
                opts.device = required_value(args, index, "--device")?.to_string();
            }
            "--profile" => {
                index += 1;
                opts.profile = required_value(args, index, "--profile")?.to_string();
            }
            "--require-same-route" => opts.require_same_route = true,
            "--require-claim-ready" => opts.require_claim_ready = true,
            "--format" => {
                index += 1;
                opts.format = required_value(args, index, "--format")?.to_string();
            }
            other => bail!("unknown llm-experience compare option {other}"),
        }
        index += 1;
    }
    Ok(opts)
}

fn parse_docs_args(args: &[String]) -> Result<DocsArgs> {
    let mut opts = DocsArgs {
        history_root: PathBuf::from("target/llm-experience-history"),
        output: PathBuf::from("docs/benchmarks/llm-experience.md"),
        check: false,
        format: "human".to_string(),
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--history-root" => {
                index += 1;
                opts.history_root = PathBuf::from(required_value(args, index, "--history-root")?);
            }
            "--output" => {
                index += 1;
                opts.output = PathBuf::from(required_value(args, index, "--output")?);
            }
            "--check" => opts.check = true,
            "--format" => {
                index += 1;
                opts.format = required_value(args, index, "--format")?.to_string();
            }
            other => bail!("unknown llm-experience docs option {other}"),
        }
        index += 1;
    }
    Ok(opts)
}

fn required_value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str> {
    args.get(index)
        .map(String::as_str)
        .filter(|value| !value.starts_with("--"))
        .with_context(|| format!("{flag} requires a value"))
}

fn print_run_help() {
    println!(
        "Build a canonical LLM experience receipt\n\nUsage: xtask.exe llm-experience run [OPTIONS]\n\nOptions:\n      --bench-receipt <PATH>       Parent benchmark receipt [default: target/bench-runs/profile-cli-stage.json]\n      --cli-stage-receipt <PATH>   Optional CLI-stage measurement supplement\n      --output <PATH>              Output receipt [default: target/llm-experience/a770-bitnet-profile-stage.json]\n      --format <FORMAT>            human or json [default: human]\n  -h, --help                       Print help"
    );
}

fn print_verify_help() {
    println!(
        "Verify an LLM experience receipt\n\nUsage: xtask.exe llm-experience verify [OPTIONS]\n\nOptions:\n      --receipt <PATH>             Experience receipt [default: target/llm-experience/a770-bitnet-profile-stage.json]\n      --require-claimable          Fail if receipt is diagnostic-only\n      --format <FORMAT>            human or json [default: human]\n  -h, --help                       Print help"
    );
}

fn print_publish_help() {
    println!(
        "Publish an LLM experience receipt to local history\n\nUsage: xtask.exe llm-experience publish [OPTIONS]\n\nOptions:\n      --receipt <PATH>             Experience receipt [default: target/llm-experience/a770-bitnet-profile-stage.json]\n      --history-root <PATH>        History root [default: target/llm-experience-history]\n      --format <FORMAT>            human or json [default: human]\n  -h, --help                       Print help"
    );
}

fn print_compare_help() {
    println!(
        "Compare LLM experience history for same-device/same-route readiness\n\nUsage: xtask.exe llm-experience compare [OPTIONS]\n\nOptions:\n      --history-root <PATH>        History root [default: target/llm-experience-history]\n      --device <SLUG>              Device slug [default: amd-5700x-intel-a770]\n      --profile <ID>               Benchmark profile [default: prefill_512_decode_64]\n      --require-same-route         Fail unless the latest pair is same-route comparable\n      --require-claim-ready        Fail unless both receipts are claim-ready\n      --format <FORMAT>            human or json [default: human]\n  -h, --help                       Print help"
    );
}

fn print_docs_help() {
    println!(
        "Generate LLM experience dashboard docs from local history\n\nUsage: xtask.exe llm-experience docs [OPTIONS]\n\nOptions:\n      --history-root <PATH>        History root [default: target/llm-experience-history]\n      --output <PATH>              Markdown output [default: docs/benchmarks/llm-experience.md]\n      --check                      Fail if output is stale\n      --format <FORMAT>            human or json [default: human]\n  -h, --help                       Print help"
    );
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

pub fn run(
    bench_receipt: &Path,
    cli_stage_receipt: Option<&Path>,
    output: Option<&Path>,
    format: &str,
) -> Result<()> {
    let bench = read_json(bench_receipt)?;
    let cli_stage = cli_stage_receipt.map(read_json).transpose()?;
    let receipt =
        build_experience_receipt(bench_receipt, &bench, cli_stage_receipt, cli_stage.as_ref());
    if let Some(output) = output {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(output, serde_json::to_vec_pretty(&receipt)?)
            .with_context(|| format!("writing {}", output.display()))?;
    }
    emit_value(&receipt, format)
}

pub fn verify(receipt: &Path, format: &str, require_claimable: bool) -> Result<()> {
    let value = read_json(receipt)?;
    let report = build_verify_report(receipt, &value, require_claimable);
    emit_value(&report, format)?;
    if !report["passed"].as_bool().unwrap_or(false) {
        bail!("llm-experience verify failed: {}", report["failures"]);
    }
    Ok(())
}

pub fn publish(receipt: &Path, history_root: &Path, format: &str) -> Result<()> {
    let value = read_json(receipt)?;
    let commit = str_at(&value, "/repo/commit").unwrap_or("unknown-commit");
    let device = str_at(&value, "/device/device_slug").unwrap_or("unknown-device");
    let profile = str_at(&value, "/benchmark_profile/id").unwrap_or("unknown-profile");
    let run_id = run_id(&value);
    let output = history_root
        .join("runs")
        .join(sanitize_path_component(commit))
        .join(sanitize_path_component(device))
        .join(sanitize_path_component(profile))
        .join(format!("{}.json", sanitize_path_component(&run_id)));
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::copy(receipt, &output)
        .with_context(|| format!("copying {} to {}", receipt.display(), output.display()))?;
    let report = json!({
        "diagnostic": "llm_experience_publish",
        "producer": "cargo xtask llm-experience publish",
        "receipt": receipt.display().to_string(),
        "history_root": history_root.display().to_string(),
        "published_path": output.display().to_string(),
        "run_id": run_id,
        "claim_allowed": bool_at(&value, "/claim_gate/claim_allowed").unwrap_or(false),
        "not_claims": CRITICAL_NOT_CLAIMS,
    });
    emit_value(&report, format)
}

pub fn compare(
    history_root: &Path,
    device: &str,
    profile: &str,
    require_same_route: bool,
    require_claim_ready: bool,
    format: &str,
) -> Result<()> {
    let mut receipts = history_receipts(history_root, device, profile)?;
    receipts.sort_by(|left, right| left.path.cmp(&right.path));
    let report = build_compare_report(
        history_root,
        device,
        profile,
        require_same_route,
        require_claim_ready,
        &receipts,
    );
    emit_value(&report, format)?;
    if !report["passed"].as_bool().unwrap_or(false) {
        bail!("llm-experience compare failed: {}", report["failures"]);
    }
    Ok(())
}

pub fn docs(history_root: &Path, output: &Path, check: bool, format: &str) -> Result<()> {
    let receipts = all_history_receipts(history_root)?;
    let rendered = render_history_docs(history_root, &receipts);
    if check {
        let existing = fs::read_to_string(output).unwrap_or_default();
        let passed = existing == rendered;
        let report = json!({
            "diagnostic": "llm_experience_docs",
            "producer": "cargo xtask llm-experience docs",
            "check": true,
            "passed": passed,
            "output": output.display().to_string(),
            "history_root": history_root.display().to_string(),
            "receipt_count": receipts.len(),
        });
        emit_value(&report, format)?;
        if !passed {
            bail!("LLM experience docs are stale: {}", output.display());
        }
        return Ok(());
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(output, rendered).with_context(|| format!("writing {}", output.display()))?;
    let report = json!({
        "diagnostic": "llm_experience_docs",
        "producer": "cargo xtask llm-experience docs",
        "check": false,
        "passed": true,
        "output": output.display().to_string(),
        "history_root": history_root.display().to_string(),
        "receipt_count": receipts.len(),
    });
    emit_value(&report, format)
}

fn build_experience_receipt(
    bench_path: &Path,
    bench: &Value,
    cli_stage_path: Option<&Path>,
    cli_stage: Option<&Value>,
) -> Value {
    let load = load_section(cli_stage);
    let ttft = ttft_section(cli_stage);
    let input_speed = input_speed_section(bench, cli_stage);
    let output_speed = output_speed_section(bench, cli_stage);
    let resource_envelope = resource_section(bench, cli_stage);
    let quality_passed = bool_at(bench, "/quality_gate/quality_passed").unwrap_or(false);
    let fallback_used = bool_at(bench, "/backend/fallback_used").unwrap_or(true);
    let route_verified = bool_at(bench, "/kernel_route/route_verified").unwrap_or(false);
    let benchmark_claim_allowed =
        bool_at(bench, "/claim_gate/benchmark_claim_allowed").unwrap_or(false);
    let model_contract_matched =
        bool_at(bench, "/claim_gate/model_contract_matched").unwrap_or(false);
    let parent_classification =
        str_at(bench, "/claim_gate/classification").unwrap_or("diagnostic_only");
    let cli_present = cli_stage.is_some();
    let cli_profile_matched =
        bool_at(bench, "/benchmark_profile/profile_identity/profile_matched").unwrap_or(false);

    let mut blocked_reasons = Vec::new();
    if !benchmark_claim_allowed {
        blocked_reasons.push("parent_benchmark_not_claim_allowed");
    }
    if !quality_passed {
        blocked_reasons.push("quality_not_passed");
    }
    if fallback_used {
        blocked_reasons.push("fallback_used");
    }
    if !route_verified {
        blocked_reasons.push("route_not_verified");
    }
    if !model_contract_matched {
        blocked_reasons.push("model_contract_not_matched");
    }
    if !load.complete {
        blocked_reasons.push("load_incomplete");
    }
    if !ttft.complete {
        blocked_reasons.push("ttft_incomplete");
    }
    if !input_speed.complete {
        blocked_reasons.push("input_speed_incomplete");
    }
    if !output_speed.complete {
        blocked_reasons.push("output_speed_incomplete");
    }
    if !resource_envelope.complete {
        blocked_reasons.push("resource_envelope_incomplete");
    }
    if cli_present {
        blocked_reasons.push("cli_stage_receipt_diagnostic_only");
    }
    blocked_reasons.sort_unstable();
    blocked_reasons.dedup();

    let claim_allowed = blocked_reasons.is_empty();
    let classification = if claim_allowed { parent_classification } else { "diagnostic_only" };

    json!({
        "schema_version": 1,
        "receipt_type": "llm_experience_run",
        "run_id": str_at(bench, "/run_id").unwrap_or("unknown-run"),
        "producer": "cargo xtask llm-experience run",
        "created_at": chrono::Utc::now().to_rfc3339(),
        "source_receipts": {
            "bench_receipt": bench_path.display().to_string(),
            "cli_stage_receipt": cli_stage_path.map(|path| path.display().to_string()),
        },
        "repo": bench.pointer("/repo").cloned().unwrap_or_else(|| json!({})),
        "model": bench.pointer("/model").cloned().unwrap_or_else(|| json!({})),
        "device": bench.pointer("/device").cloned().unwrap_or_else(|| json!({})),
        "kernel_route": bench.pointer("/kernel_route").cloned().unwrap_or_else(|| json!({})),
        "backend": bench.pointer("/backend").cloned().unwrap_or_else(|| json!({})),
        "benchmark_profile": bench.pointer("/benchmark_profile").cloned().unwrap_or_else(|| json!({})),
        "quality": {
            "required": bool_at(bench, "/quality_gate/required").unwrap_or(false),
            "passed": quality_passed,
            "receipt": str_at(bench, "/quality_gate/quality_receipt").unwrap_or(""),
        },
        "measurement_supplements": {
            "cli_stage_receipt": {
                "present": cli_present,
                "path": cli_stage_path.map(|path| path.display().to_string()),
                "diagnostic_only": cli_present,
                "claimable": false,
                "profile_matched": cli_profile_matched,
                "fields_filled": cli_stage_fields_filled(cli_stage),
                "not_claims": [
                    "cli_stage_receipt_proves_quality",
                    "cli_stage_receipt_promotes_benchmark_claim",
                    "cli_stage_receipt_promotes_residency"
                ],
            }
        },
        "load": section_value(load),
        "ttft": section_value(ttft),
        "input_speed": section_value(input_speed),
        "output_speed": section_value(output_speed),
        "resource_envelope": section_value(resource_envelope),
        "claim_gate": {
            "claim_allowed": claim_allowed,
            "classification": classification,
            "quality_passed": quality_passed,
            "fallback_used": fallback_used,
            "route_verified": route_verified,
            "model_contract_matched": model_contract_matched,
            "benchmark_claim_allowed": benchmark_claim_allowed,
            "blocked_reasons": blocked_reasons,
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    })
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
        "cpu,opencl".to_string(),
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

fn load_section(cli_stage: Option<&Value>) -> Section {
    let mut data = Map::new();
    let fields = [
        ("cold_total_ms", "/load_timing/cold_total_ms"),
        ("warm_total_ms", "/load_timing/warm_total_ms"),
        ("model_open_ms", "/load_timing/model_open_ms"),
        ("tokenizer_load_ms", "/load_timing/tokenizer_load_ms"),
        ("backend_init_ms", "/load_timing/backend_init_ms"),
        ("kernel_compile_ms", "/load_timing/kernel_compile_ms"),
        ("weight_upload_ms", "/load_timing/weight_upload_ms"),
        ("ready_to_generate_ms", "/load_timing/ready_to_generate_ms"),
    ];
    let missing = collect_section_fields(cli_stage, &mut data, &fields);
    data.insert(
        "status".to_string(),
        json!(if missing.is_empty() {
            "stage_breakdown_present"
        } else {
            "stage_breakdown_incomplete"
        }),
    );
    Section { complete: missing.is_empty(), data, missing }
}

fn ttft_section(cli_stage: Option<&Value>) -> Section {
    let mut data = Map::new();
    let fields = [
        ("end_to_end_ms", "/ttft/end_to_end_ms"),
        ("prompt_render_ms", "/ttft/prompt_render_ms"),
        ("tokenization_ms", "/ttft/tokenization_ms"),
        ("prefill_ms", "/ttft/prefill_ms"),
        ("first_decode_ms", "/ttft/first_decode_ms"),
        ("sampler_ms", "/ttft/sampler_ms"),
        ("stream_first_byte_ms", "/ttft/stream_first_byte_ms"),
    ];
    let mut missing = collect_section_fields(cli_stage, &mut data, &fields);
    if !data.contains_key("end_to_end_ms") {
        if let Some(value) = cli_stage.and_then(|json| json.pointer("/latency/cmd_to_first_ms")) {
            data.insert("end_to_end_ms".to_string(), value.clone());
            missing.retain(|field| field != "end_to_end_ms");
        }
    }
    if !data.contains_key("first_decode_ms") {
        if let Some(value) = cli_stage.and_then(|json| json.pointer("/latency/decode_first_ms")) {
            data.insert("first_decode_ms".to_string(), value.clone());
            missing.retain(|field| field != "first_decode_ms");
        }
    }
    Section { complete: missing.is_empty(), data, missing }
}

fn input_speed_section(bench: &Value, cli_stage: Option<&Value>) -> Section {
    let mut data = Map::new();
    let mut missing = Vec::new();
    push_number(
        &mut data,
        &mut missing,
        "prompt_tokens",
        bench
            .pointer("/measurements/summary/prompt_tokens")
            .or_else(|| cli_stage.and_then(|json| json.pointer("/tokens/prompt"))),
    );
    push_number(
        &mut data,
        &mut missing,
        "prefill_ms",
        bench
            .pointer("/measurements/summary/prefill_ms")
            .or_else(|| cli_stage.and_then(|json| json.pointer("/ttft/prefill_ms"))),
    );
    let complete = missing.is_empty();
    if complete {
        if let (Some(tokens), Some(prefill_ms)) = (
            data.get("prompt_tokens").and_then(Value::as_f64),
            data.get("prefill_ms").and_then(Value::as_f64),
        ) {
            if prefill_ms > 0.0 {
                data.insert(
                    "input_tokens_per_second".to_string(),
                    json!(tokens / (prefill_ms / 1000.0)),
                );
            }
        }
    }
    data.entry("input_tokens_per_second".to_string()).or_insert(Value::Null);
    Section { complete, data, missing }
}

fn output_speed_section(bench: &Value, cli_stage: Option<&Value>) -> Section {
    let mut data = Map::new();
    let mut missing = Vec::new();
    push_number(
        &mut data,
        &mut missing,
        "generated_tokens",
        bench
            .pointer("/measurements/summary/generated_tokens")
            .or_else(|| cli_stage.and_then(|json| json.pointer("/tokens/generated"))),
    );
    push_number(
        &mut data,
        &mut missing,
        "steady_state_output_tok_s",
        bench
            .pointer("/measurements/summary/steady_state_output_tok_s")
            .or_else(|| cli_stage.and_then(|json| json.pointer("/throughput/tokens_per_second"))),
    );
    push_number(
        &mut data,
        &mut missing,
        "end_to_end_output_tok_s",
        bench
            .pointer("/measurements/summary/end_to_end_output_tok_s")
            .or_else(|| cli_stage.and_then(|json| json.pointer("/throughput/tokens_per_second"))),
    );
    push_number(
        &mut data,
        &mut missing,
        "p50_inter_token_latency_ms",
        bench.pointer("/measurements/summary/p50_inter_token_latency_ms"),
    );
    push_number(
        &mut data,
        &mut missing,
        "p95_inter_token_latency_ms",
        bench.pointer("/measurements/summary/p95_inter_token_latency_ms"),
    );
    data.insert(
        "stop_reason".to_string(),
        cli_stage.and_then(|json| json.pointer("/stop_reason")).cloned().unwrap_or(Value::Null),
    );
    Section { complete: missing.is_empty(), data, missing }
}

fn resource_section(bench: &Value, cli_stage: Option<&Value>) -> Section {
    let mut data = Map::new();
    let mut missing = Vec::new();
    for field in
        ["peak_rss_bytes", "peak_vram_bytes", "host_device_transfer_bytes", "kernel_invocations"]
    {
        push_number(
            &mut data,
            &mut missing,
            field,
            bench.pointer(&format!("/measurements/summary/{field}")).or_else(|| {
                cli_stage.and_then(|json| json.pointer(&format!("/resource_envelope/{field}")))
            }),
        );
    }
    data.insert(
        "resident_ops".to_string(),
        bench
            .pointer("/measurements/summary/resident_ops")
            .or_else(|| cli_stage.and_then(|json| json.pointer("/resource_envelope/resident_ops")))
            .cloned()
            .unwrap_or_else(|| {
                json!({
                    "qk256_linears": true,
                    "embedding": true,
                    "lm_head_tied_logits": true,
                    "selected_attention": false,
                    "resident_kv_decode": false
                })
            }),
    );
    Section { complete: missing.is_empty(), data, missing }
}

fn collect_section_fields(
    source: Option<&Value>,
    data: &mut Map<String, Value>,
    fields: &[(&str, &str)],
) -> Vec<String> {
    let mut missing = Vec::new();
    for (field, pointer) in fields {
        if let Some(value) = source.and_then(|json| json.pointer(pointer)).cloned() {
            data.insert((*field).to_string(), value);
        } else {
            missing.push((*field).to_string());
        }
    }
    missing
}

fn push_number(
    data: &mut Map<String, Value>,
    missing: &mut Vec<String>,
    field: &str,
    value: Option<&Value>,
) {
    if let Some(value) = value.filter(|value| !value.is_null()) {
        data.insert(field.to_string(), value.clone());
    } else {
        data.insert(field.to_string(), Value::Null);
        missing.push(field.to_string());
    }
}

fn section_value(section: Section) -> Value {
    let mut data = section.data;
    data.insert("complete".to_string(), Value::Bool(section.complete));
    data.insert(
        "missing".to_string(),
        Value::Array(section.missing.into_iter().map(Value::String).collect()),
    );
    Value::Object(data)
}

fn cli_stage_fields_filled(cli_stage: Option<&Value>) -> Vec<String> {
    let Some(cli_stage) = cli_stage else {
        return Vec::new();
    };
    [
        ("tokenizer_load_ms", "/load_timing/tokenizer_load_ms"),
        ("prompt_render_ms", "/ttft/prompt_render_ms"),
        ("first_decode_ms", "/ttft/first_decode_ms"),
        ("latency_decode_first_ms", "/latency/decode_first_ms"),
        ("peak_rss_bytes", "/resource_envelope/peak_rss_bytes"),
        ("tokens", "/tokens"),
        ("throughput", "/throughput"),
    ]
    .into_iter()
    .filter_map(|(field, pointer)| cli_stage.pointer(pointer).map(|_| field.to_string()))
    .collect()
}

fn build_verify_report(receipt_path: &Path, value: &Value, require_claimable: bool) -> Value {
    let mut failures = Vec::new();
    for pointer in REQUIRED_EXPERIENCE_POINTERS {
        if value.pointer(pointer).is_none_or(Value::is_null) {
            failures.push(format!("missing required field {pointer}"));
        }
    }
    if str_at(value, "/receipt_type") != Some("llm_experience_run") {
        failures.push("receipt_type must be llm_experience_run".to_string());
    }
    let not_claims = array_strings(value, "/not_claims");
    for not_claim in CRITICAL_NOT_CLAIMS {
        if !not_claims.iter().any(|value| value == not_claim) {
            failures.push(format!("missing critical not-claim {not_claim}"));
        }
    }

    let claim_allowed = bool_at(value, "/claim_gate/claim_allowed").unwrap_or(false);
    let classification = str_at(value, "/claim_gate/classification").unwrap_or("diagnostic_only");
    if claim_allowed && classification == "diagnostic_only" {
        failures.push("claim_allowed=true with diagnostic_only classification".to_string());
    }
    if claim_allowed && !bool_at(value, "/quality/passed").unwrap_or(false) {
        failures.push("claim_allowed=true while quality passed=false".to_string());
    }
    if claim_allowed && bool_at(value, "/backend/fallback_used").unwrap_or(true) {
        failures.push("claim_allowed=true while fallback_used=true".to_string());
    }
    if claim_allowed && !bool_at(value, "/kernel_route/route_verified").unwrap_or(false) {
        failures.push("claim_allowed=true while route_verified=false".to_string());
    }
    if require_claimable && !claim_allowed {
        failures.push("receipt is not claimable but --require-claimable was set".to_string());
    }

    json!({
        "diagnostic": "llm_experience_verify",
        "producer": "cargo xtask llm-experience verify",
        "receipt_path": receipt_path.display().to_string(),
        "passed": failures.is_empty(),
        "claim_allowed": claim_allowed,
        "classification": classification,
        "blocked_reasons": value
            .pointer("/claim_gate/blocked_reasons")
            .cloned()
            .unwrap_or_else(|| json!([])),
        "failures": failures,
        "not_claims": not_claims,
    })
}

#[derive(Debug, Clone)]
struct HistoryReceipt {
    path: PathBuf,
    value: Value,
}

fn history_receipts(
    history_root: &Path,
    device: &str,
    profile: &str,
) -> Result<Vec<HistoryReceipt>> {
    let root = history_root.join("runs");
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut receipts = Vec::new();
    for entry in WalkDir::new(&root).into_iter().filter_map(|entry| entry.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let value = read_json(entry.path())?;
        if str_at(&value, "/device/device_slug") == Some(device)
            && str_at(&value, "/benchmark_profile/id") == Some(profile)
        {
            receipts.push(HistoryReceipt { path: entry.path().to_path_buf(), value });
        }
    }
    Ok(receipts)
}

fn all_history_receipts(history_root: &Path) -> Result<Vec<HistoryReceipt>> {
    let root = history_root.join("runs");
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut receipts = Vec::new();
    for entry in WalkDir::new(&root).into_iter().filter_map(|entry| entry.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        receipts.push(HistoryReceipt {
            path: entry.path().to_path_buf(),
            value: read_json(entry.path())?,
        });
    }
    receipts.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(receipts)
}

fn build_compare_report(
    history_root: &Path,
    device: &str,
    profile: &str,
    require_same_route: bool,
    require_claim_ready: bool,
    receipts: &[HistoryReceipt],
) -> Value {
    let mut failures = Vec::new();
    if receipts.len() < 2 {
        failures.push("history_pair_missing".to_string());
        return json!({
            "diagnostic": "llm_experience_compare",
            "producer": "cargo xtask llm-experience compare",
            "history_root": history_root.display().to_string(),
            "device": device,
            "profile": profile,
            "passed": false,
            "comparison_classification": "diagnostic_only",
            "receipt_count": receipts.len(),
            "failures": failures,
            "not_claims": CRITICAL_NOT_CLAIMS,
        });
    }

    let left = &receipts[receipts.len() - 2];
    let right = &receipts[receipts.len() - 1];
    let left_run_id = run_id(&left.value);
    let right_run_id = run_id(&right.value);
    let distinct_paths = left.path != right.path;
    let distinct_run_ids = !left_run_id.is_empty()
        && !right_run_id.is_empty()
        && left_run_id != "unknown-run"
        && right_run_id != "unknown-run"
        && left_run_id != right_run_id;
    let same_device =
        str_at(&left.value, "/device/device_slug") == str_at(&right.value, "/device/device_slug");
    let same_backend = str_at(&left.value, "/backend/selected_backend")
        == str_at(&right.value, "/backend/selected_backend");
    let same_route = route_key(&left.value) == route_key(&right.value);
    let left_claim_ready = bool_at(&left.value, "/claim_gate/claim_allowed").unwrap_or(false);
    let right_claim_ready = bool_at(&right.value, "/claim_gate/claim_allowed").unwrap_or(false);
    let claim_ready_pair = left_claim_ready && right_claim_ready;

    let comparison_classification = if !distinct_paths || !distinct_run_ids {
        "self_comparison_not_regression_comparable"
    } else if same_device && same_backend && same_route && claim_ready_pair {
        "same_device_same_route_regression"
    } else if same_device && same_backend && same_route {
        "same_device_same_route_diagnostic"
    } else if same_device && same_backend {
        "same_device_route_changed"
    } else {
        "diagnostic_only"
    };

    if require_same_route
        && !(same_device && same_backend && same_route && distinct_paths && distinct_run_ids)
    {
        failures.push("same_route_history_not_ready".to_string());
    }
    if require_claim_ready && !claim_ready_pair {
        failures.push("claim_ready_history_not_ready".to_string());
    }
    if !distinct_paths {
        failures.push("history_receipt_paths_not_distinct".to_string());
    }
    if !distinct_run_ids {
        failures.push("history_run_ids_not_distinct".to_string());
    }

    json!({
        "diagnostic": "llm_experience_compare",
        "producer": "cargo xtask llm-experience compare",
        "history_root": history_root.display().to_string(),
        "device": device,
        "profile": profile,
        "passed": failures.is_empty(),
        "comparison_classification": comparison_classification,
        "receipt_count": receipts.len(),
        "left": {
            "path": left.path.display().to_string(),
            "run_id": left_run_id,
            "claim_allowed": left_claim_ready,
            "route_key": route_key(&left.value),
        },
        "right": {
            "path": right.path.display().to_string(),
            "run_id": right_run_id,
            "claim_allowed": right_claim_ready,
            "route_key": route_key(&right.value),
        },
        "same_device": same_device,
        "same_backend": same_backend,
        "same_route": same_route,
        "distinct_paths": distinct_paths,
        "distinct_run_ids": distinct_run_ids,
        "claim_ready_pair": claim_ready_pair,
        "failures": failures,
        "not_claims": CRITICAL_NOT_CLAIMS,
    })
}

fn render_history_docs(history_root: &Path, receipts: &[HistoryReceipt]) -> String {
    let mut out = String::new();
    out.push_str("# LLM Experience History\n\n");
    out.push_str(&format!("History root: `{}`\n\n", history_root.display()));
    out.push_str("| Device | Profile | Backend | Claim | Run ID | Receipt |\n");
    out.push_str("| --- | --- | --- | --- | --- | --- |\n");
    for receipt in receipts {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            str_at(&receipt.value, "/device/device_slug").unwrap_or("unknown-device"),
            str_at(&receipt.value, "/benchmark_profile/id").unwrap_or("unknown-profile"),
            str_at(&receipt.value, "/backend/selected_backend").unwrap_or("unknown-backend"),
            bool_at(&receipt.value, "/claim_gate/claim_allowed").unwrap_or(false),
            run_id(&receipt.value),
            receipt.path.display()
        ));
    }
    out.push_str("\nNot claimed: selected attention, resident KV, attention scores, softmax, value mix, full support residency, full device residency, completion.\n");
    out
}

fn run_id(value: &Value) -> String {
    str_at(value, "/run_id")
        .or_else(|| str_at(value, "/source_receipts/parent_run_id"))
        .or_else(|| str_at(value, "/source_receipts/bench_run_id"))
        .or_else(|| str_at(value, "/benchmark_profile/run_id"))
        .unwrap_or("unknown-run")
        .to_string()
}

fn route_key(value: &Value) -> String {
    str_at(value, "/kernel_route/declared_route_id")
        .or_else(|| str_at(value, "/kernel_route/route_id"))
        .or_else(|| str_at(value, "/kernel_route/selected_backend"))
        .or_else(|| str_at(value, "/backend/selected_backend"))
        .unwrap_or("unknown-route")
        .to_string()
}

fn sanitize_path_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => ch,
            _ => '_',
        })
        .collect()
}

fn read_json(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
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

fn bool_at(value: &Value, pointer: &str) -> Option<bool> {
    value.pointer(pointer).and_then(Value::as_bool)
}

fn array_strings(value: &Value, pointer: &str) -> Vec<String> {
    value
        .pointer(pointer)
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_str).map(ToOwned::to_owned).collect())
        .unwrap_or_default()
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
            if let Some(claimable) = value
                .pointer("/claimable")
                .and_then(Value::as_bool)
                .or_else(|| value.pointer("/claim_gate/claim_allowed").and_then(Value::as_bool))
            {
                println!("claimable: {claimable}");
            }
            if let Some(classification) = str_at(value, "/claim_gate/classification")
                .or_else(|| str_at(value, "/classification"))
            {
                println!("classification: {classification}");
            }
            if let Some(profile) = str_at(value, "/profile/id") {
                println!("profile: {profile}");
            }
            if let Some(count) = value.pointer("/prompt_identity/prompt_token_count") {
                println!("prompt_token_count: {count}");
            }
            if let Some(blocked) = value
                .pointer("/claim_gate/blocked_reasons")
                .or_else(|| value.pointer("/blocked_reasons"))
            {
                println!("blocked_reasons: {blocked}");
            }
            if let Some(not_claims) = value.pointer("/not_claims") {
                println!("not_claims: {}", serde_json::to_string(not_claims)?);
            }
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
        assert!(command.windows(2).any(|args| args == ["--features", "cpu,opencl"]));
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

    #[test]
    fn manual_dispatch_leaves_profile_plan_to_clap() {
        let args = vec![
            "xtask".to_string(),
            "llm-experience".to_string(),
            "profile-cli-plan".to_string(),
            "--format".to_string(),
            "json".to_string(),
        ];
        assert!(!maybe_dispatch(&args).expect("dispatch"));
    }

    #[test]
    fn diagnostic_parent_keeps_experience_diagnostic() {
        let receipt = build_experience_receipt(
            Path::new("bench.json"),
            &bench_receipt(false),
            Some(Path::new("cli.json")),
            Some(&cli_stage()),
        );
        assert_eq!(receipt["claim_gate"]["claim_allowed"], false);
        assert_eq!(receipt["claim_gate"]["classification"], "diagnostic_only");
        assert!(
            receipt["claim_gate"]["blocked_reasons"]
                .as_array()
                .unwrap()
                .iter()
                .any(|reason| reason == "parent_benchmark_not_claim_allowed")
        );
    }

    #[test]
    fn verify_requires_critical_not_claims() {
        let mut receipt =
            build_experience_receipt(Path::new("bench.json"), &bench_receipt(false), None, None);
        receipt["not_claims"] = json!([]);
        let report = build_verify_report(Path::new("experience.json"), &receipt, false);
        assert_eq!(report["passed"], false);
        assert!(
            report["failures"].as_array().unwrap().iter().any(|failure| {
                failure.as_str().unwrap().contains("selected_attention_residency")
            })
        );
    }

    #[test]
    fn compare_rejects_self_history() {
        let receipt = build_experience_receipt(
            Path::new("bench.json"),
            &bench_receipt(false),
            Some(Path::new("cli.json")),
            Some(&cli_stage()),
        );
        let receipts = vec![
            HistoryReceipt { path: PathBuf::from("same.json"), value: receipt.clone() },
            HistoryReceipt { path: PathBuf::from("same.json"), value: receipt },
        ];
        let report = build_compare_report(
            Path::new("history"),
            "amd-5700x-intel-a770",
            "prefill_512_decode_64",
            true,
            true,
            &receipts,
        );
        assert_eq!(
            report["comparison_classification"],
            "self_comparison_not_regression_comparable"
        );
        assert_eq!(report["passed"], false);
        assert!(
            report["failures"]
                .as_array()
                .unwrap()
                .iter()
                .any(|failure| { failure == "history_receipt_paths_not_distinct" })
        );
    }

    fn bench_receipt(parent_claimable: bool) -> Value {
        json!({
            "run_id": if parent_claimable { "claimable-run" } else { "diagnostic-run" },
            "repo": { "commit": "abc", "tree": "def", "dirty": !parent_claimable },
            "model": { "contract": "docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml" },
            "device": { "device_slug": "amd-5700x-intel-a770" },
            "kernel_route": {
                "route_verified": parent_claimable,
                "device_slug": "amd-5700x-intel-a770"
            },
            "backend": {
                "selected_backend": "intel-arc-a770-opencl",
                "fallback_used": false
            },
            "benchmark_profile": {
                "id": "prefill_512_decode_64",
                "profile_hash": "sha256:profile",
                "profile_identity": { "profile_matched": true }
            },
            "quality_gate": {
                "required": true,
                "quality_passed": true,
                "quality_receipt": "quality.json"
            },
            "claim_gate": {
                "benchmark_claim_allowed": parent_claimable,
                "classification": if parent_claimable { "performance_proven" } else { "diagnostic_only" },
                "model_contract_matched": true
            },
            "measurements": {
                "summary": {
                    "prompt_tokens": 512,
                    "prefill_ms": 4000.0,
                    "generated_tokens": 64,
                    "steady_state_output_tok_s": 10.0,
                    "end_to_end_output_tok_s": 9.0,
                    "p50_inter_token_latency_ms": 100.0,
                    "p95_inter_token_latency_ms": 120.0,
                    "peak_rss_bytes": 100,
                    "peak_vram_bytes": 200,
                    "host_device_transfer_bytes": 300,
                    "kernel_invocations": 4
                }
            },
            "not_claims": CRITICAL_NOT_CLAIMS
        })
    }

    fn cli_stage() -> Value {
        json!({
            "load_timing": {
                "cold_total_ms": 1.0,
                "warm_total_ms": 1.0,
                "model_open_ms": 1.0,
                "tokenizer_load_ms": 1.0,
                "backend_init_ms": 1.0,
                "kernel_compile_ms": 1.0,
                "weight_upload_ms": 1.0,
                "ready_to_generate_ms": 1.0
            },
            "ttft": {
                "end_to_end_ms": 4.0,
                "prompt_render_ms": 0.1,
                "tokenization_ms": 0.2,
                "prefill_ms": 3.0,
                "first_decode_ms": 0.0,
                "sampler_ms": 0.1,
                "stream_first_byte_ms": 4.0
            },
            "tokens": { "prompt": 512, "generated": 64 },
            "throughput": { "tokens_per_second": 10.0 }
        })
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
