use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const DEFAULT_REFERENCE_ROOT: &str = "target/external/BitNet-reference";
const DEFAULT_CPP_ROOT: &str = "target/external/BitNet-reference/3rdparty/llama.cpp";
const DEFAULT_PATCH: &str =
    "ci/reference-instrumentation/bitnet-rs-first-token-hidden-state-main.patch";
const DEFAULT_OUTPUT: &str = "target/a770-diagnostic/bitnet-reference-hidden-state-plan.json";
const DEFAULT_RUN_OUTPUT: &str = "target/a770-diagnostic/bitnet-reference-hidden-state-run.json";
const DEFAULT_REFERENCE_PLAN: &str = "target/a770-diagnostic/bitnet-reference-plan.json";
const DEFAULT_SIDECAR: &str = "target/a770-diagnostic/reference-first-token-hidden-state.json";

const CRITICAL_NOT_CLAIMS: &[&str] = &[
    "selected_attention_residency",
    "resident_kv_decode",
    "attention_scores_residency",
    "softmax_residency",
    "attention_value_mix_residency",
    "full_support_op_residency",
    "full_device_residency",
    "completion",
    "reference_hidden_state_match_rust_cpu",
    "reference_hidden_state_match_strict_a770",
    "reference_parity_promotion",
    "a770_semantic_quality_proven",
];

#[derive(Debug)]
struct HiddenStatePlanArgs {
    cpp_root: PathBuf,
    output: Option<PathBuf>,
    format: String,
}

#[derive(Debug)]
struct HiddenStateRunArgs {
    reference_root: PathBuf,
    cpp_root: PathBuf,
    patch: PathBuf,
    plan: PathBuf,
    sidecar: PathBuf,
    output: Option<PathBuf>,
    format: String,
}

#[derive(Debug)]
struct SourceText {
    path: PathBuf,
    exists: bool,
    read_ok: bool,
    sha256: Option<String>,
    text: String,
}

#[derive(Debug)]
struct HiddenStateSourceSignals {
    llama_get_embeddings_ith: bool,
    llama_n_embd: bool,
    main_rejects_embedding_mode: bool,
    common_context_sets_embeddings_from_params: bool,
    output_reserve_has_logits_switch: bool,
    output_reserve_embeddings_only_when_cparams_embeddings: bool,
    decode_nulls_embeddings_when_not_needed: bool,
    decode_mentions_last_graph_embedding_node: bool,
    common_sampler_sample_anchor: bool,
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
    match args.get(1).map(String::as_str) {
        Some("bitnet-reference-hidden-state-plan") => {
            if args[2..].iter().any(|arg| arg == "-h" || arg == "--help") {
                print_help();
                return Ok(true);
            }
            let opts = parse_args(args)?;
            let report = build_plan(&opts)?;
            if let Some(output) = &opts.output {
                if let Some(parent) = output.parent() {
                    fs::create_dir_all(parent)
                        .with_context(|| format!("creating {}", parent.display()))?;
                }
                fs::write(output, serde_json::to_vec_pretty(&report)?)
                    .with_context(|| format!("writing {}", output.display()))?;
            }
            emit_report(&report, &opts.format)?;
            Ok(true)
        }
        Some("bitnet-reference-hidden-state-run") => {
            if args[2..].iter().any(|arg| arg == "-h" || arg == "--help") {
                print_run_help();
                return Ok(true);
            }
            let opts = parse_run_args(args)?;
            let report = run_instrumented_reference(&opts)?;
            if let Some(output) = &opts.output {
                if let Some(parent) = output.parent() {
                    fs::create_dir_all(parent)
                        .with_context(|| format!("creating {}", parent.display()))?;
                }
                fs::write(output, serde_json::to_vec_pretty(&report)?)
                    .with_context(|| format!("writing {}", output.display()))?;
            }
            emit_report(&report, &opts.format)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn print_help() {
    println!(
        "Plan target-local BitNet reference hidden-state instrumentation for first-token divergence localization\n\nUsage: xtask.exe bitnet-reference-hidden-state-plan [OPTIONS]\n\nOptions:\n      --cpp-root <PATH>  llama.cpp checkout root [default: target/external/BitNet-reference/3rdparty/llama.cpp]\n      --output <PATH>    Output JSON receipt [default: target/a770-diagnostic/bitnet-reference-hidden-state-plan.json]\n      --format <FORMAT>  Output format: human or json [default: human]\n  -h, --help             Print help"
    );
}

fn print_run_help() {
    println!(
        "Temporarily apply BitNet reference first-token hidden-state instrumentation, run the matched reference plan, and restore source worktrees\n\nUsage: xtask.exe bitnet-reference-hidden-state-run [OPTIONS]\n\nOptions:\n      --reference-root <PATH>  BitNet.cpp checkout root [default: target/external/BitNet-reference]\n      --cpp-root <PATH>        llama.cpp checkout root [default: target/external/BitNet-reference/3rdparty/llama.cpp]\n      --patch <PATH>           Hidden-state instrumentation patch [default: ci/reference-instrumentation/bitnet-rs-first-token-hidden-state-main.patch]\n      --plan <PATH>            Reference plan JSON [default: target/a770-diagnostic/bitnet-reference-plan.json]\n      --sidecar <PATH>         First-token hidden-state sidecar JSON [default: target/a770-diagnostic/reference-first-token-hidden-state.json]\n      --output <PATH>          Output JSON receipt [default: target/a770-diagnostic/bitnet-reference-hidden-state-run.json]\n      --format <FORMAT>        Output format: human or json [default: human]\n  -h, --help                   Print help"
    );
}

fn parse_args(args: &[String]) -> Result<HiddenStatePlanArgs> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-hidden-state-plan") {
        bail!("parse_args called for unexpected command");
    }
    let mut cpp_root = PathBuf::from(DEFAULT_CPP_ROOT);
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
            "--output" => output = Some(PathBuf::from(value()?)),
            "--format" => format = value()?,
            other => bail!("unknown bitnet-reference-hidden-state-plan option {other}"),
        }
    }
    Ok(HiddenStatePlanArgs { cpp_root, output, format })
}

fn parse_run_args(args: &[String]) -> Result<HiddenStateRunArgs> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-hidden-state-run") {
        bail!("parse_run_args called for unexpected command");
    }
    let mut reference_root = PathBuf::from(DEFAULT_REFERENCE_ROOT);
    let mut cpp_root = PathBuf::from(DEFAULT_CPP_ROOT);
    let mut patch = PathBuf::from(DEFAULT_PATCH);
    let mut plan = PathBuf::from(DEFAULT_REFERENCE_PLAN);
    let mut sidecar = PathBuf::from(DEFAULT_SIDECAR);
    let mut output = Some(PathBuf::from(DEFAULT_RUN_OUTPUT));
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
            "--reference-root" => reference_root = PathBuf::from(value()?),
            "--cpp-root" => cpp_root = PathBuf::from(value()?),
            "--patch" => patch = PathBuf::from(value()?),
            "--plan" => plan = PathBuf::from(value()?),
            "--sidecar" => sidecar = PathBuf::from(value()?),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--format" => format = value()?,
            other => bail!("unknown bitnet-reference-hidden-state-run option {other}"),
        }
    }
    Ok(HiddenStateRunArgs { reference_root, cpp_root, patch, plan, sidecar, output, format })
}

fn run_instrumented_reference(args: &HiddenStateRunArgs) -> Result<Value> {
    let reference_root = normalize_path(&args.reference_root)?;
    let cpp_root = normalize_path(&args.cpp_root)?;
    let patch = normalize_path(&args.patch)?;
    let plan_path = normalize_path(&args.plan)?;
    let sidecar = normalize_path(&args.sidecar)?;
    let build_dir = reference_root.join("build");
    let selected_exe = build_dir.join("bin").join(exe_name("llama-cli"));
    let generated_lut_header = reference_root.join("include/bitnet-lut-kernels.h");
    let generated_kernel_config = reference_root.join("include/kernel_config.ini");
    let plan_result = read_json(&plan_path);
    let plan_read_success = plan_result.is_ok();
    let plan = plan_result.unwrap_or(Value::Null);
    let reference_argv = reference_argv(&plan).unwrap_or_default();

    if let Some(parent) = sidecar.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    if sidecar.exists() {
        fs::remove_file(&sidecar)
            .with_context(|| format!("removing stale {}", sidecar.display()))?;
    }

    let reference_status_before = git_status(&reference_root);
    let cpp_status_before = git_status(&cpp_root);
    let clean_before = capture_success_empty(&reference_status_before)
        && capture_success_empty(&cpp_status_before);

    let mut blocked_reasons = Vec::<String>::new();
    if !reference_root.is_dir() {
        blocked_reasons.push("reference_root_missing".to_string());
    }
    if !cpp_root.is_dir() {
        blocked_reasons.push("reference_llama_cpp_root_missing".to_string());
    }
    if !patch.is_file() {
        blocked_reasons.push("reference_hidden_state_patch_missing".to_string());
    }
    if !plan_path.is_file() {
        blocked_reasons.push("reference_plan_missing".to_string());
    }
    if plan_path.is_file() && !plan_read_success {
        blocked_reasons.push("reference_plan_json_invalid".to_string());
    }
    if plan_path.is_file() && reference_argv.is_empty() {
        blocked_reasons.push("reference_plan_command_argv_missing".to_string());
    }
    if !build_dir.is_dir() {
        blocked_reasons.push("reference_build_dir_missing".to_string());
    }
    if !clean_before {
        blocked_reasons.push("reference_external_worktree_not_clean_before_run".to_string());
    }

    let generated_lut_header_exists_before = generated_lut_header.is_file();
    let generated_kernel_config_exists_before = generated_kernel_config.is_file();
    let mut generated_lut_header_exists_after_codegen = generated_lut_header_exists_before;
    let mut generated_kernel_config_exists_after_codegen = generated_kernel_config_exists_before;
    let mut compatibility = Vec::<Value>::new();
    let mut codegen_capture = None;
    let mut patch_apply = None;
    let mut build_capture = None;
    let mut run_capture = None;

    if blocked_reasons.is_empty() && !generated_lut_header_exists_before {
        codegen_capture = Some(run_reference_kernel_codegen(&reference_root)?);
        generated_lut_header_exists_after_codegen = generated_lut_header.is_file();
        generated_kernel_config_exists_after_codegen = generated_kernel_config.is_file();
        if !codegen_capture.as_ref().is_some_and(|capture| capture.success) {
            blocked_reasons.push("reference_kernel_codegen_failed".to_string());
        }
        if !generated_lut_header.is_file() {
            blocked_reasons.push("reference_generated_lut_header_missing".to_string());
        }
    }

    if blocked_reasons.is_empty() {
        compatibility = apply_windows_reference_compatibility_fixes(&reference_root)?;
        patch_apply = Some(run_git(&cpp_root, &["apply", &path_to_string(&patch)])?);
        if !patch_apply.as_ref().is_some_and(|capture| capture.success) {
            blocked_reasons.push("reference_hidden_state_patch_apply_failed".to_string());
        }
    }

    if blocked_reasons.is_empty() {
        build_capture = Some(build_reference_cli(&reference_root, &build_dir)?);
        if !build_capture.as_ref().is_some_and(|capture| capture.success) {
            blocked_reasons.push("reference_hidden_state_build_failed".to_string());
        }
    }

    if blocked_reasons.is_empty() {
        if !selected_exe.is_file() {
            blocked_reasons.push("reference_hidden_state_executable_missing".to_string());
        } else if reference_argv.is_empty() {
            blocked_reasons.push("reference_plan_command_argv_missing".to_string());
        } else {
            let mut argv = reference_argv.clone();
            argv[0] = path_to_string(&selected_exe);
            run_capture = Some(run_reference_with_sidecar(&argv, &sidecar)?);
            if !run_capture.as_ref().is_some_and(|capture| capture.success) {
                blocked_reasons.push("reference_hidden_state_run_failed".to_string());
            }
        }
    }

    let sidecar_value = if sidecar.is_file() { Some(read_json(&sidecar)?) } else { None };
    if run_capture.as_ref().is_some_and(|capture| capture.success) && sidecar_value.is_none() {
        blocked_reasons.push("reference_first_token_hidden_state_sidecar_missing".to_string());
    }

    let cleanup_capture = if reference_root.is_dir() && cpp_root.is_dir() {
        Some(cleanup_reference_sources(
            &reference_root,
            &cpp_root,
            &generated_lut_header,
            generated_lut_header_exists_before,
            &generated_kernel_config,
            generated_kernel_config_exists_before,
        )?)
    } else {
        None
    };
    let reference_status_after = git_status(&reference_root);
    let cpp_status_after = git_status(&cpp_root);
    let clean_after =
        capture_success_empty(&reference_status_after) && capture_success_empty(&cpp_status_after);
    if !clean_after {
        blocked_reasons.push("reference_external_worktree_not_clean_after_run".to_string());
    }

    blocked_reasons.sort_unstable();
    blocked_reasons.dedup();
    let reference_hidden_state_available = sidecar_value.is_some() && blocked_reasons.is_empty();

    Ok(json!({
        "schema_version": 1,
        "receipt_type": "bitnet_reference_hidden_state_run",
        "diagnostic": "bitnet_reference_hidden_state_run",
        "producer": "cargo xtask bitnet-reference-hidden-state-run",
        "created_at": chrono::Utc::now().to_rfc3339(),
        "diagnostic_only": true,
        "promotion_allowed": false,
        "claim_allowed": false,
        "classification": "diagnostic_only",
        "paths": {
            "reference_root": path_to_string(&reference_root),
            "cpp_root": path_to_string(&cpp_root),
            "patch": path_to_string(&patch),
            "plan": path_to_string(&plan_path),
            "build_dir": path_to_string(&build_dir),
            "selected_executable": path_to_string(&selected_exe),
            "sidecar": path_to_string(&sidecar),
        },
        "model": plan.pointer("/model").cloned().unwrap_or(Value::Null),
        "prompt_identity": plan.pointer("/prompt_identity").cloned().unwrap_or(Value::Null),
        "preflight": {
            "reference_root_exists": reference_root.is_dir(),
            "cpp_root_exists": cpp_root.is_dir(),
            "patch_exists": patch.is_file(),
            "plan_exists": plan_path.is_file(),
            "build_dir_exists": build_dir.is_dir(),
            "external_worktrees_clean_before_run": clean_before,
            "generated_lut_header": {
                "path": path_to_string(&generated_lut_header),
                "exists_before": generated_lut_header_exists_before,
                "exists_after_codegen": generated_lut_header_exists_after_codegen,
            },
            "generated_kernel_config": {
                "path": path_to_string(&generated_kernel_config),
                "exists_before": generated_kernel_config_exists_before,
                "exists_after_codegen": generated_kernel_config_exists_after_codegen,
            },
            "reference_status_before": capture_json(reference_status_before.as_ref()),
            "cpp_status_before": capture_json(cpp_status_before.as_ref()),
        },
        "kernel_codegen": capture_json(codegen_capture.as_ref()),
        "compatibility_fixes": compatibility,
        "patch_apply": capture_json(patch_apply.as_ref()),
        "build": capture_json(build_capture.as_ref()),
        "reference_run": capture_json(run_capture.as_ref()),
        "sidecar": {
            "exists": sidecar.is_file(),
            "sha256": sidecar.is_file().then(|| sha256_bytes(&fs::read(&sidecar).unwrap_or_default())),
            "receipt": sidecar_value,
            "policy": "reference-side first-token hidden state is diagnostic evidence only until compared with Rust CPU and strict A770 hidden-state receipts",
        },
        "cleanup": {
            "source_restore": capture_json(cleanup_capture.as_ref()),
            "external_worktrees_clean_after_run": clean_after,
            "reference_status_after": capture_json(reference_status_after.as_ref()),
            "cpp_status_after": capture_json(cpp_status_after.as_ref()),
        },
        "decision": {
            "reference_hidden_state_available": reference_hidden_state_available,
            "current_blocked_reasons": blocked_reasons,
            "next_when_available": "compare reference first-token hidden state against Rust CPU and strict A770 hidden-state receipts before changing Rust model math",
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    }))
}

fn build_plan(args: &HiddenStatePlanArgs) -> Result<Value> {
    let cpp_root = normalize_path(&args.cpp_root)?;
    let llama_h = read_source(cpp_root.join("include/llama.h"));
    let llama_cpp = read_source(cpp_root.join("src/llama.cpp"));
    let common_cpp = read_source(cpp_root.join("common/common.cpp"));
    let main_cpp = read_source(cpp_root.join("examples/main/main.cpp"));
    let signals = source_signals(&llama_h.text, &llama_cpp.text, &common_cpp.text, &main_cpp.text);
    let mut blocked_reasons = Vec::new();

    if !cpp_root.is_dir() {
        blocked_reasons.push("reference_llama_cpp_root_missing".to_string());
    }
    for source in [&llama_h, &llama_cpp, &common_cpp, &main_cpp] {
        if !source.exists {
            blocked_reasons.push(format!("source_missing:{}", path_to_string(&source.path)));
        } else if !source.read_ok {
            blocked_reasons.push(format!("source_read_failed:{}", path_to_string(&source.path)));
        }
    }
    if !signals.llama_get_embeddings_ith {
        blocked_reasons.push("llama_get_embeddings_ith_api_missing".to_string());
    }
    if !signals.llama_n_embd {
        blocked_reasons.push("llama_n_embd_api_missing".to_string());
    }
    if !signals.main_rejects_embedding_mode {
        blocked_reasons.push("llama_cli_embedding_rejection_anchor_missing".to_string());
    }
    if !signals.common_context_sets_embeddings_from_params {
        blocked_reasons.push("common_context_embedding_flag_anchor_missing".to_string());
    }
    if !signals.output_reserve_embeddings_only_when_cparams_embeddings {
        blocked_reasons.push("output_reserve_embedding_gate_anchor_missing".to_string());
    }
    if !signals.decode_nulls_embeddings_when_not_needed {
        blocked_reasons.push("decode_embedding_null_anchor_missing".to_string());
    }
    if !signals.common_sampler_sample_anchor {
        blocked_reasons.push("sampler_first_token_anchor_missing".to_string());
    }
    blocked_reasons.push("stock_reference_generation_hidden_state_not_extracted".to_string());
    blocked_reasons.sort_unstable();
    blocked_reasons.dedup();

    let source_anchors_ready = signals.llama_get_embeddings_ith
        && signals.llama_n_embd
        && signals.main_rejects_embedding_mode
        && signals.common_context_sets_embeddings_from_params
        && signals.output_reserve_embeddings_only_when_cparams_embeddings
        && signals.decode_nulls_embeddings_when_not_needed
        && signals.common_sampler_sample_anchor;

    Ok(json!({
        "schema_version": 1,
        "diagnostic": "bitnet_reference_hidden_state_plan",
        "diagnostic_only": true,
        "promotion_allowed": false,
        "claim_allowed": false,
        "classification": "diagnostic_only",
        "producer": "cargo xtask bitnet-reference-hidden-state-plan",
        "created_at": chrono::Utc::now().to_rfc3339(),
        "inputs": {
            "cpp_root": path_to_string(&cpp_root),
            "llama_h": source_receipt(&llama_h),
            "llama_cpp": source_receipt(&llama_cpp),
            "common_cpp": source_receipt(&common_cpp),
            "main_cpp": source_receipt(&main_cpp),
        },
        "source_capability": {
            "public_api": {
                "llama_get_embeddings_ith": signals.llama_get_embeddings_ith,
                "llama_n_embd": signals.llama_n_embd,
            },
            "generation_path": {
                "llama_cli_rejects_embedding_mode": signals.main_rejects_embedding_mode,
                "common_context_sets_embeddings_from_params_embedding": signals.common_context_sets_embeddings_from_params,
                "output_reserve_has_logits_switch": signals.output_reserve_has_logits_switch,
                "output_reserve_embeddings_only_when_cparams_embeddings": signals.output_reserve_embeddings_only_when_cparams_embeddings,
                "decode_nulls_embeddings_when_not_needed": signals.decode_nulls_embeddings_when_not_needed,
                "decode_mentions_last_graph_embedding_node": signals.decode_mentions_last_graph_embedding_node,
                "common_sampler_sample_anchor": signals.common_sampler_sample_anchor,
            },
            "stock_generation_hidden_state_available": false,
            "reason": "llama-cli generation keeps cparams.embeddings=false, and llama.cpp does not extract token embeddings on the logits generation path",
        },
        "instrumentation_plan": {
            "target_local_only": true,
            "target_files": [
                "target/external/BitNet-reference/3rdparty/llama.cpp/src/llama.cpp",
                "target/external/BitNet-reference/3rdparty/llama.cpp/examples/main/main.cpp"
            ],
            "environment_variable": "BITNET_RS_REFERENCE_FIRST_TOKEN_HIDDEN_STATE",
            "receipt_type_when_applied": "bitnet_reference_first_token_hidden_state",
            "required_anchors": {
                "llama_h": ["llama_get_embeddings_ith", "llama_n_embd"],
                "llama_cpp": [
                    "const bool has_logits = !cparams.embeddings",
                    "const bool has_embd   =  cparams.embeddings && (cparams.pooling_type == LLAMA_POOLING_TYPE_NONE)",
                    "embd = nullptr; // do not extract embeddings when not needed",
                    "struct ggml_tensor * embd = ggml_graph_node(gf, -2)"
                ],
                "main_cpp": [
                    "params.embedding",
                    "common_sampler_sample(smpl, ctx, -1)"
                ]
            },
            "captures": [
                "prompt_token_count",
                "n_embd",
                "final pre-logits hidden-state stats for first generated token",
                "first 16 hidden-state values for orientation",
                "vector hash if the reference patch adds a stable byte hash helper"
            ],
            "not_claim": "hidden-state sidecar will localize reference/Rust divergence only; it will not prove semantic quality or A770 support",
        },
        "decision": {
            "reference_hidden_state_available": false,
            "stock_api_generation_hidden_state_available": false,
            "source_anchors_ready_for_target_local_patch": source_anchors_ready,
            "current_blocked_reasons": blocked_reasons,
            "next_action": "add a target-local reference hidden-state instrumentation patch, run the matched prompt, then compare reference hidden state against Rust CPU and strict A770 hidden-state receipts",
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    }))
}

fn source_signals(
    llama_h: &str,
    llama_cpp: &str,
    common_cpp: &str,
    main_cpp: &str,
) -> HiddenStateSourceSignals {
    HiddenStateSourceSignals {
        llama_get_embeddings_ith: llama_h.contains("llama_get_embeddings_ith"),
        llama_n_embd: llama_h.contains("llama_n_embd"),
        main_rejects_embedding_mode: main_cpp.contains("params.embedding")
            && main_cpp.contains("please use the 'embedding' tool for embedding calculations"),
        common_context_sets_embeddings_from_params: common_cpp.contains("cparams.embeddings")
            && common_cpp.contains("params.embedding"),
        output_reserve_has_logits_switch: llama_cpp.contains("const bool has_logits = !cparams.embeddings"),
        output_reserve_embeddings_only_when_cparams_embeddings: llama_cpp
            .contains("const bool has_embd   =  cparams.embeddings && (cparams.pooling_type == LLAMA_POOLING_TYPE_NONE)"),
        decode_nulls_embeddings_when_not_needed: llama_cpp
            .contains("embd = nullptr; // do not extract embeddings when not needed"),
        decode_mentions_last_graph_embedding_node: llama_cpp
            .contains("struct ggml_tensor * embd = ggml_graph_node(gf, -2)"),
        common_sampler_sample_anchor: main_cpp.contains("common_sampler_sample(smpl, ctx, -1)"),
    }
}

fn read_source(path: PathBuf) -> SourceText {
    let exists = path.is_file();
    let read = fs::read(&path);
    let read_ok = read.is_ok();
    let bytes = read.unwrap_or_default();
    let sha256 = read_ok.then(|| sha256_bytes(&bytes));
    let text = String::from_utf8_lossy(&bytes).into_owned();
    SourceText { path, exists, read_ok, sha256, text }
}

fn source_receipt(source: &SourceText) -> Value {
    json!({
        "path": path_to_string(&source.path),
        "exists": source.exists,
        "read_ok": source.read_ok,
        "sha256": source.sha256,
    })
}

fn read_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

fn reference_argv(plan: &Value) -> Result<Vec<String>> {
    plan.pointer("/reference/command_argv")
        .and_then(Value::as_array)
        .context("plan missing /reference/command_argv")?
        .iter()
        .map(|item| {
            item.as_str().map(ToOwned::to_owned).context("command_argv item is not a string")
        })
        .collect()
}

fn apply_windows_reference_compatibility_fixes(reference_root: &Path) -> Result<Vec<Value>> {
    let mut applied = Vec::new();
    applied.push(replace_file_text(
        &reference_root.join("src/ggml-bitnet-mad.cpp"),
        "        int8_t * y_col = y + col * by;",
        "        const int8_t * y_col = y + col * by;",
        "windows_const_compatibility",
    )?);
    applied.push(insert_after_if_missing(
        &reference_root.join("3rdparty/llama.cpp/common/common.cpp"),
        "#include <ctime>",
        "#include <chrono>",
        "windows_common_chrono_include",
    )?);
    applied.push(insert_after_if_missing(
        &reference_root.join("3rdparty/llama.cpp/common/log.cpp"),
        "#include <condition_variable>",
        "#include <chrono>",
        "windows_log_chrono_include",
    )?);
    Ok(applied)
}

fn replace_file_text(path: &Path, before: &str, after: &str, fix_id: &str) -> Result<Value> {
    let content =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    if content.contains(before) {
        fs::write(path, content.replace(before, after))
            .with_context(|| format!("writing {}", path.display()))?;
        return Ok(json!({
            "fix_id": fix_id,
            "path": path_to_string(path),
            "applied": true,
            "already_present": false,
        }));
    }
    if content.contains(after) {
        return Ok(json!({
            "fix_id": fix_id,
            "path": path_to_string(path),
            "applied": false,
            "already_present": true,
        }));
    }
    bail!("expected compatibility fix target not found in {}", path.display())
}

fn insert_after_if_missing(path: &Path, anchor: &str, insert: &str, fix_id: &str) -> Result<Value> {
    let content =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    if content.contains(insert) {
        return Ok(json!({
            "fix_id": fix_id,
            "path": path_to_string(path),
            "applied": false,
            "already_present": true,
        }));
    }
    if !content.contains(anchor) {
        bail!("expected compatibility include anchor not found in {}", path.display());
    }
    fs::write(path, content.replace(anchor, &format!("{anchor}\n{insert}")))
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(json!({
        "fix_id": fix_id,
        "path": path_to_string(path),
        "applied": true,
        "already_present": false,
    }))
}

fn run_reference_kernel_codegen(reference_root: &Path) -> Result<CommandCapture> {
    run_command(Command::new("python").current_dir(reference_root).args([
        "utils/codegen_tl2.py",
        "--model",
        "bitnet_b1_58-3B",
        "--BM",
        "160,320,320",
        "--BK",
        "96,96,96",
        "--bm",
        "32,32,32",
    ]))
}

fn build_reference_cli(reference_root: &Path, build_dir: &Path) -> Result<CommandCapture> {
    if cfg!(windows) {
        let vsdevcmd = find_vsdevcmd();
        let Some(vsdevcmd) = vsdevcmd else {
            return Ok(CommandCapture {
                status_code: None,
                success: false,
                stdout: String::new(),
                stderr: "Visual Studio developer command prompt not found".to_string(),
            });
        };
        let script = reference_root.join("build_bitnet_rs_hidden_state_reference.cmd");
        let lines = [
            "@echo off".to_string(),
            format!("call {} -arch=x64 -host_arch=x64 || exit /b 1", cmd_quote(&vsdevcmd)),
            format!(
                "cmake --build {} --config Release --target llama-cli || exit /b 1",
                cmd_quote(build_dir)
            ),
        ];
        fs::write(&script, lines.join("\r\n"))
            .with_context(|| format!("writing {}", script.display()))?;
        let capture =
            run_command(Command::new("cmd.exe").args(["/d", "/c"]).arg(path_to_string(&script)))?;
        let _ = fs::remove_file(&script);
        Ok(capture)
    } else {
        run_command(Command::new("cmake").args([
            "--build",
            &path_to_string(build_dir),
            "--config",
            "Release",
            "--target",
            "llama-cli",
        ]))
    }
}

fn find_vsdevcmd() -> Option<PathBuf> {
    let program_files_x86 = std::env::var_os("ProgramFiles(x86)")?;
    let vswhere =
        PathBuf::from(program_files_x86).join("Microsoft Visual Studio/Installer/vswhere.exe");
    if !vswhere.is_file() {
        return None;
    }
    let output = Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .stdin(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if path.is_empty() {
        return None;
    }
    let vsdevcmd = PathBuf::from(path).join("Common7/Tools/VsDevCmd.bat");
    vsdevcmd.is_file().then_some(vsdevcmd)
}

fn run_reference_with_sidecar(argv: &[String], sidecar: &Path) -> Result<CommandCapture> {
    let executable = argv.first().context("empty reference command")?;
    let mut command = Command::new(executable);
    command
        .args(&argv[1..])
        .env("BITNET_RS_REFERENCE_FIRST_TOKEN_HIDDEN_STATE", sidecar)
        .stdin(Stdio::null());
    run_command(&mut command)
}

fn cleanup_reference_sources(
    reference_root: &Path,
    cpp_root: &Path,
    generated_lut_header: &Path,
    generated_lut_header_existed_before: bool,
    generated_kernel_config: &Path,
    generated_kernel_config_existed_before: bool,
) -> Result<CommandCapture> {
    let mut capture = run_git(reference_root, &["restore", "--", "src/ggml-bitnet-mad.cpp"])?;
    let cpp_capture = run_git(
        cpp_root,
        &[
            "restore",
            "--",
            "common/common.cpp",
            "common/log.cpp",
            "examples/main/main.cpp",
            "src/llama.cpp",
        ],
    )?;
    if !generated_lut_header_existed_before && generated_lut_header.exists() {
        fs::remove_file(generated_lut_header)
            .with_context(|| format!("removing {}", generated_lut_header.display()))?;
    }
    if !generated_kernel_config_existed_before && generated_kernel_config.exists() {
        fs::remove_file(generated_kernel_config)
            .with_context(|| format!("removing {}", generated_kernel_config.display()))?;
    }
    capture.success = capture.success && cpp_capture.success;
    capture.status_code = if capture.status_code == Some(0) && cpp_capture.status_code == Some(0) {
        Some(0)
    } else {
        cpp_capture.status_code.or(capture.status_code)
    };
    if !cpp_capture.stdout.is_empty() {
        capture.stdout.push_str(&cpp_capture.stdout);
    }
    if !cpp_capture.stderr.is_empty() {
        capture.stderr.push_str(&cpp_capture.stderr);
    }
    Ok(capture)
}

fn run_command(command: &mut Command) -> Result<CommandCapture> {
    let output = command.output().with_context(|| format!("running command {:?}", command))?;
    Ok(CommandCapture {
        status_code: output.status.code(),
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

fn git_status(path: &Path) -> Option<CommandCapture> {
    path.is_dir().then(|| run_git(path, &["status", "--porcelain"]).ok()).flatten()
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

fn capture_success_empty(capture: &Option<CommandCapture>) -> bool {
    capture.as_ref().is_some_and(|capture| capture.success && capture.stdout.trim().is_empty())
}

fn capture_json(capture: Option<&CommandCapture>) -> Value {
    match capture {
        Some(capture) => json!({
            "success": capture.success,
            "exit_code": capture.status_code,
            "stdout": capture.stdout.trim(),
            "stderr": capture.stderr.trim(),
        }),
        None => Value::Null,
    }
}

fn exe_name(stem: &str) -> String {
    if cfg!(windows) { format!("{stem}.exe") } else { stem.to_string() }
}

fn cmd_quote(path: &Path) -> String {
    format!("\"{}\"", path_to_string(path).replace('"', "\"\""))
}

fn normalize_path(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        path.canonicalize().with_context(|| format!("canonicalizing {}", path.display()))
    } else if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn path_to_string(path: &Path) -> String {
    let path = path.to_string_lossy().replace('\\', "/");
    path.strip_prefix("//?/").unwrap_or(&path).to_string()
}

fn emit_report(report: &Value, format: &str) -> Result<()> {
    match format {
        "json" => {
            println!("{}", serde_json::to_string_pretty(report)?);
            Ok(())
        }
        "human" => {
            println!(
                "bitnet reference hidden-state plan: diagnostic_only=true claim_allowed=false"
            );
            println!(
                "stock generation hidden state available: {}",
                report["decision"]["stock_api_generation_hidden_state_available"]
                    .as_bool()
                    .unwrap_or(false)
            );
            println!(
                "source anchors ready: {}",
                report["decision"]["source_anchors_ready_for_target_local_patch"]
                    .as_bool()
                    .unwrap_or(false)
            );
            if let Some(reasons) = report["decision"]["current_blocked_reasons"].as_array() {
                for reason in reasons {
                    if let Some(reason) = reason.as_str() {
                        println!("- {reason}");
                    }
                }
            }
            Ok(())
        }
        other => bail!("unsupported bitnet-reference-hidden-state-plan output format: {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn source_signals_detect_generation_hidden_state_blocker() {
        let signals = source_signals(
            "LLAMA_API int32_t llama_n_embd(const struct llama_model * model); LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);",
            "const bool has_logits = !cparams.embeddings; const bool has_embd   =  cparams.embeddings && (cparams.pooling_type == LLAMA_POOLING_TYPE_NONE); struct ggml_tensor * embd = ggml_graph_node(gf, -2); embd = nullptr; // do not extract embeddings when not needed",
            "cparams.embeddings        = params.embedding;",
            "if (params.embedding) { LOG_ERR(\"please use the 'embedding' tool for embedding calculations\"); } const llama_token id = common_sampler_sample(smpl, ctx, -1);",
        );

        assert!(signals.llama_get_embeddings_ith);
        assert!(signals.llama_n_embd);
        assert!(signals.main_rejects_embedding_mode);
        assert!(signals.common_context_sets_embeddings_from_params);
        assert!(signals.output_reserve_embeddings_only_when_cparams_embeddings);
        assert!(signals.decode_nulls_embeddings_when_not_needed);
        assert!(signals.common_sampler_sample_anchor);
    }

    #[test]
    fn missing_sources_keep_plan_diagnostic_and_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let report = build_plan(&HiddenStatePlanArgs {
            cpp_root: dir.path().join("missing"),
            output: None,
            format: "json".to_string(),
        })
        .unwrap();
        let reasons = report["decision"]["current_blocked_reasons"].as_array().unwrap();

        assert_eq!(report["claim_allowed"], json!(false));
        assert!(reasons.contains(&json!("reference_llama_cpp_root_missing")));
        assert!(reasons.contains(&json!("llama_get_embeddings_ith_api_missing")));
        assert!(reasons.contains(&json!("stock_reference_generation_hidden_state_not_extracted")));
    }
}
