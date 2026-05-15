use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const DEFAULT_REFERENCE_ROOT: &str = "target/external/BitNet-reference";
const DEFAULT_CPP_ROOT: &str = "target/external/BitNet-reference/3rdparty/llama.cpp";
const DEFAULT_RUST_TRANSFORMER: &str = "crates/bitnet-transformer/src/lib.rs";
const DEFAULT_PATCH: &str = "ci/reference-instrumentation/bitnet-rs-layer-trace-main.patch";
const DEFAULT_OUTPUT: &str = "target/a770-diagnostic/bitnet-reference-layer-trace-plan.json";
const DEFAULT_RUN_OUTPUT: &str = "target/a770-diagnostic/bitnet-reference-layer-trace-run.json";
const DEFAULT_REFERENCE_PLAN: &str = "target/a770-diagnostic/bitnet-reference-plan.json";
const DEFAULT_SIDECAR: &str = "target/a770-diagnostic/reference-first-token-layer-trace.json";
const DEFAULT_COMPARE_OUTPUT: &str =
    "target/a770-diagnostic/bitnet-reference-layer-trace-compare.json";
const DEFAULT_RUST_CAPTURE_OUTPUT: &str =
    "target/a770-diagnostic/bitnet-reference-layer-trace-rust-capture.json";
const DEFAULT_CPU_TRACE_DIR: &str = "target/a770-diagnostic/reference-layer-trace-rust-cpu";
const DEFAULT_A770_TRACE_DIR: &str = "target/a770-diagnostic/reference-layer-trace-rust-a770";

const CRITICAL_NOT_CLAIMS: &[&str] = &[
    "selected_attention_residency",
    "resident_kv_decode",
    "attention_scores_residency",
    "softmax_residency",
    "attention_value_mix_residency",
    "full_support_op_residency",
    "full_device_residency",
    "completion",
    "reference_trace_match_rust_cpu",
    "reference_trace_match_strict_a770",
    "reference_parity_promotion",
    "a770_semantic_quality_proven",
];

const REFERENCE_REQUIRED_ANCHORS: &[(&str, &str)] = &[
    ("bitnet_b158_builder", "struct ggml_cgraph * build_bitnet_158()"),
    ("bitnet_b158_dispatch", "result = llm.build_bitnet_158()"),
    ("graph_callback_type", "using llm_build_cb = std::function"),
    ("graph_callback_set_name", "ggml_set_name(cur, name)"),
    ("input_embedding", "cb(inpL, \"inp_embd\", -1)"),
    ("attention_norm", "cb(cur, \"attn_norm\", il)"),
    ("query_projection", "cb(Qcur, \"Qcur\", il)"),
    ("key_projection", "cb(Kcur, \"Kcur\", il)"),
    ("value_projection", "cb(Vcur, \"Vcur\", il)"),
    ("attention_subnorm", "cb(cur, \"attn_sub_norm\", il)"),
    ("attention_output", "cb(cur, \"attn_o_out\", il)"),
    ("attention_residual", "cb(ffn_inp, \"ffn_inp\", il)"),
    ("ffn_norm", "cb(cur, \"ffn_norm\", il)"),
    ("ffn_parallel_output", "cb(cur, \"ffn_out\", il)"),
    ("ffn_subnorm", "cb(cur, \"ffn_sub_norm\", il)"),
    ("ffn_down", "cb(cur, \"ffn_down\", il)"),
    ("layer_output", "cb(cur, \"l_out\", il)"),
    ("final_norm", "cb(cur, \"result_norm\", -1)"),
    ("result_output", "cb(cur, \"result_output\", -1)"),
];

const RUST_REQUIRED_ANCHORS: &[(&str, &str)] = &[
    ("trace_feature_gate", "#[cfg(feature = \"trace\")]"),
    ("trace_layer0_helper", "fn trace_layer0_tensor"),
    ("input_embedding", "t0/embeddings"),
    ("attention_norm", "attn_norm"),
    ("query_projection", "attention_q"),
    ("key_projection", "attention_k"),
    ("value_projection", "attention_v"),
    ("attention_value_mix", "attention_value_mix"),
    ("attention_subnorm", "post_attention_subnorm"),
    ("attention_output", "post_o_proj"),
    ("attention_residual", "post_attention_residual"),
    ("pre_ffn_norm", "pre_ffn_norm"),
    ("ffn_norm", "post_ffn_norm"),
    ("ffn_gate", "post_ffn_gate_proj"),
    ("ffn_activation", "post_ffn_gate_activation"),
    ("ffn_up", "post_ffn_up_proj"),
    ("ffn_parallel_output", "post_swiglu"),
    ("ffn_subnorm", "post_ffn_subnorm"),
    ("ffn_down", "post_down_proj"),
    ("layer_output", "post_layer"),
    ("final_norm", "final_norm"),
];

#[derive(Debug)]
struct LayerTracePlanArgs {
    cpp_root: PathBuf,
    rust_transformer: PathBuf,
    output: Option<PathBuf>,
    format: String,
}

#[derive(Debug)]
struct LayerTraceRunArgs {
    reference_root: PathBuf,
    cpp_root: PathBuf,
    patch: PathBuf,
    plan: PathBuf,
    sidecar: PathBuf,
    output: Option<PathBuf>,
    format: String,
}

#[derive(Debug)]
struct LayerTraceCompareArgs {
    reference: PathBuf,
    cpu_trace_dir: PathBuf,
    a770_trace_dir: Option<PathBuf>,
    output: Option<PathBuf>,
    format: String,
}

#[derive(Debug)]
struct LayerTraceRustCaptureArgs {
    plan: PathBuf,
    cpu_trace_dir: PathBuf,
    a770_trace_dir: PathBuf,
    skip_a770: bool,
    overwrite: bool,
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
struct CommandCapture {
    status_code: Option<i32>,
    success: bool,
    stdout: String,
    stderr: String,
}

#[derive(Debug, Clone)]
struct ReferenceTraceRecord {
    name: String,
    stage: String,
    graph_index: Option<i64>,
    layer: Option<i64>,
    dtype: String,
    shape: Vec<i64>,
    nelements: u64,
    rms: Option<f64>,
    values_available: bool,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct RustTraceRecord {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    #[allow(dead_code)]
    blake3: String,
    rms: f64,
    num_elements: usize,
    #[allow(dead_code)]
    seq: Option<usize>,
    layer: Option<isize>,
    stage: Option<String>,
}

pub fn maybe_dispatch_from_env() -> Result<bool> {
    let args = std::env::args().collect::<Vec<_>>();
    maybe_dispatch(&args)
}

fn maybe_dispatch(args: &[String]) -> Result<bool> {
    match args.get(1).map(String::as_str) {
        Some("bitnet-reference-layer-trace-plan") => {
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
        Some("bitnet-reference-layer-trace-run") => {
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
        Some("bitnet-reference-layer-trace-compare") => {
            if args[2..].iter().any(|arg| arg == "-h" || arg == "--help") {
                print_compare_help();
                return Ok(true);
            }
            let opts = parse_compare_args(args)?;
            let report = compare_reference_layer_trace(&opts)?;
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
        Some("bitnet-reference-layer-trace-capture-rust") => {
            if args[2..].iter().any(|arg| arg == "-h" || arg == "--help") {
                print_rust_capture_help();
                return Ok(true);
            }
            let opts = parse_rust_capture_args(args)?;
            let report = capture_rust_layer_traces(&opts)?;
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
        "Plan target-local BitNet reference layer/stage trace instrumentation\n\nUsage: xtask.exe bitnet-reference-layer-trace-plan [OPTIONS]\n\nOptions:\n      --cpp-root <PATH>          llama.cpp checkout root [default: target/external/BitNet-reference/3rdparty/llama.cpp]\n      --rust-transformer <PATH>  Rust transformer source [default: crates/bitnet-transformer/src/lib.rs]\n      --output <PATH>            Output JSON receipt [default: target/a770-diagnostic/bitnet-reference-layer-trace-plan.json]\n      --format <FORMAT>          Output format: human or json [default: human]\n  -h, --help                     Print help"
    );
}

fn print_run_help() {
    println!(
        "Temporarily apply BitNet reference layer trace instrumentation, run the matched reference plan, and restore source worktrees\n\nUsage: xtask.exe bitnet-reference-layer-trace-run [OPTIONS]\n\nOptions:\n      --reference-root <PATH>  BitNet.cpp checkout root [default: target/external/BitNet-reference]\n      --cpp-root <PATH>        llama.cpp checkout root [default: target/external/BitNet-reference/3rdparty/llama.cpp]\n      --patch <PATH>           Layer-trace instrumentation patch [default: ci/reference-instrumentation/bitnet-rs-layer-trace-main.patch]\n      --plan <PATH>            Reference plan JSON [default: target/a770-diagnostic/bitnet-reference-plan.json]\n      --sidecar <PATH>         Layer-trace sidecar JSON [default: target/a770-diagnostic/reference-first-token-layer-trace.json]\n      --output <PATH>          Output JSON receipt [default: target/a770-diagnostic/bitnet-reference-layer-trace-run.json]\n      --format <FORMAT>        Output format: human or json [default: human]\n  -h, --help                   Print help"
    );
}

fn print_compare_help() {
    println!(
        "Compare a BitNet reference layer trace receipt against Rust CPU/A770 trace directories\n\nUsage: xtask.exe bitnet-reference-layer-trace-compare [OPTIONS]\n\nOptions:\n      --reference <PATH>       Reference layer-trace run or sidecar JSON [default: target/a770-diagnostic/bitnet-reference-layer-trace-run.json]\n      --cpu-trace-dir <PATH>   Rust CPU BITNET_TRACE_DIR output\n      --a770-trace-dir <PATH>  Optional strict A770 BITNET_TRACE_DIR output\n      --output <PATH>          Output JSON receipt [default: target/a770-diagnostic/bitnet-reference-layer-trace-compare.json]\n      --format <FORMAT>        Output format: human or json [default: human]\n  -h, --help                   Print help"
    );
}

fn print_rust_capture_help() {
    println!(
        "Run the Rust CPU and strict A770 commands from the matched reference plan with BITNET_TRACE_DIR set\n\nUsage: xtask.exe bitnet-reference-layer-trace-capture-rust [OPTIONS]\n\nOptions:\n      --plan <PATH>            Reference plan JSON [default: target/a770-diagnostic/bitnet-reference-plan.json]\n      --cpu-trace-dir <PATH>   Rust CPU BITNET_TRACE_DIR output [default: target/a770-diagnostic/reference-layer-trace-rust-cpu]\n      --a770-trace-dir <PATH>  Strict A770 BITNET_TRACE_DIR output [default: target/a770-diagnostic/reference-layer-trace-rust-a770]\n      --skip-a770              Capture CPU trace only and report strict A770 as skipped\n      --overwrite              Remove existing top-level .trace files from output trace directories before running\n      --output <PATH>          Output JSON receipt [default: target/a770-diagnostic/bitnet-reference-layer-trace-rust-capture.json]\n      --format <FORMAT>        Output format: human or json [default: human]\n  -h, --help                   Print help"
    );
}

fn parse_args(args: &[String]) -> Result<LayerTracePlanArgs> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-layer-trace-plan") {
        bail!("parse_args called for unexpected command");
    }

    let mut cpp_root = PathBuf::from(DEFAULT_CPP_ROOT);
    let mut rust_transformer = PathBuf::from(DEFAULT_RUST_TRANSFORMER);
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
            "--rust-transformer" => rust_transformer = PathBuf::from(value()?),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--format" => format = value()?,
            other => bail!("unknown bitnet-reference-layer-trace-plan option {other}"),
        }
    }

    Ok(LayerTracePlanArgs { cpp_root, rust_transformer, output, format })
}

fn parse_run_args(args: &[String]) -> Result<LayerTraceRunArgs> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-layer-trace-run") {
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
            other => bail!("unknown bitnet-reference-layer-trace-run option {other}"),
        }
    }
    Ok(LayerTraceRunArgs { reference_root, cpp_root, patch, plan, sidecar, output, format })
}

fn parse_compare_args(args: &[String]) -> Result<LayerTraceCompareArgs> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-layer-trace-compare") {
        bail!("parse_compare_args called for unexpected command");
    }
    let mut reference = PathBuf::from(DEFAULT_RUN_OUTPUT);
    let mut cpu_trace_dir = None::<PathBuf>;
    let mut a770_trace_dir = None::<PathBuf>;
    let mut output = Some(PathBuf::from(DEFAULT_COMPARE_OUTPUT));
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
            "--reference" => reference = PathBuf::from(value()?),
            "--cpu-trace-dir" => cpu_trace_dir = Some(PathBuf::from(value()?)),
            "--a770-trace-dir" => a770_trace_dir = Some(PathBuf::from(value()?)),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--format" => format = value()?,
            other => bail!("unknown bitnet-reference-layer-trace-compare option {other}"),
        }
    }
    let cpu_trace_dir = cpu_trace_dir.context("--cpu-trace-dir is required")?;
    Ok(LayerTraceCompareArgs { reference, cpu_trace_dir, a770_trace_dir, output, format })
}

fn parse_rust_capture_args(args: &[String]) -> Result<LayerTraceRustCaptureArgs> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-layer-trace-capture-rust") {
        bail!("parse_rust_capture_args called for unexpected command");
    }
    let mut plan = PathBuf::from(DEFAULT_REFERENCE_PLAN);
    let mut cpu_trace_dir = PathBuf::from(DEFAULT_CPU_TRACE_DIR);
    let mut a770_trace_dir = PathBuf::from(DEFAULT_A770_TRACE_DIR);
    let mut skip_a770 = false;
    let mut overwrite = false;
    let mut output = Some(PathBuf::from(DEFAULT_RUST_CAPTURE_OUTPUT));
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
            "--cpu-trace-dir" => cpu_trace_dir = PathBuf::from(value()?),
            "--a770-trace-dir" => a770_trace_dir = PathBuf::from(value()?),
            "--skip-a770" => skip_a770 = true,
            "--overwrite" => overwrite = true,
            "--output" => output = Some(PathBuf::from(value()?)),
            "--format" => format = value()?,
            other => bail!("unknown bitnet-reference-layer-trace-capture-rust option {other}"),
        }
    }
    Ok(LayerTraceRustCaptureArgs {
        plan,
        cpu_trace_dir,
        a770_trace_dir,
        skip_a770,
        overwrite,
        output,
        format,
    })
}

fn run_instrumented_reference(args: &LayerTraceRunArgs) -> Result<Value> {
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
        blocked_reasons.push("reference_layer_trace_patch_missing".to_string());
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
            blocked_reasons.push("reference_layer_trace_patch_apply_failed".to_string());
        }
    }

    if blocked_reasons.is_empty() {
        build_capture = Some(build_reference_cli(&reference_root, &build_dir)?);
        if !build_capture.as_ref().is_some_and(|capture| capture.success) {
            blocked_reasons.push("reference_layer_trace_build_failed".to_string());
        }
    }

    if blocked_reasons.is_empty() {
        if !selected_exe.is_file() {
            blocked_reasons.push("reference_layer_trace_executable_missing".to_string());
        } else if reference_argv.is_empty() {
            blocked_reasons.push("reference_plan_command_argv_missing".to_string());
        } else {
            let mut argv = reference_argv.clone();
            argv[0] = path_to_string(&selected_exe);
            run_capture = Some(run_reference_with_sidecar(&argv, &sidecar)?);
            if !run_capture.as_ref().is_some_and(|capture| capture.success) {
                blocked_reasons.push("reference_layer_trace_run_failed".to_string());
            }
        }
    }

    let sidecar_value = if sidecar.is_file() { Some(read_json(&sidecar)?) } else { None };
    if run_capture.as_ref().is_some_and(|capture| capture.success) && sidecar_value.is_none() {
        blocked_reasons.push("reference_first_token_layer_trace_sidecar_missing".to_string());
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
    let record_count = sidecar_value
        .as_ref()
        .and_then(|sidecar| sidecar.pointer("/records"))
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let reference_layer_trace_available =
        sidecar_value.is_some() && record_count > 0 && blocked_reasons.is_empty();

    Ok(json!({
        "schema_version": 1,
        "receipt_type": "bitnet_reference_layer_trace_run",
        "diagnostic": "bitnet_reference_layer_trace_run",
        "producer": "cargo xtask bitnet-reference-layer-trace-run",
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
            "record_count": record_count,
            "receipt": sidecar_value,
            "policy": "reference-side layer trace is diagnostic evidence only until compared with Rust CPU and strict A770 layer traces",
        },
        "cleanup": {
            "source_restore": capture_json(cleanup_capture.as_ref()),
            "external_worktrees_clean_after_run": clean_after,
            "reference_status_after": capture_json(reference_status_after.as_ref()),
            "cpp_status_after": capture_json(cpp_status_after.as_ref()),
        },
        "decision": {
            "reference_layer_trace_available": reference_layer_trace_available,
            "current_blocked_reasons": blocked_reasons,
            "next_when_available": "compare reference stage trace against Rust CPU and strict A770 trace receipts before changing Rust model math",
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    }))
}

fn compare_reference_layer_trace(args: &LayerTraceCompareArgs) -> Result<Value> {
    let reference_path = normalize_path(&args.reference)?;
    let cpu_trace_dir = normalize_path(&args.cpu_trace_dir)?;
    let a770_trace_dir = match &args.a770_trace_dir {
        Some(path) => Some(normalize_path(path)?),
        None => None,
    };

    let reference_json = read_json(&reference_path)?;
    let reference_records = read_reference_records(&reference_json)?;
    let cpu_records = read_rust_trace_dir(&cpu_trace_dir)?;
    let a770_records = match &a770_trace_dir {
        Some(dir) => Some(read_rust_trace_dir(dir)?),
        None => None,
    };

    let stage_mapping = reference_stage_mapping();
    let cpu_comparison =
        compare_reference_to_rust(&reference_records, &cpu_records, &stage_mapping);
    let a770_comparison = a770_records
        .as_ref()
        .map(|records| compare_reference_to_rust(&reference_records, records, &stage_mapping));

    let mut blocked_reasons = Vec::<String>::new();
    if cpu_comparison["first_material_mismatch"].is_null() {
        blocked_reasons
            .push("rust_cpu_reference_layer_trace_no_material_mismatch_found".to_string());
    } else {
        blocked_reasons.push("rust_cpu_reference_layer_trace_divergence_unresolved".to_string());
    }
    if a770_trace_dir.is_none() {
        blocked_reasons.push("strict_a770_trace_dir_not_supplied".to_string());
    }
    blocked_reasons.sort_unstable();
    blocked_reasons.dedup();

    Ok(json!({
        "schema_version": 1,
        "receipt_type": "bitnet_reference_layer_trace_compare",
        "diagnostic": "bitnet_reference_layer_trace_compare",
        "producer": "cargo xtask bitnet-reference-layer-trace-compare",
        "created_at": chrono::Utc::now().to_rfc3339(),
        "diagnostic_only": true,
        "promotion_allowed": false,
        "claim_allowed": false,
        "classification": "diagnostic_only",
        "inputs": {
            "reference": path_to_string(&reference_path),
            "cpu_trace_dir": path_to_string(&cpu_trace_dir),
            "a770_trace_dir": a770_trace_dir.as_ref().map(|path| path_to_string(path)),
        },
        "reference": {
            "record_count": reference_records.len(),
            "stages": reference_records
                .iter()
                .map(|record| json!({
                    "name": record.name,
                    "stage": record.stage,
                    "graph_index": record.graph_index,
                    "shape": record.shape,
                    "dtype": record.dtype,
                    "nelements": record.nelements,
                    "rms": record.rms,
                    "values_available": record.values_available,
                }))
                .collect::<Vec<_>>(),
        },
        "stage_mapping": stage_mapping
            .iter()
            .map(|(reference, rust)| json!({"reference": reference, "rust": rust}))
            .collect::<Vec<_>>(),
        "cpu": cpu_comparison,
        "a770": a770_comparison,
        "decision": {
            "reference_layer_trace_compared": true,
            "claim_allowed": false,
            "current_blocked_reasons": blocked_reasons,
            "next_action": "use the first material mismatch to choose the next Rust/reference divergence capture; do not change model math until the mismatching boundary is stable",
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    }))
}

fn capture_rust_layer_traces(args: &LayerTraceRustCaptureArgs) -> Result<Value> {
    let plan_path = normalize_path(&args.plan)?;
    let cpu_trace_dir = normalize_path(&args.cpu_trace_dir)?;
    let a770_trace_dir = normalize_path(&args.a770_trace_dir)?;
    let plan_result = read_json(&plan_path);
    let plan_read_success = plan_result.is_ok();
    let plan = plan_result.unwrap_or(Value::Null);

    let cpu_argv = if plan_read_success { rust_command_argv(&plan, "cpu_argv").ok() } else { None };
    let a770_argv =
        if plan_read_success { rust_command_argv(&plan, "a770_argv").ok() } else { None };
    let (cpu_argv, cpu_trace_feature_injected) =
        cpu_argv.map(|argv| ensure_trace_feature(&argv)).unzip();
    let (a770_argv, a770_trace_feature_injected) =
        a770_argv.map(|argv| ensure_trace_feature(&argv)).unzip();

    let mut blocked_reasons = Vec::<String>::new();
    if !plan_path.is_file() {
        blocked_reasons.push("reference_plan_missing".to_string());
    } else if !plan_read_success {
        blocked_reasons.push("reference_plan_json_invalid".to_string());
    }
    if plan_read_success && cpu_argv.is_none() {
        blocked_reasons.push("rust_cpu_trace_command_missing".to_string());
    }
    if plan_read_success && !args.skip_a770 && a770_argv.is_none() {
        blocked_reasons.push("strict_a770_trace_command_missing".to_string());
    }

    let cpu_prepare = prepare_trace_dir(&cpu_trace_dir, args.overwrite)?;
    if let Some(reason) = cpu_prepare.pointer("/blocked_reason").and_then(Value::as_str) {
        blocked_reasons.push(format!("cpu_trace_dir_{reason}"));
    }
    let a770_prepare = if args.skip_a770 {
        json!({
            "trace_dir": path_to_string(&a770_trace_dir),
            "skipped": true,
            "reason": "skip_a770_requested",
        })
    } else {
        let prepare = prepare_trace_dir(&a770_trace_dir, args.overwrite)?;
        if let Some(reason) = prepare.pointer("/blocked_reason").and_then(Value::as_str) {
            blocked_reasons.push(format!("strict_a770_trace_dir_{reason}"));
        }
        prepare
    };

    let can_run_cpu = blocked_reasons.is_empty();
    let cpu = if can_run_cpu {
        run_rust_trace_capture("cpu", cpu_argv.as_deref().unwrap_or(&[]), &cpu_trace_dir)?
    } else {
        skipped_rust_trace_capture("cpu", cpu_argv.as_deref(), &cpu_trace_dir)
    };

    let can_run_a770 = blocked_reasons.is_empty() && !args.skip_a770;
    let a770 = if can_run_a770 {
        run_rust_trace_capture("strict_a770", a770_argv.as_deref().unwrap_or(&[]), &a770_trace_dir)?
    } else {
        if args.skip_a770 {
            blocked_reasons.push("strict_a770_trace_capture_skipped".to_string());
        }
        skipped_rust_trace_capture("strict_a770", a770_argv.as_deref(), &a770_trace_dir)
    };

    append_trace_capture_blockers("cpu", &cpu, &mut blocked_reasons);
    if !args.skip_a770 {
        append_trace_capture_blockers("strict_a770", &a770, &mut blocked_reasons);
    }
    blocked_reasons.sort_unstable();
    blocked_reasons.dedup();

    let cpu_records = cpu.pointer("/trace/record_count").and_then(Value::as_u64).unwrap_or(0);
    let a770_records = a770.pointer("/trace/record_count").and_then(Value::as_u64).unwrap_or(0);
    let rust_layer_traces_ready = cpu_records > 0 && (args.skip_a770 || a770_records > 0);
    let compare_ready = cpu_records > 0 && !args.skip_a770 && a770_records > 0;

    Ok(json!({
        "schema_version": 1,
        "receipt_type": "bitnet_reference_layer_trace_rust_capture",
        "diagnostic": "bitnet_reference_layer_trace_rust_capture",
        "producer": "cargo xtask bitnet-reference-layer-trace-capture-rust",
        "created_at": chrono::Utc::now().to_rfc3339(),
        "diagnostic_only": true,
        "promotion_allowed": false,
        "claim_allowed": false,
        "classification": "diagnostic_only",
        "inputs": {
            "plan": path_to_string(&plan_path),
            "cpu_trace_dir": path_to_string(&cpu_trace_dir),
            "a770_trace_dir": path_to_string(&a770_trace_dir),
            "skip_a770": args.skip_a770,
            "overwrite": args.overwrite,
        },
        "model": plan.pointer("/model").cloned().unwrap_or(Value::Null),
        "prompt_identity": plan.pointer("/prompt_identity").cloned().unwrap_or(Value::Null),
        "proof_identity": plan.pointer("/rust_commands/proof_identity").cloned().unwrap_or(Value::Null),
        "preflight": {
            "plan_exists": plan_path.is_file(),
            "plan_json_valid": plan_read_success,
            "cpu_command_present": cpu_argv.is_some(),
            "a770_command_present": a770_argv.is_some(),
            "cpu_trace_feature_injected": cpu_trace_feature_injected.unwrap_or(false),
            "a770_trace_feature_injected": a770_trace_feature_injected.unwrap_or(false),
            "cpu_trace_dir_prepare": cpu_prepare,
            "a770_trace_dir_prepare": a770_prepare,
        },
        "cpu": cpu,
        "a770": a770,
        "decision": {
            "rust_layer_traces_ready": rust_layer_traces_ready,
            "compare_ready": compare_ready,
            "claim_allowed": false,
            "current_blocked_reasons": blocked_reasons,
            "next_when_ready": "run bitnet-reference-layer-trace-compare with the reference run receipt plus the captured CPU and strict A770 trace directories",
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    }))
}

fn read_reference_records(root: &Value) -> Result<Vec<ReferenceTraceRecord>> {
    let receipt = match root.pointer("/receipt_type").and_then(Value::as_str) {
        Some("bitnet_reference_layer_trace_run") => root
            .pointer("/sidecar/receipt")
            .context("layer trace run receipt missing /sidecar/receipt")?,
        _ => root,
    };
    let records = receipt
        .pointer("/records")
        .and_then(Value::as_array)
        .context("reference layer trace receipt missing /records")?;
    records
        .iter()
        .map(|record| {
            let shape = record
                .pointer("/shape")
                .and_then(Value::as_array)
                .context("reference record missing shape")?
                .iter()
                .map(|dim| dim.as_i64().context("reference shape dim is not integer"))
                .collect::<Result<Vec<_>>>()?;
            Ok(ReferenceTraceRecord {
                name: record
                    .pointer("/name")
                    .and_then(Value::as_str)
                    .context("reference record missing name")?
                    .to_string(),
                stage: record
                    .pointer("/stage")
                    .and_then(Value::as_str)
                    .context("reference record missing stage")?
                    .to_string(),
                graph_index: record.pointer("/graph_index").and_then(Value::as_i64),
                layer: record.pointer("/layer").and_then(Value::as_i64),
                dtype: record
                    .pointer("/dtype")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                shape,
                nelements: record.pointer("/nelements").and_then(Value::as_u64).unwrap_or(0),
                rms: record.pointer("/stats/rms").and_then(Value::as_f64),
                values_available: record
                    .pointer("/values_available")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            })
        })
        .collect()
}

fn read_rust_trace_dir(dir: &Path) -> Result<BTreeMap<String, RustTraceRecord>> {
    if !dir.exists() {
        bail!("rust trace directory missing: {}", dir.display());
    }
    if !dir.is_dir() {
        bail!("rust trace path is not a directory: {}", dir.display());
    }
    let mut records = BTreeMap::new();
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry.with_context(|| format!("reading entry in {}", dir.display()))?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("trace") {
            continue;
        }
        let text =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let record: RustTraceRecord =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let stage = record
            .stage
            .clone()
            .unwrap_or_else(|| record.name.rsplit('/').next().unwrap_or(&record.name).to_string());
        let should_insert = records.get(&stage).is_none_or(|existing| {
            rust_trace_preference(&record) < rust_trace_preference(existing)
        });
        if should_insert {
            records.insert(stage, record);
        }
    }
    if records.is_empty() {
        bail!("rust trace directory has no .trace files: {}", dir.display());
    }
    Ok(records)
}

fn rust_trace_preference(record: &RustTraceRecord) -> i64 {
    match record.layer {
        Some(0) => 0,
        Some(-1) => 1,
        Some(layer) if layer > 0 => 10 + layer as i64,
        Some(layer) => 100 + layer.abs() as i64,
        None => 1_000,
    }
}

fn reference_stage_mapping() -> Vec<(&'static str, &'static str)> {
    vec![
        ("inp_embd", "embeddings"),
        ("attn_norm", "attn_norm"),
        ("Qcur", "attention_q"),
        ("Kcur", "attention_k"),
        ("Vcur", "attention_v"),
        ("attn_sub_norm", "post_attention_subnorm"),
        ("attn_o_out", "post_o_proj"),
        ("ffn_inp", "post_attention_residual"),
        ("ffn_norm", "post_ffn_norm"),
        ("ffn_out", "post_swiglu"),
        ("ffn_sub_norm", "post_ffn_subnorm"),
        ("ffn_down", "post_down_proj"),
        ("l_out", "post_layer"),
        ("result_norm", "final_norm"),
        ("result_output", "logits"),
    ]
}

fn compare_reference_to_rust(
    reference_records: &[ReferenceTraceRecord],
    rust_records: &BTreeMap<String, RustTraceRecord>,
    mapping: &[(&str, &str)],
) -> Value {
    let mut stages = Vec::new();
    let mut first_material_mismatch = Value::Null;
    let mut compared_count = 0usize;
    let mut material_mismatch_count = 0usize;
    let mut missing_reference_count = 0usize;
    let mut missing_rust_count = 0usize;

    for (reference_stage, rust_stage) in mapping {
        let candidates = reference_records
            .iter()
            .filter(|record| record.stage == *reference_stage)
            .collect::<Vec<_>>();
        let rust = rust_records.get(*rust_stage);
        let reference = rust
            .and_then(|rust| {
                candidates
                    .iter()
                    .copied()
                    .find(|record| record.nelements == rust.num_elements as u64)
            })
            .or_else(|| candidates.first().copied());

        if reference.is_none() {
            missing_reference_count += 1;
        }
        if rust.is_none() {
            missing_rust_count += 1;
        }

        let mut status = "missing";
        let mut rms_abs_delta = None::<f64>;
        let mut material_mismatch = reference.is_none() || rust.is_none();
        if let (Some(reference), Some(rust)) = (reference, rust) {
            compared_count += 1;
            let element_count_match = reference.nelements == rust.num_elements as u64;
            let dtype_match = reference.dtype.eq_ignore_ascii_case(&rust.dtype);
            rms_abs_delta = reference.rms.map(|rms| (rms - rust.rms).abs());
            let rms_material = rms_abs_delta.is_some_and(|delta| delta > 1.0e-4);
            material_mismatch = !element_count_match || !dtype_match || rms_material;
            status = if material_mismatch { "material_mismatch" } else { "summary_match" };
        }
        if material_mismatch {
            material_mismatch_count += 1;
        }

        let stage = json!({
            "reference_stage": reference_stage,
            "rust_stage": rust_stage,
            "status": status,
            "candidate_reference_records": candidates.len(),
            "reference": reference.map(reference_record_summary),
            "rust": rust.map(rust_record_summary),
            "rms_abs_delta": rms_abs_delta,
            "material_mismatch": material_mismatch,
        });
        if material_mismatch && first_material_mismatch.is_null() {
            first_material_mismatch = stage.clone();
        }
        stages.push(stage);
    }

    json!({
        "trace_record_count": rust_records.len(),
        "compared_stage_count": compared_count,
        "material_mismatch_count": material_mismatch_count,
        "missing_reference_count": missing_reference_count,
        "missing_rust_count": missing_rust_count,
        "first_material_mismatch": first_material_mismatch,
        "stages": stages,
    })
}

fn reference_record_summary(record: &ReferenceTraceRecord) -> Value {
    json!({
        "name": record.name,
        "stage": record.stage,
        "graph_index": record.graph_index,
        "layer": record.layer,
        "shape": record.shape,
        "dtype": record.dtype,
        "nelements": record.nelements,
        "rms": record.rms,
        "values_available": record.values_available,
    })
}

fn rust_record_summary(record: &RustTraceRecord) -> Value {
    json!({
        "name": record.name,
        "stage": record.stage,
        "layer": record.layer,
        "shape": record.shape,
        "dtype": record.dtype,
        "num_elements": record.num_elements,
        "rms": record.rms,
    })
}

fn build_plan(args: &LayerTracePlanArgs) -> Result<Value> {
    let cpp_root = normalize_path(&args.cpp_root)?;
    let rust_transformer_path = normalize_path(&args.rust_transformer)?;
    let llama_cpp = read_source(cpp_root.join("src/llama.cpp"));
    let rust_transformer = read_source(rust_transformer_path);
    let reference_anchors = anchor_status(&llama_cpp.text, REFERENCE_REQUIRED_ANCHORS);
    let rust_anchors = anchor_status(&rust_transformer.text, RUST_REQUIRED_ANCHORS);

    let mut blocked_reasons = Vec::<String>::new();
    if !cpp_root.is_dir() {
        blocked_reasons.push("reference_llama_cpp_root_missing".to_string());
    }
    for source in [&llama_cpp, &rust_transformer] {
        if !source.exists {
            blocked_reasons.push(format!("source_missing:{}", path_to_string(&source.path)));
        } else if !source.read_ok {
            blocked_reasons.push(format!("source_read_failed:{}", path_to_string(&source.path)));
        }
    }
    for anchor in missing_anchor_names(&reference_anchors) {
        blocked_reasons.push(format!("reference_anchor_missing:{anchor}"));
    }
    for anchor in missing_anchor_names(&rust_anchors) {
        blocked_reasons.push(format!("rust_trace_anchor_missing:{anchor}"));
    }
    blocked_reasons.push("reference_layer_trace_patch_not_applied".to_string());
    blocked_reasons.sort_unstable();
    blocked_reasons.dedup();

    let source_anchors_ready =
        blocked_reasons.iter().all(|reason| reason == "reference_layer_trace_patch_not_applied");

    Ok(json!({
        "schema_version": 1,
        "diagnostic": "bitnet_reference_layer_trace_plan",
        "diagnostic_only": true,
        "promotion_allowed": false,
        "claim_allowed": false,
        "classification": "diagnostic_only",
        "producer": "cargo xtask bitnet-reference-layer-trace-plan",
        "created_at": chrono::Utc::now().to_rfc3339(),
        "inputs": {
            "cpp_root": path_to_string(&cpp_root),
            "llama_cpp": source_receipt(&llama_cpp),
            "rust_transformer": source_receipt(&rust_transformer),
        },
        "source_capability": {
            "reference_graph_callback_labels": reference_anchors,
            "rust_trace_labels": rust_anchors,
            "source_anchors_ready_for_target_local_patch": source_anchors_ready,
            "reason": "reference llama.cpp names the BitNet b1.58 graph stages through llm_build_cb while Rust already exposes comparable layer-0 trace labels behind the trace feature",
        },
        "stage_mapping": [
            {"reference": "inp_embd", "rust": "embeddings", "scope": "prompt embedding"},
            {"reference": "attn_norm", "rust": "attn_norm", "scope": "layer0"},
            {"reference": "Qcur", "rust": "attention_q", "scope": "layer0"},
            {"reference": "Kcur", "rust": "attention_k", "scope": "layer0"},
            {"reference": "Vcur", "rust": "attention_v", "scope": "layer0"},
            {"reference": "attn_sub_norm", "rust": "post_attention_subnorm", "scope": "layer0"},
            {"reference": "attn_o_out", "rust": "post_o_proj", "scope": "layer0"},
            {"reference": "ffn_inp", "rust": "post_attention_residual", "scope": "layer0"},
            {"reference": "ffn_norm", "rust": "post_ffn_norm", "scope": "layer0"},
            {"reference": "ffn_out", "rust": "post_swiglu", "scope": "layer0"},
            {"reference": "ffn_sub_norm", "rust": "post_ffn_subnorm", "scope": "layer0"},
            {"reference": "ffn_down", "rust": "post_down_proj", "scope": "layer0"},
            {"reference": "l_out", "rust": "post_layer", "scope": "layer0"},
            {"reference": "result_norm", "rust": "final_norm", "scope": "final token"},
            {"reference": "result_output", "rust": "logits", "scope": "final token"}
        ],
        "instrumentation_plan": {
            "target_local_only": true,
            "target_files": [
                "target/external/BitNet-reference/3rdparty/llama.cpp/src/llama.cpp"
            ],
            "environment_variable": "BITNET_RS_REFERENCE_LAYER_TRACE",
            "receipt_type_when_applied": "bitnet_reference_layer_trace",
            "captures": [
                "prompt identity inherited from matched reference plan",
                "BitNet b1.58 stage name",
                "layer index",
                "shape",
                "dtype",
                "rms",
                "vector hash when a CPU-readable tensor buffer is available",
                "first-values sample when safe to extract"
            ],
            "next_action": "add a target-local reference graph-callback instrumentation patch, run the matched prompt, and compare stage mapping against Rust trace output before changing Rust model math",
            "not_claim": "layer trace localizes first numeric divergence only; it does not prove semantic quality, A770 support, or residency",
        },
        "decision": {
            "reference_layer_trace_available": false,
            "source_anchors_ready_for_target_local_patch": source_anchors_ready,
            "current_blocked_reasons": blocked_reasons,
            "next_action": "add target-local reference layer trace instrumentation, then compare reference trace stages against Rust CPU and strict A770 traces",
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    }))
}

fn anchor_status(text: &str, anchors: &[(&str, &str)]) -> Vec<Value> {
    anchors
        .iter()
        .map(|(name, needle)| {
            json!({
                "name": name,
                "needle": needle,
                "present": text.contains(needle),
            })
        })
        .collect()
}

fn missing_anchor_names(anchors: &[Value]) -> Vec<String> {
    anchors
        .iter()
        .filter(|anchor| anchor.pointer("/present").and_then(Value::as_bool) != Some(true))
        .filter_map(|anchor| anchor.pointer("/name").and_then(Value::as_str).map(str::to_string))
        .collect()
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

fn rust_command_argv(plan: &Value, key: &str) -> Result<Vec<String>> {
    let pointer = format!("/rust_commands/{key}");
    plan.pointer(&pointer)
        .and_then(Value::as_array)
        .with_context(|| format!("plan missing {pointer}"))?
        .iter()
        .map(|item| item.as_str().map(ToOwned::to_owned).context("rust argv item is not a string"))
        .collect()
}

fn ensure_trace_feature(argv: &[String]) -> (Vec<String>, bool) {
    let mut argv = argv.to_vec();
    let Some(features_index) = argv.iter().position(|arg| arg == "--features") else {
        argv.push("--features".to_string());
        argv.push("trace".to_string());
        return (argv, true);
    };
    let Some(features) = argv.get_mut(features_index + 1) else {
        argv.push("trace".to_string());
        return (argv, true);
    };
    let has_trace = features
        .split([',', ' '])
        .filter(|feature| !feature.trim().is_empty())
        .any(|feature| feature.trim() == "trace");
    if has_trace {
        return (argv, false);
    }
    if features.trim().is_empty() {
        *features = "trace".to_string();
    } else {
        features.push_str(",trace");
    }
    (argv, true)
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
        let script = reference_root.join("build_bitnet_rs_layer_trace_reference.cmd");
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
    command.args(&argv[1..]).env("BITNET_RS_REFERENCE_LAYER_TRACE", sidecar).stdin(Stdio::null());
    run_command(&mut command)
}

fn run_rust_trace_capture(label: &str, argv: &[String], trace_dir: &Path) -> Result<Value> {
    let executable = argv.first().context("empty Rust trace command")?;
    let mut command = Command::new(executable);
    command
        .args(&argv[1..])
        .env("BITNET_TRACE_DIR", trace_dir)
        .env("BITNET_DETERMINISTIC", "1")
        .env("BITNET_SEED", "0")
        .stdin(Stdio::null());
    let capture = run_command(&mut command)?;
    let trace = summarize_rust_trace_dir(trace_dir);
    Ok(json!({
        "label": label,
        "attempted": true,
        "trace_dir": path_to_string(trace_dir),
        "argv": argv,
        "command": capture_json(Some(&capture)),
        "trace": trace,
    }))
}

fn skipped_rust_trace_capture(label: &str, argv: Option<&[String]>, trace_dir: &Path) -> Value {
    json!({
        "label": label,
        "attempted": false,
        "trace_dir": path_to_string(trace_dir),
        "argv": argv.unwrap_or(&[]),
        "command": Value::Null,
        "trace": summarize_rust_trace_dir(trace_dir),
    })
}

fn append_trace_capture_blockers(prefix: &str, capture: &Value, blocked_reasons: &mut Vec<String>) {
    if capture.pointer("/attempted").and_then(Value::as_bool) == Some(true)
        && capture.pointer("/command/success").and_then(Value::as_bool) != Some(true)
    {
        blocked_reasons.push(format!("{prefix}_trace_command_failed"));
    }
    if capture.pointer("/trace/record_count").and_then(Value::as_u64).unwrap_or(0) == 0 {
        blocked_reasons.push(format!("{prefix}_trace_dir_no_trace_files"));
    }
}

fn prepare_trace_dir(dir: &Path, overwrite: bool) -> Result<Value> {
    if dir.exists() && !dir.is_dir() {
        return Ok(json!({
            "trace_dir": path_to_string(dir),
            "ready": false,
            "blocked_reason": "path_is_not_directory",
        }));
    }
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let entries = fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("reading entries in {}", dir.display()))?;
    let mut removed_trace_files = Vec::<String>::new();
    let mut non_trace_entries = Vec::<String>::new();
    for entry in entries {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("trace") {
            if overwrite {
                fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
                removed_trace_files.push(path_to_string(&path));
            }
        } else {
            non_trace_entries.push(path_to_string(&path));
        }
    }
    let trace_files_after = count_trace_files(dir)?;
    let blocked_reason = if !non_trace_entries.is_empty() {
        Some("contains_non_trace_entries")
    } else if trace_files_after > 0 && !overwrite {
        Some("contains_existing_trace_files")
    } else {
        None
    };
    Ok(json!({
        "trace_dir": path_to_string(dir),
        "ready": blocked_reason.is_none(),
        "overwrite": overwrite,
        "removed_trace_files": removed_trace_files,
        "existing_trace_files_after_prepare": trace_files_after,
        "non_trace_entries": non_trace_entries,
        "blocked_reason": blocked_reason,
    }))
}

fn summarize_rust_trace_dir(dir: &Path) -> Value {
    if !dir.exists() {
        return json!({
            "exists": false,
            "record_count": 0,
            "stages": [],
            "read_error": "trace_dir_missing",
        });
    }
    match read_rust_trace_dir(dir) {
        Ok(records) => json!({
            "exists": true,
            "record_count": records.len(),
            "stages": records
                .iter()
                .map(|(stage, record)| json!({
                    "stage": stage,
                    "name": record.name,
                    "layer": record.layer,
                    "shape": record.shape,
                    "dtype": record.dtype,
                    "num_elements": record.num_elements,
                    "rms": record.rms,
                }))
                .collect::<Vec<_>>(),
            "read_error": Value::Null,
        }),
        Err(error) => json!({
            "exists": dir.exists(),
            "record_count": 0,
            "stages": [],
            "read_error": error.to_string(),
        }),
    }
}

fn count_trace_files(dir: &Path) -> Result<usize> {
    if !dir.exists() {
        return Ok(0);
    }
    let mut count = 0usize;
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry.with_context(|| format!("reading entry in {}", dir.display()))?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("trace") {
            count += 1;
        }
    }
    Ok(count)
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
        &["restore", "--", "common/common.cpp", "common/log.cpp", "src/llama.cpp"],
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
        "json" => println!("{}", serde_json::to_string_pretty(report)?),
        "human" => {
            let receipt_type = report
                .pointer("/receipt_type")
                .and_then(Value::as_str)
                .unwrap_or("bitnet_reference_layer_trace_plan");
            if receipt_type == "bitnet_reference_layer_trace_run" {
                let available = report
                    .pointer("/decision/reference_layer_trace_available")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let record_count =
                    report.pointer("/sidecar/record_count").and_then(Value::as_u64).unwrap_or(0);
                let reasons = report
                    .pointer("/decision/current_blocked_reasons")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                println!(
                    "bitnet reference layer trace run: diagnostic_only=true claim_allowed=false available={available} records={record_count}"
                );
                if !reasons.is_empty() {
                    println!("blocked_reasons:");
                    for reason in reasons {
                        println!("  - {}", reason.as_str().unwrap_or("<non-string>"));
                    }
                }
                return Ok(());
            }
            if receipt_type == "bitnet_reference_layer_trace_compare" {
                let first = report
                    .pointer("/cpu/first_material_mismatch/status")
                    .and_then(Value::as_str)
                    .unwrap_or("none");
                let stage = report
                    .pointer("/cpu/first_material_mismatch/reference_stage")
                    .and_then(Value::as_str)
                    .unwrap_or("none");
                let mismatches = report
                    .pointer("/cpu/material_mismatch_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                println!(
                    "bitnet reference layer trace compare: diagnostic_only=true claim_allowed=false cpu_first_mismatch={stage}:{first} cpu_mismatches={mismatches}"
                );
                return Ok(());
            }
            if receipt_type == "bitnet_reference_layer_trace_rust_capture" {
                let ready = report
                    .pointer("/decision/compare_ready")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let cpu_records =
                    report.pointer("/cpu/trace/record_count").and_then(Value::as_u64).unwrap_or(0);
                let a770_records =
                    report.pointer("/a770/trace/record_count").and_then(Value::as_u64).unwrap_or(0);
                let reasons = report
                    .pointer("/decision/current_blocked_reasons")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                println!(
                    "bitnet reference layer trace rust capture: diagnostic_only=true claim_allowed=false compare_ready={ready} cpu_records={cpu_records} a770_records={a770_records}"
                );
                if !reasons.is_empty() {
                    println!("blocked_reasons:");
                    for reason in reasons {
                        println!("  - {}", reason.as_str().unwrap_or("<non-string>"));
                    }
                }
                return Ok(());
            }
            let ready = report
                .pointer("/decision/source_anchors_ready_for_target_local_patch")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let reasons = report
                .pointer("/decision/current_blocked_reasons")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            println!(
                "bitnet reference layer trace plan: diagnostic_only=true claim_allowed=false source_anchors_ready={ready}"
            );
            if !reasons.is_empty() {
                println!("blocked_reasons:");
                for reason in reasons {
                    println!("  - {}", reason.as_str().unwrap_or("<non-string>"));
                }
            }
        }
        other => bail!("unsupported bitnet-reference-layer-trace-plan output format: {other}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_file(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    fn joined_needles(anchors: &[(&str, &str)]) -> String {
        anchors.iter().map(|(_, needle)| *needle).collect::<Vec<_>>().join("\n")
    }

    fn write_rust_trace(dir: &Path, stage: &str, shape: &[usize], rms: f64) {
        let path = dir.join(format!("{stage}.trace"));
        let record = json!({
            "name": format!("t0/blk0/{stage}"),
            "shape": shape,
            "dtype": "F32",
            "blake3": "abc",
            "rms": rms,
            "num_elements": shape.iter().product::<usize>(),
            "seq": 0,
            "layer": 0,
            "stage": stage,
        });
        fs::write(path, serde_json::to_string_pretty(&record).unwrap()).unwrap();
    }

    fn write_rust_capture_plan(path: &Path) {
        write_file(
            path,
            &serde_json::to_string_pretty(&json!({
                "model": {
                    "model_path": "model.gguf",
                    "contract": "docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml"
                },
                "prompt_identity": {
                    "prompt_template": "llama3-chat",
                    "prompt_token_ids_sha256": "abc"
                },
                "rust_commands": {
                    "cpu_argv": ["definitely-not-run-when-trace-dir-is-stale"],
                    "a770_argv": ["definitely-not-run-when-trace-dir-is-stale"],
                    "proof_identity": {
                        "model_contract": "docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml",
                        "a770_kernel_route": "a770.bitnet.i2s.qk256"
                    }
                }
            }))
            .unwrap(),
        );
    }

    #[test]
    fn plan_reports_ready_when_reference_and_rust_anchors_exist() {
        let dir = tempdir().unwrap();
        let cpp_root = dir.path().join("cpp");
        let llama_cpp = cpp_root.join("src/llama.cpp");
        let rust_transformer = dir.path().join("lib.rs");
        write_file(&llama_cpp, &joined_needles(REFERENCE_REQUIRED_ANCHORS));
        write_file(&rust_transformer, &joined_needles(RUST_REQUIRED_ANCHORS));

        let report = build_plan(&LayerTracePlanArgs {
            cpp_root,
            rust_transformer,
            output: None,
            format: "json".to_string(),
        })
        .unwrap();

        assert_eq!(
            report.pointer("/decision/source_anchors_ready_for_target_local_patch"),
            Some(&json!(true))
        );
        assert_eq!(
            report.pointer("/decision/current_blocked_reasons"),
            Some(&json!(["reference_layer_trace_patch_not_applied"]))
        );
        assert_eq!(report.pointer("/claim_allowed"), Some(&json!(false)));
    }

    #[test]
    fn plan_reports_missing_reference_anchor() {
        let dir = tempdir().unwrap();
        let cpp_root = dir.path().join("cpp");
        let llama_cpp = cpp_root.join("src/llama.cpp");
        let rust_transformer = dir.path().join("lib.rs");
        write_file(&llama_cpp, "struct ggml_cgraph * build_bitnet_158()");
        write_file(&rust_transformer, &joined_needles(RUST_REQUIRED_ANCHORS));

        let report = build_plan(&LayerTracePlanArgs {
            cpp_root,
            rust_transformer,
            output: None,
            format: "json".to_string(),
        })
        .unwrap();
        let reasons =
            report.pointer("/decision/current_blocked_reasons").and_then(Value::as_array).unwrap();

        assert!(reasons.iter().any(|reason| reason == "reference_anchor_missing:result_output"));
        assert_eq!(
            report.pointer("/decision/source_anchors_ready_for_target_local_patch"),
            Some(&json!(false))
        );
    }

    #[test]
    fn run_report_stays_diagnostic_when_inputs_are_missing() {
        let dir = tempdir().unwrap();
        let report = run_instrumented_reference(&LayerTraceRunArgs {
            reference_root: dir.path().join("missing-reference"),
            cpp_root: dir.path().join("missing-cpp"),
            patch: dir.path().join("missing.patch"),
            plan: dir.path().join("missing-plan.json"),
            sidecar: dir.path().join("sidecar.json"),
            output: None,
            format: "json".to_string(),
        })
        .unwrap();
        let reasons =
            report.pointer("/decision/current_blocked_reasons").and_then(Value::as_array).unwrap();

        assert_eq!(report.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(report.pointer("/diagnostic_only"), Some(&json!(true)));
        assert!(reasons.contains(&json!("reference_root_missing")));
        assert!(reasons.contains(&json!("reference_layer_trace_patch_missing")));
        assert!(reasons.contains(&json!("reference_plan_missing")));
    }

    #[test]
    fn rust_capture_report_stays_diagnostic_when_plan_is_missing() {
        let dir = tempdir().unwrap();
        let report = capture_rust_layer_traces(&LayerTraceRustCaptureArgs {
            plan: dir.path().join("missing-plan.json"),
            cpu_trace_dir: dir.path().join("cpu"),
            a770_trace_dir: dir.path().join("a770"),
            skip_a770: false,
            overwrite: false,
            output: None,
            format: "json".to_string(),
        })
        .unwrap();
        let reasons =
            report.pointer("/decision/current_blocked_reasons").and_then(Value::as_array).unwrap();

        assert_eq!(report.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(report.pointer("/diagnostic_only"), Some(&json!(true)));
        assert!(reasons.contains(&json!("reference_plan_missing")));
        assert_eq!(report.pointer("/decision/compare_ready"), Some(&json!(false)));
    }

    #[test]
    fn rust_capture_rejects_stale_trace_dirs_without_overwrite() {
        let dir = tempdir().unwrap();
        let plan = dir.path().join("plan.json");
        let cpu = dir.path().join("cpu");
        let a770 = dir.path().join("a770");
        fs::create_dir_all(&cpu).unwrap();
        fs::create_dir_all(&a770).unwrap();
        write_rust_capture_plan(&plan);
        write_rust_trace(&cpu, "embeddings", &[1, 2], 1.0);

        let report = capture_rust_layer_traces(&LayerTraceRustCaptureArgs {
            plan,
            cpu_trace_dir: cpu,
            a770_trace_dir: a770,
            skip_a770: false,
            overwrite: false,
            output: None,
            format: "json".to_string(),
        })
        .unwrap();
        let reasons =
            report.pointer("/decision/current_blocked_reasons").and_then(Value::as_array).unwrap();

        assert!(reasons.contains(&json!("cpu_trace_dir_contains_existing_trace_files")));
        assert_eq!(report.pointer("/cpu/attempted"), Some(&json!(false)));
        assert_eq!(report.pointer("/decision/compare_ready"), Some(&json!(false)));
    }

    #[test]
    fn rust_capture_injects_trace_feature_into_plan_argv() {
        let argv = vec![
            "cargo".to_string(),
            "run".to_string(),
            "--features".to_string(),
            "opencl".to_string(),
            "--".to_string(),
            "run".to_string(),
        ];

        let (updated, injected) = ensure_trace_feature(&argv);

        assert!(injected);
        assert_eq!(updated[3], "opencl,trace");
    }

    #[test]
    fn rust_capture_does_not_duplicate_trace_feature() {
        let argv = vec![
            "cargo".to_string(),
            "run".to_string(),
            "--features".to_string(),
            "cpu,trace".to_string(),
        ];

        let (updated, injected) = ensure_trace_feature(&argv);

        assert!(!injected);
        assert_eq!(updated, argv);
    }

    #[test]
    fn compare_reports_first_material_mismatch() {
        let dir = tempdir().unwrap();
        let reference = dir.path().join("reference.json");
        let cpu = dir.path().join("cpu");
        fs::create_dir_all(&cpu).unwrap();
        fs::write(
            &reference,
            serde_json::to_string_pretty(&json!({
                "receipt_type": "bitnet_reference_layer_trace",
                "records": [
                    {
                        "name": "inp_embd",
                        "stage": "inp_embd",
                        "graph_index": 0,
                        "layer": -1,
                        "dtype": "f32",
                        "shape": [2, 2, 1, 1],
                        "nelements": 4,
                        "values_available": true,
                        "stats": {"rms": 1.0}
                    },
                    {
                        "name": "attn_norm-0",
                        "stage": "attn_norm",
                        "graph_index": 2,
                        "layer": 0,
                        "dtype": "f32",
                        "shape": [2, 2, 1, 1],
                        "nelements": 4,
                        "values_available": true,
                        "stats": {"rms": 2.0}
                    }
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        write_rust_trace(&cpu, "embeddings", &[2, 2], 1.0);
        write_rust_trace(&cpu, "attn_norm", &[2, 2], 1.0);

        let report = compare_reference_layer_trace(&LayerTraceCompareArgs {
            reference,
            cpu_trace_dir: cpu,
            a770_trace_dir: None,
            output: None,
            format: "json".to_string(),
        })
        .unwrap();

        assert_eq!(report.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(
            report.pointer("/cpu/first_material_mismatch/reference_stage"),
            Some(&json!("attn_norm"))
        );
        assert_eq!(
            report.pointer("/cpu/first_material_mismatch/status"),
            Some(&json!("material_mismatch"))
        );
        assert!(report.pointer("/decision/current_blocked_reasons").unwrap().is_array());
    }
}
