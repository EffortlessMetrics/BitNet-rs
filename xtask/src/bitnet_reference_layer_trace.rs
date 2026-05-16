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
const DEFAULT_EMBEDDING_ROW_AUTHORITY_OUTPUT: &str =
    "target/a770-diagnostic/bitnet-reference-embedding-row-authority.json";
const DEFAULT_ATTN_OUTPUT_SAME_INPUT_OUTPUT: &str =
    "target/a770-diagnostic/bitnet-reference-attn-output-same-input-parity.json";
const DEFAULT_CPU_TRACE_DIR: &str = "target/a770-diagnostic/reference-layer-trace-rust-cpu";
const DEFAULT_A770_TRACE_DIR: &str = "target/a770-diagnostic/reference-layer-trace-rust-a770";
const DEFAULT_BITNET_MODEL: &str = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf";
const DEFAULT_ATTN_OUTPUT_WEIGHT: &str = "blk.0.attn_output.weight";

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
    ("attention_scores_raw", "cb(kq, \"kq\", il)"),
    ("attention_scores_softmax", "cb(kq, \"kq_soft_max_ext\", il)"),
    ("attention_key_cache", "cb(k, \"k\", il)"),
    ("attention_value_cache", "cb(v, \"v\", il)"),
    ("attention_value_mix_premerge", "cb(kqv, \"kqv\", il)"),
    ("attention_value_mix_merged", "cb(kqv_merged, \"kqv_merged\", il)"),
    ("attention_value_mix_merged_cont", "cb(cur, \"kqv_merged_cont\", il)"),
    ("attention_value_mix_insertion_point", "cur = llm_build_kv(ctx0, lctx, kv_self, gf"),
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
    ("input_embedding", "trace_tensor_token_axis_record(\"embeddings\""),
    ("attention_norm", "attn_norm"),
    ("query_projection", "attention_q"),
    ("query_after_rope", "attention_q_rope"),
    ("key_projection", "attention_k"),
    ("value_projection", "attention_v"),
    ("attention_scores_raw_head_lanes", "attention_scores_raw_head"),
    ("attention_scores_softmax_head_lanes", "attn_scores_softmax_head"),
    ("attention_key_cache_head0_ref_layout", "attention_k_cache_head0_ref_layout_padded"),
    ("attention_key_cache_kv_head_live", "attention_k_cache_kv_head"),
    ("attention_key_cache_f16_roundtrip", "attention_k_cache_f16_roundtrip_kv_head"),
    ("attention_value_cache_head0_ref_layout", "attention_v_cache_head0_ref_layout_padded"),
    ("attention_value_cache_kv_head_live", "attention_v_cache_kv_head"),
    ("attention_value_cache_f16_roundtrip", "attention_v_cache_f16_roundtrip_kv_head"),
    ("attention_value_mix_f16_cache_head_lanes", "attention_value_mix_f16_cache_head"),
    ("attention_value_mix_f16_cache_merged", "attention_value_mix_f16_cache_merged"),
    ("attention_value_mix_head_lanes", "attention_value_mix_head"),
    ("attention_value_mix_merged", "attention_value_mix_merged"),
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
struct EmbeddingRowAuthorityArgs {
    reference: PathBuf,
    model: Option<PathBuf>,
    output: Option<PathBuf>,
    format: String,
}

#[derive(Debug)]
struct AttnOutputSameInputArgs {
    reference: PathBuf,
    model: Option<PathBuf>,
    weight: String,
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
    graph_op: Option<String>,
    graph_sources: Value,
    view_source: Value,
    view_offset: Option<u64>,
    full_shape: Vec<i64>,
    sample_offset: Option<u64>,
    token_axis: Option<i64>,
    dtype: String,
    shape: Vec<i64>,
    nelements: u64,
    rms: Option<f64>,
    first_values: Vec<f32>,
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
    #[serde(default)]
    first_values: Vec<f32>,
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
        Some("bitnet-reference-embedding-row-authority") => {
            if args[2..].iter().any(|arg| arg == "-h" || arg == "--help") {
                print_embedding_row_authority_help();
                return Ok(true);
            }
            let opts = parse_embedding_row_authority_args(args)?;
            let report = build_embedding_row_authority(&opts)?;
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
        Some("bitnet-reference-attn-output-same-input-parity") => {
            if args[2..].iter().any(|arg| arg == "-h" || arg == "--help") {
                print_attn_output_same_input_help();
                return Ok(true);
            }
            let opts = parse_attn_output_same_input_args(args)?;
            let report = build_attn_output_same_input_parity(&opts)?;
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

fn print_embedding_row_authority_help() {
    println!(
        "Compare reference token_embd.weight rows against Rust-loaded embedding rows for the captured prompt tokens\n\nUsage: xtask.exe bitnet-reference-embedding-row-authority [OPTIONS]\n\nOptions:\n      --reference <PATH>  Reference layer-trace run or sidecar JSON [default: target/a770-diagnostic/bitnet-reference-layer-trace-run.json]\n      --model <PATH>      GGUF model path [default: model path from reference receipt or models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf]\n      --output <PATH>     Output JSON receipt [default: target/a770-diagnostic/bitnet-reference-embedding-row-authority.json]\n      --format <FORMAT>   Output format: human or json [default: human]\n  -h, --help              Print help"
    );
}

fn print_attn_output_same_input_help() {
    println!(
        "Project the reference attn_sub_norm vector through Rust-loaded blk.0.attn_output.weight and compare with reference attn_o_out\n\nUsage: xtask.exe bitnet-reference-attn-output-same-input-parity [OPTIONS]\n\nOptions:\n      --reference <PATH>  Reference layer-trace run or sidecar JSON [default: target/a770-diagnostic/bitnet-reference-layer-trace-run.json]\n      --model <PATH>      GGUF model path [default: model path from reference receipt or models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf]\n      --weight <NAME>     GGUF QK256 attention output weight [default: blk.0.attn_output.weight]\n      --output <PATH>     Output JSON receipt [default: target/a770-diagnostic/bitnet-reference-attn-output-same-input-parity.json]\n      --format <FORMAT>   Output format: human or json [default: human]\n  -h, --help              Print help"
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

fn parse_embedding_row_authority_args(args: &[String]) -> Result<EmbeddingRowAuthorityArgs> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-embedding-row-authority") {
        bail!("parse_embedding_row_authority_args called for unexpected command");
    }
    let mut reference = PathBuf::from(DEFAULT_RUN_OUTPUT);
    let mut model = None::<PathBuf>;
    let mut output = Some(PathBuf::from(DEFAULT_EMBEDDING_ROW_AUTHORITY_OUTPUT));
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
            "--model" => model = Some(PathBuf::from(value()?)),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--format" => format = value()?,
            other => bail!("unknown bitnet-reference-embedding-row-authority option {other}"),
        }
    }
    Ok(EmbeddingRowAuthorityArgs { reference, model, output, format })
}

fn parse_attn_output_same_input_args(args: &[String]) -> Result<AttnOutputSameInputArgs> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-attn-output-same-input-parity") {
        bail!("parse_attn_output_same_input_args called for unexpected command");
    }
    let mut reference = PathBuf::from(DEFAULT_RUN_OUTPUT);
    let mut model = None::<PathBuf>;
    let mut weight = DEFAULT_ATTN_OUTPUT_WEIGHT.to_string();
    let mut output = Some(PathBuf::from(DEFAULT_ATTN_OUTPUT_SAME_INPUT_OUTPUT));
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
            "--model" => model = Some(PathBuf::from(value()?)),
            "--weight" => weight = value()?,
            "--output" => output = Some(PathBuf::from(value()?)),
            "--format" => format = value()?,
            other => bail!("unknown bitnet-reference-attn-output-same-input-parity option {other}"),
        }
    }
    Ok(AttnOutputSameInputArgs { reference, model, weight, output, format })
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
            "first_values_limit_env": std::env::var("BITNET_RS_REFERENCE_LAYER_TRACE_FIRST_VALUES_LIMIT").ok(),
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
    let cpu_scope_mismatch_count =
        cpu_comparison["scope_mismatch_count"].as_u64().unwrap_or_default();
    if cpu_scope_mismatch_count > 0 {
        blocked_reasons.push("rust_cpu_reference_layer_trace_scope_unaligned".to_string());
    } else if cpu_comparison["first_material_mismatch"].is_null() {
        blocked_reasons
            .push("rust_cpu_reference_layer_trace_no_material_mismatch_found".to_string());
    } else {
        blocked_reasons.push("rust_cpu_reference_layer_trace_divergence_unresolved".to_string());
    }
    if a770_comparison
        .as_ref()
        .and_then(|comparison| comparison["scope_mismatch_count"].as_u64())
        .unwrap_or_default()
        > 0
    {
        blocked_reasons.push("strict_a770_reference_layer_trace_scope_unaligned".to_string());
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
                    "graph_op": record.graph_op,
                    "graph_sources": record.graph_sources,
                    "view_source": record.view_source,
                    "view_offset": record.view_offset,
                    "full_shape": record.full_shape,
                    "sample_offset": record.sample_offset,
                    "token_axis": record.token_axis,
                    "sampled_token_index": reference_sampled_token_index(record),
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

    let (cpu_argv, cpu_command_key) = if plan_read_success {
        preferred_rust_trace_argv(&plan, "cpu_first_token_logit_argv", "cpu_argv")
    } else {
        (None, None)
    };
    let (a770_argv, a770_command_key) = if plan_read_success {
        preferred_rust_trace_argv(&plan, "a770_first_token_logit_argv", "a770_argv")
    } else {
        (None, None)
    };
    let (cpu_argv, cpu_trace_feature_injected) =
        cpu_argv.map(|argv| ensure_trace_feature(&argv)).unzip();
    let (a770_argv, a770_trace_feature_injected) =
        a770_argv.map(|argv| ensure_trace_feature(&argv)).unzip();
    let trace_target_seq =
        if plan_read_success { rust_trace_target_seq_from_plan(&plan) } else { None };

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
        run_rust_trace_capture(
            "cpu",
            cpu_argv.as_deref().unwrap_or(&[]),
            &cpu_trace_dir,
            trace_target_seq,
        )?
    } else {
        skipped_rust_trace_capture("cpu", cpu_argv.as_deref(), &cpu_trace_dir)
    };

    let can_run_a770 = blocked_reasons.is_empty() && !args.skip_a770;
    let a770 = if can_run_a770 {
        run_rust_trace_capture(
            "strict_a770",
            a770_argv.as_deref().unwrap_or(&[]),
            &a770_trace_dir,
            trace_target_seq,
        )?
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
            "cpu_command_key": cpu_command_key,
            "a770_command_key": a770_command_key,
            "trace_target_seq": trace_target_seq,
            "trace_target_source": trace_target_seq.map(|_| "prompt_identity.prompt_token_count_minus_one"),
            "cpu_trace_feature_injected": cpu_trace_feature_injected.unwrap_or(false),
            "a770_trace_feature_injected": a770_trace_feature_injected.unwrap_or(false),
            "cpu_trace_dir_prepare": cpu_prepare,
            "a770_trace_dir_prepare": a770_prepare,
            "first_values_limit_env": std::env::var("BITNET_TRACE_FIRST_VALUES_LIMIT").ok(),
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

fn build_embedding_row_authority(args: &EmbeddingRowAuthorityArgs) -> Result<Value> {
    let reference_path = normalize_path(&args.reference)?;
    let reference_root = read_json(&reference_path)?;
    let trace = reference_trace_receipt(&reference_root)?;
    let model_path = args
        .model
        .clone()
        .or_else(|| {
            reference_root.pointer("/model/model_path").and_then(Value::as_str).map(PathBuf::from)
        })
        .unwrap_or_else(|| PathBuf::from(DEFAULT_BITNET_MODEL));
    let model_path = normalize_path(&model_path)?;

    let token_ids = reference_prompt_tokens(trace);
    let sampled_output_token_id = trace.pointer("/sampled_output_token_id").and_then(Value::as_u64);
    let sampled_output_token_index =
        trace.pointer("/sampled_output_token_index").and_then(Value::as_i64);
    let inp_embd = trace.pointer("/records").and_then(Value::as_array).and_then(|records| {
        records
            .iter()
            .find(|record| record.pointer("/stage").and_then(Value::as_str) == Some("inp_embd"))
    });
    let expected_width = inp_embd
        .and_then(|record| record.pointer("/nelements").and_then(Value::as_u64))
        .unwrap_or(0) as usize;
    let reference_first_values = inp_embd
        .and_then(|record| record.pointer("/first_values").and_then(Value::as_array))
        .map(|values| {
            values.iter().filter_map(Value::as_f64).map(|value| value as f32).collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let mut blocked_reasons = Vec::<String>::new();
    if !reference_path.is_file() {
        blocked_reasons.push("reference_layer_trace_receipt_missing".to_string());
    }
    if !model_path.is_file() {
        blocked_reasons.push("model_gguf_missing".to_string());
    }
    if token_ids.is_empty() {
        blocked_reasons.push("reference_trace_prompt_tokens_missing".to_string());
    }
    if expected_width == 0 {
        blocked_reasons.push("reference_inp_embd_width_missing".to_string());
    }

    let mut model = json!({
        "path": path_to_string(&model_path),
        "exists": model_path.is_file(),
    });
    let mut tensor = Value::Null;
    let mut rows = Vec::<Value>::new();

    if model_path.is_file() && !token_ids.is_empty() && expected_width > 0 {
        match build_embedding_row_authority_rows(
            &model_path,
            &token_ids,
            sampled_output_token_index,
            sampled_output_token_id,
            expected_width,
            &reference_first_values,
        ) {
            Ok((model_report, tensor_report, row_reports)) => {
                model = model_report;
                tensor = tensor_report;
                rows = row_reports;
            }
            Err(err) => {
                blocked_reasons.push(format!("embedding_row_authority_unavailable:{err}"));
            }
        }
    }

    let sampled_row = rows
        .iter()
        .find(|row| {
            if let Some(index) = sampled_output_token_index {
                row.pointer("/prompt_index").and_then(Value::as_i64) == Some(index)
            } else {
                sampled_output_token_id.is_some_and(|token_id| {
                    row.pointer("/token_id").and_then(Value::as_u64) == Some(token_id)
                })
            }
        })
        .cloned();
    let reference_row_matches_trace_sample = sampled_row.as_ref().is_some_and(|row| {
        row_candidate_delta_le(row, "/reference_raw_vs_trace_first_values/max_abs_delta", 1.0e-3)
    });
    let rust_loaded_matches_reference_row = sampled_row.as_ref().is_some_and(|row| {
        row_candidate_delta_le(row, "/reference_raw_vs_rust_loaded/max_abs_delta", 1.0e-3)
    });
    let reference_trace_matching_layouts = sampled_row
        .as_ref()
        .map(|row| {
            row_candidate_matching_layouts(
                row,
                "/reference_raw_vs_trace_first_values/max_abs_delta",
                1.0e-3,
            )
        })
        .unwrap_or_default();
    let rust_loaded_matching_layouts = sampled_row
        .as_ref()
        .map(|row| {
            row_candidate_matching_layouts(
                row,
                "/reference_raw_vs_rust_loaded/max_abs_delta",
                1.0e-3,
            )
        })
        .unwrap_or_default();
    let shared_matching_layouts = reference_trace_matching_layouts
        .iter()
        .filter(|layout| rust_loaded_matching_layouts.contains(layout))
        .cloned()
        .collect::<Vec<_>>();
    let layout_authority_aligned = !shared_matching_layouts.is_empty();

    let input_authority_ready = blocked_reasons.is_empty() && sampled_row.is_some();
    let mut current_blocked_reasons = blocked_reasons.clone();
    if input_authority_ready
        && reference_row_matches_trace_sample
        && rust_loaded_matches_reference_row
        && !layout_authority_aligned
    {
        current_blocked_reasons
            .push("embedding_reference_and_rust_layout_authority_split".to_string());
    }

    let authority_ready = input_authority_ready && layout_authority_aligned;
    let next_action = if !input_authority_ready {
        "make the reference trace, prompt token ids, and GGUF model path available"
    } else if reference_row_matches_trace_sample
        && rust_loaded_matches_reference_row
        && !layout_authority_aligned
    {
        "inspect Rust embedding layout/transpose handling; reference trace and Rust-loaded embeddings match different raw layout candidates"
    } else if reference_row_matches_trace_sample && !rust_loaded_matches_reference_row {
        "inspect Rust GGUF embedding normalization or transformer handoff before changing downstream math"
    } else if rust_loaded_matches_reference_row && !reference_row_matches_trace_sample {
        "repair reference trace value capture for early graph nodes; raw GGUF and Rust-loaded embedding rows agree, but sampled trace values do not match token_embd.weight"
    } else if !reference_row_matches_trace_sample {
        "inspect reference trace sampling or token_embd raw row interpretation before changing Rust model math"
    } else {
        "embedding row authority matches; move the first-divergence search past prompt embedding"
    };

    Ok(json!({
        "schema_version": 1,
        "receipt_type": "bitnet_reference_embedding_row_authority",
        "diagnostic": "bitnet_reference_embedding_row_authority",
        "producer": "cargo xtask bitnet-reference-embedding-row-authority",
        "created_at": chrono::Utc::now().to_rfc3339(),
        "diagnostic_only": true,
        "claim_allowed": false,
        "promotion_allowed": false,
        "classification": "diagnostic_only",
        "inputs": {
            "reference": path_to_string(&reference_path),
            "model": path_to_string(&model_path),
        },
        "reference_trace": {
            "capture_scope": trace.pointer("/capture_scope").cloned().unwrap_or(Value::Null),
            "warmup_skip_policy": trace.pointer("/warmup_skip_policy").cloned().unwrap_or(Value::Null),
            "n_tokens": trace.pointer("/n_tokens").cloned().unwrap_or(Value::Null),
            "n_outputs": trace.pointer("/n_outputs").cloned().unwrap_or(Value::Null),
            "sampled_output_token_index": sampled_output_token_index,
            "sampled_output_token_id": sampled_output_token_id,
            "prompt_token_count": token_ids.len(),
            "prompt_token_ids": token_ids,
            "inp_embd": inp_embd.map(reference_inp_embd_summary).unwrap_or(Value::Null),
        },
        "model": model,
        "embedding_tensor": tensor,
        "rows": rows,
        "decision": {
            "embedding_row_authority_ready": authority_ready,
            "embedding_row_authority_inputs_ready": input_authority_ready,
            "reference_row_matches_trace_sample": reference_row_matches_trace_sample,
            "rust_loaded_matches_reference_row": rust_loaded_matches_reference_row,
            "reference_trace_matching_layouts": reference_trace_matching_layouts,
            "rust_loaded_matching_layouts": rust_loaded_matching_layouts,
            "shared_matching_layouts": shared_matching_layouts,
            "layout_authority_aligned": layout_authority_aligned,
            "current_blocked_reasons": current_blocked_reasons,
            "next_action": next_action,
            "claim_allowed": false,
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    }))
}

fn build_attn_output_same_input_parity(args: &AttnOutputSameInputArgs) -> Result<Value> {
    let reference_path = normalize_path(&args.reference)?;
    let reference_root = read_json(&reference_path)?;
    let reference_records = read_reference_records(&reference_root)?;
    let trace = reference_trace_receipt(&reference_root)?;
    let model_path = args
        .model
        .clone()
        .or_else(|| {
            reference_root.pointer("/model/model_path").and_then(Value::as_str).map(PathBuf::from)
        })
        .unwrap_or_else(|| PathBuf::from(DEFAULT_BITNET_MODEL));
    let model_path = normalize_path(&model_path)?;

    let input = reference_records
        .iter()
        .find(|record| record.stage == "attn_sub_norm" && record.layer == Some(0));
    let target = reference_records
        .iter()
        .find(|record| record.stage == "attn_o_out" && record.layer == Some(0));

    let mut blocked_reasons = Vec::<String>::new();
    if !reference_path.is_file() {
        blocked_reasons.push("reference_layer_trace_receipt_missing".to_string());
    }
    if !model_path.is_file() {
        blocked_reasons.push("model_gguf_missing".to_string());
    }
    if input.is_none() {
        blocked_reasons.push("reference_attn_sub_norm_layer0_missing".to_string());
    }
    if target.is_none() {
        blocked_reasons.push("reference_attn_o_out_layer0_missing".to_string());
    }

    let input_count = input.map(|record| record.first_values.len()).unwrap_or(0);
    let target_count = target.map(|record| record.first_values.len()).unwrap_or(0);
    if input_count == 0 {
        blocked_reasons.push("reference_attn_sub_norm_first_values_missing".to_string());
    }
    if target_count == 0 {
        blocked_reasons.push("reference_attn_o_out_first_values_missing".to_string());
    }

    let mut model = json!({
        "path": path_to_string(&model_path),
        "exists": model_path.is_file(),
    });
    let mut weight = json!({
        "name": args.weight,
    });
    let mut projection = Value::Null;

    if blocked_reasons.is_empty() {
        let input = input.expect("checked above");
        let target = target.expect("checked above");
        match project_attn_output_same_input(&model_path, &args.weight, &input.first_values) {
            Ok((model_report, weight_report, output)) => {
                model = model_report;
                weight = weight_report;
                let comparison = compare_prefix(
                    &output,
                    &target.first_values,
                    output.len().min(target.first_values.len()),
                );
                projection = json!({
                    "kernel": "rust_qk256_activation_quantized_scaled_same_input_cpu_oracle",
                    "input_stage": "attn_sub_norm",
                    "target_stage": "attn_o_out",
                    "input": row_report(&input.first_values),
                    "rust_same_input_output": row_report(&output),
                    "reference_target": row_report(&target.first_values),
                    "rust_same_input_vs_reference_target": comparison,
                });
            }
            Err(err) => {
                blocked_reasons
                    .push(format!("attn_output_same_input_projection_unavailable:{err}"));
            }
        }
    }

    let same_input_projection_available =
        projection.pointer("/rust_same_input_vs_reference_target").is_some();
    let same_input_projection_matches_reference = projection
        .pointer("/rust_same_input_vs_reference_target/max_abs_delta")
        .and_then(Value::as_f64)
        .is_some_and(|delta| delta <= 1.0e-3);
    let current_blocked_reasons = if blocked_reasons.is_empty() {
        if same_input_projection_matches_reference {
            vec![
                "same_input_attn_output_projection_match_does_not_explain_upstream_delta"
                    .to_string(),
            ]
        } else {
            vec!["same_input_attn_output_projection_mismatch".to_string()]
        }
    } else {
        blocked_reasons.clone()
    };
    let next_action = if !blocked_reasons.is_empty() {
        "regenerate full-prefix reference layer traces and ensure the model path is available"
    } else if same_input_projection_matches_reference {
        "treat attn_output projection math/layout as same-input compatible; localize the upstream attn_sub_norm input delta"
    } else {
        "inspect Rust QK256 attn_output weight layout, scale, and activation quantization before changing broader runtime math"
    };

    Ok(json!({
        "schema_version": 1,
        "receipt_type": "bitnet_reference_attn_output_same_input_parity",
        "diagnostic": "bitnet_reference_attn_output_same_input_parity",
        "producer": "cargo xtask bitnet-reference-attn-output-same-input-parity",
        "created_at": chrono::Utc::now().to_rfc3339(),
        "diagnostic_only": true,
        "claim_allowed": false,
        "promotion_allowed": false,
        "classification": "diagnostic_only",
        "inputs": {
            "reference": path_to_string(&reference_path),
            "model": path_to_string(&model_path),
            "weight": args.weight,
        },
        "reference_trace": {
            "capture_scope": trace.pointer("/capture_scope").cloned().unwrap_or(Value::Null),
            "warmup_skip_policy": trace.pointer("/warmup_skip_policy").cloned().unwrap_or(Value::Null),
            "n_tokens": trace.pointer("/n_tokens").cloned().unwrap_or(Value::Null),
            "n_outputs": trace.pointer("/n_outputs").cloned().unwrap_or(Value::Null),
            "sampled_output_token_index": trace.pointer("/sampled_output_token_index").cloned().unwrap_or(Value::Null),
            "sampled_output_token_id": trace.pointer("/sampled_output_token_id").cloned().unwrap_or(Value::Null),
            "attn_sub_norm": input.map(reference_record_summary).unwrap_or(Value::Null),
            "attn_o_out": target.map(reference_record_summary).unwrap_or(Value::Null),
        },
        "model": model,
        "weight": weight,
        "projection": projection,
        "decision": {
            "same_input_projection_available": same_input_projection_available,
            "same_input_projection_matches_reference": same_input_projection_matches_reference,
            "input_first_values_count": input_count,
            "target_first_values_count": target_count,
            "current_blocked_reasons": current_blocked_reasons,
            "next_action": next_action,
            "claim_allowed": false,
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    }))
}

fn project_attn_output_same_input(
    model_path: &Path,
    weight_name: &str,
    input: &[f32],
) -> Result<(Value, Value, Vec<f32>)> {
    use bitnet_models::formats::gguf::{GgufReader, GgufTensorType};
    use bitnet_models::loader::MmapFile;
    use bitnet_models::quant::i2s_qk256::gemv_qk256_activation_quantized_scaled;

    let mmap =
        MmapFile::open(model_path).map_err(|err| anyhow::anyhow!("opening GGUF model: {err}"))?;
    let reader = GgufReader::new(mmap.as_slice())
        .map_err(|err| anyhow::anyhow!("parsing GGUF model: {err}"))?;
    let info = reader
        .get_tensor_info_by_name(weight_name)
        .with_context(|| format!("QK256 weight '{weight_name}' not found in GGUF"))?;
    if info.tensor_type != GgufTensorType::I2_S {
        bail!("weight '{weight_name}' must be GGUF I2_S, got {:?}", info.tensor_type);
    }
    if info.shape.len() != 2 {
        bail!("weight '{weight_name}' must be 2D, got shape {:?}", info.shape);
    }
    let rows = info.shape[0];
    let cols = info.shape[1];
    if input.len() != cols {
        bail!(
            "reference attn_sub_norm input length {} does not match weight cols {}",
            input.len(),
            cols
        );
    }

    let data = reader
        .get_tensor_data_by_info(info)
        .map_err(|err| anyhow::anyhow!("reading raw data for '{weight_name}': {err}"))?;
    let row_stride_bytes = cols.div_ceil(256) * 64;
    let logical_bytes =
        rows.checked_mul(row_stride_bytes).context("QK256 logical byte count overflow")?;
    if data.len() < logical_bytes {
        bail!(
            "QK256 weight '{weight_name}' has {} bytes, shorter than logical {} bytes",
            data.len(),
            logical_bytes
        );
    }
    let scale = if data.len() >= logical_bytes + std::mem::size_of::<f32>() {
        f32::from_le_bytes(data[logical_bytes..logical_bytes + 4].try_into().unwrap())
    } else {
        1.0
    };
    let qk256_bytes = &data[..logical_bytes];
    let mut output = vec![0.0f32; rows];
    gemv_qk256_activation_quantized_scaled(
        qk256_bytes,
        input,
        &mut output,
        rows,
        cols,
        row_stride_bytes,
        scale,
    )
    .map_err(|err| anyhow::anyhow!("same-input QK256 projection failed: {err}"))?;

    let model_report = json!({
        "path": path_to_string(model_path),
        "exists": true,
    });
    let weight_report = json!({
        "name": weight_name,
        "gguf_dtype": format!("{:?}", info.tensor_type),
        "gguf_shape": info.shape,
        "rows": rows,
        "cols": cols,
        "row_stride_bytes": row_stride_bytes,
        "actual_bytes": data.len(),
        "logical_bytes": logical_bytes,
        "trailer_or_padding_bytes": data.len().saturating_sub(logical_bytes),
        "trailer_scale": scale,
    });
    Ok((model_report, weight_report, output))
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
            let full_shape = record
                .pointer("/full_shape")
                .and_then(Value::as_array)
                .map(|dims| {
                    dims.iter()
                        .map(|dim| dim.as_i64().context("reference full_shape dim is not integer"))
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?
                .unwrap_or_else(|| shape.clone());
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
                graph_op: record
                    .pointer("/graph_op")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                graph_sources: record
                    .pointer("/graph_sources")
                    .cloned()
                    .unwrap_or_else(|| json!([])),
                view_source: record.pointer("/view_source").cloned().unwrap_or(Value::Null),
                view_offset: record.pointer("/view_offset").and_then(Value::as_u64),
                full_shape,
                sample_offset: record.pointer("/sample_offset").and_then(Value::as_u64),
                token_axis: record.pointer("/token_axis").and_then(Value::as_i64),
                dtype: record
                    .pointer("/dtype")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                shape,
                nelements: record.pointer("/nelements").and_then(Value::as_u64).unwrap_or(0),
                rms: record.pointer("/stats/rms").and_then(Value::as_f64),
                first_values: f32_array_at(record, "/first_values"),
                values_available: record
                    .pointer("/values_available")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            })
        })
        .collect()
}

fn reference_trace_receipt(root: &Value) -> Result<&Value> {
    match root.pointer("/receipt_type").and_then(Value::as_str) {
        Some("bitnet_reference_layer_trace_run") => root
            .pointer("/sidecar/receipt")
            .context("layer trace run receipt missing /sidecar/receipt"),
        _ => Ok(root),
    }
}

fn reference_prompt_tokens(trace: &Value) -> Vec<u64> {
    trace
        .pointer("/ubatch_tokens")
        .and_then(Value::as_array)
        .map(|tokens| tokens.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default()
}

fn f32_array_at(root: &Value, pointer: &str) -> Vec<f32> {
    root.pointer(pointer)
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_f64).map(|v| v as f32).collect())
        .unwrap_or_default()
}

fn reference_inp_embd_summary(record: &Value) -> Value {
    json!({
        "name": record.pointer("/name").cloned().unwrap_or(Value::Null),
        "stage": record.pointer("/stage").cloned().unwrap_or(Value::Null),
        "graph_index": record.pointer("/graph_index").cloned().unwrap_or(Value::Null),
        "graph_op": record.pointer("/graph_op").cloned().unwrap_or(Value::Null),
        "graph_sources": record.pointer("/graph_sources").cloned().unwrap_or_else(|| json!([])),
        "shape": record.pointer("/shape").cloned().unwrap_or(Value::Null),
        "full_shape": record.pointer("/full_shape").cloned().unwrap_or(Value::Null),
        "nelements": record.pointer("/nelements").cloned().unwrap_or(Value::Null),
        "sample_offset": record.pointer("/sample_offset").cloned().unwrap_or(Value::Null),
        "token_axis": record.pointer("/token_axis").cloned().unwrap_or(Value::Null),
        "stats": record.pointer("/stats").cloned().unwrap_or(Value::Null),
        "first_values": record.pointer("/first_values").cloned().unwrap_or(Value::Null),
    })
}

fn build_embedding_row_authority_rows(
    model_path: &Path,
    token_ids: &[u64],
    sampled_output_token_index: Option<i64>,
    sampled_output_token_id: Option<u64>,
    expected_width: usize,
    reference_first_values: &[f32],
) -> Result<(Value, Value, Vec<Value>)> {
    let mmap = bitnet_models::loader::MmapFile::open(model_path)
        .map_err(|err| anyhow::anyhow!("opening GGUF model: {err}"))?;
    let reader = bitnet_models::formats::gguf::GgufReader::new(mmap.as_slice())
        .map_err(|err| anyhow::anyhow!("parsing GGUF model: {err}"))?;
    let info = ["token_embd.weight", "tok_embeddings.weight", "model.embed_tokens.weight"]
        .iter()
        .find_map(|name| reader.get_tensor_info_by_name(name))
        .context("token_embd.weight tensor not found in GGUF")?;
    let raw_data = reader
        .get_tensor_data_by_info(info)
        .map_err(|err| anyhow::anyhow!("reading raw token_embd.weight data: {err}"))?;
    let raw_layouts = embedding_raw_layouts(&info.shape, expected_width, token_ids);
    if raw_layouts.is_empty() {
        bail!("could not derive hidden-width token-row layout from GGUF token_embd.weight");
    }

    let load_result = bitnet_models::load_gguf_full(
        model_path,
        bitnet_common::Device::Cpu,
        bitnet_models::GGUFLoaderConfig::default(),
    )
    .map_err(|err| anyhow::anyhow!("Rust GGUF load failed: {err}"))?;
    let rust_embedding = ["token_embd.weight", "embed_tokens.weight", "tok_embeddings.weight"]
        .iter()
        .find_map(|name| load_result.tensors.get(*name).map(|tensor| (*name, tensor)))
        .context("Rust-loaded embedding tensor missing")?;

    let mut rows = Vec::new();
    for (prompt_index, &token_id) in token_ids.iter().enumerate() {
        let mut reference_candidates = Vec::new();
        let is_sampled_output_token = sampled_output_token_index
            .map(|index| index == prompt_index as i64)
            .unwrap_or_else(|| Some(token_id) == sampled_output_token_id);
        let rust_loaded = {
            let tensor = rust_embedding.1;
            let dims = tensor.shape().dims();
            if dims.len() != 2 {
                bail!("Rust-loaded embedding tensor must be 2D, got {dims:?}");
            }
            if dims[1] == expected_width && (token_id as usize) < dims[0] {
                tensor
                    .narrow(0, token_id as usize, 1)
                    .map_err(|err| anyhow::anyhow!("narrow Rust embedding row: {err}"))?
                    .flatten_all()
                    .map_err(|err| anyhow::anyhow!("flatten Rust embedding row: {err}"))?
                    .to_vec1::<f32>()
                    .map_err(|err| anyhow::anyhow!("copy Rust embedding row to host: {err}"))?
            } else if dims[0] == expected_width && (token_id as usize) < dims[1] {
                tensor
                    .narrow(1, token_id as usize, 1)
                    .map_err(|err| anyhow::anyhow!("narrow Rust embedding column: {err}"))?
                    .flatten_all()
                    .map_err(|err| anyhow::anyhow!("flatten Rust embedding column: {err}"))?
                    .to_vec1::<f32>()
                    .map_err(|err| anyhow::anyhow!("copy Rust embedding column to host: {err}"))?
            } else {
                bail!(
                    "could not select token {} with hidden width {} from Rust-loaded embedding shape {:?}",
                    token_id,
                    expected_width,
                    dims
                );
            }
        };
        for layout in &raw_layouts {
            let reference_raw =
                decode_embedding_row(raw_data, info.tensor_type, layout, token_id as usize)?;
            let trace_compare = if is_sampled_output_token && !reference_first_values.is_empty() {
                compare_prefix(&reference_raw, reference_first_values, reference_first_values.len())
            } else {
                Value::Null
            };
            reference_candidates.push(json!({
                "layout": layout,
                "row": row_report(&reference_raw),
                "reference_raw_vs_rust_loaded": compare_vectors(&reference_raw, &rust_loaded),
                "reference_raw_vs_trace_first_values": trace_compare,
            }));
        }
        rows.push(json!({
            "prompt_index": prompt_index,
            "token_id": token_id,
            "is_sampled_output_token": is_sampled_output_token,
            "reference_candidates": reference_candidates,
            "rust_loaded": row_report(&rust_loaded),
        }));
    }

    let model_report = json!({
        "path": path_to_string(model_path),
        "exists": true,
        "loader_mode": load_result.loader_mode.as_str(),
        "loader_config": "default_real_gguf_embedding_row_inspection",
        "fallback_used": load_result.loader_mode.fallback_used(),
        "config": {
            "vocab_size": load_result.config.model.vocab_size,
            "hidden_size": load_result.config.model.hidden_size,
            "num_layers": load_result.config.model.num_layers,
            "num_heads": load_result.config.model.num_heads,
            "num_key_value_heads": load_result.config.model.num_key_value_heads,
        },
    });
    let tensor_report = json!({
        "gguf_name": info.name,
        "gguf_dtype": format!("{:?}", info.tensor_type),
        "gguf_shape": info.shape,
        "gguf_size_bytes": info.size,
        "reference_raw_layout_candidates": raw_layouts,
        "rust_loaded_name": rust_embedding.0,
        "rust_loaded_shape": rust_embedding.1.shape().dims(),
    });

    Ok((model_report, tensor_report, rows))
}

fn embedding_raw_layouts(shape: &[usize], expected_width: usize, token_ids: &[u64]) -> Vec<Value> {
    if shape.len() != 2 || expected_width == 0 {
        return Vec::new();
    }
    let mut layouts = Vec::new();
    let max_token = token_ids.iter().copied().max().unwrap_or(0) as usize;
    if shape[0] == expected_width && max_token < shape[1] {
        layouts.push(json!({
            "kind": "ggml_ne0_hidden_by_vocab_token_column",
            "hidden": shape[0],
            "vocab": shape[1],
            "element_index_rule": "token_id * hidden + hidden_index",
        }));
        layouts.push(json!({
            "kind": "row_major_hidden_by_vocab_transposed_token_row",
            "hidden": shape[0],
            "vocab": shape[1],
            "element_index_rule": "hidden_index * vocab + token_id",
        }));
    }
    if shape[1] == expected_width && max_token < shape[0] {
        layouts.push(json!({
            "kind": "vocab_by_hidden_token_row",
            "hidden": shape[1],
            "vocab": shape[0],
            "element_index_rule": "token_id * hidden + hidden_index",
        }));
    }
    layouts
}

fn row_candidate_delta_le(row: &Value, pointer_suffix: &str, threshold: f64) -> bool {
    row.pointer("/reference_candidates").and_then(Value::as_array).is_some_and(|candidates| {
        candidates.iter().any(|candidate| {
            candidate
                .pointer(pointer_suffix)
                .and_then(Value::as_f64)
                .is_some_and(|delta| delta <= threshold)
        })
    })
}

fn row_candidate_matching_layouts(
    row: &Value,
    pointer_suffix: &str,
    threshold: f64,
) -> Vec<String> {
    row.pointer("/reference_candidates")
        .and_then(Value::as_array)
        .map(|candidates| {
            candidates
                .iter()
                .filter_map(|candidate| {
                    let matches = candidate
                        .pointer(pointer_suffix)
                        .and_then(Value::as_f64)
                        .is_some_and(|delta| delta <= threshold);
                    if !matches {
                        return None;
                    }
                    candidate.pointer("/layout/kind").and_then(Value::as_str).map(ToOwned::to_owned)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn decode_embedding_row(
    data: &[u8],
    dtype: bitnet_models::formats::gguf::GgufTensorType,
    layout: &Value,
    token_id: usize,
) -> Result<Vec<f32>> {
    let hidden = layout
        .pointer("/hidden")
        .and_then(Value::as_u64)
        .context("embedding layout missing hidden")? as usize;
    let vocab = layout
        .pointer("/vocab")
        .and_then(Value::as_u64)
        .context("embedding layout missing vocab")? as usize;
    if token_id >= vocab {
        bail!("token id {token_id} out of bounds for embedding vocab {vocab}");
    }
    let mut row = Vec::with_capacity(hidden);
    for hidden_index in 0..hidden {
        let index = match layout.pointer("/element_index_rule").and_then(Value::as_str) {
            Some("token_id * hidden + hidden_index") => token_id
                .checked_mul(hidden)
                .and_then(|start| start.checked_add(hidden_index))
                .context("embedding row offset overflow")?,
            Some("hidden_index * vocab + token_id") => hidden_index
                .checked_mul(vocab)
                .and_then(|start| start.checked_add(token_id))
                .context("embedding transposed row offset overflow")?,
            other => bail!("unsupported embedding layout index rule: {other:?}"),
        };
        row.push(decode_gguf_scalar(data, dtype, index)?);
    }
    Ok(row)
}

fn decode_gguf_scalar(
    data: &[u8],
    dtype: bitnet_models::formats::gguf::GgufTensorType,
    index: usize,
) -> Result<f32> {
    match dtype {
        bitnet_models::formats::gguf::GgufTensorType::F32 => {
            let offset = index.checked_mul(4).context("F32 offset overflow")?;
            let bytes = data
                .get(offset..offset + 4)
                .with_context(|| format!("F32 scalar index {index} out of bounds"))?;
            Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
        }
        bitnet_models::formats::gguf::GgufTensorType::F16 => {
            let offset = index.checked_mul(2).context("F16 offset overflow")?;
            let bytes = data
                .get(offset..offset + 2)
                .with_context(|| format!("F16 scalar index {index} out of bounds"))?;
            Ok(f16_bits_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())))
        }
        other => bail!("unsupported embedding dtype for row authority report: {other:?}"),
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = (bits & 0x03ff) as u32;
    let f32_bits = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut mant = frac;
                let mut e = -14i32;
                while (mant & 0x0400) == 0 {
                    mant <<= 1;
                    e -= 1;
                }
                mant &= 0x03ff;
                sign | (((e + 127) as u32) << 23) | (mant << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => {
            let exp32 = (exp as i32 - 15 + 127) as u32;
            sign | (exp32 << 23) | (frac << 13)
        }
    };
    f32::from_bits(f32_bits)
}

fn f32_to_f16_bits_nearest_even(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;

    if exp == 0 && mant == 0 {
        return (sign << 15) as u16;
    }
    if exp == 0xff {
        if mant == 0 {
            return ((sign << 15) | 0x7c00) as u16;
        }
        return ((sign << 15) | 0x7c00 | (mant >> 13).max(1)) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1f {
        return ((sign << 15) | 0x7c00) as u16;
    }

    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let full_mant = mant | 0x80_0000;
        let shift = (1 - new_exp) as u32 + 13;
        let half_mant = if shift >= 32 { 0 } else { full_mant >> shift };
        let round_bit = if shift == 0 || shift > 32 { 0 } else { (full_mant >> (shift - 1)) & 1 };
        let sticky_mask = if shift <= 1 {
            0
        } else if shift - 1 >= 32 {
            u32::MAX
        } else {
            (1u32 << (shift - 1)) - 1
        };
        let rounded = round_nearest_even(half_mant, round_bit, (full_mant & sticky_mask) != 0);
        return ((sign << 15) | rounded) as u16;
    }

    let half_mant = mant >> 13;
    let round_bit = (mant >> 12) & 1;
    let sticky = (mant & 0xfff) != 0;
    let base = (sign << 15) | ((new_exp as u32) << 10) | half_mant;
    round_nearest_even(base, round_bit, sticky) as u16
}

fn round_nearest_even(base: u32, round_bit: u32, sticky: bool) -> u32 {
    if round_bit == 1 && (sticky || (base & 1) == 1) { base + 1 } else { base }
}

fn f16_roundtrip(value: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits_nearest_even(value))
}

fn row_report(values: &[f32]) -> Value {
    let stats = vector_stats(values);
    json!({
        "count": values.len(),
        "stats": stats,
        "sha256": vector_sha256(values),
        "first_values": values.iter().take(16).copied().collect::<Vec<_>>(),
    })
}

fn vector_stats(values: &[f32]) -> Value {
    if values.is_empty() {
        return json!({
            "mean": null,
            "rms": null,
            "min": null,
            "max": null,
        });
    }
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &value in values {
        sum += value as f64;
        sum_sq += (value as f64) * (value as f64);
        min = min.min(value);
        max = max.max(value);
    }
    json!({
        "mean": sum / values.len() as f64,
        "rms": (sum_sq / values.len() as f64).sqrt(),
        "min": min,
        "max": max,
    })
}

fn vector_sha256(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn compare_prefix(left: &[f32], right: &[f32], len: usize) -> Value {
    let n = len.min(left.len()).min(right.len());
    compare_vectors(&left[..n], &right[..n])
}

fn compare_vectors(left: &[f32], right: &[f32]) -> Value {
    let n = left.len().min(right.len());
    let mut max_abs_delta = 0.0f64;
    let mut sum_sq_delta = 0.0f64;
    let mut first_mismatch_index = None::<usize>;
    for i in 0..n {
        let delta = (left[i] as f64 - right[i] as f64).abs();
        if delta > max_abs_delta {
            max_abs_delta = delta;
        }
        if delta > 1.0e-6 && first_mismatch_index.is_none() {
            first_mismatch_index = Some(i);
        }
        sum_sq_delta += delta * delta;
    }
    json!({
        "left_count": left.len(),
        "right_count": right.len(),
        "compared_count": n,
        "count_match": left.len() == right.len(),
        "max_abs_delta": max_abs_delta,
        "rms_abs_delta": if n == 0 { 0.0 } else { (sum_sq_delta / n as f64).sqrt() },
        "first_mismatch_index": first_mismatch_index,
        "sha256_match": vector_sha256(left) == vector_sha256(right),
    })
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
        ("kq", "attention_scores_raw_head0"),
        ("kq_head0", "attention_scores_raw_head0"),
        ("kq_head1", "attention_scores_raw_head1"),
        ("kq_head2", "attention_scores_raw_head2"),
        ("kq_head3", "attention_scores_raw_head3"),
        ("kq_head4", "attention_scores_raw_head4"),
        ("kq_head5", "attention_scores_raw_head5"),
        ("kq_head6", "attention_scores_raw_head6"),
        ("kq_head7", "attention_scores_raw_head7"),
        ("kq_head8", "attention_scores_raw_head8"),
        ("kq_head9", "attention_scores_raw_head9"),
        ("kq_head10", "attention_scores_raw_head10"),
        ("kq_head11", "attention_scores_raw_head11"),
        ("kq_head12", "attention_scores_raw_head12"),
        ("kq_head13", "attention_scores_raw_head13"),
        ("kq_head14", "attention_scores_raw_head14"),
        ("kq_head15", "attention_scores_raw_head15"),
        ("kq_head16", "attention_scores_raw_head16"),
        ("kq_head17", "attention_scores_raw_head17"),
        ("kq_head18", "attention_scores_raw_head18"),
        ("kq_head19", "attention_scores_raw_head19"),
        ("kq_soft_max_ext", "attn_scores_softmax_head0"),
        ("kq_soft_max_ext_head0", "attn_scores_softmax_head0"),
        ("kq_soft_max_ext_head1", "attn_scores_softmax_head1"),
        ("kq_soft_max_ext_head2", "attn_scores_softmax_head2"),
        ("kq_soft_max_ext_head3", "attn_scores_softmax_head3"),
        ("kq_soft_max_ext_head4", "attn_scores_softmax_head4"),
        ("kq_soft_max_ext_head5", "attn_scores_softmax_head5"),
        ("kq_soft_max_ext_head6", "attn_scores_softmax_head6"),
        ("kq_soft_max_ext_head7", "attn_scores_softmax_head7"),
        ("kq_soft_max_ext_head8", "attn_scores_softmax_head8"),
        ("kq_soft_max_ext_head9", "attn_scores_softmax_head9"),
        ("kq_soft_max_ext_head10", "attn_scores_softmax_head10"),
        ("kq_soft_max_ext_head11", "attn_scores_softmax_head11"),
        ("kq_soft_max_ext_head12", "attn_scores_softmax_head12"),
        ("kq_soft_max_ext_head13", "attn_scores_softmax_head13"),
        ("kq_soft_max_ext_head14", "attn_scores_softmax_head14"),
        ("kq_soft_max_ext_head15", "attn_scores_softmax_head15"),
        ("kq_soft_max_ext_head16", "attn_scores_softmax_head16"),
        ("kq_soft_max_ext_head17", "attn_scores_softmax_head17"),
        ("kq_soft_max_ext_head18", "attn_scores_softmax_head18"),
        ("kq_soft_max_ext_head19", "attn_scores_softmax_head19"),
        ("k", "attention_k_cache_head0_ref_layout_padded"),
        ("k_kv_head0_live", "attention_k_cache_kv_head0_live_ref_layout"),
        ("k_kv_head1_live", "attention_k_cache_kv_head1_live_ref_layout"),
        ("k_kv_head2_live", "attention_k_cache_kv_head2_live_ref_layout"),
        ("k_kv_head3_live", "attention_k_cache_kv_head3_live_ref_layout"),
        ("k_kv_head4_live", "attention_k_cache_kv_head4_live_ref_layout"),
        ("v", "attention_v_cache_head0_ref_layout_padded"),
        ("v_kv_head0_live", "attention_v_cache_kv_head0_live_ref_layout"),
        ("v_kv_head1_live", "attention_v_cache_kv_head1_live_ref_layout"),
        ("v_kv_head2_live", "attention_v_cache_kv_head2_live_ref_layout"),
        ("v_kv_head3_live", "attention_v_cache_kv_head3_live_ref_layout"),
        ("v_kv_head4_live", "attention_v_cache_kv_head4_live_ref_layout"),
        (
            "v_cache_rust_layout_head0_live",
            "attention_v_cache_f16_roundtrip_kv_head0_live_ref_layout",
        ),
        (
            "v_cache_rust_layout_head1_live",
            "attention_v_cache_f16_roundtrip_kv_head1_live_ref_layout",
        ),
        (
            "v_cache_rust_layout_head2_live",
            "attention_v_cache_f16_roundtrip_kv_head2_live_ref_layout",
        ),
        (
            "v_cache_rust_layout_head3_live",
            "attention_v_cache_f16_roundtrip_kv_head3_live_ref_layout",
        ),
        (
            "v_cache_rust_layout_head4_live",
            "attention_v_cache_f16_roundtrip_kv_head4_live_ref_layout",
        ),
        ("kqv", "attention_value_mix_head0"),
        ("kqv_head0", "attention_value_mix_head0"),
        ("kqv_head1", "attention_value_mix_head1"),
        ("kqv_head2", "attention_value_mix_head2"),
        ("kqv_head3", "attention_value_mix_head3"),
        ("kqv_head4", "attention_value_mix_head4"),
        ("kqv_head5", "attention_value_mix_head5"),
        ("kqv_head6", "attention_value_mix_head6"),
        ("kqv_head7", "attention_value_mix_head7"),
        ("kqv_head8", "attention_value_mix_head8"),
        ("kqv_head9", "attention_value_mix_head9"),
        ("kqv_head10", "attention_value_mix_head10"),
        ("kqv_head11", "attention_value_mix_head11"),
        ("kqv_head12", "attention_value_mix_head12"),
        ("kqv_head13", "attention_value_mix_head13"),
        ("kqv_head14", "attention_value_mix_head14"),
        ("kqv_head15", "attention_value_mix_head15"),
        ("kqv_head16", "attention_value_mix_head16"),
        ("kqv_head17", "attention_value_mix_head17"),
        ("kqv_head18", "attention_value_mix_head18"),
        ("kqv_head19", "attention_value_mix_head19"),
        ("kqv_merged", "attention_value_mix_merged"),
        ("attn_value_mix", "attention_value_mix_merged"),
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
    let mut scope_mismatch_count = 0usize;
    let mut missing_reference_count = 0usize;
    let mut missing_rust_count = 0usize;
    let mut first_scope_mismatch = Value::Null;

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
        let mut first_values_delta = Value::Null;
        let mut material_mismatch = reference.is_none() || rust.is_none();
        let scope_mismatch = trace_scope_mismatch(reference, rust);
        let has_scope_mismatch = scope_mismatch.is_some();
        if let (Some(reference), Some(rust)) = (reference, rust) {
            compared_count += 1;
            let element_count_match = reference.nelements == rust.num_elements as u64;
            let dtype_match = trace_dtype_compatible(&reference.dtype, &rust.dtype);
            rms_abs_delta = reference.rms.map(|rms| (rms - rust.rms).abs());
            let rms_material = rms_abs_delta.is_some_and(|delta| delta > 1.0e-4);
            if !reference.first_values.is_empty() && !rust.first_values.is_empty() {
                first_values_delta = compare_prefix(
                    &reference.first_values,
                    &rust.first_values,
                    reference.first_values.len().min(rust.first_values.len()),
                );
            }
            if has_scope_mismatch {
                material_mismatch = false;
                status = "scope_mismatch";
            } else {
                material_mismatch = !element_count_match || !dtype_match || rms_material;
                status = if material_mismatch { "material_mismatch" } else { "summary_match" };
            }
        }
        if has_scope_mismatch {
            scope_mismatch_count += 1;
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
            "first_values_delta": first_values_delta,
            "scope_mismatch": has_scope_mismatch,
            "scope": scope_mismatch,
            "material_mismatch": material_mismatch,
        });
        if has_scope_mismatch && first_scope_mismatch.is_null() {
            first_scope_mismatch = stage.clone();
        }
        if material_mismatch && first_material_mismatch.is_null() {
            first_material_mismatch = stage.clone();
        }
        stages.push(stage);
    }

    json!({
        "trace_record_count": rust_records.len(),
        "compared_stage_count": compared_count,
        "material_mismatch_count": material_mismatch_count,
        "scope_mismatch_count": scope_mismatch_count,
        "missing_reference_count": missing_reference_count,
        "missing_rust_count": missing_rust_count,
        "first_scope_mismatch": first_scope_mismatch,
        "first_material_mismatch": first_material_mismatch,
        "attention_query_rope_ref_layout_delta": attention_query_rope_ref_layout_delta(reference_records, rust_records),
        "attention_score_reference_scalar_recompute": attention_score_reference_scalar_recompute(reference_records),
        "attention_score_reference_semantic_variants": attention_score_reference_semantic_variants(reference_records),
        "attention_score_reference_numeric_variants": attention_score_reference_numeric_variants(reference_records),
        "attention_score_input_attribution": attention_score_input_attribution(reference_records, rust_records),
        "attention_score_rust_scalar_recompute": attention_score_rust_scalar_recompute(rust_records),
        "attention_score_raw_head_lane_best_matches": attention_score_raw_head_lane_best_matches(reference_records, rust_records),
        "attention_probability_reference_softmax_variants": attention_probability_reference_softmax_variants(reference_records),
        "attention_probability_rust_softmax_recompute": attention_probability_rust_softmax_recompute(rust_records),
        "attention_probability_head_lane_best_matches": attention_probability_head_lane_best_matches(reference_records, rust_records),
        "attention_key_cache_kv_head_best_matches": attention_key_cache_kv_head_best_matches(reference_records, rust_records),
        "attention_key_cache_f16_roundtrip_best_matches": attention_key_cache_f16_roundtrip_best_matches(reference_records, rust_records),
        "attention_key_cache_dim_major_f16_roundtrip_best_matches": attention_key_cache_dim_major_f16_roundtrip_best_matches(reference_records, rust_records),
        "attention_value_cache_kv_head_best_matches": attention_value_cache_kv_head_best_matches(reference_records, rust_records),
        "attention_value_cache_rust_layout_best_matches": attention_value_cache_rust_layout_best_matches(reference_records, rust_records),
        "attention_value_cache_f16_roundtrip_best_matches": attention_value_cache_f16_roundtrip_best_matches(reference_records, rust_records),
        "attention_value_cache_f16_amplification": attention_value_cache_f16_amplification(reference_records, rust_records),
        "attention_value_mix_reference_scalar_recompute": attention_value_mix_reference_scalar_recompute(reference_records),
        "attention_value_mix_reference_numeric_variants": attention_value_mix_reference_numeric_variants(reference_records),
        "attention_value_mix_rust_scalar_recompute": attention_value_mix_rust_scalar_recompute(rust_records),
        "attention_value_mix_input_attribution": attention_value_mix_input_attribution(reference_records, rust_records),
        "attention_value_mix_f16_cache_head_lane_best_matches": attention_value_mix_f16_cache_head_lane_best_matches(reference_records, rust_records),
        "attention_value_mix_head_lane_best_matches": attention_value_mix_head_lane_best_matches(reference_records, rust_records),
        "stages": stages,
    })
}

#[derive(Debug, Clone, Copy)]
struct HeadLaneDelta {
    reference_head: usize,
    rust_head: usize,
    compared_count: usize,
    max_abs_delta: f64,
    rms_abs_delta: f64,
}

fn attention_value_mix_head_lane_best_matches(
    reference_records: &[ReferenceTraceRecord],
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    head_lane_best_matches(
        reference_records,
        rust_records,
        "kqv_head",
        "attention_value_mix_head",
        "head-lane best matches are diagnostic mapping evidence only; they do not promote reference parity, A770 semantic quality, selected attention, value mix residency, or any support claim",
    )
}

fn attention_value_mix_f16_cache_head_lane_best_matches(
    reference_records: &[ReferenceTraceRecord],
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    head_lane_best_matches(
        reference_records,
        rust_records,
        "kqv_head",
        "attention_value_mix_f16_cache_head",
        "F16-cache value-mix head-lane best matches are diagnostic alternate-path evidence only; they do not promote reference parity, A770 semantic quality, value mix residency, resident KV, selected attention, or any support claim",
    )
}

fn attention_value_mix_reference_scalar_recompute(
    reference_records: &[ReferenceTraceRecord],
) -> Value {
    let probability_heads = reference_records
        .iter()
        .filter_map(|record| {
            parse_stage_head(&record.stage, "kq_soft_max_ext_head").map(|head| (head, record))
        })
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let value_cache_heads = reference_records
        .iter()
        .filter_map(|record| {
            parse_stage_head(&record.stage, "v_cache_rust_layout_head").map(|head| (head, record))
        })
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let value_mix_heads = reference_records
        .iter()
        .filter_map(|record| parse_stage_head(&record.stage, "kqv_head").map(|head| (head, record)))
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();

    let group_size = if value_cache_heads.is_empty()
        || value_mix_heads.is_empty()
        || !value_mix_heads.len().is_multiple_of(value_cache_heads.len())
    {
        None
    } else {
        Some(value_mix_heads.len() / value_cache_heads.len())
    };

    let mut rows = Vec::new();
    let mut compared_count = 0usize;
    let mut missing_input_count = 0usize;
    let mut max_rms_delta = 0.0f64;
    let mut max_abs_delta = 0.0f64;

    for (&head, value_mix) in &value_mix_heads {
        let kv_head = group_size.map(|group_size| head / group_size);
        let probability = probability_heads.get(&head).copied();
        let value_cache = kv_head.and_then(|kv_head| value_cache_heads.get(&kv_head).copied());
        let mut status = "missing_input";
        let mut delta = Value::Null;
        let mut recomputed_first_values = Vec::<f32>::new();
        let mut token_count = None::<usize>;
        let mut value_dim = None::<usize>;

        if let (Some(probability), Some(value_cache)) = (probability, value_cache) {
            value_dim = usize::try_from(value_mix.nelements).ok();
            token_count =
                value_cache.shape.get(1).and_then(|dim| usize::try_from(*dim).ok()).or_else(|| {
                    value_dim.and_then(|dim| {
                        usize::try_from(value_cache.nelements).ok().map(|n| n / dim)
                    })
                });
            if let (Some(value_dim), Some(token_count)) = (value_dim, token_count) {
                let value_sample_count = value_dim.saturating_mul(token_count);
                if probability.first_values.len() >= token_count
                    && value_cache.first_values.len() >= value_sample_count
                    && value_mix.first_values.len() >= value_dim
                {
                    for dim in 0..value_dim {
                        let mut sum = 0.0f64;
                        for token in 0..token_count {
                            let probability = probability.first_values[token] as f64;
                            let value = value_cache.first_values[dim * token_count + token] as f64;
                            sum += probability * value;
                        }
                        recomputed_first_values.push(sum as f32);
                    }
                    delta = compare_prefix(
                        &recomputed_first_values,
                        &value_mix.first_values,
                        value_dim,
                    );
                    max_rms_delta = max_rms_delta.max(
                        delta.pointer("/rms_abs_delta").and_then(Value::as_f64).unwrap_or(0.0),
                    );
                    max_abs_delta = max_abs_delta.max(
                        delta.pointer("/max_abs_delta").and_then(Value::as_f64).unwrap_or(0.0),
                    );
                    compared_count += 1;
                    status = "compared";
                }
            }
        }

        if status == "missing_input" {
            missing_input_count += 1;
        }
        rows.push(json!({
            "head": head,
            "kv_head": kv_head,
            "status": status,
            "probability_stage_present": probability.is_some(),
            "value_cache_stage_present": value_cache.is_some(),
            "value_mix_stage_present": true,
            "token_count": token_count,
            "value_dim": value_dim,
            "delta": delta,
            "recomputed_first_values": recomputed_first_values,
        }));
    }

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "policy": "reference scalar value-mix recompute is diagnostic arithmetic evidence only; it does not promote reference parity, A770 semantic quality, value mix residency, selected attention, resident KV, or any support claim",
        "probability_head_count": probability_heads.len(),
        "value_cache_head_count": value_cache_heads.len(),
        "value_mix_head_count": value_mix_heads.len(),
        "group_size": group_size,
        "compared_count": compared_count,
        "missing_input_count": missing_input_count,
        "max_abs_delta": max_abs_delta,
        "max_rms_delta": max_rms_delta,
        "all_compared": !value_mix_heads.is_empty() && missing_input_count == 0,
        "rows": rows,
    })
}

#[derive(Debug, Clone, Copy)]
struct ReferenceValueMixNumericVariantSpec {
    id: &'static str,
    probability_f16_roundtrip: bool,
    value_f16_roundtrip: bool,
    output_f16_roundtrip: bool,
    accum_policy: ReferenceScoreAccumPolicy,
}

fn reference_value_mix_numeric_variant_specs() -> [ReferenceValueMixNumericVariantSpec; 10] {
    [
        ReferenceValueMixNumericVariantSpec {
            id: "reference_value_mix_numeric_f64_accum_p_f32_v_f32",
            probability_f16_roundtrip: false,
            value_f16_roundtrip: false,
            output_f16_roundtrip: false,
            accum_policy: ReferenceScoreAccumPolicy::F64,
        },
        ReferenceValueMixNumericVariantSpec {
            id: "reference_value_mix_numeric_f32_accum_p_f32_v_f32",
            probability_f16_roundtrip: false,
            value_f16_roundtrip: false,
            output_f16_roundtrip: false,
            accum_policy: ReferenceScoreAccumPolicy::F32,
        },
        ReferenceValueMixNumericVariantSpec {
            id: "reference_value_mix_numeric_f32_mul_add_p_f32_v_f32",
            probability_f16_roundtrip: false,
            value_f16_roundtrip: false,
            output_f16_roundtrip: false,
            accum_policy: ReferenceScoreAccumPolicy::F32MulAdd,
        },
        ReferenceValueMixNumericVariantSpec {
            id: "reference_value_mix_numeric_f32_accum_p_f16_v_f32",
            probability_f16_roundtrip: true,
            value_f16_roundtrip: false,
            output_f16_roundtrip: false,
            accum_policy: ReferenceScoreAccumPolicy::F32,
        },
        ReferenceValueMixNumericVariantSpec {
            id: "reference_value_mix_numeric_f32_accum_p_f32_v_f16",
            probability_f16_roundtrip: false,
            value_f16_roundtrip: true,
            output_f16_roundtrip: false,
            accum_policy: ReferenceScoreAccumPolicy::F32,
        },
        ReferenceValueMixNumericVariantSpec {
            id: "reference_value_mix_numeric_f32_accum_p_f16_v_f16",
            probability_f16_roundtrip: true,
            value_f16_roundtrip: true,
            output_f16_roundtrip: false,
            accum_policy: ReferenceScoreAccumPolicy::F32,
        },
        ReferenceValueMixNumericVariantSpec {
            id: "reference_value_mix_numeric_f32_mul_add_p_f16_v_f16",
            probability_f16_roundtrip: true,
            value_f16_roundtrip: true,
            output_f16_roundtrip: false,
            accum_policy: ReferenceScoreAccumPolicy::F32MulAdd,
        },
        ReferenceValueMixNumericVariantSpec {
            id: "reference_value_mix_numeric_f32_accum_p_f32_v_f32_out_f16",
            probability_f16_roundtrip: false,
            value_f16_roundtrip: false,
            output_f16_roundtrip: true,
            accum_policy: ReferenceScoreAccumPolicy::F32,
        },
        ReferenceValueMixNumericVariantSpec {
            id: "reference_value_mix_numeric_f32_accum_p_f32_v_f16_out_f16",
            probability_f16_roundtrip: false,
            value_f16_roundtrip: true,
            output_f16_roundtrip: true,
            accum_policy: ReferenceScoreAccumPolicy::F32,
        },
        ReferenceValueMixNumericVariantSpec {
            id: "reference_value_mix_numeric_f32_accum_p_f16_v_f16_out_f16",
            probability_f16_roundtrip: true,
            value_f16_roundtrip: true,
            output_f16_roundtrip: true,
            accum_policy: ReferenceScoreAccumPolicy::F32,
        },
    ]
}

fn reference_value_mix_row_numeric(
    probability: &ReferenceTraceRecord,
    value_cache: &ReferenceTraceRecord,
    value_dim: usize,
    token_count: usize,
    variant: ReferenceValueMixNumericVariantSpec,
) -> Option<Vec<f32>> {
    if value_dim == 0 || token_count == 0 {
        return None;
    }
    if probability.first_values.len() < token_count
        || value_cache.first_values.len() < value_dim.checked_mul(token_count)?
    {
        return None;
    }

    let mut values = Vec::with_capacity(value_dim);
    for dim in 0..value_dim {
        let value = match variant.accum_policy {
            ReferenceScoreAccumPolicy::F64 => {
                let mut sum = 0.0f64;
                for token in 0..token_count {
                    let probability = numeric_variant_value(
                        probability.first_values[token],
                        variant.probability_f16_roundtrip,
                    ) as f64;
                    let value = numeric_variant_value(
                        value_cache.first_values[dim * token_count + token],
                        variant.value_f16_roundtrip,
                    ) as f64;
                    sum += probability * value;
                }
                sum as f32
            }
            ReferenceScoreAccumPolicy::F32 => {
                let mut sum = 0.0f32;
                for token in 0..token_count {
                    let probability = numeric_variant_value(
                        probability.first_values[token],
                        variant.probability_f16_roundtrip,
                    );
                    let value = numeric_variant_value(
                        value_cache.first_values[dim * token_count + token],
                        variant.value_f16_roundtrip,
                    );
                    sum += probability * value;
                }
                sum
            }
            ReferenceScoreAccumPolicy::F32MulAdd => {
                let mut sum = 0.0f32;
                for token in 0..token_count {
                    let probability = numeric_variant_value(
                        probability.first_values[token],
                        variant.probability_f16_roundtrip,
                    );
                    let value = numeric_variant_value(
                        value_cache.first_values[dim * token_count + token],
                        variant.value_f16_roundtrip,
                    );
                    sum = probability.mul_add(value, sum);
                }
                sum
            }
        };
        values.push(if variant.output_f16_roundtrip { f16_roundtrip(value) } else { value });
    }
    Some(values)
}

fn attention_value_mix_reference_numeric_variants(
    reference_records: &[ReferenceTraceRecord],
) -> Value {
    let probability_heads = reference_records
        .iter()
        .filter_map(|record| {
            parse_stage_head(&record.stage, "kq_soft_max_ext_head").map(|head| (head, record))
        })
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let value_cache_heads = reference_records
        .iter()
        .filter_map(|record| {
            parse_stage_head(&record.stage, "v_cache_rust_layout_head").map(|head| (head, record))
        })
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let value_mix_heads = reference_records
        .iter()
        .filter_map(|record| parse_stage_head(&record.stage, "kqv_head").map(|head| (head, record)))
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let variants = reference_value_mix_numeric_variant_specs();

    let group_size = if value_cache_heads.is_empty()
        || value_mix_heads.is_empty()
        || !value_mix_heads.len().is_multiple_of(value_cache_heads.len())
    {
        None
    } else {
        Some(value_mix_heads.len() / value_cache_heads.len())
    };

    let mut rows = Vec::new();
    let mut compared_count = 0usize;
    let mut missing_input_count = 0usize;
    let mut unexplained_head_count = 0usize;
    let mut max_best_abs_delta = 0.0f64;
    let mut max_best_rms_delta = 0.0f64;
    let mut best_variant_counts = BTreeMap::<String, usize>::new();

    for (&head, value_mix) in &value_mix_heads {
        let kv_head = group_size.map(|group_size| head / group_size);
        let probability = probability_heads.get(&head).copied();
        let value_cache = kv_head.and_then(|kv_head| value_cache_heads.get(&kv_head).copied());
        let value_dim = usize::try_from(value_mix.nelements).ok();
        let token_count = value_cache.and_then(|value_cache| {
            value_cache.shape.get(1).and_then(|dim| usize::try_from(*dim).ok()).or_else(|| {
                value_dim
                    .and_then(|dim| usize::try_from(value_cache.nelements).ok().map(|n| n / dim))
            })
        });

        if let (Some(probability), Some(value_cache), Some(value_dim), Some(token_count)) =
            (probability, value_cache, value_dim, token_count)
        {
            let mut variant_rows = Vec::new();
            let mut best_variant_id = "";
            let mut best_delta = Value::Null;
            let mut best_rank = (f64::INFINITY, f64::INFINITY, true);
            let mut best_accum_policy = "";
            let mut best_probability_f16_roundtrip = false;
            let mut best_value_f16_roundtrip = false;
            let mut best_output_f16_roundtrip = false;

            for variant in variants {
                if let Some(values) = reference_value_mix_row_numeric(
                    probability,
                    value_cache,
                    value_dim,
                    token_count,
                    variant,
                ) {
                    let delta = compare_vectors(&values, &value_mix.first_values);
                    let rms = delta_metric(&delta, "/rms_abs_delta");
                    let max_abs = delta_metric(&delta, "/max_abs_delta");
                    let count_mismatch =
                        !delta.pointer("/count_match").and_then(Value::as_bool).unwrap_or(false);
                    let rank = (rms, max_abs, count_mismatch);
                    if rank < best_rank {
                        best_rank = rank;
                        best_variant_id = variant.id;
                        best_delta = delta.clone();
                        best_accum_policy = variant.accum_policy.label();
                        best_probability_f16_roundtrip = variant.probability_f16_roundtrip;
                        best_value_f16_roundtrip = variant.value_f16_roundtrip;
                        best_output_f16_roundtrip = variant.output_f16_roundtrip;
                    }
                    variant_rows.push(json!({
                        "variant": variant.id,
                        "head": head,
                        "kv_head": kv_head,
                        "token_count": token_count,
                        "value_dim": value_dim,
                        "accum_policy": variant.accum_policy.label(),
                        "probability_f16_roundtrip": variant.probability_f16_roundtrip,
                        "value_f16_roundtrip": variant.value_f16_roundtrip,
                        "output_f16_roundtrip": variant.output_f16_roundtrip,
                        "max_abs_delta": max_abs,
                        "rms_delta": rms,
                        "delta": delta,
                    }));
                }
            }

            if variant_rows.is_empty() {
                missing_input_count += 1;
                rows.push(json!({
                    "head": head,
                    "kv_head": kv_head,
                    "status": "missing_input",
                    "probability_stage_present": true,
                    "value_cache_stage_present": true,
                    "value_mix_stage_present": true,
                }));
                continue;
            }

            let head_explained = reference_score_variant_explained(&best_delta);
            if !head_explained {
                unexplained_head_count += 1;
            }
            max_best_abs_delta =
                max_best_abs_delta.max(delta_metric(&best_delta, "/max_abs_delta"));
            max_best_rms_delta =
                max_best_rms_delta.max(delta_metric(&best_delta, "/rms_abs_delta"));
            *best_variant_counts.entry(best_variant_id.to_string()).or_insert(0) += 1;
            compared_count += 1;
            rows.push(json!({
                "head": head,
                "kv_head": kv_head,
                "status": "compared",
                "token_count": token_count,
                "value_dim": value_dim,
                "best_variant": best_variant_id,
                "accum_policy": best_accum_policy,
                "probability_f16_roundtrip": best_probability_f16_roundtrip,
                "value_f16_roundtrip": best_value_f16_roundtrip,
                "output_f16_roundtrip": best_output_f16_roundtrip,
                "max_abs_delta": delta_metric(&best_delta, "/max_abs_delta"),
                "rms_delta": delta_metric(&best_delta, "/rms_abs_delta"),
                "best_delta": best_delta,
                "head_explained": head_explained,
                "variants": variant_rows,
            }));
        } else {
            missing_input_count += 1;
            rows.push(json!({
                "head": head,
                "kv_head": kv_head,
                "status": "missing_input",
                "probability_stage_present": probability.is_some(),
                "value_cache_stage_present": value_cache.is_some(),
                "value_mix_stage_present": true,
                "token_count": token_count,
                "value_dim": value_dim,
            }));
        }
    }

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "policy": "reference numeric value-mix variants are diagnostic arithmetic evidence only; they do not promote reference parity, A770 semantic quality, value mix residency, selected attention, resident KV, or any support claim",
        "probability_head_count": probability_heads.len(),
        "value_cache_head_count": value_cache_heads.len(),
        "value_mix_head_count": value_mix_heads.len(),
        "group_size": group_size,
        "variant_count": variants.len(),
        "variants_tested": variants.iter().map(|variant| variant.id).collect::<Vec<_>>(),
        "explanation_abs_threshold": 1.0e-4,
        "explanation_rms_threshold": 1.0e-4,
        "compared_count": compared_count,
        "missing_input_count": missing_input_count,
        "unexplained_head_count": unexplained_head_count,
        "max_best_abs_delta": max_best_abs_delta,
        "max_best_rms_delta": max_best_rms_delta,
        "best_variant_counts": best_variant_counts,
        "all_heads_explained": !value_mix_heads.is_empty()
            && missing_input_count == 0
            && unexplained_head_count == 0,
        "rows": rows,
    })
}

fn attention_value_mix_rust_scalar_recompute(
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    let probability_heads = rust_records
        .iter()
        .filter_map(|(stage, record)| {
            parse_stage_head(stage, "attn_scores_softmax_head").map(|head| (head, record))
        })
        .filter(|(_, record)| !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let value_cache_heads = rust_records
        .iter()
        .filter_map(|(stage, record)| {
            parse_stage_head(stage, "attention_v_cache_f16_roundtrip_kv_head")
                .map(|head| (head, record))
        })
        .filter(|(_, record)| !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let value_mix_heads = rust_records
        .iter()
        .filter_map(|(stage, record)| {
            parse_stage_head(stage, "attention_value_mix_f16_cache_head").map(|head| (head, record))
        })
        .filter(|(_, record)| !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();

    let group_size = if value_cache_heads.is_empty()
        || value_mix_heads.is_empty()
        || !value_mix_heads.len().is_multiple_of(value_cache_heads.len())
    {
        None
    } else {
        Some(value_mix_heads.len() / value_cache_heads.len())
    };

    let mut rows = Vec::new();
    let mut compared_count = 0usize;
    let mut missing_input_count = 0usize;
    let mut max_rms_delta = 0.0f64;
    let mut max_abs_delta = 0.0f64;

    for (&head, value_mix) in &value_mix_heads {
        let kv_head = group_size.map(|group_size| head / group_size);
        let probability = probability_heads.get(&head).copied();
        let value_cache = kv_head.and_then(|kv_head| value_cache_heads.get(&kv_head).copied());
        let mut status = "missing_input";
        let mut delta = Value::Null;
        let mut recomputed_first_values = Vec::<f32>::new();
        let mut token_count = None::<usize>;
        let value_dim = Some(value_mix.num_elements);

        if let (Some(probability), Some(value_cache), Some(value_dim)) =
            (probability, value_cache, value_dim)
        {
            token_count = value_cache.shape.get(1).copied().or_else(|| {
                if value_dim == 0 { None } else { Some(value_cache.num_elements / value_dim) }
            });
            if let Some(token_count) = token_count {
                let value_sample_count = value_dim.saturating_mul(token_count);
                if probability.first_values.len() >= token_count
                    && value_cache.first_values.len() >= value_sample_count
                    && value_mix.first_values.len() >= value_dim
                {
                    for dim in 0..value_dim {
                        let mut sum = 0.0f64;
                        for token in 0..token_count {
                            let probability = probability.first_values[token] as f64;
                            let value = value_cache.first_values[dim * token_count + token] as f64;
                            sum += probability * value;
                        }
                        recomputed_first_values.push(sum as f32);
                    }
                    delta = compare_prefix(
                        &recomputed_first_values,
                        &value_mix.first_values,
                        value_dim,
                    );
                    max_rms_delta = max_rms_delta.max(
                        delta.pointer("/rms_abs_delta").and_then(Value::as_f64).unwrap_or(0.0),
                    );
                    max_abs_delta = max_abs_delta.max(
                        delta.pointer("/max_abs_delta").and_then(Value::as_f64).unwrap_or(0.0),
                    );
                    compared_count += 1;
                    status = "compared";
                }
            }
        }

        if status == "missing_input" {
            missing_input_count += 1;
        }
        rows.push(json!({
            "head": head,
            "kv_head": kv_head,
            "status": status,
            "probability_stage_present": probability.is_some(),
            "value_cache_stage_present": value_cache.is_some(),
            "value_mix_stage_present": true,
            "token_count": token_count,
            "value_dim": value_dim,
            "delta": delta,
            "recomputed_first_values": recomputed_first_values,
        }));
    }

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "policy": "Rust scalar value-mix recompute is diagnostic arithmetic evidence only; it compares Rust traced probability/cache inputs against the Rust trace-only F16-cache value-mix output and does not promote reference parity, A770 semantic quality, value mix residency, selected attention, resident KV, or any support claim",
        "probability_head_count": probability_heads.len(),
        "value_cache_head_count": value_cache_heads.len(),
        "value_mix_head_count": value_mix_heads.len(),
        "group_size": group_size,
        "compared_count": compared_count,
        "missing_input_count": missing_input_count,
        "max_abs_delta": max_abs_delta,
        "max_rms_delta": max_rms_delta,
        "all_compared": !value_mix_heads.is_empty() && missing_input_count == 0,
        "rows": rows,
    })
}

#[derive(Debug, Clone)]
struct ScalarTraceTensor {
    first_values: Vec<f32>,
    shape: Vec<usize>,
    num_elements: usize,
}

fn attention_value_mix_input_attribution(
    reference_records: &[ReferenceTraceRecord],
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    let reference_probability_heads = reference_records
        .iter()
        .filter_map(|record| {
            parse_stage_head(&record.stage, "kq_soft_max_ext_head")
                .and_then(|head| scalar_reference_tensor(record).map(|tensor| (head, tensor)))
        })
        .collect::<BTreeMap<_, _>>();
    let reference_value_cache_heads = reference_records
        .iter()
        .filter_map(|record| {
            parse_stage_head(&record.stage, "v_cache_rust_layout_head")
                .and_then(|head| scalar_reference_tensor(record).map(|tensor| (head, tensor)))
        })
        .collect::<BTreeMap<_, _>>();
    let reference_value_mix_heads = reference_records
        .iter()
        .filter_map(|record| {
            parse_stage_head(&record.stage, "kqv_head")
                .and_then(|head| scalar_reference_tensor(record).map(|tensor| (head, tensor)))
        })
        .collect::<BTreeMap<_, _>>();
    let rust_probability_heads = rust_records
        .iter()
        .filter_map(|(stage, record)| {
            parse_stage_head(stage, "attn_scores_softmax_head")
                .map(|head| (head, scalar_rust_tensor(record)))
        })
        .collect::<BTreeMap<_, _>>();
    let rust_value_cache_heads = rust_records
        .iter()
        .filter_map(|(stage, record)| {
            parse_stage_head(stage, "attention_v_cache_f16_roundtrip_kv_head")
                .map(|head| (head, scalar_rust_tensor(record)))
        })
        .collect::<BTreeMap<_, _>>();

    let group_size = if reference_value_cache_heads.is_empty()
        || reference_value_mix_heads.is_empty()
        || !reference_value_mix_heads.len().is_multiple_of(reference_value_cache_heads.len())
    {
        None
    } else {
        Some(reference_value_mix_heads.len() / reference_value_cache_heads.len())
    };

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "policy": "value-mix input attribution is diagnostic scalar evidence only; it compares mixed reference/Rust probability and V-cache inputs against the reference value-mix target and does not promote reference parity, A770 semantic quality, value mix residency, selected attention, resident KV, or any support claim",
        "group_size": group_size,
        "reference_probability_head_count": reference_probability_heads.len(),
        "reference_value_cache_head_count": reference_value_cache_heads.len(),
        "reference_value_mix_head_count": reference_value_mix_heads.len(),
        "rust_probability_head_count": rust_probability_heads.len(),
        "rust_value_cache_head_count": rust_value_cache_heads.len(),
        "reference_probability_rust_value_cache_vs_reference": attention_value_mix_input_attribution_section(
            &reference_probability_heads,
            &rust_value_cache_heads,
            &reference_value_mix_heads,
            group_size,
            "reference_probability",
            "rust_f16_roundtrip_value_cache",
        ),
        "rust_probability_reference_value_cache_vs_reference": attention_value_mix_input_attribution_section(
            &rust_probability_heads,
            &reference_value_cache_heads,
            &reference_value_mix_heads,
            group_size,
            "rust_probability",
            "reference_value_cache",
        ),
        "rust_probability_rust_value_cache_vs_reference": attention_value_mix_input_attribution_section(
            &rust_probability_heads,
            &rust_value_cache_heads,
            &reference_value_mix_heads,
            group_size,
            "rust_probability",
            "rust_f16_roundtrip_value_cache",
        ),
        "candidate_best_summary": attention_value_mix_input_candidate_best_summary(
            &reference_probability_heads,
            &rust_probability_heads,
            &reference_value_cache_heads,
            &rust_value_cache_heads,
            &reference_value_mix_heads,
            group_size,
        ),
    })
}

fn scalar_reference_tensor(record: &ReferenceTraceRecord) -> Option<ScalarTraceTensor> {
    if !record.values_available || record.first_values.is_empty() {
        return None;
    }
    Some(ScalarTraceTensor {
        first_values: record.first_values.clone(),
        shape: record
            .shape
            .iter()
            .map(|dim| usize::try_from(*dim).ok())
            .collect::<Option<Vec<_>>>()?,
        num_elements: usize::try_from(record.nelements).ok()?,
    })
}

fn scalar_rust_tensor(record: &RustTraceRecord) -> ScalarTraceTensor {
    ScalarTraceTensor {
        first_values: record.first_values.clone(),
        shape: record.shape.clone(),
        num_elements: record.num_elements,
    }
}

fn attention_value_mix_input_attribution_section(
    probability_heads: &BTreeMap<usize, ScalarTraceTensor>,
    value_cache_heads: &BTreeMap<usize, ScalarTraceTensor>,
    reference_value_mix_heads: &BTreeMap<usize, ScalarTraceTensor>,
    group_size: Option<usize>,
    probability_source: &str,
    value_cache_source: &str,
) -> Value {
    let mut rows = Vec::new();
    let mut compared_count = 0usize;
    let mut missing_input_count = 0usize;
    let mut max_rms_delta = 0.0f64;
    let mut max_abs_delta = 0.0f64;

    for (&head, value_mix) in reference_value_mix_heads {
        let kv_head = group_size.map(|group_size| head / group_size);
        let probability = probability_heads.get(&head);
        let value_cache = kv_head.and_then(|kv_head| value_cache_heads.get(&kv_head));
        let mut status = "missing_input";
        let mut delta = Value::Null;
        let mut recomputed_first_values = Vec::<f32>::new();
        let mut token_count = None::<usize>;
        let value_dim = value_mix.num_elements;

        if let (Some(probability), Some(value_cache)) = (probability, value_cache) {
            token_count = value_cache.shape.get(1).copied().or_else(|| {
                if value_dim == 0 { None } else { Some(value_cache.num_elements / value_dim) }
            });
            if let Some(token_count) = token_count {
                let value_sample_count = value_dim.saturating_mul(token_count);
                if probability.first_values.len() >= token_count
                    && value_cache.first_values.len() >= value_sample_count
                    && value_mix.first_values.len() >= value_dim
                {
                    for dim in 0..value_dim {
                        let mut sum = 0.0f64;
                        for token in 0..token_count {
                            let probability = probability.first_values[token] as f64;
                            let value = value_cache.first_values[dim * token_count + token] as f64;
                            sum += probability * value;
                        }
                        recomputed_first_values.push(sum as f32);
                    }
                    delta = compare_prefix(
                        &recomputed_first_values,
                        &value_mix.first_values,
                        value_dim,
                    );
                    max_rms_delta = max_rms_delta.max(
                        delta.pointer("/rms_abs_delta").and_then(Value::as_f64).unwrap_or(0.0),
                    );
                    max_abs_delta = max_abs_delta.max(
                        delta.pointer("/max_abs_delta").and_then(Value::as_f64).unwrap_or(0.0),
                    );
                    compared_count += 1;
                    status = "compared";
                }
            }
        }

        if status == "missing_input" {
            missing_input_count += 1;
        }
        rows.push(json!({
            "head": head,
            "kv_head": kv_head,
            "status": status,
            "probability_stage_present": probability.is_some(),
            "value_cache_stage_present": value_cache.is_some(),
            "reference_value_mix_stage_present": true,
            "probability_source": probability_source,
            "value_cache_source": value_cache_source,
            "target_source": "reference_value_mix",
            "token_count": token_count,
            "value_dim": value_dim,
            "delta": delta,
            "recomputed_first_values": recomputed_first_values,
        }));
    }

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "probability_source": probability_source,
        "value_cache_source": value_cache_source,
        "target_source": "reference_value_mix",
        "compared_count": compared_count,
        "missing_input_count": missing_input_count,
        "max_abs_delta": max_abs_delta,
        "max_rms_delta": max_rms_delta,
        "all_compared": !reference_value_mix_heads.is_empty() && missing_input_count == 0,
        "rows": rows,
    })
}

#[derive(Debug, Clone, Copy)]
struct ValueMixInputCandidate<'a> {
    id: &'static str,
    probability_source: &'static str,
    value_cache_source: &'static str,
    probability_heads: &'a BTreeMap<usize, ScalarTraceTensor>,
    value_cache_heads: &'a BTreeMap<usize, ScalarTraceTensor>,
}

fn value_mix_candidate_delta(
    probability: &ScalarTraceTensor,
    value_cache: &ScalarTraceTensor,
    target: &ScalarTraceTensor,
) -> Option<Value> {
    let value_dim = target.num_elements;
    if value_dim == 0 {
        return None;
    }
    let token_count =
        value_cache.shape.get(1).copied().or_else(|| Some(value_cache.num_elements / value_dim))?;
    let value_sample_count = value_dim.checked_mul(token_count)?;
    if token_count == 0
        || probability.first_values.len() < token_count
        || value_cache.first_values.len() < value_sample_count
        || target.first_values.len() < value_dim
    {
        return None;
    }

    let mut recomputed = Vec::<f32>::with_capacity(value_dim);
    for dim in 0..value_dim {
        let mut sum = 0.0f64;
        for token in 0..token_count {
            let probability = probability.first_values[token] as f64;
            let value = value_cache.first_values[dim * token_count + token] as f64;
            sum += probability * value;
        }
        recomputed.push(sum as f32);
    }
    Some(compare_prefix(&recomputed, &target.first_values, value_dim))
}

fn value_mix_candidate_rank(delta: &Value) -> (f64, f64, bool) {
    (
        delta_metric(delta, "/rms_abs_delta"),
        delta_metric(delta, "/max_abs_delta"),
        !delta.pointer("/count_match").and_then(Value::as_bool).unwrap_or(false),
    )
}

fn attention_value_mix_input_candidate_best_summary(
    reference_probability_heads: &BTreeMap<usize, ScalarTraceTensor>,
    rust_probability_heads: &BTreeMap<usize, ScalarTraceTensor>,
    reference_value_cache_heads: &BTreeMap<usize, ScalarTraceTensor>,
    rust_value_cache_heads: &BTreeMap<usize, ScalarTraceTensor>,
    reference_value_mix_heads: &BTreeMap<usize, ScalarTraceTensor>,
    group_size: Option<usize>,
) -> Value {
    let candidates = [
        ValueMixInputCandidate {
            id: "reference_probability_reference_value_cache",
            probability_source: "reference_probability",
            value_cache_source: "reference_value_cache",
            probability_heads: reference_probability_heads,
            value_cache_heads: reference_value_cache_heads,
        },
        ValueMixInputCandidate {
            id: "reference_probability_rust_value_cache",
            probability_source: "reference_probability",
            value_cache_source: "rust_f16_roundtrip_value_cache",
            probability_heads: reference_probability_heads,
            value_cache_heads: rust_value_cache_heads,
        },
        ValueMixInputCandidate {
            id: "rust_probability_reference_value_cache",
            probability_source: "rust_probability",
            value_cache_source: "reference_value_cache",
            probability_heads: rust_probability_heads,
            value_cache_heads: reference_value_cache_heads,
        },
        ValueMixInputCandidate {
            id: "rust_probability_rust_value_cache",
            probability_source: "rust_probability",
            value_cache_source: "rust_f16_roundtrip_value_cache",
            probability_heads: rust_probability_heads,
            value_cache_heads: rust_value_cache_heads,
        },
    ];

    let mut rows = Vec::new();
    let mut compared_count = 0usize;
    let mut missing_input_count = 0usize;
    let mut best_candidate_counts = BTreeMap::<String, usize>::new();
    let mut max_best_abs_delta = 0.0f64;
    let mut max_best_rms_delta = 0.0f64;

    for (&head, target) in reference_value_mix_heads {
        let kv_head = group_size.map(|group_size| head / group_size);
        let mut candidate_rows = Vec::new();
        let mut best_id = None::<&str>;
        let mut best_delta = Value::Null;
        let mut best_rank = (f64::INFINITY, f64::INFINITY, true);

        for candidate in candidates {
            let probability = candidate.probability_heads.get(&head);
            let value_cache = kv_head.and_then(|kv_head| candidate.value_cache_heads.get(&kv_head));
            if let (Some(probability), Some(value_cache)) = (probability, value_cache)
                && let Some(delta) = value_mix_candidate_delta(probability, value_cache, target)
            {
                let rank = value_mix_candidate_rank(&delta);
                if rank < best_rank {
                    best_rank = rank;
                    best_id = Some(candidate.id);
                    best_delta = delta.clone();
                }
                candidate_rows.push(json!({
                    "candidate": candidate.id,
                    "probability_source": candidate.probability_source,
                    "value_cache_source": candidate.value_cache_source,
                    "max_abs_delta": delta_metric(&delta, "/max_abs_delta"),
                    "rms_delta": delta_metric(&delta, "/rms_abs_delta"),
                    "delta": delta,
                }));
            }
        }

        if let Some(best_id) = best_id {
            compared_count += 1;
            *best_candidate_counts.entry(best_id.to_string()).or_insert(0) += 1;
            max_best_abs_delta =
                max_best_abs_delta.max(delta_metric(&best_delta, "/max_abs_delta"));
            max_best_rms_delta =
                max_best_rms_delta.max(delta_metric(&best_delta, "/rms_abs_delta"));
            rows.push(json!({
                "head": head,
                "kv_head": kv_head,
                "status": "compared",
                "best_candidate": best_id,
                "max_abs_delta": delta_metric(&best_delta, "/max_abs_delta"),
                "rms_delta": delta_metric(&best_delta, "/rms_abs_delta"),
                "candidate_count": candidate_rows.len(),
                "candidates": candidate_rows,
            }));
        } else {
            missing_input_count += 1;
            rows.push(json!({
                "head": head,
                "kv_head": kv_head,
                "status": "missing_input",
                "candidate_count": 0,
            }));
        }
    }

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "policy": "value-mix input candidate ranking is diagnostic-only evidence for whether reference value-mix residuals follow probability inputs, value-cache inputs, or both; it does not promote reference parity, A770 semantic quality, value mix residency, selected attention, resident KV, or any support claim",
        "compared_count": compared_count,
        "missing_input_count": missing_input_count,
        "best_candidate_counts": best_candidate_counts,
        "max_best_abs_delta": max_best_abs_delta,
        "max_best_rms_delta": max_best_rms_delta,
        "all_compared": !reference_value_mix_heads.is_empty() && missing_input_count == 0,
        "rows": rows,
    })
}

fn attention_query_rope_ref_layout_delta(
    reference_records: &[ReferenceTraceRecord],
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    let reference = reference_rope_query_record(reference_records);
    let rust = rust_records.get("attention_q_rope");
    let delta = reference
        .zip(rust)
        .map(|(reference, rust)| compare_vectors(&reference.first_values, &rust.first_values));

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "policy": "post-RoPE query ref-layout delta is diagnostic score-input evidence only; it does not promote reference parity, A770 semantic quality, attention score residency, selected attention, resident KV, or any support claim",
        "reference_stage": "Qcur",
        "reference_layout": "reference head-major score input selected from Qcur record with head_dim and head_count axes",
        "rust_stage": "attention_q_rope",
        "reference_present": reference.is_some(),
        "rust_present": rust.is_some(),
        "reference_shape": reference.map(|record| record.shape.clone()),
        "rust_shape": rust.map(|record| record.shape.clone()),
        "delta": delta,
    })
}

fn attention_score_reference_scalar_recompute(reference_records: &[ReferenceTraceRecord]) -> Value {
    let query = reference_rope_query_record(reference_records);
    let key_heads = reference_records
        .iter()
        .filter_map(|record| {
            parse_stage_head(&record.stage, "k_kv_head").map(|head| (head, record))
        })
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let score_heads = reference_records
        .iter()
        .filter_map(|record| parse_stage_head(&record.stage, "kq_head").map(|head| (head, record)))
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();

    let (head_dim, head_count) = query.and_then(reference_query_head_dim_count).unwrap_or((0, 0));
    let group_size = if head_count > 0 && !key_heads.is_empty() && head_count % key_heads.len() == 0
    {
        Some(head_count / key_heads.len())
    } else {
        None
    };

    let mut rows = Vec::new();
    let mut compared_count = 0usize;
    let mut missing_input_count = 0usize;
    let mut max_abs_delta = 0.0f64;
    let mut max_rms_delta = 0.0f64;

    for (&head, score) in &score_heads {
        let kv_head = group_size.map(|group_size| head / group_size);
        let key = kv_head.and_then(|kv_head| key_heads.get(&kv_head).copied());
        if let (Some(query), Some(key), Some(kv_head)) = (query, key, kv_head) {
            let token_count = reference_key_token_count(key)
                .unwrap_or(score.first_values.len())
                .min(score.first_values.len());
            if let Some(recomputed) =
                reference_score_row_from_query_key(query, key, head, head_dim, token_count)
            {
                let delta = compare_vectors(&recomputed, &score.first_values);
                max_abs_delta = max_abs_delta
                    .max(delta.pointer("/max_abs_delta").and_then(Value::as_f64).unwrap_or(0.0));
                max_rms_delta = max_rms_delta
                    .max(delta.pointer("/rms_abs_delta").and_then(Value::as_f64).unwrap_or(0.0));
                compared_count += 1;
                rows.push(json!({
                    "head": head,
                    "kv_head": kv_head,
                    "status": "compared",
                    "query_stage_present": true,
                    "key_stage_present": true,
                    "score_stage_present": true,
                    "live_token_count": token_count,
                    "recomputed_first_values": recomputed,
                    "delta": delta,
                }));
            } else {
                missing_input_count += 1;
                rows.push(json!({
                    "head": head,
                    "kv_head": kv_head,
                    "status": "missing_input",
                    "query_stage_present": true,
                    "key_stage_present": true,
                    "score_stage_present": true,
                }));
            }
        } else {
            missing_input_count += 1;
            rows.push(json!({
                "head": head,
                "kv_head": kv_head,
                "status": "missing_input",
                "query_stage_present": query.is_some(),
                "key_stage_present": key.is_some(),
                "score_stage_present": true,
            }));
        }
    }

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "policy": "reference scalar score recompute is diagnostic arithmetic evidence only; it recomputes sampled raw attention rows from reference Qcur and reference K-cache samples and does not promote reference parity, A770 semantic quality, attention score residency, selected attention, resident KV, or any support claim",
        "query_stage_present": query.is_some(),
        "query_stage": query.map(|record| record.stage.clone()),
        "key_head_count": key_heads.len(),
        "score_head_count": score_heads.len(),
        "head_dim": if head_dim == 0 { Value::Null } else { json!(head_dim) },
        "head_count": if head_count == 0 { Value::Null } else { json!(head_count) },
        "group_size": group_size,
        "compared_count": compared_count,
        "missing_input_count": missing_input_count,
        "max_abs_delta": max_abs_delta,
        "max_rms_delta": max_rms_delta,
        "all_compared": !score_heads.is_empty() && missing_input_count == 0,
        "rows": rows,
    })
}

#[derive(Debug, Clone, Copy)]
enum ReferenceScoreScalePolicy {
    Unscaled,
    HeadDimSqrtRecip,
    LlmBuildKqScale,
}

#[derive(Debug, Clone, Copy)]
enum ReferenceScoreLengthPolicy {
    LiveKeyCountOnly,
    PaddedTailZeroed,
    MaskApplied,
}

#[derive(Debug, Clone, Copy)]
struct ReferenceScoreVariantSpec {
    id: &'static str,
    scale_policy: ReferenceScoreScalePolicy,
    length_policy: ReferenceScoreLengthPolicy,
}

impl ReferenceScoreScalePolicy {
    fn label(self) -> &'static str {
        match self {
            ReferenceScoreScalePolicy::Unscaled => "unscaled",
            ReferenceScoreScalePolicy::HeadDimSqrtRecip => "1_sqrt_head_dim",
            ReferenceScoreScalePolicy::LlmBuildKqScale => "llm_build_kq_scale",
        }
    }

    fn source(self) -> &'static str {
        match self {
            ReferenceScoreScalePolicy::Unscaled => "raw_q_dot_k",
            ReferenceScoreScalePolicy::HeadDimSqrtRecip => "1.0/sqrt(head_dim)",
            ReferenceScoreScalePolicy::LlmBuildKqScale => {
                "llama.cpp llm_build_kqv fallback formula: hparams.f_attention_scale or 1.0/sqrt(head_dim)"
            }
        }
    }

    fn scale(self, head_dim: usize) -> f64 {
        match self {
            ReferenceScoreScalePolicy::Unscaled => 1.0,
            ReferenceScoreScalePolicy::HeadDimSqrtRecip
            | ReferenceScoreScalePolicy::LlmBuildKqScale => {
                if head_dim == 0 {
                    1.0
                } else {
                    1.0 / (head_dim as f64).sqrt()
                }
            }
        }
    }
}

impl ReferenceScoreLengthPolicy {
    fn label(self) -> &'static str {
        match self {
            ReferenceScoreLengthPolicy::LiveKeyCountOnly => "live_key_count_only",
            ReferenceScoreLengthPolicy::PaddedTailZeroed => "padded_tail_zeroed",
            ReferenceScoreLengthPolicy::MaskApplied => "mask_applied",
        }
    }

    fn mask_policy(self) -> &'static str {
        match self {
            ReferenceScoreLengthPolicy::LiveKeyCountOnly => "none_live_prefix_only",
            ReferenceScoreLengthPolicy::PaddedTailZeroed => {
                "padded_tail_zeroed_to_score_sample_len"
            }
            ReferenceScoreLengthPolicy::MaskApplied => {
                "causal_or_padding_tail_masked_to_large_negative_sentinel"
            }
        }
    }
}

fn reference_score_variant_specs() -> [ReferenceScoreVariantSpec; 6] {
    [
        ReferenceScoreVariantSpec {
            id: "reference_score_recompute_unscaled",
            scale_policy: ReferenceScoreScalePolicy::Unscaled,
            length_policy: ReferenceScoreLengthPolicy::LiveKeyCountOnly,
        },
        ReferenceScoreVariantSpec {
            id: "reference_score_recompute_scaled_by_1_sqrt_head_dim",
            scale_policy: ReferenceScoreScalePolicy::HeadDimSqrtRecip,
            length_policy: ReferenceScoreLengthPolicy::LiveKeyCountOnly,
        },
        ReferenceScoreVariantSpec {
            id: "reference_score_recompute_scaled_by_llm_build_kv_scale",
            scale_policy: ReferenceScoreScalePolicy::LlmBuildKqScale,
            length_policy: ReferenceScoreLengthPolicy::LiveKeyCountOnly,
        },
        ReferenceScoreVariantSpec {
            id: "reference_score_recompute_with_live_key_count_only",
            scale_policy: ReferenceScoreScalePolicy::Unscaled,
            length_policy: ReferenceScoreLengthPolicy::LiveKeyCountOnly,
        },
        ReferenceScoreVariantSpec {
            id: "reference_score_recompute_with_padded_tail_zeroed",
            scale_policy: ReferenceScoreScalePolicy::Unscaled,
            length_policy: ReferenceScoreLengthPolicy::PaddedTailZeroed,
        },
        ReferenceScoreVariantSpec {
            id: "reference_score_recompute_with_mask_applied",
            scale_policy: ReferenceScoreScalePolicy::Unscaled,
            length_policy: ReferenceScoreLengthPolicy::MaskApplied,
        },
    ]
}

fn reference_score_variant_values(
    live_scores: &[f32],
    live_token_count: usize,
    target_token_count: usize,
    scale: f64,
    length_policy: ReferenceScoreLengthPolicy,
) -> Vec<f32> {
    const MASK_SENTINEL: f32 = -1.0e30;

    let live_len = live_scores.len().min(live_token_count);
    match length_policy {
        ReferenceScoreLengthPolicy::LiveKeyCountOnly => {
            live_scores.iter().take(live_len).map(|value| (*value as f64 * scale) as f32).collect()
        }
        ReferenceScoreLengthPolicy::PaddedTailZeroed | ReferenceScoreLengthPolicy::MaskApplied => {
            let mut values = Vec::with_capacity(target_token_count);
            for token in 0..target_token_count {
                if token < live_len {
                    values.push((live_scores[token] as f64 * scale) as f32);
                } else if matches!(length_policy, ReferenceScoreLengthPolicy::MaskApplied) {
                    values.push(MASK_SENTINEL);
                } else {
                    values.push(0.0);
                }
            }
            values
        }
    }
}

fn delta_metric(delta: &Value, key: &str) -> f64 {
    delta.pointer(key).and_then(Value::as_f64).unwrap_or(f64::INFINITY)
}

fn reference_score_variant_explained(delta: &Value) -> bool {
    const ABS_THRESHOLD: f64 = 1.0e-4;
    const RMS_THRESHOLD: f64 = 1.0e-4;

    delta.pointer("/count_match").and_then(Value::as_bool).unwrap_or(false)
        && delta_metric(delta, "/max_abs_delta") <= ABS_THRESHOLD
        && delta_metric(delta, "/rms_abs_delta") <= RMS_THRESHOLD
}

fn attention_score_reference_semantic_variants(
    reference_records: &[ReferenceTraceRecord],
) -> Value {
    let query = reference_rope_query_record(reference_records);
    let key_heads = reference_records
        .iter()
        .filter_map(|record| {
            parse_stage_head(&record.stage, "k_kv_head").map(|head| (head, record))
        })
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let score_heads = reference_records
        .iter()
        .filter_map(|record| parse_stage_head(&record.stage, "kq_head").map(|head| (head, record)))
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let variants = reference_score_variant_specs();

    let (head_dim, head_count) = query.and_then(reference_query_head_dim_count).unwrap_or((0, 0));
    let group_size = if head_count > 0 && !key_heads.is_empty() && head_count % key_heads.len() == 0
    {
        Some(head_count / key_heads.len())
    } else {
        None
    };

    let mut rows = Vec::new();
    let mut compared_count = 0usize;
    let mut missing_input_count = 0usize;
    let mut unexplained_head_count = 0usize;
    let mut best_variant_counts = BTreeMap::<String, usize>::new();
    let mut max_best_abs_delta = 0.0f64;
    let mut max_best_rms_delta = 0.0f64;

    for (&head, score) in &score_heads {
        let kv_head = group_size.map(|group_size| head / group_size);
        let key = kv_head.and_then(|kv_head| key_heads.get(&kv_head).copied());
        if let (Some(query), Some(key), Some(kv_head)) = (query, key, kv_head) {
            let live_token_count = reference_key_token_count(key)
                .unwrap_or(score.first_values.len())
                .min(score.first_values.len());
            let target_token_count = score.first_values.len();
            let padded_token_count = target_token_count.saturating_sub(live_token_count);
            if let Some(live_scores) =
                reference_score_row_from_query_key(query, key, head, head_dim, live_token_count)
            {
                let mut variant_rows = Vec::new();
                let mut best_variant_id = "";
                let mut best_scale = 1.0f64;
                let mut best_mask_policy = "";
                let mut best_delta = Value::Null;
                let mut best_rank = (f64::INFINITY, f64::INFINITY, true);

                for variant in variants {
                    let scale = variant.scale_policy.scale(head_dim);
                    let values = reference_score_variant_values(
                        &live_scores,
                        live_token_count,
                        target_token_count,
                        scale,
                        variant.length_policy,
                    );
                    let delta = compare_vectors(&values, &score.first_values);
                    let rms = delta_metric(&delta, "/rms_abs_delta");
                    let max_abs = delta_metric(&delta, "/max_abs_delta");
                    let count_mismatch =
                        !delta.pointer("/count_match").and_then(Value::as_bool).unwrap_or(false);
                    let rank = (rms, max_abs, count_mismatch);
                    if rank < best_rank {
                        best_rank = rank;
                        best_variant_id = variant.id;
                        best_scale = scale;
                        best_mask_policy = variant.length_policy.mask_policy();
                        best_delta = delta.clone();
                    }
                    variant_rows.push(json!({
                        "variant": variant.id,
                        "head": head,
                        "kv_head": kv_head,
                        "token_count": values.len(),
                        "live_token_count": live_token_count,
                        "padded_token_count": padded_token_count,
                        "scale": scale,
                        "scale_policy": variant.scale_policy.label(),
                        "scale_source": variant.scale_policy.source(),
                        "mask_policy": variant.length_policy.mask_policy(),
                        "length_policy": variant.length_policy.label(),
                        "max_abs_delta": max_abs,
                        "rms_delta": rms,
                        "delta": delta,
                    }));
                }

                let head_explained = reference_score_variant_explained(&best_delta);
                if !head_explained {
                    unexplained_head_count += 1;
                }
                max_best_abs_delta =
                    max_best_abs_delta.max(delta_metric(&best_delta, "/max_abs_delta"));
                max_best_rms_delta =
                    max_best_rms_delta.max(delta_metric(&best_delta, "/rms_abs_delta"));
                *best_variant_counts.entry(best_variant_id.to_string()).or_insert(0) += 1;
                compared_count += 1;
                rows.push(json!({
                    "head": head,
                    "kv_head": kv_head,
                    "status": "compared",
                    "token_count": target_token_count,
                    "live_token_count": live_token_count,
                    "padded_token_count": padded_token_count,
                    "scale": best_scale,
                    "mask_policy": best_mask_policy,
                    "max_abs_delta": delta_metric(&best_delta, "/max_abs_delta"),
                    "rms_delta": delta_metric(&best_delta, "/rms_abs_delta"),
                    "best_variant": best_variant_id,
                    "best_delta": best_delta,
                    "head_explained": head_explained,
                    "variants": variant_rows,
                }));
            } else {
                missing_input_count += 1;
                rows.push(json!({
                    "head": head,
                    "kv_head": kv_head,
                    "status": "missing_input",
                    "query_stage_present": true,
                    "key_stage_present": true,
                    "score_stage_present": true,
                }));
            }
        } else {
            missing_input_count += 1;
            rows.push(json!({
                "head": head,
                "kv_head": kv_head,
                "status": "missing_input",
                "query_stage_present": query.is_some(),
                "key_stage_present": key.is_some(),
                "score_stage_present": true,
            }));
        }
    }

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "policy": "reference score semantic variants are diagnostic-only scale, mask, live-token, and padded-tail probes for reproducing reference kq_head rows; they do not promote reference parity, A770 semantic quality, attention score residency, selected attention, resident KV, or any support claim",
        "query_stage_present": query.is_some(),
        "query_stage": query.map(|record| record.stage.clone()),
        "key_head_count": key_heads.len(),
        "score_head_count": score_heads.len(),
        "head_dim": if head_dim == 0 { Value::Null } else { json!(head_dim) },
        "head_count": if head_count == 0 { Value::Null } else { json!(head_count) },
        "group_size": group_size,
        "variant_count": variants.len(),
        "variants_tested": variants.iter().map(|variant| variant.id).collect::<Vec<_>>(),
        "explanation_abs_threshold": 1.0e-4,
        "explanation_rms_threshold": 1.0e-4,
        "compared_count": compared_count,
        "missing_input_count": missing_input_count,
        "unexplained_head_count": unexplained_head_count,
        "max_best_abs_delta": max_best_abs_delta,
        "max_best_rms_delta": max_best_rms_delta,
        "best_variant_counts": best_variant_counts,
        "all_heads_explained": !score_heads.is_empty()
            && missing_input_count == 0
            && unexplained_head_count == 0,
        "rows": rows,
    })
}

#[derive(Debug, Clone, Copy)]
enum ReferenceScoreAccumPolicy {
    F64,
    F32,
    F32MulAdd,
}

#[derive(Debug, Clone, Copy)]
struct ReferenceScoreNumericVariantSpec {
    id: &'static str,
    query_f16_roundtrip: bool,
    key_f16_roundtrip: bool,
    accum_policy: ReferenceScoreAccumPolicy,
}

impl ReferenceScoreAccumPolicy {
    fn label(self) -> &'static str {
        match self {
            ReferenceScoreAccumPolicy::F64 => "f64_sequential",
            ReferenceScoreAccumPolicy::F32 => "f32_sequential",
            ReferenceScoreAccumPolicy::F32MulAdd => "f32_mul_add",
        }
    }
}

fn reference_score_numeric_variant_specs() -> [ReferenceScoreNumericVariantSpec; 6] {
    [
        ReferenceScoreNumericVariantSpec {
            id: "reference_score_numeric_f64_accum_q_f32_k_f32",
            query_f16_roundtrip: false,
            key_f16_roundtrip: false,
            accum_policy: ReferenceScoreAccumPolicy::F64,
        },
        ReferenceScoreNumericVariantSpec {
            id: "reference_score_numeric_f32_accum_q_f32_k_f32",
            query_f16_roundtrip: false,
            key_f16_roundtrip: false,
            accum_policy: ReferenceScoreAccumPolicy::F32,
        },
        ReferenceScoreNumericVariantSpec {
            id: "reference_score_numeric_f32_mul_add_q_f32_k_f32",
            query_f16_roundtrip: false,
            key_f16_roundtrip: false,
            accum_policy: ReferenceScoreAccumPolicy::F32MulAdd,
        },
        ReferenceScoreNumericVariantSpec {
            id: "reference_score_numeric_f32_accum_q_f16_k_f32",
            query_f16_roundtrip: true,
            key_f16_roundtrip: false,
            accum_policy: ReferenceScoreAccumPolicy::F32,
        },
        ReferenceScoreNumericVariantSpec {
            id: "reference_score_numeric_f32_accum_q_f32_k_f16",
            query_f16_roundtrip: false,
            key_f16_roundtrip: true,
            accum_policy: ReferenceScoreAccumPolicy::F32,
        },
        ReferenceScoreNumericVariantSpec {
            id: "reference_score_numeric_f32_accum_q_f16_k_f16",
            query_f16_roundtrip: true,
            key_f16_roundtrip: true,
            accum_policy: ReferenceScoreAccumPolicy::F32,
        },
    ]
}

fn numeric_variant_value(value: f32, use_f16_roundtrip: bool) -> f32 {
    if use_f16_roundtrip { f16_roundtrip(value) } else { value }
}

fn reference_score_row_from_query_key_numeric(
    query: &ReferenceTraceRecord,
    key: &ReferenceTraceRecord,
    head: usize,
    head_dim: usize,
    token_count: usize,
    variant: ReferenceScoreNumericVariantSpec,
) -> Option<Vec<f32>> {
    if head_dim == 0 || token_count == 0 {
        return None;
    }
    let q_offset = head.checked_mul(head_dim)?;
    if query.first_values.len() < q_offset + head_dim {
        return None;
    }
    if key.first_values.len() < token_count.checked_mul(head_dim)? {
        return None;
    }

    let mut scores = Vec::with_capacity(token_count);
    for token in 0..token_count {
        let score = match variant.accum_policy {
            ReferenceScoreAccumPolicy::F64 => {
                let mut sum = 0.0f64;
                for dim in 0..head_dim {
                    let q = numeric_variant_value(
                        query.first_values[q_offset + dim],
                        variant.query_f16_roundtrip,
                    ) as f64;
                    let k = numeric_variant_value(
                        key.first_values[token * head_dim + dim],
                        variant.key_f16_roundtrip,
                    ) as f64;
                    sum += q * k;
                }
                sum as f32
            }
            ReferenceScoreAccumPolicy::F32 => {
                let mut sum = 0.0f32;
                for dim in 0..head_dim {
                    let q = numeric_variant_value(
                        query.first_values[q_offset + dim],
                        variant.query_f16_roundtrip,
                    );
                    let k = numeric_variant_value(
                        key.first_values[token * head_dim + dim],
                        variant.key_f16_roundtrip,
                    );
                    sum += q * k;
                }
                sum
            }
            ReferenceScoreAccumPolicy::F32MulAdd => {
                let mut sum = 0.0f32;
                for dim in 0..head_dim {
                    let q = numeric_variant_value(
                        query.first_values[q_offset + dim],
                        variant.query_f16_roundtrip,
                    );
                    let k = numeric_variant_value(
                        key.first_values[token * head_dim + dim],
                        variant.key_f16_roundtrip,
                    );
                    sum = q.mul_add(k, sum);
                }
                sum
            }
        };
        scores.push(score);
    }
    Some(scores)
}

fn attention_score_reference_numeric_variants(reference_records: &[ReferenceTraceRecord]) -> Value {
    let query = reference_rope_query_record(reference_records);
    let key_heads = reference_records
        .iter()
        .filter_map(|record| {
            parse_stage_head(&record.stage, "k_kv_head").map(|head| (head, record))
        })
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let score_heads = reference_records
        .iter()
        .filter_map(|record| parse_stage_head(&record.stage, "kq_head").map(|head| (head, record)))
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let variants = reference_score_numeric_variant_specs();

    let (head_dim, head_count) = query.and_then(reference_query_head_dim_count).unwrap_or((0, 0));
    let group_size = if head_count > 0 && !key_heads.is_empty() && head_count % key_heads.len() == 0
    {
        Some(head_count / key_heads.len())
    } else {
        None
    };

    let mut rows = Vec::new();
    let mut compared_count = 0usize;
    let mut missing_input_count = 0usize;
    let mut unexplained_head_count = 0usize;
    let mut best_variant_counts = BTreeMap::<String, usize>::new();
    let mut max_best_abs_delta = 0.0f64;
    let mut max_best_rms_delta = 0.0f64;

    for (&head, score) in &score_heads {
        let kv_head = group_size.map(|group_size| head / group_size);
        let key = kv_head.and_then(|kv_head| key_heads.get(&kv_head).copied());
        if let (Some(query), Some(key), Some(kv_head)) = (query, key, kv_head) {
            let live_token_count = reference_key_token_count(key)
                .unwrap_or(score.first_values.len())
                .min(score.first_values.len());
            let target_token_count = score.first_values.len();
            let padded_token_count = target_token_count.saturating_sub(live_token_count);
            let mut variant_rows = Vec::new();
            let mut best_variant_id = "";
            let mut best_delta = Value::Null;
            let mut best_rank = (f64::INFINITY, f64::INFINITY, true);
            let mut best_accum_policy = "";
            let mut best_query_f16_roundtrip = false;
            let mut best_key_f16_roundtrip = false;

            for variant in variants {
                if let Some(live_scores) = reference_score_row_from_query_key_numeric(
                    query,
                    key,
                    head,
                    head_dim,
                    live_token_count,
                    variant,
                ) {
                    let values = reference_score_variant_values(
                        &live_scores,
                        live_token_count,
                        target_token_count,
                        1.0,
                        ReferenceScoreLengthPolicy::PaddedTailZeroed,
                    );
                    let delta = compare_vectors(&values, &score.first_values);
                    let rms = delta_metric(&delta, "/rms_abs_delta");
                    let max_abs = delta_metric(&delta, "/max_abs_delta");
                    let count_mismatch =
                        !delta.pointer("/count_match").and_then(Value::as_bool).unwrap_or(false);
                    let rank = (rms, max_abs, count_mismatch);
                    if rank < best_rank {
                        best_rank = rank;
                        best_variant_id = variant.id;
                        best_delta = delta.clone();
                        best_accum_policy = variant.accum_policy.label();
                        best_query_f16_roundtrip = variant.query_f16_roundtrip;
                        best_key_f16_roundtrip = variant.key_f16_roundtrip;
                    }
                    variant_rows.push(json!({
                        "variant": variant.id,
                        "head": head,
                        "kv_head": kv_head,
                        "token_count": values.len(),
                        "live_token_count": live_token_count,
                        "padded_token_count": padded_token_count,
                        "accum_policy": variant.accum_policy.label(),
                        "query_f16_roundtrip": variant.query_f16_roundtrip,
                        "key_f16_roundtrip": variant.key_f16_roundtrip,
                        "length_policy": "padded_tail_zeroed",
                        "max_abs_delta": max_abs,
                        "rms_delta": rms,
                        "delta": delta,
                    }));
                }
            }

            if variant_rows.is_empty() {
                missing_input_count += 1;
                rows.push(json!({
                    "head": head,
                    "kv_head": kv_head,
                    "status": "missing_input",
                    "query_stage_present": true,
                    "key_stage_present": true,
                    "score_stage_present": true,
                }));
                continue;
            }

            let head_explained = reference_score_variant_explained(&best_delta);
            if !head_explained {
                unexplained_head_count += 1;
            }
            max_best_abs_delta =
                max_best_abs_delta.max(delta_metric(&best_delta, "/max_abs_delta"));
            max_best_rms_delta =
                max_best_rms_delta.max(delta_metric(&best_delta, "/rms_abs_delta"));
            *best_variant_counts.entry(best_variant_id.to_string()).or_insert(0) += 1;
            compared_count += 1;
            rows.push(json!({
                "head": head,
                "kv_head": kv_head,
                "status": "compared",
                "token_count": target_token_count,
                "live_token_count": live_token_count,
                "padded_token_count": padded_token_count,
                "best_variant": best_variant_id,
                "accum_policy": best_accum_policy,
                "query_f16_roundtrip": best_query_f16_roundtrip,
                "key_f16_roundtrip": best_key_f16_roundtrip,
                "max_abs_delta": delta_metric(&best_delta, "/max_abs_delta"),
                "rms_delta": delta_metric(&best_delta, "/rms_abs_delta"),
                "best_delta": best_delta,
                "head_explained": head_explained,
                "variants": variant_rows,
            }));
        } else {
            missing_input_count += 1;
            rows.push(json!({
                "head": head,
                "kv_head": kv_head,
                "status": "missing_input",
                "query_stage_present": query.is_some(),
                "key_stage_present": key.is_some(),
                "score_stage_present": true,
            }));
        }
    }

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "policy": "reference score numeric variants are diagnostic-only probes for GGML-like score dot numeric behavior; they keep the padded-tail-zeroed row shape fixed and vary Q/K F16 roundtrip plus f32/f64 accumulation without promoting reference parity, A770 semantic quality, attention score residency, selected attention, resident KV, or any support claim",
        "query_stage_present": query.is_some(),
        "query_stage": query.map(|record| record.stage.clone()),
        "key_head_count": key_heads.len(),
        "score_head_count": score_heads.len(),
        "head_dim": if head_dim == 0 { Value::Null } else { json!(head_dim) },
        "head_count": if head_count == 0 { Value::Null } else { json!(head_count) },
        "group_size": group_size,
        "variant_count": variants.len(),
        "variants_tested": variants.iter().map(|variant| variant.id).collect::<Vec<_>>(),
        "length_policy": "padded_tail_zeroed",
        "explanation_abs_threshold": 1.0e-4,
        "explanation_rms_threshold": 1.0e-4,
        "compared_count": compared_count,
        "missing_input_count": missing_input_count,
        "unexplained_head_count": unexplained_head_count,
        "max_best_abs_delta": max_best_abs_delta,
        "max_best_rms_delta": max_best_rms_delta,
        "best_variant_counts": best_variant_counts,
        "all_heads_explained": !score_heads.is_empty()
            && missing_input_count == 0
            && unexplained_head_count == 0,
        "rows": rows,
    })
}

#[derive(Debug, Clone, Copy)]
enum ScoreKeyLayout {
    TokenMajor,
    DimMajor,
}

impl ScoreKeyLayout {
    fn label(self) -> &'static str {
        match self {
            ScoreKeyLayout::TokenMajor => "token_major",
            ScoreKeyLayout::DimMajor => "dim_major",
        }
    }
}

fn score_row_from_query_key_values(
    query_values: &[f32],
    key_values: &[f32],
    head: usize,
    head_dim: usize,
    token_count: usize,
    key_layout: ScoreKeyLayout,
    query_f16_roundtrip: bool,
    key_f16_roundtrip: bool,
    accum_policy: ReferenceScoreAccumPolicy,
) -> Option<Vec<f32>> {
    if head_dim == 0 || token_count == 0 {
        return None;
    }
    let q_offset = head.checked_mul(head_dim)?;
    if query_values.len() < q_offset + head_dim {
        return None;
    }
    if key_values.len() < head_dim.checked_mul(token_count)? {
        return None;
    }

    let key_index = |token: usize, dim: usize| match key_layout {
        ScoreKeyLayout::TokenMajor => token * head_dim + dim,
        ScoreKeyLayout::DimMajor => dim * token_count + token,
    };

    let mut scores = Vec::with_capacity(token_count);
    for token in 0..token_count {
        let score = match accum_policy {
            ReferenceScoreAccumPolicy::F64 => {
                let mut sum = 0.0f64;
                for dim in 0..head_dim {
                    let q = numeric_variant_value(query_values[q_offset + dim], query_f16_roundtrip)
                        as f64;
                    let k =
                        numeric_variant_value(key_values[key_index(token, dim)], key_f16_roundtrip)
                            as f64;
                    sum += q * k;
                }
                sum as f32
            }
            ReferenceScoreAccumPolicy::F32 => {
                let mut sum = 0.0f32;
                for dim in 0..head_dim {
                    let q =
                        numeric_variant_value(query_values[q_offset + dim], query_f16_roundtrip);
                    let k =
                        numeric_variant_value(key_values[key_index(token, dim)], key_f16_roundtrip);
                    sum += q * k;
                }
                sum
            }
            ReferenceScoreAccumPolicy::F32MulAdd => {
                let mut sum = 0.0f32;
                for dim in 0..head_dim {
                    let q =
                        numeric_variant_value(query_values[q_offset + dim], query_f16_roundtrip);
                    let k =
                        numeric_variant_value(key_values[key_index(token, dim)], key_f16_roundtrip);
                    sum = q.mul_add(k, sum);
                }
                sum
            }
        };
        scores.push(score);
    }
    Some(scores)
}

#[derive(Debug, Clone, Copy)]
struct ScoreInputAttributionCandidate<'a> {
    id: &'static str,
    query_source: &'static str,
    key_source: &'static str,
    key_layout: ScoreKeyLayout,
    query_values: &'a [f32],
    key_values: &'a [f32],
}

fn score_input_candidate_delta(
    candidate: ScoreInputAttributionCandidate<'_>,
    head: usize,
    head_dim: usize,
    live_token_count: usize,
    target: &[f32],
    length_policy: ReferenceScoreLengthPolicy,
) -> Option<Value> {
    let live_scores = score_row_from_query_key_values(
        candidate.query_values,
        candidate.key_values,
        head,
        head_dim,
        live_token_count,
        candidate.key_layout,
        true,
        false,
        ReferenceScoreAccumPolicy::F32,
    )?;
    let values = reference_score_variant_values(
        &live_scores,
        live_token_count,
        target.len(),
        1.0,
        length_policy,
    );
    Some(compare_vectors(&values, target))
}

fn score_input_best_candidate<'a>(
    candidate_deltas: impl Iterator<Item = (&'a str, &'a Value)>,
) -> Option<&'a str> {
    candidate_deltas
        .min_by(|(_, left), (_, right)| {
            let left_rank = (
                delta_metric(left, "/rms_abs_delta"),
                delta_metric(left, "/max_abs_delta"),
                !left.pointer("/count_match").and_then(Value::as_bool).unwrap_or(false),
            );
            let right_rank = (
                delta_metric(right, "/rms_abs_delta"),
                delta_metric(right, "/max_abs_delta"),
                !right.pointer("/count_match").and_then(Value::as_bool).unwrap_or(false),
            );
            left_rank.partial_cmp(&right_rank).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(id, _)| id)
}

fn attention_score_input_attribution(
    reference_records: &[ReferenceTraceRecord],
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    let reference_query = reference_rope_query_record(reference_records);
    let rust_query = rust_records.get("attention_q_rope");
    let reference_key_heads = reference_records
        .iter()
        .filter_map(|record| {
            parse_stage_head(&record.stage, "k_kv_head").map(|head| (head, record))
        })
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let rust_score_key_heads = rust_records
        .iter()
        .filter_map(|(stage, record)| {
            parse_stage_head(stage, "attention_k_score_input_head").map(|head| (head, record))
        })
        .filter(|(_, record)| !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let rust_fallback_key_heads = rust_records
        .iter()
        .filter_map(|(stage, record)| {
            parse_stage_head(stage, "attention_k_cache_f16_roundtrip_kv_head")
                .map(|head| (head, record))
        })
        .filter(|(_, record)| !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let reference_score_heads = reference_records
        .iter()
        .filter_map(|record| parse_stage_head(&record.stage, "kq_head").map(|head| (head, record)))
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let rust_score_heads = rust_records
        .iter()
        .filter_map(|(stage, record)| {
            parse_stage_head(stage, "attention_scores_raw_head").map(|head| (head, record))
        })
        .filter(|(_, record)| !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();

    let (reference_head_dim, reference_head_count) =
        reference_query.and_then(reference_query_head_dim_count).unwrap_or((0, 0));
    let (rust_head_dim, rust_head_count) =
        rust_query.and_then(rust_query_head_dim_count).unwrap_or((0, 0));
    let head_dim = if reference_head_dim != 0 { reference_head_dim } else { rust_head_dim };
    let head_count = if reference_head_count != 0 { reference_head_count } else { rust_head_count };
    let group_size = if head_count > 0
        && !reference_key_heads.is_empty()
        && head_count % reference_key_heads.len() == 0
    {
        Some(head_count / reference_key_heads.len())
    } else {
        None
    };

    let mut rows = Vec::new();
    let mut compared_count = 0usize;
    let mut missing_input_count = 0usize;
    let mut reference_best_counts = BTreeMap::<String, usize>::new();
    let mut rust_best_counts = BTreeMap::<String, usize>::new();
    let mut max_reference_best_abs_delta = 0.0f64;
    let mut max_reference_best_rms_delta = 0.0f64;
    let mut max_rust_best_abs_delta = 0.0f64;
    let mut max_rust_best_rms_delta = 0.0f64;

    for (&head, reference_score) in &reference_score_heads {
        let kv_head = group_size.map(|group_size| head / group_size);
        let rust_score = rust_score_heads.get(&head).copied();
        let reference_key = kv_head.and_then(|kv_head| reference_key_heads.get(&kv_head).copied());
        let actual_rust_key = rust_score_key_heads.get(&head).copied();
        let fallback_rust_key =
            kv_head.and_then(|kv_head| rust_fallback_key_heads.get(&kv_head).copied());
        let rust_key = actual_rust_key.or(fallback_rust_key);
        let rust_key_source = if actual_rust_key.is_some() {
            "rust_attention_k_score_input_head"
        } else {
            "rust_attention_k_cache_f16_roundtrip_kv_head"
        };
        let rust_key_source_kind = if actual_rust_key.is_some() {
            "actual_score_input"
        } else {
            "fallback_f16_cache_proxy"
        };

        if let (
            Some(reference_query),
            Some(rust_query),
            Some(reference_key),
            Some(rust_key),
            Some(rust_score),
            Some(kv_head),
        ) = (reference_query, rust_query, reference_key, rust_key, rust_score, kv_head)
        {
            let live_token_count = reference_key_token_count(reference_key)
                .unwrap_or(reference_score.first_values.len())
                .min(rust_key_token_count(rust_key).unwrap_or(rust_score.first_values.len()))
                .min(reference_score.first_values.len())
                .min(rust_score.first_values.len());
            let candidates = [
                ScoreInputAttributionCandidate {
                    id: "reference_q_reference_k",
                    query_source: "reference_Qcur",
                    key_source: "reference_k_kv_head",
                    key_layout: ScoreKeyLayout::TokenMajor,
                    query_values: &reference_query.first_values,
                    key_values: &reference_key.first_values,
                },
                ScoreInputAttributionCandidate {
                    id: "rust_q_reference_k",
                    query_source: "rust_attention_q_rope",
                    key_source: "reference_k_kv_head",
                    key_layout: ScoreKeyLayout::TokenMajor,
                    query_values: &rust_query.first_values,
                    key_values: &reference_key.first_values,
                },
                ScoreInputAttributionCandidate {
                    id: "reference_q_rust_k",
                    query_source: "reference_Qcur",
                    key_source: rust_key_source,
                    key_layout: ScoreKeyLayout::DimMajor,
                    query_values: &reference_query.first_values,
                    key_values: &rust_key.first_values,
                },
                ScoreInputAttributionCandidate {
                    id: "rust_q_rust_k",
                    query_source: "rust_attention_q_rope",
                    key_source: rust_key_source,
                    key_layout: ScoreKeyLayout::DimMajor,
                    query_values: &rust_query.first_values,
                    key_values: &rust_key.first_values,
                },
            ];

            let mut candidate_rows = Vec::new();
            let mut reference_deltas = BTreeMap::<&str, Value>::new();
            let mut rust_deltas = BTreeMap::<&str, Value>::new();
            for candidate in candidates {
                if let (Some(reference_delta), Some(rust_delta)) = (
                    score_input_candidate_delta(
                        candidate,
                        head,
                        head_dim,
                        live_token_count,
                        &reference_score.first_values,
                        ReferenceScoreLengthPolicy::PaddedTailZeroed,
                    ),
                    score_input_candidate_delta(
                        candidate,
                        head,
                        head_dim,
                        live_token_count,
                        &rust_score.first_values,
                        ReferenceScoreLengthPolicy::LiveKeyCountOnly,
                    ),
                ) {
                    candidate_rows.push(json!({
                        "candidate": candidate.id,
                        "query_source": candidate.query_source,
                        "key_source": candidate.key_source,
                        "key_layout": candidate.key_layout.label(),
                        "accum_policy": "f32_sequential",
                        "query_f16_roundtrip": true,
                        "key_f16_roundtrip": false,
                        "reference_score_delta": reference_delta,
                        "rust_score_delta": rust_delta,
                    }));
                    reference_deltas.insert(candidate.id, reference_delta);
                    rust_deltas.insert(candidate.id, rust_delta);
                }
            }

            if candidate_rows.is_empty() {
                missing_input_count += 1;
                rows.push(json!({
                    "head": head,
                    "kv_head": kv_head,
                    "status": "missing_input",
                    "candidate_count": 0,
                }));
                continue;
            }

            let reference_best =
                score_input_best_candidate(reference_deltas.iter().map(|(id, delta)| (*id, delta)));
            let rust_best =
                score_input_best_candidate(rust_deltas.iter().map(|(id, delta)| (*id, delta)));
            if let Some(reference_best) = reference_best {
                *reference_best_counts.entry(reference_best.to_string()).or_insert(0) += 1;
                if let Some(delta) = reference_deltas.get(reference_best) {
                    max_reference_best_abs_delta =
                        max_reference_best_abs_delta.max(delta_metric(delta, "/max_abs_delta"));
                    max_reference_best_rms_delta =
                        max_reference_best_rms_delta.max(delta_metric(delta, "/rms_abs_delta"));
                }
            }
            if let Some(rust_best) = rust_best {
                *rust_best_counts.entry(rust_best.to_string()).or_insert(0) += 1;
                if let Some(delta) = rust_deltas.get(rust_best) {
                    max_rust_best_abs_delta =
                        max_rust_best_abs_delta.max(delta_metric(delta, "/max_abs_delta"));
                    max_rust_best_rms_delta =
                        max_rust_best_rms_delta.max(delta_metric(delta, "/rms_abs_delta"));
                }
            }

            compared_count += 1;
            rows.push(json!({
                "head": head,
                "kv_head": kv_head,
                "status": "compared",
                "live_token_count": live_token_count,
                "reference_score_token_count": reference_score.first_values.len(),
                "rust_score_token_count": rust_score.first_values.len(),
                "rust_key_source": rust_key_source,
                "rust_key_source_kind": rust_key_source_kind,
                "reference_best_candidate": reference_best,
                "rust_best_candidate": rust_best,
                "candidates": candidate_rows,
            }));
        } else {
            missing_input_count += 1;
            rows.push(json!({
                "head": head,
                "kv_head": kv_head,
                "status": "missing_input",
                "reference_query_present": reference_query.is_some(),
                "rust_query_present": rust_query.is_some(),
                "reference_key_present": reference_key.is_some(),
                "rust_key_present": rust_key.is_some(),
                "rust_score_key_present": actual_rust_key.is_some(),
                "rust_fallback_key_present": fallback_rust_key.is_some(),
                "reference_score_present": true,
                "rust_score_present": rust_score.is_some(),
            }));
        }
    }

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "policy": "score input attribution is diagnostic-only evidence for whether raw attention score drift follows reference or Rust Q/K score inputs; it does not promote reference parity, A770 semantic quality, attention score residency, selected attention, resident KV, or any support claim",
        "formula": "f32_accum_q_f16_k_f32",
        "reference_score_length_policy": "padded_tail_zeroed",
        "rust_score_length_policy": "live_key_count_only",
        "reference_query_stage_present": reference_query.is_some(),
        "rust_query_stage_present": rust_query.is_some(),
        "reference_key_head_count": reference_key_heads.len(),
        "rust_key_head_count": if rust_score_key_heads.is_empty() {
            rust_fallback_key_heads.len()
        } else {
            rust_score_key_heads.len()
        },
        "rust_score_key_head_count": rust_score_key_heads.len(),
        "rust_fallback_key_head_count": rust_fallback_key_heads.len(),
        "rust_key_stage_source": if rust_score_key_heads.is_empty() {
            "attention_k_cache_f16_roundtrip_kv_head_fallback"
        } else {
            "attention_k_score_input_head"
        },
        "reference_score_head_count": reference_score_heads.len(),
        "rust_score_head_count": rust_score_heads.len(),
        "head_dim": if head_dim == 0 { Value::Null } else { json!(head_dim) },
        "head_count": if head_count == 0 { Value::Null } else { json!(head_count) },
        "group_size": group_size,
        "compared_count": compared_count,
        "missing_input_count": missing_input_count,
        "reference_best_candidate_counts": reference_best_counts,
        "rust_best_candidate_counts": rust_best_counts,
        "max_reference_best_abs_delta": max_reference_best_abs_delta,
        "max_reference_best_rms_delta": max_reference_best_rms_delta,
        "max_rust_best_abs_delta": max_rust_best_abs_delta,
        "max_rust_best_rms_delta": max_rust_best_rms_delta,
        "all_compared": !reference_score_heads.is_empty() && missing_input_count == 0,
        "rows": rows,
    })
}

#[derive(Debug, Clone, Copy)]
struct ReferenceProbabilitySoftmaxVariantSpec {
    id: &'static str,
    scale_policy: ReferenceScoreScalePolicy,
    length_policy: ReferenceScoreLengthPolicy,
    score_f16_roundtrip: bool,
    output_f16_roundtrip: bool,
}

fn reference_probability_softmax_variant_specs() -> [ReferenceProbabilitySoftmaxVariantSpec; 8] {
    [
        ReferenceProbabilitySoftmaxVariantSpec {
            id: "reference_probability_softmax_unscaled_live",
            scale_policy: ReferenceScoreScalePolicy::Unscaled,
            length_policy: ReferenceScoreLengthPolicy::LiveKeyCountOnly,
            score_f16_roundtrip: false,
            output_f16_roundtrip: false,
        },
        ReferenceProbabilitySoftmaxVariantSpec {
            id: "reference_probability_softmax_scaled_1_sqrt_head_dim_live",
            scale_policy: ReferenceScoreScalePolicy::HeadDimSqrtRecip,
            length_policy: ReferenceScoreLengthPolicy::LiveKeyCountOnly,
            score_f16_roundtrip: false,
            output_f16_roundtrip: false,
        },
        ReferenceProbabilitySoftmaxVariantSpec {
            id: "reference_probability_softmax_unscaled_padded_tail_zeroed",
            scale_policy: ReferenceScoreScalePolicy::Unscaled,
            length_policy: ReferenceScoreLengthPolicy::PaddedTailZeroed,
            score_f16_roundtrip: false,
            output_f16_roundtrip: false,
        },
        ReferenceProbabilitySoftmaxVariantSpec {
            id: "reference_probability_softmax_scaled_1_sqrt_head_dim_padded_tail_zeroed",
            scale_policy: ReferenceScoreScalePolicy::HeadDimSqrtRecip,
            length_policy: ReferenceScoreLengthPolicy::PaddedTailZeroed,
            score_f16_roundtrip: false,
            output_f16_roundtrip: false,
        },
        ReferenceProbabilitySoftmaxVariantSpec {
            id: "reference_probability_softmax_scaled_llm_build_kq_scale_padded_tail_zeroed",
            scale_policy: ReferenceScoreScalePolicy::LlmBuildKqScale,
            length_policy: ReferenceScoreLengthPolicy::PaddedTailZeroed,
            score_f16_roundtrip: false,
            output_f16_roundtrip: false,
        },
        ReferenceProbabilitySoftmaxVariantSpec {
            id: "reference_probability_softmax_scaled_1_sqrt_head_dim_mask_applied",
            scale_policy: ReferenceScoreScalePolicy::HeadDimSqrtRecip,
            length_policy: ReferenceScoreLengthPolicy::MaskApplied,
            score_f16_roundtrip: false,
            output_f16_roundtrip: false,
        },
        ReferenceProbabilitySoftmaxVariantSpec {
            id: "reference_probability_softmax_scaled_1_sqrt_head_dim_score_f16",
            scale_policy: ReferenceScoreScalePolicy::HeadDimSqrtRecip,
            length_policy: ReferenceScoreLengthPolicy::PaddedTailZeroed,
            score_f16_roundtrip: true,
            output_f16_roundtrip: false,
        },
        ReferenceProbabilitySoftmaxVariantSpec {
            id: "reference_probability_softmax_scaled_1_sqrt_head_dim_output_f16",
            scale_policy: ReferenceScoreScalePolicy::HeadDimSqrtRecip,
            length_policy: ReferenceScoreLengthPolicy::PaddedTailZeroed,
            score_f16_roundtrip: false,
            output_f16_roundtrip: true,
        },
    ]
}

fn reference_probability_live_token_count(probability: &ReferenceTraceRecord) -> Option<usize> {
    let mut live = probability.first_values.len();
    while live > 0 && probability.first_values[live - 1].abs() <= 1.0e-12 {
        live -= 1;
    }
    if live == 0 { None } else { Some(live) }
}

fn reference_probability_softmax_values(
    score: &ReferenceTraceRecord,
    live_token_count: usize,
    target_token_count: usize,
    scale: f64,
    variant: ReferenceProbabilitySoftmaxVariantSpec,
) -> Option<Vec<f32>> {
    probability_softmax_values_from_scores(
        &score.first_values,
        live_token_count,
        target_token_count,
        scale,
        variant.length_policy,
        variant.score_f16_roundtrip,
        variant.output_f16_roundtrip,
    )
}

fn probability_softmax_values_from_scores(
    score_values: &[f32],
    live_token_count: usize,
    target_token_count: usize,
    scale: f64,
    length_policy: ReferenceScoreLengthPolicy,
    score_f16_roundtrip: bool,
    output_f16_roundtrip: bool,
) -> Option<Vec<f32>> {
    if live_token_count == 0 || target_token_count == 0 {
        return None;
    }
    let live_len = live_token_count.min(score_values.len()).min(target_token_count);
    if live_len == 0 {
        return None;
    }

    let mut scaled_scores = Vec::with_capacity(live_len);
    for token in 0..live_len {
        let score = numeric_variant_value(score_values[token], score_f16_roundtrip);
        scaled_scores.push(score as f64 * scale);
    }
    let row_max = scaled_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut exp_values = Vec::with_capacity(live_len);
    let mut exp_sum = 0.0f64;
    for score in scaled_scores {
        let value = (score - row_max).exp();
        exp_sum += value;
        exp_values.push(value);
    }
    if exp_sum == 0.0 || !exp_sum.is_finite() {
        return None;
    }

    let mut probabilities = exp_values
        .into_iter()
        .map(|value| {
            let probability = (value / exp_sum) as f32;
            if output_f16_roundtrip { f16_roundtrip(probability) } else { probability }
        })
        .collect::<Vec<_>>();

    match length_policy {
        ReferenceScoreLengthPolicy::LiveKeyCountOnly => {}
        ReferenceScoreLengthPolicy::PaddedTailZeroed | ReferenceScoreLengthPolicy::MaskApplied => {
            probabilities.resize(target_token_count, 0.0);
        }
    }
    Some(probabilities)
}

fn attention_probability_reference_softmax_variants(
    reference_records: &[ReferenceTraceRecord],
) -> Value {
    let query = reference_rope_query_record(reference_records);
    let score_heads = reference_records
        .iter()
        .filter_map(|record| parse_stage_head(&record.stage, "kq_head").map(|head| (head, record)))
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let probability_heads = reference_records
        .iter()
        .filter_map(|record| {
            parse_stage_head(&record.stage, "kq_soft_max_ext_head").map(|head| (head, record))
        })
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let variants = reference_probability_softmax_variant_specs();
    let (head_dim, head_count) = query.and_then(reference_query_head_dim_count).unwrap_or((0, 0));

    let mut rows = Vec::new();
    let mut compared_count = 0usize;
    let mut missing_input_count = 0usize;
    let mut unexplained_head_count = 0usize;
    let mut max_best_abs_delta = 0.0f64;
    let mut max_best_rms_delta = 0.0f64;
    let mut best_variant_counts = BTreeMap::<String, usize>::new();

    for (&head, probability) in &probability_heads {
        let score = score_heads.get(&head).copied();
        if let Some(score) = score {
            let live_token_count = reference_probability_live_token_count(probability)
                .unwrap_or(probability.first_values.len())
                .min(score.first_values.len());
            let target_token_count = probability.first_values.len();
            let padded_token_count = target_token_count.saturating_sub(live_token_count);
            let mut variant_rows = Vec::new();
            let mut best_variant_id = "";
            let mut best_delta = Value::Null;
            let mut best_rank = (f64::INFINITY, f64::INFINITY, true);
            let mut best_scale = 1.0f64;
            let mut best_scale_policy = "";
            let mut best_mask_policy = "";
            let mut best_length_policy = "";
            let mut best_score_f16_roundtrip = false;
            let mut best_output_f16_roundtrip = false;

            for variant in variants {
                let scale = variant.scale_policy.scale(head_dim);
                if let Some(values) = reference_probability_softmax_values(
                    score,
                    live_token_count,
                    target_token_count,
                    scale,
                    variant,
                ) {
                    let delta = compare_vectors(&values, &probability.first_values);
                    let rms = delta_metric(&delta, "/rms_abs_delta");
                    let max_abs = delta_metric(&delta, "/max_abs_delta");
                    let count_mismatch =
                        !delta.pointer("/count_match").and_then(Value::as_bool).unwrap_or(false);
                    let rank = (rms, max_abs, count_mismatch);
                    if rank < best_rank {
                        best_rank = rank;
                        best_variant_id = variant.id;
                        best_delta = delta.clone();
                        best_scale = scale;
                        best_scale_policy = variant.scale_policy.label();
                        best_mask_policy = variant.length_policy.mask_policy();
                        best_length_policy = variant.length_policy.label();
                        best_score_f16_roundtrip = variant.score_f16_roundtrip;
                        best_output_f16_roundtrip = variant.output_f16_roundtrip;
                    }
                    variant_rows.push(json!({
                        "variant": variant.id,
                        "head": head,
                        "token_count": target_token_count,
                        "live_token_count": live_token_count,
                        "padded_token_count": padded_token_count,
                        "scale": scale,
                        "scale_policy": variant.scale_policy.label(),
                        "scale_source": variant.scale_policy.source(),
                        "mask_policy": variant.length_policy.mask_policy(),
                        "length_policy": variant.length_policy.label(),
                        "score_f16_roundtrip": variant.score_f16_roundtrip,
                        "output_f16_roundtrip": variant.output_f16_roundtrip,
                        "max_abs_delta": max_abs,
                        "rms_delta": rms,
                        "delta": delta,
                    }));
                }
            }

            if variant_rows.is_empty() {
                missing_input_count += 1;
                rows.push(json!({
                    "head": head,
                    "status": "missing_input",
                    "score_stage_present": true,
                    "probability_stage_present": true,
                }));
                continue;
            }

            let head_explained = reference_score_variant_explained(&best_delta);
            if !head_explained {
                unexplained_head_count += 1;
            }
            max_best_abs_delta =
                max_best_abs_delta.max(delta_metric(&best_delta, "/max_abs_delta"));
            max_best_rms_delta =
                max_best_rms_delta.max(delta_metric(&best_delta, "/rms_abs_delta"));
            *best_variant_counts.entry(best_variant_id.to_string()).or_insert(0) += 1;
            compared_count += 1;
            rows.push(json!({
                "head": head,
                "status": "compared",
                "token_count": target_token_count,
                "live_token_count": live_token_count,
                "padded_token_count": padded_token_count,
                "best_variant": best_variant_id,
                "scale": best_scale,
                "scale_policy": best_scale_policy,
                "mask_policy": best_mask_policy,
                "length_policy": best_length_policy,
                "score_f16_roundtrip": best_score_f16_roundtrip,
                "output_f16_roundtrip": best_output_f16_roundtrip,
                "max_abs_delta": delta_metric(&best_delta, "/max_abs_delta"),
                "rms_delta": delta_metric(&best_delta, "/rms_abs_delta"),
                "best_delta": best_delta,
                "head_explained": head_explained,
                "variants": variant_rows,
            }));
        } else {
            missing_input_count += 1;
            rows.push(json!({
                "head": head,
                "status": "missing_input",
                "score_stage_present": false,
                "probability_stage_present": true,
            }));
        }
    }

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "policy": "reference probability softmax variants are diagnostic-only probes for reproducing reference kq_soft_max_ext_head rows from reference kq_head rows; they do not promote reference parity, A770 semantic quality, softmax residency, selected attention, resident KV, or any support claim",
        "score_head_count": score_heads.len(),
        "probability_head_count": probability_heads.len(),
        "head_dim": if head_dim == 0 { Value::Null } else { json!(head_dim) },
        "head_count": if head_count == 0 { Value::Null } else { json!(head_count) },
        "variant_count": variants.len(),
        "variants_tested": variants.iter().map(|variant| variant.id).collect::<Vec<_>>(),
        "explanation_abs_threshold": 1.0e-4,
        "explanation_rms_threshold": 1.0e-4,
        "compared_count": compared_count,
        "missing_input_count": missing_input_count,
        "unexplained_head_count": unexplained_head_count,
        "max_best_abs_delta": max_best_abs_delta,
        "max_best_rms_delta": max_best_rms_delta,
        "best_variant_counts": best_variant_counts,
        "all_heads_explained": !probability_heads.is_empty()
            && missing_input_count == 0
            && unexplained_head_count == 0,
        "rows": rows,
    })
}

fn attention_probability_rust_softmax_recompute(
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    let query = rust_records.get("attention_q_rope");
    let score_heads = rust_records
        .iter()
        .filter_map(|(stage, record)| {
            parse_stage_head(stage, "attention_scores_raw_head").map(|head| (head, record))
        })
        .filter(|(_, record)| !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let probability_heads = rust_records
        .iter()
        .filter_map(|(stage, record)| {
            parse_stage_head(stage, "attn_scores_softmax_head").map(|head| (head, record))
        })
        .filter(|(_, record)| !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let variants = reference_probability_softmax_variant_specs();
    let (head_dim, head_count) = query.and_then(rust_query_head_dim_count).unwrap_or((0, 0));

    let mut rows = Vec::new();
    let mut compared_count = 0usize;
    let mut missing_input_count = 0usize;
    let mut unexplained_head_count = 0usize;
    let mut max_best_abs_delta = 0.0f64;
    let mut max_best_rms_delta = 0.0f64;
    let mut best_variant_counts = BTreeMap::<String, usize>::new();

    for (&head, probability) in &probability_heads {
        let score = score_heads.get(&head).copied();
        if let Some(score) = score {
            let target_token_count = probability.first_values.len();
            let live_token_count = target_token_count.min(score.first_values.len());
            let padded_token_count = target_token_count.saturating_sub(live_token_count);
            let mut variant_rows = Vec::new();
            let mut best_variant_id = "";
            let mut best_delta = Value::Null;
            let mut best_rank = (f64::INFINITY, f64::INFINITY, true);
            let mut best_scale = 1.0f64;
            let mut best_scale_policy = "";
            let mut best_mask_policy = "";
            let mut best_length_policy = "";
            let mut best_score_f16_roundtrip = false;
            let mut best_output_f16_roundtrip = false;

            for variant in variants {
                let scale = variant.scale_policy.scale(head_dim);
                if let Some(values) = probability_softmax_values_from_scores(
                    &score.first_values,
                    live_token_count,
                    target_token_count,
                    scale,
                    variant.length_policy,
                    variant.score_f16_roundtrip,
                    variant.output_f16_roundtrip,
                ) {
                    let delta = compare_vectors(&values, &probability.first_values);
                    let rms = delta_metric(&delta, "/rms_abs_delta");
                    let max_abs = delta_metric(&delta, "/max_abs_delta");
                    let count_mismatch =
                        !delta.pointer("/count_match").and_then(Value::as_bool).unwrap_or(false);
                    let rank = (rms, max_abs, count_mismatch);
                    if rank < best_rank {
                        best_rank = rank;
                        best_variant_id = variant.id;
                        best_delta = delta.clone();
                        best_scale = scale;
                        best_scale_policy = variant.scale_policy.label();
                        best_mask_policy = variant.length_policy.mask_policy();
                        best_length_policy = variant.length_policy.label();
                        best_score_f16_roundtrip = variant.score_f16_roundtrip;
                        best_output_f16_roundtrip = variant.output_f16_roundtrip;
                    }
                    variant_rows.push(json!({
                        "variant": variant.id,
                        "head": head,
                        "token_count": target_token_count,
                        "live_token_count": live_token_count,
                        "padded_token_count": padded_token_count,
                        "scale": scale,
                        "scale_policy": variant.scale_policy.label(),
                        "scale_source": variant.scale_policy.source(),
                        "mask_policy": variant.length_policy.mask_policy(),
                        "length_policy": variant.length_policy.label(),
                        "score_f16_roundtrip": variant.score_f16_roundtrip,
                        "output_f16_roundtrip": variant.output_f16_roundtrip,
                        "max_abs_delta": max_abs,
                        "rms_delta": rms,
                        "delta": delta,
                    }));
                }
            }

            if variant_rows.is_empty() {
                missing_input_count += 1;
                rows.push(json!({
                    "head": head,
                    "status": "missing_input",
                    "score_stage_present": true,
                    "probability_stage_present": true,
                }));
                continue;
            }

            let head_explained = reference_score_variant_explained(&best_delta);
            if !head_explained {
                unexplained_head_count += 1;
            }
            max_best_abs_delta =
                max_best_abs_delta.max(delta_metric(&best_delta, "/max_abs_delta"));
            max_best_rms_delta =
                max_best_rms_delta.max(delta_metric(&best_delta, "/rms_abs_delta"));
            *best_variant_counts.entry(best_variant_id.to_string()).or_insert(0) += 1;
            compared_count += 1;
            rows.push(json!({
                "head": head,
                "status": "compared",
                "token_count": target_token_count,
                "live_token_count": live_token_count,
                "padded_token_count": padded_token_count,
                "best_variant": best_variant_id,
                "scale": best_scale,
                "scale_policy": best_scale_policy,
                "mask_policy": best_mask_policy,
                "length_policy": best_length_policy,
                "score_f16_roundtrip": best_score_f16_roundtrip,
                "output_f16_roundtrip": best_output_f16_roundtrip,
                "max_abs_delta": delta_metric(&best_delta, "/max_abs_delta"),
                "rms_delta": delta_metric(&best_delta, "/rms_abs_delta"),
                "best_delta": best_delta,
                "head_explained": head_explained,
                "variants": variant_rows,
            }));
        } else {
            missing_input_count += 1;
            rows.push(json!({
                "head": head,
                "status": "missing_input",
                "score_stage_present": false,
                "probability_stage_present": true,
            }));
        }
    }

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "policy": "Rust probability softmax recompute is diagnostic-only evidence for whether Rust attn_scores_softmax_head rows are internally explained by Rust attention_scores_raw_head rows; it does not promote reference parity, A770 semantic quality, softmax residency, selected attention, resident KV, or any support claim",
        "score_head_count": score_heads.len(),
        "probability_head_count": probability_heads.len(),
        "head_dim": if head_dim == 0 { Value::Null } else { json!(head_dim) },
        "head_count": if head_count == 0 { Value::Null } else { json!(head_count) },
        "variant_count": variants.len(),
        "variants_tested": variants.iter().map(|variant| variant.id).collect::<Vec<_>>(),
        "explanation_abs_threshold": 1.0e-4,
        "explanation_rms_threshold": 1.0e-4,
        "compared_count": compared_count,
        "missing_input_count": missing_input_count,
        "unexplained_head_count": unexplained_head_count,
        "max_best_abs_delta": max_best_abs_delta,
        "max_best_rms_delta": max_best_rms_delta,
        "best_variant_counts": best_variant_counts,
        "all_heads_explained": !probability_heads.is_empty()
            && missing_input_count == 0
            && unexplained_head_count == 0,
        "rows": rows,
    })
}

fn attention_score_rust_scalar_recompute(
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    let query = rust_records.get("attention_q_rope");
    let key_heads = rust_records
        .iter()
        .filter_map(|(stage, record)| {
            parse_stage_head(stage, "attention_k_cache_kv_head").map(|head| (head, record))
        })
        .filter(|(_, record)| !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let score_heads = rust_records
        .iter()
        .filter_map(|(stage, record)| {
            parse_stage_head(stage, "attention_scores_raw_head").map(|head| (head, record))
        })
        .filter(|(_, record)| !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();

    let (head_dim, head_count) = query.and_then(rust_query_head_dim_count).unwrap_or((0, 0));
    let group_size = if head_count > 0 && !key_heads.is_empty() && head_count % key_heads.len() == 0
    {
        Some(head_count / key_heads.len())
    } else {
        None
    };

    let mut rows = Vec::new();
    let mut compared_count = 0usize;
    let mut missing_input_count = 0usize;
    let mut max_abs_delta = 0.0f64;
    let mut max_rms_delta = 0.0f64;

    for (&head, score) in &score_heads {
        let kv_head = group_size.map(|group_size| head / group_size);
        let key = kv_head.and_then(|kv_head| key_heads.get(&kv_head).copied());
        if let (Some(query), Some(key), Some(kv_head)) = (query, key, kv_head) {
            let token_count = rust_key_token_count(key)
                .unwrap_or(score.first_values.len())
                .min(score.first_values.len());
            if let Some(recomputed) =
                rust_score_row_from_query_key(query, key, head, head_dim, token_count)
            {
                let delta = compare_vectors(&recomputed, &score.first_values);
                max_abs_delta = max_abs_delta
                    .max(delta.pointer("/max_abs_delta").and_then(Value::as_f64).unwrap_or(0.0));
                max_rms_delta = max_rms_delta
                    .max(delta.pointer("/rms_abs_delta").and_then(Value::as_f64).unwrap_or(0.0));
                compared_count += 1;
                rows.push(json!({
                    "head": head,
                    "kv_head": kv_head,
                    "status": "compared",
                    "query_stage_present": true,
                    "key_stage_present": true,
                    "score_stage_present": true,
                    "live_token_count": token_count,
                    "recomputed_first_values": recomputed,
                    "delta": delta,
                }));
            } else {
                missing_input_count += 1;
                rows.push(json!({
                    "head": head,
                    "kv_head": kv_head,
                    "status": "missing_input",
                    "query_stage_present": true,
                    "key_stage_present": true,
                    "score_stage_present": true,
                }));
            }
        } else {
            missing_input_count += 1;
            rows.push(json!({
                "head": head,
                "kv_head": kv_head,
                "status": "missing_input",
                "query_stage_present": query.is_some(),
                "key_stage_present": key.is_some(),
                "score_stage_present": true,
            }));
        }
    }

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "policy": "Rust scalar score recompute is diagnostic arithmetic evidence only; it recomputes sampled raw attention rows from Rust traced post-RoPE Q and K-cache inputs and does not promote reference parity, A770 semantic quality, attention score residency, selected attention, resident KV, or any support claim",
        "query_stage_present": query.is_some(),
        "query_stage": query.and_then(|record| record.stage.clone()),
        "key_head_count": key_heads.len(),
        "score_head_count": score_heads.len(),
        "head_dim": if head_dim == 0 { Value::Null } else { json!(head_dim) },
        "head_count": if head_count == 0 { Value::Null } else { json!(head_count) },
        "group_size": group_size,
        "compared_count": compared_count,
        "missing_input_count": missing_input_count,
        "max_abs_delta": max_abs_delta,
        "max_rms_delta": max_rms_delta,
        "all_compared": !score_heads.is_empty() && missing_input_count == 0,
        "rows": rows,
    })
}

fn reference_rope_query_record(
    reference_records: &[ReferenceTraceRecord],
) -> Option<&ReferenceTraceRecord> {
    reference_records
        .iter()
        .find(|record| {
            record.stage == "Qcur"
                && record.values_available
                && record.shape.len() >= 2
                && record.shape[0] > 1
                && record.shape[1] > 1
                && record.first_values.len() == record.nelements as usize
        })
        .or_else(|| {
            reference_records.iter().find(|record| {
                record.stage == "Qcur" && record.values_available && !record.first_values.is_empty()
            })
        })
}

fn reference_query_head_dim_count(record: &ReferenceTraceRecord) -> Option<(usize, usize)> {
    let head_dim = usize::try_from(*record.shape.first()?).ok()?;
    let head_count = usize::try_from(*record.shape.get(1)?).ok()?;
    if head_dim == 0 || head_count == 0 || record.first_values.len() < head_dim * head_count {
        return None;
    }
    Some((head_dim, head_count))
}

fn rust_query_head_dim_count(record: &RustTraceRecord) -> Option<(usize, usize)> {
    match record.shape.as_slice() {
        [1, head_count, 1, head_dim] | [head_count, 1, head_dim] | [head_count, head_dim] => {
            if *head_dim == 0
                || *head_count == 0
                || record.first_values.len() < head_dim * head_count
            {
                None
            } else {
                Some((*head_dim, *head_count))
            }
        }
        _ => None,
    }
}

fn reference_key_token_count(record: &ReferenceTraceRecord) -> Option<usize> {
    usize::try_from(*record.shape.get(1)?).ok().filter(|count| *count > 0)
}

fn rust_key_token_count(record: &RustTraceRecord) -> Option<usize> {
    record.shape.get(1).copied().filter(|count| *count > 0)
}

fn reference_score_row_from_query_key(
    query: &ReferenceTraceRecord,
    key: &ReferenceTraceRecord,
    head: usize,
    head_dim: usize,
    token_count: usize,
) -> Option<Vec<f32>> {
    if head_dim == 0 || token_count == 0 {
        return None;
    }
    let q_offset = head.checked_mul(head_dim)?;
    if query.first_values.len() < q_offset + head_dim {
        return None;
    }
    if key.first_values.len() < token_count.checked_mul(head_dim)? {
        return None;
    }
    let mut scores = Vec::with_capacity(token_count);
    for token in 0..token_count {
        let mut sum = 0.0f64;
        for dim in 0..head_dim {
            let q = query.first_values[q_offset + dim] as f64;
            let k = key.first_values[token * head_dim + dim] as f64;
            sum += q * k;
        }
        scores.push(sum as f32);
    }
    Some(scores)
}

fn rust_score_row_from_query_key(
    query: &RustTraceRecord,
    key: &RustTraceRecord,
    head: usize,
    head_dim: usize,
    token_count: usize,
) -> Option<Vec<f32>> {
    if head_dim == 0 || token_count == 0 {
        return None;
    }
    let q_offset = head.checked_mul(head_dim)?;
    if query.first_values.len() < q_offset + head_dim {
        return None;
    }
    if key.first_values.len() < head_dim.checked_mul(token_count)? {
        return None;
    }
    let mut scores = Vec::with_capacity(token_count);
    for token in 0..token_count {
        let mut sum = 0.0f64;
        for dim in 0..head_dim {
            let q = query.first_values[q_offset + dim] as f64;
            let k = key.first_values[dim * token_count + token] as f64;
            sum += q * k;
        }
        scores.push(sum as f32);
    }
    Some(scores)
}

fn trace_dtype_compatible(reference_dtype: &str, rust_dtype: &str) -> bool {
    normalized_trace_dtype(reference_dtype) == normalized_trace_dtype(rust_dtype)
}

fn normalized_trace_dtype(dtype: &str) -> String {
    if dtype.eq_ignore_ascii_case("f32_from_f16") {
        return "f32".to_string();
    }
    dtype.to_ascii_lowercase()
}

fn attention_score_raw_head_lane_best_matches(
    reference_records: &[ReferenceTraceRecord],
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    head_lane_best_matches(
        reference_records,
        rust_records,
        "kq_head",
        "attention_scores_raw_head",
        "raw score head-lane best matches are diagnostic mapping evidence only; they do not promote reference parity, A770 semantic quality, attention score residency, selected attention, or any support claim",
    )
}

fn attention_probability_head_lane_best_matches(
    reference_records: &[ReferenceTraceRecord],
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    head_lane_best_matches(
        reference_records,
        rust_records,
        "kq_soft_max_ext_head",
        "attn_scores_softmax_head",
        "softmax probability head-lane best matches are diagnostic mapping evidence only; they do not promote reference parity, A770 semantic quality, softmax residency, selected attention, or any support claim",
    )
}

fn attention_key_cache_kv_head_best_matches(
    reference_records: &[ReferenceTraceRecord],
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    head_lane_best_matches(
        reference_records,
        rust_records,
        "k_kv_head",
        "attention_k_cache_kv_head",
        "key-cache KV-head best matches are diagnostic mapping evidence only; they do not promote reference parity, A770 semantic quality, attention score residency, resident KV, selected attention, or any support claim",
    )
}

fn attention_key_cache_f16_roundtrip_best_matches(
    reference_records: &[ReferenceTraceRecord],
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    head_lane_best_matches(
        reference_records,
        rust_records,
        "k_kv_head",
        "attention_k_cache_f16_roundtrip_kv_head",
        "key-cache F16-roundtrip best matches are diagnostic dtype-transform evidence only; they do not promote reference parity, A770 semantic quality, attention score residency, resident KV, selected attention, or any support claim",
    )
}

fn attention_key_cache_dim_major_f16_roundtrip_best_matches(
    reference_records: &[ReferenceTraceRecord],
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    let reference_heads = reference_records
        .iter()
        .filter_map(|record| {
            let head = parse_stage_head(&record.stage, "k_kv_head")?;
            let first_values = key_cache_dim_major_first_values(record)?;
            Some((head, first_values))
        })
        .collect::<BTreeMap<_, _>>();
    let rust_heads = rust_records
        .iter()
        .filter_map(|(stage, record)| {
            parse_stage_head(stage, "attention_k_cache_f16_roundtrip_kv_head")
                .map(|head| (head, record))
        })
        .filter(|(_, record)| !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();

    let mut rows = Vec::new();
    let mut identity_best_count = 0usize;
    let mut non_identity_best_count = 0usize;
    let mut missing_identity_count = 0usize;

    for (&reference_head, reference_values) in &reference_heads {
        let mut candidates = rust_heads
            .iter()
            .map(|(rust_head, rust)| {
                head_lane_delta(reference_head, *rust_head, reference_values, &rust.first_values)
            })
            .collect::<Vec<_>>();
        candidates.sort_by(head_lane_delta_order);

        let best = candidates.first().copied();
        let identity =
            candidates.iter().copied().find(|candidate| candidate.rust_head == reference_head);
        let identity_rank = identity.and_then(|identity| {
            candidates.iter().position(|candidate| {
                candidate.rust_head == identity.rust_head
                    && candidate.reference_head == identity.reference_head
            })
        });

        if let Some(best) = best {
            if best.rust_head == reference_head {
                identity_best_count += 1;
            } else {
                non_identity_best_count += 1;
            }
        }
        if identity.is_none() {
            missing_identity_count += 1;
        }

        rows.push(json!({
            "reference_head": reference_head,
            "best_rust_head": best.map(|best| best.rust_head),
            "identity_rust_head": reference_head,
            "identity_is_best": best.is_some_and(|best| best.rust_head == reference_head),
            "identity_rank": identity_rank.map(|rank| rank + 1),
            "best_delta": best.map(head_lane_delta_summary),
            "identity_delta": identity.map(head_lane_delta_summary),
            "candidate_count": candidates.len(),
        }));
    }

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "policy": "key-cache dim-major F16-roundtrip best matches are diagnostic layout evidence only; they reinterpret reference k_kv_head samples from token-major to dim-major and do not promote reference parity, A770 semantic quality, attention score residency, resident KV, selected attention, or any support claim",
        "reference_stage_prefix": "k_kv_head",
        "reference_reinterpretation": "token_major_to_dim_major",
        "rust_stage_prefix": "attention_k_cache_f16_roundtrip_kv_head",
        "reference_head_count": reference_heads.len(),
        "rust_head_count": rust_heads.len(),
        "identity_best_count": identity_best_count,
        "non_identity_best_count": non_identity_best_count,
        "missing_identity_count": missing_identity_count,
        "all_identity_best": !reference_heads.is_empty()
            && reference_heads.len() == identity_best_count
            && missing_identity_count == 0,
        "rows": rows,
    })
}

fn key_cache_dim_major_first_values(record: &ReferenceTraceRecord) -> Option<Vec<f32>> {
    if !record.values_available || record.first_values.is_empty() {
        return None;
    }
    let dim_count = usize::try_from(*record.shape.first()?).ok()?;
    let token_count = usize::try_from(*record.shape.get(1)?).ok()?;
    let sample_count = dim_count.checked_mul(token_count)?;
    if dim_count == 0 || token_count == 0 || record.first_values.len() < sample_count {
        return None;
    }
    let mut dim_major = Vec::with_capacity(sample_count);
    for dim in 0..dim_count {
        for token in 0..token_count {
            dim_major.push(record.first_values[token * dim_count + dim]);
        }
    }
    Some(dim_major)
}

fn attention_value_cache_kv_head_best_matches(
    reference_records: &[ReferenceTraceRecord],
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    head_lane_best_matches(
        reference_records,
        rust_records,
        "v_kv_head",
        "attention_v_cache_kv_head",
        "value-cache KV-head best matches are diagnostic mapping evidence only; they do not promote reference parity, A770 semantic quality, value mix residency, resident KV, selected attention, or any support claim",
    )
}

fn attention_value_cache_rust_layout_best_matches(
    reference_records: &[ReferenceTraceRecord],
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    head_lane_best_matches(
        reference_records,
        rust_records,
        "v_cache_rust_layout_head",
        "attention_v_cache_kv_head",
        "value-cache Rust-layout best matches are diagnostic layout-transform evidence only; they do not promote reference parity, A770 semantic quality, value mix residency, resident KV, selected attention, or any support claim",
    )
}

fn attention_value_cache_f16_roundtrip_best_matches(
    reference_records: &[ReferenceTraceRecord],
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    head_lane_best_matches(
        reference_records,
        rust_records,
        "v_cache_rust_layout_head",
        "attention_v_cache_f16_roundtrip_kv_head",
        "value-cache F16-roundtrip best matches are diagnostic dtype-transform evidence only; they do not promote reference parity, A770 semantic quality, value mix residency, resident KV, selected attention, or any support claim",
    )
}

fn f16_bucket_delta(left: &[f32], right: &[f32]) -> Value {
    let compared_count = left.len().min(right.len());
    let mut mismatch_count = 0usize;
    let mut first_mismatch = Value::Null;
    for index in 0..compared_count {
        let left_bits = f32_to_f16_bits_nearest_even(left[index]);
        let right_bits = f32_to_f16_bits_nearest_even(right[index]);
        if left_bits != right_bits {
            mismatch_count += 1;
            if first_mismatch.is_null() {
                first_mismatch = json!({
                    "index": index,
                    "left": left[index],
                    "right": right[index],
                    "left_f16_bits": format!("0x{left_bits:04x}"),
                    "right_f16_bits": format!("0x{right_bits:04x}"),
                    "left_f16_value": f16_bits_to_f32(left_bits),
                    "right_f16_value": f16_bits_to_f32(right_bits),
                });
            }
        }
    }
    let delta = compare_prefix(left, right, compared_count);
    json!({
        "compared_count": compared_count,
        "left_count": left.len(),
        "right_count": right.len(),
        "count_match": left.len() == right.len(),
        "f16_bucket_mismatch_count": mismatch_count,
        "f16_bucket_match_count": compared_count.saturating_sub(mismatch_count),
        "first_f16_bucket_mismatch": first_mismatch,
        "max_abs_delta": delta_metric(&delta, "/max_abs_delta"),
        "rms_abs_delta": delta_metric(&delta, "/rms_abs_delta"),
        "delta": delta,
    })
}

fn f16_bucket_delta_dim_major(
    left: &[f32],
    right: &[f32],
    dim_count: usize,
    token_count: usize,
) -> Value {
    let mut delta = f16_bucket_delta(left, right);
    let compared_count = left.len().min(right.len()).min(dim_count.saturating_mul(token_count));
    let mut token_mismatch_counts = vec![0usize; token_count];
    let mut first_layout_mismatch = Value::Null;

    for index in 0..compared_count {
        let left_bits = f32_to_f16_bits_nearest_even(left[index]);
        let right_bits = f32_to_f16_bits_nearest_even(right[index]);
        if left_bits != right_bits {
            let dim = index / token_count;
            let token = index % token_count;
            if let Some(count) = token_mismatch_counts.get_mut(token) {
                *count += 1;
            }
            if first_layout_mismatch.is_null() {
                first_layout_mismatch = json!({
                    "index": index,
                    "dim": dim,
                    "token": token,
                    "left": left[index],
                    "right": right[index],
                    "left_f16_bits": format!("0x{left_bits:04x}"),
                    "right_f16_bits": format!("0x{right_bits:04x}"),
                    "left_f16_value": f16_bits_to_f32(left_bits),
                    "right_f16_value": f16_bits_to_f32(right_bits),
                });
            }
        }
    }

    let token_rows = token_mismatch_counts
        .iter()
        .enumerate()
        .filter_map(|(token, count)| {
            if *count == 0 {
                None
            } else {
                Some(json!({
                    "token": token,
                    "f16_bucket_mismatch_count": count,
                }))
            }
        })
        .collect::<Vec<_>>();

    if let Some(object) = delta.as_object_mut() {
        object.insert("layout".to_string(), json!("dim_major_token_minor"));
        object.insert("dim_count".to_string(), json!(dim_count));
        object.insert("token_count".to_string(), json!(token_count));
        object.insert("first_f16_bucket_mismatch_layout".to_string(), first_layout_mismatch);
        object.insert("token_mismatch_counts".to_string(), json!(token_rows));
    }

    delta
}

fn attention_value_cache_f16_amplification(
    reference_records: &[ReferenceTraceRecord],
    rust_records: &BTreeMap<String, RustTraceRecord>,
) -> Value {
    let reference_projection = reference_records.iter().find(|record| {
        record.stage == "Vcur" && record.values_available && !record.first_values.is_empty()
    });
    let rust_projection =
        rust_records.get("attention_v").filter(|record| !record.first_values.is_empty());
    let projection_delta = reference_projection
        .zip(rust_projection)
        .map(|(reference, rust)| f16_bucket_delta(&reference.first_values, &rust.first_values));

    let reference_cache_heads = reference_records
        .iter()
        .filter_map(|record| {
            parse_stage_head(&record.stage, "v_cache_rust_layout_head").map(|head| (head, record))
        })
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let rust_cache_heads = rust_records
        .iter()
        .filter_map(|(stage, record)| {
            parse_stage_head(stage, "attention_v_cache_f16_roundtrip_kv_head")
                .map(|head| (head, record))
        })
        .filter(|(_, record)| !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();

    let mut rows = Vec::new();
    let mut compared_head_count = 0usize;
    let mut missing_head_count = 0usize;
    let mut total_bucket_mismatch_count = 0usize;
    let mut max_bucket_mismatch_count = 0usize;
    let mut max_abs_delta = 0.0f64;
    let mut max_rms_delta = 0.0f64;

    for (&head, reference) in &reference_cache_heads {
        if let Some(rust) = rust_cache_heads.get(&head) {
            let dim_count = reference.shape.first().and_then(|dim| usize::try_from(*dim).ok());
            let token_count = reference.shape.get(1).and_then(|dim| usize::try_from(*dim).ok());
            let delta = match (dim_count, token_count) {
                (Some(dim_count), Some(token_count)) if dim_count > 0 && token_count > 0 => {
                    f16_bucket_delta_dim_major(
                        &reference.first_values,
                        &rust.first_values,
                        dim_count,
                        token_count,
                    )
                }
                _ => f16_bucket_delta(&reference.first_values, &rust.first_values),
            };
            let bucket_mismatch_count =
                delta.pointer("/f16_bucket_mismatch_count").and_then(Value::as_u64).unwrap_or(0)
                    as usize;
            total_bucket_mismatch_count += bucket_mismatch_count;
            max_bucket_mismatch_count = max_bucket_mismatch_count.max(bucket_mismatch_count);
            max_abs_delta = max_abs_delta.max(delta_metric(&delta, "/max_abs_delta"));
            max_rms_delta = max_rms_delta.max(delta_metric(&delta, "/rms_abs_delta"));
            compared_head_count += 1;
            rows.push(json!({
                "head": head,
                "status": "compared",
                "delta": delta,
            }));
        } else {
            missing_head_count += 1;
            rows.push(json!({
                "head": head,
                "status": "missing_rust_value_cache",
            }));
        }
    }

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "policy": "value-cache F16 amplification is diagnostic-only evidence for whether tiny V-projection deltas cross F16 bucket boundaries before value mix; it does not promote reference parity, A770 semantic quality, value mix residency, resident KV, selected attention, or any support claim",
        "reference_projection_stage": "Vcur",
        "rust_projection_stage": "attention_v",
        "reference_projection_present": reference_projection.is_some(),
        "rust_projection_present": rust_projection.is_some(),
        "projection_delta": projection_delta,
        "reference_value_cache_stage_prefix": "v_cache_rust_layout_head",
        "rust_value_cache_stage_prefix": "attention_v_cache_f16_roundtrip_kv_head",
        "reference_head_count": reference_cache_heads.len(),
        "rust_head_count": rust_cache_heads.len(),
        "compared_head_count": compared_head_count,
        "missing_head_count": missing_head_count,
        "total_f16_bucket_mismatch_count": total_bucket_mismatch_count,
        "max_head_f16_bucket_mismatch_count": max_bucket_mismatch_count,
        "max_abs_delta": max_abs_delta,
        "max_rms_delta": max_rms_delta,
        "all_compared": !reference_cache_heads.is_empty() && missing_head_count == 0,
        "rows": rows,
    })
}

fn head_lane_best_matches(
    reference_records: &[ReferenceTraceRecord],
    rust_records: &BTreeMap<String, RustTraceRecord>,
    reference_prefix: &str,
    rust_prefix: &str,
    policy: &str,
) -> Value {
    let reference_heads = reference_records
        .iter()
        .filter_map(|record| {
            parse_stage_head(&record.stage, reference_prefix).map(|head| (head, record))
        })
        .filter(|(_, record)| record.values_available && !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();
    let rust_heads = rust_records
        .iter()
        .filter_map(|(stage, record)| {
            parse_stage_head(stage, rust_prefix).map(|head| (head, record))
        })
        .filter(|(_, record)| !record.first_values.is_empty())
        .collect::<BTreeMap<_, _>>();

    let mut rows = Vec::new();
    let mut identity_best_count = 0usize;
    let mut non_identity_best_count = 0usize;
    let mut missing_identity_count = 0usize;

    for (&reference_head, reference) in &reference_heads {
        let mut candidates = rust_heads
            .iter()
            .map(|(rust_head, rust)| {
                head_lane_delta(
                    reference_head,
                    *rust_head,
                    &reference.first_values,
                    &rust.first_values,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by(head_lane_delta_order);

        let best = candidates.first().copied();
        let identity =
            candidates.iter().copied().find(|candidate| candidate.rust_head == reference_head);
        let identity_rank = identity.and_then(|identity| {
            candidates.iter().position(|candidate| {
                candidate.rust_head == identity.rust_head
                    && candidate.reference_head == identity.reference_head
            })
        });

        if let Some(best) = best {
            if best.rust_head == reference_head {
                identity_best_count += 1;
            } else {
                non_identity_best_count += 1;
            }
        }
        if identity.is_none() {
            missing_identity_count += 1;
        }

        rows.push(json!({
            "reference_head": reference_head,
            "best_rust_head": best.map(|best| best.rust_head),
            "identity_rust_head": reference_head,
            "identity_is_best": best.is_some_and(|best| best.rust_head == reference_head),
            "identity_rank": identity_rank.map(|rank| rank + 1),
            "best_delta": best.map(head_lane_delta_summary),
            "identity_delta": identity.map(head_lane_delta_summary),
            "candidate_count": candidates.len(),
        }));
    }

    json!({
        "diagnostic_only": true,
        "claim_allowed": false,
        "policy": policy,
        "reference_stage_prefix": reference_prefix,
        "rust_stage_prefix": rust_prefix,
        "reference_head_count": reference_heads.len(),
        "rust_head_count": rust_heads.len(),
        "identity_best_count": identity_best_count,
        "non_identity_best_count": non_identity_best_count,
        "missing_identity_count": missing_identity_count,
        "all_identity_best": !reference_heads.is_empty()
            && reference_heads.len() == identity_best_count
            && missing_identity_count == 0,
        "rows": rows,
    })
}

fn parse_stage_head(stage: &str, prefix: &str) -> Option<usize> {
    let suffix = stage.strip_prefix(prefix)?;
    let digit_count = suffix.chars().take_while(|ch| ch.is_ascii_digit()).count();
    if digit_count == 0 {
        return None;
    }
    let rest = &suffix[digit_count..];
    if !rest.is_empty() && !rest.starts_with('_') {
        return None;
    }
    suffix[..digit_count].parse::<usize>().ok()
}

fn head_lane_delta(
    reference_head: usize,
    rust_head: usize,
    reference: &[f32],
    rust: &[f32],
) -> HeadLaneDelta {
    let compared_count = reference.len().min(rust.len());
    let mut max_abs_delta = 0.0f64;
    let mut sum_sq_delta = 0.0f64;
    for i in 0..compared_count {
        let delta = (reference[i] as f64 - rust[i] as f64).abs();
        max_abs_delta = max_abs_delta.max(delta);
        sum_sq_delta += delta * delta;
    }
    HeadLaneDelta {
        reference_head,
        rust_head,
        compared_count,
        max_abs_delta,
        rms_abs_delta: if compared_count == 0 {
            f64::INFINITY
        } else {
            (sum_sq_delta / compared_count as f64).sqrt()
        },
    }
}

fn head_lane_delta_order(left: &HeadLaneDelta, right: &HeadLaneDelta) -> std::cmp::Ordering {
    left.rms_abs_delta
        .total_cmp(&right.rms_abs_delta)
        .then_with(|| left.max_abs_delta.total_cmp(&right.max_abs_delta))
        .then_with(|| left.rust_head.cmp(&right.rust_head))
}

fn head_lane_delta_summary(delta: HeadLaneDelta) -> Value {
    json!({
        "reference_head": delta.reference_head,
        "rust_head": delta.rust_head,
        "compared_count": delta.compared_count,
        "max_abs_delta": delta.max_abs_delta,
        "rms_abs_delta": delta.rms_abs_delta,
    })
}

fn trace_scope_mismatch(
    reference: Option<&ReferenceTraceRecord>,
    rust: Option<&RustTraceRecord>,
) -> Option<Value> {
    let reference = reference?;
    let rust = rust?;
    if let Some(scope) = reference_key_cache_head0_scope(reference, rust)
        .or_else(|| reference_key_cache_kv_live_scope(reference, rust))
        .or_else(|| reference_value_cache_head0_scope(reference, rust))
        .or_else(|| reference_value_cache_kv_live_scope(reference, rust))
        .or_else(|| reference_value_mix_merged_scope(reference, rust))
    {
        return Some(scope);
    }
    let reference_sampled_token_index = reference_sampled_token_index(reference)?;
    let rust_seq = rust.seq.map(|seq| seq as u64);
    if rust_seq == Some(reference_sampled_token_index) {
        return attention_row_padded_tail_scope(reference, rust);
    }
    Some(json!({
        "reason": "reference_sampled_prompt_token_does_not_match_rust_trace_seq",
        "reference_sampled_token_index": reference_sampled_token_index,
        "reference_sample_offset": reference.sample_offset,
        "reference_token_axis": reference.token_axis,
        "rust_seq": rust_seq,
        "policy": "stage summaries with mismatched trace scope are diagnostic alignment blockers, not stable numeric divergence evidence",
    }))
}

fn reference_key_cache_head0_scope(
    reference: &ReferenceTraceRecord,
    rust: &RustTraceRecord,
) -> Option<Value> {
    if reference.stage != "k"
        || rust.stage.as_deref() != Some("attention_k_cache_head0_ref_layout_padded")
    {
        return None;
    }
    let reference_nelements = usize::try_from(reference.nelements).ok()?;
    let rust_nelements = rust.num_elements;
    let reference_values_unavailable =
        !reference.values_available || reference.first_values.is_empty();
    let reference_contains_all_kv_heads = reference_nelements > rust_nelements;
    if !reference_values_unavailable && !reference_contains_all_kv_heads {
        return None;
    }
    let reason = if reference_values_unavailable {
        "reference_key_cache_values_unavailable_for_numeric_compare"
    } else {
        "reference_key_cache_contains_all_kv_heads_rust_trace_samples_head0_reference_layout"
    };
    Some(json!({
        "reason": reason,
        "reference_nelements": reference.nelements,
        "rust_num_elements": rust.num_elements,
        "reference_values_available": reference.values_available,
        "reference_first_values_count": reference.first_values.len(),
        "compared_head0_prefix_count": rust.num_elements,
        "reference_sampled_token_index": reference_sampled_token_index(reference),
        "reference_sample_offset": reference.sample_offset,
        "reference_token_axis": reference.token_axis,
        "rust_seq": rust.seq,
        "policy": "reference cached-key tensors include all KV heads; the Rust diagnostic samples head0 in reference padded layout, so prefix deltas are layout-scope evidence, not full-cache equality claims",
    }))
}

fn reference_key_cache_kv_live_scope(
    reference: &ReferenceTraceRecord,
    rust: &RustTraceRecord,
) -> Option<Value> {
    parse_stage_head(&reference.stage, "k_kv_head")?;
    let rust_stage = rust.stage.as_deref()?;
    parse_stage_head(rust_stage, "attention_k_cache_kv_head")?;
    Some(json!({
        "reason": "reference_key_cache_live_head_token_major_not_direct_rust_dim_major_layout",
        "reference_stage": reference.stage,
        "rust_stage": rust_stage,
        "reference_shape": reference.shape,
        "rust_shape": rust.shape,
        "reference_nelements": reference.nelements,
        "rust_num_elements": rust.num_elements,
        "reference_dtype": reference.dtype,
        "rust_dtype": rust.dtype,
        "reference_values_available": reference.values_available,
        "reference_first_values_count": reference.first_values.len(),
        "rust_first_values_count": rust.first_values.len(),
        "policy": "reference key-cache live KV heads emit token-major samples while Rust diagnostic key-cache records use dim-major [head_dim, tokens] order; use attention_key_cache_dim_major_f16_roundtrip_best_matches for material evidence",
    }))
}

fn reference_value_cache_head0_scope(
    reference: &ReferenceTraceRecord,
    rust: &RustTraceRecord,
) -> Option<Value> {
    if reference.stage != "v"
        || rust.stage.as_deref() != Some("attention_v_cache_head0_ref_layout_padded")
    {
        return None;
    }
    let reference_nelements = usize::try_from(reference.nelements).ok()?;
    let rust_nelements = usize::try_from(rust.num_elements).ok()?;
    let reference_values_unavailable =
        !reference.values_available || reference.first_values.is_empty();
    let reference_contains_all_kv_heads = reference_nelements > rust_nelements;
    if !reference_values_unavailable && !reference_contains_all_kv_heads {
        return None;
    }
    let reason = if reference_values_unavailable {
        "reference_value_cache_values_unavailable_for_numeric_compare"
    } else {
        "reference_value_cache_contains_all_kv_heads_rust_trace_samples_head0_reference_layout"
    };
    Some(json!({
        "reason": reason,
        "reference_nelements": reference.nelements,
        "rust_num_elements": rust.num_elements,
        "reference_values_available": reference.values_available,
        "reference_first_values_count": reference.first_values.len(),
        "compared_head0_prefix_count": rust.num_elements,
        "reference_sampled_token_index": reference_sampled_token_index(reference),
        "reference_sample_offset": reference.sample_offset,
        "reference_token_axis": reference.token_axis,
        "rust_seq": rust.seq,
        "policy": "reference cached-value tensors include all KV heads; the Rust diagnostic samples head0 in reference key-padded layout, so prefix deltas are head0 evidence but not full-cache equality claims",
    }))
}

fn reference_value_cache_kv_live_scope(
    reference: &ReferenceTraceRecord,
    rust: &RustTraceRecord,
) -> Option<Value> {
    parse_stage_head(&reference.stage, "v_kv_head")?;
    let rust_stage = rust.stage.as_deref()?;
    parse_stage_head(rust_stage, "attention_v_cache_kv_head")?;
    let reference_shape0 = reference.shape.first().copied();
    let rust_shape0 = rust.shape.first().copied().map(|dim| dim as i64);
    let element_count_match = reference.nelements == rust.num_elements as u64;
    let head_dim_match =
        reference_shape0.is_some() && rust_shape0.is_some() && reference_shape0 == rust_shape0;
    if element_count_match && head_dim_match {
        return None;
    }
    Some(json!({
        "reason": "reference_value_cache_live_head_layout_not_direct_rust_kv_head_layout",
        "reference_stage": reference.stage,
        "rust_stage": rust_stage,
        "reference_shape": reference.shape,
        "rust_shape": rust.shape,
        "reference_nelements": reference.nelements,
        "rust_num_elements": rust.num_elements,
        "reference_dtype": reference.dtype,
        "rust_dtype": rust.dtype,
        "reference_values_available": reference.values_available,
        "reference_first_values_count": reference.first_values.len(),
        "rust_first_values_count": rust.first_values.len(),
        "policy": "reference value-cache live KV heads use the llama.cpp cache representation and are diagnostic layout evidence, not direct Rust KV-head numeric parity evidence until the value-cache layout transform is named",
    }))
}

fn reference_value_mix_merged_scope(
    reference: &ReferenceTraceRecord,
    rust: &RustTraceRecord,
) -> Option<Value> {
    if reference.stage != "kqv_merged"
        || rust.stage.as_deref() != Some("attention_value_mix_merged")
    {
        return None;
    }
    let reference_nelements = usize::try_from(reference.nelements).ok()?;
    let rust_nelements = usize::try_from(rust.num_elements).ok()?;
    let reference_values_unavailable =
        !reference.values_available || reference.first_values.is_empty();
    let reference_contains_all_tokens = reference_nelements > rust_nelements;
    if !reference_values_unavailable && !reference_contains_all_tokens {
        return None;
    }
    Some(json!({
        "reason": "reference_value_mix_merged_noncontiguous_all_tokens_scope_not_direct_numeric_compare",
        "reference_nelements": reference.nelements,
        "rust_num_elements": rust.num_elements,
        "reference_values_available": reference.values_available,
        "reference_first_values_count": reference.first_values.len(),
        "reference_sampled_token_index": reference_sampled_token_index(reference),
        "reference_sample_offset": reference.sample_offset,
        "reference_token_axis": reference.token_axis,
        "rust_seq": rust.seq,
        "policy": "reference kqv_merged is a non-contiguous all-token/all-head view in this trace; compare the contiguous attn_value_mix record against Rust attention_value_mix_merged for material evidence",
    }))
}

fn attention_row_padded_tail_scope(
    reference: &ReferenceTraceRecord,
    rust: &RustTraceRecord,
) -> Option<Value> {
    if !is_attention_row_stage(&reference.stage) {
        return None;
    }
    let reference_nelements = usize::try_from(reference.nelements).ok()?;
    let rust_nelements = usize::try_from(rust.num_elements).ok()?;
    if reference_nelements <= rust_nelements || reference.first_values.len() <= rust_nelements {
        return None;
    }
    let tail = &reference.first_values[rust_nelements..];
    if tail.iter().any(|value| value.abs() > 1.0e-12) {
        return None;
    }
    Some(json!({
        "reason": "reference_attention_row_includes_padded_zero_tail_not_emitted_by_rust",
        "reference_nelements": reference.nelements,
        "rust_num_elements": rust.num_elements,
        "compared_live_prefix_count": rust.num_elements,
        "reference_zero_tail_count": reference_nelements.saturating_sub(rust_nelements),
        "reference_sampled_zero_tail_count": tail.len(),
        "reference_sampled_token_index": reference_sampled_token_index(reference),
        "reference_sample_offset": reference.sample_offset,
        "reference_token_axis": reference.token_axis,
        "rust_seq": rust.seq,
        "policy": "attention score/probability row summaries with reference KV-cache padding are diagnostic scope differences; live-prefix deltas remain useful evidence",
    }))
}

fn is_attention_row_stage(stage: &str) -> bool {
    stage == "kq"
        || stage == "kq_soft_max_ext"
        || parse_stage_head(stage, "kq_head").is_some()
        || parse_stage_head(stage, "kq_soft_max_ext_head").is_some()
}

fn reference_sampled_token_index(record: &ReferenceTraceRecord) -> Option<u64> {
    let axis = record.token_axis?;
    if axis < 0 {
        return None;
    }
    let axis = usize::try_from(axis).ok()?;
    let shape = if record.full_shape.is_empty() { &record.shape } else { &record.full_shape };
    if axis >= shape.len() {
        return None;
    }
    let offset = record.sample_offset?;
    let stride = shape.iter().take(axis).try_fold(1u64, |acc, dim| {
        let dim = u64::try_from(*dim).ok()?;
        acc.checked_mul(dim)
    })?;
    if stride == 0 || offset % stride != 0 {
        return None;
    }
    let axis_dim = u64::try_from(shape[axis]).ok()?;
    if axis_dim == 0 {
        return None;
    }
    Some((offset / stride) % axis_dim)
}

fn reference_record_summary(record: &ReferenceTraceRecord) -> Value {
    json!({
        "name": record.name,
        "stage": record.stage,
        "graph_index": record.graph_index,
        "layer": record.layer,
        "graph_op": record.graph_op,
        "graph_sources": record.graph_sources,
        "view_source": record.view_source,
        "view_offset": record.view_offset,
        "full_shape": record.full_shape,
        "sample_offset": record.sample_offset,
        "token_axis": record.token_axis,
        "sampled_token_index": reference_sampled_token_index(record),
        "shape": record.shape,
        "dtype": record.dtype,
        "nelements": record.nelements,
        "rms": record.rms,
        "first_values": record.first_values,
        "values_available": record.values_available,
    })
}

fn rust_record_summary(record: &RustTraceRecord) -> Value {
    json!({
        "name": record.name,
        "stage": record.stage,
        "seq": record.seq,
        "layer": record.layer,
        "shape": record.shape,
        "dtype": record.dtype,
        "num_elements": record.num_elements,
        "rms": record.rms,
        "first_values": record.first_values,
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

    let mut stage_mapping = vec![
        json!({"reference": "inp_embd", "rust": "embeddings", "scope": "prompt embedding"}),
        json!({"reference": "attn_norm", "rust": "attn_norm", "scope": "layer0"}),
        json!({"reference": "Qcur", "rust": "attention_q", "scope": "layer0"}),
        json!({"reference": "Kcur", "rust": "attention_k", "scope": "layer0"}),
        json!({"reference": "Vcur", "rust": "attention_v", "scope": "layer0"}),
        json!({"reference": "kq", "rust": "attention_scores_raw_head0", "scope": "layer0 head0 sampled-query scores before scale/mask"}),
        json!({"reference": "kq_soft_max_ext", "rust": "attn_scores_softmax_head0", "scope": "layer0 head0 sampled-query probabilities"}),
        json!({"reference": "k", "rust": "attention_k_cache_head0_ref_layout_padded", "scope": "layer0 head0 cached key matrix in reference padded layout"}),
        json!({"reference": "v", "rust": "attention_v_cache_head0_ref_layout_padded", "scope": "layer0 head0 cached value matrix in reference padded layout"}),
        json!({"reference": "kqv", "rust": "attention_value_mix_head0", "scope": "layer0 head0 value-mix output before head merge"}),
    ];
    for head in 0..20 {
        stage_mapping.push(json!({
            "reference": format!("kq_head{head}"),
            "rust": format!("attention_scores_raw_head{head}"),
            "scope": format!("layer0 sampled-query score row head{head}"),
        }));
    }
    for head in 0..20 {
        stage_mapping.push(json!({
            "reference": format!("kq_soft_max_ext_head{head}"),
            "rust": format!("attn_scores_softmax_head{head}"),
            "scope": format!("layer0 sampled-query probability row head{head}"),
        }));
    }
    for kv_head in 0..5 {
        stage_mapping.push(json!({
            "reference": format!("k_kv_head{kv_head}_live"),
            "rust": format!("attention_k_cache_kv_head{kv_head}_live_ref_layout"),
            "scope": format!("layer0 live key-cache matrix KV head{kv_head} in reference layout"),
        }));
    }
    for kv_head in 0..5 {
        stage_mapping.push(json!({
            "reference": format!("v_kv_head{kv_head}_live"),
            "rust": format!("attention_v_cache_kv_head{kv_head}_live_ref_layout"),
            "scope": format!("layer0 live value-cache matrix KV head{kv_head} in reference layout"),
        }));
    }
    for kv_head in 0..5 {
        stage_mapping.push(json!({
            "reference": format!("v_cache_rust_layout_head{kv_head}_live"),
            "rust": format!("attention_v_cache_f16_roundtrip_kv_head{kv_head}_live_ref_layout"),
            "scope": format!("layer0 live value-cache matrix KV head{kv_head} transposed into Rust diagnostic layout after Rust F16 roundtrip"),
        }));
    }
    for head in 0..20 {
        stage_mapping.push(json!({
            "reference": format!("kqv_head{head}"),
            "rust": format!("attention_value_mix_head{head}"),
            "scope": format!("layer0 value-mix sampled token head{head}"),
        }));
    }
    stage_mapping.extend([
        json!({"reference": "kqv_merged", "rust": "attention_value_mix_merged", "scope": "layer0 non-contiguous all-token value-mix merge view"}),
        json!({"reference": "attn_value_mix", "rust": "attention_value_mix_merged", "scope": "layer0 contiguous merged value-mix before subnorm"}),
        json!({"reference": "attn_sub_norm", "rust": "post_attention_subnorm", "scope": "layer0"}),
        json!({"reference": "attn_o_out", "rust": "post_o_proj", "scope": "layer0"}),
        json!({"reference": "ffn_inp", "rust": "post_attention_residual", "scope": "layer0"}),
        json!({"reference": "ffn_norm", "rust": "post_ffn_norm", "scope": "layer0"}),
        json!({"reference": "ffn_out", "rust": "post_swiglu", "scope": "layer0"}),
        json!({"reference": "ffn_sub_norm", "rust": "post_ffn_subnorm", "scope": "layer0"}),
        json!({"reference": "ffn_down", "rust": "post_down_proj", "scope": "layer0"}),
        json!({"reference": "l_out", "rust": "post_layer", "scope": "layer0"}),
        json!({"reference": "result_norm", "rust": "final_norm", "scope": "final token"}),
        json!({"reference": "result_output", "rust": "logits", "scope": "final token"}),
    ]);

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
        "stage_mapping": stage_mapping,
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

fn preferred_rust_trace_argv(
    plan: &Value,
    preferred_key: &'static str,
    fallback_key: &'static str,
) -> (Option<Vec<String>>, Option<&'static str>) {
    if let Ok(argv) = rust_command_argv(plan, preferred_key) {
        return (Some(argv), Some(preferred_key));
    }
    match rust_command_argv(plan, fallback_key) {
        Ok(argv) => (Some(argv), Some(fallback_key)),
        Err(_) => (None, None),
    }
}

fn rust_trace_target_seq_from_plan(plan: &Value) -> Option<usize> {
    let token_count =
        plan.pointer("/prompt_identity/prompt_token_count").and_then(Value::as_u64)?;
    usize::try_from(token_count.checked_sub(1)?).ok()
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

fn run_rust_trace_capture(
    label: &str,
    argv: &[String],
    trace_dir: &Path,
    trace_target_seq: Option<usize>,
) -> Result<Value> {
    let executable = argv.first().context("empty Rust trace command")?;
    let mut command = Command::new(executable);
    command
        .args(&argv[1..])
        .env("BITNET_TRACE_DIR", trace_dir)
        .env("BITNET_DETERMINISTIC", "1")
        .env("BITNET_SEED", "0")
        .stdin(Stdio::null());
    if let Some(trace_target_seq) = trace_target_seq {
        command.env("BITNET_TRACE_TARGET_SEQ", trace_target_seq.to_string());
    }
    let capture = run_command(&mut command)?;
    let trace = summarize_rust_trace_dir(trace_dir);
    Ok(json!({
        "label": label,
        "attempted": true,
        "trace_dir": path_to_string(trace_dir),
        "trace_target_seq": trace_target_seq,
        "trace_target_source": trace_target_seq.map(|_| "prompt_identity.prompt_token_count_minus_one"),
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
            if receipt_type == "bitnet_reference_embedding_row_authority" {
                let ready = report
                    .pointer("/decision/embedding_row_authority_ready")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let ref_match = report
                    .pointer("/decision/reference_row_matches_trace_sample")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let rust_match = report
                    .pointer("/decision/rust_loaded_matches_reference_row")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let reasons = report
                    .pointer("/decision/current_blocked_reasons")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                println!(
                    "bitnet reference embedding row authority: diagnostic_only=true claim_allowed=false ready={ready} reference_trace_match={ref_match} rust_loaded_match={rust_match}"
                );
                if !reasons.is_empty() {
                    println!("blocked_reasons:");
                    for reason in reasons {
                        println!("  - {}", reason.as_str().unwrap_or("<non-string>"));
                    }
                }
                return Ok(());
            }
            if receipt_type == "bitnet_reference_attn_output_same_input_parity" {
                let available = report
                    .pointer("/decision/same_input_projection_available")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let matches = report
                    .pointer("/decision/same_input_projection_matches_reference")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let max_delta = report
                    .pointer("/projection/rust_same_input_vs_reference_target/max_abs_delta")
                    .and_then(Value::as_f64);
                let reasons = report
                    .pointer("/decision/current_blocked_reasons")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                println!(
                    "bitnet reference attn-output same-input parity: diagnostic_only=true claim_allowed=false available={available} matches_reference={matches} max_abs_delta={}",
                    max_delta.map(|value| value.to_string()).unwrap_or_else(|| "n/a".to_string())
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
            "first_values": [rms as f32, 0.0],
            "seq": 0,
            "layer": 0,
            "stage": stage,
        });
        fs::write(path, serde_json::to_string_pretty(&record).unwrap()).unwrap();
    }

    fn test_reference_trace_record(stage: &str, first_values: Vec<f32>) -> ReferenceTraceRecord {
        ReferenceTraceRecord {
            name: format!("{stage}-0"),
            stage: stage.to_string(),
            graph_index: Some(0),
            layer: Some(0),
            graph_op: Some("MUL_MAT".to_string()),
            graph_sources: json!([]),
            view_source: Value::Null,
            view_offset: Some(0),
            full_shape: vec![first_values.len() as i64, 1, 1, 1],
            sample_offset: Some(0),
            token_axis: Some(1),
            dtype: "f32".to_string(),
            shape: vec![first_values.len() as i64, 1, 1, 1],
            nelements: first_values.len() as u64,
            rms: Some(sample_rms(&first_values)),
            values_available: true,
            first_values,
        }
    }

    fn test_rust_trace_record(stage: &str, first_values: Vec<f32>) -> RustTraceRecord {
        RustTraceRecord {
            name: format!("t0/blk0/{stage}"),
            shape: vec![first_values.len()],
            dtype: "F32".to_string(),
            blake3: "abc".to_string(),
            rms: sample_rms(&first_values),
            num_elements: first_values.len(),
            first_values,
            seq: Some(0),
            layer: Some(0),
            stage: Some(stage.to_string()),
        }
    }

    fn sample_rms(values: &[f32]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let sum_sq = values.iter().map(|value| (*value as f64) * (*value as f64)).sum::<f64>();
        (sum_sq / values.len() as f64).sqrt()
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
        let stage_mapping = report.pointer("/stage_mapping").and_then(Value::as_array).unwrap();
        assert!(stage_mapping.iter().any(|entry| {
            entry.pointer("/reference") == Some(&json!("kq_head19"))
                && entry.pointer("/rust") == Some(&json!("attention_scores_raw_head19"))
        }));
        assert!(stage_mapping.iter().any(|entry| {
            entry.pointer("/reference") == Some(&json!("kq_soft_max_ext_head19"))
                && entry.pointer("/rust") == Some(&json!("attn_scores_softmax_head19"))
        }));
        assert!(stage_mapping.iter().any(|entry| {
            entry.pointer("/reference") == Some(&json!("k_kv_head4_live"))
                && entry.pointer("/rust")
                    == Some(&json!("attention_k_cache_kv_head4_live_ref_layout"))
        }));
        assert!(stage_mapping.iter().any(|entry| {
            entry.pointer("/reference") == Some(&json!("v_kv_head4_live"))
                && entry.pointer("/rust")
                    == Some(&json!("attention_v_cache_kv_head4_live_ref_layout"))
        }));
        assert!(stage_mapping.iter().any(|entry| {
            entry.pointer("/reference") == Some(&json!("v_cache_rust_layout_head4_live"))
                && entry.pointer("/rust")
                    == Some(&json!("attention_v_cache_f16_roundtrip_kv_head4_live_ref_layout"))
        }));
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
    fn rust_trace_capture_prefers_first_token_command_and_targets_last_prompt_token() {
        let plan = json!({
            "prompt_identity": {
                "prompt_token_count": 18
            },
            "rust_commands": {
                "cpu_argv": ["cargo", "run", "--features", "cpu"],
                "cpu_first_token_logit_argv": ["cargo", "run", "--features", "cpu", "--", "--max-new-tokens", "1"]
            }
        });

        let (argv, key) =
            preferred_rust_trace_argv(&plan, "cpu_first_token_logit_argv", "cpu_argv");

        assert_eq!(key, Some("cpu_first_token_logit_argv"));
        assert_eq!(
            argv.unwrap(),
            vec![
                "cargo".to_string(),
                "run".to_string(),
                "--features".to_string(),
                "cpu".to_string(),
                "--".to_string(),
                "--max-new-tokens".to_string(),
                "1".to_string()
            ]
        );
        assert_eq!(rust_trace_target_seq_from_plan(&plan), Some(17));
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
                        "first_values": [1.0, 0.0],
                        "values_available": true,
                        "stats": {"rms": 1.0}
                    },
                    {
                        "name": "attn_norm-0",
                        "stage": "attn_norm",
                        "graph_index": 2,
                        "layer": 0,
                        "graph_op": "RMS_NORM",
                        "graph_sources": [
                            {
                                "name": "inp_embd",
                                "op": "GET_ROWS",
                                "dtype": "f32",
                                "shape": [2, 2, 1, 1],
                                "nelements": 4
                            }
                        ],
                        "dtype": "f32",
                        "shape": [2, 2, 1, 1],
                        "nelements": 4,
                        "first_values": [2.0, 0.0],
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
        assert_eq!(
            report.pointer("/cpu/first_material_mismatch/reference/graph_op"),
            Some(&json!("RMS_NORM"))
        );
        assert_eq!(
            report.pointer("/cpu/first_material_mismatch/reference/graph_sources/0/name"),
            Some(&json!("inp_embd"))
        );
        assert_eq!(
            report.pointer("/cpu/first_material_mismatch/first_values_delta/max_abs_delta"),
            Some(&json!(1.0))
        );
        assert!(report.pointer("/decision/current_blocked_reasons").unwrap().is_array());
    }

    #[test]
    fn compare_reports_scope_mismatch_before_material_mismatch() {
        let reference = ReferenceTraceRecord {
            name: "inp_embd".to_string(),
            stage: "inp_embd".to_string(),
            graph_index: Some(0),
            layer: Some(-1),
            graph_op: Some("GET_ROWS".to_string()),
            graph_sources: json!([]),
            view_source: Value::Null,
            view_offset: Some(0),
            full_shape: vec![2, 2, 1, 1],
            sample_offset: Some(2),
            token_axis: Some(1),
            dtype: "f32".to_string(),
            shape: vec![2, 1, 1, 1],
            nelements: 2,
            rms: Some(1.0),
            values_available: true,
            first_values: vec![1.0, 2.0],
        };
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "embeddings".to_string(),
            RustTraceRecord {
                name: "t0/embeddings".to_string(),
                shape: vec![1, 1, 2],
                dtype: "F32".to_string(),
                blake3: "abc".to_string(),
                rms: 1.0,
                num_elements: 2,
                first_values: vec![1.0, 2.0],
                seq: Some(0),
                layer: Some(-1),
                stage: Some("embeddings".to_string()),
            },
        );

        let report =
            compare_reference_to_rust(&[reference], &rust_records, &[("inp_embd", "embeddings")]);

        assert_eq!(report.pointer("/scope_mismatch_count"), Some(&json!(1)));
        assert_eq!(report.pointer("/material_mismatch_count"), Some(&json!(0)));
        assert_eq!(report.pointer("/first_scope_mismatch/status"), Some(&json!("scope_mismatch")));
        assert_eq!(
            report.pointer("/first_scope_mismatch/scope/reference_sampled_token_index"),
            Some(&json!(1))
        );
        assert_eq!(report.pointer("/first_scope_mismatch/scope/rust_seq"), Some(&json!(0)));
        assert!(report.pointer("/first_material_mismatch").unwrap().is_null());
    }

    #[test]
    fn compare_marks_reference_attention_score_padding_as_scope_evidence() {
        let reference = ReferenceTraceRecord {
            name: "kq-0".to_string(),
            stage: "kq".to_string(),
            graph_index: Some(40),
            layer: Some(0),
            graph_op: Some("MUL_MAT".to_string()),
            graph_sources: json!([]),
            view_source: Value::Null,
            view_offset: Some(0),
            full_shape: vec![4, 3, 1, 1],
            sample_offset: Some(8),
            token_axis: Some(1),
            dtype: "f32".to_string(),
            shape: vec![4, 1, 1, 1],
            nelements: 4,
            rms: Some(2.0),
            values_available: true,
            first_values: vec![1.0, 2.0, 3.0, 0.0],
        };
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_scores_raw_head0".to_string(),
            RustTraceRecord {
                name: "t2/blk0/attention_scores_raw_head0".to_string(),
                shape: vec![1, 1, 1, 3],
                dtype: "F32".to_string(),
                blake3: "abc".to_string(),
                rms: 2.5,
                num_elements: 3,
                first_values: vec![1.0, 2.0, 3.0],
                seq: Some(2),
                layer: Some(0),
                stage: Some("attention_scores_raw_head0".to_string()),
            },
        );

        let report = compare_reference_to_rust(
            &[reference],
            &rust_records,
            &[("kq", "attention_scores_raw_head0")],
        );

        assert_eq!(report.pointer("/scope_mismatch_count"), Some(&json!(1)));
        assert_eq!(report.pointer("/material_mismatch_count"), Some(&json!(0)));
        assert_eq!(
            report.pointer("/first_scope_mismatch/scope/reason"),
            Some(&json!("reference_attention_row_includes_padded_zero_tail_not_emitted_by_rust"))
        );
        assert_eq!(
            report.pointer("/first_scope_mismatch/first_values_delta/max_abs_delta"),
            Some(&json!(0.0))
        );
        assert!(report.pointer("/first_material_mismatch").unwrap().is_null());
    }

    #[test]
    fn compare_marks_synthetic_attention_head_padding_as_scope_evidence() {
        let reference = ReferenceTraceRecord {
            name: "kq_head7-0".to_string(),
            stage: "kq_head7".to_string(),
            graph_index: Some(40),
            layer: Some(0),
            graph_op: Some("MUL_MAT".to_string()),
            graph_sources: json!([]),
            view_source: Value::Null,
            view_offset: Some(0),
            full_shape: vec![4, 3, 20, 1],
            sample_offset: Some(4 * 2 + 4 * 3 * 7),
            token_axis: Some(1),
            dtype: "f32".to_string(),
            shape: vec![4, 1, 1, 1],
            nelements: 4,
            rms: Some(2.0),
            values_available: true,
            first_values: vec![1.0, 2.0, 3.0, 0.0],
        };
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_scores_raw_head7".to_string(),
            RustTraceRecord {
                name: "t2/blk0/attention_scores_raw_head7".to_string(),
                shape: vec![1, 1, 1, 3],
                dtype: "F32".to_string(),
                blake3: "abc".to_string(),
                rms: 2.5,
                num_elements: 3,
                first_values: vec![1.0, 2.0, 3.0],
                seq: Some(2),
                layer: Some(0),
                stage: Some("attention_scores_raw_head7".to_string()),
            },
        );

        let report = compare_reference_to_rust(
            &[reference],
            &rust_records,
            &[("kq_head7", "attention_scores_raw_head7")],
        );

        assert_eq!(report.pointer("/scope_mismatch_count"), Some(&json!(1)));
        assert_eq!(report.pointer("/material_mismatch_count"), Some(&json!(0)));
        assert_eq!(
            report.pointer("/first_scope_mismatch/scope/reference_sampled_token_index"),
            Some(&json!(2))
        );
        assert_eq!(
            report.pointer("/first_scope_mismatch/scope/reason"),
            Some(&json!("reference_attention_row_includes_padded_zero_tail_not_emitted_by_rust"))
        );
        assert!(report.pointer("/first_material_mismatch").unwrap().is_null());
    }

    #[test]
    fn compare_marks_reference_value_cache_all_heads_as_scope_evidence() {
        let reference = ReferenceTraceRecord {
            name: "v-0".to_string(),
            stage: "v".to_string(),
            graph_index: Some(43),
            layer: Some(0),
            graph_op: Some("VIEW".to_string()),
            graph_sources: json!([]),
            view_source: Value::Null,
            view_offset: Some(0),
            full_shape: vec![4, 2, 2, 1],
            sample_offset: Some(0),
            token_axis: Some(-1),
            dtype: "f32".to_string(),
            shape: vec![4, 2, 2, 1],
            nelements: 16,
            rms: Some(3.0),
            values_available: true,
            first_values: vec![1.0, 2.0, 3.0, 4.0],
        };
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_v_cache_head0_ref_layout_padded".to_string(),
            RustTraceRecord {
                name: "t0/blk0/attention_v_cache_head0_ref_layout_padded".to_string(),
                shape: vec![2, 4],
                dtype: "F32".to_string(),
                blake3: "abc".to_string(),
                rms: 2.5,
                num_elements: 8,
                first_values: vec![1.0, 2.0, 3.0, 4.0],
                seq: Some(0),
                layer: Some(0),
                stage: Some("attention_v_cache_head0_ref_layout_padded".to_string()),
            },
        );

        let report = compare_reference_to_rust(
            &[reference],
            &rust_records,
            &[("v", "attention_v_cache_head0_ref_layout_padded")],
        );

        assert_eq!(report.pointer("/scope_mismatch_count"), Some(&json!(1)));
        assert_eq!(report.pointer("/material_mismatch_count"), Some(&json!(0)));
        assert_eq!(
            report.pointer("/first_scope_mismatch/scope/reason"),
            Some(&json!(
                "reference_value_cache_contains_all_kv_heads_rust_trace_samples_head0_reference_layout"
            ))
        );
        assert_eq!(
            report.pointer("/first_scope_mismatch/first_values_delta/max_abs_delta"),
            Some(&json!(0.0))
        );
        assert!(report.pointer("/first_material_mismatch").unwrap().is_null());
    }

    #[test]
    fn compare_marks_reference_value_cache_unavailable_values_as_scope_evidence() {
        let reference = ReferenceTraceRecord {
            name: "v-0".to_string(),
            stage: "v".to_string(),
            graph_index: Some(43),
            layer: Some(0),
            graph_op: Some("VIEW".to_string()),
            graph_sources: json!([]),
            view_source: Value::Null,
            view_offset: Some(0),
            full_shape: vec![4, 2, 2, 1],
            sample_offset: Some(0),
            token_axis: Some(-1),
            dtype: "f16".to_string(),
            shape: vec![4, 2, 2, 1],
            nelements: 16,
            rms: None,
            values_available: false,
            first_values: Vec::new(),
        };
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_v_cache_head0_ref_layout_padded".to_string(),
            RustTraceRecord {
                name: "t0/blk0/attention_v_cache_head0_ref_layout_padded".to_string(),
                shape: vec![2, 4],
                dtype: "F32".to_string(),
                blake3: "abc".to_string(),
                rms: 2.5,
                num_elements: 8,
                first_values: vec![1.0, 2.0, 3.0, 4.0],
                seq: Some(0),
                layer: Some(0),
                stage: Some("attention_v_cache_head0_ref_layout_padded".to_string()),
            },
        );

        let report = compare_reference_to_rust(
            &[reference],
            &rust_records,
            &[("v", "attention_v_cache_head0_ref_layout_padded")],
        );

        assert_eq!(report.pointer("/scope_mismatch_count"), Some(&json!(1)));
        assert_eq!(report.pointer("/material_mismatch_count"), Some(&json!(0)));
        assert_eq!(
            report.pointer("/first_scope_mismatch/scope/reason"),
            Some(&json!("reference_value_cache_values_unavailable_for_numeric_compare"))
        );
        assert_eq!(report.pointer("/first_scope_mismatch/first_values_delta"), Some(&Value::Null));
        assert!(report.pointer("/first_material_mismatch").unwrap().is_null());
    }

    #[test]
    fn compare_marks_reference_value_mix_merged_view_as_scope_evidence() {
        let reference = ReferenceTraceRecord {
            name: "kqv_merged-0".to_string(),
            stage: "kqv_merged".to_string(),
            graph_index: Some(45),
            layer: Some(0),
            graph_op: Some("PERMUTE".to_string()),
            graph_sources: json!([]),
            view_source: Value::Null,
            view_offset: None,
            full_shape: vec![4, 2, 2, 1],
            sample_offset: None,
            token_axis: None,
            dtype: "f32".to_string(),
            shape: vec![4, 2, 2, 1],
            nelements: 16,
            rms: None,
            values_available: false,
            first_values: Vec::new(),
        };
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_value_mix_merged".to_string(),
            RustTraceRecord {
                name: "t0/blk0/attention_value_mix_merged".to_string(),
                shape: vec![1, 1, 8],
                dtype: "F32".to_string(),
                blake3: "abc".to_string(),
                rms: 2.5,
                num_elements: 8,
                first_values: vec![1.0, 2.0, 3.0, 4.0],
                seq: Some(0),
                layer: Some(0),
                stage: Some("attention_value_mix_merged".to_string()),
            },
        );

        let report = compare_reference_to_rust(
            &[reference],
            &rust_records,
            &[("kqv_merged", "attention_value_mix_merged")],
        );

        assert_eq!(report.pointer("/scope_mismatch_count"), Some(&json!(1)));
        assert_eq!(report.pointer("/material_mismatch_count"), Some(&json!(0)));
        assert_eq!(
            report.pointer("/first_scope_mismatch/scope/reason"),
            Some(&json!(
                "reference_value_mix_merged_noncontiguous_all_tokens_scope_not_direct_numeric_compare"
            ))
        );
        assert!(report.pointer("/first_material_mismatch").unwrap().is_null());
    }

    #[test]
    fn reference_sampled_token_index_ignores_later_head_stride() {
        let record = ReferenceTraceRecord {
            name: "kqv_head19-0".to_string(),
            stage: "kqv_head19".to_string(),
            graph_index: Some(45),
            layer: Some(0),
            graph_op: Some("MUL_MAT".to_string()),
            graph_sources: json!([]),
            view_source: Value::Null,
            view_offset: Some(0),
            full_shape: vec![128, 18, 20, 1],
            sample_offset: Some(128 * 17 + 128 * 18 * 19),
            token_axis: Some(1),
            dtype: "f32".to_string(),
            shape: vec![128, 1, 1, 1],
            nelements: 128,
            rms: Some(3.0),
            values_available: true,
            first_values: vec![1.0],
        };

        assert_eq!(reference_sampled_token_index(&record), Some(17));
    }

    #[test]
    fn compare_maps_reference_attention_value_mix_to_rust_trace() {
        let reference = ReferenceTraceRecord {
            name: "attn_value_mix-0".to_string(),
            stage: "attn_value_mix".to_string(),
            graph_index: Some(42),
            layer: Some(0),
            graph_op: Some("MUL_MAT".to_string()),
            graph_sources: json!([]),
            view_source: Value::Null,
            view_offset: Some(0),
            full_shape: vec![2, 2, 1, 1],
            sample_offset: None,
            token_axis: None,
            dtype: "f32".to_string(),
            shape: vec![2, 2, 1, 1],
            nelements: 4,
            rms: Some(3.0),
            values_available: true,
            first_values: vec![1.0, 2.0, 3.0, 4.0],
        };
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_value_mix".to_string(),
            RustTraceRecord {
                name: "t0/blk0/attention_value_mix".to_string(),
                shape: vec![2, 2],
                dtype: "F32".to_string(),
                blake3: "abc".to_string(),
                rms: 2.5,
                num_elements: 4,
                first_values: vec![1.0, 2.5, 3.0, 4.0],
                seq: Some(0),
                layer: Some(0),
                stage: Some("attention_value_mix".to_string()),
            },
        );

        let report = compare_reference_to_rust(
            &[reference],
            &rust_records,
            &[("attn_value_mix", "attention_value_mix")],
        );

        assert_eq!(
            report.pointer("/first_material_mismatch/reference_stage"),
            Some(&json!("attn_value_mix"))
        );
        assert_eq!(
            report.pointer("/first_material_mismatch/rust_stage"),
            Some(&json!("attention_value_mix"))
        );
        assert_eq!(
            report.pointer("/first_material_mismatch/first_values_delta/max_abs_delta"),
            Some(&json!(0.5))
        );
        assert_eq!(report.pointer("/scope_mismatch_count"), Some(&json!(0)));
    }

    #[test]
    fn compare_reports_attention_value_mix_head_lane_best_matches() {
        let reference_records = vec![
            test_reference_trace_record("kqv_head0", vec![1.0, 2.0, 3.0]),
            test_reference_trace_record("kqv_head1", vec![8.0, 8.0, 8.0]),
        ];
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_value_mix_head0".to_string(),
            test_rust_trace_record("attention_value_mix_head0", vec![1.0, 2.0, 3.0]),
        );
        rust_records.insert(
            "attention_value_mix_head1".to_string(),
            test_rust_trace_record("attention_value_mix_head1", vec![0.0, 0.0, 0.0]),
        );
        rust_records.insert(
            "attention_value_mix_head2".to_string(),
            test_rust_trace_record("attention_value_mix_head2", vec![8.0, 8.0, 8.0]),
        );
        rust_records.insert(
            "attention_value_mix_f16_cache_head0".to_string(),
            test_rust_trace_record("attention_value_mix_f16_cache_head0", vec![1.0, 2.0, 3.0]),
        );
        rust_records.insert(
            "attention_value_mix_f16_cache_head1".to_string(),
            test_rust_trace_record("attention_value_mix_f16_cache_head1", vec![0.0, 0.0, 0.0]),
        );
        rust_records.insert(
            "attention_value_mix_f16_cache_head2".to_string(),
            test_rust_trace_record("attention_value_mix_f16_cache_head2", vec![8.0, 8.0, 8.0]),
        );

        let report = compare_reference_to_rust(
            &reference_records,
            &rust_records,
            &[("kqv_head0", "attention_value_mix_head0")],
        );
        let matches = report.pointer("/attention_value_mix_head_lane_best_matches").unwrap();

        assert_eq!(matches.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(matches.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(matches.pointer("/reference_head_count"), Some(&json!(2)));
        assert_eq!(matches.pointer("/rust_head_count"), Some(&json!(3)));
        assert_eq!(matches.pointer("/identity_best_count"), Some(&json!(1)));
        assert_eq!(matches.pointer("/non_identity_best_count"), Some(&json!(1)));
        assert_eq!(matches.pointer("/all_identity_best"), Some(&json!(false)));
        assert_eq!(matches.pointer("/rows/0/reference_head"), Some(&json!(0)));
        assert_eq!(matches.pointer("/rows/0/best_rust_head"), Some(&json!(0)));
        assert_eq!(matches.pointer("/rows/0/identity_is_best"), Some(&json!(true)));
        assert_eq!(matches.pointer("/rows/1/reference_head"), Some(&json!(1)));
        assert_eq!(matches.pointer("/rows/1/best_rust_head"), Some(&json!(2)));
        assert_eq!(matches.pointer("/rows/1/identity_is_best"), Some(&json!(false)));
        assert_eq!(matches.pointer("/rows/1/identity_rank"), Some(&json!(3)));

        let f16_matches =
            report.pointer("/attention_value_mix_f16_cache_head_lane_best_matches").unwrap();
        assert_eq!(f16_matches.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(f16_matches.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(f16_matches.pointer("/reference_stage_prefix"), Some(&json!("kqv_head")));
        assert_eq!(
            f16_matches.pointer("/rust_stage_prefix"),
            Some(&json!("attention_value_mix_f16_cache_head"))
        );
        assert_eq!(f16_matches.pointer("/identity_best_count"), Some(&json!(1)));
        assert_eq!(f16_matches.pointer("/non_identity_best_count"), Some(&json!(1)));
        assert_eq!(f16_matches.pointer("/rows/1/best_rust_head"), Some(&json!(2)));
    }

    #[test]
    fn compare_reports_reference_scalar_value_mix_recompute() {
        let reference_records = vec![
            ReferenceTraceRecord {
                shape: vec![2, 1, 1, 1],
                nelements: 2,
                first_values: vec![0.25, 0.75],
                ..test_reference_trace_record("kq_soft_max_ext_head0", vec![0.25, 0.75])
            },
            ReferenceTraceRecord {
                shape: vec![2, 1, 1, 1],
                nelements: 2,
                first_values: vec![0.5, 0.5],
                ..test_reference_trace_record("kq_soft_max_ext_head1", vec![0.5, 0.5])
            },
            ReferenceTraceRecord {
                shape: vec![3, 2, 1, 1],
                nelements: 6,
                token_axis: Some(-1),
                first_values: vec![2.0, 4.0, 10.0, 14.0, -2.0, 2.0],
                ..test_reference_trace_record(
                    "v_cache_rust_layout_head0_live",
                    vec![2.0, 4.0, 10.0, 14.0, -2.0, 2.0],
                )
            },
            ReferenceTraceRecord {
                shape: vec![3, 1, 1, 1],
                nelements: 3,
                first_values: vec![3.5, 13.0, 1.0],
                ..test_reference_trace_record("kqv_head0", vec![3.5, 13.0, 1.0])
            },
            ReferenceTraceRecord {
                shape: vec![3, 1, 1, 1],
                nelements: 3,
                first_values: vec![3.0, 12.0, 0.0],
                ..test_reference_trace_record("kqv_head1", vec![3.0, 12.0, 0.0])
            },
        ];

        let report = attention_value_mix_reference_scalar_recompute(&reference_records);

        assert_eq!(report.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(report.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(report.pointer("/group_size"), Some(&json!(2)));
        assert_eq!(report.pointer("/compared_count"), Some(&json!(2)));
        assert_eq!(report.pointer("/missing_input_count"), Some(&json!(0)));
        assert_eq!(report.pointer("/max_abs_delta"), Some(&json!(0.0)));
        assert_eq!(report.pointer("/max_rms_delta"), Some(&json!(0.0)));
        assert_eq!(
            report.pointer("/rows/0/recomputed_first_values"),
            Some(&json!([3.5, 13.0, 1.0]))
        );
        assert_eq!(
            report.pointer("/rows/1/recomputed_first_values"),
            Some(&json!([3.0, 12.0, 0.0]))
        );
    }

    #[test]
    fn reference_probability_softmax_variants_pin_scaled_padded_policy() {
        let reference_records = vec![
            ReferenceTraceRecord {
                shape: vec![4, 1, 1, 1],
                nelements: 4,
                first_values: vec![0.0, 0.0, 0.0, 0.0],
                ..test_reference_trace_record("Qcur", vec![0.0, 0.0, 0.0, 0.0])
            },
            ReferenceTraceRecord {
                shape: vec![4, 1, 1, 1],
                nelements: 4,
                first_values: vec![0.0, 1.0, 0.0, 0.0],
                ..test_reference_trace_record("kq_head0", vec![0.0, 1.0, 0.0, 0.0])
            },
            ReferenceTraceRecord {
                shape: vec![4, 1, 1, 1],
                nelements: 4,
                first_values: vec![0.37754068, 0.62245935, 0.0, 0.0],
                ..test_reference_trace_record(
                    "kq_soft_max_ext_head0",
                    vec![0.37754068, 0.62245935, 0.0, 0.0],
                )
            },
        ];

        let report = attention_probability_reference_softmax_variants(&reference_records);

        assert_eq!(report.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(report.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(report.pointer("/compared_count"), Some(&json!(1)));
        assert_eq!(report.pointer("/missing_input_count"), Some(&json!(0)));
        assert_eq!(report.pointer("/all_heads_explained"), Some(&json!(true)));
        assert_eq!(
            report.pointer("/rows/0/best_variant"),
            Some(&json!("reference_probability_softmax_scaled_1_sqrt_head_dim_padded_tail_zeroed"))
        );
        assert_eq!(report.pointer("/rows/0/live_token_count"), Some(&json!(2)));
        assert_eq!(report.pointer("/rows/0/padded_token_count"), Some(&json!(2)));
        assert_eq!(report.pointer("/rows/0/scale_policy"), Some(&json!("1_sqrt_head_dim")));
        assert_eq!(report.pointer("/rows/0/max_abs_delta"), Some(&json!(0.0)));
        assert_eq!(
            report.pointer("/variants_tested/0"),
            Some(&json!("reference_probability_softmax_unscaled_live"))
        );

        let compare_report = compare_reference_to_rust(&reference_records, &BTreeMap::new(), &[]);
        assert_eq!(
            compare_report
                .pointer("/attention_probability_reference_softmax_variants/claim_allowed"),
            Some(&json!(false))
        );
        assert_eq!(
            compare_report
                .pointer("/attention_probability_reference_softmax_variants/all_heads_explained"),
            Some(&json!(true))
        );
    }

    #[test]
    fn rust_probability_softmax_recompute_pins_scaled_live_policy() {
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_q_rope".to_string(),
            RustTraceRecord {
                shape: vec![1, 1, 1, 4],
                num_elements: 4,
                first_values: vec![0.0, 0.0, 0.0, 0.0],
                ..test_rust_trace_record("attention_q_rope", vec![0.0, 0.0, 0.0, 0.0])
            },
        );
        rust_records.insert(
            "attention_scores_raw_head0".to_string(),
            RustTraceRecord {
                shape: vec![2],
                num_elements: 2,
                first_values: vec![0.0, 1.0],
                ..test_rust_trace_record("attention_scores_raw_head0", vec![0.0, 1.0])
            },
        );
        rust_records.insert(
            "attn_scores_softmax_head0".to_string(),
            RustTraceRecord {
                shape: vec![2],
                num_elements: 2,
                first_values: vec![0.37754068, 0.62245935],
                ..test_rust_trace_record("attn_scores_softmax_head0", vec![0.37754068, 0.62245935])
            },
        );

        let report = attention_probability_rust_softmax_recompute(&rust_records);

        assert_eq!(report.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(report.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(report.pointer("/compared_count"), Some(&json!(1)));
        assert_eq!(report.pointer("/missing_input_count"), Some(&json!(0)));
        assert_eq!(report.pointer("/all_heads_explained"), Some(&json!(true)));
        assert_eq!(
            report.pointer("/rows/0/best_variant"),
            Some(&json!("reference_probability_softmax_scaled_1_sqrt_head_dim_live"))
        );
        assert_eq!(report.pointer("/rows/0/live_token_count"), Some(&json!(2)));
        assert_eq!(report.pointer("/rows/0/padded_token_count"), Some(&json!(0)));
        assert_eq!(report.pointer("/rows/0/scale_policy"), Some(&json!("1_sqrt_head_dim")));
        assert_eq!(report.pointer("/rows/0/max_abs_delta"), Some(&json!(0.0)));

        let compare_report = compare_reference_to_rust(&[], &rust_records, &[]);
        assert_eq!(
            compare_report.pointer("/attention_probability_rust_softmax_recompute/claim_allowed"),
            Some(&json!(false))
        );
        assert_eq!(
            compare_report
                .pointer("/attention_probability_rust_softmax_recompute/all_heads_explained"),
            Some(&json!(true))
        );
    }

    #[test]
    fn score_input_attribution_reports_reference_and_rust_qk_sources() {
        let reference_records = vec![
            ReferenceTraceRecord {
                shape: vec![2, 1],
                nelements: 2,
                first_values: vec![1.0, 2.0],
                ..test_reference_trace_record("Qcur", vec![1.0, 2.0])
            },
            ReferenceTraceRecord {
                shape: vec![2, 2],
                nelements: 4,
                first_values: vec![3.0, 4.0, 5.0, 6.0],
                ..test_reference_trace_record("k_kv_head0", vec![3.0, 4.0, 5.0, 6.0])
            },
            ReferenceTraceRecord {
                shape: vec![2],
                nelements: 2,
                first_values: vec![11.0, 17.0],
                ..test_reference_trace_record("kq_head0", vec![11.0, 17.0])
            },
        ];
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_q_rope".to_string(),
            RustTraceRecord {
                shape: vec![1, 1, 1, 2],
                num_elements: 2,
                first_values: vec![2.0, 2.0],
                ..test_rust_trace_record("attention_q_rope", vec![2.0, 2.0])
            },
        );
        rust_records.insert(
            "attention_k_cache_f16_roundtrip_kv_head0".to_string(),
            RustTraceRecord {
                shape: vec![2, 2],
                num_elements: 4,
                first_values: vec![30.0, 50.0, 40.0, 60.0],
                ..test_rust_trace_record(
                    "attention_k_cache_f16_roundtrip_kv_head0",
                    vec![30.0, 50.0, 40.0, 60.0],
                )
            },
        );
        rust_records.insert(
            "attention_scores_raw_head0".to_string(),
            RustTraceRecord {
                shape: vec![2],
                num_elements: 2,
                first_values: vec![140.0, 220.0],
                ..test_rust_trace_record("attention_scores_raw_head0", vec![140.0, 220.0])
            },
        );

        let report = attention_score_input_attribution(&reference_records, &rust_records);

        assert_eq!(report.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(report.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(report.pointer("/compared_count"), Some(&json!(1)));
        assert_eq!(report.pointer("/missing_input_count"), Some(&json!(0)));
        assert_eq!(
            report.pointer("/rust_key_stage_source"),
            Some(&json!("attention_k_cache_f16_roundtrip_kv_head_fallback"))
        );
        assert_eq!(report.pointer("/rust_score_key_head_count"), Some(&json!(0)));
        assert_eq!(report.pointer("/rust_fallback_key_head_count"), Some(&json!(1)));
        assert_eq!(
            report.pointer("/rows/0/reference_best_candidate"),
            Some(&json!("reference_q_reference_k"))
        );
        assert_eq!(report.pointer("/rows/0/rust_best_candidate"), Some(&json!("rust_q_rust_k")));
        assert_eq!(
            report.pointer("/rows/0/rust_key_source_kind"),
            Some(&json!("fallback_f16_cache_proxy"))
        );
        assert_eq!(
            report.pointer("/reference_best_candidate_counts/reference_q_reference_k"),
            Some(&json!(1))
        );
        assert_eq!(report.pointer("/rust_best_candidate_counts/rust_q_rust_k"), Some(&json!(1)));

        let compare_report = compare_reference_to_rust(&reference_records, &rust_records, &[]);
        assert_eq!(
            compare_report.pointer("/attention_score_input_attribution/claim_allowed"),
            Some(&json!(false))
        );
        assert_eq!(
            compare_report.pointer("/attention_score_input_attribution/rows/0/rust_best_candidate"),
            Some(&json!("rust_q_rust_k"))
        );
    }

    #[test]
    fn score_input_attribution_prefers_actual_score_key_input_stage() {
        let reference_records = vec![
            ReferenceTraceRecord {
                shape: vec![2, 1],
                nelements: 2,
                first_values: vec![1.0, 2.0],
                ..test_reference_trace_record("Qcur", vec![1.0, 2.0])
            },
            ReferenceTraceRecord {
                shape: vec![2, 2],
                nelements: 4,
                first_values: vec![3.0, 4.0, 5.0, 6.0],
                ..test_reference_trace_record("k_kv_head0", vec![3.0, 4.0, 5.0, 6.0])
            },
            ReferenceTraceRecord {
                shape: vec![2],
                nelements: 2,
                first_values: vec![11.0, 17.0],
                ..test_reference_trace_record("kq_head0", vec![11.0, 17.0])
            },
        ];
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_q_rope".to_string(),
            RustTraceRecord {
                shape: vec![1, 1, 1, 2],
                num_elements: 2,
                first_values: vec![2.0, 2.0],
                ..test_rust_trace_record("attention_q_rope", vec![2.0, 2.0])
            },
        );
        rust_records.insert(
            "attention_k_cache_f16_roundtrip_kv_head0".to_string(),
            RustTraceRecord {
                shape: vec![2, 2],
                num_elements: 4,
                first_values: vec![1.0, 1.0, 1.0, 1.0],
                ..test_rust_trace_record(
                    "attention_k_cache_f16_roundtrip_kv_head0",
                    vec![1.0, 1.0, 1.0, 1.0],
                )
            },
        );
        rust_records.insert(
            "attention_k_score_input_head0_live_ref_layout".to_string(),
            RustTraceRecord {
                shape: vec![2, 2],
                num_elements: 4,
                first_values: vec![30.0, 50.0, 40.0, 60.0],
                ..test_rust_trace_record(
                    "attention_k_score_input_head0_live_ref_layout",
                    vec![30.0, 50.0, 40.0, 60.0],
                )
            },
        );
        rust_records.insert(
            "attention_scores_raw_head0".to_string(),
            RustTraceRecord {
                shape: vec![2],
                num_elements: 2,
                first_values: vec![140.0, 220.0],
                ..test_rust_trace_record("attention_scores_raw_head0", vec![140.0, 220.0])
            },
        );

        let report = attention_score_input_attribution(&reference_records, &rust_records);

        assert_eq!(report.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(report.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(
            report.pointer("/rust_key_stage_source"),
            Some(&json!("attention_k_score_input_head"))
        );
        assert_eq!(report.pointer("/rust_score_key_head_count"), Some(&json!(1)));
        assert_eq!(report.pointer("/rust_fallback_key_head_count"), Some(&json!(1)));
        assert_eq!(
            report.pointer("/rows/0/rust_key_source"),
            Some(&json!("rust_attention_k_score_input_head"))
        );
        assert_eq!(
            report.pointer("/rows/0/rust_key_source_kind"),
            Some(&json!("actual_score_input"))
        );
        assert_eq!(report.pointer("/rows/0/rust_best_candidate"), Some(&json!("rust_q_rust_k")));
        assert_eq!(
            report.pointer("/rows/0/candidates/3/key_source"),
            Some(&json!("rust_attention_k_score_input_head"))
        );
    }

    #[test]
    fn reference_value_mix_numeric_variants_pin_value_f16_roundtrip_policy() {
        let reference_records = vec![
            ReferenceTraceRecord {
                shape: vec![2, 1, 1, 1],
                nelements: 2,
                first_values: vec![0.25, 0.75],
                ..test_reference_trace_record("kq_soft_max_ext_head0", vec![0.25, 0.75])
            },
            ReferenceTraceRecord {
                shape: vec![2, 2, 1, 1],
                nelements: 4,
                token_axis: Some(-1),
                first_values: vec![1.0003, 2.0007, 3.1259, -4.2509],
                ..test_reference_trace_record(
                    "v_cache_rust_layout_head0_live",
                    vec![1.0003, 2.0007, 3.1259, -4.2509],
                )
            },
            ReferenceTraceRecord {
                shape: vec![2, 1, 1, 1],
                nelements: 2,
                first_values: vec![1.75, -2.40625],
                ..test_reference_trace_record("kqv_head0", vec![1.75, -2.40625])
            },
        ];

        let report = attention_value_mix_reference_numeric_variants(&reference_records);

        assert_eq!(report.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(report.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(report.pointer("/compared_count"), Some(&json!(1)));
        assert_eq!(report.pointer("/missing_input_count"), Some(&json!(0)));
        assert_eq!(report.pointer("/all_heads_explained"), Some(&json!(true)));
        assert_eq!(
            report.pointer("/rows/0/best_variant"),
            Some(&json!("reference_value_mix_numeric_f32_accum_p_f32_v_f16"))
        );
        assert_eq!(report.pointer("/rows/0/probability_f16_roundtrip"), Some(&json!(false)));
        assert_eq!(report.pointer("/rows/0/value_f16_roundtrip"), Some(&json!(true)));
        assert_eq!(report.pointer("/rows/0/output_f16_roundtrip"), Some(&json!(false)));
        assert_eq!(report.pointer("/rows/0/max_abs_delta"), Some(&json!(0.0)));
        assert_eq!(
            report.pointer("/variants_tested/0"),
            Some(&json!("reference_value_mix_numeric_f64_accum_p_f32_v_f32"))
        );

        let compare_report = compare_reference_to_rust(&reference_records, &BTreeMap::new(), &[]);
        assert_eq!(
            compare_report.pointer("/attention_value_mix_reference_numeric_variants/claim_allowed"),
            Some(&json!(false))
        );
        assert_eq!(
            compare_report
                .pointer("/attention_value_mix_reference_numeric_variants/all_heads_explained"),
            Some(&json!(true))
        );
    }

    #[test]
    fn compare_reports_rust_scalar_value_mix_recompute() {
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attn_scores_softmax_head0".to_string(),
            RustTraceRecord {
                shape: vec![2],
                num_elements: 2,
                first_values: vec![0.25, 0.75],
                ..test_rust_trace_record("attn_scores_softmax_head0", vec![0.25, 0.75])
            },
        );
        rust_records.insert(
            "attn_scores_softmax_head1".to_string(),
            RustTraceRecord {
                shape: vec![2],
                num_elements: 2,
                first_values: vec![0.5, 0.5],
                ..test_rust_trace_record("attn_scores_softmax_head1", vec![0.5, 0.5])
            },
        );
        rust_records.insert(
            "attention_v_cache_f16_roundtrip_kv_head0_live_ref_layout".to_string(),
            RustTraceRecord {
                shape: vec![3, 2],
                num_elements: 6,
                first_values: vec![2.0, 4.0, 10.0, 14.0, -2.0, 2.0],
                ..test_rust_trace_record(
                    "attention_v_cache_f16_roundtrip_kv_head0_live_ref_layout",
                    vec![2.0, 4.0, 10.0, 14.0, -2.0, 2.0],
                )
            },
        );
        rust_records.insert(
            "attention_value_mix_f16_cache_head0".to_string(),
            RustTraceRecord {
                shape: vec![3],
                num_elements: 3,
                first_values: vec![3.5, 13.0, 1.0],
                ..test_rust_trace_record(
                    "attention_value_mix_f16_cache_head0",
                    vec![3.5, 13.0, 1.0],
                )
            },
        );
        rust_records.insert(
            "attention_value_mix_f16_cache_head1".to_string(),
            RustTraceRecord {
                shape: vec![3],
                num_elements: 3,
                first_values: vec![3.0, 12.0, 0.0],
                ..test_rust_trace_record(
                    "attention_value_mix_f16_cache_head1",
                    vec![3.0, 12.0, 0.0],
                )
            },
        );

        let report = attention_value_mix_rust_scalar_recompute(&rust_records);

        assert_eq!(report.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(report.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(report.pointer("/group_size"), Some(&json!(2)));
        assert_eq!(report.pointer("/compared_count"), Some(&json!(2)));
        assert_eq!(report.pointer("/missing_input_count"), Some(&json!(0)));
        assert_eq!(report.pointer("/max_abs_delta"), Some(&json!(0.0)));
        assert_eq!(report.pointer("/max_rms_delta"), Some(&json!(0.0)));
        assert_eq!(
            report.pointer("/rows/0/recomputed_first_values"),
            Some(&json!([3.5, 13.0, 1.0]))
        );
        assert_eq!(
            report.pointer("/rows/1/recomputed_first_values"),
            Some(&json!([3.0, 12.0, 0.0]))
        );

        let compare_report = compare_reference_to_rust(&[], &rust_records, &[]);
        assert_eq!(
            compare_report.pointer("/attention_value_mix_rust_scalar_recompute/claim_allowed"),
            Some(&json!(false))
        );
        assert_eq!(
            compare_report.pointer("/attention_value_mix_rust_scalar_recompute/compared_count"),
            Some(&json!(2))
        );
    }

    #[test]
    fn compare_reports_value_mix_input_attribution() {
        let reference_records = vec![
            ReferenceTraceRecord {
                shape: vec![2, 1, 1, 1],
                nelements: 2,
                first_values: vec![0.25, 0.75],
                ..test_reference_trace_record("kq_soft_max_ext_head0", vec![0.25, 0.75])
            },
            ReferenceTraceRecord {
                shape: vec![3, 2, 1, 1],
                nelements: 6,
                first_values: vec![2.0, 4.0, 10.0, 14.0, -2.0, 2.0],
                ..test_reference_trace_record(
                    "v_cache_rust_layout_head0_live",
                    vec![2.0, 4.0, 10.0, 14.0, -2.0, 2.0],
                )
            },
            ReferenceTraceRecord {
                shape: vec![3, 1, 1, 1],
                nelements: 3,
                first_values: vec![3.5, 13.0, 1.0],
                ..test_reference_trace_record("kqv_head0", vec![3.5, 13.0, 1.0])
            },
        ];
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attn_scores_softmax_head0".to_string(),
            RustTraceRecord {
                shape: vec![2],
                num_elements: 2,
                first_values: vec![0.5, 0.5],
                ..test_rust_trace_record("attn_scores_softmax_head0", vec![0.5, 0.5])
            },
        );
        rust_records.insert(
            "attention_v_cache_f16_roundtrip_kv_head0_live_ref_layout".to_string(),
            RustTraceRecord {
                shape: vec![3, 2],
                num_elements: 6,
                first_values: vec![2.0, 4.0, 10.0, 14.0, -2.0, 2.0],
                ..test_rust_trace_record(
                    "attention_v_cache_f16_roundtrip_kv_head0_live_ref_layout",
                    vec![2.0, 4.0, 10.0, 14.0, -2.0, 2.0],
                )
            },
        );

        let report = attention_value_mix_input_attribution(&reference_records, &rust_records);
        let reference_prob_rust_value =
            report.pointer("/reference_probability_rust_value_cache_vs_reference").unwrap();
        let rust_prob_reference_value =
            report.pointer("/rust_probability_reference_value_cache_vs_reference").unwrap();
        let rust_prob_rust_value =
            report.pointer("/rust_probability_rust_value_cache_vs_reference").unwrap();
        let candidate_best = report.pointer("/candidate_best_summary").unwrap();

        assert_eq!(report.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(report.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(report.pointer("/group_size"), Some(&json!(1)));
        assert_eq!(reference_prob_rust_value.pointer("/compared_count"), Some(&json!(1)));
        assert_eq!(reference_prob_rust_value.pointer("/max_abs_delta"), Some(&json!(0.0)));
        assert_eq!(rust_prob_reference_value.pointer("/compared_count"), Some(&json!(1)));
        assert_eq!(rust_prob_reference_value.pointer("/max_abs_delta"), Some(&json!(1.0)));
        assert_eq!(rust_prob_rust_value.pointer("/compared_count"), Some(&json!(1)));
        assert_eq!(rust_prob_rust_value.pointer("/max_abs_delta"), Some(&json!(1.0)));
        assert_eq!(candidate_best.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(candidate_best.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(candidate_best.pointer("/compared_count"), Some(&json!(1)));
        assert_eq!(
            candidate_best
                .pointer("/best_candidate_counts/reference_probability_reference_value_cache"),
            Some(&json!(1))
        );
        assert_eq!(
            candidate_best.pointer("/rows/0/best_candidate"),
            Some(&json!("reference_probability_reference_value_cache"))
        );
        assert_eq!(candidate_best.pointer("/rows/0/max_abs_delta"), Some(&json!(0.0)));
        assert_eq!(candidate_best.pointer("/rows/0/candidate_count"), Some(&json!(4)));

        let compare_report = compare_reference_to_rust(&reference_records, &rust_records, &[]);
        assert_eq!(
            compare_report.pointer("/attention_value_mix_input_attribution/claim_allowed"),
            Some(&json!(false))
        );
    }

    #[test]
    fn compare_reports_attention_score_and_probability_head_lane_best_matches() {
        let reference_records = vec![
            test_reference_trace_record("kq_head0", vec![1.0, 2.0]),
            test_reference_trace_record("kq_head1", vec![5.0, 5.0]),
            test_reference_trace_record("kq_soft_max_ext_head0", vec![0.25, 0.75]),
            test_reference_trace_record("kq_soft_max_ext_head1", vec![0.9, 0.1]),
        ];
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_scores_raw_head0".to_string(),
            test_rust_trace_record("attention_scores_raw_head0", vec![1.0, 2.0]),
        );
        rust_records.insert(
            "attention_scores_raw_head1".to_string(),
            test_rust_trace_record("attention_scores_raw_head1", vec![0.0, 0.0]),
        );
        rust_records.insert(
            "attention_scores_raw_head2".to_string(),
            test_rust_trace_record("attention_scores_raw_head2", vec![5.0, 5.0]),
        );
        rust_records.insert(
            "attn_scores_softmax_head0".to_string(),
            test_rust_trace_record("attn_scores_softmax_head0", vec![0.25, 0.75]),
        );
        rust_records.insert(
            "attn_scores_softmax_head1".to_string(),
            test_rust_trace_record("attn_scores_softmax_head1", vec![0.9, 0.1]),
        );

        let report = compare_reference_to_rust(&reference_records, &rust_records, &[]);
        let raw = report.pointer("/attention_score_raw_head_lane_best_matches").unwrap();
        let prob = report.pointer("/attention_probability_head_lane_best_matches").unwrap();

        assert_eq!(raw.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(raw.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(raw.pointer("/reference_stage_prefix"), Some(&json!("kq_head")));
        assert_eq!(raw.pointer("/rust_stage_prefix"), Some(&json!("attention_scores_raw_head")));
        assert_eq!(raw.pointer("/identity_best_count"), Some(&json!(1)));
        assert_eq!(raw.pointer("/non_identity_best_count"), Some(&json!(1)));
        assert_eq!(raw.pointer("/rows/1/best_rust_head"), Some(&json!(2)));

        assert_eq!(prob.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(prob.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(prob.pointer("/reference_stage_prefix"), Some(&json!("kq_soft_max_ext_head")));
        assert_eq!(prob.pointer("/rust_stage_prefix"), Some(&json!("attn_scores_softmax_head")));
        assert_eq!(prob.pointer("/identity_best_count"), Some(&json!(2)));
        assert_eq!(prob.pointer("/all_identity_best"), Some(&json!(true)));
    }

    #[test]
    fn compare_reports_attention_score_scalar_recompute() {
        let reference_records = vec![
            ReferenceTraceRecord {
                shape: vec![2, 2, 1, 1],
                nelements: 4,
                first_values: vec![1.0, 2.0, 3.0, 4.0],
                ..test_reference_trace_record("Qcur", vec![1.0, 2.0, 3.0, 4.0])
            },
            ReferenceTraceRecord {
                shape: vec![2, 2, 1, 1],
                nelements: 4,
                first_values: vec![5.0, 6.0, 7.0, 8.0],
                ..test_reference_trace_record("k_kv_head0_live", vec![5.0, 6.0, 7.0, 8.0])
            },
            test_reference_trace_record("kq_head0", vec![17.0, 23.0]),
            test_reference_trace_record("kq_head1", vec![39.0, 53.0]),
        ];
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_q_rope".to_string(),
            RustTraceRecord {
                shape: vec![1, 2, 1, 2],
                num_elements: 4,
                first_values: vec![1.0, 2.0, 3.0, 4.0],
                ..test_rust_trace_record("attention_q_rope", vec![1.0, 2.0, 3.0, 4.0])
            },
        );
        rust_records.insert(
            "attention_k_cache_kv_head0_live_ref_layout".to_string(),
            RustTraceRecord {
                shape: vec![2, 2],
                num_elements: 4,
                first_values: vec![5.0, 7.0, 6.0, 8.0],
                ..test_rust_trace_record(
                    "attention_k_cache_kv_head0_live_ref_layout",
                    vec![5.0, 7.0, 6.0, 8.0],
                )
            },
        );
        rust_records.insert(
            "attention_scores_raw_head0".to_string(),
            test_rust_trace_record("attention_scores_raw_head0", vec![17.0, 23.0]),
        );
        rust_records.insert(
            "attention_scores_raw_head1".to_string(),
            test_rust_trace_record("attention_scores_raw_head1", vec![39.0, 53.0]),
        );

        let report = compare_reference_to_rust(&reference_records, &rust_records, &[]);
        let query_delta = report.pointer("/attention_query_rope_ref_layout_delta").unwrap();
        let reference = report.pointer("/attention_score_reference_scalar_recompute").unwrap();
        let variants = report.pointer("/attention_score_reference_semantic_variants").unwrap();
        let numeric = report.pointer("/attention_score_reference_numeric_variants").unwrap();
        let rust = report.pointer("/attention_score_rust_scalar_recompute").unwrap();

        assert_eq!(query_delta.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(query_delta.pointer("/delta/max_abs_delta"), Some(&json!(0.0)));
        assert_eq!(reference.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(reference.pointer("/compared_count"), Some(&json!(2)));
        assert_eq!(reference.pointer("/max_abs_delta"), Some(&json!(0.0)));
        assert_eq!(variants.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(variants.pointer("/compared_count"), Some(&json!(2)));
        assert_eq!(variants.pointer("/variant_count"), Some(&json!(6)));
        assert_eq!(
            variants.pointer("/rows/0/best_variant"),
            Some(&json!("reference_score_recompute_unscaled"))
        );
        assert_eq!(variants.pointer("/rows/0/head_explained"), Some(&json!(true)));
        assert_eq!(numeric.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(numeric.pointer("/compared_count"), Some(&json!(2)));
        assert_eq!(numeric.pointer("/variant_count"), Some(&json!(6)));
        assert_eq!(rust.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(rust.pointer("/compared_count"), Some(&json!(2)));
        assert_eq!(rust.pointer("/max_abs_delta"), Some(&json!(0.0)));
    }

    #[test]
    fn reference_score_semantic_variants_pin_padded_tail_zero_policy() {
        let reference_records = vec![
            ReferenceTraceRecord {
                shape: vec![2, 1, 1, 1],
                nelements: 2,
                first_values: vec![1.0, 2.0],
                ..test_reference_trace_record("Qcur", vec![1.0, 2.0])
            },
            ReferenceTraceRecord {
                shape: vec![2, 2, 1, 1],
                nelements: 4,
                first_values: vec![5.0, 6.0, 7.0, 8.0],
                ..test_reference_trace_record("k_kv_head0_live", vec![5.0, 6.0, 7.0, 8.0])
            },
            ReferenceTraceRecord {
                shape: vec![4, 1, 1, 1],
                full_shape: vec![4, 1, 1, 1],
                nelements: 4,
                first_values: vec![17.0, 23.0, 0.0, 0.0],
                ..test_reference_trace_record("kq_head0", vec![17.0, 23.0, 0.0, 0.0])
            },
        ];
        let report = compare_reference_to_rust(&reference_records, &BTreeMap::new(), &[]);
        let variants = report.pointer("/attention_score_reference_semantic_variants").unwrap();

        assert_eq!(variants.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(variants.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(variants.pointer("/compared_count"), Some(&json!(1)));
        assert_eq!(variants.pointer("/unexplained_head_count"), Some(&json!(0)));
        assert_eq!(
            variants.pointer("/rows/0/best_variant"),
            Some(&json!("reference_score_recompute_with_padded_tail_zeroed"))
        );
        assert_eq!(variants.pointer("/rows/0/token_count"), Some(&json!(4)));
        assert_eq!(variants.pointer("/rows/0/live_token_count"), Some(&json!(2)));
        assert_eq!(variants.pointer("/rows/0/padded_token_count"), Some(&json!(2)));
        assert_eq!(
            variants.pointer("/rows/0/mask_policy"),
            Some(&json!("padded_tail_zeroed_to_score_sample_len"))
        );
        assert_eq!(variants.pointer("/rows/0/best_delta/count_match"), Some(&json!(true)));
        assert_eq!(variants.pointer("/rows/0/best_delta/max_abs_delta"), Some(&json!(0.0)));
        assert_eq!(
            variants.pointer("/variants_tested/5"),
            Some(&json!("reference_score_recompute_with_mask_applied"))
        );
    }

    #[test]
    fn reference_score_numeric_variants_pin_qk_f16_roundtrip_policy() {
        let query = ReferenceTraceRecord {
            shape: vec![4, 1, 1, 1],
            nelements: 4,
            first_values: vec![1.0003, -2.0007, 3.1259, -4.2509],
            ..test_reference_trace_record("Qcur", vec![1.0003, -2.0007, 3.1259, -4.2509])
        };
        let key = ReferenceTraceRecord {
            shape: vec![4, 2, 1, 1],
            nelements: 8,
            first_values: vec![5.0009, -6.0004, 7.1257, -8.2508, -1.3333, 2.6667, -3.9991, 4.5006],
            ..test_reference_trace_record(
                "k_kv_head0_live",
                vec![5.0009, -6.0004, 7.1257, -8.2508, -1.3333, 2.6667, -3.9991, 4.5006],
            )
        };
        let variant = ReferenceScoreNumericVariantSpec {
            id: "test_qk_f16",
            query_f16_roundtrip: true,
            key_f16_roundtrip: true,
            accum_policy: ReferenceScoreAccumPolicy::F32,
        };
        let mut target =
            reference_score_row_from_query_key_numeric(&query, &key, 0, 4, 2, variant).unwrap();
        target.extend([0.0, 0.0]);

        let reference_records = vec![
            query,
            key,
            ReferenceTraceRecord {
                shape: vec![4, 1, 1, 1],
                full_shape: vec![4, 1, 1, 1],
                nelements: 4,
                first_values: target,
                ..test_reference_trace_record("kq_head0", vec![])
            },
        ];
        let report = compare_reference_to_rust(&reference_records, &BTreeMap::new(), &[]);
        let numeric = report.pointer("/attention_score_reference_numeric_variants").unwrap();

        assert_eq!(numeric.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(numeric.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(numeric.pointer("/compared_count"), Some(&json!(1)));
        assert_eq!(numeric.pointer("/unexplained_head_count"), Some(&json!(0)));
        assert_eq!(
            numeric.pointer("/rows/0/best_variant"),
            Some(&json!("reference_score_numeric_f32_accum_q_f16_k_f16"))
        );
        assert_eq!(numeric.pointer("/rows/0/query_f16_roundtrip"), Some(&json!(true)));
        assert_eq!(numeric.pointer("/rows/0/key_f16_roundtrip"), Some(&json!(true)));
        assert_eq!(numeric.pointer("/rows/0/accum_policy"), Some(&json!("f32_sequential")));
        assert_eq!(numeric.pointer("/rows/0/best_delta/max_abs_delta"), Some(&json!(0.0)));
    }

    #[test]
    fn compare_reports_value_cache_kv_head_best_matches_with_suffix_stages() {
        let reference_records = vec![
            test_reference_trace_record("k_kv_head0_live", vec![1.0, 2.0]),
            test_reference_trace_record("k_kv_head1_live", vec![9.0, 9.0]),
            test_reference_trace_record("v_kv_head0_live", vec![1.0, 2.0]),
            test_reference_trace_record("v_kv_head1_live", vec![9.0, 9.0]),
            test_reference_trace_record("v_cache_rust_layout_head0_live", vec![1.0, 2.0]),
            test_reference_trace_record("v_cache_rust_layout_head1_live", vec![9.0, 9.0]),
        ];
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_k_cache_kv_head0_live_ref_layout".to_string(),
            test_rust_trace_record("attention_k_cache_kv_head0_live_ref_layout", vec![1.0, 2.0]),
        );
        rust_records.insert(
            "attention_k_cache_kv_head1_live_ref_layout".to_string(),
            test_rust_trace_record("attention_k_cache_kv_head1_live_ref_layout", vec![0.0, 0.0]),
        );
        rust_records.insert(
            "attention_k_cache_kv_head2_live_ref_layout".to_string(),
            test_rust_trace_record("attention_k_cache_kv_head2_live_ref_layout", vec![9.0, 9.0]),
        );
        rust_records.insert(
            "attention_k_cache_f16_roundtrip_kv_head0_live_ref_layout".to_string(),
            test_rust_trace_record(
                "attention_k_cache_f16_roundtrip_kv_head0_live_ref_layout",
                vec![1.0, 2.0],
            ),
        );
        rust_records.insert(
            "attention_k_cache_f16_roundtrip_kv_head1_live_ref_layout".to_string(),
            test_rust_trace_record(
                "attention_k_cache_f16_roundtrip_kv_head1_live_ref_layout",
                vec![0.0, 0.0],
            ),
        );
        rust_records.insert(
            "attention_k_cache_f16_roundtrip_kv_head2_live_ref_layout".to_string(),
            test_rust_trace_record(
                "attention_k_cache_f16_roundtrip_kv_head2_live_ref_layout",
                vec![9.0, 9.0],
            ),
        );
        rust_records.insert(
            "attention_v_cache_kv_head0_live_ref_layout".to_string(),
            test_rust_trace_record("attention_v_cache_kv_head0_live_ref_layout", vec![1.0, 2.0]),
        );
        rust_records.insert(
            "attention_v_cache_kv_head1_live_ref_layout".to_string(),
            test_rust_trace_record("attention_v_cache_kv_head1_live_ref_layout", vec![0.0, 0.0]),
        );
        rust_records.insert(
            "attention_v_cache_kv_head2_live_ref_layout".to_string(),
            test_rust_trace_record("attention_v_cache_kv_head2_live_ref_layout", vec![9.0, 9.0]),
        );
        rust_records.insert(
            "attention_v_cache_f16_roundtrip_kv_head0_live_ref_layout".to_string(),
            test_rust_trace_record(
                "attention_v_cache_f16_roundtrip_kv_head0_live_ref_layout",
                vec![1.0, 2.0],
            ),
        );
        rust_records.insert(
            "attention_v_cache_f16_roundtrip_kv_head1_live_ref_layout".to_string(),
            test_rust_trace_record(
                "attention_v_cache_f16_roundtrip_kv_head1_live_ref_layout",
                vec![0.0, 0.0],
            ),
        );
        rust_records.insert(
            "attention_v_cache_f16_roundtrip_kv_head2_live_ref_layout".to_string(),
            test_rust_trace_record(
                "attention_v_cache_f16_roundtrip_kv_head2_live_ref_layout",
                vec![9.0, 9.0],
            ),
        );

        let report = compare_reference_to_rust(&reference_records, &rust_records, &[]);
        let key_cache = report.pointer("/attention_key_cache_kv_head_best_matches").unwrap();
        assert_eq!(key_cache.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(key_cache.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(key_cache.pointer("/reference_stage_prefix"), Some(&json!("k_kv_head")));
        assert_eq!(
            key_cache.pointer("/rust_stage_prefix"),
            Some(&json!("attention_k_cache_kv_head"))
        );
        assert_eq!(key_cache.pointer("/identity_best_count"), Some(&json!(1)));
        assert_eq!(key_cache.pointer("/non_identity_best_count"), Some(&json!(1)));
        assert_eq!(key_cache.pointer("/rows/1/best_rust_head"), Some(&json!(2)));

        let key_f16_roundtrip =
            report.pointer("/attention_key_cache_f16_roundtrip_best_matches").unwrap();
        assert_eq!(key_f16_roundtrip.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(key_f16_roundtrip.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(key_f16_roundtrip.pointer("/reference_stage_prefix"), Some(&json!("k_kv_head")));
        assert_eq!(
            key_f16_roundtrip.pointer("/rust_stage_prefix"),
            Some(&json!("attention_k_cache_f16_roundtrip_kv_head"))
        );

        let value_cache = report.pointer("/attention_value_cache_kv_head_best_matches").unwrap();

        assert_eq!(value_cache.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(value_cache.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(value_cache.pointer("/reference_stage_prefix"), Some(&json!("v_kv_head")));
        assert_eq!(
            value_cache.pointer("/rust_stage_prefix"),
            Some(&json!("attention_v_cache_kv_head"))
        );
        assert_eq!(value_cache.pointer("/identity_best_count"), Some(&json!(1)));
        assert_eq!(value_cache.pointer("/non_identity_best_count"), Some(&json!(1)));
        assert_eq!(value_cache.pointer("/rows/1/best_rust_head"), Some(&json!(2)));

        let rust_layout =
            report.pointer("/attention_value_cache_rust_layout_best_matches").unwrap();
        assert_eq!(rust_layout.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(rust_layout.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(
            rust_layout.pointer("/reference_stage_prefix"),
            Some(&json!("v_cache_rust_layout_head"))
        );
        assert_eq!(
            rust_layout.pointer("/rust_stage_prefix"),
            Some(&json!("attention_v_cache_kv_head"))
        );
        assert_eq!(rust_layout.pointer("/identity_best_count"), Some(&json!(1)));
        assert_eq!(rust_layout.pointer("/non_identity_best_count"), Some(&json!(1)));
        assert_eq!(rust_layout.pointer("/rows/1/best_rust_head"), Some(&json!(2)));

        let f16_roundtrip =
            report.pointer("/attention_value_cache_f16_roundtrip_best_matches").unwrap();
        assert_eq!(f16_roundtrip.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(f16_roundtrip.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(
            f16_roundtrip.pointer("/reference_stage_prefix"),
            Some(&json!("v_cache_rust_layout_head"))
        );
        assert_eq!(
            f16_roundtrip.pointer("/rust_stage_prefix"),
            Some(&json!("attention_v_cache_f16_roundtrip_kv_head"))
        );
        assert_eq!(f16_roundtrip.pointer("/identity_best_count"), Some(&json!(1)));
        assert_eq!(f16_roundtrip.pointer("/non_identity_best_count"), Some(&json!(1)));
        assert_eq!(f16_roundtrip.pointer("/rows/1/best_rust_head"), Some(&json!(2)));
    }

    #[test]
    fn compare_reports_value_cache_f16_amplification() {
        let reference_records = vec![
            ReferenceTraceRecord {
                shape: vec![2, 1, 1, 1],
                nelements: 2,
                first_values: vec![1.0, 2.0],
                ..test_reference_trace_record("Vcur", vec![1.0, 2.0])
            },
            ReferenceTraceRecord {
                shape: vec![2, 2, 1, 1],
                nelements: 4,
                first_values: vec![1.0, 2.0, 3.0, 4.0],
                ..test_reference_trace_record(
                    "v_cache_rust_layout_head0_live",
                    vec![1.0, 2.0, 3.0, 4.0],
                )
            },
            ReferenceTraceRecord {
                shape: vec![2, 2, 1, 1],
                nelements: 4,
                first_values: vec![3.0, 4.0, 5.0, 6.0],
                ..test_reference_trace_record(
                    "v_cache_rust_layout_head1_live",
                    vec![3.0, 4.0, 5.0, 6.0],
                )
            },
        ];
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_v".to_string(),
            RustTraceRecord {
                shape: vec![1, 1, 2],
                num_elements: 2,
                first_values: vec![1.0, 2.5],
                ..test_rust_trace_record("attention_v", vec![1.0, 2.5])
            },
        );
        rust_records.insert(
            "attention_v_cache_f16_roundtrip_kv_head0_live_ref_layout".to_string(),
            RustTraceRecord {
                shape: vec![2, 2],
                num_elements: 4,
                first_values: vec![1.0, 2.0, 3.0, 4.0],
                ..test_rust_trace_record(
                    "attention_v_cache_f16_roundtrip_kv_head0_live_ref_layout",
                    vec![1.0, 2.0, 3.0, 4.0],
                )
            },
        );
        rust_records.insert(
            "attention_v_cache_f16_roundtrip_kv_head1_live_ref_layout".to_string(),
            RustTraceRecord {
                shape: vec![2, 2],
                num_elements: 4,
                first_values: vec![3.0, 4.0, 5.0, 6.5],
                ..test_rust_trace_record(
                    "attention_v_cache_f16_roundtrip_kv_head1_live_ref_layout",
                    vec![3.0, 4.0, 5.0, 6.5],
                )
            },
        );

        let report = compare_reference_to_rust(&reference_records, &rust_records, &[]);
        let amplification = report.pointer("/attention_value_cache_f16_amplification").unwrap();

        assert_eq!(amplification.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(amplification.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(amplification.pointer("/compared_head_count"), Some(&json!(2)));
        assert_eq!(
            amplification.pointer("/projection_delta/f16_bucket_mismatch_count"),
            Some(&json!(1))
        );
        assert_eq!(amplification.pointer("/total_f16_bucket_mismatch_count"), Some(&json!(1)));
        assert_eq!(amplification.pointer("/max_head_f16_bucket_mismatch_count"), Some(&json!(1)));
        assert_eq!(
            amplification.pointer("/rows/1/delta/first_f16_bucket_mismatch/index"),
            Some(&json!(3))
        );
        assert_eq!(
            amplification.pointer("/rows/1/delta/first_f16_bucket_mismatch_layout/dim"),
            Some(&json!(1))
        );
        assert_eq!(
            amplification.pointer("/rows/1/delta/first_f16_bucket_mismatch_layout/token"),
            Some(&json!(1))
        );
        assert_eq!(
            amplification.pointer("/rows/1/delta/token_mismatch_counts/0/token"),
            Some(&json!(1))
        );
    }

    #[test]
    fn compare_reports_key_cache_dim_major_f16_roundtrip_layout_match() {
        let mut key_head0 = test_reference_trace_record(
            "k_kv_head0_live",
            vec![
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0,
            ],
        );
        key_head0.shape = vec![3, 2, 1, 1];
        key_head0.nelements = 6;
        let mut key_head1 = test_reference_trace_record(
            "k_kv_head1_live",
            vec![
                10.0, 20.0, 30.0, //
                40.0, 50.0, 60.0,
            ],
        );
        key_head1.shape = vec![3, 2, 1, 1];
        key_head1.nelements = 6;
        let reference_records = vec![key_head0, key_head1];
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_k_cache_f16_roundtrip_kv_head0_live_ref_layout".to_string(),
            test_rust_trace_record(
                "attention_k_cache_f16_roundtrip_kv_head0_live_ref_layout",
                vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            ),
        );
        rust_records.insert(
            "attention_k_cache_f16_roundtrip_kv_head1_live_ref_layout".to_string(),
            test_rust_trace_record(
                "attention_k_cache_f16_roundtrip_kv_head1_live_ref_layout",
                vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0],
            ),
        );

        let report = compare_reference_to_rust(&reference_records, &rust_records, &[]);
        let matches =
            report.pointer("/attention_key_cache_dim_major_f16_roundtrip_best_matches").unwrap();

        assert_eq!(matches.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(matches.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(
            matches.pointer("/reference_reinterpretation"),
            Some(&json!("token_major_to_dim_major"))
        );
        assert_eq!(matches.pointer("/identity_best_count"), Some(&json!(2)));
        assert_eq!(matches.pointer("/non_identity_best_count"), Some(&json!(0)));
        assert_eq!(matches.pointer("/all_identity_best"), Some(&json!(true)));
        assert_eq!(matches.pointer("/rows/0/identity_delta/max_abs_delta"), Some(&json!(0.0)));
    }

    #[test]
    fn compare_marks_key_cache_live_layout_as_scope_evidence() {
        let reference = ReferenceTraceRecord {
            shape: vec![128, 18, 1, 1],
            nelements: 128 * 18,
            first_values: vec![0.0; 32],
            ..test_reference_trace_record("k_kv_head3_live", vec![0.0; 32])
        };
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_k_cache_kv_head3_live_ref_layout".to_string(),
            RustTraceRecord {
                name: "t17/blk0/attention_k_cache_kv_head3_live_ref_layout".to_string(),
                shape: vec![128, 18],
                dtype: "F32".to_string(),
                blake3: "abc".to_string(),
                rms: 1.0,
                num_elements: 128 * 18,
                first_values: vec![0.0; 32],
                seq: Some(17),
                layer: Some(0),
                stage: Some("attention_k_cache_kv_head3_live_ref_layout".to_string()),
            },
        );

        let report = compare_reference_to_rust(
            &[reference],
            &rust_records,
            &[("k_kv_head3_live", "attention_k_cache_kv_head3_live_ref_layout")],
        );

        assert_eq!(report.pointer("/scope_mismatch_count"), Some(&json!(1)));
        assert_eq!(report.pointer("/material_mismatch_count"), Some(&json!(0)));
        assert_eq!(
            report.pointer("/first_scope_mismatch/scope/reason"),
            Some(&json!(
                "reference_key_cache_live_head_token_major_not_direct_rust_dim_major_layout"
            ))
        );
        assert!(report.pointer("/first_material_mismatch").unwrap().is_null());
    }

    #[test]
    fn compare_marks_key_cache_all_heads_as_scope_evidence() {
        let reference = ReferenceTraceRecord {
            shape: vec![128, 32, 5, 1],
            nelements: 128 * 32 * 5,
            first_values: vec![0.0; 32],
            ..test_reference_trace_record("k", vec![0.0; 32])
        };
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_k_cache_head0_ref_layout_padded".to_string(),
            RustTraceRecord {
                name: "t17/blk0/attention_k_cache_head0_ref_layout_padded".to_string(),
                shape: vec![128, 32],
                dtype: "F32".to_string(),
                blake3: "abc".to_string(),
                rms: 1.0,
                num_elements: 128 * 32,
                first_values: vec![0.0; 32],
                seq: Some(17),
                layer: Some(0),
                stage: Some("attention_k_cache_head0_ref_layout_padded".to_string()),
            },
        );

        let report = compare_reference_to_rust(
            &[reference],
            &rust_records,
            &[("k", "attention_k_cache_head0_ref_layout_padded")],
        );

        assert_eq!(report.pointer("/scope_mismatch_count"), Some(&json!(1)));
        assert_eq!(report.pointer("/material_mismatch_count"), Some(&json!(0)));
        assert_eq!(
            report.pointer("/first_scope_mismatch/scope/reason"),
            Some(&json!(
                "reference_key_cache_contains_all_kv_heads_rust_trace_samples_head0_reference_layout"
            ))
        );
        assert!(report.pointer("/first_material_mismatch").unwrap().is_null());
    }

    #[test]
    fn compare_marks_value_cache_live_layout_as_scope_evidence() {
        let reference = ReferenceTraceRecord {
            shape: vec![32, 18, 1, 1],
            nelements: 32 * 18,
            first_values: vec![0.0; 32],
            ..test_reference_trace_record("v_kv_head3_live", vec![0.0; 32])
        };
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_v_cache_kv_head3_live_ref_layout".to_string(),
            RustTraceRecord {
                name: "t17/blk0/attention_v_cache_kv_head3_live_ref_layout".to_string(),
                shape: vec![128, 18],
                dtype: "F32".to_string(),
                blake3: "abc".to_string(),
                rms: 1.0,
                num_elements: 128 * 18,
                first_values: vec![0.0; 32],
                seq: Some(17),
                layer: Some(0),
                stage: Some("attention_v_cache_kv_head3_live_ref_layout".to_string()),
            },
        );

        let report = compare_reference_to_rust(
            &[reference],
            &rust_records,
            &[("v_kv_head3_live", "attention_v_cache_kv_head3_live_ref_layout")],
        );

        assert_eq!(report.pointer("/scope_mismatch_count"), Some(&json!(1)));
        assert_eq!(report.pointer("/material_mismatch_count"), Some(&json!(0)));
        assert_eq!(
            report.pointer("/first_scope_mismatch/scope/reason"),
            Some(&json!("reference_value_cache_live_head_layout_not_direct_rust_kv_head_layout"))
        );
        assert!(report.pointer("/first_material_mismatch").unwrap().is_null());
    }

    #[test]
    fn compare_maps_value_cache_rust_layout_to_f16_roundtrip_as_material_evidence() {
        let reference = ReferenceTraceRecord {
            dtype: "f32_from_f16".to_string(),
            shape: vec![128, 18, 1, 1],
            full_shape: vec![128, 18, 1, 1],
            nelements: 128 * 18,
            token_axis: Some(-1),
            first_values: vec![0.25; 32],
            ..test_reference_trace_record("v_cache_rust_layout_head3_live", vec![0.25; 32])
        };
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_v_cache_f16_roundtrip_kv_head3_live_ref_layout".to_string(),
            RustTraceRecord {
                name: "t17/blk0/attention_v_cache_f16_roundtrip_kv_head3_live_ref_layout"
                    .to_string(),
                shape: vec![128, 18],
                dtype: "F32".to_string(),
                blake3: "abc".to_string(),
                rms: 0.25,
                num_elements: 128 * 18,
                first_values: vec![0.25; 32],
                seq: Some(17),
                layer: Some(0),
                stage: Some("attention_v_cache_f16_roundtrip_kv_head3_live_ref_layout".to_string()),
            },
        );

        let report = compare_reference_to_rust(
            &[reference],
            &rust_records,
            &[(
                "v_cache_rust_layout_head3_live",
                "attention_v_cache_f16_roundtrip_kv_head3_live_ref_layout",
            )],
        );

        assert_eq!(report.pointer("/scope_mismatch_count"), Some(&json!(0)));
        assert_eq!(report.pointer("/material_mismatch_count"), Some(&json!(0)));
        assert_eq!(report.pointer("/stages/0/status"), Some(&json!("summary_match")));
        assert!(report.pointer("/first_scope_mismatch").unwrap().is_null());
        assert!(report.pointer("/first_material_mismatch").unwrap().is_null());
        assert_eq!(report.pointer("/stages/0/first_values_delta/max_abs_delta"), Some(&json!(0.0)));
    }

    #[test]
    fn compare_treats_reference_f32_from_f16_as_rust_f32() {
        assert!(trace_dtype_compatible("f32_from_f16", "F32"));
        assert!(trace_dtype_compatible("f32", "F32"));
        assert!(!trace_dtype_compatible("f16", "F32"));
    }

    #[test]
    fn compare_head_lane_best_matches_require_samples() {
        let reference_records = vec![ReferenceTraceRecord {
            first_values: Vec::new(),
            ..test_reference_trace_record("kqv_head0", vec![1.0])
        }];
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_value_mix_head0".to_string(),
            test_rust_trace_record("attention_value_mix_head0", vec![1.0]),
        );

        let report = compare_reference_to_rust(&reference_records, &rust_records, &[]);
        let matches = report.pointer("/attention_value_mix_head_lane_best_matches").unwrap();

        assert_eq!(matches.pointer("/reference_head_count"), Some(&json!(0)));
        assert_eq!(matches.pointer("/rust_head_count"), Some(&json!(1)));
        assert_eq!(matches.pointer("/rows"), Some(&json!([])));
        assert_eq!(matches.pointer("/all_identity_best"), Some(&json!(false)));
    }

    #[test]
    fn compare_maps_reference_attention_probability_rows_to_rust_head0_traces() {
        let kq = ReferenceTraceRecord {
            name: "kq-0".to_string(),
            stage: "kq".to_string(),
            graph_index: Some(40),
            layer: Some(0),
            graph_op: Some("MUL_MAT".to_string()),
            graph_sources: json!([]),
            view_source: Value::Null,
            view_offset: Some(0),
            full_shape: vec![3, 2, 1, 1],
            sample_offset: Some(3),
            token_axis: Some(1),
            dtype: "f32".to_string(),
            shape: vec![3, 1, 1, 1],
            nelements: 3,
            rms: Some(2.0),
            values_available: true,
            first_values: vec![1.0, 2.0, 3.0],
        };
        let softmax = ReferenceTraceRecord {
            name: "kq_soft_max_ext-0".to_string(),
            stage: "kq_soft_max_ext".to_string(),
            graph_index: Some(41),
            layer: Some(0),
            graph_op: Some("SOFT_MAX_EXT".to_string()),
            graph_sources: json!([]),
            view_source: Value::Null,
            view_offset: Some(0),
            full_shape: vec![3, 2, 1, 1],
            sample_offset: Some(3),
            token_axis: Some(1),
            dtype: "f32".to_string(),
            shape: vec![3, 1, 1, 1],
            nelements: 3,
            rms: Some(0.5),
            values_available: true,
            first_values: vec![0.1, 0.2, 0.7],
        };
        let mut rust_records = BTreeMap::new();
        rust_records.insert(
            "attention_scores_raw_head0".to_string(),
            RustTraceRecord {
                name: "t1/blk0/attention_scores_raw_head0".to_string(),
                shape: vec![1, 1, 1, 3],
                dtype: "F32".to_string(),
                blake3: "abc".to_string(),
                rms: 2.0,
                num_elements: 3,
                first_values: vec![1.0, 2.0, 3.0],
                seq: Some(1),
                layer: Some(0),
                stage: Some("attention_scores_raw_head0".to_string()),
            },
        );
        rust_records.insert(
            "attn_scores_softmax_head0".to_string(),
            RustTraceRecord {
                name: "t1/blk0/attn_scores_softmax_head0".to_string(),
                shape: vec![1, 1, 1, 3],
                dtype: "F32".to_string(),
                blake3: "def".to_string(),
                rms: 0.25,
                num_elements: 3,
                first_values: vec![0.1, 0.2, 0.8],
                seq: Some(1),
                layer: Some(0),
                stage: Some("attn_scores_softmax_head0".to_string()),
            },
        );

        let report = compare_reference_to_rust(
            &[kq, softmax],
            &rust_records,
            &[
                ("kq", "attention_scores_raw_head0"),
                ("kq_soft_max_ext", "attn_scores_softmax_head0"),
            ],
        );

        assert_eq!(report.pointer("/scope_mismatch_count"), Some(&json!(0)));
        assert_eq!(report.pointer("/stages/0/status"), Some(&json!("summary_match")));
        assert_eq!(
            report.pointer("/first_material_mismatch/reference_stage"),
            Some(&json!("kq_soft_max_ext"))
        );
        let max_delta = report
            .pointer("/first_material_mismatch/first_values_delta/max_abs_delta")
            .and_then(Value::as_f64)
            .unwrap();
        assert!((max_delta - 0.10000002).abs() < 1.0e-7);
    }

    #[test]
    fn reference_trace_receipt_supports_run_wrapper_prompt_tokens() {
        let root = json!({
            "receipt_type": "bitnet_reference_layer_trace_run",
            "sidecar": {
                "receipt": {
                    "receipt_type": "bitnet_reference_layer_trace",
                    "ubatch_tokens": [128000, 17, 271],
                    "records": []
                }
            }
        });
        let trace = reference_trace_receipt(&root).unwrap();

        assert_eq!(reference_prompt_tokens(trace), vec![128000, 17, 271]);
    }

    #[test]
    fn embedding_row_decode_uses_hidden_by_vocab_token_column() {
        let bits = [0x3c00u16, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600];
        let mut data = Vec::new();
        for value in bits {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let layouts = embedding_raw_layouts(&[2, 3], 2, &[1]);
        let layout = layouts
            .iter()
            .find(|layout| {
                layout.pointer("/kind") == Some(&json!("ggml_ne0_hidden_by_vocab_token_column"))
            })
            .unwrap();
        let row = decode_embedding_row(
            &data,
            bitnet_models::formats::gguf::GgufTensorType::F16,
            &layout,
            1,
        )
        .unwrap();

        assert_eq!(layout.pointer("/kind"), Some(&json!("ggml_ne0_hidden_by_vocab_token_column")));
        assert_eq!(row, vec![3.0, 4.0]);
    }

    #[test]
    fn row_candidate_delta_scans_layout_candidates() {
        let row = json!({
            "reference_candidates": [
                {
                    "layout": {
                        "kind": "layout_a"
                    },
                    "reference_raw_vs_rust_loaded": {
                        "max_abs_delta": 4.0
                    }
                },
                {
                    "layout": {
                        "kind": "layout_b"
                    },
                    "reference_raw_vs_rust_loaded": {
                        "max_abs_delta": 0.0
                    }
                }
            ]
        });

        assert!(row_candidate_delta_le(
            &row,
            "/reference_raw_vs_rust_loaded/max_abs_delta",
            1.0e-3
        ));
        assert!(!row_candidate_delta_le(
            &row,
            "/reference_raw_vs_trace_first_values/max_abs_delta",
            1.0e-3
        ));
    }

    #[test]
    fn embedding_layout_authority_requires_same_candidate_layout() {
        let row = json!({
            "reference_candidates": [
                {
                    "layout": {
                        "kind": "reference_layout"
                    },
                    "reference_raw_vs_trace_first_values": {
                        "max_abs_delta": 0.0
                    },
                    "reference_raw_vs_rust_loaded": {
                        "max_abs_delta": 9.0
                    }
                },
                {
                    "layout": {
                        "kind": "rust_layout"
                    },
                    "reference_raw_vs_trace_first_values": {
                        "max_abs_delta": 9.0
                    },
                    "reference_raw_vs_rust_loaded": {
                        "max_abs_delta": 0.0
                    }
                }
            ]
        });

        let reference_layouts = row_candidate_matching_layouts(
            &row,
            "/reference_raw_vs_trace_first_values/max_abs_delta",
            1.0e-3,
        );
        let rust_layouts = row_candidate_matching_layouts(
            &row,
            "/reference_raw_vs_rust_loaded/max_abs_delta",
            1.0e-3,
        );
        let shared_layouts = reference_layouts
            .iter()
            .filter(|layout| rust_layouts.contains(layout))
            .collect::<Vec<_>>();

        assert_eq!(reference_layouts, vec!["reference_layout".to_string()]);
        assert_eq!(rust_layouts, vec!["rust_layout".to_string()]);
        assert!(
            shared_layouts.is_empty(),
            "different matching layout candidates must not count as embedding authority"
        );
    }

    #[test]
    fn attn_output_same_input_args_parse_defaults_and_overrides() {
        let default_args =
            vec!["xtask".to_string(), "bitnet-reference-attn-output-same-input-parity".to_string()];
        let defaults = parse_attn_output_same_input_args(&default_args).unwrap();
        assert_eq!(defaults.reference, PathBuf::from(DEFAULT_RUN_OUTPUT));
        assert_eq!(defaults.model, None);
        assert_eq!(defaults.weight, DEFAULT_ATTN_OUTPUT_WEIGHT);
        assert_eq!(defaults.output, Some(PathBuf::from(DEFAULT_ATTN_OUTPUT_SAME_INPUT_OUTPUT)));
        assert_eq!(defaults.format, "human");

        let args = vec![
            "xtask".to_string(),
            "bitnet-reference-attn-output-same-input-parity".to_string(),
            "--reference".to_string(),
            "ref.json".to_string(),
            "--model".to_string(),
            "model.gguf".to_string(),
            "--weight".to_string(),
            "blk.1.attn_output.weight".to_string(),
            "--output".to_string(),
            "out.json".to_string(),
            "--format".to_string(),
            "json".to_string(),
        ];
        let parsed = parse_attn_output_same_input_args(&args).unwrap();
        assert_eq!(parsed.reference, PathBuf::from("ref.json"));
        assert_eq!(parsed.model, Some(PathBuf::from("model.gguf")));
        assert_eq!(parsed.weight, "blk.1.attn_output.weight");
        assert_eq!(parsed.output, Some(PathBuf::from("out.json")));
        assert_eq!(parsed.format, "json");
    }

    #[test]
    fn attn_output_same_input_report_blocks_when_reference_prefix_missing() {
        let dir = tempdir().unwrap();
        let reference = dir.path().join("reference.json");
        let model = dir.path().join("missing.gguf");
        write_file(
            &reference,
            &serde_json::to_string_pretty(&json!({
                "receipt_type": "bitnet_reference_layer_trace",
                "records": [
                    {
                        "name": "attn_sub_norm-0",
                        "stage": "attn_sub_norm",
                        "graph_index": 1,
                        "layer": 0,
                        "dtype": "f32",
                        "shape": [2, 1, 1, 1],
                        "nelements": 2,
                        "first_values": [],
                        "values_available": true,
                        "stats": {"rms": 1.0}
                    },
                    {
                        "name": "attn_o_out-0",
                        "stage": "attn_o_out",
                        "graph_index": 2,
                        "layer": 0,
                        "dtype": "f32",
                        "shape": [2, 1, 1, 1],
                        "nelements": 2,
                        "first_values": [1.0, 2.0],
                        "values_available": true,
                        "stats": {"rms": 1.0}
                    }
                ]
            }))
            .unwrap(),
        );

        let report = build_attn_output_same_input_parity(&AttnOutputSameInputArgs {
            reference,
            model: Some(model),
            weight: DEFAULT_ATTN_OUTPUT_WEIGHT.to_string(),
            output: None,
            format: "json".to_string(),
        })
        .unwrap();
        let reasons =
            report.pointer("/decision/current_blocked_reasons").and_then(Value::as_array).unwrap();

        assert_eq!(
            report.pointer("/receipt_type"),
            Some(&json!("bitnet_reference_attn_output_same_input_parity"))
        );
        assert_eq!(report.pointer("/claim_allowed"), Some(&json!(false)));
        assert_eq!(report.pointer("/diagnostic_only"), Some(&json!(true)));
        assert_eq!(
            report.pointer("/decision/same_input_projection_available"),
            Some(&json!(false))
        );
        assert!(reasons.contains(&json!("model_gguf_missing")));
        assert!(reasons.contains(&json!("reference_attn_sub_norm_first_values_missing")));
        assert_eq!(report.pointer("/not_claims"), Some(&json!(CRITICAL_NOT_CLAIMS)));
    }

    #[test]
    fn vector_compare_reports_first_mismatch_and_hash_match() {
        let same = compare_vectors(&[1.0, 2.0], &[1.0, 2.0]);
        let different = compare_vectors(&[1.0, 2.0], &[1.0, 2.5]);

        assert_eq!(same.pointer("/sha256_match"), Some(&json!(true)));
        assert_eq!(same.pointer("/first_mismatch_index"), Some(&Value::Null));
        assert_eq!(different.pointer("/sha256_match"), Some(&json!(false)));
        assert_eq!(different.pointer("/first_mismatch_index"), Some(&json!(1)));
        assert_eq!(different.pointer("/max_abs_delta"), Some(&json!(0.5)));
    }
}
