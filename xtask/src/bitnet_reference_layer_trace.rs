use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_CPP_ROOT: &str = "target/external/BitNet-reference/3rdparty/llama.cpp";
const DEFAULT_RUST_TRANSFORMER: &str = "crates/bitnet-transformer/src/lib.rs";
const DEFAULT_OUTPUT: &str = "target/a770-diagnostic/bitnet-reference-layer-trace-plan.json";

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
struct SourceText {
    path: PathBuf,
    exists: bool,
    read_ok: bool,
    sha256: Option<String>,
    text: String,
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
        _ => Ok(false),
    }
}

fn print_help() {
    println!(
        "Plan target-local BitNet reference layer/stage trace instrumentation\n\nUsage: xtask.exe bitnet-reference-layer-trace-plan [OPTIONS]\n\nOptions:\n      --cpp-root <PATH>          llama.cpp checkout root [default: target/external/BitNet-reference/3rdparty/llama.cpp]\n      --rust-transformer <PATH>  Rust transformer source [default: crates/bitnet-transformer/src/lib.rs]\n      --output <PATH>            Output JSON receipt [default: target/a770-diagnostic/bitnet-reference-layer-trace-plan.json]\n      --format <FORMAT>          Output format: human or json [default: human]\n  -h, --help                     Print help"
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
}
