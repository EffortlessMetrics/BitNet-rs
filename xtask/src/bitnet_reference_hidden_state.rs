use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_CPP_ROOT: &str = "target/external/BitNet-reference/3rdparty/llama.cpp";
const DEFAULT_OUTPUT: &str = "target/a770-diagnostic/bitnet-reference-hidden-state-plan.json";

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
        _ => Ok(false),
    }
}

fn print_help() {
    println!(
        "Plan target-local BitNet reference hidden-state instrumentation for first-token divergence localization\n\nUsage: xtask.exe bitnet-reference-hidden-state-plan [OPTIONS]\n\nOptions:\n      --cpp-root <PATH>  llama.cpp checkout root [default: target/external/BitNet-reference/3rdparty/llama.cpp]\n      --output <PATH>    Output JSON receipt [default: target/a770-diagnostic/bitnet-reference-hidden-state-plan.json]\n      --format <FORMAT>  Output format: human or json [default: human]\n  -h, --help             Print help"
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
    path.to_string_lossy().replace('\\', "/")
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
