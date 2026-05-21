//! CLI argument parsing and subcommand routing tests.
//!
//! Tests the `bitnet` binary's argument parsing using `assert_cmd` for the `Run`
//! subcommand (defined in main.rs, not exported as a library type) and
//! `clap::Parser::try_parse_from` for `InferenceCommand` (used by `inference`
//! and `chat` subcommands).
//!
//! ## Coverage vs existing test files
//!
//! | Area | Existing file | This file |
//! |------|--------------|-----------|
//! | `Run` subcommand required args | — | ✓ |
//! | `Run` subcommand type validation | — | ✓ |
//! | `Run` default values in help | — | ✓ |
//! | `generate` alias for `run` | — | ✓ |
//! | Subcommand routing (non-`run`) | cli_smoke.rs (partial) | ✓ (list-architectures, list-templates, info, config, tokenize, compat-check) |
//! | `--interface-version` flag | — | ✓ |
//! | InferenceCommand --seed | — | ✓ |
//! | InferenceCommand --deterministic | — | ✓ |
//! | InferenceCommand --top-k validation | — | ✓ |
//! | InferenceCommand --temperature range | cli_extended_tests.rs (default only) | ✓ (0.0, 2.0, rejection) |
//! | InferenceCommand full config combo | cli_arg_validation_tests.rs (partial) | ✓ (with --seed, --deterministic) |

use assert_cmd::Command;
use predicates::prelude::*;

#[allow(deprecated)]
fn bitnet() -> Command {
    Command::cargo_bin("bitnet").expect("bitnet binary must be buildable")
}

fn workspace_path(path: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(path)
}

fn write_bitnet_chat_streaming_semantics_receipt(
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_apple_m4_chat_streaming_semantics",
        "requested_backend": "apple-m4-cpu-neon",
        "selected_backend": "apple-m4-cpu-neon",
        "runtime_api": "cpu",
        "fallback_used": false,
        "streaming_semantics": {
            "token_order_preserved": true,
            "final_receipt_exported": true
        }
    });
    std::fs::write(path, serde_json::to_vec_pretty(&receipt)?)?;
    Ok(())
}

fn write_bitnet_chat_session_receipt(
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let source = workspace_path(
        "ci/hardware/apple-m4-mac-mini/2026-05-17T0847Z/bitnet-warm/variable-warm-session.json",
    );
    let mut receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(source)?)?;
    let prompt_count = receipt["session"]["prompt_count"].clone();
    let object = receipt.as_object_mut().ok_or("receipt must be object")?;
    object.insert("artifact_kind".to_string(), serde_json::json!("bitnet_apple_m4_chat_session"));
    object.insert("operator_command".to_string(), serde_json::json!("mac chat"));
    object.insert("model_id".to_string(), serde_json::json!("microsoft-bitnet-b1.58-2B-4T-i2s"));
    object.insert(
        "bitnet_chat_gate".to_string(),
        serde_json::json!({
            "path": "ci/hardware/apple-m4-mac-mini/2026-05-17T0000Z/bitnet-chat-gate/gate.json",
            "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "status": "ready_to_enable",
            "gate_passed": true,
            "validated": true,
            "work_item": "M4-BITNET-EX-006"
        }),
    );
    object.insert(
        "bitnet_chat".to_string(),
        serde_json::json!({
            "enabled": true,
            "route": "bitnet mac chat --model-family bitnet",
            "prompt_count": prompt_count,
            "serve_enabled": false,
            "streaming_requested": true,
            "per_turn_receipts_enabled": true,
            "gate_required": true
        }),
    );
    object.insert(
        "mac_bitnet_claim_boundary".to_string(),
        serde_json::json!({
            "bitnet_chat_session": true,
            "answer_corpus_proof_gate": "MODEL-ARTIFACT-007/M4-QA-001",
            "chat_gate_work_item": "M4-BITNET-EX-006",
            "requested_backend": "apple-m4-cpu-neon",
            "tokenizer_path": "models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json",
            "tokenizer_sha256": "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7",
            "chat_enabled": true,
            "serve_enabled": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false
        }),
    );
    if let Some(claim_boundary) =
        object.get_mut("claim_boundary").and_then(|value| value.as_object_mut())
    {
        claim_boundary.insert("bitnet_chat_session".to_string(), serde_json::json!(true));
        claim_boundary.insert("chat_enabled".to_string(), serde_json::json!(true));
        claim_boundary.insert("serve_enabled".to_string(), serde_json::json!(false));
        claim_boundary.insert("qk256_apple_claimed".to_string(), serde_json::json!(false));
        claim_boundary
            .insert("neural_engine_execution_claimed".to_string(), serde_json::json!(false));
        claim_boundary.insert("mpsgraph_inference_claimed".to_string(), serde_json::json!(false));
    }
    std::fs::write(path, serde_json::to_vec_pretty(&receipt)?)?;
    Ok(())
}

fn write_bitnet_serve_streaming_semantics_receipt(
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_apple_m4_serve_streaming_semantics",
        "requested_backend": "apple-m4-cpu-neon",
        "selected_backend": "apple-m4-cpu-neon",
        "runtime_api": "cpu",
        "fallback_used": false,
        "streaming_semantics": {
            "token_order_preserved": true,
            "final_receipt_exported": true,
            "sse_done_sent": true
        },
        "claim_boundary": {
            "production_hosting_claimed": false,
            "openai_compatibility_claimed": false,
            "full_metal_inference_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false
        }
    });
    std::fs::write(path, serde_json::to_vec_pretty(&receipt)?)?;
    Ok(())
}

fn write_bitnet_serve_failure_receipt(
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_apple_m4_serve_failure",
        "requested_backend": "apple-m4-cpu-neon",
        "selected_backend": "apple-m4-cpu-neon",
        "runtime_api": "cpu",
        "fallback_used": false,
        "failure": {
            "stage": "decode",
            "message": "synthetic timeout boundary fixture"
        },
        "timeout_boundary": {
            "enforced": true,
            "reached": true,
            "stage": "decode",
            "configured_seconds": 1
        },
        "mac_bitnet_claim_boundary": {
            "serve_enabled": false,
            "production_hosting_claimed": false,
            "openai_compatibility_claimed": false,
            "full_metal_inference_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false
        },
        "bitnet_quality_claimed": false
    });
    std::fs::write(path, serde_json::to_vec_pretty(&receipt)?)?;
    Ok(())
}

fn write_bitnet_serve_check_receipt(
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_apple_m4_local_server_check",
        "timestamp": "2026-05-17T00:00:00Z",
        "artifact_path": path,
        "result": "pass",
        "model_family": "bitnet",
        "requested_backend": "apple-m4-cpu-neon",
        "selected_backend": "apple-m4-cpu-neon",
        "runtime_api": "cpu",
        "fallback_used": false,
        "server": {
            "url": "http://127.0.0.1:8080",
            "health_endpoint": "/health",
            "ready_endpoint": "/ready",
            "models_endpoint": "/models",
            "completion_endpoint": "/v1/chat/completions",
            "receipt_export_endpoint": "/receipts/{id}"
        },
        "checks": {
            "health": {
                "executed": true,
                "status": 200,
                "passed": true,
                "model_family": "bitnet",
                "generation_executed": false
            },
            "ready": {
                "executed": true,
                "status": 200,
                "passed": true,
                "ready": true,
                "model_family": "bitnet",
                "selected_backend": "apple-m4-cpu-neon",
                "fallback_used": false
            },
            "models": {
                "executed": true,
                "status": 200,
                "passed": true,
                "artifact_kind": "bitnet_apple_m4_local_server_models",
                "resident_model_id": "microsoft-bitnet-b1.58-2B-4T-i2s",
                "generation_executed": false
            },
            "completion": {
                "executed": true,
                "status": 200,
                "passed": true,
                "request_id": "bitnet-serve-test",
                "receipt_id": "bitnet-serve-test",
                "generated_tokens": 1,
                "finish_reason": "length"
            },
            "receipt_export": {
                "executed": true,
                "status": 200,
                "passed": true,
                "request_id": "bitnet-serve-test",
                "artifact_kind": "bitnet_apple_m4_serve_completion",
                "selected_backend": "apple-m4-cpu-neon",
                "fallback_used": false
            }
        },
        "claim_boundary": {
            "server_health_checked": true,
            "server_readiness_checked": true,
            "model_catalog_checked": true,
            "completion_probe_executed": true,
            "receipt_export_checked": true,
            "production_readiness_claimed": false,
            "openai_compatibility_claimed": false,
            "bitnet_quality_claimed": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false
        }
    });
    std::fs::write(path, serde_json::to_vec_pretty(&receipt)?)?;
    Ok(())
}

// ============================================================================
// Run subcommand: required arguments
// ============================================================================

/// `run` requires `--model` — omitting it is a parse error.
#[test]
fn run_requires_model() {
    bitnet()
        .args(["run", "--prompt", "hello"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--model").or(predicate::str::contains("required")));
}

/// `run` requires `--prompt` — omitting it is a parse error.
#[test]
fn run_requires_prompt() {
    bitnet()
        .args(["run", "--model", "fake.gguf"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--prompt").or(predicate::str::contains("required")));
}

/// `run` with both required args missing shows usage.
#[test]
fn run_no_args_shows_usage() {
    bitnet()
        .arg("run")
        .assert()
        .failure()
        .stderr(predicate::str::contains("Usage").or(predicate::str::contains("--model")));
}

// ============================================================================
// Run subcommand: default values visible in help
// ============================================================================

/// `run --help` documents default max-new-tokens of 32.
#[test]
fn run_help_shows_default_max_new_tokens() {
    bitnet().args(["run", "--help"]).assert().success().stdout(
        predicate::str::contains("max-new-tokens").or(predicate::str::contains("max-tokens")),
    );
}

/// `run --help` documents the --temperature option.
#[test]
fn run_help_documents_temperature() {
    bitnet()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--temperature"));
}

/// `run --help` documents the --top-k option.
#[test]
fn run_help_documents_top_k() {
    bitnet().args(["run", "--help"]).assert().success().stdout(predicate::str::contains("--top-k"));
}

/// `run --help` documents the --top-p option.
#[test]
fn run_help_documents_top_p() {
    bitnet().args(["run", "--help"]).assert().success().stdout(predicate::str::contains("--top-p"));
}

/// `run --help` documents the --seed option.
#[test]
fn run_help_documents_seed() {
    bitnet().args(["run", "--help"]).assert().success().stdout(predicate::str::contains("--seed"));
}

/// `run --help` documents the --greedy flag.
#[test]
fn run_help_documents_greedy() {
    bitnet()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--greedy"));
}

#[test]
fn top_level_help_documents_apple_backend_labels() {
    bitnet()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("apple-m4-metal"))
        .stdout(predicate::str::contains("apple-m4-mpsgraph"))
        .stdout(predicate::str::contains("apple-m4-cpu-neon"));
}

#[test]
fn apple_m4_top_level_help_documents_local_answer_boundaries() {
    bitnet()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Apple M4 local answer path"))
        .stdout(predicate::str::contains("apple-m4-cpu-neon: reliable local-answer path"))
        .stdout(predicate::str::contains("apple-m4-metal: receipt-backed Metal phase"))
        .stdout(predicate::str::contains("not native Metal or Neural Engine proof"));
}

#[test]
fn run_help_documents_apple_backend_labels() {
    bitnet()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple-m4-metal"))
        .stdout(predicate::str::contains("apple-m4-mpsgraph"))
        .stdout(predicate::str::contains("apple-m4-cpu-neon"));
}

#[test]
fn apple_m4_run_help_documents_strict_cpu_neon_receipt_flow() {
    bitnet()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Apple M4 local answer path"))
        .stdout(predicate::str::contains("bitnet --device apple-m4-cpu-neon run"))
        .stdout(predicate::str::contains("--strict-loader --strict-tokenizer"))
        .stdout(predicate::str::contains("--json-out local-answer-cpu-neon.json"));
}

#[test]
fn slm_warm_session_help_documents_warm_receipts() {
    bitnet()
        .args(["slm-warm-session", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("one model/tokenizer load"))
        .stdout(predicate::str::contains("--corpus"))
        .stdout(predicate::str::contains("--prompt"))
        .stdout(predicate::str::contains("--fail-on-quality"))
        .stdout(predicate::str::contains("--require-determinism"))
        .stdout(predicate::str::contains("--allocation-audit"))
        .stdout(predicate::str::contains("--stream"))
        .stdout(predicate::str::contains("--progress"))
        .stdout(predicate::str::contains("--quiet"))
        .stdout(predicate::str::contains("--json-out"))
        .stdout(predicate::str::contains("qwen2.5"));
}

#[test]
fn cuda_warm_session_help_documents_strict_cuda_receipts() {
    bitnet()
        .args(["cuda-warm-session", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("RTX 5070 Ti CUDA"))
        .stdout(predicate::str::contains("--prompt"))
        .stdout(predicate::str::contains("--strict-loader"))
        .stdout(predicate::str::contains("--strict-tokenizer"))
        .stdout(predicate::str::contains("--fail-on-quality"))
        .stdout(predicate::str::contains("--json-out"))
        .stdout(predicate::str::contains("bitnetcpp-answer"));
}

#[test]
fn cuda_warm_session_requires_rtx5070ti_device_before_model_load() {
    bitnet()
        .args([
            "--device",
            "cpu",
            "cuda-warm-session",
            "--model",
            "missing.gguf",
            "--tokenizer",
            "missing-tokenizer.json",
            "--prompt",
            "What is 2+2?",
            "--prompt",
            "What is the capital of France?",
            "--strict-loader",
            "--strict-tokenizer",
            "--json-out",
            "target/test-cuda-warm-session.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "cuda-warm-session requires --device nvidia-rtx-5070-ti-cuda",
        ));
}

#[test]
fn cuda_warm_session_requires_multiple_prompts_before_model_load() {
    bitnet()
        .args([
            "--device",
            "nvidia-rtx-5070-ti-cuda",
            "cuda-warm-session",
            "--model",
            "missing.gguf",
            "--tokenizer",
            "missing-tokenizer.json",
            "--prompt",
            "What is 2+2?",
            "--strict-loader",
            "--strict-tokenizer",
            "--json-out",
            "target/test-cuda-warm-session.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "cuda-warm-session requires at least two --prompt values",
        ));
}

#[test]
fn cuda_warm_session_requires_strict_loader_before_model_load() {
    bitnet()
        .args([
            "--device",
            "nvidia-rtx-5070-ti-cuda",
            "cuda-warm-session",
            "--model",
            "missing.gguf",
            "--tokenizer",
            "missing-tokenizer.json",
            "--prompt",
            "What is 2+2?",
            "--prompt",
            "What is the capital of France?",
            "--strict-tokenizer",
            "--json-out",
            "target/test-cuda-warm-session.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cuda-warm-session requires --strict-loader"));
}

#[test]
fn cuda_warm_session_requires_strict_tokenizer_before_model_load() {
    bitnet()
        .args([
            "--device",
            "nvidia-rtx-5070-ti-cuda",
            "cuda-warm-session",
            "--model",
            "missing.gguf",
            "--tokenizer",
            "missing-tokenizer.json",
            "--prompt",
            "What is 2+2?",
            "--prompt",
            "What is the capital of France?",
            "--strict-loader",
            "--json-out",
            "target/test-cuda-warm-session.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cuda-warm-session requires --strict-tokenizer"));
}

#[test]
fn mac_help_documents_operator_wrappers() {
    bitnet()
        .args(["mac", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("models"))
        .stdout(predicate::str::contains("status"))
        .stdout(predicate::str::contains("report-refresh"))
        .stdout(predicate::str::contains("regression-dashboard"))
        .stdout(predicate::str::contains("check"))
        .stdout(predicate::str::contains("ask"))
        .stdout(predicate::str::contains("smoke"))
        .stdout(predicate::str::contains("bitnet-warm"))
        .stdout(predicate::str::contains("bitnet-chat-gate"))
        .stdout(predicate::str::contains("doctor"))
        .stdout(predicate::str::contains("validate"))
        .stdout(predicate::str::contains("eval"))
        .stdout(predicate::str::contains("bitnet-proof"))
        .stdout(predicate::str::contains("receipts-check"));
}

#[test]
fn mac_eval_help_documents_robustness_dry_run() {
    bitnet()
        .args(["mac", "eval", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--suite <SUITE>"))
        .stdout(predicate::str::contains("m4-robustness"))
        .stdout(predicate::str::contains("m4-long-context"))
        .stdout(predicate::str::contains("--dry-run"))
        .stdout(predicate::str::contains("--json-out <PATH>"));
}

#[test]
fn mac_eval_robustness_dry_run_writes_separate_family_summary()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let corpus = workspace_path("ci/quality/apple-m4-robustness-corpus.yaml");
    let receipt = dir.path().join("robustness-summary.json");
    let corpus_str = corpus.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "eval",
            "--suite",
            "m4-robustness",
            "--corpus",
            corpus_str.as_str(),
            "--dry-run",
            "--json-out",
            receipt_str.as_str(),
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_robustness_eval_summary"))
        .stdout(predicate::str::contains("\"model_family\": \"dense_slm\""))
        .stdout(predicate::str::contains("\"model_family\": \"bitnet\""));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "apple_m4_robustness_eval_summary");
    assert_eq!(receipt_json["suite"], "m4-robustness");
    assert_eq!(receipt_json["work_item"], "M4-ROBUSTNESS-001");
    assert_eq!(receipt_json["requested_backend"], "apple-m4-cpu-neon");
    assert_eq!(receipt_json["fallback_used"], false);
    assert_eq!(receipt_json["dry_run"], true);
    assert_eq!(receipt_json["corpus"]["mechanical_scoring_only"], true);
    assert_eq!(receipt_json["corpus"]["required_llm_judge"], false);
    assert_eq!(receipt_json["scoring_summary"]["total"], 24);
    assert_eq!(receipt_json["scoring_summary"]["not_run"], 24);
    assert_eq!(receipt_json["claim_boundary"]["broad_safety_claim"], false);
    assert_eq!(receipt_json["claim_boundary"]["dense_slm_evidence_proves_bitnet"], false);
    assert_eq!(receipt_json["claim_boundary"]["bitnet_chat_enabled"], false);
    assert_eq!(receipt_json["claim_boundary"]["bitnet_serve_enabled"], false);
    let families = receipt_json["families"].as_array().ok_or("missing families")?;
    assert_eq!(families.len(), 2);
    assert!(families.iter().any(|family| {
        family["model_family"] == "dense_slm"
            && family["cases_total"] == 12
            && family["prompt_template"] == "qwen2.5"
    }));
    assert!(families.iter().any(|family| {
        family["model_family"] == "bitnet"
            && family["cases_total"] == 12
            && family["prompt_template"] == "bitnetcpp-answer"
    }));

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_robustness_eval_summary"))
        .stdout(predicate::str::contains("\"prompt_count\": 24"))
        .stdout(predicate::str::contains("\"generated_tokens\": 0"));
    Ok(())
}

#[test]
fn mac_eval_robustness_requires_dry_run_until_live_gate() {
    bitnet()
        .args(["mac", "eval", "--suite", "m4-robustness"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("currently supports only --dry-run"));
}

#[test]
fn mac_eval_long_context_dry_run_writes_contract_summary() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = tempfile::tempdir()?;
    let corpus = workspace_path("ci/quality/apple-m4-long-context-corpus.yaml");
    let receipt = dir.path().join("long-context-summary.json");
    let corpus_str = corpus.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "eval",
            "--suite",
            "m4-long-context",
            "--corpus",
            corpus_str.as_str(),
            "--dry-run",
            "--json-out",
            receipt_str.as_str(),
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_long_context_eval_summary"))
        .stdout(predicate::str::contains("\"suite\": \"m4-long-context\""))
        .stdout(predicate::str::contains("unsupported_until_bitnet_long_context_receipts_exist"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "apple_m4_long_context_eval_summary");
    assert_eq!(receipt_json["suite"], "m4-long-context");
    assert_eq!(receipt_json["work_item"], "M4-CONTEXT-HARNESS-001");
    assert_eq!(receipt_json["requested_backend"], "apple-m4-cpu-neon");
    assert_eq!(receipt_json["fallback_used"], false);
    assert_eq!(receipt_json["dry_run"], true);
    assert_eq!(receipt_json["corpus"]["mechanical_scoring_only"], true);
    assert_eq!(receipt_json["corpus"]["required_llm_judge"], false);
    assert_eq!(receipt_json["coverage"]["retrieval_copy"], true);
    assert_eq!(receipt_json["coverage"]["table_extraction"], true);
    assert_eq!(receipt_json["coverage"]["late_context_instruction_following"], true);
    assert_eq!(receipt_json["coverage"]["truncation_behavior"], true);
    assert_eq!(receipt_json["claim_boundary"]["live_long_context_quality_claim"], false);
    assert_eq!(receipt_json["claim_boundary"]["dense_slm_evidence_proves_bitnet"], false);
    assert_eq!(receipt_json["evidence_status"]["live_quality_receipts_published"], false);
    let families = receipt_json["families"].as_array().ok_or("missing families")?;
    assert_eq!(families.len(), 2);
    assert!(families.iter().any(|family| {
        family["model_family"] == "dense_slm"
            && family["long_context_supported_for_live_run"] == true
    }));
    assert!(families.iter().any(|family| {
        family["model_family"] == "bitnet"
            && family["long_context_supported_for_live_run"] == false
            && family["unsupported_boundary"]["dense_slm_evidence_proves_bitnet"] == false
    }));

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_long_context_eval_summary"))
        .stdout(predicate::str::contains("\"prompt_count\": 8"))
        .stdout(predicate::str::contains("\"generated_tokens\": 0"));
    Ok(())
}

#[test]
fn mac_eval_long_context_live_requires_ready_model_cache_before_receipt()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache_dir = dir.path().join("cache");
    let receipt = dir.path().join("long-context-live.json");
    let cache_str = cache_dir.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "eval",
            "--suite",
            "m4-long-context",
            "--cache-dir",
            cache_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cached Apple M4 SLM model"))
        .stderr(predicate::str::contains("is not ready"));
    assert!(!receipt.exists(), "preflight failure should not write a proof receipt");
    Ok(())
}

#[test]
fn mac_eval_long_context_rejects_duplicate_case_ids() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let corpus = dir.path().join("duplicate-long-context.yaml");
    let receipt = dir.path().join("long-context-summary.json");
    std::fs::write(
        &corpus,
        r#"schema: 1
artifact_kind: apple_m4_long_context_corpus
name: duplicate-long-context
description: Duplicate case id fixture.
defaults:
  families: [dense_slm, bitnet]
  family_prompt_templates:
    dense_slm: qwen2.5
    bitnet: bitnetcpp-answer
cases:
- id: duplicate
  category: retrieval_copy
  question: "What key?"
  scoring:
    kind: normalized_match
    expected_normalized: alpha
- id: duplicate
  category: table_extraction
  question: "What route?"
  scoring:
    kind: normalized_match
    expected_normalized: beta
- id: late
  category: late_context_instruction_following
  question: "Final code?"
  scoring:
    kind: required_keywords
    required_keywords: [final]
- id: truncation
  category: truncation_behavior
  question: "State?"
  scoring:
    kind: normalized_match
    expected_normalized: unsupported_context
"#,
    )?;
    let corpus_str = corpus.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "eval",
            "--suite",
            "m4-long-context",
            "--corpus",
            corpus_str.as_str(),
            "--dry-run",
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("case id `duplicate` is duplicated"));
    Ok(())
}

#[test]
fn mac_receipts_check_rejects_robustness_broad_safety_claim()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let corpus = workspace_path("ci/quality/apple-m4-robustness-corpus.yaml");
    let receipt = dir.path().join("robustness-summary.json");
    let corpus_str = corpus.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "eval",
            "--suite",
            "m4-robustness",
            "--corpus",
            corpus_str.as_str(),
            "--dry-run",
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .success();

    let mut receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    receipt_json["claim_boundary"]["broad_safety_claim"] = serde_json::json!(true);
    std::fs::write(&receipt, serde_json::to_vec_pretty(&receipt_json)?)?;

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("claim_boundary.broad_safety_claim"));
    Ok(())
}

#[test]
fn mac_models_lists_operator_model_states() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let cache_str = cache.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "models", "--cache-dir", cache_str.as_str()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Default model: qwen2.5-0.5b-instruct-q8_0"))
        .stdout(predicate::str::contains("Disk:"))
        .stdout(predicate::str::contains("Recommendation:"))
        .stdout(predicate::str::contains("Next fetch: bitnet model fetch"))
        .stdout(predicate::str::contains("Next verify: bitnet model verify"))
        .stdout(predicate::str::contains("low_disk="))
        .stdout(predicate::str::contains("qwen2.5-0.5b-instruct-q8_0"))
        .stdout(predicate::str::contains("default"))
        .stdout(predicate::str::contains("qwen2.5-0.5b-instruct-q4_k_m"))
        .stdout(predicate::str::contains("qwen2.5-1.5b-instruct-q4_k_m"))
        .stdout(predicate::str::contains("supported"))
        .stdout(predicate::str::contains("microsoft-bitnet-b1.58-2B-4T-i2s"))
        .stdout(predicate::str::contains("supported-ask"))
        .stdout(predicate::str::contains("candidate"))
        .stdout(predicate::str::contains("rejected"))
        .stdout(predicate::str::contains("one-shot ask or fixed warm route"))
        .stdout(predicate::str::contains("Proof bridge: microsoft-bitnet-b1.58-2B-4T-i2s"))
        .stdout(predicate::str::contains("mac bitnet-proof --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"))
        .stdout(predicate::str::contains("mac bitnet-warm"))
        .stdout(predicate::str::contains("--proof-receipt ci/hardware/apple-m4-mac-mini/YYYY-MM-DD/bitnet-local-answer/bitnet-answer-corpus-full-release.json"));
    Ok(())
}

#[test]
fn mac_models_json_exposes_claim_boundaries_without_fetching()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let cache_str = cache.to_string_lossy().into_owned();

    let output = bitnet()
        .args(["mac", "models", "--cache-dir", cache_str.as_str(), "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: serde_json::Value = serde_json::from_slice(&output)?;

    assert_eq!(json["default_model_id"], "qwen2.5-0.5b-instruct-q8_0");
    assert!(
        json["disk"]["default_model_headroom_bytes"].as_u64().is_some_and(|headroom| headroom > 0)
    );
    let recommendation = json["disk"]["recommendation"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("disk recommendation"))?;
    assert!(recommendation.contains("default") || recommendation.contains("Disk"));
    let claim_boundary =
        json["claim_boundary"].as_str().ok_or_else(|| std::io::Error::other("claim boundary"))?;
    assert!(claim_boundary.contains("fixed-prompt warm-session"));
    let rows = json["rows"].as_array().ok_or_else(|| std::io::Error::other("rows"))?;
    assert!(rows.iter().any(|row| {
        row["id"] == "qwen2.5-0.5b-instruct-q8_0"
            && row["state"] == "default"
            && row["recommended_fetch_headroom_bytes"].as_u64().is_some()
    }));
    assert!(
        rows.iter().any(|row| {
            row["id"] == "qwen2.5-1.5b-instruct-q4_k_m" && row["state"] == "supported"
        })
    );
    assert!(rows.iter().any(|row| {
        row["id"] == "microsoft-bitnet-b1.58-2B-4T-i2s"
            && row["state"] == "supported-ask"
            && row["selection"]
                == "explicit --model-id with --model-path/--tokenizer for one-shot ask or fixed warm route only"
            && row["mac_ask_enabled"] == true
            && row["mac_bitnet_warm_enabled"] == true
            && row["mac_chat_enabled"] == false
            && row["mac_ask_chat_enabled"] == false
            && row["mac_serve_enabled"] == false
            && row["proof_status"] == "answer-corpus-and-warm-session-proof-passed-explicit-artifact"
            && row["proof_command"]
                .as_str()
                .is_some_and(|command| command.contains("mac bitnet-proof --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"))
            && row["proof_receipt_path"]
                == "ci/hardware/apple-m4-mac-mini/YYYY-MM-DD/bitnet-local-answer/bitnet-answer-corpus-full-release.json"
            && row["warm_command"].as_str().is_some_and(|command| {
                command.contains("mac bitnet-warm")
                    && command.contains("--model-path models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
                    && command.contains("--tokenizer models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json")
            })
            && row["warm_receipt_path"]
                == "ci/hardware/apple-m4-mac-mini/2026-05-14/bitnet-warm/bitnet-mac-bitnet-warm-runtime-receipt.json"
            && row["recommended_fetch_headroom_bytes"].as_u64().is_some()
            && row["fetch_command"].as_str().is_some_and(|command| command.contains("bitnet model fetch microsoft-bitnet"))
    }));
    assert!(rows.iter().any(|row| row["state"] == "candidate"));
    assert!(rows.iter().any(|row| row["state"] == "rejected"));
    Ok(())
}

#[test]
fn mac_status_writes_operator_readiness_receipt() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let receipt = dir.path().join("mac-status.json");
    let cache_str = cache.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "status",
            "--cache-dir",
            cache_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Apple M4 inference status"))
        .stdout(predicate::str::contains("Dense SLM:"))
        .stdout(predicate::str::contains("BitNet:"))
        .stdout(predicate::str::contains("Dense readiness:"))
        .stdout(predicate::str::contains("BitNet readiness:"))
        .stdout(predicate::str::contains("Last receipts:"))
        .stdout(predicate::str::contains("Disabled: BitNet chat=false, BitNet serve=false"))
        .stdout(predicate::str::contains("no live model run"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "apple_m4_inference_status");
    assert_eq!(receipt_json["operator_command"], "mac status");
    assert_eq!(receipt_json["fallback_used"], false);
    assert_eq!(receipt_json["dense_slm"]["supported_model_count"], 3);
    assert_eq!(receipt_json["dense_slm"]["ask_enabled"], true);
    assert_eq!(receipt_json["dense_slm"]["chat_enabled"], true);
    assert_eq!(receipt_json["dense_slm"]["serve_enabled"], true);
    assert_eq!(receipt_json["bitnet"]["ask_enabled"], true);
    assert_eq!(receipt_json["bitnet"]["warm_enabled"], true);
    assert_eq!(receipt_json["bitnet"]["chat_enabled"], false);
    assert_eq!(receipt_json["bitnet"]["serve_enabled"], false);
    assert_eq!(receipt_json["readiness"]["dense_slm"]["status"], "cache_repair_required");
    assert!(
        receipt_json["readiness"]["dense_slm"]["cache_repair_guidance"]
            .as_str()
            .is_some_and(|guidance| guidance.contains("bitnet model fetch"))
    );
    assert!(
        receipt_json["readiness"]["dense_slm"]["last_matching_receipts"]["eval"]
            .as_str()
            .is_some_and(|path| path.contains("slm-eval-v2"))
    );
    assert!(
        receipt_json["readiness"]["bitnet"]["disabled_surfaces"]
            .as_array()
            .is_some_and(|surfaces| surfaces.iter().any(|surface| surface == "chat")
                && surfaces.iter().any(|surface| surface == "serve"))
    );
    assert_eq!(receipt_json["readiness"]["bitnet"]["chat_enabled"], false);
    assert_eq!(receipt_json["readiness"]["bitnet"]["serve_enabled"], false);
    assert!(
        receipt_json["readiness"]["bitnet"]["last_matching_receipts"]["variable_warm"]
            .as_str()
            .is_some_and(|path| path.contains("bitnet-productization"))
    );
    assert!(receipt_json["readiness"]["disk_pressure"]["low_disk"].as_bool().is_some());
    assert_eq!(receipt_json["claim_boundary"]["no_live_model_run"], true);
    assert_eq!(receipt_json["claim_boundary"]["dense_slm_and_bitnet_evidence_separated"], true);
    assert_eq!(receipt_json["claim_boundary"]["full_metal_inference_claimed"], false);
    assert_eq!(receipt_json["route_state_matrix"]["work_item"], "M4-ROUTE-MATRIX-001");
    let route_rows = receipt_json["route_state_matrix"]["rows"]
        .as_array()
        .ok_or("route_state_matrix.rows must be an array")?;
    assert!(route_rows.iter().any(|row| {
        row["route_id"] == "dense_slm_serve_loopback"
            && row["state"] == "enabled"
            && row["command_surface"] == "serve"
    }));
    assert!(route_rows.iter().any(|row| {
        row["route_id"] == "bitnet_warm_session"
            && row["state"] == "batch_only"
            && row["command_surface"] == "warm_session"
    }));
    assert!(route_rows.iter().any(|row| {
        row["route_id"] == "bitnet_chat_gate_required"
            && row["state"] == "disabled"
            && row["required_gate"]["artifact_kind"] == "bitnet_apple_m4_chat_gate"
    }));
    assert!(route_rows.iter().any(|row| {
        row["route_id"] == "unsupported_apple_backends" && row["state"] == "unsupported"
    }));
    assert!(
        receipt_json["commands"]["bitnet_chat_gate"]
            .as_str()
            .is_some_and(|command| command.contains("bitnet mac bitnet-chat-gate"))
    );
    assert_eq!(receipt_json["commands"]["report_refresh"], "bitnet mac report-refresh");

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_inference_status"))
        .stdout(predicate::str::contains("\"prompt_count\": 0"));
    Ok(())
}

#[test]
fn mac_evidence_writes_operator_summary() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let report_root = workspace_path("ci/hardware/apple-m4-mac-mini");
    let receipt = dir.path().join("evidence-summary.json");
    let cache_str = cache.to_string_lossy().into_owned();
    let report_root_str = report_root.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "evidence",
            "--cache-dir",
            cache_str.as_str(),
            "--root",
            report_root_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Apple M4 evidence summary"))
        .stdout(predicate::str::contains("Last dense report:"))
        .stdout(predicate::str::contains("Last BitNet report:"))
        .stdout(predicate::str::contains("Regressions:"))
        .stdout(predicate::str::contains("no live model run"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "apple_m4_operator_evidence_summary");
    assert_eq!(receipt_json["operator_command"], "mac evidence");
    assert_eq!(receipt_json["fallback_used"], false);
    assert_eq!(receipt_json["evidence_contract"]["committed_reports_only"], true);
    assert_eq!(receipt_json["evidence_contract"]["no_live_model_run"], true);
    assert_eq!(receipt_json["default_model"]["id"], "qwen2.5-0.5b-instruct-q8_0");
    assert_eq!(receipt_json["supported_models"]["dense_slm_supported_count"], 3);
    assert_eq!(receipt_json["supported_models"]["bitnet_state"], "supported-ask");
    assert_eq!(receipt_json["supported_models"]["bitnet_chat_enabled"], false);
    assert_eq!(receipt_json["supported_models"]["bitnet_serve_enabled"], false);
    assert!(
        receipt_json["reports"]["last_dense_report"]
            .as_str()
            .is_some_and(|path| { path.contains("slm-eval-v2") && path.ends_with("summary.json") })
    );
    assert!(receipt_json["reports"]["last_bitnet_report"].as_str().is_some_and(|path| {
        path.contains("bitnet-eval-250") && path.ends_with("larger-corpus-decision.json")
    }));
    assert_eq!(receipt_json["claim_boundary"]["dense_slm_and_bitnet_evidence_separated"], true);
    assert_eq!(receipt_json["unsupported_claims"]["full_metal_inference"], false);
    assert_eq!(receipt_json["unsupported_claims"]["qk256"], false);
    assert_eq!(receipt_json["route_state_matrix"]["work_item"], "M4-ROUTE-MATRIX-001");
    assert_eq!(
        receipt_json["route_state_matrix"]["claim_boundary"]["does_not_enable_routes"],
        true
    );
    assert!(receipt_json["route_state_matrix"]["rows"].as_array().is_some_and(|rows| {
        rows.iter().any(|row| row["route_id"] == "dense_slm_streaming" && row["state"] == "enabled")
            && rows.iter().any(|row| {
                row["route_id"] == "bitnet_serve_gate_required" && row["state"] == "disabled"
            })
    }));
    assert!(
        receipt_json["recommended_next_command"]
            .as_str()
            .is_some_and(|command| command.starts_with("bitnet "))
    );

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_operator_evidence_summary"))
        .stdout(predicate::str::contains("\"prompt_count\": 0"));
    Ok(())
}

#[test]
fn mac_report_refresh_writes_model_free_manifest() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let report_root = workspace_path("ci/hardware/apple-m4-mac-mini");
    let receipt = dir.path().join("report-refresh-manifest.json");
    let report_root_str = report_root.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "report-refresh",
            "--root",
            report_root_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
            "--explain",
            "--open-targets",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Apple M4 report refresh manifest"))
        .stdout(predicate::str::contains("dense_slm_eval_v2"))
        .stdout(predicate::str::contains("bitnet_benchmark"))
        .stdout(predicate::str::contains("Status explanations"))
        .stdout(predicate::str::contains("Open targets"))
        .stdout(predicate::str::contains("no live model run"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "apple_m4_report_refresh_manifest");
    assert_eq!(receipt_json["operator_command"], "mac report-refresh");
    assert!(
        receipt_json["operator_affordances"]["explain_command"]
            .as_str()
            .is_some_and(|command| command.contains("--explain"))
    );
    assert!(
        receipt_json["operator_affordances"]["open_targets_command"]
            .as_str()
            .is_some_and(|command| command.contains("--open-targets"))
    );
    assert!(receipt_json["status_explanations"]["comparable"]["meaning"].as_str().is_some());
    assert_eq!(receipt_json["fallback_used"], false);
    assert_eq!(receipt_json["refresh_modes"]["generic_pr_ci_model_free"], true);
    assert_eq!(receipt_json["refresh_modes"]["generic_pr_ci_live_model_run"], false);
    assert_eq!(receipt_json["claim_boundary"]["manifest_only"], true);
    assert_eq!(receipt_json["claim_boundary"]["no_live_model_run"], true);
    assert_eq!(receipt_json["claim_boundary"]["dense_slm_and_bitnet_evidence_separated"], true);
    assert_eq!(receipt_json["claim_boundary"]["broad_performance_claim"], false);
    let families =
        receipt_json["families"].as_array().ok_or_else(|| std::io::Error::other("families"))?;
    assert!(families.iter().any(|family| {
        family["id"] == "dense_slm_eval_v2"
            && family["evidence_family"] == "dense_slm"
            && family["expected_artifact_kind"] == "apple_m4_slm_eval_summary"
            && family["operator_status"] == "comparable"
            && family["operator_status_reason"]
                .as_str()
                .is_some_and(|reason| reason.contains("committed report"))
            && family["report_count"].as_u64().is_some_and(|count| count >= 3)
            && family["generic_pr_ci"]["live_model_run"] == false
            && family["claim_boundary"]["bitnet_evidence"] == false
    }));
    assert!(families.iter().any(|family| {
        family["id"] == "bitnet_eval"
            && family["evidence_family"] == "bitnet"
            && family["expected_artifact_kind"] == "bitnet_apple_m4_local_answer_corpus"
            && family["report_count"].as_u64().is_some_and(|count| count >= 1)
            && family["claim_boundary"]["dense_slm_evidence"] == false
    }));
    assert!(families.iter().any(|family| {
        family["id"] == "bitnet_variable_warm"
            && family["expected_artifact_kind"] == "bitnet_apple_m4_warm_session"
    }));
    assert!(families.iter().any(|family| {
        family["id"] == "dense_slm_benchmark_variance"
            && family["expected_artifact_kind"] == "apple_m4_benchmark_variance_v1"
            && family["evidence_family"] == "dense_slm"
    }));
    assert!(families.iter().any(|family| {
        family["id"] == "bitnet_benchmark_variance"
            && family["expected_artifact_kind"] == "bitnet_apple_m4_benchmark_v1"
            && family["evidence_family"] == "bitnet"
    }));

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_report_refresh_manifest"))
        .stdout(predicate::str::contains("\"prompt_count\": 0"));
    Ok(())
}

#[test]
fn mac_regression_dashboard_writes_model_free_artifacts() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = tempfile::tempdir()?;
    let report_root = workspace_path("ci/hardware/apple-m4-mac-mini");
    let receipt = dir.path().join("regression-dashboard.json");
    let markdown = dir.path().join("regression-dashboard.md");
    let report_root_str = report_root.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();
    let markdown_str = markdown.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "regression-dashboard",
            "--root",
            report_root_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
            "--markdown-out",
            markdown_str.as_str(),
            "--explain",
            "--open-targets",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Apple M4 regression dashboard"))
        .stdout(predicate::str::contains("Group explanations"))
        .stdout(predicate::str::contains("Open targets"))
        .stdout(predicate::str::contains("dense SLM and BitNet evidence stay separate"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "apple_m4_regression_dashboard");
    assert_eq!(receipt_json["operator_command"], "mac regression-dashboard");
    assert!(
        receipt_json["operator_affordances"]["explain_command"]
            .as_str()
            .is_some_and(|command| command.contains("--explain"))
    );
    assert!(
        receipt_json["operator_affordances"]["open_markdown_hint"]
            .as_str()
            .is_some_and(|hint| hint.contains("open "))
    );
    assert!(
        receipt_json["status_explanations"]["insufficient_history"]["next_action"]
            .as_str()
            .is_some_and(|action| action.contains("second matching report"))
    );
    assert_eq!(receipt_json["fallback_used"], false);
    assert_eq!(receipt_json["dashboard_contract"]["model_free"], true);
    assert_eq!(receipt_json["dashboard_contract"]["matching_requires_same_model_id"], true);
    assert_eq!(
        receipt_json["dashboard_contract"]["matching_requires_same_tokenizer_authority"],
        true
    );
    assert_eq!(receipt_json["claim_boundary"]["dashboard_only"], true);
    assert_eq!(receipt_json["claim_boundary"]["no_live_model_run"], true);
    assert_eq!(receipt_json["claim_boundary"]["dense_slm_and_bitnet_evidence_separated"], true);
    let families =
        receipt_json["families"].as_array().ok_or_else(|| std::io::Error::other("families"))?;
    assert!(families.iter().any(|family| {
        family["id"] == "dense_slm_benchmark_v2"
            && family["evidence_family"] == "dense_slm"
            && family["group_count"].as_u64().is_some_and(|count| count >= 3)
            && family["groups"].as_array().is_some_and(|groups| {
                groups.iter().any(|group| {
                    group["operator_status"] == "comparable"
                        && group["operator_status_reason"]
                            .as_str()
                            .is_some_and(|reason| reason.contains("matching reports"))
                        && group["open_targets"]["latest_report"].as_str().is_some()
                })
            })
            && family["claim_boundary"]["bitnet_evidence"] == false
    }));
    assert!(families.iter().any(|family| {
        family["id"] == "bitnet_eval"
            && family["evidence_family"] == "bitnet"
            && family["group_count"].as_u64().is_some_and(|count| count >= 1)
            && family["claim_boundary"]["dense_slm_evidence"] == false
    }));
    let markdown_body = std::fs::read_to_string(&markdown)?;
    assert!(markdown_body.contains("Apple M4 Inference Regression Dashboard"));
    assert!(markdown_body.contains("dense_slm_benchmark_v2"));
    assert!(markdown_body.contains("dense_slm_benchmark_variance"));
    assert!(markdown_body.contains("bitnet_benchmark_variance"));
    assert!(markdown_body.contains("bitnet_eval"));

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_regression_dashboard"))
        .stdout(predicate::str::contains("\"prompt_count\": 0"));
    Ok(())
}

#[test]
fn mac_receipts_check_rejects_regression_dashboard_unsupported_claim()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let report_root = workspace_path("ci/hardware/apple-m4-mac-mini");
    let receipt = dir.path().join("regression-dashboard.json");
    let markdown = dir.path().join("regression-dashboard.md");
    let report_root_str = report_root.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();
    let markdown_str = markdown.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "regression-dashboard",
            "--root",
            report_root_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
            "--markdown-out",
            markdown_str.as_str(),
        ])
        .assert()
        .success();

    let mut receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    receipt_json["claim_boundary"]["dashboard_only"] = serde_json::json!(false);
    std::fs::write(&receipt, serde_json::to_vec_pretty(&receipt_json)?)?;

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("dashboard_only"));
    Ok(())
}

#[test]
fn mac_receipts_check_accepts_m4_report_ops_run_identity() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = tempfile::tempdir()?;
    let report_root = workspace_path("ci/hardware/apple-m4-mac-mini");
    let receipt = dir.path().join("report-refresh-manifest.json");
    let report_root_str = report_root.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "report-refresh",
            "--root",
            report_root_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .success();

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["schema_version"], "1.2.0");
    assert_eq!(receipt_json["run_identity"]["contract_version"], "m4-run-identity-v1");
    assert_eq!(receipt_json["run_identity"]["machine_id"], "apple-m4-mac-mini");
    assert_eq!(receipt_json["run_identity"]["soc"], "apple-m4");
    assert_eq!(receipt_json["run_identity"]["artifact_kind"], "apple_m4_report_refresh_manifest");
    assert_eq!(receipt_json["run_identity"]["backend"]["fallback_used"], false);
    assert!(receipt_json["run_identity_sha256"].as_str().is_some_and(|sha| sha.len() == 64));

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_report_refresh_manifest"));
    Ok(())
}

#[test]
fn mac_receipts_check_rejects_m4_report_ops_missing_run_identity()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let report_root = workspace_path("ci/hardware/apple-m4-mac-mini");
    let receipt = dir.path().join("report-refresh-manifest.json");
    let report_root_str = report_root.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "report-refresh",
            "--root",
            report_root_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .success();

    let mut receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    receipt_json.as_object_mut().unwrap().remove("run_identity");
    std::fs::write(&receipt, serde_json::to_vec_pretty(&receipt_json)?)?;

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("run_identity"));
    Ok(())
}

#[test]
fn mac_models_rejects_full_metal_request_before_cache_lookup() {
    bitnet()
        .args(["--device", "apple-m4-metal", "mac", "models"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac models routes the supported Mac local-answer path"))
        .stderr(predicate::str::contains("Full apple-m4-metal inference"));
}

#[test]
fn mac_validate_help_documents_operator_profile_set() {
    bitnet()
        .args(["mac", "validate", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--profile-set"))
        .stdout(predicate::str::contains("16/32/64 profiles"))
        .stdout(predicate::str::contains("performance"))
        .stdout(predicate::str::contains("16/32/64/128"))
        .stdout(predicate::str::contains("--allocation-audit"))
        .stdout(predicate::str::contains("--progress"))
        .stdout(predicate::str::contains("--quiet"));
}

#[test]
fn mac_check_missing_cache_points_to_model_fetch() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let cache_str = cache.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "check", "--cache-dir", cache_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("First run"))
        .stderr(predicate::str::contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0"))
        .stderr(predicate::str::contains("bitnet mac models --cache-dir"))
        .stderr(predicate::str::contains("Disk guidance:"));
    Ok(())
}

#[test]
fn mac_check_corrupt_cache_points_to_prune_and_fetch() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let model_dir = cache.join("qwen2.5-0.5b-instruct-q8_0");
    std::fs::create_dir_all(&model_dir)?;
    std::fs::write(model_dir.join("qwen2.5-0.5b-instruct-q8_0.gguf"), b"partial")?;
    let cache_str = cache.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "check", "--cache-dir", cache_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Cache repair"))
        .stderr(predicate::str::contains("bitnet model prune qwen2.5-0.5b-instruct-q8_0"))
        .stderr(predicate::str::contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0"));
    Ok(())
}

#[test]
fn mac_check_rejects_blocked_bitnet_model_before_cache_guidance() {
    bitnet()
        .args(["mac", "check", "--model-id", "microsoft-bitnet-b1.58-2B-4T-i2s"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("supported-ask for Apple M4 CPU/NEON local answers"))
        .stderr(predicate::str::contains("MODEL-ARTIFACT-007"))
        .stderr(predicate::str::contains("M4-QA-001"))
        .stderr(predicate::str::contains("one-shot `bitnet mac ask`"))
        .stderr(predicate::str::contains("fixed-prompt `bitnet mac bitnet-warm`"))
        .stderr(predicate::str::contains("bitnet mac models"))
        .stderr(predicate::str::contains("bitnet model fetch microsoft-bitnet").not());
}

#[test]
fn model_fetch_offline_missing_cache_explains_repair_options()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let cache_str = cache.to_string_lossy().into_owned();

    bitnet()
        .args([
            "model",
            "fetch",
            "qwen2.5-0.5b-instruct-q8_0",
            "--offline",
            "--cache-dir",
            cache_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("offline mode"))
        .stderr(predicate::str::contains("pre-seed"))
        .stderr(predicate::str::contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0"));
    Ok(())
}

#[test]
fn model_verify_corrupt_cache_explains_prune_and_fetch() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let model_dir = cache.join("qwen2.5-0.5b-instruct-q8_0");
    std::fs::create_dir_all(&model_dir)?;
    std::fs::write(model_dir.join("qwen2.5-0.5b-instruct-q8_0.gguf"), b"partial")?;
    let cache_str = cache.to_string_lossy().into_owned();

    bitnet()
        .args(["model", "verify", "qwen2.5-0.5b-instruct-q8_0", "--cache-dir", cache_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("repair:"))
        .stderr(predicate::str::contains("expected bytes=675710816"))
        .stderr(predicate::str::contains("got bytes=7"))
        .stderr(predicate::str::contains("bitnet model prune qwen2.5-0.5b-instruct-q8_0"))
        .stderr(predicate::str::contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0"));
    Ok(())
}

#[test]
fn model_verify_json_includes_dense_m4_artifact_provenance()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let cache_str = cache.to_string_lossy().into_owned();

    bitnet()
        .args([
            "model",
            "verify",
            "qwen2.5-0.5b-instruct-q8_0",
            "--cache-dir",
            cache_str.as_str(),
            "--json",
        ])
        .assert()
        .failure()
        .stdout(predicate::str::contains("\"artifact_kind\": \"m4_supported_model_provenance\""))
        .stdout(predicate::str::contains("\"spdx\": \"apache-2.0\""))
        .stdout(predicate::str::contains("\"tokenizer\""))
        .stdout(predicate::str::contains(
            "\"sha256_status\": \"embedded_gguf_metadata_bound_to_model_sha256\"",
        ))
        .stdout(predicate::str::contains("\"identity\": \"qwen2.5\""))
        .stdout(predicate::str::contains("\"local_cache\""))
        .stdout(predicate::str::contains("\"symlink_status\": \"not_symlink\""))
        .stdout(predicate::str::contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0"))
        .stdout(predicate::str::contains("runtime quality and performance require separate eval"));
    Ok(())
}

#[test]
fn model_verify_json_includes_bitnet_external_tokenizer_provenance()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let cache_str = cache.to_string_lossy().into_owned();

    bitnet()
        .args([
            "model",
            "verify",
            "microsoft-bitnet-b1.58-2B-4T-i2s",
            "--cache-dir",
            cache_str.as_str(),
            "--json",
        ])
        .assert()
        .failure()
        .stdout(predicate::str::contains("\"artifact_kind\": \"m4_supported_model_provenance\""))
        .stdout(predicate::str::contains(
            "\"sha256_status\": \"external_tokenizer_json_sha256_recorded\"",
        ))
        .stdout(predicate::str::contains(
            "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7",
        ))
        .stdout(predicate::str::contains("\"identity\": \"bitnetcpp-answer\""))
        .stdout(predicate::str::contains("Redistribution boundary recorded"))
        .stdout(predicate::str::contains("does not prove BitNet chat"));
    Ok(())
}

#[test]
fn mac_ask_help_documents_positional_question() {
    bitnet()
        .args(["mac", "ask", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("[QUESTION]"))
        .stdout(predicate::str::contains("--question <QUESTION>"))
        .stdout(predicate::str::contains("--timeout-seconds <SECONDS>"))
        .stdout(predicate::str::contains("--progress"))
        .stdout(predicate::str::contains("--quiet"));
}

#[test]
fn mac_ask_accepts_positional_question_and_progress_flags_before_cache_lookup()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let cache_str = cache.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "ask",
            "What is 2+2? Answer briefly.",
            "--progress",
            "--quiet",
            "--cache-dir",
            cache_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0"))
        .stderr(predicate::str::contains("bitnet mac models --cache-dir"))
        .stderr(predicate::str::contains("Disk guidance:"))
        .stderr(predicate::str::contains("unexpected argument").not());
    Ok(())
}

#[test]
fn mac_ask_rejects_positional_and_flag_question_together() {
    bitnet()
        .args(["mac", "ask", "What is 2+2?", "--question", "Name the capital of France."])
        .assert()
        .failure()
        .stderr(predicate::str::contains("either positionally"))
        .stderr(predicate::str::contains("not both"));
}

#[test]
#[cfg(debug_assertions)]
fn mac_validate_performance_requires_release_before_cache_lookup() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = dir.path().join("models");
    let cache_str = cache.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "validate",
            "--profile-set",
            "performance",
            "--cache-dir",
            cache_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must be run from a release build"));
}

#[test]
fn mac_ask_rejects_full_metal_request_before_cache_lookup() {
    bitnet()
        .args([
            "--device",
            "apple-m4-metal",
            "mac",
            "ask",
            "--question",
            "What is 2+2? Answer briefly.",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac ask routes the supported Mac local-answer path"))
        .stderr(predicate::str::contains("Full apple-m4-metal inference"));
}

#[test]
fn mac_ask_help_documents_trace_correlation() {
    bitnet()
        .args(["mac", "ask", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--trace"))
        .stdout(predicate::str::contains("trace-correlation"));
}

#[test]
fn mac_context_dense_ask_blocks_beyond_recorded_4k_before_cache_lookup()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let receipt = dir.path().join("dense-context-guardrail.json");
    let prompt = "dense context guardrail ".repeat(900);
    let cache_str = cache.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "ask",
            prompt.as_str(),
            "--cache-dir",
            cache_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac ask context guardrail blocked request"))
        .stderr(predicate::str::contains("unsupported_context_exceeds_recorded_evidence"))
        .stderr(predicate::str::contains("receipt written"))
        .stderr(predicate::str::contains("bitnet model fetch").not());

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "apple_m4_context_guardrail");
    assert_eq!(receipt_json["operator_command"], "mac ask");
    assert_eq!(receipt_json["status"], "blocked");
    assert_eq!(receipt_json["fallback_used"], false);
    assert_eq!(receipt_json["model_family"], "dense-slm");
    assert_eq!(receipt_json["context_envelope"]["work_item"], "M4-CONTEXT-001");
    assert_eq!(receipt_json["context_envelope"]["route"], "mac ask");
    assert_eq!(receipt_json["context_envelope"]["allowed"], false);
    assert_eq!(receipt_json["context_envelope"]["operator_class"], "unsupported");
    assert_eq!(
        receipt_json["context_envelope"]["status"],
        "unsupported_context_exceeds_recorded_evidence"
    );
    assert_eq!(
        receipt_json["context_envelope"]["recorded_envelope"]["evidence_profile"],
        "beyond_context_4k"
    );
    assert_eq!(receipt_json["claim_boundary"]["live_generation_executed"], false);
    assert_eq!(receipt_json["claim_boundary"]["unsupported_context_supported"], false);

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_context_guardrail"))
        .stdout(predicate::str::contains("\"prompt_count\": 0"))
        .stdout(predicate::str::contains("\"generated_tokens\": 0"));
    Ok(())
}

#[test]
fn mac_context_bitnet_ask_blocks_beyond_recorded_prompt_before_tokenizer_lookup()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt = dir.path().join("bitnet-context-guardrail.json");
    let tokenizer = dir.path().join("missing-tokenizer.json");
    let prompt = "bitnet context guardrail ".repeat(120);
    let receipt_str = receipt.to_string_lossy().into_owned();
    let tokenizer_str = tokenizer.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "ask",
            prompt.as_str(),
            "--model-id",
            "microsoft-bitnet-b1.58-2B-4T-i2s",
            "--model-path",
            "missing-bitnet.gguf",
            "--tokenizer",
            tokenizer_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac ask context guardrail blocked request"))
        .stderr(predicate::str::contains("unsupported_context_exceeds_recorded_evidence"))
        .stderr(predicate::str::contains("receipt written"))
        .stderr(predicate::str::contains("tokenizer is missing").not())
        .stderr(predicate::str::contains("requires explicit tokenizer authority").not());

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "apple_m4_context_guardrail");
    assert_eq!(receipt_json["operator_command"], "mac ask");
    assert_eq!(receipt_json["model_family"], "bitnet");
    assert_eq!(receipt_json["model_id"], "microsoft-bitnet-b1.58-2B-4T-i2s");
    assert_eq!(receipt_json["context_envelope"]["work_item"], "M4-CONTEXT-001");
    assert_eq!(receipt_json["context_envelope"]["route"], "mac ask");
    assert_eq!(receipt_json["context_envelope"]["allowed"], false);
    assert_eq!(receipt_json["context_envelope"]["operator_class"], "unsupported");
    assert_eq!(
        receipt_json["context_envelope"]["recorded_envelope"]["evidence_profile"],
        "beyond_bitnet_bounded_ask_warm"
    );
    assert_eq!(receipt_json["claim_boundary"]["bitnet_chat_enabled"], false);
    assert_eq!(receipt_json["claim_boundary"]["bitnet_serve_enabled"], false);

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_context_guardrail"))
        .stdout(predicate::str::contains("\"prompt_count\": 0"))
        .stdout(predicate::str::contains("\"generated_tokens\": 0"));
    Ok(())
}

#[test]
fn mac_context_dense_chat_blocks_beyond_recorded_4k_before_cache_lookup()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let receipt = dir.path().join("dense-chat-context-guardrail.json");
    let prompt = "dense chat context guardrail ".repeat(700);
    let cache_str = cache.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "chat",
            "--prompt",
            prompt.as_str(),
            "--prompt",
            "second bounded turn",
            "--cache-dir",
            cache_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac chat context guardrail blocked request"))
        .stderr(predicate::str::contains("unsupported_context_exceeds_recorded_evidence"))
        .stderr(predicate::str::contains("receipt written"))
        .stderr(predicate::str::contains("mac chat requires at least two prompts").not())
        .stderr(predicate::str::contains("bitnet model fetch").not());

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "apple_m4_context_guardrail");
    assert_eq!(receipt_json["operator_command"], "mac chat");
    assert_eq!(receipt_json["model_family"], "dense-slm");
    assert_eq!(receipt_json["context_envelope"]["route"], "mac chat");
    assert_eq!(receipt_json["context_envelope"]["allowed"], false);

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_context_guardrail"));
    Ok(())
}

#[test]
fn mac_context_dense_chat_smoke_blocks_large_system_before_cache_lookup()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let receipt = dir.path().join("chat-smoke-context-guardrail.json");
    let system_prompt = "dense chat smoke system context guardrail ".repeat(600);
    let cache_str = cache.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "chat-smoke",
            "--system",
            system_prompt.as_str(),
            "--cache-dir",
            cache_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac chat-smoke context guardrail blocked request"))
        .stderr(predicate::str::contains("unsupported_context_exceeds_recorded_evidence"))
        .stderr(predicate::str::contains("receipt written"))
        .stderr(predicate::str::contains("bitnet model fetch").not());

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "apple_m4_context_guardrail");
    assert_eq!(receipt_json["operator_command"], "mac chat-smoke");
    assert_eq!(receipt_json["model_family"], "dense-slm");
    assert_eq!(receipt_json["context_envelope"]["route"], "mac chat-smoke");
    assert_eq!(receipt_json["context_envelope"]["allowed"], false);

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_context_guardrail"));
    Ok(())
}

#[test]
fn mac_context_bitnet_warm_blocks_beyond_recorded_prompt_before_tokenizer_lookup()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt = dir.path().join("bitnet-warm-context-guardrail.json");
    let tokenizer = dir.path().join("missing-tokenizer.json");
    let prompt = "bitnet warm context guardrail ".repeat(120);
    let receipt_str = receipt.to_string_lossy().into_owned();
    let tokenizer_str = tokenizer.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "bitnet-warm",
            "--prompt",
            prompt.as_str(),
            "--prompt",
            prompt.as_str(),
            "--model-path",
            "missing-bitnet.gguf",
            "--tokenizer",
            tokenizer_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac bitnet-warm context guardrail blocked request"))
        .stderr(predicate::str::contains("unsupported_context_exceeds_recorded_evidence"))
        .stderr(predicate::str::contains("receipt written"))
        .stderr(predicate::str::contains("tokenizer is missing").not());

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "apple_m4_context_guardrail");
    assert_eq!(receipt_json["operator_command"], "mac bitnet-warm");
    assert_eq!(receipt_json["model_family"], "bitnet");
    assert_eq!(receipt_json["context_envelope"]["route"], "mac bitnet-warm");
    assert_eq!(receipt_json["context_envelope"]["allowed"], false);

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_context_guardrail"));
    Ok(())
}

#[test]
fn mac_bitnet_ask_timeout_seconds_is_bitnet_only_before_cache_lookup() {
    bitnet()
        .args(["mac", "ask", "What is 2+2?", "--timeout-seconds", "10"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("explicit BitNet one-shot route"))
        .stderr(predicate::str::contains("dense SLM ask does not use this timeout flag yet"));
}

#[test]
fn mac_bitnet_ask_requires_explicit_tokenizer_before_cache_lookup() {
    let dir = tempfile::tempdir().expect("tempdir");
    let receipt = dir.path().join("bitnet-ask-missing-tokenizer-authority.json");
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "ask",
            "What is 2+2?",
            "--model-id",
            "microsoft-bitnet-b1.58-2B-4T-i2s",
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("requires explicit tokenizer authority"))
        .stderr(predicate::str::contains("Repair guidance:"))
        .stderr(predicate::str::contains("does not infer tokenizer authority"))
        .stderr(predicate::str::contains(
            "--tokenizer models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json",
        ))
        .stderr(predicate::str::contains("bitnet model fetch microsoft-bitnet").not());
}

#[test]
fn mac_bitnet_ask_writes_failure_receipt_for_missing_tokenizer()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt = dir.path().join("bitnet-ask-failure.json");
    let tokenizer = dir.path().join("missing-tokenizer.json");
    let receipt_str = receipt.to_string_lossy().into_owned();
    let tokenizer_str = tokenizer.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "ask",
            "What is 2+2?",
            "--model-id",
            "microsoft-bitnet-b1.58-2B-4T-i2s",
            "--model-path",
            "missing-bitnet.gguf",
            "--tokenizer",
            tokenizer_str.as_str(),
            "--timeout-seconds",
            "60",
            "--progress",
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("tokenizer is missing"))
        .stderr(predicate::str::contains("failure receipt written"))
        .stderr(predicate::str::contains("Repair guidance:"))
        .stderr(predicate::str::contains("shasum -a 256"))
        .stderr(predicate::str::contains("mac ask progress: tokenizer_verify_start"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "bitnet_apple_m4_mac_ask_failure");
    assert_eq!(receipt_json["status"], "failed");
    assert_eq!(receipt_json["failure"]["stage"], "tokenizer_missing");
    assert_eq!(receipt_json["fallback_used"], false);
    assert_eq!(receipt_json["generation"]["generated_tokens"], 0);
    assert_eq!(receipt_json["generation"]["partial_generation_available"], false);
    assert_eq!(receipt_json["timeout_boundary"]["configured_seconds"], 60);
    assert_eq!(receipt_json["timeout_boundary"]["enforced"], true);
    assert_eq!(receipt_json["timeout_boundary"]["reached"], false);
    assert_eq!(receipt_json["timeout_boundary"]["stage"], "tokenizer_missing");
    assert_eq!(receipt_json["progress"]["enabled"], true);
    assert_eq!(receipt_json["progress"]["last_stage"], "tokenizer_missing");
    let stage_taxonomy =
        receipt_json["progress"]["stage_taxonomy"].as_array().ok_or("stage taxonomy missing")?;
    assert!(stage_taxonomy.iter().any(|stage| stage.as_str() == Some("decode")));
    assert!(stage_taxonomy.iter().any(|stage| stage.as_str() == Some("receipt_write")));
    assert_eq!(receipt_json["mac_bitnet_claim_boundary"]["chat_enabled"], false);
    assert_eq!(receipt_json["mac_bitnet_claim_boundary"]["serve_enabled"], false);

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("bitnet_apple_m4_mac_ask_failure"))
        .stdout(predicate::str::contains("\"generated_tokens\": 0"));
    Ok(())
}

#[test]
fn mac_bitnet_ask_rejects_wrong_tokenizer_sha_before_model_lookup()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let tokenizer = dir.path().join("tokenizer.json");
    let receipt = dir.path().join("bitnet-ask-tokenizer-failure.json");
    std::fs::write(&tokenizer, b"{\"not\":\"the accepted tokenizer\"}")?;
    let tokenizer_str = tokenizer.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "ask",
            "What is 2+2?",
            "--model-id",
            "microsoft-bitnet-b1.58-2B-4T-i2s",
            "--model-path",
            "missing-bitnet.gguf",
            "--tokenizer",
            tokenizer_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("requires tokenizer SHA256"))
        .stderr(predicate::str::contains("failure receipt written"))
        .stderr(predicate::str::contains("Repair guidance:"))
        .stderr(predicate::str::contains("accepted GGUF").not());

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "bitnet_apple_m4_mac_ask_failure");
    assert_eq!(receipt_json["failure"]["stage"], "tokenizer_verify_failed");
    assert_eq!(
        receipt_json["tokenizer"]["expected_sha256"],
        "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7"
    );
    Ok(())
}

#[test]
fn mac_ask_rejects_dense_model_path_tokenizer_overrides() {
    bitnet()
        .args([
            "mac",
            "ask",
            "What is 2+2?",
            "--model-path",
            "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
            "--tokenizer",
            "models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--model-path/--tokenizer only for the explicit BitNet one-shot route",
        ));
}

#[test]
fn mac_smoke_help_documents_golden_smoke() {
    bitnet()
        .args(["mac", "smoke", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("golden smoke"))
        .stdout(predicate::str::contains("--model-family"))
        .stdout(predicate::str::contains("bitnet"))
        .stdout(predicate::str::contains("--json-out <PATH>"))
        .stdout(predicate::str::contains("--max-new-tokens"));
}

#[test]
fn mac_smoke_missing_cache_points_to_model_fetch() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = dir.path().join("models");
    let cache_str = cache.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "smoke", "--cache-dir", cache_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("golden smoke cannot run"))
        .stderr(predicate::str::contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0"));
}

#[test]
fn mac_smoke_bitnet_missing_cache_writes_failure_receipt() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = dir.path().join("models");
    let json_out = dir.path().join("mac-smoke.json");
    let answer_receipt = dir.path().join("mac-smoke-answer.json");
    let cache_str = cache.to_string_lossy().into_owned();
    let json_out_str = json_out.to_string_lossy().into_owned();
    let tokenizer = dir.path().join("missing-tokenizer.json");
    let tokenizer_str = tokenizer.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "smoke",
            "--model-family",
            "bitnet",
            "--cache-dir",
            cache_str.as_str(),
            "--json-out",
            json_out_str.as_str(),
            "--tokenizer",
            tokenizer_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("failure receipt written"))
        .stderr(predicate::str::contains("tokenizer is missing"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(answer_receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "bitnet_apple_m4_mac_ask_failure");
    assert_eq!(receipt_json["failure"]["stage"], "tokenizer_missing");
    assert_eq!(receipt_json["mac_bitnet_claim_boundary"]["chat_enabled"], false);
    assert_eq!(receipt_json["mac_bitnet_claim_boundary"]["serve_enabled"], false);
    Ok(())
}

#[test]
fn mac_smoke_dense_rejects_bitnet_explicit_artifact_args() {
    bitnet()
        .args([
            "mac",
            "smoke",
            "--model-path",
            "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
            "--tokenizer",
            "models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--model-family bitnet"))
        .stderr(predicate::str::contains("dense SLM smoke uses --model-id"));
}

#[test]
fn mac_smoke_rejects_full_metal_request_before_cache_lookup() {
    bitnet()
        .args(["--device", "apple-m4-metal", "mac", "smoke"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac smoke routes the supported Mac local-answer path"))
        .stderr(predicate::str::contains("Full apple-m4-metal inference"));
}

#[test]
fn mac_smoke_rejects_diagnostic_only_model_before_cache_lookup() {
    bitnet()
        .args(["mac", "smoke", "--model-id", "qwen3-0.6b-q8_0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("diagnostic-only"))
        .stderr(predicate::str::contains("not selectable"))
        .stderr(predicate::str::contains("bitnet mac models"))
        .stderr(predicate::str::contains("bitnet model fetch").not());
}

#[test]
fn mac_bitnet_warm_help_documents_fixed_resident_proof() {
    bitnet()
        .args(["mac", "bitnet-warm", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("BitNet prompts"))
        .stdout(predicate::str::contains("--model-id"))
        .stdout(predicate::str::contains("--model-path"))
        .stdout(predicate::str::contains("--tokenizer"))
        .stdout(predicate::str::contains("--prompt <TEXT>"))
        .stdout(predicate::str::contains("--profile <PROFILE>"))
        .stdout(predicate::str::contains("resident_25"))
        .stdout(predicate::str::contains("--timeout-seconds <SECONDS>"))
        .stdout(predicate::str::contains("--progress"))
        .stdout(predicate::str::contains("--json-out"));
}

#[test]
fn mac_bitnet_chat_requires_ready_gate_before_prompt_collection() {
    bitnet()
        .args(["mac", "chat", "--model-family", "bitnet"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "BitNet Mac chat requires a ready M4-BITNET-EX-006 gate receipt",
        ))
        .stderr(predicate::str::contains("bitnet mac bitnet-chat-gate"))
        .stderr(predicate::str::contains("--bitnet-chat-gate-receipt"))
        .stderr(predicate::str::contains("mac chat requires at least two prompts").not());
}

#[test]
fn mac_bitnet_chat_gate_writes_blocked_receipt_without_required_evidence()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let warm_receipt = dir.path().join("missing-warm-session.json");
    let gate_receipt = dir.path().join("bitnet-chat-gate.json");
    let warm_receipt_str = warm_receipt.to_string_lossy().into_owned();
    let gate_receipt_str = gate_receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "bitnet-chat-gate",
            "--warm-receipt",
            warm_receipt_str.as_str(),
            "--json-out",
            gate_receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("BitNet chat gate is blocked"))
        .stderr(predicate::str::contains("receipt written"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&gate_receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "bitnet_apple_m4_chat_gate");
    assert_eq!(receipt_json["model_id"], "microsoft-bitnet-b1.58-2B-4T-i2s");
    assert_eq!(receipt_json["status"], "blocked");
    assert_eq!(receipt_json["chat_enablement"]["gate_passed"], false);
    assert_eq!(receipt_json["chat_enablement"]["chat_enabled"], false);
    assert_eq!(receipt_json["chat_enablement"]["serve_enabled"], false);
    assert_eq!(receipt_json["requirements"]["variable_warm_session_receipt"]["passed"], false);
    assert_eq!(
        receipt_json["requirements"]["variable_warm_session_receipt"]["repeated_prompt_determinism_passed"],
        false
    );
    assert_eq!(receipt_json["requirements"]["timeout_failure_receipt"]["passed"], false);
    assert_eq!(
        receipt_json["requirements"]["timeout_failure_receipt"]["timeout_boundary_recorded"],
        false
    );
    assert_eq!(receipt_json["requirements"]["streaming_semantics_receipt"]["passed"], false);
    assert_eq!(receipt_json["mac_bitnet_claim_boundary"]["chat_enabled"], false);
    assert_eq!(receipt_json["mac_bitnet_claim_boundary"]["serve_enabled"], false);

    bitnet()
        .args(["mac", "receipts-check", gate_receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("bitnet_apple_m4_chat_gate"))
        .stdout(predicate::str::contains("\"generated_tokens\": 0"));
    Ok(())
}

#[test]
fn mac_receipts_check_rejects_bitnet_chat_gate_unsupported_claim()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let warm_receipt = dir.path().join("missing-warm-session.json");
    let gate_receipt = dir.path().join("bitnet-chat-gate.json");
    let warm_receipt_str = warm_receipt.to_string_lossy().into_owned();
    let gate_receipt_str = gate_receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "bitnet-chat-gate",
            "--warm-receipt",
            warm_receipt_str.as_str(),
            "--json-out",
            gate_receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("receipt written"));

    let mut receipt_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&gate_receipt)?)?;
    receipt_json["mac_bitnet_claim_boundary"]["chat_enabled"] = serde_json::json!(true);
    std::fs::write(&gate_receipt, serde_json::to_vec_pretty(&receipt_json)?)?;

    bitnet()
        .args(["mac", "receipts-check", gate_receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("chat_enabled"));
    Ok(())
}

#[test]
fn mac_bitnet_chat_gate_writes_ready_receipt_with_required_evidence()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let failure_receipt = dir.path().join("bitnet-warm-failure.json");
    let streaming_receipt = dir.path().join("bitnet-chat-streaming.json");
    let gate_receipt = dir.path().join("bitnet-chat-gate.json");
    let missing_tokenizer = dir.path().join("missing-tokenizer.json");
    let warm_receipt = workspace_path(
        "ci/hardware/apple-m4-mac-mini/2026-05-17T0847Z/bitnet-warm/variable-warm-session.json",
    );
    let failure_receipt_str = failure_receipt.to_string_lossy().into_owned();
    let streaming_receipt_str = streaming_receipt.to_string_lossy().into_owned();
    let gate_receipt_str = gate_receipt.to_string_lossy().into_owned();
    let missing_tokenizer_str = missing_tokenizer.to_string_lossy().into_owned();
    let warm_receipt_str = warm_receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "bitnet-warm",
            "--model-path",
            "missing-bitnet.gguf",
            "--tokenizer",
            missing_tokenizer_str.as_str(),
            "--prompt",
            "Answer with a single digit: 2+2=",
            "--prompt",
            "Answer with a single digit: 2+2=",
            "--json-out",
            failure_receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("failure receipt written"));
    write_bitnet_chat_streaming_semantics_receipt(&streaming_receipt)?;

    bitnet()
        .args([
            "mac",
            "bitnet-chat-gate",
            "--model-id",
            "microsoft-bitnet-b1.58-2B-4T-i2s",
            "--warm-receipt",
            warm_receipt_str.as_str(),
            "--failure-receipt",
            failure_receipt_str.as_str(),
            "--streaming-receipt",
            streaming_receipt_str.as_str(),
            "--json-out",
            gate_receipt_str.as_str(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("ready-to-enable"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&gate_receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "bitnet_apple_m4_chat_gate");
    assert_eq!(receipt_json["status"], "ready_to_enable");
    assert_eq!(receipt_json["chat_enablement"]["gate_passed"], true);
    assert_eq!(receipt_json["chat_enablement"]["chat_enabled"], false);
    assert_eq!(receipt_json["chat_enablement"]["serve_enabled"], false);
    assert_eq!(receipt_json["requirements"]["variable_warm_session_receipt"]["passed"], true);
    assert_eq!(receipt_json["requirements"]["timeout_failure_receipt"]["passed"], true);
    assert_eq!(receipt_json["requirements"]["streaming_semantics_receipt"]["passed"], true);
    assert_eq!(receipt_json["mac_bitnet_claim_boundary"]["chat_enabled"], false);
    assert_eq!(receipt_json["mac_bitnet_claim_boundary"]["serve_enabled"], false);

    bitnet()
        .args(["mac", "receipts-check", gate_receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("bitnet_apple_m4_chat_gate"));
    Ok(())
}

#[test]
fn mac_bitnet_chat_ready_gate_reaches_model_verification() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = tempfile::tempdir()?;
    let failure_receipt = dir.path().join("bitnet-warm-failure.json");
    let streaming_receipt = dir.path().join("bitnet-chat-streaming.json");
    let gate_receipt = dir.path().join("bitnet-chat-gate.json");
    let missing_tokenizer = dir.path().join("missing-tokenizer.json");
    let warm_receipt = workspace_path(
        "ci/hardware/apple-m4-mac-mini/2026-05-17T0847Z/bitnet-warm/variable-warm-session.json",
    );
    let failure_receipt_str = failure_receipt.to_string_lossy().into_owned();
    let streaming_receipt_str = streaming_receipt.to_string_lossy().into_owned();
    let gate_receipt_str = gate_receipt.to_string_lossy().into_owned();
    let missing_tokenizer_str = missing_tokenizer.to_string_lossy().into_owned();
    let warm_receipt_str = warm_receipt.to_string_lossy().into_owned();
    let tokenizer = workspace_path("models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json");
    let tokenizer_str = tokenizer.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "bitnet-warm",
            "--model-path",
            "missing-bitnet.gguf",
            "--tokenizer",
            missing_tokenizer_str.as_str(),
            "--prompt",
            "Answer with a single digit: 2+2=",
            "--prompt",
            "Answer with a single digit: 2+2=",
            "--json-out",
            failure_receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("failure receipt written"));
    write_bitnet_chat_streaming_semantics_receipt(&streaming_receipt)?;
    bitnet()
        .args([
            "mac",
            "bitnet-chat-gate",
            "--warm-receipt",
            warm_receipt_str.as_str(),
            "--failure-receipt",
            failure_receipt_str.as_str(),
            "--streaming-receipt",
            streaming_receipt_str.as_str(),
            "--json-out",
            gate_receipt_str.as_str(),
        ])
        .assert()
        .success();

    bitnet()
        .args([
            "mac",
            "chat",
            "--model-family",
            "bitnet",
            "--model-id",
            "microsoft-bitnet-b1.58-2B-4T-i2s",
            "--model-path",
            "missing-bitnet.gguf",
            "--tokenizer",
            tokenizer_str.as_str(),
            "--bitnet-chat-gate-receipt",
            gate_receipt_str.as_str(),
            "--prompt",
            "Answer with a single digit: 2+2=",
            "--prompt",
            "Name the capital of France. Answer with one word.",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("accepted GGUF"))
        .stderr(predicate::str::contains("ready M4-BITNET-EX-006 gate receipt").not())
        .stderr(predicate::str::contains("mac chat requires at least two prompts").not());
    Ok(())
}

#[test]
fn mac_bitnet_chat_session_receipt_validates_claim_boundary()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let source = workspace_path(
        "ci/hardware/apple-m4-mac-mini/2026-05-17T0847Z/bitnet-warm/variable-warm-session.json",
    );
    let receipt_path = dir.path().join("bitnet-chat-session.json");
    let receipt_path_str = receipt_path.to_string_lossy().into_owned();
    let mut receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(source)?)?;
    let prompt_count = receipt["session"]["prompt_count"].clone();
    let object = receipt.as_object_mut().ok_or("receipt must be object")?;
    object.insert("artifact_kind".to_string(), serde_json::json!("bitnet_apple_m4_chat_session"));
    object.insert("operator_command".to_string(), serde_json::json!("mac chat"));
    object.insert("model_id".to_string(), serde_json::json!("microsoft-bitnet-b1.58-2B-4T-i2s"));
    object.insert(
        "bitnet_chat_gate".to_string(),
        serde_json::json!({
            "path": "ci/hardware/apple-m4-mac-mini/2026-05-17T0000Z/bitnet-chat-gate/gate.json",
            "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "status": "ready_to_enable",
            "gate_passed": true,
            "validated": true,
            "work_item": "M4-BITNET-EX-006"
        }),
    );
    object.insert(
        "bitnet_chat".to_string(),
        serde_json::json!({
            "enabled": true,
            "route": "bitnet mac chat --model-family bitnet",
            "prompt_count": prompt_count,
            "serve_enabled": false,
            "streaming_requested": true,
            "per_turn_receipts_enabled": true,
            "gate_required": true
        }),
    );
    object.insert(
        "mac_bitnet_claim_boundary".to_string(),
        serde_json::json!({
            "bitnet_chat_session": true,
            "answer_corpus_proof_gate": "MODEL-ARTIFACT-007/M4-QA-001",
            "chat_gate_work_item": "M4-BITNET-EX-006",
            "requested_backend": "apple-m4-cpu-neon",
            "tokenizer_path": "models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json",
            "tokenizer_sha256": "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7",
            "chat_enabled": true,
            "serve_enabled": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false
        }),
    );
    if let Some(claim_boundary) =
        object.get_mut("claim_boundary").and_then(|value| value.as_object_mut())
    {
        claim_boundary.insert("bitnet_chat_session".to_string(), serde_json::json!(true));
        claim_boundary.insert("chat_enabled".to_string(), serde_json::json!(true));
        claim_boundary.insert("serve_enabled".to_string(), serde_json::json!(false));
        claim_boundary.insert("qk256_apple_claimed".to_string(), serde_json::json!(false));
        claim_boundary
            .insert("neural_engine_execution_claimed".to_string(), serde_json::json!(false));
        claim_boundary.insert("mpsgraph_inference_claimed".to_string(), serde_json::json!(false));
    }
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;

    bitnet()
        .args(["mac", "receipts-check", receipt_path_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("bitnet_apple_m4_chat_session"))
        .stdout(predicate::str::contains("\"prompt_count\": 100"));
    Ok(())
}

#[test]
fn mac_bitnet_serve_help_documents_ready_gate() {
    bitnet()
        .args(["mac", "serve", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--model-family <MODEL_FAMILY>"))
        .stdout(predicate::str::contains("--model-path <PATH>"))
        .stdout(predicate::str::contains("--tokenizer <PATH>"))
        .stdout(predicate::str::contains("--bitnet-serve-gate-receipt <PATH>"));
}

#[test]
fn mac_bitnet_serve_requires_ready_gate_before_cache_or_bind()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_dir = dir.path().join("receipts");
    let receipt_dir_str = receipt_dir.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "serve",
            "--model-family",
            "bitnet",
            "--receipt-dir",
            receipt_dir_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "BitNet Mac serve requires a ready M4-BITNET-EX-007 gate receipt",
        ))
        .stderr(predicate::str::contains("bitnet mac bitnet-serve-gate"))
        .stderr(predicate::str::contains("--bitnet-serve-gate-receipt"))
        .stderr(predicate::str::contains("bitnet model fetch").not())
        .stderr(predicate::str::contains("failed to bind").not());
    Ok(())
}

#[test]
fn mac_bitnet_serve_gate_writes_blocked_receipt_without_required_evidence()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let chat_receipt = dir.path().join("missing-chat-session.json");
    let gate_receipt = dir.path().join("bitnet-serve-gate.json");
    let chat_receipt_str = chat_receipt.to_string_lossy().into_owned();
    let gate_receipt_str = gate_receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "bitnet-serve-gate",
            "--chat-receipt",
            chat_receipt_str.as_str(),
            "--json-out",
            gate_receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("BitNet serve gate is blocked"))
        .stderr(predicate::str::contains("receipt written"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&gate_receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "bitnet_apple_m4_serve_gate");
    assert_eq!(receipt_json["model_id"], "microsoft-bitnet-b1.58-2B-4T-i2s");
    assert_eq!(receipt_json["status"], "blocked");
    assert_eq!(receipt_json["serve_enablement"]["gate_passed"], false);
    assert_eq!(receipt_json["serve_enablement"]["serve_enabled"], false);
    assert_eq!(receipt_json["requirements"]["chat_session_receipt"]["passed"], false);
    assert_eq!(receipt_json["requirements"]["streaming_semantics_receipt"]["passed"], false);
    assert_eq!(receipt_json["requirements"]["timeout_failure_receipt"]["passed"], false);
    assert_eq!(receipt_json["requirements"]["serve_check_receipt"]["passed"], false);
    assert_eq!(receipt_json["requirements"]["health_ready_endpoints"]["passed"], false);
    assert_eq!(receipt_json["requirements"]["per_request_receipt_export"]["passed"], false);
    assert_eq!(receipt_json["mac_bitnet_claim_boundary"]["serve_enabled"], false);
    assert_eq!(receipt_json["mac_bitnet_claim_boundary"]["production_hosting_claimed"], false);
    assert_eq!(receipt_json["mac_bitnet_claim_boundary"]["openai_compatibility_claimed"], false);

    bitnet()
        .args(["mac", "receipts-check", gate_receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("bitnet_apple_m4_serve_gate"))
        .stdout(predicate::str::contains("\"generated_tokens\": 0"));
    Ok(())
}

#[test]
fn mac_receipts_check_rejects_bitnet_serve_gate_missing_fallback_state()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let chat_receipt = dir.path().join("missing-chat-session.json");
    let gate_receipt = dir.path().join("bitnet-serve-gate.json");
    let chat_receipt_str = chat_receipt.to_string_lossy().into_owned();
    let gate_receipt_str = gate_receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "bitnet-serve-gate",
            "--chat-receipt",
            chat_receipt_str.as_str(),
            "--json-out",
            gate_receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("receipt written"));

    let mut receipt_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&gate_receipt)?)?;
    receipt_json.as_object_mut().ok_or("receipt must be an object")?.remove("fallback_used");
    std::fs::write(&gate_receipt, serde_json::to_vec_pretty(&receipt_json)?)?;

    bitnet()
        .args(["mac", "receipts-check", gate_receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("fallback_used=true"));
    Ok(())
}

#[test]
fn mac_bitnet_serve_gate_writes_ready_receipt_with_required_evidence()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let chat_receipt = dir.path().join("bitnet-chat-session.json");
    let streaming_receipt = dir.path().join("bitnet-serve-streaming.json");
    let failure_receipt = dir.path().join("bitnet-serve-failure.json");
    let serve_check_receipt = dir.path().join("bitnet-serve-check.json");
    let gate_receipt = dir.path().join("bitnet-serve-gate.json");
    let chat_receipt_str = chat_receipt.to_string_lossy().into_owned();
    let streaming_receipt_str = streaming_receipt.to_string_lossy().into_owned();
    let failure_receipt_str = failure_receipt.to_string_lossy().into_owned();
    let serve_check_receipt_str = serve_check_receipt.to_string_lossy().into_owned();
    let gate_receipt_str = gate_receipt.to_string_lossy().into_owned();
    write_bitnet_chat_session_receipt(&chat_receipt)?;
    write_bitnet_serve_streaming_semantics_receipt(&streaming_receipt)?;
    write_bitnet_serve_failure_receipt(&failure_receipt)?;
    write_bitnet_serve_check_receipt(&serve_check_receipt)?;

    bitnet()
        .args([
            "mac",
            "bitnet-serve-gate",
            "--model-id",
            "microsoft-bitnet-b1.58-2B-4T-i2s",
            "--chat-receipt",
            chat_receipt_str.as_str(),
            "--streaming-receipt",
            streaming_receipt_str.as_str(),
            "--failure-receipt",
            failure_receipt_str.as_str(),
            "--serve-check-receipt",
            serve_check_receipt_str.as_str(),
            "--json-out",
            gate_receipt_str.as_str(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("ready-to-enable"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&gate_receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "bitnet_apple_m4_serve_gate");
    assert_eq!(receipt_json["status"], "ready_to_enable");
    assert_eq!(receipt_json["serve_enablement"]["gate_passed"], true);
    assert_eq!(receipt_json["serve_enablement"]["serve_enabled"], false);
    assert_eq!(receipt_json["requirements"]["chat_session_receipt"]["passed"], true);
    assert_eq!(receipt_json["requirements"]["chat_session_receipt"]["chat_enabled"], true);
    assert_eq!(receipt_json["requirements"]["chat_session_receipt"]["serve_enabled"], false);
    assert_eq!(receipt_json["requirements"]["streaming_semantics_receipt"]["passed"], true);
    assert_eq!(receipt_json["requirements"]["timeout_failure_receipt"]["passed"], true);
    assert_eq!(receipt_json["requirements"]["serve_check_receipt"]["passed"], true);
    assert_eq!(receipt_json["requirements"]["health_ready_endpoints"]["passed"], true);
    assert_eq!(receipt_json["requirements"]["per_request_receipt_export"]["passed"], true);
    assert_eq!(
        receipt_json["requirements"]["timeout_failure_receipt"]["timeout_boundary_recorded"],
        true
    );
    assert_eq!(receipt_json["mac_bitnet_claim_boundary"]["serve_enabled"], false);

    bitnet()
        .args(["mac", "receipts-check", gate_receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("bitnet_apple_m4_serve_gate"));
    Ok(())
}

#[test]
fn mac_bitnet_serve_gate_blocks_invalid_failure_evidence() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = tempfile::tempdir()?;
    let chat_receipt = dir.path().join("bitnet-chat-session.json");
    let streaming_receipt = dir.path().join("bitnet-serve-streaming.json");
    let failure_receipt = dir.path().join("bitnet-serve-failure.json");
    let serve_check_receipt = dir.path().join("bitnet-serve-check.json");
    let gate_receipt = dir.path().join("bitnet-serve-gate.json");
    let chat_receipt_str = chat_receipt.to_string_lossy().into_owned();
    let streaming_receipt_str = streaming_receipt.to_string_lossy().into_owned();
    let failure_receipt_str = failure_receipt.to_string_lossy().into_owned();
    let serve_check_receipt_str = serve_check_receipt.to_string_lossy().into_owned();
    let gate_receipt_str = gate_receipt.to_string_lossy().into_owned();
    write_bitnet_chat_session_receipt(&chat_receipt)?;
    write_bitnet_serve_streaming_semantics_receipt(&streaming_receipt)?;
    write_bitnet_serve_failure_receipt(&failure_receipt)?;
    write_bitnet_serve_check_receipt(&serve_check_receipt)?;
    let mut failure_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&failure_receipt)?)?;
    failure_json["timeout_boundary"]["enforced"] = serde_json::json!(false);
    std::fs::write(&failure_receipt, serde_json::to_vec_pretty(&failure_json)?)?;

    bitnet()
        .args([
            "mac",
            "bitnet-serve-gate",
            "--chat-receipt",
            chat_receipt_str.as_str(),
            "--streaming-receipt",
            streaming_receipt_str.as_str(),
            "--failure-receipt",
            failure_receipt_str.as_str(),
            "--serve-check-receipt",
            serve_check_receipt_str.as_str(),
            "--json-out",
            gate_receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("BitNet serve gate is blocked"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&gate_receipt)?)?;
    assert_eq!(receipt_json["status"], "blocked");
    assert_eq!(receipt_json["requirements"]["timeout_failure_receipt"]["passed"], false);
    assert!(
        receipt_json["requirements"]["timeout_failure_receipt"]["error"]
            .as_str()
            .unwrap_or_default()
            .contains("timeout_boundary.enforced")
    );
    Ok(())
}

#[test]
fn mac_receipts_check_rejects_bitnet_serve_failure_missing_timeout_boundary()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("bitnet-serve-failure.json");
    let receipt_str = receipt_path.to_string_lossy().into_owned();
    write_bitnet_serve_failure_receipt(&receipt_path)?;

    let mut receipt_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&receipt_path)?)?;
    receipt_json["timeout_boundary"]
        .as_object_mut()
        .ok_or_else(|| std::io::Error::other("timeout_boundary"))?
        .remove("reached");
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt_json)?)?;

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("timeout_boundary.reached"));
    Ok(())
}

#[test]
fn mac_bitnet_serve_ready_gate_reaches_model_verification() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = tempfile::tempdir()?;
    let chat_receipt = dir.path().join("bitnet-chat-session.json");
    let streaming_receipt = dir.path().join("bitnet-serve-streaming.json");
    let failure_receipt = dir.path().join("bitnet-serve-failure.json");
    let serve_check_receipt = dir.path().join("bitnet-serve-check.json");
    let gate_receipt = dir.path().join("bitnet-serve-gate.json");
    let receipt_dir = dir.path().join("serve-receipts");
    let chat_receipt_str = chat_receipt.to_string_lossy().into_owned();
    let streaming_receipt_str = streaming_receipt.to_string_lossy().into_owned();
    let failure_receipt_str = failure_receipt.to_string_lossy().into_owned();
    let serve_check_receipt_str = serve_check_receipt.to_string_lossy().into_owned();
    let gate_receipt_str = gate_receipt.to_string_lossy().into_owned();
    let receipt_dir_str = receipt_dir.to_string_lossy().into_owned();
    let tokenizer = workspace_path("models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json");
    let tokenizer_str = tokenizer.to_string_lossy().into_owned();
    write_bitnet_chat_session_receipt(&chat_receipt)?;
    write_bitnet_serve_streaming_semantics_receipt(&streaming_receipt)?;
    write_bitnet_serve_failure_receipt(&failure_receipt)?;
    write_bitnet_serve_check_receipt(&serve_check_receipt)?;
    bitnet()
        .args([
            "mac",
            "bitnet-serve-gate",
            "--chat-receipt",
            chat_receipt_str.as_str(),
            "--streaming-receipt",
            streaming_receipt_str.as_str(),
            "--failure-receipt",
            failure_receipt_str.as_str(),
            "--serve-check-receipt",
            serve_check_receipt_str.as_str(),
            "--json-out",
            gate_receipt_str.as_str(),
        ])
        .assert()
        .success();

    bitnet()
        .args([
            "mac",
            "serve",
            "--model-family",
            "bitnet",
            "--model-path",
            "missing-bitnet.gguf",
            "--tokenizer",
            tokenizer_str.as_str(),
            "--bitnet-serve-gate-receipt",
            gate_receipt_str.as_str(),
            "--receipt-dir",
            receipt_dir_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("accepted GGUF"))
        .stderr(predicate::str::contains("ready M4-BITNET-EX-007 gate receipt").not())
        .stderr(predicate::str::contains("failed to bind").not());
    Ok(())
}

#[test]
fn mac_bitnet_serve_completion_receipt_validates_claim_boundary()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("bitnet-serve-completion.json");
    let receipt_path_str = receipt_path.to_string_lossy().into_owned();
    let receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_apple_m4_serve_completion",
        "timestamp": "2026-05-17T00:00:00Z",
        "request_id": "bitnet-serve-test",
        "artifact_path": receipt_path_str,
        "requested_backend": "apple-m4-cpu-neon",
        "selected_backend": "apple-m4-cpu-neon",
        "runtime_api": "cpu",
        "fallback_used": false,
        "fallback_reason": null,
        "server": {
            "host": "127.0.0.1",
            "port": 8080,
            "started_at": "2026-05-17T00:00:00Z",
            "streaming_default": true,
            "receipt_dir": dir.path().join("receipts"),
            "model_family": "bitnet"
        },
        "model_family": "bitnet",
        "model": {
            "id": "microsoft-bitnet-b1.58-2B-4T-i2s",
            "display_name": "Microsoft BitNet b1.58 2B 4T I2_S",
            "path": "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
            "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
            "sha256_source": "verified_cache_metadata_and_startup_check",
            "bytes": 1187801280,
            "architecture": "bitnet_b1_58",
            "quantization": "I2_S"
        },
        "tokenizer": {
            "type": "hf-tokenizers-json",
            "source": "external_tokenizer_json",
            "strict": true,
            "pretokenizer_authority": "llama-bpe",
            "prompt_template": "bitnetcpp-answer",
            "bos": 1,
            "eos": 2
        },
        "request": {
            "model": "microsoft-bitnet-b1.58-2B-4T-i2s",
            "prompt": "Answer with one digit: 2+2=",
            "system_prompt": null,
            "stream": true,
            "max_new_tokens": 1,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "repetition_penalty": 1.1,
            "seed": null
        },
        "generation": {
            "mode": "greedy",
            "text": "4",
            "finish_reason": "length",
            "prompt_tokens": 8,
            "generated_tokens": 1,
            "prompt_token_ids": [1, 2, 3],
            "generated_token_ids": [19],
            "token_texts": ["4"]
        },
        "timing": {
            "model_load_ms": 0.0,
            "tokenizer_load_ms": 0.0,
            "tokenize_ms": 1.0,
            "prefill_ms": 1.0,
            "first_token_ms": 2,
            "time_to_first_token_ms": 2,
            "decode_ms": 1.0,
            "sampling_ms": 0.1,
            "total_ms": 3.0,
            "decode_step_ms": {"count": 1, "min": 1.0, "max": 1.0, "mean": 1.0},
            "sample_step_ms": {"count": 1, "min": 0.1, "max": 0.1, "mean": 0.1}
        },
        "session_reuse": {
            "reuse_scope": "resident_server",
            "model_loaded_at_startup": true,
            "tokenizer_loaded_at_startup": true,
            "request_serialized": true,
            "kv_cache_reuse_policy": "recreated_per_request_for_prompt_isolation"
        },
        "claim_boundary": {
            "local_server_completion_endpoint": true,
            "streaming_transport": true,
            "openai_compatibility_claimed": false,
            "production_readiness_claimed": false,
            "bitnet_quality_claimed": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false
        },
        "mac_bitnet_claim_boundary": {
            "bitnet_serve_session": true,
            "serve_gate_work_item": "M4-BITNET-EX-007",
            "serve_gate_path": "ci/hardware/apple-m4-mac-mini/2026-05-17T0000Z/bitnet-serve-gate/gate.json",
            "serve_gate_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "serve_enabled": true,
            "chat_required": true,
            "requested_backend": "apple-m4-cpu-neon",
            "tokenizer_sha256": "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7",
            "production_hosting_claimed": false,
            "openai_compatibility_claimed": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false
        }
    });
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;

    bitnet()
        .args(["mac", "receipts-check", receipt_path.to_string_lossy().as_ref(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("bitnet_apple_m4_serve_completion"))
        .stdout(predicate::str::contains("\"generated_tokens\": 1"));
    Ok(())
}

#[test]
fn mac_bitnet_benchmark_help_documents_one_shot_and_fixed_warm_paths() {
    bitnet()
        .args(["mac", "bitnet-benchmark", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("one-shot ask"))
        .stdout(predicate::str::contains("fixed warm"))
        .stdout(predicate::str::contains("--one-shot-prompt"))
        .stdout(predicate::str::contains("--model-path"))
        .stdout(predicate::str::contains("--tokenizer"))
        .stdout(predicate::str::contains("--json-out"));
}

#[test]
fn mac_bitnet_benchmark_rejects_full_metal_request_before_cache_lookup() {
    bitnet()
        .args(["--device", "apple-m4-metal", "mac", "bitnet-benchmark"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "mac bitnet-benchmark routes the supported Mac local-answer path",
        ))
        .stderr(predicate::str::contains("Full apple-m4-metal inference"));
}

#[test]
fn mac_bitnet_benchmark_requires_release_build() {
    bitnet()
        .args(["mac", "bitnet-benchmark"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac bitnet-benchmark must be run from a release build"));
}

#[test]
fn mac_bitnet_warm_rejects_full_metal_request_before_cache_lookup() {
    bitnet()
        .args(["--device", "apple-m4-metal", "mac", "bitnet-warm"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "mac bitnet-warm routes the supported Mac local-answer path",
        ))
        .stderr(predicate::str::contains("Full apple-m4-metal inference"));
}

#[test]
fn mac_bitnet_warm_rejects_non_bitnet_model_before_cache_lookup() {
    bitnet()
        .args(["mac", "bitnet-warm", "--model-id", "qwen2.5-0.5b-instruct-q8_0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "`bitnet mac bitnet-warm` only supports microsoft-bitnet-b1.58-2B-4T-i2s",
        ));
}

#[test]
fn mac_bitnet_warm_rejects_single_operator_prompt_before_cache_lookup() {
    bitnet()
        .args(["mac", "bitnet-warm", "--prompt", "What is 2+2?"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "`bitnet mac bitnet-warm --prompt` requires at least two prompt values",
        ))
        .stderr(predicate::str::contains("BitNet warm session requires").not());
}

#[test]
fn mac_bitnet_warm_rejects_non_repeated_operator_prompts_before_cache_lookup() {
    bitnet()
        .args([
            "mac",
            "bitnet-warm",
            "--prompt",
            "What is 2+2?",
            "--prompt",
            "Name the capital of France.",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("requires at least one exact repeated prompt"))
        .stderr(predicate::str::contains("BitNet warm session requires").not());
}

#[test]
fn mac_bitnet_warm_rejects_empty_operator_prompt_before_cache_lookup() {
    bitnet()
        .args([
            "mac",
            "bitnet-warm",
            "--prompt",
            "What is 2+2?",
            "--prompt",
            "",
            "--prompt",
            "What is 2+2?",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("value 2 must not be empty"))
        .stderr(predicate::str::contains("BitNet warm session requires").not());
}

#[test]
fn mac_bitnet_warm_rejects_profile_prompt_mix_before_cache_lookup() {
    bitnet()
        .args([
            "mac",
            "bitnet-warm",
            "--profile",
            "resident_25",
            "--prompt",
            "Answer with a single digit: 2+2=",
            "--prompt",
            "Answer with a single digit: 2+2=",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "`bitnet mac bitnet-warm --profile` cannot be combined with --prompt",
        ))
        .stderr(predicate::str::contains("BitNet warm session requires").not());
}

#[test]
fn mac_bitnet_warm_writes_failure_receipt_for_missing_tokenizer()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt = dir.path().join("bitnet-warm-failure.json");
    let tokenizer = dir.path().join("missing-tokenizer.json");
    let receipt_str = receipt.to_string_lossy().into_owned();
    let tokenizer_str = tokenizer.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "bitnet-warm",
            "--model-path",
            "missing-bitnet.gguf",
            "--tokenizer",
            tokenizer_str.as_str(),
            "--prompt",
            "Answer with a single digit: 2+2=",
            "--prompt",
            "Answer with a single digit: 2+2=",
            "--timeout-seconds",
            "60",
            "--progress",
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("accepted external tokenizer"))
        .stderr(predicate::str::contains("failure receipt written"))
        .stderr(predicate::str::contains("Repair guidance:"))
        .stderr(predicate::str::contains("mac bitnet-warm progress: tokenizer_verify_start"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "bitnet_apple_m4_warm_session_failure");
    assert_eq!(receipt_json["operator_command"], "mac bitnet-warm");
    assert_eq!(receipt_json["failure"]["stage"], "tokenizer_missing");
    assert_eq!(receipt_json["fallback_used"], false);
    assert_eq!(receipt_json["generation"]["generated_tokens"], 0);
    assert_eq!(receipt_json["generation"]["partial_generation_available"], false);
    assert_eq!(receipt_json["timeout_boundary"]["enforced"], true);
    assert_eq!(receipt_json["timeout_boundary"]["reached"], false);
    assert_eq!(receipt_json["progress"]["enabled"], true);
    let stage_taxonomy =
        receipt_json["progress"]["stage_taxonomy"].as_array().ok_or("stage taxonomy missing")?;
    assert!(stage_taxonomy.iter().any(|stage| stage.as_str() == Some("receipt_write")));
    assert_eq!(receipt_json["mac_bitnet_claim_boundary"]["chat_enabled"], false);
    assert_eq!(receipt_json["mac_bitnet_claim_boundary"]["serve_enabled"], false);

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("bitnet_apple_m4_warm_session_failure"))
        .stdout(predicate::str::contains("\"generated_tokens\": 0"));
    Ok(())
}

#[test]
fn mac_doctor_help_documents_health_verdict() {
    bitnet()
        .args(["mac", "doctor", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("health verdict"))
        .stdout(predicate::str::contains("--json"))
        .stdout(predicate::str::contains("--include-bitnet"))
        .stdout(predicate::str::contains("--run-smoke"))
        .stdout(predicate::str::contains("--json-out <PATH>"))
        .stdout(predicate::str::contains("--max-new-tokens"));
}

#[test]
fn mac_doctor_missing_cache_points_to_model_fetch_and_writes_receipt()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let receipt = dir.path().join("doctor.json");
    let cache_str = cache.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "doctor",
            "--cache-dir",
            cache_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
            "--json",
            "--include-bitnet",
        ])
        .assert()
        .failure()
        .stdout(predicate::str::contains("\"artifact_kind\": \"apple_m4_slm_doctor\""))
        .stdout(predicate::str::contains("\"repair_flows\""))
        .stderr(predicate::str::contains("Mac doctor cannot pass"))
        .stderr(predicate::str::contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "apple_m4_slm_doctor");
    assert_eq!(receipt_json["result"], "fail");
    assert_eq!(receipt_json["checks"]["cache"]["ready"], false);
    assert_eq!(receipt_json["checks"]["cache"]["symlink_status"], "not_symlink");
    assert_eq!(receipt_json["checks"]["unsupported_backend"]["rejected"], true);
    assert_eq!(receipt_json["checks"]["bitnet_ask"]["checked"], true);
    assert_eq!(receipt_json["checks"]["bitnet_ask"]["advisory"], true);
    assert_eq!(receipt_json["checks"]["bitnet_ask"]["blocks_doctor"], false);
    assert_eq!(
        receipt_json["checks"]["bitnet_ask"]["model"]["id"],
        "microsoft-bitnet-b1.58-2B-4T-i2s"
    );
    assert_eq!(receipt_json["checks"]["bitnet_ask"]["claim_boundary"]["chat_enabled"], false);
    assert_eq!(receipt_json["checks"]["bitnet_ask"]["claim_boundary"]["serve_enabled"], false);
    assert_eq!(
        receipt_json["checks"]["bitnet_ask"]["claim_boundary"]["bitnet_fixed_prompt_warm_session"],
        true
    );
    assert_eq!(receipt_json["checks"]["bitnet_ask"]["model"]["mac_bitnet_warm_enabled"], true);
    assert!(
        receipt_json["checks"]["bitnet_ask"]["commands"]["warm_cached_model"]
            .as_str()
            .unwrap_or_default()
            .contains("bitnet mac bitnet-warm")
    );
    assert!(
        receipt_json["checks"]["bitnet_ask"]["commands"]["models"]
            .as_str()
            .unwrap_or_default()
            .contains("bitnet mac models")
    );
    assert_eq!(receipt_json["readiness"]["dense_slm"]["status"], "cache_repair_required");
    assert!(
        receipt_json["readiness"]["dense_slm"]["cache_repair_guidance"]
            .as_str()
            .is_some_and(|guidance| guidance.contains("bitnet model fetch"))
    );
    assert_eq!(receipt_json["readiness"]["bitnet"]["chat_enabled"], false);
    assert_eq!(receipt_json["readiness"]["bitnet"]["serve_enabled"], false);
    assert!(
        receipt_json["readiness"]["bitnet"]["last_matching_receipts"]["variable_warm"]
            .as_str()
            .is_some_and(|path| path.contains("variable-warm-session.json"))
    );
    assert_eq!(
        receipt_json["readiness"]["bitnet"]["claim_boundary"]["bitnet_quality_claimed"],
        false
    );
    assert_eq!(receipt_json["mac_claim_boundary"]["bitnet_quality_claimed"], false);
    Ok(())
}

#[cfg(unix)]
#[test]
fn mac_doctor_stale_symlink_reports_repair_guidance() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let model_dir = cache.join("qwen2.5-0.5b-instruct-q8_0");
    std::fs::create_dir_all(&model_dir)?;
    let cache_file = model_dir.join("qwen2.5-0.5b-instruct-q8_0.gguf");
    std::os::unix::fs::symlink("missing-target.gguf", &cache_file)?;
    let receipt = dir.path().join("doctor-stale-symlink.json");
    let cache_str = cache.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "doctor",
            "--cache-dir",
            cache_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("stale symlink"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["checks"]["cache"]["state"], "stale-symlink");
    assert_eq!(receipt_json["checks"]["cache"]["symlink_status"], "stale_symlink");
    assert_eq!(receipt_json["checks"]["cache"]["stale_symlink"], true);
    assert!(
        receipt_json["readiness"]["repair_flows"]["stale_symlink"]
            .as_str()
            .is_some_and(|guidance| guidance.contains("prune and fetch"))
    );
    assert!(
        receipt_json["readiness"]["dense_slm"]["cache_repair_guidance"]
            .as_str()
            .is_some_and(|guidance| guidance.contains("stale symlink"))
    );
    Ok(())
}

#[test]
fn mac_doctor_rejects_full_metal_request_before_cache_lookup() {
    bitnet()
        .args(["--device", "apple-m4-metal", "mac", "doctor"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac doctor routes the supported Mac local-answer path"))
        .stderr(predicate::str::contains("Full apple-m4-metal inference"));
}

#[test]
fn model_prune_dry_run_json_defaults_to_all_without_deleting()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let cache = dir.path().join("models");
    let model_dir = cache.join("qwen2.5-0.5b-instruct-q8_0");
    std::fs::create_dir_all(&model_dir)?;
    let marker = model_dir.join("local-marker.txt");
    std::fs::write(&marker, b"cached")?;
    let cache_str = cache.to_string_lossy().into_owned();

    let output = bitnet()
        .args(["model", "prune", "--dry-run", "--json", "--cache-dir", cache_str.as_str()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let receipt: serde_json::Value = serde_json::from_slice(&output)?;

    assert_eq!(receipt["artifact_kind"], "bitnet_model_prune_dry_run");
    assert_eq!(receipt["scope"], "all_supported_models");
    assert_eq!(receipt["dry_run"], true);
    assert_eq!(receipt["deletes_user_data"], false);
    assert_eq!(receipt["would_remove_count"].as_u64(), Some(1));
    assert!(marker.exists(), "dry-run prune must not delete cache files");
    let qwen_row = receipt["results"]
        .as_array()
        .and_then(|rows| rows.iter().find(|row| row["id"] == "qwen2.5-0.5b-instruct-q8_0"))
        .ok_or("missing qwen prune dry-run row")?;
    assert_eq!(qwen_row["action"], "would_remove");
    assert_eq!(qwen_row["removed"], false);
    assert!(
        qwen_row["repair_guidance"]
            .as_str()
            .is_some_and(|guidance| guidance.contains("without --dry-run"))
    );
    Ok(())
}

#[test]
fn mac_serve_help_documents_health_ready_surface() {
    bitnet()
        .args(["mac", "serve", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("health and readiness endpoints"))
        .stdout(predicate::str::contains("--model-id <MODEL_ID>"))
        .stdout(predicate::str::contains("--host <HOST>"))
        .stdout(predicate::str::contains("--allow-non-loopback"))
        .stdout(predicate::str::contains("--port <PORT>"))
        .stdout(predicate::str::contains("--max-request-bytes <MAX_REQUEST_BYTES>"))
        .stdout(predicate::str::contains("--max-concurrent-requests <MAX_CONCURRENT_REQUESTS>"))
        .stdout(predicate::str::contains("--max-queued-requests <MAX_QUEUED_REQUESTS>"))
        .stdout(predicate::str::contains("--queue-timeout-ms <QUEUE_TIMEOUT_MS>"))
        .stdout(predicate::str::contains("--trace"))
        .stdout(predicate::str::contains("--receipt-dir <PATH>"));
}

#[test]
fn mac_serve_missing_cache_fails_before_listening() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = dir.path().join("models");
    let receipt_dir = dir.path().join("receipts");
    let cache_str = cache.to_string_lossy().into_owned();
    let receipt_str = receipt_dir.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "serve",
            "--cache-dir",
            cache_str.as_str(),
            "--receipt-dir",
            receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac serve cannot start"))
        .stderr(predicate::str::contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0"));
}

#[test]
fn mac_serve_rejects_full_metal_request_before_cache_lookup() {
    bitnet()
        .args(["--device", "apple-m4-metal", "mac", "serve"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac serve routes the supported Mac local service path"))
        .stderr(predicate::str::contains("Full apple-m4-metal inference"));
}

#[test]
fn mac_serve_requires_explicit_non_loopback_opt_in_before_cache_lookup() {
    bitnet()
        .args(["mac", "serve", "--host", "0.0.0.0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("defaults to loopback-only binding"))
        .stderr(predicate::str::contains("--allow-non-loopback"));
}

#[test]
fn mac_serve_rejects_tiny_request_size_limit_before_cache_lookup() {
    bitnet()
        .args(["mac", "serve", "--max-request-bytes", "1024"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--max-request-bytes must be at least 4096"));
}

#[test]
fn mac_serve_rejects_zero_concurrency_before_cache_lookup() {
    bitnet()
        .args(["mac", "serve", "--max-concurrent-requests", "0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--max-concurrent-requests must be at least 1"));
}

#[test]
fn mac_serve_rejects_zero_queue_timeout_before_cache_lookup() {
    bitnet()
        .args(["mac", "serve", "--queue-timeout-ms", "0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--queue-timeout-ms must be at least 1"));
}

#[test]
fn mac_serve_smoke_help_documents_dense_conformance_receipt() {
    bitnet()
        .args(["mac", "serve-smoke", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("dense M4 local-server conformance smoke"))
        .stdout(predicate::str::contains("--model-id <MODEL_ID>"))
        .stdout(predicate::str::contains("--receipt-dir <PATH>"))
        .stdout(predicate::str::contains("--json-out <PATH>"));
}

#[test]
fn mac_serve_smoke_missing_cache_fails_before_generation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = dir.path().join("models");
    let receipt_dir = dir.path().join("receipts");
    let json_out = dir.path().join("serve-smoke.json");
    let cache_str = cache.to_string_lossy().into_owned();
    let receipt_str = receipt_dir.to_string_lossy().into_owned();
    let json_out_str = json_out.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "serve-smoke",
            "--cache-dir",
            cache_str.as_str(),
            "--receipt-dir",
            receipt_str.as_str(),
            "--json-out",
            json_out_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac serve-smoke cannot start"))
        .stderr(predicate::str::contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0"));
}

#[test]
fn mac_serve_smoke_rejects_full_metal_request_before_cache_lookup() {
    bitnet()
        .args(["--device", "apple-m4-metal", "mac", "serve-smoke"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac serve routes the supported Mac local service path"))
        .stderr(predicate::str::contains("Full apple-m4-metal inference"));
}

#[test]
fn mac_serve_failure_smoke_help_documents_failure_semantics_receipt() {
    bitnet()
        .args(["mac", "serve-failure-smoke", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("timeout/cancellation/failure semantics smoke"))
        .stdout(predicate::str::contains("--timeout-seconds <TIMEOUT_SECONDS>"))
        .stdout(predicate::str::contains("--receipt-dir <PATH>"))
        .stdout(predicate::str::contains("--json-out <PATH>"));
}

#[test]
fn mac_serve_failure_smoke_rejects_full_metal_request_before_cache_lookup() {
    bitnet()
        .args(["--device", "apple-m4-metal", "mac", "serve-failure-smoke"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac serve routes the supported Mac local service path"))
        .stderr(predicate::str::contains("Full apple-m4-metal inference"));
}

#[test]
fn mac_serve_backpressure_smoke_help_documents_queue_receipt() {
    bitnet()
        .args(["mac", "serve-backpressure-smoke", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("queue/backpressure smoke"))
        .stdout(predicate::str::contains("--max-queued-requests <MAX_QUEUED_REQUESTS>"))
        .stdout(predicate::str::contains("--queue-timeout-ms <QUEUE_TIMEOUT_MS>"))
        .stdout(predicate::str::contains("--receipt-dir <PATH>"))
        .stdout(predicate::str::contains("--json-out <PATH>"));
}

#[test]
fn mac_serve_backpressure_smoke_rejects_full_metal_request_before_cache_lookup() {
    bitnet()
        .args(["--device", "apple-m4-metal", "mac", "serve-backpressure-smoke"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac serve routes the supported Mac local service path"))
        .stderr(predicate::str::contains("Full apple-m4-metal inference"));
}

#[test]
fn mac_chat_help_documents_resident_prompts() {
    bitnet()
        .args(["mac", "chat", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("resident Apple M4 CPU/NEON SLM session"))
        .stdout(predicate::str::contains("--prompt <TEXT>"))
        .stdout(predicate::str::contains("--stdin"))
        .stdout(predicate::str::contains("--interactive"))
        .stdout(predicate::str::contains("/exit"))
        .stdout(predicate::str::contains("--model-path <PATH>"))
        .stdout(predicate::str::contains("--tokenizer <PATH>"))
        .stdout(predicate::str::contains("--bitnet-chat-gate-receipt <PATH>"))
        .stdout(predicate::str::contains("--no-stream"))
        .stdout(predicate::str::contains("--no-turn-receipts"));
}

#[test]
fn mac_chat_requires_two_prompts_before_cache_lookup() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = dir.path().join("models");
    let cache_str = cache.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "chat",
            "--prompt",
            "What is 2+2? Answer briefly.",
            "--cache-dir",
            cache_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac chat requires at least two prompts"))
        .stderr(predicate::str::contains("bitnet model fetch").not());
}

#[test]
fn mac_chat_rejects_stdin_interactive_conflict_before_cache_lookup() {
    bitnet()
        .args(["mac", "chat", "--stdin", "--interactive"])
        .write_stdin("What is 2+2? Answer briefly.\nName the capital of France.\n/exit\n")
        .assert()
        .failure()
        .stderr(predicate::str::contains("--stdin and --interactive cannot be used together"))
        .stderr(predicate::str::contains("bitnet model fetch").not());
}

#[test]
fn mac_chat_interactive_collects_two_prompts_before_cache_lookup() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = dir.path().join("models");
    let cache_str = cache.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "chat", "--interactive", "--cache-dir", cache_str.as_str()])
        .write_stdin("What is 2+2? Answer briefly.\nName the capital of France.\n/exit\n")
        .assert()
        .failure()
        .stderr(predicate::str::contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0"))
        .stderr(predicate::str::contains("requires at least two prompts").not());
}

#[test]
fn mac_chat_interactive_exit_with_one_prompt_fails_before_cache_lookup() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = dir.path().join("models");
    let cache_str = cache.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "chat", "--interactive", "--cache-dir", cache_str.as_str()])
        .write_stdin("What is 2+2? Answer briefly.\n/exit\n")
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac chat requires at least two prompts"))
        .stderr(predicate::str::contains("bitnet model fetch").not());
}

#[test]
fn mac_chat_accepts_two_prompts_before_cache_lookup() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = dir.path().join("models");
    let cache_str = cache.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "chat",
            "--prompt",
            "What is 2+2? Answer briefly.",
            "--prompt",
            "Name the capital of France.",
            "--cache-dir",
            cache_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0"))
        .stderr(predicate::str::contains("requires at least two prompts").not());
}

#[test]
fn mac_chat_rejects_full_metal_request_before_cache_lookup() {
    bitnet()
        .args([
            "--device",
            "apple-m4-metal",
            "mac",
            "chat",
            "--prompt",
            "What is 2+2? Answer briefly.",
            "--prompt",
            "Name the capital of France.",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac chat routes the supported Mac local-answer path"))
        .stderr(predicate::str::contains("Full apple-m4-metal inference"));
}

#[test]
fn mac_chat_smoke_help_documents_dense_conformance_receipt() {
    bitnet()
        .args(["mac", "chat-smoke", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("fixed dense SLM resident chat conformance smoke"))
        .stdout(predicate::str::contains("--model-id <MODEL_ID>"))
        .stdout(predicate::str::contains("--timeout-seconds <TIMEOUT_SECONDS>"))
        .stdout(predicate::str::contains("--json-out <PATH>"));
}

#[test]
fn mac_chat_smoke_reaches_cache_lookup_for_default_model() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = dir.path().join("models");
    let cache_str = cache.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "chat-smoke", "--cache-dir", cache_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0"));
}

#[test]
fn mac_chat_smoke_rejects_full_metal_request_before_cache_lookup() {
    bitnet()
        .args(["--device", "apple-m4-metal", "mac", "chat-smoke"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "mac chat-smoke routes the supported Mac local-answer path",
        ))
        .stderr(predicate::str::contains("Full apple-m4-metal inference"));
}

#[test]
fn mac_regression_help_documents_advisory_mode() {
    bitnet()
        .args(["mac", "regression", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("stored local envelope"))
        .stdout(predicate::str::contains("BitNet"))
        .stdout(predicate::str::contains("--baseline <PATH>"))
        .stdout(predicate::str::contains("--fail-on-drift"));
}

#[test]
fn mac_regression_accepts_matching_warm_session_receipt() {
    let receipt = workspace_path(
        "ci/hardware/apple-m4-mac-mini/2026-05-09/M4-SLM-EX-008/resident-25-64.json",
    );
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "regression",
            receipt_str.as_str(),
            "--baseline",
            receipt_str.as_str(),
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"warning_count\": 0"))
        .stdout(predicate::str::contains("\"matched_context\": true"));
}

#[test]
fn mac_regression_fail_on_drift_turns_warning_into_error() {
    let baseline = workspace_path(
        "ci/hardware/apple-m4-mac-mini/2026-05-09/M4-SLM-EX-008/resident-25-64.json",
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let observed = dir.path().join("observed.json");
    let mut receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&baseline).expect("baseline")).expect("json");
    receipt["speed"]["throughput"]["decode_generated_tok_s"] = serde_json::json!(1.0);
    std::fs::write(&observed, serde_json::to_vec_pretty(&receipt).expect("json")).expect("write");
    let baseline_str = baseline.to_string_lossy().into_owned();
    let observed_str = observed.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "regression",
            observed_str.as_str(),
            "--baseline",
            baseline_str.as_str(),
            "--fail-on-drift",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Mac regression drift exceeded advisory thresholds"));
}

#[test]
fn mac_regression_accepts_matching_slm_eval_summary_report()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let baseline = dir.path().join("baseline-summary.json");
    let observed = dir.path().join("observed-summary.json");
    std::fs::write(&baseline, serde_json::to_vec_pretty(&slm_eval_summary_report())?)?;
    std::fs::write(&observed, serde_json::to_vec_pretty(&slm_eval_summary_report())?)?;
    let baseline_str = baseline.to_string_lossy().into_owned();
    let observed_str = observed.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "regression",
            observed_str.as_str(),
            "--baseline",
            baseline_str.as_str(),
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_slm_eval_summary"))
        .stdout(predicate::str::contains("\"warning_count\": 0"))
        .stdout(predicate::str::contains("\"matched_context\": true"));
    Ok(())
}

#[test]
fn mac_regression_fail_on_slm_eval_summary_drift_turns_warning_into_error()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let baseline = dir.path().join("baseline-summary.json");
    let observed = dir.path().join("observed-summary.json");
    let mut receipt = slm_eval_summary_report();
    receipt["accuracy"]["cases_passed"] = serde_json::json!(8);
    receipt["speed"]["decode_tok_s_p50"] = serde_json::json!(12.0);
    std::fs::write(&baseline, serde_json::to_vec_pretty(&slm_eval_summary_report())?)?;
    std::fs::write(&observed, serde_json::to_vec_pretty(&receipt)?)?;
    let baseline_str = baseline.to_string_lossy().into_owned();
    let observed_str = observed.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "regression",
            observed_str.as_str(),
            "--baseline",
            baseline_str.as_str(),
            "--fail-on-drift",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Mac regression drift exceeded advisory thresholds"));
    Ok(())
}

#[test]
fn mac_regression_rejects_slm_eval_summary_context_mismatch()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let baseline = dir.path().join("baseline-summary.json");
    let observed = dir.path().join("observed-summary.json");
    let mut receipt = slm_eval_summary_report();
    receipt["model"]["sha256"] =
        serde_json::json!("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    std::fs::write(&baseline, serde_json::to_vec_pretty(&slm_eval_summary_report())?)?;
    std::fs::write(&observed, serde_json::to_vec_pretty(&receipt)?)?;
    let baseline_str = baseline.to_string_lossy().into_owned();
    let observed_str = observed.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "regression", observed_str.as_str(), "--baseline", baseline_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("model.sha256 mismatch"));
    Ok(())
}

#[test]
fn mac_regression_receipts_check_reports_slm_eval_summary_warning()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let baseline = dir.path().join("baseline-summary.json");
    let observed = dir.path().join("observed-summary.json");
    let mut receipt = slm_eval_summary_report();
    receipt["speed"]["ttft_ms_p50"] = serde_json::json!(2080.0);
    std::fs::write(&baseline, serde_json::to_vec_pretty(&slm_eval_summary_report())?)?;
    std::fs::write(&observed, serde_json::to_vec_pretty(&receipt)?)?;
    let baseline_str = baseline.to_string_lossy().into_owned();
    let observed_str = observed.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "receipts-check",
            observed_str.as_str(),
            "--regression-baseline",
            baseline_str.as_str(),
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("speed.ttft_ms_p50"))
        .stdout(predicate::str::contains("\"warning_count\": 1"));
    Ok(())
}

#[test]
fn mac_regression_reports_slm_eval_task_family_warning() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let baseline = dir.path().join("baseline-summary.json");
    let observed = dir.path().join("observed-summary.json");
    let mut receipt = slm_eval_summary_report();
    receipt["scoring_summary"]["passed"] = serde_json::json!(8);
    receipt["scoring_summary"]["failed"] = serde_json::json!(2);
    receipt["task_families"]["arithmetic_exact"]["cases_passed"] = serde_json::json!(8);
    receipt["task_families"]["arithmetic_exact"]["pass_rate"] = serde_json::json!(0.8);
    std::fs::write(&baseline, serde_json::to_vec_pretty(&slm_eval_summary_report())?)?;
    std::fs::write(&observed, serde_json::to_vec_pretty(&receipt)?)?;
    let baseline_str = baseline.to_string_lossy().into_owned();
    let observed_str = observed.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "regression",
            observed_str.as_str(),
            "--baseline",
            baseline_str.as_str(),
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("scoring_summary.passed"))
        .stdout(predicate::str::contains("scoring_summary.failed"))
        .stdout(predicate::str::contains("task_family:arithmetic_exact"))
        .stdout(predicate::str::contains("task_families.arithmetic_exact.cases_passed"));
    Ok(())
}

#[test]
fn mac_regression_accepts_matching_slm_benchmark_v2_summary()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let baseline = dir.path().join("baseline-benchmark.json");
    let observed = dir.path().join("observed-benchmark.json");
    std::fs::write(&baseline, serde_json::to_vec_pretty(&slm_benchmark_v2_summary())?)?;
    std::fs::write(&observed, serde_json::to_vec_pretty(&slm_benchmark_v2_summary())?)?;
    let baseline_str = baseline.to_string_lossy().into_owned();
    let observed_str = observed.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "regression",
            observed_str.as_str(),
            "--baseline",
            baseline_str.as_str(),
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_slm_benchmark_v2"))
        .stdout(predicate::str::contains("\"warning_count\": 0"))
        .stdout(predicate::str::contains("\"matched_context\": true"));
    Ok(())
}

#[test]
fn mac_regression_fail_on_slm_benchmark_v2_drift_turns_warning_into_error()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let baseline = dir.path().join("baseline-benchmark.json");
    let observed = dir.path().join("observed-benchmark.json");
    let mut receipt = slm_benchmark_v2_summary();
    receipt["speed"]["input_tok_s_p50"] = serde_json::json!(80.0);
    receipt["profiles"][0]["timing"]["time_to_first_token_ms"]["p50"] = serde_json::json!(2400.0);
    receipt["profiles"][0]["timing"]["time_to_first_token_ms"]["p90"] = serde_json::json!(2500.0);
    receipt["profiles"][0]["timing"]["time_to_first_token_ms"]["p99"] = serde_json::json!(2600.0);
    std::fs::write(&baseline, serde_json::to_vec_pretty(&slm_benchmark_v2_summary())?)?;
    std::fs::write(&observed, serde_json::to_vec_pretty(&receipt)?)?;
    let baseline_str = baseline.to_string_lossy().into_owned();
    let observed_str = observed.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "regression",
            observed_str.as_str(),
            "--baseline",
            baseline_str.as_str(),
            "--fail-on-drift",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Mac regression drift exceeded advisory thresholds"));
    Ok(())
}

#[test]
fn mac_regression_rejects_slm_benchmark_v2_context_mismatch()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let baseline = dir.path().join("baseline-benchmark.json");
    let observed = dir.path().join("observed-benchmark.json");
    let mut receipt = slm_benchmark_v2_summary();
    receipt["model_cache"]["sha256"] =
        serde_json::json!("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    std::fs::write(&baseline, serde_json::to_vec_pretty(&slm_benchmark_v2_summary())?)?;
    std::fs::write(&observed, serde_json::to_vec_pretty(&receipt)?)?;
    let baseline_str = baseline.to_string_lossy().into_owned();
    let observed_str = observed.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "regression", observed_str.as_str(), "--baseline", baseline_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("model_cache.sha256 mismatch"));
    Ok(())
}

#[test]
fn mac_regression_accepts_matching_bitnet_eval_answer_corpus()
-> Result<(), Box<dyn std::error::Error>> {
    let receipt =
        workspace_path("ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-eval/answer-corpus.json");
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "regression",
            receipt_str.as_str(),
            "--baseline",
            receipt_str.as_str(),
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("bitnet_apple_m4_local_answer_corpus"))
        .stdout(predicate::str::contains("\"warning_count\": 0"))
        .stdout(predicate::str::contains("\"matched_context\": true"));
    Ok(())
}

#[test]
fn mac_regression_fail_on_bitnet_eval_quality_drift_turns_warning_into_error()
-> Result<(), Box<dyn std::error::Error>> {
    let baseline =
        workspace_path("ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-eval/answer-corpus.json");
    let dir = tempfile::tempdir()?;
    let observed = dir.path().join("observed-bitnet-eval.json");
    let mut receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(&baseline)?)?;
    receipt["quality_summary"]["passed"] = serde_json::json!(74);
    receipt["quality_summary"]["failed"] = serde_json::json!(26);
    receipt["scoring_summary"]["passed"] = serde_json::json!(74);
    receipt["scoring_summary"]["failed"] = serde_json::json!(26);
    receipt["task_family_summary"]["arithmetic_exact"]["passed"] = serde_json::json!(9);
    receipt["task_family_summary"]["arithmetic_exact"]["failed"] = serde_json::json!(1);
    receipt["task_family_summary"]["arithmetic_exact"]["scoring"]["passed"] = serde_json::json!(9);
    receipt["task_family_summary"]["arithmetic_exact"]["scoring"]["failed"] = serde_json::json!(1);
    std::fs::write(&observed, serde_json::to_vec_pretty(&receipt)?)?;
    let baseline_str = baseline.to_string_lossy().into_owned();
    let observed_str = observed.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "regression",
            observed_str.as_str(),
            "--baseline",
            baseline_str.as_str(),
            "--fail-on-drift",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Mac regression drift exceeded advisory thresholds"));
    Ok(())
}

#[test]
fn mac_regression_rejects_bitnet_eval_context_mismatch() -> Result<(), Box<dyn std::error::Error>> {
    let baseline =
        workspace_path("ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-eval/answer-corpus.json");
    let dir = tempfile::tempdir()?;
    let observed = dir.path().join("observed-bitnet-eval.json");
    let mut receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(&baseline)?)?;
    receipt["model"]["revision"] = serde_json::json!("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    std::fs::write(&observed, serde_json::to_vec_pretty(&receipt)?)?;
    let baseline_str = baseline.to_string_lossy().into_owned();
    let observed_str = observed.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "regression", observed_str.as_str(), "--baseline", baseline_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("model.revision mismatch"));
    Ok(())
}

#[test]
fn mac_regression_accepts_matching_bitnet_warm_session() -> Result<(), Box<dyn std::error::Error>> {
    let receipt = workspace_path(
        "ci/hardware/apple-m4-mac-mini/2026-05-16T0626Z/bitnet-productization/variable-warm-session.json",
    );
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "regression",
            receipt_str.as_str(),
            "--baseline",
            receipt_str.as_str(),
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("bitnet_apple_m4_warm_session"))
        .stdout(predicate::str::contains("\"warning_count\": 0"))
        .stdout(predicate::str::contains("\"matched_context\": true"));
    Ok(())
}

#[test]
fn mac_regression_fail_on_bitnet_warm_session_drift_turns_warning_into_error()
-> Result<(), Box<dyn std::error::Error>> {
    let baseline = workspace_path(
        "ci/hardware/apple-m4-mac-mini/2026-05-16T0626Z/bitnet-productization/variable-warm-session.json",
    );
    let dir = tempfile::tempdir()?;
    let observed = dir.path().join("observed-bitnet-warm.json");
    let mut receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(&baseline)?)?;
    receipt["speed"]["throughput"]["decode_generated_tok_s"] = serde_json::json!(0.5);
    std::fs::write(&observed, serde_json::to_vec_pretty(&receipt)?)?;
    let baseline_str = baseline.to_string_lossy().into_owned();
    let observed_str = observed.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "regression",
            observed_str.as_str(),
            "--baseline",
            baseline_str.as_str(),
            "--fail-on-drift",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Mac regression drift exceeded advisory thresholds"));
    Ok(())
}

#[test]
fn mac_regression_rejects_bitnet_warm_session_context_mismatch()
-> Result<(), Box<dyn std::error::Error>> {
    let baseline = workspace_path(
        "ci/hardware/apple-m4-mac-mini/2026-05-16T0626Z/bitnet-productization/variable-warm-session.json",
    );
    let dir = tempfile::tempdir()?;
    let observed = dir.path().join("observed-bitnet-warm.json");
    let mut receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(&baseline)?)?;
    receipt["tokenizer"]["pretokenizer_authority"] = serde_json::json!("other-tokenizer");
    std::fs::write(&observed, serde_json::to_vec_pretty(&receipt)?)?;
    let baseline_str = baseline.to_string_lossy().into_owned();
    let observed_str = observed.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "regression", observed_str.as_str(), "--baseline", baseline_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("tokenizer.pretokenizer_authority mismatch"));
    Ok(())
}

#[test]
fn mac_regression_receipts_check_accepts_bitnet_warm_session_baseline()
-> Result<(), Box<dyn std::error::Error>> {
    let receipt = workspace_path(
        "ci/hardware/apple-m4-mac-mini/2026-05-16T0626Z/bitnet-productization/variable-warm-session.json",
    );
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "receipts-check",
            receipt_str.as_str(),
            "--regression-baseline",
            receipt_str.as_str(),
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("bitnet_apple_m4_warm_session"))
        .stdout(predicate::str::contains("\"warning_count\": 0"))
        .stdout(predicate::str::contains("\"matched_context\": true"));
    Ok(())
}

#[test]
fn mac_regression_accepts_matching_bitnet_benchmark_v1() -> Result<(), Box<dyn std::error::Error>> {
    let receipt =
        workspace_path("ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-benchmark/summary.json");
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "regression",
            receipt_str.as_str(),
            "--baseline",
            receipt_str.as_str(),
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("bitnet_apple_m4_benchmark_v1"))
        .stdout(predicate::str::contains("\"warning_count\": 0"))
        .stdout(predicate::str::contains("\"matched_context\": true"));
    Ok(())
}

#[test]
fn mac_regression_fail_on_bitnet_benchmark_drift_turns_warning_into_error()
-> Result<(), Box<dyn std::error::Error>> {
    let baseline =
        workspace_path("ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-benchmark/summary.json");
    let dir = tempfile::tempdir()?;
    let observed = dir.path().join("observed-bitnet-benchmark.json");
    let mut receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(&baseline)?)?;
    receipt["speed"]["decode_tok_s_p50"] = serde_json::json!(0.5);
    receipt["paths"]["fixed_warm"]["throughput"]["decode_tokens_per_second"]["p50"] =
        serde_json::json!(0.5);
    std::fs::write(&observed, serde_json::to_vec_pretty(&receipt)?)?;
    let baseline_str = baseline.to_string_lossy().into_owned();
    let observed_str = observed.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "regression",
            observed_str.as_str(),
            "--baseline",
            baseline_str.as_str(),
            "--fail-on-drift",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Mac regression drift exceeded advisory thresholds"));
    Ok(())
}

#[test]
fn mac_regression_rejects_bitnet_benchmark_context_mismatch()
-> Result<(), Box<dyn std::error::Error>> {
    let baseline =
        workspace_path("ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-benchmark/summary.json");
    let dir = tempfile::tempdir()?;
    let observed = dir.path().join("observed-bitnet-benchmark.json");
    let mut receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(&baseline)?)?;
    receipt["tokenizer"]["path"] = serde_json::json!("models/other-tokenizer.json");
    std::fs::write(&observed, serde_json::to_vec_pretty(&receipt)?)?;
    let baseline_str = baseline.to_string_lossy().into_owned();
    let observed_str = observed.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "regression", observed_str.as_str(), "--baseline", baseline_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("tokenizer.path mismatch"));
    Ok(())
}

#[test]
fn mac_bitnet_proof_help_documents_blocked_contract() {
    bitnet()
        .args(["mac", "bitnet-proof", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--accepted-artifact <PATH>"))
        .stdout(predicate::str::contains("--proof-receipt <PATH>"))
        .stdout(predicate::str::contains("--tokenizer-authority <AUTHORITY>"))
        .stdout(predicate::str::contains("--strict"));
}

#[test]
fn mac_bitnet_proof_missing_inputs_fail_clearly() {
    let dir = tempfile::tempdir().expect("tempdir");
    let receipt = dir.path().join("preflight.json");
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "bitnet-proof",
            "--model",
            "missing-bitnet.gguf",
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("M4 BitNet proof is blocked"))
        .stderr(predicate::str::contains("--tokenizer-authority"))
        .stderr(predicate::str::contains("--accepted-artifact"))
        .stderr(predicate::str::contains("accepted BitNet GGUF is missing"));
}

#[test]
fn mac_bitnet_proof_preflight_accepts_artifact_contract_without_running_model()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let model = dir.path().join("accepted-bitnet.gguf");
    let accepted = dir.path().join("accepted-artifact.json");
    let receipt = dir.path().join("preflight.json");
    std::fs::write(&model, b"placeholder gguf")?;
    std::fs::write(
        &accepted,
        serde_json::to_vec_pretty(&serde_json::json!({
            "accepted": true,
            "model": {
                "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162"
            },
            "tokenizer": {
                "authority": "llama-bpe-external"
            },
            "kernel_family": "i2_s"
        }))?,
    )?;
    let model_str = model.to_string_lossy().into_owned();
    let accepted_str = accepted.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "bitnet-proof",
            "--model",
            model_str.as_str(),
            "--tokenizer-authority",
            "llama-bpe-external",
            "--accepted-artifact",
            accepted_str.as_str(),
            "--strict",
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("preflight passed"))
        .stdout(predicate::str::contains(
            "does not enable `bitnet mac chat` or `bitnet mac serve`",
        ));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "apple_m4_bitnet_proof_preflight");
    assert_eq!(receipt_json["result"], "ready");
    assert_eq!(receipt_json["proof_executed"], false);
    assert_eq!(receipt_json["claim_boundary"]["bitnet_answer_quality_claimed"], false);
    Ok(())
}

#[test]
fn mac_bitnet_proof_validates_answer_corpus_receipt_without_artifact_sweep()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let model = dir.path().join("accepted-bitnet.gguf");
    let proof = dir.path().join("answer-corpus.json");
    let receipt = dir.path().join("preflight.json");
    std::fs::write(&model, b"placeholder gguf")?;
    std::fs::write(&proof, serde_json::to_vec_pretty(&bitnet_answer_corpus_proof_fixture(true))?)?;
    let model_str = model.to_string_lossy().into_owned();
    let proof_str = proof.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "bitnet-proof",
            "--model",
            model_str.as_str(),
            "--proof-receipt",
            proof_str.as_str(),
            "--strict",
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("proof receipt verified"))
        .stdout(predicate::str::contains(
            "does not enable `bitnet mac chat` or `bitnet mac serve`",
        ));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["result"], "verified");
    assert_eq!(receipt_json["proof_executed"], true);
    assert_eq!(receipt_json["proof_receipt"]["summary"]["valid"], true);
    assert_eq!(receipt_json["tokenizer"]["authority"]["source"], "external_tokenizer_json");
    assert_eq!(receipt_json["tokenizer"]["authority"]["ggml_pre"], "llama-bpe");
    assert_eq!(receipt_json["claim_boundary"]["m4_bitnet_answer_corpus_proof_verified"], true);
    assert_eq!(receipt_json["claim_boundary"]["bitnet_answer_corpus_quality_verified"], true);
    assert_eq!(receipt_json["claim_boundary"]["bitnet_answer_quality_claimed"], false);
    assert_eq!(receipt_json["claim_boundary"]["bitnet_mac_ask_chat_enabled"], false);
    assert_eq!(receipt_json["claim_boundary"]["bitnet_mac_serve_enabled"], false);
    Ok(())
}

#[test]
fn mac_bitnet_proof_receipt_does_not_require_local_model() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = tempfile::tempdir()?;
    let model = dir.path().join("missing-after-proof.gguf");
    let proof = dir.path().join("answer-corpus.json");
    let receipt = dir.path().join("preflight.json");
    std::fs::write(&proof, serde_json::to_vec_pretty(&bitnet_answer_corpus_proof_fixture(true))?)?;
    let model_str = model.to_string_lossy().into_owned();
    let proof_str = proof.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "bitnet-proof",
            "--model",
            model_str.as_str(),
            "--proof-receipt",
            proof_str.as_str(),
            "--strict",
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("proof receipt verified"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["result"], "verified");
    assert_eq!(receipt_json["model"]["exists"], false);
    assert_eq!(receipt_json["tokenizer"]["authority"]["source"], "external_tokenizer_json");
    Ok(())
}

#[test]
fn mac_bitnet_proof_rejects_answer_corpus_receipt_without_timing()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let model = dir.path().join("accepted-bitnet.gguf");
    let proof = dir.path().join("answer-corpus.json");
    let receipt = dir.path().join("preflight.json");
    std::fs::write(&model, b"placeholder gguf")?;
    std::fs::write(&proof, serde_json::to_vec_pretty(&bitnet_answer_corpus_proof_fixture(false))?)?;
    let model_str = model.to_string_lossy().into_owned();
    let proof_str = proof.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "bitnet-proof",
            "--model",
            model_str.as_str(),
            "--proof-receipt",
            proof_str.as_str(),
            "--strict",
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("proof receipt is not usable"))
        .stderr(predicate::str::contains("timing/latency"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["result"], "blocked");
    assert_eq!(receipt_json["proof_executed"], false);
    assert_eq!(receipt_json["proof_receipt"]["summary"]["valid"], false);
    Ok(())
}

fn bitnet_answer_corpus_proof_fixture(include_timing: bool) -> serde_json::Value {
    let mut case = serde_json::json!({
        "id": "math_2_plus_2",
        "status": "passed",
        "answer": " 4",
        "backend": {
            "requested_backend": "apple-m4-cpu-neon",
            "selected_backend": "apple-m4-cpu-neon",
            "runtime_api": "cpu",
            "fallback_used": false
        },
        "model": {
            "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
            "family": "bitnet"
        },
        "loader": {
            "mode": "real_gguf"
        },
        "tokenizer": {
            "strict": true,
            "pretokenizer_authority": "llama-bpe"
        },
        "prompt_template": "bitnetcpp-answer",
        "prompt": {
            "template_family": "bitnetcpp-answer"
        },
        "prompt_prefill": {
            "exercised": true
        },
        "quality": {
            "passed": true,
            "non_empty_answer": true
        },
        "token_ids": {
            "generated": [220, 19, 128009]
        },
        "tokens": {
            "generated": 3
        }
    });
    if include_timing {
        case["timing"] = serde_json::json!({"decode_total_ms": 12.0});
        case["latency"] = serde_json::json!({"total_ms": 34.0});
    }
    serde_json::json!({
        "artifact_kind": "bitnet_apple_m4_local_answer_corpus",
        "model": {
            "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
            "answer_ready_artifact_available": true,
            "answer_ready": {
                "state": "answer_ready"
            }
        },
        "tokenizer": {
            "source": "externally_supplied_llama_bpe",
            "strict": true,
            "authority": {
                "source": "external_tokenizer_json",
                "sha256": "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7",
                "ggml_pre": "llama-bpe"
            }
        },
        "quality_summary": {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "timeout": 0,
            "not_run": 0
        },
        "claim_boundary": {
            "answer_ready_artifact_available": true,
            "backend_quality_gate_passed": true,
            "coherent_output_observed": true,
            "coherent_answer_claimed": true,
            "diagnostic_only_until_answer_ready_artifact": false,
            "full_metal_inference_claimed": false,
            "neural_engine_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claimed": false
        },
        "cases": [case]
    })
}

#[test]
fn mac_receipts_check_accepts_valid_cpu_neon_answer_receipt()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("answer.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "artifact_kind": "inference_result",
            "requested_backend": "apple-m4-cpu-neon",
            "selected_backend": "apple-m4-cpu-neon",
            "runtime_api": "cpu",
            "fallback_used": false,
            "text": "4.",
            "tokens": {
                "generated": 1,
                "generated_ids": [19]
            },
            "model": {
                "sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"
            },
            "tokenizer": {
                "source": "gguf_metadata"
            },
            "mac_claim_boundary": {
                "full_metal_inference_claimed": false,
                "neural_engine_execution_claimed": false,
                "qk256_apple_claimed": false,
                "bitnet_quality_claimed": false
            }
        }))?,
    )?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"passed\": true"))
        .stdout(predicate::str::contains("apple-m4-cpu-neon"));
    Ok(())
}

#[test]
fn mac_receipts_check_accepts_golden_smoke_receipt() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("mac-smoke.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "artifact_kind": "apple_m4_slm_golden_smoke",
            "requested_backend": "apple-m4-cpu-neon",
            "selected_backend": "apple-m4-cpu-neon",
            "runtime_api": "cpu",
            "fallback_used": false,
            "prompt": "Answer with a single digit: 2+2=",
            "expected_text_fragment": "4",
            "expected_text_fragment_found": true,
            "text": "4",
            "tokens": {
                "generated": 1,
                "generated_ids": [19]
            },
            "model": {
                "sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"
            },
            "tokenizer": {
                "source": "gguf_metadata"
            },
            "cache_health": {
                "checked": true,
                "ready": true,
                "state": "ready",
                "disk": {
                    "checked": true,
                    "low_disk": false
                }
            },
            "mac_claim_boundary": {
                "golden_smoke": true,
                "full_metal_inference_claimed": false,
                "neural_engine_execution_claimed": false,
                "qk256_apple_claimed": false,
                "bitnet_quality_claimed": false,
                "broad_performance_claim": false,
                "speedup_claim": false
            }
        }))?,
    )?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_slm_golden_smoke"))
        .stdout(predicate::str::contains("\"passed\": true"));
    Ok(())
}

#[test]
fn mac_receipts_check_accepts_bitnet_warm_session_receipt() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("bitnet-warm.json");
    let prompt = "Answer with a single digit: 2+2=";
    let prompts = serde_json::json!([
        {
            "prompt_index": 0,
            "case_id": "prompt_01",
            "prompt": prompt,
            "text": "4",
            "generated_tokens": 2,
            "generated_token_ids": [19, 128009],
            "quality": {
                "passed": true,
                "valid_utf8": true,
                "printable_utf8": true,
                "non_empty": true,
                "no_replacement_chars": true,
                "mostly_text": true,
                "non_degenerate": true,
                "generated_tokens": 2,
                "distinct_generated_tokens": 2,
                "failed_rules": []
            },
            "timing": {
                "time_to_first_token_ms": 100,
                "total_ms": 200.0
            },
            "backend": {
                "requested_backend": "apple-m4-cpu-neon",
                "selected_backend": "apple-m4-cpu-neon",
                "runtime_api": "cpu",
                "fallback_used": false
            },
            "operator_ux": {
                "time_to_first_token_receipt": true
            }
        },
        {
            "prompt_index": 1,
            "case_id": "prompt_02",
            "prompt": "Name the capital of France. Answer with one word.",
            "text": "Paris",
            "generated_tokens": 1,
            "generated_token_ids": [60704],
            "quality": {
                "passed": true,
                "valid_utf8": true,
                "printable_utf8": true,
                "non_empty": true,
                "no_replacement_chars": true,
                "mostly_text": true,
                "non_degenerate": true,
                "generated_tokens": 1,
                "distinct_generated_tokens": 1,
                "failed_rules": []
            },
            "timing": {
                "time_to_first_token_ms": 100,
                "total_ms": 200.0
            },
            "backend": {
                "requested_backend": "apple-m4-cpu-neon",
                "selected_backend": "apple-m4-cpu-neon",
                "runtime_api": "cpu",
                "fallback_used": false
            },
            "operator_ux": {
                "time_to_first_token_receipt": true
            }
        },
        {
            "prompt_index": 2,
            "case_id": "prompt_03",
            "prompt": prompt,
            "text": "4",
            "generated_tokens": 2,
            "generated_token_ids": [19, 128009],
            "quality": {
                "passed": true,
                "valid_utf8": true,
                "printable_utf8": true,
                "non_empty": true,
                "no_replacement_chars": true,
                "mostly_text": true,
                "non_degenerate": true,
                "generated_tokens": 2,
                "distinct_generated_tokens": 2,
                "failed_rules": []
            },
            "timing": {
                "time_to_first_token_ms": 100,
                "total_ms": 200.0
            },
            "backend": {
                "requested_backend": "apple-m4-cpu-neon",
                "selected_backend": "apple-m4-cpu-neon",
                "runtime_api": "cpu",
                "fallback_used": false
            },
            "operator_ux": {
                "time_to_first_token_receipt": true
            }
        }
    ]);
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "artifact_kind": "bitnet_apple_m4_warm_session",
            "operator_command": "mac bitnet-warm",
            "requested_backend": "apple-m4-cpu-neon",
            "selected_backend": "apple-m4-cpu-neon",
            "runtime_api": "cpu",
            "fallback_used": false,
            "session": {
                "model_loaded_once": true,
                "tokenizer_loaded_once": true,
                "prompt_count": 3
            },
            "model": {
                "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
                "family": "bitnet",
                "loader_mode": "real_gguf"
            },
            "tokenizer": {
                "source": "explicit",
                "strict": true,
                "pretokenizer_authority": "llama-bpe"
            },
            "quality_summary": {
                "passed": true
            },
            "operator_ux": {
                "time_to_first_token_receipts": true
            },
            "determinism": {
                "checked": true,
                "passed": true,
                "repeated_prompt_groups": 1,
                "groups": [{
                    "prompt": prompt,
                    "attempt_count": 2,
                    "stable_generated_token_ids": true,
                    "stable_text": true
                }]
            },
            "prompts": prompts,
            "claim_boundary": {
                "bitnet_warm_session": true,
                "bitnet_quality_claimed": false,
                "full_metal_inference_claimed": false,
                "qk256_apple_claimed": false,
                "neural_engine_execution_claimed": false,
                "mpsgraph_inference_claimed": false,
                "broad_performance_claim": false,
                "speedup_claim": false
            },
            "mac_bitnet_claim_boundary": {
                "bitnet_warm_session": true,
                "chat_enabled": false,
                "serve_enabled": false,
                "full_metal_inference_claimed": false,
                "qk256_apple_claimed": false,
                "broad_performance_claim": false,
                "speedup_claim": false
            },
            "bitnet_quality_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false
        }))?,
    )?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("bitnet_apple_m4_warm_session"))
        .stdout(predicate::str::contains("\"prompt_count\": 3"))
        .stdout(predicate::str::contains("\"passed\": true"));
    Ok(())
}

#[test]
fn mac_receipts_check_accepts_operator_profile_summary() {
    let dir = tempfile::tempdir().expect("tempdir");
    let receipt_path = dir.path().join("operator-profiles.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "artifact_kind": "apple_m4_slm_operator_profiles",
            "requested_backend": "apple-m4-cpu-neon",
            "selected_backend": "apple-m4-cpu-neon",
            "runtime_api": "cpu",
            "fallback_used": false,
            "operator_thresholds": {
                "cold_load_separated": true,
                "model_tokenizer_reuse_visible": true,
                "model_tokenizer_reuse_visible_per_profile": true,
                "profiles_loaded_independently": true,
                "profile_set_model_loads": 3,
                "reuse_scope": "within_each_profile",
                "profiles_required": ["warm_16", "warm_32", "warm_64"],
                "thresholds_are_claim_bounds_not_speed_guarantees": true
            },
            "profiles": [
                {
                    "profile_id": "warm_16",
                    "requested_max_new_tokens": 16,
                    "prompt_count": 1,
                    "generated_tokens": 16,
                    "quality_passed": true,
                    "cold_load_separated": true,
                    "model_loaded_once": true,
                    "tokenizer_loaded_once": true,
                    "reuse_scope": "within_profile",
                    "resident_session": resident_session_json(),
                    "timing": {
                        "model_load_ms": 1000.0,
                        "tokenizer_load_ms": 25.0,
                        "warm_prompt_wall_ms": 8000.0,
                        "decode_total_ms": 5000.0,
                        "sampling_ms": 12.0,
                        "warm_prompt_generated_tok_s": 2.0,
                        "decode_generated_tok_s": 3.2
                    }
                },
                {
                    "profile_id": "warm_32",
                    "requested_max_new_tokens": 32,
                    "prompt_count": 1,
                    "generated_tokens": 32,
                    "quality_passed": true,
                    "cold_load_separated": true,
                    "model_loaded_once": true,
                    "tokenizer_loaded_once": true,
                    "reuse_scope": "within_profile",
                    "resident_session": resident_session_json(),
                    "timing": {
                        "model_load_ms": 1000.0,
                        "tokenizer_load_ms": 25.0,
                        "warm_prompt_wall_ms": 16000.0,
                        "decode_total_ms": 10000.0,
                        "sampling_ms": 24.0,
                        "warm_prompt_generated_tok_s": 2.0,
                        "decode_generated_tok_s": 3.2
                    }
                },
                {
                    "profile_id": "warm_64",
                    "requested_max_new_tokens": 64,
                    "prompt_count": 1,
                    "generated_tokens": 64,
                    "quality_passed": true,
                    "cold_load_separated": true,
                    "model_loaded_once": true,
                    "tokenizer_loaded_once": true,
                    "reuse_scope": "within_profile",
                    "resident_session": resident_session_json(),
                    "timing": {
                        "model_load_ms": 1000.0,
                        "tokenizer_load_ms": 25.0,
                        "warm_prompt_wall_ms": 32000.0,
                        "decode_total_ms": 20000.0,
                        "sampling_ms": 48.0,
                        "warm_prompt_generated_tok_s": 2.0,
                        "decode_generated_tok_s": 3.2
                    }
                }
            ],
            "mac_claim_boundary": {
                "full_metal_inference_claimed": false,
                "neural_engine_execution_claimed": false,
                "qk256_apple_claimed": false,
                "bitnet_quality_claimed": false,
                "broad_performance_claim": false,
                "speedup_claim": false
            }
        }))
        .expect("json"),
    )
    .expect("write receipt");
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_slm_operator_profiles"))
        .stdout(predicate::str::contains("\"prompt_count\": 3"));
}

#[test]
fn mac_receipts_check_accepts_performance_profile_summary() {
    let dir = tempfile::tempdir().expect("tempdir");
    let receipt_path = dir.path().join("performance-profiles.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "artifact_kind": "apple_m4_slm_performance_profiles",
            "requested_backend": "apple-m4-cpu-neon",
            "selected_backend": "apple-m4-cpu-neon",
            "runtime_api": "cpu",
            "fallback_used": false,
            "profile_set": "performance",
            "build": {
                "profile": "release",
                "release_mode": true
            },
            "operator_thresholds": {
                "cold_load_separated": true,
                "model_tokenizer_reuse_visible": true,
                "model_tokenizer_reuse_visible_per_profile": true,
                "profiles_loaded_independently": true,
                "profile_set_model_loads": 4,
                "reuse_scope": "within_each_profile",
                "profiles_required": ["warm_16", "warm_32", "warm_64", "warm_128"],
                "thresholds_are_claim_bounds_not_speed_guarantees": true
            },
            "performance_baseline": {
                "release_mode_required": true,
                "release_mode_observed": true,
                "warm_128_included": true,
                "broad_performance_claim": false,
                "speedup_claim": false
            },
            "allocation_audit": {
                "enabled": true,
                "method": "process_global_allocator_counter_delta",
                "scope": "selected Apple M4 CPU/NEON SLM warm-session profile set",
                "optimization_deferred": true,
                "ranked_hotspots": [
                    {"component": "model.forward", "alloc_count": 10, "alloc_bytes": 1024}
                ]
            },
            "profiles": [
                performance_profile_json("warm_16", 16, 8000.0, 5000.0),
                performance_profile_json("warm_32", 32, 16000.0, 10000.0),
                performance_profile_json("warm_64", 64, 32000.0, 20000.0),
                performance_profile_json("warm_128", 128, 64000.0, 40000.0)
            ],
            "mac_claim_boundary": {
                "full_metal_inference_claimed": false,
                "neural_engine_execution_claimed": false,
                "qk256_apple_claimed": false,
                "bitnet_quality_claimed": false,
                "broad_performance_claim": false,
                "speedup_claim": false
            }
        }))
        .expect("json"),
    )
    .expect("write receipt");
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_slm_performance_profiles"))
        .stdout(predicate::str::contains("\"prompt_count\": 4"));
}

#[test]
fn mac_benchmark_requires_release_build() {
    bitnet()
        .args(["mac", "benchmark", "--profile", "short_prompt_16_out"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac benchmark must be run from a release build"));
}

#[test]
fn mac_benchmark_accepts_resident_100_profile_before_release_gate() {
    bitnet()
        .args(["mac", "benchmark", "--profile", "resident_100"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac benchmark must be run from a release build"));
}

#[test]
fn mac_benchmark_accepts_mixed_model_switch_profile_before_release_gate() {
    bitnet()
        .args(["mac", "benchmark", "--profile", "mixed_model_switch"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac benchmark must be run from a release build"));
}

#[test]
fn mac_benchmark_accepts_context_profile_alias_before_release_gate() {
    bitnet()
        .args(["mac", "benchmark", "--profile", "context"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac benchmark must be run from a release build"));
}

#[test]
fn mac_benchmark_accepts_repeat_flag_before_release_gate() {
    bitnet()
        .args(["mac", "benchmark", "--profile", "short_prompt_16_out", "--repeat", "2"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac benchmark must be run from a release build"));
}

#[test]
fn mac_benchmark_rejects_zero_repeat_before_release_gate() {
    bitnet()
        .args(["mac", "benchmark", "--profile", "short_prompt_16_out", "--repeat", "0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mac benchmark --repeat must be at least 1"));
}

#[test]
fn mac_benchmark_calibrate_writes_synthetic_receipt() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt = dir.path().join("benchmark-calibration.json");
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "benchmark", "--calibrate", "--json-out", receipt_str.as_str()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Mac benchmark calibration written"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "apple_m4_benchmark_calibration");
    assert_eq!(receipt_json["operator_command"], "mac benchmark --calibrate");
    assert_eq!(receipt_json["run_identity"]["contract_version"], "m4-run-identity-v1");
    assert_eq!(receipt_json["run_identity"]["command"]["live_model_run"], false);
    assert_eq!(receipt_json["calibration"]["live_model_run"], false);
    assert_eq!(receipt_json["calibration"]["model_inference_timing"], false);
    assert_eq!(receipt_json["calibration"]["clock"]["source"], "std::time::Instant");
    assert_eq!(receipt_json["claim_boundary"]["benchmark_calibration_only"], true);
    assert_eq!(receipt_json["claim_boundary"]["broad_performance_claim"], false);

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_benchmark_calibration"))
        .stdout(predicate::str::contains("\"prompt_count\": 0"));
    Ok(())
}

#[test]
fn mac_benchmark_calibrate_rejects_profile_combo() {
    bitnet()
        .args(["mac", "benchmark", "--calibrate", "--profile", "short_prompt_16_out"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "mac benchmark --calibrate cannot be combined with --profile",
        ));
}

#[test]
fn mac_benchmark_calibrate_rejects_repeat_combo() {
    bitnet().args(["mac", "benchmark", "--calibrate", "--repeat", "2"]).assert().failure().stderr(
        predicate::str::contains("mac benchmark --calibrate cannot be combined with --repeat"),
    );
}

#[test]
fn mac_benchmark_calibration_receipt_rejects_missing_clock()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt = dir.path().join("benchmark-calibration.json");
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "benchmark", "--calibrate", "--json-out", receipt_str.as_str()])
        .assert()
        .success();

    let mut receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    receipt_json["calibration"]
        .as_object_mut()
        .ok_or_else(|| std::io::Error::other("missing calibration"))?
        .remove("clock");
    std::fs::write(&receipt, serde_json::to_vec_pretty(&receipt_json)?)?;

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("clock"));
    Ok(())
}

#[test]
fn mac_benchmark_preflight_writes_receipt() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt = dir.path().join("benchmark-preflight.json");
    let cache_dir = dir.path().join("cache");
    let receipt_str = receipt.to_string_lossy().into_owned();
    let cache_str = cache_dir.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "benchmark-preflight",
            "--cache-dir",
            cache_str.as_str(),
            "--background-load-note",
            "test harness idle",
            "--json-out",
            receipt_str.as_str(),
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"artifact_kind\": \"apple_m4_benchmark_preflight\""))
        .stdout(predicate::str::contains("\"timing_result_recorded\": false"));

    let receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    assert_eq!(receipt_json["artifact_kind"], "apple_m4_benchmark_preflight");
    assert_eq!(receipt_json["run_identity"]["contract_version"], "m4-run-identity-v1");
    assert_eq!(receipt_json["run_identity"]["command"]["live_model_run"], false);
    assert_eq!(receipt_json["benchmark_preflight"]["live_model_run"], false);
    assert_eq!(receipt_json["claim_boundary"]["broad_performance_claim"], false);

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_benchmark_preflight"))
        .stdout(predicate::str::contains("\"prompt_count\": 0"));
    Ok(())
}

#[test]
fn mac_benchmark_preflight_receipt_rejects_missing_invalid_reasons()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt = dir.path().join("benchmark-preflight.json");
    let cache_dir = dir.path().join("cache");
    let receipt_str = receipt.to_string_lossy().into_owned();
    let cache_str = cache_dir.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "benchmark-preflight",
            "--cache-dir",
            cache_str.as_str(),
            "--json-out",
            receipt_str.as_str(),
        ])
        .assert()
        .success();

    let mut receipt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
    receipt_json["comparison_readiness"]
        .as_object_mut()
        .ok_or_else(|| std::io::Error::other("missing comparison_readiness"))?
        .remove("invalid_comparison_reasons");
    std::fs::write(&receipt, serde_json::to_vec_pretty(&receipt_json)?)?;

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid_comparison_reasons"));
    Ok(())
}

#[test]
fn mac_receipts_check_accepts_slm_benchmark_v2_summary() {
    let dir = tempfile::tempdir().expect("tempdir");
    let receipt_path = dir.path().join("slm-benchmark-v2.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": "1.1.0",
            "artifact_kind": "apple_m4_slm_benchmark_v2",
            "requested_backend": "apple-m4-cpu-neon",
            "selected_backend": "apple-m4-cpu-neon",
            "runtime_api": "cpu",
            "fallback_used": false,
            "profile_set": "slm-benchmark-v2",
            "profiles_required": ["short_prompt_16_out"],
            "build": {
                "profile": "release",
                "release_mode": true
            },
            "model_cache": {
                "id": "qwen2.5-0.5b-instruct-q8_0",
                "sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
                "architecture": "qwen2",
                "quantization": "Q8_0",
                "tokenizer_pre": "qwen2"
            },
            "profiles": [
                benchmark_profile_v2_json("short_prompt_16_out")
            ],
            "prompt_count": 3,
            "generated_tokens": 48,
            "speed": benchmark_speed_v2_json(),
            "benchmark_contract": benchmark_contract_v2_json(),
            "memory": {
                "peak_memory_mb_p50": 3900.0,
                "peak_memory_mb_p90": 3950.0,
                "peak_memory_mb_p99": 3975.0,
                "memory_drift_mb_p50": 0.0,
                "memory_drift_mb_p90": 12.0,
                "memory_drift_mb_p99": 24.0,
                "source": "getrusage.ru_maxrss process peak delta"
            },
            "evidence": {
                "profile_receipts": ["ci/hardware/apple-m4-mac-mini/2026-05-14/slm-benchmark-v2/qwen/summary-profiles/short_prompt_16_out.json"],
                "generated_text_recorded": true,
                "generated_token_ids_recorded": true,
                "operator_command": "mac benchmark"
            },
            "mac_claim_boundary": {
                "dense_slm_only": true,
                "bounded_benchmark_profiles_only": true,
                "broad_model_quality_claim": false,
                "broad_performance_claim": false,
                "speedup_claim": false,
                "bitnet_quality_claimed": false,
                "full_metal_inference_claimed": false,
                "mpsgraph_inference_claimed": false,
                "neural_engine_execution_claimed": false,
                "qk256_apple_claimed": false,
                "macbook_evidence": false
            },
            "speedup_claim": false
        }))
        .expect("json"),
    )
    .expect("write receipt");
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_slm_benchmark_v2"))
        .stdout(predicate::str::contains("\"prompt_count\": 3"));
}

#[test]
fn mac_receipts_check_accepts_slm_benchmark_v2_resident_100_profile()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("slm-benchmark-v2-resident-100.json");
    let mut receipt = slm_benchmark_v2_summary();
    receipt["profiles_required"] = serde_json::json!(["resident_100"]);
    receipt["profiles"] = serde_json::json!([benchmark_profile_v2_json("resident_100")]);
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_slm_benchmark_v2"))
        .stdout(predicate::str::contains("\"prompt_count\": 3"));
    Ok(())
}

#[test]
fn mac_benchmark_receipt_contract_rejects_missing_sampling_overhead()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("slm-benchmark-v2-missing-sampling.json");
    let mut receipt = slm_benchmark_v2_summary();
    receipt["speed"]
        .as_object_mut()
        .ok_or_else(|| std::io::Error::other("missing speed object"))?
        .remove("sampling_ms_per_token_p50");
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("sampling_ms_per_token_p50"));
    Ok(())
}

#[test]
fn mac_benchmark_receipt_contract_rejects_malformed_timing_percentiles()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("slm-benchmark-v2-bad-timing.json");
    let mut receipt = slm_benchmark_v2_summary();
    receipt["profiles"][0]["timing"]["prefill_ms"]["p90"] = serde_json::json!(100.0);
    receipt["profiles"][0]["timing"]["prefill_ms"]["p50"] = serde_json::json!(650.0);
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("p50 <= p90 <= p99"));
    Ok(())
}

#[test]
fn mac_benchmark_receipt_contract_rejects_profiles_required_mismatch()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("slm-benchmark-v2-profile-mismatch.json");
    let mut receipt = slm_benchmark_v2_summary();
    receipt["profiles_required"] = serde_json::json!(["resident_100"]);
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("profiles_required must match profiles order"));
    Ok(())
}

#[test]
fn mac_receipts_check_accepts_benchmark_variance_v1() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("benchmark-variance.json");
    let mut receipt = slm_benchmark_variance_v1_summary();
    receipt["evidence"]["child_summary_receipts"] =
        serde_json::json!(write_benchmark_variance_child_summaries(dir.path())?);
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_benchmark_variance_v1"))
        .stdout(predicate::str::contains("\"prompt_count\": 6"));
    Ok(())
}

#[test]
fn mac_benchmark_variance_receipt_rejects_missing_outlier_policy()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("benchmark-variance-missing-outlier.json");
    let mut receipt = slm_benchmark_variance_v1_summary();
    receipt["evidence"]["child_summary_receipts"] =
        serde_json::json!(write_benchmark_variance_child_summaries(dir.path())?);
    receipt
        .as_object_mut()
        .ok_or_else(|| std::io::Error::other("missing receipt object"))?
        .remove("outlier_handling");
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("outlier_handling"));
    Ok(())
}

#[test]
fn mac_benchmark_variance_receipt_rejects_profile_count_mismatch()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("benchmark-variance-profile-count.json");
    let mut receipt = slm_benchmark_variance_v1_summary();
    receipt["evidence"]["child_summary_receipts"] =
        serde_json::json!(write_benchmark_variance_child_summaries(dir.path())?);
    receipt["repeat"]["profile_count"] = serde_json::json!(2);
    receipt["repeat"]["sample_count"] = serde_json::json!(4);
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("profile_count must match profiles_required length"));
    Ok(())
}

#[test]
fn mac_benchmark_variance_receipt_rejects_metric_count_mismatch()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("benchmark-variance-metric-count.json");
    let mut receipt = slm_benchmark_variance_v1_summary();
    receipt["evidence"]["child_summary_receipts"] =
        serde_json::json!(write_benchmark_variance_child_summaries(dir.path())?);
    receipt["metrics"]["speed"]["ttft_ms_p50"]["count"] = serde_json::json!(1);
    receipt["metrics"]["speed"]["ttft_ms_p50"]["samples"] = serde_json::json!([10.0]);
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("count must equal completed repeats"));
    Ok(())
}

#[test]
fn mac_benchmark_variance_receipt_rejects_missing_child_summary()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("benchmark-variance-missing-child.json");
    let mut receipt = slm_benchmark_variance_v1_summary();
    receipt["evidence"]["child_summary_receipts"] = serde_json::json!([
        dir.path().join("missing-run-01.json").to_string_lossy(),
        dir.path().join("missing-run-02.json").to_string_lossy(),
    ]);
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("failed to read child benchmark receipt"));
    Ok(())
}

#[test]
fn mac_benchmark_variance_receipt_rejects_child_model_mismatch()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("benchmark-variance-child-model.json");
    let child_paths = write_benchmark_variance_child_summaries(dir.path())?;
    let mut child = slm_benchmark_v2_summary();
    child["model_cache"]["sha256"] =
        serde_json::json!("0000000000000000000000000000000000000000000000000000000000000000");
    std::fs::write(dir.path().join("run-02.json"), serde_json::to_vec_pretty(&child)?)?;
    let mut receipt = slm_benchmark_variance_v1_summary();
    receipt["evidence"]["child_summary_receipts"] = serde_json::json!(child_paths);
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("model_cache.sha256"));
    Ok(())
}

#[test]
fn mac_receipts_check_rejects_performance_profile_missing_warm_128() {
    let dir = tempfile::tempdir().expect("tempdir");
    let receipt_path = dir.path().join("bad-performance-profiles.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "artifact_kind": "apple_m4_slm_performance_profiles",
            "requested_backend": "apple-m4-cpu-neon",
            "selected_backend": "apple-m4-cpu-neon",
            "runtime_api": "cpu",
            "fallback_used": false,
            "profile_set": "performance",
            "build": {
                "profile": "release",
                "release_mode": true
            },
            "operator_thresholds": {
                "cold_load_separated": true,
                "model_tokenizer_reuse_visible": true,
                "model_tokenizer_reuse_visible_per_profile": true,
                "profiles_loaded_independently": true,
                "profile_set_model_loads": 3,
                "reuse_scope": "within_each_profile",
                "profiles_required": ["warm_16", "warm_32", "warm_64"],
                "thresholds_are_claim_bounds_not_speed_guarantees": true
            },
            "performance_baseline": {
                "release_mode_required": true,
                "release_mode_observed": true,
                "warm_128_included": false,
                "broad_performance_claim": false,
                "speedup_claim": false
            },
            "profiles": [
                performance_profile_json("warm_16", 16, 8000.0, 5000.0),
                performance_profile_json("warm_32", 32, 16000.0, 10000.0),
                performance_profile_json("warm_64", 64, 32000.0, 20000.0)
            ],
            "mac_claim_boundary": {
                "full_metal_inference_claimed": false,
                "neural_engine_execution_claimed": false,
                "qk256_apple_claimed": false,
                "bitnet_quality_claimed": false,
                "broad_performance_claim": false,
                "speedup_claim": false
            }
        }))
        .expect("json"),
    )
    .expect("write receipt");
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet().args(["mac", "receipts-check", receipt_str.as_str()]).assert().failure().stderr(
        predicate::str::contains(
            "profile summary must contain exactly warm_16, warm_32, warm_64, warm_128",
        ),
    );
}

#[test]
fn mac_receipts_check_reports_dense_slm_regression_no_warnings() {
    let dir = tempfile::tempdir().expect("tempdir");
    let baseline_path = dir.path().join("baseline.json");
    let current_path = dir.path().join("current.json");
    let baseline = performance_summary_receipt("same-sha");
    let current = performance_summary_receipt("same-sha");
    std::fs::write(&baseline_path, serde_json::to_vec_pretty(&baseline).expect("json"))
        .expect("write baseline");
    std::fs::write(&current_path, serde_json::to_vec_pretty(&current).expect("json"))
        .expect("write current");
    let baseline_str = baseline_path.to_string_lossy().into_owned();
    let current_str = current_path.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "receipts-check",
            current_str.as_str(),
            "--regression-baseline",
            baseline_str.as_str(),
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"warning_count\": 0"))
        .stdout(predicate::str::contains("\"advisory\": true"));
}

#[test]
fn mac_receipts_check_reports_dense_slm_regression_advisory_warning() {
    let dir = tempfile::tempdir().expect("tempdir");
    let baseline_path = dir.path().join("baseline.json");
    let current_path = dir.path().join("current.json");
    let baseline = performance_summary_receipt("same-sha");
    let mut current = performance_summary_receipt("same-sha");
    current["profiles"][0]["timing"]["decode_generated_tok_s"] = serde_json::json!(1.0);
    std::fs::write(&baseline_path, serde_json::to_vec_pretty(&baseline).expect("json"))
        .expect("write baseline");
    std::fs::write(&current_path, serde_json::to_vec_pretty(&current).expect("json"))
        .expect("write current");
    let baseline_str = baseline_path.to_string_lossy().into_owned();
    let current_str = current_path.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "receipts-check",
            current_str.as_str(),
            "--regression-baseline",
            baseline_str.as_str(),
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"warning_count\": 1"))
        .stdout(predicate::str::contains("timing.decode_generated_tok_s"));
}

#[test]
fn mac_receipts_check_uses_tightened_dense_slm_regression_thresholds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let baseline_path = dir.path().join("baseline.json");
    let current_path = dir.path().join("current.json");
    let baseline = performance_summary_receipt("same-sha");
    let mut current = performance_summary_receipt("same-sha");
    // Baseline fixture decode tok/s is 3.2. A 14% regression is above the
    // tightened 12.5% advisory band but below the original 20% band.
    current["profiles"][0]["timing"]["decode_generated_tok_s"] = serde_json::json!(2.752);
    std::fs::write(&baseline_path, serde_json::to_vec_pretty(&baseline).expect("json"))
        .expect("write baseline");
    std::fs::write(&current_path, serde_json::to_vec_pretty(&current).expect("json"))
        .expect("write current");
    let baseline_str = baseline_path.to_string_lossy().into_owned();
    let current_str = current_path.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "receipts-check",
            current_str.as_str(),
            "--regression-baseline",
            baseline_str.as_str(),
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"warning_count\": 1"))
        .stdout(predicate::str::contains("\"threshold_percent\": 12.5"));
}

#[test]
fn mac_receipts_check_rejects_dense_slm_regression_context_mismatch() {
    let dir = tempfile::tempdir().expect("tempdir");
    let baseline_path = dir.path().join("baseline.json");
    let current_path = dir.path().join("current.json");
    let baseline = performance_summary_receipt("baseline-sha");
    let current = performance_summary_receipt("different-sha");
    std::fs::write(&baseline_path, serde_json::to_vec_pretty(&baseline).expect("json"))
        .expect("write baseline");
    std::fs::write(&current_path, serde_json::to_vec_pretty(&current).expect("json"))
        .expect("write current");
    let baseline_str = baseline_path.to_string_lossy().into_owned();
    let current_str = current_path.to_string_lossy().into_owned();

    bitnet()
        .args([
            "mac",
            "receipts-check",
            current_str.as_str(),
            "--regression-baseline",
            baseline_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("model_cache.sha256 mismatch"));
}

#[test]
fn mac_receipts_check_accepts_dense_slm_quality_corpus_gate()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("warm-quality.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&dense_quality_warm_session_receipt())?,
    )?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("slm_apple_m4_warm_session"))
        .stdout(predicate::str::contains("\"prompt_count\": 14"));
    Ok(())
}

#[test]
fn mac_receipts_check_rejects_warm_session_missing_generated_token_ids()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("warm-missing-generated-ids.json");
    let mut receipt = dense_quality_warm_session_receipt();
    receipt["prompts"][0]
        .as_object_mut()
        .ok_or("prompt receipt must be an object")?
        .remove("generated_token_ids");
    receipt["prompts"][0]
        .as_object_mut()
        .ok_or("prompt receipt must be an object")?
        .remove("tokens");
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("generated token IDs"));
    Ok(())
}

#[test]
fn slm_eval_report_schema_accepts_fixture_summary() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("slm-eval-summary.json");
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&slm_eval_summary_report())?)?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_slm_eval_summary"))
        .stdout(predicate::str::contains("\"prompt_count\": 10"))
        .stdout(predicate::str::contains("\"generated_tokens\": 128"))
        .stdout(predicate::str::contains("\"passed\": true"));
    Ok(())
}

#[test]
fn slm_eval_report_schema_rejects_missing_tokenizer_identity()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("slm-eval-summary-missing-tokenizer.json");
    let mut receipt = slm_eval_summary_report();
    receipt["tokenizer"]
        .as_object_mut()
        .ok_or_else(|| std::io::Error::other("tokenizer object missing"))?
        .remove("authority");
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("tokenizer.authority"));
    Ok(())
}

#[test]
fn slm_eval_report_schema_rejects_missing_input_throughput()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("slm-eval-summary-missing-throughput.json");
    let mut receipt = slm_eval_summary_report();
    receipt["speed"]
        .as_object_mut()
        .ok_or_else(|| std::io::Error::other("speed object missing"))?
        .remove("input_tok_s_p50");
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("speed.input_tok_s_p50"));
    Ok(())
}

#[test]
fn slm_eval_report_schema_rejects_broad_quality_claim() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt_path = dir.path().join("slm-eval-summary-broad-claim.json");
    let mut receipt = slm_eval_summary_report();
    receipt["claim_boundary"]["broad_model_quality_claim"] = serde_json::json!(true);
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("claim_boundary.broad_model_quality_claim"));
    Ok(())
}

#[test]
fn mac_receipts_check_rejects_dense_slm_quality_corpus_determinism_drift() {
    let dir = tempfile::tempdir().expect("tempdir");
    let receipt_path = dir.path().join("warm-quality-drift.json");
    let mut receipt = dense_quality_warm_session_receipt();
    receipt["prompts"][1]["generated_token_ids"] = serde_json::json!([198, 999]);
    receipt["prompts"][1]["generated_tokens"] = serde_json::json!(2);
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt).expect("json"))
        .expect("write receipt");
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("changed deterministic greedy output"));
}

#[test]
fn mac_receipts_check_rejects_dense_slm_quality_corpus_degenerate_prompt() {
    let dir = tempfile::tempdir().expect("tempdir");
    let receipt_path = dir.path().join("warm-quality-degenerate.json");
    let mut receipt = dense_quality_warm_session_receipt();
    receipt["prompts"][0]["quality"]["passed"] = serde_json::json!(false);
    receipt["prompts"][0]["quality"]["non_degenerate"] = serde_json::json!(false);
    receipt["prompts"][0]["quality"]["failed_rules"] =
        serde_json::json!(["generated_token_variation"]);
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt).expect("json"))
        .expect("write receipt");
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("warm-session prompt quality failed"));
}

#[test]
fn mac_receipts_check_accepts_split_metal_phase_receipt() {
    let dir = tempfile::tempdir().expect("tempdir");
    let receipt_path = dir.path().join("metal-phase.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "artifact_kind": "phase_contribution",
            "requested_backend": "apple-m4-metal",
            "selected_backend": "apple-m4-metal",
            "runtime_api": "metal",
            "fallback_used": false,
            "kernel_id": "tiny_metal_dense_prefill_linear_projection",
            "slm_pipeline": {
                "requested_backend": "apple-m4-cpu-neon",
                "selected_backend": "apple-m4-cpu-neon",
                "runtime_api": "cpu",
                "cpu_pipeline_for_remaining_phases": true,
                "full_inference_exercised": false
            },
            "metal_phase": {
                "requested_backend": "apple-m4-metal",
                "selected_backend": "apple-m4-metal",
                "runtime_api": "metal",
                "fallback_used": false,
                "kernel_id": "tiny_metal_dense_prefill_linear_projection",
                "kernel_family": "dense_f32",
                "execution_phase": "prefill_linear_projection",
                "timing_recorded": true,
                "full_metal_inference": false,
                "full_autoregressive_decode": false
            },
            "layout": {
                "source": "fixture_dense_f32_row_major",
                "transport_layout": "row_major_f32",
                "consumes_dense_f32_directly": true,
                "dequantizes_before_compute": false,
                "batch_size": 2,
                "in_features": 8,
                "out_features": 6
            },
            "parity": {
                "reference_backend": "apple-m4-cpu-neon",
                "target_backend": "apple-m4-metal",
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
                "cpu_reference_token_id": 3,
                "metal_phase_token_id": 3,
                "greedy_token_ids_match_cpu_reference": true
            },
            "timing": {
                "scope": "single_live_phase_dispatch_readback_vs_cpu_reference_fixture",
                "cpu_reference_ms": 0.125,
                "metal_phase_ms": 0.5,
                "timing_delta_ms": 0.375,
                "speedup_claim": false
            },
            "claim_boundary": {
                "phase_contribution_only": true,
                "full_metal_inference_claimed": false,
                "neural_engine_execution_claimed": false,
                "mpsgraph_inference_claimed": false,
                "qk256_apple_claimed": false,
                "bitnet_quality_claimed": false,
                "broad_performance_claim": false,
                "speedup_claim": false
            }
        }))
        .expect("json"),
    )
    .expect("write receipt");
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("phase_contribution"))
        .stdout(predicate::str::contains("apple-m4-metal"));
}

#[test]
fn mac_receipts_check_accepts_dense_qkv_metal_phase_receipt() {
    let dir = tempfile::tempdir().expect("tempdir");
    let receipt_path = dir.path().join("metal-qkv-phase.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "artifact_kind": "phase_contribution",
            "requested_backend": "apple-m4-metal",
            "selected_backend": "apple-m4-metal",
            "runtime_api": "metal",
            "fallback_used": false,
            "kernel_id": "tiny_metal_dense_prefill_qkv_projection",
            "resolved_device": {
                "chip": "Apple M4",
                "unified_memory": true
            },
            "slm_pipeline": {
                "requested_backend": "apple-m4-cpu-neon",
                "selected_backend": "apple-m4-cpu-neon",
                "runtime_api": "cpu",
                "remaining_phases_backend": "apple-m4-cpu-neon",
                "cpu_pipeline_for_remaining_phases": true,
                "full_inference_exercised": false
            },
            "metal_phase": {
                "requested_backend": "apple-m4-metal",
                "selected_backend": "apple-m4-metal",
                "runtime_api": "metal",
                "fallback_used": false,
                "kernel_id": "tiny_metal_dense_prefill_qkv_projection",
                "kernel_family": "dense_f32",
                "execution_phase": "prefill_qkv_projection",
                "phase_scope": "qwen2_5_dense_prefill_qkv_projection_fixture",
                "prefill_tokens": 2,
                "kv_cache_behavior": "not_exercised",
                "timing_recorded": true,
                "full_metal_inference": false,
                "full_autoregressive_decode": false
            },
            "dimensions": {
                "hidden_size": 896,
                "attention_heads": 14,
                "kv_heads": 2,
                "head_dim": 64,
                "q_dim": 896,
                "kv_dim": 128,
                "q_shape": [2, 896],
                "k_shape": [2, 128],
                "v_shape": [2, 128]
            },
            "layout": {
                "source": "fixture_dense_f32_row_major",
                "transport_layout": "row_major_f32",
                "activation_layout": "row_major_f32",
                "weight_layout": "row_major_f32_out_features_by_in_features",
                "bias_layout": "concatenated_row_major_f32_q_k_v",
                "output_layout": "concatenated_row_major_f32_q_k_v",
                "consumes_dense_f32_directly": true,
                "dequantizes_before_compute": false,
                "activation_elements": 1792,
                "q_weight_elements": 802816,
                "k_weight_elements": 114688,
                "v_weight_elements": 114688
            },
            "parity": {
                "reference_backend": "apple-m4-cpu-neon",
                "target_backend": "apple-m4-metal",
                "kernel_family": "dense_f32",
                "kernel_id": "tiny_metal_dense_prefill_qkv_projection",
                "q_matches_cpu_reference": true,
                "k_matches_cpu_reference": true,
                "v_matches_cpu_reference": true,
                "q_max_abs_error": 0.0,
                "q_mean_abs_error": 0.0,
                "k_max_abs_error": 0.0,
                "k_mean_abs_error": 0.0,
                "v_max_abs_error": 0.0,
                "v_mean_abs_error": 0.0,
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0
            },
            "timing": {
                "scope": "single_live_qkv_phase_dispatch_readback_vs_cpu_reference_fixture",
                "cpu_reference_ms": 12.0,
                "metal_phase_ms": 661.0,
                "timing_delta_ms": 649.0,
                "dispatch_readback_ms": 661.0,
                "speedup_claim": false
            },
            "claim_boundary": {
                "phase_contribution_only": true,
                "full_metal_inference_claimed": false,
                "neural_engine_execution_claimed": false,
                "mpsgraph_inference_claimed": false,
                "qk256_apple_claimed": false,
                "bitnet_quality_claimed": false,
                "broad_performance_claim": false,
                "speedup_claim": false
            }
        }))
        .expect("json"),
    )
    .expect("write receipt");
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("phase_contribution"))
        .stdout(predicate::str::contains("apple-m4-metal"));
}

#[test]
fn mac_receipts_check_rejects_dense_qkv_phase_without_component_parity() {
    let dir = tempfile::tempdir().expect("tempdir");
    let receipt_path = dir.path().join("metal-qkv-phase-bad.json");
    let mut receipt = serde_json::json!({
        "artifact_kind": "phase_contribution",
        "requested_backend": "apple-m4-metal",
        "selected_backend": "apple-m4-metal",
        "runtime_api": "metal",
        "fallback_used": false,
        "kernel_id": "tiny_metal_dense_prefill_qkv_projection",
        "slm_pipeline": {
            "selected_backend": "apple-m4-cpu-neon",
            "runtime_api": "cpu",
            "cpu_pipeline_for_remaining_phases": true
        },
        "metal_phase": {
            "selected_backend": "apple-m4-metal",
            "runtime_api": "metal",
            "fallback_used": false,
            "kernel_id": "tiny_metal_dense_prefill_qkv_projection",
            "execution_phase": "prefill_qkv_projection",
            "prefill_tokens": 2,
            "full_metal_inference": false,
            "timing_recorded": true
        },
        "dimensions": {
            "hidden_size": 896,
            "attention_heads": 14,
            "kv_heads": 2,
            "head_dim": 64,
            "q_dim": 896,
            "kv_dim": 128,
            "q_shape": [2, 896],
            "k_shape": [2, 128],
            "v_shape": [2, 128]
        },
        "layout": {
            "consumes_dense_f32_directly": true,
            "dequantizes_before_compute": false,
            "bias_layout": "concatenated_row_major_f32_q_k_v",
            "output_layout": "concatenated_row_major_f32_q_k_v",
            "activation_elements": 1792,
            "q_weight_elements": 802816,
            "k_weight_elements": 114688,
            "v_weight_elements": 114688
        },
        "parity": {
            "reference_backend": "apple-m4-cpu-neon",
            "target_backend": "apple-m4-metal",
            "q_matches_cpu_reference": true,
            "k_matches_cpu_reference": true,
            "v_matches_cpu_reference": true,
            "q_max_abs_error": 0.0,
            "q_mean_abs_error": 0.0,
            "k_max_abs_error": 0.0,
            "k_mean_abs_error": 0.0,
            "v_max_abs_error": 0.0,
            "v_mean_abs_error": 0.0,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0
        },
        "timing": {
            "scope": "single_live_qkv_phase_dispatch_readback_vs_cpu_reference_fixture",
            "cpu_reference_ms": 12.0,
            "metal_phase_ms": 661.0,
            "timing_delta_ms": 649.0,
            "speedup_claim": false
        }
    });
    receipt["parity"]["v_matches_cpu_reference"] = serde_json::json!(false);
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt).expect("json"))
        .expect("write receipt");
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must record v_matches_cpu_reference=true"));
}

#[test]
fn mac_receipts_check_rejects_metal_phase_without_timing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let receipt_path = dir.path().join("metal-phase-missing-timing.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "artifact_kind": "phase_contribution",
            "requested_backend": "apple-m4-metal",
            "selected_backend": "apple-m4-metal",
            "runtime_api": "metal",
            "fallback_used": false,
            "kernel_id": "tiny_metal_dense_prefill_linear_projection",
            "slm_pipeline": {
                "selected_backend": "apple-m4-cpu-neon",
                "runtime_api": "cpu",
                "cpu_pipeline_for_remaining_phases": true
            },
            "metal_phase": {
                "selected_backend": "apple-m4-metal",
                "runtime_api": "metal",
                "fallback_used": false,
                "kernel_id": "tiny_metal_dense_prefill_linear_projection",
                "execution_phase": "prefill_linear_projection",
                "full_metal_inference": false
            },
            "layout": {
                "consumes_dense_f32_directly": true,
                "dequantizes_before_compute": false,
                "batch_size": 2,
                "in_features": 8,
                "out_features": 6
            },
            "parity": {
                "reference_backend": "apple-m4-cpu-neon",
                "target_backend": "apple-m4-metal",
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
                "greedy_token_ids_match_cpu_reference": true
            }
        }))
        .expect("json"),
    )
    .expect("write receipt");
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must record phase timing"));
}

#[test]
fn mac_receipts_check_rejects_metal_phase_with_full_inference_claim() {
    let dir = tempfile::tempdir().expect("tempdir");
    let receipt_path = dir.path().join("bad-metal-phase.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "artifact_kind": "phase_contribution",
            "requested_backend": "apple-m4-metal",
            "selected_backend": "apple-m4-metal",
            "runtime_api": "metal",
            "fallback_used": false,
            "slm_pipeline": {
                "selected_backend": "apple-m4-cpu-neon",
                "runtime_api": "cpu",
                "cpu_pipeline_for_remaining_phases": true
            },
            "metal_phase": {
                "selected_backend": "apple-m4-metal",
                "runtime_api": "metal",
                "fallback_used": false,
                "kernel_id": "tiny_metal_dense_prefill_linear_projection",
                "execution_phase": "prefill_linear_projection",
                "full_metal_inference": true
            },
            "layout": {
                "consumes_dense_f32_directly": true,
                "dequantizes_before_compute": false,
                "batch_size": 2,
                "in_features": 8,
                "out_features": 6
            },
            "parity": {
                "reference_backend": "apple-m4-cpu-neon",
                "target_backend": "apple-m4-metal",
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
                "greedy_token_ids_match_cpu_reference": true
            }
        }))
        .expect("json"),
    )
    .expect("write receipt");
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("claims full apple-m4-metal inference"));
}

#[test]
fn mac_receipts_check_rejects_operator_profile_missing_profile() {
    let dir = tempfile::tempdir().expect("tempdir");
    let receipt_path = dir.path().join("operator-profiles.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "artifact_kind": "apple_m4_slm_operator_profiles",
            "requested_backend": "apple-m4-cpu-neon",
            "selected_backend": "apple-m4-cpu-neon",
            "runtime_api": "cpu",
            "fallback_used": false,
            "operator_thresholds": {
                "cold_load_separated": true,
                "model_tokenizer_reuse_visible": true,
                "model_tokenizer_reuse_visible_per_profile": true,
                "profiles_loaded_independently": true,
                "profile_set_model_loads": 3,
                "profiles_required": ["warm_16", "warm_32", "warm_64"],
                "thresholds_are_claim_bounds_not_speed_guarantees": true
            },
            "profiles": [
                {
                    "profile_id": "warm_16",
                    "requested_max_new_tokens": 16,
                    "prompt_count": 1,
                    "generated_tokens": 16,
                    "quality_passed": true,
                    "cold_load_separated": true,
                    "model_loaded_once": true,
                    "tokenizer_loaded_once": true,
                    "reuse_scope": "within_profile",
                    "timing": {
                        "model_load_ms": 1000.0,
                        "tokenizer_load_ms": 25.0,
                        "warm_prompt_wall_ms": 8000.0,
                        "decode_total_ms": 5000.0,
                        "sampling_ms": 12.0,
                        "warm_prompt_generated_tok_s": 2.0,
                        "decode_generated_tok_s": 3.2
                    }
                }
            ],
            "mac_claim_boundary": {
                "full_metal_inference_claimed": false,
                "neural_engine_execution_claimed": false,
                "qk256_apple_claimed": false,
                "bitnet_quality_claimed": false,
                "broad_performance_claim": false,
                "speedup_claim": false
            }
        }))
        .expect("json"),
    )
    .expect("write receipt");
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet().args(["mac", "receipts-check", receipt_str.as_str()]).assert().failure().stderr(
        predicate::str::contains("profile summary must contain exactly warm_16, warm_32, warm_64"),
    );
}

fn performance_profile_json(
    profile_id: &str,
    requested_max_new_tokens: u64,
    warm_prompt_wall_ms: f64,
    decode_total_ms: f64,
) -> serde_json::Value {
    serde_json::json!({
        "profile_id": profile_id,
        "requested_max_new_tokens": requested_max_new_tokens,
        "prompt_count": 1,
        "generated_tokens": requested_max_new_tokens,
        "quality_passed": true,
        "cold_load_separated": true,
        "model_loaded_once": true,
        "tokenizer_loaded_once": true,
        "reuse_scope": "within_profile",
        "resident_session": resident_session_json(),
        "timing": {
            "model_load_ms": 1000.0,
            "tokenizer_load_ms": 25.0,
            "total_session_ms": warm_prompt_wall_ms + 1025.0,
            "tokenize_ms": 20.0,
            "prefill_ms": 100.0,
            "warm_prompt_wall_ms": warm_prompt_wall_ms,
            "first_token_ms": [100.0],
            "time_to_first_token_ms": [100.0],
            "decode_total_ms": decode_total_ms,
            "sampling_ms": 12.0,
            "warm_prompt_generated_tok_s": 2.0,
            "decode_generated_tok_s": 3.2
        },
        "memory": {
            "peak_memory_mb": 512.0,
            "peak_memory_source": "getrusage.ru_maxrss"
        },
        "allocation_audit": {
            "enabled": true,
            "scope": "selected Apple M4 CPU/NEON SLM warm-session prompt hot path",
            "ranked_hotspots": [
                {"component": "model.forward", "alloc_count": 10, "alloc_bytes": 1024}
            ]
        }
    })
}

fn benchmark_stat_v2_json(p50: f64, p90: f64, p99: f64) -> serde_json::Value {
    serde_json::json!({
        "count": 3,
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "min": p50,
        "max": p99,
        "samples": [p50, p90, p99]
    })
}

fn benchmark_variance_stat_json(first: f64, second: f64) -> serde_json::Value {
    serde_json::json!({
        "count": 2,
        "p50": first,
        "p90": second,
        "p99": second,
        "min": first,
        "max": second,
        "samples": [first, second]
    })
}

fn benchmark_variance_metric_section(metrics: &[&str]) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    for metric in metrics {
        for percentile in ["p50", "p90", "p99"] {
            object
                .insert(format!("{metric}_{percentile}"), benchmark_variance_stat_json(10.0, 12.0));
        }
    }
    serde_json::Value::Object(object)
}

fn write_benchmark_variance_child_summaries(
    dir: &std::path::Path,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut paths = Vec::new();
    for run_number in 1..=2 {
        let path = dir.join(format!("run-{run_number:02}.json"));
        std::fs::write(&path, serde_json::to_vec_pretty(&slm_benchmark_v2_summary())?)?;
        paths.push(path.to_string_lossy().into_owned());
    }
    Ok(paths)
}

fn benchmark_speed_v2_json() -> serde_json::Value {
    serde_json::json!({
        "cold_load_ms_p50": 3200.0,
        "cold_load_ms_p90": 3300.0,
        "cold_load_ms_p99": 3400.0,
        "tokenizer_load_ms_p50": 45.0,
        "tokenizer_load_ms_p90": 50.0,
        "tokenizer_load_ms_p99": 55.0,
        "prompt_tokenize_ms_p50": 4.0,
        "prompt_tokenize_ms_p90": 5.0,
        "prompt_tokenize_ms_p99": 6.0,
        "prefill_ms_p50": 650.0,
        "prefill_ms_p90": 720.0,
        "prefill_ms_p99": 800.0,
        "ttft_ms_p50": 1800.0,
        "ttft_ms_p90": 2100.0,
        "ttft_ms_p99": 2300.0,
        "sampling_ms_per_token_p50": 0.2,
        "sampling_ms_per_token_p90": 0.3,
        "sampling_ms_per_token_p99": 0.4,
        "input_tok_s_p50": 118.0,
        "input_tok_s_p90": 130.0,
        "input_tok_s_p99": 140.0,
        "output_tok_s_p50": 15.2,
        "output_tok_s_p90": 16.0,
        "output_tok_s_p99": 16.8,
        "decode_tok_s_p50": 15.5,
        "decode_tok_s_p90": 16.2,
        "decode_tok_s_p99": 17.0,
        "total_wall_ms_p50": 5200.0,
        "total_wall_ms_p90": 5600.0,
        "total_wall_ms_p99": 5900.0
    })
}

fn benchmark_contract_v2_json() -> serde_json::Value {
    serde_json::json!({
        "contract_version": "1.1.0",
        "scope": "Apple M4 Mac mini dense SLM benchmark v2",
        "profile_execution_model": "one resident warm-session run per named profile",
        "supported_profiles": [
            "short_prompt_16_out",
            "short_prompt_64_out",
            "long_prompt_16_out",
            "long_prompt_128_out",
            "context_1k",
            "context_4k",
            "resident_25",
            "resident_50",
            "resident_100"
        ],
        "required_metrics": {
            "timing": [
                "cold_load_ms",
                "tokenizer_load_ms",
                "prompt_tokenize_ms",
                "prefill_ms",
                "time_to_first_token_ms",
                "decode_total_ms",
                "sampling_ms_per_token",
                "total_wall_ms"
            ],
            "throughput": [
                "input_tokens_per_second",
                "output_tokens_per_second",
                "decode_tokens_per_second"
            ],
            "memory": [
                "peak_memory_mb",
                "memory_drift_mb"
            ],
            "aggregate_speed": [
                "cold_load_ms",
                "tokenizer_load_ms",
                "prompt_tokenize_ms",
                "prefill_ms",
                "ttft_ms",
                "sampling_ms_per_token",
                "input_tok_s",
                "output_tok_s",
                "decode_tok_s",
                "total_wall_ms"
            ]
        }
    })
}

fn benchmark_profile_v2_json(profile_id: &str) -> serde_json::Value {
    serde_json::json!({
        "profile_id": profile_id,
        "receipt_path": "ci/hardware/apple-m4-mac-mini/2026-05-14/slm-benchmark-v2/qwen/summary-profiles/short_prompt_16_out.json",
        "scenario": "short_prompt",
        "requested_max_new_tokens": 16,
        "prompt_count": 3,
        "generated_tokens": 48,
        "quality_passed": true,
        "model_loaded_once": true,
        "tokenizer_loaded_once": true,
        "reuse_scope": "resident_session",
        "prompt_tokens": benchmark_stat_v2_json(24.0, 32.0, 40.0),
        "output_tokens": benchmark_stat_v2_json(16.0, 16.0, 16.0),
        "timing": {
            "cold_load_ms": benchmark_stat_v2_json(3200.0, 3300.0, 3400.0),
            "tokenizer_load_ms": benchmark_stat_v2_json(45.0, 50.0, 55.0),
            "prompt_tokenize_ms": benchmark_stat_v2_json(4.0, 5.0, 6.0),
            "prefill_ms": benchmark_stat_v2_json(650.0, 720.0, 800.0),
            "time_to_first_token_ms": benchmark_stat_v2_json(1800.0, 2100.0, 2300.0),
            "decode_total_ms": benchmark_stat_v2_json(900.0, 1000.0, 1100.0),
            "sampling_ms_per_token": benchmark_stat_v2_json(0.2, 0.3, 0.4),
            "total_wall_ms": benchmark_stat_v2_json(5200.0, 5600.0, 5900.0)
        },
        "throughput": {
            "input_tokens_per_second": benchmark_stat_v2_json(118.0, 130.0, 140.0),
            "output_tokens_per_second": benchmark_stat_v2_json(15.2, 16.0, 16.8),
            "decode_tokens_per_second": benchmark_stat_v2_json(15.5, 16.2, 17.0)
        },
        "memory": {
            "peak_memory_mb": benchmark_stat_v2_json(3900.0, 3950.0, 3975.0),
            "memory_drift_mb": benchmark_stat_v2_json(0.0, 12.0, 24.0),
            "source": "getrusage.ru_maxrss process peak delta"
        }
    })
}

fn performance_summary_receipt(model_sha: &str) -> serde_json::Value {
    serde_json::json!({
        "artifact_kind": "apple_m4_slm_performance_profiles",
        "requested_backend": "apple-m4-cpu-neon",
        "selected_backend": "apple-m4-cpu-neon",
        "runtime_api": "cpu",
        "fallback_used": false,
        "profile_set": "performance",
        "build": {
            "profile": "release",
            "release_mode": true
        },
        "operator_thresholds": {
            "cold_load_separated": true,
            "model_tokenizer_reuse_visible": true,
            "model_tokenizer_reuse_visible_per_profile": true,
            "profiles_loaded_independently": true,
            "profile_set_model_loads": 4,
            "reuse_scope": "within_each_profile",
            "profiles_required": ["warm_16", "warm_32", "warm_64", "warm_128"],
            "thresholds_are_claim_bounds_not_speed_guarantees": true
        },
        "performance_baseline": {
            "release_mode_required": true,
            "release_mode_observed": true,
            "warm_128_included": true,
            "broad_performance_claim": false,
            "speedup_claim": false
        },
        "allocation_audit": {
            "enabled": true,
            "method": "process_global_allocator_counter_delta",
            "scope": "selected Apple M4 CPU/NEON SLM warm-session profile set",
            "optimization_deferred": true,
            "ranked_hotspots": [
                {"component": "model.forward", "alloc_count": 10, "alloc_bytes": 1024}
            ]
        },
        "model_cache": {
            "id": "qwen2.5-0.5b-instruct-q8_0",
            "sha256": model_sha,
            "architecture": "qwen2",
            "quantization": "Q8_0",
            "tokenizer_model": "gpt2",
            "tokenizer_pre": "qwen2"
        },
        "profiles": [
            performance_profile_json("warm_16", 16, 8000.0, 5000.0),
            performance_profile_json("warm_32", 32, 16000.0, 10000.0),
            performance_profile_json("warm_64", 64, 32000.0, 20000.0),
            performance_profile_json("warm_128", 128, 64000.0, 40000.0)
        ],
        "mac_claim_boundary": {
            "full_metal_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "bitnet_quality_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false
        }
    })
}

fn slm_eval_summary_report() -> serde_json::Value {
    serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "apple_m4_slm_eval_summary",
        "machine_id": "apple-m4-mac-mini",
        "model_id": "qwen2.5-0.5b-instruct-q8_0",
        "requested_backend": "apple-m4-cpu-neon",
        "selected_backend": "apple-m4-cpu-neon",
        "runtime_api": "cpu",
        "fallback_used": false,
        "model": {
            "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "file": "qwen2.5-0.5b-instruct-q8_0.gguf",
            "sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
            "family": "qwen",
            "architecture": "qwen2",
            "quantization": "Q8_0"
        },
        "tokenizer": {
            "source": "gguf_metadata",
            "authority": "gguf_metadata",
            "pretokenizer_authority": "qwen2",
            "strict": true
        },
        "prompt_template": "qwen2.5",
        "corpus": {
            "name": "apple-m4-slm-eval-seeded-corpus-v1",
            "seed": 424242,
            "case_count": 10
        },
        "accuracy": {
            "cases_total": 10,
            "cases_scored": 10,
            "cases_passed": 9,
            "exact_match": 0.8,
            "normalized_match": 0.9,
            "json_schema_pass": 1.0,
            "numeric_tolerance_pass": 1.0,
            "required_keywords_pass": 1.0,
            "forbidden_tokens_pass": 1.0
        },
        "scoring_summary": {
            "enabled": true,
            "passed": 9,
            "failed": 1,
            "not_run": 0,
            "total": 10
        },
        "task_families": {
            "arithmetic_exact": {
                "cases_total": 10,
                "cases_scored": 10,
                "cases_passed": 9,
                "pass_rate": 0.9,
                "quality_gate_cases_passed": 9,
                "quality_gate_pass_rate": 0.9,
                "scoring_kinds": ["exact_match"],
                "failure_taxonomy": {
                    "answer_content": 1
                }
            }
        },
        "evidence": {
            "generated_text_recorded": true,
            "generated_token_ids_recorded": true,
            "generated_tokens_total": 128,
            "case_receipts": [
                "ci/hardware/apple-m4-mac-mini/2026-05-13/slm-eval/qwen2.5-0.5b-instruct-q8_0/cases.json"
            ],
            "source_answer_corpus_receipt": "ci/hardware/apple-m4-mac-mini/2026-05-13/slm-eval/qwen2.5-0.5b-instruct-q8_0/answer-corpus.json"
        },
        "speed": {
            "cold_load_ms_p50": 3200.0,
            "tokenizer_load_ms_p50": 45.0,
            "prompt_tokenize_ms_p50": 4.0,
            "prefill_ms_p50": 650.0,
            "ttft_ms_p50": 1800.0,
            "ttft_ms_p90": 2100.0,
            "input_tok_s_p50": 118.0,
            "output_tok_s_p50": 15.2,
            "decode_tok_s_p50": 15.5,
            "sampling_ms_per_token_p50": 0.3,
            "total_wall_ms_p50": 5200.0
        },
        "memory": {
            "peak_memory_mb": 3900.0,
            "source": "getrusage.ru_maxrss"
        },
        "stability": {
            "resident_prompts": 50,
            "quality_passed": true,
            "memory_drift_mb": 64.0
        },
        "claim_boundary": {
            "dense_slm_only": true,
            "bounded_seeded_corpus_only": true,
            "broad_model_quality_claim": false,
            "broad_performance_claim": false,
            "bitnet_evidence": false,
            "bitnet_quality_claimed": false,
            "full_metal_inference_claimed": false,
            "qk256_apple_claimed": false,
            "neural_engine_claimed": false,
            "neural_engine_execution_claimed": false,
            "mpsgraph_inference_claimed": false,
            "macbook_evidence": false,
            "speedup_claim": false
        }
    })
}

fn slm_benchmark_v2_summary() -> serde_json::Value {
    serde_json::json!({
        "schema_version": "1.1.0",
        "artifact_kind": "apple_m4_slm_benchmark_v2",
        "requested_backend": "apple-m4-cpu-neon",
        "selected_backend": "apple-m4-cpu-neon",
        "runtime_api": "cpu",
        "fallback_used": false,
        "profile_set": "slm-benchmark-v2",
        "profiles_required": ["short_prompt_16_out"],
        "build": {
            "profile": "release",
            "release_mode": true
        },
        "model_cache": {
            "id": "qwen2.5-0.5b-instruct-q8_0",
            "sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
            "architecture": "qwen2",
            "quantization": "Q8_0",
            "tokenizer_pre": "qwen2"
        },
        "profiles": [
            benchmark_profile_v2_json("short_prompt_16_out")
        ],
        "prompt_count": 3,
        "generated_tokens": 48,
        "speed": benchmark_speed_v2_json(),
        "benchmark_contract": benchmark_contract_v2_json(),
        "memory": {
            "peak_memory_mb_p50": 3900.0,
            "peak_memory_mb_p90": 3950.0,
            "peak_memory_mb_p99": 3975.0,
            "memory_drift_mb_p50": 0.0,
            "memory_drift_mb_p90": 12.0,
            "memory_drift_mb_p99": 24.0,
            "source": "getrusage.ru_maxrss process peak delta"
        },
        "evidence": {
            "profile_receipts": ["ci/hardware/apple-m4-mac-mini/2026-05-14/slm-benchmark-v2/qwen/summary-profiles/short_prompt_16_out.json"],
            "generated_text_recorded": true,
            "generated_token_ids_recorded": true,
            "operator_command": "mac benchmark"
        },
        "mac_claim_boundary": {
            "dense_slm_only": true,
            "bounded_benchmark_profiles_only": true,
            "broad_model_quality_claim": false,
            "broad_performance_claim": false,
            "speedup_claim": false,
            "bitnet_quality_claimed": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "macbook_evidence": false
        },
        "speedup_claim": false
    })
}

fn slm_benchmark_variance_v1_summary() -> serde_json::Value {
    serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "apple_m4_benchmark_variance_v1",
        "requested_backend": "apple-m4-cpu-neon",
        "selected_backend": "apple-m4-cpu-neon",
        "runtime_api": "cpu",
        "fallback_used": false,
        "profile_set": "slm-benchmark-v2",
        "profiles_required": ["short_prompt_16_out"],
        "prompt_count": 6,
        "generated_tokens": 96,
        "repeat": {
            "requested": 2,
            "completed": 2,
            "profile_count": 1,
            "sample_count": 2
        },
        "metrics": {
            "speed": benchmark_variance_metric_section(&[
                "cold_load_ms",
                "tokenizer_load_ms",
                "prompt_tokenize_ms",
                "prefill_ms",
                "ttft_ms",
                "sampling_ms_per_token",
                "input_tok_s",
                "output_tok_s",
                "decode_tok_s",
                "total_wall_ms",
            ]),
            "memory": benchmark_variance_metric_section(&[
                "peak_memory_mb",
                "memory_drift_mb",
            ])
        },
        "variance_band": {
            "method": "min/max and p50/p90/p99 over repeated child benchmark summary aggregate metrics",
            "reported_stats": ["count", "p50", "p90", "p99", "min", "max", "samples"],
            "threshold_derivation": "uses the M4 operator envelope drift thresholds",
            "advisory_vs_failure": "timing and memory drift are advisory unless fail-on-drift is enabled"
        },
        "outlier_handling": {
            "method": "none",
            "reason": "raw repeat samples are preserved"
        },
        "invalid_comparison_reasons": [],
        "build": {
            "profile": "release",
            "release_mode": true
        },
        "model_cache": {
            "id": "qwen2.5-0.5b-instruct-q8_0",
            "sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
            "architecture": "qwen2",
            "quantization": "Q8_0",
            "tokenizer_pre": "qwen2"
        },
        "evidence": {
            "child_summary_receipts": [
                "ci/hardware/apple-m4-mac-mini/2026-05-19/benchmark-variance/run-01.json",
                "ci/hardware/apple-m4-mac-mini/2026-05-19/benchmark-variance/run-02.json"
            ],
            "child_artifact_kind": "apple_m4_slm_benchmark_v2",
            "generated_text_recorded": true,
            "generated_token_ids_recorded": true,
            "operator_command": "mac benchmark --repeat"
        },
        "mac_claim_boundary": {
            "dense_slm_only": true,
            "variance_harness_only": true,
            "final_variance_envelope": false,
            "broad_model_quality_claim": false,
            "broad_performance_claim": false,
            "speedup_claim": false,
            "bitnet_quality_claimed": false,
            "bitnet_performance_claimed": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "macbook_evidence": false
        },
        "speedup_claim": false
    })
}

fn dense_quality_warm_session_receipt() -> serde_json::Value {
    let cases = [
        ("math_2_plus_2", "What is 2+2? Answer briefly.", vec![198, 17, 10, 17]),
        ("capital_france", "Name the capital of France.", vec![785, 6722, 374, 12095]),
        ("rust_sentence", "Write one short sentence about Rust.", vec![58047, 374, 264, 4221]),
        (
            "ready_instruction",
            "In one short sentence, say that the system is ready.",
            vec![785, 1849, 374, 5527],
        ),
        (
            "answer_prefix_blue",
            "Return the answer as 'Answer: blue' for this question: What color is a clear daytime sky?",
            vec![16141, 25, 6303, 151645],
        ),
        (
            "summarize_rust_traits",
            "Summarize in three words: Rust is fast, safe, and reliable.",
            vec![14238, 510, 8545, 11, 6827],
        ),
        (
            "rewrite_cache_sentence",
            "Rewrite as a short plain sentence: The model cache is healthy.",
            vec![785, 1614, 6639, 374, 9811],
        ),
    ];
    let mut prompts = Vec::new();
    let mut groups = Vec::new();
    for (case_id, prompt, ids) in cases {
        groups.push(serde_json::json!({
            "prompt": prompt,
            "attempt_count": 2,
            "case_id": case_id,
            "prompt_indices": [prompts.len(), prompts.len() + 1],
            "stable_generated_token_ids": true,
            "stable_text": true,
            "reference_generated_ids": ids,
            "reference_text": "stable answer"
        }));
        for repeat_index in 0..2 {
            prompts.push(serde_json::json!({
                "prompt_index": prompts.len(),
                "case_id": case_id,
                "repeat_index": repeat_index,
                "prompt": prompt,
                "text": "stable answer",
                "generated_tokens": ids.len(),
                "generated_token_ids": ids,
                "quality": {
                    "passed": true,
                    "valid_utf8": true,
                    "printable_utf8": true,
                    "non_empty": true,
                    "no_replacement_chars": true,
                    "mostly_text": true,
                    "non_degenerate": true,
                    "generated_tokens": ids.len(),
                    "distinct_generated_tokens": ids.iter().collect::<std::collections::BTreeSet<_>>().len(),
                    "failed_rules": []
                },
                "timing": {
                    "model_load_ms": 0.0,
                    "tokenizer_load_ms": 0.0,
                    "total_ms": 100.0
                },
                "backend": {
                    "requested_backend": "apple-m4-cpu-neon",
                    "selected_backend": "apple-m4-cpu-neon",
                    "runtime_api": "cpu",
                    "fallback_used": false
                }
            }));
        }
    }
    serde_json::json!({
        "artifact_kind": "slm_apple_m4_warm_session",
        "requested_backend": "apple-m4-cpu-neon",
        "selected_backend": "apple-m4-cpu-neon",
        "runtime_api": "cpu",
        "fallback_used": false,
        "session": {
            "model_loaded_once": true,
            "tokenizer_loaded_once": true,
            "prompt_count": 14
        },
        "corpus": {
            "artifact_kind": "apple_m4_slm_quality_corpus",
            "name": "apple-m4-slm-quality-determinism-v2",
            "case_count": 7,
            "repeat_runs": 2
        },
        "generation": {
            "mode": "greedy",
            "temperature": 0.0,
            "top_k": 1,
            "deterministic": true
        },
        "model": {
            "sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"
        },
        "tokenizer": {
            "source": "gguf_metadata",
            "pretokenizer_authority": "present"
        },
        "quality_summary": {
            "passed": true
        },
        "determinism": {
            "checked": true,
            "passed": true,
            "repeated_prompt_groups": 7,
            "groups": groups
        },
        "prompts": prompts,
        "claim_boundary": {
            "bitnet_quality_claimed": false,
            "full_metal_inference_claimed": false,
            "broad_performance_claim": false
        }
    })
}

fn resident_session_json() -> serde_json::Value {
    serde_json::json!({
        "reuse_scope": "resident_session",
        "session_owned_buffers": true,
        "prompt_token_buffer_reused": true,
        "generated_token_buffer_reused": true,
        "timing_buffers_reused": true,
        "allocation_audit_buffers_reused": true,
        "stop_tail_buffer_reused": true,
        "kv_cache_reuse_policy": "recreated_per_prompt_for_prompt_isolation",
        "sampler_reuse_policy": "recreated_per_prompt_for_deterministic_prompt_independence",
        "logits_buffer_reuse_policy": "not_claimed_until_logits_extraction_uses_reusable_storage"
    })
}

#[test]
fn mac_receipts_check_rejects_hidden_fallback() {
    let dir = tempfile::tempdir().expect("tempdir");
    let receipt_path = dir.path().join("fallback.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "artifact_kind": "inference_result",
            "requested_backend": "apple-m4-cpu-neon",
            "selected_backend": "apple-m4-cpu-neon",
            "runtime_api": "cpu",
            "fallback_used": true,
            "text": "4.",
            "tokens": {
                "generated": 1,
                "generated_ids": [19]
            },
            "model": {
                "sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"
            },
            "tokenizer": {
                "source": "gguf_metadata"
            }
        }))
        .expect("json"),
    )
    .expect("write receipt");
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("fallback_used=true"));
}

#[test]
fn slm_warm_session_requires_multiple_prompts_before_loading_model() {
    bitnet()
        .args([
            "--device",
            "apple-m4-cpu-neon",
            "slm-warm-session",
            "--model",
            "missing.gguf",
            "--prompt",
            "Only one prompt",
            "--json-out",
            "target/test-warm-session.json",
            "--strict-loader",
            "--strict-tokenizer",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("requires at least two --prompt values or --corpus"));
}

#[test]
fn slm_warm_session_requires_supported_cpu_receipt_label_before_loading_model() {
    bitnet()
        .args([
            "--device",
            "cuda",
            "slm-warm-session",
            "--model",
            "missing.gguf",
            "--prompt",
            "One",
            "--prompt",
            "Two",
            "--json-out",
            "target/test-warm-session.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cpu, apple-m4-cpu-neon, or apple-m3-air-cpu-neon"));
}

#[test]
fn cpu_phase_warm_session_help_documents_strict_phase_receipts() {
    bitnet()
        .args(["cpu-phase-warm-session", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("258V CPU phase prompts"))
        .stdout(predicate::str::contains("--prefill-prompt-file"))
        .stdout(predicate::str::contains("--decode-tokens"))
        .stdout(predicate::str::contains("--cpu-kernel"))
        .stdout(predicate::str::contains("--strict-loader"))
        .stdout(predicate::str::contains("--strict-tokenizer"))
        .stdout(predicate::str::contains("--json-out"));
}

#[test]
fn cpu_phase_warm_session_rejects_accelerator_backend_before_loading_model() {
    bitnet()
        .args([
            "--device",
            "intel-npu",
            "cpu-phase-warm-session",
            "--model",
            "missing.gguf",
            "--strict-loader",
            "--strict-tokenizer",
            "--json-out",
            "target/test-cpu-phase-warm-session.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("scoped to --device cpu"));
}

#[test]
fn cpu_phase_warm_session_requires_strict_loader_before_loading_model() {
    bitnet()
        .args([
            "--device",
            "cpu",
            "cpu-phase-warm-session",
            "--model",
            "missing.gguf",
            "--strict-tokenizer",
            "--json-out",
            "target/test-cpu-phase-warm-session.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("requires --strict-loader"));
}

#[test]
fn cpu_phase_warm_session_requires_strict_tokenizer_before_loading_model() {
    bitnet()
        .args([
            "--device",
            "cpu",
            "cpu-phase-warm-session",
            "--model",
            "missing.gguf",
            "--strict-loader",
            "--json-out",
            "target/test-cpu-phase-warm-session.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("requires --strict-tokenizer"));
}

#[test]
fn cpu_phase_warm_session_rejects_safetensors_before_loading_model() {
    bitnet()
        .args([
            "--device",
            "cpu",
            "cpu-phase-warm-session",
            "--model",
            "missing.safetensors",
            "--cpu-kernel",
            "scalar",
            "--strict-loader",
            "--strict-tokenizer",
            "--json-out",
            "target/test-cpu-phase-warm-session.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("supports GGUF only, not safetensors"));
}

#[test]
fn slm_warm_session_rejects_non_gguf_format_before_loading_model() {
    bitnet()
        .args([
            "--device",
            "apple-m4-cpu-neon",
            "slm-warm-session",
            "--model",
            "missing-model-dir",
            "--model-format",
            "safetensors",
            "--prompt",
            "One",
            "--prompt",
            "Two",
            "--json-out",
            "target/test-warm-session.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("supports GGUF only"));
}

#[test]
fn slm_warm_session_accepts_corpus_without_prompt_before_loading_model() {
    bitnet()
        .args([
            "--device",
            "apple-m4-cpu-neon",
            "slm-warm-session",
            "--model",
            "missing.gguf",
            "--corpus",
            "missing-corpus.yaml",
            "--json-out",
            "target/test-warm-session.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("failed to read warm-session corpus"));
}

#[test]
fn apple_m4_slm_quality_corpus_tracks_seven_bounded_cases() {
    let corpus_path = workspace_path("ci/quality/apple-m4-slm-quality-corpus.yaml");
    let corpus: serde_yaml::Value =
        serde_yaml::from_slice(&std::fs::read(corpus_path).expect("read corpus"))
            .expect("parse corpus");
    let cases = corpus["cases"].as_sequence().expect("cases");
    let ids: Vec<_> = cases.iter().filter_map(|case| case["id"].as_str()).collect();

    assert_eq!(corpus["artifact_kind"].as_str(), Some("apple_m4_slm_quality_corpus"));
    assert_eq!(corpus["name"].as_str(), Some("apple-m4-slm-quality-determinism-v2"));
    assert_eq!(cases.len(), 7);
    assert!(ids.contains(&"math_2_plus_2"));
    assert!(ids.contains(&"capital_france"));
    assert!(ids.contains(&"rust_sentence"));
    assert!(ids.contains(&"ready_instruction"));
    assert!(ids.contains(&"answer_prefix_blue"));
    assert!(ids.contains(&"summarize_rust_traits"));
    assert!(ids.contains(&"rewrite_cache_sentence"));
}

#[test]
fn slm_warm_session_real_model_receipt_fields_when_enabled() {
    let Ok(model) = std::env::var("BITNET_M4_SLM_QWEN_GGUF") else {
        eprintln!("skipping real SLM warm-session receipt test; set BITNET_M4_SLM_QWEN_GGUF");
        return;
    };
    let model_path = {
        let path = std::path::PathBuf::from(&model);
        if path.is_absolute() { path } else { workspace_path(&model) }
    };
    if !model_path.exists() {
        eprintln!("skipping real SLM warm-session receipt test; missing {}", model_path.display());
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("warm-session.json");
    let out_str = out.to_string_lossy().into_owned();
    let model_str = model_path.to_string_lossy().into_owned();
    let corpus_path = workspace_path("ci/quality/apple-m4-slm-quality-corpus.yaml");
    let corpus_str = corpus_path.to_string_lossy().into_owned();

    bitnet()
        .args([
            "--device",
            "apple-m4-cpu-neon",
            "slm-warm-session",
            "--model",
            model_str.as_str(),
            "--corpus",
            corpus_str.as_str(),
            "--strict-loader",
            "--strict-tokenizer",
            "--fail-on-quality",
            "--require-determinism",
            "--json-out",
            out_str.as_str(),
        ])
        .assert()
        .success();

    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&out).expect("read aggregate receipt"))
            .expect("json aggregate receipt");
    assert_eq!(receipt["artifact_kind"], "slm_apple_m4_warm_session");
    assert_eq!(receipt["requested_backend"], "apple-m4-cpu-neon");
    assert_eq!(receipt["selected_backend"], "apple-m4-cpu-neon");
    assert_eq!(receipt["runtime_api"], "cpu");
    assert_eq!(receipt["fallback_used"], false);
    assert_eq!(receipt["session"]["model_loaded_once"], true);
    assert_eq!(receipt["session"]["tokenizer_loaded_once"], true);
    assert_eq!(receipt["session"]["prompt_count"], 10);
    assert_eq!(receipt["session"]["reuse_scope"], "resident_session");
    assert_eq!(receipt["session"]["session_owned_buffers"], true);
    assert_eq!(receipt["session"]["prompt_token_buffer_reused"], true);
    assert_eq!(receipt["session"]["generated_token_buffer_reused"], true);
    assert_eq!(receipt["operator_ux"]["stream_tokens_requested"], false);
    assert_eq!(receipt["operator_ux"]["quiet_default_logs"], true);
    assert_eq!(receipt["operator_ux"]["time_to_first_token_receipts"], true);
    assert_eq!(receipt["operator_ux"]["clear_failure_messages"], true);
    assert!(
        !receipt["speed"]["timing"]["time_to_first_token_ms"]
            .as_array()
            .expect("aggregate TTFT samples")
            .is_empty()
    );
    assert_eq!(
        receipt["session"]["kv_cache_reuse_policy"],
        "recreated_per_prompt_for_prompt_isolation"
    );
    assert_eq!(
        receipt["session"]["sampler_reuse_policy"],
        "recreated_per_prompt_for_deterministic_prompt_independence"
    );
    assert_eq!(receipt["corpus"]["artifact_kind"], "apple_m4_slm_quality_corpus");
    assert_eq!(receipt["quality_summary"]["passed"], true);
    assert_eq!(receipt["determinism"]["checked"], true);
    assert_eq!(receipt["determinism"]["passed"], true);
    assert_eq!(receipt["claim_boundary"]["speedup_claim"], false);
    assert_eq!(receipt["claim_boundary"]["full_metal_inference_claimed"], false);
    assert_eq!(receipt["claim_boundary"]["bitnet_quality_claimed"], false);
    let prompts = receipt["prompts"].as_array().expect("prompt summaries");
    assert_eq!(prompts.len(), 10);
    for prompt in prompts {
        assert_eq!(prompt["backend"]["fallback_used"], false);
        assert_eq!(prompt["quality"]["passed"], true);
        assert_eq!(prompt["timing"]["model_load_ms"], 0.0);
        assert_eq!(prompt["timing"]["tokenizer_load_ms"], 0.0);
        assert!(
            prompt["timing"]["session_model_load_ms"].as_f64().unwrap_or_default() > 0.0,
            "session model load timing should be recorded"
        );
        let prompt_receipt_path = prompt["receipt_path"].as_str().expect("prompt receipt path");
        let prompt_receipt: serde_json::Value = serde_json::from_slice(
            &std::fs::read(prompt_receipt_path).expect("read prompt receipt"),
        )
        .expect("json prompt receipt");
        assert_eq!(prompt_receipt["fallback_used"], false);
        assert_eq!(prompt_receipt["speedup_claim"], false);
        assert_eq!(prompt_receipt["session_reuse"]["reuse_scope"], "resident_session");
        assert_eq!(prompt_receipt["session_reuse"]["session_owned_buffers"], true);
        assert_eq!(prompt_receipt["session_reuse"]["prompt_token_buffer_reused"], true);
        assert_eq!(prompt_receipt["session_reuse"]["generated_token_buffer_reused"], true);
        assert_eq!(prompt_receipt["operator_ux"]["stream_tokens_requested"], false);
        assert_eq!(prompt_receipt["operator_ux"]["time_to_first_token_receipt"], true);
        assert_eq!(prompt_receipt["operator_ux"]["clear_failure_messages"], true);
        assert_eq!(prompt_receipt["timing"]["model_load_ms"], 0.0);
        assert_eq!(prompt_receipt["timing"]["tokenizer_load_ms"], 0.0);
        assert_eq!(
            prompt_receipt["timing"]["time_to_first_token_ms"],
            prompt_receipt["timing"]["first_token_ms"]
        );
        assert!(
            prompt_receipt["tokens"]["generated"].as_u64().unwrap_or_default() > 0,
            "prompt should generate at least one token"
        );
    }
}

#[test]
fn legacy_inference_apple_label_error_points_to_receipt_backed_run_path() {
    bitnet()
        .args([
            "inference",
            "--model",
            "fake.gguf",
            "--prompt",
            "hello",
            "--device",
            "apple-m4-metal",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("does not support device label 'apple-m4-metal'"))
        .stderr(predicate::str::contains("Use `bitnet run` for receipt-backed Apple proof labels"))
        .stderr(predicate::str::contains("CPU fallback cannot count as Metal execution"));
}

/// `run --help` documents the --deterministic flag.
#[test]
fn run_help_documents_deterministic() {
    bitnet()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--deterministic"));
}

/// `run --help` documents the --prompt-template option.
#[test]
fn run_help_documents_prompt_template() {
    bitnet()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--prompt-template"));
}

/// `run --help` documents the first-token logit dump aliases used by SLM divergence capture.
#[test]
fn run_help_documents_logit_dump_alias() {
    bitnet()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--dump-logit-steps"))
        .stdout(predicate::str::contains("--logits-dump-steps"));
}

/// `run --help` documents the bounded Qwen checkpoint trace flags used by SLM-CPU-007.
#[test]
fn run_help_documents_qwen_trace_flags() {
    bitnet()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--qwen-trace-jsonl"))
        .stdout(predicate::str::contains("--qwen-trace-layer"))
        .stdout(predicate::str::contains("--qwen-trace-full-prompt"))
        .stdout(predicate::str::contains("--qwen-trace-prompt-ids"));
}

/// `run --help` documents the --repetition-penalty option.
#[test]
fn run_help_documents_repetition_penalty() {
    bitnet()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--repetition-penalty"));
}

// ============================================================================
// Run subcommand: type validation (invalid values → clap error)
// ============================================================================

/// `--temperature` rejects non-numeric input.
#[test]
fn run_rejects_non_numeric_temperature() {
    bitnet()
        .args(["run", "--model", "m.gguf", "--prompt", "hi", "--temperature", "hot"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid value"));
}

/// `--top-k` rejects non-integer input.
#[test]
fn run_rejects_non_integer_top_k() {
    bitnet()
        .args(["run", "--model", "m.gguf", "--prompt", "hi", "--top-k", "abc"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid value"));
}

/// `--top-p` rejects non-numeric input.
#[test]
fn run_rejects_non_numeric_top_p() {
    bitnet()
        .args(["run", "--model", "m.gguf", "--prompt", "hi", "--top-p", "high"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid value"));
}

/// `--max-new-tokens` rejects non-integer input.
#[test]
fn run_rejects_non_integer_max_new_tokens() {
    bitnet()
        .args(["run", "--model", "m.gguf", "--prompt", "hi", "--max-new-tokens", "many"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid value"));
}

/// `--seed` rejects non-integer input.
#[test]
fn run_rejects_non_integer_seed() {
    bitnet()
        .args(["run", "--model", "m.gguf", "--prompt", "hi", "--seed", "random"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid value"));
}

/// `--repetition-penalty` rejects non-numeric input.
#[test]
fn run_rejects_non_numeric_repetition_penalty() {
    bitnet()
        .args(["run", "--model", "m.gguf", "--prompt", "hi", "--repetition-penalty", "heavy"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid value"));
}

// ============================================================================
// Run subcommand: aliases
// ============================================================================

/// `generate` is a recognized alias for the `run` subcommand.
#[test]
fn generate_alias_accepted() {
    bitnet()
        .args(["generate", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--model"));
}

/// `run --max-tokens` visible alias is accepted (aliases --max-new-tokens).
#[test]
fn run_max_tokens_alias_in_help() {
    bitnet()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("max-tokens"));
}

// ============================================================================
// Subcommand routing (subcommands not covered by cli_smoke.rs)
// ============================================================================

/// `list-architectures --help` is recognized and succeeds.
#[test]
fn list_architectures_help() {
    bitnet()
        .args(["list-architectures", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("architectures"));
}

/// `list-templates --help` is recognized and succeeds.
#[test]
fn list_templates_help() {
    bitnet().args(["list-templates", "--help"]).assert().success();
}

/// `info --help` is recognized and succeeds.
#[test]
fn info_subcommand_help() {
    bitnet().args(["info", "--help"]).assert().success();
}

/// `config --help` is recognized and lists sub-actions (show, set, reset, path).
#[test]
fn config_subcommand_help() {
    bitnet().args(["config", "--help"]).assert().success().stdout(
        predicate::str::contains("show")
            .and(predicate::str::contains("set"))
            .and(predicate::str::contains("reset"))
            .and(predicate::str::contains("path")),
    );
}

/// `compat-check --help` is recognized and succeeds.
#[test]
fn compat_check_subcommand_help() {
    bitnet()
        .args(["compat-check", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--strict"));
}

/// `tokenize --help` is recognized and requires --model.
#[test]
fn tokenize_subcommand_help() {
    bitnet()
        .args(["tokenize", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--model"));
}

// ============================================================================
// Version / interface-version
// ============================================================================

/// `--interface-version` prints the CLI interface version (1.0.0).
#[test]
fn interface_version_flag() {
    bitnet()
        .arg("--interface-version")
        .assert()
        .success()
        .stdout(predicate::str::contains("1.0.0"));
}

// ============================================================================
// Full-CLI feature-gated subcommand routing
// ============================================================================

/// `chat --help` is recognized (requires full-cli).
#[cfg(feature = "full-cli")]
#[test]
fn chat_subcommand_help() {
    bitnet()
        .args(["chat", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--model"));
}

/// `bench --help` makes clear that the legacy benchmark is not CUDA proof.
#[cfg(feature = "full-cli")]
#[test]
fn bench_help_documents_no_cuda_fallback() {
    bitnet()
        .args(["bench", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("only cpu/auto"))
        .stdout(predicate::str::contains("--cuda-benchmark-receipt"))
        .stdout(predicate::str::contains("receipt-backed CUDA"));
}

/// `bench --device cuda` must fail closed instead of silently benchmarking CPU.
#[cfg(feature = "full-cli")]
#[test]
fn bench_cuda_device_fails_closed_without_cpu_fallback() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let model = dir.path().join("placeholder.gguf");
    std::fs::write(&model, b"placeholder")?;
    let model_str = model.to_string_lossy().into_owned();

    bitnet()
        .args(["bench", "--model", model_str.as_str(), "--device", "cuda"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("does not support device label 'cuda'"))
        .stderr(predicate::str::contains("must not silently fall back to CPU"))
        .stderr(predicate::str::contains("CPU fallback cannot count as CUDA execution"));
    Ok(())
}

/// `bench --device cuda --cuda-benchmark-receipt` reports governed CUDA benchmark evidence.
#[cfg(feature = "full-cli")]
#[test]
fn bench_cuda_device_reports_governed_benchmark_receipt() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = tempfile::tempdir()?;
    let receipt = dir.path().join("dense-qwen-benchmark-qualification.json");
    write_governed_cuda_benchmark_receipt(&receipt)?;
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args(["bench", "--device", "cuda", "--cuda-benchmark-receipt", receipt_str.as_str()])
        .assert()
        .success()
        .stdout(predicate::str::contains("CUDA Benchmark Receipt Report"))
        .stdout(predicate::str::contains("dense_gguf_qwen_benchmark_qualification_review"))
        .stdout(predicate::str::contains("Route: dense_regular_llm_cuda"))
        .stdout(predicate::str::contains("Fallback: false"))
        .stdout(predicate::str::contains("Speedup claim: false"))
        .stdout(predicate::str::contains("Benchmark-qualified speedup: false"))
        .stdout(predicate::str::contains("one_token"))
        .stdout(predicate::str::contains(
            "Claim boundary: receipt-backed CUDA benchmark report only",
        ));
    Ok(())
}

/// Governed CUDA benchmark receipt reports support machine-readable JSON output.
#[cfg(feature = "full-cli")]
#[test]
fn bench_cuda_benchmark_receipt_reports_json() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt = dir.path().join("dense-qwen-benchmark-qualification.json");
    write_governed_cuda_benchmark_receipt(&receipt)?;
    let receipt_str = receipt.to_string_lossy().into_owned();

    let output = bitnet()
        .args([
            "bench",
            "--device",
            "cuda",
            "--cuda-benchmark-receipt",
            receipt_str.as_str(),
            "--format",
            "json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let report: serde_json::Value = serde_json::from_slice(&output)?;

    assert_eq!(report["artifact_kind"], "dense_gguf_qwen_benchmark_qualification_review");
    assert_eq!(report["selected_backend"], "nvidia-rtx-5070-ti-cuda");
    assert_eq!(report["runtime_api"], "cuda");
    assert_eq!(report["fallback_used"], false);
    assert_eq!(report["speedup_claim"], false);
    assert_eq!(report["benchmark_qualified_speedup"], false);
    assert_eq!(report["profiles"][0]["profile"], "one_token");
    Ok(())
}

/// Governed CUDA benchmark receipt reports support profile CSV output.
#[cfg(feature = "full-cli")]
#[test]
fn bench_cuda_benchmark_receipt_reports_csv() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let receipt = dir.path().join("dense-qwen-benchmark-qualification.json");
    write_governed_cuda_benchmark_receipt(&receipt)?;
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "bench",
            "--device",
            "cuda",
            "--cuda-benchmark-receipt",
            receipt_str.as_str(),
            "--format",
            "csv",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("profile,decision,cpu_total_ms_mean,cuda_total_ms_mean"))
        .stdout(predicate::str::contains("one_token,not_accepted,1.000000,2.000000"))
        .stdout(predicate::str::contains("CUDA Benchmark Receipt Report").not());
    Ok(())
}

#[cfg(feature = "full-cli")]
fn write_governed_cuda_benchmark_receipt(path: &std::path::Path) -> std::io::Result<()> {
    std::fs::write(
        path,
        r#"{
  "artifact_kind": "dense_gguf_qwen_benchmark_qualification_review",
  "claim": "dense_gguf_qwen_benchmark_qualification_review",
  "selected_backend": "nvidia-rtx-5070-ti-cuda",
  "runtime_api": "cuda",
  "fallback_used": false,
  "speedup_claim": false,
  "benchmark_qualified_speedup": false,
  "full_cuda_residency_claimed": false,
  "model": {
    "id": "qwen2.5-0.5b-instruct-q8_0"
  },
  "execution_plan": {
    "selected_route": "dense_regular_llm_cuda",
    "selected_backend": "nvidia-rtx-5070-ti-cuda",
    "runtime_api": "cuda",
    "fallback_used": false,
    "speedup_claim": false,
    "full_cuda_residency_claimed": false
  },
  "profile_reviews": [
    {
      "profile": "one_token",
      "decision": "not_accepted",
      "cpu_total_ms_mean": 1.0,
      "cuda_total_ms_mean": 2.0,
      "host_to_device_ms": 3.0,
      "device_to_host_ms": 0.1,
      "quality_passed": true,
      "fallback_free": true,
      "benchmark_qualified_speedup": false
    }
  ],
  "claim_boundary": {
    "speedup_claim": false,
    "benchmark_qualified_speedup": false,
    "full_cuda_residency_claimed": false,
    "bitnet_packed_i2s_qk256_proof": false
  }
}"#,
    )
}

/// `--model` is still required for legacy benchmark execution.
#[cfg(feature = "full-cli")]
#[test]
fn bench_legacy_requires_model_without_cuda_benchmark_receipt() {
    bitnet().args(["bench", "--device", "cpu"]).assert().failure().stderr(
        predicate::str::contains(
            "--model <PATH> is required unless --cuda-benchmark-receipt is provided",
        ),
    );
}

/// A CUDA benchmark receipt must not be accepted while the requested device is CPU.
#[cfg(feature = "full-cli")]
#[test]
fn bench_cuda_benchmark_receipt_requires_cuda_device() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model = dir.path().join("placeholder.gguf");
    let receipt = dir.path().join("receipt.json");
    std::fs::write(&model, b"placeholder").expect("write model placeholder");
    std::fs::write(&receipt, b"{}").expect("write receipt placeholder");
    let model_str = model.to_string_lossy().into_owned();
    let receipt_str = receipt.to_string_lossy().into_owned();

    bitnet()
        .args([
            "bench",
            "--model",
            model_str.as_str(),
            "--device",
            "cpu",
            "--cuda-benchmark-receipt",
            receipt_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--cuda-benchmark-receipt requires --device cuda"));
}

/// `answer-corpus --help` is recognized (requires full-cli).
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_subcommand_help() {
    bitnet()
        .args(["answer-corpus", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--corpus"))
        .stdout(predicate::str::contains("--case-id"))
        .stdout(predicate::str::contains("--model-id"))
        .stdout(predicate::str::contains("--dry-run"))
        .stdout(predicate::str::contains("--dump-logit-steps"));
}

/// `answer-parity --help` advertises both legacy and generic comparison inputs.
#[cfg(feature = "full-cli")]
#[test]
fn answer_parity_subcommand_help_lists_legacy_and_generic_inputs() {
    bitnet()
        .args(["answer-parity", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--scalar"))
        .stdout(predicate::str::contains("--avx2"))
        .stdout(predicate::str::contains("--left"))
        .stdout(predicate::str::contains("--right"));
}

/// Generic answer parity requires both sides.
#[cfg(feature = "full-cli")]
#[test]
fn answer_parity_rejects_partial_generic_inputs() {
    bitnet()
        .args(["answer-parity", "--left", "left.json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--right is required"));
}

/// Legacy and generic answer parity inputs are mutually exclusive.
#[cfg(feature = "full-cli")]
#[test]
fn answer_parity_rejects_mixed_input_modes() {
    bitnet()
        .args([
            "answer-parity",
            "--left",
            "left.json",
            "--right",
            "right.json",
            "--scalar",
            "scalar.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("use either --left/--right"));
}

/// `answer-corpus --dry-run` validates corpus shape without requiring a model load.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_dry_run_writes_not_run_receipt() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let corpus = dir.path().join("corpus.yaml");
    let out = dir.path().join("receipt.json");
    std::fs::write(
        &corpus,
        r#"schema: 1
artifact_kind: bitnet_answer_corpus
name: test-corpus
description: test
model:
  repo: microsoft/bitnet-b1.58-2B-4T-gguf
  file: ggml-model-i2_s.gguf
defaults:
  prompt_template: llama3-chat
  max_new_tokens: 4
  greedy: true
  deterministic: true
  strict_loader: true
  temperature: 0.0
cases:
  - id: math
    question: "What is 2+2?"
    gate:
      kind: exact_trimmed
      expected: "4"
"#,
    )?;
    let corpus_str = corpus.to_string_lossy().into_owned();
    let out_str = out.to_string_lossy().into_owned();

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--model",
            "missing.gguf",
            "--corpus",
            corpus_str.as_str(),
            "--json-out",
            out_str.as_str(),
        ])
        .assert()
        .success();

    let receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(out)?)?;
    assert_eq!(receipt["artifact_kind"], "bitnet_cpu_answer_corpus");
    assert_eq!(receipt["quality_summary"]["not_run"], 1);
    assert_eq!(receipt["cases"][0]["status"], "not_run");
    assert_eq!(receipt["model"]["answer_ready_artifact_available"], false);
    assert_eq!(receipt["claim_boundary"]["diagnostic_only_until_answer_ready_artifact"], true);
    assert_eq!(receipt["claim_boundary"]["backend_quality_gate_passed"], false);
    assert_eq!(receipt["claim_boundary"]["coherent_answer_claimed"], false);
    Ok(())
}

/// `answer-corpus --case-id` limits a diagnostic run without changing corpus identity.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_case_id_dry_run_selects_one_case() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let corpus = dir.path().join("corpus.yaml");
    let out = dir.path().join("receipt.json");
    std::fs::write(
        &corpus,
        r#"schema: 1
artifact_kind: bitnet_answer_corpus
name: test-corpus
description: test
model:
  repo: microsoft/bitnet-b1.58-2B-4T-gguf
  file: ggml-model-i2_s.gguf
defaults:
  prompt_template: llama3-chat
  max_new_tokens: 4
  greedy: true
  deterministic: true
  strict_loader: true
  temperature: 0.0
cases:
  - id: math
    question: "What is 2+2?"
    gate:
      kind: exact_trimmed
      expected: "4"
  - id: word
    question: "Answer with one word."
    gate:
      kind: contains_any
      contains_any: ["yes", "no"]
"#,
    )?;
    let corpus_str = corpus.to_string_lossy().into_owned();
    let out_str = out.to_string_lossy().into_owned();

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--case-id",
            "word",
            "--model",
            "missing.gguf",
            "--corpus",
            corpus_str.as_str(),
            "--json-out",
            out_str.as_str(),
        ])
        .assert()
        .success();

    let receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(out)?)?;
    assert_eq!(receipt["corpus"]["case_count"], 2);
    assert_eq!(receipt["corpus"]["selected_case_count"], 1);
    assert_eq!(receipt["corpus"]["selected_case_ids"][0], "word");
    assert_eq!(receipt["quality_summary"]["total"], 1);
    assert_eq!(receipt["quality_summary"]["not_run"], 1);
    assert_eq!(receipt["cases"][0]["id"], "word");
    assert_eq!(receipt["cases"][0]["status"], "not_run");
    Ok(())
}

/// `answer-corpus --case-id` fails early for unknown case IDs.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_case_id_rejects_missing_case() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let corpus = dir.path().join("corpus.yaml");
    std::fs::write(
        &corpus,
        r#"schema: 1
artifact_kind: bitnet_answer_corpus
name: test-corpus
description: test
model:
  repo: microsoft/bitnet-b1.58-2B-4T-gguf
  file: ggml-model-i2_s.gguf
defaults:
  prompt_template: llama3-chat
  max_new_tokens: 4
  greedy: true
  deterministic: true
  strict_loader: true
  temperature: 0.0
cases:
  - id: math
    question: "What is 2+2?"
    gate:
      kind: exact_trimmed
      expected: "4"
"#,
    )?;
    let corpus_str = corpus.to_string_lossy().into_owned();

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--case-id",
            "missing",
            "--model",
            "missing.gguf",
            "--corpus",
            corpus_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("answer-corpus --case-id not found: missing"));
    Ok(())
}

/// `answer-corpus` fails before child execution when the model path is missing.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_rejects_missing_model_before_child_run() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = tempfile::tempdir()?;
    let corpus = dir.path().join("corpus.yaml");
    let tokenizer = dir.path().join("tokenizer.json");
    let out = dir.path().join("receipt.json");
    std::fs::write(&tokenizer, "{}")?;
    std::fs::write(
        &corpus,
        r#"schema: 1
artifact_kind: bitnet_answer_corpus
name: strict-failure-corpus
description: test
model:
  repo: microsoft/bitnet-b1.58-2B-4T-gguf
  file: ggml-model-i2_s.gguf
  tokenizer_authority:
    source: external_tokenizer_json
defaults:
  prompt_template: bitnetcpp-answer
  max_new_tokens: 4
  greedy: true
  deterministic: true
  strict_loader: true
  temperature: 0.0
cases:
  - id: math
    question: "What is 2+2?"
    gate:
      kind: exact_trimmed
      expected: "4"
"#,
    )?;
    let missing_model = dir.path().join("missing.gguf").to_string_lossy().into_owned();
    let tokenizer_str = tokenizer.to_string_lossy().into_owned();
    let corpus_str = corpus.to_string_lossy().into_owned();
    let out_str = out.to_string_lossy().into_owned();

    bitnet()
        .args([
            "answer-corpus",
            "--device",
            "apple-m4-cpu-neon",
            "--model",
            missing_model.as_str(),
            "--tokenizer",
            tokenizer_str.as_str(),
            "--corpus",
            corpus_str.as_str(),
            "--json-out",
            out_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("answer-corpus model not found"))
        .stderr(predicate::str::contains("hidden fallback is not allowed"))
        .stderr(predicate::str::contains("--dry-run"));
    assert!(!out.exists(), "preflight failure should not write a proof receipt");
    Ok(())
}

/// `answer-corpus` requires explicit tokenizer authority for the shared BitNet corpus.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_requires_external_tokenizer_authority_before_child_run()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let model = dir.path().join("model.gguf");
    let corpus = dir.path().join("corpus.yaml");
    let out = dir.path().join("receipt.json");
    std::fs::write(&model, b"GGUF\x03\x00\x00\x00")?;
    std::fs::write(
        &corpus,
        r#"schema: 1
artifact_kind: bitnet_answer_corpus
name: strict-tokenizer-corpus
description: test
model:
  repo: microsoft/bitnet-b1.58-2B-4T-gguf
  file: ggml-model-i2_s.gguf
  tokenizer_authority:
    source: external_tokenizer_json
defaults:
  prompt_template: bitnetcpp-answer
  max_new_tokens: 4
  greedy: true
  deterministic: true
  strict_loader: true
  temperature: 0.0
cases:
  - id: math
    question: "What is 2+2?"
    gate:
      kind: exact_trimmed
      expected: "4"
"#,
    )?;
    let model_str = model.to_string_lossy().into_owned();
    let corpus_str = corpus.to_string_lossy().into_owned();
    let out_str = out.to_string_lossy().into_owned();

    bitnet()
        .args([
            "answer-corpus",
            "--device",
            "apple-m4-cpu-neon",
            "--model",
            model_str.as_str(),
            "--corpus",
            corpus_str.as_str(),
            "--json-out",
            out_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("requires --tokenizer"))
        .stderr(predicate::str::contains("external_tokenizer_json"))
        .stderr(predicate::str::contains("hidden tokenizer fallback is not allowed"));
    assert!(!out.exists(), "preflight failure should not write a proof receipt");
    Ok(())
}

/// `answer-corpus` rejects an explicit tokenizer path that cannot provide authority.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_rejects_missing_tokenizer_before_child_run()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let model = dir.path().join("model.gguf");
    let corpus = dir.path().join("corpus.yaml");
    let out = dir.path().join("receipt.json");
    std::fs::write(&model, b"GGUF\x03\x00\x00\x00")?;
    std::fs::write(
        &corpus,
        r#"schema: 1
artifact_kind: bitnet_answer_corpus
name: missing-tokenizer-corpus
description: test
model:
  repo: microsoft/bitnet-b1.58-2B-4T-gguf
  file: ggml-model-i2_s.gguf
  tokenizer_authority:
    source: external_tokenizer_json
defaults:
  prompt_template: bitnetcpp-answer
  max_new_tokens: 4
  greedy: true
  deterministic: true
  strict_loader: true
  temperature: 0.0
cases:
  - id: math
    question: "What is 2+2?"
    gate:
      kind: exact_trimmed
      expected: "4"
"#,
    )?;
    let model_str = model.to_string_lossy().into_owned();
    let missing_tokenizer =
        dir.path().join("missing-tokenizer.json").to_string_lossy().into_owned();
    let corpus_str = corpus.to_string_lossy().into_owned();
    let out_str = out.to_string_lossy().into_owned();

    bitnet()
        .args([
            "answer-corpus",
            "--device",
            "apple-m4-cpu-neon",
            "--model",
            model_str.as_str(),
            "--tokenizer",
            missing_tokenizer.as_str(),
            "--corpus",
            corpus_str.as_str(),
            "--json-out",
            out_str.as_str(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("answer-corpus tokenizer not found"))
        .stderr(predicate::str::contains("strict tokenizer authority"))
        .stderr(predicate::str::contains("hidden tokenizer fallback is not allowed"));
    assert!(!out.exists(), "preflight failure should not write a proof receipt");
    Ok(())
}

/// `answer-corpus --dry-run` accepts the SLM corpus and preserves model identity.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_dry_run_accepts_slm_answer_corpus() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("slm-answer-corpus.json");
    let corpus = workspace_path("ci/quality/slm-answer-corpus.yaml");

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--device",
            "cpu",
            "--model",
            "missing.gguf",
            "--corpus",
            corpus.to_str().unwrap(),
            "--json-out",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();

    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out).expect("read receipt")).expect("json receipt");
    assert_eq!(receipt["artifact_kind"], "slm_cpu_answer_corpus");
    assert_eq!(receipt["model"]["repo"], "Qwen/Qwen3-0.6B-GGUF");
    assert_eq!(receipt["model"]["architecture"], "qwen3");
    assert_eq!(receipt["model"]["quant_format"], "Q8_0");
    assert_eq!(receipt["model"]["tokenizer"], "gguf_metadata");
    assert_eq!(receipt["claim_boundary"]["slm_answer_path"], true);
    assert_eq!(receipt["claim_boundary"]["broad_performance_claimed"], false);
    assert_eq!(receipt["quality_summary"]["not_run"], 5);
}

/// `answer-corpus --dry-run` validates the seeded Apple M4 SLM eval scoring contract.
#[cfg(feature = "full-cli")]
#[test]
fn slm_eval_scoring_dry_run_preserves_seeded_scoring_contract()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("apple-m4-slm-eval.json");
    let corpus = workspace_path("ci/quality/apple-m4-slm-eval-seeded-corpus.yaml");

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--device",
            "cpu",
            "--model",
            "missing.gguf",
            "--corpus",
            corpus.to_str().unwrap(),
            "--json-out",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();

    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out).expect("read receipt")).expect("json receipt");
    assert_eq!(receipt["artifact_kind"], "slm_cpu_answer_corpus");
    assert_eq!(receipt["corpus"]["name"], "apple-m4-slm-eval-seeded-corpus-v1");
    assert_eq!(receipt["corpus"]["case_count"], 10);
    assert_eq!(receipt["scoring_summary"]["enabled"], true);
    assert_eq!(receipt["scoring_summary"]["total"], 10);
    assert_eq!(receipt["scoring_summary"]["not_run"], 10);
    let kinds: Vec<&str> = receipt["scoring_summary"]["kinds"]
        .as_array()
        .ok_or("missing scoring summary kinds array")?
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    for kind in [
        "exact_match",
        "json_schema",
        "normalized_match",
        "required_forbidden_tokens",
        "required_keywords",
    ] {
        assert!(kinds.contains(&kind), "missing scoring kind `{kind}`");
    }
    assert_eq!(receipt["cases"][0]["quality"]["scoring"]["kind"], "exact_match");
    assert_eq!(receipt["cases"][3]["quality"]["scoring"]["kind"], "json_schema");
    assert_eq!(receipt["cases"][9]["quality"]["scoring"]["forbidden_tokens"][0], "maybe");
    assert_eq!(receipt["claim_boundary"]["bounded_slm_answer_smoke_passed"], false);
    assert_eq!(receipt["claim_boundary"]["broad_performance_claimed"], false);
    Ok(())
}

/// `answer-corpus --dry-run` validates the Apple M4 long-context live corpus contract.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_long_context_dry_run_preserves_context_profiles()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("apple-m4-long-context-answer-corpus.json");
    let corpus = workspace_path("ci/quality/apple-m4-long-context-answer-corpus.yaml");
    let corpus_str = corpus.to_string_lossy().into_owned();
    let out_str = out.to_string_lossy().into_owned();

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--device",
            "apple-m4-cpu-neon",
            "--model",
            "missing.gguf",
            "--model-id",
            "qwen2.5-0.5b-instruct-q8_0",
            "--corpus",
            corpus_str.as_str(),
            "--json-out",
            out_str.as_str(),
        ])
        .assert()
        .success();

    let receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(out)?)?;
    assert_eq!(receipt["corpus"]["id"], "apple-m4-long-context-answer-corpus-v1");
    assert_eq!(receipt["corpus"]["name"], "apple-m4-long-context-answer-corpus-v1");
    assert_eq!(receipt["corpus"]["metadata"]["work_item"], "M4-CONTEXT-002");
    assert_eq!(receipt["corpus"]["case_count"], 4);
    assert_eq!(receipt["quality_summary"]["not_run"], 4);
    assert_eq!(receipt["scoring_summary"]["enabled"], true);
    assert_eq!(receipt["scoring_summary"]["total"], 4);
    assert_eq!(receipt["profile_summary"]["context_1k"]["total"], 2);
    assert_eq!(receipt["profile_summary"]["context_4k"]["total"], 1);
    assert_eq!(receipt["profile_summary"]["unsupported_boundary"]["total"], 1);
    assert_eq!(
        receipt["corpus"]["metadata"]["claim_boundary"]["dense_slm_evidence_proves_bitnet"],
        false
    );
    assert_eq!(
        receipt["corpus"]["metadata"]["claim_boundary"]["bitnet_long_context_proven"],
        false
    );
    Ok(())
}

/// `mac receipts-check` accepts the annotated M4-CONTEXT-002 long-context answer-corpus shape.
#[cfg(feature = "full-cli")]
#[test]
fn mac_receipts_check_accepts_long_context_answer_corpus_shape()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("apple-m4-long-context-answer-corpus.json");
    let corpus = workspace_path("ci/quality/apple-m4-long-context-answer-corpus.yaml");
    let corpus_str = corpus.to_string_lossy().into_owned();
    let out_str = out.to_string_lossy().into_owned();

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--device",
            "apple-m4-cpu-neon",
            "--model",
            "missing.gguf",
            "--model-id",
            "qwen2.5-0.5b-instruct-q8_0",
            "--corpus",
            corpus_str.as_str(),
            "--json-out",
            out_str.as_str(),
        ])
        .assert()
        .success();

    let mut receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(&out)?)?;
    receipt["artifact_kind"] = serde_json::json!("apple_m4_long_context_answer_corpus");
    receipt["suite"] = serde_json::json!("m4-long-context");
    receipt["work_item"] = serde_json::json!("M4-CONTEXT-002");
    receipt["model_id"] = serde_json::json!("qwen2.5-0.5b-instruct-q8_0");
    receipt["model_sha256"] =
        serde_json::json!("ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e");
    receipt["quality_summary"]["passed"] = serde_json::json!(4);
    receipt["quality_summary"]["failed"] = serde_json::json!(0);
    receipt["quality_summary"]["timeout"] = serde_json::json!(0);
    receipt["quality_summary"]["not_run"] = serde_json::json!(0);
    receipt["scoring_summary"]["passed"] = serde_json::json!(4);
    receipt["scoring_summary"]["failed"] = serde_json::json!(0);
    receipt["scoring_summary"]["not_run"] = serde_json::json!(0);
    receipt["m4_context_proof"] = serde_json::json!({
        "suite": "m4-long-context",
        "work_item": "M4-CONTEXT-002",
        "source_answer_corpus": "ci/quality/apple-m4-long-context-answer-corpus.yaml",
        "tested_model_id": "qwen2.5-0.5b-instruct-q8_0",
        "tested_model_sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
        "tested_backend": "apple-m4-cpu-neon",
        "quality_gate_passed": true,
        "live_quality_receipt_published": true,
        "benchmark_receipt_required": true,
        "profiles_required": ["context_1k", "context_4k", "unsupported_boundary"],
        "bitnet_long_context": {
            "status": "unsupported_until_bitnet_long_context_receipts_exist",
            "dense_slm_evidence_proves_bitnet": false
        }
    });
    receipt["claim_boundary"]["dense_slm_evidence_proves_bitnet"] = serde_json::json!(false);
    receipt["claim_boundary"]["bitnet_long_context_proven"] = serde_json::json!(false);
    receipt["claim_boundary"]["long_context_quality_receipt"] = serde_json::json!(true);
    receipt["claim_boundary"]["long_context_quality_gate_passed"] = serde_json::json!(true);
    receipt["claim_boundary"]["macbook_evidence"] = serde_json::json!(false);
    for case in receipt["cases"].as_array_mut().ok_or("missing cases")? {
        case["status"] = serde_json::json!("passed");
        case["quality"]["passed"] = serde_json::json!(true);
        case["quality"]["generated_tokens"] = serde_json::json!(1);
    }
    std::fs::write(&out, serde_json::to_vec_pretty(&receipt)?)?;

    bitnet()
        .args(["mac", "receipts-check", out_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apple_m4_long_context_answer_corpus"))
        .stdout(predicate::str::contains("\"prompt_count\": 4"))
        .stdout(predicate::str::contains("\"generated_tokens\": 4"));
    Ok(())
}

/// `answer-corpus --dry-run` validates the Apple M4 BitNet eval report schema contract.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_bitnet_eval_dry_run_preserves_task_family_and_reference_schema()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("apple-m4-bitnet-eval.json");
    let corpus = workspace_path("ci/quality/apple-m4-bitnet-eval-seeded-corpus.yaml");
    let corpus_str = corpus.to_string_lossy().into_owned();
    let out_str = out.to_string_lossy().into_owned();

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--device",
            "cpu",
            "--model",
            "missing.gguf",
            "--corpus",
            corpus_str.as_str(),
            "--json-out",
            out_str.as_str(),
        ])
        .assert()
        .success();

    let receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(&out)?)?;
    assert_eq!(receipt["artifact_kind"], "bitnet_cpu_answer_corpus");
    assert_eq!(receipt["corpus"]["id"], "apple-m4-bitnet-eval-seeded-corpus");
    assert_eq!(receipt["corpus"]["name"], "apple-m4-bitnet-eval-seeded-corpus");
    assert_eq!(receipt["corpus"]["case_count"], 100);
    assert_eq!(receipt["corpus"]["metadata"]["seed"], 912587);
    assert_eq!(
        receipt["corpus"]["metadata"]["generator_policy"],
        "deterministic-static-fixture-bitnet-v1"
    );
    assert_eq!(
        receipt["corpus"]["contract"]["contract_version"],
        "m4-eval-corpus-scorer-contract-v1"
    );
    assert_eq!(receipt["corpus"]["contract"]["corpus_version"], "1.0.0");
    assert_eq!(
        receipt["corpus"]["contract"]["scoring_schema"],
        "answer_corpus_mechanical_scoring_v1"
    );
    assert_eq!(
        receipt["scoring_contract"]["expected_output_provenance"],
        "Closed-form deterministic fixture answers derived from the prompt data in this YAML; reference-runner answers may be added as comparison evidence but do not replace the mechanical expected-output authority."
    );
    assert_eq!(
        receipt["scoring_contract"]["normalization_rules"],
        "answer_corpus_normalize_scoring_text_v1 plus normalize_match_text_v1 for normalized_match only; exact_match remains strict after trim."
    );
    assert_eq!(receipt["model"]["repo"], "microsoft/bitnet-b1.58-2B-4T-gguf");
    assert_eq!(receipt["model"]["revision"], "a1f2f1c765812aa8af3f6eda4a313707064bba15");
    assert_eq!(receipt["model"]["bytes"], 1_187_801_280u64);
    assert_eq!(
        receipt["model"]["sha256"],
        "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162"
    );
    assert_eq!(receipt["model"]["architecture"], "bitnet_b1_58");
    assert_eq!(receipt["model"]["quant_format"], "I2_S");
    assert_eq!(receipt["tokenizer"]["authority"]["source"], "external_tokenizer_json");
    assert_eq!(receipt["tokenizer"]["authority"]["ggml_pre"], "llama-bpe");
    assert_eq!(receipt["prompt_template_policy"]["family"], "bitnetcpp-answer");
    assert_eq!(receipt["scoring_summary"]["enabled"], true);
    assert_eq!(receipt["scoring_summary"]["total"], 100);
    assert_eq!(receipt["scoring_summary"]["not_run"], 100);
    assert_eq!(receipt["task_family_summary"]["arithmetic_exact"]["total"], 10);
    assert_eq!(receipt["task_family_summary"]["arithmetic_exact"]["not_run"], 10);
    assert_eq!(receipt["task_family_summary"]["required_forbidden_tokens"]["scoring"]["total"], 10);
    assert_eq!(receipt["cases"][0]["task_family"], "arithmetic_exact");
    assert!(
        receipt["cases"][0]["seed_material"]
            .as_str()
            .ok_or("missing seed material")?
            .contains("seed=912587")
    );
    assert_eq!(
        receipt["cases"][0]["reference_comparison"]["schema"],
        "bitnet_reference_vs_rust_v1"
    );
    assert_eq!(
        receipt["cases"][0]["reference_comparison"]["comparison"]["status"],
        "reference_not_supplied"
    );
    assert_eq!(receipt["reference_comparison"]["enabled"], true);
    assert!(receipt["reference_comparison"]["reference_comparison_plan"].is_null());
    assert_eq!(receipt["reference_comparison"]["rust_runner"]["fallback_used"], false);
    assert_eq!(receipt["reference_comparison"]["summary"]["reference_not_supplied"], 100);
    assert_eq!(receipt["reference_comparison"]["claim_boundary"]["dense_slm_evidence_used"], false);
    assert_eq!(receipt["reference_comparison"]["claim_boundary"]["chat_enabled"], false);
    assert_eq!(receipt["reference_comparison"]["claim_boundary"]["serve_enabled"], false);
    assert_eq!(receipt["reference_comparison"]["claim_boundary"]["performance_claimed"], false);
    assert_eq!(receipt["claim_boundary"]["broad_performance_claimed"], false);
    Ok(())
}

/// `answer-corpus --dry-run` validates the Apple M4 BitNet 250-case corpus contract.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_bitnet_250_dry_run_preserves_task_family_and_reference_schema()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("apple-m4-bitnet-eval-250.json");
    let corpus = workspace_path("ci/quality/apple-m4-bitnet-eval-seeded-corpus-250.yaml");
    let corpus_str = corpus.to_string_lossy().into_owned();
    let out_str = out.to_string_lossy().into_owned();

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--device",
            "apple-m4-cpu-neon",
            "--model",
            "missing.gguf",
            "--corpus",
            corpus_str.as_str(),
            "--json-out",
            out_str.as_str(),
        ])
        .assert()
        .success();

    let receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(&out)?)?;
    assert_eq!(receipt["artifact_kind"], "bitnet_apple_m4_local_answer_corpus");
    assert_eq!(receipt["backend_lane"], "apple_m4_cpu_neon");
    assert_eq!(receipt["corpus"]["id"], "apple-m4-bitnet-eval-seeded-corpus-250");
    assert_eq!(receipt["corpus"]["name"], "apple-m4-bitnet-eval-seeded-corpus-250");
    assert_eq!(receipt["corpus"]["case_count"], 250);
    assert_eq!(receipt["corpus"]["metadata"]["seed"], 912587);
    assert_eq!(receipt["corpus"]["metadata"]["work_item"], "M4-BITNET-EX-013");
    assert_eq!(
        receipt["corpus"]["metadata"]["generator_policy"],
        "deterministic-static-fixture-bitnet-v2"
    );
    assert_eq!(
        receipt["corpus"]["contract"]["contract_version"],
        "m4-eval-corpus-scorer-contract-v1"
    );
    assert_eq!(receipt["corpus"]["contract"]["corpus_version"], "2.1.0");
    assert_eq!(
        receipt["corpus"]["contract"]["scoring_schema"],
        "answer_corpus_mechanical_scoring_v1"
    );
    assert_eq!(
        receipt["scoring_contract"]["expected_output_provenance"],
        "Closed-form deterministic fixture answers derived from the prompt data in this YAML; reference-runner answers may be added as comparison evidence but do not replace the mechanical expected-output authority."
    );
    assert!(
        receipt["scoring_contract"]["normalization_rules"]
            .as_str()
            .ok_or("missing normalization rules")?
            .contains("normalize_match_text_v2")
    );
    assert_eq!(
        receipt["scoring_contract"]["expected_answer_authority"]["owner_work_item"],
        "M4-BITNET-EX-013"
    );
    assert_eq!(
        receipt["scoring_contract"]["reference_comparison_plan"]["status"],
        "reference_250_sidecar_not_yet_supplied"
    );
    assert_eq!(receipt["model"]["repo"], "microsoft/bitnet-b1.58-2B-4T-gguf");
    assert_eq!(receipt["model"]["revision"], "a1f2f1c765812aa8af3f6eda4a313707064bba15");
    assert_eq!(receipt["model"]["bytes"], 1_187_801_280u64);
    assert_eq!(
        receipt["model"]["sha256"],
        "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162"
    );
    assert_eq!(receipt["model"]["architecture"], "bitnet_b1_58");
    assert_eq!(receipt["model"]["quant_format"], "I2_S");
    assert_eq!(receipt["tokenizer"]["authority"]["source"], "external_tokenizer_json");
    assert_eq!(receipt["tokenizer"]["authority"]["ggml_pre"], "llama-bpe");
    assert_eq!(receipt["prompt_template_policy"]["family"], "bitnetcpp-answer");
    assert_eq!(receipt["scoring_summary"]["enabled"], true);
    assert_eq!(receipt["scoring_summary"]["total"], 250);
    assert_eq!(receipt["scoring_summary"]["not_run"], 250);

    for (family, expected_total) in [
        ("arithmetic_exact", 15),
        ("closed_label_classification", 20),
        ("constrained_summary", 30),
        ("fixed_table_qa", 35),
        ("format_constrained_json", 20),
        ("numeric_tolerance", 35),
        ("ordering_sorting", 20),
        ("required_forbidden_tokens", 30),
        ("rewrite_normalized", 20),
        ("synthetic_extraction", 25),
    ] {
        assert_eq!(receipt["task_family_summary"][family]["total"], expected_total);
        assert_eq!(receipt["task_family_summary"][family]["not_run"], expected_total);
    }

    assert_eq!(receipt["task_family_summary"]["numeric_tolerance"]["scoring"]["total"], 35);
    assert_eq!(receipt["task_family_summary"]["fixed_table_qa"]["scoring"]["total"], 35);
    assert!(
        receipt["task_family_summary"]["fixed_table_qa"]["scoring"]["kinds"]
            .as_array()
            .ok_or("missing fixed table scoring kinds")?
            .iter()
            .any(|kind| kind == "contains_expected")
    );
    assert_eq!(receipt["task_family_summary"]["required_forbidden_tokens"]["scoring"]["total"], 30);
    assert_eq!(receipt["cases"][0]["task_family"], "arithmetic_exact");
    assert!(
        receipt["cases"][0]["seed_material"]
            .as_str()
            .ok_or("missing seed material")?
            .contains("seed=912587")
    );
    assert_eq!(
        receipt["cases"][0]["reference_comparison"]["schema"],
        "bitnet_reference_vs_rust_v1"
    );
    assert_eq!(
        receipt["cases"][0]["reference_comparison"]["comparison"]["status"],
        "reference_not_supplied"
    );
    assert_eq!(receipt["reference_comparison"]["enabled"], true);
    assert_eq!(receipt["reference_comparison"]["rust_runner"]["fallback_used"], false);
    assert_eq!(receipt["reference_comparison"]["summary"]["reference_not_supplied"], 250);
    assert_eq!(receipt["reference_comparison"]["claim_boundary"]["dense_slm_evidence_used"], false);
    assert_eq!(receipt["reference_comparison"]["claim_boundary"]["chat_enabled"], false);
    assert_eq!(receipt["reference_comparison"]["claim_boundary"]["serve_enabled"], false);
    assert_eq!(receipt["reference_comparison"]["claim_boundary"]["performance_claimed"], false);
    assert_eq!(receipt["claim_boundary"]["broad_performance_claimed"], false);
    assert_eq!(receipt["claim_boundary"]["coherent_answer_claimed"], false);
    assert_eq!(receipt["claim_boundary"]["full_metal_inference_claimed"], false);
    Ok(())
}

/// `answer-corpus --model-id` pins aggregate SLM receipt identity to a supported M4 model.
#[cfg(feature = "full-cli")]
#[test]
fn slm_eval_v2_dry_run_pins_supported_dense_model_identity()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("apple-m4-slm-eval-v2.json");
    let corpus = workspace_path("ci/quality/apple-m4-slm-eval-seeded-corpus-v2.yaml");

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--device",
            "apple-m4-cpu-neon",
            "--model",
            "missing.gguf",
            "--model-id",
            "qwen2.5-1.5b-instruct-q4_k_m",
            "--corpus",
            corpus.to_str().unwrap(),
            "--json-out",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();

    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out).expect("read receipt")).expect("json receipt");
    assert_eq!(receipt["artifact_kind"], "bitnet_apple_m4_local_answer_corpus");
    assert_eq!(receipt["corpus"]["id"], "apple-m4-slm-eval-seeded-corpus-v2");
    assert_eq!(receipt["corpus"]["name"], "apple-m4-slm-eval-seeded-corpus-v2");
    assert_eq!(receipt["corpus"]["case_count"], 500);
    assert_eq!(receipt["corpus"]["metadata"]["seed"], 777331);
    assert_eq!(
        receipt["corpus"]["metadata"]["generator_policy"],
        "deterministic-static-fixture-v2"
    );
    assert_eq!(
        receipt["corpus"]["contract"]["contract_version"],
        "m4-eval-corpus-scorer-contract-v1"
    );
    assert_eq!(receipt["corpus"]["contract"]["corpus_version"], "2.2.0");
    assert_eq!(
        receipt["corpus"]["contract"]["expected_output_provenance"],
        "Closed-form deterministic fixture answers derived from the prompt data in this YAML; no model output, live run, or LLM judge is used as expected-output authority."
    );
    assert_eq!(
        receipt["corpus"]["contract"]["normalization_rules"],
        "answer_corpus_normalize_scoring_text_v2 plus normalize_match_text_v1 for normalized_match only; known Qwen ChatML stop tails and leading assistant separators are stripped before scoring, JSON/schema scoring may extract fenced or embedded JSON payloads deterministically, keyword checks use token boundaries, and exact_match remains strict after trim."
    );
    assert_eq!(
        receipt["scoring_contract"]["scoring_schema"],
        "answer_corpus_mechanical_scoring_v1"
    );
    assert_eq!(
        receipt["scoring_contract"]["receipt_contract"],
        "answer_corpus_aggregate_receipt_v1"
    );
    let scoring_kinds: Vec<&str> = receipt["scoring_contract"]["supported_scoring_kinds"]
        .as_array()
        .ok_or("missing supported scoring kinds")?
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    for kind in [
        "exact_match",
        "normalized_match",
        "json_schema",
        "numeric_tolerance",
        "required_keywords",
        "forbidden_tokens",
        "required_forbidden_tokens",
    ] {
        assert!(scoring_kinds.contains(&kind), "missing scoring contract kind `{kind}`");
    }
    let failure_categories: Vec<&str> = receipt["scoring_contract"]["supported_failure_categories"]
        .as_array()
        .ok_or("missing supported failure categories")?
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    for category in [
        "formatting",
        "factual_table",
        "extraction",
        "refusal",
        "timeout",
        "schema",
        "normalization",
    ] {
        assert!(failure_categories.contains(&category), "missing failure category `{category}`");
    }
    assert_eq!(receipt["model"]["id"], "qwen2.5-1.5b-instruct-q4_k_m");
    assert_eq!(receipt["model"]["repo"], "Qwen/Qwen2.5-1.5B-Instruct-GGUF");
    assert_eq!(receipt["model"]["revision"], "91cad51170dc346986eccefdc2dd33a9da36ead9");
    assert_eq!(receipt["model"]["file"], "qwen2.5-1.5b-instruct-q4_k_m.gguf");
    assert_eq!(
        receipt["model"]["sha256"],
        "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
    );
    assert_eq!(receipt["model"]["family"], "qwen");
    assert_eq!(receipt["model"]["architecture"], "qwen2");
    assert_eq!(receipt["model"]["quant_format"], "Q4_K_M");
    assert_eq!(receipt["model"]["tokenizer_authority"], "qwen2");
    assert_eq!(receipt["model_family"], "qwen");
    assert_eq!(receipt["model_architecture"], "qwen2");
    assert_eq!(receipt["quantization"], "Q4_K_M");
    assert_eq!(receipt["claim_boundary"]["broad_performance_claimed"], false);
    Ok(())
}

/// `answer-corpus --model-id` rejects BitNet and policy-only IDs for dense SLM reports.
#[cfg(feature = "full-cli")]
#[test]
fn slm_eval_v2_model_id_rejects_non_dense_slm_model() -> Result<(), Box<dyn std::error::Error>> {
    let corpus = workspace_path("ci/quality/apple-m4-slm-eval-seeded-corpus-v2.yaml");

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--device",
            "apple-m4-cpu-neon",
            "--model",
            "missing.gguf",
            "--model-id",
            "microsoft-bitnet-b1.58-2B-4T-i2s",
            "--corpus",
            corpus.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not selectable for dense Apple M4 SLM"));
    Ok(())
}

/// `reference-compare` validates an external SLM reference divergence artifact.
#[cfg(feature = "full-cli")]
#[test]
fn reference_compare_validates_slm_external_reference_artifact() {
    let dir = tempfile::tempdir().expect("tempdir");
    let artifact = dir.path().join("qwen3-reference.json");
    let out = dir.path().join("qwen3-reference-validation.json");
    std::fs::write(
        &artifact,
        r#"{
          "schema_version": "1.0.0",
          "artifact_kind": "backend_reference_compare",
          "model_sha256": "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
          "model_family": "qwen3",
          "prompt_text": "What is 2+2?",
          "prompt_template": "qwen",
          "bos": false,
          "reference": {
            "backend": "known-good-reference",
            "kernel": "reference",
            "prompt_ids": [1, 2, 3],
            "generated_ids": [4],
            "text": "4",
            "topk_step0": [[4, 10.0], [5, 1.0]],
            "chosen_id": 4
          },
          "bitnet_rs": {
            "backend": "cpu-rust",
            "kernel": "dense-q8_0-reference",
            "prompt_ids": [1, 2, 3],
            "generated_ids": [5],
            "text": "5",
            "topk_step0": [[5, 10.0], [4, 1.0]],
            "chosen_id": 5
          }
        }"#,
    )
    .expect("write artifact");

    bitnet()
        .args([
            "reference-compare",
            "--artifact",
            artifact.to_str().unwrap(),
            "--json-out",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();

    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out).expect("read receipt")).expect("json receipt");
    assert_eq!(receipt["artifact_kind"], "slm_reference_divergence_validation");
    assert_eq!(receipt["validation"]["passed"], true);
    assert_eq!(receipt["comparison"]["passed"], false);
    assert_eq!(receipt["comparison"]["first_divergence"]["phase"], "logits");
    assert_eq!(
        receipt["comparison"]["first_divergence"]["classification"],
        "logits_or_shared_transformer_math"
    );
    assert_eq!(receipt["comparison"]["first_divergence"]["index"], 0);
    assert_eq!(receipt["speedup_claim"], false);
}

/// `reference-compare` accepts the SmolLM2 first-token/top-k comparator shape.
#[cfg(feature = "full-cli")]
#[test]
fn reference_compare_validates_smollm2_first_token_topk_artifact()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let artifact = dir.path().join("smollm2-reference.json");
    let out = dir.path().join("smollm2-reference-validation.json");
    std::fs::write(
        &artifact,
        r#"{
          "schema_version": "1.0.0",
          "artifact_kind": "backend_reference_compare",
          "model_sha256": "48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201",
          "model_family": "smollm2",
          "prompt_text": "What is 2+2? Answer with only the number.",
          "prompt_template": "smollm2_chatml_with_explicit_system",
          "bos": true,
          "reference": {
            "backend": "llama-cli-known-good",
            "kernel": "external-reference",
            "prompt_ids": [1, 9690, 198, 2683, 359, 253, 5356, 11173, 30, 2],
            "generated_ids": [34],
            "text": "4",
            "topk_step0": [[34, 12.0], [504, 4.0]],
            "chosen_id": 34
          },
          "bitnet_rs": {
            "backend": "cpu-rust",
            "runtime_api": "cpu",
            "kernel": "dense-q8_0-reference",
            "loader_mode": "real_gguf",
            "tokenizer_source": "gguf_metadata",
            "tokenizer_strict": true,
            "fallback_used": false,
            "prompt_ids": [1, 9690, 198, 2683, 359, 253, 5356, 11173, 30, 2],
            "generated_ids": [504],
            "text": "The",
            "topk_step0": [[504, 10.0], [34, 8.0]],
            "chosen_id": 504
          }
        }"#,
    )?;

    bitnet()
        .arg("reference-compare")
        .arg("--artifact")
        .arg(&artifact)
        .arg("--json-out")
        .arg(&out)
        .assert()
        .success();

    let receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(out)?)?;
    assert_eq!(receipt["artifact_kind"], "slm_reference_divergence_validation");
    assert_eq!(receipt["model"]["family"], "smollm2");
    assert_eq!(receipt["validation"]["passed"], true);
    assert_eq!(receipt["comparison"]["passed"], false);
    assert_eq!(receipt["comparison"]["first_divergence"]["phase"], "logits");
    assert_eq!(
        receipt["comparison"]["first_divergence"]["classification"],
        "logits_or_shared_transformer_math"
    );
    assert_eq!(receipt["comparison"]["bitnet_rs"]["fallback_used"], false);
    assert_eq!(receipt["speedup_claim"], false);
    Ok(())
}

/// `--require-match` keeps the SmolLM2 comparator fail-closed when top-k diverges.
#[cfg(feature = "full-cli")]
#[test]
fn reference_compare_require_match_fails_smollm2_topk_divergence()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let artifact = dir.path().join("smollm2-reference.json");
    let out = dir.path().join("smollm2-reference-validation.json");
    std::fs::write(
        &artifact,
        r#"{
          "schema_version": "1.0.0",
          "artifact_kind": "backend_reference_compare",
          "model_sha256": "48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201",
          "model_family": "smollm2",
          "prompt_text": "What is 2+2? Answer with only the number.",
          "prompt_template": "smollm2_chatml_with_explicit_system",
          "bos": true,
          "reference": {
            "backend": "llama-cli-known-good",
            "kernel": "external-reference",
            "prompt_ids": [1, 2, 3],
            "generated_ids": [34],
            "text": "4",
            "topk_step0": [[34, 12.0], [504, 4.0]],
            "chosen_id": 34
          },
          "bitnet_rs": {
            "backend": "cpu-rust",
            "runtime_api": "cpu",
            "kernel": "dense-q8_0-reference",
            "loader_mode": "real_gguf",
            "tokenizer_source": "gguf_metadata",
            "tokenizer_strict": true,
            "fallback_used": false,
            "prompt_ids": [1, 2, 3],
            "generated_ids": [504],
            "text": "The",
            "topk_step0": [[504, 10.0], [34, 8.0]],
            "chosen_id": 504
          }
        }"#,
    )?;

    bitnet()
        .arg("reference-compare")
        .arg("--artifact")
        .arg(&artifact)
        .arg("--json-out")
        .arg(&out)
        .arg("--require-match")
        .assert()
        .failure()
        .stderr(predicate::str::contains("reference artifact diverged"));
    Ok(())
}

/// `first-token-divergence --help` documents the external reference and local CPU inputs.
#[cfg(feature = "full-cli")]
#[test]
fn first_token_divergence_subcommand_help_lists_evidence_inputs() {
    bitnet()
        .args(["first-token-divergence", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--external-reference"))
        .stdout(predicate::str::contains("--prompt-audit"))
        .stdout(predicate::str::contains("--scalar-answer-corpus"))
        .stdout(predicate::str::contains("--avx2-answer-corpus"))
        .stdout(predicate::str::contains("--answer-parity"));
}

/// `external-reference-instrumentation --help` documents the reference capability inputs.
#[cfg(feature = "full-cli")]
#[test]
fn external_reference_instrumentation_help_lists_boundary_inputs() {
    bitnet()
        .args(["external-reference-instrumentation", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--external-reference"))
        .stdout(predicate::str::contains("--runner-help"))
        .stdout(predicate::str::contains("--json-out"));
}

/// `output-head-logits-audit --help` documents tensor and top-k evidence inputs.
#[cfg(feature = "full-cli")]
#[test]
fn output_head_logits_audit_help_lists_boundary_inputs() {
    bitnet()
        .args(["output-head-logits-audit", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--model"))
        .stdout(predicate::str::contains("--machine-id"))
        .stdout(predicate::str::contains("--tokenizer"))
        .stdout(predicate::str::contains("--prompt-audit"))
        .stdout(predicate::str::contains("--scalar-answer-corpus"))
        .stdout(predicate::str::contains("--avx2-answer-corpus"))
        .stdout(predicate::str::contains("--json-out"));
}

/// `answer-corpus` can target the Apple M4 CPU/NEON local-answer lane.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_dry_run_accepts_apple_m4_cpu_neon_lane() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("apple-m4-local-answer.json");
    let corpus = workspace_path("ci/quality/apple-m4-local-answer-corpus.yaml");
    let corpus_str = corpus.to_string_lossy().into_owned();
    let out_str = out.to_string_lossy().into_owned();

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--device",
            "apple-m4-cpu-neon",
            "--model",
            "missing.gguf",
            "--corpus",
            corpus_str.as_str(),
            "--json-out",
            out_str.as_str(),
        ])
        .assert()
        .success();

    let receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(out)?)?;
    assert_eq!(receipt["artifact_kind"], "bitnet_apple_m4_local_answer_corpus");
    assert_eq!(receipt["backend"]["requested_backend"], "apple-m4-cpu-neon");
    assert_eq!(receipt["backend"]["selected_backend"], "apple-m4-cpu-neon");
    assert_eq!(receipt["backend"]["runtime_api"], "cpu");
    assert_eq!(receipt["backend"]["fallback_used"], false);
    assert_eq!(receipt["model"]["answer_ready_artifact_available"], false);
    assert_eq!(receipt["claim_boundary"]["local_answer_path"], true);
    assert_eq!(receipt["claim_boundary"]["diagnostic_only_until_answer_ready_artifact"], true);
    assert_eq!(receipt["claim_boundary"]["backend_quality_gate_passed"], false);
    assert_eq!(receipt["claim_boundary"]["full_metal_inference_claimed"], false);
    assert_eq!(receipt["quality_summary"]["not_run"], 3);
    assert_eq!(receipt["receipt_quality"]["case_receipt_checker"], "answer_receipt_failed_rules");
    assert_eq!(receipt["receipt_quality"]["checked"], false);
    let required_fields: Vec<&str> = receipt["receipt_quality"]["required_case_fields"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("required fields array"))?
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    for field in [
        "text",
        "tokens.generated",
        "model.sha256",
        "tokenizer.pretokenizer_authority",
        "fallback_used",
        "timing.decode_total_ms",
    ] {
        assert!(required_fields.contains(&field), "missing receipt field contract `{field}`");
    }
    let checked_rules: Vec<&str> = receipt["receipt_quality"]["checked_rules"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("checked rules array"))?
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    for rule in [
        "generated_text_recorded",
        "model_sha256_recorded",
        "tokenizer_pretokenizer_authority_recorded",
        "fallback_false",
        "timing_decode_total_ms_recorded",
    ] {
        assert!(checked_rules.contains(&rule), "missing receipt quality rule `{rule}`");
    }
    Ok(())
}

/// `answer-corpus` can target the RTX 5070 Ti CUDA diagnostic lane.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_dry_run_accepts_rtx5070ti_cuda_lane() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("cuda-answer-corpus.json");
    let corpus = workspace_path("ci/quality/bitnet-answer-corpus.yaml");

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--device",
            "nvidia-rtx-5070-ti-cuda",
            "--model",
            "missing.gguf",
            "--corpus",
            corpus.to_str().unwrap(),
            "--json-out",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();

    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out).expect("read receipt")).expect("json receipt");
    assert_eq!(receipt["artifact_kind"], "bitnet_cuda_answer_diagnostic_corpus");
    assert_eq!(receipt["backend"]["requested_backend"], "nvidia-rtx-5070-ti-cuda");
    assert_eq!(receipt["backend"]["selected_backend"], "nvidia-rtx-5070-ti-cuda");
    assert_eq!(receipt["backend"]["runtime_api"], "cuda");
    assert_eq!(receipt["backend"]["fallback_used"], false);
    assert_eq!(receipt["model"]["answer_ready_artifact_available"], true);
    assert_eq!(receipt["model"]["answer_ready"]["gate"], "MODEL-ARTIFACT-007");
    assert_eq!(receipt["tokenizer"]["source"], "externally_supplied_llama_bpe");
    assert_eq!(receipt["tokenizer"]["strict"], true);
    assert_eq!(
        receipt["tokenizer"]["authority"]["sha256"],
        "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7"
    );
    assert_eq!(receipt["claim_boundary"]["cuda_answer_corpus"], true);
    assert_eq!(receipt["claim_boundary"]["answer_ready_artifact_available"], true);
    assert_eq!(receipt["claim_boundary"]["diagnostic_only_until_answer_ready_artifact"], false);
    assert_eq!(receipt["claim_boundary"]["backend_quality_gate_passed"], false);
    assert_eq!(receipt["claim_boundary"]["strict_cuda_answer_claimed"], false);
    assert_eq!(receipt["claim_boundary"]["coherent_answer_claimed"], false);
    assert_eq!(receipt["quality_summary"]["not_run"], 5);
}

/// `answer-corpus` must not treat Apple Metal as the local-answer path.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_rejects_apple_m4_metal_lane() {
    let corpus = workspace_path("ci/quality/apple-m4-local-answer-corpus.yaml");

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--device",
            "apple-m4-metal",
            "--model",
            "missing.gguf",
            "--corpus",
            corpus.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "only accepts --device cpu, --device apple-m4-cpu-neon, --device cuda",
        ));
}

/// `answer-corpus` must not treat Apple MPSGraph as the local-answer path.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_rejects_apple_m4_mpsgraph_lane() {
    let corpus = workspace_path("ci/quality/apple-m4-local-answer-corpus.yaml");

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--device",
            "apple-m4-mpsgraph",
            "--model",
            "missing.gguf",
            "--corpus",
            corpus.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "only accepts --device cpu, --device apple-m4-cpu-neon, --device cuda",
        ));
}

/// `ask --help` exposes the user-answer surface.
#[test]
fn ask_subcommand_help() {
    bitnet()
        .args(["ask", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--question"))
        .stdout(predicate::str::contains("--strict-cuda"))
        .stdout(predicate::str::contains("--receipt-out"));
}

/// `ask --strict-cuda` must not silently run on auto/CPU.
#[test]
fn ask_strict_cuda_requires_lane_device() {
    bitnet()
        .args(["ask", "--model", "missing.gguf", "--question", "What is BitNet?", "--strict-cuda"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--strict-cuda requires --device nvidia-rtx-5070-ti-cuda",
        ));
}

/// `inference --help` is recognized (requires full-cli).
#[cfg(feature = "full-cli")]
#[test]
fn inference_subcommand_help() {
    bitnet()
        .args(["inference", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--prompt"));
}

/// `infer` alias routes to `inference` (requires full-cli).
#[cfg(feature = "full-cli")]
#[test]
fn infer_alias_accepted() {
    bitnet()
        .args(["infer", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--model"));
}

/// `convert --help` is recognized (requires full-cli).
#[cfg(feature = "full-cli")]
#[test]
fn convert_subcommand_help() {
    bitnet().args(["convert", "--help"]).assert().success();
}

/// `inspect --help` is recognized (requires full-cli).
#[cfg(feature = "full-cli")]
#[test]
fn inspect_subcommand_help() {
    bitnet().args(["inspect", "--help"]).assert().success();
}

// ============================================================================
// InferenceCommand try_parse_from tests (requires full-cli)
// ============================================================================

#[cfg(feature = "full-cli")]
mod inference_parsing {
    use bitnet_cli::commands::InferenceCommand;
    use clap::Parser;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        cmd: InferenceCommand,
    }

    fn parse(args: &[&str]) -> Result<InferenceCommand, clap::Error> {
        TestCli::try_parse_from(args).map(|c| c.cmd)
    }

    // -- seed --

    /// `--seed 42` sets the seed value.
    #[test]
    fn seed_sets_value() {
        let cmd = parse(&["test", "--seed", "42"]).expect("seed=42 must parse");
        assert_eq!(cmd.seed, Some(42));
    }

    /// Without `--seed`, seed defaults to `None`.
    #[test]
    fn seed_default_is_none() {
        let cmd = parse(&["test"]).expect("no-args must parse");
        assert!(cmd.seed.is_none());
    }

    /// `--seed` rejects non-integer.
    #[test]
    fn seed_rejects_non_integer() {
        assert!(parse(&["test", "--seed", "random"]).is_err());
    }

    // -- deterministic --

    /// `--deterministic` sets the flag to true.
    #[test]
    fn deterministic_sets_true() {
        let cmd = parse(&["test", "--deterministic"]).expect("must parse");
        assert!(cmd.deterministic);
    }

    /// Without `--deterministic`, defaults to false.
    #[test]
    fn deterministic_defaults_to_false() {
        let cmd = parse(&["test"]).expect("must parse");
        assert!(!cmd.deterministic);
    }

    // -- top-k --

    /// `--top-k 50` sets the value.
    #[test]
    fn top_k_accepts_positive() {
        let cmd = parse(&["test", "--top-k", "50"]).expect("must parse");
        assert_eq!(cmd.top_k, Some(50));
    }

    /// Without `--top-k`, defaults to None.
    #[test]
    fn top_k_default_is_none() {
        let cmd = parse(&["test"]).expect("must parse");
        assert!(cmd.top_k.is_none());
    }

    /// `--top-k` rejects negative (usize cannot be negative).
    #[test]
    fn top_k_rejects_negative() {
        assert!(parse(&["test", "--top-k", "-5"]).is_err());
    }

    // -- top-p --

    /// `--top-p 0.0` is accepted.
    #[test]
    fn top_p_accepts_zero() {
        let cmd = parse(&["test", "--top-p", "0.0"]).expect("must parse");
        assert!((cmd.top_p.unwrap() - 0.0).abs() < 1e-6);
    }

    /// `--top-p 1.0` is accepted.
    #[test]
    fn top_p_accepts_one() {
        let cmd = parse(&["test", "--top-p", "1.0"]).expect("must parse");
        assert!((cmd.top_p.unwrap() - 1.0).abs() < 1e-6);
    }

    /// `--top-p 0.95` is accepted.
    #[test]
    fn top_p_accepts_typical_value() {
        let cmd = parse(&["test", "--top-p", "0.95"]).expect("must parse");
        assert!((cmd.top_p.unwrap() - 0.95).abs() < 1e-6);
    }

    /// `--top-p` rejects non-numeric.
    #[test]
    fn top_p_rejects_non_numeric() {
        assert!(parse(&["test", "--top-p", "high"]).is_err());
    }

    // -- temperature --

    /// `--temperature 0.0` is accepted (greedy-equivalent).
    #[test]
    fn temperature_accepts_zero() {
        let cmd = parse(&["test", "--temperature", "0.0"]).expect("must parse");
        assert!((cmd.temperature - 0.0).abs() < 1e-6);
    }

    /// `--temperature 2.0` is accepted (high creativity).
    #[test]
    fn temperature_accepts_two() {
        let cmd = parse(&["test", "--temperature", "2.0"]).expect("must parse");
        assert!((cmd.temperature - 2.0).abs() < 1e-6);
    }

    /// `--temperature` rejects non-numeric.
    #[test]
    fn temperature_rejects_non_numeric() {
        assert!(parse(&["test", "--temperature", "warm"]).is_err());
    }

    // -- max-tokens --

    /// `--max-tokens 128` is accepted.
    #[test]
    fn max_tokens_accepts_reasonable_value() {
        let cmd = parse(&["test", "--max-tokens", "128"]).expect("must parse");
        assert_eq!(cmd.max_tokens, 128);
    }

    /// `--max-tokens 4096` is accepted.
    #[test]
    fn max_tokens_accepts_large_value() {
        let cmd = parse(&["test", "--max-tokens", "4096"]).expect("must parse");
        assert_eq!(cmd.max_tokens, 4096);
    }

    // -- model + prompt together --

    /// `--model` and `--prompt` can be provided together.
    #[test]
    fn model_and_prompt_together() {
        let cmd = parse(&["test", "--model", "m.gguf", "--prompt", "hello"]).expect("must parse");
        assert_eq!(cmd.model.as_ref().unwrap().to_str().unwrap(), "m.gguf");
        assert_eq!(cmd.prompt.as_deref(), Some("hello"));
    }

    // -- full sampling config --

    /// Complete sampling configuration parses correctly.
    #[test]
    fn full_sampling_config_parses() {
        let cmd = parse(&[
            "test",
            "--model",
            "model.gguf",
            "--prompt",
            "test",
            "--temperature",
            "0.8",
            "--top-p",
            "0.95",
            "--top-k",
            "40",
            "--max-tokens",
            "256",
            "--seed",
            "42",
            "--greedy",
            "--deterministic",
            "--repetition-penalty",
            "1.2",
        ])
        .expect("full config must parse");

        assert!((cmd.temperature - 0.8).abs() < 1e-6);
        assert!((cmd.top_p.unwrap() - 0.95).abs() < 1e-6);
        assert_eq!(cmd.top_k, Some(40));
        assert_eq!(cmd.max_tokens, 256);
        assert_eq!(cmd.seed, Some(42));
        assert!(cmd.greedy);
        assert!(cmd.deterministic);
        assert!((cmd.repetition_penalty - 1.2).abs() < 1e-6);
    }
}
