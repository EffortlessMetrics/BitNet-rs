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
        .stdout(predicate::str::contains("check"))
        .stdout(predicate::str::contains("ask"))
        .stdout(predicate::str::contains("validate"))
        .stdout(predicate::str::contains("receipts-check"));
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
fn mac_check_missing_cache_points_to_model_fetch() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = dir.path().join("models");
    let cache_str = cache.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "check", "--cache-dir", cache_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0"));
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
fn mac_receipts_check_accepts_valid_cpu_neon_answer_receipt() {
    let dir = tempfile::tempdir().expect("tempdir");
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
        }))
        .expect("json"),
    )
    .expect("write receipt");
    let receipt_str = receipt_path.to_string_lossy().into_owned();

    bitnet()
        .args(["mac", "receipts-check", receipt_str.as_str(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"passed\": true"))
        .stdout(predicate::str::contains("apple-m4-cpu-neon"));
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
            "ranked_hotspots": [
                {"component": "model.forward", "alloc_count": 10, "alloc_bytes": 1024}
            ]
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
fn slm_warm_session_requires_apple_m4_cpu_neon_before_loading_model() {
    bitnet()
        .args([
            "--device",
            "cpu",
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
        .stderr(predicate::str::contains("scoped to --device apple-m4-cpu-neon"));
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
    assert_eq!(receipt["session"]["prompt_count"], 6);
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
    assert_eq!(prompts.len(), 6);
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
        .stderr(predicate::str::contains("Use `bitnet run` for receipt-backed Apple M4 labels"))
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
fn answer_corpus_dry_run_writes_not_run_receipt() {
    let dir = tempfile::tempdir().expect("tempdir");
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
    )
    .expect("write corpus");

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
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
    assert_eq!(receipt["artifact_kind"], "bitnet_cpu_answer_corpus");
    assert_eq!(receipt["quality_summary"]["not_run"], 1);
    assert_eq!(receipt["cases"][0]["status"], "not_run");
}

/// `answer-corpus --case-id` limits a diagnostic run without changing corpus identity.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_case_id_dry_run_selects_one_case() {
    let dir = tempfile::tempdir().expect("tempdir");
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
    )
    .expect("write corpus");

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--case-id",
            "word",
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
    assert_eq!(receipt["corpus"]["case_count"], 2);
    assert_eq!(receipt["corpus"]["selected_case_count"], 1);
    assert_eq!(receipt["corpus"]["selected_case_ids"][0], "word");
    assert_eq!(receipt["quality_summary"]["total"], 1);
    assert_eq!(receipt["quality_summary"]["not_run"], 1);
    assert_eq!(receipt["cases"][0]["id"], "word");
    assert_eq!(receipt["cases"][0]["status"], "not_run");
}

/// `answer-corpus --case-id` fails early for unknown case IDs.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_case_id_rejects_missing_case() {
    let dir = tempfile::tempdir().expect("tempdir");
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
    )
    .expect("write corpus");

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--case-id",
            "missing",
            "--model",
            "missing.gguf",
            "--corpus",
            corpus.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("answer-corpus --case-id not found: missing"));
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

/// `answer-corpus` can target the Apple M4 CPU/NEON local-answer lane.
#[cfg(feature = "full-cli")]
#[test]
fn answer_corpus_dry_run_accepts_apple_m4_cpu_neon_lane() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("apple-m4-local-answer.json");
    let corpus = workspace_path("ci/quality/apple-m4-local-answer-corpus.yaml");

    bitnet()
        .args([
            "answer-corpus",
            "--dry-run",
            "--device",
            "apple-m4-cpu-neon",
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
    assert_eq!(receipt["artifact_kind"], "bitnet_apple_m4_local_answer_corpus");
    assert_eq!(receipt["backend"]["requested_backend"], "apple-m4-cpu-neon");
    assert_eq!(receipt["backend"]["selected_backend"], "apple-m4-cpu-neon");
    assert_eq!(receipt["backend"]["runtime_api"], "cpu");
    assert_eq!(receipt["backend"]["fallback_used"], false);
    assert_eq!(receipt["claim_boundary"]["local_answer_path"], true);
    assert_eq!(receipt["claim_boundary"]["full_metal_inference_claimed"], false);
    assert_eq!(receipt["quality_summary"]["not_run"], 3);
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
    assert_eq!(receipt["claim_boundary"]["cuda_answer_corpus"], true);
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
