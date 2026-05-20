//! Validators for benchmark receipt JSON artifacts.

use crate::ReceiptError;
use std::path::Path;

const QWEN25_05B_INSTRUCT_Q8_0_MODEL_ID: &str = "qwen2.5-0.5b-instruct-q8_0";
const QWEN25_05B_INSTRUCT_Q8_0_MODEL_FILE: &str = "qwen2.5-0.5b-instruct-q8_0.gguf";
const QWEN25_05B_INSTRUCT_Q8_0_MODEL_SHA256: &str =
    "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
const QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID: &str = "qwen3-0.6b-instruct-q8_0";
const QWEN3_06B_INSTRUCT_Q8_0_MODEL_FILE: &str = "Qwen3-0.6B-Q8_0.gguf";
const QWEN3_06B_INSTRUCT_Q8_0_MODEL_SHA256: &str =
    "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031";
const QWEN3_REPEATED_COMPARATOR_PROFILES: &[&str] = &[
    "one_token",
    "short_decode_8",
    "short_decode_32",
    "warm_session_3_turns",
    "decode_128_from_warm_context",
];
const STRICT_CUDA_PRODUCT_PROFILES_V1: &[&str] = &[
    "one_token",
    "short_decode_8",
    "short_decode_32",
    "warm_session_3_turns",
    "warm_session_10_turns",
];
const STRICT_CUDA_PERF_005_PROFILE_MATRIX_ID: &str = "cuda-bitnet-perf-005";
const STRICT_CUDA_PERF_005_PROFILES: &[&str] = &[
    "one_token",
    "short_decode_8",
    "short_decode_32",
    "prefill_128_decode_16",
    "prefill_512_decode_32",
    "warm_session_3_turns",
    "warm_session_10_turns",
    "decode_128_from_warm_context",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DenseQwenBenchmarkModel {
    Qwen25,
    Qwen3,
}

impl DenseQwenBenchmarkModel {
    fn min_runs_per_backend(self) -> u64 {
        match self {
            Self::Qwen25 => 3,
            Self::Qwen3 => 1,
        }
    }

    fn repeated_comparator_required(self) -> bool {
        matches!(self, Self::Qwen25)
    }
}

pub fn validate_rtx5070ti_cuda_benchmark_receipt_json(
    receipt: &serde_json::Value,
) -> Result<(), ReceiptError> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", "cuda_benchmark")?;
    require_string_eq(receipt, "machine_id", "windows-9950x3d-rtx5070ti")?;
    require_string_eq(receipt, "hardware_lane", "nvidia_rtx_5070_ti_cuda")?;
    require_string_eq(receipt, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(receipt, "runtime_api", "cuda")?;
    require_string_eq(receipt, "claim", "cuda_benchmark_baseline")?;
    require_bool_eq(receipt, "fallback_used", false)?;
    require_bool_eq(receipt, "speedup_claim", false)?;
    require_null(receipt, "fallback_backend")?;
    require_null(receipt, "fallback_reason")?;

    let cuda = require_object(receipt, "cuda")?;
    require_bool_eq(cuda, "available", true)?;
    require_u64_at_least(cuda, "device_count", 1)?;
    require_u64(cuda, "selected_device_index")?;
    let device_name = require_string(cuda, "selected_device_name")?;
    if !is_rtx5070ti_device_name(device_name) {
        return Err(validation_error(format!(
            "selected_device_name must identify NVIDIA GeForce RTX 5070 Ti, got {device_name}"
        )));
    }
    require_string_eq(cuda, "compute_capability", "12.0")?;
    require_non_empty_string(cuda, "driver_version")?;
    require_non_empty_string(cuda, "cuda_runtime_version")?;
    require_non_empty_string(cuda, "cuda_toolkit_version")?;
    require_non_empty_string(cuda, "nvrtc_version")?;
    require_u64_at_least(cuda, "vram_bytes", 1)?;

    let benchmark = require_object(receipt, "benchmark")?;
    require_string_eq(benchmark, "profile", "cuda_tiny_smoke")?;
    require_string_eq(benchmark, "kernel_id", "cuda_tiny_vector_add")?;
    require_string_eq(benchmark, "fixture_id", "cuda_tiny_vector_add_1024")?;
    require_u64_at_least(benchmark, "iterations", 1)?;
    require_string_eq(benchmark, "cpu_reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(benchmark, "cuda_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_non_negative_number(benchmark, "cpu_reference_ms")?;
    require_non_negative_number(benchmark, "cuda_total_ms")?;
    require_non_negative_number(benchmark, "cuda_kernel_ms")?;
    require_non_negative_number(benchmark, "host_to_device_ms")?;
    require_non_negative_number(benchmark, "device_to_host_ms")?;
    require_non_negative_number(benchmark, "speedup_vs_cpu")?;
    require_non_negative_number(benchmark, "max_abs_error")?;
    require_non_negative_number(benchmark, "mean_abs_error")?;
    require_bool_eq(benchmark, "passed", true)?;

    let cold_warm = require_object(benchmark, "cold_warm")?;
    require_non_negative_number(cold_warm, "compile_ms")?;
    require_non_negative_number(cold_warm, "first_iteration_total_ms")?;
    require_u64_at_least(cold_warm, "warm_iterations", 1)?;

    let profiles = receipt
        .get("profiles")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| validation_error("profiles must be an array"))?;
    for profile in [
        "cuda_tiny_smoke",
        "cuda_fp32_matmul_small",
        "cuda_i2s_matmul_small",
        "cuda_i2s_matmul_medium",
        "cuda_transfer_h2d_d2h",
    ] {
        if !profiles
            .iter()
            .any(|entry| entry.get("profile").and_then(serde_json::Value::as_str) == Some(profile))
        {
            return Err(validation_error(format!("profiles missing {profile}")));
        }
    }

    let stats = receipt
        .get("kernel_stats")
        .and_then(serde_json::Value::as_array)
        .and_then(|items| items.first())
        .ok_or_else(|| validation_error("kernel_stats must contain at least one entry"))?;
    require_string_eq(stats, "kernel_id", "cuda_tiny_vector_add")?;
    require_u64_at_least(stats, "invocations", 1)?;
    require_u64_eq(stats, "fallback_invocations", 0)?;
    require_u64_at_least(stats, "host_to_device_bytes", 1)?;
    require_u64_at_least(stats, "device_to_host_bytes", 1)?;
    require_u64_at_least(stats, "kernel_launches", 1)?;
    require_non_negative_number(stats, "kernel_time_ms")?;
    require_string_eq(stats, "selected_device_name", device_name)?;
    require_string_eq(stats, "compute_capability", "12.0")?;

    Ok(())
}

/// Validate an RTX 5070 Ti CUDA benchmark receipt file.
pub fn validate_rtx5070ti_cuda_benchmark_receipt_file(path: &Path) -> Result<(), ReceiptError> {
    let receipt = serde_json::from_slice(&std::fs::read(path)?)?;
    validate_rtx5070ti_cuda_benchmark_receipt_json(&receipt)
}

/// Validate a strict BitNet CUDA benchmark receipt for the RTX 5070 Ti lane.
///
/// This receipt is distinct from the earlier tiny-kernel CUDA benchmark. It
/// requires same-model strict BitNet decode evidence, selected RTX 5070 Ti CUDA
/// identity, a measured AVX-512 CPU reference, explicit scalar/AVX2 profile
/// disposition, CUDA kernel invocation counters, and no speedup claim.
pub fn validate_strict_bitnet_cuda_benchmark_receipt_json(
    receipt: &serde_json::Value,
) -> Result<(), ReceiptError> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", "strict_bitnet_cuda_benchmark")?;
    require_string_eq(receipt, "machine_id", "windows-9950x3d-rtx5070ti")?;
    require_string_eq(receipt, "hardware_lane", "nvidia_rtx_5070_ti_cuda")?;
    require_string_eq(receipt, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(receipt, "runtime_api", "cuda")?;
    require_string_eq(receipt, "claim", "strict_bitnet_cuda_benchmark_baseline")?;
    require_bool_eq(receipt, "fallback_used", false)?;
    require_bool_eq(receipt, "speedup_claim", false)?;
    require_null(receipt, "fallback_backend")?;
    require_null(receipt, "fallback_reason")?;

    let model = require_object(receipt, "model")?;
    require_string_eq(model, "repo", "microsoft/bitnet-b1.58-2B-4T-gguf")?;
    require_string_eq(model, "file", "ggml-model-i2_s.gguf")?;
    require_non_empty_string(model, "sha256")?;
    require_string_eq(model, "loader_mode", "strict")?;
    require_bool_eq(model, "fallback_loader_used", false)?;

    let tokenizer = require_object(receipt, "tokenizer")?;
    require_string_eq(tokenizer, "source", "explicit")?;
    require_bool_eq(tokenizer, "strict", true)?;

    let bitnet = require_object(receipt, "bitnet")?;
    require_string_eq(bitnet, "quantization", "W1.58A8")?;
    require_non_empty_string(bitnet, "kernel_family")?;
    require_non_empty_string(bitnet, "layout")?;
    require_bool(bitnet, "weights_uploaded_once")?;
    require_bool(bitnet, "per_token_weight_upload")?;

    let workload = require_object(receipt, "workload")?;
    require_string_eq(workload, "profile", "short_decode_8")?;
    require_u64_at_least(workload, "prompt_tokens", 1)?;
    require_u64_eq(workload, "generated_tokens", 8)?;
    require_non_empty_string(workload, "prompt")?;
    require_non_empty_string(workload, "generated_text")?;
    require_bool_eq(workload, "cpu_cuda_output_match", true)?;

    let contract = require_object(receipt, "comparison_contract")?;
    for field in [
        "same_model",
        "same_tokenizer",
        "same_prompt",
        "same_generated_token_count",
        "same_strict_loader_mode",
        "same_sampling_policy",
        "fallback_free",
    ] {
        require_bool_eq(contract, field, true)?;
    }

    let cuda = require_object(receipt, "cuda")?;
    require_bool_eq(cuda, "available", true)?;
    require_u64_at_least(cuda, "device_count", 1)?;
    require_u64(cuda, "device_index")?;
    let device_name = require_string(cuda, "device_name")?;
    if !is_rtx5070ti_device_name(device_name) {
        return Err(validation_error(format!(
            "cuda.device_name must identify NVIDIA GeForce RTX 5070 Ti, got {device_name}"
        )));
    }
    require_string_eq(cuda, "compute_capability", "12.0")?;
    require_non_empty_string(cuda, "driver_version")?;
    require_non_empty_string(cuda, "cuda_runtime_version")?;
    require_non_empty_string(cuda, "cuda_toolkit_version")?;
    require_non_empty_string(cuda, "nvrtc_version")?;
    require_u64_at_least(cuda, "vram_bytes", 1)?;
    require_u64_at_least(cuda, "memory_hwm_bytes", 1)?;
    require_u64_at_least(cuda, "cuda_kernel_invocations", 1)?;

    let benchmark = require_object(receipt, "benchmark")?;
    require_string_eq(benchmark, "profile", "short_decode_8")?;
    require_string_eq(benchmark, "cpu_reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(benchmark, "cuda_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_non_negative_number(benchmark, "cpu_avx512_total_ms")?;
    require_non_negative_number(benchmark, "cuda_total_ms")?;
    require_non_negative_number(benchmark, "cpu_avx512_tokens_per_second")?;
    require_non_negative_number(benchmark, "cuda_tokens_per_second")?;
    require_non_negative_number(benchmark, "cpu_avx512_total_ms_div_cuda_total_ms")?;
    require_u64_at_least(benchmark, "cuda_kernel_invocations", 1)?;
    require_bool_eq(benchmark, "cpu_cuda_output_match", true)?;
    require_bool_eq(benchmark, "speedup_claim", false)?;

    let profiles = require_array(receipt, "profiles")?;
    let cpu_scalar = require_backend_profile(profiles, "amd-9950x3d-cpu-scalar")?;
    validate_bitnet_benchmark_profile(cpu_scalar, false)?;
    let cpu_avx2 = require_backend_profile(profiles, "amd-9950x3d-cpu-avx2")?;
    validate_bitnet_benchmark_profile(cpu_avx2, false)?;
    let cpu_avx512 = require_backend_profile(profiles, "amd-9950x3d-cpu-avx512")?;
    validate_bitnet_benchmark_profile(cpu_avx512, true)?;
    let cuda_profile = require_backend_profile(profiles, "nvidia-rtx-5070-ti-cuda")?;
    validate_bitnet_benchmark_profile(cuda_profile, true)?;

    let stats = receipt
        .get("kernel_stats")
        .and_then(serde_json::Value::as_array)
        .and_then(|items| items.first())
        .ok_or_else(|| validation_error("kernel_stats must contain at least one entry"))?;
    require_string_eq(stats, "kernel_id", "qk256_gemv_cuda")?;
    require_u64_at_least(stats, "invocations", 1)?;
    require_u64_eq(stats, "fallback_invocations", 0)?;
    require_u64_at_least(stats, "kernel_launches", 1)?;

    let boundaries = require_array(receipt, "claim_boundaries")?;
    if boundaries.is_empty() {
        return Err(validation_error("claim_boundaries must not be empty"));
    }

    Ok(())
}

/// Validate a strict BitNet CUDA benchmark receipt file.
pub fn validate_strict_bitnet_cuda_benchmark_receipt_file(path: &Path) -> Result<(), ReceiptError> {
    let receipt = serde_json::from_slice(&std::fs::read(path)?)?;
    validate_strict_bitnet_cuda_benchmark_receipt_json(&receipt)
}

/// Validate a strict CUDA answer-path benchmark receipt for the RTX 5070 Ti lane.
///
/// This validator covers the product-facing ask path rather than the older
/// fixed short-decode benchmark. It requires the official BitNet artifact,
/// explicit tokenizer/prompt authority, measured CPU AVX-512 and RTX 5070 Ti
/// CUDA ask profiles, fallback-free CUDA QK256 execution, and no speedup claim.
/// Longer profiles may be recorded as blocked or not-run entries, but the
/// receipt must keep those gaps explicit.
pub fn validate_strict_cuda_answer_path_benchmark_receipt_json(
    receipt: &serde_json::Value,
) -> Result<(), ReceiptError> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", "strict_cuda_answer_path_benchmark")?;
    require_string_eq(receipt, "machine_id", "windows-9950x3d-rtx5070ti")?;
    require_string_eq(receipt, "hardware_lane", "nvidia_rtx_5070_ti_cuda")?;
    require_string_eq(receipt, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(receipt, "runtime_api", "cuda")?;
    require_string_eq(receipt, "claim", "strict_cuda_answer_path_benchmark_baseline")?;
    require_bool_eq(receipt, "fallback_used", false)?;
    require_bool_eq(receipt, "speedup_claim", false)?;
    require_bool_eq(receipt, "benchmark_qualified_speedup", false)?;
    require_null(receipt, "fallback_backend")?;
    require_null(receipt, "fallback_reason")?;
    validate_bitnet_qk256_execution_plan(require_object(receipt, "execution_plan")?)?;

    let model = require_object(receipt, "model")?;
    require_string_eq(model, "repo", "microsoft/bitnet-b1.58-2B-4T-gguf")?;
    require_string_eq(model, "file", "ggml-model-i2_s.gguf")?;
    require_non_empty_string(model, "sha256")?;
    require_string_eq(model, "loader_mode", "strict_real_gguf")?;
    require_bool_eq(model, "fallback_loader_used", false)?;

    let tokenizer = require_object(receipt, "tokenizer")?;
    require_string_eq(tokenizer, "source", "explicit")?;
    require_bool_eq(tokenizer, "strict", true)?;
    require_string_eq(tokenizer, "pretokenizer_authority", "llama-bpe")?;

    let prompt_template = require_object(receipt, "prompt_template")?;
    require_string_eq(prompt_template, "family", "bitnetcpp-answer")?;
    require_non_empty_string(prompt_template, "rendered_sha256")?;

    let workload = require_object(receipt, "workload")?;
    require_string_eq(workload, "profile", "strict_ask_math_8")?;
    require_non_empty_string(workload, "question")?;
    let answer = require_string(workload, "answer")?;
    if answer.trim() != "4" {
        return Err(validation_error(format!("workload.answer must trim to 4, got {answer:?}")));
    }
    require_u64_at_least(workload, "prompt_tokens", 1)?;
    require_u64_at_least(workload, "generated_tokens", 1)?;
    require_bool_eq(workload, "quality_passed", true)?;
    require_bool_eq(workload, "cpu_cuda_answer_match", true)?;
    require_bool_eq(workload, "cpu_cuda_generated_ids_match", true)?;

    let contract = require_object(receipt, "comparison_contract")?;
    for field in [
        "same_model",
        "same_tokenizer",
        "same_prompt_template",
        "same_question",
        "same_sampling_policy",
        "same_generated_token_ids",
        "same_answer",
        "fallback_free",
    ] {
        require_bool_eq(contract, field, true)?;
    }

    let cuda = require_object(receipt, "cuda")?;
    require_bool_eq(cuda, "available", true)?;
    require_u64_at_least(cuda, "device_count", 1)?;
    require_u64(cuda, "device_index")?;
    let device_name = require_string(cuda, "device_name")?;
    if !is_rtx5070ti_device_name(device_name) {
        return Err(validation_error(format!(
            "cuda.device_name must identify NVIDIA GeForce RTX 5070 Ti, got {device_name}"
        )));
    }
    require_string_eq(cuda, "compute_capability", "12.0")?;
    require_non_empty_string(cuda, "driver_version")?;
    require_non_empty_string(cuda, "cuda_runtime_version")?;
    require_non_empty_string(cuda, "cuda_toolkit_version")?;
    require_non_empty_string(cuda, "nvrtc_version")?;
    require_u64_at_least(cuda, "vram_bytes", 1)?;
    require_u64_at_least(cuda, "memory_hwm_bytes", 1)?;
    require_u64_at_least(cuda, "cuda_kernel_invocations", 1)?;

    let benchmark = require_object(receipt, "benchmark")?;
    require_string_eq(benchmark, "profile", "strict_ask_math_8")?;
    require_string_eq(benchmark, "cpu_reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(benchmark, "cuda_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_non_negative_number(benchmark, "cpu_avx512_total_ms")?;
    require_non_negative_number(benchmark, "cuda_total_ms")?;
    require_non_negative_number(benchmark, "cpu_avx512_tokens_per_second")?;
    require_non_negative_number(benchmark, "cuda_tokens_per_second")?;
    require_non_negative_number(benchmark, "observed_cpu_total_ms_div_cuda_total_ms")?;
    require_u64_at_least(benchmark, "cuda_kernel_invocations", 1)?;
    require_bool_eq(benchmark, "cpu_cuda_answer_match", true)?;
    require_bool_eq(benchmark, "speedup_claim", false)?;
    require_bool_eq(benchmark, "benchmark_qualified_speedup", false)?;

    let timing = require_object(receipt, "timing_split")?;
    let cpu_timing = require_object(timing, "cpu_avx512")?;
    validate_answer_path_timing(cpu_timing, false)?;
    let cuda_timing = require_object(timing, "cuda")?;
    validate_answer_path_timing(cuda_timing, true)?;

    let profiles = require_array(receipt, "profiles")?;
    validate_answer_path_profile(
        require_profile(profiles, "strict_ask_math_8", "amd-9950x3d-cpu-avx512")?,
        true,
    )?;
    validate_answer_path_profile(
        require_profile(profiles, "strict_ask_math_8", "nvidia-rtx-5070-ti-cuda")?,
        true,
    )?;
    validate_answer_path_profile(
        require_profile(profiles, "answer_corpus_5", "amd-9950x3d-cpu-avx512")?,
        false,
    )?;
    validate_answer_path_profile(
        require_profile(profiles, "answer_corpus_5", "nvidia-rtx-5070-ti-cuda")?,
        false,
    )?;
    validate_answer_path_profile(
        require_profile(profiles, "prefill_512_decode_128", "amd-9950x3d-cpu-avx512")?,
        false,
    )?;

    let stats = receipt
        .get("kernel_stats")
        .and_then(serde_json::Value::as_array)
        .and_then(|items| items.first())
        .ok_or_else(|| validation_error("kernel_stats must contain at least one entry"))?;
    require_string_eq(stats, "kernel_id", "qk256_gemv_cuda")?;
    require_u64_at_least(stats, "invocations", 1)?;
    require_u64_eq(stats, "fallback_invocations", 0)?;
    require_u64_at_least(stats, "kernel_launches", 1)?;

    let residency = require_object(receipt, "cuda_execution_residency")?;
    require_bool_eq(residency, "speedup_claim", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;

    let boundaries = require_array(receipt, "claim_boundaries")?;
    if boundaries.is_empty() {
        return Err(validation_error("claim_boundaries must not be empty"));
    }

    Ok(())
}

/// Validate a strict CUDA answer-path benchmark receipt file.
pub fn validate_strict_cuda_answer_path_benchmark_receipt_file(
    path: &Path,
) -> Result<(), ReceiptError> {
    let receipt = serde_json::from_slice(&std::fs::read(path)?)?;
    validate_strict_cuda_answer_path_benchmark_receipt_json(&receipt)
}

/// Validate a repeated strict CUDA ask benchmark receipt for the RTX 5070 Ti lane.
///
/// This receipt qualifies the single strict ask baseline with repeated
/// same-model CPU AVX-512 and RTX 5070 Ti CUDA runs. It still records
/// `speedup_claim=false`; the repeated timing ratio is evidence for later
/// review, not an accepted broad performance claim.
pub fn validate_strict_cuda_repeated_ask_benchmark_receipt_json(
    receipt: &serde_json::Value,
) -> Result<(), ReceiptError> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", "strict_cuda_repeated_ask_benchmark")?;
    require_string_eq(receipt, "machine_id", "windows-9950x3d-rtx5070ti")?;
    require_string_eq(receipt, "hardware_lane", "nvidia_rtx_5070_ti_cuda")?;
    require_string_eq(receipt, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(receipt, "runtime_api", "cuda")?;
    require_string_eq(receipt, "claim", "strict_cuda_repeated_ask_benchmark_baseline")?;
    require_bool_eq(receipt, "fallback_used", false)?;
    require_bool_eq(receipt, "speedup_claim", false)?;
    require_bool_eq(receipt, "benchmark_qualified_speedup", false)?;
    require_null(receipt, "fallback_backend")?;
    require_null(receipt, "fallback_reason")?;
    validate_bitnet_qk256_execution_plan(require_object(receipt, "execution_plan")?)?;

    let model = require_object(receipt, "model")?;
    require_string_eq(model, "repo", "microsoft/bitnet-b1.58-2B-4T-gguf")?;
    require_string_eq(model, "file", "ggml-model-i2_s.gguf")?;
    require_non_empty_string(model, "sha256")?;
    require_string_eq(model, "loader_mode", "strict_real_gguf")?;
    require_bool_eq(model, "fallback_loader_used", false)?;

    let tokenizer = require_object(receipt, "tokenizer")?;
    require_string_eq(tokenizer, "source", "explicit")?;
    require_bool_eq(tokenizer, "strict", true)?;
    require_string_eq(tokenizer, "pretokenizer_authority", "llama-bpe")?;

    let prompt_template = require_object(receipt, "prompt_template")?;
    require_string_eq(prompt_template, "family", "bitnetcpp-answer")?;
    require_non_empty_string(prompt_template, "rendered_sha256")?;

    let workload = require_object(receipt, "workload")?;
    require_string_eq(workload, "profile", "strict_ask_math_8")?;
    require_non_empty_string(workload, "question")?;
    let answer = require_string(workload, "answer")?;
    if answer.trim() != "4" {
        return Err(validation_error(format!("workload.answer must trim to 4, got {answer:?}")));
    }
    require_u64_at_least(workload, "prompt_tokens", 1)?;
    require_u64_at_least(workload, "generated_tokens", 1)?;
    require_bool_eq(workload, "quality_passed", true)?;
    require_bool_eq(workload, "cpu_cuda_answer_match", true)?;
    require_bool_eq(workload, "cpu_cuda_generated_ids_match", true)?;

    let repeat_policy = require_object(receipt, "repeat_policy")?;
    let runs_per_backend = require_u64(repeat_policy, "runs_per_backend")?;
    if runs_per_backend < 2 {
        return Err(validation_error(format!(
            "runs_per_backend must be >= 2, got {runs_per_backend}"
        )));
    }
    require_bool_eq(repeat_policy, "same_model", true)?;
    require_bool_eq(repeat_policy, "same_tokenizer", true)?;
    require_bool_eq(repeat_policy, "same_prompt_template", true)?;
    require_bool_eq(repeat_policy, "same_question", true)?;
    require_bool_eq(repeat_policy, "same_sampling_policy", true)?;
    require_bool_eq(repeat_policy, "fallback_free", true)?;
    require_non_empty_string(repeat_policy, "cold_warm_split")?;
    require_bool_eq(repeat_policy, "speedup_claim", false)?;

    let benchmark = require_object(receipt, "benchmark")?;
    require_string_eq(benchmark, "profile", "strict_ask_math_8")?;
    require_string_eq(benchmark, "cpu_reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(benchmark, "cuda_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_u64_eq(benchmark, "runs_per_backend", runs_per_backend)?;
    require_non_negative_number(benchmark, "cpu_avx512_median_total_ms")?;
    require_non_negative_number(benchmark, "cuda_median_total_ms")?;
    require_non_negative_number(benchmark, "observed_median_cpu_total_ms_div_cuda_total_ms")?;
    require_bool_eq(benchmark, "cpu_cuda_answer_match", true)?;
    require_bool_eq(benchmark, "speedup_claim", false)?;
    require_bool_eq(benchmark, "benchmark_qualified_speedup", false)?;

    let summary = require_object(receipt, "summary")?;
    validate_repeated_backend_summary(
        require_object(summary, "cpu_avx512")?,
        "amd-9950x3d-cpu-avx512",
        "cpu",
        runs_per_backend,
        false,
    )?;
    validate_repeated_backend_summary(
        require_object(summary, "cuda")?,
        "nvidia-rtx-5070-ti-cuda",
        "cuda",
        runs_per_backend,
        true,
    )?;

    let runs = require_array(receipt, "runs")?;
    let mut cpu_runs = 0;
    let mut cuda_runs = 0;
    for run in runs {
        validate_repeated_ask_run(run)?;
        match require_string(run, "backend")? {
            "amd-9950x3d-cpu-avx512" => cpu_runs += 1,
            "nvidia-rtx-5070-ti-cuda" => cuda_runs += 1,
            other => {
                return Err(validation_error(format!("unexpected repeated run backend {other}")));
            }
        }
    }
    if cpu_runs != runs_per_backend || cuda_runs != runs_per_backend {
        return Err(validation_error(format!(
            "runs must contain {runs_per_backend} CPU and {runs_per_backend} CUDA entries, got {cpu_runs} CPU and {cuda_runs} CUDA"
        )));
    }

    let pair_contracts = require_array(receipt, "pair_contracts")?;
    if pair_contracts.len() != runs_per_backend as usize {
        return Err(validation_error(format!(
            "pair_contracts must contain {runs_per_backend} entries"
        )));
    }
    for pair in pair_contracts {
        require_u64_at_least(pair, "repeat_index", 1)?;
        for field in [
            "same_model",
            "same_tokenizer",
            "same_prompt_template",
            "same_question",
            "same_sampling_policy",
            "same_generated_token_ids",
            "same_answer",
            "fallback_free",
        ] {
            require_bool_eq(pair, field, true)?;
        }
    }

    let cuda = require_object(receipt, "cuda")?;
    require_bool_eq(cuda, "available", true)?;
    require_u64_at_least(cuda, "device_count", 1)?;
    require_u64(cuda, "device_index")?;
    let device_name = require_string(cuda, "device_name")?;
    if !is_rtx5070ti_device_name(device_name) {
        return Err(validation_error(format!(
            "cuda.device_name must identify NVIDIA GeForce RTX 5070 Ti, got {device_name}"
        )));
    }
    require_string_eq(cuda, "compute_capability", "12.0")?;
    require_non_empty_string(cuda, "driver_version")?;
    require_non_empty_string(cuda, "cuda_runtime_version")?;
    require_non_empty_string(cuda, "cuda_toolkit_version")?;
    require_non_empty_string(cuda, "nvrtc_version")?;
    require_u64_at_least(cuda, "vram_bytes", 1)?;
    require_u64_at_least(cuda, "memory_hwm_bytes", 1)?;
    require_u64_at_least(cuda, "cuda_kernel_invocations", 1)?;

    let stats = receipt
        .get("kernel_stats")
        .and_then(serde_json::Value::as_array)
        .and_then(|items| items.first())
        .ok_or_else(|| validation_error("kernel_stats must contain at least one entry"))?;
    require_string_eq(stats, "kernel_id", "qk256_gemv_cuda")?;
    require_u64_at_least(stats, "invocations", 1)?;
    require_u64_eq(stats, "fallback_invocations", 0)?;
    require_u64_at_least(stats, "kernel_launches", 1)?;
    require_non_negative_number(stats, "kernel_time_ms")?;
    require_u64_at_least(stats, "host_to_device_bytes", 1)?;
    require_u64_at_least(stats, "device_to_host_bytes", 1)?;

    let residency = require_object(receipt, "cuda_execution_residency")?;
    require_bool_eq(residency, "speedup_claim", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    let transfer = require_object(residency, "host_device_transfer_accounting")?;
    require_string_eq(transfer, "status", "qk256_measured")?;
    require_u64_at_least(transfer, "host_to_device_bytes", 1)?;
    require_u64_at_least(transfer, "device_to_host_bytes", 1)?;
    require_non_negative_number(transfer, "kernel_time_ms")?;

    let boundaries = require_array(receipt, "claim_boundaries")?;
    if boundaries.is_empty() {
        return Err(validation_error("claim_boundaries must not be empty"));
    }

    Ok(())
}

/// Validate a repeated strict CUDA ask benchmark receipt file.
pub fn validate_strict_cuda_repeated_ask_benchmark_receipt_file(
    path: &Path,
) -> Result<(), ReceiptError> {
    let receipt = serde_json::from_slice(&std::fs::read(path)?)?;
    validate_strict_cuda_repeated_ask_benchmark_receipt_json(&receipt)
}

/// Validate a repeated strict CUDA warm-session benchmark receipt for the RTX 5070 Ti lane.
///
/// This receipt qualifies the CUDA warm-session path with repeated
/// same-model multi-turn sessions. It is CUDA-only baseline evidence and keeps
/// `speedup_claim=false`; CPU/CUDA speedup acceptance belongs to a later
/// benchmark qualification review.
pub fn validate_strict_cuda_warm_session_benchmark_receipt_json(
    receipt: &serde_json::Value,
) -> Result<(), ReceiptError> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", "strict_cuda_warm_session_benchmark")?;
    require_string_eq(receipt, "machine_id", "windows-9950x3d-rtx5070ti")?;
    require_string_eq(receipt, "hardware_lane", "nvidia_rtx_5070_ti_cuda")?;
    require_string_eq(receipt, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "runtime_api", "cuda")?;
    require_string_eq(receipt, "claim", "strict_cuda_warm_session_benchmark_baseline")?;
    require_bool_eq(receipt, "fallback_used", false)?;
    require_bool_eq(receipt, "speedup_claim", false)?;
    require_bool_eq(receipt, "benchmark_qualified_speedup", false)?;
    require_bool_eq(receipt, "full_cuda_residency_claimed", false)?;
    require_null(receipt, "fallback_backend")?;
    require_null(receipt, "fallback_reason")?;
    validate_bitnet_qk256_execution_plan(require_object(receipt, "execution_plan")?)?;

    let model = require_object(receipt, "model")?;
    require_string_eq(model, "repo", "microsoft/bitnet-b1.58-2B-4T-gguf")?;
    require_string_eq(model, "file", "ggml-model-i2_s.gguf")?;
    require_non_empty_string(model, "sha256")?;
    require_string_eq(model, "loader_mode", "strict_real_gguf")?;
    require_bool_eq(model, "fallback_loader_used", false)?;

    let tokenizer = require_object(receipt, "tokenizer")?;
    require_string_eq(tokenizer, "source", "explicit")?;
    require_bool_eq(tokenizer, "strict", true)?;
    require_string_eq(tokenizer, "pretokenizer_authority", "llama-bpe")?;

    let generation = require_object(receipt, "generation")?;
    require_string_eq(generation, "prompt_template", "bitnetcpp-answer")?;
    require_string_eq(generation, "mode", "greedy")?;
    require_bool_eq(generation, "deterministic", true)?;
    require_non_negative_number(generation, "temperature")?;
    require_u64_at_least(generation, "max_new_tokens", 1)?;

    let session_contract = require_object(receipt, "session_contract")?;
    let runs_per_backend = require_u64(session_contract, "runs_per_backend")?;
    if runs_per_backend < 2 {
        return Err(validation_error(format!(
            "runs_per_backend must be >= 2, got {runs_per_backend}"
        )));
    }
    let turn_count = require_u64(session_contract, "turn_count")?;
    if turn_count < 2 {
        return Err(validation_error(format!("turn_count must be >= 2, got {turn_count}")));
    }
    require_bool_eq(session_contract, "same_model", true)?;
    require_bool_eq(session_contract, "same_tokenizer", true)?;
    require_bool_eq(session_contract, "same_prompts", true)?;
    require_bool_eq(session_contract, "same_sampling_policy", true)?;
    require_bool_eq(session_contract, "fallback_free", true)?;
    require_bool_eq(session_contract, "model_loaded_once", true)?;
    require_bool_eq(session_contract, "tokenizer_loaded_once", true)?;
    require_bool_eq(session_contract, "cuda_context_initialized_once", true)?;
    require_bool_eq(session_contract, "qk256_weights_uploaded_once", true)?;
    require_bool_eq(session_contract, "per_token_weight_upload", false)?;
    require_bool_eq(session_contract, "kv_cache_reuse_claimed", false)?;
    require_bool_eq(session_contract, "speedup_claim", false)?;

    let workload = require_object(receipt, "workload")?;
    require_string_eq(workload, "profile", "strict_cuda_warm_session_2_turns")?;
    require_u64_eq(workload, "turn_count", turn_count)?;
    require_u64_at_least(workload, "generated_tokens_total", 1)?;
    require_u64_at_least(workload, "prompt_tokens_total", 1)?;
    require_bool_eq(workload, "quality_passed", true)?;
    let prompts = require_array(workload, "prompts")?;
    if prompts.len() != turn_count as usize {
        return Err(validation_error(format!(
            "workload.prompts must contain {turn_count} entries"
        )));
    }
    for prompt in prompts {
        require_u64_at_least(prompt, "turn_index", 1)?;
        require_non_empty_string(prompt, "prompt")?;
        require_non_empty_string(prompt, "expected_answer_scope")?;
    }
    let answers = require_array(workload, "answers")?;
    if answers.len() != turn_count as usize {
        return Err(validation_error(format!(
            "workload.answers must contain {turn_count} entries"
        )));
    }
    let first_answer = answers
        .first()
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| validation_error("workload.answers[0] must be a string"))?;
    if first_answer.trim() != "4" {
        return Err(validation_error(format!(
            "first warm-session answer must trim to 4, got {first_answer:?}"
        )));
    }
    for answer in answers {
        let answer = answer
            .as_str()
            .ok_or_else(|| validation_error("workload.answers entries must be strings"))?;
        if answer.trim().is_empty() {
            return Err(validation_error("workload.answers entries must not be empty"));
        }
    }

    let benchmark = require_object(receipt, "benchmark")?;
    require_string_eq(benchmark, "profile", "strict_cuda_warm_session_2_turns")?;
    require_string_eq(benchmark, "cuda_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_u64_eq(benchmark, "runs_per_backend", runs_per_backend)?;
    require_u64_eq(benchmark, "turns_per_run", turn_count)?;
    require_non_negative_number(benchmark, "cuda_median_total_session_ms")?;
    require_non_negative_number(benchmark, "cuda_median_kernel_time_ms")?;
    require_non_negative_number(benchmark, "cuda_median_generated_tokens_per_second")?;
    require_u64_at_least(benchmark, "cuda_median_host_to_device_bytes", 1)?;
    require_u64_at_least(benchmark, "cuda_median_device_to_host_bytes", 1)?;
    require_bool_eq(benchmark, "quality_passed", true)?;
    require_bool_eq(benchmark, "speedup_claim", false)?;
    require_bool_eq(benchmark, "benchmark_qualified_speedup", false)?;

    let summary = require_object(receipt, "summary")?;
    require_string_eq(summary, "backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(summary, "runtime_api", "cuda")?;
    require_u64_eq(summary, "runs", runs_per_backend)?;
    require_bool_eq(summary, "quality_passed", true)?;
    require_bool_eq(summary, "fallback_used", false)?;
    for metric in [
        "total_session_ms",
        "model_load_ms",
        "tokenizer_load_ms",
        "cuda_probe_ms",
        "kernel_time_ms",
        "generated_tokens_per_second",
    ] {
        validate_metric_summary(require_object(summary, metric)?, runs_per_backend)?;
    }
    for metric in ["host_to_device_bytes", "device_to_host_bytes", "memory_hwm_bytes"] {
        validate_u64_summary(require_object(summary, metric)?, runs_per_backend)?;
    }

    let runs = require_array(receipt, "runs")?;
    if runs.len() != runs_per_backend as usize {
        return Err(validation_error(format!("runs must contain {runs_per_backend} entries")));
    }
    for run in runs {
        validate_warm_session_run(run, turn_count)?;
    }

    let cuda = require_object(receipt, "cuda")?;
    require_bool_eq(cuda, "available", true)?;
    require_u64_at_least(cuda, "device_count", 1)?;
    require_u64(cuda, "device_index")?;
    let device_name = require_string(cuda, "device_name")?;
    if !is_rtx5070ti_device_name(device_name) {
        return Err(validation_error(format!(
            "cuda.device_name must identify NVIDIA GeForce RTX 5070 Ti, got {device_name}"
        )));
    }
    require_string_eq(cuda, "compute_capability", "12.0")?;
    require_non_empty_string(cuda, "driver_version")?;
    require_non_empty_string(cuda, "cuda_runtime_version")?;
    require_non_empty_string(cuda, "cuda_toolkit_version")?;
    require_non_empty_string(cuda, "nvrtc_version")?;
    require_u64_at_least(cuda, "vram_bytes", 1)?;
    require_u64_at_least(cuda, "memory_hwm_bytes", 1)?;
    require_u64_at_least(cuda, "cuda_kernel_invocations", 1)?;

    let stats = receipt
        .get("kernel_stats")
        .and_then(serde_json::Value::as_array)
        .and_then(|items| items.first())
        .ok_or_else(|| validation_error("kernel_stats must contain at least one entry"))?;
    require_string_eq(stats, "kernel_id", "qk256_gemv_cuda")?;
    require_u64_at_least(stats, "invocations", 1)?;
    require_u64_eq(stats, "fallback_invocations", 0)?;
    require_u64_at_least(stats, "kernel_launches", 1)?;
    require_non_negative_number(stats, "kernel_time_ms")?;
    require_u64_at_least(stats, "host_to_device_bytes", 1)?;
    require_u64_at_least(stats, "device_to_host_bytes", 1)?;

    let residency = require_object(receipt, "cuda_execution_residency")?;
    require_bool_eq(residency, "speedup_claim", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    let transfer = require_object(residency, "host_device_transfer_accounting")?;
    require_string_eq(transfer, "status", "qk256_measured")?;
    require_u64_at_least(transfer, "host_to_device_bytes", 1)?;
    require_u64_at_least(transfer, "device_to_host_bytes", 1)?;
    require_non_negative_number(transfer, "kernel_time_ms")?;

    let boundaries = require_array(receipt, "claim_boundaries")?;
    if boundaries.is_empty() {
        return Err(validation_error("claim_boundaries must not be empty"));
    }

    Ok(())
}

/// Validate a repeated strict CUDA warm-session benchmark receipt file.
pub fn validate_strict_cuda_warm_session_benchmark_receipt_file(
    path: &Path,
) -> Result<(), ReceiptError> {
    let receipt = serde_json::from_slice(&std::fs::read(path)?)?;
    validate_strict_cuda_warm_session_benchmark_receipt_json(&receipt)
}

/// Validate a benchmark qualification review receipt for the RTX 5070 Ti lane.
///
/// This receipt reviews existing repeated strict ask and warm-session evidence.
/// It is deliberately conservative: the only accepted decision in this schema
/// is to keep `speedup_claim=false` until profile-specific benchmark evidence
/// satisfies every recorded requirement.
pub fn validate_strict_cuda_benchmark_qualification_receipt_json(
    receipt: &serde_json::Value,
) -> Result<(), ReceiptError> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", "strict_cuda_benchmark_qualification_review")?;
    require_string_eq(receipt, "machine_id", "windows-9950x3d-rtx5070ti")?;
    require_string_eq(receipt, "hardware_lane", "nvidia_rtx_5070_ti_cuda")?;
    require_string_eq(receipt, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(receipt, "runtime_api", "cuda")?;
    require_string_eq(receipt, "claim", "strict_cuda_benchmark_qualification_review")?;
    require_bool_eq(receipt, "fallback_used", false)?;
    require_bool_eq(receipt, "speedup_claim", false)?;
    require_bool_eq(receipt, "benchmark_qualified_speedup", false)?;
    require_bool_eq(receipt, "full_cuda_residency_claimed", false)?;

    let model = require_object(receipt, "model")?;
    require_string_eq(model, "repo", "microsoft/bitnet-b1.58-2B-4T-gguf")?;
    require_string_eq(model, "file", "ggml-model-i2_s.gguf")?;
    require_non_empty_string(model, "sha256")?;
    require_string_eq(model, "loader_mode", "strict_real_gguf")?;
    require_bool_eq(model, "fallback_loader_used", false)?;

    let tokenizer = require_object(receipt, "tokenizer")?;
    require_string_eq(tokenizer, "source", "explicit")?;
    require_bool_eq(tokenizer, "strict", true)?;
    require_string_eq(tokenizer, "pretokenizer_authority", "llama-bpe")?;

    let prompt_template = require_object(receipt, "prompt_template")?;
    require_string_eq(prompt_template, "family", "bitnetcpp-answer")?;
    require_non_empty_string(prompt_template, "rendered_sha256")?;

    let proof_inputs = require_object(receipt, "proof_inputs")?;
    require_non_empty_string(proof_inputs, "repeated_strict_ask_receipt")?;
    require_non_empty_string(proof_inputs, "warm_session_benchmark_receipt")?;
    require_non_empty_string(proof_inputs, "answer_path_baseline_receipt")?;
    require_non_empty_string(proof_inputs, "cpu_cuda_answer_parity_receipt")?;

    let decision = require_object(receipt, "qualification_decision")?;
    require_string_eq(decision, "status", "not_accepted")?;
    require_bool_eq(decision, "speedup_claim_allowed", false)?;
    require_bool_eq(decision, "benchmark_qualified_speedup", false)?;
    require_non_empty_string(decision, "reason")?;
    let accepted_profiles = require_array(decision, "accepted_profiles")?;
    if !accepted_profiles.is_empty() {
        return Err(validation_error(
            "qualification_decision.accepted_profiles must be empty while status is not_accepted",
        ));
    }
    let blocked_profiles = require_array(decision, "blocked_profiles")?;
    if blocked_profiles.is_empty() {
        return Err(validation_error("qualification_decision.blocked_profiles must not be empty"));
    }

    let requirements = require_array(receipt, "qualification_requirements")?;
    if requirements.is_empty() {
        return Err(validation_error("qualification_requirements must not be empty"));
    }
    let mut blocked_requirements = 0;
    for requirement in requirements {
        let status = validate_qualification_requirement(requirement)?;
        if status == "blocked" {
            blocked_requirements += 1;
        }
    }
    if blocked_requirements == 0 {
        return Err(validation_error(
            "qualification_requirements must include at least one blocked requirement",
        ));
    }

    let profile_reviews = require_array(receipt, "profile_reviews")?;
    if profile_reviews.len() < 2 {
        return Err(validation_error(
            "profile_reviews must include strict ask and warm-session entries",
        ));
    }
    for profile in profile_reviews {
        validate_qualification_profile_review(profile)?;
    }

    let evidence = require_object(receipt, "evidence_summary")?;
    if receipt.get("target_profiles").is_some() {
        validate_strict_cuda_product_qualification_profiles(
            receipt,
            decision,
            profile_reviews,
            evidence,
        )?;
    } else {
        let strict_ask = require_object(evidence, "strict_ask_math_8")?;
        require_u64_at_least(strict_ask, "runs_per_backend", 2)?;
        require_bool_eq(strict_ask, "cpu_cuda_answer_match", true)?;
        require_bool_eq(strict_ask, "fallback_free", true)?;
        require_non_negative_number(strict_ask, "observed_median_cpu_total_ms_div_cuda_total_ms")?;
        require_bool_eq(strict_ask, "speedup_claim", false)?;
        let warm_session = require_object(evidence, "strict_cuda_warm_session_2_turns")?;
        require_u64_at_least(warm_session, "cuda_runs", 2)?;
        require_bool_eq(warm_session, "fallback_free", true)?;
        require_bool_eq(warm_session, "model_tokenizer_context_loaded_once", true)?;
        require_bool_eq(warm_session, "qk256_weights_uploaded_once", true)?;
        require_bool_eq(warm_session, "speedup_claim", false)?;
    }

    let cuda = require_object(receipt, "cuda")?;
    require_bool_eq(cuda, "available", true)?;
    require_u64_at_least(cuda, "device_count", 1)?;
    require_u64(cuda, "device_index")?;
    let device_name = require_string(cuda, "device_name")?;
    if !is_rtx5070ti_device_name(device_name) {
        return Err(validation_error(format!(
            "cuda.device_name must identify NVIDIA GeForce RTX 5070 Ti, got {device_name}"
        )));
    }
    require_string_eq(cuda, "compute_capability", "12.0")?;
    require_non_empty_string(cuda, "driver_version")?;
    require_non_empty_string(cuda, "cuda_runtime_version")?;
    require_non_empty_string(cuda, "cuda_toolkit_version")?;
    require_non_empty_string(cuda, "nvrtc_version")?;
    require_u64_at_least(cuda, "vram_bytes", 1)?;

    let boundaries = require_array(receipt, "claim_boundaries")?;
    if boundaries.is_empty() {
        return Err(validation_error("claim_boundaries must not be empty"));
    }

    Ok(())
}

/// Validate a benchmark qualification review receipt file.
pub fn validate_strict_cuda_benchmark_qualification_receipt_file(
    path: &Path,
) -> Result<(), ReceiptError> {
    let receipt = serde_json::from_slice(&std::fs::read(path)?)?;
    validate_strict_cuda_benchmark_qualification_receipt_json(&receipt)
}

/// Validate a dense Qwen CUDA benchmark baseline receipt.
///
/// This is baseline evidence only. It normalizes the existing one-token,
/// short-decode, and warm-session strict CUDA proof receipts for Qwen2.5 0.5B
/// Q8_0 without accepting any profile-specific speedup claim.
pub fn validate_dense_gguf_qwen_cuda_benchmark_baseline_receipt_json(
    receipt: &serde_json::Value,
) -> Result<(), ReceiptError> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", "dense_gguf_qwen_cuda_benchmark_baseline")?;
    require_string_eq(receipt, "machine_id", "windows-9950x3d-rtx5070ti")?;
    require_string_eq(receipt, "hardware_lane", "nvidia_rtx_5070_ti_cuda")?;
    require_string_eq(receipt, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(receipt, "runtime_api", "cuda")?;
    require_string_eq(receipt, "selected_route", "dense_regular_llm_cuda")?;
    require_string_eq(receipt, "claim", "dense_gguf_qwen_cuda_benchmark_baseline")?;
    require_bool_eq(receipt, "fallback_used", false)?;
    require_null(receipt, "fallback_backend")?;
    require_null(receipt, "fallback_reason")?;
    require_bool_eq(receipt, "speedup_claim", false)?;
    require_bool_eq(receipt, "benchmark_qualified_speedup", false)?;
    require_bool_eq(receipt, "full_cuda_residency_claimed", false)?;
    require_bool_eq(receipt, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(receipt, "server_ready_claimed", false)?;
    require_bool_eq(receipt, "bitnet_packed_i2s_qk256_proof", false)?;

    let claim_boundary = require_object(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_gguf_qwen_cuda_benchmark_baseline_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_warm_session_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "benchmark_qualified_speedup", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "server_ready_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;

    let model = require_object(receipt, "model")?;
    require_string_eq(model, "id", "qwen2.5-0.5b-instruct-q8_0")?;
    require_string_eq(model, "model_family", "qwen")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_string_eq(model, "file", "qwen2.5-0.5b-instruct-q8_0.gguf")?;
    require_non_empty_string(model, "sha256")?;

    let tokenizer_prompt_authority = require_object(receipt, "tokenizer_prompt_authority")?;
    require_string_eq(tokenizer_prompt_authority, "tokenizer_authority", "contract_authoritative")?;
    require_string_eq(tokenizer_prompt_authority, "prompt_authority", "contract_authoritative")?;
    require_non_empty_string(tokenizer_prompt_authority, "prompt_template")?;
    require_non_empty_string(tokenizer_prompt_authority, "rendered_prompt_sha256")?;
    require_non_empty_string(tokenizer_prompt_authority, "prompt_token_ids_sha256")?;
    require_u64_at_least(tokenizer_prompt_authority, "prompt_token_count", 1)?;

    let execution_plan = require_object(receipt, "execution_plan")?;
    require_string_eq(execution_plan, "selected_route", "dense_regular_llm_cuda")?;
    require_string_eq(execution_plan, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(execution_plan, "runtime_api", "cuda")?;
    require_string_eq(execution_plan, "strict_fallback_policy", "reject")?;
    require_bool_eq(execution_plan, "dense_regular_llm_cuda", true)?;
    require_bool_eq(execution_plan, "bitnet_packed_qk256_cuda", false)?;
    require_bool_eq(execution_plan, "fallback_used", false)?;
    require_bool_eq(execution_plan, "strict_cuda_ready", true)?;
    require_bool_eq(execution_plan, "speedup_claim", false)?;
    require_bool_eq(execution_plan, "full_cuda_residency_claimed", false)?;
    require_u64_at_least(execution_plan, "cuda_dense_regular_llm_ops", 1)?;
    require_u64_eq(execution_plan, "cuda_bitnet_qk256_ops", 0)?;
    require_u64_eq(execution_plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(execution_plan, "unsupported_ops", 0)?;

    let proof_inputs = require_object(receipt, "proof_inputs")?;
    validate_dense_qwen_proof_input(
        proof_inputs,
        "one_token",
        "dense_gguf_qwen_one_token_strict_cuda_proof",
    )?;
    validate_dense_qwen_proof_input(
        proof_inputs,
        "short_decode",
        "dense_gguf_qwen_short_decode_strict_cuda_proof",
    )?;
    validate_dense_qwen_proof_input(
        proof_inputs,
        "warm_session",
        "dense_gguf_qwen_warm_session_strict_cuda_proof",
    )?;

    let profiles = require_array(receipt, "profiles")?;
    for expected in ["one_token", "short_decode_8", "warm_session_3_turns"] {
        let profile = profiles
            .iter()
            .find(|entry| {
                entry.get("profile").and_then(serde_json::Value::as_str) == Some(expected)
            })
            .ok_or_else(|| validation_error(format!("profiles missing {expected}")))?;
        validate_dense_qwen_benchmark_profile(profile, expected)?;
    }

    let kernel_summary = require_object(receipt, "kernel_summary")?;
    require_u64_at_least(kernel_summary, "total_kernel_invocations", 1)?;
    require_u64_at_least(kernel_summary, "total_kernel_launches", 1)?;
    require_non_negative_number(kernel_summary, "total_kernel_time_ms")?;
    require_u64_at_least(kernel_summary, "total_host_to_device_bytes", 1)?;
    require_u64_at_least(kernel_summary, "total_device_to_host_bytes", 1)?;
    require_u64_eq(kernel_summary, "total_cpu_fallback_invocations", 0)?;
    require_bool_eq(kernel_summary, "fallback_used", false)?;

    let benchmark_summary = require_object(receipt, "benchmark_summary")?;
    require_string_eq(benchmark_summary, "status", "baseline_only")?;
    require_bool_eq(benchmark_summary, "speedup_claim_allowed", false)?;
    require_bool_eq(benchmark_summary, "benchmark_qualified_speedup", false)?;
    require_u64_at_least(benchmark_summary, "profiles_recorded", 3)?;
    require_non_empty_string(benchmark_summary, "next_step")?;
    let accepted_profiles = require_array(benchmark_summary, "accepted_speedup_profiles")?;
    if !accepted_profiles.is_empty() {
        return Err(validation_error("benchmark_summary.accepted_speedup_profiles must be empty"));
    }
    let blockers = require_array(benchmark_summary, "qualification_blockers")?;
    if blockers.is_empty() {
        return Err(validation_error("benchmark_summary.qualification_blockers must not be empty"));
    }

    let cuda = require_object(receipt, "cuda")?;
    require_bool_eq(cuda, "available", true)?;
    require_u64_at_least(cuda, "device_count", 1)?;
    let device_name = require_string(cuda, "device_name")?;
    if !is_rtx5070ti_device_name(device_name) {
        return Err(validation_error(format!(
            "cuda.device_name must identify NVIDIA GeForce RTX 5070 Ti, got {device_name}"
        )));
    }
    require_string_eq(cuda, "compute_capability", "12.0")?;
    require_non_empty_string(cuda, "driver_version")?;
    require_non_empty_string(cuda, "cuda_runtime_version")?;
    require_non_empty_string(cuda, "cuda_toolkit_version")?;
    require_non_empty_string(cuda, "nvrtc_version")?;
    require_u64_at_least(cuda, "vram_bytes", 1)?;

    let claim_boundaries = require_array(receipt, "claim_boundaries")?;
    if claim_boundaries.is_empty() {
        return Err(validation_error("claim_boundaries must not be empty"));
    }

    Ok(())
}

/// Validate a dense Qwen CUDA benchmark baseline receipt file.
pub fn validate_dense_gguf_qwen_cuda_benchmark_baseline_receipt_file(
    path: &Path,
) -> Result<(), ReceiptError> {
    let receipt = serde_json::from_slice(&std::fs::read(path)?)?;
    validate_dense_gguf_qwen_cuda_benchmark_baseline_receipt_json(&receipt)
}

/// Validate a repeated dense Qwen CPU/CUDA comparator receipt.
///
/// This is still benchmark evidence only. It requires repeated fallback-free
/// same-artifact CPU/CUDA comparator runs for the dense Qwen CUDA baseline
/// profiles while explicitly preserving speedup and full-residency non-claims.
pub fn validate_dense_gguf_qwen_repeated_comparator_receipt_json(
    receipt: &serde_json::Value,
) -> Result<(), ReceiptError> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", "dense_gguf_qwen_repeated_comparator")?;
    require_string_eq(receipt, "machine_id", "windows-9950x3d-rtx5070ti")?;
    require_string_eq(receipt, "hardware_lane", "nvidia_rtx_5070_ti_cuda")?;
    require_string_eq(receipt, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(receipt, "runtime_api", "cuda")?;
    require_string_eq(receipt, "selected_route", "dense_regular_llm_cuda")?;
    require_string_eq(receipt, "claim", "dense_gguf_qwen_repeated_comparator")?;
    require_bool_eq(receipt, "fallback_used", false)?;
    require_null(receipt, "fallback_backend")?;
    require_null(receipt, "fallback_reason")?;
    require_bool_eq(receipt, "speedup_claim", false)?;
    require_bool_eq(receipt, "benchmark_qualified_speedup", false)?;
    require_bool_eq(receipt, "full_cuda_residency_claimed", false)?;
    require_bool_eq(receipt, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(receipt, "server_ready_claimed", false)?;
    require_bool_eq(receipt, "bitnet_packed_i2s_qk256_proof", false)?;

    let claim_boundary = require_object(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_gguf_qwen_repeated_comparator_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_warm_session_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "benchmark_qualified_speedup", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "server_ready_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;

    let model = require_object(receipt, "model")?;
    require_string_eq(model, "id", QWEN25_05B_INSTRUCT_Q8_0_MODEL_ID)?;
    require_string_eq(model, "model_family", "qwen")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_string_eq(model, "file", QWEN25_05B_INSTRUCT_Q8_0_MODEL_FILE)?;
    require_string_eq(model, "sha256", QWEN25_05B_INSTRUCT_Q8_0_MODEL_SHA256)?;

    let tokenizer_prompt_authority = require_object(receipt, "tokenizer_prompt_authority")?;
    require_string_eq(tokenizer_prompt_authority, "tokenizer_authority", "contract_authoritative")?;
    require_string_eq(tokenizer_prompt_authority, "prompt_authority", "contract_authoritative")?;
    require_string_eq(
        tokenizer_prompt_authority,
        "prompt_template",
        "qwen-chat-raw-deterministic",
    )?;
    require_bool_eq(tokenizer_prompt_authority, "deterministic_prompt", true)?;
    require_u64_at_least(tokenizer_prompt_authority, "prompt_token_count", 1)?;
    require_non_empty_string(tokenizer_prompt_authority, "rendered_prompt_sha256")?;
    require_non_empty_string(tokenizer_prompt_authority, "prompt_token_ids_sha256")?;

    let execution_plan = require_object(receipt, "execution_plan")?;
    require_string_eq(execution_plan, "selected_route", "dense_regular_llm_cuda")?;
    require_string_eq(execution_plan, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(execution_plan, "runtime_api", "cuda")?;
    require_string_eq(execution_plan, "strict_fallback_policy", "reject")?;
    require_bool_eq(execution_plan, "dense_regular_llm_cuda", true)?;
    require_bool_eq(execution_plan, "bitnet_packed_qk256_cuda", false)?;
    require_bool_eq(execution_plan, "fallback_used", false)?;
    require_bool_eq(execution_plan, "strict_cuda_ready", true)?;
    require_bool_eq(execution_plan, "speedup_claim", false)?;
    require_bool_eq(execution_plan, "full_cuda_residency_claimed", false)?;
    require_u64_at_least(execution_plan, "cuda_dense_regular_llm_ops", 1)?;
    require_u64_eq(execution_plan, "cuda_bitnet_qk256_ops", 0)?;
    require_u64_eq(execution_plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(execution_plan, "unsupported_ops", 0)?;

    let baseline_input = require_object(receipt, "baseline_input")?;
    require_non_empty_string(baseline_input, "path")?;
    require_non_empty_string(baseline_input, "sha256")?;
    require_string_eq(baseline_input, "artifact_kind", "dense_gguf_qwen_cuda_benchmark_baseline")?;

    let profiles = require_array(receipt, "profiles")?;
    for expected in ["one_token", "short_decode_8", "warm_session_3_turns"] {
        let profile = profiles
            .iter()
            .find(|entry| {
                entry.get("profile").and_then(serde_json::Value::as_str) == Some(expected)
            })
            .ok_or_else(|| validation_error(format!("profiles missing {expected}")))?;
        validate_dense_qwen_repeated_comparator_profile(profile, expected)?;
    }

    let comparator_summary = require_object(receipt, "comparator_summary")?;
    require_string_eq(comparator_summary, "status", "repeated_comparator_only")?;
    require_u64_at_least(comparator_summary, "profiles_recorded", 3)?;
    require_u64_at_least(comparator_summary, "min_runs_per_backend", 3)?;
    require_u64_at_least(comparator_summary, "total_cpu_runs", 9)?;
    require_u64_at_least(comparator_summary, "total_cuda_runs", 9)?;
    require_bool_eq(comparator_summary, "fallback_free", true)?;
    require_bool_eq(comparator_summary, "same_artifact_sha", true)?;
    require_bool_eq(comparator_summary, "same_tokenizer_prompt_authority", true)?;
    require_bool_eq(comparator_summary, "deterministic_generation_policy", true)?;
    require_bool_eq(comparator_summary, "generated_tokens_compared", true)?;
    require_bool_eq(comparator_summary, "speedup_claim_allowed", false)?;
    require_bool_eq(comparator_summary, "benchmark_qualified_speedup", false)?;
    let accepted_profiles = require_array(comparator_summary, "accepted_speedup_profiles")?;
    if !accepted_profiles.is_empty() {
        return Err(validation_error("comparator_summary.accepted_speedup_profiles must be empty"));
    }
    let blockers = require_array(comparator_summary, "remaining_qualification_blockers")?;
    if blockers.is_empty() {
        return Err(validation_error(
            "comparator_summary.remaining_qualification_blockers must not be empty",
        ));
    }

    let transfer_timing = require_object(receipt, "transfer_timing")?;
    require_non_empty_string(transfer_timing, "status")?;
    require_non_empty_string(transfer_timing, "source")?;
    require_bool_eq(transfer_timing, "host_to_device_bytes_recorded", true)?;
    require_bool_eq(transfer_timing, "device_to_host_bytes_recorded", true)?;
    require_bool_eq(transfer_timing, "host_to_device_timing_recorded", false)?;
    require_bool_eq(transfer_timing, "device_to_host_timing_recorded", false)?;

    let hardware_context = require_object(receipt, "hardware_context")?;
    require_u64_at_least(hardware_context, "vram_bytes", 1)?;
    require_non_negative_number(hardware_context, "power_draw_watts_min")?;
    require_non_negative_number(hardware_context, "power_draw_watts_max")?;
    require_non_negative_number(hardware_context, "temperature_c_min")?;
    require_non_negative_number(hardware_context, "temperature_c_max")?;
    require_non_empty_string(hardware_context, "source")?;

    let cuda = require_object(receipt, "cuda")?;
    require_bool_eq(cuda, "available", true)?;
    require_u64_at_least(cuda, "device_count", 1)?;
    let device_name = require_string(cuda, "device_name")?;
    if !is_rtx5070ti_device_name(device_name) {
        return Err(validation_error(format!(
            "cuda.device_name must identify NVIDIA GeForce RTX 5070 Ti, got {device_name}"
        )));
    }
    require_string_eq(cuda, "compute_capability", "12.0")?;
    require_non_empty_string(cuda, "driver_version")?;
    require_non_empty_string(cuda, "cuda_runtime_version")?;
    require_non_empty_string(cuda, "cuda_toolkit_version")?;
    require_non_empty_string(cuda, "nvrtc_version")?;
    require_u64_at_least(cuda, "vram_bytes", 1)?;

    let claim_boundaries = require_array(receipt, "claim_boundaries")?;
    if claim_boundaries.is_empty() {
        return Err(validation_error("claim_boundaries must not be empty"));
    }

    Ok(())
}

/// Validate a repeated dense Qwen CPU/CUDA comparator receipt file.
pub fn validate_dense_gguf_qwen_repeated_comparator_receipt_file(
    path: &Path,
) -> Result<(), ReceiptError> {
    let receipt = serde_json::from_slice(&std::fs::read(path)?)?;
    validate_dense_gguf_qwen_repeated_comparator_receipt_json(&receipt)
}

/// Validate a Qwen3 repeated dense CPU/CUDA comparator receipt.
///
/// This is benchmark-baseline evidence only. It requires repeated
/// fallback-free same-artifact CPU/CUDA comparator runs for the exact Qwen3
/// product profiles queued by CUDA-MODEL-015 while explicitly preserving
/// speedup, benchmark-qualified, full-residency, broad dense GGUF, and BitNet
/// packed proof non-claims.
pub fn validate_qwen3_cuda_repeated_comparator_receipt_json(
    receipt: &serde_json::Value,
) -> Result<(), ReceiptError> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", "qwen3_cuda_repeated_comparator")?;
    require_string_eq(receipt, "machine_id", "windows-9950x3d-rtx5070ti")?;
    require_string_eq(receipt, "hardware_lane", "nvidia_rtx_5070_ti_cuda")?;
    require_string_eq(receipt, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(receipt, "runtime_api", "cuda")?;
    require_string_eq(receipt, "selected_route", "dense_regular_llm_cuda")?;
    require_string_eq(receipt, "claim", "qwen3_cuda_repeated_comparator")?;
    require_bool_eq(receipt, "fallback_used", false)?;
    require_null(receipt, "fallback_backend")?;
    require_null(receipt, "fallback_reason")?;
    require_bool_eq(receipt, "speedup_claim", false)?;
    require_bool_eq(receipt, "benchmark_qualified_speedup", false)?;
    require_bool_eq(receipt, "full_cuda_residency_claimed", false)?;
    require_bool_eq(receipt, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(receipt, "broad_dense_gguf_ready_claimed", false)?;
    require_bool_eq(receipt, "qwen25_proof_inherited", false)?;
    require_bool_eq(receipt, "server_ready_claimed", false)?;
    require_bool_eq(receipt, "bitnet_packed_i2s_qk256_proof", false)?;

    let claim_boundary = require_object(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "qwen3_cuda_repeated_comparator_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_warm_session_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "server_ready_claimed", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "benchmark_qualified_speedup", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "broad_dense_gguf_ready_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen25_proof_inherited", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;

    let model = require_object(receipt, "model")?;
    validate_dense_qwen_benchmark_model(model)?;
    require_string_eq(model, "id", QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID)?;

    let tokenizer_prompt_authority = require_object(receipt, "tokenizer_prompt_authority")?;
    require_string_eq(tokenizer_prompt_authority, "tokenizer_authority", "contract_authoritative")?;
    require_string_eq(tokenizer_prompt_authority, "prompt_authority", "contract_authoritative")?;
    require_string_eq(
        tokenizer_prompt_authority,
        "prompt_template",
        "qwen-chat-raw-deterministic",
    )?;
    require_bool_eq(tokenizer_prompt_authority, "deterministic_prompt", true)?;
    require_non_empty_string(tokenizer_prompt_authority, "prompt_policy")?;

    let execution_plan = require_object(receipt, "execution_plan")?;
    require_string_eq(execution_plan, "selected_route", "dense_regular_llm_cuda")?;
    require_string_eq(execution_plan, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(execution_plan, "runtime_api", "cuda")?;
    require_string_eq(execution_plan, "strict_fallback_policy", "reject")?;
    require_bool_eq(execution_plan, "dense_regular_llm_cuda", true)?;
    require_bool_eq(execution_plan, "bitnet_packed_qk256_cuda", false)?;
    require_bool_eq(execution_plan, "fallback_used", false)?;
    require_bool_eq(execution_plan, "strict_cuda_ready", true)?;
    require_bool_eq(execution_plan, "speedup_claim", false)?;
    require_bool_eq(execution_plan, "full_cuda_residency_claimed", false)?;
    require_u64_at_least(execution_plan, "cuda_dense_regular_llm_ops", 1)?;
    require_u64_eq(execution_plan, "cuda_bitnet_qk256_ops", 0)?;
    require_u64_eq(execution_plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(execution_plan, "unsupported_ops", 0)?;

    let proof_inputs = require_object(receipt, "proof_inputs")?;
    for expected in QWEN3_REPEATED_COMPARATOR_PROFILES {
        validate_dense_qwen_proof_input(proof_inputs, expected, "qwen3_profile_repeated_runs")?;
    }

    let profiles = require_array(receipt, "profiles")?;
    let expected_profile_count = QWEN3_REPEATED_COMPARATOR_PROFILES.len();
    if profiles.len() != expected_profile_count {
        return Err(validation_error(format!(
            "profiles must contain exactly {expected_profile_count} Qwen3 repeated comparator profiles"
        )));
    }
    for expected in QWEN3_REPEATED_COMPARATOR_PROFILES {
        let profile = profiles
            .iter()
            .find(|entry| {
                entry.get("profile").and_then(serde_json::Value::as_str) == Some(*expected)
            })
            .ok_or_else(|| validation_error(format!("profiles missing {expected}")))?;
        validate_qwen3_repeated_comparator_profile(profile, expected)?;
    }

    let comparator_summary = require_object(receipt, "comparator_summary")?;
    require_string_eq(comparator_summary, "status", "repeated_comparator_only")?;
    require_u64_eq(comparator_summary, "profiles_recorded", expected_profile_count as u64)?;
    require_u64_at_least(comparator_summary, "min_runs_per_backend", 3)?;
    require_u64_at_least(comparator_summary, "total_cpu_runs", 15)?;
    require_u64_at_least(comparator_summary, "total_cuda_runs", 15)?;
    require_bool_eq(comparator_summary, "fallback_free", true)?;
    require_bool_eq(comparator_summary, "same_artifact_sha", true)?;
    require_bool_eq(comparator_summary, "same_tokenizer_prompt_policy", true)?;
    require_bool_eq(comparator_summary, "deterministic_generation_policy", true)?;
    require_bool_eq(comparator_summary, "generated_tokens_compared", true)?;
    require_bool_eq(comparator_summary, "speedup_claim_allowed", false)?;
    require_bool_eq(comparator_summary, "benchmark_qualified_speedup", false)?;
    let accepted_profiles = require_array(comparator_summary, "accepted_speedup_profiles")?;
    if !accepted_profiles.is_empty() {
        return Err(validation_error("comparator_summary.accepted_speedup_profiles must be empty"));
    }
    let blockers = require_array(comparator_summary, "remaining_qualification_blockers")?;
    if blockers.is_empty() {
        return Err(validation_error(
            "comparator_summary.remaining_qualification_blockers must not be empty",
        ));
    }

    let transfer_timing = require_object(receipt, "transfer_timing")?;
    require_non_empty_string(transfer_timing, "status")?;
    require_non_empty_string(transfer_timing, "source")?;
    require_bool_eq(transfer_timing, "host_to_device_bytes_recorded", true)?;
    require_bool_eq(transfer_timing, "device_to_host_bytes_recorded", true)?;
    require_bool(transfer_timing, "host_to_device_timing_recorded")?;
    require_bool(transfer_timing, "device_to_host_timing_recorded")?;
    require_bool(transfer_timing, "pure_host_to_device_timing_recorded")?;

    let hardware_context = require_object(receipt, "hardware_context")?;
    require_u64_at_least(hardware_context, "vram_bytes", 1)?;
    require_non_negative_number(hardware_context, "power_draw_watts_min")?;
    require_non_negative_number(hardware_context, "power_draw_watts_max")?;
    require_non_negative_number(hardware_context, "temperature_c_min")?;
    require_non_negative_number(hardware_context, "temperature_c_max")?;
    require_non_empty_string(hardware_context, "source")?;

    let cuda = require_object(receipt, "cuda")?;
    require_bool_eq(cuda, "available", true)?;
    require_u64_at_least(cuda, "device_count", 1)?;
    let device_name = require_string(cuda, "device_name")?;
    if !is_rtx5070ti_device_name(device_name) {
        return Err(validation_error(format!(
            "cuda.device_name must identify NVIDIA GeForce RTX 5070 Ti, got {device_name}"
        )));
    }
    require_string_eq(cuda, "compute_capability", "12.0")?;
    require_non_empty_string(cuda, "driver_version")?;
    require_non_empty_string(cuda, "cuda_runtime_version")?;
    require_non_empty_string(cuda, "cuda_toolkit_version")?;
    require_non_empty_string(cuda, "nvrtc_version")?;
    require_u64_at_least(cuda, "vram_bytes", 1)?;

    let claim_boundaries = require_array(receipt, "claim_boundaries")?;
    if claim_boundaries.is_empty() {
        return Err(validation_error("claim_boundaries must not be empty"));
    }

    Ok(())
}

/// Validate a Qwen3 repeated dense CPU/CUDA comparator receipt file.
pub fn validate_qwen3_cuda_repeated_comparator_receipt_file(
    path: &Path,
) -> Result<(), ReceiptError> {
    let receipt = serde_json::from_slice(&std::fs::read(path)?)?;
    validate_qwen3_cuda_repeated_comparator_receipt_json(&receipt)
}

/// Validate a dense Qwen benchmark qualification review receipt.
///
/// This receipt reviews the dense Qwen CPU/CUDA comparator and transfer-timing
/// evidence without accepting a profile-specific speedup claim. The validator
/// requires at least one blocked qualification requirement so a review cannot
/// silently upgrade the dense CUDA lane without a future schema.
pub fn validate_dense_gguf_qwen_benchmark_qualification_receipt_json(
    receipt: &serde_json::Value,
) -> Result<(), ReceiptError> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", "dense_gguf_qwen_benchmark_qualification_review")?;
    require_string_eq(receipt, "machine_id", "windows-9950x3d-rtx5070ti")?;
    require_string_eq(receipt, "hardware_lane", "nvidia_rtx_5070_ti_cuda")?;
    require_string_eq(receipt, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(receipt, "runtime_api", "cuda")?;
    require_string_eq(receipt, "selected_route", "dense_regular_llm_cuda")?;
    require_string_eq(receipt, "claim", "dense_gguf_qwen_benchmark_qualification_review")?;
    require_bool_eq(receipt, "fallback_used", false)?;
    require_null(receipt, "fallback_backend")?;
    require_null(receipt, "fallback_reason")?;
    require_bool_eq(receipt, "speedup_claim", false)?;
    require_bool_eq(receipt, "benchmark_qualified_speedup", false)?;
    require_bool_eq(receipt, "full_cuda_residency_claimed", false)?;
    require_bool_eq(receipt, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(receipt, "server_ready_claimed", false)?;
    require_bool_eq(receipt, "bitnet_packed_i2s_qk256_proof", false)?;

    let claim_boundary = require_object(receipt, "claim_boundary")?;
    require_bool_eq(
        claim_boundary,
        "dense_gguf_qwen_benchmark_qualification_review_claimed",
        true,
    )?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_warm_session_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "benchmark_qualified_speedup", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "server_ready_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;

    let model = require_object(receipt, "model")?;
    let benchmark_model = validate_dense_qwen_benchmark_model(model)?;

    let tokenizer_prompt_authority = require_object(receipt, "tokenizer_prompt_authority")?;
    require_string_eq(tokenizer_prompt_authority, "tokenizer_authority", "contract_authoritative")?;
    require_string_eq(tokenizer_prompt_authority, "prompt_authority", "contract_authoritative")?;
    require_string_eq(
        tokenizer_prompt_authority,
        "prompt_template",
        "qwen-chat-raw-deterministic",
    )?;
    require_bool_eq(tokenizer_prompt_authority, "deterministic_prompt", true)?;
    require_u64_at_least(tokenizer_prompt_authority, "prompt_token_count", 1)?;
    require_non_empty_string(tokenizer_prompt_authority, "rendered_prompt_sha256")?;
    require_non_empty_string(tokenizer_prompt_authority, "prompt_token_ids_sha256")?;

    let execution_plan = require_object(receipt, "execution_plan")?;
    require_string_eq(execution_plan, "selected_route", "dense_regular_llm_cuda")?;
    require_string_eq(execution_plan, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(execution_plan, "runtime_api", "cuda")?;
    require_string_eq(execution_plan, "strict_fallback_policy", "reject")?;
    require_bool_eq(execution_plan, "dense_regular_llm_cuda", true)?;
    require_bool_eq(execution_plan, "bitnet_packed_qk256_cuda", false)?;
    require_bool_eq(execution_plan, "fallback_used", false)?;
    require_bool_eq(execution_plan, "strict_cuda_ready", true)?;
    require_bool_eq(execution_plan, "speedup_claim", false)?;
    require_bool_eq(execution_plan, "full_cuda_residency_claimed", false)?;
    require_u64_at_least(execution_plan, "cuda_dense_regular_llm_ops", 1)?;
    require_u64_eq(execution_plan, "cuda_bitnet_qk256_ops", 0)?;
    require_u64_eq(execution_plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(execution_plan, "unsupported_ops", 0)?;

    let proof_inputs = require_object(receipt, "proof_inputs")?;
    if benchmark_model.repeated_comparator_required() {
        validate_dense_qwen_proof_input(
            proof_inputs,
            "benchmark_baseline",
            "dense_gguf_qwen_cuda_benchmark_baseline",
        )?;
        validate_dense_qwen_proof_input(
            proof_inputs,
            "repeated_comparator",
            "dense_gguf_qwen_repeated_comparator",
        )?;
    } else {
        validate_missing_dense_qwen_proof_input(proof_inputs, "benchmark_baseline")?;
        validate_missing_dense_qwen_proof_input(proof_inputs, "repeated_comparator")?;
    }
    validate_dense_qwen_proof_input(
        proof_inputs,
        "one_token_transfer_timing",
        "dense_gguf_qwen_one_token_strict_cuda_proof",
    )?;
    validate_dense_qwen_proof_input(
        proof_inputs,
        "short_decode_transfer_timing",
        "dense_gguf_qwen_short_decode_strict_cuda_proof",
    )?;
    validate_dense_qwen_proof_input(
        proof_inputs,
        "warm_session_transfer_timing",
        "dense_gguf_qwen_warm_session_strict_cuda_proof",
    )?;

    let decision = require_object(receipt, "qualification_decision")?;
    require_string_eq(decision, "status", "not_accepted")?;
    require_bool_eq(decision, "speedup_claim_allowed", false)?;
    require_bool_eq(decision, "benchmark_qualified_speedup", false)?;
    require_non_empty_string(decision, "reason")?;
    let accepted_profiles = require_array(decision, "accepted_profiles")?;
    if !accepted_profiles.is_empty() {
        return Err(validation_error(
            "qualification_decision.accepted_profiles must be empty while status is not_accepted",
        ));
    }
    let blocked_profiles = require_array(decision, "blocked_profiles")?;
    if blocked_profiles.len() < 3 {
        return Err(validation_error(
            "qualification_decision.blocked_profiles must include every dense profile",
        ));
    }

    let requirements = require_array(receipt, "qualification_requirements")?;
    if requirements.is_empty() {
        return Err(validation_error("qualification_requirements must not be empty"));
    }
    let mut blocked_requirements = 0;
    for requirement in requirements {
        let status = validate_qualification_requirement(requirement)?;
        if status == "blocked" {
            blocked_requirements += 1;
        }
    }
    if blocked_requirements == 0 {
        return Err(validation_error(
            "qualification_requirements must include at least one blocked requirement",
        ));
    }

    let profile_reviews = require_array(receipt, "profile_reviews")?;
    for expected in ["one_token", "short_decode_8", "warm_session_3_turns"] {
        let profile = profile_reviews
            .iter()
            .find(|entry| {
                entry.get("profile").and_then(serde_json::Value::as_str) == Some(expected)
            })
            .ok_or_else(|| validation_error(format!("profile_reviews missing {expected}")))?;
        validate_dense_qwen_qualification_profile_review(
            profile,
            benchmark_model.min_runs_per_backend(),
        )?;
    }

    let evidence = require_object(receipt, "evidence_summary")?;
    for expected in ["one_token", "short_decode_8", "warm_session_3_turns"] {
        let profile = require_object(evidence, expected)?;
        require_u64_at_least(profile, "runs_per_backend", benchmark_model.min_runs_per_backend())?;
        if benchmark_model.min_runs_per_backend() < 3 {
            require_bool_eq(profile, "repeated_evidence", false)?;
        }
        require_bool_eq(profile, "fallback_free", true)?;
        require_bool_eq(profile, "quality_passed", true)?;
        require_bool_eq(profile, "generated_token_ids_match", true)?;
        require_non_negative_number(profile, "cpu_total_ms_mean")?;
        require_non_negative_number(profile, "cuda_total_ms_mean")?;
        require_non_negative_number(profile, "observed_cpu_total_ms_div_cuda_total_ms")?;
        require_bool_eq(profile, "cuda_mean_slower_than_cpu", true)?;
        require_non_negative_number(profile, "device_to_host_ms")?;
        validate_dense_qwen_qualification_h2d(profile)?;
        require_bool_eq(profile, "speedup_claim", false)?;
        require_bool_eq(profile, "benchmark_qualified_speedup", false)?;
    }

    let transfer_timing = require_object(receipt, "transfer_timing_review")?;
    let transfer_status = require_string(transfer_timing, "status")?;
    require_bool_eq(transfer_timing, "device_to_host_timing_recorded", true)?;
    match transfer_status {
        "device_to_host_measured_host_to_device_unmeasured" => {
            require_bool_eq(transfer_timing, "host_to_device_timing_recorded", false)?;
            require_non_empty_string(transfer_timing, "host_to_device_blocker")?;
        }
        "host_to_device_model_load_envelope_device_to_host_measured" => {
            require_bool_eq(transfer_timing, "host_to_device_timing_recorded", true)?;
            require_bool_eq(transfer_timing, "host_to_device_model_load_envelope_recorded", true)?;
            require_bool_eq(
                transfer_timing,
                "host_to_device_pure_transfer_timing_recorded",
                false,
            )?;
            require_non_empty_string(transfer_timing, "host_to_device_blocker")?;
            require_string_eq(
                transfer_timing,
                "host_to_device_source",
                "wall_clock_model_load_with_cuda_weight_upload",
            )?;
            require_string_eq(
                transfer_timing,
                "host_to_device_scope",
                "model_load_wall_clock_envelope",
            )?;
            require_bool_eq(
                transfer_timing,
                "host_to_device_ms_includes_non_transfer_overhead",
                true,
            )?;
        }
        other => {
            return Err(validation_error(format!(
                "transfer_timing_review.status must be a supported dense Qwen transfer timing status, got {other}"
            )));
        }
    }

    let hardware_context = require_object(receipt, "hardware_context")?;
    require_u64_at_least(hardware_context, "vram_bytes", 1)?;
    require_non_negative_number(hardware_context, "power_draw_watts_min")?;
    require_non_negative_number(hardware_context, "power_draw_watts_max")?;
    require_non_negative_number(hardware_context, "temperature_c_min")?;
    require_non_negative_number(hardware_context, "temperature_c_max")?;
    require_non_empty_string(hardware_context, "source")?;

    let cuda = require_object(receipt, "cuda")?;
    require_bool_eq(cuda, "available", true)?;
    require_u64_at_least(cuda, "device_count", 1)?;
    let device_name = require_string(cuda, "device_name")?;
    if !is_rtx5070ti_device_name(device_name) {
        return Err(validation_error(format!(
            "cuda.device_name must identify NVIDIA GeForce RTX 5070 Ti, got {device_name}"
        )));
    }
    require_string_eq(cuda, "compute_capability", "12.0")?;
    require_non_empty_string(cuda, "driver_version")?;
    require_non_empty_string(cuda, "cuda_runtime_version")?;
    require_non_empty_string(cuda, "cuda_toolkit_version")?;
    require_non_empty_string(cuda, "nvrtc_version")?;
    require_u64_at_least(cuda, "vram_bytes", 1)?;

    let claim_boundaries = require_array(receipt, "claim_boundaries")?;
    if claim_boundaries.is_empty() {
        return Err(validation_error("claim_boundaries must not be empty"));
    }

    Ok(())
}

/// Validate a dense Qwen benchmark qualification review receipt file.
pub fn validate_dense_gguf_qwen_benchmark_qualification_receipt_file(
    path: &Path,
) -> Result<(), ReceiptError> {
    let receipt = serde_json::from_slice(&std::fs::read(path)?)?;
    validate_dense_gguf_qwen_benchmark_qualification_receipt_json(&receipt)
}

fn validate_dense_qwen_benchmark_model(
    model: &serde_json::Value,
) -> Result<DenseQwenBenchmarkModel, ReceiptError> {
    require_string_eq(model, "model_family", "qwen")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    match require_string(model, "id")? {
        QWEN25_05B_INSTRUCT_Q8_0_MODEL_ID => {
            require_string_eq(model, "file", QWEN25_05B_INSTRUCT_Q8_0_MODEL_FILE)?;
            require_string_eq(model, "sha256", QWEN25_05B_INSTRUCT_Q8_0_MODEL_SHA256)?;
            Ok(DenseQwenBenchmarkModel::Qwen25)
        }
        QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID => {
            require_string_eq(model, "file", QWEN3_06B_INSTRUCT_Q8_0_MODEL_FILE)?;
            require_string_eq(model, "architecture", "qwen3")?;
            require_string_eq(model, "sha256", QWEN3_06B_INSTRUCT_Q8_0_MODEL_SHA256)?;
            Ok(DenseQwenBenchmarkModel::Qwen3)
        }
        other => Err(validation_error(format!(
            "model.id must be a verified dense Qwen benchmark model, got {other}"
        ))),
    }
}

fn validate_dense_qwen_repeated_comparator_profile(
    profile: &serde_json::Value,
    expected_profile: &str,
) -> Result<(), ReceiptError> {
    require_string_eq(profile, "profile", expected_profile)?;
    require_string_eq(profile, "status", "repeated_same_artifact_cpu_cuda_comparator")?;
    require_string_eq(profile, "cpu_reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(profile, "cuda_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(profile, "runtime_api", "cuda")?;
    require_string_eq(profile, "selected_route", "dense_regular_llm_cuda")?;
    require_u64_at_least(profile, "run_count", 3)?;
    require_u64_at_least(profile, "cpu_runs", 3)?;
    require_u64_at_least(profile, "cuda_runs", 3)?;
    require_u64_at_least(profile, "min_runs_per_backend", 3)?;
    require_bool_eq(profile, "fallback_free", true)?;
    require_bool_eq(profile, "same_artifact_sha", true)?;
    require_bool_eq(profile, "same_tokenizer_prompt_authority", true)?;
    require_bool_eq(profile, "deterministic_generation_policy", true)?;
    require_bool_eq(profile, "generated_token_ids_match", true)?;
    require_bool_eq(profile, "speedup_claim", false)?;
    require_bool_eq(profile, "benchmark_qualified_speedup", false)?;
    require_bool_eq(profile, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(profile, "full_cuda_residency_claimed", false)?;
    require_non_empty_string(profile, "transfer_timing_status")?;

    validate_dense_qwen_metric_summary(require_object(profile, "cpu_total_ms")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "cuda_total_ms")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "first_token_ms")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "decode_total_ms")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "kernel_time_ms")?)?;
    validate_dense_qwen_u64_summary(require_object(profile, "host_to_device_bytes")?)?;
    validate_dense_qwen_u64_summary(require_object(profile, "device_to_host_bytes")?)?;

    let runs = require_array(profile, "runs")?;
    if runs.len() < 3 {
        return Err(validation_error(format!(
            "{expected_profile}.runs must contain at least 3 runs"
        )));
    }
    let mut paths = std::collections::BTreeSet::new();
    for run in runs {
        validate_dense_qwen_repeated_comparator_run(run, expected_profile)?;
        let path = require_string(run, "source_receipt_path")?;
        if !paths.insert(path.to_owned()) {
            return Err(validation_error(format!(
                "{expected_profile}.runs source_receipt_path values must be unique"
            )));
        }
    }
    Ok(())
}

fn validate_dense_qwen_metric_summary(summary: &serde_json::Value) -> Result<(), ReceiptError> {
    require_u64_at_least(summary, "count", 3)?;
    require_non_negative_number(summary, "min")?;
    require_non_negative_number(summary, "mean")?;
    require_non_negative_number(summary, "max")?;
    Ok(())
}

fn validate_dense_qwen_u64_summary(summary: &serde_json::Value) -> Result<(), ReceiptError> {
    require_u64_at_least(summary, "count", 3)?;
    require_u64_at_least(summary, "min", 1)?;
    require_u64_at_least(summary, "max", 1)?;
    require_non_negative_number(summary, "mean")?;
    Ok(())
}

fn validate_dense_qwen_repeated_comparator_run(
    run: &serde_json::Value,
    expected_profile: &str,
) -> Result<(), ReceiptError> {
    require_non_empty_string(run, "run_id")?;
    require_string_eq(run, "profile", expected_profile)?;
    require_non_empty_string(run, "source_receipt_path")?;
    require_non_empty_string(run, "source_receipt_sha256")?;
    require_non_empty_string(run, "source_artifact_kind")?;
    require_string_eq(
        run,
        "model_sha256",
        "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
    )?;
    require_string_eq(run, "prompt_template", "qwen-chat-raw-deterministic")?;
    require_string_eq(run, "generation_policy", "greedy")?;
    require_bool_eq(run, "deterministic_generation", true)?;
    require_bool_eq(run, "fallback_used", false)?;
    require_bool_eq(run, "quality_passed", true)?;
    require_bool_eq(run, "parity_passed", true)?;
    require_bool_eq(run, "generated_token_ids_match", true)?;
    require_non_empty_string(run, "generated_token_ids_sha256")?;
    require_non_empty_string(run, "first_divergence_report")?;
    require_bool_eq(run, "speedup_claim", false)?;
    require_bool_eq(run, "benchmark_qualified_speedup", false)?;
    require_bool_eq(run, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(run, "full_cuda_residency_claimed", false)?;

    let timing = require_object(run, "timing")?;
    require_non_negative_number(timing, "cpu_total_ms")?;
    require_non_negative_number(timing, "cuda_total_ms")?;
    require_non_negative_number(timing, "first_token_ms")?;
    require_non_negative_number(timing, "decode_total_ms")?;
    require_non_negative_number(timing, "kernel_time_ms")?;
    require_u64_at_least(timing, "kernel_invocations", 1)?;
    require_u64_at_least(timing, "kernel_launches", 1)?;
    require_u64_at_least(timing, "host_to_device_bytes", 1)?;
    require_u64_at_least(timing, "device_to_host_bytes", 1)?;
    require_nullable_number_with_source(timing, "host_to_device_ms")?;
    require_nullable_number_with_source(timing, "device_to_host_ms")?;

    if expected_profile == "warm_session_3_turns" {
        require_u64_at_least(run, "turns_count", 3)?;
    }

    Ok(())
}

fn validate_qwen3_repeated_comparator_profile(
    profile: &serde_json::Value,
    expected_profile: &str,
) -> Result<(), ReceiptError> {
    require_string_eq(profile, "profile", expected_profile)?;
    require_string_eq(profile, "status", "repeated_same_artifact_cpu_cuda_comparator")?;
    require_string_eq(profile, "cpu_reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(profile, "cuda_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(profile, "runtime_api", "cuda")?;
    require_string_eq(profile, "selected_route", "dense_regular_llm_cuda")?;
    require_u64_at_least(profile, "run_count", 3)?;
    require_u64_at_least(profile, "cpu_runs", 3)?;
    require_u64_at_least(profile, "cuda_runs", 3)?;
    require_u64_at_least(profile, "min_runs_per_backend", 3)?;
    require_bool_eq(profile, "fallback_free", true)?;
    require_bool_eq(profile, "same_artifact_sha", true)?;
    require_bool_eq(profile, "same_tokenizer_prompt_policy", true)?;
    require_bool_eq(profile, "deterministic_generation_policy", true)?;
    require_bool_eq(profile, "generated_token_ids_match", true)?;
    require_bool_eq(profile, "speedup_claim", false)?;
    require_bool_eq(profile, "benchmark_qualified_speedup", false)?;
    require_bool_eq(profile, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(profile, "full_cuda_residency_claimed", false)?;
    require_bool_eq(profile, "server_ready_claimed", false)?;
    require_non_empty_string(profile, "transfer_timing_status")?;

    validate_dense_qwen_metric_summary(require_object(profile, "model_load_ms")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "tokenizer_load_ms")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "prompt_render_ms")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "tokenize_ms")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "cuda_context_init_ms")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "weight_upload_ms")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "cpu_total_ms")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "cuda_total_ms")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "prefill_ms")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "first_token_ms")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "decode_total_ms")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "steady_tok_per_s")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "kernel_time_ms")?)?;
    validate_dense_qwen_u64_summary(require_object(profile, "launch_count")?)?;
    validate_dense_qwen_u64_summary(require_object(profile, "host_to_device_bytes")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "host_to_device_ms")?)?;
    validate_dense_qwen_u64_summary(require_object(profile, "device_to_host_bytes")?)?;
    validate_dense_qwen_metric_summary(require_object(profile, "device_to_host_ms")?)?;
    validate_dense_qwen_u64_summary(require_object(profile, "vram_high_water_bytes")?)?;

    let runs = require_array(profile, "runs")?;
    if runs.len() < 3 {
        return Err(validation_error(format!(
            "{expected_profile}.runs must contain at least 3 runs"
        )));
    }
    let mut paths = std::collections::BTreeSet::new();
    for run in runs {
        validate_qwen3_repeated_comparator_run(run, expected_profile)?;
        let path = require_string(run, "source_receipt_path")?;
        if !paths.insert(path.to_owned()) {
            return Err(validation_error(format!(
                "{expected_profile}.runs source_receipt_path values must be unique"
            )));
        }
    }
    Ok(())
}

fn validate_qwen3_repeated_comparator_run(
    run: &serde_json::Value,
    expected_profile: &str,
) -> Result<(), ReceiptError> {
    require_non_empty_string(run, "run_id")?;
    require_string_eq(run, "profile", expected_profile)?;
    require_non_empty_string(run, "source_receipt_path")?;
    require_non_empty_string(run, "source_receipt_sha256")?;
    let source_artifact_kind = require_string(run, "source_artifact_kind")?;
    if !source_artifact_kind.starts_with("dense_gguf_qwen_")
        || !source_artifact_kind.ends_with("_strict_cuda_proof")
    {
        return Err(validation_error(format!(
            "{expected_profile}.source_artifact_kind must be a dense Qwen strict CUDA proof, got {source_artifact_kind}"
        )));
    }
    require_string_eq(run, "model_sha256", QWEN3_06B_INSTRUCT_Q8_0_MODEL_SHA256)?;
    require_string_eq(run, "prompt_template", "qwen-chat-raw-deterministic")?;
    require_u64_at_least(run, "prompt_token_count", 1)?;
    require_string_eq(run, "generation_policy", "greedy")?;
    require_bool_eq(run, "deterministic_generation", true)?;
    require_u64_eq(run, "generated_tokens", qwen3_expected_generated_tokens(expected_profile)?)?;
    require_bool_eq(run, "fallback_used", false)?;
    require_bool_eq(run, "quality_passed", true)?;
    require_bool_eq(run, "parity_passed", true)?;
    require_bool_eq(run, "generated_token_ids_match", true)?;
    require_non_empty_string(run, "generated_token_ids_sha256")?;
    require_non_empty_string(run, "first_divergence_report")?;
    require_bool_eq(run, "top_k_compared", true)?;
    require_bool_eq(run, "speedup_claim", false)?;
    require_bool_eq(run, "benchmark_qualified_speedup", false)?;
    require_bool_eq(run, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(run, "full_cuda_residency_claimed", false)?;
    require_bool_eq(run, "server_ready_claimed", false)?;

    if expected_profile == "warm_session_3_turns" {
        require_u64_at_least(run, "turns_count", 3)?;
    }
    if expected_profile == "decode_128_from_warm_context" {
        require_bool_eq(run, "warm_context_reused", true)?;
    }

    let timing = require_object(run, "timing")?;
    require_non_negative_number(timing, "model_load_ms")?;
    require_non_negative_number(timing, "tokenizer_load_ms")?;
    require_non_negative_number(timing, "prompt_render_ms")?;
    require_non_negative_number(timing, "tokenize_ms")?;
    require_non_negative_number(timing, "cuda_context_init_ms")?;
    require_non_negative_number(timing, "weight_upload_ms")?;
    require_non_negative_number(timing, "cpu_total_ms")?;
    require_non_negative_number(timing, "cuda_total_ms")?;
    require_non_negative_number(timing, "prefill_ms")?;
    require_non_negative_number(timing, "first_token_ms")?;
    require_non_negative_number(timing, "decode_total_ms")?;
    require_non_negative_number(timing, "steady_tok_per_s")?;
    require_non_negative_number(timing, "kernel_time_ms")?;
    require_u64_at_least(timing, "launch_count", 1)?;
    require_u64_at_least(timing, "kernel_invocations", 1)?;
    require_u64_at_least(timing, "host_to_device_bytes", 1)?;
    require_nullable_number_with_source(timing, "host_to_device_ms")?;
    require_u64_at_least(timing, "device_to_host_bytes", 1)?;
    require_nullable_number_with_source(timing, "device_to_host_ms")?;
    require_u64_at_least(timing, "vram_high_water_bytes", 1)?;
    require_non_empty_string(timing, "power_temperature_context")?;

    Ok(())
}

fn qwen3_expected_generated_tokens(profile: &str) -> Result<u64, ReceiptError> {
    match profile {
        "one_token" => Ok(1),
        "short_decode_8" => Ok(8),
        "short_decode_32" => Ok(32),
        "warm_session_3_turns" => Ok(24),
        "decode_128_from_warm_context" => Ok(128),
        other => Err(validation_error(format!("unsupported Qwen3 comparator profile {other}"))),
    }
}

fn validate_dense_qwen_proof_input(
    proof_inputs: &serde_json::Value,
    field: &str,
    expected_artifact_kind: &str,
) -> Result<(), ReceiptError> {
    let input = require_object(proof_inputs, field)?;
    require_non_empty_string(input, "path")?;
    require_non_empty_string(input, "sha256")?;
    require_string_eq(input, "artifact_kind", expected_artifact_kind)?;
    Ok(())
}

fn validate_missing_dense_qwen_proof_input(
    proof_inputs: &serde_json::Value,
    field: &str,
) -> Result<(), ReceiptError> {
    let input = require_object(proof_inputs, field)?;
    require_string_eq(input, "status", "missing")?;
    require_non_empty_string(input, "reason")?;
    Ok(())
}

fn validate_dense_qwen_benchmark_profile(
    profile: &serde_json::Value,
    expected_profile: &str,
) -> Result<(), ReceiptError> {
    require_string_eq(profile, "profile", expected_profile)?;
    require_string_eq(profile, "backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(profile, "runtime_api", "cuda")?;
    require_string_eq(profile, "selected_route", "dense_regular_llm_cuda")?;
    require_string_eq(profile, "status", "measured_existing_receipt")?;
    require_non_empty_string(profile, "source_receipt_path")?;
    require_non_empty_string(profile, "source_receipt_sha256")?;
    require_non_empty_string(profile, "source_artifact_kind")?;
    require_bool_eq(profile, "fallback_used", false)?;
    require_bool_eq(profile, "quality_passed", true)?;
    require_bool_eq(profile, "parity_passed", true)?;
    require_bool_eq(profile, "speedup_claim", false)?;
    require_bool_eq(profile, "benchmark_qualified_speedup", false)?;
    require_bool_eq(profile, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(profile, "full_cuda_residency_claimed", false)?;
    require_u64_at_least(profile, "prompt_tokens", 1)?;
    require_u64_at_least(profile, "generated_tokens", 1)?;
    require_non_negative_number(profile, "total_ms")?;
    require_non_negative_number(profile, "first_token_ms")?;
    require_non_negative_number(profile, "decode_total_ms")?;
    require_non_negative_number(profile, "kernel_time_ms")?;
    require_u64_at_least(profile, "kernel_invocations", 1)?;
    require_u64_at_least(profile, "kernel_launches", 1)?;
    require_u64_at_least(profile, "host_to_device_bytes", 1)?;
    require_u64_at_least(profile, "device_to_host_bytes", 1)?;
    require_non_negative_number(profile, "cpu_reference_total_ms")?;
    if expected_profile == "warm_session_3_turns" {
        require_u64_at_least(profile, "turns_count", 3)?;
    }
    Ok(())
}

fn validate_qualification_requirement(
    requirement: &serde_json::Value,
) -> Result<&str, ReceiptError> {
    require_non_empty_string(requirement, "id")?;
    require_non_empty_string(requirement, "description")?;
    let status = require_string(requirement, "status")?;
    match status {
        "passed" | "blocked" | "not_applicable" => {}
        other => {
            return Err(validation_error(format!(
                "qualification requirement status must be passed, blocked, or not_applicable, got {other}"
            )));
        }
    }
    if status == "blocked" {
        require_non_empty_string(requirement, "blocker")?;
    }
    Ok(status)
}

fn validate_qualification_profile_review(profile: &serde_json::Value) -> Result<(), ReceiptError> {
    require_non_empty_string(profile, "profile")?;
    let decision = require_string(profile, "decision")?;
    if decision != "not_accepted" {
        return Err(validation_error(format!(
            "profile review decision must be not_accepted, got {decision}"
        )));
    }
    require_bool_eq(profile, "speedup_claim_allowed", false)?;
    require_bool_eq(profile, "benchmark_qualified_speedup", false)?;
    let evidence_status =
        profile.get("evidence_status").and_then(serde_json::Value::as_str).unwrap_or("reviewed");
    match evidence_status {
        "reviewed" | "baseline_only" | "single_run_baseline" => {
            require_bool_eq(profile, "fallback_free", true)?;
            require_bool_eq(profile, "quality_passed", true)?;
        }
        "missing" => {
            require_bool_eq(profile, "fallback_free", false)?;
            require_bool_eq(profile, "quality_passed", false)?;
        }
        other => {
            return Err(validation_error(format!(
                "profile review evidence_status must be reviewed, baseline_only, single_run_baseline, or missing; got {other}"
            )));
        }
    }
    require_bool_eq(profile, "dense_cuda_evidence_used", false)?;
    require_non_empty_string(profile, "reason")?;
    let blockers = require_array(profile, "blockers")?;
    if blockers.is_empty() {
        return Err(validation_error("profile_review.blockers must not be empty"));
    }
    Ok(())
}

fn validate_strict_cuda_product_qualification_profiles(
    receipt: &serde_json::Value,
    decision: &serde_json::Value,
    profile_reviews: &[serde_json::Value],
    evidence: &serde_json::Value,
) -> Result<(), ReceiptError> {
    let expected_profiles = strict_cuda_expected_product_profiles(receipt)?;

    let target_profiles = require_array(receipt, "target_profiles")?;
    if target_profiles.len() != expected_profiles.len() {
        return Err(validation_error(format!(
            "target_profiles must contain exactly {} profiles",
            expected_profiles.len()
        )));
    }

    let mut seen = std::collections::BTreeSet::new();
    for value in target_profiles {
        let Some(profile) = value.as_str() else {
            return Err(validation_error("target_profiles entries must be strings"));
        };
        if !expected_profiles.contains(&profile) {
            return Err(validation_error(format!("unexpected target profile {profile}")));
        }
        if !seen.insert(profile.to_string()) {
            return Err(validation_error(format!("duplicate target profile {profile}")));
        }
    }

    if profile_reviews.len() != expected_profiles.len() {
        return Err(validation_error(format!(
            "profile_reviews must contain exactly {} product profiles",
            expected_profiles.len()
        )));
    }

    let blocked_profiles = require_array(decision, "blocked_profiles")?;
    if blocked_profiles.len() != expected_profiles.len() {
        return Err(validation_error(format!(
            "blocked_profiles must contain exactly {} product profiles",
            expected_profiles.len()
        )));
    }

    let mut seen_blocked = std::collections::BTreeSet::new();
    for value in blocked_profiles {
        let Some(profile) = value.as_str() else {
            return Err(validation_error("blocked_profiles entries must be strings"));
        };
        if !expected_profiles.contains(&profile) {
            return Err(validation_error(format!("unexpected blocked profile {profile}")));
        }
        if !seen_blocked.insert(profile.to_string()) {
            return Err(validation_error(format!("duplicate blocked profile {profile}")));
        }
    }

    let strict_ask = require_object(evidence, "strict_ask_math_8")?;
    require_u64_at_least(strict_ask, "runs_per_backend", 2)?;
    require_bool_eq(strict_ask, "cpu_cuda_answer_match", true)?;
    require_bool_eq(strict_ask, "fallback_free", true)?;
    require_non_negative_number(strict_ask, "cpu_avx512_median_total_ms")?;
    require_non_negative_number(strict_ask, "cuda_median_total_ms")?;
    require_non_negative_number(strict_ask, "observed_median_cpu_total_ms_div_cuda_total_ms")?;
    require_non_negative_number(strict_ask, "qk256_kernel_time_ms")?;
    require_bool_eq(strict_ask, "speedup_claim", false)?;

    let warm_session = require_object(evidence, "strict_cuda_warm_session_2_turns")?;
    require_u64_at_least(warm_session, "cuda_runs", 2)?;
    require_bool_eq(warm_session, "fallback_free", true)?;
    require_bool_eq(warm_session, "model_tokenizer_context_loaded_once", true)?;
    require_bool_eq(warm_session, "qk256_weights_uploaded_once", true)?;
    require_non_negative_number(warm_session, "cuda_median_kernel_time_ms")?;
    require_non_negative_number(warm_session, "cuda_median_total_session_ms")?;
    require_bool_eq(warm_session, "speedup_claim", false)?;

    for &expected in expected_profiles {
        require_string_array_contains(blocked_profiles, expected, "blocked_profiles")?;

        let review = find_object_by_string_field(profile_reviews, "profile", expected)
            .ok_or_else(|| validation_error(format!("profile_reviews missing {expected}")))?;
        require_string_eq(review, "decision", "not_accepted")?;
        require_bool_eq(review, "speedup_claim_allowed", false)?;
        require_bool_eq(review, "benchmark_qualified_speedup", false)?;
        require_bool_eq(review, "dense_cuda_evidence_used", false)?;

        let evidence_entry = require_object(evidence, expected)?;
        require_string_eq(evidence_entry, "profile", expected)?;
        require_string_eq(evidence_entry, "decision", "not_accepted")?;
        require_bool_eq(evidence_entry, "speedup_claim", false)?;
        require_bool_eq(evidence_entry, "benchmark_qualified_speedup", false)?;
        require_non_empty_string(evidence_entry, "evidence_status")?;
        require_non_empty_string(evidence_entry, "reason")?;
        let blockers = require_array(evidence_entry, "blockers")?;
        if blockers.is_empty() {
            return Err(validation_error(format!(
                "evidence_summary.{expected}.blockers must not be empty"
            )));
        }

        if expected == "short_decode_8" {
            require_string_eq(evidence_entry, "evidence_status", "single_run_baseline")?;
            require_bool_eq(evidence_entry, "fallback_free", true)?;
            require_bool_eq(evidence_entry, "quality_passed", true)?;
            require_bool_eq(evidence_entry, "cpu_cuda_output_match", true)?;
            require_non_negative_number(evidence_entry, "cpu_total_ms")?;
            require_non_negative_number(evidence_entry, "cuda_total_ms")?;
            require_non_negative_number(evidence_entry, "observed_cpu_total_ms_div_cuda_total_ms")?;
        } else {
            require_string_eq(evidence_entry, "evidence_status", "missing")?;
            require_bool_eq(evidence_entry, "fallback_free", false)?;
            require_bool_eq(evidence_entry, "quality_passed", false)?;
        }
    }

    let policy = require_object(receipt, "benchmark_policy")?;
    require_bool_eq(policy, "profile_specific_decisions_only", true)?;
    require_bool_eq(policy, "global_speedup_claim", false)?;
    require_bool_eq(policy, "dense_cuda_evidence_used", false)?;
    require_bool_eq(policy, "bitnet_packed_i2s_qk256_only", true)?;

    Ok(())
}

fn strict_cuda_expected_product_profiles(
    receipt: &serde_json::Value,
) -> Result<&'static [&'static str], ReceiptError> {
    match receipt.get("profile_matrix_id").and_then(serde_json::Value::as_str) {
        Some(STRICT_CUDA_PERF_005_PROFILE_MATRIX_ID) => Ok(STRICT_CUDA_PERF_005_PROFILES),
        Some(other) => Err(validation_error(format!("unsupported profile_matrix_id {other}"))),
        None => Ok(STRICT_CUDA_PRODUCT_PROFILES_V1),
    }
}

fn validate_dense_qwen_qualification_profile_review(
    profile: &serde_json::Value,
    min_runs_per_backend: u64,
) -> Result<(), ReceiptError> {
    require_non_empty_string(profile, "profile")?;
    let decision = require_string(profile, "decision")?;
    if decision != "not_accepted" {
        return Err(validation_error(format!(
            "dense profile review decision must be not_accepted, got {decision}"
        )));
    }
    require_bool_eq(profile, "speedup_claim_allowed", false)?;
    require_bool_eq(profile, "benchmark_qualified_speedup", false)?;
    require_bool_eq(profile, "fallback_free", true)?;
    require_bool_eq(profile, "quality_passed", true)?;
    require_bool_eq(profile, "generated_token_ids_match", true)?;
    require_bool_eq(profile, "dense_cuda_evidence_used", true)?;
    require_u64_at_least(profile, "runs_per_backend", min_runs_per_backend)?;
    if min_runs_per_backend < 3 {
        require_bool_eq(profile, "repeated_evidence", false)?;
    }
    require_non_negative_number(profile, "observed_cpu_total_ms_div_cuda_total_ms")?;
    require_bool_eq(profile, "cuda_mean_slower_than_cpu", true)?;
    require_nullable_number_with_source(profile, "host_to_device_ms")?;
    validate_dense_qwen_qualification_h2d(profile)?;
    require_non_negative_number(profile, "device_to_host_ms")?;
    require_non_empty_string(profile, "device_to_host_ms_source")?;
    require_non_empty_string(profile, "reason")?;
    let blockers = require_array(profile, "blockers")?;
    if blockers.is_empty() {
        return Err(validation_error("dense profile_review.blockers must not be empty"));
    }
    Ok(())
}

fn validate_dense_qwen_qualification_h2d(profile: &serde_json::Value) -> Result<(), ReceiptError> {
    require_non_empty_string(profile, "host_to_device_ms_source")?;
    if profile.get("host_to_device_ms").and_then(serde_json::Value::as_f64).is_some() {
        require_string_eq(
            profile,
            "host_to_device_ms_source",
            "wall_clock_model_load_with_cuda_weight_upload",
        )?;
        require_string_eq(profile, "host_to_device_ms_scope", "model_load_wall_clock_envelope")?;
        require_bool_eq(profile, "host_to_device_ms_includes_non_transfer_overhead", true)?;
        require_null(profile, "pure_host_to_device_ms")?;
        require_non_empty_string(profile, "pure_host_to_device_ms_source")?;
    }
    Ok(())
}

fn validate_warm_session_run(
    run: &serde_json::Value,
    expected_turn_count: u64,
) -> Result<(), ReceiptError> {
    require_string_eq(run, "profile", "strict_cuda_warm_session_2_turns")?;
    require_string_eq(run, "backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(run, "runtime_api", "cuda")?;
    require_string_eq(run, "status", "measured")?;
    require_string_eq(run, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(run, "kernel_id", "qk256_gemv_cuda")?;
    require_u64_at_least(run, "repeat_index", 1)?;
    require_non_empty_string(run, "source_receipt_path")?;
    require_bool_eq(run, "quality_passed", true)?;
    require_bool_eq(run, "fallback_used", false)?;
    require_bool_eq(run, "model_loaded_once", true)?;
    require_bool_eq(run, "tokenizer_loaded_once", true)?;
    require_bool_eq(run, "cuda_context_initialized_once", true)?;
    require_bool_eq(run, "qk256_weights_uploaded_once", true)?;
    require_bool_eq(run, "per_token_weight_upload", false)?;
    require_u64_eq(run, "turn_count", expected_turn_count)?;
    require_u64_at_least(run, "generated_tokens_total", 1)?;
    require_u64_at_least(run, "prompt_tokens_total", 1)?;
    require_non_negative_number(run, "total_session_ms")?;
    require_non_negative_number(run, "model_load_ms")?;
    require_non_negative_number(run, "tokenizer_load_ms")?;
    require_non_negative_number(run, "cuda_probe_ms")?;
    require_non_negative_number(run, "kernel_time_ms")?;
    require_non_negative_number(run, "generated_tokens_per_second")?;
    require_u64_at_least(run, "kernel_invocations", 1)?;
    require_u64_at_least(run, "host_to_device_bytes", 1)?;
    require_u64_at_least(run, "device_to_host_bytes", 1)?;
    require_u64_at_least(run, "memory_hwm_bytes", 1)?;

    let turns = require_array(run, "turns")?;
    if turns.len() != expected_turn_count as usize {
        return Err(validation_error(format!(
            "run.turns must contain {expected_turn_count} entries"
        )));
    }
    for turn in turns {
        require_u64_at_least(turn, "turn_index", 1)?;
        require_non_empty_string(turn, "answer_trimmed")?;
        require_u64_at_least(turn, "generated_tokens", 1)?;
        require_u64_at_least(turn, "prompt_tokens", 1)?;
        require_bool_eq(turn, "quality_passed", true)?;
        require_bool_eq(turn, "fallback_used", false)?;
        require_non_negative_number(turn, "kernel_time_ms")?;
        require_u64_at_least(turn, "host_to_device_bytes", 1)?;
        require_u64_at_least(turn, "device_to_host_bytes", 1)?;
    }

    Ok(())
}

/// Validate a strict CPU BitNet benchmark receipt.
///
/// This validator checks the benchmark evidence contract, not performance
/// quality. It requires every CPU proof benchmark profile to be present and
/// makes selected backend/kernel, fallback state, workload, model identity,
/// quantization format, and CPU context explicit before any benchmark artifact
/// can be treated as evidence.
pub fn validate_strict_cpu_benchmark_receipt_json(
    receipt: &serde_json::Value,
) -> Result<(), ReceiptError> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", "cpu_benchmark")?;
    require_string_eq(receipt, "runtime_api", "cpu")?;
    require_string_eq(receipt, "claim", "cpu_benchmark_receipt")?;
    require_string_eq(receipt, "requested_backend", "cpu")?;
    require_non_empty_string(receipt, "selected_backend")?;
    require_bool_eq(receipt, "fallback_used", false)?;
    require_null(receipt, "fallback_reason")?;
    require_bool_eq(receipt, "speedup_claim", false)?;

    let model = require_object(receipt, "model")?;
    require_non_empty_string(model, "repo")?;
    require_non_empty_string(model, "file")?;
    require_non_empty_string(model, "sha256")?;
    let quant_format = require_string(model, "quant_format")?;
    let quant_lc = quant_format.to_ascii_lowercase();
    if !(quant_lc.contains("i2_s") || quant_lc.contains("qk256")) {
        return Err(validation_error(format!(
            "model.quant_format must identify QK256/I2_S, got {quant_format}"
        )));
    }

    let tokenizer = require_object(receipt, "tokenizer")?;
    require_non_empty_string(tokenizer, "source")?;
    require_bool_eq(tokenizer, "strict", true)?;

    let kernel = require_object(receipt, "kernel")?;
    require_non_empty_string(kernel, "requested_kernel")?;
    require_non_empty_string(kernel, "selected_kernel")?;
    require_string_eq(kernel, "oracle_kernel", "qk256-scalar-gemv")?;
    require_bool_eq(kernel, "fallback_used", false)?;
    require_null(kernel, "fallback_reason")?;
    require_bool_eq(kernel, "dequantizes_before_compute", false)?;

    let cpu = require_object(receipt, "cpu")?;
    require_non_empty_string(cpu, "model")?;
    require_non_empty_string(cpu, "arch")?;
    require_u64_at_least(cpu, "threads", 1)?;
    let features = require_array(cpu, "features")?;
    if features.is_empty() {
        return Err(validation_error("cpu.features must not be empty"));
    }
    let selected_kernel = require_string(kernel, "selected_kernel")?;
    let features_lc: Vec<String> = features
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_ascii_lowercase)
                .ok_or_else(|| validation_error("cpu.features entries must be strings"))
        })
        .collect::<Result<_, _>>()?;
    if selected_kernel.to_ascii_lowercase().contains("avx2")
        && !(features_lc.iter().any(|feature| feature == "avx2")
            && features_lc.iter().any(|feature| feature == "fma"))
    {
        return Err(validation_error(
            "selected AVX2 benchmark kernel requires avx2 and fma CPU features",
        ));
    }

    let workload = require_object(receipt, "workload")?;
    require_u64_at_least(workload, "prompt_tokens", 1)?;
    require_u64_at_least(workload, "generated_tokens", 1)?;
    require_u64_at_least(workload, "batch_size", 1)?;

    let profiles = require_array(receipt, "profiles")?;
    for expected in ["micro", "layer", "prefill", "first_token", "decode"] {
        let profile = profiles
            .iter()
            .find(|entry| {
                entry.get("profile").and_then(serde_json::Value::as_str) == Some(expected)
            })
            .ok_or_else(|| validation_error(format!("profiles missing {expected}")))?;
        validate_cpu_benchmark_profile(profile, expected)?;
    }

    if receipt.get("i2s_microbench").is_some() {
        validate_i2s_microbench(require_object(receipt, "i2s_microbench")?)?;
    }
    if receipt.get("i2s_tiling_thread_matrix").is_some_and(|value| !value.is_null()) {
        validate_i2s_tiling_thread_matrix(require_object(receipt, "i2s_tiling_thread_matrix")?)?;
    }
    if receipt.get("i2s_applied_thread_matrix").is_some_and(|value| !value.is_null()) {
        validate_i2s_applied_thread_matrix(require_object(receipt, "i2s_applied_thread_matrix")?)?;
    }
    if receipt.get("embedding_quantization_evidence").is_some_and(|value| !value.is_null()) {
        validate_embedding_quantization_evidence(require_object(
            receipt,
            "embedding_quantization_evidence",
        )?)?;
    }

    Ok(())
}

/// Validate a strict CPU BitNet benchmark receipt file.
pub fn validate_strict_cpu_benchmark_receipt_file(path: &Path) -> Result<(), ReceiptError> {
    let receipt = serde_json::from_slice(&std::fs::read(path)?)?;
    validate_strict_cpu_benchmark_receipt_json(&receipt)
}

fn validate_cpu_benchmark_profile(
    profile: &serde_json::Value,
    expected_profile: &str,
) -> Result<(), ReceiptError> {
    require_string_eq(profile, "profile", expected_profile)?;
    require_string_eq(profile, "execution_phase", expected_cpu_profile_phase(expected_profile))?;
    require_non_empty_string(profile, "requested_kernel")?;
    require_non_empty_string(profile, "selected_kernel")?;
    require_bool_eq(profile, "fallback_used", false)?;
    require_null(profile, "fallback_reason")?;

    let shape = require_object(profile, "shape")?;
    require_u64_at_least(shape, "rows", 1)?;
    require_u64_at_least(shape, "cols", 1)?;
    require_u64_at_least(shape, "iterations", 1)?;

    let status = require_string(profile, "status")?;
    match status {
        "measured" => {
            require_non_negative_number(profile, "wall_time_ms")?;
            require_non_negative_number(profile, "median_ms")?;
            require_non_negative_number(profile, "p95_ms")?;
            require_non_negative_number(profile, "bandwidth_gbps")?;
            require_non_negative_number(profile, "tokens_per_second")?;
        }
        "not_run" => {
            require_non_empty_string(profile, "reason")?;
        }
        other => {
            return Err(validation_error(format!(
                "profile status must be measured or not_run, got {other}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn expected_cpu_profile_phase(profile: &str) -> &'static str {
    match profile {
        "micro" => "micro_kernel",
        "layer" => "layer_forward",
        "prefill" => "prefill",
        "first_token" => "first_token",
        "decode" => "decode_steady_state",
        _ => "unknown",
    }
}

fn validate_i2s_microbench(microbench: &serde_json::Value) -> Result<(), ReceiptError> {
    require_string_eq(microbench, "artifact_kind", "cpu_bitnet_i2s_microbench")?;
    require_string_eq(microbench, "claim", "i2_s_gemv_gemm_microbench_receipt")?;
    require_string_eq(microbench, "kernel_family", "i2_s_qk256")?;
    require_bool_eq(microbench, "speedup_claim", false)?;
    require_bool_eq(microbench, "fallback_used", false)?;
    require_null(microbench, "fallback_reason")?;

    let profiles = require_array(microbench, "profiles")?;
    for operation in ["gemv", "gemm"] {
        let profile = profiles
            .iter()
            .find(|entry| {
                entry.get("operation").and_then(serde_json::Value::as_str) == Some(operation)
            })
            .ok_or_else(|| {
                validation_error(format!("i2s_microbench.profiles missing {operation}"))
            })?;
        validate_i2s_microbench_profile(profile, operation)?;
    }

    Ok(())
}

fn validate_i2s_microbench_profile(
    profile: &serde_json::Value,
    expected_operation: &str,
) -> Result<(), ReceiptError> {
    require_non_empty_string(profile, "profile")?;
    require_string_eq(profile, "operation", expected_operation)?;
    require_non_empty_string(profile, "execution_phase")?;
    require_string_eq(profile, "status", "measured")?;
    require_non_empty_string(profile, "requested_kernel")?;
    require_non_empty_string(profile, "selected_kernel")?;
    require_bool_eq(profile, "fallback_used", false)?;
    require_null(profile, "fallback_reason")?;

    let shape = require_object(profile, "shape")?;
    require_u64_at_least(shape, "rows", 1)?;
    require_u64_at_least(shape, "cols", 1)?;
    require_u64_at_least(shape, "tokens", 1)?;
    require_u64_at_least(shape, "iterations", 1)?;

    require_non_negative_number(profile, "wall_time_ms")?;
    require_non_negative_number(profile, "median_ms")?;
    require_non_negative_number(profile, "p95_ms")?;
    require_non_negative_number(profile, "bandwidth_gbps")?;
    require_non_negative_number(profile, "tokens_per_second")?;

    Ok(())
}

fn validate_i2s_tiling_thread_matrix(matrix: &serde_json::Value) -> Result<(), ReceiptError> {
    require_string_eq(matrix, "artifact_kind", "cpu_bitnet_i2s_tiling_thread_matrix")?;
    require_string_eq(matrix, "claim", "i2_s_tiling_thread_matrix_receipt")?;
    require_string_eq(matrix, "kernel_family", "i2_s_qk256")?;
    require_bool_eq(matrix, "speedup_claim", false)?;
    require_bool_eq(matrix, "fallback_used", false)?;
    require_null(matrix, "fallback_reason")?;

    let candidate_grid = require_object(matrix, "candidate_grid")?;
    for field in ["parallelism_degrees", "row_blocks", "col_blocks", "thread_counts"] {
        let values = require_array(candidate_grid, field)?;
        if values.is_empty() {
            return Err(validation_error(format!("candidate_grid.{field} must not be empty")));
        }
    }
    require_u64_at_least(candidate_grid, "candidate_count", 1)?;

    let coverage = require_object(matrix, "coverage")?;
    require_string_eq(coverage, "status", "sampled_baseline")?;
    require_u64_at_least(coverage, "measured_candidate_count", 1)?;
    require_u64_at_least(coverage, "full_matrix_candidate_count", 1)?;
    require_bool_eq(coverage, "thread_counts_recorded_not_applied", true)?;
    require_non_empty_string(coverage, "reason")?;

    let runs = require_array(matrix, "measured_runs")?;
    let mut has_gemv = false;
    let mut has_gemm = false;
    for run in runs {
        validate_i2s_tiling_matrix_run(run)?;
        match require_string(run, "operation")? {
            "gemv" => has_gemv = true,
            "gemm" => has_gemm = true,
            other => return Err(validation_error(format!("unexpected tiling operation {other}"))),
        }
    }
    if !has_gemv {
        return Err(validation_error("i2s_tiling_thread_matrix missing gemv run"));
    }
    if !has_gemm {
        return Err(validation_error("i2s_tiling_thread_matrix missing gemm run"));
    }

    Ok(())
}

fn validate_i2s_applied_thread_matrix(matrix: &serde_json::Value) -> Result<(), ReceiptError> {
    require_string_eq(matrix, "work_item", "CPU-BITNET-PERF-003")?;
    require_string_eq(matrix, "artifact_kind", "cpu_bitnet_i2s_applied_thread_matrix")?;
    require_string_eq(matrix, "claim", "i2_s_applied_thread_matrix_receipt")?;
    require_string_eq(matrix, "kernel_family", "i2_s_qk256")?;
    require_bool_eq(matrix, "speedup_claim", false)?;
    require_bool_eq(matrix, "fallback_used", false)?;
    require_null(matrix, "fallback_reason")?;

    let candidate_grid = require_object(matrix, "candidate_grid")?;
    for field in ["parallelism_degrees", "row_blocks", "col_blocks", "thread_counts"] {
        let values = require_array(candidate_grid, field)?;
        if values.is_empty() {
            return Err(validation_error(format!("candidate_grid.{field} must not be empty")));
        }
    }
    require_u64_at_least(candidate_grid, "candidate_count", 1)?;

    let coverage = require_object(matrix, "coverage")?;
    require_string_eq(coverage, "status", "sampled_applied_thread_baseline")?;
    require_u64_at_least(coverage, "measured_candidate_count", 1)?;
    require_u64_at_least(coverage, "full_matrix_candidate_count", 1)?;
    require_bool_eq(coverage, "thread_counts_applied", true)?;
    require_string_eq(coverage, "thread_count_policy", "applied_scoped_threads")?;
    let partitions = require_array(coverage, "thread_partitions")?;
    if partitions.is_empty() {
        return Err(validation_error("coverage.thread_partitions must not be empty"));
    }
    require_non_empty_string(coverage, "reason")?;

    let runs = require_array(matrix, "measured_runs")?;
    let mut has_gemv = false;
    let mut has_gemm = false;
    for run in runs {
        validate_i2s_applied_thread_matrix_run(run)?;
        match require_string(run, "operation")? {
            "gemv" => has_gemv = true,
            "gemm" => has_gemm = true,
            other => {
                return Err(validation_error(format!(
                    "unexpected applied-thread operation {other}"
                )));
            }
        }
    }
    if !has_gemv {
        return Err(validation_error("i2s_applied_thread_matrix missing gemv run"));
    }
    if !has_gemm {
        return Err(validation_error("i2s_applied_thread_matrix missing gemm run"));
    }

    let boundary = require_array(matrix, "claim_boundary")?;
    if boundary.is_empty() {
        return Err(validation_error("claim_boundary must not be empty"));
    }

    Ok(())
}

fn validate_embedding_quantization_evidence(
    evidence: &serde_json::Value,
) -> Result<(), ReceiptError> {
    require_string_eq(evidence, "work_item", "CPU-BITNET-EMBD-001")?;
    require_string_eq(evidence, "artifact_kind", "cpu_bitnet_embedding_quantization_evidence")?;
    require_string_eq(evidence, "claim", "bitnet_embedding_quantization_evidence_receipt")?;
    require_non_empty_string(evidence, "source_tensor_boundary_audit")?;
    require_string_eq(evidence, "target_quantization", "Q6_K")?;
    require_bool_eq(evidence, "fallback_used", false)?;
    require_null(evidence, "fallback_reason")?;
    require_bool_eq(evidence, "speedup_claim", false)?;
    require_bool_eq(evidence, "answer_quality_claim", false)?;
    require_bool_eq(evidence, "acceleration_claim", false)?;
    require_bool_eq(evidence, "qk256_semantic_change_claim", false)?;
    require_non_empty_string(evidence, "current_embedding_quantization")?;
    require_bool(evidence, "current_artifact_contains_q6_k_embedding")?;
    require_bool(evidence, "q6_k_embedding_proven")?;
    require_non_empty_string(evidence, "evidence_status")?;
    require_non_empty_string(evidence, "recommended_next_step")?;

    let current_embedding = require_object(evidence, "current_embedding")?;
    require_non_empty_string(current_embedding, "name")?;
    require_non_empty_string(current_embedding, "tensor_type")?;
    require_u64_at_least(current_embedding, "size_bytes", 1)?;
    let shape = require_array(current_embedding, "shape")?;
    if shape.is_empty() {
        return Err(validation_error("current_embedding.shape must not be empty"));
    }

    let loader_scope = require_object(evidence, "loader_scope")?;
    require_bool_eq(loader_scope, "q6_k_tensor_type_known", true)?;
    require_bool_eq(loader_scope, "q6_k_dense_standard_dequantizer_present", true)?;
    require_non_empty_string(loader_scope, "q6_k_embedding_operating_path")?;
    require_non_empty_string(loader_scope, "note")?;

    let boundary = require_array(evidence, "claim_boundary")?;
    if boundary.is_empty() {
        return Err(validation_error("claim_boundary must not be empty"));
    }

    Ok(())
}

fn validate_i2s_applied_thread_matrix_run(run: &serde_json::Value) -> Result<(), ReceiptError> {
    require_non_empty_string(run, "profile")?;
    let operation = require_string(run, "operation")?;
    require_non_empty_string(run, "execution_phase")?;
    require_string_eq(run, "status", "measured")?;
    require_non_empty_string(run, "requested_kernel")?;
    require_non_empty_string(run, "selected_kernel")?;
    require_bool_eq(run, "fallback_used", false)?;
    require_null(run, "fallback_reason")?;
    require_bool_eq(run, "speedup_claim", false)?;

    let candidate = require_object(run, "candidate")?;
    require_u64_at_least(candidate, "parallelism_degree", 1)?;
    require_u64_at_least(candidate, "row_block", 1)?;
    require_u64_at_least(candidate, "col_block", 1)?;
    require_u64_at_least(candidate, "thread_count", 1)?;
    require_bool_eq(candidate, "thread_count_applied", true)?;
    require_string_eq(candidate, "thread_count_policy", "applied_scoped_threads")?;
    require_u64_at_least(candidate, "applied_thread_count", 1)?;
    let partition = require_string(candidate, "thread_partition")?;
    match (operation, partition) {
        ("gemv", "rows") | ("gemm", "tokens") => {}
        _ => {
            return Err(validation_error(format!(
                "thread_partition {partition} is not valid for operation {operation}"
            )));
        }
    }
    require_non_empty_string(candidate, "thread_count_note")?;

    let shape = require_object(run, "shape")?;
    require_u64_at_least(shape, "rows", 1)?;
    require_u64_at_least(shape, "cols", 1)?;
    require_u64_at_least(shape, "tokens", 1)?;
    require_u64_at_least(shape, "iterations", 1)?;
    require_bool(shape, "cols_rounded_to_qk256_block")?;

    require_non_negative_number(run, "wall_time_ms")?;
    require_non_negative_number(run, "median_ms")?;
    require_non_negative_number(run, "p95_ms")?;
    require_non_negative_number(run, "bandwidth_gbps")?;
    require_non_negative_number(run, "tokens_per_second")?;

    Ok(())
}

fn validate_i2s_tiling_matrix_run(run: &serde_json::Value) -> Result<(), ReceiptError> {
    require_non_empty_string(run, "profile")?;
    require_non_empty_string(run, "operation")?;
    require_non_empty_string(run, "execution_phase")?;
    require_string_eq(run, "status", "measured")?;
    require_non_empty_string(run, "requested_kernel")?;
    require_non_empty_string(run, "selected_kernel")?;
    require_bool_eq(run, "fallback_used", false)?;
    require_null(run, "fallback_reason")?;
    require_bool_eq(run, "speedup_claim", false)?;

    let candidate = require_object(run, "candidate")?;
    require_u64_at_least(candidate, "parallelism_degree", 1)?;
    require_u64_at_least(candidate, "row_block", 1)?;
    require_u64_at_least(candidate, "col_block", 1)?;
    require_u64_at_least(candidate, "thread_count", 1)?;
    require_bool_eq(candidate, "thread_count_applied", false)?;
    require_string_eq(candidate, "thread_count_policy", "recorded_not_applied")?;
    require_non_empty_string(candidate, "thread_count_note")?;

    let shape = require_object(run, "shape")?;
    require_u64_at_least(shape, "rows", 1)?;
    require_u64_at_least(shape, "cols", 1)?;
    require_u64_at_least(shape, "tokens", 1)?;
    require_u64_at_least(shape, "iterations", 1)?;
    require_bool(shape, "cols_rounded_to_qk256_block")?;

    require_non_negative_number(run, "wall_time_ms")?;
    require_non_negative_number(run, "median_ms")?;
    require_non_negative_number(run, "p95_ms")?;
    require_non_negative_number(run, "bandwidth_gbps")?;
    require_non_negative_number(run, "tokens_per_second")?;

    Ok(())
}

fn require_backend_profile<'a>(
    profiles: &'a [serde_json::Value],
    backend: &str,
) -> Result<&'a serde_json::Value, ReceiptError> {
    profiles
        .iter()
        .find(|entry| entry.get("backend").and_then(serde_json::Value::as_str) == Some(backend))
        .ok_or_else(|| validation_error(format!("profiles missing backend {backend}")))
}

fn require_profile<'a>(
    profiles: &'a [serde_json::Value],
    profile: &str,
    backend: &str,
) -> Result<&'a serde_json::Value, ReceiptError> {
    profiles
        .iter()
        .find(|entry| {
            entry.get("profile").and_then(serde_json::Value::as_str) == Some(profile)
                && entry.get("backend").and_then(serde_json::Value::as_str) == Some(backend)
        })
        .ok_or_else(|| validation_error(format!("profiles missing {profile} for {backend}")))
}

fn validate_answer_path_timing(timing: &serde_json::Value, cuda: bool) -> Result<(), ReceiptError> {
    require_non_negative_number(timing, "model_load_ms")?;
    require_non_negative_number(timing, "tokenizer_load_ms")?;
    require_non_negative_number(timing, "prompt_render_tokenize_ms")?;
    require_non_negative_number(timing, "prefill_ms")?;
    require_non_negative_number(timing, "first_token_ms")?;
    require_non_negative_number(timing, "decode_total_ms")?;
    require_non_negative_number(timing, "steady_decode_tokens_per_second")?;
    if cuda {
        require_nullable_number_with_source(timing, "cuda_context_init_ms")?;
        require_nullable_number_with_source(timing, "weight_upload_ms")?;
        require_nullable_number_with_source(timing, "kernel_time_ms")?;
        require_nullable_u64_with_source(timing, "host_to_device_bytes")?;
        require_nullable_u64_with_source(timing, "device_to_host_bytes")?;
    }
    Ok(())
}

fn validate_answer_path_profile(
    profile: &serde_json::Value,
    must_be_measured: bool,
) -> Result<(), ReceiptError> {
    require_non_empty_string(profile, "profile")?;
    require_non_empty_string(profile, "backend")?;
    let status = require_string(profile, "status")?;
    match status {
        "measured" => {
            require_non_negative_number(profile, "total_ms")?;
            require_non_negative_number(profile, "first_token_ms")?;
            require_non_negative_number(profile, "tokens_per_second")?;
            require_u64_at_least(profile, "prompt_tokens", 1)?;
            require_u64_at_least(profile, "generated_tokens", 1)?;
            require_bool_eq(profile, "quality_passed", true)?;
            require_bool_eq(profile, "fallback_used", false)?;
        }
        "measured_existing_receipt" => {
            require_non_empty_string(profile, "receipt_path")?;
            require_bool_eq(profile, "quality_passed", true)?;
            require_bool_eq(profile, "fallback_used", false)?;
        }
        "blocked_timeout" => {
            if must_be_measured {
                let backend = require_string(profile, "backend")?;
                return Err(validation_error(format!("profile {backend} must be measured")));
            }
            require_u64_at_least(profile, "timeout_seconds", 1)?;
            require_non_empty_string(profile, "reason")?;
        }
        "not_run" => {
            if must_be_measured {
                let backend = require_string(profile, "backend")?;
                return Err(validation_error(format!("profile {backend} must be measured")));
            }
            require_non_empty_string(profile, "reason")?;
        }
        other => {
            return Err(validation_error(format!(
                "answer-path profile status must be measured, measured_existing_receipt, blocked_timeout, or not_run, got {other}"
            )));
        }
    }
    Ok(())
}

fn validate_repeated_backend_summary(
    summary: &serde_json::Value,
    expected_backend: &str,
    expected_runtime_api: &str,
    expected_runs: u64,
    cuda: bool,
) -> Result<(), ReceiptError> {
    require_string_eq(summary, "backend", expected_backend)?;
    require_string_eq(summary, "runtime_api", expected_runtime_api)?;
    require_u64_eq(summary, "runs", expected_runs)?;
    require_bool_eq(summary, "quality_passed", true)?;
    require_bool_eq(summary, "fallback_used", false)?;
    validate_metric_summary(require_object(summary, "total_ms")?, expected_runs)?;
    validate_metric_summary(require_object(summary, "first_token_ms")?, expected_runs)?;
    validate_metric_summary(require_object(summary, "decode_total_ms")?, expected_runs)?;
    validate_metric_summary(require_object(summary, "tokens_per_second")?, expected_runs)?;
    if cuda {
        validate_metric_summary(require_object(summary, "kernel_time_ms")?, expected_runs)?;
        validate_u64_summary(require_object(summary, "host_to_device_bytes")?, expected_runs)?;
        validate_u64_summary(require_object(summary, "device_to_host_bytes")?, expected_runs)?;
    }
    Ok(())
}

fn validate_metric_summary(
    summary: &serde_json::Value,
    expected_samples: u64,
) -> Result<(), ReceiptError> {
    require_u64_eq(summary, "samples", expected_samples)?;
    require_non_negative_number(summary, "min")?;
    require_non_negative_number(summary, "max")?;
    require_non_negative_number(summary, "mean")?;
    require_non_negative_number(summary, "median")?;
    let min = summary["min"].as_f64().unwrap_or(0.0);
    let max = summary["max"].as_f64().unwrap_or(0.0);
    if max < min {
        return Err(validation_error(format!("metric summary max {max} is less than min {min}")));
    }
    Ok(())
}

fn validate_u64_summary(
    summary: &serde_json::Value,
    expected_samples: u64,
) -> Result<(), ReceiptError> {
    require_u64_eq(summary, "samples", expected_samples)?;
    let min = require_u64(summary, "min")?;
    let max = require_u64(summary, "max")?;
    require_non_negative_number(summary, "mean")?;
    require_non_negative_number(summary, "median")?;
    if max < min {
        return Err(validation_error(format!("u64 summary max {max} is less than min {min}")));
    }
    Ok(())
}

fn validate_bitnet_qk256_execution_plan(plan: &serde_json::Value) -> Result<(), ReceiptError> {
    require_string_eq(plan, "planner_version", "cuda-planner-004")?;
    require_string_eq(plan, "model_family", "bitnet_b1_58")?;
    require_string_eq(plan, "quantization", "i2_s_qk256")?;
    require_string_eq(plan, "selected_route", "bitnet_qk256_cuda")?;
    require_string_eq(plan, "runtime_api", "cuda")?;
    require_string_eq(plan, "strict_fallback_policy", "reject")?;
    require_bool_eq(plan, "bitnet_packed_qk256_cuda", true)?;
    require_bool_eq(plan, "dense_regular_llm_cuda", false)?;
    require_bool_eq(plan, "fallback_used", false)?;
    require_bool_eq(plan, "strict_cuda_ready", true)?;
    require_bool_eq(plan, "speedup_claim", false)?;
    require_bool_eq(plan, "full_cuda_residency_claimed", false)?;
    require_u64_at_least(plan, "cuda_bitnet_qk256_ops", 1)?;
    require_u64_eq(plan, "cuda_dense_regular_llm_ops", 0)?;
    require_u64_eq(plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(plan, "unsupported_ops", 0)?;
    require_u64_at_least(plan, "total_ops", 1)?;
    require_u64_at_least(plan, "cuda_ops", 1)?;
    Ok(())
}

fn validate_repeated_ask_run(run: &serde_json::Value) -> Result<(), ReceiptError> {
    require_string_eq(run, "profile", "strict_ask_math_8")?;
    let backend = require_string(run, "backend")?;
    let runtime_api = require_string(run, "runtime_api")?;
    match backend {
        "amd-9950x3d-cpu-avx512" => {
            if runtime_api != "cpu" {
                return Err(validation_error("CPU repeated run runtime_api must be cpu"));
            }
        }
        "nvidia-rtx-5070-ti-cuda" => {
            if runtime_api != "cuda" {
                return Err(validation_error("CUDA repeated run runtime_api must be cuda"));
            }
            require_string_eq(run, "kernel_id", "qk256_gemv_cuda")?;
            require_u64_at_least(run, "kernel_invocations", 1)?;
            require_non_negative_number(run, "kernel_time_ms")?;
            require_u64_at_least(run, "host_to_device_bytes", 1)?;
            require_u64_at_least(run, "device_to_host_bytes", 1)?;
        }
        other => return Err(validation_error(format!("unexpected repeated run backend {other}"))),
    }
    require_string_eq(run, "status", "measured")?;
    require_u64_at_least(run, "repeat_index", 1)?;
    require_non_empty_string(run, "source_receipt_path")?;
    require_non_empty_string(run, "selected_backend")?;
    require_non_empty_string(run, "kernel_id")?;
    require_non_negative_number(run, "total_ms")?;
    require_non_negative_number(run, "first_token_ms")?;
    require_non_negative_number(run, "decode_total_ms")?;
    require_non_negative_number(run, "tokens_per_second")?;
    require_u64_at_least(run, "prompt_tokens", 1)?;
    require_u64_at_least(run, "generated_tokens", 1)?;
    let answer = require_string(run, "answer_trimmed")?;
    if answer != "4" {
        return Err(validation_error(format!("answer_trimmed must be 4, got {answer:?}")));
    }
    let generated_ids = require_array(run, "generated_token_ids")?;
    if generated_ids.is_empty() {
        return Err(validation_error("generated_token_ids must not be empty"));
    }
    require_bool_eq(run, "quality_passed", true)?;
    require_bool_eq(run, "fallback_used", false)?;
    Ok(())
}

fn validate_bitnet_benchmark_profile(
    profile: &serde_json::Value,
    must_be_measured: bool,
) -> Result<(), ReceiptError> {
    require_non_empty_string(profile, "backend")?;
    require_non_empty_string(profile, "runtime_api")?;
    let status = require_string(profile, "status")?;
    match status {
        "measured" => {
            require_non_negative_number(profile, "total_ms")?;
            require_non_negative_number(profile, "first_token_ms")?;
            require_non_negative_number(profile, "tokens_per_second")?;
            require_u64_at_least(profile, "prompt_tokens", 1)?;
            require_u64_at_least(profile, "generated_tokens", 1)?;
            require_bool_eq(profile, "fallback_used", false)?;
        }
        "not_run" => {
            if must_be_measured {
                let backend = require_string(profile, "backend")?;
                return Err(validation_error(format!("profile {backend} must be measured")));
            }
            require_non_empty_string(profile, "reason")?;
        }
        other => {
            return Err(validation_error(format!(
                "profile status must be measured or not_run, got {other}"
            )));
        }
    }
    Ok(())
}

fn require_object<'a>(
    value: &'a serde_json::Value,
    field: &str,
) -> Result<&'a serde_json::Value, ReceiptError> {
    let child =
        value.get(field).ok_or_else(|| validation_error(format!("{field} must be an object")))?;
    if !child.is_object() {
        return Err(validation_error(format!("{field} must be an object")));
    }
    Ok(child)
}

fn require_array<'a>(
    value: &'a serde_json::Value,
    field: &str,
) -> Result<&'a Vec<serde_json::Value>, ReceiptError> {
    value
        .get(field)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| validation_error(format!("{field} must be an array")))
}

fn require_string_array_contains(
    values: &[serde_json::Value],
    expected: &str,
    field: &str,
) -> Result<(), ReceiptError> {
    if values.iter().any(|value| value.as_str() == Some(expected)) {
        return Ok(());
    }
    Err(validation_error(format!("{field} must include {expected}")))
}

fn find_object_by_string_field<'a>(
    values: &'a [serde_json::Value],
    field: &str,
    expected: &str,
) -> Option<&'a serde_json::Value> {
    values
        .iter()
        .find(|value| value.get(field).and_then(serde_json::Value::as_str) == Some(expected))
}

fn require_string<'a>(value: &'a serde_json::Value, field: &str) -> Result<&'a str, ReceiptError> {
    value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| validation_error(format!("{field} must be a string")))
}

fn require_non_empty_string(value: &serde_json::Value, field: &str) -> Result<(), ReceiptError> {
    let actual = require_string(value, field)?;
    if actual.trim().is_empty() {
        return Err(validation_error(format!("{field} must not be empty")));
    }
    Ok(())
}

fn require_string_eq(
    value: &serde_json::Value,
    field: &str,
    expected: &str,
) -> Result<(), ReceiptError> {
    let actual = require_string(value, field)?;
    if actual != expected {
        return Err(validation_error(format!("{field} must be {expected}, got {actual}")));
    }
    Ok(())
}

fn require_bool_eq(
    value: &serde_json::Value,
    field: &str,
    expected: bool,
) -> Result<(), ReceiptError> {
    let actual = require_bool(value, field)?;
    if actual != expected {
        return Err(validation_error(format!("{field} must be {expected}, got {actual}")));
    }
    Ok(())
}

fn require_bool(value: &serde_json::Value, field: &str) -> Result<bool, ReceiptError> {
    value
        .get(field)
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| validation_error(format!("{field} must be a boolean")))
}

fn require_null(value: &serde_json::Value, field: &str) -> Result<(), ReceiptError> {
    if !value.get(field).is_some_and(serde_json::Value::is_null) {
        return Err(validation_error(format!("{field} must be null")));
    }
    Ok(())
}

fn require_u64(value: &serde_json::Value, field: &str) -> Result<u64, ReceiptError> {
    value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| validation_error(format!("{field} must be an unsigned integer")))
}

fn require_u64_eq(
    value: &serde_json::Value,
    field: &str,
    expected: u64,
) -> Result<(), ReceiptError> {
    let actual = require_u64(value, field)?;
    if actual != expected {
        return Err(validation_error(format!("{field} must be {expected}, got {actual}")));
    }
    Ok(())
}

fn require_u64_at_least(
    value: &serde_json::Value,
    field: &str,
    minimum: u64,
) -> Result<(), ReceiptError> {
    let actual = require_u64(value, field)?;
    if actual < minimum {
        return Err(validation_error(format!("{field} must be >= {minimum}, got {actual}")));
    }
    Ok(())
}

fn require_non_negative_number(value: &serde_json::Value, field: &str) -> Result<(), ReceiptError> {
    let actual = value
        .get(field)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| validation_error(format!("{field} must be a number")))?;
    if actual < 0.0 {
        return Err(validation_error(format!("{field} must be non-negative, got {actual}")));
    }
    Ok(())
}

fn require_nullable_number_with_source(
    value: &serde_json::Value,
    field: &str,
) -> Result<(), ReceiptError> {
    if let Some(actual) = value.get(field).and_then(serde_json::Value::as_f64) {
        if actual < 0.0 {
            return Err(validation_error(format!("{field} must be non-negative, got {actual}")));
        }
    } else if !value.get(field).is_some_and(serde_json::Value::is_null) {
        return Err(validation_error(format!("{field} must be a number or null")));
    }
    require_non_empty_string(value, &format!("{field}_source"))?;
    Ok(())
}

fn require_nullable_u64_with_source(
    value: &serde_json::Value,
    field: &str,
) -> Result<(), ReceiptError> {
    if value.get(field).and_then(serde_json::Value::as_u64).is_none()
        && !value.get(field).is_some_and(serde_json::Value::is_null)
    {
        return Err(validation_error(format!("{field} must be an unsigned integer or null")));
    }
    require_non_empty_string(value, &format!("{field}_source"))?;
    Ok(())
}

fn validation_error(message: impl Into<String>) -> ReceiptError {
    ReceiptError::Validation(message.into())
}

fn is_rtx5070ti_device_name(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase();
    normalized.contains("nvidia")
        && normalized.contains("geforce")
        && normalized.contains("rtx")
        && normalized.contains("5070")
        && normalized.contains("ti")
}
