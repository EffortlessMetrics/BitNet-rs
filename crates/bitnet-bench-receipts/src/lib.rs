//! Benchmark receipts for tracking kernel performance over time.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::path::Path;

/// Errors from receipt I/O and serialization.
#[derive(Debug, thiserror::Error)]
pub enum ReceiptError {
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("validation error: {0}")]
    Validation(String),
}

/// A single benchmark measurement for a compute-kernel dispatch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchReceipt {
    pub kernel_name: String,
    pub workgroup_size: [u32; 3],
    pub dispatch_size: [u32; 3],
    pub elapsed_us: u64,
    pub throughput_gflops: f64,
    pub timestamp: u64,
    pub device_name: String,
    pub backend: String,
}

impl BenchReceipt {
    /// Create a new benchmark receipt.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel_name: impl Into<String>,
        workgroup_size: [u32; 3],
        dispatch_size: [u32; 3],
        elapsed_us: u64,
        throughput_gflops: f64,
        timestamp: u64,
        device_name: impl Into<String>,
        backend: impl Into<String>,
    ) -> Self {
        Self {
            kernel_name: kernel_name.into(),
            workgroup_size,
            dispatch_size,
            elapsed_us,
            throughput_gflops,
            timestamp,
            device_name: device_name.into(),
            backend: backend.into(),
        }
    }

    /// Serialize to a JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("BenchReceipt is always serializable")
    }

    /// Deserialize from a JSON string.
    pub fn from_json(s: &str) -> Result<Self, ReceiptError> {
        Ok(serde_json::from_str(s)?)
    }
}

/// Append-only JSON-lines store for benchmark receipts.
pub struct ReceiptStore;

impl ReceiptStore {
    /// Load all receipts from a JSON-lines file.
    pub fn load(path: &Path) -> Result<Vec<BenchReceipt>, ReceiptError> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut receipts = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            receipts.push(BenchReceipt::from_json(trimmed)?);
        }
        Ok(receipts)
    }

    /// Append a single receipt to a JSON-lines file, creating it if absent.
    pub fn append(path: &Path, receipt: &BenchReceipt) -> Result<(), ReceiptError> {
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(file, "{}", receipt.to_json())?;
        Ok(())
    }
}

/// Validate an RTX 5070 Ti CUDA benchmark receipt.
///
/// This validator is deliberately strict about backend identity and fallback
/// counters so a dense, WGPU, CPU-fallback, or generic CUDA run cannot satisfy
/// the NVIDIA CUDA proof lane.
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

fn expected_cpu_profile_phase(profile: &str) -> &'static str {
    match profile {
        "micro" => "micro_kernel",
        "layer" => "layer_forward",
        "prefill" => "prefill",
        "first_token" => "first_token",
        "decode" => "decode_steady_state",
        _ => "unknown",
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;

    fn sample_receipt(name: &str, elapsed_us: u64) -> BenchReceipt {
        BenchReceipt::new(
            name,
            [256, 1, 1],
            [1024, 1, 1],
            elapsed_us,
            42.0,
            1_700_000_000,
            "Test GPU",
            "vulkan",
        )
    }

    #[test]
    fn test_new_sets_all_fields() {
        let r = sample_receipt("matmul", 500);
        assert_eq!(r.kernel_name, "matmul");
        assert_eq!(r.workgroup_size, [256, 1, 1]);
        assert_eq!(r.dispatch_size, [1024, 1, 1]);
        assert_eq!(r.elapsed_us, 500);
        assert_eq!(r.device_name, "Test GPU");
        assert_eq!(r.backend, "vulkan");
    }

    #[test]
    fn test_to_json_produces_valid_json() {
        let r = sample_receipt("softmax", 100);
        let json = r.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["kernel_name"], "softmax");
    }

    #[test]
    fn test_from_json_roundtrip() {
        let r = sample_receipt("rms_norm", 250);
        let json = r.to_json();
        let r2 = BenchReceipt::from_json(&json).unwrap();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_from_json_invalid_returns_error() {
        let result = BenchReceipt::from_json("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_json_missing_field() {
        let result = BenchReceipt::from_json(r#"{"kernel_name":"x"}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialization_preserves_workgroup_array() {
        let r = sample_receipt("conv", 300);
        let json = r.to_json();
        assert!(json.contains("[256,1,1]"));
    }

    #[test]
    fn test_store_append_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("receipts.jsonl");

        let r1 = sample_receipt("k1", 100);
        let r2 = sample_receipt("k2", 200);
        ReceiptStore::append(&path, &r1).unwrap();
        ReceiptStore::append(&path, &r2).unwrap();

        let loaded = ReceiptStore::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0], r1);
        assert_eq!(loaded[1], r2);
    }

    #[test]
    fn test_store_load_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.jsonl");
        std::fs::File::create(&path).unwrap();

        let loaded = ReceiptStore::load(&path).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_store_load_nonexistent_file() {
        let result = ReceiptStore::load(Path::new("/nonexistent/path.jsonl"));
        assert!(result.is_err());
    }

    #[test]
    fn test_store_skips_blank_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blanks.jsonl");
        let r = sample_receipt("k1", 100);
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{}", r.to_json()).unwrap();
        writeln!(f).unwrap();
        writeln!(f, "{}", r.to_json()).unwrap();
        drop(f);

        let loaded = ReceiptStore::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_throughput_precision() {
        let r = BenchReceipt::new("k", [1, 1, 1], [1, 1, 1], 1, std::f64::consts::PI, 0, "", "");
        let r2 = BenchReceipt::from_json(&r.to_json()).unwrap();
        assert!((r2.throughput_gflops - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_store_append_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.jsonl");
        assert!(!path.exists());

        ReceiptStore::append(&path, &sample_receipt("k", 1)).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn rtx5070ti_cuda_benchmark_receipt_validates() {
        let receipt = sample_cuda_benchmark_receipt();
        validate_rtx5070ti_cuda_benchmark_receipt_json(&receipt).unwrap();
    }

    #[test]
    fn rtx5070ti_cuda_benchmark_rejects_generic_cuda_backend() {
        let mut receipt = sample_cuda_benchmark_receipt();
        receipt["selected_backend"] = json!("cuda");
        assert!(validate_rtx5070ti_cuda_benchmark_receipt_json(&receipt).is_err());
    }

    #[test]
    fn rtx5070ti_cuda_benchmark_rejects_fallback() {
        let mut receipt = sample_cuda_benchmark_receipt();
        receipt["fallback_used"] = json!(true);
        assert!(validate_rtx5070ti_cuda_benchmark_receipt_json(&receipt).is_err());
    }

    #[test]
    fn rtx5070ti_cuda_benchmark_rejects_speedup_claim() {
        let mut receipt = sample_cuda_benchmark_receipt();
        receipt["speedup_claim"] = json!(true);
        assert!(validate_rtx5070ti_cuda_benchmark_receipt_json(&receipt).is_err());
    }

    #[test]
    fn rtx5070ti_cuda_benchmark_rejects_missing_required_profile() {
        let mut receipt = sample_cuda_benchmark_receipt();
        receipt["profiles"] = json!([
            { "profile": "cuda_tiny_smoke", "status": "measured" }
        ]);
        assert!(validate_rtx5070ti_cuda_benchmark_receipt_json(&receipt).is_err());
    }

    #[test]
    fn strict_cpu_benchmark_receipt_validates() {
        let receipt = sample_cpu_benchmark_receipt();
        validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap();
    }

    #[test]
    fn strict_cpu_benchmark_rejects_missing_decode_profile() {
        let mut receipt = sample_cpu_benchmark_receipt();
        receipt["profiles"] = json!([
            measured_cpu_profile("micro"),
            measured_cpu_profile("layer"),
            measured_cpu_profile("prefill"),
            measured_cpu_profile("first_token")
        ]);

        let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("decode"), "unexpected error: {err}");
    }

    #[test]
    fn strict_cpu_benchmark_rejects_hidden_fallback() {
        let mut receipt = sample_cpu_benchmark_receipt();
        receipt["fallback_used"] = json!(true);

        let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("fallback_used"), "unexpected error: {err}");
    }

    #[test]
    fn strict_cpu_benchmark_rejects_avx2_without_fma() {
        let mut receipt = sample_cpu_benchmark_receipt();
        receipt["cpu"]["features"] = json!(["avx2"]);

        let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("avx2 and fma"), "unexpected error: {err}");
    }

    #[test]
    fn strict_cpu_benchmark_rejects_speedup_claim() {
        let mut receipt = sample_cpu_benchmark_receipt();
        receipt["speedup_claim"] = json!(true);

        let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("speedup_claim"), "unexpected error: {err}");
    }

    #[test]
    fn strict_cpu_benchmark_rejects_profile_phase_mismatch() {
        let mut receipt = sample_cpu_benchmark_receipt();
        receipt["profiles"][4]["execution_phase"] = json!("prefill");

        let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("execution_phase"), "unexpected error: {err}");
    }

    #[test]
    fn strict_cpu_benchmark_rejects_profile_without_shape() {
        let mut receipt = sample_cpu_benchmark_receipt();
        receipt["profiles"][0]["shape"] = serde_json::Value::Null;

        let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("shape"), "unexpected error: {err}");
    }

    #[test]
    fn committed_rtx5070ti_cuda_benchmark_receipt_validates() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-benchmark.json");
        validate_rtx5070ti_cuda_benchmark_receipt_file(&path).unwrap();
    }

    #[test]
    fn strict_bitnet_cuda_benchmark_receipt_validates() {
        let receipt = sample_strict_bitnet_cuda_benchmark_receipt();
        validate_strict_bitnet_cuda_benchmark_receipt_json(&receipt).unwrap();
    }

    #[test]
    fn strict_bitnet_cuda_benchmark_rejects_generic_cuda_backend() {
        let mut receipt = sample_strict_bitnet_cuda_benchmark_receipt();
        receipt["selected_backend"] = json!("cuda");

        let err =
            validate_strict_bitnet_cuda_benchmark_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("selected_backend"), "unexpected error: {err}");
    }

    #[test]
    fn strict_bitnet_cuda_benchmark_rejects_fallback() {
        let mut receipt = sample_strict_bitnet_cuda_benchmark_receipt();
        receipt["fallback_used"] = json!(true);

        let err =
            validate_strict_bitnet_cuda_benchmark_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("fallback_used"), "unexpected error: {err}");
    }

    #[test]
    fn strict_bitnet_cuda_benchmark_rejects_speedup_claim() {
        let mut receipt = sample_strict_bitnet_cuda_benchmark_receipt();
        receipt["speedup_claim"] = json!(true);

        let err =
            validate_strict_bitnet_cuda_benchmark_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("speedup_claim"), "unexpected error: {err}");
    }

    #[test]
    fn strict_bitnet_cuda_benchmark_rejects_missing_cpu_profile() {
        let mut receipt = sample_strict_bitnet_cuda_benchmark_receipt();
        receipt["profiles"] = json!([
            not_run_bitnet_profile("amd-9950x3d-cpu-scalar"),
            measured_bitnet_profile("amd-9950x3d-cpu-avx512", "cpu"),
            measured_bitnet_profile("nvidia-rtx-5070-ti-cuda", "cuda")
        ]);

        let err =
            validate_strict_bitnet_cuda_benchmark_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("amd-9950x3d-cpu-avx2"), "unexpected error: {err}");
    }

    #[test]
    fn strict_bitnet_cuda_benchmark_requires_measured_cuda_profile() {
        let mut receipt = sample_strict_bitnet_cuda_benchmark_receipt();
        receipt["profiles"][3] = not_run_bitnet_profile("nvidia-rtx-5070-ti-cuda");

        let err =
            validate_strict_bitnet_cuda_benchmark_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("must be measured"), "unexpected error: {err}");
    }

    #[test]
    fn committed_strict_bitnet_cuda_benchmark_receipt_validates() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-benchmark.json",
        );
        validate_strict_bitnet_cuda_benchmark_receipt_file(&path).unwrap();
    }

    fn sample_cpu_benchmark_receipt() -> serde_json::Value {
        json!({
            "schema": 1,
            "artifact_kind": "cpu_benchmark",
            "machine_id": "intel-i5-8250u-cpu-avx2",
            "hardware_lane": "intel-i5-8250u-cpu-avx2",
            "timestamp_utc": "2026-05-06T00:00:00Z",
            "requested_backend": "cpu",
            "selected_backend": "intel-i5-8250u-cpu-avx2",
            "runtime_api": "cpu",
            "claim": "cpu_benchmark_receipt",
            "speedup_claim": false,
            "fallback_used": false,
            "fallback_reason": null,
            "model": {
                "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
                "file": "ggml-model-i2_s.gguf",
                "sha256": "abc123def456",
                "family": "bitnet",
                "quant_format": "QK256/I2_S"
            },
            "tokenizer": {
                "source": "gguf_metadata",
                "strict": true
            },
            "kernel": {
                "requested_kernel": "qk256-avx2-gemv",
                "selected_kernel": "qk256-avx2-gemv",
                "oracle_kernel": "qk256-scalar-gemv",
                "fallback_used": false,
                "fallback_reason": null,
                "dequantizes_before_compute": false
            },
            "cpu": {
                "model": "Intel Core i5-8250U",
                "arch": "x86_64",
                "features": ["avx2", "fma"],
                "threads": 8,
                "avx512": false,
                "power_mode": "unknown",
                "temperature_c": null,
                "frequency_mhz": null
            },
            "workload": {
                "prompt_tokens": 512,
                "generated_tokens": 128,
                "batch_size": 1
            },
            "profiles": [
                measured_cpu_profile("micro"),
                measured_cpu_profile("layer"),
                measured_cpu_profile("prefill"),
                measured_cpu_profile("first_token"),
                measured_cpu_profile("decode")
            ],
            "artifact_path": "ci/hardware/intel-i5-8250u-cpu-avx2/benchmark-receipt.json"
        })
    }

    fn measured_cpu_profile(profile: &str) -> serde_json::Value {
        json!({
            "profile": profile,
            "execution_phase": expected_cpu_profile_phase(profile),
            "status": "measured",
            "requested_kernel": "qk256-avx2-gemv",
            "selected_kernel": "qk256-avx2-gemv",
            "fallback_used": false,
            "fallback_reason": null,
            "shape": {
                "rows": 512,
                "cols": 1024,
                "iterations": 8
            },
            "wall_time_ms": 1.0,
            "median_ms": 1.0,
            "p95_ms": 1.0,
            "bandwidth_gbps": 0.0,
            "tokens_per_second": 0.0
        })
    }

    fn sample_cuda_benchmark_receipt() -> serde_json::Value {
        json!({
            "schema": 1,
            "artifact_kind": "cuda_benchmark",
            "machine_id": "windows-9950x3d-rtx5070ti",
            "hardware_lane": "nvidia_rtx_5070_ti_cuda",
            "timestamp_utc": "2026-05-06T00:00:00Z",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "reference_backend": "amd-9950x3d-cpu-avx512",
            "runtime_api": "cuda",
            "claim": "cuda_benchmark_baseline",
            "speedup_claim": false,
            "fallback_used": false,
            "fallback_backend": null,
            "fallback_reason": null,
            "cuda": {
                "available": true,
                "device_count": 1,
                "selected_device_index": 0,
                "selected_device_name": "NVIDIA GeForce RTX 5070 Ti",
                "compute_capability": "12.0",
                "driver_version": "570.00",
                "cuda_runtime_version": "12.9",
                "cuda_toolkit_version": "12.9",
                "nvrtc_version": "12.9",
                "nvml_available": true,
                "vram_bytes": 17179869184u64,
                "power_limit_watts": 300.0,
                "power_draw_watts": 50.0,
                "temperature_c": 45.0
            },
            "machine": {
                "cpu": "AMD Ryzen 9 9950X3D",
                "gpu": "NVIDIA GeForce RTX 5070 Ti"
            },
            "benchmark": {
                "profile": "cuda_tiny_smoke",
                "kernel_id": "cuda_tiny_vector_add",
                "fixture_id": "cuda_tiny_vector_add_1024",
                "input_len": 1024,
                "iterations": 10,
                "cold_warm": {
                    "compile_ms": 1.0,
                    "first_iteration_total_ms": 1.0,
                    "warm_iterations": 10
                },
                "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
                "cuda_backend": "nvidia-rtx-5070-ti-cuda",
                "cpu_reference_ms": 0.1,
                "cuda_total_ms": 0.2,
                "cuda_kernel_ms": 0.1,
                "host_to_device_ms": 0.01,
                "device_to_host_ms": 0.01,
                "allocation_ms": 0.01,
                "speedup_vs_cpu": 0.5,
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
                "passed": true
            },
            "profiles": [
                { "profile": "cuda_tiny_smoke", "status": "measured" },
                { "profile": "cuda_transfer_h2d_d2h", "status": "measured" },
                { "profile": "cuda_fp32_matmul_small", "status": "not_run" },
                { "profile": "cuda_i2s_matmul_small", "status": "not_run" },
                { "profile": "cuda_i2s_matmul_medium", "status": "not_run" }
            ],
            "kernel_stats": [
                {
                    "kernel_id": "cuda_tiny_vector_add",
                    "invocations": 10,
                    "fallback_invocations": 0,
                    "host_to_device_bytes": 81920,
                    "device_to_host_bytes": 40960,
                    "kernel_launches": 10,
                    "kernel_time_ms": 0.1,
                    "selected_device_index": 0,
                    "selected_device_name": "NVIDIA GeForce RTX 5070 Ti",
                    "compute_capability": "12.0"
                }
            ],
            "artifact_path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-benchmark.json"
        })
    }

    fn sample_strict_bitnet_cuda_benchmark_receipt() -> serde_json::Value {
        json!({
            "schema": 1,
            "artifact_kind": "strict_bitnet_cuda_benchmark",
            "machine_id": "windows-9950x3d-rtx5070ti",
            "hardware_lane": "nvidia_rtx_5070_ti_cuda",
            "timestamp_utc": "2026-05-07T00:00:00Z",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "reference_backend": "amd-9950x3d-cpu-avx512",
            "runtime_api": "cuda",
            "claim": "strict_bitnet_cuda_benchmark_baseline",
            "speedup_claim": false,
            "fallback_used": false,
            "fallback_backend": null,
            "fallback_reason": null,
            "model": {
                "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
                "file": "ggml-model-i2_s.gguf",
                "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
                "loader_mode": "strict",
                "fallback_loader_used": false
            },
            "tokenizer": {
                "source": "explicit",
                "strict": true
            },
            "bitnet": {
                "quantization": "W1.58A8",
                "kernel_family": "qk256",
                "layout": "gguf_packed_i2_s",
                "weights_uploaded_once": false,
                "per_token_weight_upload": true
            },
            "workload": {
                "profile": "short_decode_8",
                "prompt": "fixture prompt",
                "prompt_tokens": 37,
                "generated_tokens": 8,
                "generated_text": "'E'E'E'E'E'E'E'E",
                "cpu_cuda_output_match": true
            },
            "comparison_contract": {
                "same_model": true,
                "same_tokenizer": true,
                "same_prompt": true,
                "same_generated_token_count": true,
                "same_strict_loader_mode": true,
                "same_sampling_policy": true,
                "fallback_free": true
            },
            "cuda": {
                "available": true,
                "device_count": 1,
                "device_index": 0,
                "device_name": "NVIDIA GeForce RTX 5070 Ti",
                "compute_capability": "12.0",
                "driver_version": "591.86",
                "cuda_runtime_version": "12.9",
                "cuda_toolkit_version": "12.9",
                "nvrtc_version": "12.9",
                "vram_bytes": 17094475776u64,
                "memory_hwm_bytes": 5949620224u64,
                "cuda_kernel_invocations": 1680
            },
            "benchmark": {
                "profile": "short_decode_8",
                "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
                "cuda_backend": "nvidia-rtx-5070-ti-cuda",
                "cpu_avx512_total_ms": 141559.0,
                "cuda_total_ms": 190129.0,
                "cpu_avx512_tokens_per_second": 0.0565,
                "cuda_tokens_per_second": 0.0421,
                "cpu_avx512_total_ms_div_cuda_total_ms": 0.7445,
                "cuda_kernel_invocations": 1680,
                "cpu_cuda_output_match": true,
                "speedup_claim": false
            },
            "profiles": [
                not_run_bitnet_profile("amd-9950x3d-cpu-scalar"),
                not_run_bitnet_profile("amd-9950x3d-cpu-avx2"),
                measured_bitnet_profile("amd-9950x3d-cpu-avx512", "cpu"),
                measured_bitnet_profile("nvidia-rtx-5070-ti-cuda", "cuda")
            ],
            "kernel_stats": [
                {
                    "kernel_id": "qk256_gemv_cuda",
                    "invocations": 1680,
                    "fallback_invocations": 0,
                    "kernel_launches": 1680,
                    "kernel_time_ms": null
                }
            ],
            "claim_boundaries": [
                "speedup_claim=false; this receipt records a baseline only.",
                "CPU scalar and AVX2 strict end-to-end profiles are explicitly present but not_run because this CLI path does not expose selectors for those modes."
            ],
            "artifact_path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-benchmark.json"
        })
    }

    fn measured_bitnet_profile(backend: &str, runtime_api: &str) -> serde_json::Value {
        json!({
            "backend": backend,
            "runtime_api": runtime_api,
            "status": "measured",
            "total_ms": 1.0,
            "first_token_ms": 1.0,
            "tokens_per_second": 1.0,
            "prompt_tokens": 37,
            "generated_tokens": 8,
            "fallback_used": false
        })
    }

    fn not_run_bitnet_profile(backend: &str) -> serde_json::Value {
        json!({
            "backend": backend,
            "runtime_api": "cpu",
            "status": "not_run",
            "reason": "current CLI does not expose a strict end-to-end selector for this CPU SIMD mode"
        })
    }
}
