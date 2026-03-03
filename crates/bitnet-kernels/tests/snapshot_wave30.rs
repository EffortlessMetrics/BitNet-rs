//! Snapshot wave 30 — bitnet-kernels
//!
//! Pins Debug/Display representations of kernel configuration types,
//! SIMD levels, pipeline stages, and capability matrices.

use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};
use bitnet_kernels::opencl_pipeline::{PipelineConfig, PipelineStage};
use bitnet_kernels::simd_diagnostics::SimdCapabilities;

// =========================================================================
// Section 1 — SimdLevel variants
// =========================================================================

#[test]
fn snapshot_wave30__simd_level_scalar_display() {
    insta::assert_snapshot!(format!("{}", SimdLevel::Scalar));
}

#[test]
fn snapshot_wave30__simd_level_avx2_display() {
    insta::assert_snapshot!(format!("{}", SimdLevel::Avx2));
}

#[test]
fn snapshot_wave30__simd_level_avx512_display() {
    insta::assert_snapshot!(format!("{}", SimdLevel::Avx512));
}

#[test]
fn snapshot_wave30__simd_level_ordering_debug() {
    let mut levels =
        [SimdLevel::Avx512, SimdLevel::Scalar, SimdLevel::Avx2, SimdLevel::Neon, SimdLevel::Sse42];
    levels.sort();
    insta::assert_debug_snapshot!(levels);
}

// =========================================================================
// Section 2 — KernelBackend variants
// =========================================================================

#[test]
fn snapshot_wave30__kernel_backend_all_display() {
    let backends = [
        KernelBackend::CpuRust,
        KernelBackend::Cuda,
        KernelBackend::Hip,
        KernelBackend::OneApi,
        KernelBackend::OpenCL,
        KernelBackend::CppFfi,
    ];
    let output: Vec<String> = backends.iter().map(|b| format!("{b}")).collect();
    insta::assert_snapshot!(output.join("\n"));
}

#[test]
fn snapshot_wave30__kernel_backend_all_debug() {
    let backends = [
        KernelBackend::CpuRust,
        KernelBackend::Cuda,
        KernelBackend::Hip,
        KernelBackend::OneApi,
        KernelBackend::OpenCL,
        KernelBackend::CppFfi,
    ];
    insta::assert_debug_snapshot!(backends);
}

// =========================================================================
// Section 3 — KernelCapabilities
// =========================================================================

#[test]
fn snapshot_wave30__kernel_caps_cpu_scalar() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        opencl_compiled: false,
        opencl_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Scalar,
    };
    insta::assert_debug_snapshot!(caps);
}

#[test]
fn snapshot_wave30__kernel_caps_cuda_avx2() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        opencl_compiled: false,
        opencl_runtime: false,
        cpp_ffi: true,
        simd_level: SimdLevel::Avx2,
    };
    insta::assert_debug_snapshot!(caps);
}

// =========================================================================
// Section 4 — PipelineStage
// =========================================================================

#[test]
fn snapshot_wave30__pipeline_stage_all_display() {
    let stages = [
        PipelineStage::Embedding,
        PipelineStage::RmsNorm,
        PipelineStage::Attention,
        PipelineStage::FeedForward,
        PipelineStage::FinalNorm,
        PipelineStage::LogitProjection,
    ];
    let output: Vec<String> = stages.iter().map(|s| format!("{s}")).collect();
    insta::assert_snapshot!(output.join("\n"));
}

#[test]
fn snapshot_wave30__pipeline_stage_all_debug() {
    let stages = [
        PipelineStage::Embedding,
        PipelineStage::RmsNorm,
        PipelineStage::Attention,
        PipelineStage::FeedForward,
        PipelineStage::FinalNorm,
        PipelineStage::LogitProjection,
    ];
    insta::assert_debug_snapshot!(stages);
}

// =========================================================================
// Section 5 — PipelineConfig serialization
// =========================================================================

#[test]
fn snapshot_wave30__pipeline_config_small_model() {
    let cfg = PipelineConfig {
        num_layers: 12,
        hidden_dim: 768,
        num_heads: 12,
        head_dim: 64,
        intermediate_dim: 3072,
        vocab_size: 32000,
        max_seq_len: 512,
        use_gpu: false,
        fallback_to_cpu: true,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave30__pipeline_config_large_model() {
    let cfg = PipelineConfig {
        num_layers: 32,
        hidden_dim: 4096,
        num_heads: 32,
        head_dim: 128,
        intermediate_dim: 11008,
        vocab_size: 128256,
        max_seq_len: 2048,
        use_gpu: true,
        fallback_to_cpu: true,
    };
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 6 — SimdCapabilities
// =========================================================================

#[test]
fn snapshot_wave30__simd_caps_none_detected() {
    let caps = SimdCapabilities {
        sse2: false,
        sse4_1: false,
        sse4_2: false,
        avx: false,
        avx2: false,
        avx512f: false,
        avx512bw: false,
        avx512vnni: false,
        fma: false,
        neon: false,
        arch: "unknown".to_string(),
    };
    insta::assert_debug_snapshot!(caps);
}

#[test]
fn snapshot_wave30__simd_caps_avx2_typical() {
    let caps = SimdCapabilities {
        sse2: true,
        sse4_1: true,
        sse4_2: true,
        avx: true,
        avx2: true,
        avx512f: false,
        avx512bw: false,
        avx512vnni: false,
        fma: true,
        neon: false,
        arch: "x86_64".to_string(),
    };
    insta::assert_debug_snapshot!(caps);
}

#[test]
fn snapshot_wave30__pipeline_config_gpu_enabled() {
    let cfg = PipelineConfig {
        num_layers: 24,
        hidden_dim: 2048,
        num_heads: 16,
        head_dim: 128,
        intermediate_dim: 5504,
        vocab_size: 32000,
        max_seq_len: 4096,
        use_gpu: true,
        fallback_to_cpu: false,
    };
    insta::assert_debug_snapshot!(cfg);
}
