#![allow(non_snake_case, clippy::manual_div_ceil)]
//! Wave 17 snapshot tests for `bitnet-kernels` — kernel configs,
//! capability matrix types, SIMD detection, quantization formats,
//! RoPE/attention/pooling/reduction configs, and activation types.
//!
//! Pins Debug representations of key configuration structs so that
//! unintentional changes to public-facing types are caught at review time.

// =========================================================================
// Section 1 — Quantization format descriptions
// =========================================================================

use bitnet_common::types::QuantizationType;

#[test]
fn snapshot_wave17__quantization_type_i2s_display() {
    insta::assert_snapshot!(format!("{}", QuantizationType::I2S));
}

#[test]
fn snapshot_wave17__quantization_type_tl1_display() {
    insta::assert_snapshot!(format!("{}", QuantizationType::TL1));
}

#[test]
fn snapshot_wave17__quantization_type_tl2_display() {
    insta::assert_snapshot!(format!("{}", QuantizationType::TL2));
}

#[test]
fn snapshot_wave17__quantization_type_i2s_debug() {
    insta::assert_debug_snapshot!(QuantizationType::I2S);
}

#[test]
fn snapshot_wave17__quantization_type_tl1_debug() {
    insta::assert_debug_snapshot!(QuantizationType::TL1);
}

#[test]
fn snapshot_wave17__quantization_type_tl2_debug() {
    insta::assert_debug_snapshot!(QuantizationType::TL2);
}

// =========================================================================
// Section 2 — SIMD capability detection
// =========================================================================

use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};

#[test]
fn snapshot_wave17__simd_level_all_variants_display() {
    let levels =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];
    let output: Vec<String> = levels.iter().map(|l| format!("{l}")).collect();
    insta::assert_snapshot!(output.join("\n"));
}

#[test]
fn snapshot_wave17__simd_level_all_variants_debug() {
    let levels =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];
    insta::assert_debug_snapshot!(levels);
}

#[test]
fn snapshot_wave17__kernel_backend_all_variants_debug() {
    let backends = [
        KernelBackend::CpuRust,
        KernelBackend::Cuda,
        KernelBackend::Hip,
        KernelBackend::OneApi,
        KernelBackend::OpenCL,
    ];
    insta::assert_debug_snapshot!(backends);
}

#[test]
fn snapshot_wave17__kernel_capabilities_cpu_only() {
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
        simd_level: SimdLevel::Avx2,
    };
    insta::assert_debug_snapshot!(caps);
}

#[test]
fn snapshot_wave17__kernel_capabilities_full_gpu() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        opencl_compiled: true,
        opencl_runtime: true,
        cpp_ffi: true,
        simd_level: SimdLevel::Avx512,
    };
    insta::assert_debug_snapshot!(caps);
}

// =========================================================================
// Section 3 — Activation function types
// =========================================================================

use bitnet_common::config::ActivationType;
use bitnet_kernels::cpu::ffn::FfnActivation;

#[test]
fn snapshot_wave17__activation_type_all_variants_debug() {
    let activations = [ActivationType::Silu, ActivationType::Relu2, ActivationType::Gelu];
    insta::assert_debug_snapshot!(activations);
}

#[test]
fn snapshot_wave17__ffn_activation_all_variants_debug() {
    let activations = [FfnActivation::GeLU, FfnActivation::SiLU, FfnActivation::ReLU];
    insta::assert_debug_snapshot!(activations);
}

// =========================================================================
// Section 4 — RoPE configuration for different model sizes
// =========================================================================

use bitnet_kernels::cpu::rope::RopeConfig;

#[test]
fn snapshot_wave17__rope_config_small_model() {
    let cfg = RopeConfig::new(64, 2048);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__rope_config_medium_model() {
    let cfg = RopeConfig::new(128, 4096);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__rope_config_large_model() {
    let cfg = RopeConfig::new(128, 8192).with_base(500_000.0);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__rope_config_with_scaling() {
    let cfg = RopeConfig::new(128, 32768).with_base(10_000.0).with_scaling_factor(4.0);
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 5 — Attention configuration
// =========================================================================

use bitnet_kernels::cpu::attention::{AttentionConfig, CpuAttentionConfig, GqaConfig};

#[test]
fn snapshot_wave17__attention_config_standard_mha() {
    let cfg = AttentionConfig {
        num_heads: 32,
        head_dim: 128,
        seq_len: 512,
        causal: true,
        use_alibi: false,
        scale: None,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__attention_config_small() {
    let cfg = AttentionConfig {
        num_heads: 8,
        head_dim: 64,
        seq_len: 256,
        causal: false,
        use_alibi: false,
        scale: Some(0.1),
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__gqa_config_4_to_1_ratio() {
    let cfg = GqaConfig {
        num_q_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        seq_len: 1024,
        causal: true,
        scale: None,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__gqa_config_mqa() {
    let cfg = GqaConfig {
        num_q_heads: 32,
        num_kv_heads: 1,
        head_dim: 128,
        seq_len: 2048,
        causal: true,
        scale: None,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__cpu_attention_config_batched() {
    let cfg = CpuAttentionConfig {
        batch_size: 4,
        num_heads: 16,
        seq_len: 512,
        head_dim: 64,
        scale: None,
        causal_mask: true,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__cpu_attention_config_single() {
    let cfg = CpuAttentionConfig {
        batch_size: 1,
        num_heads: 32,
        seq_len: 128,
        head_dim: 128,
        scale: Some(0.088),
        causal_mask: false,
    };
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 6 — Pooling operation parameters
// =========================================================================

use bitnet_kernels::cpu::pooling::{PoolConfig, PoolType};

#[test]
fn snapshot_wave17__pool_type_all_variants_debug() {
    let types = [
        PoolType::Max,
        PoolType::Average,
        PoolType::GlobalMax,
        PoolType::GlobalAverage,
        PoolType::AvgPoolCountIncludePad,
    ];
    insta::assert_debug_snapshot!(types);
}

#[test]
fn snapshot_wave17__pool_config_max_pool_3() {
    let cfg = PoolConfig {
        pool_type: PoolType::Max,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        ceil_mode: false,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__pool_config_avg_pool_2() {
    let cfg = PoolConfig {
        pool_type: PoolType::Average,
        kernel_size: 2,
        stride: 2,
        padding: 0,
        dilation: 1,
        ceil_mode: false,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__pool_config_global_max() {
    let cfg = PoolConfig {
        pool_type: PoolType::GlobalMax,
        kernel_size: 0,
        stride: 0,
        padding: 0,
        dilation: 1,
        ceil_mode: false,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__pool_config_global_average() {
    let cfg = PoolConfig {
        pool_type: PoolType::GlobalAverage,
        kernel_size: 0,
        stride: 0,
        padding: 0,
        dilation: 1,
        ceil_mode: false,
    };
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 7 — Reduction operation parameters
// =========================================================================

use bitnet_kernels::reduction::{ReductionConfig, ReductionOp};

#[test]
fn snapshot_wave17__reduction_op_all_variants_debug() {
    let ops = [
        ReductionOp::Sum,
        ReductionOp::Max,
        ReductionOp::Min,
        ReductionOp::Mean,
        ReductionOp::L2Norm,
    ];
    insta::assert_debug_snapshot!(ops);
}

#[test]
fn snapshot_wave17__reduction_config_sum_256() {
    let cfg = ReductionConfig::new(256, 64, ReductionOp::Sum).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__reduction_config_mean_1024() {
    let cfg = ReductionConfig::new(1024, 32, ReductionOp::Mean).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__reduction_config_l2norm_512() {
    let cfg = ReductionConfig::new(512, 16, ReductionOp::L2Norm).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 8 — Shaped reduction configs
// =========================================================================

use bitnet_kernels::shaped_reduction::ShapedReductionConfig;

#[test]
fn snapshot_wave17__shaped_reduction_global_sum() {
    let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__shaped_reduction_axis_mean_keepdim() {
    let cfg = ShapedReductionConfig::new(ReductionOp::Mean, Some(1), true);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__shaped_reduction_axis_max_no_keepdim() {
    let cfg = ShapedReductionConfig::new(ReductionOp::Max, Some(0), false);
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 9 — Scatter/gather configs
// =========================================================================

use bitnet_kernels::scatter_gather::{GatherConfig, ScatterMode};

#[test]
fn snapshot_wave17__scatter_mode_all_variants_debug() {
    let modes = [ScatterMode::Assign, ScatterMode::Add, ScatterMode::Max, ScatterMode::Min];
    insta::assert_debug_snapshot!(modes);
}

#[test]
fn snapshot_wave17__gather_config_axis0() {
    let cfg = GatherConfig::new(0, (4, 8), true).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__gather_config_axis1_no_bounds_check() {
    let cfg = GatherConfig::new(1, (16, 32), false).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 10 — Device capability matrix types
// =========================================================================

use bitnet_kernels::capability_matrix::{
    CapabilityEntry, DeviceClass, DeviceProfile, OperationCategory, PrecisionSupport, SupportLevel,
};

#[test]
fn snapshot_wave17__capability_entry_full_support() {
    let entry = CapabilityEntry::new(
        OperationCategory::MatrixOps,
        PrecisionSupport::FP16,
        SupportLevel::Full(0.95),
    );
    insta::assert_debug_snapshot!(entry);
}

#[test]
fn snapshot_wave17__capability_entry_partial_support() {
    let entry = CapabilityEntry::new(
        OperationCategory::QuantizedOps,
        PrecisionSupport::INT8,
        SupportLevel::Partial("no native i2s".to_string()),
    );
    insta::assert_debug_snapshot!(entry);
}

#[test]
fn snapshot_wave17__device_profile_cpu_simd() {
    let profile = DeviceProfile {
        device_class: DeviceClass::CpuSimd,
        name: "x86_64 AVX2".to_string(),
        compute_units: 8,
        memory_gb: 32,
        capabilities: vec![
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.90),
            ),
            CapabilityEntry::new(
                OperationCategory::NormOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.95),
            ),
            CapabilityEntry::new(
                OperationCategory::ActivationOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.92),
            ),
        ],
    };
    insta::assert_debug_snapshot!(profile);
}

#[test]
fn snapshot_wave17__device_profile_nvidia_gpu() {
    let profile = DeviceProfile {
        device_class: DeviceClass::NvidiaCuda,
        name: "NVIDIA RTX 4090".to_string(),
        compute_units: 128,
        memory_gb: 24,
        capabilities: vec![
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::FP16,
                SupportLevel::Full(0.98),
            ),
            CapabilityEntry::new(
                OperationCategory::AttentionOps,
                PrecisionSupport::FP16,
                SupportLevel::Full(0.96),
            ),
        ],
    };
    insta::assert_debug_snapshot!(profile);
}

// =========================================================================
// Section 11 — Common config types
// =========================================================================

use bitnet_common::config::{InferenceConfig, ModelFormat, NormType, RopeScaling};
use bitnet_common::types::Device;

#[test]
fn snapshot_wave17__inference_config_default() {
    let cfg = InferenceConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave17__device_all_variants_debug() {
    let devices = [
        Device::Cpu,
        Device::Cuda(0),
        Device::Hip(0),
        Device::Npu,
        Device::Metal,
        Device::OpenCL(0),
    ];
    insta::assert_debug_snapshot!(devices);
}

#[test]
fn snapshot_wave17__norm_type_all_variants_debug() {
    let norms = [NormType::LayerNorm, NormType::RmsNorm];
    insta::assert_debug_snapshot!(norms);
}

#[test]
fn snapshot_wave17__model_format_all_variants_debug() {
    let formats = [ModelFormat::Gguf, ModelFormat::SafeTensors, ModelFormat::HuggingFace];
    insta::assert_debug_snapshot!(formats);
}

#[test]
fn snapshot_wave17__rope_scaling_config_debug() {
    let scaling = RopeScaling { scaling_type: "linear".to_string(), factor: 4.0 };
    insta::assert_debug_snapshot!(scaling);
}
