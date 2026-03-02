//! Wave 25 snapshot tests for bitnet-kernels.
//!
//! Covers: CUDA kernel launch configurations, CPU SIMD operation results,
//! kernel registry state, and performance metrics snapshots.

// =========================================================================
// Section 1 — CUDA fusion launch configurations
// =========================================================================

use bitnet_kernels::cuda::fusion::{
    FusedElementwiseLaunchConfig, FusedMatmulLaunchConfig, FusedOp, FusionConfig, FusionError,
};

#[test]
fn w25_fusion_config_default_debug() {
    let cfg = FusionConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_fusion_config_disabled_debug() {
    let cfg = FusionConfig::disabled();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_fused_op_all_kernel_names() {
    let ops = [
        FusedOp::RmsNormLinear,
        FusedOp::GeluLinear,
        FusedOp::SoftmaxMask,
        FusedOp::AddNormalize,
        FusedOp::ScaleAndAdd,
    ];
    let names: Vec<&str> = ops.iter().map(|o| o.kernel_name()).collect();
    insta::assert_debug_snapshot!(names);
}

#[test]
fn w25_fused_op_display_strings() {
    let ops = [
        FusedOp::RmsNormLinear,
        FusedOp::GeluLinear,
        FusedOp::SoftmaxMask,
        FusedOp::AddNormalize,
        FusedOp::ScaleAndAdd,
    ];
    let displays: Vec<String> = ops.iter().map(|o| o.to_string()).collect();
    insta::assert_debug_snapshot!(displays);
}

#[test]
fn w25_fused_matmul_launch_small() {
    let cfg = FusedMatmulLaunchConfig::new(64, 32).unwrap();
    insta::assert_snapshot!(format!(
        "n={} out_dim={} tpb={} grid={:?} block={:?} shmem={}",
        cfg.n,
        cfg.out_dim,
        cfg.threads_per_block,
        cfg.grid_dim(),
        cfg.block_dim(),
        cfg.shared_mem_bytes()
    ));
}

#[test]
fn w25_fused_matmul_launch_large() {
    let cfg = FusedMatmulLaunchConfig::new(4096, 2048).unwrap();
    insta::assert_snapshot!(format!(
        "n={} out_dim={} tpb={} grid={:?} block={:?} shmem={}",
        cfg.n,
        cfg.out_dim,
        cfg.threads_per_block,
        cfg.grid_dim(),
        cfg.block_dim(),
        cfg.shared_mem_bytes()
    ));
}

#[test]
fn w25_fused_elementwise_launch_small() {
    let cfg = FusedElementwiseLaunchConfig::new(512).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_fused_elementwise_launch_large() {
    let cfg = FusedElementwiseLaunchConfig::new(1_048_576).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_fusion_error_dimension_mismatch_display() {
    let err = FusionError::DimensionMismatch { expected: 768, got: 512 };
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn w25_fusion_error_invalid_config_display() {
    let err = FusionError::InvalidConfig("min_fusion_size must be > 0".into());
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn w25_fusion_error_empty_input_display() {
    let err = FusionError::EmptyInput;
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn w25_fusion_config_validate_zero_size() {
    let cfg = FusionConfig { min_fusion_size: 0, ..FusionConfig::default() };
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

// =========================================================================
// Section 2 — CUDA matmul launch configurations
// =========================================================================

use bitnet_kernels::cuda::matmul::{MatmulConfig, MatmulDtype};

#[test]
fn w25_matmul_config_default_debug() {
    let cfg = MatmulConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_matmul_config_for_shape_small() {
    let cfg = MatmulConfig::for_shape(128, 256, 64).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_matmul_config_for_shape_large() {
    let cfg = MatmulConfig::for_shape(4096, 4096, 4096).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_matmul_dtype_all_variants() {
    let dtypes = [MatmulDtype::F32, MatmulDtype::F16];
    insta::assert_debug_snapshot!(dtypes);
}

// =========================================================================
// Section 3 — CUDA gating configurations
// =========================================================================

use bitnet_kernels::cuda::gating::{GatingConfig, GatingType};

#[test]
fn w25_gating_type_all_variants_debug() {
    let types = [GatingType::SwiGLU, GatingType::GeGLU, GatingType::ReGLU];
    insta::assert_debug_snapshot!(types);
}

#[test]
fn w25_gating_type_kernel_names() {
    let names: Vec<&str> = [GatingType::SwiGLU, GatingType::GeGLU, GatingType::ReGLU]
        .iter()
        .map(|g| g.kernel_name())
        .collect();
    insta::assert_debug_snapshot!(names);
}

#[test]
fn w25_gating_config_swiglu() {
    let cfg = GatingConfig::new(4096, GatingType::SwiGLU).unwrap();
    insta::assert_snapshot!(format!(
        "n={} tpb={} gating={:?} grid={:?} block={:?}",
        cfg.n,
        cfg.threads_per_block,
        cfg.gating,
        cfg.grid_dim(),
        cfg.block_dim()
    ));
}

#[test]
fn w25_gating_config_geglu() {
    let cfg = GatingConfig::new(2048, GatingType::GeGLU).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 4 — CUDA memory pool configuration
// =========================================================================

use bitnet_kernels::cuda::memory_pool::MemoryPoolConfig;

#[test]
fn w25_memory_pool_config_default_debug() {
    let cfg = MemoryPoolConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_memory_pool_config_custom() {
    let cfg = MemoryPoolConfig {
        initial_size: 128 * 1024 * 1024,
        max_size: 2 * 1024 * 1024 * 1024,
        block_size: 512,
        alignment: 512,
    };
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 5 — CUDA attention kernel configuration
// =========================================================================

use bitnet_kernels::cuda::attention::AttentionKernelConfig;

#[test]
fn w25_attention_kernel_config_small() {
    let cfg = AttentionKernelConfig::for_shape(12, 64, 128, 128, true).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_attention_kernel_config_decode() {
    let cfg = AttentionKernelConfig::for_shape(32, 128, 1, 2048, true).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 6 — CUDA layernorm configuration
// =========================================================================

use bitnet_kernels::cuda::layernorm::LayerNormConfig;

#[test]
fn w25_layernorm_config_default_debug() {
    let cfg = LayerNormConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_layernorm_config_custom_eps() {
    let cfg = LayerNormConfig::new(1e-6, false).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 7 — CPU fusion (mirroring CUDA fusion)
// =========================================================================

use bitnet_kernels::cpu::fusion::{
    FusedOp as CpuFusedOp, FusionConfig as CpuFusionConfig, FusionError as CpuFusionError,
};

#[test]
fn w25_cpu_fusion_config_default() {
    let cfg = CpuFusionConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_cpu_fusion_config_disabled() {
    let cfg = CpuFusionConfig::disabled();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_cpu_fused_op_all_display() {
    let ops = [
        CpuFusedOp::RmsNormLinear,
        CpuFusedOp::GeluLinear,
        CpuFusedOp::SoftmaxMask,
        CpuFusedOp::AddNormalize,
        CpuFusedOp::ScaleAndAdd,
    ];
    let displays: Vec<String> = ops.iter().map(|o| o.to_string()).collect();
    insta::assert_debug_snapshot!(displays);
}

#[test]
fn w25_cpu_fusion_error_dimension_mismatch() {
    let err = CpuFusionError::DimensionMismatch { expected: 1024, got: 768 };
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn w25_cpu_fusion_error_empty_input() {
    let err = CpuFusionError::EmptyInput;
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn w25_cpu_fusion_validate_zero() {
    let cfg = CpuFusionConfig { min_fusion_size: 0, ..CpuFusionConfig::default() };
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

// =========================================================================
// Section 8 — CPU SIMD matmul / tiling
// =========================================================================

use bitnet_kernels::cpu::simd_matmul::{SimdMatmulConfig, TileConfig};

#[test]
fn w25_tile_config_default_debug() {
    let cfg = TileConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_tile_config_custom() {
    let cfg = TileConfig::new(64, 128, 16);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_simd_matmul_config_square() {
    let cfg = SimdMatmulConfig::new(512, 512, 512);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_simd_matmul_config_transposed() {
    let cfg = SimdMatmulConfig {
        m: 128,
        n: 256,
        k: 64,
        alpha: 2.0,
        beta: 0.5,
        transpose_a: true,
        transpose_b: false,
    };
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 9 — OpenCL numerical stability
// =========================================================================

use bitnet_kernels::opencl_numerical_stability::{NumericalProfile, StabilityConfig};

#[test]
fn w25_stability_config_default_debug() {
    let cfg = StabilityConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_numerical_profile_clean_data() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
    let profile = NumericalProfile::compute(&data);
    insta::assert_snapshot!(profile.to_string());
}

#[test]
fn w25_numerical_profile_empty_data() {
    let profile = NumericalProfile::compute(&[]);
    insta::assert_snapshot!(profile.to_string());
}

#[test]
fn w25_numerical_profile_with_nans() {
    let data = vec![1.0, f32::NAN, 2.0, f32::NAN, 3.0];
    let profile = NumericalProfile::compute(&data);
    insta::assert_snapshot!(format!(
        "nan_count={} inf_count={} is_clean={}",
        profile.nan_count,
        profile.inf_count,
        profile.is_clean()
    ));
}

#[test]
fn w25_numerical_profile_with_infs() {
    let data = vec![f32::INFINITY, f32::NEG_INFINITY, 0.0];
    let profile = NumericalProfile::compute(&data);
    insta::assert_snapshot!(format!(
        "nan_count={} inf_count={} is_clean={}",
        profile.nan_count,
        profile.inf_count,
        profile.is_clean()
    ));
}

// =========================================================================
// Section 10 — CUDA pooling configuration
// =========================================================================

use bitnet_kernels::cuda::pooling::{CudaPoolType, PoolingConfig};

#[test]
fn w25_cuda_pool_type_all_variants() {
    let types = [CudaPoolType::Max, CudaPoolType::Average];
    insta::assert_debug_snapshot!(types);
}

#[test]
fn w25_pooling_config_max_pool() {
    let cfg = PoolingConfig::new(CudaPoolType::Max, 1024, 3, 2, 1).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_pooling_config_avg_pool() {
    let cfg = PoolingConfig::new(CudaPoolType::Average, 512, 4, 4, 0).unwrap();
    insta::assert_debug_snapshot!(cfg);
}
