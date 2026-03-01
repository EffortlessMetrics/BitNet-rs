//! Snapshot tests for `bitnet-common` public API surface.
//!
//! These tests pin the Display / Debug formats of key types so that
//! unintentional changes are caught at review time.

use bitnet_common::BitNetConfig;
use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};

// ---------------------------------------------------------------------------
// BitNetConfig
// ---------------------------------------------------------------------------

#[test]
fn bitnet_config_default_json_snapshot() {
    // Pin the serialized default config so that any schema change is visible at review time.
    let cfg = BitNetConfig::default();
    insta::assert_json_snapshot!("bitnet_config_default", cfg);
}

// ---------------------------------------------------------------------------
// SimdLevel
// ---------------------------------------------------------------------------

#[test]
fn simd_level_display_all_variants() {
    let levels =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];
    let displays: Vec<String> = levels.iter().map(|l| l.to_string()).collect();
    insta::assert_debug_snapshot!("simd_level_display_variants", displays);
}

#[test]
fn simd_level_ordering_is_ascending() {
    // Snapshot the sorted order to document the contract: Scalar < Neon < SSE4.2 < AVX2 < AVX512
    let ordered = {
        let mut v = [
            SimdLevel::Avx512,
            SimdLevel::Scalar,
            SimdLevel::Neon,
            SimdLevel::Avx2,
            SimdLevel::Sse42,
        ];
        v.sort();
        v.iter().map(|l| format!("{l}")).collect::<Vec<_>>()
    };
    insta::assert_debug_snapshot!("simd_level_sorted_order", ordered);
}

// ---------------------------------------------------------------------------
// KernelBackend
// ---------------------------------------------------------------------------

#[test]
fn kernel_backend_display_all_variants() {
    let backends = [KernelBackend::CpuRust, KernelBackend::Cuda, KernelBackend::CppFfi];
    let displays: Vec<String> = backends.iter().map(|b| b.to_string()).collect();
    insta::assert_debug_snapshot!("kernel_backend_display_variants", displays);
}

// ---------------------------------------------------------------------------
// KernelCapabilities
// ---------------------------------------------------------------------------

#[test]
fn kernel_capabilities_cpu_only_snapshot() {
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
    insta::assert_debug_snapshot!("kernel_capabilities_cpu_avx2", caps);
}

#[test]
fn kernel_capabilities_from_compile_time_is_deterministic() {
    // Two calls must return identical results.
    let a = KernelCapabilities::from_compile_time();
    let b = KernelCapabilities::from_compile_time();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}

// == Wave 4: memory_pool =====================================================

use bitnet_common::memory_pool::{PoolStats, TensorPool};

#[test]
fn pool_stats_default_snapshot() {
    let stats = PoolStats::default();
    insta::assert_debug_snapshot!("pool_stats_default", stats);
}

#[test]
fn pool_stats_after_allocations() {
    let pool = TensorPool::new(4096);
    let _buf1 = pool.allocate(100);
    let _buf2 = pool.allocate(200);
    let stats = pool.stats();
    insta::assert_snapshot!(
        "pool_stats_after_two_allocs",
        format!(
            "hits={} misses={} total_allocations={} active_bytes={} pooled_bytes={}",
            stats.hits,
            stats.misses,
            stats.total_allocations(),
            stats.active_bytes,
            stats.pooled_bytes,
        )
    );
}

#[test]
fn pool_stats_hit_after_recycle() {
    let pool = TensorPool::new(4096);
    let buf = pool.allocate(64);
    drop(buf);
    let _buf2 = pool.allocate(64);
    let stats = pool.stats();
    insta::assert_snapshot!(
        "pool_stats_hit_after_recycle",
        format!(
            "hits={} misses={} active_bytes={} pooled_bytes={}",
            stats.hits, stats.misses, stats.active_bytes, stats.pooled_bytes,
        )
    );
}

#[test]
fn pool_stats_after_clear() {
    let pool = TensorPool::new(4096);
    let buf = pool.allocate(128);
    drop(buf);
    pool.clear();
    let stats = pool.stats();
    insta::assert_debug_snapshot!("pool_stats_after_clear", stats);
}

// == Wave 4: tensor_validation ===============================================

use bitnet_common::tensor_validation::{
    ShapeError, broadcast_shape, validate_attention_shapes, validate_matmul_shapes,
    validate_reshape, validate_transpose_axes,
};

#[test]
fn broadcast_shape_scalar_with_matrix() {
    let result = broadcast_shape(&[], &[3, 4]).unwrap();
    insta::assert_debug_snapshot!("broadcast_scalar_with_matrix", result);
}

#[test]
fn broadcast_shape_expand_ones() {
    let result = broadcast_shape(&[1, 4], &[3, 1]).unwrap();
    insta::assert_debug_snapshot!("broadcast_expand_ones", result);
}

#[test]
fn broadcast_shape_incompatible_error() {
    let err = broadcast_shape(&[3], &[4]).unwrap_err();
    insta::assert_snapshot!("broadcast_incompatible_error", format!("{err}"));
}

#[test]
fn matmul_shapes_2d_valid() {
    let result = validate_matmul_shapes(&[2, 3], &[3, 4]).unwrap();
    insta::assert_debug_snapshot!("matmul_shapes_2d_valid", result);
}

#[test]
fn matmul_shapes_batched_valid() {
    let result = validate_matmul_shapes(&[2, 3, 4], &[2, 4, 5]).unwrap();
    insta::assert_debug_snapshot!("matmul_shapes_batched_valid", result);
}

#[test]
fn matmul_shapes_mismatch_error() {
    let err = validate_matmul_shapes(&[2, 3], &[4, 5]).unwrap_err();
    insta::assert_snapshot!("matmul_shapes_mismatch_error", format!("{err}"));
}

#[test]
fn attention_shapes_valid_gqa() {
    let result = validate_attention_shapes(&[1, 8, 16, 64], &[1, 4, 16, 64], &[1, 4, 16, 64]);
    insta::assert_snapshot!("attention_shapes_valid_gqa", format!("{result:?}"));
}

#[test]
fn attention_shapes_head_dim_mismatch() {
    let err =
        validate_attention_shapes(&[1, 8, 16, 64], &[1, 8, 16, 32], &[1, 8, 16, 32]).unwrap_err();
    insta::assert_snapshot!("attention_shapes_head_dim_mismatch", format!("{err}"));
}

#[test]
fn reshape_valid() {
    let result = validate_reshape(&[2, 3, 4], &[6, 4]);
    insta::assert_snapshot!("reshape_valid", format!("{result:?}"));
}

#[test]
fn reshape_element_count_mismatch() {
    let err = validate_reshape(&[2, 3], &[4, 4]).unwrap_err();
    insta::assert_snapshot!("reshape_element_count_mismatch", format!("{err}"));
}

#[test]
fn transpose_valid() {
    let result = validate_transpose_axes(&[2, 3, 4], &[2, 0, 1]).unwrap();
    insta::assert_debug_snapshot!("transpose_valid", result);
}

#[test]
fn transpose_axis_out_of_range() {
    let err = validate_transpose_axes(&[2, 3], &[0, 5]).unwrap_err();
    insta::assert_snapshot!("transpose_axis_out_of_range", format!("{err}"));
}

#[test]
fn shape_error_variants_display() {
    let errors: Vec<ShapeError> = vec![
        ShapeError::MatmulMismatch { a_inner: 3, b_inner: 4 },
        ShapeError::MatmulRank { a_ndim: 0, b_ndim: 2 },
        ShapeError::BroadcastIncompatible { dim: 1, a: 3, b: 4 },
        ShapeError::ReshapeElementCount { from_count: 6, to: vec![4, 4], to_count: 16 },
        ShapeError::TransposeAxisOutOfRange { axis: 5, ndim: 3 },
        ShapeError::EmptyShape,
    ];
    let displays: Vec<String> = errors.iter().map(|e| format!("{e}")).collect();
    insta::assert_debug_snapshot!("shape_error_variants_display", displays);
}
