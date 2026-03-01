//! Structure-validation tests for the ROCm HIP kernel sources.
//!
//! These tests verify that the embedded `.hip` files are well-formed,
//! use the correct HIP API surface, and contain the expected kernel
//! entry points.  They do **not** require an AMD GPU or the ROCm
//! runtime – they operate purely on the source text.

use bitnet_rocm::kernels::{
    self, ATTENTION_SRC, ELEMENTWISE_SRC, HipKernelSource, MATMUL_SRC, RMSNORM_SRC, ROPE_SRC,
    SOFTMAX_SRC, kernel_source,
};

// ── source embedding ────────────────────────────────────────────

#[test]
fn all_sources_are_non_empty() {
    for &k in HipKernelSource::ALL {
        assert!(!k.source().is_empty(), "{k:?} source is empty");
    }
}

#[test]
fn all_sources_are_valid_utf8() {
    // `include_str!` guarantees UTF-8, but belt-and-suspenders:
    for &k in HipKernelSource::ALL {
        let src = k.source();
        assert!(std::str::from_utf8(src.as_bytes()).is_ok(), "{k:?} source is not valid UTF-8");
    }
}

#[test]
fn kernel_source_fn_matches_method() {
    for &k in HipKernelSource::ALL {
        assert_eq!(
            kernel_source(k),
            k.source(),
            "kernel_source() disagrees with .source() for {k:?}"
        );
    }
}

#[test]
fn all_variants_listed_in_all_constant() {
    assert_eq!(HipKernelSource::ALL.len(), 6, "Expected 6 kernel variants");
}

// ── HIP syntax ──────────────────────────────────────────────────

#[test]
fn all_kernels_contain_global_attribute() {
    for &k in HipKernelSource::ALL {
        assert!(k.source().contains("__global__"), "{k:?} missing __global__ kernel declaration");
    }
}

#[test]
fn all_kernels_use_hip_thread_indexing() {
    for &k in HipKernelSource::ALL {
        let src = k.source();
        let has_hip_idx = src.contains("hipThreadIdx_x")
            || src.contains("hipBlockIdx_x")
            || src.contains("hipBlockDim_x");
        assert!(has_hip_idx, "{k:?} missing HIP thread/block indexing macros");
    }
}

#[test]
fn all_kernels_include_hip_runtime() {
    for &k in HipKernelSource::ALL {
        assert!(
            k.source().contains("hip/hip_runtime.h"),
            "{k:?} missing #include <hip/hip_runtime.h>"
        );
    }
}

// ── shared memory & sync ────────────────────────────────────────

#[test]
fn matmul_uses_shared_memory() {
    assert!(MATMUL_SRC.contains("__shared__"), "matmul kernel should use shared memory for tiling");
}

#[test]
fn matmul_has_sync_barrier() {
    assert!(
        MATMUL_SRC.contains("__syncthreads()"),
        "matmul kernel should synchronise after shared-memory loads"
    );
}

#[test]
fn softmax_uses_warp_reduction() {
    assert!(SOFTMAX_SRC.contains("__shfl_down"), "softmax should use warp shuffle for reduction");
}

#[test]
fn softmax_has_shared_and_sync() {
    assert!(SOFTMAX_SRC.contains("__shared__"));
    assert!(SOFTMAX_SRC.contains("__syncthreads()"));
}

#[test]
fn rmsnorm_has_shared_and_sync() {
    assert!(RMSNORM_SRC.contains("__shared__"));
    assert!(RMSNORM_SRC.contains("__syncthreads()"));
}

#[test]
fn attention_has_shared_and_sync() {
    assert!(ATTENTION_SRC.contains("__shared__"));
    assert!(ATTENTION_SRC.contains("__syncthreads()"));
}

// ── kernel entry points ─────────────────────────────────────────

#[test]
fn matmul_contains_expected_entry_points() {
    assert!(MATMUL_SRC.contains("bitnet_matmul_i2s"));
    assert!(MATMUL_SRC.contains("bitnet_matmul_simple"));
}

#[test]
fn softmax_contains_entry_point() {
    assert!(SOFTMAX_SRC.contains("softmax_forward"));
}

#[test]
fn rmsnorm_contains_entry_point() {
    assert!(RMSNORM_SRC.contains("rmsnorm_forward"));
}

#[test]
fn rope_contains_entry_points() {
    assert!(ROPE_SRC.contains("rope_forward"));
    assert!(ROPE_SRC.contains("rope_build_tables"));
}

#[test]
fn attention_contains_entry_point() {
    assert!(ATTENTION_SRC.contains("scaled_dot_product_attention"));
}

#[test]
fn elementwise_contains_all_ops() {
    assert!(ELEMENTWISE_SRC.contains("silu_forward"));
    assert!(ELEMENTWISE_SRC.contains("gelu_forward"));
    assert!(ELEMENTWISE_SRC.contains("vec_add"));
    assert!(ELEMENTWISE_SRC.contains("vec_mul"));
}

// ── numerical safety ────────────────────────────────────────────

#[test]
fn softmax_has_numerical_stability_guard() {
    // Must subtract row-max before exp to avoid overflow
    assert!(
        SOFTMAX_SRC.contains("row_max") || SOFTMAX_SRC.contains("local_max"),
        "softmax should compute row max for numerical stability"
    );
}

#[test]
fn rmsnorm_uses_epsilon() {
    assert!(
        RMSNORM_SRC.contains("eps"),
        "rmsnorm should accept an epsilon for numerical stability"
    );
}

// ── module-level constant re-exports ────────────────────────────

#[test]
fn constant_re_exports_match_enum() {
    assert_eq!(kernels::MATMUL_SRC, HipKernelSource::Matmul.source());
    assert_eq!(kernels::SOFTMAX_SRC, HipKernelSource::Softmax.source());
    assert_eq!(kernels::RMSNORM_SRC, HipKernelSource::RmsNorm.source());
    assert_eq!(kernels::ROPE_SRC, HipKernelSource::Rope.source());
    assert_eq!(kernels::ATTENTION_SRC, HipKernelSource::Attention.source());
    assert_eq!(kernels::ELEMENTWISE_SRC, HipKernelSource::Elementwise.source());
}
