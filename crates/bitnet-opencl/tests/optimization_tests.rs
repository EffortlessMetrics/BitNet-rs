//! Tests for optimized OpenCL kernel variants and the kernel selector.

use bitnet_opencl::kernel_selector::{KernelSelector, KernelVariant};
use bitnet_opencl::kernels;

// ---------------------------------------------------------------------------
// KernelSelector – matmul
// ---------------------------------------------------------------------------

#[test]
fn matmul_naive_for_small_sizes() {
    let sel = KernelSelector::new(256, false, 4);
    assert_eq!(sel.select_matmul(4, 4, 4), KernelVariant::Naive);
}

#[test]
fn matmul_naive_for_boundary_below_threshold() {
    let sel = KernelSelector::new(256, false, 4);
    // 63×63 < 64×64 threshold
    assert_eq!(sel.select_matmul(63, 63, 8), KernelVariant::Naive);
}

#[test]
fn matmul_vectorized_for_large_sizes() {
    let sel = KernelSelector::new(256, false, 4);
    assert_eq!(sel.select_matmul(512, 512, 512), KernelVariant::Vectorized,);
}

#[test]
fn matmul_tiled_when_vector_width_insufficient() {
    let sel = KernelSelector::new(256, false, 1); // vector width < 4
    assert_eq!(sel.select_matmul(256, 256, 256), KernelVariant::Tiled);
}

#[test]
fn matmul_subgroup_when_supported_and_large() {
    let sel = KernelSelector::new(256, true, 1); // subgroups, narrow vec
    assert_eq!(sel.select_matmul(256, 256, 256), KernelVariant::Subgroup,);
}

#[test]
fn matmul_non_power_of_two_dimensions() {
    let sel = KernelSelector::new(256, false, 4);
    // 100×200×300 is large enough for vectorized
    let variant = sel.select_matmul(100, 200, 300);
    assert_eq!(variant, KernelVariant::Vectorized);
}

// ---------------------------------------------------------------------------
// KernelSelector – softmax
// ---------------------------------------------------------------------------

#[test]
fn softmax_naive_for_small_n() {
    let sel = KernelSelector::new(256, false, 4);
    assert_eq!(sel.select_softmax(64), KernelVariant::Naive);
}

#[test]
fn softmax_vectorized_for_aligned_large_n() {
    let sel = KernelSelector::new(256, false, 4);
    // 1024 is large and divisible by 4
    assert_eq!(sel.select_softmax(1024), KernelVariant::Vectorized);
}

#[test]
fn softmax_tiled_for_misaligned_large_n() {
    let sel = KernelSelector::new(256, false, 4);
    // 1023 is not divisible by 4 → falls back to Tiled
    assert_eq!(sel.select_softmax(1023), KernelVariant::Tiled);
}

// ---------------------------------------------------------------------------
// KernelSelector – attention
// ---------------------------------------------------------------------------

#[test]
fn attention_naive_for_short_seq() {
    let sel = KernelSelector::new(256, false, 4);
    assert_eq!(sel.select_attention(16, 64), KernelVariant::Naive,);
}

#[test]
fn attention_vectorized_for_medium_seq() {
    let sel = KernelSelector::new(256, false, 4);
    assert_eq!(sel.select_attention(128, 64), KernelVariant::Vectorized,);
}

#[test]
fn attention_subgroup_when_supported() {
    let sel = KernelSelector::new(256, true, 4);
    assert_eq!(sel.select_attention(128, 64), KernelVariant::Subgroup,);
}

#[test]
fn attention_tiled_when_seq_exceeds_local_memory() {
    let sel = KernelSelector::new(256, true, 4);
    // Very long seq: local bytes = (16384 + 256) * 4 ≈ 66 KB > 48 KB
    assert_eq!(sel.select_attention(16384, 64), KernelVariant::Tiled,);
}

// ---------------------------------------------------------------------------
// Tile size clamping
// ---------------------------------------------------------------------------

#[test]
fn tile_size_clamped_to_device_limits() {
    // max_work_group_size = 64 → sqrt(64) = 8, so tile = 8
    let sel = KernelSelector::new(64, false, 4);
    assert_eq!(sel.tile_size(), 8);
}

#[test]
fn tile_size_default_when_device_large_enough() {
    let sel = KernelSelector::new(1024, false, 4);
    assert_eq!(sel.tile_size(), kernels::DEFAULT_TILE_SIZE);
}

// ---------------------------------------------------------------------------
// Kernel source embedding
// ---------------------------------------------------------------------------

#[test]
fn matmul_kernel_source_contains_entry_point() {
    assert!(kernels::MATMUL_VECTORIZED_SRC.contains("__kernel void matmul_vec4("));
}

#[test]
fn matmul_kernel_source_contains_wide_entry_point() {
    assert!(kernels::MATMUL_VECTORIZED_SRC.contains("__kernel void matmul_vec4_wide("));
}

#[test]
fn softmax_kernel_source_contains_entry_point() {
    assert!(kernels::SOFTMAX_OPTIMIZED_SRC.contains("__kernel void softmax_stable("));
}

#[test]
fn softmax_kernel_source_contains_vec4_entry_point() {
    assert!(kernels::SOFTMAX_OPTIMIZED_SRC.contains("__kernel void softmax_stable_vec4("));
}

#[test]
fn attention_kernel_source_contains_entry_point() {
    assert!(kernels::ATTENTION_OPTIMIZED_SRC.contains("__kernel void fused_attention("));
}

#[test]
fn attention_kernel_source_contains_causal_entry_point() {
    assert!(kernels::ATTENTION_OPTIMIZED_SRC.contains("__kernel void fused_attention_causal("));
}

// ---------------------------------------------------------------------------
// Kernel source – structural checks
// ---------------------------------------------------------------------------

#[test]
fn matmul_kernel_uses_fma() {
    assert!(
        kernels::MATMUL_VECTORIZED_SRC.contains("fma("),
        "vectorized matmul should use fma for fused multiply-add"
    );
}

#[test]
fn softmax_kernel_uses_local_memory_fence() {
    assert!(
        kernels::SOFTMAX_OPTIMIZED_SRC.contains("barrier(CLK_LOCAL_MEM_FENCE)"),
        "softmax must synchronise local memory accesses"
    );
}

#[test]
fn attention_kernel_applies_scaling() {
    assert!(
        kernels::ATTENTION_OPTIMIZED_SRC.contains("* scale"),
        "attention must scale Q·K^T dot products"
    );
}

// ---------------------------------------------------------------------------
// Hardware-specific tests (need real OpenCL runtime)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires OpenCL runtime - run manually with --ignored"]
fn matmul_correctness_on_device() {
    // Placeholder: enqueue matmul_vec4, readback, compare to CPU reference.
}

#[test]
#[ignore = "requires OpenCL runtime - run manually with --ignored"]
fn softmax_numerical_stability_large_values() {
    // Placeholder: feed [1e30, 1e30, …] into softmax_stable, verify ≈ 1/N.
}

#[test]
#[ignore = "requires OpenCL runtime - run manually with --ignored"]
fn attention_output_matches_reference() {
    // Placeholder: compare fused_attention output to naive Q·K^T·V.
}
