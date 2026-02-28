//! Unit tests verifying OpenCL kernel source correctness and contracts.
//!
//! These tests validate the .cl kernel sources and their expected behavior
//! without requiring an actual OpenCL runtime or GPU hardware.

use bitnet_kernels::kernels;

// === Kernel Source Validation ===

#[test]
fn matmul_i2s_kernel_has_correct_signature() {
    let src = kernels::MATMUL_I2S_SRC;
    assert!(src.contains("__kernel void matmul_i2s"));
    assert!(src.contains("__global const char* A"));
    assert!(src.contains("__global const uchar* B"));
    assert!(src.contains("__global float* C"));
    assert!(src.contains("const uint M"));
    assert!(src.contains("const uint N"));
    assert!(src.contains("const uint K"));
}

#[test]
fn matmul_i2s_kernel_uses_global_id() {
    let src = kernels::MATMUL_I2S_SRC;
    assert!(src.contains("get_global_id(0)"), "should use global work item ID for row");
    assert!(src.contains("get_global_id(1)"), "should use global work item ID for col");
}

#[test]
fn matmul_i2s_kernel_has_bounds_check() {
    let src = kernels::MATMUL_I2S_SRC;
    assert!(
        src.contains("if (row >= M") || src.contains("if(row >= M"),
        "kernel should check row bounds"
    );
    assert!(src.contains("col >= N"), "kernel should check col bounds");
}

#[test]
fn matmul_i2s_kernel_unpacks_2bit_values() {
    let src = kernels::MATMUL_I2S_SRC;
    // Should unpack 4 values per byte (2 bits each)
    assert!(src.contains("& 0x03") || src.contains("& 3"), "should mask to 2 bits");
    assert!(
        src.contains(">> (sub * 2)") || src.contains(">> (sub*2)"),
        "should shift by 2 bits per sub-element"
    );
}

#[test]
fn quantize_i2s_kernel_has_correct_signature() {
    let src = kernels::QUANTIZE_I2S_SRC;
    assert!(src.contains("__kernel void quantize_i2s"));
    assert!(src.contains("__global const float* input"));
    assert!(src.contains("__global uchar* output"));
    assert!(src.contains("__global float* scales"));
}

#[test]
fn quantize_i2s_kernel_computes_absmax() {
    let src = kernels::QUANTIZE_I2S_SRC;
    assert!(src.contains("fabs(") || src.contains("fabs ("), "should compute absolute value");
    assert!(src.contains("fmax(") || src.contains("fmax ("), "should find max via fmax");
}

#[test]
fn quantize_i2s_kernel_handles_zero_scale() {
    let src = kernels::QUANTIZE_I2S_SRC;
    assert!(
        src.contains("absmax > 0") || src.contains("absmax != 0"),
        "should guard against zero scale"
    );
}

#[test]
fn elementwise_kernels_have_bounds_checks() {
    let src = kernels::ELEMENTWISE_SRC;
    assert!(src.contains("if (i < N)"), "vec_add should check bounds");
}

#[test]
fn rms_norm_kernel_has_epsilon() {
    let src = kernels::ELEMENTWISE_SRC;
    assert!(src.contains("eps"), "rms_norm should use epsilon for numerical stability");
    assert!(src.contains("rsqrt("), "rms_norm should use rsqrt");
}

#[test]
fn silu_kernel_uses_sigmoid() {
    let src = kernels::ELEMENTWISE_SRC;
    assert!(
        src.contains("exp(-x)") || src.contains("exp(- x)"),
        "silu should compute sigmoid via exp(-x)"
    );
}

// === Ternary Encoding Contract Tests ===

#[test]
fn matmul_i2s_encoding_contract() {
    // Verify the 2-bit encoding used in the kernel matches our CPU implementation:
    // 0b00 = 0, 0b01 = +1, 0b11 = -1, 0b10 = unused (treated as 0)
    let src = kernels::MATMUL_I2S_SRC;
    assert!(src.contains("0x01") && src.contains("w = 1"), "0b01 should map to +1");
    assert!(src.contains("0x03") && src.contains("w = -1"), "0b11 should map to -1");
}

#[test]
fn quantize_ternary_encoding_contract() {
    let src = kernels::QUANTIZE_I2S_SRC;
    // Quantize should produce the same encoding
    assert!(
        src.contains("ternary = 1") || src.contains("ternary = 1;"),
        "positive values should quantize to 1 (0b01)"
    );
    assert!(
        src.contains("ternary = 3") || src.contains("ternary = 3;"),
        "negative values should quantize to 3 (0b11 = -1)"
    );
    assert!(
        src.contains("ternary = 0") || src.contains("ternary = 0;"),
        "near-zero values should quantize to 0 (0b00)"
    );
}

// === Kernel Source Hygiene ===

#[test]
fn no_kernel_uses_printf() {
    assert!(!kernels::MATMUL_I2S_SRC.contains("printf"), "matmul should not use printf");
    assert!(!kernels::QUANTIZE_I2S_SRC.contains("printf"), "quantize should not use printf");
    assert!(!kernels::ELEMENTWISE_SRC.contains("printf"), "elementwise should not use printf");
}

#[test]
fn no_kernel_uses_barrier_incorrectly() {
    // These are single-workitem kernels - barrier() would deadlock
    for (name, src) in
        [("matmul", kernels::MATMUL_I2S_SRC), ("quantize", kernels::QUANTIZE_I2S_SRC)]
    {
        assert!(!src.contains("barrier("), "{} should not use barrier in per-item kernel", name);
    }
}

#[test]
fn kernel_sources_are_valid_c99_ish() {
    // Basic sanity: balanced braces
    for (name, src) in [
        ("matmul", kernels::MATMUL_I2S_SRC),
        ("quantize", kernels::QUANTIZE_I2S_SRC),
        ("elementwise", kernels::ELEMENTWISE_SRC),
        ("softmax", kernels::SOFTMAX_SRC),
    ] {
        let opens = src.matches('{').count();
        let closes = src.matches('}').count();
        assert_eq!(
            opens, closes,
            "{} has unbalanced braces: {} opens, {} closes",
            name, opens, closes
        );
    }
}

// === Elementwise Kernel Validation ===

#[test]
fn elementwise_src_contains_vec_add() {
    let src = kernels::ELEMENTWISE_SRC;
    assert!(src.contains("__kernel void vec_add"), "should contain vec_add kernel");
}

#[test]
fn elementwise_src_contains_silu() {
    let src = kernels::ELEMENTWISE_SRC;
    assert!(src.contains("__kernel void silu"), "should contain silu kernel");
}

#[test]
fn elementwise_src_contains_rms_norm() {
    let src = kernels::ELEMENTWISE_SRC;
    assert!(src.contains("__kernel void rms_norm"), "should contain rms_norm kernel");
}

#[test]
fn softmax_kernel_has_correct_signature() {
    let src = kernels::SOFTMAX_SRC;
    assert!(src.contains("__kernel void softmax"), "should contain softmax kernel");
}

#[test]
fn softmax_has_numerical_stability() {
    let src = kernels::SOFTMAX_SRC;
    // Softmax should subtract max for numerical stability
    assert!(
        src.contains("row_max") || src.contains("local_max"),
        "softmax should find max for numerical stability"
    );
}

// === Cross-kernel Consistency ===

#[test]
fn quantize_and_matmul_use_same_packing_layout() {
    // Both kernels must agree on how 2-bit values are packed into bytes.
    // The packing layout: value at position j occupies bits [j*2, j*2+1].
    let matmul_src = kernels::MATMUL_I2S_SRC;
    let quantize_src = kernels::QUANTIZE_I2S_SRC;

    // Both should shift by (sub/j * 2)
    assert!(
        matmul_src.contains("* 2)") && quantize_src.contains("* 2)"),
        "both kernels must use 2-bit stride for packing"
    );

    // Both should mask with 0x03 or shift-and-mask equivalently
    assert!(matmul_src.contains("0x03"), "matmul must mask to 2 bits");
}

#[test]
fn all_kernels_use_opencl_qualifiers() {
    // Verify we're writing OpenCL, not CUDA
    for (name, src) in [
        ("matmul", kernels::MATMUL_I2S_SRC),
        ("quantize", kernels::QUANTIZE_I2S_SRC),
        ("elementwise", kernels::ELEMENTWISE_SRC),
        ("softmax", kernels::SOFTMAX_SRC),
    ] {
        assert!(
            src.contains("__kernel") || src.contains("kernel"),
            "{} should use __kernel qualifier",
            name
        );
        assert!(!src.contains("__global__"), "{} should not use CUDA __global__ qualifier", name);
        assert!(
            !src.contains("threadIdx") && !src.contains("blockIdx"),
            "{} should not use CUDA thread indexing",
            name
        );
    }
}
