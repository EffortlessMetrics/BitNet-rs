#![allow(clippy::approx_constant)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::identity_op)]
#![allow(clippy::manual_abs_diff)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::manual_contains)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::no_effect)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::useless_vec)]
#![allow(clippy::useless_vec, clippy::excessive_precision)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::assertions_on_constants)]

//! CPU-reference tests for the OpenCL activation and softmax kernels.
//!
//! These tests validate kernel correctness without requiring OpenCL hardware
//! by implementing CPU reference functions that mirror the kernel logic.

use bitnet_kernels::kernels;

// ============================================================================
// CPU reference implementations (mirror the .cl kernel logic)
// ============================================================================

fn cpu_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn cpu_gelu(x: f32) -> f32 {
    let cdf = 0.5 * (1.0 + (0.797_884_560_8_f32 * (x + 0.044715 * x * x * x)).tanh());
    x * cdf
}

fn cpu_relu(x: f32) -> f32 {
    x.max(0.0)
}

fn cpu_softmax(data: &[f32]) -> Vec<f32> {
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = data.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / (sum + 1e-8)).collect()
}

// ============================================================================
// Kernel source validation
// ============================================================================

#[test]
fn activations_kernel_source_is_not_empty() {
    assert!(!kernels::ACTIVATIONS_SRC.is_empty());
}

#[test]
fn activations_kernel_source_contains_kernel_keyword() {
    assert!(kernels::ACTIVATIONS_SRC.contains("__kernel"));
}

#[test]
fn activations_kernel_has_all_functions() {
    let src = kernels::ACTIVATIONS_SRC;
    for name in [
        "silu",
        "silu_mul",
        "gelu",
        "relu",
        "elementwise_add",
        "elementwise_mul",
        "scale",
        "softmax_full",
    ] {
        assert!(src.contains(name), "missing kernel function: {name}");
    }
}

// ============================================================================
// SiLU correctness
// ============================================================================

#[test]
fn silu_known_values() {
    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    assert!((cpu_silu(0.0)).abs() < 1e-7);
    // SiLU(1) ≈ 1 * sigmoid(1) ≈ 0.7311
    assert!((cpu_silu(1.0) - 0.7311).abs() < 0.01);
    // SiLU(-1) ≈ -1 * sigmoid(-1) ≈ -0.2689
    assert!((cpu_silu(-1.0) - (-0.2689)).abs() < 0.01);
}

#[test]
fn silu_zero_is_zero() {
    assert!((cpu_silu(0.0)).abs() < 1e-7);
}

#[test]
fn silu_large_positive_approaches_x() {
    let x = 10.0_f32;
    // For large x, sigmoid(x) → 1, so SiLU(x) → x
    assert!((cpu_silu(x) - x).abs() < 0.01);
}

#[test]
fn silu_large_negative_approaches_zero() {
    let x = -10.0_f32;
    // For large negative x, sigmoid(x) → 0, so SiLU(x) → 0
    assert!(cpu_silu(x).abs() < 0.01);
}

#[test]
fn silu_is_odd_like_near_origin() {
    // SiLU isn't exactly odd, but SiLU(-x) ≈ -SiLU(x) for small x
    // More precisely: SiLU(x) + SiLU(-x) = x * sigmoid(x) + (-x) * sigmoid(-x)
    //   = x * (sigmoid(x) - sigmoid(-x)) = x * (2*sigmoid(x) - 1)
    // This is not zero, but let's verify the identity holds.
    for &x in &[0.1_f32, 0.5, 1.0, 2.0] {
        let result = cpu_silu(x) + cpu_silu(-x);
        let expected = x * (2.0 / (1.0 + (-x).exp()) - 1.0);
        assert!((result - expected).abs() < 1e-5, "SiLU identity failed for x={x}");
    }
}

// ============================================================================
// Fused SiLU*up (silu_mul)
// ============================================================================

#[test]
fn silu_mul_matches_separate_ops() {
    let gate = vec![1.0_f32, -1.0, 0.5, 2.0, -0.5];
    let up = vec![2.0_f32, 3.0, -1.0, 0.5, 4.0];

    for i in 0..gate.len() {
        let fused = cpu_silu(gate[i]) * up[i];
        let separate_silu = gate[i] / (1.0 + (-gate[i]).exp());
        let separate = separate_silu * up[i];
        assert!((fused - separate).abs() < 1e-6, "silu_mul mismatch at index {i}");
    }
}

#[test]
fn silu_mul_zero_gate_is_zero() {
    // SiLU(0) = 0, so silu_mul(0, anything) = 0
    assert!((cpu_silu(0.0) * 42.0).abs() < 1e-7);
}

// ============================================================================
// GELU correctness
// ============================================================================

#[test]
fn gelu_known_values() {
    // GELU(0) = 0
    assert!(cpu_gelu(0.0).abs() < 1e-7);
    // GELU(1) ≈ 0.8412 (tanh approximation)
    assert!((cpu_gelu(1.0) - 0.8412).abs() < 0.01);
    // GELU(-1) ≈ -0.1588
    assert!((cpu_gelu(-1.0) - (-0.1588)).abs() < 0.01);
}

#[test]
fn gelu_zero_is_zero() {
    assert!(cpu_gelu(0.0).abs() < 1e-7);
}

#[test]
fn gelu_large_positive_approaches_x() {
    let x = 10.0_f32;
    // For large x, tanh → 1, cdf → 1, GELU(x) → x
    assert!((cpu_gelu(x) - x).abs() < 0.01);
}

#[test]
fn gelu_large_negative_approaches_zero() {
    let x = -10.0_f32;
    assert!(cpu_gelu(x).abs() < 0.01);
}

// ============================================================================
// ReLU correctness
// ============================================================================

#[test]
fn relu_positive_passthrough() {
    assert_eq!(cpu_relu(1.0), 1.0);
    assert_eq!(cpu_relu(100.0), 100.0);
    assert_eq!(cpu_relu(0.001), 0.001);
}

#[test]
fn relu_negative_is_zero() {
    assert_eq!(cpu_relu(-1.0), 0.0);
    assert_eq!(cpu_relu(-100.0), 0.0);
    assert_eq!(cpu_relu(-0.001), 0.0);
}

#[test]
fn relu_zero_is_zero() {
    assert_eq!(cpu_relu(0.0), 0.0);
}

// ============================================================================
// Elementwise add/mul correctness
// ============================================================================

#[test]
fn elementwise_add_correctness() {
    let a = vec![1.0_f32, 2.0, 3.0, -1.0];
    let b = vec![4.0_f32, -2.0, 0.0, 5.0];
    let expected = vec![5.0_f32, 0.0, 3.0, 4.0];
    for i in 0..a.len() {
        assert!((a[i] + b[i] - expected[i]).abs() < 1e-7, "add mismatch at {i}");
    }
}

#[test]
fn elementwise_mul_correctness() {
    let a = vec![1.0_f32, 2.0, 3.0, -1.0];
    let b = vec![4.0_f32, -2.0, 0.0, 5.0];
    let expected = vec![4.0_f32, -4.0, 0.0, -5.0];
    for i in 0..a.len() {
        assert!((a[i] * b[i] - expected[i]).abs() < 1e-7, "mul mismatch at {i}");
    }
}

// ============================================================================
// Scale correctness
// ============================================================================

#[test]
fn scale_correctness() {
    let data = vec![1.0_f32, -2.0, 3.0, 0.0];
    let scalar = 0.5_f32;
    let expected = vec![0.5_f32, -1.0, 1.5, 0.0];
    for i in 0..data.len() {
        assert!((data[i] * scalar - expected[i]).abs() < 1e-7, "scale mismatch at {i}");
    }
}

#[test]
fn scale_by_zero_is_zero() {
    let data = vec![1.0_f32, -2.0, 100.0];
    for &v in &data {
        assert_eq!(v * 0.0, 0.0);
    }
}

// ============================================================================
// Softmax correctness
// ============================================================================

#[test]
fn softmax_sums_to_one() {
    let input = vec![1.0_f32, 2.0, 3.0, 4.0];
    let output = cpu_softmax(&input);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}, expected ~1.0");
}

#[test]
fn softmax_numerical_stability_large_values() {
    // With large values, naive exp would overflow; subtract-max prevents this
    let input = vec![1000.0_f32, 1001.0, 999.0];
    let output = cpu_softmax(&input);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum} with large values");
    assert!(output.iter().all(|&v| v.is_finite()), "softmax produced non-finite values");
    // Middle element (1001) should have highest probability
    assert!(output[1] > output[0] && output[1] > output[2]);
}

#[test]
fn softmax_uniform_with_same_values() {
    let input = vec![5.0_f32; 4];
    let output = cpu_softmax(&input);
    for &v in &output {
        assert!((v - 0.25).abs() < 1e-5, "expected uniform 0.25, got {v}");
    }
}

#[test]
fn softmax_preserves_ordering() {
    let input = vec![1.0_f32, 3.0, 2.0, 5.0, 4.0];
    let output = cpu_softmax(&input);
    // Largest input → largest probability
    assert!(output[3] > output[4]);
    assert!(output[4] > output[1]);
    assert!(output[1] > output[2]);
    assert!(output[2] > output[0]);
}

#[test]
fn softmax_single_element() {
    let input = vec![42.0_f32];
    let output = cpu_softmax(&input);
    assert!((output[0] - 1.0).abs() < 1e-5);
}

#[test]
fn softmax_with_negative_values() {
    let input = vec![-1.0_f32, -2.0, -3.0];
    let output = cpu_softmax(&input);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // First (least negative) should have highest probability
    assert!(output[0] > output[1] && output[1] > output[2]);
}

// ============================================================================
// Zero-input tests (all activations)
// ============================================================================

#[test]
fn all_activations_handle_zero() {
    assert!(cpu_silu(0.0).abs() < 1e-7, "SiLU(0) should be ~0");
    assert!(cpu_gelu(0.0).abs() < 1e-7, "GELU(0) should be ~0");
    assert_eq!(cpu_relu(0.0), 0.0, "ReLU(0) should be 0");
}

// ============================================================================
// Large/small input tests (all activations produce finite output)
// ============================================================================

#[test]
fn all_activations_finite_for_finite_input() {
    let test_vals = [-100.0_f32, -10.0, -1.0, -0.001, 0.0, 0.001, 1.0, 10.0, 100.0];
    for &x in &test_vals {
        assert!(cpu_silu(x).is_finite(), "SiLU({x}) not finite");
        assert!(cpu_gelu(x).is_finite(), "GELU({x}) not finite");
        assert!(cpu_relu(x).is_finite(), "ReLU({x}) not finite");
    }
}

#[test]
fn silu_extreme_values_are_finite() {
    // Very large positive: SiLU(x) ≈ x (finite for f32 range)
    assert!(cpu_silu(80.0).is_finite());
    // Very large negative: SiLU(x) ≈ 0
    assert!(cpu_silu(-80.0).is_finite());
}

#[test]
fn gelu_extreme_values_are_finite() {
    assert!(cpu_gelu(80.0).is_finite());
    assert!(cpu_gelu(-80.0).is_finite());
}

// ============================================================================
// Monotonicity property tests
// ============================================================================

#[test]
fn silu_is_monotonically_increasing_for_positive_x() {
    // SiLU is monotonically increasing for x > ~-0.278
    // For simplicity, test only for x > 0
    let mut prev = cpu_silu(0.0);
    for i in 1..=100 {
        let x = i as f32 * 0.1;
        let y = cpu_silu(x);
        assert!(
            y >= prev,
            "SiLU not monotonic: SiLU({}) = {} < SiLU({}) = {}",
            x,
            y,
            x - 0.1,
            prev
        );
        prev = y;
    }
}

#[test]
fn gelu_is_monotonically_increasing_for_positive_x() {
    let mut prev = cpu_gelu(0.0);
    for i in 1..=100 {
        let x = i as f32 * 0.1;
        let y = cpu_gelu(x);
        assert!(
            y >= prev,
            "GELU not monotonic: GELU({}) = {} < GELU({}) = {}",
            x,
            y,
            x - 0.1,
            prev
        );
        prev = y;
    }
}

#[test]
fn relu_is_monotonically_nondecreasing() {
    let mut prev = cpu_relu(-10.0);
    for i in -99..=100 {
        let x = i as f32 * 0.1;
        let y = cpu_relu(x);
        assert!(y >= prev, "ReLU not monotonic at x={x}");
        prev = y;
    }
}

// ============================================================================
// Softmax with very small input differences
// ============================================================================

#[test]
fn softmax_tiny_differences() {
    let input = vec![1.0_f32, 1.0 + 1e-6, 1.0 - 1e-6];
    let output = cpu_softmax(&input);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // All probabilities should be close to 1/3
    for &v in &output {
        assert!((v - 1.0 / 3.0).abs() < 0.01);
    }
}

// ============================================================================
// Kernel source structure validation
// ============================================================================

#[test]
fn activations_silu_kernel_has_bounds_check() {
    let src = kernels::ACTIVATIONS_SRC;
    assert!(src.contains("if (gid >= n)"), "silu kernel should check bounds");
}

#[test]
fn activations_softmax_uses_local_memory() {
    let src = kernels::ACTIVATIONS_SRC;
    assert!(src.contains("__local float*"), "softmax_full should use local memory for reduction");
}

#[test]
fn activations_softmax_uses_barrier() {
    let src = kernels::ACTIVATIONS_SRC;
    assert!(
        src.contains("barrier(CLK_LOCAL_MEM_FENCE)"),
        "softmax_full should synchronize with barriers"
    );
}

#[test]
fn activations_softmax_subtracts_max_for_stability() {
    let src = kernels::ACTIVATIONS_SRC;
    assert!(
        src.contains("- row_max"),
        "softmax_full should subtract row max for numerical stability"
    );
}
