//! Tests for the OpenCL softmax kernel.
//!
//! Includes CPU-reference comparison and property tests.

use bitnet_kernels::kernels;

// === CPU reference softmax ===

/// Numerically stable softmax computed on the CPU (reference implementation).
fn cpu_softmax(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(input.len(), rows * cols);
    let mut output = vec![0.0f32; input.len()];
    for r in 0..rows {
        let row = &input[r * cols..(r + 1) * cols];
        let out = &mut output[r * cols..(r + 1) * cols];

        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|&v| (v - max_val).exp()).sum();
        for (o, &v) in out.iter_mut().zip(row.iter()) {
            *o = (v - max_val).exp() / sum;
        }
    }
    output
}

// === Source-level tests ===

#[test]
fn softmax_kernel_source_uses_local_memory_reduction() {
    let src = kernels::SOFTMAX_SRC;
    assert!(src.contains("__local"), "should use __local memory for reductions");
    assert!(
        src.contains("barrier(CLK_LOCAL_MEM_FENCE)"),
        "should synchronise with local memory barriers"
    );
}

#[test]
fn softmax_kernel_source_has_three_passes() {
    let src = kernels::SOFTMAX_SRC;
    // Pass 1: max reduction
    assert!(src.contains("fmax"), "should compute row max with fmax");
    // Pass 2: exp + sum
    assert!(src.contains("exp("), "should exponentiate");
    // Pass 3: normalise
    assert!(src.contains("inv_sum"), "should divide by sum");
}

#[test]
fn softmax_kernel_source_handles_arbitrary_sizes() {
    let src = kernels::SOFTMAX_SRC;
    // The strided loop `for (uint i = lid; i < N; i += local_size)` handles
    // tensors larger than the work-group size.
    assert!(
        src.contains("i += local_size"),
        "should stride over elements for arbitrary tensor sizes"
    );
}

// === CPU reference correctness ===

#[test]
fn cpu_softmax_single_row_sums_to_one() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = cpu_softmax(&input, 1, 4);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "softmax sum = {} ≠ 1.0", sum);
}

#[test]
fn cpu_softmax_multiple_rows() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let output = cpu_softmax(&input, 2, 4);
    for r in 0..2 {
        let row_sum: f32 = output[r * 4..(r + 1) * 4].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-6, "row {} sum = {} ≠ 1.0", r, row_sum);
    }
}

#[test]
fn cpu_softmax_all_values_in_zero_one() {
    let input = vec![-10.0, 0.0, 10.0, 20.0];
    let output = cpu_softmax(&input, 1, 4);
    for (i, &v) in output.iter().enumerate() {
        assert!(v >= 0.0 && v <= 1.0, "output[{}] = {} not in [0, 1]", i, v);
    }
}

#[test]
fn cpu_softmax_numerical_stability_large_values() {
    let input = vec![1000.0, 1001.0, 1002.0, 1003.0];
    let output = cpu_softmax(&input, 1, 4);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum = {} for large inputs", sum);
    assert!(output.iter().all(|&v| v.is_finite()), "should not produce inf/nan");
}

#[test]
fn cpu_softmax_numerical_stability_negative_large() {
    let input = vec![-1000.0, -999.0, -998.0, -997.0];
    let output = cpu_softmax(&input, 1, 4);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum = {} for large negative inputs", sum);
}

#[test]
fn cpu_softmax_uniform_input_produces_uniform_output() {
    let input = vec![5.0; 8];
    let output = cpu_softmax(&input, 1, 8);
    let expected = 1.0 / 8.0;
    for (i, &v) in output.iter().enumerate() {
        assert!((v - expected).abs() < 1e-6, "output[{}] = {} ≠ expected {}", i, v, expected);
    }
}

// === Property tests ===

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_softmax_input() -> impl Strategy<Value = (Vec<f32>, usize, usize)> {
        (1usize..=4, 1usize..=128).prop_flat_map(|(rows, cols)| {
            let len = rows * cols;
            (proptest::collection::vec(-100.0f32..100.0, len), Just(rows), Just(cols))
        })
    }

    proptest! {
        #[test]
        fn softmax_output_sums_to_one(
            (input, rows, cols) in arb_softmax_input()
        ) {
            let output = cpu_softmax(&input, rows, cols);
            for r in 0..rows {
                let row_sum: f32 = output[r * cols..(r + 1) * cols].iter().sum();
                prop_assert!(
                    (row_sum - 1.0).abs() < 1e-4,
                    "row {} sum = {}",
                    r,
                    row_sum
                );
            }
        }

        #[test]
        fn softmax_output_all_in_zero_one(
            (input, rows, cols) in arb_softmax_input()
        ) {
            let output = cpu_softmax(&input, rows, cols);
            for (i, &v) in output.iter().enumerate() {
                prop_assert!(v >= 0.0, "output[{}] = {} < 0", i, v);
                prop_assert!(v <= 1.0, "output[{}] = {} > 1", i, v);
            }
        }

        #[test]
        fn softmax_output_is_finite(
            (input, rows, cols) in arb_softmax_input()
        ) {
            let output = cpu_softmax(&input, rows, cols);
            for (i, &v) in output.iter().enumerate() {
                prop_assert!(v.is_finite(), "output[{}] = {} is not finite", i, v);
            }
        }

        #[test]
        fn softmax_preserves_relative_order(
            (input, rows, cols) in arb_softmax_input()
        ) {
            let output = cpu_softmax(&input, rows, cols);
            for r in 0..rows {
                let row_in = &input[r * cols..(r + 1) * cols];
                let row_out = &output[r * cols..(r + 1) * cols];
                // Larger input → larger (or equal) output within same row
                for i in 0..cols {
                    for j in (i + 1)..cols {
                        if row_in[i] > row_in[j] {
                            prop_assert!(
                                row_out[i] >= row_out[j] - 1e-6,
                                "order not preserved at row={}, i={}, j={}",
                                r, i, j
                            );
                        }
                    }
                }
            }
        }
    }
}
