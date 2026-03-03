//! BDD Wave 18 — Memory allocation, alignment, and cache behaviour tests.
//!
//! Verifies that kernel operations handle various allocation sizes,
//! alignment boundaries, and cache-line patterns correctly.

use bitnet_kernels::cpu::activations::{gelu_vec, relu_inplace, silu_vec};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm};
use bitnet_kernels::reduction::{ReductionOp, reduce_f32, reduce_rows_f32};

const TOL: f32 = 1e-5;

// ── Allocation size edge cases ─────────────────────────────────────

#[test]
fn given_single_element_tensor_when_reduction_applied_then_correct_result() {
    let input = vec![42.0];
    let result = reduce_f32(&input, ReductionOp::Sum);
    assert!((result - 42.0).abs() < TOL);
}

#[test]
fn given_power_of_two_sized_vector_when_gelu_applied_then_correct_length() {
    for &size in &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let input = vec![1.0; size];
        let output = gelu_vec(&input);
        assert_eq!(output.len(), size, "GELU output length mismatch for size {size}");
    }
}

#[test]
fn given_non_power_of_two_sized_vector_when_silu_applied_then_correct_length() {
    for &size in &[3, 5, 7, 13, 17, 31, 63, 127, 255, 1023] {
        let input = vec![0.5; size];
        let output = silu_vec(&input);
        assert_eq!(output.len(), size, "SiLU output length mismatch for size {size}");
    }
}

#[test]
fn given_cache_line_boundary_size_when_reduction_applied_then_correct() {
    // 64 bytes / 4 bytes per f32 = 16 floats per cache line
    let size = 16;
    let input: Vec<f32> = (1..=size as u32).map(|i| i as f32).collect();
    let result = reduce_f32(&input, ReductionOp::Sum);
    let expected = (size * (size + 1) / 2) as f32;
    assert!(
        (result - expected).abs() < TOL,
        "cache-line-aligned sum: expected {expected}, got {result}"
    );
}

#[test]
fn given_cache_line_plus_one_size_when_reduction_applied_then_handles_remainder() {
    let size = 17; // one past cache line boundary
    let input: Vec<f32> = (1..=size as u32).map(|i| i as f32).collect();
    let result = reduce_f32(&input, ReductionOp::Sum);
    let expected = (size * (size + 1) / 2) as f32;
    assert!((result - expected).abs() < TOL, "cache-line+1 sum: expected {expected}, got {result}");
}

// ── Large allocation stress ────────────────────────────────────────

#[test]
fn given_large_allocation_when_layer_norm_applied_then_no_oom_or_panic() {
    let size = 4096;
    let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
    let gamma = vec![1.0; size];
    let config = LayerNormConfig::new(vec![size]);

    let output = layer_norm(&input, &gamma, None, &config).unwrap();
    assert_eq!(output.len(), size);
    for &v in &output {
        assert!(v.is_finite(), "output should be finite");
    }
}

#[test]
fn given_multi_batch_large_allocation_when_layer_norm_applied_then_correct_size() {
    let norm_size = 512;
    let batch_size = 32;
    let total = norm_size * batch_size;
    let input: Vec<f32> = (0..total).map(|i| (i % 100) as f32).collect();
    let gamma = vec![1.0; norm_size];
    let config = LayerNormConfig::new(vec![norm_size]);

    let output = layer_norm(&input, &gamma, None, &config).unwrap();
    assert_eq!(output.len(), total);
}

// ── Alignment patterns ─────────────────────────────────────────────

#[test]
fn given_aligned_matrix_when_row_reduced_then_each_row_correct() {
    // 4 rows × 16 cols (64-byte aligned rows)
    let rows = 4;
    let cols = 16;
    let mut matrix = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            matrix[r * cols + c] = (r + 1) as f32;
        }
    }

    let result = reduce_rows_f32(&matrix, rows, cols, ReductionOp::Sum).unwrap();
    for (r, &v) in result.iter().enumerate() {
        let expected = (r + 1) as f32 * cols as f32;
        assert!((v - expected).abs() < TOL, "row {r}: expected {expected}, got {v}");
    }
}

#[test]
fn given_unaligned_matrix_when_row_reduced_then_still_correct() {
    // 3 rows × 5 cols (not cache-aligned)
    let rows = 3;
    let cols = 5;
    let matrix: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();

    let result = reduce_rows_f32(&matrix, rows, cols, ReductionOp::Mean).unwrap();
    // row 0: mean(0,1,2,3,4) = 2
    assert!((result[0] - 2.0).abs() < TOL);
    // row 1: mean(5,6,7,8,9) = 7
    assert!((result[1] - 7.0).abs() < TOL);
    // row 2: mean(10,11,12,13,14) = 12
    assert!((result[2] - 12.0).abs() < TOL);
}

// ── In-place mutation patterns ─────────────────────────────────────

#[test]
fn given_pre_allocated_buffer_when_relu_inplace_then_no_extra_allocation() {
    let mut data = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
    let ptr_before = data.as_ptr();
    relu_inplace(&mut data);
    let ptr_after = data.as_ptr();
    assert_eq!(ptr_before, ptr_after, "in-place operation should not reallocate");
    assert_eq!(data, vec![0.0, 0.0, 0.0, 1.0, 3.0]);
}

#[test]
fn given_large_buffer_when_relu_inplace_then_all_negatives_zeroed() {
    let n = 10_000;
    let mut data: Vec<f32> =
        (0..n).map(|i| if i % 2 == 0 { -(i as f32) } else { i as f32 }).collect();
    relu_inplace(&mut data);
    for (i, &v) in data.iter().enumerate() {
        assert!(v >= 0.0, "element {i} should be non-negative, got {v}");
    }
}

// ── Repeated operations (cache warming) ────────────────────────────

#[test]
fn given_same_input_when_reduction_repeated_then_results_are_deterministic() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1).collect();
    let first = reduce_f32(&input, ReductionOp::Sum);
    for _ in 0..10 {
        let result = reduce_f32(&input, ReductionOp::Sum);
        assert!(
            (result - first).abs() < TOL,
            "repeated reduction should be deterministic: {result} vs {first}"
        );
    }
}

#[test]
fn given_same_input_when_gelu_repeated_then_results_are_deterministic() {
    let input: Vec<f32> = (0..128).map(|i| (i as f32) * 0.05 - 3.0).collect();
    let first = gelu_vec(&input);
    for _ in 0..10 {
        let result = gelu_vec(&input);
        for (i, (&a, &b)) in first.iter().zip(result.iter()).enumerate() {
            assert!(
                (a - b).abs() < TOL,
                "repeated GELU should be deterministic at {i}: {a} vs {b}"
            );
        }
    }
}

// ── Stride and contiguity patterns ─────────────────────────────────

#[test]
fn given_row_major_matrix_when_row_and_col_reduced_then_consistent() {
    // 3×4 matrix
    let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let row_sums = reduce_rows_f32(&matrix, 3, 4, ReductionOp::Sum).unwrap();
    let col_sums =
        bitnet_kernels::reduction::reduce_cols_f32(&matrix, 3, 4, ReductionOp::Sum).unwrap();

    // Total from rows should equal total from cols
    let total_from_rows: f32 = row_sums.iter().sum();
    let total_from_cols: f32 = col_sums.iter().sum();
    assert!(
        (total_from_rows - total_from_cols).abs() < TOL,
        "row-sum total {total_from_rows} should equal col-sum total {total_from_cols}"
    );
    // Both should equal global sum
    let global = reduce_f32(&matrix, ReductionOp::Sum);
    assert!((total_from_rows - global).abs() < TOL);
}

#[test]
fn given_wide_matrix_when_col_reduced_then_handles_many_columns() {
    let rows = 4;
    let cols = 1024;
    let matrix: Vec<f32> = vec![1.0; rows * cols];
    let result =
        bitnet_kernels::reduction::reduce_cols_f32(&matrix, rows, cols, ReductionOp::Sum).unwrap();
    assert_eq!(result.len(), cols);
    for &v in &result {
        assert!((v - rows as f32).abs() < TOL);
    }
}
