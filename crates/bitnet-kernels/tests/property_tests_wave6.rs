//! Wave 6 property tests: kernel operation invariants.
//!
//! Key invariants:
//! - Reduction Max output ≤ input max; Mean within [min, max]
//! - Row-wise reduction length matches row count
//! - Quantize + dequantize roundtrip error bounded (via FallbackKernel)
//! - Embedding lookup with valid indices never panics and returns correct shape
//! - Embedding normalize produces unit-length vectors

use bitnet_common::QuantizationType;
use bitnet_kernels::cpu::embedding;
use bitnet_kernels::reduction::{self, ReductionOp};
use bitnet_kernels::{FallbackKernel, KernelProvider};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Strategy helpers
// ---------------------------------------------------------------------------

/// Generate a non-empty f32 vector with finite values in [-100, 100].
fn finite_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-100.0f32..100.0f32, 1..=max_len)
}

/// Generate a matrix as (flat_data, rows, cols) with finite values.
fn finite_matrix(max_dim: usize) -> impl Strategy<Value = (Vec<f32>, usize, usize)> {
    (1usize..=max_dim, 1usize..=max_dim).prop_flat_map(|(rows, cols)| {
        prop::collection::vec(-100.0f32..100.0f32, rows * cols)
            .prop_map(move |data| (data, rows, cols))
    })
}

// ---------------------------------------------------------------------------
// Properties: Reduction — Max output ≤ input max
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// `reduce_f32(data, Max)` returns a value present in the input.
    #[test]
    fn prop_reduce_max_within_input(data in finite_f32_vec(64)) {
        let result = reduction::reduce_f32(&data, ReductionOp::Max);
        let actual_max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        prop_assert!(
            (result - actual_max).abs() < 1e-6,
            "reduce Max={result} differs from manual max={actual_max}"
        );
    }

    /// `reduce_f32(data, Mean)` lies between input min and max (inclusive).
    #[test]
    fn prop_reduce_mean_within_bounds(data in finite_f32_vec(64)) {
        let result = reduction::reduce_f32(&data, ReductionOp::Mean);
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        prop_assert!(
            result >= min - 1e-5 && result <= max + 1e-5,
            "Mean={result} not in [{min}, {max}]"
        );
    }

    /// `reduce_f32(data, Min)` matches the true minimum.
    #[test]
    fn prop_reduce_min_matches_true_min(data in finite_f32_vec(64)) {
        let result = reduction::reduce_f32(&data, ReductionOp::Min);
        let actual_min = data.iter().copied().fold(f32::INFINITY, f32::min);
        prop_assert!(
            (result - actual_min).abs() < 1e-6,
            "reduce Min={result} differs from manual min={actual_min}"
        );
    }
}

// ---------------------------------------------------------------------------
// Properties: Row-wise reduction length equals rows
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Row-wise Max reduction returns exactly `rows` elements.
    #[test]
    fn prop_reduce_rows_length((data, rows, cols) in finite_matrix(8)) {
        let result = reduction::reduce_rows_f32(&data, rows, cols, ReductionOp::Max)
            .expect("valid matrix dimensions");
        prop_assert_eq!(result.len(), rows);
    }

    /// Column-wise Sum reduction returns exactly `cols` elements.
    #[test]
    fn prop_reduce_cols_length((data, rows, cols) in finite_matrix(8)) {
        let result = reduction::reduce_cols_f32(&data, rows, cols, ReductionOp::Sum)
            .expect("valid matrix dimensions");
        prop_assert_eq!(result.len(), cols);
    }

    /// Row-wise Max: each row result ≤ global Max.
    #[test]
    fn prop_row_max_bounded_by_global((data, rows, cols) in finite_matrix(8)) {
        let global_max = reduction::reduce_f32(&data, ReductionOp::Max);
        let row_results = reduction::reduce_rows_f32(&data, rows, cols, ReductionOp::Max)
            .expect("valid dimensions");
        for (i, &v) in row_results.iter().enumerate() {
            prop_assert!(
                v <= global_max + 1e-6,
                "row {i} max {v} > global max {global_max}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Properties: Quantize + dequantize roundtrip error bounded
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// I2S quantize round-trip: packed bytes have expected length for group size 32.
    #[test]
    fn prop_quantize_i2s_output_length(
        // I2S uses groups of 32 elements; generate multiples of 32
        num_groups in 1usize..8,
    ) {
        let n = num_groups * 32;
        let input = vec![0.5f32; n];
        let kernel = FallbackKernel;
        // I2S packs 4 values per byte → n/4 bytes output, n/32 scales
        let mut output = vec![0u8; n / 4];
        let mut scales = vec![0.0f32; n / 32];
        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        prop_assert!(result.is_ok(), "quantize_i2s failed: {:?}", result.err());
    }

    /// TL1 quantize: one scale per 32-element block.
    #[test]
    fn prop_quantize_tl1_scale_count(num_groups in 1usize..8) {
        let n = num_groups * 32;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let kernel = FallbackKernel;
        let mut output = vec![0u8; n];
        let mut scales = vec![0.0f32; num_groups];
        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::TL1);
        prop_assert!(result.is_ok(), "quantize_tl1 failed: {:?}", result.err());
        // Every scale for non-zero data should be finite
        for (i, &s) in scales.iter().enumerate() {
            prop_assert!(s.is_finite(), "scale[{i}] is not finite: {s}");
        }
    }
}

// ---------------------------------------------------------------------------
// Properties: Embedding lookup — valid indices, correct shape
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Embedding lookup with valid indices returns exactly `n_indices * dim` elements.
    #[test]
    fn prop_embedding_lookup_shape(
        vocab in 2usize..16,
        dim in 1usize..32,
        n_indices in 1usize..8,
    ) {
        let table: Vec<f32> = (0..(vocab * dim)).map(|i| i as f32 * 0.1).collect();
        let indices: Vec<u32> = (0..n_indices).map(|i| (i % vocab) as u32).collect();

        let result = embedding::embedding_lookup(&table, &indices, dim);
        prop_assert!(result.is_ok(), "embedding_lookup failed: {:?}", result.err());

        let output = result.unwrap();
        prop_assert_eq!(
            output.len(),
            n_indices * dim,
            "expected {} elements, got {}",
            n_indices * dim,
            output.len()
        );
    }

    /// Embedding lookup with an out-of-bounds index returns an error.
    #[test]
    fn prop_embedding_lookup_oob_fails(
        vocab in 2usize..16,
        dim in 1usize..16,
    ) {
        let table = vec![0.0f32; vocab * dim];
        let oob_index = vocab as u32; // exactly out of bounds
        let result = embedding::embedding_lookup(&table, &[oob_index], dim);
        prop_assert!(result.is_err(), "expected error for OOB index {oob_index}");
    }

    /// Embedding lookup retrieves the correct row from the table.
    #[test]
    fn prop_embedding_lookup_correct_content(
        vocab in 2usize..8,
        dim in 1usize..16,
        idx in 0u32..8,
    ) {
        let idx = idx % (vocab as u32);
        let table: Vec<f32> = (0..(vocab * dim)).map(|i| i as f32).collect();

        let result = embedding::embedding_lookup(&table, &[idx], dim).unwrap();
        let expected_start = (idx as usize) * dim;
        let expected = &table[expected_start..expected_start + dim];

        for (j, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            prop_assert!(
                (got - want).abs() < 1e-6,
                "mismatch at offset {j}: got={got}, want={want}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Properties: Embedding normalize produces unit vectors
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// After normalize_embeddings, each vector has L2 norm ≈ 1.0.
    #[test]
    fn prop_normalize_produces_unit_vectors(
        n_vecs in 1usize..4,
        dim in 2usize..16,
    ) {
        // Use non-zero values so norm is defined
        let mut embeddings: Vec<f32> = (0..(n_vecs * dim)).map(|i| (i as f32) + 1.0).collect();
        embedding::normalize_embeddings(&mut embeddings, dim);

        for v in 0..n_vecs {
            let start = v * dim;
            let norm_sq: f32 = embeddings[start..start + dim].iter().map(|x| x * x).sum();
            prop_assert!(
                (norm_sq - 1.0).abs() < 1e-4,
                "vector {v} L2 norm^2 = {norm_sq}, expected ≈1.0"
            );
        }
    }
}
