//! High-level gather/scatter operations for embedding lookups and
//! index-based tensor access.
//!
//! These functions provide the building blocks for transformer embedding
//! layers: [`gather`] selects rows from a 2-D table (the embedding
//! lookup), [`scatter_add`] accumulates gradients back into positions,
//! and [`index_select`] picks elements along an arbitrary dimension.
//!
//! All operations work on `f32` data with mandatory bounds checking.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Helpers ────────────────────────────────────────────────────────

fn invalid_args(reason: impl Into<String>) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.into() })
}

// ── gather (embedding lookup) ──────────────────────────────────────

/// Select rows from a 2-D table by index — the core embedding lookup.
///
/// `table` is a flat row-major buffer of shape `[num_rows, row_len]`.
/// For each index in `indices` the corresponding row is copied into the
/// output.
///
/// Returns a `Vec` of length `indices.len() * row_len`.
///
/// # Errors
///
/// Returns an error if `num_rows` or `row_len` is zero, `table` is too
/// short, or any index is out of bounds.
pub fn gather(
    table: &[f32],
    num_rows: usize,
    row_len: usize,
    indices: &[usize],
) -> Result<Vec<f32>> {
    if num_rows == 0 || row_len == 0 {
        return Err(invalid_args("gather: num_rows and row_len must be > 0"));
    }
    if table.len() < num_rows * row_len {
        return Err(invalid_args(format!(
            "gather: table length {} < expected {}",
            table.len(),
            num_rows * row_len,
        )));
    }

    let mut out = Vec::with_capacity(indices.len() * row_len);
    for &idx in indices {
        if idx >= num_rows {
            return Err(invalid_args(format!(
                "gather: index {idx} out of bounds for {num_rows} rows"
            )));
        }
        let start = idx * row_len;
        out.extend_from_slice(&table[start..start + row_len]);
    }
    Ok(out)
}

// ── scatter_add (gradient accumulation) ────────────────────────────

/// Accumulate rows into a 2-D table by index.
///
/// For each `(index, row)` pair, `table[index] += row` element-wise.
/// This is the backward pass of an embedding lookup — it scatters
/// gradients back to the embedding table.
///
/// `table` is a flat row-major buffer of shape `[num_rows, row_len]`.
/// `values` is a flat row-major buffer of shape `[indices.len(), row_len]`.
///
/// # Errors
///
/// Returns an error when buffer sizes are inconsistent or any index is
/// out of bounds.
pub fn scatter_add(
    table: &mut [f32],
    num_rows: usize,
    row_len: usize,
    indices: &[usize],
    values: &[f32],
) -> Result<()> {
    if num_rows == 0 || row_len == 0 {
        return Err(invalid_args("scatter_add: num_rows and row_len must be > 0"));
    }
    if table.len() < num_rows * row_len {
        return Err(invalid_args(format!(
            "scatter_add: table length {} < expected {}",
            table.len(),
            num_rows * row_len,
        )));
    }
    if values.len() < indices.len() * row_len {
        return Err(invalid_args(format!(
            "scatter_add: values length {} < expected {}",
            values.len(),
            indices.len() * row_len,
        )));
    }

    for (i, &idx) in indices.iter().enumerate() {
        if idx >= num_rows {
            return Err(invalid_args(format!(
                "scatter_add: index {idx} out of bounds for {num_rows} rows"
            )));
        }
        let dst_start = idx * row_len;
        let src_start = i * row_len;
        for j in 0..row_len {
            table[dst_start + j] += values[src_start + j];
        }
    }
    Ok(())
}

// ── index_select ───────────────────────────────────────────────────

/// Select elements along a dimension by indices.
///
/// `data` is a flat row-major buffer representing a tensor whose
/// selected dimension has `dim_size` slices.  `outer` is the product of
/// all dimensions before the selected dimension, and `inner` is the
/// product of all dimensions after it.  The total length of `data` must
/// equal `outer * dim_size * inner`.
///
/// Returns a new `Vec` of length `outer * indices.len() * inner`.
///
/// # Errors
///
/// Returns an error when dimensions are zero, the buffer is the wrong
/// length, or any index is out of bounds.
pub fn index_select(
    data: &[f32],
    outer: usize,
    dim_size: usize,
    inner: usize,
    indices: &[usize],
) -> Result<Vec<f32>> {
    if dim_size == 0 {
        return Err(invalid_args("index_select: dim_size must be > 0"));
    }
    let expected = outer * dim_size * inner;
    if expected == 0 {
        return Err(invalid_args("index_select: outer and inner must be > 0"));
    }
    if data.len() < expected {
        return Err(invalid_args(format!(
            "index_select: data length {} < expected {expected}",
            data.len(),
        )));
    }

    let n_sel = indices.len();
    let mut out = Vec::with_capacity(outer * n_sel * inner);

    for o in 0..outer {
        for &idx in indices {
            if idx >= dim_size {
                return Err(invalid_args(format!(
                    "index_select: index {idx} out of bounds for dim_size {dim_size}"
                )));
            }
            let start = (o * dim_size + idx) * inner;
            out.extend_from_slice(&data[start..start + inner]);
        }
    }
    Ok(out)
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── gather ─────────────────────────────────────────────────────

    #[test]
    fn gather_embedding_lookup() {
        // 4 embeddings of dim 3
        let table = [
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
            10.0, 11.0, 12.0, // row 3
        ];
        let out = gather(&table, 4, 3, &[2, 0, 3]).unwrap();
        assert_eq!(out, vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn gather_single_row() {
        let table = [10.0, 20.0];
        let out = gather(&table, 1, 2, &[0]).unwrap();
        assert_eq!(out, vec![10.0, 20.0]);
    }

    #[test]
    fn gather_duplicate_indices() {
        let table = [1.0, 2.0, 3.0, 4.0]; // 2×2
        let out = gather(&table, 2, 2, &[1, 1, 0, 1]).unwrap();
        assert_eq!(out, vec![3.0, 4.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn gather_empty_indices() {
        let table = [1.0, 2.0, 3.0];
        let out = gather(&table, 1, 3, &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn gather_out_of_bounds() {
        let table = [1.0, 2.0, 3.0, 4.0]; // 2×2
        let err = gather(&table, 2, 2, &[0, 5]).unwrap_err();
        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn gather_zero_rows() {
        let err = gather(&[], 0, 3, &[]).unwrap_err();
        assert!(err.to_string().contains("must be > 0"));
    }

    #[test]
    fn gather_zero_row_len() {
        let err = gather(&[], 2, 0, &[]).unwrap_err();
        assert!(err.to_string().contains("must be > 0"));
    }

    #[test]
    fn gather_table_too_short() {
        let table = [1.0, 2.0]; // only 2 elements
        let err = gather(&table, 2, 2, &[0]).unwrap_err();
        assert!(err.to_string().contains("table length"));
    }

    // ── scatter_add ────────────────────────────────────────────────

    #[test]
    fn scatter_add_accumulates() {
        let mut table = [0.0f32; 6]; // 3×2
        let values = [1.0, 2.0, 3.0, 4.0]; // 2×2
        scatter_add(&mut table, 3, 2, &[1, 1], &values).unwrap();
        // Both rows accumulated into row 1: [1+3, 2+4] = [4, 6]
        assert_eq!(table, [0.0, 0.0, 4.0, 6.0, 0.0, 0.0]);
    }

    #[test]
    fn scatter_add_disjoint() {
        let mut table = [0.0f32; 6]; // 3×2
        let values = [10.0, 20.0, 30.0, 40.0];
        scatter_add(&mut table, 3, 2, &[0, 2], &values).unwrap();
        assert_eq!(table, [10.0, 20.0, 0.0, 0.0, 30.0, 40.0]);
    }

    #[test]
    fn scatter_add_empty_indices() {
        let mut table = [1.0, 2.0];
        scatter_add(&mut table, 1, 2, &[], &[]).unwrap();
        assert_eq!(table, [1.0, 2.0]); // unchanged
    }

    #[test]
    fn scatter_add_out_of_bounds() {
        let mut table = [0.0f32; 4]; // 2×2
        let values = [1.0, 2.0];
        let err = scatter_add(&mut table, 2, 2, &[5], &values).unwrap_err();
        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn scatter_add_values_too_short() {
        let mut table = [0.0f32; 4];
        let err = scatter_add(&mut table, 2, 2, &[0, 1], &[1.0, 2.0]).unwrap_err();
        assert!(err.to_string().contains("values length"));
    }

    // ── index_select ───────────────────────────────────────────────

    #[test]
    fn index_select_along_first_dim() {
        // shape [3, 2]: select along dim 0 (outer=1, dim=3, inner=2)
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = index_select(&data, 1, 3, 2, &[2, 0]).unwrap();
        assert_eq!(out, vec![4.0, 5.0, 0.0, 1.0]);
    }

    #[test]
    fn index_select_along_middle_dim() {
        // shape [2, 3, 1]: select along dim 1 (outer=2, dim=3, inner=1)
        let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let out = index_select(&data, 2, 3, 1, &[1, 2]).unwrap();
        // outer 0: [20, 30], outer 1: [50, 60]
        assert_eq!(out, vec![20.0, 30.0, 50.0, 60.0]);
    }

    #[test]
    fn index_select_empty_indices() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let out = index_select(&data, 1, 4, 1, &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn index_select_out_of_bounds() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let err = index_select(&data, 1, 2, 2, &[5]).unwrap_err();
        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn index_select_zero_dim() {
        let err = index_select(&[1.0], 1, 0, 1, &[]).unwrap_err();
        assert!(err.to_string().contains("dim_size must be > 0"));
    }

    #[test]
    fn index_select_data_too_short() {
        let err = index_select(&[1.0], 2, 2, 2, &[0]).unwrap_err();
        assert!(err.to_string().contains("data length"));
    }

    // ── roundtrip: gather then scatter_add ─────────────────────────

    #[test]
    fn gather_scatter_add_roundtrip_disjoint() {
        // Gather rows, scatter them back — should recover original.
        let table = [
            1.0, 2.0, // row 0
            3.0, 4.0, // row 1
            5.0, 6.0, // row 2
        ];
        let indices = [2, 0, 1];
        let gathered = gather(&table, 3, 2, &indices).unwrap();

        let mut reconstructed = [0.0f32; 6];
        scatter_add(&mut reconstructed, 3, 2, &indices, &gathered).unwrap();
        assert_eq!(reconstructed, table);
    }

    // ── property tests ─────────────────────────────────────────────

    mod prop {
        use super::*;
        use proptest::prelude::*;

        /// Strategy for a gather scenario with valid parameters.
        fn gather_scenario() -> impl Strategy<Value = (Vec<f32>, usize, usize, Vec<usize>)> {
            (1..=16usize, 1..=16usize).prop_flat_map(|(num_rows, row_len)| {
                let table_len = num_rows * row_len;
                let table = proptest::collection::vec(-100.0f32..100.0, table_len);
                let indices = proptest::collection::vec(0..num_rows, 0..=20);
                (table, Just(num_rows), Just(row_len), indices)
            })
        }

        proptest! {
            #[test]
            fn gather_output_length(
                (table, num_rows, row_len, indices) in gather_scenario()
            ) {
                let out = gather(&table, num_rows, row_len, &indices).unwrap();
                prop_assert_eq!(out.len(), indices.len() * row_len);
            }

            #[test]
            fn gather_values_match_rows(
                (table, num_rows, row_len, indices) in gather_scenario()
            ) {
                let out = gather(&table, num_rows, row_len, &indices).unwrap();
                for (i, &idx) in indices.iter().enumerate() {
                    let expected = &table[idx * row_len..(idx + 1) * row_len];
                    let actual = &out[i * row_len..(i + 1) * row_len];
                    prop_assert_eq!(actual, expected);
                }
            }

            #[test]
            fn gather_then_scatter_roundtrip(
                (table, num_rows, row_len, indices) in gather_scenario()
                    .prop_filter("need unique indices for clean roundtrip",
                        |(_, num_rows, _, indices)| {
                            let mut sorted = indices.clone();
                            sorted.sort();
                            sorted.dedup();
                            sorted.len() == indices.len() && indices.len() <= *num_rows
                        })
            ) {
                let gathered = gather(&table, num_rows, row_len, &indices).unwrap();
                let mut reconstructed = vec![0.0f32; num_rows * row_len];
                scatter_add(&mut reconstructed, num_rows, row_len, &indices, &gathered).unwrap();
                // Only gathered rows should match — others stay zero.
                for &idx in &indices {
                    let expected = &table[idx * row_len..(idx + 1) * row_len];
                    let actual = &reconstructed[idx * row_len..(idx + 1) * row_len];
                    prop_assert_eq!(actual, expected);
                }
            }

            #[test]
            fn gather_oob_always_fails(
                num_rows in 1..=16usize,
                row_len in 1..=8usize,
                oob_idx in 16..=100usize,
            ) {
                let table = vec![0.0f32; num_rows * row_len];
                if oob_idx >= num_rows {
                    prop_assert!(gather(&table, num_rows, row_len, &[oob_idx]).is_err());
                }
            }

            #[test]
            fn index_select_output_length(
                dim_size in 1..=8usize,
                inner in 1..=8usize,
                indices in proptest::collection::vec(0..8usize, 0..=10),
            ) {
                let outer = 1usize;
                let data = vec![0.0f32; outer * dim_size * inner];
                let valid: Vec<usize> = indices.into_iter().filter(|&i| i < dim_size).collect();
                let out = index_select(&data, outer, dim_size, inner, &valid).unwrap();
                prop_assert_eq!(out.len(), outer * valid.len() * inner);
            }
        }
    }
}
