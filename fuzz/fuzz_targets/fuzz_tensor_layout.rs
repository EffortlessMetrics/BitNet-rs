#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::tensor_layout::{LayoutOrder, TensorLayout, broadcastable, compute_strides};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct LayoutInput {
    /// Shape dimensions (capped to reasonable sizes).
    dims: Vec<u16>,
    /// Element size selector.
    element_size: u8,
    /// _Indices placeholder (used via valid_indices below)._
    _indices: Vec<u16>,
    /// Second shape for broadcast checking.
    dims_b: Vec<u16>,
    /// Dimensions to transpose.
    transpose_dim0: u8,
    transpose_dim1: u8,
    /// New shape for reshape.
    reshape_dims: Vec<u16>,
    /// Alignment value.
    alignment: u16,
    /// Whether to use column-major order.
    col_major: bool,
}

fuzz_target!(|input: LayoutInput| {
    // Cap dimensions to avoid huge allocations
    let shape: Vec<usize> = input.dims.iter().take(6).map(|&d| (d as usize % 64) + 1).collect();

    if shape.is_empty() {
        return;
    }

    let element_size = (input.element_size as usize % 8) + 1;

    // Invariant 1: compute_strides must never panic
    let row_strides = compute_strides(&shape, LayoutOrder::RowMajor);
    let col_strides = compute_strides(&shape, LayoutOrder::ColMajor);
    assert_eq!(row_strides.len(), shape.len());
    assert_eq!(col_strides.len(), shape.len());

    // Invariant 2: TensorLayout construction must not panic
    let layout = if input.col_major {
        TensorLayout::col_major(&shape, element_size)
    } else {
        TensorLayout::contiguous(&shape, element_size)
    };

    // Invariant 3: numel must equal product of shape
    let expected_numel: usize = shape.iter().product();
    assert_eq!(layout.numel(), expected_numel);

    // Invariant 4: byte_size must equal numel * element_size
    assert_eq!(layout.byte_size(), expected_numel * element_size);

    // Invariant 5: ndim must equal shape length
    assert_eq!(layout.ndim(), shape.len());

    // Invariant 6: freshly constructed layout is contiguous
    assert!(layout.is_contiguous());

    // Invariant 7: offset with valid indices must return Some
    let valid_indices: Vec<usize> = shape.iter().map(|&d| d / 2).collect();
    let offset = layout.offset(&valid_indices);
    assert!(offset.is_some(), "valid indices should produce an offset");

    // Invariant 8: offset with out-of-bounds indices must return None
    let mut oob_indices = valid_indices.clone();
    if let Some(last) = oob_indices.last_mut() {
        *last = *shape.last().unwrap();
    }
    assert!(layout.offset(&oob_indices).is_none());

    // Invariant 9: offset with wrong rank must return None
    let wrong_rank: Vec<usize> = vec![0; shape.len() + 1];
    assert!(layout.offset(&wrong_rank).is_none());

    // Invariant 10: transpose must not panic
    let dim0 = input.transpose_dim0 as usize;
    let dim1 = input.transpose_dim1 as usize;
    match layout.transpose(dim0, dim1) {
        Some(transposed) => {
            assert_eq!(transposed.numel(), layout.numel());
            assert_eq!(transposed.ndim(), layout.ndim());
            if dim0 != dim1 {
                assert_eq!(transposed.shape[dim0], layout.shape[dim1]);
                assert_eq!(transposed.shape[dim1], layout.shape[dim0]);
            }
        }
        None => {
            // Out-of-bounds dims correctly rejected
            assert!(dim0 >= layout.ndim() || dim1 >= layout.ndim());
        }
    }

    // Invariant 11: reshape must not panic
    let new_shape: Vec<usize> =
        input.reshape_dims.iter().take(6).map(|&d| (d as usize % 64) + 1).collect();
    if !new_shape.is_empty() {
        match layout.reshape(&new_shape) {
            Some(reshaped) => {
                assert_eq!(reshaped.numel(), layout.numel());
                assert!(reshaped.is_contiguous());
            }
            None => {
                // Reshape only succeeds if numel matches and layout is contiguous
            }
        }
    }

    // Invariant 12: is_aligned must not panic
    let alignment = input.alignment as usize;
    let _ = layout.is_aligned(alignment);
    let _ = layout.is_aligned(0); // edge case: zero alignment

    // Invariant 13: broadcastable must not panic on arbitrary shapes
    let shape_b: Vec<usize> = input.dims_b.iter().take(6).map(|&d| (d as usize % 64) + 1).collect();
    if !shape_b.is_empty() {
        let result = broadcastable(&shape, &shape_b);
        // broadcastable is symmetric for equal-length shapes with 1s
        if shape.len() == shape_b.len() {
            let all_ones_a = shape.iter().all(|&d| d == 1);
            let all_ones_b = shape_b.iter().all(|&d| d == 1);
            if all_ones_a || all_ones_b {
                assert!(result, "all-ones shape should be broadcastable");
            }
        }
    }

    // Invariant 14: empty shape edge cases
    assert!(compute_strides(&[], LayoutOrder::RowMajor).is_empty());
    assert!(broadcastable(&[], &[]));
});
