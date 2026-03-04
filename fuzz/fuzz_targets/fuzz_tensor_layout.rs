#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::tensor_layout::{LayoutOrder, TensorLayout};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct LayoutInput {
    /// Shape dimensions (capped to reasonable sizes).
    dims: Vec<u16>,
    /// Whether to use column-major order.
    col_major: bool,
}

fuzz_target!(|input: LayoutInput| {
    // Cap dimensions to avoid huge allocations
    let shape: Vec<usize> = input.dims.iter().take(6).map(|&d| (d as usize % 64) + 1).collect();

    if shape.is_empty() {
        return;
    }

    // Invariant 1: TensorLayout construction must not panic
    let layout = if input.col_major {
        TensorLayout::col_major(shape.clone())
    } else {
        TensorLayout::contiguous(shape.clone(), LayoutOrder::RowMajor)
    };

    // Invariant 2: numel must equal product of shape
    let expected_numel: usize = shape.iter().product();
    assert_eq!(layout.numel(), expected_numel);

    // Invariant 3: ndim must equal shape length
    assert_eq!(layout.ndim(), shape.len());

    // Invariant 4: freshly constructed row-major layout is contiguous
    if !input.col_major {
        assert!(layout.is_contiguous());
    }

    // Invariant 5: linear_offset with valid indices must return Some
    let valid_indices: Vec<usize> = shape.iter().map(|&d| d / 2).collect();
    let offset = layout.linear_offset(&valid_indices);
    assert!(offset.is_some(), "valid indices should produce an offset");

    // Invariant 6: linear_offset with out-of-bounds indices must return None
    let mut oob_indices = valid_indices.clone();
    if let Some(last) = oob_indices.last_mut() {
        *last = *shape.last().unwrap();
    }
    assert!(layout.linear_offset(&oob_indices).is_none());

    // Invariant 7: linear_offset with wrong rank must return None
    let wrong_rank: Vec<usize> = vec![0; shape.len() + 1];
    assert!(layout.linear_offset(&wrong_rank).is_none());

    // Invariant 8: transpose must not panic (swaps last 2 dims)
    if layout.ndim() >= 2 {
        let transposed = layout.transpose();
        assert!(transposed.is_some());
        let transposed = transposed.unwrap();
        assert_eq!(transposed.numel(), layout.numel());
        assert_eq!(transposed.ndim(), layout.ndim());
    } else {
        assert!(layout.transpose().is_none());
    }
});
