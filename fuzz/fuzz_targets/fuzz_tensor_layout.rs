#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::tensor_layout::TensorLayout;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct LayoutInput {
    /// Shape dimensions (capped to reasonable sizes).
    dims: Vec<u16>,
    /// Element size selector.
    element_size: u8,
    /// _Indices placeholder (used via valid_indices below)._
    _indices: Vec<u16>,
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

    // Invariant 1: TensorLayout construction must not panic
    let layout = if input.col_major {
        TensorLayout::column_major(shape.clone())
    } else {
        TensorLayout::contiguous(shape.clone())
    };

    // Invariant 2: numel must equal product of shape
    let expected_numel: usize = shape.iter().product();
    assert_eq!(layout.numel(), expected_numel);

    // Invariant 3: size_bytes must equal numel * element_size
    assert_eq!(layout.size_bytes(element_size), expected_numel * element_size);

    // Invariant 4: ndim must equal shape length
    assert_eq!(layout.ndim(), shape.len());

    // Invariant 5: freshly constructed row-major layout is contiguous
    if !input.col_major {
        assert!(layout.is_contiguous());
    }

    // Invariant 6: flat_offset with valid indices must return Some
    let valid_indices: Vec<usize> = shape.iter().map(|&d| d / 2).collect();
    let offset = layout.flat_offset(&valid_indices);
    assert!(offset.is_some(), "valid indices should produce an offset");

    // Invariant 7: flat_offset with out-of-bounds indices must return None
    let mut oob_indices = valid_indices.clone();
    if let Some(last) = oob_indices.last_mut() {
        *last = *shape.last().unwrap();
    }
    assert!(layout.flat_offset(&oob_indices).is_none());

    // Invariant 8: flat_offset with wrong rank must return None
    let wrong_rank: Vec<usize> = vec![0; shape.len() + 1];
    assert!(layout.flat_offset(&wrong_rank).is_none());

    // Invariant 9: transpose must not panic (swaps last 2 dims)
    if layout.ndim() >= 2 {
        let transposed = layout.transpose();
        assert!(transposed.is_some());
        let transposed = transposed.unwrap();
        assert_eq!(transposed.numel(), layout.numel());
        assert_eq!(transposed.ndim(), layout.ndim());
    } else {
        assert!(layout.transpose().is_none());
    }

    // Invariant 10: reshape must not panic
    let new_shape: Vec<usize> =
        input.reshape_dims.iter().take(6).map(|&d| (d as usize % 64) + 1).collect();
    if !new_shape.is_empty() {
        match layout.reshape(new_shape) {
            Some(reshaped) => {
                assert_eq!(reshaped.numel(), layout.numel());
                assert!(reshaped.is_contiguous());
            }
            None => {
                // Reshape only succeeds if numel matches and layout is contiguous
            }
        }
    }

    // Invariant 11: is_aligned must not panic
    let alignment = input.alignment as usize;
    let _ = layout.is_aligned(alignment);
    let _ = layout.is_aligned(0); // edge case: zero alignment

    // Invariant 12: is_valid must be true for freshly constructed layouts
    assert!(layout.is_valid());
});
