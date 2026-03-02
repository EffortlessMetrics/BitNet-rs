#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::concat::ConcatKernel;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ConcatInput {
    rows_a: u8,
    rows_b: u8,
    cols: u8,
    axis: u8,
    num_splits: u8,
    data_a: Vec<u8>,
    data_b: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    data.chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: ConcatInput| {
    let rows_a = (input.rows_a as usize % 8) + 1;
    let rows_b = (input.rows_b as usize % 8) + 1;
    let cols = (input.cols as usize % 8) + 1;
    let axis = input.axis as usize % 2; // 0 or 1 for 2D

    let elems_a = rows_a * cols;
    let elems_b = rows_b * cols;

    let a = bytes_to_f32(&input.data_a, elems_a);
    let b = bytes_to_f32(&input.data_b, elems_b);

    // --- concat along axis 0 (stacking rows) ---
    if axis == 0 && a.len() >= elems_a && b.len() >= elems_b {
        let a_slice = &a[..elems_a];
        let b_slice = &b[..elems_b];
        let shape_a: &[usize] = &[rows_a, cols];
        let shape_b: &[usize] = &[rows_b, cols];
        let inputs: &[&[f32]] = &[a_slice, b_slice];
        let shapes: &[&[usize]] = &[shape_a, shape_b];

        if let Ok(out) = ConcatKernel::concat(inputs, shapes, 0) {
            assert_eq!(out.len(), elems_a + elems_b);
        }
    }

    // --- split ---
    if a.len() >= elems_a && rows_a >= 2 {
        let num_splits = ((input.num_splits as usize % rows_a) + 1).min(rows_a);
        if rows_a % num_splits == 0 {
            let shape: &[usize] = &[rows_a, cols];
            if let Ok(parts) = ConcatKernel::split(&a[..elems_a], shape, 0, num_splits) {
                assert_eq!(parts.len(), num_splits);
                let total: usize = parts.iter().map(|p| p.len()).sum();
                assert_eq!(total, elems_a);
            }
        }
    }

    // --- stack (same-shape tensors along new axis) ---
    if a.len() >= cols && b.len() >= cols {
        let a_row = &a[..cols];
        let b_row = &b[..cols];
        let shape_1d: &[usize] = &[cols];
        let inputs: &[&[f32]] = &[a_row, b_row];

        if let Ok(out) = ConcatKernel::stack(inputs, shape_1d, 0) {
            assert_eq!(out.len(), 2 * cols);
        }
    }

    // --- empty inputs should return empty ---
    {
        let empty: &[&[f32]] = &[];
        let empty_shapes: &[&[usize]] = &[];
        if let Ok(out) = ConcatKernel::concat(empty, empty_shapes, 0) {
            assert!(out.is_empty());
        }
    }
});
