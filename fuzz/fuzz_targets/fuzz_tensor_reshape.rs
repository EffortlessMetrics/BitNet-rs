#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::concat::ConcatKernel;
use libfuzzer_sys::fuzz_target;

/// Fuzz ConcatKernel operations: concat, split, stack, and shape
/// computation with random shapes and data.
#[derive(Arbitrary, Debug)]
struct ReshapeInput {
    data: Vec<u8>,
    op: u8,
    dim_a: u8,
    dim_b: u8,
    dim_c: u8,
    axis: u8,
    n_splits: u8,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: ReshapeInput| {
    let values = bytes_to_f32(&input.data, 512);
    if values.is_empty() {
        return;
    }

    match input.op % 4 {
        0 => {
            // Concat two tensors along a fuzzed axis
            let rows = (input.dim_a as usize % 4) + 1;
            let cols = (input.dim_b as usize % 4) + 1;
            let extra_cols = (input.dim_c as usize % 4) + 1;
            let shape_a = [rows, cols];
            let shape_b = [rows, extra_cols];
            let total_a = rows * cols;
            let total_b = rows * extra_cols;
            if values.len() < total_a + total_b {
                return;
            }
            let a = &values[..total_a];
            let b = &values[total_a..total_a + total_b];
            let axis = input.axis as usize % 2;

            let inputs: &[&[f32]] = &[a, b];
            let sa: &[usize] = &shape_a;
            let sb: &[usize] = &shape_b;
            let shapes: &[&[usize]] = &[sa, sb];

            // Concat along valid axis
            if axis == 1 {
                // Axis 1: rows must match
                if let Ok(result) = ConcatKernel::concat(inputs, shapes, 1) {
                    assert_eq!(result.len(), rows * (cols + extra_cols));
                }
            } else {
                // Axis 0: cols must match — only try if equal
                if cols == extra_cols {
                    if let Ok(result) = ConcatKernel::concat(inputs, shapes, 0) {
                        assert_eq!(result.len(), (rows * 2) * cols);
                    }
                }
            }

            // Verify concat_output_shape
            if axis == 1 {
                if let Ok(out_shape) = ConcatKernel::concat_output_shape(shapes, 1) {
                    assert_eq!(out_shape, vec![rows, cols + extra_cols]);
                }
            }
        }
        1 => {
            // Split then re-concat should be identity
            let rows = (input.dim_a as usize % 4) + 1;
            let cols = (input.dim_b as usize % 8) + 2;
            let total = rows * cols;
            if values.len() < total {
                return;
            }
            let data = &values[..total];
            let shape = [rows, cols];
            let n_splits = (input.n_splits as usize % cols).max(1);
            if cols % n_splits != 0 {
                return;
            }

            if let Ok(parts) = ConcatKernel::split(data, &shape, 1, n_splits) {
                assert_eq!(parts.len(), n_splits);
                let chunk_cols = cols / n_splits;
                for p in &parts {
                    assert_eq!(p.len(), rows * chunk_cols);
                }

                // Re-concat should recover original
                let refs: Vec<&[f32]> = parts.iter().map(|p| p.as_slice()).collect();
                let chunk_shape = [rows, chunk_cols];
                let shapes: Vec<&[usize]> = (0..n_splits).map(|_| chunk_shape.as_slice()).collect();
                if let Ok(recovered) = ConcatKernel::concat(&refs, &shapes, 1) {
                    assert_eq!(recovered.len(), total);
                    for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
                        if orig.is_finite() && rec.is_finite() {
                            assert!(
                                (orig - rec).abs() < 1e-6,
                                "split→concat not identity at {i}: {orig} vs {rec}"
                            );
                        }
                    }
                }
            }
        }
        2 => {
            // Stack multiple 1-D vectors
            let dim = (input.dim_a as usize % 8) + 1;
            let n = (input.dim_b as usize % 4) + 2;
            let total = n * dim;
            if values.len() < total {
                return;
            }

            let vecs: Vec<&[f32]> = (0..n).map(|i| &values[i * dim..(i + 1) * dim]).collect();
            let shape = [dim];
            let axis = input.axis as usize % 2;

            if let Ok(stacked) = ConcatKernel::stack(&vecs, &shape, axis) {
                if let Ok(expected_shape) = ConcatKernel::stack_output_shape(&shape, axis, n) {
                    let expected_numel: usize = expected_shape.iter().product();
                    assert_eq!(stacked.len(), expected_numel);
                }
            }
        }
        _ => {
            // Split with custom sizes
            let rows = (input.dim_a as usize % 4) + 1;
            let cols = (input.dim_b as usize % 8) + 2;
            let total = rows * cols;
            if values.len() < total {
                return;
            }
            let data = &values[..total];
            let shape = [rows, cols];

            // Generate sizes that sum to cols
            let s1 = (input.dim_c as usize % cols).max(1);
            let s2 = cols - s1;
            if s2 == 0 {
                return;
            }
            let sizes = [s1, s2];

            if let Ok(parts) = ConcatKernel::split_sizes(data, &shape, 1, &sizes) {
                assert_eq!(parts.len(), 2);
                assert_eq!(parts[0].len(), rows * s1);
                assert_eq!(parts[1].len(), rows * s2);
            }

            // Verify split_output_shapes with equal splits
            let n_equal = if cols >= 2 { 2 } else { 1 };
            if cols % n_equal == 0 {
                if let Ok(out_shapes) = ConcatKernel::split_output_shapes(&shape, 1, n_equal) {
                    assert_eq!(out_shapes.len(), n_equal);
                    for s in &out_shapes {
                        assert_eq!(s[0], rows);
                        assert_eq!(s[1], cols / n_equal);
                    }
                }
            }
        }
    }
});
