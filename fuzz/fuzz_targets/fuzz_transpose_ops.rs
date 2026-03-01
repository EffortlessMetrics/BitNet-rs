#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::transpose::TransposeKernel;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct TransposeInput {
    data: Vec<u8>,
    shape: Vec<u8>,
    perm: Vec<u8>,
    new_shape: Vec<u8>,
    start_dim: u8,
    end_dim: u8,
    unsqueeze_dim: u8,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: TransposeInput| {
    let values = bytes_to_f32(&input.data, 512);
    if values.is_empty() {
        return;
    }

    // --- 2D transpose ---
    let rows = (input.shape.first().copied().unwrap_or(1) as usize % 16) + 1;
    let cols = (input.shape.get(1).copied().unwrap_or(1) as usize % 16) + 1;
    if values.len() >= rows * cols {
        let mat = &values[..rows * cols];
        if let Ok(transposed) = TransposeKernel::transpose_2d(mat, rows, cols) {
            assert_eq!(transposed.len(), rows * cols);
            // Double transpose is identity
            if let Ok(back) = TransposeKernel::transpose_2d(&transposed, cols, rows) {
                assert_eq!(back.len(), mat.len());
                for (a, b) in back.iter().zip(mat.iter()) {
                    assert!(
                        (a - b).abs() < 1e-6 || (a.is_nan() && b.is_nan()),
                        "double transpose not identity: {a} vs {b}"
                    );
                }
            }
        }
    }

    // --- N-dimensional transpose ---
    let shape: Vec<usize> = input.shape.iter().take(4).map(|&d| (d as usize % 8) + 1).collect();
    let numel: usize = shape.iter().product();
    if numel > 0 && values.len() >= numel {
        let data = &values[..numel];
        let ndim = shape.len();
        let perm: Vec<usize> = if input.perm.len() >= ndim {
            // Build a permutation from fuzz bytes
            let mut p: Vec<usize> = (0..ndim).collect();
            for i in 0..ndim {
                let j = input.perm[i] as usize % (ndim - i) + i;
                p.swap(i, j);
            }
            p
        } else {
            (0..ndim).collect()
        };
        let _ = TransposeKernel::transpose_nd(data, &shape, &perm);
    }

    // --- Reshape ---
    let new_shape: Vec<usize> =
        input.new_shape.iter().take(4).map(|&d| (d as usize % 8) + 1).collect();
    let new_numel: usize = new_shape.iter().product();
    if numel > 0 && numel == new_numel && values.len() >= numel {
        let data = &values[..numel];
        let _ = TransposeKernel::reshape(data, &shape, &new_shape);
    }

    // --- Flatten ---
    if !shape.is_empty() && numel > 0 && values.len() >= numel {
        let data = &values[..numel];
        let ndim = shape.len();
        let s = input.start_dim as usize % ndim;
        let e = input.end_dim as usize % ndim;
        let (start, end) = if s <= e { (s, e) } else { (e, s) };
        let _ = TransposeKernel::flatten(data, &shape, start, end);
    }

    // --- Squeeze / Unsqueeze ---
    let _ = TransposeKernel::squeeze(&shape);
    let dim = input.unsqueeze_dim as usize;
    let _ = TransposeKernel::unsqueeze(&shape, dim);
});
