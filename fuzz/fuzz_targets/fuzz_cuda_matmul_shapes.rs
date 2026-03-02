#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cuda::{MatmulConfig, matmul_cpu};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct CudaMatmulInput {
    m: u8,
    n: u8,
    k: u8,
    batch_size: u8,
    a_data: Vec<u8>,
    b_data: Vec<u8>,
    transpose_a: bool,
    transpose_b: bool,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: CudaMatmulInput| {
    let m = (input.m as usize % 16) + 1;
    let n = (input.n as usize % 16) + 1;
    let k = (input.k as usize % 16) + 1;
    let batch_size = (input.batch_size as usize % 2) + 1;

    let a_rows = if input.transpose_a { k } else { m };
    let a_cols = if input.transpose_a { m } else { k };
    let b_rows = if input.transpose_b { n } else { k };
    let b_cols = if input.transpose_b { k } else { n };

    let a_elems = batch_size * a_rows * a_cols;
    let b_elems = batch_size * b_rows * b_cols;
    let out_elems = batch_size * m * n;

    let a = bytes_to_f32(&input.a_data, a_elems);
    let b = bytes_to_f32(&input.b_data, b_elems);

    if a.len() < a_elems || b.len() < b_elems {
        return;
    }

    // Skip non-finite inputs.
    if a[..a_elems].iter().chain(b[..b_elems].iter()).any(|x| !x.is_finite()) {
        return;
    }

    let mut config = match MatmulConfig::for_shape(m, n, k) {
        Ok(c) => c,
        Err(_) => return,
    };
    config.batch_size = batch_size;
    config.transpose_a = input.transpose_a;
    config.transpose_b = input.transpose_b;

    let mut output = vec![0.0f32; out_elems];
    if matmul_cpu(&a[..a_elems], &b[..b_elems], &mut output, &config).is_ok() {
        // Invariant 1: Output has exactly batch_size * m * n elements.
        assert_eq!(output.len(), out_elems, "matmul output shape mismatch");

        // Invariant 2: All outputs are finite (given finite inputs with small dims).
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "matmul output non-finite at index {i}: {val} (m={m}, n={n}, k={k})"
            );
        }
    }

    // Invariant 3: Zero matrix times anything = zero matrix.
    let zero_a = vec![0.0f32; a_elems];
    let mut zero_out = vec![0.0f32; out_elems];
    // Reset beta to 0 to ensure clean output.
    config.beta = 0.0;
    if matmul_cpu(&zero_a, &b[..b_elems], &mut zero_out, &config).is_ok() {
        for (i, &val) in zero_out.iter().enumerate() {
            assert!(val.abs() < 1e-10, "zero*B should be zero at index {i}, got {val}");
        }
    }
});
