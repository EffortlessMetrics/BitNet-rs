#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct MatMulShapeInput {
    m: u8,
    n: u8,
    k: u8,
    a_data: Vec<u8>,
    b_data: Vec<u8>,
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

fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn transpose(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = mat[i * cols + j];
        }
    }
    out
}

fuzz_target!(|input: MatMulShapeInput| {
    let m = (input.m as usize % 32) + 1;
    let n = (input.n as usize % 32) + 1;
    let k = (input.k as usize % 32) + 1;

    let a_elems = m * k;
    let b_elems = k * n;

    let a = bytes_to_f32(&input.a_data, a_elems);
    let b_raw = bytes_to_f32(&input.b_data, b_elems);

    if a.len() < a_elems || b_raw.len() < b_elems {
        return;
    }

    let a = &a[..a_elems];

    // Filter non-finite inputs
    if a.iter().chain(b_raw[..b_elems].iter()).any(|x| !x.is_finite()) {
        return;
    }

    let b;
    let b_slice = if input.transpose_b {
        // b_raw is [n, k], transpose to [k, n]
        b = transpose(&b_raw[..b_elems], n, k);
        &b[..]
    } else {
        &b_raw[..b_elems]
    };

    let c = matmul(a, b_slice, m, n, k);

    // Invariant 1: Output has exactly m*n elements
    assert_eq!(c.len(), m * n, "output shape: expected {}x{}={}, got {}", m, n, m * n, c.len());

    // Invariant 2: All outputs are finite (given finite inputs)
    for (i, &val) in c.iter().enumerate() {
        assert!(
            val.is_finite(),
            "matmul output non-finite at index {i}: {val} (m={m}, n={n}, k={k})"
        );
    }

    // Invariant 3: Zero matrix times anything = zero matrix
    let zero_a = vec![0.0f32; a_elems];
    let c_zero = matmul(&zero_a, b_slice, m, n, k);
    for (i, &val) in c_zero.iter().enumerate() {
        assert_eq!(val, 0.0, "zero*B should be zero at index {i}, got {val}");
    }

    // Invariant 4: Identity property for square matrices
    if m == k && k == n && m <= 16 {
        let mut identity = vec![0.0f32; m * m];
        for i in 0..m {
            identity[i * m + i] = 1.0;
        }
        let c_id = matmul(a, &identity, m, m, m);
        for (i, (&orig, &result)) in a.iter().zip(c_id.iter()).enumerate() {
            let diff = (orig - result).abs();
            assert!(diff < 1e-4, "A*I != A at index {i}: {orig} vs {result} (diff={diff})");
        }
    }

    // Invariant 5: (A*B) output dimensions match regardless of transpose path
    if input.transpose_b {
        let b_direct = &b_raw[..b_elems];
        let b_t = transpose(b_direct, n, k);
        let c2 = matmul(a, &b_t, m, n, k);
        assert_eq!(c.len(), c2.len(), "transpose path should produce same output shape");
    }
});
