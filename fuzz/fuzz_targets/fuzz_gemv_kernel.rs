#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::simd_matmul::{SimdMatmulConfig, simd_matmul_f32};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct GemvInput {
    m: u8,
    k: u8,
    batch: u8,
    a_data: Vec<u8>,
    x_data: Vec<u8>,
    use_batch: bool,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn naive_matvec(a: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m];
    for i in 0..m {
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += a[i * k + j] * x[j];
        }
        y[i] = sum;
    }
    y
}

fuzz_target!(|input: GemvInput| {
    let m = (input.m as usize % 32) + 1;
    let k = (input.k as usize % 32) + 1;
    // GEMV is matmul with n=1: C[m,1] = A[m,k] * x[k,1]
    let n = 1;

    let a_elems = m * k;
    let x_elems = k;

    let a = bytes_to_f32(&input.a_data, a_elems);
    let x = bytes_to_f32(&input.x_data, x_elems);

    if a.len() < a_elems || x.len() < x_elems {
        return;
    }

    // Filter non-finite inputs
    if a[..a_elems].iter().chain(x[..x_elems].iter()).any(|v| !v.is_finite()) {
        return;
    }

    // Use simd_matmul_f32 as GEMV: C[m,1] = A[m,k] * B[k,1]
    let cfg = SimdMatmulConfig::new(m, n, k);
    let mut y = vec![0.0f32; m * n];
    if let Ok(()) = simd_matmul_f32(&a[..a_elems], &x[..x_elems], &mut y, &cfg) {
        // Invariant 1: Output length is m
        assert_eq!(y.len(), m);

        // Invariant 2: All outputs are finite
        for (i, &val) in y.iter().enumerate() {
            assert!(val.is_finite(), "gemv non-finite at index {i}: {val} (m={m}, k={k})");
        }

        // Invariant 3: Cross-check against naive implementation
        let expected = naive_matvec(&a[..a_elems], &x[..x_elems], m, k);
        for i in 0..m {
            let diff = (y[i] - expected[i]).abs();
            let tol = 1e-3 * expected[i].abs().max(1.0);
            assert!(diff < tol, "row={i}: simd={} naive={} diff={diff}", y[i], expected[i]);
        }
    }

    // Invariant 4: Zero matrix times any vector = zero vector
    let zero_a = vec![0.0f32; a_elems];
    let mut y_zero = vec![0.0f32; m];
    let cfg_zero = SimdMatmulConfig::new(m, n, k);
    if let Ok(()) = simd_matmul_f32(&zero_a, &x[..x_elems], &mut y_zero, &cfg_zero) {
        for (i, &val) in y_zero.iter().enumerate() {
            assert_eq!(val, 0.0, "zero*x should be zero at index {i}, got {val}");
        }
    }

    // Invariant 5: Any matrix times zero vector = zero vector
    let zero_x = vec![0.0f32; k];
    let mut y_zero2 = vec![0.0f32; m];
    if let Ok(()) = simd_matmul_f32(&a[..a_elems], &zero_x, &mut y_zero2, &cfg_zero) {
        for (i, &val) in y_zero2.iter().enumerate() {
            assert_eq!(val, 0.0, "A*0 should be zero at index {i}, got {val}");
        }
    }

    // Invariant 6: Batched GEMV via multiple single-vector multiplies
    if input.use_batch {
        let batch = (input.batch as usize % 4) + 2;
        let total_a = batch * a_elems;
        let total_x = batch * x_elems;
        let a_batch = bytes_to_f32(&input.a_data, total_a);
        let x_batch = bytes_to_f32(&input.x_data, total_x);

        if a_batch.len() >= total_a
            && x_batch.len() >= total_x
            && a_batch[..total_a].iter().chain(x_batch[..total_x].iter()).all(|v| v.is_finite())
        {
            for b in 0..batch {
                let a_off = b * a_elems;
                let x_off = b * x_elems;
                let cfg_b = SimdMatmulConfig::new(m, n, k);
                let mut y_b = vec![0.0f32; m];
                if let Ok(()) = simd_matmul_f32(
                    &a_batch[a_off..a_off + a_elems],
                    &x_batch[x_off..x_off + x_elems],
                    &mut y_b,
                    &cfg_b,
                ) {
                    for (i, &val) in y_b.iter().enumerate() {
                        assert!(val.is_finite(), "batch={b} gemv non-finite at index {i}: {val}");
                    }
                }
            }
        }
    }
});
