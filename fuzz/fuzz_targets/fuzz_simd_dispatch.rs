#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::simd_math::{simd_dot_product, simd_vector_add};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct SimdInput {
    data_a: Vec<f32>,
    data_b: Vec<f32>,
    op: u8,
}

fuzz_target!(|input: SimdInput| {
    // Limit to 10000 elements as specified.
    let max_len = 10_000;
    let a: Vec<f32> = input.data_a.iter().copied().take(max_len).collect();
    let b: Vec<f32> = input.data_b.iter().copied().take(max_len).collect();

    match input.op % 2 {
        0 => {
            // Dot product: needs equal lengths.
            let len = a.len().min(b.len());
            if len == 0 {
                assert_eq!(simd_dot_product(&[], &[]), 0.0);
                return;
            }
            let a = &a[..len];
            let b = &b[..len];

            let simd_result = simd_dot_product(a, b);

            // Scalar reference.
            let scalar_result: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();

            // Both paths must agree for finite inputs.
            if a.iter().chain(b.iter()).all(|x| x.is_finite()) && scalar_result.is_finite() {
                let tol = scalar_result.abs() * 1e-4 + 1e-4;
                assert!(
                    (simd_result - scalar_result).abs() < tol,
                    "dot product mismatch: simd={simd_result}, scalar={scalar_result}, len={len}",
                );
            }
        }
        _ => {
            // Vector add: needs equal lengths.
            let len = a.len().min(b.len());
            if len == 0 {
                assert!(simd_vector_add(&[], &[]).is_empty());
                return;
            }
            let a = &a[..len];
            let b = &b[..len];

            let result = simd_vector_add(a, b);
            assert_eq!(result.len(), len, "output length mismatch");

            // Scalar reference comparison for finite inputs.
            for (i, ((&ai, &bi), &ri)) in a.iter().zip(b.iter()).zip(result.iter()).enumerate() {
                if ai.is_finite() && bi.is_finite() {
                    let expected = ai + bi;
                    if expected.is_finite() {
                        assert!(
                            (ri - expected).abs() < 1e-5,
                            "vector_add mismatch at {i}: simd={ri}, scalar={expected}",
                        );
                    }
                }
            }
        }
    }
});
