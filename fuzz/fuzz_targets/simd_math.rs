#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::simd_math::{
    fast_exp_f32, fast_sigmoid_f32, fast_tanh_f32, simd_dot_product, simd_vector_add,
};
use libfuzzer_sys::fuzz_target;

/// Fuzz the real SIMD math dispatch functions (AVX2 → scalar fallback)
/// with arbitrary f32 vectors of varying lengths.
#[derive(Arbitrary, Debug)]
struct SimdMathInput {
    /// First vector of f32 values.
    a: Vec<f32>,
    /// Second vector of f32 values (for binary ops).
    b: Vec<f32>,
    /// Which operation to fuzz (mod 5).
    op: u8,
}

fuzz_target!(|input: SimdMathInput| {
    let a: Vec<f32> = input.a.iter().copied().take(256).collect();
    let b: Vec<f32> = input.b.iter().copied().take(256).collect();

    match input.op % 5 {
        0 => {
            // --- dot product: requires equal lengths ---
            let len = a.len().min(b.len());
            if len == 0 {
                // Empty dot product should return 0.
                assert_eq!(simd_dot_product(&[], &[]), 0.0);
                return;
            }
            let a = &a[..len];
            let b = &b[..len];

            // Skip if any input is non-finite.
            if a.iter().chain(b.iter()).any(|x| !x.is_finite()) {
                // Still must not panic.
                let _ = simd_dot_product(a, b);
                return;
            }

            let result = simd_dot_product(a, b);

            // Scalar reference for comparison.
            let reference: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();

            // Both should be finite for finite inputs (barring overflow).
            if reference.is_finite() {
                let tol = reference.abs() * 1e-4 + 1e-4;
                assert!(
                    (result - reference).abs() < tol,
                    "dot product mismatch: simd={result}, scalar={reference}, len={len}"
                );
            }
        }
        1 => {
            // --- vector add: requires equal lengths ---
            let len = a.len().min(b.len());
            if len == 0 {
                assert!(simd_vector_add(&[], &[]).is_empty());
                return;
            }
            let a = &a[..len];
            let b = &b[..len];

            let result = simd_vector_add(a, b);
            assert_eq!(result.len(), len);

            // Element-wise check for finite inputs.
            for (i, ((&ai, &bi), &ri)) in a.iter().zip(b.iter()).zip(result.iter()).enumerate() {
                if ai.is_finite() && bi.is_finite() {
                    let expected = ai + bi;
                    if expected.is_finite() {
                        assert!(
                            (ri - expected).abs() < 1e-5,
                            "vector_add mismatch at {i}: {ri} vs {expected}"
                        );
                    }
                }
            }
        }
        2 => {
            // --- fast_exp_f32 ---
            let result = fast_exp_f32(&a);
            assert_eq!(result.len(), a.len());

            for (i, (&input_val, &out)) in a.iter().zip(result.iter()).enumerate() {
                if input_val.is_nan() {
                    assert!(out.is_nan(), "exp(NaN) should be NaN at {i}");
                } else if input_val.is_finite() {
                    let ref_val = input_val.exp();
                    if ref_val.is_finite() && ref_val > 1e-30 {
                        let rel = ((out - ref_val) / ref_val).abs();
                        assert!(rel < 1e-4, "exp({input_val}) rel error {rel} at {i}");
                    }
                }
            }
        }
        3 => {
            // --- fast_tanh_f32 ---
            let result = fast_tanh_f32(&a);
            assert_eq!(result.len(), a.len());

            for (i, (&input_val, &out)) in a.iter().zip(result.iter()).enumerate() {
                if input_val.is_nan() {
                    assert!(out.is_nan(), "tanh(NaN) should be NaN at {i}");
                } else if input_val.is_finite() {
                    // tanh output must be in [-1, 1].
                    assert!(
                        out >= -1.0 - 1e-5 && out <= 1.0 + 1e-5,
                        "tanh({input_val}) = {out} out of range at {i}"
                    );
                }
            }
        }
        _ => {
            // --- fast_sigmoid_f32 ---
            let result = fast_sigmoid_f32(&a);
            assert_eq!(result.len(), a.len());

            for (i, (&input_val, &out)) in a.iter().zip(result.iter()).enumerate() {
                if input_val.is_nan() {
                    assert!(out.is_nan(), "sigmoid(NaN) should be NaN at {i}");
                } else if input_val.is_finite() {
                    // sigmoid output must be in [0, 1].
                    assert!(
                        out >= -1e-5 && out <= 1.0 + 1e-5,
                        "sigmoid({input_val}) = {out} out of range at {i}"
                    );
                }
            }
        }
    }
});
