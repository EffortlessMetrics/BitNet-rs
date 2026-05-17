//! Shared helpers for AVX2-vs-scalar parity integration tests.

pub fn close(a: f32, b: f32, abs_tol: f32, rel_tol: f32) -> bool {
    let diff = (a - b).abs();
    diff <= abs_tol || diff <= rel_tol * a.abs().max(b.abs())
}

pub fn assert_vec_parity(actual: &[f32], expected: &[f32], abs_tol: f32, rel_tol: f32, ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(a, e, abs_tol, rel_tol),
            "{ctx}[{i}]: scalar={e}, dispatched={a} (diff={}, abs_tol={abs_tol}, rel_tol={rel_tol})",
            (a - e).abs()
        );
    }
}

/// Deterministic pseudo-random f32 values in [-1, 1] for reproducibility.
pub fn pseudo_rand(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}
