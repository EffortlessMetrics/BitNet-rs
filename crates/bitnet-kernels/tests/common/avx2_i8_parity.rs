//! Shared i8 helpers for AVX2-vs-scalar parity integration tests.

/// Deterministic pseudo-random i8 values in [-127, 127].
pub fn pseudo_rand_i8(len: usize, seed: u64) -> Vec<i8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state % 255) as i8).wrapping_sub(127)
        })
        .collect()
}
