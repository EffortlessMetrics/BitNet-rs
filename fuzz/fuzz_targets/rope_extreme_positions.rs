#![no_main]

//! Fuzz target: RoPE table generation with extreme position indices.
//!
//! Invariants verified:
//!   1. `build_tables` never panics for any input combination.
//!   2. When successful, `sin` and `cos` tables have exactly `max_seq_len * half_dim`
//!      elements.
//!   3. All elements of a successful table are finite (no NaN/Inf leaks into
//!      the produced tables regardless of the RoPE base).
//!   4. `build_tables` with `dim=0` always returns `Err(ZeroDimension)`.
//!   5. `build_tables` with an odd `dim` always returns `Err(OddDimension)`.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use bitnet_rope::{RopeTableError, build_tables};

#[derive(Arbitrary, Debug)]
struct Input {
    /// Head dimension (will be forced even for happy-path, kept as-is for error path).
    dim_raw: u16,
    /// Maximum sequence length — exercising both zero and u32::MAX-range values.
    max_seq_len_raw: u32,
    /// RoPE theta/base as raw bytes → reinterpreted as f32.
    base_bytes: [u8; 4],
}

fuzz_target!(|input: Input| {
    let base = f32::from_le_bytes(input.base_bytes);

    // --- Invariant 4: dim == 0 always errors ---
    let result_zero = build_tables(0, input.max_seq_len_raw as usize, base);
    assert!(
        matches!(result_zero, Err(RopeTableError::ZeroDimension)),
        "dim=0 must return ZeroDimension, got: {result_zero:?}"
    );

    // --- Invariant 5: odd dim always errors ---
    let odd_dim = (input.dim_raw as usize) | 1; // force lowest bit set → always odd
    if odd_dim > 0 {
        let result_odd = build_tables(odd_dim, input.max_seq_len_raw as usize, base);
        assert!(
            matches!(result_odd, Err(RopeTableError::OddDimension { .. })),
            "odd dim={odd_dim} must return OddDimension, got: {result_odd:?}"
        );
    }

    // --- Invariants 1-3: even dim, clamp sizes to avoid OOM ---
    // Clamp dim and seq_len to keep heap allocation bounded.
    let dim = ((input.dim_raw as usize & !1).clamp(2, 256)).max(2); // even, 2..=256
    // Allow extreme positions up to 2^18 to cover large seq_len without OOM.
    let max_seq_len = (input.max_seq_len_raw as usize).min(1 << 18);

    // Invariant 1: must not panic (may return Err for bad base).
    let result = build_tables(dim, max_seq_len, base);

    match result {
        Ok(tables) => {
            let expected_len = max_seq_len * (dim / 2);

            // Invariant 2: table lengths are exact.
            assert_eq!(
                tables.sin.len(),
                expected_len,
                "sin table length mismatch: dim={dim}, max_seq_len={max_seq_len}"
            );
            assert_eq!(
                tables.cos.len(),
                expected_len,
                "cos table length mismatch: dim={dim}, max_seq_len={max_seq_len}"
            );
            assert_eq!(
                tables.half_dim,
                dim / 2,
                "half_dim field mismatch"
            );

            // Invariant 3: no NaN or Inf in the output tables.
            for (i, &v) in tables.sin.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "sin[{i}] is not finite ({v}) for dim={dim}, max_seq_len={max_seq_len}, base={base}"
                );
            }
            for (i, &v) in tables.cos.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "cos[{i}] is not finite ({v}) for dim={dim}, max_seq_len={max_seq_len}, base={base}"
                );
            }
        }
        Err(RopeTableError::NonFiniteBase { .. } | RopeTableError::NonPositiveBase { .. }) => {
            // Expected for NaN/Inf/negative/zero base — not a bug.
        }
        Err(e) => {
            // ZeroDimension / OddDimension should not occur here since we
            // constrained dim to be an even positive value.
            panic!("unexpected error for dim={dim}, max_seq_len={max_seq_len}, base={base}: {e}");
        }
    }
});
