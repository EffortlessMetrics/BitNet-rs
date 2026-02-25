//! Property-based tests for `bitnet-rope`.
//!
//! Verifies correctness invariants of RoPE table generation:
//! - Output dimensions match input spec
//! - sin²(θ) + cos²(θ) ≈ 1 for every entry
//! - Error paths are exhaustive

use bitnet_rope::{DEFAULT_ROPE_BASE, RopeTableError, build_tables, resolve_base};
use proptest::prelude::*;

// ── Strategies ──────────────────────────────────────────────────────────────

/// A valid even dimension in [2, 128].
fn arb_even_dim() -> impl Strategy<Value = usize> {
    (1usize..=64).prop_map(|n| n * 2)
}

/// A valid sequence length in [1, 64].
fn arb_seq_len() -> impl Strategy<Value = usize> {
    1usize..=64
}

/// A valid RoPE base in (0, 1_000_000].
fn arb_valid_base() -> impl Strategy<Value = f32> {
    (1.0f32..=1_000_000.0f32)
}

// ── Property tests ───────────────────────────────────────────────────────────

proptest! {
    /// Output table dimensions match: sin.len() == cos.len() == max_seq_len × (dim/2).
    #[test]
    fn output_dimensions_are_correct(
        dim in arb_even_dim(),
        seq_len in arb_seq_len(),
        base in arb_valid_base(),
    ) {
        let tables = build_tables(dim, seq_len, base).expect("valid inputs should succeed");
        let expected_len = seq_len * (dim / 2);
        prop_assert_eq!(tables.sin.len(), expected_len);
        prop_assert_eq!(tables.cos.len(), expected_len);
        prop_assert_eq!(tables.half_dim, dim / 2);
    }

    /// sin²(θ) + cos²(θ) ≈ 1.0 for every position/frequency pair.
    #[test]
    fn pythagorean_identity_holds(
        dim in arb_even_dim(),
        seq_len in arb_seq_len(),
        base in arb_valid_base(),
    ) {
        let tables = build_tables(dim, seq_len, base).expect("valid inputs should succeed");
        for (i, (&s, &c)) in tables.sin.iter().zip(tables.cos.iter()).enumerate() {
            let identity = s * s + c * c;
            let err = (identity - 1.0_f32).abs();
            prop_assert!(
                err < 1e-5,
                "Pythagorean identity violated at index {i}: sin²+cos²={identity:.8} (err={err:.2e})"
            );
        }
    }

    /// Zero seq_len produces empty tables (no error).
    #[test]
    fn zero_seq_len_produces_empty_tables(
        dim in arb_even_dim(),
        base in arb_valid_base(),
    ) {
        let tables = build_tables(dim, 0, base).expect("zero seq_len should produce empty tables");
        prop_assert!(tables.sin.is_empty());
        prop_assert!(tables.cos.is_empty());
    }

    /// Odd dimensions always return `OddDimension` error.
    #[test]
    fn odd_dimension_returns_error(
        odd_dim in (1usize..=127).prop_map(|n| n * 2 - 1),  // 1,3,5,...,253
        seq_len in arb_seq_len(),
        base in arb_valid_base(),
    ) {
        let result = build_tables(odd_dim, seq_len, base);
        prop_assert!(
            matches!(result, Err(RopeTableError::OddDimension { dim }) if dim == odd_dim),
            "expected OddDimension error for dim={odd_dim}"
        );
    }

    /// Non-positive bases always return an error.
    #[test]
    fn non_positive_base_returns_error(
        neg_base in prop_oneof![
            (-1_000.0f32..=0.0f32),  // negative and zero
        ],
        dim in arb_even_dim(),
        seq_len in arb_seq_len(),
    ) {
        let result = build_tables(dim, seq_len, neg_base);
        prop_assert!(result.is_err(), "expected error for base={neg_base}");
    }

    /// resolve_base with Some(x) always returns x.
    #[test]
    fn resolve_base_some_returns_value(base in arb_valid_base()) {
        prop_assert_eq!(resolve_base(Some(base)), base);
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[test]
fn resolve_base_none_returns_default() {
    assert_eq!(resolve_base(None), DEFAULT_ROPE_BASE);
}

#[test]
fn zero_dimension_returns_error() {
    let err = build_tables(0, 10, DEFAULT_ROPE_BASE).unwrap_err();
    assert_eq!(err, RopeTableError::ZeroDimension);
}

#[test]
fn non_finite_base_returns_error() {
    assert!(matches!(
        build_tables(64, 10, f32::NAN).unwrap_err(),
        RopeTableError::NonFiniteBase { .. }
    ));
    assert!(matches!(
        build_tables(64, 10, f32::INFINITY).unwrap_err(),
        RopeTableError::NonFiniteBase { .. }
    ));
}

#[test]
fn typical_llama_config_builds_successfully() {
    // LLaMA-3 8B: head_dim=128, max_seq=8192
    let tables = build_tables(128, 8192, DEFAULT_ROPE_BASE).unwrap();
    assert_eq!(tables.half_dim, 64);
    assert_eq!(tables.sin.len(), 8192 * 64);
}
