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
    1.0f32..=1_000_000.0f32
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

    /// Every entry in the generated sin/cos tables is a finite f32 (no NaN or Inf).
    #[test]
    fn all_table_values_are_finite(
        dim     in arb_even_dim(),
        seq_len in arb_seq_len(),
        base    in arb_valid_base(),
    ) {
        let tables = build_tables(dim, seq_len, base).expect("valid inputs must succeed");
        for (i, &v) in tables.sin.iter().enumerate() {
            prop_assert!(v.is_finite(), "sin[{i}] is not finite: {v}");
        }
        for (i, &v) in tables.cos.iter().enumerate() {
            prop_assert!(v.is_finite(), "cos[{i}] is not finite: {v}");
        }
    }

    /// At sequence position 0 the entire row is the identity: sin ≈ 0, cos ≈ 1.
    ///
    /// This holds because every angle at position 0 is `0 * inv_freq[i] = 0`.
    #[test]
    fn position_zero_row_is_always_identity(
        dim     in arb_even_dim(),
        seq_len in arb_seq_len(),
        base    in arb_valid_base(),
    ) {
        let tables = build_tables(dim, seq_len, base).expect("valid inputs must succeed");
        let half = tables.half_dim;
        for i in 0..half {
            prop_assert!(
                tables.sin[i].abs() < 1e-6,
                "sin[{i}] at pos=0 should be ~0, got {}",
                tables.sin[i]
            );
            prop_assert!(
                (tables.cos[i] - 1.0_f32).abs() < 1e-6,
                "cos[{i}] at pos=0 should be ~1, got {}",
                tables.cos[i]
            );
        }
    }

    /// The first inv_freq component is always 1.0 (base^0 = 1), so at position 1
    /// the first sin/cos entry equals sin(1.0)/cos(1.0) regardless of base or dim.
    #[test]
    fn first_freq_component_at_position_one_is_unit(
        dim  in arb_even_dim(),
        base in arb_valid_base(),
    ) {
        // Need seq_len >= 2 to have a position-1 row.
        let tables = build_tables(dim, 2, base).expect("valid inputs must succeed");
        let half = tables.half_dim;
        let expected_sin = 1.0_f32.sin();
        let expected_cos = 1.0_f32.cos();
        prop_assert!(
            (tables.sin[half] - expected_sin).abs() < 1e-5,
            "sin at pos=1, freq=0 should be sin(1)={expected_sin}, got {}",
            tables.sin[half]
        );
        prop_assert!(
            (tables.cos[half] - expected_cos).abs() < 1e-5,
            "cos at pos=1, freq=0 should be cos(1)={expected_cos}, got {}",
            tables.cos[half]
        );
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
