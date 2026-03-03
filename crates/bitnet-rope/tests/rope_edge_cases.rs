//! Edge-case tests for the RoPE (Rotary Position Embedding) table generator.
//!
//! Tests cover:
//! - Constants (DEFAULT_ROPE_BASE)
//! - resolve_base (None/Some)
//! - build_tables: valid cases (dim=2, dim=64, dim=128, large seq_len)
//! - build_tables: error cases (zero dim, odd dim, non-finite base, non-positive base)
//! - RopeTableError Display formatting and equality
//! - RopeTables field invariants (half_dim, sin/cos lengths)
//! - Numerical properties (sin/cos in [-1, 1], position 0 gives sin=0/cos=1)

use bitnet_rope::{DEFAULT_ROPE_BASE, RopeTableError, build_tables, resolve_base};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#[test]
fn default_rope_base_is_10000() {
    assert!((DEFAULT_ROPE_BASE - 10_000.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// resolve_base
// ---------------------------------------------------------------------------

#[test]
fn resolve_base_none_returns_default() {
    let base = resolve_base(None);
    assert!((base - DEFAULT_ROPE_BASE).abs() < 1e-6);
}

#[test]
fn resolve_base_some_returns_value() {
    let base = resolve_base(Some(500_000.0));
    assert!((base - 500_000.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// build_tables — valid cases
// ---------------------------------------------------------------------------

#[test]
fn build_tables_dim_2_seq_1() {
    let tables = build_tables(2, 1, DEFAULT_ROPE_BASE).unwrap();
    assert_eq!(tables.half_dim, 1);
    assert_eq!(tables.sin.len(), 1); // 1 * 1
    assert_eq!(tables.cos.len(), 1);
}

#[test]
fn build_tables_dim_64_seq_128() {
    let tables = build_tables(64, 128, DEFAULT_ROPE_BASE).unwrap();
    assert_eq!(tables.half_dim, 32);
    assert_eq!(tables.sin.len(), 128 * 32);
    assert_eq!(tables.cos.len(), 128 * 32);
}

#[test]
fn build_tables_dim_128_seq_4096() {
    let tables = build_tables(128, 4096, DEFAULT_ROPE_BASE).unwrap();
    assert_eq!(tables.half_dim, 64);
    assert_eq!(tables.sin.len(), 4096 * 64);
    assert_eq!(tables.cos.len(), 4096 * 64);
}

#[test]
fn build_tables_large_seq_16k() {
    let tables = build_tables(128, 16384, DEFAULT_ROPE_BASE).unwrap();
    assert_eq!(tables.sin.len(), 16384 * 64);
}

#[test]
fn build_tables_seq_0_returns_empty() {
    let tables = build_tables(4, 0, DEFAULT_ROPE_BASE).unwrap();
    assert_eq!(tables.half_dim, 2);
    assert!(tables.sin.is_empty());
    assert!(tables.cos.is_empty());
}

#[test]
fn build_tables_custom_base() {
    let tables = build_tables(4, 10, 500_000.0).unwrap();
    assert_eq!(tables.half_dim, 2);
    assert_eq!(tables.sin.len(), 20); // 10 * 2
}

// ---------------------------------------------------------------------------
// build_tables — numerical properties
// ---------------------------------------------------------------------------

#[test]
fn sin_cos_values_bounded() {
    let tables = build_tables(64, 128, DEFAULT_ROPE_BASE).unwrap();
    for &v in &tables.sin {
        assert!(v >= -1.0 && v <= 1.0, "sin value out of range: {}", v);
    }
    for &v in &tables.cos {
        assert!(v >= -1.0 && v <= 1.0, "cos value out of range: {}", v);
    }
}

#[test]
fn position_0_sin_is_zero_cos_is_one() {
    let tables = build_tables(8, 4, DEFAULT_ROPE_BASE).unwrap();
    // Position 0: angle = 0 * freq = 0 for all freqs → sin=0, cos=1
    let half_dim = tables.half_dim;
    for d in 0..half_dim {
        assert!(tables.sin[d].abs() < 1e-6, "sin[0][{}] should be ~0, got {}", d, tables.sin[d]);
        assert!(
            (tables.cos[d] - 1.0).abs() < 1e-6,
            "cos[0][{}] should be ~1, got {}",
            d,
            tables.cos[d]
        );
    }
}

#[test]
fn sin_squared_plus_cos_squared_approx_one() {
    let tables = build_tables(8, 16, DEFAULT_ROPE_BASE).unwrap();
    for i in 0..tables.sin.len() {
        let s2_c2 = tables.sin[i] * tables.sin[i] + tables.cos[i] * tables.cos[i];
        assert!((s2_c2 - 1.0).abs() < 1e-5, "sin²+cos² at index {} = {}, expected ~1.0", i, s2_c2);
    }
}

// ---------------------------------------------------------------------------
// build_tables — error cases
// ---------------------------------------------------------------------------

#[test]
fn build_tables_zero_dim() {
    let err = build_tables(0, 10, DEFAULT_ROPE_BASE).unwrap_err();
    assert_eq!(err, RopeTableError::ZeroDimension);
}

#[test]
fn build_tables_odd_dim() {
    let err = build_tables(3, 10, DEFAULT_ROPE_BASE).unwrap_err();
    assert_eq!(err, RopeTableError::OddDimension { dim: 3 });
}

#[test]
fn build_tables_odd_dim_5() {
    let err = build_tables(5, 10, DEFAULT_ROPE_BASE).unwrap_err();
    assert_eq!(err, RopeTableError::OddDimension { dim: 5 });
}

#[test]
fn build_tables_nan_base() {
    let err = build_tables(4, 10, f32::NAN).unwrap_err();
    assert_eq!(err, RopeTableError::NonFiniteBase { base: f32::NAN });
}

#[test]
fn build_tables_infinity_base() {
    let err = build_tables(4, 10, f32::INFINITY).unwrap_err();
    assert_eq!(err, RopeTableError::NonFiniteBase { base: f32::INFINITY });
}

#[test]
fn build_tables_neg_infinity_base() {
    let err = build_tables(4, 10, f32::NEG_INFINITY).unwrap_err();
    match err {
        RopeTableError::NonFiniteBase { .. } => {} // expected
        other => panic!("expected NonFiniteBase, got: {:?}", other),
    }
}

#[test]
fn build_tables_zero_base() {
    let err = build_tables(4, 10, 0.0).unwrap_err();
    assert_eq!(err, RopeTableError::NonPositiveBase { base: 0.0 });
}

#[test]
fn build_tables_negative_base() {
    let err = build_tables(4, 10, -1.0).unwrap_err();
    assert_eq!(err, RopeTableError::NonPositiveBase { base: -1.0 });
}

// ---------------------------------------------------------------------------
// RopeTableError — Display and equality
// ---------------------------------------------------------------------------

#[test]
fn error_display_zero_dim() {
    let err = RopeTableError::ZeroDimension;
    let msg = format!("{}", err);
    assert!(msg.contains("greater than zero"));
}

#[test]
fn error_display_odd_dim() {
    let err = RopeTableError::OddDimension { dim: 7 };
    let msg = format!("{}", err);
    assert!(msg.contains("even") && msg.contains("7"));
}

#[test]
fn error_display_non_finite() {
    let err = RopeTableError::NonFiniteBase { base: f32::INFINITY };
    let msg = format!("{}", err);
    assert!(msg.contains("finite"));
}

#[test]
fn error_display_non_positive() {
    let err = RopeTableError::NonPositiveBase { base: -5.0 };
    let msg = format!("{}", err);
    assert!(msg.contains("greater than zero") && msg.contains("-5"));
}

#[test]
fn error_debug() {
    let err = RopeTableError::ZeroDimension;
    let dbg = format!("{:?}", err);
    assert!(dbg.contains("ZeroDimension"));
}

#[test]
fn error_clone() {
    let err = RopeTableError::OddDimension { dim: 3 };
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

#[test]
fn error_is_std_error() {
    let err = RopeTableError::ZeroDimension;
    let _: &dyn std::error::Error = &err;
}

// ---------------------------------------------------------------------------
// RopeTables — struct invariants
// ---------------------------------------------------------------------------

#[test]
fn rope_tables_debug() {
    let tables = build_tables(4, 2, DEFAULT_ROPE_BASE).unwrap();
    let dbg = format!("{:?}", tables);
    assert!(dbg.contains("half_dim"));
}

#[test]
fn rope_tables_clone() {
    let tables = build_tables(4, 2, DEFAULT_ROPE_BASE).unwrap();
    let cloned = tables.clone();
    assert_eq!(cloned.half_dim, tables.half_dim);
    assert_eq!(cloned.sin, tables.sin);
    assert_eq!(cloned.cos, tables.cos);
}
