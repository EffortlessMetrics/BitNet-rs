//! Snapshot tests for `bitnet-rope` public API surface.
//!
//! Pins error message formats and RoPE table dimensions to catch regressions
//! in the RoPE cache generation logic.

use bitnet_rope::{DEFAULT_ROPE_BASE, RopeTableError, build_tables, resolve_base};

#[test]
fn default_rope_base_snapshot() {
    insta::assert_snapshot!("default_rope_base", DEFAULT_ROPE_BASE.to_string());
}

#[test]
fn resolve_base_none_returns_default() {
    let resolved = resolve_base(None);
    insta::assert_snapshot!("resolve_base_none", resolved.to_string());
}

#[test]
fn rope_table_error_zero_dimension_display() {
    let err = RopeTableError::ZeroDimension;
    insta::assert_snapshot!("error_zero_dimension", err.to_string());
}

#[test]
fn rope_table_error_odd_dimension_display() {
    let err = RopeTableError::OddDimension { dim: 7 };
    insta::assert_snapshot!("error_odd_dimension", err.to_string());
}

#[test]
fn rope_table_error_non_positive_base_display() {
    let err = RopeTableError::NonPositiveBase { base: -1.0 };
    insta::assert_snapshot!("error_non_positive_base", err.to_string());
}

#[test]
fn rope_tables_small_shape_snapshot() {
    let tables = build_tables(4, 8, DEFAULT_ROPE_BASE).expect("4-dim, 8-seq should succeed");
    let summary = format!(
        "half_dim={} sin_len={} cos_len={}",
        tables.half_dim,
        tables.sin.len(),
        tables.cos.len()
    );
    insta::assert_snapshot!("rope_tables_small_shape", summary);
}
