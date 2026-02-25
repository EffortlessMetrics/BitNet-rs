//! Shared rotary position embedding (RoPE) cache generation.
//!
//! This crate owns only table generation for RoPE sine/cosine caches so model
//! crates can reuse one implementation and keep attention modules focused on
//! tensor-shape and execution concerns.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// Default RoPE base/theta used by LLaMA-style models.
pub const DEFAULT_ROPE_BASE: f32 = 10_000.0;

/// Generated RoPE lookup tables.
#[derive(Debug, Clone)]
pub struct RopeTables {
    /// Half of the full head dimension (`dim / 2`).
    pub half_dim: usize,
    /// Flattened sine table in row-major `[max_seq_len, half_dim]`.
    pub sin: Vec<f32>,
    /// Flattened cosine table in row-major `[max_seq_len, half_dim]`.
    pub cos: Vec<f32>,
}

/// RoPE cache generation failures.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeTableError {
    ZeroDimension,
    OddDimension { dim: usize },
    NonFiniteBase { base: f32 },
    NonPositiveBase { base: f32 },
}

impl Display for RopeTableError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "RoPE dimension must be greater than zero"),
            Self::OddDimension { dim } => {
                write!(f, "RoPE dimension must be even, got {dim}")
            }
            Self::NonFiniteBase { base } => {
                write!(f, "RoPE base must be finite, got {base}")
            }
            Self::NonPositiveBase { base } => {
                write!(f, "RoPE base must be greater than zero, got {base}")
            }
        }
    }
}

impl Error for RopeTableError {}

/// Resolve an optional RoPE base to a concrete value.
#[must_use]
pub fn resolve_base(base: Option<f32>) -> f32 {
    base.unwrap_or(DEFAULT_ROPE_BASE)
}

/// Build flattened sine/cosine RoPE tables for `[max_seq_len, dim / 2]`.
pub fn build_tables(
    dim: usize,
    max_seq_len: usize,
    base: f32,
) -> Result<RopeTables, RopeTableError> {
    if dim == 0 {
        return Err(RopeTableError::ZeroDimension);
    }
    if !dim.is_multiple_of(2) {
        return Err(RopeTableError::OddDimension { dim });
    }
    if !base.is_finite() {
        return Err(RopeTableError::NonFiniteBase { base });
    }
    if base <= 0.0 {
        return Err(RopeTableError::NonPositiveBase { base });
    }

    let half_dim = dim / 2;
    let inv_freq =
        (0..half_dim).map(|i| 1.0 / base.powf((2.0 * i as f32) / dim as f32)).collect::<Vec<_>>();

    let mut sin = Vec::with_capacity(max_seq_len * half_dim);
    let mut cos = Vec::with_capacity(max_seq_len * half_dim);

    for pos in 0..max_seq_len {
        let pos = pos as f32;
        for &freq in &inv_freq {
            let angle = pos * freq;
            sin.push(angle.sin());
            cos.push(angle.cos());
        }
    }

    Ok(RopeTables { half_dim, sin, cos })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(lhs: f32, rhs: f32, tol: f32) {
        assert!(
            (lhs - rhs).abs() <= tol,
            "expected {lhs} ~= {rhs} (tol={tol}), diff={}",
            (lhs - rhs).abs()
        );
    }

    #[test]
    fn resolve_base_uses_default_when_missing() {
        assert_eq!(resolve_base(None), DEFAULT_ROPE_BASE);
        assert_eq!(resolve_base(Some(5_000.0)), 5_000.0);
    }

    #[test]
    fn build_tables_has_expected_shape() {
        let tables = build_tables(8, 3, DEFAULT_ROPE_BASE).expect("tables");
        assert_eq!(tables.half_dim, 4);
        assert_eq!(tables.sin.len(), 12);
        assert_eq!(tables.cos.len(), 12);
    }

    #[test]
    fn position_zero_is_identity_row() {
        let tables = build_tables(8, 2, DEFAULT_ROPE_BASE).expect("tables");
        let row0 = &tables.sin[..tables.half_dim];
        for &v in row0 {
            approx_eq(v, 0.0, 1e-7);
        }

        let row0_cos = &tables.cos[..tables.half_dim];
        for &v in row0_cos {
            approx_eq(v, 1.0, 1e-7);
        }
    }

    #[test]
    fn build_tables_matches_known_values() {
        let tables = build_tables(4, 2, DEFAULT_ROPE_BASE).expect("tables");
        let row1_offset = tables.half_dim;

        approx_eq(tables.sin[row1_offset], 1.0_f32.sin(), 1e-6);
        approx_eq(tables.cos[row1_offset], 1.0_f32.cos(), 1e-6);

        approx_eq(tables.sin[row1_offset + 1], 0.01_f32.sin(), 1e-6);
        approx_eq(tables.cos[row1_offset + 1], 0.01_f32.cos(), 1e-6);
    }

    #[test]
    fn rejects_invalid_dimension_and_base() {
        assert!(matches!(
            build_tables(0, 1, DEFAULT_ROPE_BASE),
            Err(RopeTableError::ZeroDimension)
        ));
        assert!(matches!(
            build_tables(3, 1, DEFAULT_ROPE_BASE),
            Err(RopeTableError::OddDimension { dim: 3 })
        ));
        assert!(matches!(
            build_tables(4, 1, f32::INFINITY),
            Err(RopeTableError::NonFiniteBase { .. })
        ));
        assert!(matches!(build_tables(4, 1, 0.0), Err(RopeTableError::NonPositiveBase { .. })));
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // build_tables with valid even dim and positive base always succeeds.
    proptest! {
        #[test]
        fn build_tables_valid_inputs_succeed(
            dim in (1u32..=32u32).prop_map(|d| d * 2), // even dims 2..=64
            seq_len in 1usize..=64,
            base in 100f32..=1_000_000f32,
        ) {
            let result = build_tables(dim as usize, seq_len, base);
            prop_assert!(result.is_ok(), "expected Ok for dim={} seq_len={} base={}", dim, seq_len, base);
        }
    }

    // Resulting tables have the expected shape: len == seq_len * half_dim.
    proptest! {
        #[test]
        fn build_tables_shape_invariant(
            dim in (1u32..=16u32).prop_map(|d| d * 2),
            seq_len in 1usize..=32,
        ) {
            let base = DEFAULT_ROPE_BASE;
            let tables = build_tables(dim as usize, seq_len, base).unwrap();
            let expected_half = dim as usize / 2;
            prop_assert_eq!(tables.half_dim, expected_half);
            prop_assert_eq!(tables.sin.len(), seq_len * expected_half);
            prop_assert_eq!(tables.cos.len(), seq_len * expected_half);
        }
    }

    // sin^2 + cos^2 ≈ 1 for every corresponding pair.
    proptest! {
        #[test]
        fn build_tables_trig_identity(
            dim in (1u32..=8u32).prop_map(|d| d * 2),
            seq_len in 1usize..=16,
        ) {
            let tables = build_tables(dim as usize, seq_len, DEFAULT_ROPE_BASE).unwrap();
            for (s, c) in tables.sin.iter().zip(&tables.cos) {
                let norm = s * s + c * c;
                prop_assert!(
                    (norm - 1.0).abs() < 1e-5,
                    "sin²+cos²={} != 1.0 for sin={} cos={}",
                    norm, s, c
                );
            }
        }
    }

    // Odd dimensions and non-positive bases are rejected.
    proptest! {
        #[test]
        fn build_tables_rejects_odd_dim(dim in (1u32..=32u32).prop_map(|d| d * 2 - 1)) {
            let result = build_tables(dim as usize, 1, DEFAULT_ROPE_BASE);
            prop_assert!(result.is_err(), "expected Err for odd dim={}", dim);
        }
    }
}
