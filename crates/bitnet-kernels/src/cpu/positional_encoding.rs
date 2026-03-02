//! CPU positional encoding kernels.
//!
//! Provides a unified collection of position-encoding algorithms used in
//! Transformer-family architectures:
//!
//! - **Sinusoidal PE** — classic fixed encoding from "Attention Is All You Need"
//! - **RoPE** — Rotary Position Embedding with interleaved and NeoX layouts
//! - **ALiBi** — Attention with Linear Biases (no learned parameters)
//! - **Relative position bias** — learnable bias lookup for relative distances
//! - **Learnable positional embedding** — table lookup by absolute position

use std::fmt;

// ── Error type ──────────────────────────────────────────────────────

/// Errors produced by positional encoding operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PositionalEncodingError {
    /// A dimension parameter was zero when a positive value is required.
    ZeroDimension { name: &'static str },
    /// Two buffers/parameters that must agree in size do not.
    DimensionMismatch { expected: usize, actual: usize, context: &'static str },
    /// Head dimension must be even for rotation-based encodings.
    OddHeadDimension { head_dim: usize },
    /// Number of attention heads must be a positive power of two for ALiBi.
    InvalidHeadCount { num_heads: usize },
    /// A position index exceeds the table that was pre-computed.
    PositionOutOfRange { position: usize, max_len: usize },
}

impl fmt::Display for PositionalEncodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension { name } => {
                write!(f, "positional encoding: {name} must be non-zero")
            }
            Self::DimensionMismatch { expected, actual, context } => {
                write!(f, "positional encoding: {context}: expected {expected}, got {actual}")
            }
            Self::OddHeadDimension { head_dim } => {
                write!(f, "positional encoding: head_dim must be even, got {head_dim}")
            }
            Self::InvalidHeadCount { num_heads } => {
                write!(
                    f,
                    "positional encoding: num_heads must be a positive power of two, got \
                     {num_heads}"
                )
            }
            Self::PositionOutOfRange { position, max_len } => {
                write!(
                    f,
                    "positional encoding: position {position} out of range (max_len={max_len})"
                )
            }
        }
    }
}

impl std::error::Error for PositionalEncodingError {}

/// Convenience alias.
type Result<T> = std::result::Result<T, PositionalEncodingError>;

// ── Sinusoidal positional encoding ──────────────────────────────────

/// Generate a sinusoidal positional-encoding matrix.
///
/// Returns a flat `[seq_len, model_dim]` buffer where:
/// - even columns contain `sin(pos / base^(2i/d))`
/// - odd columns contain `cos(pos / base^(2i/d))`
///
/// This is the classic encoding from *Attention Is All You Need*.
pub fn sinusoidal_pe(seq_len: usize, model_dim: usize, base: f32) -> Result<Vec<f32>> {
    if model_dim == 0 {
        return Err(PositionalEncodingError::ZeroDimension { name: "model_dim" });
    }
    let d = model_dim as f32;
    let mut output = vec![0.0f32; seq_len * model_dim];
    for pos in 0..seq_len {
        let row = pos * model_dim;
        for i in 0..model_dim {
            let dim_pair = (i / 2) as f32;
            let angle = (pos as f32) / base.powf(2.0 * dim_pair / d);
            output[row + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    Ok(output)
}

// ── RoPE helpers ────────────────────────────────────────────────────

/// Build separate sin and cos tables for RoPE.
///
/// Returns `(sin_table, cos_table)` each of shape `[max_seq_len, head_dim/2]`
/// stored as flat vectors.
///
/// `theta_i = base^(-2i / head_dim)`, angle = `pos * theta_i`.
pub fn build_sin_cos_tables(
    max_seq_len: usize,
    head_dim: usize,
    base: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if head_dim == 0 {
        return Err(PositionalEncodingError::ZeroDimension { name: "head_dim" });
    }
    if !head_dim.is_multiple_of(2) {
        return Err(PositionalEncodingError::OddHeadDimension { head_dim });
    }
    let half = head_dim / 2;
    let total = max_seq_len * half;
    let mut sin_table = Vec::with_capacity(total);
    let mut cos_table = Vec::with_capacity(total);

    for pos in 0..max_seq_len {
        for i in 0..half {
            let theta = base.powf(-(2.0 * i as f32) / head_dim as f32);
            let angle = pos as f32 * theta;
            sin_table.push(angle.sin());
            cos_table.push(angle.cos());
        }
    }
    Ok((sin_table, cos_table))
}

/// Apply RoPE (interleaved layout) to a single head vector in-place.
///
/// Interleaved layout: pairs are `(x[2i], x[2i+1])`.
///
/// `sin_row` / `cos_row` must each have length ≥ `head_dim / 2` and
/// correspond to the given position.
pub fn apply_rope_pe(
    data: &mut [f32],
    head_dim: usize,
    sin_row: &[f32],
    cos_row: &[f32],
) -> Result<()> {
    if head_dim == 0 {
        return Err(PositionalEncodingError::ZeroDimension { name: "head_dim" });
    }
    if !head_dim.is_multiple_of(2) {
        return Err(PositionalEncodingError::OddHeadDimension { head_dim });
    }
    let half = head_dim / 2;
    if data.len() < head_dim {
        return Err(PositionalEncodingError::DimensionMismatch {
            expected: head_dim,
            actual: data.len(),
            context: "data length < head_dim",
        });
    }
    if sin_row.len() < half || cos_row.len() < half {
        return Err(PositionalEncodingError::DimensionMismatch {
            expected: half,
            actual: sin_row.len().min(cos_row.len()),
            context: "sin/cos row length < head_dim/2",
        });
    }
    for i in 0..half {
        let x0 = data[2 * i];
        let x1 = data[2 * i + 1];
        data[2 * i] = x0 * cos_row[i] - x1 * sin_row[i];
        data[2 * i + 1] = x0 * sin_row[i] + x1 * cos_row[i];
    }
    Ok(())
}

/// Apply RoPE in the *NeoX* (split-half) layout in-place.
///
/// NeoX layout: the first half of the head vector pairs with the second half,
/// i.e. pair `i` is `(x[i], x[i + head_dim/2])`.
pub fn apply_rope_neox(
    data: &mut [f32],
    head_dim: usize,
    sin_row: &[f32],
    cos_row: &[f32],
) -> Result<()> {
    if head_dim == 0 {
        return Err(PositionalEncodingError::ZeroDimension { name: "head_dim" });
    }
    if !head_dim.is_multiple_of(2) {
        return Err(PositionalEncodingError::OddHeadDimension { head_dim });
    }
    let half = head_dim / 2;
    if data.len() < head_dim {
        return Err(PositionalEncodingError::DimensionMismatch {
            expected: head_dim,
            actual: data.len(),
            context: "data length < head_dim",
        });
    }
    if sin_row.len() < half || cos_row.len() < half {
        return Err(PositionalEncodingError::DimensionMismatch {
            expected: half,
            actual: sin_row.len().min(cos_row.len()),
            context: "sin/cos row length < head_dim/2",
        });
    }
    for i in 0..half {
        let x0 = data[i];
        let x1 = data[i + half];
        data[i] = x0 * cos_row[i] - x1 * sin_row[i];
        data[i + half] = x0 * sin_row[i] + x1 * cos_row[i];
    }
    Ok(())
}

// ── ALiBi ───────────────────────────────────────────────────────────

/// Compute ALiBi slopes for `num_heads` attention heads.
///
/// Slopes are `2^(-8 * i / num_heads)` for `i` in `1..=num_heads`,
/// matching the ALiBi paper. `num_heads` must be a positive power of two.
pub fn compute_alibi_slopes(num_heads: usize) -> Result<Vec<f32>> {
    if num_heads == 0 || !num_heads.is_power_of_two() {
        return Err(PositionalEncodingError::InvalidHeadCount { num_heads });
    }
    let ratio = 2.0f32.powf(-8.0 / num_heads as f32);
    let slopes: Vec<f32> = (0..num_heads).map(|i| ratio.powi(i as i32 + 1)).collect();
    Ok(slopes)
}

/// Build an ALiBi bias matrix of shape `[num_heads, seq_len, seq_len]` (flat).
///
/// `bias[h, i, j] = -slopes[h] * |i - j|` — a causal-friendly linear
/// distance penalty.  Future positions (`j > i`) are set to
/// `f32::NEG_INFINITY` to enforce causal masking.
pub fn apply_alibi_bias(slopes: &[f32], seq_len: usize) -> Result<Vec<f32>> {
    let num_heads = slopes.len();
    if num_heads == 0 {
        return Err(PositionalEncodingError::ZeroDimension { name: "num_heads (slopes.len)" });
    }
    let mut bias = vec![0.0f32; num_heads * seq_len * seq_len];
    for (h, &slope) in slopes.iter().enumerate() {
        let head_offset = h * seq_len * seq_len;
        for i in 0..seq_len {
            for j in 0..seq_len {
                let idx = head_offset + i * seq_len + j;
                if j > i {
                    bias[idx] = f32::NEG_INFINITY;
                } else {
                    bias[idx] = -slope * (i - j) as f32;
                }
            }
        }
    }
    Ok(bias)
}

// ── Relative position bias ──────────────────────────────────────────

/// Compute relative position bias indices for a `[seq_len, seq_len]` window.
///
/// Returns a flat `[seq_len, seq_len]` matrix of bucket indices in
/// `[0, num_buckets)` suitable for indexing into a learnable bias table.
///
/// Follows the T5-style bucketing:
/// - Half the buckets are for exact distances, the other half for log-spaced.
/// - Negative (causal) distances are mapped to the second half of buckets.
pub fn relative_position_bias(
    seq_len: usize,
    num_buckets: usize,
    max_distance: usize,
) -> Result<Vec<usize>> {
    if num_buckets == 0 {
        return Err(PositionalEncodingError::ZeroDimension { name: "num_buckets" });
    }
    if max_distance == 0 {
        return Err(PositionalEncodingError::ZeroDimension { name: "max_distance" });
    }
    let mut indices = vec![0usize; seq_len * seq_len];
    let half = num_buckets / 2;
    let max_exact = half / 2;

    for i in 0..seq_len {
        for j in 0..seq_len {
            let rel = j as isize - i as isize;
            let (is_neg, abs_rel) =
                if rel < 0 { (true, (-rel) as usize) } else { (false, rel as usize) };

            let bucket = if max_exact == 0 || abs_rel < max_exact {
                abs_rel
            } else {
                let log_ratio = (abs_rel as f32 / max_exact as f32).ln()
                    / (max_distance as f32 / max_exact as f32).ln();
                let b = max_exact as f32 + log_ratio * (half - max_exact) as f32;
                b.min((half - 1) as f32) as usize
            };

            let final_bucket = if is_neg { bucket + half } else { bucket };
            indices[i * seq_len + j] = final_bucket.min(num_buckets - 1);
        }
    }
    Ok(indices)
}

// ── Learnable positional embedding lookup ───────────────────────────

/// Look up learnable positional embeddings by position index.
///
/// `table` is a flat `[max_len, embed_dim]` embedding table.
/// `positions` are the position indices to retrieve.
///
/// Returns a flat `[positions.len(), embed_dim]` buffer.
pub fn learnable_pe_lookup(
    table: &[f32],
    max_len: usize,
    embed_dim: usize,
    positions: &[usize],
) -> Result<Vec<f32>> {
    if embed_dim == 0 {
        return Err(PositionalEncodingError::ZeroDimension { name: "embed_dim" });
    }
    if table.len() < max_len * embed_dim {
        return Err(PositionalEncodingError::DimensionMismatch {
            expected: max_len * embed_dim,
            actual: table.len(),
            context: "table too small for max_len * embed_dim",
        });
    }
    let mut output = Vec::with_capacity(positions.len() * embed_dim);
    for &pos in positions {
        if pos >= max_len {
            return Err(PositionalEncodingError::PositionOutOfRange { position: pos, max_len });
        }
        let start = pos * embed_dim;
        output.extend_from_slice(&table[start..start + embed_dim]);
    }
    Ok(output)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::too_many_lines)]
mod tests {
    use super::*;

    // ── Sinusoidal PE ───────────────────────────────────────────────

    #[test]
    fn test_sinusoidal_pe_basic_shape() {
        let pe = sinusoidal_pe(4, 8, 10_000.0).unwrap();
        assert_eq!(pe.len(), 4 * 8);
    }

    #[test]
    fn test_sinusoidal_pe_values_position_zero() {
        let pe = sinusoidal_pe(1, 4, 10_000.0).unwrap();
        // pos=0 → angle=0 → sin(0)=0, cos(0)=1
        assert!((pe[0] - 0.0).abs() < 1e-6, "sin(0) should be 0");
        assert!((pe[1] - 1.0).abs() < 1e-6, "cos(0) should be 1");
        assert!((pe[2] - 0.0).abs() < 1e-6, "sin(0) should be 0");
        assert!((pe[3] - 1.0).abs() < 1e-6, "cos(0) should be 1");
    }

    #[test]
    fn test_sinusoidal_pe_known_values() {
        // pos=1, dim=4, base=10000
        // dim_pair 0: angle = 1/10000^0 = 1.0 → sin(1), cos(1)
        // dim_pair 1: angle = 1/10000^(2/4) = 1/100 → sin(0.01), cos(0.01)
        let pe = sinusoidal_pe(2, 4, 10_000.0).unwrap();
        let row1 = &pe[4..8]; // position 1
        assert!((row1[0] - 1.0f32.sin()).abs() < 1e-5);
        assert!((row1[1] - 1.0f32.cos()).abs() < 1e-5);
        assert!((row1[2] - 0.01f32.sin()).abs() < 1e-5);
        assert!((row1[3] - 0.01f32.cos()).abs() < 1e-5);
    }

    #[test]
    fn test_sinusoidal_pe_orthogonality() {
        let pe = sinusoidal_pe(10, 64, 10_000.0).unwrap();
        for i in 0..10 {
            for j in (i + 1)..10 {
                let a = &pe[i * 64..(i + 1) * 64];
                let b = &pe[j * 64..(j + 1) * 64];
                let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                let cos_sim = dot / (norm_a * norm_b + 1e-12);
                assert!(cos_sim < 0.999, "positions {i} and {j} too similar: cos_sim={cos_sim}");
            }
        }
    }

    #[test]
    fn test_sinusoidal_pe_bounded_values() {
        let pe = sinusoidal_pe(100, 32, 10_000.0).unwrap();
        for &v in &pe {
            assert!((-1.0..=1.0).contains(&v), "PE value {v} out of [-1,1]");
        }
    }

    #[test]
    fn test_sinusoidal_pe_deterministic() {
        let a = sinusoidal_pe(5, 16, 10_000.0).unwrap();
        let b = sinusoidal_pe(5, 16, 10_000.0).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_sinusoidal_pe_different_bases() {
        let a = sinusoidal_pe(4, 8, 10_000.0).unwrap();
        let b = sinusoidal_pe(4, 8, 1_000.0).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn test_sinusoidal_pe_zero_model_dim_error() {
        let err = sinusoidal_pe(5, 0, 10_000.0).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::ZeroDimension { name: "model_dim" }));
    }

    #[test]
    fn test_sinusoidal_pe_zero_seq_len() {
        let pe = sinusoidal_pe(0, 8, 10_000.0).unwrap();
        assert!(pe.is_empty());
    }

    #[test]
    fn test_sinusoidal_pe_single_position() {
        let pe = sinusoidal_pe(1, 8, 10_000.0).unwrap();
        assert_eq!(pe.len(), 8);
        assert!(pe.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_sinusoidal_pe_long_sequence() {
        let pe = sinusoidal_pe(4096, 4, 10_000.0).unwrap();
        assert_eq!(pe.len(), 4096 * 4);
        assert!(pe.iter().all(|v| v.is_finite()));
    }

    // ── RoPE: build_sin_cos_tables ──────────────────────────────────

    #[test]
    fn test_sin_cos_tables_shape() {
        let (sin_t, cos_t) = build_sin_cos_tables(16, 8, 10_000.0).unwrap();
        let expected = 16 * (8 / 2);
        assert_eq!(sin_t.len(), expected);
        assert_eq!(cos_t.len(), expected);
    }

    #[test]
    fn test_sin_cos_tables_position_zero() {
        let (sin_t, cos_t) = build_sin_cos_tables(1, 4, 10_000.0).unwrap();
        for &s in &sin_t {
            assert!((s - 0.0).abs() < 1e-6, "sin at pos 0 should be 0, got {s}");
        }
        for &c in &cos_t {
            assert!((c - 1.0).abs() < 1e-6, "cos at pos 0 should be 1, got {c}");
        }
    }

    #[test]
    fn test_sin_cos_tables_zero_head_dim() {
        let err = build_sin_cos_tables(8, 0, 10_000.0).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::ZeroDimension { .. }));
    }

    #[test]
    fn test_sin_cos_tables_odd_head_dim() {
        let err = build_sin_cos_tables(8, 5, 10_000.0).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::OddHeadDimension { head_dim: 5 }));
    }

    #[test]
    fn test_sin_cos_tables_frequency_decay() {
        let (sin_t, _cos_t) = build_sin_cos_tables(2, 8, 10_000.0).unwrap();
        let half = 4;
        let row1 = &sin_t[half..2 * half]; // position 1
        assert!(
            row1[0].abs() > row1[3].abs(),
            "frequency should decay: |{}| > |{}|",
            row1[0],
            row1[3]
        );
    }

    // ── RoPE: apply_rope_pe (interleaved) ───────────────────────────

    #[test]
    fn test_rope_identity_at_position_zero() {
        let (sin_t, cos_t) = build_sin_cos_tables(1, 4, 10_000.0).unwrap();
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let original = data.clone();
        apply_rope_pe(&mut data, 4, &sin_t[..2], &cos_t[..2]).unwrap();
        for (o, d) in original.iter().zip(data.iter()) {
            assert!((o - d).abs() < 1e-6, "pos 0 should be identity: {o} vs {d}");
        }
    }

    #[test]
    fn test_rope_preserves_norm() {
        let (sin_t, cos_t) = build_sin_cos_tables(32, 8, 10_000.0).unwrap();
        let half = 4;
        for pos in [1, 5, 17, 31] {
            let mut data: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.3).collect();
            let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            apply_rope_pe(
                &mut data,
                8,
                &sin_t[pos * half..(pos + 1) * half],
                &cos_t[pos * half..(pos + 1) * half],
            )
            .unwrap();
            let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm_before - norm_after).abs() < 1e-4,
                "norm not preserved at pos={pos}: {norm_before} vs {norm_after}"
            );
        }
    }

    #[test]
    fn test_rope_different_positions_differ() {
        let (sin_t, cos_t) = build_sin_cos_tables(4, 4, 10_000.0).unwrap();
        let half = 2;
        let original = vec![1.0, 2.0, 3.0, 4.0];

        let mut a = original.clone();
        apply_rope_pe(&mut a, 4, &sin_t[half..2 * half], &cos_t[half..2 * half]).unwrap();

        let mut b = original.clone();
        apply_rope_pe(&mut b, 4, &sin_t[2 * half..3 * half], &cos_t[2 * half..3 * half]).unwrap();

        assert_ne!(a, b, "different positions should produce different results");
    }

    #[test]
    fn test_rope_known_reference() {
        // head_dim=2, pos=1, base=10000 → theta=1.0, angle=1.0
        let (sin_t, cos_t) = build_sin_cos_tables(2, 2, 10_000.0).unwrap();
        let mut data = vec![1.0, 0.0];
        apply_rope_pe(&mut data, 2, &sin_t[1..2], &cos_t[1..2]).unwrap();
        assert!((data[0] - 1.0f32.cos()).abs() < 1e-5);
        assert!((data[1] - 1.0f32.sin()).abs() < 1e-5);
    }

    #[test]
    fn test_rope_data_too_short_error() {
        let (sin_t, cos_t) = build_sin_cos_tables(1, 4, 10_000.0).unwrap();
        let mut data = vec![1.0, 2.0]; // length 2, head_dim=4
        let err = apply_rope_pe(&mut data, 4, &sin_t[..2], &cos_t[..2]).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_rope_odd_head_dim_error() {
        let err = apply_rope_pe(&mut [0.0; 3], 3, &[0.0], &[0.0]).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::OddHeadDimension { head_dim: 3 }));
    }

    // ── RoPE: apply_rope_neox ───────────────────────────────────────

    #[test]
    fn test_neox_identity_at_position_zero() {
        let (sin_t, cos_t) = build_sin_cos_tables(1, 4, 10_000.0).unwrap();
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let original = data.clone();
        apply_rope_neox(&mut data, 4, &sin_t[..2], &cos_t[..2]).unwrap();
        for (o, d) in original.iter().zip(data.iter()) {
            assert!((o - d).abs() < 1e-6, "neox pos 0 should be identity: {o} vs {d}");
        }
    }

    #[test]
    fn test_neox_preserves_norm() {
        let (sin_t, cos_t) = build_sin_cos_tables(8, 8, 10_000.0).unwrap();
        let half = 4;
        let mut data: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope_neox(&mut data, 8, &sin_t[3 * half..4 * half], &cos_t[3 * half..4 * half])
            .unwrap();
        let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "neox norm: {norm_before} vs {norm_after}"
        );
    }

    #[test]
    fn test_neox_vs_interleaved_differ() {
        let (sin_t, cos_t) = build_sin_cos_tables(2, 4, 10_000.0).unwrap();
        let half = 2;
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let mut inter = input.clone();
        apply_rope_pe(&mut inter, 4, &sin_t[half..2 * half], &cos_t[half..2 * half]).unwrap();

        let mut neox = input.clone();
        apply_rope_neox(&mut neox, 4, &sin_t[half..2 * half], &cos_t[half..2 * half]).unwrap();

        assert_ne!(inter, neox, "interleaved and neox should give different results");
    }

    #[test]
    fn test_neox_known_reference() {
        // head_dim=4, pos=1: pairs are (x[0],x[2]) and (x[1],x[3])
        let (sin_t, cos_t) = build_sin_cos_tables(2, 4, 10_000.0).unwrap();
        let half = 2;
        let sin_row = &sin_t[half..2 * half];
        let cos_row = &cos_t[half..2 * half];

        let mut data = vec![1.0, 2.0, 0.5, -0.5];
        apply_rope_neox(&mut data, 4, sin_row, cos_row).unwrap();

        // pair 0: (data[0], data[2]) rotated by (cos_row[0], sin_row[0])
        let x0 = 1.0f32;
        let x2 = 0.5f32;
        let expected_0 = x0 * cos_row[0] - x2 * sin_row[0];
        let expected_2 = x0 * sin_row[0] + x2 * cos_row[0];
        assert!((data[0] - expected_0).abs() < 1e-5);
        assert!((data[2] - expected_2).abs() < 1e-5);
    }

    #[test]
    fn test_neox_data_too_short_error() {
        let err = apply_rope_neox(&mut [0.0; 2], 4, &[0.0; 2], &[0.0; 2]).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::DimensionMismatch { .. }));
    }

    // ── ALiBi: compute_alibi_slopes ─────────────────────────────────

    #[test]
    fn test_alibi_slopes_one_head() {
        let slopes = compute_alibi_slopes(1).unwrap();
        assert_eq!(slopes.len(), 1);
        assert!((slopes[0] - 2.0f32.powi(-8)).abs() < 1e-7);
    }

    #[test]
    fn test_alibi_slopes_eight_heads() {
        let slopes = compute_alibi_slopes(8).unwrap();
        assert_eq!(slopes.len(), 8);
        for (i, &s) in slopes.iter().enumerate() {
            let expected = 2.0f32.powi(-(i as i32 + 1));
            assert!((s - expected).abs() < 1e-6, "head {i}: {s} vs {expected}");
        }
    }

    #[test]
    fn test_alibi_slopes_monotonically_decreasing() {
        let slopes = compute_alibi_slopes(16).unwrap();
        for i in 1..slopes.len() {
            assert!(slopes[i] < slopes[i - 1], "slopes should decrease");
        }
    }

    #[test]
    fn test_alibi_slopes_zero_heads_error() {
        let err = compute_alibi_slopes(0).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::InvalidHeadCount { num_heads: 0 }));
    }

    #[test]
    fn test_alibi_slopes_non_power_of_two_error() {
        let err = compute_alibi_slopes(3).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::InvalidHeadCount { num_heads: 3 }));
    }

    #[test]
    fn test_alibi_slopes_various_powers() {
        for n in [1, 2, 4, 8, 16, 32, 64] {
            let slopes = compute_alibi_slopes(n).unwrap();
            assert_eq!(slopes.len(), n);
            assert!(slopes.iter().all(|s| *s > 0.0 && s.is_finite()));
        }
    }

    // ── ALiBi: apply_alibi_bias ─────────────────────────────────────

    #[test]
    fn test_alibi_bias_shape() {
        let slopes = compute_alibi_slopes(4).unwrap();
        let bias = apply_alibi_bias(&slopes, 6).unwrap();
        assert_eq!(bias.len(), 4 * 6 * 6);
    }

    #[test]
    fn test_alibi_bias_diagonal_zero() {
        let slopes = compute_alibi_slopes(2).unwrap();
        let bias = apply_alibi_bias(&slopes, 5).unwrap();
        let seq = 5;
        for h in 0..2 {
            for i in 0..seq {
                let idx = h * seq * seq + i * seq + i;
                assert!(
                    bias[idx].abs() < 1e-8,
                    "diagonal should be zero: head={h}, pos={i}, val={}",
                    bias[idx]
                );
            }
        }
    }

    #[test]
    fn test_alibi_bias_causal_mask() {
        let slopes = compute_alibi_slopes(1).unwrap();
        let bias = apply_alibi_bias(&slopes, 4).unwrap();
        let seq = 4;
        for i in 0..seq {
            for j in (i + 1)..seq {
                assert!(bias[i * seq + j] == f32::NEG_INFINITY, "future position should be -inf");
            }
        }
    }

    #[test]
    fn test_alibi_bias_non_positive() {
        let slopes = compute_alibi_slopes(4).unwrap();
        let bias = apply_alibi_bias(&slopes, 8).unwrap();
        for &v in &bias {
            assert!(v <= 0.0 || v == f32::NEG_INFINITY, "bias should be non-positive, got {v}");
        }
    }

    #[test]
    fn test_alibi_bias_empty_slopes_error() {
        let err = apply_alibi_bias(&[], 4).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::ZeroDimension { .. }));
    }

    #[test]
    fn test_alibi_bias_linear_increase() {
        let slopes = compute_alibi_slopes(1).unwrap();
        let bias = apply_alibi_bias(&slopes, 5).unwrap();
        let seq = 5;
        for j in 0..5 {
            let expected = -slopes[0] * (4 - j) as f32;
            let actual = bias[4 * seq + j];
            assert!((actual - expected).abs() < 1e-6, "j={j}: {actual} vs {expected}");
        }
    }

    // ── Relative position bias ──────────────────────────────────────

    #[test]
    fn test_relative_bias_shape() {
        let indices = relative_position_bias(6, 32, 128).unwrap();
        assert_eq!(indices.len(), 6 * 6);
    }

    #[test]
    fn test_relative_bias_diagonal_zero() {
        let indices = relative_position_bias(8, 32, 128).unwrap();
        for i in 0..8 {
            assert_eq!(indices[i * 8 + i], 0, "diagonal should be bucket 0");
        }
    }

    #[test]
    fn test_relative_bias_bounded() {
        let num_buckets = 32;
        let indices = relative_position_bias(10, num_buckets, 128).unwrap();
        for &idx in &indices {
            assert!(idx < num_buckets, "bucket {idx} >= {num_buckets}");
        }
    }

    #[test]
    fn test_relative_bias_zero_buckets_error() {
        let err = relative_position_bias(4, 0, 128).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::ZeroDimension { name: "num_buckets" }));
    }

    #[test]
    fn test_relative_bias_zero_distance_error() {
        let err = relative_position_bias(4, 32, 0).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::ZeroDimension { name: "max_distance" }));
    }

    #[test]
    fn test_relative_bias_symmetry() {
        let indices = relative_position_bias(5, 32, 128).unwrap();
        let half = 16;
        // i=0, j=1 → positive distance → bucket < half
        assert!(indices[0 * 5 + 1] < half);
        // i=1, j=0 → negative distance → bucket >= half
        assert!(indices[1 * 5 + 0] >= half);
    }

    // ── Learnable PE lookup ─────────────────────────────────────────

    #[test]
    fn test_learnable_lookup_basic() {
        let table: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out = learnable_pe_lookup(&table, 3, 4, &[0, 2]).unwrap();
        assert_eq!(out, &[0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_learnable_lookup_out_of_range_error() {
        let table = vec![0.0; 8];
        let err = learnable_pe_lookup(&table, 2, 4, &[5]).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::PositionOutOfRange { position: 5, .. }));
    }

    #[test]
    fn test_learnable_lookup_zero_embed_dim_error() {
        let err = learnable_pe_lookup(&[], 1, 0, &[0]).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::ZeroDimension { .. }));
    }

    #[test]
    fn test_learnable_lookup_table_too_small_error() {
        let table = vec![0.0; 3];
        let err = learnable_pe_lookup(&table, 2, 4, &[0]).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_learnable_lookup_empty_positions() {
        let table = vec![1.0; 16];
        let out = learnable_pe_lookup(&table, 4, 4, &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_learnable_lookup_all_positions() {
        let table: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let out = learnable_pe_lookup(&table, 3, 2, &[0, 1, 2]).unwrap();
        assert_eq!(out, table);
    }

    // ── Error Display ───────────────────────────────────────────────

    #[test]
    fn test_error_display_zero_dim() {
        let e = PositionalEncodingError::ZeroDimension { name: "model_dim" };
        assert!(e.to_string().contains("model_dim"));
    }

    #[test]
    fn test_error_display_mismatch() {
        let e = PositionalEncodingError::DimensionMismatch {
            expected: 8,
            actual: 4,
            context: "head data",
        };
        let msg = e.to_string();
        assert!(msg.contains("8") && msg.contains("4"));
    }

    #[test]
    fn test_error_implements_std_error() {
        let e: Box<dyn std::error::Error> =
            Box::new(PositionalEncodingError::ZeroDimension { name: "x" });
        assert!(!e.to_string().is_empty());
    }

    // ── Miscellaneous edge cases ────────────────────────────────────

    #[test]
    fn test_rope_large_head_dim() {
        let hd = 128;
        let (sin_t, cos_t) = build_sin_cos_tables(2, hd, 10_000.0).unwrap();
        let half = hd / 2;
        let mut data: Vec<f32> = (0..hd).map(|i| (i as f32) * 0.01).collect();
        apply_rope_pe(&mut data, hd, &sin_t[half..2 * half], &cos_t[half..2 * half]).unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_neox_large_head_dim() {
        let hd = 128;
        let (sin_t, cos_t) = build_sin_cos_tables(2, hd, 10_000.0).unwrap();
        let half = hd / 2;
        let mut data: Vec<f32> = (0..hd).map(|i| (i as f32) * 0.01).collect();
        apply_rope_neox(&mut data, hd, &sin_t[half..2 * half], &cos_t[half..2 * half]).unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_alibi_bias_single_position() {
        let slopes = compute_alibi_slopes(2).unwrap();
        let bias = apply_alibi_bias(&slopes, 1).unwrap();
        assert_eq!(bias.len(), 2);
        assert!(bias[0].abs() < 1e-8);
        assert!(bias[1].abs() < 1e-8);
    }

    #[test]
    fn test_relative_bias_single_position() {
        let indices = relative_position_bias(1, 32, 128).unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_sinusoidal_pe_odd_dim() {
        let pe = sinusoidal_pe(3, 5, 10_000.0).unwrap();
        assert_eq!(pe.len(), 15);
        assert!(pe.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rope_zero_head_dim_error() {
        let err = apply_rope_pe(&mut [], 0, &[], &[]).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::ZeroDimension { .. }));
    }

    #[test]
    fn test_neox_zero_head_dim_error() {
        let err = apply_rope_neox(&mut [], 0, &[], &[]).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::ZeroDimension { .. }));
    }

    #[test]
    fn test_rope_sin_cos_row_too_short_error() {
        let mut data = vec![0.0; 4];
        let err = apply_rope_pe(&mut data, 4, &[0.0], &[0.0; 2]).unwrap_err();
        assert!(matches!(err, PositionalEncodingError::DimensionMismatch { .. }));
    }
}
