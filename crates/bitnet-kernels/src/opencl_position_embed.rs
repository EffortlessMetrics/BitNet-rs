//! OpenCL position embedding generation for Intel A770.
//!
//! Supports multiple position encoding strategies:
//! - **Sinusoidal** — fixed positional encoding (Vaswani et al.)
//! - **RoPE** — Rotary Position Embedding (Su et al.)
//! - **ALiBi** — Attention with Linear Biases (Press et al.)
//! - **Learned** — trainable position embeddings
//! - **NTK-RoPE** — NTK-aware context-length extension
//!
//! Each method provides a CPU reference implementation with matching
//! signatures for future OpenCL kernel dispatch.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors specific to position embedding generation.
#[derive(Debug, Clone, PartialEq)]
pub enum PositionError {
    /// The requested encoding method is not available.
    UnsupportedMethod,
    /// Hidden dimension does not match expected value.
    DimensionMismatch { expected: usize, actual: usize },
    /// Requested position exceeds configured maximum.
    ExceedsMaxLength { position: usize, max: usize },
}

impl fmt::Display for PositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedMethod => write!(f, "unsupported position encoding method"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::ExceedsMaxLength { position, max } => {
                write!(f, "position {position} exceeds max length {max}")
            }
        }
    }
}

impl std::error::Error for PositionError {}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Position encoding strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum PositionMethod {
    Sinusoidal,
    RoPE { base: f32, dim: usize },
    ALiBi { num_heads: usize },
    Learned { max_positions: usize },
    NTKRoPE { base: f32, alpha: f32, dim: usize },
}

/// Configuration for position embedding generation.
#[derive(Debug, Clone)]
pub struct PositionConfig {
    pub method: PositionMethod,
    pub max_seq_len: usize,
    pub hidden_dim: usize,
}

/// A position embedding tensor.
#[derive(Debug, Clone)]
pub struct PositionEmbedding {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub method: PositionMethod,
}

/// Pre-computed RoPE cos/sin lookup tables.
#[derive(Debug, Clone)]
pub struct RoPETable {
    pub cos_table: Vec<f32>,
    pub sin_table: Vec<f32>,
    pub dim: usize,
    pub max_positions: usize,
}

/// ALiBi per-head slope values.
#[derive(Debug, Clone)]
pub struct ALiBiSlope {
    pub slopes: Vec<f32>,
    pub num_heads: usize,
}

/// Stateful position embedding generator with caching.
pub struct PositionGenerator {
    pub config: PositionConfig,
    pub rope_table: Option<RoPETable>,
    pub alibi_slopes: Option<ALiBiSlope>,
    pub cache: HashMap<usize, PositionEmbedding>,
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create a [`PositionGenerator`] pre-populated with method-specific tables.
pub fn create_position_generator(config: PositionConfig) -> PositionGenerator {
    let rope_table = match &config.method {
        PositionMethod::RoPE { base, dim } => {
            Some(cpu_build_rope_table(*base, *dim, config.max_seq_len))
        }
        PositionMethod::NTKRoPE { base, alpha, dim } => {
            Some(cpu_ntk_rope_table(*base, *alpha, *dim, config.max_seq_len))
        }
        _ => None,
    };
    let alibi_slopes = match &config.method {
        PositionMethod::ALiBi { num_heads } => Some(cpu_build_alibi_slopes(*num_heads)),
        _ => None,
    };
    PositionGenerator { config, rope_table, alibi_slopes, cache: HashMap::new() }
}

// ---------------------------------------------------------------------------
// Sinusoidal
// ---------------------------------------------------------------------------

/// CPU reference sinusoidal positional encoding.
///
/// PE(pos, 2i)   = sin(pos / 10000^(2i/d))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
pub fn cpu_sinusoidal_encoding(position: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    let pos = position as f32;
    let d = dim as f32;
    for i in 0..dim / 2 {
        let angle = pos / (10000.0_f32).powf(2.0 * i as f32 / d);
        out[2 * i] = angle.sin();
        out[2 * i + 1] = angle.cos();
    }
    // Handle odd dimension
    if dim % 2 == 1 {
        let angle = pos / (10000.0_f32).powf(2.0 * (dim / 2) as f32 / d);
        out[dim - 1] = angle.sin();
    }
    out
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

/// Build the cos/sin frequency table for standard RoPE.
pub fn cpu_build_rope_table(base: f32, dim: usize, max_positions: usize) -> RoPETable {
    let half = dim / 2;
    let mut cos_table = vec![0.0f32; max_positions * half];
    let mut sin_table = vec![0.0f32; max_positions * half];
    for pos in 0..max_positions {
        for i in 0..half {
            let freq = 1.0 / base.powf(2.0 * i as f32 / dim as f32);
            let angle = pos as f32 * freq;
            cos_table[pos * half + i] = angle.cos();
            sin_table[pos * half + i] = angle.sin();
        }
    }
    RoPETable { cos_table, sin_table, dim, max_positions }
}

/// Apply RoPE rotation to a vector at a given position.
///
/// Pairs (x[2i], x[2i+1]) are rotated by the corresponding angle.
pub fn cpu_apply_rope(x: &[f32], position: usize, table: &RoPETable) -> Vec<f32> {
    let half = table.dim / 2;
    let mut out = x.to_vec();
    for i in 0..half.min(x.len() / 2) {
        let cos = table.cos_table[position * half + i];
        let sin = table.sin_table[position * half + i];
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        out[2 * i] = x0 * cos - x1 * sin;
        out[2 * i + 1] = x0 * sin + x1 * cos;
    }
    out
}

// ---------------------------------------------------------------------------
// ALiBi
// ---------------------------------------------------------------------------

/// Build ALiBi slopes as a geometric series: slope_h = 2^(-8h/H).
pub fn cpu_build_alibi_slopes(num_heads: usize) -> ALiBiSlope {
    let slopes: Vec<f32> =
        (0..num_heads).map(|h| 2.0_f32.powf(-8.0 * (h + 1) as f32 / num_heads as f32)).collect();
    ALiBiSlope { slopes, num_heads }
}

/// Apply ALiBi position bias to attention scores (in-place).
///
/// `attention_scores` layout: `[num_heads, seq_len, seq_len]`.
pub fn cpu_apply_alibi(attention_scores: &mut [f32], slopes: &ALiBiSlope, seq_len: usize) {
    let n = slopes.num_heads;
    let block = seq_len * seq_len;
    for h in 0..n {
        let slope = slopes.slopes[h];
        for q in 0..seq_len {
            for k in 0..seq_len {
                let dist = (q as isize - k as isize).unsigned_abs();
                attention_scores[h * block + q * seq_len + k] -= slope * dist as f32;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NTK-RoPE
// ---------------------------------------------------------------------------

/// Build NTK-aware RoPE table with base scaling: base' = base * alpha^(dim/(dim-2)).
pub fn cpu_ntk_rope_table(base: f32, alpha: f32, dim: usize, max_pos: usize) -> RoPETable {
    let scaled_base = base * alpha.powf(dim as f32 / (dim as f32 - 2.0));
    cpu_build_rope_table(scaled_base, dim, max_pos)
}

// ---------------------------------------------------------------------------
// Batch generation
// ---------------------------------------------------------------------------

/// Generate embeddings for a batch of positions.
pub fn cpu_generate_positions(
    generator: &mut PositionGenerator,
    positions: &[usize],
) -> Result<Vec<PositionEmbedding>, PositionError> {
    let max = generator.config.max_seq_len;
    let dim = generator.config.hidden_dim;
    let mut results = Vec::with_capacity(positions.len());

    for &pos in positions {
        if pos >= max {
            return Err(PositionError::ExceedsMaxLength { position: pos, max });
        }
        if let Some(cached) = generator.cache.get(&pos) {
            results.push(cached.clone());
            continue;
        }
        let emb = match &generator.config.method {
            PositionMethod::Sinusoidal => {
                let data = cpu_sinusoidal_encoding(pos, dim);
                PositionEmbedding {
                    data,
                    shape: vec![1, dim],
                    method: generator.config.method.clone(),
                }
            }
            PositionMethod::RoPE { .. } | PositionMethod::NTKRoPE { .. } => {
                let table =
                    generator.rope_table.as_ref().ok_or(PositionError::UnsupportedMethod)?;
                let identity: Vec<f32> =
                    (0..dim).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
                let data = cpu_apply_rope(&identity, pos, table);
                PositionEmbedding {
                    data,
                    shape: vec![1, dim],
                    method: generator.config.method.clone(),
                }
            }
            PositionMethod::ALiBi { .. } => {
                // ALiBi doesn't produce per-position embeddings in the same sense;
                // return a placeholder of zeros — bias is applied on attention scores.
                PositionEmbedding {
                    data: vec![0.0; dim],
                    shape: vec![1, dim],
                    method: generator.config.method.clone(),
                }
            }
            PositionMethod::Learned { max_positions } => {
                if pos >= *max_positions {
                    return Err(PositionError::ExceedsMaxLength {
                        position: pos,
                        max: *max_positions,
                    });
                }
                // Deterministic pseudo-random learned embedding (hash-based).
                let data = (0..dim)
                    .map(|i| {
                        let seed = pos as f32 * 0.01 + i as f32 * 0.001;
                        seed.sin()
                    })
                    .collect();
                PositionEmbedding {
                    data,
                    shape: vec![1, dim],
                    method: generator.config.method.clone(),
                }
            }
        };
        generator.cache.insert(pos, emb.clone());
        results.push(emb);
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Cache management
// ---------------------------------------------------------------------------

/// Extend cached tables to support a larger context length.
pub fn cpu_extend_context(generator: &mut PositionGenerator, new_max: usize) {
    if new_max <= generator.config.max_seq_len {
        return;
    }
    generator.config.max_seq_len = new_max;
    match &generator.config.method {
        PositionMethod::RoPE { base, dim } => {
            generator.rope_table = Some(cpu_build_rope_table(*base, *dim, new_max));
        }
        PositionMethod::NTKRoPE { base, alpha, dim } => {
            generator.rope_table = Some(cpu_ntk_rope_table(*base, *alpha, *dim, new_max));
        }
        _ => {}
    }
    generator.cache.clear();
}

/// Number of cached position embeddings.
pub fn cpu_get_cache_size(generator: &PositionGenerator) -> usize {
    generator.cache.len()
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

/// Linearly interpolate a position embedding at a fractional position.
///
/// Given an embedding for integer position `floor(position)`, produces an
/// interpolation between `floor` and `ceil` encodings.
pub fn cpu_interpolate_position(embedding: &PositionEmbedding, position: f32) -> Vec<f32> {
    let dim = embedding.data.len();
    let floor_pos = position.floor() as usize;
    let ceil_pos = floor_pos + 1;
    let t = position - position.floor();

    match &embedding.method {
        PositionMethod::Sinusoidal => {
            let a = cpu_sinusoidal_encoding(floor_pos, dim);
            let b = cpu_sinusoidal_encoding(ceil_pos, dim);
            a.iter().zip(b.iter()).map(|(va, vb)| va * (1.0 - t) + vb * t).collect()
        }
        _ => {
            // For non-sinusoidal methods, scale the existing embedding linearly.
            embedding.data.iter().map(|v| v * (1.0 - t)).collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

/// Human-readable summary of a [`PositionConfig`].
pub fn format_position_config(config: &PositionConfig) -> String {
    let method_str = match &config.method {
        PositionMethod::Sinusoidal => "sinusoidal".to_string(),
        PositionMethod::RoPE { base, dim } => format!("RoPE(base={base}, dim={dim})"),
        PositionMethod::ALiBi { num_heads } => format!("ALiBi(heads={num_heads})"),
        PositionMethod::Learned { max_positions } => {
            format!("learned(max={max_positions})")
        }
        PositionMethod::NTKRoPE { base, alpha, dim } => {
            format!("NTK-RoPE(base={base}, α={alpha}, dim={dim})")
        }
    };
    format!(
        "PositionConfig {{ method: {method_str}, max_seq_len: {}, hidden_dim: {} }}",
        config.max_seq_len, config.hidden_dim
    )
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ------------------------------------------------------------

    fn default_sinusoidal(dim: usize) -> PositionConfig {
        PositionConfig { method: PositionMethod::Sinusoidal, max_seq_len: 4096, hidden_dim: dim }
    }

    fn default_rope(dim: usize) -> PositionConfig {
        PositionConfig {
            method: PositionMethod::RoPE { base: 10000.0, dim },
            max_seq_len: 4096,
            hidden_dim: dim,
        }
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
        for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            assert!((va - vb).abs() < tol, "{msg}: index {i}: {va} vs {vb} (tol={tol})");
        }
    }

    fn vec_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    // -----------------------------------------------------------------------
    // Sinusoidal tests
    // -----------------------------------------------------------------------

    #[test]
    fn sinusoidal_known_position_1() {
        let enc = cpu_sinusoidal_encoding(1, 4);
        // PE(1,0) = sin(1/10000^0) = sin(1)
        // PE(1,1) = cos(1/10000^0) = cos(1)
        let expected_0 = 1.0_f32.sin();
        let expected_1 = 1.0_f32.cos();
        assert!((enc[0] - expected_0).abs() < 1e-6);
        assert!((enc[1] - expected_1).abs() < 1e-6);
    }

    #[test]
    fn sinusoidal_known_position_10() {
        let enc = cpu_sinusoidal_encoding(10, 4);
        let angle0 = 10.0 / 10000.0_f32.powf(0.0 / 4.0);
        let angle1 = 10.0 / 10000.0_f32.powf(2.0 / 4.0);
        assert!((enc[0] - angle0.sin()).abs() < 1e-6);
        assert!((enc[1] - angle0.cos()).abs() < 1e-6);
        assert!((enc[2] - angle1.sin()).abs() < 1e-6);
        assert!((enc[3] - angle1.cos()).abs() < 1e-6);
    }

    #[test]
    fn sinusoidal_position_0_is_known() {
        let enc = cpu_sinusoidal_encoding(0, 8);
        // sin(0) = 0 for all even indices; cos(0) = 1 for all odd indices
        for i in (0..8).step_by(2) {
            assert!((enc[i]).abs() < 1e-6, "sin(0) should be 0 at index {i}");
            assert!((enc[i + 1] - 1.0).abs() < 1e-6, "cos(0) should be 1 at index {}", i + 1);
        }
    }

    #[test]
    fn sinusoidal_different_positions_differ() {
        let a = cpu_sinusoidal_encoding(1, 64);
        let b = cpu_sinusoidal_encoding(2, 64);
        assert_ne!(a, b);
    }

    #[test]
    fn sinusoidal_orthogonality_proxy() {
        // Dot product of sinusoidal encodings for distant positions should be small
        // relative to same-position dot product.
        let dim = 256;
        let a = cpu_sinusoidal_encoding(0, dim);
        let b = cpu_sinusoidal_encoding(100, dim);
        let self_dot: f32 = a.iter().map(|x| x * x).sum();
        let cross_dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(cross_dot.abs() < self_dot * 0.5, "cross_dot={cross_dot}, self_dot={self_dot}");
    }

    #[test]
    fn sinusoidal_bounded() {
        for pos in [0, 1, 50, 500, 4000] {
            let enc = cpu_sinusoidal_encoding(pos, 128);
            for &v in &enc {
                assert!((-1.0..=1.0).contains(&v), "value {v} out of [-1,1] at pos {pos}");
            }
        }
    }

    #[test]
    fn sinusoidal_dim_2_minimal() {
        let enc = cpu_sinusoidal_encoding(5, 2);
        assert_eq!(enc.len(), 2);
        let angle = 5.0 / 10000.0_f32.powf(0.0 / 2.0);
        assert!((enc[0] - angle.sin()).abs() < 1e-6);
        assert!((enc[1] - angle.cos()).abs() < 1e-6);
    }

    #[test]
    fn sinusoidal_large_dim() {
        let enc = cpu_sinusoidal_encoding(42, 2048);
        assert_eq!(enc.len(), 2048);
        for &v in &enc {
            assert!(v.is_finite());
        }
    }

    // -----------------------------------------------------------------------
    // RoPE tests
    // -----------------------------------------------------------------------

    #[test]
    fn rope_table_correct_cos_sin() {
        let table = cpu_build_rope_table(10000.0, 4, 8);
        let half = 2; // dim/2
        // Position 1, freq index 0: angle = 1.0 * 1/(10000^(0/4)) = 1.0
        let cos_val = table.cos_table[1 * half + 0];
        let sin_val = table.sin_table[1 * half + 0];
        assert!((cos_val - 1.0_f32.cos()).abs() < 1e-6);
        assert!((sin_val - 1.0_f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn rope_table_position_0() {
        let table = cpu_build_rope_table(10000.0, 8, 16);
        let half = 4;
        // All angles are 0 at position 0 → cos=1, sin=0
        for i in 0..half {
            assert!((table.cos_table[i] - 1.0).abs() < 1e-6);
            assert!((table.sin_table[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_table_dimensions() {
        let table = cpu_build_rope_table(10000.0, 16, 32);
        assert_eq!(table.dim, 16);
        assert_eq!(table.max_positions, 32);
        assert_eq!(table.cos_table.len(), 32 * 8); // max_pos * half
        assert_eq!(table.sin_table.len(), 32 * 8);
    }

    #[test]
    fn rope_apply_preserves_norm() {
        let table = cpu_build_rope_table(10000.0, 8, 32);
        let x: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let norm_before = vec_norm(&x);
        let rotated = cpu_apply_rope(&x, 5, &table);
        let norm_after = vec_norm(&rotated);
        assert!(
            (norm_before - norm_after).abs() < 1e-5,
            "norm changed: {norm_before} → {norm_after}"
        );
    }

    #[test]
    fn rope_apply_different_positions_differ() {
        let table = cpu_build_rope_table(10000.0, 8, 32);
        let x: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let a = cpu_apply_rope(&x, 0, &table);
        let b = cpu_apply_rope(&x, 1, &table);
        assert_ne!(a, b);
    }

    #[test]
    fn rope_apply_position_0_identity_like() {
        let table = cpu_build_rope_table(10000.0, 4, 8);
        let x = vec![3.0, 4.0, 5.0, 6.0];
        let rotated = cpu_apply_rope(&x, 0, &table);
        // At position 0 all angles are 0: cos=1, sin=0 → identity
        assert_close(&x, &rotated, 1e-6, "pos-0 should be identity");
    }

    #[test]
    fn rope_preserves_norm_many_positions() {
        let table = cpu_build_rope_table(10000.0, 64, 128);
        let x: Vec<f32> = (0..64).map(|i| ((i * 7 + 3) as f32).sin()).collect();
        let orig_norm = vec_norm(&x);
        for pos in [0, 1, 10, 63, 127] {
            let r = cpu_apply_rope(&x, pos, &table);
            assert!((vec_norm(&r) - orig_norm).abs() < 1e-4, "norm differs at pos {pos}");
        }
    }

    // -----------------------------------------------------------------------
    // ALiBi tests
    // -----------------------------------------------------------------------

    #[test]
    fn alibi_slopes_geometric() {
        let s = cpu_build_alibi_slopes(8);
        assert_eq!(s.slopes.len(), 8);
        // Each slope should be smaller than the previous (geometric decrease)
        for i in 1..8 {
            assert!(
                s.slopes[i] < s.slopes[i - 1],
                "slope[{i}]={} should be < slope[{}]={}",
                s.slopes[i],
                i - 1,
                s.slopes[i - 1]
            );
        }
    }

    #[test]
    fn alibi_slopes_decrease_with_head_index() {
        let s = cpu_build_alibi_slopes(16);
        for w in s.slopes.windows(2) {
            assert!(w[1] < w[0], "slopes must decrease: {} >= {}", w[1], w[0]);
        }
    }

    #[test]
    fn alibi_slopes_single_head() {
        let s = cpu_build_alibi_slopes(1);
        assert_eq!(s.slopes.len(), 1);
        // 2^(-8*1/1) = 2^(-8)
        let expected = 2.0_f32.powf(-8.0);
        assert!((s.slopes[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn alibi_apply_linear_bias() {
        let slopes = cpu_build_alibi_slopes(1);
        let seq_len = 4;
        let mut scores = vec![0.0f32; seq_len * seq_len];
        cpu_apply_alibi(&mut scores, &slopes, seq_len);
        // At (q=3, k=0), distance = 3 → bias = -slope * 3
        let bias = scores[3 * seq_len + 0];
        let expected = -slopes.slopes[0] * 3.0;
        assert!((bias - expected).abs() < 1e-6, "bias={bias}, expected={expected}");
    }

    #[test]
    fn alibi_apply_diagonal_zero() {
        let slopes = cpu_build_alibi_slopes(4);
        let seq_len = 8;
        let mut scores = vec![0.0f32; 4 * seq_len * seq_len];
        cpu_apply_alibi(&mut scores, &slopes, seq_len);
        // Diagonal: q == k → distance 0 → no bias
        for h in 0..4 {
            for i in 0..seq_len {
                let val = scores[h * seq_len * seq_len + i * seq_len + i];
                assert!((val).abs() < 1e-6, "diagonal should be 0, got {val} at h={h},i={i}");
            }
        }
    }

    #[test]
    fn alibi_slopes_positive() {
        let s = cpu_build_alibi_slopes(32);
        for (i, &slope) in s.slopes.iter().enumerate() {
            assert!(slope > 0.0, "slope[{i}] should be positive, got {slope}");
        }
    }

    // -----------------------------------------------------------------------
    // NTK-RoPE tests
    // -----------------------------------------------------------------------

    #[test]
    fn ntk_rope_extends_context() {
        let std_table = cpu_build_rope_table(10000.0, 8, 64);
        let ntk_table = cpu_ntk_rope_table(10000.0, 2.0, 8, 64);
        // NTK scaling changes the effective base, so tables should differ
        assert_ne!(std_table.cos_table, ntk_table.cos_table);
    }

    #[test]
    fn ntk_rope_alpha_1_matches_standard() {
        // alpha=1 → base' = base * 1^(d/(d-2)) = base
        let std_table = cpu_build_rope_table(10000.0, 8, 16);
        let ntk_table = cpu_ntk_rope_table(10000.0, 1.0, 8, 16);
        assert_close(&std_table.cos_table, &ntk_table.cos_table, 1e-5, "alpha=1 cos");
        assert_close(&std_table.sin_table, &ntk_table.sin_table, 1e-5, "alpha=1 sin");
    }

    #[test]
    fn ntk_rope_preserves_norm() {
        let table = cpu_ntk_rope_table(10000.0, 4.0, 16, 64);
        let x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.3).cos()).collect();
        let orig = vec_norm(&x);
        let rotated = cpu_apply_rope(&x, 10, &table);
        assert!((vec_norm(&rotated) - orig).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // Position generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn generate_sinusoidal_correct_shape() {
        let config = default_sinusoidal(64);
        let mut generator = create_position_generator(config);
        let embs = cpu_generate_positions(&mut generator, &[0, 1, 2]).unwrap();
        assert_eq!(embs.len(), 3);
        for e in &embs {
            assert_eq!(e.shape, vec![1, 64]);
            assert_eq!(e.data.len(), 64);
        }
    }

    #[test]
    fn generate_rope_correct_shape() {
        let config = default_rope(32);
        let mut generator = create_position_generator(config);
        let embs = cpu_generate_positions(&mut generator, &[0, 5, 10]).unwrap();
        assert_eq!(embs.len(), 3);
        for e in &embs {
            assert_eq!(e.shape, vec![1, 32]);
        }
    }

    #[test]
    fn generate_exceeds_max_length() {
        let config =
            PositionConfig { method: PositionMethod::Sinusoidal, max_seq_len: 10, hidden_dim: 4 };
        let mut generator = create_position_generator(config);
        let result = cpu_generate_positions(&mut generator, &[10]);
        assert!(matches!(result, Err(PositionError::ExceedsMaxLength { position: 10, max: 10 })));
    }

    #[test]
    fn generate_learned_exceeds_max_positions() {
        let config = PositionConfig {
            method: PositionMethod::Learned { max_positions: 5 },
            max_seq_len: 100,
            hidden_dim: 8,
        };
        let mut generator = create_position_generator(config);
        let result = cpu_generate_positions(&mut generator, &[6]);
        assert!(matches!(result, Err(PositionError::ExceedsMaxLength { .. })));
    }

    // -----------------------------------------------------------------------
    // Cache tests
    // -----------------------------------------------------------------------

    #[test]
    fn cache_reuses_same_position() {
        let config = default_sinusoidal(16);
        let mut generator = create_position_generator(config);
        let first = cpu_generate_positions(&mut generator, &[3]).unwrap();
        assert_eq!(cpu_get_cache_size(&generator), 1);
        let second = cpu_generate_positions(&mut generator, &[3]).unwrap();
        assert_eq!(cpu_get_cache_size(&generator), 1);
        assert_eq!(first[0].data, second[0].data);
    }

    #[test]
    fn cache_grows_with_new_positions() {
        let config = default_sinusoidal(16);
        let mut generator = create_position_generator(config);
        cpu_generate_positions(&mut generator, &[0, 1, 2]).unwrap();
        assert_eq!(cpu_get_cache_size(&generator), 3);
        cpu_generate_positions(&mut generator, &[3, 4]).unwrap();
        assert_eq!(cpu_get_cache_size(&generator), 5);
    }

    // -----------------------------------------------------------------------
    // Context extension tests
    // -----------------------------------------------------------------------

    #[test]
    fn extend_context_table_grows() {
        let config = default_rope(8);
        let mut generator = create_position_generator(config);
        assert_eq!(generator.rope_table.as_ref().unwrap().max_positions, 4096);
        cpu_extend_context(&mut generator, 8192);
        assert_eq!(generator.rope_table.as_ref().unwrap().max_positions, 8192);
        assert_eq!(generator.config.max_seq_len, 8192);
    }

    #[test]
    fn extend_context_clears_cache() {
        let config = default_rope(8);
        let mut generator = create_position_generator(config);
        cpu_generate_positions(&mut generator, &[0, 1]).unwrap();
        assert_eq!(cpu_get_cache_size(&generator), 2);
        cpu_extend_context(&mut generator, 8192);
        assert_eq!(cpu_get_cache_size(&generator), 0);
    }

    #[test]
    fn extend_context_no_shrink() {
        let config = default_rope(8);
        let mut generator = create_position_generator(config);
        cpu_extend_context(&mut generator, 100); // smaller → no-op
        assert_eq!(generator.config.max_seq_len, 4096);
    }

    // -----------------------------------------------------------------------
    // Interpolation tests
    // -----------------------------------------------------------------------

    #[test]
    fn interpolation_smooth_sinusoidal() {
        let a = cpu_sinusoidal_encoding(5, 16);
        let b = cpu_sinusoidal_encoding(6, 16);
        let emb = PositionEmbedding {
            data: a.clone(),
            shape: vec![1, 16],
            method: PositionMethod::Sinusoidal,
        };
        let interp = cpu_interpolate_position(&emb, 5.5);
        // Should be midpoint of a and b
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| (x + y) / 2.0).collect();
        assert_close(&interp, &expected, 1e-5, "midpoint interpolation");
    }

    #[test]
    fn interpolation_integer_position() {
        let enc = cpu_sinusoidal_encoding(3, 8);
        let emb = PositionEmbedding {
            data: enc.clone(),
            shape: vec![1, 8],
            method: PositionMethod::Sinusoidal,
        };
        let interp = cpu_interpolate_position(&emb, 3.0);
        assert_close(&interp, &enc, 1e-5, "integer position should match");
    }

    // -----------------------------------------------------------------------
    // Format / display tests
    // -----------------------------------------------------------------------

    #[test]
    fn format_sinusoidal_config() {
        let config = default_sinusoidal(128);
        let s = format_position_config(&config);
        assert!(s.contains("sinusoidal"));
        assert!(s.contains("4096"));
        assert!(s.contains("128"));
    }

    #[test]
    fn format_rope_config() {
        let config = default_rope(64);
        let s = format_position_config(&config);
        assert!(s.contains("RoPE"));
        assert!(s.contains("10000"));
    }

    #[test]
    fn format_alibi_config() {
        let config = PositionConfig {
            method: PositionMethod::ALiBi { num_heads: 8 },
            max_seq_len: 2048,
            hidden_dim: 512,
        };
        let s = format_position_config(&config);
        assert!(s.contains("ALiBi"));
        assert!(s.contains("8"));
    }

    #[test]
    fn format_ntk_config() {
        let config = PositionConfig {
            method: PositionMethod::NTKRoPE { base: 10000.0, alpha: 2.0, dim: 64 },
            max_seq_len: 8192,
            hidden_dim: 64,
        };
        let s = format_position_config(&config);
        assert!(s.contains("NTK"));
        assert!(s.contains("α=2"));
    }

    // -----------------------------------------------------------------------
    // BitNet compatibility
    // -----------------------------------------------------------------------

    #[test]
    fn bitnet_2048_hidden_dim() {
        let config = PositionConfig {
            method: PositionMethod::RoPE { base: 10000.0, dim: 2048 },
            max_seq_len: 4096,
            hidden_dim: 2048,
        };
        let mut generator = create_position_generator(config);
        let embs = cpu_generate_positions(&mut generator, &[0, 1, 100]).unwrap();
        for e in &embs {
            assert_eq!(e.data.len(), 2048);
            for &v in &e.data {
                assert!(v.is_finite(), "non-finite value in 2048-dim embedding");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Error type tests
    // -----------------------------------------------------------------------

    #[test]
    fn error_display_messages() {
        let e1 = PositionError::UnsupportedMethod;
        assert!(format!("{e1}").contains("unsupported"));

        let e2 = PositionError::DimensionMismatch { expected: 64, actual: 32 };
        assert!(format!("{e2}").contains("64"));
        assert!(format!("{e2}").contains("32"));

        let e3 = PositionError::ExceedsMaxLength { position: 999, max: 512 };
        assert!(format!("{e3}").contains("999"));
        assert!(format!("{e3}").contains("512"));
    }

    #[test]
    fn error_equality() {
        assert_eq!(PositionError::UnsupportedMethod, PositionError::UnsupportedMethod);
        assert_ne!(
            PositionError::ExceedsMaxLength { position: 1, max: 2 },
            PositionError::ExceedsMaxLength { position: 3, max: 4 },
        );
    }

    // -----------------------------------------------------------------------
    // ALiBi generation via generator
    // -----------------------------------------------------------------------

    #[test]
    fn generate_alibi_positions() {
        let config = PositionConfig {
            method: PositionMethod::ALiBi { num_heads: 4 },
            max_seq_len: 512,
            hidden_dim: 32,
        };
        let mut generator = create_position_generator(config);
        assert!(generator.alibi_slopes.is_some());
        let embs = cpu_generate_positions(&mut generator, &[0, 1, 2]).unwrap();
        assert_eq!(embs.len(), 3);
    }

    // -----------------------------------------------------------------------
    // Learned embedding generation
    // -----------------------------------------------------------------------

    #[test]
    fn generate_learned_positions() {
        let config = PositionConfig {
            method: PositionMethod::Learned { max_positions: 512 },
            max_seq_len: 1024,
            hidden_dim: 16,
        };
        let mut generator = create_position_generator(config);
        let embs = cpu_generate_positions(&mut generator, &[0, 1, 100]).unwrap();
        assert_eq!(embs.len(), 3);
        // Different positions produce different embeddings
        assert_ne!(embs[0].data, embs[1].data);
    }

    // -----------------------------------------------------------------------
    // NTK-RoPE generation
    // -----------------------------------------------------------------------

    #[test]
    fn generate_ntk_rope_positions() {
        let config = PositionConfig {
            method: PositionMethod::NTKRoPE { base: 10000.0, alpha: 2.0, dim: 16 },
            max_seq_len: 4096,
            hidden_dim: 16,
        };
        let mut generator = create_position_generator(config);
        let embs = cpu_generate_positions(&mut generator, &[0, 10, 100]).unwrap();
        assert_eq!(embs.len(), 3);
        for e in &embs {
            assert_eq!(e.data.len(), 16);
        }
    }

    #[test]
    fn extend_context_ntk_rope() {
        let config = PositionConfig {
            method: PositionMethod::NTKRoPE { base: 10000.0, alpha: 2.0, dim: 8 },
            max_seq_len: 512,
            hidden_dim: 8,
        };
        let mut generator = create_position_generator(config);
        cpu_extend_context(&mut generator, 2048);
        assert_eq!(generator.rope_table.as_ref().unwrap().max_positions, 2048);
    }
}
