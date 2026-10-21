//! SIMD-optimized embedding operations with quantized lookup and normalization.
//!
//! Extends the base [`super::embedding`] module with:
//! - SIMD-accelerated embedding table lookup (AVX2 fast path + scalar fallback)
//! - Batched embedding lookup for multi-sequence inputs
//! - Position embedding computation (absolute, sinusoidal, RoPE-based)
//! - Embedding accumulation (token + position + type embeddings)
//! - Quantized embedding lookup (INT8 / INT4 with on-the-fly dequantization)
//! - Post-embedding layer normalization
//! - Vocabulary projection (reverse embedding for logits)
//! - Sparse embedding for large vocabularies
//!
//! All public entry points auto-dispatch: AVX2 when available on x86_64,
//! otherwise portable scalar loops.
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

use bitnet_common::{BitNetError, KernelError, Result};

// ── Configuration ──────────────────────────────────────────────────────

/// Position embedding strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionEmbeddingType {
    /// Absolute learned positions from a lookup table.
    Absolute,
    /// Standard sinusoidal (Vaswani et al.): `sin/cos(pos / base^(2i/d))`.
    Sinusoidal {
        /// Base frequency (default 10 000).
        base: f32,
    },
    /// RoPE-based position encoding applied as rotation.
    Rope {
        /// Base frequency (default 10 000).
        base: f32,
    },
}

impl Default for PositionEmbeddingType {
    fn default() -> Self {
        Self::Sinusoidal { base: 10_000.0 }
    }
}

/// Configuration for SIMD embedding operations.
#[derive(Debug, Clone)]
pub struct SimdEmbeddingConfig {
    /// Number of entries (rows) in the token embedding table.
    pub vocab_size: usize,
    /// Dimensionality of each embedding vector.
    pub embed_dim: usize,
    /// Optional padding index whose embedding is always zeros.
    pub padding_idx: Option<u32>,
    /// Maximum sequence length supported for position embeddings.
    pub max_seq_len: usize,
    /// Position embedding strategy.
    pub position_type: PositionEmbeddingType,
    /// Layer-norm epsilon for post-embedding normalization.
    pub layer_norm_eps: f32,
}

impl SimdEmbeddingConfig {
    /// Create a minimal configuration.
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        Self {
            vocab_size,
            embed_dim,
            padding_idx: None,
            max_seq_len: 2048,
            position_type: PositionEmbeddingType::default(),
            layer_norm_eps: 1e-5,
        }
    }

    /// Set padding index.
    #[must_use]
    pub fn with_padding_idx(mut self, idx: u32) -> Self {
        self.padding_idx = Some(idx);
        self
    }

    /// Set position embedding type.
    #[must_use]
    pub fn with_position_type(mut self, pt: PositionEmbeddingType) -> Self {
        self.position_type = pt;
        self
    }

    /// Set max sequence length.
    #[must_use]
    pub fn with_max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        self
    }
}

/// INT4 packed embedding table with per-row scales and zero-points.
#[derive(Debug, Clone)]
pub struct Int4EmbeddingTable {
    /// Packed nibble data: two INT4 values per byte (low nibble first).
    pub data: Vec<u8>,
    /// Per-row scale factors.
    pub scales: Vec<f32>,
    /// Per-row zero-points.
    pub zero_points: Vec<f32>,
    /// Number of entries (rows).
    pub vocab_size: usize,
    /// Dimensionality of each embedding vector.
    pub embed_dim: usize,
}

/// INT8 packed embedding table with per-row scales and zero-points.
#[derive(Debug, Clone)]
pub struct Int8EmbeddingTable {
    /// Quantized embedding values (signed 8-bit).
    pub data: Vec<i8>,
    /// Per-row scale factors.
    pub scales: Vec<f32>,
    /// Per-row zero-points.
    pub zero_points: Vec<f32>,
    /// Number of entries (rows).
    pub vocab_size: usize,
    /// Dimensionality of each embedding vector.
    pub embed_dim: usize,
}

/// Sparse embedding entry for large vocabularies.
#[derive(Debug, Clone)]
pub struct SparseEmbeddingTable {
    /// Mapping from token ID to dense row index.
    pub id_to_row: std::collections::HashMap<u32, usize>,
    /// Dense embedding data `[num_stored_rows, embed_dim]`.
    pub data: Vec<f32>,
    /// Dimensionality of each embedding vector.
    pub embed_dim: usize,
    /// Default embedding returned for unmapped IDs.
    pub default_embedding: Vec<f32>,
}

// ── Error helpers ──────────────────────────────────────────────────────

fn oob_error(index: u32, vocab_size: usize) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments {
        reason: format!("embedding index {index} out of bounds for vocab_size {vocab_size}"),
    })
}

fn shape_error(reason: String) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason })
}

// ── AVX2 helpers (x86_64 only) ─────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_copy_row(src: &[f32], dst: &mut [f32]) {
    let len = src.len();
    let chunks = len / 8;
    for i in 0..chunks {
        let off = i * 8;
        let v = _mm256_loadu_ps(src.as_ptr().add(off));
        _mm256_storeu_ps(dst.as_mut_ptr().add(off), v);
    }
    for i in (chunks * 8)..len {
        *dst.get_unchecked_mut(i) = *src.get_unchecked(i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_add_row(dst: &mut [f32], src: &[f32]) {
    let len = dst.len();
    let chunks = len / 8;
    for i in 0..chunks {
        let off = i * 8;
        let d = _mm256_loadu_ps(dst.as_ptr().add(off));
        let s = _mm256_loadu_ps(src.as_ptr().add(off));
        _mm256_storeu_ps(dst.as_mut_ptr().add(off), _mm256_add_ps(d, s));
    }
    for i in (chunks * 8)..len {
        *dst.get_unchecked_mut(i) += *src.get_unchecked(i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn avx2_fmadd_row(dst: &mut [f32], src: &[f32], scale: f32) {
    let len = dst.len();
    let chunks = len / 8;
    let vs = _mm256_set1_ps(scale);
    for i in 0..chunks {
        let off = i * 8;
        let d = _mm256_loadu_ps(dst.as_ptr().add(off));
        let s = _mm256_loadu_ps(src.as_ptr().add(off));
        _mm256_storeu_ps(dst.as_mut_ptr().add(off), _mm256_fmadd_ps(s, vs, d));
    }
    for i in (chunks * 8)..len {
        *dst.get_unchecked_mut(i) += *src.get_unchecked(i) * scale;
    }
}

/// Horizontal sum of an `__m256` register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn avx2_hsum(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let sum4 = _mm_add_ps(hi, lo);
    let hi2 = _mm_movehl_ps(sum4, sum4);
    let sum2 = _mm_add_ps(sum4, hi2);
    let hi1 = _mm_shuffle_ps::<0x01>(sum2, sum2);
    _mm_cvtss_f32(_mm_add_ss(sum2, hi1))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        let off = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(off));
        let vb = _mm256_loadu_ps(b.as_ptr().add(off));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    let mut sum = avx2_hsum(acc);
    for i in (chunks * 8)..len {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
    }
    sum
}

// ── 1. SIMD-Optimized Embedding Table Lookup ───────────────────────────

/// SIMD-accelerated embedding table lookup.
///
/// Returns `[indices.len(), embed_dim]` with AVX2 fast copy when available.
pub fn simd_embedding_lookup(
    table: &[f32],
    indices: &[u32],
    config: &SimdEmbeddingConfig,
) -> Result<Vec<f32>> {
    let dim = config.embed_dim;
    if dim == 0 || indices.is_empty() {
        return Ok(Vec::new());
    }
    let vocab = config.vocab_size;
    if table.len() < vocab * dim {
        return Err(shape_error(format!(
            "table length {} < vocab_size({vocab}) * embed_dim({dim})",
            table.len()
        )));
    }

    let mut output = vec![0.0f32; indices.len() * dim];

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    for (i, &idx) in indices.iter().enumerate() {
        if config.padding_idx == Some(idx) {
            continue;
        }
        if (idx as usize) >= vocab {
            return Err(oob_error(idx, vocab));
        }
        let src = &table[(idx as usize) * dim..(idx as usize) * dim + dim];
        let dst = &mut output[i * dim..i * dim + dim];
        if use_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                avx2_copy_row(src, dst);
            }
        } else {
            dst.copy_from_slice(src);
        }
    }
    Ok(output)
}

// ── 2. Batched Embedding Lookup ────────────────────────────────────────

/// Batched embedding lookup for multiple token sequences.
///
/// Each element in `batch_indices` is a slice of token IDs for one sequence.
/// Returns `[total_tokens, embed_dim]` with sequences concatenated.
pub fn simd_embedding_lookup_batched(
    table: &[f32],
    batch_indices: &[&[u32]],
    config: &SimdEmbeddingConfig,
) -> Result<Vec<f32>> {
    let dim = config.embed_dim;
    if dim == 0 {
        return Ok(Vec::new());
    }
    let vocab = config.vocab_size;
    let total: usize = batch_indices.iter().map(|s| s.len()).sum();
    let mut output = vec![0.0f32; total * dim];

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    let mut pos = 0;
    for &seq in batch_indices {
        for &idx in seq {
            if config.padding_idx == Some(idx) {
                pos += 1;
                continue;
            }
            if (idx as usize) >= vocab {
                return Err(oob_error(idx, vocab));
            }
            let src = &table[(idx as usize) * dim..(idx as usize) * dim + dim];
            let dst = &mut output[pos * dim..pos * dim + dim];
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    avx2_copy_row(src, dst);
                }
            } else {
                dst.copy_from_slice(src);
            }
            pos += 1;
        }
    }
    Ok(output)
}

// ── 3. Position Embedding Computation ──────────────────────────────────

/// Compute sinusoidal position encodings.
///
/// Returns `[seq_len, embed_dim]` with the standard formulation:
/// - even columns: `sin(pos / base^(2i/d))`
/// - odd columns:  `cos(pos / base^(2i/d))`
pub fn compute_sinusoidal_positions(seq_len: usize, embed_dim: usize, base: f32) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * embed_dim];
    let d = embed_dim as f32;
    for pos in 0..seq_len {
        let start = pos * embed_dim;
        for i in 0..embed_dim {
            let dim_pair = (i / 2) as f32;
            let angle = (pos as f32) / base.powf(2.0 * dim_pair / d);
            output[start + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    output
}

/// Compute absolute position embeddings from a learned position table.
///
/// `position_table` has shape `[max_positions, embed_dim]`.
/// Returns `[seq_len, embed_dim]` starting at `offset`.
pub fn compute_absolute_positions(
    position_table: &[f32],
    seq_len: usize,
    embed_dim: usize,
    offset: usize,
) -> Result<Vec<f32>> {
    let max_pos = if embed_dim > 0 { position_table.len() / embed_dim } else { 0 };
    if offset + seq_len > max_pos {
        return Err(shape_error(format!(
            "offset({offset}) + seq_len({seq_len}) = {} exceeds position table rows ({max_pos})",
            offset + seq_len,
        )));
    }
    let start = offset * embed_dim;
    let end = (offset + seq_len) * embed_dim;
    Ok(position_table[start..end].to_vec())
}

/// Compute RoPE-based position encoding (rotation angles).
///
/// Returns `[seq_len, embed_dim]` with pairs `(cos θ, sin θ)` for
/// each dimension pair.
pub fn compute_rope_positions(seq_len: usize, embed_dim: usize, base: f32) -> Vec<f32> {
    let half = embed_dim / 2;
    let mut output = vec![0.0f32; seq_len * embed_dim];
    let d = embed_dim as f32;
    for pos in 0..seq_len {
        let row = pos * embed_dim;
        for i in 0..half {
            let freq = 1.0 / base.powf(2.0 * i as f32 / d);
            let angle = pos as f32 * freq;
            output[row + 2 * i] = angle.cos();
            output[row + 2 * i + 1] = angle.sin();
        }
    }
    output
}

/// Compute position embeddings according to the specified strategy.
pub fn compute_position_embeddings(
    config: &SimdEmbeddingConfig,
    seq_len: usize,
    position_table: Option<&[f32]>,
    offset: usize,
) -> Result<Vec<f32>> {
    match config.position_type {
        PositionEmbeddingType::Sinusoidal { base } => {
            Ok(compute_sinusoidal_positions(seq_len, config.embed_dim, base))
        }
        PositionEmbeddingType::Absolute => {
            let table = position_table.ok_or_else(|| {
                shape_error("absolute position embedding requires a position table".into())
            })?;
            compute_absolute_positions(table, seq_len, config.embed_dim, offset)
        }
        PositionEmbeddingType::Rope { base } => {
            Ok(compute_rope_positions(seq_len, config.embed_dim, base))
        }
    }
}

// ── 4. Embedding Accumulation ──────────────────────────────────────────

/// Accumulate token, position, and optional type embeddings.
///
/// `token_embeddings`, `position_embeddings`, and `type_embeddings` (if
/// provided) all have shape `[seq_len, embed_dim]`. The result is their
/// element-wise sum with SIMD acceleration.
pub fn accumulate_embeddings(
    token_embeddings: &[f32],
    position_embeddings: &[f32],
    type_embeddings: Option<&[f32]>,
    seq_len: usize,
    embed_dim: usize,
) -> Result<Vec<f32>> {
    let expected = seq_len * embed_dim;
    if token_embeddings.len() != expected {
        return Err(shape_error(format!(
            "token_embeddings length {} != seq_len({seq_len}) * embed_dim({embed_dim})",
            token_embeddings.len()
        )));
    }
    if position_embeddings.len() != expected {
        return Err(shape_error(format!(
            "position_embeddings length {} != expected {expected}",
            position_embeddings.len()
        )));
    }
    if let Some(te) = type_embeddings
        && te.len() != expected
    {
        return Err(shape_error(format!(
            "type_embeddings length {} != expected {expected}",
            te.len()
        )));
    }

    let mut output = token_embeddings.to_vec();

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    if use_avx2 {
        #[cfg(target_arch = "x86_64")]
        for row in 0..seq_len {
            let off = row * embed_dim;
            let dst = &mut output[off..off + embed_dim];
            let pos_row = &position_embeddings[off..off + embed_dim];
            unsafe { avx2_add_row(dst, pos_row) };
            if let Some(te) = type_embeddings {
                let type_row = &te[off..off + embed_dim];
                unsafe { avx2_add_row(dst, type_row) };
            }
        }
    } else {
        for (o, &p) in output.iter_mut().zip(position_embeddings.iter()) {
            *o += p;
        }
        if let Some(te) = type_embeddings {
            for (o, &t) in output.iter_mut().zip(te.iter()) {
                *o += t;
            }
        }
    }

    Ok(output)
}

// ── 5. Quantized Embedding Lookup ──────────────────────────────────────

/// Quantize a float embedding table to INT8 with per-row symmetric scale.
///
/// Each row is independently scaled so that `max(abs(row))` maps to 127.
/// Dequantization: `float_val = data[i] * scale`.
pub fn quantize_embedding_int8(
    table: &[f32],
    vocab_size: usize,
    embed_dim: usize,
) -> Int8EmbeddingTable {
    let mut data = vec![0i8; vocab_size * embed_dim];
    let mut scales = vec![0.0f32; vocab_size];
    let mut zero_points = vec![0.0f32; vocab_size];

    for row in 0..vocab_size {
        let start = row * embed_dim;
        let src = &table[start..start + embed_dim];
        let abs_max = src.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
        scales[row] = scale;
        zero_points[row] = 0.0;

        let dst = &mut data[start..start + embed_dim];
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = (s / scale).round().clamp(-128.0, 127.0) as i8;
        }
    }

    Int8EmbeddingTable { data, scales, zero_points, vocab_size, embed_dim }
}

/// Look up embeddings from an INT8 quantized table with on-the-fly dequantization.
pub fn int8_embedding_lookup(packed: &Int8EmbeddingTable, indices: &[u32]) -> Result<Vec<f32>> {
    let dim = packed.embed_dim;
    let vocab = packed.vocab_size;
    let mut output = vec![0.0f32; indices.len() * dim];

    for (i, &idx) in indices.iter().enumerate() {
        if (idx as usize) >= vocab {
            return Err(oob_error(idx, vocab));
        }
        let row = idx as usize;
        let scale = packed.scales[row];
        let zp = packed.zero_points[row];
        let start = row * dim;
        let src = &packed.data[start..start + dim];
        let dst = &mut output[i * dim..i * dim + dim];
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = s as f32 * scale + zp;
        }
    }
    Ok(output)
}

/// Quantize a float embedding table to INT4 with per-row scale and zero-point.
///
/// Two INT4 values are packed per byte (low nibble first). Values are in
/// the range `[0, 15]` which are dequantized via `value * scale + zero_point`.
pub fn quantize_embedding_int4(
    table: &[f32],
    vocab_size: usize,
    embed_dim: usize,
) -> Int4EmbeddingTable {
    let bytes_per_row = embed_dim.div_ceil(2);
    let mut data = vec![0u8; vocab_size * bytes_per_row];
    let mut scales = vec![0.0f32; vocab_size];
    let mut zero_points = vec![0.0f32; vocab_size];

    for row in 0..vocab_size {
        let src_start = row * embed_dim;
        let src = &table[src_start..src_start + embed_dim];
        let min_val = src.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        scales[row] = scale;
        zero_points[row] = min_val;

        let dst_start = row * bytes_per_row;
        let dst = &mut data[dst_start..dst_start + bytes_per_row];
        for (j, &val) in src.iter().enumerate() {
            let q = ((val - min_val) / scale).round().clamp(0.0, 15.0) as u8;
            let byte_idx = j / 2;
            if j % 2 == 0 {
                dst[byte_idx] = (dst[byte_idx] & 0xF0) | (q & 0x0F);
            } else {
                dst[byte_idx] = (dst[byte_idx] & 0x0F) | (q << 4);
            }
        }
    }

    Int4EmbeddingTable { data, scales, zero_points, vocab_size, embed_dim }
}

/// Look up embeddings from an INT4 quantized table with on-the-fly dequantization.
pub fn int4_embedding_lookup(packed: &Int4EmbeddingTable, indices: &[u32]) -> Result<Vec<f32>> {
    let dim = packed.embed_dim;
    let vocab = packed.vocab_size;
    let bytes_per_row = dim.div_ceil(2);
    let mut output = vec![0.0f32; indices.len() * dim];

    for (i, &idx) in indices.iter().enumerate() {
        if (idx as usize) >= vocab {
            return Err(oob_error(idx, vocab));
        }
        let row = idx as usize;
        let scale = packed.scales[row];
        let zp = packed.zero_points[row];
        let src_start = row * bytes_per_row;
        let src = &packed.data[src_start..src_start + bytes_per_row];
        let dst = &mut output[i * dim..i * dim + dim];

        for (j, d) in dst.iter_mut().enumerate() {
            let byte_idx = j / 2;
            let nibble = if j % 2 == 0 { src[byte_idx] & 0x0F } else { src[byte_idx] >> 4 };
            *d = nibble as f32 * scale + zp;
        }
    }
    Ok(output)
}

// ── 6. Embedding Normalization ─────────────────────────────────────────

/// Post-embedding layer normalization (in-place).
///
/// Applies `γ * (x - μ) / √(σ² + ε) + β` per row where `γ` and `β` are
/// the learned weight and bias, and `μ`, `σ²` are computed per row.
pub fn embedding_layer_norm(
    embeddings: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    embed_dim: usize,
    eps: f32,
) -> Result<()> {
    if gamma.len() != embed_dim || beta.len() != embed_dim {
        return Err(shape_error(format!(
            "gamma/beta length ({}/{}) != embed_dim({embed_dim})",
            gamma.len(),
            beta.len(),
        )));
    }
    if embed_dim == 0 || embeddings.is_empty() {
        return Ok(());
    }
    let n_rows = embeddings.len() / embed_dim;
    if embeddings.len() != n_rows * embed_dim {
        return Err(shape_error(format!(
            "embeddings length {} not divisible by embed_dim {embed_dim}",
            embeddings.len()
        )));
    }

    for row in 0..n_rows {
        let off = row * embed_dim;
        let slice = &mut embeddings[off..off + embed_dim];

        // Compute mean.
        let mean: f32 = slice.iter().sum::<f32>() / embed_dim as f32;

        // Compute variance.
        let var: f32 =
            slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / embed_dim as f32;

        let inv_std = 1.0 / (var + eps).sqrt();

        for (j, v) in slice.iter_mut().enumerate() {
            *v = (*v - mean) * inv_std * gamma[j] + beta[j];
        }
    }
    Ok(())
}

/// Post-embedding RMS normalization (in-place).
///
/// Applies `γ * x / √(mean(x²) + ε)` per row.
pub fn embedding_rms_norm(
    embeddings: &mut [f32],
    gamma: &[f32],
    embed_dim: usize,
    eps: f32,
) -> Result<()> {
    if gamma.len() != embed_dim {
        return Err(shape_error(format!("gamma length {} != embed_dim({embed_dim})", gamma.len())));
    }
    if embed_dim == 0 || embeddings.is_empty() {
        return Ok(());
    }

    for chunk in embeddings.chunks_exact_mut(embed_dim) {
        let rms = chunk.iter().map(|&x| x * x).sum::<f32>() / embed_dim as f32;
        let inv_rms = 1.0 / (rms + eps).sqrt();
        for (j, v) in chunk.iter_mut().enumerate() {
            *v = *v * inv_rms * gamma[j];
        }
    }
    Ok(())
}

// ── 7. Vocabulary Projection ───────────────────────────────────────────

/// Project hidden states back to vocabulary logits (reverse embedding).
///
/// Computes `hidden_states @ embedding_table^T` producing logits of shape
/// `[seq_len, vocab_size]`.
///
/// `hidden_states` has shape `[seq_len, embed_dim]`, `table` has shape
/// `[vocab_size, embed_dim]`.
pub fn vocab_projection(
    hidden_states: &[f32],
    table: &[f32],
    seq_len: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<Vec<f32>> {
    if hidden_states.len() != seq_len * embed_dim {
        return Err(shape_error(format!(
            "hidden_states length {} != seq_len({seq_len}) * embed_dim({embed_dim})",
            hidden_states.len()
        )));
    }
    if table.len() < vocab_size * embed_dim {
        return Err(shape_error(format!(
            "table length {} < vocab_size({vocab_size}) * embed_dim({embed_dim})",
            table.len()
        )));
    }

    let mut logits = vec![0.0f32; seq_len * vocab_size];

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    for s in 0..seq_len {
        let h = &hidden_states[s * embed_dim..(s + 1) * embed_dim];
        for v in 0..vocab_size {
            let w = &table[v * embed_dim..(v + 1) * embed_dim];
            let dot = if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    avx2_dot(h, w)
                }
                #[cfg(not(target_arch = "x86_64"))]
                scalar_dot(h, w)
            } else {
                scalar_dot(h, w)
            };
            logits[s * vocab_size + v] = dot;
        }
    }

    Ok(logits)
}

/// Scalar dot product.
fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Vocabulary projection with temperature scaling.
///
/// Same as [`vocab_projection`] but divides every logit by `temperature`.
pub fn vocab_projection_with_temperature(
    hidden_states: &[f32],
    table: &[f32],
    seq_len: usize,
    vocab_size: usize,
    embed_dim: usize,
    temperature: f32,
) -> Result<Vec<f32>> {
    if temperature <= 0.0 {
        return Err(shape_error("temperature must be > 0".into()));
    }
    let mut logits = vocab_projection(hidden_states, table, seq_len, vocab_size, embed_dim)?;
    let inv_t = 1.0 / temperature;
    for v in &mut logits {
        *v *= inv_t;
    }
    Ok(logits)
}

// ── 8. Sparse Embedding ────────────────────────────────────────────────

impl SparseEmbeddingTable {
    /// Create a new sparse embedding table.
    pub fn new(embed_dim: usize) -> Self {
        Self {
            id_to_row: std::collections::HashMap::new(),
            data: Vec::new(),
            embed_dim,
            default_embedding: vec![0.0f32; embed_dim],
        }
    }

    /// Set the default embedding for unmapped token IDs.
    #[must_use]
    pub fn with_default(mut self, default: Vec<f32>) -> Self {
        self.default_embedding = default;
        self
    }

    /// Insert an embedding for a token ID.
    pub fn insert(&mut self, token_id: u32, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.embed_dim {
            return Err(shape_error(format!(
                "embedding length {} != embed_dim({})",
                embedding.len(),
                self.embed_dim,
            )));
        }
        let row_idx = self.id_to_row.len();
        self.id_to_row.insert(token_id, row_idx);
        self.data.extend_from_slice(embedding);
        Ok(())
    }

    /// Number of stored embeddings.
    pub fn len(&self) -> usize {
        self.id_to_row.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_row.is_empty()
    }

    /// Check if a token ID has a stored embedding.
    pub fn contains(&self, token_id: u32) -> bool {
        self.id_to_row.contains_key(&token_id)
    }
}

/// Sparse embedding lookup for large vocabularies.
///
/// Tokens with stored embeddings use the sparse table; others get the
/// default embedding.
pub fn sparse_embedding_lookup(table: &SparseEmbeddingTable, indices: &[u32]) -> Vec<f32> {
    let dim = table.embed_dim;
    let mut output = vec![0.0f32; indices.len() * dim];

    for (i, &idx) in indices.iter().enumerate() {
        let dst = &mut output[i * dim..i * dim + dim];
        if let Some(&row_idx) = table.id_to_row.get(&idx) {
            let src_start = row_idx * dim;
            dst.copy_from_slice(&table.data[src_start..src_start + dim]);
        } else {
            dst.copy_from_slice(&table.default_embedding);
        }
    }

    output
}

/// Sparse embedding lookup with fallback to a dense table.
///
/// For tokens in the sparse table, returns the sparse embedding;
/// for others, falls back to the dense table.
pub fn sparse_embedding_lookup_with_fallback(
    sparse: &SparseEmbeddingTable,
    dense_table: &[f32],
    indices: &[u32],
    vocab_size: usize,
) -> Result<Vec<f32>> {
    let dim = sparse.embed_dim;
    let mut output = vec![0.0f32; indices.len() * dim];

    for (i, &idx) in indices.iter().enumerate() {
        let dst = &mut output[i * dim..i * dim + dim];
        if let Some(&row_idx) = sparse.id_to_row.get(&idx) {
            let src_start = row_idx * dim;
            dst.copy_from_slice(&sparse.data[src_start..src_start + dim]);
        } else {
            if (idx as usize) >= vocab_size {
                return Err(oob_error(idx, vocab_size));
            }
            let src_start = (idx as usize) * dim;
            dst.copy_from_slice(&dense_table[src_start..src_start + dim]);
        }
    }

    Ok(output)
}

// ── Full pipeline ──────────────────────────────────────────────────────

/// Full embedding pipeline: token lookup → position encoding → type
/// embedding → layer norm.
///
/// Combines token embeddings, position embeddings (from the configured
/// strategy), and optional type embeddings, then applies layer
/// normalization.
pub fn embedding_pipeline(
    token_table: &[f32],
    token_ids: &[u32],
    config: &SimdEmbeddingConfig,
    position_table: Option<&[f32]>,
    type_embeddings: Option<&[f32]>,
    ln_gamma: &[f32],
    ln_beta: &[f32],
    position_offset: usize,
) -> Result<Vec<f32>> {
    let seq_len = token_ids.len();
    let dim = config.embed_dim;

    // 1. Token lookup
    let tok_emb = simd_embedding_lookup(token_table, token_ids, config)?;

    // 2. Position embeddings
    let pos_emb = compute_position_embeddings(config, seq_len, position_table, position_offset)?;

    // 3. Accumulate
    let mut combined = accumulate_embeddings(&tok_emb, &pos_emb, type_embeddings, seq_len, dim)?;

    // 4. Layer norm
    embedding_layer_norm(&mut combined, ln_gamma, ln_beta, dim, config.layer_norm_eps)?;

    Ok(combined)
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test helpers ────────────────────────────────────────────────

    /// 4-word vocabulary, dim=4 embedding table.
    fn sample_table_4x4() -> Vec<f32> {
        vec![
            1.0, 2.0, 3.0, 4.0, // idx 0
            5.0, 6.0, 7.0, 8.0, // idx 1
            9.0, 10.0, 11.0, 12.0, // idx 2
            13.0, 14.0, 15.0, 16.0, // idx 3
        ]
    }

    fn sample_config() -> SimdEmbeddingConfig {
        SimdEmbeddingConfig::new(4, 4)
    }

    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "index {i}: {x} vs {y} (diff={})", (x - y).abs());
        }
    }

    // ── 1. SIMD Embedding Lookup Tests ─────────────────────────────

    #[test]
    fn test_simd_lookup_single() {
        let table = sample_table_4x4();
        let cfg = sample_config();
        let result = simd_embedding_lookup(&table, &[0], &cfg).unwrap();
        assert_eq!(result, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_simd_lookup_multiple() {
        let table = sample_table_4x4();
        let cfg = sample_config();
        let result = simd_embedding_lookup(&table, &[0, 2], &cfg).unwrap();
        assert_eq!(result, &[1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_simd_lookup_all_indices() {
        let table = sample_table_4x4();
        let cfg = sample_config();
        let result = simd_embedding_lookup(&table, &[0, 1, 2, 3], &cfg).unwrap();
        assert_eq!(result, table);
    }

    #[test]
    fn test_simd_lookup_duplicates() {
        let table = sample_table_4x4();
        let cfg = sample_config();
        let result = simd_embedding_lookup(&table, &[1, 1], &cfg).unwrap();
        assert_eq!(result, &[5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_simd_lookup_empty_indices() {
        let table = sample_table_4x4();
        let cfg = sample_config();
        let result = simd_embedding_lookup(&table, &[], &cfg).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_simd_lookup_zero_dim() {
        let cfg = SimdEmbeddingConfig::new(4, 0);
        let result = simd_embedding_lookup(&[], &[0], &cfg).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_simd_lookup_oob() {
        let table = sample_table_4x4();
        let cfg = sample_config();
        assert!(simd_embedding_lookup(&table, &[4], &cfg).is_err());
    }

    #[test]
    fn test_simd_lookup_oob_mixed() {
        let table = sample_table_4x4();
        let cfg = sample_config();
        assert!(simd_embedding_lookup(&table, &[0, 99], &cfg).is_err());
    }

    #[test]
    fn test_simd_lookup_padding_idx() {
        let table = sample_table_4x4();
        let cfg = sample_config().with_padding_idx(1);
        let result = simd_embedding_lookup(&table, &[0, 1, 2], &cfg).unwrap();
        assert_eq!(&result[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&result[4..8], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&result[8..12], &[9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_simd_lookup_all_padding() {
        let table = sample_table_4x4();
        let cfg = sample_config().with_padding_idx(0);
        let result = simd_embedding_lookup(&table, &[0, 0, 0], &cfg).unwrap();
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_simd_lookup_large_dim() {
        let dim = 64;
        let vocab = 128;
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.01).collect();
        let cfg = SimdEmbeddingConfig::new(vocab, dim);
        let indices: Vec<u32> = (0..vocab as u32).collect();
        let result = simd_embedding_lookup(&table, &indices, &cfg).unwrap();
        assert_eq!(result, table);
    }

    #[test]
    fn test_simd_lookup_non_multiple_of_8() {
        let dim = 13;
        let vocab = 4;
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
        let cfg = SimdEmbeddingConfig::new(vocab, dim);
        let result = simd_embedding_lookup(&table, &[2], &cfg).unwrap();
        let expected: Vec<f32> = (26..39).map(|i| i as f32).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_simd_lookup_table_too_small() {
        let table = vec![1.0, 2.0]; // only 2 elements
        let cfg = SimdEmbeddingConfig::new(4, 4);
        assert!(simd_embedding_lookup(&table, &[0], &cfg).is_err());
    }

    // ── 2. Batched Embedding Lookup Tests ──────────────────────────

    #[test]
    fn test_batched_single_sequence() {
        let table = sample_table_4x4();
        let cfg = sample_config();
        let result = simd_embedding_lookup_batched(&table, &[&[0, 1]], &cfg).unwrap();
        assert_eq!(result, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_batched_multiple_sequences() {
        let table = sample_table_4x4();
        let cfg = sample_config();
        let result = simd_embedding_lookup_batched(&table, &[&[0], &[2]], &cfg).unwrap();
        assert_eq!(result, &[1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_batched_empty_sequences() {
        let table = sample_table_4x4();
        let cfg = sample_config();
        let result = simd_embedding_lookup_batched(&table, &[&[], &[1], &[]], &cfg).unwrap();
        assert_eq!(result, &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_batched_empty_batch() {
        let table = sample_table_4x4();
        let cfg = sample_config();
        let empty: &[&[u32]] = &[];
        let result = simd_embedding_lookup_batched(&table, empty, &cfg).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_batched_oob() {
        let table = sample_table_4x4();
        let cfg = sample_config();
        assert!(simd_embedding_lookup_batched(&table, &[&[0, 99]], &cfg).is_err());
    }

    #[test]
    fn test_batched_padding() {
        let table = sample_table_4x4();
        let cfg = sample_config().with_padding_idx(0);
        let result = simd_embedding_lookup_batched(&table, &[&[0, 1]], &cfg).unwrap();
        assert_eq!(&result[0..4], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&result[4..8], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_batched_zero_dim() {
        let cfg = SimdEmbeddingConfig::new(4, 0);
        let result = simd_embedding_lookup_batched(&[], &[&[0]], &cfg).unwrap();
        assert!(result.is_empty());
    }

    // ── 3. Position Embedding Tests ────────────────────────────────

    #[test]
    fn test_sinusoidal_positions_at_zero() {
        let pe = compute_sinusoidal_positions(1, 4, 10_000.0);
        // pos=0: sin(0)=0, cos(0)=1
        assert!((pe[0] - 0.0).abs() < 1e-6);
        assert!((pe[1] - 1.0).abs() < 1e-6);
        assert!((pe[2] - 0.0).abs() < 1e-6);
        assert!((pe[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sinusoidal_positions_vary_with_pos() {
        let pe = compute_sinusoidal_positions(3, 4, 10_000.0);
        // Different positions should give different values.
        assert_ne!(&pe[0..4], &pe[4..8]);
        assert_ne!(&pe[4..8], &pe[8..12]);
    }

    #[test]
    fn test_sinusoidal_positions_shape() {
        let pe = compute_sinusoidal_positions(5, 8, 10_000.0);
        assert_eq!(pe.len(), 5 * 8);
    }

    #[test]
    fn test_absolute_positions() {
        let pos_table = vec![
            0.1, 0.2, // pos 0
            0.3, 0.4, // pos 1
            0.5, 0.6, // pos 2
        ];
        let result = compute_absolute_positions(&pos_table, 2, 2, 1).unwrap();
        assert_eq!(result, &[0.3, 0.4, 0.5, 0.6]);
    }

    #[test]
    fn test_absolute_positions_out_of_bounds() {
        let pos_table = vec![0.1, 0.2, 0.3, 0.4];
        assert!(compute_absolute_positions(&pos_table, 3, 2, 0).is_err());
    }

    #[test]
    fn test_absolute_positions_with_offset() {
        let pos_table: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let result = compute_absolute_positions(&pos_table, 2, 4, 2).unwrap();
        let expected: Vec<f32> = (8..16).map(|i| i as f32 * 0.1).collect();
        assert_approx_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_rope_positions_shape() {
        let pe = compute_rope_positions(4, 8, 10_000.0);
        assert_eq!(pe.len(), 4 * 8);
    }

    #[test]
    fn test_rope_positions_at_zero() {
        let pe = compute_rope_positions(1, 4, 10_000.0);
        // At pos=0, angle=0 for all dims: cos(0)=1, sin(0)=0
        assert!((pe[0] - 1.0).abs() < 1e-6); // cos(0)
        assert!((pe[1] - 0.0).abs() < 1e-6); // sin(0)
        assert!((pe[2] - 1.0).abs() < 1e-6); // cos(0)
        assert!((pe[3] - 0.0).abs() < 1e-6); // sin(0)
    }

    #[test]
    fn test_rope_positions_vary_with_pos() {
        let pe = compute_rope_positions(3, 4, 10_000.0);
        assert_ne!(&pe[0..4], &pe[4..8]);
    }

    #[test]
    fn test_compute_position_embeddings_sinusoidal() {
        let cfg = SimdEmbeddingConfig::new(4, 4)
            .with_position_type(PositionEmbeddingType::Sinusoidal { base: 10_000.0 });
        let pe = compute_position_embeddings(&cfg, 3, None, 0).unwrap();
        let expected = compute_sinusoidal_positions(3, 4, 10_000.0);
        assert_eq!(pe, expected);
    }

    #[test]
    fn test_compute_position_embeddings_absolute() {
        let cfg =
            SimdEmbeddingConfig::new(4, 2).with_position_type(PositionEmbeddingType::Absolute);
        let pos_table = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let pe = compute_position_embeddings(&cfg, 2, Some(&pos_table), 1).unwrap();
        assert_eq!(pe, &[0.3, 0.4, 0.5, 0.6]);
    }

    #[test]
    fn test_compute_position_embeddings_absolute_no_table() {
        let cfg =
            SimdEmbeddingConfig::new(4, 2).with_position_type(PositionEmbeddingType::Absolute);
        assert!(compute_position_embeddings(&cfg, 2, None, 0).is_err());
    }

    #[test]
    fn test_compute_position_embeddings_rope() {
        let cfg = SimdEmbeddingConfig::new(4, 4)
            .with_position_type(PositionEmbeddingType::Rope { base: 10_000.0 });
        let pe = compute_position_embeddings(&cfg, 3, None, 0).unwrap();
        let expected = compute_rope_positions(3, 4, 10_000.0);
        assert_eq!(pe, expected);
    }

    // ── 4. Embedding Accumulation Tests ────────────────────────────

    #[test]
    fn test_accumulate_token_and_position() {
        let tok = vec![1.0, 2.0, 3.0, 4.0];
        let pos = vec![0.1, 0.2, 0.3, 0.4];
        let result = accumulate_embeddings(&tok, &pos, None, 1, 4).unwrap();
        assert_approx_eq(&result, &[1.1, 2.2, 3.3, 4.4], 1e-6);
    }

    #[test]
    fn test_accumulate_with_type_embeddings() {
        let tok = vec![1.0, 2.0, 3.0, 4.0];
        let pos = vec![0.1, 0.2, 0.3, 0.4];
        let typ = vec![0.01, 0.02, 0.03, 0.04];
        let result = accumulate_embeddings(&tok, &pos, Some(&typ), 1, 4).unwrap();
        assert_approx_eq(&result, &[1.11, 2.22, 3.33, 4.44], 1e-4);
    }

    #[test]
    fn test_accumulate_multi_row() {
        let tok = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let pos = vec![0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2];
        let result = accumulate_embeddings(&tok, &pos, None, 2, 4).unwrap();
        assert_approx_eq(&result, &[1.1, 2.1, 3.1, 4.1, 5.2, 6.2, 7.2, 8.2], 1e-6);
    }

    #[test]
    fn test_accumulate_mismatched_token() {
        let tok = vec![1.0, 2.0];
        let pos = vec![0.1, 0.2, 0.3, 0.4];
        assert!(accumulate_embeddings(&tok, &pos, None, 1, 4).is_err());
    }

    #[test]
    fn test_accumulate_mismatched_position() {
        let tok = vec![1.0, 2.0, 3.0, 4.0];
        let pos = vec![0.1, 0.2];
        assert!(accumulate_embeddings(&tok, &pos, None, 1, 4).is_err());
    }

    #[test]
    fn test_accumulate_mismatched_type() {
        let tok = vec![1.0, 2.0, 3.0, 4.0];
        let pos = vec![0.1, 0.2, 0.3, 0.4];
        let typ = vec![0.01, 0.02];
        assert!(accumulate_embeddings(&tok, &pos, Some(&typ), 1, 4).is_err());
    }

    // ── 5. Quantized Embedding Tests ───────────────────────────────

    #[test]
    fn test_int8_roundtrip_uniform() {
        let table = vec![0.0, 0.5, 1.0, -1.0, -0.5, 0.0];
        let packed = quantize_embedding_int8(&table, 2, 3);
        assert_eq!(packed.vocab_size, 2);
        assert_eq!(packed.embed_dim, 3);
        let result = int8_embedding_lookup(&packed, &[0, 1]).unwrap();
        assert_approx_eq(&result, &table, 0.02);
    }

    #[test]
    fn test_int8_single_row() {
        let table = vec![1.0, 2.0, 3.0, 4.0];
        let packed = quantize_embedding_int8(&table, 1, 4);
        let result = int8_embedding_lookup(&packed, &[0]).unwrap();
        assert_approx_eq(&result, &table, 0.05);
    }

    #[test]
    fn test_int8_oob() {
        let table = vec![1.0, 2.0, 3.0, 4.0];
        let packed = quantize_embedding_int8(&table, 1, 4);
        assert!(int8_embedding_lookup(&packed, &[1]).is_err());
    }

    #[test]
    fn test_int8_constant_row() {
        let table = vec![5.0, 5.0, 5.0, 5.0];
        let packed = quantize_embedding_int8(&table, 1, 4);
        let result = int8_embedding_lookup(&packed, &[0]).unwrap();
        assert_approx_eq(&result, &table, 0.05);
    }

    #[test]
    fn test_int8_zeros() {
        let table = vec![0.0, 0.0, 0.0, 0.0];
        let packed = quantize_embedding_int8(&table, 1, 4);
        let result = int8_embedding_lookup(&packed, &[0]).unwrap();
        assert_approx_eq(&result, &table, 0.01);
    }

    #[test]
    fn test_int4_roundtrip() {
        let table = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let packed = quantize_embedding_int4(&table, 2, 4);
        assert_eq!(packed.vocab_size, 2);
        assert_eq!(packed.embed_dim, 4);
        let result = int4_embedding_lookup(&packed, &[0, 1]).unwrap();
        assert_approx_eq(&result, &table, 0.5);
    }

    #[test]
    fn test_int4_single_row() {
        let table = vec![0.0, 0.5, 1.0, 1.5];
        let packed = quantize_embedding_int4(&table, 1, 4);
        let result = int4_embedding_lookup(&packed, &[0]).unwrap();
        assert_approx_eq(&result, &table, 0.15);
    }

    #[test]
    fn test_int4_oob() {
        let table = vec![1.0, 2.0, 3.0, 4.0];
        let packed = quantize_embedding_int4(&table, 1, 4);
        assert!(int4_embedding_lookup(&packed, &[1]).is_err());
    }

    #[test]
    fn test_int4_odd_dim() {
        let table = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let packed = quantize_embedding_int4(&table, 1, 5);
        let result = int4_embedding_lookup(&packed, &[0]).unwrap();
        assert_approx_eq(&result, &table, 0.5);
    }

    #[test]
    fn test_int4_negative_values() {
        let table = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0];
        let packed = quantize_embedding_int4(&table, 2, 4);
        let result = int4_embedding_lookup(&packed, &[0, 1]).unwrap();
        assert_approx_eq(&result, &table, 0.5);
    }

    #[test]
    fn test_int8_multiple_rows() {
        let table: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let packed = quantize_embedding_int8(&table, 4, 8);
        let result = int8_embedding_lookup(&packed, &[0, 1, 2, 3]).unwrap();
        assert_approx_eq(&result, &table, 0.05);
    }

    #[test]
    fn test_int4_multiple_rows() {
        let table: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let packed = quantize_embedding_int4(&table, 4, 4);
        let result = int4_embedding_lookup(&packed, &[0, 1, 2, 3]).unwrap();
        assert_approx_eq(&result, &table, 0.6);
    }

    // ── 6. Layer Norm Tests ────────────────────────────────────────

    #[test]
    fn test_layer_norm_identity() {
        // gamma=1, beta=0 with uniform data → mean-centered, unit-variance
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        embedding_layer_norm(&mut data, &gamma, &beta, 4, 1e-5).unwrap();
        let mean: f32 = data.iter().sum::<f32>() / 4.0;
        assert!((mean).abs() < 1e-5, "mean = {mean}");
    }

    #[test]
    fn test_layer_norm_with_beta() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![10.0; 4];
        embedding_layer_norm(&mut data, &gamma, &beta, 4, 1e-5).unwrap();
        let mean: f32 = data.iter().sum::<f32>() / 4.0;
        assert!((mean - 10.0).abs() < 1e-4, "mean = {mean}");
    }

    #[test]
    fn test_layer_norm_with_gamma() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![2.0; 4];
        let beta = vec![0.0; 4];
        embedding_layer_norm(&mut data, &gamma, &beta, 4, 1e-5).unwrap();
        let var: f32 = data.iter().map(|&x| x * x).sum::<f32>() / 4.0;
        // After LN with gamma=2, variance should be ~4 (gamma²)
        assert!((var - 4.0).abs() < 0.1, "var = {var}");
    }

    #[test]
    fn test_layer_norm_multi_row() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        embedding_layer_norm(&mut data, &gamma, &beta, 4, 1e-5).unwrap();
        // Each row should have mean ~0
        let mean1: f32 = data[0..4].iter().sum::<f32>() / 4.0;
        let mean2: f32 = data[4..8].iter().sum::<f32>() / 4.0;
        assert!((mean1).abs() < 1e-5);
        assert!((mean2).abs() < 1e-5);
    }

    #[test]
    fn test_layer_norm_bad_gamma_len() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 4];
        assert!(embedding_layer_norm(&mut data, &gamma, &beta, 4, 1e-5).is_err());
    }

    #[test]
    fn test_layer_norm_bad_beta_len() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 3];
        assert!(embedding_layer_norm(&mut data, &gamma, &beta, 4, 1e-5).is_err());
    }

    #[test]
    fn test_layer_norm_empty() {
        let mut data: Vec<f32> = vec![];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        embedding_layer_norm(&mut data, &gamma, &beta, 4, 1e-5).unwrap();
    }

    #[test]
    fn test_layer_norm_zero_dim() {
        let mut data = vec![1.0, 2.0];
        let gamma: Vec<f32> = vec![];
        let beta: Vec<f32> = vec![];
        embedding_layer_norm(&mut data, &gamma, &beta, 0, 1e-5).unwrap();
        assert_eq!(data, vec![1.0, 2.0]);
    }

    #[test]
    fn test_rms_norm_identity_gamma() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let original = data.clone();
        embedding_rms_norm(&mut data, &gamma, 4, 1e-5).unwrap();
        let rms = (original.iter().map(|&x| x * x).sum::<f32>() / 4.0).sqrt();
        for (i, &v) in data.iter().enumerate() {
            assert!((v - original[i] / rms).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rms_norm_bad_gamma() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 3];
        assert!(embedding_rms_norm(&mut data, &gamma, 4, 1e-5).is_err());
    }

    #[test]
    fn test_rms_norm_empty() {
        let mut data: Vec<f32> = vec![];
        let gamma = vec![1.0; 4];
        embedding_rms_norm(&mut data, &gamma, 4, 1e-5).unwrap();
    }

    // ── 7. Vocabulary Projection Tests ─────────────────────────────

    #[test]
    fn test_vocab_projection_identity() {
        // 2×2 identity-like: hidden=[1,0], table=[[1,0],[0,1]]
        let hidden = vec![1.0, 0.0];
        let table = vec![1.0, 0.0, 0.0, 1.0];
        let logits = vocab_projection(&hidden, &table, 1, 2, 2).unwrap();
        assert_approx_eq(&logits, &[1.0, 0.0], 1e-6);
    }

    #[test]
    fn test_vocab_projection_simple() {
        let hidden = vec![1.0, 2.0, 3.0];
        let table = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2 rows
        let logits = vocab_projection(&hidden, &table, 1, 2, 3).unwrap();
        assert_approx_eq(&logits, &[1.0, 2.0], 1e-6);
    }

    #[test]
    fn test_vocab_projection_multi_seq() {
        let hidden = vec![1.0, 0.0, 0.0, 1.0]; // 2 tokens, dim=2
        let table = vec![1.0, 0.0, 0.0, 1.0]; // vocab=2
        let logits = vocab_projection(&hidden, &table, 2, 2, 2).unwrap();
        assert_approx_eq(&logits, &[1.0, 0.0, 0.0, 1.0], 1e-6);
    }

    #[test]
    fn test_vocab_projection_shape_error_hidden() {
        let hidden = vec![1.0, 2.0]; // wrong size
        let table = vec![1.0, 0.0, 0.0, 1.0];
        assert!(vocab_projection(&hidden, &table, 2, 2, 2).is_err());
    }

    #[test]
    fn test_vocab_projection_shape_error_table() {
        let hidden = vec![1.0, 2.0];
        let table = vec![1.0]; // too small
        assert!(vocab_projection(&hidden, &table, 1, 2, 2).is_err());
    }

    #[test]
    fn test_vocab_projection_with_temperature() {
        let hidden = vec![1.0, 2.0, 3.0];
        let table = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let logits = vocab_projection_with_temperature(&hidden, &table, 1, 2, 3, 2.0).unwrap();
        assert_approx_eq(&logits, &[0.5, 1.0], 1e-6);
    }

    #[test]
    fn test_vocab_projection_temperature_zero() {
        let hidden = vec![1.0, 2.0];
        let table = vec![1.0, 0.0, 0.0, 1.0];
        assert!(vocab_projection_with_temperature(&hidden, &table, 1, 2, 2, 0.0).is_err());
    }

    #[test]
    fn test_vocab_projection_temperature_negative() {
        let hidden = vec![1.0, 2.0];
        let table = vec![1.0, 0.0, 0.0, 1.0];
        assert!(vocab_projection_with_temperature(&hidden, &table, 1, 2, 2, -1.0).is_err());
    }

    #[test]
    fn test_vocab_projection_large() {
        let dim = 16;
        let vocab = 8;
        let seq = 2;
        let hidden: Vec<f32> = (0..seq * dim).map(|i| i as f32 * 0.1).collect();
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.01).collect();
        let logits = vocab_projection(&hidden, &table, seq, vocab, dim).unwrap();
        assert_eq!(logits.len(), seq * vocab);
        // Verify first logit manually.
        let expected_dot: f32 = (0..dim).map(|i| (i as f32 * 0.1) * (i as f32 * 0.01)).sum();
        assert!((logits[0] - expected_dot).abs() < 1e-3);
    }

    // ── 8. Sparse Embedding Tests ──────────────────────────────────

    #[test]
    fn test_sparse_new() {
        let table = SparseEmbeddingTable::new(4);
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert_eq!(table.embed_dim, 4);
    }

    #[test]
    fn test_sparse_insert_and_lookup() {
        let mut table = SparseEmbeddingTable::new(3);
        table.insert(5, &[1.0, 2.0, 3.0]).unwrap();
        table.insert(100, &[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(table.len(), 2);
        assert!(table.contains(5));
        assert!(!table.contains(0));

        let result = sparse_embedding_lookup(&table, &[5, 100]);
        assert_eq!(result, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sparse_default_embedding() {
        let table = SparseEmbeddingTable::new(3);
        let result = sparse_embedding_lookup(&table, &[999]);
        assert_eq!(result, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sparse_custom_default() {
        let table = SparseEmbeddingTable::new(2).with_default(vec![-1.0, -1.0]);
        let result = sparse_embedding_lookup(&table, &[42]);
        assert_eq!(result, &[-1.0, -1.0]);
    }

    #[test]
    fn test_sparse_insert_wrong_dim() {
        let mut table = SparseEmbeddingTable::new(3);
        assert!(table.insert(0, &[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_sparse_mixed_hit_miss() {
        let mut table = SparseEmbeddingTable::new(2);
        table.insert(10, &[1.0, 2.0]).unwrap();
        let result = sparse_embedding_lookup(&table, &[10, 99, 10]);
        assert_eq!(result, &[1.0, 2.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sparse_empty_indices() {
        let table = SparseEmbeddingTable::new(3);
        let result = sparse_embedding_lookup(&table, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sparse_with_fallback() {
        let mut sparse = SparseEmbeddingTable::new(2);
        sparse.insert(0, &[99.0, 99.0]).unwrap();
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 rows × 2 dim
        let result = sparse_embedding_lookup_with_fallback(&sparse, &dense, &[0, 1, 2], 3).unwrap();
        // id=0 → sparse, id=1,2 → dense
        assert_eq!(result, &[99.0, 99.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sparse_with_fallback_oob() {
        let sparse = SparseEmbeddingTable::new(2);
        let dense = vec![1.0, 2.0];
        assert!(sparse_embedding_lookup_with_fallback(&sparse, &dense, &[5], 1).is_err());
    }

    // ── Config builder tests ───────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = SimdEmbeddingConfig::new(1000, 64);
        assert_eq!(cfg.vocab_size, 1000);
        assert_eq!(cfg.embed_dim, 64);
        assert!(cfg.padding_idx.is_none());
        assert_eq!(cfg.max_seq_len, 2048);
        assert!(matches!(cfg.position_type, PositionEmbeddingType::Sinusoidal { .. }));
    }

    #[test]
    fn test_config_with_padding() {
        let cfg = SimdEmbeddingConfig::new(100, 32).with_padding_idx(0);
        assert_eq!(cfg.padding_idx, Some(0));
    }

    #[test]
    fn test_config_with_position_type() {
        let cfg = SimdEmbeddingConfig::new(100, 32)
            .with_position_type(PositionEmbeddingType::Rope { base: 5000.0 });
        assert!(
            matches!(cfg.position_type, PositionEmbeddingType::Rope { base } if (base - 5000.0).abs() < 1e-6)
        );
    }

    #[test]
    fn test_config_with_max_seq_len() {
        let cfg = SimdEmbeddingConfig::new(100, 32).with_max_seq_len(4096);
        assert_eq!(cfg.max_seq_len, 4096);
    }

    #[test]
    fn test_position_embedding_type_default() {
        let pt = PositionEmbeddingType::default();
        assert!(
            matches!(pt, PositionEmbeddingType::Sinusoidal { base } if (base - 10_000.0).abs() < 1e-6)
        );
    }

    // ── Full pipeline test ─────────────────────────────────────────

    #[test]
    fn test_embedding_pipeline_sinusoidal() {
        let table = sample_table_4x4();
        let cfg = sample_config();
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let result =
            embedding_pipeline(&table, &[0, 1], &cfg, None, None, &gamma, &beta, 0).unwrap();
        // Should have 2 rows of dim 4
        assert_eq!(result.len(), 8);
        // Each row should be mean ~0 after layer norm
        let mean: f32 = result[0..4].iter().sum::<f32>() / 4.0;
        assert!((mean).abs() < 1e-4, "mean = {mean}");
    }

    #[test]
    fn test_embedding_pipeline_with_types() {
        let table = sample_table_4x4();
        let cfg = sample_config();
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let type_emb = vec![0.01; 8]; // 2 rows × 4 dim
        let result =
            embedding_pipeline(&table, &[0, 1], &cfg, None, Some(&type_emb), &gamma, &beta, 0)
                .unwrap();
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_embedding_pipeline_absolute_positions() {
        let table = sample_table_4x4();
        let mut cfg = sample_config();
        cfg.position_type = PositionEmbeddingType::Absolute;
        let pos_table = vec![0.1; 4 * 4]; // 4 positions × 4 dim
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let result =
            embedding_pipeline(&table, &[0, 1], &cfg, Some(&pos_table), None, &gamma, &beta, 0)
                .unwrap();
        assert_eq!(result.len(), 8);
    }

    // ── Additional edge-case tests ─────────────────────────────────

    #[test]
    fn test_simd_lookup_reversed_order() {
        let table = sample_table_4x4();
        let cfg = sample_config();
        let result = simd_embedding_lookup(&table, &[3, 2, 1, 0], &cfg).unwrap();
        assert_eq!(
            result,
            &[
                13.0, 14.0, 15.0, 16.0, 9.0, 10.0, 11.0, 12.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0,
                4.0
            ]
        );
    }

    #[test]
    fn test_sparse_overwrite_same_id() {
        let mut table = SparseEmbeddingTable::new(2);
        table.insert(1, &[1.0, 2.0]).unwrap();
        table.insert(1, &[3.0, 4.0]).unwrap();
        // Last insert wins (HashMap overwrite).
        let result = sparse_embedding_lookup(&table, &[1]);
        assert_eq!(result, &[3.0, 4.0]);
    }

    #[test]
    fn test_rms_norm_multi_row() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let gamma = vec![1.0; 4];
        embedding_rms_norm(&mut data, &gamma, 4, 1e-5).unwrap();
        // Each row should have RMS ≈ 1.0 after normalization.
        let rms1 = (data[0..4].iter().map(|&x| x * x).sum::<f32>() / 4.0).sqrt();
        let rms2 = (data[4..8].iter().map(|&x| x * x).sum::<f32>() / 4.0).sqrt();
        assert!((rms1 - 1.0).abs() < 1e-4, "rms1 = {rms1}");
        assert!((rms2 - 1.0).abs() < 1e-4, "rms2 = {rms2}");
    }
}
