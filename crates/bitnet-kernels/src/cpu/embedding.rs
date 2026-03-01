//! CPU SIMD-optimized embedding lookup kernel.
//!
//! Provides embedding table lookups with optional SIMD acceleration for
//! memcpy-style fast paths, weighted accumulation for bag-of-words,
//! L2 normalization, sinusoidal/learned position encoding, max-norm
//! clipping, and 8-bit quantized embedding table support.

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

use bitnet_common::{BitNetError, KernelError, Result};

/// Configuration for an embedding table.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Number of entries (rows) in the embedding table.
    pub vocab_size: usize,
    /// Dimensionality of each embedding vector.
    pub embedding_dim: usize,
    /// Optional padding index whose embedding is always zeros.
    pub padding_idx: Option<u32>,
}

/// Error returned when an index exceeds the vocabulary size.
fn index_out_of_bounds(index: u32, vocab_size: usize) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments {
        reason: format!("embedding index {index} out of bounds for vocab_size {vocab_size}"),
    })
}

// ── Scalar implementations ──────────────────────────────────────────

/// Scalar embedding lookup — always correct, no SIMD.
fn scalar_lookup(
    table: &[f32],
    indices: &[u32],
    embedding_dim: usize,
    padding_idx: Option<u32>,
) -> Result<Vec<f32>> {
    if embedding_dim == 0 {
        return Ok(Vec::new());
    }
    let vocab_size = table.len() / embedding_dim;
    let mut output = vec![0.0f32; indices.len() * embedding_dim];

    for (i, &idx) in indices.iter().enumerate() {
        if Some(idx) == padding_idx {
            // Already zeroed by vec! initialization.
            continue;
        }
        if (idx as usize) >= vocab_size {
            return Err(index_out_of_bounds(idx, vocab_size));
        }
        let src_offset = (idx as usize) * embedding_dim;
        let dst_offset = i * embedding_dim;
        output[dst_offset..dst_offset + embedding_dim]
            .copy_from_slice(&table[src_offset..src_offset + embedding_dim]);
    }
    Ok(output)
}

/// Scalar weighted accumulation of embeddings.
fn scalar_accumulate(
    table: &[f32],
    indices: &[u32],
    weights: &[f32],
    embedding_dim: usize,
) -> Result<Vec<f32>> {
    if embedding_dim == 0 {
        return Ok(Vec::new());
    }
    let vocab_size = table.len() / embedding_dim;
    let mut output = vec![0.0f32; embedding_dim];

    for (&idx, &w) in indices.iter().zip(weights.iter()) {
        if (idx as usize) >= vocab_size {
            return Err(index_out_of_bounds(idx, vocab_size));
        }
        let src_offset = (idx as usize) * embedding_dim;
        for (o, &t) in output.iter_mut().zip(&table[src_offset..src_offset + embedding_dim]) {
            *o += w * t;
        }
    }
    Ok(output)
}

/// Scalar L2 normalization of each embedding vector in-place.
fn scalar_normalize(embeddings: &mut [f32], dim: usize) {
    if dim == 0 {
        return;
    }
    for chunk in embeddings.chunks_exact_mut(dim) {
        let norm_sq: f32 = chunk.iter().map(|&x| x * x).sum();
        if norm_sq > 0.0 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for v in chunk.iter_mut() {
                *v *= inv_norm;
            }
        }
    }
}

// ── AVX2 fast-path helpers (x86_64 only) ────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_accumulate_row(dst: &mut [f32], src: &[f32], weight: f32) {
    let len = dst.len();
    let chunks = len / 8;

    unsafe {
        let w = _mm256_set1_ps(weight);
        for i in 0..chunks {
            let off = i * 8;
            let d = _mm256_loadu_ps(dst.as_ptr().add(off));
            let s = _mm256_loadu_ps(src.as_ptr().add(off));
            _mm256_storeu_ps(dst.as_mut_ptr().add(off), _mm256_add_ps(d, _mm256_mul_ps(s, w)));
        }
    }
    for i in (chunks * 8)..len {
        dst[i] += weight * src[i];
    }
}

/// AVX2-accelerated L2 normalization of a single vector.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_normalize_vector(v: &mut [f32]) {
    let len = v.len();
    let chunks = len / 8;

    let mut norm_sq = unsafe {
        // Compute squared norm.
        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let off = i * 8;
            let x = _mm256_loadu_ps(v.as_ptr().add(off));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(x, x));
        }
        // Horizontal sum of accumulator.
        let hi128 = _mm256_extractf128_ps::<1>(acc);
        let lo128 = _mm256_castps256_ps128(acc);
        let sum4 = _mm_add_ps(hi128, lo128);
        let hi2 = _mm_movehl_ps(sum4, sum4);
        let sum2 = _mm_add_ps(sum4, hi2);
        let hi1 = _mm_shuffle_ps::<0x01>(sum2, sum2);
        _mm_cvtss_f32(_mm_add_ss(sum2, hi1))
    };

    // Scalar tail for the remainder.
    for item in v.iter().take(len).skip(chunks * 8) {
        norm_sq += item * item;
    }

    if norm_sq <= 0.0 {
        return;
    }
    let inv_norm = 1.0 / norm_sq.sqrt();

    unsafe {
        let inv = _mm256_set1_ps(inv_norm);
        for i in 0..chunks {
            let off = i * 8;
            let x = _mm256_loadu_ps(v.as_ptr().add(off));
            _mm256_storeu_ps(v.as_mut_ptr().add(off), _mm256_mul_ps(x, inv));
        }
    }
    for item in v.iter_mut().take(len).skip(chunks * 8) {
        *item *= inv_norm;
    }
}

// ── Public API ──────────────────────────────────────────────────────

/// Look up embeddings by index from a flat embedding table.
///
/// Returns a contiguous `Vec<f32>` of shape `[indices.len(), embedding_dim]`.
/// Out-of-range indices produce an error. If `padding_idx` is set in the
/// config, the corresponding row is filled with zeros.
pub fn embedding_lookup(table: &[f32], indices: &[u32], embedding_dim: usize) -> Result<Vec<f32>> {
    scalar_lookup(table, indices, embedding_dim, None)
}

/// SIMD-accelerated embedding lookup.
///
/// On x86_64 with AVX2, uses wide stores for the copy; otherwise falls back
/// to `copy_from_slice` which the compiler may auto-vectorise.
pub fn embedding_lookup_simd(
    table: &[f32],
    indices: &[u32],
    config: &EmbeddingConfig,
) -> Result<Vec<f32>> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return simd_lookup_avx2(table, indices, config);
        }
    }
    scalar_lookup(table, indices, config.embedding_dim, config.padding_idx)
}

/// AVX2 fast-path for embedding lookup.
#[cfg(target_arch = "x86_64")]
fn simd_lookup_avx2(table: &[f32], indices: &[u32], config: &EmbeddingConfig) -> Result<Vec<f32>> {
    let dim = config.embedding_dim;
    if dim == 0 {
        return Ok(Vec::new());
    }
    let vocab_size = table.len() / dim;
    let mut output = vec![0.0f32; indices.len() * dim];

    for (i, &idx) in indices.iter().enumerate() {
        if Some(idx) == config.padding_idx {
            continue;
        }
        if (idx as usize) >= vocab_size {
            return Err(index_out_of_bounds(idx, vocab_size));
        }
        let src_start = (idx as usize) * dim;
        let dst_start = i * dim;
        // Safety: AVX2 confirmed by caller via runtime check.
        unsafe {
            avx2_copy_f32(
                &table[src_start..src_start + dim],
                &mut output[dst_start..dst_start + dim],
            );
        }
    }
    Ok(output)
}

/// AVX2 memcpy-style copy for f32 slices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_copy_f32(src: &[f32], dst: &mut [f32]) {
    let len = src.len();
    let chunks = len / 8;

    unsafe {
        for i in 0..chunks {
            let off = i * 8;
            let v = _mm256_loadu_ps(src.as_ptr().add(off));
            _mm256_storeu_ps(dst.as_mut_ptr().add(off), v);
        }
    }
    // Scalar tail.
    dst[(chunks * 8)..len].copy_from_slice(&src[(chunks * 8)..len]);
}

/// Weighted accumulation of embeddings (bag-of-words).
///
/// Returns a single embedding vector of length `embedding_dim` that is
/// the weighted sum `∑ weights[i] · table[indices[i]]`.
pub fn embedding_accumulate(
    table: &[f32],
    indices: &[u32],
    weights: &[f32],
    embedding_dim: usize,
) -> Result<Vec<f32>> {
    if indices.len() != weights.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("indices length {} != weights length {}", indices.len(), weights.len()),
        }));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return avx2_accumulate(table, indices, weights, embedding_dim);
        }
    }
    scalar_accumulate(table, indices, weights, embedding_dim)
}

#[cfg(target_arch = "x86_64")]
fn avx2_accumulate(
    table: &[f32],
    indices: &[u32],
    weights: &[f32],
    embedding_dim: usize,
) -> Result<Vec<f32>> {
    if embedding_dim == 0 {
        return Ok(Vec::new());
    }
    let vocab_size = table.len() / embedding_dim;
    let mut output = vec![0.0f32; embedding_dim];

    for (&idx, &w) in indices.iter().zip(weights.iter()) {
        if (idx as usize) >= vocab_size {
            return Err(index_out_of_bounds(idx, vocab_size));
        }
        let src_offset = (idx as usize) * embedding_dim;
        // Safety: AVX2 confirmed by caller.
        unsafe {
            avx2_accumulate_row(&mut output, &table[src_offset..src_offset + embedding_dim], w);
        }
    }
    Ok(output)
}

/// L2-normalize each embedding vector in-place.
///
/// `embeddings` is treated as a flat buffer of vectors each of length `dim`.
/// Zero-norm vectors are left unchanged.
pub fn normalize_embeddings(embeddings: &mut [f32], dim: usize) {
    if dim == 0 || embeddings.is_empty() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            for chunk in embeddings.chunks_exact_mut(dim) {
                // Safety: AVX2 confirmed by runtime check.
                unsafe { avx2_normalize_vector(chunk) };
            }
            return;
        }
    }
    scalar_normalize(embeddings, dim);
}

// ── Extended CPU embedding with position encoding ───────────────

/// Extended configuration for CPU embedding operations with
/// position encoding and norm clipping support.
#[derive(Debug, Clone)]
pub struct CpuEmbeddingConfig {
    /// Number of entries (rows) in the embedding table.
    pub vocab_size: usize,
    /// Dimensionality of each embedding vector.
    pub embed_dim: usize,
    /// Optional padding index whose embedding is always zeros.
    pub padding_idx: Option<u32>,
    /// Optional max-norm constraint for embedding vectors.
    pub max_norm: Option<f32>,
}

impl CpuEmbeddingConfig {
    /// Create a new configuration with required fields.
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        Self { vocab_size, embed_dim, padding_idx: None, max_norm: None }
    }

    /// Set the padding index.
    #[must_use]
    pub fn with_padding_idx(mut self, idx: u32) -> Self {
        self.padding_idx = Some(idx);
        self
    }

    /// Set the max-norm constraint.
    #[must_use]
    pub fn with_max_norm(mut self, norm: f32) -> Self {
        self.max_norm = Some(norm);
        self
    }
}

/// Compute sinusoidal position encoding for a single position.
///
/// Even indices: `sin(pos / 10000^(2i/d))`
/// Odd indices: `cos(pos / 10000^(2i/d))`
fn compute_sinusoidal_pe(pos: usize, embed_dim: usize, output: &mut [f32]) {
    let d = embed_dim as f32;
    for (i, val) in output.iter_mut().enumerate().take(embed_dim) {
        let dim_pair = (i / 2) as f32;
        let angle = (pos as f32) / 10_000f32.powf(2.0 * dim_pair / d);
        *val = if i % 2 == 0 { angle.sin() } else { angle.cos() };
    }
}

/// Look up embeddings and add sinusoidal position encoding.
///
/// Returns `[len(indices), embed_dim]` with each row being
/// `embedding_table[id] + PE(position_offset + i)`.
pub fn embedding_with_position(
    table: &[f32],
    indices: &[u32],
    config: &CpuEmbeddingConfig,
    position_offset: usize,
) -> Result<Vec<f32>> {
    let dim = config.embed_dim;
    if dim == 0 {
        return Ok(Vec::new());
    }

    let mut output = scalar_lookup(table, indices, dim, config.padding_idx)?;
    let mut pe_buf = vec![0.0f32; dim];

    for (i, &idx) in indices.iter().enumerate() {
        if Some(idx) == config.padding_idx {
            continue;
        }
        let pos = position_offset + i;
        compute_sinusoidal_pe(pos, dim, &mut pe_buf);
        let offset = i * dim;
        for (o, &p) in output[offset..offset + dim].iter_mut().zip(pe_buf.iter()) {
            *o += p;
        }
    }

    if let Some(max_norm) = config.max_norm {
        clip_max_norm(&mut output, dim, max_norm);
    }
    Ok(output)
}

/// Look up embeddings and add learned position embeddings.
///
/// `position_table` has shape `[max_positions, embed_dim]`.
/// Position for token `i` is `position_offset + i`.
pub fn embedding_with_learned_position(
    table: &[f32],
    indices: &[u32],
    config: &CpuEmbeddingConfig,
    position_table: &[f32],
    position_offset: usize,
) -> Result<Vec<f32>> {
    let dim = config.embed_dim;
    if dim == 0 {
        return Ok(Vec::new());
    }

    let max_positions = position_table.len() / dim;
    let max_pos_needed = position_offset + indices.len();
    if max_pos_needed > max_positions {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "position_offset({position_offset}) + \
                     n_tokens({}) = {max_pos_needed} exceeds \
                     position_table rows ({max_positions})",
                indices.len(),
            ),
        }));
    }

    let mut output = scalar_lookup(table, indices, dim, config.padding_idx)?;

    for (i, &idx) in indices.iter().enumerate() {
        if Some(idx) == config.padding_idx {
            continue;
        }
        let pos = position_offset + i;
        let pe_start = pos * dim;
        let out_start = i * dim;
        for (o, &p) in output[out_start..out_start + dim]
            .iter_mut()
            .zip(position_table[pe_start..pe_start + dim].iter())
        {
            *o += p;
        }
    }

    if let Some(max_norm) = config.max_norm {
        clip_max_norm(&mut output, dim, max_norm);
    }
    Ok(output)
}

/// Clip each embedding vector to a maximum L2 norm.
fn clip_max_norm(embeddings: &mut [f32], dim: usize, max_norm: f32) {
    if dim == 0 {
        return;
    }
    for chunk in embeddings.chunks_exact_mut(dim) {
        let norm_sq: f32 = chunk.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        if norm > max_norm {
            let scale = max_norm / norm;
            for v in chunk.iter_mut() {
                *v *= scale;
            }
        }
    }
}

/// Apply norm clipping to embedding vectors in-place.
///
/// When `max_norm` is `Some(n)`, vectors exceeding L2 norm `n` are
/// scaled down. When `l2_normalize` is true, all vectors are
/// normalized to unit length after clipping.
pub fn embedding_norm(
    embeddings: &mut [f32],
    dim: usize,
    max_norm: Option<f32>,
    l2_normalize: bool,
) {
    if dim == 0 || embeddings.is_empty() {
        return;
    }
    if let Some(mn) = max_norm {
        clip_max_norm(embeddings, dim, mn);
    }
    if l2_normalize {
        scalar_normalize(embeddings, dim);
    }
}

// ── Quantized embedding packing ─────────────────────────────────

/// An 8-bit quantized embedding table with per-row scales.
#[derive(Debug, Clone)]
pub struct PackedEmbeddingTable {
    /// Quantized embedding values (signed 8-bit).
    pub data: Vec<i8>,
    /// Per-row scale factors for dequantization.
    pub scales: Vec<f32>,
    /// Number of entries (rows).
    pub vocab_size: usize,
    /// Dimensionality of each embedding vector.
    pub embed_dim: usize,
}

/// Pack a float embedding table into 8-bit quantized form.
///
/// Each row is independently scaled so that `max(abs(row))` maps
/// to 127. Dequantization: `float_val = data[i] * scale`.
pub fn pack_embedding_table(
    table: &[f32],
    vocab_size: usize,
    embed_dim: usize,
) -> PackedEmbeddingTable {
    let mut data = vec![0i8; vocab_size * embed_dim];
    let mut scales = vec![0.0f32; vocab_size];

    for (row, scale_out) in scales.iter_mut().enumerate() {
        let start = row * embed_dim;
        let end = start + embed_dim;
        let src = &table[start..end];
        let abs_max = src.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
        *scale_out = scale;

        let dst = &mut data[start..end];
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = (s / scale).round().clamp(-128.0, 127.0) as i8;
        }
    }

    PackedEmbeddingTable { data, scales, vocab_size, embed_dim }
}

/// Look up embeddings from a packed (quantized) table.
///
/// Dequantizes on-the-fly: `output[j] = data[idx][j] * scale[idx]`.
pub fn unpack_embedding_lookup(packed: &PackedEmbeddingTable, indices: &[u32]) -> Result<Vec<f32>> {
    let dim = packed.embed_dim;
    let vocab = packed.vocab_size;
    let mut output = vec![0.0f32; indices.len() * dim];

    for (i, &idx) in indices.iter().enumerate() {
        if (idx as usize) >= vocab {
            return Err(index_out_of_bounds(idx, vocab));
        }
        let row = idx as usize;
        let scale = packed.scales[row];
        let start = row * dim;
        let src = &packed.data[start..start + dim];
        let dst_start = i * dim;
        let dst = &mut output[dst_start..dst_start + dim];
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = s as f32 * scale;
        }
    }

    Ok(output)
}

// ── Convenience wrappers (usize indices) ────────────────────────────

/// Embedding lookup with explicit padding: entries matching `config.padding_idx`
/// are zeroed out in the result.
///
/// `table` is row-major `[vocab_size, embedding_dim]`.
/// Returns `[indices.len(), embedding_dim]`.
pub fn embedding_lookup_with_padding(
    table: &[f32],
    indices: &[usize],
    config: &EmbeddingConfig,
) -> Result<Vec<f32>> {
    let u32_indices: Vec<u32> = indices
        .iter()
        .map(|&i| u32::try_from(i).map_err(|_| index_out_of_bounds(u32::MAX, config.vocab_size)))
        .collect::<Result<_>>()?;
    scalar_lookup(table, &u32_indices, config.embedding_dim, config.padding_idx)
}

/// EmbeddingBag with **sum** reduction.
///
/// `offsets` defines bag boundaries: bag `b` spans
/// `indices[offsets[b]..offsets[b+1]]` (last bag runs to end of `indices`).
/// Returns `[offsets.len(), embedding_dim]`.
pub fn embedding_bag_sum(
    table: &[f32],
    indices: &[usize],
    offsets: &[usize],
    config: &EmbeddingConfig,
) -> Result<Vec<f32>> {
    embedding_bag_reduce(table, indices, offsets, config, BagReduce::Sum)
}

/// EmbeddingBag with **mean** reduction.
///
/// Same semantics as [`embedding_bag_sum`] but each bag is divided by its
/// element count. Empty bags produce zero vectors.
pub fn embedding_bag_mean(
    table: &[f32],
    indices: &[usize],
    offsets: &[usize],
    config: &EmbeddingConfig,
) -> Result<Vec<f32>> {
    embedding_bag_reduce(table, indices, offsets, config, BagReduce::Mean)
}

/// Generate a sinusoidal positional-encoding matrix.
///
/// Returns `[seq_len, embedding_dim]` using the standard formulation:
/// - even columns: `sin(pos / 10000^(2i/d))`
/// - odd columns:  `cos(pos / 10000^(2i/d))`
pub fn positional_embedding(seq_len: usize, embedding_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * embedding_dim];
    for pos in 0..seq_len {
        let start = pos * embedding_dim;
        compute_sinusoidal_pe(pos, embedding_dim, &mut output[start..start + embedding_dim]);
    }
    output
}

// ── Internal bag helpers ────────────────────────────────────────────

enum BagReduce {
    Sum,
    Mean,
}

fn embedding_bag_reduce(
    table: &[f32],
    indices: &[usize],
    offsets: &[usize],
    config: &EmbeddingConfig,
    mode: BagReduce,
) -> Result<Vec<f32>> {
    let dim = config.embedding_dim;
    if offsets.is_empty() {
        return Ok(Vec::new());
    }

    let n_bags = offsets.len();
    let mut output = vec![0.0f32; n_bags * dim];

    for bag in 0..n_bags {
        let start = offsets[bag];
        let end = if bag + 1 < n_bags { offsets[bag + 1] } else { indices.len() };
        if start > end || start > indices.len() {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!("invalid offset {start} for bag {bag}"),
            }));
        }
        let bag_indices = &indices[start..end];
        let bag_len = bag_indices.len();
        let dst_start = bag * dim;

        for &idx in bag_indices {
            if idx >= config.vocab_size {
                return Err(index_out_of_bounds(
                    u32::try_from(idx).unwrap_or(u32::MAX),
                    config.vocab_size,
                ));
            }
            if config.padding_idx.is_some_and(|p| idx == p as usize) {
                continue;
            }
            let src = idx * dim;
            for (o, &t) in output[dst_start..dst_start + dim].iter_mut().zip(&table[src..src + dim])
            {
                *o += t;
            }
        }

        if matches!(mode, BagReduce::Mean) && bag_len > 0 {
            let inv = 1.0 / bag_len as f32;
            for v in &mut output[dst_start..dst_start + dim] {
                *v *= inv;
            }
        }
    }

    Ok(output)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 4-word vocabulary, dim=3 embedding table.
    fn sample_table() -> Vec<f32> {
        vec![
            1.0, 2.0, 3.0, // idx 0
            4.0, 5.0, 6.0, // idx 1
            7.0, 8.0, 9.0, // idx 2
            10.0, 11.0, 12.0, // idx 3
        ]
    }

    fn sample_config(padding_idx: Option<u32>) -> EmbeddingConfig {
        EmbeddingConfig { vocab_size: 4, embedding_dim: 3, padding_idx }
    }

    // ── Basic lookup ────────────────────────────────────────────────

    #[test]
    fn test_basic_lookup() {
        let table = sample_table();
        let result = embedding_lookup(&table, &[0], 3).unwrap();
        assert_eq!(result, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_single_element_lookup() {
        let table = sample_table();
        let result = embedding_lookup(&table, &[3], 3).unwrap();
        assert_eq!(result, &[10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_multiple_indices() {
        let table = sample_table();
        let result = embedding_lookup(&table, &[0, 2, 1], 3).unwrap();
        assert_eq!(result, &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_duplicate_indices() {
        let table = sample_table();
        let result = embedding_lookup(&table, &[1, 1, 1], 3).unwrap();
        assert_eq!(result, &[4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0]);
    }

    // ── Empty inputs ────────────────────────────────────────────────

    #[test]
    fn test_empty_indices() {
        let table = sample_table();
        let result = embedding_lookup(&table, &[], 3).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_zero_dim() {
        let result = embedding_lookup(&[], &[0, 1], 0).unwrap();
        assert!(result.is_empty());
    }

    // ── Out-of-bounds ───────────────────────────────────────────────

    #[test]
    fn test_out_of_bounds_index() {
        let table = sample_table();
        let result = embedding_lookup(&table, &[4], 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_out_of_bounds_mixed() {
        let table = sample_table();
        let result = embedding_lookup(&table, &[0, 99], 3);
        assert!(result.is_err());
    }

    // ── Padding index ───────────────────────────────────────────────

    #[test]
    fn test_padding_idx_returns_zeros() {
        let table = sample_table();
        let config = sample_config(Some(1));
        let result = embedding_lookup_simd(&table, &[0, 1, 2], &config).unwrap();
        assert_eq!(&result[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&result[3..6], &[0.0, 0.0, 0.0]); // padding
        assert_eq!(&result[6..9], &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_padding_idx_all_padding() {
        let table = sample_table();
        let config = sample_config(Some(0));
        let result = embedding_lookup_simd(&table, &[0, 0], &config).unwrap();
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // ── SIMD vs scalar parity ───────────────────────────────────────

    #[test]
    fn test_simd_scalar_parity_small() {
        let table = sample_table();
        let indices = vec![3, 0, 2, 1];
        let config = sample_config(None);
        let scalar = embedding_lookup(&table, &indices, 3).unwrap();
        let simd = embedding_lookup_simd(&table, &indices, &config).unwrap();
        assert_eq!(scalar, simd);
    }

    #[test]
    fn test_simd_scalar_parity_large() {
        // 256-word vocab, dim=64 — exercises AVX2 8-wide loops.
        let dim = 64;
        let vocab = 256;
        let table: Vec<f32> = (0..(vocab * dim)).map(|i| i as f32 * 0.01).collect();
        let indices: Vec<u32> = (0..vocab as u32).collect();
        let config = EmbeddingConfig { vocab_size: vocab, embedding_dim: dim, padding_idx: None };

        let scalar = embedding_lookup(&table, &indices, dim).unwrap();
        let simd = embedding_lookup_simd(&table, &indices, &config).unwrap();
        assert_eq!(scalar, simd);
    }

    #[test]
    fn test_simd_scalar_parity_non_multiple_of_8() {
        // dim=13 ensures we hit the scalar tail in AVX2 code.
        let dim = 13;
        let vocab = 8;
        let table: Vec<f32> = (0..(vocab * dim)).map(|i| i as f32 * 0.1).collect();
        let indices = vec![0, 3, 7, 1];
        let config = EmbeddingConfig { vocab_size: vocab, embedding_dim: dim, padding_idx: None };

        let scalar = embedding_lookup(&table, &indices, dim).unwrap();
        let simd = embedding_lookup_simd(&table, &indices, &config).unwrap();
        assert_eq!(scalar, simd);
    }

    // ── Weighted accumulation ───────────────────────────────────────

    #[test]
    fn test_accumulate_uniform_weights() {
        let table = sample_table();
        let result = embedding_accumulate(&table, &[0, 1], &[1.0, 1.0], 3).unwrap();
        // [1+4, 2+5, 3+6] = [5, 7, 9]
        assert_eq!(result, &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_accumulate_weighted() {
        let table = sample_table();
        let result = embedding_accumulate(&table, &[0, 1], &[2.0, 0.5], 3).unwrap();
        // 2*[1,2,3] + 0.5*[4,5,6] = [2+2, 4+2.5, 6+3] = [4, 6.5, 9]
        assert_eq!(result, &[4.0, 6.5, 9.0]);
    }

    #[test]
    fn test_accumulate_single() {
        let table = sample_table();
        let result = embedding_accumulate(&table, &[2], &[3.0], 3).unwrap();
        assert_eq!(result, &[21.0, 24.0, 27.0]);
    }

    #[test]
    fn test_accumulate_mismatched_lengths() {
        let table = sample_table();
        let result = embedding_accumulate(&table, &[0, 1], &[1.0], 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_accumulate_empty() {
        let table = sample_table();
        let result = embedding_accumulate(&table, &[], &[], 3).unwrap();
        assert_eq!(result, vec![0.0; 3]);
    }

    #[test]
    fn test_accumulate_out_of_bounds() {
        let table = sample_table();
        let result = embedding_accumulate(&table, &[99], &[1.0], 3);
        assert!(result.is_err());
    }

    // ── L2 normalization ────────────────────────────────────────────

    #[test]
    fn test_normalize_unit_vector() {
        let mut data = vec![1.0, 0.0, 0.0];
        normalize_embeddings(&mut data, 3);
        assert_eq!(data, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_uniform() {
        let mut data = vec![3.0, 4.0];
        normalize_embeddings(&mut data, 2);
        let expected_norm: f32 = data.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((expected_norm - 1.0).abs() < 1e-6, "norm = {expected_norm}");
    }

    #[test]
    fn test_normalize_multiple_vectors() {
        let mut data = vec![3.0, 4.0, 0.0, 0.0, 5.0, 0.0];
        normalize_embeddings(&mut data, 2);
        // First: [3/5, 4/5]
        assert!((data[0] - 0.6).abs() < 1e-6);
        assert!((data[1] - 0.8).abs() < 1e-6);
        // Second: [0, 0] — zero vector unchanged
        assert_eq!(data[2], 0.0);
        assert_eq!(data[3], 0.0);
        // Third: [1, 0]
        assert!((data[4] - 1.0).abs() < 1e-6);
        assert_eq!(data[5], 0.0);
    }

    #[test]
    fn test_normalize_zero_vector_unchanged() {
        let mut data = vec![0.0, 0.0, 0.0];
        normalize_embeddings(&mut data, 3);
        assert_eq!(data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_empty() {
        let mut data: Vec<f32> = vec![];
        normalize_embeddings(&mut data, 3);
        assert!(data.is_empty());
    }

    #[test]
    fn test_normalize_zero_dim() {
        let mut data = vec![1.0, 2.0];
        normalize_embeddings(&mut data, 0);
        assert_eq!(data, vec![1.0, 2.0]); // unchanged
    }

    #[test]
    fn test_normalize_large_dim_simd_parity() {
        // dim=64 exercises AVX2 loops; compare against scalar.
        let dim = 64;
        let mut simd_data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let mut scalar_data = simd_data.clone();

        normalize_embeddings(&mut simd_data, dim);
        scalar_normalize(&mut scalar_data, dim);

        for (s, sc) in simd_data.iter().zip(scalar_data.iter()) {
            assert!((s - sc).abs() < 1e-6, "simd={s} scalar={sc}");
        }
    }

    // ── CpuEmbeddingConfig ──────────────────────────────────

    #[test]
    fn test_cpu_config_builder() {
        let cfg = CpuEmbeddingConfig::new(1000, 64);
        assert_eq!(cfg.vocab_size, 1000);
        assert_eq!(cfg.embed_dim, 64);
        assert!(cfg.padding_idx.is_none());
        assert!(cfg.max_norm.is_none());
    }

    #[test]
    fn test_cpu_config_with_padding() {
        let cfg = CpuEmbeddingConfig::new(100, 32).with_padding_idx(0);
        assert_eq!(cfg.padding_idx, Some(0));
    }

    #[test]
    fn test_cpu_config_with_max_norm() {
        let cfg = CpuEmbeddingConfig::new(100, 32).with_max_norm(1.5);
        assert_eq!(cfg.max_norm, Some(1.5));
    }

    // ── Sinusoidal position encoding ────────────────────────

    #[test]
    fn test_sinusoidal_pe_at_zero() {
        let mut pe = vec![0.0f32; 4];
        compute_sinusoidal_pe(0, 4, &mut pe);
        // sin(0) = 0, cos(0) = 1
        assert!((pe[0] - 0.0).abs() < 1e-6);
        assert!((pe[1] - 1.0).abs() < 1e-6);
        assert!((pe[2] - 0.0).abs() < 1e-6);
        assert!((pe[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sinusoidal_pe_varies_with_position() {
        let dim = 8;
        let mut pe0 = vec![0.0f32; dim];
        let mut pe1 = vec![0.0f32; dim];
        compute_sinusoidal_pe(0, dim, &mut pe0);
        compute_sinusoidal_pe(1, dim, &mut pe1);
        assert_ne!(pe0, pe1);
    }

    #[test]
    fn test_sinusoidal_pe_bounded() {
        let dim = 64;
        let mut pe = vec![0.0f32; dim];
        for pos in [0, 1, 50, 1000, 10_000] {
            compute_sinusoidal_pe(pos, dim, &mut pe);
            for &v in &pe {
                assert!((-1.0..=1.0).contains(&v), "PE out of [-1,1] at pos={pos}: {v}");
            }
        }
    }

    // ── embedding_with_position ─────────────────────────────

    #[test]
    fn test_embedding_with_position_basic() {
        // All-zero table so output == PE only.
        let table = vec![0.0f32; 6]; // vocab=2, dim=3
        let cfg = CpuEmbeddingConfig::new(2, 3);
        let result = embedding_with_position(&table, &[0], &cfg, 1).unwrap();

        let mut expected = vec![0.0f32; 3];
        compute_sinusoidal_pe(1, 3, &mut expected);
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_embedding_with_position_adds_to_embedding() {
        let table = vec![
            1.0, 2.0, 3.0, // token 0
            4.0, 5.0, 6.0, // token 1
        ];
        let cfg = CpuEmbeddingConfig::new(2, 3);
        let result = embedding_with_position(&table, &[0], &cfg, 0).unwrap();

        let mut pe = vec![0.0f32; 3];
        compute_sinusoidal_pe(0, 3, &mut pe);
        assert!((result[0] - (1.0 + pe[0])).abs() < 1e-6);
        assert!((result[1] - (2.0 + pe[1])).abs() < 1e-6);
        assert!((result[2] - (3.0 + pe[2])).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_with_position_offset() {
        let table = vec![0.0f32; 4]; // vocab=2, dim=2
        let cfg = CpuEmbeddingConfig::new(2, 2);
        let r0 = embedding_with_position(&table, &[0], &cfg, 0).unwrap();
        let r5 = embedding_with_position(&table, &[0], &cfg, 5).unwrap();
        // Different offsets → different PE.
        assert_ne!(r0, r5);
    }

    #[test]
    fn test_embedding_with_position_padding_zeroed() {
        let table = vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
        ];
        let cfg = CpuEmbeddingConfig::new(2, 2).with_padding_idx(1);
        let result = embedding_with_position(&table, &[0, 1], &cfg, 0).unwrap();
        // Token 1 is padding → stays zero.
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 0.0);
    }

    #[test]
    fn test_embedding_with_position_empty_input() {
        let table = vec![1.0, 2.0];
        let cfg = CpuEmbeddingConfig::new(1, 2);
        let result = embedding_with_position(&table, &[], &cfg, 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_embedding_with_position_batch() {
        let table = vec![0.0f32; 8]; // vocab=4, dim=2
        let cfg = CpuEmbeddingConfig::new(4, 2);
        let result = embedding_with_position(&table, &[0, 1, 2, 3], &cfg, 0).unwrap();
        assert_eq!(result.len(), 8);
        // Each pair should differ due to different positions.
        assert_ne!(&result[0..2], &result[2..4]);
    }

    #[test]
    fn test_embedding_with_position_out_of_range() {
        let table = vec![1.0, 2.0]; // vocab=1, dim=2
        let cfg = CpuEmbeddingConfig::new(1, 2);
        let result = embedding_with_position(&table, &[5], &cfg, 0);
        assert!(result.is_err());
    }

    // ── embedding_with_learned_position ─────────────────────

    #[test]
    fn test_learned_position_basic() {
        let table = vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
        ];
        let pos_table = vec![
            0.1, 0.2, // position 0
            0.3, 0.4, // position 1
        ];
        let cfg = CpuEmbeddingConfig::new(2, 2);
        let result = embedding_with_learned_position(&table, &[0, 1], &cfg, &pos_table, 0).unwrap();
        assert!((result[0] - 1.1).abs() < 1e-6);
        assert!((result[1] - 2.2).abs() < 1e-6);
        assert!((result[2] - 3.3).abs() < 1e-6);
        assert!((result[3] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn test_learned_position_with_offset() {
        let table = vec![1.0, 2.0]; // vocab=1, dim=2
        let pos_table = vec![
            0.1, 0.2, // position 0
            0.3, 0.4, // position 1
            0.5, 0.6, // position 2
        ];
        let cfg = CpuEmbeddingConfig::new(1, 2);
        let result = embedding_with_learned_position(&table, &[0], &cfg, &pos_table, 2).unwrap();
        // Uses position 2.
        assert!((result[0] - 1.5).abs() < 1e-6);
        assert!((result[1] - 2.6).abs() < 1e-6);
    }

    #[test]
    fn test_learned_position_out_of_bounds() {
        let table = vec![1.0, 2.0];
        let pos_table = vec![0.1, 0.2]; // only 1 position
        let cfg = CpuEmbeddingConfig::new(1, 2);
        let result = embedding_with_learned_position(&table, &[0], &cfg, &pos_table, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_learned_position_padding_zeroed() {
        let table = vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
        ];
        let pos_table = vec![0.5, 0.5, 0.5, 0.5];
        let cfg = CpuEmbeddingConfig::new(2, 2).with_padding_idx(0);
        let result = embedding_with_learned_position(&table, &[0, 1], &cfg, &pos_table, 0).unwrap();
        // Token 0 is padding → zeros.
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
        // Token 1 gets embedding + position.
        assert!((result[2] - 3.5).abs() < 1e-6);
        assert!((result[3] - 4.5).abs() < 1e-6);
    }

    // ── embedding_norm ──────────────────────────────────────

    #[test]
    fn test_embedding_norm_max_norm_clips() {
        // Vector [3, 4] has norm 5; clip to 2.5.
        let mut data = vec![3.0, 4.0];
        embedding_norm(&mut data, 2, Some(2.5), false);
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_embedding_norm_no_clip_when_within() {
        // Vector [1, 0] has norm 1; max_norm=5 should not clip.
        let mut data = vec![1.0, 0.0];
        embedding_norm(&mut data, 2, Some(5.0), false);
        assert_eq!(data, vec![1.0, 0.0]);
    }

    #[test]
    fn test_embedding_norm_l2_normalize() {
        let mut data = vec![3.0, 4.0];
        embedding_norm(&mut data, 2, None, true);
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_norm_combined() {
        // Clip to 2.5 then normalize to unit.
        let mut data = vec![3.0, 4.0];
        embedding_norm(&mut data, 2, Some(2.5), true);
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_norm_zero_vector() {
        let mut data = vec![0.0, 0.0, 0.0];
        embedding_norm(&mut data, 3, Some(1.0), true);
        assert_eq!(data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_embedding_norm_empty() {
        let mut data: Vec<f32> = vec![];
        embedding_norm(&mut data, 3, Some(1.0), true);
        assert!(data.is_empty());
    }

    // ── Quantized embedding packing ─────────────────────────

    #[test]
    fn test_pack_embedding_roundtrip() {
        let table = vec![
            1.0, 2.0, 3.0, // token 0
            -1.0, -2.0, -3.0, // token 1
        ];
        let packed = pack_embedding_table(&table, 2, 3);
        let unpacked = unpack_embedding_lookup(&packed, &[0, 1]).unwrap();
        for (orig, reconst) in table.iter().zip(unpacked.iter()) {
            assert!((orig - reconst).abs() < 0.05, "orig={orig} reconst={reconst}");
        }
    }

    #[test]
    fn test_pack_embedding_zero_row() {
        let table = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0];
        let packed = pack_embedding_table(&table, 2, 3);
        let unpacked = unpack_embedding_lookup(&packed, &[0]).unwrap();
        // Zero row stays zero.
        assert!(unpacked.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_pack_embedding_out_of_bounds() {
        let table = vec![1.0, 2.0];
        let packed = pack_embedding_table(&table, 1, 2);
        assert!(unpack_embedding_lookup(&packed, &[5]).is_err());
    }

    #[test]
    fn test_pack_preserves_sign() {
        let table = vec![-10.0, 10.0, -5.0, 5.0];
        let packed = pack_embedding_table(&table, 2, 2);
        let unpacked = unpack_embedding_lookup(&packed, &[0, 1]).unwrap();
        assert!(unpacked[0] < 0.0);
        assert!(unpacked[1] > 0.0);
        assert!(unpacked[2] < 0.0);
        assert!(unpacked[3] > 0.0);
    }

    // ── max_norm via position-encoding path ─────────────────

    #[test]
    fn test_embedding_with_position_max_norm() {
        // Large embeddings + PE; max_norm should clip result.
        let table = vec![100.0, 200.0]; // vocab=1, dim=2
        let cfg = CpuEmbeddingConfig::new(1, 2).with_max_norm(1.0);
        let result = embedding_with_position(&table, &[0], &cfg, 0).unwrap();
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_learned_position_max_norm() {
        let table = vec![50.0, 60.0];
        let pos = vec![10.0, 20.0];
        let cfg = CpuEmbeddingConfig::new(1, 2).with_max_norm(2.0);
        let result = embedding_with_learned_position(&table, &[0], &cfg, &pos, 0).unwrap();
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 2.0).abs() < 1e-4);
    }

    // ── embedding_lookup_with_padding tests ─────────────────────

    #[test]
    fn test_lookup_with_padding_basic() {
        let table = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
        let cfg = EmbeddingConfig { vocab_size: 2, embedding_dim: 2, padding_idx: None };
        let out = embedding_lookup_with_padding(&table, &[0, 1], &cfg).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_lookup_with_padding_zeros_pad() {
        let table = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = EmbeddingConfig { vocab_size: 2, embedding_dim: 2, padding_idx: Some(0) };
        let out = embedding_lookup_with_padding(&table, &[0, 1], &cfg).unwrap();
        assert_eq!(out, vec![0.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn test_lookup_with_padding_all_pad() {
        let table = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = EmbeddingConfig { vocab_size: 2, embedding_dim: 2, padding_idx: Some(1) };
        let out = embedding_lookup_with_padding(&table, &[1, 1], &cfg).unwrap();
        assert_eq!(out, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_lookup_with_padding_empty_indices() {
        let table = vec![1.0, 2.0];
        let cfg = EmbeddingConfig { vocab_size: 1, embedding_dim: 2, padding_idx: None };
        let out = embedding_lookup_with_padding(&table, &[], &cfg).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_lookup_with_padding_oob() {
        let table = vec![1.0, 2.0];
        let cfg = EmbeddingConfig { vocab_size: 1, embedding_dim: 2, padding_idx: None };
        assert!(embedding_lookup_with_padding(&table, &[5], &cfg).is_err());
    }

    #[test]
    fn test_lookup_with_padding_single_element() {
        let table = vec![7.0, 8.0, 9.0];
        let cfg = EmbeddingConfig { vocab_size: 1, embedding_dim: 3, padding_idx: None };
        let out = embedding_lookup_with_padding(&table, &[0], &cfg).unwrap();
        assert_eq!(out, vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_lookup_with_padding_duplicate_indices() {
        let table = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = EmbeddingConfig { vocab_size: 2, embedding_dim: 2, padding_idx: None };
        let out = embedding_lookup_with_padding(&table, &[0, 0, 1], &cfg).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0]);
    }

    // ── embedding_bag_sum tests ─────────────────────────────────

    #[test]
    fn test_bag_sum_single_bag() {
        let table = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
        let cfg = EmbeddingConfig { vocab_size: 2, embedding_dim: 2, padding_idx: None };
        let out = embedding_bag_sum(&table, &[0, 1], &[0], &cfg).unwrap();
        assert_eq!(out, vec![4.0, 6.0]);
    }

    #[test]
    fn test_bag_sum_two_bags() {
        let table = vec![1.0, 10.0, 100.0]; // vocab=3, dim=1
        let cfg = EmbeddingConfig { vocab_size: 3, embedding_dim: 1, padding_idx: None };
        let out = embedding_bag_sum(&table, &[0, 1, 2], &[0, 2], &cfg).unwrap();
        assert_eq!(out, vec![11.0, 100.0]);
    }

    #[test]
    fn test_bag_sum_empty_offsets() {
        let table = vec![1.0];
        let cfg = EmbeddingConfig { vocab_size: 1, embedding_dim: 1, padding_idx: None };
        let out = embedding_bag_sum(&table, &[0], &[], &cfg).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_bag_sum_oob_index() {
        let table = vec![1.0, 2.0];
        let cfg = EmbeddingConfig { vocab_size: 1, embedding_dim: 2, padding_idx: None };
        assert!(embedding_bag_sum(&table, &[5], &[0], &cfg).is_err());
    }

    #[test]
    fn test_bag_sum_with_padding() {
        let table = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
        let cfg = EmbeddingConfig { vocab_size: 2, embedding_dim: 2, padding_idx: Some(0) };
        // Bag contains [0(pad), 1] -> only row 1 contributes
        let out = embedding_bag_sum(&table, &[0, 1], &[0], &cfg).unwrap();
        assert_eq!(out, vec![3.0, 4.0]);
    }

    #[test]
    fn test_bag_sum_single_element_bags() {
        let table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // vocab=3, dim=2
        let cfg = EmbeddingConfig { vocab_size: 3, embedding_dim: 2, padding_idx: None };
        let out = embedding_bag_sum(&table, &[0, 1, 2], &[0, 1, 2], &cfg).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_bag_sum_empty_bag_at_end() {
        // Last bag has no indices (offset == indices.len()).
        let table = vec![1.0, 2.0];
        let cfg = EmbeddingConfig { vocab_size: 1, embedding_dim: 2, padding_idx: None };
        let out = embedding_bag_sum(&table, &[0], &[0, 1], &cfg).unwrap();
        // Bag 0 = [0], Bag 1 = [] (empty)
        assert_eq!(out, vec![1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_bag_sum_all_same_index() {
        let table = vec![2.0, 3.0]; // vocab=1, dim=2
        let cfg = EmbeddingConfig { vocab_size: 1, embedding_dim: 2, padding_idx: None };
        let out = embedding_bag_sum(&table, &[0, 0, 0], &[0], &cfg).unwrap();
        assert_eq!(out, vec![6.0, 9.0]);
    }

    // ── embedding_bag_mean tests ────────────────────────────────

    #[test]
    fn test_bag_mean_single_bag() {
        let table = vec![2.0, 4.0, 6.0, 8.0]; // vocab=2, dim=2
        let cfg = EmbeddingConfig { vocab_size: 2, embedding_dim: 2, padding_idx: None };
        let out = embedding_bag_mean(&table, &[0, 1], &[0], &cfg).unwrap();
        assert_eq!(out, vec![4.0, 6.0]);
    }

    #[test]
    fn test_bag_mean_two_bags() {
        let table = vec![1.0, 3.0, 5.0]; // vocab=3, dim=1
        let cfg = EmbeddingConfig { vocab_size: 3, embedding_dim: 1, padding_idx: None };
        let out = embedding_bag_mean(&table, &[0, 1, 2], &[0, 2], &cfg).unwrap();
        assert_eq!(out, vec![2.0, 5.0]);
    }

    #[test]
    fn test_bag_mean_single_element() {
        let table = vec![10.0, 20.0]; // vocab=1, dim=2
        let cfg = EmbeddingConfig { vocab_size: 1, embedding_dim: 2, padding_idx: None };
        let out = embedding_bag_mean(&table, &[0], &[0], &cfg).unwrap();
        assert_eq!(out, vec![10.0, 20.0]);
    }

    #[test]
    fn test_bag_mean_empty_bag_at_end() {
        let table = vec![4.0];
        let cfg = EmbeddingConfig { vocab_size: 1, embedding_dim: 1, padding_idx: None };
        let out = embedding_bag_mean(&table, &[0], &[0, 1], &cfg).unwrap();
        // Bag 0 = mean([4.0]) = 4.0, Bag 1 = empty → 0.0
        assert_eq!(out, vec![4.0, 0.0]);
    }

    #[test]
    fn test_bag_mean_uniform_values() {
        let table = vec![5.0, 5.0, 5.0]; // vocab=3, dim=1
        let cfg = EmbeddingConfig { vocab_size: 3, embedding_dim: 1, padding_idx: None };
        let out = embedding_bag_mean(&table, &[0, 1, 2], &[0], &cfg).unwrap();
        assert_eq!(out, vec![5.0]);
    }

    #[test]
    fn test_bag_mean_with_padding() {
        let table = vec![10.0, 20.0]; // vocab=2, dim=1
        let cfg = EmbeddingConfig { vocab_size: 2, embedding_dim: 1, padding_idx: Some(0) };
        // Bag has [0(pad), 1] — sum = 20, count = 2 → mean = 10
        let out = embedding_bag_mean(&table, &[0, 1], &[0], &cfg).unwrap();
        assert_eq!(out, vec![10.0]);
    }

    #[test]
    fn test_bag_mean_oob() {
        let table = vec![1.0];
        let cfg = EmbeddingConfig { vocab_size: 1, embedding_dim: 1, padding_idx: None };
        assert!(embedding_bag_mean(&table, &[9], &[0], &cfg).is_err());
    }

    // ── positional_embedding tests ──────────────────────────────

    #[test]
    fn test_positional_embedding_shape() {
        let pe = positional_embedding(5, 8);
        assert_eq!(pe.len(), 40);
    }

    #[test]
    fn test_positional_embedding_position_zero() {
        let pe = positional_embedding(1, 4);
        // pos=0: sin(0)=0, cos(0)=1 for the first pair
        assert!((pe[0] - 0.0).abs() < 1e-6); // sin(0)
        assert!((pe[1] - 1.0).abs() < 1e-6); // cos(0)
    }

    #[test]
    fn test_positional_embedding_distinct_positions() {
        let pe = positional_embedding(3, 4);
        let row0 = &pe[0..4];
        let row1 = &pe[4..8];
        let row2 = &pe[8..12];
        // Each position should differ
        assert_ne!(row0, row1);
        assert_ne!(row1, row2);
        assert_ne!(row0, row2);
    }

    #[test]
    fn test_positional_embedding_zero_seq_len() {
        let pe = positional_embedding(0, 8);
        assert!(pe.is_empty());
    }

    #[test]
    fn test_positional_embedding_zero_dim() {
        let pe = positional_embedding(5, 0);
        assert!(pe.is_empty());
    }

    #[test]
    fn test_positional_embedding_bounded_values() {
        let pe = positional_embedding(100, 64);
        for &v in &pe {
            assert!(v >= -1.0 && v <= 1.0, "PE value {v} out of [-1,1]");
        }
    }

    #[test]
    fn test_positional_embedding_sin_cos_pattern() {
        let pe = positional_embedding(1, 6);
        // Even indices are sin, odd are cos; for pos=0 and first pair:
        // sin(0)=0, cos(0)=1
        assert!((pe[0]).abs() < 1e-6);
        assert!((pe[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_positional_embedding_deterministic() {
        let pe1 = positional_embedding(10, 16);
        let pe2 = positional_embedding(10, 16);
        assert_eq!(pe1, pe2);
    }

    #[test]
    fn test_positional_embedding_dim_1() {
        let pe = positional_embedding(3, 1);
        assert_eq!(pe.len(), 3);
        // dim=1 → only sin column
        assert!((pe[0] - 0.0f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn test_positional_embedding_large_position() {
        let pe = positional_embedding(1000, 4);
        assert_eq!(pe.len(), 4000);
        // Values should still be bounded
        for &v in &pe {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }
}
