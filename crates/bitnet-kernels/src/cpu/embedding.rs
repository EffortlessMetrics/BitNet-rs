//! CPU SIMD-optimized embedding lookup kernel.
//!
//! Provides embedding table lookups with optional SIMD acceleration for
//! memcpy-style fast paths, weighted accumulation for bag-of-words, and
//! L2 normalization of embedding vectors.

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
}
