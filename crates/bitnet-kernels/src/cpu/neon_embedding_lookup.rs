//! ARM NEON-optimized embedding lookup kernels for Apple Silicon.
//!
//! Provides vectorized embedding table operations using NEON SIMD intrinsics
//! on AArch64: f32 lookup, batched lookup with prefetching, multi-embedding
//! summation, embedding scaling, and quantized i8 table lookup with
//! dequantization.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── f32 Embedding Lookup ────────────────────────────────────────────

/// NEON-accelerated f32 embedding lookup.
///
/// Copies `embedding_dim` floats from `table[index * embedding_dim ..]`
/// into `output` using 4-wide NEON loads/stores.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64)
/// and that `index < vocab_size`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_lookup_f32_single(
    table: &[f32],
    index: usize,
    embedding_dim: usize,
    output: &mut [f32],
) {
    let src = table.as_ptr().add(index * embedding_dim);
    let dst = output.as_mut_ptr();
    let chunks = embedding_dim / 4;
    let remainder = embedding_dim % 4;

    for i in 0..chunks {
        let v = vld1q_f32(src.add(i * 4));
        vst1q_f32(dst.add(i * 4), v);
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        *dst.add(tail + i) = *src.add(tail + i);
    }
}

/// Scalar fallback for f32 embedding lookup.
fn scalar_lookup_f32_single(table: &[f32], index: usize, embedding_dim: usize, output: &mut [f32]) {
    let start = index * embedding_dim;
    output[..embedding_dim].copy_from_slice(&table[start..start + embedding_dim]);
}

/// Look up a single embedding vector from an f32 table.
///
/// Returns a vector of `embedding_dim` floats for the given `index`.
///
/// # Errors
///
/// Returns an error if `index >= vocab_size`.
pub fn embedding_lookup_f32(
    table: &[f32],
    index: u32,
    embedding_dim: usize,
) -> Result<Vec<f32>, String> {
    if embedding_dim == 0 {
        return Ok(Vec::new());
    }
    let vocab_size = table.len() / embedding_dim;
    let idx = index as usize;
    if idx >= vocab_size {
        return Err(format!(
            "embedding index {index} out of bounds for vocab_size {vocab_size}"
        ));
    }
    let mut output = vec![0.0f32; embedding_dim];

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_lookup_f32_single(table, idx, embedding_dim, &mut output);
            }
            return Ok(output);
        }
    }

    scalar_lookup_f32_single(table, idx, embedding_dim, &mut output);
    Ok(output)
}

// ── Batched Embedding Lookup ────────────────────────────────────────

/// NEON-accelerated batched embedding lookup with software prefetching.
///
/// # Safety
///
/// Caller must ensure the target supports NEON and all indices are valid.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_batched_lookup(
    table: &[f32],
    indices: &[u32],
    embedding_dim: usize,
    output: &mut [f32],
) {
    let chunks = embedding_dim / 4;
    let remainder = embedding_dim % 4;

    for (tok_i, &idx) in indices.iter().enumerate() {
        let src = table.as_ptr().add((idx as usize) * embedding_dim);
        let dst = output.as_mut_ptr().add(tok_i * embedding_dim);

        // Prefetch the next embedding row if available.
        if tok_i + 1 < indices.len() {
            let next_src = table
                .as_ptr()
                .add((indices[tok_i + 1] as usize) * embedding_dim);
            #[cfg(target_arch = "aarch64")]
            {
                std::arch::asm!(
                    "prfm pldl1keep, [{addr}]",
                    addr = in(reg) next_src,
                    options(nostack, preserves_flags, readonly)
                );
            }
        }

        for i in 0..chunks {
            let v = vld1q_f32(src.add(i * 4));
            vst1q_f32(dst.add(i * 4), v);
        }

        let tail = chunks * 4;
        for i in 0..remainder {
            *dst.add(tail + i) = *src.add(tail + i);
        }
    }
}

/// Scalar fallback for batched embedding lookup.
fn scalar_batched_lookup(
    table: &[f32],
    indices: &[u32],
    embedding_dim: usize,
    output: &mut [f32],
) {
    for (tok_i, &idx) in indices.iter().enumerate() {
        let src_off = (idx as usize) * embedding_dim;
        let dst_off = tok_i * embedding_dim;
        output[dst_off..dst_off + embedding_dim]
            .copy_from_slice(&table[src_off..src_off + embedding_dim]);
    }
}

/// Look up multiple embeddings in a single call.
///
/// Returns a flat vector of `indices.len() * embedding_dim` floats.
///
/// # Errors
///
/// Returns an error if any index is out of bounds.
pub fn batched_embedding_lookup(
    table: &[f32],
    indices: &[u32],
    embedding_dim: usize,
) -> Result<Vec<f32>, String> {
    if embedding_dim == 0 || indices.is_empty() {
        return Ok(Vec::new());
    }
    let vocab_size = table.len() / embedding_dim;
    for &idx in indices {
        if (idx as usize) >= vocab_size {
            return Err(format!(
                "embedding index {idx} out of bounds for vocab_size {vocab_size}"
            ));
        }
    }
    let mut output = vec![0.0f32; indices.len() * embedding_dim];

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_batched_lookup(table, indices, embedding_dim, &mut output);
            }
            return Ok(output);
        }
    }

    scalar_batched_lookup(table, indices, embedding_dim, &mut output);
    Ok(output)
}

// ── Embedding Sum ───────────────────────────────────────────────────

/// NEON-accelerated element-wise sum of embedding vectors.
///
/// # Safety
///
/// Caller must ensure the target supports NEON and all slices have
/// length `embedding_dim`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_embedding_sum_impl(embeddings: &[&[f32]], embedding_dim: usize, output: &mut [f32]) {
    let chunks = embedding_dim / 4;
    let remainder = embedding_dim % 4;
    let dst = output.as_mut_ptr();

    // Initialize from first embedding.
    let first = embeddings[0].as_ptr();
    for i in 0..chunks {
        let v = vld1q_f32(first.add(i * 4));
        vst1q_f32(dst.add(i * 4), v);
    }
    let tail = chunks * 4;
    for i in 0..remainder {
        *dst.add(tail + i) = *first.add(tail + i);
    }

    // Accumulate remaining embeddings.
    for emb in &embeddings[1..] {
        let src = emb.as_ptr();
        for i in 0..chunks {
            let a = vld1q_f32(dst.add(i * 4));
            let b = vld1q_f32(src.add(i * 4));
            vst1q_f32(dst.add(i * 4), vaddq_f32(a, b));
        }
        for i in 0..remainder {
            *dst.add(tail + i) += *src.add(tail + i);
        }
    }
}

/// Scalar fallback for embedding sum.
fn scalar_embedding_sum(embeddings: &[&[f32]], embedding_dim: usize, output: &mut [f32]) {
    output[..embedding_dim].copy_from_slice(&embeddings[0][..embedding_dim]);
    for emb in &embeddings[1..] {
        for j in 0..embedding_dim {
            output[j] += emb[j];
        }
    }
}

/// Sum multiple embedding vectors element-wise (e.g. token + positional + type).
///
/// All input slices must have length `embedding_dim`.
///
/// # Errors
///
/// Returns an error if `embeddings` is empty or any slice has wrong length.
pub fn embedding_sum(embeddings: &[&[f32]], embedding_dim: usize) -> Result<Vec<f32>, String> {
    if embeddings.is_empty() {
        return Err("embedding_sum: no embeddings provided".to_string());
    }
    if embedding_dim == 0 {
        return Ok(Vec::new());
    }
    for (i, emb) in embeddings.iter().enumerate() {
        if emb.len() != embedding_dim {
            return Err(format!(
                "embedding_sum: embedding[{i}] has length {} but expected {embedding_dim}",
                emb.len()
            ));
        }
    }
    let mut output = vec![0.0f32; embedding_dim];

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_embedding_sum_impl(embeddings, embedding_dim, &mut output);
            }
            return Ok(output);
        }
    }

    scalar_embedding_sum(embeddings, embedding_dim, &mut output);
    Ok(output)
}

// ── Embedding Scale ─────────────────────────────────────────────────

/// NEON-accelerated in-place embedding scaling.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_scale_impl(data: &mut [f32], factor: f32) {
    let len = data.len();
    let chunks = len / 4;
    let remainder = len % 4;
    let ptr = data.as_mut_ptr();
    let vfactor = vdupq_n_f32(factor);

    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        vst1q_f32(ptr.add(i * 4), vmulq_f32(v, vfactor));
    }
    let tail = chunks * 4;
    for i in 0..remainder {
        *ptr.add(tail + i) *= factor;
    }
}

/// Scalar fallback for embedding scaling.
fn scalar_scale(data: &mut [f32], factor: f32) {
    for v in data.iter_mut() {
        *v *= factor;
    }
}

/// Scale an embedding vector by a constant factor (for pre-norm architectures).
///
/// Modifies `embedding` in place and also returns a copy for convenience.
pub fn embedding_scale(embedding: &mut [f32], factor: f32) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_scale_impl(embedding, factor);
            }
            return embedding.to_vec();
        }
    }

    scalar_scale(embedding, factor);
    embedding.to_vec()
}

// ── Packed i8 Embedding Lookup ──────────────────────────────────────

/// NEON-accelerated dequantizing lookup from a quantized i8 table.
///
/// Each i8 value is dequantized as: `f32_val = (i8_val as f32) * scale`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON and `index < vocab_size`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_packed_lookup_i8(
    table: &[i8],
    index: usize,
    embedding_dim: usize,
    scale: f32,
    output: &mut [f32],
) {
    let src = table.as_ptr().add(index * embedding_dim);
    let dst = output.as_mut_ptr();
    let vscale = vdupq_n_f32(scale);
    let chunks = embedding_dim / 8;
    let remainder = embedding_dim % 8;

    for i in 0..chunks {
        let off = i * 8;
        // Load 8 × i8 into a NEON register.
        let raw = vld1_s8(src.add(off));
        // Widen low 4 to i16, then to i32, then to f32.
        let wide16 = vmovl_s8(raw);
        let lo32 = vmovl_s16(vget_low_s16(wide16));
        let hi32 = vmovl_s16(vget_high_s16(wide16));
        let lo_f = vcvtq_f32_s32(lo32);
        let hi_f = vcvtq_f32_s32(hi32);
        vst1q_f32(dst.add(off), vmulq_f32(lo_f, vscale));
        vst1q_f32(dst.add(off + 4), vmulq_f32(hi_f, vscale));
    }

    let tail = chunks * 8;
    for i in 0..remainder {
        *dst.add(tail + i) = (*src.add(tail + i) as f32) * scale;
    }
}

/// Scalar fallback for i8 dequantizing lookup.
fn scalar_packed_lookup_i8(
    table: &[i8],
    index: usize,
    embedding_dim: usize,
    scale: f32,
    output: &mut [f32],
) {
    let start = index * embedding_dim;
    for i in 0..embedding_dim {
        output[i] = (table[start + i] as f32) * scale;
    }
}

/// Look up an embedding from a quantized i8 table and dequantize to f32.
///
/// Each i8 entry is converted via `f32_val = i8_val as f32 * scale`.
///
/// # Errors
///
/// Returns an error if `index >= vocab_size`.
pub fn packed_embedding_lookup_i8(
    table: &[i8],
    index: u32,
    embedding_dim: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    if embedding_dim == 0 {
        return Ok(Vec::new());
    }
    let vocab_size = table.len() / embedding_dim;
    let idx = index as usize;
    if idx >= vocab_size {
        return Err(format!(
            "embedding index {index} out of bounds for vocab_size {vocab_size}"
        ));
    }
    let mut output = vec![0.0f32; embedding_dim];

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_packed_lookup_i8(table, idx, embedding_dim, scale, &mut output);
            }
            return Ok(output);
        }
    }

    scalar_packed_lookup_i8(table, idx, embedding_dim, scale, &mut output);
    Ok(output)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple embedding table where row `i` is filled with `(i+1) as f32`.
    fn make_table(vocab_size: usize, dim: usize) -> Vec<f32> {
        (0..vocab_size)
            .flat_map(|i| std::iter::repeat_n((i + 1) as f32, dim))
            .collect()
    }

    /// Build a ramp table: element `[i][j] = (i * dim + j) as f32`.
    fn make_ramp_table(vocab_size: usize, dim: usize) -> Vec<f32> {
        (0..vocab_size * dim).map(|v| v as f32).collect()
    }

    /// Build a quantized i8 table where row `i` is filled with `(i+1) as i8`.
    fn make_i8_table(vocab_size: usize, dim: usize) -> Vec<i8> {
        (0..vocab_size)
            .flat_map(|i| std::iter::repeat_n(((i + 1) % 127) as i8, dim))
            .collect()
    }

    // ── embedding_lookup_f32 ────────────────────────────────────────

    #[test]
    fn test_lookup_f32_basic() {
        let table = make_table(4, 8);
        let result = embedding_lookup_f32(&table, 0, 8).unwrap();
        assert_eq!(result, vec![1.0; 8]);
    }

    #[test]
    fn test_lookup_f32_last_row() {
        let table = make_table(4, 8);
        let result = embedding_lookup_f32(&table, 3, 8).unwrap();
        assert_eq!(result, vec![4.0; 8]);
    }

    #[test]
    fn test_lookup_f32_out_of_bounds() {
        let table = make_table(4, 8);
        assert!(embedding_lookup_f32(&table, 4, 8).is_err());
    }

    #[test]
    fn test_lookup_f32_zero_dim() {
        let result = embedding_lookup_f32(&[], 0, 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_lookup_f32_single_element() {
        let table = vec![42.0];
        let result = embedding_lookup_f32(&table, 0, 1).unwrap();
        assert_eq!(result, vec![42.0]);
    }

    #[test]
    fn test_lookup_f32_non_aligned_dim() {
        let table = make_ramp_table(3, 5);
        let result = embedding_lookup_f32(&table, 1, 5).unwrap();
        assert_eq!(result, vec![5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_lookup_f32_large_dim() {
        let dim = 1024;
        let table = make_ramp_table(2, dim);
        let result = embedding_lookup_f32(&table, 1, dim).unwrap();
        for (j, &v) in result.iter().enumerate() {
            assert_eq!(v, (dim + j) as f32);
        }
    }

    #[test]
    fn test_lookup_f32_dim_3() {
        let table = make_ramp_table(4, 3);
        let result = embedding_lookup_f32(&table, 2, 3).unwrap();
        assert_eq!(result, vec![6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_lookup_f32_dim_7() {
        let table = make_ramp_table(2, 7);
        let result = embedding_lookup_f32(&table, 1, 7).unwrap();
        assert_eq!(result, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_lookup_f32_neon_scalar_parity() {
        let dim = 33; // intentionally non-aligned
        let table = make_ramp_table(8, dim);
        for idx in 0..8u32 {
            let via_pub = embedding_lookup_f32(&table, idx, dim).unwrap();
            let mut scalar = vec![0.0f32; dim];
            scalar_lookup_f32_single(&table, idx as usize, dim, &mut scalar);
            assert_eq!(via_pub, scalar, "mismatch at index {idx}");
        }
    }

    // ── batched_embedding_lookup ────────────────────────────────────

    #[test]
    fn test_batched_basic() {
        let table = make_table(4, 8);
        let result = batched_embedding_lookup(&table, &[0, 2], 8).unwrap();
        assert_eq!(result.len(), 16);
        assert_eq!(&result[..8], &[1.0; 8]);
        assert_eq!(&result[8..], &[3.0; 8]);
    }

    #[test]
    fn test_batched_empty_indices() {
        let table = make_table(4, 8);
        let result = batched_embedding_lookup(&table, &[], 8).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_batched_zero_dim() {
        let result = batched_embedding_lookup(&[], &[0], 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_batched_single_index() {
        let table = make_table(4, 8);
        let result = batched_embedding_lookup(&table, &[3], 8).unwrap();
        assert_eq!(result, vec![4.0; 8]);
    }

    #[test]
    fn test_batched_out_of_bounds() {
        let table = make_table(4, 8);
        assert!(batched_embedding_lookup(&table, &[0, 5], 8).is_err());
    }

    #[test]
    fn test_batched_repeated_indices() {
        let table = make_table(4, 8);
        let result = batched_embedding_lookup(&table, &[1, 1, 1], 8).unwrap();
        assert_eq!(result.len(), 24);
        for chunk in result.chunks(8) {
            assert_eq!(chunk, &[2.0; 8]);
        }
    }

    #[test]
    fn test_batched_large() {
        let dim = 512;
        let vocab = 1000;
        let table = make_ramp_table(vocab, dim);
        let indices: Vec<u32> = (0..64).collect();
        let result = batched_embedding_lookup(&table, &indices, dim).unwrap();
        assert_eq!(result.len(), 64 * dim);
        for (tok_i, idx) in indices.iter().enumerate() {
            let row = &result[tok_i * dim..(tok_i + 1) * dim];
            for (j, &v) in row.iter().enumerate() {
                assert_eq!(v, (*idx as usize * dim + j) as f32);
            }
        }
    }

    #[test]
    fn test_batched_non_aligned_dim() {
        let table = make_ramp_table(4, 5);
        let result = batched_embedding_lookup(&table, &[0, 3], 5).unwrap();
        assert_eq!(&result[..5], &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&result[5..], &[15.0, 16.0, 17.0, 18.0, 19.0]);
    }

    #[test]
    fn test_batched_neon_scalar_parity() {
        let dim = 17;
        let table = make_ramp_table(10, dim);
        let indices: Vec<u32> = vec![0, 3, 7, 9, 1];
        let via_pub = batched_embedding_lookup(&table, &indices, dim).unwrap();
        let mut scalar = vec![0.0f32; indices.len() * dim];
        scalar_batched_lookup(&table, &indices, dim, &mut scalar);
        assert_eq!(via_pub, scalar);
    }

    #[test]
    fn test_batched_all_same_index() {
        let table = make_table(4, 8);
        let indices = vec![2u32; 16];
        let result = batched_embedding_lookup(&table, &indices, 8).unwrap();
        for chunk in result.chunks(8) {
            assert_eq!(chunk, &[3.0; 8]);
        }
    }

    // ── embedding_sum ───────────────────────────────────────────────

    #[test]
    fn test_sum_two_embeddings() {
        let a = vec![1.0f32; 8];
        let b = vec![2.0f32; 8];
        let result = embedding_sum(&[&a, &b], 8).unwrap();
        assert_eq!(result, vec![3.0; 8]);
    }

    #[test]
    fn test_sum_three_embeddings() {
        let a = vec![1.0f32; 8];
        let b = vec![2.0f32; 8];
        let c = vec![3.0f32; 8];
        let result = embedding_sum(&[&a, &b, &c], 8).unwrap();
        assert_eq!(result, vec![6.0; 8]);
    }

    #[test]
    fn test_sum_single_embedding() {
        let a = vec![5.0f32; 8];
        let result = embedding_sum(&[&a], 8).unwrap();
        assert_eq!(result, vec![5.0; 8]);
    }

    #[test]
    fn test_sum_empty_list() {
        let result = embedding_sum(&[], 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_sum_zero_dim() {
        let a: Vec<f32> = vec![];
        let result = embedding_sum(&[&a[..]], 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_sum_mismatched_length() {
        let a = vec![1.0f32; 8];
        let b = vec![2.0f32; 4];
        assert!(embedding_sum(&[&a, &b], 8).is_err());
    }

    #[test]
    fn test_sum_non_aligned_dim() {
        let a: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let b: Vec<f32> = (10..15).map(|i| i as f32).collect();
        let result = embedding_sum(&[&a, &b], 5).unwrap();
        assert_eq!(result, vec![10.0, 12.0, 14.0, 16.0, 18.0]);
    }

    #[test]
    fn test_sum_large_dim() {
        let dim = 1024;
        let a: Vec<f32> = vec![1.0; dim];
        let b: Vec<f32> = vec![2.0; dim];
        let c: Vec<f32> = vec![3.0; dim];
        let result = embedding_sum(&[&a, &b, &c], dim).unwrap();
        assert_eq!(result, vec![6.0; dim]);
    }

    #[test]
    fn test_sum_neon_scalar_parity() {
        let dim = 19;
        let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..dim).map(|i| i as f32 * 0.2).collect();
        let c: Vec<f32> = (0..dim).map(|i| i as f32 * 0.3).collect();
        let via_pub = embedding_sum(&[&a, &b, &c], dim).unwrap();
        let mut scalar = vec![0.0f32; dim];
        scalar_embedding_sum(&[&a[..], &b[..], &c[..]], dim, &mut scalar);
        for (j, (&a_v, &b_v)) in via_pub.iter().zip(scalar.iter()).enumerate() {
            assert!(
                (a_v - b_v).abs() < 1e-6,
                "mismatch at [{j}]: neon={a_v} scalar={b_v}"
            );
        }
    }

    #[test]
    fn test_sum_negative_values() {
        let a = vec![-1.0f32; 8];
        let b = vec![1.0f32; 8];
        let result = embedding_sum(&[&a, &b], 8).unwrap();
        for &v in &result {
            assert!((v).abs() < 1e-6);
        }
    }

    // ── embedding_scale ─────────────────────────────────────────────

    #[test]
    fn test_scale_basic() {
        let mut emb = vec![2.0f32; 8];
        let result = embedding_scale(&mut emb, 3.0);
        assert_eq!(result, vec![6.0; 8]);
        assert_eq!(emb, vec![6.0; 8]);
    }

    #[test]
    fn test_scale_zero() {
        let mut emb = vec![5.0f32; 8];
        let result = embedding_scale(&mut emb, 0.0);
        assert_eq!(result, vec![0.0; 8]);
    }

    #[test]
    fn test_scale_one() {
        let mut emb = vec![3.0f32; 8];
        let result = embedding_scale(&mut emb, 1.0);
        assert_eq!(result, vec![3.0; 8]);
    }

    #[test]
    fn test_scale_negative() {
        let mut emb = vec![2.0f32; 8];
        let result = embedding_scale(&mut emb, -1.0);
        assert_eq!(result, vec![-2.0; 8]);
    }

    #[test]
    fn test_scale_empty() {
        let mut emb: Vec<f32> = vec![];
        let result = embedding_scale(&mut emb, 5.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_scale_non_aligned() {
        let mut emb: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let result = embedding_scale(&mut emb, 2.0);
        assert_eq!(result, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_scale_large() {
        let dim = 1024;
        let mut emb = vec![1.0f32; dim];
        let result = embedding_scale(&mut emb, 0.5);
        assert_eq!(result, vec![0.5; dim]);
    }

    #[test]
    fn test_scale_fractional() {
        let mut emb = vec![10.0f32; 4];
        let result = embedding_scale(&mut emb, 0.1);
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_scale_neon_scalar_parity() {
        let dim = 33;
        let factor = 1.5f32;
        let original: Vec<f32> = (0..dim).map(|i| i as f32 * 0.7).collect();

        let mut neon_copy = original.clone();
        let neon_result = embedding_scale(&mut neon_copy, factor);

        let mut scalar_copy = original.clone();
        scalar_scale(&mut scalar_copy, factor);

        for (j, (&a, &b)) in neon_result.iter().zip(scalar_copy.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "mismatch at [{j}]: neon={a} scalar={b}"
            );
        }
    }

    // ── packed_embedding_lookup_i8 ──────────────────────────────────

    #[test]
    fn test_i8_lookup_basic() {
        let table = make_i8_table(4, 8);
        let result = packed_embedding_lookup_i8(&table, 0, 8, 1.0).unwrap();
        assert_eq!(result, vec![1.0; 8]);
    }

    #[test]
    fn test_i8_lookup_with_scale() {
        let table = make_i8_table(4, 8);
        let result = packed_embedding_lookup_i8(&table, 0, 8, 0.5).unwrap();
        assert_eq!(result, vec![0.5; 8]);
    }

    #[test]
    fn test_i8_lookup_last_row() {
        let table = make_i8_table(4, 8);
        let result = packed_embedding_lookup_i8(&table, 3, 8, 1.0).unwrap();
        assert_eq!(result, vec![4.0; 8]);
    }

    #[test]
    fn test_i8_lookup_out_of_bounds() {
        let table = make_i8_table(4, 8);
        assert!(packed_embedding_lookup_i8(&table, 4, 8, 1.0).is_err());
    }

    #[test]
    fn test_i8_lookup_zero_dim() {
        let result = packed_embedding_lookup_i8(&[], 0, 0, 1.0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_i8_lookup_single_element() {
        let table = vec![42i8];
        let result = packed_embedding_lookup_i8(&table, 0, 1, 1.0).unwrap();
        assert_eq!(result, vec![42.0]);
    }

    #[test]
    fn test_i8_lookup_negative_values() {
        let table = vec![-10i8; 8];
        let result = packed_embedding_lookup_i8(&table, 0, 8, 1.0).unwrap();
        assert_eq!(result, vec![-10.0; 8]);
    }

    #[test]
    fn test_i8_lookup_non_aligned_dim() {
        let dim = 5;
        let table: Vec<i8> = (0..20).map(|v| (v % 127) as i8).collect();
        let result = packed_embedding_lookup_i8(&table, 1, dim, 2.0).unwrap();
        let expected: Vec<f32> = (5..10).map(|v| v as f32 * 2.0).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_i8_lookup_large_dim() {
        let dim = 1024;
        let table: Vec<i8> = (0..dim * 2).map(|v| ((v % 127) as i8)).collect();
        let result = packed_embedding_lookup_i8(&table, 1, dim, 0.1).unwrap();
        assert_eq!(result.len(), dim);
        for (j, &v) in result.iter().enumerate() {
            let expected = ((dim + j) % 127) as f32 * 0.1;
            assert!(
                (v - expected).abs() < 1e-5,
                "mismatch at [{j}]: got={v} expected={expected}"
            );
        }
    }

    #[test]
    fn test_i8_lookup_neon_scalar_parity() {
        let dim = 33;
        let scale = 0.25f32;
        let table: Vec<i8> = (0..10 * dim)
            .map(|v| ((v as i32 % 255) - 128) as i8)
            .collect();
        for idx in 0..10u32 {
            let via_pub = packed_embedding_lookup_i8(&table, idx, dim, scale).unwrap();
            let mut scalar = vec![0.0f32; dim];
            scalar_packed_lookup_i8(&table, idx as usize, dim, scale, &mut scalar);
            for (j, (&a, &b)) in via_pub.iter().zip(scalar.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "mismatch at idx={idx} [{j}]: neon={a} scalar={b}"
                );
            }
        }
    }

    #[test]
    fn test_i8_lookup_dim_7() {
        let table: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 10, 20, 30, 40, 50, 60, 70];
        let result = packed_embedding_lookup_i8(&table, 1, 7, 1.0).unwrap();
        assert_eq!(result, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]);
    }

    #[test]
    fn test_i8_lookup_extreme_scale() {
        let table = vec![100i8; 8];
        let result = packed_embedding_lookup_i8(&table, 0, 8, 1000.0).unwrap();
        assert_eq!(result, vec![100_000.0; 8]);
    }

    #[test]
    fn test_i8_lookup_zero_scale() {
        let table = vec![100i8; 8];
        let result = packed_embedding_lookup_i8(&table, 0, 8, 0.0).unwrap();
        assert_eq!(result, vec![0.0; 8]);
    }

    #[test]
    fn test_i8_lookup_negative_scale() {
        let table = vec![10i8; 8];
        let result = packed_embedding_lookup_i8(&table, 0, 8, -1.0).unwrap();
        assert_eq!(result, vec![-10.0; 8]);
    }

    // ── Cross-function integration tests ────────────────────────────

    #[test]
    fn test_lookup_then_scale() {
        let table = make_table(4, 8);
        let mut emb = embedding_lookup_f32(&table, 2, 8).unwrap();
        assert_eq!(emb, vec![3.0; 8]);
        let scaled = embedding_scale(&mut emb, 2.0);
        assert_eq!(scaled, vec![6.0; 8]);
    }

    #[test]
    fn test_lookup_then_sum() {
        let table = make_table(4, 8);
        let tok = embedding_lookup_f32(&table, 0, 8).unwrap();
        let pos = embedding_lookup_f32(&table, 1, 8).unwrap();
        let result = embedding_sum(&[&tok, &pos], 8).unwrap();
        assert_eq!(result, vec![3.0; 8]);
    }

    #[test]
    fn test_batched_then_sum() {
        let table = make_table(4, 8);
        let batch = batched_embedding_lookup(&table, &[0, 1], 8).unwrap();
        let tok = &batch[..8];
        let pos = &batch[8..];
        let result = embedding_sum(&[tok, pos], 8).unwrap();
        assert_eq!(result, vec![3.0; 8]);
    }

    #[test]
    fn test_i8_lookup_then_scale() {
        let table = make_i8_table(4, 8);
        let mut emb = packed_embedding_lookup_i8(&table, 1, 8, 0.5).unwrap();
        assert_eq!(emb, vec![1.0; 8]);
        let scaled = embedding_scale(&mut emb, 3.0);
        assert_eq!(scaled, vec![3.0; 8]);
    }

    #[test]
    fn test_sum_i8_and_f32() {
        let f32_table = make_table(4, 8);
        let i8_table = make_i8_table(4, 8);
        let tok = embedding_lookup_f32(&f32_table, 0, 8).unwrap();
        let pos = packed_embedding_lookup_i8(&i8_table, 1, 8, 1.0).unwrap();
        let result = embedding_sum(&[&tok, &pos], 8).unwrap();
        assert_eq!(result, vec![3.0; 8]);
    }

    // ── Stress / edge-case tests ────────────────────────────────────

    #[test]
    fn test_batched_large_vocab() {
        let vocab = 50_000;
        let dim = 128;
        let table = make_ramp_table(vocab, dim);
        let indices: Vec<u32> = vec![0, 49_999, 25_000];
        let result = batched_embedding_lookup(&table, &indices, dim).unwrap();
        assert_eq!(result.len(), 3 * dim);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[dim], (49_999 * dim) as f32);
    }

    #[test]
    fn test_scale_preserves_nan() {
        let mut emb = vec![f32::NAN; 4];
        let result = embedding_scale(&mut emb, 2.0);
        for v in &result {
            assert!(v.is_nan());
        }
    }

    #[test]
    fn test_scale_inf() {
        let mut emb = vec![f32::INFINITY; 4];
        let result = embedding_scale(&mut emb, 2.0);
        assert_eq!(result, vec![f32::INFINITY; 4]);
    }

    #[test]
    fn test_lookup_f32_dim_1() {
        let table = vec![10.0, 20.0, 30.0];
        let result = embedding_lookup_f32(&table, 2, 1).unwrap();
        assert_eq!(result, vec![30.0]);
    }

    #[test]
    fn test_batched_dim_1() {
        let table = vec![10.0, 20.0, 30.0];
        let result = batched_embedding_lookup(&table, &[2, 0, 1], 1).unwrap();
        assert_eq!(result, vec![30.0, 10.0, 20.0]);
    }

    #[test]
    fn test_i8_min_max_values() {
        let table = vec![i8::MIN, i8::MAX, 0, 1, -1, 50, -50, 127];
        let result = packed_embedding_lookup_i8(&table, 0, 8, 1.0).unwrap();
        assert_eq!(result[0], -128.0);
        assert_eq!(result[1], 127.0);
        assert_eq!(result[2], 0.0);
    }

    #[test]
    fn test_sum_many_embeddings() {
        let dim = 16;
        let count = 10;
        let vecs: Vec<Vec<f32>> = (0..count).map(|_| vec![1.0; dim]).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let result = embedding_sum(&refs, dim).unwrap();
        assert_eq!(result, vec![count as f32; dim]);
    }
}
