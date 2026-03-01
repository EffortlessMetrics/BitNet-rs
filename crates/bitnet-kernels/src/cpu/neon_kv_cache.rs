//! ARM NEON optimized KV cache operations for Apple Silicon.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Fast cache line copy using NEON vld1q/vst1q intrinsics.
/// Copies `len` elements from `src[offset..]` into `dst[..len]`.
#[cfg(target_arch = "aarch64")]
pub fn neon_kv_cache_copy_f32(src: &[f32], dst: &mut [f32], offset: usize, len: usize) {
    assert!(
        offset + len <= src.len(),
        "source range out of bounds: offset={offset} len={len} src.len={}",
        src.len()
    );
    assert!(len <= dst.len(), "destination too small: len={len} dst.len={}", dst.len());

    let src = &src[offset..offset + len];
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let base = i * 4;
        unsafe {
            let v = vld1q_f32(src.as_ptr().add(base));
            vst1q_f32(dst.as_mut_ptr().add(base), v);
        }
    }

    let tail_start = chunks * 4;
    dst[tail_start..tail_start + remainder]
        .copy_from_slice(&src[tail_start..tail_start + remainder]);
}

/// In-place scale of cached f32 values using NEON vmulq.
#[cfg(target_arch = "aarch64")]
pub fn neon_kv_cache_scale_f32(data: &mut [f32], scale: f32) {
    let len = data.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let scale_vec = unsafe { vdupq_n_f32(scale) };

    for i in 0..chunks {
        let base = i * 4;
        unsafe {
            let v = vld1q_f32(data.as_ptr().add(base));
            let scaled = vmulq_f32(v, scale_vec);
            vst1q_f32(data.as_mut_ptr().add(base), scaled);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        data[tail_start + i] *= scale;
    }
}

/// Concatenate existing KV cache entries with new entries into `output`.
/// `output` must have length >= `existing.len() + new.len()`.
#[cfg(target_arch = "aarch64")]
pub fn neon_kv_concat_f32(existing: &[f32], new: &[f32], output: &mut [f32]) {
    let total = existing.len() + new.len();
    assert!(output.len() >= total, "output too small: need {total}, got {}", output.len());

    if !existing.is_empty() {
        neon_kv_cache_copy_f32(existing, output, 0, existing.len());
    }

    let dst = &mut output[existing.len()..existing.len() + new.len()];
    if !new.is_empty() {
        neon_kv_cache_copy_f32(new, dst, 0, new.len());
    }
}

/// Compute attention scores Q·K^T for cached keys.
///
/// - `query`: shape `[head_dim]` — a single query vector
/// - `key_cache`: shape `[num_positions * head_dim]` — flattened key vectors
/// - `scores`: shape `[num_positions]` — output dot-product scores
///
/// Each score\[i\] = dot(query, key\[i\]) / sqrt(head_dim).
#[cfg(target_arch = "aarch64")]
pub fn neon_kv_attention_score_f32(
    query: &[f32],
    key_cache: &[f32],
    head_dim: usize,
    num_positions: usize,
    scores: &mut [f32],
) {
    assert!(query.len() >= head_dim, "query too short: need {head_dim}, got {}", query.len());
    assert!(
        key_cache.len() >= num_positions * head_dim,
        "key_cache too small: need {}, got {}",
        num_positions * head_dim,
        key_cache.len()
    );
    assert!(
        scores.len() >= num_positions,
        "scores too small: need {num_positions}, got {}",
        scores.len()
    );

    let inv_sqrt = 1.0 / (head_dim as f32).sqrt();

    for (pos, score) in scores.iter_mut().enumerate().take(num_positions) {
        let key_offset = pos * head_dim;
        let key = &key_cache[key_offset..key_offset + head_dim];

        let chunks = head_dim / 4;
        let remainder = head_dim % 4;

        let mut acc = unsafe { vdupq_n_f32(0.0) };

        for c in 0..chunks {
            let base = c * 4;
            unsafe {
                let q = vld1q_f32(query.as_ptr().add(base));
                let k = vld1q_f32(key.as_ptr().add(base));
                acc = vfmaq_f32(acc, q, k);
            }
        }

        let mut dot: f32 = unsafe { vaddvq_f32(acc) };

        let tail_start = chunks * 4;
        for i in 0..remainder {
            dot += query[tail_start + i] * key[tail_start + i];
        }

        *score = dot * inv_sqrt;
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    #[test]
    fn test_cache_copy_basic() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dst = vec![0.0f32; 8];
        neon_kv_cache_copy_f32(&src, &mut dst, 0, 8);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_cache_copy_partial() {
        let src = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut dst = vec![0.0f32; 3];
        neon_kv_cache_copy_f32(&src, &mut dst, 2, 3);
        assert_eq!(dst, vec![30.0, 40.0, 50.0]);
    }

    #[test]
    fn test_cache_scale() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        neon_kv_cache_scale_f32(&mut data, 0.5);
        assert_eq!(data, vec![0.5, 1.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    fn test_concat_basic() {
        let existing = vec![1.0f32, 2.0, 3.0, 4.0];
        let new = vec![5.0f32, 6.0, 7.0];
        let mut output = vec![0.0f32; 7];
        neon_kv_concat_f32(&existing, &new, &mut output);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_attention_scores() {
        let head_dim = 4;
        let num_positions = 4;
        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let key_cache = vec![
            1.0f32, 0.0, 0.0, 0.0, // pos0: dot=1
            0.0, 1.0, 0.0, 0.0, // pos1: dot=0
            0.5, 0.5, 0.0, 0.0, // pos2: dot=0.5
            2.0, 0.0, 0.0, 0.0, // pos3: dot=2
        ];
        let mut scores = vec![0.0f32; num_positions];
        neon_kv_attention_score_f32(&query, &key_cache, head_dim, num_positions, &mut scores);

        let inv_sqrt = 1.0 / (head_dim as f32).sqrt(); // 0.5
        let expected = [1.0 * inv_sqrt, 0.0, 0.5 * inv_sqrt, 2.0 * inv_sqrt];
        for (s, e) in scores.iter().zip(expected.iter()) {
            assert!((s - e).abs() < 1e-6, "score {s} != expected {e}");
        }
    }

    #[test]
    fn test_empty_cache() {
        let existing: Vec<f32> = vec![];
        let new = vec![1.0f32, 2.0];
        let mut output = vec![0.0f32; 2];
        neon_kv_concat_f32(&existing, &new, &mut output);
        assert_eq!(output, vec![1.0, 2.0]);

        let src = vec![1.0f32];
        let mut dst: Vec<f32> = vec![];
        neon_kv_cache_copy_f32(&src, &mut dst, 0, 0);
        assert!(dst.is_empty());

        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let key_cache: Vec<f32> = vec![];
        let mut scores: Vec<f32> = vec![];
        neon_kv_attention_score_f32(&query, &key_cache, 4, 0, &mut scores);
        assert!(scores.is_empty());
    }
}
