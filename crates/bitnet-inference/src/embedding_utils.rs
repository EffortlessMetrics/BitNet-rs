//! Embedding table utilities for model inference.
//!
//! Token embedding lookup, position embedding, and tied embeddings.

/// Look up token embeddings from a flat weight table.
pub fn lookup_embeddings(weight: &[f32], token_ids: &[u32], hidden_size: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(token_ids.len() * hidden_size);
    for &id in token_ids {
        let start = id as usize * hidden_size;
        let end = start + hidden_size;
        if end <= weight.len() {
            out.extend_from_slice(&weight[start..end]);
        } else {
            // OOV: zero embedding
            out.extend(std::iter::repeat(0.0f32).take(hidden_size));
        }
    }
    out
}

/// Apply RoPE-style sinusoidal position embeddings in-place.
/// Modifies `embeddings` which is [seq_len, hidden_size].
pub fn add_sinusoidal_positions(
    embeddings: &mut [f32],
    seq_len: usize,
    hidden_size: usize,
    base: f32,
) {
    for pos in 0..seq_len {
        for i in 0..hidden_size / 2 {
            let angle = (pos as f32) / base.powf(2.0 * i as f32 / hidden_size as f32);
            let sin_val = angle.sin();
            let cos_val = angle.cos();
            let idx = pos * hidden_size;
            embeddings[idx + 2 * i] += sin_val;
            if 2 * i + 1 < hidden_size {
                embeddings[idx + 2 * i + 1] += cos_val;
            }
        }
    }
}

/// Check if an embedding table is tied to the output projection.
pub fn check_tied_embeddings(
    embed_weight: &[f32],
    lm_head_weight: &[f32],
    sample_size: usize,
) -> bool {
    if embed_weight.len() != lm_head_weight.len() {
        return false;
    }
    let check = sample_size.min(embed_weight.len());
    embed_weight[..check] == lm_head_weight[..check]
}

/// Compute embedding table statistics.
#[derive(Debug, Clone)]
pub struct EmbeddingStats {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub mean: f32,
    pub std_dev: f32,
    pub zero_rows: usize,
}

pub fn embedding_stats(weight: &[f32], vocab_size: usize, hidden_size: usize) -> EmbeddingStats {
    if vocab_size == 0 || hidden_size == 0 || weight.len() < vocab_size * hidden_size {
        return EmbeddingStats { vocab_size, hidden_size, mean: 0.0, std_dev: 0.0, zero_rows: 0 };
    }

    let total = (vocab_size * hidden_size) as f64;
    let sum: f64 = weight[..vocab_size * hidden_size].iter().map(|&v| v as f64).sum();
    let mean = sum / total;

    let var: f64 = weight[..vocab_size * hidden_size]
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / total;

    let mut zero_rows = 0;
    for row in 0..vocab_size {
        let start = row * hidden_size;
        let end = start + hidden_size;
        if weight[start..end].iter().all(|&v| v == 0.0) {
            zero_rows += 1;
        }
    }

    EmbeddingStats {
        vocab_size,
        hidden_size,
        mean: mean as f32,
        std_dev: var.sqrt() as f32,
        zero_rows,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_basic() {
        // 3-token vocab, hidden=2
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ids = vec![0, 2, 1];
        let out = lookup_embeddings(&w, &ids, 2);
        assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0]);
    }

    #[test]
    fn test_lookup_oov() {
        let w = vec![1.0, 2.0];
        let ids = vec![0, 5]; // id 5 is out of range
        let out = lookup_embeddings(&w, &ids, 2);
        assert_eq!(out.len(), 4);
        assert_eq!(out[2], 0.0); // zero padding for OOV
    }

    #[test]
    fn test_lookup_empty() {
        let w = vec![1.0, 2.0];
        let out = lookup_embeddings(&w, &[], 2);
        assert!(out.is_empty());
    }

    #[test]
    fn test_sinusoidal() {
        let mut emb = vec![0.0f32; 8]; // 2 positions, hidden=4
        add_sinusoidal_positions(&mut emb, 2, 4, 10000.0);
        // Position 0 should have sin(0)=0, cos(0)=1 pattern
        assert!((emb[0] - 0.0).abs() < 0.01); // sin(0) = 0
        assert!((emb[1] - 1.0).abs() < 0.01); // cos(0) = 1
    }

    #[test]
    fn test_tied_same() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        assert!(check_tied_embeddings(&a, &b, 4));
    }

    #[test]
    fn test_tied_different() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 5.0];
        assert!(!check_tied_embeddings(&a, &b, 4));
    }

    #[test]
    fn test_tied_different_size() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(!check_tied_embeddings(&a, &b, 2));
    }

    #[test]
    fn test_stats_basic() {
        // 2 tokens, hidden=2: [[1,1], [0,0]]
        let w = vec![1.0, 1.0, 0.0, 0.0];
        let s = embedding_stats(&w, 2, 2);
        assert_eq!(s.vocab_size, 2);
        assert_eq!(s.zero_rows, 1);
        assert!((s.mean - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_stats_empty() {
        let s = embedding_stats(&[], 0, 0);
        assert_eq!(s.mean, 0.0);
    }

    #[test]
    fn test_stats_no_zeros() {
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let s = embedding_stats(&w, 2, 2);
        assert_eq!(s.zero_rows, 0);
    }

    #[test]
    fn test_stats_std_dev() {
        let w = vec![1.0; 4];
        let s = embedding_stats(&w, 2, 2);
        assert!(s.std_dev < 0.01); // all same values
    }

    #[test]
    fn test_lookup_single() {
        let w = vec![10.0, 20.0];
        let out = lookup_embeddings(&w, &[0], 2);
        assert_eq!(out, vec![10.0, 20.0]);
    }
}
