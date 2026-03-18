//! Embedding table operations for token/position lookups.
//!
//! Token embedding lookup, position embedding, embedding+position
//! fusion, vocabulary projection (output logits).

/// Look up embeddings for a sequence of token IDs.
/// Returns a flat vec of [seq_len * embed_dim] f32 values.
pub fn token_embed_lookup(
    token_ids: &[u32],
    embed_table: &[f32],
    embed_dim: usize,
    vocab_size: usize,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(token_ids.len() * embed_dim);
    for &id in token_ids {
        let idx = (id as usize).min(vocab_size - 1);
        let start = idx * embed_dim;
        let end = start + embed_dim;
        if end <= embed_table.len() {
            output.extend_from_slice(&embed_table[start..end]);
        } else {
            output.extend(std::iter::repeat_n(0.0f32, embed_dim));
        }
    }
    output
}

/// Simple sinusoidal position embeddings (Vaswani et al.).
/// Returns [max_len * embed_dim] f32 values.
pub fn sinusoidal_position_embed(max_len: usize, embed_dim: usize) -> Vec<f32> {
    let mut pe = vec![0.0f32; max_len * embed_dim];
    for pos in 0..max_len {
        for i in 0..embed_dim / 2 {
            let angle = pos as f64 / (10000.0f64).powf(2.0 * i as f64 / embed_dim as f64);
            pe[pos * embed_dim + 2 * i] = angle.sin() as f32;
            pe[pos * embed_dim + 2 * i + 1] = angle.cos() as f32;
        }
    }
    pe
}

/// Add position embeddings to token embeddings (in-place).
/// Both are flat [seq_len * embed_dim] arrays.
pub fn add_position_embed(
    token_embeds: &mut [f32],
    pos_embeds: &[f32],
    embed_dim: usize,
    start_pos: usize,
) {
    let seq_len = token_embeds.len() / embed_dim;
    for i in 0..seq_len {
        let tok_offset = i * embed_dim;
        let pos_offset = (start_pos + i) * embed_dim;
        if pos_offset + embed_dim <= pos_embeds.len() {
            for j in 0..embed_dim {
                token_embeds[tok_offset + j] += pos_embeds[pos_offset + j];
            }
        }
    }
}

/// Project hidden states to vocabulary logits.
/// hidden: [seq_len * hidden_dim], weight: [vocab_size * hidden_dim]
/// Returns: [seq_len * vocab_size].
pub fn vocab_projection(
    hidden: &[f32],
    weight: &[f32],
    hidden_dim: usize,
    vocab_size: usize,
) -> Vec<f32> {
    let seq_len = hidden.len() / hidden_dim;
    let mut logits = vec![0.0f32; seq_len * vocab_size];
    for s in 0..seq_len {
        let h_offset = s * hidden_dim;
        let l_offset = s * vocab_size;
        for v in 0..vocab_size {
            let w_offset = v * hidden_dim;
            let mut dot = 0.0f32;
            for d in 0..hidden_dim {
                dot += hidden[h_offset + d] * weight[w_offset + d];
            }
            logits[l_offset + v] = dot;
        }
    }
    logits
}

/// Embedding table statistics.
#[derive(Debug)]
pub struct EmbedStats {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub mean_norm: f32,
    pub max_norm: f32,
    pub min_norm: f32,
}

/// Compute norms for each embedding vector.
pub fn embed_norms(table: &[f32], vocab_size: usize, embed_dim: usize) -> Vec<f32> {
    (0..vocab_size)
        .map(|i| {
            let start = i * embed_dim;
            let end = (start + embed_dim).min(table.len());
            if start >= table.len() {
                return 0.0;
            }
            table[start..end].iter().map(|x| x * x).sum::<f32>().sqrt()
        })
        .collect()
}

/// Compute embedding statistics.
pub fn compute_embed_stats(table: &[f32], vocab_size: usize, embed_dim: usize) -> EmbedStats {
    let norms = embed_norms(table, vocab_size, embed_dim);
    let n = norms.len();
    if n == 0 {
        return EmbedStats { vocab_size, embed_dim, mean_norm: 0.0, max_norm: 0.0, min_norm: 0.0 };
    }
    let sum: f32 = norms.iter().sum();
    EmbedStats {
        vocab_size,
        embed_dim,
        mean_norm: sum / n as f32,
        max_norm: norms.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        min_norm: norms.iter().copied().fold(f32::INFINITY, f32::min),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_embed_lookup() {
        // vocab_size=3, embed_dim=2
        let table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ids = vec![0, 2, 1];
        let out = token_embed_lookup(&ids, &table, 2, 3);
        assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0]);
    }

    #[test]
    fn test_embed_lookup_oob() {
        let table = vec![1.0, 2.0, 3.0, 4.0];
        let ids = [100]; // out of bounds
        let out = token_embed_lookup(&ids, &table, 2, 2);
        // Clamped to vocab_size-1 = 1
        assert_eq!(out, vec![3.0, 4.0]);
    }

    #[test]
    fn test_sinusoidal_pe() {
        let pe = sinusoidal_position_embed(4, 8);
        assert_eq!(pe.len(), 4 * 8);
        // Position 0 should have sin(0)=0 in first element
        assert!((pe[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_position_embed() {
        let mut tok = vec![1.0, 2.0, 3.0, 4.0]; // 2 tokens, dim=2
        let pos = vec![0.1, 0.2, 0.3, 0.4];
        add_position_embed(&mut tok, &pos, 2, 0);
        assert!((tok[0] - 1.1).abs() < 1e-6);
        assert!((tok[2] - 3.3).abs() < 1e-6);
    }

    #[test]
    fn test_add_position_embed_offset() {
        let mut tok = vec![1.0, 2.0]; // 1 token, dim=2
        let pos = vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0]; // 3 positions
        add_position_embed(&mut tok, &pos, 2, 1);
        assert!((tok[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_vocab_projection() {
        // hidden_dim=2, vocab_size=3, seq_len=1
        let hidden = vec![1.0, 2.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3 rows
        let logits = vocab_projection(&hidden, &weight, 2, 3);
        assert_eq!(logits.len(), 3);
        assert!((logits[0] - 1.0).abs() < 1e-6); // [1,0]·[1,2]=1
        assert!((logits[1] - 2.0).abs() < 1e-6); // [0,1]·[1,2]=2
        assert!((logits[2] - 3.0).abs() < 1e-6); // [1,1]·[1,2]=3
    }

    #[test]
    fn test_embed_norms() {
        let table = vec![3.0, 4.0, 0.0, 1.0]; // 2 vectors, dim=2
        let norms = embed_norms(&table, 2, 2);
        assert!((norms[0] - 5.0).abs() < 1e-6);
        assert!((norms[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embed_stats() {
        let table = vec![3.0, 4.0, 0.0, 1.0];
        let stats = compute_embed_stats(&table, 2, 2);
        assert_eq!(stats.vocab_size, 2);
        assert!((stats.max_norm - 5.0).abs() < 1e-6);
        assert!((stats.min_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sinusoidal_monotone() {
        let pe = sinusoidal_position_embed(10, 4);
        // Different positions should produce different embeddings
        let pos0 = &pe[0..4];
        let pos1 = &pe[4..8];
        assert!(pos0 != pos1);
    }

    #[test]
    fn test_projection_multi_seq() {
        let hidden = vec![1.0, 0.0, 0.0, 1.0]; // 2 tokens, dim=2
        let weight = vec![1.0, 1.0]; // 1 vocab entry
        let logits = vocab_projection(&hidden, &weight, 2, 1);
        assert_eq!(logits.len(), 2);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_lookup() {
        let table = vec![1.0, 2.0];
        let out = token_embed_lookup(&[], &table, 2, 1);
        assert!(out.is_empty());
    }

    #[test]
    fn test_embed_stats_empty() {
        let stats = compute_embed_stats(&[], 0, 4);
        assert_eq!(stats.mean_norm, 0.0);
    }
}
