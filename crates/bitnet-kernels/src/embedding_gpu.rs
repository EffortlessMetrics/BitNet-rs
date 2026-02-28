//! GPU-accelerated embedding table lookup for transformer inference.
//! Provides a CPU reference implementation with the same layout as the OpenCL kernels.

/// Configuration for an embedding table.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
}

/// GPU embedding table with CPU reference implementation.
/// Storage layout: flat [vocab_size * embed_dim] matching the OpenCL kernel.
#[derive(Debug)]
pub struct GpuEmbeddingTable {
    config: EmbeddingConfig,
    weights: Vec<f32>,
}

impl GpuEmbeddingTable {
    /// Create a new embedding table from a flat weight vector.
    ///
    /// # Panics
    /// Panics if weights.len() != vocab_size * embed_dim.
    pub fn new(config: EmbeddingConfig, weights: Vec<f32>) -> Self {
        let expected = config.vocab_size * config.embed_dim;
        assert_eq!(
            weights.len(),
            expected,
            "weight length {} != vocab_size*embed_dim {}",
            weights.len(),
            expected,
        );
        Self { config, weights }
    }

    /// Look up the embedding for a single token (CPU reference path).
    ///
    /// # Panics
    /// Panics if 	oken_id >= vocab_size.
    pub fn lookup(&self, token_id: u32) -> Vec<f32> {
        let tid = token_id as usize;
        assert!(tid < self.config.vocab_size, "token_id {tid} >= vocab_size {}", self.config.vocab_size);
        let start = tid * self.config.embed_dim;
        self.weights[start..start + self.config.embed_dim].to_vec()
    }

    /// Batched lookup: fetch embeddings for a sequence of token IDs (CPU reference path).
    /// Returns a flat vector of length 	okens.len() * embed_dim.
    ///
    /// # Panics
    /// Panics if any token_id >= vocab_size.
    pub fn lookup_batched(&self, tokens: &[u32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(tokens.len() * self.config.embed_dim);
        for &tok in tokens {
            out.extend_from_slice(&self.lookup(tok));
        }
        out
    }

    pub fn vocab_size(&self) -> usize { self.config.vocab_size }
    pub fn embed_dim(&self) -> usize { self.config.embed_dim }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_table(vocab: usize, dim: usize) -> GpuEmbeddingTable {
        // Fill weights so that table[token][d] = token * 100 + d
        let weights: Vec<f32> = (0..vocab)
            .flat_map(|t| (0..dim).map(move |d| (t * 100 + d) as f32))
            .collect();
        GpuEmbeddingTable::new(EmbeddingConfig { vocab_size: vocab, embed_dim: dim }, weights)
    }

    #[test]
    fn lookup_single_token() {
        let table = make_table(8, 4);
        let emb = table.lookup(3);
        assert_eq!(emb, vec![300.0, 301.0, 302.0, 303.0]);
    }

    #[test]
    fn lookup_first_and_last_token() {
        let table = make_table(10, 3);
        assert_eq!(table.lookup(0), vec![0.0, 1.0, 2.0]);
        assert_eq!(table.lookup(9), vec![900.0, 901.0, 902.0]);
    }

    #[test]
    fn batched_lookup() {
        let table = make_table(8, 4);
        let out = table.lookup_batched(&[1, 3]);
        assert_eq!(out.len(), 8);
        assert_eq!(&out[0..4], &[100.0, 101.0, 102.0, 103.0]);
        assert_eq!(&out[4..8], &[300.0, 301.0, 302.0, 303.0]);
    }

    #[test]
    fn batched_lookup_empty() {
        let table = make_table(4, 2);
        let out = table.lookup_batched(&[]);
        assert!(out.is_empty());
    }

    #[test]
    #[should_panic(expected = "token_id")]
    fn lookup_out_of_range() {
        let table = make_table(4, 2);
        table.lookup(4);
    }

    #[test]
    #[should_panic(expected = "weight length")]
    fn wrong_weight_length() {
        GpuEmbeddingTable::new(
            EmbeddingConfig { vocab_size: 2, embed_dim: 3 },
            vec![0.0; 5],
        );
    }

    #[test]
    fn large_embed_dim() {
        let dim = 256;
        let table = make_table(4, dim);
        let emb = table.lookup(2);
        assert_eq!(emb.len(), dim);
        assert_eq!(emb[0], 200.0);
        assert_eq!(emb[255], 455.0);
    }

    #[test]
    fn vocab_and_dim_accessors() {
        let table = make_table(16, 64);
        assert_eq!(table.vocab_size(), 16);
        assert_eq!(table.embed_dim(), 64);
    }
}
