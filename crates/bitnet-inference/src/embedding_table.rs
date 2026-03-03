//! Embedding table operations.
//!
//! Vocabulary embedding lookup with support for positional embeddings,
//! normalization, and batch lookups.

/// An embedding table backed by a flat f32 buffer.
#[derive(Debug, Clone)]
pub struct EmbeddingTable {
    data: Vec<f32>,
    vocab_size: usize,
    embed_dim: usize,
}

impl EmbeddingTable {
    /// Create an embedding table from a flat buffer.
    pub fn new(data: Vec<f32>, vocab_size: usize, embed_dim: usize) -> Option<Self> {
        if data.len() != vocab_size * embed_dim {
            return None;
        }
        Some(Self { data, vocab_size, embed_dim })
    }

    /// Create a zero-initialized table.
    pub fn zeros(vocab_size: usize, embed_dim: usize) -> Self {
        Self { data: vec![0.0; vocab_size * embed_dim], vocab_size, embed_dim }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    pub fn total_params(&self) -> usize {
        self.vocab_size * self.embed_dim
    }

    /// Size in bytes (f32).
    pub fn size_bytes(&self) -> usize {
        self.data.len() * 4
    }

    /// Look up a single token ID.
    pub fn lookup(&self, token_id: u32) -> Option<&[f32]> {
        let idx = token_id as usize;
        if idx >= self.vocab_size {
            return None;
        }
        let start = idx * self.embed_dim;
        Some(&self.data[start..start + self.embed_dim])
    }

    /// Look up multiple token IDs.
    pub fn lookup_batch(&self, token_ids: &[u32]) -> Option<Vec<f32>> {
        let mut result = Vec::with_capacity(token_ids.len() * self.embed_dim);
        for &id in token_ids {
            let embedding = self.lookup(id)?;
            result.extend_from_slice(embedding);
        }
        Some(result)
    }

    /// Get the L2 norm of an embedding.
    pub fn embedding_norm(&self, token_id: u32) -> Option<f32> {
        let embedding = self.lookup(token_id)?;
        let sum_sq: f32 = embedding.iter().map(|x| x * x).sum();
        Some(sum_sq.sqrt())
    }

    /// Cosine similarity between two embeddings.
    pub fn cosine_similarity(&self, id_a: u32, id_b: u32) -> Option<f32> {
        let a = self.lookup(id_a)?;
        let b = self.lookup(id_b)?;
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return Some(0.0);
        }
        Some(dot / (norm_a * norm_b))
    }

    /// Set an embedding for a token.
    pub fn set(&mut self, token_id: u32, values: &[f32]) -> bool {
        let idx = token_id as usize;
        if idx >= self.vocab_size || values.len() != self.embed_dim {
            return false;
        }
        let start = idx * self.embed_dim;
        self.data[start..start + self.embed_dim].copy_from_slice(values);
        true
    }

    /// Mean embedding across all tokens.
    pub fn mean_embedding(&self) -> Vec<f32> {
        let mut mean = vec![0.0f64; self.embed_dim];
        for row in 0..self.vocab_size {
            let start = row * self.embed_dim;
            for (j, val) in self.data[start..start + self.embed_dim].iter().enumerate() {
                mean[j] += *val as f64;
            }
        }
        let n = self.vocab_size as f64;
        mean.iter().map(|v| (*v / n) as f32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_table() -> EmbeddingTable {
        // 4 tokens, dim=3
        EmbeddingTable::new(
            vec![
                1.0, 0.0, 0.0, // token 0
                0.0, 1.0, 0.0, // token 1
                0.0, 0.0, 1.0, // token 2
                1.0, 1.0, 1.0, // token 3
            ],
            4,
            3,
        )
        .unwrap()
    }

    #[test]
    fn test_lookup() {
        let table = sample_table();
        assert_eq!(table.lookup(0), Some([1.0, 0.0, 0.0].as_slice()));
        assert_eq!(table.lookup(2), Some([0.0, 0.0, 1.0].as_slice()));
    }

    #[test]
    fn test_lookup_oob() {
        let table = sample_table();
        assert!(table.lookup(10).is_none());
    }

    #[test]
    fn test_lookup_batch() {
        let table = sample_table();
        let result = table.lookup_batch(&[0, 2]).unwrap();
        assert_eq!(result.len(), 6);
        assert_eq!(&result[..3], &[1.0, 0.0, 0.0]);
        assert_eq!(&result[3..], &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_lookup_batch_invalid() {
        let table = sample_table();
        assert!(table.lookup_batch(&[0, 99]).is_none());
    }

    #[test]
    fn test_embedding_norm() {
        let table = sample_table();
        assert!((table.embedding_norm(0).unwrap() - 1.0).abs() < 1e-6);
        let expected = (3.0f32).sqrt();
        assert!((table.embedding_norm(3).unwrap() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let table = sample_table();
        let sim = table.cosine_similarity(0, 0).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let table = sample_table();
        let sim = table.cosine_similarity(0, 1).unwrap();
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_set_embedding() {
        let mut table = sample_table();
        assert!(table.set(0, &[9.0, 8.0, 7.0]));
        assert_eq!(table.lookup(0), Some([9.0, 8.0, 7.0].as_slice()));
    }

    #[test]
    fn test_set_invalid() {
        let mut table = sample_table();
        assert!(!table.set(0, &[1.0, 2.0])); // wrong dim
        assert!(!table.set(99, &[1.0, 2.0, 3.0])); // oob
    }

    #[test]
    fn test_zeros() {
        let table = EmbeddingTable::zeros(10, 5);
        assert_eq!(table.vocab_size(), 10);
        assert_eq!(table.embed_dim(), 5);
        assert_eq!(table.lookup(0), Some(vec![0.0; 5].as_slice()));
    }

    #[test]
    fn test_size() {
        let table = sample_table();
        assert_eq!(table.total_params(), 12);
        assert_eq!(table.size_bytes(), 48);
    }

    #[test]
    fn test_new_invalid_size() {
        assert!(EmbeddingTable::new(vec![1.0, 2.0], 3, 3).is_none());
    }
}
