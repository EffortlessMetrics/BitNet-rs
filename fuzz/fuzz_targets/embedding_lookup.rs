#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct EmbeddingInput {
    /// Number of vocabulary entries (clamped to small range).
    vocab_size: u8,
    /// Embedding dimension (clamped to small range).
    embed_dim: u8,
    /// Raw embedding table data (f32 bytes).
    table_data: Vec<u8>,
    /// Token IDs to look up.
    token_ids: Vec<u16>,
}

// Minimal embedding table for fuzzing lookup logic.
struct EmbeddingTable {
    data: Vec<f32>,
    vocab_size: usize,
    embed_dim: usize,
}

impl EmbeddingTable {
    fn new(data: Vec<f32>, vocab_size: usize, embed_dim: usize) -> Self {
        Self { data, vocab_size, embed_dim }
    }

    /// Look up a single token's embedding. Returns None for out-of-bounds IDs.
    fn lookup(&self, token_id: usize) -> Option<&[f32]> {
        if token_id >= self.vocab_size {
            return None;
        }
        let start = token_id * self.embed_dim;
        let end = start + self.embed_dim;
        if end > self.data.len() {
            return None;
        }
        Some(&self.data[start..end])
    }

    /// Batch lookup: gather embeddings for a sequence of token IDs.
    fn batch_lookup(&self, token_ids: &[usize]) -> Vec<f32> {
        let mut result = Vec::with_capacity(token_ids.len() * self.embed_dim);
        for &tid in token_ids {
            if let Some(emb) = self.lookup(tid) {
                result.extend_from_slice(emb);
            }
            // Out-of-bounds tokens are silently skipped (no panic).
        }
        result
    }
}

fuzz_target!(|input: EmbeddingInput| {
    let vocab_size = (input.vocab_size as usize % 64) + 1;
    let embed_dim = (input.embed_dim as usize % 32) + 1;
    let required = vocab_size * embed_dim;

    // Build embedding table from raw bytes, padding with zeros if needed.
    let aligned_len = (input.table_data.len() / 4) * 4;
    let mut table: Vec<f32> = input.table_data[..aligned_len]
        .chunks_exact(4)
        .take(256)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Pad to required size.
    table.resize(required, 0.0);

    let emb = EmbeddingTable::new(table, vocab_size, embed_dim);

    // Invariant 1: Valid lookups return correct dimension.
    let token_ids: Vec<usize> = input.token_ids.iter().take(256).map(|&t| t as usize).collect();

    for &tid in &token_ids {
        if tid < vocab_size {
            let result = emb.lookup(tid);
            assert!(result.is_some(), "valid token {tid} returned None");
            assert_eq!(result.unwrap().len(), embed_dim, "wrong embedding dim for token {tid}");
        } else {
            // Invariant 2: Out-of-bounds tokens return None (no panic).
            assert!(emb.lookup(tid).is_none(), "OOB token {tid} should be None");
        }
    }

    // Invariant 3: Batch lookup output dimension is correct.
    let valid_ids: Vec<usize> = token_ids.iter().copied().filter(|&t| t < vocab_size).collect();
    let batch = emb.batch_lookup(&valid_ids);
    assert_eq!(
        batch.len(),
        valid_ids.len() * embed_dim,
        "batch output dimension mismatch: expected {} got {}",
        valid_ids.len() * embed_dim,
        batch.len()
    );

    // Invariant 4: Batch with all-OOB tokens produces empty output.
    let oob_ids: Vec<usize> =
        token_ids.iter().copied().filter(|&t| t >= vocab_size).take(16).collect();
    let oob_batch = emb.batch_lookup(&oob_ids);
    assert!(oob_batch.is_empty(), "OOB batch should be empty, got len {}", oob_batch.len());
});
