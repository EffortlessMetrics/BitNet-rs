#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct EmbeddingLookupInput {
    vocab_size: u8,
    embed_dim: u8,
    table_data: Vec<u8>,
    /// Token IDs as u32 to cover a wider OOB range.
    token_ids: Vec<u32>,
    /// Whether to test gather (batch lookup with possible duplicates).
    test_gather: bool,
}

struct EmbeddingTable {
    data: Vec<f32>,
    vocab_size: usize,
    embed_dim: usize,
}

impl EmbeddingTable {
    fn new(data: Vec<f32>, vocab_size: usize, embed_dim: usize) -> Self {
        Self { data, vocab_size, embed_dim }
    }

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

    fn gather(&self, ids: &[usize]) -> Vec<f32> {
        let mut out = Vec::with_capacity(ids.len() * self.embed_dim);
        for &id in ids {
            if let Some(emb) = self.lookup(id) {
                out.extend_from_slice(emb);
            }
        }
        out
    }

    /// Compute L2 norm of an embedding vector.
    fn l2_norm(&self, token_id: usize) -> Option<f32> {
        self.lookup(token_id).map(|emb| emb.iter().map(|x| x * x).sum::<f32>().sqrt())
    }
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: EmbeddingLookupInput| {
    let vocab_size = (input.vocab_size as usize % 64) + 1;
    let embed_dim = (input.embed_dim as usize % 32) + 1;
    let required = vocab_size * embed_dim;

    let mut table = bytes_to_f32(&input.table_data, required);
    table.resize(required, 0.0);

    let emb = EmbeddingTable::new(table, vocab_size, embed_dim);

    let ids: Vec<usize> = input.token_ids.iter().take(128).map(|&t| t as usize).collect();

    // Invariant 1: Valid lookups return correct dimension.
    for &tid in &ids {
        match emb.lookup(tid) {
            Some(vec) => {
                assert!(tid < vocab_size, "OOB token {tid} returned Some");
                assert_eq!(vec.len(), embed_dim, "wrong dim for token {tid}");
            }
            None => {
                assert!(tid >= vocab_size, "valid token {tid} returned None");
            }
        }
    }

    // Invariant 2: Same token ID always yields identical embedding.
    for &tid in &ids {
        if let (Some(a), Some(b)) = (emb.lookup(tid), emb.lookup(tid)) {
            assert_eq!(a, b, "lookup should be deterministic for token {tid}");
        }
    }

    // Invariant 3: Gather output dimension is correct.
    let valid_ids: Vec<usize> = ids.iter().copied().filter(|&t| t < vocab_size).collect();
    let gathered = emb.gather(&valid_ids);
    assert_eq!(gathered.len(), valid_ids.len() * embed_dim, "gather output dim mismatch");

    // Invariant 4: Gather with duplicates produces repeated embeddings.
    if let Some(&first_valid) = valid_ids.first() {
        let dup_ids = vec![first_valid; 3];
        let dup_gathered = emb.gather(&dup_ids);
        assert_eq!(dup_gathered.len(), 3 * embed_dim);
        let e0 = &dup_gathered[..embed_dim];
        let e1 = &dup_gathered[embed_dim..2 * embed_dim];
        let e2 = &dup_gathered[2 * embed_dim..3 * embed_dim];
        assert_eq!(e0, e1, "duplicate gather entries should match");
        assert_eq!(e1, e2, "duplicate gather entries should match");
    }

    // Invariant 5: All-OOB gather produces empty output.
    let oob_ids: Vec<usize> = ids.iter().copied().filter(|&t| t >= vocab_size).take(8).collect();
    let oob_gathered = emb.gather(&oob_ids);
    assert!(oob_gathered.is_empty(), "all-OOB gather should be empty");

    // Invariant 6: L2 norm is non-negative for valid tokens.
    if input.test_gather {
        for &tid in &valid_ids {
            if let Some(norm) = emb.l2_norm(tid) {
                assert!(norm >= 0.0 || norm.is_nan(), "L2 norm must be non-negative");
            }
        }
    }
});
