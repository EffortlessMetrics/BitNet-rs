#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

/// Fuzz SIMD embedding lookup with random vocab sizes and dimensions,
/// verifying bounds safety and output consistency.
#[derive(Arbitrary, Debug)]
struct SimdEmbeddingInput {
    vocab_size: u8,
    embed_dim: u8,
    table_bytes: Vec<u8>,
    token_ids: Vec<u16>,
    scale_byte: u8,
    normalize: bool,
    _gather_mode: u8,
}

struct SimdEmbeddingTable {
    data: Vec<f32>,
    vocab_size: usize,
    embed_dim: usize,
}

impl SimdEmbeddingTable {
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

    /// Scaled lookup: multiply embedding by a scale factor.
    fn scaled_lookup(&self, token_id: usize, scale: f32) -> Option<Vec<f32>> {
        self.lookup(token_id).map(|emb| emb.iter().map(|&v| v * scale).collect())
    }

    /// Gather: batch lookup with accumulation into output buffer.
    fn gather(&self, ids: &[usize], output: &mut [f32]) {
        let dim = self.embed_dim;
        for (i, &id) in ids.iter().enumerate() {
            let out_start = i * dim;
            let out_end = out_start + dim;
            if out_end > output.len() {
                break;
            }
            if let Some(emb) = self.lookup(id) {
                output[out_start..out_end].copy_from_slice(emb);
            } else {
                // Zero-fill for OOB tokens
                for v in &mut output[out_start..out_end] {
                    *v = 0.0;
                }
            }
        }
    }

    /// L2 normalize an embedding vector in-place.
    fn normalize_vec(vec: &mut [f32]) {
        let norm_sq: f32 = vec.iter().map(|v| v * v).sum();
        if norm_sq > 1e-12 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for v in vec.iter_mut() {
                *v *= inv_norm;
            }
        }
    }
}

fuzz_target!(|input: SimdEmbeddingInput| {
    let vocab_size = (input.vocab_size as usize % 128) + 1;
    let embed_dim = (input.embed_dim as usize % 64) + 1;
    let required = vocab_size * embed_dim;

    // Build table from raw bytes, pad with zeros.
    let aligned_len = (input.table_bytes.len() / 4) * 4;
    let mut table: Vec<f32> = input.table_bytes[..aligned_len]
        .chunks_exact(4)
        .take(required)
        .map(|b| {
            let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            if v.is_finite() { v } else { 0.0 }
        })
        .collect();
    table.resize(required, 0.0);

    let emb = SimdEmbeddingTable::new(table, vocab_size, embed_dim);
    let scale = (input.scale_byte as f32 / 255.0) * 4.0 - 2.0;
    let ids: Vec<usize> = input.token_ids.iter().take(64).map(|&t| t as usize).collect();

    // Invariant 1: Valid lookups return correct dimension.
    for &id in &ids {
        if id < vocab_size {
            let result = emb.lookup(id);
            assert!(result.is_some(), "valid token {id} returned None");
            assert_eq!(result.unwrap().len(), embed_dim);
        } else {
            assert!(emb.lookup(id).is_none());
        }
    }

    // Invariant 2: Scaled lookup preserves dimension.
    for &id in ids.iter().take(16) {
        if id < vocab_size {
            let scaled = emb.scaled_lookup(id, scale);
            assert!(scaled.is_some());
            let sv = scaled.unwrap();
            assert_eq!(sv.len(), embed_dim);
            // Verify scaling: each element should be original * scale
            if let Some(orig) = emb.lookup(id) {
                for (i, (&o, &s)) in orig.iter().zip(sv.iter()).enumerate() {
                    let expected = o * scale;
                    if expected.is_finite() && s.is_finite() {
                        assert!(
                            (s - expected).abs() < 1e-5,
                            "scaled mismatch at {i}: {s} vs {expected}"
                        );
                    }
                }
            }
        }
    }

    // Invariant 3: Gather fills output buffer correctly.
    let valid_ids: Vec<usize> =
        ids.iter().copied().filter(|&id| id < vocab_size).take(16).collect();
    if !valid_ids.is_empty() {
        let mut output = vec![f32::NAN; valid_ids.len() * embed_dim];
        emb.gather(&valid_ids, &mut output);
        // All values should be overwritten (no NaN remaining for valid lookups)
        for (i, &v) in output.iter().enumerate() {
            assert!(!v.is_nan(), "gather left NaN at index {i}");
        }
    }

    // Invariant 4: Gather with OOB tokens zero-fills.
    let oob_ids: Vec<usize> = ids.iter().copied().filter(|&id| id >= vocab_size).take(4).collect();
    if !oob_ids.is_empty() {
        let mut output = vec![f32::NAN; oob_ids.len() * embed_dim];
        emb.gather(&oob_ids, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert_eq!(v, 0.0, "OOB gather should zero-fill, got {v} at {i}");
        }
    }

    // Invariant 5: Normalization produces unit-length vectors.
    if input.normalize {
        for &id in ids.iter().take(8) {
            if id < vocab_size {
                if let Some(orig) = emb.lookup(id) {
                    let mut vec = orig.to_vec();
                    SimdEmbeddingTable::normalize_vec(&mut vec);
                    let norm_sq: f32 = vec.iter().map(|v| v * v).sum();
                    // Zero vectors stay zero
                    let orig_norm_sq: f32 = orig.iter().map(|v| v * v).sum();
                    if orig_norm_sq > 1e-12 && norm_sq.is_finite() {
                        assert!(
                            (norm_sq - 1.0).abs() < 1e-4,
                            "normalized vector norm²={norm_sq}, expected ~1.0"
                        );
                    }
                }
            }
        }
    }

    // Invariant 6: Determinism — same lookup twice yields identical results.
    if let (Some(a), Some(b)) = (emb.lookup(0), emb.lookup(0)) {
        assert_eq!(a, b, "non-deterministic embedding lookup");
    }
});
