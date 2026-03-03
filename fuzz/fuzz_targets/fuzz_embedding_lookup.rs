#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::embedding::{EmbeddingConfig, embedding_lookup, embedding_lookup_simd};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct EmbeddingLookupInput {
    vocab_size: u8,
    embed_dim: u8,
    table_data: Vec<u8>,
    indices: Vec<u16>,
    use_simd: bool,
    padding_idx: Option<u8>,
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
    // Pad to required size and replace non-finite values
    table.resize(required, 0.0);
    for v in table.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }

    // Build indices — mix of valid and OOB
    let indices: Vec<u32> = input.indices.iter().take(64).map(|&i| i as u32).collect();
    let valid_indices: Vec<u32> =
        indices.iter().copied().filter(|&i| (i as usize) < vocab_size).collect();
    let oob_indices: Vec<u32> =
        indices.iter().copied().filter(|&i| (i as usize) >= vocab_size).collect();

    // --- Test basic embedding_lookup ---
    // Valid indices should succeed
    if !valid_indices.is_empty() {
        match embedding_lookup(&table, &valid_indices, embed_dim) {
            Ok(result) => {
                // Invariant 1: Output shape is [n_indices, embed_dim]
                assert_eq!(
                    result.len(),
                    valid_indices.len() * embed_dim,
                    "embedding output shape mismatch"
                );

                // Invariant 2: Each row matches the table row
                for (idx_pos, &token_id) in valid_indices.iter().enumerate() {
                    let out_start = idx_pos * embed_dim;
                    let tbl_start = token_id as usize * embed_dim;
                    for d in 0..embed_dim {
                        assert_eq!(
                            result[out_start + d],
                            table[tbl_start + d],
                            "mismatch at token={token_id} dim={d}"
                        );
                    }
                }
            }
            Err(_) => {} // Graceful error is acceptable
        }
    }

    // OOB indices should error, not panic
    if !oob_indices.is_empty() {
        let result = embedding_lookup(&table, &oob_indices, embed_dim);
        // Must not panic — error is expected
        assert!(result.is_err(), "OOB indices should produce an error");
    }

    // --- Test SIMD path ---
    if input.use_simd {
        let padding_idx = input.padding_idx.map(|pad| pad as u32 % (vocab_size as u32 + 1));
        let config = EmbeddingConfig { vocab_size, embedding_dim: embed_dim, padding_idx };

        if !valid_indices.is_empty() {
            match embedding_lookup_simd(&table, &valid_indices, &config) {
                Ok(simd_result) => {
                    // Invariant 3: SIMD result matches basic result
                    if let Ok(basic_result) = embedding_lookup(&table, &valid_indices, embed_dim) {
                        // If padding_idx is set, padded rows are zeroed in simd but not basic
                        if padding_idx.is_none() {
                            assert_eq!(
                                simd_result.len(),
                                basic_result.len(),
                                "simd/basic length mismatch"
                            );
                            for (i, (&s, &b)) in
                                simd_result.iter().zip(basic_result.iter()).enumerate()
                            {
                                assert_eq!(s, b, "simd/basic mismatch at index {i}");
                            }
                        }
                    }
                }
                Err(_) => {} // Graceful error is acceptable
            }
        }

        // OOB via SIMD should also error
        if !oob_indices.is_empty() {
            let result = embedding_lookup_simd(&table, &oob_indices, &config);
            assert!(result.is_err(), "OOB SIMD indices should produce an error");
        }
    }

    // --- Edge cases ---
    // Empty indices should produce empty output
    let empty: Vec<u32> = vec![];
    if let Ok(result) = embedding_lookup(&table, &empty, embed_dim) {
        assert!(result.is_empty(), "empty indices should produce empty output");
    }

    // Duplicate indices should produce duplicate rows
    if vocab_size >= 1 {
        let dupes = vec![0u32; 3];
        if let Ok(result) = embedding_lookup(&table, &dupes, embed_dim) {
            assert_eq!(result.len(), 3 * embed_dim);
            // All three rows should be identical
            for d in 0..embed_dim {
                assert_eq!(result[d], result[embed_dim + d], "duplicate row 0 vs 1 at dim {d}");
                assert_eq!(result[d], result[2 * embed_dim + d], "duplicate row 0 vs 2 at dim {d}");
            }
        }
    }
});
