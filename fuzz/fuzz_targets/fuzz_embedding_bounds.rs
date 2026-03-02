#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::embedding::{EmbeddingConfig, embedding_lookup, embedding_lookup_simd};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct EmbeddingBoundsInput {
    vocab_size: u8,
    embed_dim: u8,
    table_data: Vec<u8>,
    indices: Vec<u16>,
    padding_idx: Option<u8>,
    test_boundary: bool,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .map(|v| if v.is_finite() { v } else { 0.0 })
        .collect()
}

fuzz_target!(|input: EmbeddingBoundsInput| {
    let vocab_size = (input.vocab_size as usize % 64) + 1;
    let embed_dim = (input.embed_dim as usize % 32) + 1;
    let table_len = vocab_size * embed_dim;

    let mut table = bytes_to_f32(&input.table_data, table_len);
    table.resize(table_len, 0.0);

    // Build index list from fuzzed bytes, including boundary values.
    let mut indices: Vec<u32> = input.indices.iter().take(128).map(|&i| i as u32).collect();

    if input.test_boundary {
        // Inject boundary indices: 0, vocab_size-1, vocab_size, u32::MAX.
        indices.push(0);
        if vocab_size > 0 {
            indices.push((vocab_size - 1) as u32);
        }
        indices.push(vocab_size as u32);
        indices.push(u32::MAX);
    }

    if indices.is_empty() {
        return;
    }

    // --- scalar embedding_lookup ---
    // Only valid indices should succeed; any OOB should return Err, never panic.
    let all_valid = indices.iter().all(|&idx| (idx as usize) < vocab_size);

    match embedding_lookup(&table, &indices, embed_dim) {
        Ok(result) => {
            // Invariant 1: All indices must have been in-bounds.
            assert!(all_valid, "embedding_lookup succeeded with OOB indices");

            // Invariant 2: Output length = indices.len() * embed_dim.
            assert_eq!(
                result.len(),
                indices.len() * embed_dim,
                "output length mismatch: expected {}, got {}",
                indices.len() * embed_dim,
                result.len()
            );

            // Invariant 3: All output values are finite.
            for (i, &v) in result.iter().enumerate() {
                assert!(v.is_finite(), "non-finite output at index {i}: {v}");
            }

            // Invariant 4: Lookup result matches table slice for each index.
            for (i, &idx) in indices.iter().enumerate() {
                let start = (idx as usize) * embed_dim;
                let expected = &table[start..start + embed_dim];
                let actual = &result[i * embed_dim..(i + 1) * embed_dim];
                assert_eq!(expected, actual, "lookup mismatch at token {idx}");
            }
        }
        Err(_) => {
            // OOB index detected — this is expected, not a panic.
        }
    }

    // --- SIMD embedding_lookup_simd ---
    let padding_idx = input.padding_idx.map(|p| p as u32);
    let config = EmbeddingConfig { vocab_size, embedding_dim: embed_dim, padding_idx };

    // Filter to valid-only indices for SIMD path comparison.
    let valid_indices: Vec<u32> = indices
        .iter()
        .copied()
        .filter(|&idx| (idx as usize) < vocab_size)
        .filter(|&idx| Some(idx) != padding_idx)
        .collect();

    if !valid_indices.is_empty() {
        match embedding_lookup_simd(&table, &valid_indices, &config) {
            Ok(simd_result) => {
                // Invariant 5: SIMD result length matches.
                assert_eq!(
                    simd_result.len(),
                    valid_indices.len() * embed_dim,
                    "SIMD output length mismatch"
                );

                // Invariant 6: SIMD matches scalar for same valid inputs.
                if let Ok(scalar_result) = embedding_lookup(&table, &valid_indices, embed_dim) {
                    assert_eq!(
                        scalar_result.len(),
                        simd_result.len(),
                        "scalar vs SIMD length mismatch"
                    );
                    for (i, (&s, &d)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
                        assert_eq!(s, d, "scalar vs SIMD mismatch at index {i}: {s} vs {d}");
                    }
                }
            }
            Err(_) => {
                // Acceptable — SIMD path may also reject.
            }
        }
    }

    // --- Padding index path ---
    if let Some(pad) = padding_idx {
        if (pad as usize) < vocab_size {
            let pad_indices = vec![pad];
            match embedding_lookup_simd(&table, &pad_indices, &config) {
                Ok(result) => {
                    // Invariant 7: Padding index produces all zeros.
                    assert_eq!(result.len(), embed_dim);
                    for (i, &v) in result.iter().enumerate() {
                        assert_eq!(v, 0.0, "padding index output non-zero at {i}: {v}");
                    }
                }
                Err(_) => {}
            }
        }
    }
});
