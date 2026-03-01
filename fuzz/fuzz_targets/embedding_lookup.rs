#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::embedding::{
    EmbeddingConfig, embedding_accumulate, embedding_lookup, embedding_lookup_simd,
    normalize_embeddings,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct EmbeddingInput {
    ops: Vec<EmbedOp>,
}

#[derive(Arbitrary, Debug)]
enum EmbedOp {
    Lookup { vocab_size: u8, dim: u8, indices: Vec<u8> },
    SimdLookup { vocab_size: u8, dim: u8, indices: Vec<u8>, padding_idx: Option<u8> },
    Accumulate { vocab_size: u8, dim: u8, indices: Vec<u8>, weights: Vec<f32> },
    Normalize { dim: u8, data: Vec<f32> },
}

fuzz_target!(|input: EmbeddingInput| {
    for op in input.ops.into_iter().take(256) {
        match op {
            EmbedOp::Lookup { vocab_size, dim, indices } => {
                let vs = (vocab_size as usize).clamp(1, 64);
                let d = (dim as usize).clamp(1, 32);
                let table: Vec<f32> = (0..vs * d).map(|i| (i as f32) * 0.01).collect();
                let idx: Vec<u32> = indices.iter().take(16).map(|&i| i as u32).collect();
                let _ = embedding_lookup(&table, &idx, d);
            }
            EmbedOp::SimdLookup { vocab_size, dim, indices, padding_idx } => {
                let vs = (vocab_size as usize).clamp(1, 64);
                let d = (dim as usize).clamp(1, 32);
                let table: Vec<f32> = (0..vs * d).map(|i| (i as f32) * 0.01).collect();
                let idx: Vec<u32> = indices.iter().take(16).map(|&i| i as u32).collect();
                let pad = padding_idx.map(|p| p as u32);
                let config = EmbeddingConfig { vocab_size: vs, embedding_dim: d, padding_idx: pad };

                if let Ok(out) = embedding_lookup_simd(&table, &idx, &config) {
                    assert_eq!(out.len(), idx.len() * d);
                    if let Some(pi) = pad {
                        for (i, &id) in idx.iter().enumerate() {
                            if id == pi && (id as usize) < vs {
                                let slice = &out[i * d..(i + 1) * d];
                                assert!(slice.iter().all(|&v| v == 0.0), "padding idx not zeroed");
                            }
                        }
                    }
                }
            }
            EmbedOp::Accumulate { vocab_size, dim, indices, weights } => {
                let vs = (vocab_size as usize).clamp(1, 64);
                let d = (dim as usize).clamp(1, 32);
                let table: Vec<f32> = (0..vs * d).map(|i| (i as f32) * 0.01).collect();
                let n = indices.len().min(16);
                let idx: Vec<u32> = indices.iter().take(n).map(|&i| i as u32).collect();
                let w: Vec<f32> = weights
                    .iter()
                    .take(n)
                    .map(|&v| if v.is_finite() { v } else { 0.0 })
                    .chain(std::iter::repeat_n(1.0f32, n.saturating_sub(weights.len())))
                    .take(n)
                    .collect();
                let _ = embedding_accumulate(&table, &idx, &w, d);
            }
            EmbedOp::Normalize { dim, data } => {
                let d = (dim as usize).clamp(1, 32);
                let mut buf: Vec<f32> = data
                    .into_iter()
                    .take(d * 8)
                    .map(|v| if v.is_finite() { v } else { 0.0 })
                    .collect();
                normalize_embeddings(&mut buf, d);
                for chunk in buf.chunks(d) {
                    if chunk.len() == d {
                        let norm_sq: f32 = chunk.iter().map(|&x| x * x).sum();
                        assert!(
                            norm_sq < 1e-6 || (norm_sq - 1.0).abs() < 1e-4,
                            "unexpected norm: {norm_sq}"
                        );
                    }
                }
            }
        }
    }
});
