#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::embedding::{
    CpuEmbeddingConfig, embedding_with_position, positional_embedding,
};
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, apply_rope_batch, compute_frequencies};
use libfuzzer_sys::fuzz_target;

/// Fuzz sinusoidal positional embeddings, RoPE frequency tables,
/// and RoPE application to random vectors.
#[derive(Arbitrary, Debug)]
struct PosEncInput {
    op: u8,
    seq_len: u8,
    embed_dim: u8,
    position: u8,
    base: f32,
    #[allow(dead_code)]
    scaling_factor: f32,
    data: Vec<u8>,
    indices: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: PosEncInput| {
    match input.op % 4 {
        0 => {
            // Sinusoidal positional embedding
            let seq_len = (input.seq_len as usize % 32) + 1;
            let dim = ((input.embed_dim as usize % 16) + 1) * 2; // must be even
            let pe = positional_embedding(seq_len, dim);
            assert_eq!(pe.len(), seq_len * dim, "PE shape mismatch");
            for (i, &v) in pe.iter().enumerate() {
                assert!(v.is_finite(), "PE non-finite at {i}: {v}");
                assert!(v.abs() <= 1.0 + 1e-6, "PE out of [-1,1] at {i}: {v}");
            }
        }
        1 => {
            // RoPE compute_frequencies + apply_rope
            let head_dim = ((input.embed_dim as usize % 16) + 1) * 2;
            let max_seq = (input.seq_len as usize % 32) + 1;
            let base = input.base.abs();
            if !base.is_finite() || base <= 0.0 {
                return;
            }

            let config = RopeConfig::new(head_dim, max_seq).with_base(base);
            let freqs = compute_frequencies(&config);
            assert_eq!(freqs.len(), head_dim / 2, "freq table size mismatch");
            for (i, &f) in freqs.iter().enumerate() {
                assert!(f.is_finite(), "freq non-finite at {i}: {f}");
            }

            // Apply to a single position
            let mut vec = bytes_to_f32(&input.data, head_dim);
            if vec.len() < head_dim {
                vec.resize(head_dim, 0.0);
            }
            for v in &mut vec {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }
            let norm_before: f32 = vec.iter().map(|x| x * x).sum();

            let pos = input.position as usize % max_seq;
            apply_rope(&mut vec, pos, head_dim, &freqs);

            for (i, &v) in vec.iter().enumerate() {
                assert!(v.is_finite(), "RoPE output non-finite at {i}: {v}");
            }
            // Norm preservation: RoPE is a rotation
            if norm_before > 1e-12 {
                let norm_after: f32 = vec.iter().map(|x| x * x).sum();
                let ratio = norm_after / norm_before;
                assert!((ratio - 1.0).abs() < 1e-4, "RoPE norm not preserved: ratio={ratio}");
            }
        }
        2 => {
            // RoPE batch application
            let head_dim = ((input.embed_dim as usize % 8) + 1) * 2;
            let num_heads = (input.indices.first().copied().unwrap_or(1) as usize % 4) + 1;
            let seq_len = (input.seq_len as usize % 8) + 1;
            let total = seq_len * num_heads * head_dim;
            let max_seq = seq_len + (input.position as usize % 8);

            let base = input.base.abs();
            if !base.is_finite() || base <= 0.0 {
                return;
            }
            let config = RopeConfig::new(head_dim, max_seq).with_base(base);
            let freqs = compute_frequencies(&config);

            let mut data = bytes_to_f32(&input.data, total);
            if data.len() < total {
                data.resize(total, 0.0);
            }
            for v in &mut data {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }

            let start_pos = input.position as usize % max_seq.saturating_sub(seq_len).max(1);
            apply_rope_batch(&mut data, start_pos, seq_len, num_heads, head_dim, &freqs);

            for (i, &v) in data.iter().enumerate() {
                assert!(v.is_finite(), "RoPE batch non-finite at {i}: {v}");
            }
        }
        _ => {
            // embedding_with_position: sinusoidal PE added to embeddings
            let vocab_size = (input.indices.first().copied().unwrap_or(4) as usize % 16) + 4;
            let dim = ((input.embed_dim as usize % 8) + 1) * 2;
            let table_size = vocab_size * dim;

            let mut table = bytes_to_f32(&input.data, table_size);
            if table.len() < table_size {
                table.resize(table_size, 0.0);
            }
            for v in &mut table {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }

            let indices: Vec<u32> =
                input.indices.iter().take(8).map(|&i| (i as u32) % (vocab_size as u32)).collect();
            if indices.is_empty() {
                return;
            }

            let config = CpuEmbeddingConfig::new(vocab_size, dim);
            let pos_offset = input.position as usize;
            if let Ok(result) = embedding_with_position(&table, &indices, &config, pos_offset) {
                assert_eq!(result.len(), indices.len() * dim);
                for (i, &v) in result.iter().enumerate() {
                    assert!(v.is_finite(), "embed+PE non-finite at {i}: {v}");
                }
            }
        }
    }
});
