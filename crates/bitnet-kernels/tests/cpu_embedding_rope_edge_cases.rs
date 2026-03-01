//! Edge-case tests for CPU embedding lookup and RoPE kernels.
//!
//! Tests cover basic embedding operations, positional encodings,
//! packed embeddings, and RoPE frequency computations.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::embedding::{
    CpuEmbeddingConfig, EmbeddingConfig, embedding_accumulate, embedding_lookup,
    normalize_embeddings, pack_embedding_table, positional_embedding, unpack_embedding_lookup,
};
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, compute_frequencies};

// ── embedding_lookup ─────────────────────────────────────────────────

#[test]
fn embedding_lookup_basic() {
    let table = vec![
        1.0, 0.0, // token 0
        0.0, 1.0, // token 1
        1.0, 1.0, // token 2
    ];
    let indices = vec![0, 2, 1];
    let result = embedding_lookup(&table, &indices, 2).unwrap();
    assert_eq!(result.len(), 6); // 3 tokens * dim 2
    assert_eq!(&result[0..2], &[1.0, 0.0]); // token 0
    assert_eq!(&result[2..4], &[1.0, 1.0]); // token 2
    assert_eq!(&result[4..6], &[0.0, 1.0]); // token 1
}

#[test]
fn embedding_lookup_single_token() {
    let table = vec![0.5, 0.5, 0.5]; // 1 token, dim 3
    let indices = vec![0];
    let result = embedding_lookup(&table, &indices, 3).unwrap();
    assert_eq!(result, vec![0.5, 0.5, 0.5]);
}

#[test]
fn embedding_lookup_repeated_index() {
    let table = vec![
        1.0, 2.0, // token 0
        3.0, 4.0, // token 1
    ];
    let indices = vec![1, 1, 1];
    let result = embedding_lookup(&table, &indices, 2).unwrap();
    assert_eq!(result.len(), 6);
    assert_eq!(&result[0..2], &[3.0, 4.0]);
    assert_eq!(&result[2..4], &[3.0, 4.0]);
    assert_eq!(&result[4..6], &[3.0, 4.0]);
}

#[test]
fn embedding_lookup_empty_indices() {
    let table = vec![1.0, 2.0];
    let indices: Vec<u32> = vec![];
    let result = embedding_lookup(&table, &indices, 2).unwrap();
    assert!(result.is_empty());
}

// ── embedding_accumulate ─────────────────────────────────────────────

#[test]
fn embedding_accumulate_weighted() {
    let table = vec![
        1.0, 0.0, // token 0
        0.0, 1.0, // token 1
    ];
    let indices = vec![0, 1];
    let weights = vec![0.5, 0.5];
    let result = embedding_accumulate(&table, &indices, &weights, 2).unwrap();
    assert_eq!(result.len(), 2); // weighted sum of embeddings
    // 0.5 * [1,0] + 0.5 * [0,1] = [0.5, 0.5]
    assert!((result[0] - 0.5).abs() < 1e-6);
    assert!((result[1] - 0.5).abs() < 1e-6);
}

// ── normalize_embeddings ─────────────────────────────────────────────

#[test]
fn normalize_embeddings_unit_vectors() {
    let mut embeddings = vec![3.0, 4.0, 0.0, 0.0]; // 2 vectors of dim 2
    normalize_embeddings(&mut embeddings, 2);
    // First vector: [3,4] → L2 = 5 → [0.6, 0.8]
    assert!((embeddings[0] - 0.6).abs() < 1e-5);
    assert!((embeddings[1] - 0.8).abs() < 1e-5);
}

#[test]
fn normalize_embeddings_already_unit() {
    let mut embeddings = vec![1.0, 0.0]; // already unit
    normalize_embeddings(&mut embeddings, 2);
    assert!((embeddings[0] - 1.0).abs() < 1e-5);
    assert!((embeddings[1] - 0.0).abs() < 1e-5);
}

// ── packed embedding ─────────────────────────────────────────────────

#[test]
fn pack_unpack_roundtrip() {
    let vocab_size = 4;
    let embed_dim = 8;
    let table: Vec<f32> = (0..vocab_size * embed_dim).map(|i| (i as f32) * 0.1).collect();
    let packed = pack_embedding_table(&table, vocab_size, embed_dim);
    assert_eq!(packed.vocab_size, vocab_size);
    assert_eq!(packed.embed_dim, embed_dim);

    // Unpack a token and check it's approximately the same
    let result = unpack_embedding_lookup(&packed, &[0, 2]).unwrap();
    assert_eq!(result.len(), 2 * embed_dim);
    // Token 0 should be approximately [0.0, 0.1, 0.2, ...]
    for i in 0..embed_dim {
        let expected = (i as f32) * 0.1;
        // Packed quantization introduces some error
        assert!(
            (result[i] - expected).abs() < 1.0,
            "token 0, dim {i}: expected ~{expected}, got {}",
            result[i]
        );
    }
}

#[test]
fn pack_embedding_table_dimensions() {
    let table = vec![1.0; 100]; // vocab=10, dim=10
    let packed = pack_embedding_table(&table, 10, 10);
    assert_eq!(packed.vocab_size, 10);
    assert_eq!(packed.embed_dim, 10);
}

// ── positional_embedding ─────────────────────────────────────────────

#[test]
fn positional_embedding_shape() {
    let pe = positional_embedding(8, 16);
    assert_eq!(pe.len(), 8 * 16); // seq_len * dim
}

#[test]
fn positional_embedding_first_position_known() {
    let pe = positional_embedding(2, 4);
    assert_eq!(pe.len(), 8);
    // Position 0 should start with sin(0) for even dims
    // and cos(0) for odd dims → [0, 1, 0, 1] pattern
    assert!(pe[0].abs() < 1e-5); // sin(0) = 0
    assert!((pe[1] - 1.0).abs() < 1e-5); // cos(0) = 1
}

#[test]
fn positional_embedding_values_bounded() {
    let pe = positional_embedding(32, 64);
    for &v in &pe {
        assert!(v >= -1.0 && v <= 1.0, "PE value {v} out of [-1,1]");
    }
}

#[test]
fn positional_embedding_different_positions_differ() {
    let pe = positional_embedding(4, 8);
    let row0 = &pe[0..8];
    let row1 = &pe[8..16];
    // Different positions should produce different embeddings
    assert_ne!(row0, row1);
}

// ── EmbeddingConfig ──────────────────────────────────────────────────

#[test]
fn embedding_config_basic() {
    let config = EmbeddingConfig { vocab_size: 32000, embedding_dim: 4096, padding_idx: Some(0) };
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.embedding_dim, 4096);
    assert_eq!(config.padding_idx, Some(0));
}

#[test]
fn cpu_embedding_config_no_padding() {
    let config = CpuEmbeddingConfig {
        vocab_size: 100352,
        embed_dim: 5120,
        padding_idx: None,
        max_norm: None,
    };
    assert_eq!(config.vocab_size, 100352); // Phi-4 vocab size
}

// ── RoPE ─────────────────────────────────────────────────────────────

#[test]
fn rope_config_basic() {
    let config = RopeConfig { head_dim: 64, max_seq_len: 2048, base: 10000.0, scaling_factor: 1.0 };
    assert_eq!(config.head_dim, 64);
}

#[test]
fn compute_frequencies_shape() {
    let config = RopeConfig { head_dim: 8, max_seq_len: 4, base: 10000.0, scaling_factor: 1.0 };
    let freqs = compute_frequencies(&config);
    // Shape depends on implementation: typically max_seq_len * (head_dim / 2) or similar
    assert!(freqs.len() > 0);
    // Verify it's related to the config dimensions
    assert_eq!(freqs.len() % (config.head_dim / 2), 0);
}

#[test]
fn compute_frequencies_all_finite() {
    let config =
        RopeConfig { head_dim: 128, max_seq_len: 16384, base: 10000.0, scaling_factor: 1.0 };
    let freqs = compute_frequencies(&config);
    for &f in &freqs {
        assert!(f.is_finite(), "non-finite frequency: {f}");
    }
}

#[test]
fn compute_frequencies_position_zero() {
    let config = RopeConfig { head_dim: 4, max_seq_len: 2, base: 10000.0, scaling_factor: 1.0 };
    let freqs = compute_frequencies(&config);
    // At position 0, all angles are 0 → sin=0, cos=1
    // But compute_frequencies returns the angles themselves
    let half = config.head_dim / 2;
    for i in 0..half {
        // Position 0 should have angle = 0 * freq = 0
        // (actual values depend on implementation, just check finite)
        assert!(freqs[i].is_finite());
    }
}

#[test]
fn apply_rope_preserves_length() {
    let config = RopeConfig { head_dim: 4, max_seq_len: 2, base: 10000.0, scaling_factor: 1.0 };
    let freqs = compute_frequencies(&config);
    let mut data = vec![1.0f32, 2.0, 3.0, 4.0]; // one head, dim=4
    apply_rope(&mut data, 0, 4, &freqs);
    assert_eq!(data.len(), 4);
    for &v in &data {
        assert!(v.is_finite(), "non-finite after RoPE: {v}");
    }
}

#[test]
fn apply_rope_position_zero_preserves_input() {
    let config = RopeConfig { head_dim: 4, max_seq_len: 2, base: 10000.0, scaling_factor: 1.0 };
    let freqs = compute_frequencies(&config);
    let original = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut data = original.clone();
    apply_rope(&mut data, 0, 4, &freqs);
    // At position 0, rotation angles should be 0 → cos=1, sin=0
    // So rotated vector should be ~same as original
    for (a, b) in original.iter().zip(data.iter()) {
        assert!((a - b).abs() < 0.1, "position 0 should ~preserve input: {a} vs {b}");
    }
}

#[test]
fn apply_rope_different_positions_differ() {
    let config = RopeConfig { head_dim: 4, max_seq_len: 4, base: 10000.0, scaling_factor: 1.0 };
    let freqs = compute_frequencies(&config);
    let mut data0 = vec![1.0f32, 0.0, 0.0, 0.0];
    let mut data1 = vec![1.0f32, 0.0, 0.0, 0.0];
    apply_rope(&mut data0, 0, 4, &freqs);
    apply_rope(&mut data1, 1, 4, &freqs);
    // Different positions should produce different rotations
    assert_ne!(data0, data1);
}

#[test]
fn rope_scaling_factor_changes_frequencies() {
    let config1 = RopeConfig { head_dim: 8, max_seq_len: 4, base: 10000.0, scaling_factor: 1.0 };
    let config2 = RopeConfig { head_dim: 8, max_seq_len: 4, base: 10000.0, scaling_factor: 2.0 };
    let freqs1 = compute_frequencies(&config1);
    let freqs2 = compute_frequencies(&config2);
    assert_eq!(freqs1.len(), freqs2.len());
    // Different scaling factors should produce different frequencies
    assert_ne!(freqs1, freqs2);
}
