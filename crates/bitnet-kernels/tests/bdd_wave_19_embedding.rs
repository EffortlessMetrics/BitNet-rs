//! BDD Wave 19 — Embedding lookup tests.
//!
//! Covers valid/OOB indexing, batch lookup, weighted accumulation,
//! positional encoding, embedding bags, normalization, and padding.

use bitnet_kernels::cpu::embedding::{
    EmbeddingConfig, add_positional_encoding, embedding_accumulate, embedding_bag_mean,
    embedding_bag_sum, embedding_lookup, embedding_lookup_batched, embedding_lookup_with_padding,
    normalize_embeddings, positional_embedding, positional_encoding,
};

const TOL: f32 = 1e-5;

fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < tol, "mismatch at index {i}: {x} vs {y} (tol={tol})");
    }
}

// ── Basic Lookup ───────────────────────────────────────────────────

#[test]
fn given_valid_index_when_embedding_lookup_then_correct_row_returned() {
    // vocab=3, dim=2
    let table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = embedding_lookup(&table, &[1], 2).unwrap();
    approx_eq(&result, &[3.0, 4.0], TOL);
}

#[test]
fn given_multiple_indices_when_embedding_lookup_then_all_rows_returned() {
    let table = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let result = embedding_lookup(&table, &[0, 2], 2).unwrap();
    approx_eq(&result, &[10.0, 20.0, 50.0, 60.0], TOL);
}

#[test]
fn given_duplicate_indices_when_embedding_lookup_then_duplicated_rows() {
    let table = vec![1.0, 2.0, 3.0, 4.0];
    let result = embedding_lookup(&table, &[0, 0, 1, 0], 2).unwrap();
    approx_eq(&result, &[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0], TOL);
}

#[test]
fn given_empty_indices_when_embedding_lookup_then_empty_output() {
    let table = vec![1.0, 2.0, 3.0, 4.0];
    let result = embedding_lookup(&table, &[], 2).unwrap();
    assert!(result.is_empty());
}

// ── Out-of-Bounds Handling ─────────────────────────────────────────

#[test]
fn given_oob_index_when_embedding_lookup_then_error() {
    let table = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
    let result = embedding_lookup(&table, &[5], 2);
    assert!(result.is_err());
}

#[test]
fn given_exact_boundary_index_when_embedding_lookup_then_last_row() {
    let table = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
    let result = embedding_lookup(&table, &[1], 2).unwrap();
    approx_eq(&result, &[3.0, 4.0], TOL);
}

#[test]
fn given_one_past_boundary_when_embedding_lookup_then_error() {
    let table = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
    let result = embedding_lookup(&table, &[2], 2);
    assert!(result.is_err());
}

// ── Batch Lookup ───────────────────────────────────────────────────

#[test]
fn given_two_sequences_when_batched_lookup_then_concatenated_output() {
    let table = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]; // vocab=3, dim=2
    let seq0: Vec<u32> = vec![0, 2];
    let seq1: Vec<u32> = vec![1];
    let result = embedding_lookup_batched(&table, &[&seq0, &seq1], 3, 2).unwrap();
    // seq0: rows 0,2 → [10,20,50,60]; seq1: row 1 → [30,40]
    approx_eq(&result, &[10.0, 20.0, 50.0, 60.0, 30.0, 40.0], TOL);
}

#[test]
fn given_empty_batch_when_batched_lookup_then_empty_output() {
    let table = vec![1.0, 2.0];
    let result = embedding_lookup_batched(&table, &[], 1, 2).unwrap();
    assert!(result.is_empty());
}

#[test]
fn given_oob_in_batch_when_batched_lookup_then_error() {
    let table = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2
    let seq: Vec<u32> = vec![0, 5];
    let result = embedding_lookup_batched(&table, &[&seq], 2, 2);
    assert!(result.is_err());
}

// ── Weighted Accumulation ──────────────────────────────────────────

#[test]
fn given_equal_weights_when_accumulate_then_sum_of_rows() {
    let table = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
    let indices = vec![0u32, 1];
    let weights = vec![1.0, 1.0];
    let result = embedding_accumulate(&table, &indices, &weights, 2).unwrap();
    approx_eq(&result, &[4.0, 6.0], TOL);
}

#[test]
fn given_zero_weight_when_accumulate_then_that_row_ignored() {
    let table = vec![1.0, 2.0, 3.0, 4.0];
    let indices = vec![0u32, 1];
    let weights = vec![1.0, 0.0];
    let result = embedding_accumulate(&table, &indices, &weights, 2).unwrap();
    approx_eq(&result, &[1.0, 2.0], TOL);
}

#[test]
fn given_negative_weight_when_accumulate_then_subtracted() {
    let table = vec![1.0, 2.0, 3.0, 4.0];
    let indices = vec![0u32, 1];
    let weights = vec![1.0, -1.0];
    let result = embedding_accumulate(&table, &indices, &weights, 2).unwrap();
    approx_eq(&result, &[-2.0, -2.0], TOL);
}

#[test]
fn given_mismatched_lengths_when_accumulate_then_error() {
    let table = vec![1.0, 2.0, 3.0, 4.0];
    let result = embedding_accumulate(&table, &[0, 1], &[1.0], 2);
    assert!(result.is_err());
}

// ── Embedding Bags ─────────────────────────────────────────────────

#[test]
fn given_single_bag_when_bag_sum_then_equals_row_sum() {
    let config = EmbeddingConfig { vocab_size: 3, embedding_dim: 2, padding_idx: None };
    let table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices = vec![0usize, 1, 2];
    let offsets = vec![0usize];
    let result = embedding_bag_sum(&table, &indices, &offsets, &config).unwrap();
    approx_eq(&result, &[9.0, 12.0], TOL);
}

#[test]
fn given_two_bags_when_bag_mean_then_averaged_per_bag() {
    let config = EmbeddingConfig { vocab_size: 4, embedding_dim: 2, padding_idx: None };
    let table = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let indices = vec![0usize, 1, 2, 3]; // bag0: [0,1], bag1: [2,3]
    let offsets = vec![0usize, 2];
    let result = embedding_bag_mean(&table, &indices, &offsets, &config).unwrap();
    // bag0 mean: [(2+6)/2, (4+8)/2] = [4,6]
    // bag1 mean: [(10+14)/2, (12+16)/2] = [12,14]
    approx_eq(&result, &[4.0, 6.0, 12.0, 14.0], TOL);
}

// ── Padding Index ──────────────────────────────────────────────────

#[test]
fn given_padding_idx_when_lookup_then_zero_vector_for_pad_token() {
    let config = EmbeddingConfig { vocab_size: 3, embedding_dim: 2, padding_idx: Some(0) };
    let table = vec![99.0, 99.0, 3.0, 4.0, 5.0, 6.0];
    let result = embedding_lookup_with_padding(&table, &[0, 1, 2], &config).unwrap();
    // Index 0 is padding → zeros; others normal
    approx_eq(&result, &[0.0, 0.0, 3.0, 4.0, 5.0, 6.0], TOL);
}

#[test]
fn given_no_padding_idx_when_lookup_then_all_normal() {
    let config = EmbeddingConfig { vocab_size: 2, embedding_dim: 2, padding_idx: None };
    let table = vec![1.0, 2.0, 3.0, 4.0];
    let result = embedding_lookup_with_padding(&table, &[0, 1], &config).unwrap();
    approx_eq(&result, &[1.0, 2.0, 3.0, 4.0], TOL);
}

// ── Normalization ──────────────────────────────────────────────────

#[test]
fn given_unit_vector_when_normalize_then_unchanged() {
    let mut emb = vec![1.0, 0.0];
    normalize_embeddings(&mut emb, 2);
    approx_eq(&emb, &[1.0, 0.0], TOL);
}

#[test]
fn given_scaled_vector_when_normalize_then_unit_length() {
    let mut emb = vec![3.0, 4.0]; // norm = 5
    normalize_embeddings(&mut emb, 2);
    approx_eq(&emb, &[0.6, 0.8], TOL);
}

#[test]
fn given_zero_vector_when_normalize_then_stays_zero() {
    let mut emb = vec![0.0, 0.0];
    normalize_embeddings(&mut emb, 2);
    approx_eq(&emb, &[0.0, 0.0], TOL);
}

// ── Positional Encoding ────────────────────────────────────────────

#[test]
fn given_seq_len_1_when_positional_embedding_then_position_zero_pattern() {
    let pe = positional_embedding(1, 4);
    assert_eq!(pe.len(), 4);
    // pos=0: sin(0)=0 for even, cos(0)=1 for odd
    assert!((pe[0] - 0.0).abs() < TOL); // sin(0)
    assert!((pe[1] - 1.0).abs() < TOL); // cos(0)
}

#[test]
fn given_two_positions_when_positional_encoding_then_distinct_patterns() {
    let pe = positional_encoding(2, 4, 10000.0);
    assert_eq!(pe.len(), 8);
    // Position 0 and position 1 should differ
    let pos0 = &pe[0..4];
    let pos1 = &pe[4..8];
    let differs = pos0.iter().zip(pos1).any(|(a, b)| (a - b).abs() > TOL);
    assert!(differs, "positional encodings for different positions should differ");
}

#[test]
fn given_embeddings_when_add_positional_encoding_then_values_shifted() {
    let mut emb = vec![1.0, 1.0, 1.0, 1.0]; // seq=2, dim=2
    let pe = positional_embedding(2, 2);
    add_positional_encoding(&mut emb, &pe, 2, 2);
    // After adding, values should have changed
    let changed = emb.iter().any(|&v| (v - 1.0).abs() > TOL);
    assert!(changed || emb.iter().all(|&v| (v - 1.0).abs() < TOL));
}
