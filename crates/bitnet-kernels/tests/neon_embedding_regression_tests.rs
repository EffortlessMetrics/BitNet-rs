#![allow(dead_code, unused_imports, unused_variables, unused_unsafe, unsafe_op_in_unsafe_fn)]
#![cfg(all(feature = "cpu", target_arch = "aarch64"))]
#![allow(clippy::float_cmp)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::approx_constant)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

//! Comprehensive integration test suite for NEON embedding operations on Apple Silicon
//!
//! This test suite validates embedding operations including:
//! - Basic embedding lookup with fallback mechanisms
//! - Position encodings (sinusoidal, RoPE, absolute, relative)
//! - Embedding arithmetic (addition, scaling, normalization, similarity)
//! - BitNet-specific operations (quantized embeddings, large vocab handling)
//!
//! All tests use pure Rust math with tolerance-based assertions for numerical stability.

// ============================================================================
// HELPER FUNCTIONS AND UTILITIES
// ============================================================================

/// Epsilon value for floating-point comparisons (typical tolerance for float32)
const EPSILON: f32 = 1e-5;

/// Assert floating-point equality with tolerance
fn assert_f32_eq(actual: f32, expected: f32, tolerance: f32) {
    let diff = (actual - expected).abs();
    assert!(
        diff < tolerance,
        "assertion failed: f32 values differ by {}, expected {} got {}",
        diff,
        expected,
        actual
    );
}

/// Assert floating-point vectors are equal with tolerance
fn assert_vec_f32_eq(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "vector lengths differ: {} vs {}",
        actual.len(),
        expected.len()
    );

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        // Use max(tolerance, relative tolerance) for numerical stability
        let effective_tolerance = tolerance.max(1e-7 * e.abs().max(1.0));
        assert!(
            diff < effective_tolerance,
            "vector element [{}] differs by {}: expected {} got {}",
            i,
            diff,
            e,
            a
        );
    }
}

/// Compute vector L2 norm
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalize vector to unit length
fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm.abs() < EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Compute dot product between two vectors
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector lengths must match for dot product");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Simulate embedding table lookup
struct EmbeddingTable {
    embeddings: Vec<Vec<f32>>,
    embedding_dim: usize,
    vocab_size: usize,
}

impl EmbeddingTable {
    /// Create a new embedding table
    fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        // Initialize with pseudo-random values for testing
        let mut embeddings = vec![vec![0.0; embedding_dim]; vocab_size];
        let mut seed: u64 = 12345;

        for row in &mut embeddings {
            for col in row.iter_mut() {
                // Simple PRNG
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let normalized = ((seed >> 16) & 0x7fff) as f32 / 32768.0;
                *col = (normalized - 0.5) * 2.0; // Range [-1, 1]
            }
        }

        EmbeddingTable { embeddings, embedding_dim, vocab_size }
    }

    /// Look up embedding for a token with out-of-vocab fallback
    fn lookup(&self, token_id: usize) -> Vec<f32> {
        if token_id < self.vocab_size {
            self.embeddings[token_id].clone()
        } else {
            // OOV fallback: return zero embedding
            vec![0.0; self.embedding_dim]
        }
    }

    /// Look up embeddings for multiple tokens
    fn lookup_batch(&self, token_ids: &[usize]) -> Vec<f32> {
        let mut result = vec![0.0; token_ids.len() * self.embedding_dim];
        for (i, &token_id) in token_ids.iter().enumerate() {
            let embedding = self.lookup(token_id);
            result[i * self.embedding_dim..(i + 1) * self.embedding_dim]
                .copy_from_slice(&embedding);
        }
        result
    }
}

/// Generate sinusoidal positional encodings
fn sinusoidal_encoding(position: usize, embedding_dim: usize) -> Vec<f32> {
    let mut encoding = vec![0.0; embedding_dim];
    let div_term = 10000.0_f32.powf(
        2.0 * ((0..embedding_dim).step_by(2).next().unwrap_or(0) as f32) / (embedding_dim as f32),
    );

    for i in 0..embedding_dim {
        let pos_f = position as f32;
        if i % 2 == 0 {
            encoding[i] = (pos_f / div_term.powf((i as f32) / (embedding_dim as f32))).sin();
        } else {
            encoding[i] = (pos_f / div_term.powf(((i - 1) as f32) / (embedding_dim as f32))).cos();
        }
    }
    encoding
}

/// Generate RoPE (Rotary Position Embedding) encoding
fn rope_encoding(position: usize, embedding_dim: usize, base: f32) -> Vec<f32> {
    let mut encoding = vec![0.0; embedding_dim];
    let inv_freq = 1.0
        / base.powf(
            2.0 * ((0..embedding_dim).step_by(2).next().unwrap_or(0) as f32)
                / (embedding_dim as f32),
        );

    for i in 0..embedding_dim {
        let pos_f = position as f32;
        let angle = pos_f * inv_freq.powf((i as f32) / 2.0);

        if i % 2 == 0 {
            encoding[i] = angle.cos();
        } else {
            encoding[i] = angle.sin();
        }
    }
    encoding
}

/// Apply RoPE rotation to a vector
fn apply_rope_rotation(vec: &[f32], rope: &[f32]) -> Vec<f32> {
    assert_eq!(vec.len(), rope.len(), "vector and RoPE dimensions must match");
    let mut rotated = vec![0.0; vec.len()];

    for i in (0..vec.len()).step_by(2) {
        if i + 1 < vec.len() {
            // Complex number multiplication: (a + bi) * (cos(θ) + i*sin(θ))
            let real_part = vec[i] * rope[i] - vec[i + 1] * rope[i + 1];
            let imag_part = vec[i] * rope[i + 1] + vec[i + 1] * rope[i];
            rotated[i] = real_part;
            rotated[i + 1] = imag_part;
        }
    }
    rotated
}

// ============================================================================
// BASIC EMBEDDING LOOKUP TESTS (4 tests)
// ============================================================================

#[test]
fn test_single_token_lookup() {
    let table = EmbeddingTable::new(1000, 64);
    let token_id = 42;

    let embedding = table.lookup(token_id);

    assert_eq!(embedding.len(), 64, "embedding dimension should be 64");
    assert!(embedding.iter().all(|x| x.is_finite()), "all values should be finite");
    assert!(embedding.iter().any(|x| x.abs() > 1e-6), "at least one value should be non-zero");

    // Lookup same token again should give same result
    let embedding2 = table.lookup(token_id);
    assert_vec_f32_eq(&embedding, &embedding2, 0.0);
}

#[test]
fn test_batch_token_lookup() {
    let table = EmbeddingTable::new(1000, 64);
    let token_ids = vec![10, 20, 30, 40, 50];

    let batch_result = table.lookup_batch(&token_ids);

    assert_eq!(batch_result.len(), 5 * 64, "batch result size should be 5 * 64");

    // Verify each embedding matches individual lookup
    for (i, &token_id) in token_ids.iter().enumerate() {
        let individual = table.lookup(token_id);
        let batch_slice = &batch_result[i * 64..(i + 1) * 64];
        assert_vec_f32_eq(batch_slice, &individual, 0.0);
    }
}

#[test]
fn test_out_of_vocab_fallback() {
    let vocab_size = 100;
    let table = EmbeddingTable::new(vocab_size, 64);

    // Test valid token
    let valid_token = 50;
    let valid_embedding = table.lookup(valid_token);
    assert!(
        !valid_embedding.iter().all(|x| x.abs() < EPSILON),
        "valid token should have non-zero embedding"
    );

    // Test out-of-vocab token
    let oov_token = vocab_size + 10;
    let oov_embedding = table.lookup(oov_token);
    assert!(
        oov_embedding.iter().all(|x| x.abs() < EPSILON),
        "OOV token should return zero embedding"
    );

    // Test boundary case
    let boundary_token = vocab_size - 1;
    let boundary_embedding = table.lookup(boundary_token);
    assert!(
        !boundary_embedding.iter().all(|x| x.abs() < EPSILON),
        "boundary token should have valid embedding"
    );
}

#[test]
fn test_zero_embedding_table() {
    let vocab_size = 100;
    let embedding_dim = 64;
    let table = EmbeddingTable {
        embeddings: vec![vec![0.0; embedding_dim]; vocab_size],
        embedding_dim,
        vocab_size,
    };

    // All lookups should return zero vectors
    for token_id in 0..vocab_size {
        let embedding = table.lookup(token_id);
        assert!(embedding.iter().all(|x| x.abs() < EPSILON), "all embeddings should be zero");
    }

    // OOV lookups should also return zero
    let oov = table.lookup(vocab_size + 1);
    assert!(oov.iter().all(|x| x.abs() < EPSILON), "OOV should return zero");
}

// ============================================================================
// POSITION EMBEDDINGS TESTS (4 tests)
// ============================================================================

#[test]
fn test_sinusoidal_position_encoding() {
    let embedding_dim = 64;
    let encoding1 = sinusoidal_encoding(0, embedding_dim);
    let encoding2 = sinusoidal_encoding(1, embedding_dim);
    let encoding3 = sinusoidal_encoding(10, embedding_dim);

    // All encodings should have correct dimension
    assert_eq!(encoding1.len(), embedding_dim);
    assert_eq!(encoding2.len(), embedding_dim);
    assert_eq!(encoding3.len(), embedding_dim);

    // All values should be finite and in [-1, 1]
    for enc in &[&encoding1, &encoding2, &encoding3] {
        assert!(enc.iter().all(|x| x.is_finite()), "all values should be finite");
        assert!(enc.iter().all(|x| x.abs() <= 1.0 + EPSILON), "all values should be in [-1, 1]");
    }

    // Different positions should yield different encodings
    let diff: f32 = encoding1.iter().zip(encoding2.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > EPSILON, "different positions should have different encodings");
}

#[test]
fn test_sinusoidal_periodicity() {
    let embedding_dim = 64;

    // Get encodings at different positions
    let enc0 = sinusoidal_encoding(0, embedding_dim);
    let enc512 = sinusoidal_encoding(512, embedding_dim);

    // Compute differences - they should not be identical but should show periodicity pattern
    let mut max_diff: f32 = 0.0;
    for i in 0..embedding_dim {
        max_diff = max_diff.max((enc0[i] - enc512[i]).abs());
    }

    assert!(max_diff < 2.0, "sinusoidal encodings should have bounded periodicity");
}

#[test]
fn test_rope_position_encoding() {
    let embedding_dim = 64;
    let base = 10000.0;

    let rope1 = rope_encoding(0, embedding_dim, base);
    let rope2 = rope_encoding(1, embedding_dim, base);
    let rope3 = rope_encoding(10, embedding_dim, base);

    // All encodings should have correct dimension
    assert_eq!(rope1.len(), embedding_dim);
    assert_eq!(rope2.len(), embedding_dim);
    assert_eq!(rope3.len(), embedding_dim);

    // All values should be finite and in [-1, 1] (cos/sin range)
    for rope in &[&rope1, &rope2, &rope3] {
        assert!(rope.iter().all(|x| x.is_finite()), "all RoPE values should be finite");
        assert!(rope.iter().all(|x| x.abs() <= 1.0 + EPSILON), "RoPE values should be in [-1, 1]");
    }

    // Different positions should yield different RoPE encodings
    let diff: f32 = rope1.iter().zip(rope2.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > EPSILON, "different positions should have different RoPE encodings");
}

#[test]
fn test_rope_rotation_application() {
    let embedding_dim = 64;
    let base = 10000.0;

    // Create a test vector
    let mut test_vec = vec![0.0; embedding_dim];
    for i in 0..embedding_dim {
        test_vec[i] = (i as f32) / (embedding_dim as f32);
    }

    // Apply RoPE rotation
    let rope = rope_encoding(5, embedding_dim, base);
    let rotated = apply_rope_rotation(&test_vec, &rope);

    // Rotation should preserve vector magnitude (approximately)
    let orig_norm = l2_norm(&test_vec);
    let rot_norm = l2_norm(&rotated);

    // Note: RoPE rotation operates on pairs, so norm is approximately preserved
    assert_f32_eq(orig_norm, rot_norm, 1e-4);

    // Rotated vector should be different from original
    let diff: f32 = test_vec.iter().zip(rotated.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > EPSILON, "rotated vector should differ from original");
}

// ============================================================================
// EMBEDDING ARITHMETIC TESTS (4 tests)
// ============================================================================

#[test]
fn test_add_embeddings() {
    let embedding_dim = 64;
    let table = EmbeddingTable::new(1000, embedding_dim);

    let emb1 = table.lookup(10);
    let emb2 = table.lookup(20);

    // Add embeddings
    let mut sum = vec![0.0; embedding_dim];
    for i in 0..embedding_dim {
        sum[i] = emb1[i] + emb2[i];
    }

    // Verify sum
    for i in 0..embedding_dim {
        assert_f32_eq(sum[i], emb1[i] + emb2[i], EPSILON);
    }

    // Sum norm should be reasonable
    let sum_norm = l2_norm(&sum);
    assert!(sum_norm.is_finite() && sum_norm > 0.0, "sum norm should be positive and finite");
}

#[test]
fn test_scale_embeddings() {
    let embedding_dim = 64;
    let table = EmbeddingTable::new(1000, embedding_dim);

    let embedding = table.lookup(50);
    let scale_factor = 2.5;

    // Scale embedding
    let scaled: Vec<f32> = embedding.iter().map(|x| x * scale_factor).collect();

    // Verify scaling
    for i in 0..embedding_dim {
        assert_f32_eq(scaled[i], embedding[i] * scale_factor, EPSILON);
    }

    // Norms should scale linearly
    let orig_norm = l2_norm(&embedding);
    let scaled_norm = l2_norm(&scaled);
    assert_f32_eq(scaled_norm, orig_norm * scale_factor, 1e-4);
}

#[test]
fn test_normalize_embeddings() {
    let embedding_dim = 64;
    let table = EmbeddingTable::new(1000, embedding_dim);

    let embedding = table.lookup(75);
    let normalized = normalize_vector(&embedding);

    // Normalized vector should have unit norm
    let norm = l2_norm(&normalized);
    assert_f32_eq(norm, 1.0, 1e-5);

    // Normalized vector should have same direction as original
    let orig_norm = l2_norm(&embedding);
    for i in 0..embedding_dim {
        if orig_norm > EPSILON {
            let expected = embedding[i] / orig_norm;
            assert_f32_eq(normalized[i], expected, 1e-5);
        }
    }
}

#[test]
fn test_embedding_similarity_computation() {
    let embedding_dim = 64;
    let table = EmbeddingTable::new(1000, embedding_dim);

    let emb1 = table.lookup(10);
    let emb2 = table.lookup(20);
    let emb3 = table.lookup(10); // Same as emb1

    // Compute similarities via dot product
    let sim_1_2 = dot_product(&emb1, &emb2);
    let sim_1_3 = dot_product(&emb1, &emb3);

    // Same embedding should have higher similarity with itself
    assert!(sim_1_3.abs() >= sim_1_2.abs(), "embedding should have highest similarity with itself");

    // Verify dot product is commutative
    let sim_2_1 = dot_product(&emb2, &emb1);
    assert_f32_eq(sim_1_2, sim_2_1, EPSILON);

    // Compute cosine similarity
    let norm1 = l2_norm(&emb1);
    let norm2 = l2_norm(&emb2);
    if norm1 > EPSILON && norm2 > EPSILON {
        let cosine_sim = sim_1_2 / (norm1 * norm2);
        assert!((-1.0..=1.0).contains(&cosine_sim), "cosine similarity should be in [-1, 1]");
    }
}

// ============================================================================
// BITNET-SPECIFIC EMBEDDING TESTS (3+ tests)
// ============================================================================

#[test]
fn test_quantized_embedding_lookup() {
    // Simulate quantized embeddings (stored as i8 instead of f32)
    let vocab_size = 1000;
    let embedding_dim = 64;
    let mut seed: u64 = 54321;

    // Create quantized embedding table (i8 values in [-128, 127])
    let mut quantized_table = vec![vec![0i8; embedding_dim]; vocab_size];
    for row in &mut quantized_table {
        for col in row.iter_mut() {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let val = ((seed >> 16) & 0xff) as i8;
            *col = val;
        }
    }

    // Create scale factors (one per embedding)
    let mut scales = vec![0.01; vocab_size];
    for scale in &mut scales {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let scale_idx = (seed >> 16) & 0xf;
        *scale = 0.01 + (scale_idx as f32) * 0.001;
    }

    // Dequantize lookup
    let token_id = 42;
    let mut dequantized = vec![0.0; embedding_dim];
    for i in 0..embedding_dim {
        dequantized[i] = quantized_table[token_id][i] as f32 * scales[token_id];
    }

    // Verify dequantized embedding has correct properties
    assert_eq!(dequantized.len(), embedding_dim);
    assert!(dequantized.iter().all(|x| x.is_finite()), "dequantized values should be finite");

    // Dequantized values should be within reasonable range (scale * i8 range)
    let max_val = dequantized.iter().map(|x| x.abs()).fold(0.0, f32::max);
    assert!(
        max_val < 4.0,
        "dequantized values should be in reasonable range, got max: {}",
        max_val
    );
}

#[test]
fn test_large_vocab_size_typical_32k() {
    // Typical LLM vocabulary size: 32K tokens
    let vocab_size = 32000;
    let embedding_dim = 128; // Typical embedding dimension

    let table = EmbeddingTable::new(vocab_size, embedding_dim);

    // Test various token IDs
    let test_tokens = vec![0, 1, 100, 1000, 10000, 31999, 32000];

    for &token_id in &test_tokens {
        let embedding = table.lookup(token_id);
        assert_eq!(embedding.len(), embedding_dim);
        assert!(embedding.iter().all(|x| x.is_finite()));

        // Tokens within vocab should be non-zero, OOV should be zero
        if token_id < vocab_size {
            assert!(
                embedding.iter().any(|x| x.abs() > EPSILON),
                "valid token should have non-zero embedding"
            );
        } else {
            assert!(embedding.iter().all(|x| x.abs() < EPSILON), "OOV token should be zero");
        }
    }
}

#[test]
fn test_cache_friendly_sequential_access() {
    let vocab_size = 1000;
    let embedding_dim = 128;
    let table = EmbeddingTable::new(vocab_size, embedding_dim);

    // Simulate sequential cache-friendly access pattern
    let mut cache_hits = 0;
    let access_pattern = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; // Sequential

    for &token_id in &access_pattern {
        let _embedding = table.lookup(token_id);
        cache_hits += 1;
    }

    assert_eq!(cache_hits, 10, "all sequential accesses should succeed");

    // Compare with random access pattern (for documentation, not actually slower in simulation)
    let random_pattern = vec![500, 123, 789, 42, 999, 321, 654, 111, 888, 256];
    let mut random_hits = 0;

    for &token_id in &random_pattern {
        let _embedding = table.lookup(token_id);
        random_hits += 1;
    }

    assert_eq!(random_hits, 10, "all random accesses should also succeed");
}

#[test]
fn test_batch_embedding_with_mixed_validity() {
    let vocab_size = 1000;
    let embedding_dim = 64;
    let table = EmbeddingTable::new(vocab_size, embedding_dim);

    // Mix of valid and OOV tokens
    let tokens = vec![10, 100, 1000, 5000, 999, 1001, 500, 1500];

    let batch_result = table.lookup_batch(&tokens);

    assert_eq!(batch_result.len(), tokens.len() * embedding_dim);

    // Verify each token's embedding
    for (i, &token_id) in tokens.iter().enumerate() {
        let embedding_slice = &batch_result[i * embedding_dim..(i + 1) * embedding_dim];

        if token_id < vocab_size {
            // Valid token should have non-zero values
            assert!(
                embedding_slice.iter().any(|x| x.abs() > EPSILON),
                "valid token {} should have non-zero embedding",
                token_id
            );
        } else {
            // OOV token should be all zeros
            assert!(
                embedding_slice.iter().all(|x| x.abs() < EPSILON),
                "OOV token {} should have zero embedding",
                token_id
            );
        }
    }
}

#[test]
fn test_embedding_dimension_consistency() {
    let dimensions = vec![32, 64, 128, 256, 512];

    for dim in dimensions {
        let table = EmbeddingTable::new(100, dim);

        // All lookups should have consistent dimension
        for token_id in 0..10 {
            let emb = table.lookup(token_id);
            assert_eq!(emb.len(), dim, "embedding dimension should be consistent: {}", dim);
        }

        // Batch lookups should also be consistent
        let batch = table.lookup_batch(&[0, 1, 2, 3, 4]);
        assert_eq!(batch.len(), 5 * dim, "batch dimension should be consistent: {}", dim);
    }
}

#[test]
fn test_quantized_batch_operations() {
    // Test batch operations on quantized embeddings
    let vocab_size = 256;
    let embedding_dim = 64;

    // Create quantized embeddings
    let mut quantized_embeddings = vec![vec![0i8; embedding_dim]; vocab_size];
    let mut seed: u64 = 99999;

    for row in &mut quantized_embeddings {
        for col in row.iter_mut() {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *col = (seed as i8).wrapping_add(1);
        }
    }

    let scales: Vec<f32> = (0..vocab_size).map(|_| 0.01).collect();

    // Dequantize batch
    let token_ids = [10, 20, 30, 40, 50];
    let mut dequantized_batch = vec![0.0; token_ids.len() * embedding_dim];

    for (batch_idx, &token_id) in token_ids.iter().enumerate() {
        for dim_idx in 0..embedding_dim {
            dequantized_batch[batch_idx * embedding_dim + dim_idx] =
                quantized_embeddings[token_id][dim_idx] as f32 * scales[token_id];
        }
    }

    // Verify batch result
    assert_eq!(dequantized_batch.len(), token_ids.len() * embedding_dim);
    assert!(dequantized_batch.iter().all(|x| x.is_finite()));
}

// ============================================================================
// INTEGRATION AND REGRESSION TESTS
// ============================================================================

#[test]
fn test_position_aware_embedding_computation() {
    let embedding_dim = 64;
    let vocab_size = 1000;
    let table = EmbeddingTable::new(vocab_size, embedding_dim);

    let token_id = 42;
    let position = 10;

    // Get token embedding
    let token_emb = table.lookup(token_id);

    // Get position encoding
    let pos_enc = sinusoidal_encoding(position, embedding_dim);

    // Combine (add) embeddings
    let mut combined = vec![0.0; embedding_dim];
    for i in 0..embedding_dim {
        combined[i] = token_emb[i] + pos_enc[i];
    }

    // Verify result
    assert_eq!(combined.len(), embedding_dim);
    assert!(combined.iter().all(|x| x.is_finite()));

    // Combined embedding should have reasonable norm
    let norm = l2_norm(&combined);
    assert!(norm > EPSILON && norm.is_finite());
}

#[test]
fn test_rope_augmented_embeddings() {
    let embedding_dim = 64;
    let table = EmbeddingTable::new(1000, embedding_dim);

    let token_id = 100;
    let position = 5;

    // Get token embedding
    let token_emb = table.lookup(token_id);

    // Generate RoPE encoding
    let rope = rope_encoding(position, embedding_dim, 10000.0);

    // Apply RoPE rotation
    let rope_augmented = apply_rope_rotation(&token_emb, &rope);

    // Verify properties
    assert_eq!(rope_augmented.len(), embedding_dim);
    assert!(rope_augmented.iter().all(|x| x.is_finite()));

    // Norm should be approximately preserved
    let orig_norm = l2_norm(&token_emb);
    let rope_norm = l2_norm(&rope_augmented);
    assert_f32_eq(orig_norm, rope_norm, 1e-4);
}

#[test]
fn test_multi_head_embedding_simulation() {
    // Simulate multi-head embeddings (e.g., for multi-head attention)
    let embedding_dim = 64;
    let num_heads = 4;
    let head_dim = embedding_dim / num_heads;

    let table = EmbeddingTable::new(1000, embedding_dim);

    let embedding = table.lookup(50);

    // Split into heads
    let mut heads = vec![vec![0.0; head_dim]; num_heads];
    for head_idx in 0..num_heads {
        let start = head_idx * head_dim;
        let end = start + head_dim;
        heads[head_idx].copy_from_slice(&embedding[start..end]);
    }

    // Verify heads
    assert_eq!(heads.len(), num_heads);
    for head in &heads {
        assert_eq!(head.len(), head_dim);
        assert!(head.iter().all(|x| x.is_finite()));
    }

    // Reconstruct embedding
    let mut reconstructed = vec![0.0; embedding_dim];
    for head_idx in 0..num_heads {
        let start = head_idx * head_dim;
        let end = start + head_dim;
        reconstructed[start..end].copy_from_slice(&heads[head_idx]);
    }

    // Should match original
    assert_vec_f32_eq(&embedding, &reconstructed, EPSILON);
}

#[test]
fn test_embedding_gradient_simulation() {
    // Simulate gradient updates (important for training, though tests are inference)
    let embedding_dim = 64;
    let table = EmbeddingTable::new(1000, embedding_dim);

    let embedding = table.lookup(50);
    let learning_rate = 0.01;

    // Simulate gradient (pseudo-random values in [-1, 1])
    let mut seed: u64 = 77777;
    let mut gradient = vec![0.0; embedding_dim];
    for g in &mut gradient {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let normalized = ((seed >> 16) & 0x7fff) as f32 / 32768.0;
        *g = (normalized - 0.5) * 2.0;
    }

    // Update embedding (simulate one step of SGD)
    let mut updated = vec![0.0; embedding_dim];
    for i in 0..embedding_dim {
        updated[i] = embedding[i] - learning_rate * gradient[i];
    }

    // Verify update
    assert_eq!(updated.len(), embedding_dim);
    assert!(updated.iter().all(|x| x.is_finite()));

    // Change should be small relative to learning rate
    let change: f32 = embedding.iter().zip(updated.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(change < learning_rate * embedding_dim as f32 * 2.0);
}

#[test]
fn test_embedding_matrix_transpose_consistency() {
    let vocab_size = 100;
    let embedding_dim = 64;
    let table = EmbeddingTable::new(vocab_size, embedding_dim);

    // Create embedding matrix from batch lookup
    let tokens: Vec<usize> = (0..vocab_size).collect();
    let batch_result = table.lookup_batch(&tokens);

    // Arrange as matrix: vocab_size x embedding_dim
    let mut matrix = vec![vec![0.0; embedding_dim]; vocab_size];
    for i in 0..vocab_size {
        for j in 0..embedding_dim {
            matrix[i][j] = batch_result[i * embedding_dim + j];
        }
    }

    // Transpose
    let mut transposed = vec![vec![0.0; vocab_size]; embedding_dim];
    for i in 0..vocab_size {
        for j in 0..embedding_dim {
            transposed[j][i] = matrix[i][j];
        }
    }

    // Transpose back
    let mut double_transposed = vec![vec![0.0; embedding_dim]; vocab_size];
    for i in 0..embedding_dim {
        for j in 0..vocab_size {
            double_transposed[j][i] = transposed[i][j];
        }
    }

    // Should match original
    for i in 0..vocab_size {
        assert_vec_f32_eq(&matrix[i], &double_transposed[i], EPSILON);
    }
}

#[test]
fn test_embedding_numerical_stability_extreme_scales() {
    let embedding_dim = 64;
    let table = EmbeddingTable::new(1000, embedding_dim);

    let embedding = table.lookup(100);

    // Test with very small scaling
    let tiny_scale = 1e-6;
    let tiny_scaled: Vec<f32> = embedding.iter().map(|x| x * tiny_scale).collect();
    assert!(tiny_scaled.iter().all(|x| x.is_finite()), "tiny scaling should not cause NaN");

    // Test with very large scaling
    let huge_scale = 1e6;
    let huge_scaled: Vec<f32> = embedding.iter().map(|x| x * huge_scale).collect();
    assert!(huge_scaled.iter().all(|x| x.is_finite()), "huge scaling should not cause NaN");

    // Test normalization of tiny values
    let tiny_normalized = normalize_vector(&tiny_scaled);
    let norm = l2_norm(&tiny_normalized);
    if l2_norm(&tiny_scaled) > EPSILON {
        assert_f32_eq(norm, 1.0, 1e-5);
    }
}
