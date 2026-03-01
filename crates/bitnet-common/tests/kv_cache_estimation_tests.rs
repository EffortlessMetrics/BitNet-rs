//! KV Cache memory estimation tests
//!
//! Tests the KV cache memory estimation formula:
//! cache_size = 2 * num_layers * num_kv_heads * head_dim * max_seq_len * sizeof(f32)
//! where head_dim = hidden_size / num_heads

use bitnet_common::ModelConfig;

/// Calculate KV cache size in bytes using the estimation formula
fn calculate_kv_cache_size(config: &ModelConfig) -> usize {
    let num_kv_heads =
        if config.num_key_value_heads > 0 { config.num_key_value_heads } else { config.num_heads };

    let head_dim = config.hidden_size / config.num_heads;
    let sizeof_f32 = 4; // f32 is 4 bytes

    // 2 for K and V caches
    2 * config.num_layers * num_kv_heads * head_dim * config.max_position_embeddings * sizeof_f32
}

/// Convert bytes to megabytes
fn bytes_to_mb(bytes: usize) -> f32 {
    bytes as f32 / (1024.0 * 1024.0)
}

/// Convert bytes to gigabytes
fn bytes_to_gb(bytes: usize) -> f32 {
    bytes as f32 / (1024.0 * 1024.0 * 1024.0)
}

// ============================================================================
// Basic Model Configuration Tests
// ============================================================================

#[test]
fn test_bitnet_2b_kv_cache_estimation() {
    // BitNet 2B config: 30 layers, 20 heads, 5 KV heads, hidden=2560, max_seq=4096
    // head_dim = 2560 / 20 = 128
    // cache = 2 * 30 * 5 * 128 * 4096 * 4 bytes = ~600MB
    let config = ModelConfig {
        num_layers: 30,
        num_heads: 20,
        num_key_value_heads: 5,
        hidden_size: 2560,
        max_position_embeddings: 4096,
        ..Default::default()
    };

    let cache_size = calculate_kv_cache_size(&config);
    let head_dim = config.hidden_size / config.num_heads;

    assert_eq!(head_dim, 128, "BitNet 2B head_dim should be 128");
    assert_eq!(cache_size, 2 * 30 * 5 * 128 * 4096 * 4);

    let cache_mb = bytes_to_mb(cache_size);
    assert!(
        cache_mb > 580.0 && cache_mb < 620.0,
        "BitNet 2B cache should be ~600MB, got {:.1}MB",
        cache_mb
    );
}

#[test]
fn test_phi_4_14b_kv_cache_estimation() {
    // Phi-4 14B config: 40 layers, 40 heads, 10 KV heads, hidden=5120, max_seq=16384
    // head_dim = 5120 / 40 = 128
    // cache = 2 * 40 * 10 * 128 * 16384 * 4 bytes = ~13GB
    let config = ModelConfig {
        num_layers: 40,
        num_heads: 40,
        num_key_value_heads: 10,
        hidden_size: 5120,
        max_position_embeddings: 16384,
        ..Default::default()
    };

    let cache_size = calculate_kv_cache_size(&config);
    let head_dim = config.hidden_size / config.num_heads;

    assert_eq!(head_dim, 128, "Phi-4 head_dim should be 128");
    assert_eq!(cache_size, 2 * 40 * 10 * 128 * 16384 * 4);

    let cache_gb = bytes_to_gb(cache_size);
    assert!(
        cache_gb > 6.0 && cache_gb < 7.0,
        "Phi-4 cache should be ~6.25GB, got {:.2}GB",
        cache_gb
    );
}

#[test]
fn test_llama_7b_kv_cache_estimation() {
    // LLaMA 7B config: 32 layers, 32 heads, 32 KV heads (no GQA), hidden=4096, max_seq=4096
    // head_dim = 4096 / 32 = 128
    // cache = 2 * 32 * 32 * 128 * 4096 * 4 bytes
    let config = ModelConfig {
        num_layers: 32,
        num_heads: 32,
        num_key_value_heads: 32,
        hidden_size: 4096,
        max_position_embeddings: 4096,
        ..Default::default()
    };

    let cache_size = calculate_kv_cache_size(&config);
    let head_dim = config.hidden_size / config.num_heads;

    assert_eq!(head_dim, 128, "LLaMA 7B head_dim should be 128");
    assert_eq!(cache_size, 2 * 32 * 32 * 128 * 4096 * 4);

    let cache_gb = bytes_to_gb(cache_size);
    assert!(
        cache_gb > 3.5 && cache_gb < 4.5,
        "LLaMA 7B cache should be ~4GB, got {:.2}GB",
        cache_gb
    );
}

#[test]
fn test_small_model_kv_cache_estimation() {
    // Small test model: 8 layers, 8 heads, 8 KV heads, hidden=512, max_seq=512
    // head_dim = 512 / 8 = 64
    // cache = 2 * 8 * 8 * 64 * 512 * 4 bytes = 2MB
    let config = ModelConfig {
        num_layers: 8,
        num_heads: 8,
        num_key_value_heads: 8,
        hidden_size: 512,
        max_position_embeddings: 512,
        ..Default::default()
    };

    let cache_size = calculate_kv_cache_size(&config);
    let head_dim = config.hidden_size / config.num_heads;

    assert_eq!(head_dim, 64, "Small model head_dim should be 64");
    assert_eq!(cache_size, 2 * 8 * 8 * 64 * 512 * 4);

    let cache_mb = bytes_to_mb(cache_size);
    assert!(
        cache_mb > 15.0 && cache_mb < 17.0,
        "Small model cache should be ~16MB, got {:.2}MB",
        cache_mb
    );
}

// ============================================================================
// Head Dimension Verification Tests
// ============================================================================

#[test]
fn test_head_dim_calculation() {
    // Verify head_dim = hidden_size / num_heads for various configs
    let test_cases = vec![
        (512, 8, 64),
        (1024, 8, 128),
        (2048, 16, 128),
        (4096, 32, 128),
        (5120, 40, 128),
        (8192, 64, 128),
    ];

    for (hidden_size, num_heads, expected_head_dim) in test_cases {
        let config = ModelConfig { hidden_size, num_heads, ..Default::default() };
        let head_dim = config.hidden_size / config.num_heads;
        assert_eq!(
            head_dim, expected_head_dim,
            "For hidden_size={}, num_heads={}, expected head_dim={}, got {}",
            hidden_size, num_heads, expected_head_dim, head_dim
        );
    }
}

#[test]
fn test_head_dim_divisibility() {
    // Verify that hidden_size is properly divisible by num_heads
    let config = ModelConfig { hidden_size: 4096, num_heads: 32, ..Default::default() };

    assert_eq!(config.hidden_size % config.num_heads, 0);

    let head_dim = config.hidden_size / config.num_heads;
    assert_eq!(head_dim * config.num_heads, config.hidden_size);
}

// ============================================================================
// Linear Scaling Tests
// ============================================================================

#[test]
fn test_cache_scales_linearly_with_sequence_length() {
    // Test that cache size scales linearly with max_seq_len
    let base_config = ModelConfig {
        num_layers: 10,
        num_heads: 8,
        num_key_value_heads: 8,
        hidden_size: 512,
        max_position_embeddings: 1024,
        ..Default::default()
    };

    let cache_size_1k = calculate_kv_cache_size(&base_config);

    // Double the sequence length
    let doubled_config = ModelConfig { max_position_embeddings: 2048, ..base_config.clone() };
    let cache_size_2k = calculate_kv_cache_size(&doubled_config);

    assert_eq!(cache_size_2k, cache_size_1k * 2, "Doubling seq_len should double cache size");

    // Triple the sequence length
    let tripled_config = ModelConfig { max_position_embeddings: 3072, ..base_config.clone() };
    let cache_size_3k = calculate_kv_cache_size(&tripled_config);

    assert_eq!(cache_size_3k, cache_size_1k * 3, "Tripling seq_len should triple cache size");
}

#[test]
fn test_cache_scales_linearly_with_num_layers() {
    // Test that cache size scales linearly with num_layers
    let base_config = ModelConfig {
        num_layers: 10,
        num_heads: 8,
        num_key_value_heads: 8,
        hidden_size: 512,
        max_position_embeddings: 1024,
        ..Default::default()
    };

    let cache_size_10l = calculate_kv_cache_size(&base_config);

    // Double the number of layers
    let doubled_config = ModelConfig { num_layers: 20, ..base_config.clone() };
    let cache_size_20l = calculate_kv_cache_size(&doubled_config);

    assert_eq!(cache_size_20l, cache_size_10l * 2, "Doubling num_layers should double cache size");

    // Triple the number of layers
    let tripled_config = ModelConfig { num_layers: 30, ..base_config.clone() };
    let cache_size_30l = calculate_kv_cache_size(&tripled_config);

    assert_eq!(cache_size_30l, cache_size_10l * 3, "Tripling num_layers should triple cache size");
}

#[test]
fn test_cache_scales_linearly_with_kv_heads() {
    // Test that cache size scales linearly with num_kv_heads
    let base_config = ModelConfig {
        num_layers: 10,
        num_heads: 16,
        num_key_value_heads: 4,
        hidden_size: 1024,
        max_position_embeddings: 1024,
        ..Default::default()
    };

    let cache_size_4kv = calculate_kv_cache_size(&base_config);

    // Double the number of KV heads
    let doubled_config = ModelConfig { num_key_value_heads: 8, ..base_config.clone() };
    let cache_size_8kv = calculate_kv_cache_size(&doubled_config);

    assert_eq!(
        cache_size_8kv,
        cache_size_4kv * 2,
        "Doubling num_kv_heads should double cache size"
    );

    // Triple the number of KV heads
    let tripled_config = ModelConfig { num_key_value_heads: 12, ..base_config.clone() };
    let cache_size_12kv = calculate_kv_cache_size(&tripled_config);

    assert_eq!(
        cache_size_12kv,
        cache_size_4kv * 3,
        "Tripling num_kv_heads should triple cache size"
    );
}

// ============================================================================
// GQA (Grouped Query Attention) Tests
// ============================================================================

#[test]
fn test_gqa_effect_on_cache_size() {
    // Test that fewer KV heads = smaller cache (GQA effect)
    // Same model, but with different KV head counts
    let base_config = ModelConfig {
        num_layers: 32,
        num_heads: 32,
        hidden_size: 4096,
        max_position_embeddings: 4096,
        ..Default::default()
    };

    // Full Attention (all KV heads = all attention heads)
    let full_attention = ModelConfig { num_key_value_heads: 32, ..base_config.clone() };
    let cache_full = calculate_kv_cache_size(&full_attention);

    // GQA with 8 KV heads (4:1 compression)
    let gqa_8kv = ModelConfig { num_key_value_heads: 8, ..base_config.clone() };
    let cache_gqa_8 = calculate_kv_cache_size(&gqa_8kv);

    // Cache should be 4x smaller (32/8 = 4)
    assert_eq!(
        cache_gqa_8,
        cache_full / 4,
        "GQA with 8 KV heads should be 4x smaller than full attention"
    );

    // GQA with 4 KV heads (8:1 compression)
    let gqa_4kv = ModelConfig { num_key_value_heads: 4, ..base_config.clone() };
    let cache_gqa_4 = calculate_kv_cache_size(&gqa_4kv);

    // Cache should be 8x smaller (32/4 = 8)
    assert_eq!(
        cache_gqa_4,
        cache_full / 8,
        "GQA with 4 KV heads should be 8x smaller than full attention"
    );

    // Verify ordering
    assert!(
        cache_gqa_4 < cache_gqa_8 && cache_gqa_8 < cache_full,
        "Cache size should be: full > gqa_8 > gqa_4"
    );
}

#[test]
fn test_gqa_head_ratio_calculation() {
    // Test that the compression ratio is exactly num_heads / num_kv_heads
    let base_config = ModelConfig {
        num_layers: 10,
        num_heads: 32,
        hidden_size: 4096,
        max_position_embeddings: 2048,
        ..Default::default()
    };

    let compression_ratios = vec![
        (32, 1), // full attention, no compression
        (16, 2), // 2x compression
        (8, 4),  // 4x compression
        (4, 8),  // 8x compression
        (2, 16), // 16x compression
        (1, 32), // extreme compression (MQA)
    ];

    let cache_full = {
        let config = ModelConfig { num_key_value_heads: 32, ..base_config.clone() };
        calculate_kv_cache_size(&config)
    };

    for (num_kv_heads, expected_ratio) in compression_ratios {
        let config = ModelConfig { num_key_value_heads: num_kv_heads, ..base_config.clone() };
        let cache_size = calculate_kv_cache_size(&config);
        let actual_ratio = cache_full / cache_size;

        assert_eq!(
            actual_ratio, expected_ratio as usize,
            "For num_kv_heads={}, compression ratio should be {}, got {}",
            num_kv_heads, expected_ratio, actual_ratio
        );
    }
}

#[test]
fn test_default_num_key_value_heads_fallback() {
    // When num_key_value_heads is 0 (default), it should use num_heads
    let config = ModelConfig {
        num_layers: 10,
        num_heads: 16,
        num_key_value_heads: 0, // default/unset
        hidden_size: 1024,
        max_position_embeddings: 1024,
        ..Default::default()
    };

    let cache_with_default = calculate_kv_cache_size(&config);

    // Manually set num_key_value_heads to num_heads
    let config_explicit = ModelConfig {
        num_layers: 10,
        num_heads: 16,
        num_key_value_heads: 16,
        hidden_size: 1024,
        max_position_embeddings: 1024,
        ..Default::default()
    };

    let cache_explicit = calculate_kv_cache_size(&config_explicit);

    assert_eq!(
        cache_with_default, cache_explicit,
        "Default num_key_value_heads=0 should use num_heads"
    );
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_single_layer_model() {
    // Minimal single-layer model
    let config = ModelConfig {
        num_layers: 1,
        num_heads: 4,
        num_key_value_heads: 4,
        hidden_size: 256,
        max_position_embeddings: 512,
        ..Default::default()
    };

    let cache_size = calculate_kv_cache_size(&config);
    let expected = 2 * 1 * 4 * 64 * 512 * 4; // head_dim = 256/4 = 64

    assert_eq!(cache_size, expected);
}

#[test]
fn test_single_head_model() {
    // Model with single attention head (degenerate case)
    let config = ModelConfig {
        num_layers: 4,
        num_heads: 1,
        num_key_value_heads: 1,
        hidden_size: 128,
        max_position_embeddings: 256,
        ..Default::default()
    };

    let cache_size = calculate_kv_cache_size(&config);
    let expected = 2 * 4 * 1 * 128 * 256 * 4; // head_dim = 128/1 = 128

    assert_eq!(cache_size, expected);
}

#[test]
fn test_minimal_sequence_length() {
    // Test with minimal sequence length
    let config = ModelConfig {
        num_layers: 8,
        num_heads: 8,
        num_key_value_heads: 8,
        hidden_size: 512,
        max_position_embeddings: 1,
        ..Default::default()
    };

    let cache_size = calculate_kv_cache_size(&config);
    let expected = 2 * 8 * 8 * 64 * 1 * 4; // head_dim = 512/8 = 64

    assert_eq!(cache_size, expected);
}

#[test]
fn test_power_of_two_dimensions() {
    // Test with all power-of-two dimensions (common in practice)
    let config = ModelConfig {
        num_layers: 16,
        num_heads: 16,
        num_key_value_heads: 4,
        hidden_size: 2048,
        max_position_embeddings: 2048,
        ..Default::default()
    };

    let cache_size = calculate_kv_cache_size(&config);
    let head_dim = 2048 / 16; // 128
    let expected = 2 * 16 * 4 * 128 * 2048 * 4;

    assert_eq!(cache_size, expected);
    assert_eq!(head_dim, 128);
}

// ============================================================================
// Composite/Combination Tests
// ============================================================================

#[test]
fn test_simultaneous_scaling() {
    // Test that multiple dimensions scale together correctly
    // Double num_layers AND double seq_len -> 4x cache
    let base_config = ModelConfig {
        num_layers: 10,
        num_heads: 8,
        num_key_value_heads: 8,
        hidden_size: 512,
        max_position_embeddings: 512,
        ..Default::default()
    };

    let cache_base = calculate_kv_cache_size(&base_config);

    let doubled_config =
        ModelConfig { num_layers: 20, max_position_embeddings: 1024, ..base_config.clone() };

    let cache_doubled = calculate_kv_cache_size(&doubled_config);

    assert_eq!(cache_doubled, cache_base * 4, "Doubling both layers and seq_len should 4x cache");
}

#[test]
fn test_cache_size_bytes_precision() {
    // Verify the cache size calculation is exact (no rounding errors)
    let config = ModelConfig {
        num_layers: 12,
        num_heads: 12,
        num_key_value_heads: 12,
        hidden_size: 768,
        max_position_embeddings: 512,
        ..Default::default()
    };

    let cache_size = calculate_kv_cache_size(&config);
    let _head_dim = 768 / 12; // 64

    // Manual calculation
    let manual = 2 * 12 * 12 * 64 * 512 * 4;

    assert_eq!(cache_size, manual, "Cache size calculation must be exact");
}

#[test]
fn test_large_model_realistic_config() {
    // Test with a large, realistic model configuration
    // Based on typical LLM dimensions
    let config = ModelConfig {
        vocab_size: 128256, // Typical for newer models
        hidden_size: 8192,
        num_layers: 80,
        num_heads: 64,
        num_key_value_heads: 8,
        intermediate_size: 28672,
        max_position_embeddings: 32768,
        ..Default::default()
    };

    let cache_size = calculate_kv_cache_size(&config);
    let _head_dim = 8192 / 64; // 128

    // Expected calculation
    let expected = 2 * 80 * 8 * 128 * 32768 * 4;
    assert_eq!(cache_size, expected);

    // This should be on the order of tens of GB
    let cache_gb = bytes_to_gb(cache_size);
    assert!(
        cache_gb > 10.0 && cache_gb < 100.0,
        "Large model cache should be tens of GB, got {:.2}GB",
        cache_gb
    );
}

#[test]
fn test_zero_cache_edge_case() {
    // Verify behavior with minimal config (not practically useful but tests math)
    let config = ModelConfig {
        num_layers: 0, // Invalid in practice, but math should still work
        num_heads: 1,
        num_key_value_heads: 1,
        hidden_size: 64,
        max_position_embeddings: 1,
        ..Default::default()
    };

    let cache_size = calculate_kv_cache_size(&config);
    assert_eq!(cache_size, 0, "Cache with 0 layers should be 0 bytes");
}

// ============================================================================
// Formula Verification Tests
// ============================================================================

#[test]
fn test_formula_components() {
    // Verify each component of the formula independently
    let config = ModelConfig {
        num_layers: 24,
        num_heads: 32,
        num_key_value_heads: 8,
        hidden_size: 2048,
        max_position_embeddings: 8192,
        ..Default::default()
    };

    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.hidden_size / config.num_heads;
    let sizeof_f32 = 4;

    let cache_size = calculate_kv_cache_size(&config);

    // Verify formula: 2 * num_layers * num_kv_heads * head_dim * max_seq_len * sizeof_f32
    let expected = 2
        * config.num_layers
        * num_kv_heads
        * head_dim
        * config.max_position_embeddings
        * sizeof_f32;

    assert_eq!(cache_size, expected, "Cache size must exactly match formula");

    // Verify individual components
    assert_eq!(num_kv_heads, 8);
    assert_eq!(head_dim, 64);
    assert_eq!(sizeof_f32, 4);
}

#[test]
fn test_memory_estimation_correctness() {
    // Comprehensive verification of memory estimation across multiple dimensions
    let test_cases = vec![
        // (num_layers, num_heads, num_kv_heads, hidden_size, max_seq_len, description)
        (8, 8, 8, 512, 256, "tiny"),
        (16, 16, 16, 1024, 512, "small"),
        (24, 32, 8, 2048, 2048, "medium-gqa"),
        (32, 32, 32, 4096, 4096, "large-full"),
        (40, 40, 10, 5120, 8192, "xlarge-gqa"),
    ];

    for (num_layers, num_heads, num_kv_heads, hidden_size, max_seq_len, desc) in test_cases {
        let config = ModelConfig {
            num_layers,
            num_heads,
            num_key_value_heads: num_kv_heads,
            hidden_size,
            max_position_embeddings: max_seq_len,
            ..Default::default()
        };

        let head_dim = hidden_size / num_heads;
        let cache_size = calculate_kv_cache_size(&config);

        // Verify the formula
        assert_eq!(
            cache_size,
            2 * num_layers * num_kv_heads * head_dim * max_seq_len * 4,
            "Formula verification failed for config: {}",
            desc
        );

        // Ensure cache size is positive
        assert!(cache_size > 0, "Cache size must be positive for config: {}", desc);
    }
}
