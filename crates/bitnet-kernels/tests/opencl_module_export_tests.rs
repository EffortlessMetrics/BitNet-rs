//! OpenCL module export tests (Wave 8 — unignored from opencl_e2e_tests.rs scaffolds)
//!
//! These tests validate that the 5 OpenCL modules (opencl_attention, opencl_ffn,
//! opencl_memory, opencl_quantized, opencl_transformer) are correctly exported
//! from bitnet-kernels and that their public APIs function as documented.
//!
//! Replaces the 6 scaffold stubs at opencl_e2e_tests.rs:968-1002 which were
//! ignored pending module exports that have since been completed.

// ---------------------------------------------------------------------------
// opencl_attention module
// ---------------------------------------------------------------------------

/// Replaces: e2e_attention_with_rope_and_kv_cache (opencl_e2e_tests.rs:968)
/// Tests AttentionConfig creation with standard and GQA configurations.
#[test]
fn test_attention_config_creation() {
    use bitnet_kernels::opencl_attention::AttentionConfig;

    let cfg = AttentionConfig::new(16, 64, 2048, true).expect("valid config");
    assert_eq!(cfg.heads_per_kv_group(), 1);
    assert!(!cfg.is_gqa());

    let gqa = AttentionConfig::new_gqa(32, 8, 64, 2048, true).expect("valid GQA config");
    assert_eq!(gqa.heads_per_kv_group(), 4);
    assert!(gqa.is_gqa());
}

/// Tests that causal mask blocks future positions and allows past positions.
#[test]
fn test_attention_mask_causal() {
    use bitnet_kernels::opencl_attention::AttentionMask;

    let mask = AttentionMask::causal(4, 4, 0);
    // Position i can attend to positions j <= i
    assert!(mask.allows(0, 0), "position 0 attends to itself");
    assert!(mask.allows(3, 0), "position 3 attends to position 0");
    assert!(!mask.allows(0, 1), "position 0 must not attend to future position 1");
    assert!(!mask.allows(1, 3), "position 1 must not attend to future position 3");

    // None mask allows all positions
    let none_mask = AttentionMask::none(4, 4);
    assert!(none_mask.allows(0, 3), "none mask allows all positions");
}

/// Tests raw attention score computation with a small 2x2 matrix.
#[test]
fn test_attention_scores_compute() {
    use bitnet_kernels::opencl_attention::AttentionScores;

    let q = vec![1.0_f32, 0.0, 0.0, 1.0]; // 2 queries, head_dim=2
    let k = vec![1.0_f32, 0.0, 0.0, 1.0]; // 2 keys, head_dim=2
    let scale = 1.0 / (2.0_f32).sqrt();
    let mut scores = AttentionScores::compute_raw(&q, &k, 2, 2, 2, scale);
    // q[0]·k[0]*scale = scale, q[0]·k[1]*scale = 0
    assert!(
        (scores.weights[0] - scale).abs() < 1e-5,
        "q0·k0 should be {scale}, got {}",
        scores.weights[0]
    );
    assert!(scores.weights[1].abs() < 1e-5, "q0·k1 should be ~0");

    // Apply mask and softmax without panicking
    let mask = bitnet_kernels::opencl_attention::AttentionMask::causal(2, 2, 0);
    scores.apply_mask(&mask);
    scores.softmax();
    // After softmax, rows should sum to ~1.0
    let row0_sum: f32 = scores.weights[..2].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-5, "softmax row 0 should sum to 1.0");
}

/// Tests KVCacheEntry append and retrieval.
#[test]
fn test_kv_cache_entry_append() {
    use bitnet_kernels::opencl_attention::KVCacheEntry;

    let mut cache = KVCacheEntry::new(4, 128);
    assert_eq!(cache.keys().len(), 0);
    assert_eq!(cache.values().len(), 0);

    let keys = vec![1.0_f32, 2.0, 3.0, 4.0];
    let vals = vec![5.0_f32, 6.0, 7.0, 8.0];
    cache.append(&keys, &vals).expect("append should succeed");

    assert_eq!(cache.keys(), &keys[..]);
    assert_eq!(cache.values(), &vals[..]);

    // Append a second entry
    let k2 = vec![9.0_f32, 10.0, 11.0, 12.0];
    let v2 = vec![13.0_f32, 14.0, 15.0, 16.0];
    cache.append(&k2, &v2).expect("second append should succeed");
    assert_eq!(cache.keys().len(), 8);
    assert_eq!(cache.values().len(), 8);

    // Clear resets
    cache.clear();
    assert_eq!(cache.keys().len(), 0);
}

// ---------------------------------------------------------------------------
// opencl_ffn module
// ---------------------------------------------------------------------------

/// Replaces: e2e_gated_ffn_with_silu_activation (opencl_e2e_tests.rs:974)
/// Tests gated FFN forward pass with known inputs producing expected outputs.
#[test]
fn test_gated_ffn_forward() {
    use bitnet_kernels::opencl_ffn::{ActivationType, gated_ffn_forward_ref};

    let hidden_size = 4;
    let intermediate_size = 8;
    let seq_len = 1;
    // Weights: gate, up, down (all zeros → output should be zero)
    let gate_w = vec![0.0_f32; hidden_size * intermediate_size];
    let up_w = vec![0.0_f32; hidden_size * intermediate_size];
    let down_w = vec![0.0_f32; intermediate_size * hidden_size];
    let input = vec![1.0_f32; hidden_size * seq_len];
    let mut output = vec![0.0_f32; hidden_size * seq_len];

    gated_ffn_forward_ref(
        &input,
        &gate_w,
        &up_w,
        &down_w,
        &mut output,
        seq_len,
        hidden_size,
        intermediate_size,
        ActivationType::SiLU,
    )
    .expect("gated FFN forward should succeed");

    assert_eq!(output.len(), hidden_size);
    for (i, &v) in output.iter().enumerate() {
        assert!(v.abs() < 1e-6, "zero weights → zero output at index {i}, got {v}");
    }
}

/// Tests that each activation type computes expected value for known input.
#[test]
fn test_ffn_activation_types() {
    use bitnet_kernels::opencl_ffn::ActivationType;

    // SiLU(1.0) = 1.0 * sigmoid(1.0) ≈ 0.7311
    let silu = ActivationType::SiLU.apply(1.0);
    assert!((silu - 0.7311).abs() < 0.01, "SiLU(1.0) ≈ 0.7311, got {silu}");

    // ReLU(1.0) = 1.0, ReLU(-1.0) = 0.0
    assert_eq!(ActivationType::ReLU.apply(1.0), 1.0);
    assert_eq!(ActivationType::ReLU.apply(-1.0), 0.0);

    // GELU(0.0) = 0.0 (by symmetry)
    let gelu_zero = ActivationType::GELU.apply(0.0);
    assert!(gelu_zero.abs() < 1e-6, "GELU(0) should be ~0, got {gelu_zero}");

    // GELU(1.0) ≈ 0.8412
    let gelu_one = ActivationType::GELU.apply(1.0);
    assert!((gelu_one - 0.8412).abs() < 0.01, "GELU(1.0) ≈ 0.8412, got {gelu_one}");
}

// ---------------------------------------------------------------------------
// opencl_memory module
// ---------------------------------------------------------------------------

/// Replaces: e2e_memory_transfer_tracking_full_pipeline (opencl_e2e_tests.rs:980)
/// Tests MemoryTransferTracker records transfers and computes stats.
#[test]
fn test_memory_transfer_tracking() {
    use bitnet_kernels::opencl_memory::{MemoryTransferTracker, TransferDirection};

    let mut tracker = MemoryTransferTracker::new();
    assert_eq!(tracker.transfer_count(), 0);
    assert_eq!(tracker.total_bytes_transferred(), 0);

    // Record a host-to-device transfer (1 MB in 1ms)
    tracker.record_transfer(TransferDirection::HostToDevice, 1_048_576, 1_000_000);
    assert_eq!(tracker.transfer_count(), 1);
    assert_eq!(tracker.total_bytes_transferred(), 1_048_576);

    // Record a device-to-host transfer (512 KB in 0.5ms)
    tracker.record_transfer(TransferDirection::DeviceToHost, 524_288, 500_000);
    assert_eq!(tracker.transfer_count(), 2);
    assert_eq!(tracker.total_bytes_transferred(), 1_048_576 + 524_288);

    // Stats should be computable
    let h2d_stats = tracker.host_to_device_stats();
    assert!(h2d_stats.avg_bandwidth_gbps() > 0.0, "H2D bandwidth should be positive");
}

/// Tests memory budget allocation and utilization tracking.
#[test]
fn test_memory_budget_allocation() {
    use bitnet_kernels::opencl_memory::MemoryBudget;

    let mut budget = MemoryBudget::new(1024);
    assert!(budget.can_allocate(512));
    assert_eq!(budget.free_bytes(), 1024);
    assert!((budget.utilization() - 0.0).abs() < 1e-6);

    budget.allocate(512).expect("should allocate 512 bytes");
    assert_eq!(budget.free_bytes(), 512);
    assert!((budget.utilization() - 0.5).abs() < 1e-6);

    assert!(!budget.can_allocate(1024), "should not allocate more than free");
    budget.free(256);
    assert_eq!(budget.free_bytes(), 768);
}

// ---------------------------------------------------------------------------
// opencl_quantized module
// ---------------------------------------------------------------------------

/// Replaces: e2e_quantized_matmul_i2s_round_trip (opencl_e2e_tests.rs:986)
/// Tests I2S pack → unpack round-trip preserves 2-bit signed values.
#[test]
fn test_i2s_pack_unpack_round_trip() {
    use bitnet_kernels::opencl_quantized::I2sPackedFormat;

    // 2-bit signed values: -1, 0, 1, -1, 0, 1, -1, 0
    let original: Vec<i8> = vec![-1, 0, 1, -1, 0, 1, -1, 0];
    let packed = I2sPackedFormat::pack(&original);

    // Packed length for 8 values at 2 bits each = 2 bytes
    assert_eq!(I2sPackedFormat::packed_len(8), 2);
    assert_eq!(packed.len(), 2);

    // Unpack and verify round-trip
    let unpacked = I2sPackedFormat::unpack(&packed, 8);
    assert_eq!(unpacked, original, "pack/unpack round-trip must preserve values");

    // Individual element access
    for (i, &expected) in original.iter().enumerate() {
        let actual = I2sPackedFormat::unpack_one(&packed, i);
        assert_eq!(actual, expected, "element {i} mismatch: expected {expected}, got {actual}");
    }
}

/// Tests I2S block layout properties for different block sizes.
#[test]
fn test_i2s_block_layouts() {
    use bitnet_kernels::opencl_quantized::I2sBlockLayout;

    let qk256 = I2sBlockLayout::Qk256;
    assert_eq!(qk256.block_size(), 256);
    assert_eq!(qk256.blocks_per_row(512), 2);
    assert_eq!(qk256.blocks_per_row(256), 1);

    let bitnet32 = I2sBlockLayout::BitNet32F16;
    assert_eq!(bitnet32.block_size(), 32);
    assert_eq!(bitnet32.blocks_per_row(256), 8);
}

// ---------------------------------------------------------------------------
// opencl_transformer module
// ---------------------------------------------------------------------------

/// Replaces: e2e_transformer_layer_full_forward (opencl_e2e_tests.rs:992)
/// Tests TransformerLayerConfig validation and standard configs.
#[test]
fn test_transformer_layer_config_validation() {
    use bitnet_kernels::opencl_transformer::bitnet_2b_config;

    let cfg = bitnet_2b_config();
    assert!(cfg.validate().is_ok(), "bitnet_2b_config should be valid");
    assert!(cfg.gqa_ratio() >= 1, "GQA ratio must be >= 1");
    assert!(cfg.kv_size() > 0, "KV size must be positive");
}

/// Tests LayerWeights creation with zeros and validation.
#[test]
fn test_transformer_layer_weights() {
    use bitnet_kernels::opencl_transformer::{LayerWeights, bitnet_2b_config};

    let config = bitnet_2b_config();
    let weights = LayerWeights::zeros(&config);
    assert!(weights.validate(&config).is_ok(), "zero weights should pass shape validation");

    let ones_weights = LayerWeights::ones(&config);
    assert!(ones_weights.validate(&config).is_ok(), "ones weights should pass shape validation");
}

// ---------------------------------------------------------------------------
// CPU fallback provider
// ---------------------------------------------------------------------------

/// Replaces: e2e_opencl_provider_fallback_to_cpu (opencl_e2e_tests.rs:998)
/// Tests that FallbackKernel (CPU) is always available as a provider.
#[test]
fn test_cpu_fallback_always_available() {
    use bitnet_kernels::{FallbackKernel, KernelProvider};

    let fallback = FallbackKernel;
    assert!(fallback.is_available(), "CPU fallback must always be available");
    assert!(!fallback.name().is_empty(), "fallback kernel must have a name");
}
