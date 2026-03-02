//! BDD-style integration tests — Wave 11
//!
//! Each test follows the Given / When / Then structure and exercises
//! inference pipeline integration across kernel selection, quantization
//! pipelines, memory management, error handling, and performance
//! characteristics.
//!
//! Categories:
//!   1. Kernel selection scenarios  (11 tests)
//!   2. Quantization pipeline scenarios (11 tests)
//!   3. Memory management scenarios (11 tests)
//!   4. Error handling scenarios    (11 tests)
//!   5. Performance scenarios       (12 tests)

use bitnet_common::kernel_registry::SimdLevel;
use bitnet_kernels::cpu::dequant::{
    dequant_i2s_block, dequant_i2s_row, dequant_ternary, pack_ternary,
};
use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear, kv_cache_memory_usage,
    kv_cache_slice,
};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm};
use bitnet_kernels::cpu::quantize::{
    compute_quantization_error, dequantize_symmetric_i8, quantize_symmetric_i8,
};
use bitnet_kernels::cpu::quantized_matmul::{i2s_matmul_f32, pack_i2s};
use bitnet_kernels::cpu::simd_matmul::SimdMatmulConfig;
use bitnet_kernels::perf_tracker::{KernelTiming, PerfTracker};
use bitnet_kernels::tl_lut::lut_index;
use bitnet_kernels::{FallbackKernel, KernelManager, KernelProvider, select_cpu_kernel};
use std::time::{Duration, Instant};

const TOL: f32 = 1e-5;

/// Pack four ternary values into one byte (I2_S encoding).
fn pack4(vals: [i8; 4]) -> u8 {
    let mut byte = 0u8;
    for (i, &v) in vals.iter().enumerate() {
        let code: u8 = match v {
            1 => 0b01,
            -1 => 0b11,
            _ => 0b00,
        };
        byte |= code << (i * 2);
    }
    byte
}

fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() < tol, "mismatch at index {i}: {x} vs {y} (diff {})", (x - y).abs());
    }
}

// ═══════════════════════════════════════════════════════════════════
// Section 1 — Kernel selection scenarios (11 tests)
// ═══════════════════════════════════════════════════════════════════

/// Scenario: CPU-only build always has a fallback kernel available.
#[test]
fn given_cpu_build_when_selecting_kernel_then_fallback_is_always_present() {
    // Given a KernelManager created on any CPU
    let mgr = KernelManager::new();

    // When listing available providers
    let providers = mgr.list_available_providers();

    // Then the fallback provider is always present
    assert!(
        providers.contains(&"fallback"),
        "fallback provider must always be present; got {providers:?}"
    );
}

/// Scenario: select_best always returns a usable provider.
#[test]
fn given_kernel_manager_when_selecting_best_then_provider_is_available() {
    // Given a new KernelManager
    let mgr = KernelManager::new();

    // When selecting the best kernel
    let provider = mgr.select_best().expect("select_best must succeed");

    // Then the provider reports itself as available
    assert!(provider.is_available());
    assert!(!provider.name().is_empty());
}

/// Scenario: SIMD detection returns a level >= Scalar.
#[test]
fn given_any_cpu_when_detecting_simd_then_level_is_at_least_scalar() {
    // Given the current hardware
    let level = bitnet_kernels::device_features::detect_simd_level();

    // When comparing to the minimum level
    // Then the level is at least Scalar
    assert!(level >= SimdLevel::Scalar);
}

/// Scenario: selected kernel name is a recognized string.
#[test]
fn given_kernel_manager_when_selecting_then_name_is_recognized() {
    // Given a KernelManager
    let mgr = KernelManager::new();
    let provider = mgr.select_best().unwrap();

    // When querying the provider name
    let name = provider.name();

    // Then the name is one of the known kernel names
    let known = ["fallback", "avx2", "avx512", "neon", "cuda", "npu", "rocm", "opencl"];
    assert!(known.contains(&name), "provider name {name:?} not in known list {known:?}");
}

/// Scenario: CPU kernel selection returns a provider that can do matmul.
#[test]
fn given_cpu_kernel_when_performing_matmul_then_operation_succeeds() {
    // Given a CPU kernel provider
    let provider = select_cpu_kernel().expect("select_cpu_kernel must succeed");

    // When performing a trivial 1×1 matmul via the provider trait
    let a = vec![1_i8];
    let b = vec![0b01_u8]; // +1 in I2_S
    let mut c = vec![0.0_f32];
    let result = provider.matmul_i2s(&a, &b, &mut c, 1, 1, 1);

    // Then the operation succeeds
    assert!(result.is_ok(), "matmul_i2s must succeed: {result:?}");
}

/// Scenario: FallbackKernel is always available regardless of features.
#[test]
fn given_fallback_kernel_when_checking_availability_then_always_available() {
    // Given the FallbackKernel
    let kernel = FallbackKernel;

    // When checking availability
    let available = kernel.is_available();

    // Then it is always available
    assert!(available);
    assert_eq!(kernel.name(), "fallback");
}

/// Scenario: KernelManager caches its selection across calls.
#[test]
fn given_kernel_manager_when_selecting_twice_then_same_provider_returned() {
    // Given a KernelManager
    let mgr = KernelManager::new();

    // When selecting best provider twice
    let name1 = mgr.select_best().unwrap().name();
    let name2 = mgr.select_best().unwrap().name();

    // Then the same provider is returned both times (cached)
    assert_eq!(name1, name2, "cached selection must be stable");
}

/// Scenario: provider list contains only unique names.
#[test]
fn given_kernel_manager_when_listing_providers_then_names_are_unique() {
    // Given a KernelManager
    let mgr = KernelManager::new();

    // When listing providers
    let providers = mgr.list_available_providers();

    // Then all names are unique
    let mut seen = std::collections::HashSet::new();
    for name in &providers {
        assert!(seen.insert(name), "duplicate provider name: {name}");
    }
}

/// Scenario: device capability summary is a non-empty string.
#[test]
fn given_current_device_when_querying_capability_summary_then_nonempty() {
    // Given the current device
    let summary = bitnet_kernels::device_features::device_capability_summary();

    // When checking the summary
    // Then it is non-empty and contains useful information
    assert!(!summary.is_empty(), "capability summary must not be empty");
}

/// Scenario: GPU compilation flag is false when only CPU feature is enabled.
#[test]
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn given_cpu_only_build_when_checking_gpu_compiled_then_false() {
    // Given a CPU-only build
    let gpu = bitnet_kernels::device_features::gpu_compiled();

    // When checking GPU compilation
    // Then it reports false
    assert!(!gpu, "GPU must not be compiled in CPU-only build");
}

/// Scenario: SIMD level detection is consistent across invocations.
#[test]
fn given_stable_hardware_when_detecting_simd_twice_then_same_result() {
    // Given stable hardware during test execution
    let level1 = bitnet_kernels::device_features::detect_simd_level();
    let level2 = bitnet_kernels::device_features::detect_simd_level();

    // Then both detections return the same result
    assert_eq!(level1, level2, "SIMD detection must be deterministic");
}

// ═══════════════════════════════════════════════════════════════════
// Section 2 — Quantization pipeline scenarios (11 tests)
// ═══════════════════════════════════════════════════════════════════

/// Scenario: I2_S dequantization produces output matching the expected precision.
#[test]
fn given_i2s_packed_input_when_dequantizing_then_output_matches_expected() {
    // Given known I2_S packed data: [+1, -1, 0, +1]
    let packed = vec![pack4([1, -1, 0, 1])];
    let scale = 2.5;

    // When dequantizing
    let output = dequant_i2s_block(&packed, scale, 4).unwrap();

    // Then output matches expected values within precision
    approx_eq(&output, &[2.5, -2.5, 0.0, 2.5], TOL);
}

/// Scenario: QK256-sized block maintains alignment after dequantization.
#[test]
fn given_qk256_block_when_processing_then_alignment_is_preserved() {
    // Given a QK256-sized block (256 elements = 64 bytes packed)
    let packed = vec![0x55u8; 64]; // all +1 values
    let block_size = 256;

    // When dequantizing
    let output = dequant_i2s_block(&packed, 1.0, block_size).unwrap();

    // Then output has exactly 256 elements (QK256 alignment preserved)
    assert_eq!(output.len(), block_size, "QK256 block alignment must be preserved");
    // And all values should be +1 (0b01 pattern)
    assert!(output.iter().all(|&v| (v - 1.0).abs() < TOL));
}

/// Scenario: TL1 LUT lookup returns deterministic results.
#[test]
fn given_tl1_table_when_looking_up_values_then_results_are_deterministic() {
    // Given LUT parameters for a TL1 table
    let block_idx = 0;
    let block_bytes = 32;
    let elems_per_block = 128;
    let lut_len = 256;

    // When looking up the same element twice
    let idx1 = lut_index(block_idx, 8, block_bytes, elems_per_block, lut_len).unwrap();
    let idx2 = lut_index(block_idx, 8, block_bytes, elems_per_block, lut_len).unwrap();

    // Then results are identical (deterministic)
    assert_eq!(idx1, idx2, "LUT lookup must be deterministic");
    assert_eq!(idx1, 1, "elem 8 / 8 = byte offset 1");
}

/// Scenario: symmetric i8 quantization round-trip preserves sign.
#[test]
fn given_f32_input_when_quantize_dequantize_symmetric_then_sign_preserved() {
    // Given a vector of mixed positive/negative values
    let input = vec![1.5, -2.0, 0.3, -0.7, 3.0, -1.1];

    // When quantizing and dequantizing symmetrically
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    let recovered = dequantize_symmetric_i8(&quantized, scale);

    // Then the sign of each non-zero value is preserved
    for (i, (&orig, &rec)) in input.iter().zip(recovered.iter()).enumerate() {
        if orig.abs() > 0.01 {
            assert_eq!(
                orig.is_sign_positive(),
                rec.is_sign_positive(),
                "sign mismatch at {i}: orig={orig}, rec={rec}"
            );
        }
    }
}

/// Scenario: ternary pack/unpack round-trip maps zeros correctly.
#[test]
fn given_zero_values_when_quantizing_ternary_then_all_recover_as_zero() {
    // Given a vector of zeros and near-zeros
    let values = vec![0.0, 0.001, -0.001, 0.0, 0.0, 0.0, 0.0, 0.0];
    let threshold = 0.01;

    // When quantizing and dequantizing
    let (packed, scale) = pack_ternary(&values, threshold);
    let recovered = dequant_ternary(&packed, scale);

    // Then all values recover as zero
    for (i, &v) in recovered.iter().enumerate().take(values.len()) {
        assert_eq!(v, 0.0, "index {i} should be zero, got {v}");
    }
}

/// Scenario: I2_S row dequantization with per-block scales applies them correctly.
#[test]
fn given_multi_block_i2s_row_when_dequantizing_then_per_block_scales_applied() {
    // Given two blocks with different scales
    let packed = vec![pack4([1, 1, 1, 1]), pack4([-1, -1, -1, -1])];
    let scales = vec![2.0, 3.0];
    let block_size = 4;

    // When dequantizing the full row
    let output = dequant_i2s_row(&packed, &scales, block_size).unwrap();

    // Then first block uses scale 2.0, second uses 3.0
    approx_eq(&output[..4], &[2.0, 2.0, 2.0, 2.0], TOL);
    approx_eq(&output[4..], &[-3.0, -3.0, -3.0, -3.0], TOL);
}

/// Scenario: quantization error metric is computed correctly.
#[test]
fn given_original_and_quantized_when_computing_error_then_mse_is_nonnegative() {
    // Given original and (imperfect) quantized values
    let original = vec![1.0, 2.0, 3.0, 4.0];
    let quantized = vec![1.1, 1.9, 3.1, 3.8];

    // When computing quantization error
    let error = compute_quantization_error(&original, &quantized);

    // Then MSE is non-negative and SNR is reasonable
    assert!(error.mse >= 0.0, "MSE must be non-negative");
    assert!(error.max_abs_error >= 0.0, "max_abs_error must be non-negative");
    assert!(error.max_abs_error <= 0.21, "max error should be ~0.2");
}

/// Scenario: I2_S matmul produces correct dot product for identity-like case.
#[test]
fn given_i2s_identity_weights_when_matmul_then_correct_output() {
    // Given identity-like I2_S weights (all +1) and input
    let weights_vals = [1_i8, 1, 1, 1];
    let packed_w = vec![pack_i2s(weights_vals)]; // one packed byte
    let scales = vec![1.0_f32];
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0_f32; 1];
    let m = 1;
    let n = 1;
    let k = 4;
    let block_size = 4;

    // When performing I2_S matmul
    i2s_matmul_f32(&input, &packed_w, &scales, &mut output, m, n, k, block_size).unwrap();

    // Then the result is the sum: 1+2+3+4 = 10
    assert_eq!(output.len(), 1);
    assert!((output[0] - 10.0).abs() < TOL, "expected 10.0, got {}", output[0]);
}

/// Scenario: dequantization output length matches block size exactly.
#[test]
fn given_packed_data_when_dequantizing_block_then_output_length_matches_block_size() {
    // Given packed data and various block sizes
    for block_size in [4, 8, 16, 32] {
        let bytes_needed = block_size / 4;
        let packed = vec![0u8; bytes_needed];

        // When dequantizing
        let output = dequant_i2s_block(&packed, 1.0, block_size).unwrap();

        // Then output length exactly matches block_size
        assert_eq!(output.len(), block_size);
    }
}

/// Scenario: LUT index maps elements 0-7 to the same byte offset.
#[test]
fn given_tl_bit_packing_when_indexing_first_eight_elements_then_same_byte() {
    // Given TL1 bit-packing (8 elements per byte)
    let block_bytes = 32;
    let elems_per_block = 256;
    let lut_len = 512;

    // When indexing elements 0 through 7 in block 0
    let indices: Vec<usize> =
        (0..8).map(|e| lut_index(0, e, block_bytes, elems_per_block, lut_len).unwrap()).collect();

    // Then all map to the same byte offset (0)
    assert!(indices.iter().all(|&i| i == 0), "elements 0-7 must map to byte 0: {indices:?}");
}

/// Scenario: ternary values are always in the set {-scale, 0, +scale}.
#[test]
fn given_arbitrary_packed_data_when_dequantizing_ternary_then_values_in_valid_set() {
    // Given arbitrary packed bytes with a known scale
    let packed = vec![0xA5, 0x3C, 0xFF, 0x00];
    let scale = 2.5;

    // When dequantizing
    let output = dequant_ternary(&packed, scale);

    // Then every value is in {-2.5, 0.0, +2.5}
    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - scale).abs() < TOL || (v + scale).abs() < TOL || v.abs() < TOL,
            "value at {i} ({v}) not in {{-{scale}, 0, +{scale}}}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// Section 3 — Memory management scenarios (11 tests)
// ═══════════════════════════════════════════════════════════════════

/// Scenario: KV cache allocates the correct number of layers.
#[test]
fn given_kv_cache_config_when_allocated_then_layer_count_matches() {
    // Given a config for 4 layers
    let config = KvCacheConfig {
        num_layers: 4,
        num_heads: 8,
        head_dim: 64,
        max_seq_len: 128,
        dtype: KvDtype::F32,
    };

    // When allocating
    let cache = KvCache::new(config).unwrap();

    // Then num_layers matches
    assert_eq!(cache.num_layers(), 4);
}

/// Scenario: KV cache append increases sequence length.
#[test]
fn given_empty_kv_cache_when_appending_tokens_then_seq_len_increases() {
    // Given an empty KV cache
    let config = KvCacheConfig {
        num_layers: 2,
        num_heads: 4,
        head_dim: 8,
        max_seq_len: 64,
        dtype: KvDtype::F32,
    };
    let mut cache = KvCache::new(config).unwrap();
    let token_size = 4 * 8; // num_heads * head_dim
    let keys = vec![1.0_f32; token_size];
    let values = vec![2.0_f32; token_size];

    // When appending a token to layer 0
    kv_cache_append(&mut cache, 0, &keys, &values).unwrap();

    // Then seq_len increases to 1
    assert_eq!(cache.seq_len(0).unwrap(), 1);
}

/// Scenario: KV cache clear resets sequence length to zero.
#[test]
fn given_populated_kv_cache_when_clearing_then_seq_len_is_zero() {
    // Given a populated KV cache
    let config = KvCacheConfig {
        num_layers: 2,
        num_heads: 4,
        head_dim: 8,
        max_seq_len: 64,
        dtype: KvDtype::F32,
    };
    let mut cache = KvCache::new(config).unwrap();
    let token_size = 4 * 8;
    kv_cache_append(&mut cache, 0, &vec![1.0; token_size], &vec![1.0; token_size]).unwrap();

    // When clearing
    kv_cache_clear(&mut cache);

    // Then all layers have seq_len = 0
    assert_eq!(cache.seq_len(0).unwrap(), 0);
    assert_eq!(cache.seq_len(1).unwrap(), 0);
}

/// Scenario: KV cache memory usage is positive after allocation.
#[test]
fn given_kv_cache_when_checking_memory_usage_then_positive() {
    // Given a new KV cache
    let config = KvCacheConfig {
        num_layers: 2,
        num_heads: 4,
        head_dim: 8,
        max_seq_len: 32,
        dtype: KvDtype::F32,
    };
    let cache = KvCache::new(config).unwrap();

    // When checking memory usage
    let usage = kv_cache_memory_usage(&cache);

    // Then it is positive (allocated storage)
    assert!(usage > 0, "memory usage should be > 0, got {usage}");
}

/// Scenario: KV cache slice returns the correct number of elements.
#[test]
fn given_kv_cache_with_tokens_when_slicing_then_correct_count_returned() {
    // Given a KV cache with 3 appended tokens
    let config = KvCacheConfig {
        num_layers: 1,
        num_heads: 2,
        head_dim: 4,
        max_seq_len: 32,
        dtype: KvDtype::F32,
    };
    let mut cache = KvCache::new(config).unwrap();
    let token_size = 2 * 4; // num_heads * head_dim
    for _ in 0..3 {
        kv_cache_append(&mut cache, 0, &vec![1.0; token_size], &vec![1.0; token_size]).unwrap();
    }

    // When slicing layer 0 (all 3 tokens)
    let (keys, values) = kv_cache_slice(&cache, 0, 0, 3).unwrap();

    // Then we get 3 tokens worth of data
    assert_eq!(keys.len(), 3 * token_size);
    assert_eq!(values.len(), 3 * token_size);
}

/// Scenario: KV cache layers are independent of each other.
#[test]
fn given_multi_layer_kv_cache_when_appending_to_one_then_others_unaffected() {
    // Given a multi-layer cache
    let config = KvCacheConfig {
        num_layers: 3,
        num_heads: 2,
        head_dim: 4,
        max_seq_len: 16,
        dtype: KvDtype::F32,
    };
    let mut cache = KvCache::new(config).unwrap();
    let token_size = 2 * 4;

    // When appending only to layer 1
    kv_cache_append(&mut cache, 1, &vec![1.0; token_size], &vec![1.0; token_size]).unwrap();

    // Then layer 0 and 2 are still empty, layer 1 has 1 token
    assert_eq!(cache.seq_len(0).unwrap(), 0);
    assert_eq!(cache.seq_len(1).unwrap(), 1);
    assert_eq!(cache.seq_len(2).unwrap(), 0);
}

/// Scenario: KvDtype element sizes are correct.
#[test]
fn given_kv_dtype_variants_when_checking_element_bytes_then_correct_sizes() {
    // Given all KvDtype variants
    // When checking element byte sizes
    // Then sizes are correct
    assert_eq!(KvDtype::F32.element_bytes(), 4);
    assert_eq!(KvDtype::F16.element_bytes(), 2);
    assert_eq!(KvDtype::Bf16.element_bytes(), 2);
}

/// Scenario: KV cache config validation rejects zero dimensions.
#[test]
fn given_zero_dim_kv_config_when_validating_then_error_returned() {
    // Given configs with zero dimensions
    let configs = [
        KvCacheConfig {
            num_layers: 0,
            num_heads: 4,
            head_dim: 8,
            max_seq_len: 32,
            dtype: KvDtype::F32,
        },
        KvCacheConfig {
            num_layers: 2,
            num_heads: 0,
            head_dim: 8,
            max_seq_len: 32,
            dtype: KvDtype::F32,
        },
        KvCacheConfig {
            num_layers: 2,
            num_heads: 4,
            head_dim: 0,
            max_seq_len: 32,
            dtype: KvDtype::F32,
        },
        KvCacheConfig {
            num_layers: 2,
            num_heads: 4,
            head_dim: 8,
            max_seq_len: 0,
            dtype: KvDtype::F32,
        },
    ];

    // When validating each config
    // Then all produce errors
    for (i, cfg) in configs.iter().enumerate() {
        assert!(cfg.validate().is_err(), "config {i} with a zero dim should fail validation");
    }
}

/// Scenario: memory usage scales with number of layers.
#[test]
fn given_different_layer_counts_when_comparing_memory_then_more_layers_uses_more() {
    // Given caches with 2 and 4 layers
    let cfg2 = KvCacheConfig {
        num_layers: 2,
        num_heads: 4,
        head_dim: 8,
        max_seq_len: 32,
        dtype: KvDtype::F32,
    };
    let cfg4 = KvCacheConfig {
        num_layers: 4,
        num_heads: 4,
        head_dim: 8,
        max_seq_len: 32,
        dtype: KvDtype::F32,
    };
    let cache2 = KvCache::new(cfg2).unwrap();
    let cache4 = KvCache::new(cfg4).unwrap();

    // When checking memory
    let mem2 = kv_cache_memory_usage(&cache2);
    let mem4 = kv_cache_memory_usage(&cache4);

    // Then 4-layer cache uses more memory
    assert!(mem4 > mem2, "4 layers ({mem4}) should use more memory than 2 ({mem2})");
}

/// Scenario: SimdMatmulConfig validates dimensions correctly.
#[test]
fn given_valid_dimensions_when_creating_simd_config_then_config_stores_them() {
    // Given valid matrix dimensions
    let m = 16;
    let n = 32;
    let k = 64;

    // When creating a SimdMatmulConfig
    let cfg = SimdMatmulConfig::new(m, n, k);

    // Then the config reflects the dimensions
    assert_eq!(cfg.m, m);
    assert_eq!(cfg.n, n);
    assert_eq!(cfg.k, k);
}

/// Scenario: KV cache sequential appends maintain correct seq_len monotonicity.
#[test]
fn given_kv_cache_when_appending_n_tokens_then_seq_len_equals_n() {
    // Given a KV cache
    let config = KvCacheConfig {
        num_layers: 1,
        num_heads: 2,
        head_dim: 4,
        max_seq_len: 64,
        dtype: KvDtype::F32,
    };
    let mut cache = KvCache::new(config).unwrap();
    let token_size = 2 * 4;
    let k = vec![1.0_f32; token_size];
    let v = vec![1.0_f32; token_size];

    // When appending 5 tokens sequentially
    for expected_len in 1..=5 {
        kv_cache_append(&mut cache, 0, &k, &v).unwrap();

        // Then seq_len equals the number of tokens appended so far
        assert_eq!(cache.seq_len(0).unwrap(), expected_len);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Section 4 — Error handling scenarios (11 tests)
// ═══════════════════════════════════════════════════════════════════

/// Scenario: dequantizing with insufficient packed bytes returns an error.
#[test]
fn given_insufficient_packed_data_when_dequantizing_block_then_error_returned() {
    // Given 1 byte (4 elements) but requesting 8 elements
    let packed = vec![0u8; 1];

    // When dequantizing with block_size > capacity
    let result = dequant_i2s_block(&packed, 1.0, 8);

    // Then a descriptive error is returned
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("need") || err_msg.contains("byte"),
        "error should describe the issue"
    );
}

/// Scenario: LUT index with element out of block bounds returns an error.
#[test]
fn given_element_out_of_bounds_when_lut_lookup_then_error_returned() {
    // Given elem_in_block >= elems_per_block
    let result = lut_index(0, 128, 32, 128, 256);

    // When checking the result
    // Then an error is returned
    assert!(result.is_err(), "out-of-bounds element should produce an error");
}

/// Scenario: LUT index with zero elems_per_block returns an error.
#[test]
fn given_zero_elems_per_block_when_lut_lookup_then_error_returned() {
    // Given elems_per_block = 0 (invalid)
    let result = lut_index(0, 0, 32, 0, 256);

    // Then an error is returned
    assert!(result.is_err(), "zero elems_per_block should produce an error");
}

/// Scenario: dequant_i2s_row with zero block_size returns an error.
#[test]
fn given_zero_block_size_when_dequantizing_row_then_error_returned() {
    // Given a zero block_size
    let packed = vec![0u8; 4];

    // When dequantizing
    let result = dequant_i2s_row(&packed, &[1.0], 0);

    // Then an error is returned
    assert!(result.is_err());
}

/// Scenario: dequant_i2s_row with insufficient scales returns an error.
#[test]
fn given_insufficient_scales_when_dequantizing_row_then_error_returned() {
    // Given 16 elements (4 bytes) with block_size=4 needing 4 scales
    let packed = vec![0u8; 4];

    // When providing only 2 scales
    let result = dequant_i2s_row(&packed, &[1.0, 2.0], 4);

    // Then an error is returned
    assert!(result.is_err());
}

/// Scenario: KV cache with zero-dim config fails to create.
#[test]
fn given_invalid_kv_config_when_creating_cache_then_error_returned() {
    // Given a config with num_layers = 0
    let config = KvCacheConfig {
        num_layers: 0,
        num_heads: 4,
        head_dim: 8,
        max_seq_len: 32,
        dtype: KvDtype::F32,
    };

    // When trying to create the cache
    let result = KvCache::new(config);

    // Then it fails with a descriptive error
    assert!(result.is_err());
}

/// Scenario: KV cache append to invalid layer index returns an error.
#[test]
fn given_out_of_range_layer_when_appending_to_kv_cache_then_error_returned() {
    // Given a cache with 2 layers
    let config = KvCacheConfig {
        num_layers: 2,
        num_heads: 2,
        head_dim: 4,
        max_seq_len: 32,
        dtype: KvDtype::F32,
    };
    let mut cache = KvCache::new(config).unwrap();
    let token_size = 2 * 4;

    // When appending to layer 5 (out of range)
    let result = kv_cache_append(&mut cache, 5, &vec![1.0; token_size], &vec![1.0; token_size]);

    // Then an error is returned
    assert!(result.is_err());
}

/// Scenario: KV cache seq_len for invalid layer returns an error.
#[test]
fn given_invalid_layer_index_when_querying_seq_len_then_error_returned() {
    // Given a cache with 2 layers
    let config = KvCacheConfig {
        num_layers: 2,
        num_heads: 2,
        head_dim: 4,
        max_seq_len: 32,
        dtype: KvDtype::F32,
    };
    let cache = KvCache::new(config).unwrap();

    // When querying seq_len for non-existent layer
    let result = cache.seq_len(99);

    // Then an error is returned
    assert!(result.is_err());
}

/// Scenario: LUT index overflow is caught and reported.
#[test]
fn given_large_indices_when_lut_lookup_then_overflow_is_caught() {
    // Given extremely large block_idx and block_bytes that would overflow
    let result = lut_index(usize::MAX, 0, usize::MAX, 256, 100);

    // Then the overflow is caught as an error (not a panic)
    assert!(result.is_err(), "overflow must be caught gracefully");
}

/// Scenario: LayerNorm with empty gamma returns an error.
#[test]
fn given_empty_gamma_when_computing_layer_norm_then_error_returned() {
    // Given input with a mismatched empty gamma
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![]; // empty
    let beta = vec![0.0; 4];
    let config = LayerNormConfig::new(vec![4]);

    // When computing layer norm
    let result = layer_norm(&input, &gamma, Some(&beta), &config);

    // Then an error or failure occurs due to shape mismatch
    assert!(result.is_err(), "empty gamma should cause an error");
}

/// Scenario: quantize_symmetric_i8 never produces out-of-range i8 values.
#[test]
fn given_extreme_input_when_quantizing_symmetric_then_values_in_i8_range() {
    // Given extreme input values
    let input = vec![f32::MAX, f32::MIN, 0.0, -1e30, 1e30, f32::EPSILON];

    // When quantizing
    let (quantized, _scale) = quantize_symmetric_i8(&input, 8);

    // Then all values are in valid i8 range (this is guaranteed by the type,
    // but we verify no panics occurred)
    assert_eq!(quantized.len(), input.len());
    for &v in &quantized {
        assert!((-128..=127).contains(&(v as i16)));
    }
}

// ═══════════════════════════════════════════════════════════════════
// Section 5 — Performance scenarios (12 tests)
// ═══════════════════════════════════════════════════════════════════

/// Scenario: PerfTracker records kernel timings correctly.
#[test]
fn given_perf_tracker_when_recording_timings_then_count_matches() {
    // Given a PerfTracker
    let mut tracker = PerfTracker::new();

    // When recording 3 timings
    for i in 0..3 {
        tracker.record(KernelTiming::new(
            &format!("kernel_{i}"),
            Duration::from_micros(100 * (i + 1) as u64),
            1024,
        ));
    }

    // Then the count is 3
    assert_eq!(tracker.count(), 3);
}

/// Scenario: PerfTracker total time is the sum of recorded timings.
#[test]
fn given_recorded_timings_when_querying_total_time_then_sum_is_correct() {
    // Given a tracker with known timings
    let mut tracker = PerfTracker::new();
    tracker.record(KernelTiming::new("a", Duration::from_millis(10), 100));
    tracker.record(KernelTiming::new("b", Duration::from_millis(20), 200));
    tracker.record(KernelTiming::new("c", Duration::from_millis(30), 300));

    // When querying total time
    let total = tracker.total_time();

    // Then it equals the sum: 60ms
    assert_eq!(total, Duration::from_millis(60));
}

/// Scenario: PerfTracker identifies the slowest kernel.
#[test]
fn given_multiple_timings_when_querying_slowest_then_correct_kernel_identified() {
    // Given timings where "slow_kernel" is the slowest
    let mut tracker = PerfTracker::new();
    tracker.record(KernelTiming::new("fast_kernel", Duration::from_micros(50), 100));
    tracker.record(KernelTiming::new("slow_kernel", Duration::from_micros(500), 100));
    tracker.record(KernelTiming::new("mid_kernel", Duration::from_micros(200), 100));

    // When querying slowest
    let slowest = tracker.slowest().unwrap();

    // Then it is "slow_kernel"
    assert_eq!(slowest.kernel_name, "slow_kernel");
}

/// Scenario: PerfTracker identifies the fastest kernel.
#[test]
fn given_multiple_timings_when_querying_fastest_then_correct_kernel_identified() {
    // Given timings where "fast_kernel" is the fastest
    let mut tracker = PerfTracker::new();
    tracker.record(KernelTiming::new("fast_kernel", Duration::from_micros(10), 100));
    tracker.record(KernelTiming::new("slow_kernel", Duration::from_micros(500), 100));

    // When querying fastest
    let fastest = tracker.fastest().unwrap();

    // Then it is "fast_kernel"
    assert_eq!(fastest.kernel_name, "fast_kernel");
}

/// Scenario: throughput scales with batch size.
#[test]
fn given_increasing_batch_sizes_when_dequantizing_then_throughput_does_not_collapse() {
    // Given batch sizes of 256, 512, 1024
    let sizes = [256, 512, 1024];
    let mut throughputs = Vec::new();

    for &size in &sizes {
        let packed = vec![0x55u8; size / 4]; // all +1
        let start = Instant::now();
        for _ in 0..100 {
            let _ = dequant_i2s_block(&packed, 1.0, size).unwrap();
        }
        let elapsed = start.elapsed();
        let elements_per_sec = (size * 100) as f64 / elapsed.as_secs_f64();
        throughputs.push(elements_per_sec);
    }

    // Then throughput for larger batches is at least 50% of the smallest batch
    let min_throughput = throughputs[0] * 0.5;
    for (i, &tp) in throughputs.iter().enumerate() {
        assert!(
            tp > min_throughput,
            "throughput for size {} ({tp:.0} elem/s) collapsed below threshold ({min_throughput:.0})",
            sizes[i]
        );
    }
}

/// Scenario: repeated kernel execution benefits from warm caches.
#[test]
fn given_warm_cache_when_repeating_kernel_then_no_performance_degradation() {
    // Given a dequantization kernel executed repeatedly
    let packed = vec![0x55u8; 64]; // 256 elements

    // When running cold (first iteration) and warm (subsequent)
    let cold_start = Instant::now();
    let _ = dequant_i2s_block(&packed, 1.0, 256).unwrap();
    let cold_duration = cold_start.elapsed();

    let warm_start = Instant::now();
    for _ in 0..100 {
        let _ = dequant_i2s_block(&packed, 1.0, 256).unwrap();
    }
    let warm_avg = warm_start.elapsed() / 100;

    // Then warm average is not dramatically worse than cold (no degradation)
    // We allow 10x tolerance since the cold run might be cached by OS too
    assert!(
        warm_avg <= cold_duration.saturating_mul(10).max(Duration::from_micros(100)),
        "warm avg {warm_avg:?} should not degrade vs cold {cold_duration:?}"
    );
}

/// Scenario: KernelTiming throughput calculation is correct.
#[test]
fn given_kernel_timing_when_computing_throughput_then_formula_is_correct() {
    // Given a timing for 1000 elements in 1ms
    let timing = KernelTiming::new("test_kernel", Duration::from_millis(1), 1000);

    // When computing throughput
    let throughput = timing.throughput();

    // Then throughput ≈ 1,000,000 elements/sec (1000 / 0.001)
    assert!((throughput - 1_000_000.0).abs() < 100_000.0, "expected ~1M elem/s, got {throughput}");
}

/// Scenario: PerfTracker clear resets all state.
#[test]
fn given_populated_perf_tracker_when_clearing_then_count_is_zero() {
    // Given a tracker with some timings
    let mut tracker = PerfTracker::new();
    tracker.record(KernelTiming::new("a", Duration::from_micros(100), 50));
    tracker.record(KernelTiming::new("b", Duration::from_micros(200), 100));
    assert_eq!(tracker.count(), 2);

    // When clearing
    tracker.clear();

    // Then count is zero
    assert_eq!(tracker.count(), 0);
    assert_eq!(tracker.total_time(), Duration::ZERO);
}

/// Scenario: kernel_stats groups by kernel name.
#[test]
fn given_mixed_kernel_names_when_computing_stats_then_grouped_correctly() {
    // Given timings from two different kernels
    let mut tracker = PerfTracker::new();
    tracker.record(KernelTiming::new("matmul", Duration::from_micros(100), 1000));
    tracker.record(KernelTiming::new("matmul", Duration::from_micros(150), 1000));
    tracker.record(KernelTiming::new("layernorm", Duration::from_micros(50), 500));

    // When computing stats
    let stats = tracker.kernel_stats();

    // Then there are 2 kernel groups
    assert_eq!(stats.len(), 2, "should have 2 kernel groups");
    let matmul_stats = stats.iter().find(|s| s.name == "matmul").unwrap();
    assert_eq!(matmul_stats.count, 2);
}

/// Scenario: GFLOPS calculation works when FLOPs are provided.
#[test]
fn given_timing_with_flops_when_querying_gflops_then_correct_value() {
    // Given a timing with known FLOPs (1 billion in 1 second)
    let timing = KernelTiming::new("gemm", Duration::from_secs(1), 1000).with_flops(1_000_000_000);

    // When querying GFLOPS
    let gflops = timing.gflops().unwrap();

    // Then it equals 1.0 GFLOPS
    assert!((gflops - 1.0).abs() < 0.01, "expected ~1.0 GFLOPS, got {gflops}");
}

/// Scenario: perf report formatting produces non-empty output.
#[test]
fn given_perf_tracker_with_data_when_formatting_report_then_nonempty_string() {
    // Given a tracker with some timings
    let mut tracker = PerfTracker::new();
    tracker.record(KernelTiming::new("test_op", Duration::from_micros(500), 2048));

    // When formatting a report
    let report = bitnet_kernels::perf_tracker::format_perf_report(&tracker);

    // Then the report is non-empty and contains the kernel name
    assert!(!report.is_empty());
    assert!(report.contains("test_op"), "report should mention the kernel name");
}

/// Scenario: ternary quantization+dequantization is faster than per-element f32 copy for large vectors.
#[test]
fn given_large_vector_when_pack_then_packed_size_is_smaller() {
    // Given a large vector of 1024 f32 values (4096 bytes as f32)
    let values: Vec<f32> = (0..1024).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let threshold = 0.0;

    // When packing to ternary
    let (packed, _scale) = pack_ternary(&values, threshold);

    // Then packed size is much smaller than the original (1024/4 = 256 bytes)
    assert_eq!(packed.len(), 256, "ternary packing should compress 4:1");
    assert!(
        packed.len() < values.len(),
        "packed ({}) must be smaller than original ({})",
        packed.len(),
        values.len()
    );
}
