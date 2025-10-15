// AC:1 - CPU Forward Pass with Real Inference Test Scaffolding
//
// This test file validates AC1 from Issue #462:
// - Single-layer forward pass with I2_S quantization
// - Multi-layer forward pass with KV cache updates
// - End-to-end logits validation (BOS → non-zero finite output)
// - Quantized linear path enforcement (strict mode)
//
// Test Plan Reference: docs/explanation/cpu-inference-test-plan.md
// Architecture Spec: docs/explanation/cpu-inference-architecture.md
// API Contracts: docs/explanation/cpu-inference-api-contracts.md

#![cfg(feature = "cpu")]

use anyhow::{Context, Result};
use std::path::PathBuf;

/// Test utilities for CPU forward pass validation
mod test_utils {
    use super::*;

    /// Get test model path via auto-discovery or BITNET_GGUF env var
    ///
    /// # Search Order
    /// 1. BITNET_GGUF environment variable
    /// 2. models/ directory (workspace root)
    /// 3. Return error if not found
    ///
    /// # Example
    /// ```bash
    /// export BITNET_GGUF=/path/to/model.gguf
    /// cargo test -p bitnet-inference test_ac1 --features cpu
    /// ```
    pub fn get_test_model_path() -> Result<PathBuf> {
        // Try BITNET_GGUF env var first
        if let Ok(path) = std::env::var("BITNET_GGUF") {
            let model_path = PathBuf::from(&path);
            if model_path.exists() {
                return Ok(model_path);
            }
            anyhow::bail!("BITNET_GGUF set to '{}' but file does not exist", path);
        }

        // Try models/ directory auto-discovery
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .ok_or_else(|| anyhow::anyhow!("Failed to find workspace root"))?;

        let models_dir = workspace_root.join("models");
        if !models_dir.exists() {
            anyhow::bail!(
                "No test model found. Set BITNET_GGUF env var or place model in models/ directory.\n\
                 Example: cargo run -p xtask -- download-model"
            );
        }

        // Find first .gguf file in models/
        let model_file = std::fs::read_dir(&models_dir)
            .context("Failed to read models/ directory")?
            .filter_map(|entry| entry.ok())
            .find(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    == Some("gguf")
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No .gguf files found in models/ directory.\n\
                     Run: cargo run -p xtask -- download-model"
                )
            })?;

        Ok(model_file.path())
    }

    /// Get test tokenizer path via auto-discovery or BITNET_TOKENIZER env var
    ///
    /// # Returns
    /// Optional tokenizer path (None if not found, which is acceptable for some tests)
    pub fn get_test_tokenizer_path() -> Option<PathBuf> {
        // Try BITNET_TOKENIZER env var first
        if let Ok(path) = std::env::var("BITNET_TOKENIZER") {
            let tokenizer_path = PathBuf::from(&path);
            if tokenizer_path.exists() {
                return Some(tokenizer_path);
            }
        }

        // Try co-located tokenizer.json in models/
        if let Ok(model_path) = get_test_model_path() {
            if let Some(parent) = model_path.parent() {
                let tokenizer_path = parent.join("tokenizer.json");
                if tokenizer_path.exists() {
                    return Some(tokenizer_path);
                }
            }
        }

        None
    }

    /// Enable deterministic inference mode
    ///
    /// Sets environment variables:
    /// - BITNET_DETERMINISTIC=1
    /// - BITNET_SEED=42
    /// - RAYON_NUM_THREADS=1
    ///
    /// # Safety
    /// Uses unsafe set_var which is safe in single-threaded test contexts
    pub fn enable_deterministic_mode() {
        unsafe {
            std::env::set_var("BITNET_DETERMINISTIC", "1");
            std::env::set_var("BITNET_SEED", "42");
            std::env::set_var("RAYON_NUM_THREADS", "1");
        }
    }

    /// Enable strict mode (no FP32 staging)
    ///
    /// # Safety
    /// Uses unsafe set_var which is safe in single-threaded test contexts
    pub fn enable_strict_mode() {
        unsafe {
            std::env::set_var("BITNET_STRICT_MODE", "1");
        }
    }
}

// ============================================================================
// AC:1 - Test 1.1: BOS Token Returns Non-Zero Finite Logits
// ============================================================================

/// AC:1 - T1.1: Single-layer forward pass with I2_S quantization
///
/// Test Plan: docs/explanation/cpu-inference-test-plan.md#test-11
/// Validates that forward pass on BOS token returns non-zero finite logits
///
/// # Expected Behavior
/// - Input: BOS token (ID = 1 or 2)
/// - Output: Logits tensor [1, vocab_size]
/// - All logits finite (no NaN/Inf)
/// - At least 50% of logits are non-zero
/// - KV cache updated at position 0
#[test]
#[cfg(feature = "cpu")]
fn test_ac1_cpu_forward_bos_nonzero_logits() -> Result<()> {
    // Skip if no test model available (graceful degradation for CI)
    let _model_path = match test_utils::get_test_model_path() {
        Ok(path) => path,
        Err(e) => {
            eprintln!("SKIP: {}", e);
            return Ok(());
        }
    };

    test_utils::enable_deterministic_mode();

    // TODO: Create CpuInferenceEngine from model
    // let engine = CpuInferenceEngine::new(&model_path)?;

    // TODO: Create BOS token tensor [1u32] with shape [1]
    // let bos_token_id = 1u32; // Standard BOS token
    // let bos_tensor = BitNetTensor::from_slice(&[bos_token_id], &[1], DType::U32, &Device::Cpu)?;

    // TODO: Forward pass at step 0
    // let logits = engine.forward_parallel(&bos_tensor, 0)?;

    // TODO: Validate output shape
    // assert_eq!(logits.shape()[0], 1, "Batch size should be 1");
    // assert!(logits.shape()[1] > 0, "Vocab size should be positive");

    // TODO: Extract logits as f32 vec
    // let logits_data = logits.to_vec1::<f32>()?;

    // TODO: Check all logits are finite
    // assert!(
    //     logits_data.iter().all(|&x| x.is_finite()),
    //     "All logits should be finite (no NaN/Inf)"
    // );

    // TODO: Check at least 50% non-zero
    // let non_zero_count = logits_data.iter().filter(|&&x| x.abs() > 1e-6).count();
    // let non_zero_ratio = non_zero_count as f32 / logits_data.len() as f32;
    // assert!(
    //     non_zero_ratio >= 0.5,
    //     "At least 50% of logits should be non-zero (got {}/{})",
    //     non_zero_count,
    //     logits_data.len()
    // );

    // TODO: Validate KV cache updated at position 0
    // let cache = engine.kv_cache.read()?;
    // assert_eq!(cache.len(), 1, "KV cache should have length 1 after BOS token");

    // TEMPORARY: Test fails with unimplemented error (TDD Red phase)
    anyhow::bail!(
        "UNIMPLEMENTED: CpuInferenceEngine::forward_parallel() not yet implemented.\n\
         Expected: Non-zero finite logits [1, vocab_size]\n\
         This test will pass once AC1 CPU forward pass is implemented."
    );
}

// ============================================================================
// AC:1 - Test 1.2: 16-Token Greedy Decode Without Panic
// ============================================================================

/// AC:1 - T1.2: Multi-layer forward pass with KV cache updates
///
/// Test Plan: docs/explanation/cpu-inference-test-plan.md#test-12
/// Validates 16-token autoregressive generation with greedy decoding
///
/// # Expected Behavior
/// - Greedy sampling (temperature = 0.0, argmax)
/// - Deterministic output (same seed → same tokens)
/// - All token IDs within vocab range
/// - KV cache length increments correctly: 1, 2, 3, ..., 17
#[test]
#[cfg(feature = "cpu")]
fn test_ac1_greedy_decode_16_tokens() -> Result<()> {
    let _model_path = match test_utils::get_test_model_path() {
        Ok(path) => path,
        Err(e) => {
            eprintln!("SKIP: {}", e);
            return Ok(());
        }
    };

    test_utils::enable_deterministic_mode();

    // TODO: Create engine
    // let engine = CpuInferenceEngine::new(&model_path)?;

    // TODO: Initialize with BOS token
    // let bos_token_id = 1u32;
    // let mut current_token = bos_token_id;

    // TODO: Generate 16 tokens with greedy sampling
    // let mut generated_tokens = Vec::new();
    // for step in 1..=16 {
    //     // Forward pass with previous token
    //     let input = BitNetTensor::from_slice(&[current_token], &[1], DType::U32, &Device::Cpu)?;
    //     let logits = engine.forward_parallel(&input, step)?;
    //
    //     // Greedy sampling: argmax
    //     let logits_data = logits.to_vec1::<f32>()?;
    //     let next_token_id = logits_data
    //         .iter()
    //         .enumerate()
    //         .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    //         .map(|(idx, _)| idx as u32)
    //         .unwrap();
    //
    //     // Validate token ID within vocab range
    //     assert!(
    //         next_token_id < logits_data.len() as u32,
    //         "Token ID {} exceeds vocab size {}",
    //         next_token_id,
    //         logits_data.len()
    //     );
    //
    //     generated_tokens.push(next_token_id);
    //     current_token = next_token_id;
    //
    //     // Validate KV cache length
    //     let cache = engine.kv_cache.read()?;
    //     assert_eq!(
    //         cache.len(),
    //         step + 1,
    //         "KV cache length should be {} after step {}",
    //         step + 1,
    //         step
    //     );
    // }

    // TODO: Validate 16 tokens generated
    // assert_eq!(generated_tokens.len(), 16, "Should generate exactly 16 tokens");

    // TODO: Validate determinism (run twice, compare output)
    // Note: This requires running the full decode loop twice with same seed

    anyhow::bail!(
        "UNIMPLEMENTED: 16-token greedy decode not yet implemented.\n\
         Expected: 16 valid tokens, deterministic output, KV cache management.\n\
         This test will pass once AC1 autoregressive generation is implemented."
    );
}

// ============================================================================
// AC:1 - Test 1.3: Quantized Linear Path Enforcement
// ============================================================================

/// AC:1 - T1.3: End-to-end logits validation with strict mode
///
/// Test Plan: docs/explanation/cpu-inference-test-plan.md#test-13
/// Validates that forward pass uses QuantizedLinear I2S/TL1/TL2 paths only
///
/// # Expected Behavior
/// - Strict mode enabled (BITNET_STRICT_MODE=1)
/// - No FP32 staging kernels allowed
/// - Receipt contains ≥1 quantized kernel (i2s_*, tl1_*, tl2_*)
/// - No fallback kernels (fp32_*, fallback_*, dequant*)
#[test]
#[cfg(feature = "cpu")]
fn test_ac1_quantized_linear_strict_mode() -> Result<()> {
    let _model_path = match test_utils::get_test_model_path() {
        Ok(path) => path,
        Err(e) => {
            eprintln!("SKIP: {}", e);
            return Ok(());
        }
    };

    test_utils::enable_deterministic_mode();
    test_utils::enable_strict_mode();

    // TODO: Create engine with strict mode config
    // let config = InferenceConfig {
    //     strict_mode: true,
    //     ..Default::default()
    // };
    // let engine = CpuInferenceEngine::new_with_config(&model_path, config)?;

    // TODO: Forward pass with BOS token
    // let bos_token = BitNetTensor::from_slice(&[1u32], &[1], DType::U32, &Device::Cpu)?;
    // let logits = engine.forward_parallel(&bos_token, 0)?;

    // TODO: Validate output
    // assert!(logits.to_vec1::<f32>()?.iter().all(|&x| x.is_finite()));

    // TODO: Get receipt and validate kernels
    // let receipt = engine.get_receipt()?;
    // assert_eq!(receipt.compute_path, "real", "compute_path should be 'real'");
    //
    // // Check for quantized kernels
    // let quantized_kernels: Vec<&String> = receipt
    //     .kernels
    //     .iter()
    //     .filter(|k| k.starts_with("i2s_") || k.starts_with("tl1_") || k.starts_with("tl2_"))
    //     .collect();
    //
    // assert!(
    //     !quantized_kernels.is_empty(),
    //     "Receipt should contain at least one quantized kernel (i2s_/tl1_/tl2_)"
    // );
    //
    // // Check no fallback kernels
    // let fallback_kernels: Vec<&String> = receipt
    //     .kernels
    //     .iter()
    //     .filter(|k| {
    //         k.starts_with("fp32_") || k.starts_with("fallback_") || k.contains("dequant")
    //     })
    //     .collect();
    //
    // assert!(
    //     fallback_kernels.is_empty(),
    //     "Receipt should not contain fallback kernels: {:?}",
    //     fallback_kernels
    // );

    anyhow::bail!(
        "UNIMPLEMENTED: Strict mode quantized path enforcement not yet implemented.\n\
         Expected: Receipt with quantized kernels only (no FP32 staging).\n\
         This test will pass once AC1 strict mode is enforced."
    );
}

// ============================================================================
// AC:1 - Test 1.4: KV Cache Population and Retrieval
// ============================================================================

/// AC:1 - T1.4: KV cache update and retrieval correctness
///
/// Test Plan: docs/explanation/cpu-inference-test-plan.md#test-14
/// Validates KV cache management during multi-step forward pass
///
/// # Expected Behavior
/// - Cache length increments: 1, 2, 3
/// - K,V shapes: [current_len, num_heads, head_dim]
/// - Cache contains non-zero values (not all zeros)
/// - Cache accessible for all layers
#[test]
#[cfg(feature = "cpu")]
fn test_ac1_kv_cache_update_retrieval() -> Result<()> {
    let _model_path = match test_utils::get_test_model_path() {
        Ok(path) => path,
        Err(e) => {
            eprintln!("SKIP: {}", e);
            return Ok(());
        }
    };

    test_utils::enable_deterministic_mode();

    // TODO: Create engine
    // let engine = CpuInferenceEngine::new(&model_path)?;

    // TODO: Forward BOS token (step 0)
    // let bos_token = BitNetTensor::from_slice(&[1u32], &[1], DType::U32, &Device::Cpu)?;
    // let _logits = engine.forward_parallel(&bos_token, 0)?;

    // TODO: Validate cache length = 1
    // let cache = engine.kv_cache.read()?;
    // assert_eq!(cache.len(), 1, "Cache length should be 1 after BOS");
    //
    // // Validate K,V shape for layer 0
    // let (k_cache, v_cache) = cache.get(0)?;
    // assert_eq!(k_cache.shape()[0], 1, "K cache seq length should be 1");
    // assert_eq!(v_cache.shape()[0], 1, "V cache seq length should be 1");

    // TODO: Forward token1 (step 1)
    // let token1 = BitNetTensor::from_slice(&[50u32], &[1], DType::U32, &Device::Cpu)?;
    // let _logits = engine.forward_parallel(&token1, 1)?;
    //
    // // Validate cache length = 2
    // let cache = engine.kv_cache.read()?;
    // assert_eq!(cache.len(), 2, "Cache length should be 2 after token1");

    // TODO: Forward token2 (step 2)
    // let token2 = BitNetTensor::from_slice(&[100u32], &[1], DType::U32, &Device::Cpu)?;
    // let _logits = engine.forward_parallel(&token2, 2)?;
    //
    // // Validate cache length = 3
    // let cache = engine.kv_cache.read()?;
    // assert_eq!(cache.len(), 3, "Cache length should be 3 after token2");
    //
    // // Validate K,V shapes
    // let (k_cache, v_cache) = cache.get(0)?;
    // assert_eq!(k_cache.shape()[0], 3, "K cache seq length should be 3");
    // assert_eq!(v_cache.shape()[0], 3, "V cache seq length should be 3");
    //
    // // Validate non-zero values
    // let k_data = k_cache.to_vec2::<f32>()?;
    // let has_nonzero = k_data.iter().any(|row| row.iter().any(|&x| x.abs() > 1e-6));
    // assert!(has_nonzero, "KV cache should contain non-zero values");

    anyhow::bail!(
        "UNIMPLEMENTED: KV cache update/retrieval not yet implemented.\n\
         Expected: Cache length increments correctly, K,V shapes valid, non-zero values.\n\
         This test will pass once AC1 KV cache management is implemented."
    );
}
