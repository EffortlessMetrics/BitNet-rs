//! Parity testing utilities for cross-validation with C++ implementation
//!
//! This module provides deterministic inference for comparing outputs with
//! the C++ BitNet implementation. It bypasses all async operations and
//! provides simple, synchronous evaluation.

use anyhow::Result;
use bitnet_common::{Device, Tensor};
use bitnet_models::transformer::KVCache;
use bitnet_models::{BitNetModel, Model, load_gguf_full};
use candle_core::{DType, IndexOp};
use std::path::Path;

/// Perform a single forward pass and return logits for the last token (PRODUCTION)
///
/// This function supports all BitNet quantization formats including:
/// - BitNet I2_S (32-element blocks with inline F16 scales)
/// - GGML I2_S (QK256 - 256-element blocks, pure Rust kernel)
/// - Other standard quantization formats
///
/// # Arguments
/// * `model_path` - Path to the GGUF model file
/// * `tokens` - Token IDs
///
/// # Returns
/// * Logits vector for the last token position (vocab_size elements)
///
/// # Errors
/// * Fails if model file cannot be loaded or if unsupported format is detected
pub fn eval_logits_once(model_path: &str, tokens: &[i32]) -> Result<Vec<f32>> {
    // Load model tensors with Rust GGUF loader (fail-closed, no FFI routing)
    let (config, model) = match load_gguf_full(
        Path::new(model_path),
        Device::Cpu,
        bitnet_models::GGUFLoaderConfig::default(),
    ) {
        Ok(result) => {
            eprintln!(
                "DEBUG parity: load_gguf_full returned config: hidden={}, n_heads={}, n_kv_heads={}",
                result.config.model.hidden_size,
                result.config.model.num_heads,
                result.config.model.num_key_value_heads
            );

            // Convert i2s_qk256 map to raw_tensors map with key remapping
            // QK256 tensors are stored as raw bytes in I2SQk256NoScale, we need to convert them to Candle tensors
            // and remap GGUF keys (e.g., "blk.0.attn_q.weight") to model keys (e.g., "layers.0.attention.q_proj.weight")
            let mut raw_tensors_unmapped = std::collections::HashMap::new();
            for (key, qk256_tensor) in result.i2s_qk256.iter() {
                // Create a U8 tensor from the raw bytes with shape [rows, row_stride_bytes]
                let raw_bytes_tensor = candle_core::Tensor::from_raw_buffer(
                    &qk256_tensor.qs,
                    candle_core::DType::U8,
                    &[qk256_tensor.rows, qk256_tensor.row_stride_bytes],
                    &candle_core::Device::Cpu,
                )
                .map_err(|e| anyhow::anyhow!("Failed to create raw tensor for '{}': {}", key, e))?;

                // Store with .qk256_qs suffix (GGUF key format)
                let qk256_key = format!("{}.qk256_qs", key);
                raw_tensors_unmapped.insert(qk256_key, raw_bytes_tensor);
            }

            eprintln!(
                "DEBUG parity: Converted {} QK256 tensors to raw_tensors (pre-remap)",
                raw_tensors_unmapped.len()
            );

            // Remap keys from GGUF format to model format
            let raw_tensors = bitnet_models::weight_mapper::remap_gguf_weights_with_options(
                &raw_tensors_unmapped,
                false, // non-strict
            )?;

            eprintln!("DEBUG parity: Remapped raw_tensors keys ({} tensors)", raw_tensors.len());

            let model = BitNetModel::from_gguf(
                result.config.clone(),
                result.tensors,
                raw_tensors,
                Device::Cpu,
            )?;
            (result.config, model)
        }
        Err(e) => {
            // Propagate the error with context (fail-closed for ggml I2_S)
            anyhow::bail!("Failed to load GGUF model: {}", e);
        }
    };

    // Convert i32 tokens to u32
    let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();

    // Create KV cache for the model (batch size 1, CPU)
    let cache = KVCache::new(&config, 1, &candle_core::Device::Cpu)?;
    let mut any_cache: Box<dyn std::any::Any> = Box::new(cache);

    // Get embeddings for the tokens
    let embedded = model.embed(&tokens_u32)?;

    // Run forward pass through the model
    let output = model.forward(&embedded, any_cache.as_mut())?;

    // Get logits from the output
    let logits = model.logits(&output)?;

    // Extract logits for the last token position
    let logits = extract_last_token_logits(logits)?;

    Ok(logits)
}

/// Perform a single forward pass for PARITY VALIDATION ONLY (pure Rust with optional C++ comparison)
///
/// This function uses the pure-Rust path for all supported formats including GGML I2_S (QK256).
/// Since QK256 support is now complete in pure Rust, there is no automatic FFI routing.
/// For C++ parity validation, use the explicit parity harness in `crossval/`.
///
/// # Arguments
/// * `model_path` - Path to the GGUF model file
/// * `tokens` - Token IDs
///
/// # Returns
/// * Logits vector for the last token position (vocab_size elements)
pub fn eval_logits_once_for_parity(model_path: &str, tokens: &[i32]) -> Result<Vec<f32>> {
    // Use the pure-Rust path for all supported formats (including QK256)
    eval_logits_once(model_path, tokens)
}

/// Evaluate logits using C++ FFI (for ggml I2_S models - parity only)
///
/// DEPRECATED: Use `eval_logits_via_ffi_session` instead to prevent memory corruption.
/// This function creates a new model/context for each call, which can cause munmap_chunk()
/// crashes when called repeatedly in tests.
#[deprecated(since = "0.10.0", note = "Use eval_logits_via_ffi_session instead")]
#[allow(dead_code)]
#[cfg(feature = "ffi")]
fn eval_logits_via_ffi(model_path: &str, tokens: &[i32]) -> Result<Vec<f32>> {
    use bitnet_sys::{
        BitnetContext, BitnetModel, bitnet_eval_tokens, bitnet_prefill, cpp_vocab_size,
    };

    // Load model via C++ FFI
    let cpp_model = BitnetModel::from_file(model_path)
        .map_err(|e| anyhow::anyhow!("C++ FFI model load failed: {:?}", e))?;

    // Create context (4096 max tokens, batch size 1, no threads override)
    let cpp_ctx = BitnetContext::new(&cpp_model, 4096, 1, 0)
        .map_err(|e| anyhow::anyhow!("C++ FFI context creation failed: {:?}", e))?;

    // Get vocab size from C++
    let vocab_size = cpp_vocab_size(&cpp_ctx)
        .map_err(|e| anyhow::anyhow!("C++ FFI vocab_size failed: {:?}", e))?;

    // Prefill the context with all tokens
    bitnet_prefill(&cpp_ctx, tokens)
        .map_err(|e| anyhow::anyhow!("C++ FFI prefill failed: {:?}", e))?;

    // Evaluate to get logits
    let logits = bitnet_eval_tokens(&cpp_ctx, tokens, vocab_size)
        .map_err(|e| anyhow::anyhow!("C++ FFI eval failed: {:?}", e))?;

    // Paranoia: ensure non-zero last-step logits (catches KV/logits wiring issues)
    let sum_abs: f32 = logits.iter().map(|x| x.abs()).sum();
    anyhow::ensure!(
        sum_abs > 1e-6,
        "C++ last-step logits near zero (sum_abs={:.2e}); KV/logits wiring off or weights not loaded",
        sum_abs
    );

    // Note: Let Rust's Drop trait handle cleanup automatically
    // Manual drop() was causing "free(): invalid pointer" errors
    Ok(logits)
}

/// Evaluate logits using C++ FFI session (for ggml I2_S models - parity only)
///
/// This function uses a reusable FFI session to prevent repeated model/context
/// allocation that causes munmap_chunk() crashes. The session is shared globally
/// and thread-safe via Mutex.
#[allow(dead_code)]
#[cfg(feature = "ffi")]
fn eval_logits_via_ffi_session(model_path: &str, tokens: &[i32]) -> Result<Vec<f32>> {
    use crate::ffi_session::parity_cpp_session;

    // Get or initialize the global session
    let session_mutex = parity_cpp_session(model_path)?;

    // Lock the session and perform evaluation
    let session =
        session_mutex.lock().map_err(|e| anyhow::anyhow!("Failed to lock FFI session: {}", e))?;

    // Prefill the context with all tokens
    session.prefill(tokens)?;

    // Evaluate to get logits
    session.eval_last_logits(tokens)
}

/// Perform step-by-step generation with per-token logit comparison
///
/// This function processes tokens one at a time, returning logits at each step.
/// Useful for debugging divergences in multi-step generation.
///
/// # Arguments
/// * `model_path` - Path to the GGUF model file
/// * `tokens` - Initial token sequence
/// * `n_past` - Number of tokens already processed (for KV cache)
///
/// # Returns
/// * Logits for the last token in the sequence
pub fn eval_logits_incremental(
    model_path: &str,
    tokens: &[i32],
    _n_past: usize,
) -> Result<Vec<f32>> {
    // For now, just call the single-shot version
    // In a full implementation, this would maintain state across calls
    eval_logits_once(model_path, tokens)
}

/// Extract logits for the last token from the model output
fn extract_last_token_logits(logits: bitnet_common::ConcreteTensor) -> Result<Vec<f32>> {
    use bitnet_common::ConcreteTensor;

    match logits {
        ConcreteTensor::BitNet(tensor) => {
            // Get the underlying Candle tensor
            let candle_tensor = tensor.as_candle();

            // Shape should be [batch, seq_len, vocab_size]
            let dims = candle_tensor.dims();
            if dims.len() != 3 {
                anyhow::bail!("Expected 3D logits tensor, got {:?}", dims);
            }

            let seq_len = dims[1];

            // Extract last token: narrow to last position in sequence dimension
            let last_token_logits = candle_tensor
                .narrow(1, seq_len - 1, 1)?  // Get last position
                .squeeze(1)?                  // Remove seq dimension
                .i(0)?; // Get first (and only) batch

            // Convert to F32 if needed
            let last_token_logits = if last_token_logits.dtype() != DType::F32 {
                last_token_logits.to_dtype(DType::F32)?
            } else {
                last_token_logits.clone()
            };

            // Convert to Vec<f32>
            Ok(last_token_logits.to_vec1::<f32>()?)
        }
        ConcreteTensor::Mock(mock) => {
            // For mock tensors, return zeros
            let vocab_size = mock.shape()[2];
            Ok(vec![0.0f32; vocab_size])
        }
    }
}

/// Load model and return vocabulary size for validation
pub fn get_model_vocab_size(model_path: &str) -> Result<usize> {
    let result = load_gguf_full(
        Path::new(model_path),
        Device::Cpu,
        bitnet_models::GGUFLoaderConfig::default(),
    )?;
    Ok(result.config.model.vocab_size)
}

/// Load model and return configuration for validation
pub fn get_model_config(model_path: &str) -> Result<bitnet_common::BitNetConfig> {
    let result = load_gguf_full(
        Path::new(model_path),
        Device::Cpu,
        bitnet_models::GGUFLoaderConfig::default(),
    )?;
    Ok(result.config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_eval() {
        // Test with non-existent model file
        // After error propagation fix, this should properly fail
        let tokens = vec![1, 2, 3, 4];

        // This should fail since the file doesn't exist
        let result = eval_logits_once("nonexistent_test.gguf", &tokens);

        // Should fail with proper error message (fail-closed behavior)
        assert!(result.is_err(), "Non-existent model should fail to load");

        // Verify it's a proper file error, not a ggml I2_S error
        let err = result.unwrap_err();
        let err_str = format!("{:?}", err);
        assert!(
            err_str.contains("Failed to open GGUF file") || err_str.contains("No such file"),
            "Expected file not found error, got: {}",
            err_str
        );
    }
}
