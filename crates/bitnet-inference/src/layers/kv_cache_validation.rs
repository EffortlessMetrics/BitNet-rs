//! K/V Cache Dimension Validation (AC3: Issue #469)
//!
//! This module provides dimension guardrails for K/V cache tensors to catch
//! shape mismatches early during inference. Validation uses debug assertions
//! in hot paths (zero overhead in release) and explicit checks in cold paths
//! (initialization).
//!
//! ## Design Principles
//!
//! - **Hot Path**: `debug_assert_eq!` for tensor rank checks (compiled out in --release)
//! - **Cold Path**: `anyhow::ensure!` for explicit initialization validation
//! - **Once-per-layer warnings**: `std::sync::Once` guards prevent log spam
//!
//! ## Usage
//!
//! ```no_run
//! use bitnet_inference::layers::kv_cache_validation::validate_kv_cache_dims;
//! use bitnet_common::BitNetTensor;
//!
//! # fn example(k_cache: &BitNetTensor) -> anyhow::Result<()> {
//! // Validate K-cache dimensions (called in KVCache::get or attention layer)
//! validate_kv_cache_dims(
//!     k_cache,
//!     0,      // layer_idx
//!     1,      // expected_batch (always 1 for now)
//!     16,     // expected_n_heads (from model config)
//!     2048,   // max_seq_len
//!     64,     // expected_head_dim
//! )?;
//! # Ok(())
//! # }
//! ```

use anyhow::{Result, ensure};
use bitnet_common::BitNetTensor;
use std::sync::Once;

/// Validate K/V cache tensor dimensions
///
/// This function validates that a K/V cache tensor has the expected shape:
/// `[batch=1, n_heads, seq_len, head_dim]`
///
/// ## Validation Rules
///
/// - Batch dimension must be 1 (batching not supported yet)
/// - Number of heads must match model config (supports GQA: num_kv_heads)
/// - Sequence length must not exceed max_seq_len
/// - Head dimension must match model config
///
/// ## Hot Path Optimization
///
/// - Uses `debug_assert_eq!` for tensor rank check (compiled out in release)
/// - Explicit dimension checks always run (minimal overhead)
/// - Once-per-layer warning guards prevent log spam
///
/// # Arguments
///
/// * `cache` - K or V cache tensor to validate
/// * `layer_idx` - Layer index for diagnostic messages
/// * `expected_batch` - Expected batch size (always 1 for now)
/// * `expected_n_heads` - Expected number of attention heads (num_kv_heads for GQA)
/// * `max_seq_len` - Maximum sequence length from model config
/// * `expected_head_dim` - Expected head dimension from model config
///
/// # Returns
///
/// `Ok(())` if dimensions are valid, otherwise error with diagnostic context.
///
/// # Examples
///
/// ```no_run
/// use bitnet_inference::layers::kv_cache_validation::validate_kv_cache_dims;
/// use bitnet_common::{BitNetTensor, Device};
///
/// # fn example() -> anyhow::Result<()> {
/// let k_cache = BitNetTensor::zeros(
///     &[1, 16, 128, 64],
///     candle_core::DType::F32,
///     &Device::Cpu,
/// )?;
///
/// // Valid cache passes validation
/// validate_kv_cache_dims(&k_cache, 0, 1, 16, 2048, 64)?;
/// # Ok(())
/// # }
/// ```
pub fn validate_kv_cache_dims(
    cache: &BitNetTensor,
    layer_idx: usize,
    expected_batch: usize,
    expected_n_heads: usize,
    max_seq_len: usize,
    expected_head_dim: usize,
) -> Result<()> {
    let shape = cache.as_candle().dims();

    // AC3: Hot-path debug assertion (zero overhead in release)
    debug_assert_eq!(
        shape.len(),
        4,
        "K/V cache must be 4D tensor [batch, n_heads, seq_len, head_dim], got rank {}",
        shape.len()
    );

    // AC3: Explicit dimension validation (always runs)
    ensure!(
        shape.len() == 4,
        "K/V cache must be 4D tensor [batch, n_heads, seq_len, head_dim], got rank {}",
        shape.len()
    );

    let (actual_batch, actual_n_heads, actual_seq_len, actual_head_dim) =
        (shape[0], shape[1], shape[2], shape[3]);

    // AC3: Batch dimension validation (batching not supported yet)
    if actual_batch != expected_batch {
        warn_once_per_layer(
            layer_idx,
            "batch_mismatch",
            &format!(
                "Layer {} K/V cache batch mismatch: expected {}, got {}. Batching not supported yet.",
                layer_idx, expected_batch, actual_batch
            ),
        );
        anyhow::bail!(
            "Layer {} K/V cache batch dimension mismatch: expected {}, got {}",
            layer_idx,
            expected_batch,
            actual_batch
        );
    }

    // AC3: Number of heads validation (supports GQA)
    if actual_n_heads != expected_n_heads {
        warn_once_per_layer(
            layer_idx,
            "heads_mismatch",
            &format!(
                "Layer {} K/V cache heads mismatch: expected {} (model config), got {}. This indicates a cache management bug.",
                layer_idx, expected_n_heads, actual_n_heads
            ),
        );
        anyhow::bail!(
            "Layer {} K/V cache heads dimension mismatch: expected {}, got {}",
            layer_idx,
            expected_n_heads,
            actual_n_heads
        );
    }

    // AC3: Sequence length validation
    if actual_seq_len > max_seq_len {
        warn_once_per_layer(
            layer_idx,
            "seq_overflow",
            &format!(
                "Layer {} K/V cache sequence length exceeds max: seq_len={} > max_seq_len={}. This indicates context window overflow.",
                layer_idx, actual_seq_len, max_seq_len
            ),
        );
        anyhow::bail!(
            "Layer {} K/V cache sequence length exceeds max: {} > {}",
            layer_idx,
            actual_seq_len,
            max_seq_len
        );
    }

    // AC3: Head dimension validation
    if actual_head_dim != expected_head_dim {
        warn_once_per_layer(
            layer_idx,
            "head_dim_mismatch",
            &format!(
                "Layer {} K/V cache head dimension mismatch: expected {} (model config), got {}. This indicates a cache management bug.",
                layer_idx, expected_head_dim, actual_head_dim
            ),
        );
        anyhow::bail!(
            "Layer {} K/V cache head dimension mismatch: expected {}, got {}",
            layer_idx,
            expected_head_dim,
            actual_head_dim
        );
    }

    Ok(())
}

/// Log warning once per layer using std::sync::Once guards
///
/// This function ensures warnings are logged at most once per layer,
/// preventing log spam during inference loops.
///
/// # Implementation
///
/// Uses a static array of `Once` guards indexed by (layer_idx, warning_kind).
/// The warning_kind is hashed to a small index space to limit memory usage.
fn warn_once_per_layer(layer_idx: usize, warning_kind: &str, message: &str) {
    // AC3: Once-per-layer warning guards (prevent log spam)
    // Use a static array of Once guards for common layer counts (up to 48 layers)
    // Hash warning_kind to a small index space (4 warning types: batch, heads, seq, head_dim)
    static WARNINGS: [Once; 192] = [const { Once::new() }; 192]; // 48 layers * 4 warning types

    let warning_idx = match warning_kind {
        "batch_mismatch" => 0,
        "heads_mismatch" => 1,
        "seq_overflow" => 2,
        "head_dim_mismatch" => 3,
        _ => return, // Unknown warning kind, skip
    };

    let idx = (layer_idx % 48) * 4 + warning_idx;
    if idx < WARNINGS.len() {
        WARNINGS[idx].call_once(|| {
            tracing::warn!("{}", message);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_common::Device;
    use candle_core::DType;

    #[test]
    fn test_valid_cache_dimensions() -> Result<()> {
        let valid_cache = BitNetTensor::zeros(&[1, 16, 128, 64], DType::F32, &Device::Cpu)?;

        let result = validate_kv_cache_dims(&valid_cache, 0, 1, 16, 2048, 64);

        assert!(result.is_ok(), "Valid cache shape should pass validation");
        Ok(())
    }

    #[test]
    fn test_invalid_batch_dimension() -> Result<()> {
        let invalid_batch = BitNetTensor::zeros(&[2, 16, 128, 64], DType::F32, &Device::Cpu)?;

        let result = validate_kv_cache_dims(&invalid_batch, 0, 1, 16, 2048, 64);

        assert!(result.is_err(), "Invalid batch should fail validation");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("batch dimension mismatch"),
            "Error should mention batch dimension"
        );
        Ok(())
    }

    #[test]
    fn test_invalid_heads_dimension() -> Result<()> {
        let invalid_heads = BitNetTensor::zeros(&[1, 8, 128, 64], DType::F32, &Device::Cpu)?;

        let result = validate_kv_cache_dims(&invalid_heads, 0, 1, 16, 2048, 64);

        assert!(result.is_err(), "Invalid heads should fail validation");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("heads dimension mismatch"),
            "Error should mention heads dimension"
        );
        Ok(())
    }

    #[test]
    fn test_sequence_length_overflow() -> Result<()> {
        let seq_overflow = BitNetTensor::zeros(&[1, 16, 2100, 64], DType::F32, &Device::Cpu)?;

        let result = validate_kv_cache_dims(&seq_overflow, 0, 1, 16, 2048, 64);

        assert!(result.is_err(), "seq_len overflow should fail validation");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("sequence length exceeds max"),
            "Error should mention sequence length overflow"
        );
        Ok(())
    }

    #[test]
    fn test_invalid_head_dimension() -> Result<()> {
        let invalid_head_dim = BitNetTensor::zeros(&[1, 16, 128, 32], DType::F32, &Device::Cpu)?;

        let result = validate_kv_cache_dims(&invalid_head_dim, 0, 1, 16, 2048, 64);

        assert!(result.is_err(), "Invalid head_dim should fail validation");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("head dimension mismatch"), "Error should mention head dimension");
        Ok(())
    }

    #[test]
    fn test_gqa_validation() -> Result<()> {
        // GQA: num_q_heads=32, num_kv_heads=8
        // Cache should have num_kv_heads=8 (not num_q_heads=32)
        let gqa_cache = BitNetTensor::zeros(&[1, 8, 128, 64], DType::F32, &Device::Cpu)?;

        let result = validate_kv_cache_dims(&gqa_cache, 0, 1, 8, 2048, 64);

        assert!(result.is_ok(), "GQA cache with num_kv_heads=8 should pass validation");
        Ok(())
    }
}
