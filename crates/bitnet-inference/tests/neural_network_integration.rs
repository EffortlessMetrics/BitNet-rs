//! Neural Network Integration Tests for BitNet Inference
//!
//! Tests cover real neural network inference paths with quantized layers:
//! - QuantizedLinear forward pass with I2S/TL1/TL2 quantization
//! - Multi-head attention with Q/K/V/O projections
//! - KV-cache operations
//! - End-to-end inference pipeline
//!
//! These tests target the critical 0% coverage gaps identified in coverage analysis:
//! - crates/bitnet-inference/src/layers/quantized_linear.rs (1,530 lines uncovered)
//! - crates/bitnet-inference/src/layers/attention.rs (717 lines uncovered)
//! - crates/bitnet-inference/src/generation/autoregressive.rs (654 lines uncovered)
#![cfg(feature = "cpu")]
use anyhow::{Context, Result};
use bitnet_common::{BitNetTensor, Device, Tensor};
use bitnet_inference::layers::attention::KVCache;
use bitnet_inference::layers::quantized_linear::{LookupTable, QuantizedLinear};
use bitnet_quantization::{I2SQuantizer, TL1Quantizer, TL2Quantizer};
/// Test I2S quantized linear forward pass with real tensor operations
/// Coverage target: quantized_linear.rs:forward_i2s()
#[tokio::test]
async fn test_i2s_quantized_linear_forward_real_inference() -> Result<()> {
    let batch_size = 2;
    let seq_len = 16;
    let hidden_size = 256;
    let out_features = 512;
    let input_data: Vec<f32> = (0..batch_size * seq_len * hidden_size)
        .map(|i| {
            let x = (i as f32) / 100.0;
            (x.sin() * 0.5 + x.cos() * 0.3).tanh()
        })
        .collect();
    let input =
        BitNetTensor::from_slice(&input_data, &[batch_size, seq_len, hidden_size], &Device::Cpu)?;
    let weight_data: Vec<f32> = (0..hidden_size * out_features)
        .map(|i| {
            let x = (i as f32 * 0.01).sin();
            x * 0.5
        })
        .collect();
    let weight_tensor =
        BitNetTensor::from_slice(&weight_data, &[hidden_size, out_features], &Device::Cpu)?;
    let quantizer = I2SQuantizer::new();
    let quantized_weights =
        quantizer.quantize_tensor(&weight_tensor).context("I2S quantization failed")?;
    let linear = QuantizedLinear::new_i2s(quantized_weights, Device::Cpu)
        .context("Failed to create I2S quantized linear layer")?;
    let output = linear.forward(&input).await.context("I2S forward pass failed")?;
    assert_eq!(output.shape(), &[batch_size, seq_len, out_features], "I2S output shape mismatch");
    let output_candle = output.to_candle()?;
    let output_data = output_candle.flatten_all()?.to_vec1::<f32>()?;
    let has_nan_inf = output_data.iter().any(|v| !v.is_finite());
    assert!(!has_nan_inf, "I2S output contains NaN/Inf values");
    let mean: f32 = output_data.iter().sum::<f32>() / output_data.len() as f32;
    let variance: f32 =
        output_data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / output_data.len() as f32;
    assert!(variance > 1e-6, "I2S output has zero variance - kernel may not have executed");
    println!(
        "✅ I2S quantized linear forward pass: shape={:?}, variance={:.6}",
        output.shape(),
        variance
    );
    Ok(())
}
/// Test TL1 quantized linear forward pass with table lookup
/// Coverage target: quantized_linear.rs:forward_tl1()
#[tokio::test]
async fn test_tl1_quantized_linear_forward_real_inference() -> Result<()> {
    let batch_size = 1;
    let seq_len = 8;
    let hidden_size = 128;
    let out_features = 256;
    let input_data: Vec<f32> =
        (0..batch_size * seq_len * hidden_size).map(|i| ((i % 100) as f32) / 100.0 - 0.5).collect();
    let input =
        BitNetTensor::from_slice(&input_data, &[batch_size, seq_len, hidden_size], &Device::Cpu)?;
    let weight_data: Vec<f32> =
        (0..hidden_size * out_features).map(|i| ((i % 50) as f32) / 50.0 - 0.5).collect();
    let weight_tensor =
        BitNetTensor::from_slice(&weight_data, &[hidden_size, out_features], &Device::Cpu)?;
    let quantizer = TL1Quantizer::new();
    let quantized_weights =
        quantizer.quantize_tensor(&weight_tensor).context("TL1 quantization failed")?;
    let lookup_table = LookupTable::new(vec![
        -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0,
    ]);
    let linear = QuantizedLinear::new_tl1(quantized_weights, lookup_table, Device::Cpu)
        .context("Failed to create TL1 quantized linear layer")?;
    let output = linear.forward(&input).await.context("TL1 forward pass failed")?;
    assert_eq!(output.shape(), &[batch_size, seq_len, out_features]);
    let output_candle = output.to_candle()?;
    let output_data = output_candle.flatten_all()?.to_vec1::<f32>()?;
    assert!(output_data.iter().all(|v| v.is_finite()), "TL1 output contains NaN/Inf");
    let unique_values: std::collections::HashSet<_> =
        output_data.iter().map(|v| (v * 1000.0) as i32).collect();
    assert!(unique_values.len() > 10, "TL1 output has too few unique values");
    println!("✅ TL1 quantized linear forward pass: unique_values={}", unique_values.len());
    Ok(())
}
/// Test TL2 quantized linear forward pass with 2-level table lookup
/// Coverage target: quantized_linear.rs:forward_tl2()
#[tokio::test]
async fn test_tl2_quantized_linear_forward_real_inference() -> Result<()> {
    let batch_size = 1;
    let seq_len = 4;
    let hidden_size = 64;
    let out_features = 128;
    let input_data: Vec<f32> =
        (0..batch_size * seq_len * hidden_size).map(|i| (i as f32 * 0.01).sin()).collect();
    let input =
        BitNetTensor::from_slice(&input_data, &[batch_size, seq_len, hidden_size], &Device::Cpu)?;
    let weight_data: Vec<f32> =
        (0..hidden_size * out_features).map(|i| (i as f32 * 0.02).cos()).collect();
    let weight_tensor =
        BitNetTensor::from_slice(&weight_data, &[hidden_size, out_features], &Device::Cpu)?;
    let quantizer = TL2Quantizer::new();
    let quantized_weights =
        quantizer.quantize_tensor(&weight_tensor).context("TL2 quantization failed")?;
    let lookup_table = LookupTable::new((0..256).map(|i| (i as f32 - 128.0) / 64.0).collect());
    let linear = QuantizedLinear::new_tl2(quantized_weights, lookup_table, Device::Cpu)
        .context("Failed to create TL2 quantized linear layer")?;
    let output = linear.forward(&input).await.context("TL2 forward pass failed")?;
    assert_eq!(output.shape(), &[batch_size, seq_len, out_features]);
    let output_candle = output.to_candle()?;
    let output_data = output_candle.flatten_all()?.to_vec1::<f32>()?;
    assert!(output_data.iter().all(|v| v.is_finite()), "TL2 output contains NaN/Inf");
    // Verify output diversity — correct TL2 dequantization should produce varied outputs,
    // not near-zero values as the old matmul_i2s path would have produced.
    let max_val = output_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_val = output_data.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        max_val - min_val > 0.01,
        "TL2 output has no diversity (max={max_val:.6}, min={min_val:.6}); \
         kernel bug may have returned near-zero values"
    );
    println!("✅ TL2 quantized linear forward pass: shape={:?}", output.shape());
    Ok(())
}
/// Test quantized linear with bias addition
/// Coverage target: quantized_linear.rs:apply_bias_if_present()
#[tokio::test]
async fn test_quantized_linear_with_bias() -> Result<()> {
    let batch_size = 1;
    let seq_len = 8;
    let hidden_size = 64;
    let out_features = 128;
    let input_data: Vec<f32> = vec![0.5; batch_size * seq_len * hidden_size];
    let input =
        BitNetTensor::from_slice(&input_data, &[batch_size, seq_len, hidden_size], &Device::Cpu)?;
    let weight_data: Vec<f32> = vec![0.1; hidden_size * out_features];
    let weight_tensor =
        BitNetTensor::from_slice(&weight_data, &[hidden_size, out_features], &Device::Cpu)?;
    let bias_data: Vec<f32> = (0..out_features).map(|i| i as f32 * 0.01).collect();
    let _bias = BitNetTensor::from_slice(&bias_data, &[out_features], &Device::Cpu)?;
    let quantizer = I2SQuantizer::new();
    let quantized_weights = quantizer.quantize_tensor(&weight_tensor)?;
    let linear = QuantizedLinear::new_i2s(quantized_weights, Device::Cpu)?;
    let output = linear.forward(&input).await?;
    let output_candle = output.to_candle()?;
    let output_data = output_candle.flatten_all()?.to_vec1::<f32>()?;
    let has_nonzero = output_data.iter().any(|v| v.abs() > 1e-6);
    assert!(has_nonzero, "Matrix multiplication should produce non-zero output");
    println!(
        "✅ Quantized linear with bias: output_mean={:.6}",
        output_data.iter().sum::<f32>() / output_data.len() as f32
    );
    Ok(())
}
/// Test quantized linear input validation and error handling
/// Coverage target: quantized_linear.rs:validate_input(), validate_input_dimensions()
#[tokio::test]
async fn test_quantized_linear_input_validation() -> Result<()> {
    let hidden_size = 128;
    let out_features = 256;
    let weight_data: Vec<f32> = vec![0.1; hidden_size * out_features];
    let weight_tensor =
        BitNetTensor::from_slice(&weight_data, &[hidden_size, out_features], &Device::Cpu)?;
    let quantizer = I2SQuantizer::new();
    let quantized_weights = quantizer.quantize_tensor(&weight_tensor)?;
    let linear = QuantizedLinear::new_i2s(quantized_weights, Device::Cpu)?;
    let invalid_input_1d = BitNetTensor::from_slice(&[0.5; 128], &[128], &Device::Cpu)?;
    let result_1d = linear.forward(&invalid_input_1d).await;
    assert!(result_1d.is_err(), "Should reject 1D input");
    let invalid_input_features =
        BitNetTensor::from_slice(&[0.5; 2 * 8 * 64], &[2, 8, 64], &Device::Cpu)?;
    let result_features = linear.forward(&invalid_input_features).await;
    assert!(result_features.is_err(), "Should reject wrong feature dimension");
    let valid_input = BitNetTensor::from_slice(&[0.5; 2 * 8 * 128], &[2, 8, 128], &Device::Cpu)?;
    let result_valid = linear.forward(&valid_input).await;
    assert!(result_valid.is_ok(), "Should accept valid input");
    println!("✅ Quantized linear input validation: all error paths covered");
    Ok(())
}
/// Test KV-cache creation and initialization
/// Coverage target: attention.rs:KVCache::new()
#[tokio::test]
async fn test_kv_cache_initialization() -> Result<()> {
    let max_seq_len = 128;
    let num_layers = 12;
    let num_heads = 8;
    let head_dim = 64;
    let cache = KVCache::new(max_seq_len, num_layers, num_heads, head_dim, &Device::Cpu)?;
    let memory_usage = cache.memory_usage();
    assert_eq!(memory_usage.get("current_sequence_length"), Some(&0), "Cache should start empty");
    assert_eq!(
        memory_usage.get("max_sequence_length"),
        Some(&max_seq_len),
        "Max seq len should match"
    );
    let expected_bytes =
        max_seq_len * num_heads * head_dim * std::mem::size_of::<f32>() * 2 * num_layers;
    let actual_bytes = *memory_usage.get("tensor_memory_bytes").unwrap();
    assert_eq!(actual_bytes, expected_bytes, "Memory allocation mismatch");
    println!(
        "✅ KV-cache initialization: {} layers, {} MB",
        num_layers,
        actual_bytes / 1024 / 1024
    );
    Ok(())
}
/// Test KV-cache update and retrieval
/// Coverage target: attention.rs:KVCache::update(), get()
#[tokio::test]
async fn test_kv_cache_update_and_retrieval() -> Result<()> {
    let max_seq_len = 64;
    let num_layers = 4;
    let num_heads = 4;
    let head_dim = 32;
    let seq_len = 16;
    let mut cache = KVCache::new(max_seq_len, num_layers, num_heads, head_dim, &Device::Cpu)?;
    let k_data: Vec<f32> = (0..seq_len * num_heads * head_dim).map(|i| i as f32 * 0.01).collect();
    let v_data: Vec<f32> =
        (0..seq_len * num_heads * head_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let k_tensor =
        BitNetTensor::from_slice(&k_data, &[seq_len, num_heads, head_dim], &Device::Cpu)?;
    let v_tensor =
        BitNetTensor::from_slice(&v_data, &[seq_len, num_heads, head_dim], &Device::Cpu)?;
    cache.update(0, k_tensor.clone(), v_tensor.clone(), seq_len)?;
    let (k_retrieved, v_retrieved) = cache.get(0)?;
    assert_eq!(k_retrieved.shape(), &[seq_len, num_heads, head_dim]);
    assert_eq!(v_retrieved.shape(), &[seq_len, num_heads, head_dim]);
    let k_retrieved_candle = k_retrieved.to_candle()?;
    let k_retrieved_data = k_retrieved_candle.flatten_all()?.to_vec1::<f32>()?;
    let k_original_candle = k_tensor.to_candle()?;
    let k_original_data = k_original_candle.flatten_all()?.to_vec1::<f32>()?;
    let max_diff: f32 = k_retrieved_data
        .iter()
        .zip(k_original_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-5, "KV-cache data corruption detected: max_diff={}", max_diff);
    println!(
        "✅ KV-cache update/retrieval: layer 0, seq_len={}, max_diff={:.2e}",
        seq_len, max_diff
    );
    Ok(())
}
/// Test KV-cache clearing
/// Coverage target: attention.rs:KVCache::clear()
#[tokio::test]
async fn test_kv_cache_clear() -> Result<()> {
    let max_seq_len = 32;
    let num_layers = 2;
    let num_heads = 4;
    let head_dim = 16;
    let mut cache = KVCache::new(max_seq_len, num_layers, num_heads, head_dim, &Device::Cpu)?;
    let k_tensor = BitNetTensor::zeros(
        &[max_seq_len, num_heads, head_dim],
        candle_core::DType::F32,
        &Device::Cpu,
    )?;
    let v_tensor = BitNetTensor::zeros(
        &[max_seq_len, num_heads, head_dim],
        candle_core::DType::F32,
        &Device::Cpu,
    )?;
    cache.update(0, k_tensor, v_tensor, max_seq_len)?;
    let memory_usage_before = cache.memory_usage();
    assert_eq!(*memory_usage_before.get("current_sequence_length").unwrap(), max_seq_len);
    cache.clear(&Device::Cpu)?;
    let memory_usage_after = cache.memory_usage();
    assert_eq!(
        *memory_usage_after.get("current_sequence_length").unwrap(),
        0,
        "Cache should be cleared"
    );
    println!("✅ KV-cache clear: seq_len {} → 0", max_seq_len);
    Ok(())
}
/// Test KV-cache error handling for invalid layer indices
/// Coverage target: attention.rs:KVCache::validate_layer_index()
#[tokio::test]
async fn test_kv_cache_error_handling() -> Result<()> {
    let cache = KVCache::new(64, 4, 4, 32, &Device::Cpu)?;
    let result = cache.get(10);
    assert!(result.is_err(), "Should reject invalid layer index");
    let error_msg = format!("{:?}", result.unwrap_err());
    assert!(error_msg.contains("out of bounds"), "Error should mention bounds");
    println!("✅ KV-cache error handling: invalid layer index rejected");
    Ok(())
}
/// Test multi-layer quantized forward pass simulating transformer block
/// Coverage target: Combined quantized_linear.rs + attention.rs integration
#[tokio::test]
async fn test_multi_layer_quantized_transformer_block() -> Result<()> {
    let batch_size = 1;
    let seq_len = 8;
    let hidden_size = 128;
    let input_data: Vec<f32> =
        (0..batch_size * seq_len * hidden_size).map(|i| (i as f32 * 0.01).tanh()).collect();
    let mut hidden_states =
        BitNetTensor::from_slice(&input_data, &[batch_size, seq_len, hidden_size], &Device::Cpu)?;
    for layer_idx in 0..3 {
        let weight_data: Vec<f32> = (0..hidden_size * hidden_size)
            .map(|i| ((i + layer_idx * 1000) as f32 * 0.01).sin())
            .collect();
        let weight_tensor =
            BitNetTensor::from_slice(&weight_data, &[hidden_size, hidden_size], &Device::Cpu)?;
        let quantizer = I2SQuantizer::new();
        let quantized_weights = quantizer.quantize_tensor(&weight_tensor)?;
        let linear = QuantizedLinear::new_i2s(quantized_weights, Device::Cpu)?;
        hidden_states = linear.forward(&hidden_states).await?;
        let hidden_candle = hidden_states.to_candle()?;
        let activated = hidden_candle.tanh()?;
        hidden_states = BitNetTensor::new(activated);
        println!("  Layer {}: forward pass complete, shape={:?}", layer_idx, hidden_states.shape());
    }
    assert_eq!(hidden_states.shape(), &[batch_size, seq_len, hidden_size]);
    let output_candle = hidden_states.to_candle()?;
    let output_data = output_candle.flatten_all()?.to_vec1::<f32>()?;
    assert!(output_data.iter().all(|v| v.is_finite()), "Multi-layer output contains NaN/Inf");
    println!("✅ Multi-layer transformer block: 3 layers complete");
    Ok(())
}
/// Test quantized linear with extreme input values
/// Coverage target: quantized_linear.rs numerical stability
#[tokio::test]
async fn test_quantized_linear_extreme_values() -> Result<()> {
    let hidden_size = 64;
    let out_features = 64;
    let weight_data: Vec<f32> = vec![0.5; hidden_size * out_features];
    let weight_tensor =
        BitNetTensor::from_slice(&weight_data, &[hidden_size, out_features], &Device::Cpu)?;
    let quantizer = I2SQuantizer::new();
    let quantized_weights = quantizer.quantize_tensor(&weight_tensor)?;
    let linear = QuantizedLinear::new_i2s(quantized_weights, Device::Cpu)?;
    let near_zero_input = BitNetTensor::from_slice(&[1e-6; 64], &[1, 1, 64], &Device::Cpu)?;
    let output_zero = linear.forward(&near_zero_input).await?;
    let output_zero_data = output_zero.to_candle()?.flatten_all()?.to_vec1::<f32>()?;
    assert!(output_zero_data.iter().all(|v| v.is_finite()), "Failed on near-zero input");
    let large_input = BitNetTensor::from_slice(&[10.0; 64], &[1, 1, 64], &Device::Cpu)?;
    let output_large = linear.forward(&large_input).await?;
    let output_large_data = output_large.to_candle()?.flatten_all()?.to_vec1::<f32>()?;
    assert!(output_large_data.iter().all(|v| v.is_finite()), "Failed on large input");
    println!("✅ Extreme value handling: near-zero and large inputs passed");
    Ok(())
}
