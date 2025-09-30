# Stub code: `validate_quantization_sanity` in `engine.rs` is a placeholder

The `validate_quantization_sanity` function in `crates/bitnet-inference/src/engine.rs` is called during engine initialization, but it only prints a message and mentions what a real implementation would do. It doesn't perform actual quantization sanity checks. This is a form of stubbing and should be replaced with a real implementation.

**File:** `crates/bitnet-inference/src/engine.rs`

**Function:** `validate_quantization_sanity`

**Code:**
```rust
    fn validate_quantization_sanity(&self) -> Result<()> {
        eprintln!("=== Quantization Sanity Check ===");

        // This is a simplified version since the full quantization validation would require
        // access to the actual quantized tensors and multiple dequantization backends.
        // In a real implementation, we would:
        // 1. Pick a small quantized tensor from the model
        // 2. Dequantize using different backends (CPU vs GPU, different SIMD paths)
        // 3. Compare results with MSE < threshold
        // 4. Ensure scales and blocks are correct

        // For now, we'll validate that basic quantization parameters are reasonable
        let _config = self.model.config();
        eprintln!("Quantization validation:");
        eprintln!("  Model appears to use quantized weights");

        // In a full implementation, this would do actual dequantization comparison:
        // let test_data = get_small_quantized_block();
        // let cpu_result = dequant_i2s_cpu(&test_data)?;
        // let gpu_result = dequant_i2s_gpu(&test_data)?;
        // let mse = mean_squared_error(&cpu_result, &gpu_result);
        // if mse > 1e-6 { return Err(...) }

        eprintln!("✅ Quantization sanity check passed (basic validation)");
        eprintln!("================================");

        Ok(())
    }
```

## Proposed Fix

The `validate_quantization_sanity` function should be implemented to perform actual quantization sanity checks. This would involve:

1.  **Picking a small quantized tensor from the model:** Select a representative quantized tensor from the loaded model.
2.  **Dequantizing using different backends:** Dequantize the selected tensor using different backends (e.g., CPU and GPU, or different SIMD paths).
3.  **Comparing results:** Compare the dequantized results from different backends using a metric like Mean Squared Error (MSE). If the MSE exceeds a predefined threshold, it indicates a quantization issue.
4.  **Ensuring scales and blocks are correct:** Validate that the quantization scales and block sizes are correctly applied.

### Example Implementation

```rust
    fn validate_quantization_sanity(&self) -> Result<()> {
        eprintln!("=== Quantization Sanity Check ===");

        let config = self.model.config();
        if let Some(q_type) = config.quantization {
            eprintln!("Model uses quantization type: {:?}", q_type);

            // Example: Pick a small quantized tensor (this would need actual model access)
            // let test_tensor = self.model.get_some_quantized_tensor()?; 

            // Simulate dequantization for now
            let cpu_dequant_result = vec![0.1, 0.2, 0.3]; // Simulate CPU dequantization
            let gpu_dequant_result = vec![0.100001, 0.200001, 0.300001]; // Simulate GPU dequantization

            let mse = calculate_mse(&cpu_dequant_result, &gpu_dequant_result);
            if mse > 1e-6 {
                return Err(anyhow::anyhow!("Quantization sanity check failed: MSE between CPU and GPU dequantization is too high ({})", mse));
            }
            eprintln!("  MSE between CPU and GPU dequantization: {:.8}", mse);
        } else {
            eprintln!("Model does not appear to use quantized weights");
        }

        eprintln!("✅ Quantization sanity check passed");
        eprintln!("================================");

        Ok(())
    }

    fn calculate_mse(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum::<f64>() / a.len() as f64
    }
```
