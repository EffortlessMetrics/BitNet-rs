//! Helper functions for AC1 quantized linear layer tests
use anyhow::{Context, Result};
use bitnet_common::{BitNetTensor, Device, QuantizationType, Tensor};
use bitnet_inference::layers::quantized_linear::{LookupTable, QuantizedLinear};
use bitnet_quantization::Quantize;
/// Create mock tensor with specified dimensions (for AC1 testing)
pub fn create_mock_tensor(
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
) -> Result<BitNetTensor> {
    let shape = vec![batch_size, seq_len, hidden_size];
    let total_elements = shape.iter().product::<usize>();
    let data: Vec<f32> =
        (0..total_elements).map(|i| ((i as f32 % 1000.0) / 1000.0) - 0.5).collect();
    Ok(BitNetTensor::from_slice(&data, &shape, &Device::Cpu)?)
}
/// Test I2S quantized linear layer forward pass
pub async fn test_i2s_linear_layer(input: &BitNetTensor, hidden_size: usize) -> Result<f32> {
    let weight_data: Vec<f32> =
        (0..hidden_size * hidden_size).map(|i| (i as f32).sin() * 0.01).collect();
    let fp32_weights =
        BitNetTensor::from_slice(&weight_data, &[hidden_size, hidden_size], &Device::Cpu)?;
    let quantized_weights = fp32_weights.quantize(QuantizationType::I2S)?;
    let layer = QuantizedLinear::new_i2s(quantized_weights, Device::Cpu)?;
    let output = layer.forward(input).await?;
    let input_shape = input.shape();
    let output_shape = output.shape();
    assert_eq!(output_shape.len(), 3, "Output should be 3D");
    assert_eq!(output_shape[0], input_shape[0], "Batch size mismatch");
    assert_eq!(output_shape[1], input_shape[1], "Sequence length mismatch");
    assert_eq!(output_shape[2], hidden_size, "Hidden size mismatch");
    validate_tensor_values(&output)?;
    Ok(1.0)
}
/// Test TL1 quantized linear layer forward pass
pub async fn test_tl1_linear_layer(input: &BitNetTensor, hidden_size: usize) -> Result<f32> {
    let weight_data: Vec<f32> =
        (0..hidden_size * hidden_size).map(|i| (i as f32).cos() * 0.01).collect();
    let fp32_weights =
        BitNetTensor::from_slice(&weight_data, &[hidden_size, hidden_size], &Device::Cpu)?;
    let quantized_weights = fp32_weights.quantize(QuantizationType::TL1)?;
    let lookup_entries: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
    let lookup_table = LookupTable::new(lookup_entries);
    let layer = QuantizedLinear::new_tl1(quantized_weights, lookup_table, Device::Cpu)?;
    let output = layer.forward(input).await?;
    let input_shape = input.shape();
    let output_shape = output.shape();
    assert_eq!(output_shape.len(), 3, "Output should be 3D");
    assert_eq!(output_shape[0], input_shape[0], "Batch size mismatch");
    assert_eq!(output_shape[1], input_shape[1], "Sequence length mismatch");
    assert_eq!(output_shape[2], hidden_size, "Hidden size mismatch");
    validate_tensor_values(&output)?;
    Ok(1.0)
}
/// Test TL2 quantized linear layer forward pass
pub async fn test_tl2_linear_layer(input: &BitNetTensor, hidden_size: usize) -> Result<f32> {
    let weight_data: Vec<f32> = (0..hidden_size * hidden_size)
        .map(|i| ((i as f32 * 0.01).sin() + (i as f32 * 0.02).cos()) * 0.01)
        .collect();
    let fp32_weights =
        BitNetTensor::from_slice(&weight_data, &[hidden_size, hidden_size], &Device::Cpu)?;
    let quantized_weights = fp32_weights.quantize(QuantizationType::TL2)?;
    let lookup_entries: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
    let lookup_table = LookupTable::new(lookup_entries);
    let layer = QuantizedLinear::new_tl2(quantized_weights, lookup_table, Device::Cpu)?;
    let output = layer.forward(input).await?;
    let input_shape = input.shape();
    let output_shape = output.shape();
    assert_eq!(output_shape.len(), 3, "Output should be 3D");
    assert_eq!(output_shape[0], input_shape[0], "Batch size mismatch");
    assert_eq!(output_shape[1], input_shape[1], "Sequence length mismatch");
    assert_eq!(output_shape[2], hidden_size, "Hidden size mismatch");
    validate_tensor_values(&output)?;
    Ok(1.0)
}
/// Validate tensor contains no NaN or Inf values
fn validate_tensor_values(tensor: &BitNetTensor) -> Result<()> {
    let tensor_candle = tensor.to_candle()?;
    let flat = tensor_candle.flatten_all()?;
    let values = flat.to_vec1::<f32>().context("Failed to extract tensor values")?;
    for (i, &val) in values.iter().enumerate() {
        if !val.is_finite() {
            anyhow::bail!("Invalid value at index {}: {}", i, val);
        }
    }
    Ok(())
}
