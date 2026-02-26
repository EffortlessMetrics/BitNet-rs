//! # Candle Tensor Interoperability Example
//!
//! This example demonstrates how to integrate BitNet-rs with the Candle tensor library
//! for seamless tensor operations and device management.

use anyhow::Result;
use candle_core::{Device, Tensor as CandleTensor, DType};
use bitnet_common::{BitNetConfig, Tensor};
use bitnet_models::{BitNetModel, ModelLoader};
use bitnet_inference::InferenceEngine;
use std::path::Path;

/// Demonstrates converting between BitNet tensors and Candle tensors
fn tensor_conversion_example() -> Result<()> {
    println!("=== Candle Tensor Interoperability Example ===\n");

    // Create a Candle device (CPU or CUDA)
    let device = Device::Cpu;
    println!("Using device: {:?}", device);

    // Create a sample Candle tensor
    let candle_tensor = CandleTensor::randn(0f32, 1f32, (4, 512), &device)?;
    println!("Created Candle tensor with shape: {:?}", candle_tensor.shape());

    // Convert Candle tensor to BitNet tensor
    let bitnet_tensor = candle_to_bitnet_tensor(&candle_tensor)?;
    println!("Converted to BitNet tensor with shape: {:?}", bitnet_tensor.shape());

    // Perform operations on BitNet tensor
    let processed_tensor = process_with_bitnet(bitnet_tensor)?;

    // Convert back to Candle tensor
    let result_candle = bitnet_to_candle_tensor(&processed_tensor, &device)?;
    println!("Converted back to Candle tensor with shape: {:?}", result_candle.shape());

    Ok(())
}

/// Convert Candle tensor to BitNet tensor format
fn candle_to_bitnet_tensor(candle_tensor: &CandleTensor) -> Result<Box<dyn Tensor>> {
    // Extract data from Candle tensor
    let data = candle_tensor.to_vec1::<f32>()?;
    let shape = candle_tensor.shape().dims().to_vec();

    // Create BitNet tensor wrapper
    Ok(Box::new(BitNetTensorWrapper {
        data,
        shape,
        dtype: bitnet_common::DType::F32,
    }))
}

/// Convert BitNet tensor to Candle tensor
fn bitnet_to_candle_tensor(bitnet_tensor: &dyn Tensor, device: &Device) -> Result<CandleTensor> {
    let data = bitnet_tensor.as_slice::<f32>()?;
    let shape = bitnet_tensor.shape();

    CandleTensor::from_slice(data, shape, device)
}

/// Example BitNet tensor wrapper for Candle interoperability
struct BitNetTensorWrapper {
    data: Vec<f32>,
    shape: Vec<usize>,
    dtype: bitnet_common::DType,
}

impl Tensor for BitNetTensorWrapper {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> bitnet_common::DType {
        self.dtype
    }

    fn device(&self) -> &bitnet_common::Device {
        &bitnet_common::Device::Cpu
    }

    fn as_slice<T>(&self) -> Result<&[T]> {
        // Safety: This is a simplified example - in production, proper type checking is required
        unsafe {
            let ptr = self.data.as_ptr() as *const T;
            let slice = std::slice::from_raw_parts(ptr, self.data.len());
            Ok(slice)
        }
    }
}

/// Process tensor using BitNet operations
fn process_with_bitnet(tensor: Box<dyn Tensor>) -> Result<Box<dyn Tensor>> {
    // Example processing - in practice, this would use BitNet quantization/inference
    println!("Processing tensor with BitNet operations...");

    // Simulate some processing
    let shape = tensor.shape().to_vec();
    let processed_data = vec![1.0f32; shape.iter().product()];

    Ok(Box::new(BitNetTensorWrapper {
        data: processed_data,
        shape,
        dtype: bitnet_common::DType::F32,
    }))
}

/// Demonstrates using Candle for preprocessing and BitNet for inference
fn hybrid_inference_pipeline() -> Result<()> {
    println!("\n=== Hybrid Candle + BitNet Inference Pipeline ===\n");

    let device = Device::Cpu;

    // Step 1: Preprocess input with Candle
    println!("1. Preprocessing input with Candle...");
    let input_text = "Hello, world!";
    let token_ids = vec![15496, 11, 995, 0]; // Example token IDs

    let input_tensor = CandleTensor::from_slice(
        &token_ids,
        (1, token_ids.len()),
        &device
    )?;

    // Apply Candle preprocessing (normalization, etc.)
    let normalized = input_tensor.to_dtype(DType::F32)? / 1000.0;
    println!("Preprocessed tensor shape: {:?}", normalized.shape());

    // Step 2: Convert to BitNet format for inference
    println!("2. Converting to BitNet format...");
    let bitnet_input = candle_to_bitnet_tensor(&normalized)?;

    // Step 3: Run BitNet inference (simulated)
    println!("3. Running BitNet inference...");
    let inference_result = simulate_bitnet_inference(bitnet_input)?;

    // Step 4: Convert back to Candle for postprocessing
    println!("4. Converting back to Candle for postprocessing...");
    let candle_result = bitnet_to_candle_tensor(&*inference_result, &device)?;

    // Step 5: Postprocess with Candle
    let final_result = candle_result.softmax(1)?;
    println!("Final result shape: {:?}", final_result.shape());

    // Extract top predictions
    let probabilities = final_result.to_vec2::<f32>()?;
    println!("Top predictions: {:?}", &probabilities[0][..5]);

    Ok(())
}

/// Simulate BitNet inference for demonstration
fn simulate_bitnet_inference(input: Box<dyn Tensor>) -> Result<Box<dyn Tensor>> {
    let shape = input.shape();
    println!("Running inference on tensor with shape: {:?}", shape);

    // Simulate model output (vocabulary size = 50257 for GPT-2)
    let vocab_size = 50257;
    let batch_size = shape[0];
    let output_shape = vec![batch_size, vocab_size];

    // Generate random logits for demonstration
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let logits: Vec<f32> = (0..batch_size * vocab_size)
        .map(|_| rng.gen_range(-5.0..5.0))
        .collect();

    Ok(Box::new(BitNetTensorWrapper {
        data: logits,
        shape: output_shape,
        dtype: bitnet_common::DType::F32,
    }))
}

/// Demonstrates device management between Candle and BitNet
fn device_management_example() -> Result<()> {
    println!("\n=== Device Management Example ===\n");

    // Try to use CUDA if available, fallback to CPU
    let candle_device = if candle_core::utils::cuda_is_available() {
        println!("CUDA available, using GPU");
        Device::new_cuda(0)?
    } else {
        println!("CUDA not available, using CPU");
        Device::Cpu
    };

    // Create tensor on the selected device
    let tensor = CandleTensor::randn(0f32, 1f32, (2, 1024), &candle_device)?;
    println!("Created tensor on device: {:?}", candle_device);

    // Convert to BitNet (which will handle device mapping internally)
    let bitnet_tensor = candle_to_bitnet_tensor(&tensor)?;
    println!("BitNet tensor device: {:?}", bitnet_tensor.device());

    // Demonstrate moving between devices (if multiple devices available)
    if candle_core::utils::cuda_is_available() {
        let cpu_tensor = tensor.to_device(&Device::Cpu)?;
        println!("Moved tensor to CPU: {:?}", cpu_tensor.device());
    }

    Ok(())
}

fn main() -> Result<()> {
    // Run all examples
    tensor_conversion_example()?;
    hybrid_inference_pipeline()?;
    device_management_example()?;

    println!("\n=== Candle Integration Examples Complete ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_conversion() -> Result<()> {
        let device = Device::Cpu;
        let candle_tensor = CandleTensor::zeros((2, 3), DType::F32, &device)?;

        let bitnet_tensor = candle_to_bitnet_tensor(&candle_tensor)?;
        assert_eq!(bitnet_tensor.shape(), &[2, 3]);

        let converted_back = bitnet_to_candle_tensor(&*bitnet_tensor, &device)?;
        assert_eq!(converted_back.shape().dims(), &[2, 3]);

        Ok(())
    }

    #[test]
    fn test_device_compatibility() -> Result<()> {
        let device = Device::Cpu;
        let tensor = CandleTensor::ones((1, 4), DType::F32, &device)?;

        let bitnet_tensor = candle_to_bitnet_tensor(&tensor)?;
        // Verify device compatibility
        assert_eq!(bitnet_tensor.device(), &bitnet_common::Device::Cpu);

        Ok(())
    }
}
