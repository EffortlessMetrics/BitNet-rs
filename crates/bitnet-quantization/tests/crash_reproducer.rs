//! Test I2S quantization crash reproducers to understand the issues before fixing

use bitnet_common::BitNetTensor;
use bitnet_quantization::I2SQuantizer;
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use std::fs;

#[test]
fn test_crash_1849515_i2s_extreme_float_values() {
    let crash_file = "/home/steven/code/Rust/BitNet-rs/fuzz/artifacts/quantization_i2s/crash-1849515c7958976d1cf7360b3e0d75d04115d96c";
    if let Ok(data) = fs::read(crash_file) {
        println!("Testing I2S crash 1849515 with {} bytes", data.len());

        // Try to parse the data as f32 values that might cause overflow
        if data.len() >= 4 {
            let float_values: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .take(256) // Limit to reasonable size
                .collect();

            if !float_values.is_empty() {
                println!("1849515: Testing with {} float values", float_values.len());
                println!("Sample values: {:?}", &float_values[..float_values.len().min(5)]);

                // Check for problematic values
                let has_nan = float_values.iter().any(|f| f.is_nan());
                let has_inf = float_values.iter().any(|f| f.is_infinite());
                let max_abs = float_values.iter().map(|f| f.abs()).fold(0.0f32, f32::max);

                println!(
                    "1849515: Has NaN: {}, Has Inf: {}, Max abs: {}",
                    has_nan, has_inf, max_abs
                );

                // Try to create tensor and quantize - this should not panic
                let shape = vec![float_values.len()];
                if let Ok(tensor) =
                    CandleTensor::from_vec(float_values, shape.as_slice(), &CandleDevice::Cpu)
                {
                    let bitnet_tensor = BitNetTensor::new(tensor);
                    let quantizer = I2SQuantizer::new();

                    match quantizer.quantize_tensor(&bitnet_tensor) {
                        Ok(_) => println!("1849515: Quantization succeeded unexpectedly"),
                        Err(e) => println!("1849515: Expected error: {}", e),
                    }
                }
            }
        }
    } else {
        println!("I2S crash file 1849515 not found - skipping");
    }
}

#[test]
fn test_crash_79f55aa_i2s_nan_infinite_values() {
    let crash_file = "/home/steven/code/Rust/BitNet-rs/fuzz/artifacts/quantization_i2s/crash-79f55aabbc9a4b9b83da759a0853dc61a66318d2";
    if let Ok(data) = fs::read(crash_file) {
        println!("Testing I2S crash 79f55aa with {} bytes", data.len());

        // Try to parse the data as f32 values that might cause overflow
        if data.len() >= 4 {
            let float_values: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .take(256) // Limit to reasonable size
                .collect();

            if !float_values.is_empty() {
                println!("79f55aa: Testing with {} float values", float_values.len());
                println!("Sample values: {:?}", &float_values[..float_values.len().min(5)]);

                // Check for problematic values
                let has_nan = float_values.iter().any(|f| f.is_nan());
                let has_inf = float_values.iter().any(|f| f.is_infinite());
                let max_abs = float_values.iter().map(|f| f.abs()).fold(0.0f32, f32::max);

                println!(
                    "79f55aa: Has NaN: {}, Has Inf: {}, Max abs: {}",
                    has_nan, has_inf, max_abs
                );

                // Try to create tensor and quantize - this should not panic
                let shape = vec![float_values.len()];
                if let Ok(tensor) =
                    CandleTensor::from_vec(float_values, shape.as_slice(), &CandleDevice::Cpu)
                {
                    let bitnet_tensor = BitNetTensor::new(tensor);
                    let quantizer = I2SQuantizer::new();

                    match quantizer.quantize_tensor(&bitnet_tensor) {
                        Ok(_) => println!("79f55aa: Quantization succeeded unexpectedly"),
                        Err(e) => println!("79f55aa: Expected error: {}", e),
                    }
                }
            }
        }
    } else {
        println!("I2S crash file 79f55aa not found - skipping");
    }
}
