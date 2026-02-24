#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::{QuantizationType, Result};
use bitnet_quantization::Quantize;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    data: Vec<f32>,
    shape: Vec<usize>,
}

fuzz_target!(|input: FuzzInput| {
    // Skip empty or invalid inputs
    if input.data.is_empty() || input.shape.is_empty() {
        return;
    }

    // Ensure shape product doesn't overflow and is consistent with data length.
    // Fuzzer can supply [usize::MAX, 2] which would panic with .product().
    const MAX_FUZZ_ELEMENTS: usize = 1_000_000;
    let Some(total_elements) =
        input.shape.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    else {
        return;
    };
    if total_elements == 0
        || total_elements > MAX_FUZZ_ELEMENTS
        || total_elements != input.data.len()
    {
        return;
    }

    // Filter out problematic values that could cause issues
    let filtered_data: Vec<f32> = input
        .data
        .into_iter()
        .map(|x| {
            if x.is_nan() || x.is_infinite() {
                0.0
            } else if x.abs() > 1000.0 {
                x.signum() * 1000.0 // Clamp to reasonable range
            } else {
                x
            }
        })
        .collect();

    // Create a mock tensor for testing
    let tensor = FuzzMockTensor::new(filtered_data, input.shape);

    // Test I2S quantization - create a BitNetTensor for testing
    if let Ok(bitnet_tensor) = tensor.to_bitnet_tensor()
        && let Ok(quantized) = bitnet_tensor.quantize(QuantizationType::I2S)
    {
        // Test that quantized data is valid
        assert!(!quantized.data.is_empty());
        assert!(!quantized.scales.is_empty());
        assert_eq!(quantized.qtype, QuantizationType::I2S);

        // Test dequantization doesn't panic
        let _ = quantized.dequantize();
    }
});

// Enhanced mock tensor for fuzzing with conversion capability
struct FuzzMockTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl FuzzMockTensor {
    fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    // Convert to BitNetTensor for quantization testing
    fn to_bitnet_tensor(&self) -> Result<bitnet_common::BitNetTensor> {
        use bitnet_common::{BitNetTensor, Device};
        BitNetTensor::from_slice(&self.data, &self.shape, &Device::Cpu)
    }
}
