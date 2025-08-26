#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use bitnet_quantization::{QuantizationType, Quantize};
use bitnet_common::Device;

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
    
    // Ensure shape is consistent with data length
    let total_elements: usize = input.shape.iter().product();
    if total_elements == 0 || total_elements != input.data.len() {
        return;
    }
    
    // Filter out problematic values that could cause issues
    let filtered_data: Vec<f32> = input.data.into_iter()
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
    let tensor = MockTensor::new(filtered_data, input.shape);
    
    // Test I2S quantization
    if let Ok(quantized) = tensor.quantize(QuantizationType::I2S) {
        // Test that quantized data is valid
        assert!(!quantized.data.is_empty());
        assert!(!quantized.scales.is_empty());
        assert_eq!(quantized.qtype, QuantizationType::I2S);
        
        // Test dequantization doesn't panic
        let _ = quantized.dequantize(&Device::Cpu);
    }
});

// Mock tensor implementation for fuzzing
struct MockTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl MockTensor {
    fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

impl bitnet_quantization::Tensor for MockTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> bitnet_quantization::DType {
        bitnet_quantization::DType::F32
    }

    fn device(&self) -> &bitnet_quantization::Device {
        &bitnet_quantization::Device::Cpu
    }

    fn as_slice<T>(&self) -> Result<&[T], bitnet_quantization::BitNetError> {
        // Unsafe cast for fuzzing - this is acceptable in fuzz tests
        unsafe {
            let ptr = self.data.as_ptr() as *const T;
            let slice = std::slice::from_raw_parts(ptr, self.data.len());
            Ok(slice)
        }
    }
}

impl Quantize for MockTensor {
    fn quantize(&self, qtype: QuantizationType) -> Result<bitnet_quantization::QuantizedTensor, bitnet_quantization::BitNetError> {
        match qtype {
            QuantizationType::I2S => bitnet_quantization::i2s::quantize_i2s(self),
            QuantizationType::TL1 => bitnet_quantization::tl1::quantize_tl1(self),
            QuantizationType::TL2 => bitnet_quantization::tl2::quantize_tl2(self),
        }
    }
}