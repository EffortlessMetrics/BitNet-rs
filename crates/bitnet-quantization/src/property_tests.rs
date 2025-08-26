// Property-based testing for quantization operations
use proptest::prelude::*;
use crate::{QuantizationType, Quantize, QuantizedTensor, Device};
use std::collections::HashMap;

/// Test that quantization followed by dequantization preserves approximate values
pub fn quantization_roundtrip_property() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-10.0f32..10.0f32, 1..1000)
}

/// Test that quantization is deterministic
pub fn quantization_deterministic_property() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-5.0f32..5.0f32, 10..100)
}

/// Test that quantization handles edge cases properly
pub fn quantization_edge_cases_property() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        prop_oneof![
            Just(0.0f32),
            Just(f32::INFINITY),
            Just(f32::NEG_INFINITY),
            Just(f32::NAN),
            -1000.0f32..1000.0f32,
        ],
        1..50
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MockTensor; // Assuming we have a mock tensor for testing

    proptest! {
        #[test]
        fn test_i2s_quantization_roundtrip(data in quantization_roundtrip_property()) {
            let tensor = MockTensor::from_vec(data.clone());
            let quantized = tensor.quantize(QuantizationType::I2S).unwrap();
            let dequantized = quantized.dequantize(&Device::Cpu).unwrap();
            
            // Check that the dequantized values are close to original
            let original_slice = tensor.as_slice::<f32>().unwrap();
            let dequant_slice = dequantized.as_slice::<f32>().unwrap();
            
            for (orig, dequant) in original_slice.iter().zip(dequant_slice.iter()) {
                if orig.is_finite() && dequant.is_finite() {
                    let relative_error = (orig - dequant).abs() / orig.abs().max(1e-6);
                    prop_assert!(relative_error < 0.1, 
                        "Relative error too large: {} vs {}", orig, dequant);
                }
            }
        }
    }
}   
 proptest! {
        #[test]
        fn test_quantization_deterministic(data in quantization_deterministic_property()) {
            let tensor = MockTensor::from_vec(data);
            
            // Quantize the same data twice
            let quantized1 = tensor.quantize(QuantizationType::I2S).unwrap();
            let quantized2 = tensor.quantize(QuantizationType::I2S).unwrap();
            
            // Results should be identical
            prop_assert_eq!(quantized1.data, quantized2.data);
            prop_assert_eq!(quantized1.scales, quantized2.scales);
        }

        #[test]
        fn test_quantization_preserves_shape(data in quantization_roundtrip_property()) {
            let original_shape = vec![data.len()];
            let tensor = MockTensor::from_vec_with_shape(data, original_shape.clone());
            
            let quantized = tensor.quantize(QuantizationType::I2S).unwrap();
            prop_assert_eq!(quantized.shape, original_shape);
            
            let dequantized = quantized.dequantize(&Device::Cpu).unwrap();
            prop_assert_eq!(dequantized.shape(), &original_shape[..]);
        }

        #[test]
        fn test_quantization_handles_edge_cases(data in quantization_edge_cases_property()) {
            let tensor = MockTensor::from_vec(data);
            
            // Quantization should not panic on edge cases
            let result = tensor.quantize(QuantizationType::I2S);
            
            // Either succeeds or fails gracefully
            match result {
                Ok(quantized) => {
                    // If successful, dequantization should also work
                    let _dequantized = quantized.dequantize(&Device::Cpu).unwrap();
                }
                Err(_) => {
                    // Graceful failure is acceptable for edge cases
                }
            }
        }

        #[test]
        fn test_tl1_quantization_properties(data in quantization_roundtrip_property()) {
            let tensor = MockTensor::from_vec(data.clone());
            
            if let Ok(quantized) = tensor.quantize(QuantizationType::TL1) {
                // TL1 should use lookup tables efficiently
                prop_assert!(quantized.data.len() <= data.len() / 2, 
                    "TL1 quantization should compress data");
                
                // Scales should be reasonable
                for scale in &quantized.scales {
                    prop_assert!(scale.is_finite() && *scale >= 0.0,
                        "Scale values should be finite and non-negative");
                }
            }
        }

        #[test]
        fn test_tl2_quantization_properties(data in quantization_roundtrip_property()) {
            let tensor = MockTensor::from_vec(data.clone());
            
            if let Ok(quantized) = tensor.quantize(QuantizationType::TL2) {
                // TL2 should also compress data
                prop_assert!(quantized.data.len() <= data.len() / 2,
                    "TL2 quantization should compress data");
                
                // Check that quantization type is preserved
                prop_assert_eq!(quantized.qtype, QuantizationType::TL2);
            }
        }
    }

    /// Test quantization with specific patterns
    #[test]
    fn test_quantization_with_patterns() {
        // Test with all zeros
        let zeros = vec![0.0f32; 100];
        let tensor = MockTensor::from_vec(zeros);
        let quantized = tensor.quantize(QuantizationType::I2S).unwrap();
        let dequantized = quantized.dequantize(&Device::Cpu).unwrap();
        
        for value in dequantized.as_slice::<f32>().unwrap() {
            assert!((value.abs() < 1e-6), "Zero values should remain close to zero");
        }

        // Test with alternating pattern
        let alternating: Vec<f32> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let tensor = MockTensor::from_vec(alternating);
        let quantized = tensor.quantize(QuantizationType::I2S).unwrap();
        let _dequantized = quantized.dequantize(&Device::Cpu).unwrap();
        // Should not panic and should produce reasonable results
    }

    /// Test quantization memory usage
    #[test]
    fn test_quantization_memory_efficiency() {
        let large_data: Vec<f32> = (0..10000).map(|i| (i as f32) * 0.001).collect();
        let tensor = MockTensor::from_vec(large_data.clone());
        
        let quantized = tensor.quantize(QuantizationType::I2S).unwrap();
        
        // I2S should use 2 bits per weight, so roughly 4x compression
        let original_bytes = large_data.len() * 4; // f32 = 4 bytes
        let quantized_bytes = quantized.data.len() + quantized.scales.len() * 4;
        
        assert!(quantized_bytes < original_bytes / 2, 
            "Quantization should significantly reduce memory usage");
    }
}

/// Mock tensor implementation for testing
#[cfg(test)]
pub struct MockTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

#[cfg(test)]
impl MockTensor {
    pub fn from_vec(data: Vec<f32>) -> Self {
        let shape = vec![data.len()];
        Self { data, shape }
    }

    pub fn from_vec_with_shape(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

#[cfg(test)]
impl crate::Tensor for MockTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> crate::DType {
        crate::DType::F32
    }

    fn device(&self) -> &crate::Device {
        &crate::Device::Cpu
    }

    fn as_slice<T>(&self) -> Result<&[T], crate::BitNetError> {
        // Unsafe cast for testing - in real implementation this would be properly typed
        unsafe {
            let ptr = self.data.as_ptr() as *const T;
            let slice = std::slice::from_raw_parts(ptr, self.data.len());
            Ok(slice)
        }
    }
}

#[cfg(test)]
impl Quantize for MockTensor {
    fn quantize(&self, qtype: QuantizationType) -> Result<QuantizedTensor, crate::BitNetError> {
        match qtype {
            QuantizationType::I2S => crate::i2s::quantize_i2s(self),
            QuantizationType::TL1 => crate::tl1::quantize_tl1(self),
            QuantizationType::TL2 => crate::tl2::quantize_tl2(self),
        }
    }

    fn dequantize(&self, device: &Device) -> Result<BitNetTensor, crate::BitNetError> {
        BitNetTensor::from_slice(&self.data, &self.shape, device)
    }
}