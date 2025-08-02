//! Quantization algorithms for BitNet models

use bitnet_common::{BitNetTensor, QuantizationType, Result, Tensor};

pub mod i2s;
pub mod tl1;
pub mod tl2;

/// Quantization trait
pub trait Quantize {
    fn quantize(&self, qtype: QuantizationType) -> Result<QuantizedTensor>;
    fn dequantize(&self) -> Result<BitNetTensor>;
}

/// Quantized tensor representation
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
    pub shape: Vec<usize>,
    pub qtype: QuantizationType,
}

impl QuantizedTensor {
    pub fn new(
        data: Vec<u8>,
        scales: Vec<f32>,
        shape: Vec<usize>,
        qtype: QuantizationType,
    ) -> Self {
        Self {
            data,
            scales,
            shape,
            qtype,
        }
    }
}