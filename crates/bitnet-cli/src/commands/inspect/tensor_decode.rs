//! Tensor decoding and RMS computation primitives for the inspect command.
//!
//! Single responsibility: convert raw GGUF tensor bytes into a Candle
//! `Tensor`, and compute the root-mean-square statistic over its values.
//! Only float tensor types (F32, F16) are supported — quantized projection
//! weights are filtered out earlier in the scan pipeline.

use anyhow::Result;
use bitnet_common::BitNetError;
use bitnet_models::formats::gguf::GgufTensorType;
use candle_core::{DType, Tensor};

use super::tensor_scanner::TensorKind;

/// Decode a tensor from raw GGUF bytes into a Candle `Tensor`.
///
/// Errors if the tensor type is anything other than F32 or F16; quantized
/// inputs are not valid for RMS validation and are filtered out upstream.
pub(crate) fn decode_tensor(
    name: &str,
    shape: &[usize],
    tensor_type: GgufTensorType,
    data: &[u8],
    tensor_kind: TensorKind,
) -> Result<Tensor> {
    let tensor = match tensor_type {
        GgufTensorType::F32 => {
            let float_data = bytemuck::cast_slice::<u8, f32>(data);
            Tensor::from_slice(float_data, shape, &candle_core::Device::Cpu)
                .map_err(|e| anyhow::anyhow!("Failed to create F32 tensor '{}': {}", name, e))?
        }
        GgufTensorType::F16 => {
            let half_data = bytemuck::cast_slice::<u8, u16>(data);
            let float_data: Vec<f32> =
                half_data.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect();
            Tensor::from_slice(&float_data, shape, &candle_core::Device::Cpu)
                .map_err(|e| anyhow::anyhow!("Failed to create F16 tensor '{}': {}", name, e))?
        }
        _ => {
            let kind_str = match tensor_kind {
                TensorKind::LayerNorm => "LayerNorm",
                TensorKind::Projection => "Projection",
            };
            return Err(anyhow::anyhow!(
                "{} tensor '{}' has quantized type {:?}, expected float (F32/F16) for RMS validation",
                kind_str,
                name,
                tensor_type
            ));
        }
    };

    Ok(tensor)
}

/// Compute the root-mean-square statistic of `tensor`.
pub(crate) fn compute_rms(tensor: &Tensor) -> Result<f32> {
    let t32 = tensor.to_dtype(DType::F32).map_err(|e| BitNetError::Validation(e.to_string()))?;

    let mean_sq = t32
        .sqr()
        .map_err(|e| BitNetError::Validation(e.to_string()))?
        .mean_all()
        .map_err(|e| BitNetError::Validation(e.to_string()))?
        .to_scalar::<f32>()
        .map_err(|e| BitNetError::Validation(e.to_string()))?;

    Ok(mean_sq.sqrt())
}
