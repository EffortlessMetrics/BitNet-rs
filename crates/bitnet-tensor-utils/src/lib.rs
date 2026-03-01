//! Tensor extraction helpers shared across CLI/runtime crates.

use anyhow::Result;
use bitnet_common::{BitNetError, ConcreteTensor, Tensor};
use candle_core::{DType, IndexOp};

/// Extract last token hidden state from 3D tensor `[B, T, H] -> [B, H]`.
pub fn extract_last_token_hidden(tensor: &ConcreteTensor) -> Result<ConcreteTensor> {
    let shape = tensor.shape();
    if shape.len() != 3 {
        return Err(BitNetError::Validation("Expected 3D tensor".into()).into());
    }

    let (batch_size, seq_len, hidden_size) = (shape[0], shape[1], shape[2]);

    match tensor {
        ConcreteTensor::BitNet(t) => {
            let candle = t.as_candle();
            let last = candle.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
            Ok(ConcreteTensor::BitNet(bitnet_common::BitNetTensor::new(last)))
        }
        ConcreteTensor::Mock(_) => Ok(ConcreteTensor::mock(vec![batch_size, hidden_size])),
    }
}

/// Extract logits vector from 2D tensor `[B, V] -> Vec<f32>`.
pub fn extract_logits_2d(tensor: &ConcreteTensor) -> Result<Vec<f32>> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(BitNetError::Validation("Expected 2D tensor".into()).into());
    }

    match tensor {
        ConcreteTensor::BitNet(t) => {
            let candle = t.as_candle();
            let batch_0 = candle.i(0)?;
            let batch_0 =
                if batch_0.dtype() != DType::F32 { batch_0.to_dtype(DType::F32)? } else { batch_0 };
            Ok(batch_0.to_vec1::<f32>()?)
        }
        ConcreteTensor::Mock(_) => Ok(vec![0.1; 50257]),
    }
}

/// Extract logits vector from 3D tensor `[B, T, V] -> Vec<f32>` using the last token of batch 0.
pub fn extract_logits_3d_last_token(tensor: &ConcreteTensor) -> Result<Vec<f32>> {
    let shape = tensor.shape();
    if shape.len() != 3 {
        return Err(BitNetError::Validation("Expected 3D tensor".into()).into());
    }

    let seq_len = shape[1];

    match tensor {
        ConcreteTensor::BitNet(t) => {
            let candle = t.as_candle();
            let last = candle.narrow(1, seq_len - 1, 1)?.squeeze(1)?.i(0)?;
            let last = if last.dtype() != DType::F32 { last.to_dtype(DType::F32)? } else { last };
            Ok(last.to_vec1::<f32>()?)
        }
        ConcreteTensor::Mock(_) => Ok(vec![0.1; 50257]),
    }
}

/// Convert any tensor shape to a flattened `Vec<f32>` for diagnostics.
pub fn tensor_to_vec(tensor: &ConcreteTensor) -> Result<Vec<f32>> {
    match tensor {
        ConcreteTensor::BitNet(t) => {
            let candle = t.as_candle();
            let candle_f32 = if candle.dtype() != DType::F32 {
                candle.to_dtype(DType::F32)?
            } else {
                candle.clone()
            };
            let flattened = candle_f32.flatten_all()?;
            Ok(flattened.to_vec1::<f32>()?)
        }
        ConcreteTensor::Mock(mock) => {
            let size: usize = mock.shape().iter().product();
            Ok(vec![0.1; size])
        }
    }
}

/// Compute root-mean-square value for a slice.
#[inline]
pub fn compute_rms(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = xs.iter().map(|x| x * x).sum();
    (sum_sq / (xs.len() as f32)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_last_token_hidden_mock_shape() {
        let t = ConcreteTensor::mock(vec![2, 4, 8]);
        let out = extract_last_token_hidden(&t).expect("shape should be valid");
        assert_eq!(out.shape(), &[2, 8]);
    }

    #[test]
    fn extract_logits_2d_mock_size() {
        let t = ConcreteTensor::mock(vec![1, 32]);
        let out = extract_logits_2d(&t).expect("shape should be valid");
        assert_eq!(out.len(), 50_257);
    }

    #[test]
    fn tensor_to_vec_mock_uses_shape_product() {
        let t = ConcreteTensor::mock(vec![2, 3, 5]);
        let out = tensor_to_vec(&t).expect("mock tensor conversion should work");
        assert_eq!(out.len(), 30);
    }

    #[test]
    fn compute_rms_empty_is_zero() {
        assert_eq!(compute_rms(&[]), 0.0);
    }
}
