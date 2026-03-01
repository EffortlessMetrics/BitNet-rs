//! SRP microcrate for reusable tensor extraction and diagnostics helpers.

use bitnet_common::{BitNetError, ConcreteTensor, Result, Tensor};
use candle_core::IndexOp;

/// Extract last token hidden state from a 3D tensor `[B, T, H] -> [B, H]`.
pub fn extract_last_token_hidden(tensor: &ConcreteTensor) -> Result<ConcreteTensor> {
    let shape = tensor.shape();
    if shape.len() != 3 {
        return Err(BitNetError::Validation("Expected 3D tensor".into()));
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

/// Extract logits vector from a 2D tensor `[B, V] -> Vec<f32>`.
pub fn extract_logits_2d(tensor: &ConcreteTensor) -> Result<Vec<f32>> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(BitNetError::Validation("Expected 2D tensor".into()));
    }

    match tensor {
        ConcreteTensor::BitNet(t) => {
            let candle = t.as_candle();
            let batch_0 = candle.i(0)?;
            let batch_0 = if batch_0.dtype() != candle_core::DType::F32 {
                batch_0.to_dtype(candle_core::DType::F32)?
            } else {
                batch_0
            };
            Ok(batch_0.to_vec1::<f32>()?)
        }
        ConcreteTensor::Mock(_) => Ok(vec![0.1; 50_257]),
    }
}

/// Extract logits vector from a legacy 3D tensor `[B, T, V] -> Vec<f32>`.
pub fn extract_logits_3d_legacy(tensor: &ConcreteTensor) -> Result<Vec<f32>> {
    let shape = tensor.shape();
    if shape.len() != 3 {
        return Err(BitNetError::Validation("Expected 3D tensor".into()));
    }

    let seq_len = shape[1];
    match tensor {
        ConcreteTensor::BitNet(t) => {
            let candle = t.as_candle();
            let last = candle.narrow(1, seq_len - 1, 1)?.squeeze(1)?.i(0)?;
            let last = if last.dtype() != candle_core::DType::F32 {
                last.to_dtype(candle_core::DType::F32)?
            } else {
                last
            };
            Ok(last.to_vec1::<f32>()?)
        }
        ConcreteTensor::Mock(_) => Ok(vec![0.1; 50_257]),
    }
}

/// Flatten a tensor to f32 values for diagnostics.
pub fn tensor_to_vec(tensor: &ConcreteTensor) -> Result<Vec<f32>> {
    match tensor {
        ConcreteTensor::BitNet(t) => {
            let candle = t.as_candle();
            let candle_f32 = if candle.dtype() != candle_core::DType::F32 {
                candle.to_dtype(candle_core::DType::F32)?
            } else {
                candle.clone()
            };
            Ok(candle_f32.flatten_all()?.to_vec1::<f32>()?)
        }
        ConcreteTensor::Mock(mock) => {
            let size: usize = mock.shape().iter().product();
            Ok(vec![0.1; size])
        }
    }
}

/// Compute RMS (root mean square) of values.
#[inline]
#[must_use]
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
    fn mock_extract_last_token_hidden_returns_bxh_shape() {
        let t = ConcreteTensor::mock(vec![2, 4, 8]);
        let out = extract_last_token_hidden(&t).expect("extract should succeed");
        assert_eq!(out.shape(), vec![2, 8]);
    }

    #[test]
    fn extract_logits_2d_rejects_non_2d_shapes() {
        let t = ConcreteTensor::mock(vec![2, 4, 8]);
        let err = extract_logits_2d(&t).expect_err("expected validation error");
        assert!(matches!(err, BitNetError::Validation(_)));
    }

    #[test]
    fn tensor_to_vec_mock_uses_shape_product() {
        let t = ConcreteTensor::mock(vec![2, 3, 5]);
        let values = tensor_to_vec(&t).expect("flatten should succeed");
        assert_eq!(values.len(), 30);
    }

    #[test]
    fn compute_rms_handles_empty_and_non_empty() {
        assert_eq!(compute_rms(&[]), 0.0);
        let rms = compute_rms(&[3.0, 4.0]);
        assert!((rms - 3.535534).abs() < 1e-5);
    }
}
