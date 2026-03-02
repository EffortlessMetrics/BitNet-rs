//! Reusable tensor extraction and diagnostics helpers for the BitNet CLI.

use anyhow::Result;
use bitnet_common::Tensor;
use candle_core::{DType, IndexOp};

/// Extract last token hidden state from 3D tensor [B,T,H] -> [B,H].
pub fn extract_last_token_hidden(
    tensor: &bitnet_common::ConcreteTensor,
) -> Result<bitnet_common::ConcreteTensor> {
    use bitnet_common::{BitNetError, ConcreteTensor};

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

/// Extract logits vector from 2D tensor [B,V] -> `Vec<f32>`.
pub fn extract_logits_2d(tensor: &bitnet_common::ConcreteTensor) -> Result<Vec<f32>> {
    use bitnet_common::{BitNetError, ConcreteTensor};

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

/// Extract logits vector from 3D tensor [B,T,V], taking last timestep and first batch.
pub fn extract_logits(tensor: &bitnet_common::ConcreteTensor) -> Result<Vec<f32>> {
    use bitnet_common::{BitNetError, ConcreteTensor};

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

/// Convert tensor to flattened f32 vector for diagnostics.
pub fn tensor_to_vec(tensor: &bitnet_common::ConcreteTensor) -> Result<Vec<f32>> {
    use bitnet_common::ConcreteTensor;

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

/// Compute RMS (root mean square) of a vector.
#[must_use]
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
    use bitnet_common::ConcreteTensor;

    #[test]
    fn compute_rms_empty_is_zero() {
        assert_eq!(compute_rms(&[]), 0.0);
    }

    #[test]
    fn compute_rms_matches_expected_value() {
        let got = compute_rms(&[3.0, 4.0]);
        let expected = (12.5_f32).sqrt();
        assert!((got - expected).abs() < 1e-6);
    }

    #[test]
    fn tensor_to_vec_mock_matches_flattened_size() {
        let t = ConcreteTensor::mock(vec![2, 3, 4]);
        let values = tensor_to_vec(&t).expect("mock tensor conversion should work");
        assert_eq!(values.len(), 24);
    }

    #[test]
    fn extract_last_token_hidden_rejects_non_3d() {
        let t = ConcreteTensor::mock(vec![2, 3]);
        let err = extract_last_token_hidden(&t).expect_err("shape validation should fail");
        assert!(err.to_string().contains("Expected 3D tensor"));
    }

    #[test]
    fn extract_logits_2d_rejects_non_2d() {
        let t = ConcreteTensor::mock(vec![2, 3, 4]);
        let err = extract_logits_2d(&t).expect_err("shape validation should fail");
        assert!(err.to_string().contains("Expected 2D tensor"));
    }

    #[test]
    fn extract_logits_mock_returns_vocab_sized_vector() {
        let t = ConcreteTensor::mock(vec![1, 5, 50257]);
        let logits = extract_logits(&t).expect("mock logits extraction should work");
        assert_eq!(logits.len(), 50257);
    }
}
