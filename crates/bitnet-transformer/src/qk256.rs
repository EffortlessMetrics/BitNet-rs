//! QK256 raw-tensor naming and inline scale extraction helpers.

use bitnet_common::{BitNetError, Result};
use candle_core::Tensor;

fn qk256_scale_key(qk256_key: &str) -> String {
    if let Some(base) = qk256_key.strip_suffix(".qk256_qs") {
        format!("{base}.qk256_scale")
    } else {
        format!("{qk256_key}.qk256_scale")
    }
}

pub(crate) fn qk256_inline_scale(
    raw_tensors: &std::collections::HashMap<String, Tensor>,
    qk256_key: &str,
) -> Result<Option<f32>> {
    let scale_key = qk256_scale_key(qk256_key);
    let Some(scale_tensor) = raw_tensors.get(&scale_key) else {
        return Ok(None);
    };

    let scale_values = scale_tensor.flatten_all()?.to_vec1::<f32>().map_err(|e| {
        BitNetError::Validation(format!("failed to read QK256 inline scale {scale_key}: {e}"))
    })?;
    let [scale] = scale_values.as_slice() else {
        return Err(BitNetError::Validation(format!(
            "QK256 inline scale {scale_key} must contain exactly one value, got {}",
            scale_values.len()
        )));
    };
    let scale = *scale;
    if !scale.is_finite() {
        return Err(BitNetError::Validation(format!(
            "QK256 inline scale {scale_key} is not finite: {scale}"
        )));
    }

    Ok(Some(scale))
}

pub(crate) const TIED_EMBED_QK256_KEY: &str = "embed_tokens.weight.qk256_qs";
