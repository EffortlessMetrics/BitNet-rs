//! Builders for Candle layers whose checkpoint tensors may omit optional bias.
//!
//! Keeping these construction policies here avoids mixing GGUF compatibility
//! decisions into attention, feed-forward, and model assembly code.

use bitnet_common::config::NormType;
use candle_nn::{LayerNorm, Linear, VarBuilder};

/// Helper to create linear layers with optional bias tensors.
pub(crate) fn linear_with_optional_bias(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
) -> candle_core::Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;

    // Missing bias is semantically a no-bias linear layer. Avoid materializing
    // zero tensors and a runtime add for dense GGUF weights that omit bias.
    let bias = match vb.get(out_dim, "bias") {
        Ok(b) => Some(b),
        Err(_) => {
            tracing::debug!("Bias tensor missing for linear layer; using no-bias path [{out_dim}]");
            None
        }
    };

    Ok(Linear::new(weight, bias))
}

/// Helper to create layer norm with optional bias.
/// If `bias` is missing we use no-bias LayerNorm by default, or error when
/// `BITNET_REQUIRE_LAYER_NORM_BIAS=1`.
#[cfg(test)]
pub(crate) fn layer_norm_with_optional_bias(
    normalized_shape: usize,
    eps: f64,
    vb: VarBuilder,
) -> candle_core::Result<LayerNorm> {
    norm_with_optional_bias(NormType::LayerNorm, normalized_shape, eps, vb)
}

pub(crate) fn norm_with_optional_bias(
    norm_type: NormType,
    normalized_shape: usize,
    eps: f64,
    vb: VarBuilder,
) -> candle_core::Result<LayerNorm> {
    let weight = vb.get((normalized_shape,), "weight")?;
    if matches!(norm_type, NormType::RmsNorm) {
        if vb.get((normalized_shape,), "bias").is_ok() {
            tracing::debug!(
                "Bias tensor present for RMSNorm layer; ignoring bias [{}]",
                normalized_shape
            );
        }
        tracing::debug!("Using RMSNorm without mean subtraction [{}]", normalized_shape);
        return Ok(LayerNorm::rms_norm(weight, eps));
    }

    match vb.get((normalized_shape,), "bias") {
        Ok(bias) => {
            // Bias exists → standard LayerNorm (with mean subtraction and bias)
            tracing::debug!("Using LayerNorm with bias [{}]", normalized_shape);
            Ok(LayerNorm::new(weight, bias, eps))
        }
        Err(err) => {
            if std::env::var("BITNET_REQUIRE_LAYER_NORM_BIAS")
                .ok()
                .is_some_and(|value| value == "1")
            {
                return Err(candle_core::Error::Msg(format!(
                    "LayerNorm bias tensor is required but missing (set BITNET_REQUIRE_LAYER_NORM_BIAS=0 or unset to allow no-bias LayerNorm): {err}"
                )));
            }

            // No bias → LayerNorm without bias (but WITH mean subtraction)
            // IMPORTANT: Use LayerNorm::new_no_bias (remove_mean=true) NOT rms_norm (remove_mean=false)
            // because these gamma weights are calibrated for LayerNorm semantics
            // (mean subtraction). RMSNorm callers return earlier in this helper.
            tracing::debug!(
                "Bias tensor missing for norm layer; using LayerNorm without bias (mean subtraction enabled) [{}]",
                normalized_shape
            );
            Ok(LayerNorm::new_no_bias(weight, eps))
        }
    }
}

pub(crate) fn optional_layer_norm_with_optional_bias(
    norm_type: NormType,
    normalized_shape: usize,
    eps: f64,
    vb: VarBuilder,
) -> candle_core::Result<Option<LayerNorm>> {
    if !vb.contains_tensor("weight") {
        return Ok(None);
    }

    Ok(Some(norm_with_optional_bias(norm_type, normalized_shape, eps, vb)?))
}
