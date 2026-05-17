//! Single-pass tensor scan that classifies and validates LayerNorm and
//! projection weights.
//!
//! Single responsibility: walk the GGUF tensor table once, classify each
//! tensor by name (LayerNorm vs projection), decode its bytes via
//! [`super::tensor_decode`], compute its RMS, and apply the appropriate
//! validation rule from the supplied [`Ruleset`]. The aggregated counts
//! and per-tensor statistics are returned in [`ScanResults`] for the
//! output stage to render.

use anyhow::Result;
use bitnet_models::formats::gguf::{GgufReader, GgufTensorType, TensorInfo};
use bitnet_models::names::{is_layernorm_weight, is_projection_weight};
use tracing::debug;

use super::tensor_decode;
use crate::ln_rules::Ruleset;

/// Statistics for a single inspected tensor.
#[derive(Debug)]
pub(crate) struct TensorStat {
    pub(crate) name: String,
    pub(crate) rms: f32,
    pub(crate) is_ok: bool,
    pub(crate) kind: TensorKind,
}

/// Type of tensor being validated.
#[derive(Debug, Clone, Copy)]
pub(crate) enum TensorKind {
    LayerNorm,
    Projection,
}

/// Aggregated scan output: per-tensor stats plus per-kind counts.
#[derive(Debug, Default)]
pub(crate) struct ScanResults {
    pub(crate) stats: Vec<TensorStat>,
    pub(crate) ln_bad_count: usize,
    pub(crate) ln_total_count: usize,
    pub(crate) proj_bad_count: usize,
    pub(crate) proj_total_count: usize,
}

impl ScanResults {
    /// Total number of tensors that failed validation across both kinds.
    pub(crate) fn total_bad(&self) -> usize {
        self.ln_bad_count + self.proj_bad_count
    }
}

/// Walk the tensor table, validating LayerNorm and projection weights.
pub(crate) fn scan(reader: &GgufReader, rules: &Ruleset) -> Result<ScanResults> {
    let tensor_count = reader.tensor_count() as usize;
    debug!("Inspecting {} tensors for LayerNorm gamma statistics", tensor_count);

    let mut results = ScanResults::default();

    for i in 0..tensor_count {
        let info = reader.get_tensor_info(i)?;

        if is_layernorm_weight(&info.name) {
            scan_layernorm(reader, i, info, rules, &mut results)?;
        } else if is_projection_weight(&info.name) {
            scan_projection(reader, i, info, rules, &mut results)?;
        }
    }

    Ok(results)
}

fn scan_layernorm(
    reader: &GgufReader,
    index: usize,
    info: &TensorInfo,
    rules: &Ruleset,
    results: &mut ScanResults,
) -> Result<()> {
    debug!("Processing LayerNorm tensor: {} (type: {:?})", info.name, info.tensor_type);
    results.ln_total_count += 1;

    let tensor_data = reader.get_tensor_data(index)?;
    let tensor = tensor_decode::decode_tensor(
        &info.name,
        &info.shape,
        info.tensor_type,
        tensor_data,
        TensorKind::LayerNorm,
    )?;

    let rms = tensor_decode::compute_rms(&tensor)?;
    let is_ok = rules.check_ln(&info.name, rms);

    if !is_ok {
        results.ln_bad_count += 1;
    }

    results.stats.push(TensorStat {
        name: info.name.clone(),
        rms,
        is_ok,
        kind: TensorKind::LayerNorm,
    });

    Ok(())
}

fn scan_projection(
    reader: &GgufReader,
    index: usize,
    info: &TensorInfo,
    rules: &Ruleset,
    results: &mut ScanResults,
) -> Result<()> {
    // Quantized projection weights (I2_S, etc.) are expected and don't carry
    // a meaningful RMS — skip them entirely instead of trying to decode.
    if !matches!(info.tensor_type, GgufTensorType::F32 | GgufTensorType::F16) {
        debug!(
            "Skipping RMS validation for quantized projection tensor: {} (type: {:?})",
            info.name, info.tensor_type
        );
        return Ok(());
    }

    results.proj_total_count += 1;

    let tensor_data = reader.get_tensor_data(index)?;
    let tensor = tensor_decode::decode_tensor(
        &info.name,
        &info.shape,
        info.tensor_type,
        tensor_data,
        TensorKind::Projection,
    )?;

    let rms = tensor_decode::compute_rms(&tensor)?;
    let is_ok = rules.check_proj_rms(rms);

    if !is_ok {
        results.proj_bad_count += 1;
    }

    results.stats.push(TensorStat {
        name: info.name.clone(),
        rms,
        is_ok,
        kind: TensorKind::Projection,
    });

    Ok(())
}
