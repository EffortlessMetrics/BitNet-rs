//! Environment-gated diagnostics and Qwen trace helpers.
//!
//! This module owns debug flag caching, tensor statistics logging, and JSONL
//! trace emission so the transformer layers can stay focused on model math.

use candle_core::{DType, Tensor};
use std::sync::OnceLock;

#[derive(Clone, Debug)]
struct QwenTraceConfig {
    path: Option<std::path::PathBuf>,
    stderr_enabled: bool,
    layer: usize,
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok()
}

fn env_flag_eq_1(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

fn qwen_trace_config() -> &'static QwenTraceConfig {
    static CONFIG: OnceLock<QwenTraceConfig> = OnceLock::new();
    CONFIG.get_or_init(|| QwenTraceConfig {
        path: std::env::var("BITNET_QWEN_TRACE_JSONL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(std::path::PathBuf::from),
        stderr_enabled: env_flag_eq_1("BITNET_QWEN_TRACE"),
        layer: std::env::var("BITNET_QWEN_TRACE_LAYER")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0),
    })
}

pub(crate) fn debug_attn_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("DEBUG_ATTN"))
}

pub(crate) fn debug_attn_scale_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("BITNET_DEBUG_ATTN_SCALE"))
}

pub(crate) fn debug_gqa_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("BITNET_DEBUG_GQA"))
}

pub(crate) fn debug_rope_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("BITNET_DEBUG_ROPE"))
}

pub(crate) fn debug_rmsnorm_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("BITNET_DEBUG_RMSNORM"))
}

pub(crate) fn debug_mlp_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("BITNET_DEBUG_MLP"))
}

pub(crate) fn trace_rms_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_eq_1("BITNET_TRACE_RMS"))
}

fn qwen_trace_path() -> Option<std::path::PathBuf> {
    qwen_trace_config().path.clone()
}

fn qwen_trace_enabled() -> bool {
    let config = qwen_trace_config();
    config.path.is_some() || config.stderr_enabled
}

fn qwen_trace_active() -> bool {
    qwen_trace_enabled() && std::env::var("BITNET_QWEN_TRACE_ACTIVE").as_deref() == Ok("1")
}

pub(crate) fn qwen_trace_layer_enabled(layer_idx: usize) -> bool {
    if !qwen_trace_active() {
        return false;
    }
    qwen_trace_config().layer == layer_idx
}

fn qwen_trace_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\r', "\\r")
}

pub(crate) fn qwen_trace_number(value: f64) -> String {
    if value.is_finite() { format!("{value:.9}") } else { "null".to_string() }
}

fn qwen_trace_write_line(line: &str) {
    if let Some(path) = qwen_trace_path() {
        if let Some(parent) = path.parent()
            && let Err(err) = std::fs::create_dir_all(parent)
        {
            eprintln!("qwen_trace_write_failed: create_dir_all {}: {err}", parent.display());
            return;
        }
        match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            Ok(mut file) => {
                if let Err(err) = std::io::Write::write_all(&mut file, line.as_bytes())
                    .and_then(|_| std::io::Write::write_all(&mut file, b"\n"))
                {
                    eprintln!("qwen_trace_write_failed: {}: {err}", path.display());
                }
            }
            Err(err) => eprintln!("qwen_trace_write_failed: {}: {err}", path.display()),
        }
    } else if std::env::var("BITNET_QWEN_TRACE").as_deref() == Ok("1") {
        eprintln!("{line}");
    }
}

pub(crate) fn qwen_trace_event(stage: &str, fields_json: &str) {
    if !qwen_trace_enabled() {
        return;
    }
    let step = std::env::var("BITNET_QWEN_TRACE_STEP").unwrap_or_else(|_| "null".to_string());
    qwen_trace_write_line(&format!(
        "{{\"kind\":\"qwen_trace_event\",\"stage\":\"{}\",\"step\":{},{} }}",
        qwen_trace_escape(stage),
        step,
        fields_json
    ));
}

pub(crate) fn qwen_trace_events_enabled() -> bool {
    qwen_trace_enabled()
}

pub(crate) fn qwen_trace_tensor(
    stage: &str,
    layer_idx: Option<usize>,
    tensor: &Tensor,
) -> candle_core::Result<()> {
    if !qwen_trace_active() {
        return Ok(());
    }
    if let Some(layer_idx) = layer_idx
        && !qwen_trace_layer_enabled(layer_idx)
    {
        return Ok(());
    }

    let tensor_f32 =
        if tensor.dtype() == DType::F32 { tensor.clone() } else { tensor.to_dtype(DType::F32)? };
    let values = tensor_f32.flatten_all()?.to_vec1::<f32>()?;
    let mut finite_count = 0usize;
    let mut nonfinite_count = 0usize;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut checksum = 0.0f64;
    for (idx, value) in values.iter().enumerate() {
        let value = *value as f64;
        if value.is_finite() {
            finite_count += 1;
            sum += value;
            sum_sq += value * value;
            min = min.min(value);
            max = max.max(value);
            if idx < 4096 {
                checksum += value * ((idx % 257) + 1) as f64;
            }
        } else {
            nonfinite_count += 1;
        }
    }

    let denom = finite_count.max(1) as f64;
    let mean = sum / denom;
    let rms = (sum_sq / denom).sqrt();
    let sample = values
        .iter()
        .take(8)
        .map(|value| qwen_trace_number(*value as f64))
        .collect::<Vec<_>>()
        .join(",");
    let dims = tensor.dims().iter().map(|dim| dim.to_string()).collect::<Vec<_>>().join(",");
    let layer_json = layer_idx.map(|idx| idx.to_string()).unwrap_or_else(|| "null".to_string());
    let step = std::env::var("BITNET_QWEN_TRACE_STEP").unwrap_or_else(|_| "null".to_string());

    qwen_trace_write_line(&format!(
        "{{\"kind\":\"qwen_trace_tensor\",\"stage\":\"{}\",\"step\":{},\"layer\":{},\"dtype\":\"{:?}\",\"dims\":[{}],\"len\":{},\"finite\":{},\"nonfinite\":{},\"mean\":{},\"rms\":{},\"min\":{},\"max\":{},\"checksum\":{},\"sample\":[{}]}}",
        qwen_trace_escape(stage),
        step,
        layer_json,
        tensor.dtype(),
        dims,
        values.len(),
        finite_count,
        nonfinite_count,
        qwen_trace_number(mean),
        qwen_trace_number(rms),
        qwen_trace_number(min),
        qwen_trace_number(max),
        qwen_trace_number(checksum),
        sample
    ));
    Ok(())
}

/// Debug helper for tensor statistics (only runs if DEBUG_ATTN env var is set)
pub(crate) fn dbg_stats(tag: &str, t: &Tensor) -> candle_core::Result<()> {
    if debug_attn_enabled() {
        let mean = t.mean_all()?.to_scalar::<f32>()?;
        // Compute std manually: sqrt(E[(x - mean)^2])
        let diff = t.broadcast_sub(&t.mean_all()?)?;
        let variance = diff.sqr()?.mean_all()?;
        let std = variance.sqrt()?.to_scalar::<f32>()?;
        eprintln!("[dbg] {tag}: mean={mean:.6} std={std:.6}");
    }
    Ok(())
}

/// Debug helper for checking finite values
pub(crate) fn dbg_finite(tag: &str, t: &Tensor) -> candle_core::Result<()> {
    if debug_attn_enabled() {
        let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
        let n = v.len().min(4096);
        let mut n_nan = 0;
        let mut n_inf = 0;
        for &x in &v[..n] {
            if !x.is_finite() {
                if x.is_nan() {
                    n_nan += 1;
                } else {
                    n_inf += 1;
                }
            }
        }
        if n_nan + n_inf > 0 {
            eprintln!(
                "⚠️  [dbg] {tag}: non-finite values: NaN={n_nan} Inf={n_inf} (in first {n} elems)"
            );
        }
    }
    Ok(())
}
