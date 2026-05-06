//! QK256 linear dispatch for BitNet transformer layers.
//!
//! The public `forward_qk256` entry point is used by the transformer whenever a
//! `.qk256_qs` raw tensor is present. It records coverage counters for the
//! BitNet linear path and, when the selected backend is CUDA, attempts the CUDA
//! QK256 kernel before falling back according to strict-mode policy.

use bitnet_common::{BitNetError, Result};
use bitnet_qk256_layout_core::{
    Qk256InputShape, Qk256Layout, parse_input_shape, parse_qk256_layout, validate_input_cols,
};
use candle_core::Tensor;
use std::sync::atomic::{AtomicU64, Ordering};

static BITNET_LINEAR_TOTAL: AtomicU64 = AtomicU64::new(0);
static BITNET_LINEAR_ON_CUDA: AtomicU64 = AtomicU64::new(0);
static BITNET_LINEAR_CPU_FALLBACK: AtomicU64 = AtomicU64::new(0);
static BITNET_LINEAR_UNSUPPORTED: AtomicU64 = AtomicU64::new(0);

/// Coverage counters for BitNet QK256 linear dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qk256DispatchCoverageCounters {
    /// Total BitNet linear dispatch points observed by this crate.
    pub bitnet_linear_layers_total: u64,
    /// Dispatch points routed through the CUDA QK256 kernel.
    pub bitnet_linear_layers_on_cuda: u64,
    /// Dispatch points that used CPU fallback while a CUDA backend was requested.
    pub bitnet_linear_layers_cpu_fallback: u64,
    /// Unsupported operations that prevent a full CUDA inference claim.
    pub unsupported_ops: Vec<String>,
    /// Human-readable claim boundary for partial routing.
    pub execution_claim: &'static str,
}

/// Snapshot the current QK256 dispatch coverage counters.
pub fn qk256_dispatch_coverage() -> Qk256DispatchCoverageCounters {
    let cpu_fallback = BITNET_LINEAR_CPU_FALLBACK.load(Ordering::Relaxed);
    let unsupported = BITNET_LINEAR_UNSUPPORTED.load(Ordering::Relaxed);
    let on_cuda = BITNET_LINEAR_ON_CUDA.load(Ordering::Relaxed);
    let mut unsupported_ops = Vec::new();
    if cpu_fallback > 0 {
        unsupported_ops.push("qk256_cpu_fallback".to_string());
    }
    if unsupported > 0 {
        unsupported_ops.push("qk256_strict_cuda_unsupported".to_string());
    }

    Qk256DispatchCoverageCounters {
        bitnet_linear_layers_total: BITNET_LINEAR_TOTAL.load(Ordering::Relaxed),
        bitnet_linear_layers_on_cuda: on_cuda,
        bitnet_linear_layers_cpu_fallback: cpu_fallback,
        unsupported_ops,
        execution_claim: if on_cuda > 0 {
            "cuda_inference_contribution"
        } else if cuda_bitnet_backend_requested() {
            "cuda_bitnet_not_routed"
        } else {
            "cpu_reference"
        },
    }
}

/// Reset dispatch coverage counters.
///
/// This is public so CLI and integration tests can scope receipt counters to a
/// single run without relying on process lifetime.
pub fn reset_qk256_dispatch_coverage() {
    BITNET_LINEAR_TOTAL.store(0, Ordering::Relaxed);
    BITNET_LINEAR_ON_CUDA.store(0, Ordering::Relaxed);
    BITNET_LINEAR_CPU_FALLBACK.store(0, Ordering::Relaxed);
    BITNET_LINEAR_UNSUPPORTED.store(0, Ordering::Relaxed);
}

/// Record a BitNet linear CPU fallback outside the QK256 raw-tensor path.
pub fn record_bitnet_linear_cpu_fallback() {
    BITNET_LINEAR_TOTAL.fetch_add(1, Ordering::Relaxed);
    if cuda_bitnet_backend_requested() {
        BITNET_LINEAR_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
    }
}

/// Record a BitNet linear dispatch point that strict CUDA cannot support.
pub fn record_bitnet_linear_unsupported() {
    BITNET_LINEAR_TOTAL.fetch_add(1, Ordering::Relaxed);
    BITNET_LINEAR_UNSUPPORTED.fetch_add(1, Ordering::Relaxed);
}

/// True when the run selected or requested the RTX 5070 Ti CUDA BitNet lane.
pub fn cuda_bitnet_backend_requested() -> bool {
    backend_env_matches("BITNET_SELECTED_BACKEND")
        || backend_env_matches("BITNET_REQUESTED_BACKEND")
        || backend_env_matches("BITNET_BACKEND")
}

/// True when strict mode forbids CPU fallback for the CUDA BitNet lane.
pub fn strict_cuda_bitnet_backend_requested() -> bool {
    (cuda_bitnet_backend_requested() && env_truthy("BITNET_STRICT_MODE"))
        || env_truthy("BITNET_STRICT_CUDA_BACKEND")
}

/// Runs I2_S QK256 forward pass for input tensor shapes [B, T, H] or [B, H].
pub fn forward_qk256(input: &Tensor, qk256_tensor: &Tensor, weight_name: &str) -> Result<Tensor> {
    BITNET_LINEAR_TOTAL.fetch_add(1, Ordering::Relaxed);

    if cuda_bitnet_backend_requested() {
        #[cfg(feature = "cuda")]
        {
            match forward_qk256_cuda(input, qk256_tensor, weight_name) {
                Ok(output) => {
                    BITNET_LINEAR_ON_CUDA.fetch_add(1, Ordering::Relaxed);
                    return Ok(output);
                }
                Err(err) if strict_cuda_bitnet_backend_requested() => {
                    return Err(BitNetError::Validation(format!(
                        "strict CUDA BitNet linear dispatch failed for {weight_name}: {err}"
                    )));
                }
                Err(err) => {
                    BITNET_LINEAR_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(
                        "CUDA QK256 dispatch failed for {}; using CPU fallback: {}",
                        weight_name,
                        err
                    );
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            if strict_cuda_bitnet_backend_requested() {
                return Err(BitNetError::Validation(format!(
                    "strict CUDA BitNet linear dispatch requested for {weight_name}, but bitnet-qk256-dispatch was built without the cuda feature"
                )));
            }
            BITNET_LINEAR_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
        }
    }

    forward_qk256_cpu(input, qk256_tensor, weight_name)
}

fn forward_qk256_cpu(input: &Tensor, qk256_tensor: &Tensor, weight_name: &str) -> Result<Tensor> {
    use bitnet_quantization::i2s_qk256::gemv_qk256;

    let prepared = prepare_qk256_forward(input, qk256_tensor, weight_name)?;
    let mut output_rows = vec![
        vec![0.0f32; prepared.layout.rows];
        prepared.shape.batch_size * prepared.shape.seq_len
    ];

    if std::env::var("BITNET_TRACE_RMS").as_deref() == Ok("1") && weight_name.contains("layers.0.")
    {
        static DIM_LOGGED: std::sync::Once = std::sync::Once::new();
        DIM_LOGGED.call_once(|| {
            eprintln!(
                "trace_qk256: weight={} rows={} cols={} row_stride_bytes={} qk256_shape={:?}",
                weight_name,
                prepared.layout.rows,
                prepared.layout.cols,
                prepared.layout.row_stride_bytes,
                qk256_tensor.dims()
            );
        });
    }

    for (row_index, input_row) in prepared.input_rows.iter().enumerate() {
        gemv_qk256(
            &prepared.flat_bytes,
            input_row,
            &mut output_rows[row_index],
            prepared.layout.rows,
            prepared.layout.cols,
            prepared.layout.row_stride_bytes,
        )
        .map_err(|e| {
            BitNetError::Validation(format!(
                "QK256 GEMV failed for {} at row {}: {}",
                weight_name, row_index, e
            ))
        })?;
    }

    tensor_from_flat_output(
        output_rows.into_iter().flatten().collect(),
        &prepared.shape,
        &prepared.layout,
        input,
    )
}

#[cfg(feature = "cuda")]
fn forward_qk256_cuda(input: &Tensor, qk256_tensor: &Tensor, weight_name: &str) -> Result<Tensor> {
    let prepared = prepare_qk256_forward(input, qk256_tensor, weight_name)?;
    let mut input_flat = Vec::with_capacity(prepared.input_rows.len() * prepared.layout.cols);
    for row in &prepared.input_rows {
        input_flat.extend_from_slice(row);
    }
    let mut output_flat =
        vec![0.0f32; prepared.shape.batch_size * prepared.shape.seq_len * prepared.layout.rows];
    let config = bitnet_kernels::cuda::Qk256GemvConfig::for_shape(
        prepared.shape.batch_size * prepared.shape.seq_len,
        prepared.layout.rows,
        prepared.layout.cols,
    )
    .map_err(BitNetError::from)?;

    bitnet_kernels::cuda::launch_qk256_gemv(
        &prepared.flat_bytes,
        &[],
        &input_flat,
        &mut output_flat,
        &config,
    )
    .map_err(BitNetError::from)?;

    tensor_from_flat_output(output_flat, &prepared.shape, &prepared.layout, input)
}

struct PreparedQk256Forward {
    layout: Qk256Layout,
    shape: Qk256InputShape,
    flat_bytes: Vec<u8>,
    input_rows: Vec<Vec<f32>>,
}

fn prepare_qk256_forward(
    input: &Tensor,
    qk256_tensor: &Tensor,
    weight_name: &str,
) -> Result<PreparedQk256Forward> {
    let qk256_dims = qk256_tensor.dims();
    let layout = parse_qk256_layout(weight_name, qk256_dims)
        .map_err(|e| BitNetError::Validation(e.to_string()))?;

    debug_assert!(
        layout.row_stride_bytes.is_multiple_of(64),
        "QK256 row_stride_bytes must be multiple of 64"
    );

    let bytes_2d = qk256_tensor.to_vec2::<u8>().map_err(|e| {
        BitNetError::Validation(format!("Failed to extract QK256 bytes for {}: {}", weight_name, e))
    })?;
    let mut flat_bytes = Vec::with_capacity(layout.rows * layout.row_stride_bytes);
    for row in bytes_2d {
        flat_bytes.extend_from_slice(&row);
    }

    let shape =
        parse_input_shape(input.dims()).map_err(|e| BitNetError::Validation(e.to_string()))?;

    validate_input_cols(weight_name, shape.cols, layout.cols)
        .map_err(|e| BitNetError::Validation(e.to_string()))?;

    let input_flat = input.reshape(&[shape.batch_size * shape.seq_len, layout.cols])?;
    let input_rows = input_flat.to_vec2::<f32>().map_err(|e| {
        BitNetError::Validation(format!(
            "Failed to convert input to f32 for {}: {}",
            weight_name, e
        ))
    })?;

    Ok(PreparedQk256Forward { layout, shape, flat_bytes, input_rows })
}

fn tensor_from_flat_output(
    output_flat: Vec<f32>,
    shape: &Qk256InputShape,
    layout: &Qk256Layout,
    input: &Tensor,
) -> Result<Tensor> {
    let output_tensor = if shape.input_rank == 3 {
        Tensor::from_vec(
            output_flat,
            (shape.batch_size, shape.seq_len, layout.rows),
            input.device(),
        )?
    } else {
        Tensor::from_vec(output_flat, (shape.batch_size, layout.rows), input.device())?
    };

    Ok(output_tensor)
}

fn backend_env_matches(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        matches!(value.to_ascii_lowercase().as_str(), "nvidia-rtx-5070-ti-cuda" | "cuda")
    })
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}
