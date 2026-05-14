#[cfg(feature = "opencl")]
mod opencl;

#[cfg(feature = "opencl")]
pub use opencl::{QK256_OPENCL_KERNEL_NAME, QK256_OPENCL_KERNEL_SRC, gemm_qk256_opencl};

use bitnet_common::{BitNetError, KernelError, Result};
use bitnet_qk256_layout_core::{parse_input_shape, parse_qk256_layout, validate_input_cols};
use candle_core::Tensor;

const NOT_CLAIMED_OPENCL_QK256: &[&str] =
    &["a770_qk256_opencl_execution", "a770_qk256_opencl_performance", "a770_full_device_residency"];

/// Describes which QK256 runtime is currently used by this dispatch crate.
///
/// OpenCL/oneAPI features currently compile the route dependencies only. The actual
/// GGML QK256 no-scale GEMV remains the CPU implementation until a format-correct
/// OpenCL QK256 runtime is wired here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qk256DispatchStatus {
    pub compiled_opencl: bool,
    pub compiled_oneapi: bool,
    pub opencl_launcher_available: bool,
    pub runtime_backend: &'static str,
    pub accelerator_claimable: bool,
    pub blocker: Option<&'static str>,
    pub not_claims: &'static [&'static str],
}

/// Returns the non-promoting QK256 dispatch status for proof receipts.
pub fn qk256_dispatch_status() -> Qk256DispatchStatus {
    let compiled_opencl = cfg!(feature = "opencl");
    let compiled_oneapi = cfg!(feature = "oneapi");
    let blocker = if compiled_oneapi {
        Some("oneapi_qk256_transformer_dispatch_not_wired")
    } else if compiled_opencl {
        Some("opencl_qk256_transformer_dispatch_not_wired")
    } else {
        Some("cpu_qk256_dispatch_only")
    };

    Qk256DispatchStatus {
        compiled_opencl,
        compiled_oneapi,
        opencl_launcher_available: cfg!(feature = "opencl"),
        runtime_backend: "cpu_qk256_reference",
        accelerator_claimable: false,
        blocker,
        not_claims: NOT_CLAIMED_OPENCL_QK256,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qk256DispatchBackend {
    Cpu,
    OpenCl,
}

/// Runs I2_S QK256 forward pass for input tensor shapes [B, T, H] or [B, H].
pub fn forward_qk256(input: &Tensor, qk256_tensor: &Tensor, weight_name: &str) -> Result<Tensor> {
    forward_qk256_with_backend(input, qk256_tensor, weight_name, Qk256DispatchBackend::Cpu)
}

/// Runs I2_S QK256 forward pass using an explicit dispatch backend.
pub fn forward_qk256_with_backend(
    input: &Tensor,
    qk256_tensor: &Tensor,
    weight_name: &str,
    backend: Qk256DispatchBackend,
) -> Result<Tensor> {
    use bitnet_quantization::i2s_qk256::gemv_qk256;

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

    let input_dims = input.dims();
    let shape =
        parse_input_shape(input_dims).map_err(|e| BitNetError::Validation(e.to_string()))?;

    validate_input_cols(weight_name, shape.cols, layout.cols)
        .map_err(|e| BitNetError::Validation(e.to_string()))?;

    let input_flat = input.reshape(&[shape.batch_size * shape.seq_len, layout.cols])?;
    let input_vec = input_flat.to_vec2::<f32>().map_err(|e| {
        BitNetError::Validation(format!(
            "Failed to convert input to f32 for {}: {}",
            weight_name, e
        ))
    })?;

    let input_rows = shape.batch_size * shape.seq_len;
    let mut output_flat = vec![0.0f32; input_rows * layout.rows];

    if std::env::var("BITNET_TRACE_RMS").as_deref() == Ok("1") && weight_name.contains("layers.0.")
    {
        static DIM_LOGGED: std::sync::Once = std::sync::Once::new();
        DIM_LOGGED.call_once(|| {
            eprintln!(
                "trace_qk256: weight={} rows={} cols={} row_stride_bytes={} qk256_shape={:?}",
                weight_name, layout.rows, layout.cols, layout.row_stride_bytes, qk256_dims
            );
        });
    }

    match backend {
        Qk256DispatchBackend::Cpu => {
            for (i, input_row) in input_vec.iter().enumerate() {
                let start = i * layout.rows;
                let end = start + layout.rows;
                gemv_qk256(
                    &flat_bytes,
                    input_row,
                    &mut output_flat[start..end],
                    layout.rows,
                    layout.cols,
                    layout.row_stride_bytes,
                )
                .map_err(|e| {
                    BitNetError::Validation(format!(
                        "QK256 GEMV failed for {} at row {}: {}",
                        weight_name, i, e
                    ))
                })?;
            }
        }
        Qk256DispatchBackend::OpenCl => {
            run_opencl_qk256(
                &flat_bytes,
                &input_vec,
                &mut output_flat,
                layout.rows,
                layout.cols,
                layout.row_stride_bytes,
                weight_name,
            )?;
        }
    }

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

fn run_opencl_qk256(
    flat_bytes: &[u8],
    input_vec: &[Vec<f32>],
    output_flat: &mut [f32],
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
    weight_name: &str,
) -> Result<()> {
    #[cfg(feature = "opencl")]
    {
        let input_flat: Vec<f32> = input_vec.iter().flat_map(|row| row.iter().copied()).collect();
        opencl::gemm_qk256_opencl(
            flat_bytes,
            &input_flat,
            output_flat,
            input_vec.len(),
            rows,
            cols,
            row_stride_bytes,
        )
        .map_err(|e| {
            BitNetError::Kernel(KernelError::GpuError {
                reason: format!("OpenCL QK256 GEMV failed for {weight_name}: {e}"),
            })
        })
    }

    #[cfg(not(feature = "opencl"))]
    {
        let _ = (flat_bytes, input_vec, output_flat, rows, cols, row_stride_bytes);
        Err(BitNetError::Kernel(KernelError::DeviceUnavailable {
            reason: format!(
                "OpenCL QK256 dispatch requested for {weight_name}, but opencl feature is disabled"
            ),
        }))
    }
}
