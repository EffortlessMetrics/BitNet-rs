use bitnet_common::{BitNetError, Result};
use candle_core::Tensor;

/// Runs I2_S QK256 forward pass for input tensor shapes [B, T, H] or [B, H].
pub fn forward_qk256(input: &Tensor, qk256_tensor: &Tensor, weight_name: &str) -> Result<Tensor> {
    use bitnet_quantization::i2s_qk256::gemv_qk256;

    let dims = qk256_tensor.dims();
    if dims.len() != 2 {
        return Err(BitNetError::Validation(format!(
            "QK256 tensor {} has invalid shape: {:?}",
            weight_name, dims
        )));
    }

    let rows = dims[0];
    let row_stride_bytes = dims[1];

    debug_assert!(
        row_stride_bytes.is_multiple_of(64),
        "QK256 row_stride_bytes must be multiple of 64"
    );
    let cols = row_stride_bytes.checked_mul(4).ok_or_else(|| {
        BitNetError::Validation(format!(
            "QK256: row_stride_bytes overflow computing cols (row_stride={})",
            row_stride_bytes
        ))
    })?;

    let bytes_2d = qk256_tensor.to_vec2::<u8>().map_err(|e| {
        BitNetError::Validation(format!("Failed to extract QK256 bytes for {}: {}", weight_name, e))
    })?;
    let mut flat_bytes = Vec::with_capacity(rows * row_stride_bytes);
    for row in bytes_2d {
        flat_bytes.extend_from_slice(&row);
    }

    let input_dims = input.dims();
    let rank = input_dims.len();
    let (batch_size, seq_len, input_cols) = match rank {
        3 => (input_dims[0], input_dims[1], input_dims[2]),
        2 => (input_dims[0], 1, input_dims[1]),
        _ => {
            return Err(BitNetError::Validation(format!(
                "Unsupported input shape for QK256: {:?}",
                input_dims
            )));
        }
    };

    if input_cols != cols {
        return Err(BitNetError::Validation(format!(
            "QK256 dimension mismatch for {}: input has {} cols but QK256 tensor expects {} cols",
            weight_name, input_cols, cols
        )));
    }

    let input_flat = input.reshape(&[batch_size * seq_len, cols])?;
    let input_vec = input_flat.to_vec2::<f32>().map_err(|e| {
        BitNetError::Validation(format!(
            "Failed to convert input to f32 for {}: {}",
            weight_name, e
        ))
    })?;

    let mut output_vec = vec![vec![0.0f32; rows]; batch_size * seq_len];

    if std::env::var("BITNET_TRACE_RMS").as_deref() == Ok("1") && weight_name.contains("layers.0.")
    {
        static DIM_LOGGED: std::sync::Once = std::sync::Once::new();
        DIM_LOGGED.call_once(|| {
            eprintln!(
                "trace_qk256: weight={} rows={} cols={} row_stride_bytes={} qk256_shape={:?}",
                weight_name, rows, cols, row_stride_bytes, dims
            );
        });
    }

    for (i, input_row) in input_vec.iter().enumerate() {
        gemv_qk256(&flat_bytes, input_row, &mut output_vec[i], rows, cols, row_stride_bytes)
            .map_err(|e| {
                BitNetError::Validation(format!(
                    "QK256 GEMV failed for {} at row {}: {}",
                    weight_name, i, e
                ))
            })?;
    }

    let output_flat: Vec<f32> = output_vec.into_iter().flatten().collect();
    let output_tensor = if rank == 3 {
        Tensor::from_vec(output_flat, (batch_size, seq_len, rows), input.device())?
    } else {
        Tensor::from_vec(output_flat, (batch_size, rows), input.device())?
    };

    Ok(output_tensor)
}
