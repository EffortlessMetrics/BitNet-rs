use bitnet_common::{BitNetError, Result};
use bitnet_qk256_layout_core::{parse_input_shape, parse_qk256_layout, validate_input_cols};
use candle_core::Tensor;

/// Runs I2_S QK256 forward pass for input tensor shapes [B, T, H] or [B, H].
pub fn forward_qk256(input: &Tensor, qk256_tensor: &Tensor, weight_name: &str) -> Result<Tensor> {
    use bitnet_quantization::i2s_qk256::gemv_qk256;

    let qk256_dims = qk256_tensor.dims();
    let layout = parse_qk256_layout(weight_name, &qk256_dims)
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
        parse_input_shape(&input_dims).map_err(|e| BitNetError::Validation(e.to_string()))?;

    validate_input_cols(weight_name, shape.cols, layout.cols)
        .map_err(|e| BitNetError::Validation(e.to_string()))?;

    let input_flat = input.reshape(&[shape.batch_size * shape.seq_len, layout.cols])?;
    let input_vec = input_flat.to_vec2::<f32>().map_err(|e| {
        BitNetError::Validation(format!(
            "Failed to convert input to f32 for {}: {}",
            weight_name, e
        ))
    })?;

    let mut output_vec = vec![vec![0.0f32; layout.rows]; shape.batch_size * shape.seq_len];

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

    for (i, input_row) in input_vec.iter().enumerate() {
        gemv_qk256(
            &flat_bytes,
            input_row,
            &mut output_vec[i],
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

    let output_flat: Vec<f32> = output_vec.into_iter().flatten().collect();
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
