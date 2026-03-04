use thiserror::Error;

pub type Result<T> = std::result::Result<T, Qk256LayoutError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qk256Layout {
    pub rows: usize,
    pub row_stride_bytes: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qk256InputShape {
    pub batch_size: usize,
    pub seq_len: usize,
    pub cols: usize,
    pub input_rank: usize,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Qk256LayoutError {
    #[error("QK256 tensor {weight_name} has invalid shape: {dims:?}")]
    InvalidQk256Shape { weight_name: String, dims: Vec<usize> },

    #[error("QK256: row_stride_bytes overflow computing cols (row_stride={row_stride_bytes})")]
    RowStrideOverflow { row_stride_bytes: usize },

    #[error("Unsupported input shape for QK256: {dims:?}")]
    UnsupportedInputShape { dims: Vec<usize> },

    #[error(
        "QK256 dimension mismatch for {weight_name}: input has {input_cols} cols but QK256 tensor expects {expected_cols} cols"
    )]
    DimensionMismatch { weight_name: String, input_cols: usize, expected_cols: usize },
}

pub fn parse_qk256_layout(weight_name: &str, qk256_dims: &[usize]) -> Result<Qk256Layout> {
    if qk256_dims.len() != 2 {
        return Err(Qk256LayoutError::InvalidQk256Shape {
            weight_name: weight_name.to_owned(),
            dims: qk256_dims.to_vec(),
        });
    }

    let rows = qk256_dims[0];
    let row_stride_bytes = qk256_dims[1];
    let cols = row_stride_bytes
        .checked_mul(4)
        .ok_or(Qk256LayoutError::RowStrideOverflow { row_stride_bytes })?;

    Ok(Qk256Layout { rows, row_stride_bytes, cols })
}

pub fn parse_input_shape(input_dims: &[usize]) -> Result<Qk256InputShape> {
    let (batch_size, seq_len, cols) = match input_dims {
        [batch_size, seq_len, cols] => (*batch_size, *seq_len, *cols),
        [batch_size, cols] => (*batch_size, 1, *cols),
        _ => {
            return Err(Qk256LayoutError::UnsupportedInputShape { dims: input_dims.to_vec() });
        }
    };

    Ok(Qk256InputShape { batch_size, seq_len, cols, input_rank: input_dims.len() })
}

pub fn validate_input_cols(
    weight_name: &str,
    input_cols: usize,
    expected_cols: usize,
) -> Result<()> {
    if input_cols != expected_cols {
        return Err(Qk256LayoutError::DimensionMismatch {
            weight_name: weight_name.to_owned(),
            input_cols,
            expected_cols,
        });
    }

    Ok(())
}
