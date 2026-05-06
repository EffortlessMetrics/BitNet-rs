use thiserror::Error;

pub type Result<T> = std::result::Result<T, Qk256LayoutError>;

/// Number of matrix columns encoded by one QK256 block.
pub const QK256_BLOCK_COLS: usize = 256;

/// Number of bits used by one packed QK256 code.
pub const QK256_BITS_PER_CODE: usize = 2;

/// Number of packed bytes in one QK256 block.
pub const QK256_PACKED_BYTES_PER_BLOCK: usize = QK256_BLOCK_COLS * QK256_BITS_PER_CODE / 8;

/// QK256 rows are stored as whole packed blocks, so row strides are block aligned.
pub const QK256_ROW_ALIGNMENT_BYTES: usize = QK256_PACKED_BYTES_PER_BLOCK;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qk256Layout {
    pub rows: usize,
    pub row_stride_bytes: usize,
    pub cols: usize,
    pub blocks_per_row: usize,
    pub packed_len_bytes: usize,
}

impl Qk256Layout {
    pub fn from_rows_cols(rows: usize, cols: usize) -> Result<Self> {
        let blocks_per_row = qk256_blocks_per_row(cols);
        let row_stride_bytes = blocks_per_row
            .checked_mul(QK256_PACKED_BYTES_PER_BLOCK)
            .ok_or(Qk256LayoutError::PackedLengthOverflow { rows, cols })?;
        let packed_len_bytes = rows
            .checked_mul(row_stride_bytes)
            .ok_or(Qk256LayoutError::PackedLengthOverflow { rows, cols })?;

        Ok(Self { rows, row_stride_bytes, cols, blocks_per_row, packed_len_bytes })
    }

    pub fn from_rows_stride(rows: usize, row_stride_bytes: usize) -> Result<Self> {
        if !row_stride_bytes.is_multiple_of(QK256_ROW_ALIGNMENT_BYTES) {
            return Err(Qk256LayoutError::InvalidRowStride { row_stride_bytes });
        }

        let blocks_per_row = row_stride_bytes / QK256_PACKED_BYTES_PER_BLOCK;
        let cols = blocks_per_row
            .checked_mul(QK256_BLOCK_COLS)
            .ok_or(Qk256LayoutError::RowStrideOverflow { row_stride_bytes })?;
        let packed_len_bytes = rows
            .checked_mul(row_stride_bytes)
            .ok_or(Qk256LayoutError::PackedLengthOverflow { rows, cols })?;

        Ok(Self { rows, row_stride_bytes, cols, blocks_per_row, packed_len_bytes })
    }

    pub fn validate_packed_len(&self, actual_len: usize) -> Result<()> {
        if actual_len != self.packed_len_bytes {
            return Err(Qk256LayoutError::PackedLengthMismatch {
                rows: self.rows,
                cols: self.cols,
                actual_len,
                expected_len: self.packed_len_bytes,
            });
        }

        Ok(())
    }

    pub fn row_range(&self, row: usize) -> Result<std::ops::Range<usize>> {
        if row >= self.rows {
            return Err(Qk256LayoutError::RowOutOfBounds { row, rows: self.rows });
        }

        let start = row * self.row_stride_bytes;
        Ok(start..start + self.row_stride_bytes)
    }

    pub fn block_range(&self, row: usize, block: usize) -> Result<std::ops::Range<usize>> {
        if block >= self.blocks_per_row {
            return Err(Qk256LayoutError::BlockOutOfBounds {
                block,
                blocks_per_row: self.blocks_per_row,
            });
        }

        let row_start = self.row_range(row)?.start;
        let start = row_start + block * QK256_PACKED_BYTES_PER_BLOCK;
        Ok(start..start + QK256_PACKED_BYTES_PER_BLOCK)
    }

    pub fn row_ranges(&self) -> impl ExactSizeIterator<Item = std::ops::Range<usize>> + '_ {
        (0..self.rows).map(|row| {
            let start = row * self.row_stride_bytes;
            start..start + self.row_stride_bytes
        })
    }
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

    #[error("QK256: invalid row_stride_bytes {row_stride_bytes}; expected a multiple of 64")]
    InvalidRowStride { row_stride_bytes: usize },

    #[error("QK256: packed length overflow computing rows={rows}, cols={cols}")]
    PackedLengthOverflow { rows: usize, cols: usize },

    #[error(
        "QK256 packed length mismatch for rows={rows}, cols={cols}: got {actual_len}, expected {expected_len}"
    )]
    PackedLengthMismatch { rows: usize, cols: usize, actual_len: usize, expected_len: usize },

    #[error("QK256 row index {row} is out of bounds for rows={rows}")]
    RowOutOfBounds { row: usize, rows: usize },

    #[error("QK256 block index {block} is out of bounds for blocks_per_row={blocks_per_row}")]
    BlockOutOfBounds { block: usize, blocks_per_row: usize },

    #[error("QK256 code at offset {offset} has invalid value {code}; expected 0..=3")]
    InvalidCode { offset: usize, code: u8 },

    #[error("Unsupported input shape for QK256: {dims:?}")]
    UnsupportedInputShape { dims: Vec<usize> },

    #[error(
        "QK256 dimension mismatch for {weight_name}: input has {input_cols} cols but QK256 tensor expects {expected_cols} cols"
    )]
    DimensionMismatch { weight_name: String, input_cols: usize, expected_cols: usize },
}

pub fn qk256_blocks_per_row(cols: usize) -> usize {
    cols.div_ceil(QK256_BLOCK_COLS)
}

pub fn qk256_row_stride_bytes(cols: usize) -> Result<usize> {
    qk256_blocks_per_row(cols)
        .checked_mul(QK256_PACKED_BYTES_PER_BLOCK)
        .ok_or(Qk256LayoutError::PackedLengthOverflow { rows: 1, cols })
}

pub fn qk256_packed_len_bytes(rows: usize, cols: usize) -> Result<usize> {
    Qk256Layout::from_rows_cols(rows, cols).map(|layout| layout.packed_len_bytes)
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
    Qk256Layout::from_rows_stride(rows, row_stride_bytes)
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

pub fn pack_qk256_codes(
    codes: &[u8; QK256_BLOCK_COLS],
) -> Result<[u8; QK256_PACKED_BYTES_PER_BLOCK]> {
    let mut packed = [0u8; QK256_PACKED_BYTES_PER_BLOCK];
    for (offset, code) in codes.iter().copied().enumerate() {
        if code > 3 {
            return Err(Qk256LayoutError::InvalidCode { offset, code });
        }
        packed[offset / 4] |= code << ((offset % 4) * 2);
    }

    Ok(packed)
}

pub fn unpack_qk256_codes(packed: &[u8; QK256_PACKED_BYTES_PER_BLOCK]) -> [u8; QK256_BLOCK_COLS] {
    let mut codes = [0u8; QK256_BLOCK_COLS];
    for (offset, code) in codes.iter_mut().enumerate() {
        *code = (packed[offset / 4] >> ((offset % 4) * 2)) & 0b11;
    }
    codes
}
