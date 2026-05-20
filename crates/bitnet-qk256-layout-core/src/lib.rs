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

#[cfg(test)]
mod tests {
    use super::*;

    fn patterned_codes() -> [u8; QK256_BLOCK_COLS] {
        let mut codes = [0u8; QK256_BLOCK_COLS];
        for (offset, code) in codes.iter_mut().enumerate() {
            *code = (offset % 4) as u8;
        }
        codes
    }

    fn layout_2x257() -> Qk256Layout {
        Qk256Layout {
            rows: 2,
            row_stride_bytes: 128,
            cols: 257,
            blocks_per_row: 2,
            packed_len_bytes: 256,
        }
    }

    fn layout_3x257() -> Qk256Layout {
        Qk256Layout {
            rows: 3,
            row_stride_bytes: 128,
            cols: 257,
            blocks_per_row: 2,
            packed_len_bytes: 384,
        }
    }

    #[test]
    fn constants_match_qk256_block_geometry() {
        assert_eq!(QK256_BLOCK_COLS, 256);
        assert_eq!(QK256_BITS_PER_CODE, 2);
        assert_eq!(QK256_PACKED_BYTES_PER_BLOCK, 64);
        assert_eq!(QK256_ROW_ALIGNMENT_BYTES, QK256_PACKED_BYTES_PER_BLOCK);
    }

    #[test]
    fn blocks_per_row_rounds_up_to_whole_qk256_blocks() {
        assert_eq!(qk256_blocks_per_row(0), 0);
        assert_eq!(qk256_blocks_per_row(1), 1);
        assert_eq!(qk256_blocks_per_row(255), 1);
        assert_eq!(qk256_blocks_per_row(256), 1);
        assert_eq!(qk256_blocks_per_row(257), 2);
        assert_eq!(qk256_blocks_per_row(512), 2);
    }

    #[test]
    fn row_stride_bytes_uses_packed_block_size() {
        assert_eq!(qk256_row_stride_bytes(0), Ok(0));
        assert_eq!(qk256_row_stride_bytes(1), Ok(64));
        assert_eq!(qk256_row_stride_bytes(256), Ok(64));
        assert_eq!(qk256_row_stride_bytes(257), Ok(128));
    }

    #[test]
    fn layout_from_rows_cols_records_derived_sizes() {
        assert_eq!(Qk256Layout::from_rows_cols(3, 257), Ok(layout_3x257()));
        assert_eq!(qk256_packed_len_bytes(3, 257), Ok(384));
    }

    #[test]
    fn layout_from_rows_cols_allows_zero_sized_layouts() {
        assert_eq!(
            Qk256Layout::from_rows_cols(0, 257),
            Ok(Qk256Layout {
                rows: 0,
                row_stride_bytes: 128,
                cols: 257,
                blocks_per_row: 2,
                packed_len_bytes: 0,
            })
        );
        assert_eq!(
            Qk256Layout::from_rows_cols(3, 0),
            Ok(Qk256Layout {
                rows: 3,
                row_stride_bytes: 0,
                cols: 0,
                blocks_per_row: 0,
                packed_len_bytes: 0,
            })
        );
    }

    #[test]
    fn layout_from_rows_cols_reports_packed_length_overflow() {
        assert_eq!(
            Qk256Layout::from_rows_cols(usize::MAX, 256),
            Err(Qk256LayoutError::PackedLengthOverflow { rows: usize::MAX, cols: 256 })
        );
    }

    #[test]
    fn layout_from_rows_stride_converts_aligned_stride_to_cols() {
        assert_eq!(
            Qk256Layout::from_rows_stride(2, 128),
            Ok(Qk256Layout {
                rows: 2,
                row_stride_bytes: 128,
                cols: 512,
                blocks_per_row: 2,
                packed_len_bytes: 256,
            })
        );
    }

    #[test]
    fn layout_from_rows_stride_rejects_unaligned_stride() {
        assert_eq!(
            Qk256Layout::from_rows_stride(1, 63),
            Err(Qk256LayoutError::InvalidRowStride { row_stride_bytes: 63 })
        );
        assert_eq!(
            Qk256Layout::from_rows_stride(1, 65),
            Err(Qk256LayoutError::InvalidRowStride { row_stride_bytes: 65 })
        );
    }

    #[test]
    fn layout_from_rows_stride_reports_cols_overflow() {
        let row_stride_bytes =
            (usize::MAX / QK256_PACKED_BYTES_PER_BLOCK) * QK256_PACKED_BYTES_PER_BLOCK;

        assert_eq!(
            Qk256Layout::from_rows_stride(1, row_stride_bytes),
            Err(Qk256LayoutError::RowStrideOverflow { row_stride_bytes })
        );
    }

    #[test]
    fn layout_from_rows_stride_reports_packed_length_overflow() {
        assert_eq!(
            Qk256Layout::from_rows_stride(usize::MAX, 64),
            Err(Qk256LayoutError::PackedLengthOverflow { rows: usize::MAX, cols: 256 })
        );
    }

    #[test]
    fn validate_packed_len_accepts_exact_length() {
        let layout = layout_2x257();
        assert_eq!(layout.validate_packed_len(256), Ok(()));
    }

    #[test]
    fn validate_packed_len_reports_mismatch_details() {
        let layout = layout_2x257();

        assert_eq!(
            layout.validate_packed_len(255),
            Err(Qk256LayoutError::PackedLengthMismatch {
                rows: 2,
                cols: 257,
                actual_len: 255,
                expected_len: 256,
            })
        );
    }

    #[test]
    fn row_range_returns_stride_sized_ranges() {
        let layout = layout_3x257();

        assert_eq!(layout.row_range(0), Ok(0..128));
        assert_eq!(layout.row_range(1), Ok(128..256));
        assert_eq!(layout.row_range(2), Ok(256..384));
    }

    #[test]
    fn row_range_rejects_out_of_bounds_rows() {
        let layout = layout_3x257();

        assert_eq!(layout.row_range(3), Err(Qk256LayoutError::RowOutOfBounds { row: 3, rows: 3 }));
    }

    #[test]
    fn block_range_returns_block_sized_ranges_within_row() {
        let layout = layout_2x257();

        assert_eq!(layout.block_range(0, 0), Ok(0..64));
        assert_eq!(layout.block_range(0, 1), Ok(64..128));
        assert_eq!(layout.block_range(1, 0), Ok(128..192));
        assert_eq!(layout.block_range(1, 1), Ok(192..256));
    }

    #[test]
    fn block_range_rejects_out_of_bounds_blocks_before_rows() {
        let layout = layout_2x257();

        assert_eq!(
            layout.block_range(99, 2),
            Err(Qk256LayoutError::BlockOutOfBounds { block: 2, blocks_per_row: 2 })
        );
    }

    #[test]
    fn block_range_propagates_row_bounds_for_valid_block() {
        let layout = layout_2x257();

        assert_eq!(
            layout.block_range(2, 1),
            Err(Qk256LayoutError::RowOutOfBounds { row: 2, rows: 2 })
        );
    }

    #[test]
    fn row_ranges_iterates_all_rows_with_exact_size() {
        let layout = layout_3x257();
        let mut ranges = layout.row_ranges();

        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges.next(), Some(0..128));
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges.collect::<Vec<_>>(), vec![128..256, 256..384]);
    }

    #[test]
    fn parse_qk256_layout_accepts_two_dimensional_layout_shape() {
        assert_eq!(
            parse_qk256_layout("blk.0.weight", &[2, 64]),
            Ok(Qk256Layout {
                rows: 2,
                row_stride_bytes: 64,
                cols: 256,
                blocks_per_row: 1,
                packed_len_bytes: 128,
            })
        );
    }

    #[test]
    fn parse_qk256_layout_rejects_non_matrix_shapes() {
        assert_eq!(
            parse_qk256_layout("blk.0.weight", &[2, 64, 1]),
            Err(Qk256LayoutError::InvalidQk256Shape {
                weight_name: "blk.0.weight".to_owned(),
                dims: vec![2, 64, 1],
            })
        );
    }

    #[test]
    fn parse_input_shape_accepts_rank_two_as_single_token_sequence() {
        assert_eq!(
            parse_input_shape(&[4, 1024]),
            Ok(Qk256InputShape { batch_size: 4, seq_len: 1, cols: 1024, input_rank: 2 })
        );
    }

    #[test]
    fn parse_input_shape_accepts_rank_three_sequence_batches() {
        assert_eq!(
            parse_input_shape(&[4, 8, 1024]),
            Ok(Qk256InputShape { batch_size: 4, seq_len: 8, cols: 1024, input_rank: 3 })
        );
    }

    #[test]
    fn parse_input_shape_rejects_unsupported_ranks() {
        assert_eq!(
            parse_input_shape(&[1024]),
            Err(Qk256LayoutError::UnsupportedInputShape { dims: vec![1024] })
        );
        assert_eq!(
            parse_input_shape(&[1, 2, 3, 4]),
            Err(Qk256LayoutError::UnsupportedInputShape { dims: vec![1, 2, 3, 4] })
        );
    }

    #[test]
    fn validate_input_cols_accepts_exact_match() {
        assert_eq!(validate_input_cols("blk.0.weight", 1024, 1024), Ok(()));
    }

    #[test]
    fn validate_input_cols_reports_mismatch() {
        assert_eq!(
            validate_input_cols("blk.0.weight", 512, 1024),
            Err(Qk256LayoutError::DimensionMismatch {
                weight_name: "blk.0.weight".to_owned(),
                input_cols: 512,
                expected_cols: 1024,
            })
        );
    }

    #[test]
    fn pack_qk256_codes_packs_four_two_bit_codes_per_byte_little_endian() {
        assert_eq!(
            pack_qk256_codes(&patterned_codes()),
            Ok([0b11_10_01_00; QK256_PACKED_BYTES_PER_BLOCK])
        );
    }

    #[test]
    fn pack_qk256_codes_rejects_codes_larger_than_two_bits() {
        let mut codes = patterned_codes();
        codes[17] = 4;

        assert_eq!(
            pack_qk256_codes(&codes),
            Err(Qk256LayoutError::InvalidCode { offset: 17, code: 4 })
        );
    }

    #[test]
    fn unpack_qk256_codes_extracts_four_two_bit_codes_per_byte_little_endian() {
        let packed = [0b11_10_01_00; QK256_PACKED_BYTES_PER_BLOCK];

        assert_eq!(unpack_qk256_codes(&packed), patterned_codes());
    }

    #[test]
    fn pack_and_unpack_round_trip_for_non_repeating_codes() {
        let mut codes = [0u8; QK256_BLOCK_COLS];
        for (offset, code) in codes.iter_mut().enumerate() {
            *code = ((offset * 17 + offset / 7) % 4) as u8;
        }

        assert_eq!(pack_qk256_codes(&codes).map(|packed| unpack_qk256_codes(&packed)), Ok(codes));
    }
}
