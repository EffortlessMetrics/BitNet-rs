use anyhow::{Result, bail};

pub const QK256_BLOCK: usize = 256;
pub const QK256_PACKED_BYTES: usize = 64;

#[derive(Clone, Debug)]
pub struct I2SQk256NoScale {
    pub rows: usize,
    pub cols: usize,
    pub row_stride_bytes: usize,
    pub qs: Vec<u8>,
}

impl I2SQk256NoScale {
    pub fn new(rows: usize, cols: usize, qs: Vec<u8>) -> Result<Self> {
        let blocks_per_row = cols.div_ceil(QK256_BLOCK);
        let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;
        let expected_bytes = rows * row_stride_bytes;

        const TOLERANCE: usize = 128;
        let size_diff = qs.len().abs_diff(expected_bytes);

        if size_diff > TOLERANCE {
            bail!(
                "I2SQk256NoScale: data size mismatch: got {} bytes, expected {} for {}×{} matrix. \
                 Check tensor orientation: QK256 requires [out_dim, in_dim] layout.",
                qs.len(),
                expected_bytes,
                rows,
                cols
            );
        }

        Ok(Self { rows, cols, row_stride_bytes, qs })
    }

    #[inline]
    pub fn row_bytes(&self, row: usize) -> &[u8] {
        debug_assert!(row < self.rows, "I2SQk256NoScale: row {} >= rows {}", row, self.rows);
        let start = row * self.row_stride_bytes;
        let end = start + self.row_stride_bytes;
        &self.qs[start..end]
    }
}

#[inline]
pub fn code_to_f32(code: u8) -> f32 {
    debug_assert!(code < 4, "I2S_QK256: code must be 0..=3, got {}", code);
    const LUT: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];
    LUT[code as usize]
}

#[inline]
pub fn unpack_qk256_block(qs64: &[u8; QK256_PACKED_BYTES], out_codes256: &mut [u8; QK256_BLOCK]) {
    for (i, &b) in qs64.iter().enumerate() {
        let base = i * 4;
        out_codes256[base] = b & 0x03;
        out_codes256[base + 1] = (b >> 2) & 0x03;
        out_codes256[base + 2] = (b >> 4) & 0x03;
        out_codes256[base + 3] = (b >> 6) & 0x03;
    }
}

#[inline]
pub fn gemv_qk256_row(qs_row: &[u8], x: &[f32], cols: usize) -> f32 {
    let blocks_needed = cols.div_ceil(QK256_BLOCK);
    let expected_bytes = blocks_needed * QK256_PACKED_BYTES;

    debug_assert_eq!(
        qs_row.len(),
        expected_bytes,
        "I2S_QK256: row bytes mismatch: got {}, expected {} for {} cols",
        qs_row.len(),
        expected_bytes,
        cols
    );
    debug_assert!(x.len() >= cols, "I2S_QK256: x too short: {} < {}", x.len(), cols);

    let mut acc = 0.0f32;
    let mut codes = [0u8; QK256_BLOCK];

    let mut col = 0usize;
    for blk in qs_row.chunks_exact(QK256_PACKED_BYTES) {
        let blk_arr: &[u8; QK256_PACKED_BYTES] =
            blk.try_into().expect("QK256: block must be 64 bytes");
        unpack_qk256_block(blk_arr, &mut codes);

        let take = QK256_BLOCK.min(cols - col);
        for j in 0..take {
            acc += code_to_f32(codes[j]) * x[col + j];
        }

        col += take;
        if col >= cols {
            break;
        }
    }

    acc
}

pub fn gemv_qk256_scalar(
    qs_data: &[u8],
    x: &[f32],
    y_out: &mut [f32],
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
) -> Result<()> {
    if y_out.len() != rows {
        bail!("I2S_QK256: y_out length {} != rows {}", y_out.len(), rows);
    }
    if x.len() < cols {
        bail!("I2S_QK256: x length {} < cols {}", x.len(), cols);
    }

    let expected_total = rows * row_stride_bytes;
    if qs_data.len() < expected_total {
        bail!("I2S_QK256: data too short: {} < {}", qs_data.len(), expected_total);
    }

    for (row, output) in y_out.iter_mut().enumerate().take(rows) {
        let start = row * row_stride_bytes;
        let end = start + row_stride_bytes;
        *output = gemv_qk256_row(&qs_data[start..end], x, cols);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_values_match_spec() {
        assert_eq!(code_to_f32(0), -2.0);
        assert_eq!(code_to_f32(1), -1.0);
        assert_eq!(code_to_f32(2), 1.0);
        assert_eq!(code_to_f32(3), 2.0);
    }

    #[test]
    fn unpack_block_smoke() {
        let mut qs = [0u8; QK256_PACKED_BYTES];
        for (i, b) in qs.iter_mut().enumerate() {
            *b = 0b_11_10_01_00u8.wrapping_add(i as u8 & 0x03);
        }
        let mut codes = [0u8; QK256_BLOCK];
        unpack_qk256_block(&qs, &mut codes);
        assert_eq!(&codes[..4], &[0, 1, 2, 3]);
    }

    #[test]
    fn gemv_row_with_tail() {
        let cols = 300usize;
        let blocks_needed = cols.div_ceil(QK256_BLOCK);
        let qs_row = vec![0xAAu8; blocks_needed * QK256_PACKED_BYTES];
        let x: Vec<f32> = (0..cols).map(|i| (i % 7) as f32).collect();

        let got = gemv_qk256_row(&qs_row, &x, cols);
        let expected: f32 = x.iter().sum();
        assert!((got - expected).abs() < 1e-3);
    }

    #[test]
    fn gemv_multi_row() {
        let rows = 3usize;
        let cols = 256usize;
        let row_stride_bytes = QK256_PACKED_BYTES;
        let qs_data = vec![0x55u8; rows * row_stride_bytes];
        let x: Vec<f32> = (0..cols).map(|i| i as f32).collect();
        let mut y_out = vec![0.0f32; rows];

        gemv_qk256_scalar(&qs_data, &x, &mut y_out, rows, cols, row_stride_bytes).unwrap();

        let expected: f32 = -x.iter().sum::<f32>();
        for &val in &y_out {
            assert!((val - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn qk256_size_tolerance() {
        let rows = 512usize;
        let cols = 1024usize;
        let exact_size = rows * (cols.div_ceil(QK256_BLOCK) * QK256_PACKED_BYTES);

        assert!(I2SQk256NoScale::new(rows, cols, vec![0u8; exact_size]).is_ok());
        assert!(I2SQk256NoScale::new(rows, cols, vec![0u8; exact_size + 32]).is_ok());
        assert!(I2SQk256NoScale::new(rows, cols, vec![0u8; exact_size + 128]).is_ok());
        assert!(I2SQk256NoScale::new(rows, cols, vec![0u8; exact_size + 129]).is_err());
    }
}
