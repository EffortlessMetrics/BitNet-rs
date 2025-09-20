//! Native Rust implementation of I2_S (BitNet 2-bit signed) dequantization
//!
//! Layout per block:
//! - 256 elements per block
//! - 64 bytes of packed 2-bit values
//! - 2 bytes for f16 scale
//! - Total: 66 bytes per block

use anyhow::{Result, bail};
use half::f16;
use tracing::{debug, warn};

/// I2_S block constants
const I2S_DEFAULT_BLOCK: usize = 256;

fn expected_bytes(rows: usize, cols: usize, block: usize) -> usize {
    let blocks_per_row = cols.div_ceil(block);
    let qbits = block.div_ceil(4); // 2 bits per weight
    (rows * blocks_per_row) * (qbits + 2) // +2 bytes f16 scale
}

fn infer_block_size(bytes: usize, rows: usize, cols: usize) -> Option<usize> {
    // Most common variants
    for b in [256, 128, 64, 32] {
        if expected_bytes(rows, cols, b) == bytes {
            return Some(b);
        }
    }
    None
}

/// Extract (rows, cols) from tensor shape, where last dim is the column count
fn rows_cols(dims: &[usize]) -> Result<(usize, usize)> {
    match dims.len() {
        0 => bail!("I2_S: empty tensor dims"),
        1 => Ok((1, dims[0])),
        _ => {
            let cols = *dims.last().unwrap();
            let rows = dims[..dims.len() - 1].iter().product();
            Ok((rows, cols))
        }
    }
}

/// Unpack 2-bit values from packed bytes
/// Each byte contains 4 values: bits [0:1]=v0, [2:3]=v1, [4:5]=v2, [6:7]=v3
#[inline]
fn unpack_2bit_signed(dst: &mut [i8], src_qbits: &[u8], n: usize) {
    // Process 4 values per byte
    let chunks = n / 4;
    let rem = n % 4;
    for i in 0..chunks {
        let byte = src_qbits[i];
        dst[i * 4] = (byte & 0b11) as i8 - 2;
        dst[i * 4 + 1] = ((byte >> 2) & 0b11) as i8 - 2;
        dst[i * 4 + 2] = ((byte >> 4) & 0b11) as i8 - 2;
        dst[i * 4 + 3] = ((byte >> 6) & 0b11) as i8 - 2;
    }
    // Handle remainder
    if rem > 0 {
        let byte = src_qbits[chunks];
        for j in 0..rem {
            dst[chunks * 4 + j] = ((byte >> (j * 2)) & 0b11) as i8 - 2;
        }
    }
}

/// Dequantize I2_S formatted tensor data to f32
///
/// # Arguments
/// * `bytes` - Raw quantized data in I2_S format
/// * `shape` - Tensor dimensions
///
/// # Returns
/// Dequantized f32 values in row-major order
pub fn dequantize_to_f32(bytes: &[u8], shape: &[usize]) -> Result<Vec<f32>> {
    let (rows, cols) = rows_cols(shape)?;
    let mut block = I2S_DEFAULT_BLOCK;
    let default_expected = expected_bytes(rows, cols, block);

    if bytes.len() != default_expected {
        if let Some(b) = infer_block_size(bytes.len(), rows, cols) {
            // Only warn if the difference is significant (more than one block worth of data)
            let one_block_bytes = expected_bytes(1, I2S_DEFAULT_BLOCK, I2S_DEFAULT_BLOCK);
            let shortfall = default_expected.saturating_sub(bytes.len());
            if shortfall > one_block_bytes {
                warn!(
                    "I2_S: non-default block size detected: {} (default {})",
                    b, I2S_DEFAULT_BLOCK
                );
            } else {
                debug!(
                    "I2_S: non-default block size detected: {} (default {})",
                    b, I2S_DEFAULT_BLOCK
                );
            }
            block = b;
        } else {
            // Partial data fallback: process what we can, zero-fill the rest
            let qbits = block.div_ceil(4);
            let per_block = qbits + 2;
            let available_blocks = bytes.len() / per_block;
            debug!(
                "I2_S: byte length mismatch (got {}, expected {}), processing {} blocks then zero-fill",
                bytes.len(),
                default_expected,
                available_blocks
            );
            return dequantize_partial_blocks(bytes, shape, block, available_blocks);
        }
    }
    dequantize_to_f32_with_block(bytes, shape, block)
}

fn dequantize_to_f32_with_block(bytes: &[u8], shape: &[usize], block: usize) -> Result<Vec<f32>> {
    let (rows, cols) = rows_cols(shape)?;
    let blocks_per_row = cols.div_ceil(block);
    let qbits = block.div_ceil(4);
    let per_block = qbits + 2;

    if bytes.len() != rows * blocks_per_row * per_block {
        bail!("I2_S: internal size mismatch for block={}", block);
    }

    let mut out = vec![0f32; rows * cols];
    let mut off = 0usize;
    let mut scratch = vec![0i8; block];

    for r in 0..rows {
        let row_base = r * cols;
        let mut c = 0usize;
        for _ in 0..blocks_per_row {
            let n = (cols - c).min(block);
            let qbits_slice = &bytes[off..off + n.div_ceil(4)];
            off += qbits_slice.len();
            let scale_bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            off += 2;
            let scale = f16::from_bits(scale_bits).to_f32();

            unpack_2bit_signed(&mut scratch[..n], qbits_slice, n);
            for i in 0..n {
                out[row_base + c + i] = scale * scratch[i] as f32;
            }
            c += n;
        }
    }
    Ok(out)
}

fn dequantize_partial_blocks(
    bytes: &[u8],
    shape: &[usize],
    block: usize,
    available_blocks: usize,
) -> Result<Vec<f32>> {
    let (rows, cols) = rows_cols(shape)?;
    let blocks_per_row = cols.div_ceil(block);
    let qbits = block.div_ceil(4);
    let per_block = qbits + 2;

    let mut out = vec![0f32; rows * cols];
    let mut off = 0usize;
    let mut processed = 0usize;
    let mut scratch = vec![0i8; block];

    'rows: for r in 0..rows {
        let row_base = r * cols;
        let mut c = 0usize;
        for _ in 0..blocks_per_row {
            if processed == available_blocks {
                break 'rows;
            }
            let n = (cols - c).min(block);
            if off + per_block > bytes.len() {
                break 'rows;
            }

            let qbits_slice = &bytes[off..off + n.div_ceil(4)];
            off += qbits_slice.len();
            let scale_bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            off += 2;
            let scale = f16::from_bits(scale_bits).to_f32();

            unpack_2bit_signed(&mut scratch[..n], qbits_slice, n);
            for i in 0..n {
                out[row_base + c + i] = scale * scratch[i] as f32;
            }
            c += n;
            processed += 1;
        }
    }
    // remaining values stay zero (explicit)
    Ok(out)
}

/// Get the number of elements per I2_S block
pub fn block_elems() -> usize {
    I2S_DEFAULT_BLOCK
}

/// Get the byte size of an I2_S block
pub fn block_bytes() -> usize {
    expected_bytes(1, I2S_DEFAULT_BLOCK, I2S_DEFAULT_BLOCK)
}

/// Dequantize I2_S directly into the transposed layout:
/// input logical shape = [rows, cols]; output shape = [cols, rows].
pub fn dequantize_to_f32_transposed(bytes: &[u8], shape: &[usize]) -> Result<Vec<f32>> {
    let (rows, cols) = rows_cols(shape)?;
    let mut block = infer_block_size(bytes.len(), rows, cols).unwrap_or(I2S_DEFAULT_BLOCK);

    let default_expected = expected_bytes(rows, cols, block);
    if bytes.len() != default_expected {
        if let Some(b) = infer_block_size(bytes.len(), rows, cols) {
            // Only warn if the difference is significant (more than one block worth of data)
            let one_block_bytes = expected_bytes(1, I2S_DEFAULT_BLOCK, I2S_DEFAULT_BLOCK);
            let shortfall = default_expected.saturating_sub(bytes.len());
            if shortfall > one_block_bytes {
                warn!(
                    "I2_S: non-default block size detected: {} (default {})",
                    b, I2S_DEFAULT_BLOCK
                );
            } else {
                debug!(
                    "I2_S: non-default block size detected: {} (default {})",
                    b, I2S_DEFAULT_BLOCK
                );
            }
            block = b;
        } else {
            // Partial fallback: decode what we can into transposed output
            let qbits = block.div_ceil(4);
            let per_block = qbits + 2;
            let available_blocks = bytes.len() / per_block;
            warn!(
                "I2_S: byte length mismatch (got {}, expected {}), processing {} blocks then zero-fill (transposed)",
                bytes.len(),
                default_expected,
                available_blocks
            );
            return dequantize_partial_blocks_transposed(bytes, shape, block, available_blocks);
        }
    }

    // Normal path: direct transposed dequant
    let blocks_per_row = cols.div_ceil(block);
    let mut out = vec![0f32; rows * cols]; // output logical shape [cols, rows]
    let mut off = 0usize;
    let mut scratch = vec![0i8; block];

    for r in 0..rows {
        let mut c = 0usize;
        for _ in 0..blocks_per_row {
            let n = (cols - c).min(block);
            let qbits_len = n.div_ceil(4);
            let qbits_slice = &bytes[off..off + qbits_len];
            off += qbits_len;

            let scale_bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            off += 2;
            let scale = f16::from_bits(scale_bits).to_f32();

            unpack_2bit_signed(&mut scratch[..n], qbits_slice, n);

            // Write transposed: input (r, c+i) -> output (c+i, r)
            for i in 0..n {
                out[(c + i) * rows + r] = scale * scratch[i] as f32;
            }
            c += n;
        }
    }
    Ok(out)
}

fn dequantize_partial_blocks_transposed(
    bytes: &[u8],
    shape: &[usize],
    block: usize,
    available_blocks: usize,
) -> Result<Vec<f32>> {
    let (rows, cols) = rows_cols(shape)?;
    let blocks_per_row = cols.div_ceil(block);
    let qbits = block.div_ceil(4);
    let per_block = qbits + 2;

    let mut out = vec![0f32; rows * cols]; // transposed output
    let mut off = 0usize;
    let mut processed = 0usize;
    let mut scratch = vec![0i8; block];

    'rows: for r in 0..rows {
        let mut c = 0usize;
        for _ in 0..blocks_per_row {
            if processed == available_blocks {
                break 'rows;
            }
            let n = (cols - c).min(block);
            if off + per_block > bytes.len() {
                break 'rows;
            }

            let qbits_len = n.div_ceil(4);
            let qbits_slice = &bytes[off..off + qbits_len];
            off += qbits_len;

            let scale_bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            off += 2;
            let scale = f16::from_bits(scale_bits).to_f32();

            unpack_2bit_signed(&mut scratch[..n], qbits_slice, n);
            for i in 0..n {
                out[(c + i) * rows + r] = scale * scratch[i] as f32;
            }
            c += n;
            processed += 1;
        }
    }
    // remainder stays zero
    Ok(out)
}
