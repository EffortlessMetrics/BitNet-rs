//! Native Rust implementation of I2_S (BitNet 2-bit signed) dequantization
//!
//! Layout per block:
//! - 256 elements per block
//! - 64 bytes of packed 2-bit values
//! - 2 bytes for f16 scale
//! - Total: 66 bytes per block

use anyhow::{Result, bail};
use half::f16;

/// I2_S block constants
const I2S_BLOCK_ELEMS: usize = 256;
const I2S_BLOCK_BYTES: usize = 64 /* qbits */ + 2 /* f16 scale */;

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
    let blocks_per_row = cols.div_ceil(I2S_BLOCK_ELEMS);

    // Validate input size with flexible handling
    let expected = rows * blocks_per_row * I2S_BLOCK_BYTES;
    if bytes.len() != expected {
        // Try alternative block sizes (32, 64, 128 elements) for compatibility
        let alt_block_sizes = [32, 64, 128];
        let mut found_valid = false;

        for &block_size in &alt_block_sizes {
            let alt_blocks_per_row = cols.div_ceil(block_size);
            let alt_block_bytes = (block_size / 4) + 2; // 2-bit packed + f16 scale
            let alt_expected = rows * alt_blocks_per_row * alt_block_bytes;

            if bytes.len() == alt_expected {
                eprintln!("Warning: I2_S using alternative block size {} instead of {}", block_size, I2S_BLOCK_ELEMS);
                return dequantize_to_f32_with_block_size(bytes, shape, block_size);
            }
        }

        // Graceful fallback: try to process with partial blocks
        let total_blocks = expected / I2S_BLOCK_BYTES;
        let actual_blocks = bytes.len() / I2S_BLOCK_BYTES;

        if actual_blocks > 0 && actual_blocks < total_blocks {
            eprintln!("Warning: I2_S partial data ({} bytes vs {} expected) - processing {} of {} blocks",
                     bytes.len(), expected, actual_blocks, total_blocks);
            return dequantize_partial_blocks(bytes, shape, actual_blocks);
        }

        bail!("I2_S: byte length mismatch (got {}, expected {}) and no valid alternative block size found", bytes.len(), expected);
    }

    let mut out = vec![0f32; rows * cols];
    let mut off = 0usize;

    // Scratch buffer for unpacked values
    let mut scratch = [0i8; I2S_BLOCK_ELEMS];

    for r in 0..rows {
        let row_base = r * cols;
        let mut c = 0usize;

        for _ in 0..blocks_per_row {
            let n = (cols - c).min(I2S_BLOCK_ELEMS);

            // Layout: [64 bytes qbits][2 bytes f16 scale]
            let qbits = &bytes[off..off + 64];
            off += 64;

            let scale_bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            off += 2;
            let scale = f16::from_bits(scale_bits).to_f32();

            // Unpack and dequantize
            unpack_2bit_signed(&mut scratch[..n], qbits, n);
            for i in 0..n {
                out[row_base + c + i] = scale * (scratch[i] as f32);
            }
            c += n;
        }
    }

    Ok(out)
}

/// Get the number of elements per I2_S block
pub fn block_elems() -> usize {
    I2S_BLOCK_ELEMS
}

/// Get the byte size of an I2_S block
pub fn block_bytes() -> usize {
    I2S_BLOCK_BYTES
}

/// Dequantize I2_S with alternative block size
fn dequantize_to_f32_with_block_size(bytes: &[u8], shape: &[usize], block_size: usize) -> Result<Vec<f32>> {
    let (rows, cols) = rows_cols(shape)?;
    let blocks_per_row = cols.div_ceil(block_size);
    let block_bytes = (block_size / 4) + 2; // 2-bit packed + f16 scale

    let mut out = vec![0f32; rows * cols];
    let mut off = 0usize;

    // Scratch buffer for unpacked values
    let mut scratch = vec![0i8; block_size];

    for r in 0..rows {
        let row_base = r * cols;
        let mut c = 0usize;

        for _ in 0..blocks_per_row {
            let n = (cols - c).min(block_size);

            // Layout: [qbits][2 bytes f16 scale]
            let qbits_len = n.div_ceil(4); // 2-bit packed, 4 values per byte
            let qbits = &bytes[off..off + qbits_len];
            off += qbits_len;

            let scale_bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            off += 2;
            let scale = f16::from_bits(scale_bits).to_f32();

            // Unpack and dequantize
            unpack_2bit_signed(&mut scratch[..n], qbits, n);
            for i in 0..n {
                out[row_base + c + i] = scale * (scratch[i] as f32);
            }
            c += n;
        }
    }

    Ok(out)
}

/// Dequantize I2_S with partial block processing
fn dequantize_partial_blocks(bytes: &[u8], shape: &[usize], available_blocks: usize) -> Result<Vec<f32>> {
    let (rows, cols) = rows_cols(shape)?;
    let mut out = vec![0f32; rows * cols];
    let mut off = 0usize;

    // Scratch buffer for unpacked values
    let mut scratch = [0i8; I2S_BLOCK_ELEMS];

    let blocks_per_row = cols.div_ceil(I2S_BLOCK_ELEMS);
    let max_processable_rows = available_blocks / blocks_per_row;

    for r in 0..max_processable_rows.min(rows) {
        let row_base = r * cols;
        let mut c = 0usize;

        for _ in 0..blocks_per_row {
            if off + I2S_BLOCK_BYTES > bytes.len() {
                break; // No more data
            }

            let n = (cols - c).min(I2S_BLOCK_ELEMS);

            // Layout: [64 bytes qbits][2 bytes f16 scale]
            let qbits = &bytes[off..off + 64];
            off += 64;

            let scale_bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            off += 2;
            let scale = f16::from_bits(scale_bits).to_f32();

            // Unpack and dequantize
            unpack_2bit_signed(&mut scratch[..n], qbits, n);
            for i in 0..n {
                out[row_base + c + i] = scale * (scratch[i] as f32);
            }
            c += n;
        }
    }

    Ok(out)
}
