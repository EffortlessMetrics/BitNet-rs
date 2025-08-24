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

    // Validate input size
    let expected = rows * blocks_per_row * I2S_BLOCK_BYTES;
    if bytes.len() != expected {
        bail!("I2_S: byte length mismatch (got {}, expected {})", bytes.len(), expected);
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
