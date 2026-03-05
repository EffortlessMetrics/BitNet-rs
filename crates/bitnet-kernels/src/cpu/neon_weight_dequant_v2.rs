#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::missing_safety_doc,
    clippy::too_many_arguments,
    clippy::ptr_as_ptr,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! NEON-optimized weight dequantization for BitNet ternary weights.
//!
//! Implements high-throughput unpacking of I2_S (2-bit signed) packed weights
//! using ARM NEON SIMD intrinsics on Apple Silicon (aarch64).
//!
//! ## I2_S encoding (2 bits per value, 4 values per byte, LSB-first)
//!
//! | 2-bit code | value |
//! |------------|-------|
//! | `0b00`     |  0    |
//! | `0b01`     | +1    |
//! | `0b11`     | −1    |
//! | `0b10`     | reserved (treated as 0) |
//!
//! ## Block formats
//!
//! - **32-element blocks**: standard BitNet32-F16 (32 values, 8 packed bytes,
//!   1 f32 scale per block)
//! - **256-element blocks (QK256)**: GGML-compatible (256 values, 64 packed
//!   bytes, 1 f32 scale per block)
//!
//! All public functions have scalar fallbacks for non-aarch64 targets.

// ── Scalar helpers ─────────────────────────────────────────────────────

/// Decode a single 2-bit I2_S code to its signed integer value.
#[inline(always)]
fn decode_i2s_scalar(bits: u8) -> i8 {
    match bits & 0x03 {
        0b01 => 1,
        0b11 => -1,
        _ => 0,
    }
}

/// Unpack a single byte into 4 ternary i8 values (LSB-first).
#[inline(always)]
fn unpack_byte_scalar(byte: u8) -> [i8; 4] {
    [
        decode_i2s_scalar(byte),
        decode_i2s_scalar(byte >> 2),
        decode_i2s_scalar(byte >> 4),
        decode_i2s_scalar(byte >> 6),
    ]
}

// ── NEON unpack core ───────────────────────────────────────────────────

/// Unpack 8 bytes (32 ternary values) into `i8x16` lane pairs using NEON.
///
/// Returns two `int8x16_t` vectors containing 16 ternary values each
/// (total 32 values from the 8 input bytes).
///
/// # Safety
///
/// Caller must ensure `neon` target feature is available and `packed`
/// contains at least 8 bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_unpack_32_values(
    packed: &[u8],
) -> (std::arch::aarch64::int8x16_t, std::arch::aarch64::int8x16_t) {
    use std::arch::aarch64::*;

    debug_assert!(packed.len() >= 8);

    // LUT: 2-bit code → signed value. Index by 2-bit code.
    // [0b00→0, 0b01→1, 0b10→0, 0b11→-1]  repeated across lanes
    let lut_vals: [i8; 4] = [0, 1, 0, -1];

    let mut lo_arr = [0i8; 16];
    let mut hi_arr = [0i8; 16];

    // Process 8 bytes → 32 ternary values
    for i in 0..8 {
        let byte = packed[i];
        let v0 = lut_vals[(byte & 0x03) as usize];
        let v1 = lut_vals[((byte >> 2) & 0x03) as usize];
        let v2 = lut_vals[((byte >> 4) & 0x03) as usize];
        let v3 = lut_vals[((byte >> 6) & 0x03) as usize];
        if i < 4 {
            lo_arr[i * 4] = v0;
            lo_arr[i * 4 + 1] = v1;
            lo_arr[i * 4 + 2] = v2;
            lo_arr[i * 4 + 3] = v3;
        } else {
            let j = i - 4;
            hi_arr[j * 4] = v0;
            hi_arr[j * 4 + 1] = v1;
            hi_arr[j * 4 + 2] = v2;
            hi_arr[j * 4 + 3] = v3;
        }
    }

    unsafe {
        let lo = vld1q_s8(lo_arr.as_ptr());
        let hi = vld1q_s8(hi_arr.as_ptr());
        (lo, hi)
    }
}

// ── Public scalar fallback: unpack 32 ─────────────────────────────────

/// Unpack 8 packed bytes into 32 ternary i8 values (scalar fallback).
pub fn unpack_32_values_scalar(packed: &[u8], out: &mut [i8]) {
    debug_assert!(packed.len() >= 8);
    debug_assert!(out.len() >= 32);
    for i in 0..8 {
        let vals = unpack_byte_scalar(packed[i]);
        out[i * 4] = vals[0];
        out[i * 4 + 1] = vals[1];
        out[i * 4 + 2] = vals[2];
        out[i * 4 + 3] = vals[3];
    }
}

// ── I2_S dequantization: single block ──────────────────────────────────

/// Dequantize a block of I2_S packed weights applying a scalar scale.
///
/// Each byte of `packed` encodes 4 ternary values. The `block_size`
/// values are unpacked and multiplied by `scale`, writing into `out`.
///
/// On aarch64 this uses NEON SIMD; elsewhere it falls back to scalar.
pub fn dequant_i2s_block_v2(packed: &[u8], scale: f32, block_size: usize, out: &mut [f32]) {
    assert!(packed.len() >= block_size.div_ceil(4), "not enough packed bytes");
    assert!(out.len() >= block_size, "output buffer too small");

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 always has NEON
        unsafe { dequant_i2s_block_neon(packed, scale, block_size, out) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        dequant_i2s_block_scalar(packed, scale, block_size, out);
    }
}

/// Scalar implementation of I2_S block dequantization.
pub fn dequant_i2s_block_scalar(packed: &[u8], scale: f32, block_size: usize, out: &mut [f32]) {
    for (i, out_val) in out.iter_mut().enumerate().take(block_size) {
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        let bits = (packed[byte_idx] >> bit_off) & 0x03;
        *out_val = decode_i2s_scalar(bits) as f32 * scale;
    }
}

/// NEON-accelerated I2_S block dequantization.
///
/// Processes 16 values at a time using `int8x16_t` → `float32x4_t`
/// conversion with NEON multiply.
///
/// # Safety
///
/// Requires `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dequant_i2s_block_neon(packed: &[u8], scale: f32, block_size: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;

    let lut: [i8; 4] = [0, 1, 0, -1];

    // Process 16 values (4 bytes) at a time
    let full_chunks = block_size / 16;
    let remainder = block_size % 16;

    for chunk in 0..full_chunks {
        let base_byte = chunk * 4;
        let base_out = chunk * 16;

        // Unpack 4 bytes → 16 i8 values
        let mut i8_arr = [0i8; 16];
        for b in 0..4 {
            let byte = packed[base_byte + b];
            i8_arr[b * 4] = lut[(byte & 0x03) as usize];
            i8_arr[b * 4 + 1] = lut[((byte >> 2) & 0x03) as usize];
            i8_arr[b * 4 + 2] = lut[((byte >> 4) & 0x03) as usize];
            i8_arr[b * 4 + 3] = lut[((byte >> 6) & 0x03) as usize];
        }

        unsafe {
            let scale_v = vdupq_n_f32(scale);
            let i8v = vld1q_s8(i8_arr.as_ptr());

            // Convert lower 8 i8 → two f32x4 vectors
            let i16_lo = vmovl_s8(vget_low_s8(i8v));
            let i32_lo_lo = vmovl_s16(vget_low_s16(i16_lo));
            let i32_lo_hi = vmovl_s16(vget_high_s16(i16_lo));
            let f0 = vmulq_f32(vcvtq_f32_s32(i32_lo_lo), scale_v);
            let f1 = vmulq_f32(vcvtq_f32_s32(i32_lo_hi), scale_v);

            // Convert upper 8 i8 → two f32x4 vectors
            let i16_hi = vmovl_s8(vget_high_s8(i8v));
            let i32_hi_lo = vmovl_s16(vget_low_s16(i16_hi));
            let i32_hi_hi = vmovl_s16(vget_high_s16(i16_hi));
            let f2 = vmulq_f32(vcvtq_f32_s32(i32_hi_lo), scale_v);
            let f3 = vmulq_f32(vcvtq_f32_s32(i32_hi_hi), scale_v);

            vst1q_f32(out.as_mut_ptr().add(base_out), f0);
            vst1q_f32(out.as_mut_ptr().add(base_out + 4), f1);
            vst1q_f32(out.as_mut_ptr().add(base_out + 8), f2);
            vst1q_f32(out.as_mut_ptr().add(base_out + 12), f3);
        }
    }

    // Scalar tail
    let tail_start = full_chunks * 16;
    let tail_byte_start = full_chunks * 4;
    for i in 0..remainder {
        let global_i = tail_start + i;
        let byte_idx = tail_byte_start + i / 4;
        let bit_off = (i % 4) * 2;
        let bits = (packed[byte_idx] >> bit_off) & 0x03;
        out[global_i] = decode_i2s_scalar(bits) as f32 * scale;
    }
}

// ── 32-element block dequantization ────────────────────────────────────

/// Block size for standard BitNet32-F16 blocks.
pub const BLOCK_SIZE_32: usize = 32;

/// Block size for QK256 blocks.
pub const BLOCK_SIZE_QK256: usize = 256;

/// Bytes per 32-element block.
pub const BYTES_PER_BLOCK_32: usize = BLOCK_SIZE_32 / 4;

/// Bytes per QK256 block.
pub const BYTES_PER_BLOCK_QK256: usize = BLOCK_SIZE_QK256 / 4;

/// Dequantize a single 32-element block with its scale.
///
/// `packed` must contain at least 8 bytes. `out` must have space for 32
/// f32 values.
pub fn dequant_block32(packed: &[u8], scale: f32, out: &mut [f32]) {
    dequant_i2s_block_v2(packed, scale, BLOCK_SIZE_32, out);
}

/// Dequantize a single 256-element (QK256) block with its scale.
///
/// `packed` must contain at least 64 bytes. `out` must have space for 256
/// f32 values.
pub fn dequant_block_qk256(packed: &[u8], scale: f32, out: &mut [f32]) {
    dequant_i2s_block_v2(packed, scale, BLOCK_SIZE_QK256, out);
}

// ── Row dequantization with per-block scales ───────────────────────────

/// Dequantize a row of I2_S packed weights with per-block f32 scales.
///
/// The row has `num_elements` total values, divided into blocks of
/// `block_size`. Each block has an entry in `scales`.
///
/// On aarch64 this dispatches to NEON; on other architectures it uses
/// the scalar path.
pub fn dequant_row_blocked(
    packed: &[u8],
    scales: &[f32],
    block_size: usize,
    num_elements: usize,
    out: &mut [f32],
) {
    assert!(block_size > 0, "block_size must be > 0");
    let num_blocks = num_elements.div_ceil(block_size);
    assert!(
        scales.len() >= num_blocks,
        "not enough scales: need {num_blocks}, got {}",
        scales.len()
    );
    let bytes_needed = num_elements.div_ceil(4);
    assert!(
        packed.len() >= bytes_needed,
        "not enough packed bytes: need {bytes_needed}, got {}",
        packed.len()
    );
    assert!(out.len() >= num_elements, "output too small: need {num_elements}, got {}", out.len());

    let bytes_per_block = block_size / 4;
    for (blk, &scale) in scales.iter().enumerate().take(num_blocks) {
        let elem_start = blk * block_size;
        let elem_end = (elem_start + block_size).min(num_elements);
        let this_block_size = elem_end - elem_start;
        let byte_start = blk * bytes_per_block;
        dequant_i2s_block_v2(&packed[byte_start..], scale, this_block_size, &mut out[elem_start..]);
    }
}

/// Dequantize a row using 32-element blocks.
pub fn dequant_row_block32(packed: &[u8], scales: &[f32], num_elements: usize, out: &mut [f32]) {
    dequant_row_blocked(packed, scales, BLOCK_SIZE_32, num_elements, out);
}

/// Dequantize a row using 256-element (QK256) blocks.
pub fn dequant_row_qk256(packed: &[u8], scales: &[f32], num_elements: usize, out: &mut [f32]) {
    dequant_row_blocked(packed, scales, BLOCK_SIZE_QK256, num_elements, out);
}

// ── Batch dequantization ───────────────────────────────────────────────

/// Dequantize multiple weight rows simultaneously.
///
/// Each row is `row_elements` wide, packed contiguously in `packed_rows`
/// (each row occupies `ceil(row_elements / 4)` bytes). `scales_per_row`
/// provides the per-block scales for each row. Results are written into
/// the flat `out` buffer (row-major, `num_rows × row_elements`).
pub fn dequant_batch_rows(
    packed_rows: &[u8],
    scales_per_row: &[&[f32]],
    block_size: usize,
    row_elements: usize,
    num_rows: usize,
    out: &mut [f32],
) {
    assert!(num_rows > 0);
    assert_eq!(scales_per_row.len(), num_rows);
    let bytes_per_row = row_elements.div_ceil(4);
    assert!(packed_rows.len() >= num_rows * bytes_per_row, "packed_rows too small");
    assert!(out.len() >= num_rows * row_elements, "output buffer too small");

    for (row, &row_scales) in scales_per_row.iter().enumerate().take(num_rows) {
        let packed_start = row * bytes_per_row;
        let out_start = row * row_elements;
        dequant_row_blocked(
            &packed_rows[packed_start..packed_start + bytes_per_row],
            row_scales,
            block_size,
            row_elements,
            &mut out[out_start..out_start + row_elements],
        );
    }
}

/// Dequantize a batch with a flat scales array.
///
/// `all_scales` is laid out as `[row0_block0, row0_block1, ..., row1_block0, ...]`.
/// Each row has `blocks_per_row = ceil(row_elements / block_size)` scale entries.
pub fn dequant_batch_rows_flat_scales(
    packed_rows: &[u8],
    all_scales: &[f32],
    block_size: usize,
    row_elements: usize,
    num_rows: usize,
    out: &mut [f32],
) {
    let blocks_per_row = row_elements.div_ceil(block_size);
    assert!(all_scales.len() >= num_rows * blocks_per_row, "not enough scales");
    let bytes_per_row = row_elements.div_ceil(4);

    for row in 0..num_rows {
        let packed_start = row * bytes_per_row;
        let scale_start = row * blocks_per_row;
        let out_start = row * row_elements;
        dequant_row_blocked(
            &packed_rows[packed_start..packed_start + bytes_per_row],
            &all_scales[scale_start..scale_start + blocks_per_row],
            block_size,
            row_elements,
            &mut out[out_start..out_start + row_elements],
        );
    }
}

// ── Scale application ──────────────────────────────────────────────────

/// Multiply a slice of f32 values by a scalar scale, in place.
///
/// On aarch64 uses NEON `vmulq_f32` for 4-wide processing.
pub fn apply_scale_inplace(data: &mut [f32], scale: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 always has NEON.
        unsafe { apply_scale_inplace_neon(data, scale) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for v in data.iter_mut() {
            *v *= scale;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_scale_inplace_neon(data: &mut [f32], scale: f32) {
    use std::arch::aarch64::*;

    let chunks = data.len() / 4;
    let remainder = data.len() % 4;

    for i in 0..chunks {
        unsafe {
            let scale_v = vdupq_n_f32(scale);
            let ptr = data.as_mut_ptr().add(i * 4);
            let v = vld1q_f32(ptr);
            let r = vmulq_f32(v, scale_v);
            vst1q_f32(ptr, r);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        data[tail_start + i] *= scale;
    }
}

/// Multiply a slice of f32 values by a scalar scale, writing to `out`.
pub fn apply_scale(data: &[f32], scale: f32, out: &mut [f32]) {
    assert!(out.len() >= data.len());

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { apply_scale_neon(data, scale, out) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for (o, &v) in out.iter_mut().zip(data.iter()) {
            *o = v * scale;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_scale_neon(data: &[f32], scale: f32, out: &mut [f32]) {
    use std::arch::aarch64::*;

    let chunks = data.len() / 4;
    let remainder = data.len() % 4;

    for i in 0..chunks {
        unsafe {
            let scale_v = vdupq_n_f32(scale);
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            let r = vmulq_f32(v, scale_v);
            vst1q_f32(out.as_mut_ptr().add(i * 4), r);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        out[tail_start + i] = data[tail_start + i] * scale;
    }
}

// ── Zero-point offset ──────────────────────────────────────────────────

/// Apply a zero-point offset to dequantized values:
/// `out[i] = (dequant[i] - zero_point) * scale`.
///
/// This handles quantization schemes where the zero-point is not 0.
pub fn dequant_with_zero_point(
    packed: &[u8],
    scale: f32,
    zero_point: f32,
    block_size: usize,
    out: &mut [f32],
) {
    assert!(out.len() >= block_size);
    assert!(packed.len() >= block_size.div_ceil(4));

    // First dequantize without scale (scale=1.0)
    dequant_i2s_block_v2(packed, 1.0, block_size, out);

    // Apply zero-point and scale
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            apply_zero_point_scale_neon(out, block_size, zero_point, scale);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..block_size {
            out[i] = (out[i] - zero_point) * scale;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_zero_point_scale_neon(out: &mut [f32], count: usize, zero_point: f32, scale: f32) {
    use std::arch::aarch64::*;

    let chunks = count / 4;
    let remainder = count % 4;

    for i in 0..chunks {
        unsafe {
            let zp_v = vdupq_n_f32(zero_point);
            let scale_v = vdupq_n_f32(scale);
            let ptr = out.as_mut_ptr().add(i * 4);
            let v = vld1q_f32(ptr);
            let shifted = vsubq_f32(v, zp_v);
            let scaled = vmulq_f32(shifted, scale_v);
            vst1q_f32(ptr, scaled);
        }
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        out[tail + i] = (out[tail + i] - zero_point) * scale;
    }
}

// ── Interleaved layout support ─────────────────────────────────────────

/// Dequantize weights stored in an interleaved (NEON-friendly) layout.
///
/// In the interleaved layout, 4 consecutive weight rows have their
/// elements interleaved at a granularity of 4, so that a single NEON
/// `vld4q` can load one element from each of the 4 rows.
///
/// `packed_interleaved`: interleaved packed bytes for `num_rows` rows
///   (must be a multiple of 4). Row stride is `elements_per_row` values.
/// `scales`: `[num_rows][blocks_per_row]` flat array of per-block scales.
/// `out`: deinterleaved output, row-major `[num_rows][elements_per_row]`.
pub fn dequant_interleaved_4row(
    packed_interleaved: &[u8],
    scales: &[f32],
    elements_per_row: usize,
    num_rows: usize,
    out: &mut [f32],
) {
    assert!(num_rows.is_multiple_of(4), "num_rows must be multiple of 4");
    assert!(out.len() >= num_rows * elements_per_row, "output too small");

    let blocks_per_row = elements_per_row.div_ceil(BLOCK_SIZE_32);
    assert!(scales.len() >= num_rows * blocks_per_row, "not enough scales");

    // The interleaved layout groups 4 rows together. For each group of 4
    // rows, the packed bytes are interleaved at the byte level.
    let bytes_per_row = elements_per_row.div_ceil(4);
    let group_stride = bytes_per_row * 4; // 4 interleaved rows

    for group in 0..(num_rows / 4) {
        let group_packed = &packed_interleaved[group * group_stride..];

        for col_byte in 0..bytes_per_row {
            // In interleaved layout: bytes for the same column position
            // from 4 rows are consecutive.
            let interleaved_offset = col_byte * 4;

            for row_in_group in 0..4u8 {
                let global_row = group * 4 + row_in_group as usize;
                let byte = group_packed[interleaved_offset + row_in_group as usize];
                let vals = unpack_byte_scalar(byte);

                let col_start = col_byte * 4;
                let blk = col_start / BLOCK_SIZE_32;
                let scale = scales[global_row * blocks_per_row + blk.min(blocks_per_row - 1)];

                let out_base = global_row * elements_per_row + col_start;
                for j in 0..4 {
                    if col_start + j < elements_per_row {
                        out[out_base + j] = vals[j] as f32 * scale;
                    }
                }
            }
        }
    }
}

/// Pack weight rows into the interleaved 4-row layout.
///
/// Takes row-major packed bytes and reorders them for NEON-friendly
/// access. `num_rows` must be a multiple of 4.
pub fn pack_interleaved_4row(
    packed_rows: &[u8],
    elements_per_row: usize,
    num_rows: usize,
) -> Vec<u8> {
    assert!(num_rows.is_multiple_of(4), "num_rows must be multiple of 4");
    let bytes_per_row = elements_per_row.div_ceil(4);
    let mut interleaved = vec![0u8; num_rows * bytes_per_row];

    for group in 0..(num_rows / 4) {
        for col_byte in 0..bytes_per_row {
            for row_in_group in 0..4usize {
                let global_row = group * 4 + row_in_group;
                let src = global_row * bytes_per_row + col_byte;
                let dst = group * bytes_per_row * 4 + col_byte * 4 + row_in_group;
                interleaved[dst] = packed_rows[src];
            }
        }
    }

    interleaved
}

// ── Unpack to i8 buffer ────────────────────────────────────────────────

/// Unpack all packed bytes into a contiguous i8 buffer.
pub fn unpack_to_i8(packed: &[u8], num_elements: usize, out: &mut [i8]) {
    assert!(out.len() >= num_elements);
    assert!(packed.len() >= num_elements.div_ceil(4));

    let full_bytes = num_elements / 4;
    let remainder = num_elements % 4;

    for i in 0..full_bytes {
        let vals = unpack_byte_scalar(packed[i]);
        out[i * 4] = vals[0];
        out[i * 4 + 1] = vals[1];
        out[i * 4 + 2] = vals[2];
        out[i * 4 + 3] = vals[3];
    }

    if remainder > 0 {
        let vals = unpack_byte_scalar(packed[full_bytes]);
        for j in 0..remainder {
            out[full_bytes * 4 + j] = vals[j];
        }
    }
}

/// Unpack packed bytes and return a new `Vec<i8>`.
pub fn unpack_to_i8_vec(packed: &[u8], num_elements: usize) -> Vec<i8> {
    let mut out = vec![0i8; num_elements];
    unpack_to_i8(packed, num_elements, &mut out);
    out
}

// ── Pack from i8 ───────────────────────────────────────────────────────

/// Pack ternary i8 values ({-1, 0, +1}) into I2_S packed bytes.
pub fn pack_from_i8(values: &[i8]) -> Vec<u8> {
    let num_bytes = values.len().div_ceil(4);
    let mut packed = vec![0u8; num_bytes];

    for (i, &v) in values.iter().enumerate() {
        let code: u8 = match v {
            1 => 0b01,
            -1 => 0b11,
            _ => 0b00,
        };
        packed[i / 4] |= code << ((i % 4) * 2);
    }

    packed
}

// ── Dequant + accumulate ───────────────────────────────────────────────

/// Dequantize a block and accumulate (add) into an existing f32 buffer.
///
/// Useful for fused residual or bias-add operations.
pub fn dequant_i2s_block_accumulate(packed: &[u8], scale: f32, block_size: usize, acc: &mut [f32]) {
    assert!(packed.len() >= block_size.div_ceil(4));
    assert!(acc.len() >= block_size);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            dequant_i2s_block_accumulate_neon(packed, scale, block_size, acc);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..block_size {
            let byte_idx = i / 4;
            let bit_off = (i % 4) * 2;
            let bits = (packed[byte_idx] >> bit_off) & 0x03;
            acc[i] += decode_i2s_scalar(bits) as f32 * scale;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dequant_i2s_block_accumulate_neon(
    packed: &[u8],
    scale: f32,
    block_size: usize,
    acc: &mut [f32],
) {
    use std::arch::aarch64::*;

    let lut: [i8; 4] = [0, 1, 0, -1];

    let full_chunks = block_size / 16;
    let remainder = block_size % 16;

    for chunk in 0..full_chunks {
        let base_byte = chunk * 4;
        let base_out = chunk * 16;

        let mut i8_arr = [0i8; 16];
        for b in 0..4 {
            let byte = packed[base_byte + b];
            i8_arr[b * 4] = lut[(byte & 0x03) as usize];
            i8_arr[b * 4 + 1] = lut[((byte >> 2) & 0x03) as usize];
            i8_arr[b * 4 + 2] = lut[((byte >> 4) & 0x03) as usize];
            i8_arr[b * 4 + 3] = lut[((byte >> 6) & 0x03) as usize];
        }

        unsafe {
            let scale_v = vdupq_n_f32(scale);
            let i8v = vld1q_s8(i8_arr.as_ptr());
            let i16_lo = vmovl_s8(vget_low_s8(i8v));
            let i32_ll = vmovl_s16(vget_low_s16(i16_lo));
            let i32_lh = vmovl_s16(vget_high_s16(i16_lo));
            let i16_hi = vmovl_s8(vget_high_s8(i8v));
            let i32_hl = vmovl_s16(vget_low_s16(i16_hi));
            let i32_hh = vmovl_s16(vget_high_s16(i16_hi));

            let ptr = acc.as_mut_ptr().add(base_out);
            let a0 = vld1q_f32(ptr);
            let a1 = vld1q_f32(ptr.add(4));
            let a2 = vld1q_f32(ptr.add(8));
            let a3 = vld1q_f32(ptr.add(12));

            vst1q_f32(ptr, vaddq_f32(a0, vmulq_f32(vcvtq_f32_s32(i32_ll), scale_v)));
            vst1q_f32(ptr.add(4), vaddq_f32(a1, vmulq_f32(vcvtq_f32_s32(i32_lh), scale_v)));
            vst1q_f32(ptr.add(8), vaddq_f32(a2, vmulq_f32(vcvtq_f32_s32(i32_hl), scale_v)));
            vst1q_f32(ptr.add(12), vaddq_f32(a3, vmulq_f32(vcvtq_f32_s32(i32_hh), scale_v)));
        }
    }

    let tail_start = full_chunks * 16;
    let tail_byte_start = full_chunks * 4;
    for i in 0..remainder {
        let global_i = tail_start + i;
        let byte_idx = tail_byte_start + i / 4;
        let bit_off = (i % 4) * 2;
        let bits = (packed[byte_idx] >> bit_off) & 0x03;
        acc[global_i] += decode_i2s_scalar(bits) as f32 * scale;
    }
}

// ── Convenience: allocating dequant ────────────────────────────────────

/// Dequantize a row and return a new `Vec<f32>`.
pub fn dequant_row_to_vec(
    packed: &[u8],
    scales: &[f32],
    block_size: usize,
    num_elements: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_elements];
    dequant_row_blocked(packed, scales, block_size, num_elements, &mut out);
    out
}

// ── Statistics helpers ─────────────────────────────────────────────────

/// Count the number of +1, -1, and 0 values in packed data.
pub fn count_ternary_values(packed: &[u8], num_elements: usize) -> (usize, usize, usize) {
    let mut pos = 0usize;
    let mut neg = 0usize;
    let mut zero = 0usize;

    for i in 0..num_elements {
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        let bits = (packed[byte_idx] >> bit_off) & 0x03;
        match bits {
            0b01 => pos += 1,
            0b11 => neg += 1,
            _ => zero += 1,
        }
    }

    (pos, neg, zero)
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    // ── Test helpers ───────────────────────────────────────────────────

    fn pack4(vals: [i8; 4]) -> u8 {
        let mut byte = 0u8;
        for (i, &v) in vals.iter().enumerate() {
            let code: u8 = match v {
                1 => 0b01,
                -1 => 0b11,
                _ => 0b00,
            };
            byte |= code << (i * 2);
        }
        byte
    }

    /// Reference scalar dequantization for correctness checks.
    fn reference_dequant(packed: &[u8], scale: f32, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let byte_idx = i / 4;
            let bit_off = (i % 4) * 2;
            let bits = (packed[byte_idx] >> bit_off) & 0x03;
            let val = match bits & 0x03 {
                0b01 => 1.0f32,
                0b11 => -1.0f32,
                _ => 0.0f32,
            };
            out.push(val * scale);
        }
        out
    }

    fn assert_f32_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at [{i}]: {x} vs {y} (tol={tol})");
        }
    }

    // ── decode_i2s_scalar ─────────────────────────────────────────────

    #[test]
    fn test_decode_i2s_00() {
        assert_eq!(decode_i2s_scalar(0b00), 0);
    }

    #[test]
    fn test_decode_i2s_01() {
        assert_eq!(decode_i2s_scalar(0b01), 1);
    }

    #[test]
    fn test_decode_i2s_10() {
        assert_eq!(decode_i2s_scalar(0b10), 0);
    }

    #[test]
    fn test_decode_i2s_11() {
        assert_eq!(decode_i2s_scalar(0b11), -1);
    }

    #[test]
    fn test_decode_i2s_masks_upper_bits() {
        assert_eq!(decode_i2s_scalar(0xFF), -1); // 0xFF & 0x03 = 0b11
        assert_eq!(decode_i2s_scalar(0xFD), 1); // 0xFD & 0x03 = 0b01
    }

    // ── unpack_byte_scalar ────────────────────────────────────────────

    #[test]
    fn test_unpack_byte_all_zeros() {
        assert_eq!(unpack_byte_scalar(0x00), [0, 0, 0, 0]);
    }

    #[test]
    fn test_unpack_byte_all_plus_one() {
        assert_eq!(unpack_byte_scalar(0x55), [1, 1, 1, 1]);
    }

    #[test]
    fn test_unpack_byte_all_minus_one() {
        assert_eq!(unpack_byte_scalar(0xFF), [-1, -1, -1, -1]);
    }

    #[test]
    fn test_unpack_byte_mixed() {
        // [+1, -1, 0, +1] → 0b01_00_11_01 = 0x4D
        let byte = pack4([1, -1, 0, 1]);
        assert_eq!(unpack_byte_scalar(byte), [1, -1, 0, 1]);
    }

    #[test]
    fn test_unpack_byte_alternating() {
        let byte = pack4([1, -1, 1, -1]);
        assert_eq!(unpack_byte_scalar(byte), [1, -1, 1, -1]);
    }

    // ── unpack_32_values_scalar ───────────────────────────────────────

    #[test]
    fn test_unpack_32_values_scalar_zeros() {
        let packed = [0u8; 8];
        let mut out = [0i8; 32];
        unpack_32_values_scalar(&packed, &mut out);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_unpack_32_values_scalar_all_pos() {
        let packed = [0x55u8; 8];
        let mut out = [0i8; 32];
        unpack_32_values_scalar(&packed, &mut out);
        assert!(out.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_unpack_32_values_scalar_all_neg() {
        let packed = [0xFFu8; 8];
        let mut out = [0i8; 32];
        unpack_32_values_scalar(&packed, &mut out);
        assert!(out.iter().all(|&v| v == -1));
    }

    #[test]
    fn test_unpack_32_values_scalar_pattern() {
        let packed = [
            pack4([1, 0, -1, 0]),
            pack4([0, 1, 0, -1]),
            pack4([-1, -1, 1, 1]),
            pack4([0, 0, 0, 0]),
            pack4([1, 1, 1, 1]),
            pack4([-1, -1, -1, -1]),
            pack4([1, -1, 1, -1]),
            pack4([0, 1, -1, 0]),
        ];
        let mut out = [0i8; 32];
        unpack_32_values_scalar(&packed, &mut out);
        assert_eq!(out[0], 1);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], -1);
        assert_eq!(out[3], 0);
        assert_eq!(out[4], 0);
        assert_eq!(out[5], 1);
    }

    // ── NEON unpack_32_values (aarch64 only) ──────────────────────────

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_unpack_32_matches_scalar() {
        let packed = [
            pack4([1, -1, 0, 1]),
            pack4([0, 0, -1, -1]),
            pack4([1, 1, 1, 1]),
            pack4([-1, -1, -1, -1]),
            pack4([0, 1, -1, 0]),
            pack4([1, 0, 0, -1]),
            pack4([-1, 1, -1, 1]),
            pack4([0, 0, 0, 0]),
        ];

        let mut scalar_out = [0i8; 32];
        unpack_32_values_scalar(&packed, &mut scalar_out);

        unsafe {
            let (lo, hi) = neon_unpack_32_values(&packed);
            let mut neon_out = [0i8; 32];
            std::arch::aarch64::vst1q_s8(neon_out.as_mut_ptr(), lo);
            std::arch::aarch64::vst1q_s8(neon_out.as_mut_ptr().add(16), hi);
            assert_eq!(neon_out, scalar_out);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_unpack_32_all_zeros() {
        let packed = [0u8; 8];
        unsafe {
            let (lo, hi) = neon_unpack_32_values(&packed);
            let mut out = [0i8; 32];
            std::arch::aarch64::vst1q_s8(out.as_mut_ptr(), lo);
            std::arch::aarch64::vst1q_s8(out.as_mut_ptr().add(16), hi);
            assert!(out.iter().all(|&v| v == 0));
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_unpack_32_all_plus() {
        let packed = [0x55u8; 8];
        unsafe {
            let (lo, hi) = neon_unpack_32_values(&packed);
            let mut out = [0i8; 32];
            std::arch::aarch64::vst1q_s8(out.as_mut_ptr(), lo);
            std::arch::aarch64::vst1q_s8(out.as_mut_ptr().add(16), hi);
            assert!(out.iter().all(|&v| v == 1));
        }
    }

    // ── dequant_i2s_block_scalar ──────────────────────────────────────

    #[test]
    fn test_scalar_block_known() {
        let packed = [pack4([1, -1, 0, 1])];
        let mut out = [0.0f32; 4];
        dequant_i2s_block_scalar(&packed, 2.0, 4, &mut out);
        assert_eq!(out, [2.0, -2.0, 0.0, 2.0]);
    }

    #[test]
    fn test_scalar_block_all_zeros() {
        let packed = [0u8; 2];
        let mut out = [99.0f32; 8];
        dequant_i2s_block_scalar(&packed, 5.0, 8, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_scalar_block_all_pos() {
        let packed = [0x55u8; 2];
        let mut out = [0.0f32; 8];
        dequant_i2s_block_scalar(&packed, 3.0, 8, &mut out);
        assert!(out.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_scalar_block_all_neg() {
        let packed = [0xFFu8; 2];
        let mut out = [0.0f32; 8];
        dequant_i2s_block_scalar(&packed, 1.5, 8, &mut out);
        assert!(out.iter().all(|&v| (v + 1.5).abs() < 1e-6));
    }

    #[test]
    fn test_scalar_block_partial() {
        let packed = [pack4([1, -1, 1, 0])];
        let mut out = [0.0f32; 3];
        dequant_i2s_block_scalar(&packed, 1.0, 3, &mut out);
        assert_eq!(out, [1.0, -1.0, 1.0]);
    }

    // ── dequant_i2s_block_v2 (dispatched) ─────────────────────────────

    #[test]
    fn test_v2_block_matches_reference_4() {
        let packed = [pack4([1, -1, 0, 1])];
        let mut out = [0.0f32; 4];
        dequant_i2s_block_v2(&packed, 2.5, 4, &mut out);
        let expected = reference_dequant(&packed, 2.5, 4);
        assert_f32_eq(&out, &expected, 1e-6);
    }

    #[test]
    fn test_v2_block_matches_reference_16() {
        let packed = [0x55, 0xFF, 0x00, 0xAA];
        let mut out = [0.0f32; 16];
        dequant_i2s_block_v2(&packed, 1.0, 16, &mut out);
        let expected = reference_dequant(&packed, 1.0, 16);
        assert_f32_eq(&out, &expected, 1e-6);
    }

    #[test]
    fn test_v2_block_matches_reference_32() {
        let packed: Vec<u8> = (0..8).map(|i| pack4([1, -1, 0, (i % 3 - 1) as i8])).collect();
        let mut out = [0.0f32; 32];
        dequant_i2s_block_v2(&packed, 0.75, 32, &mut out);
        let expected = reference_dequant(&packed, 0.75, 32);
        assert_f32_eq(&out, &expected, 1e-6);
    }

    #[test]
    fn test_v2_block_17_elements() {
        let packed = [0x55u8; 5]; // 20 values, only use 17
        let mut out = [0.0f32; 17];
        dequant_i2s_block_v2(&packed, 1.0, 17, &mut out);
        let expected = reference_dequant(&packed, 1.0, 17);
        assert_f32_eq(&out, &expected, 1e-6);
    }

    #[test]
    fn test_v2_block_1_element() {
        let packed = [pack4([1, 0, 0, 0])];
        let mut out = [0.0f32; 1];
        dequant_i2s_block_v2(&packed, 4.0, 1, &mut out);
        assert_eq!(out[0], 4.0);
    }

    #[test]
    fn test_v2_block_negative_scale() {
        let packed = [pack4([1, -1, 0, 1])];
        let mut out = [0.0f32; 4];
        dequant_i2s_block_v2(&packed, -2.0, 4, &mut out);
        assert_eq!(out, [-2.0, 2.0, 0.0, -2.0]);
    }

    #[test]
    fn test_v2_block_zero_scale() {
        let packed = [0xFFu8; 4];
        let mut out = [99.0f32; 16];
        dequant_i2s_block_v2(&packed, 0.0, 16, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    // ── dequant_block32 ───────────────────────────────────────────────

    #[test]
    fn test_block32_basic() {
        let packed = [0x55u8; 8];
        let mut out = [0.0f32; 32];
        dequant_block32(&packed, 1.0, &mut out);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_block32_matches_reference() {
        let packed: Vec<u8> = (0..8).map(|i| (i * 37) as u8).collect();
        let mut out = [0.0f32; 32];
        dequant_block32(&packed, 3.14, &mut out);
        let expected = reference_dequant(&packed, 3.14, 32);
        assert_f32_eq(&out, &expected, 1e-5);
    }

    // ── dequant_block_qk256 ──────────────────────────────────────────

    #[test]
    fn test_block_qk256_basic() {
        let packed = [0x55u8; 64];
        let mut out = [0.0f32; 256];
        dequant_block_qk256(&packed, 1.0, &mut out);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_block_qk256_matches_reference() {
        let packed: Vec<u8> = (0..64).map(|i| (i * 13 + 7) as u8).collect();
        let mut out = [0.0f32; 256];
        dequant_block_qk256(&packed, 0.5, &mut out);
        let expected = reference_dequant(&packed, 0.5, 256);
        assert_f32_eq(&out, &expected, 1e-6);
    }

    #[test]
    fn test_block_qk256_all_neg() {
        let packed = [0xFFu8; 64];
        let mut out = [0.0f32; 256];
        dequant_block_qk256(&packed, 2.0, &mut out);
        assert!(out.iter().all(|&v| (v + 2.0).abs() < 1e-6));
    }

    // ── dequant_row_blocked ──────────────────────────────────────────

    #[test]
    fn test_row_blocked_single_block() {
        let packed = [pack4([1, -1, 0, 1])];
        let scales = [2.0f32];
        let mut out = [0.0f32; 4];
        dequant_row_blocked(&packed, &scales, 4, 4, &mut out);
        assert_eq!(out, [2.0, -2.0, 0.0, 2.0]);
    }

    #[test]
    fn test_row_blocked_two_blocks() {
        let packed = [pack4([1, -1, 0, 1]), pack4([-1, 1, 1, 0])];
        let scales = [2.0f32, 3.0];
        let mut out = [0.0f32; 8];
        dequant_row_blocked(&packed, &scales, 4, 8, &mut out);
        assert_eq!(out, [2.0, -2.0, 0.0, 2.0, -3.0, 3.0, 3.0, 0.0]);
    }

    #[test]
    fn test_row_blocked_block32() {
        let packed = vec![0x55u8; 16]; // 64 elements, 2 blocks of 32
        let scales = [1.0f32, 2.0];
        let mut out = vec![0.0f32; 64];
        dequant_row_blocked(&packed, &scales, 32, 64, &mut out);
        assert!(out[..32].iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(out[32..].iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_row_block32_convenience() {
        let packed = vec![0x55u8; 8];
        let scales = [1.5f32];
        let mut out = vec![0.0f32; 32];
        dequant_row_block32(&packed, &scales, 32, &mut out);
        assert!(out.iter().all(|&v| (v - 1.5).abs() < 1e-6));
    }

    #[test]
    fn test_row_qk256_convenience() {
        let packed = vec![0xFFu8; 64];
        let scales = [0.5f32];
        let mut out = vec![0.0f32; 256];
        dequant_row_qk256(&packed, &scales, 256, &mut out);
        assert!(out.iter().all(|&v| (v + 0.5).abs() < 1e-6));
    }

    // ── dequant_row_to_vec ───────────────────────────────────────────

    #[test]
    fn test_row_to_vec() {
        let packed = [pack4([1, 0, -1, 1])];
        let scales = [3.0f32];
        let out = dequant_row_to_vec(&packed, &scales, 4, 4);
        assert_eq!(out, vec![3.0, 0.0, -3.0, 3.0]);
    }

    #[test]
    fn test_row_to_vec_multi_block() {
        let packed = vec![0x55u8; 16];
        let scales = vec![1.0f32; 2];
        let out = dequant_row_to_vec(&packed, &scales, 32, 64);
        assert_eq!(out.len(), 64);
    }

    // ── dequant_batch_rows ───────────────────────────────────────────

    #[test]
    fn test_batch_rows_2x4() {
        let packed = [pack4([1, -1, 0, 1]), pack4([-1, 0, 1, -1])];
        let s0 = [2.0f32];
        let s1 = [3.0f32];
        let scales: Vec<&[f32]> = vec![&s0, &s1];
        let mut out = [0.0f32; 8];
        dequant_batch_rows(&packed, &scales, 4, 4, 2, &mut out);
        assert_eq!(out[..4], [2.0, -2.0, 0.0, 2.0]);
        assert_eq!(out[4..], [-3.0, 0.0, 3.0, -3.0]);
    }

    #[test]
    fn test_batch_rows_4x8() {
        let packed = vec![0x55u8; 8]; // 4 rows × 2 bytes
        let s = [1.0f32, 1.0];
        let scales: Vec<&[f32]> = vec![&s; 4];
        let mut out = vec![0.0f32; 32];
        dequant_batch_rows(&packed, &scales, 4, 8, 4, &mut out);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    // ── dequant_batch_rows_flat_scales ───────────────────────────────

    #[test]
    fn test_batch_flat_scales() {
        let packed = [pack4([1, 1, 1, 1]), pack4([-1, -1, -1, -1])];
        let scales = [2.0f32, 3.0]; // row0 scale=2, row1 scale=3
        let mut out = [0.0f32; 8];
        dequant_batch_rows_flat_scales(&packed, &scales, 4, 4, 2, &mut out);
        assert!(out[..4].iter().all(|&v| (v - 2.0).abs() < 1e-6));
        assert!(out[4..].iter().all(|&v| (v + 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_batch_flat_scales_multi_block() {
        let packed = vec![0x55u8; 16]; // 2 rows × 8 bytes = 2×32 elements
        let scales = [1.0f32, 2.0, 3.0, 4.0]; // 2 rows × 2 blocks of 16
        let mut out = vec![0.0f32; 64];
        dequant_batch_rows_flat_scales(&packed, &scales, 16, 32, 2, &mut out);
        assert!(out[..16].iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(out[16..32].iter().all(|&v| (v - 2.0).abs() < 1e-6));
        assert!(out[32..48].iter().all(|&v| (v - 3.0).abs() < 1e-6));
        assert!(out[48..64].iter().all(|&v| (v - 4.0).abs() < 1e-6));
    }

    // ── apply_scale_inplace ──────────────────────────────────────────

    #[test]
    fn test_scale_inplace_basic() {
        let mut data = vec![1.0, -1.0, 0.0, 2.0];
        apply_scale_inplace(&mut data, 3.0);
        assert_eq!(data, vec![3.0, -3.0, 0.0, 6.0]);
    }

    #[test]
    fn test_scale_inplace_zero() {
        let mut data = vec![5.0; 8];
        apply_scale_inplace(&mut data, 0.0);
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_scale_inplace_one() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut data = original.clone();
        apply_scale_inplace(&mut data, 1.0);
        assert_eq!(data, original);
    }

    #[test]
    fn test_scale_inplace_negative() {
        let mut data = vec![1.0, -2.0, 3.0];
        apply_scale_inplace(&mut data, -1.0);
        assert_eq!(data, vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_scale_inplace_odd_len() {
        let mut data = vec![2.0; 7];
        apply_scale_inplace(&mut data, 0.5);
        assert!(data.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    // ── apply_scale ──────────────────────────────────────────────────

    #[test]
    fn test_scale_out_of_place() {
        let data = vec![1.0, -1.0, 0.0, 2.0, -2.0];
        let mut out = vec![0.0f32; 5];
        apply_scale(&data, 3.0, &mut out);
        assert_eq!(out, vec![3.0, -3.0, 0.0, 6.0, -6.0]);
    }

    #[test]
    fn test_scale_out_of_place_large() {
        let data: Vec<f32> = (0..33).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; 33];
        apply_scale(&data, 2.0, &mut out);
        for i in 0..33 {
            assert!((out[i] - i as f32 * 2.0).abs() < 1e-5, "mismatch at {i}");
        }
    }

    // ── zero_point ───────────────────────────────────────────────────

    #[test]
    fn test_zero_point_basic() {
        let packed = [pack4([1, -1, 0, 1])];
        let mut out = [0.0f32; 4];
        dequant_with_zero_point(&packed, 2.0, 0.0, 4, &mut out);
        // zero_point=0 → same as normal dequant * scale
        assert_eq!(out, [2.0, -2.0, 0.0, 2.0]);
    }

    #[test]
    fn test_zero_point_nonzero() {
        let packed = [pack4([1, -1, 0, 1])];
        let mut out = [0.0f32; 4];
        // dequant values: [1, -1, 0, 1], zero_point=0.5, scale=2
        // result: [(1-0.5)*2, (-1-0.5)*2, (0-0.5)*2, (1-0.5)*2]
        //       = [1.0, -3.0, -1.0, 1.0]
        dequant_with_zero_point(&packed, 2.0, 0.5, 4, &mut out);
        assert_f32_eq(&out, &[1.0, -3.0, -1.0, 1.0], 1e-6);
    }

    #[test]
    fn test_zero_point_all_zeros_input() {
        let packed = [0x00u8; 2];
        let mut out = [0.0f32; 8];
        // dequant → all 0.0, then (0 - 1.0) * 3.0 = -3.0
        dequant_with_zero_point(&packed, 3.0, 1.0, 8, &mut out);
        assert!(out.iter().all(|&v| (v + 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_zero_point_odd_count() {
        let packed = [pack4([1, 0, -1, 0])];
        let mut out = [0.0f32; 3];
        dequant_with_zero_point(&packed, 1.0, 0.0, 3, &mut out);
        assert_f32_eq(&out, &[1.0, 0.0, -1.0], 1e-6);
    }

    // ── interleaved layout ───────────────────────────────────────────

    #[test]
    fn test_interleaved_roundtrip_identity() {
        // 4 rows of 4 elements each
        let row0 = pack4([1, 0, -1, 1]);
        let row1 = pack4([-1, 1, 0, 0]);
        let row2 = pack4([0, -1, 1, -1]);
        let row3 = pack4([1, 1, -1, 0]);
        let packed_rows = vec![row0, row1, row2, row3];

        let interleaved = pack_interleaved_4row(&packed_rows, 4, 4);
        assert_eq!(interleaved.len(), 4);

        let scales = vec![1.0f32; 4]; // 4 rows × 1 block each
        let mut out = vec![0.0f32; 16];
        dequant_interleaved_4row(&interleaved, &scales, 4, 4, &mut out);

        // Compare against direct scalar dequant
        let mut expected = [0.0f32; 16];
        for (i, &byte) in packed_rows.iter().enumerate() {
            dequant_i2s_block_scalar(&[byte], 1.0, 4, &mut expected[i * 4..]);
        }
        assert_f32_eq(&out, &expected, 1e-6);
    }

    #[test]
    fn test_interleaved_8_rows() {
        let packed_rows = vec![0x55u8; 8]; // 8 rows × 1 byte (4 elem)
        let interleaved = pack_interleaved_4row(&packed_rows, 4, 8);
        let scales = vec![1.0f32; 8];
        let mut out = vec![0.0f32; 32];
        dequant_interleaved_4row(&interleaved, &scales, 4, 8, &mut out);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_interleaved_with_different_scales() {
        let packed_rows = vec![0x55u8; 4]; // 4 rows, all +1
        let interleaved = pack_interleaved_4row(&packed_rows, 4, 4);
        let scales = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 16];
        dequant_interleaved_4row(&interleaved, &scales, 4, 4, &mut out);
        assert!(out[0..4].iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(out[4..8].iter().all(|&v| (v - 2.0).abs() < 1e-6));
        assert!(out[8..12].iter().all(|&v| (v - 3.0).abs() < 1e-6));
        assert!(out[12..16].iter().all(|&v| (v - 4.0).abs() < 1e-6));
    }

    // ── pack_interleaved_4row ────────────────────────────────────────

    #[test]
    fn test_pack_interleaved_preserves_size() {
        let packed = vec![0u8; 32]; // 8 rows × 4 bytes
        let interleaved = pack_interleaved_4row(&packed, 16, 8);
        assert_eq!(interleaved.len(), packed.len());
    }

    #[test]
    #[should_panic(expected = "num_rows must be multiple of 4")]
    fn test_pack_interleaved_bad_rows() {
        let packed = vec![0u8; 3];
        let _ = pack_interleaved_4row(&packed, 4, 3);
    }

    // ── unpack_to_i8 ─────────────────────────────────────────────────

    #[test]
    fn test_unpack_to_i8_basic() {
        let packed = [pack4([1, -1, 0, 1])];
        let mut out = [0i8; 4];
        unpack_to_i8(&packed, 4, &mut out);
        assert_eq!(out, [1, -1, 0, 1]);
    }

    #[test]
    fn test_unpack_to_i8_partial() {
        let packed = [pack4([1, -1, 0, 1])];
        let mut out = [0i8; 3];
        unpack_to_i8(&packed, 3, &mut out);
        assert_eq!(out, [1, -1, 0]);
    }

    #[test]
    fn test_unpack_to_i8_vec() {
        let packed = [0x55u8; 4];
        let out = unpack_to_i8_vec(&packed, 16);
        assert_eq!(out.len(), 16);
        assert!(out.iter().all(|&v| v == 1));
    }

    // ── pack_from_i8 ─────────────────────────────────────────────────

    #[test]
    fn test_pack_from_i8_basic() {
        let values = [1i8, -1, 0, 1];
        let packed = pack_from_i8(&values);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], pack4([1, -1, 0, 1]));
    }

    #[test]
    fn test_pack_from_i8_roundtrip() {
        let original = vec![1i8, -1, 0, 1, 0, -1, 1, -1];
        let packed = pack_from_i8(&original);
        let unpacked = unpack_to_i8_vec(&packed, original.len());
        assert_eq!(unpacked, original);
    }

    #[test]
    fn test_pack_from_i8_partial() {
        let values = [1i8, -1, 0]; // 3 values → 1 byte
        let packed = pack_from_i8(&values);
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_to_i8_vec(&packed, 3);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_pack_unpack_all_combos() {
        // Test all 81 possible 4-value ternary combinations
        let ternary = [-1i8, 0, 1];
        for &a in &ternary {
            for &b in &ternary {
                for &c in &ternary {
                    for &d in &ternary {
                        let vals = [a, b, c, d];
                        let packed = pack_from_i8(&vals);
                        let unpacked = unpack_to_i8_vec(&packed, 4);
                        assert_eq!(unpacked, vals, "roundtrip failed for [{a},{b},{c},{d}]");
                    }
                }
            }
        }
    }

    // ── dequant_i2s_block_accumulate ─────────────────────────────────

    #[test]
    fn test_accumulate_zeros_acc() {
        let packed = [pack4([1, -1, 0, 1])];
        let mut acc = [0.0f32; 4];
        dequant_i2s_block_accumulate(&packed, 2.0, 4, &mut acc);
        assert_eq!(acc, [2.0, -2.0, 0.0, 2.0]);
    }

    #[test]
    fn test_accumulate_existing_values() {
        let packed = [pack4([1, -1, 0, 1])];
        let mut acc = [10.0f32; 4];
        dequant_i2s_block_accumulate(&packed, 2.0, 4, &mut acc);
        assert_eq!(acc, [12.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_accumulate_multiple() {
        let p1 = [pack4([1, 0, 0, 0])];
        let p2 = [pack4([0, 1, 0, 0])];
        let p3 = [pack4([0, 0, 1, 0])];
        let mut acc = [0.0f32; 4];
        dequant_i2s_block_accumulate(&p1, 1.0, 4, &mut acc);
        dequant_i2s_block_accumulate(&p2, 1.0, 4, &mut acc);
        dequant_i2s_block_accumulate(&p3, 1.0, 4, &mut acc);
        assert_eq!(acc, [1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_accumulate_16_elements() {
        let packed = vec![0x55u8; 4]; // 16 all-positive
        let mut acc = vec![1.0f32; 16];
        dequant_i2s_block_accumulate(&packed, 2.0, 16, &mut acc);
        assert!(acc.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_accumulate_32_elements() {
        let packed = vec![0xFFu8; 8]; // 32 all-negative
        let mut acc = vec![5.0f32; 32];
        dequant_i2s_block_accumulate(&packed, 1.0, 32, &mut acc);
        assert!(acc.iter().all(|&v| (v - 4.0).abs() < 1e-6));
    }

    #[test]
    fn test_accumulate_matches_dequant_plus_add() {
        let packed: Vec<u8> = (0..5).map(|i| (i * 0x37) as u8).collect();
        let scale = 1.5;
        let n = 20;
        let initial = vec![7.0f32; n];

        let mut via_acc = initial.clone();
        dequant_i2s_block_accumulate(&packed, scale, n, &mut via_acc);

        let dequant = reference_dequant(&packed, scale, n);
        let via_add: Vec<f32> = initial.iter().zip(dequant.iter()).map(|(a, d)| a + d).collect();

        assert_f32_eq(&via_acc, &via_add, 1e-6);
    }

    // ── count_ternary_values ─────────────────────────────────────────

    #[test]
    fn test_count_all_zeros() {
        let packed = [0x00u8; 4];
        let (p, n, z) = count_ternary_values(&packed, 16);
        assert_eq!((p, n, z), (0, 0, 16));
    }

    #[test]
    fn test_count_all_positive() {
        let packed = [0x55u8; 2];
        let (p, n, z) = count_ternary_values(&packed, 8);
        assert_eq!((p, n, z), (8, 0, 0));
    }

    #[test]
    fn test_count_all_negative() {
        let packed = [0xFFu8; 2];
        let (p, n, z) = count_ternary_values(&packed, 8);
        assert_eq!((p, n, z), (0, 8, 0));
    }

    #[test]
    fn test_count_mixed() {
        let packed = [pack4([1, -1, 0, 1]), pack4([0, -1, -1, 0])];
        let (p, n, z) = count_ternary_values(&packed, 8);
        assert_eq!((p, n, z), (2, 3, 3));
    }

    #[test]
    fn test_count_partial() {
        let packed = [pack4([1, -1, 0, 1])];
        let (p, n, z) = count_ternary_values(&packed, 3);
        assert_eq!((p, n, z), (1, 1, 1));
    }

    // ── Constants ────────────────────────────────────────────────────

    #[test]
    fn test_block_size_constants() {
        assert_eq!(BLOCK_SIZE_32, 32);
        assert_eq!(BLOCK_SIZE_QK256, 256);
        assert_eq!(BYTES_PER_BLOCK_32, 8);
        assert_eq!(BYTES_PER_BLOCK_QK256, 64);
    }

    // ── Edge cases and panics ────────────────────────────────────────

    #[test]
    #[should_panic(expected = "not enough packed bytes")]
    fn test_v2_block_too_few_bytes() {
        let packed = [0u8; 1]; // only 4 elements
        let mut out = [0.0f32; 8];
        dequant_i2s_block_v2(&packed, 1.0, 8, &mut out);
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn test_v2_block_output_too_small() {
        let packed = [0u8; 4];
        let mut out = [0.0f32; 4]; // need 16
        dequant_i2s_block_v2(&packed, 1.0, 16, &mut out);
    }

    #[test]
    #[should_panic(expected = "block_size must be > 0")]
    fn test_row_blocked_zero_block_size() {
        let packed = [0u8; 4];
        let scales = [1.0f32];
        let mut out = [0.0f32; 16];
        dequant_row_blocked(&packed, &scales, 0, 16, &mut out);
    }

    #[test]
    #[should_panic(expected = "not enough scales")]
    fn test_row_blocked_too_few_scales() {
        let packed = [0u8; 4];
        let scales = [1.0f32]; // need 4 for block_size=4
        let mut out = [0.0f32; 16];
        dequant_row_blocked(&packed, &scales, 4, 16, &mut out);
    }

    #[test]
    #[should_panic(expected = "num_rows must be multiple of 4")]
    fn test_interleaved_bad_num_rows() {
        let packed = [0u8; 12];
        let scales = [1.0f32; 3];
        let mut out = [0.0f32; 12];
        dequant_interleaved_4row(&packed, &scales, 4, 3, &mut out);
    }

    // ── Large-ish correctness checks ─────────────────────────────────

    #[test]
    fn test_v2_block_256_matches_reference() {
        let packed: Vec<u8> = (0..64).map(|i| (i * 41 + 3) as u8).collect();
        let mut out = vec![0.0f32; 256];
        dequant_i2s_block_v2(&packed, 1.23, 256, &mut out);
        let expected = reference_dequant(&packed, 1.23, 256);
        assert_f32_eq(&out, &expected, 1e-5);
    }

    #[test]
    fn test_v2_block_1024_matches_reference() {
        let packed: Vec<u8> = (0..256).map(|i| (i * 17 + 5) as u8).collect();
        let mut out = vec![0.0f32; 1024];
        dequant_i2s_block_v2(&packed, 0.001, 1024, &mut out);
        let expected = reference_dequant(&packed, 0.001, 1024);
        assert_f32_eq(&out, &expected, 1e-7);
    }

    #[test]
    fn test_batch_8_rows_32_elem_matches_scalar() {
        let num_rows = 8;
        let elements = 32;
        let bytes_per_row = 8;
        let packed: Vec<u8> =
            (0..(num_rows * bytes_per_row) as u8).map(|i| i.wrapping_mul(37)).collect();
        let all_scales: Vec<f32> = (0..num_rows).map(|i| 0.5 + i as f32 * 0.1).collect();

        let mut batch_out = vec![0.0f32; num_rows * elements];
        dequant_batch_rows_flat_scales(
            &packed,
            &all_scales,
            32,
            elements,
            num_rows,
            &mut batch_out,
        );

        // Compare row-by-row against scalar reference
        for row in 0..num_rows {
            let start = row * bytes_per_row;
            let expected =
                reference_dequant(&packed[start..start + bytes_per_row], all_scales[row], elements);
            let row_start = row * elements;
            assert_f32_eq(&batch_out[row_start..row_start + elements], &expected, 1e-5);
        }
    }

    #[test]
    fn test_zero_point_large_block() {
        let packed = vec![0x55u8; 8]; // 32 all-positive
        let mut out = vec![0.0f32; 32];
        dequant_with_zero_point(&packed, 2.0, 1.0, 32, &mut out);
        // dequant=1.0, (1.0 - 1.0)*2.0 = 0.0
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn test_empty_operations() {
        // 0-element operations shouldn't panic
        let packed: &[u8] = &[];
        let mut out: Vec<f32> = vec![];
        dequant_i2s_block_v2(packed, 1.0, 0, &mut out);
        assert!(out.is_empty());
    }
}
