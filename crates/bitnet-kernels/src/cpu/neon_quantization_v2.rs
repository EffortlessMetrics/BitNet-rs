//! NEON-optimized quantization v2 kernel for Apple Silicon (aarch64).
//!
//! Provides six quantization operations with NEON intrinsics and scalar
//! fallbacks:
//!
//! 1. `quantize_f32_to_i2` — f32 → 2-bit packed (I2_S) with per-block scales
//! 2. `dequantize_i2_to_f32` — 2-bit packed → f32
//! 3. `quantize_f32_to_i8` — symmetric int8 quantization
//! 4. `dequantize_i8_to_f32` — int8 → f32
//! 5. `requantize_i8_to_i2` — int8 → 2-bit ternary (-1/0/+1)
//! 6. `compute_block_scales` — per-block absmax scales
//!
//! I2_S encoding (2 bits per value, 4 values per byte, LSB-first):
//! - `0b00` → 0
//! - `0b01` → +1
//! - `0b11` → −1
//! - `0b10` → unused (treated as 0)

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::missing_safety_doc,
    clippy::float_cmp,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::let_and_return,
    clippy::derivable_impls,
    clippy::excessive_precision,
    clippy::manual_is_multiple_of,
    clippy::manual_memcpy,
    dead_code,
    unused_unsafe
)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── I2_S encode/decode helpers ─────────────────────────────────────

/// Encode a ternary value (-1, 0, +1) to its 2-bit I2_S code.
#[inline(always)]
fn encode_i2s(val: i8) -> u8 {
    match val {
        1 => 0b01,
        -1 => 0b11,
        _ => 0b00,
    }
}

/// Decode a 2-bit I2_S code to its ternary f32 value.
#[inline(always)]
fn decode_i2s(bits: u8) -> f32 {
    match bits & 0x03 {
        0b01 => 1.0,
        0b11 => -1.0,
        _ => 0.0,
    }
}

/// Pack 4 ternary values (each -1, 0, or +1 as i8) into one byte.
#[inline(always)]
fn pack_4_i2s(vals: &[i8]) -> u8 {
    let v0 = encode_i2s(vals[0]);
    let v1 = encode_i2s(vals[1]);
    let v2 = encode_i2s(vals[2]);
    let v3 = encode_i2s(vals[3]);
    v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
}

/// Unpack one byte into 4 f32 ternary values.
#[inline(always)]
fn unpack_4_i2s(byte: u8) -> [f32; 4] {
    [decode_i2s(byte), decode_i2s(byte >> 2), decode_i2s(byte >> 4), decode_i2s(byte >> 6)]
}

// ── 1. quantize_f32_to_i2 ─────────────────────────────────────────

/// Quantize f32 values to 2-bit I2_S packed format with per-block scales.
///
/// Each block of `block_size` values gets a scale = absmax of the block.
/// Values are mapped to ternary: v/scale rounds to -1, 0, or +1.
/// 4 ternary values are packed per byte (LSB-first).
///
/// `output` must have length `ceil(input.len() / 4)`.
/// `scales` must have length `ceil(input.len() / block_size)`.
pub fn quantize_f32_to_i2(input: &[f32], output: &mut [u8], scales: &mut [f32], block_size: usize) {
    assert!(block_size == 32 || block_size == 256, "block_size must be 32 or 256");
    let n = input.len();
    assert!(output.len() >= (n + 3) / 4);
    let num_blocks = (n + block_size - 1) / block_size;
    assert!(scales.len() >= num_blocks);

    #[cfg(target_arch = "aarch64")]
    {
        // Safety: aarch64 always has NEON.
        unsafe { neon_quantize_f32_to_i2(input, output, scales, block_size) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_quantize_f32_to_i2(input, output, scales, block_size);
    }
}

/// Scalar fallback for f32 → I2_S quantization.
fn scalar_quantize_f32_to_i2(
    input: &[f32],
    output: &mut [u8],
    scales: &mut [f32],
    block_size: usize,
) {
    let n = input.len();

    // Compute per-block scales.
    let num_blocks = (n + block_size - 1) / block_size;
    for b in 0..num_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(n);
        let mut abs_max: f32 = 0.0;
        for &v in &input[start..end] {
            abs_max = abs_max.max(v.abs());
        }
        scales[b] = abs_max;
    }

    // Quantize to ternary and pack.
    output.iter_mut().for_each(|b| *b = 0);
    for i in 0..n {
        let block_idx = i / block_size;
        let scale = scales[block_idx];
        let ternary = if scale == 0.0 {
            0i8
        } else {
            let normalized = input[i] / scale;
            if normalized > 0.5 {
                1
            } else if normalized < -0.5 {
                -1
            } else {
                0
            }
        };
        let byte_idx = i / 4;
        let bit_pos = (i % 4) * 2;
        output[byte_idx] |= encode_i2s(ternary) << bit_pos;
    }
}

/// NEON-accelerated f32 → I2_S quantization.
///
/// # Safety
///
/// Caller must ensure `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_quantize_f32_to_i2(
    input: &[f32],
    output: &mut [u8],
    scales: &mut [f32],
    block_size: usize,
) {
    let n = input.len();
    let num_blocks = (n + block_size - 1) / block_size;
    let ptr = input.as_ptr();

    // Phase 1: compute per-block scales with NEON absmax.
    for b in 0..num_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(n);
        let block_len = end - start;

        let mut acc = unsafe { vdupq_n_f32(0.0) };
        let chunks = block_len / 4;
        let rem = block_len % 4;

        for c in 0..chunks {
            let v = unsafe { vld1q_f32(ptr.add(start + c * 4)) };
            let abs_v = unsafe { vabsq_f32(v) };
            acc = unsafe { vmaxq_f32(acc, abs_v) };
        }
        let mut abs_max = unsafe { vmaxvq_f32(acc) };
        for r in 0..rem {
            abs_max = abs_max.max(input[start + chunks * 4 + r].abs());
        }
        scales[b] = abs_max;
    }

    // Phase 2: quantize to ternary and pack.
    output.iter_mut().for_each(|b| *b = 0);
    let half = unsafe { vdupq_n_f32(0.5) };
    let neg_half = unsafe { vdupq_n_f32(-0.5) };

    // Process 4 values at a time (one packed byte).
    let full_quads = n / 4;
    for q in 0..full_quads {
        let base = q * 4;
        let v = unsafe { vld1q_f32(ptr.add(base)) };

        // Determine scale for each of the 4 values.
        // All 4 may be in the same block or span two blocks.
        let block_idx = base / block_size;
        let scale = scales[block_idx];

        if scale == 0.0 {
            output[q] = 0;
            continue;
        }

        let inv_scale = 1.0 / scale;
        // If the quad spans a block boundary, fall back to scalar per-element.
        let end_block = (base + 3) / block_size;
        if end_block != block_idx {
            let mut byte = 0u8;
            for k in 0..4 {
                let bi = (base + k) / block_size;
                let s = scales[bi];
                let t = if s == 0.0 {
                    0i8
                } else {
                    let norm = input[base + k] / s;
                    if norm > 0.5 {
                        1
                    } else if norm < -0.5 {
                        -1
                    } else {
                        0
                    }
                };
                byte |= encode_i2s(t) << (k * 2);
            }
            output[q] = byte;
            continue;
        }

        let sv = unsafe { vdupq_n_f32(inv_scale) };
        let normalized = unsafe { vmulq_f32(v, sv) };

        // Compare: > 0.5 → +1, < -0.5 → -1, else 0.
        let pos_mask = unsafe { vcgtq_f32(normalized, half) };
        let neg_mask = unsafe { vcltq_f32(normalized, neg_half) };

        // Extract lane results.
        let mut byte = 0u8;
        let pos_bits: [u32; 4] = unsafe { std::mem::transmute(pos_mask) };
        let neg_bits: [u32; 4] = unsafe { std::mem::transmute(neg_mask) };
        for k in 0..4 {
            let code = if pos_bits[k] != 0 {
                0b01u8 // +1
            } else if neg_bits[k] != 0 {
                0b11u8 // -1
            } else {
                0b00u8 // 0
            };
            byte |= code << (k * 2);
        }
        output[q] = byte;
    }

    // Handle remaining values (0..3 tail).
    let tail_start = full_quads * 4;
    if tail_start < n {
        let byte_idx = full_quads;
        let mut byte = 0u8;
        for k in 0..(n - tail_start) {
            let i = tail_start + k;
            let bi = i / block_size;
            let s = scales[bi];
            let t = if s == 0.0 {
                0i8
            } else {
                let norm = input[i] / s;
                if norm > 0.5 {
                    1
                } else if norm < -0.5 {
                    -1
                } else {
                    0
                }
            };
            byte |= encode_i2s(t) << (k * 2);
        }
        output[byte_idx] = byte;
    }
}

// ── 2. dequantize_i2_to_f32 ───────────────────────────────────────

/// Dequantize 2-bit I2_S packed values to f32 using per-block scales.
///
/// `input` has packed bytes (4 values per byte).
/// `output` receives the dequantized f32 values (length must match element count).
/// `block_size` must be 32 or 256.
pub fn dequantize_i2_to_f32(input: &[u8], scales: &[f32], output: &mut [f32], block_size: usize) {
    assert!(block_size == 32 || block_size == 256, "block_size must be 32 or 256");
    let n = output.len();
    assert!(input.len() >= (n + 3) / 4);
    let num_blocks = (n + block_size - 1) / block_size;
    assert!(scales.len() >= num_blocks);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_dequantize_i2_to_f32(input, scales, output, block_size) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_dequantize_i2_to_f32(input, scales, output, block_size);
    }
}

/// Scalar fallback for I2_S → f32 dequantization.
fn scalar_dequantize_i2_to_f32(
    input: &[u8],
    scales: &[f32],
    output: &mut [f32],
    block_size: usize,
) {
    let n = output.len();
    for i in 0..n {
        let byte_idx = i / 4;
        let bit_pos = (i % 4) * 2;
        let code = (input[byte_idx] >> bit_pos) & 0x03;
        let ternary = decode_i2s(code);
        let block_idx = i / block_size;
        output[i] = ternary * scales[block_idx];
    }
}

/// NEON-accelerated I2_S → f32 dequantization.
///
/// # Safety
///
/// Caller must ensure `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dequantize_i2_to_f32(
    input: &[u8],
    scales: &[f32],
    output: &mut [f32],
    block_size: usize,
) {
    let n = output.len();
    let out_ptr = output.as_mut_ptr();

    // Process 4 values (1 byte) at a time.
    let full_bytes = n / 4;
    for b in 0..full_bytes {
        let byte = input[b];
        let base = b * 4;

        // Decode 4 ternary values.
        let t0 = decode_i2s(byte) as f32;
        let t1 = decode_i2s(byte >> 2) as f32;
        let t2 = decode_i2s(byte >> 4) as f32;
        let t3 = decode_i2s(byte >> 6) as f32;

        // Check if all 4 values are in the same block.
        let block_idx = base / block_size;
        let end_block = (base + 3) / block_size;

        if block_idx == end_block {
            // Same block: broadcast scale and multiply.
            let scale_vec = unsafe { vdupq_n_f32(scales[block_idx]) };
            let ternary_arr: [f32; 4] = [t0, t1, t2, t3];
            let ternary_vec = unsafe { vld1q_f32(ternary_arr.as_ptr()) };
            let result = unsafe { vmulq_f32(ternary_vec, scale_vec) };
            unsafe { vst1q_f32(out_ptr.add(base), result) };
        } else {
            // Cross-block boundary: per-element.
            for k in 0..4 {
                let idx = base + k;
                let bi = idx / block_size;
                let tv = [t0, t1, t2, t3][k];
                unsafe { *out_ptr.add(idx) = tv * scales[bi] };
            }
        }
    }

    // Handle tail (0..3 remaining values).
    let tail_start = full_bytes * 4;
    if tail_start < n {
        let byte = input[full_bytes];
        for k in 0..(n - tail_start) {
            let idx = tail_start + k;
            let code = (byte >> (k * 2)) & 0x03;
            let ternary = decode_i2s(code);
            let block_idx = idx / block_size;
            unsafe { *out_ptr.add(idx) = ternary * scales[block_idx] };
        }
    }
}

// ── 3. quantize_f32_to_i8 ─────────────────────────────────────────

/// Symmetric int8 quantization: maps f32 to [-127, 127] with a single scale.
///
/// `scale` is set to `absmax / 127.0`. Zero input produces scale=0 and zero output.
pub fn quantize_f32_to_i8(input: &[f32], output: &mut [i8], scale: &mut f32) {
    assert!(output.len() >= input.len());

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_quantize_f32_to_i8(input, output, scale) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_quantize_f32_to_i8(input, output, scale);
    }
}

/// Scalar fallback for symmetric f32 → i8 quantization.
fn scalar_quantize_f32_to_i8(input: &[f32], output: &mut [i8], scale: &mut f32) {
    let abs_max = input.iter().copied().fold(0.0_f32, |m, v| m.max(v.abs()));
    if abs_max == 0.0 {
        *scale = 0.0;
        output[..input.len()].fill(0);
        return;
    }
    *scale = abs_max / 127.0;
    let inv_scale = 127.0 / abs_max;
    for (i, &v) in input.iter().enumerate() {
        output[i] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
    }
}

/// NEON-accelerated symmetric f32 → i8 quantization.
///
/// # Safety
///
/// Caller must ensure `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_quantize_f32_to_i8(input: &[f32], output: &mut [i8], scale: &mut f32) {
    let n = input.len();
    let ptr = input.as_ptr();

    // Phase 1: find absmax with NEON.
    let mut acc = unsafe { vdupq_n_f32(0.0) };
    let chunks = n / 4;
    let rem = n % 4;

    for c in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(c * 4)) };
        let abs_v = unsafe { vabsq_f32(v) };
        acc = unsafe { vmaxq_f32(acc, abs_v) };
    }
    let mut abs_max = unsafe { vmaxvq_f32(acc) };
    for r in 0..rem {
        abs_max = abs_max.max(input[chunks * 4 + r].abs());
    }

    if abs_max == 0.0 {
        *scale = 0.0;
        output[..n].fill(0);
        return;
    }

    *scale = abs_max / 127.0;
    let inv_scale = 127.0 / abs_max;

    // Phase 2: quantize with NEON multiply + clamp.
    let inv_scale_v = unsafe { vdupq_n_f32(inv_scale) };
    let min_v = unsafe { vdupq_n_f32(-127.0) };
    let max_v = unsafe { vdupq_n_f32(127.0) };
    let out_ptr = output.as_mut_ptr();

    for c in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(c * 4)) };
        let scaled = unsafe { vmulq_f32(v, inv_scale_v) };
        // Round using vrndnq_f32 (round to nearest, ties to even).
        let rounded = unsafe { vrndnq_f32(scaled) };
        let clamped = unsafe { vminq_f32(vmaxq_f32(rounded, min_v), max_v) };

        // Convert f32 → i32 → i16 → i8.
        let i32_vals = unsafe { vcvtq_s32_f32(clamped) };
        let i16_vals = unsafe { vmovn_s32(i32_vals) };
        // We need to narrow further: i16 → i8. Use saturating narrow.
        // But vmovn_s16 needs two int16x4 halves to make int8x8.
        // For 4 values, duplicate the low half and narrow.
        let i16_combined = unsafe { vcombine_s16(i16_vals, i16_vals) };
        let i8_vals = unsafe { vmovn_s16(i16_combined) };

        // Store 4 i8 values.
        let base = c * 4;
        let arr: [i8; 8] = unsafe { std::mem::transmute(i8_vals) };
        for k in 0..4 {
            unsafe { *out_ptr.add(base + k) = arr[k] };
        }
    }

    // Tail scalar.
    for r in 0..rem {
        let idx = chunks * 4 + r;
        output[idx] = (input[idx] * inv_scale).round().clamp(-127.0, 127.0) as i8;
    }
}

// ── 4. dequantize_i8_to_f32 ───────────────────────────────────────

/// Dequantize int8 values to f32: `output[i] = input[i] as f32 * scale`.
pub fn dequantize_i8_to_f32(input: &[i8], scale: f32, output: &mut [f32]) {
    assert!(output.len() >= input.len());

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_dequantize_i8_to_f32(input, scale, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_dequantize_i8_to_f32(input, scale, output);
    }
}

/// Scalar fallback for i8 → f32 dequantization.
fn scalar_dequantize_i8_to_f32(input: &[i8], scale: f32, output: &mut [f32]) {
    for (i, &v) in input.iter().enumerate() {
        output[i] = v as f32 * scale;
    }
}

/// NEON-accelerated i8 → f32 dequantization using vcvtq_f32_s32.
///
/// # Safety
///
/// Caller must ensure `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dequantize_i8_to_f32(input: &[i8], scale: f32, output: &mut [f32]) {
    let n = input.len();
    let scale_v = unsafe { vdupq_n_f32(scale) };
    let out_ptr = output.as_mut_ptr();
    let in_ptr = input.as_ptr();

    let chunks = n / 4;
    let rem = n % 4;

    for c in 0..chunks {
        let base = c * 4;
        // Load 4 i8 values, widen to i32, convert to f32, multiply by scale.
        let vals: [i8; 4] = [
            unsafe { *in_ptr.add(base) },
            unsafe { *in_ptr.add(base + 1) },
            unsafe { *in_ptr.add(base + 2) },
            unsafe { *in_ptr.add(base + 3) },
        ];
        let i32_arr: [i32; 4] = [vals[0] as i32, vals[1] as i32, vals[2] as i32, vals[3] as i32];
        let i32_vec = unsafe { vld1q_s32(i32_arr.as_ptr()) };
        let f32_vec = unsafe { vcvtq_f32_s32(i32_vec) };
        let result = unsafe { vmulq_f32(f32_vec, scale_v) };
        unsafe { vst1q_f32(out_ptr.add(base), result) };
    }

    // Tail scalar.
    for r in 0..rem {
        let idx = chunks * 4 + r;
        output[idx] = input[idx] as f32 * scale;
    }
}

// ── 5. requantize_i8_to_i2 ────────────────────────────────────────

/// Requantize int8 values to 2-bit ternary I2_S (-1/0/+1).
///
/// Values with `|v| > threshold * 127` map to ±1, else 0.
/// `output` must have length `ceil(input.len() / 4)`.
pub fn requantize_i8_to_i2(input: &[i8], output: &mut [u8], threshold: f32) {
    assert!(output.len() >= (input.len() + 3) / 4);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_requantize_i8_to_i2(input, output, threshold) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_requantize_i8_to_i2(input, output, threshold);
    }
}

/// Scalar fallback for i8 → I2_S requantization.
fn scalar_requantize_i8_to_i2(input: &[i8], output: &mut [u8], threshold: f32) {
    let thresh_i8 = (threshold * 127.0).round() as i8;
    let n = input.len();
    output.iter_mut().for_each(|b| *b = 0);

    for i in 0..n {
        let v = input[i];
        let ternary = if v > thresh_i8 {
            1i8
        } else if v < -thresh_i8 {
            -1i8
        } else {
            0i8
        };
        let byte_idx = i / 4;
        let bit_pos = (i % 4) * 2;
        output[byte_idx] |= encode_i2s(ternary) << bit_pos;
    }
}

/// NEON-accelerated i8 → I2_S requantization.
///
/// # Safety
///
/// Caller must ensure `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_requantize_i8_to_i2(input: &[i8], output: &mut [u8], threshold: f32) {
    let thresh_i8 = (threshold * 127.0).round() as i8;
    let n = input.len();
    output.iter_mut().for_each(|b| *b = 0);

    let thresh_pos = unsafe { vdupq_n_s32(thresh_i8 as i32) };
    let thresh_neg = unsafe { vdupq_n_s32(-(thresh_i8 as i32)) };

    let full_quads = n / 4;
    let in_ptr = input.as_ptr();

    for q in 0..full_quads {
        let base = q * 4;
        let vals: [i32; 4] = [
            unsafe { *in_ptr.add(base) } as i32,
            unsafe { *in_ptr.add(base + 1) } as i32,
            unsafe { *in_ptr.add(base + 2) } as i32,
            unsafe { *in_ptr.add(base + 3) } as i32,
        ];
        let v = unsafe { vld1q_s32(vals.as_ptr()) };

        // Compare: v > threshold → positive, v < -threshold → negative.
        let pos_mask = unsafe { vcgtq_s32(v, thresh_pos) };
        let neg_mask = unsafe { vcltq_s32(v, thresh_neg) };

        let pos_bits: [u32; 4] = unsafe { std::mem::transmute(pos_mask) };
        let neg_bits: [u32; 4] = unsafe { std::mem::transmute(neg_mask) };

        let mut byte = 0u8;
        for k in 0..4 {
            let code = if pos_bits[k] != 0 {
                0b01u8
            } else if neg_bits[k] != 0 {
                0b11u8
            } else {
                0b00u8
            };
            byte |= code << (k * 2);
        }
        output[q] = byte;
    }

    // Tail.
    let tail_start = full_quads * 4;
    if tail_start < n {
        let byte_idx = full_quads;
        let mut byte = 0u8;
        for k in 0..(n - tail_start) {
            let v = input[tail_start + k];
            let ternary = if v > thresh_i8 {
                1i8
            } else if v < -thresh_i8 {
                -1i8
            } else {
                0i8
            };
            byte |= encode_i2s(ternary) << (k * 2);
        }
        output[byte_idx] = byte;
    }
}

// ── 6. compute_block_scales ────────────────────────────────────────

/// Compute per-block absmax scales using NEON horizontal max.
///
/// `scales` must have length `ceil(input.len() / block_size)`.
pub fn compute_block_scales(input: &[f32], scales: &mut [f32], block_size: usize) {
    assert!(block_size == 32 || block_size == 256, "block_size must be 32 or 256");
    let num_blocks = (input.len() + block_size - 1) / block_size;
    assert!(scales.len() >= num_blocks);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_compute_block_scales(input, scales, block_size) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_compute_block_scales(input, scales, block_size);
    }
}

/// Scalar fallback for block-scale computation.
fn scalar_compute_block_scales(input: &[f32], scales: &mut [f32], block_size: usize) {
    let n = input.len();
    let num_blocks = (n + block_size - 1) / block_size;
    for b in 0..num_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(n);
        let mut abs_max: f32 = 0.0;
        for &v in &input[start..end] {
            abs_max = abs_max.max(v.abs());
        }
        scales[b] = abs_max;
    }
}

/// NEON-accelerated per-block absmax scale computation.
///
/// # Safety
///
/// Caller must ensure `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_compute_block_scales(input: &[f32], scales: &mut [f32], block_size: usize) {
    let n = input.len();
    let num_blocks = (n + block_size - 1) / block_size;
    let ptr = input.as_ptr();

    for b in 0..num_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(n);
        let block_len = end - start;

        let mut acc = unsafe { vdupq_n_f32(0.0) };
        let chunks = block_len / 4;
        let rem = block_len % 4;

        for c in 0..chunks {
            let v = unsafe { vld1q_f32(ptr.add(start + c * 4)) };
            let abs_v = unsafe { vabsq_f32(v) };
            acc = unsafe { vmaxq_f32(acc, abs_v) };
        }
        let mut abs_max = unsafe { vmaxvq_f32(acc) };
        for r in 0..rem {
            abs_max = abs_max.max(input[start + chunks * 4 + r].abs());
        }
        scales[b] = abs_max;
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: compute relative or absolute error tolerance.
    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ── quantize_f32_to_i2 / dequantize_i2_to_f32 round-trip ──────

    #[test]
    fn test_i2_roundtrip_zeros_block32() {
        let input = vec![0.0f32; 32];
        let mut packed = vec![0u8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 32);
        assert_eq!(scales[0], 0.0);

        let mut output = vec![0.0f32; 32];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 32);
        for v in &output {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_i2_roundtrip_zeros_block256() {
        let input = vec![0.0f32; 256];
        let mut packed = vec![0u8; 64];
        let mut scales = vec![0.0f32; 1];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 256);
        assert_eq!(scales[0], 0.0);

        let mut output = vec![0.0f32; 256];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 256);
        for v in &output {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_i2_roundtrip_all_ones_block32() {
        let input = vec![1.0f32; 32];
        let mut packed = vec![0u8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 32);
        assert_eq!(scales[0], 1.0);

        let mut output = vec![0.0f32; 32];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 32);
        for v in &output {
            assert_eq!(*v, 1.0);
        }
    }

    #[test]
    fn test_i2_roundtrip_all_negative_block32() {
        let input = vec![-1.0f32; 32];
        let mut packed = vec![0u8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 32);
        assert_eq!(scales[0], 1.0);

        let mut output = vec![0.0f32; 32];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 32);
        for v in &output {
            assert_eq!(*v, -1.0);
        }
    }

    #[test]
    fn test_i2_roundtrip_mixed_pattern_block32() {
        // Pattern: +1, -1, 0, +1 repeated.
        let mut input = vec![0.0f32; 32];
        for i in 0..32 {
            input[i] = match i % 4 {
                0 => 1.0,
                1 => -1.0,
                2 => 0.0,
                3 => 0.8, // maps to +1
                _ => unreachable!(),
            };
        }
        let mut packed = vec![0u8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 32);

        let mut output = vec![0.0f32; 32];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 32);

        // Values get quantized to ternary then scaled back.
        for i in 0..32 {
            let expected = match i % 4 {
                0 => 1.0,
                1 => -1.0,
                2 => 0.0,
                3 => 1.0, // 0.8/1.0 > 0.5 → +1 → 1.0
                _ => unreachable!(),
            };
            assert!(
                approx_eq(output[i], expected, 1e-6),
                "index {i}: got {} expected {}",
                output[i],
                expected
            );
        }
    }

    #[test]
    fn test_i2_roundtrip_block256() {
        let mut input = vec![0.0f32; 256];
        for i in 0..256 {
            input[i] = if i % 3 == 0 {
                1.0
            } else if i % 3 == 1 {
                -1.0
            } else {
                0.0
            };
        }
        let mut packed = vec![0u8; 64];
        let mut scales = vec![0.0f32; 1];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 256);

        let mut output = vec![0.0f32; 256];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 256);

        for i in 0..256 {
            let expected = if i % 3 == 0 {
                1.0
            } else if i % 3 == 1 {
                -1.0
            } else {
                0.0
            };
            assert!(approx_eq(output[i], expected, 1e-6), "idx {i}: {} != {}", output[i], expected);
        }
    }

    #[test]
    fn test_i2_multiple_blocks() {
        // 64 values = 2 blocks of 32.
        let mut input = vec![0.0f32; 64];
        for i in 0..32 {
            input[i] = 2.0; // block 0: scale=2.0
        }
        for i in 32..64 {
            input[i] = -3.0; // block 1: scale=3.0
        }
        let mut packed = vec![0u8; 16];
        let mut scales = vec![0.0f32; 2];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 32);
        assert_eq!(scales[0], 2.0);
        assert_eq!(scales[1], 3.0);

        let mut output = vec![0.0f32; 64];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 32);
        for i in 0..32 {
            assert!(approx_eq(output[i], 2.0, 1e-6), "idx {i}: {}", output[i]);
        }
        for i in 32..64 {
            assert!(approx_eq(output[i], -3.0, 1e-6), "idx {i}: {}", output[i]);
        }
    }

    #[test]
    fn test_i2_tail_handling_5_values() {
        let input = vec![1.0f32, -1.0, 0.0, 1.0, -1.0];
        let mut packed = vec![0u8; 2]; // ceil(5/4) = 2 bytes
        let mut scales = vec![0.0f32; 1];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 32);

        let mut output = vec![0.0f32; 5];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 32);
        let expected = [1.0, -1.0, 0.0, 1.0, -1.0];
        for i in 0..5 {
            assert!(approx_eq(output[i], expected[i], 1e-6), "idx {i}");
        }
    }

    #[test]
    fn test_i2_tail_handling_1_value() {
        let input = vec![-0.9f32];
        let mut packed = vec![0u8; 1];
        let mut scales = vec![0.0f32; 1];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 32);
        assert!(approx_eq(scales[0], 0.9, 1e-6));

        let mut output = vec![0.0f32; 1];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 32);
        // -0.9 / 0.9 = -1.0 → ternary -1 → -0.9
        assert!(approx_eq(output[0], -0.9, 1e-6));
    }

    #[test]
    fn test_i2_tail_handling_3_values() {
        let input = vec![0.5f32, -0.5, 0.1];
        let mut packed = vec![0u8; 1];
        let mut scales = vec![0.0f32; 1];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 32);

        let mut output = vec![0.0f32; 3];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 32);
        // scale = 0.5, 0.5/0.5=1.0 > 0.5 → +1, -0.5/0.5=-1.0 < -0.5 → -1, 0.1/0.5=0.2 → 0
        assert!(approx_eq(output[0], 0.5, 1e-6));
        assert!(approx_eq(output[1], -0.5, 1e-6));
        assert!(approx_eq(output[2], 0.0, 1e-6));
    }

    #[test]
    fn test_i2_large_values() {
        let input = vec![100.0f32; 32];
        let mut packed = vec![0u8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 32);
        assert_eq!(scales[0], 100.0);

        let mut output = vec![0.0f32; 32];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 32);
        for v in &output {
            assert!(approx_eq(*v, 100.0, 1e-4));
        }
    }

    #[test]
    fn test_i2_near_threshold_values() {
        // Values near the 0.5 threshold.
        let input = vec![0.49f32, 0.51, -0.49, -0.51];
        let mut packed = vec![0u8; 1];
        let mut scales = vec![0.0f32; 1];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 32);

        let mut output = vec![0.0f32; 4];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 32);
        let scale = scales[0]; // 0.51
        // 0.49/0.51 ≈ 0.96 > 0.5 → +1 → 0.51
        // 0.51/0.51 = 1.0 > 0.5 → +1 → 0.51
        // -0.49/0.51 ≈ -0.96 < -0.5 → -1 → -0.51
        // -0.51/0.51 = -1.0 < -0.5 → -1 → -0.51
        assert!(approx_eq(output[0], scale, 1e-6));
        assert!(approx_eq(output[1], scale, 1e-6));
        assert!(approx_eq(output[2], -scale, 1e-6));
        assert!(approx_eq(output[3], -scale, 1e-6));
    }

    #[test]
    fn test_i2_block_boundary_crossing() {
        // 48 values = 1.5 blocks of 32. The second block is partial.
        let mut input = vec![0.0f32; 48];
        for i in 0..32 {
            input[i] = 1.0;
        }
        for i in 32..48 {
            input[i] = -2.0;
        }
        let mut packed = vec![0u8; 12]; // ceil(48/4)
        let mut scales = vec![0.0f32; 2]; // ceil(48/32)
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 32);
        assert_eq!(scales[0], 1.0);
        assert_eq!(scales[1], 2.0);

        let mut output = vec![0.0f32; 48];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 32);
        for i in 0..32 {
            assert!(approx_eq(output[i], 1.0, 1e-6), "idx {i}: {}", output[i]);
        }
        for i in 32..48 {
            assert!(approx_eq(output[i], -2.0, 1e-6), "idx {i}: {}", output[i]);
        }
    }

    // ── I2_S encoding tests ────────────────────────────────────────

    #[test]
    fn test_encode_i2s_values() {
        assert_eq!(encode_i2s(0), 0b00);
        assert_eq!(encode_i2s(1), 0b01);
        assert_eq!(encode_i2s(-1), 0b11);
        assert_eq!(encode_i2s(2), 0b00); // out of range → 0
        assert_eq!(encode_i2s(-2), 0b00);
    }

    #[test]
    fn test_decode_i2s_values() {
        assert_eq!(decode_i2s(0b00), 0.0);
        assert_eq!(decode_i2s(0b01), 1.0);
        assert_eq!(decode_i2s(0b11), -1.0);
        assert_eq!(decode_i2s(0b10), 0.0); // unused code
    }

    #[test]
    fn test_pack_4_i2s() {
        let vals = [1i8, -1, 0, 1];
        let packed = pack_4_i2s(&vals);
        // 01 | 11<<2 | 00<<4 | 01<<6
        // = 0b01 | 0b1100 | 0b000000 | 0b01000000
        // = 0b01_00_11_01
        assert_eq!(packed & 0x03, 0b01); // +1
        assert_eq!((packed >> 2) & 0x03, 0b11); // -1
        assert_eq!((packed >> 4) & 0x03, 0b00); // 0
        assert_eq!((packed >> 6) & 0x03, 0b01); // +1
    }

    #[test]
    fn test_unpack_4_i2s() {
        // Pack [+1, -1, 0, +1] then unpack.
        let packed = 0b01_00_11_01u8;
        let vals = unpack_4_i2s(packed);
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], -1.0);
        assert_eq!(vals[2], 0.0);
        assert_eq!(vals[3], 1.0);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let patterns: Vec<[i8; 4]> = vec![
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [-1, -1, -1, -1],
            [1, -1, 0, 1],
            [-1, 0, 1, -1],
            [0, 1, -1, 0],
        ];
        for vals in &patterns {
            let packed = pack_4_i2s(vals);
            let unpacked = unpack_4_i2s(packed);
            for k in 0..4 {
                assert_eq!(unpacked[k], vals[k] as f32, "mismatch at index {k} for {:?}", vals);
            }
        }
    }

    #[test]
    fn test_encode_decode_roundtrip_all_valid() {
        for &val in &[-1i8, 0, 1] {
            let code = encode_i2s(val);
            let decoded = decode_i2s(code);
            assert_eq!(decoded, val as f32);
        }
    }

    // ── quantize_f32_to_i8 / dequantize_i8_to_f32 round-trip ──────

    #[test]
    fn test_i8_roundtrip_zeros() {
        let input = vec![0.0f32; 16];
        let mut output = vec![0i8; 16];
        let mut scale = 0.0f32;
        quantize_f32_to_i8(&input, &mut output, &mut scale);
        assert_eq!(scale, 0.0);
        for v in &output {
            assert_eq!(*v, 0);
        }

        let mut dequant = vec![0.0f32; 16];
        dequantize_i8_to_f32(&output, scale, &mut dequant);
        for v in &dequant {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_i8_roundtrip_ones() {
        let input = vec![1.0f32; 16];
        let mut output = vec![0i8; 16];
        let mut scale = 0.0f32;
        quantize_f32_to_i8(&input, &mut output, &mut scale);
        assert!(approx_eq(scale, 1.0 / 127.0, 1e-6));
        for v in &output {
            assert_eq!(*v, 127);
        }

        let mut dequant = vec![0.0f32; 16];
        dequantize_i8_to_f32(&output, scale, &mut dequant);
        for v in &dequant {
            assert!(approx_eq(*v, 1.0, 0.01));
        }
    }

    #[test]
    fn test_i8_roundtrip_negative() {
        let input = vec![-1.0f32; 16];
        let mut output = vec![0i8; 16];
        let mut scale = 0.0f32;
        quantize_f32_to_i8(&input, &mut output, &mut scale);
        for v in &output {
            assert_eq!(*v, -127);
        }

        let mut dequant = vec![0.0f32; 16];
        dequantize_i8_to_f32(&output, scale, &mut dequant);
        for v in &dequant {
            assert!(approx_eq(*v, -1.0, 0.01));
        }
    }

    #[test]
    fn test_i8_roundtrip_mixed() {
        let input = vec![0.5, -0.5, 0.25, -0.25, 1.0, -1.0, 0.0, 0.75];
        let mut output = vec![0i8; 8];
        let mut scale = 0.0f32;
        quantize_f32_to_i8(&input, &mut output, &mut scale);

        let mut dequant = vec![0.0f32; 8];
        dequantize_i8_to_f32(&output, scale, &mut dequant);

        for i in 0..8 {
            assert!(
                approx_eq(dequant[i], input[i], 0.02),
                "idx {}: {} != {}",
                i,
                dequant[i],
                input[i]
            );
        }
    }

    #[test]
    fn test_i8_scale_computation() {
        let input = vec![0.0, 0.5, -3.0, 1.0];
        let mut output = vec![0i8; 4];
        let mut scale = 0.0f32;
        quantize_f32_to_i8(&input, &mut output, &mut scale);
        assert!(approx_eq(scale, 3.0 / 127.0, 1e-6));
    }

    #[test]
    fn test_i8_single_value() {
        let input = vec![0.42f32];
        let mut output = vec![0i8; 1];
        let mut scale = 0.0f32;
        quantize_f32_to_i8(&input, &mut output, &mut scale);

        let mut dequant = vec![0.0f32; 1];
        dequantize_i8_to_f32(&output, scale, &mut dequant);
        assert!(approx_eq(dequant[0], 0.42, 0.01));
    }

    #[test]
    fn test_i8_large_values() {
        let input = vec![1000.0, -1000.0, 500.0, -500.0];
        let mut output = vec![0i8; 4];
        let mut scale = 0.0f32;
        quantize_f32_to_i8(&input, &mut output, &mut scale);
        assert_eq!(output[0], 127);
        assert_eq!(output[1], -127);

        let mut dequant = vec![0.0f32; 4];
        dequantize_i8_to_f32(&output, scale, &mut dequant);
        assert!(approx_eq(dequant[0], 1000.0, 10.0));
        assert!(approx_eq(dequant[1], -1000.0, 10.0));
    }

    #[test]
    fn test_i8_tail_handling() {
        // 7 values: 1 chunk of 4 + 3 tail.
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let mut output = vec![0i8; 7];
        let mut scale = 0.0f32;
        quantize_f32_to_i8(&input, &mut output, &mut scale);

        let mut dequant = vec![0.0f32; 7];
        dequantize_i8_to_f32(&output, scale, &mut dequant);
        for i in 0..7 {
            assert!(approx_eq(dequant[i], input[i], 0.02), "idx {i}");
        }
    }

    #[test]
    fn test_i8_symmetric_property() {
        // Quantizing x and -x should give negated results.
        let input = vec![0.3, 0.6, 0.9, 1.2];
        let neg_input: Vec<f32> = input.iter().map(|v| -v).collect();
        let mut out1 = vec![0i8; 4];
        let mut out2 = vec![0i8; 4];
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        quantize_f32_to_i8(&input, &mut out1, &mut s1);
        quantize_f32_to_i8(&neg_input, &mut out2, &mut s2);
        assert!(approx_eq(s1, s2, 1e-6));
        for i in 0..4 {
            assert_eq!(out1[i], -out2[i]);
        }
    }

    // ── dequantize_i8_to_f32 standalone ────────────────────────────

    #[test]
    fn test_i8_dequant_zero_scale() {
        let input = vec![100i8, -100, 50, -50];
        let mut output = vec![0.0f32; 4];
        dequantize_i8_to_f32(&input, 0.0, &mut output);
        for v in &output {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_i8_dequant_known_values() {
        let input = vec![127i8, -127, 0, 64];
        let scale = 0.5f32;
        let mut output = vec![0.0f32; 4];
        dequantize_i8_to_f32(&input, scale, &mut output);
        assert!(approx_eq(output[0], 63.5, 1e-4));
        assert!(approx_eq(output[1], -63.5, 1e-4));
        assert!(approx_eq(output[2], 0.0, 1e-4));
        assert!(approx_eq(output[3], 32.0, 1e-4));
    }

    // ── requantize_i8_to_i2 ───────────────────────────────────────

    #[test]
    fn test_requantize_zeros() {
        let input = vec![0i8; 16];
        let mut output = vec![0u8; 4];
        requantize_i8_to_i2(&input, &mut output, 0.5);
        for v in &output {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn test_requantize_all_positive() {
        let input = vec![127i8; 8];
        let mut output = vec![0u8; 2];
        requantize_i8_to_i2(&input, &mut output, 0.5);
        // All > threshold → +1 → 0b01 packed.
        // 4 values of +1: 01_01_01_01 = 0x55
        assert_eq!(output[0], 0x55);
        assert_eq!(output[1], 0x55);
    }

    #[test]
    fn test_requantize_all_negative() {
        let input = vec![-127i8; 8];
        let mut output = vec![0u8; 2];
        requantize_i8_to_i2(&input, &mut output, 0.5);
        // All < -threshold → -1 → 0b11 packed.
        // 4 values of -1: 11_11_11_11 = 0xFF
        assert_eq!(output[0], 0xFF);
        assert_eq!(output[1], 0xFF);
    }

    #[test]
    fn test_requantize_mixed() {
        let input = vec![100i8, -100, 10, -10];
        let mut output = vec![0u8; 1];
        requantize_i8_to_i2(&input, &mut output, 0.5);
        // threshold = 0.5 * 127 ≈ 64
        // 100 > 64 → +1 (0b01), -100 < -64 → -1 (0b11), 10 → 0 (0b00), -10 → 0 (0b00)
        let byte = output[0];
        assert_eq!(byte & 0x03, 0b01);
        assert_eq!((byte >> 2) & 0x03, 0b11);
        assert_eq!((byte >> 4) & 0x03, 0b00);
        assert_eq!((byte >> 6) & 0x03, 0b00);
    }

    #[test]
    fn test_requantize_threshold_boundary() {
        // threshold = 0.5 → thresh_i8 = 64.
        // Values at exactly 64 should NOT map to ±1 (strictly greater).
        let input = vec![64i8, -64, 65, -65];
        let mut output = vec![0u8; 1];
        requantize_i8_to_i2(&input, &mut output, 0.5);
        let byte = output[0];
        assert_eq!(byte & 0x03, 0b00); // 64 not > 64
        assert_eq!((byte >> 2) & 0x03, 0b00); // -64 not < -64
        assert_eq!((byte >> 4) & 0x03, 0b01); // 65 > 64 → +1
        assert_eq!((byte >> 6) & 0x03, 0b11); // -65 < -64 → -1
    }

    #[test]
    fn test_requantize_zero_threshold() {
        // threshold = 0 → thresh_i8 = 0. Any nonzero value maps to ±1.
        let input = vec![1i8, -1, 0, 50];
        let mut output = vec![0u8; 1];
        requantize_i8_to_i2(&input, &mut output, 0.0);
        let byte = output[0];
        assert_eq!(byte & 0x03, 0b01); // 1 > 0
        assert_eq!((byte >> 2) & 0x03, 0b11); // -1 < 0
        assert_eq!((byte >> 4) & 0x03, 0b00); // 0 is not > 0
        assert_eq!((byte >> 6) & 0x03, 0b01); // 50 > 0
    }

    #[test]
    fn test_requantize_tail_5_values() {
        let input = vec![100i8, -100, 0, 80, -80];
        let mut output = vec![0u8; 2]; // ceil(5/4)
        requantize_i8_to_i2(&input, &mut output, 0.5);
        // First byte: [100→+1, -100→-1, 0→0, 80→+1]
        let byte0 = output[0];
        assert_eq!(byte0 & 0x03, 0b01);
        assert_eq!((byte0 >> 2) & 0x03, 0b11);
        assert_eq!((byte0 >> 4) & 0x03, 0b00);
        assert_eq!((byte0 >> 6) & 0x03, 0b01);
        // Second byte: [-80→-1, 0, 0, 0]
        let byte1 = output[1];
        assert_eq!(byte1 & 0x03, 0b11);
    }

    #[test]
    fn test_requantize_full_range() {
        // Sweep all i8 values with threshold 0.5.
        let input: Vec<i8> = (-127..=127).map(|v| v as i8).collect();
        let n = input.len(); // 255
        let mut output = vec![0u8; (n + 3) / 4];
        requantize_i8_to_i2(&input, &mut output, 0.5);
        // Verify each value.
        let thresh = 64i8; // 0.5 * 127 rounded
        for i in 0..n {
            let byte_idx = i / 4;
            let bit_pos = (i % 4) * 2;
            let code = (output[byte_idx] >> bit_pos) & 0x03;
            let expected = if input[i] > thresh {
                0b01
            } else if input[i] < -thresh {
                0b11
            } else {
                0b00
            };
            assert_eq!(code, expected, "mismatch at i={}, v={}", i, input[i]);
        }
    }

    // ── compute_block_scales ───────────────────────────────────────

    #[test]
    fn test_block_scales_zeros() {
        let input = vec![0.0f32; 32];
        let mut scales = vec![0.0f32; 1];
        compute_block_scales(&input, &mut scales, 32);
        assert_eq!(scales[0], 0.0);
    }

    #[test]
    fn test_block_scales_uniform() {
        let input = vec![2.5f32; 32];
        let mut scales = vec![0.0f32; 1];
        compute_block_scales(&input, &mut scales, 32);
        assert!(approx_eq(scales[0], 2.5, 1e-6));
    }

    #[test]
    fn test_block_scales_negative() {
        let input = vec![-3.0f32; 32];
        let mut scales = vec![0.0f32; 1];
        compute_block_scales(&input, &mut scales, 32);
        assert!(approx_eq(scales[0], 3.0, 1e-6));
    }

    #[test]
    fn test_block_scales_mixed() {
        let mut input = vec![0.0f32; 32];
        input[0] = 5.0;
        input[15] = -7.0;
        input[31] = 3.0;
        let mut scales = vec![0.0f32; 1];
        compute_block_scales(&input, &mut scales, 32);
        assert!(approx_eq(scales[0], 7.0, 1e-6));
    }

    #[test]
    fn test_block_scales_multiple_blocks() {
        let mut input = vec![0.0f32; 64];
        input[0] = 1.0;
        input[31] = 2.0;
        input[32] = -5.0;
        input[63] = 3.0;
        let mut scales = vec![0.0f32; 2];
        compute_block_scales(&input, &mut scales, 32);
        assert!(approx_eq(scales[0], 2.0, 1e-6));
        assert!(approx_eq(scales[1], 5.0, 1e-6));
    }

    #[test]
    fn test_block_scales_block256() {
        let mut input = vec![0.0f32; 256];
        input[0] = 10.0;
        input[255] = -20.0;
        let mut scales = vec![0.0f32; 1];
        compute_block_scales(&input, &mut scales, 256);
        assert!(approx_eq(scales[0], 20.0, 1e-6));
    }

    #[test]
    fn test_block_scales_partial_last_block() {
        // 48 values = 1 full block of 32 + 16 remainder.
        let mut input = vec![0.0f32; 48];
        input[0] = 1.0; // block 0
        input[40] = 9.0; // block 1 (partial)
        let mut scales = vec![0.0f32; 2];
        compute_block_scales(&input, &mut scales, 32);
        assert!(approx_eq(scales[0], 1.0, 1e-6));
        assert!(approx_eq(scales[1], 9.0, 1e-6));
    }

    #[test]
    fn test_block_scales_matches_quantize() {
        // Verify compute_block_scales gives same scales as quantize_f32_to_i2.
        let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.2).collect();
        let mut scales_standalone = vec![0.0f32; 2];
        compute_block_scales(&input, &mut scales_standalone, 32);

        let mut packed = vec![0u8; 16];
        let mut scales_quant = vec![0.0f32; 2];
        quantize_f32_to_i2(&input, &mut packed, &mut scales_quant, 32);

        for i in 0..2 {
            assert!(
                approx_eq(scales_standalone[i], scales_quant[i], 1e-6),
                "block {i}: {} != {}",
                scales_standalone[i],
                scales_quant[i]
            );
        }
    }

    // ── Cross-function integration tests ───────────────────────────

    #[test]
    fn test_i8_to_i2_pipeline() {
        // f32 → i8 → i2 pipeline.
        let input = vec![1.0f32, -1.0, 0.1, 0.0, -0.9, 0.9, -0.1, 0.5];
        let mut i8_out = vec![0i8; 8];
        let mut scale = 0.0f32;
        quantize_f32_to_i8(&input, &mut i8_out, &mut scale);

        let mut i2_out = vec![0u8; 2];
        requantize_i8_to_i2(&i8_out, &mut i2_out, 0.3);

        // Verify ternary values are reasonable.
        for i in 0..8 {
            let byte_idx = i / 4;
            let bit_pos = (i % 4) * 2;
            let code = (i2_out[byte_idx] >> bit_pos) & 0x03;
            let ternary = decode_i2s(code);
            // Large magnitude inputs should be ±1, near-zero should be 0.
            if input[i].abs() > 0.5 {
                assert!(ternary.abs() == 1.0, "idx {i}: expected ±1 for input {}", input[i]);
            }
        }
    }

    #[test]
    fn test_full_i2_pipeline_32_block() {
        // Full roundtrip: f32 → i2 → f32 for a 32-element block.
        let input: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let mut packed = vec![0u8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 32);

        let mut output = vec![0.0f32; 32];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 32);

        for i in 0..32 {
            assert!(approx_eq(output[i], input[i], 1e-6), "idx {i}");
        }
    }

    #[test]
    fn test_full_i2_pipeline_256_block() {
        // Full roundtrip with QK256 block size.
        let input: Vec<f32> = (0..512)
            .map(|i| match i % 3 {
                0 => 2.0,
                1 => -2.0,
                _ => 0.0,
            })
            .collect();
        let mut packed = vec![0u8; 128];
        let mut scales = vec![0.0f32; 2];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 256);

        let mut output = vec![0.0f32; 512];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 256);

        for i in 0..512 {
            let expected = match i % 3 {
                0 => 2.0,
                1 => -2.0,
                _ => 0.0,
            };
            assert!(approx_eq(output[i], expected, 1e-4), "idx {i}: {} != {}", output[i], expected);
        }
    }

    #[test]
    fn test_dequant_i2_all_codes() {
        // Manually set all 4 I2_S codes and verify dequantization.
        let packed = vec![0b10_11_01_00u8]; // codes: 00, 01, 11, 10
        let scales = vec![2.0f32];
        let mut output = vec![0.0f32; 4];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 32);
        assert_eq!(output[0], 0.0); // 00 → 0
        assert_eq!(output[1], 2.0); // 01 → +1 * 2.0
        assert_eq!(output[2], -2.0); // 11 → -1 * 2.0
        assert_eq!(output[3], 0.0); // 10 → 0 (unused)
    }

    #[test]
    fn test_dequant_i8_tail_1() {
        let input = vec![50i8];
        let mut output = vec![0.0f32; 1];
        dequantize_i8_to_f32(&input, 0.1, &mut output);
        assert!(approx_eq(output[0], 5.0, 1e-4));
    }

    #[test]
    fn test_dequant_i8_tail_3() {
        let input = vec![10i8, -20, 30];
        let mut output = vec![0.0f32; 3];
        dequantize_i8_to_f32(&input, 2.0, &mut output);
        assert!(approx_eq(output[0], 20.0, 1e-4));
        assert!(approx_eq(output[1], -40.0, 1e-4));
        assert!(approx_eq(output[2], 60.0, 1e-4));
    }

    #[test]
    fn test_i8_quantize_clamp() {
        // Verify clamping to [-127, 127].
        let input = vec![f32::MAX, f32::MIN];
        let mut output = vec![0i8; 2];
        let mut scale = 0.0f32;
        quantize_f32_to_i8(&input, &mut output, &mut scale);
        assert_eq!(output[0], 127);
        assert_eq!(output[1], -127);
    }

    #[test]
    fn test_i2_alternating_block32() {
        // +1, -1, +1, -1, ... for a full block.
        let input: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let mut packed = vec![0u8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 32);

        let mut output = vec![0.0f32; 32];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 32);
        for i in 0..32 {
            let expected = if i % 2 == 0 { 1.0 } else { -1.0 };
            assert_eq!(output[i], expected, "idx {i}");
        }
    }

    #[test]
    fn test_i2_small_magnitudes_become_zero() {
        // Values close to zero should quantize to 0.
        let input = vec![0.1f32, -0.1, 0.2, -0.2];
        // scale will be 0.2. 0.1/0.2 = 0.5, not > 0.5, so → 0.
        // 0.2/0.2 = 1.0 > 0.5 → +1, -0.2/0.2 = -1.0 < -0.5 → -1.
        let mut packed = vec![0u8; 1];
        let mut scales = vec![0.0f32; 1];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 32);

        let mut output = vec![0.0f32; 4];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 32);
        assert_eq!(output[0], 0.0);
        assert_eq!(output[1], 0.0);
        assert!(approx_eq(output[2], 0.2, 1e-6));
        assert!(approx_eq(output[3], -0.2, 1e-6));
    }

    #[test]
    fn test_block_scales_single_element() {
        let input = vec![42.0f32];
        let mut scales = vec![0.0f32; 1];
        compute_block_scales(&input, &mut scales, 32);
        assert!(approx_eq(scales[0], 42.0, 1e-6));
    }

    #[test]
    fn test_i2_roundtrip_with_varied_scales() {
        // Each block has a different scale.
        let mut input = vec![0.0f32; 96]; // 3 blocks of 32
        for i in 0..32 {
            input[i] = 1.0;
        }
        for i in 32..64 {
            input[i] = -5.0;
        }
        for i in 64..96 {
            input[i] = 0.3; // 0.3/0.3 = 1.0 > 0.5 → +1
        }
        let mut packed = vec![0u8; 24];
        let mut scales = vec![0.0f32; 3];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 32);
        assert!(approx_eq(scales[0], 1.0, 1e-6));
        assert!(approx_eq(scales[1], 5.0, 1e-6));
        assert!(approx_eq(scales[2], 0.3, 1e-6));

        let mut output = vec![0.0f32; 96];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 32);
        for i in 0..32 {
            assert!(approx_eq(output[i], 1.0, 1e-6), "idx {i}");
        }
        for i in 32..64 {
            assert!(approx_eq(output[i], -5.0, 1e-4), "idx {i}");
        }
        for i in 64..96 {
            assert!(approx_eq(output[i], 0.3, 1e-4), "idx {i}");
        }
    }

    // ── Edge cases / panics ────────────────────────────────────────

    #[test]
    #[should_panic(expected = "block_size must be 32 or 256")]
    fn test_quantize_i2_invalid_block_size() {
        let input = vec![0.0f32; 8];
        let mut packed = vec![0u8; 2];
        let mut scales = vec![0.0f32; 1];
        quantize_f32_to_i2(&input, &mut packed, &mut scales, 64);
    }

    #[test]
    #[should_panic(expected = "block_size must be 32 or 256")]
    fn test_dequantize_i2_invalid_block_size() {
        let packed = vec![0u8; 2];
        let scales = vec![0.0f32; 1];
        let mut output = vec![0.0f32; 8];
        dequantize_i2_to_f32(&packed, &scales, &mut output, 64);
    }

    #[test]
    #[should_panic(expected = "block_size must be 32 or 256")]
    fn test_block_scales_invalid_block_size() {
        let input = vec![0.0f32; 8];
        let mut scales = vec![0.0f32; 1];
        compute_block_scales(&input, &mut scales, 128);
    }
}
