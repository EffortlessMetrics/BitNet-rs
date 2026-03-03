//! NEON-optimized dynamic quantization v2 for Apple Silicon.
//! Per-channel and per-token quantization with calibration
//! and symmetric/asymmetric modes.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Configuration types ───────────────────────────────────────────────

/// Quantization mode for dynamic quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    /// Symmetric: zero-point is always 0, range = [-max_abs, +max_abs].
    Symmetric,
    /// Asymmetric: zero-point computed from min/max.
    Asymmetric,
}

/// Precision level for mixed-precision quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// 2-bit signed (I2_S ternary: -1, 0, +1).
    I2,
    /// 4-bit signed.
    I4,
    /// 8-bit signed.
    I8,
}

/// Pre-computed calibration statistics for a tensor.
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    /// Per-channel (or per-token) scale factors.
    pub scales: Vec<f32>,
    /// Per-channel (or per-token) zero-points.
    pub zero_points: Vec<f32>,
    /// Number of calibration samples observed.
    pub num_samples: usize,
}

/// Result of a quantization operation.
#[derive(Debug, Clone)]
pub struct QuantizedOutput {
    /// Quantized values (INT8).
    pub data: Vec<i8>,
    /// Scale factor per row/channel.
    pub scales: Vec<f32>,
    /// Zero-point per row/channel (0.0 for symmetric).
    pub zero_points: Vec<f32>,
}

/// Result of 2-bit symmetric quantization.
#[derive(Debug, Clone)]
pub struct QuantizedI2Output {
    /// Packed 2-bit values (4 values per byte, LSB-first).
    pub packed: Vec<u8>,
    /// Scale factor per row.
    pub scales: Vec<f32>,
}

/// Result of mixed-precision quantization.
#[derive(Debug, Clone)]
pub struct MixedPrecisionOutput {
    /// Per-block precision selected.
    pub precisions: Vec<Precision>,
    /// INT8 quantized data (all blocks stored as i8, lower bits used for I2/I4).
    pub data: Vec<i8>,
    /// Per-block scales.
    pub scales: Vec<f32>,
    /// Per-block zero-points.
    pub zero_points: Vec<f32>,
    /// Block size used.
    pub block_size: usize,
}

// ── Scalar helpers ────────────────────────────────────────────────────

/// Find min and max of a slice (scalar).
#[inline]
fn scalar_min_max(data: &[f32]) -> (f32, f32) {
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &v in data {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }
    (min_val, max_val)
}

/// Compute scale and zero-point for INT8 symmetric quantization.
#[inline]
fn compute_symmetric_scale(min_val: f32, max_val: f32) -> (f32, f32) {
    let abs_max = min_val.abs().max(max_val.abs());
    let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
    (scale, 0.0)
}

/// Compute scale and zero-point for INT8 asymmetric quantization.
#[inline]
fn compute_asymmetric_scale(min_val: f32, max_val: f32) -> (f32, f32) {
    let range = max_val - min_val;
    let scale = if range == 0.0 { 1.0 } else { range / 255.0 };
    let zero_point = -128.0 - min_val / scale;
    (scale, zero_point.round().clamp(-128.0, 127.0))
}

/// Clamp and round to INT8.
#[inline]
fn quantize_val_i8(val: f32, scale: f32, zero_point: f32) -> i8 {
    let q = (val / scale + zero_point).round();
    q.clamp(-128.0, 127.0) as i8
}

/// Dequantize a single INT8 value.
#[inline]
fn dequantize_val_i8(val: i8, scale: f32, zero_point: f32) -> f32 {
    (val as f32 - zero_point) * scale
}

// ── NEON min/max reduction ────────────────────────────────────────────

/// Find min and max of a slice using NEON.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_min_max(data: &[f32]) -> (f32, f32) {
    let n = data.len();
    let ptr = data.as_ptr();
    let chunks = n / 4;

    let mut vmin = vdupq_n_f32(f32::INFINITY);
    let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);

    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        vmin = vminq_f32(vmin, v);
        vmax = vmaxq_f32(vmax, v);
    }

    let mut min_val = vminvq_f32(vmin);
    let mut max_val = vmaxvq_f32(vmax);

    for i in (chunks * 4)..n {
        let v = *ptr.add(i);
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    (min_val, max_val)
}

// ── NEON quantize core (shared by per-channel and calibrated) ─────────

/// Quantize a row of f32 to i8 using NEON.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_quantize_row(
    src: *const f32,
    dst: *mut i8,
    len: usize,
    inv_scale: f32,
    zp: f32,
) {
    let chunks = len / 4;

    let vzp = vdupq_n_f32(zp);
    let vinv = vdupq_n_f32(inv_scale);
    let vmin_clamp = vdupq_n_f32(-128.0);
    let vmax_clamp = vdupq_n_f32(127.0);

    for i in 0..chunks {
        let v = vld1q_f32(src.add(i * 4));
        let q = vaddq_f32(vmulq_f32(v, vinv), vzp);
        let q = vrndnq_f32(q);
        let q = vmaxq_f32(vminq_f32(q, vmax_clamp), vmin_clamp);
        *dst.add(i * 4) = vgetq_lane_f32(q, 0) as i8;
        *dst.add(i * 4 + 1) = vgetq_lane_f32(q, 1) as i8;
        *dst.add(i * 4 + 2) = vgetq_lane_f32(q, 2) as i8;
        *dst.add(i * 4 + 3) = vgetq_lane_f32(q, 3) as i8;
    }

    // Scalar tail
    let scale = 1.0 / inv_scale;
    for i in (chunks * 4)..len {
        *dst.add(i) = quantize_val_i8(*src.add(i), scale, zp);
    }
}

// ── Per-channel quantize INT8 ─────────────────────────────────────────

/// Per-channel INT8 quantization with NEON acceleration.
///
/// `data` is row-major `[num_channels, channel_size]`.
/// Returns quantized values plus per-channel scale and zero-point.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn per_channel_quantize_i8(
    data: &[f32],
    num_channels: usize,
    channel_size: usize,
    mode: QuantMode,
) -> QuantizedOutput {
    assert_eq!(data.len(), num_channels * channel_size, "data length mismatch");

    let mut out_data = vec![0i8; data.len()];
    let mut scales = Vec::with_capacity(num_channels);
    let mut zero_points = Vec::with_capacity(num_channels);

    for ch in 0..num_channels {
        let start = ch * channel_size;
        let end = start + channel_size;
        let channel = &data[start..end];

        let (min_val, max_val) = neon_min_max(channel);
        let (scale, zp) = match mode {
            QuantMode::Symmetric => compute_symmetric_scale(min_val, max_val),
            QuantMode::Asymmetric => compute_asymmetric_scale(min_val, max_val),
        };

        scales.push(scale);
        zero_points.push(zp);

        let inv_scale = 1.0 / scale;
        neon_quantize_row(
            channel.as_ptr(),
            out_data[start..end].as_mut_ptr(),
            channel_size,
            inv_scale,
            zp,
        );
    }

    QuantizedOutput { data: out_data, scales, zero_points }
}

/// Per-channel INT8 quantization — scalar fallback.
pub fn per_channel_quantize_i8_scalar(
    data: &[f32],
    num_channels: usize,
    channel_size: usize,
    mode: QuantMode,
) -> QuantizedOutput {
    assert_eq!(data.len(), num_channels * channel_size, "data length mismatch");

    let mut out_data = vec![0i8; data.len()];
    let mut scales = Vec::with_capacity(num_channels);
    let mut zero_points = Vec::with_capacity(num_channels);

    for ch in 0..num_channels {
        let start = ch * channel_size;
        let end = start + channel_size;
        let channel = &data[start..end];

        let (min_val, max_val) = scalar_min_max(channel);
        let (scale, zp) = match mode {
            QuantMode::Symmetric => compute_symmetric_scale(min_val, max_val),
            QuantMode::Asymmetric => compute_asymmetric_scale(min_val, max_val),
        };

        scales.push(scale);
        zero_points.push(zp);

        for i in 0..channel_size {
            out_data[start + i] = quantize_val_i8(channel[i], scale, zp);
        }
    }

    QuantizedOutput { data: out_data, scales, zero_points }
}

// ── Per-token quantize INT8 ───────────────────────────────────────────

/// Per-token INT8 quantization with NEON acceleration.
///
/// `data` is row-major `[num_tokens, hidden_dim]`.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn per_token_quantize_i8(
    data: &[f32],
    num_tokens: usize,
    hidden_dim: usize,
    mode: QuantMode,
) -> QuantizedOutput {
    per_channel_quantize_i8(data, num_tokens, hidden_dim, mode)
}

/// Per-token INT8 quantization — scalar fallback.
pub fn per_token_quantize_i8_scalar(
    data: &[f32],
    num_tokens: usize,
    hidden_dim: usize,
    mode: QuantMode,
) -> QuantizedOutput {
    per_channel_quantize_i8_scalar(data, num_tokens, hidden_dim, mode)
}

// ── Symmetric 2-bit quantization (I2_S) ──────────────────────────────

/// Pack ternary codes into bytes. Shared by NEON and scalar paths.
fn pack_ternary(row_data: &[f32], thresh: f32, out: &mut [u8]) {
    let row_size = row_data.len();
    let packed_row_bytes = (row_size + 3) / 4;
    for byte_idx in 0..packed_row_bytes {
        let mut byte_val: u8 = 0;
        for bit_pos in 0..4 {
            let elem_idx = byte_idx * 4 + bit_pos;
            if elem_idx < row_size {
                let v = row_data[elem_idx];
                let code: u8 = if v > thresh {
                    0b01 // +1
                } else if v < -thresh {
                    0b11 // -1
                } else {
                    0b00 // 0
                };
                byte_val |= code << (bit_pos * 2);
            }
        }
        out[byte_idx] = byte_val;
    }
}

/// Symmetric 2-bit quantization with NEON.
///
/// Quantizes to ternary {-1, 0, +1}. Values within `threshold * scale`
/// of zero are mapped to 0. Packs 4 values per byte (LSB-first).
///
/// Encoding: 0b00 = 0, 0b01 = +1, 0b11 = -1.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn symmetric_quantize_i2(
    data: &[f32],
    num_rows: usize,
    row_size: usize,
    threshold: f32,
) -> QuantizedI2Output {
    assert_eq!(data.len(), num_rows * row_size, "data length mismatch");

    let packed_row_bytes = (row_size + 3) / 4;
    let mut packed = vec![0u8; num_rows * packed_row_bytes];
    let mut scales = Vec::with_capacity(num_rows);

    for row in 0..num_rows {
        let start = row * row_size;
        let end = start + row_size;
        let row_data = &data[start..end];

        // Find absolute max using NEON
        let ptr = row_data.as_ptr();
        let chunks = row_size / 4;
        let mut vabsmax = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            let va = vabsq_f32(v);
            vabsmax = vmaxq_f32(vabsmax, va);
        }

        let mut abs_max = vmaxvq_f32(vabsmax);
        for i in (chunks * 4)..row_size {
            abs_max = abs_max.max((*ptr.add(i)).abs());
        }

        let scale = if abs_max == 0.0 { 1.0 } else { abs_max };
        scales.push(scale);

        let thresh = threshold * scale;
        let pack_start = row * packed_row_bytes;
        pack_ternary(row_data, thresh, &mut packed[pack_start..]);
    }

    QuantizedI2Output { packed, scales }
}

/// Symmetric 2-bit quantization — scalar fallback.
pub fn symmetric_quantize_i2_scalar(
    data: &[f32],
    num_rows: usize,
    row_size: usize,
    threshold: f32,
) -> QuantizedI2Output {
    assert_eq!(data.len(), num_rows * row_size, "data length mismatch");

    let packed_row_bytes = (row_size + 3) / 4;
    let mut packed = vec![0u8; num_rows * packed_row_bytes];
    let mut scales = Vec::with_capacity(num_rows);

    for row in 0..num_rows {
        let start = row * row_size;
        let end = start + row_size;
        let row_data = &data[start..end];

        let abs_max = row_data.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max };
        scales.push(scale);

        let thresh = threshold * scale;
        let pack_start = row * packed_row_bytes;
        pack_ternary(row_data, thresh, &mut packed[pack_start..]);
    }

    QuantizedI2Output { packed, scales }
}

// ── Calibrated quantize ───────────────────────────────────────────────

/// Quantize using pre-computed calibration statistics (NEON).
///
/// Uses the provided `CalibrationStats` (scales/zero-points) instead
/// of computing them from the data, which is useful for post-training
/// quantization with a representative calibration dataset.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn calibrated_quantize(
    data: &[f32],
    num_rows: usize,
    row_size: usize,
    stats: &CalibrationStats,
) -> QuantizedOutput {
    assert_eq!(data.len(), num_rows * row_size, "data length mismatch");
    assert!(stats.scales.len() >= num_rows, "not enough calibration scales");
    assert!(stats.zero_points.len() >= num_rows, "not enough calibration zero_points");

    let mut out_data = vec![0i8; data.len()];

    for row in 0..num_rows {
        let start = row * row_size;
        let scale = stats.scales[row];
        let zp = stats.zero_points[row];
        let inv_scale = 1.0 / scale;

        neon_quantize_row(
            data[start..].as_ptr(),
            out_data[start..].as_mut_ptr(),
            row_size,
            inv_scale,
            zp,
        );
    }

    QuantizedOutput {
        data: out_data,
        scales: stats.scales[..num_rows].to_vec(),
        zero_points: stats.zero_points[..num_rows].to_vec(),
    }
}

/// Calibrated quantize — scalar fallback.
pub fn calibrated_quantize_scalar(
    data: &[f32],
    num_rows: usize,
    row_size: usize,
    stats: &CalibrationStats,
) -> QuantizedOutput {
    assert_eq!(data.len(), num_rows * row_size, "data length mismatch");
    assert!(stats.scales.len() >= num_rows, "not enough calibration scales");
    assert!(stats.zero_points.len() >= num_rows, "not enough calibration zero_points");

    let mut out_data = vec![0i8; data.len()];

    for row in 0..num_rows {
        let start = row * row_size;
        let scale = stats.scales[row];
        let zp = stats.zero_points[row];

        for i in 0..row_size {
            out_data[start + i] = quantize_val_i8(data[start + i], scale, zp);
        }
    }

    QuantizedOutput {
        data: out_data,
        scales: stats.scales[..num_rows].to_vec(),
        zero_points: stats.zero_points[..num_rows].to_vec(),
    }
}

// ── Dequantize INT8 → F32 ────────────────────────────────────────────

/// Dequantize INT8 data back to F32 using NEON.
///
/// `data` is row-major `[num_rows, row_size]` with per-row scale/zero-point.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dequantize_i8_f32(
    data: &[i8],
    num_rows: usize,
    row_size: usize,
    scales: &[f32],
    zero_points: &[f32],
) -> Vec<f32> {
    assert_eq!(data.len(), num_rows * row_size, "data length mismatch");
    assert!(scales.len() >= num_rows, "not enough scales");
    assert!(zero_points.len() >= num_rows, "not enough zero_points");

    let mut output = vec![0.0f32; data.len()];

    for row in 0..num_rows {
        let start = row * row_size;
        let scale = scales[row];
        let zp = zero_points[row];

        let chunks = row_size / 4;

        let in_ptr = data[start..].as_ptr();
        let out_ptr = output[start..].as_mut_ptr();

        let vscale = vdupq_n_f32(scale);
        let vzp = vdupq_n_f32(zp);

        for i in 0..chunks {
            let offset = i * 4;
            let v0 = *in_ptr.add(offset) as f32;
            let v1 = *in_ptr.add(offset + 1) as f32;
            let v2 = *in_ptr.add(offset + 2) as f32;
            let v3 = *in_ptr.add(offset + 3) as f32;

            let arr = [v0, v1, v2, v3];
            let vi = vld1q_f32(arr.as_ptr());
            let dq = vmulq_f32(vsubq_f32(vi, vzp), vscale);
            vst1q_f32(out_ptr.add(offset), dq);
        }

        // Scalar tail
        for i in (chunks * 4)..row_size {
            output[start + i] = dequantize_val_i8(data[start + i], scale, zp);
        }
    }

    output
}

/// Dequantize INT8 → F32 — scalar fallback.
pub fn dequantize_i8_f32_scalar(
    data: &[i8],
    num_rows: usize,
    row_size: usize,
    scales: &[f32],
    zero_points: &[f32],
) -> Vec<f32> {
    assert_eq!(data.len(), num_rows * row_size, "data length mismatch");
    assert!(scales.len() >= num_rows, "not enough scales");
    assert!(zero_points.len() >= num_rows, "not enough zero_points");

    let mut output = vec![0.0f32; data.len()];

    for row in 0..num_rows {
        let start = row * row_size;
        let scale = scales[row];
        let zp = zero_points[row];

        for i in 0..row_size {
            output[start + i] = dequantize_val_i8(data[start + i], scale, zp);
        }
    }

    output
}

// ── Mixed-precision quantize ──────────────────────────────────────────

/// Select precision based on dynamic range.
///
/// - I2 for small range (mostly near zero — ternary is sufficient)
/// - I4 for medium range
/// - I8 for large range
#[inline]
fn select_precision(dynamic_range: f32, i4_threshold: f32, i8_threshold: f32) -> Precision {
    if dynamic_range <= i4_threshold {
        Precision::I2
    } else if dynamic_range <= i8_threshold {
        Precision::I4
    } else {
        Precision::I8
    }
}

/// Quantize a single block at the selected precision. Returns (scale, zp).
fn quantize_block(
    block: &[f32],
    out: &mut [i8],
    precision: Precision,
    min_val: f32,
    max_val: f32,
) -> (f32, f32) {
    match precision {
        Precision::I2 => {
            let abs_max = min_val.abs().max(max_val.abs());
            let s = if abs_max == 0.0 { 1.0 } else { abs_max };
            for i in 0..block.len() {
                let normalized = block[i] / s;
                out[i] = if normalized > 0.5 {
                    1
                } else if normalized < -0.5 {
                    -1
                } else {
                    0
                };
            }
            (s, 0.0)
        }
        Precision::I4 => {
            let abs_max = min_val.abs().max(max_val.abs());
            let s = if abs_max == 0.0 { 1.0 } else { abs_max / 7.0 };
            for i in 0..block.len() {
                let q = (block[i] / s).round().clamp(-8.0, 7.0) as i8;
                out[i] = q;
            }
            (s, 0.0)
        }
        Precision::I8 => {
            let (s, z) = compute_symmetric_scale(min_val, max_val);
            for i in 0..block.len() {
                out[i] = quantize_val_i8(block[i], s, z);
            }
            (s, z)
        }
    }
}

/// Mixed-precision quantization with NEON (dynamic per-block precision).
///
/// Divides `data` into blocks of `block_size`. Each block independently
/// selects I2/I4/I8 precision based on its dynamic range.
///
/// Thresholds:
/// - `i4_threshold`: blocks with range ≤ this use I2
/// - `i8_threshold`: blocks with range ≤ this use I4; above uses I8
///
/// # Safety
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn mixed_precision_quantize(
    data: &[f32],
    block_size: usize,
    i4_threshold: f32,
    i8_threshold: f32,
) -> MixedPrecisionOutput {
    assert!(block_size > 0, "block_size must be > 0");

    let num_blocks = (data.len() + block_size - 1) / block_size;
    let mut precisions = Vec::with_capacity(num_blocks);
    let mut out_data = vec![0i8; data.len()];
    let mut scales = Vec::with_capacity(num_blocks);
    let mut zero_points = Vec::with_capacity(num_blocks);

    for blk in 0..num_blocks {
        let start = blk * block_size;
        let end = (start + block_size).min(data.len());
        let block = &data[start..end];

        let (min_val, max_val) = neon_min_max(block);
        let range = max_val - min_val;
        let precision = select_precision(range, i4_threshold, i8_threshold);
        precisions.push(precision);

        let (scale, zp) =
            quantize_block(block, &mut out_data[start..end], precision, min_val, max_val);

        scales.push(scale);
        zero_points.push(zp);
    }

    MixedPrecisionOutput { precisions, data: out_data, scales, zero_points, block_size }
}

/// Mixed-precision quantization — scalar fallback.
pub fn mixed_precision_quantize_scalar(
    data: &[f32],
    block_size: usize,
    i4_threshold: f32,
    i8_threshold: f32,
) -> MixedPrecisionOutput {
    assert!(block_size > 0, "block_size must be > 0");

    let num_blocks = (data.len() + block_size - 1) / block_size;
    let mut precisions = Vec::with_capacity(num_blocks);
    let mut out_data = vec![0i8; data.len()];
    let mut scales = Vec::with_capacity(num_blocks);
    let mut zero_points = Vec::with_capacity(num_blocks);

    for blk in 0..num_blocks {
        let start = blk * block_size;
        let end = (start + block_size).min(data.len());
        let block = &data[start..end];

        let (min_val, max_val) = scalar_min_max(block);
        let range = max_val - min_val;
        let precision = select_precision(range, i4_threshold, i8_threshold);
        precisions.push(precision);

        let (scale, zp) =
            quantize_block(block, &mut out_data[start..end], precision, min_val, max_val);

        scales.push(scale);
        zero_points.push(zp);
    }

    MixedPrecisionOutput { precisions, data: out_data, scales, zero_points, block_size }
}

// ── Runtime dispatch ──────────────────────────────────────────────────

/// Per-channel INT8 quantization with runtime NEON dispatch.
pub fn per_channel_quantize_i8_dispatch(
    data: &[f32],
    num_channels: usize,
    channel_size: usize,
    mode: QuantMode,
) -> QuantizedOutput {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe {
                per_channel_quantize_i8(data, num_channels, channel_size, mode)
            };
        }
    }
    per_channel_quantize_i8_scalar(data, num_channels, channel_size, mode)
}

/// Per-token INT8 quantization with runtime NEON dispatch.
pub fn per_token_quantize_i8_dispatch(
    data: &[f32],
    num_tokens: usize,
    hidden_dim: usize,
    mode: QuantMode,
) -> QuantizedOutput {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe {
                per_token_quantize_i8(data, num_tokens, hidden_dim, mode)
            };
        }
    }
    per_token_quantize_i8_scalar(data, num_tokens, hidden_dim, mode)
}

/// Symmetric I2 quantization with runtime NEON dispatch.
pub fn symmetric_quantize_i2_dispatch(
    data: &[f32],
    num_rows: usize,
    row_size: usize,
    threshold: f32,
) -> QuantizedI2Output {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe {
                symmetric_quantize_i2(data, num_rows, row_size, threshold)
            };
        }
    }
    symmetric_quantize_i2_scalar(data, num_rows, row_size, threshold)
}

/// Calibrated quantization with runtime NEON dispatch.
pub fn calibrated_quantize_dispatch(
    data: &[f32],
    num_rows: usize,
    row_size: usize,
    stats: &CalibrationStats,
) -> QuantizedOutput {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { calibrated_quantize(data, num_rows, row_size, stats) };
        }
    }
    calibrated_quantize_scalar(data, num_rows, row_size, stats)
}

/// INT8 → F32 dequantization with runtime NEON dispatch.
pub fn dequantize_i8_f32_dispatch(
    data: &[i8],
    num_rows: usize,
    row_size: usize,
    scales: &[f32],
    zero_points: &[f32],
) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe {
                dequantize_i8_f32(data, num_rows, row_size, scales, zero_points)
            };
        }
    }
    dequantize_i8_f32_scalar(data, num_rows, row_size, scales, zero_points)
}

/// Mixed-precision quantization with runtime NEON dispatch.
pub fn mixed_precision_quantize_dispatch(
    data: &[f32],
    block_size: usize,
    i4_threshold: f32,
    i8_threshold: f32,
) -> MixedPrecisionOutput {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe {
                mixed_precision_quantize(data, block_size, i4_threshold, i8_threshold)
            };
        }
    }
    mixed_precision_quantize_scalar(data, block_size, i4_threshold, i8_threshold)
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert two f32 slices are close within `eps`.
    fn assert_close(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= eps,
                "index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    fn assert_i8_close(a: &[i8], b: &[i8], max_diff: i8) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x as i16 - y as i16).unsigned_abs() <= max_diff as u16,
                "index {i}: {x} vs {y}"
            );
        }
    }

    // ── per_channel_quantize_i8 tests ─────────────────────────────

    #[test]
    fn test_per_channel_symmetric_basic() {
        let data = vec![1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.0, 0.0];
        let result = per_channel_quantize_i8_scalar(&data, 2, 4, QuantMode::Symmetric);
        assert_eq!(result.scales.len(), 2);
        assert_eq!(result.zero_points.len(), 2);
        assert_eq!(result.data.len(), 8);
        for &zp in &result.zero_points {
            assert_eq!(zp, 0.0);
        }
    }

    #[test]
    fn test_per_channel_asymmetric_basic() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let result = per_channel_quantize_i8_scalar(&data, 1, 4, QuantMode::Asymmetric);
        assert_eq!(result.scales.len(), 1);
        assert!(result.scales[0] > 0.0);
    }

    #[test]
    fn test_per_channel_all_zeros() {
        let data = vec![0.0; 8];
        let result = per_channel_quantize_i8_scalar(&data, 2, 4, QuantMode::Symmetric);
        for &v in &result.data {
            assert_eq!(v, 0);
        }
        for &s in &result.scales {
            assert_eq!(s, 1.0);
        }
    }

    #[test]
    fn test_per_channel_single_element() {
        let data = vec![42.0];
        let result = per_channel_quantize_i8_scalar(&data, 1, 1, QuantMode::Symmetric);
        assert_eq!(result.data.len(), 1);
        assert_eq!(result.data[0], 127);
    }

    #[test]
    fn test_per_channel_negative_only() {
        let data = vec![-1.0, -2.0, -3.0, -4.0];
        let result = per_channel_quantize_i8_scalar(&data, 1, 4, QuantMode::Symmetric);
        for &v in &result.data {
            assert!(v <= 0);
        }
    }

    #[test]
    fn test_per_channel_large_values() {
        let data = vec![1e6, -1e6, 5e5, -5e5];
        let result = per_channel_quantize_i8_scalar(&data, 1, 4, QuantMode::Symmetric);
        assert_eq!(result.data[0], 127);
        assert_eq!(result.data[1], -127);
    }

    #[test]
    fn test_per_channel_many_channels() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.2).collect();
        let result = per_channel_quantize_i8_scalar(&data, 16, 4, QuantMode::Symmetric);
        assert_eq!(result.scales.len(), 16);
        assert_eq!(result.data.len(), 64);
    }

    #[test]
    #[should_panic(expected = "data length mismatch")]
    fn test_per_channel_size_mismatch() {
        per_channel_quantize_i8_scalar(&[1.0, 2.0], 1, 4, QuantMode::Symmetric);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_per_channel_neon_vs_scalar_symmetric() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.05 - 3.0).collect();
        let scalar = per_channel_quantize_i8_scalar(&data, 4, 32, QuantMode::Symmetric);
        let neon = unsafe { per_channel_quantize_i8(&data, 4, 32, QuantMode::Symmetric) };
        assert_i8_close(&neon.data, &scalar.data, 1);
        assert_close(&neon.scales, &scalar.scales, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_per_channel_neon_vs_scalar_asymmetric() {
        let data: Vec<f32> = (0..48).map(|i| (i as f32) * 0.1).collect();
        let scalar = per_channel_quantize_i8_scalar(&data, 3, 16, QuantMode::Asymmetric);
        let neon = unsafe { per_channel_quantize_i8(&data, 3, 16, QuantMode::Asymmetric) };
        assert_i8_close(&neon.data, &scalar.data, 1);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_per_channel_neon_odd_channel_size() {
        let data: Vec<f32> = (0..15).map(|i| i as f32 - 7.0).collect();
        let scalar = per_channel_quantize_i8_scalar(&data, 3, 5, QuantMode::Symmetric);
        let neon = unsafe { per_channel_quantize_i8(&data, 3, 5, QuantMode::Symmetric) };
        assert_i8_close(&neon.data, &scalar.data, 1);
    }

    // ── per_token_quantize_i8 tests ───────────────────────────────

    #[test]
    fn test_per_token_basic() {
        let data = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0];
        let result = per_token_quantize_i8_scalar(&data, 2, 4, QuantMode::Symmetric);
        assert_eq!(result.scales.len(), 2);
        assert_eq!(result.data.len(), 8);
    }

    #[test]
    fn test_per_token_single_token() {
        let data = vec![3.0, -1.0, 2.0, 0.0];
        let result = per_token_quantize_i8_scalar(&data, 1, 4, QuantMode::Symmetric);
        assert_eq!(result.data[0], 127);
    }

    #[test]
    fn test_per_token_many_tokens() {
        let data: Vec<f32> = (0..256).map(|i| ((i % 17) as f32) - 8.0).collect();
        let result = per_token_quantize_i8_scalar(&data, 32, 8, QuantMode::Symmetric);
        assert_eq!(result.scales.len(), 32);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_per_token_neon_vs_scalar() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.2 - 6.0).collect();
        let scalar = per_token_quantize_i8_scalar(&data, 8, 8, QuantMode::Symmetric);
        let neon = unsafe { per_token_quantize_i8(&data, 8, 8, QuantMode::Symmetric) };
        assert_i8_close(&neon.data, &scalar.data, 1);
    }

    #[test]
    fn test_per_token_dispatch() {
        let data: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
        let result = per_token_quantize_i8_dispatch(&data, 4, 8, QuantMode::Symmetric);
        assert_eq!(result.data.len(), 32);
    }

    // ── symmetric_quantize_i2 tests ───────────────────────────────

    #[test]
    fn test_i2_basic_ternary() {
        let data = vec![1.0, -1.0, 0.0, 0.5];
        let result = symmetric_quantize_i2_scalar(&data, 1, 4, 0.3);
        let byte = result.packed[0];
        assert_eq!(byte & 0x03, 0b01);       // +1
        assert_eq!((byte >> 2) & 0x03, 0b11); // -1
        assert_eq!((byte >> 4) & 0x03, 0b00); // 0
        assert_eq!((byte >> 6) & 0x03, 0b01); // +1
    }

    #[test]
    fn test_i2_all_zeros() {
        let data = vec![0.0; 8];
        let result = symmetric_quantize_i2_scalar(&data, 1, 8, 0.3);
        for &b in &result.packed {
            assert_eq!(b, 0);
        }
    }

    #[test]
    fn test_i2_all_positive() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = symmetric_quantize_i2_scalar(&data, 1, 4, 0.1);
        let byte = result.packed[0];
        assert_eq!(byte & 0x03, 0b01);
        assert_eq!((byte >> 2) & 0x03, 0b01);
        assert_eq!((byte >> 4) & 0x03, 0b01);
        assert_eq!((byte >> 6) & 0x03, 0b01);
    }

    #[test]
    fn test_i2_multi_row() {
        let data = vec![1.0, -1.0, 0.0, 0.0, -2.0, 2.0, 0.0, 0.0];
        let result = symmetric_quantize_i2_scalar(&data, 2, 4, 0.3);
        assert_eq!(result.scales.len(), 2);
        assert_eq!(result.packed.len(), 2);
    }

    #[test]
    fn test_i2_non_multiple_of_4() {
        let data = vec![1.0, -1.0, 0.5, 0.0, -0.5];
        let result = symmetric_quantize_i2_scalar(&data, 1, 5, 0.3);
        assert_eq!(result.packed.len(), 2);
    }

    #[test]
    fn test_i2_threshold_zero() {
        let data = vec![0.1, -0.1, 0.0, 0.001];
        let result = symmetric_quantize_i2_scalar(&data, 1, 4, 0.0);
        let byte = result.packed[0];
        assert_eq!(byte & 0x03, 0b01);       // +1
        assert_eq!((byte >> 2) & 0x03, 0b11); // -1
        assert_eq!((byte >> 4) & 0x03, 0b00); // 0
        assert_eq!((byte >> 6) & 0x03, 0b01); // +1
    }

    #[test]
    fn test_i2_high_threshold() {
        let data = vec![0.5, -0.5, 0.1, -0.1];
        let result = symmetric_quantize_i2_scalar(&data, 1, 4, 0.99);
        let byte = result.packed[0];
        // 0.5/0.5 = 1.0 > 0.99*0.5=0.495 → +1
        assert_eq!(byte & 0x03, 0b01);
        // -0.5/0.5 = -1.0 < -0.495 → -1
        assert_eq!((byte >> 2) & 0x03, 0b11);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_i2_neon_vs_scalar() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let scalar = symmetric_quantize_i2_scalar(&data, 4, 8, 0.3);
        let neon = unsafe { symmetric_quantize_i2(&data, 4, 8, 0.3) };
        assert_eq!(neon.packed, scalar.packed);
        assert_close(&neon.scales, &scalar.scales, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_i2_neon_vs_scalar_odd_size() {
        let data: Vec<f32> = (0..15).map(|i| (i as f32) - 7.0).collect();
        let scalar = symmetric_quantize_i2_scalar(&data, 3, 5, 0.2);
        let neon = unsafe { symmetric_quantize_i2(&data, 3, 5, 0.2) };
        assert_eq!(neon.packed, scalar.packed);
    }

    #[test]
    fn test_i2_dispatch() {
        let data = vec![1.0, -1.0, 0.5, -0.5];
        let result = symmetric_quantize_i2_dispatch(&data, 1, 4, 0.3);
        assert_eq!(result.scales.len(), 1);
    }

    // ── calibrated_quantize tests ─────────────────────────────────

    #[test]
    fn test_calibrated_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let stats = CalibrationStats {
            scales: vec![0.1],
            zero_points: vec![0.0],
            num_samples: 100,
        };
        let result = calibrated_quantize_scalar(&data, 1, 4, &stats);
        assert_eq!(result.data[0], 10);
        assert_eq!(result.data[1], 20);
        assert_eq!(result.data[2], 30);
        assert_eq!(result.data[3], 40);
    }

    #[test]
    fn test_calibrated_with_zero_point() {
        let data = vec![0.0, 0.5, 1.0, 1.5];
        let stats = CalibrationStats {
            scales: vec![0.01],
            zero_points: vec![-50.0],
            num_samples: 50,
        };
        let result = calibrated_quantize_scalar(&data, 1, 4, &stats);
        assert_eq!(result.data[0], -50);
        assert_eq!(result.data[1], 0);
        assert_eq!(result.data[2], 50);
        assert_eq!(result.data[3], 100);
    }

    #[test]
    fn test_calibrated_clamp() {
        let data = vec![100.0];
        let stats = CalibrationStats {
            scales: vec![0.1],
            zero_points: vec![0.0],
            num_samples: 10,
        };
        let result = calibrated_quantize_scalar(&data, 1, 1, &stats);
        assert_eq!(result.data[0], 127);
    }

    #[test]
    fn test_calibrated_multi_row() {
        let data = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        let stats = CalibrationStats {
            scales: vec![0.1, 0.2],
            zero_points: vec![0.0, 0.0],
            num_samples: 20,
        };
        let result = calibrated_quantize_scalar(&data, 2, 4, &stats);
        assert_eq!(result.scales.len(), 2);
    }

    #[test]
    #[should_panic(expected = "not enough calibration scales")]
    fn test_calibrated_insufficient_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let stats = CalibrationStats {
            scales: vec![0.1],
            zero_points: vec![0.0],
            num_samples: 1,
        };
        calibrated_quantize_scalar(&data, 2, 4, &stats);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_calibrated_neon_vs_scalar() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 - 1.5).collect();
        let stats = CalibrationStats {
            scales: vec![0.05, 0.1, 0.02, 0.08],
            zero_points: vec![0.0, 10.0, -5.0, 3.0],
            num_samples: 100,
        };
        let scalar = calibrated_quantize_scalar(&data, 4, 16, &stats);
        let neon = unsafe { calibrated_quantize(&data, 4, 16, &stats) };
        assert_i8_close(&neon.data, &scalar.data, 1);
    }

    #[test]
    fn test_calibrated_dispatch() {
        let data = vec![1.0, -1.0, 0.5, -0.5];
        let stats = CalibrationStats {
            scales: vec![0.1],
            zero_points: vec![0.0],
            num_samples: 10,
        };
        let result = calibrated_quantize_dispatch(&data, 1, 4, &stats);
        assert_eq!(result.data.len(), 4);
    }

    // ── dequantize_i8_f32 tests ───────────────────────────────────

    #[test]
    fn test_dequantize_basic() {
        let data = vec![127i8, -127, 0, 64];
        let scales = vec![0.01];
        let zero_points = vec![0.0];
        let result = dequantize_i8_f32_scalar(&data, 1, 4, &scales, &zero_points);
        assert_close(&result, &[1.27, -1.27, 0.0, 0.64], 1e-5);
    }

    #[test]
    fn test_dequantize_with_zero_point() {
        let data = vec![0i8, 50, 100, -50];
        let scales = vec![0.1];
        let zero_points = vec![10.0];
        let result = dequantize_i8_f32_scalar(&data, 1, 4, &scales, &zero_points);
        assert_close(&result, &[-1.0, 4.0, 9.0, -6.0], 1e-5);
    }

    #[test]
    fn test_dequantize_multi_row() {
        let data = vec![10i8, 20, -10, -20];
        let scales = vec![0.5, 1.0];
        let zero_points = vec![0.0, 0.0];
        let result = dequantize_i8_f32_scalar(&data, 2, 2, &scales, &zero_points);
        assert_close(&result, &[5.0, 10.0, -10.0, -20.0], 1e-5);
    }

    #[test]
    fn test_dequantize_all_zeros() {
        let data = vec![0i8; 8];
        let scales = vec![0.5, 0.5];
        let zero_points = vec![0.0, 0.0];
        let result = dequantize_i8_f32_scalar(&data, 2, 4, &scales, &zero_points);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dequantize_neon_vs_scalar() {
        let data: Vec<i8> = (0..64).map(|i| (i as i8) - 32).collect();
        let scales = vec![0.1, 0.2, 0.05, 0.15];
        let zero_points = vec![0.0, 5.0, -3.0, 2.0];
        let scalar = dequantize_i8_f32_scalar(&data, 4, 16, &scales, &zero_points);
        let neon = unsafe { dequantize_i8_f32(&data, 4, 16, &scales, &zero_points) };
        assert_close(&neon, &scalar, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dequantize_neon_odd_size() {
        let data: Vec<i8> = vec![10, 20, 30, -10, -20];
        let scales = vec![0.5];
        let zero_points = vec![0.0];
        let scalar = dequantize_i8_f32_scalar(&data, 1, 5, &scales, &zero_points);
        let neon = unsafe { dequantize_i8_f32(&data, 1, 5, &scales, &zero_points) };
        assert_close(&neon, &scalar, 1e-5);
    }

    #[test]
    fn test_dequantize_dispatch() {
        let data = vec![10i8, -10, 0, 5];
        let result = dequantize_i8_f32_dispatch(&data, 1, 4, &[0.1], &[0.0]);
        assert_close(&result, &[1.0, -1.0, 0.0, 0.5], 1e-5);
    }

    // ── round-trip tests ──────────────────────────────────────────

    #[test]
    fn test_roundtrip_symmetric_i8() {
        let data = vec![1.0, -0.5, 0.25, -0.75, 0.0, 0.1, -0.1, 0.9];
        let q = per_channel_quantize_i8_scalar(&data, 1, 8, QuantMode::Symmetric);
        let dq = dequantize_i8_f32_scalar(&q.data, 1, 8, &q.scales, &q.zero_points);
        for (i, (&orig, &restored)) in data.iter().zip(dq.iter()).enumerate() {
            let max_err = q.scales[0];
            assert!(
                (orig - restored).abs() <= max_err + 1e-5,
                "index {i}: orig={orig}, restored={restored}, err={}",
                (orig - restored).abs()
            );
        }
    }

    #[test]
    fn test_roundtrip_asymmetric_i8() {
        let data = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let q = per_channel_quantize_i8_scalar(&data, 1, 5, QuantMode::Asymmetric);
        let dq = dequantize_i8_f32_scalar(&q.data, 1, 5, &q.scales, &q.zero_points);
        for (i, (&orig, &restored)) in data.iter().zip(dq.iter()).enumerate() {
            let max_err = q.scales[0] * 2.0;
            assert!(
                (orig - restored).abs() <= max_err + 1e-4,
                "index {i}: orig={orig}, restored={restored}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_roundtrip_neon_symmetric() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let q = unsafe { per_channel_quantize_i8(&data, 2, 16, QuantMode::Symmetric) };
        let dq = unsafe {
            dequantize_i8_f32(&q.data, 2, 16, &q.scales, &q.zero_points)
        };
        for (i, (&orig, &restored)) in data.iter().zip(dq.iter()).enumerate() {
            let row = i / 16;
            let max_err = q.scales[row] + 1e-4;
            assert!(
                (orig - restored).abs() <= max_err,
                "index {i}: orig={orig}, restored={restored}"
            );
        }
    }

    // ── mixed_precision_quantize tests ────────────────────────────

    #[test]
    fn test_mixed_all_i2() {
        let data = vec![0.01, -0.01, 0.0, 0.005];
        let result = mixed_precision_quantize_scalar(&data, 4, 0.05, 0.5);
        assert_eq!(result.precisions[0], Precision::I2);
    }

    #[test]
    fn test_mixed_all_i4() {
        let data = vec![0.1, -0.1, 0.2, -0.2];
        let result = mixed_precision_quantize_scalar(&data, 4, 0.05, 0.5);
        assert_eq!(result.precisions[0], Precision::I4);
    }

    #[test]
    fn test_mixed_all_i8() {
        let data = vec![10.0, -10.0, 5.0, -5.0];
        let result = mixed_precision_quantize_scalar(&data, 4, 0.05, 0.5);
        assert_eq!(result.precisions[0], Precision::I8);
    }

    #[test]
    fn test_mixed_multiple_blocks() {
        let mut data = vec![0.01, -0.01, 0.0, 0.005];
        data.extend_from_slice(&[10.0, -10.0, 5.0, -5.0]);
        let result = mixed_precision_quantize_scalar(&data, 4, 0.05, 0.5);
        assert_eq!(result.precisions.len(), 2);
        assert_eq!(result.precisions[0], Precision::I2);
        assert_eq!(result.precisions[1], Precision::I8);
    }

    #[test]
    fn test_mixed_partial_last_block() {
        let data = vec![1.0, -1.0, 0.5, -0.5, 0.1];
        let result = mixed_precision_quantize_scalar(&data, 4, 0.05, 0.5);
        assert_eq!(result.precisions.len(), 2);
        assert_eq!(result.data.len(), 5);
    }

    #[test]
    fn test_mixed_block_size_1() {
        let data = vec![0.01, 5.0, 0.2];
        let result = mixed_precision_quantize_scalar(&data, 1, 0.05, 0.5);
        assert_eq!(result.precisions.len(), 3);
        // Single-element blocks have range 0 → I2
        assert_eq!(result.precisions[0], Precision::I2);
        assert_eq!(result.precisions[1], Precision::I2);
        assert_eq!(result.precisions[2], Precision::I2);
    }

    #[test]
    #[should_panic(expected = "block_size must be > 0")]
    fn test_mixed_zero_block_size() {
        mixed_precision_quantize_scalar(&[1.0], 0, 0.05, 0.5);
    }

    #[test]
    fn test_mixed_empty_data() {
        let data: Vec<f32> = vec![];
        let result = mixed_precision_quantize_scalar(&data, 4, 0.05, 0.5);
        assert_eq!(result.precisions.len(), 0);
        assert_eq!(result.data.len(), 0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_mixed_neon_vs_scalar() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.5 - 8.0).collect();
        let scalar = mixed_precision_quantize_scalar(&data, 8, 0.5, 5.0);
        let neon = unsafe { mixed_precision_quantize(&data, 8, 0.5, 5.0) };
        assert_eq!(neon.precisions, scalar.precisions);
        assert_i8_close(&neon.data, &scalar.data, 1);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_mixed_neon_vs_scalar_varied() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0.001, -0.001, 0.0, 0.0]);
        data.extend_from_slice(&[0.3, -0.3, 0.1, -0.1]);
        data.extend_from_slice(&[5.0, -5.0, 3.0, -3.0]);
        let scalar = mixed_precision_quantize_scalar(&data, 4, 0.01, 1.0);
        let neon = unsafe { mixed_precision_quantize(&data, 4, 0.01, 1.0) };
        assert_eq!(neon.precisions, scalar.precisions);
    }

    #[test]
    fn test_mixed_dispatch() {
        let data = vec![1.0, -1.0, 0.5, -0.5];
        let result = mixed_precision_quantize_dispatch(&data, 4, 0.05, 0.5);
        assert_eq!(result.precisions.len(), 1);
    }

    // ── edge case tests ───────────────────────────────────────────

    #[test]
    fn test_nan_handling() {
        let data = vec![1.0, f32::NAN, 0.0, -1.0];
        let result = per_channel_quantize_i8_scalar(&data, 1, 4, QuantMode::Symmetric);
        assert_eq!(result.data.len(), 4);
    }

    #[test]
    fn test_inf_handling() {
        let data = vec![f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.0];
        let result = per_channel_quantize_i8_scalar(&data, 1, 4, QuantMode::Symmetric);
        assert_eq!(result.data.len(), 4);
    }

    #[test]
    fn test_subnormal_values() {
        let data = vec![f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 0.0, f32::MIN_POSITIVE / 2.0];
        let result = per_channel_quantize_i8_scalar(&data, 1, 4, QuantMode::Symmetric);
        assert_eq!(result.data.len(), 4);
    }

    #[test]
    fn test_large_tensor_shape() {
        let data: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let result = per_channel_quantize_i8_scalar(&data, 32, 32, QuantMode::Symmetric);
        assert_eq!(result.data.len(), 1024);
        assert_eq!(result.scales.len(), 32);
    }

    #[test]
    fn test_single_element_tensor() {
        let data = vec![0.42];
        let q = per_channel_quantize_i8_scalar(&data, 1, 1, QuantMode::Symmetric);
        let dq = dequantize_i8_f32_scalar(&q.data, 1, 1, &q.scales, &q.zero_points);
        assert!((data[0] - dq[0]).abs() < q.scales[0] + 1e-5);
    }

    #[test]
    fn test_identical_values() {
        let data = vec![3.14; 16];
        let result = per_channel_quantize_i8_scalar(&data, 1, 16, QuantMode::Symmetric);
        let first = result.data[0];
        for &v in &result.data {
            assert_eq!(v, first);
        }
    }

    #[test]
    fn test_alternating_sign_pattern() {
        let data: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let result = per_channel_quantize_i8_scalar(&data, 1, 16, QuantMode::Symmetric);
        for (i, &v) in result.data.iter().enumerate() {
            if i % 2 == 0 {
                assert!(v > 0);
            } else {
                assert!(v < 0);
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_large_tensor() {
        let data: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.001 - 2.0).collect();
        let scalar = per_channel_quantize_i8_scalar(&data, 64, 64, QuantMode::Symmetric);
        let neon = unsafe { per_channel_quantize_i8(&data, 64, 64, QuantMode::Symmetric) };
        assert_i8_close(&neon.data, &scalar.data, 1);
    }

    #[test]
    fn test_precision_enum_equality() {
        assert_eq!(Precision::I2, Precision::I2);
        assert_ne!(Precision::I2, Precision::I4);
        assert_ne!(Precision::I4, Precision::I8);
    }

    #[test]
    fn test_quant_mode_enum() {
        assert_eq!(QuantMode::Symmetric, QuantMode::Symmetric);
        assert_ne!(QuantMode::Symmetric, QuantMode::Asymmetric);
    }

    #[test]
    fn test_calibration_stats_clone() {
        let stats = CalibrationStats {
            scales: vec![1.0, 2.0],
            zero_points: vec![0.0, 0.0],
            num_samples: 42,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.num_samples, 42);
        assert_eq!(cloned.scales, stats.scales);
    }

    #[test]
    fn test_quantized_output_debug() {
        let out = QuantizedOutput {
            data: vec![1, -1, 0],
            scales: vec![0.5],
            zero_points: vec![0.0],
        };
        let dbg = format!("{out:?}");
        assert!(dbg.contains("QuantizedOutput"));
    }
}
