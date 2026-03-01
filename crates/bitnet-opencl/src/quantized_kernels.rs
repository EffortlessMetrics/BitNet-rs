//! `OpenCL` kernel source strings for quantized inference operations.
//!
//! Each kernel is embedded as a `const &str` so the crate remains
//! self-contained (no external `.cl` files needed at runtime).

// ── I2_S dequantization ──────────────────────────────────────────────────────

/// `OpenCL` kernel for `I2_S` dequantization.
///
/// **Workgroup**: 256 work-items (one per output element group).
/// **Memory**: No local memory; reads from global `packed` buffer.
///
/// Each work-item unpacks one byte (4 weights) and writes 4 consecutive
/// output floats.
///
/// Args (via `clSetKernelArg`):
///   0: `__global const uchar *packed` — `I2_S` packed weight bytes
///   1: `__global float *output`       — dequantized f32 output
///   2: `float scale`                  — per-tensor scale factor
///   3: `uint n_bytes`                 — number of packed bytes
pub const DEQUANTIZE_I2S_CL: &str = r"
__kernel void dequantize_i2s(
    __global const uchar *packed,
    __global float *output,
    const float scale,
    const uint n_bytes)
{
    uint gid = get_global_id(0);
    if (gid >= n_bytes) return;

    uchar byte = packed[gid];
    uint base = gid * 4;

    // 2-bit decode: 00->0, 01->+1, 10->-1, 11->0(reserved)
    for (int i = 0; i < 4; i++) {
        uchar code = (byte >> (i * 2)) & 0x3;
        float w = (code == 1) ? 1.0f : ((code == 2) ? -1.0f : 0.0f);
        output[base + i] = w * scale;
    }
}
";

/// Recommended workgroup size for [`DEQUANTIZE_I2S_CL`].
pub const DEQUANTIZE_I2S_WORKGROUP: usize = 256;

// ── Ternary matrix multiply ──────────────────────────────────────────────────

/// `OpenCL` kernel for ternary matrix-vector multiply.
///
/// **Workgroup**: 256 work-items (one per output row tile).
/// **Memory**: No local memory; row-parallel reduction.
///
/// Each work-item computes one output row by iterating over packed weight
/// bytes and accumulating dot products.
///
/// Args:
///   0: `__global const uchar *weights` — packed ternary weights
///   1: `__global const float *input`   — input vector
///   2: `__global float *output`        — output vector
///   3: `float scale`                   — scale factor
///   4: `uint rows`
///   5: `uint cols`
///   6: `uint cols_packed`              — `cols.div_ceil(4)`
pub const TERNARY_MATMUL_CL: &str = r"
__kernel void ternary_matmul(
    __global const uchar *weights,
    __global const float *input,
    __global float *output,
    const float scale,
    const uint rows,
    const uint cols,
    const uint cols_packed)
{
    uint row = get_global_id(0);
    if (row >= rows) return;

    float acc = 0.0f;
    uint row_offset = row * cols_packed;

    for (uint byte_idx = 0; byte_idx < cols_packed; byte_idx++) {
        uchar byte = weights[row_offset + byte_idx];
        uint base_col = byte_idx * 4;

        for (int sub = 0; sub < 4; sub++) {
            uint col = base_col + sub;
            if (col >= cols) break;
            uchar code = (byte >> (sub * 2)) & 0x3;
            float w = (code == 1) ? 1.0f : ((code == 2) ? -1.0f : 0.0f);
            acc += w * input[col];
        }
    }

    output[row] = acc * scale;
}
";

/// Recommended workgroup size for [`TERNARY_MATMUL_CL`].
pub const TERNARY_MATMUL_WORKGROUP: usize = 256;

/// Minimum global memory required (bytes) for the ternary matmul kernel,
/// given matrix dimensions.
pub const fn ternary_matmul_mem_bytes(rows: usize, cols: usize) -> usize {
    let cols_packed = cols.div_ceil(4);
    // weights + input + output
    rows * cols_packed + cols * 4 + rows * 4
}

// ── QK256 block dequantization ───────────────────────────────────────────────

/// `OpenCL` kernel for QK256 block dequantization.
///
/// **Workgroup**: 256 work-items (one per block).
/// **Memory**: No local memory.
///
/// Each work-item dequantizes one 66-byte QK256 block into 256 floats.
///
/// Args:
///   0: `__global const uchar *blocks` — packed QK256 blocks (66 bytes each)
///   1: `__global float *output`       — dequantized output
///   2: `uint n_blocks`
pub const QK256_DEQUANT_CL: &str = r"
__kernel void qk256_dequant(
    __global const uchar *blocks,
    __global float *output,
    const uint n_blocks)
{
    uint bid = get_global_id(0);
    if (bid >= n_blocks) return;

    // 66 bytes per block: 2-byte f16 scale + 64 bytes of 2-bit weights
    uint block_offset = bid * 66;
    uchar lo = blocks[block_offset];
    uchar hi = blocks[block_offset + 1];

    // f16 -> f32 conversion
    ushort f16_bits = (ushort)lo | ((ushort)hi << 8);
    uint sign  = (f16_bits >> 15) & 1;
    uint exp   = (f16_bits >> 10) & 0x1F;
    uint frac  = f16_bits & 0x3FF;
    float scale;
    if (exp == 0) {
        scale = (float)frac * (1.0f / 16777216.0f);
        if (sign) scale = -scale;
    } else if (exp == 0x1F) {
        scale = 0.0f;
    } else {
        uint f32_bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
        scale = as_float(f32_bits);
    }

    uint out_base = bid * 256;
    for (uint i = 0; i < 64; i++) {
        uchar byte = blocks[block_offset + 2 + i];
        for (int j = 0; j < 4; j++) {
            uchar code = (byte >> (j * 2)) & 0x3;
            float w = (code == 1) ? 1.0f : ((code == 2) ? -1.0f : 0.0f);
            output[out_base + i * 4 + j] = w * scale;
        }
    }
}
";

/// Recommended workgroup size for [`QK256_DEQUANT_CL`].
pub const QK256_DEQUANT_WORKGROUP: usize = 256;

/// Returns all kernel sources paired with their entry-point names.
///
/// Useful for batch compilation into an `OpenCL` program.
pub fn all_kernel_sources() -> Vec<(&'static str, &'static str)> {
    vec![
        ("dequantize_i2s", DEQUANTIZE_I2S_CL),
        ("ternary_matmul", TERNARY_MATMUL_CL),
        ("qk256_dequant", QK256_DEQUANT_CL),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_sources_non_empty() {
        assert!(!DEQUANTIZE_I2S_CL.is_empty());
        assert!(!TERNARY_MATMUL_CL.is_empty());
        assert!(!QK256_DEQUANT_CL.is_empty());
    }

    #[test]
    fn kernel_sources_contain_entry_points() {
        assert!(DEQUANTIZE_I2S_CL.contains("dequantize_i2s"));
        assert!(TERNARY_MATMUL_CL.contains("ternary_matmul"));
        assert!(QK256_DEQUANT_CL.contains("qk256_dequant"));
    }

    #[test]
    fn all_kernel_sources_complete() {
        let sources = all_kernel_sources();
        assert_eq!(sources.len(), 3);
    }

    #[test]
    fn ternary_matmul_mem_estimate() {
        // 128×64 matrix
        let mem = ternary_matmul_mem_bytes(128, 64);
        let cols_packed = (64 + 3) / 4; // 16
        let expected = 128 * cols_packed + 64 * 4 + 128 * 4;
        assert_eq!(mem, expected);
    }

    #[test]
    fn workgroup_sizes_are_power_of_two() {
        assert!(DEQUANTIZE_I2S_WORKGROUP.is_power_of_two());
        assert!(TERNARY_MATMUL_WORKGROUP.is_power_of_two());
        assert!(QK256_DEQUANT_WORKGROUP.is_power_of_two());
    }
}
