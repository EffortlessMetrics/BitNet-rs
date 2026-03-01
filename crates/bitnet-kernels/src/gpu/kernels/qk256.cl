/// QK256 quantization kernels for BitNet GPU inference.
///
/// QK256 uses 256-element blocks with per-block float16 scales.
/// Each block stores 256 ternary values packed as 2-bit pairs (4 per byte = 64 bytes)
/// plus a float16 scale factor.

/// Dequantize a QK256 block: unpack 2-bit ternary values and apply scale.
__kernel void qk256_dequantize(
    __global const uchar* packed_data,
    __global const half* scales,
    __global float* output,
    const uint num_blocks
) {
    const uint block_id = get_global_id(0) / 256;
    const uint elem_id = get_global_id(0) % 256;
    if (block_id >= num_blocks) return;

    const uint byte_offset = block_id * 64 + elem_id / 4;
    const uint bit_shift = (elem_id % 4) * 2;
    uchar packed_byte = packed_data[byte_offset];
    uchar bits = (packed_byte >> bit_shift) & 0x03;

    float value;
    if (bits == 0x01) value = 1.0f;
    else if (bits == 0x03) value = -1.0f;
    else value = 0.0f;

    float scale = vload_half(block_id, (__global const half*)scales);
    output[block_id * 256 + elem_id] = value * scale;
}

/// QK256 matrix multiplication: quantized weight * float activation.
__kernel void qk256_matmul(
    __global const uchar* weights,
    __global const half* weight_scales,
    __global const float* activations,
    __global float* output,
    const uint N,
    const uint K,
    const uint blocks_per_row
) {
    const uint row = get_global_id(0);
    if (row >= N) return;

    float sum = 0.0f;
    for (uint b = 0; b < blocks_per_row; b++) {
        uint block_idx = row * blocks_per_row + b;
        float scale = vload_half(block_idx, (__global const half*)weight_scales);
        uint data_offset = block_idx * 64;
        uint k_base = b * 256;

        for (uint byte_i = 0; byte_i < 64; byte_i++) {
            uchar packed = weights[data_offset + byte_i];
            for (uint sub = 0; sub < 4; sub++) {
                uint k = k_base + byte_i * 4 + sub;
                if (k >= K) break;
                uchar bits = (packed >> (sub * 2)) & 0x03;
                float w;
                if (bits == 0x01) w = 1.0f;
                else if (bits == 0x03) w = -1.0f;
                else w = 0.0f;
                sum += w * scale * activations[k];
            }
        }
    }
    output[row] = sum;
}

/// Apply per-block scales to pre-accumulated integer results.
__kernel void qk256_apply_scales(
    __global const int* accumulated,
    __global const half* scales,
    __global float* output,
    const uint num_entries
) {
    const uint i = get_global_id(0);
    if (i >= num_entries) return;
    float scale = vload_half(i, (__global const half*)scales);
    output[i] = (float)accumulated[i] * scale;
}
