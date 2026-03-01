/// TL2 (Ternary Level 2) kernels for BitNet GPU inference.
///
/// TL2 extends TL1 with per-group scale factors (finer granularity).
/// Groups are typically 32 or 64 elements, each with an f32 scale.

/// Dequantize TL2 packed data with per-group scales.
__kernel void tl2_dequantize(
    __global const uchar* packed_data,
    __global const float* group_scales,
    __global float* output,
    const uint N,
    const uint group_size
) {
    const uint i = get_global_id(0);
    if (i >= N) return;

    uint group_id = i / group_size;
    float scale = group_scales[group_id];
    uint byte_idx = i / 4;
    uint bit_shift = (i % 4) * 2;

    uchar packed = packed_data[byte_idx];
    uchar bits = (packed >> bit_shift) & 0x03;

    float value;
    if (bits == 0x01) value = 1.0f;
    else if (bits == 0x03) value = -1.0f;
    else value = 0.0f;

    output[i] = value * scale;
}

/// TL2 quantize: float32 to packed ternary with per-group scales.
__kernel void tl2_quantize(
    __global const float* input,
    __global uchar* packed_output,
    __global float* group_scales,
    const uint N,
    const uint group_size
) {
    const uint group_id = get_global_id(0);
    const uint num_groups = (N + group_size - 1) / group_size;
    if (group_id >= num_groups) return;

    const uint start = group_id * group_size;
    const uint end = min(start + group_size, N);

    float absmax = 0.0f;
    for (uint i = start; i < end; i++) {
        absmax = fmax(absmax, fabs(input[i]));
    }
    float scale = absmax > 0.0f ? absmax : 1.0f;
    group_scales[group_id] = scale;

    for (uint i = start; i < end; i += 4) {
        uchar packed = 0;
        for (uint sub = 0; sub < 4 && (i + sub) < end; sub++) {
            float normalized = input[i + sub] / scale;
            uchar ternary;
            if (normalized > 0.5f) ternary = 0x01;
            else if (normalized < -0.5f) ternary = 0x03;
            else ternary = 0x00;
            packed |= ternary << (sub * 2);
        }
        packed_output[i / 4] = packed;
    }
}

/// TL2 matrix multiplication with per-group scales.
/// W is [N x K/4] packed ternary with per-group scales.
__kernel void tl2_matmul(
    __global const uchar* weights,
    __global const float* group_scales,
    __global const float* input,
    __global float* output,
    const uint N,
    const uint K,
    const uint group_size
) {
    const uint row = get_global_id(0);
    if (row >= N) return;

    const uint k_packed = K / 4;
    const uint groups_per_row = (K + group_size - 1) / group_size;
    float sum = 0.0f;

    for (uint kp = 0; kp < k_packed; kp++) {
        uchar packed = weights[row * k_packed + kp];
        uint k_base = kp * 4;

        for (uint sub = 0; sub < 4; sub++) {
            uint k = k_base + sub;
            if (k >= K) break;
            uint group_id = row * groups_per_row + k / group_size;
            float scale = group_scales[group_id];
            uchar bits = (packed >> (sub * 2)) & 0x03;
            float w;
            if (bits == 0x01) w = scale;
            else if (bits == 0x03) w = -scale;
            else w = 0.0f;
            sum += w * input[k];
        }
    }
    output[row] = sum;
}
