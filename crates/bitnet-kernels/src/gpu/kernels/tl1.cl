/// TL1 (Ternary Level 1) kernels for BitNet GPU inference.
///
/// TL1 packs ternary values {-1, 0, +1} into 2 bits per element
/// with a single per-tensor or per-row scale factor.

/// Pack float32 values into TL1 ternary format.
/// Encoding: 0b00 = 0, 0b01 = +1, 0b11 = -1
__kernel void tl1_pack(
    __global const float* input,
    __global uchar* packed_output,
    __global float* scales,
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
    scales[group_id] = scale;

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

/// Unpack TL1 ternary format to float32.
__kernel void tl1_unpack(
    __global const uchar* packed_input,
    __global const float* scales,
    __global float* output,
    const uint N,
    const uint group_size
) {
    const uint i = get_global_id(0);
    if (i >= N) return;

    uint group_id = i / group_size;
    float scale = scales[group_id];
    uint byte_idx = i / 4;
    uint bit_shift = (i % 4) * 2;

    uchar packed = packed_input[byte_idx];
    uchar bits = (packed >> bit_shift) & 0x03;

    float value;
    if (bits == 0x01) value = 1.0f;
    else if (bits == 0x03) value = -1.0f;
    else value = 0.0f;

    output[i] = value * scale;
}

/// TL1 matrix multiplication: packed ternary weights * float activations.
/// W is [N x K/4] packed, x is [K] float, output is [N] float.
__kernel void tl1_matmul(
    __global const uchar* weights,
    __global const float* scales,
    __global const float* input,
    __global float* output,
    const uint N,
    const uint K
) {
    const uint row = get_global_id(0);
    if (row >= N) return;

    float sum = 0.0f;
    const uint k_packed = K / 4;

    for (uint kp = 0; kp < k_packed; kp++) {
        uchar packed = weights[row * k_packed + kp];
        for (uint sub = 0; sub < 4; sub++) {
            uint k = kp * 4 + sub;
            if (k >= K) break;
            uchar bits = (packed >> (sub * 2)) & 0x03;
            float w;
            if (bits == 0x01) w = 1.0f;
            else if (bits == 0x03) w = -1.0f;
            else w = 0.0f;
            sum += w * input[k];
        }
    }
    output[row] = sum * scales[row];
}
