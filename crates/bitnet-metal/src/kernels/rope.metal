#include <metal_stdlib>
using namespace metal;

/// Rotary Position Embeddings (RoPE).
/// Applies rotation to pairs of elements in the input tensor.
/// dims.x = batch_size, dims.y = seq_len, dims.z = head_dim
/// The head_dim is expected to be even (pairs of elements are rotated).
kernel void rope(
    device float* input [[buffer(0)]],
    device const float* cos_table [[buffer(1)]],
    device const float* sin_table [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    constant uint& position_offset [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint batch = gid.z;
    const uint seq_pos = gid.y;
    const uint pair_idx = gid.x;

    const uint batch_size = dims.x;
    const uint seq_len = dims.y;
    const uint head_dim = dims.z;
    const uint half_dim = head_dim / 2;

    if (batch >= batch_size || seq_pos >= seq_len || pair_idx >= half_dim) {
        return;
    }

    const uint pos = position_offset + seq_pos;
    const uint cos_sin_idx = pos * half_dim + pair_idx;
    const float cos_val = cos_table[cos_sin_idx];
    const float sin_val = sin_table[cos_sin_idx];

    const uint base_idx = batch * seq_len * head_dim + seq_pos * head_dim;
    const uint idx_re = base_idx + pair_idx;
    const uint idx_im = base_idx + pair_idx + half_dim;

    const float x_re = input[idx_re];
    const float x_im = input[idx_im];

    // Apply rotation: [cos, -sin; sin, cos] * [x_re, x_im]
    input[idx_re] = x_re * cos_val - x_im * sin_val;
    input[idx_im] = x_re * sin_val + x_im * cos_val;
}

/// Build cos/sin frequency tables for RoPE.
/// dims.x = max_seq_len, dims.y = half_head_dim
kernel void rope_build_tables(
    device float* cos_table [[buffer(0)]],
    device float* sin_table [[buffer(1)]],
    constant uint2& dims [[buffer(2)]],
    constant float& theta_base [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint pos = gid.y;
    const uint dim_idx = gid.x;
    const uint max_seq_len = dims.x;
    const uint half_dim = dims.y;

    if (pos >= max_seq_len || dim_idx >= half_dim) {
        return;
    }

    const float freq = 1.0f / pow(theta_base, float(2 * dim_idx) / float(2 * half_dim));
    const float angle = float(pos) * freq;

    const uint idx = pos * half_dim + dim_idx;
    cos_table[idx] = cos(angle);
    sin_table[idx] = sin(angle);
}
