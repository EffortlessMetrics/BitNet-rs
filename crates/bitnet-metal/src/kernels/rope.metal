// Metal Shading Language (MSL) — Rotary Position Embedding kernel
// Applies rotary embeddings to query/key vectors in-place.
// Each thread handles one (position, dimension_pair).

#include <metal_stdlib>
using namespace metal;

struct RopeParams {
    uint seq_len;
    uint head_dim;
    uint num_heads;
};

kernel void rope(
    device float* data       [[buffer(0)]],
    device const float* freq_cos [[buffer(1)]],
    device const float* freq_sin [[buffer(2)]],
    constant RopeParams& params  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint half_dim = params.head_dim / 2;
    uint total_pairs = params.seq_len * params.num_heads * half_dim;

    if (gid >= total_pairs) return;

    uint pair_in_head = gid % half_dim;
    uint head_and_pos = gid / half_dim;
    uint head = head_and_pos % params.num_heads;
    uint pos = head_and_pos / params.num_heads;

    uint base = (pos * params.num_heads + head) * params.head_dim;
    uint idx0 = base + pair_in_head;
    uint idx1 = base + pair_in_head + half_dim;

    uint freq_idx = pos * half_dim + pair_in_head;
    float cos_val = freq_cos[freq_idx];
    float sin_val = freq_sin[freq_idx];

    float x0 = data[idx0];
    float x1 = data[idx1];

    data[idx0] = x0 * cos_val - x1 * sin_val;
    data[idx1] = x0 * sin_val + x1 * cos_val;
}
