// Metal Shading Language (MSL) — Scaled dot-product attention kernel
// attn = softmax(Q @ K^T / sqrt(d_k)) @ V
// Each threadgroup processes one query position.

#include <metal_stdlib>
using namespace metal;

struct AttentionParams {
    uint seq_len;
    uint head_dim;
    uint kv_len;
};

kernel void attention(
    device const float* q     [[buffer(0)]],
    device const float* k     [[buffer(1)]],
    device const float* v     [[buffer(2)]],
    device float* output      [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg  [[threadgroup_position_in_grid]])
{
    uint query_pos = tg;
    float scale = rsqrt(float(params.head_dim));

    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];

    // Phase 1: compute Q·K^T scores and find max
    float local_max = -MAXFLOAT;
    for (uint kv = tid; kv < params.kv_len; kv += 256) {
        float dot = 0.0f;
        for (uint d = 0; d < params.head_dim; d++) {
            dot += q[query_pos * params.head_dim + d] * k[kv * params.head_dim + d];
        }
        float score = dot * scale;
        local_max = max(local_max, score);
    }
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_max[0];

    // Phase 2: softmax denominator
    float local_sum = 0.0f;
    for (uint kv = tid; kv < params.kv_len; kv += 256) {
        float dot = 0.0f;
        for (uint d = 0; d < params.head_dim; d++) {
            dot += q[query_pos * params.head_dim + d] * k[kv * params.head_dim + d];
        }
        local_sum += exp(dot * scale - row_max);
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total = shared_sum[0];

    // Phase 3: weighted sum of values
    for (uint d = tid; d < params.head_dim; d += 256) {
        float acc = 0.0f;
        for (uint kv = 0; kv < params.kv_len; kv++) {
            float dot = 0.0f;
            for (uint dd = 0; dd < params.head_dim; dd++) {
                dot += q[query_pos * params.head_dim + dd] * k[kv * params.head_dim + dd];
            }
            float w = exp(dot * scale - row_max) / total;
            acc += w * v[kv * params.head_dim + d];
        }
        output[query_pos * params.head_dim + d] = acc;
    }
}
