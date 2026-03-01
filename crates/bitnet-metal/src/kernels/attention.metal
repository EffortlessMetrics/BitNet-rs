#include <metal_stdlib>
using namespace metal;

/// Scaled dot-product attention with optional causal mask.
/// Q: [batch, heads, seq_q, head_dim]   - query
/// K: [batch, heads, seq_k, head_dim]   - key
/// V: [batch, heads, seq_k, head_dim]   - value
/// output: [batch, heads, seq_q, head_dim]
///
/// attention_dims: (batch, heads, seq_q, seq_k)
/// head_dim passed separately in buffer(5)
/// causal: if non-zero, applies causal mask
constant uint ATTN_THREADGROUP_SIZE = 256;

kernel void attention_scores(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint4& attention_dims [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& causal [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint batch = gid.z / attention_dims.y;
    const uint head = gid.z % attention_dims.y;
    const uint q_pos = gid.y;
    const uint k_pos = gid.x;

    const uint n_batch = attention_dims.x;
    const uint n_heads = attention_dims.y;
    const uint seq_q = attention_dims.z;
    const uint seq_k = attention_dims.w;

    if (batch >= n_batch || head >= n_heads ||
        q_pos >= seq_q || k_pos >= seq_k) {
        return;
    }

    // Causal mask: future positions get -inf
    if (causal != 0 && k_pos > q_pos) {
        const uint score_idx = ((batch * n_heads + head) * seq_q + q_pos) * seq_k + k_pos;
        scores[score_idx] = -INFINITY;
        return;
    }

    const float scale = rsqrt(float(head_dim));

    // Compute dot product Q[q_pos] · K[k_pos]
    const uint q_offset = ((batch * n_heads + head) * seq_q + q_pos) * head_dim;
    const uint k_offset = ((batch * n_heads + head) * seq_k + k_pos) * head_dim;

    float dot = 0.0f;
    for (uint i = 0; i < head_dim; i++) {
        dot += q[q_offset + i] * k[k_offset + i];
    }

    const uint score_idx = ((batch * n_heads + head) * seq_q + q_pos) * seq_k + k_pos;
    scores[score_idx] = dot * scale;
}

/// Attention weighted sum: output = softmax(scores) @ V
/// scores: [batch, heads, seq_q, seq_k] (already softmaxed)
/// V: [batch, heads, seq_k, head_dim]
/// output: [batch, heads, seq_q, head_dim]
kernel void attention_weighted_sum(
    device const float* scores [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint4& attention_dims [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint batch = gid.z / attention_dims.y;
    const uint head = gid.z % attention_dims.y;
    const uint q_pos = gid.y;
    const uint d = gid.x;

    const uint n_batch = attention_dims.x;
    const uint n_heads = attention_dims.y;
    const uint seq_q = attention_dims.z;
    const uint seq_k = attention_dims.w;

    if (batch >= n_batch || head >= n_heads ||
        q_pos >= seq_q || d >= head_dim) {
        return;
    }

    float sum = 0.0f;
    const uint score_offset = ((batch * n_heads + head) * seq_q + q_pos) * seq_k;
    const uint v_base = (batch * n_heads + head) * seq_k;

    for (uint k = 0; k < seq_k; k++) {
        sum += scores[score_offset + k] * v[(v_base + k) * head_dim + d];
    }

    const uint out_idx = ((batch * n_heads + head) * seq_q + q_pos) * head_dim + d;
    output[out_idx] = sum;
}
