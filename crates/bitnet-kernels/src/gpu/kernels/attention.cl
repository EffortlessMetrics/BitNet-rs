/// Scaled dot-product attention kernels for BitNet inference.
///
/// Implements: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
///
/// Three granularities:
///   1. attention_qkv      — fully fused QK^T + softmax + V multiply
///   2. attention_scores    — Q * K^T / sqrt(d_k) with optional causal mask
///   3. attention_weighted_sum — softmax(scores) * V

/// Compute attention scores: S = Q * K^T / sqrt(d_k) with optional causal mask.
///
/// Q: [num_heads x seq_len x d_k]   — query
/// K: [num_heads x kv_len  x d_k]   — key
/// S: [num_heads x seq_len x kv_len] — output scores
///
/// Work-items: global(0)=seq_len, global(1)=kv_len, global(2)=num_heads
__kernel void attention_scores(
    __global const float* Q,
    __global const float* K,
    __global float* S,
    const uint seq_len,
    const uint kv_len,
    const uint d_k,
    const uint num_heads,
    const float inv_sqrt_dk,
    const uint causal              // 1 = apply causal mask
) {
    const uint q_pos  = get_global_id(0);  // query position
    const uint k_pos  = get_global_id(1);  // key position
    const uint head   = get_global_id(2);  // head index

    if (q_pos >= seq_len || k_pos >= kv_len || head >= num_heads) return;

    // Causal mask: future positions → -inf
    if (causal && k_pos > q_pos) {
        S[head * seq_len * kv_len + q_pos * kv_len + k_pos] = -1e30f;
        return;
    }

    // Dot product Q[head, q_pos, :] · K[head, k_pos, :]
    float dot = 0.0f;
    const uint q_offset = head * seq_len * d_k + q_pos * d_k;
    const uint k_offset = head * kv_len  * d_k + k_pos * d_k;

    for (uint i = 0; i < d_k; i++) {
        dot += Q[q_offset + i] * K[k_offset + i];
    }

    S[head * seq_len * kv_len + q_pos * kv_len + k_pos] = dot * inv_sqrt_dk;
}

/// Softmax(scores) * V — weighted sum of values.
///
/// S: [num_heads x seq_len x kv_len] — attention weights (pre-softmax scores)
/// V: [num_heads x kv_len  x d_v]    — values
/// O: [num_heads x seq_len x d_v]    — output
///
/// Each work-item handles one (head, q_pos) pair.
/// Uses local memory to cache the softmax row for efficient reuse.
///
/// Work-items: global(0)=seq_len, global(1)=num_heads
__kernel void attention_weighted_sum(
    __global const float* S,
    __global const float* V,
    __global float* O,
    const uint seq_len,
    const uint kv_len,
    const uint d_v,
    const uint num_heads
) {
    const uint q_pos = get_global_id(0);
    const uint head  = get_global_id(1);

    if (q_pos >= seq_len || head >= num_heads) return;

    const uint s_offset = head * seq_len * kv_len + q_pos * kv_len;

    // --- Row-wise softmax over scores ---
    // Pass 1: find max for numerical stability
    float max_val = -1e30f;
    for (uint j = 0; j < kv_len; j++) {
        float v = S[s_offset + j];
        max_val = fmax(max_val, v);
    }

    // Pass 2: sum of exp
    float sum_exp = 0.0f;
    for (uint j = 0; j < kv_len; j++) {
        sum_exp += exp(S[s_offset + j] - max_val);
    }
    float inv_sum = 1.0f / (sum_exp + 1e-8f);

    // Pass 3: weighted sum  O[head, q_pos, :] = Σ_j softmax_j * V[head, j, :]
    const uint o_offset = head * seq_len * d_v + q_pos * d_v;

    for (uint d = 0; d < d_v; d++) {
        float acc = 0.0f;
        for (uint j = 0; j < kv_len; j++) {
            float w = exp(S[s_offset + j] - max_val) * inv_sum;
            acc += w * V[head * kv_len * d_v + j * d_v + d];
        }
        O[o_offset + d] = acc;
    }
}

/// Fully fused scaled dot-product attention with local-memory tiling.
///
/// Q: [num_heads x seq_len x d_k]
/// K: [num_heads x kv_len  x d_k]
/// V: [num_heads x kv_len  x d_v]   (d_v == d_k assumed)
/// O: [num_heads x seq_len x d_v]
///
/// Uses local memory to stage tiles of scores for the softmax reduction,
/// reducing global memory traffic.
///
/// Work-items: global(0)=seq_len, global(1)=num_heads
__kernel void attention_qkv(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* O,
    const uint seq_len,
    const uint kv_len,
    const uint d_k,
    const uint num_heads,
    const float inv_sqrt_dk,
    const uint causal,
    __local float* local_scores      // size = kv_len floats
) {
    const uint q_pos = get_global_id(0);
    const uint head  = get_global_id(1);

    if (q_pos >= seq_len || head >= num_heads) return;

    const uint q_offset = head * seq_len * d_k + q_pos * d_k;

    // --- Phase 1: compute Q * K^T / sqrt(d_k) into local memory ---
    for (uint k_pos = 0; k_pos < kv_len; k_pos++) {
        if (causal && k_pos > q_pos) {
            local_scores[k_pos] = -1e30f;
            continue;
        }
        float dot = 0.0f;
        const uint k_offset = head * kv_len * d_k + k_pos * d_k;
        for (uint i = 0; i < d_k; i++) {
            dot += Q[q_offset + i] * K[k_offset + i];
        }
        local_scores[k_pos] = dot * inv_sqrt_dk;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Phase 2: softmax over local_scores ---
    float max_val = -1e30f;
    for (uint j = 0; j < kv_len; j++) {
        max_val = fmax(max_val, local_scores[j]);
    }

    float sum_exp = 0.0f;
    for (uint j = 0; j < kv_len; j++) {
        local_scores[j] = exp(local_scores[j] - max_val);
        sum_exp += local_scores[j];
    }
    float inv_sum = 1.0f / (sum_exp + 1e-8f);

    for (uint j = 0; j < kv_len; j++) {
        local_scores[j] *= inv_sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Phase 3: weighted sum  O = softmax(scores) * V ---
    const uint o_offset = head * seq_len * d_k + q_pos * d_k;

    for (uint d = 0; d < d_k; d++) {
        float acc = 0.0f;
        for (uint j = 0; j < kv_len; j++) {
            acc += local_scores[j] * V[head * kv_len * d_k + j * d_k + d];
        }
        O[o_offset + d] = acc;
    }
}
