// Scaled dot-product attention kernel for Intel Arc GPUs
// Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

// Kernel 1: Compute attention scores (QK^T / sqrt(d_k))
__kernel void attention_scores(
    __global const float* Q,      // [batch, heads, seq_q, d_k]
    __global const float* K,      // [batch, heads, seq_k, d_k]
    __global float* scores,       // [batch, heads, seq_q, seq_k]
    const int seq_q,
    const int seq_k,
    const int d_k,
    const float inv_sqrt_dk,
    const int causal                // 1 = causal masking, 0 = no mask
) {
    const int gid_q = get_global_id(0);   // query position
    const int gid_k = get_global_id(1);   // key position
    const int head  = get_global_id(2);   // batch*heads linear index

    if (gid_q >= seq_q || gid_k >= seq_k) return;

    // Causal mask: future positions get -inf
    if (causal && gid_k > gid_q) {
        scores[head * seq_q * seq_k + gid_q * seq_k + gid_k] = -1e9f;
        return;
    }

    float dot = 0.0f;
    const int q_offset = head * seq_q * d_k + gid_q * d_k;
    const int k_offset = head * seq_k * d_k + gid_k * d_k;

    // Vectorized dot product (float4)
    int i = 0;
    for (; i + 3 < d_k; i += 4) {
        float4 q4 = vload4(0, Q + q_offset + i);
        float4 k4 = vload4(0, K + k_offset + i);
        dot += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
    }
    for (; i < d_k; i++) {
        dot += Q[q_offset + i] * K[k_offset + i];
    }

    scores[head * seq_q * seq_k + gid_q * seq_k + gid_k] = dot * inv_sqrt_dk;
}

// Kernel 2: Row-wise softmax of attention scores
__kernel void attention_softmax(
    __global float* scores,       // [batch*heads, seq_q, seq_k] — modified in place
    const int seq_k,
    __local float* local_buf      // [workgroup_size]
) {
    const int row = get_global_id(1);     // which query row
    const int head = get_global_id(2);    // batch*heads index
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    const int row_offset = head * get_global_size(1) * seq_k + row * seq_k;

    // Phase 1: Find max (for numerical stability)
    float thread_max = -1e9f;
    for (int i = lid; i < seq_k; i += local_size) {
        thread_max = fmax(thread_max, scores[row_offset + i]);
    }
    local_buf[lid] = thread_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction for max
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            local_buf[lid] = fmax(local_buf[lid], local_buf[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_max = local_buf[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = lid; i < seq_k; i += local_size) {
        float val = exp(scores[row_offset + i] - row_max);
        scores[row_offset + i] = val;
        thread_sum += val;
    }
    local_buf[lid] = thread_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction for sum
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            local_buf[lid] += local_buf[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_sum = local_buf[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 3: Normalize
    float inv_sum = 1.0f / (row_sum + 1e-8f);
    for (int i = lid; i < seq_k; i += local_size) {
        scores[row_offset + i] *= inv_sum;
    }
}

// Kernel 3: Weighted sum (attention_weights * V)
__kernel void attention_weighted_sum(
    __global const float* weights,  // [batch*heads, seq_q, seq_k]
    __global const float* V,        // [batch*heads, seq_k, d_v]
    __global float* output,         // [batch*heads, seq_q, d_v]
    const int seq_q,
    const int seq_k,
    const int d_v
) {
    const int gid_q = get_global_id(0);   // query position
    const int gid_d = get_global_id(1);   // value dimension
    const int head  = get_global_id(2);   // batch*heads

    if (gid_q >= seq_q || gid_d >= d_v) return;

    float sum = 0.0f;
    const int w_offset = head * seq_q * seq_k + gid_q * seq_k;
    const int v_offset = head * seq_k * d_v;

    for (int k = 0; k < seq_k; k++) {
        sum += weights[w_offset + k] * V[v_offset + k * d_v + gid_d];
    }

    output[head * seq_q * d_v + gid_q * d_v + gid_d] = sum;
}
