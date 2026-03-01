// Scaled dot-product attention kernels for Intel Arc A770
//
// Three kernels implementing multi-head attention:
// - attention_scores: QK^T / sqrt(d_k) with optional causal masking
// - attention_softmax: numerically stable row-wise softmax with tree reduction
// - attention_weighted_sum: attention_weights × V

// Compute attention scores: QK^T / sqrt(d_k) with optional causal mask.
// Grid: (seq_len, seq_len, num_heads)
__kernel void attention_scores(
    __global const float* Q,        // [num_heads, seq_len, head_dim]
    __global const float* K,        // [num_heads, seq_len, head_dim]
    __global float* scores,         // [num_heads, seq_len, seq_len]
    const int seq_len,
    const int head_dim,
    const float inv_sqrt_dk,        // 1.0 / sqrt(head_dim)
    const int causal                // 1 = apply causal mask, 0 = no mask
) {
    const int q_pos = get_global_id(0);
    const int k_pos = get_global_id(1);
    const int head  = get_global_id(2);

    if (q_pos >= seq_len || k_pos >= seq_len) return;

    // Causal mask: future positions get -inf
    if (causal != 0 && k_pos > q_pos) {
        scores[head * seq_len * seq_len + q_pos * seq_len + k_pos] = -1e9f;
        return;
    }

    const int q_offset = head * seq_len * head_dim + q_pos * head_dim;
    const int k_offset = head * seq_len * head_dim + k_pos * head_dim;

    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        dot += Q[q_offset + d] * K[k_offset + d];
    }

    scores[head * seq_len * seq_len + q_pos * seq_len + k_pos] = dot * inv_sqrt_dk;
}

// Row-wise softmax with tree reduction using local memory.
// Each workgroup processes one row. Launch: global(local_size * rows), local(local_size).
__kernel void attention_softmax(
    __global float* scores,         // [rows, cols] — modified in place
    const int cols,
    __local float* local_buf
) {
    const int row = get_global_id(0) / get_local_size(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    const int row_offset = row * cols;

    // Phase 1: find row max for numerical stability
    float thread_max = -INFINITY;
    for (int i = lid; i < cols; i += local_size) {
        thread_max = fmax(thread_max, scores[row_offset + i]);
    }
    local_buf[lid] = thread_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) local_buf[lid] = fmax(local_buf[lid], local_buf[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_max = local_buf[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: compute exp(x - max) and partial sums
    float thread_sum = 0.0f;
    for (int i = lid; i < cols; i += local_size) {
        float val = exp(scores[row_offset + i] - row_max);
        scores[row_offset + i] = val;
        thread_sum += val;
    }
    local_buf[lid] = thread_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) local_buf[lid] += local_buf[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_sum = 1.0f / (local_buf[0] + 1e-8f);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 3: normalize
    for (int i = lid; i < cols; i += local_size) {
        scores[row_offset + i] *= inv_sum;
    }
}

// Compute attention output: weights × V.
// Grid: (seq_len, head_dim, num_heads)
__kernel void attention_weighted_sum(
    __global const float* weights,  // [num_heads, seq_len, seq_len]
    __global const float* V,        // [num_heads, seq_len, head_dim]
    __global float* output,         // [num_heads, seq_len, head_dim]
    const int seq_len,
    const int head_dim
) {
    const int q_pos = get_global_id(0);
    const int d     = get_global_id(1);
    const int head  = get_global_id(2);

    if (q_pos >= seq_len || d >= head_dim) return;

    const int w_offset = head * seq_len * seq_len + q_pos * seq_len;
    const int v_offset = head * seq_len * head_dim;

    float sum = 0.0f;
    for (int k = 0; k < seq_len; k++) {
        sum += weights[w_offset + k] * V[v_offset + k * head_dim + d];
    }

    output[head * seq_len * head_dim + q_pos * head_dim + d] = sum;
}
