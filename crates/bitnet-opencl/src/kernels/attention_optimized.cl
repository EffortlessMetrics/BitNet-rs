// Fused scaled dot-product attention kernel.
//
// Computes: output = softmax(Q · K^T / scale) · V
//
// Each work-group processes one query row (one head position).
// Local memory `shared` must be allocated to at least
// seq_len * sizeof(float) bytes for the attention score row.
//
// Q:      [seq_len, head_dim]  — query matrix
// K:      [seq_len, head_dim]  — key matrix
// V:      [seq_len, head_dim]  — value matrix
// output: [seq_len, head_dim]  — result

__kernel void fused_attention(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* output,
    __local float* shared,
    const uint seq_len,
    const uint head_dim,
    const float scale)
{
    const uint q_row = get_group_id(0);   // Which query position
    const uint lid   = get_local_id(0);
    const uint wg    = get_local_size(0);

    if (q_row >= seq_len) return;

    // ------ Phase 1: Q·K^T scores into local memory ------
    for (uint j = lid; j < seq_len; j += wg) {
        float dot = 0.0f;
        uint q_base = q_row * head_dim;
        uint k_base = j * head_dim;

        // Scalar accumulation (vectorised variant below).
        uint d = 0;
        for (; d + 3 < head_dim; d += 4) {
            float4 qv = vload4(0, &Q[q_base + d]);
            float4 kv = vload4(0, &K[k_base + d]);
            dot += qv.s0 * kv.s0 + qv.s1 * kv.s1
                 + qv.s2 * kv.s2 + qv.s3 * kv.s3;
        }
        for (; d < head_dim; d++) {
            dot += Q[q_base + d] * K[k_base + d];
        }

        shared[j] = dot * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ------ Phase 2: numerically stable softmax in-place ------
    // 2a. Find max.
    // Use scratch at offset seq_len for the reduction buffer.
    __local float* scratch = shared + seq_len;

    float local_max = -INFINITY;
    for (uint j = lid; j < seq_len; j += wg) {
        local_max = fmax(local_max, shared[j]);
    }
    scratch[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = wg / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_val = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2b. Exp + sum.
    float local_sum = 0.0f;
    for (uint j = lid; j < seq_len; j += wg) {
        float e = exp(shared[j] - max_val);
        shared[j] = e;
        local_sum += e;
    }
    scratch[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = wg / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_sum = 1.0f / scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2c. Normalise weights.
    for (uint j = lid; j < seq_len; j += wg) {
        shared[j] *= inv_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ------ Phase 3: weighted sum of V ------
    for (uint d = lid; d < head_dim; d += wg) {
        float acc = 0.0f;
        for (uint j = 0; j < seq_len; j++) {
            acc = fma(shared[j], V[j * head_dim + d], acc);
        }
        output[q_row * head_dim + d] = acc;
    }
}

// Causal (masked) variant – zeros out attention for positions j > q_row.
__kernel void fused_attention_causal(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* output,
    __local float* shared,
    const uint seq_len,
    const uint head_dim,
    const float scale)
{
    const uint q_row = get_group_id(0);
    const uint lid   = get_local_id(0);
    const uint wg    = get_local_size(0);

    if (q_row >= seq_len) return;

    // Phase 1: scores with causal mask.
    for (uint j = lid; j < seq_len; j += wg) {
        if (j > q_row) {
            shared[j] = -INFINITY;
        } else {
            float dot = 0.0f;
            uint q_base = q_row * head_dim;
            uint k_base = j * head_dim;
            for (uint d = 0; d < head_dim; d++) {
                dot = fma(Q[q_base + d], K[k_base + d], dot);
            }
            shared[j] = dot * scale;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: softmax (same algorithm).
    __local float* scratch = shared + seq_len;
    float local_max = -INFINITY;
    for (uint j = lid; j <= q_row && j < seq_len; j += wg) {
        local_max = fmax(local_max, shared[j]);
    }
    scratch[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = wg / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_val = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float local_sum = 0.0f;
    for (uint j = lid; j < seq_len; j += wg) {
        float e = (j <= q_row) ? exp(shared[j] - max_val) : 0.0f;
        shared[j] = e;
        local_sum += e;
    }
    scratch[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = wg / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_sum = 1.0f / scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint j = lid; j < seq_len; j += wg) {
        shared[j] *= inv_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 3: weighted V.
    for (uint d = lid; d < head_dim; d += wg) {
        float acc = 0.0f;
        for (uint j = 0; j <= q_row; j++) {
            acc = fma(shared[j], V[j * head_dim + d], acc);
        }
        output[q_row * head_dim + d] = acc;
    }
}
