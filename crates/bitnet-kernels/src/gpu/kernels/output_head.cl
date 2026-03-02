// Output projection head kernels for Intel Arc A770 (Xe-HPG)
//
// Kernels:
// - output_head_projection: hidden @ weight^T with optional bias
// - output_head_projection_tiled: tiled variant for large vocabularies
// - partial_vocab_topk: select top-K logit candidates
// - logit_normalize: subtract max for numerical stability

#define TILE_SIZE 16

// Full output projection: logits = hidden @ weight^T + bias
// hidden: [seq_len, hidden_dim], weight: [vocab_size, hidden_dim]
// output: [seq_len, vocab_size]
__kernel void output_head_projection(
    __global const float* hidden,
    __global const float* weight,
    __global const float* bias,
    __global float* output,
    const int seq_len,
    const int hidden_dim,
    const int vocab_size,
    const int use_bias
) {
    const int s = get_global_id(1);
    const int v = get_global_id(0);

    if (s >= seq_len || v >= vocab_size) return;

    float acc = 0.0f;
    for (int k = 0; k < hidden_dim; k++) {
        acc += hidden[s * hidden_dim + k] * weight[v * hidden_dim + k];
    }
    if (use_bias) {
        acc += bias[v];
    }
    output[s * vocab_size + v] = acc;
}

// Tiled output projection using local memory for large vocabularies
__kernel void output_head_projection_tiled(
    __global const float* hidden,
    __global const float* weight,
    __global float* output,
    const int seq_len,
    const int hidden_dim,
    const int vocab_size,
    __local float* tile_h,
    __local float* tile_w
) {
    const int row = get_global_id(1);  // seq index
    const int col = get_global_id(0);  // vocab index
    const int lr  = get_local_id(1);
    const int lc  = get_local_id(0);

    float acc = 0.0f;
    const int num_tiles = (hidden_dim + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int h_col = t * TILE_SIZE + lc;
        if (row < seq_len && h_col < hidden_dim)
            tile_h[lr * TILE_SIZE + lc] = hidden[row * hidden_dim + h_col];
        else
            tile_h[lr * TILE_SIZE + lc] = 0.0f;

        int w_col = t * TILE_SIZE + lr;
        if (col < vocab_size && w_col < hidden_dim)
            tile_w[lr * TILE_SIZE + lc] = weight[col * hidden_dim + w_col];
        else
            tile_w[lr * TILE_SIZE + lc] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            acc += tile_h[lr * TILE_SIZE + k] * tile_w[k * TILE_SIZE + lc];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < seq_len && col < vocab_size) {
        output[row * vocab_size + col] = acc;
    }
}

// Subtract per-row max for numerical stability
// logits: [seq_len, vocab_size], modified in-place
__kernel void logit_normalize(
    __global float* logits,
    const int vocab_size
) {
    const int s = get_global_id(0);
    const int offset = s * vocab_size;

    float max_val = logits[offset];
    for (int v = 1; v < vocab_size; v++) {
        float val = logits[offset + v];
        if (val > max_val) max_val = val;
    }
    for (int v = 0; v < vocab_size; v++) {
        logits[offset + v] -= max_val;
    }
}

// Select top-K indices from logits for partial vocab decode
// logits: [vocab_size], indices_out: [k], values_out: [k]
__kernel void partial_vocab_topk(
    __global const float* logits,
    __global int* indices_out,
    __global float* values_out,
    const int vocab_size,
    const int k
) {
    // Single work-item kernel for simplicity; K is small
    for (int i = 0; i < k; i++) {
        float best = -1e30f;
        int best_idx = 0;
        for (int v = 0; v < vocab_size; v++) {
            float val = logits[v];
            // Skip already-selected indices
            int skip = 0;
            for (int j = 0; j < i; j++) {
                if (indices_out[j] == v) { skip = 1; break; }
            }
            if (!skip && val > best) {
                best = val;
                best_idx = v;
            }
        }
        indices_out[i] = best_idx;
        values_out[i] = best;
    }
}
