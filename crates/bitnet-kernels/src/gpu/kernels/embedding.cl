/// Embedding lookup and output projection kernels for BitNet inference.

/// Embedding lookup: output[token_idx, dim_idx] = weight[token_id, dim_idx]
/// Out-of-vocabulary token IDs (>= vocab_size) produce a zero vector.
/// Global work: [n_tokens, embedding_dim]
__kernel void embedding_lookup(
    __global const uint* token_ids,
    __global const float* weight,
    __global float* output,
    const int embedding_dim,
    const int vocab_size
) {
    const int token_idx = get_global_id(0);
    const int dim_idx = get_global_id(1);
    if (dim_idx >= embedding_dim) return;

    const uint token_id = token_ids[token_idx];
    const int out_offset = token_idx * embedding_dim + dim_idx;

    if (token_id < (uint)vocab_size) {
        output[out_offset] = weight[token_id * embedding_dim + dim_idx];
    } else {
        output[out_offset] = 0.0f;
    }
}

/// Embedding lookup with padding index support.
/// Tokens matching padding_idx also produce a zero vector.
/// Global work: [n_tokens, embedding_dim]
__kernel void embedding_lookup_padded(
    __global const uint* token_ids,
    __global const float* weight,
    __global float* output,
    const int embedding_dim,
    const int vocab_size,
    const int padding_idx
) {
    const int token_idx = get_global_id(0);
    const int dim_idx = get_global_id(1);
    if (dim_idx >= embedding_dim) return;

    const uint token_id = token_ids[token_idx];
    const int out_offset = token_idx * embedding_dim + dim_idx;

    if (token_id >= (uint)vocab_size || (int)token_id == padding_idx) {
        output[out_offset] = 0.0f;
    } else {
        output[out_offset] = weight[token_id * embedding_dim + dim_idx];
    }
}

/// Output projection (lm_head): logits = hidden @ weight^T
/// hidden: [seq_len, hidden_size], weight: [vocab_size, hidden_size]
/// output: [seq_len, vocab_size]
/// Each work-item computes one element of the output matrix.
/// Global work: [seq_len, vocab_size]
__kernel void output_projection(
    __global const float* hidden,
    __global const float* weight,
    __global float* output,
    const int hidden_size,
    const int vocab_size
) {
    const int seq_idx = get_global_id(0);
    const int vocab_idx = get_global_id(1);
    if (vocab_idx >= vocab_size) return;

    float acc = 0.0f;
    const int h_offset = seq_idx * hidden_size;
    const int w_offset = vocab_idx * hidden_size;
    for (int k = 0; k < hidden_size; k++) {
        acc += hidden[h_offset + k] * weight[w_offset + k];
    }
    output[seq_idx * vocab_size + vocab_idx] = acc;
}

/// Embedding normalization (L2-norm per token).
/// output[i] = input[i] / (sqrt(sum(input[i]^2) / dim) + eps)
/// Global work: [n_tokens]
__kernel void embedding_rms_norm(
    __global const float* input,
    __global float* output,
    const int embedding_dim,
    const float eps
) {
    const int token_idx = get_global_id(0);
    const int offset = token_idx * embedding_dim;

    float sum_sq = 0.0f;
    for (int i = 0; i < embedding_dim; i++) {
        float val = input[offset + i];
        sum_sq += val * val;
    }

    float scale = rsqrt(sum_sq / (float)embedding_dim + eps);
    for (int i = 0; i < embedding_dim; i++) {
        output[offset + i] = input[offset + i] * scale;
    }
}

/// Add absolute position embeddings element-wise.
/// output[token_idx, dim] = input[token_idx, dim] + pos_weight[position_offset + token_idx, dim]
/// Global work: [n_tokens, embedding_dim]
__kernel void add_position_embedding(
    __global const float* input,
    __global const float* pos_weight,
    __global float* output,
    const int embedding_dim,
    const int position_offset
) {
    const int token_idx = get_global_id(0);
    const int dim_idx = get_global_id(1);
    if (dim_idx >= embedding_dim) return;

    const int in_offset = token_idx * embedding_dim + dim_idx;
    const int pos = position_offset + token_idx;
    output[in_offset] = input[in_offset] + pos_weight[pos * embedding_dim + dim_idx];
}
