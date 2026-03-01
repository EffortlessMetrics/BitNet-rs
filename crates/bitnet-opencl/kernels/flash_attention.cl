/// Flash Attention OpenCL kernel — memory-efficient tiled attention.
///
/// Processes attention in BLOCK_SIZE tiles so intermediate scores stay in
/// local memory, achieving O(N) global-memory usage instead of O(N²).
///
/// Uses the online softmax trick (Milakov & Gimelshein, 2018):
///   m_new  = max(m_old, rowmax(S_block))
///   l_new  = l_old * exp(m_old - m_new) + rowsum(exp(S_block - m_new))

// BLOCK_SIZE is passed as a compile-time define (-DBLOCK_SIZE=64).
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

/// Tiled flash-attention forward pass.
///
/// Grid: (num_heads, ceil(seq_len_q / BLOCK_SIZE))
/// Work-group: (BLOCK_SIZE,)
///
/// @param Q        [num_heads, seq_len_q, head_dim]
/// @param K        [num_heads, seq_len_kv, head_dim]
/// @param V        [num_heads, seq_len_kv, head_dim]
/// @param O        [num_heads, seq_len_q, head_dim]  (output)
/// @param scale    1/sqrt(head_dim)
/// @param seq_len_q   query sequence length
/// @param seq_len_kv  key/value sequence length
/// @param head_dim    dimension per head
__kernel void flash_attention_forward(
    __global const float* restrict Q,
    __global const float* restrict K,
    __global const float* restrict V,
    __global       float* restrict O,
    const float scale,
    const int seq_len_q,
    const int seq_len_kv,
    const int head_dim)
{
    const int head_idx   = get_group_id(0);
    const int block_row  = get_group_id(1);
    const int local_id   = get_local_id(0);

    const int q_row = block_row * BLOCK_SIZE + local_id;
    if (q_row >= seq_len_q) return;

    const int head_offset_q  = head_idx * seq_len_q  * head_dim;
    const int head_offset_kv = head_idx * seq_len_kv * head_dim;

    // Per-row online-softmax state.
    float m_i = -INFINITY;
    float l_i = 0.0f;

    const int num_kv_blocks = (seq_len_kv + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int kv_start = kv_block * BLOCK_SIZE;
        const int kv_end   = min(kv_start + BLOCK_SIZE, seq_len_kv);

        float m_ij = -INFINITY;

        // First pass: compute raw scores and local max.
        for (int kv = kv_start; kv < kv_end; ++kv) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += Q[head_offset_q + q_row * head_dim + d]
                       * K[head_offset_kv + kv * head_dim + d];
            }
            score *= scale;
            if (score > m_ij) m_ij = score;
        }

        // Online softmax update.
        float m_new = fmax(m_i, m_ij);
        float alpha = exp(m_i - m_new);
        float l_ij  = 0.0f;

        // Second pass: accumulate weighted V.
        for (int kv = kv_start; kv < kv_end; ++kv) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += Q[head_offset_q + q_row * head_dim + d]
                       * K[head_offset_kv + kv * head_dim + d];
            }
            score *= scale;
            float p = exp(score - m_new);
            l_ij += p;

            for (int d = 0; d < head_dim; ++d) {
                float o_old = O[head_offset_q + q_row * head_dim + d];
                float v_val = V[head_offset_kv + kv * head_dim + d];
                O[head_offset_q + q_row * head_dim + d] =
                    o_old * alpha + p * v_val;
            }
        }

        l_i = l_i * alpha + l_ij;
        m_i = m_new;
    }

    // Final normalisation: O[q_row] /= l_i
    if (l_i > 0.0f) {
        float inv_l = 1.0f / l_i;
        for (int d = 0; d < head_dim; ++d) {
            O[head_offset_q + q_row * head_dim + d] *= inv_l;
        }
    }
}

/// Causal (masked) variant — identical algorithm but ignores kv > q_row.
__kernel void flash_attention_causal(
    __global const float* restrict Q,
    __global const float* restrict K,
    __global const float* restrict V,
    __global       float* restrict O,
    const float scale,
    const int seq_len_q,
    const int seq_len_kv,
    const int head_dim)
{
    const int head_idx  = get_group_id(0);
    const int block_row = get_group_id(1);
    const int local_id  = get_local_id(0);

    const int q_row = block_row * BLOCK_SIZE + local_id;
    if (q_row >= seq_len_q) return;

    const int head_offset_q  = head_idx * seq_len_q  * head_dim;
    const int head_offset_kv = head_idx * seq_len_kv * head_dim;

    float m_i = -INFINITY;
    float l_i = 0.0f;

    // Causal mask: only attend to kv <= q_row.
    const int causal_limit = min(q_row + 1, seq_len_kv);
    const int num_kv_blocks = (causal_limit + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int kv_start = kv_block * BLOCK_SIZE;
        const int kv_end   = min(kv_start + BLOCK_SIZE, causal_limit);

        float m_ij = -INFINITY;
        for (int kv = kv_start; kv < kv_end; ++kv) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += Q[head_offset_q + q_row * head_dim + d]
                       * K[head_offset_kv + kv * head_dim + d];
            }
            score *= scale;
            if (score > m_ij) m_ij = score;
        }

        float m_new = fmax(m_i, m_ij);
        float alpha = exp(m_i - m_new);
        float l_ij  = 0.0f;

        for (int kv = kv_start; kv < kv_end; ++kv) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += Q[head_offset_q + q_row * head_dim + d]
                       * K[head_offset_kv + kv * head_dim + d];
            }
            score *= scale;
            float p = exp(score - m_new);
            l_ij += p;

            for (int d = 0; d < head_dim; ++d) {
                float o_old = O[head_offset_q + q_row * head_dim + d];
                float v_val = V[head_offset_kv + kv * head_dim + d];
                O[head_offset_q + q_row * head_dim + d] =
                    o_old * alpha + p * v_val;
            }
        }

        l_i = l_i * alpha + l_ij;
        m_i = m_new;
    }

    if (l_i > 0.0f) {
        float inv_l = 1.0f / l_i;
        for (int d = 0; d < head_dim; ++d) {
            O[head_offset_q + q_row * head_dim + d] *= inv_l;
        }
    }
}
