/**
 * HIP rotary position embedding (RoPE) kernel for AMD GPUs.
 *
 * Applies rotation to pairs of elements (x[2i], x[2i+1]) using
 * precomputed cos/sin tables.
 */

#include <hip/hip_runtime.h>

/**
 * In-place RoPE application.
 *
 * @param x         Tensor of shape [batch, seq_len, dim]  (dim is even)
 * @param cos_table Precomputed cos values [max_seq, dim/2]
 * @param sin_table Precomputed sin values [max_seq, dim/2]
 * @param seq_len   Current sequence length
 * @param dim       Hidden dimension (must be even)
 * @param offset    Positional offset for KV-cache continuation
 */
extern "C" __global__ void rope_forward(
    float* __restrict__ x,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int seq_len, int dim, int offset
) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int half_dim = dim / 2;
    int total = seq_len * half_dim;

    if (idx >= total) return;

    int pos = idx / half_dim;
    int i   = idx % half_dim;

    int abs_pos = pos + offset;

    float cos_val = cos_table[abs_pos * half_dim + i];
    float sin_val = sin_table[abs_pos * half_dim + i];

    int base = pos * dim;
    float x0 = x[base + 2 * i];
    float x1 = x[base + 2 * i + 1];

    x[base + 2 * i]     = x0 * cos_val - x1 * sin_val;
    x[base + 2 * i + 1] = x0 * sin_val + x1 * cos_val;
}

/**
 * Build cos/sin tables on-device (avoids host→device copy).
 *
 * theta_i = 1 / (base ^ (2i / dim))
 */
extern "C" __global__ void rope_build_tables(
    float* __restrict__ cos_table,
    float* __restrict__ sin_table,
    int max_seq, int half_dim, float theta_base
) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx >= max_seq * half_dim) return;

    int pos = idx / half_dim;
    int i   = idx % half_dim;

    float freq = 1.0f / powf(theta_base,
                              2.0f * (float)i / (float)(half_dim * 2));
    float angle = (float)pos * freq;

    cos_table[idx] = cosf(angle);
    sin_table[idx] = sinf(angle);
}
