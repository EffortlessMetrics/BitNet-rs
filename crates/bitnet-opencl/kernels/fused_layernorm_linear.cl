/// Fused LayerNorm + Linear projection kernel.
///
/// Performs normalisation, optional affine transform (scale + bias),
/// and a linear projection (matmul) in a single kernel launch,
/// avoiding intermediate global-memory writes.
///
/// y = Linear(LayerNorm(x))
///   = W * ((x - mean) / sqrt(var + eps) * gamma + beta) + bias_linear
///
/// Work-group: one work-group per row (token position).
/// Each work-item in the group processes a subset of the hidden dimension
/// and participates in the parallel mean/variance reduction.

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

/// Fused LayerNorm → Linear projection.
///
/// @param input       [batch, hidden_dim]
/// @param gamma       [hidden_dim]         (LayerNorm scale)
/// @param beta        [hidden_dim]         (LayerNorm bias)
/// @param weight      [out_dim, hidden_dim] (linear weight, row-major)
/// @param bias_linear [out_dim]            (linear bias, may be NULL)
/// @param output      [batch, out_dim]
/// @param hidden_dim  input feature dimension
/// @param out_dim     output feature dimension
/// @param eps         LayerNorm epsilon (e.g. 1e-5)
/// @param fused       if 0, skip LayerNorm (pass-through to linear only)
__kernel void fused_layernorm_linear(
    __global const float* restrict input,
    __global const float* restrict gamma,
    __global const float* restrict beta,
    __global const float* restrict weight,
    __global const float* restrict bias_linear,
    __global       float* restrict output,
    const int hidden_dim,
    const int out_dim,
    const float eps,
    const int fused)
{
    const int row = get_group_id(0);
    const int lid = get_local_id(0);

    __local float smem_sum[WG_SIZE];
    __local float smem_sq[WG_SIZE];

    const int row_off = row * hidden_dim;

    // --- Step 1: compute mean and variance in shared memory ---
    float local_sum = 0.0f;
    float local_sq  = 0.0f;

    if (fused) {
        for (int i = lid; i < hidden_dim; i += WG_SIZE) {
            float val = input[row_off + i];
            local_sum += val;
            local_sq  += val * val;
        }
    }

    smem_sum[lid] = local_sum;
    smem_sq[lid]  = local_sq;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction.
    for (int stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            smem_sum[lid] += smem_sum[lid + stride];
            smem_sq[lid]  += smem_sq[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float mean = smem_sum[0] / (float)hidden_dim;
    float var  = smem_sq[0] / (float)hidden_dim - mean * mean;
    float inv_std = rsqrt(var + eps);

    // --- Step 2: for each output dimension, compute dot product ---
    // Each work-item handles a subset of out_dim columns.
    for (int o = lid; o < out_dim; o += WG_SIZE) {
        float acc = 0.0f;
        const int w_off = o * hidden_dim;

        for (int i = 0; i < hidden_dim; ++i) {
            float x;
            if (fused) {
                // Normalise + affine.
                float normed = (input[row_off + i] - mean) * inv_std;
                x = normed * gamma[i] + beta[i];
            } else {
                x = input[row_off + i];
            }
            acc += weight[w_off + i] * x;
        }

        if (bias_linear != 0) {
            acc += bias_linear[o];
        }
        output[row * out_dim + o] = acc;
    }
}
