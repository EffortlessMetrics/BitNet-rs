/**
 * HIP scaled dot-product attention kernel for AMD GPUs.
 *
 * Implements: Attention(Q,K,V) = softmax(Q·K^T / sqrt(d_k)) · V
 *
 * Single-head, row-parallel: each block computes one query row.
 * For production multi-head attention the host dispatches per-head slices.
 */

#include <hip/hip_runtime.h>

#define WARP_SIZE 64

__device__ float attn_warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

__device__ float attn_warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

/**
 * Scaled dot-product attention.
 *
 * @param Q      Query  [seq_q,  d_k]
 * @param K      Key    [seq_kv, d_k]
 * @param V      Value  [seq_kv, d_v]
 * @param output Result [seq_q,  d_v]
 * @param seq_q  Number of query positions
 * @param seq_kv Number of key/value positions
 * @param d_k    Key dimension
 * @param d_v    Value dimension
 * @param scale  Typically 1/sqrt(d_k)
 */
extern "C" __global__ void scaled_dot_product_attention(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int seq_q, int seq_kv, int d_k, int d_v,
    float scale
) {
    int q_row = hipBlockIdx_x;
    if (q_row >= seq_q) return;

    int tid = hipThreadIdx_x;

    // -- Scores: Q[q_row] · K^T  (length = seq_kv) --
    extern __shared__ float smem[];          // dynamically sized
    float* scores = smem;                    // [seq_kv]

    for (int kv = tid; kv < seq_kv; kv += hipBlockDim_x) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; ++d) {
            dot += Q[q_row * d_k + d] * K[kv * d_k + d];
        }
        scores[kv] = dot * scale;
    }
    __syncthreads();

    // -- Numerically-stable softmax over scores --
    float local_max = -1e30f;
    for (int kv = tid; kv < seq_kv; kv += hipBlockDim_x) {
        local_max = fmaxf(local_max, scores[kv]);
    }

    __shared__ float smax[64];
    int lane = tid % WARP_SIZE;
    int wid  = tid / WARP_SIZE;

    float warp_max = attn_warp_reduce_max(local_max);
    if (lane == 0) smax[wid] = warp_max;
    __syncthreads();

    if (tid < WARP_SIZE) {
        float v = (tid < (hipBlockDim_x + WARP_SIZE - 1) / WARP_SIZE)
            ? smax[tid] : -1e30f;
        warp_max = attn_warp_reduce_max(v);
        if (tid == 0) smax[0] = warp_max;
    }
    __syncthreads();
    float row_max = smax[0];

    float local_sum = 0.0f;
    for (int kv = tid; kv < seq_kv; kv += hipBlockDim_x) {
        float e = expf(scores[kv] - row_max);
        scores[kv] = e;
        local_sum += e;
    }

    __shared__ float ssum[64];
    float warp_sum = attn_warp_reduce_sum(local_sum);
    if (lane == 0) ssum[wid] = warp_sum;
    __syncthreads();

    if (tid < WARP_SIZE) {
        float v = (tid < (hipBlockDim_x + WARP_SIZE - 1) / WARP_SIZE)
            ? ssum[tid] : 0.0f;
        warp_sum = attn_warp_reduce_sum(v);
        if (tid == 0) ssum[0] = warp_sum;
    }
    __syncthreads();

    float inv_sum = 1.0f / (ssum[0] + 1e-8f);
    for (int kv = tid; kv < seq_kv; kv += hipBlockDim_x) {
        scores[kv] *= inv_sum;
    }
    __syncthreads();

    // -- Weighted sum: output[q_row] = scores · V --
    for (int d = tid; d < d_v; d += hipBlockDim_x) {
        float acc = 0.0f;
        for (int kv = 0; kv < seq_kv; ++kv) {
            acc += scores[kv] * V[kv * d_v + d];
        }
        output[q_row * d_v + d] = acc;
    }
}
