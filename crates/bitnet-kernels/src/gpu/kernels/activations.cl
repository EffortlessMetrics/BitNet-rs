// Activation function kernels for Intel Arc A770
//
// Provides SiLU (Swish), GELU, ReLU activations, fused SiLU*up gate pattern,
// elementwise operations, and numerically stable softmax with tree reduction.

// SiLU (Swish) activation: x * sigmoid(x)
// Used in LLaMA FFN: gate = SiLU(W1 * x) * (W3 * x)
__kernel void silu(
    __global const float* input,
    __global float* output,
    const int n
) {
    const int gid = get_global_id(0);
    if (gid >= n) return;
    float x = input[gid];
    output[gid] = x / (1.0f + exp(-x));
}

// Fused SiLU + elementwise multiply (gate * up pattern)
__kernel void silu_mul(
    __global const float* gate,     // W1 * x
    __global const float* up,       // W3 * x
    __global float* output,
    const int n
) {
    const int gid = get_global_id(0);
    if (gid >= n) return;
    float g = gate[gid];
    float s = g / (1.0f + exp(-g));
    output[gid] = s * up[gid];
}

// GELU activation (tanh approximation)
__kernel void gelu(
    __global const float* input,
    __global float* output,
    const int n
) {
    const int gid = get_global_id(0);
    if (gid >= n) return;
    float x = input[gid];
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[gid] = x * cdf;
}

// ReLU activation
__kernel void relu(
    __global const float* input,
    __global float* output,
    const int n
) {
    const int gid = get_global_id(0);
    if (gid >= n) return;
    output[gid] = fmax(0.0f, input[gid]);
}

// Elementwise add
__kernel void elementwise_add(
    __global const float* a,
    __global const float* b,
    __global float* output,
    const int n
) {
    const int gid = get_global_id(0);
    if (gid >= n) return;
    output[gid] = a[gid] + b[gid];
}

// Elementwise multiply
__kernel void elementwise_mul(
    __global const float* a,
    __global const float* b,
    __global float* output,
    const int n
) {
    const int gid = get_global_id(0);
    if (gid >= n) return;
    output[gid] = a[gid] * b[gid];
}

// Scale (multiply by scalar)
__kernel void scale(
    __global float* data,
    const int n,
    const float scalar
) {
    const int gid = get_global_id(0);
    if (gid >= n) return;
    data[gid] *= scalar;
}

// Numerically stable softmax (full implementation)
// Each workgroup processes one row. Workgroup dim 0 = local threads, dim 1 = row index.
__kernel void softmax_full(
    __global float* data,           // [rows, cols] — modified in place
    const int cols,
    __local float* local_buf
) {
    const int row = get_global_id(1);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    const int row_offset = row * cols;

    // Phase 1: Find max (for numerical stability)
    float thread_max = -INFINITY;
    for (int i = lid; i < cols; i += local_size) {
        thread_max = fmax(thread_max, data[row_offset + i]);
    }
    local_buf[lid] = thread_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) local_buf[lid] = fmax(local_buf[lid], local_buf[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_max = local_buf[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = lid; i < cols; i += local_size) {
        float val = exp(data[row_offset + i] - row_max);
        data[row_offset + i] = val;
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

    // Phase 3: Normalize
    for (int i = lid; i < cols; i += local_size) {
        data[row_offset + i] *= inv_sum;
    }
}
