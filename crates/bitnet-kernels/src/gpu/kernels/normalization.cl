// Normalization kernels for Intel Arc A770
//
// RMSNorm and LayerNorm with tree reductions optimized for Xe-HPG.
// Each workgroup processes one row/token of dimension `dim`.

// RMSNorm: output = x * rsqrt(mean(x²) + eps) * weight
// Each workgroup handles one row. Launch: global(local_size * rows), local(local_size).
__kernel void rmsnorm(
    __global const float* input,    // [rows, dim]
    __global const float* weight,   // [dim]
    __global float* output,         // [rows, dim]
    const int dim,
    const float eps,
    __local float* local_buf
) {
    const int row = get_global_id(0) / get_local_size(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    const int row_offset = row * dim;

    // Phase 1: compute partial sum of squares
    float partial_ss = 0.0f;
    for (int i = lid; i < dim; i += local_size) {
        float x = input[row_offset + i];
        partial_ss += x * x;
    }
    local_buf[lid] = partial_ss;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction for sum of squares
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) local_buf[lid] += local_buf[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float rms_scale = rsqrt(local_buf[0] / (float)dim + eps);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: normalize and scale
    for (int i = lid; i < dim; i += local_size) {
        output[row_offset + i] = input[row_offset + i] * rms_scale * weight[i];
    }
}

// LayerNorm: output = (x - mean) / sqrt(var + eps) * gamma + beta
// Each workgroup handles one row. Launch: global(local_size * rows), local(local_size).
__kernel void layernorm(
    __global const float* input,    // [rows, dim]
    __global const float* gamma,    // [dim]
    __global const float* beta,     // [dim]
    __global float* output,         // [rows, dim]
    const int dim,
    const float eps,
    __local float* local_buf
) {
    const int row = get_global_id(0) / get_local_size(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    const int row_offset = row * dim;

    // Phase 1: compute partial sum for mean
    float partial_sum = 0.0f;
    for (int i = lid; i < dim; i += local_size) {
        partial_sum += input[row_offset + i];
    }
    local_buf[lid] = partial_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) local_buf[lid] += local_buf[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = local_buf[0] / (float)dim;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: compute partial sum of (x - mean)² for variance
    float partial_var = 0.0f;
    for (int i = lid; i < dim; i += local_size) {
        float diff = input[row_offset + i] - mean;
        partial_var += diff * diff;
    }
    local_buf[lid] = partial_var;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) local_buf[lid] += local_buf[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_std = rsqrt(local_buf[0] / (float)dim + eps);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 3: normalize, scale, and shift
    for (int i = lid; i < dim; i += local_size) {
        output[row_offset + i] = (input[row_offset + i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
