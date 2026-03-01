// Normalization kernels for Intel Arc A770 (Xe-HPG)
// Optimized with workgroup-level tree reductions in local memory.
// Workgroup size of 256 (8 × Xe-HPG 32-wide subgroups) recommended.

// RMSNorm kernel
// RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
__kernel void rmsnorm(
    __global const float* input,    // [batch, hidden_dim]
    __global const float* weight,   // [hidden_dim]
    __global float* output,         // [batch, hidden_dim]
    const int hidden_dim,
    const float eps,
    __local float* local_buf        // [workgroup_size]
) {
    const int batch_idx = get_global_id(1);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    const int row_offset = batch_idx * hidden_dim;

    // Phase 1: Compute sum of squares (parallel reduction)
    float sum_sq = 0.0f;
    for (int i = lid; i < hidden_dim; i += local_size) {
        float val = input[row_offset + i];
        sum_sq += val * val;
    }
    local_buf[lid] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) local_buf[lid] += local_buf[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float rms = rsqrt(local_buf[0] / (float)hidden_dim + eps);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Normalize and scale
    for (int i = lid; i < hidden_dim; i += local_size) {
        output[row_offset + i] = input[row_offset + i] * rms * weight[i];
    }
}

// LayerNorm kernel
// LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta
__kernel void layernorm(
    __global const float* input,    // [batch, hidden_dim]
    __global const float* gamma,    // [hidden_dim]
    __global const float* beta,     // [hidden_dim]
    __global float* output,         // [batch, hidden_dim]
    const int hidden_dim,
    const float eps,
    __local float* local_buf        // [workgroup_size * 2] for mean and var
) {
    const int batch_idx = get_global_id(1);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    const int row_offset = batch_idx * hidden_dim;

    // Phase 1: Compute mean
    float sum = 0.0f;
    for (int i = lid; i < hidden_dim; i += local_size) {
        sum += input[row_offset + i];
    }
    local_buf[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) local_buf[lid] += local_buf[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = local_buf[0] / (float)hidden_dim;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Compute variance
    float var_sum = 0.0f;
    for (int i = lid; i < hidden_dim; i += local_size) {
        float diff = input[row_offset + i] - mean;
        var_sum += diff * diff;
    }
    local_buf[lid] = var_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) local_buf[lid] += local_buf[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_std = rsqrt(local_buf[0] / (float)hidden_dim + eps);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 3: Normalize, scale, and shift
    for (int i = lid; i < hidden_dim; i += local_size) {
        output[row_offset + i] = (input[row_offset + i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
