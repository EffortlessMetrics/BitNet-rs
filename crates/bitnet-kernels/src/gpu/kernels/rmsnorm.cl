/// Optimized RMS normalization kernels for Intel Arc GPUs.
///
/// Uses work-group parallel reduction for the sum-of-squares pass,
/// then a parallel normalize+scale pass.  One work-group processes
/// one row (hidden dimension), so launch with:
///   global_work_size = [rows * local_size]
///   local_work_size  = [local_size]        (e.g. 256)

/// Two-pass RMSNorm with work-group parallel reduction.
///
/// Pass 1 — parallel sum-of-squares via local memory tree reduction.
/// Pass 2 — each work-item normalizes a strided slice of the row.
///
/// Args:
///   input   — [rows × N] flattened row-major
///   weight  — [N] per-element learnable scale
///   output  — [rows × N]
///   N       — hidden dimension (number of columns)
///   eps     — numerical stability constant (e.g. 1e-5)
__kernel void rms_norm_parallel(
    __global const float* input,
    __global const float* weight,
    __global float* output,
    const uint N,
    const float eps
) {
    const uint lid  = get_local_id(0);
    const uint lsz  = get_local_size(0);
    const uint group = get_group_id(0);

    // Each work-group handles one row
    const uint row_offset = group * N;

    // --- Pass 1: parallel sum of squares via reduction ---
    __local float scratch[256];

    float partial = 0.0f;
    for (uint i = lid; i < N; i += lsz) {
        float v = input[row_offset + i];
        partial += v * v;
    }
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction
    for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Broadcast the reciprocal RMS factor
    __local float rms_factor;
    if (lid == 0) {
        rms_factor = rsqrt(scratch[0] / (float)N + eps);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float rms = rms_factor;

    // --- Pass 2: normalize and scale ---
    for (uint i = lid; i < N; i += lsz) {
        output[row_offset + i] = input[row_offset + i] * rms * weight[i];
    }
}

/// Fused RMSNorm + residual add.
///
/// out[i] = (input[i] + residual[i]) * rsqrt(mean((input+residual)^2) + eps) * weight[i]
///
/// Saves one global memory round-trip compared to separate add + norm.
__kernel void rms_norm_residual(
    __global const float* input,
    __global const float* residual,
    __global const float* weight,
    __global float* output,
    const uint N,
    const float eps
) {
    const uint lid  = get_local_id(0);
    const uint lsz  = get_local_size(0);
    const uint group = get_group_id(0);

    const uint row_offset = group * N;

    __local float scratch[256];

    // Pass 1: sum of squares of (input + residual)
    float partial = 0.0f;
    for (uint i = lid; i < N; i += lsz) {
        float v = input[row_offset + i] + residual[row_offset + i];
        partial += v * v;
    }
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    __local float rms_factor;
    if (lid == 0) {
        rms_factor = rsqrt(scratch[0] / (float)N + eps);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float rms = rms_factor;

    // Pass 2: normalize and scale
    for (uint i = lid; i < N; i += lsz) {
        float v = input[row_offset + i] + residual[row_offset + i];
        output[row_offset + i] = v * rms * weight[i];
    }
}
