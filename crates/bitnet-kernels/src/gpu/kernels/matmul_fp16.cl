/// FP16 matrix multiplication kernel for Intel Arc GPUs.
///
/// Uses cl_khr_fp16 extension for half-precision computation.
/// Accumulates in FP32, stores FP16 intermediates for 2x throughput.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void matmul_i2s_fp16(
    __global const half* A,
    __global const uchar* B,
    __global float* C,
    const uint M,
    const uint N,
    const uint K
) {
    const uint row = get_global_id(0);
    const uint col = get_global_id(1);
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    const uint k_packed = K / 4;

    for (uint kb = 0; kb < k_packed; kb++) {
        uchar packed = B[kb * N + col];
        uint base_k = kb * 4;
        for (uint sub = 0; sub < 4; sub++) {
            int bits = (packed >> (sub * 2)) & 0x3;
            float w;
            if (bits == 0x1) w = 1.0f;
            else if (bits == 0x3) w = -1.0f;
            else w = 0.0f;
            float a_val = vload_half(0, &A[row * K + base_k + sub]);
            acc += a_val * w;
        }
    }
    C[row * N + col] = acc;
}

__kernel void matmul_fp16_full(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const uint M,
    const uint N,
    const uint K
) {
    const uint row = get_global_id(0);
    const uint col = get_global_id(1);
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        float a_val = vload_half(0, &A[row * K + k]);
        float b_val = vload_half(0, &B[k * N + col]);
        acc += a_val * b_val;
    }
    vstore_half(acc, 0, &C[row * N + col]);
}
