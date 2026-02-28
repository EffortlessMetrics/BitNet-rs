/// INT8 matrix multiplication kernel.
/// INT8 weight x FP16 activation -> FP32 accumulate.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void matmul_int8_fp16(
    __global const half* A,
    __global const char* B,
    __global float* C,
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
        float b_val = (float)B[k * N + col];
        acc += a_val * b_val;
    }
    C[row * N + col] = acc;
}

__kernel void matmul_int8_int8(
    __global const char* A,
    __global const char* B,
    __global float* C,
    const uint M,
    const uint N,
    const uint K,
    const float scale_a,
    const float scale_b
) {
    const uint row = get_global_id(0);
    const uint col = get_global_id(1);
    if (row >= M || col >= N) return;

    int acc = 0;
    for (uint k = 0; k < K; k++) {
        acc += (int)A[row * K + k] * (int)B[k * N + col];
    }
    C[row * N + col] = (float)acc * scale_a * scale_b;
}
