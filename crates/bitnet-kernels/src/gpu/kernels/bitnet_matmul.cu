extern "C" __global__ void bitnet_matmul_i2s(
    const signed char* __restrict__ A,
    const unsigned char* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
        signed char a = A[row * K + i];
        unsigned char b = B[i * N + col];
        acc += static_cast<float>(a) * static_cast<float>(b);
    }

    C[row * N + col] = acc;
}
