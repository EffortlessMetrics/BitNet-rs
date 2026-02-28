//! HIP C++ kernel source strings for AMD GPU inference.
//!
//! Each kernel is embedded as a `&str` constant containing HIP C++ source.
//! These are compiled at runtime via `hiprtcCompileProgram` (or loaded as
//! pre-compiled `.hsaco` blobs when available).

/// HIP C++ source for tiled matrix multiplication.
pub const MATMUL_HIP: &str = r#"
extern "C" __global__ void matmul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K)
{
    const int TILE = 16;
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
"#;

/// HIP C++ source for row-wise softmax.
pub const SOFTMAX_HIP: &str = r#"
extern "C" __global__ void softmax(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int N)
{
    __shared__ float shared_max[256];
    __shared__ float shared_sum[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int row_start = row * N;

    // Phase 1: find row max
    float local_max = -3.402823e+38f;
    for (int i = tid; i < N; i += 256) {
        float val = input[row_start + i];
        if (val > local_max) local_max = val;
    }
    shared_max[tid] = local_max;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && shared_max[tid + s] > shared_max[tid])
            shared_max[tid] = shared_max[tid + s];
        __syncthreads();
    }
    float row_max = shared_max[0];

    // Phase 2: exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += 256) {
        float e = expf(input[row_start + i] - row_max);
        output[row_start + i] = e;
        local_sum += e;
    }
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) shared_sum[tid] += shared_sum[tid + s];
        __syncthreads();
    }
    float total = shared_sum[0];

    // Phase 3: normalize
    for (int i = tid; i < N; i += 256) {
        output[row_start + i] /= total;
    }
}
"#;

/// HIP C++ source for RMS normalization.
pub const RMSNORM_HIP: &str = r#"
extern "C" __global__ void rmsnorm(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    unsigned int N, float eps)
{
    __shared__ float shared_sq[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int row_start = row * N;

    // Phase 1: sum of squares
    float local_sq = 0.0f;
    for (int i = tid; i < N; i += 256) {
        float val = input[row_start + i];
        local_sq += val * val;
    }
    shared_sq[tid] = local_sq;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) shared_sq[tid] += shared_sq[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(shared_sq[0] / (float)N + eps);

    // Phase 2: normalize and scale
    for (int i = tid; i < N; i += 256) {
        output[row_start + i] = (input[row_start + i] / rms) * weight[i];
    }
}
"#;

/// HIP C++ source for scaled dot-product attention.
pub const ATTENTION_HIP: &str = r#"
extern "C" __global__ void attention(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    unsigned int seq_len, unsigned int head_dim, unsigned int kv_len)
{
    __shared__ float shared_max[256];
    __shared__ float shared_sum[256];

    int query_pos = blockIdx.x;
    int tid = threadIdx.x;
    float scale = rsqrtf((float)head_dim);

    // Phase 1: compute attention scores and find max
    float local_max = -3.402823e+38f;
    for (int kv = tid; kv < kv_len; kv += 256) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += Q[query_pos * head_dim + d] * K[kv * head_dim + d];
        }
        float score = dot * scale;
        if (score > local_max) local_max = score;
    }
    shared_max[tid] = local_max;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && shared_max[tid + s] > shared_max[tid])
            shared_max[tid] = shared_max[tid + s];
        __syncthreads();
    }
    float row_max = shared_max[0];

    // Phase 2: softmax denominator
    float local_sum = 0.0f;
    for (int kv = tid; kv < kv_len; kv += 256) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += Q[query_pos * head_dim + d] * K[kv * head_dim + d];
        }
        local_sum += expf(dot * scale - row_max);
    }
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) shared_sum[tid] += shared_sum[tid + s];
        __syncthreads();
    }
    float total = shared_sum[0];

    // Phase 3: weighted sum of values
    for (int d = tid; d < head_dim; d += 256) {
        float acc = 0.0f;
        for (int kv = 0; kv < kv_len; kv++) {
            float dot = 0.0f;
            for (int dd = 0; dd < head_dim; dd++) {
                dot += Q[query_pos * head_dim + dd] * K[kv * head_dim + dd];
            }
            float w = expf(dot * scale - row_max) / total;
            acc += w * V[kv * head_dim + d];
        }
        output[query_pos * head_dim + d] = acc;
    }
}
"#;

/// All available HIP kernel source entries.
pub const ALL_KERNELS: &[(&str, &str)] = &[
    ("matmul", MATMUL_HIP),
    ("softmax", SOFTMAX_HIP),
    ("rmsnorm", RMSNORM_HIP),
    ("attention", ATTENTION_HIP),
];

/// Look up a kernel source by name.
pub fn get_kernel_source(name: &str) -> Option<&'static str> {
    ALL_KERNELS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, src)| *src)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_source_is_valid_hip() {
        assert!(MATMUL_HIP.contains("__global__"));
        assert!(MATMUL_HIP.contains("__shared__"));
        assert!(MATMUL_HIP.contains("__syncthreads()"));
        assert!(MATMUL_HIP.contains("extern \"C\""));
    }

    #[test]
    fn softmax_source_has_reduction() {
        assert!(SOFTMAX_HIP.contains("__shared__ float shared_max"));
        assert!(SOFTMAX_HIP.contains("__shared__ float shared_sum"));
        assert!(SOFTMAX_HIP.contains("expf("));
    }

    #[test]
    fn rmsnorm_source_has_eps() {
        assert!(RMSNORM_HIP.contains("float eps"));
        assert!(RMSNORM_HIP.contains("sqrtf("));
        assert!(RMSNORM_HIP.contains("weight[i]"));
    }

    #[test]
    fn attention_source_has_qkv() {
        assert!(ATTENTION_HIP.contains("const float* __restrict__ Q"));
        assert!(ATTENTION_HIP.contains("const float* __restrict__ K"));
        assert!(ATTENTION_HIP.contains("const float* __restrict__ V"));
        assert!(ATTENTION_HIP.contains("rsqrtf("));
    }

    #[test]
    fn all_kernels_list_has_four_entries() {
        assert_eq!(ALL_KERNELS.len(), 4);
    }

    #[test]
    fn get_kernel_source_finds_existing() {
        assert!(get_kernel_source("matmul").is_some());
        assert!(get_kernel_source("softmax").is_some());
        assert!(get_kernel_source("rmsnorm").is_some());
        assert!(get_kernel_source("attention").is_some());
    }

    #[test]
    fn get_kernel_source_returns_none_for_missing() {
        assert!(get_kernel_source("nonexistent").is_none());
    }

    #[test]
    fn all_sources_are_non_empty() {
        for (name, src) in ALL_KERNELS {
            assert!(!src.is_empty(), "kernel '{name}' has empty source");
        }
    }

    #[test]
    fn all_kernels_use_extern_c() {
        for (name, src) in ALL_KERNELS {
            assert!(
                src.contains("extern \"C\""),
                "kernel '{name}' must use extern \"C\" linkage"
            );
        }
    }

    #[test]
    fn all_kernels_use_global_qualifier() {
        for (name, src) in ALL_KERNELS {
            assert!(
                src.contains("__global__"),
                "kernel '{name}' must use __global__ qualifier"
            );
        }
    }
}
