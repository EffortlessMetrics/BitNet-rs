//! OpenCL kernel source management and compilation.
//!
//! Manages OpenCL .cl kernel source code as embedded strings,
//! providing compilation, caching, and build-option management.

use std::collections::HashMap;
use std::fmt;

/// Identifies an OpenCL kernel program.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum KernelProgramId {
    Matmul,
    MatmulTiled,
    MatmulBatched,
    Softmax,
    SoftmaxTemperature,
    LayerNorm,
    RmsNorm,
    RoPE,
    Elementwise,
    Quantized,
    Attention,
    Custom(String),
}

impl fmt::Display for KernelProgramId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matmul => write!(f, "matmul"),
            Self::MatmulTiled => write!(f, "matmul_tiled"),
            Self::MatmulBatched => write!(f, "matmul_batched"),
            Self::Softmax => write!(f, "softmax"),
            Self::SoftmaxTemperature => write!(f, "softmax_temperature"),
            Self::LayerNorm => write!(f, "layer_norm"),
            Self::RmsNorm => write!(f, "rms_norm"),
            Self::RoPE => write!(f, "rope"),
            Self::Elementwise => write!(f, "elementwise"),
            Self::Quantized => write!(f, "quantized"),
            Self::Attention => write!(f, "attention"),
            Self::Custom(name) => write!(f, "custom:{name}"),
        }
    }
}

/// OpenCL kernel source code with metadata.
#[derive(Debug, Clone)]
pub struct KernelSource {
    /// Program identifier
    pub id: KernelProgramId,
    /// OpenCL C source code
    pub source: &'static str,
    /// Kernel function names within the source
    pub entry_points: Vec<String>,
    /// Required OpenCL version
    pub required_version: &'static str,
    /// Build options (e.g., "-DTILE_SIZE=16")
    pub build_options: Vec<String>,
    /// Estimated local memory usage per work group (bytes)
    pub local_mem_estimate: usize,
    /// Preferred work group size
    pub preferred_work_group: usize,
}

/// Registry of all available OpenCL kernel sources.
pub struct KernelSourceRegistry {
    sources: HashMap<KernelProgramId, KernelSource>,
}

impl KernelSourceRegistry {
    /// Create a new registry with all built-in kernels.
    pub fn new() -> Self {
        let mut sources = HashMap::new();

        sources.insert(
            KernelProgramId::Matmul,
            KernelSource {
                id: KernelProgramId::Matmul,
                source: MATMUL_CL_SOURCE,
                entry_points: vec!["matmul_naive".to_string()],
                required_version: "1.2",
                build_options: vec![],
                local_mem_estimate: 0,
                preferred_work_group: 64,
            },
        );

        sources.insert(
            KernelProgramId::MatmulTiled,
            KernelSource {
                id: KernelProgramId::MatmulTiled,
                source: MATMUL_TILED_CL_SOURCE,
                entry_points: vec!["matmul_tiled".to_string()],
                required_version: "1.2",
                build_options: vec!["-DTILE_SIZE=16".to_string()],
                local_mem_estimate: 16 * 16 * 4 * 2, // 2 tiles of float
                preferred_work_group: 256,
            },
        );

        sources.insert(
            KernelProgramId::MatmulBatched,
            KernelSource {
                id: KernelProgramId::MatmulBatched,
                source: MATMUL_BATCHED_CL_SOURCE,
                entry_points: vec!["matmul_batched".to_string()],
                required_version: "1.2",
                build_options: vec![],
                local_mem_estimate: 0,
                preferred_work_group: 64,
            },
        );

        sources.insert(
            KernelProgramId::Softmax,
            KernelSource {
                id: KernelProgramId::Softmax,
                source: SOFTMAX_CL_SOURCE,
                entry_points: vec!["softmax_row".to_string()],
                required_version: "1.2",
                build_options: vec![],
                local_mem_estimate: 256 * 4, // reduction buffer
                preferred_work_group: 256,
            },
        );

        sources.insert(
            KernelProgramId::SoftmaxTemperature,
            KernelSource {
                id: KernelProgramId::SoftmaxTemperature,
                source: SOFTMAX_TEMP_CL_SOURCE,
                entry_points: vec!["softmax_temperature".to_string()],
                required_version: "1.2",
                build_options: vec![],
                local_mem_estimate: 256 * 4,
                preferred_work_group: 256,
            },
        );

        sources.insert(
            KernelProgramId::LayerNorm,
            KernelSource {
                id: KernelProgramId::LayerNorm,
                source: LAYER_NORM_CL_SOURCE,
                entry_points: vec!["layer_norm".to_string()],
                required_version: "1.2",
                build_options: vec![],
                local_mem_estimate: 256 * 4 * 2, // mean + var reduction
                preferred_work_group: 256,
            },
        );

        sources.insert(
            KernelProgramId::RmsNorm,
            KernelSource {
                id: KernelProgramId::RmsNorm,
                source: RMS_NORM_CL_SOURCE,
                entry_points: vec!["rms_norm".to_string()],
                required_version: "1.2",
                build_options: vec![],
                local_mem_estimate: 256 * 4,
                preferred_work_group: 256,
            },
        );

        sources.insert(
            KernelProgramId::RoPE,
            KernelSource {
                id: KernelProgramId::RoPE,
                source: ROPE_CL_SOURCE,
                entry_points: vec!["rope_forward".to_string()],
                required_version: "1.2",
                build_options: vec![],
                local_mem_estimate: 0,
                preferred_work_group: 64,
            },
        );

        sources.insert(
            KernelProgramId::Elementwise,
            KernelSource {
                id: KernelProgramId::Elementwise,
                source: ELEMENTWISE_CL_SOURCE,
                entry_points: vec![
                    "add".to_string(),
                    "mul".to_string(),
                    "silu".to_string(),
                    "gelu".to_string(),
                ],
                required_version: "1.2",
                build_options: vec![],
                local_mem_estimate: 0,
                preferred_work_group: 256,
            },
        );

        sources.insert(
            KernelProgramId::Quantized,
            KernelSource {
                id: KernelProgramId::Quantized,
                source: QUANTIZED_CL_SOURCE,
                entry_points: vec!["dequantize_i2s".to_string(), "quantized_matvec".to_string()],
                required_version: "1.2",
                build_options: vec![],
                local_mem_estimate: 0,
                preferred_work_group: 128,
            },
        );

        Self { sources }
    }

    /// Get a kernel source by ID.
    pub fn get(&self, id: &KernelProgramId) -> Option<&KernelSource> {
        self.sources.get(id)
    }

    /// Get all registered kernel IDs.
    pub fn kernel_ids(&self) -> Vec<&KernelProgramId> {
        self.sources.keys().collect()
    }

    /// Get total number of registered kernels.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Whether registry is empty.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Register a custom kernel source.
    pub fn register(&mut self, source: KernelSource) {
        self.sources.insert(source.id.clone(), source);
    }

    /// Get all entry point names across all kernels.
    pub fn all_entry_points(&self) -> Vec<String> {
        self.sources.values().flat_map(|s| s.entry_points.clone()).collect()
    }

    /// Get build options string for a kernel.
    pub fn build_options_string(&self, id: &KernelProgramId) -> String {
        self.sources.get(id).map(|s| s.build_options.join(" ")).unwrap_or_default()
    }
}

impl Default for KernelSourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Embedded kernel sources as static strings.
// In a full implementation, these would be include_str!("opencl/matmul.cl")
// etc. For now, we inline minimal valid OpenCL C.

const MATMUL_CL_SOURCE: &str = r#"
__kernel void matmul_naive(
    __global const float* A, __global const float* B,
    __global float* C,
    const int M, const int N, const int K,
    const float alpha, const float beta)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}
"#;

const MATMUL_TILED_CL_SOURCE: &str = r#"
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

__kernel void matmul_tiled(
    __global const float* A, __global const float* B,
    __global float* C,
    const int M, const int N, const int K)
{
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];
    int bx = get_group_id(1), by = get_group_id(0);
    int tx = get_local_id(1), ty = get_local_id(0);
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int k_idx = t * TILE_SIZE;
        As[ty][tx] = (row < M && k_idx + tx < K)
            ? A[row * K + k_idx + tx] : 0.0f;
        Bs[ty][tx] = (k_idx + ty < K && col < N)
            ? B[(k_idx + ty) * N + col] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[ty][k] * Bs[k][tx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
"#;

const MATMUL_BATCHED_CL_SOURCE: &str = r#"
__kernel void matmul_batched(
    __global const float* A, __global const float* B,
    __global float* C,
    const int M, const int N, const int K,
    const int batch_count)
{
    int batch = get_global_id(2);
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (batch < batch_count && row < M && col < N) {
        int stride_a = M * K;
        int stride_b = K * N;
        int stride_c = M * N;
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[batch * stride_a + row * K + k]
                 * B[batch * stride_b + k * N + col];
        }
        C[batch * stride_c + row * N + col] = sum;
    }
}
"#;

const SOFTMAX_CL_SOURCE: &str = r#"
__kernel void softmax_row(
    __global const float* input, __global float* output,
    const int rows, const int cols)
{
    int row = get_global_id(0);
    if (row >= rows) return;
    int base = row * cols;
    float max_val = input[base];
    for (int j = 1; j < cols; j++)
        max_val = fmax(max_val, input[base + j]);
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        float e = exp(input[base + j] - max_val);
        output[base + j] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (int j = 0; j < cols; j++) output[base + j] *= inv;
}
"#;

const SOFTMAX_TEMP_CL_SOURCE: &str = r#"
__kernel void softmax_temperature(
    __global const float* input, __global float* output,
    const int rows, const int cols, const float temperature)
{
    int row = get_global_id(0);
    if (row >= rows) return;
    int base = row * cols;
    float inv_temp = 1.0f / temperature;
    float max_val = input[base] * inv_temp;
    for (int j = 1; j < cols; j++)
        max_val = fmax(max_val, input[base + j] * inv_temp);
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        float e = exp(input[base + j] * inv_temp - max_val);
        output[base + j] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (int j = 0; j < cols; j++) output[base + j] *= inv;
}
"#;

const LAYER_NORM_CL_SOURCE: &str = r#"
__kernel void layer_norm(
    __global const float* input, __global const float* gamma,
    __global const float* beta, __global float* output,
    const int rows, const int hidden_size, const float eps)
{
    int row = get_global_id(0);
    if (row >= rows) return;
    int base = row * hidden_size;
    float mean = 0.0f;
    for (int i = 0; i < hidden_size; i++)
        mean += input[base + i];
    mean /= (float)hidden_size;
    float var = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        float d = input[base + i] - mean;
        var += d * d;
    }
    var /= (float)hidden_size;
    float inv_std = 1.0f / sqrt(var + eps);
    for (int i = 0; i < hidden_size; i++) {
        output[base + i] =
            (input[base + i] - mean) * inv_std * gamma[i]
            + beta[i];
    }
}
"#;

const RMS_NORM_CL_SOURCE: &str = r#"
__kernel void rms_norm(
    __global const float* input, __global const float* gamma,
    __global float* output,
    const int rows, const int hidden_size, const float eps)
{
    int row = get_global_id(0);
    if (row >= rows) return;
    int base = row * hidden_size;
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_size; i++)
        sum_sq += input[base + i] * input[base + i];
    float rms = sqrt(sum_sq / (float)hidden_size + eps);
    float inv = 1.0f / rms;
    for (int i = 0; i < hidden_size; i++) {
        output[base + i] = input[base + i] * inv * gamma[i];
    }
}
"#;

const ROPE_CL_SOURCE: &str = r#"
__kernel void rope_forward(
    __global float* x,
    const int seq_len, const int head_dim,
    const float theta_base)
{
    int pos = get_global_id(0);
    int d = get_global_id(1);
    if (pos >= seq_len || d >= head_dim / 2) return;
    float freq = 1.0f / pow(theta_base,
        2.0f * (float)d / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);
    int idx0 = pos * head_dim + 2 * d;
    int idx1 = idx0 + 1;
    float x0 = x[idx0], x1 = x[idx1];
    x[idx0] = x0 * cos_val - x1 * sin_val;
    x[idx1] = x0 * sin_val + x1 * cos_val;
}
"#;

const ELEMENTWISE_CL_SOURCE: &str = r#"
__kernel void add(
    __global const float* a, __global const float* b,
    __global float* c, const int n)
{
    int i = get_global_id(0);
    if (i < n) c[i] = a[i] + b[i];
}

__kernel void mul(
    __global const float* a, __global const float* b,
    __global float* c, const int n)
{
    int i = get_global_id(0);
    if (i < n) c[i] = a[i] * b[i];
}

__kernel void silu(
    __global const float* x, __global float* y,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) y[i] = x[i] / (1.0f + exp(-x[i]));
}

__kernel void gelu(
    __global const float* x, __global float* y,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        float v = x[i];
        float cdf = 0.5f * (1.0f + tanh(
            0.7978845608f * (v + 0.044715f * v * v * v)));
        y[i] = v * cdf;
    }
}
"#;

const QUANTIZED_CL_SOURCE: &str = r#"
__kernel void dequantize_i2s(
    __global const uchar* packed,
    __global const float* scales,
    __global float* output,
    const int num_elements, const int block_size)
{
    int i = get_global_id(0);
    if (i >= num_elements) return;
    int byte_idx = i / 4;
    int bit_pos = (i % 4) * 2;
    int val = (packed[byte_idx] >> bit_pos) & 0x03;
    float fval = (float)(val - 1);
    int block_idx = i / block_size;
    output[i] = fval * scales[block_idx];
}

__kernel void quantized_matvec(
    __global const uchar* packed_w,
    __global const float* scales,
    __global const float* x,
    __global float* y,
    const int rows, const int cols, const int block_size)
{
    int row = get_global_id(0);
    if (row >= rows) return;
    float sum = 0.0f;
    for (int col = 0; col < cols; col++) {
        int idx = row * cols + col;
        int byte_idx = idx / 4;
        int bit_pos = (idx % 4) * 2;
        int val = (packed_w[byte_idx] >> bit_pos) & 0x03;
        float w = (float)(val - 1)
            * scales[idx / block_size];
        sum += w * x[col];
    }
    y[row] = sum;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    // ── Registry construction ──────────────────────────────────────

    #[test]
    fn test_registry_has_10_builtin_kernels() {
        let reg = KernelSourceRegistry::new();
        assert_eq!(reg.len(), 10);
    }

    #[test]
    fn test_registry_is_not_empty() {
        let reg = KernelSourceRegistry::new();
        assert!(!reg.is_empty());
    }

    #[test]
    fn test_empty_registry_after_clear() {
        let mut reg = KernelSourceRegistry::new();
        reg.sources.clear();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_default_equals_new() {
        let a = KernelSourceRegistry::new();
        let b = KernelSourceRegistry::default();
        assert_eq!(a.len(), b.len());
        for id in a.kernel_ids() {
            assert!(b.get(id).is_some());
        }
    }

    // ── Lookup: get() returns correct source ───────────────────────

    #[test]
    fn test_get_matmul() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Matmul).unwrap();
        assert_eq!(src.id, KernelProgramId::Matmul);
    }

    #[test]
    fn test_get_matmul_tiled() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::MatmulTiled).unwrap();
        assert_eq!(src.id, KernelProgramId::MatmulTiled);
    }

    #[test]
    fn test_get_matmul_batched() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::MatmulBatched).unwrap();
        assert_eq!(src.id, KernelProgramId::MatmulBatched);
    }

    #[test]
    fn test_get_softmax() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Softmax).unwrap();
        assert_eq!(src.id, KernelProgramId::Softmax);
    }

    #[test]
    fn test_get_softmax_temperature() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::SoftmaxTemperature).unwrap();
        assert_eq!(src.id, KernelProgramId::SoftmaxTemperature);
    }

    #[test]
    fn test_get_layer_norm() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::LayerNorm).unwrap();
        assert_eq!(src.id, KernelProgramId::LayerNorm);
    }

    #[test]
    fn test_get_rms_norm() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::RmsNorm).unwrap();
        assert_eq!(src.id, KernelProgramId::RmsNorm);
    }

    #[test]
    fn test_get_rope() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::RoPE).unwrap();
        assert_eq!(src.id, KernelProgramId::RoPE);
    }

    #[test]
    fn test_get_elementwise() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Elementwise).unwrap();
        assert_eq!(src.id, KernelProgramId::Elementwise);
    }

    #[test]
    fn test_get_quantized() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Quantized).unwrap();
        assert_eq!(src.id, KernelProgramId::Quantized);
    }

    // ── Missing: get() returns None ────────────────────────────────

    #[test]
    fn test_get_missing_custom_returns_none() {
        let reg = KernelSourceRegistry::new();
        assert!(reg.get(&KernelProgramId::Custom("missing".into())).is_none());
    }

    #[test]
    fn test_get_attention_returns_none() {
        let reg = KernelSourceRegistry::new();
        assert!(reg.get(&KernelProgramId::Attention).is_none());
    }

    // ── Entry points ───────────────────────────────────────────────

    #[test]
    fn test_matmul_entry_point() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Matmul).unwrap();
        assert_eq!(src.entry_points, vec!["matmul_naive"]);
    }

    #[test]
    fn test_elementwise_has_four_entry_points() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Elementwise).unwrap();
        assert_eq!(src.entry_points.len(), 4);
        assert!(src.entry_points.contains(&"add".to_string()));
        assert!(src.entry_points.contains(&"mul".to_string()));
        assert!(src.entry_points.contains(&"silu".to_string()));
        assert!(src.entry_points.contains(&"gelu".to_string()));
    }

    #[test]
    fn test_quantized_has_two_entry_points() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Quantized).unwrap();
        assert_eq!(src.entry_points.len(), 2);
        assert!(src.entry_points.contains(&"dequantize_i2s".to_string()));
        assert!(src.entry_points.contains(&"quantized_matvec".to_string()));
    }

    #[test]
    fn test_all_entry_points_count() {
        let reg = KernelSourceRegistry::new();
        let eps = reg.all_entry_points();
        // 1+1+1+1+1+1+1+1+4+2 = 14
        assert_eq!(eps.len(), 14);
    }

    #[test]
    fn test_all_entry_points_contains_matmul_naive() {
        let reg = KernelSourceRegistry::new();
        let eps = reg.all_entry_points();
        assert!(eps.contains(&"matmul_naive".to_string()));
    }

    #[test]
    fn test_all_entry_points_contains_gelu() {
        let reg = KernelSourceRegistry::new();
        let eps = reg.all_entry_points();
        assert!(eps.contains(&"gelu".to_string()));
    }

    #[test]
    fn test_all_entry_points_contains_rope_forward() {
        let reg = KernelSourceRegistry::new();
        let eps = reg.all_entry_points();
        assert!(eps.contains(&"rope_forward".to_string()));
    }

    // ── Build options ──────────────────────────────────────────────

    #[test]
    fn test_matmul_tiled_build_options() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::MatmulTiled).unwrap();
        assert_eq!(src.build_options, vec!["-DTILE_SIZE=16"]);
    }

    #[test]
    fn test_build_options_string_tiled() {
        let reg = KernelSourceRegistry::new();
        let opts = reg.build_options_string(&KernelProgramId::MatmulTiled);
        assert_eq!(opts, "-DTILE_SIZE=16");
    }

    #[test]
    fn test_build_options_string_empty_for_matmul() {
        let reg = KernelSourceRegistry::new();
        let opts = reg.build_options_string(&KernelProgramId::Matmul);
        assert_eq!(opts, "");
    }

    #[test]
    fn test_build_options_string_missing_kernel() {
        let reg = KernelSourceRegistry::new();
        let opts = reg.build_options_string(&KernelProgramId::Attention);
        assert_eq!(opts, "");
    }

    // ── Custom registration ────────────────────────────────────────

    #[test]
    fn test_register_custom_kernel() {
        let mut reg = KernelSourceRegistry::new();
        let custom_id = KernelProgramId::Custom("my_kernel".into());
        reg.register(KernelSource {
            id: custom_id.clone(),
            source: "__kernel void my_fn() {}",
            entry_points: vec!["my_fn".to_string()],
            required_version: "2.0",
            build_options: vec![],
            local_mem_estimate: 0,
            preferred_work_group: 64,
        });
        assert_eq!(reg.len(), 11);
        let src = reg.get(&custom_id).unwrap();
        assert_eq!(src.entry_points, vec!["my_fn"]);
    }

    #[test]
    fn test_register_custom_overwrites() {
        let mut reg = KernelSourceRegistry::new();
        let custom_id = KernelProgramId::Custom("test".into());
        reg.register(KernelSource {
            id: custom_id.clone(),
            source: "__kernel void v1() {}",
            entry_points: vec!["v1".to_string()],
            required_version: "1.2",
            build_options: vec![],
            local_mem_estimate: 0,
            preferred_work_group: 64,
        });
        reg.register(KernelSource {
            id: custom_id.clone(),
            source: "__kernel void v2() {}",
            entry_points: vec!["v2".to_string()],
            required_version: "1.2",
            build_options: vec![],
            local_mem_estimate: 0,
            preferred_work_group: 64,
        });
        assert_eq!(reg.len(), 11);
        let src = reg.get(&custom_id).unwrap();
        assert_eq!(src.entry_points, vec!["v2"]);
    }

    #[test]
    fn test_register_attention_kernel() {
        let mut reg = KernelSourceRegistry::new();
        reg.register(KernelSource {
            id: KernelProgramId::Attention,
            source: "__kernel void attention() {}",
            entry_points: vec!["attention".to_string()],
            required_version: "1.2",
            build_options: vec![],
            local_mem_estimate: 0,
            preferred_work_group: 256,
        });
        assert_eq!(reg.len(), 11);
        assert!(reg.get(&KernelProgramId::Attention).is_some());
    }

    // ── Display trait ──────────────────────────────────────────────

    #[test]
    fn test_display_matmul() {
        assert_eq!(KernelProgramId::Matmul.to_string(), "matmul");
    }

    #[test]
    fn test_display_matmul_tiled() {
        assert_eq!(KernelProgramId::MatmulTiled.to_string(), "matmul_tiled");
    }

    #[test]
    fn test_display_matmul_batched() {
        assert_eq!(KernelProgramId::MatmulBatched.to_string(), "matmul_batched");
    }

    #[test]
    fn test_display_softmax() {
        assert_eq!(KernelProgramId::Softmax.to_string(), "softmax");
    }

    #[test]
    fn test_display_softmax_temperature() {
        assert_eq!(KernelProgramId::SoftmaxTemperature.to_string(), "softmax_temperature");
    }

    #[test]
    fn test_display_layer_norm() {
        assert_eq!(KernelProgramId::LayerNorm.to_string(), "layer_norm");
    }

    #[test]
    fn test_display_rms_norm() {
        assert_eq!(KernelProgramId::RmsNorm.to_string(), "rms_norm");
    }

    #[test]
    fn test_display_rope() {
        assert_eq!(KernelProgramId::RoPE.to_string(), "rope");
    }

    #[test]
    fn test_display_elementwise() {
        assert_eq!(KernelProgramId::Elementwise.to_string(), "elementwise");
    }

    #[test]
    fn test_display_quantized() {
        assert_eq!(KernelProgramId::Quantized.to_string(), "quantized");
    }

    #[test]
    fn test_display_attention() {
        assert_eq!(KernelProgramId::Attention.to_string(), "attention");
    }

    #[test]
    fn test_display_custom() {
        assert_eq!(KernelProgramId::Custom("foo".into()).to_string(), "custom:foo");
    }

    // ── Source validity ────────────────────────────────────────────

    #[test]
    fn test_matmul_source_contains_function() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Matmul).unwrap();
        assert!(!src.source.is_empty());
        assert!(src.source.contains("matmul_naive"));
    }

    #[test]
    fn test_matmul_tiled_source_contains_function() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::MatmulTiled).unwrap();
        assert!(!src.source.is_empty());
        assert!(src.source.contains("matmul_tiled"));
    }

    #[test]
    fn test_matmul_batched_source_contains_function() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::MatmulBatched).unwrap();
        assert!(!src.source.is_empty());
        assert!(src.source.contains("matmul_batched"));
    }

    #[test]
    fn test_softmax_source_contains_function() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Softmax).unwrap();
        assert!(!src.source.is_empty());
        assert!(src.source.contains("softmax_row"));
    }

    #[test]
    fn test_softmax_temp_source_contains_function() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::SoftmaxTemperature).unwrap();
        assert!(!src.source.is_empty());
        assert!(src.source.contains("softmax_temperature"));
    }

    #[test]
    fn test_layer_norm_source_contains_function() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::LayerNorm).unwrap();
        assert!(!src.source.is_empty());
        assert!(src.source.contains("layer_norm"));
    }

    #[test]
    fn test_rms_norm_source_contains_function() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::RmsNorm).unwrap();
        assert!(!src.source.is_empty());
        assert!(src.source.contains("rms_norm"));
    }

    #[test]
    fn test_rope_source_contains_function() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::RoPE).unwrap();
        assert!(!src.source.is_empty());
        assert!(src.source.contains("rope_forward"));
    }

    #[test]
    fn test_elementwise_source_contains_all_functions() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Elementwise).unwrap();
        assert!(!src.source.is_empty());
        for name in &["add", "mul", "silu", "gelu"] {
            assert!(src.source.contains(name), "missing function: {name}");
        }
    }

    #[test]
    fn test_quantized_source_contains_all_functions() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Quantized).unwrap();
        assert!(!src.source.is_empty());
        assert!(src.source.contains("dequantize_i2s"));
        assert!(src.source.contains("quantized_matvec"));
    }

    #[test]
    fn test_all_sources_contain_kernel_keyword() {
        let reg = KernelSourceRegistry::new();
        for id in reg.kernel_ids() {
            let src = reg.get(id).unwrap();
            assert!(src.source.contains("__kernel"), "{id} source missing __kernel keyword");
        }
    }

    // ── Metadata correctness ───────────────────────────────────────

    #[test]
    fn test_all_kernels_require_opencl_1_2() {
        let reg = KernelSourceRegistry::new();
        for id in reg.kernel_ids() {
            let src = reg.get(id).unwrap();
            assert_eq!(src.required_version, "1.2", "{id} should require OpenCL 1.2");
        }
    }

    #[test]
    fn test_matmul_tiled_local_mem_estimate() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::MatmulTiled).unwrap();
        assert_eq!(src.local_mem_estimate, 16 * 16 * 4 * 2);
    }

    #[test]
    fn test_matmul_tiled_preferred_work_group() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::MatmulTiled).unwrap();
        assert_eq!(src.preferred_work_group, 256);
    }

    #[test]
    fn test_quantized_preferred_work_group() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Quantized).unwrap();
        assert_eq!(src.preferred_work_group, 128);
    }

    #[test]
    fn test_rope_preferred_work_group() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::RoPE).unwrap();
        assert_eq!(src.preferred_work_group, 64);
    }

    // ── Clone / Eq / Hash on KernelProgramId ───────────────────────

    #[test]
    fn test_kernel_program_id_clone() {
        let id = KernelProgramId::Matmul;
        let cloned = id.clone();
        assert_eq!(id, cloned);
    }

    #[test]
    fn test_kernel_program_id_eq() {
        assert_eq!(KernelProgramId::Softmax, KernelProgramId::Softmax);
        assert_ne!(KernelProgramId::Softmax, KernelProgramId::RmsNorm);
    }

    #[test]
    fn test_kernel_program_id_custom_eq() {
        let a = KernelProgramId::Custom("x".into());
        let b = KernelProgramId::Custom("x".into());
        let c = KernelProgramId::Custom("y".into());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_kernel_program_id_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(KernelProgramId::Matmul);
        set.insert(KernelProgramId::Matmul);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_kernel_program_id_debug() {
        let dbg = format!("{:?}", KernelProgramId::RoPE);
        assert_eq!(dbg, "RoPE");
    }

    #[test]
    fn test_kernel_source_debug() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Matmul).unwrap();
        let dbg = format!("{:?}", src);
        assert!(dbg.contains("Matmul"));
    }

    #[test]
    fn test_kernel_source_clone() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Matmul).unwrap();
        let cloned = src.clone();
        assert_eq!(cloned.id, src.id);
        assert_eq!(cloned.source, src.source);
    }
}
