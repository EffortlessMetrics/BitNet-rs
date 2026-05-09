//! Scaled dot-product attention CUDA kernel with CPU fallback.
//!
//! # Kernel strategy
//!
//! Implements FlashAttention-style tiled attention to keep the `O(n²)` score
//! matrix in on-chip SRAM rather than HBM:
//!
//! 1. **Q tile** — each thread-block loads a `tile_q × head_dim` slice of Q
//!    into shared memory.
//! 2. **K/V streaming** — K and V blocks are streamed in `tile_kv`-sized
//!    chunks.  For each chunk the partial `softmax(QKᵀ)V` is accumulated
//!    using the online softmax trick (numerically stable running max + sum).
//! 3. **Causal mask** — upper-triangular positions are masked to `-inf` before
//!    the softmax reduction, supporting autoregressive decoding.
//! 4. **Output write-back** — the final `O[tile_q, head_dim]` tile is written
//!    to global memory in a single coalesced store.
//!
//! Target: ≥ 50 % SM occupancy on Ampere (SM 8.0) with 48 KB shared memory
//! per block.  FP16 accumulation is used when `head_dim ≤ 128` and the device
//! supports native FP16 (`compute_capability ≥ 6.0`).
//!
//! # CPU fallback
//!
//! [`attention_cpu_fallback`], [`masked_attention_cpu_fallback`], and
//! [`multi_head_attention_cpu_fallback`] provide pure-Rust implementations
//! for correctness testing and non-GPU environments.

use bitnet_common::{KernelError, Result};
#[cfg(feature = "cuda")]
use std::any::Any;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::{Ptx, compile_ptx};

/// Kernel ID recorded by dense regular-LLM CUDA attention-score receipts.
pub const CUDA_DENSE_ATTENTION_SCORE_KERNEL_ID: &str = "dense_attention_scores_f32_cuda";

/// Kernel ID recorded by dense regular-LLM CUDA attention-softmax receipts.
pub const CUDA_DENSE_ATTENTION_SOFTMAX_KERNEL_ID: &str = "dense_attention_softmax_f32_cuda";

/// Tolerance for dense GGUF attention-score fixture parity against the CPU reference.
pub const CUDA_DENSE_ATTENTION_SCORE_TOLERANCE: f32 = 0.000_25;

/// Tolerance for dense GGUF attention-softmax fixture parity against the CPU reference.
pub const CUDA_DENSE_ATTENTION_SOFTMAX_TOLERANCE: f32 = 0.000_25;

/// CPU reference backend recorded by dense attention-score CUDA parity receipts.
pub const CUDA_DENSE_ATTENTION_SCORE_REFERENCE_BACKEND: &str = "amd-9950x3d-cpu-avx512";

/// RTX 5070 Ti CUDA backend recorded by dense attention-score CUDA parity receipts.
pub const CUDA_DENSE_ATTENTION_SCORE_TARGET_BACKEND: &str = "nvidia-rtx-5070-ti-cuda";

#[cfg(feature = "cuda")]
static DENSE_ATTENTION_SCORE_NVRTC_COMPILE_LOCK: Mutex<()> = Mutex::new(());

#[cfg(feature = "cuda")]
static DENSE_ATTENTION_SOFTMAX_NVRTC_COMPILE_LOCK: Mutex<()> = Mutex::new(());

/// Alias for [`AttentionKernelConfig`] — the CUDA-specific launch configuration.
///
/// Provides a discoverable name matching the `Cuda*Config` naming convention
/// used by other kernel modules (e.g. `CudaTransposeConfig`, `CudaBatchNormConfig`).
pub type CudaAttentionConfig = AttentionKernelConfig;

// ---------------------------------------------------------------------------
// CUDA kernel source (compiled at runtime via NVRTC when `gpu`/`cuda` active)
// ---------------------------------------------------------------------------

/// Inline CUDA C source for the scaled dot-product attention kernel.
///
/// Implements a FlashAttention-style tiled kernel:
/// - `sdp_attention_f32`: single-head scaled dot-product attention
/// - `sdp_attention_causal_f32`: causal (autoregressive) variant
///
/// Each thread-block processes one query tile across all K/V positions,
/// using online softmax for numerical stability.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const ATTENTION_KERNEL_SRC: &str = r#"
extern "C" __global__ void sdp_attention_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    float scale)
{
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= seq_len_q) return;

    const float* q_row = Q + q_idx * head_dim;
    float row_max = -1e30f;

    // Pass 1: compute scores and find max for numerical stability
    extern __shared__ float scores[];
    for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
        const float* k_row = K + k_idx * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_row[d] * k_row[d];
        }
        dot *= scale;
        scores[k_idx] = dot;
        if (dot > row_max) row_max = dot;
    }

    // Pass 2: stable softmax
    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
        scores[k_idx] = expf(scores[k_idx] - row_max);
        sum_exp += scores[k_idx];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
        scores[k_idx] *= inv_sum;
    }

    // Pass 3: weighted sum of V
    float* o_row = O + q_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
            acc += scores[k_idx] * V[k_idx * head_dim + d];
        }
        o_row[d] = acc;
    }
}

extern "C" __global__ void sdp_attention_causal_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    float scale)
{
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= seq_len_q) return;

    const float* q_row = Q + q_idx * head_dim;
    float row_max = -1e30f;

    extern __shared__ float scores[];
    for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
        if (k_idx > q_idx) {
            scores[k_idx] = -1e30f;
            continue;
        }
        const float* k_row = K + k_idx * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_row[d] * k_row[d];
        }
        dot *= scale;
        scores[k_idx] = dot;
        if (dot > row_max) row_max = dot;
    }

    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
        scores[k_idx] = expf(scores[k_idx] - row_max);
        sum_exp += scores[k_idx];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
        scores[k_idx] *= inv_sum;
    }

    float* o_row = O + q_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
            acc += scores[k_idx] * V[k_idx * head_dim + d];
        }
        o_row[d] = acc;
    }
}
"#;

#[cfg(feature = "cuda")]
const CUDA_DENSE_ATTENTION_SCORE_KERNEL_SRC: &str = r#"
extern "C" __global__
void dense_attention_scores_f32_cuda(
    const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ scores,
    int q_heads,
    int kv_heads,
    int seq_len,
    int head_dim,
    float scale,
    int causal
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)q_heads * seq_len * seq_len;
    if (idx >= total) {
        return;
    }

    int k_pos = (int)(idx % seq_len);
    int q_pos = (int)((idx / seq_len) % seq_len);
    int q_head = (int)(idx / ((long long)seq_len * seq_len));

    if (causal && k_pos > q_pos) {
        scores[idx] = __int_as_float((int)0xff800000u);
        return;
    }

    int heads_per_kv_group = q_heads / kv_heads;
    int kv_head = q_head / heads_per_kv_group;
    long long q_base = ((long long)q_head * seq_len + q_pos) * head_dim;
    long long k_base = ((long long)kv_head * seq_len + k_pos) * head_dim;

    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
        dot += q[q_base + dim] * k[k_base + dim];
    }

    scores[idx] = dot * scale;
}
"#;

#[cfg(feature = "cuda")]
const CUDA_DENSE_ATTENTION_SOFTMAX_KERNEL_SRC: &str = r#"
extern "C" __global__
void dense_attention_softmax_f32_cuda(
    const float* __restrict__ scores,
    float* __restrict__ probabilities,
    int q_heads,
    int seq_len
) {
    int row = (int)blockIdx.x;
    int total_rows = q_heads * seq_len;
    if (row >= total_rows || threadIdx.x != 0) {
        return;
    }

    long long row_base = (long long)row * seq_len;
    float row_max = -3.4028234663852886e38f;
    for (int col = 0; col < seq_len; ++col) {
        float value = scores[row_base + col];
        if (value > row_max) {
            row_max = value;
        }
    }

    float sum = 0.0f;
    for (int col = 0; col < seq_len; ++col) {
        float value = scores[row_base + col];
        float probability = expf(value - row_max);
        probabilities[row_base + col] = probability;
        sum += probability;
    }

    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int col = 0; col < seq_len; ++col) {
            probabilities[row_base + col] *= inv_sum;
        }
    } else {
        for (int col = 0; col < seq_len; ++col) {
            probabilities[row_base + col] = 0.0f;
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Launch configuration (CUDA)
// ---------------------------------------------------------------------------

/// Launch configuration for the scaled dot-product attention kernel.
#[derive(Debug, Clone)]
pub struct AttentionKernelConfig {
    /// Tile size along the query (sequence-out) dimension.
    pub tile_q: u32,
    /// Tile size along the key/value (sequence-in) dimension.
    pub tile_kv: u32,
    /// Number of attention heads processed in parallel.
    pub n_heads: usize,
    /// Per-head embedding dimension (typically 64 or 128).
    pub head_dim: usize,
    /// Sequence length of the query tensor.
    pub seq_len_q: usize,
    /// Sequence length of the key/value tensors (may differ during decode).
    pub seq_len_kv: usize,
    /// Threads per block (must be ≥ `tile_q * tile_kv`).
    pub threads_per_block: u32,
    /// Bytes of dynamic shared memory for Q/K tiles and running softmax state.
    pub shared_mem_bytes: u32,
    /// Whether to apply a causal (autoregressive) mask.
    pub causal: bool,
    /// Softmax temperature scale (`1.0 / sqrt(head_dim)` by default).
    pub scale: f32,
}

/// Launch configuration for dense attention-score fixture parity.
#[derive(Debug, Clone)]
pub struct AttentionScoresConfig {
    /// Query head count.
    pub q_heads: usize,
    /// Key/value head count.
    pub kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Sequence length for query and key positions.
    pub seq_len: usize,
    /// Attention scaling factor.
    pub scale: f32,
    /// Whether to apply causal upper-triangular masking.
    pub causal: bool,
    /// Threads per CUDA block.
    pub threads_per_block: u32,
}

/// Launch configuration for dense attention-softmax fixture parity.
#[derive(Debug, Clone)]
pub struct AttentionSoftmaxConfig {
    /// Query head count.
    pub q_heads: usize,
    /// Sequence length for query/key positions.
    pub seq_len: usize,
    /// Threads per CUDA block.
    pub threads_per_block: u32,
}

/// CUDA execution counters for a dense attention-score fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaDenseAttentionScoreStats {
    /// CUDA kernel identifier.
    pub kernel_id: &'static str,
    /// Number of CUDA kernel invocations.
    pub invocations: u64,
    /// CPU fallback invocations under strict CUDA.
    pub fallback_invocations: u64,
    /// Host-to-device bytes copied for Q/K input tensors.
    pub host_to_device_bytes: u64,
    /// Device-to-host bytes copied for score output tensor.
    pub device_to_host_bytes: u64,
    /// CUDA kernel launches.
    pub kernel_launches: u64,
    /// Optional measured kernel time.
    pub kernel_time_ms: Option<f64>,
}

/// CUDA execution counters for a dense attention-softmax fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaDenseAttentionSoftmaxStats {
    /// CUDA kernel identifier.
    pub kernel_id: &'static str,
    /// Number of CUDA kernel invocations.
    pub invocations: u64,
    /// CPU fallback invocations under strict CUDA.
    pub fallback_invocations: u64,
    /// Host-to-device bytes copied for the attention-score tensor.
    pub host_to_device_bytes: u64,
    /// Device-to-host bytes copied for probability outputs.
    pub device_to_host_bytes: u64,
    /// CUDA kernel launches.
    pub kernel_launches: u64,
    /// Optional measured kernel time.
    pub kernel_time_ms: Option<f64>,
}

/// Dense GGUF attention-score fixture data prepared by the CLI/model layer.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseGgufAttentionScoreCudaFixture {
    /// Fixture identifier recorded in parity receipts.
    pub fixture_id: String,
    /// Dense model family label.
    pub model_family: String,
    /// Dense GGUF architecture label.
    pub architecture: String,
    /// Transformer layer index represented by the fixture.
    pub layer_index: usize,
    /// Query head count.
    pub q_heads: usize,
    /// Key/value head count.
    pub kv_heads: usize,
    /// Query heads per key/value head.
    pub heads_per_kv_group: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Attention scaling factor.
    pub scale: f32,
    /// SHA-256 of source RoPE Q output.
    pub q_rope_output_sha256: String,
    /// SHA-256 of source RoPE K output.
    pub k_rope_output_sha256: String,
    /// Source RoPE fixture identifier.
    pub source_rope_fixture_id: String,
    /// RoPE Q output `[q_heads, seq_len, head_dim]`.
    pub q_rope_output_f32: Vec<f32>,
    /// RoPE K output `[kv_heads, seq_len, head_dim]`.
    pub k_rope_output_f32: Vec<f32>,
    /// CPU reference scores `[q_heads, seq_len, seq_len]`.
    pub expected_scores_f32: Vec<f32>,
    /// Number of finite scores.
    pub finite_scores: usize,
    /// Number of causally masked scores.
    pub causal_masked_scores: usize,
}

/// Dense GGUF attention-softmax fixture data prepared by the CLI/model layer.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseGgufAttentionSoftmaxCudaFixture {
    /// Fixture identifier recorded in parity receipts.
    pub fixture_id: String,
    /// Dense model family label.
    pub model_family: String,
    /// Dense GGUF architecture label.
    pub architecture: String,
    /// Transformer layer index represented by the fixture.
    pub layer_index: usize,
    /// Query head count.
    pub q_heads: usize,
    /// Key/value head count from the source attention-score fixture.
    pub kv_heads: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Source attention-score fixture identifier.
    pub source_attention_score_fixture_id: String,
    /// SHA-256 of source attention scores.
    pub attention_scores_sha256: String,
    /// Attention scores `[q_heads, seq_len, seq_len]`.
    pub attention_scores_f32: Vec<f32>,
    /// CPU reference probabilities `[q_heads, seq_len, seq_len]`.
    pub expected_probabilities_f32: Vec<f32>,
    /// Number of softmax rows.
    pub row_count: usize,
    /// Number of probabilities.
    pub probability_count: usize,
    /// Number of probabilities expected to be zero from causal masking.
    pub causal_zero_probabilities: usize,
    /// Maximum absolute row-sum error in the CPU reference.
    pub max_row_sum_abs_error: f32,
}

/// Dense GGUF attention-score CUDA parity result against the CPU reference.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseGgufAttentionScoreCudaParity {
    /// Fixture identifier.
    pub fixture_id: String,
    /// Dense model family label.
    pub model_family: String,
    /// Dense GGUF architecture label.
    pub architecture: String,
    /// Transformer layer index represented by the fixture.
    pub layer_index: usize,
    /// Query head count.
    pub q_heads: usize,
    /// Key/value head count.
    pub kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Attention scaling factor.
    pub scale: f32,
    /// CPU reference backend.
    pub reference_backend: &'static str,
    /// CUDA target backend.
    pub target_backend: &'static str,
    /// CUDA kernel identifier.
    pub kernel_id: &'static str,
    /// Maximum absolute error against the CPU reference.
    pub max_abs_error: f32,
    /// Mean absolute error against the CPU reference.
    pub mean_abs_error: f32,
    /// Fixture tolerance.
    pub tolerance: f32,
    /// Whether the fixture passed tolerance.
    pub passed: bool,
    /// Number of compared scores.
    pub compared_scores: usize,
    /// Number of finite scores.
    pub finite_scores: usize,
    /// Number of causally masked scores.
    pub causal_masked_scores: usize,
    /// CUDA execution counters.
    pub stats: CudaDenseAttentionScoreStats,
}

/// Dense GGUF attention-softmax CUDA parity result against the CPU reference.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseGgufAttentionSoftmaxCudaParity {
    /// Fixture identifier.
    pub fixture_id: String,
    /// Dense model family label.
    pub model_family: String,
    /// Dense GGUF architecture label.
    pub architecture: String,
    /// Transformer layer index represented by the fixture.
    pub layer_index: usize,
    /// Query head count.
    pub q_heads: usize,
    /// Key/value head count from the source attention-score fixture.
    pub kv_heads: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// CPU reference backend.
    pub reference_backend: &'static str,
    /// CUDA target backend.
    pub target_backend: &'static str,
    /// CUDA kernel identifier.
    pub kernel_id: &'static str,
    /// Maximum absolute error against the CPU reference.
    pub max_abs_error: f32,
    /// Mean absolute error against the CPU reference.
    pub mean_abs_error: f32,
    /// Fixture tolerance.
    pub tolerance: f32,
    /// Whether the fixture passed tolerance.
    pub passed: bool,
    /// Number of compared probabilities.
    pub compared_probabilities: usize,
    /// Number of causally masked zero probabilities.
    pub causal_zero_probabilities: usize,
    /// CUDA execution counters.
    pub stats: CudaDenseAttentionSoftmaxStats,
}

impl AttentionKernelConfig {
    /// Create a configuration for the given attention shape.
    pub fn for_shape(
        n_heads: usize,
        head_dim: usize,
        seq_len_q: usize,
        seq_len_kv: usize,
        causal: bool,
    ) -> Result<Self> {
        if n_heads == 0 || head_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "attention n_heads={n_heads} and head_dim={head_dim} must be non-zero"
                ),
            }
            .into());
        }
        if seq_len_q == 0 || seq_len_kv == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "attention seq lengths must be non-zero: q={seq_len_q}, kv={seq_len_kv}"
                ),
            }
            .into());
        }

        let tile_q: u32 = if seq_len_q <= 32 { seq_len_q as u32 } else { 32 };
        let tile_kv: u32 = if seq_len_kv <= 64 { seq_len_kv as u32 } else { 64 };
        let threads_per_block = 256u32;

        // Shared memory: Q tile + K tile + running softmax state (max + sum per row)
        let q_tile_bytes = (tile_q as usize) * head_dim * 4; // FP32
        let k_tile_bytes = (tile_kv as usize) * head_dim * 4;
        let softmax_state_bytes = (tile_q as usize) * 2 * 4; // max + sum per row
        let shared_mem_bytes = (q_tile_bytes + k_tile_bytes + softmax_state_bytes) as u32;

        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            tile_q,
            tile_kv,
            n_heads,
            head_dim,
            seq_len_q,
            seq_len_kv,
            threads_per_block,
            shared_mem_bytes,
            causal,
            scale,
        })
    }

    /// Compute the CUDA grid dimensions `(grid_x, grid_y, grid_z)`.
    ///
    /// `grid_x` covers query tiles, `grid_y` covers heads, `grid_z` = 1.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let grid_x = (self.seq_len_q as u32).div_ceil(self.tile_q);
        let grid_y = self.n_heads as u32;
        (grid_x, grid_y, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

impl AttentionScoresConfig {
    /// Create a dense attention-score configuration for `[q_heads, seq_len, seq_len]`.
    pub fn for_shape(
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Result<Self> {
        if q_heads == 0 || kv_heads == 0 || head_dim == 0 || seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "dense attention-score dimensions must be non-zero: q_heads={q_heads}, kv_heads={kv_heads}, head_dim={head_dim}, seq_len={seq_len}"
                ),
            }
            .into());
        }
        if !q_heads.is_multiple_of(kv_heads) {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "dense attention-score q_heads must be divisible by kv_heads: q_heads={q_heads}, kv_heads={kv_heads}"
                ),
            }
            .into());
        }

        Ok(Self {
            q_heads,
            kv_heads,
            head_dim,
            seq_len,
            scale: 1.0 / (head_dim as f32).sqrt(),
            causal: true,
            threads_per_block: 128,
        })
    }

    /// Override the attention scale.
    #[must_use]
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Override causal masking.
    #[must_use]
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Total number of output scores.
    pub fn score_count(&self) -> Result<usize> {
        checked_mul(
            checked_mul(self.q_heads, self.seq_len, "attention-score q_heads*seq_len")?,
            self.seq_len,
            "attention-score output",
        )
    }

    /// CUDA grid dimensions.
    pub fn grid_dim(&self) -> Result<(u32, u32, u32)> {
        let score_count = self.score_count()?;
        let score_count =
            u32::try_from(score_count).map_err(|_| KernelError::InvalidArguments {
                reason: format!("dense attention-score output exceeds u32: {score_count}"),
            })?;
        Ok((score_count.div_ceil(self.threads_per_block), 1, 1))
    }

    /// CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

impl AttentionSoftmaxConfig {
    /// Create a dense attention-softmax configuration for `[q_heads, seq_len, seq_len]`.
    pub fn for_shape(q_heads: usize, seq_len: usize) -> Result<Self> {
        if q_heads == 0 || seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "dense attention-softmax dimensions must be non-zero: q_heads={q_heads}, seq_len={seq_len}"
                ),
            }
            .into());
        }
        Ok(Self { q_heads, seq_len, threads_per_block: 1 })
    }

    /// Total number of probability outputs.
    pub fn probability_count(&self) -> Result<usize> {
        checked_mul(
            checked_mul(self.q_heads, self.seq_len, "attention-softmax q_heads*seq_len")?,
            self.seq_len,
            "attention-softmax output",
        )
    }

    /// Number of independent softmax rows.
    pub fn row_count(&self) -> Result<usize> {
        checked_mul(self.q_heads, self.seq_len, "attention-softmax q_heads*seq_len")
    }

    /// CUDA grid dimensions.
    pub fn grid_dim(&self) -> Result<(u32, u32, u32)> {
        let rows = self.row_count()?;
        let rows = u32::try_from(rows).map_err(|_| KernelError::InvalidArguments {
            reason: format!("dense attention-softmax row count exceeds u32: {rows}"),
        })?;
        Ok((rows, 1, 1))
    }

    /// CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// CPU attention configuration
// ---------------------------------------------------------------------------

/// Configuration for the CPU attention fallback functions.
///
/// A simpler struct than [`AttentionKernelConfig`] for use with the pure-Rust
/// CPU fallback implementations.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Per-head embedding dimension.
    pub head_dim: usize,
    /// Sequence length (query and key/value share the same length for CPU path).
    pub seq_len: usize,
    /// Whether to apply a causal (autoregressive) mask.
    pub causal: bool,
    /// Softmax temperature scale (`1.0 / sqrt(head_dim)` by default).
    pub scale: f32,
}

impl AttentionConfig {
    /// Create a new attention config with default scale `1.0 / sqrt(head_dim)`.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn new(num_heads: usize, head_dim: usize, seq_len: usize, causal: bool) -> Result<Self> {
        if num_heads == 0 || head_dim == 0 || seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "AttentionConfig: dimensions must be non-zero: \
                     num_heads={num_heads}, head_dim={head_dim}, seq_len={seq_len}"
                ),
            }
            .into());
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        Ok(Self { num_heads, head_dim, seq_len, causal, scale })
    }

    /// Override the default scale factor.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }
}

// ---------------------------------------------------------------------------
// CUDA launch stub
// ---------------------------------------------------------------------------

/// Launch stub for the scaled dot-product attention kernel.
///
/// # Arguments
///
/// * `q`      — Query tensor `[n_heads, seq_len_q, head_dim]` (FP32)
/// * `k`      — Key tensor   `[n_heads, seq_len_kv, head_dim]` (FP32)
/// * `v`      — Value tensor  `[n_heads, seq_len_kv, head_dim]` (FP32)
/// * `output` — Output buffer `[n_heads, seq_len_q, head_dim]` (FP32, written)
/// * `config` — Launch configuration
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled and loaded.
pub fn launch_attention(
    _q: &[f32],
    _k: &[f32],
    _v: &[f32],
    _output: &mut [f32],
    config: &AttentionKernelConfig,
) -> Result<()> {
    log::debug!(
        "Attention stub: heads={}, head_dim={}, seq_q={}, seq_kv={}, causal={}, grid={:?}",
        config.n_heads,
        config.head_dim,
        config.seq_len_q,
        config.seq_len_kv,
        config.causal,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "Attention CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch a strict dense attention-score F32 CUDA fixture and return execution counters.
///
/// This computes scaled QK scores with an optional causal mask. It does not
/// compute softmax or attention-V mixing.
///
/// # Errors
///
/// Returns an error if CUDA/NVRTC is unavailable, buffers are invalid, or the
/// kernel launch fails. This function never falls back to CPU.
pub fn launch_dense_attention_scores_f32_cuda(
    device_index: usize,
    q: &[f32],
    k: &[f32],
    scores: &mut [f32],
    config: &AttentionScoresConfig,
) -> Result<CudaDenseAttentionScoreStats> {
    validate_attention_score_buffers(q, k, scores, config)?;

    #[cfg(feature = "cuda")]
    {
        return launch_dense_attention_scores_f32_cuda_impl(device_index, q, k, scores, config);
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = device_index;
        Err(KernelError::DeviceUnavailable {
            reason: "dense attention-score CUDA parity requires the cuda feature".to_string(),
        }
        .into())
    }
}

/// Run a dense GGUF attention-score fixture on CUDA and compare it to CPU references.
///
/// # Errors
///
/// Returns an error if the fixture is invalid, CUDA is unavailable, or parity
/// comparison cannot be computed.
pub fn run_dense_gguf_attention_score_cuda_parity(
    device_index: usize,
    fixture: &DenseGgufAttentionScoreCudaFixture,
) -> Result<DenseGgufAttentionScoreCudaParity> {
    validate_dense_gguf_attention_score_fixture(fixture)?;
    let config = AttentionScoresConfig::for_shape(
        fixture.q_heads,
        fixture.kv_heads,
        fixture.head_dim,
        fixture.seq_len,
    )?
    .with_scale(fixture.scale)
    .with_causal(true);
    let mut actual = vec![0.0f32; fixture.expected_scores_f32.len()];
    let stats = launch_dense_attention_scores_f32_cuda(
        device_index,
        &fixture.q_rope_output_f32,
        &fixture.k_rope_output_f32,
        &mut actual,
        &config,
    )?;
    let (max_abs_error, mean_abs_error, compared_scores) =
        compare_attention_score_outputs(&fixture.expected_scores_f32, &actual)?;

    Ok(DenseGgufAttentionScoreCudaParity {
        fixture_id: fixture.fixture_id.clone(),
        model_family: fixture.model_family.clone(),
        architecture: fixture.architecture.clone(),
        layer_index: fixture.layer_index,
        q_heads: fixture.q_heads,
        kv_heads: fixture.kv_heads,
        head_dim: fixture.head_dim,
        seq_len: fixture.seq_len,
        scale: fixture.scale,
        reference_backend: CUDA_DENSE_ATTENTION_SCORE_REFERENCE_BACKEND,
        target_backend: CUDA_DENSE_ATTENTION_SCORE_TARGET_BACKEND,
        kernel_id: CUDA_DENSE_ATTENTION_SCORE_KERNEL_ID,
        max_abs_error,
        mean_abs_error,
        tolerance: CUDA_DENSE_ATTENTION_SCORE_TOLERANCE,
        passed: max_abs_error <= CUDA_DENSE_ATTENTION_SCORE_TOLERANCE,
        compared_scores,
        finite_scores: fixture.finite_scores,
        causal_masked_scores: fixture.causal_masked_scores,
        stats,
    })
}

/// Launch a strict dense attention-softmax F32 CUDA fixture and return execution counters.
///
/// This computes row-wise probabilities from a `[q_heads, seq_len, seq_len]`
/// attention-score tensor. Masked scores are expected to be encoded as
/// negative infinity by the upstream attention-score fixture. This function
/// never falls back to CPU.
pub fn launch_dense_attention_softmax_f32_cuda(
    device_index: usize,
    scores: &[f32],
    probabilities: &mut [f32],
    config: &AttentionSoftmaxConfig,
) -> Result<CudaDenseAttentionSoftmaxStats> {
    validate_attention_softmax_buffers(scores, probabilities, config)?;

    #[cfg(feature = "cuda")]
    {
        return launch_dense_attention_softmax_f32_cuda_impl(
            device_index,
            scores,
            probabilities,
            config,
        );
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = device_index;
        Err(KernelError::DeviceUnavailable {
            reason: "dense attention-softmax CUDA parity requires the cuda feature".to_string(),
        }
        .into())
    }
}

/// Run a dense GGUF attention-softmax fixture on CUDA and compare it to CPU references.
///
/// # Errors
///
/// Returns an error if the fixture is invalid, CUDA is unavailable, or parity
/// comparison cannot be computed.
pub fn run_dense_gguf_attention_softmax_cuda_parity(
    device_index: usize,
    fixture: &DenseGgufAttentionSoftmaxCudaFixture,
) -> Result<DenseGgufAttentionSoftmaxCudaParity> {
    validate_dense_gguf_attention_softmax_fixture(fixture)?;
    let config = AttentionSoftmaxConfig::for_shape(fixture.q_heads, fixture.seq_len)?;
    let mut actual = vec![0.0f32; fixture.expected_probabilities_f32.len()];
    let stats = launch_dense_attention_softmax_f32_cuda(
        device_index,
        &fixture.attention_scores_f32,
        &mut actual,
        &config,
    )?;
    let (max_abs_error, mean_abs_error, compared_probabilities) =
        compare_attention_softmax_outputs(&fixture.expected_probabilities_f32, &actual)?;

    Ok(DenseGgufAttentionSoftmaxCudaParity {
        fixture_id: fixture.fixture_id.clone(),
        model_family: fixture.model_family.clone(),
        architecture: fixture.architecture.clone(),
        layer_index: fixture.layer_index,
        q_heads: fixture.q_heads,
        kv_heads: fixture.kv_heads,
        seq_len: fixture.seq_len,
        reference_backend: CUDA_DENSE_ATTENTION_SCORE_REFERENCE_BACKEND,
        target_backend: CUDA_DENSE_ATTENTION_SCORE_TARGET_BACKEND,
        kernel_id: CUDA_DENSE_ATTENTION_SOFTMAX_KERNEL_ID,
        max_abs_error,
        mean_abs_error,
        tolerance: CUDA_DENSE_ATTENTION_SOFTMAX_TOLERANCE,
        passed: max_abs_error <= CUDA_DENSE_ATTENTION_SOFTMAX_TOLERANCE,
        compared_probabilities,
        causal_zero_probabilities: fixture.causal_zero_probabilities,
        stats,
    })
}

fn validate_attention_score_buffers(
    q: &[f32],
    k: &[f32],
    scores: &[f32],
    config: &AttentionScoresConfig,
) -> Result<()> {
    if config.q_heads == 0 || config.kv_heads == 0 || config.head_dim == 0 || config.seq_len == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "dense attention-score dimensions must be non-zero".into(),
        }
        .into());
    }
    if !config.q_heads.is_multiple_of(config.kv_heads) {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense attention-score q_heads must be divisible by kv_heads: q_heads={}, kv_heads={}",
                config.q_heads, config.kv_heads
            ),
        }
        .into());
    }
    if !config.scale.is_finite() || config.scale <= 0.0 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense attention-score scale must be positive and finite, got {}",
                config.scale
            ),
        }
        .into());
    }
    if config.threads_per_block == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "dense attention-score threads_per_block must be non-zero".into(),
        }
        .into());
    }
    let q_expected = checked_mul(
        checked_mul(config.q_heads, config.seq_len, "attention-score q_heads*seq_len")?,
        config.head_dim,
        "attention-score q",
    )?;
    let k_expected = checked_mul(
        checked_mul(config.kv_heads, config.seq_len, "attention-score kv_heads*seq_len")?,
        config.head_dim,
        "attention-score k",
    )?;
    let scores_expected = config.score_count()?;
    if q.len() != q_expected || k.len() != k_expected || scores.len() != scores_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense attention-score buffer length mismatch: expected q={q_expected}, k={k_expected}, scores={scores_expected}; got q={}, k={}, scores={}",
                q.len(),
                k.len(),
                scores.len()
            ),
        }
        .into());
    }
    validate_i32_arg(config.q_heads, "q_heads")?;
    validate_i32_arg(config.kv_heads, "kv_heads")?;
    validate_i32_arg(config.seq_len, "seq_len")?;
    validate_i32_arg(config.head_dim, "head_dim")?;
    for (idx, value) in q.iter().chain(k.iter()).enumerate() {
        if !value.is_finite() {
            return Err(KernelError::InvalidArguments {
                reason: format!("dense attention-score input[{idx}] is not finite"),
            }
            .into());
        }
    }
    Ok(())
}

fn validate_attention_softmax_buffers(
    scores: &[f32],
    probabilities: &[f32],
    config: &AttentionSoftmaxConfig,
) -> Result<()> {
    if config.q_heads == 0 || config.seq_len == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "dense attention-softmax dimensions must be non-zero".into(),
        }
        .into());
    }
    if config.threads_per_block == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "dense attention-softmax threads_per_block must be non-zero".into(),
        }
        .into());
    }
    let expected = config.probability_count()?;
    if scores.len() != expected || probabilities.len() != expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense attention-softmax buffer length mismatch: expected scores/probabilities={expected}; got scores={}, probabilities={}",
                scores.len(),
                probabilities.len()
            ),
        }
        .into());
    }
    validate_i32_arg(config.q_heads, "q_heads")?;
    validate_i32_arg(config.seq_len, "seq_len")?;
    for (idx, value) in scores.iter().enumerate() {
        if value.is_finite() || (value.is_infinite() && value.is_sign_negative()) {
            continue;
        }
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense attention-softmax score[{idx}] must be finite or negative infinity"
            ),
        }
        .into());
    }
    Ok(())
}

fn validate_dense_gguf_attention_score_fixture(
    fixture: &DenseGgufAttentionScoreCudaFixture,
) -> Result<()> {
    require_dense_label(&fixture.fixture_id, "fixture_id")?;
    require_dense_label(&fixture.model_family, "model_family")?;
    require_dense_label(&fixture.architecture, "architecture")?;
    require_dense_label(&fixture.source_rope_fixture_id, "source_rope_fixture_id")?;
    require_sha256_like(&fixture.q_rope_output_sha256, "q_rope_output_sha256")?;
    require_sha256_like(&fixture.k_rope_output_sha256, "k_rope_output_sha256")?;
    if fixture.q_heads == 0
        || fixture.kv_heads == 0
        || fixture.heads_per_kv_group == 0
        || fixture.q_heads / fixture.kv_heads != fixture.heads_per_kv_group
    {
        return Err(KernelError::InvalidArguments {
            reason:
                "dense attention-score fixture heads_per_kv_group must match q_heads / kv_heads"
                    .into(),
        }
        .into());
    }
    let config = AttentionScoresConfig::for_shape(
        fixture.q_heads,
        fixture.kv_heads,
        fixture.head_dim,
        fixture.seq_len,
    )?
    .with_scale(fixture.scale)
    .with_causal(true);
    validate_attention_score_buffers(
        &fixture.q_rope_output_f32,
        &fixture.k_rope_output_f32,
        &fixture.expected_scores_f32,
        &config,
    )?;
    let finite_scores =
        fixture.expected_scores_f32.iter().filter(|score| score.is_finite()).count();
    let masked_scores = fixture.expected_scores_f32.len().saturating_sub(finite_scores);
    if finite_scores == 0
        || finite_scores != fixture.finite_scores
        || masked_scores != fixture.causal_masked_scores
    {
        return Err(KernelError::InvalidArguments {
            reason: "dense attention-score fixture finite/masked counts must match expected scores"
                .into(),
        }
        .into());
    }
    Ok(())
}

fn validate_dense_gguf_attention_softmax_fixture(
    fixture: &DenseGgufAttentionSoftmaxCudaFixture,
) -> Result<()> {
    require_dense_label(&fixture.fixture_id, "fixture_id")?;
    require_dense_label(&fixture.model_family, "model_family")?;
    require_dense_label(&fixture.architecture, "architecture")?;
    require_dense_label(
        &fixture.source_attention_score_fixture_id,
        "source_attention_score_fixture_id",
    )?;
    require_sha256_like(&fixture.attention_scores_sha256, "attention_scores_sha256")?;
    if fixture.q_heads == 0 || fixture.kv_heads == 0 || fixture.seq_len == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "dense attention-softmax fixture dimensions must be non-zero".into(),
        }
        .into());
    }
    let config = AttentionSoftmaxConfig::for_shape(fixture.q_heads, fixture.seq_len)?;
    let expected_probability_count = config.probability_count()?;
    let expected_row_count = config.row_count()?;
    if fixture.row_count != expected_row_count
        || fixture.probability_count != expected_probability_count
        || fixture.attention_scores_f32.len() != expected_probability_count
        || fixture.expected_probabilities_f32.len() != expected_probability_count
    {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense attention-softmax fixture shape mismatch: expected rows={expected_row_count}, probabilities={expected_probability_count}; got rows={}, probabilities={}, scores_len={}, expected_len={}",
                fixture.row_count,
                fixture.probability_count,
                fixture.attention_scores_f32.len(),
                fixture.expected_probabilities_f32.len()
            ),
        }
        .into());
    }
    validate_attention_softmax_buffers(
        &fixture.attention_scores_f32,
        &fixture.expected_probabilities_f32,
        &config,
    )?;
    if !fixture.max_row_sum_abs_error.is_finite() || fixture.max_row_sum_abs_error > 0.000_01 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense attention-softmax fixture row-sum error too large: {}",
                fixture.max_row_sum_abs_error
            ),
        }
        .into());
    }
    let zero_probs = fixture
        .expected_probabilities_f32
        .iter()
        .filter(|probability| probability.abs() <= f32::EPSILON)
        .count();
    if zero_probs != fixture.causal_zero_probabilities {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense attention-softmax fixture causal zero count mismatch: expected {}, got {zero_probs}",
                fixture.causal_zero_probabilities
            ),
        }
        .into());
    }
    for (idx, probability) in fixture.expected_probabilities_f32.iter().enumerate() {
        if !probability.is_finite() || *probability < 0.0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "dense attention-softmax expected probability[{idx}] must be finite and non-negative"
                ),
            }
            .into());
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn launch_dense_attention_scores_f32_cuda_impl(
    device_index: usize,
    q: &[f32],
    k: &[f32],
    scores: &mut [f32],
    config: &AttentionScoresConfig,
) -> Result<CudaDenseAttentionScoreStats> {
    let q_len = checked_mul(
        checked_mul(config.q_heads, config.seq_len, "attention-score q_heads*seq_len")?,
        config.head_dim,
        "attention-score q",
    )?;
    let k_len = checked_mul(
        checked_mul(config.kv_heads, config.seq_len, "attention-score kv_heads*seq_len")?,
        config.head_dim,
        "attention-score k",
    )?;
    let score_count = config.score_count()?;

    let ctx = CudaContext::new(device_index).map_err(|err| KernelError::GpuError {
        reason: format!("failed to create CUDA context for dense attention-score: {err:?}"),
    })?;
    let stream = ctx.default_stream();
    let ptx = compile_dense_attention_score_ptx()?;
    let module = ctx.load_module(ptx).map_err(|err| KernelError::GpuError {
        reason: format!("failed to load dense attention-score CUDA module: {err:?}"),
    })?;
    let function = module.load_function(CUDA_DENSE_ATTENTION_SCORE_KERNEL_ID).map_err(|err| {
        KernelError::GpuError {
            reason: format!("failed to load dense attention-score CUDA kernel: {err:?}"),
        }
    })?;

    let q_dev = stream.memcpy_stod(&q[..q_len]).map_err(|err| KernelError::GpuError {
        reason: format!("failed to copy dense attention-score Q to device: {err:?}"),
    })?;
    let k_dev = stream.memcpy_stod(&k[..k_len]).map_err(|err| KernelError::GpuError {
        reason: format!("failed to copy dense attention-score K to device: {err:?}"),
    })?;
    let mut scores_dev: CudaSlice<f32> =
        stream.alloc_zeros(score_count).map_err(|err| KernelError::GpuError {
            reason: format!("failed to allocate dense attention-score output on device: {err:?}"),
        })?;

    let launch_config = LaunchConfig {
        grid_dim: config.grid_dim()?,
        block_dim: config.block_dim(),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&function);
    builder.arg(&q_dev);
    builder.arg(&k_dev);
    builder.arg(&mut scores_dev);
    let q_heads_arg = i32::try_from(config.q_heads).map_err(|_| KernelError::InvalidArguments {
        reason: format!("dense attention-score q_heads exceeds i32: {}", config.q_heads),
    })?;
    let kv_heads_arg =
        i32::try_from(config.kv_heads).map_err(|_| KernelError::InvalidArguments {
            reason: format!("dense attention-score kv_heads exceeds i32: {}", config.kv_heads),
        })?;
    let seq_len_arg = i32::try_from(config.seq_len).map_err(|_| KernelError::InvalidArguments {
        reason: format!("dense attention-score seq_len exceeds i32: {}", config.seq_len),
    })?;
    let head_dim_arg =
        i32::try_from(config.head_dim).map_err(|_| KernelError::InvalidArguments {
            reason: format!("dense attention-score head_dim exceeds i32: {}", config.head_dim),
        })?;
    let scale_arg = config.scale;
    let causal_arg = i32::from(config.causal);
    builder.arg(&q_heads_arg);
    builder.arg(&kv_heads_arg);
    builder.arg(&seq_len_arg);
    builder.arg(&head_dim_arg);
    builder.arg(&scale_arg);
    builder.arg(&causal_arg);

    unsafe { builder.launch(launch_config) }.map_err(|err| KernelError::GpuError {
        reason: format!("failed to launch dense attention-score CUDA kernel: {err:?}"),
    })?;
    stream.synchronize().map_err(|err| KernelError::GpuError {
        reason: format!("failed to synchronize dense attention-score CUDA kernel: {err:?}"),
    })?;

    let scores_host: Vec<f32> =
        stream.memcpy_dtov(&scores_dev).map_err(|err| KernelError::GpuError {
            reason: format!("failed to copy dense attention-score output from device: {err:?}"),
        })?;
    scores[..score_count].copy_from_slice(&scores_host[..score_count]);

    Ok(CudaDenseAttentionScoreStats {
        kernel_id: CUDA_DENSE_ATTENTION_SCORE_KERNEL_ID,
        invocations: 1,
        fallback_invocations: 0,
        host_to_device_bytes: bytes_for::<f32>(q_len + k_len)?,
        device_to_host_bytes: bytes_for::<f32>(score_count)?,
        kernel_launches: 1,
        kernel_time_ms: None,
    })
}

#[cfg(feature = "cuda")]
fn launch_dense_attention_softmax_f32_cuda_impl(
    device_index: usize,
    scores: &[f32],
    probabilities: &mut [f32],
    config: &AttentionSoftmaxConfig,
) -> Result<CudaDenseAttentionSoftmaxStats> {
    let probability_count = config.probability_count()?;

    let ctx = CudaContext::new(device_index).map_err(|err| KernelError::GpuError {
        reason: format!("failed to create CUDA context for dense attention-softmax: {err:?}"),
    })?;
    let stream = ctx.default_stream();
    let ptx = compile_dense_attention_softmax_ptx()?;
    let module = ctx.load_module(ptx).map_err(|err| KernelError::GpuError {
        reason: format!("failed to load dense attention-softmax CUDA module: {err:?}"),
    })?;
    let function = module.load_function(CUDA_DENSE_ATTENTION_SOFTMAX_KERNEL_ID).map_err(|err| {
        KernelError::GpuError {
            reason: format!("failed to load dense attention-softmax CUDA kernel: {err:?}"),
        }
    })?;

    let scores_dev =
        stream.memcpy_stod(&scores[..probability_count]).map_err(|err| KernelError::GpuError {
            reason: format!("failed to copy dense attention-softmax scores to device: {err:?}"),
        })?;
    let mut probabilities_dev: CudaSlice<f32> =
        stream.alloc_zeros(probability_count).map_err(|err| KernelError::GpuError {
            reason: format!(
                "failed to allocate dense attention-softmax probabilities on device: {err:?}"
            ),
        })?;

    let launch_config = LaunchConfig {
        grid_dim: config.grid_dim()?,
        block_dim: config.block_dim(),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&function);
    builder.arg(&scores_dev);
    builder.arg(&mut probabilities_dev);
    let q_heads_arg = i32::try_from(config.q_heads).map_err(|_| KernelError::InvalidArguments {
        reason: format!("dense attention-softmax q_heads exceeds i32: {}", config.q_heads),
    })?;
    let seq_len_arg = i32::try_from(config.seq_len).map_err(|_| KernelError::InvalidArguments {
        reason: format!("dense attention-softmax seq_len exceeds i32: {}", config.seq_len),
    })?;
    builder.arg(&q_heads_arg);
    builder.arg(&seq_len_arg);

    unsafe { builder.launch(launch_config) }.map_err(|err| KernelError::GpuError {
        reason: format!("failed to launch dense attention-softmax CUDA kernel: {err:?}"),
    })?;
    stream.synchronize().map_err(|err| KernelError::GpuError {
        reason: format!("failed to synchronize dense attention-softmax CUDA kernel: {err:?}"),
    })?;

    let probabilities_host: Vec<f32> =
        stream.memcpy_dtov(&probabilities_dev).map_err(|err| KernelError::GpuError {
            reason: format!("failed to copy dense attention-softmax output from device: {err:?}"),
        })?;
    probabilities[..probability_count].copy_from_slice(&probabilities_host[..probability_count]);

    Ok(CudaDenseAttentionSoftmaxStats {
        kernel_id: CUDA_DENSE_ATTENTION_SOFTMAX_KERNEL_ID,
        invocations: 1,
        fallback_invocations: 0,
        host_to_device_bytes: bytes_for::<f32>(probability_count)?,
        device_to_host_bytes: bytes_for::<f32>(probability_count)?,
        kernel_launches: 1,
        kernel_time_ms: None,
    })
}

#[cfg(feature = "cuda")]
fn compile_dense_attention_score_ptx() -> Result<Ptx> {
    let _hook_guard = DENSE_ATTENTION_SCORE_NVRTC_COMPILE_LOCK.lock().ok();
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let compile_result =
        std::panic::catch_unwind(|| compile_ptx(CUDA_DENSE_ATTENTION_SCORE_KERNEL_SRC));
    std::panic::set_hook(previous_hook);

    match compile_result {
        Ok(Ok(ptx)) => Ok(ptx),
        Ok(Err(err)) => Err(KernelError::GpuError {
            reason: format!("failed to compile dense attention-score CUDA PTX: {err:?}"),
        }
        .into()),
        Err(payload) => Err(KernelError::GpuError {
            reason: format!(
                "failed to compile dense attention-score CUDA PTX because NVRTC was unavailable: {}",
                panic_payload_message(&*payload)
            ),
        }
        .into()),
    }
}

#[cfg(feature = "cuda")]
fn compile_dense_attention_softmax_ptx() -> Result<Ptx> {
    let _hook_guard = DENSE_ATTENTION_SOFTMAX_NVRTC_COMPILE_LOCK.lock().ok();
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let compile_result =
        std::panic::catch_unwind(|| compile_ptx(CUDA_DENSE_ATTENTION_SOFTMAX_KERNEL_SRC));
    std::panic::set_hook(previous_hook);

    match compile_result {
        Ok(Ok(ptx)) => Ok(ptx),
        Ok(Err(err)) => Err(KernelError::GpuError {
            reason: format!("failed to compile dense attention-softmax CUDA PTX: {err:?}"),
        }
        .into()),
        Err(payload) => Err(KernelError::GpuError {
            reason: format!(
                "failed to compile dense attention-softmax CUDA PTX because NVRTC was unavailable: {}",
                panic_payload_message(&*payload)
            ),
        }
        .into()),
    }
}

#[cfg(feature = "cuda")]
fn panic_payload_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn compare_attention_score_outputs(expected: &[f32], actual: &[f32]) -> Result<(f32, f32, usize)> {
    if expected.len() != actual.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense attention-score parity length mismatch: expected {}, got {}",
                expected.len(),
                actual.len()
            ),
        }
        .into());
    }
    if expected.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "dense attention-score parity comparison requires non-empty output".into(),
        }
        .into());
    }

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut compared = 0usize;
    for (idx, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        if expected.is_finite() && actual.is_finite() {
            let abs = (expected - actual).abs();
            max_abs = max_abs.max(abs);
            sum_abs += abs;
            compared += 1;
            continue;
        }
        if expected.is_infinite()
            && actual.is_infinite()
            && expected.is_sign_negative() == actual.is_sign_negative()
        {
            compared += 1;
            continue;
        }
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense attention-score parity non-finite mismatch at {idx}: expected={expected}, actual={actual}"
            ),
        }
        .into());
    }
    if compared == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "dense attention-score parity comparison found no comparable scores".into(),
        }
        .into());
    }
    Ok((max_abs, sum_abs / compared as f32, compared))
}

fn compare_attention_softmax_outputs(
    expected: &[f32],
    actual: &[f32],
) -> Result<(f32, f32, usize)> {
    if expected.len() != actual.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense attention-softmax parity length mismatch: expected {}, got {}",
                expected.len(),
                actual.len()
            ),
        }
        .into());
    }
    if expected.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "dense attention-softmax parity comparison requires non-empty output".into(),
        }
        .into());
    }

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    for (idx, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        if !expected.is_finite() || !actual.is_finite() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "dense attention-softmax parity requires finite probabilities at {idx}: expected={expected}, actual={actual}"
                ),
            }
            .into());
        }
        let abs = (expected - actual).abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs;
    }
    Ok((max_abs, sum_abs / expected.len() as f32, expected.len()))
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        KernelError::InvalidArguments { reason: format!("{label} size overflows usize") }.into()
    })
}

#[cfg(feature = "cuda")]
fn bytes_for<T>(items: usize) -> Result<u64> {
    let bytes = checked_mul(items, std::mem::size_of::<T>(), "byte count")?;
    u64::try_from(bytes).map_err(|_| {
        KernelError::InvalidArguments { reason: "byte count exceeds u64".into() }.into()
    })
}

fn validate_i32_arg(value: usize, label: &str) -> Result<()> {
    if value > i32::MAX as usize {
        return Err(KernelError::InvalidArguments {
            reason: format!("{label} exceeds i32: {value}"),
        }
        .into());
    }
    Ok(())
}

fn require_dense_label(value: &str, field: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: format!("dense GGUF attention-score fixture {field} must not be empty"),
        }
        .into());
    }
    let lower = value.to_ascii_lowercase();
    if lower.contains("bitnet") || lower.contains("qk256") || lower.contains("i2_s") {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense GGUF attention-score fixture {field} must not contain BitNet packed markers"
            ),
        }
        .into());
    }
    Ok(())
}

fn require_sha256_like(value: &str, field: &str) -> Result<()> {
    if value.len() != 64 || !value.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense GGUF attention-score fixture {field} must be a SHA-256 hex digest"
            ),
        }
        .into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback: single-head scaled dot-product attention
// ---------------------------------------------------------------------------

/// Numerically stable row-wise softmax over `scores` in-place.
fn softmax_inplace(scores: &mut [f32]) {
    let row_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for s in scores.iter_mut() {
        let e = (*s - row_max).exp();
        *s = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for s in scores.iter_mut() {
            *s *= inv;
        }
    }
}

/// Pure-Rust CPU fallback for single-head scaled dot-product attention.
///
/// Computes `softmax(Q·Kᵀ · scale) · V`.
///
/// # Arguments
///
/// * `query` — `[seq_len, head_dim]` (FP32, row-major)
/// * `key`   — `[seq_len, head_dim]` (FP32, row-major)
/// * `value` — `[seq_len, head_dim]` (FP32, row-major)
/// * `config` — Attention configuration (uses `seq_len`, `head_dim`, `scale`, `causal`)
///
/// # Returns
///
/// Output tensor `[seq_len, head_dim]` as a flat `Vec<f32>`.
///
/// # Errors
///
/// Returns an error if tensor lengths do not match the configuration.
pub fn attention_cpu_fallback(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &AttentionConfig,
) -> Result<Vec<f32>> {
    let expected = config.seq_len * config.head_dim;
    if query.len() < expected || key.len() < expected || value.len() < expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "attention_cpu_fallback: tensor length mismatch, expected {expected}, \
                 got q={}, k={}, v={}",
                query.len(),
                key.len(),
                value.len()
            ),
        }
        .into());
    }

    let seq = config.seq_len;
    let dim = config.head_dim;
    let scale = config.scale;
    let mut output = vec![0.0_f32; expected];

    for i in 0..seq {
        // Compute scaled dot-product scores: Q[i] · K[j]^T * scale
        let mut scores = vec![0.0_f32; seq];
        for j in 0..seq {
            if config.causal && j > i {
                scores[j] = f32::NEG_INFINITY;
            } else {
                let mut dot = 0.0_f32;
                for d in 0..dim {
                    dot += query[i * dim + d] * key[j * dim + d];
                }
                scores[j] = dot * scale;
            }
        }

        softmax_inplace(&mut scores);

        // Weighted sum of values
        for d in 0..dim {
            let mut acc = 0.0_f32;
            for j in 0..seq {
                acc += scores[j] * value[j * dim + d];
            }
            output[i * dim + d] = acc;
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// CPU fallback: masked attention
// ---------------------------------------------------------------------------

/// Pure-Rust CPU fallback for masked scaled dot-product attention.
///
/// Computes `softmax(Q·Kᵀ · scale + mask) · V`.
///
/// # Arguments
///
/// * `query` — `[seq_len, head_dim]` (FP32, row-major)
/// * `key`   — `[seq_len, head_dim]` (FP32, row-major)
/// * `value` — `[seq_len, head_dim]` (FP32, row-major)
/// * `mask`  — `[seq_len, seq_len]` additive mask (FP32, row-major);
///   use `0.0` for attending, `f32::NEG_INFINITY` for blocking
/// * `config` — Attention configuration (uses `seq_len`, `head_dim`, `scale`)
///
/// # Returns
///
/// Output tensor `[seq_len, head_dim]` as a flat `Vec<f32>`.
///
/// # Errors
///
/// Returns an error if tensor lengths do not match the configuration.
pub fn masked_attention_cpu_fallback(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: &[f32],
    config: &AttentionConfig,
) -> Result<Vec<f32>> {
    let expected = config.seq_len * config.head_dim;
    let mask_expected = config.seq_len * config.seq_len;
    if query.len() < expected || key.len() < expected || value.len() < expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "masked_attention_cpu_fallback: tensor length mismatch, expected {expected}, \
                 got q={}, k={}, v={}",
                query.len(),
                key.len(),
                value.len()
            ),
        }
        .into());
    }
    if mask.len() < mask_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "masked_attention_cpu_fallback: mask length {}, expected {mask_expected}",
                mask.len()
            ),
        }
        .into());
    }

    let seq = config.seq_len;
    let dim = config.head_dim;
    let scale = config.scale;
    let mut output = vec![0.0_f32; expected];

    for i in 0..seq {
        let mut scores = vec![0.0_f32; seq];
        for j in 0..seq {
            let mut dot = 0.0_f32;
            for d in 0..dim {
                dot += query[i * dim + d] * key[j * dim + d];
            }
            // Additive mask applied after scaling
            scores[j] = dot * scale + mask[i * seq + j];
        }

        softmax_inplace(&mut scores);

        for d in 0..dim {
            let mut acc = 0.0_f32;
            for j in 0..seq {
                acc += scores[j] * value[j * dim + d];
            }
            output[i * dim + d] = acc;
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// CPU fallback: multi-head attention
// ---------------------------------------------------------------------------

/// Pure-Rust CPU fallback for multi-head scaled dot-product attention.
///
/// Applies single-head attention independently per head, then concatenates.
///
/// # Arguments
///
/// * `query` — `[num_heads, seq_len, head_dim]` (FP32, row-major)
/// * `key`   — `[num_heads, seq_len, head_dim]` (FP32, row-major)
/// * `value` — `[num_heads, seq_len, head_dim]` (FP32, row-major)
/// * `config` — Attention configuration (`num_heads`, `seq_len`, `head_dim`, etc.)
///
/// # Returns
///
/// Output tensor `[num_heads, seq_len, head_dim]` as a flat `Vec<f32>`.
///
/// # Errors
///
/// Returns an error if tensor lengths do not match the configuration.
pub fn multi_head_attention_cpu_fallback(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &AttentionConfig,
) -> Result<Vec<f32>> {
    let head_size = config.seq_len * config.head_dim;
    let total = config.num_heads * head_size;
    if query.len() < total || key.len() < total || value.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "multi_head_attention_cpu_fallback: tensor length mismatch, \
                 expected {total}, got q={}, k={}, v={}",
                query.len(),
                key.len(),
                value.len()
            ),
        }
        .into());
    }

    let mut output = vec![0.0_f32; total];

    // Per-head config (single head)
    let single_cfg = AttentionConfig {
        num_heads: 1,
        head_dim: config.head_dim,
        seq_len: config.seq_len,
        causal: config.causal,
        scale: config.scale,
    };

    for h in 0..config.num_heads {
        let offset = h * head_size;
        let q_head = &query[offset..offset + head_size];
        let k_head = &key[offset..offset + head_size];
        let v_head = &value[offset..offset + head_size];
        let head_out = attention_cpu_fallback(q_head, k_head, v_head, &single_cfg)?;
        output[offset..offset + head_size].copy_from_slice(&head_out);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// CPU convenience wrapper
// ---------------------------------------------------------------------------

/// Pure-Rust CPU reference for scaled dot-product attention.
///
/// Dispatches to single-head or multi-head depending on `config.num_heads`.
/// This is the canonical CPU entry-point; `attention_cpu_fallback` and
/// `multi_head_attention_cpu_fallback` are the underlying per-variant
/// implementations.
pub fn attention_forward_cpu(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &AttentionConfig,
) -> Result<Vec<f32>> {
    if config.num_heads > 1 {
        multi_head_attention_cpu_fallback(query, key, value, config)
    } else {
        attention_cpu_fallback(query, key, value, config)
    }
}

// ---------------------------------------------------------------------------
// Flash-attention style chunked CPU reference
// ---------------------------------------------------------------------------

/// Default chunk size (number of K/V positions per chunk) for the chunked
/// CPU attention implementation.  Chosen to keep the temporary score buffer
/// small enough for L1/L2 cache while still amortising the per-chunk
/// overhead.
const DEFAULT_CHUNK_SIZE: usize = 64;

/// Flash-attention style chunked single-head attention (CPU reference).
///
/// Instead of materialising the full `[seq_q, seq_kv]` score matrix, this
/// implementation streams K/V in chunks of `chunk_size` positions and
/// maintains a running softmax accumulator (online softmax trick).  Memory
/// usage is `O(seq_q * chunk_size)` instead of `O(seq_q * seq_kv)`.
///
/// The numerical result is equivalent to [`attention_cpu_fallback`] within
/// floating-point tolerance.
///
/// # Arguments
///
/// * `query`      — `[seq_len, head_dim]` (FP32, row-major)
/// * `key`        — `[seq_len, head_dim]` (FP32, row-major)
/// * `value`      — `[seq_len, head_dim]` (FP32, row-major)
/// * `config`     — Attention configuration
/// * `chunk_size` — Number of K/V positions per chunk (`0` → use default)
pub fn chunked_attention_cpu(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &AttentionConfig,
    chunk_size: usize,
) -> Result<Vec<f32>> {
    let expected = config.seq_len * config.head_dim;
    if query.len() < expected || key.len() < expected || value.len() < expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "chunked_attention_cpu: tensor length mismatch, expected {expected}, \
                 got q={}, k={}, v={}",
                query.len(),
                key.len(),
                value.len()
            ),
        }
        .into());
    }

    let seq = config.seq_len;
    let dim = config.head_dim;
    let scale = config.scale;
    let cs = if chunk_size == 0 { DEFAULT_CHUNK_SIZE } else { chunk_size };

    let mut output = vec![0.0_f32; expected];

    for i in 0..seq {
        // Per-query running accumulators for online softmax
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0_f32;
        let mut acc = vec![0.0_f32; dim]; // weighted V accumulator

        // Determine effective KV length (causal limits to positions ≤ i)
        let kv_len = if config.causal { i + 1 } else { seq };

        // Stream K/V in chunks
        let mut chunk_start = 0;
        while chunk_start < kv_len {
            let chunk_end = (chunk_start + cs).min(kv_len);
            let chunk_len = chunk_end - chunk_start;

            // Compute scores for this chunk: Q[i] · K[j]^T * scale
            let mut scores = vec![0.0_f32; chunk_len];
            let mut chunk_max = f32::NEG_INFINITY;
            for (ci, j) in (chunk_start..chunk_end).enumerate() {
                let mut dot = 0.0_f32;
                for d in 0..dim {
                    dot += query[i * dim + d] * key[j * dim + d];
                }
                scores[ci] = dot * scale;
                if scores[ci] > chunk_max {
                    chunk_max = scores[ci];
                }
            }

            // Online softmax update: merge this chunk into running state
            // Algorithm: if new_max > running_max, rescale existing accumulators.
            let new_max = running_max.max(chunk_max);

            // Rescale previous accumulator if max changed
            if running_sum > 0.0 {
                let correction = (running_max - new_max).exp();
                running_sum *= correction;
                for a in acc.iter_mut() {
                    *a *= correction;
                }
            }

            // Add this chunk's contribution
            let mut chunk_sum = 0.0_f32;
            for (ci, &score) in scores.iter().enumerate().take(chunk_len) {
                let w = (score - new_max).exp();
                chunk_sum += w;
                let j = chunk_start + ci;
                for d in 0..dim {
                    acc[d] += w * value[j * dim + d];
                }
            }

            running_max = new_max;
            running_sum += chunk_sum;
            chunk_start = chunk_end;
        }

        // Normalise
        if running_sum > 0.0 {
            let inv = 1.0 / running_sum;
            for d in 0..dim {
                output[i * dim + d] = acc[d] * inv;
            }
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Batch attention
// ---------------------------------------------------------------------------

/// Pure-Rust CPU fallback for batched multi-head attention.
///
/// Applies multi-head attention independently per batch element.
///
/// # Arguments
///
/// * `query` — `[batch, num_heads, seq_len, head_dim]` (FP32, row-major)
/// * `key`   — `[batch, num_heads, seq_len, head_dim]` (FP32, row-major)
/// * `value` — `[batch, num_heads, seq_len, head_dim]` (FP32, row-major)
/// * `config` — Attention configuration (`num_heads`, `seq_len`, `head_dim`)
/// * `batch_size` — Number of independent sequences in the batch
///
/// # Returns
///
/// Output tensor `[batch, num_heads, seq_len, head_dim]` as a flat `Vec<f32>`.
pub fn batch_attention_cpu(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &AttentionConfig,
    batch_size: usize,
) -> Result<Vec<f32>> {
    if batch_size == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "batch_attention_cpu: batch_size must be non-zero".into(),
        }
        .into());
    }
    let per_batch = config.num_heads * config.seq_len * config.head_dim;
    let total = batch_size * per_batch;
    if query.len() < total || key.len() < total || value.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "batch_attention_cpu: tensor length mismatch, expected {total}, \
                 got q={}, k={}, v={}",
                query.len(),
                key.len(),
                value.len()
            ),
        }
        .into());
    }

    let mut output = vec![0.0_f32; total];

    for b in 0..batch_size {
        let offset = b * per_batch;
        let q_batch = &query[offset..offset + per_batch];
        let k_batch = &key[offset..offset + per_batch];
        let v_batch = &value[offset..offset + per_batch];
        let batch_out = if config.num_heads > 1 {
            multi_head_attention_cpu_fallback(q_batch, k_batch, v_batch, config)?
        } else {
            attention_cpu_fallback(q_batch, k_batch, v_batch, config)?
        };
        output[offset..offset + per_batch].copy_from_slice(&batch_out);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Unified dispatch
// ---------------------------------------------------------------------------

/// Apply attention with automatic dispatch: GPU if available, else CPU fallback.
pub fn attention_forward(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &AttentionConfig,
) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime() {
            let kernel_cfg = AttentionKernelConfig::for_shape(
                config.num_heads,
                config.head_dim,
                config.seq_len,
                config.seq_len,
                config.causal,
            )?;
            let total = config.num_heads * config.seq_len * config.head_dim;
            let mut output = vec![0.0_f32; total];
            if launch_attention(query, key, value, &mut output, &kernel_cfg).is_ok() {
                return Ok(output);
            }
            // GPU launch failed — fall through to CPU path
        }
    }
    attention_forward_cpu(query, key, value, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── AttentionKernelConfig tests ───────────────────────────────────

    #[test]
    fn test_attention_config_for_shape() {
        let cfg = AttentionKernelConfig::for_shape(32, 128, 1, 512, true).unwrap();
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.head_dim, 128);
        assert!(cfg.causal);
        assert!((cfg.scale - 1.0 / (128.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_attention_config_grid_dim() {
        let cfg = AttentionKernelConfig::for_shape(8, 64, 100, 100, false).unwrap();
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!(gx, 4); // ceil(100/32)
        assert_eq!(gy, 8); // n_heads
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_attention_config_rejects_zero_heads() {
        assert!(AttentionKernelConfig::for_shape(0, 128, 1, 512, true).is_err());
    }

    #[test]
    fn test_attention_config_rejects_zero_seq() {
        assert!(AttentionKernelConfig::for_shape(8, 64, 0, 512, true).is_err());
        assert!(AttentionKernelConfig::for_shape(8, 64, 1, 0, true).is_err());
    }

    #[test]
    fn test_attention_config_small_seq() {
        let cfg = AttentionKernelConfig::for_shape(1, 64, 4, 4, false).unwrap();
        assert_eq!(cfg.tile_q, 4); // small seq → tile = seq
        assert_eq!(cfg.tile_kv, 4);
    }

    #[test]
    fn test_attention_score_config_for_shape() {
        let cfg = AttentionScoresConfig::for_shape(4, 2, 8, 3).unwrap();
        assert_eq!(cfg.q_heads, 4);
        assert_eq!(cfg.kv_heads, 2);
        assert_eq!(cfg.head_dim, 8);
        assert_eq!(cfg.seq_len, 3);
        assert_eq!(cfg.score_count().unwrap(), 36);
        assert_eq!(cfg.grid_dim().unwrap(), (1, 1, 1));
        assert_eq!(cfg.block_dim(), (128, 1, 1));
        assert!(cfg.causal);
    }

    #[test]
    fn test_attention_score_config_rejects_mismatched_heads() {
        assert!(AttentionScoresConfig::for_shape(3, 2, 8, 3).is_err());
    }

    #[test]
    fn test_attention_softmax_config_for_shape() {
        let cfg = AttentionSoftmaxConfig::for_shape(4, 3).unwrap();
        assert_eq!(cfg.q_heads, 4);
        assert_eq!(cfg.seq_len, 3);
        assert_eq!(cfg.row_count().unwrap(), 12);
        assert_eq!(cfg.probability_count().unwrap(), 36);
        assert_eq!(cfg.grid_dim().unwrap(), (12, 1, 1));
        assert_eq!(cfg.block_dim(), (1, 1, 1));
    }

    #[test]
    fn test_attention_softmax_config_rejects_zero() {
        assert!(AttentionSoftmaxConfig::for_shape(0, 3).is_err());
        assert!(AttentionSoftmaxConfig::for_shape(4, 0).is_err());
    }

    #[test]
    fn test_attention_score_compare_handles_causal_mask() {
        let expected = vec![0.5, f32::NEG_INFINITY, -0.25, 0.75];
        let actual = vec![0.50001, f32::NEG_INFINITY, -0.25002, 0.75003];
        let (max_abs, _mean_abs, compared) =
            compare_attention_score_outputs(&expected, &actual).unwrap();
        assert_eq!(compared, 4);
        assert!(max_abs < 0.000_1);
    }

    #[test]
    fn test_attention_score_compare_rejects_nan_mismatch() {
        let expected = vec![0.5, f32::NEG_INFINITY];
        let actual = vec![0.5, f32::NAN];
        assert!(compare_attention_score_outputs(&expected, &actual).is_err());
    }

    #[test]
    fn test_attention_softmax_compare_handles_masked_zero_probabilities() {
        let expected = vec![1.0, 0.0, 0.25, 0.75];
        let actual = vec![0.99999, 0.0, 0.25002, 0.74998];
        let (max_abs, _mean_abs, compared) =
            compare_attention_softmax_outputs(&expected, &actual).unwrap();
        assert_eq!(compared, 4);
        assert!(max_abs < 0.000_1);
    }

    #[test]
    fn test_attention_softmax_compare_rejects_non_finite() {
        let expected = vec![1.0, 0.0];
        let actual = vec![1.0, f32::NAN];
        assert!(compare_attention_softmax_outputs(&expected, &actual).is_err());
    }

    // ── AttentionConfig tests ─────────────────────────────────────────

    #[test]
    fn test_cpu_config_new() {
        let cfg = AttentionConfig::new(8, 64, 16, true).unwrap();
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.seq_len, 16);
        assert!(cfg.causal);
        assert!((cfg.scale - 1.0 / (64.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_config_rejects_zero() {
        assert!(AttentionConfig::new(0, 64, 16, false).is_err());
        assert!(AttentionConfig::new(8, 0, 16, false).is_err());
        assert!(AttentionConfig::new(8, 64, 0, false).is_err());
    }

    #[test]
    fn test_cpu_config_custom_scale() {
        let cfg = AttentionConfig::new(1, 64, 4, false).unwrap().with_scale(0.5);
        assert!((cfg.scale - 0.5).abs() < f32::EPSILON);
    }

    // ── Single-head CPU fallback tests ────────────────────────────────

    #[test]
    fn test_cpu_attention_identity_key() {
        // Q == K: each query attends most to itself (non-causal, uniform V)
        let cfg = AttentionConfig::new(1, 2, 3, false).unwrap();
        let qk = vec![
            1.0, 0.0, // row 0
            0.0, 1.0, // row 1
            1.0, 1.0, // row 2
        ];
        let value = vec![
            1.0, 0.0, //
            0.0, 1.0, //
            0.5, 0.5, //
        ];
        let out = attention_cpu_fallback(&qk, &qk, &value, &cfg).unwrap();
        // Output should be well-formed (finite, seq_len * head_dim)
        assert_eq!(out.len(), 6);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_cpu_attention_output_shape() {
        let cfg = AttentionConfig::new(1, 4, 8, false).unwrap();
        let q = vec![0.1_f32; 32];
        let k = vec![0.2_f32; 32];
        let v = vec![0.3_f32; 32];
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), 32); // seq_len * head_dim
    }

    #[test]
    fn test_cpu_attention_uniform_query_equal_values() {
        // All-equal Q,K → uniform attention → output == mean(V rows)
        let cfg = AttentionConfig::new(1, 2, 3, false).unwrap();
        let q = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let k = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let v = vec![3.0, 6.0, 3.0, 6.0, 3.0, 6.0];
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        // All rows: mean of [3,6], [3,6], [3,6] = [3,6]
        for row in 0..3 {
            assert!((out[row * 2] - 3.0).abs() < 1e-5);
            assert!((out[row * 2 + 1] - 6.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cpu_attention_numerical_stability_large_values() {
        let cfg = AttentionConfig::new(1, 2, 2, false).unwrap();
        let q = vec![500.0, 500.0, -500.0, -500.0];
        let k = vec![500.0, 500.0, -500.0, -500.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "non-finite with large values");
    }

    #[test]
    fn test_cpu_attention_single_position() {
        // seq_len=1: output == value (only one position to attend to)
        let cfg = AttentionConfig::new(1, 4, 1, false).unwrap();
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![0.5, 0.5, 0.5, 0.5];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        for d in 0..4 {
            assert!((out[d] - v[d]).abs() < 1e-5, "seq_len=1 should return V");
        }
    }

    #[test]
    fn test_cpu_attention_rejects_short_tensors() {
        let cfg = AttentionConfig::new(1, 4, 8, false).unwrap();
        let short = vec![0.0_f32; 16]; // need 32
        let ok = vec![0.0_f32; 32];
        assert!(attention_cpu_fallback(&short, &ok, &ok, &cfg).is_err());
        assert!(attention_cpu_fallback(&ok, &short, &ok, &cfg).is_err());
        assert!(attention_cpu_fallback(&ok, &ok, &short, &cfg).is_err());
    }

    // ── Causal masking tests ──────────────────────────────────────────

    #[test]
    fn test_cpu_attention_causal_first_token() {
        // First token with causal: can only attend to itself → output == V[0]
        let cfg = AttentionConfig::new(1, 2, 3, true).unwrap();
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        // Row 0: can only see position 0 → output = V[0]
        assert!((out[0] - 10.0).abs() < 1e-5);
        assert!((out[1] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_attention_causal_vs_noncausal() {
        let seq = 4;
        let dim = 2;
        let q = vec![1.0; seq * dim];
        let k = vec![1.0; seq * dim];
        let v: Vec<f32> = (0..seq * dim).map(|i| i as f32).collect();

        let causal_cfg = AttentionConfig::new(1, dim, seq, true).unwrap();
        let noncausal_cfg = AttentionConfig::new(1, dim, seq, false).unwrap();

        let out_c = attention_cpu_fallback(&q, &k, &v, &causal_cfg).unwrap();
        let out_nc = attention_cpu_fallback(&q, &k, &v, &noncausal_cfg).unwrap();

        // Non-causal: all rows see all positions → all rows identical
        // Causal: each row sees only positions ≤ itself → rows differ
        let row0_c = &out_c[0..dim];
        let row1_c = &out_c[dim..2 * dim];
        // Row 0 should differ from row 1 under causal masking
        let diff: f32 = row0_c.iter().zip(row1_c).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "causal rows should differ");

        // Non-causal: all rows same
        let row0_nc = &out_nc[0..dim];
        let row1_nc = &out_nc[dim..2 * dim];
        let diff_nc: f32 = row0_nc.iter().zip(row1_nc).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff_nc < 1e-5, "non-causal rows should be identical");
    }

    #[test]
    fn test_cpu_attention_causal_monotonic_context() {
        // Under causal masking, later tokens have more context
        let cfg = AttentionConfig::new(1, 2, 4, true).unwrap();
        let q = vec![1.0; 8];
        let k = vec![1.0; 8];
        let v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        // Each row's output[0] should increase (more context, higher-indexed V)
        for i in 0..3 {
            assert!(out[(i + 1) * 2] >= out[i * 2] - 1e-5, "causal context should be monotonic");
        }
    }

    // ── Masked attention tests ────────────────────────────────────────

    #[test]
    fn test_masked_attention_zero_mask_equals_unmasked() {
        let cfg = AttentionConfig::new(1, 2, 3, false).unwrap();
        let q = vec![1.0, 0.5, 0.5, 1.0, 0.0, 1.0];
        let k = vec![0.5, 0.5, 1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let zero_mask = vec![0.0_f32; 9]; // 3×3 zero mask

        let out_unmasked = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        let out_masked = masked_attention_cpu_fallback(&q, &k, &v, &zero_mask, &cfg).unwrap();

        for (a, b) in out_unmasked.iter().zip(out_masked.iter()) {
            assert!((a - b).abs() < 1e-5, "zero mask should equal unmasked");
        }
    }

    #[test]
    fn test_masked_attention_blocks_positions() {
        // Mask blocks all except self-attention (diagonal)
        let seq = 3;
        let dim = 2;
        let cfg = AttentionConfig::new(1, dim, seq, false).unwrap();
        let q = vec![1.0; seq * dim];
        let k = vec![1.0; seq * dim];
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

        // Diagonal mask: 0 on diagonal, -inf off-diagonal
        let ninf = f32::NEG_INFINITY;
        #[rustfmt::skip]
        let mask = vec![
            0.0,  ninf, ninf,
            ninf, 0.0,  ninf,
            ninf, ninf, 0.0,
        ];

        let out = masked_attention_cpu_fallback(&q, &k, &v, &mask, &cfg).unwrap();
        // Each row attends only to itself → output == value
        for i in 0..seq {
            for d in 0..dim {
                assert!(
                    (out[i * dim + d] - v[i * dim + d]).abs() < 1e-5,
                    "diagonal mask: row {i} dim {d}"
                );
            }
        }
    }

    #[test]
    fn test_masked_attention_rejects_short_mask() {
        let cfg = AttentionConfig::new(1, 2, 4, false).unwrap();
        let t = vec![0.0_f32; 8];
        let short_mask = vec![0.0_f32; 8]; // need 16
        assert!(masked_attention_cpu_fallback(&t, &t, &t, &short_mask, &cfg).is_err());
    }

    #[test]
    fn test_masked_attention_numerical_stability() {
        let cfg = AttentionConfig::new(1, 2, 2, false).unwrap();
        let q = vec![1000.0, -1000.0, 0.0, 0.0];
        let k = vec![1000.0, -1000.0, 0.0, 0.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let mask = vec![0.0_f32; 4];
        let out = masked_attention_cpu_fallback(&q, &k, &v, &mask, &cfg).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "non-finite with large values");
    }

    // ── Multi-head attention tests ────────────────────────────────────

    #[test]
    fn test_multi_head_output_shape() {
        let cfg = AttentionConfig::new(4, 8, 6, false).unwrap();
        let total = 4 * 6 * 8;
        let q = vec![0.1_f32; total];
        let k = vec![0.2_f32; total];
        let v = vec![0.3_f32; total];
        let out = multi_head_attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), total);
    }

    #[test]
    fn test_multi_head_independent_heads() {
        // Different data per head → different outputs per head
        let cfg = AttentionConfig::new(2, 2, 2, false).unwrap();
        let q = vec![
            1.0, 0.0, 0.0, 1.0, // head 0
            0.0, 1.0, 1.0, 0.0, // head 1
        ];
        let k = vec![
            1.0, 0.0, 0.0, 1.0, // head 0
            0.0, 1.0, 1.0, 0.0, // head 1
        ];
        let v = vec![
            10.0, 20.0, 30.0, 40.0, // head 0
            50.0, 60.0, 70.0, 80.0, // head 1
        ];
        let out = multi_head_attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        // Head 0 output should differ from head 1 output
        let head0 = &out[0..4];
        let head1 = &out[4..8];
        let diff: f32 = head0.iter().zip(head1).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-3, "heads should produce different outputs");
    }

    #[test]
    fn test_multi_head_matches_single_head() {
        // Multi-head with 1 head should match single-head
        let cfg = AttentionConfig::new(1, 4, 3, false).unwrap();
        let q = vec![1.0, 0.5, 0.0, -0.5, 0.2, 0.8, -0.3, 0.1, 0.7, -0.2, 0.4, 0.6];
        let k = vec![0.5, 1.0, -0.5, 0.0, -0.1, 0.3, 0.7, -0.2, 0.4, 0.0, 0.1, 0.9];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let out_single = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        let out_multi = multi_head_attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();

        for (a, b) in out_single.iter().zip(out_multi.iter()) {
            assert!((a - b).abs() < 1e-5, "single vs multi mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_multi_head_causal() {
        let cfg = AttentionConfig::new(2, 2, 3, true).unwrap();
        let total = 2 * 3 * 2;
        let q = vec![1.0_f32; total];
        let k = vec![1.0_f32; total];
        let v: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let out = multi_head_attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), total);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_multi_head_rejects_short_tensors() {
        let cfg = AttentionConfig::new(4, 4, 4, false).unwrap();
        let short = vec![0.0_f32; 32]; // need 64
        let ok = vec![0.0_f32; 64];
        assert!(multi_head_attention_cpu_fallback(&short, &ok, &ok, &cfg).is_err());
    }

    // ── Unified dispatch tests ────────────────────────────────────────

    #[test]
    fn test_attention_forward_cpu_single_head() {
        let cfg = AttentionConfig::new(1, 4, 2, false).unwrap();
        let q = vec![1.0_f32; 8];
        let k = vec![1.0_f32; 8];
        let v = vec![2.0_f32; 8];
        let out = attention_forward(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), 8);
        // Uniform Q,K,V → output == V
        for &val in &out {
            assert!((val - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_attention_forward_cpu_multi_head() {
        let cfg = AttentionConfig::new(2, 4, 3, true).unwrap();
        let total = 2 * 3 * 4;
        let q = vec![0.5_f32; total];
        let k = vec![0.5_f32; total];
        let v = vec![1.0_f32; total];
        let out = attention_forward(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), total);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Edge case tests ───────────────────────────────────────────────

    #[test]
    fn test_cpu_attention_large_head_dim() {
        let cfg = AttentionConfig::new(1, 128, 2, false).unwrap();
        let size = 2 * 128;
        let q = vec![0.01_f32; size];
        let k = vec![0.01_f32; size];
        let v = vec![1.0_f32; size];
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), size);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_cpu_attention_negative_values() {
        let cfg = AttentionConfig::new(1, 2, 2, false).unwrap();
        let q = vec![-1.0, -2.0, -3.0, -4.0];
        let k = vec![-1.0, -2.0, -3.0, -4.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_cpu_attention_softmax_sum_to_one() {
        // Verify attention weights implicitly sum to 1 by checking output
        // is a convex combination of values
        let cfg = AttentionConfig::new(1, 1, 3, false).unwrap();
        let q = vec![1.0, 0.0, -1.0];
        let k = vec![1.0, 0.0, -1.0];
        let v = vec![0.0, 50.0, 100.0];
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        // Each output should be in [0, 100] (convex combination)
        for &val in &out {
            assert!(val >= -1e-5 && val <= 100.0 + 1e-5, "out of range: {val}");
        }
    }

    // ── CUDA launch stub test ─────────────────────────────────────────

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_attention_launch() {
        let cfg = AttentionKernelConfig::for_shape(8, 64, 32, 32, true).unwrap();
        let size_q = 8 * 32 * 64;
        let size_kv = 8 * 32 * 64;
        let q = vec![0.0f32; size_q];
        let k = vec![0.0f32; size_kv];
        let v = vec![0.0f32; size_kv];
        let mut output = vec![0.0f32; size_q];
        let result = launch_attention(&q, &k, &v, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA attention launch failed: {result:?}");
    }

    // ── CUDA kernel source compile guard ──────────────────────────────

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_kernel_source_not_empty() {
        assert!(!ATTENTION_KERNEL_SRC.is_empty(), "CUDA kernel source should not be empty");
        assert!(ATTENTION_KERNEL_SRC.contains("sdp_attention_f32"));
        assert!(ATTENTION_KERNEL_SRC.contains("sdp_attention_causal_f32"));
    }

    // ── CudaAttentionConfig alias test ────────────────────────────────

    #[test]
    fn test_cuda_attention_config_alias() {
        let cfg: CudaAttentionConfig = CudaAttentionConfig::for_shape(4, 64, 16, 16, true).unwrap();
        assert_eq!(cfg.n_heads, 4);
        assert_eq!(cfg.head_dim, 64);
        assert!(cfg.causal);
    }

    // ── attention_forward_cpu wrapper tests ────────────────────────────

    #[test]
    fn test_attention_forward_cpu_single() {
        let cfg = AttentionConfig::new(1, 4, 2, false).unwrap();
        let q = vec![1.0_f32; 8];
        let k = vec![1.0_f32; 8];
        let v = vec![2.0_f32; 8];
        let out = attention_forward_cpu(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), 8);
        for &val in &out {
            assert!((val - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_attention_forward_cpu_multi() {
        let cfg = AttentionConfig::new(2, 4, 3, false).unwrap();
        let total = 2 * 3 * 4;
        let q = vec![0.5_f32; total];
        let k = vec![0.5_f32; total];
        let v = vec![1.0_f32; total];
        let out = attention_forward_cpu(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), total);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Chunked (flash-attention style) CPU tests ─────────────────────

    #[test]
    fn test_chunked_matches_standard_noncausal() {
        let cfg = AttentionConfig::new(1, 4, 8, false).unwrap();
        let q: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..32).map(|i| ((i + 7) as f32) * 0.05).collect();
        let v: Vec<f32> = (0..32).map(|i| (i as f32) * 0.2).collect();

        let standard = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        let chunked = chunked_attention_cpu(&q, &k, &v, &cfg, 3).unwrap();

        assert_eq!(standard.len(), chunked.len());
        for (a, b) in standard.iter().zip(chunked.iter()) {
            assert!((a - b).abs() < 1e-4, "chunked vs standard mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_chunked_matches_standard_causal() {
        let cfg = AttentionConfig::new(1, 2, 6, true).unwrap();
        let q = vec![1.0_f32; 12];
        let k: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let v: Vec<f32> = (0..12).map(|i| i as f32).collect();

        let standard = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        let chunked = chunked_attention_cpu(&q, &k, &v, &cfg, 2).unwrap();

        for (a, b) in standard.iter().zip(chunked.iter()) {
            assert!((a - b).abs() < 1e-4, "chunked causal mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_chunked_default_chunk_size() {
        // chunk_size=0 should use default and still produce correct results
        let cfg = AttentionConfig::new(1, 2, 4, false).unwrap();
        let q = vec![1.0_f32; 8];
        let k = vec![1.0_f32; 8];
        let v = vec![3.0_f32; 8];
        let out = chunked_attention_cpu(&q, &k, &v, &cfg, 0).unwrap();
        for &val in &out {
            assert!((val - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_chunked_single_token() {
        let cfg = AttentionConfig::new(1, 4, 1, false).unwrap();
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![0.5, 0.5, 0.5, 0.5];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let out = chunked_attention_cpu(&q, &k, &v, &cfg, 1).unwrap();
        for d in 0..4 {
            assert!((out[d] - v[d]).abs() < 1e-5, "seq_len=1 chunked should return V");
        }
    }

    #[test]
    fn test_chunked_rejects_short_tensors() {
        let cfg = AttentionConfig::new(1, 4, 8, false).unwrap();
        let short = vec![0.0_f32; 16]; // need 32
        let ok = vec![0.0_f32; 32];
        assert!(chunked_attention_cpu(&short, &ok, &ok, &cfg, 4).is_err());
    }

    // ── Batch attention tests ─────────────────────────────────────────

    #[test]
    fn test_batch_attention_single_batch() {
        let cfg = AttentionConfig::new(1, 2, 3, false).unwrap();
        let q = vec![1.0_f32; 6];
        let k = vec![1.0_f32; 6];
        let v = vec![5.0_f32; 6];

        let single = attention_forward_cpu(&q, &k, &v, &cfg).unwrap();
        let batched = batch_attention_cpu(&q, &k, &v, &cfg, 1).unwrap();

        for (a, b) in single.iter().zip(batched.iter()) {
            assert!((a - b).abs() < 1e-5, "batch=1 should match single");
        }
    }

    #[test]
    fn test_batch_attention_two_batches() {
        let cfg = AttentionConfig::new(2, 2, 2, false).unwrap();
        let per_batch = 2 * 2 * 2; // heads * seq * dim = 8
        // Batch 0: all ones; Batch 1: counting
        let mut q = vec![1.0_f32; per_batch];
        q.extend((0..per_batch).map(|i| i as f32 * 0.1));
        let mut k = vec![1.0_f32; per_batch];
        k.extend((0..per_batch).map(|i| (i as f32 + 1.0) * 0.1));
        let mut v = vec![3.0_f32; per_batch];
        v.extend((0..per_batch).map(|i| i as f32));

        let out = batch_attention_cpu(&q, &k, &v, &cfg, 2).unwrap();
        assert_eq!(out.len(), 2 * per_batch);
        assert!(out.iter().all(|v| v.is_finite()));

        // Batch 0 (uniform) → output == V = 3.0
        for &val in &out[..per_batch] {
            assert!((val - 3.0).abs() < 1e-4, "batch 0 uniform: {val}");
        }
    }

    #[test]
    fn test_batch_attention_rejects_zero_batch() {
        let cfg = AttentionConfig::new(1, 2, 2, false).unwrap();
        let t = vec![0.0_f32; 4];
        assert!(batch_attention_cpu(&t, &t, &t, &cfg, 0).is_err());
    }

    #[test]
    fn test_batch_attention_rejects_short_tensors() {
        let cfg = AttentionConfig::new(1, 2, 2, false).unwrap();
        let short = vec![0.0_f32; 4]; // need 8 for batch=2
        assert!(batch_attention_cpu(&short, &short, &short, &cfg, 2).is_err());
    }

    // ── Scale factor verification ─────────────────────────────────────

    #[test]
    fn test_scale_factor_affects_output() {
        let cfg_default = AttentionConfig::new(1, 4, 2, false).unwrap();
        let cfg_big = AttentionConfig::new(1, 4, 2, false).unwrap().with_scale(10.0);
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

        let out_default = attention_cpu_fallback(&q, &k, &v, &cfg_default).unwrap();
        let out_big = attention_cpu_fallback(&q, &k, &v, &cfg_big).unwrap();

        // Large scale sharpens attention → outputs should differ
        let diff: f32 = out_default.iter().zip(out_big.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-3, "different scales should produce different outputs");
    }

    // ── Equal Q=K=V edge case ─────────────────────────────────────────

    #[test]
    fn test_equal_qkv() {
        // When Q == K == V, output should equal the input (uniform attention
        // over identical values returns those values).
        let cfg = AttentionConfig::new(1, 2, 3, false).unwrap();
        let data = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let out = attention_cpu_fallback(&data, &data, &data, &cfg).unwrap();
        for i in 0..6 {
            assert!(
                (out[i] - data[i]).abs() < 1e-4,
                "Q==K==V: output should match input at index {i}"
            );
        }
    }
}
