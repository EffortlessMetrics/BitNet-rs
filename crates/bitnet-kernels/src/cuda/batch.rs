//! CUDA batched operation kernels with CPU fallback.
//!
//! # Kernel strategy
//!
//! Provides batched operations over contiguous `f32` slices:
//!
//! - **Batched matmul**: Independent matrix multiplications across a batch
//!   dimension: (B × M × K) × (B × K × N) → (B × M × N).
//! - **Batched add**: Element-wise addition across corresponding batch elements.
//! - **Batched scale**: Per-batch-element scalar multiplication.
//! - **Batch norm inference**: Batch normalization using pre-computed statistics.
//! - **Dynamic batching**: Group variable-length sequences into padded batches.
//! - **Unbatch**: Split a batched tensor back into individual tensors.
//! - **Pad to batch**: Pad variable-length inputs to a uniform batch size.
//!
//! # CPU fallback
//!
//! Pure-Rust implementations are provided for correctness testing and
//! non-GPU environments. The `*_forward` functions dispatch to GPU when
//! available, falling back to CPU otherwise.

use bitnet_common::{KernelError, Result};

// ── Batched matmul ────────────────────────────────────────────────────

/// Configuration for batched matrix multiplication.
#[derive(Debug, Clone)]
pub struct BatchedMatmulConfig {
    /// Number of independent matmuls in the batch.
    pub batch_size: usize,
    /// Number of rows in each A matrix.
    pub m: usize,
    /// Number of columns in each B matrix (output columns).
    pub n: usize,
    /// Inner (reduction) dimension.
    pub k: usize,
}

impl BatchedMatmulConfig {
    /// Create a validated configuration.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] when any dimension is zero.
    pub fn new(batch_size: usize, m: usize, n: usize, k: usize) -> Result<Self> {
        if batch_size == 0 || m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "batched matmul dimensions must be non-zero: batch={batch_size}, m={m}, n={n}, k={k}"
                ),
            }
            .into());
        }
        Ok(Self { batch_size, m, n, k })
    }
}

/// CPU fallback for batched matrix multiplication.
///
/// Computes `C[b] = A[b] · B[b]` for each batch element independently.
///
/// # Layout
///
/// - `a`: `[batch_size, m, k]` row-major
/// - `b`: `[batch_size, k, n]` row-major
/// - `output`: `[batch_size, m, n]` row-major
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn batched_matmul(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    config: &BatchedMatmulConfig,
) -> Result<()> {
    let BatchedMatmulConfig { batch_size, m, n, k } = *config;
    let a_stride = m * k;
    let b_stride = k * n;
    let out_stride = m * n;

    if a.len() < batch_size * a_stride {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "A buffer too small: expected >= {}, got {}",
                batch_size * a_stride,
                a.len()
            ),
        }
        .into());
    }
    if b.len() < batch_size * b_stride {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "B buffer too small: expected >= {}, got {}",
                batch_size * b_stride,
                b.len()
            ),
        }
        .into());
    }
    if output.len() < batch_size * out_stride {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "output buffer too small: expected >= {}, got {}",
                batch_size * out_stride,
                output.len()
            ),
        }
        .into());
    }

    for batch in 0..batch_size {
        let a_off = batch * a_stride;
        let b_off = batch * b_stride;
        let o_off = batch * out_stride;

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in 0..k {
                    acc += a[a_off + i * k + l] * b[b_off + l * n + j];
                }
                output[o_off + i * n + j] = acc;
            }
        }
    }
    Ok(())
}

/// Dispatch batched matmul: GPU if available, else CPU fallback.
pub fn batched_matmul_forward(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    config: &BatchedMatmulConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_batched_matmul(a, b, output, config)
        {
            return Ok(());
        }
    }
    batched_matmul(a, b, output, config)
}

/// CUDA launch stub for batched matmul.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_batched_matmul(
    _a: &[f32],
    _b: &[f32],
    _output: &mut [f32],
    config: &BatchedMatmulConfig,
) -> Result<()> {
    log::debug!(
        "batched_matmul CUDA stub: batch={}, m={}, n={}, k={}",
        config.batch_size,
        config.m,
        config.n,
        config.k,
    );
    Err(KernelError::GpuError {
        reason: "batched matmul CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ── Batched add ───────────────────────────────────────────────────────

/// CPU fallback for batched element-wise addition.
///
/// Computes `output[b][i] = a[b][i] + b_data[b][i]` for each batch element.
///
/// - `a`, `b_data`: `[batch_size, elem_count]` row-major
/// - `output`: `[batch_size, elem_count]` row-major
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn batched_add(
    a: &[f32],
    b_data: &[f32],
    output: &mut [f32],
    batch_size: usize,
    elem_count: usize,
) -> Result<()> {
    let total = batch_size * elem_count;
    if a.len() < total || b_data.len() < total || output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "batched_add buffer size mismatch: need {total}, got a={}, b={}, out={}",
                a.len(),
                b_data.len(),
                output.len()
            ),
        }
        .into());
    }
    for i in 0..total {
        output[i] = a[i] + b_data[i];
    }
    Ok(())
}

/// Dispatch batched add: GPU if available, else CPU fallback.
pub fn batched_add_forward(
    a: &[f32],
    b_data: &[f32],
    output: &mut [f32],
    batch_size: usize,
    elem_count: usize,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_batched_add(a, b_data, output, batch_size, elem_count)
        {
            return Ok(());
        }
    }
    batched_add(a, b_data, output, batch_size, elem_count)
}

/// CUDA launch stub for batched add.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_batched_add(
    _a: &[f32],
    _b_data: &[f32],
    _output: &mut [f32],
    _batch_size: usize,
    _elem_count: usize,
) -> Result<()> {
    Err(KernelError::GpuError {
        reason: "batched add CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ── Batched scale ─────────────────────────────────────────────────────

/// CPU fallback for batched scaling.
///
/// Scales each batch element by its corresponding factor:
/// `output[b][i] = input[b][i] * scales[b]`
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn batched_scale(
    input: &[f32],
    scales: &[f32],
    output: &mut [f32],
    batch_size: usize,
    elem_count: usize,
) -> Result<()> {
    let total = batch_size * elem_count;
    if input.len() < total || output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "batched_scale buffer mismatch: need {total}, got input={}, out={}",
                input.len(),
                output.len()
            ),
        }
        .into());
    }
    if scales.len() < batch_size {
        return Err(KernelError::InvalidArguments {
            reason: format!("scales buffer too small: need {batch_size}, got {}", scales.len()),
        }
        .into());
    }
    for (b, &s) in scales.iter().enumerate().take(batch_size) {
        let off = b * elem_count;
        for i in 0..elem_count {
            output[off + i] = input[off + i] * s;
        }
    }
    Ok(())
}

/// Dispatch batched scale: GPU if available, else CPU fallback.
pub fn batched_scale_forward(
    input: &[f32],
    scales: &[f32],
    output: &mut [f32],
    batch_size: usize,
    elem_count: usize,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_batched_scale(input, scales, output, batch_size, elem_count)
        {
            return Ok(());
        }
    }
    batched_scale(input, scales, output, batch_size, elem_count)
}

/// CUDA launch stub for batched scale.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_batched_scale(
    _input: &[f32],
    _scales: &[f32],
    _output: &mut [f32],
    _batch_size: usize,
    _elem_count: usize,
) -> Result<()> {
    Err(KernelError::GpuError {
        reason: "batched scale CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ── Batch norm inference ──────────────────────────────────────────────

/// Configuration for batch normalization (inference mode).
#[derive(Debug, Clone)]
pub struct BatchNormInferenceConfig {
    /// Number of samples in the batch.
    pub batch_size: usize,
    /// Number of features (channels).
    pub num_features: usize,
    /// Small constant for numerical stability.
    pub eps: f32,
}

/// CPU fallback for batch normalization in inference mode.
///
/// Uses pre-computed running statistics:
/// `y[b, c] = (x[b, c] - mean[c]) / sqrt(var[c] + eps) * gamma[c] + beta[c]`
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn batch_norm_inference(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    config: &BatchNormInferenceConfig,
) -> Result<()> {
    let total = config.batch_size * config.num_features;
    if input.len() < total || output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "batch_norm buffer mismatch: need {total}, got input={}, out={}",
                input.len(),
                output.len()
            ),
        }
        .into());
    }
    let nf = config.num_features;
    if gamma.len() < nf || beta.len() < nf || running_mean.len() < nf || running_var.len() < nf {
        return Err(KernelError::InvalidArguments {
            reason: "batch_norm parameter buffer too small".into(),
        }
        .into());
    }

    for b in 0..config.batch_size {
        for c in 0..nf {
            let idx = b * nf + c;
            let inv_std = 1.0 / (running_var[c] + config.eps).sqrt();
            output[idx] = (input[idx] - running_mean[c]) * inv_std * gamma[c] + beta[c];
        }
    }
    Ok(())
}

/// Dispatch batch norm inference: GPU if available, else CPU fallback.
pub fn batch_norm_inference_forward(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    config: &BatchNormInferenceConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_batch_norm_inference(
                input,
                output,
                gamma,
                beta,
                running_mean,
                running_var,
                config,
            )
        {
            return Ok(());
        }
    }
    batch_norm_inference(input, output, gamma, beta, running_mean, running_var, config)
}

/// CUDA launch stub for batch norm inference.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_batch_norm_inference(
    _input: &[f32],
    _output: &mut [f32],
    _gamma: &[f32],
    _beta: &[f32],
    _running_mean: &[f32],
    _running_var: &[f32],
    _config: &BatchNormInferenceConfig,
) -> Result<()> {
    Err(KernelError::GpuError {
        reason: "batch norm inference CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ── Dynamic batching ──────────────────────────────────────────────────

/// Result of dynamic batching: variable-length sequences grouped into
/// padded batches.
#[derive(Debug, Clone)]
pub struct DynamicBatchResult {
    /// The padded batch tensor `[num_batches, max_batch_len, feature_dim]`.
    pub data: Vec<f32>,
    /// Number of resulting batches.
    pub num_batches: usize,
    /// Maximum sequence length per batch.
    pub max_batch_len: usize,
    /// Feature dimension (preserved from input).
    pub feature_dim: usize,
    /// Mapping from original sequence index to (batch_idx, position_in_batch).
    pub batch_assignments: Vec<(usize, usize)>,
    /// Original lengths of sequences in each batch.
    pub batch_lengths: Vec<Vec<usize>>,
}

/// Group variable-length sequences into padded batches for efficient
/// processing.
///
/// Sequences are greedily packed into batches of at most `max_batch_size`
/// elements. Each batch is padded to the longest sequence in that batch.
///
/// # Arguments
///
/// - `sequences`: Slice of variable-length sequences, each `[seq_len, feature_dim]`.
/// - `feature_dim`: Feature dimension of each element.
/// - `max_batch_size`: Maximum number of sequences per batch.
/// - `pad_value`: Value used for padding.
///
/// # Errors
///
/// Returns an error if any sequence has length not divisible by `feature_dim`.
pub fn dynamic_batching(
    sequences: &[&[f32]],
    feature_dim: usize,
    max_batch_size: usize,
    pad_value: f32,
) -> Result<DynamicBatchResult> {
    if feature_dim == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "feature_dim must be > 0".into() }.into()
        );
    }
    if max_batch_size == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "max_batch_size must be > 0".into() }.into()
        );
    }

    // Compute sequence lengths (in tokens, not elements).
    let seq_lengths: Vec<usize> = sequences
        .iter()
        .map(|s| {
            if s.len() % feature_dim != 0 {
                Err(KernelError::InvalidArguments {
                    reason: format!(
                        "sequence length {} not divisible by feature_dim {feature_dim}",
                        s.len()
                    ),
                }
                .into())
            } else {
                Ok(s.len() / feature_dim)
            }
        })
        .collect::<Result<Vec<_>>>()?;

    if sequences.is_empty() {
        return Ok(DynamicBatchResult {
            data: vec![],
            num_batches: 0,
            max_batch_len: 0,
            feature_dim,
            batch_assignments: vec![],
            batch_lengths: vec![],
        });
    }

    // Greedy packing: fill batches in order.
    let mut batch_assignments = vec![(0usize, 0usize); sequences.len()];
    let mut batch_lengths: Vec<Vec<usize>> = vec![];
    let mut current_batch: Vec<usize> = vec![]; // indices into sequences

    for (idx, &len) in seq_lengths.iter().enumerate() {
        if current_batch.len() >= max_batch_size {
            batch_lengths.push(current_batch.iter().map(|&i| seq_lengths[i]).collect());
            current_batch.clear();
        }
        let pos = current_batch.len();
        batch_assignments[idx] = (batch_lengths.len(), pos);
        current_batch.push(idx);
        let _ = len; // used via seq_lengths[idx]
    }
    if !current_batch.is_empty() {
        batch_lengths.push(current_batch.iter().map(|&i| seq_lengths[i]).collect());
    }

    let num_batches = batch_lengths.len();

    // For each batch, find max length.
    let batch_max_lens: Vec<usize> =
        batch_lengths.iter().map(|lens| lens.iter().copied().max().unwrap_or(0)).collect();
    let max_batch_len = batch_max_lens.iter().copied().max().unwrap_or(0);

    // Allocate output: [num_batches, max_batch_size, max_batch_len, feature_dim]
    // Simplified: flatten each batch to [max_batch_size * max_batch_len * feature_dim]
    // Actually, use [num_batches * max_batch_len * feature_dim] per batch slot.
    // We flatten to [total_batch_slots, max_batch_len, feature_dim].
    let total_slots: usize = batch_lengths.iter().map(|b| b.len()).sum();
    let slot_size = max_batch_len * feature_dim;
    let mut data = vec![pad_value; total_slots * slot_size];

    let mut slot_idx = 0;
    for (batch_idx, lens) in batch_lengths.iter().enumerate() {
        for (pos, &seq_len) in lens.iter().enumerate() {
            let orig_idx = sequences
                .iter()
                .enumerate()
                .find(|(i, _)| batch_assignments[*i] == (batch_idx, pos))
                .map(|(i, _)| i)
                .unwrap();
            let dst_off = slot_idx * slot_size;
            let src = sequences[orig_idx];
            let copy_len = seq_len * feature_dim;
            data[dst_off..dst_off + copy_len].copy_from_slice(&src[..copy_len]);
            slot_idx += 1;
        }
    }

    Ok(DynamicBatchResult {
        data,
        num_batches,
        max_batch_len,
        feature_dim,
        batch_assignments,
        batch_lengths,
    })
}

// ── Unbatch ───────────────────────────────────────────────────────────

/// Split a batched tensor back into individual tensors.
///
/// # Layout
///
/// - `batched`: `[batch_size, elem_count]` row-major
///
/// # Errors
///
/// Returns an error if buffer size is inconsistent.
pub fn unbatch(batched: &[f32], batch_size: usize, elem_count: usize) -> Result<Vec<Vec<f32>>> {
    let total = batch_size * elem_count;
    if batched.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("unbatch buffer too small: need {total}, got {}", batched.len()),
        }
        .into());
    }
    let mut result = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let off = b * elem_count;
        result.push(batched[off..off + elem_count].to_vec());
    }
    Ok(result)
}

// ── Pad to batch ──────────────────────────────────────────────────────

/// Pad variable-length inputs to a uniform batch size.
///
/// Each input is padded (or truncated) to `target_len` elements with `pad_value`.
///
/// # Errors
///
/// Returns an error if `target_len` is zero.
pub fn pad_to_batch(inputs: &[&[f32]], target_len: usize, pad_value: f32) -> Result<Vec<f32>> {
    if target_len == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "target_len must be > 0".into() }.into()
        );
    }
    let batch_size = inputs.len();
    let mut output = vec![pad_value; batch_size * target_len];
    for (b, input) in inputs.iter().enumerate() {
        let copy_len = input.len().min(target_len);
        let off = b * target_len;
        output[off..off + copy_len].copy_from_slice(&input[..copy_len]);
    }
    Ok(output)
}

// ── CUDA kernel source (feature-gated) ───────────────────────────────

/// Inline CUDA C source for batched matmul kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const BATCHED_MATMUL_KERNEL_SRC: &str = r#"
extern "C" __global__ void batched_matmul_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int batch_size,
    int m,
    int n,
    int k)
{
    int batch = blockIdx.z;
    int row   = blockIdx.y * blockDim.y + threadIdx.y;
    int col   = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && row < m && col < n) {
        int a_off = batch * m * k;
        int b_off = batch * k * n;
        int o_off = batch * m * n;
        float acc = 0.0f;
        for (int i = 0; i < k; i++) {
            acc += a[a_off + row * k + i] * b[b_off + i * n + col];
        }
        out[o_off + row * n + col] = acc;
    }
}
"#;

/// Inline CUDA C source for batched element-wise add kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const BATCHED_ADD_KERNEL_SRC: &str = r#"
extern "C" __global__ void batched_add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = a[i] + b[i];
    }
}
"#;

/// Inline CUDA C source for batched scale kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const BATCHED_SCALE_KERNEL_SRC: &str = r#"
extern "C" __global__ void batched_scale_f32(
    const float* __restrict__ input,
    const float* __restrict__ scales,
    float* __restrict__ out,
    int batch_size,
    int elem_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * elem_count;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int b = i / elem_count;
        out[i] = input[i] * scales[b];
    }
}
"#;

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y} (tol {tol})");
        }
    }

    // ── batched_matmul tests ──────────────────────────────────────

    #[test]
    fn test_batched_matmul_single_batch_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
        let b = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity
        let mut out = vec![0.0f32; 4];
        let cfg = BatchedMatmulConfig::new(1, 2, 2, 2).unwrap();
        batched_matmul(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &a, 1e-6);
    }

    #[test]
    fn test_batched_matmul_two_batches() {
        // batch 0: [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        // batch 1: [[1,0],[0,1]] × [[9,10],[11,12]] = [[9,10],[11,12]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0];
        let b = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = vec![0.0f32; 8];
        let cfg = BatchedMatmulConfig::new(2, 2, 2, 2).unwrap();
        batched_matmul(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[19.0, 22.0, 43.0, 50.0, 9.0, 10.0, 11.0, 12.0], 1e-6);
    }

    #[test]
    fn test_batched_matmul_rectangular() {
        // (2, 2, 3) × (2, 3, 1) → (2, 2, 1)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let b = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchedMatmulConfig::new(2, 2, 1, 3).unwrap();
        batched_matmul(&a, &b, &mut out, &cfg).unwrap();
        // batch 0: [1+2+3, 4+5+6] = [6, 15]
        // batch 1: [14+16+18, 20+22+24] = [48, 66]
        assert_close(&out, &[6.0, 15.0, 48.0, 66.0], 1e-6);
    }

    #[test]
    fn test_batched_matmul_large_batch() {
        let batch = 8;
        let m = 3;
        let n = 3;
        let k = 3;
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i % 7) as f32).collect();
        let b: Vec<f32> = (0..batch * k * n).map(|i| ((i + 3) % 5) as f32).collect();
        let mut out = vec![0.0f32; batch * m * n];
        let cfg = BatchedMatmulConfig::new(batch, m, n, k).unwrap();
        batched_matmul(&a, &b, &mut out, &cfg).unwrap();
        // Verify against sequential naive matmul.
        for batch_idx in 0..batch {
            let a_off = batch_idx * m * k;
            let b_off = batch_idx * k * n;
            let o_off = batch_idx * m * n;
            for i in 0..m {
                for j in 0..n {
                    let mut expected = 0.0f32;
                    for l in 0..k {
                        expected += a[a_off + i * k + l] * b[b_off + l * n + j];
                    }
                    assert!(
                        (out[o_off + i * n + j] - expected).abs() < 1e-4,
                        "mismatch at batch={batch_idx}, i={i}, j={j}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_batched_matmul_1x1() {
        // Scalar multiply: (B, 1, 1) × (B, 1, 1)
        let a = vec![3.0, 5.0, 7.0];
        let b = vec![2.0, 4.0, 6.0];
        let mut out = vec![0.0f32; 3];
        let cfg = BatchedMatmulConfig::new(3, 1, 1, 1).unwrap();
        batched_matmul(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[6.0, 20.0, 42.0], 1e-6);
    }

    #[test]
    fn test_batched_matmul_zero_dims_error() {
        assert!(BatchedMatmulConfig::new(0, 2, 2, 2).is_err());
        assert!(BatchedMatmulConfig::new(1, 0, 2, 2).is_err());
        assert!(BatchedMatmulConfig::new(1, 2, 0, 2).is_err());
        assert!(BatchedMatmulConfig::new(1, 2, 2, 0).is_err());
    }

    #[test]
    fn test_batched_matmul_buffer_too_small_a() {
        let a = vec![1.0; 3]; // too small for (1, 2, 2)
        let b = vec![1.0; 4];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchedMatmulConfig::new(1, 2, 2, 2).unwrap();
        assert!(batched_matmul(&a, &b, &mut out, &cfg).is_err());
    }

    #[test]
    fn test_batched_matmul_buffer_too_small_b() {
        let a = vec![1.0; 4];
        let b = vec![1.0; 3]; // too small
        let mut out = vec![0.0f32; 4];
        let cfg = BatchedMatmulConfig::new(1, 2, 2, 2).unwrap();
        assert!(batched_matmul(&a, &b, &mut out, &cfg).is_err());
    }

    #[test]
    fn test_batched_matmul_buffer_too_small_out() {
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let mut out = vec![0.0f32; 3]; // too small
        let cfg = BatchedMatmulConfig::new(1, 2, 2, 2).unwrap();
        assert!(batched_matmul(&a, &b, &mut out, &cfg).is_err());
    }

    #[test]
    fn test_batched_matmul_forward_uses_cpu() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchedMatmulConfig::new(1, 2, 2, 2).unwrap();
        batched_matmul_forward(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[5.0, 6.0, 7.0, 8.0], 1e-6);
    }

    #[test]
    fn test_batched_matmul_wide_inner_dim() {
        // (1, 1, 1) with k=4
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 1×4
        let b = vec![1.0, 1.0, 1.0, 1.0]; // 4×1
        let mut out = vec![0.0f32; 1];
        let cfg = BatchedMatmulConfig::new(1, 1, 1, 4).unwrap();
        batched_matmul(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[10.0], 1e-6);
    }

    // ── batched_add tests ─────────────────────────────────────────

    #[test]
    fn test_batched_add_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut out = vec![0.0f32; 6];
        batched_add(&a, &b, &mut out, 2, 3).unwrap();
        assert_close(&out, &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0], 1e-6);
    }

    #[test]
    fn test_batched_add_single_element() {
        let a = vec![3.0];
        let b = vec![7.0];
        let mut out = vec![0.0f32; 1];
        batched_add(&a, &b, &mut out, 1, 1).unwrap();
        assert_close(&out, &[10.0], 1e-6);
    }

    #[test]
    fn test_batched_add_buffer_error() {
        let a = vec![1.0; 3];
        let b = vec![1.0; 6];
        let mut out = vec![0.0f32; 6];
        assert!(batched_add(&a, &b, &mut out, 2, 3).is_err());
    }

    #[test]
    fn test_batched_add_forward_cpu() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let mut out = vec![0.0f32; 2];
        batched_add_forward(&a, &b, &mut out, 1, 2).unwrap();
        assert_close(&out, &[4.0, 6.0], 1e-6);
    }

    #[test]
    fn test_batched_add_zeros() {
        let a = vec![0.0; 4];
        let b = vec![0.0; 4];
        let mut out = vec![0.0f32; 4];
        batched_add(&a, &b, &mut out, 2, 2).unwrap();
        assert_close(&out, &[0.0; 4], 1e-6);
    }

    // ── batched_scale tests ───────────────────────────────────────

    #[test]
    fn test_batched_scale_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let scales = vec![2.0, 0.5];
        let mut out = vec![0.0f32; 6];
        batched_scale(&input, &scales, &mut out, 2, 3).unwrap();
        assert_close(&out, &[2.0, 4.0, 6.0, 2.0, 2.5, 3.0], 1e-6);
    }

    #[test]
    fn test_batched_scale_single() {
        let input = vec![5.0, 10.0];
        let scales = vec![3.0];
        let mut out = vec![0.0f32; 2];
        batched_scale(&input, &scales, &mut out, 1, 2).unwrap();
        assert_close(&out, &[15.0, 30.0], 1e-6);
    }

    #[test]
    fn test_batched_scale_zero_scale() {
        let input = vec![1.0, 2.0, 3.0];
        let scales = vec![0.0];
        let mut out = vec![0.0f32; 3];
        batched_scale(&input, &scales, &mut out, 1, 3).unwrap();
        assert_close(&out, &[0.0, 0.0, 0.0], 1e-6);
    }

    #[test]
    fn test_batched_scale_negative() {
        let input = vec![1.0, 2.0];
        let scales = vec![-1.0];
        let mut out = vec![0.0f32; 2];
        batched_scale(&input, &scales, &mut out, 1, 2).unwrap();
        assert_close(&out, &[-1.0, -2.0], 1e-6);
    }

    #[test]
    fn test_batched_scale_buffer_error() {
        let input = vec![1.0; 4];
        let scales = vec![1.0]; // too few for batch_size=2
        let mut out = vec![0.0f32; 4];
        assert!(batched_scale(&input, &scales, &mut out, 2, 2).is_err());
    }

    #[test]
    fn test_batched_scale_forward_cpu() {
        let input = vec![2.0, 4.0];
        let scales = vec![0.5];
        let mut out = vec![0.0f32; 2];
        batched_scale_forward(&input, &scales, &mut out, 1, 2).unwrap();
        assert_close(&out, &[1.0, 2.0], 1e-6);
    }

    // ── batch_norm_inference tests ────────────────────────────────

    #[test]
    fn test_batch_norm_inference_identity() {
        // mean=0, var=1, gamma=1, beta=0 → identity
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        let mean = vec![0.0, 0.0];
        let var = vec![1.0, 1.0];
        let cfg = BatchNormInferenceConfig { batch_size: 2, num_features: 2, eps: 0.0 };
        batch_norm_inference(&input, &mut output, &gamma, &beta, &mean, &var, &cfg).unwrap();
        assert_close(&output, &input, 1e-6);
    }

    #[test]
    fn test_batch_norm_inference_shift() {
        // mean=0, var=1, gamma=1, beta=5 → input + 5
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0f32; 2];
        let gamma = vec![1.0];
        let beta = vec![5.0];
        let mean = vec![0.0];
        let var = vec![1.0];
        let cfg = BatchNormInferenceConfig { batch_size: 2, num_features: 1, eps: 0.0 };
        batch_norm_inference(&input, &mut output, &gamma, &beta, &mean, &var, &cfg).unwrap();
        assert_close(&output, &[6.0, 7.0], 1e-6);
    }

    #[test]
    fn test_batch_norm_inference_scale_and_shift() {
        // gamma=2, beta=1, mean=1, var=4 → 2*(x-1)/2 + 1 = (x-1) + 1 = x
        let input = vec![3.0, 5.0];
        let mut output = vec![0.0f32; 2];
        let gamma = vec![2.0];
        let beta = vec![1.0];
        let mean = vec![1.0];
        let var = vec![4.0];
        let cfg = BatchNormInferenceConfig { batch_size: 2, num_features: 1, eps: 0.0 };
        batch_norm_inference(&input, &mut output, &gamma, &beta, &mean, &var, &cfg).unwrap();
        // (3-1)/2 * 2 + 1 = 3, (5-1)/2 * 2 + 1 = 5
        assert_close(&output, &[3.0, 5.0], 1e-6);
    }

    #[test]
    fn test_batch_norm_inference_eps() {
        let input = vec![2.0];
        let mut output = vec![0.0f32; 1];
        let gamma = vec![1.0];
        let beta = vec![0.0];
        let mean = vec![0.0];
        let var = vec![0.0]; // zero variance, eps prevents division by zero
        let cfg = BatchNormInferenceConfig { batch_size: 1, num_features: 1, eps: 1e-5 };
        batch_norm_inference(&input, &mut output, &gamma, &beta, &mean, &var, &cfg).unwrap();
        let expected = 2.0 / (1e-5f32).sqrt();
        assert!((output[0] - expected).abs() < 1e-1);
    }

    #[test]
    fn test_batch_norm_inference_buffer_error() {
        let input = vec![1.0]; // too small for batch=2, features=2
        let mut output = vec![0.0f32; 4];
        let cfg = BatchNormInferenceConfig { batch_size: 2, num_features: 2, eps: 1e-5 };
        assert!(
            batch_norm_inference(
                &input,
                &mut output,
                &[1.0, 1.0],
                &[0.0, 0.0],
                &[0.0, 0.0],
                &[1.0, 1.0],
                &cfg
            )
            .is_err()
        );
    }

    #[test]
    fn test_batch_norm_inference_forward_cpu() {
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0f32; 2];
        let cfg = BatchNormInferenceConfig { batch_size: 1, num_features: 2, eps: 0.0 };
        batch_norm_inference_forward(
            &input,
            &mut output,
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[1.0, 1.0],
            &cfg,
        )
        .unwrap();
        assert_close(&output, &[1.0, 2.0], 1e-6);
    }

    #[test]
    fn test_batch_norm_inference_multi_feature() {
        // 2 batches, 3 features each
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0f32; 6];
        let gamma = vec![1.0, 2.0, 0.5];
        let beta = vec![0.0, 0.0, 0.0];
        let mean = vec![0.0, 0.0, 0.0];
        let var = vec![1.0, 1.0, 1.0];
        let cfg = BatchNormInferenceConfig { batch_size: 2, num_features: 3, eps: 0.0 };
        batch_norm_inference(&input, &mut output, &gamma, &beta, &mean, &var, &cfg).unwrap();
        assert_close(&output, &[1.0, 4.0, 1.5, 4.0, 10.0, 3.0], 1e-6);
    }

    // ── dynamic_batching tests ────────────────────────────────────

    #[test]
    fn test_dynamic_batching_single_seq() {
        let seq = vec![1.0, 2.0, 3.0, 4.0];
        let seqs: Vec<&[f32]> = vec![&seq];
        let result = dynamic_batching(&seqs, 2, 4, 0.0).unwrap();
        assert_eq!(result.num_batches, 1);
        assert_eq!(result.batch_assignments[0], (0, 0));
        assert_eq!(result.feature_dim, 2);
    }

    #[test]
    fn test_dynamic_batching_empty() {
        let seqs: Vec<&[f32]> = vec![];
        let result = dynamic_batching(&seqs, 2, 4, 0.0).unwrap();
        assert_eq!(result.num_batches, 0);
        assert!(result.data.is_empty());
    }

    #[test]
    fn test_dynamic_batching_multiple_batches() {
        let s1 = vec![1.0, 2.0];
        let s2 = vec![3.0, 4.0];
        let s3 = vec![5.0, 6.0];
        let seqs: Vec<&[f32]> = vec![&s1, &s2, &s3];
        let result = dynamic_batching(&seqs, 1, 2, 0.0).unwrap();
        // max_batch_size=2, so 2 batches: [s1, s2] and [s3]
        assert_eq!(result.num_batches, 2);
        assert_eq!(result.batch_lengths.len(), 2);
    }

    #[test]
    fn test_dynamic_batching_padding() {
        // Two sequences of different lengths, feature_dim=1
        let s1 = vec![1.0, 2.0, 3.0]; // len 3
        let s2 = vec![4.0]; // len 1
        let seqs: Vec<&[f32]> = vec![&s1, &s2];
        let result = dynamic_batching(&seqs, 1, 2, -1.0).unwrap();
        assert_eq!(result.num_batches, 1);
        assert_eq!(result.max_batch_len, 3);
        // s2 should be padded to length 3 with -1.0
        let slot_size = result.max_batch_len * result.feature_dim;
        // Second slot starts at offset slot_size
        assert_eq!(result.data[slot_size], 4.0);
        assert_eq!(result.data[slot_size + 1], -1.0);
        assert_eq!(result.data[slot_size + 2], -1.0);
    }

    #[test]
    fn test_dynamic_batching_feature_dim_error() {
        let s1 = vec![1.0, 2.0, 3.0]; // 3 not divisible by 2
        let seqs: Vec<&[f32]> = vec![&s1];
        assert!(dynamic_batching(&seqs, 2, 4, 0.0).is_err());
    }

    #[test]
    fn test_dynamic_batching_zero_feature_dim() {
        let seqs: Vec<&[f32]> = vec![];
        assert!(dynamic_batching(&seqs, 0, 4, 0.0).is_err());
    }

    #[test]
    fn test_dynamic_batching_zero_max_batch() {
        let seqs: Vec<&[f32]> = vec![];
        assert!(dynamic_batching(&seqs, 2, 0, 0.0).is_err());
    }

    // ── unbatch tests ─────────────────────────────────────────────

    #[test]
    fn test_unbatch_basic() {
        let batched = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = unbatch(&batched, 2, 3).unwrap();
        assert_eq!(result.len(), 2);
        assert_close(&result[0], &[1.0, 2.0, 3.0], 1e-6);
        assert_close(&result[1], &[4.0, 5.0, 6.0], 1e-6);
    }

    #[test]
    fn test_unbatch_single() {
        let batched = vec![1.0, 2.0];
        let result = unbatch(&batched, 1, 2).unwrap();
        assert_eq!(result.len(), 1);
        assert_close(&result[0], &[1.0, 2.0], 1e-6);
    }

    #[test]
    fn test_unbatch_buffer_error() {
        let batched = vec![1.0; 3];
        assert!(unbatch(&batched, 2, 3).is_err());
    }

    #[test]
    fn test_unbatch_roundtrip() {
        // Batch then unbatch should be identity.
        let tensors: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let flat: Vec<f32> = tensors.iter().flatten().copied().collect();
        let result = unbatch(&flat, 2, 3).unwrap();
        for (i, t) in tensors.iter().enumerate() {
            assert_close(&result[i], t, 1e-6);
        }
    }

    // ── pad_to_batch tests ────────────────────────────────────────

    #[test]
    fn test_pad_to_batch_basic() {
        let s1 = vec![1.0, 2.0];
        let s2 = vec![3.0];
        let inputs: Vec<&[f32]> = vec![&s1, &s2];
        let result = pad_to_batch(&inputs, 3, 0.0).unwrap();
        assert_close(&result, &[1.0, 2.0, 0.0, 3.0, 0.0, 0.0], 1e-6);
    }

    #[test]
    fn test_pad_to_batch_truncate() {
        let s1 = vec![1.0, 2.0, 3.0, 4.0];
        let inputs: Vec<&[f32]> = vec![&s1];
        let result = pad_to_batch(&inputs, 2, 0.0).unwrap();
        assert_close(&result, &[1.0, 2.0], 1e-6);
    }

    #[test]
    fn test_pad_to_batch_exact_fit() {
        let s1 = vec![1.0, 2.0, 3.0];
        let inputs: Vec<&[f32]> = vec![&s1];
        let result = pad_to_batch(&inputs, 3, 0.0).unwrap();
        assert_close(&result, &[1.0, 2.0, 3.0], 1e-6);
    }

    #[test]
    fn test_pad_to_batch_empty_inputs() {
        let inputs: Vec<&[f32]> = vec![];
        let result = pad_to_batch(&inputs, 3, 0.0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_pad_to_batch_zero_target_error() {
        let s1 = vec![1.0];
        let inputs: Vec<&[f32]> = vec![&s1];
        assert!(pad_to_batch(&inputs, 0, 0.0).is_err());
    }

    #[test]
    fn test_pad_to_batch_custom_pad_value() {
        let s1 = vec![1.0];
        let inputs: Vec<&[f32]> = vec![&s1];
        let result = pad_to_batch(&inputs, 3, -1.0).unwrap();
        assert_close(&result, &[1.0, -1.0, -1.0], 1e-6);
    }

    #[test]
    fn test_pad_to_batch_multiple_sequences() {
        let s1 = vec![1.0];
        let s2 = vec![2.0, 3.0];
        let s3 = vec![4.0, 5.0, 6.0];
        let inputs: Vec<&[f32]> = vec![&s1, &s2, &s3];
        let result = pad_to_batch(&inputs, 3, 0.0).unwrap();
        assert_eq!(result.len(), 9);
        assert_close(&result[0..3], &[1.0, 0.0, 0.0], 1e-6);
        assert_close(&result[3..6], &[2.0, 3.0, 0.0], 1e-6);
        assert_close(&result[6..9], &[4.0, 5.0, 6.0], 1e-6);
    }
}
