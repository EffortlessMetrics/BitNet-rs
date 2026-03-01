//! Softmax CUDA kernel with numerically stable computation.
//!
//! # Kernel strategy
//!
//! Row-wise softmax over the logits (vocabulary) dimension using the
//! three-pass stable algorithm:
//!
//! 1. **Row max** — each thread-block cooperatively reduces one row to find
//!    `m = max(x)`, preventing overflow in the exponentiation step.
//! 2. **Shifted exp + sum** — every element is transformed to
//!    `e = exp((x[i] - m) / T)` where `T` is an optional temperature
//!    parameter (default `1.0`).  A parallel reduction accumulates `sum(e)`.
//! 3. **Normalise** — each element is divided by `sum(e)` to yield a valid
//!    probability distribution.
//!
//! One thread-block handles one row.  Grid size equals the batch/sequence
//! dimension (`n_rows`).  For typical vocabulary sizes (32 000 – 128 000) the
//! kernel achieves high memory-bandwidth utilisation on Ampere+.
//!
//! # Online (one-pass) softmax
//!
//! [`online_softmax_cpu`] implements the *online softmax* algorithm
//! (Milakov & Gimelshein, 2018) which fuses the max-finding and
//! exp-sum-accumulation into a single pass.  A running maximum `m` is
//! maintained; whenever a new maximum is encountered the partial sum is
//! rescaled by `exp(m_old − m_new)`.  This avoids a separate max-reduction
//! pass and halves memory traffic for bandwidth-bound workloads while
//! remaining numerically equivalent to the three-pass version.
//!
//! # Enhanced features
//!
//! - **Causal masking** — optional upper-triangular mask that sets future
//!   positions to `−∞` before the softmax, used in autoregressive attention.
//! - **Log-softmax** — returns `log(softmax(x))` directly, avoiding a
//!   separate `log()` pass and improving numerical precision for
//!   cross-entropy losses.
//! - **In-place mode** — writes results back into the input buffer,
//!   reducing memory traffic for bandwidth-bound workloads.
//! - **Batched multi-head attention** — [`batched_softmax_cpu`] operates
//!   over `[batch, n_heads, seq_len, seq_len]` attention score tensors
//!   with optional causal masking per head.
//! - **Backward pass** — [`softmax_backward_cpu`] computes the Jacobian-
//!   vector product `dL/dx = softmax(x) ⊙ (dL/dy − dot(dL/dy, softmax(x)))`
//!   for gradient computation.
//!
//! # CUDA kernel
//!
//! `SOFTMAX_KERNEL_SRC` contains a CUDA C kernel string that uses warp-
//! level `__shfl_xor_sync` intrinsics for intra-warp reductions and
//! shared memory for cross-warp communication.  It supports temperature
//! scaling and causal masking via kernel parameters.
//!
//! # CPU fallback
//!
//! [`softmax_cpu`] provides an equivalent pure-Rust implementation for
//! correctness testing and non-GPU environments.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// CUDA kernel source — warp-level online softmax
// ---------------------------------------------------------------------------

/// CUDA C kernel implementing numerically stable online softmax with
/// warp-level `__shfl_xor_sync` reductions.
///
/// **Algorithm** (per row, one thread-block):
/// 1. Each thread computes a local `(max, exp_sum)` pair over its strided
///    slice of the row using the online update rule.
/// 2. Warp-level butterfly reduction (`__shfl_xor_sync`) merges the 32
///    lanes into a single warp-level `(max, sum)`.
/// 3. The first lane of each warp writes to shared memory; after a
///    `__syncthreads()` barrier the first warp reduces across warps.
/// 4. A final broadcast gives every thread the global `(max, sum)`.
/// 5. Each thread normalises its elements: `exp((x - max) / T) / sum`.
///
/// Supports temperature scaling (`inv_temp`) and causal masking
/// (`causal_mask` flag — positions where `col > row_idx` are treated as
/// `−∞`).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const SOFTMAX_KERNEL_SRC: &str = r#"
// Online softmax with warp-level reductions.
// Grid: (n_rows, 1, 1)   Block: (blockDim.x, 1, 1)

extern "C" __global__ void softmax_online(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int   n_cols,
    float inv_temp,
    int   causal_mask,
    int   log_mode)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int row_off = row * n_cols;

    // --- Phase 1: online max + exp-sum per thread -----------------------
    float local_max = -1e38f;
    float local_sum = 0.0f;

    for (int col = tid; col < n_cols; col += blockDim.x) {
        float val = input[row_off + col];
        if (causal_mask && col > row) val = -1e38f;
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum += expf(val - local_max);
        }
    }

    // --- Phase 2: warp-level butterfly reduction -----------------------
    const unsigned FULL_MASK = 0xFFFFFFFFu;
    for (int offset = 16; offset >= 1; offset >>= 1) {
        float other_max = __shfl_xor_sync(FULL_MASK, local_max, offset);
        float other_sum = __shfl_xor_sync(FULL_MASK, local_sum, offset);
        float new_max = fmaxf(local_max, other_max);
        local_sum = local_sum * expf(local_max - new_max)
                   + other_sum * expf(other_max - new_max);
        local_max = new_max;
    }

    // --- Phase 3: cross-warp reduction via shared memory ---------------
    __shared__ float smem_max[32];
    __shared__ float smem_sum[32];
    const int lane   = tid & 31;
    const int warpId = tid >> 5;

    if (lane == 0) {
        smem_max[warpId] = local_max;
        smem_sum[warpId] = local_sum;
    }
    __syncthreads();

    const int n_warps = (blockDim.x + 31) / 32;
    if (tid < 32) {
        local_max = (tid < n_warps) ? smem_max[tid] : -1e38f;
        local_sum = (tid < n_warps) ? smem_sum[tid] : 0.0f;
        for (int offset = 16; offset >= 1; offset >>= 1) {
            float om = __shfl_xor_sync(FULL_MASK, local_max, offset);
            float os = __shfl_xor_sync(FULL_MASK, local_sum, offset);
            float nm = fmaxf(local_max, om);
            local_sum = local_sum * expf(local_max - nm)
                       + os * expf(om - nm);
            local_max = nm;
        }
    }
    __syncthreads();

    // Broadcast final (max, sum) from lane 0 of warp 0.
    if (tid == 0) {
        smem_max[0] = local_max;
        smem_sum[0] = local_sum;
    }
    __syncthreads();

    const float row_max = smem_max[0];
    const float row_sum = smem_sum[0];
    const float log_sum = logf(row_sum);

    // --- Phase 4: write normalised output ------------------------------
    for (int col = tid; col < n_cols; col += blockDim.x) {
        float val = input[row_off + col];
        if (causal_mask && col > row) {
            output[row_off + col] = log_mode ? -1e38f : 0.0f;
        } else {
            float shifted = (val - row_max) * inv_temp;
            if (log_mode) {
                output[row_off + col] = shifted - log_sum;
            } else {
                output[row_off + col] = expf(shifted) / row_sum;
            }
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Softmax mode
// ---------------------------------------------------------------------------

/// Selects between standard softmax and log-softmax output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxMode {
    /// Standard softmax: `exp(x_i - m) / sum(exp(x_j - m))`.
    Standard,
    /// Log-softmax: `(x_i - m) - log(sum(exp(x_j - m)))`.
    ///
    /// Numerically more stable than computing `log(softmax(x))` in a
    /// separate pass, and preferred for cross-entropy loss computation.
    LogSoftmax,
}

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

/// Launch configuration for the softmax kernel.
#[derive(Debug, Clone)]
pub struct SoftmaxConfig {
    /// Number of columns per row (vocabulary / logits dimension).
    pub n_cols: usize,
    /// Number of rows (batch * sequence length).
    pub n_rows: usize,
    /// Threads per block — typically `min(n_cols, 1024)`.
    pub threads_per_block: u32,
    /// Temperature scaling factor applied before exponentiation.
    /// Values `> 1.0` soften the distribution; values in `(0, 1)`
    /// sharpen it.
    pub temperature: f32,
    /// When `true`, applies an upper-triangular causal mask before the
    /// softmax.  Positions where `col > row` are set to negative
    /// infinity so that each token can only attend to itself and
    /// earlier positions.
    ///
    /// Only meaningful when `n_cols == n_rows` (square attention
    /// matrix); for non-square shapes the mask is applied to columns
    /// beyond the current row index.
    pub causal_mask: bool,
    /// Output mode — standard probabilities or log-probabilities.
    pub mode: SoftmaxMode,
    /// When `true`, the CPU fallback writes results back into the input
    /// buffer and ignores the output slice, reducing memory traffic.
    pub in_place: bool,
}

impl SoftmaxConfig {
    /// Create a configuration for the given shape with `temperature = 1.0`.
    pub fn for_shape(n_cols: usize, n_rows: usize) -> Result<Self> {
        if n_cols == 0 || n_rows == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "softmax dimensions must be non-zero: \
                     n_cols={n_cols}, n_rows={n_rows}"
                ),
            }
            .into());
        }

        let threads_per_block = (n_cols as u32).min(1024);

        Ok(Self {
            n_cols,
            n_rows,
            threads_per_block,
            temperature: 1.0,
            causal_mask: false,
            mode: SoftmaxMode::Standard,
            in_place: false,
        })
    }

    /// Override the temperature value (default `1.0`).
    ///
    /// # Errors
    ///
    /// Returns an error if `temperature` is not positive and finite.
    pub fn with_temperature(mut self, temperature: f32) -> Result<Self> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "softmax temperature must be positive \
                     and finite, got {temperature}"
                ),
            }
            .into());
        }
        self.temperature = temperature;
        Ok(self)
    }

    /// Enable causal (upper-triangular) masking.
    ///
    /// Positions where `col > row` are set to negative infinity before
    /// the softmax so each token attends only to itself and earlier
    /// positions.
    pub fn with_causal_mask(mut self) -> Self {
        self.causal_mask = true;
        self
    }

    /// Switch to log-softmax output mode.
    pub fn with_log_softmax(mut self) -> Self {
        self.mode = SoftmaxMode::LogSoftmax;
        self
    }

    /// Enable in-place operation — results are written back into the
    /// input buffer and the output slice is unused.
    pub fn with_in_place(mut self) -> Self {
        self.in_place = true;
        self
    }

    /// Compute the CUDA grid dimensions `(n_rows, 1, 1)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        (self.n_rows as u32, 1, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// Batched multi-head attention softmax config
// ---------------------------------------------------------------------------

/// Configuration for batched softmax over multi-head attention scores.
///
/// The input tensor is `[batch, n_heads, seq_len, seq_len]` (row-major).
/// Each `(batch, head)` slice is an independent `[seq_len, seq_len]`
/// attention matrix that is softmax-normalised row-wise.
#[derive(Debug, Clone)]
pub struct BatchedSoftmaxConfig {
    /// Batch size.
    pub batch_size: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Sequence length (both query and key dimensions).
    pub seq_len: usize,
    /// Temperature scaling.
    pub temperature: f32,
    /// Apply causal mask to every head.
    pub causal_mask: bool,
    /// Output mode.
    pub mode: SoftmaxMode,
}

impl BatchedSoftmaxConfig {
    /// Create a batched config with defaults (`temperature = 1.0`, no
    /// causal mask, standard mode).
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn new(batch_size: usize, n_heads: usize, seq_len: usize) -> Result<Self> {
        if batch_size == 0 || n_heads == 0 || seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "batched softmax dimensions must be non-zero: \
                     batch={batch_size}, heads={n_heads}, \
                     seq_len={seq_len}"
                ),
            }
            .into());
        }
        Ok(Self {
            batch_size,
            n_heads,
            seq_len,
            temperature: 1.0,
            causal_mask: false,
            mode: SoftmaxMode::Standard,
        })
    }

    /// Override the temperature value.
    ///
    /// # Errors
    ///
    /// Returns an error if `temperature` is not positive and finite.
    pub fn with_temperature(mut self, temperature: f32) -> Result<Self> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "batched softmax temperature must be positive \
                     and finite, got {temperature}"
                ),
            }
            .into());
        }
        self.temperature = temperature;
        Ok(self)
    }

    /// Enable causal masking for all heads.
    pub fn with_causal_mask(mut self) -> Self {
        self.causal_mask = true;
        self
    }

    /// Switch to log-softmax output.
    pub fn with_log_softmax(mut self) -> Self {
        self.mode = SoftmaxMode::LogSoftmax;
        self
    }

    /// Total number of elements in the tensor.
    pub fn total_elements(&self) -> usize {
        self.batch_size * self.n_heads * self.seq_len * self.seq_len
    }
}

// ---------------------------------------------------------------------------
// CPU fallback — core row-wise softmax
// ---------------------------------------------------------------------------

/// Numerically stable row-wise softmax on the CPU.
///
/// Computes `softmax(input / temperature)` for each row independently.
/// Supports causal masking, log-softmax mode, and in-place operation.
///
/// # Arguments
///
/// * `input`  — Input logits `[n_rows, n_cols]` (FP32, row-major)
/// * `output` — Output probabilities `[n_rows, n_cols]` (FP32, row-major,
///   written).  Ignored when `config.in_place` is `true`.
/// * `config` — Configuration (uses `n_rows`, `n_cols`, `temperature`,
///   `causal_mask`, `mode`, `in_place`)
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if the slice lengths do not
/// match `n_rows * n_cols`.
pub fn softmax_cpu(input: &[f32], output: &mut [f32], config: &SoftmaxConfig) -> Result<()> {
    let total = config.n_rows * config.n_cols;
    if input.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("softmax input length {} < expected {}", input.len(), total),
        }
        .into());
    }
    if !config.in_place && output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("softmax output length {} < expected {}", output.len(), total),
        }
        .into());
    }

    let inv_temp = 1.0_f32 / config.temperature;

    for row in 0..config.n_rows {
        let start = row * config.n_cols;
        let end = start + config.n_cols;
        let row_in = &input[start..end];

        // --- Pass 1: find row max (after masking) ---
        let row_max = row_in
            .iter()
            .enumerate()
            .map(|(col, &v)| if config.causal_mask && col > row { f32::NEG_INFINITY } else { v })
            .fold(f32::NEG_INFINITY, f32::max);

        // --- Pass 2: shifted exp + sum ---
        let mut sum = 0.0_f32;

        if config.in_place {
            // In-place: write back into the input buffer via raw
            // pointer.  This is safe because we process each element
            // exactly once and never re-read a written position
            // within the same pass.
            let ptr = input.as_ptr() as *mut f32;
            for col in 0..config.n_cols {
                let val = if config.causal_mask && col > row {
                    0.0_f32
                } else {
                    // SAFETY: col < n_cols <= input.len()
                    let x = unsafe { *ptr.add(start + col) };
                    ((x - row_max) * inv_temp).exp()
                };
                unsafe { *ptr.add(start + col) = val };
                sum += val;
            }
            // --- Pass 3: normalise ---
            let log_sum = sum.ln();
            if sum > 0.0 {
                let inv_sum = 1.0 / sum;
                for col in 0..config.n_cols {
                    let p = unsafe { *ptr.add(start + col) };
                    let out_val = match config.mode {
                        SoftmaxMode::Standard => p * inv_sum,
                        SoftmaxMode::LogSoftmax => {
                            if p == 0.0 {
                                f32::NEG_INFINITY
                            } else {
                                p.ln() - log_sum
                            }
                        }
                    };
                    unsafe { *ptr.add(start + col) = out_val };
                }
            }
        } else {
            let row_out = &mut output[start..end];
            for (col, (out, &x)) in row_out.iter_mut().zip(row_in.iter()).enumerate() {
                if config.causal_mask && col > row {
                    *out = 0.0;
                } else {
                    let e = ((x - row_max) * inv_temp).exp();
                    *out = e;
                    sum += e;
                }
            }
            // --- Pass 3: normalise ---
            let log_sum = sum.ln();
            if sum > 0.0 {
                let inv_sum = 1.0 / sum;
                for (col, val) in row_out.iter_mut().enumerate() {
                    match config.mode {
                        SoftmaxMode::Standard => *val *= inv_sum,
                        SoftmaxMode::LogSoftmax => {
                            if (config.causal_mask && col > row) || *val == 0.0 {
                                *val = f32::NEG_INFINITY;
                            } else {
                                *val = val.ln() - log_sum;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback — in-place convenience wrapper
// ---------------------------------------------------------------------------

/// In-place softmax: reads from and writes to the same buffer.
///
/// Equivalent to calling [`softmax_cpu`] with `config.in_place = true`.
///
/// # Errors
///
/// Propagates errors from [`softmax_cpu`].
pub fn softmax_cpu_inplace(data: &mut [f32], config: &SoftmaxConfig) -> Result<()> {
    let mut cfg = config.clone();
    cfg.in_place = true;
    // output slice is unused in in-place mode; pass an empty slice.
    softmax_cpu(data, &mut [], &cfg)
}

// ---------------------------------------------------------------------------
// CPU fallback — batched multi-head attention softmax
// ---------------------------------------------------------------------------

/// Batched softmax over `[batch, n_heads, seq_len, seq_len]` attention
/// scores.
///
/// Each `(batch, head)` slice is an independent `[seq_len, seq_len]`
/// matrix that is softmax-normalised row-wise with the given
/// configuration (temperature, causal mask, mode).
///
/// # Arguments
///
/// * `input`  — Attention scores (row-major, FP32)
/// * `output` — Normalised attention weights (row-major, FP32, written)
/// * `config` — Batched configuration
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if slices are too small.
pub fn batched_softmax_cpu(
    input: &[f32],
    output: &mut [f32],
    config: &BatchedSoftmaxConfig,
) -> Result<()> {
    let total = config.total_elements();
    if input.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("batched softmax input length {} < expected {}", input.len(), total),
        }
        .into());
    }
    if output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("batched softmax output length {} < expected {}", output.len(), total),
        }
        .into());
    }

    let slice_size = config.seq_len * config.seq_len;
    let per_row_cfg = SoftmaxConfig {
        n_cols: config.seq_len,
        n_rows: config.seq_len,
        threads_per_block: (config.seq_len as u32).min(1024),
        temperature: config.temperature,
        causal_mask: config.causal_mask,
        mode: config.mode,
        in_place: false,
    };

    for i in 0..(config.batch_size * config.n_heads) {
        let off = i * slice_size;
        softmax_cpu(
            &input[off..off + slice_size],
            &mut output[off..off + slice_size],
            &per_row_cfg,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback — online (one-pass) softmax
// ---------------------------------------------------------------------------

/// One-pass *online* softmax on the CPU (Milakov & Gimelshein, 2018).
///
/// Instead of a separate max-reduction pass, this algorithm maintains a
/// running maximum `m` and rescales the partial exponential sum whenever a
/// larger element is encountered:
///
/// ```text
/// for x_i in row:
///     if x_i > m:
///         sum = sum * exp(m_old − x_i) + 1
///         m   = x_i
///     else:
///         sum = sum + exp(x_i − m)
/// ```
///
/// The result is numerically equivalent to the three-pass [`softmax_cpu`]
/// but touches memory only twice (read + write) instead of three times.
///
/// Supports temperature scaling and causal masking identical to
/// [`softmax_cpu`].
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if slice lengths are
/// inconsistent with `config`.
pub fn online_softmax_cpu(input: &[f32], output: &mut [f32], config: &SoftmaxConfig) -> Result<()> {
    let total = config.n_rows * config.n_cols;
    if input.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("online softmax input length {} < expected {}", input.len(), total),
        }
        .into());
    }
    if output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("online softmax output length {} < expected {}", output.len(), total),
        }
        .into());
    }

    let inv_temp = 1.0_f32 / config.temperature;

    for row in 0..config.n_rows {
        let start = row * config.n_cols;
        let row_in = &input[start..start + config.n_cols];
        let row_out = &mut output[start..start + config.n_cols];

        // --- Single-pass: running (max, exp-sum) on scaled values ---
        // We work on u_i = x_i * inv_temp so that the online accumulation
        // is equivalent to the three-pass version with temperature.
        let mut m = f32::NEG_INFINITY;
        let mut s = 0.0_f32;

        for (col, &x) in row_in.iter().enumerate() {
            let v = if config.causal_mask && col > row { f32::NEG_INFINITY } else { x * inv_temp };
            if v > m {
                s = s * (m - v).exp() + 1.0;
                m = v;
            } else {
                s += (v - m).exp();
            }
        }

        // --- Write normalised output ---
        let log_sum = s.ln();
        for (col, (&x, out)) in row_in.iter().zip(row_out.iter_mut()).enumerate() {
            if config.causal_mask && col > row {
                *out = match config.mode {
                    SoftmaxMode::Standard => 0.0,
                    SoftmaxMode::LogSoftmax => f32::NEG_INFINITY,
                };
            } else {
                let u = x * inv_temp;
                *out = match config.mode {
                    SoftmaxMode::Standard => (u - m).exp() / s,
                    SoftmaxMode::LogSoftmax => (u - m) - log_sum,
                };
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback — softmax backward (Jacobian-vector product)
// ---------------------------------------------------------------------------

/// Compute the backward pass of softmax: `dL/dx = y ⊙ (dL/dy − dot(dL/dy, y))`
/// where `y = softmax(x)`.
///
/// This is the Jacobian-vector product of the softmax with respect to its
/// input, given upstream gradients `grad_output` and the forward-pass
/// output `softmax_output`.
///
/// Each row is processed independently.
///
/// # Arguments
///
/// * `softmax_output` — Forward-pass softmax output `[n_rows, n_cols]`
/// * `grad_output`    — Upstream gradient `dL/dy` `[n_rows, n_cols]`
/// * `grad_input`     — Output gradient `dL/dx` `[n_rows, n_cols]` (written)
/// * `n_rows`         — Number of rows
/// * `n_cols`         — Number of columns per row
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if any slice is too short.
pub fn softmax_backward_cpu(
    softmax_output: &[f32],
    grad_output: &[f32],
    grad_input: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) -> Result<()> {
    let total = n_rows * n_cols;
    if softmax_output.len() < total || grad_output.len() < total || grad_input.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "softmax_backward: need at least {total} elements, got \
                 softmax_output={}, grad_output={}, grad_input={}",
                softmax_output.len(),
                grad_output.len(),
                grad_input.len()
            ),
        }
        .into());
    }

    for row in 0..n_rows {
        let off = row * n_cols;
        let y = &softmax_output[off..off + n_cols];
        let dy = &grad_output[off..off + n_cols];
        let dx = &mut grad_input[off..off + n_cols];

        // dot = sum_j (y_j * dy_j)
        let dot: f32 = y.iter().zip(dy.iter()).map(|(&yi, &dyi)| yi * dyi).sum();

        // dx_i = y_i * (dy_i - dot)
        for ((dxi, &yi), &dyi) in dx.iter_mut().zip(y.iter()).zip(dy.iter()) {
            *dxi = yi * (dyi - dot);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA launch stub
// ---------------------------------------------------------------------------

/// Launch stub for the softmax CUDA kernel.
///
/// # Arguments
///
/// * `input`  — Input logits `[n_rows, n_cols]` (FP32)
/// * `output` — Output probabilities `[n_rows, n_cols]` (FP32, written)
/// * `config` — Launch configuration
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled and
/// loaded.
pub fn launch_softmax(_input: &[f32], _output: &mut [f32], config: &SoftmaxConfig) -> Result<()> {
    log::debug!(
        "softmax stub: n_cols={}, n_rows={}, temperature={}, \
         causal={}, mode={:?}, grid={:?}",
        config.n_cols,
        config.n_rows,
        config.temperature,
        config.causal_mask,
        config.mode,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "softmax CUDA kernel not yet compiled \
                 — scaffold only"
            .into(),
    }
    .into())
}

/// Launch log-softmax CUDA kernel stub.
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_log_softmax(input: &[f32], output: &mut [f32], config: &SoftmaxConfig) -> Result<()> {
    let mut cfg = config.clone();
    cfg.mode = SoftmaxMode::LogSoftmax;
    launch_softmax(input, output, &cfg)
}

/// Convenience CPU fallback returning an allocated `Vec<f32>`.
pub fn softmax_cpu_fallback(
    input: &[f32],
    rows: usize,
    cols: usize,
    config: &SoftmaxConfig,
) -> Vec<f32> {
    let mut output = vec![0.0f32; rows * cols];
    let _ = softmax_cpu(input, &mut output, config);
    output
}

// ---------------------------------------------------------------------------
// Unified dispatch
// ---------------------------------------------------------------------------

/// Apply softmax with automatic dispatch: GPU if available, else CPU
/// fallback.
///
/// # Arguments
///
/// * `input`  — Input logits `[n_rows, n_cols]` (FP32, row-major)
/// * `output` — Output probabilities `[n_rows, n_cols]` (FP32, row-major,
///   written)
/// * `config` — Launch configuration
pub fn softmax_forward(input: &[f32], output: &mut [f32], config: &SoftmaxConfig) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_softmax(input, output, config)
        {
            return Ok(());
        }
        // GPU launch failed — fall through to CPU path
    }
    softmax_cpu(input, output, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    // == Config tests ====================================================

    #[test]
    fn test_softmax_config_for_shape() {
        let cfg = SoftmaxConfig::for_shape(32000, 1).unwrap();
        assert_eq!(cfg.n_cols, 32000);
        assert_eq!(cfg.n_rows, 1);
        assert_eq!(cfg.threads_per_block, 1024); // capped
        assert!((cfg.temperature - 1.0).abs() < f32::EPSILON);
        assert!(!cfg.causal_mask);
        assert_eq!(cfg.mode, SoftmaxMode::Standard);
        assert!(!cfg.in_place);
    }

    #[test]
    fn test_softmax_config_small_vocab() {
        let cfg = SoftmaxConfig::for_shape(64, 10).unwrap();
        assert_eq!(cfg.threads_per_block, 64);
        assert_eq!(cfg.grid_dim(), (10, 1, 1));
        assert_eq!(cfg.block_dim(), (64, 1, 1));
    }

    #[test]
    fn test_softmax_config_rejects_zero() {
        assert!(SoftmaxConfig::for_shape(0, 1).is_err());
        assert!(SoftmaxConfig::for_shape(32000, 0).is_err());
    }

    #[test]
    fn test_softmax_config_with_temperature() {
        let cfg = SoftmaxConfig::for_shape(128, 4).unwrap().with_temperature(0.7).unwrap();
        assert!((cfg.temperature - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_config_rejects_bad_temperature() {
        let cfg = SoftmaxConfig::for_shape(128, 4).unwrap();
        assert!(cfg.clone().with_temperature(0.0).is_err());
        assert!(cfg.clone().with_temperature(-1.0).is_err());
        assert!(cfg.clone().with_temperature(f32::NAN).is_err());
        assert!(cfg.with_temperature(f32::INFINITY).is_err());
    }

    #[test]
    fn test_softmax_grid_dim() {
        let cfg = SoftmaxConfig::for_shape(32000, 32).unwrap();
        assert_eq!(cfg.grid_dim(), (32, 1, 1));
        assert_eq!(cfg.block_dim(), (1024, 1, 1));
    }

    #[test]
    fn test_softmax_config_builder_chain() {
        let cfg = SoftmaxConfig::for_shape(64, 8)
            .unwrap()
            .with_temperature(0.5)
            .unwrap()
            .with_causal_mask()
            .with_log_softmax()
            .with_in_place();
        assert!((cfg.temperature - 0.5).abs() < 1e-6);
        assert!(cfg.causal_mask);
        assert_eq!(cfg.mode, SoftmaxMode::LogSoftmax);
        assert!(cfg.in_place);
    }

    // == CPU fallback tests ==============================================

    #[test]
    fn test_cpu_softmax_single_row() {
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");

        for i in 0..3 {
            assert!(output[i] < output[i + 1], "output not monotonic at {i}");
        }
    }

    #[test]
    fn test_cpu_softmax_multiple_rows() {
        let cfg = SoftmaxConfig::for_shape(3, 2).unwrap();
        let input = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let mut output = [0.0_f32; 6];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        let sum_row0: f32 = output[0..3].iter().sum();
        let sum_row1: f32 = output[3..6].iter().sum();
        assert!((sum_row0 - 1.0).abs() < 1e-6, "row0 sum={sum_row0}");
        assert!((sum_row1 - 1.0).abs() < 1e-6, "row1 sum={sum_row1}");
    }

    #[test]
    fn test_cpu_softmax_numerical_stability() {
        let cfg = SoftmaxConfig::for_shape(3, 1).unwrap();
        let input = [1000.0_f32, 1001.0, 1002.0];
        let mut output = [0.0_f32; 3];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
        assert!(output.iter().all(|&v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn test_cpu_softmax_uniform_input() {
        let cfg = SoftmaxConfig::for_shape(5, 1).unwrap();
        let input = [3.0_f32; 5];
        let mut output = [0.0_f32; 5];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        for &v in &output {
            assert!((v - 0.2).abs() < 1e-6, "expected 0.2, got {v}");
        }
    }

    #[test]
    fn test_cpu_softmax_with_temperature() {
        let n_cols = 4;
        let input = [1.0_f32, 2.0, 3.0, 4.0];

        let cfg_t1 = SoftmaxConfig::for_shape(n_cols, 1).unwrap();
        let mut out_t1 = [0.0_f32; 4];
        softmax_cpu(&input, &mut out_t1, &cfg_t1).unwrap();

        let cfg_hot = SoftmaxConfig::for_shape(n_cols, 1).unwrap().with_temperature(10.0).unwrap();
        let mut out_hot = [0.0_f32; 4];
        softmax_cpu(&input, &mut out_hot, &cfg_hot).unwrap();

        let cfg_cold = SoftmaxConfig::for_shape(n_cols, 1).unwrap().with_temperature(0.1).unwrap();
        let mut out_cold = [0.0_f32; 4];
        softmax_cpu(&input, &mut out_cold, &cfg_cold).unwrap();

        let max_hot = out_hot.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_t1 = out_t1.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_cold = out_cold.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_hot < max_t1, "high temp should be flatter: {max_hot} >= {max_t1}");
        assert!(max_t1 < max_cold, "low temp should be peakier: {max_t1} >= {max_cold}");
    }

    #[test]
    fn test_cpu_softmax_single_element() {
        let cfg = SoftmaxConfig::for_shape(1, 1).unwrap();
        let input = [42.0_f32];
        let mut output = [0.0_f32; 1];
        softmax_cpu(&input, &mut output, &cfg).unwrap();
        assert!((output[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_softmax_negative_values() {
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap();
        let input = [-10.0_f32, -5.0, 0.0, 5.0];
        let mut output = [0.0_f32; 4];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
        assert!(output.iter().all(|&v| v.is_finite()), "non-finite output");
        for i in 0..3 {
            assert!(output[i] < output[i + 1], "output not monotonic at {i}");
        }
    }

    #[test]
    fn test_cpu_softmax_very_large_values() {
        let cfg = SoftmaxConfig::for_shape(3, 1).unwrap();
        let input = [f32::MAX / 2.0, f32::MAX / 2.0 - 1.0, f32::MAX / 2.0 - 2.0];
        let mut output = [0.0_f32; 3];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        assert!(output.iter().all(|&v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn test_cpu_softmax_temperature_preserves_sum() {
        let input = [1.0_f32, 3.0, 5.0, 2.0, 4.0];
        for temp in [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0] {
            let cfg = SoftmaxConfig::for_shape(5, 1).unwrap().with_temperature(temp).unwrap();
            let mut output = [0.0_f32; 5];
            softmax_cpu(&input, &mut output, &cfg).unwrap();

            let sum: f32 = output.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "temp={temp}: sum={sum}");
        }
    }

    #[test]
    fn test_cpu_softmax_rejects_short_input() {
        let cfg = SoftmaxConfig::for_shape(4, 2).unwrap();
        let input = [1.0_f32; 4]; // need 8
        let mut output = [0.0_f32; 8];
        assert!(softmax_cpu(&input, &mut output, &cfg).is_err());
    }

    #[test]
    fn test_cpu_softmax_rejects_short_output() {
        let cfg = SoftmaxConfig::for_shape(4, 2).unwrap();
        let input = [1.0_f32; 8];
        let mut output = [0.0_f32; 4]; // need 8
        assert!(softmax_cpu(&input, &mut output, &cfg).is_err());
    }

    // == Causal mask tests ===============================================

    #[test]
    fn test_cpu_softmax_causal_mask_identity_row0() {
        // Row 0: only column 0 is visible
        let cfg = SoftmaxConfig::for_shape(3, 3).unwrap().with_causal_mask();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut output = [0.0_f32; 9];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        assert!((output[0] - 1.0).abs() < 1e-6, "row0 col0 should be 1.0, got {}", output[0]);
        assert!(output[1].abs() < 1e-6, "row0 col1 should be 0, got {}", output[1]);
        assert!(output[2].abs() < 1e-6, "row0 col2 should be 0, got {}", output[2]);
    }

    #[test]
    fn test_cpu_softmax_causal_mask_last_row_full() {
        let n = 4;
        let cfg = SoftmaxConfig::for_shape(n, n).unwrap().with_causal_mask();
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut output = vec![0.0_f32; 16];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        let sum_last: f32 = output[12..16].iter().sum();
        assert!((sum_last - 1.0).abs() < 1e-6, "last row sum={sum_last}");
        assert!(output[12..16].iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_cpu_softmax_causal_mask_rows_sum_to_one() {
        let n = 5;
        let cfg = SoftmaxConfig::for_shape(n, n).unwrap().with_causal_mask();
        let input: Vec<f32> = (0..25).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0_f32; 25];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        for row in 0..n {
            let start = row * n;
            let sum: f32 = output[start..start + n].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {row} sum={sum}");
            for col in (row + 1)..n {
                assert!(
                    output[start + col].abs() < 1e-7,
                    "row={row} col={col} should be masked, \
                     got {}",
                    output[start + col]
                );
            }
        }
    }

    // == Log-softmax tests ===============================================

    #[test]
    fn test_cpu_log_softmax_basic() {
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap().with_log_softmax();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        assert!(output.iter().all(|&v| v <= 0.0));

        let sum: f32 = output.iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5, "exp sum={sum}");

        for i in 0..3 {
            assert!(output[i] < output[i + 1], "log-softmax not monotonic at {i}");
        }
    }

    #[test]
    fn test_cpu_log_softmax_matches_log_of_softmax() {
        let input = [0.5_f32, 1.5, -0.3, 2.1, 0.0];

        let cfg_std = SoftmaxConfig::for_shape(5, 1).unwrap();
        let mut out_std = [0.0_f32; 5];
        softmax_cpu(&input, &mut out_std, &cfg_std).unwrap();
        let log_of_std: Vec<f32> = out_std.iter().map(|v| v.ln()).collect();

        let cfg_log = SoftmaxConfig::for_shape(5, 1).unwrap().with_log_softmax();
        let mut out_log = [0.0_f32; 5];
        softmax_cpu(&input, &mut out_log, &cfg_log).unwrap();

        for (i, (&ls, &log_s)) in out_log.iter().zip(log_of_std.iter()).enumerate() {
            assert!(
                (ls - log_s).abs() < 1e-5,
                "log-softmax mismatch at {i}: direct={ls}, \
                 log(softmax)={log_s}"
            );
        }
    }

    #[test]
    fn test_cpu_log_softmax_numerical_stability() {
        let cfg = SoftmaxConfig::for_shape(3, 1).unwrap().with_log_softmax();
        let input = [1000.0_f32, 1001.0, 1002.0];
        let mut output = [0.0_f32; 3];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        assert!(output.iter().all(|&v| v.is_finite()), "log-softmax produced non-finite values");
        let sum: f32 = output.iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5, "exp sum={sum}");
    }

    #[test]
    fn test_cpu_log_softmax_with_causal_mask() {
        let n = 3;
        let cfg = SoftmaxConfig::for_shape(n, n).unwrap().with_causal_mask().with_log_softmax();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut output = [0.0_f32; 9];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        // Row 0: only col 0 visible => log(1.0) = 0.0
        assert!(output[0].abs() < 1e-6, "row0 col0 log-prob should be 0, got {}", output[0]);
        // Masked positions => -inf
        assert!(output[1] == f32::NEG_INFINITY, "row0 col1 should be -inf, got {}", output[1]);

        for row in 0..n {
            let start = row * n;
            let sum: f32 = output[start..start + row + 1].iter().map(|&v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {row} exp-sum={sum}");
        }
    }

    // == In-place tests ==================================================

    #[test]
    fn test_cpu_softmax_inplace() {
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap();
        let input_orig = [1.0_f32, 2.0, 3.0, 4.0];

        let mut out_ref = [0.0_f32; 4];
        softmax_cpu(&input_orig, &mut out_ref, &cfg).unwrap();

        let mut data = input_orig;
        softmax_cpu_inplace(&mut data, &cfg).unwrap();

        for (i, (&ip, &oop)) in data.iter().zip(out_ref.iter()).enumerate() {
            assert!((ip - oop).abs() < 1e-6, "in-place mismatch at {i}: {ip} vs {oop}");
        }
    }

    #[test]
    fn test_cpu_softmax_inplace_multi_row() {
        let cfg = SoftmaxConfig::for_shape(3, 2).unwrap();
        let input_orig = [1.0_f32, 2.0, 3.0, 10.0, 20.0, 30.0];

        let mut out_ref = [0.0_f32; 6];
        softmax_cpu(&input_orig, &mut out_ref, &cfg).unwrap();

        let mut data = input_orig;
        softmax_cpu_inplace(&mut data, &cfg).unwrap();

        for (i, (&ip, &oop)) in data.iter().zip(out_ref.iter()).enumerate() {
            assert!((ip - oop).abs() < 1e-6, "in-place mismatch at {i}: {ip} vs {oop}");
        }
    }

    // == Batched multi-head softmax tests ================================

    #[test]
    fn test_batched_softmax_config_rejects_zero() {
        assert!(BatchedSoftmaxConfig::new(0, 4, 8).is_err());
        assert!(BatchedSoftmaxConfig::new(2, 0, 8).is_err());
        assert!(BatchedSoftmaxConfig::new(2, 4, 0).is_err());
    }

    #[test]
    fn test_batched_softmax_basic() {
        let cfg = BatchedSoftmaxConfig::new(2, 2, 3).unwrap();
        let total = cfg.total_elements(); // 2*2*3*3 = 36
        assert_eq!(total, 36);

        let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0_f32; total];
        batched_softmax_cpu(&input, &mut output, &cfg).unwrap();

        let seq = cfg.seq_len;
        for slice_idx in 0..(cfg.batch_size * cfg.n_heads) {
            for row in 0..seq {
                let off = slice_idx * seq * seq + row * seq;
                let sum: f32 = output[off..off + seq].iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "slice={slice_idx} row={row} sum={sum}");
            }
        }
    }

    #[test]
    fn test_batched_softmax_with_causal_mask() {
        let cfg = BatchedSoftmaxConfig::new(1, 1, 4).unwrap().with_causal_mask();
        let total = cfg.total_elements(); // 16
        let input: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let mut output = vec![0.0_f32; total];
        batched_softmax_cpu(&input, &mut output, &cfg).unwrap();

        let seq = cfg.seq_len;
        for row in 0..seq {
            for col in (row + 1)..seq {
                let idx = row * seq + col;
                assert!(output[idx].abs() < 1e-7, "row={row} col={col} should be masked");
            }
            let sum: f32 = output[row * seq..row * seq + row + 1].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row={row} sum={sum}");
        }
    }

    #[test]
    fn test_batched_softmax_rejects_short_slices() {
        let cfg = BatchedSoftmaxConfig::new(2, 2, 4).unwrap();
        let short = vec![0.0_f32; 10]; // need 64
        let mut out = vec![0.0_f32; 64];
        assert!(batched_softmax_cpu(&short, &mut out, &cfg).is_err());

        let full = vec![0.0_f32; 64];
        let mut short_out = vec![0.0_f32; 10];
        assert!(batched_softmax_cpu(&full, &mut short_out, &cfg).is_err());
    }

    // == NaN / special-value handling ====================================

    #[test]
    fn test_cpu_softmax_all_negative_infinity() {
        // When all inputs are -inf, row_max = -inf and (x - row_max) is
        // NaN under IEEE 754.  The result is mathematically undefined
        // (0/0), so we just verify the function does not panic.
        let cfg = SoftmaxConfig::for_shape(3, 1).unwrap();
        let input = [f32::NEG_INFINITY; 3];
        let mut output = [99.0_f32; 3];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        // Accept NaN or zero — both are valid degenerate outputs.
        for &v in &output {
            assert!(
                v.is_nan() || v.abs() < 1e-7,
                "expected NaN or ~0 for all-neginf input, got {v}"
            );
        }
    }

    #[test]
    fn test_cpu_softmax_mixed_inf() {
        // +inf in input produces row_max = +inf, then (inf - inf) = NaN
        // under IEEE 754.  We verify the function doesn't panic and
        // the +inf element gets the largest share (or NaN, which is
        // a valid IEEE result for this degenerate input).
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap();
        let input = [1.0_f32, f32::INFINITY, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        // Accept NaN or ~1.0 for the inf element.
        assert!(
            output[1].is_nan() || (output[1] - 1.0).abs() < 1e-5,
            "inf element should dominate or be NaN, got {}",
            output[1]
        );
    }

    #[test]
    fn test_cpu_softmax_with_temperature_and_causal_mask() {
        let n = 3;
        let cfg = SoftmaxConfig::for_shape(n, n)
            .unwrap()
            .with_temperature(0.5)
            .unwrap()
            .with_causal_mask();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut output = [0.0_f32; 9];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        for row in 0..n {
            let start = row * n;
            let sum: f32 = output[start..start + n].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {row} sum={sum}");
            for col in (row + 1)..n {
                assert!(output[start + col].abs() < 1e-7, "masked position row={row} col={col}");
            }
        }
    }

    // == Unified dispatch tests ==========================================

    #[test]
    fn test_softmax_forward_dispatches_cpu() {
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];

        let result = softmax_forward(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CPU dispatch should succeed: {result:?}");

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn test_softmax_forward_matches_cpu() {
        let cfg = SoftmaxConfig::for_shape(5, 2).unwrap().with_temperature(0.7).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let mut out_forward = [0.0_f32; 10];
        let mut out_cpu = [0.0_f32; 10];

        softmax_forward(&input, &mut out_forward, &cfg).unwrap();
        softmax_cpu(&input, &mut out_cpu, &cfg).unwrap();

        for (i, (&fwd, &cpu)) in out_forward.iter().zip(out_cpu.iter()).enumerate() {
            assert!(
                (fwd - cpu).abs() < 1e-6,
                "dispatch mismatch at elem {i}: \
                 forward={fwd}, cpu={cpu}"
            );
        }
    }

    // == GPU launch stub tests ===========================================

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu"]
    fn test_cuda_softmax_launch() {
        let cfg = SoftmaxConfig::for_shape(32000, 4).unwrap();
        let input = vec![1.0_f32; 32000 * 4];
        let mut output = vec![0.0_f32; 32000 * 4];
        let result = launch_softmax(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA softmax launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu"]
    fn test_cuda_softmax_with_temperature() {
        let cfg = SoftmaxConfig::for_shape(32000, 1).unwrap().with_temperature(0.7).unwrap();
        let input = vec![0.0_f32; 32000];
        let mut output = vec![0.0_f32; 32000];
        let result = launch_softmax(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA softmax launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu"]
    fn test_cuda_softmax_causal_mask() {
        let cfg = SoftmaxConfig::for_shape(128, 128).unwrap().with_causal_mask();
        let input = vec![1.0_f32; 128 * 128];
        let mut output = vec![0.0_f32; 128 * 128];
        let result = launch_softmax(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA causal softmax failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu"]
    fn test_cuda_log_softmax() {
        let cfg = SoftmaxConfig::for_shape(32000, 1).unwrap().with_log_softmax();
        let input = vec![1.0_f32; 32000];
        let mut output = vec![0.0_f32; 32000];
        let result = launch_softmax(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA log-softmax failed: {result:?}");
    }

    // == Online softmax tests ============================================

    #[test]
    fn test_online_softmax_matches_standard() {
        let cfg = SoftmaxConfig::for_shape(6, 2).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut out_std = [0.0_f32; 12];
        let mut out_online = [0.0_f32; 12];
        softmax_cpu(&input, &mut out_std, &cfg).unwrap();
        online_softmax_cpu(&input, &mut out_online, &cfg).unwrap();

        for (i, (&a, &b)) in out_std.iter().zip(out_online.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "online vs standard mismatch at {i}: std={a}, online={b}"
            );
        }
    }

    #[test]
    fn test_online_softmax_numerical_stability() {
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap();
        let input = [1e30_f32, 1e30 + 1.0, 1e30 - 1.0, 1e30 + 0.5];
        let mut output = [0.0_f32; 4];
        online_softmax_cpu(&input, &mut output, &cfg).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        assert!(output.iter().all(|&v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn test_online_softmax_with_temperature() {
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap().with_temperature(0.5).unwrap();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut out_online = [0.0_f32; 4];
        let mut out_std = [0.0_f32; 4];
        online_softmax_cpu(&input, &mut out_online, &cfg).unwrap();
        softmax_cpu(&input, &mut out_std, &cfg).unwrap();

        for (i, (&a, &b)) in out_std.iter().zip(out_online.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "temp mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn test_online_softmax_causal_mask() {
        let n = 4;
        let cfg = SoftmaxConfig::for_shape(n, n).unwrap().with_causal_mask();
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut out_online = vec![0.0_f32; 16];
        let mut out_std = vec![0.0_f32; 16];
        online_softmax_cpu(&input, &mut out_online, &cfg).unwrap();
        softmax_cpu(&input, &mut out_std, &cfg).unwrap();

        for (i, (&a, &b)) in out_std.iter().zip(out_online.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "causal mismatch at {i}: std={a}, online={b}");
        }
    }

    #[test]
    fn test_online_softmax_log_mode() {
        let cfg = SoftmaxConfig::for_shape(5, 1).unwrap().with_log_softmax();
        let input = [0.5_f32, 1.5, -0.3, 2.1, 0.0];
        let mut out_online = [0.0_f32; 5];
        let mut out_std = [0.0_f32; 5];
        online_softmax_cpu(&input, &mut out_online, &cfg).unwrap();
        softmax_cpu(&input, &mut out_std, &cfg).unwrap();

        for (i, (&a, &b)) in out_std.iter().zip(out_online.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "log-mode mismatch at {i}: std={a}, online={b}");
        }
    }

    #[test]
    fn test_online_softmax_single_element() {
        let cfg = SoftmaxConfig::for_shape(1, 1).unwrap();
        let input = [42.0_f32];
        let mut output = [0.0_f32; 1];
        online_softmax_cpu(&input, &mut output, &cfg).unwrap();
        assert!((output[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_online_softmax_all_zeros() {
        let n = 8;
        let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
        let input = vec![0.0_f32; n];
        let mut output = vec![0.0_f32; n];
        online_softmax_cpu(&input, &mut output, &cfg).unwrap();

        let expected = 1.0 / n as f32;
        for (i, &v) in output.iter().enumerate() {
            assert!((v - expected).abs() < 1e-6, "all-zeros: elem {i} = {v}, expected {expected}");
        }
    }

    #[test]
    fn test_online_softmax_all_same_values() {
        let n = 16;
        let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
        let input = vec![7.7_f32; n];
        let mut output = vec![0.0_f32; n];
        online_softmax_cpu(&input, &mut output, &cfg).unwrap();

        let expected = 1.0 / n as f32;
        for &v in &output {
            assert!((v - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_online_softmax_rejects_short_slices() {
        let cfg = SoftmaxConfig::for_shape(4, 2).unwrap();
        let short = [1.0_f32; 4]; // need 8
        let mut out = [0.0_f32; 8];
        assert!(online_softmax_cpu(&short, &mut out, &cfg).is_err());

        let full = [1.0_f32; 8];
        let mut short_out = [0.0_f32; 4];
        assert!(online_softmax_cpu(&full, &mut short_out, &cfg).is_err());
    }

    // == Large vocabulary tests ==========================================

    #[test]
    fn test_cpu_softmax_large_vocab_32k() {
        let n = 32_000;
        let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) - 16.0).collect();
        let mut output = vec![0.0_f32; n];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "32k vocab sum={sum}");
        assert!(output.iter().all(|&v| v.is_finite() && v >= 0.0));
    }

    #[test]
    fn test_cpu_softmax_large_vocab_128k() {
        let n = 128_000;
        let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0001) - 6.4).collect();
        let mut output = vec![0.0_f32; n];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "128k vocab sum={sum}");
        assert!(output.iter().all(|&v| v.is_finite() && v >= 0.0));
    }

    #[test]
    fn test_online_softmax_large_vocab_32k() {
        let n = 32_000;
        let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) - 16.0).collect();
        let mut out_std = vec![0.0_f32; n];
        let mut out_online = vec![0.0_f32; n];
        softmax_cpu(&input, &mut out_std, &cfg).unwrap();
        online_softmax_cpu(&input, &mut out_online, &cfg).unwrap();

        let max_diff: f32 = out_std
            .iter()
            .zip(out_online.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5, "32k vocab max_diff={max_diff}");
    }

    // == Softmax backward tests ==========================================

    #[test]
    fn test_softmax_backward_basic() {
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut y = [0.0_f32; 4];
        softmax_cpu(&input, &mut y, &cfg).unwrap();

        let dy = [1.0_f32, 0.0, 0.0, 0.0];
        let mut dx = [0.0_f32; 4];
        softmax_backward_cpu(&y, &dy, &mut dx, 1, 4).unwrap();

        let sum: f32 = dx.iter().sum();
        assert!(sum.abs() < 1e-6, "backward sum={sum}, expected ~0");
        assert!(dx[0] > 0.0, "dx[0] should be positive, got {}", dx[0]);
        for i in 1..4 {
            assert!(dx[i] < 0.0, "dx[{i}] should be negative, got {}", dx[i]);
        }
    }

    #[test]
    fn test_softmax_backward_gradient_sum_zero() {
        let cfg = SoftmaxConfig::for_shape(5, 1).unwrap();
        let input = [0.5_f32, 1.5, -0.3, 2.1, 0.0];
        let mut y = [0.0_f32; 5];
        softmax_cpu(&input, &mut y, &cfg).unwrap();

        let dy = [0.2, -0.5, 0.8, 0.1, -0.3];
        let mut dx = [0.0_f32; 5];
        softmax_backward_cpu(&y, &dy, &mut dx, 1, 5).unwrap();

        let sum: f32 = dx.iter().sum();
        assert!(sum.abs() < 1e-6, "backward sum={sum}");
    }

    #[test]
    fn test_softmax_backward_identity_gradient() {
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut y = [0.0_f32; 4];
        softmax_cpu(&input, &mut y, &cfg).unwrap();

        let mut dx = [0.0_f32; 4];
        softmax_backward_cpu(&y, &y, &mut dx, 1, 4).unwrap();

        let sum_sq: f32 = y.iter().map(|v| v * v).sum();
        for (i, (&dxi, &yi)) in dx.iter().zip(y.iter()).enumerate() {
            let expected = yi * (yi - sum_sq);
            assert!(
                (dxi - expected).abs() < 1e-7,
                "identity grad mismatch at {i}: {dxi} vs {expected}"
            );
        }
    }

    #[test]
    fn test_softmax_backward_multi_row() {
        let cfg = SoftmaxConfig::for_shape(3, 2).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut y = [0.0_f32; 6];
        softmax_cpu(&input, &mut y, &cfg).unwrap();

        let dy = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut dx = [0.0_f32; 6];
        softmax_backward_cpu(&y, &dy, &mut dx, 2, 3).unwrap();

        let sum_r0: f32 = dx[0..3].iter().sum();
        let sum_r1: f32 = dx[3..6].iter().sum();
        assert!(sum_r0.abs() < 1e-6, "row0 backward sum={sum_r0}");
        assert!(sum_r1.abs() < 1e-6, "row1 backward sum={sum_r1}");
    }

    #[test]
    fn test_softmax_backward_rejects_short_slices() {
        let y = [0.25_f32; 4];
        let dy = [1.0_f32; 4];
        let mut dx = [0.0_f32; 2]; // too short
        assert!(softmax_backward_cpu(&y, &dy, &mut dx, 1, 4).is_err());
    }

    // == Comparison: naive vs stable =====================================

    /// Naive (numerically unstable) softmax for comparison.
    fn naive_softmax(input: &[f32]) -> Vec<f32> {
        let exps: Vec<f32> = input.iter().map(|&x| x.exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    #[test]
    fn test_naive_vs_stable_small_values() {
        let input = [0.1_f32, 0.2, 0.3, 0.4];
        let naive = naive_softmax(&input);

        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap();
        let mut stable = [0.0_f32; 4];
        softmax_cpu(&input, &mut stable, &cfg).unwrap();

        for (i, (&n, &s)) in naive.iter().zip(stable.iter()).enumerate() {
            assert!((n - s).abs() < 1e-6, "small values differ at {i}: naive={n}, stable={s}");
        }
    }

    #[test]
    fn test_naive_overflows_stable_does_not() {
        let input = [500.0_f32, 501.0, 502.0];
        let naive = naive_softmax(&input);
        let naive_has_nan = naive.iter().any(|v| v.is_nan() || v.is_infinite());

        let cfg = SoftmaxConfig::for_shape(3, 1).unwrap();
        let mut stable = [0.0_f32; 3];
        softmax_cpu(&input, &mut stable, &cfg).unwrap();

        assert!(stable.iter().all(|&v| v.is_finite()), "stable produced non-finite");
        let sum: f32 = stable.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "stable sum={sum}");

        if naive_has_nan {
            assert!(stable.iter().all(|&v| v.is_finite()));
        }
    }

    // == CUDA kernel source tests ========================================

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_softmax_kernel_src_is_nonempty() {
        assert!(!SOFTMAX_KERNEL_SRC.is_empty());
        assert!(SOFTMAX_KERNEL_SRC.contains("softmax_online"));
        assert!(SOFTMAX_KERNEL_SRC.contains("__shfl_xor_sync"));
        assert!(SOFTMAX_KERNEL_SRC.contains("__shared__"));
    }
}

// ---------------------------------------------------------------------------
// Property tests (proptest)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating a row of logits with reasonable values.
    fn logits_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-50.0_f32..50.0_f32, 1..max_len)
    }

    proptest! {
        #[test]
        fn prop_softmax_output_sums_to_one(input in logits_vec(256)) {
            let n = input.len();
            let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
            let mut output = vec![0.0_f32; n];
            softmax_cpu(&input, &mut output, &cfg).unwrap();

            let sum: f32 = output.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-4, "sum={sum}");
        }

        #[test]
        fn prop_softmax_output_non_negative(input in logits_vec(256)) {
            let n = input.len();
            let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
            let mut output = vec![0.0_f32; n];
            softmax_cpu(&input, &mut output, &cfg).unwrap();

            for (i, &v) in output.iter().enumerate() {
                prop_assert!(v >= 0.0, "negative at {i}: {v}");
            }
        }

        #[test]
        fn prop_softmax_preserves_order(input in logits_vec(128)) {
            let n = input.len();
            let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
            let mut output = vec![0.0_f32; n];
            softmax_cpu(&input, &mut output, &cfg).unwrap();

            for i in 0..n {
                for j in (i + 1)..n {
                    if input[i] > input[j] {
                        prop_assert!(output[i] >= output[j],
                            "order violation: input[{i}]={} > input[{j}]={} \
                             but output[{i}]={} < output[{j}]={}",
                            input[i], input[j], output[i], output[j]);
                    }
                }
            }
        }

        #[test]
        fn prop_online_matches_standard(input in logits_vec(256)) {
            let n = input.len();
            let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
            let mut out_std = vec![0.0_f32; n];
            let mut out_online = vec![0.0_f32; n];
            softmax_cpu(&input, &mut out_std, &cfg).unwrap();
            online_softmax_cpu(&input, &mut out_online, &cfg).unwrap();

            let max_diff: f32 = out_std.iter().zip(out_online.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            prop_assert!(max_diff < 1e-5, "max_diff={max_diff}");
        }

        #[test]
        fn prop_softmax_backward_sums_to_zero(
            input in logits_vec(64),
            grad in logits_vec(64),
        ) {
            let n = input.len().min(grad.len());
            if n == 0 { return Ok(()); }
            let input = &input[..n];
            let grad = &grad[..n];

            let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
            let mut y = vec![0.0_f32; n];
            softmax_cpu(input, &mut y, &cfg).unwrap();

            let mut dx = vec![0.0_f32; n];
            softmax_backward_cpu(&y, grad, &mut dx, 1, n).unwrap();

            let sum: f32 = dx.iter().sum();
            prop_assert!(sum.abs() < 1e-4, "backward sum={sum}");
        }

        #[test]
        fn prop_temperature_preserves_probability(
            input in logits_vec(64),
            temp in 0.01_f32..100.0_f32,
        ) {
            let n = input.len();
            let cfg = SoftmaxConfig::for_shape(n, 1).unwrap()
                .with_temperature(temp).unwrap();
            let mut output = vec![0.0_f32; n];
            softmax_cpu(&input, &mut output, &cfg).unwrap();

            let sum: f32 = output.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-3, "temp={temp} sum={sum}");
            for &v in &output {
                prop_assert!(v >= 0.0 && v.is_finite());
            }
        }

        #[test]
        fn prop_log_softmax_exp_sums_to_one(input in logits_vec(128)) {
            let n = input.len();
            let cfg = SoftmaxConfig::for_shape(n, 1).unwrap().with_log_softmax();
            let mut output = vec![0.0_f32; n];
            softmax_cpu(&input, &mut output, &cfg).unwrap();

            let sum: f32 = output.iter().map(|&v| v.exp()).sum();
            prop_assert!((sum - 1.0).abs() < 1e-4, "exp(log-softmax) sum={sum}");
        }
    }
}
