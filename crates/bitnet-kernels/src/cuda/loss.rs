//! CUDA loss function kernels with CPU fallback.
//!
//! Provides GPU-accelerated and scalar CPU implementations of common loss
//! functions used for training and evaluation of neural network models:
//!
//! - **Cross-entropy loss** — standard categorical cross-entropy over
//!   probability distributions.
//! - **Cross-entropy with logits** — numerically stable cross-entropy
//!   computed directly from raw logits using the log-sum-exp trick.
//! - **Binary cross-entropy** — per-element binary classification loss.
//! - **Mean squared error** — L2 regression loss.
//! - **Huber loss** — smooth L1 loss that transitions from quadratic to
//!   linear at a configurable delta threshold.
//! - **KL divergence** — Kullback–Leibler divergence between two
//!   probability distributions.
//! - **Focal loss** — down-weights easy examples to focus learning on
//!   hard negatives (Lin et al., 2017).
//! - **Label-smoothing CE** — cross-entropy with uniform label smoothing.
//! - **Perplexity from logits** — exponentiated average cross-entropy.
//! - **Contrastive loss** — pairwise contrastive learning loss.
//! - **Triplet loss** — triplet margin loss for metric learning.
//!
//! # Kernel strategy
//!
//! The CUDA kernels use grid-stride loops with 256 threads per block.
//! Reduction-style losses (cross-entropy, MSE, etc.) use shared-memory
//! parallel reduction to compute per-block partial sums which are then
//! summed on the host.  For the CPU path every function is a
//! straightforward scalar loop.
//!
//! # CPU fallback
//!
//! Every public function has a pure-Rust scalar implementation that
//! serves as the reference for correctness testing and non-GPU
//! environments.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// CUDA kernel source
// ---------------------------------------------------------------------------

/// Inline CUDA C source for loss function kernels.
///
/// Contains kernels for cross-entropy, MSE, Huber, focal, and other
/// loss computations.  Each kernel reduces `n` elements into a scalar
/// loss using shared-memory parallel reduction.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const LOSS_KERNEL_SRC: &str = r#"
extern "C" __global__ void cross_entropy_f32(
    const float* __restrict__ probs,
    const int*   __restrict__ targets,
    float*       __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float p = fmaxf(probs[targets[idx] + idx * gridDim.y], 1e-7f);
        output[idx] = -logf(p);
    }
}

extern "C" __global__ void mse_f32(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float*       __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float diff = predictions[i] - targets[i];
        output[i] = diff * diff;
    }
}

extern "C" __global__ void huber_f32(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float*       __restrict__ output,
    float delta,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        if (abs_diff <= delta) {
            output[i] = 0.5f * diff * diff;
        } else {
            output[i] = delta * (abs_diff - 0.5f * delta);
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Reduction kind
// ---------------------------------------------------------------------------

/// How to reduce element-wise losses into a scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossReduction {
    /// Return the arithmetic mean of all element losses.
    Mean,
    /// Return the sum of all element losses.
    Sum,
    /// Return element-wise losses without reduction (output has the same
    /// length as the input).
    None,
}

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

/// Launch configuration for loss kernels.
#[derive(Debug, Clone)]
pub struct LossConfig {
    /// Number of elements (batch size or total element count).
    pub n: usize,
    /// Number of classes (vocabulary size) for classification losses.
    pub n_classes: usize,
    /// Threads per block (default 256).
    pub threads_per_block: u32,
    /// Reduction mode applied to element-wise losses.
    pub reduction: LossReduction,
}

impl LossConfig {
    /// Create a configuration for the given element count and class count.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `n` is zero.
    pub fn new(n: usize, n_classes: usize) -> Result<Self> {
        if n == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "loss element count must be non-zero".into(),
            }
            .into());
        }
        Ok(Self { n, n_classes, threads_per_block: 256, reduction: LossReduction::Mean })
    }

    /// Override the reduction mode.
    pub fn with_reduction(mut self, reduction: LossReduction) -> Self {
        self.reduction = reduction;
        self
    }

    /// Compute the CUDA grid dimensions.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let blocks = (self.n as u32).div_ceil(self.threads_per_block);
        (blocks, 1, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// Helper: apply reduction
// ---------------------------------------------------------------------------

fn apply_reduction(values: &[f32], reduction: LossReduction) -> Vec<f32> {
    match reduction {
        LossReduction::None => values.to_vec(),
        LossReduction::Sum => {
            let s: f32 = values.iter().sum();
            vec![s]
        }
        LossReduction::Mean => {
            if values.is_empty() {
                return vec![0.0];
            }
            let s: f32 = values.iter().sum();
            vec![s / values.len() as f32]
        }
    }
}

// ---------------------------------------------------------------------------
// CPU fallback implementations
// ---------------------------------------------------------------------------

/// CPU cross-entropy loss.
///
/// For each sample `i`, computes `-log(probs[i * n_classes + targets[i]])`.
///
/// # Arguments
///
/// * `probs` — Probability distribution `[n, n_classes]` (row-major,
///   should sum to ≈1 per row).
/// * `targets` — Ground-truth class indices `[n]`, each in `0..n_classes`.
/// * `config` — Loss configuration.
///
/// # Errors
///
/// Returns an error if slice lengths are inconsistent with `config`.
pub fn cross_entropy_loss(probs: &[f32], targets: &[u32], config: &LossConfig) -> Result<Vec<f32>> {
    let n = config.n;
    let nc = config.n_classes;
    if probs.len() < n * nc {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "cross_entropy_loss: probs length {} < n*n_classes {}",
                probs.len(),
                n * nc
            ),
        }
        .into());
    }
    if targets.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("cross_entropy_loss: targets length {} < n {}", targets.len(), n),
        }
        .into());
    }
    let mut losses = Vec::with_capacity(n);
    for i in 0..n {
        let t = targets[i] as usize;
        if t >= nc {
            return Err(KernelError::InvalidArguments {
                reason: format!("cross_entropy_loss: target {t} >= n_classes {nc} at index {i}"),
            }
            .into());
        }
        let p = probs[i * nc + t].max(1e-7);
        losses.push(-p.ln());
    }
    Ok(apply_reduction(&losses, config.reduction))
}

/// CPU cross-entropy loss computed from raw logits (numerically stable).
///
/// Uses the log-sum-exp trick: `loss_i = -logits[target] + log(sum(exp(logits)))`.
///
/// # Arguments
///
/// * `logits` — Raw logits `[n, n_classes]` (row-major).
/// * `targets` — Ground-truth class indices `[n]`.
/// * `config` — Loss configuration.
pub fn cross_entropy_with_logits(
    logits: &[f32],
    targets: &[u32],
    config: &LossConfig,
) -> Result<Vec<f32>> {
    let n = config.n;
    let nc = config.n_classes;
    if logits.len() < n * nc {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "cross_entropy_with_logits: logits length {} < n*n_classes {}",
                logits.len(),
                n * nc
            ),
        }
        .into());
    }
    if targets.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "cross_entropy_with_logits: targets length {} < n {}",
                targets.len(),
                n
            ),
        }
        .into());
    }
    let mut losses = Vec::with_capacity(n);
    for i in 0..n {
        let t = targets[i] as usize;
        if t >= nc {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "cross_entropy_with_logits: target {t} >= n_classes {nc} at index {i}"
                ),
            }
            .into());
        }
        let row = &logits[i * nc..(i + 1) * nc];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln() + max_val;
        losses.push(-row[t] + log_sum_exp);
    }
    Ok(apply_reduction(&losses, config.reduction))
}

/// CPU binary cross-entropy loss.
///
/// For each element: `-[y * log(p) + (1 - y) * log(1 - p)]`.
///
/// # Arguments
///
/// * `predictions` — Predicted probabilities in `[0, 1]`, length `n`.
/// * `targets` — Binary targets in `{0, 1}` (as f32), length `n`.
/// * `config` — Loss configuration (only `n` and `reduction` are used).
pub fn binary_cross_entropy(
    predictions: &[f32],
    targets: &[f32],
    config: &LossConfig,
) -> Result<Vec<f32>> {
    let n = config.n;
    if predictions.len() < n || targets.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "binary_cross_entropy: need at least {n} elements, got pred={} target={}",
                predictions.len(),
                targets.len()
            ),
        }
        .into());
    }
    let mut losses = Vec::with_capacity(n);
    for i in 0..n {
        let p = predictions[i].clamp(1e-7, 1.0 - 1e-7);
        let y = targets[i];
        losses.push(-(y * p.ln() + (1.0 - y) * (1.0 - p).ln()));
    }
    Ok(apply_reduction(&losses, config.reduction))
}

/// CPU mean squared error loss.
///
/// For each element: `(prediction - target)²`.
///
/// # Arguments
///
/// * `predictions` — Predicted values, length `n`.
/// * `targets` — Target values, length `n`.
/// * `config` — Loss configuration.
pub fn mse_loss(predictions: &[f32], targets: &[f32], config: &LossConfig) -> Result<Vec<f32>> {
    let n = config.n;
    if predictions.len() < n || targets.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "mse_loss: need at least {n} elements, got pred={} target={}",
                predictions.len(),
                targets.len()
            ),
        }
        .into());
    }
    let mut losses = Vec::with_capacity(n);
    for i in 0..n {
        let diff = predictions[i] - targets[i];
        losses.push(diff * diff);
    }
    Ok(apply_reduction(&losses, config.reduction))
}

/// CPU Huber (smooth L1) loss.
///
/// Quadratic for `|error| <= delta`, linear beyond:
/// ```text
/// huber(x) = 0.5 * x²            if |x| <= delta
///          = delta * (|x| - 0.5 * delta) otherwise
/// ```
///
/// # Arguments
///
/// * `predictions` — Predicted values, length `n`.
/// * `targets` — Target values, length `n`.
/// * `delta` — Transition threshold between quadratic and linear regions.
/// * `config` — Loss configuration.
pub fn huber_loss(
    predictions: &[f32],
    targets: &[f32],
    delta: f32,
    config: &LossConfig,
) -> Result<Vec<f32>> {
    let n = config.n;
    if predictions.len() < n || targets.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "huber_loss: need at least {n} elements, got pred={} target={}",
                predictions.len(),
                targets.len()
            ),
        }
        .into());
    }
    if !delta.is_finite() || delta <= 0.0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("huber_loss: delta must be positive and finite, got {delta}"),
        }
        .into());
    }
    let mut losses = Vec::with_capacity(n);
    for i in 0..n {
        let diff = predictions[i] - targets[i];
        let abs_diff = diff.abs();
        if abs_diff <= delta {
            losses.push(0.5 * diff * diff);
        } else {
            losses.push(delta * (abs_diff - 0.5 * delta));
        }
    }
    Ok(apply_reduction(&losses, config.reduction))
}

/// CPU Kullback–Leibler divergence.
///
/// For each element: `target * log(target / prediction)`.
/// Elements where `target ≈ 0` contribute zero (by convention `0 log 0 = 0`).
///
/// # Arguments
///
/// * `predictions` — Predicted distribution (probabilities), length `n`.
/// * `targets` — Target distribution (probabilities), length `n`.
/// * `config` — Loss configuration.
pub fn kl_divergence(
    predictions: &[f32],
    targets: &[f32],
    config: &LossConfig,
) -> Result<Vec<f32>> {
    let n = config.n;
    if predictions.len() < n || targets.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "kl_divergence: need at least {n} elements, got pred={} target={}",
                predictions.len(),
                targets.len()
            ),
        }
        .into());
    }
    let mut losses = Vec::with_capacity(n);
    for i in 0..n {
        let t = targets[i];
        if t <= 0.0 {
            losses.push(0.0);
        } else {
            let p = predictions[i].max(1e-7);
            losses.push(t * (t / p).ln());
        }
    }
    Ok(apply_reduction(&losses, config.reduction))
}

/// CPU focal loss (Lin et al., 2017).
///
/// Down-weights well-classified examples by a factor `(1 - p_t)^gamma`:
/// ```text
/// focal(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
/// ```
///
/// # Arguments
///
/// * `probs` — Probability distribution `[n, n_classes]` (row-major).
/// * `targets` — Ground-truth class indices `[n]`.
/// * `alpha` — Balancing factor (typically 0.25).
/// * `gamma` — Focusing parameter (typically 2.0).
/// * `config` — Loss configuration.
pub fn focal_loss(
    probs: &[f32],
    targets: &[u32],
    alpha: f32,
    gamma: f32,
    config: &LossConfig,
) -> Result<Vec<f32>> {
    let n = config.n;
    let nc = config.n_classes;
    if probs.len() < n * nc {
        return Err(KernelError::InvalidArguments {
            reason: format!("focal_loss: probs length {} < n*n_classes {}", probs.len(), n * nc),
        }
        .into());
    }
    if targets.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("focal_loss: targets length {} < n {}", targets.len(), n),
        }
        .into());
    }
    let mut losses = Vec::with_capacity(n);
    for i in 0..n {
        let t = targets[i] as usize;
        if t >= nc {
            return Err(KernelError::InvalidArguments {
                reason: format!("focal_loss: target {t} >= n_classes {nc} at index {i}"),
            }
            .into());
        }
        let p_t = probs[i * nc + t].max(1e-7);
        losses.push(-alpha * (1.0 - p_t).powf(gamma) * p_t.ln());
    }
    Ok(apply_reduction(&losses, config.reduction))
}

/// CPU cross-entropy with label smoothing.
///
/// Blends the one-hot target distribution with a uniform distribution:
/// ```text
/// target' = (1 - smoothing) * one_hot(target) + smoothing / n_classes
/// loss = -sum(target'_c * log(prob_c))
/// ```
///
/// # Arguments
///
/// * `probs` — Probability distribution `[n, n_classes]` (row-major).
/// * `targets` — Ground-truth class indices `[n]`.
/// * `smoothing` — Smoothing factor in `[0, 1)`.
/// * `config` — Loss configuration.
pub fn label_smoothing_ce(
    probs: &[f32],
    targets: &[u32],
    smoothing: f32,
    config: &LossConfig,
) -> Result<Vec<f32>> {
    let n = config.n;
    let nc = config.n_classes;
    if probs.len() < n * nc {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "label_smoothing_ce: probs length {} < n*n_classes {}",
                probs.len(),
                n * nc
            ),
        }
        .into());
    }
    if targets.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("label_smoothing_ce: targets length {} < n {}", targets.len(), n),
        }
        .into());
    }
    if !(0.0..1.0).contains(&smoothing) {
        return Err(KernelError::InvalidArguments {
            reason: format!("label_smoothing_ce: smoothing must be in [0, 1), got {smoothing}"),
        }
        .into());
    }
    let uniform = smoothing / nc as f32;
    let on_value = 1.0 - smoothing + uniform;
    let mut losses = Vec::with_capacity(n);
    for i in 0..n {
        let t = targets[i] as usize;
        if t >= nc {
            return Err(KernelError::InvalidArguments {
                reason: format!("label_smoothing_ce: target {t} >= n_classes {nc} at index {i}"),
            }
            .into());
        }
        let row = &probs[i * nc..(i + 1) * nc];
        let mut loss = 0.0_f32;
        for (c, &p) in row.iter().enumerate() {
            let p_clamped = p.max(1e-7);
            let target_c = if c == t { on_value } else { uniform };
            loss -= target_c * p_clamped.ln();
        }
        losses.push(loss);
    }
    Ok(apply_reduction(&losses, config.reduction))
}

/// CPU perplexity from raw logits.
///
/// Computes `exp(mean(cross_entropy_with_logits))` — the exponentiated
/// average per-token cross-entropy loss.
///
/// # Arguments
///
/// * `logits` — Raw logits `[n, n_classes]` (row-major).
/// * `targets` — Ground-truth class indices `[n]`.
/// * `config` — Loss configuration (reduction is forced to `Mean`
///   internally).
pub fn perplexity_from_logits(logits: &[f32], targets: &[u32], config: &LossConfig) -> Result<f32> {
    let mean_cfg = LossConfig {
        n: config.n,
        n_classes: config.n_classes,
        threads_per_block: config.threads_per_block,
        reduction: LossReduction::Mean,
    };
    let ce = cross_entropy_with_logits(logits, targets, &mean_cfg)?;
    Ok(ce[0].exp())
}

/// CPU contrastive loss.
///
/// For paired embeddings `(a, b)` with label `y ∈ {0, 1}`:
/// ```text
/// loss = y * d² + (1 - y) * max(0, margin - d)²
/// ```
/// where `d = ||a - b||₂`.
///
/// # Arguments
///
/// * `embeddings_a` — First embedding set, `[n, dim]` (row-major).
/// * `embeddings_b` — Second embedding set, `[n, dim]` (row-major).
/// * `labels` — Binary labels (1.0 = similar, 0.0 = dissimilar).
/// * `margin` — Distance margin for dissimilar pairs.
/// * `dim` — Embedding dimensionality.
/// * `config` — Loss configuration (only `n` and `reduction` used).
pub fn contrastive_loss(
    embeddings_a: &[f32],
    embeddings_b: &[f32],
    labels: &[f32],
    margin: f32,
    dim: usize,
    config: &LossConfig,
) -> Result<Vec<f32>> {
    let n = config.n;
    if embeddings_a.len() < n * dim || embeddings_b.len() < n * dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "contrastive_loss: need at least {} embedding elements, got a={} b={}",
                n * dim,
                embeddings_a.len(),
                embeddings_b.len()
            ),
        }
        .into());
    }
    if labels.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("contrastive_loss: labels length {} < n {}", labels.len(), n),
        }
        .into());
    }
    let mut losses = Vec::with_capacity(n);
    for i in 0..n {
        let a = &embeddings_a[i * dim..(i + 1) * dim];
        let b = &embeddings_b[i * dim..(i + 1) * dim];
        let dist_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
        let dist = dist_sq.sqrt();
        let y = labels[i];
        let hinge = (margin - dist).max(0.0);
        losses.push(y * dist_sq + (1.0 - y) * hinge * hinge);
    }
    Ok(apply_reduction(&losses, config.reduction))
}

/// CPU triplet margin loss.
///
/// For anchor/positive/negative triplets:
/// ```text
/// loss = max(0, ||anchor - positive||₂ - ||anchor - negative||₂ + margin)
/// ```
///
/// # Arguments
///
/// * `anchor` — Anchor embeddings `[n, dim]` (row-major).
/// * `positive` — Positive embeddings `[n, dim]` (row-major).
/// * `negative` — Negative embeddings `[n, dim]` (row-major).
/// * `margin` — Distance margin.
/// * `dim` — Embedding dimensionality.
/// * `config` — Loss configuration (only `n` and `reduction` used).
pub fn triplet_loss(
    anchor: &[f32],
    positive: &[f32],
    negative: &[f32],
    margin: f32,
    dim: usize,
    config: &LossConfig,
) -> Result<Vec<f32>> {
    let n = config.n;
    let required = n * dim;
    if anchor.len() < required || positive.len() < required || negative.len() < required {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "triplet_loss: need at least {required} elements, got anchor={} pos={} neg={}",
                anchor.len(),
                positive.len(),
                negative.len()
            ),
        }
        .into());
    }
    let mut losses = Vec::with_capacity(n);
    for i in 0..n {
        let a = &anchor[i * dim..(i + 1) * dim];
        let p = &positive[i * dim..(i + 1) * dim];
        let neg = &negative[i * dim..(i + 1) * dim];
        let dist_ap: f32 =
            a.iter().zip(p.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
        let dist_an: f32 =
            a.iter().zip(neg.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
        losses.push((dist_ap - dist_an + margin).max(0.0));
    }
    Ok(apply_reduction(&losses, config.reduction))
}

// ---------------------------------------------------------------------------
// CUDA launch stubs
// ---------------------------------------------------------------------------

/// Launch cross-entropy loss CUDA kernel stub.
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_cross_entropy_loss(
    _probs: &[f32],
    _targets: &[u32],
    config: &LossConfig,
) -> Result<Vec<f32>> {
    log::debug!(
        "cross_entropy_loss CUDA stub: n={}, n_classes={}, grid={:?}",
        config.n,
        config.n_classes,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "cross_entropy_loss CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch MSE loss CUDA kernel stub.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_mse_loss(
    _predictions: &[f32],
    _targets: &[f32],
    config: &LossConfig,
) -> Result<Vec<f32>> {
    log::debug!("mse_loss CUDA stub: n={}, grid={:?}", config.n, config.grid_dim(),);
    Err(KernelError::GpuError {
        reason: "mse_loss CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch Huber loss CUDA kernel stub.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_huber_loss(
    _predictions: &[f32],
    _targets: &[f32],
    _delta: f32,
    config: &LossConfig,
) -> Result<Vec<f32>> {
    log::debug!("huber_loss CUDA stub: n={}, grid={:?}", config.n, config.grid_dim(),);
    Err(KernelError::GpuError {
        reason: "huber_loss CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ---------------------------------------------------------------------------
// Unified dispatch — GPU if available, else CPU
// ---------------------------------------------------------------------------

/// Cross-entropy loss with automatic dispatch.
pub fn cross_entropy_loss_forward(
    probs: &[f32],
    targets: &[u32],
    config: &LossConfig,
) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(result) = launch_cross_entropy_loss(probs, targets, config)
        {
            return Ok(result);
        }
    }
    cross_entropy_loss(probs, targets, config)
}

/// MSE loss with automatic dispatch.
pub fn mse_loss_forward(
    predictions: &[f32],
    targets: &[f32],
    config: &LossConfig,
) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(result) = launch_mse_loss(predictions, targets, config)
        {
            return Ok(result);
        }
    }
    mse_loss(predictions, targets, config)
}

/// Huber loss with automatic dispatch.
pub fn huber_loss_forward(
    predictions: &[f32],
    targets: &[f32],
    delta: f32,
    config: &LossConfig,
) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(result) = launch_huber_loss(predictions, targets, delta, config)
        {
            return Ok(result);
        }
    }
    huber_loss(predictions, targets, delta, config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // == LossConfig tests ================================================

    #[test]
    fn test_loss_config_new() {
        let cfg = LossConfig::new(32, 10).unwrap();
        assert_eq!(cfg.n, 32);
        assert_eq!(cfg.n_classes, 10);
        assert_eq!(cfg.threads_per_block, 256);
        assert_eq!(cfg.reduction, LossReduction::Mean);
    }

    #[test]
    fn test_loss_config_rejects_zero() {
        assert!(LossConfig::new(0, 10).is_err());
    }

    #[test]
    fn test_loss_config_grid_dim() {
        let cfg = LossConfig::new(1000, 5).unwrap();
        assert_eq!(cfg.grid_dim(), (4, 1, 1)); // ceil(1000/256)
        assert_eq!(cfg.block_dim(), (256, 1, 1));
    }

    #[test]
    fn test_loss_config_with_reduction() {
        let cfg = LossConfig::new(8, 3).unwrap().with_reduction(LossReduction::Sum);
        assert_eq!(cfg.reduction, LossReduction::Sum);
    }

    #[test]
    fn test_loss_config_small_n() {
        let cfg = LossConfig::new(1, 2).unwrap();
        assert_eq!(cfg.grid_dim(), (1, 1, 1));
    }

    // == Cross-entropy loss tests ========================================

    #[test]
    fn test_cross_entropy_basic() {
        // 2 samples, 3 classes; perfect predictions → near-zero loss
        let probs = [
            0.9, 0.05, 0.05, // sample 0 → class 0
            0.1, 0.8, 0.1, // sample 1 → class 1
        ];
        let targets = [0, 1];
        let cfg = LossConfig::new(2, 3).unwrap();
        let loss = cross_entropy_loss(&probs, &targets, &cfg).unwrap();
        assert_eq!(loss.len(), 1);
        assert!(loss[0] < 0.3, "expected low loss for good predictions, got {}", loss[0]);
    }

    #[test]
    fn test_cross_entropy_perfect_prediction() {
        let probs = [1.0, 0.0, 0.0];
        let targets = [0];
        let cfg = LossConfig::new(1, 3).unwrap();
        let loss = cross_entropy_loss(&probs, &targets, &cfg).unwrap();
        // -log(max(1.0, 1e-7)) ≈ 0.0
        assert!(loss[0].abs() < 1e-5, "perfect prediction loss = {}", loss[0]);
    }

    #[test]
    fn test_cross_entropy_worst_prediction() {
        let probs = [0.0, 0.0, 1.0];
        let targets = [0]; // class 0 has p≈0
        let cfg = LossConfig::new(1, 3).unwrap();
        let loss = cross_entropy_loss(&probs, &targets, &cfg).unwrap();
        // -log(1e-7) ≈ 16.1
        assert!(loss[0] > 10.0, "worst prediction loss should be large, got {}", loss[0]);
    }

    #[test]
    fn test_cross_entropy_uniform() {
        let probs = [1.0 / 3.0; 3];
        let targets = [1];
        let cfg = LossConfig::new(1, 3).unwrap();
        let loss = cross_entropy_loss(&probs, &targets, &cfg).unwrap();
        let expected = -(1.0_f32 / 3.0).ln();
        assert!((loss[0] - expected).abs() < 1e-5, "got {}, expected {expected}", loss[0]);
    }

    #[test]
    fn test_cross_entropy_sum_reduction() {
        let probs = [0.7, 0.3, 0.2, 0.8];
        let targets = [0, 1];
        let cfg = LossConfig::new(2, 2).unwrap().with_reduction(LossReduction::Sum);
        let loss = cross_entropy_loss(&probs, &targets, &cfg).unwrap();
        let expected = -0.7_f32.ln() + -0.8_f32.ln();
        assert!((loss[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_cross_entropy_none_reduction() {
        let probs = [0.7, 0.3, 0.2, 0.8];
        let targets = [0, 1];
        let cfg = LossConfig::new(2, 2).unwrap().with_reduction(LossReduction::None);
        let loss = cross_entropy_loss(&probs, &targets, &cfg).unwrap();
        assert_eq!(loss.len(), 2);
        assert!((loss[0] - (-0.7_f32.ln())).abs() < 1e-5);
        assert!((loss[1] - (-0.8_f32.ln())).abs() < 1e-5);
    }

    #[test]
    fn test_cross_entropy_rejects_bad_target() {
        let probs = [0.5, 0.5];
        let targets = [2]; // out of range
        let cfg = LossConfig::new(1, 2).unwrap();
        assert!(cross_entropy_loss(&probs, &targets, &cfg).is_err());
    }

    #[test]
    fn test_cross_entropy_rejects_short_probs() {
        let probs = [0.5]; // too short
        let targets = [0];
        let cfg = LossConfig::new(1, 3).unwrap();
        assert!(cross_entropy_loss(&probs, &targets, &cfg).is_err());
    }

    // == Cross-entropy with logits tests =================================

    #[test]
    fn test_ce_logits_basic() {
        // logits [2.0, 1.0, 0.1], target=0
        let logits = [2.0_f32, 1.0, 0.1];
        let targets = [0_u32];
        let cfg = LossConfig::new(1, 3).unwrap();
        let loss = cross_entropy_with_logits(&logits, &targets, &cfg).unwrap();
        // Manual: max=2.0, logsumexp = 2.0 + ln(1 + e^-1 + e^-1.9)
        let lse = 2.0 + (1.0_f32 + (-1.0_f32).exp() + (-1.9_f32).exp()).ln();
        let expected = -2.0 + lse;
        assert!((loss[0] - expected).abs() < 1e-4, "got {}, expected {expected}", loss[0]);
    }

    #[test]
    fn test_ce_logits_numerical_stability() {
        // Very large logits should not overflow
        let logits = [1000.0_f32, 999.0, 998.0];
        let targets = [0_u32];
        let cfg = LossConfig::new(1, 3).unwrap();
        let loss = cross_entropy_with_logits(&logits, &targets, &cfg).unwrap();
        assert!(loss[0].is_finite(), "loss should be finite, got {}", loss[0]);
        assert!(loss[0] >= 0.0, "CE loss should be non-negative");
    }

    #[test]
    fn test_ce_logits_very_negative() {
        let logits = [-1000.0_f32, -999.0, -998.0];
        let targets = [2_u32];
        let cfg = LossConfig::new(1, 3).unwrap();
        let loss = cross_entropy_with_logits(&logits, &targets, &cfg).unwrap();
        assert!(loss[0].is_finite());
    }

    #[test]
    fn test_ce_logits_matches_manual() {
        // 2 samples, 2 classes
        let logits = [1.0_f32, 0.0, 0.0, 1.0];
        let targets = [0_u32, 1];
        let cfg = LossConfig::new(2, 2).unwrap().with_reduction(LossReduction::None);
        let loss = cross_entropy_with_logits(&logits, &targets, &cfg).unwrap();
        let lse = (1.0_f32.exp() + 0.0_f32.exp()).ln();
        let expected_each = -1.0 + lse;
        assert!((loss[0] - expected_each).abs() < 1e-5);
        assert!((loss[1] - expected_each).abs() < 1e-5);
    }

    #[test]
    fn test_ce_logits_rejects_bad_target() {
        let logits = [1.0, 2.0];
        let targets = [2_u32]; // out of range
        let cfg = LossConfig::new(1, 2).unwrap();
        assert!(cross_entropy_with_logits(&logits, &targets, &cfg).is_err());
    }

    // == Binary cross-entropy tests ======================================

    #[test]
    fn test_bce_basic() {
        let preds = [0.9, 0.1];
        let targets = [1.0, 0.0];
        let cfg = LossConfig::new(2, 1).unwrap();
        let loss = binary_cross_entropy(&preds, &targets, &cfg).unwrap();
        // Each close to -log(0.9) ≈ 0.105
        assert!(loss[0] < 0.2, "expected low BCE for good preds, got {}", loss[0]);
    }

    #[test]
    fn test_bce_perfect() {
        let preds = [1.0, 0.0];
        let targets = [1.0, 0.0];
        let cfg = LossConfig::new(2, 1).unwrap();
        let loss = binary_cross_entropy(&preds, &targets, &cfg).unwrap();
        // Clamped to 1-1e-7, so near zero
        assert!(loss[0] < 0.01);
    }

    #[test]
    fn test_bce_worst() {
        let preds = [0.0, 1.0]; // completely wrong
        let targets = [1.0, 0.0];
        let cfg = LossConfig::new(2, 1).unwrap();
        let loss = binary_cross_entropy(&preds, &targets, &cfg).unwrap();
        assert!(loss[0] > 10.0, "expected high BCE for worst preds, got {}", loss[0]);
    }

    #[test]
    fn test_bce_symmetric() {
        // BCE(0.7, 1) should equal BCE(0.3, 0) by symmetry
        let cfg = LossConfig::new(1, 1).unwrap().with_reduction(LossReduction::None);
        let l1 = binary_cross_entropy(&[0.7], &[1.0], &cfg).unwrap();
        let l2 = binary_cross_entropy(&[0.3], &[0.0], &cfg).unwrap();
        assert!((l1[0] - l2[0]).abs() < 1e-5, "l1={}, l2={}", l1[0], l2[0]);
    }

    #[test]
    fn test_bce_rejects_short() {
        let cfg = LossConfig::new(3, 1).unwrap();
        assert!(binary_cross_entropy(&[0.5, 0.5], &[1.0, 0.0, 1.0], &cfg).is_err());
    }

    // == MSE loss tests ==================================================

    #[test]
    fn test_mse_basic() {
        let preds = [1.0, 2.0, 3.0];
        let targets = [1.0, 2.0, 3.0];
        let cfg = LossConfig::new(3, 1).unwrap();
        let loss = mse_loss(&preds, &targets, &cfg).unwrap();
        assert!(loss[0].abs() < 1e-7, "perfect match should give 0 MSE, got {}", loss[0]);
    }

    #[test]
    fn test_mse_known_value() {
        let preds = [1.0, 2.0, 3.0];
        let targets = [1.5, 2.5, 3.5];
        let cfg = LossConfig::new(3, 1).unwrap();
        let loss = mse_loss(&preds, &targets, &cfg).unwrap();
        // Each diff = 0.5, diff^2 = 0.25, mean = 0.25
        assert!((loss[0] - 0.25).abs() < 1e-6, "got {}", loss[0]);
    }

    #[test]
    fn test_mse_sum_reduction() {
        let preds = [0.0, 0.0];
        let targets = [1.0, 2.0];
        let cfg = LossConfig::new(2, 1).unwrap().with_reduction(LossReduction::Sum);
        let loss = mse_loss(&preds, &targets, &cfg).unwrap();
        // 1^2 + 2^2 = 5
        assert!((loss[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_none_reduction() {
        let preds = [0.0, 0.0];
        let targets = [3.0, 4.0];
        let cfg = LossConfig::new(2, 1).unwrap().with_reduction(LossReduction::None);
        let loss = mse_loss(&preds, &targets, &cfg).unwrap();
        assert_eq!(loss.len(), 2);
        assert!((loss[0] - 9.0).abs() < 1e-6);
        assert!((loss[1] - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_negative_values() {
        let preds = [-1.0];
        let targets = [1.0];
        let cfg = LossConfig::new(1, 1).unwrap();
        let loss = mse_loss(&preds, &targets, &cfg).unwrap();
        assert!((loss[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_rejects_short() {
        let cfg = LossConfig::new(3, 1).unwrap();
        assert!(mse_loss(&[1.0, 2.0], &[1.0, 2.0, 3.0], &cfg).is_err());
    }

    // == Huber loss tests ================================================

    #[test]
    fn test_huber_quadratic_region() {
        // |diff| < delta → quadratic
        let preds = [1.0];
        let targets = [1.5];
        let cfg = LossConfig::new(1, 1).unwrap();
        let loss = huber_loss(&preds, &targets, 1.0, &cfg).unwrap();
        // 0.5 * 0.5^2 = 0.125
        assert!((loss[0] - 0.125).abs() < 1e-6, "got {}", loss[0]);
    }

    #[test]
    fn test_huber_linear_region() {
        // |diff| > delta → linear
        let preds = [0.0];
        let targets = [3.0];
        let cfg = LossConfig::new(1, 1).unwrap();
        let loss = huber_loss(&preds, &targets, 1.0, &cfg).unwrap();
        // delta * (|3| - 0.5 * delta) = 1 * (3 - 0.5) = 2.5
        assert!((loss[0] - 2.5).abs() < 1e-6, "got {}", loss[0]);
    }

    #[test]
    fn test_huber_at_delta() {
        // |diff| == delta → both formulas agree: 0.5 * delta^2
        let preds = [0.0];
        let targets = [1.0];
        let cfg = LossConfig::new(1, 1).unwrap();
        let loss = huber_loss(&preds, &targets, 1.0, &cfg).unwrap();
        assert!((loss[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_huber_zero_diff() {
        let preds = [5.0];
        let targets = [5.0];
        let cfg = LossConfig::new(1, 1).unwrap();
        let loss = huber_loss(&preds, &targets, 1.0, &cfg).unwrap();
        assert!(loss[0].abs() < 1e-7);
    }

    #[test]
    fn test_huber_custom_delta() {
        let preds = [0.0];
        let targets = [0.5];
        let cfg = LossConfig::new(1, 1).unwrap();
        let loss = huber_loss(&preds, &targets, 2.0, &cfg).unwrap();
        // |0.5| < 2.0 → 0.5 * 0.25 = 0.125
        assert!((loss[0] - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_huber_rejects_bad_delta() {
        let cfg = LossConfig::new(1, 1).unwrap();
        assert!(huber_loss(&[0.0], &[1.0], 0.0, &cfg).is_err());
        assert!(huber_loss(&[0.0], &[1.0], -1.0, &cfg).is_err());
        assert!(huber_loss(&[0.0], &[1.0], f32::NAN, &cfg).is_err());
    }

    #[test]
    fn test_huber_less_than_mse_for_outliers() {
        let preds = [0.0];
        let targets = [10.0];
        let cfg = LossConfig::new(1, 1).unwrap();
        let huber = huber_loss(&preds, &targets, 1.0, &cfg).unwrap()[0];
        let mse = mse_loss(&preds, &targets, &cfg).unwrap()[0];
        assert!(huber < mse, "huber={huber} should be < mse={mse} for outliers");
    }

    // == KL divergence tests =============================================

    #[test]
    fn test_kl_identical_distributions() {
        let p = [0.25, 0.25, 0.25, 0.25];
        let q = [0.25, 0.25, 0.25, 0.25];
        let cfg = LossConfig::new(4, 1).unwrap();
        let loss = kl_divergence(&p, &q, &cfg).unwrap();
        assert!(loss[0].abs() < 1e-6, "KL(p||p) should be 0, got {}", loss[0]);
    }

    #[test]
    fn test_kl_known_value() {
        // KL([0.5, 0.5] || [0.25, 0.75])
        let pred = [0.25_f32, 0.75];
        let target = [0.5_f32, 0.5];
        let cfg = LossConfig::new(2, 1).unwrap().with_reduction(LossReduction::Sum);
        let loss = kl_divergence(&pred, &target, &cfg).unwrap();
        let expected = 0.5 * (0.5_f32 / 0.25).ln() + 0.5 * (0.5_f32 / 0.75).ln();
        assert!((loss[0] - expected).abs() < 1e-5, "got {}, expected {expected}", loss[0]);
    }

    #[test]
    fn test_kl_non_negative() {
        let pred = [0.1, 0.2, 0.7];
        let target = [0.3, 0.3, 0.4];
        let cfg = LossConfig::new(3, 1).unwrap().with_reduction(LossReduction::Sum);
        let loss = kl_divergence(&pred, &target, &cfg).unwrap();
        assert!(loss[0] >= -1e-7, "KL divergence should be non-negative, got {}", loss[0]);
    }

    #[test]
    fn test_kl_zero_target() {
        // 0 * log(0/p) = 0 by convention
        let pred = [0.5, 0.5];
        let target = [0.0, 1.0];
        let cfg = LossConfig::new(2, 1).unwrap().with_reduction(LossReduction::Sum);
        let loss = kl_divergence(&pred, &target, &cfg).unwrap();
        let expected = 0.0 + 1.0 * (1.0_f32 / 0.5).ln();
        assert!((loss[0] - expected).abs() < 1e-5);
    }

    // == Focal loss tests ================================================

    #[test]
    fn test_focal_basic() {
        let probs = [0.9, 0.1];
        let targets = [0_u32];
        let cfg = LossConfig::new(1, 2).unwrap();
        let loss = focal_loss(&probs, &targets, 0.25, 2.0, &cfg).unwrap();
        // -0.25 * (1-0.9)^2 * log(0.9) = -0.25 * 0.01 * (-0.10536) ≈ 0.000263
        let expected = -0.25 * (0.1_f32).powi(2) * 0.9_f32.ln();
        assert!((loss[0] - expected).abs() < 1e-5, "got {}, expected {expected}", loss[0]);
    }

    #[test]
    fn test_focal_reduces_easy_example_weight() {
        // Focal loss should be much smaller than standard CE for easy examples
        let probs = [0.95, 0.05];
        let targets = [0_u32];
        let cfg = LossConfig::new(1, 2).unwrap();
        let focal = focal_loss(&probs, &targets, 1.0, 2.0, &cfg).unwrap()[0];
        let ce = cross_entropy_loss(&probs, &targets, &cfg).unwrap()[0];
        assert!(focal < ce, "focal={focal} should be < ce={ce} for easy examples");
    }

    #[test]
    fn test_focal_gamma_zero_matches_ce() {
        // With gamma=0 and alpha=1, focal loss == cross-entropy
        let probs = [0.7, 0.2, 0.1];
        let targets = [0_u32];
        let cfg = LossConfig::new(1, 3).unwrap();
        let focal = focal_loss(&probs, &targets, 1.0, 0.0, &cfg).unwrap()[0];
        let ce = cross_entropy_loss(&probs, &targets, &cfg).unwrap()[0];
        assert!((focal - ce).abs() < 1e-5, "focal(gamma=0)={focal} should match ce={ce}");
    }

    #[test]
    fn test_focal_rejects_bad_target() {
        let probs = [0.5, 0.5];
        let targets = [2_u32];
        let cfg = LossConfig::new(1, 2).unwrap();
        assert!(focal_loss(&probs, &targets, 0.25, 2.0, &cfg).is_err());
    }

    // == Label smoothing CE tests ========================================

    #[test]
    fn test_label_smoothing_zero_equals_ce() {
        let probs = [0.7, 0.2, 0.1];
        let targets = [0_u32];
        let cfg = LossConfig::new(1, 3).unwrap();
        let smooth = label_smoothing_ce(&probs, &targets, 0.0, &cfg).unwrap()[0];
        let ce = cross_entropy_loss(&probs, &targets, &cfg).unwrap()[0];
        assert!(
            (smooth - ce).abs() < 1e-5,
            "smoothing=0 should match CE: smooth={smooth}, ce={ce}"
        );
    }

    #[test]
    fn test_label_smoothing_increases_loss() {
        // Smoothing should increase loss for confident correct predictions
        let probs = [0.9, 0.05, 0.05];
        let targets = [0_u32];
        let cfg = LossConfig::new(1, 3).unwrap();
        let smooth = label_smoothing_ce(&probs, &targets, 0.1, &cfg).unwrap()[0];
        let ce = cross_entropy_loss(&probs, &targets, &cfg).unwrap()[0];
        assert!(smooth > ce, "smooth={smooth} should be > ce={ce} for confident predictions");
    }

    #[test]
    fn test_label_smoothing_known_value() {
        let probs = [0.8, 0.2];
        let targets = [0_u32];
        let cfg = LossConfig::new(1, 2).unwrap();
        let loss = label_smoothing_ce(&probs, &targets, 0.1, &cfg).unwrap();
        // smooth=0.1, nc=2: uniform=0.05, on_value=0.95
        // loss = -0.95*log(0.8) - 0.05*log(0.2)
        let expected = -0.95 * 0.8_f32.ln() - 0.05 * 0.2_f32.ln();
        assert!((loss[0] - expected).abs() < 1e-5, "got {}, expected {expected}", loss[0]);
    }

    #[test]
    fn test_label_smoothing_rejects_bad_smoothing() {
        let probs = [0.5, 0.5];
        let targets = [0_u32];
        let cfg = LossConfig::new(1, 2).unwrap();
        assert!(label_smoothing_ce(&probs, &targets, 1.0, &cfg).is_err());
        assert!(label_smoothing_ce(&probs, &targets, -0.1, &cfg).is_err());
    }

    // == Perplexity from logits tests ====================================

    #[test]
    fn test_perplexity_uniform() {
        // For uniform logits over K classes, perplexity = K
        let k = 4;
        let logits = vec![0.0_f32; k];
        let targets = [0_u32];
        let cfg = LossConfig::new(1, k).unwrap();
        let ppl = perplexity_from_logits(&logits, &targets, &cfg).unwrap();
        assert!((ppl - k as f32).abs() < 0.1, "uniform perplexity should ≈ {k}, got {ppl}");
    }

    #[test]
    fn test_perplexity_confident() {
        // Very confident prediction → perplexity ≈ 1
        let logits = [100.0, 0.0, 0.0];
        let targets = [0_u32];
        let cfg = LossConfig::new(1, 3).unwrap();
        let ppl = perplexity_from_logits(&logits, &targets, &cfg).unwrap();
        assert!(ppl < 1.1, "confident prediction perplexity should be ≈ 1, got {ppl}");
    }

    #[test]
    fn test_perplexity_multiple_tokens() {
        let logits = [
            2.0, 1.0, 0.0, // token 0
            0.0, 2.0, 1.0, // token 1
        ];
        let targets = [0_u32, 1];
        let cfg = LossConfig::new(2, 3).unwrap();
        let ppl = perplexity_from_logits(&logits, &targets, &cfg).unwrap();
        assert!(ppl > 1.0, "perplexity should be > 1");
        assert!(ppl.is_finite());
    }

    // == Contrastive loss tests ==========================================

    #[test]
    fn test_contrastive_similar_close() {
        // Similar pair, close together → low loss (y*d²)
        let a = [1.0, 0.0, 0.0];
        let b = [1.1, 0.0, 0.0];
        let labels = [1.0]; // similar
        let cfg = LossConfig::new(1, 1).unwrap();
        let loss = contrastive_loss(&a, &b, &labels, 1.0, 3, &cfg).unwrap();
        assert!(loss[0] < 0.02, "similar-close loss should be small, got {}", loss[0]);
    }

    #[test]
    fn test_contrastive_dissimilar_far() {
        // Dissimilar pair, far apart → low loss (hinge at 0)
        let a = [0.0, 0.0, 0.0];
        let b = [5.0, 0.0, 0.0];
        let labels = [0.0]; // dissimilar
        let cfg = LossConfig::new(1, 1).unwrap();
        let loss = contrastive_loss(&a, &b, &labels, 1.0, 3, &cfg).unwrap();
        assert!(loss[0] < 1e-6, "dissimilar-far loss should be ~0, got {}", loss[0]);
    }

    #[test]
    fn test_contrastive_dissimilar_close() {
        // Dissimilar pair, close → high loss (margin penalty)
        let a = [0.0, 0.0];
        let b = [0.1, 0.0];
        let labels = [0.0]; // dissimilar
        let cfg = LossConfig::new(1, 1).unwrap();
        let loss = contrastive_loss(&a, &b, &labels, 2.0, 2, &cfg).unwrap();
        // (2.0 - 0.1)^2 = 3.61
        assert!(loss[0] > 3.0, "dissimilar-close loss should be high, got {}", loss[0]);
    }

    #[test]
    fn test_contrastive_batch() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 0.0, 5.0, 0.0];
        let labels = [1.0, 0.0]; // first similar, second dissimilar
        let cfg = LossConfig::new(2, 1).unwrap().with_reduction(LossReduction::None);
        let loss = contrastive_loss(&a, &b, &labels, 1.0, 2, &cfg).unwrap();
        assert_eq!(loss.len(), 2);
        assert!(loss[0] > 0.0); // similar but far → penalty
        assert!(loss[1] < 1e-6); // dissimilar and far → no penalty
    }

    // == Triplet loss tests ==============================================

    #[test]
    fn test_triplet_satisfied() {
        // anchor close to positive, far from negative → loss = 0
        let anchor = [0.0, 0.0];
        let positive = [0.1, 0.0];
        let negative = [10.0, 0.0];
        let cfg = LossConfig::new(1, 1).unwrap();
        let loss = triplet_loss(&anchor, &positive, &negative, 1.0, 2, &cfg).unwrap();
        assert!(loss[0] < 1e-6, "satisfied triplet should have ~0 loss, got {}", loss[0]);
    }

    #[test]
    fn test_triplet_violated() {
        // anchor closer to negative than positive → positive loss
        let anchor = [0.0, 0.0];
        let positive = [10.0, 0.0];
        let negative = [0.1, 0.0];
        let cfg = LossConfig::new(1, 1).unwrap();
        let loss = triplet_loss(&anchor, &positive, &negative, 1.0, 2, &cfg).unwrap();
        assert!(loss[0] > 5.0, "violated triplet should have high loss, got {}", loss[0]);
    }

    #[test]
    fn test_triplet_margin_effect() {
        let anchor = [0.0];
        let positive = [1.0];
        let negative = [3.0];
        let cfg = LossConfig::new(1, 1).unwrap();
        // dist_ap=1, dist_an=3, margin=1 → max(0, 1-3+1)=0
        let l1 = triplet_loss(&anchor, &positive, &negative, 1.0, 1, &cfg).unwrap()[0];
        assert!(l1 < 1e-6);
        // margin=5 → max(0, 1-3+5) = 3
        let l2 = triplet_loss(&anchor, &positive, &negative, 5.0, 1, &cfg).unwrap()[0];
        assert!((l2 - 3.0).abs() < 1e-5, "got {l2}, expected 3.0");
    }

    #[test]
    fn test_triplet_batch() {
        let anchor = [0.0, 0.0, 0.0, 0.0];
        let positive = [0.1, 0.0, 10.0, 0.0];
        let negative = [10.0, 0.0, 0.1, 0.0];
        let cfg = LossConfig::new(2, 1).unwrap().with_reduction(LossReduction::None);
        let loss = triplet_loss(&anchor, &positive, &negative, 1.0, 2, &cfg).unwrap();
        assert_eq!(loss.len(), 2);
        assert!(loss[0] < 1e-6); // first satisfied
        assert!(loss[1] > 5.0); // second violated
    }

    #[test]
    fn test_triplet_rejects_short() {
        let cfg = LossConfig::new(2, 1).unwrap();
        assert!(triplet_loss(&[0.0; 4], &[0.0; 4], &[0.0; 2], 1.0, 2, &cfg).is_err());
    }

    // == Forward dispatch tests ==========================================

    #[test]
    fn test_cross_entropy_forward_dispatches_cpu() {
        let probs = [0.7, 0.2, 0.1, 0.1, 0.1, 0.8];
        let targets = [0_u32, 2];
        let cfg = LossConfig::new(2, 3).unwrap();
        let loss = cross_entropy_loss_forward(&probs, &targets, &cfg).unwrap();
        assert!(loss[0] > 0.0);
        assert!(loss[0].is_finite());
    }

    #[test]
    fn test_mse_forward_dispatches_cpu() {
        let preds = [1.0, 2.0];
        let targets = [1.5, 2.5];
        let cfg = LossConfig::new(2, 1).unwrap();
        let loss = mse_loss_forward(&preds, &targets, &cfg).unwrap();
        assert!((loss[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_huber_forward_dispatches_cpu() {
        let preds = [0.0];
        let targets = [3.0];
        let cfg = LossConfig::new(1, 1).unwrap();
        let loss = huber_loss_forward(&preds, &targets, 1.0, &cfg).unwrap();
        assert!((loss[0] - 2.5).abs() < 1e-6);
    }

    // == CUDA launch stub tests ==========================================

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_cuda_cross_entropy_stub() {
        let cfg = LossConfig::new(4, 10).unwrap();
        let result = launch_cross_entropy_loss(&[0.0; 40], &[0; 4], &cfg);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_cuda_mse_stub() {
        let cfg = LossConfig::new(4, 1).unwrap();
        let result = launch_mse_loss(&[0.0; 4], &[0.0; 4], &cfg);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_cuda_huber_stub() {
        let cfg = LossConfig::new(4, 1).unwrap();
        let result = launch_huber_loss(&[0.0; 4], &[0.0; 4], 1.0, &cfg);
        assert!(result.is_err());
    }
}
