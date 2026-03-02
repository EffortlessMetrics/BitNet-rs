//! CUDA embedding lookup and operations kernel module.
//!
//! Extends the basic embedding lookup from [`super::embedding`] with advanced
//! operations commonly needed in transformer inference:
//!
//! - **Sparse lookups** and **embedding bags** (Sum/Mean/Max pooling)
//! - **Sinusoidal position embeddings** (Vaswani et al. 2017)
//! - **Learned position embeddings** with table-based lookup
//! - **Embedding norm clamping** (max-norm re-projection)
//! - **Sparse gradient accumulation** for backward pass
//! - **Fused embedding + LayerNorm** to reduce memory round-trips
//!
//! All GPU kernels are gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! Every operation has a CPU fallback that is used in tests and on non-GPU
//! hardware.

use bitnet_common::{KernelError, Result};

// ───────────────────────────────────────────────────────────────────
// Configuration types
// ───────────────────────────────────────────────────────────────────

/// Full embedding configuration covering vocabulary, dimensions, and
/// optional norm / sparsity settings.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Number of entries (rows) in the embedding table.
    pub vocab_size: usize,
    /// Dimensionality of each embedding vector.
    pub embed_dim: usize,
    /// Token index whose embedding is always zero.
    pub padding_idx: Option<u32>,
    /// If set, embeddings with norm > `max_norm` are re-normalised.
    pub max_norm: Option<f32>,
    /// Norm type for `max_norm` clamping (default 2.0 = L2).
    pub norm_type: f32,
    /// Hint that gradients will be sparse (advisory only on CPU).
    pub sparse: bool,
}

impl EmbeddingConfig {
    /// Create a new configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when `vocab_size` or `embed_dim` is zero.
    pub fn new(vocab_size: usize, embed_dim: usize) -> Result<Self> {
        if vocab_size == 0 || embed_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "embedding dimensions must be non-zero: vocab_size={vocab_size}, embed_dim={embed_dim}"
                ),
            }
            .into());
        }
        Ok(Self {
            vocab_size,
            embed_dim,
            padding_idx: None,
            max_norm: None,
            norm_type: 2.0,
            sparse: false,
        })
    }

    /// Set the padding index.
    pub fn with_padding_idx(mut self, idx: u32) -> Self {
        self.padding_idx = Some(idx);
        self
    }

    /// Set the max-norm clamping threshold.
    pub fn with_max_norm(mut self, max_norm: f32) -> Self {
        self.max_norm = Some(max_norm);
        self
    }

    /// Set the norm type used for max-norm clamping.
    pub fn with_norm_type(mut self, norm_type: f32) -> Self {
        self.norm_type = norm_type;
        self
    }

    /// Mark the embedding as sparse.
    pub fn with_sparse(mut self, sparse: bool) -> Self {
        self.sparse = sparse;
        self
    }
}

/// Dense weight matrix for an embedding table (row-major, `vocab_size × embed_dim`).
#[derive(Debug, Clone)]
pub struct EmbeddingTable {
    /// Flat weight buffer in row-major order.
    pub weights: Vec<f32>,
    /// Number of vocabulary entries.
    pub vocab_size: usize,
    /// Dimensionality of each vector.
    pub embed_dim: usize,
}

impl EmbeddingTable {
    /// Create a new table from a flat weight buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if `weights.len() != vocab_size * embed_dim`.
    pub fn new(weights: Vec<f32>, vocab_size: usize, embed_dim: usize) -> Result<Self> {
        if weights.len() != vocab_size * embed_dim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "weights length {} != vocab_size ({}) * embed_dim ({})",
                    weights.len(),
                    vocab_size,
                    embed_dim,
                ),
            }
            .into());
        }
        Ok(Self { weights, vocab_size, embed_dim })
    }

    /// Return the embedding vector for `idx`.
    ///
    /// # Errors
    ///
    /// Returns an error if `idx >= vocab_size`.
    pub fn row(&self, idx: usize) -> Result<&[f32]> {
        if idx >= self.vocab_size {
            return Err(KernelError::InvalidArguments {
                reason: format!("index {idx} out of bounds for vocab_size {}", self.vocab_size),
            }
            .into());
        }
        let start = idx * self.embed_dim;
        Ok(&self.weights[start..start + self.embed_dim])
    }
}

/// Pooling mode for [`embedding_bag`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingBagMode {
    /// Sum all looked-up vectors per bag.
    Sum,
    /// Mean of all looked-up vectors per bag.
    Mean,
    /// Element-wise maximum across looked-up vectors per bag.
    Max,
}

// ───────────────────────────────────────────────────────────────────
// CUDA kernel sources
// ───────────────────────────────────────────────────────────────────

/// CUDA kernel for batched embedding lookup.
///
/// Grid: `(total_tokens, 1, 1)`, Block: `(min(embed_dim, 1024), 1, 1)`.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const EMBEDDING_OPS_LOOKUP_KERNEL_SRC: &str = r#"
extern "C" __global__ void embedding_ops_lookup_f32(
    const float* __restrict__ table,
    const unsigned int* __restrict__ indices,
    float* __restrict__ output,
    int vocab_size,
    int embed_dim,
    int total_tokens,
    int padding_idx)
{
    int token_pos = blockIdx.x;
    if (token_pos >= total_tokens) return;
    unsigned int idx = indices[token_pos];
    float* out_row = output + token_pos * embed_dim;
    if ((int)idx == padding_idx) {
        for (int d = threadIdx.x; d < embed_dim; d += blockDim.x)
            out_row[d] = 0.0f;
        return;
    }
    if (idx >= (unsigned int)vocab_size)
        idx = (unsigned int)(vocab_size - 1);
    const float* src = table + idx * embed_dim;
    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x)
        out_row[d] = src[d];
}
"#;

/// CUDA kernel for embedding bag (Sum mode).
///
/// Grid: `(num_bags, 1, 1)`, Block: `(min(embed_dim, 1024), 1, 1)`.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const EMBEDDING_BAG_SUM_KERNEL_SRC: &str = r#"
extern "C" __global__ void embedding_bag_sum_f32(
    const float* __restrict__ table,
    const unsigned int* __restrict__ indices,
    const int* __restrict__ offsets,
    float* __restrict__ output,
    int vocab_size,
    int embed_dim,
    int num_bags)
{
    int bag = blockIdx.x;
    if (bag >= num_bags) return;
    int start = offsets[bag];
    int end = (bag + 1 < num_bags) ? offsets[bag + 1] : start;
    float* out_row = output + bag * embed_dim;
    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = start; i < end; i++) {
            unsigned int idx = indices[i];
            if (idx < (unsigned int)vocab_size)
                acc += table[idx * embed_dim + d];
        }
        out_row[d] = acc;
    }
}
"#;

/// CUDA kernel for sinusoidal position embedding generation.
///
/// Grid: `(num_positions, 1, 1)`, Block: `(min(embed_dim/2, 1024), 1, 1)`.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const SINUSOIDAL_POSITION_KERNEL_SRC: &str = r#"
extern "C" __global__ void sinusoidal_position_f32(
    const int* __restrict__ positions,
    float* __restrict__ output,
    int num_positions,
    int embed_dim)
{
    int pos_idx = blockIdx.x;
    if (pos_idx >= num_positions) return;
    int pos = positions[pos_idx];
    float* out_row = output + pos_idx * embed_dim;
    int half_dim = embed_dim / 2;
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        float freq = 1.0f / powf(10000.0f, (float)(2 * i) / (float)embed_dim);
        float angle = (float)pos * freq;
        out_row[i] = sinf(angle);
        out_row[i + half_dim] = cosf(angle);
    }
}
"#;

/// CUDA kernel for embedding max-norm clamping.
///
/// Grid: `(num_vectors, 1, 1)`, Block: `(min(embed_dim, 1024), 1, 1)`.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const EMBEDDING_NORM_KERNEL_SRC: &str = r#"
extern "C" __global__ void embedding_norm_f32(
    float* __restrict__ embeddings,
    int num_vectors,
    int embed_dim,
    float max_norm,
    float norm_type)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= num_vectors) return;
    float* vec = embeddings + vec_idx * embed_dim;

    // Compute p-norm using shared memory reduction.
    extern __shared__ float sdata[];
    float local_sum = 0.0f;
    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x)
        local_sum += powf(fabsf(vec[d]), norm_type);
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float norm_val = powf(sdata[0], 1.0f / norm_type);
    if (norm_val <= max_norm) return;
    float scale = max_norm / (norm_val + 1e-7f);
    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x)
        vec[d] *= scale;
}
"#;

/// CUDA kernel for fused embedding lookup + LayerNorm.
///
/// Grid: `(num_tokens, 1, 1)`, Block: `(min(embed_dim, 1024), 1, 1)`.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const FUSED_EMBEDDING_LAYERNORM_KERNEL_SRC: &str = r#"
extern "C" __global__ void fused_embedding_layernorm_f32(
    const float* __restrict__ table,
    const unsigned int* __restrict__ indices,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int vocab_size,
    int embed_dim,
    int num_tokens,
    float eps)
{
    int tok = blockIdx.x;
    if (tok >= num_tokens) return;
    unsigned int idx = indices[tok];
    if (idx >= (unsigned int)vocab_size) idx = (unsigned int)(vocab_size - 1);
    const float* src = table + idx * embed_dim;
    float* out = output + tok * embed_dim;

    extern __shared__ float sdata[];
    // Pass 1: compute mean
    float local_sum = 0.0f;
    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x)
        local_sum += src[d];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)embed_dim;
    __syncthreads();

    // Pass 2: compute variance
    float local_var = 0.0f;
    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        float diff = src[d] - mean;
        local_var += diff * diff;
    }
    sdata[threadIdx.x] = local_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / (float)embed_dim + eps);

    // Pass 3: normalize and apply affine
    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x)
        out[d] = (src[d] - mean) * inv_std * gamma[d] + beta[d];
}
"#;

/// CUDA kernel for sparse gradient accumulation.
///
/// Grid: `(num_indices, 1, 1)`, Block: `(min(embed_dim, 1024), 1, 1)`.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const EMBEDDING_GRADIENT_KERNEL_SRC: &str = r#"
extern "C" __global__ void embedding_gradient_f32(
    const float* __restrict__ grad_output,
    const unsigned int* __restrict__ indices,
    float* __restrict__ grad_table,
    int embed_dim,
    int num_indices)
{
    int i = blockIdx.x;
    if (i >= num_indices) return;
    unsigned int idx = indices[i];
    const float* grad_row = grad_output + i * embed_dim;
    float* table_row = grad_table + idx * embed_dim;
    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x)
        atomicAdd(&table_row[d], grad_row[d]);
}
"#;

// ───────────────────────────────────────────────────────────────────
// CPU fallback: embedding lookup (batched)
// ───────────────────────────────────────────────────────────────────

/// Batched embedding lookup (CPU).
///
/// `indices` has shape `[batch * seq_len]`. Returns a flat buffer of shape
/// `[batch * seq_len, embed_dim]`.
///
/// # Errors
///
/// Returns an error if any index exceeds `vocab_size` or buffer sizes
/// are inconsistent.
pub fn embedding_lookup(
    table: &EmbeddingTable,
    indices: &[u32],
    config: &EmbeddingConfig,
) -> Result<Vec<f32>> {
    let dim = config.embed_dim;
    if table.embed_dim != dim || table.vocab_size != config.vocab_size {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "config/table mismatch: config=({}, {}), table=({}, {})",
                config.vocab_size, config.embed_dim, table.vocab_size, table.embed_dim,
            ),
        }
        .into());
    }
    let n = indices.len();
    let mut output = vec![0.0_f32; n * dim];
    for (i, &idx) in indices.iter().enumerate() {
        if config.padding_idx == Some(idx) {
            continue; // already zeroed
        }
        if (idx as usize) >= config.vocab_size {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "embedding index {} out of bounds for vocab_size {}",
                    idx, config.vocab_size,
                ),
            }
            .into());
        }
        let src_start = idx as usize * dim;
        let dst_start = i * dim;
        output[dst_start..dst_start + dim]
            .copy_from_slice(&table.weights[src_start..src_start + dim]);
    }
    if let Some(max_norm) = config.max_norm {
        embedding_norm(&mut output, n, dim, max_norm, config.norm_type)?;
    }
    Ok(output)
}

// ───────────────────────────────────────────────────────────────────
// CPU fallback: sparse embedding lookup
// ───────────────────────────────────────────────────────────────────

/// Sparse embedding lookup (CPU).
///
/// `sparse_indices` contains the non-zero token IDs. `offsets` marks
/// the start of each bag/row in `sparse_indices`. Returns a buffer of
/// shape `[num_bags, embed_dim]` where each row is the sum of the
/// looked-up vectors for that bag.
///
/// # Errors
///
/// Returns an error if any index exceeds `vocab_size` or offsets are
/// inconsistent.
pub fn embedding_lookup_sparse(
    table: &EmbeddingTable,
    sparse_indices: &[u32],
    offsets: &[usize],
    config: &EmbeddingConfig,
) -> Result<Vec<f32>> {
    if offsets.is_empty() {
        return Ok(vec![]);
    }
    let dim = config.embed_dim;
    let num_bags = offsets.len();
    let mut output = vec![0.0_f32; num_bags * dim];
    for bag in 0..num_bags {
        let start = offsets[bag];
        let end = if bag + 1 < num_bags { offsets[bag + 1] } else { sparse_indices.len() };
        if start > end || end > sparse_indices.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "invalid offsets: bag {bag} start={start} end={end} total={}",
                    sparse_indices.len(),
                ),
            }
            .into());
        }
        for &idx in &sparse_indices[start..end] {
            if (idx as usize) >= config.vocab_size {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "sparse index {} out of bounds for vocab_size {}",
                        idx, config.vocab_size,
                    ),
                }
                .into());
            }
            let src = idx as usize * dim;
            let dst = bag * dim;
            for j in 0..dim {
                output[dst + j] += table.weights[src + j];
            }
        }
    }
    Ok(output)
}

// ───────────────────────────────────────────────────────────────────
// CPU fallback: embedding bag
// ───────────────────────────────────────────────────────────────────

/// Embedding bag with Sum / Mean / Max pooling (CPU).
///
/// `indices` is a flat list of token IDs. `offsets` marks the start of
/// each bag. Returns `[num_bags, embed_dim]`.
///
/// # Errors
///
/// Returns an error if any index is out of bounds or offsets are invalid.
pub fn embedding_bag(
    table: &EmbeddingTable,
    indices: &[u32],
    offsets: &[usize],
    mode: EmbeddingBagMode,
    config: &EmbeddingConfig,
) -> Result<Vec<f32>> {
    if offsets.is_empty() {
        return Ok(vec![]);
    }
    let dim = config.embed_dim;
    let num_bags = offsets.len();
    let mut output = vec![f32::NEG_INFINITY; num_bags * dim];
    if mode != EmbeddingBagMode::Max {
        output.fill(0.0);
    }

    for bag in 0..num_bags {
        let start = offsets[bag];
        let end = if bag + 1 < num_bags { offsets[bag + 1] } else { indices.len() };
        if start > end || end > indices.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "invalid offsets: bag {bag} start={start} end={end} total={}",
                    indices.len(),
                ),
            }
            .into());
        }
        let count = end - start;
        let dst = bag * dim;
        for &idx in &indices[start..end] {
            if (idx as usize) >= config.vocab_size {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "embedding_bag index {} out of bounds for vocab_size {}",
                        idx, config.vocab_size,
                    ),
                }
                .into());
            }
            let src = idx as usize * dim;
            match mode {
                EmbeddingBagMode::Sum | EmbeddingBagMode::Mean => {
                    for j in 0..dim {
                        output[dst + j] += table.weights[src + j];
                    }
                }
                EmbeddingBagMode::Max => {
                    for j in 0..dim {
                        output[dst + j] = output[dst + j].max(table.weights[src + j]);
                    }
                }
            }
        }
        if mode == EmbeddingBagMode::Mean && count > 0 {
            let inv = 1.0 / count as f32;
            for j in 0..dim {
                output[dst + j] *= inv;
            }
        }
        // For Max mode with empty bags, replace -inf with 0.
        if mode == EmbeddingBagMode::Max && count == 0 {
            for j in 0..dim {
                output[dst + j] = 0.0;
            }
        }
    }
    Ok(output)
}

// ───────────────────────────────────────────────────────────────────
// CPU fallback: position embeddings
// ───────────────────────────────────────────────────────────────────

/// Create a learned position embedding table (CPU).
///
/// Returns a zero-initialised buffer of shape `[max_positions, embed_dim]`.
/// The caller is expected to fill it with trained weights or pass it to
/// [`learned_position_embedding`] for lookup.
pub fn position_embedding(max_positions: usize, embed_dim: usize) -> Result<Vec<f32>> {
    if max_positions == 0 || embed_dim == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "position embedding dims must be non-zero: max_positions={max_positions}, embed_dim={embed_dim}"
            ),
        }
        .into());
    }
    Ok(vec![0.0_f32; max_positions * embed_dim])
}

/// Learned position embedding lookup (CPU).
///
/// Gathers rows from `table` for each position in `positions`.
/// Returns `[positions.len(), embed_dim]`.
///
/// # Errors
///
/// Returns an error if any position index exceeds the table row count.
pub fn learned_position_embedding(
    table: &[f32],
    positions: &[u32],
    max_positions: usize,
    embed_dim: usize,
) -> Result<Vec<f32>> {
    let n = positions.len();
    let mut output = vec![0.0_f32; n * embed_dim];
    for (i, &pos) in positions.iter().enumerate() {
        if (pos as usize) >= max_positions {
            return Err(KernelError::InvalidArguments {
                reason: format!("position {pos} out of bounds for max_positions {max_positions}"),
            }
            .into());
        }
        let src = pos as usize * embed_dim;
        let dst = i * embed_dim;
        output[dst..dst + embed_dim].copy_from_slice(&table[src..src + embed_dim]);
    }
    Ok(output)
}

/// Sinusoidal position embedding (Vaswani et al. 2017) (CPU).
///
/// For each position `p` and dimension `i`:
///   `PE(p, 2i)   = sin(p / 10000^(2i/d))`
///   `PE(p, 2i+1) = cos(p / 10000^(2i/d))`
///
/// Returns `[positions.len(), embed_dim]`.
///
/// # Errors
///
/// Returns an error if `embed_dim` is zero.
pub fn sinusoidal_position_embedding(positions: &[u32], embed_dim: usize) -> Result<Vec<f32>> {
    if embed_dim == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "embed_dim must be non-zero for sinusoidal embeddings".into(),
        }
        .into());
    }
    let n = positions.len();
    let half = embed_dim / 2;
    let mut output = vec![0.0_f32; n * embed_dim];
    for (i, &pos) in positions.iter().enumerate() {
        let base = i * embed_dim;
        for j in 0..half {
            let freq = 1.0_f32 / (10000.0_f32.powf(2.0 * j as f32 / embed_dim as f32));
            let angle = pos as f32 * freq;
            output[base + j] = angle.sin();
            output[base + half + j] = angle.cos();
        }
        // For odd embed_dim the last element stays zero.
    }
    Ok(output)
}

// ───────────────────────────────────────────────────────────────────
// CPU fallback: embedding norm
// ───────────────────────────────────────────────────────────────────

/// Clamp embedding vectors to `max_norm` using the given p-norm (in-place).
///
/// # Errors
///
/// Returns an error if the buffer length is not a multiple of `embed_dim`.
pub fn embedding_norm(
    embeddings: &mut [f32],
    num_vectors: usize,
    embed_dim: usize,
    max_norm: f32,
    norm_type: f32,
) -> Result<()> {
    if embed_dim == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "embed_dim must be non-zero for norm clamping".into(),
        }
        .into());
    }
    if embeddings.len() < num_vectors * embed_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "embeddings length {} < num_vectors ({}) * embed_dim ({})",
                embeddings.len(),
                num_vectors,
                embed_dim,
            ),
        }
        .into());
    }
    for v in 0..num_vectors {
        let start = v * embed_dim;
        let end = start + embed_dim;
        let slice = &mut embeddings[start..end];
        let norm: f32 =
            slice.iter().map(|x| x.abs().powf(norm_type)).sum::<f32>().powf(1.0 / norm_type);
        if norm > max_norm {
            let scale = max_norm / (norm + 1e-7);
            for x in slice.iter_mut() {
                *x *= scale;
            }
        }
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────
// CPU fallback: embedding gradient
// ───────────────────────────────────────────────────────────────────

/// Sparse gradient accumulation for embedding backward (CPU).
///
/// `grad_output` has shape `[num_indices, embed_dim]`.
/// Returns a sparse gradient table of shape `[vocab_size, embed_dim]`
/// where only rows indexed by `indices` may be non-zero.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn embedding_gradient(
    grad_output: &[f32],
    indices: &[u32],
    vocab_size: usize,
    embed_dim: usize,
) -> Result<Vec<f32>> {
    let n = indices.len();
    if grad_output.len() < n * embed_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "grad_output length {} < num_indices ({}) * embed_dim ({})",
                grad_output.len(),
                n,
                embed_dim,
            ),
        }
        .into());
    }
    let mut grad_table = vec![0.0_f32; vocab_size * embed_dim];
    for (i, &idx) in indices.iter().enumerate() {
        if (idx as usize) >= vocab_size {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "gradient index {} out of bounds for vocab_size {}",
                    idx, vocab_size,
                ),
            }
            .into());
        }
        let src = i * embed_dim;
        let dst = idx as usize * embed_dim;
        for j in 0..embed_dim {
            grad_table[dst + j] += grad_output[src + j];
        }
    }
    Ok(grad_table)
}

// ───────────────────────────────────────────────────────────────────
// CPU fallback: fused embedding + LayerNorm
// ───────────────────────────────────────────────────────────────────

/// Fused embedding lookup followed by LayerNorm (CPU).
///
/// Performs `LayerNorm(table[indices], gamma, beta)` without
/// materialising the intermediate embedding buffer.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn fused_embedding_layernorm(
    table: &EmbeddingTable,
    indices: &[u32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    config: &EmbeddingConfig,
) -> Result<Vec<f32>> {
    let dim = config.embed_dim;
    if gamma.len() < dim || beta.len() < dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "gamma/beta length ({}, {}) must be >= embed_dim ({})",
                gamma.len(),
                beta.len(),
                dim,
            ),
        }
        .into());
    }
    let n = indices.len();
    let mut output = vec![0.0_f32; n * dim];
    for (i, &idx) in indices.iter().enumerate() {
        if (idx as usize) >= config.vocab_size {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "fused_embedding_layernorm index {} out of bounds for vocab_size {}",
                    idx, config.vocab_size,
                ),
            }
            .into());
        }
        let src_start = idx as usize * dim;
        let src = &table.weights[src_start..src_start + dim];
        let dst_start = i * dim;

        // Compute mean.
        let mean: f32 = src.iter().sum::<f32>() / dim as f32;

        // Compute variance.
        let var: f32 = src.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / dim as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        // Normalise and apply affine transform.
        for j in 0..dim {
            output[dst_start + j] = (src[j] - mean) * inv_std * gamma[j] + beta[j];
        }
    }
    Ok(output)
}

// ───────────────────────────────────────────────────────────────────
// CUDA launch stubs
// ───────────────────────────────────────────────────────────────────

/// Launch stub for the batched embedding lookup CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is loaded.
pub fn launch_embedding_ops_lookup(
    _table: &[f32],
    _indices: &[u32],
    _output: &mut [f32],
    _config: &EmbeddingConfig,
) -> Result<()> {
    Err(KernelError::GpuError {
        reason: "embedding_ops lookup CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch stub for the embedding bag CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is loaded.
pub fn launch_embedding_bag(
    _table: &[f32],
    _indices: &[u32],
    _offsets: &[usize],
    _output: &mut [f32],
    _mode: EmbeddingBagMode,
    _config: &EmbeddingConfig,
) -> Result<()> {
    Err(KernelError::GpuError {
        reason: "embedding_bag CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch stub for sinusoidal position embedding generation.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is loaded.
pub fn launch_sinusoidal_position(
    _positions: &[u32],
    _output: &mut [f32],
    _embed_dim: usize,
) -> Result<()> {
    Err(KernelError::GpuError {
        reason: "sinusoidal position CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch stub for embedding norm clamping.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is loaded.
pub fn launch_embedding_norm(
    _embeddings: &mut [f32],
    _num_vectors: usize,
    _embed_dim: usize,
    _max_norm: f32,
    _norm_type: f32,
) -> Result<()> {
    Err(KernelError::GpuError {
        reason: "embedding_norm CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch stub for fused embedding + LayerNorm.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is loaded.
pub fn launch_fused_embedding_layernorm(
    _table: &[f32],
    _indices: &[u32],
    _gamma: &[f32],
    _beta: &[f32],
    _output: &mut [f32],
    _config: &EmbeddingConfig,
    _eps: f32,
) -> Result<()> {
    Err(KernelError::GpuError {
        reason: "fused_embedding_layernorm CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch stub for sparse gradient accumulation.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is loaded.
pub fn launch_embedding_gradient(
    _grad_output: &[f32],
    _indices: &[u32],
    _grad_table: &mut [f32],
    _embed_dim: usize,
) -> Result<()> {
    Err(KernelError::GpuError {
        reason: "embedding_gradient CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ───────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────

    /// 4-word vocab, dim=3 embedding table.
    fn sample_table() -> EmbeddingTable {
        EmbeddingTable::new(
            vec![
                1.0, 2.0, 3.0, // idx 0
                4.0, 5.0, 6.0, // idx 1
                7.0, 8.0, 9.0, // idx 2
                10.0, 11.0, 12.0, // idx 3
            ],
            4,
            3,
        )
        .unwrap()
    }

    fn sample_config() -> EmbeddingConfig {
        EmbeddingConfig::new(4, 3).unwrap()
    }

    // ── EmbeddingConfig tests ───────────────────────────────────

    #[test]
    fn config_new_valid() {
        let cfg = EmbeddingConfig::new(32000, 768).unwrap();
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.embed_dim, 768);
        assert!(cfg.padding_idx.is_none());
        assert!(cfg.max_norm.is_none());
        assert!((cfg.norm_type - 2.0).abs() < f32::EPSILON);
        assert!(!cfg.sparse);
    }

    #[test]
    fn config_rejects_zero_vocab() {
        assert!(EmbeddingConfig::new(0, 768).is_err());
    }

    #[test]
    fn config_rejects_zero_dim() {
        assert!(EmbeddingConfig::new(32000, 0).is_err());
    }

    #[test]
    fn config_with_padding_idx() {
        let cfg = EmbeddingConfig::new(100, 64).unwrap().with_padding_idx(0);
        assert_eq!(cfg.padding_idx, Some(0));
    }

    #[test]
    fn config_with_max_norm() {
        let cfg = EmbeddingConfig::new(100, 64).unwrap().with_max_norm(1.0);
        assert_eq!(cfg.max_norm, Some(1.0));
    }

    #[test]
    fn config_with_norm_type() {
        let cfg = EmbeddingConfig::new(100, 64).unwrap().with_norm_type(1.0);
        assert!((cfg.norm_type - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn config_with_sparse() {
        let cfg = EmbeddingConfig::new(100, 64).unwrap().with_sparse(true);
        assert!(cfg.sparse);
    }

    // ── EmbeddingTable tests ────────────────────────────────────

    #[test]
    fn table_new_valid() {
        let t = EmbeddingTable::new(vec![0.0; 12], 4, 3).unwrap();
        assert_eq!(t.vocab_size, 4);
        assert_eq!(t.embed_dim, 3);
    }

    #[test]
    fn table_new_size_mismatch() {
        assert!(EmbeddingTable::new(vec![0.0; 10], 4, 3).is_err());
    }

    #[test]
    fn table_row_valid() {
        let t = sample_table();
        assert_eq!(t.row(0).unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(t.row(3).unwrap(), &[10.0, 11.0, 12.0]);
    }

    #[test]
    fn table_row_oob() {
        let t = sample_table();
        assert!(t.row(4).is_err());
    }

    // ── embedding_lookup tests ──────────────────────────────────

    #[test]
    fn lookup_single() {
        let t = sample_table();
        let cfg = sample_config();
        let out = embedding_lookup(&t, &[2], &cfg).unwrap();
        assert_eq!(out, &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn lookup_multiple() {
        let t = sample_table();
        let cfg = sample_config();
        let out = embedding_lookup(&t, &[0, 3, 1], &cfg).unwrap();
        assert_eq!(out, &[1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn lookup_duplicate_ids() {
        let t = sample_table();
        let cfg = sample_config();
        let out = embedding_lookup(&t, &[1, 1, 1], &cfg).unwrap();
        assert_eq!(out, &[4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn lookup_with_padding() {
        let t = sample_table();
        let cfg = sample_config().with_padding_idx(1);
        let out = embedding_lookup(&t, &[0, 1, 2], &cfg).unwrap();
        assert_eq!(&out[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&out[3..6], &[0.0, 0.0, 0.0]);
        assert_eq!(&out[6..9], &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn lookup_oob() {
        let t = sample_table();
        let cfg = sample_config();
        assert!(embedding_lookup(&t, &[4], &cfg).is_err());
    }

    #[test]
    fn lookup_empty_indices() {
        let t = sample_table();
        let cfg = sample_config();
        let out = embedding_lookup(&t, &[], &cfg).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn lookup_config_mismatch() {
        let t = sample_table();
        let cfg = EmbeddingConfig::new(8, 3).unwrap(); // wrong vocab
        assert!(embedding_lookup(&t, &[0], &cfg).is_err());
    }

    #[test]
    fn lookup_all_padding() {
        let t = sample_table();
        let cfg = sample_config().with_padding_idx(0);
        let out = embedding_lookup(&t, &[0, 0, 0], &cfg).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn lookup_with_max_norm() {
        let t = sample_table();
        let cfg = sample_config().with_max_norm(1.0);
        let out = embedding_lookup(&t, &[3], &cfg).unwrap();
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm <= 1.0 + 1e-5, "norm should be clamped: {norm}");
    }

    // ── embedding_lookup_sparse tests ───────────────────────────

    #[test]
    fn sparse_lookup_single_bag() {
        let t = sample_table();
        let cfg = sample_config();
        let out = embedding_lookup_sparse(&t, &[0, 1], &[0], &cfg).unwrap();
        // sum of row 0 + row 1
        assert_eq!(out, &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn sparse_lookup_two_bags() {
        let t = sample_table();
        let cfg = sample_config();
        let out = embedding_lookup_sparse(&t, &[0, 1, 2, 3], &[0, 2], &cfg).unwrap();
        assert_eq!(&out[0..3], &[5.0, 7.0, 9.0]); // rows 0+1
        assert_eq!(&out[3..6], &[17.0, 19.0, 21.0]); // rows 2+3
    }

    #[test]
    fn sparse_lookup_empty_offsets() {
        let t = sample_table();
        let cfg = sample_config();
        let out = embedding_lookup_sparse(&t, &[], &[], &cfg).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn sparse_lookup_oob_index() {
        let t = sample_table();
        let cfg = sample_config();
        assert!(embedding_lookup_sparse(&t, &[10], &[0], &cfg).is_err());
    }

    #[test]
    fn sparse_lookup_empty_bag() {
        let t = sample_table();
        let cfg = sample_config();
        // Two bags, first empty, second has one entry
        let out = embedding_lookup_sparse(&t, &[2], &[0, 0], &cfg).unwrap();
        assert_eq!(&out[0..3], &[0.0, 0.0, 0.0]); // empty bag
        assert_eq!(&out[3..6], &[7.0, 8.0, 9.0]); // row 2
    }

    // ── embedding_bag tests ─────────────────────────────────────

    #[test]
    fn bag_sum_single() {
        let t = sample_table();
        let cfg = sample_config();
        let out = embedding_bag(&t, &[0, 1], &[0], EmbeddingBagMode::Sum, &cfg).unwrap();
        assert_eq!(out, &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn bag_mean_single() {
        let t = sample_table();
        let cfg = sample_config();
        let out = embedding_bag(&t, &[0, 1], &[0], EmbeddingBagMode::Mean, &cfg).unwrap();
        assert_eq!(out, &[2.5, 3.5, 4.5]);
    }

    #[test]
    fn bag_max_single() {
        let t = sample_table();
        let cfg = sample_config();
        let out = embedding_bag(&t, &[0, 1], &[0], EmbeddingBagMode::Max, &cfg).unwrap();
        assert_eq!(out, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn bag_sum_two_bags() {
        let t = sample_table();
        let cfg = sample_config();
        let out = embedding_bag(&t, &[0, 1, 2, 3], &[0, 2], EmbeddingBagMode::Sum, &cfg).unwrap();
        assert_eq!(&out[0..3], &[5.0, 7.0, 9.0]);
        assert_eq!(&out[3..6], &[17.0, 19.0, 21.0]);
    }

    #[test]
    fn bag_mean_two_bags() {
        let t = sample_table();
        let cfg = sample_config();
        let out = embedding_bag(&t, &[0, 1, 2, 3], &[0, 2], EmbeddingBagMode::Mean, &cfg).unwrap();
        assert_eq!(&out[0..3], &[2.5, 3.5, 4.5]);
        assert_eq!(&out[3..6], &[8.5, 9.5, 10.5]);
    }

    #[test]
    fn bag_max_two_bags() {
        let t = sample_table();
        let cfg = sample_config();
        let out = embedding_bag(&t, &[0, 1, 2, 3], &[0, 2], EmbeddingBagMode::Max, &cfg).unwrap();
        assert_eq!(&out[0..3], &[4.0, 5.0, 6.0]);
        assert_eq!(&out[3..6], &[10.0, 11.0, 12.0]);
    }

    #[test]
    fn bag_empty_offsets() {
        let t = sample_table();
        let cfg = sample_config();
        let out = embedding_bag(&t, &[], &[], EmbeddingBagMode::Sum, &cfg).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn bag_oob_index() {
        let t = sample_table();
        let cfg = sample_config();
        assert!(embedding_bag(&t, &[10], &[0], EmbeddingBagMode::Sum, &cfg).is_err());
    }

    #[test]
    fn bag_empty_bag_sum() {
        let t = sample_table();
        let cfg = sample_config();
        let out = embedding_bag(&t, &[1], &[0, 0], EmbeddingBagMode::Sum, &cfg).unwrap();
        assert_eq!(&out[0..3], &[0.0, 0.0, 0.0]); // empty bag
        assert_eq!(&out[3..6], &[4.0, 5.0, 6.0]); // row 1
    }

    #[test]
    fn bag_empty_bag_max() {
        let t = sample_table();
        let cfg = sample_config();
        let out = embedding_bag(&t, &[1], &[0, 0], EmbeddingBagMode::Max, &cfg).unwrap();
        assert_eq!(&out[0..3], &[0.0, 0.0, 0.0]); // empty bag → 0
        assert_eq!(&out[3..6], &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn bag_sum_single_element_bags() {
        let t = sample_table();
        let cfg = sample_config();
        let out =
            embedding_bag(&t, &[0, 1, 2, 3], &[0, 1, 2, 3], EmbeddingBagMode::Sum, &cfg).unwrap();
        assert_eq!(&out[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&out[3..6], &[4.0, 5.0, 6.0]);
        assert_eq!(&out[6..9], &[7.0, 8.0, 9.0]);
        assert_eq!(&out[9..12], &[10.0, 11.0, 12.0]);
    }

    #[test]
    fn bag_mean_single_element_equals_sum() {
        let t = sample_table();
        let cfg = sample_config();
        let sum =
            embedding_bag(&t, &[0, 1, 2, 3], &[0, 1, 2, 3], EmbeddingBagMode::Sum, &cfg).unwrap();
        let mean =
            embedding_bag(&t, &[0, 1, 2, 3], &[0, 1, 2, 3], EmbeddingBagMode::Mean, &cfg).unwrap();
        assert_eq!(sum, mean);
    }

    // ── position_embedding tests ────────────────────────────────

    #[test]
    fn position_embedding_creates_zeros() {
        let buf = position_embedding(128, 64).unwrap();
        assert_eq!(buf.len(), 128 * 64);
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn position_embedding_rejects_zero_positions() {
        assert!(position_embedding(0, 64).is_err());
    }

    #[test]
    fn position_embedding_rejects_zero_dim() {
        assert!(position_embedding(128, 0).is_err());
    }

    // ── learned_position_embedding tests ────────────────────────

    #[test]
    fn learned_position_basic() {
        let table = vec![
            0.1, 0.2, // pos 0
            0.3, 0.4, // pos 1
            0.5, 0.6, // pos 2
        ];
        let out = learned_position_embedding(&table, &[0, 2], 3, 2).unwrap();
        assert_eq!(out, &[0.1, 0.2, 0.5, 0.6]);
    }

    #[test]
    fn learned_position_oob() {
        let table = vec![0.0; 6];
        assert!(learned_position_embedding(&table, &[3], 3, 2).is_err());
    }

    #[test]
    fn learned_position_duplicate() {
        let table = vec![1.0, 2.0, 3.0, 4.0];
        let out = learned_position_embedding(&table, &[0, 0, 1, 1], 2, 2).unwrap();
        assert_eq!(out, &[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn learned_position_empty() {
        let table = vec![0.0; 6];
        let out = learned_position_embedding(&table, &[], 3, 2).unwrap();
        assert!(out.is_empty());
    }

    // ── sinusoidal_position_embedding tests ─────────────────────

    #[test]
    fn sinusoidal_basic_shape() {
        let out = sinusoidal_position_embedding(&[0, 1, 2], 4).unwrap();
        assert_eq!(out.len(), 3 * 4);
    }

    #[test]
    fn sinusoidal_position_zero() {
        let out = sinusoidal_position_embedding(&[0], 4).unwrap();
        // sin(0)=0 for both dims, cos(0)=1 for both dims
        assert!((out[0] - 0.0).abs() < 1e-6); // sin(0*f0)
        assert!((out[1] - 0.0).abs() < 1e-6); // sin(0*f1)
        assert!((out[2] - 1.0).abs() < 1e-6); // cos(0*f0)
        assert!((out[3] - 1.0).abs() < 1e-6); // cos(0*f1)
    }

    #[test]
    fn sinusoidal_different_positions_differ() {
        let out = sinusoidal_position_embedding(&[0, 1], 8).unwrap();
        let pos0 = &out[0..8];
        let pos1 = &out[8..16];
        assert_ne!(pos0, pos1);
    }

    #[test]
    fn sinusoidal_rejects_zero_dim() {
        assert!(sinusoidal_position_embedding(&[0], 0).is_err());
    }

    #[test]
    fn sinusoidal_odd_dim() {
        let out = sinusoidal_position_embedding(&[1], 5).unwrap();
        assert_eq!(out.len(), 5);
        // Last element should be zero (unset for odd dim).
        assert!((out[4] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn sinusoidal_empty_positions() {
        let out = sinusoidal_position_embedding(&[], 4).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn sinusoidal_large_position() {
        let out = sinusoidal_position_embedding(&[10000], 4).unwrap();
        assert_eq!(out.len(), 4);
        // Values should be finite.
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── embedding_norm tests ────────────────────────────────────

    #[test]
    fn norm_clamps_l2() {
        let mut emb = vec![3.0, 4.0]; // L2 norm = 5
        embedding_norm(&mut emb, 1, 2, 1.0, 2.0).unwrap();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm <= 1.0 + 1e-5, "norm should be clamped: {norm}");
    }

    #[test]
    fn norm_preserves_below_max() {
        let mut emb = vec![0.3, 0.4]; // L2 norm = 0.5
        let orig = emb.clone();
        embedding_norm(&mut emb, 1, 2, 1.0, 2.0).unwrap();
        assert_eq!(emb, orig);
    }

    #[test]
    fn norm_clamps_l1() {
        let mut emb = vec![3.0, 4.0]; // L1 norm = 7
        embedding_norm(&mut emb, 1, 2, 2.0, 1.0).unwrap();
        let norm: f32 = emb.iter().map(|x| x.abs()).sum();
        assert!(norm <= 2.0 + 1e-4, "L1 norm should be clamped: {norm}");
    }

    #[test]
    fn norm_multiple_vectors() {
        let mut emb = vec![3.0, 4.0, 6.0, 8.0]; // norms 5 and 10
        embedding_norm(&mut emb, 2, 2, 1.0, 2.0).unwrap();
        let n1: f32 = emb[0..2].iter().map(|x| x * x).sum::<f32>().sqrt();
        let n2: f32 = emb[2..4].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(n1 <= 1.0 + 1e-5);
        assert!(n2 <= 1.0 + 1e-5);
    }

    #[test]
    fn norm_zero_dim_err() {
        let mut emb = vec![];
        assert!(embedding_norm(&mut emb, 0, 0, 1.0, 2.0).is_err());
    }

    #[test]
    fn norm_short_buffer_err() {
        let mut emb = vec![1.0];
        assert!(embedding_norm(&mut emb, 1, 2, 1.0, 2.0).is_err());
    }

    #[test]
    fn norm_zero_vector_preserved() {
        let mut emb = vec![0.0, 0.0];
        embedding_norm(&mut emb, 1, 2, 1.0, 2.0).unwrap();
        assert_eq!(emb, vec![0.0, 0.0]);
    }

    // ── embedding_gradient tests ────────────────────────────────

    #[test]
    fn gradient_basic() {
        let grad_out = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = [0_u32, 2];
        let grad = embedding_gradient(&grad_out, &indices, 4, 3).unwrap();
        assert_eq!(grad.len(), 12);
        assert_eq!(&grad[0..3], &[1.0, 2.0, 3.0]); // row 0
        assert_eq!(&grad[3..6], &[0.0, 0.0, 0.0]); // row 1 (unused)
        assert_eq!(&grad[6..9], &[4.0, 5.0, 6.0]); // row 2
        assert_eq!(&grad[9..12], &[0.0, 0.0, 0.0]); // row 3 (unused)
    }

    #[test]
    fn gradient_accumulates_duplicates() {
        let grad_out = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let indices = [1_u32, 1];
        let grad = embedding_gradient(&grad_out, &indices, 4, 3).unwrap();
        assert_eq!(&grad[3..6], &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn gradient_oob_index() {
        let grad_out = vec![1.0, 2.0, 3.0];
        assert!(embedding_gradient(&grad_out, &[5], 4, 3).is_err());
    }

    #[test]
    fn gradient_short_grad_output() {
        assert!(embedding_gradient(&[1.0], &[0], 4, 3).is_err());
    }

    #[test]
    fn gradient_empty() {
        let grad = embedding_gradient(&[], &[], 4, 3).unwrap();
        assert_eq!(grad.len(), 12);
        assert!(grad.iter().all(|&v| v == 0.0));
    }

    // ── fused_embedding_layernorm tests ─────────────────────────

    #[test]
    fn fused_ln_basic() {
        let t = sample_table();
        let cfg = sample_config();
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 3];
        let out = fused_embedding_layernorm(&t, &[0], &gamma, &beta, 1e-5, &cfg).unwrap();
        assert_eq!(out.len(), 3);
        // LayerNorm of [1,2,3]: mean=2, var=2/3 → each normalised
        let mean: f32 = out.iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-5, "mean should be ~0 after LN: {mean}");
    }

    #[test]
    fn fused_ln_with_affine() {
        let t = sample_table();
        let cfg = sample_config();
        let gamma = vec![2.0; 3];
        let beta = vec![1.0; 3];
        let out = fused_embedding_layernorm(&t, &[0], &gamma, &beta, 1e-5, &cfg).unwrap();
        // After LN with gamma=2, beta=1 the mean should be ~1.0
        let mean: f32 = out.iter().sum::<f32>() / 3.0;
        assert!((mean - 1.0).abs() < 1e-4, "affine mean should be ~1: {mean}");
    }

    #[test]
    fn fused_ln_multiple_tokens() {
        let t = sample_table();
        let cfg = sample_config();
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 3];
        let out = fused_embedding_layernorm(&t, &[0, 1, 2, 3], &gamma, &beta, 1e-5, &cfg).unwrap();
        assert_eq!(out.len(), 12);
        // Each group of 3 should have mean ≈ 0.
        for tok in 0..4 {
            let m: f32 = out[tok * 3..(tok + 1) * 3].iter().sum::<f32>() / 3.0;
            assert!(m.abs() < 1e-4, "token {tok} mean should be ~0: {m}");
        }
    }

    #[test]
    fn fused_ln_oob_index() {
        let t = sample_table();
        let cfg = sample_config();
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 3];
        assert!(fused_embedding_layernorm(&t, &[4], &gamma, &beta, 1e-5, &cfg).is_err());
    }

    #[test]
    fn fused_ln_short_gamma() {
        let t = sample_table();
        let cfg = sample_config();
        assert!(fused_embedding_layernorm(&t, &[0], &[1.0], &[0.0; 3], 1e-5, &cfg).is_err());
    }

    #[test]
    fn fused_ln_short_beta() {
        let t = sample_table();
        let cfg = sample_config();
        assert!(fused_embedding_layernorm(&t, &[0], &[1.0; 3], &[0.0], 1e-5, &cfg).is_err());
    }

    #[test]
    fn fused_ln_empty_indices() {
        let t = sample_table();
        let cfg = sample_config();
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 3];
        let out = fused_embedding_layernorm(&t, &[], &gamma, &beta, 1e-5, &cfg).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn fused_ln_constant_embedding() {
        // All-same values → variance = 0 → output = beta
        let weights = vec![5.0; 12]; // all 5s
        let t = EmbeddingTable::new(weights, 4, 3).unwrap();
        let cfg = sample_config();
        let gamma = vec![1.0; 3];
        let beta = vec![7.0; 3];
        let out = fused_embedding_layernorm(&t, &[0], &gamma, &beta, 1e-5, &cfg).unwrap();
        for &v in &out {
            assert!((v - 7.0).abs() < 1e-3, "constant input → output ≈ beta: {v}");
        }
    }

    // ── CUDA kernel source tests ────────────────────────────────

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn kernel_src_lookup_not_empty() {
        assert!(!EMBEDDING_OPS_LOOKUP_KERNEL_SRC.is_empty());
        assert!(EMBEDDING_OPS_LOOKUP_KERNEL_SRC.contains("embedding_ops_lookup_f32"));
    }

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn kernel_src_bag_sum_not_empty() {
        assert!(!EMBEDDING_BAG_SUM_KERNEL_SRC.is_empty());
        assert!(EMBEDDING_BAG_SUM_KERNEL_SRC.contains("embedding_bag_sum_f32"));
    }

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn kernel_src_sinusoidal_not_empty() {
        assert!(!SINUSOIDAL_POSITION_KERNEL_SRC.is_empty());
        assert!(SINUSOIDAL_POSITION_KERNEL_SRC.contains("sinusoidal_position_f32"));
    }

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn kernel_src_norm_not_empty() {
        assert!(!EMBEDDING_NORM_KERNEL_SRC.is_empty());
        assert!(EMBEDDING_NORM_KERNEL_SRC.contains("embedding_norm_f32"));
    }

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn kernel_src_fused_ln_not_empty() {
        assert!(!FUSED_EMBEDDING_LAYERNORM_KERNEL_SRC.is_empty());
        assert!(FUSED_EMBEDDING_LAYERNORM_KERNEL_SRC.contains("fused_embedding_layernorm_f32"));
    }

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn kernel_src_gradient_not_empty() {
        assert!(!EMBEDDING_GRADIENT_KERNEL_SRC.is_empty());
        assert!(EMBEDDING_GRADIENT_KERNEL_SRC.contains("embedding_gradient_f32"));
    }

    // ── Launch stub tests ───────────────────────────────────────

    #[test]
    fn launch_lookup_returns_gpu_error() {
        let cfg = sample_config();
        let result = launch_embedding_ops_lookup(&[0.0; 12], &[0], &mut [0.0; 3], &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn launch_bag_returns_gpu_error() {
        let cfg = sample_config();
        let result = launch_embedding_bag(
            &[0.0; 12],
            &[0],
            &[0],
            &mut [0.0; 3],
            EmbeddingBagMode::Sum,
            &cfg,
        );
        assert!(result.is_err());
    }

    #[test]
    fn launch_sinusoidal_returns_gpu_error() {
        assert!(launch_sinusoidal_position(&[0], &mut [0.0; 4], 4).is_err());
    }

    #[test]
    fn launch_norm_returns_gpu_error() {
        assert!(launch_embedding_norm(&mut [0.0; 4], 1, 4, 1.0, 2.0).is_err());
    }

    #[test]
    fn launch_fused_ln_returns_gpu_error() {
        let cfg = sample_config();
        assert!(
            launch_fused_embedding_layernorm(
                &[0.0; 12],
                &[0],
                &[1.0; 3],
                &[0.0; 3],
                &mut [0.0; 3],
                &cfg,
                1e-5
            )
            .is_err()
        );
    }

    #[test]
    fn launch_gradient_returns_gpu_error() {
        assert!(launch_embedding_gradient(&[0.0; 3], &[0], &mut [0.0; 12], 3).is_err());
    }

    // ── Large-scale tests ───────────────────────────────────────

    #[test]
    fn lookup_large_vocab_32k() {
        let vocab = 32_000;
        let dim = 4;
        let weights: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
        let t = EmbeddingTable::new(weights, vocab, dim).unwrap();
        let cfg = EmbeddingConfig::new(vocab, dim).unwrap();
        let out = embedding_lookup(&t, &[0, 31_999], &cfg).unwrap();
        assert_eq!(&out[0..dim], &[0.0, 1.0, 2.0, 3.0]);
        let base = 31_999.0 * dim as f32;
        assert_eq!(&out[dim..2 * dim], &[base, base + 1.0, base + 2.0, base + 3.0]);
    }

    #[test]
    fn bag_large_single_bag() {
        let vocab = 100;
        let dim = 2;
        let weights = vec![1.0_f32; vocab * dim];
        let t = EmbeddingTable::new(weights, vocab, dim).unwrap();
        let cfg = EmbeddingConfig::new(vocab, dim).unwrap();
        let indices: Vec<u32> = (0..50).collect();
        let out = embedding_bag(&t, &indices, &[0], EmbeddingBagMode::Sum, &cfg).unwrap();
        assert_eq!(out, &[50.0, 50.0]);
    }

    #[test]
    fn sinusoidal_many_positions() {
        let positions: Vec<u32> = (0..128).collect();
        let out = sinusoidal_position_embedding(&positions, 64).unwrap();
        assert_eq!(out.len(), 128 * 64);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Cross-operation consistency tests ───────────────────────

    #[test]
    fn sparse_equals_bag_sum() {
        let t = sample_table();
        let cfg = sample_config();
        let sparse = embedding_lookup_sparse(&t, &[0, 1, 2, 3], &[0, 2], &cfg).unwrap();
        let bag = embedding_bag(&t, &[0, 1, 2, 3], &[0, 2], EmbeddingBagMode::Sum, &cfg).unwrap();
        assert_eq!(sparse, bag);
    }

    #[test]
    fn fused_ln_matches_separate_ops() {
        let t = sample_table();
        let cfg = sample_config();
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 3];

        // Fused path.
        let fused = fused_embedding_layernorm(&t, &[0, 1], &gamma, &beta, 1e-5, &cfg).unwrap();

        // Separate path: lookup then LN.
        let emb = embedding_lookup(&t, &[0, 1], &cfg).unwrap();
        let mut separate = vec![0.0_f32; 6];
        for tok in 0..2 {
            let s = &emb[tok * 3..(tok + 1) * 3];
            let mean: f32 = s.iter().sum::<f32>() / 3.0;
            let var: f32 = s.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 3.0;
            let inv_std = 1.0 / (var + 1e-5_f32).sqrt();
            for j in 0..3 {
                separate[tok * 3 + j] = (s[j] - mean) * inv_std * gamma[j] + beta[j];
            }
        }
        for (a, b) in fused.iter().zip(separate.iter()) {
            assert!((a - b).abs() < 1e-5, "fused vs separate mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn gradient_round_trip() {
        let t = sample_table();
        let cfg = sample_config();
        let indices = [0_u32, 1, 2, 3];
        let emb = embedding_lookup(&t, &indices, &cfg).unwrap();
        // Use embeddings as grad_output.
        let grad = embedding_gradient(&emb, &indices, 4, 3).unwrap();
        // Each row in grad should match the original embedding row.
        assert_eq!(grad, t.weights);
    }

    // ── Edge-case / numerical tests ─────────────────────────────

    #[test]
    fn norm_preserves_direction() {
        let mut emb = vec![3.0, 4.0];
        let orig_ratio = emb[0] / emb[1];
        embedding_norm(&mut emb, 1, 2, 1.0, 2.0).unwrap();
        let new_ratio = emb[0] / emb[1];
        assert!((orig_ratio - new_ratio).abs() < 1e-5);
    }

    #[test]
    fn sinusoidal_orthogonality() {
        // Distant positions should have low dot product for large dims.
        let out = sinusoidal_position_embedding(&[0, 100], 128).unwrap();
        let a = &out[0..128];
        let b = &out[128..256];
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_sim = dot / (na * nb + 1e-8);
        assert!(cos_sim.abs() < 0.5, "distant positions should have low similarity: {cos_sim}");
    }

    #[test]
    fn lookup_negative_weights() {
        let weights = vec![-1.0, -2.0, -3.0, 4.0, 5.0, 6.0];
        let t = EmbeddingTable::new(weights, 2, 3).unwrap();
        let cfg = EmbeddingConfig::new(2, 3).unwrap();
        let out = embedding_lookup(&t, &[0, 1], &cfg).unwrap();
        assert_eq!(out, &[-1.0, -2.0, -3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn bag_sum_commutative() {
        let t = sample_table();
        let cfg = sample_config();
        let a = embedding_bag(&t, &[0, 1, 2], &[0], EmbeddingBagMode::Sum, &cfg).unwrap();
        let b = embedding_bag(&t, &[2, 0, 1], &[0], EmbeddingBagMode::Sum, &cfg).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn bag_max_commutative() {
        let t = sample_table();
        let cfg = sample_config();
        let a = embedding_bag(&t, &[0, 1, 2], &[0], EmbeddingBagMode::Max, &cfg).unwrap();
        let b = embedding_bag(&t, &[2, 1, 0], &[0], EmbeddingBagMode::Max, &cfg).unwrap();
        assert_eq!(a, b);
    }

    // ── Property tests ──────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        prop_compose! {
            fn embedding_args()(
                vocab in 1_usize..32,
                dim in 1_usize..16,
                n in 1_usize..16,
            )(
                ids in proptest::collection::vec(0..vocab as u32, n),
                weights in proptest::collection::vec(-100.0_f32..100.0, vocab * dim),
                vocab in Just(vocab),
                dim in Just(dim),
                n in Just(n),
            ) -> (Vec<f32>, Vec<u32>, usize, usize, usize) {
                (weights, ids, vocab, dim, n)
            }
        }

        proptest! {
            #[test]
            fn prop_lookup_output_shape(
                (weights, ids, vocab, dim, n) in embedding_args()
            ) {
                let t = EmbeddingTable::new(weights, vocab, dim).unwrap();
                let cfg = EmbeddingConfig::new(vocab, dim).unwrap();
                let out = embedding_lookup(&t, &ids, &cfg).unwrap();
                prop_assert_eq!(out.len(), n * dim);
            }

            #[test]
            fn prop_lookup_is_gather(
                (weights, ids, vocab, dim, _n) in embedding_args()
            ) {
                let t = EmbeddingTable::new(weights.clone(), vocab, dim).unwrap();
                let cfg = EmbeddingConfig::new(vocab, dim).unwrap();
                let out = embedding_lookup(&t, &ids, &cfg).unwrap();
                for (i, &id) in ids.iter().enumerate() {
                    let src = &weights[id as usize * dim..(id as usize + 1) * dim];
                    prop_assert_eq!(&out[i * dim..(i + 1) * dim], src);
                }
            }

            #[test]
            fn prop_padding_zeroed(
                (weights, _ids, vocab, dim, n) in embedding_args()
            ) {
                let t = EmbeddingTable::new(weights, vocab, dim).unwrap();
                let cfg = EmbeddingConfig::new(vocab, dim).unwrap().with_padding_idx(0);
                let ids = vec![0_u32; n];
                let out = embedding_lookup(&t, &ids, &cfg).unwrap();
                prop_assert!(out.iter().all(|&v| v == 0.0));
            }

            #[test]
            fn prop_bag_sum_matches_sparse(
                (weights, ids, vocab, dim, _n) in embedding_args()
            ) {
                let t = EmbeddingTable::new(weights, vocab, dim).unwrap();
                let cfg = EmbeddingConfig::new(vocab, dim).unwrap();
                let sparse = embedding_lookup_sparse(&t, &ids, &[0], &cfg).unwrap();
                let bag = embedding_bag(&t, &ids, &[0], EmbeddingBagMode::Sum, &cfg).unwrap();
                for (a, b) in sparse.iter().zip(bag.iter()) {
                    prop_assert!((a - b).abs() < 1e-4, "mismatch: {} vs {}", a, b);
                }
            }

            #[test]
            fn prop_gradient_shape(
                (weights, ids, vocab, dim, n) in embedding_args()
            ) {
                let _ = weights; // unused for gradient
                let grad_output: Vec<f32> = vec![1.0; n * dim];
                let grad = embedding_gradient(&grad_output, &ids, vocab, dim).unwrap();
                prop_assert_eq!(grad.len(), vocab * dim);
            }

            #[test]
            fn prop_sinusoidal_bounded(
                pos in proptest::collection::vec(0_u32..1000, 1..16),
                dim in (2_usize..64).prop_map(|d| d & !1), // even dims
            ) {
                let out = sinusoidal_position_embedding(&pos, dim).unwrap();
                for v in &out {
                    prop_assert!(v.is_finite());
                    prop_assert!(*v >= -1.0 - 1e-6 && *v <= 1.0 + 1e-6,
                        "sinusoidal value out of [-1,1]: {}", v);
                }
            }

            #[test]
            fn prop_norm_clamps(
                (weights, ids, vocab, dim, _n) in embedding_args()
            ) {
                let t = EmbeddingTable::new(weights, vocab, dim).unwrap();
                let cfg = EmbeddingConfig::new(vocab, dim).unwrap().with_max_norm(1.0);
                let out = embedding_lookup(&t, &ids, &cfg).unwrap();
                let n_vecs = ids.len();
                for v in 0..n_vecs {
                    let slice = &out[v * dim..(v + 1) * dim];
                    let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
                    prop_assert!(norm <= 1.0 + 1e-4, "norm should be clamped: {}", norm);
                }
            }
        }
    }
}
