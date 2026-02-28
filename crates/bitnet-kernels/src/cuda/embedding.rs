//! CUDA embedding lookup kernel.
//!
//! # Kernel strategy
//!
//! Embedding lookup maps integer token IDs to dense floating-point vectors by
//! indexing into a weight matrix `[vocab_size, embed_dim]`.  The operation is
//! entirely memory-bound — no arithmetic beyond address computation — so the
//! kernel is optimised for coalesced global-memory reads:
//!
//! 1. Each thread-block handles one token.  Threads within the block
//!    cooperatively load the `embed_dim`-wide row into a coalesced read
//!    pattern and write it to the output buffer.
//! 2. For **batched** lookups the grid is 1-D with `batch_size` blocks.
//! 3. **Tied embeddings** share the same weight matrix for both input
//!    embedding and output projection (LM head), avoiding a redundant copy
//!    on-device.
//!
//! A CPU fallback (`embedding_lookup_cpu`) is always available and used when
//! no GPU runtime is present.
//!
//! Target: full memory-bandwidth utilisation on Ampere+ for `embed_dim ≥ 128`.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Launch configuration for the embedding lookup kernel.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Vocabulary size (number of rows in the embedding matrix).
    pub vocab_size: usize,
    /// Embedding dimension (number of columns / elements per vector).
    pub embed_dim: usize,
    /// Number of tokens in the batch to look up.
    pub batch_size: usize,
    /// Threads per block — typically `min(embed_dim, 1024)`.
    pub threads_per_block: u32,
}

impl EmbeddingConfig {
    /// Create a configuration for the given vocabulary and embedding shape.
    ///
    /// # Errors
    ///
    /// Returns `KernelError::InvalidArguments` when any dimension is zero.
    pub fn for_shape(vocab_size: usize, embed_dim: usize, batch_size: usize) -> Result<Self> {
        if vocab_size == 0 || embed_dim == 0 || batch_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "Embedding dimensions must be non-zero: \
                     vocab_size={vocab_size}, embed_dim={embed_dim}, batch_size={batch_size}"
                ),
            }
            .into());
        }

        let threads_per_block = (embed_dim as u32).min(1024);
        Ok(Self { vocab_size, embed_dim, batch_size, threads_per_block })
    }

    /// Compute the CUDA grid dimensions `(batch_size, 1, 1)`.
    ///
    /// One thread-block per token in the batch.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        (self.batch_size as u32, 1, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// Tied embedding marker
// ---------------------------------------------------------------------------

/// Describes whether embedding weights are shared with the output projection.
///
/// When `Tied`, the same `[vocab_size, embed_dim]` weight matrix is used for
/// both the token embedding layer and the final LM-head linear projection,
/// saving approximately `vocab_size × embed_dim × 4` bytes of device memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingTied {
    /// Weights are independent — separate matrices for embedding and LM head.
    Independent,
    /// Weights are shared (tied) with the output projection.
    Tied,
}

// ---------------------------------------------------------------------------
// GPU launch stub
// ---------------------------------------------------------------------------

/// Launch stub for the CUDA embedding lookup kernel.
///
/// # Arguments
///
/// * `weights`   — Embedding weight matrix `[vocab_size, embed_dim]` (FP32, row-major)
/// * `token_ids` — Batch of token IDs `[batch_size]` (`u32`).  Each ID must be
///   `< config.vocab_size`.
/// * `output`    — Output buffer `[batch_size, embed_dim]` (FP32, written)
/// * `config`    — Launch configuration
/// * `_tied`     — Whether the weight matrix is shared with the output projection.
///   This flag is informational for the kernel scheduler; the launch
///   signature is identical either way since the same pointer is reused.
///
/// # Errors
///
/// * `KernelError::InvalidArguments` if any token ID is out of range.
/// * `KernelError::GpuError` until a real PTX kernel is compiled and loaded.
pub fn launch_embedding_lookup(
    weights: &[f32],
    token_ids: &[u32],
    output: &mut [f32],
    config: &EmbeddingConfig,
    _tied: EmbeddingTied,
) -> Result<()> {
    // Pre-validate buffer sizes and token IDs on the host side so that we
    // never launch a kernel with out-of-bounds reads.
    validate_args(weights, token_ids, output, config)?;

    log::debug!(
        "Embedding lookup stub: vocab_size={}, embed_dim={}, batch_size={}, grid={:?}",
        config.vocab_size,
        config.embed_dim,
        config.batch_size,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "Embedding lookup CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ---------------------------------------------------------------------------
// CPU fallback
// ---------------------------------------------------------------------------

/// CPU fallback for embedding lookup.
///
/// Performs the same operation as the CUDA kernel but entirely on the host.
/// This is always available regardless of feature flags and serves as the
/// reference implementation for correctness testing.
///
/// # Errors
///
/// Returns `KernelError::InvalidArguments` on shape mismatches or
/// out-of-range token IDs.
pub fn embedding_lookup_cpu(
    weights: &[f32],
    token_ids: &[u32],
    output: &mut [f32],
    config: &EmbeddingConfig,
) -> Result<()> {
    validate_args(weights, token_ids, output, config)?;

    let embed_dim = config.embed_dim;
    for (i, &tid) in token_ids.iter().enumerate() {
        let src_offset = (tid as usize) * embed_dim;
        let dst_offset = i * embed_dim;
        output[dst_offset..dst_offset + embed_dim]
            .copy_from_slice(&weights[src_offset..src_offset + embed_dim]);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Batched helper
// ---------------------------------------------------------------------------

/// Convenience wrapper for batched embedding lookup with automatic fallback.
///
/// Tries the CUDA kernel first; if that fails (no runtime, scaffold-only, etc.)
/// it falls back to the CPU implementation transparently.
///
/// # Errors
///
/// Returns an error only when both the GPU and CPU paths fail (which should
/// never happen for valid inputs).
pub fn batched_embedding_lookup(
    weights: &[f32],
    token_ids: &[u32],
    output: &mut [f32],
    config: &EmbeddingConfig,
    tied: EmbeddingTied,
) -> Result<()> {
    match launch_embedding_lookup(weights, token_ids, output, config, tied) {
        Ok(()) => Ok(()),
        Err(_gpu_err) => {
            log::debug!("GPU embedding unavailable, falling back to CPU");
            embedding_lookup_cpu(weights, token_ids, output, config)
        }
    }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Shared argument validation for both GPU and CPU paths.
fn validate_args(
    weights: &[f32],
    token_ids: &[u32],
    output: &[f32],
    config: &EmbeddingConfig,
) -> Result<()> {
    let expected_weights = config.vocab_size * config.embed_dim;
    if weights.len() < expected_weights {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "weights buffer too small: expected at least {expected_weights}, got {}",
                weights.len()
            ),
        }
        .into());
    }

    if token_ids.len() < config.batch_size {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "token_ids buffer too small: expected at least {}, got {}",
                config.batch_size,
                token_ids.len()
            ),
        }
        .into());
    }

    let expected_output = config.batch_size * config.embed_dim;
    if output.len() < expected_output {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "output buffer too small: expected at least {expected_output}, got {}",
                output.len()
            ),
        }
        .into());
    }

    for (i, &tid) in token_ids.iter().take(config.batch_size).enumerate() {
        if (tid as usize) >= config.vocab_size {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "token_ids[{i}]={tid} out of range for vocab_size={}",
                    config.vocab_size
                ),
            }
            .into());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config tests -------------------------------------------------------

    #[test]
    fn test_embedding_config_for_shape() {
        let cfg = EmbeddingConfig::for_shape(32000, 2048, 4).unwrap();
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.embed_dim, 2048);
        assert_eq!(cfg.batch_size, 4);
        assert_eq!(cfg.threads_per_block, 1024); // capped
    }

    #[test]
    fn test_embedding_config_small_embed() {
        let cfg = EmbeddingConfig::for_shape(100, 64, 1).unwrap();
        assert_eq!(cfg.threads_per_block, 64);
    }

    #[test]
    fn test_embedding_config_rejects_zero() {
        assert!(EmbeddingConfig::for_shape(0, 2048, 1).is_err());
        assert!(EmbeddingConfig::for_shape(32000, 0, 1).is_err());
        assert!(EmbeddingConfig::for_shape(32000, 2048, 0).is_err());
    }

    #[test]
    fn test_embedding_grid_dim() {
        let cfg = EmbeddingConfig::for_shape(32000, 2048, 8).unwrap();
        assert_eq!(cfg.grid_dim(), (8, 1, 1));
        assert_eq!(cfg.block_dim(), (1024, 1, 1));
    }

    // -- CPU fallback tests -------------------------------------------------

    #[test]
    fn test_cpu_embedding_single_token() {
        // Tiny vocab: 3 tokens, embed_dim = 4
        let weights = vec![
            1.0, 2.0, 3.0, 4.0, // token 0
            5.0, 6.0, 7.0, 8.0, // token 1
            9.0, 10.0, 11.0, 12.0, // token 2
        ];
        let token_ids = vec![1u32];
        let mut output = vec![0.0f32; 4];
        let cfg = EmbeddingConfig::for_shape(3, 4, 1).unwrap();

        embedding_lookup_cpu(&weights, &token_ids, &mut output, &cfg).unwrap();
        assert_eq!(output, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_cpu_embedding_batch() {
        let weights = vec![
            0.0, 0.0, // token 0
            1.0, 1.0, // token 1
            2.0, 2.0, // token 2
        ];
        let token_ids = vec![2u32, 0, 1];
        let mut output = vec![0.0f32; 6];
        let cfg = EmbeddingConfig::for_shape(3, 2, 3).unwrap();

        embedding_lookup_cpu(&weights, &token_ids, &mut output, &cfg).unwrap();
        assert_eq!(output, vec![2.0, 2.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_cpu_embedding_oob_token() {
        let weights = vec![0.0f32; 8]; // vocab=2, dim=4
        let token_ids = vec![5u32]; // out of range
        let mut output = vec![0.0f32; 4];
        let cfg = EmbeddingConfig::for_shape(2, 4, 1).unwrap();

        assert!(embedding_lookup_cpu(&weights, &token_ids, &mut output, &cfg).is_err());
    }

    #[test]
    fn test_cpu_embedding_weight_buffer_too_small() {
        let weights = vec![0.0f32; 4]; // only 4, need 8 (vocab=2, dim=4)
        let token_ids = vec![0u32];
        let mut output = vec![0.0f32; 4];
        let cfg = EmbeddingConfig::for_shape(2, 4, 1).unwrap();

        assert!(embedding_lookup_cpu(&weights, &token_ids, &mut output, &cfg).is_err());
    }

    #[test]
    fn test_cpu_embedding_output_buffer_too_small() {
        let weights = vec![0.0f32; 8];
        let token_ids = vec![0u32];
        let mut output = vec![0.0f32; 2]; // need 4
        let cfg = EmbeddingConfig::for_shape(2, 4, 1).unwrap();

        assert!(embedding_lookup_cpu(&weights, &token_ids, &mut output, &cfg).is_err());
    }

    // -- Tied embedding tests -----------------------------------------------

    #[test]
    fn test_tied_enum_values() {
        assert_ne!(EmbeddingTied::Independent, EmbeddingTied::Tied);
    }

    #[test]
    fn test_tied_embedding_uses_same_weights() {
        // Tied and independent should produce identical results given the
        // same weight pointer — the flag is informational.
        let weights = vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
        ];
        let token_ids = vec![1u32];
        let mut out_a = vec![0.0f32; 2];
        let mut out_b = vec![0.0f32; 2];
        let cfg = EmbeddingConfig::for_shape(2, 2, 1).unwrap();

        batched_embedding_lookup(&weights, &token_ids, &mut out_a, &cfg, EmbeddingTied::Tied)
            .unwrap();
        batched_embedding_lookup(
            &weights,
            &token_ids,
            &mut out_b,
            &cfg,
            EmbeddingTied::Independent,
        )
        .unwrap();
        assert_eq!(out_a, out_b);
    }

    // -- Batched fallback test ----------------------------------------------

    #[test]
    fn test_batched_fallback_to_cpu() {
        let weights = vec![
            10.0, 20.0, 30.0, // token 0
            40.0, 50.0, 60.0, // token 1
        ];
        let token_ids = vec![0u32, 1];
        let mut output = vec![0.0f32; 6];
        let cfg = EmbeddingConfig::for_shape(2, 3, 2).unwrap();

        // GPU stub returns Err, so this should transparently fall back to CPU.
        batched_embedding_lookup(
            &weights,
            &token_ids,
            &mut output,
            &cfg,
            EmbeddingTied::Independent,
        )
        .unwrap();
        assert_eq!(output, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    // -- GPU stub test (always scaffold error) ------------------------------

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_embedding_launch() {
        let weights = vec![0.0f32; 32000 * 2048];
        let token_ids = vec![0u32; 16];
        let mut output = vec![0.0f32; 16 * 2048];
        let cfg = EmbeddingConfig::for_shape(32000, 2048, 16).unwrap();
        let result = launch_embedding_lookup(
            &weights,
            &token_ids,
            &mut output,
            &cfg,
            EmbeddingTied::Independent,
        );
        assert!(result.is_ok(), "CUDA embedding launch failed: {result:?}");
    }
}
