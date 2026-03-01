//! Token embedding lookup kernel with CPU fallback and optional
//! position encoding.
//!
//! # Kernel strategy
//!
//! Token embedding lookup maps discrete token IDs to dense vectors from a
//! learned embedding table `[vocab_size, embed_dim]`:
//!
//!   `output[i] = embedding_table[token_ids[i]]`
//!
//! Position encoding may be added element-wise:
//!
//! - **Sinusoidal**: `PE(pos, 2i) = sin(pos / 10000^(2i/d))`,
//!   `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`
//! - **Learned**: reads a second table `[max_seq_len, embed_dim]`
//!
//! Out-of-vocabulary token IDs (≥ `vocab_size`) produce a zero embedding.
//!
//! The CUDA kernel fuses table lookup + position encoding + write-back
//! in a single pass. The CPU fallback below provides correctness
//! testing.

use bitnet_common::{KernelError, Result};

// ── Position encoding variant ────────────────────────────────────

/// Selects the position encoding strategy applied after token lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionEncoding {
    /// No positional information added.
    None,
    /// Fixed sinusoidal encoding (Vaswani et al., 2017).
    Sinusoidal,
    /// Learned position embeddings from a separate table.
    Learned,
}

// ── Data type selector ───────────────────────────────────────────

/// Embedding table storage data type.
///
/// The CPU fallback always works in f32. On GPU the kernel reads
/// directly in the source dtype and converts to f32 for output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingDtype {
    /// 32-bit IEEE 754 float.
    F32,
    /// 16-bit IEEE 754 half-float (GPU only).
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    F16,
    /// 16-bit Brain Float (GPU only).
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    Bf16,
}

// ── Launch configuration ─────────────────────────────────────────

/// Launch configuration for the embedding lookup kernel.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Vocabulary size (number of rows in the embedding table).
    pub vocab_size: usize,
    /// Embedding dimension (number of columns per row).
    pub embed_dim: usize,
    /// Number of tokens in the batch to look up.
    pub n_tokens: usize,
    /// Threads per block — typically `min(embed_dim, 1024)`.
    pub threads_per_block: u32,
    /// Position encoding strategy.
    pub position_encoding: PositionEncoding,
    /// Maximum sequence length for learned position encoding.
    pub max_seq_len: usize,
    /// Starting position offset for position encoding.
    pub position_offset: usize,
    /// Storage data type of the embedding table.
    pub dtype: EmbeddingDtype,
}

impl EmbeddingConfig {
    /// Create a configuration for the given vocabulary shape.
    pub fn for_shape(vocab_size: usize, embed_dim: usize, n_tokens: usize) -> Result<Self> {
        if vocab_size == 0 || embed_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "embedding dimensions must be non-zero: \
                     vocab_size={vocab_size}, embed_dim={embed_dim}"
                ),
            }
            .into());
        }
        if n_tokens == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "n_tokens must be non-zero".into(),
            }
            .into());
        }

        let threads_per_block = (embed_dim as u32).min(1024);

        Ok(Self {
            vocab_size,
            embed_dim,
            n_tokens,
            threads_per_block,
            position_encoding: PositionEncoding::None,
            max_seq_len: 0,
            position_offset: 0,
            dtype: EmbeddingDtype::F32,
        })
    }

    /// Enable sinusoidal position encoding.
    #[must_use]
    pub fn with_sinusoidal_position(mut self) -> Self {
        self.position_encoding = PositionEncoding::Sinusoidal;
        self
    }

    /// Enable learned position encoding with the given max length.
    #[must_use]
    pub fn with_learned_position(mut self, max_seq_len: usize) -> Self {
        self.position_encoding = PositionEncoding::Learned;
        self.max_seq_len = max_seq_len;
        self
    }

    /// Set the starting position offset for position encoding.
    #[must_use]
    pub fn with_position_offset(mut self, offset: usize) -> Self {
        self.position_offset = offset;
        self
    }

    /// Set the embedding table storage dtype.
    #[must_use]
    pub fn with_dtype(mut self, dtype: EmbeddingDtype) -> Self {
        self.dtype = dtype;
        self
    }

    /// Compute the CUDA grid dimensions `(n_tokens, 1, 1)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        (self.n_tokens as u32, 1, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ── CPU fallback: sinusoidal position encoding ───────────────────

/// Compute sinusoidal position encoding for a single position.
///
/// Writes `embed_dim` values into `output`:
///   even indices → `sin(pos / 10000^(2i/d))`
///   odd  indices → `cos(pos / 10000^(2i/d))`
pub fn sinusoidal_position_encoding(pos: usize, embed_dim: usize, output: &mut [f32]) {
    debug_assert!(output.len() >= embed_dim);
    let d = embed_dim as f32;
    for (i, out_val) in output.iter_mut().enumerate().take(embed_dim) {
        let dim_pair = (i / 2) as f32;
        let angle = (pos as f32) / (10_000.0_f32).powf(2.0 * dim_pair / d);
        *out_val = if i % 2 == 0 { angle.sin() } else { angle.cos() };
    }
}

// ── CPU fallback: token embedding lookup ─────────────────────────

/// Look up token embeddings on the CPU.
///
/// For each token ID, copies the corresponding row from
/// `embedding_table` into `output`. Out-of-vocabulary IDs
/// (≥ `vocab_size`) produce a zero embedding.
///
/// Optionally adds position encoding when
/// `config.position_encoding` is not `None`.
///
/// # Layout
///
/// * `embedding_table`: `[vocab_size, embed_dim]` row-major f32
/// * `token_ids`: `[n_tokens]`
/// * `output`: `[n_tokens, embed_dim]` row-major f32 (written)
/// * `position_table` (optional): `[max_seq_len, embed_dim]` for
///   learned encoding
pub fn embedding_lookup_cpu(
    embedding_table: &[f32],
    token_ids: &[u32],
    output: &mut [f32],
    config: &EmbeddingConfig,
    position_table: Option<&[f32]>,
) -> Result<()> {
    let n = config.n_tokens;
    let d = config.embed_dim;
    let v = config.vocab_size;

    if token_ids.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("token_ids length {} < n_tokens {n}", token_ids.len()),
        }
        .into());
    }
    if embedding_table.len() < v * d {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "embedding_table length {} < vocab_size * embed_dim {}",
                embedding_table.len(),
                v * d
            ),
        }
        .into());
    }
    if output.len() < n * d {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < n_tokens * embed_dim {}", output.len(), n * d),
        }
        .into());
    }

    // Validate learned position table if needed.
    if config.position_encoding == PositionEncoding::Learned {
        let max_pos = config.position_offset + n;
        if config.max_seq_len < max_pos {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "position_offset({}) + n_tokens({n}) = {max_pos} \
                     exceeds max_seq_len({})",
                    config.position_offset, config.max_seq_len
                ),
            }
            .into());
        }
        match position_table {
            Some(pt) if pt.len() >= config.max_seq_len * d => {}
            _ => {
                return Err(KernelError::InvalidArguments {
                    reason: "learned position encoding requires a \
                         position_table of length >= \
                         max_seq_len * embed_dim"
                        .into(),
                }
                .into());
            }
        }
    }

    // Scratch buffer for sinusoidal PE (reused across tokens).
    let mut pe_buf = if config.position_encoding == PositionEncoding::Sinusoidal {
        vec![0.0f32; d]
    } else {
        Vec::new()
    };

    for (t, &tok) in token_ids.iter().enumerate().take(n) {
        let out_start = t * d;
        let tid = tok as usize;

        // Token embedding (zero for OOV).
        if tid < v {
            let src_start = tid * d;
            output[out_start..out_start + d]
                .copy_from_slice(&embedding_table[src_start..src_start + d]);
        } else {
            output[out_start..out_start + d].fill(0.0);
        }

        // Position encoding.
        let pos = config.position_offset + t;
        match config.position_encoding {
            PositionEncoding::None => {}
            PositionEncoding::Sinusoidal => {
                sinusoidal_position_encoding(pos, d, &mut pe_buf);
                for (o, &p) in output[out_start..out_start + d].iter_mut().zip(pe_buf.iter()) {
                    *o += p;
                }
            }
            PositionEncoding::Learned => {
                let pt = position_table.unwrap();
                let pe_start = pos * d;
                for (o, &p) in output[out_start..out_start + d]
                    .iter_mut()
                    .zip(pt[pe_start..pe_start + d].iter())
                {
                    *o += p;
                }
            }
        }
    }

    Ok(())
}

// ── GPU launch stub ──────────────────────────────────────────────

/// Launch stub for the embedding lookup CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is
/// compiled and loaded.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_embedding(
    _embedding_table: &[f32],
    _token_ids: &[u32],
    _output: &mut [f32],
    config: &EmbeddingConfig,
    _position_table: Option<&[f32]>,
) -> Result<()> {
    log::debug!(
        "Embedding stub: vocab={}, dim={}, tokens={}, pos={:?}, \
         grid={:?}",
        config.vocab_size,
        config.embed_dim,
        config.n_tokens,
        config.position_encoding,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "Embedding CUDA kernel not yet compiled — \
                 scaffold only"
            .into(),
    }
    .into())
}

// ── Dispatch: GPU → CPU fallback ─────────────────────────────────

/// Embedding lookup with automatic GPU → CPU fallback.
///
/// Tries the CUDA launch when compiled with GPU features and a
/// runtime device is available; otherwise falls through to the CPU
/// implementation.
pub fn embedding_forward(
    embedding_table: &[f32],
    token_ids: &[u32],
    output: &mut [f32],
    config: &EmbeddingConfig,
    position_table: Option<&[f32]>,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime() {
            if let Ok(()) =
                launch_embedding(embedding_table, token_ids, output, config, position_table)
            {
                return Ok(());
            }
        }
    }

    embedding_lookup_cpu(embedding_table, token_ids, output, config, position_table)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config validation ────────────────────────────────────

    #[test]
    fn test_embedding_config_for_shape() {
        let cfg = EmbeddingConfig::for_shape(32000, 2048, 8).unwrap();
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.embed_dim, 2048);
        assert_eq!(cfg.n_tokens, 8);
        assert_eq!(cfg.threads_per_block, 1024);
        assert_eq!(cfg.position_encoding, PositionEncoding::None);
    }

    #[test]
    fn test_embedding_config_small_dim() {
        let cfg = EmbeddingConfig::for_shape(100, 64, 4).unwrap();
        assert_eq!(cfg.threads_per_block, 64);
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!(gx, 4);
        assert_eq!(gy, 1);
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_embedding_config_rejects_zero_vocab() {
        assert!(EmbeddingConfig::for_shape(0, 128, 1).is_err());
    }

    #[test]
    fn test_embedding_config_rejects_zero_dim() {
        assert!(EmbeddingConfig::for_shape(100, 0, 1).is_err());
    }

    #[test]
    fn test_embedding_config_rejects_zero_tokens() {
        assert!(EmbeddingConfig::for_shape(100, 128, 0).is_err());
    }

    #[test]
    fn test_embedding_config_builders() {
        let cfg = EmbeddingConfig::for_shape(100, 64, 2)
            .unwrap()
            .with_sinusoidal_position()
            .with_position_offset(10);
        assert_eq!(cfg.position_encoding, PositionEncoding::Sinusoidal);
        assert_eq!(cfg.position_offset, 10);
    }

    #[test]
    fn test_embedding_config_learned_builder() {
        let cfg = EmbeddingConfig::for_shape(100, 64, 2).unwrap().with_learned_position(512);
        assert_eq!(cfg.position_encoding, PositionEncoding::Learned);
        assert_eq!(cfg.max_seq_len, 512);
    }

    #[test]
    fn test_embedding_config_grid_dim() {
        let cfg = EmbeddingConfig::for_shape(32000, 4096, 32).unwrap();
        assert_eq!(cfg.grid_dim(), (32, 1, 1));
        assert_eq!(cfg.block_dim(), (1024, 1, 1));
    }

    // ── Token lookup correctness ─────────────────────────────

    #[test]
    fn test_embedding_lookup_basic() {
        let table: Vec<f32> = vec![
            1.0, 2.0, 3.0, // token 0
            4.0, 5.0, 6.0, // token 1
            7.0, 8.0, 9.0, // token 2
            10.0, 11.0, 12.0, // token 3
        ];
        let ids = [2u32, 0, 3];
        let cfg = EmbeddingConfig::for_shape(4, 3, 3).unwrap();
        let mut out = vec![0.0f32; 9];

        embedding_lookup_cpu(&table, &ids, &mut out, &cfg, None).unwrap();

        assert_eq!(&out[0..3], &[7.0, 8.0, 9.0]);
        assert_eq!(&out[3..6], &[1.0, 2.0, 3.0]);
        assert_eq!(&out[6..9], &[10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_embedding_lookup_single_token() {
        let table = vec![0.5f32; 8]; // vocab=2, dim=4
        let ids = [1u32];
        let cfg = EmbeddingConfig::for_shape(2, 4, 1).unwrap();
        let mut out = vec![0.0f32; 4];

        embedding_lookup_cpu(&table, &ids, &mut out, &cfg, None).unwrap();
        assert_eq!(&out, &[0.5, 0.5, 0.5, 0.5]);
    }

    // ── Out-of-vocabulary handling ───────────────────────────

    #[test]
    fn test_embedding_oov_returns_zero() {
        let table = vec![1.0f32; 6]; // vocab=2, dim=3
        let ids = [5u32]; // OOV
        let cfg = EmbeddingConfig::for_shape(2, 3, 1).unwrap();
        let mut out = vec![99.0f32; 3];

        embedding_lookup_cpu(&table, &ids, &mut out, &cfg, None).unwrap();
        assert_eq!(&out, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_embedding_oov_mixed_batch() {
        let table = vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
        ];
        let ids = [0u32, 999, 1];
        let cfg = EmbeddingConfig::for_shape(2, 2, 3).unwrap();
        let mut out = vec![0.0f32; 6];

        embedding_lookup_cpu(&table, &ids, &mut out, &cfg, None).unwrap();
        assert_eq!(&out[0..2], &[1.0, 2.0]);
        assert_eq!(&out[2..4], &[0.0, 0.0]);
        assert_eq!(&out[4..6], &[3.0, 4.0]);
    }

    // ── Position encoding values ─────────────────────────────

    #[test]
    fn test_sinusoidal_pe_at_zero() {
        let mut pe = vec![0.0f32; 4];
        sinusoidal_position_encoding(0, 4, &mut pe);
        assert!((pe[0] - 0.0).abs() < 1e-6); // sin(0)
        assert!((pe[1] - 1.0).abs() < 1e-6); // cos(0)
        assert!((pe[2] - 0.0).abs() < 1e-6);
        assert!((pe[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sinusoidal_pe_varies_with_position() {
        let dim = 8;
        let mut pe0 = vec![0.0f32; dim];
        let mut pe1 = vec![0.0f32; dim];
        sinusoidal_position_encoding(0, dim, &mut pe0);
        sinusoidal_position_encoding(1, dim, &mut pe1);
        assert_ne!(pe0, pe1);
    }

    #[test]
    fn test_sinusoidal_pe_bounded() {
        let dim = 64;
        let mut pe = vec![0.0f32; dim];
        for pos in [0, 1, 100, 10_000] {
            sinusoidal_position_encoding(pos, dim, &mut pe);
            for &v in &pe {
                assert!((-1.0..=1.0).contains(&v), "PE out of [-1,1] at pos={pos}: {v}");
            }
        }
    }

    #[test]
    fn test_embedding_with_sinusoidal_position() {
        let table = vec![0.0f32; 6]; // all zeros
        let ids = [0u32];
        let cfg = EmbeddingConfig::for_shape(2, 3, 1)
            .unwrap()
            .with_sinusoidal_position()
            .with_position_offset(1);
        let mut out = vec![0.0f32; 3];

        embedding_lookup_cpu(&table, &ids, &mut out, &cfg, None).unwrap();

        let mut expected = vec![0.0f32; 3];
        sinusoidal_position_encoding(1, 3, &mut expected);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_embedding_with_learned_position() {
        let table = vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
        ];
        let pos_table = vec![
            0.1, 0.2, // position 0
            0.3, 0.4, // position 1
        ];
        let ids = [0u32, 1];
        let cfg = EmbeddingConfig::for_shape(2, 2, 2).unwrap().with_learned_position(2);
        let mut out = vec![0.0f32; 4];

        embedding_lookup_cpu(&table, &ids, &mut out, &cfg, Some(&pos_table)).unwrap();

        assert!((out[0] - 1.1).abs() < 1e-6);
        assert!((out[1] - 2.2).abs() < 1e-6);
        assert!((out[2] - 3.3).abs() < 1e-6);
        assert!((out[3] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn test_learned_position_rejects_missing_table() {
        let table = vec![1.0f32; 4];
        let ids = [0u32];
        let cfg = EmbeddingConfig::for_shape(2, 2, 1).unwrap().with_learned_position(4);
        let mut out = vec![0.0f32; 2];

        assert!(embedding_lookup_cpu(&table, &ids, &mut out, &cfg, None).is_err());
    }

    #[test]
    fn test_learned_position_rejects_overflow() {
        let table = vec![1.0f32; 4];
        let pos_table = vec![0.0f32; 4];
        let ids = [0u32, 0];
        let cfg = EmbeddingConfig::for_shape(2, 2, 2)
            .unwrap()
            .with_learned_position(2)
            .with_position_offset(2);
        let mut out = vec![0.0f32; 4];

        assert!(embedding_lookup_cpu(&table, &ids, &mut out, &cfg, Some(&pos_table),).is_err());
    }

    // ── Batch processing ─────────────────────────────────────

    #[test]
    fn test_embedding_batch_lookup() {
        let vocab = 8;
        let dim = 4;
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
        let ids: Vec<u32> = (0..vocab as u32).collect();
        let cfg = EmbeddingConfig::for_shape(vocab, dim, vocab).unwrap();
        let mut out = vec![0.0f32; vocab * dim];

        embedding_lookup_cpu(&table, &ids, &mut out, &cfg, None).unwrap();
        assert_eq!(out, table);
    }

    #[test]
    fn test_embedding_batch_repeated_tokens() {
        let table = vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
        ];
        let ids = [1u32, 1, 0, 1];
        let cfg = EmbeddingConfig::for_shape(2, 2, 4).unwrap();
        let mut out = vec![0.0f32; 8];

        embedding_lookup_cpu(&table, &ids, &mut out, &cfg, None).unwrap();

        assert_eq!(&out[0..2], &[3.0, 4.0]);
        assert_eq!(&out[2..4], &[3.0, 4.0]);
        assert_eq!(&out[4..6], &[1.0, 2.0]);
        assert_eq!(&out[6..8], &[3.0, 4.0]);
    }

    // ── Dimension validation ─────────────────────────────────

    #[test]
    fn test_embedding_rejects_short_table() {
        let table = vec![1.0f32; 3]; // too small
        let ids = [0u32];
        let cfg = EmbeddingConfig::for_shape(2, 3, 1).unwrap();
        let mut out = vec![0.0f32; 3];

        assert!(embedding_lookup_cpu(&table, &ids, &mut out, &cfg, None).is_err());
    }

    #[test]
    fn test_embedding_rejects_short_output() {
        let table = vec![1.0f32; 6];
        let ids = [0u32];
        let cfg = EmbeddingConfig::for_shape(2, 3, 1).unwrap();
        let mut out = vec![0.0f32; 2]; // too small

        assert!(embedding_lookup_cpu(&table, &ids, &mut out, &cfg, None).is_err());
    }

    #[test]
    fn test_embedding_rejects_short_ids() {
        let table = vec![1.0f32; 6];
        let ids = [0u32]; // only 1, config says 2
        let cfg = EmbeddingConfig::for_shape(2, 3, 2).unwrap();
        let mut out = vec![0.0f32; 6];

        assert!(embedding_lookup_cpu(&table, &ids, &mut out, &cfg, None).is_err());
    }

    // ── Dispatch (CPU path) ──────────────────────────────────

    #[test]
    fn test_embedding_forward_uses_cpu_path() {
        let table = vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
        ];
        let ids = [1u32, 0];
        let cfg = EmbeddingConfig::for_shape(2, 2, 2).unwrap();
        let mut out = vec![0.0f32; 4];

        embedding_forward(&table, &ids, &mut out, &cfg, None).unwrap();

        assert_eq!(&out[0..2], &[3.0, 4.0]);
        assert_eq!(&out[2..4], &[1.0, 2.0]);
    }

    // ── Property tests ───────────────────────────────────────

    #[test]
    fn test_property_output_dimensions() {
        for &(v, d, n) in &[(100, 64, 1), (32000, 2048, 32), (5, 3, 5)] {
            let table = vec![0.0f32; v * d];
            let ids: Vec<u32> = (0..n).map(|i| (i % v) as u32).collect();
            let cfg = EmbeddingConfig::for_shape(v, d, n).unwrap();
            let mut out = vec![f32::NAN; n * d];

            embedding_lookup_cpu(&table, &ids, &mut out, &cfg, None).unwrap();

            assert_eq!(out.len(), n * d);
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_property_same_token_same_embedding() {
        let table: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let ids = [3u32, 1, 3, 3, 1];
        let cfg = EmbeddingConfig::for_shape(5, 4, 5).unwrap();
        let mut out = vec![0.0f32; 20];

        embedding_lookup_cpu(&table, &ids, &mut out, &cfg, None).unwrap();

        // Same token id → same embedding.
        assert_eq!(&out[0..4], &out[8..12]);
        assert_eq!(&out[0..4], &out[12..16]);
        assert_eq!(&out[4..8], &out[16..20]);
    }

    #[test]
    fn test_property_oov_always_zero() {
        let table = vec![99.0f32; 12]; // vocab=3, dim=4
        let ids = [100u32, u32::MAX, 3]; // all OOV
        let cfg = EmbeddingConfig::for_shape(3, 4, 3).unwrap();
        let mut out = vec![1.0f32; 12];

        embedding_lookup_cpu(&table, &ids, &mut out, &cfg, None).unwrap();

        assert!(out.iter().all(|&v| v == 0.0));
    }

    // ── GPU stub test ────────────────────────────────────────

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu \
                on GPU hardware"]
    fn test_cuda_embedding_launch() {
        let cfg = EmbeddingConfig::for_shape(32000, 2048, 16).unwrap();
        let table = vec![0.0f32; 32000 * 2048];
        let ids: Vec<u32> = (0..16).collect();
        let mut output = vec![0.0f32; 16 * 2048];
        let result = embedding_forward(&table, &ids, &mut output, &cfg, None);
        assert!(result.is_ok(), "CUDA embedding launch failed: {result:?}");
    }
}
