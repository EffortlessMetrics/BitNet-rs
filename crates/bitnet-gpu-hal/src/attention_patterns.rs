//! Module stub - implementation pending merge from feature branch
//! Attention pattern analysis and mask generation for efficient attention.
//!
//! Provides various attention patterns (Dense, Causal, Sliding, Sparse, Block,
//! Longformer, `BigBird`) with mask generation, block-sparse representation,
//! pattern analysis, and an orchestrator engine.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during attention pattern operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttentionPatternError {
    /// An invalid configuration was provided.
    InvalidConfig(String),
    /// Sequence length is incompatible with the requested operation.
    InvalidSequenceLength(String),
    /// Block size does not evenly divide the sequence length.
    BlockSizeMismatch { seq_len: usize, block_size: usize },
    /// A required field was missing.
    MissingField(String),
}

impl fmt::Display for AttentionPatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::InvalidSequenceLength(msg) => write!(f, "invalid sequence length: {msg}"),
            Self::BlockSizeMismatch { seq_len, block_size } => {
                write!(f, "block size {block_size} does not divide seq_len {seq_len}")
            }
            Self::MissingField(name) => write!(f, "missing field: {name}"),
        }
    }
}

impl std::error::Error for AttentionPatternError {}

// ---------------------------------------------------------------------------
// 1. AttentionPattern enum
// ---------------------------------------------------------------------------

/// Supported attention pattern types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionPattern {
    /// Full quadratic attention – every token attends to every other.
    Dense,
    /// Lower-triangular causal mask (autoregressive).
    Causal,
    /// Fixed-width sliding window.
    Sliding,
    /// General sparse pattern (caller-defined).
    Sparse,
    /// Block-sparse attention (fixed block size).
    Block,
    /// Longformer-style global + local sliding window.
    Longformer,
    /// BigBird-style: global + sliding window + random blocks.
    BigBird,
}

impl fmt::Display for AttentionPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Dense => "Dense",
            Self::Causal => "Causal",
            Self::Sliding => "Sliding",
            Self::Sparse => "Sparse",
            Self::Block => "Block",
            Self::Longformer => "Longformer",
            Self::BigBird => "BigBird",
        };
        f.write_str(name)
    }
}

impl AttentionPattern {
    /// Returns `true` if the pattern is causal (autoregressive).
    pub const fn is_causal(&self) -> bool {
        matches!(self, Self::Causal)
    }

    /// Returns `true` if the pattern uses sparse representations.
    pub const fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse | Self::Block | Self::Longformer | Self::BigBird)
    }

    /// Theoretical sparsity ratio for a given sequence length and config.
    #[allow(clippy::cast_precision_loss)]
    pub fn sparsity_ratio(&self, seq_len: usize, config: &PatternConfig) -> f64 {
        if seq_len == 0 {
            return 0.0;
        }
        let total = (seq_len * seq_len) as f64;
        let active = match self {
            Self::Dense => total,
            Self::Causal => {
                let n = seq_len as f64;
                n * (n + 1.0) / 2.0
            }
            Self::Sliding => {
                let w = config.window_size.min(seq_len);
                let mut count = 0usize;
                for i in 0..seq_len {
                    let lo = i.saturating_sub(w / 2);
                    let hi = (i + w / 2 + 1).min(seq_len);
                    count += hi - lo;
                }
                count as f64
            }
            Self::Sparse | Self::Block | Self::Longformer | Self::BigBird => {
                // Estimate: window + global tokens
                let w = config.window_size.min(seq_len);
                let g = config.global_tokens.min(seq_len);
                let window_active = {
                    let mut c = 0usize;
                    for i in 0..seq_len {
                        let lo = i.saturating_sub(w / 2);
                        let hi = (i + w / 2 + 1).min(seq_len);
                        c += hi - lo;
                    }
                    c
                };
                let global_active = g * seq_len * 2; // row + col
                (window_active + global_active).min(seq_len * seq_len) as f64
            }
        };
        1.0 - (active / total)
    }
}

// ---------------------------------------------------------------------------
// 2. PatternConfig
// ---------------------------------------------------------------------------

/// Configuration for attention pattern generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternConfig {
    /// Window size for sliding-window patterns.
    pub window_size: usize,
    /// Block size for block-sparse patterns.
    pub block_size: usize,
    /// Number of global tokens (Longformer / `BigBird`).
    pub global_tokens: usize,
    /// Number of random blocks per row (`BigBird`).
    pub random_blocks: usize,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self { window_size: 256, block_size: 64, global_tokens: 0, random_blocks: 0 }
    }
}

impl PatternConfig {
    /// Create a new config with the given window size.
    #[must_use]
    pub const fn with_window(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set block size.
    #[must_use]
    pub const fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    /// Set global token count.
    #[must_use]
    pub const fn with_global_tokens(mut self, count: usize) -> Self {
        self.global_tokens = count;
        self
    }

    /// Set random block count.
    #[must_use]
    pub const fn with_random_blocks(mut self, count: usize) -> Self {
        self.random_blocks = count;
        self
    }

    /// Validate the config.
    pub fn validate(&self) -> Result<(), AttentionPatternError> {
        if self.block_size == 0 {
            return Err(AttentionPatternError::InvalidConfig("block_size must be > 0".into()));
        }
        if self.window_size == 0 {
            return Err(AttentionPatternError::InvalidConfig("window_size must be > 0".into()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 3. MaskGenerator
// ---------------------------------------------------------------------------

/// Generates attention masks from a pattern and config.
pub struct MaskGenerator {
    pattern: AttentionPattern,
    config: PatternConfig,
}

impl MaskGenerator {
    /// Create a new mask generator.
    pub const fn new(pattern: AttentionPattern, config: PatternConfig) -> Self {
        Self { pattern, config }
    }

    /// Generate a dense boolean mask of shape `[seq_len, seq_len]`.
    ///
    /// `true` means the position is *attended*.
    #[allow(clippy::too_many_lines)]
    pub fn generate(&self, seq_len: usize) -> Result<Vec<bool>, AttentionPatternError> {
        if seq_len == 0 {
            return Err(AttentionPatternError::InvalidSequenceLength("seq_len must be > 0".into()));
        }
        self.config.validate()?;

        let mut mask = vec![false; seq_len * seq_len];
        match self.pattern {
            AttentionPattern::Dense => mask.fill(true),
            AttentionPattern::Causal => {
                for i in 0..seq_len {
                    for j in 0..=i {
                        mask[i * seq_len + j] = true;
                    }
                }
            }
            AttentionPattern::Sliding => {
                let half = self.config.window_size / 2;
                for i in 0..seq_len {
                    let lo = i.saturating_sub(half);
                    let hi = (i + half + 1).min(seq_len);
                    for j in lo..hi {
                        mask[i * seq_len + j] = true;
                    }
                }
            }
            AttentionPattern::Sparse => {
                // Sparse uses the sliding window as a baseline.
                let half = self.config.window_size / 2;
                for i in 0..seq_len {
                    let lo = i.saturating_sub(half);
                    let hi = (i + half + 1).min(seq_len);
                    for j in lo..hi {
                        mask[i * seq_len + j] = true;
                    }
                }
            }
            AttentionPattern::Block => {
                let bs = self.config.block_size;
                if !seq_len.is_multiple_of(bs) {
                    return Err(AttentionPatternError::BlockSizeMismatch {
                        seq_len,
                        block_size: bs,
                    });
                }
                let n_blocks = seq_len / bs;
                for b in 0..n_blocks {
                    let start = b * bs;
                    let end = start + bs;
                    for i in start..end {
                        for j in start..end {
                            mask[i * seq_len + j] = true;
                        }
                    }
                }
            }
            AttentionPattern::Longformer => {
                // Sliding window
                let half = self.config.window_size / 2;
                for i in 0..seq_len {
                    let lo = i.saturating_sub(half);
                    let hi = (i + half + 1).min(seq_len);
                    for j in lo..hi {
                        mask[i * seq_len + j] = true;
                    }
                }
                // Global tokens: first `global_tokens` positions attend everywhere
                for g in 0..self.config.global_tokens.min(seq_len) {
                    for j in 0..seq_len {
                        mask[g * seq_len + j] = true; // global row
                        mask[j * seq_len + g] = true; // global col
                    }
                }
            }
            AttentionPattern::BigBird => {
                // Sliding window
                let half = self.config.window_size / 2;
                for i in 0..seq_len {
                    let lo = i.saturating_sub(half);
                    let hi = (i + half + 1).min(seq_len);
                    for j in lo..hi {
                        mask[i * seq_len + j] = true;
                    }
                }
                // Global tokens
                for g in 0..self.config.global_tokens.min(seq_len) {
                    for j in 0..seq_len {
                        mask[g * seq_len + j] = true;
                        mask[j * seq_len + g] = true;
                    }
                }
                // Random blocks (deterministic with simple hash for reproducibility)
                if self.config.block_size > 0 && self.config.random_blocks > 0 {
                    let bs = self.config.block_size;
                    let n_blocks = seq_len / bs.max(1);
                    for i in 0..seq_len {
                        let block_i = i / bs.max(1);
                        for r in 0..self.config.random_blocks {
                            let target = (block_i.wrapping_mul(31) + r.wrapping_mul(17) + 7)
                                % n_blocks.max(1);
                            let start = target * bs;
                            let end = (start + bs).min(seq_len);
                            for j in start..end {
                                mask[i * seq_len + j] = true;
                            }
                        }
                    }
                }
            }
        }
        Ok(mask)
    }

    /// Count the number of attended positions in the generated mask.
    pub fn count_active(&self, seq_len: usize) -> Result<usize, AttentionPatternError> {
        let mask = self.generate(seq_len)?;
        Ok(mask.iter().filter(|&&v| v).count())
    }

    /// Return the pattern used by this generator.
    pub const fn pattern(&self) -> AttentionPattern {
        self.pattern
    }

    /// Return a reference to the config.
    pub const fn config(&self) -> &PatternConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// 4. SparseAttentionMask – compressed block-sparse representation
// ---------------------------------------------------------------------------

/// A compressed block-sparse attention mask.
///
/// Stores which blocks are active in a block-sparse grid, along with the
/// associated metadata for efficient GPU dispatch.
#[derive(Debug, Clone)]
pub struct SparseAttentionMask {
    /// Sequence length.
    pub seq_len: usize,
    /// Block size.
    pub block_size: usize,
    /// Number of blocks along each axis.
    pub num_blocks: usize,
    /// Flat bitmap: `active_blocks[row * num_blocks + col]` is `true` if block
    /// at `(row, col)` is active.
    pub active_blocks: Vec<bool>,
}

impl SparseAttentionMask {
    /// Build a `SparseAttentionMask` from a dense boolean mask.
    pub fn from_dense(
        mask: &[bool],
        seq_len: usize,
        block_size: usize,
    ) -> Result<Self, AttentionPatternError> {
        if block_size == 0 {
            return Err(AttentionPatternError::InvalidConfig("block_size must be > 0".into()));
        }
        if !seq_len.is_multiple_of(block_size) {
            return Err(AttentionPatternError::BlockSizeMismatch { seq_len, block_size });
        }
        if mask.len() != seq_len * seq_len {
            return Err(AttentionPatternError::InvalidSequenceLength(format!(
                "expected mask len {}, got {}",
                seq_len * seq_len,
                mask.len()
            )));
        }

        let num_blocks = seq_len / block_size;
        let mut active_blocks = vec![false; num_blocks * num_blocks];

        for br in 0..num_blocks {
            for bc in 0..num_blocks {
                let mut any_active = false;
                'block: for i in 0..block_size {
                    for j in 0..block_size {
                        let row = br * block_size + i;
                        let col = bc * block_size + j;
                        if mask[row * seq_len + col] {
                            any_active = true;
                            break 'block;
                        }
                    }
                }
                active_blocks[br * num_blocks + bc] = any_active;
            }
        }

        Ok(Self { seq_len, block_size, num_blocks, active_blocks })
    }

    /// Number of active blocks.
    pub fn active_count(&self) -> usize {
        self.active_blocks.iter().filter(|&&b| b).count()
    }

    /// Total number of blocks.
    pub const fn total_blocks(&self) -> usize {
        self.num_blocks * self.num_blocks
    }

    /// Sparsity ratio (fraction of inactive blocks).
    #[allow(clippy::cast_precision_loss)]
    pub fn sparsity(&self) -> f64 {
        let total = self.total_blocks();
        if total == 0 {
            return 0.0;
        }
        1.0 - (self.active_count() as f64 / total as f64)
    }

    /// Check if block `(block_row, block_col)` is active.
    pub fn is_block_active(&self, block_row: usize, block_col: usize) -> bool {
        if block_row >= self.num_blocks || block_col >= self.num_blocks {
            return false;
        }
        self.active_blocks[block_row * self.num_blocks + block_col]
    }

    /// Return list of active block indices as `(row, col)`.
    pub fn active_block_indices(&self) -> Vec<(usize, usize)> {
        let mut indices = Vec::new();
        for r in 0..self.num_blocks {
            for c in 0..self.num_blocks {
                if self.active_blocks[r * self.num_blocks + c] {
                    indices.push((r, c));
                }
            }
        }
        indices
    }

    /// Expand back to a dense boolean mask.
    pub fn to_dense(&self) -> Vec<bool> {
        let mut mask = vec![false; self.seq_len * self.seq_len];
        for br in 0..self.num_blocks {
            for bc in 0..self.num_blocks {
                if self.active_blocks[br * self.num_blocks + bc] {
                    for i in 0..self.block_size {
                        for j in 0..self.block_size {
                            let row = br * self.block_size + i;
                            let col = bc * self.block_size + j;
                            mask[row * self.seq_len + col] = true;
                        }
                    }
                }
            }
        }
        mask
    }
}

// ---------------------------------------------------------------------------
// 5. CausalMaskBuilder
// ---------------------------------------------------------------------------

/// Builds lower-triangular causal masks with optional prefix length.
pub struct CausalMaskBuilder {
    /// Number of prefix tokens that can attend to all positions.
    prefix_len: usize,
}

impl CausalMaskBuilder {
    /// Create a builder with no prefix.
    pub const fn new() -> Self {
        Self { prefix_len: 0 }
    }

    /// Create a builder with a prefix (e.g., for encoder-decoder cross-attn).
    pub const fn with_prefix(prefix_len: usize) -> Self {
        Self { prefix_len }
    }

    /// Generate a causal mask for `seq_len`.
    pub fn build(&self, seq_len: usize) -> Result<Vec<bool>, AttentionPatternError> {
        if seq_len == 0 {
            return Err(AttentionPatternError::InvalidSequenceLength("seq_len must be > 0".into()));
        }
        let mut mask = vec![false; seq_len * seq_len];
        for i in 0..seq_len {
            if i < self.prefix_len {
                // Prefix tokens attend to all positions.
                for j in 0..seq_len {
                    mask[i * seq_len + j] = true;
                }
            } else {
                for j in 0..=i {
                    mask[i * seq_len + j] = true;
                }
            }
        }
        Ok(mask)
    }

    /// Return the prefix length.
    pub const fn prefix_len(&self) -> usize {
        self.prefix_len
    }

    /// Count active (attended) positions for the given `seq_len`.
    pub fn count_active(&self, seq_len: usize) -> Result<usize, AttentionPatternError> {
        let mask = self.build(seq_len)?;
        Ok(mask.iter().filter(|&&v| v).count())
    }
}

impl Default for CausalMaskBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 6. SlidingWindowMask
// ---------------------------------------------------------------------------

/// Builds sliding-window attention masks.
pub struct SlidingWindowMask {
    /// Window size (total, centered on each token).
    window_size: usize,
    /// If true, apply causal constraint (no future tokens).
    causal: bool,
}

impl SlidingWindowMask {
    /// New symmetric sliding window.
    pub const fn new(window_size: usize) -> Self {
        Self { window_size, causal: false }
    }

    /// New causal sliding window (only past + current within window).
    pub const fn causal(window_size: usize) -> Self {
        Self { window_size, causal: true }
    }

    /// Generate the mask.
    pub fn build(&self, seq_len: usize) -> Result<Vec<bool>, AttentionPatternError> {
        if seq_len == 0 {
            return Err(AttentionPatternError::InvalidSequenceLength("seq_len must be > 0".into()));
        }
        if self.window_size == 0 {
            return Err(AttentionPatternError::InvalidConfig("window_size must be > 0".into()));
        }

        let half = self.window_size / 2;
        let mut mask = vec![false; seq_len * seq_len];
        for i in 0..seq_len {
            let lo = i.saturating_sub(half);
            let hi = if self.causal {
                (i + 1).min(seq_len) // only current + past
            } else {
                (i + half + 1).min(seq_len)
            };
            for j in lo..hi {
                mask[i * seq_len + j] = true;
            }
        }
        Ok(mask)
    }

    /// Window size.
    pub const fn window_size(&self) -> usize {
        self.window_size
    }

    /// Whether this is a causal window.
    pub const fn is_causal(&self) -> bool {
        self.causal
    }
}

// ---------------------------------------------------------------------------
// 7. GlobalLocalMask – Longformer-style
// ---------------------------------------------------------------------------

/// Longformer-style global + local attention mask builder.
pub struct GlobalLocalMask {
    /// Indices of global tokens.
    global_indices: Vec<usize>,
    /// Local window size.
    window_size: usize,
}

impl GlobalLocalMask {
    /// Create with global token indices and local window size.
    pub const fn new(global_indices: Vec<usize>, window_size: usize) -> Self {
        Self { global_indices, window_size }
    }

    /// Generate the mask.
    pub fn build(&self, seq_len: usize) -> Result<Vec<bool>, AttentionPatternError> {
        if seq_len == 0 {
            return Err(AttentionPatternError::InvalidSequenceLength("seq_len must be > 0".into()));
        }
        if self.window_size == 0 {
            return Err(AttentionPatternError::InvalidConfig("window_size must be > 0".into()));
        }
        let half = self.window_size / 2;
        let mut mask = vec![false; seq_len * seq_len];

        // Local sliding window
        for i in 0..seq_len {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(seq_len);
            for j in lo..hi {
                mask[i * seq_len + j] = true;
            }
        }

        // Global tokens: full row and full column
        for &g in &self.global_indices {
            if g >= seq_len {
                continue;
            }
            for j in 0..seq_len {
                mask[g * seq_len + j] = true; // row
                mask[j * seq_len + g] = true; // col
            }
        }

        Ok(mask)
    }

    /// Return global token indices.
    pub fn global_indices(&self) -> &[usize] {
        &self.global_indices
    }

    /// Return window size.
    pub const fn window_size(&self) -> usize {
        self.window_size
    }

    /// Count active positions.
    pub fn count_active(&self, seq_len: usize) -> Result<usize, AttentionPatternError> {
        let mask = self.build(seq_len)?;
        Ok(mask.iter().filter(|&&v| v).count())
    }
}

// ---------------------------------------------------------------------------
// 8. BlockSparsePattern – BigBird-style
// ---------------------------------------------------------------------------

/// Block-sparse attention pattern for BigBird-style models.
///
/// Combines: block-diagonal + sliding-window-blocks + global blocks + random blocks.
pub struct BlockSparsePattern {
    /// Block size.
    block_size: usize,
    /// Number of sliding-window blocks on each side.
    window_blocks: usize,
    /// Indices of global blocks.
    global_block_indices: Vec<usize>,
    /// Number of random blocks per row of blocks.
    random_blocks: usize,
}

impl BlockSparsePattern {
    /// Create a new block-sparse pattern.
    pub const fn new(
        block_size: usize,
        window_blocks: usize,
        global_block_indices: Vec<usize>,
        random_blocks: usize,
    ) -> Self {
        Self { block_size, window_blocks, global_block_indices, random_blocks }
    }

    /// Generate block-level mask (`num_blocks x num_blocks`).
    pub fn build_block_mask(&self, seq_len: usize) -> Result<Vec<bool>, AttentionPatternError> {
        if self.block_size == 0 {
            return Err(AttentionPatternError::InvalidConfig("block_size must be > 0".into()));
        }
        if seq_len == 0 {
            return Err(AttentionPatternError::InvalidSequenceLength("seq_len must be > 0".into()));
        }
        if !seq_len.is_multiple_of(self.block_size) {
            return Err(AttentionPatternError::BlockSizeMismatch {
                seq_len,
                block_size: self.block_size,
            });
        }

        let nb = seq_len / self.block_size;
        let mut bmask = vec![false; nb * nb];

        // Block-diagonal
        for b in 0..nb {
            bmask[b * nb + b] = true;
        }

        // Sliding window blocks
        for b in 0..nb {
            for d in 1..=self.window_blocks {
                if b + d < nb {
                    bmask[b * nb + (b + d)] = true;
                    bmask[(b + d) * nb + b] = true;
                }
            }
        }

        // Global blocks
        for &g in &self.global_block_indices {
            if g < nb {
                for b in 0..nb {
                    bmask[g * nb + b] = true;
                    bmask[b * nb + g] = true;
                }
            }
        }

        // Random blocks (deterministic)
        for b in 0..nb {
            for r in 0..self.random_blocks {
                let target = (b.wrapping_mul(31) + r.wrapping_mul(17) + 7) % nb;
                bmask[b * nb + target] = true;
                bmask[target * nb + b] = true;
            }
        }

        Ok(bmask)
    }

    /// Expand block mask to element-level dense mask.
    pub fn build_dense_mask(&self, seq_len: usize) -> Result<Vec<bool>, AttentionPatternError> {
        let nb = seq_len / self.block_size;
        let bmask = self.build_block_mask(seq_len)?;
        let mut mask = vec![false; seq_len * seq_len];

        for br in 0..nb {
            for bc in 0..nb {
                if bmask[br * nb + bc] {
                    for i in 0..self.block_size {
                        for j in 0..self.block_size {
                            let row = br * self.block_size + i;
                            let col = bc * self.block_size + j;
                            mask[row * seq_len + col] = true;
                        }
                    }
                }
            }
        }

        Ok(mask)
    }

    /// Block size.
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of window blocks per side.
    pub const fn window_blocks(&self) -> usize {
        self.window_blocks
    }

    /// Active block count for a given `seq_len`.
    pub fn active_block_count(&self, seq_len: usize) -> Result<usize, AttentionPatternError> {
        let bmask = self.build_block_mask(seq_len)?;
        Ok(bmask.iter().filter(|&&v| v).count())
    }
}

// ---------------------------------------------------------------------------
// 9. PatternAnalyzer – analyze attention weights
// ---------------------------------------------------------------------------

/// Analysis result from inspecting attention weights.
#[derive(Debug, Clone)]
pub struct PatternAnalysis {
    /// Detected dominant pattern.
    pub detected_pattern: AttentionPattern,
    /// Fraction of weight mass on the diagonal.
    pub diagonal_mass: f64,
    /// Fraction of weight mass in the lower triangle.
    pub causal_mass: f64,
    /// Average bandwidth (non-zero band width around diagonal).
    pub avg_bandwidth: f64,
    /// Sparsity (fraction of near-zero entries, below threshold).
    pub sparsity: f64,
    /// Entropy of the average attention distribution.
    pub avg_entropy: f64,
}

/// Analyzes attention weight matrices to detect patterns.
pub struct PatternAnalyzer {
    /// Threshold below which a weight is considered zero.
    zero_threshold: f64,
}

impl PatternAnalyzer {
    /// Create with default threshold (1e-4).
    pub const fn new() -> Self {
        Self { zero_threshold: 1e-4 }
    }

    /// Create with a custom zero threshold.
    pub const fn with_threshold(threshold: f64) -> Self {
        Self { zero_threshold: threshold }
    }

    /// Analyze an attention weight matrix of shape `[seq_len, seq_len]`.
    ///
    /// Weights are in row-major order and should be non-negative (post-softmax).
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(
        &self,
        weights: &[f64],
        seq_len: usize,
    ) -> Result<PatternAnalysis, AttentionPatternError> {
        if seq_len == 0 {
            return Err(AttentionPatternError::InvalidSequenceLength("seq_len must be > 0".into()));
        }
        if weights.len() != seq_len * seq_len {
            return Err(AttentionPatternError::InvalidSequenceLength(format!(
                "expected weights len {}, got {}",
                seq_len * seq_len,
                weights.len()
            )));
        }

        let total: f64 = weights.iter().sum();
        let total = if total > 0.0 { total } else { 1.0 };

        // Diagonal mass
        let diag_sum: f64 = (0..seq_len).map(|i| weights[i * seq_len + i]).sum();
        let diagonal_mass = diag_sum / total;

        // Causal (lower-triangle) mass
        let mut lower_sum = 0.0;
        for i in 0..seq_len {
            for j in 0..=i {
                lower_sum += weights[i * seq_len + j];
            }
        }
        let causal_mass = lower_sum / total;

        // Sparsity
        let zero_count = weights.iter().filter(|&&w| w.abs() < self.zero_threshold).count();
        let sparsity = zero_count as f64 / weights.len() as f64;

        // Average bandwidth
        let mut total_bw = 0.0;
        for i in 0..seq_len {
            let mut max_dist = 0usize;
            for j in 0..seq_len {
                if weights[i * seq_len + j].abs() >= self.zero_threshold {
                    let dist = j.abs_diff(i);
                    max_dist = max_dist.max(dist);
                }
            }
            total_bw += max_dist as f64;
        }
        let avg_bandwidth = total_bw / seq_len as f64;

        // Average row entropy
        let mut total_entropy = 0.0;
        for i in 0..seq_len {
            let row_start = i * seq_len;
            let row = &weights[row_start..row_start + seq_len];
            let row_sum: f64 = row.iter().sum();
            if row_sum > 0.0 {
                let mut entropy = 0.0;
                for &w in row {
                    let p = w / row_sum;
                    if p > 0.0 {
                        entropy -= p * p.ln();
                    }
                }
                total_entropy += entropy;
            }
        }
        let avg_entropy = total_entropy / seq_len as f64;

        // Pattern detection heuristic
        let detected_pattern = if causal_mass > 0.99 {
            AttentionPattern::Causal
        } else if sparsity > 0.7 && avg_bandwidth < (seq_len as f64 * 0.3) {
            AttentionPattern::Sliding
        } else if sparsity > 0.5 {
            AttentionPattern::Sparse
        } else {
            AttentionPattern::Dense
        };

        Ok(PatternAnalysis {
            detected_pattern,
            diagonal_mass,
            causal_mass,
            avg_bandwidth,
            sparsity,
            avg_entropy,
        })
    }

    /// Return the zero threshold.
    pub const fn zero_threshold(&self) -> f64 {
        self.zero_threshold
    }
}

impl Default for PatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 10. AttentionPatternEngine – orchestrator
// ---------------------------------------------------------------------------

/// Orchestrator: configure pattern → build mask → apply → analyze.
pub struct AttentionPatternEngine {
    pattern: AttentionPattern,
    config: PatternConfig,
    analyzer: PatternAnalyzer,
}

impl AttentionPatternEngine {
    /// Create a new engine with the given pattern and config.
    pub const fn new(pattern: AttentionPattern, config: PatternConfig) -> Self {
        Self { pattern, config, analyzer: PatternAnalyzer::new() }
    }

    /// Set analyzer threshold.
    #[must_use]
    pub const fn with_analyzer_threshold(mut self, threshold: f64) -> Self {
        self.analyzer = PatternAnalyzer::with_threshold(threshold);
        self
    }

    /// Generate the attention mask for `seq_len`.
    pub fn generate_mask(&self, seq_len: usize) -> Result<Vec<bool>, AttentionPatternError> {
        let mgen = MaskGenerator::new(self.pattern, self.config.clone());
        mgen.generate(seq_len)
    }

    /// Generate a `SparseAttentionMask` for `seq_len`.
    pub fn generate_sparse_mask(
        &self,
        seq_len: usize,
    ) -> Result<SparseAttentionMask, AttentionPatternError> {
        let mask = self.generate_mask(seq_len)?;
        SparseAttentionMask::from_dense(&mask, seq_len, self.config.block_size)
    }

    /// Apply the mask to attention weights (zeroing out masked positions).
    ///
    /// `weights` is `[seq_len, seq_len]` in row-major order.
    pub fn apply_mask(
        &self,
        weights: &mut [f64],
        seq_len: usize,
    ) -> Result<(), AttentionPatternError> {
        let mask = self.generate_mask(seq_len)?;
        if weights.len() != seq_len * seq_len {
            return Err(AttentionPatternError::InvalidSequenceLength(format!(
                "expected weights len {}, got {}",
                seq_len * seq_len,
                weights.len()
            )));
        }
        for (w, &m) in weights.iter_mut().zip(mask.iter()) {
            if !m {
                *w = 0.0;
            }
        }
        Ok(())
    }

    /// Analyze attention weights to detect the dominant pattern.
    pub fn analyze(
        &self,
        weights: &[f64],
        seq_len: usize,
    ) -> Result<PatternAnalysis, AttentionPatternError> {
        self.analyzer.analyze(weights, seq_len)
    }

    /// Full pipeline: generate mask → apply to weights → analyze result.
    pub fn process(
        &self,
        weights: &mut [f64],
        seq_len: usize,
    ) -> Result<PatternAnalysis, AttentionPatternError> {
        self.apply_mask(weights, seq_len)?;
        self.analyze(weights, seq_len)
    }

    /// Return the configured pattern.
    pub const fn pattern(&self) -> AttentionPattern {
        self.pattern
    }

    /// Return a reference to the config.
    pub const fn config(&self) -> &PatternConfig {
        &self.config
    }

    /// Compute theoretical sparsity for the given `seq_len`.
    pub fn theoretical_sparsity(&self, seq_len: usize) -> f64 {
        self.pattern.sparsity_ratio(seq_len, &self.config)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- AttentionPattern enum tests ----------------------------------------

    #[test]
    fn test_pattern_display() {
        assert_eq!(AttentionPattern::Dense.to_string(), "Dense");
        assert_eq!(AttentionPattern::Causal.to_string(), "Causal");
        assert_eq!(AttentionPattern::Sliding.to_string(), "Sliding");
        assert_eq!(AttentionPattern::Sparse.to_string(), "Sparse");
        assert_eq!(AttentionPattern::Block.to_string(), "Block");
        assert_eq!(AttentionPattern::Longformer.to_string(), "Longformer");
        assert_eq!(AttentionPattern::BigBird.to_string(), "BigBird");
    }

    #[test]
    fn test_pattern_is_causal() {
        assert!(AttentionPattern::Causal.is_causal());
        assert!(!AttentionPattern::Dense.is_causal());
        assert!(!AttentionPattern::Sliding.is_causal());
    }

    #[test]
    fn test_pattern_is_sparse() {
        assert!(AttentionPattern::Sparse.is_sparse());
        assert!(AttentionPattern::Block.is_sparse());
        assert!(AttentionPattern::Longformer.is_sparse());
        assert!(AttentionPattern::BigBird.is_sparse());
        assert!(!AttentionPattern::Dense.is_sparse());
        assert!(!AttentionPattern::Causal.is_sparse());
        assert!(!AttentionPattern::Sliding.is_sparse());
    }

    #[test]
    fn test_pattern_eq_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(AttentionPattern::Dense);
        set.insert(AttentionPattern::Dense);
        assert_eq!(set.len(), 1);
        set.insert(AttentionPattern::Causal);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_pattern_clone_copy() {
        let p = AttentionPattern::Sliding;
        let p2 = p;
        assert_eq!(p, p2);
    }

    #[test]
    fn test_sparsity_ratio_dense() {
        let cfg = PatternConfig::default();
        let r = AttentionPattern::Dense.sparsity_ratio(8, &cfg);
        assert!((r - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_sparsity_ratio_causal() {
        let cfg = PatternConfig::default();
        let r = AttentionPattern::Causal.sparsity_ratio(4, &cfg);
        // 4*5/2 = 10 active out of 16 → sparsity = 6/16 = 0.375
        assert!((r - 0.375).abs() < 1e-9);
    }

    #[test]
    fn test_sparsity_ratio_zero_seq() {
        let cfg = PatternConfig::default();
        assert_eq!(AttentionPattern::Dense.sparsity_ratio(0, &cfg), 0.0);
    }

    // -- PatternConfig tests ------------------------------------------------

    #[test]
    fn test_config_default() {
        let cfg = PatternConfig::default();
        assert_eq!(cfg.window_size, 256);
        assert_eq!(cfg.block_size, 64);
        assert_eq!(cfg.global_tokens, 0);
        assert_eq!(cfg.random_blocks, 0);
    }

    #[test]
    fn test_config_builder_chain() {
        let cfg = PatternConfig::default()
            .with_window(128)
            .with_block_size(32)
            .with_global_tokens(4)
            .with_random_blocks(2);
        assert_eq!(cfg.window_size, 128);
        assert_eq!(cfg.block_size, 32);
        assert_eq!(cfg.global_tokens, 4);
        assert_eq!(cfg.random_blocks, 2);
    }

    #[test]
    fn test_config_validate_ok() {
        assert!(PatternConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_block() {
        let cfg = PatternConfig::default().with_block_size(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_window() {
        let cfg = PatternConfig::default().with_window(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_eq() {
        let a = PatternConfig::default();
        let b = PatternConfig::default();
        assert_eq!(a, b);
    }

    // -- MaskGenerator tests -----------------------------------------------

    #[test]
    fn test_mask_dense_4x4() {
        let mgen = MaskGenerator::new(AttentionPattern::Dense, PatternConfig::default());
        let mask = mgen.generate(4).unwrap();
        assert_eq!(mask.len(), 16);
        assert!(mask.iter().all(|&v| v));
    }

    #[test]
    fn test_mask_causal_4x4() {
        let mgen = MaskGenerator::new(AttentionPattern::Causal, PatternConfig::default());
        let mask = mgen.generate(4).unwrap();
        // Row 0: [T F F F], Row 1: [T T F F], Row 2: [T T T F], Row 3: [T T T T]
        let expected = [
            true, false, false, false, true, true, false, false, true, true, true, false, true,
            true, true, true,
        ];
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_mask_causal_1x1() {
        let mgen = MaskGenerator::new(AttentionPattern::Causal, PatternConfig::default());
        let mask = mgen.generate(1).unwrap();
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn test_mask_sliding_window_3() {
        let cfg = PatternConfig::default().with_window(3);
        let mgen = MaskGenerator::new(AttentionPattern::Sliding, cfg);
        let mask = mgen.generate(5).unwrap();
        // window=3, half=1 → each token attends to [i-1, i, i+1]
        for i in 0..5 {
            for j in 0..5 {
                let in_window = (j as isize - i as isize).unsigned_abs() <= 1;
                assert_eq!(mask[i * 5 + j], in_window, "mismatch at ({i}, {j})");
            }
        }
    }

    #[test]
    fn test_mask_sparse_uses_window() {
        let cfg = PatternConfig::default().with_window(3);
        let sparse = MaskGenerator::new(AttentionPattern::Sparse, cfg.clone());
        let sliding = MaskGenerator::new(AttentionPattern::Sliding, cfg);
        assert_eq!(sparse.generate(5).unwrap(), sliding.generate(5).unwrap());
    }

    #[test]
    fn test_mask_block_4x4_bs2() {
        let cfg = PatternConfig::default().with_block_size(2);
        let mgen = MaskGenerator::new(AttentionPattern::Block, cfg);
        let mask = mgen.generate(4).unwrap();
        // Block [0,1] and block [2,3]
        let expected = [
            true, true, false, false, true, true, false, false, false, false, true, true, false,
            false, true, true,
        ];
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_mask_block_misaligned() {
        let cfg = PatternConfig::default().with_block_size(3);
        let mgen = MaskGenerator::new(AttentionPattern::Block, cfg);
        let result = mgen.generate(4);
        assert!(matches!(result, Err(AttentionPatternError::BlockSizeMismatch { .. })));
    }

    #[test]
    fn test_mask_longformer() {
        let cfg = PatternConfig::default().with_window(3).with_global_tokens(1);
        let mgen = MaskGenerator::new(AttentionPattern::Longformer, cfg);
        let mask = mgen.generate(4).unwrap();
        // Token 0 is global → row 0 all true, col 0 all true
        for j in 0..4 {
            assert!(mask[0 * 4 + j], "global row at col {j}");
            assert!(mask[j * 4 + 0], "global col at row {j}");
        }
    }

    #[test]
    fn test_mask_bigbird_has_window_and_global() {
        let cfg = PatternConfig::default()
            .with_window(3)
            .with_global_tokens(1)
            .with_block_size(2)
            .with_random_blocks(1);
        let mgen = MaskGenerator::new(AttentionPattern::BigBird, cfg);
        let mask = mgen.generate(4).unwrap();
        // Global: token 0 → full row & col
        for j in 0..4 {
            assert!(mask[j], "global row 0 col {j}");
            assert!(mask[j * 4], "global col 0 row {j}");
        }
        // Diagonal should always be attended (window)
        for i in 0..4 {
            assert!(mask[i * 4 + i], "diagonal at {i}");
        }
    }

    #[test]
    fn test_mask_zero_seq_len() {
        let mgen = MaskGenerator::new(AttentionPattern::Dense, PatternConfig::default());
        assert!(mgen.generate(0).is_err());
    }

    #[test]
    fn test_mask_count_active_dense() {
        let mgen = MaskGenerator::new(AttentionPattern::Dense, PatternConfig::default());
        assert_eq!(mgen.count_active(4).unwrap(), 16);
    }

    #[test]
    fn test_mask_count_active_causal() {
        let mgen = MaskGenerator::new(AttentionPattern::Causal, PatternConfig::default());
        assert_eq!(mgen.count_active(4).unwrap(), 10); // 1+2+3+4
    }

    #[test]
    fn test_mask_pattern_accessor() {
        let mgen = MaskGenerator::new(AttentionPattern::Dense, PatternConfig::default());
        assert_eq!(mgen.pattern(), AttentionPattern::Dense);
    }

    #[test]
    fn test_mask_config_accessor() {
        let cfg = PatternConfig::default().with_window(42);
        let mgen = MaskGenerator::new(AttentionPattern::Dense, cfg);
        assert_eq!(mgen.config().window_size, 42);
    }

    // -- SparseAttentionMask tests ------------------------------------------

    #[test]
    fn test_sparse_mask_from_dense_block() {
        let cfg = PatternConfig::default().with_block_size(2);
        let mgen = MaskGenerator::new(AttentionPattern::Block, cfg);
        let mask = mgen.generate(4).unwrap();
        let sparse = SparseAttentionMask::from_dense(&mask, 4, 2).unwrap();
        assert_eq!(sparse.num_blocks, 2);
        assert_eq!(sparse.active_count(), 2); // diagonal blocks only
        assert!(sparse.is_block_active(0, 0));
        assert!(sparse.is_block_active(1, 1));
        assert!(!sparse.is_block_active(0, 1));
        assert!(!sparse.is_block_active(1, 0));
    }

    #[test]
    fn test_sparse_mask_from_dense_full() {
        let mask = vec![true; 16];
        let sparse = SparseAttentionMask::from_dense(&mask, 4, 2).unwrap();
        assert_eq!(sparse.active_count(), 4); // all blocks active
        assert!((sparse.sparsity() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_sparse_mask_sparsity() {
        let mask = vec![false; 16];
        let sparse = SparseAttentionMask::from_dense(&mask, 4, 2).unwrap();
        assert_eq!(sparse.active_count(), 0);
        assert!((sparse.sparsity() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sparse_mask_total_blocks() {
        let mask = vec![true; 64];
        let sparse = SparseAttentionMask::from_dense(&mask, 8, 4).unwrap();
        assert_eq!(sparse.total_blocks(), 4);
    }

    #[test]
    fn test_sparse_mask_active_block_indices() {
        let mut mask = vec![false; 16];
        // Activate block (0,0) and block (1,1)
        mask[0] = true;
        mask[2 * 4 + 2] = true;
        let sparse = SparseAttentionMask::from_dense(&mask, 4, 2).unwrap();
        let indices = sparse.active_block_indices();
        assert!(indices.contains(&(0, 0)));
        assert!(indices.contains(&(1, 1)));
    }

    #[test]
    fn test_sparse_mask_to_dense_roundtrip() {
        let cfg = PatternConfig::default().with_block_size(2);
        let mgen = MaskGenerator::new(AttentionPattern::Block, cfg);
        let original = mgen.generate(4).unwrap();
        let sparse = SparseAttentionMask::from_dense(&original, 4, 2).unwrap();
        let reconstructed = sparse.to_dense();
        assert_eq!(original, reconstructed);
    }

    #[test]
    fn test_sparse_mask_invalid_block_size() {
        let mask = vec![true; 16];
        assert!(SparseAttentionMask::from_dense(&mask, 4, 0).is_err());
    }

    #[test]
    fn test_sparse_mask_misaligned() {
        let mask = vec![true; 9];
        assert!(SparseAttentionMask::from_dense(&mask, 3, 2).is_err());
    }

    #[test]
    fn test_sparse_mask_wrong_len() {
        let mask = vec![true; 10];
        assert!(SparseAttentionMask::from_dense(&mask, 4, 2).is_err());
    }

    #[test]
    fn test_sparse_mask_out_of_bounds_check() {
        let mask = vec![true; 16];
        let sparse = SparseAttentionMask::from_dense(&mask, 4, 2).unwrap();
        assert!(!sparse.is_block_active(5, 0));
        assert!(!sparse.is_block_active(0, 5));
    }

    // -- CausalMaskBuilder tests --------------------------------------------

    #[test]
    fn test_causal_builder_basic() {
        let builder = CausalMaskBuilder::new();
        let mask = builder.build(3).unwrap();
        let expected = [true, false, false, true, true, false, true, true, true];
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_causal_builder_with_prefix() {
        let builder = CausalMaskBuilder::with_prefix(2);
        let mask = builder.build(4).unwrap();
        // Prefix rows 0,1 attend everywhere
        for j in 0..4 {
            assert!(mask[0 * 4 + j]);
            assert!(mask[1 * 4 + j]);
        }
        // Row 2: causal
        assert!(mask[2 * 4 + 0]);
        assert!(mask[2 * 4 + 1]);
        assert!(mask[2 * 4 + 2]);
        assert!(!mask[2 * 4 + 3]);
    }

    #[test]
    fn test_causal_builder_prefix_exceeds_seq() {
        let builder = CausalMaskBuilder::with_prefix(10);
        let mask = builder.build(4).unwrap();
        // All rows are prefix → full attention
        assert!(mask.iter().all(|&v| v));
    }

    #[test]
    fn test_causal_builder_zero_seq() {
        let builder = CausalMaskBuilder::new();
        assert!(builder.build(0).is_err());
    }

    #[test]
    fn test_causal_builder_prefix_len_accessor() {
        assert_eq!(CausalMaskBuilder::new().prefix_len(), 0);
        assert_eq!(CausalMaskBuilder::with_prefix(5).prefix_len(), 5);
    }

    #[test]
    fn test_causal_builder_count_active() {
        let builder = CausalMaskBuilder::new();
        assert_eq!(builder.count_active(4).unwrap(), 10);
    }

    #[test]
    fn test_causal_builder_default() {
        let builder = CausalMaskBuilder::default();
        assert_eq!(builder.prefix_len(), 0);
    }

    // -- SlidingWindowMask tests --------------------------------------------

    #[test]
    fn test_sliding_window_symmetric() {
        let sw = SlidingWindowMask::new(3);
        let mask = sw.build(5).unwrap();
        // Each token attends to [i-1, i, i+1]
        for i in 0..5 {
            assert!(mask[i * 5 + i], "diagonal at {i}");
        }
        // Check boundaries
        assert!(!mask[0 * 5 + 2]); // 0 can't see 2
        assert!(mask[2 * 5 + 1]); // 2 can see 1
        assert!(mask[2 * 5 + 3]); // 2 can see 3
    }

    #[test]
    fn test_sliding_window_causal() {
        let sw = SlidingWindowMask::causal(3);
        let mask = sw.build(4).unwrap();
        // Causal: only past within window
        assert!(mask[2 * 4 + 1]); // 2 sees 1
        assert!(mask[2 * 4 + 2]); // 2 sees self
        assert!(!mask[2 * 4 + 3]); // 2 doesn't see 3
        assert!(!mask[0 * 4 + 1]); // 0 doesn't see 1
    }

    #[test]
    fn test_sliding_window_full_coverage() {
        // Window >= 2*(seq_len-1)+1 → all attended
        let sw = SlidingWindowMask::new(9);
        let mask = sw.build(4).unwrap();
        assert!(mask.iter().all(|&v| v));
    }

    #[test]
    fn test_sliding_window_zero_seq() {
        let sw = SlidingWindowMask::new(3);
        assert!(sw.build(0).is_err());
    }

    #[test]
    fn test_sliding_window_zero_window() {
        let sw = SlidingWindowMask::new(0);
        assert!(sw.build(4).is_err());
    }

    #[test]
    fn test_sliding_window_accessors() {
        let sw = SlidingWindowMask::new(7);
        assert_eq!(sw.window_size(), 7);
        assert!(!sw.is_causal());
        let sc = SlidingWindowMask::causal(5);
        assert!(sc.is_causal());
    }

    // -- GlobalLocalMask tests ----------------------------------------------

    #[test]
    fn test_global_local_no_globals() {
        let gl = GlobalLocalMask::new(vec![], 3);
        let sw = SlidingWindowMask::new(3);
        let mask_gl = gl.build(5).unwrap();
        let mask_sw = sw.build(5).unwrap();
        assert_eq!(mask_gl, mask_sw);
    }

    #[test]
    fn test_global_local_with_globals() {
        let gl = GlobalLocalMask::new(vec![0, 2], 1);
        let mask = gl.build(4).unwrap();
        // Token 0 and 2 are global
        for j in 0..4 {
            assert!(mask[0 * 4 + j], "global row 0");
            assert!(mask[j * 4 + 0], "global col 0");
            assert!(mask[2 * 4 + j], "global row 2");
            assert!(mask[j * 4 + 2], "global col 2");
        }
    }

    #[test]
    fn test_global_local_out_of_range_index() {
        let gl = GlobalLocalMask::new(vec![100], 3);
        let mask = gl.build(4).unwrap();
        // out-of-range global is ignored → same as no globals
        let sw = SlidingWindowMask::new(3);
        assert_eq!(mask, sw.build(4).unwrap());
    }

    #[test]
    fn test_global_local_zero_seq() {
        let gl = GlobalLocalMask::new(vec![0], 3);
        assert!(gl.build(0).is_err());
    }

    #[test]
    fn test_global_local_zero_window() {
        let gl = GlobalLocalMask::new(vec![], 0);
        assert!(gl.build(4).is_err());
    }

    #[test]
    fn test_global_local_accessors() {
        let gl = GlobalLocalMask::new(vec![0, 1], 5);
        assert_eq!(gl.global_indices(), &[0, 1]);
        assert_eq!(gl.window_size(), 5);
    }

    #[test]
    fn test_global_local_count_active() {
        let gl = GlobalLocalMask::new(vec![0], 3);
        let count = gl.count_active(4).unwrap();
        assert!(count > 0);
    }

    // -- BlockSparsePattern tests -------------------------------------------

    #[test]
    fn test_block_sparse_diagonal() {
        let bsp = BlockSparsePattern::new(2, 0, vec![], 0);
        let bmask = bsp.build_block_mask(4).unwrap();
        // 2 blocks, diagonal only
        assert!(bmask[0]); // (0,0)
        assert!(!bmask[1]); // (0,1)
        assert!(!bmask[2]); // (1,0)
        assert!(bmask[3]); // (1,1)
    }

    #[test]
    fn test_block_sparse_with_window() {
        let bsp = BlockSparsePattern::new(2, 1, vec![], 0);
        let bmask = bsp.build_block_mask(4).unwrap();
        // Diagonal + 1 neighbor on each side → all blocks active
        assert!(bmask.iter().all(|&v| v));
    }

    #[test]
    fn test_block_sparse_with_global() {
        let bsp = BlockSparsePattern::new(2, 0, vec![0], 0);
        let bmask = bsp.build_block_mask(6).unwrap();
        // 3 blocks; block 0 is global
        let nb = 3;
        for b in 0..nb {
            assert!(bmask[0 * nb + b], "global row");
            assert!(bmask[b * nb + 0], "global col");
        }
    }

    #[test]
    fn test_block_sparse_with_random() {
        let bsp = BlockSparsePattern::new(2, 0, vec![], 1);
        let bmask = bsp.build_block_mask(8).unwrap();
        let nb = 4;
        // At least diagonal blocks active
        for b in 0..nb {
            assert!(bmask[b * nb + b]);
        }
        // Plus random → more than just diagonal
        let active: usize = bmask.iter().filter(|&&v| v).count();
        assert!(active > nb);
    }

    #[test]
    fn test_block_sparse_dense_expansion() {
        let bsp = BlockSparsePattern::new(2, 0, vec![], 0);
        let dense = bsp.build_dense_mask(4).unwrap();
        // Same as MaskGenerator block pattern
        let cfg = PatternConfig::default().with_block_size(2);
        let mgen = MaskGenerator::new(AttentionPattern::Block, cfg);
        assert_eq!(dense, mgen.generate(4).unwrap());
    }

    #[test]
    fn test_block_sparse_misaligned() {
        let bsp = BlockSparsePattern::new(3, 0, vec![], 0);
        assert!(bsp.build_block_mask(4).is_err());
    }

    #[test]
    fn test_block_sparse_zero_block_size() {
        let bsp = BlockSparsePattern::new(0, 0, vec![], 0);
        assert!(bsp.build_block_mask(4).is_err());
    }

    #[test]
    fn test_block_sparse_zero_seq() {
        let bsp = BlockSparsePattern::new(2, 0, vec![], 0);
        assert!(bsp.build_block_mask(0).is_err());
    }

    #[test]
    fn test_block_sparse_accessors() {
        let bsp = BlockSparsePattern::new(4, 2, vec![0], 3);
        assert_eq!(bsp.block_size(), 4);
        assert_eq!(bsp.window_blocks(), 2);
    }

    #[test]
    fn test_block_sparse_active_count() {
        let bsp = BlockSparsePattern::new(2, 0, vec![], 0);
        let count = bsp.active_block_count(4).unwrap();
        assert_eq!(count, 2); // diagonal only
    }

    // -- PatternAnalyzer tests ----------------------------------------------

    fn make_causal_weights(n: usize) -> Vec<f64> {
        let mut w = vec![0.0; n * n];
        for i in 0..n {
            let count = (i + 1) as f64;
            for j in 0..=i {
                w[i * n + j] = 1.0 / count;
            }
        }
        w
    }

    fn make_uniform_weights(n: usize) -> Vec<f64> {
        vec![1.0 / (n * n) as f64; n * n]
    }

    fn make_diagonal_weights(n: usize) -> Vec<f64> {
        let mut w = vec![0.0; n * n];
        for i in 0..n {
            w[i * n + i] = 1.0;
        }
        w
    }

    #[test]
    fn test_analyzer_dense_detection() {
        let analyzer = PatternAnalyzer::new();
        let w = make_uniform_weights(8);
        let analysis = analyzer.analyze(&w, 8).unwrap();
        assert_eq!(analysis.detected_pattern, AttentionPattern::Dense);
    }

    #[test]
    fn test_analyzer_causal_detection() {
        let analyzer = PatternAnalyzer::new();
        let w = make_causal_weights(8);
        let analysis = analyzer.analyze(&w, 8).unwrap();
        assert_eq!(analysis.detected_pattern, AttentionPattern::Causal);
    }

    #[test]
    fn test_analyzer_sparse_detection() {
        let analyzer = PatternAnalyzer::new();
        let n = 16;
        let mut w = vec![0.0; n * n];
        // Only diagonal + one neighbor
        for i in 0..n {
            w[i * n + i] = 0.5;
            if i + 1 < n {
                w[i * n + i + 1] = 0.5;
            }
        }
        let analysis = analyzer.analyze(&w, n).unwrap();
        assert!(analysis.sparsity > 0.5);
    }

    #[test]
    fn test_analyzer_diagonal_mass() {
        let analyzer = PatternAnalyzer::new();
        let w = make_diagonal_weights(4);
        let analysis = analyzer.analyze(&w, 4).unwrap();
        assert!((analysis.diagonal_mass - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_analyzer_causal_mass() {
        let analyzer = PatternAnalyzer::new();
        let w = make_causal_weights(4);
        let analysis = analyzer.analyze(&w, 4).unwrap();
        assert!((analysis.causal_mass - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_analyzer_custom_threshold() {
        let analyzer = PatternAnalyzer::with_threshold(0.5);
        assert!((analyzer.zero_threshold() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_analyzer_zero_seq() {
        let analyzer = PatternAnalyzer::new();
        assert!(analyzer.analyze(&[], 0).is_err());
    }

    #[test]
    fn test_analyzer_wrong_len() {
        let analyzer = PatternAnalyzer::new();
        assert!(analyzer.analyze(&[1.0; 5], 4).is_err());
    }

    #[test]
    fn test_analyzer_default() {
        let a = PatternAnalyzer::default();
        assert!((a.zero_threshold() - 1e-4).abs() < 1e-9);
    }

    #[test]
    fn test_analyzer_entropy_uniform() {
        let analyzer = PatternAnalyzer::new();
        let w = make_uniform_weights(4);
        let analysis = analyzer.analyze(&w, 4).unwrap();
        // Uniform → max entropy ≈ ln(4)
        assert!(analysis.avg_entropy > 1.0);
    }

    #[test]
    fn test_analyzer_entropy_diagonal() {
        let analyzer = PatternAnalyzer::new();
        let w = make_diagonal_weights(4);
        let analysis = analyzer.analyze(&w, 4).unwrap();
        // Diagonal → row entropy = 0 (all mass on one element)
        assert!(analysis.avg_entropy < 0.01);
    }

    // -- AttentionPatternEngine tests ---------------------------------------

    #[test]
    fn test_engine_generate_mask() {
        let engine =
            AttentionPatternEngine::new(AttentionPattern::Causal, PatternConfig::default());
        let mask = engine.generate_mask(4).unwrap();
        assert_eq!(mask.len(), 16);
        assert_eq!(mask.iter().filter(|&&v| v).count(), 10);
    }

    #[test]
    fn test_engine_generate_sparse_mask() {
        let cfg = PatternConfig::default().with_block_size(2);
        let engine = AttentionPatternEngine::new(AttentionPattern::Block, cfg);
        let sparse = engine.generate_sparse_mask(4).unwrap();
        assert_eq!(sparse.active_count(), 2);
    }

    #[test]
    fn test_engine_apply_mask() {
        let engine =
            AttentionPatternEngine::new(AttentionPattern::Causal, PatternConfig::default());
        let mut weights = vec![1.0; 16];
        engine.apply_mask(&mut weights, 4).unwrap();
        // Upper triangle should be zeroed
        assert_eq!(weights[0 * 4 + 1], 0.0);
        assert_eq!(weights[0 * 4 + 2], 0.0);
        assert_eq!(weights[0 * 4 + 3], 0.0);
        // Lower triangle preserved
        assert_eq!(weights[1 * 4 + 0], 1.0);
        assert_eq!(weights[2 * 4 + 1], 1.0);
    }

    #[test]
    fn test_engine_apply_mask_wrong_len() {
        let engine = AttentionPatternEngine::new(AttentionPattern::Dense, PatternConfig::default());
        let mut weights = vec![1.0; 5];
        assert!(engine.apply_mask(&mut weights, 4).is_err());
    }

    #[test]
    fn test_engine_analyze() {
        let engine = AttentionPatternEngine::new(AttentionPattern::Dense, PatternConfig::default());
        let w = make_uniform_weights(4);
        let analysis = engine.analyze(&w, 4).unwrap();
        assert_eq!(analysis.detected_pattern, AttentionPattern::Dense);
    }

    #[test]
    fn test_engine_process_pipeline() {
        let engine =
            AttentionPatternEngine::new(AttentionPattern::Causal, PatternConfig::default());
        let mut weights = make_uniform_weights(4);
        let analysis = engine.process(&mut weights, 4).unwrap();
        // After causal masking + analysis
        assert!((analysis.causal_mass - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_engine_pattern_accessor() {
        let engine =
            AttentionPatternEngine::new(AttentionPattern::Sliding, PatternConfig::default());
        assert_eq!(engine.pattern(), AttentionPattern::Sliding);
    }

    #[test]
    fn test_engine_config_accessor() {
        let cfg = PatternConfig::default().with_window(99);
        let engine = AttentionPatternEngine::new(AttentionPattern::Dense, cfg);
        assert_eq!(engine.config().window_size, 99);
    }

    #[test]
    fn test_engine_theoretical_sparsity() {
        let engine = AttentionPatternEngine::new(AttentionPattern::Dense, PatternConfig::default());
        assert!((engine.theoretical_sparsity(8) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_engine_with_analyzer_threshold() {
        let engine = AttentionPatternEngine::new(AttentionPattern::Dense, PatternConfig::default())
            .with_analyzer_threshold(0.1);
        let w = make_uniform_weights(4);
        let analysis = engine.analyze(&w, 4).unwrap();
        // Uniform weights are 1/16 ≈ 0.0625 < 0.1 threshold → high sparsity
        assert!(analysis.sparsity > 0.5);
    }

    // -- Error display tests ------------------------------------------------

    #[test]
    fn test_error_display_invalid_config() {
        let e = AttentionPatternError::InvalidConfig("test".into());
        assert!(e.to_string().contains("test"));
    }

    #[test]
    fn test_error_display_invalid_seq() {
        let e = AttentionPatternError::InvalidSequenceLength("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn test_error_display_block_mismatch() {
        let e = AttentionPatternError::BlockSizeMismatch { seq_len: 5, block_size: 2 };
        let s = e.to_string();
        assert!(s.contains("5") && s.contains("2"));
    }

    #[test]
    fn test_error_display_missing_field() {
        let e = AttentionPatternError::MissingField("foo".into());
        assert!(e.to_string().contains("foo"));
    }

    #[test]
    fn test_error_eq() {
        let a = AttentionPatternError::InvalidConfig("x".into());
        let b = AttentionPatternError::InvalidConfig("x".into());
        assert_eq!(a, b);
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> =
            Box::new(AttentionPatternError::InvalidConfig("x".into()));
        assert!(e.to_string().contains("x"));
    }
}
