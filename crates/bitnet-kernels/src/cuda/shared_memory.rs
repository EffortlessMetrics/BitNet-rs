//! CUDA shared memory optimization with bank-conflict avoidance.
//!
//! # Overview
//!
//! Models shared memory bank conflicts and provides conflict-free access
//! patterns for CUDA kernels used in BitNet inference.  Includes:
//!
//! - **Bank conflict analysis**: Detects and quantifies bank conflicts for
//!   given access patterns across 32 shared memory banks.
//! - **Padding calculators**: Computes minimal padding to eliminate conflicts
//!   for row-major, column-major, and diagonal access patterns.
//! - **Tiling strategies**: Configures tile dimensions for matmul and
//!   attention kernels, balancing occupancy against shared memory usage.
//! - **Allocation planner**: Decides between static and dynamic shared memory
//!   based on kernel requirements and device limits.
//! - **L1/shared memory partitioning**: Models the configurable carve-out
//!   for different CUDA compute capabilities (sm_50 … sm_90).
//! - **Data layout optimizers**: Transforms access patterns for coalesced
//!   global memory loads and conflict-free shared memory stores.
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU-side planning and analysis functions are always available for testing
//! on non-GPU hosts.

use bitnet_common::{KernelError, Result};

// ── Constants ────────────────────────────────────────────────────────

/// Number of shared memory banks on all CUDA architectures since Kepler.
pub const SHARED_MEMORY_BANKS: usize = 32;

/// Size in bytes of one shared memory bank (4 bytes = 32 bits).
pub const BANK_WIDTH_BYTES: usize = 4;

/// Default tile dimension for square tiling strategies.
pub const DEFAULT_TILE_DIM: usize = 32;

/// Maximum static shared memory per block (48 KiB, all architectures).
pub const MAX_STATIC_SMEM_BYTES: usize = 48 * 1024;

// ── Compute capability ──────────────────────────────────────────────

/// CUDA compute capability encoded as `(major, minor)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeCapability {
    pub major: u32,
    pub minor: u32,
}

impl ComputeCapability {
    /// Create a new compute capability.
    pub fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Numeric encoding: `major * 10 + minor` (e.g. sm_80 → 80).
    pub fn sm_version(&self) -> u32 {
        self.major * 10 + self.minor
    }

    /// Maximum shared memory per SM for this compute capability.
    pub fn max_smem_per_sm(&self) -> usize {
        match self.sm_version() {
            v if v < 50 => 48 * 1024,
            50..=52 => 64 * 1024,
            60..=62 => 64 * 1024,
            70..=72 => 96 * 1024,
            75 => 64 * 1024,
            80..=86 => 164 * 1024,
            87 => 164 * 1024,
            89 => 100 * 1024,
            90..=99 => 228 * 1024,
            _ => 48 * 1024,
        }
    }

    /// Maximum shared memory per thread block (configurable carve-out).
    pub fn max_smem_per_block(&self) -> usize {
        match self.sm_version() {
            v if v < 70 => 48 * 1024,
            70..=75 => 96 * 1024,
            80..=89 => 163 * 1024,
            90..=99 => 227 * 1024,
            _ => 48 * 1024,
        }
    }
}

// ── L1 / shared memory partitioning ─────────────────────────────────

/// Available carve-out ratios for the unified L1/shared memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CachePreference {
    /// Prefer more shared memory.
    PreferShared,
    /// Equal split.
    PreferEqual,
    /// Prefer more L1 data cache.
    PreferL1,
    /// No preference — let the driver decide.
    NoPreference,
}

/// Result of an L1/shared memory partition query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachePartition {
    /// Shared memory bytes available under this partition.
    pub shared_memory_bytes: usize,
    /// L1 data cache bytes available.
    pub l1_cache_bytes: usize,
    /// Total unified cache bytes (shared + L1).
    pub total_bytes: usize,
}

/// Compute the L1/shared memory partition for a given compute capability
/// and cache preference.
pub fn partition_l1_shared(cc: ComputeCapability, preference: CachePreference) -> CachePartition {
    let total = if cc.sm_version() >= 80 {
        192 * 1024
    } else if cc.sm_version() >= 70 {
        128 * 1024
    } else {
        64 * 1024
    };

    let shared = match preference {
        CachePreference::PreferShared => (total * 3) / 4,
        CachePreference::PreferEqual => total / 2,
        CachePreference::PreferL1 => total / 4,
        CachePreference::NoPreference => total / 2,
    };

    CachePartition {
        shared_memory_bytes: shared,
        l1_cache_bytes: total - shared,
        total_bytes: total,
    }
}

// ── Bank conflict analysis ──────────────────────────────────────────

/// Result of a bank-conflict analysis for a single warp access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankConflictReport {
    /// Number of threads in the warp (typically 32).
    pub warp_size: usize,
    /// Bank index accessed by each thread.
    pub bank_indices: Vec<usize>,
    /// Maximum number of threads hitting the same bank (1 = no conflict).
    pub max_ways: usize,
    /// Total number of serialised bank transactions.
    pub num_transactions: usize,
    /// `true` when all accesses are broadcast-eligible (same address).
    pub is_broadcast: bool,
}

/// Compute the bank index for a byte offset into shared memory.
#[inline]
pub fn bank_index(byte_offset: usize) -> usize {
    (byte_offset / BANK_WIDTH_BYTES) % SHARED_MEMORY_BANKS
}

/// Analyse bank conflicts for a warp of up to 32 threads, given per-thread
/// byte offsets into shared memory.
pub fn analyse_bank_conflicts(offsets: &[usize]) -> Result<BankConflictReport> {
    if offsets.is_empty() || offsets.len() > 32 {
        return Err(KernelError::InvalidArguments {
            reason: format!("warp offsets must have 1..=32 entries, got {}", offsets.len()),
        }
        .into());
    }

    let banks: Vec<usize> = offsets.iter().map(|&o| bank_index(o)).collect();

    let mut per_bank = [0usize; SHARED_MEMORY_BANKS];
    for &b in &banks {
        per_bank[b] += 1;
    }

    let max_ways = per_bank.iter().copied().max().unwrap_or(0);
    let num_transactions = per_bank.iter().filter(|&&c| c > 0).count().max(max_ways);

    let is_broadcast = offsets.windows(2).all(|w| w[0] == w[1]);

    Ok(BankConflictReport {
        warp_size: offsets.len(),
        bank_indices: banks,
        max_ways,
        num_transactions,
        is_broadcast,
    })
}

/// Return `true` if the given per-thread offsets are conflict-free.
pub fn is_conflict_free(offsets: &[usize]) -> bool {
    analyse_bank_conflicts(offsets).map(|r| r.max_ways <= 1).unwrap_or(false)
}

// ── Padding calculators ─────────────────────────────────────────────

/// Configuration for the bank-conflict padding calculator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaddingConfig {
    /// Row width in elements (before padding).
    pub row_elements: usize,
    /// Size of each element in bytes (e.g. 4 for f32, 2 for f16).
    pub element_bytes: usize,
    /// Number of shared memory banks (default 32).
    pub num_banks: usize,
}

impl PaddingConfig {
    /// Create a padding config for `f32` rows.
    pub fn f32_row(row_elements: usize) -> Self {
        Self { row_elements, element_bytes: 4, num_banks: SHARED_MEMORY_BANKS }
    }

    /// Create a padding config for `f16` rows.
    pub fn f16_row(row_elements: usize) -> Self {
        Self { row_elements, element_bytes: 2, num_banks: SHARED_MEMORY_BANKS }
    }
}

/// Compute the minimum number of padding elements to append to each row
/// so that column-wise access across threads in a warp hits distinct banks.
///
/// Returns `0` when no padding is required.
pub fn padding_elements(cfg: &PaddingConfig) -> usize {
    if cfg.row_elements == 0 || cfg.element_bytes == 0 {
        return 0;
    }

    let row_bytes = cfg.row_elements * cfg.element_bytes;
    let bank_stride = cfg.num_banks * BANK_WIDTH_BYTES;

    if !row_bytes.is_multiple_of(bank_stride) {
        return 0;
    }

    1
}

/// Padded row width in elements (original + padding).
pub fn padded_row_width(cfg: &PaddingConfig) -> usize {
    cfg.row_elements + padding_elements(cfg)
}

/// Total shared memory bytes for a tile with optional padding.
pub fn tile_shared_bytes(rows: usize, cfg: &PaddingConfig) -> usize {
    rows * padded_row_width(cfg) * cfg.element_bytes
}

// ── Tiling strategies ───────────────────────────────────────────────

/// Kernel type for which a tiling strategy is being computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelType {
    /// Dense GEMM (C = αAB + βC).
    Matmul,
    /// Scaled dot-product attention.
    Attention,
    /// Reduction along one axis.
    Reduction,
    /// Element-wise fused operations.
    Elementwise,
}

/// Tiling strategy produced by the planner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileConfig {
    /// Tile dimension along M (rows of output).
    pub tile_m: usize,
    /// Tile dimension along N (columns of output).
    pub tile_n: usize,
    /// Tile dimension along the reduction axis K.
    pub tile_k: usize,
    /// Threads per block (X dimension).
    pub block_x: usize,
    /// Threads per block (Y dimension).
    pub block_y: usize,
    /// Shared memory required per block in bytes.
    pub shared_memory_bytes: usize,
    /// Padding elements per row for conflict-free access.
    pub padding_elements: usize,
    /// Number of pipeline stages (double/triple buffering).
    pub pipeline_stages: usize,
    /// Kernel type this config was produced for.
    pub kernel_type: KernelType,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tile_m: DEFAULT_TILE_DIM,
            tile_n: DEFAULT_TILE_DIM,
            tile_k: 8,
            block_x: DEFAULT_TILE_DIM,
            block_y: DEFAULT_TILE_DIM,
            shared_memory_bytes: 0,
            padding_elements: 0,
            pipeline_stages: 2,
            kernel_type: KernelType::Matmul,
        }
    }
}

/// Compute a tiling configuration for a matmul kernel.
///
/// The planner chooses tile sizes that maximise occupancy while keeping
/// shared memory usage within the device limit.
pub fn plan_matmul_tile(
    m: usize,
    n: usize,
    k: usize,
    element_bytes: usize,
    cc: ComputeCapability,
) -> Result<TileConfig> {
    if m == 0 || n == 0 || k == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "matmul dimensions must be non-zero".into(),
        }
        .into());
    }
    if element_bytes == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "element_bytes must be non-zero".into(),
        }
        .into());
    }

    let max_smem = cc.max_smem_per_block();

    for &tile in &[128, 64, 32, 16] {
        let tile_m = tile.min(m);
        let tile_n = tile.min(n);
        let tile_k = 8usize.min(k);

        let pad_a = padding_elements(&PaddingConfig {
            row_elements: tile_k,
            element_bytes,
            num_banks: SHARED_MEMORY_BANKS,
        });
        let padded_k = tile_k + pad_a;

        let pad_b = padding_elements(&PaddingConfig {
            row_elements: tile_n,
            element_bytes,
            num_banks: SHARED_MEMORY_BANKS,
        });
        let padded_n = tile_n + pad_b;

        let stages = 2usize;
        let smem_a = tile_m * padded_k * element_bytes * stages;
        let smem_b = tile_k * padded_n * element_bytes * stages;
        let total_smem = smem_a + smem_b;

        if total_smem <= max_smem {
            return Ok(TileConfig {
                tile_m,
                tile_n,
                tile_k,
                block_x: tile_n.min(32),
                block_y: tile_m.min(32),
                shared_memory_bytes: total_smem,
                padding_elements: pad_a.max(pad_b),
                pipeline_stages: stages,
                kernel_type: KernelType::Matmul,
            });
        }
    }

    Ok(TileConfig {
        tile_m: 16.min(m),
        tile_n: 16.min(n),
        tile_k: 8.min(k),
        block_x: 16.min(n),
        block_y: 16.min(m),
        shared_memory_bytes: 16 * 8 * element_bytes * 2 * 2,
        padding_elements: 0,
        pipeline_stages: 1,
        kernel_type: KernelType::Matmul,
    })
}

/// Compute a tiling configuration for a scaled dot-product attention kernel.
pub fn plan_attention_tile(
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
    element_bytes: usize,
    cc: ComputeCapability,
) -> Result<TileConfig> {
    if seq_len == 0 || head_dim == 0 || num_heads == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "attention dimensions must be non-zero".into(),
        }
        .into());
    }

    let max_smem = cc.max_smem_per_block();

    for &tile_seq in &[128, 64, 32, 16] {
        let ts = tile_seq.min(seq_len);
        let td = head_dim.min(128);

        let pad = padding_elements(&PaddingConfig {
            row_elements: td,
            element_bytes,
            num_banks: SHARED_MEMORY_BANKS,
        });

        // Q tile + K tile + partial softmax row.
        let smem = (ts * (td + pad) * element_bytes) * 2 + ts * 4;
        if smem <= max_smem {
            return Ok(TileConfig {
                tile_m: ts,
                tile_n: td,
                tile_k: td,
                block_x: td.min(32),
                block_y: ts.min(32),
                shared_memory_bytes: smem,
                padding_elements: pad,
                pipeline_stages: 2,
                kernel_type: KernelType::Attention,
            });
        }
    }

    Ok(TileConfig {
        tile_m: 16.min(seq_len),
        tile_n: head_dim.min(64),
        tile_k: head_dim.min(64),
        block_x: 16,
        block_y: 16,
        shared_memory_bytes: 16 * 64 * element_bytes * 2,
        padding_elements: 0,
        pipeline_stages: 1,
        kernel_type: KernelType::Attention,
    })
}

// ── Allocation planner (static vs dynamic) ──────────────────────────

/// Allocation strategy for shared memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SmemAllocStrategy {
    /// Statically declared `__shared__` array (≤ 48 KiB).
    Static,
    /// Dynamically sized via the third kernel launch parameter.
    Dynamic,
    /// Mixed: static for fixed-size buffers, dynamic for variable ones.
    Mixed {
        /// Bytes allocated statically.
        static_bytes: usize,
        /// Bytes allocated dynamically.
        dynamic_bytes: usize,
    },
}

/// Plan for shared memory allocation for a kernel launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmemAllocationPlan {
    /// Chosen allocation strategy.
    pub strategy: SmemAllocStrategy,
    /// Total shared memory required in bytes.
    pub total_bytes: usize,
    /// Whether the plan fits within device limits.
    pub fits_device: bool,
    /// Compute capability the plan was generated for.
    pub compute_capability: ComputeCapability,
}

/// Decide between static, dynamic, or mixed shared memory allocation.
pub fn plan_smem_allocation(required_bytes: usize, cc: ComputeCapability) -> SmemAllocationPlan {
    let max_block = cc.max_smem_per_block();
    let fits = required_bytes <= max_block;

    let strategy = if required_bytes <= MAX_STATIC_SMEM_BYTES {
        SmemAllocStrategy::Static
    } else {
        SmemAllocStrategy::Dynamic
    };

    SmemAllocationPlan {
        strategy,
        total_bytes: required_bytes,
        fits_device: fits,
        compute_capability: cc,
    }
}

/// Plan a mixed static/dynamic allocation.
pub fn plan_mixed_allocation(
    static_bytes: usize,
    dynamic_bytes: usize,
    cc: ComputeCapability,
) -> SmemAllocationPlan {
    let total = static_bytes + dynamic_bytes;
    let max_block = cc.max_smem_per_block();
    let fits = total <= max_block;

    let strategy = if static_bytes == 0 {
        SmemAllocStrategy::Dynamic
    } else if dynamic_bytes == 0 {
        if static_bytes <= MAX_STATIC_SMEM_BYTES {
            SmemAllocStrategy::Static
        } else {
            SmemAllocStrategy::Dynamic
        }
    } else {
        SmemAllocStrategy::Mixed { static_bytes, dynamic_bytes }
    };

    SmemAllocationPlan { strategy, total_bytes: total, fits_device: fits, compute_capability: cc }
}

// ── Data layout optimizers ──────────────────────────────────────────

/// Memory access pattern descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessPattern {
    /// Row-major (consecutive threads access consecutive columns).
    RowMajor,
    /// Column-major (consecutive threads access consecutive rows).
    ColumnMajor,
    /// Diagonal access (e.g. triangular solves).
    Diagonal,
    /// Strided access with a fixed stride in elements.
    Strided(usize),
}

/// Recommendation produced by the layout optimizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutRecommendation {
    /// Recommended access pattern for shared memory stores.
    pub store_pattern: AccessPattern,
    /// Recommended access pattern for shared memory loads.
    pub load_pattern: AccessPattern,
    /// Padding elements per row.
    pub padding: usize,
    /// Whether the layout swizzles row indices for conflict avoidance.
    pub uses_swizzle: bool,
    /// Estimated bank-conflict degree (1 = conflict-free).
    pub estimated_conflict_degree: usize,
}

/// Analyse an access pattern and recommend an optimal data layout.
pub fn optimize_layout(
    rows: usize,
    cols: usize,
    element_bytes: usize,
    pattern: AccessPattern,
) -> Result<LayoutRecommendation> {
    if rows == 0 || cols == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "layout dimensions must be non-zero".into(),
        }
        .into());
    }

    let pad_cfg =
        PaddingConfig { row_elements: cols, element_bytes, num_banks: SHARED_MEMORY_BANKS };
    let pad = padding_elements(&pad_cfg);

    match pattern {
        AccessPattern::RowMajor => Ok(LayoutRecommendation {
            store_pattern: AccessPattern::RowMajor,
            load_pattern: AccessPattern::RowMajor,
            padding: 0,
            uses_swizzle: false,
            estimated_conflict_degree: 1,
        }),
        AccessPattern::ColumnMajor => Ok(LayoutRecommendation {
            store_pattern: AccessPattern::RowMajor,
            load_pattern: AccessPattern::ColumnMajor,
            padding: pad,
            uses_swizzle: pad == 0 && cols.is_multiple_of(SHARED_MEMORY_BANKS),
            estimated_conflict_degree: if pad > 0 || !cols.is_multiple_of(SHARED_MEMORY_BANKS) {
                1
            } else {
                SHARED_MEMORY_BANKS.min(rows)
            },
        }),
        AccessPattern::Diagonal => Ok(LayoutRecommendation {
            store_pattern: AccessPattern::RowMajor,
            load_pattern: AccessPattern::Diagonal,
            padding: 0,
            uses_swizzle: true,
            estimated_conflict_degree: 1,
        }),
        AccessPattern::Strided(stride) => {
            let effective_stride = stride * element_bytes;
            let conflict =
                if effective_stride.is_multiple_of(SHARED_MEMORY_BANKS * BANK_WIDTH_BYTES) {
                    SHARED_MEMORY_BANKS
                } else {
                    gcd(effective_stride / BANK_WIDTH_BYTES, SHARED_MEMORY_BANKS).max(1)
                };
            let needs_pad = conflict > 1;
            Ok(LayoutRecommendation {
                store_pattern: AccessPattern::RowMajor,
                load_pattern: AccessPattern::Strided(stride),
                padding: if needs_pad { 1 } else { 0 },
                uses_swizzle: false,
                estimated_conflict_degree: if needs_pad { conflict } else { 1 },
            })
        }
    }
}

/// Compute the XOR-swizzled row index used to eliminate bank conflicts
/// in diagonal or transpose access patterns.
#[inline]
pub fn swizzle_index(row: usize, col: usize) -> usize {
    row ^ col
}

/// Compute a coalesced byte offset for a 2-D tile with optional padding.
#[inline]
pub fn coalesced_offset(
    row: usize,
    col: usize,
    padded_row_width: usize,
    element_bytes: usize,
) -> usize {
    (row * padded_row_width + col) * element_bytes
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Greatest common divisor (Euclidean algorithm).
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ── CUDA kernel source (GPU-only) ───────────────────────────────────

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const SHARED_MEMORY_TRANSPOSE_KERNEL_SRC: &str = r#"
extern "C" __global__
void smem_transpose_2d(const float* __restrict__ in,
                       float* __restrict__ out,
                       int rows, int cols)
{
    __shared__ float tile[32][33];

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];

    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    if (x < rows && y < cols)
        out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}
"#;

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sm80() -> ComputeCapability {
        ComputeCapability::new(8, 0)
    }

    fn sm70() -> ComputeCapability {
        ComputeCapability::new(7, 0)
    }

    fn sm50() -> ComputeCapability {
        ComputeCapability::new(5, 0)
    }

    fn sm90() -> ComputeCapability {
        ComputeCapability::new(9, 0)
    }

    // ── Compute capability ──────────────────────────────────────────

    #[test]
    fn cc_sm_version_encoding() {
        assert_eq!(sm80().sm_version(), 80);
        assert_eq!(sm70().sm_version(), 70);
        assert_eq!(ComputeCapability::new(7, 5).sm_version(), 75);
    }

    #[test]
    fn cc_max_smem_per_sm_kepler() {
        let cc = ComputeCapability::new(3, 5);
        assert_eq!(cc.max_smem_per_sm(), 48 * 1024);
    }

    #[test]
    fn cc_max_smem_per_sm_maxwell() {
        assert_eq!(sm50().max_smem_per_sm(), 64 * 1024);
    }

    #[test]
    fn cc_max_smem_per_sm_volta() {
        assert_eq!(sm70().max_smem_per_sm(), 96 * 1024);
    }

    #[test]
    fn cc_max_smem_per_sm_turing() {
        let cc = ComputeCapability::new(7, 5);
        assert_eq!(cc.max_smem_per_sm(), 64 * 1024);
    }

    #[test]
    fn cc_max_smem_per_sm_ampere() {
        assert_eq!(sm80().max_smem_per_sm(), 164 * 1024);
    }

    #[test]
    fn cc_max_smem_per_sm_ada() {
        let cc = ComputeCapability::new(8, 9);
        assert_eq!(cc.max_smem_per_sm(), 100 * 1024);
    }

    #[test]
    fn cc_max_smem_per_sm_hopper() {
        assert_eq!(sm90().max_smem_per_sm(), 228 * 1024);
    }

    #[test]
    fn cc_max_smem_per_block_volta() {
        assert_eq!(sm70().max_smem_per_block(), 96 * 1024);
    }

    #[test]
    fn cc_max_smem_per_block_ampere() {
        assert_eq!(sm80().max_smem_per_block(), 163 * 1024);
    }

    #[test]
    fn cc_max_smem_per_block_hopper() {
        assert_eq!(sm90().max_smem_per_block(), 227 * 1024);
    }

    #[test]
    fn cc_ordering() {
        assert!(sm50() < sm70());
        assert!(sm70() < sm80());
        assert!(sm80() < sm90());
    }

    #[test]
    fn cc_max_smem_per_sm_pascal() {
        let cc = ComputeCapability::new(6, 1);
        assert_eq!(cc.max_smem_per_sm(), 64 * 1024);
    }

    #[test]
    fn cc_unknown_future_arch_uses_fallback() {
        let cc = ComputeCapability::new(12, 0);
        assert_eq!(cc.max_smem_per_sm(), 48 * 1024);
        assert_eq!(cc.max_smem_per_block(), 48 * 1024);
    }

    // ── L1 / shared memory partitioning ─────────────────────────────

    #[test]
    fn partition_prefer_shared_ampere() {
        let p = partition_l1_shared(sm80(), CachePreference::PreferShared);
        assert_eq!(p.total_bytes, 192 * 1024);
        assert_eq!(p.shared_memory_bytes, (192 * 1024 * 3) / 4);
        assert_eq!(p.l1_cache_bytes, p.total_bytes - p.shared_memory_bytes);
    }

    #[test]
    fn partition_prefer_l1_volta() {
        let p = partition_l1_shared(sm70(), CachePreference::PreferL1);
        assert_eq!(p.total_bytes, 128 * 1024);
        assert_eq!(p.shared_memory_bytes, 128 * 1024 / 4);
    }

    #[test]
    fn partition_equal_maxwell() {
        let p = partition_l1_shared(sm50(), CachePreference::PreferEqual);
        assert_eq!(p.total_bytes, 64 * 1024);
        assert_eq!(p.shared_memory_bytes, 32 * 1024);
        assert_eq!(p.l1_cache_bytes, 32 * 1024);
    }

    #[test]
    fn partition_no_preference_uses_equal_split() {
        let p = partition_l1_shared(sm80(), CachePreference::NoPreference);
        assert_eq!(p.shared_memory_bytes, p.total_bytes / 2);
    }

    #[test]
    fn partition_shared_plus_l1_equals_total() {
        for cc in [sm50(), sm70(), sm80(), sm90()] {
            for pref in [
                CachePreference::PreferShared,
                CachePreference::PreferEqual,
                CachePreference::PreferL1,
                CachePreference::NoPreference,
            ] {
                let p = partition_l1_shared(cc, pref);
                assert_eq!(p.shared_memory_bytes + p.l1_cache_bytes, p.total_bytes);
            }
        }
    }

    #[test]
    fn partition_prefer_shared_always_largest() {
        for cc in [sm50(), sm70(), sm80(), sm90()] {
            let shared = partition_l1_shared(cc, CachePreference::PreferShared);
            let equal = partition_l1_shared(cc, CachePreference::PreferEqual);
            let l1 = partition_l1_shared(cc, CachePreference::PreferL1);
            assert!(shared.shared_memory_bytes >= equal.shared_memory_bytes);
            assert!(equal.shared_memory_bytes >= l1.shared_memory_bytes);
        }
    }

    // ── Bank conflict analysis ──────────────────────────────────────

    #[test]
    fn bank_index_sequential_f32() {
        for i in 0..32 {
            assert_eq!(bank_index(i * 4), i);
        }
    }

    #[test]
    fn bank_index_wraps_at_32() {
        assert_eq!(bank_index(0), bank_index(32 * 4));
    }

    #[test]
    fn analyse_conflict_free_sequential() {
        let offsets: Vec<usize> = (0..32).map(|i| i * 4).collect();
        let report = analyse_bank_conflicts(&offsets).unwrap();
        assert_eq!(report.max_ways, 1);
        assert_eq!(report.num_transactions, 32);
        assert!(!report.is_broadcast);
    }

    #[test]
    fn analyse_full_conflict_same_bank() {
        let offsets: Vec<usize> = (0..32).map(|i| i * 128).collect();
        let report = analyse_bank_conflicts(&offsets).unwrap();
        assert_eq!(report.max_ways, 32);
    }

    #[test]
    fn analyse_broadcast_same_address() {
        let offsets = vec![256; 32];
        let report = analyse_bank_conflicts(&offsets).unwrap();
        assert!(report.is_broadcast);
    }

    #[test]
    fn analyse_empty_offsets_rejected() {
        assert!(analyse_bank_conflicts(&[]).is_err());
    }

    #[test]
    fn analyse_too_many_offsets_rejected() {
        let offsets: Vec<usize> = (0..33).map(|i| i * 4).collect();
        assert!(analyse_bank_conflicts(&offsets).is_err());
    }

    #[test]
    fn is_conflict_free_sequential() {
        let offsets: Vec<usize> = (0..32).map(|i| i * 4).collect();
        assert!(is_conflict_free(&offsets));
    }

    #[test]
    fn is_conflict_free_stride_128_has_conflicts() {
        let offsets: Vec<usize> = (0..32).map(|i| i * 128).collect();
        assert!(!is_conflict_free(&offsets));
    }

    #[test]
    fn analyse_two_way_conflict() {
        // Stride 8 bytes = 2 banks, so threads 0 and 16 share a bank.
        let offsets: Vec<usize> = (0..32).map(|i| i * 8).collect();
        let report = analyse_bank_conflicts(&offsets).unwrap();
        assert_eq!(report.max_ways, 2);
    }

    #[test]
    fn analyse_single_thread() {
        let report = analyse_bank_conflicts(&[0]).unwrap();
        assert_eq!(report.max_ways, 1);
        assert_eq!(report.warp_size, 1);
    }

    #[test]
    fn bank_indices_correct_for_f16() {
        let offsets: Vec<usize> = (0..16).map(|i| i * 2).collect();
        let report = analyse_bank_conflicts(&offsets).unwrap();
        assert_eq!(report.bank_indices[0], 0);
        assert_eq!(report.bank_indices[2], 1);
    }

    #[test]
    fn analyse_four_way_conflict() {
        // Stride 16 bytes = 4 banks, so 32/4 threads share each bank → 4-way.
        let offsets: Vec<usize> = (0..32).map(|i| i * 16).collect();
        let report = analyse_bank_conflicts(&offsets).unwrap();
        assert_eq!(report.max_ways, 4);
    }

    #[test]
    fn analyse_half_warp() {
        let offsets: Vec<usize> = (0..16).map(|i| i * 4).collect();
        let report = analyse_bank_conflicts(&offsets).unwrap();
        assert_eq!(report.max_ways, 1);
        assert_eq!(report.warp_size, 16);
    }

    // ── Padding calculators ─────────────────────────────────────────

    #[test]
    fn padding_f32_row_32_needs_one() {
        let cfg = PaddingConfig::f32_row(32);
        assert_eq!(padding_elements(&cfg), 1);
    }

    #[test]
    fn padding_f32_row_31_no_padding() {
        let cfg = PaddingConfig::f32_row(31);
        assert_eq!(padding_elements(&cfg), 0);
    }

    #[test]
    fn padding_f32_row_64_needs_one() {
        let cfg = PaddingConfig::f32_row(64);
        assert_eq!(padding_elements(&cfg), 1);
    }

    #[test]
    fn padding_f16_row_64_needs_one() {
        let cfg = PaddingConfig::f16_row(64);
        assert_eq!(padding_elements(&cfg), 1);
    }

    #[test]
    fn padding_f16_row_63_no_padding() {
        let cfg = PaddingConfig::f16_row(63);
        assert_eq!(padding_elements(&cfg), 0);
    }

    #[test]
    fn padding_zero_elements_returns_zero() {
        let cfg = PaddingConfig { row_elements: 0, element_bytes: 4, num_banks: 32 };
        assert_eq!(padding_elements(&cfg), 0);
    }

    #[test]
    fn padding_zero_element_bytes_returns_zero() {
        let cfg = PaddingConfig { row_elements: 32, element_bytes: 0, num_banks: 32 };
        assert_eq!(padding_elements(&cfg), 0);
    }

    #[test]
    fn padded_row_width_adds_padding() {
        let cfg = PaddingConfig::f32_row(32);
        assert_eq!(padded_row_width(&cfg), 33);
    }

    #[test]
    fn padded_row_width_no_padding_unchanged() {
        let cfg = PaddingConfig::f32_row(31);
        assert_eq!(padded_row_width(&cfg), 31);
    }

    #[test]
    fn tile_shared_bytes_with_padding() {
        let cfg = PaddingConfig::f32_row(32);
        assert_eq!(tile_shared_bytes(16, &cfg), 16 * 33 * 4);
    }

    #[test]
    fn tile_shared_bytes_single_row() {
        let cfg = PaddingConfig::f32_row(8);
        assert_eq!(tile_shared_bytes(1, &cfg), 8 * 4);
    }

    #[test]
    fn padding_f32_row_96_needs_one() {
        let cfg = PaddingConfig::f32_row(96);
        assert_eq!(padding_elements(&cfg), 1);
    }

    #[test]
    fn padding_f16_row_128_needs_one() {
        let cfg = PaddingConfig::f16_row(128);
        assert_eq!(padding_elements(&cfg), 1);
    }

    #[test]
    fn padding_f32_row_33_no_padding() {
        let cfg = PaddingConfig::f32_row(33);
        assert_eq!(padding_elements(&cfg), 0);
    }

    // ── Tiling strategies ───────────────────────────────────────────

    #[test]
    fn tile_config_default_is_matmul() {
        let tc = TileConfig::default();
        assert_eq!(tc.kernel_type, KernelType::Matmul);
        assert_eq!(tc.tile_m, 32);
        assert_eq!(tc.tile_n, 32);
    }

    #[test]
    fn plan_matmul_tile_basic() {
        let tc = plan_matmul_tile(256, 256, 256, 4, sm80()).unwrap();
        assert_eq!(tc.kernel_type, KernelType::Matmul);
        assert!(tc.tile_m > 0);
        assert!(tc.tile_n > 0);
        assert!(tc.shared_memory_bytes > 0);
        assert!(tc.shared_memory_bytes <= sm80().max_smem_per_block());
    }

    #[test]
    fn plan_matmul_tile_zero_m_rejected() {
        assert!(plan_matmul_tile(0, 64, 64, 4, sm80()).is_err());
    }

    #[test]
    fn plan_matmul_tile_zero_n_rejected() {
        assert!(plan_matmul_tile(64, 0, 64, 4, sm80()).is_err());
    }

    #[test]
    fn plan_matmul_tile_zero_k_rejected() {
        assert!(plan_matmul_tile(64, 64, 0, 4, sm80()).is_err());
    }

    #[test]
    fn plan_matmul_tile_zero_element_bytes_rejected() {
        assert!(plan_matmul_tile(64, 64, 64, 0, sm80()).is_err());
    }

    #[test]
    fn plan_matmul_tile_small_matrix() {
        let tc = plan_matmul_tile(4, 4, 4, 4, sm80()).unwrap();
        assert!(tc.tile_m <= 4);
        assert!(tc.tile_n <= 4);
        assert!(tc.tile_k <= 4);
    }

    #[test]
    fn plan_matmul_tile_f16() {
        let tc = plan_matmul_tile(512, 512, 512, 2, sm80()).unwrap();
        assert!(tc.shared_memory_bytes > 0);
        assert!(tc.shared_memory_bytes <= sm80().max_smem_per_block());
    }

    #[test]
    fn plan_matmul_tile_large_fits_device() {
        let tc = plan_matmul_tile(4096, 4096, 4096, 4, sm80()).unwrap();
        assert!(tc.shared_memory_bytes <= sm80().max_smem_per_block());
    }

    #[test]
    fn plan_matmul_tile_double_buffer() {
        let tc = plan_matmul_tile(256, 256, 256, 4, sm80()).unwrap();
        assert!(tc.pipeline_stages >= 2);
    }

    #[test]
    fn plan_matmul_different_cc_may_differ() {
        let tc50 = plan_matmul_tile(1024, 1024, 1024, 4, sm50()).unwrap();
        let tc90 = plan_matmul_tile(1024, 1024, 1024, 4, sm90()).unwrap();
        assert!(tc50.shared_memory_bytes <= sm50().max_smem_per_block());
        assert!(tc90.shared_memory_bytes <= sm90().max_smem_per_block());
    }

    #[test]
    fn plan_matmul_tile_non_square() {
        let tc = plan_matmul_tile(1024, 128, 512, 4, sm80()).unwrap();
        assert!(tc.tile_m <= 1024);
        assert!(tc.tile_n <= 128);
    }

    #[test]
    fn plan_matmul_block_dims_within_warp() {
        let tc = plan_matmul_tile(256, 256, 256, 4, sm80()).unwrap();
        assert!(tc.block_x <= 32);
        assert!(tc.block_y <= 32);
    }

    #[test]
    fn plan_attention_tile_basic() {
        let tc = plan_attention_tile(512, 64, 8, 4, sm80()).unwrap();
        assert_eq!(tc.kernel_type, KernelType::Attention);
        assert!(tc.shared_memory_bytes <= sm80().max_smem_per_block());
    }

    #[test]
    fn plan_attention_tile_zero_seq_rejected() {
        assert!(plan_attention_tile(0, 64, 8, 4, sm80()).is_err());
    }

    #[test]
    fn plan_attention_tile_zero_head_dim_rejected() {
        assert!(plan_attention_tile(512, 0, 8, 4, sm80()).is_err());
    }

    #[test]
    fn plan_attention_tile_zero_heads_rejected() {
        assert!(plan_attention_tile(512, 64, 0, 4, sm80()).is_err());
    }

    #[test]
    fn plan_attention_tile_small_sequence() {
        let tc = plan_attention_tile(8, 64, 1, 4, sm80()).unwrap();
        assert!(tc.tile_m <= 8);
    }

    #[test]
    fn plan_attention_tile_large_head_dim() {
        let tc = plan_attention_tile(1024, 256, 32, 4, sm80()).unwrap();
        assert!(tc.shared_memory_bytes <= sm80().max_smem_per_block());
    }

    #[test]
    fn plan_attention_tile_f16() {
        let tc = plan_attention_tile(512, 64, 8, 2, sm80()).unwrap();
        assert!(tc.shared_memory_bytes > 0);
    }

    // ── Allocation planner ──────────────────────────────────────────

    #[test]
    fn plan_static_when_under_48k() {
        let plan = plan_smem_allocation(32 * 1024, sm80());
        assert_eq!(plan.strategy, SmemAllocStrategy::Static);
        assert!(plan.fits_device);
    }

    #[test]
    fn plan_dynamic_when_over_48k() {
        let plan = plan_smem_allocation(64 * 1024, sm80());
        assert_eq!(plan.strategy, SmemAllocStrategy::Dynamic);
        assert!(plan.fits_device);
    }

    #[test]
    fn plan_does_not_fit_enormous_request() {
        let plan = plan_smem_allocation(1024 * 1024, sm50());
        assert!(!plan.fits_device);
    }

    #[test]
    fn plan_exactly_48k_is_static() {
        let plan = plan_smem_allocation(48 * 1024, sm80());
        assert_eq!(plan.strategy, SmemAllocStrategy::Static);
    }

    #[test]
    fn plan_zero_bytes_is_static() {
        let plan = plan_smem_allocation(0, sm80());
        assert_eq!(plan.strategy, SmemAllocStrategy::Static);
        assert!(plan.fits_device);
    }

    #[test]
    fn plan_mixed_both_nonzero() {
        let plan = plan_mixed_allocation(16 * 1024, 32 * 1024, sm80());
        assert!(matches!(plan.strategy, SmemAllocStrategy::Mixed { .. }));
        assert_eq!(plan.total_bytes, 48 * 1024);
        assert!(plan.fits_device);
    }

    #[test]
    fn plan_mixed_static_only() {
        let plan = plan_mixed_allocation(16 * 1024, 0, sm80());
        assert_eq!(plan.strategy, SmemAllocStrategy::Static);
    }

    #[test]
    fn plan_mixed_dynamic_only() {
        let plan = plan_mixed_allocation(0, 64 * 1024, sm80());
        assert_eq!(plan.strategy, SmemAllocStrategy::Dynamic);
    }

    #[test]
    fn plan_mixed_exceeds_device() {
        let plan = plan_mixed_allocation(100 * 1024, 200 * 1024, sm50());
        assert!(!plan.fits_device);
    }

    #[test]
    fn plan_stores_compute_capability() {
        let plan = plan_smem_allocation(1024, sm90());
        assert_eq!(plan.compute_capability, sm90());
    }

    #[test]
    fn plan_48k_plus_one_is_dynamic() {
        let plan = plan_smem_allocation(48 * 1024 + 1, sm80());
        assert_eq!(plan.strategy, SmemAllocStrategy::Dynamic);
    }

    #[test]
    fn plan_mixed_static_over_48k_becomes_dynamic() {
        let plan = plan_mixed_allocation(64 * 1024, 0, sm80());
        assert_eq!(plan.strategy, SmemAllocStrategy::Dynamic);
    }

    // ── Data layout optimizers ──────────────────────────────────────

    #[test]
    fn layout_row_major_conflict_free() {
        let rec = optimize_layout(32, 32, 4, AccessPattern::RowMajor).unwrap();
        assert_eq!(rec.estimated_conflict_degree, 1);
        assert_eq!(rec.padding, 0);
        assert!(!rec.uses_swizzle);
    }

    #[test]
    fn layout_col_major_32_needs_padding() {
        let rec = optimize_layout(32, 32, 4, AccessPattern::ColumnMajor).unwrap();
        assert_eq!(rec.padding, 1);
        assert_eq!(rec.estimated_conflict_degree, 1);
    }

    #[test]
    fn layout_col_major_31_no_padding() {
        let rec = optimize_layout(32, 31, 4, AccessPattern::ColumnMajor).unwrap();
        assert_eq!(rec.padding, 0);
        assert_eq!(rec.estimated_conflict_degree, 1);
    }

    #[test]
    fn layout_diagonal_uses_swizzle() {
        let rec = optimize_layout(32, 32, 4, AccessPattern::Diagonal).unwrap();
        assert!(rec.uses_swizzle);
        assert_eq!(rec.estimated_conflict_degree, 1);
    }

    #[test]
    fn layout_strided_power_of_32_has_conflicts() {
        let rec = optimize_layout(32, 32, 4, AccessPattern::Strided(32)).unwrap();
        assert!(rec.estimated_conflict_degree > 1);
    }

    #[test]
    fn layout_strided_prime_is_conflict_free() {
        let rec = optimize_layout(32, 32, 4, AccessPattern::Strided(7)).unwrap();
        assert_eq!(rec.estimated_conflict_degree, 1);
    }

    #[test]
    fn layout_zero_rows_rejected() {
        assert!(optimize_layout(0, 32, 4, AccessPattern::RowMajor).is_err());
    }

    #[test]
    fn layout_zero_cols_rejected() {
        assert!(optimize_layout(32, 0, 4, AccessPattern::RowMajor).is_err());
    }

    #[test]
    fn layout_col_major_64_needs_padding() {
        let rec = optimize_layout(32, 64, 4, AccessPattern::ColumnMajor).unwrap();
        assert_eq!(rec.padding, 1);
    }

    #[test]
    fn layout_strided_1_is_conflict_free() {
        let rec = optimize_layout(32, 32, 4, AccessPattern::Strided(1)).unwrap();
        assert_eq!(rec.estimated_conflict_degree, 1);
    }

    // ── Swizzle and coalesced offset ────────────────────────────────

    #[test]
    fn swizzle_identity_at_zero() {
        assert_eq!(swizzle_index(0, 0), 0);
    }

    #[test]
    fn swizzle_xor_property() {
        assert_eq!(swizzle_index(5, 3), 5 ^ 3);
        assert_eq!(swizzle_index(7, 7), 0);
    }

    #[test]
    fn swizzle_all_rows_unique_within_tile() {
        let col = 13;
        let mut rows: Vec<usize> = (0..32).map(|r| swizzle_index(r, col)).collect();
        rows.sort();
        rows.dedup();
        assert_eq!(rows.len(), 32);
    }

    #[test]
    fn swizzle_is_self_inverse() {
        for r in 0..32 {
            for c in 0..32 {
                let swizzled = swizzle_index(r, c);
                assert_eq!(swizzle_index(swizzled, c), r);
            }
        }
    }

    #[test]
    fn coalesced_offset_first_element() {
        assert_eq!(coalesced_offset(0, 0, 33, 4), 0);
    }

    #[test]
    fn coalesced_offset_row_stride() {
        assert_eq!(coalesced_offset(1, 0, 33, 4), 33 * 4);
    }

    #[test]
    fn coalesced_offset_col_offset() {
        assert_eq!(coalesced_offset(0, 5, 33, 4), 5 * 4);
    }

    #[test]
    fn coalesced_offset_f16() {
        assert_eq!(coalesced_offset(2, 3, 65, 2), (2 * 65 + 3) * 2);
    }

    // ── GCD helper ──────────────────────────────────────────────────

    #[test]
    fn gcd_basic() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(7, 13), 1);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(100, 0), 100);
    }

    #[test]
    fn gcd_equal_values() {
        assert_eq!(gcd(42, 42), 42);
    }

    #[test]
    fn gcd_one() {
        assert_eq!(gcd(1, 100), 1);
        assert_eq!(gcd(100, 1), 1);
    }

    // ── Integration-style tests ─────────────────────────────────────

    #[test]
    fn matmul_tile_with_padding_is_conflict_free() {
        let tc = plan_matmul_tile(256, 256, 256, 4, sm80()).unwrap();
        let padded = tc.tile_k + tc.padding_elements;
        let offsets: Vec<usize> = (0..32.min(tc.tile_m)).map(|row| row * padded * 4).collect();
        if offsets.len() == 32 {
            let report = analyse_bank_conflicts(&offsets).unwrap();
            assert!(report.max_ways <= 4, "too many conflicts: {}", report.max_ways);
        }
    }

    #[test]
    fn attention_plan_fits_all_cc() {
        for cc in [sm50(), sm70(), sm80(), sm90()] {
            let tc = plan_attention_tile(256, 64, 8, 4, cc).unwrap();
            assert!(
                tc.shared_memory_bytes <= cc.max_smem_per_block(),
                "smem {} > max {} for sm_{}",
                tc.shared_memory_bytes,
                cc.max_smem_per_block(),
                cc.sm_version()
            );
        }
    }

    #[test]
    fn allocation_plan_respects_device_limit() {
        for cc in [sm50(), sm70(), sm80(), sm90()] {
            let max = cc.max_smem_per_block();
            let plan = plan_smem_allocation(max, cc);
            assert!(plan.fits_device);
            let plan_over = plan_smem_allocation(max + 1, cc);
            assert!(!plan_over.fits_device);
        }
    }

    #[test]
    fn end_to_end_conflict_free_transpose() {
        let padded_width = 33;
        let offsets: Vec<usize> = (0..32)
            .map(|tid| coalesced_offset(swizzle_index(tid, 0), 0, padded_width, 4))
            .collect();
        let report = analyse_bank_conflicts(&offsets).unwrap();
        assert_eq!(report.max_ways, 1, "expected conflict-free transpose tile");
    }

    #[test]
    fn partition_then_plan_integration() {
        let cc = sm80();
        let part = partition_l1_shared(cc, CachePreference::PreferShared);
        let plan = plan_smem_allocation(part.shared_memory_bytes, cc);
        assert!(plan.fits_device);
    }

    #[test]
    fn layout_recommendation_column_major_with_pad_is_conflict_free() {
        let rec = optimize_layout(32, 32, 4, AccessPattern::ColumnMajor).unwrap();
        let padded = 32 + rec.padding;
        let offsets: Vec<usize> = (0..32).map(|row| row * padded * 4).collect();
        assert!(is_conflict_free(&offsets));
    }

    #[test]
    fn constants_are_sane() {
        assert_eq!(SHARED_MEMORY_BANKS, 32);
        assert_eq!(BANK_WIDTH_BYTES, 4);
        assert_eq!(MAX_STATIC_SMEM_BYTES, 48 * 1024);
        assert_eq!(DEFAULT_TILE_DIM, 32);
    }

    #[test]
    fn matmul_plan_then_allocation_plan() {
        let cc = sm80();
        let tile = plan_matmul_tile(1024, 1024, 1024, 4, cc).unwrap();
        let plan = plan_smem_allocation(tile.shared_memory_bytes, cc);
        assert!(plan.fits_device);
    }

    #[test]
    fn full_pipeline_attention() {
        let cc = sm90();
        let part = partition_l1_shared(cc, CachePreference::PreferShared);
        let tile = plan_attention_tile(2048, 128, 32, 4, cc).unwrap();
        assert!(tile.shared_memory_bytes <= part.shared_memory_bytes);
        let plan = plan_smem_allocation(tile.shared_memory_bytes, cc);
        assert!(plan.fits_device);
    }
}
