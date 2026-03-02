//! CUDA memory bandwidth optimization: coalescing, alignment, and prefetch.
//!
//! # Overview
//!
//! Provides analysis and optimization primitives for GPU memory access patterns.
//! Coalesced, aligned, and prefetched accesses are critical for maximising global
//! memory bandwidth on NVIDIA GPUs (typically 80–90 % of peak with coalesced
//! 128-byte transactions).
//!
//! Key components:
//!
//! - [`MemoryAccessPattern`] — classifies access patterns (coalesced, strided,
//!   random, sequential).
//! - [`BandwidthConfig`] — alignment, prefetch distance, vectorisation width,
//!   and cache policy.
//! - [`MemoryAnalysis`] — bandwidth utilisation metrics and access statistics.
//! - [`CachePolicy`] — L1/L2 caching hints for load/store instructions.
//! - [`MemoryHierarchyStats`] — per-level hit/miss statistics.
//!
//! # Functions
//!
//! - [`analyze_access_pattern`] — classify an index sequence into a pattern.
//! - [`optimize_layout`] — reorder data for coalesced access.
//! - [`coalesced_copy`] — aligned bulk copy respecting warp transaction size.
//! - [`vectorized_load`] / [`vectorized_store`] — wide loads/stores (float4).
//! - [`prefetch_buffer`] — software prefetch simulation.
//! - [`align_buffer`] — pad a buffer to the requested alignment.
//! - [`transpose_for_coalescing`] — SoA ↔ AoS transpose for coalesced reads.
//! - [`bank_conflict_free_access`] — shared-memory index remapping (+1 padding).
//! - [`estimate_bandwidth`] — estimate achieved bandwidth from size and duration.
//! - [`memory_hierarchy_stats`] — simulated L1/L2/DRAM hit-rate model.
//!
//! All GPU code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU implementations model memory hierarchy behaviour for testing and analysis.

use bitnet_common::{KernelError, Result};

// ── Constants ────────────────────────────────────────────────────────

/// Typical GPU cache-line / transaction size in bytes.
pub const CACHE_LINE_BYTES: usize = 128;

/// Warp width on NVIDIA GPUs.
pub const WARP_SIZE: usize = 32;

/// Shared-memory bank count on modern NVIDIA GPUs.
pub const SHARED_MEM_BANKS: usize = 32;

/// Default alignment in bytes (matches GPU global-memory transaction).
pub const DEFAULT_ALIGNMENT: usize = 128;

/// Default vectorisation width in f32 elements (float4 = 4).
pub const DEFAULT_VECTOR_WIDTH: usize = 4;

// ── MemoryAccessPattern ──────────────────────────────────────────────

/// Classification of a memory access pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryAccessPattern {
    /// Consecutive threads access consecutive addresses — optimal.
    Coalesced,
    /// Threads access addresses separated by a fixed stride.
    Strided {
        /// Stride in elements between consecutive thread accesses.
        stride: usize,
    },
    /// No discernible regularity in the access pattern.
    Random,
    /// Single-threaded sequential scan (e.g. CPU memcpy).
    Sequential,
}

impl MemoryAccessPattern {
    /// Estimated efficiency relative to peak bandwidth (0.0–1.0).
    pub fn efficiency(&self) -> f64 {
        match self {
            Self::Coalesced => 1.0,
            Self::Sequential => 0.85,
            Self::Strided { stride } => {
                // Larger strides waste more of each cache-line transaction.
                let useful = 1.0 / (*stride).max(1) as f64;
                useful.clamp(0.05, 1.0)
            }
            Self::Random => 0.05,
        }
    }
}

// ── CachePolicy ──────────────────────────────────────────────────────

/// Cache-policy hint for load/store instructions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CachePolicy {
    /// Cache at all levels (`.ca` — default).
    #[default]
    CacheAll,
    /// Cache in L2 only, streaming through L1 (`.cg`).
    CacheGlobal,
    /// Streaming — evict early from all caches (`.cs`).
    Streaming,
    /// Last-use — hint that data will not be reused (`.lu`).
    LastUse,
}

// ── BandwidthConfig ──────────────────────────────────────────────────

/// Configuration knobs for memory bandwidth optimisation.
#[derive(Debug, Clone)]
pub struct BandwidthConfig {
    /// Required byte-alignment for buffers (must be a power of two).
    pub alignment: usize,
    /// Software prefetch distance in cache lines.
    pub prefetch_distance: usize,
    /// Vectorisation width in f32 elements (1, 2, or 4).
    pub vectorize_width: usize,
    /// Cache policy hint for loads.
    pub cache_policy: CachePolicy,
}

impl Default for BandwidthConfig {
    fn default() -> Self {
        Self {
            alignment: DEFAULT_ALIGNMENT,
            prefetch_distance: 4,
            vectorize_width: DEFAULT_VECTOR_WIDTH,
            cache_policy: CachePolicy::default(),
        }
    }
}

impl BandwidthConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<()> {
        if !self.alignment.is_power_of_two() {
            return Err(KernelError::InvalidArguments {
                reason: "alignment must be a power of two".into(),
            }
            .into());
        }
        if self.alignment == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "alignment must be non-zero".into(),
            }
            .into());
        }
        if !matches!(self.vectorize_width, 1 | 2 | 4) {
            return Err(KernelError::InvalidArguments {
                reason: "vectorize_width must be 1, 2, or 4".into(),
            }
            .into());
        }
        Ok(())
    }
}

// ── MemoryAnalysis ───────────────────────────────────────────────────

/// Bandwidth utilisation metrics for an access pattern.
#[derive(Debug, Clone)]
pub struct MemoryAnalysis {
    /// Detected access pattern.
    pub pattern: MemoryAccessPattern,
    /// Achieved bandwidth in bytes/second (0 until measured).
    pub achieved_bandwidth_bps: f64,
    /// Peak theoretical bandwidth in bytes/second.
    pub peak_bandwidth_bps: f64,
    /// Bandwidth utilisation ratio (achieved / peak, 0.0–1.0).
    pub utilization: f64,
    /// Total bytes transferred (including wasted bytes from misaligned txns).
    pub total_bytes_transferred: usize,
    /// Useful bytes actually consumed by the kernel.
    pub useful_bytes: usize,
    /// Number of cache-line transactions issued.
    pub transactions: usize,
    /// Number of transactions that would be needed with perfect coalescing.
    pub ideal_transactions: usize,
}

impl MemoryAnalysis {
    /// Transaction efficiency (ideal / actual).
    pub fn transaction_efficiency(&self) -> f64 {
        if self.transactions == 0 {
            return 0.0;
        }
        self.ideal_transactions as f64 / self.transactions as f64
    }

    /// Wasted bytes ratio.
    pub fn waste_ratio(&self) -> f64 {
        if self.total_bytes_transferred == 0 {
            return 0.0;
        }
        1.0 - (self.useful_bytes as f64 / self.total_bytes_transferred as f64)
    }
}

// ── MemoryHierarchyStats ─────────────────────────────────────────────

/// Simulated per-level cache statistics.
#[derive(Debug, Clone, Default)]
pub struct MemoryHierarchyStats {
    /// L1 hits.
    pub l1_hits: usize,
    /// L1 misses.
    pub l1_misses: usize,
    /// L2 hits.
    pub l2_hits: usize,
    /// L2 misses.
    pub l2_misses: usize,
    /// DRAM accesses (L2 misses that go to global memory).
    pub dram_accesses: usize,
    /// Estimated average latency in cycles.
    pub avg_latency_cycles: f64,
}

impl MemoryHierarchyStats {
    /// L1 hit rate.
    pub fn l1_hit_rate(&self) -> f64 {
        let total = self.l1_hits + self.l1_misses;
        if total == 0 {
            return 0.0;
        }
        self.l1_hits as f64 / total as f64
    }

    /// L2 hit rate.
    pub fn l2_hit_rate(&self) -> f64 {
        let total = self.l2_hits + self.l2_misses;
        if total == 0 {
            return 0.0;
        }
        self.l2_hits as f64 / total as f64
    }
}

// ── Helper: round-up to alignment ────────────────────────────────────

/// Round `n` up to the next multiple of `align` (must be power of two).
#[inline]
fn align_up(n: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (n + align - 1) & !(align - 1)
}

// ── analyze_access_pattern ───────────────────────────────────────────

/// Classify an index sequence into a [`MemoryAccessPattern`].
///
/// Examines up to `WARP_SIZE` consecutive indices to determine the dominant
/// pattern. Returns `Coalesced` when successive indices differ by exactly 1,
/// `Strided` when a constant stride > 1 is detected, `Sequential` for a
/// length-1 sequence, and `Random` otherwise.
pub fn analyze_access_pattern(indices: &[usize]) -> MemoryAccessPattern {
    if indices.len() <= 1 {
        return MemoryAccessPattern::Sequential;
    }

    let window = &indices[..indices.len().min(WARP_SIZE + 1)];

    // Compute deltas.
    let first_delta = window[1] as isize - window[0] as isize;
    let mut all_same_delta = true;
    for pair in window.windows(2) {
        let delta = pair[1] as isize - pair[0] as isize;
        if delta != first_delta {
            all_same_delta = false;
            break;
        }
    }

    if !all_same_delta {
        return MemoryAccessPattern::Random;
    }

    if first_delta == 1 {
        MemoryAccessPattern::Coalesced
    } else if first_delta > 1 {
        MemoryAccessPattern::Strided { stride: first_delta as usize }
    } else {
        // Negative or zero stride counts as random.
        MemoryAccessPattern::Random
    }
}

// ── optimize_layout ──────────────────────────────────────────────────

/// Reorder `data` so that the elements accessed by `indices` are laid out
/// contiguously, returning the reordered buffer and a mapping from new to
/// old positions.
///
/// This is a gather-based layout transformation: element `i` in the output
/// comes from `data[indices[i]]`.
pub fn optimize_layout(data: &[f32], indices: &[usize]) -> Result<(Vec<f32>, Vec<usize>)> {
    for &idx in indices {
        if idx >= data.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!("index {idx} out of bounds for buffer of length {}", data.len()),
            }
            .into());
        }
    }
    let reordered: Vec<f32> = indices.iter().map(|&i| data[i]).collect();
    let mapping: Vec<usize> = indices.to_vec();
    Ok((reordered, mapping))
}

// ── coalesced_copy ───────────────────────────────────────────────────

/// Copy `src` into `dst` in chunks aligned to `config.alignment` bytes.
///
/// On GPU this would map to coalesced warp-wide loads/stores. The CPU
/// fallback copies in aligned chunks, modelling the transaction pattern.
pub fn coalesced_copy(src: &[f32], dst: &mut [f32], config: &BandwidthConfig) -> Result<usize> {
    config.validate()?;
    let len = src.len().min(dst.len());
    if len == 0 {
        return Ok(0);
    }

    let elem_bytes = std::mem::size_of::<f32>();
    let elems_per_txn = config.alignment / elem_bytes;
    let mut transactions = 0usize;

    let mut offset = 0;
    while offset < len {
        let chunk = (len - offset).min(elems_per_txn);
        dst[offset..offset + chunk].copy_from_slice(&src[offset..offset + chunk]);
        transactions += 1;
        offset += chunk;
    }

    Ok(transactions)
}

// ── vectorized_load ──────────────────────────────────────────────────

/// Wide load: read `data` in groups of `config.vectorize_width` elements.
///
/// Returns the loaded values (identical to `data` for the CPU path) and the
/// number of vector load operations performed.
pub fn vectorized_load(data: &[f32], config: &BandwidthConfig) -> Result<(Vec<f32>, usize)> {
    config.validate()?;
    let width = config.vectorize_width;
    let full_vecs = data.len() / width;
    let remainder = data.len() % width;
    let ops = full_vecs + if remainder > 0 { 1 } else { 0 };
    Ok((data.to_vec(), ops))
}

// ── vectorized_store ─────────────────────────────────────────────────

/// Wide store: write `data` into `dst` in groups of `config.vectorize_width`.
///
/// Returns the number of vector store operations performed.
pub fn vectorized_store(data: &[f32], dst: &mut [f32], config: &BandwidthConfig) -> Result<usize> {
    config.validate()?;
    let len = data.len().min(dst.len());
    dst[..len].copy_from_slice(&data[..len]);
    let width = config.vectorize_width;
    let full_vecs = len / width;
    let remainder = len % width;
    Ok(full_vecs + if remainder > 0 { 1 } else { 0 })
}

// ── prefetch_buffer ──────────────────────────────────────────────────

/// Simulate software prefetch by touching cache-line–aligned offsets.
///
/// On GPU this maps to `__prefetch_global_l2` or `ld.global.L2::128B`
/// instructions. The CPU fallback returns the set of byte offsets that
/// would be prefetched.
pub fn prefetch_buffer(len_bytes: usize, config: &BandwidthConfig) -> Result<Vec<usize>> {
    config.validate()?;
    let line = config.alignment;
    let distance = config.prefetch_distance;
    let mut offsets = Vec::new();
    let mut off = 0;
    while off < len_bytes {
        // Prefetch `distance` lines ahead.
        let prefetch_off = off + distance * line;
        if prefetch_off < len_bytes {
            offsets.push(prefetch_off);
        }
        off += line;
    }
    Ok(offsets)
}

// ── align_buffer ─────────────────────────────────────────────────────

/// Pad `data` so its length (in elements) is a multiple of
/// `config.alignment / sizeof(f32)`.
///
/// Returns the padded buffer and the number of padding elements added.
pub fn align_buffer(data: &[f32], config: &BandwidthConfig) -> Result<(Vec<f32>, usize)> {
    config.validate()?;
    let elem_bytes = std::mem::size_of::<f32>();
    let elems_per_align = config.alignment / elem_bytes;
    let aligned_len = align_up(data.len(), elems_per_align);
    let pad = aligned_len - data.len();
    let mut buf = data.to_vec();
    buf.resize(aligned_len, 0.0);
    Ok((buf, pad))
}

// ── transpose_for_coalescing ─────────────────────────────────────────

/// Transpose a row-major `[rows × cols]` matrix to column-major order so
/// that column-wise reads become coalesced.
///
/// The output has shape `[cols × rows]` in row-major layout, which is
/// equivalent to the input in column-major layout.
pub fn transpose_for_coalescing(data: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
    if data.len() != rows * cols {
        return Err(KernelError::InvalidArguments {
            reason: format!("buffer length {} != rows({}) * cols({})", data.len(), rows, cols),
        }
        .into());
    }
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    Ok(out)
}

// ── bank_conflict_free_access ────────────────────────────────────────

/// Remap a shared-memory index to avoid bank conflicts.
///
/// Uses the classic +1 padding trick: for a 2-D shared-memory tile of width
/// `tile_width`, the padded width becomes `tile_width + 1`, spreading
/// successive columns across different banks.
///
/// Returns the remapped linear index.
pub fn bank_conflict_free_access(row: usize, col: usize, tile_width: usize) -> usize {
    let padded_width = tile_width + 1;
    row * padded_width + col
}

// ── estimate_bandwidth ───────────────────────────────────────────────

/// Estimate achieved bandwidth given a transfer size and elapsed duration.
///
/// Returns bandwidth in bytes per second. `duration_secs` must be positive.
pub fn estimate_bandwidth(bytes: usize, duration_secs: f64) -> Result<f64> {
    if duration_secs <= 0.0 {
        return Err(KernelError::InvalidArguments {
            reason: "duration_secs must be positive".into(),
        }
        .into());
    }
    Ok(bytes as f64 / duration_secs)
}

// ── memory_hierarchy_stats ───────────────────────────────────────────

/// Produce a simplified memory-hierarchy simulation for `access_count`
/// accesses with a given working-set size and cache capacities.
///
/// The model assumes:
/// - L1 = 48 KiB per SM, ~28-cycle latency on hit
/// - L2 = 6 MiB (A100), ~200-cycle latency on hit
/// - DRAM = ~400-cycle latency
///
/// `working_set_bytes` is the total unique data footprint; accesses that
/// fit in a cache level hit there.
pub fn memory_hierarchy_stats(
    access_count: usize,
    working_set_bytes: usize,
) -> MemoryHierarchyStats {
    const L1_SIZE: usize = 48 * 1024;
    const L2_SIZE: usize = 6 * 1024 * 1024;
    const L1_LATENCY: f64 = 28.0;
    const L2_LATENCY: f64 = 200.0;
    const DRAM_LATENCY: f64 = 400.0;

    if access_count == 0 {
        return MemoryHierarchyStats::default();
    }

    let (l1_hits, l1_misses, l2_hits, l2_misses, dram);
    if working_set_bytes <= L1_SIZE {
        l1_hits = access_count;
        l1_misses = 0;
        l2_hits = 0;
        l2_misses = 0;
        dram = 0;
    } else if working_set_bytes <= L2_SIZE {
        // Simple model: fraction that fits in L1 hits there.
        let l1_frac = L1_SIZE as f64 / working_set_bytes as f64;
        l1_hits = (access_count as f64 * l1_frac) as usize;
        l1_misses = access_count - l1_hits;
        l2_hits = l1_misses;
        l2_misses = 0;
        dram = 0;
    } else {
        let l1_frac = L1_SIZE as f64 / working_set_bytes as f64;
        let l2_frac = L2_SIZE as f64 / working_set_bytes as f64;
        l1_hits = (access_count as f64 * l1_frac) as usize;
        l1_misses = access_count - l1_hits;
        l2_hits = (l1_misses as f64 * l2_frac) as usize;
        l2_misses = l1_misses - l2_hits;
        dram = l2_misses;
    }

    let total = access_count as f64;
    let avg_latency =
        (l1_hits as f64 * L1_LATENCY + l2_hits as f64 * L2_LATENCY + dram as f64 * DRAM_LATENCY)
            / total;

    MemoryHierarchyStats {
        l1_hits,
        l1_misses,
        l2_hits,
        l2_misses,
        dram_accesses: dram,
        avg_latency_cycles: avg_latency,
    }
}

// ── CUDA kernel source (GPU only) ────────────────────────────────────

/// CUDA C kernel source for memory bandwidth optimisation primitives.
///
/// Contains kernels for coalesced copy, vectorised load/store (`float4`),
/// software prefetch, and bank-conflict-free shared-memory transpose.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const MEMORY_BANDWIDTH_KERNEL_SRC: &str = r#"
// Coalesced copy — each warp loads/stores a contiguous 128-byte segment.
extern "C" __global__ void coalesced_copy_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Vectorised copy using float4 (128-bit loads/stores).
extern "C" __global__ void vectorized_copy_f32x4(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    int n_vec)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        dst[idx] = src[idx];
    }
}

// Bank-conflict-free transpose using shared memory with +1 padding.
#define TILE 32
extern "C" __global__ void transpose_coalesced_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
    __shared__ float tile[TILE][TILE + 1];

    int bx = blockIdx.x * TILE;
    int by = blockIdx.y * TILE;

    int ix = bx + threadIdx.x;
    int iy = by + threadIdx.y;
    if (ix < cols && iy < rows) {
        tile[threadIdx.y][threadIdx.x] = input[iy * cols + ix];
    }
    __syncthreads();

    int ox = by + threadIdx.x;
    int oy = bx + threadIdx.y;
    if (ox < rows && oy < cols) {
        output[oy * rows + ox] = tile[threadIdx.x][threadIdx.y];
    }
}

// Prefetch-assisted coalesced copy.
extern "C" __global__ void prefetch_copy_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n,
    int prefetch_dist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        // Prefetch ahead
        int pf = i + prefetch_dist * blockDim.x;
        if (pf < n) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"(src + pf));
        }
        dst[i] = src[i];
    }
}
"#;

/// CUDA launch configuration for the coalesced copy kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_coalesced_copy(n: usize) -> Result<(u32, u32)> {
    if n == 0 {
        return Err(KernelError::InvalidArguments { reason: "n must be non-zero".into() }.into());
    }
    let block_size = 256u32;
    let grid_size = ((n as u32) + block_size - 1) / block_size;
    Ok((grid_size, block_size))
}

/// CUDA launch configuration for the vectorised float4 copy kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_vectorized_copy(n_elements: usize) -> Result<(u32, u32)> {
    if n_elements == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "n_elements must be non-zero".into() }.into()
        );
    }
    let n_vec = ((n_elements + 3) / 4) as u32;
    let block_size = 256u32;
    let grid_size = (n_vec + block_size - 1) / block_size;
    Ok((grid_size, block_size))
}

/// CUDA launch configuration for the bank-conflict-free transpose kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_transpose_coalesced(rows: usize, cols: usize) -> Result<(u32, u32, u32, u32)> {
    if rows == 0 || cols == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "rows and cols must be non-zero".into(),
        }
        .into());
    }
    let tile = 32u32;
    let grid_x = ((cols as u32) + tile - 1) / tile;
    let grid_y = ((rows as u32) + tile - 1) / tile;
    Ok((grid_x, grid_y, tile, tile))
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── MemoryAccessPattern ──────────────────────────────────────────

    #[test]
    fn test_pattern_coalesced() {
        let idx: Vec<usize> = (0..32).collect();
        assert_eq!(analyze_access_pattern(&idx), MemoryAccessPattern::Coalesced);
    }

    #[test]
    fn test_pattern_strided() {
        let idx: Vec<usize> = (0..32).map(|i| i * 4).collect();
        assert_eq!(analyze_access_pattern(&idx), MemoryAccessPattern::Strided { stride: 4 });
    }

    #[test]
    fn test_pattern_random() {
        let idx = vec![0, 5, 2, 17, 3, 100, 7];
        assert_eq!(analyze_access_pattern(&idx), MemoryAccessPattern::Random);
    }

    #[test]
    fn test_pattern_sequential_single() {
        assert_eq!(analyze_access_pattern(&[42]), MemoryAccessPattern::Sequential,);
    }

    #[test]
    fn test_pattern_empty() {
        assert_eq!(analyze_access_pattern(&[]), MemoryAccessPattern::Sequential,);
    }

    #[test]
    fn test_pattern_two_elements_coalesced() {
        assert_eq!(analyze_access_pattern(&[10, 11]), MemoryAccessPattern::Coalesced,);
    }

    #[test]
    fn test_pattern_two_elements_strided() {
        assert_eq!(analyze_access_pattern(&[0, 8]), MemoryAccessPattern::Strided { stride: 8 },);
    }

    #[test]
    fn test_pattern_negative_stride_is_random() {
        assert_eq!(analyze_access_pattern(&[10, 5, 0]), MemoryAccessPattern::Random,);
    }

    #[test]
    fn test_pattern_zero_stride_is_random() {
        assert_eq!(analyze_access_pattern(&[5, 5, 5]), MemoryAccessPattern::Random,);
    }

    #[test]
    fn test_pattern_large_stride() {
        let idx: Vec<usize> = (0..32).map(|i| i * 1024).collect();
        assert_eq!(analyze_access_pattern(&idx), MemoryAccessPattern::Strided { stride: 1024 },);
    }

    // ── MemoryAccessPattern::efficiency ──────────────────────────────

    #[test]
    fn test_efficiency_coalesced() {
        assert!((MemoryAccessPattern::Coalesced.efficiency() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_efficiency_sequential() {
        assert!((MemoryAccessPattern::Sequential.efficiency() - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_efficiency_strided() {
        let e = MemoryAccessPattern::Strided { stride: 4 }.efficiency();
        assert!((e - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_efficiency_random() {
        assert!((MemoryAccessPattern::Random.efficiency() - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_efficiency_stride_1_is_max() {
        let e = MemoryAccessPattern::Strided { stride: 1 }.efficiency();
        assert!((e - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_efficiency_stride_0_clamped() {
        let e = MemoryAccessPattern::Strided { stride: 0 }.efficiency();
        assert!((e - 1.0).abs() < f64::EPSILON);
    }

    // ── BandwidthConfig ──────────────────────────────────────────────

    #[test]
    fn test_config_default_is_valid() {
        BandwidthConfig::default().validate().unwrap();
    }

    #[test]
    fn test_config_bad_alignment() {
        let cfg = BandwidthConfig { alignment: 3, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_bad_vector_width() {
        let cfg = BandwidthConfig { vectorize_width: 3, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_vector_width_1() {
        let cfg = BandwidthConfig { vectorize_width: 1, ..Default::default() };
        cfg.validate().unwrap();
    }

    #[test]
    fn test_config_vector_width_2() {
        let cfg = BandwidthConfig { vectorize_width: 2, ..Default::default() };
        cfg.validate().unwrap();
    }

    #[test]
    fn test_config_vector_width_4() {
        let cfg = BandwidthConfig { vectorize_width: 4, ..Default::default() };
        cfg.validate().unwrap();
    }

    #[test]
    fn test_config_alignment_power_of_two() {
        for exp in 0..12 {
            let cfg = BandwidthConfig { alignment: 1 << exp, ..Default::default() };
            cfg.validate().unwrap();
        }
    }

    // ── coalesced_copy ───────────────────────────────────────────────

    #[test]
    fn test_coalesced_copy_basic() {
        let src: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 64];
        let cfg = BandwidthConfig::default();
        let txns = coalesced_copy(&src, &mut dst, &cfg).unwrap();
        assert_eq!(src, dst);
        assert!(txns >= 1);
    }

    #[test]
    fn test_coalesced_copy_empty() {
        let cfg = BandwidthConfig::default();
        let txns = coalesced_copy(&[], &mut [], &cfg).unwrap();
        assert_eq!(txns, 0);
    }

    #[test]
    fn test_coalesced_copy_smaller_dst() {
        let src = vec![1.0f32; 100];
        let mut dst = vec![0.0f32; 50];
        let cfg = BandwidthConfig::default();
        coalesced_copy(&src, &mut dst, &cfg).unwrap();
        assert!(dst.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_coalesced_copy_transactions() {
        // 128-byte alignment = 32 f32 per txn.
        let src = vec![0.0f32; 64];
        let mut dst = vec![0.0f32; 64];
        let cfg = BandwidthConfig { alignment: 128, ..Default::default() };
        let txns = coalesced_copy(&src, &mut dst, &cfg).unwrap();
        assert_eq!(txns, 2);
    }

    #[test]
    fn test_coalesced_copy_unaligned_len() {
        let src = vec![1.0f32; 33];
        let mut dst = vec![0.0f32; 33];
        let cfg = BandwidthConfig { alignment: 128, ..Default::default() };
        let txns = coalesced_copy(&src, &mut dst, &cfg).unwrap();
        assert_eq!(txns, 2); // 32 + 1
        assert_eq!(src, dst);
    }

    // ── vectorized_load ──────────────────────────────────────────────

    #[test]
    fn test_vectorized_load_basic() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let cfg = BandwidthConfig::default(); // width=4
        let (loaded, ops) = vectorized_load(&data, &cfg).unwrap();
        assert_eq!(loaded, data);
        assert_eq!(ops, 4);
    }

    #[test]
    fn test_vectorized_load_remainder() {
        let data = vec![1.0f32; 7];
        let cfg = BandwidthConfig { vectorize_width: 4, ..Default::default() };
        let (_, ops) = vectorized_load(&data, &cfg).unwrap();
        assert_eq!(ops, 2); // 1 full + 1 remainder
    }

    #[test]
    fn test_vectorized_load_empty() {
        let cfg = BandwidthConfig::default();
        let (loaded, ops) = vectorized_load(&[], &cfg).unwrap();
        assert!(loaded.is_empty());
        assert_eq!(ops, 0);
    }

    #[test]
    fn test_vectorized_load_width_1() {
        let data = vec![1.0f32; 5];
        let cfg = BandwidthConfig { vectorize_width: 1, ..Default::default() };
        let (_, ops) = vectorized_load(&data, &cfg).unwrap();
        assert_eq!(ops, 5);
    }

    #[test]
    fn test_vectorized_load_width_2() {
        let data = vec![1.0f32; 5];
        let cfg = BandwidthConfig { vectorize_width: 2, ..Default::default() };
        let (_, ops) = vectorized_load(&data, &cfg).unwrap();
        assert_eq!(ops, 3); // 2 full + 1 remainder
    }

    // ── vectorized_store ─────────────────────────────────────────────

    #[test]
    fn test_vectorized_store_basic() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 16];
        let cfg = BandwidthConfig::default();
        let ops = vectorized_store(&data, &mut dst, &cfg).unwrap();
        assert_eq!(dst, data);
        assert_eq!(ops, 4);
    }

    #[test]
    fn test_vectorized_store_empty() {
        let cfg = BandwidthConfig::default();
        let ops = vectorized_store(&[], &mut [], &cfg).unwrap();
        assert_eq!(ops, 0);
    }

    #[test]
    fn test_vectorized_store_partial_dst() {
        let data = vec![1.0f32; 10];
        let mut dst = vec![0.0f32; 5];
        let cfg = BandwidthConfig { vectorize_width: 4, ..Default::default() };
        let ops = vectorized_store(&data, &mut dst, &cfg).unwrap();
        assert_eq!(ops, 2); // 1 full + 1 remainder
        assert!(dst.iter().all(|&v| v == 1.0));
    }

    // ── prefetch_buffer ──────────────────────────────────────────────

    #[test]
    fn test_prefetch_buffer_basic() {
        let cfg = BandwidthConfig { alignment: 128, prefetch_distance: 2, ..Default::default() };
        let offsets = prefetch_buffer(1024, &cfg).unwrap();
        assert!(!offsets.is_empty());
        // First prefetch: line 0 prefetches 2 lines ahead → offset 256.
        assert_eq!(offsets[0], 256);
    }

    #[test]
    fn test_prefetch_buffer_empty() {
        let cfg = BandwidthConfig::default();
        let offsets = prefetch_buffer(0, &cfg).unwrap();
        assert!(offsets.is_empty());
    }

    #[test]
    fn test_prefetch_buffer_small_buffer() {
        let cfg = BandwidthConfig { alignment: 128, prefetch_distance: 100, ..Default::default() };
        // Buffer too small for any prefetch to land inside.
        let offsets = prefetch_buffer(128, &cfg).unwrap();
        assert!(offsets.is_empty());
    }

    #[test]
    fn test_prefetch_offsets_ascending() {
        let cfg = BandwidthConfig { alignment: 64, prefetch_distance: 1, ..Default::default() };
        let offsets = prefetch_buffer(512, &cfg).unwrap();
        for pair in offsets.windows(2) {
            assert!(pair[1] > pair[0]);
        }
    }

    // ── align_buffer ─────────────────────────────────────────────────

    #[test]
    fn test_align_buffer_already_aligned() {
        let cfg = BandwidthConfig { alignment: 16, ..Default::default() };
        let data = vec![1.0f32; 4]; // 4 elems × 4 bytes = 16 bytes
        let (buf, pad) = align_buffer(&data, &cfg).unwrap();
        assert_eq!(pad, 0);
        assert_eq!(buf, data);
    }

    #[test]
    fn test_align_buffer_needs_padding() {
        let cfg = BandwidthConfig { alignment: 128, ..Default::default() };
        let data = vec![1.0f32; 5]; // 20 bytes → needs padding to 128 bytes (32 elems)
        let (buf, pad) = align_buffer(&data, &cfg).unwrap();
        assert_eq!(buf.len(), 32);
        assert_eq!(pad, 27);
        assert!(buf[5..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_align_buffer_empty() {
        let cfg = BandwidthConfig { alignment: 128, ..Default::default() };
        let (buf, pad) = align_buffer(&[], &cfg).unwrap();
        assert!(buf.is_empty());
        assert_eq!(pad, 0);
    }

    #[test]
    fn test_align_buffer_preserves_data() {
        let cfg = BandwidthConfig { alignment: 64, ..Default::default() };
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let (buf, _) = align_buffer(&data, &cfg).unwrap();
        assert_eq!(&buf[..10], &data[..]);
    }

    // ── transpose_for_coalescing ─────────────────────────────────────

    #[test]
    fn test_transpose_identity() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let out = transpose_for_coalescing(&data, 2, 2).unwrap();
        assert_eq!(out, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_3x2() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = transpose_for_coalescing(&data, 3, 2).unwrap();
        // Row-major [3×2] → Row-major [2×3]:
        // [[1,3,5],[2,4,6]]
        assert_eq!(out, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_transpose_1x1() {
        let out = transpose_for_coalescing(&[42.0], 1, 1).unwrap();
        assert_eq!(out, vec![42.0]);
    }

    #[test]
    fn test_transpose_bad_size() {
        let r = transpose_for_coalescing(&[1.0, 2.0], 3, 3);
        assert!(r.is_err());
    }

    #[test]
    fn test_transpose_roundtrip() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t1 = transpose_for_coalescing(&data, 3, 4).unwrap();
        let t2 = transpose_for_coalescing(&t1, 4, 3).unwrap();
        assert_eq!(t2, data);
    }

    // ── bank_conflict_free_access ────────────────────────────────────

    #[test]
    fn test_bank_conflict_free_linear() {
        // Row 0, cols 0..31 → 0, 1, 2, … 31 with padded width 33.
        for c in 0..32 {
            let idx = bank_conflict_free_access(0, c, 32);
            assert_eq!(idx, c);
        }
    }

    #[test]
    fn test_bank_conflict_free_row1() {
        let idx = bank_conflict_free_access(1, 0, 32);
        assert_eq!(idx, 33); // padded_width = 33
    }

    #[test]
    fn test_bank_conflict_free_different_rows() {
        // Two accesses to the same column in different rows should map to
        // different shared-memory banks.
        let a = bank_conflict_free_access(0, 0, 32) % SHARED_MEM_BANKS;
        let b = bank_conflict_free_access(1, 0, 32) % SHARED_MEM_BANKS;
        assert_ne!(a, b);
    }

    #[test]
    fn test_bank_conflict_free_tile16() {
        let idx = bank_conflict_free_access(2, 3, 16);
        // padded_width = 17, idx = 2*17+3 = 37
        assert_eq!(idx, 37);
    }

    // ── estimate_bandwidth ───────────────────────────────────────────

    #[test]
    fn test_estimate_bandwidth_basic() {
        let bw = estimate_bandwidth(1_000_000, 0.001).unwrap();
        assert!((bw - 1e9).abs() < 1.0); // 1 GB/s
    }

    #[test]
    fn test_estimate_bandwidth_zero_duration() {
        assert!(estimate_bandwidth(100, 0.0).is_err());
    }

    #[test]
    fn test_estimate_bandwidth_negative_duration() {
        assert!(estimate_bandwidth(100, -1.0).is_err());
    }

    #[test]
    fn test_estimate_bandwidth_zero_bytes() {
        let bw = estimate_bandwidth(0, 1.0).unwrap();
        assert!((bw - 0.0).abs() < f64::EPSILON);
    }

    // ── memory_hierarchy_stats ───────────────────────────────────────

    #[test]
    fn test_hierarchy_all_l1() {
        let stats = memory_hierarchy_stats(1000, 1024);
        assert_eq!(stats.l1_hits, 1000);
        assert_eq!(stats.l1_misses, 0);
        assert_eq!(stats.l2_hits, 0);
        assert_eq!(stats.dram_accesses, 0);
    }

    #[test]
    fn test_hierarchy_l2_spill() {
        let stats = memory_hierarchy_stats(1000, 128 * 1024); // 128 KiB > L1 (48 KiB)
        assert!(stats.l1_hits > 0);
        assert!(stats.l1_misses > 0);
        assert!(stats.l2_hits > 0);
        assert_eq!(stats.dram_accesses, 0);
    }

    #[test]
    fn test_hierarchy_dram_spill() {
        let stats = memory_hierarchy_stats(1000, 64 * 1024 * 1024); // 64 MiB > L2
        // L1 fraction is tiny (48 KiB / 64 MiB ≈ 0.07 %), so l1_hits may be 0.
        assert!(stats.l1_misses > 0);
        assert!(stats.l2_hits > 0);
        assert!(stats.dram_accesses > 0);
    }

    #[test]
    fn test_hierarchy_zero_accesses() {
        let stats = memory_hierarchy_stats(0, 1024);
        assert_eq!(stats.l1_hits, 0);
        assert_eq!(stats.l2_hits, 0);
        assert_eq!(stats.dram_accesses, 0);
    }

    #[test]
    fn test_hierarchy_l1_hit_rate() {
        let stats = memory_hierarchy_stats(1000, 1024);
        assert!((stats.l1_hit_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hierarchy_avg_latency_all_l1() {
        let stats = memory_hierarchy_stats(1000, 1024);
        assert!((stats.avg_latency_cycles - 28.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hierarchy_avg_latency_increases_with_working_set() {
        let small = memory_hierarchy_stats(1000, 1024);
        let large = memory_hierarchy_stats(1000, 64 * 1024 * 1024);
        assert!(large.avg_latency_cycles > small.avg_latency_cycles);
    }

    // ── MemoryAnalysis ───────────────────────────────────────────────

    #[test]
    fn test_analysis_transaction_efficiency() {
        let analysis = MemoryAnalysis {
            pattern: MemoryAccessPattern::Coalesced,
            achieved_bandwidth_bps: 0.0,
            peak_bandwidth_bps: 0.0,
            utilization: 0.0,
            total_bytes_transferred: 256,
            useful_bytes: 256,
            transactions: 2,
            ideal_transactions: 2,
        };
        assert!((analysis.transaction_efficiency() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analysis_transaction_efficiency_low() {
        let analysis = MemoryAnalysis {
            pattern: MemoryAccessPattern::Random,
            achieved_bandwidth_bps: 0.0,
            peak_bandwidth_bps: 0.0,
            utilization: 0.0,
            total_bytes_transferred: 4096,
            useful_bytes: 128,
            transactions: 32,
            ideal_transactions: 1,
        };
        assert!((analysis.transaction_efficiency() - 1.0 / 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_analysis_zero_transactions() {
        let analysis = MemoryAnalysis {
            pattern: MemoryAccessPattern::Coalesced,
            achieved_bandwidth_bps: 0.0,
            peak_bandwidth_bps: 0.0,
            utilization: 0.0,
            total_bytes_transferred: 0,
            useful_bytes: 0,
            transactions: 0,
            ideal_transactions: 0,
        };
        assert!((analysis.transaction_efficiency() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analysis_waste_ratio_zero() {
        let analysis = MemoryAnalysis {
            pattern: MemoryAccessPattern::Coalesced,
            achieved_bandwidth_bps: 0.0,
            peak_bandwidth_bps: 0.0,
            utilization: 0.0,
            total_bytes_transferred: 128,
            useful_bytes: 128,
            transactions: 1,
            ideal_transactions: 1,
        };
        assert!((analysis.waste_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analysis_waste_ratio_half() {
        let analysis = MemoryAnalysis {
            pattern: MemoryAccessPattern::Strided { stride: 2 },
            achieved_bandwidth_bps: 0.0,
            peak_bandwidth_bps: 0.0,
            utilization: 0.0,
            total_bytes_transferred: 256,
            useful_bytes: 128,
            transactions: 2,
            ideal_transactions: 1,
        };
        assert!((analysis.waste_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analysis_waste_ratio_zero_transfer() {
        let analysis = MemoryAnalysis {
            pattern: MemoryAccessPattern::Coalesced,
            achieved_bandwidth_bps: 0.0,
            peak_bandwidth_bps: 0.0,
            utilization: 0.0,
            total_bytes_transferred: 0,
            useful_bytes: 0,
            transactions: 0,
            ideal_transactions: 0,
        };
        assert!((analysis.waste_ratio() - 0.0).abs() < f64::EPSILON);
    }

    // ── optimize_layout ──────────────────────────────────────────────

    #[test]
    fn test_optimize_layout_basic() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let indices = vec![4, 2, 0];
        let (reordered, mapping) = optimize_layout(&data, &indices).unwrap();
        assert_eq!(reordered, vec![50.0, 30.0, 10.0]);
        assert_eq!(mapping, vec![4, 2, 0]);
    }

    #[test]
    fn test_optimize_layout_identity() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let indices: Vec<usize> = (0..8).collect();
        let (reordered, _) = optimize_layout(&data, &indices).unwrap();
        assert_eq!(reordered, data);
    }

    #[test]
    fn test_optimize_layout_out_of_bounds() {
        let data = vec![1.0, 2.0, 3.0];
        let indices = vec![0, 5];
        assert!(optimize_layout(&data, &indices).is_err());
    }

    #[test]
    fn test_optimize_layout_empty() {
        let data = vec![1.0, 2.0];
        let (reordered, mapping) = optimize_layout(&data, &[]).unwrap();
        assert!(reordered.is_empty());
        assert!(mapping.is_empty());
    }

    #[test]
    fn test_optimize_layout_duplicate_indices() {
        let data = vec![10.0, 20.0, 30.0];
        let indices = vec![1, 1, 1];
        let (reordered, _) = optimize_layout(&data, &indices).unwrap();
        assert_eq!(reordered, vec![20.0, 20.0, 20.0]);
    }

    // ── CachePolicy ─────────────────────────────────────────────────

    #[test]
    fn test_cache_policy_default() {
        assert_eq!(CachePolicy::default(), CachePolicy::CacheAll);
    }

    // ── Constants ────────────────────────────────────────────────────

    #[test]
    fn test_constants() {
        assert_eq!(CACHE_LINE_BYTES, 128);
        assert_eq!(WARP_SIZE, 32);
        assert_eq!(SHARED_MEM_BANKS, 32);
        assert_eq!(DEFAULT_ALIGNMENT, 128);
        assert_eq!(DEFAULT_VECTOR_WIDTH, 4);
    }

    // ── align_up helper ──────────────────────────────────────────────

    #[test]
    fn test_align_up_already_aligned() {
        assert_eq!(align_up(128, 128), 128);
    }

    #[test]
    fn test_align_up_needs_rounding() {
        assert_eq!(align_up(129, 128), 256);
    }

    #[test]
    fn test_align_up_zero() {
        assert_eq!(align_up(0, 128), 0);
    }

    #[test]
    fn test_align_up_one() {
        assert_eq!(align_up(1, 128), 128);
    }

    #[test]
    fn test_align_up_small_alignment() {
        assert_eq!(align_up(5, 4), 8);
    }

    // ── GPU launch configs ───────────────────────────────────────────

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    mod gpu_tests {
        use super::super::*;

        #[test]
        fn test_launch_coalesced_copy() {
            let (grid, block) = launch_coalesced_copy(1024).unwrap();
            assert_eq!(block, 256);
            assert_eq!(grid, 4);
        }

        #[test]
        fn test_launch_coalesced_copy_zero() {
            assert!(launch_coalesced_copy(0).is_err());
        }

        #[test]
        fn test_launch_vectorized_copy() {
            let (grid, block) = launch_vectorized_copy(1024).unwrap();
            assert_eq!(block, 256);
            assert_eq!(grid, 1); // 256 float4 ops
        }

        #[test]
        fn test_launch_vectorized_copy_zero() {
            assert!(launch_vectorized_copy(0).is_err());
        }

        #[test]
        fn test_launch_transpose_coalesced() {
            let (gx, gy, bx, by) = launch_transpose_coalesced(64, 128).unwrap();
            assert_eq!(bx, 32);
            assert_eq!(by, 32);
            assert_eq!(gx, 4);
            assert_eq!(gy, 2);
        }

        #[test]
        fn test_launch_transpose_coalesced_zero() {
            assert!(launch_transpose_coalesced(0, 10).is_err());
            assert!(launch_transpose_coalesced(10, 0).is_err());
        }

        #[test]
        fn test_kernel_src_not_empty() {
            assert!(!MEMORY_BANDWIDTH_KERNEL_SRC.is_empty());
            assert!(MEMORY_BANDWIDTH_KERNEL_SRC.contains("coalesced_copy_f32"));
            assert!(MEMORY_BANDWIDTH_KERNEL_SRC.contains("vectorized_copy_f32x4"));
            assert!(MEMORY_BANDWIDTH_KERNEL_SRC.contains("transpose_coalesced_f32"));
            assert!(MEMORY_BANDWIDTH_KERNEL_SRC.contains("prefetch_copy_f32"));
        }
    }

    // ── MemoryHierarchyStats rates ───────────────────────────────────

    #[test]
    fn test_hierarchy_l2_hit_rate_when_no_spill() {
        let stats = memory_hierarchy_stats(1000, 1024);
        // No L2 accesses → hit rate is 0.
        assert!((stats.l2_hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hierarchy_l2_hit_rate_with_l2_usage() {
        let stats = memory_hierarchy_stats(1000, 128 * 1024);
        assert!(stats.l2_hit_rate() > 0.0);
    }

    #[test]
    fn test_hierarchy_access_count_conservation() {
        let stats = memory_hierarchy_stats(1000, 64 * 1024 * 1024);
        let total = stats.l1_hits + stats.l1_misses;
        assert_eq!(total, 1000);
        // l2_hits + dram_accesses = l1_misses
        assert_eq!(stats.l2_hits + stats.dram_accesses, stats.l1_misses);
    }

    #[test]
    fn test_hierarchy_access_count_conservation_small() {
        let stats = memory_hierarchy_stats(1000, 1024);
        assert_eq!(stats.l1_hits + stats.l2_hits + stats.dram_accesses, 1000);
    }

    #[test]
    fn test_hierarchy_access_count_conservation_medium() {
        let stats = memory_hierarchy_stats(1000, 128 * 1024);
        assert_eq!(stats.l1_hits + stats.l2_hits + stats.dram_accesses, 1000);
    }
}
