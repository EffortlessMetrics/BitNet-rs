//! Cooperative groups kernel module for block/grid-level parallelism.
//!
//! This module provides CPU-simulated cooperative group primitives that mirror
//! CUDA Cooperative Groups API. On GPU these map to hardware-accelerated
//! `cooperative_groups::thread_block`, `cooperative_groups::grid_group`, and
//! `cooperative_groups::coalesced_threads()`. The CPU fallback performs equivalent
//! sequential simulation for correctness testing and non-GPU environments.
//!
//! # Group types
//!
//! - [`ThreadBlockGroup`] — thread indexing and synchronization within a block
//! - [`GridGroup`] — grid-wide synchronization across all blocks
//! - [`CoalescedGroup`] — dynamic warp-level cooperation for active threads
//!
//! # Collective operations
//!
//! - [`cooperative_reduce`] — block-level reduction using shared memory
//! - [`cooperative_scan`] — inclusive prefix scan across threads
//! - [`cooperative_broadcast`] — single-thread value broadcast to all
//! - [`cooperative_sort`] — bitonic sort within a cooperative group
//! - [`cooperative_histogram`] — block-level histogram computation
//! - [`cooperative_matmul`] — tiled matrix multiplication using thread cooperation
//!
//! # CUDA kernel source
//!
//! `COOPERATIVE_GROUPS_KERNEL_SRC` contains CUDA C kernels that use the
//! CUDA Cooperative Groups API. Feature-gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// CUDA kernel source — cooperative groups
// ---------------------------------------------------------------------------

/// CUDA C kernel source implementing cooperative group operations.
///
/// Contains kernels for block-level reduction, prefix scan, broadcast,
/// bitonic sort, histogram, and tiled matrix multiplication using
/// CUDA Cooperative Groups API (`cooperative_groups.h`).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const COOPERATIVE_GROUPS_KERNEL_SRC: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Block-level reduction using cooperative groups shared memory.
// Supports sum, min, max, product via template-style op selection.
extern "C" __global__ void coop_reduce_sum_f32(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int n)
{
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ float sdata[];
    int tid = block.thread_rank();
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    block.sync();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        block.sync();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

extern "C" __global__ void coop_reduce_max_f32(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int n)
{
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ float sdata[];
    int tid = block.thread_rank();
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? input[idx] : -1e38f;
    block.sync();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        block.sync();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

extern "C" __global__ void coop_reduce_min_f32(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int n)
{
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ float sdata[];
    int tid = block.thread_rank();
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? input[idx] : 1e38f;
    block.sync();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        block.sync();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

extern "C" __global__ void coop_reduce_product_f32(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int n)
{
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ float sdata[];
    int tid = block.thread_rank();
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? input[idx] : 1.0f;
    block.sync();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] *= sdata[tid + s];
        block.sync();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Inclusive prefix scan (Blelloch-style up-sweep/down-sweep).
extern "C" __global__ void coop_scan_f32(
    float* __restrict__ data,
    int n)
{
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ float sdata[];
    int tid = block.thread_rank();
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
    block.sync();
    // Up-sweep
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        float val = 0.0f;
        if (tid >= stride) val = sdata[tid - stride];
        block.sync();
        if (tid >= stride) sdata[tid] += val;
        block.sync();
    }
    if (idx < n) data[idx] = sdata[tid];
}

// Broadcast value from thread 0 to all threads in block.
extern "C" __global__ void coop_broadcast_f32(
    float* __restrict__ data,
    int n,
    int src_thread)
{
    cg::thread_block block = cg::this_thread_block();
    __shared__ float broadcast_val;
    int tid = block.thread_rank();
    int idx = blockIdx.x * blockDim.x + tid;
    if (tid == src_thread && idx < n) broadcast_val = data[idx];
    block.sync();
    if (idx < n) data[idx] = broadcast_val;
}

// Block-level histogram using shared memory atomics.
extern "C" __global__ void coop_histogram_u32(
    const unsigned int* __restrict__ input,
    unsigned int*       __restrict__ output,
    int n,
    int num_bins)
{
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ unsigned int shist[];
    int tid = block.thread_rank();
    for (int b = tid; b < num_bins; b += blockDim.x) shist[b] = 0;
    block.sync();
    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < n && input[idx] < (unsigned int)num_bins) {
        atomicAdd(&shist[input[idx]], 1);
    }
    block.sync();
    for (int b = tid; b < num_bins; b += blockDim.x) {
        atomicAdd(&output[b], shist[b]);
    }
}

// Tiled matrix multiplication using cooperative thread block.
extern "C" __global__ void coop_matmul_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    const int TILE = 16;
    cg::thread_block block = cg::this_thread_block();
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        block.sync();
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        block.sync();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// Grid-wide synchronization kernel (requires cooperative launch).
extern "C" __global__ void coop_grid_sync_f32(
    float* __restrict__ data,
    int n)
{
    cg::grid_group grid = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
    grid.sync();
    if (idx < n) data[idx] += 1.0f;
}
"#;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Reduction operation type for cooperative reduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CooperativeReduceOp {
    /// Sum all values.
    Sum,
    /// Find maximum value.
    Max,
    /// Find minimum value.
    Min,
    /// Multiply all values.
    Product,
}

/// Configuration for cooperative group operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CooperativeGroupConfig {
    /// Number of threads per group (must be power of 2 for some ops, max 1024).
    pub group_size: u32,
    /// Enable grid-wide synchronization (requires cooperative kernel launch).
    pub grid_sync: bool,
    /// Enable thread block cluster support (SM 9.0+, Hopper architecture).
    pub thread_block_cluster: bool,
    /// Cluster size in blocks (1–16, only used when `thread_block_cluster` is true).
    pub cluster_size: u32,
    /// Shared memory size in bytes (0 = auto-computed).
    pub shared_mem_bytes: u32,
}

impl Default for CooperativeGroupConfig {
    fn default() -> Self {
        Self {
            group_size: 256,
            grid_sync: false,
            thread_block_cluster: false,
            cluster_size: 1,
            shared_mem_bytes: 0,
        }
    }
}

impl CooperativeGroupConfig {
    /// Create a new config with the given group size.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `group_size` is 0 or exceeds 1024.
    pub fn new(group_size: u32) -> Result<Self> {
        if group_size == 0 || group_size > 1024 {
            return Err(KernelError::InvalidArguments {
                reason: format!("group_size must be 1..=1024, got {group_size}"),
            }
            .into());
        }
        Ok(Self { group_size, ..Default::default() })
    }

    /// Enable grid-wide synchronization.
    #[must_use]
    pub fn with_grid_sync(mut self) -> Self {
        self.grid_sync = true;
        self
    }

    /// Enable thread block cluster with the given cluster size.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `size` is 0 or exceeds 16.
    pub fn with_cluster(mut self, size: u32) -> Result<Self> {
        if size == 0 || size > 16 {
            return Err(KernelError::InvalidArguments {
                reason: format!("cluster_size must be 1..=16, got {size}"),
            }
            .into());
        }
        self.thread_block_cluster = true;
        self.cluster_size = size;
        Ok(self)
    }

    /// Set explicit shared memory allocation in bytes.
    #[must_use]
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    /// Compute the required shared memory for a reduction on this group.
    pub fn reduction_shared_mem(&self) -> u32 {
        self.group_size * 4 // f32 per thread
    }

    /// Check whether `group_size` is a power of two.
    pub fn is_power_of_two(&self) -> bool {
        self.group_size.is_power_of_two()
    }
}

// ---------------------------------------------------------------------------
// Group types — CPU simulation
// ---------------------------------------------------------------------------

/// Simulated thread block group for indexing and synchronization.
///
/// Models a CUDA `cooperative_groups::thread_block` with thread rank,
/// block dimensions, and synchronization barrier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadBlockGroup {
    /// Number of threads in the block.
    pub num_threads: u32,
    /// Block index within the grid.
    pub block_idx: u32,
}

impl ThreadBlockGroup {
    /// Create a new thread block group.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `num_threads` is 0.
    pub fn new(num_threads: u32, block_idx: u32) -> Result<Self> {
        if num_threads == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "thread block must have at least 1 thread".into(),
            }
            .into());
        }
        Ok(Self { num_threads, block_idx })
    }

    /// Get the thread rank (0-based index) for a given global thread index.
    pub fn thread_rank(&self, global_idx: u32) -> u32 {
        global_idx % self.num_threads
    }

    /// Total number of threads in this block.
    pub fn size(&self) -> u32 {
        self.num_threads
    }

    /// Check whether a global index belongs to this block.
    pub fn contains(&self, global_idx: u32) -> bool {
        global_idx / self.num_threads == self.block_idx
    }
}

/// Simulated grid group for grid-wide synchronization.
///
/// Models a CUDA `cooperative_groups::grid_group` with grid dimensions
/// and a global synchronization barrier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridGroup {
    /// Threads per block.
    pub block_size: u32,
    /// Number of blocks in the grid.
    pub num_blocks: u32,
}

impl GridGroup {
    /// Create a new grid group.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if either dimension is 0.
    pub fn new(block_size: u32, num_blocks: u32) -> Result<Self> {
        if block_size == 0 || num_blocks == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "grid dimensions must be non-zero: block_size={block_size}, num_blocks={num_blocks}"
                ),
            }
            .into());
        }
        Ok(Self { block_size, num_blocks })
    }

    /// Total number of threads across all blocks.
    pub fn total_threads(&self) -> u64 {
        u64::from(self.block_size) * u64::from(self.num_blocks)
    }

    /// Get the block index for a given global thread index.
    pub fn block_of(&self, global_idx: u32) -> u32 {
        global_idx / self.block_size
    }

    /// Get the thread rank within its block.
    pub fn thread_rank(&self, global_idx: u32) -> u32 {
        global_idx % self.block_size
    }
}

/// Simulated coalesced group for warp-level cooperation.
///
/// Models a CUDA `cooperative_groups::coalesced_threads()` representing the
/// dynamically converged subset of threads within a warp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoalescedGroup {
    /// Active thread mask within the warp.
    pub active_mask: u32,
}

impl CoalescedGroup {
    /// Create a coalesced group from an active thread mask.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if the mask is zero.
    pub fn new(active_mask: u32) -> Result<Self> {
        if active_mask == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "coalesced group mask must have at least one active thread".into(),
            }
            .into());
        }
        Ok(Self { active_mask })
    }

    /// Create a fully active coalesced group (all 32 warp lanes).
    pub fn full_warp() -> Self {
        Self { active_mask: 0xFFFF_FFFF }
    }

    /// Number of active threads in the group.
    pub fn size(&self) -> u32 {
        self.active_mask.count_ones()
    }

    /// Get the n-th active thread's lane index.
    pub fn nth_active(&self, n: u32) -> Option<u32> {
        let mut count = 0u32;
        for lane in 0..32u32 {
            if self.active_mask & (1 << lane) != 0 {
                if count == n {
                    return Some(lane);
                }
                count += 1;
            }
        }
        None
    }

    /// Check if a specific lane is active.
    pub fn is_active(&self, lane: u32) -> bool {
        lane < 32 && (self.active_mask & (1 << lane)) != 0
    }
}

// ---------------------------------------------------------------------------
// Collective operations — CPU fallback implementations
// ---------------------------------------------------------------------------

/// Block-level cooperative reduction.
///
/// Reduces `input` using the specified operation, simulating a CUDA block-level
/// reduction with shared memory. The group config determines the block size;
/// input is logically partitioned into blocks and each block produces one output.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `input` is empty.
pub fn cooperative_reduce(
    input: &[f32],
    op: CooperativeReduceOp,
    config: &CooperativeGroupConfig,
) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "cooperative_reduce input must not be empty".into(),
        }
        .into());
    }
    let block_size = config.group_size as usize;
    let num_blocks = input.len().div_ceil(block_size);
    let mut output = Vec::with_capacity(num_blocks);
    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(input.len());
        let chunk = &input[start..end];
        let val = match op {
            CooperativeReduceOp::Sum => chunk.iter().sum(),
            CooperativeReduceOp::Max => chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            CooperativeReduceOp::Min => chunk.iter().copied().fold(f32::INFINITY, f32::min),
            CooperativeReduceOp::Product => chunk.iter().product(),
        };
        output.push(val);
    }
    Ok(output)
}

/// Inclusive prefix scan across threads.
///
/// Computes an inclusive prefix sum in-place, simulating a CUDA block-level
/// Blelloch scan with shared memory. Works on the full slice regardless of
/// group size configuration (the GPU kernel processes per-block chunks).
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is empty.
pub fn cooperative_scan(data: &mut [f32], config: &CooperativeGroupConfig) -> Result<()> {
    if data.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "cooperative_scan input must not be empty".into(),
        }
        .into());
    }
    let block_size = config.group_size as usize;
    for block_start in (0..data.len()).step_by(block_size) {
        let block_end = (block_start + block_size).min(data.len());
        let mut running = 0.0f32;
        for val in &mut data[block_start..block_end] {
            running += *val;
            *val = running;
        }
    }
    Ok(())
}

/// Broadcast a value from `src_thread` to all threads in the group.
///
/// Simulates shared-memory broadcast within a CUDA thread block. Each
/// logical block of `group_size` threads broadcasts from its local
/// `src_thread` index.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is empty or
/// `src_thread` >= `group_size`.
pub fn cooperative_broadcast(
    data: &mut [f32],
    src_thread: u32,
    config: &CooperativeGroupConfig,
) -> Result<()> {
    if data.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "cooperative_broadcast input must not be empty".into(),
        }
        .into());
    }
    if src_thread >= config.group_size {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "src_thread ({src_thread}) must be < group_size ({})",
                config.group_size
            ),
        }
        .into());
    }
    let block_size = config.group_size as usize;
    let src = src_thread as usize;
    for block_start in (0..data.len()).step_by(block_size) {
        let block_end = (block_start + block_size).min(data.len());
        let src_idx = block_start + src;
        if src_idx < block_end {
            let val = data[src_idx];
            for item in &mut data[block_start..block_end] {
                *item = val;
            }
        }
    }
    Ok(())
}

/// Bitonic sort within a cooperative group.
///
/// Sorts `data` in ascending order using the bitonic sort network, which maps
/// naturally to CUDA thread cooperation. Input length is rounded up internally
/// to the next power of two using `f32::INFINITY` padding.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is empty.
pub fn cooperative_sort(data: &mut [f32], _config: &CooperativeGroupConfig) -> Result<()> {
    if data.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "cooperative_sort input must not be empty".into(),
        }
        .into());
    }
    let n = data.len();
    if n == 1 {
        return Ok(());
    }
    // Pad to next power of two
    let padded_len = n.next_power_of_two();
    let mut buf = data.to_vec();
    buf.resize(padded_len, f32::INFINITY);

    // Bitonic sort network
    let mut k = 2;
    while k <= padded_len {
        let mut j = k / 2;
        while j > 0 {
            for i in 0..padded_len {
                let partner = i ^ j;
                if partner > i {
                    let ascending = (i & k) == 0;
                    if (ascending && buf[i] > buf[partner]) || (!ascending && buf[i] < buf[partner])
                    {
                        buf.swap(i, partner);
                    }
                }
            }
            j /= 2;
        }
        k *= 2;
    }
    data.copy_from_slice(&buf[..n]);
    Ok(())
}

/// Block-level histogram computation.
///
/// Counts occurrences of each value in `input` that falls within `[0, num_bins)`.
/// Values outside this range are ignored. Simulates shared-memory atomic
/// histogram on GPU.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `num_bins` is 0.
pub fn cooperative_histogram(
    input: &[u32],
    num_bins: u32,
    _config: &CooperativeGroupConfig,
) -> Result<Vec<u32>> {
    if num_bins == 0 {
        return Err(KernelError::InvalidArguments { reason: "num_bins must be > 0".into() }.into());
    }
    let mut hist = vec![0u32; num_bins as usize];
    for &val in input {
        if val < num_bins {
            hist[val as usize] += 1;
        }
    }
    Ok(hist)
}

/// Tiled matrix multiplication using cooperative thread blocks.
///
/// Computes `C = A × B` where `A` is `M×K` and `B` is `K×N` (row-major).
/// The CPU fallback performs a straightforward triple-loop; the corresponding
/// CUDA kernel uses 16×16 shared-memory tiles with block synchronization.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if dimensions are zero or
/// slice lengths do not match the declared dimensions.
pub fn cooperative_matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    _config: &CooperativeGroupConfig,
) -> Result<Vec<f32>> {
    if m == 0 || n == 0 || k == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("matmul dimensions must be non-zero: M={m}, N={n}, K={k}"),
        }
        .into());
    }
    if a.len() < m * k {
        return Err(KernelError::InvalidArguments {
            reason: format!("A length {} < M*K = {}", a.len(), m * k),
        }
        .into());
    }
    if b.len() < k * n {
        return Err(KernelError::InvalidArguments {
            reason: format!("B length {} < K*N = {}", b.len(), k * n),
        }
        .into());
    }
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    Ok(c)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===== CooperativeGroupConfig tests =====

    #[test]
    fn test_config_default() {
        let cfg = CooperativeGroupConfig::default();
        assert_eq!(cfg.group_size, 256);
        assert!(!cfg.grid_sync);
        assert!(!cfg.thread_block_cluster);
        assert_eq!(cfg.cluster_size, 1);
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn test_config_new_valid() {
        let cfg = CooperativeGroupConfig::new(128).unwrap();
        assert_eq!(cfg.group_size, 128);
    }

    #[test]
    fn test_config_new_min() {
        let cfg = CooperativeGroupConfig::new(1).unwrap();
        assert_eq!(cfg.group_size, 1);
    }

    #[test]
    fn test_config_new_max() {
        let cfg = CooperativeGroupConfig::new(1024).unwrap();
        assert_eq!(cfg.group_size, 1024);
    }

    #[test]
    fn test_config_new_zero_errors() {
        assert!(CooperativeGroupConfig::new(0).is_err());
    }

    #[test]
    fn test_config_new_over_max_errors() {
        assert!(CooperativeGroupConfig::new(1025).is_err());
    }

    #[test]
    fn test_config_with_grid_sync() {
        let cfg = CooperativeGroupConfig::new(256).unwrap().with_grid_sync();
        assert!(cfg.grid_sync);
    }

    #[test]
    fn test_config_with_cluster_valid() {
        let cfg = CooperativeGroupConfig::new(256).unwrap().with_cluster(4).unwrap();
        assert!(cfg.thread_block_cluster);
        assert_eq!(cfg.cluster_size, 4);
    }

    #[test]
    fn test_config_with_cluster_max() {
        let cfg = CooperativeGroupConfig::new(256).unwrap().with_cluster(16).unwrap();
        assert_eq!(cfg.cluster_size, 16);
    }

    #[test]
    fn test_config_with_cluster_zero_errors() {
        assert!(CooperativeGroupConfig::new(256).unwrap().with_cluster(0).is_err());
    }

    #[test]
    fn test_config_with_cluster_over_max_errors() {
        assert!(CooperativeGroupConfig::new(256).unwrap().with_cluster(17).is_err());
    }

    #[test]
    fn test_config_with_shared_mem() {
        let cfg = CooperativeGroupConfig::new(256).unwrap().with_shared_mem(4096);
        assert_eq!(cfg.shared_mem_bytes, 4096);
    }

    #[test]
    fn test_config_reduction_shared_mem() {
        let cfg = CooperativeGroupConfig::new(256).unwrap();
        assert_eq!(cfg.reduction_shared_mem(), 1024); // 256 * 4
    }

    #[test]
    fn test_config_is_power_of_two() {
        assert!(CooperativeGroupConfig::new(256).unwrap().is_power_of_two());
        assert!(CooperativeGroupConfig::new(1).unwrap().is_power_of_two());
        assert!(!CooperativeGroupConfig::new(3).unwrap().is_power_of_two());
        assert!(!CooperativeGroupConfig::new(100).unwrap().is_power_of_two());
    }

    #[test]
    fn test_config_chained_builder() {
        let cfg = CooperativeGroupConfig::new(512)
            .unwrap()
            .with_grid_sync()
            .with_cluster(8)
            .unwrap()
            .with_shared_mem(8192);
        assert_eq!(cfg.group_size, 512);
        assert!(cfg.grid_sync);
        assert!(cfg.thread_block_cluster);
        assert_eq!(cfg.cluster_size, 8);
        assert_eq!(cfg.shared_mem_bytes, 8192);
    }

    // ===== ThreadBlockGroup tests =====

    #[test]
    fn test_thread_block_new() {
        let tb = ThreadBlockGroup::new(128, 0).unwrap();
        assert_eq!(tb.num_threads, 128);
        assert_eq!(tb.block_idx, 0);
    }

    #[test]
    fn test_thread_block_zero_threads_errors() {
        assert!(ThreadBlockGroup::new(0, 0).is_err());
    }

    #[test]
    fn test_thread_block_size() {
        let tb = ThreadBlockGroup::new(256, 0).unwrap();
        assert_eq!(tb.size(), 256);
    }

    #[test]
    fn test_thread_block_thread_rank() {
        let tb = ThreadBlockGroup::new(128, 0).unwrap();
        assert_eq!(tb.thread_rank(0), 0);
        assert_eq!(tb.thread_rank(1), 1);
        assert_eq!(tb.thread_rank(127), 127);
        assert_eq!(tb.thread_rank(128), 0); // wraps around
    }

    #[test]
    fn test_thread_block_contains() {
        let tb = ThreadBlockGroup::new(128, 1).unwrap();
        assert!(!tb.contains(0)); // block 0
        assert!(!tb.contains(127)); // block 0
        assert!(tb.contains(128)); // block 1
        assert!(tb.contains(255)); // block 1
        assert!(!tb.contains(256)); // block 2
    }

    #[test]
    fn test_thread_block_rank_multi_block() {
        let tb = ThreadBlockGroup::new(64, 2).unwrap();
        assert_eq!(tb.thread_rank(128), 0); // first thread of block 2
        assert_eq!(tb.thread_rank(191), 63); // last thread of block 2
    }

    // ===== GridGroup tests =====

    #[test]
    fn test_grid_group_new() {
        let gg = GridGroup::new(256, 4).unwrap();
        assert_eq!(gg.block_size, 256);
        assert_eq!(gg.num_blocks, 4);
    }

    #[test]
    fn test_grid_group_zero_block_size_errors() {
        assert!(GridGroup::new(0, 4).is_err());
    }

    #[test]
    fn test_grid_group_zero_num_blocks_errors() {
        assert!(GridGroup::new(256, 0).is_err());
    }

    #[test]
    fn test_grid_group_total_threads() {
        let gg = GridGroup::new(256, 4).unwrap();
        assert_eq!(gg.total_threads(), 1024);
    }

    #[test]
    fn test_grid_group_total_threads_large() {
        let gg = GridGroup::new(1024, 65535).unwrap();
        assert_eq!(gg.total_threads(), 1024 * 65535);
    }

    #[test]
    fn test_grid_group_block_of() {
        let gg = GridGroup::new(256, 4).unwrap();
        assert_eq!(gg.block_of(0), 0);
        assert_eq!(gg.block_of(255), 0);
        assert_eq!(gg.block_of(256), 1);
        assert_eq!(gg.block_of(768), 3);
    }

    #[test]
    fn test_grid_group_thread_rank() {
        let gg = GridGroup::new(128, 8).unwrap();
        assert_eq!(gg.thread_rank(0), 0);
        assert_eq!(gg.thread_rank(127), 127);
        assert_eq!(gg.thread_rank(128), 0);
        assert_eq!(gg.thread_rank(130), 2);
    }

    // ===== CoalescedGroup tests =====

    #[test]
    fn test_coalesced_full_warp() {
        let cg = CoalescedGroup::full_warp();
        assert_eq!(cg.size(), 32);
        assert_eq!(cg.active_mask, 0xFFFF_FFFF);
    }

    #[test]
    fn test_coalesced_new_valid() {
        let cg = CoalescedGroup::new(0b1010_1010).unwrap();
        assert_eq!(cg.size(), 4);
    }

    #[test]
    fn test_coalesced_new_zero_errors() {
        assert!(CoalescedGroup::new(0).is_err());
    }

    #[test]
    fn test_coalesced_is_active() {
        let cg = CoalescedGroup::new(0b1101).unwrap();
        assert!(cg.is_active(0));
        assert!(!cg.is_active(1));
        assert!(cg.is_active(2));
        assert!(cg.is_active(3));
        assert!(!cg.is_active(4));
        assert!(!cg.is_active(32));
    }

    #[test]
    fn test_coalesced_nth_active() {
        let cg = CoalescedGroup::new(0b1010_0110).unwrap();
        // Active lanes: 1, 2, 5, 7
        assert_eq!(cg.nth_active(0), Some(1));
        assert_eq!(cg.nth_active(1), Some(2));
        assert_eq!(cg.nth_active(2), Some(5));
        assert_eq!(cg.nth_active(3), Some(7));
        assert_eq!(cg.nth_active(4), None);
    }

    #[test]
    fn test_coalesced_single_thread() {
        let cg = CoalescedGroup::new(1).unwrap();
        assert_eq!(cg.size(), 1);
        assert!(cg.is_active(0));
        assert!(!cg.is_active(1));
        assert_eq!(cg.nth_active(0), Some(0));
        assert_eq!(cg.nth_active(1), None);
    }

    // ===== cooperative_reduce tests =====

    fn default_config() -> CooperativeGroupConfig {
        CooperativeGroupConfig::default()
    }

    #[test]
    fn test_reduce_sum_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result =
            cooperative_reduce(&input, CooperativeReduceOp::Sum, &default_config()).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_sum_multi_block() {
        let cfg = CooperativeGroupConfig::new(4).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let result = cooperative_reduce(&input, CooperativeReduceOp::Sum, &cfg).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 10.0).abs() < 1e-6);
        assert!((result[1] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_sum_partial_last_block() {
        let cfg = CooperativeGroupConfig::new(4).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = cooperative_reduce(&input, CooperativeReduceOp::Sum, &cfg).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 10.0).abs() < 1e-6);
        assert!((result[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_max_basic() {
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let result =
            cooperative_reduce(&input, CooperativeReduceOp::Max, &default_config()).unwrap();
        assert!((result[0] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_max_multi_block() {
        let cfg = CooperativeGroupConfig::new(4).unwrap();
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let result = cooperative_reduce(&input, CooperativeReduceOp::Max, &cfg).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 4.0).abs() < 1e-6);
        assert!((result[1] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_min_basic() {
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let result =
            cooperative_reduce(&input, CooperativeReduceOp::Min, &default_config()).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_min_multi_block() {
        let cfg = CooperativeGroupConfig::new(4).unwrap();
        let input = vec![3.0, 7.0, 4.0, 8.0, 5.0, 1.0, 2.0, 6.0];
        let result = cooperative_reduce(&input, CooperativeReduceOp::Min, &cfg).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 3.0).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_product_basic() {
        let input = vec![2.0, 3.0, 4.0];
        let result =
            cooperative_reduce(&input, CooperativeReduceOp::Product, &default_config()).unwrap();
        assert!((result[0] - 24.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_product_with_one() {
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let result =
            cooperative_reduce(&input, CooperativeReduceOp::Product, &default_config()).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_product_with_zero() {
        let input = vec![5.0, 3.0, 0.0, 7.0];
        let result =
            cooperative_reduce(&input, CooperativeReduceOp::Product, &default_config()).unwrap();
        assert!((result[0]).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_empty_errors() {
        let input: Vec<f32> = vec![];
        assert!(cooperative_reduce(&input, CooperativeReduceOp::Sum, &default_config()).is_err());
    }

    #[test]
    fn test_reduce_single_element() {
        let input = vec![42.0];
        let result =
            cooperative_reduce(&input, CooperativeReduceOp::Sum, &default_config()).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_sum_negative_values() {
        let input = vec![-1.0, -2.0, -3.0, -4.0];
        let result =
            cooperative_reduce(&input, CooperativeReduceOp::Sum, &default_config()).unwrap();
        assert!((result[0] - (-10.0)).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_max_all_negative() {
        let input = vec![-10.0, -5.0, -20.0, -1.0];
        let result =
            cooperative_reduce(&input, CooperativeReduceOp::Max, &default_config()).unwrap();
        assert!((result[0] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_min_all_positive() {
        let input = vec![100.0, 50.0, 200.0, 1.0];
        let result =
            cooperative_reduce(&input, CooperativeReduceOp::Min, &default_config()).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_sum_large_block() {
        let cfg = CooperativeGroupConfig::new(1024).unwrap();
        let input: Vec<f32> = (1..=1024).map(|x| x as f32).collect();
        let result = cooperative_reduce(&input, CooperativeReduceOp::Sum, &cfg).unwrap();
        let expected: f32 = (1..=1024).sum::<u32>() as f32;
        assert!((result[0] - expected).abs() < 1.0); // f32 summation tolerance
    }

    // ===== cooperative_scan tests =====

    #[test]
    fn test_scan_basic() {
        let cfg = CooperativeGroupConfig::new(8).unwrap();
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        cooperative_scan(&mut data, &cfg).unwrap();
        let expected = [1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0];
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn test_scan_multi_block() {
        let cfg = CooperativeGroupConfig::new(4).unwrap();
        let mut data = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        cooperative_scan(&mut data, &cfg).unwrap();
        let expected = [1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0];
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn test_scan_single_element() {
        let cfg = CooperativeGroupConfig::new(4).unwrap();
        let mut data = vec![7.0];
        cooperative_scan(&mut data, &cfg).unwrap();
        assert!((data[0] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_scan_empty_errors() {
        let cfg = default_config();
        let mut data: Vec<f32> = vec![];
        assert!(cooperative_scan(&mut data, &cfg).is_err());
    }

    #[test]
    fn test_scan_all_zeros() {
        let cfg = CooperativeGroupConfig::new(4).unwrap();
        let mut data = vec![0.0, 0.0, 0.0, 0.0];
        cooperative_scan(&mut data, &cfg).unwrap();
        for val in &data {
            assert!((val - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_scan_partial_block() {
        let cfg = CooperativeGroupConfig::new(4).unwrap();
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        cooperative_scan(&mut data, &cfg).unwrap();
        // Block 0: [1, 3, 6, 10], Block 1: [5, 11]
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[3] - 10.0).abs() < 1e-6);
        assert!((data[4] - 5.0).abs() < 1e-6);
        assert!((data[5] - 11.0).abs() < 1e-6);
    }

    // ===== cooperative_broadcast tests =====

    #[test]
    fn test_broadcast_from_thread_0() {
        let cfg = CooperativeGroupConfig::new(4).unwrap();
        let mut data = vec![10.0, 20.0, 30.0, 40.0];
        cooperative_broadcast(&mut data, 0, &cfg).unwrap();
        assert!(data.iter().all(|&v| (v - 10.0).abs() < 1e-6));
    }

    #[test]
    fn test_broadcast_from_thread_2() {
        let cfg = CooperativeGroupConfig::new(4).unwrap();
        let mut data = vec![10.0, 20.0, 30.0, 40.0];
        cooperative_broadcast(&mut data, 2, &cfg).unwrap();
        assert!(data.iter().all(|&v| (v - 30.0).abs() < 1e-6));
    }

    #[test]
    fn test_broadcast_multi_block() {
        let cfg = CooperativeGroupConfig::new(2).unwrap();
        let mut data = vec![10.0, 20.0, 30.0, 40.0];
        cooperative_broadcast(&mut data, 0, &cfg).unwrap();
        // Block 0: [10, 10], Block 1: [30, 30]
        assert!((data[0] - 10.0).abs() < 1e-6);
        assert!((data[1] - 10.0).abs() < 1e-6);
        assert!((data[2] - 30.0).abs() < 1e-6);
        assert!((data[3] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_broadcast_empty_errors() {
        let cfg = default_config();
        let mut data: Vec<f32> = vec![];
        assert!(cooperative_broadcast(&mut data, 0, &cfg).is_err());
    }

    #[test]
    fn test_broadcast_src_out_of_range_errors() {
        let cfg = CooperativeGroupConfig::new(4).unwrap();
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(cooperative_broadcast(&mut data, 4, &cfg).is_err());
    }

    #[test]
    fn test_broadcast_single_element() {
        let cfg = CooperativeGroupConfig::new(4).unwrap();
        let mut data = vec![42.0];
        cooperative_broadcast(&mut data, 0, &cfg).unwrap();
        assert!((data[0] - 42.0).abs() < 1e-6);
    }

    // ===== cooperative_sort tests =====

    #[test]
    fn test_sort_basic() {
        let cfg = default_config();
        let mut data = vec![4.0, 2.0, 7.0, 1.0, 3.0];
        cooperative_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 7.0]);
    }

    #[test]
    fn test_sort_already_sorted() {
        let cfg = default_config();
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        cooperative_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sort_reverse() {
        let cfg = default_config();
        let mut data = vec![4.0, 3.0, 2.0, 1.0];
        cooperative_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sort_single_element() {
        let cfg = default_config();
        let mut data = vec![42.0];
        cooperative_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![42.0]);
    }

    #[test]
    fn test_sort_two_elements() {
        let cfg = default_config();
        let mut data = vec![5.0, 2.0];
        cooperative_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![2.0, 5.0]);
    }

    #[test]
    fn test_sort_power_of_two_size() {
        let cfg = default_config();
        let mut data = vec![8.0, 4.0, 2.0, 6.0, 1.0, 3.0, 7.0, 5.0];
        cooperative_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_sort_non_power_of_two_size_3() {
        let cfg = default_config();
        let mut data = vec![3.0, 1.0, 2.0];
        cooperative_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sort_non_power_of_two_size_5() {
        let cfg = default_config();
        let mut data = vec![5.0, 3.0, 1.0, 4.0, 2.0];
        cooperative_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sort_non_power_of_two_size_7() {
        let cfg = default_config();
        let mut data = vec![7.0, 2.0, 5.0, 1.0, 6.0, 3.0, 4.0];
        cooperative_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_sort_duplicates() {
        let cfg = default_config();
        let mut data = vec![3.0, 1.0, 3.0, 1.0, 2.0];
        cooperative_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![1.0, 1.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_sort_all_same() {
        let cfg = default_config();
        let mut data = vec![5.0; 6];
        cooperative_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![5.0; 6]);
    }

    #[test]
    fn test_sort_empty_errors() {
        let cfg = default_config();
        let mut data: Vec<f32> = vec![];
        assert!(cooperative_sort(&mut data, &cfg).is_err());
    }

    #[test]
    fn test_sort_negative_values() {
        let cfg = default_config();
        let mut data = vec![-1.0, -5.0, -3.0, -2.0];
        cooperative_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![-5.0, -3.0, -2.0, -1.0]);
    }

    #[test]
    fn test_sort_mixed_sign() {
        let cfg = default_config();
        let mut data = vec![-2.0, 3.0, -1.0, 0.0, 2.0];
        cooperative_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![-2.0, -1.0, 0.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sort_large_power_of_two() {
        let cfg = default_config();
        let mut data: Vec<f32> = (0..16).rev().map(|x| x as f32).collect();
        cooperative_sort(&mut data, &cfg).unwrap();
        let expected: Vec<f32> = (0..16).map(|x| x as f32).collect();
        assert_eq!(data, expected);
    }

    // ===== cooperative_histogram tests =====

    #[test]
    fn test_histogram_basic() {
        let cfg = default_config();
        let input = vec![0, 1, 2, 0, 1, 0];
        let hist = cooperative_histogram(&input, 3, &cfg).unwrap();
        assert_eq!(hist, vec![3, 2, 1]);
    }

    #[test]
    fn test_histogram_all_same() {
        let cfg = default_config();
        let input = vec![2, 2, 2, 2];
        let hist = cooperative_histogram(&input, 4, &cfg).unwrap();
        assert_eq!(hist, vec![0, 0, 4, 0]);
    }

    #[test]
    fn test_histogram_out_of_range_ignored() {
        let cfg = default_config();
        let input = vec![0, 1, 5, 10, 0];
        let hist = cooperative_histogram(&input, 3, &cfg).unwrap();
        assert_eq!(hist, vec![2, 1, 0]); // 5 and 10 ignored
    }

    #[test]
    fn test_histogram_empty_input() {
        let cfg = default_config();
        let input: Vec<u32> = vec![];
        let hist = cooperative_histogram(&input, 4, &cfg).unwrap();
        assert_eq!(hist, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_histogram_zero_bins_errors() {
        let cfg = default_config();
        let input = vec![1, 2, 3];
        assert!(cooperative_histogram(&input, 0, &cfg).is_err());
    }

    #[test]
    fn test_histogram_single_bin() {
        let cfg = default_config();
        let input = vec![0, 0, 0];
        let hist = cooperative_histogram(&input, 1, &cfg).unwrap();
        assert_eq!(hist, vec![3]);
    }

    #[test]
    fn test_histogram_uniform_distribution() {
        let cfg = default_config();
        let input: Vec<u32> = (0..100).map(|x| x % 10).collect();
        let hist = cooperative_histogram(&input, 10, &cfg).unwrap();
        assert!(hist.iter().all(|&c| c == 10));
    }

    #[test]
    fn test_histogram_large_bins() {
        let cfg = default_config();
        let input = vec![999];
        let hist = cooperative_histogram(&input, 1000, &cfg).unwrap();
        assert_eq!(hist[999], 1);
        assert_eq!(hist.iter().sum::<u32>(), 1);
    }

    // ===== cooperative_matmul tests =====

    #[test]
    fn test_matmul_identity() {
        let cfg = default_config();
        // 2x2 identity * 2x2 matrix
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = cooperative_matmul(&a, &b, 2, 2, 2, &cfg).unwrap();
        assert!((c[0] - 5.0).abs() < 1e-6);
        assert!((c[1] - 6.0).abs() < 1e-6);
        assert!((c[2] - 7.0).abs() < 1e-6);
        assert!((c[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_2x2() {
        let cfg = default_config();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = cooperative_matmul(&a, &b, 2, 2, 2, &cfg).unwrap();
        // [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
        assert!((c[0] - 19.0).abs() < 1e-6);
        assert!((c[1] - 22.0).abs() < 1e-6);
        assert!((c[2] - 43.0).abs() < 1e-6);
        assert!((c[3] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_non_square() {
        let cfg = default_config();
        // 2x3 * 3x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = cooperative_matmul(&a, &b, 2, 2, 3, &cfg).unwrap();
        // [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
        // = [58, 64, 139, 154]
        assert!((c[0] - 58.0).abs() < 1e-6);
        assert!((c[1] - 64.0).abs() < 1e-6);
        assert!((c[2] - 139.0).abs() < 1e-6);
        assert!((c[3] - 154.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_1x1() {
        let cfg = default_config();
        let c = cooperative_matmul(&[3.0], &[4.0], 1, 1, 1, &cfg).unwrap();
        assert!((c[0] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_zero_result() {
        let cfg = default_config();
        let a = vec![0.0; 4];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = cooperative_matmul(&a, &b, 2, 2, 2, &cfg).unwrap();
        assert!(c.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn test_matmul_row_vector() {
        let cfg = default_config();
        // 1x3 * 3x1
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = cooperative_matmul(&a, &b, 1, 1, 3, &cfg).unwrap();
        assert!((c[0] - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_col_vector() {
        let cfg = default_config();
        // 3x1 * 1x3 = 3x3 outer product
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = cooperative_matmul(&a, &b, 3, 3, 1, &cfg).unwrap();
        let expected = vec![4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0];
        for (a, e) in c.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matmul_zero_m_errors() {
        let cfg = default_config();
        assert!(cooperative_matmul(&[], &[], 0, 1, 1, &cfg).is_err());
    }

    #[test]
    fn test_matmul_zero_n_errors() {
        let cfg = default_config();
        assert!(cooperative_matmul(&[], &[], 1, 0, 1, &cfg).is_err());
    }

    #[test]
    fn test_matmul_zero_k_errors() {
        let cfg = default_config();
        assert!(cooperative_matmul(&[], &[], 1, 1, 0, &cfg).is_err());
    }

    #[test]
    fn test_matmul_a_too_short_errors() {
        let cfg = default_config();
        assert!(cooperative_matmul(&[1.0], &[1.0, 2.0, 3.0, 4.0], 2, 2, 2, &cfg).is_err());
    }

    #[test]
    fn test_matmul_b_too_short_errors() {
        let cfg = default_config();
        assert!(cooperative_matmul(&[1.0, 2.0, 3.0, 4.0], &[1.0], 2, 2, 2, &cfg).is_err());
    }

    #[test]
    fn test_matmul_larger() {
        let cfg = default_config();
        // 4x4 identity
        let mut a = vec![0.0f32; 16];
        for i in 0..4 {
            a[i * 4 + i] = 1.0;
        }
        let b: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let c = cooperative_matmul(&a, &b, 4, 4, 4, &cfg).unwrap();
        for (cv, bv) in c.iter().zip(b.iter()) {
            assert!((cv - bv).abs() < 1e-6);
        }
    }

    // ===== CUDA kernel source compile gate test =====

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_kernel_src_not_empty() {
        assert!(!COOPERATIVE_GROUPS_KERNEL_SRC.is_empty());
        assert!(COOPERATIVE_GROUPS_KERNEL_SRC.contains("coop_reduce_sum_f32"));
        assert!(COOPERATIVE_GROUPS_KERNEL_SRC.contains("coop_matmul_f32"));
        assert!(COOPERATIVE_GROUPS_KERNEL_SRC.contains("cooperative_groups"));
    }

    // ===== ReduceOp enum tests =====

    #[test]
    fn test_reduce_op_debug() {
        let op = CooperativeReduceOp::Sum;
        assert_eq!(format!("{op:?}"), "Sum");
    }

    #[test]
    fn test_reduce_op_clone() {
        let op = CooperativeReduceOp::Max;
        let op2 = op;
        assert_eq!(op, op2);
    }

    #[test]
    fn test_reduce_op_all_variants() {
        let ops = [
            CooperativeReduceOp::Sum,
            CooperativeReduceOp::Max,
            CooperativeReduceOp::Min,
            CooperativeReduceOp::Product,
        ];
        assert_eq!(ops.len(), 4);
    }

    // ===== Edge case and stress tests =====

    #[test]
    fn test_config_single_thread_group() {
        let cfg = CooperativeGroupConfig::new(1).unwrap();
        let input = vec![42.0];
        let result = cooperative_reduce(&input, CooperativeReduceOp::Sum, &cfg).unwrap();
        assert!((result[0] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_scan_large_input() {
        let cfg = CooperativeGroupConfig::new(256).unwrap();
        let mut data: Vec<f32> = vec![1.0; 256];
        cooperative_scan(&mut data, &cfg).unwrap();
        for (i, val) in data.iter().enumerate() {
            assert!((val - (i + 1) as f32).abs() < 1e-4, "scan[{i}] = {val}, expected {}", i + 1);
        }
    }

    #[test]
    fn test_reduce_sum_matches_scan_last() {
        let cfg = CooperativeGroupConfig::new(8).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let reduce_result = cooperative_reduce(&input, CooperativeReduceOp::Sum, &cfg).unwrap();
        let mut scan_data = input;
        cooperative_scan(&mut scan_data, &cfg).unwrap();
        assert!((reduce_result[0] - scan_data[7]).abs() < 1e-6);
    }

    #[test]
    fn test_grid_group_single_block() {
        let gg = GridGroup::new(256, 1).unwrap();
        assert_eq!(gg.total_threads(), 256);
        assert_eq!(gg.block_of(0), 0);
        assert_eq!(gg.block_of(255), 0);
    }

    #[test]
    fn test_coalesced_group_first_half() {
        let cg = CoalescedGroup::new(0x0000_FFFF).unwrap();
        assert_eq!(cg.size(), 16);
        assert!(cg.is_active(0));
        assert!(cg.is_active(15));
        assert!(!cg.is_active(16));
    }

    #[test]
    fn test_coalesced_group_last_half() {
        let cg = CoalescedGroup::new(0xFFFF_0000).unwrap();
        assert_eq!(cg.size(), 16);
        assert!(!cg.is_active(0));
        assert!(cg.is_active(16));
        assert!(cg.is_active(31));
    }

    #[test]
    fn test_sort_size_16_non_power_of_two_values() {
        let cfg = default_config();
        let mut data = vec![
            16.0, 8.0, 4.0, 12.0, 2.0, 14.0, 6.0, 10.0, 1.0, 9.0, 5.0, 13.0, 3.0, 15.0, 7.0, 11.0,
        ];
        cooperative_sort(&mut data, &cfg).unwrap();
        let expected: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        assert_eq!(data, expected);
    }

    #[test]
    fn test_broadcast_preserves_other_blocks() {
        let cfg = CooperativeGroupConfig::new(3).unwrap();
        let mut data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        cooperative_broadcast(&mut data, 1, &cfg).unwrap();
        // Block 0: all become 20.0, Block 1: all become 50.0
        assert!((data[0] - 20.0).abs() < 1e-6);
        assert!((data[1] - 20.0).abs() < 1e-6);
        assert!((data[2] - 20.0).abs() < 1e-6);
        assert!((data[3] - 50.0).abs() < 1e-6);
        assert!((data[4] - 50.0).abs() < 1e-6);
        assert!((data[5] - 50.0).abs() < 1e-6);
    }
}
