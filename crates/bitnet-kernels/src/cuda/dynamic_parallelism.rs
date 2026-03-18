//! CUDA dynamic parallelism — child kernel launches from within kernels.
//!
//! # Overview
//!
//! CUDA dynamic parallelism allows a running GPU kernel (the *parent*) to
//! launch new kernels (*children*) without returning to the host. This module
//! provides:
//!
//! - [`DynamicLaunchConfig`] — launch envelope (max nesting depth, shared
//!   memory budget, stream priority, synchronisation policy)
//! - [`ChildKernelDescriptor`] — specification for a nested kernel launch
//! - [`DynamicParallelismManager`] — bookkeeping for parent→child
//!   relationships, depth tracking, and resource accounting
//!
//! # Operations
//!
//! - [`launch_child_kernel`] — enqueue a child kernel from within a parent
//! - [`synchronize_children`] — barrier-wait on all outstanding children
//! - [`dynamic_reduce`] — hierarchical parallel reduction via child kernels
//! - [`dynamic_scan`] — hierarchical exclusive prefix scan
//! - [`recursive_merge_sort`] — recursive merge-sort with child kernel launches
//! - [`adaptive_grid_launch`] — data-driven grid sizing with child splits
//! - [`nested_matmul`] — block-recursive matrix multiplication
//!
//! # GPU / CPU duality
//!
//! All CUDA kernel source constants and GPU launch stubs are feature-gated
//! behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`. Every operation
//! has a pure-Rust CPU fallback that mirrors the algorithmic behaviour so
//! that tests pass on non-GPU hosts.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use bitnet_common::{KernelError, Result};

// ── Identifiers ──────────────────────────────────────────────────────

static NEXT_DESCRIPTOR_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for a child kernel descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorId(u64);

impl DescriptorId {
    fn next() -> Self {
        Self(NEXT_DESCRIPTOR_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for DescriptorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "child-{}", self.0)
    }
}

// ── Constants ────────────────────────────────────────────────────────

/// Maximum supported nesting depth for dynamic parallelism.
pub const MAX_NESTING_DEPTH: u32 = 24;

/// Default shared-memory budget per child block (bytes).
pub const DEFAULT_SHARED_MEM_PER_BLOCK: usize = 48 * 1024;

/// Default stream priority for child kernels (0 = default).
pub const DEFAULT_STREAM_PRIORITY: i32 = 0;

/// Threshold below which a recursive algorithm switches to sequential CPU work.
pub const SEQUENTIAL_THRESHOLD: usize = 1024;

// ── Synchronisation policy ───────────────────────────────────────────

/// How the parent waits for child kernels to complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncPolicy {
    /// Implicit synchronisation at parent kernel exit.
    #[default]
    Implicit,
    /// Explicit `cudaDeviceSynchronize()` inside the parent.
    Explicit,
    /// Stream-ordered synchronisation (most efficient).
    StreamOrdered,
}

// ── DynamicLaunchConfig ──────────────────────────────────────────────

/// Configuration envelope for dynamic-parallelism launches.
#[derive(Debug, Clone)]
pub struct DynamicLaunchConfig {
    /// Maximum kernel nesting depth (1 = one level of children).
    pub max_depth: u32,
    /// Per-block shared memory budget in bytes.
    pub shared_mem_per_block: usize,
    /// CUDA stream priority for child kernels.
    pub stream_priority: i32,
    /// Synchronisation policy between parent and children.
    pub sync_policy: SyncPolicy,
    /// Maximum number of concurrent child kernels.
    pub max_pending_children: u32,
    /// Minimum work items before spawning a child (avoids launch overhead).
    pub min_items_per_child: usize,
    /// Whether to enable device-side memory allocation in children.
    pub device_malloc_enabled: bool,
}

impl Default for DynamicLaunchConfig {
    fn default() -> Self {
        Self {
            max_depth: 4,
            shared_mem_per_block: DEFAULT_SHARED_MEM_PER_BLOCK,
            stream_priority: DEFAULT_STREAM_PRIORITY,
            sync_policy: SyncPolicy::default(),
            max_pending_children: 1024,
            min_items_per_child: SEQUENTIAL_THRESHOLD,
            device_malloc_enabled: false,
        }
    }
}

impl DynamicLaunchConfig {
    /// Create a new config with the given maximum nesting depth.
    pub fn new(max_depth: u32) -> Result<Self> {
        if max_depth == 0 || max_depth > MAX_NESTING_DEPTH {
            return Err(KernelError::InvalidArguments {
                reason: format!("max_depth must be in 1..={MAX_NESTING_DEPTH}, got {max_depth}"),
            }
            .into());
        }
        Ok(Self { max_depth, ..Default::default() })
    }

    /// Builder: set shared memory per block.
    pub fn with_shared_mem(mut self, bytes: usize) -> Self {
        self.shared_mem_per_block = bytes;
        self
    }

    /// Builder: set stream priority.
    pub fn with_stream_priority(mut self, priority: i32) -> Self {
        self.stream_priority = priority;
        self
    }

    /// Builder: set synchronisation policy.
    pub fn with_sync_policy(mut self, policy: SyncPolicy) -> Self {
        self.sync_policy = policy;
        self
    }

    /// Builder: set minimum items per child.
    pub fn with_min_items(mut self, n: usize) -> Self {
        self.min_items_per_child = n;
        self
    }

    /// Validate that configuration is internally consistent.
    pub fn validate(&self) -> Result<()> {
        if self.max_depth == 0 || self.max_depth > MAX_NESTING_DEPTH {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "max_depth must be in 1..={MAX_NESTING_DEPTH}, got {}",
                    self.max_depth
                ),
            }
            .into());
        }
        if self.shared_mem_per_block > 163_840 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "shared_mem_per_block exceeds 160 KiB limit: {}",
                    self.shared_mem_per_block
                ),
            }
            .into());
        }
        Ok(())
    }
}

// ── ChildKernelDescriptor ────────────────────────────────────────────

/// Specification for a child kernel that will be launched from a parent.
#[derive(Debug, Clone)]
pub struct ChildKernelDescriptor {
    /// Unique identifier.
    pub id: DescriptorId,
    /// Human-readable kernel name.
    pub name: String,
    /// Grid dimensions `[x, y, z]`.
    pub grid: [u32; 3],
    /// Block dimensions `[x, y, z]`.
    pub block: [u32; 3],
    /// Per-block dynamic shared memory (bytes).
    pub shared_mem_bytes: usize,
    /// Nesting depth at which this child runs (1 = first child level).
    pub depth: u32,
    /// Index range `[start, end)` into the parent data this child processes.
    pub work_range: (usize, usize),
}

impl ChildKernelDescriptor {
    /// Create a new descriptor with default 1-D launch geometry.
    pub fn new(name: impl Into<String>, n: usize) -> Self {
        let block_size = 256u32;
        let grid_x = (n as u32).div_ceil(block_size);
        Self {
            id: DescriptorId::next(),
            name: name.into(),
            grid: [grid_x.max(1), 1, 1],
            block: [block_size, 1, 1],
            shared_mem_bytes: 0,
            depth: 1,
            work_range: (0, n),
        }
    }

    /// Builder: override grid dimensions.
    pub fn with_grid(mut self, grid: [u32; 3]) -> Self {
        self.grid = grid;
        self
    }

    /// Builder: override block dimensions.
    pub fn with_block(mut self, block: [u32; 3]) -> Self {
        self.block = block;
        self
    }

    /// Builder: override depth.
    pub fn with_depth(mut self, depth: u32) -> Self {
        self.depth = depth;
        self
    }

    /// Builder: override work range.
    pub fn with_work_range(mut self, start: usize, end: usize) -> Self {
        self.work_range = (start, end);
        self
    }

    /// Total threads launched by this child.
    pub fn total_threads(&self) -> u64 {
        let g = self.grid;
        let b = self.block;
        u64::from(g[0])
            * u64::from(g[1])
            * u64::from(g[2])
            * u64::from(b[0])
            * u64::from(b[1])
            * u64::from(b[2])
    }
}

impl fmt::Display for ChildKernelDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(grid={:?}, block={:?}, depth={}, range={:?})",
            self.name, self.grid, self.block, self.depth, self.work_range
        )
    }
}

// ── DynamicParallelismManager ────────────────────────────────────────

/// Tracks parent–child kernel relationships and resource consumption.
#[derive(Debug)]
pub struct DynamicParallelismManager {
    /// Launch configuration.
    config: DynamicLaunchConfig,
    /// Descriptors for all children launched so far.
    children: Vec<ChildKernelDescriptor>,
    /// Current nesting depth.
    current_depth: u32,
    /// Total shared-memory bytes allocated across all active children.
    total_shared_mem: usize,
    /// Whether all children have been synchronised.
    synchronised: bool,
}

impl DynamicParallelismManager {
    /// Create a new manager with the given configuration.
    pub fn new(config: DynamicLaunchConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            children: Vec::new(),
            current_depth: 0,
            total_shared_mem: 0,
            synchronised: true,
        })
    }

    /// Number of children launched so far.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Current nesting depth.
    pub fn current_depth(&self) -> u32 {
        self.current_depth
    }

    /// Whether all children have been synchronised.
    pub fn is_synchronised(&self) -> bool {
        self.synchronised
    }

    /// Maximum allowed depth.
    pub fn max_depth(&self) -> u32 {
        self.config.max_depth
    }

    /// Reference to the launch configuration.
    pub fn config(&self) -> &DynamicLaunchConfig {
        &self.config
    }

    /// Register a child kernel launch.
    pub fn register_child(&mut self, desc: ChildKernelDescriptor) -> Result<()> {
        if desc.depth > self.config.max_depth {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "child depth {} exceeds max_depth {}",
                    desc.depth, self.config.max_depth
                ),
            }
            .into());
        }
        if self.children.len() as u32 >= self.config.max_pending_children {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "max pending children ({}) reached",
                    self.config.max_pending_children
                ),
            }
            .into());
        }
        self.total_shared_mem += desc.shared_mem_bytes;
        if desc.depth > self.current_depth {
            self.current_depth = desc.depth;
        }
        self.synchronised = false;
        self.children.push(desc);
        Ok(())
    }

    /// Mark all children as synchronised and reset pending list.
    pub fn synchronise(&mut self) {
        self.synchronised = true;
        self.children.clear();
        self.total_shared_mem = 0;
    }

    /// Total shared memory consumed by pending children.
    pub fn total_shared_mem(&self) -> usize {
        self.total_shared_mem
    }

    /// Return immutable slice of pending child descriptors.
    pub fn pending_children(&self) -> &[ChildKernelDescriptor] {
        &self.children
    }
}

// ── CUDA kernel source ───────────────────────────────────────────────

/// CUDA C kernel source for dynamic-parallelism primitives.
///
/// Includes parent kernels that launch child grids for reduction, scan,
/// merge-sort, adaptive grid sizing, and nested matrix multiplication.
/// Requires `-rdc=true` (relocatable device code) at compile time.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const DYNAMIC_PARALLELISM_KERNEL_SRC: &str = r#"
// ---------- child-launch helper ----------
extern "C" __device__ void launch_child(
    void (*kernel)(float*, float*, int),
    float* in, float* out, int n,
    int grid, int block, int smem, cudaStream_t stream)
{
    kernel<<<grid, block, smem, stream>>>(in, out, n);
}

// ---------- dynamic reduce ----------
// Parent kernel: each block reduces its tile, then block-0 launches a child
// to reduce partial results.
extern "C" __global__ void dynamic_reduce_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int n)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = smem[0];

    __syncthreads();
    if (tid == 0 && blockIdx.x == 0 && gridDim.x > 1) {
        dynamic_reduce_kernel<<<1, blockDim.x, blockDim.x * sizeof(float), 0>>>(
            output, output, gridDim.x);
        cudaDeviceSynchronize();
    }
}

// ---------- dynamic scan (Blelloch) ----------
extern "C" __global__ void dynamic_scan_kernel(
    float* __restrict__ data,
    float* __restrict__ block_sums,
    int n)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    smem[tid] = (gid < n) ? data[gid] : 0.0f;
    __syncthreads();

    // up-sweep
    for (int d = 1; d < blockDim.x; d <<= 1) {
        int idx = (tid + 1) * (d << 1) - 1;
        if (idx < blockDim.x) smem[idx] += smem[idx - d];
        __syncthreads();
    }

    if (tid == blockDim.x - 1) {
        if (block_sums) block_sums[blockIdx.x] = smem[tid];
        smem[tid] = 0.0f;
    }
    __syncthreads();

    // down-sweep
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        int idx = (tid + 1) * (d << 1) - 1;
        if (idx < blockDim.x) {
            float tmp = smem[idx - d];
            smem[idx - d] = smem[idx];
            smem[idx] += tmp;
        }
        __syncthreads();
    }

    if (gid < n) data[gid] = smem[tid];
}

// ---------- recursive merge-sort ----------
extern "C" __global__ void merge_sort_kernel(
    float* __restrict__ data,
    float* __restrict__ tmp,
    int n,
    int depth)
{
    if (n <= 1) return;

    int mid = n / 2;
    if (depth > 0) {
        cudaStream_t s1, s2;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

        merge_sort_kernel<<<1, 1, 0, s1>>>(data, tmp, mid, depth - 1);
        merge_sort_kernel<<<1, 1, 0, s2>>>(data + mid, tmp + mid, n - mid, depth - 1);
        cudaDeviceSynchronize();

        cudaStreamDestroy(s1);
        cudaStreamDestroy(s2);
    }

    // merge
    int i = 0, j = mid, k = 0;
    while (i < mid && j < n) {
        tmp[k++] = (data[i] <= data[j]) ? data[i++] : data[j++];
    }
    while (i < mid) tmp[k++] = data[i++];
    while (j < n)   tmp[k++] = data[j++];
    for (int x = 0; x < n; ++x) data[x] = tmp[x];
}

// ---------- adaptive grid launch ----------
extern "C" __global__ void adaptive_grid_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int n,
    int threshold)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        output[gid] = input[gid] * input[gid]; // placeholder op
    }

    // If work is large enough, split into children
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0 && n > threshold) {
        int half = n / 2;
        int blk = 256;
        int g1 = (half + blk - 1) / blk;
        int g2 = ((n - half) + blk - 1) / blk;
        adaptive_grid_kernel<<<g1, blk, 0, 0>>>(input, output, half, threshold);
        adaptive_grid_kernel<<<g2, blk, 0, 0>>>(input + half, output + half, n - half, threshold);
        cudaDeviceSynchronize();
    }
}

// ---------- nested matmul ----------
// Recursively splits C = A * B into quadrants.
extern "C" __global__ void nested_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K, int lda, int ldb, int ldc, int depth)
{
    if (M <= 64 || N <= 64 || K <= 64 || depth <= 0) {
        // Base case: naive matmul
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * lda + k] * B[k * ldb + col];
            }
            C[row * ldc + col] += sum;
        }
        return;
    }

    // Recursive split
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        int mh = M / 2, nh = N / 2, kh = K / 2;
        dim3 blk(16, 16);
        dim3 g1((nh + 15) / 16, (mh + 15) / 16);
        dim3 g2(((N - nh) + 15) / 16, (mh + 15) / 16);
        dim3 g3((nh + 15) / 16, ((M - mh) + 15) / 16);
        dim3 g4(((N - nh) + 15) / 16, ((M - mh) + 15) / 16);

        // C11 += A11 * B11
        nested_matmul_kernel<<<g1, blk>>>(A, B, C, mh, nh, kh, lda, ldb, ldc, depth - 1);
        // C12 += A11 * B12
        nested_matmul_kernel<<<g2, blk>>>(A, B + nh, C + nh, mh, N - nh, kh, lda, ldb, ldc, depth - 1);
        // C21 += A21 * B11
        nested_matmul_kernel<<<g3, blk>>>(A + mh * lda, B, C + mh * ldc, M - mh, nh, kh, lda, ldb, ldc, depth - 1);
        // C22 += A21 * B12
        nested_matmul_kernel<<<g4, blk>>>(A + mh * lda, B + nh, C + mh * ldc + nh, M - mh, N - nh, kh, lda, ldb, ldc, depth - 1);
        cudaDeviceSynchronize();

        // Second half of K dimension
        nested_matmul_kernel<<<g1, blk>>>(A + kh, B + kh * ldb, C, mh, nh, K - kh, lda, ldb, ldc, depth - 1);
        nested_matmul_kernel<<<g2, blk>>>(A + kh, B + kh * ldb + nh, C + nh, mh, N - nh, K - kh, lda, ldb, ldc, depth - 1);
        nested_matmul_kernel<<<g3, blk>>>(A + mh * lda + kh, B + kh * ldb, C + mh * ldc, M - mh, nh, K - kh, lda, ldb, ldc, depth - 1);
        nested_matmul_kernel<<<g4, blk>>>(A + mh * lda + kh, B + kh * ldb + nh, C + mh * ldc + nh, M - mh, N - nh, K - kh, lda, ldb, ldc, depth - 1);
        cudaDeviceSynchronize();
    }
}
"#;

// ── GPU launch stubs ─────────────────────────────────────────────────

/// Launch the dynamic-reduce kernel on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_dynamic_reduce(
    input: &[f32],
    output: &mut [f32],
    config: &DynamicLaunchConfig,
) -> Result<()> {
    let _ = (input, output, config);
    Err(KernelError::GpuError { reason: "CUDA runtime not linked".into() }.into())
}

/// Launch the dynamic-scan kernel on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_dynamic_scan(data: &mut [f32], config: &DynamicLaunchConfig) -> Result<()> {
    let _ = (data, config);
    Err(KernelError::GpuError { reason: "CUDA runtime not linked".into() }.into())
}

/// Launch the recursive merge-sort kernel on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_recursive_merge_sort(data: &mut [f32], config: &DynamicLaunchConfig) -> Result<()> {
    let _ = (data, config);
    Err(KernelError::GpuError { reason: "CUDA runtime not linked".into() }.into())
}

/// Launch the adaptive-grid kernel on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_adaptive_grid(
    input: &[f32],
    output: &mut [f32],
    config: &DynamicLaunchConfig,
) -> Result<()> {
    let _ = (input, output, config);
    Err(KernelError::GpuError { reason: "CUDA runtime not linked".into() }.into())
}

/// Launch the nested-matmul kernel on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_nested_matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    config: &DynamicLaunchConfig,
) -> Result<()> {
    let _ = (a, b, c, m, n, k, config);
    Err(KernelError::GpuError { reason: "CUDA runtime not linked".into() }.into())
}

// ── CPU fallback: launch_child_kernel ────────────────────────────────

/// Simulate launching a child kernel on the CPU.
///
/// On GPU this would enqueue a child grid from within a parent kernel.
/// The CPU fallback simply applies `op` to the slice defined by the
/// descriptor's `work_range`.
pub fn launch_child_kernel(
    data: &mut [f32],
    desc: &ChildKernelDescriptor,
    manager: &mut DynamicParallelismManager,
    op: fn(&mut [f32]),
) -> Result<()> {
    if desc.depth > manager.max_depth() {
        return Err(KernelError::InvalidArguments {
            reason: format!("child depth {} exceeds max_depth {}", desc.depth, manager.max_depth()),
        }
        .into());
    }
    let (start, end) = desc.work_range;
    if end > data.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("work_range ({start}, {end}) exceeds data length {}", data.len()),
        }
        .into());
    }
    manager.register_child(desc.clone())?;
    op(&mut data[start..end]);
    Ok(())
}

// ── CPU fallback: synchronize_children ───────────────────────────────

/// Synchronise all outstanding child kernels (CPU no-op / bookkeeping reset).
pub fn synchronize_children(manager: &mut DynamicParallelismManager) -> Result<()> {
    manager.synchronise();
    Ok(())
}

// ── CPU fallback: dynamic_reduce ─────────────────────────────────────

/// Hierarchical parallel reduction via recursive child kernel launches (CPU).
///
/// Computes the sum of `input` using a tree of reductions that mirrors
/// the GPU dynamic-parallelism pattern.
pub fn dynamic_reduce(input: &[f32], config: &DynamicLaunchConfig) -> Result<f32> {
    config.validate()?;
    if input.is_empty() {
        return Ok(0.0);
    }
    Ok(dynamic_reduce_recursive(input, config.max_depth))
}

fn dynamic_reduce_recursive(data: &[f32], depth_remaining: u32) -> f32 {
    if data.len() <= SEQUENTIAL_THRESHOLD || depth_remaining == 0 {
        return data.iter().sum();
    }
    let mid = data.len() / 2;
    let left = dynamic_reduce_recursive(&data[..mid], depth_remaining - 1);
    let right = dynamic_reduce_recursive(&data[mid..], depth_remaining - 1);
    left + right
}

/// Unified dispatch for dynamic reduction.
pub fn dynamic_reduce_forward(input: &[f32], config: &DynamicLaunchConfig) -> Result<f32> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let mut output = [0.0f32; 1];
        if crate::device_features::gpu_available_runtime()
            && launch_dynamic_reduce(input, &mut output, config).is_ok()
        {
            return Ok(output[0]);
        }
    }
    dynamic_reduce(input, config)
}

// ── CPU fallback: dynamic_scan ───────────────────────────────────────

/// Hierarchical exclusive prefix scan via recursive child launches (CPU).
///
/// After this call, `data[i]` contains the sum of all elements before
/// index `i` in the original array (exclusive scan).
pub fn dynamic_scan(data: &mut [f32], config: &DynamicLaunchConfig) -> Result<()> {
    config.validate()?;
    if data.is_empty() {
        return Ok(());
    }
    dynamic_scan_recursive(data, config.max_depth);
    Ok(())
}

fn dynamic_scan_recursive(data: &mut [f32], depth_remaining: u32) {
    let n = data.len();
    if n <= 1 {
        if n == 1 {
            data[0] = 0.0;
        }
        return;
    }
    if n <= SEQUENTIAL_THRESHOLD || depth_remaining == 0 {
        // Sequential exclusive scan.
        let mut acc = 0.0f32;
        for v in data.iter_mut() {
            let cur = *v;
            *v = acc;
            acc += cur;
        }
        return;
    }

    // Block-level scan: compute block sums, scan them recursively, then
    // propagate offsets. This mirrors the GPU three-phase scan.
    let block_size = SEQUENTIAL_THRESHOLD;
    let num_blocks = n.div_ceil(block_size);

    // Phase 1: inclusive scan each block, collect block totals.
    let mut block_sums = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(n);
        let mut sum = 0.0f32;
        for v in &data[start..end] {
            sum += *v;
        }
        block_sums.push(sum);
    }

    // Phase 2: exclusive-scan block sums recursively.
    dynamic_scan_recursive(&mut block_sums, depth_remaining - 1);

    // Phase 3: sequential exclusive scan within each block, offset by
    // the block's prefix.
    for (b, &prefix) in block_sums.iter().enumerate().take(num_blocks) {
        let start = b * block_size;
        let end = (start + block_size).min(n);
        let mut acc = prefix;
        for v in &mut data[start..end] {
            let cur = *v;
            *v = acc;
            acc += cur;
        }
    }
}

/// Unified dispatch for dynamic scan.
pub fn dynamic_scan_forward(data: &mut [f32], config: &DynamicLaunchConfig) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && launch_dynamic_scan(data, config).is_ok()
        {
            return Ok(());
        }
    }
    dynamic_scan(data, config)
}

// ── CPU fallback: recursive_merge_sort ───────────────────────────────

/// Recursive merge-sort mirroring the GPU dynamic-parallelism pattern.
pub fn recursive_merge_sort(data: &mut [f32], config: &DynamicLaunchConfig) -> Result<()> {
    config.validate()?;
    if data.len() <= 1 {
        return Ok(());
    }
    let mut tmp = vec![0.0f32; data.len()];
    merge_sort_recursive(data, &mut tmp, config.max_depth);
    Ok(())
}

fn merge_sort_recursive(data: &mut [f32], tmp: &mut [f32], depth: u32) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    if n <= SEQUENTIAL_THRESHOLD || depth == 0 {
        // Insertion sort for small/base case.
        for i in 1..n {
            let key = data[i];
            let mut j = i;
            while j > 0 && data[j - 1] > key {
                data[j] = data[j - 1];
                j -= 1;
            }
            data[j] = key;
        }
        return;
    }
    let mid = n / 2;
    merge_sort_recursive(&mut data[..mid], &mut tmp[..mid], depth - 1);
    merge_sort_recursive(&mut data[mid..], &mut tmp[mid..], depth - 1);

    // Merge.
    let (mut i, mut j, mut k) = (0, mid, 0);
    while i < mid && j < n {
        if data[i] <= data[j] {
            tmp[k] = data[i];
            i += 1;
        } else {
            tmp[k] = data[j];
            j += 1;
        }
        k += 1;
    }
    while i < mid {
        tmp[k] = data[i];
        i += 1;
        k += 1;
    }
    while j < n {
        tmp[k] = data[j];
        j += 1;
        k += 1;
    }
    data[..n].copy_from_slice(&tmp[..n]);
}

/// Unified dispatch for recursive merge-sort.
pub fn recursive_merge_sort_forward(data: &mut [f32], config: &DynamicLaunchConfig) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && launch_recursive_merge_sort(data, config).is_ok()
        {
            return Ok(());
        }
    }
    recursive_merge_sort(data, config)
}

// ── CPU fallback: adaptive_grid_launch ───────────────────────────────

/// Adaptive grid launch: applies `f(x) = x * x` with recursive splitting
/// when the input size exceeds the configured threshold.
///
/// On GPU the kernel would dynamically launch child grids for large inputs.
pub fn adaptive_grid_launch(
    input: &[f32],
    output: &mut [f32],
    config: &DynamicLaunchConfig,
) -> Result<()> {
    config.validate()?;
    let n = input.len();
    if output.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < input length {n}", output.len()),
        }
        .into());
    }
    adaptive_grid_recursive(input, output, config.min_items_per_child, config.max_depth);
    Ok(())
}

fn adaptive_grid_recursive(input: &[f32], output: &mut [f32], threshold: usize, depth: u32) {
    let n = input.len();
    if n <= threshold || depth == 0 {
        for (o, &v) in output.iter_mut().zip(input.iter()) {
            *o = v * v;
        }
        return;
    }
    let mid = n / 2;
    adaptive_grid_recursive(&input[..mid], &mut output[..mid], threshold, depth - 1);
    adaptive_grid_recursive(&input[mid..], &mut output[mid..], threshold, depth - 1);
}

/// Unified dispatch for adaptive grid launch.
pub fn adaptive_grid_launch_forward(
    input: &[f32],
    output: &mut [f32],
    config: &DynamicLaunchConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && launch_adaptive_grid(input, output, config).is_ok()
        {
            return Ok(());
        }
    }
    adaptive_grid_launch(input, output, config)
}

// ── CPU fallback: nested_matmul ──────────────────────────────────────

/// Block-recursive matrix multiplication mirroring the GPU nested-matmul
/// dynamic-parallelism pattern.
///
/// Computes `C += A × B` where `A` is `m × k`, `B` is `k × n`, `C` is
/// `m × n`, all row-major.
pub fn nested_matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    config: &DynamicLaunchConfig,
) -> Result<()> {
    config.validate()?;
    if a.len() < m * k {
        return Err(KernelError::InvalidArguments {
            reason: format!("a.len() {} < m*k = {}", a.len(), m * k),
        }
        .into());
    }
    if b.len() < k * n {
        return Err(KernelError::InvalidArguments {
            reason: format!("b.len() {} < k*n = {}", b.len(), k * n),
        }
        .into());
    }
    if c.len() < m * n {
        return Err(KernelError::InvalidArguments {
            reason: format!("c.len() {} < m*n = {}", c.len(), m * n),
        }
        .into());
    }
    nested_matmul_recursive(a, b, c, m, n, k, k, n, n, config.max_depth);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn nested_matmul_recursive(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    depth: u32,
) {
    const BASE: usize = 64;
    if m <= BASE || n <= BASE || k <= BASE || depth == 0 {
        // Naive matmul base case.
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[row * lda + p] * b[p * ldb + col];
                }
                c[row * ldc + col] += sum;
            }
        }
        return;
    }

    let mh = m / 2;
    let nh = n / 2;
    let kh = k / 2;

    // C11 += A11 * B11
    nested_matmul_recursive(a, b, c, mh, nh, kh, lda, ldb, ldc, depth - 1);
    // C12 += A11 * B12
    nested_matmul_recursive(a, &b[nh..], &mut c[nh..], mh, n - nh, kh, lda, ldb, ldc, depth - 1);
    // C21 += A21 * B11
    nested_matmul_recursive(
        &a[mh * lda..],
        b,
        &mut c[mh * ldc..],
        m - mh,
        nh,
        kh,
        lda,
        ldb,
        ldc,
        depth - 1,
    );
    // C22 += A21 * B12
    nested_matmul_recursive(
        &a[mh * lda..],
        &b[nh..],
        &mut c[mh * ldc + nh..],
        m - mh,
        n - nh,
        kh,
        lda,
        ldb,
        ldc,
        depth - 1,
    );

    // Second half of K dimension.
    nested_matmul_recursive(&a[kh..], &b[kh * ldb..], c, mh, nh, k - kh, lda, ldb, ldc, depth - 1);
    nested_matmul_recursive(
        &a[kh..],
        &b[kh * ldb + nh..],
        &mut c[nh..],
        mh,
        n - nh,
        k - kh,
        lda,
        ldb,
        ldc,
        depth - 1,
    );
    nested_matmul_recursive(
        &a[mh * lda + kh..],
        &b[kh * ldb..],
        &mut c[mh * ldc..],
        m - mh,
        nh,
        k - kh,
        lda,
        ldb,
        ldc,
        depth - 1,
    );
    nested_matmul_recursive(
        &a[mh * lda + kh..],
        &b[kh * ldb + nh..],
        &mut c[mh * ldc + nh..],
        m - mh,
        n - nh,
        k - kh,
        lda,
        ldb,
        ldc,
        depth - 1,
    );
}

/// Unified dispatch for nested matmul.
pub fn nested_matmul_forward(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    config: &DynamicLaunchConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && launch_nested_matmul(a, b, c, m, n, k, config).is_ok()
        {
            return Ok(());
        }
    }
    nested_matmul(a, b, c, m, n, k, config)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────

    fn default_config() -> DynamicLaunchConfig {
        DynamicLaunchConfig::default()
    }

    fn config_depth(d: u32) -> DynamicLaunchConfig {
        DynamicLaunchConfig::new(d).unwrap()
    }

    fn ascending(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i + 1) as f32).collect()
    }

    fn descending(n: usize) -> Vec<f32> {
        (0..n).rev().map(|i| (i + 1) as f32).collect()
    }

    fn constant(n: usize, v: f32) -> Vec<f32> {
        vec![v; n]
    }

    fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for r in 0..m {
            for c_col in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[r * k + p] * b[p * n + c_col];
                }
                c[r * n + c_col] = s;
            }
        }
        c
    }

    // ── DynamicLaunchConfig ─────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = default_config();
        assert_eq!(cfg.max_depth, 4);
        assert_eq!(cfg.shared_mem_per_block, DEFAULT_SHARED_MEM_PER_BLOCK);
        assert_eq!(cfg.stream_priority, 0);
        assert_eq!(cfg.sync_policy, SyncPolicy::Implicit);
        assert_eq!(cfg.max_pending_children, 1024);
        assert!(!cfg.device_malloc_enabled);
    }

    #[test]
    fn test_config_new_valid() {
        let cfg = DynamicLaunchConfig::new(8).unwrap();
        assert_eq!(cfg.max_depth, 8);
    }

    #[test]
    fn test_config_new_max_valid() {
        let cfg = DynamicLaunchConfig::new(MAX_NESTING_DEPTH).unwrap();
        assert_eq!(cfg.max_depth, MAX_NESTING_DEPTH);
    }

    #[test]
    fn test_config_new_zero_depth_fails() {
        assert!(DynamicLaunchConfig::new(0).is_err());
    }

    #[test]
    fn test_config_new_exceeds_max_depth_fails() {
        assert!(DynamicLaunchConfig::new(MAX_NESTING_DEPTH + 1).is_err());
    }

    #[test]
    fn test_config_validate_ok() {
        default_config().validate().unwrap();
    }

    #[test]
    fn test_config_validate_excessive_shared_mem() {
        let mut cfg = default_config();
        cfg.shared_mem_per_block = 200_000;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_builder_chain() {
        let cfg = DynamicLaunchConfig::new(2)
            .unwrap()
            .with_shared_mem(1024)
            .with_stream_priority(-1)
            .with_sync_policy(SyncPolicy::Explicit)
            .with_min_items(512);
        assert_eq!(cfg.max_depth, 2);
        assert_eq!(cfg.shared_mem_per_block, 1024);
        assert_eq!(cfg.stream_priority, -1);
        assert_eq!(cfg.sync_policy, SyncPolicy::Explicit);
        assert_eq!(cfg.min_items_per_child, 512);
    }

    #[test]
    fn test_sync_policy_default_is_implicit() {
        assert_eq!(SyncPolicy::default(), SyncPolicy::Implicit);
    }

    #[test]
    fn test_config_stream_ordered_policy() {
        let cfg = DynamicLaunchConfig::new(1).unwrap().with_sync_policy(SyncPolicy::StreamOrdered);
        assert_eq!(cfg.sync_policy, SyncPolicy::StreamOrdered);
    }

    // ── ChildKernelDescriptor ───────────────────────────────────────

    #[test]
    fn test_descriptor_new_small() {
        let d = ChildKernelDescriptor::new("test_kern", 128);
        assert_eq!(d.name, "test_kern");
        assert_eq!(d.block, [256, 1, 1]);
        assert_eq!(d.grid, [1, 1, 1]);
        assert_eq!(d.depth, 1);
        assert_eq!(d.work_range, (0, 128));
    }

    #[test]
    fn test_descriptor_new_large() {
        let d = ChildKernelDescriptor::new("big", 10_000);
        assert_eq!(d.grid[0], (10_000 + 255) / 256);
        assert_eq!(d.work_range, (0, 10_000));
    }

    #[test]
    fn test_descriptor_new_zero() {
        let d = ChildKernelDescriptor::new("empty", 0);
        assert_eq!(d.grid, [1, 1, 1]);
        assert_eq!(d.work_range, (0, 0));
    }

    #[test]
    fn test_descriptor_builders() {
        let d = ChildKernelDescriptor::new("k", 100)
            .with_grid([4, 2, 1])
            .with_block([128, 1, 1])
            .with_depth(3)
            .with_work_range(10, 90);
        assert_eq!(d.grid, [4, 2, 1]);
        assert_eq!(d.block, [128, 1, 1]);
        assert_eq!(d.depth, 3);
        assert_eq!(d.work_range, (10, 90));
    }

    #[test]
    fn test_descriptor_total_threads() {
        let d = ChildKernelDescriptor::new("t", 256);
        assert_eq!(d.total_threads(), 256);
    }

    #[test]
    fn test_descriptor_total_threads_multidim() {
        let d = ChildKernelDescriptor::new("t", 1).with_grid([2, 3, 4]).with_block([8, 8, 2]);
        assert_eq!(d.total_threads(), 2 * 3 * 4 * 8 * 8 * 2);
    }

    #[test]
    fn test_descriptor_display() {
        let d = ChildKernelDescriptor::new("relu", 64);
        let s = format!("{d}");
        assert!(s.contains("relu"));
        assert!(s.contains("depth=1"));
    }

    #[test]
    fn test_descriptor_id_unique() {
        let d1 = ChildKernelDescriptor::new("a", 1);
        let d2 = ChildKernelDescriptor::new("b", 1);
        assert_ne!(d1.id, d2.id);
    }

    #[test]
    fn test_descriptor_id_display() {
        let d = ChildKernelDescriptor::new("x", 1);
        let s = format!("{}", d.id);
        assert!(s.starts_with("child-"));
    }

    // ── DynamicParallelismManager ───────────────────────────────────

    #[test]
    fn test_manager_new() {
        let mgr = DynamicParallelismManager::new(default_config()).unwrap();
        assert_eq!(mgr.child_count(), 0);
        assert_eq!(mgr.current_depth(), 0);
        assert!(mgr.is_synchronised());
    }

    #[test]
    fn test_manager_register_child() {
        let mut mgr = DynamicParallelismManager::new(default_config()).unwrap();
        let desc = ChildKernelDescriptor::new("c1", 64);
        mgr.register_child(desc).unwrap();
        assert_eq!(mgr.child_count(), 1);
        assert!(!mgr.is_synchronised());
    }

    #[test]
    fn test_manager_register_updates_depth() {
        let mut mgr = DynamicParallelismManager::new(default_config()).unwrap();
        mgr.register_child(ChildKernelDescriptor::new("a", 1).with_depth(2)).unwrap();
        assert_eq!(mgr.current_depth(), 2);
        mgr.register_child(ChildKernelDescriptor::new("b", 1).with_depth(3)).unwrap();
        assert_eq!(mgr.current_depth(), 3);
    }

    #[test]
    fn test_manager_register_exceeds_depth_fails() {
        let mut mgr = DynamicParallelismManager::new(config_depth(2)).unwrap();
        let desc = ChildKernelDescriptor::new("deep", 1).with_depth(3);
        assert!(mgr.register_child(desc).is_err());
    }

    #[test]
    fn test_manager_register_max_children_fails() {
        let mut cfg = default_config();
        cfg.max_pending_children = 2;
        let mut mgr = DynamicParallelismManager::new(cfg).unwrap();
        mgr.register_child(ChildKernelDescriptor::new("a", 1)).unwrap();
        mgr.register_child(ChildKernelDescriptor::new("b", 1)).unwrap();
        assert!(mgr.register_child(ChildKernelDescriptor::new("c", 1)).is_err());
    }

    #[test]
    fn test_manager_synchronise_resets() {
        let mut mgr = DynamicParallelismManager::new(default_config()).unwrap();
        mgr.register_child(ChildKernelDescriptor::new("c", 32)).unwrap();
        assert!(!mgr.is_synchronised());
        mgr.synchronise();
        assert!(mgr.is_synchronised());
        assert_eq!(mgr.child_count(), 0);
        assert_eq!(mgr.total_shared_mem(), 0);
    }

    #[test]
    fn test_manager_tracks_shared_mem() {
        let mut mgr = DynamicParallelismManager::new(default_config()).unwrap();
        let mut d = ChildKernelDescriptor::new("s", 1);
        d.shared_mem_bytes = 4096;
        mgr.register_child(d.clone()).unwrap();
        mgr.register_child(d).unwrap();
        assert_eq!(mgr.total_shared_mem(), 8192);
    }

    #[test]
    fn test_manager_pending_children_slice() {
        let mut mgr = DynamicParallelismManager::new(default_config()).unwrap();
        mgr.register_child(ChildKernelDescriptor::new("a", 1)).unwrap();
        mgr.register_child(ChildKernelDescriptor::new("b", 1)).unwrap();
        assert_eq!(mgr.pending_children().len(), 2);
    }

    #[test]
    fn test_manager_config_accessor() {
        let cfg = config_depth(6);
        let mgr = DynamicParallelismManager::new(cfg).unwrap();
        assert_eq!(mgr.config().max_depth, 6);
        assert_eq!(mgr.max_depth(), 6);
    }

    #[test]
    fn test_manager_invalid_config_fails() {
        let mut cfg = default_config();
        cfg.max_depth = 0;
        assert!(DynamicParallelismManager::new(cfg).is_err());
    }

    // ── launch_child_kernel ─────────────────────────────────────────

    #[test]
    fn test_launch_child_kernel_basic() {
        let cfg = default_config();
        let mut mgr = DynamicParallelismManager::new(cfg).unwrap();
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let desc = ChildKernelDescriptor::new("double", 4);
        launch_child_kernel(&mut data, &desc, &mut mgr, |s| {
            for v in s.iter_mut() {
                *v *= 2.0;
            }
        })
        .unwrap();
        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(mgr.child_count(), 1);
    }

    #[test]
    fn test_launch_child_kernel_sub_range() {
        let cfg = default_config();
        let mut mgr = DynamicParallelismManager::new(cfg).unwrap();
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let desc = ChildKernelDescriptor::new("negate", 3).with_work_range(1, 4);
        launch_child_kernel(&mut data, &desc, &mut mgr, |s| {
            for v in s.iter_mut() {
                *v = -*v;
            }
        })
        .unwrap();
        assert_eq!(data, vec![1.0, -2.0, -3.0, -4.0, 5.0]);
    }

    #[test]
    fn test_launch_child_kernel_depth_exceeded() {
        let cfg = config_depth(1);
        let mut mgr = DynamicParallelismManager::new(cfg).unwrap();
        let mut data = [1.0];
        let desc = ChildKernelDescriptor::new("bad", 1).with_depth(2);
        assert!(launch_child_kernel(&mut data, &desc, &mut mgr, |_| {}).is_err());
    }

    #[test]
    fn test_launch_child_kernel_range_oob() {
        let cfg = default_config();
        let mut mgr = DynamicParallelismManager::new(cfg).unwrap();
        let mut data = vec![1.0, 2.0];
        let desc = ChildKernelDescriptor::new("oob", 5).with_work_range(0, 5);
        assert!(launch_child_kernel(&mut data, &desc, &mut mgr, |_| {}).is_err());
    }

    #[test]
    fn test_launch_child_empty_range() {
        let cfg = default_config();
        let mut mgr = DynamicParallelismManager::new(cfg).unwrap();
        let mut data = vec![1.0, 2.0];
        let desc = ChildKernelDescriptor::new("noop", 0).with_work_range(1, 1);
        launch_child_kernel(&mut data, &desc, &mut mgr, |_| {}).unwrap();
        assert_eq!(data, vec![1.0, 2.0]);
    }

    // ── synchronize_children ────────────────────────────────────────

    #[test]
    fn test_synchronize_children_basic() {
        let cfg = default_config();
        let mut mgr = DynamicParallelismManager::new(cfg).unwrap();
        mgr.register_child(ChildKernelDescriptor::new("c", 1)).unwrap();
        assert!(!mgr.is_synchronised());
        synchronize_children(&mut mgr).unwrap();
        assert!(mgr.is_synchronised());
        assert_eq!(mgr.child_count(), 0);
    }

    #[test]
    fn test_synchronize_children_idempotent() {
        let cfg = default_config();
        let mut mgr = DynamicParallelismManager::new(cfg).unwrap();
        synchronize_children(&mut mgr).unwrap();
        synchronize_children(&mut mgr).unwrap();
        assert!(mgr.is_synchronised());
    }

    // ── dynamic_reduce ──────────────────────────────────────────────

    #[test]
    fn test_reduce_empty() {
        let cfg = default_config();
        assert_eq!(dynamic_reduce(&[], &cfg).unwrap(), 0.0);
    }

    #[test]
    fn test_reduce_single() {
        let cfg = default_config();
        assert_eq!(dynamic_reduce(&[42.0], &cfg).unwrap(), 42.0);
    }

    #[test]
    fn test_reduce_small() {
        let cfg = default_config();
        let data = ascending(10);
        let sum = dynamic_reduce(&data, &cfg).unwrap();
        assert!((sum - 55.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_large_recursive() {
        let cfg = config_depth(4);
        let n = 8192;
        let data = constant(n, 1.0);
        let sum = dynamic_reduce(&data, &cfg).unwrap();
        assert!((sum - n as f32).abs() < 1.0);
    }

    #[test]
    fn test_reduce_ascending() {
        let cfg = default_config();
        let data = ascending(100);
        let expected: f32 = (1..=100).map(|i| i as f32).sum();
        let sum = dynamic_reduce(&data, &cfg).unwrap();
        assert!((sum - expected).abs() < 1e-3);
    }

    #[test]
    fn test_reduce_negative_values() {
        let cfg = default_config();
        let data = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
        let sum = dynamic_reduce(&data, &cfg).unwrap();
        assert!((sum - (-15.0)).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_mixed_sign() {
        let cfg = default_config();
        let data = vec![1.0, -1.0, 2.0, -2.0, 3.0];
        let sum = dynamic_reduce(&data, &cfg).unwrap();
        assert!((sum - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_depth_1() {
        let cfg = config_depth(1);
        let data = ascending(2048);
        let expected: f32 = (1..=2048).map(|i| i as f32).sum();
        let sum = dynamic_reduce(&data, &cfg).unwrap();
        assert!((sum - expected).abs() / expected < 1e-4);
    }

    #[test]
    fn test_reduce_forward_dispatch() {
        let cfg = default_config();
        let data = vec![1.0, 2.0, 3.0];
        let sum = dynamic_reduce_forward(&data, &cfg).unwrap();
        assert!((sum - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_invalid_config() {
        let mut cfg = default_config();
        cfg.max_depth = 0;
        assert!(dynamic_reduce(&[1.0], &cfg).is_err());
    }

    // ── dynamic_scan ────────────────────────────────────────────────

    #[test]
    fn test_scan_empty() {
        let cfg = default_config();
        let mut data: Vec<f32> = vec![];
        dynamic_scan(&mut data, &cfg).unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn test_scan_single() {
        let cfg = default_config();
        let mut data = [5.0f32];
        dynamic_scan(&mut data, &cfg).unwrap();
        assert_eq!(data.to_vec(), vec![0.0]);
    }

    #[test]
    fn test_scan_small() {
        let cfg = default_config();
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        dynamic_scan(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![0.0, 1.0, 3.0, 6.0]);
    }

    #[test]
    fn test_scan_ones() {
        let cfg = default_config();
        let n = 16;
        let mut data = constant(n, 1.0);
        dynamic_scan(&mut data, &cfg).unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!((v - i as f32).abs() < 1e-5, "scan[{i}] = {v}, expected {i}");
        }
    }

    #[test]
    fn test_scan_large_recursive() {
        let cfg = config_depth(4);
        let n = 4096;
        let mut data = constant(n, 1.0);
        dynamic_scan(&mut data, &cfg).unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!((v - i as f32).abs() < 1.0, "scan[{i}] = {v}, expected {i}");
        }
    }

    #[test]
    fn test_scan_ascending() {
        let cfg = default_config();
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        dynamic_scan(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![0.0, 1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_scan_two_elements() {
        let cfg = default_config();
        let mut data = vec![7.0, 3.0];
        dynamic_scan(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![0.0, 7.0]);
    }

    #[test]
    fn test_scan_zeroes() {
        let cfg = default_config();
        let mut data = [0.0f32; 8];
        dynamic_scan(&mut data, &cfg).unwrap();
        assert_eq!(data.to_vec(), vec![0.0; 8]);
    }

    #[test]
    fn test_scan_forward_dispatch() {
        let cfg = default_config();
        let mut data = vec![1.0, 1.0, 1.0];
        dynamic_scan_forward(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_scan_invalid_config() {
        let mut cfg = default_config();
        cfg.max_depth = 0;
        let mut data = [1.0];
        assert!(dynamic_scan(&mut data, &cfg).is_err());
    }

    // ── recursive_merge_sort ────────────────────────────────────────

    #[test]
    fn test_sort_empty() {
        let cfg = default_config();
        let mut data: Vec<f32> = vec![];
        recursive_merge_sort(&mut data, &cfg).unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn test_sort_single() {
        let cfg = default_config();
        let mut data = [42.0f32];
        recursive_merge_sort(&mut data, &cfg).unwrap();
        assert_eq!(data.to_vec(), vec![42.0]);
    }

    #[test]
    fn test_sort_two_sorted() {
        let cfg = default_config();
        let mut data = vec![1.0, 2.0];
        recursive_merge_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![1.0, 2.0]);
    }

    #[test]
    fn test_sort_two_reversed() {
        let cfg = default_config();
        let mut data = vec![2.0, 1.0];
        recursive_merge_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![1.0, 2.0]);
    }

    #[test]
    fn test_sort_descending() {
        let cfg = default_config();
        let mut data = descending(10);
        recursive_merge_sort(&mut data, &cfg).unwrap();
        let expected = ascending(10);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_sort_ascending_stable() {
        let cfg = default_config();
        let mut data = ascending(20);
        let expected = data.clone();
        recursive_merge_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, expected);
    }

    #[test]
    fn test_sort_duplicates() {
        let cfg = default_config();
        let mut data = vec![3.0, 1.0, 2.0, 1.0, 3.0, 2.0];
        recursive_merge_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_sort_all_same() {
        let cfg = default_config();
        let mut data = constant(16, 5.0);
        recursive_merge_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, constant(16, 5.0));
    }

    #[test]
    fn test_sort_negative_values() {
        let cfg = default_config();
        let mut data = vec![-3.0, -1.0, -4.0, -1.5, -2.0];
        recursive_merge_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![-4.0, -3.0, -2.0, -1.5, -1.0]);
    }

    #[test]
    fn test_sort_mixed_sign() {
        let cfg = default_config();
        let mut data = vec![3.0, -1.0, 0.0, 2.0, -2.0];
        recursive_merge_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![-2.0, -1.0, 0.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sort_large_recursive() {
        let cfg = config_depth(4);
        let n = 4096;
        let mut data = descending(n);
        recursive_merge_sort(&mut data, &cfg).unwrap();
        for i in 1..n {
            assert!(data[i - 1] <= data[i], "not sorted at index {i}");
        }
    }

    #[test]
    fn test_sort_depth_1() {
        let cfg = config_depth(1);
        let mut data = vec![5.0, 3.0, 4.0, 1.0, 2.0];
        recursive_merge_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sort_forward_dispatch() {
        let cfg = default_config();
        let mut data = vec![3.0, 1.0, 2.0];
        recursive_merge_sort_forward(&mut data, &cfg).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sort_invalid_config() {
        let mut cfg = default_config();
        cfg.max_depth = 0;
        assert!(recursive_merge_sort(&mut vec![1.0], &cfg).is_err());
    }

    // ── adaptive_grid_launch ────────────────────────────────────────

    #[test]
    fn test_adaptive_grid_basic() {
        let cfg = default_config();
        let input = vec![1.0, 2.0, 3.0];
        let mut output = [0.0f32; 3];
        adaptive_grid_launch(&input, &mut output, &cfg).unwrap();
        assert_eq!(output.to_vec(), vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_adaptive_grid_empty() {
        let cfg = default_config();
        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        adaptive_grid_launch(&input, &mut output, &cfg).unwrap();
    }

    #[test]
    fn test_adaptive_grid_single() {
        let cfg = default_config();
        let input = [5.0f32];
        let mut output = [0.0f32];
        adaptive_grid_launch(&input, &mut output, &cfg).unwrap();
        assert_eq!(output.to_vec(), vec![25.0]);
    }

    #[test]
    fn test_adaptive_grid_zeros() {
        let cfg = default_config();
        let input = constant(8, 0.0);
        let mut output = [0.0f32; 8];
        adaptive_grid_launch(&input, &mut output, &cfg).unwrap();
        assert_eq!(output.to_vec(), constant(8, 0.0));
    }

    #[test]
    fn test_adaptive_grid_negative() {
        let cfg = default_config();
        let input = vec![-3.0, -2.0, -1.0];
        let mut output = [0.0f32; 3];
        adaptive_grid_launch(&input, &mut output, &cfg).unwrap();
        assert_eq!(output.to_vec(), vec![9.0, 4.0, 1.0]);
    }

    #[test]
    fn test_adaptive_grid_large_recursive() {
        let cfg = config_depth(4).with_min_items(256);
        let n = 4096;
        let input = ascending(n);
        let mut output = vec![0.0; n];
        adaptive_grid_launch(&input, &mut output, &cfg).unwrap();
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert!((out - inp * inp).abs() < 1e-3, "mismatch at {i}: {out} vs {}", inp * inp);
        }
    }

    #[test]
    fn test_adaptive_grid_output_too_short() {
        let cfg = default_config();
        let input = vec![1.0, 2.0, 3.0];
        let mut output = [0.0; 2];
        assert!(adaptive_grid_launch(&input, &mut output, &cfg).is_err());
    }

    #[test]
    fn test_adaptive_grid_output_longer_ok() {
        let cfg = default_config();
        let input = [2.0];
        let mut output = [0.0; 5];
        adaptive_grid_launch(&input, &mut output, &cfg).unwrap();
        assert_eq!(output[0], 4.0);
    }

    #[test]
    fn test_adaptive_grid_forward_dispatch() {
        let cfg = default_config();
        let input = [3.0f32];
        let mut output = [0.0f32];
        adaptive_grid_launch_forward(&input, &mut output, &cfg).unwrap();
        assert_eq!(output.to_vec(), vec![9.0]);
    }

    #[test]
    fn test_adaptive_grid_invalid_config() {
        let mut cfg = default_config();
        cfg.max_depth = 0;
        let mut out = [0.0];
        assert!(adaptive_grid_launch(&[1.0], &mut out, &cfg).is_err());
    }

    // ── nested_matmul ───────────────────────────────────────────────

    #[test]
    fn test_matmul_1x1() {
        let cfg = default_config();
        let a = [3.0];
        let b = [4.0];
        let mut c = [0.0];
        nested_matmul(&a, &b, &mut c, 1, 1, 1, &cfg).unwrap();
        assert!((c[0] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_2x2_identity() {
        let cfg = default_config();
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        nested_matmul(&a, &b, &mut c, 2, 2, 2, &cfg).unwrap();
        assert_eq!(c.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_matmul_2x3_3x2() {
        let cfg = default_config();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3×2
        let mut c = [0.0; 4];
        nested_matmul(&a, &b, &mut c, 2, 2, 3, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, 2, 2, 3);
        for i in 0..4 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-3,
                "c[{i}] = {}, expected {}",
                c[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_matmul_accumulates_into_c() {
        let cfg = default_config();
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![10.0, 20.0, 30.0, 40.0];
        nested_matmul(&a, &b, &mut c, 2, 2, 2, &cfg).unwrap();
        assert_eq!(c, vec![11.0, 20.0, 30.0, 41.0]);
    }

    #[test]
    fn test_matmul_zeros() {
        let cfg = default_config();
        let a = [0.0; 4];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = [0.0f32; 4];
        nested_matmul(&a, &b, &mut c, 2, 2, 2, &cfg).unwrap();
        assert_eq!(c.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_matmul_4x4() {
        let cfg = default_config();
        let a: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let mut c = [0.0f32; 16];
        nested_matmul(&a, &b, &mut c, 4, 4, 4, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, 4, 4, 4);
        for i in 0..16 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-2,
                "c[{i}] = {}, expected {}",
                c[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_matmul_non_square() {
        let cfg = default_config();
        let m = 3;
        let n = 5;
        let k = 4;
        let a: Vec<f32> = (0..m * k).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i + 1) as f32).collect();
        let mut c = vec![0.0f32; m * n];
        nested_matmul(&a, &b, &mut c, m, n, k, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        for i in 0..m * n {
            assert!(
                (c[i] - expected[i]).abs() < 1e-2,
                "c[{i}] = {}, expected {}",
                c[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_matmul_a_too_short() {
        let cfg = default_config();
        let a = [1.0; 3];
        let b = [1.0; 4];
        let mut c = [0.0; 4];
        assert!(nested_matmul(&a, &b, &mut c, 2, 2, 2, &cfg).is_err());
    }

    #[test]
    fn test_matmul_b_too_short() {
        let cfg = default_config();
        let a = [1.0; 4];
        let b = [1.0; 3];
        let mut c = [0.0; 4];
        assert!(nested_matmul(&a, &b, &mut c, 2, 2, 2, &cfg).is_err());
    }

    #[test]
    fn test_matmul_c_too_short() {
        let cfg = default_config();
        let a = [1.0; 4];
        let b = [1.0; 4];
        let mut c = [0.0; 3];
        assert!(nested_matmul(&a, &b, &mut c, 2, 2, 2, &cfg).is_err());
    }

    #[test]
    fn test_matmul_invalid_config() {
        let mut cfg = default_config();
        cfg.max_depth = 0;
        let mut c = [0.0; 4];
        assert!(nested_matmul(&[1.0; 4], &[1.0; 4], &mut c, 2, 2, 2, &cfg).is_err());
    }

    #[test]
    fn test_matmul_forward_dispatch() {
        let cfg = default_config();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0; 4];
        nested_matmul_forward(&a, &b, &mut c, 2, 2, 2, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, 2, 2, 2);
        for i in 0..4 {
            assert!((c[i] - expected[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn test_matmul_depth_1() {
        let cfg = config_depth(1);
        let a: Vec<f32> = (1..=9).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=9).map(|i| i as f32).collect();
        let mut c = [0.0f32; 9];
        nested_matmul(&a, &b, &mut c, 3, 3, 3, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, 3, 3, 3);
        for i in 0..9 {
            assert!((c[i] - expected[i]).abs() < 1e-2);
        }
    }

    // ── Cross-operation consistency ─────────────────────────────────

    #[test]
    fn test_reduce_then_scan_consistency() {
        // Sum via reduce should equal last element of inclusive scan.
        let cfg = default_config();
        let data = ascending(64);
        let sum = dynamic_reduce(&data, &cfg).unwrap();
        let mut scan_data = data.clone();
        dynamic_scan(&mut scan_data, &cfg).unwrap();
        // Exclusive scan last element + original last element = total sum.
        let n = data.len();
        let inclusive_last = scan_data[n - 1] + data[n - 1];
        assert!((sum - inclusive_last).abs() < 1e-3);
    }

    #[test]
    fn test_sort_preserves_sum() {
        let cfg = default_config();
        let mut data = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        let sum_before: f32 = data.iter().sum();
        recursive_merge_sort(&mut data, &cfg).unwrap();
        let sum_after: f32 = data.iter().sum();
        assert!((sum_before - sum_after).abs() < 1e-5);
    }

    #[test]
    fn test_sort_preserves_length() {
        let cfg = default_config();
        let mut data = descending(100);
        recursive_merge_sort(&mut data, &cfg).unwrap();
        assert_eq!(data.len(), 100);
    }

    #[test]
    fn test_adaptive_grid_matches_naive_square() {
        let cfg = default_config();
        let input = ascending(50);
        let mut output = [0.0; 50];
        adaptive_grid_launch(&input, &mut output, &cfg).unwrap();
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert!((out - inp * inp).abs() < 1e-3, "mismatch at {i}");
        }
    }

    #[test]
    fn test_matmul_commutativity_1x1() {
        let cfg = default_config();
        let mut c1 = [0.0];
        let mut c2 = [0.0];
        nested_matmul(&[3.0], &[7.0], &mut c1, 1, 1, 1, &cfg).unwrap();
        nested_matmul(&[7.0], &[3.0], &mut c2, 1, 1, 1, &cfg).unwrap();
        assert!((c1[0] - c2[0]).abs() < 1e-5);
    }

    #[test]
    fn test_scan_strictly_non_decreasing_for_positive() {
        let cfg = default_config();
        let mut data = constant(32, 1.0);
        dynamic_scan(&mut data, &cfg).unwrap();
        for i in 1..data.len() {
            assert!(data[i] >= data[i - 1]);
        }
    }

    #[test]
    fn test_reduce_all_ones() {
        let cfg = default_config();
        let n = 500;
        let data = constant(n, 1.0);
        let sum = dynamic_reduce(&data, &cfg).unwrap();
        assert!((sum - n as f32).abs() < 1e-2);
    }

    #[test]
    fn test_sort_already_sorted_large() {
        let cfg = config_depth(3);
        let mut data = ascending(2048);
        let expected = data.clone();
        recursive_merge_sort(&mut data, &cfg).unwrap();
        assert_eq!(data, expected);
    }

    #[test]
    fn test_scan_first_is_zero() {
        let cfg = default_config();
        let mut data = vec![99.0, 1.0, 2.0];
        dynamic_scan(&mut data, &cfg).unwrap();
        assert_eq!(data[0], 0.0);
    }
}
