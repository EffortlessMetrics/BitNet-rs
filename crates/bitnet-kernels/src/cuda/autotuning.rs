//! Autotuning framework for selecting optimal CUDA kernel parameters.
//!
//! # Overview
//!
//! CUDA kernel performance depends heavily on launch configuration parameters
//! (block size, grid size, shared memory allocation, unroll factor). This module
//! provides a systematic framework for discovering optimal configurations
//! through empirical search and heuristic fallback.
//!
//! # Workflow
//!
//! 1. Define a search space via [`AutotuneConfig`].
//! 2. Run [`autotune_matmul`], [`autotune_softmax`], or [`autotune_attention`]
//!    to benchmark candidate configurations.
//! 3. Cache the best result with [`cache_autotune_result`] for reuse.
//! 4. On cache miss or CPU-only builds, [`heuristic_config`] provides a fast
//!    rule-based fallback.
//!
//! # CPU fallback
//!
//! All public functions work on CPU-only builds, returning heuristic-based
//! configurations. Actual GPU benchmarking requires the `gpu` or `cuda`
//! feature.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ── Valid block sizes (must be multiples of warp size 32) ─────────────

/// Valid CUDA thread block sizes (multiples of warp size 32).
pub const VALID_BLOCK_SIZES: &[u32] = &[32, 64, 128, 256, 512, 1024];

/// CUDA warp size.
const WARP_SIZE: u32 = 32;

/// Maximum threads per SM on modern NVIDIA GPUs (Ampere+).
const MAX_THREADS_PER_SM: u32 = 2048;

/// Maximum shared memory per block (48 KB on most architectures).
const MAX_SHARED_MEM_PER_BLOCK: u32 = 49152;

/// Maximum registers per thread (typical limit).
const MAX_REGISTERS_PER_THREAD: u32 = 255;

// ── Kernel configuration ──────────────────────────────────────────────

/// Configurable CUDA kernel launch parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelConfig {
    /// Threads per block (must be a multiple of 32).
    pub block_size: u32,
    /// Grid dimensions (x, y, z).
    pub grid_size: (u32, u32, u32),
    /// Dynamic shared memory in bytes.
    pub shared_mem_bytes: u32,
    /// Loop unroll factor (1 = no unrolling).
    pub unroll_factor: u32,
}

impl KernelConfig {
    /// Create a new kernel configuration.
    ///
    /// Returns `None` if `block_size` is not a valid multiple of 32, is zero,
    /// or exceeds 1024.
    pub fn new(
        block_size: u32,
        grid_size: (u32, u32, u32),
        shared_mem_bytes: u32,
        unroll_factor: u32,
    ) -> Option<Self> {
        if !is_valid_block_size(block_size) {
            return None;
        }
        if grid_size.0 == 0 || grid_size.1 == 0 || grid_size.2 == 0 {
            return None;
        }
        if unroll_factor == 0 {
            return None;
        }
        Some(Self { block_size, grid_size, shared_mem_bytes, unroll_factor })
    }

    /// Total number of threads launched.
    pub fn total_threads(&self) -> u64 {
        self.block_size as u64
            * self.grid_size.0 as u64
            * self.grid_size.1 as u64
            * self.grid_size.2 as u64
    }
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self { block_size: 256, grid_size: (1, 1, 1), shared_mem_bytes: 0, unroll_factor: 1 }
    }
}

// ── Autotune configuration ────────────────────────────────────────────

/// Configuration for the autotuning search process.
#[derive(Debug, Clone)]
pub struct AutotuneConfig {
    /// Block sizes to evaluate.
    pub search_space: Vec<u32>,
    /// Maximum number of candidate configurations to try.
    pub max_trials: usize,
    /// Per-trial timeout in milliseconds.
    pub timeout_ms: u64,
    /// Number of warmup iterations before timing.
    pub warmup_iters: usize,
    /// Number of timed iterations for averaging.
    pub bench_iters: usize,
}

impl Default for AutotuneConfig {
    fn default() -> Self {
        Self {
            search_space: VALID_BLOCK_SIZES.to_vec(),
            max_trials: 20,
            timeout_ms: 5000,
            warmup_iters: 3,
            bench_iters: 10,
        }
    }
}

impl AutotuneConfig {
    /// Create a config with a custom search space.
    ///
    /// Invalid block sizes (not a multiple of 32, or > 1024) are silently
    /// removed.
    pub fn with_search_space(mut self, sizes: &[u32]) -> Self {
        self.search_space = sizes.iter().copied().filter(|&s| is_valid_block_size(s)).collect();
        self
    }

    /// Override the maximum number of trials.
    pub fn with_max_trials(mut self, n: usize) -> Self {
        self.max_trials = n;
        self
    }

    /// Override the per-trial timeout.
    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Generate the full search space of [`KernelConfig`] candidates for
    /// the given problem size.
    pub fn generate_search_space(
        &self,
        total_elements: usize,
        shared_mem_options: &[u32],
        unroll_options: &[u32],
    ) -> Vec<KernelConfig> {
        let mut configs = Vec::new();
        let shared_opts =
            if shared_mem_options.is_empty() { &[0u32][..] } else { shared_mem_options };
        let unroll_opts = if unroll_options.is_empty() { &[1u32][..] } else { unroll_options };

        for &block in &self.search_space {
            let grid_x = compute_grid_dim(total_elements as u32, block);
            for &smem in shared_opts {
                if smem > MAX_SHARED_MEM_PER_BLOCK {
                    continue;
                }
                for &unroll in unroll_opts {
                    if unroll == 0 {
                        continue;
                    }
                    configs.push(KernelConfig {
                        block_size: block,
                        grid_size: (grid_x, 1, 1),
                        shared_mem_bytes: smem,
                        unroll_factor: unroll,
                    });
                    if configs.len() >= self.max_trials {
                        return configs;
                    }
                }
            }
        }
        configs
    }
}

// ── Autotune result ───────────────────────────────────────────────────

/// Result of an autotuning search.
#[derive(Debug, Clone)]
pub struct AutotuneResult {
    /// Best kernel configuration found.
    pub best_config: KernelConfig,
    /// Best observed execution time.
    pub best_time: Duration,
    /// Number of configurations evaluated.
    pub configs_evaluated: usize,
    /// Total time spent autotuning.
    pub total_search_time: Duration,
    /// Per-configuration timing results (config index → duration).
    pub all_timings: Vec<(KernelConfig, Duration)>,
}

// ── Block size search ─────────────────────────────────────────────────

/// Find the optimal thread block size for a 1-D kernel over
/// `total_elements` elements.
///
/// `timing_fn` receives a block size and returns the measured execution
/// time. On CPU-only builds, pass a simulated timing function.
pub fn search_block_size<F>(
    config: &AutotuneConfig,
    total_elements: usize,
    mut timing_fn: F,
) -> AutotuneResult
where
    F: FnMut(u32) -> Duration,
{
    let start = Instant::now();
    let mut best_config = KernelConfig::default();
    let mut best_time = Duration::MAX;
    let mut all_timings = Vec::new();
    let mut evaluated = 0;
    let deadline = Duration::from_millis(config.timeout_ms);

    for &block_size in &config.search_space {
        if start.elapsed() > deadline {
            break;
        }
        if evaluated >= config.max_trials {
            break;
        }

        let grid_x = compute_grid_dim(total_elements as u32, block_size);
        let candidate = KernelConfig {
            block_size,
            grid_size: (grid_x, 1, 1),
            shared_mem_bytes: 0,
            unroll_factor: 1,
        };

        let elapsed = timing_fn(block_size);
        all_timings.push((candidate.clone(), elapsed));

        if elapsed < best_time {
            best_time = elapsed;
            best_config = candidate;
        }
        evaluated += 1;
    }

    AutotuneResult {
        best_config,
        best_time,
        configs_evaluated: evaluated,
        total_search_time: start.elapsed(),
        all_timings,
    }
}

// ── Grid size search ──────────────────────────────────────────────────

/// Compute the optimal 1-D grid dimension for the given total work items
/// and block size.
pub fn search_grid_size(total_elements: u32, block_size: u32) -> (u32, u32, u32) {
    let grid_x = compute_grid_dim(total_elements, block_size);
    (grid_x, 1, 1)
}

/// Compute optimal 2-D grid dimensions for a matrix operation.
pub fn search_grid_size_2d(rows: u32, cols: u32, block_size: u32) -> (u32, u32, u32) {
    let block_dim = (block_size as f32).sqrt().ceil() as u32;
    let block_dim = block_dim.max(1);
    let grid_x = compute_grid_dim(cols, block_dim);
    let grid_y = compute_grid_dim(rows, block_dim);
    (grid_x, grid_y, 1)
}

/// Compute optimal 3-D grid dimensions for a batched matrix operation.
pub fn search_grid_size_3d(rows: u32, cols: u32, batch: u32, block_size: u32) -> (u32, u32, u32) {
    let block_dim = (block_size as f32).sqrt().ceil() as u32;
    let block_dim = block_dim.max(1);
    let grid_x = compute_grid_dim(cols, block_dim);
    let grid_y = compute_grid_dim(rows, block_dim);
    (grid_x, grid_y, batch.max(1))
}

// ── Shared memory search ──────────────────────────────────────────────

/// Find the optimal shared memory allocation for a given block size and
/// element size.
///
/// Returns the largest shared memory allocation (in bytes) that fits
/// within the hardware limit and covers the requested tile.
pub fn search_shared_memory(_block_size: u32, element_size_bytes: u32, tile_elements: u32) -> u32 {
    let desired = tile_elements * element_size_bytes;
    // Align to 128-byte boundary for coalesced access.
    let aligned = (desired + 127) & !127;
    aligned.min(MAX_SHARED_MEM_PER_BLOCK)
}

// ── Kernel-specific autotune entry points ─────────────────────────────

/// Autotune matrix multiplication kernel parameters.
///
/// On CPU-only builds this returns a heuristic configuration.
pub fn autotune_matmul(m: usize, n: usize, k: usize, config: &AutotuneConfig) -> AutotuneResult {
    let total = m * n;
    // CPU fallback: use heuristic timing model based on FLOPs.
    let flops_per_element = 2 * k; // multiply + accumulate
    search_block_size(config, total, |block_size| {
        // Simulated cost model: penalise very small or very large blocks.
        let occupancy = estimate_occupancy(block_size, 0, 32);
        let eff = occupancy.max(0.01);
        let base_ns = (flops_per_element as f64) / eff;
        Duration::from_nanos(base_ns as u64)
    })
}

/// Autotune softmax kernel parameters.
///
/// Softmax is row-wise, so block size should ideally cover one row.
pub fn autotune_softmax(rows: usize, cols: usize, config: &AutotuneConfig) -> AutotuneResult {
    search_block_size(config, rows, |block_size| {
        // Cost model: each row needs cols reads + reductions.
        let passes = (cols as u32).div_ceil(block_size);
        let occupancy = estimate_occupancy(block_size, 0, 16);
        let eff = occupancy.max(0.01);
        let base_ns = (passes as f64 * cols as f64) / eff;
        Duration::from_nanos(base_ns as u64)
    })
}

/// Autotune attention kernel parameters.
///
/// Attention is memory-bandwidth-bound for long sequences; the optimal
/// block size balances parallelism with shared memory pressure.
pub fn autotune_attention(
    seq_len: usize,
    head_dim: usize,
    n_heads: usize,
    config: &AutotuneConfig,
) -> AutotuneResult {
    let total = seq_len * n_heads;
    let smem_per_row = (head_dim * 4) as u32; // f32 elements
    search_block_size(config, total, |block_size| {
        let smem = search_shared_memory(block_size, 4, head_dim as u32 * 2);
        let occupancy = estimate_occupancy(block_size, smem, 32);
        let eff = occupancy.max(0.01);
        let work = (seq_len as f64) * (head_dim as f64);
        let base_ns = (work * smem_per_row as f64) / (eff * 1e3);
        Duration::from_nanos(base_ns as u64)
    })
}

// ── Caching ───────────────────────────────────────────────────────────

/// Cache key: (kernel_name, input_shape_tuple).
type CacheKey = (String, Vec<usize>);

/// Global autotune cache.
static AUTOTUNE_CACHE: Mutex<Option<HashMap<CacheKey, KernelConfig>>> = Mutex::new(None);

fn with_cache<F, R>(f: F) -> R
where
    F: FnOnce(&mut HashMap<CacheKey, KernelConfig>) -> R,
{
    let mut guard = AUTOTUNE_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    f(map)
}

/// Cache an autotuning result keyed by kernel name and input shape.
pub fn cache_autotune_result(kernel_name: &str, input_shape: &[usize], config: &KernelConfig) {
    let key = (kernel_name.to_string(), input_shape.to_vec());
    with_cache(|cache| {
        cache.insert(key, config.clone());
    });
}

/// Load a cached autotuning result.
///
/// Returns `None` on cache miss.
pub fn load_autotune_cache(kernel_name: &str, input_shape: &[usize]) -> Option<KernelConfig> {
    let key = (kernel_name.to_string(), input_shape.to_vec());
    with_cache(|cache| cache.get(&key).cloned())
}

/// Clear the global autotune cache. Primarily useful for testing.
pub fn clear_autotune_cache() {
    with_cache(|cache| cache.clear());
}

// ── Heuristic fallback ────────────────────────────────────────────────

/// Fast heuristic configuration when autotuning is not available.
///
/// Selects block size based on problem size and estimates grid/shared
/// memory from matrix dimensions.
pub fn heuristic_config(m: usize, n: usize, k: usize) -> KernelConfig {
    let total = m * n;
    let block_size = heuristic_block_size(total);
    let grid_x = compute_grid_dim(total as u32, block_size);

    // Shared memory: one tile of A (block_size × tile_k) + one tile of B.
    let tile_k: u32 = 16;
    let smem = 2 * block_size * tile_k * 4; // 2 tiles × f32
    let smem = smem.min(MAX_SHARED_MEM_PER_BLOCK);

    let unroll = if k >= 256 {
        4
    } else if k >= 64 {
        2
    } else {
        1
    };

    KernelConfig {
        block_size,
        grid_size: (grid_x, 1, 1),
        shared_mem_bytes: smem,
        unroll_factor: unroll,
    }
}

/// Select a heuristic block size from the problem size.
fn heuristic_block_size(total_elements: usize) -> u32 {
    if total_elements <= 64 {
        32
    } else if total_elements <= 512 {
        64
    } else if total_elements <= 4096 {
        128
    } else {
        256
    }
}

// ── Occupancy calculator ──────────────────────────────────────────────

/// Estimate SM occupancy (0.0 – 1.0) for the given configuration.
///
/// This is a simplified model based on threads-per-SM limits and shared
/// memory constraints. Real occupancy depends on the specific GPU
/// architecture and register usage.
pub fn occupancy_calculator(config: &KernelConfig, regs_per_thread: u32) -> f64 {
    estimate_occupancy(config.block_size, config.shared_mem_bytes, regs_per_thread)
}

/// Core occupancy estimation.
fn estimate_occupancy(block_size: u32, shared_mem_bytes: u32, regs_per_thread: u32) -> f64 {
    if block_size == 0 {
        return 0.0;
    }

    // Thread-based limit: how many blocks fit based on thread count.
    let blocks_by_threads = MAX_THREADS_PER_SM / block_size;

    // Shared memory limit: how many blocks fit in 96 KB shared mem budget.
    let total_smem: u32 = 98304; // 96 KB (Ampere)
    let blocks_by_smem = if shared_mem_bytes == 0 {
        blocks_by_threads // no limit from shared memory
    } else {
        total_smem / shared_mem_bytes
    };

    // Register limit: 65536 registers per SM (Ampere).
    let total_regs: u32 = 65536;
    let regs_per_block = block_size * regs_per_thread.min(MAX_REGISTERS_PER_THREAD);
    let blocks_by_regs =
        if regs_per_block == 0 { blocks_by_threads } else { total_regs / regs_per_block };

    let max_blocks = blocks_by_threads.min(blocks_by_smem).min(blocks_by_regs).max(1);
    let active_threads = max_blocks * block_size;
    (active_threads as f64) / (MAX_THREADS_PER_SM as f64)
}

// ── Validation helpers ────────────────────────────────────────────────

/// Returns `true` if `block_size` is a valid CUDA block size (multiple of
/// warp size 32 and ≤ 1024).
pub fn is_valid_block_size(block_size: u32) -> bool {
    block_size > 0 && block_size <= 1024 && block_size.is_multiple_of(WARP_SIZE)
}

/// Validate a [`KernelConfig`] against hardware constraints.
///
/// Returns a human-readable error message on failure.
pub fn validate_config(config: &KernelConfig) -> std::result::Result<(), String> {
    if !is_valid_block_size(config.block_size) {
        return Err(format!(
            "invalid block size {}: must be a multiple of {} and ≤ 1024",
            config.block_size, WARP_SIZE
        ));
    }
    if config.grid_size.0 == 0 || config.grid_size.1 == 0 || config.grid_size.2 == 0 {
        return Err("grid dimensions must be non-zero".to_string());
    }
    if config.shared_mem_bytes > MAX_SHARED_MEM_PER_BLOCK {
        return Err(format!(
            "shared memory {} exceeds maximum {} bytes",
            config.shared_mem_bytes, MAX_SHARED_MEM_PER_BLOCK
        ));
    }
    if config.unroll_factor == 0 {
        return Err("unroll factor must be ≥ 1".to_string());
    }
    Ok(())
}

// ── Internal helpers ──────────────────────────────────────────────────

/// Ceiling division for grid dimension computation.
fn compute_grid_dim(total: u32, block: u32) -> u32 {
    if block == 0 {
        return 1;
    }
    total.div_ceil(block).max(1)
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── KernelConfig tests ────────────────────────────────────────────

    #[test]
    fn test_kernel_config_default() {
        let cfg = KernelConfig::default();
        assert_eq!(cfg.block_size, 256);
        assert_eq!(cfg.grid_size, (1, 1, 1));
        assert_eq!(cfg.shared_mem_bytes, 0);
        assert_eq!(cfg.unroll_factor, 1);
    }

    #[test]
    fn test_kernel_config_new_valid() {
        let cfg = KernelConfig::new(128, (4, 2, 1), 1024, 2);
        assert!(cfg.is_some());
        let cfg = cfg.unwrap();
        assert_eq!(cfg.block_size, 128);
        assert_eq!(cfg.grid_size, (4, 2, 1));
    }

    #[test]
    fn test_kernel_config_new_invalid_block_size() {
        assert!(KernelConfig::new(0, (1, 1, 1), 0, 1).is_none());
        assert!(KernelConfig::new(33, (1, 1, 1), 0, 1).is_none());
        assert!(KernelConfig::new(2048, (1, 1, 1), 0, 1).is_none());
    }

    #[test]
    fn test_kernel_config_new_invalid_grid() {
        assert!(KernelConfig::new(256, (0, 1, 1), 0, 1).is_none());
        assert!(KernelConfig::new(256, (1, 0, 1), 0, 1).is_none());
        assert!(KernelConfig::new(256, (1, 1, 0), 0, 1).is_none());
    }

    #[test]
    fn test_kernel_config_new_invalid_unroll() {
        assert!(KernelConfig::new(256, (1, 1, 1), 0, 0).is_none());
    }

    #[test]
    fn test_kernel_config_total_threads() {
        let cfg = KernelConfig::new(128, (4, 2, 3), 0, 1).unwrap();
        assert_eq!(cfg.total_threads(), 128 * 4 * 2 * 3);
    }

    #[test]
    fn test_kernel_config_total_threads_large() {
        let cfg = KernelConfig::new(1024, (65535, 1, 1), 0, 1).unwrap();
        assert_eq!(cfg.total_threads(), 1024 * 65535);
    }

    // ── Block size validation ─────────────────────────────────────────

    #[test]
    fn test_valid_block_sizes() {
        for &bs in VALID_BLOCK_SIZES {
            assert!(is_valid_block_size(bs), "block size {bs} should be valid");
        }
    }

    #[test]
    fn test_invalid_block_sizes() {
        assert!(!is_valid_block_size(0));
        assert!(!is_valid_block_size(1));
        assert!(!is_valid_block_size(31));
        assert!(!is_valid_block_size(33));
        assert!(!is_valid_block_size(100));
        assert!(!is_valid_block_size(1025));
        assert!(!is_valid_block_size(2048));
    }

    // ── AutotuneConfig tests ──────────────────────────────────────────

    #[test]
    fn test_autotune_config_default() {
        let cfg = AutotuneConfig::default();
        assert_eq!(cfg.search_space, VALID_BLOCK_SIZES);
        assert_eq!(cfg.max_trials, 20);
        assert_eq!(cfg.timeout_ms, 5000);
    }

    #[test]
    fn test_autotune_config_with_search_space_filters_invalid() {
        let cfg = AutotuneConfig::default().with_search_space(&[32, 33, 64, 100, 128]);
        assert_eq!(cfg.search_space, vec![32, 64, 128]);
    }

    #[test]
    fn test_autotune_config_with_max_trials() {
        let cfg = AutotuneConfig::default().with_max_trials(5);
        assert_eq!(cfg.max_trials, 5);
    }

    #[test]
    fn test_autotune_config_with_timeout() {
        let cfg = AutotuneConfig::default().with_timeout_ms(1000);
        assert_eq!(cfg.timeout_ms, 1000);
    }

    // ── Search space generation ───────────────────────────────────────

    #[test]
    fn test_generate_search_space_basic() {
        let cfg = AutotuneConfig::default().with_max_trials(100);
        let space = cfg.generate_search_space(1024, &[], &[]);
        assert!(!space.is_empty());
        for c in &space {
            assert!(is_valid_block_size(c.block_size));
            assert!(c.grid_size.0 > 0);
        }
    }

    #[test]
    fn test_generate_search_space_with_smem_options() {
        let cfg = AutotuneConfig::default().with_max_trials(100);
        let space = cfg.generate_search_space(1024, &[0, 4096, 8192], &[1]);
        assert!(space.len() >= VALID_BLOCK_SIZES.len());
    }

    #[test]
    fn test_generate_search_space_with_unroll_options() {
        let cfg = AutotuneConfig::default().with_max_trials(100);
        let space = cfg.generate_search_space(1024, &[0], &[1, 2, 4]);
        assert!(space.len() >= VALID_BLOCK_SIZES.len());
    }

    #[test]
    fn test_generate_search_space_respects_max_trials() {
        let cfg = AutotuneConfig::default().with_max_trials(3);
        let space = cfg.generate_search_space(1024, &[0, 4096], &[1, 2, 4]);
        assert!(space.len() <= 3);
    }

    #[test]
    fn test_generate_search_space_skips_excess_smem() {
        let cfg = AutotuneConfig::default().with_max_trials(100);
        let space = cfg.generate_search_space(1024, &[0, MAX_SHARED_MEM_PER_BLOCK + 1], &[1]);
        // Only the 0-smem variants should appear.
        for c in &space {
            assert!(c.shared_mem_bytes <= MAX_SHARED_MEM_PER_BLOCK);
        }
    }

    #[test]
    fn test_generate_search_space_skips_zero_unroll() {
        let cfg = AutotuneConfig::default().with_max_trials(100);
        let space = cfg.generate_search_space(1024, &[0], &[0, 1, 2]);
        for c in &space {
            assert!(c.unroll_factor >= 1);
        }
    }

    // ── Block size search ─────────────────────────────────────────────

    #[test]
    fn test_search_block_size_finds_best() {
        let cfg = AutotuneConfig::default();
        // Mock: 128 is fastest.
        let result = search_block_size(&cfg, 4096, |bs| {
            let cost = if bs == 128 { 10 } else { 100 };
            Duration::from_nanos(cost)
        });
        assert_eq!(result.best_config.block_size, 128);
        assert_eq!(result.configs_evaluated, VALID_BLOCK_SIZES.len());
    }

    #[test]
    fn test_search_block_size_evaluates_all_candidates() {
        let cfg = AutotuneConfig::default();
        let result = search_block_size(&cfg, 1024, |_bs| Duration::from_nanos(50));
        assert_eq!(result.configs_evaluated, VALID_BLOCK_SIZES.len());
        assert_eq!(result.all_timings.len(), VALID_BLOCK_SIZES.len());
    }

    #[test]
    fn test_search_block_size_respects_max_trials() {
        let cfg = AutotuneConfig::default().with_max_trials(2);
        let result = search_block_size(&cfg, 1024, |_bs| Duration::from_nanos(50));
        assert_eq!(result.configs_evaluated, 2);
    }

    #[test]
    fn test_search_block_size_grid_computation() {
        let cfg = AutotuneConfig::default().with_search_space(&[64]);
        let result = search_block_size(&cfg, 1000, |_| Duration::from_nanos(10));
        // ceil(1000 / 64) = 16
        assert_eq!(result.best_config.grid_size.0, 16);
    }

    #[test]
    fn test_search_block_size_single_candidate() {
        let cfg = AutotuneConfig::default().with_search_space(&[256]);
        let result = search_block_size(&cfg, 512, |_bs| Duration::from_nanos(42));
        assert_eq!(result.best_config.block_size, 256);
        assert_eq!(result.configs_evaluated, 1);
        assert_eq!(result.best_time, Duration::from_nanos(42));
    }

    // ── Grid size search ──────────────────────────────────────────────

    #[test]
    fn test_search_grid_size_1d() {
        assert_eq!(search_grid_size(1024, 256), (4, 1, 1));
    }

    #[test]
    fn test_search_grid_size_1d_not_divisible() {
        assert_eq!(search_grid_size(1000, 256), (4, 1, 1));
    }

    #[test]
    fn test_search_grid_size_1d_small() {
        assert_eq!(search_grid_size(1, 256), (1, 1, 1));
    }

    #[test]
    fn test_search_grid_size_2d() {
        let (gx, gy, gz) = search_grid_size_2d(512, 512, 256);
        assert!(gx > 0);
        assert!(gy > 0);
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_search_grid_size_3d_batched() {
        let (gx, gy, gz) = search_grid_size_3d(128, 128, 4, 256);
        assert!(gx > 0);
        assert!(gy > 0);
        assert_eq!(gz, 4);
    }

    #[test]
    fn test_search_grid_size_3d_batch_zero_becomes_one() {
        let (_, _, gz) = search_grid_size_3d(128, 128, 0, 256);
        assert_eq!(gz, 1);
    }

    // ── Shared memory search ──────────────────────────────────────────

    #[test]
    fn test_search_shared_memory_basic() {
        let smem = search_shared_memory(256, 4, 256);
        assert_eq!(smem, 1024); // 256 * 4 = 1024, already 128-aligned
    }

    #[test]
    fn test_search_shared_memory_alignment() {
        let smem = search_shared_memory(256, 4, 100);
        // 100 * 4 = 400, aligned to 128 → 512
        assert_eq!(smem % 128, 0);
        assert!(smem >= 400);
    }

    #[test]
    fn test_search_shared_memory_clamped() {
        let smem = search_shared_memory(1024, 4, 1_000_000);
        assert!(smem <= MAX_SHARED_MEM_PER_BLOCK);
    }

    // ── Autotune matmul ───────────────────────────────────────────────

    #[test]
    fn test_autotune_matmul_returns_valid() {
        let cfg = AutotuneConfig::default();
        let result = autotune_matmul(64, 64, 64, &cfg);
        assert!(is_valid_block_size(result.best_config.block_size));
        assert!(result.configs_evaluated > 0);
    }

    #[test]
    fn test_autotune_matmul_large() {
        let cfg = AutotuneConfig::default();
        let result = autotune_matmul(1024, 1024, 1024, &cfg);
        assert!(result.best_config.block_size >= 32);
        assert!(result.best_time < Duration::from_secs(10));
    }

    // ── Autotune softmax ──────────────────────────────────────────────

    #[test]
    fn test_autotune_softmax_returns_valid() {
        let cfg = AutotuneConfig::default();
        let result = autotune_softmax(32, 50257, &cfg);
        assert!(is_valid_block_size(result.best_config.block_size));
        assert!(result.configs_evaluated > 0);
    }

    #[test]
    fn test_autotune_softmax_single_row() {
        let cfg = AutotuneConfig::default();
        let result = autotune_softmax(1, 128, &cfg);
        assert!(result.configs_evaluated > 0);
    }

    // ── Autotune attention ────────────────────────────────────────────

    #[test]
    fn test_autotune_attention_returns_valid() {
        let cfg = AutotuneConfig::default();
        let result = autotune_attention(128, 64, 8, &cfg);
        assert!(is_valid_block_size(result.best_config.block_size));
        assert!(result.configs_evaluated > 0);
    }

    #[test]
    fn test_autotune_attention_small_seq() {
        let cfg = AutotuneConfig::default();
        let result = autotune_attention(1, 64, 1, &cfg);
        assert!(result.configs_evaluated > 0);
    }

    // ── Cache tests ───────────────────────────────────────────────────

    #[test]
    fn test_cache_store_and_load() {
        clear_autotune_cache();
        let cfg = KernelConfig::new(128, (4, 1, 1), 1024, 2).unwrap();
        cache_autotune_result("matmul", &[64, 64, 64], &cfg);
        let loaded = load_autotune_cache("matmul", &[64, 64, 64]);
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().block_size, 128);
    }

    #[test]
    fn test_cache_miss() {
        clear_autotune_cache();
        let result = load_autotune_cache("nonexistent", &[1, 2, 3]);
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_different_shapes() {
        clear_autotune_cache();
        let cfg1 = KernelConfig::new(128, (4, 1, 1), 0, 1).unwrap();
        let cfg2 = KernelConfig::new(256, (8, 1, 1), 0, 1).unwrap();
        cache_autotune_result("matmul", &[64, 64], &cfg1);
        cache_autotune_result("matmul", &[128, 128], &cfg2);

        let r1 = load_autotune_cache("matmul", &[64, 64]).unwrap();
        let r2 = load_autotune_cache("matmul", &[128, 128]).unwrap();
        assert_eq!(r1.block_size, 128);
        assert_eq!(r2.block_size, 256);
    }

    #[test]
    fn test_cache_different_kernels() {
        clear_autotune_cache();
        let cfg1 = KernelConfig::new(64, (1, 1, 1), 0, 1).unwrap();
        let cfg2 = KernelConfig::new(512, (1, 1, 1), 0, 1).unwrap();
        cache_autotune_result("matmul", &[64], &cfg1);
        cache_autotune_result("softmax", &[64], &cfg2);

        assert_eq!(load_autotune_cache("matmul", &[64]).unwrap().block_size, 64);
        assert_eq!(load_autotune_cache("softmax", &[64]).unwrap().block_size, 512);
    }

    #[test]
    fn test_cache_overwrite() {
        clear_autotune_cache();
        let cfg1 = KernelConfig::new(64, (1, 1, 1), 0, 1).unwrap();
        let cfg2 = KernelConfig::new(256, (1, 1, 1), 0, 1).unwrap();
        cache_autotune_result("matmul", &[64], &cfg1);
        cache_autotune_result("matmul", &[64], &cfg2);
        assert_eq!(load_autotune_cache("matmul", &[64]).unwrap().block_size, 256);
    }

    #[test]
    fn test_cache_clear() {
        clear_autotune_cache();
        let cfg = KernelConfig::new(128, (1, 1, 1), 0, 1).unwrap();
        cache_autotune_result("test", &[1], &cfg);
        assert!(load_autotune_cache("test", &[1]).is_some());
        clear_autotune_cache();
        assert!(load_autotune_cache("test", &[1]).is_none());
    }

    // ── Heuristic config tests ────────────────────────────────────────

    #[test]
    fn test_heuristic_config_small() {
        let cfg = heuristic_config(8, 8, 8);
        assert!(is_valid_block_size(cfg.block_size));
        assert_eq!(cfg.block_size, 32); // small problem → small block
    }

    #[test]
    fn test_heuristic_config_medium() {
        let cfg = heuristic_config(64, 64, 64);
        assert!(is_valid_block_size(cfg.block_size));
        assert!(cfg.block_size >= 128);
    }

    #[test]
    fn test_heuristic_config_large() {
        let cfg = heuristic_config(1024, 1024, 1024);
        assert!(is_valid_block_size(cfg.block_size));
        assert_eq!(cfg.block_size, 256);
    }

    #[test]
    fn test_heuristic_config_unroll_increases_with_k() {
        let small = heuristic_config(64, 64, 32);
        let large = heuristic_config(64, 64, 512);
        assert!(large.unroll_factor >= small.unroll_factor);
    }

    #[test]
    fn test_heuristic_config_shared_mem_bounded() {
        let cfg = heuristic_config(4096, 4096, 4096);
        assert!(cfg.shared_mem_bytes <= MAX_SHARED_MEM_PER_BLOCK);
    }

    #[test]
    fn test_heuristic_config_grid_covers_problem() {
        let cfg = heuristic_config(100, 100, 100);
        let total_threads = cfg.grid_size.0 as u64 * cfg.block_size as u64;
        assert!(total_threads >= 100 * 100);
    }

    // ── Occupancy calculator tests ────────────────────────────────────

    #[test]
    fn test_occupancy_full() {
        // 256 threads, no shared mem, few registers → should achieve high occupancy.
        let cfg = KernelConfig::new(256, (1, 1, 1), 0, 1).unwrap();
        let occ = occupancy_calculator(&cfg, 16);
        assert!(occ > 0.5, "occupancy {occ} should be > 0.5");
        assert!(occ <= 1.0);
    }

    #[test]
    fn test_occupancy_large_block() {
        let cfg = KernelConfig::new(1024, (1, 1, 1), 0, 1).unwrap();
        let occ = occupancy_calculator(&cfg, 32);
        assert!(occ > 0.0);
        assert!(occ <= 1.0);
    }

    #[test]
    fn test_occupancy_high_smem_reduces() {
        let low_smem = KernelConfig::new(256, (1, 1, 1), 1024, 1).unwrap();
        let high_smem = KernelConfig::new(256, (1, 1, 1), 48000, 1).unwrap();
        let occ_low = occupancy_calculator(&low_smem, 32);
        let occ_high = occupancy_calculator(&high_smem, 32);
        assert!(occ_low >= occ_high, "more smem should reduce occupancy");
    }

    #[test]
    fn test_occupancy_high_regs_reduces() {
        let cfg = KernelConfig::new(256, (1, 1, 1), 0, 1).unwrap();
        let occ_low_regs = occupancy_calculator(&cfg, 16);
        let occ_high_regs = occupancy_calculator(&cfg, 128);
        assert!(occ_low_regs >= occ_high_regs, "more registers should reduce occupancy");
    }

    #[test]
    fn test_occupancy_bounded_0_1() {
        for &bs in VALID_BLOCK_SIZES {
            let cfg = KernelConfig::new(bs, (1, 1, 1), 0, 1).unwrap();
            let occ = occupancy_calculator(&cfg, 32);
            assert!(occ >= 0.0 && occ <= 1.0, "occupancy {occ} out of range for bs={bs}");
        }
    }

    // ── Validate config tests ─────────────────────────────────────────

    #[test]
    fn test_validate_config_valid() {
        let cfg = KernelConfig::new(256, (4, 2, 1), 4096, 2).unwrap();
        assert!(validate_config(&cfg).is_ok());
    }

    #[test]
    fn test_validate_config_invalid_block_size() {
        let cfg = KernelConfig { block_size: 33, ..KernelConfig::default() };
        assert!(validate_config(&cfg).is_err());
        assert!(validate_config(&cfg).unwrap_err().contains("block size"));
    }

    #[test]
    fn test_validate_config_zero_grid() {
        let cfg = KernelConfig { grid_size: (0, 1, 1), ..KernelConfig::default() };
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn test_validate_config_excess_smem() {
        let cfg = KernelConfig {
            shared_mem_bytes: MAX_SHARED_MEM_PER_BLOCK + 1,
            ..KernelConfig::default()
        };
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn test_validate_config_zero_unroll() {
        let cfg = KernelConfig { unroll_factor: 0, ..KernelConfig::default() };
        assert!(validate_config(&cfg).is_err());
    }

    // ── Grid dimension computation ────────────────────────────────────

    #[test]
    fn test_compute_grid_dim_exact() {
        assert_eq!(compute_grid_dim(1024, 256), 4);
    }

    #[test]
    fn test_compute_grid_dim_with_remainder() {
        assert_eq!(compute_grid_dim(1025, 256), 5);
    }

    #[test]
    fn test_compute_grid_dim_small() {
        assert_eq!(compute_grid_dim(1, 256), 1);
    }

    #[test]
    fn test_compute_grid_dim_zero_block() {
        assert_eq!(compute_grid_dim(1024, 0), 1);
    }

    // ── AutotuneResult structure tests ────────────────────────────────

    #[test]
    fn test_autotune_result_timings_match_evaluated() {
        let cfg = AutotuneConfig::default();
        let result = search_block_size(&cfg, 1024, |_| Duration::from_nanos(50));
        assert_eq!(result.all_timings.len(), result.configs_evaluated);
    }

    #[test]
    fn test_autotune_result_best_time_in_timings() {
        let cfg = AutotuneConfig::default();
        let result = search_block_size(&cfg, 1024, |bs| {
            Duration::from_nanos(if bs == 256 { 1 } else { 100 })
        });
        let min_timing = result.all_timings.iter().map(|(_, d)| *d).min().unwrap();
        assert_eq!(result.best_time, min_timing);
    }

    // ── Integration-style tests ───────────────────────────────────────

    #[test]
    fn test_autotune_then_cache_round_trip() {
        clear_autotune_cache();
        let cfg = AutotuneConfig::default();
        let result = autotune_matmul(128, 128, 128, &cfg);
        cache_autotune_result("matmul", &[128, 128, 128], &result.best_config);
        let loaded = load_autotune_cache("matmul", &[128, 128, 128]).unwrap();
        assert_eq!(loaded.block_size, result.best_config.block_size);
    }

    #[test]
    fn test_heuristic_always_valid() {
        for size in [1, 8, 32, 64, 128, 256, 512, 1024, 4096] {
            let cfg = heuristic_config(size, size, size);
            assert!(validate_config(&cfg).is_ok(), "invalid heuristic for size={size}");
        }
    }

    #[test]
    fn test_search_space_all_configs_valid() {
        let ac = AutotuneConfig::default().with_max_trials(100);
        let space = ac.generate_search_space(4096, &[0, 4096], &[1, 2]);
        for cfg in &space {
            assert!(validate_config(cfg).is_ok(), "invalid config in search space: {cfg:?}");
        }
    }
}
