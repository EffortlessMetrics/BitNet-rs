//! OpenCL kernel auto-tuning framework for Intel Arc A770 hardware.
//!
//! Benchmarks kernel configurations and selects optimal workgroup sizes,
//! tiling parameters, and vectorization widths. Provides CPU-simulated
//! reference implementations for offline tuning without device access.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Tuning parameter
// ---------------------------------------------------------------------------

/// A single tuning knob for an OpenCL kernel dispatch.
#[derive(Debug, Clone, PartialEq)]
pub enum TuningParam {
    WorkgroupSizeX(usize),
    WorkgroupSizeY(usize),
    TileSize(usize),
    VectorWidth(usize),
    UnrollFactor(usize),
    UseLocalMemory(bool),
}

impl fmt::Display for TuningParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkgroupSizeX(v) => write!(f, "wg_x={v}"),
            Self::WorkgroupSizeY(v) => write!(f, "wg_y={v}"),
            Self::TileSize(v) => write!(f, "tile={v}"),
            Self::VectorWidth(v) => write!(f, "vec={v}"),
            Self::UnrollFactor(v) => write!(f, "unroll={v}"),
            Self::UseLocalMemory(v) => write!(f, "local_mem={v}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tuning configuration
// ---------------------------------------------------------------------------

/// A concrete set of parameters for a single kernel variant.
#[derive(Debug, Clone, PartialEq)]
pub struct TuningConfig {
    pub params: Vec<TuningParam>,
    pub name: String,
}

impl TuningConfig {
    /// Extract the workgroup-size-X value, if present.
    pub fn workgroup_x(&self) -> Option<usize> {
        self.params.iter().find_map(|p| match p {
            TuningParam::WorkgroupSizeX(v) => Some(*v),
            _ => None,
        })
    }

    /// Extract the workgroup-size-Y value, if present.
    pub fn workgroup_y(&self) -> Option<usize> {
        self.params.iter().find_map(|p| match p {
            TuningParam::WorkgroupSizeY(v) => Some(*v),
            _ => None,
        })
    }

    /// Extract tile size, if present.
    pub fn tile_size(&self) -> Option<usize> {
        self.params.iter().find_map(|p| match p {
            TuningParam::TileSize(v) => Some(*v),
            _ => None,
        })
    }

    /// Extract vector width, if present.
    pub fn vector_width(&self) -> Option<usize> {
        self.params.iter().find_map(|p| match p {
            TuningParam::VectorWidth(v) => Some(*v),
            _ => None,
        })
    }

    /// Extract unroll factor, if present.
    pub fn unroll_factor(&self) -> Option<usize> {
        self.params.iter().find_map(|p| match p {
            TuningParam::UnrollFactor(v) => Some(*v),
            _ => None,
        })
    }

    /// Check whether local memory is enabled.
    pub fn uses_local_memory(&self) -> Option<bool> {
        self.params.iter().find_map(|p| match p {
            TuningParam::UseLocalMemory(v) => Some(*v),
            _ => None,
        })
    }

    /// Total workgroup size (X × Y, defaulting each to 1).
    pub fn total_workgroup_size(&self) -> usize {
        self.workgroup_x().unwrap_or(1) * self.workgroup_y().unwrap_or(1)
    }
}

impl fmt::Display for TuningConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: [", self.name)?;
        for (i, p) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{p}")?;
        }
        write!(f, "]")
    }
}

// ---------------------------------------------------------------------------
// Search space
// ---------------------------------------------------------------------------

/// The full set of configurations to explore for a kernel.
#[derive(Debug, Clone)]
pub struct TuningSpace {
    pub configs: Vec<TuningConfig>,
    pub kernel_name: String,
}

// ---------------------------------------------------------------------------
// Tuning result / report
// ---------------------------------------------------------------------------

/// Outcome of evaluating a single configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct TuningResult {
    pub config: TuningConfig,
    pub elapsed_us: u64,
    pub gflops: f64,
    pub memory_bandwidth_gbps: f64,
    pub valid: bool,
}

/// Summary produced after a tuning run.
#[derive(Debug, Clone, PartialEq)]
pub struct TuningReport {
    pub kernel_name: String,
    pub best_config: TuningConfig,
    pub all_results: Vec<TuningResult>,
    pub search_time_us: u64,
    pub speedup_vs_default: f64,
}

// ---------------------------------------------------------------------------
// Search strategy
// ---------------------------------------------------------------------------

/// Strategy used by [`AutoTuner`] to explore the search space.
#[derive(Debug, Clone, PartialEq)]
pub enum SearchStrategy {
    Exhaustive,
    RandomSample(usize),
    GridSearch { steps: usize },
    HillClimbing { restarts: usize },
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during auto-tuning.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TuneError {
    #[error("no valid configurations found in search space")]
    NoValidConfigs,
    #[error("benchmark failed: {0}")]
    BenchmarkFailed(String),
    #[error("tuning cache corrupted")]
    CacheCorrupted,
    #[error("invalid tuning parameter: {0}")]
    InvalidParam(String),
}

// ---------------------------------------------------------------------------
// AutoTuner
// ---------------------------------------------------------------------------

/// Persistent tuner that caches optimal configurations across runs.
#[derive(Debug)]
pub struct AutoTuner {
    pub cache: HashMap<String, TuningConfig>,
    pub history: Vec<TuningReport>,
    pub strategy: SearchStrategy,
}

// ---------------------------------------------------------------------------
// Public API — construction
// ---------------------------------------------------------------------------

/// Create a new [`AutoTuner`] with the given search strategy.
pub fn create_auto_tuner(strategy: SearchStrategy) -> AutoTuner {
    AutoTuner {
        cache: HashMap::new(),
        history: Vec::new(),
        strategy,
    }
}

// ---------------------------------------------------------------------------
// Public API — search-space generation
// ---------------------------------------------------------------------------

/// Enumerate valid kernel configurations for the given constraints.
///
/// Generates combinations of workgroup sizes (powers of 2 up to
/// `max_workgroup`), tile sizes, vector widths, and unroll factors.
pub fn cpu_generate_search_space(
    kernel_name: &str,
    max_workgroup: usize,
    max_tile: usize,
) -> TuningSpace {
    let wg_sizes: Vec<usize> =
        (0..).map(|i| 1usize << i).take_while(|&v| v <= max_workgroup).collect();
    let tile_sizes: Vec<usize> =
        (0..).map(|i| 1usize << i).take_while(|&v| v <= max_tile).collect();
    let vector_widths: Vec<usize> = vec![1, 2, 4, 8];
    let unroll_factors: Vec<usize> = vec![1, 2, 4];
    let local_mem_opts: Vec<bool> = vec![false, true];

    let mut configs = Vec::new();
    for &wg_x in &wg_sizes {
        for &wg_y in &wg_sizes {
            if wg_x * wg_y > max_workgroup {
                continue;
            }
            for &tile in &tile_sizes {
                for &vec_w in &vector_widths {
                    for &unroll in &unroll_factors {
                        for &local_mem in &local_mem_opts {
                            let name = format!(
                                "{kernel_name}_wg{wg_x}x{wg_y}_t{tile}\
                                 _v{vec_w}_u{unroll}_lm{local_mem}"
                            );
                            configs.push(TuningConfig {
                                params: vec![
                                    TuningParam::WorkgroupSizeX(wg_x),
                                    TuningParam::WorkgroupSizeY(wg_y),
                                    TuningParam::TileSize(tile),
                                    TuningParam::VectorWidth(vec_w),
                                    TuningParam::UnrollFactor(unroll),
                                    TuningParam::UseLocalMemory(local_mem),
                                ],
                                name,
                            });
                        }
                    }
                }
            }
        }
    }

    TuningSpace { configs, kernel_name: kernel_name.to_string() }
}

// ---------------------------------------------------------------------------
// Public API — evaluation (CPU-simulated)
// ---------------------------------------------------------------------------

/// Simulate evaluating a kernel configuration on CPU.
///
/// The performance model rewards larger workgroups, wider vectors, and
/// local-memory usage — mimicking real GPU behaviour without device access.
pub fn cpu_evaluate_config(
    config: &TuningConfig,
    problem_size: (usize, usize, usize),
) -> TuningResult {
    let (m, n, k) = problem_size;
    let total_ops = 2.0 * m as f64 * n as f64 * k as f64; // FMA = 2 ops
    let total_bytes = ((m * k + k * n + m * n) * 4) as f64; // f32

    let wg_x = config.workgroup_x().unwrap_or(1) as f64;
    let wg_y = config.workgroup_y().unwrap_or(1) as f64;
    let tile = config.tile_size().unwrap_or(1) as f64;
    let vec_w = config.vector_width().unwrap_or(1) as f64;
    let unroll = config.unroll_factor().unwrap_or(1) as f64;
    let local_mem = config.uses_local_memory().unwrap_or(false);

    // Simulated throughput model: larger workgroups & wider vectors are faster
    let wg_factor = (wg_x * wg_y).sqrt().max(1.0);
    let tile_factor = tile.sqrt().max(1.0);
    let vec_factor = vec_w.max(1.0);
    let unroll_factor = unroll.sqrt().max(1.0);
    let local_mem_factor = if local_mem { 1.3 } else { 1.0 };

    let throughput =
        wg_factor * tile_factor * vec_factor * unroll_factor * local_mem_factor;

    // Simulated elapsed time inversely proportional to throughput
    let base_us = (total_ops / 1e6).max(1.0);
    let elapsed_us = (base_us / throughput).max(1.0) as u64;

    let elapsed_s = elapsed_us as f64 / 1e6;
    let gflops = total_ops / elapsed_s / 1e9;
    let memory_bandwidth_gbps = total_bytes / elapsed_s / 1e9;

    let valid = cpu_validate_config(config, 1024);

    TuningResult { config: config.clone(), elapsed_us, gflops, memory_bandwidth_gbps, valid }
}

// ---------------------------------------------------------------------------
// Public API — search strategies
// ---------------------------------------------------------------------------

/// Run the full tuning loop using the strategy stored in `tuner`.
pub fn cpu_run_tuning(
    tuner: &mut AutoTuner,
    space: &TuningSpace,
    problem_size: (usize, usize, usize),
) -> Result<TuningReport, TuneError> {
    let start = std::time::Instant::now();

    let results = match &tuner.strategy {
        SearchStrategy::Exhaustive => cpu_exhaustive_search(space, problem_size),
        SearchStrategy::RandomSample(n) => {
            cpu_random_search(space, *n, problem_size)
        }
        SearchStrategy::GridSearch { steps } => {
            // Grid search samples evenly across the space
            let step = space.configs.len().max(1) / (*steps).max(1);
            let step = step.max(1);
            space
                .configs
                .iter()
                .step_by(step)
                .map(|c| cpu_evaluate_config(c, problem_size))
                .collect()
        }
        SearchStrategy::HillClimbing { restarts } => {
            cpu_hill_climbing(space, *restarts, problem_size)
        }
    };

    let valid_results: Vec<TuningResult> =
        results.iter().filter(|r| r.valid).cloned().collect();
    if valid_results.is_empty() {
        return Err(TuneError::NoValidConfigs);
    }

    let best = valid_results
        .iter()
        .max_by(|a, b| a.gflops.partial_cmp(&b.gflops).unwrap())
        .unwrap()
        .clone();

    // Compute speedup vs the first (default) config
    let default_result = cpu_evaluate_config(
        space.configs.first().ok_or(TuneError::NoValidConfigs)?,
        problem_size,
    );
    let speedup = cpu_compute_speedup(default_result.elapsed_us, best.elapsed_us);

    let report = TuningReport {
        kernel_name: space.kernel_name.clone(),
        best_config: best.config.clone(),
        all_results: results,
        search_time_us: start.elapsed().as_micros() as u64,
        speedup_vs_default: speedup,
    };

    cpu_cache_store(tuner, &space.kernel_name, best.config);
    tuner.history.push(report.clone());

    Ok(report)
}

/// Evaluate every configuration in the search space.
pub fn cpu_exhaustive_search(
    space: &TuningSpace,
    problem_size: (usize, usize, usize),
) -> Vec<TuningResult> {
    space
        .configs
        .iter()
        .map(|c| cpu_evaluate_config(c, problem_size))
        .collect()
}

/// Evaluate a random subset of the search space.
pub fn cpu_random_search(
    space: &TuningSpace,
    samples: usize,
    problem_size: (usize, usize, usize),
) -> Vec<TuningResult> {
    if space.configs.is_empty() {
        return Vec::new();
    }
    // Deterministic pseudo-random selection for reproducibility
    let count = samples.min(space.configs.len());
    let step = space.configs.len().max(1) / count.max(1);
    let step = step.max(1);
    space
        .configs
        .iter()
        .step_by(step)
        .take(count)
        .map(|c| cpu_evaluate_config(c, problem_size))
        .collect()
}

/// Hill-climbing search with random restarts.
pub fn cpu_hill_climbing(
    space: &TuningSpace,
    restarts: usize,
    problem_size: (usize, usize, usize),
) -> Vec<TuningResult> {
    if space.configs.is_empty() {
        return Vec::new();
    }
    let mut all_results = Vec::new();
    let n = space.configs.len();

    for r in 0..restarts.max(1) {
        // Start at a different point each restart
        let start_idx = (r * 37) % n; // simple deterministic scatter
        let mut current = cpu_evaluate_config(&space.configs[start_idx], problem_size);
        all_results.push(current.clone());

        // Walk neighbours (adjacent indices) looking for improvements
        let mut idx = start_idx;
        for _ in 0..n.min(64) {
            let next_idx = (idx + 1) % n;
            let neighbour =
                cpu_evaluate_config(&space.configs[next_idx], problem_size);
            all_results.push(neighbour.clone());
            if neighbour.valid && neighbour.gflops > current.gflops {
                current = neighbour;
                idx = next_idx;
            } else {
                break;
            }
        }
    }

    all_results
}

// ---------------------------------------------------------------------------
// Public API — cache
// ---------------------------------------------------------------------------

/// Look up a previously cached optimal configuration.
pub fn cpu_cache_lookup<'a>(
    tuner: &'a AutoTuner,
    kernel_name: &str,
) -> Option<&'a TuningConfig> {
    tuner.cache.get(kernel_name)
}

/// Store an optimal configuration in the cache.
pub fn cpu_cache_store(tuner: &mut AutoTuner, kernel_name: &str, config: TuningConfig) {
    tuner.cache.insert(kernel_name.to_string(), config);
}

// ---------------------------------------------------------------------------
// Public API — validation helpers
// ---------------------------------------------------------------------------

/// Validate that a configuration respects hardware limits.
///
/// Checks that total workgroup size ≤ `max_workgroup` and that vector width
/// and tile size are powers of two.
pub fn cpu_validate_config(config: &TuningConfig, max_workgroup: usize) -> bool {
    let total_wg = config.total_workgroup_size();
    if total_wg > max_workgroup || total_wg == 0 {
        return false;
    }
    if let Some(v) = config.vector_width()
        && (!v.is_power_of_two() || v > 16)
    {
        return false;
    }
    if let Some(t) = config.tile_size()
        && !t.is_power_of_two()
    {
        return false;
    }
    true
}

/// Compute the speedup ratio between the default and tuned elapsed times.
pub fn cpu_compute_speedup(default_us: u64, tuned_us: u64) -> f64 {
    if tuned_us == 0 {
        return 1.0;
    }
    default_us as f64 / tuned_us as f64
}

// ---------------------------------------------------------------------------
// Public API — reporting
// ---------------------------------------------------------------------------

/// Format a [`TuningReport`] into a human-readable string.
pub fn format_tuning_report(report: &TuningReport) -> String {
    let valid_count = report.all_results.iter().filter(|r| r.valid).count();
    format!(
        "=== Tuning Report: {} ===\n\
         Best config: {}\n\
         GFLOPS:      {:.2}\n\
         Bandwidth:   {:.2} GB/s\n\
         Elapsed:     {} µs\n\
         Speedup:     {:.2}×\n\
         Search time: {} µs\n\
         Configs:     {} total, {} valid",
        report.kernel_name,
        report.best_config,
        report
            .all_results
            .iter()
            .filter(|r| r.valid)
            .map(|r| r.gflops)
            .fold(f64::NEG_INFINITY, f64::max),
        report
            .all_results
            .iter()
            .filter(|r| r.valid)
            .map(|r| r.memory_bandwidth_gbps)
            .fold(f64::NEG_INFINITY, f64::max),
        report
            .all_results
            .iter()
            .filter(|r| r.valid)
            .map(|r| r.elapsed_us)
            .min()
            .unwrap_or(0),
        report.speedup_vs_default,
        report.search_time_us,
        report.all_results.len(),
        valid_count,
    )
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn small_space() -> TuningSpace {
        cpu_generate_search_space("matmul_i2s", 64, 16)
    }

    fn default_problem() -> (usize, usize, usize) {
        (256, 256, 256)
    }

    fn single_config_space() -> TuningSpace {
        TuningSpace {
            configs: vec![TuningConfig {
                params: vec![
                    TuningParam::WorkgroupSizeX(16),
                    TuningParam::WorkgroupSizeY(1),
                    TuningParam::TileSize(4),
                    TuningParam::VectorWidth(4),
                    TuningParam::UnrollFactor(1),
                    TuningParam::UseLocalMemory(false),
                ],
                name: "single".into(),
            }],
            kernel_name: "single_kernel".into(),
        }
    }

    // -----------------------------------------------------------------------
    // Search-space generation
    // -----------------------------------------------------------------------

    #[test]
    fn generate_space_nonempty() {
        let space = small_space();
        assert!(!space.configs.is_empty());
    }

    #[test]
    fn generate_space_count() {
        let space = cpu_generate_search_space("test", 32, 8);
        // Every generated config must respect max_workgroup constraint
        for cfg in &space.configs {
            assert!(cfg.total_workgroup_size() <= 32);
        }
        assert!(space.configs.len() > 1);
    }

    #[test]
    fn generate_space_kernel_name_preserved() {
        let space = cpu_generate_search_space("rope_fwd", 64, 8);
        assert_eq!(space.kernel_name, "rope_fwd");
    }

    #[test]
    fn generate_space_all_power_of_two_sizes() {
        let space = small_space();
        for cfg in &space.configs {
            let wg_x = cfg.workgroup_x().unwrap();
            let wg_y = cfg.workgroup_y().unwrap();
            assert!(wg_x.is_power_of_two());
            assert!(wg_y.is_power_of_two());
        }
    }

    #[test]
    fn generate_space_max_workgroup_respected() {
        let space = cpu_generate_search_space("k", 128, 4);
        for cfg in &space.configs {
            assert!(cfg.total_workgroup_size() <= 128);
        }
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    #[test]
    fn validate_config_within_limits() {
        let cfg = TuningConfig {
            params: vec![
                TuningParam::WorkgroupSizeX(16),
                TuningParam::WorkgroupSizeY(16),
                TuningParam::TileSize(8),
                TuningParam::VectorWidth(4),
                TuningParam::UnrollFactor(2),
                TuningParam::UseLocalMemory(true),
            ],
            name: "good".into(),
        };
        assert!(cpu_validate_config(&cfg, 1024));
    }

    #[test]
    fn validate_config_exceeds_workgroup() {
        let cfg = TuningConfig {
            params: vec![
                TuningParam::WorkgroupSizeX(64),
                TuningParam::WorkgroupSizeY(64),
            ],
            name: "too_big".into(),
        };
        assert!(!cpu_validate_config(&cfg, 1024)); // 64*64 = 4096 > 1024
    }

    #[test]
    fn validate_config_non_power_of_two_vec() {
        let cfg = TuningConfig {
            params: vec![
                TuningParam::WorkgroupSizeX(8),
                TuningParam::VectorWidth(3),
            ],
            name: "bad_vec".into(),
        };
        assert!(!cpu_validate_config(&cfg, 1024));
    }

    #[test]
    fn validate_config_vec_too_wide() {
        let cfg = TuningConfig {
            params: vec![
                TuningParam::WorkgroupSizeX(8),
                TuningParam::VectorWidth(32),
            ],
            name: "wide_vec".into(),
        };
        assert!(!cpu_validate_config(&cfg, 1024));
    }

    #[test]
    fn validate_config_empty_params_ok() {
        let cfg = TuningConfig { params: vec![], name: "empty".into() };
        // total_wg = 1*1 = 1, no vec/tile to check
        assert!(cpu_validate_config(&cfg, 1024));
    }

    #[test]
    fn validate_config_exact_max() {
        let cfg = TuningConfig {
            params: vec![
                TuningParam::WorkgroupSizeX(32),
                TuningParam::WorkgroupSizeY(32),
            ],
            name: "exact".into(),
        };
        assert!(cpu_validate_config(&cfg, 1024)); // 32*32 = 1024
    }

    // -----------------------------------------------------------------------
    // Evaluate
    // -----------------------------------------------------------------------

    #[test]
    fn evaluate_produces_result() {
        let cfg = TuningConfig {
            params: vec![
                TuningParam::WorkgroupSizeX(16),
                TuningParam::WorkgroupSizeY(1),
                TuningParam::TileSize(4),
                TuningParam::VectorWidth(4),
                TuningParam::UnrollFactor(1),
                TuningParam::UseLocalMemory(false),
            ],
            name: "test".into(),
        };
        let result = cpu_evaluate_config(&cfg, default_problem());
        assert!(result.elapsed_us > 0);
        assert!(result.gflops > 0.0);
        assert!(result.memory_bandwidth_gbps > 0.0);
    }

    #[test]
    fn evaluate_larger_wg_faster() {
        let small = TuningConfig {
            params: vec![
                TuningParam::WorkgroupSizeX(1),
                TuningParam::WorkgroupSizeY(1),
                TuningParam::TileSize(1),
                TuningParam::VectorWidth(1),
                TuningParam::UnrollFactor(1),
                TuningParam::UseLocalMemory(false),
            ],
            name: "small".into(),
        };
        let large = TuningConfig {
            params: vec![
                TuningParam::WorkgroupSizeX(16),
                TuningParam::WorkgroupSizeY(16),
                TuningParam::TileSize(8),
                TuningParam::VectorWidth(8),
                TuningParam::UnrollFactor(4),
                TuningParam::UseLocalMemory(true),
            ],
            name: "large".into(),
        };
        let rs = cpu_evaluate_config(&small, default_problem());
        let rl = cpu_evaluate_config(&large, default_problem());
        assert!(rl.gflops > rs.gflops);
    }

    #[test]
    fn evaluate_local_mem_boost() {
        let no_lm = TuningConfig {
            params: vec![
                TuningParam::WorkgroupSizeX(16),
                TuningParam::VectorWidth(4),
                TuningParam::UseLocalMemory(false),
            ],
            name: "no_lm".into(),
        };
        let with_lm = TuningConfig {
            params: vec![
                TuningParam::WorkgroupSizeX(16),
                TuningParam::VectorWidth(4),
                TuningParam::UseLocalMemory(true),
            ],
            name: "with_lm".into(),
        };
        let rn = cpu_evaluate_config(&no_lm, default_problem());
        let rl = cpu_evaluate_config(&with_lm, default_problem());
        assert!(rl.gflops > rn.gflops);
    }

    // -----------------------------------------------------------------------
    // Exhaustive search
    // -----------------------------------------------------------------------

    #[test]
    fn exhaustive_search_evaluates_all() {
        let space = small_space();
        let results = cpu_exhaustive_search(&space, default_problem());
        assert_eq!(results.len(), space.configs.len());
    }

    #[test]
    fn exhaustive_search_finds_best() {
        let space = small_space();
        let results = cpu_exhaustive_search(&space, default_problem());
        let best = results
            .iter()
            .filter(|r| r.valid)
            .max_by(|a, b| a.gflops.partial_cmp(&b.gflops).unwrap());
        assert!(best.is_some());
    }

    // -----------------------------------------------------------------------
    // Random search
    // -----------------------------------------------------------------------

    #[test]
    fn random_search_returns_requested_count() {
        let space = small_space();
        let results = cpu_random_search(&space, 10, default_problem());
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn random_search_clamped_to_space_size() {
        let space = single_config_space();
        let results = cpu_random_search(&space, 100, default_problem());
        assert!(results.len() <= space.configs.len());
    }

    #[test]
    fn random_search_empty_space() {
        let space = TuningSpace { configs: vec![], kernel_name: "empty".into() };
        let results = cpu_random_search(&space, 10, default_problem());
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // Hill climbing
    // -----------------------------------------------------------------------

    #[test]
    fn hill_climbing_returns_results() {
        let space = small_space();
        let results = cpu_hill_climbing(&space, 3, default_problem());
        assert!(!results.is_empty());
    }

    #[test]
    fn hill_climbing_more_restarts_more_results() {
        let space = small_space();
        let r1 = cpu_hill_climbing(&space, 1, default_problem());
        let r3 = cpu_hill_climbing(&space, 5, default_problem());
        assert!(r3.len() >= r1.len());
    }

    #[test]
    fn hill_climbing_empty_space() {
        let space = TuningSpace { configs: vec![], kernel_name: "e".into() };
        let results = cpu_hill_climbing(&space, 3, default_problem());
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // Cache
    // -----------------------------------------------------------------------

    #[test]
    fn cache_store_and_retrieve() {
        let mut tuner = create_auto_tuner(SearchStrategy::Exhaustive);
        let cfg = TuningConfig {
            params: vec![TuningParam::WorkgroupSizeX(16)],
            name: "cached".into(),
        };
        cpu_cache_store(&mut tuner, "matmul", cfg.clone());
        let found = cpu_cache_lookup(&tuner, "matmul");
        assert_eq!(found, Some(&cfg));
    }

    #[test]
    fn cache_miss_returns_none() {
        let tuner = create_auto_tuner(SearchStrategy::Exhaustive);
        assert_eq!(cpu_cache_lookup(&tuner, "nonexistent"), None);
    }

    #[test]
    fn cache_overwrite() {
        let mut tuner = create_auto_tuner(SearchStrategy::Exhaustive);
        let cfg1 = TuningConfig {
            params: vec![TuningParam::WorkgroupSizeX(8)],
            name: "v1".into(),
        };
        let cfg2 = TuningConfig {
            params: vec![TuningParam::WorkgroupSizeX(16)],
            name: "v2".into(),
        };
        cpu_cache_store(&mut tuner, "k", cfg1);
        cpu_cache_store(&mut tuner, "k", cfg2.clone());
        assert_eq!(cpu_cache_lookup(&tuner, "k"), Some(&cfg2));
    }

    // -----------------------------------------------------------------------
    // Run tuning (integration)
    // -----------------------------------------------------------------------

    #[test]
    fn run_tuning_exhaustive() {
        let mut tuner = create_auto_tuner(SearchStrategy::Exhaustive);
        let space = small_space();
        let report = cpu_run_tuning(&mut tuner, &space, default_problem());
        assert!(report.is_ok());
        let report = report.unwrap();
        assert_eq!(report.kernel_name, space.kernel_name);
        assert!(!report.all_results.is_empty());
    }

    #[test]
    fn run_tuning_random() {
        let mut tuner = create_auto_tuner(SearchStrategy::RandomSample(5));
        let space = small_space();
        let report = cpu_run_tuning(&mut tuner, &space, default_problem());
        assert!(report.is_ok());
    }

    #[test]
    fn run_tuning_grid_search() {
        let mut tuner =
            create_auto_tuner(SearchStrategy::GridSearch { steps: 4 });
        let space = small_space();
        let report = cpu_run_tuning(&mut tuner, &space, default_problem());
        assert!(report.is_ok());
    }

    #[test]
    fn run_tuning_hill_climbing() {
        let mut tuner =
            create_auto_tuner(SearchStrategy::HillClimbing { restarts: 2 });
        let space = small_space();
        let report = cpu_run_tuning(&mut tuner, &space, default_problem());
        assert!(report.is_ok());
    }

    #[test]
    fn run_tuning_caches_result() {
        let mut tuner = create_auto_tuner(SearchStrategy::Exhaustive);
        let space = small_space();
        let _ = cpu_run_tuning(&mut tuner, &space, default_problem()).unwrap();
        assert!(cpu_cache_lookup(&tuner, &space.kernel_name).is_some());
    }

    #[test]
    fn run_tuning_records_history() {
        let mut tuner = create_auto_tuner(SearchStrategy::Exhaustive);
        let space = small_space();
        let _ = cpu_run_tuning(&mut tuner, &space, default_problem()).unwrap();
        assert_eq!(tuner.history.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Speedup
    // -----------------------------------------------------------------------

    #[test]
    fn speedup_correct_ratio() {
        assert!((cpu_compute_speedup(200, 100) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn speedup_same_time() {
        assert!((cpu_compute_speedup(100, 100) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn speedup_zero_tuned() {
        assert!((cpu_compute_speedup(100, 0) - 1.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Report formatting
    // -----------------------------------------------------------------------

    #[test]
    fn format_report_contains_kernel_name() {
        let mut tuner = create_auto_tuner(SearchStrategy::Exhaustive);
        let space = small_space();
        let report = cpu_run_tuning(&mut tuner, &space, default_problem()).unwrap();
        let formatted = format_tuning_report(&report);
        assert!(formatted.contains(&report.kernel_name));
    }

    #[test]
    fn format_report_contains_speedup() {
        let mut tuner = create_auto_tuner(SearchStrategy::Exhaustive);
        let space = small_space();
        let report = cpu_run_tuning(&mut tuner, &space, default_problem()).unwrap();
        let formatted = format_tuning_report(&report);
        assert!(formatted.contains("Speedup"));
    }

    #[test]
    fn format_report_contains_gflops() {
        let mut tuner = create_auto_tuner(SearchStrategy::Exhaustive);
        let space = small_space();
        let report = cpu_run_tuning(&mut tuner, &space, default_problem()).unwrap();
        let formatted = format_tuning_report(&report);
        assert!(formatted.contains("GFLOPS"));
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn single_config_space_works() {
        let mut tuner = create_auto_tuner(SearchStrategy::Exhaustive);
        let space = single_config_space();
        let report = cpu_run_tuning(&mut tuner, &space, default_problem());
        assert!(report.is_ok());
        assert_eq!(report.unwrap().all_results.len(), 1);
    }

    #[test]
    fn all_configs_invalid_returns_error() {
        let space = TuningSpace {
            configs: vec![TuningConfig {
                params: vec![
                    TuningParam::WorkgroupSizeX(2048),
                    TuningParam::WorkgroupSizeY(2048),
                ],
                name: "invalid".into(),
            }],
            kernel_name: "bad".into(),
        };
        let mut tuner = create_auto_tuner(SearchStrategy::Exhaustive);
        let result = cpu_run_tuning(&mut tuner, &space, default_problem());
        assert_eq!(result, Err(TuneError::NoValidConfigs));
    }

    #[test]
    fn problem_size_1x1x1() {
        let cfg = TuningConfig {
            params: vec![
                TuningParam::WorkgroupSizeX(1),
                TuningParam::VectorWidth(1),
            ],
            name: "tiny".into(),
        };
        let r = cpu_evaluate_config(&cfg, (1, 1, 1));
        assert!(r.elapsed_us >= 1);
        assert!(r.gflops > 0.0);
    }

    #[test]
    fn empty_space_returns_error() {
        let space = TuningSpace { configs: vec![], kernel_name: "e".into() };
        let mut tuner = create_auto_tuner(SearchStrategy::Exhaustive);
        let result = cpu_run_tuning(&mut tuner, &space, default_problem());
        assert_eq!(result, Err(TuneError::NoValidConfigs));
    }

    // -----------------------------------------------------------------------
    // Properties
    // -----------------------------------------------------------------------

    #[test]
    fn best_result_has_highest_gflops() {
        let space = small_space();
        let results = cpu_exhaustive_search(&space, default_problem());
        let valid: Vec<_> = results.iter().filter(|r| r.valid).collect();
        let best = valid
            .iter()
            .max_by(|a, b| a.gflops.partial_cmp(&b.gflops).unwrap())
            .unwrap();
        for r in &valid {
            assert!(best.gflops >= r.gflops);
        }
    }

    #[test]
    fn speedup_ge_one_when_valid() {
        let mut tuner = create_auto_tuner(SearchStrategy::Exhaustive);
        let space = small_space();
        let report = cpu_run_tuning(&mut tuner, &space, default_problem()).unwrap();
        assert!(report.speedup_vs_default >= 1.0);
    }

    // -----------------------------------------------------------------------
    // A770 hardware constraints
    // -----------------------------------------------------------------------

    #[test]
    fn a770_max_workgroup_1024() {
        let space = cpu_generate_search_space("a770_matmul", 1024, 32);
        for cfg in &space.configs {
            assert!(cfg.total_workgroup_size() <= 1024);
        }
    }

    #[test]
    fn a770_subgroup_sizes() {
        // Intel Arc supports subgroup sizes 8, 16, 32
        let valid_subgroups = [8, 16, 32];
        for &sg in &valid_subgroups {
            let cfg = TuningConfig {
                params: vec![TuningParam::WorkgroupSizeX(sg)],
                name: format!("sg_{sg}"),
            };
            assert!(cpu_validate_config(&cfg, 1024));
        }
    }

    #[test]
    fn a770_subgroup_size_16_preferred() {
        // Intel Arc prefers subgroup size 16
        let cfg16 = TuningConfig {
            params: vec![
                TuningParam::WorkgroupSizeX(16),
                TuningParam::WorkgroupSizeY(16),
                TuningParam::TileSize(16),
                TuningParam::VectorWidth(8),
                TuningParam::UnrollFactor(4),
                TuningParam::UseLocalMemory(true),
            ],
            name: "a770_16".into(),
        };
        assert!(cpu_validate_config(&cfg16, 1024));
        let r = cpu_evaluate_config(&cfg16, default_problem());
        assert!(r.valid);
        assert!(r.gflops > 0.0);
    }

    // -----------------------------------------------------------------------
    // Display / misc
    // -----------------------------------------------------------------------

    #[test]
    fn tuning_param_display() {
        assert_eq!(TuningParam::WorkgroupSizeX(16).to_string(), "wg_x=16");
        assert_eq!(TuningParam::UseLocalMemory(true).to_string(), "local_mem=true");
    }

    #[test]
    fn tuning_config_display() {
        let cfg = TuningConfig {
            params: vec![TuningParam::WorkgroupSizeX(8), TuningParam::TileSize(4)],
            name: "test".into(),
        };
        let s = cfg.to_string();
        assert!(s.contains("wg_x=8"));
        assert!(s.contains("tile=4"));
    }

    #[test]
    fn tune_error_display() {
        let e = TuneError::NoValidConfigs;
        assert!(e.to_string().contains("no valid"));
        let e2 = TuneError::BenchmarkFailed("timeout".into());
        assert!(e2.to_string().contains("timeout"));
    }

    #[test]
    fn create_auto_tuner_default_state() {
        let tuner = create_auto_tuner(SearchStrategy::Exhaustive);
        assert!(tuner.cache.is_empty());
        assert!(tuner.history.is_empty());
        assert_eq!(tuner.strategy, SearchStrategy::Exhaustive);
    }

    #[test]
    fn search_strategy_variants() {
        assert_eq!(SearchStrategy::Exhaustive, SearchStrategy::Exhaustive);
        assert_ne!(
            SearchStrategy::RandomSample(10),
            SearchStrategy::RandomSample(20)
        );
        assert_eq!(
            SearchStrategy::GridSearch { steps: 5 },
            SearchStrategy::GridSearch { steps: 5 }
        );
        assert_eq!(
            SearchStrategy::HillClimbing { restarts: 3 },
            SearchStrategy::HillClimbing { restarts: 3 }
        );
    }
}
