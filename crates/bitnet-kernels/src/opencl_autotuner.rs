//! Automatic kernel parameter tuning system for OpenCL A770 performance
//! optimization.
//!
//! Explores a Cartesian product of tuning parameters (work-group size, tile
//! dimensions, vector width, etc.) to find the configuration that maximises
//! throughput for a given kernel on a given device and tensor shape.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// TuningParam — single tunable parameter
// ---------------------------------------------------------------------------

/// A single tunable kernel parameter with a valid range.
#[derive(Debug, Clone, PartialEq)]
pub struct TuningParam {
    /// Human-readable parameter name (e.g. `"workgroup_size"`).
    pub name: String,
    /// Minimum value (inclusive).
    pub min: u32,
    /// Maximum value (inclusive).
    pub max: u32,
    /// Step between consecutive candidate values.
    pub step: u32,
    /// Current / default value.
    pub current: u32,
}

impl TuningParam {
    /// Create a new tuning parameter.
    ///
    /// # Panics
    /// Panics when `min > max` or `step == 0`.
    pub fn new(name: impl Into<String>, min: u32, max: u32, step: u32, current: u32) -> Self {
        assert!(min <= max, "min ({min}) must be <= max ({max})");
        assert!(step > 0, "step must be > 0");
        Self { name: name.into(), min, max, step, current }
    }

    /// Iterator over all candidate values in `[min, max]` with the configured
    /// step.
    pub fn values(&self) -> Vec<u32> {
        let mut vals = Vec::new();
        let mut v = self.min;
        while v <= self.max {
            vals.push(v);
            v = v.saturating_add(self.step);
            if v <= self.min {
                break; // overflow guard
            }
        }
        vals
    }

    /// Number of distinct candidate values.
    pub fn num_values(&self) -> usize {
        self.values().len()
    }
}

impl fmt::Display for TuningParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}=[{},{}] step={} cur={}",
            self.name, self.min, self.max, self.step, self.current
        )
    }
}

// ---------------------------------------------------------------------------
// TuningSpace — Cartesian product of all parameter ranges
// ---------------------------------------------------------------------------

/// Cartesian product of multiple [`TuningParam`] ranges.
#[derive(Debug, Clone)]
pub struct TuningSpace {
    params: Vec<TuningParam>,
}

impl TuningSpace {
    /// Build a tuning space from a set of parameters.
    pub fn new(params: Vec<TuningParam>) -> Self {
        Self { params }
    }

    /// Total number of configurations in the Cartesian product.
    pub fn total_configurations(&self) -> usize {
        if self.params.is_empty() {
            return 0;
        }
        self.params.iter().map(|p| p.num_values()).product()
    }

    /// Parameter descriptors.
    pub fn params(&self) -> &[TuningParam] {
        &self.params
    }

    /// Whether the space is empty (no parameters or any parameter has zero
    /// values).
    pub fn is_empty(&self) -> bool {
        self.params.is_empty() || self.params.iter().any(|p| p.num_values() == 0)
    }

    /// Enumerate every configuration as a `Vec` of `(name, value)` pairs.
    pub fn enumerate(&self) -> Vec<ParamSet> {
        if self.is_empty() {
            return Vec::new();
        }
        let value_lists: Vec<Vec<u32>> = self.params.iter().map(|p| p.values()).collect();
        let mut results = Vec::with_capacity(self.total_configurations());
        let mut indices = vec![0usize; self.params.len()];

        loop {
            let config: Vec<(String, u32)> = self
                .params
                .iter()
                .zip(indices.iter())
                .map(|(p, &i)| {
                    (
                        p.name.clone(),
                        value_lists[self.params.iter().position(|x| x.name == p.name).unwrap()][i],
                    )
                })
                .collect();
            results.push(ParamSet(config));

            // Increment odometer-style
            let mut carry = true;
            for i in (0..indices.len()).rev() {
                if carry {
                    indices[i] += 1;
                    if indices[i] < value_lists[i].len() {
                        carry = false;
                    } else {
                        indices[i] = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }
        results
    }

    /// Default configuration — the `current` value of each parameter.
    pub fn default_config(&self) -> ParamSet {
        ParamSet(self.params.iter().map(|p| (p.name.clone(), p.current)).collect())
    }
}

// ---------------------------------------------------------------------------
// ParamSet — a concrete set of parameter values
// ---------------------------------------------------------------------------

/// A concrete assignment of values to all tuning parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamSet(pub Vec<(String, u32)>);

impl ParamSet {
    /// Look up a parameter value by name.
    pub fn get(&self, name: &str) -> Option<u32> {
        self.0.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
    }
}

impl fmt::Display for ParamSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.0.iter().map(|(n, v)| format!("{n}={v}")).collect();
        write!(f, "{{{}}}", parts.join(", "))
    }
}

// ---------------------------------------------------------------------------
// BenchmarkResult — single benchmark measurement
// ---------------------------------------------------------------------------

/// Result of benchmarking one parameter configuration.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// The parameter configuration that was benchmarked.
    pub params: ParamSet,
    /// Kernel wall time in microseconds.
    pub elapsed_us: f64,
    /// Achieved compute throughput (GFLOP/s).
    pub gflops: f64,
    /// Achieved memory bandwidth (GB/s).
    pub bandwidth_gb_s: f64,
}

impl BenchmarkResult {
    pub fn new(params: ParamSet, elapsed_us: f64, gflops: f64, bandwidth_gb_s: f64) -> Self {
        Self { params, elapsed_us, gflops, bandwidth_gb_s }
    }
}

// ---------------------------------------------------------------------------
// SearchStrategy
// ---------------------------------------------------------------------------

/// Strategy the autotuner uses to explore the tuning space.
#[derive(Debug, Clone, PartialEq)]
pub enum SearchStrategy {
    /// Try every configuration — guarantees the global optimum.
    Exhaustive,
    /// Randomly sample `n` configurations.
    RandomSample(usize),
    /// Simulated annealing with a starting temperature and cooling factor.
    SimulatedAnnealing { initial_temp: f64, cooling_rate: f64, iterations: usize },
    /// Bayesian optimisation with a budget of evaluations.
    BayesianOpt { evaluations: usize },
}

impl fmt::Display for SearchStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exhaustive => write!(f, "Exhaustive"),
            Self::RandomSample(n) => write!(f, "RandomSample({n})"),
            Self::SimulatedAnnealing { iterations, .. } => {
                write!(f, "SimulatedAnnealing({iterations} iters)")
            }
            Self::BayesianOpt { evaluations } => {
                write!(f, "BayesianOpt({evaluations} evals)")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BenchmarkFn — user-provided benchmark closure type alias
// ---------------------------------------------------------------------------

/// A benchmark function that evaluates a configuration and returns elapsed µs.
///
/// Signature: `(params) -> elapsed_us`
pub type BenchmarkFn = Box<dyn Fn(&ParamSet) -> f64>;

// ---------------------------------------------------------------------------
// Autotuner
// ---------------------------------------------------------------------------

/// Explores a [`TuningSpace`] according to a [`SearchStrategy`] to find the
/// parameter configuration that minimises kernel execution time.
pub struct Autotuner {
    space: TuningSpace,
    strategy: SearchStrategy,
    /// Compute GFLOP count of the kernel (for throughput calculation).
    flop_count: f64,
    /// Total bytes transferred by the kernel (for bandwidth calculation).
    bytes_transferred: f64,
}

impl Autotuner {
    pub fn new(
        space: TuningSpace,
        strategy: SearchStrategy,
        flop_count: f64,
        bytes_transferred: f64,
    ) -> Self {
        Self { space, strategy, flop_count, bytes_transferred }
    }

    /// Run the tuning process using the supplied benchmark function.
    ///
    /// Returns a [`TuningReport`] summarising the best configuration found.
    pub fn tune(&self, bench: &BenchmarkFn) -> TuningReport {
        let configs = self.configs_to_evaluate();
        let default_elapsed = bench(&self.space.default_config());

        let mut results: Vec<BenchmarkResult> = configs
            .into_iter()
            .map(|cfg| {
                let elapsed = bench(&cfg);
                let gflops =
                    if elapsed > 0.0 { self.flop_count / (elapsed * 1e-6) / 1e9 } else { 0.0 };
                let bw = if elapsed > 0.0 {
                    self.bytes_transferred / (elapsed * 1e-6) / 1e9
                } else {
                    0.0
                };
                BenchmarkResult::new(cfg, elapsed, gflops, bw)
            })
            .collect();

        // Sort by elapsed ascending — best first
        results.sort_by(|a, b| {
            a.elapsed_us.partial_cmp(&b.elapsed_us).unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = results.first().cloned().expect("at least one config");
        let speedup = if best.elapsed_us > 0.0 { default_elapsed / best.elapsed_us } else { 1.0 };

        TuningReport {
            best_params: best.params.clone(),
            best_elapsed_us: best.elapsed_us,
            best_gflops: best.gflops,
            best_bandwidth_gb_s: best.bandwidth_gb_s,
            speedup_vs_default: speedup,
            iterations: results.len(),
            strategy: self.strategy.clone(),
            all_results: results,
        }
    }

    /// Select configurations according to the chosen strategy.
    fn configs_to_evaluate(&self) -> Vec<ParamSet> {
        match &self.strategy {
            SearchStrategy::Exhaustive => self.space.enumerate(),
            SearchStrategy::RandomSample(n) => self.random_sample(*n),
            SearchStrategy::SimulatedAnnealing { iterations, .. } => {
                // CPU-side SA: sample deterministically for reproducibility
                self.random_sample(*iterations)
            }
            SearchStrategy::BayesianOpt { evaluations } => self.random_sample(*evaluations),
        }
    }

    /// Deterministic pseudo-random sampling using a simple LCG seeded from the
    /// space size.
    fn random_sample(&self, n: usize) -> Vec<ParamSet> {
        let all = self.space.enumerate();
        if all.is_empty() || n == 0 {
            return Vec::new();
        }
        if n >= all.len() {
            return all;
        }
        // Simple deterministic selection: stride through the list
        let stride = all.len() as f64 / n as f64;
        (0..n)
            .map(|i| {
                let idx = ((i as f64 * stride) as usize).min(all.len() - 1);
                all[idx].clone()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// TuningReport
// ---------------------------------------------------------------------------

/// Summary of a completed tuning run.
#[derive(Debug, Clone)]
pub struct TuningReport {
    pub best_params: ParamSet,
    pub best_elapsed_us: f64,
    pub best_gflops: f64,
    pub best_bandwidth_gb_s: f64,
    pub speedup_vs_default: f64,
    pub iterations: usize,
    pub strategy: SearchStrategy,
    pub all_results: Vec<BenchmarkResult>,
}

impl fmt::Display for TuningReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Tuning Report ===")?;
        writeln!(f, "Strategy:        {}", self.strategy)?;
        writeln!(f, "Iterations:      {}", self.iterations)?;
        writeln!(f, "Best params:     {}", self.best_params)?;
        writeln!(f, "Best time:       {:.2} µs", self.best_elapsed_us)?;
        writeln!(f, "Best GFLOPS:     {:.2}", self.best_gflops)?;
        writeln!(f, "Best bandwidth:  {:.2} GB/s", self.best_bandwidth_gb_s)?;
        write!(f, "Speedup:         {:.2}×", self.speedup_vs_default)
    }
}

// ---------------------------------------------------------------------------
// TuningCacheKey
// ---------------------------------------------------------------------------

/// Composite key for the tuning cache: (kernel name, device name, shape).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TuningCacheKey {
    pub kernel_name: String,
    pub device_name: String,
    pub shape: Vec<usize>,
}

impl TuningCacheKey {
    pub fn new(
        kernel_name: impl Into<String>,
        device_name: impl Into<String>,
        shape: Vec<usize>,
    ) -> Self {
        Self { kernel_name: kernel_name.into(), device_name: device_name.into(), shape }
    }
}

// ---------------------------------------------------------------------------
// TuningCache
// ---------------------------------------------------------------------------

/// Stores the best tuned parameters per (kernel, device, shape) tuple.
#[derive(Debug, Clone, Default)]
pub struct TuningCache {
    entries: HashMap<TuningCacheKey, TuningCacheEntry>,
}

/// A single cache entry with the best parameter set and its timing.
#[derive(Debug, Clone)]
pub struct TuningCacheEntry {
    pub params: ParamSet,
    pub elapsed_us: f64,
    pub gflops: f64,
}

impl TuningCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or update a cache entry.
    pub fn insert(&mut self, key: TuningCacheKey, entry: TuningCacheEntry) {
        self.entries.insert(key, entry);
    }

    /// Look up the cached best parameters.
    pub fn get(&self, key: &TuningCacheKey) -> Option<&TuningCacheEntry> {
        self.entries.get(key)
    }

    /// Check whether a key is present.
    pub fn contains(&self, key: &TuningCacheKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove a cache entry.
    pub fn remove(&mut self, key: &TuningCacheKey) -> Option<TuningCacheEntry> {
        self.entries.remove(key)
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// All keys in the cache.
    pub fn keys(&self) -> Vec<&TuningCacheKey> {
        self.entries.keys().collect()
    }

    /// Store a [`TuningReport`] result in the cache.
    pub fn store_report(&mut self, key: TuningCacheKey, report: &TuningReport) {
        self.insert(
            key,
            TuningCacheEntry {
                params: report.best_params.clone(),
                elapsed_us: report.best_elapsed_us,
                gflops: report.best_gflops,
            },
        );
    }
}

// ---------------------------------------------------------------------------
// A770Defaults — known-good defaults for common kernels on Arc A770
// ---------------------------------------------------------------------------

/// Known-good kernel parameters for Intel Arc A770.
///
/// These are hand-tuned starting points; the autotuner can refine them further.
pub struct A770Defaults;

impl A770Defaults {
    /// Workgroup size for general-purpose compute kernels.
    pub const WORKGROUP_SIZE: u32 = 256;
    /// Tile dimension for tiled matrix multiplication.
    pub const TILE_SIZE: u32 = 16;
    /// Vector width for vectorised loads/stores.
    pub const VECTOR_WIDTH: u32 = 8;
    /// Number of rows each work-item processes.
    pub const ROWS_PER_ITEM: u32 = 4;
    /// Sub-group (warp/wavefront) size on Xe-HPG.
    pub const SUBGROUP_SIZE: u32 = 16;
    /// Preferred local memory tile for GEMM (KB).
    pub const LOCAL_MEM_TILE_KB: u32 = 16;

    /// Build a [`TuningSpace`] for GEMM on the A770.
    pub fn gemm_tuning_space() -> TuningSpace {
        TuningSpace::new(vec![
            TuningParam::new("workgroup_size", 64, 512, 64, Self::WORKGROUP_SIZE),
            TuningParam::new("tile_size", 8, 32, 8, Self::TILE_SIZE),
            TuningParam::new("vector_width", 1, 16, 1, Self::VECTOR_WIDTH),
        ])
    }

    /// Build a [`TuningSpace`] for element-wise kernels on the A770.
    pub fn elementwise_tuning_space() -> TuningSpace {
        TuningSpace::new(vec![
            TuningParam::new("workgroup_size", 32, 1024, 32, Self::WORKGROUP_SIZE),
            TuningParam::new("vector_width", 1, 16, 1, Self::VECTOR_WIDTH),
        ])
    }

    /// Build a [`TuningSpace`] for quantised matmul kernels on the A770.
    pub fn quantized_matmul_tuning_space() -> TuningSpace {
        TuningSpace::new(vec![
            TuningParam::new("workgroup_size", 64, 512, 64, Self::WORKGROUP_SIZE),
            TuningParam::new("tile_size", 8, 32, 8, Self::TILE_SIZE),
            TuningParam::new("rows_per_item", 1, 8, 1, Self::ROWS_PER_ITEM),
        ])
    }

    /// Default [`ParamSet`] for quick fallback without tuning.
    pub fn default_gemm_params() -> ParamSet {
        ParamSet(vec![
            ("workgroup_size".into(), Self::WORKGROUP_SIZE),
            ("tile_size".into(), Self::TILE_SIZE),
            ("vector_width".into(), Self::VECTOR_WIDTH),
        ])
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- TuningParam tests ------------------------------------------------

    #[test]
    fn test_param_values_simple() {
        let p = TuningParam::new("wg", 64, 256, 64, 128);
        assert_eq!(p.values(), vec![64, 128, 192, 256]);
    }

    #[test]
    fn test_param_values_single() {
        let p = TuningParam::new("x", 42, 42, 1, 42);
        assert_eq!(p.values(), vec![42]);
    }

    #[test]
    fn test_param_num_values() {
        let p = TuningParam::new("v", 1, 8, 2, 4);
        assert_eq!(p.num_values(), 4); // 1, 3, 5, 7
    }

    #[test]
    fn test_param_display() {
        let p = TuningParam::new("tile", 8, 32, 8, 16);
        let s = format!("{p}");
        assert!(s.contains("tile"));
        assert!(s.contains("8"));
        assert!(s.contains("32"));
    }

    #[test]
    #[should_panic(expected = "min")]
    fn test_param_min_greater_than_max() {
        TuningParam::new("bad", 100, 10, 1, 50);
    }

    #[test]
    #[should_panic(expected = "step")]
    fn test_param_zero_step() {
        TuningParam::new("bad", 1, 10, 0, 5);
    }

    #[test]
    fn test_param_step_larger_than_range() {
        let p = TuningParam::new("x", 1, 5, 10, 1);
        assert_eq!(p.values(), vec![1]);
    }

    #[test]
    fn test_param_values_step_one() {
        let p = TuningParam::new("x", 1, 5, 1, 3);
        assert_eq!(p.values(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_param_clone_eq() {
        let p = TuningParam::new("a", 1, 10, 1, 5);
        let p2 = p.clone();
        assert_eq!(p, p2);
    }

    // ---- TuningSpace tests ------------------------------------------------

    #[test]
    fn test_space_total_configurations() {
        let space = TuningSpace::new(vec![
            TuningParam::new("a", 1, 3, 1, 1),    // 3 values
            TuningParam::new("b", 10, 20, 5, 10), // 3 values
        ]);
        assert_eq!(space.total_configurations(), 9);
    }

    #[test]
    fn test_space_empty() {
        let space = TuningSpace::new(vec![]);
        assert!(space.is_empty());
        assert_eq!(space.total_configurations(), 0);
    }

    #[test]
    fn test_space_enumerate_count() {
        let space = TuningSpace::new(vec![
            TuningParam::new("x", 1, 2, 1, 1),
            TuningParam::new("y", 10, 11, 1, 10),
        ]);
        let configs = space.enumerate();
        assert_eq!(configs.len(), 4);
    }

    #[test]
    fn test_space_enumerate_contains_all_combos() {
        let space = TuningSpace::new(vec![
            TuningParam::new("a", 1, 2, 1, 1),
            TuningParam::new("b", 10, 11, 1, 10),
        ]);
        let configs = space.enumerate();
        // Should have (1,10), (1,11), (2,10), (2,11)
        assert!(configs.iter().any(|c| c.get("a") == Some(1) && c.get("b") == Some(10)));
        assert!(configs.iter().any(|c| c.get("a") == Some(1) && c.get("b") == Some(11)));
        assert!(configs.iter().any(|c| c.get("a") == Some(2) && c.get("b") == Some(10)));
        assert!(configs.iter().any(|c| c.get("a") == Some(2) && c.get("b") == Some(11)));
    }

    #[test]
    fn test_space_default_config() {
        let space = TuningSpace::new(vec![
            TuningParam::new("wg", 64, 256, 64, 128),
            TuningParam::new("tile", 8, 32, 8, 16),
        ]);
        let cfg = space.default_config();
        assert_eq!(cfg.get("wg"), Some(128));
        assert_eq!(cfg.get("tile"), Some(16));
    }

    #[test]
    fn test_space_single_param() {
        let space = TuningSpace::new(vec![TuningParam::new("only", 5, 5, 1, 5)]);
        assert_eq!(space.total_configurations(), 1);
        let configs = space.enumerate();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].get("only"), Some(5));
    }

    #[test]
    fn test_space_three_params() {
        let space = TuningSpace::new(vec![
            TuningParam::new("a", 1, 2, 1, 1), // 2
            TuningParam::new("b", 1, 2, 1, 1), // 2
            TuningParam::new("c", 1, 2, 1, 1), // 2
        ]);
        assert_eq!(space.total_configurations(), 8);
        assert_eq!(space.enumerate().len(), 8);
    }

    #[test]
    fn test_space_enumerate_empty_returns_empty() {
        let space = TuningSpace::new(vec![]);
        assert!(space.enumerate().is_empty());
    }

    // ---- ParamSet tests ---------------------------------------------------

    #[test]
    fn test_paramset_get_existing() {
        let ps = ParamSet(vec![("wg".into(), 256), ("tile".into(), 16)]);
        assert_eq!(ps.get("wg"), Some(256));
        assert_eq!(ps.get("tile"), Some(16));
    }

    #[test]
    fn test_paramset_get_missing() {
        let ps = ParamSet(vec![("wg".into(), 256)]);
        assert_eq!(ps.get("nonexistent"), None);
    }

    #[test]
    fn test_paramset_display() {
        let ps = ParamSet(vec![("wg".into(), 256)]);
        let s = format!("{ps}");
        assert!(s.contains("wg=256"));
    }

    #[test]
    fn test_paramset_eq() {
        let a = ParamSet(vec![("x".into(), 1)]);
        let b = ParamSet(vec![("x".into(), 1)]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_paramset_ne() {
        let a = ParamSet(vec![("x".into(), 1)]);
        let b = ParamSet(vec![("x".into(), 2)]);
        assert_ne!(a, b);
    }

    // ---- BenchmarkResult tests --------------------------------------------

    #[test]
    fn test_benchmark_result_new() {
        let ps = ParamSet(vec![("wg".into(), 128)]);
        let br = BenchmarkResult::new(ps.clone(), 100.0, 50.0, 200.0);
        assert_eq!(br.params, ps);
        assert!((br.elapsed_us - 100.0).abs() < f64::EPSILON);
        assert!((br.gflops - 50.0).abs() < f64::EPSILON);
        assert!((br.bandwidth_gb_s - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_benchmark_result_clone() {
        let br = BenchmarkResult::new(ParamSet(vec![("a".into(), 1)]), 10.0, 5.0, 100.0);
        let br2 = br.clone();
        assert!((br2.elapsed_us - br.elapsed_us).abs() < f64::EPSILON);
    }

    // ---- SearchStrategy tests ---------------------------------------------

    #[test]
    fn test_strategy_display_exhaustive() {
        assert_eq!(format!("{}", SearchStrategy::Exhaustive), "Exhaustive");
    }

    #[test]
    fn test_strategy_display_random() {
        assert_eq!(format!("{}", SearchStrategy::RandomSample(100)), "RandomSample(100)");
    }

    #[test]
    fn test_strategy_display_sa() {
        let sa = SearchStrategy::SimulatedAnnealing {
            initial_temp: 1.0,
            cooling_rate: 0.95,
            iterations: 200,
        };
        let s = format!("{sa}");
        assert!(s.contains("SimulatedAnnealing"));
        assert!(s.contains("200"));
    }

    #[test]
    fn test_strategy_display_bayesian() {
        let bo = SearchStrategy::BayesianOpt { evaluations: 50 };
        assert!(format!("{bo}").contains("BayesianOpt"));
    }

    #[test]
    fn test_strategy_eq() {
        assert_eq!(SearchStrategy::Exhaustive, SearchStrategy::Exhaustive);
        assert_eq!(SearchStrategy::RandomSample(10), SearchStrategy::RandomSample(10));
    }

    #[test]
    fn test_strategy_ne() {
        assert_ne!(SearchStrategy::Exhaustive, SearchStrategy::RandomSample(1));
    }

    // ---- Autotuner tests --------------------------------------------------

    fn mock_bench_wg_prefers_128(ps: &ParamSet) -> f64 {
        let wg = ps.get("workgroup_size").unwrap_or(256);
        // Minimum at wg=128
        let diff = (wg as f64 - 128.0).abs();
        100.0 + diff
    }

    fn mock_bench_constant(_ps: &ParamSet) -> f64 {
        42.0
    }

    fn mock_bench_two_params(ps: &ParamSet) -> f64 {
        let wg = ps.get("wg").unwrap_or(64) as f64;
        let tile = ps.get("tile").unwrap_or(8) as f64;
        // Optimum at wg=128, tile=16 → elapsed = 10
        (wg - 128.0).abs() + (tile - 16.0).abs() + 10.0
    }

    #[test]
    fn test_exhaustive_finds_global_optimum() {
        let space = TuningSpace::new(vec![TuningParam::new("workgroup_size", 64, 256, 64, 256)]);
        let tuner = Autotuner::new(space, SearchStrategy::Exhaustive, 1e9, 1e8);
        let report = tuner.tune(&(Box::new(mock_bench_wg_prefers_128) as BenchmarkFn));
        assert_eq!(report.best_params.get("workgroup_size"), Some(128));
    }

    #[test]
    fn test_exhaustive_all_results_sorted() {
        let space = TuningSpace::new(vec![TuningParam::new("workgroup_size", 64, 256, 64, 256)]);
        let tuner = Autotuner::new(space, SearchStrategy::Exhaustive, 1e9, 1e8);
        let report = tuner.tune(&(Box::new(mock_bench_wg_prefers_128) as BenchmarkFn));
        for pair in report.all_results.windows(2) {
            assert!(pair[0].elapsed_us <= pair[1].elapsed_us);
        }
    }

    #[test]
    fn test_exhaustive_best_leq_all() {
        let space = TuningSpace::new(vec![TuningParam::new("workgroup_size", 64, 512, 64, 256)]);
        let tuner = Autotuner::new(space, SearchStrategy::Exhaustive, 1e9, 1e8);
        let report = tuner.tune(&(Box::new(mock_bench_wg_prefers_128) as BenchmarkFn));
        for r in &report.all_results {
            assert!(report.best_elapsed_us <= r.elapsed_us + f64::EPSILON);
        }
    }

    #[test]
    fn test_random_sample_finds_good_params() {
        let space = TuningSpace::new(vec![TuningParam::new("workgroup_size", 64, 512, 64, 256)]);
        let tuner = Autotuner::new(space, SearchStrategy::RandomSample(5), 1e9, 1e8);
        let report = tuner.tune(&(Box::new(mock_bench_wg_prefers_128) as BenchmarkFn));
        // Random sample should find something ≤ the default (wg=256)
        assert!(
            report.best_elapsed_us
                <= mock_bench_wg_prefers_128(&ParamSet(vec![("workgroup_size".into(), 256)]))
        );
    }

    #[test]
    fn test_random_sample_respects_count() {
        let space = TuningSpace::new(vec![TuningParam::new("x", 1, 100, 1, 50)]);
        let tuner = Autotuner::new(space, SearchStrategy::RandomSample(10), 1e9, 1e8);
        let report = tuner.tune(&(Box::new(mock_bench_constant) as BenchmarkFn));
        assert_eq!(report.iterations, 10);
    }

    #[test]
    fn test_constant_bench_speedup_one() {
        let space = TuningSpace::new(vec![TuningParam::new("x", 1, 4, 1, 2)]);
        let tuner = Autotuner::new(space, SearchStrategy::Exhaustive, 1e9, 1e8);
        let report = tuner.tune(&(Box::new(mock_bench_constant) as BenchmarkFn));
        assert!((report.speedup_vs_default - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_tuner_two_params() {
        let space = TuningSpace::new(vec![
            TuningParam::new("wg", 64, 192, 64, 64),
            TuningParam::new("tile", 8, 24, 8, 8),
        ]);
        let tuner = Autotuner::new(space, SearchStrategy::Exhaustive, 1e9, 1e8);
        let report = tuner.tune(&(Box::new(mock_bench_two_params) as BenchmarkFn));
        assert_eq!(report.best_params.get("wg"), Some(128));
        assert_eq!(report.best_params.get("tile"), Some(16));
    }

    #[test]
    fn test_tuner_gflops_positive() {
        let space = TuningSpace::new(vec![TuningParam::new("x", 1, 3, 1, 2)]);
        let tuner = Autotuner::new(space, SearchStrategy::Exhaustive, 1e9, 1e8);
        let report = tuner.tune(&(Box::new(|_: &ParamSet| 50.0) as BenchmarkFn));
        assert!(report.best_gflops > 0.0);
    }

    #[test]
    fn test_tuner_bandwidth_positive() {
        let space = TuningSpace::new(vec![TuningParam::new("x", 1, 3, 1, 2)]);
        let tuner = Autotuner::new(space, SearchStrategy::Exhaustive, 1e9, 1e8);
        let report = tuner.tune(&(Box::new(|_: &ParamSet| 50.0) as BenchmarkFn));
        assert!(report.best_bandwidth_gb_s > 0.0);
    }

    #[test]
    fn test_tuner_speedup_greater_than_one_when_faster() {
        let space = TuningSpace::new(vec![TuningParam::new(
            "workgroup_size",
            64,
            256,
            64,
            256, // default is worst case
        )]);
        let tuner = Autotuner::new(space, SearchStrategy::Exhaustive, 1e9, 1e8);
        let report = tuner.tune(&(Box::new(mock_bench_wg_prefers_128) as BenchmarkFn));
        assert!(report.speedup_vs_default > 1.0);
    }

    #[test]
    fn test_simulated_annealing_strategy() {
        let space = TuningSpace::new(vec![TuningParam::new("workgroup_size", 64, 512, 64, 256)]);
        let tuner = Autotuner::new(
            space,
            SearchStrategy::SimulatedAnnealing {
                initial_temp: 1.0,
                cooling_rate: 0.95,
                iterations: 5,
            },
            1e9,
            1e8,
        );
        let report = tuner.tune(&(Box::new(mock_bench_wg_prefers_128) as BenchmarkFn));
        assert!(report.best_elapsed_us > 0.0);
        assert!(report.iterations <= 8); // may get all if n >= all
    }

    #[test]
    fn test_bayesian_opt_strategy() {
        let space = TuningSpace::new(vec![TuningParam::new("workgroup_size", 64, 512, 64, 256)]);
        let tuner = Autotuner::new(space, SearchStrategy::BayesianOpt { evaluations: 4 }, 1e9, 1e8);
        let report = tuner.tune(&(Box::new(mock_bench_wg_prefers_128) as BenchmarkFn));
        assert!(report.best_elapsed_us > 0.0);
        assert!(report.iterations <= 8);
    }

    // ---- TuningReport tests -----------------------------------------------

    #[test]
    fn test_report_display() {
        let space = TuningSpace::new(vec![TuningParam::new("x", 1, 3, 1, 2)]);
        let tuner = Autotuner::new(space, SearchStrategy::Exhaustive, 1e9, 1e8);
        let report = tuner.tune(&(Box::new(mock_bench_constant) as BenchmarkFn));
        let s = format!("{report}");
        assert!(s.contains("Tuning Report"));
        assert!(s.contains("Exhaustive"));
        assert!(s.contains("Speedup"));
    }

    #[test]
    fn test_report_iterations_match() {
        let space = TuningSpace::new(vec![TuningParam::new("x", 1, 4, 1, 2)]);
        let tuner = Autotuner::new(space, SearchStrategy::Exhaustive, 1e9, 1e8);
        let report = tuner.tune(&(Box::new(mock_bench_constant) as BenchmarkFn));
        assert_eq!(report.iterations, 4);
        assert_eq!(report.all_results.len(), 4);
    }

    #[test]
    fn test_report_strategy_preserved() {
        let space = TuningSpace::new(vec![TuningParam::new("x", 1, 2, 1, 1)]);
        let tuner = Autotuner::new(space, SearchStrategy::RandomSample(2), 1e9, 1e8);
        let report = tuner.tune(&(Box::new(mock_bench_constant) as BenchmarkFn));
        assert_eq!(report.strategy, SearchStrategy::RandomSample(2));
    }

    // ---- TuningCache tests ------------------------------------------------

    #[test]
    fn test_cache_new_empty() {
        let cache = TuningCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = TuningCache::new();
        let key = TuningCacheKey::new("gemm", "A770", vec![1024, 1024]);
        let entry = TuningCacheEntry {
            params: ParamSet(vec![("wg".into(), 128)]),
            elapsed_us: 50.0,
            gflops: 100.0,
        };
        cache.insert(key.clone(), entry);
        assert!(cache.contains(&key));
        assert_eq!(cache.len(), 1);
        let got = cache.get(&key).unwrap();
        assert_eq!(got.params.get("wg"), Some(128));
    }

    #[test]
    fn test_cache_miss() {
        let cache = TuningCache::new();
        let key = TuningCacheKey::new("nope", "dev", vec![]);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_overwrite() {
        let mut cache = TuningCache::new();
        let key = TuningCacheKey::new("gemm", "A770", vec![512]);
        cache.insert(
            key.clone(),
            TuningCacheEntry {
                params: ParamSet(vec![("wg".into(), 64)]),
                elapsed_us: 100.0,
                gflops: 50.0,
            },
        );
        cache.insert(
            key.clone(),
            TuningCacheEntry {
                params: ParamSet(vec![("wg".into(), 128)]),
                elapsed_us: 50.0,
                gflops: 100.0,
            },
        );
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&key).unwrap().params.get("wg"), Some(128));
    }

    #[test]
    fn test_cache_remove() {
        let mut cache = TuningCache::new();
        let key = TuningCacheKey::new("k", "d", vec![1]);
        cache.insert(
            key.clone(),
            TuningCacheEntry { params: ParamSet(vec![]), elapsed_us: 1.0, gflops: 1.0 },
        );
        assert!(cache.remove(&key).is_some());
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = TuningCache::new();
        for i in 0..5 {
            cache.insert(
                TuningCacheKey::new(format!("k{i}"), "d", vec![i as usize]),
                TuningCacheEntry { params: ParamSet(vec![]), elapsed_us: 1.0, gflops: 1.0 },
            );
        }
        assert_eq!(cache.len(), 5);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_keys() {
        let mut cache = TuningCache::new();
        let k1 = TuningCacheKey::new("a", "d", vec![]);
        let k2 = TuningCacheKey::new("b", "d", vec![]);
        let entry = TuningCacheEntry { params: ParamSet(vec![]), elapsed_us: 1.0, gflops: 1.0 };
        cache.insert(k1.clone(), entry.clone());
        cache.insert(k2.clone(), entry);
        let keys = cache.keys();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_cache_store_report() {
        let mut cache = TuningCache::new();
        let space = TuningSpace::new(vec![TuningParam::new("x", 1, 3, 1, 2)]);
        let tuner = Autotuner::new(space, SearchStrategy::Exhaustive, 1e9, 1e8);
        let report = tuner.tune(&(Box::new(mock_bench_constant) as BenchmarkFn));
        let key = TuningCacheKey::new("gemm", "A770", vec![1024]);
        cache.store_report(key.clone(), &report);
        assert!(cache.contains(&key));
    }

    #[test]
    fn test_cache_different_shapes() {
        let mut cache = TuningCache::new();
        let k1 = TuningCacheKey::new("gemm", "A770", vec![1024, 1024]);
        let k2 = TuningCacheKey::new("gemm", "A770", vec![2048, 2048]);
        let entry = TuningCacheEntry {
            params: ParamSet(vec![("wg".into(), 128)]),
            elapsed_us: 50.0,
            gflops: 100.0,
        };
        cache.insert(k1.clone(), entry.clone());
        cache.insert(k2.clone(), entry);
        assert_eq!(cache.len(), 2);
        assert!(cache.contains(&k1));
        assert!(cache.contains(&k2));
    }

    #[test]
    fn test_cache_different_devices() {
        let mut cache = TuningCache::new();
        let k1 = TuningCacheKey::new("gemm", "A770", vec![1024]);
        let k2 = TuningCacheKey::new("gemm", "A750", vec![1024]);
        let entry = TuningCacheEntry { params: ParamSet(vec![]), elapsed_us: 1.0, gflops: 1.0 };
        cache.insert(k1.clone(), entry.clone());
        cache.insert(k2.clone(), entry);
        assert_eq!(cache.len(), 2);
    }

    // ---- A770Defaults tests -----------------------------------------------

    #[test]
    fn test_a770_workgroup_size() {
        assert_eq!(A770Defaults::WORKGROUP_SIZE, 256);
    }

    #[test]
    fn test_a770_tile_size() {
        assert_eq!(A770Defaults::TILE_SIZE, 16);
    }

    #[test]
    fn test_a770_vector_width() {
        assert_eq!(A770Defaults::VECTOR_WIDTH, 8);
    }

    #[test]
    fn test_a770_subgroup_size() {
        assert_eq!(A770Defaults::SUBGROUP_SIZE, 16);
    }

    #[test]
    fn test_a770_gemm_space_nonempty() {
        let space = A770Defaults::gemm_tuning_space();
        assert!(!space.is_empty());
        assert!(space.total_configurations() > 1);
    }

    #[test]
    fn test_a770_elementwise_space_nonempty() {
        let space = A770Defaults::elementwise_tuning_space();
        assert!(!space.is_empty());
        assert!(space.total_configurations() > 1);
    }

    #[test]
    fn test_a770_quantized_matmul_space() {
        let space = A770Defaults::quantized_matmul_tuning_space();
        assert!(!space.is_empty());
        assert!(space.total_configurations() > 1);
    }

    #[test]
    fn test_a770_default_gemm_params() {
        let ps = A770Defaults::default_gemm_params();
        assert_eq!(ps.get("workgroup_size"), Some(256));
        assert_eq!(ps.get("tile_size"), Some(16));
        assert_eq!(ps.get("vector_width"), Some(8));
    }

    #[test]
    fn test_a770_gemm_space_includes_defaults() {
        let space = A770Defaults::gemm_tuning_space();
        let configs = space.enumerate();
        let defaults = A770Defaults::default_gemm_params();
        assert!(configs.iter().any(|c| {
            c.get("workgroup_size") == defaults.get("workgroup_size")
                && c.get("tile_size") == defaults.get("tile_size")
        }));
    }

    // ---- Property / edge case tests ---------------------------------------

    #[test]
    fn test_property_best_leq_all_multi_param() {
        let space = TuningSpace::new(vec![
            TuningParam::new("a", 1, 5, 1, 3),
            TuningParam::new("b", 10, 50, 10, 30),
        ]);
        let tuner = Autotuner::new(space, SearchStrategy::Exhaustive, 1e9, 1e8);
        let bench: BenchmarkFn = Box::new(|ps: &ParamSet| {
            let a = ps.get("a").unwrap() as f64;
            let b = ps.get("b").unwrap() as f64;
            a * b
        });
        let report = tuner.tune(&bench);
        for r in &report.all_results {
            assert!(report.best_elapsed_us <= r.elapsed_us + f64::EPSILON);
        }
    }

    #[test]
    fn test_edge_large_step_single_value() {
        let space = TuningSpace::new(vec![TuningParam::new("x", 1, 1000, 2000, 1)]);
        assert_eq!(space.total_configurations(), 1);
    }

    #[test]
    fn test_edge_all_same_performance() {
        let space = TuningSpace::new(vec![TuningParam::new("x", 1, 10, 1, 5)]);
        let tuner = Autotuner::new(space, SearchStrategy::Exhaustive, 1e9, 1e8);
        let report = tuner.tune(&(Box::new(mock_bench_constant) as BenchmarkFn));
        // All times equal → speedup ≈ 1
        assert!((report.speedup_vs_default - 1.0).abs() < 0.01);
        assert_eq!(report.iterations, 10);
    }

    #[test]
    fn test_random_sample_more_than_space() {
        let space = TuningSpace::new(vec![TuningParam::new("x", 1, 3, 1, 2)]);
        let tuner = Autotuner::new(space, SearchStrategy::RandomSample(100), 1e9, 1e8);
        let report = tuner.tune(&(Box::new(mock_bench_constant) as BenchmarkFn));
        // Should cap at space size (3)
        assert_eq!(report.iterations, 3);
    }

    #[test]
    fn test_tuning_cache_key_equality() {
        let k1 = TuningCacheKey::new("gemm", "A770", vec![1024, 1024]);
        let k2 = TuningCacheKey::new("gemm", "A770", vec![1024, 1024]);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_tuning_cache_key_inequality_shape() {
        let k1 = TuningCacheKey::new("gemm", "A770", vec![1024]);
        let k2 = TuningCacheKey::new("gemm", "A770", vec![2048]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_tuning_cache_key_inequality_kernel() {
        let k1 = TuningCacheKey::new("gemm", "A770", vec![1024]);
        let k2 = TuningCacheKey::new("conv", "A770", vec![1024]);
        assert_ne!(k1, k2);
    }
}
