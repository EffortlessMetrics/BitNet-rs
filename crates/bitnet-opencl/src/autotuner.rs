//! Auto-tuning framework for GPU kernels.
//!
//! Benchmarks all kernel variants across a parameter space (work-group size,
//! tile size, vector width), persists results to a JSON cache file, and
//! maintains device-specific tuning profiles.

use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Parameter space
// ---------------------------------------------------------------------------

/// A single point in the kernel parameter space.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KernelParams {
    /// OpenCL work-group size (e.g. 64, 128, 256).
    pub work_group_size: u32,
    /// 2-D tile dimension used by tiled matmul (e.g. 8, 16, 32).
    pub tile_size: u32,
    /// SIMD vector width (1, 4, 8, 16).
    pub vector_width: u32,
}

/// Defines the search space for auto-tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSpace {
    pub work_group_sizes: Vec<u32>,
    pub tile_sizes: Vec<u32>,
    pub vector_widths: Vec<u32>,
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self {
            work_group_sizes: vec![64, 128, 256],
            tile_sizes: vec![8, 16, 32],
            vector_widths: vec![1, 4, 8],
        }
    }
}

impl ParameterSpace {
    /// Iterator over every combination in the space.
    pub fn iter(&self) -> impl Iterator<Item = KernelParams> + '_ {
        self.work_group_sizes.iter().flat_map(move |&wg| {
            self.tile_sizes.iter().flat_map(move |&ts| {
                self.vector_widths
                    .iter()
                    .map(move |&vw| KernelParams {
                        work_group_size: wg,
                        tile_size: ts,
                        vector_width: vw,
                    })
            })
        })
    }

    /// Total number of combinations.
    pub fn total_combinations(&self) -> usize {
        self.work_group_sizes.len() * self.tile_sizes.len() * self.vector_widths.len()
    }
}

// ---------------------------------------------------------------------------
// Benchmark result
// ---------------------------------------------------------------------------

/// Timing result for a single kernel parameter combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub params: KernelParams,
    /// Median execution time across measurement iterations.
    pub median_time: Duration,
    /// Minimum execution time observed.
    pub min_time: Duration,
    /// Maximum execution time observed.
    pub max_time: Duration,
    /// Number of measurement iterations.
    pub iterations: u32,
    /// Whether the kernel produced correct results.
    pub valid: bool,
}

// ---------------------------------------------------------------------------
// Device profile
// ---------------------------------------------------------------------------

/// A device-specific tuning profile containing the best kernel parameters
/// found for each kernel name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceProfile {
    /// Device name string (e.g. "Intel Arc A770").
    pub device_name: String,
    /// Driver/runtime version at the time of tuning.
    pub driver_version: String,
    /// Best parameters for each kernel by name.
    pub best_params: HashMap<String, KernelParams>,
    /// Full benchmark results per kernel.
    pub results: HashMap<String, Vec<BenchmarkResult>>,
    /// Timestamp when the profile was created.
    pub created_at: String,
}

impl DeviceProfile {
    pub fn new(device_name: impl Into<String>, driver_version: impl Into<String>) -> Self {
        Self {
            device_name: device_name.into(),
            driver_version: driver_version.into(),
            best_params: HashMap::new(),
            results: HashMap::new(),
            created_at: chrono_now_iso(),
        }
    }

    /// Record a full set of benchmark results for `kernel_name` and pick the
    /// best (lowest median time among valid results).
    pub fn record_results(&mut self, kernel_name: &str, results: Vec<BenchmarkResult>) {
        let best = results
            .iter()
            .filter(|r| r.valid)
            .min_by_key(|r| r.median_time)
            .map(|r| r.params.clone());

        if let Some(params) = best {
            info!(
                "Best params for '{}': wg={}, tile={}, vec={}",
                kernel_name, params.work_group_size, params.tile_size, params.vector_width
            );
            self.best_params.insert(kernel_name.to_owned(), params);
        } else {
            warn!("No valid results for kernel '{}'", kernel_name);
        }
        self.results.insert(kernel_name.to_owned(), results);
    }

    /// Get the best parameters for a specific kernel, if tuned.
    pub fn best_for(&self, kernel_name: &str) -> Option<&KernelParams> {
        self.best_params.get(kernel_name)
    }
}

fn chrono_now_iso() -> String {
    use std::time::SystemTime;
    let d = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("epoch:{}", d.as_secs())
}

// ---------------------------------------------------------------------------
// Results persistence (JSON cache)
// ---------------------------------------------------------------------------

/// Cache file containing one or more device profiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningCache {
    /// Schema version for forward-compatibility.
    pub version: String,
    /// Map of device_name → profile.
    pub profiles: HashMap<String, DeviceProfile>,
}

impl Default for TuningCache {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_owned(),
            profiles: HashMap::new(),
        }
    }
}

impl TuningCache {
    /// Load from a JSON file, returning an empty cache on any error.
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(json) => serde_json::from_str(&json).unwrap_or_else(|e| {
                warn!("Corrupt tuning cache {}: {}", path.display(), e);
                Self::default()
            }),
            Err(_) => {
                debug!("No tuning cache at {}, starting fresh", path.display());
                Self::default()
            }
        }
    }

    /// Persist to a JSON file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, json)?;
        info!("Saved tuning cache to {}", path.display());
        Ok(())
    }

    /// Upsert a device profile.
    pub fn upsert_profile(&mut self, profile: DeviceProfile) {
        self.profiles.insert(profile.device_name.clone(), profile);
    }

    /// Retrieve a profile by device name.
    pub fn get_profile(&self, device_name: &str) -> Option<&DeviceProfile> {
        self.profiles.get(device_name)
    }
}

// ---------------------------------------------------------------------------
// Autotuner
// ---------------------------------------------------------------------------

/// A closure that runs a kernel with the given parameters and returns the
/// wall-clock execution time.  Returns `None` if the parameters are invalid
/// (e.g. unsupported work-group size).
pub type KernelBenchFn = Box<dyn Fn(&KernelParams) -> Option<Duration>>;

/// Auto-tuner that benchmarks all kernel variants in a parameter space.
pub struct Autotuner {
    parameter_space: ParameterSpace,
    cache_path: Option<std::path::PathBuf>,
    cache: TuningCache,
    warmup_iterations: u32,
    bench_iterations: u32,
}

impl std::fmt::Debug for Autotuner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Autotuner")
            .field("parameter_space", &self.parameter_space)
            .field("cache_path", &self.cache_path)
            .field("warmup_iterations", &self.warmup_iterations)
            .field("bench_iterations", &self.bench_iterations)
            .finish()
    }
}

impl Autotuner {
    /// Create a new auto-tuner with default parameter space.
    pub fn new() -> Self {
        Self {
            parameter_space: ParameterSpace::default(),
            cache_path: None,
            cache: TuningCache::default(),
            warmup_iterations: 3,
            bench_iterations: 5,
        }
    }

    /// Override the parameter space.
    pub fn with_parameter_space(mut self, space: ParameterSpace) -> Self {
        self.parameter_space = space;
        self
    }

    /// Set the path for the JSON cache file and load existing results.
    pub fn with_cache_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        let p = path.into();
        self.cache = TuningCache::load(&p);
        self.cache_path = Some(p);
        self
    }

    /// Set the number of warmup iterations (discarded).
    pub fn with_warmup(mut self, n: u32) -> Self {
        self.warmup_iterations = n;
        self
    }

    /// Set the number of benchmark iterations.
    pub fn with_iterations(mut self, n: u32) -> Self {
        self.bench_iterations = n;
        self
    }

    /// Total combinations in the current parameter space.
    pub fn total_combinations(&self) -> usize {
        self.parameter_space.total_combinations()
    }

    /// Run the auto-tuner for `kernel_name` using the provided benchmark
    /// closure.  Returns the list of results (also stored in the profile).
    pub fn tune(
        &mut self,
        kernel_name: &str,
        device_name: &str,
        driver_version: &str,
        bench_fn: &KernelBenchFn,
    ) -> Vec<BenchmarkResult> {
        info!(
            "Auto-tuning '{}' on '{}': {} combinations, {} warmup + {} bench iters",
            kernel_name,
            device_name,
            self.parameter_space.total_combinations(),
            self.warmup_iterations,
            self.bench_iterations,
        );

        let mut results = Vec::new();

        for params in self.parameter_space.iter() {
            // Warmup
            for _ in 0..self.warmup_iterations {
                let _ = bench_fn(&params);
            }

            // Benchmark
            let mut times = Vec::new();
            let mut valid = true;
            for _ in 0..self.bench_iterations {
                match bench_fn(&params) {
                    Some(t) => times.push(t),
                    None => {
                        valid = false;
                        break;
                    }
                }
            }

            if times.is_empty() {
                results.push(BenchmarkResult {
                    params,
                    median_time: Duration::MAX,
                    min_time: Duration::MAX,
                    max_time: Duration::MAX,
                    iterations: 0,
                    valid: false,
                });
                continue;
            }

            times.sort();
            let median = times[times.len() / 2];
            let min_t = times[0];
            let max_t = times[times.len() - 1];

            debug!(
                "  wg={} tile={} vec={} → median={:?}{}",
                params.work_group_size,
                params.tile_size,
                params.vector_width,
                median,
                if valid { "" } else { " (INVALID)" }
            );

            results.push(BenchmarkResult {
                params,
                median_time: median,
                min_time: min_t,
                max_time: max_t,
                iterations: times.len() as u32,
                valid,
            });
        }

        // Store in profile
        let profile = self
            .cache
            .profiles
            .entry(device_name.to_owned())
            .or_insert_with(|| DeviceProfile::new(device_name, driver_version));
        profile.record_results(kernel_name, results.clone());

        // Persist
        if let Some(ref path) = self.cache_path {
            if let Err(e) = self.cache.save(path) {
                warn!("Failed to save tuning cache: {}", e);
            }
        }

        results
    }

    /// Look up the best parameters from the cache without re-tuning.
    pub fn cached_best(
        &self,
        device_name: &str,
        kernel_name: &str,
    ) -> Option<&KernelParams> {
        self.cache
            .get_profile(device_name)
            .and_then(|p| p.best_for(kernel_name))
    }

    /// Access the underlying cache (read-only).
    pub fn cache(&self) -> &TuningCache {
        &self.cache
    }
}

impl Default for Autotuner {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    // -- ParameterSpace tests --

    #[test]
    fn test_default_parameter_space_combinations() {
        let space = ParameterSpace::default();
        // 3 work_group × 3 tile × 3 vector = 27
        assert_eq!(space.total_combinations(), 27);
    }

    #[test]
    fn test_parameter_space_iter_count() {
        let space = ParameterSpace {
            work_group_sizes: vec![64, 128],
            tile_sizes: vec![8],
            vector_widths: vec![4, 8],
        };
        let items: Vec<_> = space.iter().collect();
        assert_eq!(items.len(), 4); // 2 × 1 × 2
    }

    #[test]
    fn test_parameter_space_iter_values() {
        let space = ParameterSpace {
            work_group_sizes: vec![128],
            tile_sizes: vec![16],
            vector_widths: vec![4],
        };
        let items: Vec<_> = space.iter().collect();
        assert_eq!(items.len(), 1);
        assert_eq!(
            items[0],
            KernelParams {
                work_group_size: 128,
                tile_size: 16,
                vector_width: 4,
            }
        );
    }

    // -- DeviceProfile tests --

    #[test]
    fn test_device_profile_best_selection() {
        let mut profile = DeviceProfile::new("TestGPU", "1.0");
        let results = vec![
            BenchmarkResult {
                params: KernelParams {
                    work_group_size: 64,
                    tile_size: 8,
                    vector_width: 4,
                },
                median_time: Duration::from_millis(10),
                min_time: Duration::from_millis(9),
                max_time: Duration::from_millis(11),
                iterations: 5,
                valid: true,
            },
            BenchmarkResult {
                params: KernelParams {
                    work_group_size: 128,
                    tile_size: 16,
                    vector_width: 8,
                },
                median_time: Duration::from_millis(5),
                min_time: Duration::from_millis(4),
                max_time: Duration::from_millis(6),
                iterations: 5,
                valid: true,
            },
        ];
        profile.record_results("matmul", results);
        let best = profile.best_for("matmul").unwrap();
        assert_eq!(best.work_group_size, 128);
        assert_eq!(best.tile_size, 16);
    }

    #[test]
    fn test_device_profile_skips_invalid() {
        let mut profile = DeviceProfile::new("TestGPU", "1.0");
        let results = vec![
            BenchmarkResult {
                params: KernelParams {
                    work_group_size: 64,
                    tile_size: 8,
                    vector_width: 4,
                },
                median_time: Duration::from_millis(1),
                min_time: Duration::from_millis(1),
                max_time: Duration::from_millis(1),
                iterations: 5,
                valid: false, // invalid!
            },
            BenchmarkResult {
                params: KernelParams {
                    work_group_size: 128,
                    tile_size: 16,
                    vector_width: 8,
                },
                median_time: Duration::from_millis(10),
                min_time: Duration::from_millis(9),
                max_time: Duration::from_millis(11),
                iterations: 5,
                valid: true,
            },
        ];
        profile.record_results("matmul", results);
        let best = profile.best_for("matmul").unwrap();
        assert_eq!(best.work_group_size, 128);
    }

    // -- TuningCache tests --

    #[test]
    fn test_cache_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.json");

        let mut cache = TuningCache::default();
        let mut profile = DeviceProfile::new("Arc A770", "23.1");
        profile.best_params.insert(
            "matmul".into(),
            KernelParams {
                work_group_size: 256,
                tile_size: 32,
                vector_width: 8,
            },
        );
        cache.upsert_profile(profile);
        cache.save(&path).unwrap();

        let loaded = TuningCache::load(&path);
        let p = loaded.get_profile("Arc A770").unwrap();
        assert_eq!(p.best_for("matmul").unwrap().work_group_size, 256);
    }

    #[test]
    fn test_cache_load_missing_file() {
        let cache = TuningCache::load(Path::new("/nonexistent/cache.json"));
        assert!(cache.profiles.is_empty());
    }

    #[test]
    fn test_cache_load_corrupt_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.json");
        std::fs::write(&path, "not valid json {{{").unwrap();
        let cache = TuningCache::load(&path);
        assert!(cache.profiles.is_empty());
    }

    // -- Autotuner tests --

    #[test]
    fn test_autotuner_total_combinations() {
        let tuner = Autotuner::new().with_parameter_space(ParameterSpace {
            work_group_sizes: vec![64, 128],
            tile_sizes: vec![8, 16],
            vector_widths: vec![4],
        });
        assert_eq!(tuner.total_combinations(), 4);
    }

    #[test]
    fn test_autotuner_tune_picks_best() {
        let bench: KernelBenchFn = Box::new(|params| {
            Some(Duration::from_micros(params.work_group_size as u64))
        });

        let mut tuner = Autotuner::new()
            .with_parameter_space(ParameterSpace {
                work_group_sizes: vec![64, 256],
                tile_sizes: vec![8],
                vector_widths: vec![1],
            })
            .with_warmup(0)
            .with_iterations(3);

        let results = tuner.tune("test_kernel", "FakeGPU", "1.0", &bench);
        assert_eq!(results.len(), 2);

        let best = tuner.cached_best("FakeGPU", "test_kernel").unwrap();
        assert_eq!(best.work_group_size, 64); // fastest
    }

    #[test]
    fn test_autotuner_handles_invalid_params() {
        let call_count = AtomicU32::new(0);
        let bench: KernelBenchFn = Box::new(|params| {
            call_count.fetch_add(1, Ordering::Relaxed);
            if params.vector_width > 4 {
                None // unsupported
            } else {
                Some(Duration::from_micros(100))
            }
        });

        let mut tuner = Autotuner::new()
            .with_parameter_space(ParameterSpace {
                work_group_sizes: vec![128],
                tile_sizes: vec![16],
                vector_widths: vec![4, 8],
            })
            .with_warmup(1)
            .with_iterations(2);

        let results = tuner.tune("test_kernel", "FakeGPU", "1.0", &bench);
        assert_eq!(results.len(), 2);

        let valid: Vec<_> = results.iter().filter(|r| r.valid).collect();
        assert_eq!(valid.len(), 1);
        assert_eq!(valid[0].params.vector_width, 4);
    }

    #[test]
    fn test_autotuner_cache_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("tuning.json");

        let bench: KernelBenchFn =
            Box::new(|_| Some(Duration::from_micros(42)));

        let mut tuner = Autotuner::new()
            .with_parameter_space(ParameterSpace {
                work_group_sizes: vec![128],
                tile_sizes: vec![16],
                vector_widths: vec![4],
            })
            .with_cache_path(&cache_path)
            .with_warmup(0)
            .with_iterations(1);

        tuner.tune("matmul", "TestGPU", "1.0", &bench);

        assert!(cache_path.exists());
        let loaded = TuningCache::load(&cache_path);
        assert!(loaded.get_profile("TestGPU").is_some());
    }
}
