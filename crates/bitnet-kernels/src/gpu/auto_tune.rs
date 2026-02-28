//! Work-group size auto-tuning for OpenCL kernels.
//!
//! Selects optimal `local_work_size` per (device, kernel) pair by running
//! micro-benchmarks and caching results. Supports an environment variable
//! override (`BITNET_OPENCL_LOCAL_SIZE`) for manual tuning.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// A work-group configuration for an OpenCL kernel dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkGroupConfig {
    /// Local work-group size in the x-dimension.
    pub local_x: u32,
    /// Local work-group size in the y-dimension (1 for 1-D kernels).
    pub local_y: u32,
    /// Local work-group size in the z-dimension (1 for 1-D/2-D kernels).
    pub local_z: u32,
}

impl WorkGroupConfig {
    /// Create a 1-D work-group config.
    pub fn new_1d(local_x: u32) -> Self {
        Self { local_x, local_y: 1, local_z: 1 }
    }

    /// Create a 2-D work-group config.
    pub fn new_2d(local_x: u32, local_y: u32) -> Self {
        Self { local_x, local_y, local_z: 1 }
    }

    /// Create a 3-D work-group config.
    pub fn new_3d(local_x: u32, local_y: u32, local_z: u32) -> Self {
        Self { local_x, local_y, local_z }
    }

    /// Total number of work-items per work-group.
    pub fn total_size(&self) -> u32 {
        self.local_x * self.local_y * self.local_z
    }
}

impl Default for WorkGroupConfig {
    fn default() -> Self {
        Self::new_1d(64)
    }
}

/// Candidate configurations to benchmark during auto-tuning.
pub const DEFAULT_1D_CANDIDATES: &[u32] = &[32, 64, 128, 256, 512];
pub const DEFAULT_2D_CANDIDATES: &[(u32, u32)] = &[
    (8, 8),
    (16, 8),
    (8, 16),
    (16, 16),
    (32, 8),
    (8, 32),
];

/// Key for cache lookups: (device_id, kernel_name).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TuningKey {
    pub device_id: String,
    pub kernel_name: String,
}

/// Result of a single tuning trial.
#[derive(Debug, Clone)]
pub struct TuningResult {
    pub config: WorkGroupConfig,
    pub elapsed: Duration,
}

/// Cache of auto-tuned work-group configurations.
#[derive(Debug, Clone)]
pub struct TuningCache {
    entries: Arc<RwLock<HashMap<TuningKey, WorkGroupConfig>>>,
}

impl Default for TuningCache {
    fn default() -> Self {
        Self::new()
    }
}

impl TuningCache {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Look up a cached config.
    pub fn get(&self, key: &TuningKey) -> Option<WorkGroupConfig> {
        self.entries.read().ok()?.get(key).copied()
    }

    /// Store a tuned config.
    pub fn insert(&self, key: TuningKey, config: WorkGroupConfig) {
        if let Ok(mut map) = self.entries.write() {
            map.insert(key, config);
        }
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.read().map(|m| m.len()).unwrap_or(0)
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        if let Ok(mut map) = self.entries.write() {
            map.clear();
        }
    }
}

/// Parse `BITNET_OPENCL_LOCAL_SIZE` environment variable.
///
/// Accepts formats: `"128"` (1-D), `"16,16"` (2-D), `"8,8,4"` (3-D).
/// Returns `None` if unset or unparseable.
pub fn parse_local_size_env() -> Option<WorkGroupConfig> {
    let val = std::env::var("BITNET_OPENCL_LOCAL_SIZE").ok()?;
    let parts: Vec<&str> = val.trim().split(',').collect();
    match parts.len() {
        1 => {
            let x = parts[0].trim().parse::<u32>().ok()?;
            Some(WorkGroupConfig::new_1d(x))
        }
        2 => {
            let x = parts[0].trim().parse::<u32>().ok()?;
            let y = parts[1].trim().parse::<u32>().ok()?;
            Some(WorkGroupConfig::new_2d(x, y))
        }
        3 => {
            let x = parts[0].trim().parse::<u32>().ok()?;
            let y = parts[1].trim().parse::<u32>().ok()?;
            let z = parts[2].trim().parse::<u32>().ok()?;
            Some(WorkGroupConfig::new_3d(x, y, z))
        }
        _ => None,
    }
}

/// Run auto-tuning for a kernel by benchmarking candidate work-group sizes.
///
/// `benchmark_fn` is called with each candidate config and should return the
/// time taken to execute the kernel with that configuration. The fastest
/// config wins.
///
/// If `BITNET_OPENCL_LOCAL_SIZE` is set, the env override is returned
/// immediately without benchmarking.
///
/// Results are stored in `cache` for subsequent lookups.
pub fn auto_tune_kernel<F>(
    device_id: &str,
    kernel_name: &str,
    candidates: &[WorkGroupConfig],
    cache: &TuningCache,
    benchmark_fn: F,
) -> WorkGroupConfig
where
    F: Fn(&WorkGroupConfig) -> Duration,
{
    let key = TuningKey {
        device_id: device_id.to_string(),
        kernel_name: kernel_name.to_string(),
    };

    // Check env override first.
    if let Some(env_config) = parse_local_size_env() {
        cache.insert(key, env_config);
        return env_config;
    }

    // Check cache.
    if let Some(cached) = cache.get(&key) {
        return cached;
    }

    // Benchmark each candidate, pick the fastest.
    let best = candidates
        .iter()
        .map(|config| {
            let elapsed = benchmark_fn(config);
            TuningResult { config: *config, elapsed }
        })
        .min_by_key(|r| r.elapsed)
        .map(|r| r.config)
        .unwrap_or_default();

    cache.insert(key, best);
    best
}

/// Generate 1-D candidate configs from the default sizes, filtered by
/// `max_work_group_size` (device limit).
pub fn candidates_1d(max_work_group_size: u32) -> Vec<WorkGroupConfig> {
    DEFAULT_1D_CANDIDATES
        .iter()
        .filter(|&&size| size <= max_work_group_size)
        .map(|&size| WorkGroupConfig::new_1d(size))
        .collect()
}

/// Generate 2-D candidate configs from the default sizes, filtered by
/// `max_work_group_size` (device limit).
pub fn candidates_2d(max_work_group_size: u32) -> Vec<WorkGroupConfig> {
    DEFAULT_2D_CANDIDATES
        .iter()
        .filter(|&&(x, y)| x * y <= max_work_group_size)
        .map(|&(x, y)| WorkGroupConfig::new_2d(x, y))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_1d() {
        let c = WorkGroupConfig::new_1d(128);
        assert_eq!(c.total_size(), 128);
        assert_eq!(c.local_y, 1);
        assert_eq!(c.local_z, 1);
    }

    #[test]
    fn config_2d() {
        let c = WorkGroupConfig::new_2d(16, 16);
        assert_eq!(c.total_size(), 256);
    }

    #[test]
    fn config_3d() {
        let c = WorkGroupConfig::new_3d(8, 8, 4);
        assert_eq!(c.total_size(), 256);
    }

    #[test]
    fn config_default() {
        let c = WorkGroupConfig::default();
        assert_eq!(c.local_x, 64);
        assert_eq!(c.total_size(), 64);
    }

    #[test]
    fn cache_insert_get() {
        let cache = TuningCache::new();
        assert!(cache.is_empty());

        let key = TuningKey {
            device_id: "dev0".to_string(),
            kernel_name: "matmul".to_string(),
        };
        let config = WorkGroupConfig::new_1d(256);

        cache.insert(key.clone(), config);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&key), Some(config));
    }

    #[test]
    fn cache_clear() {
        let cache = TuningCache::new();
        let key = TuningKey {
            device_id: "dev0".to_string(),
            kernel_name: "kern".to_string(),
        };
        cache.insert(key, WorkGroupConfig::default());
        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn auto_tune_selects_fastest() {
        let cache = TuningCache::new();
        let candidates = vec![
            WorkGroupConfig::new_1d(32),
            WorkGroupConfig::new_1d(64),
            WorkGroupConfig::new_1d(128),
        ];

        // Simulate: 64 is the fastest.
        let best = auto_tune_kernel(
            "test_dev",
            "test_kern",
            &candidates,
            &cache,
            |config| match config.local_x {
                32 => Duration::from_millis(10),
                64 => Duration::from_millis(5),
                128 => Duration::from_millis(8),
                _ => Duration::from_millis(100),
            },
        );

        assert_eq!(best.local_x, 64);

        // Verify it was cached.
        let key = TuningKey {
            device_id: "test_dev".to_string(),
            kernel_name: "test_kern".to_string(),
        };
        assert_eq!(cache.get(&key), Some(best));
    }

    #[test]
    fn auto_tune_uses_cache() {
        let cache = TuningCache::new();
        let key = TuningKey {
            device_id: "dev".to_string(),
            kernel_name: "kern".to_string(),
        };
        let preset = WorkGroupConfig::new_1d(256);
        cache.insert(key, preset);

        let mut called = false;
        let result = auto_tune_kernel(
            "dev",
            "kern",
            &[WorkGroupConfig::new_1d(64)],
            &cache,
            |_| {
                called = true;
                Duration::from_millis(1)
            },
        );

        assert_eq!(result, preset);
        assert!(!called, "benchmark_fn should not be called when cache hit");
    }

    #[test]
    fn candidates_1d_respects_max() {
        let cands = candidates_1d(128);
        assert!(cands.iter().all(|c| c.total_size() <= 128));
        assert_eq!(cands.len(), 3); // 32, 64, 128
    }

    #[test]
    fn candidates_2d_respects_max() {
        let cands = candidates_2d(128);
        assert!(cands.iter().all(|c| c.total_size() <= 128));
        // (8,8)=64, (16,8)=128, (8,16)=128
        assert_eq!(cands.len(), 3);
    }

    #[test]
    fn tuning_result_ordering() {
        let a = TuningResult {
            config: WorkGroupConfig::new_1d(32),
            elapsed: Duration::from_millis(10),
        };
        let b = TuningResult {
            config: WorkGroupConfig::new_1d(64),
            elapsed: Duration::from_millis(5),
        };
        assert!(b.elapsed < a.elapsed);
    }

    #[test]
    fn auto_tune_fallback_on_empty_candidates() {
        let cache = TuningCache::new();
        let result = auto_tune_kernel(
            "dev",
            "kern",
            &[],
            &cache,
            |_| Duration::from_millis(1),
        );
        // Falls back to default
        assert_eq!(result, WorkGroupConfig::default());
    }
}
