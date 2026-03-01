//! Workgroup size tuning grid for wgpu compute kernel dispatch.

/// A candidate workgroup configuration for kernel dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkgroupConfig {
    pub size: [u32; 3],
    pub label: String,
}

impl WorkgroupConfig {
    pub fn new(size: [u32; 3], label: impl Into<String>) -> Self {
        Self { size, label: label.into() }
    }

    /// Total number of invocations per workgroup.
    pub fn total_invocations(&self) -> u32 {
        self.size[0] * self.size[1] * self.size[2]
    }
}

/// The result of running a single workgroup configuration.
#[derive(Debug, Clone)]
pub struct TuningResult {
    pub config: WorkgroupConfig,
    pub elapsed_us: u64,
    pub throughput: f64,
}

/// A search space of workgroup configurations to evaluate.
#[derive(Debug, Clone)]
pub struct TuningGrid {
    pub candidates: Vec<WorkgroupConfig>,
}

impl TuningGrid {
    pub fn new(candidates: Vec<WorkgroupConfig>) -> Self {
        Self { candidates }
    }

    /// NVIDIA-optimized defaults: warp-aligned sizes (multiples of 32).
    pub fn nvidia_defaults() -> Self {
        Self {
            candidates: vec![
                WorkgroupConfig::new([64, 1, 1], "nvidia_64"),
                WorkgroupConfig::new([128, 1, 1], "nvidia_128"),
                WorkgroupConfig::new([256, 1, 1], "nvidia_256"),
                WorkgroupConfig::new([512, 1, 1], "nvidia_512"),
            ],
        }
    }

    /// Number of candidates in the grid.
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Whether the grid is empty.
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workgroup_config_new() {
        let cfg = WorkgroupConfig::new([256, 1, 1], "test");
        assert_eq!(cfg.size, [256, 1, 1]);
        assert_eq!(cfg.label, "test");
    }

    #[test]
    fn test_total_invocations_1d() {
        let cfg = WorkgroupConfig::new([256, 1, 1], "1d");
        assert_eq!(cfg.total_invocations(), 256);
    }

    #[test]
    fn test_total_invocations_3d() {
        let cfg = WorkgroupConfig::new([8, 8, 4], "3d");
        assert_eq!(cfg.total_invocations(), 256);
    }

    #[test]
    fn test_nvidia_defaults_count() {
        let grid = TuningGrid::nvidia_defaults();
        assert_eq!(grid.len(), 4);
        assert!(!grid.is_empty());
    }

    #[test]
    fn test_nvidia_defaults_all_warp_aligned() {
        let grid = TuningGrid::nvidia_defaults();
        for cfg in &grid.candidates {
            assert_eq!(cfg.total_invocations() % 32, 0, "{} is not warp-aligned", cfg.label);
        }
    }

    #[test]
    fn test_nvidia_defaults_sizes() {
        let grid = TuningGrid::nvidia_defaults();
        let sizes: Vec<u32> = grid.candidates.iter().map(|c| c.size[0]).collect();
        assert!(sizes.contains(&64));
        assert!(sizes.contains(&128));
        assert!(sizes.contains(&256));
        assert!(sizes.contains(&512));
    }

    #[test]
    fn test_custom_grid() {
        let grid = TuningGrid::new(vec![
            WorkgroupConfig::new([32, 1, 1], "small"),
            WorkgroupConfig::new([16, 16, 1], "square"),
        ]);
        assert_eq!(grid.len(), 2);
        assert_eq!(grid.candidates[1].total_invocations(), 256);
    }

    #[test]
    fn test_empty_grid() {
        let grid = TuningGrid::new(vec![]);
        assert!(grid.is_empty());
        assert_eq!(grid.len(), 0);
    }
}
