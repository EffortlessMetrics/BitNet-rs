use bitnet_nvidia::NVIDIA_1D_WORKGROUP_CANDIDATES;

use super::WorkgroupConfig;

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
        let candidates = NVIDIA_1D_WORKGROUP_CANDIDATES
            .iter()
            .map(|&size| WorkgroupConfig::new([size, 1, 1], format!("nvidia_{size}")))
            .collect();
        Self { candidates }
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
