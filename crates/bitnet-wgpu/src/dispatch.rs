//! Workgroup dispatch helpers and NVIDIA-tuned sizing.

use std::fmt;

/// Configuration for a single compute dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchConfig {
    /// Workgroup size declared in the shader (x, y, z).
    pub workgroup_size: [u32; 3],
    /// Number of workgroups to dispatch (x, y, z).
    pub dispatch_size: [u32; 3],
}

impl DispatchConfig {
    /// Total number of threads launched across all workgroups.
    pub fn total_threads(&self) -> u64 {
        let wg: u64 = self.workgroup_size.iter().map(|&v| v as u64).product();
        let ds: u64 = self.dispatch_size.iter().map(|&v| v as u64).product();
        wg * ds
    }
}

impl fmt::Display for DispatchConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "workgroup[{}×{}×{}] dispatch[{}×{}×{}]",
            self.workgroup_size[0],
            self.workgroup_size[1],
            self.workgroup_size[2],
            self.dispatch_size[0],
            self.dispatch_size[1],
            self.dispatch_size[2],
        )
    }
}

/// Compute the number of workgroups needed to cover `total_elements`,
/// rounding up so no element is missed.
pub fn compute_dispatch_size(total_elements: u32, workgroup_size: u32) -> u32 {
    assert!(workgroup_size > 0, "workgroup_size must be > 0");
    total_elements.div_ceil(workgroup_size)
}

/// NVIDIA warp size (threads per warp on all NVIDIA architectures).
const NVIDIA_WARP_SIZE: u32 = 32;

/// Return an NVIDIA-tuned workgroup size for 1-D dispatches.
///
/// Heuristics:
/// - Always warp-aligned (multiple of 32).
/// - Prefer 256 for Blackwell / Ada / Ampere (good occupancy).
/// - Fall back to 128 for very small workloads, 64 for tiny ones.
pub fn optimal_workgroup_size_nvidia(elements: u32) -> u32 {
    if elements == 0 {
        return NVIDIA_WARP_SIZE;
    }
    if elements <= 64 {
        // Tiny: one or two warps.
        NVIDIA_WARP_SIZE * 2 // 64
    } else if elements <= 256 {
        // Small: four warps.
        128
    } else {
        // Default: eight warps — best occupancy on Blackwell / RTX 5070 Ti.
        256
    }
}

/// Records dispatches for profiling / diagnostics.
#[derive(Debug, Default)]
pub struct DispatchRecorder {
    entries: Vec<DispatchEntry>,
}

/// A single recorded dispatch.
#[derive(Debug, Clone)]
pub struct DispatchEntry {
    /// Human-readable label (e.g. kernel name).
    pub label: String,
    /// The dispatch configuration used.
    pub config: DispatchConfig,
}

impl DispatchRecorder {
    /// Create an empty recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a dispatch.
    pub fn record(&mut self, label: impl Into<String>, config: DispatchConfig) {
        self.entries.push(DispatchEntry { label: label.into(), config });
    }

    /// Number of recorded dispatches.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the recorder is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over recorded entries.
    pub fn entries(&self) -> &[DispatchEntry] {
        &self.entries
    }

    /// Total threads across all recorded dispatches.
    pub fn total_threads(&self) -> u64 {
        self.entries.iter().map(|e| e.config.total_threads()).sum()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_size_exact_multiple() {
        assert_eq!(compute_dispatch_size(256, 256), 1);
        assert_eq!(compute_dispatch_size(512, 256), 2);
        assert_eq!(compute_dispatch_size(1024, 256), 4);
    }

    #[test]
    fn dispatch_size_rounds_up() {
        assert_eq!(compute_dispatch_size(1, 256), 1);
        assert_eq!(compute_dispatch_size(257, 256), 2);
        assert_eq!(compute_dispatch_size(513, 256), 3);
    }

    #[test]
    fn dispatch_size_single_element() {
        assert_eq!(compute_dispatch_size(1, 1), 1);
        assert_eq!(compute_dispatch_size(1, 64), 1);
    }

    #[test]
    #[should_panic(expected = "workgroup_size must be > 0")]
    fn dispatch_size_zero_workgroup_panics() {
        compute_dispatch_size(100, 0);
    }

    #[test]
    fn nvidia_workgroup_zero_elements() {
        let ws = optimal_workgroup_size_nvidia(0);
        assert_eq!(ws, NVIDIA_WARP_SIZE);
    }

    #[test]
    fn nvidia_workgroup_tiny() {
        assert_eq!(optimal_workgroup_size_nvidia(1), 64);
        assert_eq!(optimal_workgroup_size_nvidia(32), 64);
        assert_eq!(optimal_workgroup_size_nvidia(64), 64);
    }

    #[test]
    fn nvidia_workgroup_small() {
        assert_eq!(optimal_workgroup_size_nvidia(65), 128);
        assert_eq!(optimal_workgroup_size_nvidia(128), 128);
        assert_eq!(optimal_workgroup_size_nvidia(256), 128);
    }

    #[test]
    fn nvidia_workgroup_large() {
        assert_eq!(optimal_workgroup_size_nvidia(257), 256);
        assert_eq!(optimal_workgroup_size_nvidia(1024), 256);
        assert_eq!(optimal_workgroup_size_nvidia(1_000_000), 256);
    }

    #[test]
    fn nvidia_workgroup_always_warp_aligned() {
        for n in [0, 1, 32, 33, 64, 65, 128, 256, 257, 512, 100_000] {
            let ws = optimal_workgroup_size_nvidia(n);
            assert_eq!(ws % NVIDIA_WARP_SIZE, 0, "ws={ws} not warp-aligned for n={n}");
        }
    }

    #[test]
    fn dispatch_config_total_threads() {
        let cfg = DispatchConfig { workgroup_size: [256, 1, 1], dispatch_size: [4, 1, 1] };
        assert_eq!(cfg.total_threads(), 1024);
    }

    #[test]
    fn dispatch_config_total_threads_3d() {
        let cfg = DispatchConfig { workgroup_size: [8, 8, 4], dispatch_size: [2, 2, 2] };
        // 8*8*4 = 256 threads per WG, 2*2*2 = 8 WGs → 2048
        assert_eq!(cfg.total_threads(), 2048);
    }

    #[test]
    fn dispatch_config_display() {
        let cfg = DispatchConfig { workgroup_size: [256, 1, 1], dispatch_size: [4, 2, 1] };
        let s = cfg.to_string();
        assert!(s.contains("256"), "display should contain workgroup x");
        assert!(s.contains("dispatch"), "display should contain 'dispatch'");
    }

    #[test]
    fn recorder_basic_usage() {
        let mut rec = DispatchRecorder::new();
        assert!(rec.is_empty());
        assert_eq!(rec.len(), 0);

        let cfg = DispatchConfig { workgroup_size: [256, 1, 1], dispatch_size: [4, 1, 1] };
        rec.record("matmul", cfg);
        assert_eq!(rec.len(), 1);
        assert!(!rec.is_empty());
        assert_eq!(rec.entries()[0].label, "matmul");
        assert_eq!(rec.total_threads(), 1024);
    }

    #[test]
    fn recorder_clear() {
        let mut rec = DispatchRecorder::new();
        let cfg = DispatchConfig { workgroup_size: [64, 1, 1], dispatch_size: [1, 1, 1] };
        rec.record("a", cfg);
        rec.record("b", cfg);
        assert_eq!(rec.len(), 2);
        rec.clear();
        assert!(rec.is_empty());
        assert_eq!(rec.total_threads(), 0);
    }
}
