//! OpenCL device warmup for A770 GPU initialization.
//!
//! GPU first-use has high latency due to JIT compilation and device init.
//! This module manages warmup routines to pre-initialize the GPU.

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// WarmupPhase
// ---------------------------------------------------------------------------

/// Phases executed during device warmup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarmupPhase {
    DeviceInit,
    KernelCompilation,
    MemoryAllocation,
    DryRun,
    Complete,
}

impl WarmupPhase {
    /// Return the ordered sequence of active phases for a given config.
    fn active_phases(config: &WarmupConfig) -> Vec<WarmupPhase> {
        let mut phases = vec![WarmupPhase::DeviceInit];
        if config.compile_all_kernels {
            phases.push(WarmupPhase::KernelCompilation);
        }
        if config.allocate_pools {
            phases.push(WarmupPhase::MemoryAllocation);
        }
        if config.dry_run_matmul {
            phases.push(WarmupPhase::DryRun);
        }
        phases.push(WarmupPhase::Complete);
        phases
    }
}

impl fmt::Display for WarmupPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceInit => write!(f, "DeviceInit"),
            Self::KernelCompilation => write!(f, "KernelCompilation"),
            Self::MemoryAllocation => write!(f, "MemoryAllocation"),
            Self::DryRun => write!(f, "DryRun"),
            Self::Complete => write!(f, "Complete"),
        }
    }
}

// ---------------------------------------------------------------------------
// WarmupConfig
// ---------------------------------------------------------------------------

/// Configuration for the warmup sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarmupConfig {
    /// Pre-compile all `.cl` kernel sources.
    pub compile_all_kernels: bool,
    /// Pre-allocate buffer pools.
    pub allocate_pools: bool,
    /// Run a small matmul to warm caches.
    pub dry_run_matmul: bool,
    /// Matrix dimension for the dry-run matmul.
    pub dry_run_size: usize,
    /// Overall warmup timeout in milliseconds.
    pub timeout_ms: u64,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self::thorough()
    }
}

impl WarmupConfig {
    /// Fast preset – skip kernel compilation and memory pools.
    pub fn fast() -> Self {
        Self {
            compile_all_kernels: false,
            allocate_pools: false,
            dry_run_matmul: true,
            dry_run_size: 64,
            timeout_ms: 5_000,
        }
    }

    /// Thorough preset – all warmup phases enabled.
    pub fn thorough() -> Self {
        Self {
            compile_all_kernels: true,
            allocate_pools: true,
            dry_run_matmul: true,
            dry_run_size: 64,
            timeout_ms: 30_000,
        }
    }

    /// Minimal preset – only device init, no extra work.
    pub fn minimal() -> Self {
        Self {
            compile_all_kernels: false,
            allocate_pools: false,
            dry_run_matmul: false,
            dry_run_size: 64,
            timeout_ms: 2_000,
        }
    }

    /// Validate the config, returning an error if invalid.
    pub fn validate(&self) -> Result<(), WarmupError> {
        if self.dry_run_matmul && self.dry_run_size == 0 {
            return Err(WarmupError::InvalidConfig(
                "dry_run_size must be > 0 when dry_run_matmul is enabled".into(),
            ));
        }
        if self.timeout_ms == 0 {
            return Err(WarmupError::InvalidConfig("timeout_ms must be > 0".into()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// WarmupResult / WarmupReport
// ---------------------------------------------------------------------------

/// Outcome of a single warmup phase.
#[derive(Debug, Clone)]
pub struct WarmupResult {
    pub phase: WarmupPhase,
    pub duration_ms: u64,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Aggregate report produced by [`DeviceWarmer::run_warmup`].
#[derive(Debug, Clone)]
pub struct WarmupReport {
    pub results: Vec<WarmupResult>,
    pub total_duration_ms: u64,
    pub all_succeeded: bool,
}

impl WarmupReport {
    /// Human-readable summary line.
    pub fn summary(&self) -> String {
        let ok = self.results.iter().filter(|r| r.success).count();
        let total = self.results.len();
        format!("Warmup: {ok}/{total} phases succeeded in {ms}ms", ms = self.total_duration_ms,)
    }

    /// Return phases that failed.
    pub fn failed_phases(&self) -> Vec<&WarmupResult> {
        self.results.iter().filter(|r| !r.success).collect()
    }

    /// Return the phase that took the longest.
    pub fn slowest_phase(&self) -> Option<&WarmupResult> {
        self.results.iter().max_by_key(|r| r.duration_ms)
    }
}

// ---------------------------------------------------------------------------
// WarmupError
// ---------------------------------------------------------------------------

/// Errors that may occur during warmup.
#[derive(Debug, Clone)]
pub enum WarmupError {
    /// The supplied configuration is invalid.
    InvalidConfig(String),
    /// The warmup exceeded its timeout.
    Timeout { phase: WarmupPhase, elapsed_ms: u64 },
    /// A phase failed for a device-specific reason.
    PhaseFailed { phase: WarmupPhase, reason: String },
}

impl fmt::Display for WarmupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid warmup config: {msg}"),
            Self::Timeout { phase, elapsed_ms } => {
                write!(f, "warmup timeout in {phase} after {elapsed_ms}ms")
            }
            Self::PhaseFailed { phase, reason } => {
                write!(f, "warmup phase {phase} failed: {reason}")
            }
        }
    }
}

impl std::error::Error for WarmupError {}

// ---------------------------------------------------------------------------
// DeviceWarmer
// ---------------------------------------------------------------------------

/// Manages the warmup lifecycle for an OpenCL device.
pub struct DeviceWarmer {
    config: WarmupConfig,
    completed: bool,
}

impl DeviceWarmer {
    pub fn new(config: WarmupConfig) -> Self {
        Self { config, completed: false }
    }

    /// Whether warmup has completed successfully.
    pub fn is_warmed(&self) -> bool {
        self.completed
    }

    /// Estimate total warmup time for a given config (heuristic, in ms).
    pub fn estimated_warmup_time_ms(config: &WarmupConfig) -> u64 {
        let mut est: u64 = 50; // DeviceInit baseline
        if config.compile_all_kernels {
            est += 200;
        }
        if config.allocate_pools {
            est += 100;
        }
        if config.dry_run_matmul {
            let size = config.dry_run_size as u64;
            est += 50 + size; // rough heuristic
        }
        est += 10; // Complete phase overhead
        est
    }

    /// Execute the full warmup sequence (CPU-only simulation).
    pub fn run_warmup(&mut self) -> Result<WarmupReport, WarmupError> {
        self.config.validate()?;

        let phases = WarmupPhase::active_phases(&self.config);
        let global_start = Instant::now();
        let mut results = Vec::with_capacity(phases.len());

        for &phase in &phases {
            let elapsed_global = global_start.elapsed().as_millis() as u64;
            if elapsed_global > self.config.timeout_ms {
                return Err(WarmupError::Timeout { phase, elapsed_ms: elapsed_global });
            }

            let phase_start = Instant::now();
            let outcome = self.execute_phase(phase);
            let duration_ms = phase_start.elapsed().as_millis() as u64;

            results.push(WarmupResult {
                phase,
                duration_ms,
                success: outcome.is_ok(),
                error_message: outcome.err().map(|e| e.to_string()),
            });
        }

        let total_duration_ms = global_start.elapsed().as_millis() as u64;
        let all_succeeded = results.iter().all(|r| r.success);

        if all_succeeded {
            self.completed = true;
        }

        Ok(WarmupReport { results, total_duration_ms, all_succeeded })
    }

    /// Simulate a single phase (CPU-only stub).
    fn execute_phase(&self, phase: WarmupPhase) -> Result<(), WarmupError> {
        match phase {
            WarmupPhase::DeviceInit => {
                // Simulate device enumeration
                Ok(())
            }
            WarmupPhase::KernelCompilation => {
                // Simulate .cl source compilation
                Ok(())
            }
            WarmupPhase::MemoryAllocation => {
                // Simulate buffer pool allocation
                Ok(())
            }
            WarmupPhase::DryRun => {
                // Simulate a small matmul
                let n = self.config.dry_run_size;
                let _scratch: Vec<f32> = vec![0.0; n * n];
                Ok(())
            }
            WarmupPhase::Complete => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config presets -----------------------------------------------------

    #[test]
    fn test_fast_preset() {
        let c = WarmupConfig::fast();
        assert!(!c.compile_all_kernels);
        assert!(!c.allocate_pools);
        assert!(c.dry_run_matmul);
        assert_eq!(c.dry_run_size, 64);
        assert_eq!(c.timeout_ms, 5_000);
    }

    #[test]
    fn test_thorough_preset() {
        let c = WarmupConfig::thorough();
        assert!(c.compile_all_kernels);
        assert!(c.allocate_pools);
        assert!(c.dry_run_matmul);
        assert_eq!(c.timeout_ms, 30_000);
    }

    #[test]
    fn test_minimal_preset() {
        let c = WarmupConfig::minimal();
        assert!(!c.compile_all_kernels);
        assert!(!c.allocate_pools);
        assert!(!c.dry_run_matmul);
        assert_eq!(c.timeout_ms, 2_000);
    }

    #[test]
    fn test_default_is_thorough() {
        assert_eq!(WarmupConfig::default(), WarmupConfig::thorough());
    }

    #[test]
    fn test_fast_differs_from_thorough() {
        assert_ne!(WarmupConfig::fast(), WarmupConfig::thorough());
    }

    #[test]
    fn test_minimal_differs_from_fast() {
        assert_ne!(WarmupConfig::minimal(), WarmupConfig::fast());
    }

    // -- Config validation --------------------------------------------------

    #[test]
    fn test_valid_config_ok() {
        assert!(WarmupConfig::thorough().validate().is_ok());
    }

    #[test]
    fn test_zero_dry_run_size_invalid() {
        let mut c = WarmupConfig::fast();
        c.dry_run_size = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_zero_timeout_invalid() {
        let mut c = WarmupConfig::fast();
        c.timeout_ms = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_zero_dry_run_size_ok_when_disabled() {
        let mut c = WarmupConfig::minimal();
        c.dry_run_size = 0;
        assert!(c.validate().is_ok());
    }

    // -- Phase Display formatting -------------------------------------------

    #[test]
    fn test_display_device_init() {
        assert_eq!(WarmupPhase::DeviceInit.to_string(), "DeviceInit");
    }

    #[test]
    fn test_display_kernel_compilation() {
        assert_eq!(WarmupPhase::KernelCompilation.to_string(), "KernelCompilation");
    }

    #[test]
    fn test_display_memory_allocation() {
        assert_eq!(WarmupPhase::MemoryAllocation.to_string(), "MemoryAllocation");
    }

    #[test]
    fn test_display_dry_run() {
        assert_eq!(WarmupPhase::DryRun.to_string(), "DryRun");
    }

    #[test]
    fn test_display_complete() {
        assert_eq!(WarmupPhase::Complete.to_string(), "Complete");
    }

    // -- Full warmup run ----------------------------------------------------

    #[test]
    fn test_warmup_thorough_report() {
        let mut w = DeviceWarmer::new(WarmupConfig::thorough());
        let report = w.run_warmup().unwrap();
        assert!(report.all_succeeded);
        assert!(!report.results.is_empty());
    }

    #[test]
    fn test_warmup_fast_report() {
        let mut w = DeviceWarmer::new(WarmupConfig::fast());
        let report = w.run_warmup().unwrap();
        assert!(report.all_succeeded);
    }

    #[test]
    fn test_warmup_minimal_report() {
        let mut w = DeviceWarmer::new(WarmupConfig::minimal());
        let report = w.run_warmup().unwrap();
        assert!(report.all_succeeded);
        // minimal: DeviceInit + Complete
        assert_eq!(report.results.len(), 2);
    }

    // -- Phase ordering -----------------------------------------------------

    #[test]
    fn test_thorough_phase_order() {
        let mut w = DeviceWarmer::new(WarmupConfig::thorough());
        let report = w.run_warmup().unwrap();
        let phases: Vec<_> = report.results.iter().map(|r| r.phase).collect();
        assert_eq!(
            phases,
            vec![
                WarmupPhase::DeviceInit,
                WarmupPhase::KernelCompilation,
                WarmupPhase::MemoryAllocation,
                WarmupPhase::DryRun,
                WarmupPhase::Complete,
            ]
        );
    }

    #[test]
    fn test_fast_phase_order() {
        let mut w = DeviceWarmer::new(WarmupConfig::fast());
        let report = w.run_warmup().unwrap();
        let phases: Vec<_> = report.results.iter().map(|r| r.phase).collect();
        assert_eq!(
            phases,
            vec![WarmupPhase::DeviceInit, WarmupPhase::DryRun, WarmupPhase::Complete]
        );
    }

    #[test]
    fn test_minimal_phase_order() {
        let mut w = DeviceWarmer::new(WarmupConfig::minimal());
        let report = w.run_warmup().unwrap();
        let phases: Vec<_> = report.results.iter().map(|r| r.phase).collect();
        assert_eq!(phases, vec![WarmupPhase::DeviceInit, WarmupPhase::Complete]);
    }

    // -- Report methods -----------------------------------------------------

    #[test]
    fn test_summary_contains_all_phases() {
        let mut w = DeviceWarmer::new(WarmupConfig::thorough());
        let report = w.run_warmup().unwrap();
        let summary = report.summary();
        assert!(summary.contains("5/5"));
    }

    #[test]
    fn test_failed_phases_empty_on_success() {
        let mut w = DeviceWarmer::new(WarmupConfig::fast());
        let report = w.run_warmup().unwrap();
        assert!(report.failed_phases().is_empty());
    }

    #[test]
    fn test_slowest_phase_present() {
        let mut w = DeviceWarmer::new(WarmupConfig::thorough());
        let report = w.run_warmup().unwrap();
        assert!(report.slowest_phase().is_some());
    }

    #[test]
    fn test_failed_phases_detection() {
        let report = WarmupReport {
            results: vec![
                WarmupResult {
                    phase: WarmupPhase::DeviceInit,
                    duration_ms: 10,
                    success: true,
                    error_message: None,
                },
                WarmupResult {
                    phase: WarmupPhase::KernelCompilation,
                    duration_ms: 5,
                    success: false,
                    error_message: Some("compile error".into()),
                },
            ],
            total_duration_ms: 15,
            all_succeeded: false,
        };
        let failed = report.failed_phases();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].phase, WarmupPhase::KernelCompilation);
    }

    #[test]
    fn test_slowest_phase_identification() {
        let report = WarmupReport {
            results: vec![
                WarmupResult {
                    phase: WarmupPhase::DeviceInit,
                    duration_ms: 10,
                    success: true,
                    error_message: None,
                },
                WarmupResult {
                    phase: WarmupPhase::DryRun,
                    duration_ms: 200,
                    success: true,
                    error_message: None,
                },
                WarmupResult {
                    phase: WarmupPhase::Complete,
                    duration_ms: 1,
                    success: true,
                    error_message: None,
                },
            ],
            total_duration_ms: 211,
            all_succeeded: true,
        };
        let slowest = report.slowest_phase().unwrap();
        assert_eq!(slowest.phase, WarmupPhase::DryRun);
        assert_eq!(slowest.duration_ms, 200);
    }

    // -- is_warmed state tracking -------------------------------------------

    #[test]
    fn test_not_warmed_initially() {
        let w = DeviceWarmer::new(WarmupConfig::fast());
        assert!(!w.is_warmed());
    }

    #[test]
    fn test_warmed_after_run() {
        let mut w = DeviceWarmer::new(WarmupConfig::fast());
        w.run_warmup().unwrap();
        assert!(w.is_warmed());
    }

    #[test]
    fn test_repeated_warmup_idempotent() {
        let mut w = DeviceWarmer::new(WarmupConfig::thorough());
        let r1 = w.run_warmup().unwrap();
        let r2 = w.run_warmup().unwrap();
        assert!(r1.all_succeeded);
        assert!(r2.all_succeeded);
        assert!(w.is_warmed());
    }

    // -- Estimated time -----------------------------------------------------

    #[test]
    fn test_estimated_time_thorough() {
        let est = DeviceWarmer::estimated_warmup_time_ms(&WarmupConfig::thorough());
        // DeviceInit(50) + Compile(200) + Alloc(100) + DryRun(50+64) + Complete(10)
        assert_eq!(est, 474);
    }

    #[test]
    fn test_estimated_time_minimal() {
        let est = DeviceWarmer::estimated_warmup_time_ms(&WarmupConfig::minimal());
        // DeviceInit(50) + Complete(10)
        assert_eq!(est, 60);
    }

    #[test]
    fn test_estimated_time_fast() {
        let est = DeviceWarmer::estimated_warmup_time_ms(&WarmupConfig::fast());
        // DeviceInit(50) + DryRun(50+64) + Complete(10)
        assert_eq!(est, 174);
    }

    #[test]
    fn test_estimated_time_increases_with_dry_run_size() {
        let mut c = WarmupConfig::fast();
        let est_small = DeviceWarmer::estimated_warmup_time_ms(&c);
        c.dry_run_size = 256;
        let est_large = DeviceWarmer::estimated_warmup_time_ms(&c);
        assert!(est_large > est_small);
    }

    // -- Timeout handling ---------------------------------------------------

    #[test]
    fn test_timeout_zero_rejected() {
        let mut c = WarmupConfig::minimal();
        c.timeout_ms = 0;
        let mut w = DeviceWarmer::new(c);
        assert!(w.run_warmup().is_err());
    }

    // -- Empty warmup (all phases disabled) ---------------------------------

    #[test]
    fn test_empty_warmup_all_disabled() {
        let c = WarmupConfig {
            compile_all_kernels: false,
            allocate_pools: false,
            dry_run_matmul: false,
            dry_run_size: 64,
            timeout_ms: 1_000,
        };
        let mut w = DeviceWarmer::new(c);
        let report = w.run_warmup().unwrap();
        // DeviceInit + Complete only
        assert_eq!(report.results.len(), 2);
        assert!(report.all_succeeded);
    }

    // -- WarmupError display ------------------------------------------------

    #[test]
    fn test_error_display_invalid_config() {
        let e = WarmupError::InvalidConfig("bad".into());
        assert_eq!(e.to_string(), "invalid warmup config: bad");
    }

    #[test]
    fn test_error_display_timeout() {
        let e = WarmupError::Timeout { phase: WarmupPhase::DryRun, elapsed_ms: 999 };
        assert_eq!(e.to_string(), "warmup timeout in DryRun after 999ms");
    }

    #[test]
    fn test_error_display_phase_failed() {
        let e =
            WarmupError::PhaseFailed { phase: WarmupPhase::MemoryAllocation, reason: "OOM".into() };
        assert_eq!(e.to_string(), "warmup phase MemoryAllocation failed: OOM");
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(WarmupError::InvalidConfig("x".into()));
        assert!(!e.to_string().is_empty());
    }

    // -- Misc ---------------------------------------------------------------

    #[test]
    fn test_warmup_report_total_duration_nonnegative() {
        let mut w = DeviceWarmer::new(WarmupConfig::fast());
        let report = w.run_warmup().unwrap();
        // total_duration_ms is u64, always >= 0; just sanity-check it's <= sum
        let sum: u64 = report.results.iter().map(|r| r.duration_ms).sum();
        assert!(report.total_duration_ms <= sum + 5); // small slack
    }

    #[test]
    fn test_phase_eq_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(WarmupPhase::DeviceInit);
        set.insert(WarmupPhase::DeviceInit);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_warmup_result_error_message_none_on_success() {
        let mut w = DeviceWarmer::new(WarmupConfig::minimal());
        let report = w.run_warmup().unwrap();
        for r in &report.results {
            assert!(r.success);
            assert!(r.error_message.is_none());
        }
    }

    #[test]
    fn test_config_clone() {
        let c = WarmupConfig::thorough();
        let c2 = c.clone();
        assert_eq!(c, c2);
    }
}
