//! OpenCL error recovery with automatic CPU fallback.
//!
//! When an OpenCL kernel execution fails (OOM, device lost, compilation failure,
//! timeout, etc.), this module maps the error to a [`RecoveryStrategy`] and
//! executes it — retrying on the GPU, falling back to a CPU implementation,
//! or both.
//!
//! # Usage
//!
//! ```rust,ignore
//! use bitnet_kernels::gpu::opencl_recovery::*;
//!
//! let policy = RecoveryPolicy::default();
//! let mut executor = RecoveryExecutor::new(policy);
//!
//! let (result, path) = executor.execute(
//!     || gpu_matmul(a, b),      // GPU attempt
//!     || cpu_matmul(a, b),      // CPU fallback
//! )?;
//! ```

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

// ---------------------------------------------------------------------------
// OpenClError
// ---------------------------------------------------------------------------

/// OpenCL error codes relevant to kernel execution recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenClError {
    /// `CL_DEVICE_NOT_FOUND` / device was lost mid-execution.
    DeviceLost,
    /// `CL_OUT_OF_HOST_MEMORY` or `CL_MEM_OBJECT_ALLOCATION_FAILURE`.
    OutOfMemory,
    /// `CL_BUILD_PROGRAM_FAILURE` — kernel source could not compile.
    CompilationFailed,
    /// `CL_INVALID_KERNEL` / `CL_INVALID_KERNEL_NAME`.
    InvalidKernel,
    /// `CL_INVALID_ARG_SIZE`.
    InvalidArgSize,
    /// `CL_INVALID_WORK_GROUP_SIZE`.
    InvalidWorkGroupSize,
    /// Execution exceeded a user-defined wall-clock timeout.
    Timeout,
    /// Driver crashed or returned an unexpected error.
    DriverCrash,
    /// No OpenCL platform was found on the system.
    PlatformNotFound,
    /// `CL_DEVICE_NOT_AVAILABLE`.
    DeviceNotAvailable,
}

impl fmt::Display for OpenClError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceLost => write!(f, "OpenCL device lost"),
            Self::OutOfMemory => write!(f, "OpenCL out of memory"),
            Self::CompilationFailed => write!(f, "OpenCL kernel compilation failed"),
            Self::InvalidKernel => write!(f, "OpenCL invalid kernel"),
            Self::InvalidArgSize => write!(f, "OpenCL invalid argument size"),
            Self::InvalidWorkGroupSize => write!(f, "OpenCL invalid work-group size"),
            Self::Timeout => write!(f, "OpenCL execution timeout"),
            Self::DriverCrash => write!(f, "OpenCL driver crash"),
            Self::PlatformNotFound => write!(f, "OpenCL platform not found"),
            Self::DeviceNotAvailable => write!(f, "OpenCL device not available"),
        }
    }
}

impl std::error::Error for OpenClError {}

// ---------------------------------------------------------------------------
// RecoveryStrategy
// ---------------------------------------------------------------------------

/// What to do when a GPU kernel execution fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Retry on the GPU up to `max_attempts` times with exponential backoff.
    RetryGpu { max_attempts: u32, backoff_ms: u64 },
    /// Immediately fall back to the CPU implementation.
    FallbackCpu,
    /// Retry on the GPU first; if all retries are exhausted, fall back to CPU.
    RetryThenFallback { max_attempts: u32 },
    /// Propagate the error — no recovery is possible.
    Fail,
}

// ---------------------------------------------------------------------------
// ExecutionPath
// ---------------------------------------------------------------------------

/// Describes which path produced the final result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionPath {
    /// The GPU produced the result on its first (or only) attempt.
    Gpu,
    /// The GPU was retried and succeeded on attempt number `attempt`.
    GpuRetry { attempt: u32 },
    /// The CPU fallback was used after the GPU failed.
    CpuFallback { reason: OpenClError, attempts: u32 },
}

// ---------------------------------------------------------------------------
// RecoveryPolicy
// ---------------------------------------------------------------------------

/// Maps each [`OpenClError`] variant to a [`RecoveryStrategy`].
///
/// Construct with [`RecoveryPolicy::default()`] for sensible defaults, or use
/// the builder methods to customise individual mappings.
#[derive(Debug, Clone)]
pub struct RecoveryPolicy {
    device_lost: RecoveryStrategy,
    out_of_memory: RecoveryStrategy,
    compilation_failed: RecoveryStrategy,
    invalid_kernel: RecoveryStrategy,
    invalid_arg_size: RecoveryStrategy,
    invalid_work_group_size: RecoveryStrategy,
    timeout: RecoveryStrategy,
    driver_crash: RecoveryStrategy,
    platform_not_found: RecoveryStrategy,
    device_not_available: RecoveryStrategy,
}

impl Default for RecoveryPolicy {
    fn default() -> Self {
        Self {
            device_lost: RecoveryStrategy::FallbackCpu,
            out_of_memory: RecoveryStrategy::FallbackCpu,
            compilation_failed: RecoveryStrategy::Fail,
            invalid_kernel: RecoveryStrategy::Fail,
            invalid_arg_size: RecoveryStrategy::Fail,
            invalid_work_group_size: RecoveryStrategy::RetryGpu { max_attempts: 3, backoff_ms: 10 },
            timeout: RecoveryStrategy::RetryThenFallback { max_attempts: 2 },
            driver_crash: RecoveryStrategy::FallbackCpu,
            platform_not_found: RecoveryStrategy::FallbackCpu,
            device_not_available: RecoveryStrategy::FallbackCpu,
        }
    }
}

impl RecoveryPolicy {
    /// Create a new builder initialised with [`Default`] values.
    pub fn builder() -> RecoveryPolicyBuilder {
        RecoveryPolicyBuilder { policy: Self::default() }
    }

    /// Look up the strategy for a given error.
    pub fn strategy_for(&self, err: &OpenClError) -> &RecoveryStrategy {
        match err {
            OpenClError::DeviceLost => &self.device_lost,
            OpenClError::OutOfMemory => &self.out_of_memory,
            OpenClError::CompilationFailed => &self.compilation_failed,
            OpenClError::InvalidKernel => &self.invalid_kernel,
            OpenClError::InvalidArgSize => &self.invalid_arg_size,
            OpenClError::InvalidWorkGroupSize => &self.invalid_work_group_size,
            OpenClError::Timeout => &self.timeout,
            OpenClError::DriverCrash => &self.driver_crash,
            OpenClError::PlatformNotFound => &self.platform_not_found,
            OpenClError::DeviceNotAvailable => &self.device_not_available,
        }
    }
}

// ---------------------------------------------------------------------------
// RecoveryPolicyBuilder
// ---------------------------------------------------------------------------

/// Builder for [`RecoveryPolicy`].
#[derive(Debug, Clone)]
pub struct RecoveryPolicyBuilder {
    policy: RecoveryPolicy,
}

impl RecoveryPolicyBuilder {
    /// Override the strategy for a specific error variant.
    pub fn on_error(mut self, err: OpenClError, strategy: RecoveryStrategy) -> Self {
        match err {
            OpenClError::DeviceLost => self.policy.device_lost = strategy,
            OpenClError::OutOfMemory => self.policy.out_of_memory = strategy,
            OpenClError::CompilationFailed => self.policy.compilation_failed = strategy,
            OpenClError::InvalidKernel => self.policy.invalid_kernel = strategy,
            OpenClError::InvalidArgSize => self.policy.invalid_arg_size = strategy,
            OpenClError::InvalidWorkGroupSize => {
                self.policy.invalid_work_group_size = strategy;
            }
            OpenClError::Timeout => self.policy.timeout = strategy,
            OpenClError::DriverCrash => self.policy.driver_crash = strategy,
            OpenClError::PlatformNotFound => self.policy.platform_not_found = strategy,
            OpenClError::DeviceNotAvailable => self.policy.device_not_available = strategy,
        }
        self
    }

    /// Consume the builder and return the finished [`RecoveryPolicy`].
    pub fn build(self) -> RecoveryPolicy {
        self.policy
    }
}

// ---------------------------------------------------------------------------
// RecoveryStats
// ---------------------------------------------------------------------------

/// Cumulative statistics gathered by [`RecoveryExecutor`].
///
/// All counters are atomic so the struct can be shared across threads.
#[derive(Debug, Default)]
pub struct RecoveryStats {
    pub gpu_successes: AtomicU64,
    pub gpu_failures: AtomicU64,
    pub cpu_fallbacks: AtomicU64,
    pub retries: AtomicU64,
    // last_error is behind a parking_lot or std Mutex in a real implementation,
    // but for simplicity we use Option guarded externally.
}

impl RecoveryStats {
    /// Snapshot the current counters into plain integers.
    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            gpu_successes: self.gpu_successes.load(Ordering::Relaxed),
            gpu_failures: self.gpu_failures.load(Ordering::Relaxed),
            cpu_fallbacks: self.cpu_fallbacks.load(Ordering::Relaxed),
            retries: self.retries.load(Ordering::Relaxed),
        }
    }
}

/// Non-atomic snapshot of [`RecoveryStats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatsSnapshot {
    pub gpu_successes: u64,
    pub gpu_failures: u64,
    pub cpu_fallbacks: u64,
    pub retries: u64,
}

// ---------------------------------------------------------------------------
// RecoveryExecutor
// ---------------------------------------------------------------------------

/// Executes GPU kernels with automatic error recovery.
///
/// Wraps a [`RecoveryPolicy`] and accumulates [`RecoveryStats`].
pub struct RecoveryExecutor {
    policy: RecoveryPolicy,
    stats: RecoveryStats,
    last_error: Option<OpenClError>,
}

impl RecoveryExecutor {
    pub fn new(policy: RecoveryPolicy) -> Self {
        Self { policy, stats: RecoveryStats::default(), last_error: None }
    }

    /// Return a reference to the live stats.
    pub fn stats(&self) -> &RecoveryStats {
        &self.stats
    }

    /// Return the last OpenCL error observed, if any.
    pub fn last_error(&self) -> Option<OpenClError> {
        self.last_error
    }

    /// Execute `gpu_fn`. On failure, apply the recovery strategy from the
    /// policy, potentially retrying or falling back to `cpu_fallback`.
    ///
    /// Returns the result together with the [`ExecutionPath`] that produced it.
    pub fn execute<R>(
        &mut self,
        gpu_fn: impl Fn() -> Result<R, OpenClError>,
        cpu_fallback: impl FnOnce() -> Result<R, OpenClError>,
    ) -> Result<(R, ExecutionPath), OpenClError> {
        // First attempt on GPU.
        match gpu_fn() {
            Ok(val) => {
                self.stats.gpu_successes.fetch_add(1, Ordering::Relaxed);
                Ok((val, ExecutionPath::Gpu))
            }
            Err(first_err) => {
                self.last_error = Some(first_err);
                self.stats.gpu_failures.fetch_add(1, Ordering::Relaxed);
                let strategy = self.policy.strategy_for(&first_err).clone();
                self.apply_strategy(strategy, first_err, gpu_fn, cpu_fallback)
            }
        }
    }

    /// Internal: apply a [`RecoveryStrategy`] after the initial GPU attempt
    /// failed with `initial_err`.
    fn apply_strategy<R>(
        &mut self,
        strategy: RecoveryStrategy,
        initial_err: OpenClError,
        gpu_fn: impl Fn() -> Result<R, OpenClError>,
        cpu_fallback: impl FnOnce() -> Result<R, OpenClError>,
    ) -> Result<(R, ExecutionPath), OpenClError> {
        match strategy {
            RecoveryStrategy::FallbackCpu => {
                self.stats.cpu_fallbacks.fetch_add(1, Ordering::Relaxed);
                let val = cpu_fallback()?;
                Ok((val, ExecutionPath::CpuFallback { reason: initial_err, attempts: 1 }))
            }

            RecoveryStrategy::RetryGpu { max_attempts, backoff_ms } => {
                self.retry_gpu(max_attempts, backoff_ms, initial_err, &gpu_fn)
            }

            RecoveryStrategy::RetryThenFallback { max_attempts } => {
                match self.retry_gpu(max_attempts, 50, initial_err, &gpu_fn) {
                    Ok(val) => Ok(val),
                    Err(err) => {
                        self.stats.cpu_fallbacks.fetch_add(1, Ordering::Relaxed);
                        let val = cpu_fallback()?;
                        let attempts = max_attempts.saturating_add(1); // +1 for initial
                        Ok((val, ExecutionPath::CpuFallback { reason: err, attempts }))
                    }
                }
            }

            RecoveryStrategy::Fail => Err(initial_err),
        }
    }

    /// Retry `gpu_fn` up to `max_attempts` times with exponential backoff.
    fn retry_gpu<R>(
        &mut self,
        max_attempts: u32,
        backoff_ms: u64,
        initial_err: OpenClError,
        gpu_fn: &dyn Fn() -> Result<R, OpenClError>,
    ) -> Result<(R, ExecutionPath), OpenClError> {
        let mut last_err = initial_err;
        for attempt in 1..=max_attempts {
            self.stats.retries.fetch_add(1, Ordering::Relaxed);
            let sleep_ms = backoff_ms.saturating_mul(1u64 << attempt.min(10));
            thread::sleep(Duration::from_millis(sleep_ms));

            match gpu_fn() {
                Ok(val) => {
                    self.stats.gpu_successes.fetch_add(1, Ordering::Relaxed);
                    return Ok((val, ExecutionPath::GpuRetry { attempt: attempt + 1 }));
                }
                Err(err) => {
                    self.last_error = Some(err);
                    self.stats.gpu_failures.fetch_add(1, Ordering::Relaxed);
                    last_err = err;
                }
            }
        }
        Err(last_err)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    // ---- helpers -----------------------------------------------------------

    fn fail_gpu(err: OpenClError) -> impl Fn() -> Result<i32, OpenClError> {
        move || Err(err)
    }

    fn ok_cpu() -> Result<i32, OpenClError> {
        Ok(-1) // sentinel for "CPU produced this"
    }

    /// GPU that fails `n` times, then succeeds.
    fn flaky_gpu(n: u32) -> impl Fn() -> Result<i32, OpenClError> {
        let counter = AtomicU32::new(0);
        move || {
            let attempt = counter.fetch_add(1, Ordering::SeqCst);
            if attempt < n { Err(OpenClError::Timeout) } else { Ok(42) }
        }
    }

    // ---- GPU success -------------------------------------------------------

    #[test]
    fn gpu_success_returns_gpu_path() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());
        let (val, path) = exec.execute(|| Ok(99i32), ok_cpu).unwrap();
        assert_eq!(val, 99);
        assert_eq!(path, ExecutionPath::Gpu);
    }

    #[test]
    fn gpu_success_increments_gpu_successes() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());
        let _ = exec.execute(|| Ok(1i32), ok_cpu);
        let _ = exec.execute(|| Ok(2i32), ok_cpu);
        assert_eq!(exec.stats().gpu_successes.load(Ordering::Relaxed), 2);
    }

    // ---- FallbackCpu -------------------------------------------------------

    #[test]
    fn oom_triggers_cpu_fallback() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());
        let (val, path) = exec.execute(fail_gpu(OpenClError::OutOfMemory), ok_cpu).unwrap();
        assert_eq!(val, -1);
        assert_eq!(
            path,
            ExecutionPath::CpuFallback { reason: OpenClError::OutOfMemory, attempts: 1 }
        );
    }

    #[test]
    fn device_lost_triggers_cpu_fallback() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());
        let (_, path) = exec.execute(fail_gpu(OpenClError::DeviceLost), ok_cpu).unwrap();
        assert!(matches!(path, ExecutionPath::CpuFallback { reason: OpenClError::DeviceLost, .. }));
    }

    #[test]
    fn driver_crash_triggers_cpu_fallback() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());
        let (_, path) = exec.execute(fail_gpu(OpenClError::DriverCrash), ok_cpu).unwrap();
        assert!(matches!(
            path,
            ExecutionPath::CpuFallback { reason: OpenClError::DriverCrash, .. }
        ));
    }

    #[test]
    fn platform_not_found_triggers_cpu_fallback() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());
        let (_, path) = exec.execute(fail_gpu(OpenClError::PlatformNotFound), ok_cpu).unwrap();
        assert!(matches!(
            path,
            ExecutionPath::CpuFallback { reason: OpenClError::PlatformNotFound, .. }
        ));
    }

    #[test]
    fn device_not_available_triggers_cpu_fallback() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());
        let (_, path) = exec.execute(fail_gpu(OpenClError::DeviceNotAvailable), ok_cpu).unwrap();
        assert!(matches!(
            path,
            ExecutionPath::CpuFallback { reason: OpenClError::DeviceNotAvailable, .. }
        ));
    }

    #[test]
    fn cpu_fallback_always_succeeds_when_cpu_ok() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());
        // Every FallbackCpu-mapped error should succeed via CPU.
        for err in [
            OpenClError::DeviceLost,
            OpenClError::OutOfMemory,
            OpenClError::DriverCrash,
            OpenClError::PlatformNotFound,
            OpenClError::DeviceNotAvailable,
        ] {
            let result = exec.execute(fail_gpu(err), ok_cpu);
            assert!(result.is_ok(), "expected Ok for {err}");
        }
    }

    // ---- Fail strategy -----------------------------------------------------

    #[test]
    fn compilation_failed_does_not_retry() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());
        let result = exec.execute(fail_gpu(OpenClError::CompilationFailed), ok_cpu);
        assert!(result.is_err());
        assert_eq!(exec.stats().retries.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn invalid_kernel_propagates_error() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());
        let result = exec.execute(fail_gpu(OpenClError::InvalidKernel), ok_cpu);
        assert_eq!(result.unwrap_err(), OpenClError::InvalidKernel);
    }

    #[test]
    fn invalid_arg_size_propagates_error() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());
        let result = exec.execute(fail_gpu(OpenClError::InvalidArgSize), ok_cpu);
        assert_eq!(result.unwrap_err(), OpenClError::InvalidArgSize);
    }

    // ---- RetryGpu ----------------------------------------------------------

    #[test]
    fn retry_succeeds_on_second_attempt() {
        let policy = RecoveryPolicy::builder()
            .on_error(
                OpenClError::Timeout,
                RecoveryStrategy::RetryGpu { max_attempts: 3, backoff_ms: 1 },
            )
            .build();
        let mut exec = RecoveryExecutor::new(policy);
        let (val, path) = exec.execute(flaky_gpu(1), ok_cpu).unwrap();
        assert_eq!(val, 42);
        assert!(matches!(path, ExecutionPath::GpuRetry { attempt: 2 }));
    }

    #[test]
    fn invalid_work_group_size_retries() {
        let counter = AtomicU32::new(0);
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());
        let gpu = || {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            if n == 0 { Err(OpenClError::InvalidWorkGroupSize) } else { Ok(7i32) }
        };
        let (val, path) = exec.execute(gpu, ok_cpu).unwrap();
        assert_eq!(val, 7);
        assert!(matches!(path, ExecutionPath::GpuRetry { .. }));
    }

    #[test]
    fn retry_exhausted_returns_error() {
        let policy = RecoveryPolicy::builder()
            .on_error(
                OpenClError::Timeout,
                RecoveryStrategy::RetryGpu { max_attempts: 2, backoff_ms: 1 },
            )
            .build();
        let mut exec = RecoveryExecutor::new(policy);
        let result = exec.execute(fail_gpu(OpenClError::Timeout), ok_cpu);
        assert!(result.is_err());
        assert_eq!(exec.stats().retries.load(Ordering::Relaxed), 2);
    }

    // ---- RetryThenFallback -------------------------------------------------

    #[test]
    fn timeout_retry_then_fallback_succeeds_on_retry() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());
        // flaky_gpu(1) fails once then succeeds — the retry should catch it.
        let (val, path) = exec.execute(flaky_gpu(1), ok_cpu).unwrap();
        assert_eq!(val, 42);
        assert!(matches!(path, ExecutionPath::GpuRetry { .. }));
    }

    #[test]
    fn timeout_retry_exhausted_falls_back_to_cpu() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());
        // Always fail — retries exhausted, should fall back.
        let (val, path) = exec.execute(fail_gpu(OpenClError::Timeout), ok_cpu).unwrap();
        assert_eq!(val, -1);
        assert!(matches!(path, ExecutionPath::CpuFallback { reason: OpenClError::Timeout, .. }));
    }

    // ---- Stats tracking ----------------------------------------------------

    #[test]
    fn stats_track_across_multiple_executions() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());

        // 1 success
        let _ = exec.execute(|| Ok(1i32), ok_cpu);
        // 1 failure → CPU fallback
        let _ = exec.execute(fail_gpu(OpenClError::OutOfMemory), ok_cpu);
        // 1 success
        let _ = exec.execute(|| Ok(3i32), ok_cpu);

        let snap = exec.stats().snapshot();
        assert_eq!(snap.gpu_successes, 2);
        assert_eq!(snap.gpu_failures, 1);
        assert_eq!(snap.cpu_fallbacks, 1);
    }

    #[test]
    fn last_error_tracks_most_recent_failure() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());
        assert!(exec.last_error().is_none());

        let _ = exec.execute(fail_gpu(OpenClError::OutOfMemory), ok_cpu);
        assert_eq!(exec.last_error(), Some(OpenClError::OutOfMemory));

        let _ = exec.execute(fail_gpu(OpenClError::DeviceLost), ok_cpu);
        assert_eq!(exec.last_error(), Some(OpenClError::DeviceLost));
    }

    // ---- Builder pattern ---------------------------------------------------

    #[test]
    fn builder_overrides_default_strategy() {
        let policy = RecoveryPolicy::builder()
            .on_error(OpenClError::OutOfMemory, RecoveryStrategy::Fail)
            .build();
        assert_eq!(*policy.strategy_for(&OpenClError::OutOfMemory), RecoveryStrategy::Fail,);
        // Other defaults still intact.
        assert_eq!(*policy.strategy_for(&OpenClError::DeviceLost), RecoveryStrategy::FallbackCpu,);
    }

    #[test]
    fn builder_multiple_overrides() {
        let policy = RecoveryPolicy::builder()
            .on_error(OpenClError::OutOfMemory, RecoveryStrategy::Fail)
            .on_error(
                OpenClError::DeviceLost,
                RecoveryStrategy::RetryGpu { max_attempts: 5, backoff_ms: 100 },
            )
            .build();
        assert_eq!(*policy.strategy_for(&OpenClError::OutOfMemory), RecoveryStrategy::Fail,);
        assert_eq!(
            *policy.strategy_for(&OpenClError::DeviceLost),
            RecoveryStrategy::RetryGpu { max_attempts: 5, backoff_ms: 100 },
        );
    }

    #[test]
    fn custom_policy_oom_retry_then_fallback() {
        let policy = RecoveryPolicy::builder()
            .on_error(
                OpenClError::OutOfMemory,
                RecoveryStrategy::RetryThenFallback { max_attempts: 1 },
            )
            .build();
        let mut exec = RecoveryExecutor::new(policy);
        let (val, path) = exec.execute(fail_gpu(OpenClError::OutOfMemory), ok_cpu).unwrap();
        assert_eq!(val, -1);
        assert!(matches!(path, ExecutionPath::CpuFallback { .. }));
    }

    // ---- Display -----------------------------------------------------------

    #[test]
    fn all_opencl_error_variants_display_correctly() {
        let errors = [
            (OpenClError::DeviceLost, "OpenCL device lost"),
            (OpenClError::OutOfMemory, "OpenCL out of memory"),
            (OpenClError::CompilationFailed, "OpenCL kernel compilation failed"),
            (OpenClError::InvalidKernel, "OpenCL invalid kernel"),
            (OpenClError::InvalidArgSize, "OpenCL invalid argument size"),
            (OpenClError::InvalidWorkGroupSize, "OpenCL invalid work-group size"),
            (OpenClError::Timeout, "OpenCL execution timeout"),
            (OpenClError::DriverCrash, "OpenCL driver crash"),
            (OpenClError::PlatformNotFound, "OpenCL platform not found"),
            (OpenClError::DeviceNotAvailable, "OpenCL device not available"),
        ];
        for (err, expected) in errors {
            assert_eq!(err.to_string(), expected, "Display mismatch for {err:?}");
        }
    }

    // ---- Edge cases --------------------------------------------------------

    #[test]
    fn max_attempts_zero_skips_retries_and_returns_error() {
        let policy = RecoveryPolicy::builder()
            .on_error(
                OpenClError::Timeout,
                RecoveryStrategy::RetryGpu { max_attempts: 0, backoff_ms: 1 },
            )
            .build();
        let mut exec = RecoveryExecutor::new(policy);
        let result = exec.execute(fail_gpu(OpenClError::Timeout), ok_cpu);
        assert!(result.is_err());
        assert_eq!(exec.stats().retries.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn retry_then_fallback_max_attempts_zero_falls_back_immediately() {
        let policy = RecoveryPolicy::builder()
            .on_error(OpenClError::Timeout, RecoveryStrategy::RetryThenFallback { max_attempts: 0 })
            .build();
        let mut exec = RecoveryExecutor::new(policy);
        let (val, path) = exec.execute(fail_gpu(OpenClError::Timeout), ok_cpu).unwrap();
        assert_eq!(val, -1);
        assert!(matches!(path, ExecutionPath::CpuFallback { .. }));
    }

    #[test]
    fn nested_execute_calls_work() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());

        // First execution: GPU fails, falls back.
        let (v1, _) = exec.execute(fail_gpu(OpenClError::OutOfMemory), ok_cpu).unwrap();
        assert_eq!(v1, -1);

        // Second execution: GPU succeeds.
        let (v2, path) = exec.execute(|| Ok(100i32), ok_cpu).unwrap();
        assert_eq!(v2, 100);
        assert_eq!(path, ExecutionPath::Gpu);

        let snap = exec.stats().snapshot();
        assert_eq!(snap.gpu_successes, 1);
        assert_eq!(snap.cpu_fallbacks, 1);
    }

    #[test]
    fn cpu_fallback_error_propagates() {
        let mut exec = RecoveryExecutor::new(RecoveryPolicy::default());
        let result = exec.execute(
            fail_gpu(OpenClError::OutOfMemory),
            || Err(OpenClError::DeviceNotAvailable), // CPU also fails
        );
        assert_eq!(result.unwrap_err(), OpenClError::DeviceNotAvailable);
    }

    #[test]
    fn retry_stats_accumulate_correctly() {
        let policy = RecoveryPolicy::builder()
            .on_error(
                OpenClError::Timeout,
                RecoveryStrategy::RetryGpu { max_attempts: 3, backoff_ms: 1 },
            )
            .build();
        let mut exec = RecoveryExecutor::new(policy);

        // flaky_gpu(2): fails 2 times, succeeds on 3rd (attempt index 2).
        let (val, _) = exec.execute(flaky_gpu(2), ok_cpu).unwrap();
        assert_eq!(val, 42);

        let snap = exec.stats().snapshot();
        // initial fail + 1 retry fail = 2 gpu_failures, then success on retry 2
        assert_eq!(snap.gpu_failures, 2); // 1 initial + 1 retry that failed
        assert_eq!(snap.gpu_successes, 1);
        assert_eq!(snap.retries, 2); // 2 retry attempts (fail, then succeed)
    }

    #[test]
    fn default_policy_maps_all_errors() {
        let policy = RecoveryPolicy::default();
        // Just ensure strategy_for doesn't panic for every variant.
        let all_errors = [
            OpenClError::DeviceLost,
            OpenClError::OutOfMemory,
            OpenClError::CompilationFailed,
            OpenClError::InvalidKernel,
            OpenClError::InvalidArgSize,
            OpenClError::InvalidWorkGroupSize,
            OpenClError::Timeout,
            OpenClError::DriverCrash,
            OpenClError::PlatformNotFound,
            OpenClError::DeviceNotAvailable,
        ];
        for err in all_errors {
            let _ = policy.strategy_for(&err);
        }
    }
}
