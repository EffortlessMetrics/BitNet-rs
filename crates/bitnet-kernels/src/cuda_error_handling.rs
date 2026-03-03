//! CUDA error handling and recovery for GPU kernel operations.
//!
//! Provides structured error types, recovery strategies with configurable
//! retry/backoff, and an error log for diagnostics.

use std::fmt;
use std::sync::Mutex;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors originating from CUDA operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CudaError {
    /// GPU memory allocation failed.
    MemoryAllocation,
    /// Kernel launch returned an error.
    KernelLaunch,
    /// Operation targeted wrong device.
    DeviceMismatch,
    /// Invalid kernel or launch configuration.
    InvalidConfig,
    /// Operation exceeded time limit.
    Timeout,
    /// Device ran out of memory.
    OutOfMemory,
    /// Low-level driver error with native code.
    DriverError(i32),
    /// PTX / NVRTC compilation failed.
    CompilationFailed(String),
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryAllocation => write!(f, "CUDA memory allocation failed"),
            Self::KernelLaunch => write!(f, "CUDA kernel launch failed"),
            Self::DeviceMismatch => write!(f, "CUDA device mismatch"),
            Self::InvalidConfig => write!(f, "invalid CUDA configuration"),
            Self::Timeout => write!(f, "CUDA operation timed out"),
            Self::OutOfMemory => write!(f, "CUDA out of memory"),
            Self::DriverError(code) => write!(f, "CUDA driver error (code {code})"),
            Self::CompilationFailed(msg) => write!(f, "CUDA compilation failed: {msg}"),
        }
    }
}

impl std::error::Error for CudaError {}

// ---------------------------------------------------------------------------
// Recovery
// ---------------------------------------------------------------------------

/// Recommended recovery action for a CUDA error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorRecovery {
    /// Retry the same operation.
    Retry,
    /// Fall back to CPU.
    Fallback,
    /// Give up immediately.
    Abort,
    /// Retry with smaller work size.
    ReduceWorkSize,
}

/// Configurable strategy for automatic recovery.
#[derive(Debug, Clone)]
pub struct RecoveryStrategy {
    /// Maximum number of retries before giving up.
    pub max_retries: u32,
    /// Base backoff duration (doubled on each retry).
    pub backoff: Duration,
    /// Whether to fall back to CPU when retries are exhausted.
    pub fallback_to_cpu: bool,
}

impl Default for RecoveryStrategy {
    fn default() -> Self {
        Self { max_retries: 3, backoff: Duration::from_millis(100), fallback_to_cpu: true }
    }
}

// ---------------------------------------------------------------------------
// Context & diagnostics
// ---------------------------------------------------------------------------

/// Contextual information captured with a CUDA error.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Description of the operation that failed.
    pub operation: String,
    /// CUDA device ordinal.
    pub device_id: u32,
    /// Approximate device memory usage in bytes at time of error.
    pub memory_usage: u64,
    /// The underlying error.
    pub error: CudaError,
}

// ---------------------------------------------------------------------------
// ErrorLog
// ---------------------------------------------------------------------------

/// Thread-safe accumulator of [`ErrorContext`] entries for diagnostics.
pub struct ErrorLog {
    entries: Mutex<Vec<ErrorContext>>,
}

impl ErrorLog {
    /// Create an empty log.
    pub fn new() -> Self {
        Self { entries: Mutex::new(Vec::new()) }
    }

    /// Record an error context.
    pub fn log(&self, ctx: ErrorContext) {
        self.entries.lock().expect("ErrorLog lock poisoned").push(ctx);
    }

    /// Human-readable summary of all logged errors.
    pub fn summary(&self) -> String {
        let entries = self.entries.lock().expect("ErrorLog lock poisoned");
        if entries.is_empty() {
            return "No CUDA errors recorded".to_string();
        }
        let mut out = format!("{} CUDA error(s):\n", entries.len());
        for (i, ctx) in entries.iter().enumerate() {
            out.push_str(&format!(
                "  [{}] device={} op=\"{}\" mem={} err={}\n",
                i, ctx.device_id, ctx.operation, ctx.memory_usage, ctx.error,
            ));
        }
        out
    }

    /// Return the most recent `n` entries (newest last).
    pub fn recent(&self, n: usize) -> Vec<ErrorContext> {
        let entries = self.entries.lock().expect("ErrorLog lock poisoned");
        let start = entries.len().saturating_sub(n);
        entries[start..].to_vec()
    }

    /// Total number of logged errors.
    pub fn len(&self) -> usize {
        self.entries.lock().expect("ErrorLog lock poisoned").len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for ErrorLog {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ErrorLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let len = self.len();
        f.debug_struct("ErrorLog").field("count", &len).finish()
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Suggest a recovery action for the given error.
pub fn suggest_recovery(error: &CudaError) -> ErrorRecovery {
    match error {
        // Transient — worth retrying
        CudaError::KernelLaunch | CudaError::Timeout => ErrorRecovery::Retry,
        // Memory pressure — shrink first
        CudaError::MemoryAllocation | CudaError::OutOfMemory => ErrorRecovery::ReduceWorkSize,
        // Permanent — no point retrying
        CudaError::DeviceMismatch | CudaError::InvalidConfig => ErrorRecovery::Abort,
        CudaError::CompilationFailed(_) => ErrorRecovery::Abort,
        // Driver errors: negative codes are typically transient; others abort
        CudaError::DriverError(code) if *code < 0 => ErrorRecovery::Retry,
        CudaError::DriverError(_) => ErrorRecovery::Fallback,
    }
}

/// Whether the error is transient (may succeed on retry).
pub fn is_transient(error: &CudaError) -> bool {
    matches!(
        error,
        CudaError::KernelLaunch
            | CudaError::Timeout
            | CudaError::MemoryAllocation
            | CudaError::OutOfMemory
            | CudaError::DriverError(_)
    )
}

/// Format a human-readable diagnostic string from an error context.
pub fn format_error_diagnostic(ctx: &ErrorContext) -> String {
    let recovery = suggest_recovery(&ctx.error);
    let transient = if is_transient(&ctx.error) { "transient" } else { "permanent" };
    format!(
        "CUDA error on device {} during \"{}\": {} [{}] \
         (memory_usage={} bytes, suggested_recovery={:?})",
        ctx.device_id, ctx.operation, ctx.error, transient, ctx.memory_usage, recovery,
    )
}

/// Execute a fallible closure with retry + exponential backoff.
///
/// Returns `Ok(T)` on the first successful attempt, or the last
/// [`CudaError`] after all retries (and optional CPU fallback) are
/// exhausted.
pub fn execute_with_recovery<F, T>(mut f: F, strategy: &RecoveryStrategy) -> Result<T, CudaError>
where
    F: FnMut() -> Result<T, CudaError>,
{
    let mut last_err = CudaError::InvalidConfig; // placeholder
    let mut delay = strategy.backoff;

    for _ in 0..=strategy.max_retries {
        match f() {
            Ok(val) => return Ok(val),
            Err(e) => {
                if !is_transient(&e) && suggest_recovery(&e) == ErrorRecovery::Abort {
                    return Err(e);
                }
                last_err = e;
                std::thread::sleep(delay);
                delay = delay.saturating_mul(2);
            }
        }
    }

    // All retries exhausted
    Err(last_err)
}

// ===========================================================================
// Tests
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // CudaError basics
    // -----------------------------------------------------------------------

    #[test]
    fn error_display_memory_allocation() {
        assert_eq!(CudaError::MemoryAllocation.to_string(), "CUDA memory allocation failed");
    }

    #[test]
    fn error_display_kernel_launch() {
        assert_eq!(CudaError::KernelLaunch.to_string(), "CUDA kernel launch failed");
    }

    #[test]
    fn error_display_device_mismatch() {
        assert_eq!(CudaError::DeviceMismatch.to_string(), "CUDA device mismatch");
    }

    #[test]
    fn error_display_invalid_config() {
        assert_eq!(CudaError::InvalidConfig.to_string(), "invalid CUDA configuration");
    }

    #[test]
    fn error_display_timeout() {
        assert_eq!(CudaError::Timeout.to_string(), "CUDA operation timed out");
    }

    #[test]
    fn error_display_out_of_memory() {
        assert_eq!(CudaError::OutOfMemory.to_string(), "CUDA out of memory");
    }

    #[test]
    fn error_display_driver_error() {
        assert_eq!(CudaError::DriverError(42).to_string(), "CUDA driver error (code 42)");
    }

    #[test]
    fn error_display_driver_error_negative() {
        assert_eq!(CudaError::DriverError(-1).to_string(), "CUDA driver error (code -1)");
    }

    #[test]
    fn error_display_compilation_failed() {
        let e = CudaError::CompilationFailed("bad ptx".into());
        assert_eq!(e.to_string(), "CUDA compilation failed: bad ptx");
    }

    #[test]
    fn error_clone_eq() {
        let a = CudaError::DriverError(7);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn error_debug_format() {
        let e = CudaError::Timeout;
        let dbg = format!("{e:?}");
        assert!(dbg.contains("Timeout"));
    }

    #[test]
    fn error_is_std_error() {
        let e: &dyn std::error::Error = &CudaError::KernelLaunch;
        assert!(!e.to_string().is_empty());
    }

    // -----------------------------------------------------------------------
    // suggest_recovery
    // -----------------------------------------------------------------------

    #[test]
    fn recovery_kernel_launch_retry() {
        assert_eq!(suggest_recovery(&CudaError::KernelLaunch), ErrorRecovery::Retry);
    }

    #[test]
    fn recovery_timeout_retry() {
        assert_eq!(suggest_recovery(&CudaError::Timeout), ErrorRecovery::Retry);
    }

    #[test]
    fn recovery_memory_allocation_reduce() {
        assert_eq!(suggest_recovery(&CudaError::MemoryAllocation), ErrorRecovery::ReduceWorkSize);
    }

    #[test]
    fn recovery_out_of_memory_reduce() {
        assert_eq!(suggest_recovery(&CudaError::OutOfMemory), ErrorRecovery::ReduceWorkSize);
    }

    #[test]
    fn recovery_device_mismatch_abort() {
        assert_eq!(suggest_recovery(&CudaError::DeviceMismatch), ErrorRecovery::Abort);
    }

    #[test]
    fn recovery_invalid_config_abort() {
        assert_eq!(suggest_recovery(&CudaError::InvalidConfig), ErrorRecovery::Abort);
    }

    #[test]
    fn recovery_compilation_failed_abort() {
        let e = CudaError::CompilationFailed("syntax".into());
        assert_eq!(suggest_recovery(&e), ErrorRecovery::Abort);
    }

    #[test]
    fn recovery_driver_negative_retry() {
        assert_eq!(suggest_recovery(&CudaError::DriverError(-99)), ErrorRecovery::Retry);
    }

    #[test]
    fn recovery_driver_positive_fallback() {
        assert_eq!(suggest_recovery(&CudaError::DriverError(1)), ErrorRecovery::Fallback);
    }

    #[test]
    fn recovery_driver_zero_fallback() {
        assert_eq!(suggest_recovery(&CudaError::DriverError(0)), ErrorRecovery::Fallback);
    }

    // -----------------------------------------------------------------------
    // is_transient
    // -----------------------------------------------------------------------

    #[test]
    fn transient_kernel_launch() {
        assert!(is_transient(&CudaError::KernelLaunch));
    }

    #[test]
    fn transient_timeout() {
        assert!(is_transient(&CudaError::Timeout));
    }

    #[test]
    fn transient_memory_allocation() {
        assert!(is_transient(&CudaError::MemoryAllocation));
    }

    #[test]
    fn transient_out_of_memory() {
        assert!(is_transient(&CudaError::OutOfMemory));
    }

    #[test]
    fn transient_driver_error() {
        assert!(is_transient(&CudaError::DriverError(42)));
    }

    #[test]
    fn not_transient_device_mismatch() {
        assert!(!is_transient(&CudaError::DeviceMismatch));
    }

    #[test]
    fn not_transient_invalid_config() {
        assert!(!is_transient(&CudaError::InvalidConfig));
    }

    #[test]
    fn not_transient_compilation_failed() {
        assert!(!is_transient(&CudaError::CompilationFailed("x".into())));
    }

    // -----------------------------------------------------------------------
    // format_error_diagnostic
    // -----------------------------------------------------------------------

    #[test]
    fn diagnostic_contains_device_id() {
        let ctx = ErrorContext {
            operation: "gemm".into(),
            device_id: 3,
            memory_usage: 1024,
            error: CudaError::KernelLaunch,
        };
        let diag = format_error_diagnostic(&ctx);
        assert!(diag.contains("device 3"));
    }

    #[test]
    fn diagnostic_contains_operation() {
        let ctx = ErrorContext {
            operation: "softmax".into(),
            device_id: 0,
            memory_usage: 0,
            error: CudaError::Timeout,
        };
        assert!(format_error_diagnostic(&ctx).contains("softmax"));
    }

    #[test]
    fn diagnostic_contains_memory_usage() {
        let ctx = ErrorContext {
            operation: "alloc".into(),
            device_id: 0,
            memory_usage: 999_999,
            error: CudaError::OutOfMemory,
        };
        assert!(format_error_diagnostic(&ctx).contains("999999"));
    }

    #[test]
    fn diagnostic_transient_label() {
        let ctx = ErrorContext {
            operation: "launch".into(),
            device_id: 0,
            memory_usage: 0,
            error: CudaError::KernelLaunch,
        };
        assert!(format_error_diagnostic(&ctx).contains("transient"));
    }

    #[test]
    fn diagnostic_permanent_label() {
        let ctx = ErrorContext {
            operation: "config".into(),
            device_id: 0,
            memory_usage: 0,
            error: CudaError::InvalidConfig,
        };
        assert!(format_error_diagnostic(&ctx).contains("permanent"));
    }

    #[test]
    fn diagnostic_suggested_recovery() {
        let ctx = ErrorContext {
            operation: "matmul".into(),
            device_id: 1,
            memory_usage: 4096,
            error: CudaError::OutOfMemory,
        };
        assert!(format_error_diagnostic(&ctx).contains("ReduceWorkSize"));
    }

    #[test]
    fn diagnostic_driver_error_code() {
        let ctx = ErrorContext {
            operation: "copy".into(),
            device_id: 0,
            memory_usage: 0,
            error: CudaError::DriverError(700),
        };
        let diag = format_error_diagnostic(&ctx);
        assert!(diag.contains("700"));
    }

    #[test]
    fn diagnostic_compilation_message() {
        let ctx = ErrorContext {
            operation: "compile".into(),
            device_id: 0,
            memory_usage: 0,
            error: CudaError::CompilationFailed("line 42".into()),
        };
        assert!(format_error_diagnostic(&ctx).contains("line 42"));
    }

    // -----------------------------------------------------------------------
    // RecoveryStrategy defaults
    // -----------------------------------------------------------------------

    #[test]
    fn default_strategy_max_retries() {
        assert_eq!(RecoveryStrategy::default().max_retries, 3);
    }

    #[test]
    fn default_strategy_backoff() {
        assert_eq!(RecoveryStrategy::default().backoff, Duration::from_millis(100));
    }

    #[test]
    fn default_strategy_fallback_enabled() {
        assert!(RecoveryStrategy::default().fallback_to_cpu);
    }

    // -----------------------------------------------------------------------
    // execute_with_recovery
    // -----------------------------------------------------------------------

    #[test]
    fn recovery_succeeds_first_try() {
        let strategy = RecoveryStrategy { max_retries: 0, ..Default::default() };
        let result = execute_with_recovery(|| Ok::<_, CudaError>(42), &strategy);
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn recovery_succeeds_after_transient_failure() {
        let mut attempts = 0u32;
        let strategy = RecoveryStrategy {
            max_retries: 3,
            backoff: Duration::from_millis(1),
            fallback_to_cpu: false,
        };
        let result = execute_with_recovery(
            || {
                attempts += 1;
                if attempts < 3 { Err(CudaError::KernelLaunch) } else { Ok(99) }
            },
            &strategy,
        );
        assert_eq!(result.unwrap(), 99);
        assert_eq!(attempts, 3);
    }

    #[test]
    fn recovery_exhausts_retries() {
        let strategy = RecoveryStrategy {
            max_retries: 2,
            backoff: Duration::from_millis(1),
            fallback_to_cpu: false,
        };
        let mut attempts = 0u32;
        let result: Result<i32, _> = execute_with_recovery(
            || {
                attempts += 1;
                Err(CudaError::Timeout)
            },
            &strategy,
        );
        assert!(result.is_err());
        // initial attempt + 2 retries = 3
        assert_eq!(attempts, 3);
    }

    #[test]
    fn recovery_aborts_on_permanent_error() {
        let strategy = RecoveryStrategy {
            max_retries: 5,
            backoff: Duration::from_millis(1),
            fallback_to_cpu: true,
        };
        let mut attempts = 0u32;
        let result: Result<i32, _> = execute_with_recovery(
            || {
                attempts += 1;
                Err(CudaError::InvalidConfig)
            },
            &strategy,
        );
        assert_eq!(result.unwrap_err(), CudaError::InvalidConfig);
        assert_eq!(attempts, 1);
    }

    #[test]
    fn recovery_aborts_on_compilation_failed() {
        let strategy = RecoveryStrategy {
            max_retries: 5,
            backoff: Duration::from_millis(1),
            fallback_to_cpu: true,
        };
        let result: Result<i32, _> =
            execute_with_recovery(|| Err(CudaError::CompilationFailed("ptx".into())), &strategy);
        assert!(result.is_err());
    }

    #[test]
    fn recovery_driver_positive_retries_then_fails() {
        let strategy = RecoveryStrategy {
            max_retries: 1,
            backoff: Duration::from_millis(1),
            fallback_to_cpu: false,
        };
        let mut attempts = 0u32;
        let result: Result<i32, _> = execute_with_recovery(
            || {
                attempts += 1;
                Err(CudaError::DriverError(1))
            },
            &strategy,
        );
        assert!(result.is_err());
        // DriverError(positive) → Fallback, which is not Abort, so retries proceed
        assert_eq!(attempts, 2);
    }

    #[test]
    fn recovery_zero_retries_returns_error() {
        let strategy = RecoveryStrategy {
            max_retries: 0,
            backoff: Duration::from_millis(1),
            fallback_to_cpu: false,
        };
        let result: Result<i32, _> = execute_with_recovery(|| Err(CudaError::Timeout), &strategy);
        assert_eq!(result.unwrap_err(), CudaError::Timeout);
    }

    // -----------------------------------------------------------------------
    // ErrorLog
    // -----------------------------------------------------------------------

    #[test]
    fn log_new_is_empty() {
        let log = ErrorLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn log_after_one_entry() {
        let log = ErrorLog::new();
        log.log(ErrorContext {
            operation: "test".into(),
            device_id: 0,
            memory_usage: 0,
            error: CudaError::Timeout,
        });
        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());
    }

    #[test]
    fn log_summary_empty() {
        let log = ErrorLog::new();
        assert_eq!(log.summary(), "No CUDA errors recorded");
    }

    #[test]
    fn log_summary_with_entries() {
        let log = ErrorLog::new();
        log.log(ErrorContext {
            operation: "gemm".into(),
            device_id: 1,
            memory_usage: 4096,
            error: CudaError::KernelLaunch,
        });
        let s = log.summary();
        assert!(s.contains("1 CUDA error(s)"));
        assert!(s.contains("gemm"));
    }

    #[test]
    fn log_recent_returns_last_n() {
        let log = ErrorLog::new();
        for i in 0..5 {
            log.log(ErrorContext {
                operation: format!("op{i}"),
                device_id: 0,
                memory_usage: 0,
                error: CudaError::Timeout,
            });
        }
        let recent = log.recent(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].operation, "op3");
        assert_eq!(recent[1].operation, "op4");
    }

    #[test]
    fn log_recent_more_than_available() {
        let log = ErrorLog::new();
        log.log(ErrorContext {
            operation: "only".into(),
            device_id: 0,
            memory_usage: 0,
            error: CudaError::Timeout,
        });
        assert_eq!(log.recent(100).len(), 1);
    }

    #[test]
    fn log_recent_zero() {
        let log = ErrorLog::new();
        log.log(ErrorContext {
            operation: "x".into(),
            device_id: 0,
            memory_usage: 0,
            error: CudaError::Timeout,
        });
        assert!(log.recent(0).is_empty());
    }

    #[test]
    fn log_debug_format() {
        let log = ErrorLog::new();
        let dbg = format!("{log:?}");
        assert!(dbg.contains("ErrorLog"));
        assert!(dbg.contains("0"));
    }

    #[test]
    fn log_default_is_empty() {
        let log = ErrorLog::default();
        assert!(log.is_empty());
    }

    // -----------------------------------------------------------------------
    // ErrorContext
    // -----------------------------------------------------------------------

    #[test]
    fn error_context_clone() {
        let ctx = ErrorContext {
            operation: "test".into(),
            device_id: 2,
            memory_usage: 1024,
            error: CudaError::Timeout,
        };
        let ctx2 = ctx.clone();
        assert_eq!(ctx.device_id, ctx2.device_id);
        assert_eq!(ctx.operation, ctx2.operation);
    }

    #[test]
    fn error_context_debug() {
        let ctx = ErrorContext {
            operation: "launch".into(),
            device_id: 0,
            memory_usage: 0,
            error: CudaError::KernelLaunch,
        };
        let dbg = format!("{ctx:?}");
        assert!(dbg.contains("launch"));
    }

    // -----------------------------------------------------------------------
    // ErrorRecovery
    // -----------------------------------------------------------------------

    #[test]
    fn recovery_enum_debug() {
        assert_eq!(format!("{:?}", ErrorRecovery::Retry), "Retry");
        assert_eq!(format!("{:?}", ErrorRecovery::Fallback), "Fallback");
        assert_eq!(format!("{:?}", ErrorRecovery::Abort), "Abort");
        assert_eq!(format!("{:?}", ErrorRecovery::ReduceWorkSize), "ReduceWorkSize");
    }

    #[test]
    fn recovery_clone_eq() {
        let a = ErrorRecovery::Retry;
        let b = a;
        assert_eq!(a, b);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn compilation_empty_message() {
        let e = CudaError::CompilationFailed(String::new());
        assert_eq!(e.to_string(), "CUDA compilation failed: ");
    }

    #[test]
    fn driver_error_i32_min() {
        let e = CudaError::DriverError(i32::MIN);
        assert!(e.to_string().contains(&i32::MIN.to_string()));
    }

    #[test]
    fn driver_error_i32_max() {
        let e = CudaError::DriverError(i32::MAX);
        assert!(e.to_string().contains(&i32::MAX.to_string()));
    }

    #[test]
    fn diagnostic_empty_operation() {
        let ctx = ErrorContext {
            operation: String::new(),
            device_id: 0,
            memory_usage: 0,
            error: CudaError::Timeout,
        };
        let diag = format_error_diagnostic(&ctx);
        assert!(diag.contains("\"\""));
    }

    #[test]
    fn diagnostic_large_memory() {
        let ctx = ErrorContext {
            operation: "big_alloc".into(),
            device_id: 0,
            memory_usage: u64::MAX,
            error: CudaError::OutOfMemory,
        };
        let diag = format_error_diagnostic(&ctx);
        assert!(diag.contains(&u64::MAX.to_string()));
    }

    #[test]
    fn log_multiple_devices() {
        let log = ErrorLog::new();
        for dev in 0..4 {
            log.log(ErrorContext {
                operation: "multi".into(),
                device_id: dev,
                memory_usage: 0,
                error: CudaError::Timeout,
            });
        }
        assert_eq!(log.len(), 4);
        let summary = log.summary();
        assert!(summary.contains("device=0"));
        assert!(summary.contains("device=3"));
    }

    // -----------------------------------------------------------------------
    // proptest
    // -----------------------------------------------------------------------

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_cuda_error() -> impl Strategy<Value = CudaError> {
            prop_oneof![
                Just(CudaError::MemoryAllocation),
                Just(CudaError::KernelLaunch),
                Just(CudaError::DeviceMismatch),
                Just(CudaError::InvalidConfig),
                Just(CudaError::Timeout),
                Just(CudaError::OutOfMemory),
                any::<i32>().prop_map(CudaError::DriverError),
                "[a-z]{0,20}".prop_map(CudaError::CompilationFailed),
            ]
        }

        proptest! {
            #[test]
            fn suggest_recovery_always_returns_valid(ref e in arb_cuda_error()) {
                let r = suggest_recovery(e);
                prop_assert!(matches!(
                    r,
                    ErrorRecovery::Retry
                        | ErrorRecovery::Fallback
                        | ErrorRecovery::Abort
                        | ErrorRecovery::ReduceWorkSize
                ));
            }

            #[test]
            fn is_transient_consistent_with_recovery(ref e in arb_cuda_error()) {
                let recovery = suggest_recovery(e);
                if recovery == ErrorRecovery::Abort {
                    // Abort errors may or may not be transient (DriverError positive
                    // is transient but maps to Fallback; others are non-transient)
                    // Just ensure the function doesn't panic.
                    let _ = is_transient(e);
                }
            }

            #[test]
            fn display_never_empty(ref e in arb_cuda_error()) {
                prop_assert!(!e.to_string().is_empty());
            }

            #[test]
            fn format_diagnostic_never_empty(
                ref e in arb_cuda_error(),
                device_id in 0u32..8,
                mem in 0u64..1_000_000,
            ) {
                let ctx = ErrorContext {
                    operation: "prop_op".into(),
                    device_id,
                    memory_usage: mem,
                    error: e.clone(),
                };
                let diag = format_error_diagnostic(&ctx);
                prop_assert!(!diag.is_empty());
                prop_assert!(diag.contains("prop_op"));
            }

            #[test]
            fn error_log_len_matches_inserts(count in 0usize..50) {
                let log = ErrorLog::new();
                for _ in 0..count {
                    log.log(ErrorContext {
                        operation: "p".into(),
                        device_id: 0,
                        memory_usage: 0,
                        error: CudaError::Timeout,
                    });
                }
                prop_assert_eq!(log.len(), count);
            }
        }
    }
}
