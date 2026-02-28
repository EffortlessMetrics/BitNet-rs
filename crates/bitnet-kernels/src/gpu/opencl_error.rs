//! Typed OpenCL errors and CPU fallback logic.
//!
//! Maps raw OpenCL error codes to a Rust enum so callers can pattern-match
//! on specific failure modes (out of memory, device lost, etc.).
//!
//! The [`FallbackProvider`] wraps an OpenCL kernel and transparently
//! falls back to CPU when the GPU fails, with configurable retry limits
//! controlled by `BITNET_OPENCL_MAX_RETRIES` (default: 3).

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

// ── OpenCL error enum ──────────────────────────────────────────────

/// Categorised OpenCL error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenClError {
    /// Device was lost or reset during execution.
    DeviceLost(String),
    /// GPU ran out of memory or resources.
    OutOfMemory(String),
    /// Kernel program failed to compile.
    BuildFailed(String),
    /// Kernel name not found in the compiled program.
    InvalidKernel(String),
    /// Invalid argument passed to a kernel.
    InvalidArgument(String),
    /// Any other OpenCL error not specifically categorised.
    Other { code: i32, detail: String },
}

impl OpenClError {
    /// Create from a raw OpenCL error code and optional context message.
    pub fn from_code(code: i32, detail: impl Into<String>) -> Self {
        let detail = detail.into();
        match code {
            // CL_DEVICE_NOT_FOUND / CL_DEVICE_NOT_AVAILABLE
            -1 | -2 => Self::DeviceLost(detail),
            // CL_MEM_OBJECT_ALLOCATION_FAILURE / CL_OUT_OF_RESOURCES / CL_OUT_OF_HOST_MEMORY
            -4 | -5 | -6 => Self::OutOfMemory(detail),
            // CL_BUILD_PROGRAM_FAILURE
            -11 => Self::BuildFailed(detail),
            // CL_INVALID_KERNEL_NAME / CL_INVALID_KERNEL
            -46 | -48 => Self::InvalidKernel(detail),
            // CL_INVALID_ARG_INDEX / CL_INVALID_ARG_VALUE / CL_INVALID_ARG_SIZE
            -49 | -50 | -51 => Self::InvalidArgument(detail),
            _ => Self::Other { code, detail },
        }
    }

    /// Returns `true` when a retry with smaller buffers might succeed.
    pub fn is_retriable_with_smaller_buffers(&self) -> bool {
        matches!(self, Self::OutOfMemory(_))
    }

    /// Returns `true` when the GPU should be permanently disabled.
    pub fn is_fatal(&self) -> bool {
        matches!(self, Self::DeviceLost(_) | Self::BuildFailed(_))
    }
}

impl fmt::Display for OpenClError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceLost(d) => write!(f, "OpenCL device lost: {}", d),
            Self::OutOfMemory(d) => write!(f, "OpenCL out of memory: {}", d),
            Self::BuildFailed(d) => write!(f, "OpenCL kernel build failed: {}", d),
            Self::InvalidKernel(d) => write!(f, "OpenCL invalid kernel: {}", d),
            Self::InvalidArgument(d) => write!(f, "OpenCL invalid argument: {}", d),
            Self::Other { code, detail } => {
                write!(f, "OpenCL error (code {}): {}", code, detail)
            }
        }
    }
}

impl std::error::Error for OpenClError {}

// ── Fallback provider ──────────────────────────────────────────────

/// Read `BITNET_OPENCL_MAX_RETRIES` from the environment (default: 3).
fn max_retries() -> u32 {
    std::env::var("BITNET_OPENCL_MAX_RETRIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
}

/// Wraps an OpenCL kernel provider with automatic CPU fallback.
///
/// If the GPU kernel fails, the operation is retried on the CPU.
/// After `BITNET_OPENCL_MAX_RETRIES` consecutive GPU failures the GPU
/// is permanently disabled for the lifetime of this provider.
pub struct FallbackProvider {
    /// Consecutive failure counter.
    consecutive_failures: AtomicU32,
    /// Once set, GPU is permanently disabled.
    gpu_disabled: AtomicU32, // 0 = enabled, 1 = disabled
    /// Maximum consecutive failures before disabling GPU.
    max_retries: u32,
}

impl FallbackProvider {
    /// Create a new fallback provider.
    pub fn new() -> Self {
        Self {
            consecutive_failures: AtomicU32::new(0),
            gpu_disabled: AtomicU32::new(0),
            max_retries: max_retries(),
        }
    }

    /// Returns `true` when the GPU has been permanently disabled.
    pub fn is_gpu_disabled(&self) -> bool {
        self.gpu_disabled.load(Ordering::Relaxed) != 0
    }

    /// Record a successful GPU operation (resets failure counter).
    pub fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
    }

    /// Record a GPU failure.
    ///
    /// Returns `true` when the GPU has just been permanently disabled
    /// (i.e. the failure count crossed the threshold).
    pub fn record_failure(&self, error: &OpenClError) -> bool {
        if error.is_fatal() {
            log::warn!(
                "Fatal OpenCL error, disabling GPU permanently: {}",
                error
            );
            self.gpu_disabled.store(1, Ordering::Relaxed);
            return true;
        }

        let prev = self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
        let new_count = prev + 1;

        if new_count >= self.max_retries {
            log::warn!(
                "OpenCL kernel failed {} consecutive times (max {}), disabling GPU: {}",
                new_count,
                self.max_retries,
                error
            );
            self.gpu_disabled.store(1, Ordering::Relaxed);
            true
        } else {
            log::warn!(
                "OpenCL kernel failed ({}/{}), falling back to CPU: {}",
                new_count,
                self.max_retries,
                error
            );
            false
        }
    }

    /// Current consecutive failure count.
    pub fn failure_count(&self) -> u32 {
        self.consecutive_failures.load(Ordering::Relaxed)
    }
}

impl Default for FallbackProvider {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- Error mapping tests --

    #[test]
    fn device_lost_from_code() {
        let e = OpenClError::from_code(-1, "gone");
        assert!(matches!(e, OpenClError::DeviceLost(_)));
        assert!(e.is_fatal());
    }

    #[test]
    fn out_of_memory_from_code() {
        for code in [-4, -5, -6] {
            let e = OpenClError::from_code(code, "oom");
            assert!(matches!(e, OpenClError::OutOfMemory(_)), "code {}", code);
            assert!(e.is_retriable_with_smaller_buffers());
            assert!(!e.is_fatal());
        }
    }

    #[test]
    fn build_failed_from_code() {
        let e = OpenClError::from_code(-11, "syntax");
        assert!(matches!(e, OpenClError::BuildFailed(_)));
        assert!(e.is_fatal());
    }

    #[test]
    fn invalid_kernel_from_code() {
        let e = OpenClError::from_code(-46, "bad name");
        assert!(matches!(e, OpenClError::InvalidKernel(_)));
    }

    #[test]
    fn invalid_argument_from_code() {
        let e = OpenClError::from_code(-50, "bad arg");
        assert!(matches!(e, OpenClError::InvalidArgument(_)));
    }

    #[test]
    fn unknown_code_maps_to_other() {
        let e = OpenClError::from_code(-999, "mystery");
        assert!(matches!(e, OpenClError::Other { code: -999, .. }));
    }

    #[test]
    fn display_impl() {
        let e = OpenClError::DeviceLost("test".into());
        let s = format!("{}", e);
        assert!(s.contains("device lost"), "{}", s);
    }

    #[test]
    fn std_error_impl() {
        let e: Box<dyn std::error::Error> =
            Box::new(OpenClError::OutOfMemory("test".into()));
        assert!(e.to_string().contains("out of memory"));
    }

    // -- Fallback provider tests --

    #[test]
    fn fallback_provider_starts_enabled() {
        let fp = FallbackProvider::new();
        assert!(!fp.is_gpu_disabled());
        assert_eq!(fp.failure_count(), 0);
    }

    #[test]
    fn record_success_resets_counter() {
        let fp = FallbackProvider::new();
        fp.record_failure(&OpenClError::Other {
            code: -99,
            detail: "test".into(),
        });
        assert_eq!(fp.failure_count(), 1);
        fp.record_success();
        assert_eq!(fp.failure_count(), 0);
    }

    #[test]
    fn gpu_disabled_after_max_retries() {
        let fp = FallbackProvider {
            consecutive_failures: AtomicU32::new(0),
            gpu_disabled: AtomicU32::new(0),
            max_retries: 3,
        };
        let err = OpenClError::Other { code: -99, detail: "flaky".into() };

        assert!(!fp.record_failure(&err)); // 1/3
        assert!(!fp.record_failure(&err)); // 2/3
        assert!(fp.record_failure(&err)); // 3/3 → disabled
        assert!(fp.is_gpu_disabled());
    }

    #[test]
    fn fatal_error_disables_immediately() {
        let fp = FallbackProvider::new();
        let err = OpenClError::DeviceLost("hardware failure".into());
        assert!(fp.record_failure(&err));
        assert!(fp.is_gpu_disabled());
    }

    #[test]
    fn non_fatal_below_threshold_keeps_gpu() {
        let fp = FallbackProvider {
            consecutive_failures: AtomicU32::new(0),
            gpu_disabled: AtomicU32::new(0),
            max_retries: 5,
        };
        let err = OpenClError::OutOfMemory("oom".into());
        fp.record_failure(&err);
        fp.record_failure(&err);
        assert!(!fp.is_gpu_disabled());
    }

    #[test]
    fn success_after_failures_resets() {
        let fp = FallbackProvider {
            consecutive_failures: AtomicU32::new(0),
            gpu_disabled: AtomicU32::new(0),
            max_retries: 3,
        };
        let err = OpenClError::Other { code: -99, detail: "x".into() };
        fp.record_failure(&err); // 1
        fp.record_failure(&err); // 2
        fp.record_success(); // reset
        fp.record_failure(&err); // 1 again
        assert!(!fp.is_gpu_disabled());
        assert_eq!(fp.failure_count(), 1);
    }

    #[test]
    fn oom_is_retriable() {
        let e = OpenClError::OutOfMemory("big tensor".into());
        assert!(e.is_retriable_with_smaller_buffers());
    }

    #[test]
    fn non_oom_is_not_retriable() {
        let e = OpenClError::InvalidKernel("bad".into());
        assert!(!e.is_retriable_with_smaller_buffers());
    }
}
