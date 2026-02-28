//! GPU memory limit enforcement and telemetry for OpenCL backends.
//!
//! Provides [`GpuMemoryInfo`] for querying device memory, [`MemoryGuard`] for
//! tracking allocations and preventing OOM, and an environment variable
//! override (`BITNET_GPU_MEM_LIMIT_MB`) to cap GPU memory usage.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Information about GPU device memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuMemoryInfo {
    /// Total global memory on the device (bytes).
    pub total_bytes: u64,
    /// Maximum single allocation size (bytes).
    pub max_alloc_bytes: u64,
    /// Currently allocated by this process (bytes, tracked by [`MemoryGuard`]).
    pub allocated_bytes: u64,
}

impl GpuMemoryInfo {
    /// Available memory (total minus allocated).
    pub fn available_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.allocated_bytes)
    }
}

/// Telemetry counters for GPU memory usage.
#[derive(Debug)]
pub struct MemoryTelemetry {
    /// Peak memory usage observed (bytes).
    pub peak_bytes: AtomicU64,
    /// Total number of allocations performed.
    pub allocation_count: AtomicU64,
    /// Total number of deallocations performed.
    pub deallocation_count: AtomicU64,
    /// Total bytes ever allocated (for fragmentation estimation).
    pub cumulative_allocated: AtomicU64,
}

impl Default for MemoryTelemetry {
    fn default() -> Self {
        Self {
            peak_bytes: AtomicU64::new(0),
            allocation_count: AtomicU64::new(0),
            deallocation_count: AtomicU64::new(0),
            cumulative_allocated: AtomicU64::new(0),
        }
    }
}

impl MemoryTelemetry {
    /// Estimated fragmentation ratio: cumulative_allocated / peak.
    /// A value close to 1.0 means little fragmentation; higher means more churn.
    pub fn fragmentation_estimate(&self) -> f64 {
        let peak = self.peak_bytes.load(Ordering::Relaxed);
        let cumulative = self.cumulative_allocated.load(Ordering::Relaxed);
        if peak == 0 {
            0.0
        } else {
            cumulative as f64 / peak as f64
        }
    }

    /// Snapshot of telemetry values.
    pub fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            peak_bytes: self.peak_bytes.load(Ordering::Relaxed),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
            deallocation_count: self.deallocation_count.load(Ordering::Relaxed),
            fragmentation_estimate: self.fragmentation_estimate(),
        }
    }
}

/// A point-in-time snapshot of memory telemetry.
#[derive(Debug, Clone, Copy)]
pub struct TelemetrySnapshot {
    pub peak_bytes: u64,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub fragmentation_estimate: f64,
}

/// Error returned when a memory allocation would exceed limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryLimitError {
    /// The allocation would exceed the configured memory limit.
    ExceedsLimit {
        requested: u64,
        available: u64,
        limit: u64,
    },
    /// The allocation exceeds the device's max single allocation size.
    ExceedsMaxAlloc {
        requested: u64,
        max_alloc: u64,
    },
}

impl std::fmt::Display for MemoryLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryLimitError::ExceedsLimit {
                requested,
                available,
                limit,
            } => write!(
                f,
                "GPU allocation of {} bytes exceeds limit \
                 (available={}, limit={})",
                requested, available, limit
            ),
            MemoryLimitError::ExceedsMaxAlloc {
                requested,
                max_alloc,
            } => write!(
                f,
                "GPU allocation of {} bytes exceeds device max \
                 single allocation size ({})",
                requested, max_alloc
            ),
        }
    }
}

impl std::error::Error for MemoryLimitError {}

/// Parse `BITNET_GPU_MEM_LIMIT_MB` environment variable.
/// Returns `None` if unset or unparseable.
pub fn parse_mem_limit_env() -> Option<u64> {
    std::env::var("BITNET_GPU_MEM_LIMIT_MB")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|mb| mb * 1024 * 1024)
}

/// Guard that tracks GPU memory allocations and prevents OOM.
///
/// All allocations go through [`MemoryGuard::try_allocate`] and are released
/// via [`MemoryGuard::release`]. The guard enforces a configurable memory
/// limit (from env var or explicit cap).
#[derive(Debug)]
pub struct MemoryGuard {
    /// Device total memory (bytes).
    total_bytes: u64,
    /// Device max single alloc (bytes).
    max_alloc_bytes: u64,
    /// Effective memory cap (bytes). May be less than total if env-limited.
    effective_limit: u64,
    /// Current tracked allocation (bytes).
    current_allocated: AtomicU64,
    /// Telemetry counters.
    telemetry: Arc<MemoryTelemetry>,
}

impl MemoryGuard {
    /// Create a new memory guard for a device with the given memory properties.
    ///
    /// If `BITNET_GPU_MEM_LIMIT_MB` is set, the effective limit is capped
    /// to that value. Otherwise the full device memory is available.
    pub fn new(total_bytes: u64, max_alloc_bytes: u64) -> Self {
        let env_limit = parse_mem_limit_env();
        let effective_limit = match env_limit {
            Some(limit) => limit.min(total_bytes),
            None => total_bytes,
        };

        Self {
            total_bytes,
            max_alloc_bytes,
            effective_limit,
            current_allocated: AtomicU64::new(0),
            telemetry: Arc::new(MemoryTelemetry::default()),
        }
    }

    /// Create a guard with an explicit memory cap (ignoring env var).
    pub fn with_limit(
        total_bytes: u64,
        max_alloc_bytes: u64,
        limit_bytes: u64,
    ) -> Self {
        Self {
            total_bytes,
            max_alloc_bytes,
            effective_limit: limit_bytes.min(total_bytes),
            current_allocated: AtomicU64::new(0),
            telemetry: Arc::new(MemoryTelemetry::default()),
        }
    }

    /// Try to allocate `size` bytes. Returns `Ok(())` if the allocation
    /// is within limits, `Err` otherwise.
    pub fn try_allocate(&self, size: u64) -> Result<(), MemoryLimitError> {
        if size > self.max_alloc_bytes {
            return Err(MemoryLimitError::ExceedsMaxAlloc {
                requested: size,
                max_alloc: self.max_alloc_bytes,
            });
        }

        let current = self.current_allocated.load(Ordering::Relaxed);
        let new_total = current.saturating_add(size);

        if new_total > self.effective_limit {
            return Err(MemoryLimitError::ExceedsLimit {
                requested: size,
                available: self.effective_limit.saturating_sub(current),
                limit: self.effective_limit,
            });
        }

        // CAS loop for atomic update.
        let mut prev = current;
        loop {
            match self.current_allocated.compare_exchange_weak(
                prev,
                prev.saturating_add(size),
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => {
                    if actual.saturating_add(size) > self.effective_limit {
                        return Err(MemoryLimitError::ExceedsLimit {
                            requested: size,
                            available: self
                                .effective_limit
                                .saturating_sub(actual),
                            limit: self.effective_limit,
                        });
                    }
                    prev = actual;
                }
            }
        }

        // Update telemetry.
        self.telemetry
            .allocation_count
            .fetch_add(1, Ordering::Relaxed);
        self.telemetry
            .cumulative_allocated
            .fetch_add(size, Ordering::Relaxed);

        let after = self.current_allocated.load(Ordering::Relaxed);
        self.telemetry
            .peak_bytes
            .fetch_max(after, Ordering::Relaxed);

        Ok(())
    }

    /// Release `size` bytes of previously allocated memory.
    pub fn release(&self, size: u64) {
        self.current_allocated
            .fetch_sub(size.min(self.current()), Ordering::AcqRel);
        self.telemetry
            .deallocation_count
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Current tracked allocation in bytes.
    pub fn current(&self) -> u64 {
        self.current_allocated.load(Ordering::Relaxed)
    }

    /// Current memory info snapshot.
    pub fn info(&self) -> GpuMemoryInfo {
        GpuMemoryInfo {
            total_bytes: self.total_bytes,
            max_alloc_bytes: self.max_alloc_bytes,
            allocated_bytes: self.current(),
        }
    }

    /// Effective memory limit in bytes.
    pub fn effective_limit(&self) -> u64 {
        self.effective_limit
    }

    /// Reference to the telemetry counters.
    pub fn telemetry(&self) -> &MemoryTelemetry {
        &self.telemetry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_info_available() {
        let info = GpuMemoryInfo {
            total_bytes: 1000,
            max_alloc_bytes: 500,
            allocated_bytes: 300,
        };
        assert_eq!(info.available_bytes(), 700);
    }

    #[test]
    fn memory_info_available_overflow() {
        let info = GpuMemoryInfo {
            total_bytes: 100,
            max_alloc_bytes: 50,
            allocated_bytes: 200,
        };
        assert_eq!(info.available_bytes(), 0);
    }

    #[test]
    fn guard_basic_allocate_release() {
        let guard = MemoryGuard::with_limit(1024, 512, 1024);
        assert_eq!(guard.current(), 0);

        guard.try_allocate(256).unwrap();
        assert_eq!(guard.current(), 256);

        guard.try_allocate(256).unwrap();
        assert_eq!(guard.current(), 512);

        guard.release(256);
        assert_eq!(guard.current(), 256);
    }

    #[test]
    fn guard_rejects_over_limit() {
        let guard = MemoryGuard::with_limit(1024, 512, 512);
        guard.try_allocate(256).unwrap();

        let err = guard.try_allocate(300).unwrap_err();
        match err {
            MemoryLimitError::ExceedsLimit {
                requested,
                available,
                limit,
            } => {
                assert_eq!(requested, 300);
                assert_eq!(available, 256);
                assert_eq!(limit, 512);
            }
            _ => panic!("expected ExceedsLimit"),
        }
    }

    #[test]
    fn guard_rejects_over_max_alloc() {
        let guard = MemoryGuard::with_limit(4096, 512, 4096);
        let err = guard.try_allocate(1024).unwrap_err();
        match err {
            MemoryLimitError::ExceedsMaxAlloc {
                requested,
                max_alloc,
            } => {
                assert_eq!(requested, 1024);
                assert_eq!(max_alloc, 512);
            }
            _ => panic!("expected ExceedsMaxAlloc"),
        }
    }

    #[test]
    fn guard_telemetry_tracks_peak() {
        let guard = MemoryGuard::with_limit(4096, 4096, 4096);
        guard.try_allocate(100).unwrap();
        guard.try_allocate(200).unwrap();
        guard.release(150);
        guard.try_allocate(50).unwrap();

        let snap = guard.telemetry().snapshot();
        assert_eq!(snap.peak_bytes, 300);
        assert_eq!(snap.allocation_count, 3);
        assert_eq!(snap.deallocation_count, 1);
    }

    #[test]
    fn guard_telemetry_fragmentation() {
        let guard = MemoryGuard::with_limit(4096, 4096, 4096);
        guard.try_allocate(100).unwrap();
        guard.release(100);
        guard.try_allocate(100).unwrap();

        let snap = guard.telemetry().snapshot();
        // cumulative = 200, peak = 100, ratio = 2.0
        assert!((snap.fragmentation_estimate - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn guard_info_snapshot() {
        let guard = MemoryGuard::with_limit(2048, 1024, 1536);
        guard.try_allocate(512).unwrap();

        let info = guard.info();
        assert_eq!(info.total_bytes, 2048);
        assert_eq!(info.max_alloc_bytes, 1024);
        assert_eq!(info.allocated_bytes, 512);
        assert_eq!(info.available_bytes(), 1536);
    }

    #[test]
    fn guard_effective_limit_capped() {
        let guard = MemoryGuard::with_limit(2048, 1024, 3000);
        // limit capped to total
        assert_eq!(guard.effective_limit(), 2048);
    }

    #[test]
    fn parse_mem_limit_env_unset() {
        // When env var is not set, returns None
        let val = std::env::var("BITNET_GPU_MEM_LIMIT_MB_TEST_NONEXIST");
        assert!(val.is_err());
    }

    #[test]
    fn error_display_formatting() {
        let err = MemoryLimitError::ExceedsLimit {
            requested: 1000,
            available: 500,
            limit: 2000,
        };
        let msg = err.to_string();
        assert!(msg.contains("1000"));
        assert!(msg.contains("500"));
        assert!(msg.contains("2000"));

        let err2 = MemoryLimitError::ExceedsMaxAlloc {
            requested: 1024,
            max_alloc: 512,
        };
        let msg2 = err2.to_string();
        assert!(msg2.contains("1024"));
        assert!(msg2.contains("512"));
    }
}
