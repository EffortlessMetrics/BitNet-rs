//! wgpu Metal compute kernel runner scaffold for Apple Silicon.
//!
//! Provides `MetalKernelRunner` — a mock interface for dispatching compute
//! kernels on Apple Silicon via wgpu's Metal backend.  This is a scaffold
//! (no actual wgpu dependency) that validates Metal-specific constraints
//! and defines the API surface for future integration.
//!
//! # Metal constraints enforced
//!
//! | Limit                          | Value |
//! |--------------------------------|-------|
//! | Buffer alignment               |   256 |
//! | Max threads per threadgroup     |  1024 |
//! | SIMD group (wave) size          |    32 |

use std::fmt;
use std::time::{Duration, Instant};

// ── Constants ───────────────────────────────────────────────────────

/// Metal requires 256-byte buffer alignment.
pub const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Maximum threads per threadgroup on Apple Silicon.
pub const METAL_MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// SIMD group (wave) size on Apple Silicon GPUs.
pub const METAL_SIMD_GROUP_SIZE: u32 = 32;

/// Maximum single buffer allocation (256 MiB scaffold limit).
pub const METAL_MAX_BUFFER_SIZE: usize = 256 * 1024 * 1024;

// ── Errors ──────────────────────────────────────────────────────────

/// Errors from the Metal kernel runner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalRunnerError {
    /// Failed to acquire a wgpu `Device`.
    DeviceCreationFailed(String),
    /// Compute pipeline compilation error.
    PipelineCreationFailed(String),
    /// Buffer allocation or mapping error.
    BufferError(String),
    /// Dispatch validation or execution error.
    DispatchError(String),
    /// Workgroup configuration violates Metal limits.
    InvalidWorkgroupConfig(String),
}

impl fmt::Display for MetalRunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceCreationFailed(msg) => {
                write!(f, "Metal device creation failed: {msg}")
            }
            Self::PipelineCreationFailed(msg) => {
                write!(f, "Metal pipeline creation failed: {msg}")
            }
            Self::BufferError(msg) => {
                write!(f, "Metal buffer error: {msg}")
            }
            Self::DispatchError(msg) => {
                write!(f, "Metal dispatch error: {msg}")
            }
            Self::InvalidWorkgroupConfig(msg) => {
                write!(f, "Invalid workgroup config: {msg}")
            }
        }
    }
}

impl std::error::Error for MetalRunnerError {}

// ── Configuration ───────────────────────────────────────────────────

/// Configuration for a Metal compute dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalKernelConfig {
    /// Workgroup size (x, y, z). Product must be ≤ 1024.
    pub workgroup_size: [u32; 3],
    /// Required buffer alignment in bytes (must be ≥ 256 for Metal).
    pub buffer_alignment: usize,
    /// Grid dispatch dimensions (x, y, z) in workgroups.
    pub dispatch_dimensions: [u32; 3],
}

impl Default for MetalKernelConfig {
    fn default() -> Self {
        Self {
            workgroup_size: [32, 1, 1],
            buffer_alignment: METAL_BUFFER_ALIGNMENT,
            dispatch_dimensions: [1, 1, 1],
        }
    }
}

impl MetalKernelConfig {
    /// Validate the configuration against Metal hardware limits.
    pub fn validate(&self) -> Result<(), MetalRunnerError> {
        // Workgroup dimensions must be non-zero.
        if self.workgroup_size.contains(&0) {
            return Err(MetalRunnerError::InvalidWorkgroupConfig(
                "workgroup dimensions must be non-zero".into(),
            ));
        }

        // Total threads ≤ 1024.
        let total_threads: u64 = self.workgroup_size.iter().map(|&d| u64::from(d)).product();
        if total_threads > u64::from(METAL_MAX_THREADS_PER_THREADGROUP) {
            return Err(MetalRunnerError::InvalidWorkgroupConfig(format!(
                "total threads per threadgroup ({total_threads}) exceeds Metal limit ({METAL_MAX_THREADS_PER_THREADGROUP})"
            )));
        }

        // Buffer alignment must meet Metal minimum.
        if self.buffer_alignment < METAL_BUFFER_ALIGNMENT {
            return Err(MetalRunnerError::InvalidWorkgroupConfig(format!(
                "buffer alignment ({}) is below Metal minimum ({METAL_BUFFER_ALIGNMENT})",
                self.buffer_alignment
            )));
        }

        // Alignment must be a power of two.
        if !self.buffer_alignment.is_power_of_two() {
            return Err(MetalRunnerError::InvalidWorkgroupConfig(format!(
                "buffer alignment ({}) must be a power of two",
                self.buffer_alignment
            )));
        }

        // Dispatch dimensions must be non-zero.
        if self.dispatch_dimensions.contains(&0) {
            return Err(MetalRunnerError::InvalidWorkgroupConfig(
                "dispatch dimensions must be non-zero".into(),
            ));
        }

        Ok(())
    }

    /// Total number of workgroups in the dispatch grid.
    pub fn total_workgroups(&self) -> u64 {
        self.dispatch_dimensions.iter().map(|&d| u64::from(d)).product()
    }

    /// Total number of threads across the entire dispatch.
    pub fn total_threads(&self) -> u64 {
        let per_group: u64 = self.workgroup_size.iter().map(|&d| u64::from(d)).product();
        per_group * self.total_workgroups()
    }
}

// ── Dispatch result ─────────────────────────────────────────────────

/// Result of a compute dispatch.
#[derive(Debug, Clone)]
pub struct MetalDispatchResult {
    /// Wall-clock time for the dispatch (mock: always near zero).
    pub elapsed: Duration,
    /// Size of the output buffer in bytes.
    pub output_buffer_size: usize,
    /// Number of workgroups dispatched.
    pub workgroup_count: u64,
}

// ── Buffer handle ───────────────────────────────────────────────────

/// Opaque handle to a GPU buffer (mock: tracks metadata only).
#[derive(Debug, Clone)]
pub struct MetalBuffer {
    /// Unique id for this buffer.
    pub id: u64,
    /// Size in bytes (aligned).
    pub size: usize,
    /// True if the buffer is mappable for read-back.
    pub mappable: bool,
}

// ── Runner ──────────────────────────────────────────────────────────

/// Mock wgpu Metal compute kernel runner.
///
/// Validates Metal-specific constraints and simulates the lifecycle of
/// device creation → buffer allocation → dispatch → read-back.
#[derive(Debug)]
pub struct MetalKernelRunner {
    /// Whether the mock device is "connected".
    initialized: bool,
    /// Monotonic buffer id counter.
    next_buffer_id: u64,
    /// Buffers currently allocated.
    buffers: Vec<MetalBuffer>,
}

impl MetalKernelRunner {
    /// Create a new runner, simulating wgpu device/queue creation.
    ///
    /// Returns `Err` if `simulate_failure` is `true` (for testing error
    /// paths).
    pub fn new(simulate_failure: bool) -> Result<Self, MetalRunnerError> {
        if simulate_failure {
            return Err(MetalRunnerError::DeviceCreationFailed(
                "simulated: no Metal-capable GPU found".into(),
            ));
        }
        Ok(Self { initialized: true, next_buffer_id: 1, buffers: Vec::new() })
    }

    /// Allocate a buffer of `size` bytes on the (mock) device.
    ///
    /// `size` is rounded up to the next multiple of
    /// [`METAL_BUFFER_ALIGNMENT`]. Zero-size buffers are rejected.
    pub fn create_buffer(
        &mut self,
        size: usize,
        mappable: bool,
    ) -> Result<MetalBuffer, MetalRunnerError> {
        if !self.initialized {
            return Err(MetalRunnerError::DeviceCreationFailed("runner not initialized".into()));
        }
        if size == 0 {
            return Err(MetalRunnerError::BufferError(
                "zero-size buffer allocation is not allowed".into(),
            ));
        }
        if size > METAL_MAX_BUFFER_SIZE {
            return Err(MetalRunnerError::BufferError(format!(
                "requested size ({size}) exceeds maximum ({METAL_MAX_BUFFER_SIZE})"
            )));
        }

        let aligned_size = align_up(size, METAL_BUFFER_ALIGNMENT);
        let id = self.next_buffer_id;
        self.next_buffer_id += 1;

        let buf = MetalBuffer { id, size: aligned_size, mappable };
        self.buffers.push(buf.clone());
        Ok(buf)
    }

    /// Submit a compute dispatch with the given configuration.
    pub fn dispatch_compute(
        &self,
        config: &MetalKernelConfig,
        output_buffer: &MetalBuffer,
    ) -> Result<MetalDispatchResult, MetalRunnerError> {
        if !self.initialized {
            return Err(MetalRunnerError::DispatchError("runner not initialized".into()));
        }
        config.validate()?;

        // Verify the output buffer belongs to this runner.
        if !self.buffers.iter().any(|b| b.id == output_buffer.id) {
            return Err(MetalRunnerError::DispatchError(
                "output buffer not owned by this runner".into(),
            ));
        }

        let start = Instant::now();
        // Mock: no actual GPU work.
        let elapsed = start.elapsed();

        Ok(MetalDispatchResult {
            elapsed,
            output_buffer_size: output_buffer.size,
            workgroup_count: config.total_workgroups(),
        })
    }

    /// Read back the contents of a mappable buffer.
    ///
    /// Returns a zero-filled `Vec<u8>` of the buffer's aligned size
    /// (mock implementation).
    pub fn read_buffer(&self, buffer: &MetalBuffer) -> Result<Vec<u8>, MetalRunnerError> {
        if !self.initialized {
            return Err(MetalRunnerError::BufferError("runner not initialized".into()));
        }
        if !buffer.mappable {
            return Err(MetalRunnerError::BufferError(
                "buffer is not mappable for read-back".into(),
            ));
        }
        if !self.buffers.iter().any(|b| b.id == buffer.id) {
            return Err(MetalRunnerError::BufferError("buffer not owned by this runner".into()));
        }
        Ok(vec![0u8; buffer.size])
    }

    /// Wait for all in-flight GPU work to complete (mock: no-op).
    pub fn synchronize(&self) -> Result<(), MetalRunnerError> {
        if !self.initialized {
            return Err(MetalRunnerError::DeviceCreationFailed("runner not initialized".into()));
        }
        Ok(())
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Round `value` up to the next multiple of `alignment`.
///
/// `alignment` **must** be a power of two.
#[inline]
fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

/// Check whether a workgroup size is a multiple of the SIMD group size.
pub fn is_simd_aligned(workgroup_x: u32) -> bool {
    workgroup_x > 0 && workgroup_x.is_multiple_of(METAL_SIMD_GROUP_SIZE)
}

/// Compute the aligned buffer size for a given byte count.
pub fn aligned_buffer_size(bytes: usize) -> usize {
    if bytes == 0 {
        return 0;
    }
    align_up(bytes, METAL_BUFFER_ALIGNMENT)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config validation ───────────────────────────────────────────

    #[test]
    fn test_default_config_is_valid() {
        let cfg = MetalKernelConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_workgroup_exceeds_max_threads() {
        let cfg = MetalKernelConfig {
            workgroup_size: [32, 33, 1], // 1056 > 1024
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, MetalRunnerError::InvalidWorkgroupConfig(_)));
    }

    #[test]
    fn test_config_workgroup_exactly_max_threads() {
        let cfg = MetalKernelConfig {
            workgroup_size: [32, 32, 1], // 1024
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_workgroup_zero_dimension() {
        let cfg = MetalKernelConfig { workgroup_size: [0, 1, 1], ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_dispatch_zero_dimension() {
        let cfg = MetalKernelConfig { dispatch_dimensions: [0, 1, 1], ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_alignment_below_minimum() {
        let cfg = MetalKernelConfig { buffer_alignment: 128, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_alignment_not_power_of_two() {
        let cfg = MetalKernelConfig { buffer_alignment: 300, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_large_valid_alignment() {
        let cfg = MetalKernelConfig { buffer_alignment: 512, ..Default::default() };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_total_threads() {
        let cfg = MetalKernelConfig {
            workgroup_size: [8, 4, 2],
            dispatch_dimensions: [2, 3, 1],
            ..Default::default()
        };
        // per_group = 64, workgroups = 6, total = 384
        assert_eq!(cfg.total_threads(), 384);
    }

    #[test]
    fn test_config_total_workgroups() {
        let cfg = MetalKernelConfig { dispatch_dimensions: [4, 5, 6], ..Default::default() };
        assert_eq!(cfg.total_workgroups(), 120);
    }

    // ── Runner lifecycle ────────────────────────────────────────────

    #[test]
    fn test_runner_creation_success() {
        let runner = MetalKernelRunner::new(false);
        assert!(runner.is_ok());
    }

    #[test]
    fn test_runner_creation_failure() {
        let err = MetalKernelRunner::new(true).unwrap_err();
        assert!(matches!(err, MetalRunnerError::DeviceCreationFailed(_)));
    }

    // ── Buffer creation ─────────────────────────────────────────────

    #[test]
    fn test_buffer_zero_size_rejected() {
        let mut runner = MetalKernelRunner::new(false).unwrap();
        let err = runner.create_buffer(0, false).unwrap_err();
        assert!(matches!(err, MetalRunnerError::BufferError(_)));
    }

    #[test]
    fn test_buffer_exceeds_max_size() {
        let mut runner = MetalKernelRunner::new(false).unwrap();
        let err = runner.create_buffer(METAL_MAX_BUFFER_SIZE + 1, false).unwrap_err();
        assert!(matches!(err, MetalRunnerError::BufferError(_)));
    }

    #[test]
    fn test_buffer_at_max_size() {
        let mut runner = MetalKernelRunner::new(false).unwrap();
        let buf = runner.create_buffer(METAL_MAX_BUFFER_SIZE, false).unwrap();
        assert_eq!(buf.size, METAL_MAX_BUFFER_SIZE);
    }

    #[test]
    fn test_buffer_alignment_rounding() {
        let mut runner = MetalKernelRunner::new(false).unwrap();
        let buf = runner.create_buffer(1, false).unwrap();
        assert_eq!(buf.size, METAL_BUFFER_ALIGNMENT);
    }

    #[test]
    fn test_buffer_exact_alignment_no_padding() {
        let mut runner = MetalKernelRunner::new(false).unwrap();
        let buf = runner.create_buffer(METAL_BUFFER_ALIGNMENT, false).unwrap();
        assert_eq!(buf.size, METAL_BUFFER_ALIGNMENT);
    }

    #[test]
    fn test_buffer_ids_are_unique() {
        let mut runner = MetalKernelRunner::new(false).unwrap();
        let b1 = runner.create_buffer(256, false).unwrap();
        let b2 = runner.create_buffer(256, false).unwrap();
        assert_ne!(b1.id, b2.id);
    }

    #[test]
    fn test_buffer_mappable_flag() {
        let mut runner = MetalKernelRunner::new(false).unwrap();
        let buf = runner.create_buffer(512, true).unwrap();
        assert!(buf.mappable);
    }

    // ── Dispatch ────────────────────────────────────────────────────

    #[test]
    fn test_dispatch_valid() {
        let mut runner = MetalKernelRunner::new(false).unwrap();
        let buf = runner.create_buffer(1024, true).unwrap();
        let cfg = MetalKernelConfig::default();
        let result = runner.dispatch_compute(&cfg, &buf).unwrap();
        assert_eq!(result.workgroup_count, 1);
        assert_eq!(result.output_buffer_size, buf.size);
    }

    #[test]
    fn test_dispatch_invalid_config_rejected() {
        let mut runner = MetalKernelRunner::new(false).unwrap();
        let buf = runner.create_buffer(256, true).unwrap();
        let cfg = MetalKernelConfig { workgroup_size: [0, 1, 1], ..Default::default() };
        assert!(runner.dispatch_compute(&cfg, &buf).is_err());
    }

    #[test]
    fn test_dispatch_unknown_buffer_rejected() {
        let runner = MetalKernelRunner::new(false).unwrap();
        let foreign = MetalBuffer { id: 999, size: 256, mappable: true };
        let cfg = MetalKernelConfig::default();
        let err = runner.dispatch_compute(&cfg, &foreign).unwrap_err();
        assert!(matches!(err, MetalRunnerError::DispatchError(_)));
    }

    // ── Read-back ───────────────────────────────────────────────────

    #[test]
    fn test_read_mappable_buffer() {
        let mut runner = MetalKernelRunner::new(false).unwrap();
        let buf = runner.create_buffer(300, true).unwrap();
        let data = runner.read_buffer(&buf).unwrap();
        assert_eq!(data.len(), buf.size);
    }

    #[test]
    fn test_read_non_mappable_buffer_rejected() {
        let mut runner = MetalKernelRunner::new(false).unwrap();
        let buf = runner.create_buffer(256, false).unwrap();
        let err = runner.read_buffer(&buf).unwrap_err();
        assert!(matches!(err, MetalRunnerError::BufferError(_)));
    }

    #[test]
    fn test_read_foreign_buffer_rejected() {
        let runner = MetalKernelRunner::new(false).unwrap();
        let foreign = MetalBuffer { id: 999, size: 256, mappable: true };
        assert!(runner.read_buffer(&foreign).is_err());
    }

    // ── Synchronize ─────────────────────────────────────────────────

    #[test]
    fn test_synchronize_ok() {
        let runner = MetalKernelRunner::new(false).unwrap();
        assert!(runner.synchronize().is_ok());
    }

    // ── SIMD helpers ────────────────────────────────────────────────

    #[test]
    fn test_simd_aligned_32() {
        assert!(is_simd_aligned(32));
    }

    #[test]
    fn test_simd_aligned_64() {
        assert!(is_simd_aligned(64));
    }

    #[test]
    fn test_simd_not_aligned() {
        assert!(!is_simd_aligned(48));
    }

    #[test]
    fn test_simd_zero_not_aligned() {
        assert!(!is_simd_aligned(0));
    }

    // ── Alignment helper ────────────────────────────────────────────

    #[test]
    fn test_aligned_buffer_size_zero() {
        assert_eq!(aligned_buffer_size(0), 0);
    }

    #[test]
    fn test_aligned_buffer_size_rounds_up() {
        assert_eq!(aligned_buffer_size(1), METAL_BUFFER_ALIGNMENT);
        assert_eq!(aligned_buffer_size(257), 2 * METAL_BUFFER_ALIGNMENT);
    }

    // ── Error display ───────────────────────────────────────────────

    #[test]
    fn test_error_display_device() {
        let e = MetalRunnerError::DeviceCreationFailed("no gpu".into());
        assert!(e.to_string().contains("no gpu"));
    }

    #[test]
    fn test_error_display_pipeline() {
        let e = MetalRunnerError::PipelineCreationFailed("bad shader".into());
        assert!(e.to_string().contains("bad shader"));
    }

    #[test]
    fn test_error_display_buffer() {
        let e = MetalRunnerError::BufferError("oom".into());
        assert!(e.to_string().contains("oom"));
    }

    #[test]
    fn test_error_display_dispatch() {
        let e = MetalRunnerError::DispatchError("dim".into());
        assert!(e.to_string().contains("dim"));
    }

    #[test]
    fn test_error_display_workgroup() {
        let e = MetalRunnerError::InvalidWorkgroupConfig("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    // ── 3D workgroup edge cases ─────────────────────────────────────

    #[test]
    fn test_config_3d_workgroup_at_limit() {
        // 8 × 8 × 16 = 1024
        let cfg = MetalKernelConfig { workgroup_size: [8, 8, 16], ..Default::default() };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_3d_workgroup_over_limit() {
        // 8 × 8 × 17 = 1088 > 1024
        let cfg = MetalKernelConfig { workgroup_size: [8, 8, 17], ..Default::default() };
        assert!(cfg.validate().is_err());
    }
}
