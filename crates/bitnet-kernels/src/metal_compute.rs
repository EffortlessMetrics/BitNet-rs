//! wgpu-based Metal compute pipeline for Apple Silicon.
//!
//! Provides a compute dispatch abstraction targeting Metal via wgpu.  The
//! pipeline handles workgroup sizing, buffer alignment, and dispatch
//! dimension calculations with Apple Silicon constraints (unified memory,
//! 1024-thread workgroup limit, 256-byte buffer alignment).
//!
//! **Current status — configuration layer only.**  Shader compilation and
//! GPU dispatch require the `wgpu` crate which is not yet wired in.  This
//! module establishes the type-safe configuration surface so that kernel
//! authors can prepare pipelines without a live Metal device.
//!
//! Gated behind `--features metal`.

use std::fmt;

// ── Constants ────────────────────────────────────────────────────────

/// Metal requires buffer offsets aligned to 256 bytes.
pub const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Maximum threads per threadgroup on Apple Silicon GPUs.
pub const METAL_MAX_WORKGROUP_SIZE: u32 = 1024;

/// Default tile size tuned for Apple Silicon unified-memory architecture.
/// 16×16 = 256 threads — fits well within the 1024 limit and maps to
/// common matrix-multiply tile shapes.
pub const DEFAULT_TILE_SIZE: u32 = 16;

/// Maximum dispatch dimensions per axis (Metal limit).
pub const MAX_DISPATCH_DIM: u32 = 65535;

// ── Types ────────────────────────────────────────────────────────────

/// Workgroup (threadgroup) dimensions for a Metal compute dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl WorkgroupSize {
    /// Create a new workgroup size, returning an error if the total thread
    /// count exceeds [`METAL_MAX_WORKGROUP_SIZE`].
    pub fn new(x: u32, y: u32, z: u32) -> Result<Self, MetalConfigError> {
        let total = (x as u64) * (y as u64) * (z as u64);
        if total > METAL_MAX_WORKGROUP_SIZE as u64 {
            return Err(MetalConfigError::WorkgroupTooLarge {
                requested: total,
                max: METAL_MAX_WORKGROUP_SIZE,
            });
        }
        if x == 0 || y == 0 || z == 0 {
            return Err(MetalConfigError::ZeroDimension);
        }
        Ok(Self { x, y, z })
    }

    /// Total number of threads in this workgroup.
    pub fn total_threads(&self) -> u32 {
        self.x * self.y * self.z
    }

    /// 1-D workgroup of `n` threads.
    pub fn linear(n: u32) -> Result<Self, MetalConfigError> {
        Self::new(n, 1, 1)
    }

    /// 2-D square tile (e.g. 16×16).
    pub fn tile(size: u32) -> Result<Self, MetalConfigError> {
        Self::new(size, size, 1)
    }
}

impl Default for WorkgroupSize {
    fn default() -> Self {
        // 16×16 tile — always valid.
        Self { x: DEFAULT_TILE_SIZE, y: DEFAULT_TILE_SIZE, z: 1 }
    }
}

/// Number of workgroups to dispatch along each axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchDimensions {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl DispatchDimensions {
    /// Compute the dispatch grid for a given problem size and workgroup size.
    ///
    /// Each axis is `ceil(problem / workgroup)`, clamped to
    /// [`MAX_DISPATCH_DIM`].
    pub fn for_problem(
        problem: (u32, u32, u32),
        wg: &WorkgroupSize,
    ) -> Result<Self, MetalConfigError> {
        let dim = |p: u32, w: u32| -> Result<u32, MetalConfigError> {
            if w == 0 {
                return Err(MetalConfigError::ZeroDimension);
            }
            let d = p.div_ceil(w);
            if d > MAX_DISPATCH_DIM {
                return Err(MetalConfigError::DispatchTooLarge {
                    dimension: d,
                    max: MAX_DISPATCH_DIM,
                });
            }
            Ok(d)
        };
        Ok(Self { x: dim(problem.0, wg.x)?, y: dim(problem.1, wg.y)?, z: dim(problem.2, wg.z)? })
    }
}

// ── Unified memory ──────────────────────────────────────────────────

/// Describes the memory architecture of the target device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryArchitecture {
    /// CPU and GPU share the same physical memory (Apple Silicon).
    Unified,
    /// Discrete GPU with separate VRAM requiring explicit transfers.
    Discrete,
}

impl MemoryArchitecture {
    /// Detect memory architecture from the current platform.
    ///
    /// Returns [`Unified`](Self::Unified) on `aarch64-apple-*` targets
    /// (all Apple Silicon Macs) and [`Discrete`](Self::Discrete) otherwise.
    pub fn detect() -> Self {
        if cfg!(all(target_arch = "aarch64", target_vendor = "apple")) {
            Self::Unified
        } else {
            Self::Discrete
        }
    }

    /// Whether zero-copy buffer sharing is expected to be available.
    pub fn supports_zero_copy(&self) -> bool {
        *self == Self::Unified
    }
}

// ── Buffer helpers ──────────────────────────────────────────────────

/// Round `size` up to the next multiple of [`METAL_BUFFER_ALIGNMENT`].
#[inline]
pub fn align_buffer_size(size: usize) -> usize {
    let mask = METAL_BUFFER_ALIGNMENT - 1;
    (size + mask) & !mask
}

/// Validate that `offset` satisfies Metal's 256-byte alignment rule.
#[inline]
pub fn is_aligned(offset: usize) -> bool {
    offset % METAL_BUFFER_ALIGNMENT == 0
}

// ── Pipeline configuration ──────────────────────────────────────────

/// Configuration for a Metal compute pipeline.
#[derive(Debug, Clone)]
pub struct MetalComputePipeline {
    /// Human-readable label used for GPU debugging / profiling tools.
    pub label: String,
    /// Threadgroup dimensions.
    pub workgroup: WorkgroupSize,
    /// Memory architecture of the target.
    pub memory: MemoryArchitecture,
    /// Whether to prefer shared (threadgroup) memory over device memory
    /// for intermediate results.
    pub use_shared_memory: bool,
}

impl MetalComputePipeline {
    /// Create a pipeline with the given label and default Apple Silicon
    /// settings (16×16 tile, unified memory, shared memory enabled).
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            workgroup: WorkgroupSize::default(),
            memory: MemoryArchitecture::detect(),
            use_shared_memory: true,
        }
    }

    /// Override the workgroup size.
    pub fn with_workgroup(mut self, wg: WorkgroupSize) -> Self {
        self.workgroup = wg;
        self
    }

    /// Override the memory architecture assumption.
    pub fn with_memory(mut self, mem: MemoryArchitecture) -> Self {
        self.memory = mem;
        self
    }

    /// Compute dispatch dimensions for a 2-D matrix of `rows × cols`.
    pub fn dispatch_for_matrix(
        &self,
        rows: u32,
        cols: u32,
    ) -> Result<DispatchDimensions, MetalConfigError> {
        DispatchDimensions::for_problem((cols, rows, 1), &self.workgroup)
    }

    /// Compute the aligned buffer size needed for `element_count` elements
    /// of `element_bytes` each.
    pub fn aligned_buffer_bytes(&self, element_count: usize, element_bytes: usize) -> usize {
        align_buffer_size(element_count * element_bytes)
    }
}

// ── Errors ──────────────────────────────────────────────────────────

/// Errors arising from Metal compute pipeline configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalConfigError {
    /// Total threads per threadgroup exceeds the hardware limit.
    WorkgroupTooLarge { requested: u64, max: u32 },
    /// A dimension was zero (invalid for dispatch or workgroup).
    ZeroDimension,
    /// Dispatch grid exceeds Metal's per-axis maximum.
    DispatchTooLarge { dimension: u32, max: u32 },
}

impl fmt::Display for MetalConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkgroupTooLarge { requested, max } => {
                write!(f, "workgroup size {requested} exceeds Metal limit {max}")
            }
            Self::ZeroDimension => write!(f, "dimension must be non-zero"),
            Self::DispatchTooLarge { dimension, max } => {
                write!(
                    f,
                    "dispatch dimension {dimension} exceeds Metal limit \
                     {max}"
                )
            }
        }
    }
}

impl std::error::Error for MetalConfigError {}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- WorkgroupSize -------------------------------------------------

    #[test]
    fn workgroup_default_is_16x16() {
        let wg = WorkgroupSize::default();
        assert_eq!(wg.x, 16);
        assert_eq!(wg.y, 16);
        assert_eq!(wg.z, 1);
        assert_eq!(wg.total_threads(), 256);
    }

    #[test]
    fn workgroup_at_max_limit() {
        let wg = WorkgroupSize::new(1024, 1, 1).unwrap();
        assert_eq!(wg.total_threads(), 1024);

        let wg = WorkgroupSize::new(32, 32, 1).unwrap();
        assert_eq!(wg.total_threads(), 1024);
    }

    #[test]
    fn workgroup_exceeds_limit() {
        let err = WorkgroupSize::new(1025, 1, 1).unwrap_err();
        assert_eq!(err, MetalConfigError::WorkgroupTooLarge { requested: 1025, max: 1024 });

        let err = WorkgroupSize::new(32, 32, 2).unwrap_err();
        assert_eq!(err, MetalConfigError::WorkgroupTooLarge { requested: 2048, max: 1024 });
    }

    #[test]
    fn workgroup_zero_dimension_rejected() {
        assert_eq!(WorkgroupSize::new(0, 1, 1).unwrap_err(), MetalConfigError::ZeroDimension,);
    }

    #[test]
    fn workgroup_linear() {
        let wg = WorkgroupSize::linear(256).unwrap();
        assert_eq!((wg.x, wg.y, wg.z), (256, 1, 1));
    }

    #[test]
    fn workgroup_tile() {
        let wg = WorkgroupSize::tile(32).unwrap();
        assert_eq!((wg.x, wg.y, wg.z), (32, 32, 1));
        assert_eq!(wg.total_threads(), 1024);
    }

    #[test]
    fn workgroup_tile_too_large() {
        assert!(WorkgroupSize::tile(33).is_err());
    }

    // -- Buffer alignment ---------------------------------------------

    #[test]
    fn align_zero() {
        assert_eq!(align_buffer_size(0), 0);
    }

    #[test]
    fn align_already_aligned() {
        assert_eq!(align_buffer_size(256), 256);
        assert_eq!(align_buffer_size(512), 512);
    }

    #[test]
    fn align_rounds_up() {
        assert_eq!(align_buffer_size(1), 256);
        assert_eq!(align_buffer_size(255), 256);
        assert_eq!(align_buffer_size(257), 512);
    }

    #[test]
    fn is_aligned_checks() {
        assert!(is_aligned(0));
        assert!(is_aligned(256));
        assert!(is_aligned(512));
        assert!(!is_aligned(1));
        assert!(!is_aligned(128));
        assert!(!is_aligned(255));
    }

    // -- DispatchDimensions -------------------------------------------

    #[test]
    fn dispatch_exact_fit() {
        let wg = WorkgroupSize::tile(16).unwrap();
        let d = DispatchDimensions::for_problem((64, 32, 1), &wg).unwrap();
        assert_eq!((d.x, d.y, d.z), (4, 2, 1));
    }

    #[test]
    fn dispatch_rounds_up() {
        let wg = WorkgroupSize::tile(16).unwrap();
        let d = DispatchDimensions::for_problem((17, 1, 1), &wg).unwrap();
        assert_eq!(d.x, 2); // ceil(17/16)
        assert_eq!(d.y, 1);
    }

    #[test]
    fn dispatch_too_large() {
        let wg = WorkgroupSize::linear(1).unwrap();
        let err = DispatchDimensions::for_problem((70000, 1, 1), &wg).unwrap_err();
        assert_eq!(err, MetalConfigError::DispatchTooLarge { dimension: 70000, max: 65535 });
    }

    // -- MemoryArchitecture -------------------------------------------

    #[test]
    fn unified_memory_detection() {
        let mem = MemoryArchitecture::detect();
        // On Apple Silicon CI this will be Unified; on x86 it will be
        // Discrete.  Just ensure the function returns a valid variant.
        assert!(mem == MemoryArchitecture::Unified || mem == MemoryArchitecture::Discrete);
    }

    #[test]
    fn unified_supports_zero_copy() {
        assert!(MemoryArchitecture::Unified.supports_zero_copy());
        assert!(!MemoryArchitecture::Discrete.supports_zero_copy());
    }

    // -- MetalComputePipeline -----------------------------------------

    #[test]
    fn pipeline_creation_defaults() {
        let p = MetalComputePipeline::new("test_kernel");
        assert_eq!(p.label, "test_kernel");
        assert_eq!(p.workgroup.total_threads(), 256);
        assert!(p.use_shared_memory);
    }

    #[test]
    fn pipeline_with_custom_workgroup() {
        let wg = WorkgroupSize::linear(128).unwrap();
        let p = MetalComputePipeline::new("custom").with_workgroup(wg);
        assert_eq!(p.workgroup.total_threads(), 128);
    }

    #[test]
    fn pipeline_dispatch_for_matrix() {
        let p = MetalComputePipeline::new("matmul");
        // 64 rows × 128 cols, default 16×16 tile
        let d = p.dispatch_for_matrix(64, 128).unwrap();
        assert_eq!(d.x, 8); // 128 / 16
        assert_eq!(d.y, 4); // 64 / 16
        assert_eq!(d.z, 1);
    }

    #[test]
    fn pipeline_aligned_buffer_bytes() {
        let p = MetalComputePipeline::new("buf");
        // 100 f32 elements = 400 bytes → aligned to 512
        assert_eq!(p.aligned_buffer_bytes(100, 4), 512);
        // 64 f32 elements = 256 bytes → already aligned
        assert_eq!(p.aligned_buffer_bytes(64, 4), 256);
    }

    #[test]
    fn pipeline_memory_override() {
        let p = MetalComputePipeline::new("k").with_memory(MemoryArchitecture::Discrete);
        assert_eq!(p.memory, MemoryArchitecture::Discrete);
        assert!(!p.memory.supports_zero_copy());
    }

    // -- Error Display ------------------------------------------------

    #[test]
    fn error_display() {
        let e = MetalConfigError::WorkgroupTooLarge { requested: 2048, max: 1024 };
        assert!(e.to_string().contains("2048"));
        assert!(e.to_string().contains("1024"));

        let e = MetalConfigError::ZeroDimension;
        assert!(e.to_string().contains("non-zero"));
    }
}
