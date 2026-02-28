//! OpenCL backend for BitNet GPU inference.
//!
//! Provides optimized OpenCL primitives for Intel Arc GPU inference,
//! including context pooling, local memory optimizations, prefetch
//! pipelines, KV cache, paged attention, and multi-backend GPU dispatch
//! with automatic selection.
//! pipelines, KV cache, paged attention, and CPU reference implementations
//! with OpenCL kernel sources for I2_S dequantization, QK256 block
//! dequantization, and ternary matrix multiply.
//! pipelines, KV cache, paged attention, SPIR-V compilation, and kernel registry.

pub mod backend_dispatcher;
pub mod backend_registry;
pub mod context_pool;
pub mod kv_cache;
pub mod paged_attention;

pub use backend_dispatcher::{
    BackendCapabilityMatrix, BackendDispatcher, BackendStatus, DispatchDecision, DispatchError,
    DispatchLog, DispatchStrategy, Operation,
};
pub use backend_registry::{BackendInfo, BackendProvider, BackendRegistry};
pub mod quantized_kernels;
pub mod quantized_ops;
pub mod spirv;
pub mod spirv_kernels;

// Re-exports for convenience.
pub use spirv::{
    CompileOptions, CompilerBackend, OptimizationLevel, SPIRV_MAGIC, SpirVCache, SpirVCompiler,
    SpirVError, SpirVModule, SpirVValidator,
};
pub use spirv_kernels::{KernelSource, SpirvKernelRegistry};
//! OpenCL kernel infrastructure for BitNet GPU inference.
//!
//! This crate provides kernel source management, workgroup configuration,
//! buffer size calculations, and kernel variant selection for OpenCL-based
//! GPU acceleration of BitNet inference operations.
//!
//! **Current status — stubs only.** All kernel launch entry-points return
//! errors until an OpenCL runtime is integrated. The crate is gated behind
//! `--features oneapi` and requires `BITNET_ENABLE_OPENCL=1` at runtime.
//!
//! # Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`kernels`] | OpenCL C kernel source strings |
//! | [`selector`] | Problem-size-aware kernel variant selection |
//! | [`workgroup`] | Workgroup / NDRange size computation |
//! | [`buffers`] | Buffer size calculations for tensor dimensions |

pub mod buffers;
pub mod kernels;
pub mod selector;
pub mod workgroup;

use bitnet_common::{BitNetError, KernelError, QuantizationType, Result};

// ── Device information ───────────────────────────────────────────────

/// OpenCL device capabilities snapshot.
#[derive(Debug, Clone)]
pub struct OpenClDeviceInfo {
    /// Device ordinal index.
    pub device_id: usize,
    /// Device marketing name (e.g. "Intel Arc A770").
    pub name: String,
    /// Maximum work-group size supported by the device.
    pub max_workgroup_size: usize,
    /// Maximum number of compute units.
    pub max_compute_units: u32,
    /// Global memory size in bytes.
    pub global_mem_bytes: u64,
    /// Local (shared) memory size in bytes per work-group.
    pub local_mem_bytes: u64,
    /// Whether the device supports FP16 (`cl_khr_fp16`).
    pub supports_fp16: bool,
    /// Preferred work-group size multiple (wavefront / warp size).
    pub preferred_workgroup_multiple: usize,
}

impl Default for OpenClDeviceInfo {
    fn default() -> Self {
        Self {
            device_id: 0,
            name: "Generic OpenCL Device".into(),
            max_workgroup_size: 256,
            max_compute_units: 24,
            global_mem_bytes: 8 * 1024 * 1024 * 1024, // 8 GiB
            local_mem_bytes: 64 * 1024,               // 64 KiB
            supports_fp16: true,
            preferred_workgroup_multiple: 32,
        }
    }
}

impl OpenClDeviceInfo {
    /// Create device info with a specific max workgroup size.
    #[must_use]
    pub fn with_max_workgroup_size(mut self, size: usize) -> Self {
        self.max_workgroup_size = size;
        self
    }

    /// Create device info with a specific local memory limit.
    #[must_use]
    pub fn with_local_mem_bytes(mut self, bytes: u64) -> Self {
        self.local_mem_bytes = bytes;
        self
    }
}

// ── Kernel provider ──────────────────────────────────────────────────

/// OpenCL kernel provider (stub).
///
/// Mirrors [`bitnet_kernels::rocm::RocmKernel`] but targets Intel/AMD
/// GPUs via the OpenCL runtime.
#[derive(Debug, Clone, Default)]
pub struct OpenClKernel {
    _private: (),
}

impl OpenClKernel {
    /// Create a new OpenCL kernel provider (stub).
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Whether `oneapi` / OpenCL support was compiled into this build.
    pub fn compiled() -> bool {
        cfg!(feature = "oneapi")
    }

    /// Runtime opt-in via `BITNET_ENABLE_OPENCL=1`.
    pub fn opencl_enabled() -> bool {
        std::env::var("BITNET_ENABLE_OPENCL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    /// Check if this provider is available (compiled + enabled).
    pub fn is_available(&self) -> bool {
        Self::compiled() && Self::opencl_enabled()
    }

    /// Provider name for display / logging.
    pub fn name(&self) -> &'static str {
        "opencl"
    }

    /// Matrix multiply (stub — returns error).
    pub fn matmul_i2s(
        &self,
        _a: &[i8],
        _b: &[u8],
        _c: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<()> {
        Err(self.unavailable_err("matmul_i2s"))
    }

    /// Quantize (stub — returns error).
    pub fn quantize(
        &self,
        _input: &[f32],
        _output: &mut [u8],
        _scales: &mut [f32],
        _qtype: QuantizationType,
    ) -> Result<()> {
        Err(self.unavailable_err("quantize"))
    }

    fn unavailable_err(&self, op: &str) -> BitNetError {
        BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("OpenCL operation '{op}' is not yet wired to an OpenCL runtime"),
        })
    }
}

/// Check whether an OpenCL runtime is available on the system.
///
/// Stub — always returns `false` until device enumeration is wired in.
pub fn is_opencl_available() -> bool {
    false
}

/// Return the number of OpenCL-visible GPU devices.
///
/// Stub — always returns `0`.
pub fn opencl_device_count() -> usize {
    0
}
