//! OpenCL context and resource management for GPU compute.
//!
//! Provides a safe abstraction over OpenCL platform/device/context/queue
//! lifecycle management. When the `oneapi` feature is disabled, this module
//! provides CPU-only stub implementations.

use std::fmt;

/// OpenCL platform information.
#[derive(Debug, Clone)]
pub struct OpenClPlatformInfo {
    /// Platform name
    pub name: String,
    /// Platform vendor
    pub vendor: String,
    /// Platform version string
    pub version: String,
    /// Platform profile
    pub profile: String,
    /// Platform extensions
    pub extensions: Vec<String>,
}

/// OpenCL device information.
#[derive(Debug, Clone)]
pub struct OpenClDeviceInfo {
    /// Device name (e.g., "Intel(R) Arc(TM) A770 Graphics")
    pub name: String,
    /// Device vendor
    pub vendor: String,
    /// Device type (GPU, CPU, Accelerator)
    pub device_type: OpenClDeviceType,
    /// Maximum compute units (Xe-cores for Intel Arc)
    pub max_compute_units: u32,
    /// Maximum clock frequency in MHz
    pub max_clock_freq_mhz: u32,
    /// Global memory size in bytes
    pub global_mem_bytes: u64,
    /// Local memory size in bytes
    pub local_mem_bytes: u64,
    /// Maximum work group size
    pub max_work_group_size: usize,
    /// Maximum work item dimensions
    pub max_work_item_dimensions: u32,
    /// Maximum work item sizes per dimension
    pub max_work_item_sizes: Vec<usize>,
    /// OpenCL version
    pub opencl_version: String,
    /// Driver version
    pub driver_version: String,
    /// Whether device supports FP16
    pub supports_fp16: bool,
    /// Whether device supports FP64
    pub supports_fp64: bool,
    /// Preferred vector width for float
    pub preferred_vector_width_float: u32,
}

/// OpenCL device type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenClDeviceType {
    Gpu,
    Cpu,
    Accelerator,
    Custom,
    Unknown,
}

impl fmt::Display for OpenClDeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu => write!(f, "GPU"),
            Self::Cpu => write!(f, "CPU"),
            Self::Accelerator => write!(f, "Accelerator"),
            Self::Custom => write!(f, "Custom"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// OpenCL context configuration.
#[derive(Debug, Clone)]
pub struct OpenClContextConfig {
    /// Platform index (0-based)
    pub platform_index: usize,
    /// Device index within platform (0-based)
    pub device_index: usize,
    /// Enable profiling on command queue
    pub enable_profiling: bool,
    /// Enable out-of-order execution
    pub enable_out_of_order: bool,
    /// Kernel source cache directory (None = runtime compile only)
    pub kernel_cache_dir: Option<String>,
}

impl Default for OpenClContextConfig {
    fn default() -> Self {
        Self {
            platform_index: 0,
            device_index: 0,
            enable_profiling: true,
            enable_out_of_order: false,
            kernel_cache_dir: None,
        }
    }
}

/// Manages OpenCL context lifecycle and resources.
///
/// This is the primary entry point for OpenCL compute operations.
/// Create via `OpenClContextManager::new()` or `::with_config()`.
pub struct OpenClContextManager {
    config: OpenClContextConfig,
    platform_info: OpenClPlatformInfo,
    device_info: OpenClDeviceInfo,
    initialized: bool,
}

impl OpenClContextManager {
    /// Create a new context manager with default configuration.
    pub fn new() -> Result<Self, OpenClError> {
        Self::with_config(OpenClContextConfig::default())
    }

    /// Create a new context manager with custom configuration.
    pub fn with_config(config: OpenClContextConfig) -> Result<Self, OpenClError> {
        // Without actual OpenCL, return a CPU-mode stub
        let platform_info = OpenClPlatformInfo {
            name: "CPU Reference (no OpenCL runtime)".to_string(),
            vendor: "BitNet-rs".to_string(),
            version: "OpenCL 1.2 (CPU reference)".to_string(),
            profile: "FULL_PROFILE".to_string(),
            extensions: vec![],
        };

        let device_info = OpenClDeviceInfo {
            name: "CPU Reference Device".to_string(),
            vendor: "BitNet-rs".to_string(),
            device_type: OpenClDeviceType::Cpu,
            max_compute_units: num_cpus_fallback(),
            max_clock_freq_mhz: 0,
            global_mem_bytes: 0,
            local_mem_bytes: 65536,
            max_work_group_size: 1024,
            max_work_item_dimensions: 3,
            max_work_item_sizes: vec![1024, 1024, 1024],
            opencl_version: "1.2".to_string(),
            driver_version: "cpu-ref".to_string(),
            supports_fp16: false,
            supports_fp64: true,
            preferred_vector_width_float: 4,
        };

        Ok(Self { config, platform_info, device_info, initialized: true })
    }

    /// Get platform information.
    pub fn platform_info(&self) -> &OpenClPlatformInfo {
        &self.platform_info
    }

    /// Get device information.
    pub fn device_info(&self) -> &OpenClDeviceInfo {
        &self.device_info
    }

    /// Get configuration.
    pub fn config(&self) -> &OpenClContextConfig {
        &self.config
    }

    /// Whether the context is initialized and ready.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Whether this is a real GPU context (vs CPU reference).
    pub fn is_gpu(&self) -> bool {
        self.device_info.device_type == OpenClDeviceType::Gpu
    }

    /// Recommended work group size for the device.
    pub fn recommended_work_group_size(&self) -> usize {
        if self.is_gpu() {
            // Intel Arc Xe-cores prefer multiples of 16
            256
        } else {
            // CPU: use compute unit count
            self.device_info.max_compute_units as usize
        }
    }

    /// Maximum global work size for a given buffer element count.
    pub fn max_global_work_size(&self, elements: usize) -> usize {
        let wg = self.recommended_work_group_size();
        elements.div_ceil(wg) * wg
    }
}

impl fmt::Debug for OpenClContextManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenClContextManager")
            .field("platform", &self.platform_info.name)
            .field("device", &self.device_info.name)
            .field("initialized", &self.initialized)
            .finish()
    }
}

/// Errors from OpenCL operations.
#[derive(Debug, Clone)]
pub enum OpenClError {
    /// No OpenCL platform found
    NoPlatform,
    /// No suitable device found
    NoDevice,
    /// Context creation failed
    ContextCreationFailed(String),
    /// Kernel compilation failed
    KernelCompileFailed { kernel_name: String, log: String },
    /// Buffer allocation failed
    BufferAllocationFailed { size_bytes: usize, reason: String },
    /// Kernel execution failed
    KernelExecutionFailed { kernel_name: String, reason: String },
    /// Data transfer failed
    DataTransferFailed(String),
    /// Invalid argument
    InvalidArgument(String),
    /// Device not initialized
    NotInitialized,
}

impl fmt::Display for OpenClError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoPlatform => write!(f, "No OpenCL platform found"),
            Self::NoDevice => write!(f, "No suitable OpenCL device found"),
            Self::ContextCreationFailed(msg) => {
                write!(f, "Context creation failed: {msg}")
            }
            Self::KernelCompileFailed { kernel_name, log } => {
                write!(f, "Kernel '{kernel_name}' compile failed: {log}")
            }
            Self::BufferAllocationFailed { size_bytes, reason } => {
                write!(f, "Buffer allocation ({size_bytes} bytes) failed: {reason}")
            }
            Self::KernelExecutionFailed { kernel_name, reason } => {
                write!(f, "Kernel '{kernel_name}' execution failed: {reason}")
            }
            Self::DataTransferFailed(msg) => {
                write!(f, "Data transfer failed: {msg}")
            }
            Self::InvalidArgument(msg) => {
                write!(f, "Invalid argument: {msg}")
            }
            Self::NotInitialized => {
                write!(f, "OpenCL context not initialized")
            }
        }
    }
}

impl std::error::Error for OpenClError {}

fn num_cpus_fallback() -> u32 {
    std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Default config tests ────────────────────────────────────────

    #[test]
    fn default_config_platform_index() {
        let cfg = OpenClContextConfig::default();
        assert_eq!(cfg.platform_index, 0);
    }

    #[test]
    fn default_config_device_index() {
        let cfg = OpenClContextConfig::default();
        assert_eq!(cfg.device_index, 0);
    }

    #[test]
    fn default_config_profiling_enabled() {
        let cfg = OpenClContextConfig::default();
        assert!(cfg.enable_profiling);
    }

    #[test]
    fn default_config_out_of_order_disabled() {
        let cfg = OpenClContextConfig::default();
        assert!(!cfg.enable_out_of_order);
    }

    #[test]
    fn default_config_no_kernel_cache() {
        let cfg = OpenClContextConfig::default();
        assert!(cfg.kernel_cache_dir.is_none());
    }

    // ── Config customization tests ──────────────────────────────────

    #[test]
    fn config_custom_platform_index() {
        let cfg = OpenClContextConfig { platform_index: 2, ..Default::default() };
        assert_eq!(cfg.platform_index, 2);
    }

    #[test]
    fn config_custom_device_index() {
        let cfg = OpenClContextConfig { device_index: 3, ..Default::default() };
        assert_eq!(cfg.device_index, 3);
    }

    #[test]
    fn config_disable_profiling() {
        let cfg = OpenClContextConfig { enable_profiling: false, ..Default::default() };
        assert!(!cfg.enable_profiling);
    }

    #[test]
    fn config_enable_out_of_order() {
        let cfg = OpenClContextConfig { enable_out_of_order: true, ..Default::default() };
        assert!(cfg.enable_out_of_order);
    }

    #[test]
    fn config_custom_kernel_cache_dir() {
        let cfg = OpenClContextConfig {
            kernel_cache_dir: Some("/tmp/kernels".to_string()),
            ..Default::default()
        };
        assert_eq!(cfg.kernel_cache_dir.as_deref(), Some("/tmp/kernels"));
    }

    #[test]
    fn config_debug_impl() {
        let cfg = OpenClContextConfig::default();
        let dbg = format!("{cfg:?}");
        assert!(dbg.contains("OpenClContextConfig"));
    }

    // ── Context creation tests ──────────────────────────────────────

    #[test]
    fn context_new_succeeds() {
        let ctx = OpenClContextManager::new();
        assert!(ctx.is_ok());
    }

    #[test]
    fn context_with_default_config_succeeds() {
        let ctx = OpenClContextManager::with_config(OpenClContextConfig::default());
        assert!(ctx.is_ok());
    }

    #[test]
    fn context_with_custom_config_succeeds() {
        let cfg = OpenClContextConfig {
            platform_index: 1,
            device_index: 2,
            enable_profiling: false,
            enable_out_of_order: true,
            kernel_cache_dir: Some("/cache".to_string()),
        };
        let ctx = OpenClContextManager::with_config(cfg);
        assert!(ctx.is_ok());
    }

    #[test]
    fn context_preserves_config() {
        let cfg = OpenClContextConfig { platform_index: 5, device_index: 7, ..Default::default() };
        let ctx = OpenClContextManager::with_config(cfg).unwrap();
        assert_eq!(ctx.config().platform_index, 5);
        assert_eq!(ctx.config().device_index, 7);
    }

    // ── Platform info tests ─────────────────────────────────────────

    #[test]
    fn platform_name_populated() {
        let ctx = OpenClContextManager::new().unwrap();
        assert!(!ctx.platform_info().name.is_empty());
    }

    #[test]
    fn platform_vendor_populated() {
        let ctx = OpenClContextManager::new().unwrap();
        assert!(!ctx.platform_info().vendor.is_empty());
    }

    #[test]
    fn platform_version_populated() {
        let ctx = OpenClContextManager::new().unwrap();
        assert!(!ctx.platform_info().version.is_empty());
    }

    #[test]
    fn platform_profile_is_full() {
        let ctx = OpenClContextManager::new().unwrap();
        assert_eq!(ctx.platform_info().profile, "FULL_PROFILE");
    }

    #[test]
    fn platform_info_clone() {
        let ctx = OpenClContextManager::new().unwrap();
        let cloned = ctx.platform_info().clone();
        assert_eq!(cloned.name, ctx.platform_info().name);
    }

    #[test]
    fn platform_info_debug() {
        let ctx = OpenClContextManager::new().unwrap();
        let dbg = format!("{:?}", ctx.platform_info());
        assert!(dbg.contains("OpenClPlatformInfo"));
    }

    // ── Device info tests ───────────────────────────────────────────

    #[test]
    fn device_name_populated() {
        let ctx = OpenClContextManager::new().unwrap();
        assert!(!ctx.device_info().name.is_empty());
    }

    #[test]
    fn device_vendor_populated() {
        let ctx = OpenClContextManager::new().unwrap();
        assert!(!ctx.device_info().vendor.is_empty());
    }

    #[test]
    fn device_type_is_cpu_for_stub() {
        let ctx = OpenClContextManager::new().unwrap();
        assert_eq!(ctx.device_info().device_type, OpenClDeviceType::Cpu);
    }

    #[test]
    fn device_compute_units_positive() {
        let ctx = OpenClContextManager::new().unwrap();
        assert!(ctx.device_info().max_compute_units > 0);
    }

    #[test]
    fn device_local_mem_nonzero() {
        let ctx = OpenClContextManager::new().unwrap();
        assert!(ctx.device_info().local_mem_bytes > 0);
    }

    #[test]
    fn device_max_work_group_size_nonzero() {
        let ctx = OpenClContextManager::new().unwrap();
        assert!(ctx.device_info().max_work_group_size > 0);
    }

    #[test]
    fn device_work_item_dimensions_is_3() {
        let ctx = OpenClContextManager::new().unwrap();
        assert_eq!(ctx.device_info().max_work_item_dimensions, 3);
    }

    #[test]
    fn device_work_item_sizes_length_matches_dimensions() {
        let ctx = OpenClContextManager::new().unwrap();
        assert_eq!(
            ctx.device_info().max_work_item_sizes.len(),
            ctx.device_info().max_work_item_dimensions as usize
        );
    }

    #[test]
    fn device_work_item_sizes_all_positive() {
        let ctx = OpenClContextManager::new().unwrap();
        for &size in &ctx.device_info().max_work_item_sizes {
            assert!(size > 0);
        }
    }

    #[test]
    fn device_opencl_version_populated() {
        let ctx = OpenClContextManager::new().unwrap();
        assert!(!ctx.device_info().opencl_version.is_empty());
    }

    #[test]
    fn device_driver_version_populated() {
        let ctx = OpenClContextManager::new().unwrap();
        assert!(!ctx.device_info().driver_version.is_empty());
    }

    #[test]
    fn device_supports_fp64_for_cpu_stub() {
        let ctx = OpenClContextManager::new().unwrap();
        assert!(ctx.device_info().supports_fp64);
    }

    #[test]
    fn device_preferred_vector_width_positive() {
        let ctx = OpenClContextManager::new().unwrap();
        assert!(ctx.device_info().preferred_vector_width_float > 0);
    }

    #[test]
    fn device_info_clone() {
        let ctx = OpenClContextManager::new().unwrap();
        let cloned = ctx.device_info().clone();
        assert_eq!(cloned.name, ctx.device_info().name);
        assert_eq!(cloned.device_type, ctx.device_info().device_type);
    }

    #[test]
    fn device_info_debug() {
        let ctx = OpenClContextManager::new().unwrap();
        let dbg = format!("{:?}", ctx.device_info());
        assert!(dbg.contains("OpenClDeviceInfo"));
    }

    // ── Device type tests ───────────────────────────────────────────

    #[test]
    fn device_type_display_gpu() {
        assert_eq!(OpenClDeviceType::Gpu.to_string(), "GPU");
    }

    #[test]
    fn device_type_display_cpu() {
        assert_eq!(OpenClDeviceType::Cpu.to_string(), "CPU");
    }

    #[test]
    fn device_type_display_accelerator() {
        assert_eq!(OpenClDeviceType::Accelerator.to_string(), "Accelerator");
    }

    #[test]
    fn device_type_display_custom() {
        assert_eq!(OpenClDeviceType::Custom.to_string(), "Custom");
    }

    #[test]
    fn device_type_display_unknown() {
        assert_eq!(OpenClDeviceType::Unknown.to_string(), "Unknown");
    }

    #[test]
    fn device_type_eq() {
        assert_eq!(OpenClDeviceType::Gpu, OpenClDeviceType::Gpu);
        assert_ne!(OpenClDeviceType::Gpu, OpenClDeviceType::Cpu);
    }

    #[test]
    fn device_type_clone() {
        let t = OpenClDeviceType::Accelerator;
        let c = t;
        assert_eq!(t, c);
    }

    #[test]
    fn device_type_debug() {
        let dbg = format!("{:?}", OpenClDeviceType::Gpu);
        assert_eq!(dbg, "Gpu");
    }

    // ── Error type tests ────────────────────────────────────────────

    #[test]
    fn error_display_no_platform() {
        let e = OpenClError::NoPlatform;
        assert_eq!(e.to_string(), "No OpenCL platform found");
    }

    #[test]
    fn error_display_no_device() {
        let e = OpenClError::NoDevice;
        assert_eq!(e.to_string(), "No suitable OpenCL device found");
    }

    #[test]
    fn error_display_context_creation_failed() {
        let e = OpenClError::ContextCreationFailed("timeout".to_string());
        assert!(e.to_string().contains("timeout"));
    }

    #[test]
    fn error_display_kernel_compile_failed() {
        let e = OpenClError::KernelCompileFailed {
            kernel_name: "matmul".to_string(),
            log: "syntax error".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("matmul"));
        assert!(s.contains("syntax error"));
    }

    #[test]
    fn error_display_buffer_allocation_failed() {
        let e = OpenClError::BufferAllocationFailed {
            size_bytes: 1024,
            reason: "out of memory".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("1024"));
        assert!(s.contains("out of memory"));
    }

    #[test]
    fn error_display_kernel_execution_failed() {
        let e = OpenClError::KernelExecutionFailed {
            kernel_name: "dequant".to_string(),
            reason: "invalid args".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("dequant"));
        assert!(s.contains("invalid args"));
    }

    #[test]
    fn error_display_data_transfer_failed() {
        let e = OpenClError::DataTransferFailed("bus error".to_string());
        assert!(e.to_string().contains("bus error"));
    }

    #[test]
    fn error_display_invalid_argument() {
        let e = OpenClError::InvalidArgument("null pointer".to_string());
        assert!(e.to_string().contains("null pointer"));
    }

    #[test]
    fn error_display_not_initialized() {
        let e = OpenClError::NotInitialized;
        assert_eq!(e.to_string(), "OpenCL context not initialized");
    }

    #[test]
    fn error_debug_impl() {
        let e = OpenClError::NoPlatform;
        let dbg = format!("{e:?}");
        assert!(dbg.contains("NoPlatform"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(OpenClError::NoPlatform);
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn error_clone() {
        let e = OpenClError::ContextCreationFailed("test".to_string());
        let c = e.clone();
        assert_eq!(e.to_string(), c.to_string());
    }

    // ── Context manager method tests ────────────────────────────────

    #[test]
    fn is_initialized_true_after_new() {
        let ctx = OpenClContextManager::new().unwrap();
        assert!(ctx.is_initialized());
    }

    #[test]
    fn is_gpu_false_for_cpu_stub() {
        let ctx = OpenClContextManager::new().unwrap();
        assert!(!ctx.is_gpu());
    }

    #[test]
    fn recommended_work_group_size_cpu() {
        let ctx = OpenClContextManager::new().unwrap();
        let wg = ctx.recommended_work_group_size();
        // CPU stub: should equal compute unit count
        assert_eq!(wg, ctx.device_info().max_compute_units as usize);
    }

    #[test]
    fn max_global_work_size_exact_multiple() {
        let ctx = OpenClContextManager::new().unwrap();
        let wg = ctx.recommended_work_group_size();
        let elements = wg * 4;
        assert_eq!(ctx.max_global_work_size(elements), elements);
    }

    #[test]
    fn max_global_work_size_rounds_up() {
        let ctx = OpenClContextManager::new().unwrap();
        let wg = ctx.recommended_work_group_size();
        let elements = wg * 3 + 1;
        let global = ctx.max_global_work_size(elements);
        assert_eq!(global, wg * 4);
        assert!(global >= elements);
        assert_eq!(global % wg, 0);
    }

    #[test]
    fn max_global_work_size_single_element() {
        let ctx = OpenClContextManager::new().unwrap();
        let wg = ctx.recommended_work_group_size();
        assert_eq!(ctx.max_global_work_size(1), wg);
    }

    #[test]
    fn max_global_work_size_zero_elements() {
        let ctx = OpenClContextManager::new().unwrap();
        assert_eq!(ctx.max_global_work_size(0), 0);
    }

    #[test]
    fn context_debug_impl() {
        let ctx = OpenClContextManager::new().unwrap();
        let dbg = format!("{ctx:?}");
        assert!(dbg.contains("OpenClContextManager"));
        assert!(dbg.contains("initialized"));
    }

    #[test]
    fn context_debug_shows_platform_name() {
        let ctx = OpenClContextManager::new().unwrap();
        let dbg = format!("{ctx:?}");
        assert!(dbg.contains("CPU Reference"));
    }

    #[test]
    fn context_debug_shows_device_name() {
        let ctx = OpenClContextManager::new().unwrap();
        let dbg = format!("{ctx:?}");
        assert!(dbg.contains("CPU Reference Device"));
    }

    // ── num_cpus_fallback tests ─────────────────────────────────────

    #[test]
    fn num_cpus_fallback_positive() {
        assert!(num_cpus_fallback() > 0);
    }

    #[test]
    fn num_cpus_fallback_reasonable() {
        let n = num_cpus_fallback();
        // Should be between 1 and 1024 for any reasonable machine
        assert!(n >= 1 && n <= 1024);
    }
}
