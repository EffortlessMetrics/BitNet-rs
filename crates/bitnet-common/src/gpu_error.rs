//! Comprehensive GPU error code mapping for OpenCL, Vulkan, and CUDA backends.
//!
//! Provides a unified [`GpuError`] enum that maps vendor-specific error codes
//! to a common representation with severity levels and recovery hints.

use std::fmt;

/// Severity level of a GPU error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorSeverity {
    /// Unrecoverable — the device or context is likely corrupt.
    Fatal,
    /// The operation failed but the device is still usable.
    Recoverable,
    /// A transient condition (e.g. resource busy) that may succeed on retry.
    Transient,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fatal => write!(f, "FATAL"),
            Self::Recoverable => write!(f, "RECOVERABLE"),
            Self::Transient => write!(f, "TRANSIENT"),
        }
    }
}

/// Unified GPU error covering OpenCL, Vulkan, and CUDA error codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuError {
    // ── OpenCL errors ───────────────────────────────────────────────
    /// CL_DEVICE_NOT_FOUND (-1)
    ClDeviceNotFound,
    /// CL_DEVICE_NOT_AVAILABLE (-2)
    ClDeviceNotAvailable,
    /// CL_COMPILER_NOT_AVAILABLE (-3)
    ClCompilerNotAvailable,
    /// CL_MEM_OBJECT_ALLOCATION_FAILURE (-4)
    ClMemObjectAllocationFailure,
    /// CL_OUT_OF_RESOURCES (-5)
    ClOutOfResources,
    /// CL_OUT_OF_HOST_MEMORY (-6)
    ClOutOfHostMemory,
    /// CL_BUILD_PROGRAM_FAILURE (-11)
    ClBuildProgramFailure,
    /// CL_INVALID_VALUE (-30)
    ClInvalidValue,
    /// CL_INVALID_KERNEL (-48)
    ClInvalidKernel,
    /// CL_INVALID_WORK_GROUP_SIZE (-54)
    ClInvalidWorkGroupSize,
    /// CL_INVALID_COMMAND_QUEUE (-36)
    ClInvalidCommandQueue,
    /// CL_INVALID_CONTEXT (-34)
    ClInvalidContext,

    // ── Vulkan errors ───────────────────────────────────────────────
    /// VK_ERROR_OUT_OF_HOST_MEMORY (-1)
    VkOutOfHostMemory,
    /// VK_ERROR_OUT_OF_DEVICE_MEMORY (-2)
    VkOutOfDeviceMemory,
    /// VK_ERROR_INITIALIZATION_FAILED (-3)
    VkInitializationFailed,
    /// VK_ERROR_DEVICE_LOST (-4)
    VkDeviceLost,
    /// VK_ERROR_MEMORY_MAP_FAILED (-5)
    VkMemoryMapFailed,
    /// VK_ERROR_LAYER_NOT_PRESENT (-6)
    VkLayerNotPresent,
    /// VK_ERROR_TOO_MANY_OBJECTS (-10)
    VkTooManyObjects,

    // ── CUDA errors ─────────────────────────────────────────────────
    /// CUDA_ERROR_OUT_OF_MEMORY (2)
    CudaOutOfMemory,
    /// CUDA_ERROR_NOT_INITIALIZED (3)
    CudaNotInitialized,
    /// CUDA_ERROR_INVALID_VALUE (1)
    CudaInvalidValue,
    /// CUDA_ERROR_LAUNCH_FAILED (719)
    CudaLaunchFailed,
    /// CUDA_ERROR_ILLEGAL_ADDRESS (700)
    CudaIllegalAddress,
    /// CUDA_ERROR_NO_DEVICE (100)
    CudaNoDevice,

    // ── Generic / fallback ──────────────────────────────────────────
    /// Unknown vendor error with raw code.
    Unknown {
        /// Name of the backend (e.g. "OpenCL", "Vulkan", "CUDA").
        backend: String,
        /// Raw numeric error code.
        code: i64,
    },
}

impl GpuError {
    /// Return the severity level for this error.
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            // Fatal — device/context likely unusable
            Self::ClDeviceNotFound
            | Self::ClDeviceNotAvailable
            | Self::ClCompilerNotAvailable
            | Self::ClInvalidContext
            | Self::VkDeviceLost
            | Self::VkInitializationFailed
            | Self::CudaNotInitialized
            | Self::CudaNoDevice
            | Self::CudaIllegalAddress => ErrorSeverity::Fatal,

            // Transient — may succeed on retry
            Self::ClOutOfResources
            | Self::ClOutOfHostMemory
            | Self::ClMemObjectAllocationFailure
            | Self::VkOutOfHostMemory
            | Self::VkOutOfDeviceMemory
            | Self::VkTooManyObjects
            | Self::CudaOutOfMemory => ErrorSeverity::Transient,

            // Recoverable — bad params, can fix and retry
            Self::ClBuildProgramFailure
            | Self::ClInvalidValue
            | Self::ClInvalidKernel
            | Self::ClInvalidWorkGroupSize
            | Self::ClInvalidCommandQueue
            | Self::VkMemoryMapFailed
            | Self::VkLayerNotPresent
            | Self::CudaInvalidValue
            | Self::CudaLaunchFailed => ErrorSeverity::Recoverable,

            Self::Unknown { .. } => ErrorSeverity::Recoverable,
        }
    }

    /// Human-readable recovery hint.
    pub fn recovery_hint(&self) -> &str {
        match self {
            Self::ClDeviceNotFound => "No OpenCL device found. Check driver installation.",
            Self::ClDeviceNotAvailable => {
                "OpenCL device exists but is unavailable. Another process may hold it."
            }
            Self::ClCompilerNotAvailable => {
                "OpenCL compiler missing. Install the full ICD driver (not runtime-only)."
            }
            Self::ClMemObjectAllocationFailure | Self::ClOutOfResources => {
                "GPU memory exhausted. Reduce batch size or model size."
            }
            Self::ClOutOfHostMemory => "Host (RAM) allocation failed. Free system memory.",
            Self::ClBuildProgramFailure => {
                "Kernel compilation failed. Check kernel source for syntax errors."
            }
            Self::ClInvalidValue => "Invalid argument passed to OpenCL API.",
            Self::ClInvalidKernel => {
                "Kernel object is invalid. Ensure the program was built successfully."
            }
            Self::ClInvalidWorkGroupSize => {
                "Work-group size exceeds device limit. Query CL_DEVICE_MAX_WORK_GROUP_SIZE."
            }
            Self::ClInvalidCommandQueue => "Command queue is invalid. It may have been released.",
            Self::ClInvalidContext => "OpenCL context is invalid or was released.",

            Self::VkOutOfHostMemory => "Vulkan host memory exhausted. Free system memory.",
            Self::VkOutOfDeviceMemory => "Vulkan device memory exhausted. Reduce allocation sizes.",
            Self::VkInitializationFailed => {
                "Vulkan initialization failed. Check driver and loader installation."
            }
            Self::VkDeviceLost => {
                "Vulkan device lost. The GPU may have hung — reset or restart required."
            }
            Self::VkMemoryMapFailed => "Vulkan memory mapping failed. Check memory type flags.",
            Self::VkLayerNotPresent => {
                "Requested Vulkan layer is not available. Remove it from enabled layers."
            }
            Self::VkTooManyObjects => {
                "Too many Vulkan objects allocated. Destroy unused resources."
            }

            Self::CudaOutOfMemory => "CUDA out of memory. Reduce batch size or free GPU memory.",
            Self::CudaNotInitialized => "CUDA not initialized. Call cuInit() first.",
            Self::CudaInvalidValue => "Invalid value passed to CUDA API.",
            Self::CudaLaunchFailed => {
                "CUDA kernel launch failed. Check grid/block dimensions and shared memory."
            }
            Self::CudaIllegalAddress => {
                "CUDA illegal memory access. Check buffer bounds and pointer validity."
            }
            Self::CudaNoDevice => "No CUDA-capable device found. Check driver installation.",

            Self::Unknown { .. } => "Unknown GPU error. Check vendor documentation.",
        }
    }

    /// The vendor-specific numeric error code (where applicable).
    pub fn raw_code(&self) -> Option<i64> {
        match self {
            Self::ClDeviceNotFound => Some(-1),
            Self::ClDeviceNotAvailable => Some(-2),
            Self::ClCompilerNotAvailable => Some(-3),
            Self::ClMemObjectAllocationFailure => Some(-4),
            Self::ClOutOfResources => Some(-5),
            Self::ClOutOfHostMemory => Some(-6),
            Self::ClBuildProgramFailure => Some(-11),
            Self::ClInvalidValue => Some(-30),
            Self::ClInvalidContext => Some(-34),
            Self::ClInvalidCommandQueue => Some(-36),
            Self::ClInvalidKernel => Some(-48),
            Self::ClInvalidWorkGroupSize => Some(-54),

            Self::VkOutOfHostMemory => Some(-1),
            Self::VkOutOfDeviceMemory => Some(-2),
            Self::VkInitializationFailed => Some(-3),
            Self::VkDeviceLost => Some(-4),
            Self::VkMemoryMapFailed => Some(-5),
            Self::VkLayerNotPresent => Some(-6),
            Self::VkTooManyObjects => Some(-10),

            Self::CudaInvalidValue => Some(1),
            Self::CudaOutOfMemory => Some(2),
            Self::CudaNotInitialized => Some(3),
            Self::CudaNoDevice => Some(100),
            Self::CudaIllegalAddress => Some(700),
            Self::CudaLaunchFailed => Some(719),

            Self::Unknown { code, .. } => Some(*code),
        }
    }

    /// Construct a [`GpuError`] from an OpenCL error code.
    pub fn from_opencl(code: i32) -> Self {
        match code {
            -1 => Self::ClDeviceNotFound,
            -2 => Self::ClDeviceNotAvailable,
            -3 => Self::ClCompilerNotAvailable,
            -4 => Self::ClMemObjectAllocationFailure,
            -5 => Self::ClOutOfResources,
            -6 => Self::ClOutOfHostMemory,
            -11 => Self::ClBuildProgramFailure,
            -30 => Self::ClInvalidValue,
            -34 => Self::ClInvalidContext,
            -36 => Self::ClInvalidCommandQueue,
            -48 => Self::ClInvalidKernel,
            -54 => Self::ClInvalidWorkGroupSize,
            _ => Self::Unknown { backend: "OpenCL".into(), code: code as i64 },
        }
    }

    /// Construct a [`GpuError`] from a Vulkan result code.
    pub fn from_vulkan(code: i32) -> Self {
        match code {
            -1 => Self::VkOutOfHostMemory,
            -2 => Self::VkOutOfDeviceMemory,
            -3 => Self::VkInitializationFailed,
            -4 => Self::VkDeviceLost,
            -5 => Self::VkMemoryMapFailed,
            -6 => Self::VkLayerNotPresent,
            -10 => Self::VkTooManyObjects,
            _ => Self::Unknown { backend: "Vulkan".into(), code: code as i64 },
        }
    }

    /// Construct a [`GpuError`] from a CUDA error code.
    pub fn from_cuda(code: u32) -> Self {
        match code {
            1 => Self::CudaInvalidValue,
            2 => Self::CudaOutOfMemory,
            3 => Self::CudaNotInitialized,
            100 => Self::CudaNoDevice,
            700 => Self::CudaIllegalAddress,
            719 => Self::CudaLaunchFailed,
            _ => Self::Unknown { backend: "CUDA".into(), code: code as i64 },
        }
    }

    /// The backend name for this error.
    pub fn backend_name(&self) -> &str {
        match self {
            Self::ClDeviceNotFound
            | Self::ClDeviceNotAvailable
            | Self::ClCompilerNotAvailable
            | Self::ClMemObjectAllocationFailure
            | Self::ClOutOfResources
            | Self::ClOutOfHostMemory
            | Self::ClBuildProgramFailure
            | Self::ClInvalidValue
            | Self::ClInvalidKernel
            | Self::ClInvalidWorkGroupSize
            | Self::ClInvalidCommandQueue
            | Self::ClInvalidContext => "OpenCL",

            Self::VkOutOfHostMemory
            | Self::VkOutOfDeviceMemory
            | Self::VkInitializationFailed
            | Self::VkDeviceLost
            | Self::VkMemoryMapFailed
            | Self::VkLayerNotPresent
            | Self::VkTooManyObjects => "Vulkan",

            Self::CudaOutOfMemory
            | Self::CudaNotInitialized
            | Self::CudaInvalidValue
            | Self::CudaLaunchFailed
            | Self::CudaIllegalAddress
            | Self::CudaNoDevice => "CUDA",

            Self::Unknown { backend, .. } => backend,
        }
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} (severity: {})",
            self.backend_name(),
            self.recovery_hint(),
            self.severity()
        )
    }
}

impl std::error::Error for GpuError {}

/// Additional context captured when a GPU error occurs.
#[derive(Debug, Clone)]
pub struct GpuErrorContext {
    /// The underlying GPU error.
    pub error: GpuError,
    /// Device name or identifier (e.g. "Intel Arc A770").
    pub device_name: String,
    /// Name of the kernel that was executing (if applicable).
    pub kernel_name: Option<String>,
    /// Buffer sizes involved in the failed operation (bytes).
    pub buffer_sizes: Vec<usize>,
    /// Free device memory at the time of the error (bytes), if known.
    pub free_memory: Option<u64>,
}

impl GpuErrorContext {
    /// Create a new context wrapping a [`GpuError`].
    pub fn new(error: GpuError, device_name: impl Into<String>) -> Self {
        Self {
            error,
            device_name: device_name.into(),
            kernel_name: None,
            buffer_sizes: Vec::new(),
            free_memory: None,
        }
    }

    /// Attach the kernel name to this context.
    pub fn with_kernel(mut self, name: impl Into<String>) -> Self {
        self.kernel_name = Some(name.into());
        self
    }

    /// Attach buffer sizes to this context.
    pub fn with_buffers(mut self, sizes: Vec<usize>) -> Self {
        self.buffer_sizes = sizes;
        self
    }

    /// Attach free-memory information.
    pub fn with_free_memory(mut self, bytes: u64) -> Self {
        self.free_memory = Some(bytes);
        self
    }

    /// Shortcut to the error severity.
    pub fn severity(&self) -> ErrorSeverity {
        self.error.severity()
    }

    /// Shortcut to the recovery hint.
    pub fn recovery_hint(&self) -> &str {
        self.error.recovery_hint()
    }
}

impl fmt::Display for GpuErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} on device '{}'", self.error, self.device_name)?;
        if let Some(ref k) = self.kernel_name {
            write!(f, " in kernel '{k}'")?;
        }
        if !self.buffer_sizes.is_empty() {
            write!(f, " buffers={:?}", self.buffer_sizes)?;
        }
        if let Some(free) = self.free_memory {
            write!(f, " free_mem={free}B")?;
        }
        Ok(())
    }
}

impl std::error::Error for GpuErrorContext {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opencl_error_from_code() {
        assert_eq!(GpuError::from_opencl(-1), GpuError::ClDeviceNotFound);
        assert_eq!(GpuError::from_opencl(-5), GpuError::ClOutOfResources);
        assert_eq!(GpuError::from_opencl(-54), GpuError::ClInvalidWorkGroupSize);
    }

    #[test]
    fn test_opencl_unknown_code() {
        let err = GpuError::from_opencl(-999);
        assert_eq!(err, GpuError::Unknown { backend: "OpenCL".into(), code: -999 });
    }

    #[test]
    fn test_vulkan_error_from_code() {
        assert_eq!(GpuError::from_vulkan(-4), GpuError::VkDeviceLost);
        assert_eq!(GpuError::from_vulkan(-2), GpuError::VkOutOfDeviceMemory);
    }

    #[test]
    fn test_cuda_error_from_code() {
        assert_eq!(GpuError::from_cuda(2), GpuError::CudaOutOfMemory);
        assert_eq!(GpuError::from_cuda(719), GpuError::CudaLaunchFailed);
    }

    #[test]
    fn test_severity_fatal() {
        assert_eq!(GpuError::ClDeviceNotFound.severity(), ErrorSeverity::Fatal);
        assert_eq!(GpuError::VkDeviceLost.severity(), ErrorSeverity::Fatal);
        assert_eq!(GpuError::CudaNoDevice.severity(), ErrorSeverity::Fatal);
    }

    #[test]
    fn test_severity_transient() {
        assert_eq!(GpuError::ClOutOfResources.severity(), ErrorSeverity::Transient);
        assert_eq!(GpuError::CudaOutOfMemory.severity(), ErrorSeverity::Transient);
        assert_eq!(GpuError::VkOutOfDeviceMemory.severity(), ErrorSeverity::Transient);
    }

    #[test]
    fn test_severity_recoverable() {
        assert_eq!(GpuError::ClBuildProgramFailure.severity(), ErrorSeverity::Recoverable);
        assert_eq!(GpuError::CudaLaunchFailed.severity(), ErrorSeverity::Recoverable);
    }

    #[test]
    fn test_recovery_hint_not_empty() {
        let errors = [
            GpuError::ClDeviceNotFound,
            GpuError::VkDeviceLost,
            GpuError::CudaOutOfMemory,
            GpuError::Unknown { backend: "test".into(), code: 0 },
        ];
        for err in &errors {
            assert!(!err.recovery_hint().is_empty(), "{err:?} has empty hint");
        }
    }

    #[test]
    fn test_raw_code_roundtrip_opencl() {
        for code in [-1, -2, -3, -4, -5, -6, -11, -30, -34, -36, -48, -54] {
            let err = GpuError::from_opencl(code);
            assert_eq!(err.raw_code(), Some(code as i64));
        }
    }

    #[test]
    fn test_backend_name() {
        assert_eq!(GpuError::ClOutOfResources.backend_name(), "OpenCL");
        assert_eq!(GpuError::VkDeviceLost.backend_name(), "Vulkan");
        assert_eq!(GpuError::CudaLaunchFailed.backend_name(), "CUDA");
    }

    #[test]
    fn test_display_format() {
        let msg = format!("{}", GpuError::ClDeviceNotFound);
        assert!(msg.contains("OpenCL"));
        assert!(msg.contains("FATAL"));
    }

    #[test]
    fn test_error_context_display() {
        let ctx = GpuErrorContext::new(GpuError::CudaOutOfMemory, "RTX 4090")
            .with_kernel("matmul_i2s")
            .with_buffers(vec![1024, 2048])
            .with_free_memory(0);
        let msg = format!("{ctx}");
        assert!(msg.contains("RTX 4090"));
        assert!(msg.contains("matmul_i2s"));
        assert!(msg.contains("1024"));
        assert!(msg.contains("free_mem=0B"));
    }

    #[test]
    fn test_error_context_severity_delegation() {
        let ctx = GpuErrorContext::new(GpuError::VkDeviceLost, "Arc A770");
        assert_eq!(ctx.severity(), ErrorSeverity::Fatal);
        assert!(!ctx.recovery_hint().is_empty());
    }

    #[test]
    fn test_error_severity_display() {
        assert_eq!(format!("{}", ErrorSeverity::Fatal), "FATAL");
        assert_eq!(format!("{}", ErrorSeverity::Recoverable), "RECOVERABLE");
        assert_eq!(format!("{}", ErrorSeverity::Transient), "TRANSIENT");
    }
}
