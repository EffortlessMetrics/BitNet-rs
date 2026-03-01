//! Comprehensive GPU error catalog for all GPU operations across backends.
//!
//! Provides structured error types with numeric codes organized into 7
//! categories (Device, Kernel, Memory, Pipeline, Execution, Format, Backend),
//! an error registry with descriptions, common causes, and suggestions, and
//! full `std::error::Error` support including source chaining.

use std::collections::HashMap;
use std::fmt;

// ── ErrorCode ────────────────────────────────────────────────────────

/// Numeric error codes grouped by category range.
///
/// - 1xxx: Device errors
/// - 2xxx: Kernel errors
/// - 3xxx: Memory errors
/// - 4xxx: Pipeline errors
/// - 5xxx: Execution errors
/// - 6xxx: Format errors
/// - 7xxx: Backend errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ErrorCode {
    // Device errors (1xxx)
    DeviceNotFound = 1001,
    DeviceNotSupported = 1002,
    DeviceLost = 1003,
    InsufficientMemory = 1004,
    DriverVersionMismatch = 1005,
    DriverNotInstalled = 1006,

    // Kernel errors (2xxx)
    KernelCompilationFailed = 2001,
    KernelNotFound = 2002,
    InvalidKernelArgs = 2003,
    KernelTimeout = 2004,
    WorkgroupSizeExceeded = 2005,

    // Memory errors (3xxx)
    AllocationFailed = 3001,
    BufferOverflow = 3002,
    InvalidMemoryAccess = 3003,
    TransferFailed = 3004,
    MappingFailed = 3005,

    // Pipeline errors (4xxx)
    PipelineCreationFailed = 4001,
    ShaderCompilationFailed = 4002,
    DescriptorSetError = 4003,
    InvalidPipelineState = 4004,

    // Execution errors (5xxx)
    QueueSubmissionFailed = 5001,
    SynchronizationError = 5002,
    CommandBufferError = 5003,
    TimeoutExpired = 5004,

    // Format errors (6xxx)
    UnsupportedFormat = 6001,
    QuantizationError = 6002,
    ShapeMismatch = 6003,
    DtypeMismatch = 6004,

    // Backend errors (7xxx)
    BackendNotAvailable = 7001,
    FeatureNotSupported = 7002,
    BackendInitFailed = 7003,
    FallbackFailed = 7004,
}

impl ErrorCode {
    /// All defined error codes, in declaration order.
    pub const ALL: &[ErrorCode] = &[
        ErrorCode::DeviceNotFound,
        ErrorCode::DeviceNotSupported,
        ErrorCode::DeviceLost,
        ErrorCode::InsufficientMemory,
        ErrorCode::DriverVersionMismatch,
        ErrorCode::DriverNotInstalled,
        ErrorCode::KernelCompilationFailed,
        ErrorCode::KernelNotFound,
        ErrorCode::InvalidKernelArgs,
        ErrorCode::KernelTimeout,
        ErrorCode::WorkgroupSizeExceeded,
        ErrorCode::AllocationFailed,
        ErrorCode::BufferOverflow,
        ErrorCode::InvalidMemoryAccess,
        ErrorCode::TransferFailed,
        ErrorCode::MappingFailed,
        ErrorCode::PipelineCreationFailed,
        ErrorCode::ShaderCompilationFailed,
        ErrorCode::DescriptorSetError,
        ErrorCode::InvalidPipelineState,
        ErrorCode::QueueSubmissionFailed,
        ErrorCode::SynchronizationError,
        ErrorCode::CommandBufferError,
        ErrorCode::TimeoutExpired,
        ErrorCode::UnsupportedFormat,
        ErrorCode::QuantizationError,
        ErrorCode::ShapeMismatch,
        ErrorCode::DtypeMismatch,
        ErrorCode::BackendNotAvailable,
        ErrorCode::FeatureNotSupported,
        ErrorCode::BackendInitFailed,
        ErrorCode::FallbackFailed,
    ];

    /// Numeric value of this error code.
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    /// Category this error code belongs to.
    pub fn category(self) -> ErrorCategory {
        match self.as_u32() / 1000 {
            1 => ErrorCategory::Device,
            2 => ErrorCategory::Kernel,
            3 => ErrorCategory::Memory,
            4 => ErrorCategory::Pipeline,
            5 => ErrorCategory::Execution,
            6 => ErrorCategory::Format,
            7 => ErrorCategory::Backend,
            _ => unreachable!("invalid error code range"),
        }
    }

    /// Default severity for this error code.
    pub fn severity(self) -> ErrorSeverity {
        match self {
            // Fatal — unrecoverable hardware/driver issues
            ErrorCode::DeviceLost | ErrorCode::DriverNotInstalled => ErrorSeverity::Fatal,

            // Warnings — degraded but potentially recoverable
            ErrorCode::DriverVersionMismatch | ErrorCode::FeatureNotSupported => {
                ErrorSeverity::Warning
            }

            // Everything else is a standard error
            _ => ErrorSeverity::Error,
        }
    }

    /// Whether this error is potentially recoverable (retry, fallback, etc.).
    pub fn is_recoverable(self) -> bool {
        matches!(
            self,
            ErrorCode::KernelTimeout
                | ErrorCode::TimeoutExpired
                | ErrorCode::TransferFailed
                | ErrorCode::QueueSubmissionFailed
                | ErrorCode::SynchronizationError
                | ErrorCode::BackendNotAvailable
                | ErrorCode::FallbackFailed
                | ErrorCode::DriverVersionMismatch
                | ErrorCode::FeatureNotSupported
        )
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{}", self.as_u32())
    }
}

// ── ErrorCategory ────────────────────────────────────────────────────

/// High-level grouping for error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    Device,
    Kernel,
    Memory,
    Pipeline,
    Execution,
    Format,
    Backend,
}

impl ErrorCategory {
    /// All categories.
    pub const ALL: &[ErrorCategory] = &[
        ErrorCategory::Device,
        ErrorCategory::Kernel,
        ErrorCategory::Memory,
        ErrorCategory::Pipeline,
        ErrorCategory::Execution,
        ErrorCategory::Format,
        ErrorCategory::Backend,
    ];
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            ErrorCategory::Device => "Device",
            ErrorCategory::Kernel => "Kernel",
            ErrorCategory::Memory => "Memory",
            ErrorCategory::Pipeline => "Pipeline",
            ErrorCategory::Execution => "Execution",
            ErrorCategory::Format => "Format",
            ErrorCategory::Backend => "Backend",
        };
        f.write_str(label)
    }
}

// ── ErrorSeverity ────────────────────────────────────────────────────

/// Severity classification for GPU errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorSeverity {
    Warning,
    Error,
    Fatal,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            ErrorSeverity::Warning => "WARNING",
            ErrorSeverity::Error => "ERROR",
            ErrorSeverity::Fatal => "FATAL",
        };
        f.write_str(label)
    }
}

// ── GpuError ─────────────────────────────────────────────────────────

/// Rich GPU error with code, context, backend info, and source chaining.
pub struct GpuError {
    code: ErrorCode,
    message: String,
    backend: Option<String>,
    context: Vec<(String, String)>,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
    suggestion: Option<String>,
}

impl GpuError {
    /// Create a new GPU error with the given code and message.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            backend: None,
            context: Vec::new(),
            source: None,
            suggestion: None,
        }
    }

    /// Attach a backend name (e.g. `"cuda"`, `"vulkan"`).
    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.backend = Some(backend.into());
        self
    }

    /// Append a key-value context entry.
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.push((key.into(), value.into()));
        self
    }

    /// Attach a human-readable suggestion for resolving this error.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Attach a source/cause error for chaining.
    pub fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    /// The error code.
    pub fn code(&self) -> ErrorCode {
        self.code
    }

    /// The human-readable message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The backend name, if set.
    pub fn backend(&self) -> Option<&str> {
        self.backend.as_deref()
    }

    /// Context key-value pairs.
    pub fn context(&self) -> &[(String, String)] {
        &self.context
    }

    /// Suggestion string, if any.
    pub fn suggestion(&self) -> Option<&str> {
        self.suggestion.as_deref()
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)?;
        if let Some(ref backend) = self.backend {
            write!(f, " (backend: {backend})")?;
        }
        for (k, v) in &self.context {
            write!(f, " [{k}={v}]")?;
        }
        if let Some(ref suggestion) = self.suggestion {
            write!(f, " — suggestion: {suggestion}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuError")
            .field("code", &self.code)
            .field("message", &self.message)
            .field("backend", &self.backend)
            .field("context", &self.context)
            .field("source", &self.source)
            .field("suggestion", &self.suggestion)
            .finish()
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as &(dyn std::error::Error + 'static))
    }
}

// ── ErrorInfo ────────────────────────────────────────────────────────

/// Static metadata about an error code, stored in the [`ErrorRegistry`].
#[derive(Debug, Clone)]
pub struct ErrorInfo {
    code: ErrorCode,
    category: ErrorCategory,
    severity: ErrorSeverity,
    description: String,
    common_causes: Vec<String>,
    suggestions: Vec<String>,
    documentation_url: Option<String>,
}

impl ErrorInfo {
    pub fn code(&self) -> ErrorCode {
        self.code
    }

    pub fn category(&self) -> ErrorCategory {
        self.category
    }

    pub fn severity(&self) -> ErrorSeverity {
        self.severity
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn common_causes(&self) -> &[String] {
        &self.common_causes
    }

    pub fn suggestions(&self) -> &[String] {
        &self.suggestions
    }

    pub fn documentation_url(&self) -> Option<&str> {
        self.documentation_url.as_deref()
    }
}

// ── ErrorRegistry ────────────────────────────────────────────────────

/// Pre-populated registry mapping every [`ErrorCode`] to its [`ErrorInfo`].
#[derive(Debug)]
pub struct ErrorRegistry {
    errors: HashMap<ErrorCode, ErrorInfo>,
}

impl ErrorRegistry {
    /// Build the default registry with entries for every error code.
    pub fn new() -> Self {
        let mut errors = HashMap::new();

        // --- Device errors (1xxx) ---
        errors.insert(
            ErrorCode::DeviceNotFound,
            ErrorInfo {
                code: ErrorCode::DeviceNotFound,
                category: ErrorCategory::Device,
                severity: ErrorSeverity::Error,
                description: "No compatible GPU device was found.".to_string(),
                common_causes: vec![
                    "No GPU installed in the system".to_string(),
                    "GPU driver not loaded".to_string(),
                ],
                suggestions: vec![
                    "Install a supported GPU and driver".to_string(),
                    "Check device manager for GPU presence".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::DeviceNotSupported,
            ErrorInfo {
                code: ErrorCode::DeviceNotSupported,
                category: ErrorCategory::Device,
                severity: ErrorSeverity::Error,
                description: "The detected GPU is not supported by this backend.".to_string(),
                common_causes: vec![
                    "GPU too old for required compute capability".to_string(),
                    "Backend does not support this vendor".to_string(),
                ],
                suggestions: vec![
                    "Use a newer GPU with the required features".to_string(),
                    "Switch to a backend that supports your hardware".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::DeviceLost,
            ErrorInfo {
                code: ErrorCode::DeviceLost,
                category: ErrorCategory::Device,
                severity: ErrorSeverity::Fatal,
                description: "GPU device was lost during operation.".to_string(),
                common_causes: vec![
                    "Driver crash or TDR timeout".to_string(),
                    "Hardware failure".to_string(),
                ],
                suggestions: vec![
                    "Restart the application".to_string(),
                    "Update GPU drivers".to_string(),
                    "Check GPU thermals and power supply".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::InsufficientMemory,
            ErrorInfo {
                code: ErrorCode::InsufficientMemory,
                category: ErrorCategory::Device,
                severity: ErrorSeverity::Error,
                description: "Not enough GPU memory for the requested operation.".to_string(),
                common_causes: vec![
                    "Model too large for available VRAM".to_string(),
                    "Memory fragmentation".to_string(),
                ],
                suggestions: vec![
                    "Use a smaller model or reduce batch size".to_string(),
                    "Free other GPU workloads".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::DriverVersionMismatch,
            ErrorInfo {
                code: ErrorCode::DriverVersionMismatch,
                category: ErrorCategory::Device,
                severity: ErrorSeverity::Warning,
                description: "GPU driver version does not meet requirements.".to_string(),
                common_causes: vec![
                    "Outdated driver installed".to_string(),
                    "CUDA toolkit / driver version incompatibility".to_string(),
                ],
                suggestions: vec![
                    "Update GPU driver to the recommended version".to_string(),
                    "Check compatibility matrix for your toolkit".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::DriverNotInstalled,
            ErrorInfo {
                code: ErrorCode::DriverNotInstalled,
                category: ErrorCategory::Device,
                severity: ErrorSeverity::Fatal,
                description: "Required GPU driver is not installed.".to_string(),
                common_causes: vec![
                    "Driver package not installed".to_string(),
                    "Kernel module not loaded".to_string(),
                ],
                suggestions: vec![
                    "Install the appropriate GPU driver".to_string(),
                    "Run `nvidia-smi` or equivalent to verify".to_string(),
                ],
                documentation_url: None,
            },
        );

        // --- Kernel errors (2xxx) ---
        errors.insert(
            ErrorCode::KernelCompilationFailed,
            ErrorInfo {
                code: ErrorCode::KernelCompilationFailed,
                category: ErrorCategory::Kernel,
                severity: ErrorSeverity::Error,
                description: "GPU kernel failed to compile.".to_string(),
                common_causes: vec![
                    "Syntax error in kernel source".to_string(),
                    "Unsupported intrinsic for target architecture".to_string(),
                ],
                suggestions: vec![
                    "Check kernel source for errors".to_string(),
                    "Verify target compute capability".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::KernelNotFound,
            ErrorInfo {
                code: ErrorCode::KernelNotFound,
                category: ErrorCategory::Kernel,
                severity: ErrorSeverity::Error,
                description: "Requested kernel is not registered or available.".to_string(),
                common_causes: vec![
                    "Kernel name misspelled".to_string(),
                    "Kernel not compiled for this backend".to_string(),
                ],
                suggestions: vec![
                    "Check kernel registry for available kernels".to_string(),
                    "Ensure the correct feature flags are enabled".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::InvalidKernelArgs,
            ErrorInfo {
                code: ErrorCode::InvalidKernelArgs,
                category: ErrorCategory::Kernel,
                severity: ErrorSeverity::Error,
                description: "Arguments passed to the kernel are invalid.".to_string(),
                common_causes: vec![
                    "Wrong number of arguments".to_string(),
                    "Argument type mismatch".to_string(),
                ],
                suggestions: vec![
                    "Verify argument count and types".to_string(),
                    "Check kernel signature documentation".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::KernelTimeout,
            ErrorInfo {
                code: ErrorCode::KernelTimeout,
                category: ErrorCategory::Kernel,
                severity: ErrorSeverity::Error,
                description: "Kernel execution exceeded the time limit.".to_string(),
                common_causes: vec![
                    "Infinite loop in kernel".to_string(),
                    "Workload too large for timeout setting".to_string(),
                ],
                suggestions: vec![
                    "Increase timeout or reduce workload".to_string(),
                    "Check for divergent control flow".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::WorkgroupSizeExceeded,
            ErrorInfo {
                code: ErrorCode::WorkgroupSizeExceeded,
                category: ErrorCategory::Kernel,
                severity: ErrorSeverity::Error,
                description: "Requested workgroup size exceeds device limits.".to_string(),
                common_causes: vec![
                    "Block dimensions too large".to_string(),
                    "Shared memory requirement exceeds limit".to_string(),
                ],
                suggestions: vec![
                    "Reduce workgroup dimensions".to_string(),
                    "Query device limits before dispatch".to_string(),
                ],
                documentation_url: None,
            },
        );

        // --- Memory errors (3xxx) ---
        errors.insert(
            ErrorCode::AllocationFailed,
            ErrorInfo {
                code: ErrorCode::AllocationFailed,
                category: ErrorCategory::Memory,
                severity: ErrorSeverity::Error,
                description: "GPU memory allocation failed.".to_string(),
                common_causes: vec![
                    "Out of device memory".to_string(),
                    "Allocation size exceeds device limit".to_string(),
                ],
                suggestions: vec![
                    "Free unused buffers".to_string(),
                    "Reduce allocation size".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::BufferOverflow,
            ErrorInfo {
                code: ErrorCode::BufferOverflow,
                category: ErrorCategory::Memory,
                severity: ErrorSeverity::Error,
                description: "Write exceeds allocated buffer bounds.".to_string(),
                common_causes: vec![
                    "Index out of range".to_string(),
                    "Incorrect buffer size calculation".to_string(),
                ],
                suggestions: vec![
                    "Verify buffer size matches data dimensions".to_string(),
                    "Enable bounds checking in debug mode".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::InvalidMemoryAccess,
            ErrorInfo {
                code: ErrorCode::InvalidMemoryAccess,
                category: ErrorCategory::Memory,
                severity: ErrorSeverity::Error,
                description: "Illegal GPU memory access detected.".to_string(),
                common_causes: vec![
                    "Dangling pointer / freed buffer".to_string(),
                    "Race condition in memory access".to_string(),
                ],
                suggestions: vec![
                    "Run with GPU memory sanitizer".to_string(),
                    "Check buffer lifetimes".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::TransferFailed,
            ErrorInfo {
                code: ErrorCode::TransferFailed,
                category: ErrorCategory::Memory,
                severity: ErrorSeverity::Error,
                description: "Host-device memory transfer failed.".to_string(),
                common_causes: vec![
                    "PCIe bus error".to_string(),
                    "Device disconnected during transfer".to_string(),
                ],
                suggestions: vec![
                    "Retry the transfer".to_string(),
                    "Check PCIe link health".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::MappingFailed,
            ErrorInfo {
                code: ErrorCode::MappingFailed,
                category: ErrorCategory::Memory,
                severity: ErrorSeverity::Error,
                description: "Failed to map GPU buffer to host address space.".to_string(),
                common_causes: vec![
                    "Buffer not created with mappable flag".to_string(),
                    "Already mapped elsewhere".to_string(),
                ],
                suggestions: vec![
                    "Create buffer with MAP_READ/WRITE usage".to_string(),
                    "Unmap existing mapping first".to_string(),
                ],
                documentation_url: None,
            },
        );

        // --- Pipeline errors (4xxx) ---
        errors.insert(
            ErrorCode::PipelineCreationFailed,
            ErrorInfo {
                code: ErrorCode::PipelineCreationFailed,
                category: ErrorCategory::Pipeline,
                severity: ErrorSeverity::Error,
                description: "Failed to create a compute/render pipeline.".to_string(),
                common_causes: vec![
                    "Invalid pipeline descriptor".to_string(),
                    "Shader module missing".to_string(),
                ],
                suggestions: vec![
                    "Validate pipeline descriptor fields".to_string(),
                    "Ensure all shader stages are compiled".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::ShaderCompilationFailed,
            ErrorInfo {
                code: ErrorCode::ShaderCompilationFailed,
                category: ErrorCategory::Pipeline,
                severity: ErrorSeverity::Error,
                description: "Shader program failed to compile.".to_string(),
                common_causes: vec![
                    "GLSL/HLSL/WGSL syntax error".to_string(),
                    "Unsupported shader extension".to_string(),
                ],
                suggestions: vec![
                    "Check shader source for compilation errors".to_string(),
                    "Use naga or glslc for offline validation".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::DescriptorSetError,
            ErrorInfo {
                code: ErrorCode::DescriptorSetError,
                category: ErrorCategory::Pipeline,
                severity: ErrorSeverity::Error,
                description: "Descriptor set / bind group creation failed.".to_string(),
                common_causes: vec!["Layout mismatch".to_string(), "Missing binding".to_string()],
                suggestions: vec![
                    "Verify bind group layout matches shader".to_string(),
                    "Check that all bindings are provided".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::InvalidPipelineState,
            ErrorInfo {
                code: ErrorCode::InvalidPipelineState,
                category: ErrorCategory::Pipeline,
                severity: ErrorSeverity::Error,
                description: "Pipeline is in an invalid state for this operation.".to_string(),
                common_causes: vec![
                    "Using pipeline before initialization".to_string(),
                    "Pipeline already destroyed".to_string(),
                ],
                suggestions: vec![
                    "Ensure pipeline lifecycle is correct".to_string(),
                    "Check pipeline state before dispatch".to_string(),
                ],
                documentation_url: None,
            },
        );

        // --- Execution errors (5xxx) ---
        errors.insert(
            ErrorCode::QueueSubmissionFailed,
            ErrorInfo {
                code: ErrorCode::QueueSubmissionFailed,
                category: ErrorCategory::Execution,
                severity: ErrorSeverity::Error,
                description: "Failed to submit work to the GPU queue.".to_string(),
                common_causes: vec![
                    "Queue full or stalled".to_string(),
                    "Invalid command buffer".to_string(),
                ],
                suggestions: vec![
                    "Wait for previous submissions to complete".to_string(),
                    "Validate command buffer before submission".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::SynchronizationError,
            ErrorInfo {
                code: ErrorCode::SynchronizationError,
                category: ErrorCategory::Execution,
                severity: ErrorSeverity::Error,
                description: "GPU synchronization primitive failed.".to_string(),
                common_causes: vec![
                    "Fence/semaphore wait timeout".to_string(),
                    "Deadlock between GPU operations".to_string(),
                ],
                suggestions: vec![
                    "Review synchronization dependencies".to_string(),
                    "Increase fence timeout".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::CommandBufferError,
            ErrorInfo {
                code: ErrorCode::CommandBufferError,
                category: ErrorCategory::Execution,
                severity: ErrorSeverity::Error,
                description: "Error recording or executing a command buffer.".to_string(),
                common_causes: vec![
                    "Command buffer already submitted".to_string(),
                    "Invalid command sequence".to_string(),
                ],
                suggestions: vec![
                    "Reset command buffer before reuse".to_string(),
                    "Check command recording order".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::TimeoutExpired,
            ErrorInfo {
                code: ErrorCode::TimeoutExpired,
                category: ErrorCategory::Execution,
                severity: ErrorSeverity::Error,
                description: "Operation did not complete within the timeout.".to_string(),
                common_causes: vec![
                    "GPU overloaded".to_string(),
                    "Timeout value too short".to_string(),
                ],
                suggestions: vec![
                    "Increase the timeout duration".to_string(),
                    "Reduce workload or batch size".to_string(),
                ],
                documentation_url: None,
            },
        );

        // --- Format errors (6xxx) ---
        errors.insert(
            ErrorCode::UnsupportedFormat,
            ErrorInfo {
                code: ErrorCode::UnsupportedFormat,
                category: ErrorCategory::Format,
                severity: ErrorSeverity::Error,
                description: "The tensor/data format is not supported.".to_string(),
                common_causes: vec![
                    "Format not implemented for this backend".to_string(),
                    "Unknown quantization type".to_string(),
                ],
                suggestions: vec![
                    "Convert data to a supported format".to_string(),
                    "Check backend format support table".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::QuantizationError,
            ErrorInfo {
                code: ErrorCode::QuantizationError,
                category: ErrorCategory::Format,
                severity: ErrorSeverity::Error,
                description: "Error during quantization or dequantization.".to_string(),
                common_causes: vec![
                    "Invalid scale factor".to_string(),
                    "Block size mismatch".to_string(),
                ],
                suggestions: vec![
                    "Verify quantization parameters".to_string(),
                    "Check that input dimensions are block-aligned".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::ShapeMismatch,
            ErrorInfo {
                code: ErrorCode::ShapeMismatch,
                category: ErrorCategory::Format,
                severity: ErrorSeverity::Error,
                description: "Tensor shapes are incompatible.".to_string(),
                common_causes: vec![
                    "Matrix dimensions don't align for matmul".to_string(),
                    "Broadcast rules violated".to_string(),
                ],
                suggestions: vec![
                    "Check tensor dimensions before operation".to_string(),
                    "Reshape or pad tensors as needed".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::DtypeMismatch,
            ErrorInfo {
                code: ErrorCode::DtypeMismatch,
                category: ErrorCategory::Format,
                severity: ErrorSeverity::Error,
                description: "Tensor data types do not match the expected type.".to_string(),
                common_causes: vec![
                    "Mixed precision without explicit cast".to_string(),
                    "Expected f16 but received f32".to_string(),
                ],
                suggestions: vec![
                    "Cast tensors to the expected dtype".to_string(),
                    "Check model config for expected precision".to_string(),
                ],
                documentation_url: None,
            },
        );

        // --- Backend errors (7xxx) ---
        errors.insert(
            ErrorCode::BackendNotAvailable,
            ErrorInfo {
                code: ErrorCode::BackendNotAvailable,
                category: ErrorCategory::Backend,
                severity: ErrorSeverity::Error,
                description: "Requested GPU backend is not available.".to_string(),
                common_causes: vec![
                    "Backend library not found".to_string(),
                    "Feature not compiled in".to_string(),
                ],
                suggestions: vec![
                    "Enable the backend feature flag at build time".to_string(),
                    "Install the required runtime library".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::FeatureNotSupported,
            ErrorInfo {
                code: ErrorCode::FeatureNotSupported,
                category: ErrorCategory::Backend,
                severity: ErrorSeverity::Warning,
                description: "Requested feature is not supported by this backend.".to_string(),
                common_causes: vec![
                    "Hardware lacks required capability".to_string(),
                    "Feature gated behind a newer API version".to_string(),
                ],
                suggestions: vec![
                    "Check backend capability matrix".to_string(),
                    "Fall back to a compatible code path".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::BackendInitFailed,
            ErrorInfo {
                code: ErrorCode::BackendInitFailed,
                category: ErrorCategory::Backend,
                severity: ErrorSeverity::Error,
                description: "Backend initialization failed.".to_string(),
                common_causes: vec![
                    "Missing runtime dependency".to_string(),
                    "Conflicting backend instances".to_string(),
                ],
                suggestions: vec![
                    "Verify runtime dependencies are installed".to_string(),
                    "Ensure only one backend instance is active".to_string(),
                ],
                documentation_url: None,
            },
        );
        errors.insert(
            ErrorCode::FallbackFailed,
            ErrorInfo {
                code: ErrorCode::FallbackFailed,
                category: ErrorCategory::Backend,
                severity: ErrorSeverity::Error,
                description: "All fallback backends also failed.".to_string(),
                common_causes: vec![
                    "No compatible backend available".to_string(),
                    "Each fallback hit its own error".to_string(),
                ],
                suggestions: vec![
                    "Check individual backend errors".to_string(),
                    "Install at least one supported backend".to_string(),
                ],
                documentation_url: None,
            },
        );

        Self { errors }
    }

    /// Look up metadata for an error code.
    pub fn lookup(&self, code: ErrorCode) -> Option<&ErrorInfo> {
        self.errors.get(&code)
    }

    /// Convenience: return suggestion strings for an error code.
    pub fn suggestions_for(&self, code: ErrorCode) -> Vec<String> {
        self.errors.get(&code).map(|info| info.suggestions.clone()).unwrap_or_default()
    }

    /// All errors belonging to the given category.
    pub fn errors_in_category(&self, category: ErrorCategory) -> Vec<&ErrorInfo> {
        self.errors.values().filter(|info| info.category == category).collect()
    }
}

impl Default for ErrorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Registry completeness ────────────────────────────────────

    #[test]
    fn all_error_codes_have_registry_entries() {
        let reg = ErrorRegistry::new();
        for &code in ErrorCode::ALL {
            assert!(reg.lookup(code).is_some(), "Missing registry entry for {code:?}");
        }
    }

    #[test]
    fn registry_contains_exactly_all_codes() {
        let reg = ErrorRegistry::new();
        assert_eq!(reg.errors.len(), ErrorCode::ALL.len());
    }

    #[test]
    fn all_codes_have_at_least_one_suggestion() {
        let reg = ErrorRegistry::new();
        for &code in ErrorCode::ALL {
            let suggestions = reg.suggestions_for(code);
            assert!(!suggestions.is_empty(), "{code:?} has no suggestions");
        }
    }

    #[test]
    fn all_codes_have_at_least_one_common_cause() {
        let reg = ErrorRegistry::new();
        for &code in ErrorCode::ALL {
            let info = reg.lookup(code).unwrap();
            assert!(!info.common_causes.is_empty(), "{code:?} has no common causes");
        }
    }

    #[test]
    fn all_codes_have_nonempty_description() {
        let reg = ErrorRegistry::new();
        for &code in ErrorCode::ALL {
            let info = reg.lookup(code).unwrap();
            assert!(!info.description.is_empty(), "{code:?} has empty description");
        }
    }

    // ── ErrorCode ranges ─────────────────────────────────────────

    #[test]
    fn error_code_numeric_ranges_do_not_overlap() {
        let mut seen = HashMap::<u32, ErrorCode>::new();
        for &code in ErrorCode::ALL {
            let v = code.as_u32();
            assert!(
                !seen.contains_key(&v),
                "Duplicate numeric value {v} for {code:?} and {:?}",
                seen[&v]
            );
            seen.insert(v, code);
        }
    }

    #[test]
    fn device_codes_in_1xxx_range() {
        for &code in ErrorCode::ALL {
            if code.category() == ErrorCategory::Device {
                let v = code.as_u32();
                assert!((1000..2000).contains(&v), "{code:?} ({v}) not in 1xxx");
            }
        }
    }

    #[test]
    fn kernel_codes_in_2xxx_range() {
        for &code in ErrorCode::ALL {
            if code.category() == ErrorCategory::Kernel {
                let v = code.as_u32();
                assert!((2000..3000).contains(&v), "{code:?} ({v}) not in 2xxx");
            }
        }
    }

    #[test]
    fn memory_codes_in_3xxx_range() {
        for &code in ErrorCode::ALL {
            if code.category() == ErrorCategory::Memory {
                let v = code.as_u32();
                assert!((3000..4000).contains(&v), "{code:?} ({v}) not in 3xxx");
            }
        }
    }

    #[test]
    fn pipeline_codes_in_4xxx_range() {
        for &code in ErrorCode::ALL {
            if code.category() == ErrorCategory::Pipeline {
                let v = code.as_u32();
                assert!((4000..5000).contains(&v), "{code:?} ({v}) not in 4xxx");
            }
        }
    }

    #[test]
    fn execution_codes_in_5xxx_range() {
        for &code in ErrorCode::ALL {
            if code.category() == ErrorCategory::Execution {
                let v = code.as_u32();
                assert!((5000..6000).contains(&v), "{code:?} ({v}) not in 5xxx");
            }
        }
    }

    #[test]
    fn format_codes_in_6xxx_range() {
        for &code in ErrorCode::ALL {
            if code.category() == ErrorCategory::Format {
                let v = code.as_u32();
                assert!((6000..7000).contains(&v), "{code:?} ({v}) not in 6xxx");
            }
        }
    }

    #[test]
    fn backend_codes_in_7xxx_range() {
        for &code in ErrorCode::ALL {
            if code.category() == ErrorCategory::Backend {
                let v = code.as_u32();
                assert!((7000..8000).contains(&v), "{code:?} ({v}) not in 7xxx");
            }
        }
    }

    // ── Category grouping ────────────────────────────────────────

    #[test]
    fn every_category_has_at_least_one_error() {
        let reg = ErrorRegistry::new();
        for &cat in ErrorCategory::ALL {
            assert!(!reg.errors_in_category(cat).is_empty(), "Category {cat:?} has no errors");
        }
    }

    #[test]
    fn errors_in_category_returns_correct_items() {
        let reg = ErrorRegistry::new();
        for info in reg.errors_in_category(ErrorCategory::Device) {
            assert_eq!(info.category, ErrorCategory::Device);
        }
    }

    #[test]
    fn device_category_has_six_errors() {
        let reg = ErrorRegistry::new();
        assert_eq!(reg.errors_in_category(ErrorCategory::Device).len(), 6);
    }

    #[test]
    fn kernel_category_has_five_errors() {
        let reg = ErrorRegistry::new();
        assert_eq!(reg.errors_in_category(ErrorCategory::Kernel).len(), 5);
    }

    #[test]
    fn memory_category_has_five_errors() {
        let reg = ErrorRegistry::new();
        assert_eq!(reg.errors_in_category(ErrorCategory::Memory).len(), 5);
    }

    #[test]
    fn pipeline_category_has_four_errors() {
        let reg = ErrorRegistry::new();
        assert_eq!(reg.errors_in_category(ErrorCategory::Pipeline).len(), 4);
    }

    #[test]
    fn execution_category_has_four_errors() {
        let reg = ErrorRegistry::new();
        assert_eq!(reg.errors_in_category(ErrorCategory::Execution).len(), 4);
    }

    #[test]
    fn format_category_has_four_errors() {
        let reg = ErrorRegistry::new();
        assert_eq!(reg.errors_in_category(ErrorCategory::Format).len(), 4);
    }

    #[test]
    fn backend_category_has_four_errors() {
        let reg = ErrorRegistry::new();
        assert_eq!(reg.errors_in_category(ErrorCategory::Backend).len(), 4);
    }

    #[test]
    fn registry_info_category_matches_code_category() {
        let reg = ErrorRegistry::new();
        for &code in ErrorCode::ALL {
            let info = reg.lookup(code).unwrap();
            assert_eq!(info.category(), code.category(), "Category mismatch for {code:?}");
        }
    }

    #[test]
    fn registry_info_severity_matches_code_severity() {
        let reg = ErrorRegistry::new();
        for &code in ErrorCode::ALL {
            let info = reg.lookup(code).unwrap();
            assert_eq!(info.severity(), code.severity(), "Severity mismatch for {code:?}");
        }
    }

    // ── Severity classification ──────────────────────────────────

    #[test]
    fn device_lost_is_fatal() {
        assert_eq!(ErrorCode::DeviceLost.severity(), ErrorSeverity::Fatal);
    }

    #[test]
    fn driver_not_installed_is_fatal() {
        assert_eq!(ErrorCode::DriverNotInstalled.severity(), ErrorSeverity::Fatal);
    }

    #[test]
    fn driver_version_mismatch_is_warning() {
        assert_eq!(ErrorCode::DriverVersionMismatch.severity(), ErrorSeverity::Warning);
    }

    #[test]
    fn feature_not_supported_is_warning() {
        assert_eq!(ErrorCode::FeatureNotSupported.severity(), ErrorSeverity::Warning);
    }

    #[test]
    fn allocation_failed_is_error() {
        assert_eq!(ErrorCode::AllocationFailed.severity(), ErrorSeverity::Error);
    }

    // ── Recoverability ───────────────────────────────────────────

    #[test]
    fn kernel_timeout_is_recoverable() {
        assert!(ErrorCode::KernelTimeout.is_recoverable());
    }

    #[test]
    fn timeout_expired_is_recoverable() {
        assert!(ErrorCode::TimeoutExpired.is_recoverable());
    }

    #[test]
    fn transfer_failed_is_recoverable() {
        assert!(ErrorCode::TransferFailed.is_recoverable());
    }

    #[test]
    fn device_lost_is_not_recoverable() {
        assert!(!ErrorCode::DeviceLost.is_recoverable());
    }

    #[test]
    fn buffer_overflow_is_not_recoverable() {
        assert!(!ErrorCode::BufferOverflow.is_recoverable());
    }

    #[test]
    fn fallback_failed_is_recoverable() {
        assert!(ErrorCode::FallbackFailed.is_recoverable());
    }

    // ── GpuError construction ────────────────────────────────────

    #[test]
    fn new_error_has_code_and_message() {
        let err = GpuError::new(ErrorCode::DeviceNotFound, "no GPU found");
        assert_eq!(err.code(), ErrorCode::DeviceNotFound);
        assert_eq!(err.message(), "no GPU found");
    }

    #[test]
    fn error_without_backend_returns_none() {
        let err = GpuError::new(ErrorCode::DeviceNotFound, "msg");
        assert!(err.backend().is_none());
    }

    #[test]
    fn with_backend_sets_backend_name() {
        let err = GpuError::new(ErrorCode::DeviceNotFound, "msg").with_backend("cuda");
        assert_eq!(err.backend(), Some("cuda"));
    }

    #[test]
    fn with_context_adds_key_value_pair() {
        let err = GpuError::new(ErrorCode::AllocationFailed, "oom")
            .with_context("requested_bytes", "1073741824");
        assert_eq!(err.context().len(), 1);
        assert_eq!(err.context()[0].0, "requested_bytes");
        assert_eq!(err.context()[0].1, "1073741824");
    }

    #[test]
    fn multiple_context_entries() {
        let err = GpuError::new(ErrorCode::AllocationFailed, "oom")
            .with_context("requested", "1GB")
            .with_context("available", "512MB");
        assert_eq!(err.context().len(), 2);
    }

    #[test]
    fn empty_context_by_default() {
        let err = GpuError::new(ErrorCode::DeviceNotFound, "msg");
        assert!(err.context().is_empty());
    }

    #[test]
    fn with_suggestion_sets_suggestion() {
        let err =
            GpuError::new(ErrorCode::DeviceNotFound, "msg").with_suggestion("install GPU driver");
        assert_eq!(err.suggestion(), Some("install GPU driver"));
    }

    #[test]
    fn no_suggestion_by_default() {
        let err = GpuError::new(ErrorCode::DeviceNotFound, "msg");
        assert!(err.suggestion().is_none());
    }

    // ── Display format ───────────────────────────────────────────

    #[test]
    fn display_includes_code_and_message() {
        let err = GpuError::new(ErrorCode::DeviceNotFound, "no GPU");
        let s = format!("{err}");
        assert!(s.contains("E1001"));
        assert!(s.contains("no GPU"));
    }

    #[test]
    fn display_includes_backend_when_set() {
        let err = GpuError::new(ErrorCode::DeviceNotFound, "msg").with_backend("vulkan");
        let s = format!("{err}");
        assert!(s.contains("backend: vulkan"));
    }

    #[test]
    fn display_omits_backend_when_unset() {
        let err = GpuError::new(ErrorCode::DeviceNotFound, "msg");
        let s = format!("{err}");
        assert!(!s.contains("backend"));
    }

    #[test]
    fn display_includes_context_pairs() {
        let err = GpuError::new(ErrorCode::AllocationFailed, "oom").with_context("size", "4GB");
        let s = format!("{err}");
        assert!(s.contains("[size=4GB]"));
    }

    #[test]
    fn display_includes_suggestion_when_set() {
        let err = GpuError::new(ErrorCode::DeviceNotFound, "msg").with_suggestion("try again");
        let s = format!("{err}");
        assert!(s.contains("suggestion: try again"));
    }

    // ── std::error::Error impl ───────────────────────────────────

    #[test]
    fn error_source_is_none_by_default() {
        let err = GpuError::new(ErrorCode::DeviceNotFound, "msg");
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn error_source_chains_inner_error() {
        let inner = std::io::Error::new(std::io::ErrorKind::NotFound, "device file missing");
        let err = GpuError::new(ErrorCode::DeviceLost, "lost").with_source(inner);
        let src = std::error::Error::source(&err).unwrap();
        assert!(src.to_string().contains("device file missing"));
    }

    #[test]
    fn error_can_downcast_source() {
        let inner = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "no access");
        let err = GpuError::new(ErrorCode::DeviceLost, "lost").with_source(inner);
        let src = std::error::Error::source(&err).unwrap();
        assert!(src.downcast_ref::<std::io::Error>().is_some());
    }

    #[test]
    fn gpu_error_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GpuError>();
    }

    // ── ErrorCode Display ────────────────────────────────────────

    #[test]
    fn error_code_display_format() {
        assert_eq!(format!("{}", ErrorCode::DeviceNotFound), "E1001");
        assert_eq!(format!("{}", ErrorCode::FallbackFailed), "E7004");
    }

    #[test]
    fn error_code_as_u32() {
        assert_eq!(ErrorCode::DeviceNotFound.as_u32(), 1001);
        assert_eq!(ErrorCode::KernelCompilationFailed.as_u32(), 2001);
        assert_eq!(ErrorCode::AllocationFailed.as_u32(), 3001);
        assert_eq!(ErrorCode::PipelineCreationFailed.as_u32(), 4001);
        assert_eq!(ErrorCode::QueueSubmissionFailed.as_u32(), 5001);
        assert_eq!(ErrorCode::UnsupportedFormat.as_u32(), 6001);
        assert_eq!(ErrorCode::BackendNotAvailable.as_u32(), 7001);
    }

    // ── ErrorCategory Display ────────────────────────────────────

    #[test]
    fn category_display() {
        assert_eq!(format!("{}", ErrorCategory::Device), "Device");
        assert_eq!(format!("{}", ErrorCategory::Kernel), "Kernel");
        assert_eq!(format!("{}", ErrorCategory::Memory), "Memory");
        assert_eq!(format!("{}", ErrorCategory::Pipeline), "Pipeline");
        assert_eq!(format!("{}", ErrorCategory::Execution), "Execution");
        assert_eq!(format!("{}", ErrorCategory::Format), "Format");
        assert_eq!(format!("{}", ErrorCategory::Backend), "Backend");
    }

    // ── ErrorSeverity Display ────────────────────────────────────

    #[test]
    fn severity_display() {
        assert_eq!(format!("{}", ErrorSeverity::Warning), "WARNING");
        assert_eq!(format!("{}", ErrorSeverity::Error), "ERROR");
        assert_eq!(format!("{}", ErrorSeverity::Fatal), "FATAL");
    }

    // ── ErrorRegistry ────────────────────────────────────────────

    #[test]
    fn registry_lookup_returns_correct_info() {
        let reg = ErrorRegistry::new();
        let info = reg.lookup(ErrorCode::DeviceNotFound).unwrap();
        assert_eq!(info.code(), ErrorCode::DeviceNotFound);
        assert_eq!(info.category(), ErrorCategory::Device);
    }

    #[test]
    fn registry_suggestions_for_returns_vec() {
        let reg = ErrorRegistry::new();
        let suggestions = reg.suggestions_for(ErrorCode::InsufficientMemory);
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn registry_default_equals_new() {
        let a = ErrorRegistry::new();
        let b = ErrorRegistry::default();
        assert_eq!(a.errors.len(), b.errors.len());
    }

    // ── Builder chaining ─────────────────────────────────────────

    #[test]
    fn full_builder_chain() {
        let inner = std::io::Error::new(std::io::ErrorKind::Other, "PCIe timeout");
        let err = GpuError::new(ErrorCode::TransferFailed, "DMA transfer timed out")
            .with_backend("cuda")
            .with_context("direction", "host_to_device")
            .with_context("bytes", "67108864")
            .with_suggestion("reduce transfer size")
            .with_source(inner);

        assert_eq!(err.code(), ErrorCode::TransferFailed);
        assert_eq!(err.backend(), Some("cuda"));
        assert_eq!(err.context().len(), 2);
        assert_eq!(err.suggestion(), Some("reduce transfer size"));
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn debug_format_is_nonempty() {
        let err = GpuError::new(ErrorCode::DeviceNotFound, "not found");
        let dbg = format!("{err:?}");
        assert!(!dbg.is_empty());
        assert!(dbg.contains("GpuError"));
    }

    // ── ErrorCode::ALL completeness ──────────────────────────────

    #[test]
    fn all_constant_has_32_entries() {
        assert_eq!(ErrorCode::ALL.len(), 32);
    }

    #[test]
    fn all_numeric_values_are_unique() {
        let mut values: Vec<u32> = ErrorCode::ALL.iter().map(|c| c.as_u32()).collect();
        values.sort();
        values.dedup();
        assert_eq!(values.len(), ErrorCode::ALL.len());
    }
}
