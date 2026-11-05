//! Comprehensive error taxonomy and handling for OpenCL operations.
//!
//! Maps every CL_* error code to a Rust enum, classifies errors into
//! categories, provides recovery strategies, and includes aggregation,
//! rate-tracking with circuit-breaker, and human-readable diagnostics.
//! All logic is CPU-only — no `opencl3` imports.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// OpenCL error codes
// ---------------------------------------------------------------------------

/// Every standard OpenCL error code (CL_SUCCESS through vendor range).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenClErrorCode {
    Success,
    DeviceNotFound,
    DeviceNotAvailable,
    CompilerNotAvailable,
    MemObjectAllocationFailure,
    OutOfResources,
    OutOfHostMemory,
    ProfilingInfoNotAvailable,
    MemCopyOverlap,
    ImageFormatMismatch,
    ImageFormatNotSupported,
    BuildProgramFailure,
    MapFailure,
    MisalignedSubBufferOffset,
    ExecStatusErrorForEventsInWaitList,
    CompileProgramFailure,
    LinkerNotAvailable,
    LinkProgramFailure,
    DevicePartitionFailed,
    KernelArgInfoNotAvailable,
    InvalidValue,
    InvalidDeviceType,
    InvalidPlatform,
    InvalidDevice,
    InvalidContext,
    InvalidQueueProperties,
    InvalidCommandQueue,
    InvalidHostPtr,
    InvalidMemObject,
    InvalidImageFormatDescriptor,
    InvalidImageSize,
    InvalidSampler,
    InvalidBinary,
    InvalidBuildOptions,
    InvalidProgram,
    InvalidProgramExecutable,
    InvalidKernelName,
    InvalidKernelDefinition,
    InvalidKernel,
    InvalidArgIndex,
    InvalidArgValue,
    InvalidArgSize,
    InvalidKernelArgs,
    InvalidWorkDimension,
    InvalidWorkGroupSize,
    InvalidWorkItemSize,
    InvalidGlobalOffset,
    InvalidEventWaitList,
    InvalidEvent,
    InvalidOperation,
    InvalidGlObject,
    InvalidBufferSize,
    InvalidMipLevel,
    InvalidGlobalWorkSize,
    InvalidProperty,
    InvalidImageDescriptor,
    InvalidCompilerOptions,
    InvalidLinkerOptions,
    InvalidDevicePartitionCount,
    InvalidPipeSize,
    InvalidDeviceQueue,
    /// Vendor-specific or unrecognised error code.
    Unknown(i32),
}

impl OpenClErrorCode {
    /// Convert a raw CL status code to the typed enum.
    pub fn from_raw(code: i32) -> Self {
        match code {
            0 => Self::Success,
            -1 => Self::DeviceNotFound,
            -2 => Self::DeviceNotAvailable,
            -3 => Self::CompilerNotAvailable,
            -4 => Self::MemObjectAllocationFailure,
            -5 => Self::OutOfResources,
            -6 => Self::OutOfHostMemory,
            -7 => Self::ProfilingInfoNotAvailable,
            -8 => Self::MemCopyOverlap,
            -9 => Self::ImageFormatMismatch,
            -10 => Self::ImageFormatNotSupported,
            -11 => Self::BuildProgramFailure,
            -12 => Self::MapFailure,
            -13 => Self::MisalignedSubBufferOffset,
            -14 => Self::ExecStatusErrorForEventsInWaitList,
            -15 => Self::CompileProgramFailure,
            -16 => Self::LinkerNotAvailable,
            -17 => Self::LinkProgramFailure,
            -18 => Self::DevicePartitionFailed,
            -19 => Self::KernelArgInfoNotAvailable,
            -30 => Self::InvalidValue,
            -31 => Self::InvalidDeviceType,
            -32 => Self::InvalidPlatform,
            -33 => Self::InvalidDevice,
            -34 => Self::InvalidContext,
            -35 => Self::InvalidQueueProperties,
            -36 => Self::InvalidCommandQueue,
            -37 => Self::InvalidHostPtr,
            -38 => Self::InvalidMemObject,
            -39 => Self::InvalidImageFormatDescriptor,
            -40 => Self::InvalidImageSize,
            -41 => Self::InvalidSampler,
            -42 => Self::InvalidBinary,
            -43 => Self::InvalidBuildOptions,
            -44 => Self::InvalidProgram,
            -45 => Self::InvalidProgramExecutable,
            -46 => Self::InvalidKernelName,
            -47 => Self::InvalidKernelDefinition,
            -48 => Self::InvalidKernel,
            -49 => Self::InvalidArgIndex,
            -50 => Self::InvalidArgValue,
            -51 => Self::InvalidArgSize,
            -52 => Self::InvalidKernelArgs,
            -53 => Self::InvalidWorkDimension,
            -54 => Self::InvalidWorkGroupSize,
            -55 => Self::InvalidWorkItemSize,
            -56 => Self::InvalidGlobalOffset,
            -57 => Self::InvalidEventWaitList,
            -58 => Self::InvalidEvent,
            -59 => Self::InvalidOperation,
            -60 => Self::InvalidGlObject,
            -61 => Self::InvalidBufferSize,
            -62 => Self::InvalidMipLevel,
            -63 => Self::InvalidGlobalWorkSize,
            -64 => Self::InvalidProperty,
            -65 => Self::InvalidImageDescriptor,
            -66 => Self::InvalidCompilerOptions,
            -67 => Self::InvalidLinkerOptions,
            -68 => Self::InvalidDevicePartitionCount,
            -69 => Self::InvalidPipeSize,
            -70 => Self::InvalidDeviceQueue,
            other => Self::Unknown(other),
        }
    }

    /// Return the raw `i32` value.
    pub fn raw_code(&self) -> i32 {
        match self {
            Self::Success => 0,
            Self::DeviceNotFound => -1,
            Self::DeviceNotAvailable => -2,
            Self::CompilerNotAvailable => -3,
            Self::MemObjectAllocationFailure => -4,
            Self::OutOfResources => -5,
            Self::OutOfHostMemory => -6,
            Self::ProfilingInfoNotAvailable => -7,
            Self::MemCopyOverlap => -8,
            Self::ImageFormatMismatch => -9,
            Self::ImageFormatNotSupported => -10,
            Self::BuildProgramFailure => -11,
            Self::MapFailure => -12,
            Self::MisalignedSubBufferOffset => -13,
            Self::ExecStatusErrorForEventsInWaitList => -14,
            Self::CompileProgramFailure => -15,
            Self::LinkerNotAvailable => -16,
            Self::LinkProgramFailure => -17,
            Self::DevicePartitionFailed => -18,
            Self::KernelArgInfoNotAvailable => -19,
            Self::InvalidValue => -30,
            Self::InvalidDeviceType => -31,
            Self::InvalidPlatform => -32,
            Self::InvalidDevice => -33,
            Self::InvalidContext => -34,
            Self::InvalidQueueProperties => -35,
            Self::InvalidCommandQueue => -36,
            Self::InvalidHostPtr => -37,
            Self::InvalidMemObject => -38,
            Self::InvalidImageFormatDescriptor => -39,
            Self::InvalidImageSize => -40,
            Self::InvalidSampler => -41,
            Self::InvalidBinary => -42,
            Self::InvalidBuildOptions => -43,
            Self::InvalidProgram => -44,
            Self::InvalidProgramExecutable => -45,
            Self::InvalidKernelName => -46,
            Self::InvalidKernelDefinition => -47,
            Self::InvalidKernel => -48,
            Self::InvalidArgIndex => -49,
            Self::InvalidArgValue => -50,
            Self::InvalidArgSize => -51,
            Self::InvalidKernelArgs => -52,
            Self::InvalidWorkDimension => -53,
            Self::InvalidWorkGroupSize => -54,
            Self::InvalidWorkItemSize => -55,
            Self::InvalidGlobalOffset => -56,
            Self::InvalidEventWaitList => -57,
            Self::InvalidEvent => -58,
            Self::InvalidOperation => -59,
            Self::InvalidGlObject => -60,
            Self::InvalidBufferSize => -61,
            Self::InvalidMipLevel => -62,
            Self::InvalidGlobalWorkSize => -63,
            Self::InvalidProperty => -64,
            Self::InvalidImageDescriptor => -65,
            Self::InvalidCompilerOptions => -66,
            Self::InvalidLinkerOptions => -67,
            Self::InvalidDevicePartitionCount => -68,
            Self::InvalidPipeSize => -69,
            Self::InvalidDeviceQueue => -70,
            Self::Unknown(v) => *v,
        }
    }

    /// CL name string, e.g. `"CL_DEVICE_NOT_FOUND"`.
    pub fn cl_name(&self) -> &'static str {
        match self {
            Self::Success => "CL_SUCCESS",
            Self::DeviceNotFound => "CL_DEVICE_NOT_FOUND",
            Self::DeviceNotAvailable => "CL_DEVICE_NOT_AVAILABLE",
            Self::CompilerNotAvailable => "CL_COMPILER_NOT_AVAILABLE",
            Self::MemObjectAllocationFailure => "CL_MEM_OBJECT_ALLOCATION_FAILURE",
            Self::OutOfResources => "CL_OUT_OF_RESOURCES",
            Self::OutOfHostMemory => "CL_OUT_OF_HOST_MEMORY",
            Self::ProfilingInfoNotAvailable => "CL_PROFILING_INFO_NOT_AVAILABLE",
            Self::MemCopyOverlap => "CL_MEM_COPY_OVERLAP",
            Self::ImageFormatMismatch => "CL_IMAGE_FORMAT_MISMATCH",
            Self::ImageFormatNotSupported => "CL_IMAGE_FORMAT_NOT_SUPPORTED",
            Self::BuildProgramFailure => "CL_BUILD_PROGRAM_FAILURE",
            Self::MapFailure => "CL_MAP_FAILURE",
            Self::MisalignedSubBufferOffset => "CL_MISALIGNED_SUB_BUFFER_OFFSET",
            Self::ExecStatusErrorForEventsInWaitList => {
                "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"
            }
            Self::CompileProgramFailure => "CL_COMPILE_PROGRAM_FAILURE",
            Self::LinkerNotAvailable => "CL_LINKER_NOT_AVAILABLE",
            Self::LinkProgramFailure => "CL_LINK_PROGRAM_FAILURE",
            Self::DevicePartitionFailed => "CL_DEVICE_PARTITION_FAILED",
            Self::KernelArgInfoNotAvailable => "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",
            Self::InvalidValue => "CL_INVALID_VALUE",
            Self::InvalidDeviceType => "CL_INVALID_DEVICE_TYPE",
            Self::InvalidPlatform => "CL_INVALID_PLATFORM",
            Self::InvalidDevice => "CL_INVALID_DEVICE",
            Self::InvalidContext => "CL_INVALID_CONTEXT",
            Self::InvalidQueueProperties => "CL_INVALID_QUEUE_PROPERTIES",
            Self::InvalidCommandQueue => "CL_INVALID_COMMAND_QUEUE",
            Self::InvalidHostPtr => "CL_INVALID_HOST_PTR",
            Self::InvalidMemObject => "CL_INVALID_MEM_OBJECT",
            Self::InvalidImageFormatDescriptor => "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
            Self::InvalidImageSize => "CL_INVALID_IMAGE_SIZE",
            Self::InvalidSampler => "CL_INVALID_SAMPLER",
            Self::InvalidBinary => "CL_INVALID_BINARY",
            Self::InvalidBuildOptions => "CL_INVALID_BUILD_OPTIONS",
            Self::InvalidProgram => "CL_INVALID_PROGRAM",
            Self::InvalidProgramExecutable => "CL_INVALID_PROGRAM_EXECUTABLE",
            Self::InvalidKernelName => "CL_INVALID_KERNEL_NAME",
            Self::InvalidKernelDefinition => "CL_INVALID_KERNEL_DEFINITION",
            Self::InvalidKernel => "CL_INVALID_KERNEL",
            Self::InvalidArgIndex => "CL_INVALID_ARG_INDEX",
            Self::InvalidArgValue => "CL_INVALID_ARG_VALUE",
            Self::InvalidArgSize => "CL_INVALID_ARG_SIZE",
            Self::InvalidKernelArgs => "CL_INVALID_KERNEL_ARGS",
            Self::InvalidWorkDimension => "CL_INVALID_WORK_DIMENSION",
            Self::InvalidWorkGroupSize => "CL_INVALID_WORK_GROUP_SIZE",
            Self::InvalidWorkItemSize => "CL_INVALID_WORK_ITEM_SIZE",
            Self::InvalidGlobalOffset => "CL_INVALID_GLOBAL_OFFSET",
            Self::InvalidEventWaitList => "CL_INVALID_EVENT_WAIT_LIST",
            Self::InvalidEvent => "CL_INVALID_EVENT",
            Self::InvalidOperation => "CL_INVALID_OPERATION",
            Self::InvalidGlObject => "CL_INVALID_GL_OBJECT",
            Self::InvalidBufferSize => "CL_INVALID_BUFFER_SIZE",
            Self::InvalidMipLevel => "CL_INVALID_MIP_LEVEL",
            Self::InvalidGlobalWorkSize => "CL_INVALID_GLOBAL_WORK_SIZE",
            Self::InvalidProperty => "CL_INVALID_PROPERTY",
            Self::InvalidImageDescriptor => "CL_INVALID_IMAGE_DESCRIPTOR",
            Self::InvalidCompilerOptions => "CL_INVALID_COMPILER_OPTIONS",
            Self::InvalidLinkerOptions => "CL_INVALID_LINKER_OPTIONS",
            Self::InvalidDevicePartitionCount => "CL_INVALID_DEVICE_PARTITION_COUNT",
            Self::InvalidPipeSize => "CL_INVALID_PIPE_SIZE",
            Self::InvalidDeviceQueue => "CL_INVALID_DEVICE_QUEUE",
            Self::Unknown(_) => "CL_UNKNOWN",
        }
    }
}

impl fmt::Display for OpenClErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.cl_name(), self.raw_code())
    }
}

// ---------------------------------------------------------------------------
// Error category
// ---------------------------------------------------------------------------

/// High-level classification of an OpenCL error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Runtime dispatch or execution failure.
    Runtime,
    /// Memory allocation or access error.
    Memory,
    /// Kernel/program compilation or link failure.
    Compilation,
    /// Device enumeration or availability issue.
    Device,
    /// Invalid argument or parameter validation.
    Validation,
    /// Timeout waiting for a kernel or event.
    Timeout,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime => write!(f, "Runtime"),
            Self::Memory => write!(f, "Memory"),
            Self::Compilation => write!(f, "Compilation"),
            Self::Device => write!(f, "Device"),
            Self::Validation => write!(f, "Validation"),
            Self::Timeout => write!(f, "Timeout"),
        }
    }
}

impl OpenClErrorCode {
    /// Classify this error code into a high-level category.
    pub fn category(&self) -> ErrorCategory {
        match self {
            // Device
            Self::DeviceNotFound
            | Self::DeviceNotAvailable
            | Self::DevicePartitionFailed
            | Self::InvalidDevice
            | Self::InvalidDeviceType
            | Self::InvalidPlatform
            | Self::InvalidDevicePartitionCount
            | Self::InvalidDeviceQueue => ErrorCategory::Device,

            // Memory
            Self::MemObjectAllocationFailure
            | Self::OutOfResources
            | Self::OutOfHostMemory
            | Self::MemCopyOverlap
            | Self::MapFailure
            | Self::MisalignedSubBufferOffset
            | Self::InvalidHostPtr
            | Self::InvalidMemObject
            | Self::InvalidBufferSize => ErrorCategory::Memory,

            // Compilation
            Self::CompilerNotAvailable
            | Self::BuildProgramFailure
            | Self::CompileProgramFailure
            | Self::LinkerNotAvailable
            | Self::LinkProgramFailure
            | Self::InvalidBinary
            | Self::InvalidBuildOptions
            | Self::InvalidProgram
            | Self::InvalidProgramExecutable
            | Self::InvalidCompilerOptions
            | Self::InvalidLinkerOptions => ErrorCategory::Compilation,

            // Validation (invalid arguments / parameters)
            Self::InvalidValue
            | Self::InvalidQueueProperties
            | Self::InvalidCommandQueue
            | Self::InvalidContext
            | Self::InvalidImageFormatDescriptor
            | Self::InvalidImageSize
            | Self::InvalidSampler
            | Self::InvalidKernelName
            | Self::InvalidKernelDefinition
            | Self::InvalidKernel
            | Self::InvalidArgIndex
            | Self::InvalidArgValue
            | Self::InvalidArgSize
            | Self::InvalidKernelArgs
            | Self::InvalidWorkDimension
            | Self::InvalidWorkGroupSize
            | Self::InvalidWorkItemSize
            | Self::InvalidGlobalOffset
            | Self::InvalidEventWaitList
            | Self::InvalidEvent
            | Self::InvalidGlObject
            | Self::InvalidMipLevel
            | Self::InvalidGlobalWorkSize
            | Self::InvalidProperty
            | Self::InvalidImageDescriptor
            | Self::InvalidPipeSize
            | Self::InvalidOperation => ErrorCategory::Validation,

            // Runtime
            Self::ProfilingInfoNotAvailable
            | Self::ImageFormatMismatch
            | Self::ImageFormatNotSupported
            | Self::ExecStatusErrorForEventsInWaitList
            | Self::KernelArgInfoNotAvailable => ErrorCategory::Runtime,

            // Success never really needs a category, but default to Runtime.
            Self::Success => ErrorCategory::Runtime,

            // Unknown codes are treated as Runtime.
            Self::Unknown(_) => ErrorCategory::Runtime,
        }
    }
}

// ---------------------------------------------------------------------------
// Error severity
// ---------------------------------------------------------------------------

/// How severe the error is for continued operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ErrorSeverity {
    /// May resolve on its own (transient resource pressure).
    Transient,
    /// Worth noting but non-blocking.
    Warning,
    /// Significant failure; operation cannot complete.
    Error,
    /// Unrecoverable; the device or context is unusable.
    Fatal,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transient => write!(f, "Transient"),
            Self::Warning => write!(f, "Warning"),
            Self::Error => write!(f, "Error"),
            Self::Fatal => write!(f, "Fatal"),
        }
    }
}

impl OpenClErrorCode {
    /// Determine severity for this error code.
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Success => ErrorSeverity::Transient,

            // Transient — may resolve with retry
            Self::OutOfResources
            | Self::OutOfHostMemory
            | Self::MemObjectAllocationFailure
            | Self::MapFailure => ErrorSeverity::Transient,

            // Warning — non-ideal but not fatal
            Self::ProfilingInfoNotAvailable
            | Self::KernelArgInfoNotAvailable
            | Self::ImageFormatNotSupported
            | Self::ImageFormatMismatch => ErrorSeverity::Warning,

            // Fatal — device/context unusable
            Self::DeviceNotFound
            | Self::DeviceNotAvailable
            | Self::CompilerNotAvailable
            | Self::LinkerNotAvailable
            | Self::DevicePartitionFailed
            | Self::InvalidContext
            | Self::InvalidDevice
            | Self::InvalidPlatform => ErrorSeverity::Fatal,

            // Everything else is a hard error.
            _ => ErrorSeverity::Error,
        }
    }
}

// ---------------------------------------------------------------------------
// Error recovery strategy
// ---------------------------------------------------------------------------

/// Recommended recovery action for a given error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorRecovery {
    /// Retry the same operation (possibly after a back-off).
    Retry,
    /// Fall back to CPU execution path.
    FallbackCpu,
    /// Abort — no automatic recovery possible.
    Abort,
    /// Rebuild kernels and retry (compilation-related).
    Recompile,
    /// Re-initialise the OpenCL context and retry.
    ReinitContext,
}

impl fmt::Display for ErrorRecovery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Retry => write!(f, "Retry"),
            Self::FallbackCpu => write!(f, "FallbackCpu"),
            Self::Abort => write!(f, "Abort"),
            Self::Recompile => write!(f, "Recompile"),
            Self::ReinitContext => write!(f, "ReinitContext"),
        }
    }
}

impl OpenClErrorCode {
    /// Suggest a recovery strategy for this error code.
    pub fn recovery(&self) -> ErrorRecovery {
        match self.category() {
            ErrorCategory::Memory => {
                if self.severity() == ErrorSeverity::Transient {
                    ErrorRecovery::Retry
                } else {
                    ErrorRecovery::FallbackCpu
                }
            }
            ErrorCategory::Compilation => ErrorRecovery::Recompile,
            ErrorCategory::Device => {
                if matches!(self, Self::DeviceNotFound | Self::DeviceNotAvailable) {
                    ErrorRecovery::FallbackCpu
                } else {
                    ErrorRecovery::ReinitContext
                }
            }
            ErrorCategory::Validation => ErrorRecovery::Abort,
            ErrorCategory::Timeout => ErrorRecovery::Retry,
            ErrorCategory::Runtime => {
                if matches!(self, Self::ExecStatusErrorForEventsInWaitList | Self::Success) {
                    ErrorRecovery::Retry
                } else {
                    ErrorRecovery::FallbackCpu
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Error context
// ---------------------------------------------------------------------------

/// Rich context captured at the point of an OpenCL error.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// The CL error code that triggered this context.
    pub error_code: OpenClErrorCode,
    /// Name of the kernel being executed (if applicable).
    pub kernel_name: Option<String>,
    /// Device description string.
    pub device_name: Option<String>,
    /// Size of the buffer involved (bytes), if applicable.
    pub buffer_size: Option<u64>,
    /// Logical call-site breadcrumbs (not a real stack trace).
    pub call_stack: Vec<String>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
    /// Timestamp of error occurrence.
    pub timestamp: Instant,
}

impl ErrorContext {
    /// Create a minimal context from an error code.
    pub fn new(code: OpenClErrorCode) -> Self {
        Self {
            error_code: code,
            kernel_name: None,
            device_name: None,
            buffer_size: None,
            call_stack: Vec::new(),
            metadata: HashMap::new(),
            timestamp: Instant::now(),
        }
    }

    /// Builder: attach a kernel name.
    pub fn with_kernel(mut self, name: impl Into<String>) -> Self {
        self.kernel_name = Some(name.into());
        self
    }

    /// Builder: attach a device description.
    pub fn with_device(mut self, name: impl Into<String>) -> Self {
        self.device_name = Some(name.into());
        self
    }

    /// Builder: attach buffer size.
    pub fn with_buffer_size(mut self, size: u64) -> Self {
        self.buffer_size = Some(size);
        self
    }

    /// Builder: push a call-site breadcrumb.
    pub fn with_call_site(mut self, site: impl Into<String>) -> Self {
        self.call_stack.push(site.into());
        self
    }

    /// Builder: add a metadata key-value pair.
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.error_code)?;
        if let Some(ref k) = self.kernel_name {
            write!(f, " kernel={k}")?;
        }
        if let Some(ref d) = self.device_name {
            write!(f, " device={d}")?;
        }
        if let Some(sz) = self.buffer_size {
            write!(f, " buf={sz}B")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Diagnostic info
// ---------------------------------------------------------------------------

/// Snapshot of system state at the time an error occurs.
#[derive(Debug, Clone)]
pub struct DiagnosticInfo {
    /// Approximate free host memory in bytes.
    pub free_host_memory_bytes: u64,
    /// Approximate total host memory in bytes.
    pub total_host_memory_bytes: u64,
    /// Number of outstanding commands in the queue (estimate).
    pub queue_depth: u32,
    /// Number of live OpenCL buffer objects (estimate).
    pub live_buffers: u32,
    /// Total bytes allocated in OpenCL buffers (estimate).
    pub allocated_buffer_bytes: u64,
    /// Optional driver version string.
    pub driver_version: Option<String>,
}

impl DiagnosticInfo {
    /// Capture a diagnostic snapshot with the given estimates.
    pub fn capture(queue_depth: u32, live_buffers: u32, allocated_buffer_bytes: u64) -> Self {
        let (free, total) = Self::host_memory_info();
        Self {
            free_host_memory_bytes: free,
            total_host_memory_bytes: total,
            queue_depth,
            live_buffers,
            allocated_buffer_bytes,
            driver_version: None,
        }
    }

    /// Builder: attach driver version.
    pub fn with_driver(mut self, ver: impl Into<String>) -> Self {
        self.driver_version = Some(ver.into());
        self
    }

    /// Query host memory via `sysinfo`.
    fn host_memory_info() -> (u64, u64) {
        use sysinfo::System;
        let sys = System::new_all();
        let total = sys.total_memory();
        let available = sys.available_memory();
        (available, total)
    }

    /// Memory utilisation ratio (0.0 – 1.0).
    pub fn host_memory_utilisation(&self) -> f64 {
        if self.total_host_memory_bytes == 0 {
            return 0.0;
        }
        let used = self.total_host_memory_bytes.saturating_sub(self.free_host_memory_bytes);
        used as f64 / self.total_host_memory_bytes as f64
    }
}

impl fmt::Display for DiagnosticInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "host_mem={}/{}MB queue={} bufs={} alloc={}MB",
            self.free_host_memory_bytes / (1024 * 1024),
            self.total_host_memory_bytes / (1024 * 1024),
            self.queue_depth,
            self.live_buffers,
            self.allocated_buffer_bytes / (1024 * 1024),
        )?;
        if let Some(ref v) = self.driver_version {
            write!(f, " driver={v}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Error report
// ---------------------------------------------------------------------------

/// Human-readable error report with root-cause analysis.
#[derive(Debug, Clone)]
pub struct ErrorReport {
    /// The error context that triggered this report.
    pub context: ErrorContext,
    /// Classified category.
    pub category: ErrorCategory,
    /// Assessed severity.
    pub severity: ErrorSeverity,
    /// Recommended recovery.
    pub recovery: ErrorRecovery,
    /// Root-cause hypothesis.
    pub root_cause: String,
    /// Optional diagnostic snapshot.
    pub diagnostics: Option<DiagnosticInfo>,
    /// Suggested human-readable remediation steps.
    pub remediation: Vec<String>,
}

impl ErrorReport {
    /// Generate a report from an [`ErrorContext`].
    pub fn from_context(ctx: ErrorContext) -> Self {
        let code = ctx.error_code;
        let category = code.category();
        let severity = code.severity();
        let recovery = code.recovery();
        let root_cause = Self::analyse_root_cause(&ctx);
        let remediation = Self::suggest_remediation(code, &ctx);
        Self {
            context: ctx,
            category,
            severity,
            recovery,
            root_cause,
            diagnostics: None,
            remediation,
        }
    }

    /// Builder: attach diagnostics.
    pub fn with_diagnostics(mut self, diag: DiagnosticInfo) -> Self {
        self.diagnostics = Some(diag);
        self
    }

    /// Simple root-cause analysis heuristic.
    fn analyse_root_cause(ctx: &ErrorContext) -> String {
        let code = ctx.error_code;
        match code.category() {
            ErrorCategory::Memory => {
                if let Some(sz) = ctx.buffer_size {
                    format!("Memory allocation of {} bytes failed ({})", sz, code.cl_name())
                } else {
                    format!("GPU memory pressure detected ({})", code.cl_name())
                }
            }
            ErrorCategory::Compilation => {
                let kernel = ctx.kernel_name.as_deref().unwrap_or("<unknown>");
                format!("Kernel '{}' failed to compile/link ({})", kernel, code.cl_name())
            }
            ErrorCategory::Device => {
                let dev = ctx.device_name.as_deref().unwrap_or("<unknown>");
                format!("Device '{}' unavailable ({})", dev, code.cl_name())
            }
            ErrorCategory::Validation => {
                let kernel = ctx.kernel_name.as_deref().unwrap_or("<unknown>");
                format!("Invalid parameter in kernel '{}' dispatch ({})", kernel, code.cl_name())
            }
            ErrorCategory::Timeout => "Kernel execution exceeded timeout".to_string(),
            ErrorCategory::Runtime => {
                format!("Runtime error during execution ({})", code.cl_name())
            }
        }
    }

    /// Produce remediation suggestions based on the error.
    fn suggest_remediation(code: OpenClErrorCode, ctx: &ErrorContext) -> Vec<String> {
        let mut steps = Vec::new();
        match code.category() {
            ErrorCategory::Memory => {
                steps.push("Reduce batch size or sequence length".into());
                steps.push("Free unused GPU buffers before retrying".into());
                if ctx.buffer_size.is_some_and(|sz| sz > 512 * 1024 * 1024) {
                    steps.push("Buffer >512 MB — consider tiled execution".into());
                }
            }
            ErrorCategory::Compilation => {
                steps.push("Check kernel source for syntax errors".into());
                steps.push("Verify OpenCL compiler version compatibility".into());
            }
            ErrorCategory::Device => {
                steps.push("Ensure GPU drivers are installed and up to date".into());
                steps.push("Try a different OpenCL platform or device".into());
            }
            ErrorCategory::Validation => {
                steps.push("Review kernel argument types and sizes".into());
                steps.push("Validate work group dimensions against device limits".into());
            }
            ErrorCategory::Timeout => {
                steps.push("Increase timeout threshold".into());
                steps.push("Break computation into smaller chunks".into());
            }
            ErrorCategory::Runtime => {
                steps.push("Check event dependencies for deadlocks".into());
                steps.push("Fall back to CPU path if error persists".into());
            }
        }
        steps
    }
}

impl fmt::Display for ErrorReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== OpenCL Error Report ===")?;
        writeln!(f, "Code:      {}", self.context.error_code)?;
        writeln!(f, "Category:  {}", self.category)?;
        writeln!(f, "Severity:  {}", self.severity)?;
        writeln!(f, "Recovery:  {}", self.recovery)?;
        writeln!(f, "Root cause: {}", self.root_cause)?;
        if let Some(ref k) = self.context.kernel_name {
            writeln!(f, "Kernel:    {k}")?;
        }
        if let Some(ref d) = self.context.device_name {
            writeln!(f, "Device:    {d}")?;
        }
        if let Some(sz) = self.context.buffer_size {
            writeln!(f, "Buffer:    {sz} bytes")?;
        }
        if !self.context.call_stack.is_empty() {
            writeln!(f, "Call stack:")?;
            for (i, frame) in self.context.call_stack.iter().enumerate() {
                writeln!(f, "  {i}: {frame}")?;
            }
        }
        if let Some(ref diag) = self.diagnostics {
            writeln!(f, "Diagnostics: {diag}")?;
        }
        if !self.remediation.is_empty() {
            writeln!(f, "Remediation:")?;
            for step in &self.remediation {
                writeln!(f, "  - {step}")?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Error aggregator
// ---------------------------------------------------------------------------

/// A record of a single aggregated error kind.
#[derive(Debug, Clone)]
pub struct AggregatedError {
    /// The error code.
    pub code: OpenClErrorCode,
    /// Kernel name (if consistent across occurrences).
    pub kernel_name: Option<String>,
    /// Total number of times this error has been seen.
    pub count: u64,
    /// Timestamp of the first occurrence.
    pub first_seen: Instant,
    /// Timestamp of the most recent occurrence.
    pub last_seen: Instant,
}

/// Collects and deduplicates errors over time.
///
/// Errors are grouped by `(OpenClErrorCode, Option<kernel_name>)`.
#[derive(Debug)]
pub struct ErrorAggregator {
    /// Map from `(raw_code, kernel_name)` → aggregated record.
    errors: HashMap<(i32, Option<String>), AggregatedError>,
    /// Maximum number of distinct error keys to track before evicting oldest.
    max_keys: usize,
}

impl ErrorAggregator {
    /// Create a new aggregator with a max-key limit.
    pub fn new(max_keys: usize) -> Self {
        Self { errors: HashMap::new(), max_keys }
    }

    /// Record an error occurrence.
    pub fn record(&mut self, ctx: &ErrorContext) {
        let key = (ctx.error_code.raw_code(), ctx.kernel_name.clone());
        let now = Instant::now();
        if let Some(entry) = self.errors.get_mut(&key) {
            entry.count += 1;
            entry.last_seen = now;
        } else {
            // Evict oldest entry if at capacity.
            if self.errors.len() >= self.max_keys {
                self.evict_oldest();
            }
            self.errors.insert(
                key,
                AggregatedError {
                    code: ctx.error_code,
                    kernel_name: ctx.kernel_name.clone(),
                    count: 1,
                    first_seen: now,
                    last_seen: now,
                },
            );
        }
    }

    /// Number of distinct error keys currently tracked.
    pub fn distinct_count(&self) -> usize {
        self.errors.len()
    }

    /// Total error occurrences across all keys.
    pub fn total_count(&self) -> u64 {
        self.errors.values().map(|e| e.count).sum()
    }

    /// Get the most frequent error.
    pub fn most_frequent(&self) -> Option<&AggregatedError> {
        self.errors.values().max_by_key(|e| e.count)
    }

    /// Iterate over all aggregated errors.
    pub fn iter(&self) -> impl Iterator<Item = &AggregatedError> {
        self.errors.values()
    }

    /// Clear all recorded errors.
    pub fn clear(&mut self) {
        self.errors.clear();
    }

    /// Remove the entry with the oldest `last_seen`.
    fn evict_oldest(&mut self) {
        if let Some(oldest_key) =
            self.errors.iter().min_by_key(|(_, v)| v.last_seen).map(|(k, _)| k.clone())
        {
            self.errors.remove(&oldest_key);
        }
    }
}

// ---------------------------------------------------------------------------
// Error rate tracker with circuit breaker
// ---------------------------------------------------------------------------

/// Tracks error rates per [`ErrorCategory`] and triggers a circuit breaker
/// when the rate exceeds a configured threshold.
#[derive(Debug)]
pub struct ErrorRateTracker {
    inner: Arc<Mutex<RateTrackerInner>>,
}

#[derive(Debug)]
struct RateTrackerInner {
    /// Per-category sliding window of error timestamps.
    windows: HashMap<ErrorCategory, Vec<Instant>>,
    /// Window duration for rate calculation.
    window_duration: Duration,
    /// Maximum errors per window before tripping the breaker.
    threshold: u32,
    /// Categories currently tripped (breaker open).
    tripped: HashMap<ErrorCategory, Instant>,
    /// Cool-down period after breaker trips before allowing retry.
    cooldown: Duration,
}

/// State of the circuit breaker for a category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation — errors below threshold.
    Closed,
    /// Breaker tripped — blocking operations.
    Open,
    /// Cool-down elapsed — allowing a single probe.
    HalfOpen,
}

impl fmt::Display for CircuitState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => write!(f, "Closed"),
            Self::Open => write!(f, "Open"),
            Self::HalfOpen => write!(f, "HalfOpen"),
        }
    }
}

impl ErrorRateTracker {
    /// Create a new rate tracker.
    ///
    /// * `window` — sliding window duration for counting errors.
    /// * `threshold` — max errors in the window before tripping.
    /// * `cooldown` — time after trip before half-open probe.
    pub fn new(window: Duration, threshold: u32, cooldown: Duration) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RateTrackerInner {
                windows: HashMap::new(),
                window_duration: window,
                threshold,
                tripped: HashMap::new(),
                cooldown,
            })),
        }
    }

    /// Record an error in the given category and return the resulting
    /// [`CircuitState`].
    pub fn record(&self, category: ErrorCategory) -> CircuitState {
        let mut inner = self.inner.lock().unwrap();
        let now = Instant::now();
        let window_duration = inner.window_duration;
        let threshold = inner.threshold;
        let cooldown = inner.cooldown;

        // Append timestamp to the category window.
        let window = inner.windows.entry(category).or_default();
        window.push(now);

        // Prune timestamps outside the sliding window.
        let cutoff = now.checked_sub(window_duration).unwrap_or(now);
        window.retain(|&t| t >= cutoff);

        // Check threshold.
        if window.len() as u32 >= threshold {
            inner.tripped.insert(category, now);
            return CircuitState::Open;
        }

        // Check if previously tripped.
        if let Some(&trip_time) = inner.tripped.get(&category) {
            if now.duration_since(trip_time) >= cooldown {
                CircuitState::HalfOpen
            } else {
                CircuitState::Open
            }
        } else {
            CircuitState::Closed
        }
    }

    /// Query the circuit state for a category without recording an error.
    pub fn state(&self, category: ErrorCategory) -> CircuitState {
        let inner = self.inner.lock().unwrap();
        let now = Instant::now();
        if let Some(&trip_time) = inner.tripped.get(&category) {
            if now.duration_since(trip_time) >= inner.cooldown {
                CircuitState::HalfOpen
            } else {
                CircuitState::Open
            }
        } else {
            CircuitState::Closed
        }
    }

    /// Reset a tripped breaker (e.g. after a successful half-open probe).
    pub fn reset(&self, category: ErrorCategory) {
        let mut inner = self.inner.lock().unwrap();
        inner.tripped.remove(&category);
        inner.windows.remove(&category);
    }

    /// Current error count in the window for a category.
    pub fn current_count(&self, category: ErrorCategory) -> u32 {
        let mut inner = self.inner.lock().unwrap();
        let now = Instant::now();
        let window_duration = inner.window_duration;
        let window = inner.windows.entry(category).or_default();
        let cutoff = now.checked_sub(window_duration).unwrap_or(now);
        window.retain(|&t| t >= cutoff);
        window.len() as u32
    }
}

impl Clone for ErrorRateTracker {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

// ---------------------------------------------------------------------------
// Known CL error code constants (for property tests)
// ---------------------------------------------------------------------------

/// All known (non-Unknown) raw CL error codes.
pub const ALL_KNOWN_CL_CODES: &[i32] = &[
    0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -30,
    -31, -32, -33, -34, -35, -36, -37, -38, -39, -40, -41, -42, -43, -44, -45, -46, -47, -48, -49,
    -50, -51, -52, -53, -54, -55, -56, -57, -58, -59, -60, -61, -62, -63, -64, -65, -66, -67, -68,
    -69, -70,
];

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // ----- Error code mapping completeness -----

    #[test]
    fn test_all_known_codes_round_trip() {
        for &raw in ALL_KNOWN_CL_CODES {
            let code = OpenClErrorCode::from_raw(raw);
            assert_eq!(code.raw_code(), raw, "round-trip failed for {raw}");
        }
    }

    #[test]
    fn test_all_known_codes_are_not_unknown() {
        for &raw in ALL_KNOWN_CL_CODES {
            let code = OpenClErrorCode::from_raw(raw);
            assert!(!matches!(code, OpenClErrorCode::Unknown(_)), "code {raw} mapped to Unknown");
        }
    }

    #[test]
    fn test_unknown_code_preserved() {
        let code = OpenClErrorCode::from_raw(-9999);
        assert_eq!(code, OpenClErrorCode::Unknown(-9999));
        assert_eq!(code.raw_code(), -9999);
    }

    #[test]
    fn test_positive_unknown_code() {
        let code = OpenClErrorCode::from_raw(42);
        assert_eq!(code, OpenClErrorCode::Unknown(42));
    }

    #[test]
    fn test_gap_codes_are_unknown() {
        // Codes -20 through -29 are not defined in standard OpenCL.
        for raw in -29..=-20 {
            let code = OpenClErrorCode::from_raw(raw);
            assert!(matches!(code, OpenClErrorCode::Unknown(_)), "code {raw} should be Unknown");
        }
    }

    #[test]
    fn test_cl_name_for_known_codes() {
        assert_eq!(OpenClErrorCode::DeviceNotFound.cl_name(), "CL_DEVICE_NOT_FOUND");
        assert_eq!(OpenClErrorCode::OutOfResources.cl_name(), "CL_OUT_OF_RESOURCES");
        assert_eq!(OpenClErrorCode::InvalidKernel.cl_name(), "CL_INVALID_KERNEL");
    }

    #[test]
    fn test_cl_name_for_unknown() {
        assert_eq!(OpenClErrorCode::Unknown(-999).cl_name(), "CL_UNKNOWN");
    }

    #[test]
    fn test_display_format() {
        let code = OpenClErrorCode::OutOfHostMemory;
        let s = format!("{code}");
        assert!(s.contains("CL_OUT_OF_HOST_MEMORY"));
        assert!(s.contains("-6"));
    }

    #[test]
    fn test_known_code_count() {
        // 1 success + 19 runtime/memory codes (-1…-19) + 41 invalid codes (-30…-70) = 61
        assert_eq!(ALL_KNOWN_CL_CODES.len(), 61);
    }

    // ----- Category classification -----

    #[test]
    fn test_device_errors_classified() {
        let device_codes = [
            OpenClErrorCode::DeviceNotFound,
            OpenClErrorCode::DeviceNotAvailable,
            OpenClErrorCode::DevicePartitionFailed,
            OpenClErrorCode::InvalidDevice,
            OpenClErrorCode::InvalidPlatform,
        ];
        for code in &device_codes {
            assert_eq!(code.category(), ErrorCategory::Device, "{code:?}");
        }
    }

    #[test]
    fn test_memory_errors_classified() {
        let mem_codes = [
            OpenClErrorCode::MemObjectAllocationFailure,
            OpenClErrorCode::OutOfResources,
            OpenClErrorCode::OutOfHostMemory,
            OpenClErrorCode::MemCopyOverlap,
            OpenClErrorCode::MapFailure,
            OpenClErrorCode::InvalidBufferSize,
        ];
        for code in &mem_codes {
            assert_eq!(code.category(), ErrorCategory::Memory, "{code:?}");
        }
    }

    #[test]
    fn test_compilation_errors_classified() {
        let comp_codes = [
            OpenClErrorCode::CompilerNotAvailable,
            OpenClErrorCode::BuildProgramFailure,
            OpenClErrorCode::CompileProgramFailure,
            OpenClErrorCode::LinkProgramFailure,
            OpenClErrorCode::InvalidBinary,
            OpenClErrorCode::InvalidProgram,
        ];
        for code in &comp_codes {
            assert_eq!(code.category(), ErrorCategory::Compilation, "{code:?}");
        }
    }

    #[test]
    fn test_validation_errors_classified() {
        let val_codes = [
            OpenClErrorCode::InvalidValue,
            OpenClErrorCode::InvalidKernelArgs,
            OpenClErrorCode::InvalidWorkGroupSize,
            OpenClErrorCode::InvalidArgSize,
            OpenClErrorCode::InvalidGlobalWorkSize,
        ];
        for code in &val_codes {
            assert_eq!(code.category(), ErrorCategory::Validation, "{code:?}");
        }
    }

    #[test]
    fn test_runtime_errors_classified() {
        let rt_codes = [
            OpenClErrorCode::ProfilingInfoNotAvailable,
            OpenClErrorCode::ImageFormatMismatch,
            OpenClErrorCode::ExecStatusErrorForEventsInWaitList,
        ];
        for code in &rt_codes {
            assert_eq!(code.category(), ErrorCategory::Runtime, "{code:?}");
        }
    }

    #[test]
    fn test_unknown_code_category_is_runtime() {
        assert_eq!(OpenClErrorCode::Unknown(-1234).category(), ErrorCategory::Runtime);
    }

    #[test]
    fn test_success_category_is_runtime() {
        assert_eq!(OpenClErrorCode::Success.category(), ErrorCategory::Runtime);
    }

    #[test]
    fn test_every_known_code_has_a_category() {
        for &raw in ALL_KNOWN_CL_CODES {
            let code = OpenClErrorCode::from_raw(raw);
            // Should not panic — just ensure it returns a valid category.
            let _ = code.category();
        }
    }

    // ----- Severity -----

    #[test]
    fn test_transient_severity() {
        assert_eq!(OpenClErrorCode::OutOfResources.severity(), ErrorSeverity::Transient);
        assert_eq!(OpenClErrorCode::OutOfHostMemory.severity(), ErrorSeverity::Transient);
    }

    #[test]
    fn test_fatal_severity() {
        assert_eq!(OpenClErrorCode::DeviceNotFound.severity(), ErrorSeverity::Fatal);
        assert_eq!(OpenClErrorCode::CompilerNotAvailable.severity(), ErrorSeverity::Fatal);
    }

    #[test]
    fn test_warning_severity() {
        assert_eq!(OpenClErrorCode::ProfilingInfoNotAvailable.severity(), ErrorSeverity::Warning);
    }

    #[test]
    fn test_error_severity_default() {
        assert_eq!(OpenClErrorCode::InvalidKernelArgs.severity(), ErrorSeverity::Error);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(ErrorSeverity::Transient < ErrorSeverity::Warning);
        assert!(ErrorSeverity::Warning < ErrorSeverity::Error);
        assert!(ErrorSeverity::Error < ErrorSeverity::Fatal);
    }

    // ----- Recovery strategy -----

    #[test]
    fn test_memory_transient_recovery_is_retry() {
        assert_eq!(OpenClErrorCode::OutOfResources.recovery(), ErrorRecovery::Retry);
    }

    #[test]
    fn test_memory_non_transient_recovery_is_fallback() {
        assert_eq!(OpenClErrorCode::InvalidBufferSize.recovery(), ErrorRecovery::FallbackCpu);
    }

    #[test]
    fn test_compilation_recovery_is_recompile() {
        assert_eq!(OpenClErrorCode::BuildProgramFailure.recovery(), ErrorRecovery::Recompile);
    }

    #[test]
    fn test_device_not_found_recovery_is_fallback() {
        assert_eq!(OpenClErrorCode::DeviceNotFound.recovery(), ErrorRecovery::FallbackCpu);
    }

    #[test]
    fn test_device_partition_recovery_is_reinit() {
        assert_eq!(OpenClErrorCode::DevicePartitionFailed.recovery(), ErrorRecovery::ReinitContext);
    }

    #[test]
    fn test_validation_recovery_is_abort() {
        assert_eq!(OpenClErrorCode::InvalidKernelArgs.recovery(), ErrorRecovery::Abort);
    }

    #[test]
    fn test_every_known_code_has_recovery() {
        for &raw in ALL_KNOWN_CL_CODES {
            let code = OpenClErrorCode::from_raw(raw);
            let _ = code.recovery();
        }
    }

    // ----- ErrorContext -----

    #[test]
    fn test_context_builder_chain() {
        let ctx = ErrorContext::new(OpenClErrorCode::OutOfResources)
            .with_kernel("matmul_i2s")
            .with_device("Arc A770")
            .with_buffer_size(1024 * 1024)
            .with_call_site("dispatch_kernel")
            .with_meta("batch", "16");

        assert_eq!(ctx.kernel_name.as_deref(), Some("matmul_i2s"));
        assert_eq!(ctx.device_name.as_deref(), Some("Arc A770"));
        assert_eq!(ctx.buffer_size, Some(1024 * 1024));
        assert_eq!(ctx.call_stack.len(), 1);
        assert_eq!(ctx.metadata.get("batch").map(|s| s.as_str()), Some("16"));
    }

    #[test]
    fn test_context_display() {
        let ctx = ErrorContext::new(OpenClErrorCode::MapFailure).with_kernel("softmax");
        let s = format!("{ctx}");
        assert!(s.contains("CL_MAP_FAILURE"));
        assert!(s.contains("softmax"));
    }

    #[test]
    fn test_context_minimal() {
        let ctx = ErrorContext::new(OpenClErrorCode::Success);
        assert!(ctx.kernel_name.is_none());
        assert!(ctx.device_name.is_none());
        assert!(ctx.buffer_size.is_none());
        assert!(ctx.call_stack.is_empty());
    }

    #[test]
    fn test_context_multiple_call_sites() {
        let ctx = ErrorContext::new(OpenClErrorCode::InvalidKernel)
            .with_call_site("layer_1")
            .with_call_site("dispatch")
            .with_call_site("run_model");
        assert_eq!(ctx.call_stack.len(), 3);
        assert_eq!(ctx.call_stack[0], "layer_1");
        assert_eq!(ctx.call_stack[2], "run_model");
    }

    // ----- DiagnosticInfo -----

    #[test]
    fn test_diagnostic_capture() {
        let diag = DiagnosticInfo::capture(10, 5, 256 * 1024 * 1024);
        assert!(diag.total_host_memory_bytes > 0);
        assert_eq!(diag.queue_depth, 10);
        assert_eq!(diag.live_buffers, 5);
    }

    #[test]
    fn test_diagnostic_with_driver() {
        let diag = DiagnosticInfo::capture(0, 0, 0).with_driver("31.0.101.5768");
        assert_eq!(diag.driver_version.as_deref(), Some("31.0.101.5768"));
    }

    #[test]
    fn test_diagnostic_utilisation_zero_total() {
        let diag = DiagnosticInfo {
            free_host_memory_bytes: 0,
            total_host_memory_bytes: 0,
            queue_depth: 0,
            live_buffers: 0,
            allocated_buffer_bytes: 0,
            driver_version: None,
        };
        assert_eq!(diag.host_memory_utilisation(), 0.0);
    }

    #[test]
    fn test_diagnostic_utilisation_half() {
        let diag = DiagnosticInfo {
            free_host_memory_bytes: 500,
            total_host_memory_bytes: 1000,
            queue_depth: 0,
            live_buffers: 0,
            allocated_buffer_bytes: 0,
            driver_version: None,
        };
        let util = diag.host_memory_utilisation();
        assert!((util - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_diagnostic_display() {
        let diag = DiagnosticInfo::capture(3, 2, 128 * 1024 * 1024);
        let s = format!("{diag}");
        assert!(s.contains("queue=3"));
        assert!(s.contains("bufs=2"));
    }

    // ----- ErrorReport -----

    #[test]
    fn test_report_from_memory_error() {
        let ctx = ErrorContext::new(OpenClErrorCode::OutOfResources)
            .with_kernel("matmul")
            .with_buffer_size(1_000_000);
        let report = ErrorReport::from_context(ctx);
        assert_eq!(report.category, ErrorCategory::Memory);
        assert_eq!(report.severity, ErrorSeverity::Transient);
        assert_eq!(report.recovery, ErrorRecovery::Retry);
        assert!(report.root_cause.contains("1000000"));
    }

    #[test]
    fn test_report_from_compilation_error() {
        let ctx = ErrorContext::new(OpenClErrorCode::BuildProgramFailure).with_kernel("layer_norm");
        let report = ErrorReport::from_context(ctx);
        assert_eq!(report.category, ErrorCategory::Compilation);
        assert!(report.root_cause.contains("layer_norm"));
    }

    #[test]
    fn test_report_display_contains_sections() {
        let ctx = ErrorContext::new(OpenClErrorCode::InvalidWorkGroupSize)
            .with_kernel("rms_norm")
            .with_device("Arc A770");
        let report = ErrorReport::from_context(ctx);
        let s = format!("{report}");
        assert!(s.contains("=== OpenCL Error Report ==="));
        assert!(s.contains("Validation"));
        assert!(s.contains("rms_norm"));
        assert!(s.contains("Arc A770"));
        assert!(s.contains("Remediation"));
    }

    #[test]
    fn test_report_with_diagnostics() {
        let ctx = ErrorContext::new(OpenClErrorCode::OutOfHostMemory);
        let diag = DiagnosticInfo::capture(0, 10, 2 * 1024 * 1024 * 1024);
        let report = ErrorReport::from_context(ctx).with_diagnostics(diag);
        assert!(report.diagnostics.is_some());
        let s = format!("{report}");
        assert!(s.contains("Diagnostics"));
    }

    #[test]
    fn test_report_large_buffer_remediation() {
        let ctx = ErrorContext::new(OpenClErrorCode::MemObjectAllocationFailure)
            .with_buffer_size(1024 * 1024 * 1024); // 1 GB
        let report = ErrorReport::from_context(ctx);
        assert!(
            report.remediation.iter().any(|s| s.contains("tiled")),
            "should suggest tiled execution for large buffers"
        );
    }

    #[test]
    fn test_report_device_error_remediation() {
        let ctx = ErrorContext::new(OpenClErrorCode::DeviceNotFound).with_device("missing_gpu");
        let report = ErrorReport::from_context(ctx);
        assert!(report.remediation.iter().any(|s| s.contains("driver")));
    }

    #[test]
    fn test_report_call_stack_displayed() {
        let ctx = ErrorContext::new(OpenClErrorCode::InvalidKernel)
            .with_call_site("frame0")
            .with_call_site("frame1");
        let report = ErrorReport::from_context(ctx);
        let s = format!("{report}");
        assert!(s.contains("Call stack"));
        assert!(s.contains("frame0"));
        assert!(s.contains("frame1"));
    }

    // ----- ErrorAggregator -----

    #[test]
    fn test_aggregator_basic_record() {
        let mut agg = ErrorAggregator::new(100);
        let ctx = ErrorContext::new(OpenClErrorCode::OutOfResources).with_kernel("matmul");
        agg.record(&ctx);
        assert_eq!(agg.distinct_count(), 1);
        assert_eq!(agg.total_count(), 1);
    }

    #[test]
    fn test_aggregator_deduplicates() {
        let mut agg = ErrorAggregator::new(100);
        let ctx = ErrorContext::new(OpenClErrorCode::OutOfResources).with_kernel("matmul");
        for _ in 0..10 {
            agg.record(&ctx);
        }
        assert_eq!(agg.distinct_count(), 1);
        assert_eq!(agg.total_count(), 10);
    }

    #[test]
    fn test_aggregator_different_kernels_are_separate() {
        let mut agg = ErrorAggregator::new(100);
        let ctx1 = ErrorContext::new(OpenClErrorCode::OutOfResources).with_kernel("matmul");
        let ctx2 = ErrorContext::new(OpenClErrorCode::OutOfResources).with_kernel("softmax");
        agg.record(&ctx1);
        agg.record(&ctx2);
        assert_eq!(agg.distinct_count(), 2);
        assert_eq!(agg.total_count(), 2);
    }

    #[test]
    fn test_aggregator_different_codes_are_separate() {
        let mut agg = ErrorAggregator::new(100);
        let ctx1 = ErrorContext::new(OpenClErrorCode::OutOfResources);
        let ctx2 = ErrorContext::new(OpenClErrorCode::OutOfHostMemory);
        agg.record(&ctx1);
        agg.record(&ctx2);
        assert_eq!(agg.distinct_count(), 2);
    }

    #[test]
    fn test_aggregator_most_frequent() {
        let mut agg = ErrorAggregator::new(100);
        let ctx_a = ErrorContext::new(OpenClErrorCode::OutOfResources);
        let ctx_b = ErrorContext::new(OpenClErrorCode::MapFailure);
        agg.record(&ctx_a);
        agg.record(&ctx_a);
        agg.record(&ctx_a);
        agg.record(&ctx_b);
        let top = agg.most_frequent().unwrap();
        assert_eq!(top.code, OpenClErrorCode::OutOfResources);
        assert_eq!(top.count, 3);
    }

    #[test]
    fn test_aggregator_clear() {
        let mut agg = ErrorAggregator::new(100);
        agg.record(&ErrorContext::new(OpenClErrorCode::OutOfResources));
        agg.clear();
        assert_eq!(agg.distinct_count(), 0);
        assert_eq!(agg.total_count(), 0);
    }

    #[test]
    fn test_aggregator_eviction() {
        let mut agg = ErrorAggregator::new(2);
        let ctx1 = ErrorContext::new(OpenClErrorCode::from_raw(-1));
        let ctx2 = ErrorContext::new(OpenClErrorCode::from_raw(-2));
        let ctx3 = ErrorContext::new(OpenClErrorCode::from_raw(-3));
        agg.record(&ctx1);
        agg.record(&ctx2);
        agg.record(&ctx3); // Should evict oldest.
        assert_eq!(agg.distinct_count(), 2);
    }

    #[test]
    fn test_aggregator_most_frequent_empty() {
        let agg = ErrorAggregator::new(10);
        assert!(agg.most_frequent().is_none());
    }

    #[test]
    fn test_aggregator_iter() {
        let mut agg = ErrorAggregator::new(10);
        agg.record(&ErrorContext::new(OpenClErrorCode::OutOfResources));
        agg.record(&ErrorContext::new(OpenClErrorCode::MapFailure));
        let codes: Vec<_> = agg.iter().map(|e| e.code).collect();
        assert_eq!(codes.len(), 2);
    }

    // ----- ErrorRateTracker / circuit breaker -----

    #[test]
    fn test_rate_tracker_closed_initially() {
        let tracker = ErrorRateTracker::new(Duration::from_mins(1), 5, Duration::from_secs(30));
        assert_eq!(tracker.state(ErrorCategory::Memory), CircuitState::Closed);
    }

    #[test]
    fn test_rate_tracker_stays_closed_below_threshold() {
        let tracker = ErrorRateTracker::new(Duration::from_mins(1), 5, Duration::from_secs(30));
        for _ in 0..4 {
            let state = tracker.record(ErrorCategory::Memory);
            assert_eq!(state, CircuitState::Closed);
        }
    }

    #[test]
    fn test_rate_tracker_opens_at_threshold() {
        let tracker = ErrorRateTracker::new(Duration::from_mins(1), 3, Duration::from_secs(30));
        tracker.record(ErrorCategory::Runtime);
        tracker.record(ErrorCategory::Runtime);
        let state = tracker.record(ErrorCategory::Runtime);
        assert_eq!(state, CircuitState::Open);
    }

    #[test]
    fn test_rate_tracker_categories_independent() {
        let tracker = ErrorRateTracker::new(Duration::from_mins(1), 2, Duration::from_secs(30));
        tracker.record(ErrorCategory::Memory);
        let state = tracker.record(ErrorCategory::Runtime);
        // Runtime has only 1 error — should still be closed.
        assert_eq!(state, CircuitState::Closed);
    }

    #[test]
    fn test_rate_tracker_reset() {
        let tracker = ErrorRateTracker::new(Duration::from_mins(1), 2, Duration::from_secs(30));
        tracker.record(ErrorCategory::Memory);
        tracker.record(ErrorCategory::Memory);
        assert_eq!(tracker.state(ErrorCategory::Memory), CircuitState::Open);
        tracker.reset(ErrorCategory::Memory);
        assert_eq!(tracker.state(ErrorCategory::Memory), CircuitState::Closed);
    }

    #[test]
    fn test_rate_tracker_current_count() {
        let tracker = ErrorRateTracker::new(Duration::from_mins(1), 10, Duration::from_secs(5));
        tracker.record(ErrorCategory::Compilation);
        tracker.record(ErrorCategory::Compilation);
        assert_eq!(tracker.current_count(ErrorCategory::Compilation), 2);
        assert_eq!(tracker.current_count(ErrorCategory::Device), 0);
    }

    #[test]
    fn test_rate_tracker_clone_shares_state() {
        let tracker = ErrorRateTracker::new(Duration::from_mins(1), 5, Duration::from_secs(5));
        let clone = tracker.clone();
        tracker.record(ErrorCategory::Memory);
        assert_eq!(clone.current_count(ErrorCategory::Memory), 1);
    }

    #[test]
    fn test_rate_tracker_rapid_burst_trips() {
        let tracker = ErrorRateTracker::new(Duration::from_secs(1), 10, Duration::from_secs(30));
        for _ in 0..9 {
            tracker.record(ErrorCategory::Runtime);
        }
        let state = tracker.record(ErrorCategory::Runtime);
        assert_eq!(state, CircuitState::Open);
    }

    #[test]
    fn test_rate_tracker_half_open_after_cooldown() {
        let tracker = ErrorRateTracker::new(Duration::from_mins(1), 2, Duration::from_millis(50));
        tracker.record(ErrorCategory::Memory);
        tracker.record(ErrorCategory::Memory);
        assert_eq!(tracker.state(ErrorCategory::Memory), CircuitState::Open);
        thread::sleep(Duration::from_millis(80));
        assert_eq!(tracker.state(ErrorCategory::Memory), CircuitState::HalfOpen);
    }

    // ----- Property-style tests -----

    #[test]
    fn test_all_known_codes_have_category_and_recovery() {
        for &raw in ALL_KNOWN_CL_CODES {
            let code = OpenClErrorCode::from_raw(raw);
            let cat = code.category();
            let sev = code.severity();
            let rec = code.recovery();
            // Ensure the combination is self-consistent.
            match cat {
                ErrorCategory::Memory => {
                    assert!(
                        matches!(rec, ErrorRecovery::Retry | ErrorRecovery::FallbackCpu),
                        "memory code {raw} has unexpected recovery {rec:?}"
                    );
                }
                ErrorCategory::Compilation => {
                    assert_eq!(rec, ErrorRecovery::Recompile);
                }
                ErrorCategory::Device => {
                    assert!(matches!(
                        rec,
                        ErrorRecovery::FallbackCpu | ErrorRecovery::ReinitContext
                    ));
                }
                ErrorCategory::Validation => {
                    assert_eq!(rec, ErrorRecovery::Abort);
                }
                _ => {
                    // Runtime / Timeout — any recovery is fine.
                    let _ = sev;
                }
            }
        }
    }

    #[test]
    fn test_category_display_all_variants() {
        let cats = [
            ErrorCategory::Runtime,
            ErrorCategory::Memory,
            ErrorCategory::Compilation,
            ErrorCategory::Device,
            ErrorCategory::Validation,
            ErrorCategory::Timeout,
        ];
        for cat in &cats {
            let s = format!("{cat}");
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_severity_display_all_variants() {
        let sevs = [
            ErrorSeverity::Transient,
            ErrorSeverity::Warning,
            ErrorSeverity::Error,
            ErrorSeverity::Fatal,
        ];
        for s in &sevs {
            let d = format!("{s}");
            assert!(!d.is_empty());
        }
    }

    #[test]
    fn test_recovery_display_all_variants() {
        let recs = [
            ErrorRecovery::Retry,
            ErrorRecovery::FallbackCpu,
            ErrorRecovery::Abort,
            ErrorRecovery::Recompile,
            ErrorRecovery::ReinitContext,
        ];
        for r in &recs {
            let d = format!("{r}");
            assert!(!d.is_empty());
        }
    }

    #[test]
    fn test_circuit_state_display() {
        assert_eq!(format!("{}", CircuitState::Closed), "Closed");
        assert_eq!(format!("{}", CircuitState::Open), "Open");
        assert_eq!(format!("{}", CircuitState::HalfOpen), "HalfOpen");
    }

    // ----- Edge cases -----

    #[test]
    fn test_i32_min_unknown() {
        let code = OpenClErrorCode::from_raw(i32::MIN);
        assert!(matches!(code, OpenClErrorCode::Unknown(i32::MIN)));
    }

    #[test]
    fn test_i32_max_unknown() {
        let code = OpenClErrorCode::from_raw(i32::MAX);
        assert!(matches!(code, OpenClErrorCode::Unknown(i32::MAX)));
    }

    #[test]
    fn test_report_from_unknown_code() {
        let ctx = ErrorContext::new(OpenClErrorCode::Unknown(-5000));
        let report = ErrorReport::from_context(ctx);
        assert_eq!(report.category, ErrorCategory::Runtime);
    }

    #[test]
    fn test_aggregator_no_kernel_name() {
        let mut agg = ErrorAggregator::new(10);
        let ctx = ErrorContext::new(OpenClErrorCode::OutOfResources);
        agg.record(&ctx);
        agg.record(&ctx);
        assert_eq!(agg.distinct_count(), 1);
        assert_eq!(agg.total_count(), 2);
    }

    #[test]
    fn test_error_code_eq_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(OpenClErrorCode::InvalidKernel);
        set.insert(OpenClErrorCode::InvalidKernel);
        set.insert(OpenClErrorCode::OutOfResources);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_error_context_multiple_meta() {
        let ctx = ErrorContext::new(OpenClErrorCode::Success)
            .with_meta("key1", "val1")
            .with_meta("key2", "val2");
        assert_eq!(ctx.metadata.len(), 2);
    }

    #[test]
    fn test_aggregator_timestamp_ordering() {
        let mut agg = ErrorAggregator::new(100);
        let ctx = ErrorContext::new(OpenClErrorCode::OutOfResources);
        agg.record(&ctx);
        thread::sleep(Duration::from_millis(10));
        agg.record(&ctx);
        let entry = agg.most_frequent().unwrap();
        assert!(entry.last_seen >= entry.first_seen);
    }
}
