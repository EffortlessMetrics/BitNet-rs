//! Structured error taxonomy for GPU HAL operations.
//!
//! Provides a layered error hierarchy for GPU hardware abstraction:
//! - [`GpuHalError`] — top-level error enum covering all GPU HAL errors
//! - [`DeviceError`] — device initialization, capability, and driver errors
//! - [`MemoryError`] — allocation failures, OOM, corruption, fragmentation
//! - [`KernelError`] — compilation, launch, timeout, invalid arguments
//! - [`TransferError`] — host↔device transfer failures
//! - [`BackendError`] — backend-specific errors (CUDA, OpenCL, Vulkan, etc.)
//! - [`ErrorContext`] — rich error context with device info and operation
//! - [`ErrorRecovery`] — recovery strategies (retry, fallback, abort, degrade)
//! - [`ErrorReporter`] — structured error reporting for telemetry/logging
//! - [`ErrorEngine`] — central error handling with recovery and reporting

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ── DeviceError ─────────────────────────────────────────────────────────────

/// Errors related to GPU device initialization and capability queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceError {
    /// No compatible device found on the system.
    NotFound { query: String },
    /// Device exists but is unavailable (busy, in exclusive mode, etc.).
    Unavailable { device_id: u32, reason: String },
    /// Driver version mismatch or missing driver.
    DriverError { expected: String, found: String },
    /// Device does not support a required capability.
    CapabilityMissing { device_id: u32, capability: String },
    /// Device initialization failed.
    InitFailed { device_id: u32, message: String },
    /// Device was lost or reset during operation.
    DeviceLost { device_id: u32 },
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound { query } => {
                write!(f, "no compatible device found for query: {query}")
            }
            Self::Unavailable { device_id, reason } => {
                write!(f, "device {device_id} unavailable: {reason}")
            }
            Self::DriverError { expected, found } => {
                write!(f, "driver mismatch: expected {expected}, found {found}")
            }
            Self::CapabilityMissing { device_id, capability } => {
                write!(f, "device {device_id} missing capability: {capability}")
            }
            Self::InitFailed { device_id, message } => {
                write!(f, "device {device_id} init failed: {message}")
            }
            Self::DeviceLost { device_id } => {
                write!(f, "device {device_id} lost")
            }
        }
    }
}

impl std::error::Error for DeviceError {}

// ── MemoryError ─────────────────────────────────────────────────────────────

/// Errors related to GPU memory operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    /// Allocation failed (out of memory).
    OutOfMemory { requested: usize, available: usize },
    /// Buffer access out of bounds.
    OutOfBounds { offset: usize, length: usize, buffer_size: usize },
    /// Memory corruption detected (checksum mismatch, etc.).
    Corruption { buffer_id: u64, details: String },
    /// Memory fragmentation prevents allocation despite sufficient free memory.
    Fragmentation { requested: usize, total_free: usize, largest_block: usize },
    /// Attempted to use a freed or invalid buffer.
    InvalidBuffer { buffer_id: u64 },
    /// Host-device mapping failed.
    MappingFailed { reason: String },
    /// Alignment requirement not met.
    AlignmentError { required: usize, actual: usize },
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory { requested, available } => {
                write!(
                    f,
                    "out of memory: requested {requested} B, \
                     available {available} B"
                )
            }
            Self::OutOfBounds { offset, length, buffer_size } => {
                write!(
                    f,
                    "out of bounds: offset {offset} + length {length} \
                     exceeds buffer size {buffer_size}"
                )
            }
            Self::Corruption { buffer_id, details } => {
                write!(f, "memory corruption in buffer {buffer_id}: {details}")
            }
            Self::Fragmentation { requested, total_free, largest_block } => {
                write!(
                    f,
                    "fragmentation: need {requested} B, {total_free} B free \
                     but largest block is {largest_block} B"
                )
            }
            Self::InvalidBuffer { buffer_id } => {
                write!(f, "invalid buffer: {buffer_id}")
            }
            Self::MappingFailed { reason } => {
                write!(f, "memory mapping failed: {reason}")
            }
            Self::AlignmentError { required, actual } => {
                write!(f, "alignment error: required {required}, actual {actual}")
            }
        }
    }
}

impl std::error::Error for MemoryError {}

// ── KernelError ─────────────────────────────────────────────────────────────

/// Errors related to GPU kernel compilation and execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    /// Kernel source failed to compile.
    CompilationFailed { kernel_name: String, log: String },
    /// Kernel launch configuration is invalid.
    LaunchConfigInvalid { kernel_name: String, reason: String },
    /// Kernel execution timed out.
    Timeout { kernel_name: String, elapsed_ms: u64, limit_ms: u64 },
    /// Invalid kernel argument at the given index.
    InvalidArgument { kernel_name: String, index: usize, reason: String },
    /// Kernel not found in the compiled program.
    NotFound { kernel_name: String },
    /// Kernel execution produced an error on device.
    ExecutionFailed { kernel_name: String, error_code: i64 },
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CompilationFailed { kernel_name, log } => {
                write!(f, "kernel '{kernel_name}' compilation failed: {log}")
            }
            Self::LaunchConfigInvalid { kernel_name, reason } => {
                write!(f, "kernel '{kernel_name}' invalid launch config: {reason}")
            }
            Self::Timeout { kernel_name, elapsed_ms, limit_ms } => {
                write!(
                    f,
                    "kernel '{kernel_name}' timed out after {elapsed_ms} ms \
                     (limit: {limit_ms} ms)"
                )
            }
            Self::InvalidArgument { kernel_name, index, reason } => {
                write!(
                    f,
                    "kernel '{kernel_name}' invalid argument at index \
                     {index}: {reason}"
                )
            }
            Self::NotFound { kernel_name } => {
                write!(f, "kernel '{kernel_name}' not found")
            }
            Self::ExecutionFailed { kernel_name, error_code } => {
                write!(
                    f,
                    "kernel '{kernel_name}' execution failed with code \
                     {error_code}"
                )
            }
        }
    }
}

impl std::error::Error for KernelError {}

// ── TransferError ───────────────────────────────────────────────────────────

/// Direction of a host↔device transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

impl fmt::Display for TransferDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostToDevice => write!(f, "host→device"),
            Self::DeviceToHost => write!(f, "device→host"),
            Self::DeviceToDevice => write!(f, "device→device"),
        }
    }
}

/// Errors related to data transfers between host and device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferError {
    /// Transfer of the given size failed.
    Failed { direction: TransferDirection, size_bytes: usize, reason: String },
    /// Transfer timed out.
    Timeout { direction: TransferDirection, elapsed_ms: u64 },
    /// Source or destination buffer is invalid.
    InvalidBuffer { direction: TransferDirection, buffer_id: u64 },
    /// Size mismatch between source and destination.
    SizeMismatch { source_size: usize, dest_size: usize },
    /// DMA channel error.
    DmaError { channel: u32, message: String },
}

impl fmt::Display for TransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Failed { direction, size_bytes, reason } => {
                write!(f, "{direction} transfer of {size_bytes} B failed: {reason}")
            }
            Self::Timeout { direction, elapsed_ms } => {
                write!(f, "{direction} transfer timed out after {elapsed_ms} ms")
            }
            Self::InvalidBuffer { direction, buffer_id } => {
                write!(f, "{direction} transfer: invalid buffer {buffer_id}")
            }
            Self::SizeMismatch { source_size, dest_size } => {
                write!(
                    f,
                    "transfer size mismatch: source {source_size} B, \
                     dest {dest_size} B"
                )
            }
            Self::DmaError { channel, message } => {
                write!(f, "DMA channel {channel} error: {message}")
            }
        }
    }
}

impl std::error::Error for TransferError {}

// ── BackendError ────────────────────────────────────────────────────────────

/// Backend kind for backend-specific error categorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    CUDA,
    OpenCL,
    Vulkan,
    Metal,
    ROCm,
    LevelZero,
    WebGPU,
    Other,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CUDA => write!(f, "CUDA"),
            Self::OpenCL => write!(f, "OpenCL"),
            Self::Vulkan => write!(f, "Vulkan"),
            Self::Metal => write!(f, "Metal"),
            Self::ROCm => write!(f, "ROCm"),
            Self::LevelZero => write!(f, "LevelZero"),
            Self::WebGPU => write!(f, "WebGPU"),
            Self::Other => write!(f, "Other"),
        }
    }
}

/// Backend-specific errors with native error codes and messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendError {
    /// Which backend produced the error.
    pub kind: BackendKind,
    /// Native error code from the backend API (-1 if unavailable).
    pub native_code: i64,
    /// Human-readable error message.
    pub message: String,
    /// Optional backend API function that failed.
    pub api_call: Option<String>,
}

impl BackendError {
    pub fn new(kind: BackendKind, native_code: i64, message: impl Into<String>) -> Self {
        Self { kind, native_code, message: message.into(), api_call: None }
    }

    pub fn with_api_call(mut self, call: impl Into<String>) -> Self {
        self.api_call = Some(call.into());
        self
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] ", self.kind)?;
        if let Some(ref api) = self.api_call {
            write!(f, "{api}: ")?;
        }
        write!(f, "{} (code {})", self.message, self.native_code)
    }
}

impl std::error::Error for BackendError {}

// ── GpuHalError (top-level) ─────────────────────────────────────────────────

/// Top-level error enum covering all GPU HAL error categories.
#[derive(Debug)]
pub enum GpuHalError {
    Device(DeviceError),
    Memory(MemoryError),
    Kernel(KernelError),
    Transfer(TransferError),
    Backend(BackendError),
    /// An error with attached rich context.
    Contextualized {
        error: Box<GpuHalError>,
        context: ErrorContext,
    },
    /// Catch-all for errors that don't fit other categories.
    Other(String),
}

impl fmt::Display for GpuHalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device(e) => write!(f, "device error: {e}"),
            Self::Memory(e) => write!(f, "memory error: {e}"),
            Self::Kernel(e) => write!(f, "kernel error: {e}"),
            Self::Transfer(e) => write!(f, "transfer error: {e}"),
            Self::Backend(e) => write!(f, "backend error: {e}"),
            Self::Contextualized { error, context } => {
                write!(f, "{error} [context: {context}]")
            }
            Self::Other(msg) => write!(f, "gpu hal error: {msg}"),
        }
    }
}

impl std::error::Error for GpuHalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Device(e) => Some(e),
            Self::Memory(e) => Some(e),
            Self::Kernel(e) => Some(e),
            Self::Transfer(e) => Some(e),
            Self::Backend(e) => Some(e),
            Self::Contextualized { error, .. } => Some(error.as_ref()),
            Self::Other(_) => None,
        }
    }
}

impl GpuHalError {
    /// Attach rich context to this error.
    pub fn with_context(self, context: ErrorContext) -> Self {
        Self::Contextualized { error: Box::new(self), context }
    }

    /// Returns the error category as a static string.
    pub fn category(&self) -> &'static str {
        match self {
            Self::Device(_) => "device",
            Self::Memory(_) => "memory",
            Self::Kernel(_) => "kernel",
            Self::Transfer(_) => "transfer",
            Self::Backend(_) => "backend",
            Self::Contextualized { error, .. } => error.category(),
            Self::Other(_) => "other",
        }
    }

    /// Returns true if this error is likely transient and retryable.
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            Self::Memory(MemoryError::OutOfMemory { .. })
                | Self::Kernel(KernelError::Timeout { .. })
                | Self::Transfer(TransferError::Timeout { .. })
                | Self::Device(DeviceError::Unavailable { .. })
        )
    }
}

// From conversions for ergonomic `?` usage
impl From<DeviceError> for GpuHalError {
    fn from(e: DeviceError) -> Self {
        Self::Device(e)
    }
}

impl From<MemoryError> for GpuHalError {
    fn from(e: MemoryError) -> Self {
        Self::Memory(e)
    }
}

impl From<KernelError> for GpuHalError {
    fn from(e: KernelError) -> Self {
        Self::Kernel(e)
    }
}

impl From<TransferError> for GpuHalError {
    fn from(e: TransferError) -> Self {
        Self::Transfer(e)
    }
}

impl From<BackendError> for GpuHalError {
    fn from(e: BackendError) -> Self {
        Self::Backend(e)
    }
}

/// Convenience alias for GPU HAL results.
pub type GpuHalResult<T> = Result<T, GpuHalError>;

// ── ErrorContext ─────────────────────────────────────────────────────────────

/// Rich context attached to errors for diagnostics.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// The high-level operation being performed when the error occurred.
    pub operation: String,
    /// Device ID where the error occurred (if applicable).
    pub device_id: Option<u32>,
    /// Backend that produced the error (if known).
    pub backend: Option<BackendKind>,
    /// Timestamp when the error occurred (millis since UNIX epoch).
    pub timestamp_ms: u64,
    /// Optional key-value metadata for additional context.
    pub metadata: HashMap<String, String>,
    /// Optional call stack or location hint.
    pub location: Option<String>,
}

impl ErrorContext {
    pub fn new(operation: impl Into<String>) -> Self {
        let ts =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
        Self {
            operation: operation.into(),
            device_id: None,
            backend: None,
            timestamp_ms: ts,
            metadata: HashMap::new(),
            location: None,
        }
    }

    pub fn with_device(mut self, device_id: u32) -> Self {
        self.device_id = Some(device_id);
        self
    }

    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = Some(backend);
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "op={}", self.operation)?;
        if let Some(id) = self.device_id {
            write!(f, " device={id}")?;
        }
        if let Some(ref b) = self.backend {
            write!(f, " backend={b}")?;
        }
        if let Some(ref loc) = self.location {
            write!(f, " at {loc}")?;
        }
        if !self.metadata.is_empty() {
            let pairs: Vec<String> =
                self.metadata.iter().map(|(k, v)| format!("{k}={v}")).collect();
            write!(f, " {{{}}}", pairs.join(", "))?;
        }
        Ok(())
    }
}

// ── ErrorRecovery ───────────────────────────────────────────────────────────

/// Suggested recovery strategy for a GPU HAL error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Retry the same operation (possibly after a delay).
    Retry { max_attempts: u32, delay: Duration },
    /// Fall back to an alternative backend or code path.
    Fallback { target: String },
    /// Abort the operation; no recovery is possible.
    Abort { reason: String },
    /// Degrade: continue with reduced functionality.
    Degrade { description: String },
    /// Reset the device and retry.
    ResetAndRetry { device_id: u32 },
}

impl fmt::Display for RecoveryAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Retry { max_attempts, delay } => {
                write!(f, "retry (max {max_attempts}, delay {}ms)", delay.as_millis())
            }
            Self::Fallback { target } => {
                write!(f, "fallback to {target}")
            }
            Self::Abort { reason } => write!(f, "abort: {reason}"),
            Self::Degrade { description } => {
                write!(f, "degrade: {description}")
            }
            Self::ResetAndRetry { device_id } => {
                write!(f, "reset device {device_id} and retry")
            }
        }
    }
}

/// Determines recovery strategies for GPU HAL errors.
pub struct ErrorRecovery {
    /// Maximum retry attempts for transient errors.
    pub max_retries: u32,
    /// Base delay between retries.
    pub retry_delay: Duration,
    /// Fallback backend to use when primary fails.
    pub fallback_backend: Option<String>,
}

impl Default for ErrorRecovery {
    fn default() -> Self {
        Self { max_retries: 3, retry_delay: Duration::from_millis(100), fallback_backend: None }
    }
}

impl ErrorRecovery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    pub fn with_retry_delay(mut self, delay: Duration) -> Self {
        self.retry_delay = delay;
        self
    }

    pub fn with_fallback(mut self, backend: impl Into<String>) -> Self {
        self.fallback_backend = Some(backend.into());
        self
    }

    /// Suggest a recovery action for the given error.
    pub fn suggest(&self, error: &GpuHalError) -> RecoveryAction {
        match error {
            GpuHalError::Memory(MemoryError::OutOfMemory { .. }) => {
                if let Some(ref fb) = self.fallback_backend {
                    RecoveryAction::Fallback { target: fb.clone() }
                } else {
                    RecoveryAction::Degrade {
                        description: "reduce batch size or model precision".into(),
                    }
                }
            }
            GpuHalError::Memory(MemoryError::Fragmentation { .. }) => RecoveryAction::Degrade {
                description: "defragment memory pool or reduce allocation".into(),
            },
            GpuHalError::Kernel(KernelError::Timeout { .. }) => {
                RecoveryAction::Retry { max_attempts: self.max_retries, delay: self.retry_delay }
            }
            GpuHalError::Transfer(TransferError::Timeout { .. }) => {
                RecoveryAction::Retry { max_attempts: self.max_retries, delay: self.retry_delay }
            }
            GpuHalError::Device(DeviceError::Unavailable { .. }) => RecoveryAction::Retry {
                max_attempts: self.max_retries,
                delay: self.retry_delay * 2,
            },
            GpuHalError::Device(DeviceError::DeviceLost { device_id }) => {
                RecoveryAction::ResetAndRetry { device_id: *device_id }
            }
            GpuHalError::Device(DeviceError::DriverError { .. }) => RecoveryAction::Abort {
                reason: "driver mismatch requires manual intervention".into(),
            },
            GpuHalError::Kernel(KernelError::CompilationFailed { .. }) => {
                RecoveryAction::Abort { reason: "kernel compilation failed; check source".into() }
            }
            GpuHalError::Backend(_) => {
                if let Some(ref fb) = self.fallback_backend {
                    RecoveryAction::Fallback { target: fb.clone() }
                } else {
                    RecoveryAction::Abort { reason: "backend error with no fallback".into() }
                }
            }
            GpuHalError::Contextualized { error, .. } => self.suggest(error),
            _ => RecoveryAction::Abort { reason: "no recovery strategy available".into() },
        }
    }
}

// ── ErrorReporter ───────────────────────────────────────────────────────────

/// Severity level for reported errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ErrorSeverity {
    /// Informational; operation succeeded with caveats.
    Info,
    /// Warning; non-fatal issue detected.
    Warning,
    /// Error; operation failed but system is stable.
    Error,
    /// Critical; system may be in an unstable state.
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// A structured error report suitable for logging and telemetry.
#[derive(Debug, Clone)]
pub struct ErrorReport {
    /// Unique report identifier.
    pub id: u64,
    /// Severity of the error.
    pub severity: ErrorSeverity,
    /// Error category (device, memory, kernel, transfer, backend, other).
    pub category: String,
    /// Human-readable error message.
    pub message: String,
    /// Optional context.
    pub context: Option<ErrorContext>,
    /// Suggested recovery action.
    pub recovery: Option<String>,
    /// Timestamp (millis since UNIX epoch).
    pub timestamp_ms: u64,
}

impl fmt::Display for ErrorReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] #{} [{}] {}", self.severity, self.id, self.category, self.message)?;
        if let Some(ref ctx) = self.context {
            write!(f, " ({ctx})")?;
        }
        if let Some(ref rec) = self.recovery {
            write!(f, " → {rec}")?;
        }
        Ok(())
    }
}

/// Structured error reporter that collects and formats error reports.
pub struct ErrorReporter {
    next_id: AtomicU64,
    reports: Vec<ErrorReport>,
    min_severity: ErrorSeverity,
}

impl ErrorReporter {
    pub fn new() -> Self {
        Self { next_id: AtomicU64::new(1), reports: Vec::new(), min_severity: ErrorSeverity::Info }
    }

    pub fn with_min_severity(mut self, severity: ErrorSeverity) -> Self {
        self.min_severity = severity;
        self
    }

    /// Classify the severity of a `GpuHalError`.
    pub fn classify_severity(error: &GpuHalError) -> ErrorSeverity {
        match error {
            GpuHalError::Device(DeviceError::DeviceLost { .. }) => ErrorSeverity::Critical,
            GpuHalError::Memory(MemoryError::Corruption { .. }) => ErrorSeverity::Critical,
            GpuHalError::Device(DeviceError::DriverError { .. }) => ErrorSeverity::Error,
            GpuHalError::Kernel(KernelError::CompilationFailed { .. }) => ErrorSeverity::Error,
            GpuHalError::Memory(MemoryError::OutOfMemory { .. }) => ErrorSeverity::Error,
            GpuHalError::Kernel(KernelError::Timeout { .. }) => ErrorSeverity::Warning,
            GpuHalError::Transfer(TransferError::Timeout { .. }) => ErrorSeverity::Warning,
            GpuHalError::Device(DeviceError::Unavailable { .. }) => ErrorSeverity::Warning,
            GpuHalError::Memory(MemoryError::Fragmentation { .. }) => ErrorSeverity::Warning,
            GpuHalError::Contextualized { error, .. } => Self::classify_severity(error),
            _ => ErrorSeverity::Error,
        }
    }

    /// Report an error, creating a structured report.
    pub fn report(
        &mut self,
        error: &GpuHalError,
        context: Option<ErrorContext>,
        recovery: Option<&RecoveryAction>,
    ) -> &ErrorReport {
        let severity = Self::classify_severity(error);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let ts =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
        let report = ErrorReport {
            id,
            severity,
            category: error.category().to_string(),
            message: error.to_string(),
            context,
            recovery: recovery.map(|r| r.to_string()),
            timestamp_ms: ts,
        };
        self.reports.push(report);
        self.reports.last().unwrap()
    }

    /// Returns all collected reports.
    pub fn reports(&self) -> &[ErrorReport] {
        &self.reports
    }

    /// Returns reports filtered by minimum configured severity.
    pub fn filtered_reports(&self) -> Vec<&ErrorReport> {
        self.reports.iter().filter(|r| r.severity >= self.min_severity).collect()
    }

    /// Total number of reports.
    pub fn count(&self) -> usize {
        self.reports.len()
    }

    /// Number of reports at or above a given severity.
    pub fn count_at_severity(&self, severity: ErrorSeverity) -> usize {
        self.reports.iter().filter(|r| r.severity >= severity).count()
    }

    /// Clear all collected reports.
    pub fn clear(&mut self) {
        self.reports.clear();
    }
}

impl Default for ErrorReporter {
    fn default() -> Self {
        Self::new()
    }
}

// ── ErrorEngine ─────────────────────────────────────────────────────────────

/// Central error handling engine combining recovery and reporting.
///
/// The `ErrorEngine` wires together [`ErrorRecovery`] and [`ErrorReporter`]
/// to provide a single entry point for handling GPU HAL errors: it
/// determines recovery strategy, records a report, and returns both.
pub struct ErrorEngine {
    recovery: ErrorRecovery,
    reporter: ErrorReporter,
    total_errors: AtomicU64,
    total_recovered: AtomicU64,
}

/// Result of processing an error through the engine.
#[derive(Debug)]
pub struct HandleResult {
    pub recovery_action: RecoveryAction,
    pub report_id: u64,
    pub severity: ErrorSeverity,
}

impl ErrorEngine {
    pub fn new(recovery: ErrorRecovery, reporter: ErrorReporter) -> Self {
        Self {
            recovery,
            reporter,
            total_errors: AtomicU64::new(0),
            total_recovered: AtomicU64::new(0),
        }
    }

    /// Handle an error: determine recovery, record a report, return both.
    pub fn handle(&mut self, error: &GpuHalError, context: Option<ErrorContext>) -> HandleResult {
        self.total_errors.fetch_add(1, Ordering::Relaxed);

        let action = self.recovery.suggest(error);
        let report = self.reporter.report(error, context, Some(&action));

        let result = HandleResult {
            recovery_action: action.clone(),
            report_id: report.id,
            severity: report.severity,
        };

        // Count non-abort recoveries.
        if !matches!(action, RecoveryAction::Abort { .. }) {
            self.total_recovered.fetch_add(1, Ordering::Relaxed);
        }

        result
    }

    /// Handle error and execute a retry closure if recovery says retry.
    pub fn handle_with_retry<F, T>(
        &mut self,
        error: &GpuHalError,
        context: Option<ErrorContext>,
        mut retry_fn: F,
    ) -> Result<T, HandleResult>
    where
        F: FnMut(u32) -> Result<T, GpuHalError>,
    {
        let result = self.handle(error, context);
        if let RecoveryAction::Retry { max_attempts, delay } = &result.recovery_action {
            for attempt in 1..=*max_attempts {
                std::thread::sleep(*delay);
                match retry_fn(attempt) {
                    Ok(val) => return Ok(val),
                    Err(ref retry_err) if attempt < *max_attempts => {
                        self.reporter.report(retry_err, None, None);
                    }
                    Err(_) => {}
                }
            }
        }
        Err(result)
    }

    /// Total errors processed.
    pub fn total_errors(&self) -> u64 {
        self.total_errors.load(Ordering::Relaxed)
    }

    /// Total errors where a non-abort recovery was suggested.
    pub fn total_recovered(&self) -> u64 {
        self.total_recovered.load(Ordering::Relaxed)
    }

    /// Access the reporter's collected reports.
    pub fn reports(&self) -> &[ErrorReport] {
        self.reporter.reports()
    }

    /// Access the recovery configuration.
    pub fn recovery(&self) -> &ErrorRecovery {
        &self.recovery
    }

    /// Clear all reports and reset counters.
    pub fn reset(&mut self) {
        self.reporter.clear();
        self.total_errors.store(0, Ordering::Relaxed);
        self.total_recovered.store(0, Ordering::Relaxed);
    }
}

impl Default for ErrorEngine {
    fn default() -> Self {
        Self::new(ErrorRecovery::default(), ErrorReporter::new())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DeviceError tests ───────────────────────────────────────────────

    #[test]
    fn device_not_found_display() {
        let e = DeviceError::NotFound { query: "GPU:0".into() };
        assert!(e.to_string().contains("GPU:0"));
        assert!(e.to_string().contains("no compatible device"));
    }

    #[test]
    fn device_unavailable_display() {
        let e = DeviceError::Unavailable { device_id: 1, reason: "exclusive mode".into() };
        assert!(e.to_string().contains("device 1"));
        assert!(e.to_string().contains("exclusive mode"));
    }

    #[test]
    fn device_driver_error_display() {
        let e = DeviceError::DriverError { expected: "535.0".into(), found: "520.0".into() };
        assert!(e.to_string().contains("535.0"));
        assert!(e.to_string().contains("520.0"));
    }

    #[test]
    fn device_capability_missing_display() {
        let e = DeviceError::CapabilityMissing { device_id: 0, capability: "fp16".into() };
        assert!(e.to_string().contains("fp16"));
    }

    #[test]
    fn device_init_failed_display() {
        let e = DeviceError::InitFailed { device_id: 2, message: "context creation failed".into() };
        assert!(e.to_string().contains("device 2"));
        assert!(e.to_string().contains("context creation"));
    }

    #[test]
    fn device_lost_display() {
        let e = DeviceError::DeviceLost { device_id: 3 };
        assert!(e.to_string().contains("device 3 lost"));
    }

    #[test]
    fn device_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(DeviceError::DeviceLost { device_id: 0 });
        assert!(e.source().is_none());
    }

    #[test]
    fn device_error_debug() {
        let e = DeviceError::NotFound { query: "test".into() };
        let debug = format!("{e:?}");
        assert!(debug.contains("NotFound"));
    }

    // ── MemoryError tests ───────────────────────────────────────────────

    #[test]
    fn memory_oom_display() {
        let e = MemoryError::OutOfMemory { requested: 1024, available: 512 };
        assert!(e.to_string().contains("1024"));
        assert!(e.to_string().contains("512"));
    }

    #[test]
    fn memory_out_of_bounds_display() {
        let e = MemoryError::OutOfBounds { offset: 100, length: 200, buffer_size: 150 };
        let msg = e.to_string();
        assert!(msg.contains("100"));
        assert!(msg.contains("200"));
        assert!(msg.contains("150"));
    }

    #[test]
    fn memory_corruption_display() {
        let e = MemoryError::Corruption { buffer_id: 42, details: "checksum mismatch".into() };
        assert!(e.to_string().contains("42"));
        assert!(e.to_string().contains("checksum"));
    }

    #[test]
    fn memory_fragmentation_display() {
        let e =
            MemoryError::Fragmentation { requested: 1000, total_free: 2000, largest_block: 500 };
        let msg = e.to_string();
        assert!(msg.contains("1000"));
        assert!(msg.contains("2000"));
        assert!(msg.contains("500"));
    }

    #[test]
    fn memory_invalid_buffer_display() {
        let e = MemoryError::InvalidBuffer { buffer_id: 99 };
        assert!(e.to_string().contains("99"));
    }

    #[test]
    fn memory_mapping_failed_display() {
        let e = MemoryError::MappingFailed { reason: "no host pointer".into() };
        assert!(e.to_string().contains("no host pointer"));
    }

    #[test]
    fn memory_alignment_error_display() {
        let e = MemoryError::AlignmentError { required: 256, actual: 64 };
        assert!(e.to_string().contains("256"));
        assert!(e.to_string().contains("64"));
    }

    #[test]
    fn memory_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(MemoryError::InvalidBuffer { buffer_id: 1 });
        assert!(e.source().is_none());
    }

    // ── KernelError tests ───────────────────────────────────────────────

    #[test]
    fn kernel_compilation_failed_display() {
        let e = KernelError::CompilationFailed {
            kernel_name: "matmul".into(),
            log: "syntax error line 5".into(),
        };
        assert!(e.to_string().contains("matmul"));
        assert!(e.to_string().contains("syntax error"));
    }

    #[test]
    fn kernel_launch_config_invalid_display() {
        let e = KernelError::LaunchConfigInvalid {
            kernel_name: "reduce".into(),
            reason: "block size too large".into(),
        };
        assert!(e.to_string().contains("reduce"));
        assert!(e.to_string().contains("block size"));
    }

    #[test]
    fn kernel_timeout_display() {
        let e = KernelError::Timeout {
            kernel_name: "attention".into(),
            elapsed_ms: 5000,
            limit_ms: 3000,
        };
        let msg = e.to_string();
        assert!(msg.contains("attention"));
        assert!(msg.contains("5000"));
        assert!(msg.contains("3000"));
    }

    #[test]
    fn kernel_invalid_argument_display() {
        let e = KernelError::InvalidArgument {
            kernel_name: "softmax".into(),
            index: 2,
            reason: "wrong type".into(),
        };
        assert!(e.to_string().contains("softmax"));
        assert!(e.to_string().contains("index 2"));
    }

    #[test]
    fn kernel_not_found_display() {
        let e = KernelError::NotFound { kernel_name: "missing_kernel".into() };
        assert!(e.to_string().contains("missing_kernel"));
    }

    #[test]
    fn kernel_execution_failed_display() {
        let e = KernelError::ExecutionFailed { kernel_name: "gelu".into(), error_code: -700 };
        assert!(e.to_string().contains("gelu"));
        assert!(e.to_string().contains("-700"));
    }

    #[test]
    fn kernel_error_is_std_error() {
        let e: Box<dyn std::error::Error> =
            Box::new(KernelError::NotFound { kernel_name: "x".into() });
        assert!(e.source().is_none());
    }

    // ── TransferError tests ─────────────────────────────────────────────

    #[test]
    fn transfer_direction_display() {
        assert_eq!(TransferDirection::HostToDevice.to_string(), "host→device");
        assert_eq!(TransferDirection::DeviceToHost.to_string(), "device→host");
        assert_eq!(TransferDirection::DeviceToDevice.to_string(), "device→device");
    }

    #[test]
    fn transfer_failed_display() {
        let e = TransferError::Failed {
            direction: TransferDirection::HostToDevice,
            size_bytes: 4096,
            reason: "bus error".into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("host→device"));
        assert!(msg.contains("4096"));
        assert!(msg.contains("bus error"));
    }

    #[test]
    fn transfer_timeout_display() {
        let e =
            TransferError::Timeout { direction: TransferDirection::DeviceToHost, elapsed_ms: 1000 };
        assert!(e.to_string().contains("device→host"));
        assert!(e.to_string().contains("1000"));
    }

    #[test]
    fn transfer_invalid_buffer_display() {
        let e = TransferError::InvalidBuffer {
            direction: TransferDirection::DeviceToDevice,
            buffer_id: 77,
        };
        assert!(e.to_string().contains("device→device"));
        assert!(e.to_string().contains("77"));
    }

    #[test]
    fn transfer_size_mismatch_display() {
        let e = TransferError::SizeMismatch { source_size: 1024, dest_size: 512 };
        assert!(e.to_string().contains("1024"));
        assert!(e.to_string().contains("512"));
    }

    #[test]
    fn transfer_dma_error_display() {
        let e = TransferError::DmaError { channel: 2, message: "channel reset".into() };
        assert!(e.to_string().contains("channel 2"));
        assert!(e.to_string().contains("channel reset"));
    }

    #[test]
    fn transfer_error_is_std_error() {
        let e: Box<dyn std::error::Error> =
            Box::new(TransferError::SizeMismatch { source_size: 1, dest_size: 2 });
        assert!(e.source().is_none());
    }

    // ── BackendError tests ──────────────────────────────────────────────

    #[test]
    fn backend_kind_display() {
        assert_eq!(BackendKind::CUDA.to_string(), "CUDA");
        assert_eq!(BackendKind::OpenCL.to_string(), "OpenCL");
        assert_eq!(BackendKind::Vulkan.to_string(), "Vulkan");
        assert_eq!(BackendKind::Metal.to_string(), "Metal");
        assert_eq!(BackendKind::ROCm.to_string(), "ROCm");
        assert_eq!(BackendKind::LevelZero.to_string(), "LevelZero");
        assert_eq!(BackendKind::WebGPU.to_string(), "WebGPU");
        assert_eq!(BackendKind::Other.to_string(), "Other");
    }

    #[test]
    fn backend_error_display_without_api_call() {
        let e = BackendError::new(BackendKind::CUDA, 2, "out of memory");
        let msg = e.to_string();
        assert!(msg.contains("[CUDA]"));
        assert!(msg.contains("out of memory"));
        assert!(msg.contains("code 2"));
    }

    #[test]
    fn backend_error_display_with_api_call() {
        let e = BackendError::new(BackendKind::Vulkan, -3, "device lost")
            .with_api_call("vkQueueSubmit");
        let msg = e.to_string();
        assert!(msg.contains("[Vulkan]"));
        assert!(msg.contains("vkQueueSubmit"));
        assert!(msg.contains("device lost"));
    }

    #[test]
    fn backend_error_is_std_error() {
        let e: Box<dyn std::error::Error> =
            Box::new(BackendError::new(BackendKind::OpenCL, 0, "ok"));
        assert!(e.source().is_none());
    }

    // ── GpuHalError tests ───────────────────────────────────────────────

    #[test]
    fn gpu_hal_error_device_display() {
        let e = GpuHalError::Device(DeviceError::DeviceLost { device_id: 0 });
        assert!(e.to_string().contains("device error"));
        assert!(e.to_string().contains("device 0 lost"));
    }

    #[test]
    fn gpu_hal_error_memory_display() {
        let e = GpuHalError::Memory(MemoryError::OutOfMemory { requested: 100, available: 50 });
        assert!(e.to_string().contains("memory error"));
    }

    #[test]
    fn gpu_hal_error_kernel_display() {
        let e = GpuHalError::Kernel(KernelError::NotFound { kernel_name: "test".into() });
        assert!(e.to_string().contains("kernel error"));
    }

    #[test]
    fn gpu_hal_error_transfer_display() {
        let e = GpuHalError::Transfer(TransferError::SizeMismatch { source_size: 1, dest_size: 2 });
        assert!(e.to_string().contains("transfer error"));
    }

    #[test]
    fn gpu_hal_error_backend_display() {
        let e = GpuHalError::Backend(BackendError::new(BackendKind::Metal, 1, "unsupported"));
        assert!(e.to_string().contains("backend error"));
    }

    #[test]
    fn gpu_hal_error_other_display() {
        let e = GpuHalError::Other("something weird".into());
        assert!(e.to_string().contains("something weird"));
    }

    #[test]
    fn gpu_hal_error_category() {
        assert_eq!(
            GpuHalError::Device(DeviceError::DeviceLost { device_id: 0 }).category(),
            "device"
        );
        assert_eq!(
            GpuHalError::Memory(MemoryError::InvalidBuffer { buffer_id: 0 }).category(),
            "memory"
        );
        assert_eq!(
            GpuHalError::Kernel(KernelError::NotFound { kernel_name: "x".into() }).category(),
            "kernel"
        );
        assert_eq!(
            GpuHalError::Transfer(TransferError::SizeMismatch { source_size: 0, dest_size: 0 })
                .category(),
            "transfer"
        );
        assert_eq!(
            GpuHalError::Backend(BackendError::new(BackendKind::CUDA, 0, "")).category(),
            "backend"
        );
        assert_eq!(GpuHalError::Other("x".into()).category(), "other");
    }

    #[test]
    fn gpu_hal_error_is_transient() {
        assert!(
            GpuHalError::Memory(MemoryError::OutOfMemory { requested: 1, available: 0 })
                .is_transient()
        );
        assert!(
            GpuHalError::Kernel(KernelError::Timeout {
                kernel_name: "x".into(),
                elapsed_ms: 1,
                limit_ms: 1,
            })
            .is_transient()
        );
        assert!(
            GpuHalError::Transfer(TransferError::Timeout {
                direction: TransferDirection::HostToDevice,
                elapsed_ms: 1,
            })
            .is_transient()
        );
        assert!(
            GpuHalError::Device(DeviceError::Unavailable { device_id: 0, reason: "busy".into() })
                .is_transient()
        );
        // Non-transient
        assert!(
            !GpuHalError::Kernel(KernelError::NotFound { kernel_name: "x".into() }).is_transient()
        );
    }

    #[test]
    fn gpu_hal_error_source_chain() {
        let e = GpuHalError::Device(DeviceError::DeviceLost { device_id: 0 });
        let src = std::error::Error::source(&e);
        assert!(src.is_some());
    }

    #[test]
    fn gpu_hal_error_other_has_no_source() {
        let e = GpuHalError::Other("test".into());
        assert!(std::error::Error::source(&e).is_none());
    }

    // ── From conversions ────────────────────────────────────────────────

    #[test]
    fn from_device_error() {
        let e: GpuHalError = DeviceError::DeviceLost { device_id: 0 }.into();
        assert_eq!(e.category(), "device");
    }

    #[test]
    fn from_memory_error() {
        let e: GpuHalError = MemoryError::InvalidBuffer { buffer_id: 0 }.into();
        assert_eq!(e.category(), "memory");
    }

    #[test]
    fn from_kernel_error() {
        let e: GpuHalError = KernelError::NotFound { kernel_name: "k".into() }.into();
        assert_eq!(e.category(), "kernel");
    }

    #[test]
    fn from_transfer_error() {
        let e: GpuHalError = TransferError::SizeMismatch { source_size: 1, dest_size: 2 }.into();
        assert_eq!(e.category(), "transfer");
    }

    #[test]
    fn from_backend_error() {
        let e: GpuHalError = BackendError::new(BackendKind::CUDA, 0, "").into();
        assert_eq!(e.category(), "backend");
    }

    #[test]
    fn question_mark_propagation() {
        fn inner() -> GpuHalResult<()> {
            Err(DeviceError::NotFound { query: "test".into() })?;
            Ok(())
        }
        assert!(inner().is_err());
    }

    // ── ErrorContext tests ───────────────────────────────────────────────

    #[test]
    fn error_context_basic() {
        let ctx = ErrorContext::new("inference");
        assert_eq!(ctx.operation, "inference");
        assert!(ctx.timestamp_ms > 0);
        assert!(ctx.device_id.is_none());
        assert!(ctx.backend.is_none());
        assert!(ctx.metadata.is_empty());
    }

    #[test]
    fn error_context_with_device() {
        let ctx = ErrorContext::new("alloc").with_device(2);
        assert_eq!(ctx.device_id, Some(2));
    }

    #[test]
    fn error_context_with_backend() {
        let ctx = ErrorContext::new("launch").with_backend(BackendKind::CUDA);
        assert_eq!(ctx.backend, Some(BackendKind::CUDA));
    }

    #[test]
    fn error_context_with_metadata() {
        let ctx = ErrorContext::new("op")
            .with_metadata("batch_size", "32")
            .with_metadata("model", "bitnet-2b");
        assert_eq!(ctx.metadata.get("batch_size").unwrap(), "32");
        assert_eq!(ctx.metadata.get("model").unwrap(), "bitnet-2b");
    }

    #[test]
    fn error_context_with_location() {
        let ctx = ErrorContext::new("op").with_location("kernel_launch:42");
        assert_eq!(ctx.location.as_deref(), Some("kernel_launch:42"));
    }

    #[test]
    fn error_context_display() {
        let ctx = ErrorContext::new("inference").with_device(0).with_backend(BackendKind::Vulkan);
        let msg = ctx.to_string();
        assert!(msg.contains("op=inference"));
        assert!(msg.contains("device=0"));
        assert!(msg.contains("backend=Vulkan"));
    }

    #[test]
    fn error_context_display_with_location() {
        let ctx = ErrorContext::new("op").with_location("file.rs:10");
        assert!(ctx.to_string().contains("at file.rs:10"));
    }

    #[test]
    fn gpu_hal_error_with_context() {
        let e = GpuHalError::Memory(MemoryError::OutOfMemory { requested: 1024, available: 0 })
            .with_context(ErrorContext::new("model_load").with_device(0));
        assert_eq!(e.category(), "memory");
        let msg = e.to_string();
        assert!(msg.contains("context"));
        assert!(msg.contains("model_load"));
    }

    #[test]
    fn contextualized_error_source() {
        let e = GpuHalError::Memory(MemoryError::OutOfMemory { requested: 1, available: 0 })
            .with_context(ErrorContext::new("test"));
        assert!(std::error::Error::source(&e).is_some());
    }

    #[test]
    fn contextualized_category_delegates() {
        let e = GpuHalError::Kernel(KernelError::Timeout {
            kernel_name: "k".into(),
            elapsed_ms: 1,
            limit_ms: 1,
        })
        .with_context(ErrorContext::new("test"));
        assert_eq!(e.category(), "kernel");
    }

    // ── ErrorRecovery tests ─────────────────────────────────────────────

    #[test]
    fn recovery_default() {
        let r = ErrorRecovery::default();
        assert_eq!(r.max_retries, 3);
        assert_eq!(r.retry_delay, Duration::from_millis(100));
        assert!(r.fallback_backend.is_none());
    }

    #[test]
    fn recovery_builder() {
        let r = ErrorRecovery::new()
            .with_max_retries(5)
            .with_retry_delay(Duration::from_secs(1))
            .with_fallback("CPU");
        assert_eq!(r.max_retries, 5);
        assert_eq!(r.retry_delay, Duration::from_secs(1));
        assert_eq!(r.fallback_backend.as_deref(), Some("CPU"));
    }

    #[test]
    fn recovery_suggest_oom_with_fallback() {
        let r = ErrorRecovery::new().with_fallback("CPU");
        let e = GpuHalError::Memory(MemoryError::OutOfMemory { requested: 1024, available: 0 });
        match r.suggest(&e) {
            RecoveryAction::Fallback { target } => {
                assert_eq!(target, "CPU");
            }
            other => panic!("expected Fallback, got {other:?}"),
        }
    }

    #[test]
    fn recovery_suggest_oom_without_fallback() {
        let r = ErrorRecovery::new();
        let e = GpuHalError::Memory(MemoryError::OutOfMemory { requested: 1024, available: 0 });
        assert!(matches!(r.suggest(&e), RecoveryAction::Degrade { .. }));
    }

    #[test]
    fn recovery_suggest_kernel_timeout() {
        let r = ErrorRecovery::new().with_max_retries(2);
        let e = GpuHalError::Kernel(KernelError::Timeout {
            kernel_name: "attn".into(),
            elapsed_ms: 5000,
            limit_ms: 3000,
        });
        match r.suggest(&e) {
            RecoveryAction::Retry { max_attempts, .. } => {
                assert_eq!(max_attempts, 2);
            }
            other => panic!("expected Retry, got {other:?}"),
        }
    }

    #[test]
    fn recovery_suggest_transfer_timeout() {
        let r = ErrorRecovery::new();
        let e = GpuHalError::Transfer(TransferError::Timeout {
            direction: TransferDirection::HostToDevice,
            elapsed_ms: 500,
        });
        assert!(matches!(r.suggest(&e), RecoveryAction::Retry { .. }));
    }

    #[test]
    fn recovery_suggest_device_unavailable() {
        let r = ErrorRecovery::new();
        let e =
            GpuHalError::Device(DeviceError::Unavailable { device_id: 0, reason: "busy".into() });
        assert!(matches!(r.suggest(&e), RecoveryAction::Retry { .. }));
    }

    #[test]
    fn recovery_suggest_device_lost() {
        let r = ErrorRecovery::new();
        let e = GpuHalError::Device(DeviceError::DeviceLost { device_id: 1 });
        match r.suggest(&e) {
            RecoveryAction::ResetAndRetry { device_id } => {
                assert_eq!(device_id, 1);
            }
            other => panic!("expected ResetAndRetry, got {other:?}"),
        }
    }

    #[test]
    fn recovery_suggest_driver_error_aborts() {
        let r = ErrorRecovery::new();
        let e = GpuHalError::Device(DeviceError::DriverError {
            expected: "a".into(),
            found: "b".into(),
        });
        assert!(matches!(r.suggest(&e), RecoveryAction::Abort { .. }));
    }

    #[test]
    fn recovery_suggest_compilation_aborts() {
        let r = ErrorRecovery::new();
        let e = GpuHalError::Kernel(KernelError::CompilationFailed {
            kernel_name: "k".into(),
            log: "error".into(),
        });
        assert!(matches!(r.suggest(&e), RecoveryAction::Abort { .. }));
    }

    #[test]
    fn recovery_suggest_backend_with_fallback() {
        let r = ErrorRecovery::new().with_fallback("OpenCL");
        let e = GpuHalError::Backend(BackendError::new(BackendKind::CUDA, 1, "fail"));
        match r.suggest(&e) {
            RecoveryAction::Fallback { target } => {
                assert_eq!(target, "OpenCL");
            }
            other => panic!("expected Fallback, got {other:?}"),
        }
    }

    #[test]
    fn recovery_suggest_backend_no_fallback_aborts() {
        let r = ErrorRecovery::new();
        let e = GpuHalError::Backend(BackendError::new(BackendKind::CUDA, 1, "fail"));
        assert!(matches!(r.suggest(&e), RecoveryAction::Abort { .. }));
    }

    #[test]
    fn recovery_suggest_fragmentation() {
        let r = ErrorRecovery::new();
        let e = GpuHalError::Memory(MemoryError::Fragmentation {
            requested: 1000,
            total_free: 2000,
            largest_block: 500,
        });
        assert!(matches!(r.suggest(&e), RecoveryAction::Degrade { .. }));
    }

    #[test]
    fn recovery_suggest_other_aborts() {
        let r = ErrorRecovery::new();
        let e = GpuHalError::Other("unknown".into());
        assert!(matches!(r.suggest(&e), RecoveryAction::Abort { .. }));
    }

    #[test]
    fn recovery_suggest_contextualized_delegates() {
        let r = ErrorRecovery::new().with_max_retries(7);
        let e = GpuHalError::Kernel(KernelError::Timeout {
            kernel_name: "k".into(),
            elapsed_ms: 1,
            limit_ms: 1,
        })
        .with_context(ErrorContext::new("test"));
        match r.suggest(&e) {
            RecoveryAction::Retry { max_attempts, .. } => {
                assert_eq!(max_attempts, 7);
            }
            other => panic!("expected Retry, got {other:?}"),
        }
    }

    #[test]
    fn recovery_action_display() {
        let a = RecoveryAction::Retry { max_attempts: 3, delay: Duration::from_millis(200) };
        assert!(a.to_string().contains("retry"));
        assert!(a.to_string().contains("200ms"));

        let a = RecoveryAction::Fallback { target: "CPU".into() };
        assert!(a.to_string().contains("CPU"));

        let a = RecoveryAction::Abort { reason: "fatal".into() };
        assert!(a.to_string().contains("fatal"));

        let a = RecoveryAction::Degrade { description: "reduce batch".into() };
        assert!(a.to_string().contains("reduce batch"));

        let a = RecoveryAction::ResetAndRetry { device_id: 5 };
        assert!(a.to_string().contains("device 5"));
    }

    // ── ErrorReporter tests ─────────────────────────────────────────────

    #[test]
    fn reporter_new_empty() {
        let r = ErrorReporter::new();
        assert_eq!(r.count(), 0);
        assert!(r.reports().is_empty());
    }

    #[test]
    fn reporter_report_basic() {
        let mut r = ErrorReporter::new();
        let e = GpuHalError::Memory(MemoryError::OutOfMemory { requested: 1, available: 0 });
        let report = r.report(&e, None, None);
        assert_eq!(report.id, 1);
        assert_eq!(report.severity, ErrorSeverity::Error);
        assert_eq!(report.category, "memory");
        assert!(report.timestamp_ms > 0);
    }

    #[test]
    fn reporter_increments_ids() {
        let mut r = ErrorReporter::new();
        let e = GpuHalError::Other("a".into());
        r.report(&e, None, None);
        let report = r.report(&e, None, None);
        assert_eq!(report.id, 2);
        assert_eq!(r.count(), 2);
    }

    #[test]
    fn reporter_with_context_and_recovery() {
        let mut r = ErrorReporter::new();
        let e = GpuHalError::Device(DeviceError::DeviceLost { device_id: 0 });
        let ctx = ErrorContext::new("inference").with_device(0);
        let action = RecoveryAction::ResetAndRetry { device_id: 0 };
        let report = r.report(&e, Some(ctx), Some(&action));
        assert!(report.context.is_some());
        assert!(report.recovery.is_some());
        assert_eq!(report.severity, ErrorSeverity::Critical);
    }

    #[test]
    fn reporter_classify_severity_critical() {
        assert_eq!(
            ErrorReporter::classify_severity(&GpuHalError::Device(DeviceError::DeviceLost {
                device_id: 0
            })),
            ErrorSeverity::Critical
        );
        assert_eq!(
            ErrorReporter::classify_severity(&GpuHalError::Memory(MemoryError::Corruption {
                buffer_id: 0,
                details: "x".into()
            })),
            ErrorSeverity::Critical
        );
    }

    #[test]
    fn reporter_classify_severity_warning() {
        assert_eq!(
            ErrorReporter::classify_severity(&GpuHalError::Kernel(KernelError::Timeout {
                kernel_name: "k".into(),
                elapsed_ms: 1,
                limit_ms: 1,
            })),
            ErrorSeverity::Warning
        );
        assert_eq!(
            ErrorReporter::classify_severity(&GpuHalError::Transfer(TransferError::Timeout {
                direction: TransferDirection::HostToDevice,
                elapsed_ms: 1,
            })),
            ErrorSeverity::Warning
        );
    }

    #[test]
    fn reporter_classify_severity_contextualized() {
        let e = GpuHalError::Device(DeviceError::DeviceLost { device_id: 0 })
            .with_context(ErrorContext::new("test"));
        assert_eq!(ErrorReporter::classify_severity(&e), ErrorSeverity::Critical);
    }

    #[test]
    fn reporter_filtered_reports() {
        let mut r = ErrorReporter::new().with_min_severity(ErrorSeverity::Error);
        let warn = GpuHalError::Kernel(KernelError::Timeout {
            kernel_name: "k".into(),
            elapsed_ms: 1,
            limit_ms: 1,
        });
        let err = GpuHalError::Memory(MemoryError::OutOfMemory { requested: 1, available: 0 });
        r.report(&warn, None, None);
        r.report(&err, None, None);
        assert_eq!(r.count(), 2);
        assert_eq!(r.filtered_reports().len(), 1);
    }

    #[test]
    fn reporter_count_at_severity() {
        let mut r = ErrorReporter::new();
        let crit = GpuHalError::Device(DeviceError::DeviceLost { device_id: 0 });
        let warn = GpuHalError::Kernel(KernelError::Timeout {
            kernel_name: "k".into(),
            elapsed_ms: 1,
            limit_ms: 1,
        });
        r.report(&crit, None, None);
        r.report(&warn, None, None);
        assert_eq!(r.count_at_severity(ErrorSeverity::Critical), 1);
        assert_eq!(r.count_at_severity(ErrorSeverity::Warning), 2);
    }

    #[test]
    fn reporter_clear() {
        let mut r = ErrorReporter::new();
        r.report(&GpuHalError::Other("x".into()), None, None);
        assert_eq!(r.count(), 1);
        r.clear();
        assert_eq!(r.count(), 0);
    }

    #[test]
    fn error_report_display() {
        let mut r = ErrorReporter::new();
        let e = GpuHalError::Other("test".into());
        let report = r.report(&e, None, None);
        let msg = report.to_string();
        assert!(msg.contains("[ERROR]"));
        assert!(msg.contains("[other]"));
    }

    #[test]
    fn error_severity_ordering() {
        assert!(ErrorSeverity::Info < ErrorSeverity::Warning);
        assert!(ErrorSeverity::Warning < ErrorSeverity::Error);
        assert!(ErrorSeverity::Error < ErrorSeverity::Critical);
    }

    #[test]
    fn error_severity_display() {
        assert_eq!(ErrorSeverity::Info.to_string(), "INFO");
        assert_eq!(ErrorSeverity::Warning.to_string(), "WARN");
        assert_eq!(ErrorSeverity::Error.to_string(), "ERROR");
        assert_eq!(ErrorSeverity::Critical.to_string(), "CRITICAL");
    }

    // ── ErrorEngine tests ───────────────────────────────────────────────

    #[test]
    fn engine_default() {
        let engine = ErrorEngine::default();
        assert_eq!(engine.total_errors(), 0);
        assert_eq!(engine.total_recovered(), 0);
        assert!(engine.reports().is_empty());
    }

    #[test]
    fn engine_handle_records_and_suggests() {
        let mut engine = ErrorEngine::default();
        let e = GpuHalError::Kernel(KernelError::Timeout {
            kernel_name: "attn".into(),
            elapsed_ms: 5000,
            limit_ms: 3000,
        });
        let result = engine.handle(&e, None);
        assert_eq!(engine.total_errors(), 1);
        assert_eq!(engine.total_recovered(), 1);
        assert!(matches!(result.recovery_action, RecoveryAction::Retry { .. }));
        assert_eq!(result.severity, ErrorSeverity::Warning);
        assert_eq!(engine.reports().len(), 1);
    }

    #[test]
    fn engine_handle_abort_not_counted_as_recovered() {
        let mut engine = ErrorEngine::default();
        let e = GpuHalError::Device(DeviceError::DriverError {
            expected: "a".into(),
            found: "b".into(),
        });
        let result = engine.handle(&e, None);
        assert_eq!(engine.total_errors(), 1);
        assert_eq!(engine.total_recovered(), 0);
        assert!(matches!(result.recovery_action, RecoveryAction::Abort { .. }));
    }

    #[test]
    fn engine_handle_with_context() {
        let mut engine = ErrorEngine::default();
        let e = GpuHalError::Memory(MemoryError::OutOfMemory { requested: 1024, available: 0 });
        let ctx = ErrorContext::new("model_load").with_device(0);
        let result = engine.handle(&e, Some(ctx));
        assert!(result.report_id > 0);
        let report = &engine.reports()[0];
        assert!(report.context.is_some());
    }

    #[test]
    fn engine_handle_with_retry_success() {
        let recovery =
            ErrorRecovery::new().with_max_retries(3).with_retry_delay(Duration::from_millis(1));
        let mut engine = ErrorEngine::new(recovery, ErrorReporter::new());
        let e = GpuHalError::Kernel(KernelError::Timeout {
            kernel_name: "k".into(),
            elapsed_ms: 1,
            limit_ms: 1,
        });
        let result = engine.handle_with_retry(&e, None, |attempt| {
            if attempt >= 2 {
                Ok(42)
            } else {
                Err(GpuHalError::Kernel(KernelError::Timeout {
                    kernel_name: "k".into(),
                    elapsed_ms: 1,
                    limit_ms: 1,
                }))
            }
        });
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn engine_handle_with_retry_exhausted() {
        let recovery =
            ErrorRecovery::new().with_max_retries(2).with_retry_delay(Duration::from_millis(1));
        let mut engine = ErrorEngine::new(recovery, ErrorReporter::new());
        let e = GpuHalError::Kernel(KernelError::Timeout {
            kernel_name: "k".into(),
            elapsed_ms: 1,
            limit_ms: 1,
        });
        let result: Result<i32, _> = engine.handle_with_retry(&e, None, |_attempt| {
            Err(GpuHalError::Kernel(KernelError::Timeout {
                kernel_name: "k".into(),
                elapsed_ms: 1,
                limit_ms: 1,
            }))
        });
        assert!(result.is_err());
    }

    #[test]
    fn engine_handle_with_retry_non_retryable_returns_err() {
        let mut engine = ErrorEngine::default();
        let e = GpuHalError::Device(DeviceError::DriverError {
            expected: "a".into(),
            found: "b".into(),
        });
        let result: Result<i32, _> = engine.handle_with_retry(&e, None, |_| Ok(1));
        // Abort action → no retry → returns Err
        assert!(result.is_err());
    }

    #[test]
    fn engine_reset() {
        let mut engine = ErrorEngine::default();
        engine.handle(&GpuHalError::Other("test".into()), None);
        assert_eq!(engine.total_errors(), 1);
        engine.reset();
        assert_eq!(engine.total_errors(), 0);
        assert_eq!(engine.total_recovered(), 0);
        assert!(engine.reports().is_empty());
    }

    #[test]
    fn engine_multiple_errors() {
        let mut engine = ErrorEngine::default();
        for i in 0..5 {
            engine.handle(&GpuHalError::Other(format!("error {i}")), None);
        }
        assert_eq!(engine.total_errors(), 5);
        assert_eq!(engine.reports().len(), 5);
    }

    #[test]
    fn engine_recovery_accessor() {
        let recovery = ErrorRecovery::new().with_max_retries(10);
        let engine = ErrorEngine::new(recovery, ErrorReporter::new());
        assert_eq!(engine.recovery().max_retries, 10);
    }
}
