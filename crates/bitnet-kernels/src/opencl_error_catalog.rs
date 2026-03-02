//! Unified error taxonomy and recovery guide for OpenCL kernel failures.
//!
//! Provides a structured catalog of all known OpenCL error codes, their
//! severities, human-readable descriptions, and recommended recovery actions.
//! Includes Intel Arc A770-specific error handling guidance.

use std::fmt;

// ---------------------------------------------------------------------------
// Error code taxonomy
// ---------------------------------------------------------------------------

/// Comprehensive OpenCL error codes covering device, memory, kernel, numeric,
/// and runtime failure modes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpenClErrorCode {
    DeviceNotFound,
    DeviceNotAvailable,
    CompilerNotAvailable,
    MemObjectAllocationFailure,
    OutOfResources,
    OutOfHostMemory,
    KernelCompilationFailed(String),
    KernelArgInvalid,
    InvalidWorkGroupSize,
    InvalidBufferSize,
    InvalidGlobalOffset,
    InvalidWorkDimension,
    NumericalPrecisionLoss,
    NaNDetected,
    InfDetected,
    Timeout,
    DeviceLost,
    DriverCrash,
    UnsupportedExtension(String),
    IncompatibleDevice,
    InternalError(String),
}

impl fmt::Display for OpenClErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotFound => write!(f, "CL_DEVICE_NOT_FOUND"),
            Self::DeviceNotAvailable => write!(f, "CL_DEVICE_NOT_AVAILABLE"),
            Self::CompilerNotAvailable => {
                write!(f, "CL_COMPILER_NOT_AVAILABLE")
            }
            Self::MemObjectAllocationFailure => {
                write!(f, "CL_MEM_OBJECT_ALLOCATION_FAILURE")
            }
            Self::OutOfResources => write!(f, "CL_OUT_OF_RESOURCES"),
            Self::OutOfHostMemory => write!(f, "CL_OUT_OF_HOST_MEMORY"),
            Self::KernelCompilationFailed(msg) => {
                write!(f, "CL_BUILD_PROGRAM_FAILURE: {msg}")
            }
            Self::KernelArgInvalid => write!(f, "CL_INVALID_KERNEL_ARGS"),
            Self::InvalidWorkGroupSize => {
                write!(f, "CL_INVALID_WORK_GROUP_SIZE")
            }
            Self::InvalidBufferSize => write!(f, "CL_INVALID_BUFFER_SIZE"),
            Self::InvalidGlobalOffset => {
                write!(f, "CL_INVALID_GLOBAL_OFFSET")
            }
            Self::InvalidWorkDimension => {
                write!(f, "CL_INVALID_WORK_DIMENSION")
            }
            Self::NumericalPrecisionLoss => write!(f, "NUMERICAL_PRECISION_LOSS"),
            Self::NaNDetected => write!(f, "NAN_DETECTED"),
            Self::InfDetected => write!(f, "INF_DETECTED"),
            Self::Timeout => write!(f, "CL_TIMEOUT"),
            Self::DeviceLost => write!(f, "CL_DEVICE_LOST"),
            Self::DriverCrash => write!(f, "DRIVER_CRASH"),
            Self::UnsupportedExtension(ext) => {
                write!(f, "UNSUPPORTED_EXTENSION: {ext}")
            }
            Self::IncompatibleDevice => write!(f, "INCOMPATIBLE_DEVICE"),
            Self::InternalError(msg) => write!(f, "INTERNAL_ERROR: {msg}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Severity, recovery, and diagnostic types
// ---------------------------------------------------------------------------

/// How severe is this error?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Recoverable,
    Fatal,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Recoverable => write!(f, "RECOVERABLE"),
            Self::Fatal => write!(f, "FATAL"),
        }
    }
}

/// Suggested recovery action for a given error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecoveryAction {
    RetryWithFallback,
    ReduceWorkgroupSize,
    ReduceBatchSize,
    FallbackToCPU,
    AllocateSmaller,
    RetryAfterDelay,
    Abort,
}

impl fmt::Display for RecoveryAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RetryWithFallback => write!(f, "Retry with fallback backend"),
            Self::ReduceWorkgroupSize => write!(f, "Reduce workgroup size"),
            Self::ReduceBatchSize => write!(f, "Reduce batch size"),
            Self::FallbackToCPU => write!(f, "Fall back to CPU kernels"),
            Self::AllocateSmaller => write!(f, "Allocate smaller buffers"),
            Self::RetryAfterDelay => write!(f, "Retry after brief delay"),
            Self::Abort => write!(f, "Abort operation"),
        }
    }
}

/// A single entry in the error catalog.
#[derive(Debug, Clone)]
pub struct ErrorEntry {
    pub code: OpenClErrorCode,
    pub severity: ErrorSeverity,
    pub message: String,
    pub recovery: Vec<RecoveryAction>,
    pub a770_specific: bool,
    pub documentation: String,
}

/// The full error catalog.
#[derive(Debug, Clone)]
pub struct ErrorCatalog {
    pub entries: Vec<ErrorEntry>,
}

/// Runtime diagnostic snapshot attached to error reports.
#[derive(Debug, Clone)]
pub struct DiagnosticInfo {
    pub device_name: String,
    pub driver_version: String,
    pub memory_used: u64,
    pub memory_total: u64,
    pub workgroup_size: u32,
    pub kernel_name: String,
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Construct the full catalog with all known errors, severities, and recovery
/// actions.
pub fn build_error_catalog() -> ErrorCatalog {
    let entries = vec![
        ErrorEntry {
            code: OpenClErrorCode::DeviceNotFound,
            severity: ErrorSeverity::Fatal,
            message: "No OpenCL device found on this system".into(),
            recovery: vec![RecoveryAction::FallbackToCPU, RecoveryAction::Abort],
            a770_specific: false,
            documentation: "Ensure an OpenCL-capable GPU is installed and \
                the ICD loader can discover it."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::DeviceNotAvailable,
            severity: ErrorSeverity::Fatal,
            message: "OpenCL device exists but is not available".into(),
            recovery: vec![RecoveryAction::FallbackToCPU, RecoveryAction::Abort],
            a770_specific: false,
            documentation: "The device may be in use by another process or \
                disabled in BIOS/driver settings."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::CompilerNotAvailable,
            severity: ErrorSeverity::Fatal,
            message: "Online kernel compiler not available".into(),
            recovery: vec![RecoveryAction::FallbackToCPU, RecoveryAction::Abort],
            a770_specific: false,
            documentation: "The OpenCL runtime does not support online \
                compilation. Use pre-compiled SPIR-V binaries."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::MemObjectAllocationFailure,
            severity: ErrorSeverity::Recoverable,
            message: "Failed to allocate device memory object".into(),
            recovery: vec![
                RecoveryAction::AllocateSmaller,
                RecoveryAction::ReduceBatchSize,
                RecoveryAction::FallbackToCPU,
            ],
            a770_specific: false,
            documentation: "Device memory is exhausted. Reduce allocation \
                size or free unused buffers."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::OutOfResources,
            severity: ErrorSeverity::Recoverable,
            message: "Device ran out of resources".into(),
            recovery: vec![
                RecoveryAction::ReduceBatchSize,
                RecoveryAction::ReduceWorkgroupSize,
                RecoveryAction::FallbackToCPU,
            ],
            a770_specific: false,
            documentation: "The device cannot satisfy the resource request. \
                Reduce workload or workgroup dimensions."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::OutOfHostMemory,
            severity: ErrorSeverity::Fatal,
            message: "Host memory allocation failed".into(),
            recovery: vec![RecoveryAction::Abort],
            a770_specific: false,
            documentation: "The system is out of host RAM. Close other \
                applications or reduce model size."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::KernelCompilationFailed(String::new()),
            severity: ErrorSeverity::Fatal,
            message: "Kernel source failed to compile".into(),
            recovery: vec![
                RecoveryAction::RetryWithFallback,
                RecoveryAction::FallbackToCPU,
                RecoveryAction::Abort,
            ],
            a770_specific: false,
            documentation: "Check the kernel source for syntax errors or \
                unsupported extensions on this device."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::KernelArgInvalid,
            severity: ErrorSeverity::Fatal,
            message: "Invalid kernel argument".into(),
            recovery: vec![RecoveryAction::Abort],
            a770_specific: false,
            documentation: "A kernel argument is null or has an incorrect \
                type/size. This is usually a programming error."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::InvalidWorkGroupSize,
            severity: ErrorSeverity::Recoverable,
            message: "Work-group size exceeds device limits".into(),
            recovery: vec![RecoveryAction::ReduceWorkgroupSize, RecoveryAction::RetryWithFallback],
            a770_specific: true,
            documentation: "The A770 supports max 1024 work-items per \
                work-group. Reduce local dimensions to fit."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::InvalidBufferSize,
            severity: ErrorSeverity::Recoverable,
            message: "Buffer size is invalid or exceeds limits".into(),
            recovery: vec![RecoveryAction::AllocateSmaller, RecoveryAction::ReduceBatchSize],
            a770_specific: false,
            documentation: "The requested buffer exceeds \
                CL_DEVICE_MAX_MEM_ALLOC_SIZE. Split into smaller buffers."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::InvalidGlobalOffset,
            severity: ErrorSeverity::Recoverable,
            message: "Global offset is out of range".into(),
            recovery: vec![RecoveryAction::RetryWithFallback],
            a770_specific: false,
            documentation: "The global work offset must be within the \
                valid range for the NDRange."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::InvalidWorkDimension,
            severity: ErrorSeverity::Recoverable,
            message: "Work dimension is not 1, 2, or 3".into(),
            recovery: vec![RecoveryAction::RetryWithFallback],
            a770_specific: false,
            documentation: "OpenCL supports 1D, 2D, and 3D work \
                dimensions only."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::NumericalPrecisionLoss,
            severity: ErrorSeverity::Warning,
            message: "Numerical precision loss detected in kernel output".into(),
            recovery: vec![RecoveryAction::FallbackToCPU, RecoveryAction::RetryWithFallback],
            a770_specific: true,
            documentation: "FP16 intermediate values may lose precision \
                on the A770. Consider FP32 accumulation."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::NaNDetected,
            severity: ErrorSeverity::Warning,
            message: "NaN value detected in kernel output".into(),
            recovery: vec![RecoveryAction::FallbackToCPU, RecoveryAction::RetryWithFallback],
            a770_specific: false,
            documentation: "NaN propagation indicates a numerical bug. \
                Check input data and kernel arithmetic."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::InfDetected,
            severity: ErrorSeverity::Warning,
            message: "Infinity value detected in kernel output".into(),
            recovery: vec![RecoveryAction::FallbackToCPU, RecoveryAction::RetryWithFallback],
            a770_specific: false,
            documentation: "Overflow to ±Inf suggests accumulator width \
                is too narrow or inputs are un-normalised."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::Timeout,
            severity: ErrorSeverity::Recoverable,
            message: "Kernel execution timed out".into(),
            recovery: vec![
                RecoveryAction::RetryAfterDelay,
                RecoveryAction::ReduceBatchSize,
                RecoveryAction::FallbackToCPU,
            ],
            a770_specific: false,
            documentation: "The kernel did not finish within the TDR \
                deadline. Reduce workload or increase timeout."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::DeviceLost,
            severity: ErrorSeverity::Fatal,
            message: "GPU device was lost (TDR or hardware reset)".into(),
            recovery: vec![RecoveryAction::FallbackToCPU, RecoveryAction::Abort],
            a770_specific: true,
            documentation: "The A770 triggered a TDR reset. Reduce kernel \
                duration or increase the TDR timeout in the driver."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::DriverCrash,
            severity: ErrorSeverity::Fatal,
            message: "GPU driver crashed during execution".into(),
            recovery: vec![RecoveryAction::FallbackToCPU, RecoveryAction::Abort],
            a770_specific: false,
            documentation: "A driver-level crash occurred. Update the GPU \
                driver and check for known errata."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::UnsupportedExtension(String::new()),
            severity: ErrorSeverity::Recoverable,
            message: "Required OpenCL extension is not supported".into(),
            recovery: vec![RecoveryAction::RetryWithFallback, RecoveryAction::FallbackToCPU],
            a770_specific: false,
            documentation: "The kernel requires an extension (e.g. \
                cl_intel_subgroups) not present on this device."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::IncompatibleDevice,
            severity: ErrorSeverity::Fatal,
            message: "Device is incompatible with required capabilities".into(),
            recovery: vec![RecoveryAction::FallbackToCPU, RecoveryAction::Abort],
            a770_specific: false,
            documentation: "The device does not meet minimum compute \
                capability requirements for BitNet inference."
                .into(),
        },
        ErrorEntry {
            code: OpenClErrorCode::InternalError(String::new()),
            severity: ErrorSeverity::Fatal,
            message: "An unexpected internal error occurred".into(),
            recovery: vec![RecoveryAction::Abort],
            a770_specific: false,
            documentation: "This is a catch-all for unrecognised error \
                codes. File a bug report with the error details."
                .into(),
        },
    ];
    ErrorCatalog { entries }
}

/// Find an error entry by its code. Variant payloads (strings) are ignored
/// during comparison so that e.g. any `KernelCompilationFailed` matches.
pub fn lookup_error<'a>(
    catalog: &'a ErrorCatalog,
    code: &OpenClErrorCode,
) -> Option<&'a ErrorEntry> {
    catalog.entries.iter().find(|e| variant_matches(&e.code, code))
}

/// Get recovery actions for a given error code.
pub fn suggest_recovery(catalog: &ErrorCatalog, code: &OpenClErrorCode) -> Vec<RecoveryAction> {
    lookup_error(catalog, code).map(|e| e.recovery.clone()).unwrap_or_default()
}

/// Whether this error is known to be Intel Arc A770-specific.
pub fn is_a770_specific(code: &OpenClErrorCode) -> bool {
    let catalog = build_error_catalog();
    lookup_error(&catalog, code).is_some_and(|e| e.a770_specific)
}

/// Classify the severity of an error code.
pub fn classify_severity(code: &OpenClErrorCode) -> ErrorSeverity {
    let catalog = build_error_catalog();
    lookup_error(&catalog, code).map(|e| e.severity).unwrap_or(ErrorSeverity::Fatal)
}

/// Produce a human-readable error report with optional diagnostic context.
pub fn format_error_report(entry: &ErrorEntry, diag: Option<&DiagnosticInfo>) -> String {
    let mut report = format!(
        "[{}] {}\n  Error: {}\n  Message: {}\n  A770-specific: {}\n  \
         Documentation: {}\n  Recovery actions:\n",
        entry.severity,
        entry.code,
        entry.code,
        entry.message,
        entry.a770_specific,
        entry.documentation,
    );
    for action in &entry.recovery {
        report.push_str(&format!("    - {action}\n"));
    }
    if let Some(d) = diag {
        report.push_str(&format!(
            "  Diagnostics:\n    Device: {}\n    Driver: {}\n    \
             Memory: {}/{} bytes\n    Workgroup size: {}\n    \
             Kernel: {}\n",
            d.device_name,
            d.driver_version,
            d.memory_used,
            d.memory_total,
            d.workgroup_size,
            d.kernel_name,
        ));
    }
    report
}

/// Simulate whether the given recovery actions would succeed for a code.
/// Recoverable/Warning/Info errors succeed; Fatal errors only succeed if
/// `FallbackToCPU` is among the actions.
pub fn cpu_simulate_recovery(code: &OpenClErrorCode, actions: &[RecoveryAction]) -> bool {
    let severity = classify_severity(code);
    match severity {
        ErrorSeverity::Fatal => actions.contains(&RecoveryAction::FallbackToCPU),
        _ => !actions.is_empty(),
    }
}

/// Filter the catalog to entries matching a specific severity.
pub fn filter_by_severity(catalog: &ErrorCatalog, severity: ErrorSeverity) -> Vec<&ErrorEntry> {
    catalog.entries.iter().filter(|e| e.severity == severity).collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compare two `OpenClErrorCode` values by discriminant only, ignoring any
/// inner `String` payload.
fn variant_matches(a: &OpenClErrorCode, b: &OpenClErrorCode) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- catalog construction -----------------------------------------------

    #[test]
    fn test_catalog_has_all_error_codes() {
        let catalog = build_error_catalog();
        assert_eq!(catalog.entries.len(), 21);
    }

    #[test]
    fn test_catalog_no_duplicate_discriminants() {
        let catalog = build_error_catalog();
        let mut seen = std::collections::HashSet::new();
        for entry in &catalog.entries {
            let disc = std::mem::discriminant(&entry.code);
            assert!(seen.insert(disc), "duplicate: {:?}", entry.code);
        }
    }

    #[test]
    fn test_catalog_contains_device_not_found() {
        let cat = build_error_catalog();
        assert!(lookup_error(&cat, &OpenClErrorCode::DeviceNotFound).is_some());
    }

    #[test]
    fn test_catalog_contains_timeout() {
        let cat = build_error_catalog();
        assert!(lookup_error(&cat, &OpenClErrorCode::Timeout).is_some());
    }

    // -- lookup -------------------------------------------------------------

    #[test]
    fn test_lookup_known_code() {
        let cat = build_error_catalog();
        let entry = lookup_error(&cat, &OpenClErrorCode::OutOfResources);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().severity, ErrorSeverity::Recoverable);
    }

    #[test]
    fn test_lookup_kernel_compilation_ignores_payload() {
        let cat = build_error_catalog();
        let code = OpenClErrorCode::KernelCompilationFailed("some specific error".into());
        assert!(lookup_error(&cat, &code).is_some());
    }

    #[test]
    fn test_lookup_unsupported_extension_ignores_payload() {
        let cat = build_error_catalog();
        let code = OpenClErrorCode::UnsupportedExtension("cl_khr_fp64".into());
        assert!(lookup_error(&cat, &code).is_some());
    }

    #[test]
    fn test_lookup_internal_error_ignores_payload() {
        let cat = build_error_catalog();
        let code = OpenClErrorCode::InternalError("oops".into());
        assert!(lookup_error(&cat, &code).is_some());
    }

    // -- recovery suggestions -----------------------------------------------

    #[test]
    fn test_suggest_recovery_non_empty() {
        let cat = build_error_catalog();
        for entry in &cat.entries {
            let actions = suggest_recovery(&cat, &entry.code);
            assert!(!actions.is_empty(), "no recovery for {:?}", entry.code);
        }
    }

    #[test]
    fn test_suggest_recovery_unknown_code_empty() {
        let _cat = build_error_catalog();
        // Create a code that doesn't exist by using InternalError then
        // removing it from the catalog. Instead, just use an empty catalog.
        let empty = ErrorCatalog { entries: vec![] };
        let actions = suggest_recovery(&empty, &OpenClErrorCode::DeviceNotFound);
        assert!(actions.is_empty());
    }

    #[test]
    fn test_out_of_resources_suggests_reduce() {
        let cat = build_error_catalog();
        let actions = suggest_recovery(&cat, &OpenClErrorCode::OutOfResources);
        assert!(actions.contains(&RecoveryAction::ReduceBatchSize));
    }

    #[test]
    fn test_timeout_suggests_retry_after_delay() {
        let cat = build_error_catalog();
        let actions = suggest_recovery(&cat, &OpenClErrorCode::Timeout);
        assert!(actions.contains(&RecoveryAction::RetryAfterDelay));
    }

    // -- A770-specific classification ----------------------------------------

    #[test]
    fn test_device_lost_is_a770_specific() {
        assert!(is_a770_specific(&OpenClErrorCode::DeviceLost));
    }

    #[test]
    fn test_invalid_work_group_size_is_a770_specific() {
        assert!(is_a770_specific(&OpenClErrorCode::InvalidWorkGroupSize));
    }

    #[test]
    fn test_numerical_precision_loss_is_a770_specific() {
        assert!(is_a770_specific(&OpenClErrorCode::NumericalPrecisionLoss));
    }

    #[test]
    fn test_out_of_resources_not_a770_specific() {
        assert!(!is_a770_specific(&OpenClErrorCode::OutOfResources));
    }

    #[test]
    fn test_driver_crash_not_a770_specific() {
        assert!(!is_a770_specific(&OpenClErrorCode::DriverCrash));
    }

    // -- severity classification --------------------------------------------

    #[test]
    fn test_severity_fatal_for_device_not_found() {
        assert_eq!(classify_severity(&OpenClErrorCode::DeviceNotFound), ErrorSeverity::Fatal,);
    }

    #[test]
    fn test_severity_recoverable_for_timeout() {
        assert_eq!(classify_severity(&OpenClErrorCode::Timeout), ErrorSeverity::Recoverable,);
    }

    #[test]
    fn test_severity_warning_for_nan() {
        assert_eq!(classify_severity(&OpenClErrorCode::NaNDetected), ErrorSeverity::Warning,);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(ErrorSeverity::Fatal > ErrorSeverity::Recoverable);
        assert!(ErrorSeverity::Recoverable > ErrorSeverity::Warning);
        assert!(ErrorSeverity::Warning > ErrorSeverity::Info);
    }

    // -- error report formatting --------------------------------------------

    #[test]
    fn test_format_report_contains_code() {
        let cat = build_error_catalog();
        let entry = lookup_error(&cat, &OpenClErrorCode::DeviceNotFound).unwrap();
        let report = format_error_report(entry, None);
        assert!(report.contains("CL_DEVICE_NOT_FOUND"));
    }

    #[test]
    fn test_format_report_contains_message() {
        let cat = build_error_catalog();
        let entry = lookup_error(&cat, &OpenClErrorCode::DeviceNotFound).unwrap();
        let report = format_error_report(entry, None);
        assert!(report.contains(&entry.message));
    }

    #[test]
    fn test_format_report_contains_recovery_actions() {
        let cat = build_error_catalog();
        let entry = lookup_error(&cat, &OpenClErrorCode::OutOfResources).unwrap();
        let report = format_error_report(entry, None);
        assert!(report.contains("Reduce batch size"));
    }

    #[test]
    fn test_format_report_with_diagnostics() {
        let cat = build_error_catalog();
        let entry = lookup_error(&cat, &OpenClErrorCode::DeviceLost).unwrap();
        let diag = DiagnosticInfo {
            device_name: "Intel Arc A770".into(),
            driver_version: "23.17.26241".into(),
            memory_used: 4_000_000_000,
            memory_total: 16_000_000_000,
            workgroup_size: 256,
            kernel_name: "matmul_i2s".into(),
        };
        let report = format_error_report(entry, Some(&diag));
        assert!(report.contains("Intel Arc A770"));
        assert!(report.contains("matmul_i2s"));
    }

    // -- fatal errors suggest Abort or FallbackToCPU -------------------------

    #[test]
    fn test_all_fatal_errors_suggest_abort_or_fallback() {
        let cat = build_error_catalog();
        for entry in filter_by_severity(&cat, ErrorSeverity::Fatal) {
            let has_abort = entry.recovery.contains(&RecoveryAction::Abort);
            let has_cpu = entry.recovery.contains(&RecoveryAction::FallbackToCPU);
            assert!(
                has_abort || has_cpu,
                "fatal error {:?} needs Abort or FallbackToCPU",
                entry.code
            );
        }
    }

    // -- recoverable errors have non-empty recovery -------------------------

    #[test]
    fn test_all_recoverable_non_empty_recovery() {
        let cat = build_error_catalog();
        for entry in filter_by_severity(&cat, ErrorSeverity::Recoverable) {
            assert!(!entry.recovery.is_empty(), "recoverable {:?} has no recovery", entry.code);
        }
    }

    // -- simulation ----------------------------------------------------------

    #[test]
    fn test_simulation_recoverable_succeeds() {
        let ok =
            cpu_simulate_recovery(&OpenClErrorCode::Timeout, &[RecoveryAction::RetryAfterDelay]);
        assert!(ok);
    }

    #[test]
    fn test_simulation_fatal_fails_without_cpu_fallback() {
        let ok = cpu_simulate_recovery(&OpenClErrorCode::DeviceNotFound, &[RecoveryAction::Abort]);
        assert!(!ok);
    }

    #[test]
    fn test_simulation_fatal_succeeds_with_cpu_fallback() {
        let ok = cpu_simulate_recovery(
            &OpenClErrorCode::DeviceNotFound,
            &[RecoveryAction::FallbackToCPU],
        );
        assert!(ok);
    }

    #[test]
    fn test_simulation_warning_succeeds() {
        let ok =
            cpu_simulate_recovery(&OpenClErrorCode::NaNDetected, &[RecoveryAction::FallbackToCPU]);
        assert!(ok);
    }

    // -- filter by severity --------------------------------------------------

    #[test]
    fn test_filter_fatal_count() {
        let cat = build_error_catalog();
        let fatal = filter_by_severity(&cat, ErrorSeverity::Fatal);
        // DeviceNotFound, DeviceNotAvailable, CompilerNotAvailable,
        // OutOfHostMemory, KernelCompilationFailed, KernelArgInvalid,
        // DeviceLost, DriverCrash, IncompatibleDevice, InternalError
        assert_eq!(fatal.len(), 10);
    }

    #[test]
    fn test_filter_recoverable_count() {
        let cat = build_error_catalog();
        let rec = filter_by_severity(&cat, ErrorSeverity::Recoverable);
        // MemObjectAllocationFailure, OutOfResources,
        // InvalidWorkGroupSize, InvalidBufferSize, InvalidGlobalOffset,
        // InvalidWorkDimension, Timeout, UnsupportedExtension
        assert_eq!(rec.len(), 8);
    }

    #[test]
    fn test_filter_warning_count() {
        let cat = build_error_catalog();
        let warn = filter_by_severity(&cat, ErrorSeverity::Warning);
        // NumericalPrecisionLoss, NaNDetected, InfDetected
        assert_eq!(warn.len(), 3);
    }

    #[test]
    fn test_filter_info_empty() {
        let cat = build_error_catalog();
        let info = filter_by_severity(&cat, ErrorSeverity::Info);
        assert!(info.is_empty());
    }

    // -- property: every entry has documentation ----------------------------

    #[test]
    fn test_every_entry_has_documentation() {
        let cat = build_error_catalog();
        for entry in &cat.entries {
            assert!(!entry.documentation.is_empty(), "missing docs for {:?}", entry.code);
        }
    }

    // -- property: every entry has at least one recovery action ---------------

    #[test]
    fn test_every_entry_has_recovery_action() {
        let cat = build_error_catalog();
        for entry in &cat.entries {
            assert!(!entry.recovery.is_empty(), "no recovery for {:?}", entry.code);
        }
    }

    // -- edge cases ----------------------------------------------------------

    #[test]
    fn test_format_report_empty_diagnostics() {
        let cat = build_error_catalog();
        let entry = lookup_error(&cat, &OpenClErrorCode::Timeout).unwrap();
        let report = format_error_report(entry, None);
        assert!(!report.contains("Diagnostics:"));
    }

    #[test]
    fn test_lookup_in_empty_catalog() {
        let empty = ErrorCatalog { entries: vec![] };
        assert!(lookup_error(&empty, &OpenClErrorCode::Timeout).is_none());
    }

    #[test]
    fn test_display_error_code() {
        let code = OpenClErrorCode::DeviceNotFound;
        assert_eq!(format!("{code}"), "CL_DEVICE_NOT_FOUND");
    }

    #[test]
    fn test_display_severity() {
        assert_eq!(format!("{}", ErrorSeverity::Fatal), "FATAL");
    }

    #[test]
    fn test_display_recovery_action() {
        assert_eq!(format!("{}", RecoveryAction::FallbackToCPU), "Fall back to CPU kernels",);
    }
}
