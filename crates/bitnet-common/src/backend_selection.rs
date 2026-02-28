//! Runtime backend selection and validation.
//!
//! Provides the capability snapshot that answers:
//! "requested X, detected Y, selected Z" — and logs / returns that string
//! so it can be embedded in inference receipts.

use crate::kernel_registry::{KernelBackend, KernelCapabilities};
use std::fmt;

/// Startup summary of what backend was requested, detected, and selected.
///
/// Designed for inclusion in `InferenceReceipt` and startup log output.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BackendStartupSummary {
    /// The backend the user (or config) requested (e.g. `"auto"`, `"cpu"`, `"gpu"`).
    pub requested: String,
    /// Backends detected as available at runtime (e.g. `["cpu-rust"]`).
    pub detected: Vec<String>,
    /// The backend that was ultimately selected (e.g. `"cpu-rust"`).
    pub selected: String,
}

impl BackendStartupSummary {
    /// Construct a new summary from string slices.
    pub fn new(requested: &str, detected: Vec<String>, selected: &str) -> Self {
        Self { requested: requested.to_string(), detected, selected: selected.to_string() }
    }

    /// One-line format suitable for log output.
    ///
    /// Example: `"requested=auto detected=[cpu-rust] selected=cpu-rust"`
    pub fn log_line(&self) -> String {
        format!(
            "requested={} detected=[{}] selected={}",
            self.requested,
            self.detected.join(", "),
            self.selected,
        )
    }
}

/// A user's backend preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendRequest {
    /// Automatically select the best available backend.
    Auto,
    /// Prefer CPU even if GPU is available.
    Cpu,
    /// Require GPU; error if not available.
    Gpu,
    /// Require CUDA specifically.
    Cuda,
    /// Require Intel oneAPI specifically.
    OneApi,
}

impl fmt::Display for BackendRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendRequest::Auto => write!(f, "auto"),
            BackendRequest::Cpu => write!(f, "cpu"),
            BackendRequest::Gpu => write!(f, "gpu"),
            BackendRequest::Cuda => write!(f, "cuda"),
            BackendRequest::OneApi => write!(f, "oneapi"),
        }
    }
}

/// The outcome of backend selection.
#[derive(Debug, Clone)]
pub struct BackendSelectionResult {
    /// What the user requested.
    pub requested: BackendRequest,
    /// What was detected as available.
    pub detected: Vec<KernelBackend>,
    /// What was actually selected.
    pub selected: KernelBackend,
    /// Human-readable rationale for the selection.
    pub rationale: String,
}

impl BackendSelectionResult {
    /// A compact one-line summary for receipts and logs.
    ///
    /// Format: `requested=auto detected=[cuda,cpu-rust] selected=cpu-rust`
    pub fn summary(&self) -> String {
        let detected: Vec<String> = self.detected.iter().map(|b| b.to_string()).collect();
        format!(
            "requested={} detected=[{}] selected={}",
            self.requested,
            detected.join(","),
            self.selected,
        )
    }
}

/// Select the best backend given the request and available capabilities.
///
/// Returns an error if the requested backend is not available.
pub fn select_backend(
    request: BackendRequest,
    caps: &KernelCapabilities,
) -> Result<BackendSelectionResult, BackendSelectionError> {
    let detected = caps.compiled_backends();

    let (selected, rationale) = match request {
        BackendRequest::Auto => {
            let best = caps.best_available().ok_or(BackendSelectionError::NoBackendAvailable)?;
            (best, "auto-selected best available backend".to_string())
        }
        BackendRequest::Cpu => {
            if !caps.cpu_rust {
                return Err(BackendSelectionError::RequestedUnavailable {
                    requested: request,
                    available: detected.clone(),
                });
            }
            (KernelBackend::CpuRust, "CPU explicitly requested".to_string())
        }
        BackendRequest::Gpu => {
            if caps.cuda_compiled && caps.cuda_runtime {
                (KernelBackend::Cuda, "CUDA GPU available and requested".to_string())
            } else if caps.oneapi_compiled && caps.oneapi_runtime {
                (KernelBackend::OneApi, "Intel oneAPI GPU available and requested".to_string())
            } else if caps.cuda_compiled && !caps.cuda_runtime {
                // GPU requested but no runtime — fall back to CPU with warning
                if caps.cpu_rust {
                    (
                        KernelBackend::CpuRust,
                        "CUDA compiled but no GPU runtime detected; falling back to CPU"
                            .to_string(),
                    )
                } else {
                    return Err(BackendSelectionError::RequestedUnavailable {
                        requested: request,
                        available: detected.clone(),
                    });
                }
            } else {
                return Err(BackendSelectionError::RequestedUnavailable {
                    requested: request,
                    available: detected.clone(),
                });
            }
        }
        BackendRequest::Cuda => {
            // Cuda is a strict requirement — no silent fallback to CPU.
            if caps.cuda_compiled && caps.cuda_runtime {
                (KernelBackend::Cuda, "CUDA GPU available and requested".to_string())
            } else {
                return Err(BackendSelectionError::RequestedUnavailable {
                    requested: request,
                    available: detected.clone(),
                });
            }
        }
        BackendRequest::OneApi => {
            if caps.oneapi_compiled && caps.oneapi_runtime {
                (KernelBackend::OneApi, "Intel oneAPI GPU available and requested".to_string())
            } else {
                return Err(BackendSelectionError::RequestedUnavailable {
                    requested: request,
                    available: detected.clone(),
                });
            }
        }
    };

    Ok(BackendSelectionResult { requested: request, detected, selected, rationale })
}

/// Errors from backend selection.
#[derive(Debug)]
pub enum BackendSelectionError {
    /// The requested backend is not compiled or available.
    RequestedUnavailable { requested: BackendRequest, available: Vec<KernelBackend> },
    /// No backend is available at all.
    NoBackendAvailable,
}

impl fmt::Display for BackendSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendSelectionError::RequestedUnavailable { requested, available } => {
                let avail: Vec<String> = available.iter().map(|b| b.to_string()).collect();
                write!(
                    f,
                    "requested backend '{}' is not available; compiled backends: [{}]",
                    requested,
                    avail.join(", ")
                )
            }
            BackendSelectionError::NoBackendAvailable => {
                write!(
                    f,
                    "no kernel backend is compiled; build with --features cpu or --features gpu"
                )
            }
        }
    }
}

impl std::error::Error for BackendSelectionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_registry::{KernelCapabilities, SimdLevel};

    fn cpu_only_caps() -> KernelCapabilities {
        KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: false,
            cuda_runtime: false,
            oneapi_compiled: false,
            oneapi_runtime: false,
            cpp_ffi: false,
            vulkan_compiled: false,
            vulkan_runtime: false,
            simd_level: SimdLevel::Avx2,
        }
    }

    fn cuda_caps() -> KernelCapabilities {
        KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: true,
            cuda_runtime: true,
            oneapi_compiled: false,
            oneapi_runtime: false,
            cpp_ffi: false,
            vulkan_compiled: false,
            vulkan_runtime: false,
            simd_level: SimdLevel::Avx2,
        }
    }

    fn cuda_no_runtime_caps() -> KernelCapabilities {
        KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: true,
            cuda_runtime: false,
            oneapi_compiled: false,
            oneapi_runtime: false,
            cpp_ffi: false,
            vulkan_compiled: false,
            vulkan_runtime: false,
            simd_level: SimdLevel::Avx2,
        }
    }

    #[test]
    fn auto_selects_cpu_when_only_cpu() {
        let result = select_backend(BackendRequest::Auto, &cpu_only_caps()).unwrap();
        assert_eq!(result.selected, KernelBackend::CpuRust);
    }

    #[test]
    fn auto_selects_cuda_when_available() {
        let result = select_backend(BackendRequest::Auto, &cuda_caps()).unwrap();
        assert_eq!(result.selected, KernelBackend::Cuda);
    }

    #[test]
    fn gpu_request_falls_back_to_cpu_when_no_runtime() {
        let result = select_backend(BackendRequest::Gpu, &cuda_no_runtime_caps()).unwrap();
        assert_eq!(result.selected, KernelBackend::CpuRust);
        assert!(result.rationale.contains("falling back to CPU"));
    }

    #[test]
    fn gpu_request_fails_when_no_cuda_compiled() {
        let err = select_backend(BackendRequest::Gpu, &cpu_only_caps()).unwrap_err();
        assert!(matches!(err, BackendSelectionError::RequestedUnavailable { .. }));
        let msg = err.to_string();
        assert!(msg.contains("not available"));
    }

    #[test]
    fn cuda_request_fails_when_no_runtime_available() {
        // BackendRequest::Cuda is strict: no silent CPU fallback
        let err = select_backend(BackendRequest::Cuda, &cuda_no_runtime_caps()).unwrap_err();
        assert!(matches!(err, BackendSelectionError::RequestedUnavailable { .. }));
    }

    #[test]
    fn cuda_request_succeeds_with_full_cuda_caps() {
        let result = select_backend(BackendRequest::Cuda, &cuda_caps()).unwrap();
        assert_eq!(result.selected, KernelBackend::Cuda);
    }

    #[test]
    fn cpu_request_succeeds_with_cpu_caps() {
        let result = select_backend(BackendRequest::Cpu, &cpu_only_caps()).unwrap();
        assert_eq!(result.selected, KernelBackend::CpuRust);
    }

    #[test]
    fn summary_format_is_stable() {
        let result = select_backend(BackendRequest::Auto, &cpu_only_caps()).unwrap();
        let summary = result.summary();
        assert!(summary.contains("requested=auto"), "got: {summary}");
        assert!(summary.contains("selected=cpu-rust"), "got: {summary}");
    }

    fn oneapi_caps() -> KernelCapabilities {
        KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: false,
            cuda_runtime: false,
            oneapi_compiled: true,
            oneapi_runtime: true,
            cpp_ffi: false,
            simd_level: SimdLevel::Avx2,
        }
    }

    #[test]
    fn oneapi_request_selects_oneapi_when_available() {
        let result = select_backend(BackendRequest::OneApi, &oneapi_caps()).unwrap();
        assert_eq!(result.selected, KernelBackend::OneApi);
    }

    #[test]
    fn oneapi_request_fails_without_runtime() {
        let err = select_backend(BackendRequest::OneApi, &cpu_only_caps()).unwrap_err();
        assert!(matches!(err, BackendSelectionError::RequestedUnavailable { .. }));
    }

    #[test]
    fn gpu_request_prefers_oneapi_when_no_cuda() {
        let result = select_backend(BackendRequest::Gpu, &oneapi_caps()).unwrap();
        assert_eq!(result.selected, KernelBackend::OneApi);
    }

    #[test]
    fn no_backend_available_error() {
        let empty_caps = KernelCapabilities {
            cpu_rust: false,
            cuda_compiled: false,
            cuda_runtime: false,
            oneapi_compiled: false,
            oneapi_runtime: false,
            cpp_ffi: false,
            vulkan_compiled: false,
            vulkan_runtime: false,
            simd_level: SimdLevel::Scalar,
        };
        let err = select_backend(BackendRequest::Auto, &empty_caps).unwrap_err();
        assert!(matches!(err, BackendSelectionError::NoBackendAvailable));
    }

    fn oneapi_caps() -> KernelCapabilities {
        KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: false,
            cuda_runtime: false,
            oneapi_compiled: true,
            oneapi_runtime: true,
            cpp_ffi: false,
            simd_level: SimdLevel::Avx2,
        }
    }

    #[test]
    fn oneapi_request_selects_oneapi_when_available() {
        let result = select_backend(BackendRequest::OneApi, &oneapi_caps()).unwrap();
        assert_eq!(result.selected, KernelBackend::OneApi);
    }

    #[test]
    fn oneapi_request_fails_without_runtime() {
        let err = select_backend(BackendRequest::OneApi, &cpu_only_caps()).unwrap_err();
        assert!(matches!(err, BackendSelectionError::RequestedUnavailable { .. }));
    }

    #[test]
    fn gpu_request_prefers_oneapi_when_no_cuda() {
        let result = select_backend(BackendRequest::Gpu, &oneapi_caps()).unwrap();
        assert_eq!(result.selected, KernelBackend::OneApi);
    }
}
