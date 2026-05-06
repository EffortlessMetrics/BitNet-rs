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
    /// Require AMD HIP specifically.
    Hip,
    /// Require Intel oneAPI specifically.
    OneApi,
    /// Require native Metal compute without assuming a specific Apple machine.
    Metal,
    /// Require MPSGraph graph execution without treating it as native Metal kernels.
    MpsGraph,
    /// Require the Apple M4 native Metal lane.
    AppleM4Metal,
    /// Require the Apple M4 MPSGraph graph/reference lane.
    AppleM4MpsGraph,
    /// Require the Apple M4 CPU/NEON fallback/parity lane.
    AppleM4CpuNeon,
}

impl BackendRequest {
    /// Parse a CLI/config backend label without collapsing Apple proof lanes.
    pub fn from_label(label: &str) -> Option<Self> {
        match label.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(BackendRequest::Auto),
            "cpu" => Some(BackendRequest::Cpu),
            "gpu" => Some(BackendRequest::Gpu),
            "cuda" => Some(BackendRequest::Cuda),
            "hip" | "rocm" => Some(BackendRequest::Hip),
            "oneapi" => Some(BackendRequest::OneApi),
            "metal" => Some(BackendRequest::Metal),
            "mpsgraph" => Some(BackendRequest::MpsGraph),
            "apple-m4-metal" => Some(BackendRequest::AppleM4Metal),
            "apple-m4-mpsgraph" => Some(BackendRequest::AppleM4MpsGraph),
            "apple-m4-cpu-neon" => Some(BackendRequest::AppleM4CpuNeon),
            _ => None,
        }
    }
}

impl fmt::Display for BackendRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendRequest::Auto => write!(f, "auto"),
            BackendRequest::Cpu => write!(f, "cpu"),
            BackendRequest::Gpu => write!(f, "gpu"),
            BackendRequest::Cuda => write!(f, "cuda"),
            BackendRequest::Hip => write!(f, "hip"),
            BackendRequest::OneApi => write!(f, "oneapi"),
            BackendRequest::Metal => write!(f, "metal"),
            BackendRequest::MpsGraph => write!(f, "mpsgraph"),
            BackendRequest::AppleM4Metal => write!(f, "apple-m4-metal"),
            BackendRequest::AppleM4MpsGraph => write!(f, "apple-m4-mpsgraph"),
            BackendRequest::AppleM4CpuNeon => write!(f, "apple-m4-cpu-neon"),
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

    /// Requested backend label for receipt/log fields.
    pub fn requested_backend(&self) -> String {
        self.requested.to_string()
    }

    /// Selected backend label for receipt/log fields.
    pub fn selected_backend(&self) -> String {
        match (self.requested, self.selected) {
            (BackendRequest::AppleM4CpuNeon, KernelBackend::CpuRust) => {
                "apple-m4-cpu-neon".to_string()
            }
            _ => self.selected.to_string(),
        }
    }

    /// Runtime API implied by the selected backend label.
    pub fn runtime_api(&self) -> &'static str {
        match self.selected_backend().as_str() {
            "cuda" => "cuda",
            "hip" => "hip",
            "oneapi" => "oneapi",
            "opencl" => "opencl",
            "apple-m4-metal" | "metal" => "metal",
            "apple-m4-mpsgraph" | "mpsgraph" => "mpsgraph",
            _ => "cpu",
        }
    }

    /// Whether backend selection changed the requested backend identity.
    pub fn fallback_used(&self) -> bool {
        match self.requested {
            BackendRequest::Auto => false,
            BackendRequest::Cpu => self.selected != KernelBackend::CpuRust,
            BackendRequest::Gpu => !self.selected.requires_gpu(),
            BackendRequest::Cuda => self.selected != KernelBackend::Cuda,
            BackendRequest::Hip => self.selected != KernelBackend::Hip,
            BackendRequest::OneApi => self.selected != KernelBackend::OneApi,
            BackendRequest::Metal
            | BackendRequest::MpsGraph
            | BackendRequest::AppleM4Metal
            | BackendRequest::AppleM4MpsGraph
            | BackendRequest::AppleM4CpuNeon => self.requested_backend() != self.selected_backend(),
        }
    }

    /// Human-readable fallback reason, when fallback happened.
    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback_used().then_some(self.rationale.as_str())
    }

    /// Receipt-oriented one-line identity summary for logs.
    pub fn identity_summary(&self) -> String {
        let fallback_reason = self
            .fallback_reason()
            .map(|reason| format!(" fallback_reason={reason}"))
            .unwrap_or_default();
        format!(
            "requested_backend={} selected_backend={} runtime_api={} fallback_used={}{}",
            self.requested_backend(),
            self.selected_backend(),
            self.runtime_api(),
            self.fallback_used(),
            fallback_reason,
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
            } else if caps.hip_compiled && caps.hip_runtime {
                (KernelBackend::Hip, "AMD HIP GPU available and requested".to_string())
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
        BackendRequest::Hip => {
            if caps.hip_compiled && caps.hip_runtime {
                (KernelBackend::Hip, "AMD HIP GPU available and requested".to_string())
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
        BackendRequest::Metal | BackendRequest::AppleM4Metal => {
            return Err(BackendSelectionError::RequestedUnavailable {
                requested: request,
                available: detected.clone(),
            });
        }
        BackendRequest::MpsGraph | BackendRequest::AppleM4MpsGraph => {
            return Err(BackendSelectionError::RequestedUnavailable {
                requested: request,
                available: detected.clone(),
            });
        }
        BackendRequest::AppleM4CpuNeon => {
            if caps.cpu_rust
                && matches!(caps.simd_level, crate::kernel_registry::SimdLevel::Neon)
                && cfg!(all(target_os = "macos", target_arch = "aarch64"))
            {
                (KernelBackend::CpuRust, "Apple M4 CPU/NEON lane requested".to_string())
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
            hip_compiled: false,
            hip_runtime: false,
            oneapi_compiled: false,
            oneapi_runtime: false,
            opencl_compiled: false,
            opencl_runtime: false,
            cpp_ffi: false,
            simd_level: SimdLevel::Avx2,
        }
    }

    fn cuda_caps() -> KernelCapabilities {
        KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: true,
            cuda_runtime: true,
            hip_compiled: false,
            hip_runtime: false,
            oneapi_compiled: false,
            oneapi_runtime: false,
            opencl_compiled: false,
            opencl_runtime: false,
            cpp_ffi: false,
            simd_level: SimdLevel::Avx2,
        }
    }

    fn cuda_no_runtime_caps() -> KernelCapabilities {
        KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: true,
            cuda_runtime: false,
            hip_compiled: false,
            hip_runtime: false,
            oneapi_compiled: false,
            oneapi_runtime: false,
            opencl_compiled: false,
            opencl_runtime: false,
            cpp_ffi: false,
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

    #[test]
    fn apple_backend_labels_parse_without_aliasing() {
        assert_eq!(BackendRequest::from_label("metal"), Some(BackendRequest::Metal));
        assert_eq!(
            BackendRequest::from_label("apple-m4-metal"),
            Some(BackendRequest::AppleM4Metal)
        );
        assert_eq!(BackendRequest::from_label("mpsgraph"), Some(BackendRequest::MpsGraph));
        assert_eq!(
            BackendRequest::from_label("apple-m4-mpsgraph"),
            Some(BackendRequest::AppleM4MpsGraph)
        );
        assert_eq!(
            BackendRequest::from_label("apple-m4-cpu-neon"),
            Some(BackendRequest::AppleM4CpuNeon)
        );
    }

    #[test]
    fn apple_m4_metal_request_is_strict_until_probe_work_lands() {
        let err = select_backend(BackendRequest::AppleM4Metal, &cpu_only_caps()).unwrap_err();
        assert!(matches!(err, BackendSelectionError::RequestedUnavailable { .. }));
        assert!(err.to_string().contains("apple-m4-metal"));
    }

    #[test]
    fn identity_summary_records_fallback_status() {
        let result = select_backend(BackendRequest::Gpu, &cuda_no_runtime_caps()).unwrap();
        let summary = result.identity_summary();
        assert!(summary.contains("requested_backend=gpu"), "got: {summary}");
        assert!(summary.contains("selected_backend=cpu-rust"), "got: {summary}");
        assert!(summary.contains("runtime_api=cpu"), "got: {summary}");
        assert!(summary.contains("fallback_used=true"), "got: {summary}");
    }

    #[test]
    fn no_backend_available_error() {
        let empty_caps = KernelCapabilities {
            cpu_rust: false,
            cuda_compiled: false,
            cuda_runtime: false,
            hip_compiled: false,
            hip_runtime: false,
            oneapi_compiled: false,
            oneapi_runtime: false,
            opencl_compiled: false,
            opencl_runtime: false,
            cpp_ffi: false,
            simd_level: SimdLevel::Scalar,
        };
        let err = select_backend(BackendRequest::Auto, &empty_caps).unwrap_err();
        assert!(matches!(err, BackendSelectionError::NoBackendAvailable));
    }
}
