//! GPU backend selection for the CLI.
//!
//! Provides the `BackendArg` enum for `--backend` flag parsing and
//! a `--list-backends` helper that prints detected availability.

use std::fmt;

use bitnet_common::BackendRequest;

/// GPU backend selection for the `--backend` CLI flag.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum BackendArg {
    /// Automatically detect best available backend
    #[default]
    Auto,
    /// CPU only (no GPU)
    Cpu,
    /// NVIDIA CUDA
    Cuda,
    /// Intel OpenCL (Arc, Xe)
    Opencl,
    /// Vulkan compute
    Vulkan,
    /// Apple Metal
    Metal,
    /// AMD ROCm/HIP
    Rocm,
    /// WebGPU (experimental)
    Webgpu,
}

impl BackendArg {
    /// Returns `true` if this variant selects a GPU backend.
    #[allow(dead_code)]
    pub fn is_gpu(self) -> bool {
        !matches!(self, Self::Auto | Self::Cpu)
    }

    /// Convert to the existing `BackendRequest` used by the kernel layer.
    pub fn to_backend_request(self) -> BackendRequest {
        match self {
            Self::Auto => BackendRequest::Auto,
            Self::Cpu => BackendRequest::Cpu,
            Self::Cuda => BackendRequest::Cuda,
            Self::Opencl | Self::Vulkan | Self::Webgpu => {
                // These backends are not yet wired into the kernel layer;
                // fall back to the generic GPU request so the selector can
                // return the best available GPU or an informative error.
                BackendRequest::Gpu
            }
            Self::Metal => BackendRequest::Gpu,
            Self::Rocm => BackendRequest::Hip,
        }
    }

    /// All variants in definition order.
    #[allow(dead_code)]
    pub const ALL: &'static [BackendArg] = &[
        Self::Auto,
        Self::Cpu,
        Self::Cuda,
        Self::Opencl,
        Self::Vulkan,
        Self::Metal,
        Self::Rocm,
        Self::Webgpu,
    ];
}

impl fmt::Display for BackendArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda => write!(f, "cuda"),
            Self::Opencl => write!(f, "opencl"),
            Self::Vulkan => write!(f, "vulkan"),
            Self::Metal => write!(f, "metal"),
            Self::Rocm => write!(f, "rocm"),
            Self::Webgpu => write!(f, "webgpu"),
        }
    }
}

/// Describes a single backend's runtime availability.
pub struct BackendStatus {
    pub name: &'static str,
    pub status: &'static str,
}

/// Probe the system and return the availability of every known backend.
pub fn list_backends() -> Vec<BackendStatus> {
    use bitnet_kernels::device_features::{gpu_available_runtime, gpu_compiled};

    let cuda_status = if gpu_compiled() && gpu_available_runtime() {
        "Available"
    } else if gpu_compiled() {
        "Compiled but no runtime detected"
    } else {
        "Not compiled"
    };

    let metal_status = if cfg!(target_os = "macos") {
        if cfg!(feature = "metal") { "Available" } else { "Not compiled" }
    } else {
        "Not available (not macOS)"
    };

    vec![
        BackendStatus { name: "cpu", status: "Always available" },
        BackendStatus { name: "cuda", status: cuda_status },
        BackendStatus { name: "opencl", status: "Not yet implemented" },
        BackendStatus { name: "vulkan", status: "Not yet implemented" },
        BackendStatus { name: "metal", status: metal_status },
        BackendStatus { name: "rocm", status: "Not yet implemented" },
        BackendStatus { name: "webgpu", status: "Experimental (not yet implemented)" },
    ]
}

/// Print the backend availability table to stdout.
pub fn print_backends() {
    println!("Available backends:");
    for b in list_backends() {
        println!("  {:<10}{}", b.name, b.status);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Minimal CLI wrapper for testing `BackendArg` parsing.
    #[derive(Parser, Debug)]
    struct StubCli {
        #[arg(long, default_value = "auto")]
        backend: BackendArg,

        #[arg(long)]
        list_backends: bool,
    }

    fn parse(args: &[&str]) -> StubCli {
        StubCli::try_parse_from(args).expect("parse failed")
    }

    // ── Parsing ─────────────────────────────────────────────────────────

    #[test]
    fn default_is_auto() {
        let cli = parse(&["stub"]);
        assert_eq!(cli.backend, BackendArg::Auto);
    }

    #[test]
    fn parse_auto() {
        let cli = parse(&["stub", "--backend", "auto"]);
        assert_eq!(cli.backend, BackendArg::Auto);
    }

    #[test]
    fn parse_cpu() {
        let cli = parse(&["stub", "--backend", "cpu"]);
        assert_eq!(cli.backend, BackendArg::Cpu);
    }

    #[test]
    fn parse_cuda() {
        let cli = parse(&["stub", "--backend", "cuda"]);
        assert_eq!(cli.backend, BackendArg::Cuda);
    }

    #[test]
    fn parse_opencl() {
        let cli = parse(&["stub", "--backend", "opencl"]);
        assert_eq!(cli.backend, BackendArg::Opencl);
    }

    #[test]
    fn parse_vulkan() {
        let cli = parse(&["stub", "--backend", "vulkan"]);
        assert_eq!(cli.backend, BackendArg::Vulkan);
    }

    #[test]
    fn parse_metal() {
        let cli = parse(&["stub", "--backend", "metal"]);
        assert_eq!(cli.backend, BackendArg::Metal);
    }

    #[test]
    fn parse_rocm() {
        let cli = parse(&["stub", "--backend", "rocm"]);
        assert_eq!(cli.backend, BackendArg::Rocm);
    }

    #[test]
    fn parse_webgpu() {
        let cli = parse(&["stub", "--backend", "webgpu"]);
        assert_eq!(cli.backend, BackendArg::Webgpu);
    }

    #[test]
    fn invalid_backend_rejected() {
        let result = StubCli::try_parse_from(["stub", "--backend", "directx"]);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_backend_empty_rejected() {
        let result = StubCli::try_parse_from(["stub", "--backend", ""]);
        assert!(result.is_err());
    }

    // ── Display ─────────────────────────────────────────────────────────

    #[test]
    fn display_auto() {
        assert_eq!(BackendArg::Auto.to_string(), "auto");
    }

    #[test]
    fn display_cpu() {
        assert_eq!(BackendArg::Cpu.to_string(), "cpu");
    }

    #[test]
    fn display_cuda() {
        assert_eq!(BackendArg::Cuda.to_string(), "cuda");
    }

    #[test]
    fn display_opencl() {
        assert_eq!(BackendArg::Opencl.to_string(), "opencl");
    }

    #[test]
    fn display_metal() {
        assert_eq!(BackendArg::Metal.to_string(), "metal");
    }

    #[test]
    fn display_rocm() {
        assert_eq!(BackendArg::Rocm.to_string(), "rocm");
    }

    #[test]
    fn display_webgpu() {
        assert_eq!(BackendArg::Webgpu.to_string(), "webgpu");
    }

    // ── is_gpu ──────────────────────────────────────────────────────────

    #[test]
    fn auto_is_not_gpu() {
        assert!(!BackendArg::Auto.is_gpu());
    }

    #[test]
    fn cpu_is_not_gpu() {
        assert!(!BackendArg::Cpu.is_gpu());
    }

    #[test]
    fn cuda_is_gpu() {
        assert!(BackendArg::Cuda.is_gpu());
    }

    #[test]
    fn opencl_is_gpu() {
        assert!(BackendArg::Opencl.is_gpu());
    }

    #[test]
    fn vulkan_is_gpu() {
        assert!(BackendArg::Vulkan.is_gpu());
    }

    #[test]
    fn metal_is_gpu() {
        assert!(BackendArg::Metal.is_gpu());
    }

    #[test]
    fn rocm_is_gpu() {
        assert!(BackendArg::Rocm.is_gpu());
    }

    #[test]
    fn webgpu_is_gpu() {
        assert!(BackendArg::Webgpu.is_gpu());
    }

    // ── BackendRequest mapping ──────────────────────────────────────────

    #[test]
    fn to_backend_request_auto() {
        assert_eq!(BackendArg::Auto.to_backend_request(), BackendRequest::Auto);
    }

    #[test]
    fn to_backend_request_cpu() {
        assert_eq!(BackendArg::Cpu.to_backend_request(), BackendRequest::Cpu);
    }

    #[test]
    fn to_backend_request_cuda() {
        assert_eq!(BackendArg::Cuda.to_backend_request(), BackendRequest::Cuda);
    }

    #[test]
    fn to_backend_request_rocm() {
        assert_eq!(BackendArg::Rocm.to_backend_request(), BackendRequest::Hip);
    }

    // ── ALL constant ────────────────────────────────────────────────────

    #[test]
    fn all_has_eight_variants() {
        assert_eq!(BackendArg::ALL.len(), 8);
    }

    // ── Default ─────────────────────────────────────────────────────────

    #[test]
    fn default_trait_is_auto() {
        assert_eq!(BackendArg::default(), BackendArg::Auto);
    }
}
