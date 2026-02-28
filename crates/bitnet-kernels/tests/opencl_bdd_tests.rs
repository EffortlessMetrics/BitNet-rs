//! BDD-style tests for OpenCL device selection and backend behavior.
//!
//! Tests follow Given-When-Then pattern to verify:
//! - Device detection and selection logic
//! - Backend priority (CUDA > OneAPI > CPU)
//! - Fallback behavior when backends are unavailable
//! - Environment variable overrides

use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};
use bitnet_common::Device;

// === Scenario: Device::OpenCL variant behavior ===

mod device_opencl_variant {
    use super::*;

    #[test]
    fn given_opencl_device_when_checked_then_is_opencl_true() {
        // Given an OpenCL device
        let device = Device::OpenCL(0);
        // When we check its type
        // Then it should report as OpenCL
        assert!(device.is_opencl());
        assert!(!device.is_cpu());
        assert!(!device.is_cuda());
    }

    #[test]
    fn given_opencl_device_when_converted_to_candle_then_falls_back_to_cpu() {
        // Given an OpenCL device
        let device = Device::OpenCL(0);
        // When converted to Candle device
        let candle = device.to_candle().unwrap();
        // Then it falls back to CPU (OpenCL uses its own buffer management)
        assert!(device.is_opencl());
        assert_eq!(Device::from(&candle), Device::Cpu);
    }

    #[test]
    fn given_opencl_device_with_different_indices_when_compared_then_distinct() {
        let dev0 = Device::OpenCL(0);
        let dev1 = Device::OpenCL(1);
        assert_ne!(dev0, dev1);
        assert_eq!(dev0, Device::OpenCL(0));
    }
}

// === Scenario: KernelCapabilities with OneAPI ===

mod kernel_capabilities_oneapi {
    use super::*;

    #[test]
    fn given_oneapi_compiled_when_capabilities_checked_then_reflects_feature() {
        let caps = KernelCapabilities::from_compile_time();
        assert_eq!(caps.oneapi_compiled, cfg!(feature = "oneapi"));
    }

    #[test]
    fn given_oneapi_runtime_available_when_capabilities_built_then_shows_runtime() {
        let caps = KernelCapabilities::from_compile_time().with_oneapi_runtime(true);
        assert!(caps.oneapi_runtime);
    }

    #[test]
    fn given_no_oneapi_runtime_when_capabilities_built_then_runtime_false() {
        let caps = KernelCapabilities::from_compile_time().with_oneapi_runtime(false);
        assert!(!caps.oneapi_runtime);
    }

    #[test]
    fn given_oneapi_compiled_when_backends_listed_then_includes_oneapi() {
        let caps = KernelCapabilities::from_compile_time();
        let backends = caps.compiled_backends();
        if cfg!(feature = "oneapi") {
            assert!(
                backends.contains(&KernelBackend::OneApi),
                "oneapi should be in compiled backends when feature enabled"
            );
        }
    }
}

// === Scenario: Backend requires_gpu check ===

mod backend_gpu_requirement {
    use super::*;

    #[test]
    fn given_oneapi_backend_when_checked_then_requires_gpu() {
        assert!(
            KernelBackend::OneApi.requires_gpu(),
            "OneAPI backend should require GPU"
        );
    }

    #[test]
    fn given_cpu_backend_when_checked_then_no_gpu_required() {
        assert!(!KernelBackend::CpuRust.requires_gpu());
    }

    #[test]
    fn given_oneapi_backend_when_display_then_shows_oneapi() {
        assert_eq!(format!("{}", KernelBackend::OneApi), "oneapi");
    }
}

// === Scenario: Backend priority ordering ===

mod backend_priority {
    use super::*;

    #[test]
    fn given_all_backends_compiled_when_listed_then_cuda_before_oneapi() {
        // Simulate all backends compiled
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: true,
            cuda_runtime: false,
            oneapi_compiled: true,
            oneapi_runtime: false,
                        cpp_ffi: false,
            vulkan_compiled: false,
            vulkan_runtime: false,
            simd_level: SimdLevel::Avx2,
        };

        let backends = caps.compiled_backends();
        if let (Some(cuda_pos), Some(oneapi_pos)) = (
            backends.iter().position(|b| *b == KernelBackend::Cuda),
            backends.iter().position(|b| *b == KernelBackend::OneApi),
        ) {
            assert!(
                cuda_pos < oneapi_pos,
                "CUDA should have higher priority than OneAPI"
            );
        }
    }

    #[test]
    fn given_only_oneapi_when_listed_then_oneapi_present() {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: false,
            cuda_runtime: false,
            oneapi_compiled: true,
            oneapi_runtime: false,
                        cpp_ffi: false,
            vulkan_compiled: false,
            vulkan_runtime: false,
            simd_level: SimdLevel::Scalar,
        };

        let backends = caps.compiled_backends();
        assert!(backends.contains(&KernelBackend::OneApi));
        assert!(!backends.contains(&KernelBackend::Cuda));
    }

    #[test]
    fn given_oneapi_runtime_available_when_best_available_then_selects_oneapi() {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: false,
            cuda_runtime: false,
            oneapi_compiled: true,
            oneapi_runtime: true,
                        cpp_ffi: false,
            vulkan_compiled: false,
            vulkan_runtime: false,
            simd_level: SimdLevel::Scalar,
        };

        assert_eq!(caps.best_available(), Some(KernelBackend::OneApi));
    }

    #[test]
    fn given_cuda_and_oneapi_runtime_when_best_available_then_prefers_cuda() {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: true,
            cuda_runtime: true,
            oneapi_compiled: true,
            oneapi_runtime: true,
                        cpp_ffi: false,
            vulkan_compiled: false,
            vulkan_runtime: false,
            simd_level: SimdLevel::Avx2,
        };

        assert_eq!(
            caps.best_available(),
            Some(KernelBackend::Cuda),
            "CUDA should be preferred over OneAPI when both available"
        );
    }

    #[test]
    fn given_no_gpu_runtime_when_best_available_then_falls_back_to_cpu() {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: true,
            cuda_runtime: false,
            oneapi_compiled: true,
            oneapi_runtime: false,
                        cpp_ffi: false,
            vulkan_compiled: false,
            vulkan_runtime: false,
            simd_level: SimdLevel::Avx2,
        };

        assert_eq!(
            caps.best_available(),
            Some(KernelBackend::CpuRust),
            "Should fall back to CPU when no GPU runtime available"
        );
    }
}

// === Scenario: Device serialization ===

mod device_serialization {
    use super::*;

    #[test]
    fn given_opencl_device_when_serialized_then_deserializes_correctly() {
        let device = Device::OpenCL(42);
        let json = serde_json::to_string(&device).unwrap();
        let deserialized: Device = serde_json::from_str(&json).unwrap();
        assert_eq!(device, deserialized);
    }

    #[test]
    fn given_opencl_device_when_debug_formatted_then_readable() {
        let device = Device::OpenCL(0);
        let debug = format!("{:?}", device);
        assert!(debug.contains("OpenCL"));
        assert!(debug.contains("0"));
    }
}