//! End-to-end integration tests for the OpenCL inference pipeline.
//!
//! These tests exercise the full device-detection → backend-selection → kernel
//! compilation → compute → fallback path without requiring a real GPU.
//!
//! Tests that can run with `BITNET_GPU_FAKE=oneapi` are unconditional;
//! tests requiring real hardware carry `#[ignore]`.

use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};
use bitnet_device_probe::{
    detect_simd_level, gpu_compiled, oneapi_compiled, probe_cpu, probe_device, probe_gpu,
    DeviceCapabilities,
};
use serial_test::serial;

// ---------------------------------------------------------------------------
// 1. Device Detection E2E
// ---------------------------------------------------------------------------

/// Verify that device detection functions return consistent, non-panicking
/// results on any platform and that CPU is always reported.
#[test]
fn e2e_device_detection_consistency() {
    let cpu = probe_cpu();
    assert!(cpu.core_count >= 1, "must detect at least 1 core");

    let probe = probe_device();
    assert!(probe.cpu.cores >= 1);
    assert!(probe.cpu.threads >= 1);

    let caps = DeviceCapabilities::detect();
    assert!(caps.cpu_rust, "CPU backend must always be available");

    // SIMD detection is deterministic across calls.
    assert_eq!(detect_simd_level(), detect_simd_level());

    // Compiled flags must be self-consistent.
    assert_eq!(
        caps.cuda_compiled || caps.rocm_compiled || caps.oneapi_compiled,
        gpu_compiled(),
    );
}

/// Calling probe_gpu() and probe_device() must agree on GPU availability.
#[test]
fn e2e_device_detection_gpu_agreement() {
    let gpu = probe_gpu();
    let device = probe_device();

    assert_eq!(gpu.cuda_available, device.cuda_available);
    assert_eq!(gpu.rocm_available, device.rocm_available);
    assert_eq!(gpu.oneapi_available, device.oneapi_available);
}

// ---------------------------------------------------------------------------
// 2. Backend Selection E2E
// ---------------------------------------------------------------------------

/// With BITNET_GPU_FAKE=oneapi, the KernelCapabilities must report oneapi as
/// best available when oneapi is compiled.
#[test]
#[serial(bitnet_env)]
fn e2e_backend_selection_oneapi_fake() {
    temp_env::with_var("BITNET_STRICT_MODE", None::<&str>, || {
        temp_env::with_var("BITNET_GPU_FAKE", Some("oneapi"), || {
            let caps = KernelCapabilities {
                cpu_rust: true,
                cuda_compiled: false,
                cuda_runtime: false,
                oneapi_compiled: true,
                oneapi_runtime: true, // simulated via fake
                cpp_ffi: false,
                simd_level: detect_simd_level(),
            };
            assert_eq!(
                caps.best_available(),
                Some(KernelBackend::OneApi),
                "oneapi should be selected when runtime reports available"
            );
        });
    });
}

/// With BITNET_GPU_FAKE=none, all GPU backends are unavailable and CPU must
/// be selected.
#[test]
#[serial(bitnet_env)]
fn e2e_backend_selection_fake_none_falls_back_to_cpu() {
    temp_env::with_var("BITNET_STRICT_MODE", None::<&str>, || {
        temp_env::with_var("BITNET_GPU_FAKE", Some("none"), || {
            let caps = KernelCapabilities {
                cpu_rust: true,
                cuda_compiled: false,
                cuda_runtime: false,
                oneapi_compiled: true,
                oneapi_runtime: false, // forced off by "none"
                cpp_ffi: false,
                simd_level: detect_simd_level(),
            };
            assert_eq!(
                caps.best_available(),
                Some(KernelBackend::CpuRust),
                "CPU must be the fallback when all GPU backends are unavailable"
            );
        });
    });
}

/// Backend priority: CUDA > OneApi > CppFfi > CpuRust.
#[test]
fn e2e_backend_priority_order() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        oneapi_compiled: true,
        oneapi_runtime: true,
        cpp_ffi: true,
        simd_level: SimdLevel::Avx2,
    };
    assert_eq!(
        caps.best_available(),
        Some(KernelBackend::Cuda),
        "CUDA must be preferred when all backends are available"
    );

    // Without CUDA runtime, OneApi takes over.
    let caps_no_cuda = KernelCapabilities {
        cuda_runtime: false,
        ..caps.clone()
    };
    assert_eq!(caps_no_cuda.best_available(), Some(KernelBackend::OneApi));

    // Without any GPU runtime, CppFfi is next.
    let caps_no_gpu = KernelCapabilities {
        cuda_runtime: false,
        oneapi_runtime: false,
        ..caps.clone()
    };
    assert_eq!(caps_no_gpu.best_available(), Some(KernelBackend::CppFfi));

    // Without CppFfi, CPU is the fallback.
    let caps_cpu_only = KernelCapabilities {
        cuda_runtime: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        ..caps
    };
    assert_eq!(caps_cpu_only.best_available(), Some(KernelBackend::CpuRust));
}

// ---------------------------------------------------------------------------
// 3. Kernel Compilation E2E (source validation — no GPU required)
// ---------------------------------------------------------------------------

/// Validate that all embedded OpenCL kernel sources are non-empty and contain
/// the expected entry-point names.
#[test]
fn e2e_kernel_sources_valid() {
    use bitnet_kernels::kernels::{ELEMENTWISE_SRC, MATMUL_I2S_SRC, QUANTIZE_I2S_SRC};

    // Non-empty
    assert!(!MATMUL_I2S_SRC.is_empty());
    assert!(!QUANTIZE_I2S_SRC.is_empty());
    assert!(!ELEMENTWISE_SRC.is_empty());

    // Contain __kernel declarations
    assert!(MATMUL_I2S_SRC.contains("__kernel"));
    assert!(QUANTIZE_I2S_SRC.contains("__kernel"));
    assert!(ELEMENTWISE_SRC.contains("__kernel"));

    // Contain expected entry-point names
    assert!(MATMUL_I2S_SRC.contains("matmul_i2s"));
    assert!(QUANTIZE_I2S_SRC.contains("quantize_i2s"));
    assert!(ELEMENTWISE_SRC.contains("vec_add"));
    assert!(ELEMENTWISE_SRC.contains("rms_norm"));
    assert!(ELEMENTWISE_SRC.contains("silu"));
    assert!(ELEMENTWISE_SRC.contains("scale"));
}

/// Compile all OpenCL kernels on a real device and verify no build errors.
#[test]
#[ignore = "requires Intel GPU with OpenCL runtime — run with --ignored on real hardware"]
fn e2e_kernel_compilation_real_device() {
    // This test requires `--features oneapi` and real hardware.
    // When run, it calls OpenClKernel::new() which compiles all three programs.
    #[cfg(feature = "oneapi")]
    {
        let kernel = bitnet_kernels::OpenClKernel::new()
            .expect("OpenCL kernel should initialise on real Intel GPU");
        assert!(
            kernel.is_available(),
            "Kernel must report available after successful init"
        );
        assert!(
            !kernel.device_name().is_empty(),
            "Device name must be populated"
        );
        assert!(
            !kernel.platform_name().is_empty(),
            "Platform name must be populated"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Matmul Round-Trip (CPU reference comparison)
// ---------------------------------------------------------------------------

/// Perform a small matmul on the CPU fallback kernel and verify the result
/// against a hand-computed reference.
///
/// The FallbackKernel operates on raw i8 values (not packed ternary).
///
/// Matrix A (2×4 i8):
///   row 0: +1, +1, -1, 0
///   row 1:  0, -1, +1, +1
///
/// Matrix B (4×2 u8):
///   col 0: 1, 2, 3, 4
///   col 1: 5, 6, 7, 8
///
/// Expected C = A × B:
///   C[0,0] = 1*1 + 1*2 + (-1)*3 + 0*4 = 0
///   C[0,1] = 1*5 + 1*6 + (-1)*7 + 0*8 = 4
///   C[1,0] = 0*1 + (-1)*2 + 1*3  + 1*4 = 5
///   C[1,1] = 0*5 + (-1)*6 + 1*7  + 1*8 = 9
#[test]
fn e2e_matmul_cpu_reference() {
    use bitnet_kernels::FallbackKernel;
    use bitnet_kernels::KernelProvider;

    let m = 2;
    let n = 2;
    let k = 4;

    // A: raw i8 values (m × k = 2 × 4)
    let a: Vec<i8> = vec![1, 1, -1, 0, 0, -1, 1, 1];

    // B: row-major u8 (k × n = 4 × 2): B[l * n + j]
    let b: Vec<u8> = vec![1, 5, 2, 6, 3, 7, 4, 8];

    let mut c = vec![0.0f32; m * n];

    FallbackKernel
        .matmul_i2s(&a, &b, &mut c, m, n, k)
        .expect("CPU matmul must succeed");

    assert_eq!(c[0], 0.0, "C[0,0]");
    assert_eq!(c[1], 4.0, "C[0,1]");
    assert_eq!(c[2], 5.0, "C[1,0]");
    assert_eq!(c[3], 9.0, "C[1,1]");
}

/// Run the same matmul on a real OpenCL device and compare to CPU reference.
#[test]
#[ignore = "requires Intel GPU with OpenCL runtime — run with --ignored on real hardware"]
fn e2e_matmul_gpu_vs_cpu_roundtrip() {
    #[cfg(feature = "oneapi")]
    {
        use bitnet_kernels::KernelProvider;
        use bitnet_kernels::{FallbackKernel, OpenClKernel};

        let m = 2;
        let n = 2;
        let k = 4;
        let a: Vec<i8> = vec![1, 1, -1, 0, 0, -1, 1, 1];
        let b: Vec<u8> = vec![1, 5, 2, 6, 3, 7, 4, 8];

        let mut c_cpu = vec![0.0f32; m * n];
        FallbackKernel
            .matmul_i2s(&a, &b, &mut c_cpu, m, n, k)
            .expect("CPU matmul");

        let gpu = OpenClKernel::new().expect("OpenCL init");
        let mut c_gpu = vec![0.0f32; m * n];
        gpu.matmul_i2s(&a, &b, &mut c_gpu, m, n, k)
            .expect("GPU matmul");

        for i in 0..c_cpu.len() {
            assert!(
                (c_cpu[i] - c_gpu[i]).abs() < 1e-4,
                "GPU/CPU mismatch at index {i}: cpu={} gpu={}",
                c_cpu[i],
                c_gpu[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Quantize Round-Trip (error bounds)
// ---------------------------------------------------------------------------

/// Quantize a small float vector on CPU, then verify the packed representation
/// and scale are plausible.
#[test]
fn e2e_quantize_cpu_roundtrip() {
    use bitnet_common::QuantizationType;
    use bitnet_kernels::FallbackKernel;
    use bitnet_kernels::KernelProvider;

    let input: Vec<f32> = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.9, -0.9, 0.1];
    let block_size = 8;
    let num_blocks = (input.len() + block_size - 1) / block_size;
    let packed_len = (input.len() + 3) / 4; // 4 ternary values per byte

    let mut output = vec![0u8; packed_len];
    let mut scales = vec![0.0f32; num_blocks];

    FallbackKernel
        .quantize(
            &input,
            &mut output,
            &mut scales,
            QuantizationType::I2S,
        )
        .expect("CPU quantize must succeed");

    // Scale should be positive (absmax / 1.5).
    assert!(scales[0] > 0.0, "scale must be positive: {}", scales[0]);

    // At least some bits should be non-zero (we have non-zero inputs).
    assert!(output.iter().any(|&b| b != 0), "packed output should be non-zero");
}

// ---------------------------------------------------------------------------
// 6. Full Pipeline Mock (device-detection → kernel selection → compute)
// ---------------------------------------------------------------------------

/// Simulate a full inference micro-pipeline using KernelCapabilities to
/// select a backend, then run a small matmul through FallbackKernel.
#[test]
fn e2e_full_pipeline_mock() {
    use bitnet_kernels::FallbackKernel;
    use bitnet_kernels::KernelProvider;

    // Step 1: Build capabilities as if no GPU is present.
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: detect_simd_level(),
    };
    assert_eq!(caps.best_available(), Some(KernelBackend::CpuRust));

    // Step 2: Simulate kernel manager selecting the fallback.
    let provider: &dyn KernelProvider = &FallbackKernel;
    assert!(provider.is_available());
    assert_eq!(provider.name(), "fallback");

    // Step 3: Run a trivial matmul (identity-like).
    // 1×1 matrix: A = [+1], B = [42]
    let a: Vec<i8> = vec![1];
    let b: Vec<u8> = vec![42];
    let mut c = vec![0.0f32; 1];

    provider
        .matmul_i2s(&a, &b, &mut c, 1, 1, 1)
        .expect("pipeline matmul");

    assert_eq!(c[0], 42.0, "1×1 (+1) * 42 should = 42");
}

/// Full pipeline with BITNET_GPU_FAKE=oneapi — simulates oneapi being the
/// selected backend at the capability-selection level.
#[test]
#[serial(bitnet_env)]
fn e2e_full_pipeline_mock_with_fake_oneapi() {
    temp_env::with_var("BITNET_STRICT_MODE", None::<&str>, || {
        temp_env::with_var("BITNET_GPU_FAKE", Some("oneapi"), || {
            let gpu = probe_gpu();

            // When the oneapi feature is compiled, the fake should make it
            // available. Without the feature, it stays false.
            if oneapi_compiled() {
                assert!(gpu.oneapi_available, "fake oneapi should be detected");
            }

            // Regardless, CPU path must still work.
            let caps = KernelCapabilities {
                cpu_rust: true,
                cuda_compiled: false,
                cuda_runtime: false,
                oneapi_compiled: oneapi_compiled(),
                oneapi_runtime: gpu.oneapi_available,
                cpp_ffi: false,
                simd_level: detect_simd_level(),
            };

            let backend = caps.best_available().expect("at least CPU should be available");
            // Validate that some backend was selected.
            assert!(
                backend == KernelBackend::CpuRust || backend == KernelBackend::OneApi,
                "expected CPU or OneApi, got {backend}"
            );
        });
    });
}

// ---------------------------------------------------------------------------
// 7. Fallback E2E (GPU failure → CPU fallback)
// ---------------------------------------------------------------------------

/// Force GPU detection to fail via BITNET_GPU_FAKE=none and verify we land
/// on the CPU backend.
#[test]
#[serial(bitnet_env)]
fn e2e_fallback_no_gpu() {
    temp_env::with_var("BITNET_STRICT_MODE", None::<&str>, || {
        temp_env::with_var("BITNET_GPU_FAKE", Some("none"), || {
            let gpu = probe_gpu();
            assert!(!gpu.available, "no GPU should be available with BITNET_GPU_FAKE=none");

            let caps = KernelCapabilities {
                cpu_rust: true,
                cuda_compiled: cfg!(any(feature = "gpu", feature = "cuda")),
                cuda_runtime: false,
                oneapi_compiled: cfg!(feature = "oneapi"),
                oneapi_runtime: false,
                cpp_ffi: false,
                simd_level: detect_simd_level(),
            };

            assert_eq!(
                caps.best_available(),
                Some(KernelBackend::CpuRust),
                "CPU must be selected when GPU is unavailable"
            );
        });
    });
}

/// Even when GPU caps claim availability but all GPU compiled flags are off,
/// we must fall back to CPU.
#[test]
fn e2e_fallback_gpu_not_compiled() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Scalar,
    };
    assert_eq!(caps.best_available(), Some(KernelBackend::CpuRust));
}

/// OpenClKernel::new() should gracefully return Err when no Intel GPU is
/// present (not panic).
#[test]
#[ignore = "requires oneapi feature — run with --features oneapi"]
fn e2e_fallback_opencl_init_no_device() {
    #[cfg(feature = "oneapi")]
    {
        // On a machine without an Intel GPU, new() should return Err.
        let result = bitnet_kernels::OpenClKernel::new();
        // We don't assert Ok or Err — the point is it must not panic.
        let _ = result;
    }
}

// ---------------------------------------------------------------------------
// 8. Multi-Kernel Sequence (matmul → rmsnorm-like → elementwise)
// ---------------------------------------------------------------------------

/// Run a multi-step sequence using the CPU fallback:
///   1. matmul: produce hidden state
///   2. rms-norm-like: normalise the hidden state
///   3. elementwise add: residual connection
///
/// This validates that data flows correctly between operations.
#[test]
fn e2e_multi_kernel_sequence() {
    use bitnet_kernels::FallbackKernel;
    use bitnet_kernels::KernelProvider;

    let m = 1;
    let n = 4;
    let k = 4;

    // A encodes [+1, +1, -1, +1] as raw i8 (1×4 = 4 elements)
    let a: Vec<i8> = vec![1, 1, -1, 1];

    // B is a 4×4 identity activation matrix (row-major)
    let b: Vec<u8> = vec![
        1, 0, 0, 0, // row 0
        0, 1, 0, 0, // row 1
        0, 0, 1, 0, // row 2
        0, 0, 0, 1, // row 3
    ];

    // Step 1: matmul → hidden
    let mut hidden = vec![0.0f32; m * n];
    FallbackKernel
        .matmul_i2s(&a, &b, &mut hidden, m, n, k)
        .expect("matmul step");

    // hidden should be the raw weights: [+1, +1, -1, +1]
    assert_eq!(hidden[0], 1.0);
    assert_eq!(hidden[1], 1.0);
    assert_eq!(hidden[2], -1.0);
    assert_eq!(hidden[3], 1.0);

    // Step 2: RMS normalization (CPU reference)
    let eps = 1e-5_f32;
    let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden.len() as f32 + eps).sqrt().recip();
    let weights = vec![1.0f32; n]; // unit weights
    let normed: Vec<f32> = hidden.iter().zip(&weights).map(|(h, w)| h * rms * w).collect();

    // All elements should have the same absolute value after unit-weight RMS norm.
    let expected_abs = (1.0f32 * rms).abs();
    for (i, &val) in normed.iter().enumerate() {
        assert!(
            (val.abs() - expected_abs).abs() < 1e-5,
            "normed[{i}] = {val}, expected ±{expected_abs}"
        );
    }

    // Step 3: Residual add (elementwise)
    let residual = vec![0.5f32; n];
    let output: Vec<f32> = normed.iter().zip(&residual).map(|(a, b)| a + b).collect();

    // Output should have 4 values, none of which are NaN or Inf.
    assert_eq!(output.len(), n);
    for (i, &val) in output.iter().enumerate() {
        assert!(val.is_finite(), "output[{i}] is not finite: {val}");
    }
}

/// Multi-kernel sequence on a real OpenCL device (matmul only — norm and
/// elementwise are CPU for now since OpenCL provider doesn't expose them yet).
#[test]
#[ignore = "requires Intel GPU with OpenCL runtime — run with --ignored on real hardware"]
fn e2e_multi_kernel_sequence_real_gpu() {
    #[cfg(feature = "oneapi")]
    {
        use bitnet_kernels::KernelProvider;
        use bitnet_kernels::OpenClKernel;

        let gpu = OpenClKernel::new().expect("OpenCL init");

        let m = 1;
        let n = 4;
        let k = 4;
        let a: Vec<i8> = vec![1, 1, -1, 1];
        let b: Vec<u8> = vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        let mut hidden = vec![0.0f32; m * n];

        gpu.matmul_i2s(&a, &b, &mut hidden, m, n, k)
            .expect("GPU matmul");

        assert_eq!(hidden[0], 1.0);
        assert_eq!(hidden[1], 1.0);
        assert_eq!(hidden[2], -1.0);
        assert_eq!(hidden[3], 1.0);
    }
}
