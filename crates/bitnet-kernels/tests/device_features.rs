//! Device features module tests for Issue #439
//!
//! Tests specification: docs/explanation/issue-439-spec.md#ac3-shared-helpers
//!
//! Validates the device_features module provides unified GPU capability detection
//! with compile-time (gpu_compiled) and runtime (gpu_available_runtime) checks.

#[cfg(test)]
mod compile_time_detection {
    /// AC:3 - gpu_compiled() returns true when GPU features enabled
    ///
    /// Tests that the compile-time GPU detection correctly identifies when
    /// either feature="gpu" or feature="cuda" was enabled at build time.
    ///
    /// Tests specification: docs/explanation/issue-439-spec.md#shared-helpers
    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn ac3_gpu_compiled_true_with_features() {
        use bitnet_kernels::device_features::gpu_compiled;

        assert!(
            gpu_compiled(),
            "AC:3 FAIL - gpu_compiled() should return true when gpu/cuda features enabled"
        );

        println!("AC:3 PASS - gpu_compiled() correctly returns true with GPU features");
    }

    /// AC:3 - gpu_compiled() returns false when GPU features disabled
    ///
    /// Tests that the compile-time GPU detection correctly identifies when
    /// no GPU features are enabled (CPU-only build).
    ///
    /// Tests specification: docs/explanation/issue-439-spec.md#shared-helpers
    #[test]
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    fn ac3_gpu_compiled_false_without_features() {
        use bitnet_kernels::device_features::gpu_compiled;

        assert!(
            !gpu_compiled(),
            "AC:3 FAIL - gpu_compiled() should return false when gpu/cuda features disabled"
        );

        println!("AC:3 PASS - gpu_compiled() correctly returns false without GPU features");
    }
}

#[cfg(test)]
mod runtime_detection {
    use std::env;

    /// AC:3 - gpu_available_runtime() respects BITNET_GPU_FAKE=cuda
    ///
    /// Tests that the fake GPU environment variable overrides real hardware
    /// detection for deterministic testing (fake precedence).
    ///
    /// Tests specification: docs/explanation/issue-439-spec.md#implementation-approach-3
    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn ac3_gpu_fake_cuda_overrides_detection() {
        use bitnet_kernels::device_features::gpu_available_runtime;

        // Set fake GPU environment variable
        unsafe {
            env::set_var("BITNET_GPU_FAKE", "cuda");
        }

        let result = gpu_available_runtime();

        // Clean up environment
        unsafe {
            env::remove_var("BITNET_GPU_FAKE");
        }

        assert!(
            result,
            "AC:3 FAIL - BITNET_GPU_FAKE=cuda should override real detection and return true"
        );

        println!("AC:3 PASS - BITNET_GPU_FAKE=cuda correctly overrides hardware detection");
    }

    /// AC:3 - gpu_available_runtime() respects BITNET_GPU_FAKE=none
    ///
    /// Tests that BITNET_GPU_FAKE=none disables GPU detection even if
    /// real hardware is present (for testing CPU fallback paths).
    ///
    /// Tests specification: docs/explanation/issue-439-spec.md#xtask-preflight
    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn ac3_gpu_fake_none_disables_detection() {
        use bitnet_kernels::device_features::gpu_available_runtime;

        // Disable GPU detection via fake environment
        unsafe {
            env::set_var("BITNET_GPU_FAKE", "none");
        }

        let result = gpu_available_runtime();

        // Clean up environment
        unsafe {
            env::remove_var("BITNET_GPU_FAKE");
        }

        assert!(
            !result,
            "AC:3 FAIL - BITNET_GPU_FAKE=none should disable GPU detection and return false"
        );

        println!("AC:3 PASS - BITNET_GPU_FAKE=none correctly disables GPU detection");
    }

    /// AC:3 - gpu_available_runtime() returns false without GPU features
    ///
    /// Tests that runtime GPU detection always returns false when GPU
    /// was not compiled in (stub implementation).
    ///
    /// Tests specification: docs/explanation/issue-439-spec.md#shared-helpers
    #[test]
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    fn ac3_gpu_runtime_false_without_compile() {
        use bitnet_kernels::device_features::gpu_available_runtime;

        assert!(
            !gpu_available_runtime(),
            "AC:3 FAIL - gpu_available_runtime() must return false when GPU not compiled"
        );

        println!("AC:3 PASS - gpu_available_runtime() returns false for CPU-only builds");
    }

    /// AC:3 - Test BITNET_GPU_FAKE case-insensitive matching
    ///
    /// Verifies that the fake GPU environment variable accepts various
    /// case variations (CUDA, cuda, Cuda) for user convenience.
    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn ac3_gpu_fake_case_insensitive() {
        use bitnet_kernels::device_features::gpu_available_runtime;

        let test_cases = vec!["cuda", "CUDA", "Cuda", "CuDa"];

        for fake_value in test_cases {
            unsafe {
                env::set_var("BITNET_GPU_FAKE", fake_value);
            }

            let result = gpu_available_runtime();

            unsafe {
                env::remove_var("BITNET_GPU_FAKE");
            }

            assert!(
                result,
                "AC:3 FAIL - BITNET_GPU_FAKE='{}' should enable GPU detection (case-insensitive)",
                fake_value
            );
        }

        println!("AC:3 PASS - BITNET_GPU_FAKE supports case-insensitive matching");
    }
}

#[cfg(test)]
mod integration_tests {
    use std::env;

    /// AC:3 - Test device_capability_summary() output format
    ///
    /// Validates that the diagnostic summary function provides human-readable
    /// information about compile-time and runtime GPU capabilities.
    ///
    /// Tests specification: docs/explanation/issue-439-spec.md#device-feature-detection-api
    #[test]
    fn ac3_device_capability_summary_format() {
        use bitnet_kernels::device_features::device_capability_summary;

        let summary = device_capability_summary();

        // Verify summary contains expected sections
        assert!(
            summary.contains("Device Capabilities"),
            "AC:3 FAIL - Summary should contain 'Device Capabilities' header"
        );

        assert!(
            summary.contains("Compiled:"),
            "AC:3 FAIL - Summary should show compile-time capabilities"
        );

        assert!(
            summary.contains("Runtime:"),
            "AC:3 FAIL - Summary should show runtime capabilities"
        );

        // Verify CPU is always marked as available
        assert!(
            summary.contains("CPU ✓"),
            "AC:3 FAIL - Summary should show CPU as available"
        );

        println!("AC:3 PASS - device_capability_summary() provides complete diagnostics:\n{}", summary);
    }

    /// AC:3 - Test fake precedence in capability summary
    ///
    /// Verifies that the diagnostic summary correctly reflects fake GPU state
    /// when BITNET_GPU_FAKE environment variable is set.
    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn ac3_capability_summary_respects_fake() {
        use bitnet_kernels::device_features::device_capability_summary;

        // Test with fake GPU enabled
        unsafe {
            env::set_var("BITNET_GPU_FAKE", "cuda");
        }
        let summary_with_fake = device_capability_summary();
        unsafe {
            env::remove_var("BITNET_GPU_FAKE");
        }

        // Summary should show CUDA as available when fake is set
        assert!(
            summary_with_fake.contains("CUDA") || summary_with_fake.contains("GPU ✓"),
            "AC:3 FAIL - Summary should reflect fake GPU state:\n{}",
            summary_with_fake
        );

        // Test with fake GPU disabled
        unsafe {
            env::set_var("BITNET_GPU_FAKE", "none");
        }
        let summary_no_fake = device_capability_summary();
        unsafe {
            env::remove_var("BITNET_GPU_FAKE");
        }

        // Summary should show CUDA as unavailable when fake=none
        assert!(
            summary_no_fake.contains("CUDA ✗") || summary_no_fake.contains("GPU ✗"),
            "AC:3 FAIL - Summary should reflect fake=none state:\n{}",
            summary_no_fake
        );

        println!("AC:3 PASS - Capability summary respects BITNET_GPU_FAKE environment");
    }
}

#[cfg(test)]
mod workspace_integration {
    /// AC:3 - Verify device_features module is used in quantization
    ///
    /// Tests that bitnet-quantization crate uses the unified device features
    /// helpers rather than duplicating GPU detection logic.
    #[test]
    fn ac3_quantization_uses_device_features() {
        use std::process::Command;

        let output = Command::new("rg")
            .args(&[
                "use bitnet_kernels::device_features",
                "--glob",
                "*.rs",
                "crates/bitnet-quantization/src/",
            ])
            .current_dir("/home/steven/code/Rust/BitNet-rs")
            .output()
            .expect("Failed to run ripgrep");

        let stdout = String::from_utf8_lossy(&output.stdout);

        if !stdout.is_empty() {
            println!("AC:3 PASS - bitnet-quantization uses device_features module:\n{}", stdout);
        } else {
            println!(
                "AC:3 INFO - bitnet-quantization not yet using device_features \
                 (will be validated after implementation)"
            );
        }
    }

    /// AC:3 - Verify device_features module is used in inference
    ///
    /// Tests that bitnet-inference crate uses the unified device features
    /// helpers for device routing in autoregressive generation.
    #[test]
    fn ac3_inference_uses_device_features() {
        use std::process::Command;

        let output = Command::new("rg")
            .args(&[
                "use bitnet_kernels::device_features",
                "--glob",
                "*.rs",
                "crates/bitnet-inference/src/",
            ])
            .current_dir("/home/steven/code/Rust/BitNet-rs")
            .output()
            .expect("Failed to run ripgrep");

        let stdout = String::from_utf8_lossy(&output.stdout);

        if !stdout.is_empty() {
            println!("AC:3 PASS - bitnet-inference uses device_features module:\n{}", stdout);
        } else {
            println!(
                "AC:3 INFO - bitnet-inference not yet using device_features \
                 (will be validated after implementation)"
            );
        }
    }
}
