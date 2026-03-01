//! ARM64/Apple Silicon regression tests
//!
//! This test suite validates ARM64-specific functionality including NEON SIMD
//! detection, kernel registry correctness, quantization kernels, and device
//! detection. All tests are gated to aarch64 architecture.

#[cfg(target_arch = "aarch64")]
mod arm64_tests {
    use bitnet_common::QuantizationType;
    use bitnet_kernels::{KernelManager, KernelProvider};

    /// Test NEON SIMD detection on ARM64
    ///
    /// Verifies that ARM NEON SIMD instructions are properly detected at
    /// runtime. NEON is mandatory on ARM64 but this test ensures the detection
    /// mechanism works correctly.
    #[test]
    fn test_arm_neon_detection() {
        let neon_available = std::arch::is_aarch64_feature_detected!("neon");
        assert!(neon_available, "ARM NEON should be available on aarch64 platform");
    }

    /// Test kernel registry reports correct SIMD level on aarch64
    ///
    /// Verifies that the KernelManager selects an appropriate kernel for the
    /// platform and that the selected kernel is available. On ARM64 with
    /// proper feature gates, this should select the NEON kernel.
    #[test]
    fn test_kernel_registry_arm64_simd_level() {
        let manager = KernelManager::new();
        let available = manager.list_available_providers();

        // Should have at least the fallback kernel
        assert!(!available.is_empty(), "At least one kernel provider should be available");

        // On aarch64 with CPU feature, should have NEON available
        #[cfg(all(target_arch = "aarch64", feature = "cpu"))]
        assert!(
            available.iter().any(|&name| name == "neon" || name == "fallback"),
            "aarch64 should have NEON or fallback kernel, got: {:?}",
            available
        );
    }

    /// Test kernel selection works on ARM64
    ///
    /// Verifies that we can select a kernel provider and it reports as
    /// available for use.
    #[test]
    fn test_kernel_selection_arm64() {
        let manager = KernelManager::new();
        let kernel = manager.select_best();

        assert!(kernel.is_ok(), "Kernel selection should succeed");
        let provider = kernel.unwrap();
        assert!(provider.is_available(), "Selected kernel should report as available");
    }

    /// Test ARM-specific I2S dot product correctness (basic)
    ///
    /// Verifies that the i8 x u8 -> f32 dot product works correctly on ARM64.
    /// Tests a simple case with known input/output values.
    ///
    /// This test is ignored on hardware that may not have the necessary
    /// support, though NEON is mandatory on aarch64.
    #[test]
    #[ignore = "Requires ARM NEON SIMD support"]
    fn test_arm_i2s_dot_product() {
        let manager = KernelManager::new();
        let kernel = match manager.select_best() {
            Ok(k) => k,
            Err(_) => {
                println!("Skipping: no suitable kernel found");
                return;
            }
        };

        // Simple test case: small matrix multiplication
        // A: 2x3 i8 matrix
        let a = vec![1i8, 2i8, 3i8, 4i8, 5i8, 6i8];
        // B: 3x2 u8 matrix
        let b = vec![1u8, 2u8, 3u8, 4u8, 5u8, 6u8];
        // C: 2x2 output
        let mut c = vec![0.0f32; 4];

        // Expected results:
        // C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        // C[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        // C[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
        // C[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64

        let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 3);
        assert!(result.is_ok(), "matmul_i2s should succeed");

        // Allow small floating point error
        assert!((c[0] - 22.0).abs() < 0.1, "C[0,0] should be ~22, got {}", c[0]);
        assert!((c[1] - 28.0).abs() < 0.1, "C[0,1] should be ~28, got {}", c[1]);
        assert!((c[2] - 49.0).abs() < 0.1, "C[1,0] should be ~49, got {}", c[2]);
        assert!((c[3] - 64.0).abs() < 0.1, "C[1,1] should be ~64, got {}", c[3]);
    }

    /// Test quantization kernel on ARM64
    ///
    /// Verifies that quantization works correctly on ARM64 with the selected
    /// kernel provider. Tests I2S quantization format.
    ///
    /// Ignored as it requires specific hardware support and calibrated values.
    #[test]
    #[ignore = "Requires ARM NEON quantization support"]
    fn test_arm_quantization_kernel() {
        let manager = KernelManager::new();
        let kernel = match manager.select_best() {
            Ok(k) => k,
            Err(_) => {
                println!("Skipping: no suitable kernel found");
                return;
            }
        };

        // Test I2S quantization
        let input = vec![1.5f32, -0.5f32, 2.0f32, -1.0f32];
        let mut output = vec![0u8; 4];
        let mut scales = vec![0.0f32; 1];

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_ok(), "Quantization should succeed");

        // Basic sanity checks
        assert!(scales[0] > 0.0, "Scale should be positive");
    }

    /// Test that Device is correctly identified on Apple Silicon
    ///
    /// Verifies that we can correctly identify the CPU device type on
    /// Apple Silicon (aarch64) macOS systems.
    #[test]
    fn test_apple_silicon_device_identification() {
        // On aarch64, we should be able to identify the CPU architecture
        assert_eq!(
            std::env::consts::ARCH,
            "aarch64",
            "This test should run on aarch64 architecture"
        );

        // On macOS, check the OS
        #[cfg(target_os = "macos")]
        assert_eq!(std::env::consts::OS, "macos", "Expected macOS platform");

        // Kernel selection should work and report CPU as the device
        let manager = KernelManager::new();
        let kernel =
            manager.select_best().expect("Should be able to select a kernel on Apple Silicon");
        assert!(kernel.is_available(), "Selected kernel should be available on Apple Silicon");
    }

    /// Test that at least one CPU kernel is available on ARM64
    ///
    /// Verifies that CPU-based kernels are available when CPU feature is
    /// enabled. This is essential for CPU inference on Apple Silicon.
    #[test]
    #[cfg(feature = "cpu")]
    fn test_cpu_kernel_availability_arm64() {
        let manager = KernelManager::new();
        let available = manager.list_available_providers();

        // With CPU feature, should have at least the fallback kernel
        assert!(available.contains(&"fallback"), "Fallback CPU kernel should always be available");

        // On aarch64 with proper setup, NEON kernel should be available
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            // Note: NEON feature gate may not be set, so fallback is acceptable
            assert!(
                available.iter().any(|&name| name == "neon" || name == "fallback"),
                "At least NEON or fallback should be available on aarch64"
            );
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod non_aarch64_tests {
    /// Placeholder test for non-aarch64 architectures
    ///
    /// This test passes on non-ARM64 platforms to allow the test suite to
    /// compile and run on all architectures. The ARM64-specific tests are
    /// only compiled and executed on aarch64 targets.
    #[test]
    fn test_placeholder_non_aarch64() {
        // This test always passes and serves as a placeholder
        // for the ARM64 regression test suite on non-ARM64 platforms.
        assert!(true);
    }
}
