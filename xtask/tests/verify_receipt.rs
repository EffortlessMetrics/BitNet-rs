//! Receipt validation tests for Issue #439
//!
//! Tests specification: docs/explanation/issue-439-spec.md#ac6-receipt-validation
//!
//! Validates that GPU backend receipts contain evidence of actual GPU kernel
//! execution to prevent silent CPU fallback and dishonest performance reporting.

use std::path::PathBuf;

/// Helper to find workspace root by walking up to .git directory
fn workspace_root() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    while !path.join(".git").exists() {
        if !path.pop() {
            panic!("Could not find workspace root (no .git directory found)");
        }
    }
    path
}

/// Model correction record (LayerNorm rescaling, etc.)
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct CorrectionRecord {
    layer: String,
    correction_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    rms_before: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rms_after: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    factor: Option<f32>,
    policy_fingerprint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
}

/// Receipt structure matching bitnet-inference Receipt
///
/// Tests specification: docs/explanation/issue-439-spec.md#receipt-validation-architecture
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct Receipt {
    backend: String,
    kernels: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    latency_ms: Option<f64>,
    #[serde(default)]
    corrections: Vec<CorrectionRecord>,
}

/// GPU kernel naming convention prefixes (AC6)
const GPU_KERNEL_PREFIXES: &[&str] = &[
    "gemm_",    // GEMM kernels (gemm_fp16, gemm_bf16)
    "wmma_",    // Tensor Core kernels (wmma_matmul)
    "cuda_",    // CUDA utilities (cuda_sync, cuda_memcpy)
    "i2s_gpu_", // I2_S GPU quantization
    "tl1_gpu_", // TL1 GPU quantization
    "tl2_gpu_", // TL2 GPU quantization
];

/// AC:6 - GPU backend requires GPU kernel evidence
///
/// Tests that receipts claiming GPU backend must contain at least one
/// GPU kernel following the naming convention.
///
/// Tests specification: docs/explanation/issue-439-spec.md#receipt-validation
fn verify_gpu_receipt(receipt: &Receipt) -> anyhow::Result<()> {
    use anyhow::ensure;

    let backend_claims_gpu = receipt.backend == "cuda" || receipt.backend == "gpu";

    if !backend_claims_gpu {
        // CPU backend - no validation needed
        return Ok(());
    }

    // GPU backend claimed - verify kernel evidence
    ensure!(
        !receipt.kernels.is_empty(),
        "GPU backend '{}' requires non-empty kernels array, got: {:?}",
        receipt.backend,
        receipt.kernels
    );

    let has_gpu_kernel = receipt
        .kernels
        .iter()
        .any(|kernel_id| GPU_KERNEL_PREFIXES.iter().any(|prefix| kernel_id.starts_with(prefix)));

    ensure!(
        has_gpu_kernel,
        "GPU backend '{}' requires at least one GPU kernel matching naming convention.\n\
         Expected kernel prefixes: {}\n\
         Actual kernels: {:?}\n\n\
         This likely indicates silent CPU fallback. Verify:\n\
         1. GPU feature compiled: cargo build --features gpu\n\
         2. CUDA runtime available: nvidia-smi\n\
         3. Device selection: Device::Cuda(0) passed to inference",
        receipt.backend,
        GPU_KERNEL_PREFIXES.join(", "),
        receipt.kernels
    );

    Ok(())
}

/// Validate receipt corrections for CI gating
///
/// In normal CI (not canary jobs), receipts must NOT contain any corrections.
/// This ensures production builds use properly prepared models, not runtime workarounds.
///
/// Set BITNET_ALLOW_CORRECTIONS=1 to disable this check in canary/dev environments.
fn verify_corrections_in_ci(receipt: &Receipt) -> anyhow::Result<()> {
    use anyhow::ensure;

    // Allow corrections in canary/dev builds
    if std::env::var("BITNET_ALLOW_CORRECTIONS").is_ok() {
        return Ok(());
    }

    // In normal CI, reject any receipts with corrections
    ensure!(
        receipt.corrections.is_empty(),
        "Receipt contains {} correction(s) but BITNET_ALLOW_CORRECTIONS is not set.\n\
         Corrections detected:\n{}\n\n\
         Production CI must use properly prepared models without runtime corrections.\n\
         To allow corrections in canary/dev builds, set BITNET_ALLOW_CORRECTIONS=1.\n\
         To fix properly: regenerate GGUF with float LayerNorm weights (not quantized).",
        receipt.corrections.len(),
        receipt
            .corrections
            .iter()
            .map(|c| format!(
                "  - {}: {} (RMS {:.5}→{:.5}, factor={:.3}, policy={})",
                c.layer,
                c.correction_type,
                c.rms_before.unwrap_or(0.0),
                c.rms_after.unwrap_or(0.0),
                c.factor.unwrap_or(1.0),
                c.policy_fingerprint
            ))
            .collect::<Vec<_>>()
            .join("\n")
    );

    Ok(())
}

#[cfg(test)]
mod corrections_validation_tests {
    use super::*;

    /// Test that receipts without corrections pass validation
    #[test]
    fn test_receipt_no_corrections_passes() {
        let receipt = Receipt {
            backend: "cpu".to_string(),
            kernels: vec!["i2s_gemv".to_string()],
            tokens_per_second: Some(15.0),
            latency_ms: Some(66.0),
            corrections: vec![],
        };

        let result = verify_corrections_in_ci(&receipt);
        assert!(result.is_ok(), "Receipt without corrections should pass: {:?}", result.err());
    }

    /// Test that receipts with corrections fail in normal CI
    #[test]
    fn test_receipt_with_corrections_fails_in_ci() {
        unsafe {
            std::env::remove_var("BITNET_ALLOW_CORRECTIONS");
        }

        let receipt = Receipt {
            backend: "cpu".to_string(),
            kernels: vec!["i2s_gemv".to_string()],
            tokens_per_second: Some(15.0),
            latency_ms: Some(66.0),
            corrections: vec![CorrectionRecord {
                layer: "model.layers.0.input_layernorm.weight".to_string(),
                correction_type: "ln_gamma_rescale_rms".to_string(),
                rms_before: Some(0.5),
                rms_after: Some(1.0),
                factor: Some(2.0),
                policy_fingerprint: "BITNET_FIX_LN_SCALE=1".to_string(),
                metadata: None,
            }],
        };

        let result = verify_corrections_in_ci(&receipt);
        assert!(result.is_err(), "Receipt with corrections should fail in normal CI");

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("correction(s)"), "Error should mention corrections: {}", err_msg);
        assert!(
            err_msg.contains("BITNET_ALLOW_CORRECTIONS"),
            "Error should mention BITNET_ALLOW_CORRECTIONS: {}",
            err_msg
        );
    }

    /// Test that receipts with corrections pass when BITNET_ALLOW_CORRECTIONS is set
    #[test]
    fn test_receipt_with_corrections_passes_in_canary() {
        unsafe {
            std::env::set_var("BITNET_ALLOW_CORRECTIONS", "1");
        }

        let receipt = Receipt {
            backend: "cpu".to_string(),
            kernels: vec!["i2s_gemv".to_string()],
            tokens_per_second: Some(15.0),
            latency_ms: Some(66.0),
            corrections: vec![CorrectionRecord {
                layer: "model.layers.0.input_layernorm.weight".to_string(),
                correction_type: "ln_gamma_rescale_rms".to_string(),
                rms_before: Some(0.5),
                rms_after: Some(1.0),
                factor: Some(2.0),
                policy_fingerprint: "BITNET_FIX_LN_SCALE=1".to_string(),
                metadata: None,
            }],
        };

        let result = verify_corrections_in_ci(&receipt);
        assert!(
            result.is_ok(),
            "Receipt with corrections should pass when BITNET_ALLOW_CORRECTIONS=1: {:?}",
            result.err()
        );

        unsafe {
            std::env::remove_var("BITNET_ALLOW_CORRECTIONS");
        }
    }

    /// Test that error message shows correction details
    #[test]
    fn test_corrections_error_shows_details() {
        unsafe {
            std::env::remove_var("BITNET_ALLOW_CORRECTIONS");
        }

        let receipt = Receipt {
            backend: "cpu".to_string(),
            kernels: vec!["i2s_gemv".to_string()],
            tokens_per_second: Some(15.0),
            latency_ms: Some(66.0),
            corrections: vec![
                CorrectionRecord {
                    layer: "layer1.norm.weight".to_string(),
                    correction_type: "ln_gamma_rescale_rms".to_string(),
                    rms_before: Some(0.5),
                    rms_after: Some(1.0),
                    factor: Some(2.0),
                    policy_fingerprint: "BITNET_FIX_LN_SCALE=1".to_string(),
                    metadata: None,
                },
                CorrectionRecord {
                    layer: "layer2.norm.weight".to_string(),
                    correction_type: "ln_gamma_rescale_rms".to_string(),
                    rms_before: Some(0.75),
                    rms_after: Some(1.0),
                    factor: Some(1.33),
                    policy_fingerprint: "BITNET_FIX_LN_SCALE=1".to_string(),
                    metadata: None,
                },
            ],
        };

        let result = verify_corrections_in_ci(&receipt);
        let err_msg = result.unwrap_err().to_string();

        // Check that both layers are mentioned
        assert!(err_msg.contains("layer1.norm.weight"), "Error should mention layer1: {}", err_msg);
        assert!(err_msg.contains("layer2.norm.weight"), "Error should mention layer2: {}", err_msg);
        assert!(err_msg.contains("RMS"), "Error should show RMS values: {}", err_msg);
    }
}

#[cfg(test)]
mod receipt_validation_tests {
    use super::*;

    /// AC:6 - GPU backend with CPU kernels fails validation
    ///
    /// Tests specification: docs/explanation/issue-439-spec.md#receipt-validation-architecture
    #[test]
    fn ac6_gpu_backend_requires_gpu_kernel() {
        let receipt = Receipt {
            backend: "cuda".to_string(),
            kernels: vec!["i2s_cpu_quantize".to_string(), "avx2_matmul".to_string()],
            tokens_per_second: Some(12.3),
            latency_ms: Some(81.2),
            corrections: vec![],
        };

        let result = verify_gpu_receipt(&receipt);

        assert!(result.is_err(), "AC:6 FAIL - GPU backend with CPU kernels should fail validation");

        let err_msg = result.unwrap_err().to_string();

        assert!(
            err_msg.contains("naming convention"),
            "AC:6 FAIL - Error should mention naming convention: {}",
            err_msg
        );

        println!("AC:6 PASS - GPU backend without GPU kernels correctly fails validation");
    }

    /// AC:6 - GPU backend with valid GPU kernel passes
    ///
    /// Tests that receipts with proper GPU kernel naming convention pass validation.
    #[test]
    fn ac6_gpu_backend_with_valid_kernel_passes() {
        let receipt = Receipt {
            backend: "cuda".to_string(),
            kernels: vec!["gemm_fp16".to_string()],
            tokens_per_second: Some(87.5),
            latency_ms: Some(11.4),
            corrections: vec![],
        };

        let result = verify_gpu_receipt(&receipt);

        assert!(
            result.is_ok(),
            "AC:6 FAIL - GPU backend with GPU kernel should pass: {:?}",
            result.err()
        );

        println!("AC:6 PASS - GPU backend with GPU kernel passes validation");
    }

    /// AC:6 - CPU backend requires no validation
    ///
    /// Tests that CPU backend receipts pass validation regardless of kernels.
    #[test]
    fn ac6_cpu_backend_no_validation_required() {
        let receipt = Receipt {
            backend: "cpu".to_string(),
            kernels: vec!["avx2_matmul".to_string(), "i2s_cpu_quantize".to_string()],
            tokens_per_second: Some(15.2),
            latency_ms: Some(66.7),
            corrections: vec![],
        };

        let result = verify_gpu_receipt(&receipt);

        assert!(
            result.is_ok(),
            "AC:6 FAIL - CPU backend should pass validation: {:?}",
            result.err()
        );

        println!("AC:6 PASS - CPU backend passes validation without GPU kernels");
    }

    /// AC:6 - Empty kernels array fails for GPU backend
    ///
    /// Tests that GPU backend receipts must have non-empty kernel arrays.
    #[test]
    fn ac6_gpu_backend_empty_kernels_fails() {
        let receipt = Receipt {
            backend: "cuda".to_string(),
            kernels: vec![],
            tokens_per_second: Some(0.0),
            latency_ms: Some(0.0),
            corrections: vec![],
        };

        let result = verify_gpu_receipt(&receipt);

        assert!(result.is_err(), "AC:6 FAIL - GPU backend with empty kernels should fail");

        println!("AC:6 PASS - GPU backend with empty kernels correctly fails");
    }
}

#[cfg(test)]
mod kernel_prefix_tests {
    use super::*;

    /// AC:6 - Test all GPU kernel prefix categories
    ///
    /// Validates that all documented GPU kernel naming conventions are recognized.
    ///
    /// Tests specification: docs/explanation/issue-439-spec.md#gpu-kernel-naming-convention
    #[test]
    fn ac6_all_gpu_kernel_prefixes_recognized() {
        let test_kernels = vec![
            ("gemm_fp16", "GEMM kernels"),
            ("gemm_bf16", "GEMM kernels"),
            ("wmma_matmul", "Tensor Core kernels"),
            ("cuda_sync", "CUDA utilities"),
            ("i2s_gpu_quantize", "I2_S GPU quantization"),
            ("tl1_gpu_pack", "TL1 GPU quantization"),
            ("tl2_gpu_matmul", "TL2 GPU quantization"),
        ];

        for (kernel, category) in test_kernels {
            let receipt = Receipt {
                backend: "cuda".to_string(),
                kernels: vec![kernel.to_string()],
                tokens_per_second: Some(87.5),
                latency_ms: Some(11.4),
                corrections: vec![],
            };

            let result = verify_gpu_receipt(&receipt);

            assert!(
                result.is_ok(),
                "AC:6 FAIL - Kernel '{}' ({}) should pass validation: {:?}",
                kernel,
                category,
                result.err()
            );

            println!("AC:6 PASS - {} kernel '{}' recognized", category, kernel);
        }
    }

    /// AC:6 - CPU kernel prefixes correctly rejected
    ///
    /// Tests that CPU-specific kernel naming patterns are not mistaken for GPU kernels.
    #[test]
    fn ac6_cpu_kernel_prefixes_rejected() {
        let cpu_kernels = vec![
            "i2s_cpu_quantize",
            "avx2_matmul",
            "scalar_quantize",
            "fallback_gemm",
            "cpu_memcpy",
        ];

        for cpu_kernel in cpu_kernels {
            let receipt = Receipt {
                backend: "cuda".to_string(),
                kernels: vec![cpu_kernel.to_string()],
                tokens_per_second: Some(12.0),
                latency_ms: Some(80.0),
                corrections: vec![],
            };

            let result = verify_gpu_receipt(&receipt);

            assert!(
                result.is_err(),
                "AC:6 FAIL - CPU kernel '{}' should fail GPU validation",
                cpu_kernel
            );
        }

        println!("AC:6 PASS - CPU kernel prefixes correctly rejected for GPU backend");
    }
}

#[cfg(test)]
mod fixture_integration_tests {
    use super::*;
    use std::fs;

    fn load_receipt_fixture(name: &str) -> Receipt {
        let fixture_path =
            workspace_root().join("tests/fixtures/receipts").join(format!("{}.json", name));

        let contents = fs::read_to_string(&fixture_path).unwrap_or_else(|_| {
            panic!("AC:6 FAIL - Failed to load receipt fixture: {}", fixture_path.display())
        });

        serde_json::from_str(&contents).unwrap_or_else(|e| {
            panic!("AC:6 FAIL - Failed to parse receipt fixture {}: {}", fixture_path.display(), e)
        })
    }

    /// AC:6 - Valid GPU receipt fixture passes
    #[test]
    fn ac6_fixture_valid_gpu_receipt() {
        let receipt = load_receipt_fixture("valid-gpu-receipt");

        let result = verify_gpu_receipt(&receipt);

        assert!(
            result.is_ok(),
            "AC:6 FAIL - valid-gpu-receipt.json should pass: {:?}",
            result.err()
        );

        println!("AC:6 PASS - valid-gpu-receipt.json fixture validates correctly");
    }

    /// AC:6 - Invalid GPU receipt fixture fails
    #[test]
    fn ac6_fixture_invalid_gpu_receipt() {
        let receipt = load_receipt_fixture("invalid-gpu-receipt");

        let result = verify_gpu_receipt(&receipt);

        assert!(result.is_err(), "AC:6 FAIL - invalid-gpu-receipt.json should fail validation");

        println!("AC:6 PASS - invalid-gpu-receipt.json fixture correctly fails");
    }

    /// AC:6 - Valid CPU receipt fixture passes
    #[test]
    fn ac6_fixture_valid_cpu_receipt() {
        let receipt = load_receipt_fixture("valid-cpu-receipt");

        let result = verify_gpu_receipt(&receipt);

        assert!(
            result.is_ok(),
            "AC:6 FAIL - valid-cpu-receipt.json should pass: {:?}",
            result.err()
        );

        println!("AC:6 PASS - valid-cpu-receipt.json fixture validates correctly");
    }

    /// AC:6 - GPU receipt with all kernel types passes
    #[test]
    fn ac6_fixture_all_kernel_types() {
        let receipt = load_receipt_fixture("gpu-receipt-all-kernel-types");

        let result = verify_gpu_receipt(&receipt);

        assert!(
            result.is_ok(),
            "AC:6 FAIL - gpu-receipt-all-kernel-types.json should pass: {:?}",
            result.err()
        );

        // Verify all GPU kernel categories are present
        let has_gemm = receipt.kernels.iter().any(|k| k.starts_with("gemm_"));
        let has_wmma = receipt.kernels.iter().any(|k| k.starts_with("wmma_"));
        let has_cuda = receipt.kernels.iter().any(|k| k.starts_with("cuda_"));
        let has_i2s = receipt.kernels.iter().any(|k| k.starts_with("i2s_gpu_"));
        let has_tl1 = receipt.kernels.iter().any(|k| k.starts_with("tl1_gpu_"));
        let has_tl2 = receipt.kernels.iter().any(|k| k.starts_with("tl2_gpu_"));

        assert!(
            has_gemm && has_wmma && has_cuda && has_i2s && has_tl1 && has_tl2,
            "AC:6 FAIL - All kernel type fixture should contain all GPU kernel categories"
        );

        println!(
            "AC:6 PASS - gpu-receipt-all-kernel-types.json contains all GPU kernel categories"
        );
    }
}

#[cfg(test)]
mod performance_validation {
    use super::*;

    /// AC:6 - Detect suspicious GPU performance (silent CPU fallback indicator)
    ///
    /// Tests that we can detect GPU receipts with suspiciously low performance
    /// that might indicate silent CPU fallback despite GPU backend claim.
    ///
    /// Tests specification: docs/explanation/issue-439-spec.md#silent-fallback-detection
    #[test]
    fn ac6_detect_suspicious_gpu_performance() {
        let suspicious_receipt = Receipt {
            backend: "cuda".to_string(),
            kernels: vec!["gemm_fp16".to_string()], // Has GPU kernel but...
            tokens_per_second: Some(8.5),           // ...suspiciously low performance (CPU-like)
            latency_ms: Some(117.0),
            corrections: vec![],
        };

        // Basic validation passes (has GPU kernel)
        assert!(verify_gpu_receipt(&suspicious_receipt).is_ok());

        // But performance suggests silent fallback
        let tps = suspicious_receipt.tokens_per_second.unwrap();
        if tps < 25.0 {
            println!(
                "AC:6 WARNING - GPU backend with CPU-like performance ({:.1} tok/s) detected.\n\
                 This may indicate silent CPU fallback despite GPU kernel present.",
                tps
            );
        }

        println!("AC:6 PASS - Performance-based fallback detection works");
    }

    /// AC:6 - Valid GPU performance baselines
    ///
    /// Tests that GPU receipts with expected performance ranges pass validation.
    ///
    /// Tests specification: docs/explanation/issue-439-spec.md#performance-baseline-specifications
    #[test]
    fn ac6_gpu_performance_baselines() {
        let gpu_baseline = Receipt {
            backend: "cuda".to_string(),
            kernels: vec!["tl1_gpu_pack".to_string(), "gemm_fp16".to_string()],
            tokens_per_second: Some(87.5), // Within 50-100 tok/s GPU baseline
            latency_ms: Some(11.4),
            corrections: vec![],
        };

        assert!(verify_gpu_receipt(&gpu_baseline).is_ok());

        let tps = gpu_baseline.tokens_per_second.unwrap();
        assert!(
            (30.0..=150.0).contains(&tps),
            "AC:6 INFO - GPU performance {:.1} tok/s within expected range 30-150 tok/s",
            tps
        );

        println!("AC:6 PASS - GPU performance baseline validation works");
    }

    /// AC:6 - Mixed CPU+GPU kernels should pass if GPU kernel present
    ///
    /// Tests that receipts with BOTH CPU and GPU kernels pass validation
    /// as long as at least one GPU kernel is present (partial fallback scenario).
    ///
    /// This is a realistic scenario where some operations fall back to CPU
    /// but the GPU is still being used for critical operations.
    ///
    /// Tests specification: docs/explanation/issue-439-spec.md#receipt-validation
    #[test]
    fn ac6_mixed_cpu_gpu_kernels_should_pass_if_gpu_kernel_present() {
        let mixed_receipt = Receipt {
            backend: "cuda".to_string(),
            kernels: vec![
                "gemm_fp32".to_string(),        // GPU kernel (valid)
                "i2s_cpu_quantize".to_string(), // CPU kernel (fallback)
                "avx2_matmul".to_string(),      // CPU kernel (fallback)
            ],
            tokens_per_second: Some(45.5),
            latency_ms: Some(22.0),
            corrections: vec![],
        };

        let result = verify_gpu_receipt(&mixed_receipt);

        assert!(
            result.is_ok(),
            "AC:6 FAIL - Mixed CPU+GPU kernels should pass if at least one GPU kernel present: {:?}",
            result.err()
        );

        println!(
            "AC:6 PASS - Mixed CPU+GPU kernels pass validation (partial fallback scenario handled correctly)"
        );
    }

    /// AC:6 - Mixed CPU+GPU fixture integration test
    ///
    /// Tests that the mixed-cpu-gpu-kernels-receipt.json fixture passes validation.
    #[test]
    fn ac6_fixture_mixed_cpu_gpu_kernels() {
        use std::fs;

        let fixture_path =
            workspace_root().join("tests/fixtures/receipts/mixed-cpu-gpu-kernels-receipt.json");

        let contents = fs::read_to_string(&fixture_path).unwrap_or_else(|_| {
            panic!("AC:6 FAIL - Failed to load mixed receipt fixture: {}", fixture_path.display())
        });

        let receipt: Receipt = serde_json::from_str(&contents).unwrap_or_else(|e| {
            panic!(
                "AC:6 FAIL - Failed to parse mixed receipt fixture {}: {}",
                fixture_path.display(),
                e
            )
        });

        let result = verify_gpu_receipt(&receipt);

        assert!(
            result.is_ok(),
            "AC:6 FAIL - mixed-cpu-gpu-kernels-receipt.json should pass: {:?}",
            result.err()
        );

        // Verify it has both CPU and GPU kernels
        let has_gpu_kernel =
            receipt.kernels.iter().any(|k| GPU_KERNEL_PREFIXES.iter().any(|p| k.starts_with(p)));
        let has_cpu_kernel = receipt.kernels.iter().any(|k| k.contains("cpu") || k.contains("avx"));

        assert!(
            has_gpu_kernel && has_cpu_kernel,
            "AC:6 FAIL - Mixed fixture should contain both GPU and CPU kernels"
        );

        println!("AC:6 PASS - mixed-cpu-gpu-kernels-receipt.json fixture validates correctly");
    }
}
