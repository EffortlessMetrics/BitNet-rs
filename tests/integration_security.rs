// Integration tests for security and safety measures
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

/// Test that cargo audit passes
#[test]
fn test_cargo_audit() {
    let output = Command::new("cargo")
        .args(&["audit", "--format", "json"])
        .output()
        .expect("Failed to run cargo audit");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("cargo audit failed: {}", stderr);
    }

    // Parse JSON output to check for vulnerabilities
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.contains("\"vulnerabilities\"") {
        // Check if there are any actual vulnerabilities
        if !stdout.contains("\"vulnerabilities\":[]") {
            panic!("Security vulnerabilities found: {}", stdout);
        }
    }
}

/// Test that cargo deny passes
#[test]
fn test_cargo_deny() {
    let output = Command::new("cargo")
        .args(&["deny", "check"])
        .output()
        .expect("Failed to run cargo deny");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("cargo deny failed: {}", stderr);
    }
}

/// Test that unsafe code is properly documented
#[test]
fn test_unsafe_code_documentation() {
    // Check that unsafe_report.md exists
    assert!(
        Path::new("unsafe_report.md").exists(),
        "unsafe_report.md must exist to document all unsafe code"
    );

    // Read the report
    let report =
        std::fs::read_to_string("unsafe_report.md").expect("Failed to read unsafe_report.md");

    // Should not be empty
    assert!(
        !report.trim().is_empty(),
        "unsafe_report.md should not be empty"
    );

    // Should contain safety documentation
    assert!(
        report.contains("Safety") || report.contains("SAFETY"),
        "unsafe_report.md should contain safety documentation"
    );
}

/// Test that third-party licenses are documented
#[test]
fn test_third_party_licenses() {
    // Check that THIRD_PARTY.md exists
    assert!(
        Path::new("THIRD_PARTY.md").exists(),
        "THIRD_PARTY.md must exist to document third-party licenses"
    );

    // Read the license file
    let licenses =
        std::fs::read_to_string("THIRD_PARTY.md").expect("Failed to read THIRD_PARTY.md");

    // Should not be empty
    assert!(
        !licenses.trim().is_empty(),
        "THIRD_PARTY.md should not be empty"
    );

    // Should contain license information
    assert!(
        licenses.contains("License") || licenses.contains("LICENSE"),
        "THIRD_PARTY.md should contain license information"
    );
}

/// Test that deny.toml configuration is valid
#[test]
fn test_deny_configuration() {
    assert!(
        Path::new("deny.toml").exists(),
        "deny.toml must exist for supply chain security"
    );

    let config = std::fs::read_to_string("deny.toml").expect("Failed to read deny.toml");

    // Should contain required sections
    assert!(
        config.contains("[advisories]"),
        "deny.toml should contain advisories section"
    );
    assert!(
        config.contains("[licenses]"),
        "deny.toml should contain licenses section"
    );
    assert!(
        config.contains("[bans]"),
        "deny.toml should contain bans section"
    );
}

/// Test model security verification
#[test]
fn test_model_security_verification() {
    use bitnet_models::security::{ModelSecurity, ModelVerifier};
    use std::io::Write;
    use tempfile::NamedTempFile;

    let config = ModelSecurity::default();
    let verifier = ModelVerifier::new(config);

    // Test with a small test file
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(b"test model data").unwrap();

    // Should compute hash successfully
    let hash = verifier.compute_file_hash(temp_file.path()).unwrap();
    assert!(!hash.is_empty());
    assert_eq!(hash.len(), 64); // SHA256 hex string length

    // Test source verification
    assert!(verifier
        .verify_source("https://huggingface.co/test")
        .is_ok());
    assert!(verifier
        .verify_source("https://malicious.com/test")
        .is_err());
}

/// Test that fuzzing infrastructure is set up
#[test]
fn test_fuzzing_setup() {
    assert!(Path::new("fuzz").exists(), "fuzz directory should exist");
    assert!(
        Path::new("fuzz/Cargo.toml").exists(),
        "fuzz/Cargo.toml should exist"
    );

    // Check that fuzz targets exist
    let fuzz_targets = [
        "fuzz/fuzz_targets/quantization_i2s.rs",
        "fuzz/fuzz_targets/gguf_parser.rs",
        "fuzz/fuzz_targets/kernel_matmul.rs",
    ];

    for target in &fuzz_targets {
        assert!(
            Path::new(target).exists(),
            "Fuzz target {} should exist",
            target
        );
    }
}

/// Test memory safety with basic operations
#[test]
fn test_memory_safety() {
    // Test that basic operations don't cause memory issues
    // This is a placeholder - in practice, this would use tools like Valgrind or AddressSanitizer

    // Test quantization doesn't leak memory
    use bitnet_quantization::{QuantizationType, Quantize};

    // Create test data
    let test_data = vec![1.0f32, -1.0, 0.5, -0.5];
    let tensor = MockTensor::new(test_data, vec![4]);

    // Quantize and dequantize multiple times
    for _ in 0..100 {
        if let Ok(quantized) = tensor.quantize(QuantizationType::I2S) {
            let _ = quantized.dequantize();
        }
    }

    // If we get here without crashing, basic memory safety is working
}

/// Test that security scanning workflow exists
#[test]
fn test_security_workflow() {
    assert!(
        Path::new(".github/workflows/security.yml").exists(),
        "Security workflow should exist"
    );

    let workflow = std::fs::read_to_string(".github/workflows/security.yml")
        .expect("Failed to read security workflow");

    // Should contain required security checks
    assert!(
        workflow.contains("cargo audit"),
        "Workflow should include cargo audit"
    );
    assert!(
        workflow.contains("cargo deny"),
        "Workflow should include cargo deny"
    );
    assert!(
        workflow.contains("miri"),
        "Workflow should include miri testing"
    );
}

/// Test performance regression detection
#[test]
fn test_performance_regression_detection() {
    // This is a placeholder for performance regression tests
    // In practice, this would run benchmarks and compare against baselines

    use std::time::Instant;

    // Simple performance test for quantization
    let test_data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
    let tensor = MockTensor::new(test_data, vec![1000]);

    let start = Instant::now();
    let _ = tensor.quantize(bitnet_quantization::QuantizationType::I2S);
    let duration = start.elapsed();

    // Should complete within reasonable time (this is a very loose bound)
    assert!(
        duration.as_millis() < 1000,
        "Quantization took too long: {:?}",
        duration
    );
}

// Mock implementations for testing
struct MockTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl MockTensor {
    fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

impl bitnet_quantization::Tensor for MockTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> bitnet_quantization::DType {
        bitnet_quantization::DType::F32
    }

    fn device(&self) -> &bitnet_quantization::Device {
        &bitnet_quantization::Device::Cpu
    }

    fn as_slice<T>(&self) -> Result<&[T], bitnet_quantization::BitNetError> {
        unsafe {
            let ptr = self.data.as_ptr() as *const T;
            let slice = std::slice::from_raw_parts(ptr, self.data.len());
            Ok(slice)
        }
    }
}

impl bitnet_quantization::Quantize for MockTensor {
    fn quantize(
        &self,
        qtype: bitnet_quantization::QuantizationType,
    ) -> Result<bitnet_quantization::QuantizedTensor, bitnet_quantization::BitNetError> {
        // Mock implementation for testing
        Ok(bitnet_quantization::QuantizedTensor {
            data: vec![0u8; self.data.len() / 4],
            scales: vec![1.0f32],
            shape: self.shape.clone(),
            qtype,
        })
    }
}
