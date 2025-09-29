//! Simple Fixture Loader for Issue #260 Mock Elimination Tests
//!
//! Provides basic fixture loading utilities without complex global state management.
//! This approach is more suitable for test environments with deterministic behavior.

#![allow(dead_code)]

// Re-export fixture modules from parent
#[allow(unused_imports)]
pub use super::crossval;
pub use super::models;
pub use super::quantization;
pub use super::strict_mode;

// Re-export key fixture types
pub use models::qlinear_layer_data::{
    GgufModelFixture, LayerReplacementScenario, QLinearLayerFixture,
};
#[allow(unused_imports)]
pub use quantization::i2s_test_data::{DeviceType, I2SAccuracyFixture, I2STestFixture};
#[allow(unused_imports)]
pub use quantization::tl_lookup_table_data::{MemoryLayoutFixture, TL1TestFixture, TL2TestFixture};
pub use strict_mode::mock_detection_data::{
    MockDetectionFixture, StatisticalAnalysisFixture, StrictModeFixture,
};

// Note: crossval feature not available in bitnet-quantization
// pub use crossval::cpp_reference_data::{CppReferenceFixture, ValidationResult};

/// Test environment configuration
#[derive(Debug, Clone, Default)]
pub struct TestEnvironment {
    pub deterministic_mode: bool,
    pub seed: u64,
    pub strict_mode: bool,
    pub enable_gpu_tests: bool,
    pub enable_crossval: bool,
}

impl TestEnvironment {
    /// Create test environment from environment variables
    pub fn from_env() -> Self {
        Self {
            deterministic_mode: std::env::var("BITNET_DETERMINISTIC").unwrap_or_default() == "1",
            seed: std::env::var("BITNET_SEED").unwrap_or_default().parse().unwrap_or(42),
            strict_mode: std::env::var("BITNET_STRICT_MODE").unwrap_or_default() == "1",
            enable_gpu_tests: cfg!(feature = "gpu")
                && std::env::var("BITNET_STRICT_NO_FAKE_GPU").unwrap_or_default() != "1",
            enable_crossval: false, // crossval feature not available in this crate
        }
    }
}

/// Load all I2S quantization test fixtures
pub fn load_i2s_fixtures() -> Vec<I2STestFixture> {
    quantization::i2s_test_data::load_i2s_cpu_fixtures()
}

/// Load I2S fixtures filtered by device type
pub fn load_i2s_fixtures_by_device(device_type: DeviceType) -> Vec<I2STestFixture> {
    load_i2s_fixtures().into_iter().filter(|f| f.device_type == device_type).collect()
}

/// Load GPU-specific I2S fixtures
#[cfg(feature = "gpu")]
pub fn load_i2s_gpu_fixtures() -> Vec<I2STestFixture> {
    quantization::i2s_test_data::load_i2s_gpu_fixtures()
}

/// Load TL1 lookup table fixtures
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub fn load_tl1_fixtures() -> Vec<TL1TestFixture> {
    quantization::tl_lookup_table_data::load_tl1_neon_fixtures()
}

/// Load TL2 lookup table fixtures
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub fn load_tl2_fixtures() -> Vec<TL2TestFixture> {
    quantization::tl_lookup_table_data::load_tl2_avx_fixtures()
}

/// Load memory layout validation fixtures
pub fn load_memory_layout_fixtures() -> Vec<MemoryLayoutFixture> {
    quantization::tl_lookup_table_data::load_memory_layout_fixtures()
}

/// Load QLinear layer replacement fixtures
pub fn load_qlinear_fixtures() -> Vec<QLinearLayerFixture> {
    models::qlinear_layer_data::load_qlinear_layer_fixtures()
}

/// Load QLinear fixtures filtered by quantization type
pub fn load_qlinear_fixtures_by_type(
    qtype: models::qlinear_layer_data::QuantizationType,
) -> Vec<QLinearLayerFixture> {
    load_qlinear_fixtures().into_iter().filter(|f| f.quantization_type == qtype).collect()
}

/// Load GGUF model test fixtures
pub fn load_gguf_model_fixtures() -> Vec<GgufModelFixture> {
    models::qlinear_layer_data::load_gguf_model_fixtures()
}

/// Load layer replacement test scenarios
pub fn load_layer_replacement_scenarios() -> Vec<LayerReplacementScenario> {
    models::qlinear_layer_data::load_layer_replacement_scenarios()
}

/// Load mock detection test fixtures
pub fn load_mock_detection_fixtures() -> Vec<MockDetectionFixture> {
    strict_mode::mock_detection_data::load_mock_detection_fixtures()
}

/// Load strict mode test fixtures
pub fn load_strict_mode_fixtures() -> Vec<StrictModeFixture> {
    strict_mode::mock_detection_data::load_strict_mode_fixtures()
}

/// Load statistical analysis fixtures
pub fn load_statistical_analysis_fixtures() -> Vec<StatisticalAnalysisFixture> {
    strict_mode::mock_detection_data::load_statistical_analysis_fixtures()
}

// Cross-validation fixtures not available in bitnet-quantization crate
// They would be available in a higher-level crate that includes the crossval feature

/// Get fixture by name (searches across all types)
pub fn find_fixture_by_name(name: &str) -> Option<FixtureInfo> {
    // Search I2S fixtures
    if let Some(fixture) = load_i2s_fixtures().into_iter().find(|f| f.name == name) {
        return Some(FixtureInfo::I2S(fixture));
    }

    // Search QLinear fixtures
    if let Some(fixture) = load_qlinear_fixtures().into_iter().find(|f| f.layer_name == name) {
        return Some(FixtureInfo::QLinear(fixture));
    }

    // Search GGUF model fixtures
    if let Some(fixture) = load_gguf_model_fixtures().into_iter().find(|f| f.model_name == name) {
        return Some(FixtureInfo::GgufModel(fixture));
    }

    // Search mock detection fixtures
    if let Some(fixture) = load_mock_detection_fixtures().into_iter().find(|f| f.test_name == name)
    {
        return Some(FixtureInfo::MockDetection(fixture));
    }

    // Search strict mode fixtures
    if let Some(fixture) = load_strict_mode_fixtures().into_iter().find(|f| f.scenario == name) {
        return Some(FixtureInfo::StrictMode(fixture));
    }

    None
}

/// Unified fixture information
#[derive(Debug, Clone)]
pub enum FixtureInfo {
    I2S(I2STestFixture),
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    TL1(TL1TestFixture),
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    TL2(TL2TestFixture),
    QLinear(QLinearLayerFixture),
    GgufModel(GgufModelFixture),
    MockDetection(MockDetectionFixture),
    StrictMode(StrictModeFixture),
    // CrossVal not available in this crate
}

/// Fixture statistics for reporting
#[derive(Debug, Clone)]
pub struct FixtureStats {
    pub i2s_count: usize,
    pub tl1_count: usize,
    pub tl2_count: usize,
    pub qlinear_count: usize,
    pub gguf_model_count: usize,
    pub mock_detection_count: usize,
    pub strict_mode_count: usize,
    pub crossval_count: usize,
    pub total_count: usize,
}

/// Get statistics about loaded fixtures
pub fn get_fixture_stats() -> FixtureStats {
    let i2s_count = load_i2s_fixtures().len();
    let qlinear_count = load_qlinear_fixtures().len();
    let gguf_model_count = load_gguf_model_fixtures().len();
    let mock_detection_count = load_mock_detection_fixtures().len();
    let strict_mode_count = load_strict_mode_fixtures().len();

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    let tl1_count = load_tl1_fixtures().len();
    #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
    let tl1_count = 0;

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let tl2_count = load_tl2_fixtures().len();
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    let tl2_count = 0;

    let crossval_count = 0; // Not available in this crate

    let total_count = i2s_count
        + tl1_count
        + tl2_count
        + qlinear_count
        + gguf_model_count
        + mock_detection_count
        + strict_mode_count
        + crossval_count;

    FixtureStats {
        i2s_count,
        tl1_count,
        tl2_count,
        qlinear_count,
        gguf_model_count,
        mock_detection_count,
        strict_mode_count,
        crossval_count,
        total_count,
    }
}

/// Print fixture summary to stdout
pub fn print_fixture_summary() {
    let stats = get_fixture_stats();
    println!("BitNet.rs Issue #260 Test Fixture Summary");
    println!("========================================");
    println!("I2S Quantization fixtures: {}", stats.i2s_count);

    if stats.tl1_count > 0 {
        println!("TL1 Lookup Table fixtures: {}", stats.tl1_count);
    }

    if stats.tl2_count > 0 {
        println!("TL2 Lookup Table fixtures: {}", stats.tl2_count);
    }

    println!("QLinear Layer fixtures: {}", stats.qlinear_count);
    println!("GGUF Model fixtures: {}", stats.gguf_model_count);
    println!("Mock Detection fixtures: {}", stats.mock_detection_count);
    println!("Strict Mode fixtures: {}", stats.strict_mode_count);

    if stats.crossval_count > 0 {
        println!("Cross-Validation fixtures: {}", stats.crossval_count);
    }

    println!("Total fixtures: {}", stats.total_count);

    // Environment info
    let env = TestEnvironment::from_env();
    println!("\nEnvironment Configuration:");
    println!("Deterministic mode: {}", env.deterministic_mode);
    println!("Seed: {}", env.seed);
    println!("Strict mode: {}", env.strict_mode);
    println!("GPU tests enabled: {}", env.enable_gpu_tests);
    println!("Cross-validation enabled: {}", env.enable_crossval);
}

/// Validate all fixtures (basic integrity check)
pub fn validate_fixtures() -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    // Validate I2S fixtures
    for fixture in load_i2s_fixtures() {
        if let Err(e) = quantization::i2s_test_data::validate_fixture_integrity(&fixture) {
            errors.push(format!("I2S fixture '{}': {}", fixture.name, e));
        }
    }

    // Validate QLinear fixtures
    for fixture in load_qlinear_fixtures() {
        if let Err(e) = models::qlinear_layer_data::validate_qlinear_fixture(&fixture) {
            errors.push(format!("QLinear fixture '{}': {}", fixture.layer_name, e));
        }
    }

    // Validate TL1 fixtures
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    for fixture in load_tl1_fixtures() {
        if let Err(e) = quantization::tl_lookup_table_data::validate_tl1_fixture(&fixture) {
            errors.push(format!("TL1 fixture '{}': {}", fixture.name, e));
        }
    }

    // Validate TL2 fixtures
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    for fixture in load_tl2_fixtures() {
        if let Err(e) = quantization::tl_lookup_table_data::validate_tl2_fixture(&fixture) {
            errors.push(format!("TL2 fixture '{}': {}", fixture.name, e));
        }
    }

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixture_loading() {
        let i2s_fixtures = load_i2s_fixtures();
        assert!(!i2s_fixtures.is_empty(), "Should have I2S fixtures");

        let qlinear_fixtures = load_qlinear_fixtures();
        assert!(!qlinear_fixtures.is_empty(), "Should have QLinear fixtures");

        let gguf_fixtures = load_gguf_model_fixtures();
        assert!(!gguf_fixtures.is_empty(), "Should have GGUF model fixtures");
    }

    #[test]
    fn test_fixture_stats() {
        let stats = get_fixture_stats();
        assert!(stats.total_count > 0, "Should have fixtures loaded");
        assert!(stats.i2s_count > 0, "Should have I2S fixtures");
        assert!(stats.qlinear_count > 0, "Should have QLinear fixtures");
    }

    #[test]
    fn test_fixture_search() {
        let fixture = find_fixture_by_name("small_embedding_256");
        assert!(fixture.is_some(), "Should find I2S fixture by name");

        let qlinear_fixture = find_fixture_by_name("attention_query_small");
        assert!(qlinear_fixture.is_some(), "Should find QLinear fixture by name");
    }

    #[test]
    fn test_environment_configuration() {
        let env = TestEnvironment::from_env();
        // Basic environment should be valid
        assert!(env.seed > 0, "Seed should be positive");
    }

    #[test]
    fn test_fixture_validation() {
        match validate_fixtures() {
            Ok(()) => println!("All fixtures validated successfully"),
            Err(errors) => {
                println!("Fixture validation errors:");
                for error in errors {
                    println!("  - {}", error);
                }
                panic!("Fixture validation failed");
            }
        }
    }

    #[test]
    fn test_filtered_loading() {
        let cpu_fixtures = load_i2s_fixtures_by_device(DeviceType::Cpu);
        assert!(!cpu_fixtures.is_empty(), "Should have CPU I2S fixtures");

        let i2s_qlinear =
            load_qlinear_fixtures_by_type(models::qlinear_layer_data::QuantizationType::I2S);
        assert!(!i2s_qlinear.is_empty(), "Should have I2S QLinear fixtures");
    }
}
