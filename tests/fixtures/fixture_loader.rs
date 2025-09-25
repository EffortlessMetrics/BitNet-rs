//! Fixture loading utilities with proper feature gates and workspace paths for BitNet.rs
//!
//! Provides centralized loading utilities for all test fixtures with support for
//! deterministic testing, feature-gated compilation, and workspace-aware paths.

use crate::fixtures::{
    cross_validation_data::{CrossValidationFixtures, CrossValidationTestCase},
    network_mocks::{NetworkMockFixtures, NetworkTestScenario},
    quantization_test_vectors::{QuantizationFixtures, QuantizationTestVector},
    tokenizer_fixtures::{TokenizerFixtures, TokenizerTestFixture, TokenizerType},
};
use bitnet_common::{BitNetError, QuantizationType, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use tokio::fs;

/// Global fixture loader instance for test coordination
static FIXTURE_LOADER: LazyLock<Mutex<FixtureLoader>> =
    LazyLock::new(|| Mutex::new(FixtureLoader::new()));

/// Main fixture loader with centralized management
pub struct FixtureLoader {
    pub fixtures_dir: PathBuf,
    pub tokenizer_fixtures: Option<TokenizerFixtures>,
    pub quantization_fixtures: Option<QuantizationFixtures>,
    pub crossval_fixtures: Option<CrossValidationFixtures>,
    pub network_fixtures: Option<NetworkMockFixtures>,
    pub initialized: bool,
    pub deterministic_mode: bool,
    pub seed: Option<u64>,
}

/// Configuration for fixture loading behavior
#[derive(Debug, Clone)]
pub struct FixtureConfig {
    pub fixtures_directory: PathBuf,
    pub enable_deterministic: bool,
    pub seed: Option<u64>,
    pub force_regenerate: bool,
    pub feature_gates: Vec<String>,
    pub test_tier: TestTier,
}

/// Test execution tier for fixture selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestTier {
    Fast,     // Mock data only, minimal fixtures
    Standard, // Mix of mock and small real fixtures
    Full,     // Complete fixture set including large files
}

/// Fixture loading result with metadata
#[derive(Debug, Clone)]
pub struct FixtureLoadResult<T> {
    pub data: T,
    pub source: FixtureSource,
    pub load_time_ms: u64,
    pub file_size_bytes: Option<u64>,
}

/// Source of fixture data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureSource {
    Static,    // Compiled-in static data
    Generated, // Dynamically generated
    Cached,    // Loaded from disk cache
    Network,   // Downloaded (not used in tests, but tracked)
}

impl FixtureLoader {
    /// Create new fixture loader with default configuration
    pub fn new() -> Self {
        Self {
            fixtures_dir: Self::default_fixtures_dir(),
            tokenizer_fixtures: None,
            quantization_fixtures: None,
            crossval_fixtures: None,
            network_fixtures: None,
            initialized: false,
            deterministic_mode: std::env::var("BITNET_DETERMINISTIC").is_ok(),
            seed: std::env::var("BITNET_SEED").ok().and_then(|s| s.parse().ok()),
        }
    }

    /// Get default fixtures directory based on workspace structure
    fn default_fixtures_dir() -> PathBuf {
        if let Ok(workspace_root) = std::env::var("CARGO_MANIFEST_DIR") {
            PathBuf::from(workspace_root).join("tests").join("fixtures")
        } else {
            // Fallback for when running from different contexts
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
        }
    }

    /// Initialize all fixtures with given configuration
    pub async fn initialize(&mut self, config: FixtureConfig) -> Result<()> {
        if self.initialized && !config.force_regenerate {
            return Ok(());
        }

        self.fixtures_dir = config.fixtures_directory.clone();
        self.deterministic_mode = config.enable_deterministic;
        self.seed = config.seed.or(self.seed);

        // Create fixtures directory if it doesn't exist
        fs::create_dir_all(&self.fixtures_dir).await.map_err(BitNetError::Io)?;

        // Initialize fixtures based on feature gates and test tier
        self.initialize_tokenizer_fixtures(&config).await?;
        self.initialize_quantization_fixtures(&config).await?;
        self.initialize_crossval_fixtures(&config).await?;
        self.initialize_network_fixtures(&config).await?;

        self.initialized = true;
        Ok(())
    }

    /// Initialize tokenizer fixtures
    async fn initialize_tokenizer_fixtures(&mut self, config: &FixtureConfig) -> Result<()> {
        let fixtures = TokenizerFixtures::new();

        // Write fixture files based on test tier
        match config.test_tier {
            TestTier::Fast => {
                // Only create minimal mock files
                self.create_minimal_tokenizer_fixtures(&fixtures).await?;
            }
            TestTier::Standard | TestTier::Full => {
                // Create complete fixture set
                fixtures.write_all_fixtures().await?;
            }
        }

        self.tokenizer_fixtures = Some(fixtures);
        Ok(())
    }

    /// Initialize quantization fixtures
    async fn initialize_quantization_fixtures(&mut self, config: &FixtureConfig) -> Result<()> {
        let fixtures = QuantizationFixtures::new();

        // Write binary fixtures for performance testing
        if config.test_tier != TestTier::Fast {
            fixtures.write_binary_fixtures(&self.fixtures_dir).await?;
        }

        self.quantization_fixtures = Some(fixtures);
        Ok(())
    }

    /// Initialize cross-validation fixtures
    async fn initialize_crossval_fixtures(&mut self, config: &FixtureConfig) -> Result<()> {
        let fixtures = CrossValidationFixtures::new();

        // Write cross-validation data files
        if config.feature_gates.contains(&"crossval".to_string()) {
            fixtures.write_crossval_data(&self.fixtures_dir).await?;
        }

        self.crossval_fixtures = Some(fixtures);
        Ok(())
    }

    /// Initialize network mock fixtures
    async fn initialize_network_fixtures(&mut self, _config: &FixtureConfig) -> Result<()> {
        let fixtures = NetworkMockFixtures::new();

        // Write network mock data
        fixtures.write_network_mocks(&self.fixtures_dir).await.map_err(|e| {
            BitNetError::Configuration(format!("Failed to write network mocks: {}", e))
        })?;

        self.network_fixtures = Some(fixtures);
        Ok(())
    }

    /// Create minimal tokenizer fixtures for fast testing
    async fn create_minimal_tokenizer_fixtures(&self, fixtures: &TokenizerFixtures) -> Result<()> {
        let tokenizers_dir = self.fixtures_dir.join("tokenizers");
        fs::create_dir_all(&tokenizers_dir).await.map_err(BitNetError::Io)?;

        // Create minimal LLaMA-3 tokenizer
        let minimal_llama3 = serde_json::json!({
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": {
                    "<|begin_of_text|>": 128000,
                    "<|end_of_text|>": 128001,
                    "Hello": 9906,
                    "world": 1917
                },
                "merges": ["H e", "l l", "o o"]
            }
        });

        fs::write(
            tokenizers_dir.join("minimal_llama3_tokenizer.json"),
            serde_json::to_string_pretty(&minimal_llama3).unwrap(),
        )
        .await
        .map_err(BitNetError::Io)?;

        Ok(())
    }

    /// Load tokenizer fixture by type with proper error handling
    pub fn load_tokenizer_fixture(
        &self,
        tokenizer_type: TokenizerType,
    ) -> Result<FixtureLoadResult<&TokenizerTestFixture>> {
        let start = std::time::Instant::now();

        let fixtures = self.tokenizer_fixtures.as_ref().ok_or_else(|| {
            BitNetError::Configuration("Tokenizer fixtures not initialized".to_string())
        })?;

        let fixture = fixtures.get_fixture(&tokenizer_type).ok_or_else(|| {
            BitNetError::Configuration(format!(
                "Fixture not found for tokenizer type: {:?}",
                tokenizer_type
            ))
        })?;

        Ok(FixtureLoadResult {
            data: fixture,
            source: FixtureSource::Static,
            load_time_ms: start.elapsed().as_millis() as u64,
            file_size_bytes: None,
        })
    }

    /// Load quantization test vectors for specific type and vocab size
    pub fn load_quantization_vectors(
        &self,
        quant_type: QuantizationType,
        vocab_size: Option<u32>,
    ) -> Result<FixtureLoadResult<Vec<QuantizationTestVector>>> {
        let start = std::time::Instant::now();

        let fixtures = self.quantization_fixtures.as_ref().ok_or_else(|| {
            BitNetError::Configuration("Quantization fixtures not initialized".to_string())
        })?;

        let vectors = if let Some(vocab_size) = vocab_size {
            fixtures
                .get_vocab_compatible_vectors(vocab_size)
                .into_iter()
                .filter(|v| v.quantization_type == quant_type)
                .cloned()
                .collect()
        } else if let Some(vectors) = fixtures.get_test_vectors(&quant_type) {
            vectors.clone()
        } else {
            return Err(BitNetError::Configuration(format!(
                "No test vectors found for quantization type: {:?}",
                quant_type
            )));
        };

        Ok(FixtureLoadResult {
            data: vectors,
            source: FixtureSource::Static,
            load_time_ms: start.elapsed().as_millis() as u64,
            file_size_bytes: None,
        })
    }

    /// Load cross-validation test cases for specific architecture
    pub fn load_crossval_cases(
        &self,
        architecture: &str,
    ) -> Result<FixtureLoadResult<Vec<&CrossValidationTestCase>>> {
        let start = std::time::Instant::now();

        let fixtures = self.crossval_fixtures.as_ref().ok_or_else(|| {
            BitNetError::Configuration("Cross-validation fixtures not initialized".to_string())
        })?;

        let cases = fixtures.get_cases_for_architecture(architecture);

        if cases.is_empty() {
            return Err(BitNetError::Configuration(format!(
                "No cross-validation cases found for architecture: {}",
                architecture
            )));
        }

        Ok(FixtureLoadResult {
            data: cases,
            source: FixtureSource::Static,
            load_time_ms: start.elapsed().as_millis() as u64,
            file_size_bytes: None,
        })
    }

    /// Load network test scenario
    pub fn load_network_scenario(
        &self,
        scenario_type: &str,
    ) -> Result<FixtureLoadResult<NetworkTestScenario>> {
        let start = std::time::Instant::now();

        let fixtures = self.network_fixtures.as_ref().ok_or_else(|| {
            BitNetError::Configuration("Network fixtures not initialized".to_string())
        })?;

        let scenario = fixtures.create_test_scenario(scenario_type);

        Ok(FixtureLoadResult {
            data: scenario,
            source: FixtureSource::Generated,
            load_time_ms: start.elapsed().as_millis() as u64,
            file_size_bytes: None,
        })
    }

    /// Generate deterministic test data with current seed
    pub fn generate_deterministic_data<T>(
        &self,
        generator: impl Fn(u64) -> T,
    ) -> Result<FixtureLoadResult<T>> {
        let start = std::time::Instant::now();

        if !self.deterministic_mode {
            return Err(BitNetError::Configuration(
                "Deterministic mode not enabled. Set BITNET_DETERMINISTIC=1".to_string(),
            ));
        }

        let seed = self.seed.unwrap_or(42);
        let data = generator(seed);

        Ok(FixtureLoadResult {
            data,
            source: FixtureSource::Generated,
            load_time_ms: start.elapsed().as_millis() as u64,
            file_size_bytes: None,
        })
    }

    /// Load GGUF model fixture by vocabulary size
    pub async fn load_gguf_fixture(&self, vocab_size: u32) -> Result<FixtureLoadResult<Vec<u8>>> {
        let start = std::time::Instant::now();

        let fixtures = self.tokenizer_fixtures.as_ref().ok_or_else(|| {
            BitNetError::Configuration("Tokenizer fixtures not initialized".to_string())
        })?;

        let model = fixtures.get_gguf_model_by_vocab(vocab_size).ok_or_else(|| {
            BitNetError::Configuration(format!(
                "No GGUF model found for vocab size: {}",
                vocab_size
            ))
        })?;

        // Check if file exists, create if needed
        let model_path = &model.file_path;
        if !model_path.exists() {
            fs::create_dir_all(model_path.parent().unwrap()).await.map_err(BitNetError::Io)?;
            fs::write(model_path, &model.file_content).await.map_err(BitNetError::Io)?;
        }

        let file_content = fs::read(model_path).await.map_err(BitNetError::Io)?;
        let file_size = file_content.len() as u64;

        Ok(FixtureLoadResult {
            data: file_content,
            source: FixtureSource::Cached,
            load_time_ms: start.elapsed().as_millis() as u64,
            file_size_bytes: Some(file_size),
        })
    }

    /// Check if fixtures are initialized and ready
    pub fn is_ready(&self) -> bool {
        self.initialized
            && self.tokenizer_fixtures.is_some()
            && self.quantization_fixtures.is_some()
            && self.crossval_fixtures.is_some()
            && self.network_fixtures.is_some()
    }

    /// Get fixture statistics for debugging
    pub fn get_fixture_stats(&self) -> FixtureStats {
        FixtureStats {
            tokenizer_fixtures_count: self
                .tokenizer_fixtures
                .as_ref()
                .map(|f| f.fixtures.len())
                .unwrap_or(0),
            quantization_vectors_count: self
                .quantization_fixtures
                .as_ref()
                .map(|f| f.test_vectors.values().map(|v| v.len()).sum())
                .unwrap_or(0),
            crossval_cases_count: self
                .crossval_fixtures
                .as_ref()
                .map(|f| f.llama3_cases.len() + f.llama2_cases.len())
                .unwrap_or(0),
            network_scenarios_count: self
                .network_fixtures
                .as_ref()
                .map(|f| f.error_scenarios.len())
                .unwrap_or(0),
            total_disk_usage_bytes: self.calculate_disk_usage(),
            initialized: self.initialized,
        }
    }

    /// Calculate total disk usage of fixture files
    fn calculate_disk_usage(&self) -> u64 {
        // This would be implemented to scan the fixtures directory
        // For now, return estimated size based on loaded fixtures
        let mut total_size = 0u64;

        if let Some(_tokenizer_fixtures) = &self.tokenizer_fixtures {
            total_size += 50_000_000; // ~50MB for tokenizer files
        }

        if let Some(_quant_fixtures) = &self.quantization_fixtures {
            total_size += 10_000_000; // ~10MB for quantization vectors
        }

        if let Some(_crossval_fixtures) = &self.crossval_fixtures {
            total_size += 5_000_000; // ~5MB for cross-validation data
        }

        if let Some(_network_fixtures) = &self.network_fixtures {
            total_size += 2_000_000; // ~2MB for network mocks
        }

        total_size
    }

    /// Cleanup all fixture files
    pub async fn cleanup(&mut self) -> Result<()> {
        if !self.initialized {
            return Ok(());
        }

        // Remove generated fixture files (but keep static ones)
        let fixtures_dir = &self.fixtures_dir;

        // Clean up GGUF models
        let gguf_dir = fixtures_dir.join("gguf_models");
        if gguf_dir.exists() {
            fs::remove_dir_all(&gguf_dir).await.map_err(BitNetError::Io)?;
        }

        // Clean up binary quantization files
        let quant_dir = fixtures_dir.join("quantization");
        if quant_dir.exists() {
            let mut read_dir = fs::read_dir(&quant_dir).await.map_err(BitNetError::Io)?;
            while let Some(entry) = read_dir.next_entry().await.map_err(BitNetError::Io)? {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("bin") {
                    fs::remove_file(&path).await.map_err(BitNetError::Io)?;
                }
            }
        }

        self.initialized = false;
        Ok(())
    }
}

/// Fixture statistics for debugging and monitoring
#[derive(Debug, Clone)]
pub struct FixtureStats {
    pub tokenizer_fixtures_count: usize,
    pub quantization_vectors_count: usize,
    pub crossval_cases_count: usize,
    pub network_scenarios_count: usize,
    pub total_disk_usage_bytes: u64,
    pub initialized: bool,
}

/// Global fixture loader access functions

/// Initialize global fixture loader with configuration
pub async fn initialize_fixtures(config: FixtureConfig) -> Result<()> {
    let mut loader = FIXTURE_LOADER.lock().unwrap();
    loader.initialize(config).await
}

/// Load tokenizer fixture using global loader
pub fn load_tokenizer_fixture(
    tokenizer_type: TokenizerType,
) -> Result<FixtureLoadResult<&'static TokenizerTestFixture>> {
    let loader = FIXTURE_LOADER.lock().unwrap();
    // Note: This is a simplified version - in practice, we'd need to handle static lifetime properly
    loader.load_tokenizer_fixture(tokenizer_type).map(|result| FixtureLoadResult {
        data: unsafe { std::mem::transmute(result.data) }, // Unsafe but necessary for static lifetime
        source: result.source,
        load_time_ms: result.load_time_ms,
        file_size_bytes: result.file_size_bytes,
    })
}

/// Load quantization vectors using global loader
pub fn load_quantization_vectors(
    quant_type: QuantizationType,
    vocab_size: Option<u32>,
) -> Result<FixtureLoadResult<Vec<QuantizationTestVector>>> {
    let loader = FIXTURE_LOADER.lock().unwrap();
    loader.load_quantization_vectors(quant_type, vocab_size)
}

/// CPU-specific fixture loading utilities
#[cfg(feature = "cpu")]
pub mod cpu_fixtures {
    use super::*;

    pub async fn initialize_cpu_fixtures() -> Result<()> {
        let config = FixtureConfig {
            fixtures_directory: FixtureLoader::default_fixtures_dir(),
            enable_deterministic: true,
            seed: Some(42),
            force_regenerate: false,
            feature_gates: vec!["cpu".to_string()],
            test_tier: TestTier::Standard,
        };

        initialize_fixtures(config).await
    }

    pub fn load_cpu_quantization_data() -> Result<Vec<QuantizationTestVector>> {
        let result = load_quantization_vectors(QuantizationType::I2S, Some(32000))?;
        Ok(result.data)
    }

    pub fn load_simd_test_data() -> Vec<([f32; 8], [i8; 8], f32)> {
        vec![
            ([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -1.5, 0.5], [-1, -1, 0, 1, 1, 1, -1, 0], 2.0),
            ([1.2, -1.8, 2.5, -0.3, 0.7, -2.1, 1.9, -0.6], [1, -1, 1, 0, 0, -1, 1, 0], 2.5),
        ]
    }
}

/// GPU-specific fixture loading utilities
#[cfg(feature = "gpu")]
pub mod gpu_fixtures {
    use super::*;

    pub async fn initialize_gpu_fixtures() -> Result<()> {
        let config = FixtureConfig {
            fixtures_directory: FixtureLoader::default_fixtures_dir(),
            enable_deterministic: true,
            seed: Some(42),
            force_regenerate: false,
            feature_gates: vec!["gpu".to_string()],
            test_tier: TestTier::Full,
        };

        initialize_fixtures(config).await
    }

    pub fn load_gpu_quantization_data() -> Result<Vec<QuantizationTestVector>> {
        let result = load_quantization_vectors(QuantizationType::I2S, Some(128256))?;
        Ok(result.data)
    }

    pub async fn load_mixed_precision_fixtures() -> Result<FixtureLoadResult<Vec<u8>>> {
        let loader = FIXTURE_LOADER.lock().unwrap();
        loader.load_gguf_fixture(128256).await
    }
}

/// FFI bridge fixture loading utilities
#[cfg(feature = "ffi")]
pub mod ffi_fixtures {
    use super::*;

    pub async fn initialize_ffi_fixtures() -> Result<()> {
        let config = FixtureConfig {
            fixtures_directory: FixtureLoader::default_fixtures_dir(),
            enable_deterministic: true,
            seed: Some(42),
            force_regenerate: false,
            feature_gates: vec!["ffi".to_string(), "crossval".to_string()],
            test_tier: TestTier::Full,
        };

        initialize_fixtures(config).await
    }

    pub fn load_ffi_crossval_cases() -> Result<Vec<CrossValidationTestCase>> {
        let loader = FIXTURE_LOADER.lock().unwrap();
        let result = loader.load_crossval_cases("BitNet-b1.58")?;
        Ok(result.data.into_iter().cloned().collect())
    }
}

/// Test utilities for fixture management
#[cfg(test)]
pub mod test_utilities {
    use super::*;

    /// Create test configuration for fast testing
    pub fn create_fast_test_config() -> FixtureConfig {
        FixtureConfig {
            fixtures_directory: std::env::temp_dir().join("bitnet_test_fixtures"),
            enable_deterministic: true,
            seed: Some(12345),
            force_regenerate: true,
            feature_gates: vec!["cpu".to_string()],
            test_tier: TestTier::Fast,
        }
    }

    /// Create test configuration for comprehensive testing
    pub fn create_full_test_config() -> FixtureConfig {
        FixtureConfig {
            fixtures_directory: FixtureLoader::default_fixtures_dir(),
            enable_deterministic: true,
            seed: Some(42),
            force_regenerate: false,
            feature_gates: vec![
                "cpu".to_string(),
                "gpu".to_string(),
                "ffi".to_string(),
                "crossval".to_string(),
            ],
            test_tier: TestTier::Full,
        }
    }

    /// Initialize fixtures for testing with cleanup
    pub async fn with_test_fixtures<F, R>(config: FixtureConfig, test_fn: F) -> Result<R>
    where
        F: Fn() -> R,
    {
        initialize_fixtures(config).await?;

        let result = test_fn();

        // Cleanup
        let mut loader = FIXTURE_LOADER.lock().unwrap();
        loader.cleanup().await?;

        Ok(result)
    }
}

/// Default initialization for common test scenarios
impl Default for FixtureConfig {
    fn default() -> Self {
        let tier = if std::env::var("CI").is_ok() {
            TestTier::Fast // Use fast fixtures in CI
        } else {
            TestTier::Standard // Use standard fixtures locally
        };

        Self {
            fixtures_directory: FixtureLoader::default_fixtures_dir(),
            enable_deterministic: std::env::var("BITNET_DETERMINISTIC").is_ok(),
            seed: std::env::var("BITNET_SEED").ok().and_then(|s| s.parse().ok()),
            force_regenerate: false,
            feature_gates: Self::detect_available_features(),
            test_tier: tier,
        }
    }
}

impl FixtureConfig {
    /// Detect available features based on compilation flags
    fn detect_available_features() -> Vec<String> {
        #[allow(unused_mut)] // Conditional compilation may not use mutation
        let mut features = vec!["cpu".to_string()]; // CPU always available

        #[cfg(feature = "gpu")]
        features.push("gpu".to_string());

        #[cfg(feature = "ffi")]
        features.push("ffi".to_string());

        #[cfg(feature = "spm")]
        features.push("smp".to_string());

        #[cfg(feature = "crossval")]
        features.push("crossval".to_string());

        features
    }
}
