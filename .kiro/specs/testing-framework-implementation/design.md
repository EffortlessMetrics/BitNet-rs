# Testing Framework Implementation Design

## Overview

This document outlines the design for implementing the foundational components of the comprehensive testing framework for BitNet-rs. The design focuses on creating a robust, extensible foundation that can support the full testing framework while providing immediate value through basic cross-implementation comparison and comprehensive unit testing.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Testing Framework Core                      │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│ │   Test Harness  │ │ Fixture Manager │ │  Config Manager │   │
│ │                 │ │                 │ │                 │   │
│ │ • Test execution│ │ • Data loading  │ │ • Test config   │   │
│ │ • Result collect│ │ • Caching       │ │ • Environment   │   │
│ │ • Parallel runs │ │ • Cleanup       │ │ • Validation    │   │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│ │   Unit Tests    │ │ Integration     │ │ Cross-Impl      │   │
│ │                 │ │    Tests        │ │  Comparison     │   │
│ │ • Per-crate     │ │ • Workflows     │ │ • Rust vs C++   │   │
│ │ • API coverage  │ │ • Component     │ │ • Accuracy      │   │
│ │ • Edge cases    │ │   interaction   │ │ • Performance   │   │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│ │   Reporting     │ │   CI/CD         │ │   Utilities     │   │
│ │                 │ │  Integration    │ │                 │   │
│ │ • Coverage      │ │ • GitHub Actions│ │ • Logging       │   │
│ │ • Metrics       │ │ • Caching       │ │ • Debugging     │   │
│ │ • Visualizations│ │ • Artifacts     │ │ • Helpers       │   │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Core Test Infrastructure

#### Test Harness (`tests/common/harness.rs`)
```rust
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

pub struct TestHarness {
    config: TestConfig,
    fixtures: FixtureManager,
    reporters: Vec<Box<dyn TestReporter>>,
    semaphore: Semaphore,
}

impl TestHarness {
    pub fn new(config: TestConfig) -> Self {
        let max_parallel = config.max_parallel_tests;
        Self {
            config,
            fixtures: FixtureManager::new(),
            reporters: Vec::new(),
            semaphore: Semaphore::new(max_parallel),
        }
    }

    pub async fn run_test_suite<T: TestSuite>(&self, suite: T) -> TestSuiteResult {
        let start_time = Instant::now();
        let mut results = Vec::new();

        for test_case in suite.test_cases() {
            let _permit = self.semaphore.acquire().await.unwrap();
            let result = self.run_single_test(test_case).await;
            results.push(result);
        }

        TestSuiteResult {
            suite_name: suite.name().to_string(),
            total_duration: start_time.elapsed(),
            test_results: results,
            summary: self.calculate_summary(&results),
        }
    }

    async fn run_single_test(&self, test_case: Box<dyn TestCase>) -> TestResult {
        let start_time = Instant::now();

        // Setup phase
        let setup_result = test_case.setup(&self.fixtures).await;
        if let Err(e) = setup_result {
            return TestResult::failed(test_case.name(), e, start_time.elapsed());
        }

        // Execute phase
        let execute_result = test_case.execute().await;
        let duration = start_time.elapsed();

        // Cleanup phase
        let _ = test_case.cleanup().await;

        match execute_result {
            Ok(metrics) => TestResult::passed(test_case.name(), metrics, duration),
            Err(e) => TestResult::failed(test_case.name(), e, duration),
        }
    }
}

pub trait TestCase: Send + Sync {
    fn name(&self) -> &str;
    async fn setup(&self, fixtures: &FixtureManager) -> Result<(), TestError>;
    async fn execute(&self) -> Result<TestMetrics, TestError>;
    async fn cleanup(&self) -> Result<(), TestError>;
}

pub trait TestSuite {
    fn name(&self) -> &str;
    fn test_cases(&self) -> Vec<Box<dyn TestCase>>;
}
```

#### Test Results and Metrics (`tests/common/results.rs`)
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_name: String,
    pub status: TestStatus,
    pub duration: Duration,
    pub metrics: TestMetrics,
    pub error: Option<TestError>,
    pub artifacts: Vec<TestArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestStatus {
    Passed,
    Failed,
    Skipped,
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMetrics {
    pub memory_peak: Option<u64>,
    pub memory_average: Option<u64>,
    pub cpu_time: Option<Duration>,
    pub wall_time: Duration,
    pub custom_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuiteResult {
    pub suite_name: String,
    pub total_duration: Duration,
    pub test_results: Vec<TestResult>,
    pub summary: TestSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    pub total_tests: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub success_rate: f64,
    pub total_duration: Duration,
}
```

### 2. Fixture Management System

#### Fixture Manager (`tests/common/fixtures.rs`)
```rust
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use sha2::{Sha256, Digest};

pub struct FixtureManager {
    cache_dir: PathBuf,
    fixtures: HashMap<String, FixtureInfo>,
    downloads: HashMap<String, DownloadInfo>,
}

impl FixtureManager {
    pub fn new() -> Self {
        let cache_dir = std::env::var("BITNET_TEST_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("tests/cache"));

        Self {
            cache_dir,
            fixtures: HashMap::new(),
            downloads: HashMap::new(),
        }
    }

    pub async fn get_model_fixture(&self, name: &str) -> Result<PathBuf, FixtureError> {
        if let Some(fixture) = self.fixtures.get(name) {
            let path = self.cache_dir.join(&fixture.filename);
            if path.exists() && self.verify_checksum(&path, &fixture.checksum).await? {
                return Ok(path);
            }
        }

        // Download if not cached or checksum mismatch
        self.download_fixture(name).await
    }

    async fn download_fixture(&self, name: &str) -> Result<PathBuf, FixtureError> {
        let download_info = self.downloads.get(name)
            .ok_or_else(|| FixtureError::UnknownFixture(name.to_string()))?;

        let target_path = self.cache_dir.join(&download_info.filename);

        // Create cache directory if it doesn't exist
        fs::create_dir_all(&self.cache_dir).await?;

        // Download file
        let response = reqwest::get(&download_info.url).await?;
        let bytes = response.bytes().await?;

        // Verify checksum
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hash = format!("{:x}", hasher.finalize());

        if hash != download_info.checksum {
            return Err(FixtureError::ChecksumMismatch {
                expected: download_info.checksum.clone(),
                actual: hash,
            });
        }

        // Write to cache
        fs::write(&target_path, bytes).await?;

        Ok(target_path)
    }

    async fn verify_checksum(&self, path: &Path, expected: &str) -> Result<bool, FixtureError> {
        let bytes = fs::read(path).await?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hash = format!("{:x}", hasher.finalize());
        Ok(hash == expected)
    }

    pub async fn cleanup_old_fixtures(&self, max_age: Duration) -> Result<(), FixtureError> {
        let cutoff = SystemTime::now() - max_age;

        let mut entries = fs::read_dir(&self.cache_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let metadata = entry.metadata().await?;
            if let Ok(modified) = metadata.modified() {
                if modified < cutoff {
                    let _ = fs::remove_file(entry.path()).await;
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FixtureInfo {
    pub name: String,
    pub filename: String,
    pub checksum: String,
    pub size: u64,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct DownloadInfo {
    pub url: String,
    pub filename: String,
    pub checksum: String,
}
```

### 3. Cross-Implementation Comparison

#### Implementation Abstraction (`tests/crossval/implementation.rs`)
```rust
use async_trait::async_trait;
use std::path::Path;

#[async_trait]
pub trait BitNetImplementation: Send + Sync {
    async fn load_model(&mut self, model_path: &Path) -> Result<(), ImplementationError>;
    async fn tokenize(&self, text: &str) -> Result<Vec<u32>, ImplementationError>;
    async fn inference(&self, tokens: &[u32], config: &InferenceConfig) -> Result<InferenceResult, ImplementationError>;
    fn get_metrics(&self) -> PerformanceMetrics;
    fn implementation_name(&self) -> &str;
}

pub struct RustImplementation {
    model: Option<bitnet::BitNetModel>,
    metrics: PerformanceMetrics,
}

impl RustImplementation {
    pub fn new() -> Self {
        Self {
            model: None,
            metrics: PerformanceMetrics::default(),
        }
    }
}

#[async_trait]
impl BitNetImplementation for RustImplementation {
    async fn load_model(&mut self, model_path: &Path) -> Result<(), ImplementationError> {
        let start = Instant::now();

        let model = bitnet::BitNetModel::from_file(model_path).await
            .map_err(|e| ImplementationError::ModelLoadError(e.to_string()))?;

        self.model = Some(model);
        self.metrics.model_load_time = start.elapsed();

        Ok(())
    }

    async fn tokenize(&self, text: &str) -> Result<Vec<u32>, ImplementationError> {
        let model = self.model.as_ref()
            .ok_or(ImplementationError::ModelNotLoaded)?;

        let tokens = model.tokenize(text).await
            .map_err(|e| ImplementationError::TokenizationError(e.to_string()))?;

        Ok(tokens)
    }

    async fn inference(&self, tokens: &[u32], config: &InferenceConfig) -> Result<InferenceResult, ImplementationError> {
        let model = self.model.as_ref()
            .ok_or(ImplementationError::ModelNotLoaded)?;

        let start = Instant::now();
        let start_memory = get_memory_usage();

        let result = model.generate_from_tokens(tokens, config).await
            .map_err(|e| ImplementationError::InferenceError(e.to_string()))?;

        let duration = start.elapsed();
        let peak_memory = get_peak_memory_usage();

        self.metrics.inference_time = duration;
        self.metrics.peak_memory = peak_memory;

        Ok(InferenceResult {
            tokens: result.tokens,
            probabilities: result.probabilities,
            logits: result.logits,
            duration,
            memory_usage: peak_memory - start_memory,
        })
    }

    fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.clone()
    }

    fn implementation_name(&self) -> &str {
        "BitNet-rs"
    }
}

pub struct CppImplementation {
    // FFI bindings to C++ implementation
    handle: Option<*mut c_void>,
    metrics: PerformanceMetrics,
}

// Similar implementation for C++ wrapper...
```

#### Comparison Framework (`tests/crossval/comparison.rs`)
```rust
pub struct CrossValidationSuite {
    rust_impl: Box<dyn BitNetImplementation>,
    cpp_impl: Box<dyn BitNetImplementation>,
    tolerance: ComparisonTolerance,
    test_cases: Vec<ComparisonTestCase>,
}

impl CrossValidationSuite {
    pub fn new(tolerance: ComparisonTolerance) -> Self {
        Self {
            rust_impl: Box::new(RustImplementation::new()),
            cpp_impl: Box::new(CppImplementation::new()),
            tolerance,
            test_cases: Vec::new(),
        }
    }

    pub async fn run_comparison(&mut self, model_path: &Path) -> Result<ComparisonResult, ComparisonError> {
        // Load model in both implementations
        self.rust_impl.load_model(model_path).await?;
        self.cpp_impl.load_model(model_path).await?;

        let mut test_results = Vec::new();

        for test_case in &self.test_cases {
            let result = self.run_single_comparison(test_case).await?;
            test_results.push(result);
        }

        Ok(ComparisonResult {
            model_path: model_path.to_path_buf(),
            test_results,
            summary: self.calculate_summary(&test_results),
            rust_metrics: self.rust_impl.get_metrics(),
            cpp_metrics: self.cpp_impl.get_metrics(),
        })
    }

    async fn run_single_comparison(&mut self, test_case: &ComparisonTestCase) -> Result<SingleComparisonResult, ComparisonError> {
        // Tokenize input
        let rust_tokens = self.rust_impl.tokenize(&test_case.input).await?;
        let cpp_tokens = self.cpp_impl.tokenize(&test_case.input).await?;

        // Check tokenization consistency
        let tokenization_match = rust_tokens == cpp_tokens;

        // Run inference
        let rust_result = self.rust_impl.inference(&rust_tokens, &test_case.config).await?;
        let cpp_result = self.cpp_impl.inference(&cpp_tokens, &test_case.config).await?;

        // Compare results
        let accuracy_result = self.compare_accuracy(&rust_result, &cpp_result)?;
        let performance_comparison = self.compare_performance(&rust_result, &cpp_result);

        Ok(SingleComparisonResult {
            test_case: test_case.clone(),
            tokenization_match,
            accuracy_result,
            performance_comparison,
            rust_result,
            cpp_result,
        })
    }

    fn compare_accuracy(&self, rust_result: &InferenceResult, cpp_result: &InferenceResult) -> Result<AccuracyResult, ComparisonError> {
        let token_matches = rust_result.tokens.iter()
            .zip(cpp_result.tokens.iter())
            .filter(|(r, c)| r == c)
            .count();

        let total_tokens = rust_result.tokens.len().min(cpp_result.tokens.len());
        let token_accuracy = token_matches as f64 / total_tokens as f64;

        // Find first mismatch
        let first_mismatch = rust_result.tokens.iter()
            .zip(cpp_result.tokens.iter())
            .enumerate()
            .find(|(_, (r, c))| r != c)
            .map(|(idx, (rust_token, cpp_token))| TokenMismatch {
                position: idx,
                rust_token: *rust_token,
                cpp_token: *cpp_token,
                context: self.get_context(&rust_result.tokens, idx),
            });

        // Compare probability distributions if available
        let probability_similarity = if let (Some(rust_probs), Some(cpp_probs)) =
            (&rust_result.probabilities, &cpp_result.probabilities) {
            self.calculate_probability_similarity(rust_probs, cpp_probs)
        } else {
            None
        };

        Ok(AccuracyResult {
            token_accuracy,
            total_tokens,
            matches: token_matches,
            first_mismatch,
            probability_similarity,
            passes_tolerance: token_accuracy >= self.tolerance.min_token_accuracy,
        })
    }

    fn compare_performance(&self, rust_result: &InferenceResult, cpp_result: &InferenceResult) -> PerformanceComparison {
        let throughput_ratio = if cpp_result.duration.as_secs_f64() > 0.0 {
            rust_result.duration.as_secs_f64() / cpp_result.duration.as_secs_f64()
        } else {
            1.0
        };

        let memory_ratio = if cpp_result.memory_usage > 0 {
            rust_result.memory_usage as f64 / cpp_result.memory_usage as f64
        } else {
            1.0
        };

        PerformanceComparison {
            rust_duration: rust_result.duration,
            cpp_duration: cpp_result.duration,
            throughput_ratio, // < 1.0 means Rust is faster
            rust_memory: rust_result.memory_usage,
            cpp_memory: cpp_result.memory_usage,
            memory_ratio, // < 1.0 means Rust uses less memory
        }
    }
}
```

## Data Models

### Configuration Schema (`tests/common/config.rs`)
```rust
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    pub max_parallel_tests: usize,
    pub test_timeout: Duration,
    pub cache_dir: PathBuf,
    pub log_level: String,
    pub coverage_threshold: f64,
    pub fixtures: FixtureConfig,
    pub crossval: CrossValidationConfig,
    pub reporting: ReportingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureConfig {
    pub auto_download: bool,
    pub max_cache_size: u64,
    pub cleanup_interval: Duration,
    pub download_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationConfig {
    pub enabled: bool,
    pub tolerance: ComparisonTolerance,
    pub cpp_binary_path: Option<PathBuf>,
    pub test_cases: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonTolerance {
    pub min_token_accuracy: f64,
    pub max_probability_divergence: f64,
    pub max_performance_regression: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    pub output_dir: PathBuf,
    pub formats: Vec<ReportFormat>,
    pub include_artifacts: bool,
    pub generate_coverage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Html,
    Json,
    Junit,
    Markdown,
}
```

## Error Handling

### Test Error Types (`tests/common/errors.rs`)
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TestError {
    #[error("Test setup failed: {0}")]
    SetupError(String),

    #[error("Test execution failed: {0}")]
    ExecutionError(String),

    #[error("Test timeout after {timeout:?}")]
    TimeoutError { timeout: Duration },

    #[error("Assertion failed: {message}")]
    AssertionError { message: String },

    #[error("Fixture error: {0}")]
    FixtureError(#[from] FixtureError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

#[derive(Debug, Error)]
pub enum FixtureError {
    #[error("Unknown fixture: {0}")]
    UnknownFixture(String),

    #[error("Download failed: {0}")]
    DownloadError(String),

    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    #[error("Cache error: {0}")]
    CacheError(String),
}

#[derive(Debug, Error)]
pub enum ComparisonError {
    #[error("Implementation error: {0}")]
    ImplementationError(#[from] ImplementationError),

    #[error("Accuracy comparison failed: {0}")]
    AccuracyError(String),

    #[error("Performance comparison failed: {0}")]
    PerformanceError(String),
}

#[derive(Debug, Error)]
pub enum ImplementationError {
    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("Model load error: {0}")]
    ModelLoadError(String),

    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),
}
```

## Implementation Strategy

### Phase 1: Core Infrastructure (Week 1-2)
1. Set up test harness with basic execution and reporting
2. Implement fixture management with download and caching
3. Create configuration system with validation
4. Add basic error handling and logging

### Phase 2: Unit Testing Framework (Week 2-3)
1. Create unit test templates for each crate
2. Implement coverage collection and reporting
3. Add property-based testing integration
4. Create CI integration for unit tests

### Phase 3: Cross-Implementation Comparison (Week 3-4)
1. Implement BitNet implementation trait
2. Create Rust implementation wrapper
3. Add C++ implementation wrapper with FFI
4. Implement basic accuracy and performance comparison

### Phase 4: Integration Testing (Week 4-5)
1. Create integration test framework
2. Add workflow validation tests
3. Implement component interaction tests
4. Add resource management validation

### Phase 5: Reporting and CI Integration (Week 5-6)
1. Implement comprehensive reporting system
2. Add HTML and JSON report generation
3. Create GitHub Actions integration
4. Add artifact collection and publishing

This design provides a solid foundation for the comprehensive testing framework while delivering immediate value through basic cross-implementation comparison and thorough unit testing coverage.
