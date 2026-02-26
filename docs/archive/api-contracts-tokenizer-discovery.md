# API Contracts: Tokenizer Discovery and Automatic Download

## Overview

This document defines the public API contracts for the BitNet-rs tokenizer discovery system, providing comprehensive interfaces for automatic tokenizer resolution, smart downloading, and neural network model integration.

## Core API Contracts

### 1. TokenizerDiscovery Interface

```rust
/// Primary tokenizer discovery engine for BitNet-rs neural network models
pub struct TokenizerDiscovery {
    gguf_reader: Arc<GgufReader<'static>>,
    model_path: PathBuf,
    vocab_size: usize,
    model_type: String,
    quantization_type: Option<QuantizationType>,
}

impl TokenizerDiscovery {
    /// Create discovery engine from GGUF model file
    ///
    /// # Arguments
    /// * `path` - Path to GGUF model file
    ///
    /// # Returns
    /// * `Ok(TokenizerDiscovery)` - Successfully initialized discovery engine
    /// * `Err(BitNetError::Model)` - GGUF parsing failed or metadata missing
    ///
    /// # Example
    /// ```rust
    /// let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;
    /// assert_eq!(discovery.vocab_size(), 128256); // LLaMA-3
    /// ```
    pub fn from_gguf(path: &Path) -> Result<Self>;

    /// Discover optimal tokenizer strategy for the loaded model
    ///
    /// # Returns
    /// * `TokenizerStrategy::Discovered` - Compatible tokenizer found locally
    /// * `TokenizerStrategy::NeedsDownload` - Smart download required
    /// * `TokenizerStrategy::EmbeddedGguf` - GGUF contains embedded tokenizer
    /// * `TokenizerStrategy::Mock` - Fallback for testing (non-strict mode only)
    ///
    /// # Errors
    /// * `BitNetError::Inference` - No compatible tokenizer found in strict mode
    pub fn discover_tokenizer_strategy(&self) -> Result<TokenizerStrategy>;

    /// Get vocabulary size from model metadata
    pub fn vocab_size(&self) -> usize;

    /// Get model architecture type (e.g., "llama", "gpt2")
    pub fn model_type(&self) -> &str;

    /// Get quantization type if specified in model
    pub fn quantization_type(&self) -> Option<QuantizationType>;

    /// Check if model requires large vocabulary optimization (>64K tokens)
    pub fn requires_large_vocab_optimization(&self) -> bool {
        self.vocab_size > 65536
    }
}
```

### 2. SmartTokenizerDownload Interface

```rust
/// Intelligent tokenizer downloading with caching and resume capability
pub struct SmartTokenizerDownload {
    cache_dir: PathBuf,
    client: reqwest::Client,
    compatibility_matrix: ModelCompatibilityMatrix,
}

impl SmartTokenizerDownload {
    /// Initialize download system with default cache directory
    ///
    /// # Returns
    /// * `Ok(SmartTokenizerDownload)` - Successfully initialized
    /// * `Err(BitNetError::Model)` - Cache directory creation failed
    pub fn new() -> Result<Self>;

    /// Download tokenizer files for given download info
    ///
    /// # Arguments
    /// * `info` - Download metadata including repo, files, and cache key
    ///
    /// # Returns
    /// * `Ok(PathBuf)` - Path to primary tokenizer file
    /// * `Err(BitNetError::Model)` - Download failed or network error
    ///
    /// # Example
    /// ```rust
    /// let info = TokenizerDownloadInfo {
    ///     repo: "meta-llama/Llama-2-7b-hf".to_string(),
    ///     files: vec!["tokenizer.json".to_string()],
    ///     cache_key: "llama2-32k".to_string(),
    /// };
    /// let tokenizer_path = downloader.download_tokenizer(&info).await?;
    /// ```
    pub async fn download_tokenizer(&self, info: &TokenizerDownloadInfo) -> Result<PathBuf>;

    /// Check if tokenizer is already cached
    ///
    /// # Returns
    /// * `Some(PathBuf)` - Path to cached tokenizer
    /// * `None` - Tokenizer not in cache
    pub fn find_cached_tokenizer(&self, cache_key: &str) -> Option<PathBuf>;

    /// Clear cache for specific tokenizer or all cached tokenizers
    pub fn clear_cache(&self, cache_key: Option<&str>) -> Result<()>;
}

/// Download metadata for tokenizer acquisition
#[derive(Debug, Clone)]
pub struct TokenizerDownloadInfo {
    pub repo: String,           // HuggingFace repo (e.g., "meta-llama/Llama-2-7b-hf")
    pub files: Vec<String>,     // Required files (e.g., ["tokenizer.json"])
    pub cache_key: String,      // Cache identifier (e.g., "llama2-32k")
    pub expected_vocab: Option<usize>, // Expected vocabulary size for validation
}
```

### 3. TokenizerStrategy Enumeration

```rust
/// Comprehensive tokenizer resolution strategy
#[derive(Debug, Clone)]
pub enum TokenizerStrategy {
    /// User explicitly specified tokenizer path
    Exact(PathBuf),

    /// Auto-discovered compatible tokenizer in model directory
    Discovered(PathBuf),

    /// Smart download required from HuggingFace Hub
    NeedsDownload(TokenizerDownloadInfo),

    /// GGUF file contains embedded tokenizer data
    EmbeddedGguf(Arc<dyn Tokenizer>),

    /// Mock tokenizer for testing (non-strict mode only)
    Mock,
}

impl TokenizerStrategy {
    /// Check if strategy requires network access
    pub fn requires_network(&self) -> bool {
        matches!(self, TokenizerStrategy::NeedsDownload(_))
    }

    /// Check if strategy uses cached resources
    pub fn uses_cache(&self) -> bool {
        matches!(self, TokenizerStrategy::Discovered(_) | TokenizerStrategy::NeedsDownload(_))
    }

    /// Get description for logging and error messages
    pub fn description(&self) -> &'static str {
        match self {
            TokenizerStrategy::Exact(_) => "user-specified tokenizer",
            TokenizerStrategy::Discovered(_) => "auto-discovered tokenizer",
            TokenizerStrategy::NeedsDownload(_) => "smart download required",
            TokenizerStrategy::EmbeddedGguf(_) => "GGUF-embedded tokenizer",
            TokenizerStrategy::Mock => "mock tokenizer (testing only)",
        }
    }
}
```

### 4. TokenizerStrategyResolver Interface

```rust
/// Unified tokenizer strategy resolution with neural network model integration
pub struct TokenizerStrategyResolver {
    discovery: TokenizerDiscovery,
    downloader: SmartTokenizerDownload,
    fallback_chain: TokenizerFallbackChain,
}

impl TokenizerStrategyResolver {
    /// Create resolver with discovery engine and downloader
    pub async fn new(discovery: TokenizerDiscovery) -> Result<Self>;

    /// Resolve tokenizer strategy to concrete tokenizer implementation
    ///
    /// # Arguments
    /// * `strategy` - Tokenizer strategy to resolve
    ///
    /// # Returns
    /// * `Arc<dyn Tokenizer>` - Concrete tokenizer implementation
    ///
    /// # Errors
    /// * `BitNetError::Inference` - Strategy resolution failed
    /// * `BitNetError::Model` - Tokenizer loading or download failed
    pub async fn resolve_tokenizer(&self, strategy: TokenizerStrategy) -> Result<Arc<dyn Tokenizer>>;

    /// Resolve with automatic fallback chain
    ///
    /// Attempts multiple strategies in order:
    /// 1. GGUF embedded tokenizer
    /// 2. Co-located files
    /// 3. Standard cache directories
    /// 4. Smart download
    /// 5. Mock fallback (non-strict mode)
    pub async fn resolve_with_fallback(&self) -> Result<Arc<dyn Tokenizer>>;
}
```

### 5. Neural Network Model Wrappers

```rust
/// LLaMA model-specific tokenizer wrapper
pub struct LlamaTokenizerWrapper {
    inner: Arc<dyn Tokenizer>,
    vocab_size: usize,
    model_variant: LlamaVariant,
}

/// GPT-2 model-specific tokenizer wrapper
pub struct Gpt2TokenizerWrapper {
    inner: Arc<dyn Tokenizer>,
}

/// BitNet model-specific tokenizer wrapper
pub struct BitNetTokenizerWrapper {
    inner: Arc<dyn Tokenizer>,
    quantization_type: QuantizationType,
}

/// Model variant enumeration
#[derive(Debug, Clone, Copy)]
pub enum LlamaVariant {
    Llama2,     // 32K vocabulary, legacy special tokens
    Llama3,     // 128K vocabulary, enhanced special tokens
    CodeLlama,  // Code-optimized vocabulary
}
```

## Error Handling Contracts

### Error Types

```rust
/// Comprehensive error types for tokenizer discovery
#[derive(Debug, thiserror::Error)]
pub enum TokenizerDiscoveryError {
    #[error("GGUF metadata parsing failed: {reason}")]
    GgufMetadataError { reason: String },

    #[error("Tokenizer download failed from {repo}: {reason}")]
    DownloadError { repo: String, reason: String },

    #[error("Tokenizer validation failed: expected vocab {expected}, got {actual}")]
    ValidationError { expected: usize, actual: usize },

    #[error("No compatible tokenizer found: {reason}")]
    NoCompatibleTokenizer { reason: String },

    #[error("Strict mode violation: {reason}")]
    StrictModeViolation { reason: String },

    #[error("Cache operation failed: {reason}")]
    CacheError { reason: String },
}
```

### Error Recovery Patterns

```rust
/// Error recovery with actionable suggestions
impl TokenizerDiscoveryError {
    /// Get user-actionable suggestions for resolving the error
    pub fn suggestions(&self) -> Vec<String> {
        match self {
            TokenizerDiscoveryError::NoCompatibleTokenizer { .. } => vec![
                "Place tokenizer.json in the same directory as the model".to_string(),
                "Use --tokenizer path/to/tokenizer.json to specify manually".to_string(),
                "Use --auto-download to enable automatic downloading".to_string(),
            ],
            TokenizerDiscoveryError::DownloadError { .. } => vec![
                "Check internet connection and retry".to_string(),
                "Use --tokenizer to specify local tokenizer file".to_string(),
                "Clear cache with: rm -rf ~/.cache/bitnet/tokenizers".to_string(),
            ],
            TokenizerDiscoveryError::StrictModeViolation { .. } => vec![
                "Remove BITNET_STRICT_TOKENIZERS=1 to enable fallback tokenizers".to_string(),
                "Provide compatible tokenizer with --tokenizer flag".to_string(),
            ],
            _ => vec!["See documentation for troubleshooting guidance".to_string()],
        }
    }
}
```

## Environment Variable Contracts

### Configuration Variables

```rust
/// Environment variable configuration for tokenizer system
pub struct TokenizerEnvironment;

impl TokenizerEnvironment {
    /// Check if strict tokenizer mode is enabled
    ///
    /// When enabled:
    /// - No mock tokenizer fallbacks
    /// - Validation failures are hard errors
    /// - Network downloads must succeed or fail
    pub fn is_strict_mode() -> bool {
        std::env::var("BITNET_STRICT_TOKENIZERS").as_deref() == Ok("1")
    }

    /// Check if deterministic mode is enabled
    ///
    /// When enabled:
    /// - Tokenizer selection is deterministic across runs
    /// - Caching behavior is consistent
    /// - Random fallbacks are disabled
    pub fn is_deterministic() -> bool {
        std::env::var("BITNET_DETERMINISTIC").as_deref() == Ok("1")
    }

    /// Check if offline mode is enabled
    ///
    /// When enabled:
    /// - No network downloads attempted
    /// - Only local and cached tokenizers used
    /// - Smart download strategies are skipped
    pub fn is_offline_mode() -> bool {
        std::env::var("BITNET_OFFLINE").as_deref() == Ok("1")
    }

    /// Get custom cache directory if specified
    pub fn cache_directory() -> Option<PathBuf> {
        std::env::var("BITNET_TOKENIZER_CACHE")
            .ok()
            .map(PathBuf::from)
    }
}
```

## Feature Flag Contracts

### Build Feature Matrix

```rust
/// Feature flag configuration for tokenizer discovery system
pub struct TokenizerFeatures;

impl TokenizerFeatures {
    /// Check if SentencePiece support is available
    #[cfg(feature = "spm")]
    pub fn has_sentencepiece() -> bool { true }
    #[cfg(not(feature = "spm"))]
    pub fn has_sentencepiece() -> bool { false }

    /// Check if async downloading is available
    #[cfg(any(feature = "cpu", feature = "gpu"))]
    pub fn has_async_download() -> bool { true }
    #[cfg(not(any(feature = "cpu", feature = "gpu")))]
    pub fn has_async_download() -> bool { false }

    /// Check if GPU acceleration is available for large vocabularies
    #[cfg(feature = "gpu")]
    pub fn has_gpu_acceleration() -> bool { true }
    #[cfg(not(feature = "gpu"))]
    pub fn has_gpu_acceleration() -> bool { false }

    /// Get supported tokenizer formats based on enabled features
    pub fn supported_formats() -> Vec<&'static str> {
        let mut formats = vec!["json"]; // HuggingFace JSON always supported

        #[cfg(feature = "spm")]
        formats.push("model"); // SentencePiece models

        formats
    }
}
```

## Testing Contracts

### Test Trait Implementations

```rust
/// Testing utilities for tokenizer discovery system
pub mod testing {
    use super::*;

    /// Mock discovery engine for testing
    pub struct MockTokenizerDiscovery {
        vocab_size: usize,
        model_type: String,
        strategy: TokenizerStrategy,
    }

    impl MockTokenizerDiscovery {
        pub fn new(vocab_size: usize, model_type: String) -> Self {
            Self {
                vocab_size,
                model_type,
                strategy: TokenizerStrategy::Mock,
            }
        }

        pub fn with_strategy(mut self, strategy: TokenizerStrategy) -> Self {
            self.strategy = strategy;
            self
        }
    }

    /// Test utilities for validation
    pub struct TokenizerTestUtils;

    impl TokenizerTestUtils {
        /// Create test GGUF file with specified metadata
        pub fn create_test_gguf(
            path: &Path,
            vocab_size: usize,
            model_type: &str,
        ) -> Result<()>;

        /// Verify tokenizer compatibility with quantization format
        pub fn verify_quantization_compatibility(
            tokenizer: &dyn Tokenizer,
            quant_type: QuantizationType,
        ) -> Result<()>;

        /// Compare tokenizer outputs for compatibility testing
        pub fn compare_tokenizer_outputs(
            tokenizer1: &dyn Tokenizer,
            tokenizer2: &dyn Tokenizer,
            test_texts: &[&str],
        ) -> Result<f64>; // Returns similarity score 0.0-1.0
    }
}
```

## Performance Contracts

### Benchmarking Interface

```rust
/// Performance benchmarking for tokenizer discovery
pub struct TokenizerBenchmark;

impl TokenizerBenchmark {
    /// Benchmark tokenizer discovery overhead
    pub fn measure_discovery_overhead(model_path: &Path) -> Result<Duration>;

    /// Benchmark large vocabulary tokenization performance
    pub fn measure_large_vocab_performance(
        tokenizer: &dyn Tokenizer,
        test_corpus: &str,
    ) -> Result<TokenizationMetrics>;

    /// Benchmark download and caching performance
    pub async fn measure_download_performance(
        info: &TokenizerDownloadInfo,
    ) -> Result<DownloadMetrics>;
}

#[derive(Debug)]
pub struct TokenizationMetrics {
    pub tokens_per_second: f64,
    pub memory_usage_mb: f64,
    pub cache_hit_rate: f64,
}

#[derive(Debug)]
pub struct DownloadMetrics {
    pub download_time: Duration,
    pub cache_time: Duration,
    pub network_efficiency: f64,
}
```

## Conclusion

These API contracts provide a comprehensive interface for BitNet-rs tokenizer discovery and automatic download functionality. The contracts emphasize:

- **Type Safety**: Comprehensive error types with actionable suggestions
- **Performance**: Benchmarking interfaces for validation
- **Neural Network Integration**: Model-specific wrappers and quantization compatibility
- **Testing**: Mock implementations and validation utilities
- **Configuration**: Environment variables and feature flags for flexible deployment

All interfaces maintain backward compatibility with existing BitNet-rs tokenizer infrastructure while enabling advanced automatic discovery and download capabilities for production neural network inference.
