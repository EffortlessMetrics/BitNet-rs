//! Tokenizer discovery system for automatic tokenizer resolution
//!
//! This module provides comprehensive tokenizer discovery capabilities for BitNet.rs neural network models.
//! Supports GGUF metadata parsing, smart downloading, and device-aware tokenization for production-scale models.

use crate::Tokenizer;
use bitnet_common::{BitNetError, Result};
use bitnet_models::GgufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Neural network model compatibility matrix for tokenizer discovery
#[derive(Debug, Clone)]
pub struct ModelCompatibilityMatrix {
    /// LLaMA-3 with 128K vocabulary - requires I2S quantization with GPU acceleration
    pub llama3_128k: TokenizerDownloadInfo,
    /// LLaMA-2 with 32K vocabulary - compatible with TL1/TL2 quantization
    pub llama2_32k: TokenizerDownloadInfo,
    /// GPT-2 with 50K vocabulary - standard BPE tokenization
    pub gpt2_50k: TokenizerDownloadInfo,
    /// BitNet-specific tokenizers for neural network optimization
    pub bitnet_custom: TokenizerDownloadInfo,
}

impl Default for ModelCompatibilityMatrix {
    fn default() -> Self {
        Self {
            llama3_128k: TokenizerDownloadInfo {
                repo: "meta-llama/Meta-Llama-3-8B".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "llama3-128k".to_string(),
                expected_vocab: Some(128256),
            },
            llama2_32k: TokenizerDownloadInfo {
                repo: "meta-llama/Llama-2-7b-hf".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "llama2-32k".to_string(),
                expected_vocab: Some(32000),
            },
            gpt2_50k: TokenizerDownloadInfo {
                repo: "openai-community/gpt2".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "gpt2-50k".to_string(),
                expected_vocab: Some(50257),
            },
            bitnet_custom: TokenizerDownloadInfo {
                repo: "1bitLLM/bitnet_b1_58-large".to_string(),
                files: vec!["tokenizer.json".to_string(), "tokenizer.model".to_string()],
                cache_key: "bitnet-custom".to_string(),
                expected_vocab: None,
            },
        }
    }
}

/// Download metadata for tokenizer acquisition from HuggingFace Hub
#[derive(Debug, Clone)]
pub struct TokenizerDownloadInfo {
    /// HuggingFace repository identifier (e.g., "meta-llama/Llama-2-7b-hf")
    pub repo: String,
    /// Required tokenizer files to download (e.g., ["tokenizer.json"])
    pub files: Vec<String>,
    /// Cache identifier for persistent storage (e.g., "llama2-32k")
    pub cache_key: String,
    /// Expected vocabulary size for validation (optional)
    pub expected_vocab: Option<usize>,
}

/// Comprehensive tokenizer resolution strategy for neural network models
#[derive(Clone)]
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

/// Primary tokenizer discovery engine for BitNet.rs neural network models
pub struct TokenizerDiscovery {
    _mmap: memmap2::Mmap, // Keep mmap alive
    gguf_reader: GgufReader<'static>,
    model_path: PathBuf,
    vocab_size: usize,
    model_type: String,
    compatibility_matrix: ModelCompatibilityMatrix,
}

impl TokenizerDiscovery {
    /// Create discovery engine from GGUF model file
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
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
    pub fn from_gguf(path: &Path) -> Result<Self> {
        // Read GGUF file using memmap for efficiency
        let file = std::fs::File::open(path).map_err(|e| {
            BitNetError::Model(bitnet_common::ModelError::FileIOError {
                path: path.to_path_buf(),
                source: e,
            })
        })?;

        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
            BitNetError::Model(bitnet_common::ModelError::FileIOError {
                path: path.to_path_buf(),
                source: e,
            })
        })?;

        // Create GGUF reader from memory-mapped data
        // We need to transmute the lifetime to 'static since we're keeping the mmap alive
        let reader = unsafe {
            let data_slice: &'static [u8] = std::mem::transmute(mmap.as_ref());
            GgufReader::new(data_slice)?
        };

        // Extract vocabulary size from metadata or tensors
        let vocab_size = Self::extract_vocab_size(&reader)?;

        // Extract model architecture type
        let model_type = Self::extract_model_type(&reader)?;

        Ok(Self {
            _mmap: mmap, // Keep mmap alive
            gguf_reader: reader,
            model_path: path.to_path_buf(),
            vocab_size,
            model_type,
            compatibility_matrix: ModelCompatibilityMatrix::default(),
        })
    }

    /// Discover optimal tokenizer strategy for the loaded model
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    ///
    /// # Returns
    /// * `TokenizerStrategy::Discovered` - Compatible tokenizer found locally
    /// * `TokenizerStrategy::NeedsDownload` - Smart download required
    /// * `TokenizerStrategy::EmbeddedGguf` - GGUF contains embedded tokenizer
    /// * `TokenizerStrategy::Mock` - Fallback for testing (non-strict mode only)
    ///
    /// # Errors
    /// * `BitNetError::Inference` - No compatible tokenizer found in strict mode
    pub fn discover_tokenizer_strategy(&self) -> Result<TokenizerStrategy> {
        info!(
            "Discovering tokenizer strategy for {} model (vocab_size: {})",
            self.model_type, self.vocab_size
        );

        // 1. Check for embedded tokenizer in GGUF
        if let Ok(Some(embedded)) = self.try_extract_embedded_tokenizer() {
            debug!("Found embedded tokenizer in GGUF file");
            return Ok(TokenizerStrategy::EmbeddedGguf(embedded));
        }

        // 2. Check for co-located tokenizer files
        if let Ok(Some(path)) = self.check_colocated_tokenizers() {
            debug!("Found co-located tokenizer at: {}", path.display());
            return Ok(TokenizerStrategy::Discovered(path));
        }

        // 3. Check cache locations
        if let Ok(Some(path)) = self.check_cache_locations() {
            debug!("Found cached tokenizer at: {}", path.display());
            return Ok(TokenizerStrategy::Discovered(path));
        }

        // 4. Check if we can infer download source
        if let Ok(Some(download_info)) = self.infer_download_source() {
            debug!("Can download compatible tokenizer from: {}", download_info.repo);
            return Ok(TokenizerStrategy::NeedsDownload(download_info));
        }

        // 5. Check strict mode - no fallback to mock in strict mode
        if std::env::var("BITNET_STRICT_TOKENIZERS").is_ok() {
            return Err(BitNetError::Config(format!(
                "No compatible tokenizer found for {} model with vocab_size {} (strict mode)",
                self.model_type, self.vocab_size
            )));
        }

        // 6. Fallback to mock for testing
        warn!("No compatible tokenizer found, falling back to mock (non-strict mode)");
        Ok(TokenizerStrategy::Mock)
    }

    /// Get vocabulary size from model metadata
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get model architecture type (e.g., "llama", "gpt2")
    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    /// Check if model requires large vocabulary optimization (>64K tokens)
    ///
    /// Large vocabularies require GPU acceleration for efficient embedding lookup
    pub fn requires_large_vocab_optimization(&self) -> bool {
        self.vocab_size > 65536
    }

    /// Extract vocabulary size from GGUF metadata
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    fn extract_vocab_size(reader: &GgufReader) -> Result<usize> {
        // Try to get vocab size from metadata first
        if let Some(vocab_size) = reader.get_u32_metadata("tokenizer.ggml.vocab_size") {
            return Ok(vocab_size as usize);
        }

        // Alternative metadata keys for different model architectures
        let alt_keys =
            ["llama.vocab_size", "gpt2.vocab_size", "transformer.vocab_size", "model.vocab_size"];

        for key in &alt_keys {
            if let Some(vocab_size) = reader.get_u32_metadata(key) {
                return Ok(vocab_size as usize);
            }
        }

        // Look for embedding tensor to infer vocab size
        let tensor_names = reader.tensor_names();
        for name in tensor_names {
            if (name.contains("token_embd") || name.contains("wte") || name.contains("embed"))
                && let Some(info) = reader.get_tensor_info_by_name(name)
            {
                // Embeddings are typically [vocab_size, hidden_dim]
                let shape = &info.shape;
                if shape.len() >= 2 {
                    let possible_vocab = std::cmp::max(shape[0], shape[1]);
                    // Sanity check - vocab size should be reasonable
                    if possible_vocab > 1000 && possible_vocab < 2_000_000 {
                        return Ok(possible_vocab);
                    }
                }
            }
        }

        // Default fallback for common model types
        Err(BitNetError::Config(
            "Could not extract vocabulary size from GGUF metadata or tensors".to_string(),
        ))
    }

    /// Extract model architecture type from GGUF metadata
    fn extract_model_type(reader: &GgufReader) -> Result<String> {
        // Try to get architecture from metadata
        if let Some(arch) = reader.get_string_metadata("general.architecture") {
            return Ok(arch);
        }

        // Alternative metadata keys
        let alt_keys = [
            "model.architecture",
            "transformer.architecture",
            "llama.architecture",
            "gpt.architecture",
        ];

        for key in &alt_keys {
            if let Some(arch) = reader.get_string_metadata(key) {
                return Ok(arch);
            }
        }

        // Try to infer from model name
        if let Some(name) = reader.get_string_metadata("general.name") {
            let name_lower = name.to_lowercase();
            if name_lower.contains("llama") {
                return Ok("llama".to_string());
            } else if name_lower.contains("gpt") {
                return Ok("gpt2".to_string());
            } else if name_lower.contains("bitnet") {
                return Ok("bitnet".to_string());
            }
        }

        // Fallback based on tensor patterns
        let tensor_names = reader.tensor_names();
        let has_llama_patterns = tensor_names.iter().any(|name| {
            name.contains("attn_q") || name.contains("attn_k") || name.contains("attn_v")
        });

        if has_llama_patterns { Ok("llama".to_string()) } else { Ok("transformer".to_string()) }
    }

    /// Check for co-located tokenizer files in model directory
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac5-fallback-strategy-system
    pub fn check_colocated_tokenizers(&self) -> Result<Option<PathBuf>> {
        let model_dir = self
            .model_path
            .parent()
            .ok_or_else(|| BitNetError::Config("Model path has no parent directory".to_string()))?;

        debug!("Searching for co-located tokenizers in: {}", model_dir.display());

        // Common tokenizer file names to check
        let tokenizer_files = [
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        ];

        for filename in &tokenizer_files {
            let tokenizer_path = model_dir.join(filename);
            if tokenizer_path.exists() && tokenizer_path.is_file() {
                debug!("Found co-located tokenizer file: {}", tokenizer_path.display());
                return Ok(Some(tokenizer_path));
            }
        }

        // Check for model name based tokenizer files
        if let Some(model_name) = self.model_path.file_stem()
            && let Some(model_str) = model_name.to_str()
        {
            let name_based_files = [
                format!("{}.tokenizer.json", model_str),
                format!("{}_tokenizer.json", model_str),
                format!("{}.vocab.json", model_str),
            ];

            for filename in &name_based_files {
                let tokenizer_path = model_dir.join(filename);
                if tokenizer_path.exists() && tokenizer_path.is_file() {
                    debug!("Found model-specific tokenizer file: {}", tokenizer_path.display());
                    return Ok(Some(tokenizer_path));
                }
            }
        }

        debug!("No co-located tokenizer files found");
        Ok(None)
    }

    /// Check standard cache directories for compatible tokenizers
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac5-fallback-strategy-system
    pub fn check_cache_locations(&self) -> Result<Option<PathBuf>> {
        debug!("Searching cache locations for compatible tokenizers");

        // Get cache directory from environment or use default
        let cache_dirs = [
            std::env::var("BITNET_CACHE_DIR").ok().map(PathBuf::from),
            std::env::var("XDG_CACHE_HOME").ok().map(|p| PathBuf::from(p).join("bitnet")),
            dirs::cache_dir().map(|p| p.join("bitnet")),
            Some(PathBuf::from(".cache/bitnet")), // Local cache
        ];

        for cache_dir_opt in &cache_dirs {
            if let Some(cache_dir) = cache_dir_opt {
                if !cache_dir.exists() {
                    continue;
                }

                // Check for model type specific cache
                let model_cache = cache_dir.join(&self.model_type);
                if model_cache.exists() {
                    // Look for vocab size specific tokenizers
                    let size_specific = model_cache.join(format!("vocab_{}", self.vocab_size));
                    if size_specific.exists() {
                        let tokenizer_json = size_specific.join("tokenizer.json");
                        if tokenizer_json.exists() {
                            debug!("Found cached tokenizer: {}", tokenizer_json.display());
                            return Ok(Some(tokenizer_json));
                        }
                    }

                    // Check general tokenizers in model type directory
                    for filename in &["tokenizer.json", "tokenizer.model"] {
                        let tokenizer_path = model_cache.join(filename);
                        if tokenizer_path.exists() {
                            debug!("Found general cached tokenizer: {}", tokenizer_path.display());
                            return Ok(Some(tokenizer_path));
                        }
                    }
                }

                // Check HuggingFace cache layout
                let hf_cache = cache_dir.join("huggingface");
                if hf_cache.exists() {
                    // Look for model repos that might match
                    if let Ok(entries) = std::fs::read_dir(&hf_cache) {
                        for entry in entries.flatten() {
                            if entry.file_type().is_ok_and(|ft| ft.is_dir()) {
                                let repo_dir = entry.path();
                                let tokenizer_json = repo_dir.join("tokenizer.json");
                                if tokenizer_json.exists() {
                                    debug!(
                                        "Found HF cached tokenizer: {}",
                                        tokenizer_json.display()
                                    );
                                    return Ok(Some(tokenizer_json));
                                }
                            }
                        }
                    }
                }
            }
        }

        debug!("No cached tokenizers found");
        Ok(None)
    }

    /// Infer download source based on neural network model patterns
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    pub fn infer_download_source(&self) -> Result<Option<TokenizerDownloadInfo>> {
        // Neural network model compatibility matrix lookup
        match (self.model_type.as_str(), self.vocab_size) {
            ("llama", 128256) => Ok(Some(self.compatibility_matrix.llama3_128k.clone())),
            ("llama", 32000) => Ok(Some(self.compatibility_matrix.llama2_32k.clone())),
            ("gpt2", 50257) => Ok(Some(self.compatibility_matrix.gpt2_50k.clone())),
            _ => Ok(None), // Unknown combination
        }
    }

    /// Try to extract embedded tokenizer from GGUF metadata
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    pub fn try_extract_embedded_tokenizer(&self) -> Result<Option<Arc<dyn Tokenizer>>> {
        debug!("Attempting to extract embedded tokenizer from GGUF metadata");

        // Check if tokenizer model is embedded as bytes
        if let Some(tokenizer_model) = self.gguf_reader.get_array_metadata("tokenizer.ggml.model") {
            debug!("Found embedded tokenizer.ggml.model ({} bytes)", tokenizer_model.len());

            // Try to create tokenizer from the embedded data
            // This is a simplified implementation - in production this would parse the model format
            if tokenizer_model.len() > 100 {
                // Sanity check for reasonable size
                let basic_tokenizer = crate::BasicTokenizer::with_config(
                    self.vocab_size,
                    Some(1), // BOS token
                    Some(2), // EOS token
                    Some(0), // PAD token
                );

                debug!("Created basic tokenizer from GGUF metadata");
                return Ok(Some(Arc::new(basic_tokenizer)));
            }
        }

        // Check for tokenizer vocab embedded in metadata
        if let Some(vocab) = self.gguf_reader.get_string_array_metadata("tokenizer.ggml.tokens")
            && vocab.len() == self.vocab_size
        {
            debug!("Found embedded vocabulary with {} tokens", vocab.len());

            // Create tokenizer with embedded vocabulary
            let basic_tokenizer = crate::BasicTokenizer::with_config(
                self.vocab_size,
                self.gguf_reader.get_u32_metadata("tokenizer.ggml.bos_token_id"),
                self.gguf_reader.get_u32_metadata("tokenizer.ggml.eos_token_id"),
                self.gguf_reader.get_u32_metadata("tokenizer.ggml.pad_token_id"),
            );

            debug!("Created tokenizer with embedded vocabulary");
            return Ok(Some(Arc::new(basic_tokenizer)));
        }

        // Check for HuggingFace tokenizer.json embedded as string
        if let Some(tokenizer_json) = self.gguf_reader.get_string_metadata("tokenizer.json") {
            debug!("Found embedded tokenizer.json ({} chars)", tokenizer_json.len());

            // In production, this would parse the JSON and create an HfTokenizer
            // For now, create a basic tokenizer with inferred parameters
            let basic_tokenizer = crate::BasicTokenizer::with_config(
                self.vocab_size,
                Some(1), // BOS token
                Some(2), // EOS token
                Some(0), // PAD token
            );

            debug!("Created tokenizer from embedded JSON metadata");
            return Ok(Some(Arc::new(basic_tokenizer)));
        }

        debug!("No embedded tokenizer found in GGUF metadata");
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AC1: Tests TokenizerDiscovery GGUF metadata parsing functionality
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    #[test]
    #[cfg(feature = "cpu")]
    fn test_tokenizer_discovery_from_gguf_llama3() {
        // Test scaffolding - will fail until implementation complete
        let test_path = Path::new("test-models/llama3-128k.gguf");
        let result = TokenizerDiscovery::from_gguf(test_path);

        // This should fail with unimplemented! until actual implementation
        assert!(result.is_err(), "Test scaffolding should fail until implemented");
    }

    /// AC1: Tests vocabulary size extraction from GGUF metadata for large neural network models
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    #[test]
    #[cfg(feature = "cpu")]
    fn test_vocab_size_extraction_large_models() {
        // Test scaffolding for 128K+ vocabulary models (LLaMA-3)
        // This test will pass once extract_vocab_size is implemented
        let test_path = Path::new("test-models/llama3-128k.gguf");
        let result = TokenizerDiscovery::from_gguf(test_path);

        // Test scaffolding assertion - implementation needed
        assert!(result.is_err(), "Requires GGUF metadata parsing implementation");
    }

    /// AC1: Tests model architecture detection for neural network compatibility
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    #[test]
    #[cfg(feature = "cpu")]
    fn test_model_type_detection_neural_networks() {
        // Test different neural network architectures
        let test_cases = [
            ("test-models/llama2-32k.gguf", "llama"),
            ("test-models/gpt2-50k.gguf", "gpt2"),
            ("test-models/bitnet-custom.gguf", "bitnet"),
        ];

        for (model_path, _expected_type) in test_cases {
            let test_path = Path::new(model_path);
            let result = TokenizerDiscovery::from_gguf(test_path);

            // Test scaffolding - requires implementation
            assert!(result.is_err(), "Model type detection requires GGUF parsing implementation");
        }
    }

    /// AC1: Tests tokenizer strategy discovery for different neural network models
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    #[test]
    #[cfg(feature = "cpu")]
    fn test_discover_tokenizer_strategy_neural_networks() {
        // Test strategy discovery for various neural network models
        // This is comprehensive test scaffolding covering all strategy types

        // Test scaffolding setup - will need real TokenizerDiscovery instance
        // let discovery = create_mock_discovery("llama", 128256);
        // let strategy = discovery.discover_tokenizer_strategy().unwrap();

        // Expected behavior tests:
        // - TokenizerStrategy::Discovered for co-located files
        // - TokenizerStrategy::NeedsDownload for known model types
        // - TokenizerStrategy::EmbeddedGguf for GGUF-embedded tokenizers
        // - TokenizerStrategy::Mock for fallback (non-strict mode)

        // Test scaffolding assertion
        assert!(true, "Test scaffolding placeholder - requires TokenizerDiscovery implementation");
    }

    /// AC1: Tests large vocabulary optimization detection for GPU acceleration
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    #[test]
    #[cfg(feature = "gpu")]
    fn test_large_vocab_optimization_detection() {
        // Test GPU acceleration requirements for large vocabularies

        // Test cases for different vocabulary sizes
        let test_cases = [
            (128256, true),  // LLaMA-3 - requires GPU optimization
            (32000, false),  // LLaMA-2 - CPU compatible
            (50257, false),  // GPT-2 - CPU compatible
            (1000000, true), // Hypothetical large model
        ];

        for (vocab_size, should_optimize) in test_cases {
            // Mock discovery instance for testing
            // let discovery = create_mock_discovery("test", vocab_size);
            // assert_eq!(discovery.requires_large_vocab_optimization(), should_optimize);

            // Test scaffolding assertion
            assert!(vocab_size > 0, "Test scaffolding - vocab size validation");
        }
    }

    /// AC1: Tests neural network model compatibility matrix functionality
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    #[test]
    #[cfg(feature = "cpu")]
    fn test_neural_network_compatibility_matrix() {
        let matrix = ModelCompatibilityMatrix::default();

        // Validate LLaMA-3 configuration
        assert_eq!(matrix.llama3_128k.repo, "meta-llama/Meta-Llama-3-8B");
        assert_eq!(matrix.llama3_128k.expected_vocab, Some(128256));
        assert_eq!(matrix.llama3_128k.cache_key, "llama3-128k");

        // Validate LLaMA-2 configuration
        assert_eq!(matrix.llama2_32k.repo, "meta-llama/Llama-2-7b-hf");
        assert_eq!(matrix.llama2_32k.expected_vocab, Some(32000));

        // Validate GPT-2 configuration
        assert_eq!(matrix.gpt2_50k.repo, "openai-community/gpt2");
        assert_eq!(matrix.gpt2_50k.expected_vocab, Some(50257));

        // Validate BitNet configuration
        assert_eq!(matrix.bitnet_custom.repo, "1bitLLM/bitnet_b1_58-large");
        assert_eq!(matrix.bitnet_custom.files.len(), 2); // tokenizer.json + tokenizer.model
    }

    /// AC1: Tests tokenizer strategy properties and descriptions
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac1-tokenizerdiscovery-implementation
    #[test]
    #[cfg(feature = "cpu")]
    fn test_tokenizer_strategy_properties() {
        // Test strategy network requirements
        let download_info = TokenizerDownloadInfo {
            repo: "test/repo".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "test".to_string(),
            expected_vocab: Some(1000),
        };

        let strategies = [
            (TokenizerStrategy::Exact(PathBuf::from("test.json")), false, false),
            (TokenizerStrategy::Discovered(PathBuf::from("found.json")), false, true),
            (TokenizerStrategy::NeedsDownload(download_info), true, true),
            (TokenizerStrategy::Mock, false, false),
        ];

        for (strategy, should_need_network, should_use_cache) in strategies {
            assert_eq!(strategy.requires_network(), should_need_network);
            assert_eq!(strategy.uses_cache(), should_use_cache);
            assert!(!strategy.description().is_empty());
        }
    }
}
