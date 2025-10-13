# Issue #249: Complete Tokenizer Integration and Automatic Discovery - Neural Network Technical Specification

## Executive Summary

This specification defines the technical implementation approach for automatic tokenizer discovery and integration in BitNet.rs neural network inference engine. The implementation will enable seamless model inference without manual tokenizer configuration, supporting production-grade neural networks with large vocabularies (LLaMA-3: 128K tokens, LLaMA-2: 32K tokens, GPT-2: 50K tokens) and proper device-aware quantization compatibility.

**Flow Status**: generative:gate:spec = pass (Neural Network Implementation Analysis Complete)

## Neural Network Context Analysis

### Current BitNet.rs Inference Pipeline
```
Model Loading → Quantization → Kernels → Inference → Output
     ↓             ↓           ↓         ↓        ↓
  GGUF Parse   I2S/TL1/TL2   GPU/CPU   Stream   Tokens
     ↓             ↓           ↓         ↓        ↓
[TOKENIZER DISCOVERY INJECTION POINT]    ←← CRITICAL INTEGRATION POINT
```

### Neural Network Requirements Analysis

**Vocabulary Scale Impact on Quantization**:
- **LLaMA-3** (128,256 vocab): Requires I2S quantization with GPU acceleration for efficient embedding lookup
- **LLaMA-2** (32,000 vocab): Compatible with TL1/TL2 quantization, CPU-optimized embedding tables
- **GPT-2** (50,257 vocab): Standard BPE tokenization with mixed precision support

**Quantization Format Compatibility**:
- **I2S**: Native 2-bit signed quantization - optimal for large vocabularies with GPU acceleration
- **TL1/TL2**: Table lookup quantization - efficient for smaller vocabularies with vectorized operations
- **IQ2_S**: GGML-compatible quantization with 82-byte blocks - universal compatibility

**Performance Considerations**:
- **Token Lookup**: O(1) byte lookup performance critical for large vocabularies
- **Memory Bandwidth**: Device-aware tokenization to minimize GPU/CPU transfer overhead
- **Quantization Overhead**: Tokenizer discovery must not impact I2S/TL1/TL2 quantization pipeline

## Requirements Analysis

### AC1: TokenizerDiscovery Implementation
**Neural Network Requirement**: Automatic detection of tokenizer type from GGUF model metadata without manual configuration.

**Implementation Strategy**:
```rust
// bitnet-tokenizers/src/discovery.rs
pub struct TokenizerDiscovery {
    gguf_reader: Arc<GgufReader<'static>>,
    model_path: PathBuf,
    vocab_size: usize,
    model_type: String,
}

impl TokenizerDiscovery {
    pub fn from_gguf(path: &Path) -> Result<Self> {
        let mmap = MmapFile::open(path)?;
        let reader = GgufReader::new(mmap.as_slice())?;

        // Extract tokenizer metadata from GGUF
        let vocab_size = Self::extract_vocab_size(&reader)?;
        let model_type = Self::extract_model_type(&reader)?;

        Ok(Self {
            gguf_reader: Arc::new(reader),
            model_path: path.to_path_buf(),
            vocab_size,
            model_type,
        })
    }

    pub fn discover_tokenizer_strategy(&self) -> Result<TokenizerStrategy> {
        // 1. Check co-located files first
        if let Ok(colocated) = self.check_colocated_tokenizers() {
            return Ok(TokenizerStrategy::Discovered(colocated));
        }

        // 2. Check standard cache locations
        if let Ok(cached) = self.check_cache_locations() {
            return Ok(TokenizerStrategy::Discovered(cached));
        }

        // 3. Determine smart download strategy
        if let Some(download_info) = self.infer_download_source()? {
            return Ok(TokenizerStrategy::NeedsDownload(download_info));
        }

        // 4. Fallback to mock (non-strict mode only)
        if std::env::var("BITNET_STRICT_TOKENIZERS").as_deref() != Ok("1") {
            return Ok(TokenizerStrategy::Mock);
        }

        Err(BitNetError::Inference(InferenceError::TokenizationFailed {
            reason: "No compatible tokenizer found and strict mode enabled".to_string(),
        }))
    }

    fn extract_vocab_size(reader: &GgufReader) -> Result<usize> {
        // Parse vocab from GGUF metadata
        if let Some(tokens) = reader.get_string_array_metadata("tokenizer.ggml.tokens") {
            return Ok(tokens.len());
        }

        // Fallback to architecture-specific vocab size
        if let Some(arch) = reader.get_string_metadata("general.architecture") {
            match arch.as_str() {
                "llama" => {
                    if let Some(vocab_size) = reader.get_u32_metadata("llama.vocab_size") {
                        return Ok(vocab_size as usize);
                    }
                }
                "gpt2" => return Ok(50257), // GPT-2 standard vocab
                _ => {}
            }
        }

        Err(BitNetError::Model(ModelError::LoadingFailed {
            reason: "Cannot determine vocab size from GGUF metadata".to_string(),
        }))
    }

    fn infer_download_source(&self) -> Result<Option<TokenizerDownloadInfo>> {
        // Neural network model compatibility matrix
        match (self.model_type.as_str(), self.vocab_size) {
            ("llama", 128256) => Ok(Some(TokenizerDownloadInfo {
                repo: "meta-llama/Meta-Llama-3-8B".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "llama3-128k".to_string(),
            })),
            ("llama", 32000) => Ok(Some(TokenizerDownloadInfo {
                repo: "meta-llama/Llama-2-7b-hf".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "llama2-32k".to_string(),
            })),
            ("gpt2", 50257) => Ok(Some(TokenizerDownloadInfo {
                repo: "openai-community/gpt2".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "gpt2-50k".to_string(),
            })),
            _ => Ok(None), // Unknown combination
        }
    }
}
```

**Risk Assessment**: Low risk - leverages existing GGUF reader infrastructure.

**Validation Commands**:
```bash
cargo test --no-default-features -p bitnet-tokenizers test_tokenizer_discovery_from_gguf --no-default-features --features cpu
cargo test --no-default-features -p bitnet-tokenizers test_vocab_size_extraction --no-default-features --features cpu
```

### AC2: SmartTokenizerDownload Implementation
**Neural Network Requirement**: Automatic downloading of missing tokenizer files when referenced in model metadata.

**Implementation Strategy**:
```rust
// bitnet-tokenizers/src/downloader.rs
pub struct SmartTokenizerDownload {
    cache_dir: PathBuf,
    client: reqwest::Client,
}

impl SmartTokenizerDownload {
    pub fn new() -> Result<Self> {
        let cache_dir = Self::cache_directory()?;
        let client = reqwest::Client::builder()
            .user_agent("bitnet-rs/0.1.0")
            .timeout(std::time::Duration::from_secs(300))
            .build()?;

        Ok(Self { cache_dir, client })
    }

    pub async fn download_tokenizer(&self, info: &TokenizerDownloadInfo) -> Result<PathBuf> {
        let cache_path = self.cache_dir.join(&info.cache_key);

        // Check if already cached
        if let Ok(cached) = self.find_cached_tokenizer(&cache_path) {
            tracing::info!("Using cached tokenizer: {}", cached.display());
            return Ok(cached);
        }

        // Create cache directory
        std::fs::create_dir_all(&cache_path)?;

        // Download all required files
        for file in &info.files {
            let url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                info.repo, file
            );

            let file_path = cache_path.join(file);
            self.download_file(&url, &file_path).await?;
        }

        // Return primary tokenizer file
        Ok(cache_path.join("tokenizer.json"))
    }

    async fn download_file(&self, url: &str, path: &Path) -> Result<()> {
        tracing::info!("Downloading tokenizer: {}", url);

        let response = self.client.get(url).send().await?;
        if !response.status().is_success() {
            return Err(BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("HTTP {} for {}", response.status(), url),
            }));
        }

        let bytes = response.bytes().await?;
        std::fs::write(path, bytes)?;

        tracing::info!("Downloaded tokenizer to: {}", path.display());
        Ok(())
    }

    fn cache_directory() -> Result<PathBuf> {
        let cache = dirs::cache_dir()
            .ok_or_else(|| BitNetError::Model(ModelError::LoadingFailed {
                reason: "Cannot determine cache directory".to_string(),
            }))?
            .join("bitnet")
            .join("tokenizers");

        Ok(cache)
    }
}

#[derive(Debug, Clone)]
pub struct TokenizerDownloadInfo {
    pub repo: String,
    pub files: Vec<String>,
    pub cache_key: String,
}
```

**Neural Network Considerations**:
- **Memory Efficiency**: Downloads are cached to minimize network overhead for large tokenizer files
- **Async Integration**: Non-blocking downloads compatible with inference pipeline
- **Error Recovery**: Graceful degradation to cached or mock tokenizers on network failure

**Validation Commands**:
```bash
cargo test --no-default-features -p bitnet-tokenizers test_smart_tokenizer_download --no-default-features --features cpu
cargo test --no-default-features -p bitnet-tokenizers test_download_cache_management --no-default-features --features cpu
```

### AC3: Production TokenizerStrategy Implementation
**Neural Network Requirement**: Production-ready tokenizer strategy implementations for LLaMA-2/3 and GPT-2 with proper special token handling.

**Implementation Strategy**:
```rust
// bitnet-tokenizers/src/strategy.rs
#[derive(Debug, Clone)]
pub enum TokenizerStrategy {
    Exact(PathBuf),              // User-specified tokenizer path
    Discovered(PathBuf),         // Auto-discovered compatible tokenizer
    NeedsDownload(TokenizerDownloadInfo), // Smart download required
    Mock,                        // Testing fallback (non-strict mode)
}

pub struct TokenizerStrategyResolver {
    discovery: TokenizerDiscovery,
    downloader: SmartTokenizerDownload,
}

impl TokenizerStrategyResolver {
    pub async fn resolve_tokenizer(&self, strategy: TokenizerStrategy) -> Result<Arc<dyn Tokenizer>> {
        match strategy {
            TokenizerStrategy::Exact(path) => {
                tracing::info!("Loading exact tokenizer: {}", path.display());
                self.load_tokenizer_from_path(&path)
            }

            TokenizerStrategy::Discovered(path) => {
                tracing::info!("Loading discovered tokenizer: {}", path.display());
                self.load_tokenizer_from_path(&path)
            }

            TokenizerStrategy::NeedsDownload(info) => {
                tracing::info!("Downloading tokenizer: {}", info.repo);
                let downloaded_path = self.downloader.download_tokenizer(&info).await?;
                self.load_tokenizer_from_path(&downloaded_path)
            }

            TokenizerStrategy::Mock => {
                if std::env::var("BITNET_STRICT_TOKENIZERS").as_deref() == Ok("1") {
                    return Err(BitNetError::Inference(InferenceError::TokenizationFailed {
                        reason: "Mock tokenizer disabled in strict mode".to_string(),
                    }));
                }
                tracing::warn!("Using mock tokenizer - real inference not possible");
                Ok(Arc::new(MockTokenizer::with_vocab_size(self.discovery.vocab_size)))
            }
        }
    }

    fn load_tokenizer_from_path(&self, path: &Path) -> Result<Arc<dyn Tokenizer>> {
        // Use existing tokenizer loading infrastructure
        let (tokenizer, kind) = bitnet_tokenizers::from_path(path)?;

        // Apply neural network model-specific configurations
        match self.discovery.model_type.as_str() {
            "llama" => self.configure_llama_tokenizer(tokenizer),
            "gpt2" => self.configure_gpt2_tokenizer(tokenizer),
            _ => Ok(tokenizer), // Use as-is for unknown types
        }
    }

    fn configure_llama_tokenizer(&self, tokenizer: Arc<dyn Tokenizer>) -> Result<Arc<dyn Tokenizer>> {
        // LLaMA-specific tokenizer configuration for neural network inference
        // - Proper BOS/EOS token handling
        // - Byte fallback for unknown tokens
        // - Space prefix handling
        Ok(Arc::new(LlamaTokenizerWrapper::new(tokenizer, self.discovery.vocab_size)?))
    }

    fn configure_gpt2_tokenizer(&self, tokenizer: Arc<dyn Tokenizer>) -> Result<Arc<dyn Tokenizer>> {
        // GPT-2 specific configuration
        // - No BOS token
        // - EOS token = 50256
        // - Standard BPE encoding
        Ok(Arc::new(Gpt2TokenizerWrapper::new(tokenizer)?))
    }
}

// Neural network model-specific tokenizer wrappers
pub struct LlamaTokenizerWrapper {
    inner: Arc<dyn Tokenizer>,
    vocab_size: usize,
}

impl Tokenizer for LlamaTokenizerWrapper {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        let mut tokens = self.inner.encode(text, false, false)?;

        // Add BOS token for LLaMA models
        if add_bos {
            tokens.insert(0, 1); // LLaMA BOS token ID
        }

        // Validate token IDs are within vocab range
        for &token in &tokens {
            if token as usize >= self.vocab_size {
                return Err(BitNetError::Inference(InferenceError::TokenizationFailed {
                    reason: format!("Token ID {} exceeds vocab size {}", token, self.vocab_size),
                }));
            }
        }

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner.decode(tokens)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.inner.token_to_piece(token)
    }

    fn bos_token_id(&self) -> Option<u32> {
        Some(1) // LLaMA BOS token
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(2) // LLaMA EOS token
    }
}
```

**Neural Network Validation**:
- **Large Vocabulary Handling**: Efficient token validation for 128K+ vocabularies
- **Special Token Correctness**: Proper BOS/EOS handling for different model architectures
- **Quantization Compatibility**: Token IDs within valid ranges for I2S/TL1/TL2 quantization

**Validation Commands**:
```bash
cargo test --no-default-features -p bitnet-tokenizers test_llama_tokenizer_wrapper --no-default-features --features cpu
cargo test --no-default-features -p bitnet-tokenizers test_gpt2_tokenizer_wrapper --no-default-features --features cpu
cargo test --no-default-features -p bitnet-tokenizers test_vocab_size_validation --no-default-features --features cpu
```

### AC4: xtask Integration
**Neural Network Requirement**: Integrate automatic tokenizer discovery with `cargo xtask infer` command for seamless model inference.

**Implementation Strategy**:
```rust
// xtask/src/commands/infer.rs (modifications)
pub async fn run_infer(args: &InferArgs) -> Result<()> {
    let model_path = &args.model;

    // Enhanced tokenizer resolution with automatic discovery
    let tokenizer = if let Some(tokenizer_path) = &args.tokenizer {
        // User-specified tokenizer (existing behavior)
        let strategy = TokenizerStrategy::Exact(tokenizer_path.clone());
        TokenizerStrategyResolver::new(model_path).await?.resolve_tokenizer(strategy).await?
    } else {
        // NEW: Automatic tokenizer discovery
        let discovery = TokenizerDiscovery::from_gguf(model_path)?;
        let strategy = discovery.discover_tokenizer_strategy()?;

        match strategy {
            TokenizerStrategy::NeedsDownload(ref info) => {
                if !args.auto_download {
                    return Err(anyhow::anyhow!(
                        "Model requires tokenizer from {}. Use --auto-download or specify --tokenizer path",
                        info.repo
                    ));
                }

                eprintln!("🔍 Auto-discovering tokenizer for model...");
                eprintln!("📥 Downloading tokenizer from {}...", info.repo);
            }
            TokenizerStrategy::Mock => {
                if !args.allow_mock {
                    return Err(anyhow::anyhow!(
                        "No compatible tokenizer found. Use --allow-mock for testing or specify --tokenizer path"
                    ));
                }
                eprintln!("⚠️  Using mock tokenizer - real inference not possible");
            }
            _ => {}
        }

        let resolver = TokenizerStrategyResolver::new(discovery).await?;
        resolver.resolve_tokenizer(strategy).await?
    };

    // Continue with existing inference logic...
    let model = load_model(model_path)?;
    let inference_engine = InferenceEngine::new(model, tokenizer)?;

    // Apply deterministic settings if requested
    if args.deterministic {
        std::env::set_var("RAYON_NUM_THREADS", "1");
        std::env::set_var("BITNET_DETERMINISTIC", "1");
    }

    let result = inference_engine.generate(&args.prompt, &args.into())?;

    match args.format {
        OutputFormat::Human => println!("{}", result.text),
        OutputFormat::Json => println!("{}", serde_json::to_string(&result)?),
    }

    Ok(())
}

// Enhanced command line arguments
#[derive(Debug, Clone, Args)]
pub struct InferArgs {
    // ... existing fields ...

    #[arg(long, help = "Automatically download missing tokenizers")]
    pub auto_download: bool,

    #[arg(long, help = "Use strict tokenizer mode (no mock fallbacks)")]
    pub strict: bool,
}
```

**Neural Network Integration Points**:
- **Seamless UX**: Zero-configuration inference for supported neural network models
- **Performance**: Tokenizer discovery occurs once and cached for subsequent runs
- **Compatibility**: Maintains backward compatibility with existing --tokenizer flag

**Validation Commands**:
```bash
cargo test --no-default-features -p xtask test_infer_auto_discovery --no-default-features --features cpu
cargo run -p xtask -- infer --model models/ggml-model-i2_s.gguf --prompt "Test" --auto-download
```

### AC5: Fallback Strategy System
**Neural Network Requirement**: Robust fallback strategy system with proper error reporting.

**Implementation Strategy**:
```rust
// bitnet-tokenizers/src/fallback.rs
pub struct TokenizerFallbackChain {
    strategies: Vec<FallbackStrategy>,
    strict_mode: bool,
}

#[derive(Debug)]
enum FallbackStrategy {
    GgufMetadata,
    ColocatedFiles,
    StandardCache,
    SmartDownload,
    MockFallback,
}

impl TokenizerFallbackChain {
    pub fn new() -> Self {
        let strict_mode = std::env::var("BITNET_STRICT_TOKENIZERS").as_deref() == Ok("1");

        let strategies = if strict_mode {
            vec![
                FallbackStrategy::GgufMetadata,
                FallbackStrategy::ColocatedFiles,
                FallbackStrategy::StandardCache,
                FallbackStrategy::SmartDownload,
                // No MockFallback in strict mode
            ]
        } else {
            vec![
                FallbackStrategy::GgufMetadata,
                FallbackStrategy::ColocatedFiles,
                FallbackStrategy::StandardCache,
                FallbackStrategy::SmartDownload,
                FallbackStrategy::MockFallback,
            ]
        };

        Self { strategies, strict_mode }
    }

    pub async fn resolve_tokenizer(&self, discovery: &TokenizerDiscovery) -> Result<TokenizerResolution> {
        let mut errors = Vec::new();

        for strategy in &self.strategies {
            match self.try_strategy(strategy, discovery).await {
                Ok(resolution) => {
                    tracing::info!("Tokenizer resolved using strategy: {:?}", strategy);
                    return Ok(resolution);
                }
                Err(e) => {
                    tracing::debug!("Strategy {:?} failed: {}", strategy, e);
                    errors.push((strategy, e));
                }
            }
        }

        // All strategies failed - generate comprehensive error message
        let error_summary = self.generate_error_summary(&errors);
        Err(BitNetError::Inference(InferenceError::TokenizationFailed {
            reason: error_summary,
        }))
    }

    async fn try_strategy(&self, strategy: &FallbackStrategy, discovery: &TokenizerDiscovery) -> Result<TokenizerResolution> {
        match strategy {
            FallbackStrategy::GgufMetadata => {
                // Try to extract tokenizer from GGUF metadata directly
                if let Some(embedded_tokenizer) = discovery.try_extract_embedded_tokenizer()? {
                    return Ok(TokenizerResolution::Embedded(embedded_tokenizer));
                }
                Err(BitNetError::Model(ModelError::LoadingFailed {
                    reason: "No embedded tokenizer in GGUF".to_string(),
                }))
            }

            FallbackStrategy::ColocatedFiles => {
                if let Some(colocated) = discovery.check_colocated_tokenizers()? {
                    return Ok(TokenizerResolution::File(colocated));
                }
                Err(BitNetError::Model(ModelError::LoadingFailed {
                    reason: "No colocated tokenizer files".to_string(),
                }))
            }

            FallbackStrategy::StandardCache => {
                if let Some(cached) = discovery.check_cache_locations()? {
                    return Ok(TokenizerResolution::File(cached));
                }
                Err(BitNetError::Model(ModelError::LoadingFailed {
                    reason: "No cached tokenizer found".to_string(),
                }))
            }

            FallbackStrategy::SmartDownload => {
                if let Some(download_info) = discovery.infer_download_source()? {
                    let downloader = SmartTokenizerDownload::new()?;
                    let downloaded_path = downloader.download_tokenizer(&download_info).await?;
                    return Ok(TokenizerResolution::File(downloaded_path));
                }
                Err(BitNetError::Model(ModelError::LoadingFailed {
                    reason: "No known download source for model".to_string(),
                }))
            }

            FallbackStrategy::MockFallback => {
                if self.strict_mode {
                    return Err(BitNetError::Inference(InferenceError::TokenizationFailed {
                        reason: "Mock fallback disabled in strict mode".to_string(),
                    }));
                }
                Ok(TokenizerResolution::Mock(MockTokenizer::with_vocab_size(discovery.vocab_size)))
            }
        }
    }

    fn generate_error_summary(&self, errors: &[(FallbackStrategy, BitNetError)]) -> String {
        let mut summary = String::from("All tokenizer resolution strategies failed:\n");

        for (strategy, error) in errors {
            summary.push_str(&format!("  {:?}: {}\n", strategy, error));
        }

        summary.push_str("\nSuggestions:\n");
        summary.push_str("  1. Place tokenizer.json in the same directory as the model\n");
        summary.push_str("  2. Use --tokenizer path/to/tokenizer.json to specify manually\n");

        if self.strict_mode {
            summary.push_str("  3. Remove BITNET_STRICT_TOKENIZERS=1 to enable mock fallback\n");
        } else {
            summary.push_str("  3. Use --allow-mock for testing (produces placeholder output)\n");
        }

        summary
    }
}

#[derive(Debug)]
pub enum TokenizerResolution {
    File(PathBuf),
    Embedded(Arc<dyn Tokenizer>),
    Mock(MockTokenizer),
}
```

**Error Handling Strategy**:
- **Actionable Messages**: Clear guidance on how to resolve tokenizer issues
- **Progressive Degradation**: Graceful fallback without silent failures
- **Debug Information**: Detailed logging for troubleshooting complex scenarios

**Validation Commands**:
```bash
# Test fallback chain with various scenarios
cargo test --no-default-features -p bitnet-tokenizers test_fallback_chain_success --no-default-features --features cpu
cargo test --no-default-features -p bitnet-tokenizers test_fallback_chain_strict_mode --no-default-features --features cpu
BITNET_STRICT_TOKENIZERS=1 cargo test --no-default-features -p bitnet-tokenizers test_no_mock_in_strict_mode --no-default-features --features cpu
```

### AC6: Cross-Validation Tests
**Neural Network Requirement**: Verify tokenizer compatibility against existing universal tokenizer architecture.

**Implementation Strategy**:
```rust
// bitnet-tokenizers/tests/tokenizer_discovery_crossval.rs
use bitnet_tokenizers::*;

#[tokio::test]
async fn test_tokenizer_discovery_cross_validation() -> Result<()> {
    // AC6: Cross-validation against existing universal tokenizer
    let test_cases = [
        ("models/test-llama3.gguf", 128256, "llama"),
        ("models/test-llama2.gguf", 32000, "llama"),
        ("models/test-gpt2.gguf", 50257, "gpt2"),
    ];

    for (model_path, expected_vocab, expected_type) in test_cases {
        let path = Path::new(model_path);
        if !path.exists() {
            continue; // Skip if test model not available
        }

        // Test new discovery system
        let discovery = TokenizerDiscovery::from_gguf(path)?;
        assert_eq!(discovery.vocab_size, expected_vocab);
        assert_eq!(discovery.model_type, expected_type);

        let strategy = discovery.discover_tokenizer_strategy()?;
        let resolver = TokenizerStrategyResolver::new(discovery).await?;
        let discovered_tokenizer = resolver.resolve_tokenizer(strategy).await?;

        // Cross-validate against existing UniversalTokenizer
        let universal_tokenizer = UniversalTokenizer::from_gguf(path)?;

        // Test with same inputs
        let test_texts = [
            "Hello world",
            "The quick brown fox",
            "Neural network inference with BitNet",
        ];

        for text in test_texts {
            let discovered_tokens = discovered_tokenizer.encode(text, true, true)?;
            let universal_tokens = universal_tokenizer.encode(text, true, true)?;

            // Tokens should be compatible (allowing for minor differences in special token handling)
            assert_token_compatibility(&discovered_tokens, &universal_tokens, expected_vocab);

            // Decode should produce similar results
            let discovered_decoded = discovered_tokenizer.decode(&discovered_tokens)?;
            let universal_decoded = universal_tokenizer.decode(&universal_tokens)?;
            assert_text_similarity(&discovered_decoded, &universal_decoded);
        }

        println!("✅ Cross-validation passed for {}", model_path);
    }

    Ok(())
}

fn assert_token_compatibility(discovered: &[u32], universal: &[u32], vocab_size: usize) {
    // All tokens should be within vocab range
    for &token in discovered {
        assert!((token as usize) < vocab_size, "Token {} exceeds vocab size {}", token, vocab_size);
    }

    // Length should be similar (within reasonable bounds)
    let len_diff = (discovered.len() as i32 - universal.len() as i32).abs();
    assert!(len_diff <= 2, "Token length difference too large: {} vs {}", discovered.len(), universal.len());
}

fn assert_text_similarity(discovered: &str, universal: &str) {
    // For mock tokenizers, allow generic output
    if discovered.starts_with("Generated text from") || universal.starts_with("Generated text from") {
        return; // Skip similarity check for mock tokenizers
    }

    // Real tokenizers should produce similar output
    let discovered_words: Vec<&str> = discovered.split_whitespace().collect();
    let universal_words: Vec<&str> = universal.split_whitespace().collect();

    // Allow some variation in decoded output
    let similarity = text_similarity(&discovered_words, &universal_words);
    assert!(similarity > 0.7, "Text similarity too low: {:.2} for '{}' vs '{}'", similarity, discovered, universal);
}

#[tokio::test]
async fn test_quantization_compatibility() -> Result<()> {
    // AC6: Test tokenizer compatibility with different quantization formats
    use bitnet_quantization::*;

    let discovery = TokenizerDiscovery::from_gguf(Path::new("models/test-llama3.gguf"))?;
    if discovery.vocab_size != 128256 {
        return Ok(()); // Skip if not LLaMA-3 model
    }

    let strategy = discovery.discover_tokenizer_strategy()?;
    let resolver = TokenizerStrategyResolver::new(discovery).await?;
    let tokenizer = resolver.resolve_tokenizer(strategy).await?;

    // Test with different quantization types
    let quantization_types = [
        QuantizationType::I2S,
        QuantizationType::TL1,
        QuantizationType::TL2,
    ];

    for quant_type in quantization_types {
        let tokens = tokenizer.encode("Test quantization compatibility", true, true)?;

        // Validate all tokens are within quantization-safe ranges
        match quant_type {
            QuantizationType::I2S => {
                // I2S supports full vocab range with GPU acceleration
                assert!(tokens.iter().all(|&t| (t as usize) < 128256));
            }
            QuantizationType::TL1 | QuantizationType::TL2 => {
                // TL1/TL2 may have lookup table size constraints
                assert!(tokens.iter().all(|&t| (t as usize) < 65536)); // 16-bit lookup table limit
            }
        }

        println!("✅ Quantization compatibility verified for {:?}", quant_type);
    }

    Ok(())
}
```

**Cross-Validation Strategy**:
- **Backward Compatibility**: Ensure new discovery system maintains compatibility with existing UniversalTokenizer
- **Quantization Integration**: Validate tokenizer outputs work with I2S/TL1/TL2 quantization pipelines
- **Performance Regression**: No degradation in tokenization performance

**Validation Commands**:
```bash
cargo test --no-default-features -p bitnet-tokenizers --no-default-features --features cpu tokenizer_discovery_crossval
cargo test --no-default-features --test crossval_integration --no-default-features --features cpu
```

### AC7: Integration Tests with Real Models
**Neural Network Requirement**: End-to-end validation with real model files.

**Implementation Strategy**:
```rust
// tests/tokenizer_integration_e2e.rs
#[tokio::test]
async fn test_end_to_end_tokenizer_discovery_integration() -> Result<()> {
    // AC7: Integration test with real models

    // Download a small test model if not available
    let model_path = setup_test_model().await?;

    // Test complete workflow: Discovery → Download → Inference
    let result = tokio::process::Command::new("cargo")
        .args(&[
            "run", "-p", "xtask", "--",
            "infer",
            "--model", model_path.to_str().unwrap(),
            "--prompt", "The capital of France is",
            "--max-new-tokens", "5",
            "--auto-download",
            "--deterministic"
        ])
        .output()
        .await?;

    assert!(result.status.success(), "xtask infer failed: {}",
        String::from_utf8_lossy(&result.stderr));

    let output = String::from_utf8(result.stdout)?;

    // Verify output contains reasonable text (not mock tokenizer output)
    assert!(!output.contains("Generated text from"), "Mock tokenizer was used unexpectedly");
    assert!(!output.is_empty(), "No inference output generated");

    // Verify tokenizer was automatically resolved
    let stderr = String::from_utf8(result.stderr)?;
    assert!(stderr.contains("Auto-discovering tokenizer") ||
           stderr.contains("Loading discovered tokenizer") ||
           stderr.contains("Using cached tokenizer"));

    println!("✅ End-to-end integration test passed");
    println!("Output: {}", output);

    Ok(())
}

async fn setup_test_model() -> Result<PathBuf> {
    let model_dir = Path::new("models/test");
    std::fs::create_dir_all(model_dir)?;

    let model_path = model_dir.join("test-model.gguf");

    if !model_path.exists() {
        // Download or generate a minimal test model
        let result = tokio::process::Command::new("cargo")
            .args(&[
                "run", "-p", "xtask", "--",
                "gen-mini-gguf",
                "--output", model_path.to_str().unwrap(),
                "--vocab-size", "32000",
                "--model-type", "llama"
            ])
            .output()
            .await?;

        assert!(result.status.success(), "Failed to generate test model");
    }

    Ok(model_path)
}

#[tokio::test]
async fn test_gpu_cpu_tokenizer_parity() -> Result<()> {
    // AC7: Test tokenizer works with both GPU and CPU inference

    let model_path = setup_test_model().await?;

    // Test CPU inference
    let cpu_result = tokio::process::Command::new("cargo")
        .args(&[
            "run", "-p", "xtask", "--",
            "infer",
            "--model", model_path.to_str().unwrap(),
            "--prompt", "Test prompt",
            "--max-new-tokens", "10",
            "--auto-download",
            "--deterministic",
            // No --gpu flag for CPU inference
        ])
        .output()
        .await?;

    assert!(cpu_result.status.success(), "CPU inference failed");
    let cpu_output = String::from_utf8(cpu_result.stdout)?;

    // Test GPU inference (if available)
    let gpu_result = tokio::process::Command::new("cargo")
        .args(&[
            "run", "-p", "xtask", "--",
            "infer",
            "--model", model_path.to_str().unwrap(),
            "--prompt", "Test prompt",
            "--max-new-tokens", "10",
            "--auto-download",
            "--deterministic",
            "--gpu"
        ])
        .output()
        .await;

    if let Ok(gpu_result) = gpu_result {
        if gpu_result.status.success() {
            let gpu_output = String::from_utf8(gpu_result.stdout)?;

            // With deterministic settings, outputs should be identical
            assert_eq!(cpu_output.trim(), gpu_output.trim(),
                "GPU and CPU inference outputs differ with same tokenizer");

            println!("✅ GPU/CPU tokenizer parity verified");
        } else {
            println!("⚠️  GPU inference not available, skipping parity test");
        }
    }

    Ok(())
}
```

**Validation Commands**:
```bash
# Full integration test suite
cargo test --no-default-features --test tokenizer_integration_e2e --no-default-features --features cpu

# Real model verification
cargo run -p xtask -- verify --model models/test/test-model.gguf --expect-tokenizer-auto-discovery
```

### AC8-AC10: Documentation, Determinism, and Error Handling

**AC8 Implementation**: Comprehensive documentation with neural network context
**AC9 Implementation**: Deterministic behavior support with reproducible tokenization
**AC10 Implementation**: Robust error handling with actionable messages

## Neural Network Implementation Architecture

### Crate-Level Integration

**bitnet-tokenizers** (Primary):
- `TokenizerDiscovery`: GGUF metadata parsing and strategy resolution
- `SmartTokenizerDownload`: Intelligent downloading with caching
- `TokenizerStrategyResolver`: Unified resolution interface
- `TokenizerFallbackChain`: Robust fallback system

**bitnet-models** (GGUF Integration):
- Enhanced GGUF reader with tokenizer metadata extraction
- Validation of tokenizer compatibility with model architecture
- Memory-mapped model loading with tokenizer discovery

**bitnet-inference** (Pipeline Integration):
- Seamless tokenizer integration in inference pipeline
- Device-aware tokenization for GPU/CPU backends
- Performance optimization for large vocabulary models

**xtask** (CLI Integration):
- Enhanced `infer` command with automatic discovery
- Progress indicators for downloads and discovery
- Comprehensive error messages with suggestions

### Performance Specifications

**Neural Network Scale Requirements**:
- **Large Vocabulary Support**: Efficient handling of 128K+ token vocabularies (LLaMA-3)
- **Memory Efficiency**: O(1) token lookup performance with device-aware memory management
- **Quantization Compatibility**: Seamless integration with I2S/TL1/TL2 quantization pipelines
- **GPU Acceleration**: Automatic GPU/CPU selection with mixed precision support

**Throughput Targets**:
- **Tokenizer Discovery**: <100ms for cached tokenizers, <5s for downloads
- **Token Encoding**: >10K tokens/sec for large vocabularies on GPU
- **Memory Usage**: <100MB additional overhead for tokenizer caching
- **Network Efficiency**: Resume capability for large tokenizer downloads

### Feature Flag Analysis

**Build Configurations**:
```bash
# CPU-only build with SentencePiece support
cargo build --no-default-features --features cpu,spm

# GPU build with all tokenizer formats
cargo build --no-default-features --features gpu,spm,ffi

# WebAssembly build with browser compatibility
cargo build --target wasm32-unknown-unknown --no-default-features --features browser
```

**Feature Dependencies**:
- `cpu`: Basic tokenizer discovery and HuggingFace JSON support
- `gpu`: GPU-accelerated tokenization for large vocabularies
- `smp`: SentencePiece tokenizer support via feature flag
- `ffi`: C++ bridge compatibility for cross-validation
- `browser`: WebAssembly-compatible tokenizer discovery

### Risk Assessment and Mitigation

**Technical Risks**:

1. **Network Dependency Risk**:
   - *Impact*: Tokenizer downloads may fail in restricted environments
   - *Mitigation*: Comprehensive caching strategy, offline mode support, graceful degradation to cached tokenizers
   - *Validation*: `BITNET_OFFLINE=1 cargo test` simulates network-free environment

2. **Large Vocabulary Performance Risk**:
   - *Impact*: 128K+ vocabularies may impact tokenization performance
   - *Mitigation*: GPU acceleration for large vocabularies, efficient O(1) lookup tables, memory-mapped tokenizer files
   - *Validation*: Performance benchmarks with `cargo run -p xtask -- benchmark --vocab-stress-test`

3. **GGUF Metadata Compatibility Risk**:
   - *Impact*: Model files may have inconsistent or missing tokenizer metadata
   - *Mitigation*: Robust fallback chain, metadata validation, compatibility fixes for common issues
   - *Validation*: `cargo run -p bitnet-cli -- compat-check <model.gguf>` validates metadata

4. **Quantization Integration Risk**:
   - *Impact*: New tokenizer system may not integrate properly with I2S/TL1/TL2 quantization
   - *Mitigation*: Comprehensive cross-validation tests, token range validation, device-aware quantization
   - *Validation*: `cargo test -p bitnet-quantization test_tokenizer_quantization_compatibility`

5. **Cross-Platform Compatibility Risk**:
   - *Impact*: Tokenizer discovery may behave differently across platforms
   - *Mitigation*: Platform-specific caching strategies, WebAssembly compatibility, comprehensive CI testing
   - *Validation*: Multi-platform CI with `cargo test --target wasm32-unknown-unknown`

### Success Criteria and Validation

**Measurable Acceptance Criteria**:

1. **Zero-Configuration Success Rate**: >95% of supported models work without manual tokenizer specification
2. **Performance Benchmarks**: Tokenizer discovery adds <5% overhead to inference pipeline
3. **Network Efficiency**: Smart caching reduces duplicate downloads by >90%
4. **Error Rate Reduction**: Actionable error messages reduce user support requests by >80%
5. **Cross-Validation Accuracy**: 100% compatibility with existing UniversalTokenizer for supported models

**Validation Commands**:
```bash
# Comprehensive test suite
cargo test --no-default-features --workspace --no-default-features --features cpu tokenizer
./scripts/verify-tests.sh

# Performance validation
cargo run -p xtask -- benchmark --tokenizer-discovery-overhead
RUSTFLAGS="-C target-cpu=native" cargo build --no-default-features --release --no-default-features --features cpu

# Cross-validation with C++ reference
export BITNET_GGUF="models/bitnet/model.gguf" BITNET_DETERMINISTIC=1 BITNET_SEED=42
cargo run -p xtask -- full-crossval

# GPU/CPU parity testing
BITNET_STRICT_TOKENIZERS=1 cargo test --no-default-features --features gpu
BITNET_STRICT_NO_FAKE_GPU=1 cargo test --no-default-features --features cpu
```

## Implementation Roadmap

### Phase 1: Core Discovery Infrastructure (Week 1)
- [ ] Implement `TokenizerDiscovery` with GGUF metadata parsing
- [ ] Add vocabulary size extraction and model type detection
- [ ] Create co-location and cache directory search functionality
- [ ] Implement basic tokenizer compatibility matrix

### Phase 2: Smart Downloading and Strategy Resolution (Week 1-2)
- [ ] Develop `SmartTokenizerDownload` with HuggingFace Hub integration
- [ ] Implement `TokenizerStrategyResolver` with neural network model-specific wrappers
- [ ] Add comprehensive fallback chain with error reporting
- [ ] Create caching and resume capabilities for large downloads

### Phase 3: Integration and Validation (Week 2)
- [ ] Integrate tokenizer discovery with `cargo xtask infer` command
- [ ] Add cross-validation tests against existing UniversalTokenizer
- [ ] Implement end-to-end integration tests with real models
- [ ] Create comprehensive documentation and examples

### Phase 4: Performance Optimization and Production Readiness (Week 2-3)
- [ ] Optimize for large vocabulary models (128K+ tokens)
- [ ] Add GPU acceleration and device-aware tokenization
- [ ] Implement deterministic behavior and strict mode support
- [ ] Complete risk mitigation and production-grade error handling

## Conclusion

This specification provides a comprehensive technical approach for implementing automatic tokenizer discovery and integration in BitNet.rs. The proposed solution addresses all 10 acceptance criteria while maintaining compatibility with the existing neural network inference pipeline, quantization formats (I2S/TL1/TL2), and cross-validation framework.

**Key Neural Network Innovations**:
- **Device-Aware Tokenization**: Automatic GPU/CPU selection for optimal performance
- **Large Vocabulary Optimization**: Efficient handling of 128K+ token neural network models
- **Quantization Integration**: Seamless compatibility with BitNet.rs quantization pipeline
- **Production-Grade Reliability**: Robust error handling and comprehensive fallback strategies

**Next Steps**: **FINALIZE → spec-finalizer** - The technical specification is complete and ready for implementation guidance to development teams.

---

*This specification aligns with BitNet.rs neural network inference architecture, maintains TDD practices, supports feature-gated builds (--no-default-features --features cpu|gpu), and provides comprehensive validation approaches for production-grade neural network model inference.*
