# Issue #336: Universal Tokenizer Discovery System for Production Neural Network Inference

## Context

BitNet.rs requires a production-ready universal tokenizer discovery system to enable automatic tokenizer resolution for quantized neural network models. The current implementation in `bitnet-tokenizers` contains multiple stub implementations that prevent proper tokenizer extraction from GGUF metadata, model architecture detection, and intelligent fallback strategies.

This specification defines the comprehensive architecture for implementing 5 core components that enable seamless tokenizer discovery, smart downloading, and production-scale neural network inference across BitNet, LLaMA, GPT-2, GPT-Neo, BERT, and T5 architectures.

**Pipeline Integration**: Model Loading → **Tokenizer Discovery** → Quantization → Inference → Output

## User Stories

### Story 1: Automatic Tokenizer Discovery from GGUF Models
**As a** BitNet.rs user
**I want** automatic tokenizer discovery from GGUF model files
**So that** I can run neural network inference without manual tokenizer configuration

**Business Value**: Eliminates manual tokenizer setup, reduces configuration errors, enables production-ready inference workflows

### Story 2: Smart Tokenizer Downloading
**As a** BitNet.rs developer
**I want** automatic tokenizer downloading from HuggingFace Hub
**So that** missing tokenizers are automatically acquired with proper caching

**Business Value**: Seamless model deployment, reduced manual intervention, improved developer experience

### Story 3: Model Architecture Auto-Detection
**As a** neural network researcher
**I want** automatic model architecture detection from tensor patterns
**So that** the system selects optimal tokenizers for different model types (LLaMA, GPT-2, BitNet, etc.)

**Business Value**: Supports diverse model architectures, enables flexible experimentation, maintains compatibility

### Story 4: Production-Ready Fallback Strategies
**As a** production ML engineer
**I want** intelligent fallback chains for tokenizer resolution
**So that** the system gracefully handles missing tokenizers with actionable error messages

**Business Value**: Robust production deployments, predictable behavior, clear error diagnostics

### Story 5: Cross-Validation with C++ Reference
**As a** BitNet.rs maintainer
**I want** tokenizer discovery validated against Microsoft BitNet C++ reference
**So that** we maintain >99% compatibility with the official implementation

**Business Value**: Ensures correctness, maintains ecosystem compatibility, builds user trust

## Acceptance Criteria

### AC1: Embedded Tokenizer Support
**Given** a GGUF model file with embedded HuggingFace or SentencePiece tokenizer data
**When** `TokenizerDiscovery::try_extract_embedded_tokenizer()` is called
**Then** the system correctly extracts and validates the embedded tokenizer
**And** fallback to `BasicTokenizer` occurs only when embedded data is invalid or missing

**Test Tag**: `// AC1:embedded_extraction`

**Validation**:
- HuggingFace JSON tokenizers parse correctly from GGUF string metadata
- SentencePiece models load correctly from GGUF binary blobs
- BOS/EOS/PAD token IDs are correctly extracted from metadata
- Vocabulary size validation ensures reasonable bounds (1K-2M tokens)
- Invalid embedded data triggers appropriate error messages

### AC2: Model Architecture Detection
**Given** a GGUF model file with tensor patterns
**When** `TokenizerDiscovery::extract_model_type()` analyzes tensor names
**Then** the system correctly identifies model architecture (BitNet, LLaMA, GPT-2, GPT-Neo, BERT, T5)
**And** unknown architectures gracefully fallback to "transformer" generic type

**Test Tag**: `// AC2:architecture_detection`

**Validation**:
- BitNet models detected via `bitnet` or `bitlinear` tensor patterns
- LLaMA models detected via `attn_q/k/v` patterns
- GPT-Neo/GPT-J detected via `transformer.h.` and `mlp.c_fc` patterns
- BERT models detected via `bert.encoder.layer` patterns
- T5 models detected via `encoder.block` or `decoder.block` patterns
- Confidence scoring helps disambiguate similar architectures

### AC3: Vocabulary Size Resolution
**Given** a GGUF model file with complete or incomplete metadata
**When** `TokenizerDiscovery::extract_vocab_size()` extracts vocabulary information
**Then** the system determines vocabulary size via metadata, tensor dimensions, or architecture defaults
**And** clear error messages guide resolution when size cannot be determined

**Test Tag**: `// AC3:vocab_resolution`

**Validation**:
- Primary extraction from `tokenizer.ggml.vocab_size` metadata
- Alternative keys checked (`llama.vocab_size`, `gpt2.vocab_size`, etc.)
- Embedding tensor dimensions used when metadata missing
- Architecture-specific defaults: LLaMA-2 (32000), LLaMA-3 (128256), GPT-2 (50257), BERT (30522), T5 (32128), BitNet (50257)
- Vocabulary size sanity checking (1000 < size < 2,000,000)

### AC4: Smart Download Integration
**Given** a model requiring a tokenizer not available locally
**When** `TokenizerStrategyResolver::resolve_with_fallback()` attempts smart download
**Then** the system downloads compatible tokenizers from HuggingFace Hub with caching
**And** network failures are handled gracefully with retry logic and offline mode support

**Test Tag**: `// AC4:smart_download`

**Validation**:
- HuggingFace Hub integration downloads tokenizer.json files
- Download progress reporting and error handling
- Local caching prevents redundant downloads
- Network timeout and retry logic
- Offline mode (`BITNET_OFFLINE=1`) skips download attempts
- Cache validation ensures downloaded files are complete

### AC5: Production Readiness
**Given** the complete tokenizer discovery system
**When** cross-validated with Microsoft BitNet C++ reference implementation
**Then** the system achieves >99% compatibility with existing GGUF models
**And** performance is comparable to or better than existing implementations
**And** comprehensive error messages guide troubleshooting

**Test Tag**: `// AC5:production_validation`

**Validation**:
- Cross-validation with `cargo run -p xtask -- crossval` passes
- Performance benchmarks establish baselines for discovery latency (<100ms for metadata, <200ms for large vocabularies)
- All error paths tested with actionable error messages
- Integration tests cover LLaMA-2, LLaMA-3, GPT-2, BitNet model types
- Strict mode (`BITNET_STRICT_TOKENIZERS=1`) prevents mock fallbacks

## Technical Requirements

### Scope

**Affected Workspace Crates**:
- `bitnet-tokenizers` (primary): Discovery, strategy, and download implementations
- `bitnet-models`: GGUF metadata parsing integration
- `bitnet-inference`: Tokenizer usage in generation pipeline
- `bitnet-cli`: User-facing inference commands
- `xtask`: Developer tooling and cross-validation

**Pipeline Stages**:
1. **Model Loading**: GGUF file parsing, metadata extraction
2. **Tokenizer Discovery**: Embedded extraction, co-located file search, cache lookup
3. **Smart Download**: HuggingFace Hub integration (when needed)
4. **Tokenizer Loading**: Instantiation with model-specific wrappers
5. **Inference Integration**: Seamless text → tokens → inference → output

### Constraints

**Performance Targets**:
- Tokenizer discovery latency: <100ms for metadata extraction, <200ms for large vocabularies (128K+ tokens)
- Memory efficiency: Zero-copy GGUF metadata parsing via memory mapping
- GPU acceleration: Automatic for large vocabularies (>65K tokens)
- Download caching: Persistent cache with content validation

**Quantization Accuracy**:
- Tokenizer output must maintain >99.8% correlation with reference implementations
- Support for I2_S, TL1, TL2 quantization types with device-aware selection
- Cross-validation against Microsoft BitNet C++ reference implementation

**GPU/CPU Compatibility**:
- Device-aware tokenization for large vocabularies (LLaMA-3: 128K tokens)
- Automatic CPU fallback when GPU unavailable
- Feature flag discipline: `--no-default-features --features cpu|gpu`
- SIMD optimization for CPU tokenization (AVX2/AVX-512/NEON)

**Production Reliability**:
- Strict mode enforcement via `BITNET_STRICT_TOKENIZERS=1`
- Deterministic behavior with `BITNET_DETERMINISTIC=1 BITNET_SEED=42`
- Comprehensive error handling with actionable messages
- Graceful degradation when tokenizers unavailable

### Public Contracts

**Core Discovery API**:
```rust
/// Primary tokenizer discovery engine for BitNet.rs neural network models
pub struct TokenizerDiscovery {
    // Memory-mapped GGUF data with 'static lifetime management
    _mmap: memmap2::Mmap,
    gguf_reader: GgufReader<'static>,
    model_path: PathBuf,
    vocab_size: usize,
    model_type: String,
    compatibility_matrix: ModelCompatibilityMatrix,
}

impl TokenizerDiscovery {
    /// Create discovery engine from GGUF model file
    /// Tests: // AC1:embedded_extraction
    pub fn from_gguf(path: &Path) -> Result<Self>;

    /// Discover optimal tokenizer strategy for the loaded model
    /// Returns: Exact, Discovered, NeedsDownload, EmbeddedGguf, or Mock (non-strict only)
    /// Tests: // AC2:architecture_detection, AC3:vocab_resolution
    pub fn discover_tokenizer_strategy(&self) -> Result<TokenizerStrategy>;

    /// Extract embedded tokenizer from GGUF metadata
    /// Tests: // AC1:embedded_extraction
    pub fn try_extract_embedded_tokenizer(&self) -> Result<Option<Arc<dyn Tokenizer>>>;

    /// Check for co-located tokenizer files in model directory
    /// Tests: // AC4:smart_download
    pub fn check_colocated_tokenizers(&self) -> Result<Option<PathBuf>>;

    /// Check standard cache directories for compatible tokenizers
    /// Tests: // AC4:smart_download
    pub fn check_cache_locations(&self) -> Result<Option<PathBuf>>;

    /// Infer download source based on neural network model patterns
    /// Tests: // AC4:smart_download
    pub fn infer_download_source(&self) -> Result<Option<TokenizerDownloadInfo>>;

    /// Get vocabulary size from model metadata
    pub fn vocab_size(&self) -> usize;

    /// Get model architecture type (e.g., "llama", "gpt2", "bitnet")
    pub fn model_type(&self) -> &str;

    /// Check if model requires large vocabulary optimization (>64K tokens)
    pub fn requires_large_vocab_optimization(&self) -> bool;
}
```

**Strategy Resolution API**:
```rust
/// Unified tokenizer strategy resolution with neural network model integration
pub struct TokenizerStrategyResolver {
    discovery: TokenizerDiscovery,
    downloader: SmartTokenizerDownload,
    fallback_chain: TokenizerFallbackChain,
}

impl TokenizerStrategyResolver {
    /// Create resolver with discovery engine and downloader
    /// Tests: // AC4:smart_download
    pub async fn new(discovery: TokenizerDiscovery) -> Result<Self>;

    /// Resolve tokenizer strategy to concrete tokenizer implementation
    /// Tests: // AC5:production_validation
    pub async fn resolve_tokenizer(&self, strategy: TokenizerStrategy) -> Result<Arc<dyn Tokenizer>>;

    /// Resolve with automatic fallback chain
    /// Strategy order: GGUF embedded → Co-located files → Cache → Smart download → Mock (non-strict)
    /// Tests: // AC5:production_validation
    pub async fn resolve_with_fallback(&self) -> Result<Arc<dyn Tokenizer>>;
}
```

**Tokenizer Strategy Enum**:
```rust
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
    pub fn requires_network(&self) -> bool;
    /// Check if strategy uses cached resources
    pub fn uses_cache(&self) -> bool;
    /// Get description for logging and error messages
    pub fn description(&self) -> &'static str;
}
```

**Download Integration API**:
```rust
/// Smart tokenizer downloading from HuggingFace Hub
pub struct SmartTokenizerDownload {
    cache_manager: CacheManager,
    http_client: reqwest::Client,
}

impl SmartTokenizerDownload {
    /// Create downloader with cache management
    pub fn new() -> Result<Self>;

    /// Download tokenizer from HuggingFace Hub
    /// Tests: // AC4:smart_download
    pub async fn download_tokenizer(&self, info: &TokenizerDownloadInfo) -> Result<PathBuf>;

    /// Check if tokenizer is already cached
    pub fn is_cached(&self, info: &TokenizerDownloadInfo) -> bool;

    /// Validate cached tokenizer integrity
    pub fn validate_cache(&self, path: &Path) -> Result<()>;
}

/// Download metadata for tokenizer acquisition
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
```

**Model-Specific Wrappers**:
```rust
/// LLaMA model-specific tokenizer wrapper with neural network optimizations
pub struct LlamaTokenizerWrapper {
    inner: Arc<dyn Tokenizer>,
    vocab_size: usize,
    model_variant: LlamaVariant,
}

/// GPT-2 model-specific tokenizer wrapper
pub struct Gpt2TokenizerWrapper {
    inner: Arc<dyn Tokenizer>,
}

/// BitNet model-specific tokenizer wrapper with quantization awareness
pub struct BitNetTokenizerWrapper {
    inner: Arc<dyn Tokenizer>,
    quantization_type: QuantizationType,
}

/// LLaMA model variant enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaVariant {
    Llama2,      // 32K vocabulary, legacy special tokens
    Llama3,      // 128K vocabulary, enhanced special tokens
    CodeLlama,   // 32016 vocabulary, code-optimized
}

impl LlamaVariant {
    /// Get expected vocabulary size for variant
    pub fn expected_vocab_size(&self) -> usize;
    /// Check if variant requires GPU acceleration for large vocabulary
    pub fn requires_gpu_acceleration(&self) -> bool;
}
```

### Risks

**Performance Impact**:
- **Risk**: Large vocabulary models (LLaMA-3: 128K tokens) may cause memory pressure during discovery
- **Mitigation**: Memory-mapped GGUF files with zero-copy metadata parsing, lazy tokenizer instantiation
- **Validation**: Performance benchmarks with `cargo bench --no-default-features --features cpu` establish baselines

**Quantization Accuracy**:
- **Risk**: Tokenizer output may not align correctly with quantized tensors
- **Mitigation**: Cross-validation against C++ reference implementation, >99% compatibility requirement
- **Validation**: Integration tests with `cargo run -p xtask -- crossval` verify accuracy

**Network Dependency**:
- **Risk**: Smart download failures may block inference when tokenizers unavailable
- **Mitigation**: Offline mode (`BITNET_OFFLINE=1`), local cache fallback, clear error messages
- **Validation**: Integration tests simulate network failures and validate fallback behavior

**Model Compatibility**:
- **Risk**: New model architectures may not be detected correctly
- **Mitigation**: Generic "transformer" fallback, extensible architecture detection patterns
- **Validation**: Test suite covers LLaMA, GPT-2, GPT-Neo, BERT, T5, BitNet architectures

**GGUF Format Changes**:
- **Risk**: GGUF specification updates may break metadata parsing
- **Mitigation**: Defensive parsing with multiple metadata key variants, graceful degradation
- **Validation**: Compatibility tests with various GGUF versions from HuggingFace Hub

## Design Components

### Component 1: Embedded Tokenizer Extraction

**Location**: `crates/bitnet-tokenizers/src/discovery.rs:462-522`

**Functionality**: Extract HuggingFace JSON or SentencePiece tokenizers embedded in GGUF metadata

**Implementation**:
```rust
pub fn try_extract_embedded_tokenizer(&self) -> Result<Option<Arc<dyn Tokenizer>>> {
    debug!("Attempting to extract embedded tokenizer from GGUF metadata");

    // Check for HuggingFace tokenizer.json embedded as string
    if let Some(tokenizer_json) = self.gguf_reader.get_string_metadata("tokenizer.json") {
        debug!("Found embedded tokenizer.json ({} chars)", tokenizer_json.len());
        let hf_tokenizer = crate::hf_tokenizer::HfTokenizer::from_json_string(&tokenizer_json)?;
        return Ok(Some(Arc::new(hf_tokenizer)));
    }

    // Check for SentencePiece model embedded as bytes
    if let Some(model_bytes) = self.gguf_reader.get_array_metadata("tokenizer.ggml.model") {
        debug!("Found embedded tokenizer.ggml.model ({} bytes)", model_bytes.len());

        let bos = self.gguf_reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
        let eos = self.gguf_reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
        let pad = self.gguf_reader.get_u32_metadata("tokenizer.ggml.pad_token_id");

        let spm_tokenizer = crate::sp_tokenizer::SpTokenizer::from_gguf_blob(
            &model_bytes,
            bos,
            eos,
            pad
        )?;
        return Ok(Some(Arc::new(spm_tokenizer)));
    }

    // Check for embedded vocabulary tokens
    if let Some(vocab) = self.gguf_reader.get_string_array_metadata("tokenizer.ggml.tokens")
        && vocab.len() == self.vocab_size
    {
        debug!("Found embedded vocabulary with {} tokens", vocab.len());
        let basic_tokenizer = crate::BasicTokenizer::with_config(
            self.vocab_size,
            self.gguf_reader.get_u32_metadata("tokenizer.ggml.bos_token_id"),
            self.gguf_reader.get_u32_metadata("tokenizer.ggml.eos_token_id"),
            self.gguf_reader.get_u32_metadata("tokenizer.ggml.pad_token_id"),
        );
        return Ok(Some(Arc::new(basic_tokenizer)));
    }

    debug!("No embedded tokenizer found in GGUF metadata");
    Ok(None)
}
```

**Device-Aware Considerations**:
- No GPU acceleration needed for metadata parsing
- Memory-mapped GGUF files prevent large memory allocations
- Lazy tokenizer instantiation defers heavy initialization

### Component 2: Model Type Detection

**Location**: `crates/bitnet-tokenizers/src/discovery.rs:285-320`

**Functionality**: Analyze tensor patterns to identify model architecture (BitNet, LLaMA, GPT-2, GPT-Neo, BERT, T5)

**Implementation**:
```rust
fn extract_model_type(reader: &GgufReader) -> Result<String> {
    // Try metadata first
    if let Some(arch) = reader.get_string_metadata("general.architecture") {
        return Ok(arch);
    }

    // Try alternative metadata keys
    for key in &["model.architecture", "transformer.architecture"] {
        if let Some(arch) = reader.get_string_metadata(key) {
            return Ok(arch);
        }
    }

    // Infer from model name
    if let Some(name) = reader.get_string_metadata("general.name") {
        let name_lower = name.to_lowercase();
        if name_lower.contains("bitnet") {
            return Ok("bitnet".to_string());
        } else if name_lower.contains("llama") {
            return Ok("llama".to_string());
        } else if name_lower.contains("gpt") {
            return Ok("gpt2".to_string());
        }
    }

    // Comprehensive tensor pattern analysis
    let tensor_names = reader.tensor_names();

    // BitNet architecture patterns
    if tensor_names.iter().any(|name| name.contains("bitnet") || name.contains("bitlinear")) {
        return Ok("bitnet".to_string());
    }

    // LLaMA patterns
    if tensor_names.iter().any(|name|
        name.contains("attn_q") || name.contains("attn_k") || name.contains("attn_v")
    ) {
        return Ok("llama".to_string());
    }

    // GPT-Neo/GPT-J patterns
    if tensor_names.iter().any(|name|
        name.contains("transformer.h.") && name.contains("mlp.c_fc")
    ) {
        return Ok("gpt-neo".to_string());
    }

    // BERT patterns
    if tensor_names.iter().any(|name| name.contains("bert.encoder.layer")) {
        return Ok("bert".to_string());
    }

    // T5 patterns
    if tensor_names.iter().any(|name|
        name.contains("encoder.block") || name.contains("decoder.block")
    ) {
        return Ok("t5".to_string());
    }

    // Generic transformer fallback
    Ok("transformer".to_string())
}
```

**Confidence Scoring** (future enhancement):
```rust
struct ArchitectureMatch {
    architecture: String,
    confidence: f32,  // 0.0-1.0
    matched_patterns: Vec<String>,
}

fn detect_with_confidence(reader: &GgufReader) -> Vec<ArchitectureMatch> {
    // Return ranked list for disambiguation
}
```

### Component 3: Vocabulary Size Fallback

**Location**: `crates/bitnet-tokenizers/src/discovery.rs:322-365`

**Functionality**: Extract vocabulary size from metadata, tensor dimensions, or architecture defaults

**Implementation**:
```rust
fn extract_vocab_size(reader: &GgufReader) -> Result<usize> {
    // Primary: Try standard metadata key
    if let Some(vocab_size) = reader.get_u32_metadata("tokenizer.ggml.vocab_size") {
        return Ok(vocab_size as usize);
    }

    // Secondary: Try alternative metadata keys
    for key in &["llama.vocab_size", "gpt2.vocab_size", "transformer.vocab_size", "model.vocab_size"] {
        if let Some(vocab_size) = reader.get_u32_metadata(key) {
            return Ok(vocab_size as usize);
        }
    }

    // Tertiary: Infer from embedding tensor dimensions
    let tensor_names = reader.tensor_names();
    for name in tensor_names {
        if (name.contains("token_embd") || name.contains("wte") || name.contains("embed"))
            && let Some(info) = reader.get_tensor_info_by_name(name)
        {
            let shape = &info.shape;
            if shape.len() >= 2 {
                let possible_vocab = std::cmp::max(shape[0], shape[1]);
                // Sanity check - vocab size should be reasonable
                if possible_vocab > 1000 && possible_vocab < 2_000_000 {
                    debug!("Inferred vocab_size {} from tensor: {}", possible_vocab, name);
                    return Ok(possible_vocab);
                }
            }
        }
    }

    // Quaternary: Architecture-specific defaults
    if let Some(model_type) = reader.get_string_metadata("general.architecture") {
        let default_vocab = match model_type.as_str() {
            "llama" => 32000,      // LLaMA-2 default
            "llama3" => 128256,    // LLaMA-3 default
            "gpt2" => 50257,       // GPT-2 default
            "bitnet" => 50257,     // BitNet typically uses GPT-2 vocab
            "bert" => 30522,       // BERT default
            "t5" => 32128,         // T5 default
            "gpt-neo" => 50257,    // GPT-Neo default
            _ => {
                return Err(TokenizerErrorHandler::config_error(
                    format!("Unknown architecture '{}' with no explicit vocab size", model_type)
                ));
            }
        };

        warn!("Using default vocab_size {} for architecture '{}'", default_vocab, model_type);
        return Ok(default_vocab);
    }

    Err(TokenizerErrorHandler::config_error(
        "Could not extract vocabulary size from GGUF metadata, tensors, or architecture defaults"
    ))
}
```

### Component 4: Smart Download Strategy

**Location**: `crates/bitnet-tokenizers/src/strategy.rs:93-211`

**Functionality**: Download compatible tokenizers from HuggingFace Hub with caching and retry logic

**Implementation**:
```rust
pub async fn download_tokenizer(&self, info: &TokenizerDownloadInfo) -> Result<PathBuf> {
    info!("Downloading tokenizer from HuggingFace Hub: {}", info.repo);

    // Check if already cached
    let cache_path = self.cache_manager.model_cache_dir(&info.cache_key, None)?;
    let target_file = cache_path.join("tokenizer.json");

    if target_file.exists() {
        match self.validate_cache(&target_file) {
            Ok(()) => {
                info!("Using cached tokenizer at: {}", target_file.display());
                return Ok(target_file);
            }
            Err(e) => {
                warn!("Cached tokenizer invalid: {}, re-downloading", e);
                std::fs::remove_file(&target_file).ok();
            }
        }
    }

    // Create cache directory
    std::fs::create_dir_all(&cache_path)?;

    // Download from HuggingFace Hub
    let url = format!(
        "https://huggingface.co/{}/resolve/main/tokenizer.json",
        info.repo
    );

    let mut response = self.http_client
        .get(&url)
        .timeout(Duration::from_secs(300))  // 5 minute timeout
        .send()
        .await
        .map_err(|e| TokenizerErrorHandler::download_error(&url, e))?;

    if !response.status().is_success() {
        return Err(TokenizerErrorHandler::config_error(
            format!("Download failed with status: {}", response.status())
        ));
    }

    // Stream download to file
    let mut file = std::fs::File::create(&target_file)?;
    while let Some(chunk) = response.chunk().await? {
        std::io::Write::write_all(&mut file, &chunk)?;
    }

    // Validate downloaded file
    self.validate_cache(&target_file)?;

    // Validate expected vocabulary size if specified
    if let Some(expected_vocab) = info.expected_vocab {
        let tokenizer = crate::from_path(&target_file)?.0;
        let actual_vocab = tokenizer.vocab_size();
        if actual_vocab != expected_vocab {
            warn!(
                "Vocabulary size mismatch: expected {}, got {}",
                expected_vocab, actual_vocab
            );
        }
    }

    info!("Successfully downloaded tokenizer to: {}", target_file.display());
    Ok(target_file)
}

fn validate_cache(&self, path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(TokenizerErrorHandler::file_not_found(path));
    }

    let metadata = std::fs::metadata(path)?;
    if metadata.len() == 0 {
        return Err(TokenizerErrorHandler::config_error(
            format!("Cached tokenizer file is empty: {}", path.display())
        ));
    }

    // Validate JSON structure
    let content = std::fs::read_to_string(path)?;
    serde_json::from_str::<serde_json::Value>(&content)
        .map_err(|e| TokenizerErrorHandler::config_error(
            format!("Invalid JSON in cached tokenizer: {}", e)
        ))?;

    Ok(())
}
```

**Network Configuration**:
- Retry logic: 3 attempts with exponential backoff (1s, 2s, 4s)
- Timeout: 5 minutes for download completion
- Offline mode: Skip download when `BITNET_OFFLINE=1` set
- Progress reporting: Log download progress for large files

### Component 5: Fallback Chain Integration

**Location**: `crates/bitnet-tokenizers/src/strategy.rs:18-35`

**Functionality**: Systematic fallback through discovery strategies with proper error aggregation

**Implementation**:
```rust
pub async fn resolve_with_fallback(&self) -> Result<Arc<dyn Tokenizer>> {
    let mut errors = Vec::new();

    // Strategy 1: GGUF embedded tokenizer
    match self.discovery.try_extract_embedded_tokenizer() {
        Ok(Some(embedded_tokenizer)) => {
            info!("Successfully resolved tokenizer from GGUF metadata");
            return self.configure_model_specific_wrapper(embedded_tokenizer);
        }
        Ok(None) => debug!("No embedded tokenizer found in GGUF"),
        Err(e) => {
            warn!("Failed to extract embedded tokenizer: {}", e);
            errors.push(("GGUF embedded", e));
        }
    }

    // Strategy 2: Co-located files
    match self.discovery.check_colocated_tokenizers() {
        Ok(Some(path)) => {
            info!("Found co-located tokenizer at: {}", path.display());
            match self.load_tokenizer_from_path(&path) {
                Ok(tokenizer) => return Ok(tokenizer),
                Err(e) => {
                    warn!("Failed to load co-located tokenizer: {}", e);
                    errors.push(("co-located files", e));
                }
            }
        }
        Ok(None) => debug!("No co-located tokenizer files found"),
        Err(e) => {
            warn!("Error checking co-located tokenizers: {}", e);
            errors.push(("co-located search", e));
        }
    }

    // Strategy 3: Standard cache directories
    match self.discovery.check_cache_locations() {
        Ok(Some(path)) => {
            info!("Found cached tokenizer at: {}", path.display());
            match self.load_tokenizer_from_path(&path) {
                Ok(tokenizer) => return Ok(tokenizer),
                Err(e) => {
                    warn!("Failed to load cached tokenizer: {}", e);
                    errors.push(("cache directories", e));
                }
            }
        }
        Ok(None) => debug!("No cached tokenizer found"),
        Err(e) => {
            warn!("Error checking cache locations: {}", e);
            errors.push(("cache search", e));
        }
    }

    // Strategy 4: Smart download (if not in offline mode)
    if std::env::var("BITNET_OFFLINE").is_err() {
        match self.discovery.infer_download_source() {
            Ok(Some(download_info)) => {
                info!("Attempting smart download from: {}", download_info.repo);
                match self.downloader.download_tokenizer(&download_info).await {
                    Ok(downloaded_path) => {
                        match self.load_tokenizer_from_path(&downloaded_path) {
                            Ok(tokenizer) => return Ok(tokenizer),
                            Err(e) => {
                                warn!("Failed to load downloaded tokenizer: {}", e);
                                errors.push(("smart download loading", e));
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Smart download failed: {}", e);
                        errors.push(("smart download", e));
                    }
                }
            }
            Ok(None) => debug!("No download source available for this model"),
            Err(e) => {
                warn!("Error determining download source: {}", e);
                errors.push(("download source detection", e));
            }
        }
    } else {
        debug!("Skipping smart download (offline mode)");
    }

    // Strategy 5: Mock fallback (non-strict mode only)
    if std::env::var("BITNET_STRICT_TOKENIZERS").is_err() {
        info!("Falling back to mock tokenizer");
        let mock_tokenizer = Arc::new(crate::MockTokenizer::new());
        return self.configure_model_specific_wrapper(mock_tokenizer);
    }

    // All strategies failed - provide comprehensive error summary
    let error_summary = format!(
        "All tokenizer resolution strategies failed. Tried {} strategies.\n\
         Errors encountered:\n{}",
        errors.len(),
        errors.iter()
            .map(|(strategy, err)| format!("  - {}: {}", strategy, err))
            .collect::<Vec<_>>()
            .join("\n")
    );

    Err(TokenizerErrorHandler::config_error(error_summary))
}
```

## Integration Points

### GGUF Parsing Integration

**Dependency**: `bitnet-models::GgufReader`

**Integration**:
```rust
// TokenizerDiscovery relies on GgufReader for metadata extraction
impl TokenizerDiscovery {
    pub fn from_gguf(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Create GGUF reader with static lifetime via transmute
        // (safe because we keep mmap alive)
        let reader = unsafe {
            let data_slice: &'static [u8] = std::mem::transmute(mmap.as_ref());
            GgufReader::new(data_slice)?
        };

        // Extract metadata required for discovery
        let vocab_size = Self::extract_vocab_size(&reader)?;
        let model_type = Self::extract_model_type(&reader)?;

        Ok(Self {
            _mmap: mmap,
            gguf_reader: reader,
            model_path: path.to_path_buf(),
            vocab_size,
            model_type,
            compatibility_matrix: ModelCompatibilityMatrix::default(),
        })
    }
}
```

### Inference Pipeline Integration

**Usage in Inference Engine**:
```rust
// bitnet-inference integration
use bitnet_tokenizers::{TokenizerDiscovery, TokenizerStrategyResolver};

pub async fn create_inference_engine(model_path: &Path) -> Result<InferenceEngine> {
    // Automatic tokenizer discovery
    let discovery = TokenizerDiscovery::from_gguf(model_path)?;
    let resolver = TokenizerStrategyResolver::new(discovery).await?;
    let tokenizer = resolver.resolve_with_fallback().await?;

    // Load model and create engine
    let model = GgufModel::load(model_path)?;
    Ok(InferenceEngine::new(model, tokenizer))
}
```

### CLI Integration

**Usage in `bitnet-cli`**:
```rust
// cargo run -p bitnet-cli -- infer --model model.gguf --prompt "Test"
pub async fn infer_command(model_path: &Path, prompt: &str) -> Result<String> {
    // Automatic tokenizer discovery (no manual specification needed)
    let discovery = TokenizerDiscovery::from_gguf(model_path)?;
    let resolver = TokenizerStrategyResolver::new(discovery).await?;
    let tokenizer = resolver.resolve_with_fallback().await?;

    // Tokenize input
    let tokens = tokenizer.encode(prompt, true, false)?;

    // Run inference
    let engine = InferenceEngine::new_from_gguf(model_path)?;
    let output_tokens = engine.generate(&tokens, 50)?;

    // Decode output
    let output_text = tokenizer.decode(&output_tokens)?;
    Ok(output_text)
}
```

### xtask Integration

**Cross-Validation Command**:
```bash
# Automatic tokenizer discovery in cross-validation
export BITNET_GGUF="model.gguf"
cargo run -p xtask -- crossval

# Behind the scenes:
# 1. TokenizerDiscovery::from_gguf() parses model metadata
# 2. Resolver attempts embedded → co-located → cache → download
# 3. Validates tokenizer output against C++ reference
# 4. Reports >99% compatibility or failures
```

## Testing Strategy

### Unit Tests

**Embedded Extraction Tests**:
```rust
// AC1:embedded_extraction
#[test]
fn test_extract_hf_embedded_tokenizer() {
    let gguf_with_hf = create_mock_gguf_with_hf_json();
    let discovery = TokenizerDiscovery::from_gguf(&gguf_with_hf).unwrap();
    let tokenizer = discovery.try_extract_embedded_tokenizer().unwrap();

    assert!(tokenizer.is_some());
    assert_eq!(tokenizer.unwrap().vocab_size(), 128256);
}

// AC1:embedded_extraction
#[test]
fn test_extract_spm_embedded_tokenizer() {
    let gguf_with_spm = create_mock_gguf_with_spm_blob();
    let discovery = TokenizerDiscovery::from_gguf(&gguf_with_spm).unwrap();
    let tokenizer = discovery.try_extract_embedded_tokenizer().unwrap();

    assert!(tokenizer.is_some());
    assert_eq!(tokenizer.unwrap().bos_token_id(), Some(1));
}
```

**Architecture Detection Tests**:
```rust
// AC2:architecture_detection
#[test]
fn test_detect_bitnet_architecture() {
    let gguf_bitnet = create_mock_gguf_with_bitnet_tensors();
    let discovery = TokenizerDiscovery::from_gguf(&gguf_bitnet).unwrap();
    assert_eq!(discovery.model_type(), "bitnet");
}

// AC2:architecture_detection
#[test]
fn test_detect_llama_architecture() {
    let gguf_llama = create_mock_gguf_with_llama_tensors();
    let discovery = TokenizerDiscovery::from_gguf(&gguf_llama).unwrap();
    assert_eq!(discovery.model_type(), "llama");
}

// AC2:architecture_detection
#[test]
fn test_detect_gpt2_architecture() {
    let gguf_gpt2 = create_mock_gguf_with_gpt2_metadata();
    let discovery = TokenizerDiscovery::from_gguf(&gguf_gpt2).unwrap();
    assert_eq!(discovery.model_type(), "gpt2");
}
```

**Vocabulary Size Tests**:
```rust
// AC3:vocab_resolution
#[test]
fn test_extract_vocab_from_metadata() {
    let gguf = create_mock_gguf_with_vocab_metadata(128256);
    let discovery = TokenizerDiscovery::from_gguf(&gguf).unwrap();
    assert_eq!(discovery.vocab_size(), 128256);
}

// AC3:vocab_resolution
#[test]
fn test_infer_vocab_from_embedding_tensor() {
    let gguf = create_mock_gguf_with_embedding_tensor(50257, 768);
    let discovery = TokenizerDiscovery::from_gguf(&gguf).unwrap();
    assert_eq!(discovery.vocab_size(), 50257);
}

// AC3:vocab_resolution
#[test]
fn test_fallback_to_architecture_default() {
    let gguf = create_mock_gguf_with_architecture("llama");
    let discovery = TokenizerDiscovery::from_gguf(&gguf).unwrap();
    assert_eq!(discovery.vocab_size(), 32000);  // LLaMA-2 default
}
```

### Integration Tests

**Download Integration Tests**:
```rust
// AC4:smart_download
#[tokio::test]
async fn test_smart_download_llama_tokenizer() {
    let downloader = SmartTokenizerDownload::new().unwrap();
    let download_info = TokenizerDownloadInfo {
        repo: "meta-llama/Llama-2-7b-hf".to_string(),
        files: vec!["tokenizer.json".to_string()],
        cache_key: "llama2-32k".to_string(),
        expected_vocab: Some(32000),
    };

    let downloaded_path = downloader.download_tokenizer(&download_info).await.unwrap();
    assert!(downloaded_path.exists());

    let tokenizer = crate::from_path(&downloaded_path).unwrap().0;
    assert_eq!(tokenizer.vocab_size(), 32000);
}

// AC4:smart_download
#[tokio::test]
async fn test_download_with_cache_hit() {
    let downloader = SmartTokenizerDownload::new().unwrap();
    let download_info = create_test_download_info();

    // First download
    let path1 = downloader.download_tokenizer(&download_info).await.unwrap();

    // Second download should use cache
    let path2 = downloader.download_tokenizer(&download_info).await.unwrap();
    assert_eq!(path1, path2);
}

// AC4:smart_download
#[tokio::test]
async fn test_download_network_failure_handling() {
    let downloader = SmartTokenizerDownload::new().unwrap();
    let invalid_info = TokenizerDownloadInfo {
        repo: "invalid/nonexistent-model".to_string(),
        files: vec!["tokenizer.json".to_string()],
        cache_key: "invalid".to_string(),
        expected_vocab: None,
    };

    let result = downloader.download_tokenizer(&invalid_info).await;
    assert!(result.is_err());
}
```

**End-to-End Tests**:
```rust
// AC5:production_validation
#[tokio::test]
async fn test_e2e_llama2_inference() {
    let model_path = download_test_model("llama2-7b-chat.gguf");

    // Automatic tokenizer discovery
    let discovery = TokenizerDiscovery::from_gguf(&model_path).unwrap();
    let resolver = TokenizerStrategyResolver::new(discovery).await.unwrap();
    let tokenizer = resolver.resolve_with_fallback().await.unwrap();

    // Validate tokenization
    let prompt = "Hello, world!";
    let tokens = tokenizer.encode(prompt, true, false).unwrap();
    assert!(!tokens.is_empty());
    assert_eq!(tokens[0], 1);  // LLaMA-2 BOS token

    // Validate decoding
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert!(decoded.contains("Hello"));
}

// AC5:production_validation
#[tokio::test]
async fn test_e2e_llama3_large_vocab() {
    let model_path = download_test_model("llama3-8b.gguf");

    let discovery = TokenizerDiscovery::from_gguf(&model_path).unwrap();
    assert_eq!(discovery.vocab_size(), 128256);
    assert!(discovery.requires_large_vocab_optimization());

    let resolver = TokenizerStrategyResolver::new(discovery).await.unwrap();
    let tokenizer = resolver.resolve_with_fallback().await.unwrap();

    // Large vocabulary should work seamlessly
    let tokens = tokenizer.encode("Test with large vocabulary", true, false).unwrap();
    assert!(!tokens.is_empty());
}

// AC5:production_validation
#[tokio::test]
async fn test_e2e_bitnet_quantized_inference() {
    let model_path = download_test_model("bitnet-b1.58-2B.gguf");

    let discovery = TokenizerDiscovery::from_gguf(&model_path).unwrap();
    let resolver = TokenizerStrategyResolver::new(discovery).await.unwrap();
    let tokenizer = resolver.resolve_with_fallback().await.unwrap();

    // BitNet-specific validation
    let tokens = tokenizer.encode("BitNet quantization test", true, false).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert!(decoded.contains("quantization"));
}
```

### Cross-Validation Tests

**C++ Reference Validation**:
```bash
# AC5:production_validation
export BITNET_GGUF="models/bitnet-b1.58-2B.gguf"
export BITNET_STRICT_TOKENIZERS=1
cargo run -p xtask -- crossval

# Expected output:
# ✅ Tokenizer discovery: GGUF embedded extraction successful
# ✅ Tokenizer compatibility: >99% match with C++ reference
# ✅ Vocabulary size: 50257 (GPT-2 compatible)
# ✅ Special tokens: BOS=1, EOS=2, PAD=0
# ✅ Encoding accuracy: 100% token match with reference
# ✅ Decoding accuracy: 100% text match with reference
```

### Performance Benchmarks

**Discovery Latency Benchmarks**:
```rust
// AC5:production_validation
#[bench]
fn bench_tokenizer_discovery_metadata(b: &mut Bencher) {
    let model_path = get_test_model("llama2-7b.gguf");
    b.iter(|| {
        let discovery = TokenizerDiscovery::from_gguf(&model_path).unwrap();
        black_box(discovery.vocab_size());
    });
}

// AC5:production_validation
#[bench]
fn bench_tokenizer_discovery_large_vocab(b: &mut Bencher) {
    let model_path = get_test_model("llama3-8b.gguf");
    b.iter(|| {
        let discovery = TokenizerDiscovery::from_gguf(&model_path).unwrap();
        assert!(discovery.requires_large_vocab_optimization());
    });
}

// Expected performance targets:
// - Metadata extraction: <100ms (small vocabularies <64K)
// - Large vocabulary (128K+): <200ms (includes GPU detection)
// - Embedded tokenizer extraction: <50ms (memory-mapped access)
// - Download (cached): <10ms (local filesystem access)
```

## Environment Variables

**Configuration Options**:
```bash
# Strict mode: Prevent mock tokenizer fallbacks
BITNET_STRICT_TOKENIZERS=1

# Offline mode: Skip smart download attempts
BITNET_OFFLINE=1

# Deterministic inference: Reproducible tokenization
BITNET_DETERMINISTIC=1
BITNET_SEED=42

# Cache directory override
BITNET_CACHE_DIR=/custom/cache/path

# Download timeout (seconds)
BITNET_DOWNLOAD_TIMEOUT=300

# Test fixture paths
SPM_MODEL=/path/to/test/tokenizer.model
HF_TOKENIZER=/path/to/test/tokenizer.json
```

## Documentation Requirements

**User-Facing Documentation**:
- `docs/tutorials/tokenizer-discovery.md`: Guide for automatic tokenizer usage
- `docs/reference/tokenizer-api.md`: Complete API reference for discovery system
- `docs/explanation/tokenizer-architecture.md`: Architecture deep-dive

**Developer Documentation**:
- `docs/development/implementing-tokenizers.md`: Guide for adding new tokenizer types
- `docs/development/testing-tokenizers.md`: Testing strategies and fixtures
- `docs/architecture/tokenizer-discovery-design.md`: Design rationale and decisions

**Examples**:
```rust
// examples/tokenizer_discovery_basic.rs
use bitnet_tokenizers::{TokenizerDiscovery, TokenizerStrategyResolver};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Automatic discovery from GGUF model
    let model_path = std::path::Path::new("model.gguf");
    let discovery = TokenizerDiscovery::from_gguf(model_path)?;

    println!("Model type: {}", discovery.model_type());
    println!("Vocabulary size: {}", discovery.vocab_size());

    // Resolve tokenizer with automatic fallback
    let resolver = TokenizerStrategyResolver::new(discovery).await?;
    let tokenizer = resolver.resolve_with_fallback().await?;

    // Use tokenizer
    let tokens = tokenizer.encode("Hello, world!", true, false)?;
    println!("Tokens: {:?}", tokens);

    Ok(())
}
```

## Implementation Roadmap

### Phase 1: Core Discovery (Week 1-2)
- [ ] Implement `TokenizerDiscovery::from_gguf()` with memory-mapped GGUF parsing
- [ ] Complete `extract_model_type()` with comprehensive tensor pattern matching
- [ ] Implement `extract_vocab_size()` with metadata, tensor, and default fallbacks
- [ ] Add unit tests for AC2 (architecture detection) and AC3 (vocabulary resolution)

### Phase 2: Embedded Extraction (Week 2-3)
- [ ] Implement HuggingFace JSON tokenizer extraction from GGUF metadata
- [ ] Implement SentencePiece model loading from GGUF binary blobs
- [ ] Add special token extraction (BOS/EOS/PAD) from metadata
- [ ] Add unit tests for AC1 (embedded extraction)

### Phase 3: Smart Download (Week 3-4)
- [ ] Implement `SmartTokenizerDownload` with HuggingFace Hub integration
- [ ] Add download caching with integrity validation
- [ ] Implement retry logic with exponential backoff
- [ ] Add offline mode support and network error handling
- [ ] Add integration tests for AC4 (smart download)

### Phase 4: Strategy Resolution (Week 4-5)
- [ ] Implement `TokenizerStrategyResolver::resolve_with_fallback()`
- [ ] Complete fallback chain: GGUF → co-located → cache → download → mock
- [ ] Add model-specific wrappers (LLaMA, GPT-2, BitNet)
- [ ] Implement strict mode enforcement
- [ ] Add integration tests for AC5 (production validation)

### Phase 5: CLI & xtask Integration (Week 5-6)
- [ ] Integrate discovery into `bitnet-cli infer` command
- [ ] Add cross-validation to `xtask crossval` workflow
- [ ] Implement performance benchmarks and baselines
- [ ] Add comprehensive error messages and logging

### Phase 6: Documentation & Polish (Week 6)
- [ ] Write user-facing tutorials and examples
- [ ] Complete API reference documentation
- [ ] Add architecture documentation
- [ ] Create migration guide for existing users
- [ ] Final cross-validation and performance testing

## Success Metrics

**Functional Success**:
- ✅ All 5 acceptance criteria pass with comprehensive test coverage
- ✅ Cross-validation achieves >99% compatibility with C++ reference
- ✅ End-to-end inference works for LLaMA-2, LLaMA-3, GPT-2, BitNet models
- ✅ Strict mode prevents all mock fallbacks
- ✅ Smart download successfully acquires missing tokenizers

**Performance Success**:
- ✅ Discovery latency <100ms for small vocabularies (<64K tokens)
- ✅ Discovery latency <200ms for large vocabularies (128K+ tokens)
- ✅ Download caching reduces repeated downloads by >95%
- ✅ Memory efficiency: Zero-copy GGUF parsing with memory mapping
- ✅ GPU acceleration for large vocabularies (LLaMA-3)

**Quality Success**:
- ✅ Test coverage >90% for all discovery components
- ✅ All error paths tested with actionable error messages
- ✅ Feature flag discipline maintained (`--no-default-features --features cpu|gpu`)
- ✅ Documentation complete for all public APIs
- ✅ Examples demonstrate common usage patterns

## Related Issues

- **Issue #249**: Complete Tokenizer Integration (parent issue)
- **Issue #251**: Production Inference Server (downstream consumer)
- **Issue #260**: Mock Elimination (related quality improvement)

## References

- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/index)
- [SentencePiece Documentation](https://github.com/google/sentencepiece)
- [Microsoft BitNet C++ Reference](https://github.com/microsoft/BitNet)
- [BitNet.rs Architecture Overview](../architecture-overview.md)
- [Tokenizer Architecture Documentation](../tokenizer-architecture.md)
