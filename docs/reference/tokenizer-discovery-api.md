# Tokenizer Discovery API Reference

**Version**: 1.0.0 (Issue #336)
**Module**: `bitnet-tokenizers`
**Feature Flags**: `cpu`, `gpu`, `spm` (SentencePiece support)

This document provides the complete API reference for BitNet-rs universal tokenizer discovery system, enabling automatic tokenizer resolution for quantized neural network models.

## Table of Contents

- [Core Discovery API](#core-discovery-api)
- [Strategy Resolution API](#strategy-resolution-api)
- [Download Integration API](#download-integration-api)
- [Model-Specific Wrappers](#model-specific-wrappers)
- [Error Handling](#error-handling)
- [Usage Examples](#usage-examples)

---

## Core Discovery API

### `TokenizerDiscovery`

Primary tokenizer discovery engine for BitNet-rs neural network models. Parses GGUF metadata, analyzes tensor patterns, and determines optimal tokenizer strategies.

#### Constructor

```rust
pub fn from_gguf(path: &Path) -> Result<Self>
```

Create discovery engine from GGUF model file.

**Parameters**:
- `path: &Path` - Path to GGUF model file

**Returns**:
- `Ok(TokenizerDiscovery)` - Successfully initialized discovery engine
- `Err(BitNetError::Model)` - GGUF parsing failed or metadata missing
- `Err(BitNetError::Config)` - Vocabulary size extraction failed

**Example**:
```rust
use bitnet_tokenizers::TokenizerDiscovery;
use std::path::Path;

let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;
assert_eq!(discovery.vocab_size(), 128256); // LLaMA-3
```

**Test Tag**: `// AC1:embedded_extraction`

#### Methods

##### `discover_tokenizer_strategy()`

```rust
pub fn discover_tokenizer_strategy(&self) -> Result<TokenizerStrategy>
```

Discover optimal tokenizer strategy for the loaded model.

**Returns**:
- `TokenizerStrategy::EmbeddedGguf` - GGUF contains embedded tokenizer
- `TokenizerStrategy::Discovered` - Compatible tokenizer found locally
- `TokenizerStrategy::NeedsDownload` - Smart download required
- `TokenizerStrategy::Mock` - Fallback for testing (non-strict mode only)

**Errors**:
- `BitNetError::Inference` - No compatible tokenizer found in strict mode

**Example**:
```rust
let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;
let strategy = discovery.discover_tokenizer_strategy()?;

match strategy {
    TokenizerStrategy::EmbeddedGguf(tokenizer) => {
        println!("Using embedded tokenizer");
    }
    TokenizerStrategy::NeedsDownload(info) => {
        println!("Will download from: {}", info.repo);
    }
    _ => {}
}
```

**Test Tag**: `// AC2:architecture_detection, AC3:vocab_resolution`

##### `try_extract_embedded_tokenizer()`

```rust
pub fn try_extract_embedded_tokenizer(&self) -> Result<Option<Arc<dyn Tokenizer>>>
```

Extract embedded tokenizer from GGUF metadata.

**Returns**:
- `Ok(Some(tokenizer))` - Successfully extracted HuggingFace or SentencePiece tokenizer
- `Ok(None)` - No embedded tokenizer found in GGUF
- `Err` - Embedded data exists but is invalid

**Supported Formats**:
- **HuggingFace JSON**: Embedded in `tokenizer.json` string metadata
- **SentencePiece**: Embedded in `tokenizer.ggml.model` binary metadata
- **Vocabulary tokens**: Embedded in `tokenizer.ggml.tokens` array metadata

**Example**:
```rust
let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;

if let Some(embedded) = discovery.try_extract_embedded_tokenizer()? {
    println!("Extracted embedded tokenizer with vocab: {}", embedded.vocab_size());
} else {
    println!("No embedded tokenizer, will try other strategies");
}
```

**Test Tag**: `// AC1:embedded_extraction`

##### `check_colocated_tokenizers()`

```rust
pub fn check_colocated_tokenizers(&self) -> Result<Option<PathBuf>>
```

Check for co-located tokenizer files in model directory.

**Returns**:
- `Ok(Some(path))` - Found compatible tokenizer file
- `Ok(None)` - No co-located tokenizer files found

**Searched Files** (in order):
1. `tokenizer.json` - HuggingFace tokenizer
2. `tokenizer.model` - SentencePiece model
3. `vocab.json` - Vocabulary file
4. `merges.txt` - BPE merges
5. `special_tokens_map.json` - Special token mapping
6. `{model_name}.tokenizer.json` - Model-specific naming
7. `{model_name}_tokenizer.json` - Alternative naming
8. `{model_name}.vocab.json` - Vocabulary-specific naming

**Example**:
```rust
let discovery = TokenizerDiscovery::from_gguf(Path::new("/models/llama2/model.gguf"))?;

if let Some(tokenizer_path) = discovery.check_colocated_tokenizers()? {
    println!("Found co-located tokenizer: {}", tokenizer_path.display());
    // Likely found: /models/llama2/tokenizer.json
}
```

**Test Tag**: `// AC4:smart_download`

##### `check_cache_locations()`

```rust
pub fn check_cache_locations(&self) -> Result<Option<PathBuf>>
```

Check standard cache directories for compatible tokenizers.

**Returns**:
- `Ok(Some(path))` - Found cached tokenizer
- `Ok(None)` - No cached tokenizer found

**Cache Locations** (in order):
1. **Model-specific cache**: `{BITNET_CACHE_DIR}/{model_type}/{vocab_size}/tokenizer.json`
2. **General model cache**: `{BITNET_CACHE_DIR}/{model_type}/tokenizer.json`
3. **HuggingFace cache**: `~/.cache/huggingface/{repo}/tokenizer.json`

**Example**:
```rust
let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;

if let Some(cached_path) = discovery.check_cache_locations()? {
    println!("Using cached tokenizer: {}", cached_path.display());
}
```

**Test Tag**: `// AC4:smart_download`

##### `infer_download_source()`

```rust
pub fn infer_download_source(&self) -> Result<Option<TokenizerDownloadInfo>>
```

Infer download source based on neural network model patterns.

**Returns**:
- `Ok(Some(download_info))` - Compatible download source identified
- `Ok(None)` - Unknown model type, no download source available

**Compatibility Matrix**:
| Model Type | Vocab Size | HuggingFace Repo | Cache Key |
|-----------|-----------|------------------|-----------|
| `llama` | 32000 | `meta-llama/Llama-2-7b-hf` | `llama2-32k` |
| `llama` | 128256 | `meta-llama/Meta-Llama-3-8B` | `llama3-128k` |
| `gpt2` | 50257 | `openai-community/gpt2` | `gpt2-50k` |
| `bitnet` | * | `1bitLLM/bitnet_b1_58-large` | `bitnet-custom` |

**Example**:
```rust
let discovery = TokenizerDiscovery::from_gguf(Path::new("llama2-model.gguf"))?;

if let Some(download_info) = discovery.infer_download_source()? {
    println!("Can download from: {}", download_info.repo);
    println!("Expected vocab: {:?}", download_info.expected_vocab);
}
```

**Test Tag**: `// AC4:smart_download`

##### Getter Methods

```rust
/// Get vocabulary size from model metadata
pub fn vocab_size(&self) -> usize

/// Get model architecture type (e.g., "llama", "gpt2", "bitnet")
pub fn model_type(&self) -> &str

/// Check if model requires large vocabulary optimization (>64K tokens)
pub fn requires_large_vocab_optimization(&self) -> bool
```

**Example**:
```rust
let discovery = TokenizerDiscovery::from_gguf(Path::new("llama3-8b.gguf"))?;

println!("Model: {}", discovery.model_type());          // "llama"
println!("Vocab: {}", discovery.vocab_size());          // 128256
println!("GPU opt: {}", discovery.requires_large_vocab_optimization()); // true
```

---

## Strategy Resolution API

### `TokenizerStrategyResolver`

Unified tokenizer strategy resolution with neural network model integration. Orchestrates fallback chains and applies model-specific configurations.

#### Constructor

```rust
pub async fn new(discovery: TokenizerDiscovery) -> Result<Self>
```

Create resolver with discovery engine and downloader.

**Parameters**:
- `discovery: TokenizerDiscovery` - Discovery engine from `from_gguf()`

**Returns**:
- `Ok(TokenizerStrategyResolver)` - Successfully initialized resolver
- `Err` - Download infrastructure initialization failed

**Example**:
```rust
let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;
let resolver = TokenizerStrategyResolver::new(discovery).await?;
```

**Test Tag**: `// AC4:smart_download`

#### Methods

##### `resolve_tokenizer()`

```rust
pub async fn resolve_tokenizer(
    &self,
    strategy: TokenizerStrategy
) -> Result<Arc<dyn Tokenizer>>
```

Resolve tokenizer strategy to concrete tokenizer implementation.

**Parameters**:
- `strategy: TokenizerStrategy` - Strategy from `discover_tokenizer_strategy()`

**Returns**:
- `Arc<dyn Tokenizer>` - Concrete tokenizer with model-specific wrapper
- `Err(BitNetError::Inference)` - Strategy resolution failed
- `Err(BitNetError::Model)` - Tokenizer loading or download failed

**Model-Specific Wrappers Applied**:
- **LLaMA models**: `LlamaTokenizerWrapper` with variant detection (LLaMA-2/3/CodeLlama)
- **GPT-2 models**: `Gpt2TokenizerWrapper` with proper special token handling
- **BitNet models**: `BitNetTokenizerWrapper` with quantization awareness

**Example**:
```rust
let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;
let resolver = TokenizerStrategyResolver::new(discovery).await?;

let strategy = discovery.discover_tokenizer_strategy()?;
let tokenizer = resolver.resolve_tokenizer(strategy).await?;

// tokenizer is now wrapped with model-specific behavior
let tokens = tokenizer.encode("Hello, world!", true, false)?;
```

**Test Tag**: `// AC5:production_validation`

##### `resolve_with_fallback()`

```rust
pub async fn resolve_with_fallback(&self) -> Result<Arc<dyn Tokenizer>>
```

Resolve with automatic fallback chain.

**Fallback Order**:
1. **GGUF embedded tokenizer** - Extract from model metadata
2. **Co-located files** - Search model directory
3. **Standard cache** - Check BitNet-rs and HuggingFace caches
4. **Smart download** - Download from HuggingFace Hub (if not offline)
5. **Mock fallback** - Testing only (non-strict mode)

**Environment Variables**:
- `BITNET_STRICT_TOKENIZERS=1` - Disable mock fallback (production mode)
- `BITNET_OFFLINE=1` - Skip smart download attempts
- `BITNET_CACHE_DIR=/custom/path` - Override cache directory

**Returns**:
- `Arc<dyn Tokenizer>` - Successfully resolved tokenizer
- `Err` - All strategies failed (with comprehensive error summary)

**Example**:
```rust
// Production usage: automatic discovery with full fallback chain
let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;
let resolver = TokenizerStrategyResolver::new(discovery).await?;
let tokenizer = resolver.resolve_with_fallback().await?;

// tokenizer is ready for inference with optimal configuration
let engine = InferenceEngine::new(model, tokenizer);
```

**Test Tag**: `// AC5:production_validation`

---

## Download Integration API

### `SmartTokenizerDownload`

Smart tokenizer downloading from HuggingFace Hub with caching and retry logic.

#### Constructor

```rust
pub fn new() -> Result<Self>
```

Create downloader with cache management.

**Returns**:
- `Ok(SmartTokenizerDownload)` - Successfully initialized downloader
- `Err` - Cache directory initialization failed

#### Methods

##### `download_tokenizer()`

```rust
pub async fn download_tokenizer(
    &self,
    info: &TokenizerDownloadInfo
) -> Result<PathBuf>
```

Download tokenizer from HuggingFace Hub.

**Parameters**:
- `info: &TokenizerDownloadInfo` - Download metadata from `infer_download_source()`

**Returns**:
- `Ok(PathBuf)` - Path to downloaded (or cached) tokenizer file
- `Err` - Download failed or cache validation failed

**Behavior**:
- **Cache hit**: Returns cached file path immediately
- **Cache miss**: Downloads from HuggingFace Hub, validates, and caches
- **Cache invalid**: Re-downloads and overwrites corrupted cache

**Network Configuration**:
- **Timeout**: 5 minutes (configurable via `BITNET_DOWNLOAD_TIMEOUT`)
- **Retry**: 3 attempts with exponential backoff (1s, 2s, 4s)
- **Offline mode**: Skips download when `BITNET_OFFLINE=1` set

**Example**:
```rust
let downloader = SmartTokenizerDownload::new()?;
let download_info = TokenizerDownloadInfo {
    repo: "meta-llama/Llama-2-7b-hf".to_string(),
    files: vec!["tokenizer.json".to_string()],
    cache_key: "llama2-32k".to_string(),
    expected_vocab: Some(32000),
};

let tokenizer_path = downloader.download_tokenizer(&download_info).await?;
println!("Downloaded to: {}", tokenizer_path.display());

// Subsequent calls use cache
let cached_path = downloader.download_tokenizer(&download_info).await?;
assert_eq!(tokenizer_path, cached_path);
```

**Test Tag**: `// AC4:smart_download`

##### `is_cached()`

```rust
pub fn is_cached(&self, info: &TokenizerDownloadInfo) -> bool
```

Check if tokenizer is already cached.

**Returns**:
- `true` - Tokenizer exists in cache and is valid
- `false` - Tokenizer not cached or cache is invalid

**Example**:
```rust
let downloader = SmartTokenizerDownload::new()?;
let download_info = create_download_info("llama2-32k");

if downloader.is_cached(&download_info) {
    println!("Using cached tokenizer");
} else {
    println!("Will download tokenizer");
}
```

##### `validate_cache()`

```rust
pub fn validate_cache(&self, path: &Path) -> Result<()>
```

Validate cached tokenizer integrity.

**Parameters**:
- `path: &Path` - Path to cached tokenizer file

**Returns**:
- `Ok(())` - Cache is valid
- `Err` - Cache is corrupted or invalid

**Validation Checks**:
1. File exists and is readable
2. File size > 0 bytes
3. JSON structure is valid
4. Required fields present (for HuggingFace tokenizers)

**Example**:
```rust
let downloader = SmartTokenizerDownload::new()?;
let cache_path = Path::new("/cache/tokenizers/llama2-32k/tokenizer.json");

match downloader.validate_cache(&cache_path) {
    Ok(()) => println!("Cache is valid"),
    Err(e) => println!("Cache corrupted: {}, will re-download", e),
}
```

### `TokenizerDownloadInfo`

Download metadata for tokenizer acquisition.

```rust
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

**Example**:
```rust
let download_info = TokenizerDownloadInfo {
    repo: "meta-llama/Meta-Llama-3-8B".to_string(),
    files: vec!["tokenizer.json".to_string()],
    cache_key: "llama3-128k".to_string(),
    expected_vocab: Some(128256),
};
```

---

## Model-Specific Wrappers

### `LlamaTokenizerWrapper`

LLaMA model-specific tokenizer wrapper with neural network optimizations.

```rust
pub struct LlamaTokenizerWrapper {
    inner: Arc<dyn Tokenizer>,
    vocab_size: usize,
    model_variant: LlamaVariant,
}
```

#### Constructor

```rust
pub fn new(inner: Arc<dyn Tokenizer>, vocab_size: usize) -> Result<Self>
```

Create LLaMA tokenizer wrapper with variant-specific configuration.

**Parameters**:
- `inner: Arc<dyn Tokenizer>` - Base tokenizer implementation
- `vocab_size: usize` - Expected vocabulary size for variant detection

**Variant Detection**:
- **LLaMA-2**: `vocab_size == 32000`
- **LLaMA-3**: `vocab_size == 128256`
- **CodeLlama**: `vocab_size == 32016`

**Example**:
```rust
let base_tokenizer = load_tokenizer_from_path("tokenizer.json")?;
let llama_wrapper = LlamaTokenizerWrapper::new(base_tokenizer, 32000)?;

// Wrapper applies LLaMA-2 specific special token handling
let tokens = llama_wrapper.encode("Hello", true, false)?;
assert_eq!(tokens[0], 1);  // LLaMA-2 BOS token
```

**Test Tag**: `// AC5:production_validation`

#### Special Token Handling

**LLaMA-2**:
- BOS: `1`
- EOS: `2`
- Special tokens: `0-2` (UNK, BOS, EOS)

**LLaMA-3**:
- BOS: `128000`
- EOS: `128001`
- Special tokens: `128000-128002` (enhanced special token range)

**CodeLlama**:
- BOS: `1`
- EOS: `2`
- Special tokens: `0-2` (same as LLaMA-2)

### `Gpt2TokenizerWrapper`

GPT-2 model-specific tokenizer wrapper.

```rust
pub struct Gpt2TokenizerWrapper {
    inner: Arc<dyn Tokenizer>,
}
```

#### Constructor

```rust
pub fn new(inner: Arc<dyn Tokenizer>) -> Result<Self>
```

Create GPT-2 tokenizer wrapper.

**Vocabulary Validation**:
- Expected: `50257` tokens
- Warning logged if mismatch detected

**Example**:
```rust
let base_tokenizer = load_tokenizer_from_path("tokenizer.json")?;
let gpt2_wrapper = Gpt2TokenizerWrapper::new(base_tokenizer)?;

// GPT-2 doesn't use BOS tokens
let tokens = gpt2_wrapper.encode("Hello", true, false)?;
// add_bos=true is ignored with warning
```

**Test Tag**: `// AC5:production_validation`

#### Special Token Handling

**GPT-2**:
- BOS: **None** (GPT-2 doesn't use BOS tokens)
- EOS: `50256`
- Note: `add_bos=true` is ignored with warning

### `BitNetTokenizerWrapper`

BitNet model-specific tokenizer wrapper with quantization awareness.

```rust
pub struct BitNetTokenizerWrapper {
    inner: Arc<dyn Tokenizer>,
    quantization_type: QuantizationType,
}
```

#### Constructor

```rust
pub fn new(
    inner: Arc<dyn Tokenizer>,
    quantization_type: QuantizationType
) -> Result<Self>
```

Create BitNet tokenizer wrapper with quantization-specific optimizations.

**Parameters**:
- `inner: Arc<dyn Tokenizer>` - Base tokenizer implementation
- `quantization_type: QuantizationType` - I2_S, TL1, or TL2

**Quantization Compatibility Warnings**:
- **I2_S**: Warning if `vocab_size > 200000` (very large vocabulary)
- **TL1/TL2**: Warning if `vocab_size > 65536` (exceeds table lookup efficiency)

**Example**:
```rust
let base_tokenizer = load_tokenizer_from_path("tokenizer.json")?;
let bitnet_wrapper = BitNetTokenizerWrapper::new(
    base_tokenizer,
    QuantizationType::I2S
)?;

// Validates token IDs are compatible with quantization format
let tokens = bitnet_wrapper.encode("Test", true, false)?;
```

**Test Tag**: `// AC5:production_validation`

#### Methods

```rust
/// Get the quantization type used by this tokenizer
pub fn quantization_type(&self) -> QuantizationType

/// Validate token IDs are compatible with quantization format
fn validate_quantization_compatibility(&self, tokens: &[u32]) -> Result<()>
```

### `LlamaVariant`

LLaMA model variant enumeration.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaVariant {
    Llama2,      // 32K vocabulary, legacy special tokens
    Llama3,      // 128K vocabulary, enhanced special tokens
    CodeLlama,   // 32016 vocabulary, code-optimized
}
```

#### Methods

```rust
/// Get expected vocabulary size for variant
pub fn expected_vocab_size(&self) -> usize

/// Check if variant requires GPU acceleration for large vocabulary
pub fn requires_gpu_acceleration(&self) -> bool
```

**Example**:
```rust
let variant = LlamaVariant::Llama3;
assert_eq!(variant.expected_vocab_size(), 128256);
assert!(variant.requires_gpu_acceleration());  // true for LLaMA-3
```

---

## Tokenizer Strategy Enum

### `TokenizerStrategy`

Comprehensive tokenizer resolution strategy for neural network models.

```rust
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
```

#### Methods

```rust
/// Check if strategy requires network access
pub fn requires_network(&self) -> bool

/// Check if strategy uses cached resources
pub fn uses_cache(&self) -> bool

/// Get description for logging and error messages
pub fn description(&self) -> &'static str
```

**Example**:
```rust
let strategy = discovery.discover_tokenizer_strategy()?;

if strategy.requires_network() {
    println!("Will download tokenizer from HuggingFace Hub");
}

if strategy.uses_cache() {
    println!("Using cached tokenizer resources");
}

println!("Strategy: {}", strategy.description());
```

---

## Error Handling

### Error Types

**BitNetError Variants**:
- `BitNetError::Model` - GGUF parsing failed or model invalid
- `BitNetError::Config` - Configuration error (vocab size, architecture, etc.)
- `BitNetError::Inference` - Tokenizer resolution failed
- `BitNetError::Io` - File I/O error (file not found, permissions, etc.)
- `BitNetError::Network` - Download failed (network timeout, 404, etc.)

### Error Messages

All errors provide actionable messages with context:

```rust
// Example error messages:
"Could not extract vocabulary size from GGUF metadata, tensors, or architecture defaults"
"No compatible tokenizer found for llama model with vocab_size 128256 (strict mode)"
"Download failed from https://huggingface.co/meta-llama/Llama-2-7b-hf: HTTP 404"
"Cached tokenizer file is corrupted: /cache/llama2-32k/tokenizer.json"
"All tokenizer resolution strategies failed. Tried: GGUF embedded, co-located files, cache, smart download"
```

### Error Handling Patterns

```rust
// Pattern 1: Simple error propagation
let discovery = TokenizerDiscovery::from_gguf(path)?;

// Pattern 2: Fallback on error
let tokenizer = match discovery.try_extract_embedded_tokenizer()? {
    Some(t) => t,
    None => {
        // Fallback to other strategies
        resolver.resolve_with_fallback().await?
    }
};

// Pattern 3: Collect errors for comprehensive reporting
let mut errors = Vec::new();
for strategy in strategies {
    match try_strategy(strategy) {
        Ok(tokenizer) => return Ok(tokenizer),
        Err(e) => errors.push((strategy, e)),
    }
}
return Err(create_comprehensive_error(errors));
```

---

## Usage Examples

### Basic Usage

```rust
use bitnet_tokenizers::{TokenizerDiscovery, TokenizerStrategyResolver};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Automatic discovery from GGUF model
    let model_path = Path::new("model.gguf");
    let discovery = TokenizerDiscovery::from_gguf(model_path)?;

    println!("Model type: {}", discovery.model_type());
    println!("Vocabulary size: {}", discovery.vocab_size());

    // Resolve tokenizer with automatic fallback
    let resolver = TokenizerStrategyResolver::new(discovery).await?;
    let tokenizer = resolver.resolve_with_fallback().await?;

    // Use tokenizer
    let tokens = tokenizer.encode("Hello, world!", true, false)?;
    println!("Tokens: {:?}", tokens);

    let decoded = tokenizer.decode(&tokens)?;
    println!("Decoded: {}", decoded);

    Ok(())
}
```

### Manual Strategy Selection

```rust
use bitnet_tokenizers::{TokenizerDiscovery, TokenizerStrategyResolver};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;
    let resolver = TokenizerStrategyResolver::new(discovery).await?;

    // Try strategies manually
    if let Ok(Some(embedded)) = discovery.try_extract_embedded_tokenizer() {
        println!("Using embedded tokenizer");
        let tokenizer = resolver.resolve_tokenizer(
            TokenizerStrategy::EmbeddedGguf(embedded)
        ).await?;
    } else if let Ok(Some(path)) = discovery.check_colocated_tokenizers() {
        println!("Using co-located tokenizer");
        let tokenizer = resolver.resolve_tokenizer(
            TokenizerStrategy::Discovered(path)
        ).await?;
    } else {
        println!("Using fallback chain");
        let tokenizer = resolver.resolve_with_fallback().await?;
    }

    Ok(())
}
```

### Production Inference

```rust
use bitnet_tokenizers::{TokenizerDiscovery, TokenizerStrategyResolver};
use bitnet_inference::InferenceEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Enable strict mode for production
    std::env::set_var("BITNET_STRICT_TOKENIZERS", "1");

    let model_path = Path::new("llama2-7b-chat.gguf");

    // Automatic tokenizer discovery
    let discovery = TokenizerDiscovery::from_gguf(model_path)?;
    let resolver = TokenizerStrategyResolver::new(discovery).await?;
    let tokenizer = resolver.resolve_with_fallback().await?;

    // Load model and create inference engine
    let engine = InferenceEngine::new_from_gguf(model_path, tokenizer)?;

    // Generate text
    let prompt = "Hello, how are you?";
    let output = engine.generate_text(prompt, 50)?;
    println!("Generated: {}", output);

    Ok(())
}
```

### Custom Download Source

```rust
use bitnet_tokenizers::{SmartTokenizerDownload, TokenizerDownloadInfo};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let downloader = SmartTokenizerDownload::new()?;

    // Custom download configuration
    let download_info = TokenizerDownloadInfo {
        repo: "my-org/custom-model".to_string(),
        files: vec!["tokenizer.json".to_string()],
        cache_key: "custom-model".to_string(),
        expected_vocab: Some(50000),
    };

    // Download with caching
    let tokenizer_path = downloader.download_tokenizer(&download_info).await?;
    println!("Downloaded to: {}", tokenizer_path.display());

    // Load tokenizer
    let (tokenizer, _kind) = bitnet_tokenizers::from_path(&tokenizer_path)?;
    println!("Loaded tokenizer with vocab: {}", tokenizer.vocab_size());

    Ok(())
}
```

### Model-Specific Wrappers

```rust
use bitnet_tokenizers::{LlamaTokenizerWrapper, Gpt2TokenizerWrapper, BitNetTokenizerWrapper};
use bitnet_common::QuantizationType;

fn configure_llama_tokenizer(base: Arc<dyn Tokenizer>) -> Result<Arc<dyn Tokenizer>> {
    let wrapper = LlamaTokenizerWrapper::new(base, 128256)?;  // LLaMA-3
    Ok(Arc::new(wrapper))
}

fn configure_gpt2_tokenizer(base: Arc<dyn Tokenizer>) -> Result<Arc<dyn Tokenizer>> {
    let wrapper = Gpt2TokenizerWrapper::new(base)?;
    Ok(Arc::new(wrapper))
}

fn configure_bitnet_tokenizer(base: Arc<dyn Tokenizer>) -> Result<Arc<dyn Tokenizer>> {
    let wrapper = BitNetTokenizerWrapper::new(base, QuantizationType::I2S)?;
    Ok(Arc::new(wrapper))
}
```

---

## Environment Variables

**Configuration Options**:

```bash
# Strict mode: Prevent mock tokenizer fallbacks (production)
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

**Example Usage**:
```bash
# Production inference with strict mode
BITNET_STRICT_TOKENIZERS=1 cargo run -p bitnet-cli -- infer --model model.gguf --prompt "Test"

# Offline development
BITNET_OFFLINE=1 cargo test --no-default-features --features cpu

# Reproducible testing
BITNET_DETERMINISTIC=1 BITNET_SEED=42 cargo test
```

---

## Feature Flags

**Compilation Features**:

```bash
# CPU-only build (default recommended)
cargo build --no-default-features --features cpu

# GPU acceleration build
cargo build --no-default-features --features gpu

# SentencePiece support (optional)
cargo build --no-default-features --features cpu,spm

# All features (for development)
cargo build --all-features
```

**Feature Matrix**:
| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `cpu` | CPU inference with SIMD optimization | Required for tokenizer discovery |
| `gpu` | GPU acceleration for large vocabularies | CUDA toolkit |
| `spm` | SentencePiece tokenizer support | `sentencepiece-rs` |
| `ffi` | C++ FFI bridge (gradual migration) | C++ compiler |

---

## Performance Considerations

**Discovery Latency**:
- Metadata extraction: <100ms (small vocabularies <64K)
- Large vocabulary (128K+): <200ms (includes GPU detection)
- Embedded tokenizer extraction: <50ms (memory-mapped access)
- Download (cached): <10ms (local filesystem access)

**Memory Efficiency**:
- Zero-copy GGUF parsing via memory mapping
- Lazy tokenizer instantiation (deferred initialization)
- GPU memory optimization for large vocabularies

**GPU Acceleration Thresholds**:
- Vocabulary size ≤ 65536: CPU sufficient
- Vocabulary size > 65536: GPU recommended
- LLaMA-3 (128256 tokens): GPU required for optimal performance

---

## Testing

**Run Tests**:
```bash
# Unit tests
cargo test --no-default-features --features cpu -p bitnet-tokenizers

# Integration tests
cargo test --no-default-features --features cpu --test tokenizer_discovery_integration

# Cross-validation
export BITNET_GGUF="model.gguf"
cargo run -p xtask -- crossval

# Benchmarks
cargo bench --no-default-features --features cpu --bench tokenizer_discovery_bench
```

**Test Tags**:
- `// AC1:embedded_extraction` - Embedded tokenizer tests
- `// AC2:architecture_detection` - Model type detection tests
- `// AC3:vocab_resolution` - Vocabulary size extraction tests
- `// AC4:smart_download` - Download integration tests
- `// AC5:production_validation` - End-to-end validation tests

---

## See Also

- [Feature Specification](../explanation/issue-336-universal-tokenizer-discovery-spec.md)
- [Tokenizer Architecture](../tokenizer-architecture.md)
- [GGUF Weight Loading](../explanation/gguf-weight-loading.md)
- [Build Commands](../development/build-commands.md)
- [GPU Development Guide](../development/gpu-development.md)
