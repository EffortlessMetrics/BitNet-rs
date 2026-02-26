# [Tokenizer] Implement Tokenizer Fallback Chain Resolution

## Problem Description

The `TokenizerStrategyResolver` in `crates/bitnet-tokenizers/src/strategy.rs` contains an unused `_fallback_chain` field, indicating that the fallback mechanism for tokenizer resolution is not implemented. This prevents the system from gracefully handling tokenizer loading failures and reduces robustness when dealing with various model formats.

## Environment

- **Component**: `crates/bitnet-tokenizers/src/strategy.rs`
- **Struct**: `TokenizerStrategyResolver`
- **Field**: `_fallback_chain: TokenizerFallbackChain`
- **Impact**: Tokenizer loading reliability and fallback handling

## Current Implementation Analysis

```rust
pub struct TokenizerStrategyResolver {
    discovery: TokenizerDiscovery,
    downloader: SmartTokenizerDownload,
    _fallback_chain: TokenizerFallbackChain,  // Unused field
}

impl TokenizerStrategyResolver {
    pub async fn new(discovery: TokenizerDiscovery) -> Result<Self> {
        let downloader = SmartTokenizerDownload::new()?;
        let fallback_chain = TokenizerFallbackChain::new();

        Ok(Self { discovery, downloader, _fallback_chain: fallback_chain })
    }
}
```

**Issues Identified:**
1. **Unused fallback mechanism**: `_fallback_chain` field is not utilized
2. **No fallback resolution**: No methods to use the fallback chain
3. **Reduced reliability**: Single point of failure for tokenizer loading
4. **Missing graceful degradation**: No alternative strategies when primary loading fails

## Impact Assessment

**Severity**: Medium
**Affected Users**: Users with models using non-standard tokenizer formats or network connectivity issues
**Functional Impact**:
- Tokenizer loading fails completely when primary strategy fails
- No graceful degradation for different tokenizer sources
- Reduced robustness for various model architectures

## Proposed Solution

### 1. Implement Comprehensive Fallback Chain System

```rust
impl TokenizerStrategyResolver {
    pub async fn new(discovery: TokenizerDiscovery) -> Result<Self> {
        info!("Initializing TokenizerStrategyResolver for {} model", discovery.model_type());

        let downloader = SmartTokenizerDownload::new()?;
        let fallback_chain = TokenizerFallbackChain::new();

        Ok(Self {
            discovery,
            downloader,
            fallback_chain,  // Remove underscore prefix
        })
    }

    /// Resolve tokenizer with automatic fallback handling
    pub async fn resolve_with_fallback(&self, model_path: &Path) -> Result<Arc<dyn Tokenizer>> {
        // Try primary resolution strategy first
        match self.resolve_primary(model_path).await {
            Ok(tokenizer) => {
                info!("Successfully resolved tokenizer using primary strategy");
                Ok(tokenizer)
            }
            Err(primary_error) => {
                warn!("Primary tokenizer resolution failed: {}", primary_error);

                // Fall back to fallback chain
                self.fallback_chain.resolve_tokenizer(&self.discovery, model_path).await
                    .with_context(|| format!("All fallback strategies failed after primary error: {}", primary_error))
            }
        }
    }

    async fn resolve_primary(&self, model_path: &Path) -> Result<Arc<dyn Tokenizer>> {
        // Existing primary resolution logic
        self.discovery.discover_tokenizer(model_path).await
    }
}

pub struct TokenizerFallbackChain {
    strategies: Vec<Box<dyn FallbackStrategy>>,
}

impl TokenizerFallbackChain {
    pub fn new() -> Self {
        let strategies: Vec<Box<dyn FallbackStrategy>> = vec![
            Box::new(ModelEmbeddedStrategy::new()),
            Box::new(HuggingFaceStrategy::new()),
            Box::new(LocalCacheStrategy::new()),
            Box::new(GenericStrategy::new()),
            Box::new(BackupTokenizerStrategy::new()),
        ];

        Self { strategies }
    }

    pub async fn resolve_tokenizer(
        &self,
        discovery: &TokenizerDiscovery,
        model_path: &Path,
    ) -> Result<Arc<dyn Tokenizer>> {
        let mut last_error = None;

        for (index, strategy) in self.strategies.iter().enumerate() {
            info!("Attempting fallback strategy {}: {}", index + 1, strategy.name());

            match strategy.try_resolve(discovery, model_path).await {
                Ok(tokenizer) => {
                    info!("Successfully resolved tokenizer using fallback strategy: {}", strategy.name());
                    return Ok(tokenizer);
                }
                Err(error) => {
                    warn!("Fallback strategy '{}' failed: {}", strategy.name(), error);
                    last_error = Some(error);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("No fallback strategies available")))
    }
}

#[async_trait]
trait FallbackStrategy: Send + Sync {
    fn name(&self) -> &'static str;
    async fn try_resolve(
        &self,
        discovery: &TokenizerDiscovery,
        model_path: &Path,
    ) -> Result<Arc<dyn Tokenizer>>;
    fn priority(&self) -> u8; // Higher number = higher priority
}
```

### 2. Specific Fallback Strategy Implementations

```rust
struct ModelEmbeddedStrategy;

impl ModelEmbeddedStrategy {
    fn new() -> Self { Self }
}

#[async_trait]
impl FallbackStrategy for ModelEmbeddedStrategy {
    fn name(&self) -> &'static str { "ModelEmbedded" }

    fn priority(&self) -> u8 { 100 } // Highest priority

    async fn try_resolve(
        &self,
        discovery: &TokenizerDiscovery,
        model_path: &Path,
    ) -> Result<Arc<dyn Tokenizer>> {
        // Try to extract tokenizer from model file itself
        if model_path.extension() == Some(std::ffi::OsStr::new("gguf")) {
            return self.try_gguf_embedded_tokenizer(model_path).await;
        }

        Err(anyhow::anyhow!("No embedded tokenizer found in model"))
    }

    async fn try_gguf_embedded_tokenizer(&self, model_path: &Path) -> Result<Arc<dyn Tokenizer>> {
        // Implementation to extract tokenizer from GGUF metadata
        let mut reader = GgufReader::from_file(model_path)?;

        // Check if tokenizer metadata exists
        if let Ok(tokenizer_model) = reader.get_metadata_value("tokenizer.ggml.model") {
            let vocab = reader.get_string_array("tokenizer.ggml.tokens")?;
            return Ok(Arc::new(GgufTokenizer::from_vocab(vocab)?));
        }

        Err(anyhow::anyhow!("No tokenizer metadata in GGUF file"))
    }
}

struct HuggingFaceStrategy {
    client: Option<HfClient>,
}

impl HuggingFaceStrategy {
    fn new() -> Self {
        let client = HfClient::new().ok();
        Self { client }
    }
}

#[async_trait]
impl FallbackStrategy for HuggingFaceStrategy {
    fn name(&self) -> &'static str { "HuggingFace" }

    fn priority(&self) -> u8 { 80 }

    async fn try_resolve(
        &self,
        discovery: &TokenizerDiscovery,
        _model_path: &Path,
    ) -> Result<Arc<dyn Tokenizer>> {
        let client = self.client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("HuggingFace client not available"))?;

        // Try to infer model name and download tokenizer
        let model_name = discovery.infer_model_name()?;

        info!("Attempting to download tokenizer for model: {}", model_name);

        let tokenizer_files = client.download_tokenizer_files(&model_name).await?;
        let tokenizer = HfTokenizer::from_files(tokenizer_files)?;

        Ok(Arc::new(tokenizer))
    }
}

struct LocalCacheStrategy {
    cache_dir: PathBuf,
}

impl LocalCacheStrategy {
    fn new() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("bitnet")
            .join("tokenizers");

        Self { cache_dir }
    }
}

#[async_trait]
impl FallbackStrategy for LocalCacheStrategy {
    fn name(&self) -> &'static str { "LocalCache" }

    fn priority(&self) -> u8 { 60 }

    async fn try_resolve(
        &self,
        discovery: &TokenizerDiscovery,
        model_path: &Path,
    ) -> Result<Arc<dyn Tokenizer>> {
        // Generate cache key based on model path and metadata
        let cache_key = self.generate_cache_key(model_path)?;
        let cache_path = self.cache_dir.join(format!("{}.tokenizer", cache_key));

        if cache_path.exists() {
            info!("Found cached tokenizer at: {}", cache_path.display());
            let tokenizer = CachedTokenizer::from_file(&cache_path)?;
            return Ok(Arc::new(tokenizer));
        }

        Err(anyhow::anyhow!("No cached tokenizer found"))
    }

    fn generate_cache_key(&self, model_path: &Path) -> Result<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let metadata = std::fs::metadata(model_path)?;
        let mut hasher = DefaultHasher::new();

        model_path.hash(&mut hasher);
        metadata.len().hash(&mut hasher);
        metadata.modified()?.duration_since(std::time::UNIX_EPOCH)?.hash(&mut hasher);

        Ok(format!("{:x}", hasher.finish()))
    }
}

struct GenericStrategy;

impl GenericStrategy {
    fn new() -> Self { Self }
}

#[async_trait]
impl FallbackStrategy for GenericStrategy {
    fn name(&self) -> &'static str { "Generic" }

    fn priority(&self) -> u8 { 40 }

    async fn try_resolve(
        &self,
        discovery: &TokenizerDiscovery,
        model_path: &Path,
    ) -> Result<Arc<dyn Tokenizer>> {
        // Try common tokenizer file patterns in the same directory
        let model_dir = model_path.parent()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine model directory"))?;

        let tokenizer_patterns = [
            "tokenizer.json",
            "tokenizer.model",
            "vocab.txt",
            "merges.txt",
        ];

        for pattern in &tokenizer_patterns {
            let tokenizer_path = model_dir.join(pattern);
            if tokenizer_path.exists() {
                info!("Found tokenizer file: {}", tokenizer_path.display());

                match self.load_tokenizer_from_file(&tokenizer_path).await {
                    Ok(tokenizer) => return Ok(tokenizer),
                    Err(e) => warn!("Failed to load tokenizer from {}: {}", tokenizer_path.display(), e),
                }
            }
        }

        Err(anyhow::anyhow!("No generic tokenizer files found"))
    }

    async fn load_tokenizer_from_file(&self, path: &Path) -> Result<Arc<dyn Tokenizer>> {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => {
                let tokenizer = HfTokenizer::from_file(path)?;
                Ok(Arc::new(tokenizer))
            }
            Some("model") => {
                let tokenizer = SentencePieceTokenizer::from_file(path)?;
                Ok(Arc::new(tokenizer))
            }
            Some("txt") => {
                let tokenizer = VocabTokenizer::from_file(path)?;
                Ok(Arc::new(tokenizer))
            }
            _ => Err(anyhow::anyhow!("Unsupported tokenizer file format")),
        }
    }
}

struct BackupTokenizerStrategy;

impl BackupTokenizerStrategy {
    fn new() -> Self { Self }
}

#[async_trait]
impl FallbackStrategy for BackupTokenizerStrategy {
    fn name(&self) -> &'static str { "Backup" }

    fn priority(&self) -> u8 { 20 } // Lowest priority

    async fn try_resolve(
        &self,
        discovery: &TokenizerDiscovery,
        _model_path: &Path,
    ) -> Result<Arc<dyn Tokenizer>> {
        warn!("Using backup tokenizer - this may result in suboptimal performance");

        // Create a basic tokenizer with reasonable defaults
        let vocab_size = 32000; // Common vocab size
        let backup_tokenizer = BasicTokenizer::with_vocab_size(vocab_size)?;

        Ok(Arc::new(backup_tokenizer))
    }
}
```

### 3. Enhanced Resolution with Caching

```rust
impl TokenizerStrategyResolver {
    pub async fn resolve_with_caching(&self, model_path: &Path) -> Result<Arc<dyn Tokenizer>> {
        // Check if we have a cached successful resolution
        if let Some(cached) = self.check_resolution_cache(model_path).await? {
            return Ok(cached);
        }

        // Perform resolution with fallback
        let tokenizer = self.resolve_with_fallback(model_path).await?;

        // Cache the successful resolution
        self.cache_successful_resolution(model_path, &tokenizer).await?;

        Ok(tokenizer)
    }

    async fn check_resolution_cache(&self, model_path: &Path) -> Result<Option<Arc<dyn Tokenizer>>> {
        // Implementation to check if we have a cached tokenizer for this model
        // This could use file system cache, in-memory cache, or database
        Ok(None) // Placeholder
    }

    async fn cache_successful_resolution(
        &self,
        model_path: &Path,
        tokenizer: &Arc<dyn Tokenizer>
    ) -> Result<()> {
        // Implementation to cache the successful tokenizer resolution
        // This helps avoid repeating the fallback chain for the same model
        Ok(()) // Placeholder
    }
}
```

## Implementation Breakdown

### Phase 1: Core Fallback Infrastructure
- [ ] Remove underscore prefix from fallback_chain field
- [ ] Implement TokenizerFallbackChain with strategy pattern
- [ ] Add FallbackStrategy trait definition
- [ ] Create resolve_with_fallback method

### Phase 2: Strategy Implementations
- [ ] Implement ModelEmbeddedStrategy for GGUF tokenizers
- [ ] Add HuggingFaceStrategy for remote download
- [ ] Create LocalCacheStrategy for cached tokenizers
- [ ] Implement GenericStrategy for common file patterns

### Phase 3: Advanced Features
- [ ] Add BackupTokenizerStrategy as last resort
- [ ] Implement resolution caching system
- [ ] Add strategy priority ordering
- [ ] Create comprehensive error handling

### Phase 4: Integration and Testing
- [ ] Update existing tokenizer loading code
- [ ] Add comprehensive test coverage
- [ ] Create integration tests with various model types
- [ ] Add performance monitoring

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fallback_chain_priority_order() {
        let fallback_chain = TokenizerFallbackChain::new();

        // Verify strategies are ordered by priority
        let priorities: Vec<u8> = fallback_chain.strategies.iter()
            .map(|s| s.priority())
            .collect();

        let mut sorted_priorities = priorities.clone();
        sorted_priorities.sort_by(|a, b| b.cmp(a)); // Descending order

        assert_eq!(priorities, sorted_priorities);
    }

    #[tokio::test]
    async fn test_fallback_after_primary_failure() {
        let resolver = create_test_resolver().await;
        let test_model_path = Path::new("tests/data/invalid_model.gguf");

        // Should succeed via fallback even if primary fails
        let result = resolver.resolve_with_fallback(test_model_path).await;
        assert!(result.is_ok());
    }
}
```

### Integration Tests
```rust
#[tokio::test]
async fn test_real_model_fallback_resolution() {
    let discovery = TokenizerDiscovery::new();
    let resolver = TokenizerStrategyResolver::new(discovery).await.unwrap();

    // Test with a model that requires fallback
    let model_path = Path::new("tests/data/model_without_tokenizer.gguf");
    let tokenizer = resolver.resolve_with_fallback(model_path).await.unwrap();

    // Verify tokenizer works
    let tokens = tokenizer.encode("Hello, world!").unwrap();
    assert!(!tokens.is_empty());
}
```

## Acceptance Criteria

- [ ] Remove unused field indicator (_fallback_chain -> fallback_chain)
- [ ] Implement comprehensive fallback strategy system
- [ ] Multiple fallback strategies with priority ordering
- [ ] Graceful degradation when primary resolution fails
- [ ] Caching of successful resolutions
- [ ] Comprehensive error handling and logging
- [ ] Test coverage for all fallback scenarios

## Related Issues/PRs

- **Related to**: Tokenizer discovery improvements
- **Depends on**: Model format detection enhancements
- **Blocks**: Robust model loading pipeline
- **References**: Error handling framework improvements

## Additional Context

Implementing the fallback chain is crucial for making BitNet-rs robust across different model formats and deployment environments. The fallback system should provide multiple strategies for tokenizer resolution while maintaining performance and reliability.
