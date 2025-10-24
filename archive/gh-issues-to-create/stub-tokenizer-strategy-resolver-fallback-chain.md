# Stub code: `TokenizerStrategyResolver::new` in `strategy.rs` has a placeholder for `_fallback_chain`

The `_fallback_chain` field in `TokenizerStrategyResolver::new` in `crates/bitnet-tokenizers/src/strategy.rs` is marked with `_` indicating it's unused. This suggests that the fallback chain is not actually being used in the `TokenizerStrategyResolver`. This is a form of stubbing.

**File:** `crates/bitnet-tokenizers/src/strategy.rs`

**Function:** `TokenizerStrategyResolver::new`

**Code:**
```rust
pub struct TokenizerStrategyResolver {
    discovery: TokenizerDiscovery,
    downloader: SmartTokenizerDownload,
    _fallback_chain: TokenizerFallbackChain,
}

impl TokenizerStrategyResolver {
    /// Create resolver with discovery engine and downloader
    pub async fn new(discovery: TokenizerDiscovery) -> Result<Self> {
        info!("Initializing TokenizerStrategyResolver for {} model", discovery.model_type());

        let downloader = SmartTokenizerDownload::new()?;
        let fallback_chain = TokenizerFallbackChain::new();

        Ok(Self { discovery, downloader, _fallback_chain: fallback_chain })
    }
```

## Proposed Fix

The `_fallback_chain` field should be used in the `TokenizerStrategyResolver`. This would involve using the `fallback_chain` to resolve tokenizer strategies.

### Example Implementation

```rust
pub struct TokenizerStrategyResolver {
    discovery: TokenizerDiscovery,
    downloader: SmartTokenizerDownload,
    fallback_chain: TokenizerFallbackChain,
}

impl TokenizerStrategyResolver {
    /// Create resolver with discovery engine and downloader
    pub async fn new(discovery: TokenizerDiscovery) -> Result<Self> {
        info!("Initializing TokenizerStrategyResolver for {} model", discovery.model_type());

        let downloader = SmartTokenizerDownload::new()?;
        let fallback_chain = TokenizerFallbackChain::new();

        Ok(Self { discovery, downloader, fallback_chain })
    }

    /// Resolve with automatic fallback chain
    pub async fn resolve_with_fallback(&self) -> Result<Arc<dyn Tokenizer>> {
        self.fallback_chain.resolve_tokenizer(&self.discovery).await.and_then(|resolution| resolution.into_tokenizer())
    }
```
