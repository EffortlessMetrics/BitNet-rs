# [Tokenizer] Replace hardcoded LLaMA special token detection with dynamic configuration

## Problem Description

The `LlamaTokenizerWrapper::is_special_token` method in `crates/bitnet-tokenizers/src/strategy.rs` contains hardcoded values for LLaMA-2, LLaMA-3, and CodeLlama special tokens. This implementation creates several critical problems for the tokenizer system's flexibility, accuracy, and maintainability.

## Environment
- **Affected File**: `crates/bitnet-tokenizers/src/strategy.rs`
- **Function**: `LlamaTokenizerWrapper::is_special_token`
- **Lines**: 326-338
- **Impact**: Token classification accuracy, model compatibility, tokenizer flexibility
- **Architecture**: Universal tokenizer system with neural network model integration

## Root Cause Investigation

### Current Hardcoded Implementation
```rust
fn is_special_token(&self, token: u32) -> bool {
    match self.model_variant {
        LlamaVariant::Llama2 => {
            matches!(token, 0..=2) // UNK, BOS, EOS
        }
        LlamaVariant::Llama3 => {
            matches!(token, 128000..=128002) // LLaMA-3 special tokens
        }
        LlamaVariant::CodeLlama => {
            matches!(token, 0..=2) // Similar to LLaMA-2
        }
    }
}
```

### Identified Problems

#### 1. **Inaccurate Special Token Detection**
- **LLaMA-2 Issue**: Assumes special tokens are only 0-2, but models may have different UNK/PAD token configurations
- **LLaMA-3 Issue**: Hardcoded range 128000-128002 doesn't account for model variations or extended special token sets
- **CodeLlama Issue**: Uses LLaMA-2 assumptions without considering code-specific special tokens

#### 2. **Missing UNK Token Support**
- Current Tokenizer trait lacks `unk_token_id()` method despite `TokenizerConfig` supporting it
- Inner tokenizers (GGUF, HF, SPM) provide UNK token information but wrapper ignores it
- Creates inconsistency between available token metadata and usage

#### 3. **Variant Detection Brittleness**
- Vocabulary size-based variant detection is fragile:
  - Custom LLaMA models may have different vocab sizes
  - Fine-tuned models often extend vocabularies
  - Quantized models may alter special token mappings

#### 4. **Limited Extensibility**
- Adding new LLaMA variants requires code changes
- Custom/fine-tuned models cannot specify their special tokens
- No mechanism for runtime special token discovery

#### 5. **Information Loss**
- Inner tokenizers provide rich special token metadata through trait methods
- Wrapper discards this information in favor of hardcoded assumptions
- Results in suboptimal token classification and processing

## Impact Assessment

### Severity: **High**
- **Token Classification Errors**: Incorrect special token detection affects text generation quality
- **Model Compatibility**: Prevents proper support for custom and fine-tuned LLaMA models
- **Information Loss**: Available tokenizer metadata ignored in favor of hardcoded values
- **Maintenance Burden**: Each new LLaMA variant requires code changes

### Affected Components
- **Tokenizer Strategy System**: All LLaMA tokenizer wrapper usage
- **Text Generation**: Special token handling during encoding/decoding
- **Model Loading**: Tokenizer compatibility validation
- **Cross-Validation**: Comparison with reference implementations

### User Impact
- **Incorrect Token Handling**: Special tokens may be incorrectly processed or filtered
- **Model Support Limitations**: Custom LLaMA models may not work correctly
- **Generation Quality**: Poor special token handling affects output quality
- **Development Friction**: Adding new model support requires code changes

## Technical Analysis

### Available Inner Tokenizer Information
The current Tokenizer trait provides special token accessors that should be leveraged:

```rust
pub trait Tokenizer: Send + Sync {
    // Current methods providing special token information
    fn bos_token_id(&self) -> Option<u32>;
    fn eos_token_id(&self) -> Option<u32>;
    fn pad_token_id(&self) -> Option<u32>;
    // Missing: fn unk_token_id(&self) -> Option<u32>;
}
```

### TokenizerConfig Support
The configuration system already supports UNK tokens:
```rust
pub struct TokenizerConfig {
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub unk_token_id: Option<u32>, // Available but not used in trait
    // ...
}
```

### GGUF Metadata Extraction
The universal tokenizer extracts UNK token information from GGUF metadata:
```rust
// From universal.rs
unk_token_id: reader.get_u32_metadata("tokenizer.ggml.unknown_token_id"),
```

## Proposed Solution

### Phase 1: Extend Tokenizer Trait with UNK Token Support

**A. Add Missing UNK Token Method**
```rust
// crates/bitnet-tokenizers/src/lib.rs
pub trait Tokenizer: Send + Sync {
    // Existing methods...
    fn bos_token_id(&self) -> Option<u32> { None }
    fn eos_token_id(&self) -> Option<u32> { None }
    fn pad_token_id(&self) -> Option<u32> { None }

    // New method for UNK token support
    fn unk_token_id(&self) -> Option<u32> { None }
}
```

**B. Update All Tokenizer Implementations**
```rust
// GGUF tokenizer
impl Tokenizer for GgufTokenizer {
    fn unk_token_id(&self) -> Option<u32> {
        self.unk_token_id
    }
}

// HF tokenizer
impl Tokenizer for HfTokenizer {
    fn unk_token_id(&self) -> Option<u32> {
        self.unk_id
    }
}

// SPM tokenizer
impl Tokenizer for SpmTokenizer {
    fn unk_token_id(&self) -> Option<u32> {
        self.unk_id
    }
}
```

### Phase 2: Dynamic Special Token Configuration

**A. Enhanced LlamaTokenizerWrapper Structure**
```rust
pub struct LlamaTokenizerWrapper {
    inner: Arc<dyn Tokenizer>,
    vocab_size: usize,
    model_variant: LlamaVariant,

    // Dynamic special token configuration
    special_tokens: SpecialTokenConfig,
}

#[derive(Debug, Clone)]
pub struct SpecialTokenConfig {
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,

    // Extended special tokens for specific variants
    pub additional_special_tokens: HashSet<u32>,
}
```

**B. Dynamic Special Token Detection**
```rust
impl LlamaTokenizerWrapper {
    pub fn new(inner: Arc<dyn Tokenizer>, vocab_size: usize) -> Result<Self> {
        let model_variant = Self::detect_variant(vocab_size);

        // Extract special tokens from inner tokenizer
        let special_tokens = SpecialTokenConfig {
            bos_token_id: inner.bos_token_id(),
            eos_token_id: inner.eos_token_id(),
            unk_token_id: inner.unk_token_id(),
            pad_token_id: inner.pad_token_id(),
            additional_special_tokens: Self::detect_additional_special_tokens(&model_variant, &inner),
        };

        debug!(
            "LLaMA tokenizer special tokens - BOS: {:?}, EOS: {:?}, UNK: {:?}, PAD: {:?}",
            special_tokens.bos_token_id,
            special_tokens.eos_token_id,
            special_tokens.unk_token_id,
            special_tokens.pad_token_id
        );

        Ok(Self { inner, vocab_size, model_variant, special_tokens })
    }

    fn is_special_token(&self, token: u32) -> bool {
        // Check primary special tokens
        if self.special_tokens.bos_token_id.map_or(false, |id| token == id) ||
           self.special_tokens.eos_token_id.map_or(false, |id| token == id) ||
           self.special_tokens.unk_token_id.map_or(false, |id| token == id) ||
           self.special_tokens.pad_token_id.map_or(false, |id| token == id) {
            return true;
        }

        // Check additional special tokens
        self.special_tokens.additional_special_tokens.contains(&token)
    }

    fn detect_additional_special_tokens(
        variant: &LlamaVariant,
        inner: &Arc<dyn Tokenizer>,
    ) -> HashSet<u32> {
        let mut additional = HashSet::new();

        match variant {
            LlamaVariant::Llama3 => {
                // LLaMA-3 has extended special tokens in 128000+ range
                // Detect them dynamically by checking token pieces
                for token_id in 128000..128010 {
                    if let Some(piece) = inner.token_to_piece(token_id) {
                        if piece.starts_with('<') && piece.ends_with('>') {
                            additional.insert(token_id);
                            debug!("Detected LLaMA-3 special token: {} -> '{}'", token_id, piece);
                        }
                    }
                }
            }
            LlamaVariant::CodeLlama => {
                // CodeLlama may have code-specific special tokens
                // Look for tokens with code-related patterns
                for token_id in 0..100 {
                    if let Some(piece) = inner.token_to_piece(token_id) {
                        if piece.contains("CODE") || piece.contains("FILL") {
                            additional.insert(token_id);
                            debug!("Detected CodeLlama special token: {} -> '{}'", token_id, piece);
                        }
                    }
                }
            }
            LlamaVariant::Llama2 => {
                // LLaMA-2 typically uses minimal special tokens
                // No additional detection needed beyond primary tokens
            }
        }

        additional
    }
}
```

### Phase 3: Fallback and Validation Strategy

**A. Robust Fallback Logic**
```rust
impl LlamaTokenizerWrapper {
    fn is_special_token_with_fallback(&self, token: u32) -> bool {
        // Primary: Use dynamically detected special tokens
        if self.is_special_token(token) {
            return true;
        }

        // Fallback: Use variant-specific heuristics if dynamic detection failed
        match self.model_variant {
            LlamaVariant::Llama2 => {
                // Conservative fallback for LLaMA-2
                token <= 2 || (token < 100 && self.token_looks_special(token))
            }
            LlamaVariant::Llama3 => {
                // LLaMA-3 has special tokens in high range
                (token >= 128000 && token <= 128256) ||
                (token <= 2) ||
                (token < 100 && self.token_looks_special(token))
            }
            LlamaVariant::CodeLlama => {
                // CodeLlama similar to LLaMA-2 with code extensions
                token <= 2 ||
                (token < 100 && self.token_looks_special(token)) ||
                self.token_looks_code_special(token)
            }
        }
    }

    fn token_looks_special(&self, token: u32) -> bool {
        if let Some(piece) = self.inner.token_to_piece(token) {
            // Heuristic: special tokens often have angle brackets or special patterns
            piece.starts_with('<') && piece.ends_with('>') ||
            piece.contains("SPECIAL") ||
            piece.len() == 0 // Empty piece indicates special token
        } else {
            false
        }
    }

    fn token_looks_code_special(&self, token: u32) -> bool {
        if let Some(piece) = self.inner.token_to_piece(token) {
            piece.contains("CODE") ||
            piece.contains("FILL") ||
            piece.contains("PREFIX") ||
            piece.contains("SUFFIX")
        } else {
            false
        }
    }
}
```

**B. Configuration Override Support**
```rust
impl LlamaTokenizerWrapper {
    /// Create with explicit special token configuration
    pub fn with_special_token_config(
        inner: Arc<dyn Tokenizer>,
        vocab_size: usize,
        special_tokens: SpecialTokenConfig,
    ) -> Result<Self> {
        let model_variant = Self::detect_variant(vocab_size);

        info!(
            "Creating LLaMA tokenizer with explicit special token config: {:?}",
            special_tokens
        );

        Ok(Self { inner, vocab_size, model_variant, special_tokens })
    }

    /// Update special token configuration at runtime
    pub fn update_special_tokens(&mut self, new_config: SpecialTokenConfig) {
        self.special_tokens = new_config;

        debug!("Updated LLaMA tokenizer special token configuration");
    }
}
```

### Phase 4: Enhanced Error Handling and Validation

**A. Configuration Validation**
```rust
impl SpecialTokenConfig {
    pub fn validate(&self, vocab_size: usize) -> Result<()> {
        // Validate token IDs are within vocabulary range
        if let Some(bos) = self.bos_token_id {
            if bos as usize >= vocab_size {
                return Err(BitNetError::Config(format!(
                    "BOS token ID {} exceeds vocabulary size {}",
                    bos, vocab_size
                )));
            }
        }

        if let Some(eos) = self.eos_token_id {
            if eos as usize >= vocab_size {
                return Err(BitNetError::Config(format!(
                    "EOS token ID {} exceeds vocabulary size {}",
                    eos, vocab_size
                )));
            }
        }

        if let Some(unk) = self.unk_token_id {
            if unk as usize >= vocab_size {
                return Err(BitNetError::Config(format!(
                    "UNK token ID {} exceeds vocabulary size {}",
                    unk, vocab_size
                )));
            }
        }

        if let Some(pad) = self.pad_token_id {
            if pad as usize >= vocab_size {
                return Err(BitNetError::Config(format!(
                    "PAD token ID {} exceeds vocabulary size {}",
                    pad, vocab_size
                )));
            }
        }

        // Validate additional special tokens
        for &token in &self.additional_special_tokens {
            if token as usize >= vocab_size {
                return Err(BitNetError::Config(format!(
                    "Additional special token ID {} exceeds vocabulary size {}",
                    token, vocab_size
                )));
            }
        }

        Ok(())
    }

    pub fn detect_conflicts(&self) -> Vec<String> {
        let mut conflicts = Vec::new();
        let mut used_tokens = HashSet::new();

        // Check for duplicate token IDs
        if let Some(bos) = self.bos_token_id {
            if !used_tokens.insert(bos) {
                conflicts.push(format!("BOS token ID {} conflicts with another special token", bos));
            }
        }

        if let Some(eos) = self.eos_token_id {
            if !used_tokens.insert(eos) {
                conflicts.push(format!("EOS token ID {} conflicts with another special token", eos));
            }
        }

        if let Some(unk) = self.unk_token_id {
            if !used_tokens.insert(unk) {
                conflicts.push(format!("UNK token ID {} conflicts with another special token", unk));
            }
        }

        if let Some(pad) = self.pad_token_id {
            if !used_tokens.insert(pad) {
                conflicts.push(format!("PAD token ID {} conflicts with another special token", pad));
            }
        }

        for &token in &self.additional_special_tokens {
            if !used_tokens.insert(token) {
                conflicts.push(format!("Additional special token ID {} conflicts with another special token", token));
            }
        }

        conflicts
    }
}
```

**B. Enhanced Diagnostic Information**
```rust
impl LlamaTokenizerWrapper {
    /// Get diagnostic information about special token configuration
    pub fn special_token_diagnostics(&self) -> SpecialTokenDiagnostics {
        SpecialTokenDiagnostics {
            variant: self.model_variant,
            vocab_size: self.vocab_size,
            special_tokens: self.special_tokens.clone(),
            detection_method: self.get_detection_method(),
            validation_status: self.special_tokens.validate(self.vocab_size),
            conflicts: self.special_tokens.detect_conflicts(),
            inner_tokenizer_info: self.get_inner_tokenizer_info(),
        }
    }

    fn get_detection_method(&self) -> DetectionMethod {
        if self.special_tokens.bos_token_id.is_some() ||
           self.special_tokens.eos_token_id.is_some() ||
           self.special_tokens.unk_token_id.is_some() ||
           self.special_tokens.pad_token_id.is_some() {
            DetectionMethod::DynamicFromInner
        } else {
            DetectionMethod::FallbackHeuristic
        }
    }

    fn get_inner_tokenizer_info(&self) -> InnerTokenizerInfo {
        InnerTokenizerInfo {
            bos_available: self.inner.bos_token_id().is_some(),
            eos_available: self.inner.eos_token_id().is_some(),
            unk_available: self.inner.unk_token_id().is_some(),
            pad_available: self.inner.pad_token_id().is_some(),
            vocab_size: self.inner.vocab_size(),
        }
    }
}

#[derive(Debug)]
pub struct SpecialTokenDiagnostics {
    pub variant: LlamaVariant,
    pub vocab_size: usize,
    pub special_tokens: SpecialTokenConfig,
    pub detection_method: DetectionMethod,
    pub validation_status: Result<()>,
    pub conflicts: Vec<String>,
    pub inner_tokenizer_info: InnerTokenizerInfo,
}

#[derive(Debug)]
pub enum DetectionMethod {
    DynamicFromInner,
    FallbackHeuristic,
    ExplicitConfiguration,
}

#[derive(Debug)]
pub struct InnerTokenizerInfo {
    pub bos_available: bool,
    pub eos_available: bool,
    pub unk_available: bool,
    pub pad_available: bool,
    pub vocab_size: usize,
}
```

## Implementation Plan

### Phase 1: Foundation (Week 1)
- [ ] **Add `unk_token_id()` method to Tokenizer trait**
  - Update trait definition with default implementation
  - Document new method in trait documentation
- [ ] **Update all tokenizer implementations**
  - Implement `unk_token_id()` in GgufTokenizer, HfTokenizer, SpmTokenizer
  - Ensure MockTokenizer and BasicTokenizer support new method
- [ ] **Add comprehensive unit tests**
  - Test UNK token access across all tokenizer types
  - Verify backward compatibility

### Phase 2: Core Implementation (Week 2)
- [ ] **Design SpecialTokenConfig structure**
  - Define configuration struct with validation
  - Implement conflict detection and diagnostics
- [ ] **Refactor LlamaTokenizerWrapper constructor**
  - Extract special tokens from inner tokenizer
  - Add dynamic special token detection
- [ ] **Implement dynamic is_special_token method**
  - Replace hardcoded logic with configuration-based detection
  - Add fallback heuristics for robustness

### Phase 3: Enhanced Features (Week 3)
- [ ] **Add explicit configuration override support**
  - Implement `with_special_token_config` constructor
  - Add runtime configuration update methods
- [ ] **Implement variant-specific detection heuristics**
  - Add LLaMA-3 extended special token detection
  - Add CodeLlama code-specific token detection
- [ ] **Add comprehensive validation and diagnostics**
  - Implement configuration validation
  - Add diagnostic information system

### Phase 4: Integration and Testing (Week 4)
- [ ] **Update tokenizer strategy resolver**
  - Ensure special token configuration propagates correctly
  - Add integration with discovery system
- [ ] **Comprehensive testing suite**
  - Unit tests for all special token detection scenarios
  - Integration tests with real LLaMA models
  - Cross-validation tests with reference implementations
- [ ] **Documentation and examples**
  - Update tokenizer architecture documentation
  - Add special token configuration examples

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_special_token_detection() {
        let base_tokenizer = Arc::new(BasicTokenizer::with_config(
            32000,
            Some(1),    // BOS
            Some(2),    // EOS
            Some(0),    // UNK
        ));

        let wrapper = LlamaTokenizerWrapper::new(base_tokenizer, 32000)
            .expect("Wrapper should initialize");

        // Test dynamic detection works
        assert!(wrapper.is_special_token(0), "UNK token should be detected");
        assert!(wrapper.is_special_token(1), "BOS token should be detected");
        assert!(wrapper.is_special_token(2), "EOS token should be detected");
        assert!(!wrapper.is_special_token(3), "Regular token should not be special");
    }

    #[test]
    fn test_llama3_extended_special_tokens() {
        let base_tokenizer = Arc::new(create_llama3_tokenizer_mock());
        let wrapper = LlamaTokenizerWrapper::new(base_tokenizer, 128256)
            .expect("LLaMA-3 wrapper should initialize");

        // Test LLaMA-3 specific special tokens
        assert!(wrapper.is_special_token(128000), "LLaMA-3 BOS should be detected");
        assert!(wrapper.is_special_token(128001), "LLaMA-3 EOS should be detected");
        assert!(wrapper.is_special_token(128002), "LLaMA-3 special should be detected");
    }

    #[test]
    fn test_explicit_configuration_override() {
        let base_tokenizer = Arc::new(BasicTokenizer::new());

        let special_tokens = SpecialTokenConfig {
            bos_token_id: Some(100),
            eos_token_id: Some(101),
            unk_token_id: Some(102),
            pad_token_id: Some(103),
            additional_special_tokens: [200, 201, 202].into_iter().collect(),
        };

        let wrapper = LlamaTokenizerWrapper::with_special_token_config(
            base_tokenizer,
            32000,
            special_tokens,
        ).expect("Explicit configuration should work");

        assert!(wrapper.is_special_token(100), "Custom BOS should be detected");
        assert!(wrapper.is_special_token(200), "Additional special token should be detected");
        assert!(!wrapper.is_special_token(1), "Default BOS should not be special");
    }

    #[test]
    fn test_special_token_validation() {
        let config = SpecialTokenConfig {
            bos_token_id: Some(50000), // Exceeds vocab size
            eos_token_id: Some(2),
            unk_token_id: Some(0),
            pad_token_id: Some(3),
            additional_special_tokens: HashSet::new(),
        };

        let validation_result = config.validate(32000);
        assert!(validation_result.is_err(), "Should fail validation for out-of-range token");
    }

    #[test]
    fn test_conflict_detection() {
        let config = SpecialTokenConfig {
            bos_token_id: Some(1),
            eos_token_id: Some(1), // Conflicts with BOS
            unk_token_id: Some(0),
            pad_token_id: Some(3),
            additional_special_tokens: [1].into_iter().collect(), // Also conflicts
        };

        let conflicts = config.detect_conflicts();
        assert!(!conflicts.is_empty(), "Should detect token ID conflicts");
        assert!(conflicts.len() >= 2, "Should detect multiple conflicts");
    }
}
```

### Integration Tests
```rust
#[cfg(test)]
mod integration_tests {
    #[test]
    #[ignore = "requires real model files"]
    fn test_real_llama2_model_special_tokens() {
        let model_path = std::env::var("LLAMA2_MODEL_PATH")
            .expect("Set LLAMA2_MODEL_PATH for integration test");

        let discovery = TokenizerDiscovery::from_gguf(&model_path)
            .expect("Should discover tokenizer from real model");

        let resolver = TokenizerStrategyResolver::new(discovery).await
            .expect("Should create resolver");

        let tokenizer = resolver.resolve_with_fallback().await
            .expect("Should resolve tokenizer");

        // Verify special token detection works with real model
        if let Some(bos) = tokenizer.bos_token_id() {
            // Test actual special token detection
        }
    }

    #[test]
    #[ignore = "requires real model files"]
    fn test_real_llama3_model_special_tokens() {
        // Similar test for LLaMA-3 with extended special tokens
    }

    #[test]
    fn test_cross_validation_special_tokens() {
        // Cross-validate special token detection with reference implementation
    }
}
```

### Property-Based Tests
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_special_token_consistency(
        vocab_size in 1000..200000_usize,
        bos_id in prop::option::of(0..1000_u32),
        eos_id in prop::option::of(0..1000_u32),
        unk_id in prop::option::of(0..1000_u32),
    ) {
        let base_tokenizer = Arc::new(BasicTokenizer::with_config(
            vocab_size,
            bos_id,
            eos_id,
            unk_id,
        ));

        let wrapper = LlamaTokenizerWrapper::new(base_tokenizer, vocab_size);

        if let Ok(wrapper) = wrapper {
            // Property: special token detection should be consistent
            if let Some(bos) = bos_id {
                prop_assert!(wrapper.is_special_token(bos));
            }
            if let Some(eos) = eos_id {
                prop_assert!(wrapper.is_special_token(eos));
            }
            if let Some(unk) = unk_id {
                prop_assert!(wrapper.is_special_token(unk));
            }
        }
    }
}
```

## Acceptance Criteria

### ✅ Core Functionality
- [ ] **Dynamic Special Token Detection**: `is_special_token` uses inner tokenizer information instead of hardcoded ranges
- [ ] **UNK Token Support**: Tokenizer trait includes `unk_token_id()` method with full implementation coverage
- [ ] **Variant Compatibility**: All LLaMA variants (LLaMA-2, LLaMA-3, CodeLlama) work correctly with dynamic detection
- [ ] **Backward Compatibility**: Existing tokenizer usage continues to work without changes

### ✅ Enhanced Features
- [ ] **Configuration Override**: Support for explicit special token configuration via `with_special_token_config`
- [ ] **Runtime Updates**: Ability to update special token configuration after wrapper creation
- [ ] **Extended Token Detection**: Automatic detection of variant-specific special tokens (LLaMA-3 extended range, CodeLlama code tokens)
- [ ] **Fallback Robustness**: Graceful handling when inner tokenizer lacks special token information

### ✅ Quality Assurance
- [ ] **Input Validation**: Comprehensive validation of special token configurations with helpful error messages
- [ ] **Conflict Detection**: Detection and reporting of conflicting special token ID assignments
- [ ] **Diagnostic Information**: Rich diagnostic API for troubleshooting special token issues
- [ ] **Cross-Validation**: Special token detection matches reference implementations

### ✅ Testing Coverage
- [ ] **Unit Test Coverage**: >95% line coverage for special token detection logic
- [ ] **Integration Tests**: Tests with real LLaMA model files verify correct behavior
- [ ] **Property-Based Tests**: Fuzz testing ensures robustness across input variations
- [ ] **Performance Tests**: Special token detection performance remains optimal

### ✅ Documentation
- [ ] **API Documentation**: Complete documentation of new methods and configuration options
- [ ] **Architecture Updates**: Tokenizer architecture guide reflects dynamic special token system
- [ ] **Migration Guide**: Clear guidance for users with custom tokenizer configurations
- [ ] **Troubleshooting**: Diagnostic examples for common special token issues

## Breaking Changes and Migration

### API Changes
- **New Method**: `Tokenizer::unk_token_id()` - purely additive, no breaking changes
- **Enhanced Constructor**: `LlamaTokenizerWrapper::new()` behavior changes but maintains compatibility
- **New Constructors**: `with_special_token_config()` and update methods are purely additive

### Migration Path
1. **Immediate**: All existing code continues to work with improved special token detection
2. **Optional Enhancement**: Users can opt into explicit special token configuration
3. **Future Optimization**: Custom models can provide special token metadata for optimal detection

### Configuration Migration
```rust
// Before: Relied on hardcoded detection
let wrapper = LlamaTokenizerWrapper::new(tokenizer, vocab_size)?;

// After: Same API, improved detection (no code changes needed)
let wrapper = LlamaTokenizerWrapper::new(tokenizer, vocab_size)?;

// New: Explicit configuration for custom models
let special_tokens = SpecialTokenConfig {
    bos_token_id: Some(1),
    eos_token_id: Some(2),
    unk_token_id: Some(0),
    pad_token_id: Some(3),
    additional_special_tokens: custom_tokens,
};
let wrapper = LlamaTokenizerWrapper::with_special_token_config(
    tokenizer, vocab_size, special_tokens
)?;
```

## Performance Considerations

### Computational Overhead
- **Special Token Detection**: O(1) lookup in HashSet vs O(1) range checks - comparable performance
- **Initialization**: Slight overhead for special token discovery, offset by accuracy gains
- **Memory Usage**: Minimal increase for storing special token configuration

### Optimization Strategies
- **Lazy Detection**: Additional special tokens detected only when needed
- **Caching**: Special token sets cached for repeated lookups
- **Branch Prediction**: Common special tokens checked first for optimal performance

## Cross-References to Related Issues

### Direct Dependencies
- **Issue #249**: Tokenizer Discovery System - Dynamic configuration integrates with discovery
- **Configuration System Issues**: Special token configuration aligns with broader config improvements
- **Universal Tokenizer Issues**: Enhanced trait methods benefit all tokenizer implementations

### Related Enhancements
- **Tokenizer Strategy Resolution**: Improved special token handling enhances strategy effectiveness
- **Model Loading**: Better special token detection improves model compatibility validation
- **Cross-Validation**: Accurate special token handling essential for reference comparison

### Architecture Alignment
- **ADR-005**: Tokenizer Discovery System architectural decision alignment
- **Neural Network Integration**: Special token handling critical for text generation quality
- **Production Readiness**: Robust special token detection essential for production deployment

## Labels
- `tokenizer`
- `enhancement`
- `compatibility`
- `priority-high`
- `api-improvement`
- `llama-models`
- `special-tokens`
- `dynamic-configuration`

## Related Issues and PRs
- **Issue #249**: Tokenizer Discovery Neural Network Integration
- **Hardcoded Values Configuration Issue**: Broader configuration system improvements
- **Universal Tokenizer Enhancement**: Trait method additions benefit entire system
- **Cross-Validation Improvements**: Accurate special token handling essential for validation

## Benefits After Implementation

### ✅ Immediate Benefits
- **Accurate Token Detection**: Special tokens correctly identified for all LLaMA variants
- **Model Compatibility**: Custom and fine-tuned LLaMA models work correctly
- **Information Utilization**: Available tokenizer metadata fully leveraged
- **Maintenance Reduction**: New LLaMA variants require no code changes

### ✅ Long-term Benefits
- **Extensibility**: Easy addition of new special token types and detection methods
- **Robustness**: Graceful handling of edge cases and unusual configurations
- **Debuggability**: Rich diagnostic information simplifies troubleshooting
- **Performance**: Optimized special token detection maintains system performance

This comprehensive solution transforms hardcoded special token detection into a flexible, robust, and maintainable dynamic configuration system that improves accuracy while maintaining performance and compatibility.