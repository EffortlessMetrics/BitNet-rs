# [HARDCODED] GPT-2 model type detection uses string matching instead of proper model configuration

## Problem Description

The model loader uses hardcoded string matching (`model.model_type.contains("gpt2")`) to determine BOS token behavior, creating fragile model detection that may fail with model variants or similar architectures.

## Environment

**File**: `crates/bitnet-inference/src/loader.rs`
**Component**: Model Loading and Configuration
**Issue Type**: Hardcoded Values / Fragile Model Detection

## Root Cause Analysis

**Current Implementation:**
```rust
// Model-specific overrides
if config.model_type.contains("gpt2") {
    policy.add_bos = false;  // GPT-2 doesn't use BOS
}
```

**Analysis:**
1. **String Matching Fragility**: Uses substring matching which may match unintended models
2. **Limited Extensibility**: Adding new model types requires code changes
3. **Maintenance Burden**: Model-specific logic scattered throughout codebase
4. **Configuration Inconsistency**: BOS token behavior not properly configurable

## Impact Assessment

**Severity**: Medium
**Affected Areas**:
- Model compatibility and detection
- Token scoring and generation accuracy
- Support for new model architectures
- Configuration management

**Compatibility Impact**:
- May incorrectly identify non-GPT-2 models containing "gpt2" in their name
- Missing support for GPT-2 variants with different naming conventions
- Inconsistent behavior across different model formats

**Maintenance Impact**:
- Requires code changes for each new model architecture
- Hardcoded logic scattered across multiple files
- Difficult to test and validate model-specific behaviors

## Proposed Solution

### Configurable Model Behavior System

```rust
// Enhanced ModelConfig with comprehensive behavior configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,

    // Token behavior configuration
    pub tokenizer_config: TokenizerConfig,

    // Model architecture configuration
    pub architecture_config: ArchitectureConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Whether to add BOS token at the beginning of sequences
    pub add_bos_token: bool,

    /// Whether to add EOS token at the end of sequences
    pub add_eos_token: bool,

    /// Whether to use special tokens in scoring
    pub use_special_tokens_in_scoring: bool,

    /// Maximum sequence length supported
    pub max_sequence_length: usize,

    /// Padding strategy
    pub padding_strategy: PaddingStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    /// Model family (GPT, BERT, T5, etc.)
    pub model_family: ModelFamily,

    /// Attention mechanism type
    pub attention_type: AttentionType,

    /// Position encoding type
    pub position_encoding: PositionEncoding,

    /// Whether model supports bidirectional attention
    pub bidirectional: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelFamily {
    GPT,
    GPT2,
    BERT,
    T5,
    LLaMA,
    BitNet,
    Custom(u32), // For custom architectures
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PaddingStrategy {
    Left,
    Right,
    None,
}

impl ModelConfig {
    /// Create configuration for well-known model architectures
    pub fn for_model_family(family: ModelFamily) -> Self {
        match family {
            ModelFamily::GPT | ModelFamily::GPT2 => Self {
                model_type: "gpt2".to_string(),
                tokenizer_config: TokenizerConfig {
                    add_bos_token: false,  // GPT-2 doesn't use BOS
                    add_eos_token: true,
                    use_special_tokens_in_scoring: false,
                    max_sequence_length: 1024,
                    padding_strategy: PaddingStrategy::Left,
                },
                architecture_config: ArchitectureConfig {
                    model_family: family,
                    attention_type: AttentionType::CausalSelfAttention,
                    position_encoding: PositionEncoding::Learned,
                    bidirectional: false,
                },
                // ... other default values
            },
            ModelFamily::BERT => Self {
                tokenizer_config: TokenizerConfig {
                    add_bos_token: true,   // BERT uses [CLS] as BOS
                    add_eos_token: true,   // BERT uses [SEP] as EOS
                    use_special_tokens_in_scoring: true,
                    max_sequence_length: 512,
                    padding_strategy: PaddingStrategy::Right,
                },
                architecture_config: ArchitectureConfig {
                    model_family: family,
                    attention_type: AttentionType::BidirectionalSelfAttention,
                    position_encoding: PositionEncoding::Learned,
                    bidirectional: true,
                },
                // ... other values
            },
            ModelFamily::BitNet => Self {
                tokenizer_config: TokenizerConfig {
                    add_bos_token: true,
                    add_eos_token: true,
                    use_special_tokens_in_scoring: true,
                    max_sequence_length: 2048,
                    padding_strategy: PaddingStrategy::Left,
                },
                architecture_config: ArchitectureConfig {
                    model_family: family,
                    attention_type: AttentionType::CausalSelfAttention,
                    position_encoding: PositionEncoding::RoPE,
                    bidirectional: false,
                },
                // ... other values
            },
            _ => Self::default(),
        }
    }

    /// Detect model family from model metadata
    pub fn detect_model_family(model_type: &str, metadata: &ModelMetadata) -> ModelFamily {
        let model_type_lower = model_type.to_lowercase();

        // Check for exact matches first
        if model_type_lower == "gpt2" || model_type_lower == "gpt-2" {
            return ModelFamily::GPT2;
        }

        // Check for family patterns
        if model_type_lower.starts_with("gpt") {
            return ModelFamily::GPT;
        }

        if model_type_lower.contains("bert") {
            return ModelFamily::BERT;
        }

        if model_type_lower.contains("llama") {
            return ModelFamily::LLaMA;
        }

        if model_type_lower.contains("bitnet") {
            return ModelFamily::BitNet;
        }

        // Check metadata for additional clues
        if let Some(arch_hint) = metadata.architecture_hint() {
            return Self::family_from_architecture_hint(arch_hint);
        }

        // Default to custom
        ModelFamily::Custom(0)
    }
}

// Updated loader implementation
impl ModelLoader {
    fn determine_scoring_policy(
        &self,
        model: &Arc<dyn Model>,
        tokenizer: &Arc<dyn Tokenizer>
    ) -> ScoringPolicy {
        let config = model.config();
        let mut policy = ScoringPolicy::default();

        // Use configuration-driven approach instead of hardcoded string matching
        policy.add_bos = config.tokenizer_config.add_bos_token;
        policy.add_eos = config.tokenizer_config.add_eos_token;
        policy.use_special_tokens = config.tokenizer_config.use_special_tokens_in_scoring;
        policy.max_length = config.tokenizer_config.max_sequence_length;

        // Architecture-specific adjustments
        match config.architecture_config.model_family {
            ModelFamily::GPT | ModelFamily::GPT2 => {
                policy.causal_mask = true;
                policy.position_offset = 0;
            },
            ModelFamily::BERT => {
                policy.causal_mask = false;
                policy.bidirectional = true;
            },
            ModelFamily::BitNet => {
                policy.quantized_attention = true;
                policy.use_kv_cache = true;
            },
            _ => {
                // Use safe defaults
                warn!("Unknown model family, using conservative defaults");
            }
        }

        // Validate policy consistency
        self.validate_scoring_policy(&policy, config)?;

        policy
    }

    fn validate_scoring_policy(
        &self,
        policy: &ScoringPolicy,
        config: &ModelConfig
    ) -> Result<()> {
        // Check for configuration conflicts
        if policy.bidirectional && policy.causal_mask {
            return Err(anyhow::anyhow!(
                "Configuration conflict: bidirectional attention incompatible with causal masking"
            ));
        }

        if policy.max_length > config.tokenizer_config.max_sequence_length {
            warn!(
                "Policy max length {} exceeds model capability {}",
                policy.max_length, config.tokenizer_config.max_sequence_length
            );
        }

        Ok(())
    }
}
```

## Implementation Plan

### Task 1: Model Configuration System
- [ ] Implement `TokenizerConfig` and `ArchitectureConfig` structures
- [ ] Add `ModelFamily` enum with support for major architectures
- [ ] Create configuration presets for common model families
- [ ] Add model family detection from metadata

### Task 2: Configuration-Driven Loading
- [ ] Replace hardcoded string matching with configuration lookup
- [ ] Implement `determine_scoring_policy` using model configuration
- [ ] Add validation for configuration consistency
- [ ] Support custom model configurations

### Task 3: Metadata Integration
- [ ] Enhance model metadata parsing to extract architecture hints
- [ ] Add support for loading configuration from model files
- [ ] Implement fallback detection for unknown model types
- [ ] Add configuration override mechanisms

### Task 4: Testing and Validation
- [ ] Add comprehensive tests for model family detection
- [ ] Test configuration-driven policy determination
- [ ] Validate behavior across different model architectures
- [ ] Add regression tests for existing model support

## Testing Strategy

### Model Detection Tests
```rust
#[test]
fn test_model_family_detection() {
    // Test exact matches
    assert_eq!(
        ModelConfig::detect_model_family("gpt2", &ModelMetadata::default()),
        ModelFamily::GPT2
    );

    // Test case insensitive matching
    assert_eq!(
        ModelConfig::detect_model_family("GPT-2", &ModelMetadata::default()),
        ModelFamily::GPT2
    );

    // Test family patterns
    assert_eq!(
        ModelConfig::detect_model_family("gpt-neo-1.3B", &ModelMetadata::default()),
        ModelFamily::GPT
    );

    // Test BitNet models
    assert_eq!(
        ModelConfig::detect_model_family("bitnet-1.58b", &ModelMetadata::default()),
        ModelFamily::BitNet
    );
}

#[test]
fn test_scoring_policy_generation() {
    let loader = ModelLoader::new();

    // Test GPT-2 policy
    let gpt2_config = ModelConfig::for_model_family(ModelFamily::GPT2);
    let mock_model = create_mock_model(gpt2_config);
    let policy = loader.determine_scoring_policy(&mock_model, &mock_tokenizer);

    assert!(!policy.add_bos);  // GPT-2 doesn't use BOS
    assert!(policy.add_eos);
    assert!(policy.causal_mask);

    // Test BERT policy
    let bert_config = ModelConfig::for_model_family(ModelFamily::BERT);
    let mock_model = create_mock_model(bert_config);
    let policy = loader.determine_scoring_policy(&mock_model, &mock_tokenizer);

    assert!(policy.add_bos);   // BERT uses [CLS]
    assert!(policy.add_eos);   // BERT uses [SEP]
    assert!(policy.bidirectional);
}
```

### Configuration Validation Tests
```rust
#[test]
fn test_configuration_validation() {
    let loader = ModelLoader::new();

    // Test valid configuration
    let valid_config = ModelConfig::for_model_family(ModelFamily::GPT2);
    let policy = ScoringPolicy::from_config(&valid_config);
    assert!(loader.validate_scoring_policy(&policy, &valid_config).is_ok());

    // Test conflicting configuration
    let mut invalid_config = ModelConfig::for_model_family(ModelFamily::BERT);
    invalid_config.architecture_config.bidirectional = true;
    let mut policy = ScoringPolicy::from_config(&invalid_config);
    policy.causal_mask = true; // Conflict!

    assert!(loader.validate_scoring_policy(&policy, &invalid_config).is_err());
}
```

## Related Issues/PRs

- Part of comprehensive model configuration system
- Related to tokenizer and model compatibility improvements
- Connected to extensible architecture support

## Acceptance Criteria

- [ ] GPT-2 model detection uses configuration instead of string matching
- [ ] Model family detection supports common architectures (GPT, BERT, LLaMA, BitNet)
- [ ] Configuration system is extensible for new model types
- [ ] BOS/EOS token behavior is properly configurable per model family
- [ ] Validation prevents configuration conflicts
- [ ] All existing model support continues to work
- [ ] Performance is not degraded by configuration lookup

## Risk Assessment

**Low-Medium Risk**: Configuration-driven approach is more robust but requires careful migration of existing logic.

**Mitigation Strategies**:
- Maintain backwards compatibility during migration
- Add comprehensive testing for all supported model types
- Implement graceful fallback for unknown model families
- Provide clear error messages for configuration issues
- Document configuration options clearly for users