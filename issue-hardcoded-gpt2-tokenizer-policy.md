# [Configuration] Replace hardcoded GPT-2 tokenizer policy with extensible model-aware configuration system

## Problem Description

The current inference loader contains a hardcoded check for GPT-2 models that directly impacts scoring policy determination, making the system inflexible for supporting diverse model architectures with different tokenization requirements.

**Location**: `crates/bitnet-inference/src/loader.rs:255-257`

```rust
// Model-specific overrides
if config.model_type.contains("gpt2") {
    policy.add_bos = false;  // GPT-2 doesn't use BOS
}
```

**Critical Issues:**
1. **Architectural Inflexibility**: String matching against model types creates brittle dependencies
2. **Maintenance Burden**: Each new model architecture requires code changes to scoring policy logic
3. **Configuration Inconsistency**: Tokenization behavior is hardcoded rather than model-configurable
4. **Cross-Validation Risk**: Hardcoded policies may not match reference implementations for new architectures
5. **Scale Challenge**: Current approach doesn't scale for the growing ecosystem of transformer architectures

## Root Cause Analysis

### Current Architecture Problems
1. **Tight Coupling**: Scoring policy determination is directly embedded in loader logic
2. **Limited Extensibility**: No mechanism to specify tokenizer policies per model type
3. **Missing Abstraction**: No configuration layer between model metadata and inference behavior
4. **Validation Gap**: Scoring policies aren't validated against model-specific requirements

### Affected Components
- **Primary**: `bitnet-inference/src/loader.rs` (scoring policy determination)
- **Secondary**: `bitnet-common/src/config.rs` (model configuration)
- **Tertiary**: Model loading pipeline and tokenizer integration

## Technical Impact Assessment

### Immediate Impact
- **Scope**: All model loading with non-standard tokenization requirements
- **Severity**: Medium - affects inference quality for specific model types
- **Users Affected**: Developers using GPT-2 models and future model architectures

### Future Scaling Issues
- Each new model architecture (Mistral, Qwen, Claude, etc.) requires code modifications
- Risk of inconsistent tokenization behavior across model families
- Difficulty maintaining parity with reference implementations

## Proposed Solution

### Phase 1: Configuration-Driven Tokenizer Policies

Replace hardcoded checks with a flexible configuration system that maps model architectures to tokenizer policies.

#### 1.1 Extend ModelConfig with Tokenizer Policies

```rust
// crates/bitnet-common/src/config.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerPolicy {
    pub add_bos_token: bool,
    pub add_eos_token: bool,
    pub mask_pad_tokens: bool,
    pub special_tokens: HashMap<String, u32>,
}

impl Default for TokenizerPolicy {
    fn default() -> Self {
        Self {
            add_bos_token: true,     // Most models use BOS
            add_eos_token: false,    // EOS typically not added for evaluation
            mask_pad_tokens: true,   // Standard for most architectures
            special_tokens: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // ... existing fields ...
    pub model_type: Option<String>,
    pub tokenizer_policy: TokenizerPolicy,
}
```

#### 1.2 Model Architecture Registry

```rust
// crates/bitnet-common/src/model_registry.rs

pub struct ModelArchitectureRegistry {
    policies: HashMap<String, TokenizerPolicy>,
}

impl ModelArchitectureRegistry {
    pub fn new() -> Self {
        let mut policies = HashMap::new();

        // GPT-2 family
        policies.insert("gpt2".to_string(), TokenizerPolicy {
            add_bos_token: false,  // GPT-2 doesn't use BOS
            add_eos_token: false,
            mask_pad_tokens: true,
            special_tokens: HashMap::new(),
        });

        // LLaMA family
        policies.insert("llama".to_string(), TokenizerPolicy {
            add_bos_token: true,
            add_eos_token: false,
            mask_pad_tokens: true,
            special_tokens: HashMap::new(),
        });

        // Mistral family
        policies.insert("mistral".to_string(), TokenizerPolicy {
            add_bos_token: true,
            add_eos_token: false,
            mask_pad_tokens: true,
            special_tokens: HashMap::new(),
        });

        Self { policies }
    }

    pub fn get_policy(&self, model_type: &str) -> TokenizerPolicy {
        // Try exact match first
        if let Some(policy) = self.policies.get(model_type) {
            return policy.clone();
        }

        // Try prefix matching for model families
        for (registered_type, policy) in &self.policies {
            if model_type.starts_with(registered_type) ||
               model_type.contains(registered_type) {
                return policy.clone();
            }
        }

        // Return default policy
        TokenizerPolicy::default()
    }
}
```

#### 1.3 Updated Loader Implementation

```rust
// crates/bitnet-inference/src/loader.rs

impl ModelLoader {
    fn determine_scoring_policy(&self, model: &Arc<dyn Model>, tokenizer: &Arc<dyn Tokenizer>) -> ScoringPolicy {
        let config = model.config();
        let registry = ModelArchitectureRegistry::new();

        // Determine model type from multiple sources
        let model_type = self.determine_model_type(config, model);

        // Get architecture-specific tokenizer policy
        let tokenizer_policy = if let Some(explicit_policy) = &config.model.tokenizer_policy {
            // Use explicit configuration if provided
            explicit_policy.clone()
        } else {
            // Fallback to registry-based lookup
            registry.get_policy(&model_type)
        };

        // Create scoring policy from tokenizer policy
        let mut policy = ScoringPolicy {
            add_bos: tokenizer_policy.add_bos_token,
            append_eos: tokenizer_policy.add_eos_token,
            mask_pad: tokenizer_policy.mask_pad_tokens,
        };

        // Validate against tokenizer capabilities
        if policy.add_bos && tokenizer.bos_token_id().is_none() {
            warn!("Model policy requires BOS token, but tokenizer doesn't provide one");
            policy.add_bos = false;
        }

        if policy.append_eos && tokenizer.eos_token_id().is_none() {
            warn!("Model policy requires EOS token, but tokenizer doesn't provide one");
            policy.append_eos = false;
        }

        debug!("Determined scoring policy for {}: {:?}", model_type, policy);
        policy
    }

    fn determine_model_type(&self, config: &BitNetConfig, model: &Arc<dyn Model>) -> String {
        // Priority order for model type determination:
        // 1. Explicit configuration
        if let Some(model_type) = &config.model.model_type {
            return model_type.clone();
        }

        // 2. Model metadata (from GGUF, SafeTensors, etc.)
        if let Some(metadata) = model.metadata() {
            if let Some(arch) = metadata.get("general.architecture") {
                return arch.clone();
            }
        }

        // 3. Inference from model structure/tensors
        // This could analyze tensor names, dimensions, etc.

        // 4. Default fallback
        "unknown".to_string()
    }
}
```

### Phase 2: Runtime Configuration Support

#### 2.1 Environment Variable Overrides

```rust
// Support runtime configuration via environment variables
if let Ok(policy_override) = env::var("BITNET_TOKENIZER_POLICY") {
    let override_policy: TokenizerPolicy = serde_json::from_str(&policy_override)
        .context("Failed to parse BITNET_TOKENIZER_POLICY")?;
    config.model.tokenizer_policy = Some(override_policy);
}
```

#### 2.2 Configuration File Support

```toml
# bitnet.toml
[model]
model_type = "custom-gpt2-variant"

[model.tokenizer_policy]
add_bos_token = false
add_eos_token = false
mask_pad_tokens = true

[model.tokenizer_policy.special_tokens]
"<custom_token>" = 50256
```

### Phase 3: Validation and Testing Framework

#### 3.1 Policy Validation

```rust
pub struct TokenizerPolicyValidator;

impl TokenizerPolicyValidator {
    pub fn validate_policy(
        policy: &TokenizerPolicy,
        tokenizer: &dyn Tokenizer,
        model_type: &str,
    ) -> Result<Vec<ValidationWarning>> {
        let mut warnings = Vec::new();

        // Check BOS token consistency
        if policy.add_bos_token && tokenizer.bos_token_id().is_none() {
            warnings.push(ValidationWarning::MissingBosToken);
        }

        // Check EOS token consistency
        if policy.add_eos_token && tokenizer.eos_token_id().is_none() {
            warnings.push(ValidationWarning::MissingEosToken);
        }

        // Model-specific validation
        match model_type {
            t if t.contains("gpt2") => {
                if policy.add_bos_token {
                    warnings.push(ValidationWarning::UnexpectedBosForGPT2);
                }
            }
            _ => {}
        }

        Ok(warnings)
    }
}
```

#### 3.2 Cross-Validation Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_tokenizer_policy() {
        let registry = ModelArchitectureRegistry::new();
        let policy = registry.get_policy("gpt2");

        assert!(!policy.add_bos_token, "GPT-2 should not use BOS tokens");
        assert!(!policy.add_eos_token, "GPT-2 should not add EOS for evaluation");
        assert!(policy.mask_pad_tokens, "GPT-2 should mask padding tokens");
    }

    #[test]
    fn test_llama_tokenizer_policy() {
        let registry = ModelArchitectureRegistry::new();
        let policy = registry.get_policy("llama");

        assert!(policy.add_bos_token, "LLaMA should use BOS tokens");
        assert!(!policy.add_eos_token, "LLaMA should not add EOS for evaluation");
    }

    #[test]
    fn test_custom_model_fallback() {
        let registry = ModelArchitectureRegistry::new();
        let policy = registry.get_policy("custom-unknown-model");

        // Should return default policy
        assert_eq!(policy.add_bos_token, TokenizerPolicy::default().add_bos_token);
    }
}
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)
1. **Model Configuration Extension**
   - [ ] Add `TokenizerPolicy` struct to `bitnet-common`
   - [ ] Extend `ModelConfig` with tokenizer policy field
   - [ ] Update configuration serialization/deserialization
   - [ ] Add configuration validation

2. **Architecture Registry**
   - [ ] Create `ModelArchitectureRegistry` in `bitnet-common`
   - [ ] Implement policy lookup with fallback logic
   - [ ] Add support for model family matching
   - [ ] Create comprehensive policy database

3. **Loader Refactoring**
   - [ ] Replace hardcoded GPT-2 check with registry lookup
   - [ ] Implement model type determination logic
   - [ ] Add policy validation and warning system
   - [ ] Ensure backward compatibility

### Phase 2: Configuration Support (Week 3)
4. **Runtime Configuration**
   - [ ] Add environment variable support for policy overrides
   - [ ] Implement configuration file loading for tokenizer policies
   - [ ] Add CLI flags for policy specification
   - [ ] Create configuration validation tools

5. **Documentation and Examples**
   - [ ] Update model loading documentation
   - [ ] Create configuration examples for common architectures
   - [ ] Add troubleshooting guide for tokenizer issues
   - [ ] Document migration from hardcoded approach

### Phase 3: Validation and Testing (Week 4)
6. **Testing Infrastructure**
   - [ ] Create comprehensive test suite for tokenizer policies
   - [ ] Add cross-validation tests against reference implementations
   - [ ] Implement property-based testing for policy combinations
   - [ ] Add integration tests with real model files

7. **Quality Assurance**
   - [ ] Performance benchmarking of new configuration system
   - [ ] Memory usage analysis for registry system
   - [ ] Error handling and recovery testing
   - [ ] Documentation review and updates

## Acceptance Criteria

### Core Functionality
- [ ] **AC1**: Remove hardcoded `config.model_type.contains("gpt2")` check from loader
- [ ] **AC2**: Implement `TokenizerPolicy` configuration struct with serialization support
- [ ] **AC3**: Create `ModelArchitectureRegistry` with policy lookup for major architectures
- [ ] **AC4**: Support explicit tokenizer policy specification in model configuration
- [ ] **AC5**: Implement fallback behavior for unknown model architectures

### Configuration Support
- [ ] **AC6**: Support tokenizer policy specification via configuration files (TOML/JSON)
- [ ] **AC7**: Enable runtime policy overrides via environment variables
- [ ] **AC8**: Provide CLI options for tokenizer policy specification
- [ ] **AC9**: Validate tokenizer policies against tokenizer capabilities

### Backward Compatibility
- [ ] **AC10**: Maintain existing behavior for GPT-2 models without configuration changes
- [ ] **AC11**: Ensure all existing tests pass without modification
- [ ] **AC12**: Preserve current default behavior for unspecified model types

### Quality Assurance
- [ ] **AC13**: Achieve 100% test coverage for new tokenizer policy logic
- [ ] **AC14**: Add integration tests for all supported model architectures
- [ ] **AC15**: Performance impact < 5% for model loading operations
- [ ] **AC16**: Memory overhead < 1MB for registry system

### Documentation
- [ ] **AC17**: Update API documentation for new configuration options
- [ ] **AC18**: Create migration guide from hardcoded to configuration-based approach
- [ ] **AC19**: Document tokenizer policy specification for each supported architecture
- [ ] **AC20**: Provide troubleshooting guide for tokenizer configuration issues

## Testing Strategy

### Unit Tests
- Model architecture registry lookup logic
- Tokenizer policy validation
- Configuration serialization/deserialization
- Fallback behavior for unknown architectures

### Integration Tests
- End-to-end model loading with various tokenizer policies
- Configuration file parsing and application
- Environment variable override behavior
- Cross-validation with reference implementations

### Property-Based Tests
- Policy combinations produce valid scoring configurations
- Registry lookups are deterministic and consistent
- Configuration merging preserves semantic correctness

### Performance Tests
- Model loading latency with new configuration system
- Memory usage of architecture registry
- Policy lookup performance for large model catalogs

## Related Issues and Components

### Cross-References
- **Related to**: Model configuration modernization initiatives
- **Blocks**: Support for Mistral, Qwen, and other transformer architectures
- **Enables**: Automated model compatibility testing framework
- **Dependencies**: Configuration system validation enhancements

### Affected Documentation
- `/docs/development/model-configuration.md`
- `/docs/reference/tokenizer-architecture.md`
- `/docs/troubleshooting/model-loading.md`
- `/docs/examples/custom-model-integration.md`

## Labels and Priority

**Labels**: `enhancement`, `configuration`, `tokenizer`, `model-loading`, `breaking-change`
**Priority**: `high`
**Effort**: `medium`
**Complexity**: `medium`

**Milestone**: Configuration Modernization v1.2
**Team**: `inference`, `models`
**Reviewer**: Architecture team, model integration specialists

---

**Note**: This implementation will significantly improve BitNet.rs's ability to support diverse transformer architectures while maintaining backward compatibility and providing clear migration paths for existing users.
