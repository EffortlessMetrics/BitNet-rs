# [HARDCODED] LLaMA tokenizer uses hardcoded special token handling instead of configurable approach

## Problem Description

The LLaMA tokenizer implementation uses hardcoded special token logic for `is_special_token` checks, making it inflexible for different LLaMA variants and preventing proper customization for deployment scenarios.

## Environment

**File**: LLaMA Tokenizer Implementation
**Component**: Special Token Processing
**Issue Type**: Hardcoded Values / Inflexible Configuration

## Root Cause Analysis

**Current Implementation:**
```rust
fn is_special_token(&self, token: &str) -> bool {
    // Hardcoded LLaMA special tokens
    matches!(token, "<bos>" | "<eos>" | "<unk>" | "<pad>")
}
```

**Analysis:**
1. **Fixed Token Set**: Only handles standard LLaMA special tokens
2. **No Variant Support**: Different LLaMA models may use different special tokens
3. **Configuration Rigidity**: Cannot adapt to custom or fine-tuned models
4. **Missing Extensibility**: No mechanism to add new special tokens

## Impact Assessment

**Severity**: Medium
**Affected Areas**:
- Multi-model LLaMA support
- Custom model integration
- Special token processing accuracy
- Deployment flexibility

## Proposed Solution

### Configurable Special Token System

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokenConfig {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
    pub pad_token: Option<String>,
    pub mask_token: Option<String>,
    pub additional_special_tokens: Vec<String>,
}

impl SpecialTokenConfig {
    pub fn llama_v1() -> Self {
        Self {
            bos_token: Some("<s>".to_string()),
            eos_token: Some("</s>".to_string()),
            unk_token: Some("<unk>".to_string()),
            pad_token: None, // LLaMA v1 doesn't use padding
            mask_token: None,
            additional_special_tokens: vec![],
        }
    }

    pub fn llama_v2() -> Self {
        Self {
            bos_token: Some("<s>".to_string()),
            eos_token: Some("</s>".to_string()),
            unk_token: Some("<unk>".to_string()),
            pad_token: Some("<pad>".to_string()),
            mask_token: None,
            additional_special_tokens: vec![
                "<<SYS>>".to_string(),
                "<</SYS>>".to_string(),
            ],
        }
    }

    pub fn code_llama() -> Self {
        let mut config = Self::llama_v2();
        config.additional_special_tokens.extend(vec![
            "<PRE>".to_string(),
            "<SUF>".to_string(),
            "<MID>".to_string(),
            "<EOD>".to_string(),
        ]);
        config
    }

    pub fn all_special_tokens(&self) -> HashSet<String> {
        let mut tokens = HashSet::new();

        if let Some(ref token) = self.bos_token {
            tokens.insert(token.clone());
        }
        if let Some(ref token) = self.eos_token {
            tokens.insert(token.clone());
        }
        if let Some(ref token) = self.unk_token {
            tokens.insert(token.clone());
        }
        if let Some(ref token) = self.pad_token {
            tokens.insert(token.clone());
        }
        if let Some(ref token) = self.mask_token {
            tokens.insert(token.clone());
        }

        tokens.extend(self.additional_special_tokens.iter().cloned());
        tokens
    }
}

pub struct LlamaTokenizer {
    vocab: HashMap<String, u32>,
    special_token_config: SpecialTokenConfig,
    special_token_set: HashSet<String>,
}

impl LlamaTokenizer {
    pub fn new(vocab: HashMap<String, u32>, config: SpecialTokenConfig) -> Self {
        let special_token_set = config.all_special_tokens();

        Self {
            vocab,
            special_token_config: config,
            special_token_set,
        }
    }

    pub fn from_model_variant(variant: LlamaVariant, vocab: HashMap<String, u32>) -> Self {
        let config = match variant {
            LlamaVariant::V1 => SpecialTokenConfig::llama_v1(),
            LlamaVariant::V2 => SpecialTokenConfig::llama_v2(),
            LlamaVariant::CodeLlama => SpecialTokenConfig::code_llama(),
            LlamaVariant::Custom(config) => config,
        };

        Self::new(vocab, config)
    }

    fn is_special_token(&self, token: &str) -> bool {
        self.special_token_set.contains(token)
    }

    pub fn add_special_token(&mut self, token: String) -> Result<()> {
        if self.special_token_set.contains(&token) {
            return Err(anyhow::anyhow!("Special token '{}' already exists", token));
        }

        self.special_token_config.additional_special_tokens.push(token.clone());
        self.special_token_set.insert(token);
        Ok(())
    }

    pub fn remove_special_token(&mut self, token: &str) -> bool {
        if self.special_token_set.remove(token) {
            self.special_token_config.additional_special_tokens.retain(|t| t != token);
            true
        } else {
            false
        }
    }

    pub fn get_special_token_id(&self, token_type: SpecialTokenType) -> Option<u32> {
        let token_str = match token_type {
            SpecialTokenType::Bos => self.special_token_config.bos_token.as_ref()?,
            SpecialTokenType::Eos => self.special_token_config.eos_token.as_ref()?,
            SpecialTokenType::Unk => self.special_token_config.unk_token.as_ref()?,
            SpecialTokenType::Pad => self.special_token_config.pad_token.as_ref()?,
            SpecialTokenType::Mask => self.special_token_config.mask_token.as_ref()?,
        };

        self.vocab.get(token_str).copied()
    }

    pub fn is_system_prompt_token(&self, token: &str) -> bool {
        // LLaMA v2 specific system prompt tokens
        matches!(token, "<<SYS>>" | "<</SYS>>")
    }

    pub fn is_code_special_token(&self, token: &str) -> bool {
        // Code Llama specific tokens
        matches!(token, "<PRE>" | "<SUF>" | "<MID>" | "<EOD>")
    }
}

#[derive(Debug, Clone)]
pub enum LlamaVariant {
    V1,
    V2,
    CodeLlama,
    Custom(SpecialTokenConfig),
}

#[derive(Debug, Clone, Copy)]
pub enum SpecialTokenType {
    Bos,
    Eos,
    Unk,
    Pad,
    Mask,
}
```

## Implementation Plan

### Task 1: Special Token Configuration System
- [ ] Implement configurable special token management
- [ ] Add support for different LLaMA variants
- [ ] Create preset configurations for known models
- [ ] Add runtime special token modification

### Task 2: Model Variant Support
- [ ] Add LLaMA v1, v2, and Code Llama presets
- [ ] Implement variant auto-detection from model metadata
- [ ] Support custom special token configurations
- [ ] Add validation for special token consistency

### Task 3: Integration and Testing
- [ ] Update tokenizer initialization to use configurations
- [ ] Add comprehensive testing for different variants
- [ ] Validate special token handling across models
- [ ] Test custom special token scenarios

## Testing Strategy

### Special Token Tests
```rust
#[test]
fn test_llama_v1_special_tokens() {
    let config = SpecialTokenConfig::llama_v1();
    let mut vocab = HashMap::new();
    vocab.insert("<s>".to_string(), 1);
    vocab.insert("</s>".to_string(), 2);
    vocab.insert("<unk>".to_string(), 0);

    let tokenizer = LlamaTokenizer::new(vocab, config);

    assert!(tokenizer.is_special_token("<s>"));
    assert!(tokenizer.is_special_token("</s>"));
    assert!(tokenizer.is_special_token("<unk>"));
    assert!(!tokenizer.is_special_token("<pad>")); // Not in v1
}

#[test]
fn test_code_llama_special_tokens() {
    let config = SpecialTokenConfig::code_llama();
    let vocab = create_code_llama_vocab();
    let tokenizer = LlamaTokenizer::new(vocab, config);

    assert!(tokenizer.is_special_token("<PRE>"));
    assert!(tokenizer.is_special_token("<SUF>"));
    assert!(tokenizer.is_code_special_token("<MID>"));
    assert!(!tokenizer.is_special_token("<CUSTOM>"));
}

#[test]
fn test_custom_special_tokens() {
    let mut config = SpecialTokenConfig::llama_v2();
    config.additional_special_tokens.push("<CUSTOM>".to_string());

    let vocab = create_test_vocab();
    let mut tokenizer = LlamaTokenizer::new(vocab, config);

    assert!(tokenizer.is_special_token("<CUSTOM>"));

    // Test runtime addition
    tokenizer.add_special_token("<RUNTIME>".to_string()).unwrap();
    assert!(tokenizer.is_special_token("<RUNTIME>"));
}
```

## Acceptance Criteria

- [ ] Different LLaMA variants use appropriate special token sets
- [ ] Special tokens are configurable at runtime
- [ ] Model variant detection works automatically
- [ ] Custom special tokens can be added and removed
- [ ] All existing functionality continues to work
- [ ] Performance impact is minimal

## Risk Assessment

**Low Risk**: Configuration change that improves flexibility without breaking existing functionality.

**Mitigation Strategies**:
- Maintain backwards compatibility with existing hardcoded behavior
- Provide sensible defaults for unknown model variants
- Add comprehensive validation for special token configurations
- Test across multiple LLaMA model versions
