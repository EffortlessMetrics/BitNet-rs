# [HARDCODED] Universal Tokenizer Hardcodes GPT-2 Fallback - Breaks Model Compatibility

## Problem Description

The `UniversalTokenizer::from_gguf` function in `crates/bitnet-tokenizers/src/universal.rs` hardcodes "gpt2" as a fallback when the `tokenizer.ggml.model` metadata is missing from GGUF files. This inappropriate default can lead to incorrect tokenization for non-GPT-2 models, causing inference failures and model incompatibility issues.

## Environment

- **File**: `crates/bitnet-tokenizers/src/universal.rs`
- **Function**: `UniversalTokenizer::from_gguf` (line 18)
- **Component**: Universal tokenizer and GGUF model loading
- **Build Configuration**: All feature configurations
- **Context**: Model loading with embedded tokenizer metadata

## Root Cause Analysis

### Technical Issues

1. **Inappropriate Hardcoded Fallback**:
   ```rust
   let model_type = reader.get_string_metadata("tokenizer.ggml.model")
       .unwrap_or_else(|| "gpt2".into()); // PROBLEMATIC HARDCODE
   ```

2. **Model Compatibility Issues**:
   - LLaMA models use SentencePiece tokenization, not GPT-2 BPE
   - Mistral, Phi, and other models have different tokenization schemes
   - GPT-2 tokenizer may not handle special tokens correctly for other models

3. **Silent Failure Mode**:
   - No error or warning when metadata is missing
   - Incorrect tokenization leads to degraded model performance
   - Difficult to debug tokenization-related inference issues

4. **Tokenization Mismatch Consequences**:
   - Wrong vocabulary mapping for input text
   - Incorrect special token handling (BOS, EOS, PAD)
   - Potential OOV (out-of-vocabulary) token mishandling
   - Model accuracy degradation or complete failure

### Impact Assessment

- **Model Compatibility**: Breaks inference for non-GPT-2 models
- **Accuracy**: Silent degradation of model performance
- **Debugging**: Difficult to identify tokenization as root cause
- **User Experience**: Confusing behavior when loading various model types

## Reproduction Steps

1. Load a LLaMA or Mistral model with missing tokenizer metadata:
   ```rust
   let tokenizer = UniversalTokenizer::from_gguf(Path::new("llama_model.gguf"))?;
   // Silently uses GPT-2 tokenizer instead of SentencePiece
   ```

2. Attempt tokenization with model-specific text:
   ```rust
   let tokens = tokenizer.encode("Hello, world!", false)?;
   // Wrong tokenization for LLaMA model
   ```

3. **Expected**: Clear error indicating missing tokenizer metadata
4. **Actual**: Silent fallback to incompatible GPT-2 tokenizer

## Proposed Solution

### Primary Approach: Explicit Error Handling with Model-Aware Detection

Replace hardcoded fallback with intelligent model detection and explicit error handling:

```rust
use anyhow::{Context, Result};
use std::path::Path;

impl UniversalTokenizer {
    pub fn from_gguf(path: &Path) -> Result<Self> {
        use bitnet_models::{GgufReader, loader::MmapFile};

        let mmap = MmapFile::open(path)
            .with_context(|| format!("Failed to open GGUF file: {}", path.display()))?;
        let reader = GgufReader::new(mmap.as_slice())
            .with_context(|| "Failed to parse GGUF file format")?;

        // Attempt to get tokenizer model type from metadata
        let model_type = match reader.get_string_metadata("tokenizer.ggml.model") {
            Some(model_type) => model_type,
            None => {
                // Instead of hardcoding, try to infer from other metadata
                Self::infer_tokenizer_type(&reader)
                    .with_context(|| {
                        "Missing 'tokenizer.ggml.model' metadata and could not infer tokenizer type. \
                         Please ensure the GGUF file contains proper tokenizer metadata."
                    })?
            }
        };

        tracing::info!("Loading tokenizer of type: {}", model_type);

        // Validate tokenizer type is supported
        Self::validate_tokenizer_type(&model_type)?;

        // Create tokenizer based on detected/specified type
        Self::create_tokenizer(&reader, &model_type, path)
    }

    /// Infer tokenizer type from model architecture and other metadata
    fn infer_tokenizer_type(reader: &GgufReader) -> Result<String> {
        // Try to infer from model architecture
        if let Some(arch) = reader.get_string_metadata("general.architecture") {
            let inferred_type = match arch.as_str() {
                "llama" | "code_llama" => "llama",
                "mistral" => "llama", // Mistral uses llama-style tokenization
                "phi" | "phi-msft" => "gpt2", // Phi models often use GPT-2 style
                "qwen" | "qwen2" => "qwen",
                "gpt2" | "gpt-2" => "gpt2",
                "bert" => "bert",
                arch => {
                    tracing::warn!("Unknown architecture '{}', cannot infer tokenizer type", arch);
                    return Err(anyhow::anyhow!(
                        "Cannot infer tokenizer type from unknown architecture: {}",
                        arch
                    ));
                }
            };

            tracing::info!("Inferred tokenizer type '{}' from architecture '{}'", inferred_type, arch);
            return Ok(inferred_type.to_string());
        }

        // Try to infer from vocabulary size and structure
        if let Some(vocab_size) = reader.get_u32_metadata("tokenizer.ggml.tokens") {
            let inferred_type = match vocab_size {
                32000 | 32001 => "llama", // Common LLaMA vocab sizes
                50257 | 50258 => "gpt2",  // GPT-2 vocab sizes
                vocab_size if vocab_size > 100000 => "qwen", // Large vocab often indicates Qwen
                _ => {
                    return Err(anyhow::anyhow!(
                        "Cannot infer tokenizer type from vocabulary size: {}",
                        vocab_size
                    ));
                }
            };

            tracing::info!("Inferred tokenizer type '{}' from vocabulary size {}", inferred_type, vocab_size);
            return Ok(inferred_type.to_string());
        }

        // Try to infer from tokenizer-specific metadata presence
        if reader.get_metadata("tokenizer.ggml.merges").is_some() {
            tracing::info!("Found BPE merges, inferring GPT-2 style tokenizer");
            return Ok("gpt2".to_string());
        }

        if reader.get_metadata("tokenizer.ggml.scores").is_some() {
            tracing::info!("Found token scores, inferring SentencePiece style tokenizer");
            return Ok("llama".to_string());
        }

        Err(anyhow::anyhow!(
            "Could not infer tokenizer type from available metadata. \
             Please ensure GGUF file contains 'tokenizer.ggml.model' metadata."
        ))
    }

    /// Validate that the tokenizer type is supported
    fn validate_tokenizer_type(model_type: &str) -> Result<()> {
        match model_type {
            "gpt2" | "llama" | "qwen" | "bert" | "t5" => Ok(()),
            unsupported => Err(anyhow::anyhow!(
                "Unsupported tokenizer type: '{}'. Supported types: gpt2, llama, qwen, bert, t5",
                unsupported
            )),
        }
    }

    /// Create appropriate tokenizer based on type
    fn create_tokenizer(reader: &GgufReader, model_type: &str, path: &Path) -> Result<Self> {
        match model_type {
            "gpt2" => Self::create_gpt2_tokenizer(reader, path),
            "llama" => Self::create_llama_tokenizer(reader, path),
            "qwen" => Self::create_qwen_tokenizer(reader, path),
            "bert" => Self::create_bert_tokenizer(reader, path),
            "t5" => Self::create_t5_tokenizer(reader, path),
            _ => Err(anyhow::anyhow!("Tokenizer type '{}' not implemented", model_type)),
        }
    }

    fn create_gpt2_tokenizer(reader: &GgufReader, path: &Path) -> Result<Self> {
        tracing::debug!("Creating GPT-2 style tokenizer");

        // Extract GPT-2 specific metadata
        let vocab = Self::extract_gpt2_vocab(reader)?;
        let merges = Self::extract_gpt2_merges(reader)?;
        let special_tokens = Self::extract_gpt2_special_tokens(reader)?;

        Ok(UniversalTokenizer {
            tokenizer_type: TokenizerType::Gpt2(Gpt2Tokenizer::new(vocab, merges, special_tokens)?),
            model_path: path.to_path_buf(),
            vocab_size: reader.get_u32_metadata("tokenizer.ggml.tokens").unwrap_or(50257) as usize,
        })
    }

    fn create_llama_tokenizer(reader: &GgufReader, path: &Path) -> Result<Self> {
        tracing::debug!("Creating LLaMA style (SentencePiece) tokenizer");

        // Extract SentencePiece specific metadata
        let vocab = Self::extract_sentencepiece_vocab(reader)?;
        let scores = Self::extract_sentencepiece_scores(reader)?;
        let special_tokens = Self::extract_llama_special_tokens(reader)?;

        Ok(UniversalTokenizer {
            tokenizer_type: TokenizerType::SentencePiece(SentencePieceTokenizer::new(
                vocab, scores, special_tokens
            )?),
            model_path: path.to_path_buf(),
            vocab_size: reader.get_u32_metadata("tokenizer.ggml.tokens").unwrap_or(32000) as usize,
        })
    }

    fn create_qwen_tokenizer(reader: &GgufReader, path: &Path) -> Result<Self> {
        tracing::debug!("Creating Qwen style tokenizer");

        // Qwen typically uses a variant of GPT-2 style tokenization with larger vocab
        let vocab = Self::extract_qwen_vocab(reader)?;
        let merges = Self::extract_qwen_merges(reader)?;
        let special_tokens = Self::extract_qwen_special_tokens(reader)?;

        Ok(UniversalTokenizer {
            tokenizer_type: TokenizerType::Qwen(QwenTokenizer::new(vocab, merges, special_tokens)?),
            model_path: path.to_path_buf(),
            vocab_size: reader.get_u32_metadata("tokenizer.ggml.tokens").unwrap_or(151936) as usize,
        })
    }

    // Enhanced extraction methods with proper error handling
    fn extract_gpt2_vocab(reader: &GgufReader) -> Result<Vec<String>> {
        let vocab_data = reader.get_string_array_metadata("tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow::anyhow!("Missing GPT-2 vocabulary tokens"))?;
        Ok(vocab_data)
    }

    fn extract_gpt2_merges(reader: &GgufReader) -> Result<Vec<String>> {
        let merges_data = reader.get_string_array_metadata("tokenizer.ggml.merges")
            .ok_or_else(|| anyhow::anyhow!("Missing GPT-2 BPE merges"))?;
        Ok(merges_data)
    }

    fn extract_gpt2_special_tokens(reader: &GgufReader) -> Result<SpecialTokens> {
        let bos_token_id = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
        let eos_token_id = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
        let pad_token_id = reader.get_u32_metadata("tokenizer.ggml.pad_token_id");
        let unk_token_id = reader.get_u32_metadata("tokenizer.ggml.unk_token_id");

        Ok(SpecialTokens {
            bos_token_id,
            eos_token_id,
            pad_token_id,
            unk_token_id,
        })
    }

    fn extract_sentencepiece_vocab(reader: &GgufReader) -> Result<Vec<String>> {
        let vocab_data = reader.get_string_array_metadata("tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow::anyhow!("Missing SentencePiece vocabulary tokens"))?;
        Ok(vocab_data)
    }

    fn extract_sentencepiece_scores(reader: &GgufReader) -> Result<Vec<f32>> {
        let scores_data = reader.get_f32_array_metadata("tokenizer.ggml.scores")
            .ok_or_else(|| anyhow::anyhow!("Missing SentencePiece token scores"))?;
        Ok(scores_data)
    }

    fn extract_llama_special_tokens(reader: &GgufReader) -> Result<SpecialTokens> {
        // LLaMA typically has specific token IDs
        let bos_token_id = reader.get_u32_metadata("tokenizer.ggml.bos_token_id").or(Some(1));
        let eos_token_id = reader.get_u32_metadata("tokenizer.ggml.eos_token_id").or(Some(2));
        let pad_token_id = reader.get_u32_metadata("tokenizer.ggml.pad_token_id"); // Often None
        let unk_token_id = reader.get_u32_metadata("tokenizer.ggml.unk_token_id").or(Some(0));

        Ok(SpecialTokens {
            bos_token_id,
            eos_token_id,
            pad_token_id,
            unk_token_id,
        })
    }
}

// Enhanced tokenizer types enum
#[derive(Debug)]
pub enum TokenizerType {
    Gpt2(Gpt2Tokenizer),
    SentencePiece(SentencePieceTokenizer),
    Qwen(QwenTokenizer),
    Bert(BertTokenizer),
    T5(T5Tokenizer),
}

#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
}

// Configuration-based tokenizer creation
pub struct TokenizerConfig {
    pub tokenizer_type: Option<String>,
    pub fallback_strategy: FallbackStrategy,
    pub strict_mode: bool,
}

#[derive(Debug, Clone)]
pub enum FallbackStrategy {
    Error,              // Return error on missing metadata
    InferFromModel,     // Try to infer from model metadata
    UseDefault(String), // Use specified default (with warning)
}

impl UniversalTokenizer {
    pub fn from_gguf_with_config(path: &Path, config: TokenizerConfig) -> Result<Self> {
        let mmap = MmapFile::open(path)?;
        let reader = GgufReader::new(mmap.as_slice())?;

        let model_type = match reader.get_string_metadata("tokenizer.ggml.model") {
            Some(model_type) => model_type,
            None => match config.fallback_strategy {
                FallbackStrategy::Error => {
                    return Err(anyhow::anyhow!(
                        "Missing 'tokenizer.ggml.model' metadata in GGUF file"
                    ));
                }
                FallbackStrategy::InferFromModel => {
                    Self::infer_tokenizer_type(&reader)?
                }
                FallbackStrategy::UseDefault(default_type) => {
                    tracing::warn!(
                        "Using default tokenizer type '{}' due to missing metadata",
                        default_type
                    );
                    default_type
                }
            }
        };

        if config.strict_mode {
            Self::validate_tokenizer_compatibility(&reader, &model_type)?;
        }

        Self::create_tokenizer(&reader, &model_type, path)
    }

    fn validate_tokenizer_compatibility(reader: &GgufReader, model_type: &str) -> Result<()> {
        // Validate that the tokenizer type matches expected model characteristics
        match model_type {
            "gpt2" => {
                if reader.get_metadata("tokenizer.ggml.merges").is_none() {
                    return Err(anyhow::anyhow!(
                        "GPT-2 tokenizer requires BPE merges but none found"
                    ));
                }
            }
            "llama" => {
                if reader.get_metadata("tokenizer.ggml.scores").is_none() {
                    return Err(anyhow::anyhow!(
                        "LLaMA tokenizer requires token scores but none found"
                    ));
                }
            }
            _ => {
                // Additional validation for other tokenizer types
            }
        }

        Ok(())
    }
}
```

### Alternative Approaches

1. **Configurable Default**: Allow users to specify fallback tokenizer type
2. **Multi-Stage Inference**: Use multiple heuristics to determine tokenizer type
3. **External Tokenizer Registry**: Maintain database of model→tokenizer mappings

## Implementation Plan

### Phase 1: Remove Hardcoded Fallback (Priority: Critical)
- [ ] Replace hardcoded "gpt2" with explicit error handling
- [ ] Implement model architecture-based tokenizer inference
- [ ] Add comprehensive error messages for missing metadata
- [ ] Create migration guide for affected code

### Phase 2: Intelligent Inference (Priority: High)
- [ ] Implement robust tokenizer type inference from metadata
- [ ] Add support for multiple inference strategies
- [ ] Create validation for tokenizer compatibility
- [ ] Add comprehensive test suite for inference logic

### Phase 3: Enhanced Configuration (Priority: Medium)
- [ ] Add configurable fallback strategies
- [ ] Implement strict validation mode
- [ ] Support custom tokenizer type overrides
- [ ] Add tokenizer compatibility database

### Phase 4: Integration & Testing (Priority: High)
- [ ] Integration with model loading pipeline
- [ ] Cross-validation with different model types
- [ ] Performance testing for inference logic
- [ ] Documentation and usage examples

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_no_hardcoded_fallback() {
    // Create GGUF file without tokenizer.ggml.model metadata
    let temp_file = create_test_gguf_without_tokenizer_metadata();

    let result = UniversalTokenizer::from_gguf(&temp_file);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("tokenizer.ggml.model"));
}

#[test]
fn test_tokenizer_inference_llama() {
    let temp_file = create_test_gguf_llama_architecture();

    let tokenizer = UniversalTokenizer::from_gguf(&temp_file).unwrap();
    assert!(matches!(tokenizer.tokenizer_type, TokenizerType::SentencePiece(_)));
}

#[test]
fn test_tokenizer_inference_gpt2() {
    let temp_file = create_test_gguf_gpt2_architecture();

    let tokenizer = UniversalTokenizer::from_gguf(&temp_file).unwrap();
    assert!(matches!(tokenizer.tokenizer_type, TokenizerType::Gpt2(_)));
}

#[test]
fn test_unsupported_architecture_error() {
    let temp_file = create_test_gguf_unknown_architecture();

    let result = UniversalTokenizer::from_gguf(&temp_file);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("unknown architecture"));
}
```

### Integration Tests
```bash
# Test with real model files
cargo test --no-default-features --features cpu test_tokenizer_loading_real_models

# Test tokenizer inference accuracy
cargo test test_tokenizer_inference_validation

# Cross-validation with reference tokenizers
cargo run -p xtask -- crossval --component tokenizer
```

## Acceptance Criteria

### Functional Requirements
- [ ] No hardcoded tokenizer type fallbacks
- [ ] Clear error messages for missing metadata
- [ ] Accurate tokenizer type inference from model architecture
- [ ] Support for all major model architectures (LLaMA, GPT-2, Qwen, etc.)

### Quality Requirements
- [ ] 100% test coverage for tokenizer inference logic
- [ ] Cross-validation with reference tokenizer implementations
- [ ] Clear error handling and debugging information
- [ ] Comprehensive documentation for supported tokenizer types

### Compatibility Requirements
- [ ] Proper tokenization for each supported model type
- [ ] No silent failures or incorrect tokenization
- [ ] Backward compatibility with existing explicit tokenizer metadata
- [ ] Support for emerging model architectures

## Related Issues

- Model loading validation and compatibility checking
- Tokenizer cross-validation with reference implementations
- GGUF metadata standardization and validation
- Universal tokenizer architecture improvements

## Dependencies

- GGUF metadata parsing and validation utilities
- Model architecture detection and classification
- Tokenizer implementations for different types (GPT-2, SentencePiece, etc.)
- Error handling and logging infrastructure

## Migration Impact

- **Breaking Change**: May break code relying on silent GPT-2 fallback
- **Error Handling**: Explicit error handling required for missing metadata
- **Model Compatibility**: Improved compatibility with non-GPT-2 models
- **Debugging**: Better error messages for tokenization issues

---

**Labels**: `critical`, `hardcoded-value`, `tokenizer`, `model-compatibility`, `gguf-metadata`, `error-handling`
**Assignee**: Core team member with tokenizer and model loading experience
**Milestone**: Robust Tokenizer Loading (v0.3.0)
**Estimated Effort**: 1-2 weeks for implementation and comprehensive testing
