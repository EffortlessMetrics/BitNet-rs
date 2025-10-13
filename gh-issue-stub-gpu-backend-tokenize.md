# [GPU Backend] Replace placeholder tokenization with production implementation

## Problem Description

The `GpuBackend::tokenize` function in `crates/bitnet-inference/src/gpu.rs` contains a placeholder implementation that simply converts characters to u32 values. This naive approach doesn't use proper tokenization, which will cause significant issues with text processing, model compatibility, and inference quality.

## Environment

- **File**: `crates/bitnet-inference/src/gpu.rs`
- **Function**: `GpuBackend::tokenize`
- **Architecture**: GPU inference backend with CUDA acceleration
- **Dependencies**: `bitnet_tokenizers` crate integration needed

## Root Cause Analysis

### Current Implementation
```rust
fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
    // Placeholder implementation - in practice would use a proper tokenizer
    Ok(text.chars().map(|c| c as u32).collect())
}
```

### Issues Identified
1. **Character-Level Tokenization**: Converts characters directly to u32, ignoring subword tokenization
2. **No Vocabulary Mapping**: Doesn't use model's actual vocabulary
3. **Missing Special Tokens**: No handling of BOS, EOS, PAD, or UNK tokens
4. **Unicode Issues**: Direct char to u32 casting may not handle Unicode properly
5. **Model Incompatibility**: Generated token IDs won't match model's expected input format

## Impact Assessment

- **Severity**: Critical - Breaks actual text inference
- **Model Compatibility**: Critical - Token IDs won't match trained vocabulary
- **Inference Quality**: Critical - Will produce nonsensical outputs
- **Production Readiness**: Blocking - Cannot be used with real models

## Proposed Solution

### Integration with BitNet Tokenizers

```rust
impl GpuBackend {
    /// Tokenize text using the model's associated tokenizer
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        debug!("Tokenizing text for GPU inference: {} chars", text.len());

        // Get tokenizer from model or use configured tokenizer
        let tokenizer = self.get_tokenizer()?;

        // Use proper tokenization with model-specific settings
        let tokens = tokenizer.encode(text, true, true)
            .map_err(|e| BitNetError::Tokenization(format!("GPU tokenization failed: {}", e)))?;

        // Validate token IDs are within model vocabulary
        self.validate_token_ids(&tokens)?;

        debug!("Tokenized to {} tokens", tokens.len());
        Ok(tokens)
    }

    /// Get the tokenizer associated with this GPU backend
    fn get_tokenizer(&self) -> Result<Arc<dyn Tokenizer>> {
        match &self.tokenizer {
            Some(tokenizer) => Ok(tokenizer.clone()),
            None => {
                // Try to get tokenizer from model if available
                if let Some(model_tokenizer) = self.model.get_tokenizer() {
                    Ok(model_tokenizer)
                } else {
                    Err(BitNetError::Configuration(
                        "No tokenizer configured for GPU backend".to_string()
                    ))
                }
            }
        }
    }

    /// Validate that token IDs are within the model's vocabulary range
    fn validate_token_ids(&self, tokens: &[u32]) -> Result<()> {
        let vocab_size = self.model.config().vocab_size as u32;

        for (idx, &token) in tokens.iter().enumerate() {
            if token >= vocab_size {
                return Err(BitNetError::Tokenization(format!(
                    "Token ID {} at position {} exceeds vocabulary size {}",
                    token, idx, vocab_size
                )));
            }
        }

        Ok(())
    }

    /// Enhanced tokenization with preprocessing options
    pub fn tokenize_with_options(
        &self,
        text: &str,
        options: &TokenizationOptions,
    ) -> Result<TokenizationResult> {
        let tokenizer = self.get_tokenizer()?;

        // Apply preprocessing based on options
        let preprocessed_text = self.preprocess_text(text, options)?;

        // Tokenize with proper settings
        let tokens = tokenizer.encode(
            &preprocessed_text,
            options.add_bos_token,
            options.add_special_tokens,
        ).map_err(|e| BitNetError::Tokenization(format!("Tokenization failed: {}", e)))?;

        // Post-process tokens if needed
        let processed_tokens = self.post_process_tokens(tokens, options)?;

        // Validate and prepare for GPU
        self.validate_token_ids(&processed_tokens)?;
        let gpu_tokens = self.prepare_tokens_for_gpu(&processed_tokens)?;

        Ok(TokenizationResult {
            tokens: processed_tokens,
            gpu_tokens,
            original_text: text.to_string(),
            preprocessed_text,
            metadata: self.create_tokenization_metadata(&options),
        })
    }

    fn preprocess_text(&self, text: &str, options: &TokenizationOptions) -> Result<String> {
        let mut processed = text.to_string();

        // Apply text normalization if requested
        if options.normalize_text {
            processed = self.normalize_text(&processed)?;
        }

        // Handle whitespace according to model requirements
        if options.preserve_whitespace {
            // Some models require exact whitespace preservation
            processed = processed;
        } else {
            // Standard whitespace normalization
            processed = processed.trim().to_string();
        }

        // Apply any model-specific preprocessing
        if let Some(ref processor) = options.text_processor {
            processed = processor.process(&processed)?;
        }

        Ok(processed)
    }

    fn post_process_tokens(&self, tokens: Vec<u32>, options: &TokenizationOptions) -> Result<Vec<u32>> {
        let mut processed = tokens;

        // Apply token filtering if specified
        if let Some(ref filter) = options.token_filter {
            processed = filter.filter(processed)?;
        }

        // Handle sequence length limits
        if let Some(max_length) = options.max_length {
            if processed.len() > max_length {
                processed.truncate(max_length);
                debug!("Truncated tokens to max length: {}", max_length);
            }
        }

        // Apply padding if needed for GPU efficiency
        if let Some(target_length) = options.pad_to_length {
            let pad_token = options.pad_token_id.unwrap_or(0);
            while processed.len() < target_length {
                processed.push(pad_token);
            }
        }

        Ok(processed)
    }

    fn prepare_tokens_for_gpu(&self, tokens: &[u32]) -> Result<GpuTokens> {
        // Convert tokens to GPU-friendly format
        let gpu_tokens = match self.device_type {
            GpuDeviceType::Cuda => {
                // Copy tokens to CUDA memory
                let cuda_tokens = self.cuda_context.copy_to_device(tokens)?;
                GpuTokens::Cuda(cuda_tokens)
            },
            GpuDeviceType::Metal => {
                // Copy tokens to Metal buffer
                let metal_tokens = self.metal_context.copy_to_buffer(tokens)?;
                GpuTokens::Metal(metal_tokens)
            },
            GpuDeviceType::OpenCL => {
                // Copy tokens to OpenCL buffer
                let opencl_tokens = self.opencl_context.copy_to_buffer(tokens)?;
                GpuTokens::OpenCL(opencl_tokens)
            },
        };

        Ok(gpu_tokens)
    }

    fn normalize_text(&self, text: &str) -> Result<String> {
        // Apply Unicode normalization
        use unicode_normalization::UnicodeNormalization;
        let normalized = text.nfd().collect::<String>();

        // Apply any model-specific normalization
        // This could include lowercasing, accent removal, etc.
        Ok(normalized)
    }

    fn create_tokenization_metadata(&self, options: &TokenizationOptions) -> TokenizationMetadata {
        TokenizationMetadata {
            tokenizer_type: self.get_tokenizer_type(),
            vocab_size: self.model.config().vocab_size,
            special_tokens_used: options.add_special_tokens,
            preprocessing_applied: options.normalize_text,
            max_length_applied: options.max_length,
            padding_applied: options.pad_to_length.is_some(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenizationOptions {
    pub add_bos_token: bool,
    pub add_special_tokens: bool,
    pub normalize_text: bool,
    pub preserve_whitespace: bool,
    pub max_length: Option<usize>,
    pub pad_to_length: Option<usize>,
    pub pad_token_id: Option<u32>,
    pub token_filter: Option<Box<dyn TokenFilter>>,
    pub text_processor: Option<Box<dyn TextProcessor>>,
}

impl Default for TokenizationOptions {
    fn default() -> Self {
        Self {
            add_bos_token: true,
            add_special_tokens: true,
            normalize_text: true,
            preserve_whitespace: false,
            max_length: None,
            pad_to_length: None,
            pad_token_id: None,
            token_filter: None,
            text_processor: None,
        }
    }
}

#[derive(Debug)]
pub struct TokenizationResult {
    pub tokens: Vec<u32>,
    pub gpu_tokens: GpuTokens,
    pub original_text: String,
    pub preprocessed_text: String,
    pub metadata: TokenizationMetadata,
}

#[derive(Debug)]
pub enum GpuTokens {
    Cuda(CudaBuffer<u32>),
    Metal(MetalBuffer<u32>),
    OpenCL(OpenCLBuffer<u32>),
}

#[derive(Debug, Clone)]
pub struct TokenizationMetadata {
    pub tokenizer_type: String,
    pub vocab_size: usize,
    pub special_tokens_used: bool,
    pub preprocessing_applied: bool,
    pub max_length_applied: Option<usize>,
    pub padding_applied: bool,
}

pub trait TokenFilter: Send + Sync {
    fn filter(&self, tokens: Vec<u32>) -> Result<Vec<u32>>;
}

pub trait TextProcessor: Send + Sync {
    fn process(&self, text: &str) -> Result<String>;
}
```

### Enhanced GPU Backend Structure

```rust
pub struct GpuBackend {
    device_type: GpuDeviceType,
    model: Arc<dyn Model>,
    tokenizer: Option<Arc<dyn Tokenizer>>, // Now properly integrated
    cuda_context: Option<CudaContext>,
    metal_context: Option<MetalContext>,
    opencl_context: Option<OpenCLContext>,
    config: GpuBackendConfig,
}

impl GpuBackend {
    pub fn new_with_tokenizer(
        device_type: GpuDeviceType,
        model: Arc<dyn Model>,
        tokenizer: Arc<dyn Tokenizer>,
        config: GpuBackendConfig,
    ) -> Result<Self> {
        let mut backend = Self::new(device_type, model, config)?;
        backend.tokenizer = Some(tokenizer);
        Ok(backend)
    }

    pub fn set_tokenizer(&mut self, tokenizer: Arc<dyn Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }
}
```

## Implementation Plan

### Phase 1: Basic Integration (Week 1)
- [ ] Integrate `bitnet_tokenizers` crate with GPU backend
- [ ] Implement basic tokenization using proper tokenizer
- [ ] Add token validation and error handling
- [ ] Create comprehensive test suite

### Phase 2: Advanced Features (Week 2)
- [ ] Add tokenization options and preprocessing
- [ ] Implement GPU memory management for tokens
- [ ] Add support for different GPU backends (CUDA, Metal, OpenCL)
- [ ] Create performance optimization

### Phase 3: Production Features (Week 3)
- [ ] Add batch tokenization support
- [ ] Implement caching for tokenization results
- [ ] Add detailed logging and metrics
- [ ] Create comprehensive documentation

### Phase 4: Integration and Testing (Week 4)
- [ ] Integrate with existing GPU inference pipeline
- [ ] Add cross-validation with CPU tokenization
- [ ] Performance benchmarking
- [ ] Real-world model testing

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_gpu_tokenization() {
        let backend = create_test_gpu_backend_with_tokenizer();
        let tokens = backend.tokenize("Hello, world!").unwrap();

        assert!(!tokens.is_empty());
        assert!(tokens.iter().all(|&t| t < backend.model.config().vocab_size as u32));
    }

    #[test]
    fn test_tokenization_with_special_tokens() {
        let backend = create_test_gpu_backend_with_tokenizer();
        let options = TokenizationOptions {
            add_bos_token: true,
            add_special_tokens: true,
            ..Default::default()
        };

        let result = backend.tokenize_with_options("Test text", &options).unwrap();

        // Should include BOS token
        assert!(result.metadata.special_tokens_used);
    }

    #[test]
    fn test_gpu_token_preparation() {
        let backend = create_test_gpu_backend_cuda();
        let tokens = vec![1, 2, 3, 4, 5];

        let gpu_tokens = backend.prepare_tokens_for_gpu(&tokens).unwrap();

        match gpu_tokens {
            GpuTokens::Cuda(_) => assert!(true),
            _ => panic!("Expected CUDA tokens"),
        }
    }

    #[test]
    fn test_tokenization_error_handling() {
        let backend = create_test_gpu_backend_no_tokenizer();
        let result = backend.tokenize("Test");

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No tokenizer"));
    }
}
```

## Acceptance Criteria

- [ ] Proper integration with `bitnet_tokenizers` crate
- [ ] Support for all standard tokenization features (BOS, EOS, special tokens)
- [ ] GPU memory management for tokenized data
- [ ] Comprehensive error handling and validation
- [ ] Performance equivalent to CPU tokenization
- [ ] Support for multiple GPU backends (CUDA, Metal, OpenCL)
- [ ] Batch tokenization capabilities
- [ ] Complete test coverage including edge cases

## Priority: Critical

This is a blocking issue for actual GPU inference functionality, as the current placeholder makes the GPU backend unusable with real models and text input.
