# [Feature] Comprehensive Tokenizer Discovery and Embedded Parser Implementation

## Problem Description

BitNet-rs tokenizer discovery system contains multiple critical implementation gaps that prevent robust model loading and tokenizer extraction. Three major components require complete implementation:

1. **Test Infrastructure**: `test_tokenizer_discovery_from_gguf_llama3()` is a placeholder that always expects failure
2. **Model Type Inference**: `extract_model_type()` has incomplete tensor pattern matching for architecture detection
3. **Embedded Tokenizer Parsing**: `try_extract_embedded_tokenizer()` creates basic tokenizers instead of parsing actual embedded data

These limitations prevent automatic tokenizer discovery from GGUF files and force manual tokenizer specification.

## Environment

- **Affected Crates**: `bitnet-tokenizers`, `bitnet-models`
- **Primary Files**:
  - `crates/bitnet-tokenizers/src/discovery.rs`
  - `crates/bitnet-models/src/gguf.rs` (GGUF reader integration)
- **Build Configuration**: `--no-default-features --features cpu,spm`
- **Supported Formats**: GGUF files with embedded tokenizers (SentencePiece, HuggingFace tokenizer.json)
- **Target Models**: LLaMA-3, GPT-2, BERT, T5, BitNet variants

## Root Cause Analysis

### Implementation Gaps

1. **Test Scaffolding**: Placeholder test prevents validation of GGUF parsing functionality
   ```rust
   // Current: Always expects failure
   assert!(result.is_err(), "Test scaffolding should fail until implemented");
   ```

2. **Limited Architecture Detection**: Only checks basic LLaMA patterns, missing comprehensive model type inference
   ```rust
   // Current: Minimal pattern matching
   let has_llama_patterns = tensor_names.iter().any(|name| {
       name.contains("attn_q") || name.contains("attn_k") || name.contains("attn_v")
   });
   ```

3. **Basic Tokenizer Fallback**: Creates `BasicTokenizer` instead of parsing embedded tokenizer data
   ```rust
   // Current: Simplified implementation
   let basic_tokenizer = crate::BasicTokenizer::with_config(self.vocab_size, Some(1), Some(2), Some(0));
   ```

### Functional Limitations

- **No Real Tokenizer Extraction**: Embedded SentencePiece models not parsed
- **Missing HuggingFace Support**: tokenizer.json parsing not implemented
- **Incomplete Model Detection**: Many transformer architectures unrecognized
- **Poor Error Handling**: Placeholder tests mask real implementation issues

## Impact Assessment

- **Severity**: High - Blocks automatic tokenizer loading for most models
- **User Experience**: Manual tokenizer specification required for all models
- **Model Compatibility**: Limited to manually configured tokenizers
- **Development Efficiency**: Placeholder tests prevent proper validation
- **Production Readiness**: Cannot handle diverse model formats in deployment

## Proposed Solution

### 1. Comprehensive GGUF Test Infrastructure

**Real GGUF Metadata Testing**:
```rust
// crates/bitnet-tokenizers/src/discovery.rs
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_tokenizer_discovery_from_gguf_llama3() {
        let test_gguf = create_test_gguf_llama3();
        let discovery = TokenizerDiscovery::from_gguf(&test_gguf).unwrap();

        assert_eq!(discovery.vocab_size(), 128256);
        assert_eq!(discovery.model_type(), "llama");
        assert_eq!(discovery.special_tokens().bos_token_id, Some(128000));
        assert_eq!(discovery.special_tokens().eos_token_id, Some(128001));
    }

    #[test]
    fn test_tokenizer_discovery_from_gguf_gpt2() {
        let test_gguf = create_test_gguf_gpt2();
        let discovery = TokenizerDiscovery::from_gguf(&test_gguf).unwrap();

        assert_eq!(discovery.vocab_size(), 50257);
        assert_eq!(discovery.model_type(), "gpt2");
        assert_eq!(discovery.special_tokens().eos_token_id, Some(50256));
    }

    #[test]
    fn test_embedded_sentencepiece_extraction() {
        let test_gguf = create_test_gguf_with_spm();
        let discovery = TokenizerDiscovery::from_gguf(&test_gguf).unwrap();

        let tokenizer = discovery.try_extract_embedded_tokenizer().unwrap().unwrap();

        // Test actual tokenization
        let tokens = tokenizer.tokenize("Hello, world!").unwrap();
        assert!(tokens.len() > 0);

        let reconstructed = tokenizer.detokenize(&tokens).unwrap();
        assert_eq!(reconstructed.trim(), "Hello, world!");
    }

    fn create_test_gguf_llama3() -> PathBuf {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_path_buf();

        // Write proper GGUF header
        let mut gguf_writer = GgufWriter::new(&mut temp_file);

        // Add LLaMA-3 metadata
        gguf_writer.add_string("general.architecture", "llama");
        gguf_writer.add_string("general.name", "LLaMA-3-8B");
        gguf_writer.add_u32("tokenizer.ggml.vocab_size", 128256);
        gguf_writer.add_u32("tokenizer.ggml.bos_token_id", 128000);
        gguf_writer.add_u32("tokenizer.ggml.eos_token_id", 128001);

        // Add embedded SentencePiece model
        let spm_data = create_test_sentencepiece_model();
        gguf_writer.add_bytes("tokenizer.ggml.model", &smp_data);

        // Add sample tensor metadata
        gguf_writer.add_tensor("model.embed_tokens.weight", &[128256, 4096], GgufDataType::F16);
        gguf_writer.add_tensor("model.layers.0.self_attn.q_proj.weight", &[4096, 4096], GgufDataType::F16);

        gguf_writer.finalize();
        path
    }

    fn create_test_sentencepiece_model() -> Vec<u8> {
        // Create minimal valid SentencePiece model binary
        // This would include proper protobuf serialization of sentencepiece::ModelProto
        let mut model_data = Vec::new();

        // Minimal SentencePiece header
        model_data.extend_from_slice(b"\x08\x01\x12\x04test");

        // Add vocabulary entries
        for i in 0..1000 {
            let piece = format!("token_{}", i);
            model_data.extend_from_slice(&[0x1a, piece.len() as u8]);
            model_data.extend_from_slice(piece.as_bytes());
            model_data.extend_from_slice(&[0x0d, 0x00, 0x00, 0x80, 0x3f]); // score = 1.0
        }

        model_data
    }
}
```

### 2. Advanced Model Architecture Detection

**Comprehensive Tensor Pattern Analysis**:
```rust
impl TokenizerDiscovery {
    fn extract_model_type(reader: &GgufReader) -> Result<String> {
        // Try explicit architecture metadata first
        if let Some(arch) = reader.get_string_metadata("general.architecture") {
            return Ok(arch);
        }

        // Try alternative metadata keys
        let metadata_keys = [
            "model.architecture",
            "transformer.architecture",
            "llama.architecture",
            "gpt.architecture",
            "bert.architecture",
        ];

        for key in &metadata_keys {
            if let Some(arch) = reader.get_string_metadata(key) {
                return Ok(arch);
            }
        }

        // Infer from model name patterns
        if let Some(name) = reader.get_string_metadata("general.name") {
            if let Some(inferred_type) = Self::infer_from_model_name(&name) {
                return Ok(inferred_type);
            }
        }

        // Advanced tensor pattern analysis
        Self::infer_from_tensor_patterns(reader)
    }

    fn infer_from_model_name(name: &str) -> Option<String> {
        let name_lower = name.to_lowercase();

        // LLaMA variants
        if name_lower.contains("llama") || name_lower.contains("alpaca") || name_lower.contains("vicuna") {
            return Some("llama".to_string());
        }

        // GPT variants
        if name_lower.contains("gpt") || name_lower.contains("chatgpt") {
            return Some("gpt2".to_string());
        }

        // BERT variants
        if name_lower.contains("bert") || name_lower.contains("roberta") || name_lower.contains("distilbert") {
            return Some("bert".to_string());
        }

        // T5 variants
        if name_lower.contains("t5") || name_lower.contains("flan") {
            return Some("t5".to_string());
        }

        // BitNet variants
        if name_lower.contains("bitnet") {
            return Some("bitnet".to_string());
        }

        // Falcon
        if name_lower.contains("falcon") {
            return Some("falcon".to_string());
        }

        // Mistral
        if name_lower.contains("mistral") || name_lower.contains("mixtral") {
            return Some("mistral".to_string());
        }

        None
    }

    fn infer_from_tensor_patterns(reader: &GgufReader) -> Result<String> {
        let tensor_names = reader.tensor_names();
        let patterns = TensorPatternAnalyzer::new(&tensor_names);

        // LLaMA family patterns
        if patterns.has_llama_attention_pattern() {
            return Ok("llama".to_string());
        }

        // GPT family patterns
        if patterns.has_gpt_pattern() {
            return Ok("gpt2".to_string());
        }

        // BERT family patterns
        if patterns.has_bert_pattern() {
            return Ok("bert".to_string());
        }

        // T5 family patterns
        if patterns.has_t5_pattern() {
            return Ok("t5".to_string());
        }

        // Falcon patterns
        if patterns.has_falcon_pattern() {
            return Ok("falcon".to_string());
        }

        // BitNet specific patterns
        if patterns.has_bitnet_pattern() {
            return Ok("bitnet".to_string());
        }

        // Default fallback
        Ok("transformer".to_string())
    }
}

struct TensorPatternAnalyzer<'a> {
    tensor_names: &'a [String],
}

impl<'a> TensorPatternAnalyzer<'a> {
    fn new(tensor_names: &'a [String]) -> Self {
        Self { tensor_names }
    }

    fn has_llama_attention_pattern(&self) -> bool {
        self.has_patterns(&[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ]) && self.has_patterns(&[
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ])
    }

    fn has_gpt_pattern(&self) -> bool {
        self.has_patterns(&[
            "attn.c_attn",
            "attn.c_proj",
            "mlp.c_fc",
            "mlp.c_proj",
        ]) || self.has_patterns(&[
            "attn.qkv_proj",
            "attn.out_proj",
        ])
    }

    fn has_bert_pattern(&self) -> bool {
        self.has_patterns(&[
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
        ]) && self.has_patterns(&[
            "intermediate.dense",
            "output.dense",
        ])
    }

    fn has_t5_pattern(&self) -> bool {
        self.has_patterns(&[
            "SelfAttention.q",
            "SelfAttention.k",
            "SelfAttention.v",
            "SelfAttention.o",
        ]) && (
            self.has_patterns(&["EncDecAttention"]) ||
            self.has_patterns(&["DenseReluDense"])
        )
    }

    fn has_falcon_pattern(&self) -> bool {
        self.has_patterns(&[
            "self_attention.query_key_value",
            "self_attention.dense",
        ]) && self.has_patterns(&[
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        ])
    }

    fn has_bitnet_pattern(&self) -> bool {
        self.has_patterns(&[
            "bitlinear",
            "activation_quant",
            "weight_quant",
        ]) || self.tensor_names.iter().any(|name| {
            name.contains("i2_s") || name.contains("tl1") || name.contains("tl2")
        })
    }

    fn has_patterns(&self, patterns: &[&str]) -> bool {
        patterns.iter().all(|pattern| {
            self.tensor_names.iter().any(|name| name.contains(pattern))
        })
    }
}
```

### 3. Robust Embedded Tokenizer Parsing

**Full SentencePiece and HuggingFace Support**:
```rust
impl TokenizerDiscovery {
    pub fn try_extract_embedded_tokenizer(&self) -> Result<Option<Arc<dyn Tokenizer + Send + Sync>>> {
        debug!("Attempting to extract embedded tokenizer from GGUF metadata");

        // Priority 1: HuggingFace tokenizer.json (most complete)
        if let Some(tokenizer_json) = self.gguf_reader.get_string_metadata("tokenizer.json") {
            debug!("Found embedded tokenizer.json ({} chars)", tokenizer_json.len());
            return self.parse_huggingface_tokenizer(&tokenizer_json);
        }

        // Priority 2: SentencePiece model bytes
        if let Some(spm_bytes) = self.gguf_reader.get_array_metadata("tokenizer.ggml.model") {
            debug!("Found embedded SentencePiece model ({} bytes)", smp_bytes.len());
            return self.parse_sentencepiece_tokenizer(&spm_bytes);
        }

        // Priority 3: Embedded vocabulary with score arrays
        if let Some(vocab) = self.try_extract_vocabulary_tokenizer()? {
            debug!("Created tokenizer from embedded vocabulary");
            return Ok(Some(vocab));
        }

        // Priority 4: Legacy GGML tokenizer formats
        if let Some(legacy) = self.try_extract_legacy_tokenizer()? {
            debug!("Created tokenizer from legacy format");
            return Ok(Some(legacy));
        }

        debug!("No embedded tokenizer found in GGUF metadata");
        Ok(None)
    }

    fn parse_huggingface_tokenizer(&self, json_str: &str) -> Result<Option<Arc<dyn Tokenizer + Send + Sync>>> {
        use serde_json::Value;

        let tokenizer_config: Value = serde_json::from_str(json_str)
            .context("Failed to parse tokenizer.json")?;

        // Extract tokenizer type
        let tokenizer_type = tokenizer_config
            .get("model")
            .and_then(|m| m.get("type"))
            .and_then(|t| t.as_str())
            .unwrap_or("BPE");

        match tokenizer_type {
            "BPE" => self.create_bpe_tokenizer(&tokenizer_config),
            "WordPiece" => self.create_wordpiece_tokenizer(&tokenizer_config),
            "SentencePiece" => self.create_sp_from_json(&tokenizer_config),
            _ => {
                warn!("Unknown tokenizer type: {}", tokenizer_type);
                Ok(None)
            }
        }
    }

    fn create_bpe_tokenizer(&self, config: &Value) -> Result<Option<Arc<dyn Tokenizer + Send + Sync>>> {
        // Extract vocabulary
        let vocab = config
            .get("model")
            .and_then(|m| m.get("vocab"))
            .context("Missing vocabulary in BPE tokenizer")?;

        // Extract merges
        let merges = config
            .get("model")
            .and_then(|m| m.get("merges"))
            .and_then(|m| m.as_array())
            .context("Missing merges in BPE tokenizer")?;

        // Extract special tokens
        let special_tokens = self.extract_special_tokens(config)?;

        let bpe_tokenizer = BpeTokenizer::new(
            vocab.clone(),
            merges.clone(),
            special_tokens,
        )?;

        Ok(Some(Arc::new(bpe_tokenizer)))
    }

    fn parse_sentencepiece_tokenizer(&self, smp_bytes: &[u8]) -> Result<Option<Arc<dyn Tokenizer + Send + Sync>>> {
        // Parse SentencePiece protobuf
        let model_proto = sentencepiece::ModelProto::parse_from_bytes(smp_bytes)
            .context("Failed to parse SentencePiece model")?;

        // Extract special token IDs from GGUF metadata
        let bos_id = self.gguf_reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
        let eos_id = self.gguf_reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
        let pad_id = self.gguf_reader.get_u32_metadata("tokenizer.ggml.pad_token_id");
        let unk_id = self.gguf_reader.get_u32_metadata("tokenizer.ggml.unk_token_id");

        let special_tokens = SpecialTokens {
            bos_token_id: bos_id,
            eos_token_id: eos_id,
            pad_token_id: pad_id,
            unk_token_id: unk_id,
        };

        let sp_tokenizer = SpTokenizer::from_model_proto(model_proto, special_tokens)?;
        Ok(Some(Arc::new(sp_tokenizer)))
    }

    fn try_extract_vocabulary_tokenizer(&self) -> Result<Option<Arc<dyn Tokenizer + Send + Sync>>> {
        // Try to get vocabulary tokens
        let vocab_tokens = match self.gguf_reader.get_string_array_metadata("tokenizer.ggml.tokens") {
            Some(tokens) => tokens,
            None => return Ok(None),
        };

        // Get token scores if available
        let token_scores = self.gguf_reader.get_f32_array_metadata("tokenizer.ggml.scores")
            .unwrap_or_else(|| vec![0.0; vocab_tokens.len()]);

        // Get token types if available (for special token identification)
        let token_types = self.gguf_reader.get_i32_array_metadata("tokenizer.ggml.token_type")
            .unwrap_or_else(|| vec![0; vocab_tokens.len()]);

        // Extract special tokens from metadata
        let special_tokens = SpecialTokens {
            bos_token_id: self.gguf_reader.get_u32_metadata("tokenizer.ggml.bos_token_id"),
            eos_token_id: self.gguf_reader.get_u32_metadata("tokenizer.ggml.eos_token_id"),
            pad_token_id: self.gguf_reader.get_u32_metadata("tokenizer.ggml.pad_token_id"),
            unk_token_id: self.gguf_reader.get_u32_metadata("tokenizer.ggml.unk_token_id"),
        };

        let vocab_tokenizer = VocabularyTokenizer::new(
            vocab_tokens,
            token_scores,
            token_types,
            special_tokens,
        )?;

        Ok(Some(Arc::new(vocab_tokenizer)))
    }

    fn extract_special_tokens(&self, config: &Value) -> Result<SpecialTokens> {
        let added_tokens = config
            .get("added_tokens")
            .and_then(|t| t.as_array())
            .unwrap_or(&vec![]);

        let mut special_tokens = SpecialTokens::default();

        for token in added_tokens {
            if let (Some(content), Some(id)) = (
                token.get("content").and_then(|c| c.as_str()),
                token.get("id").and_then(|i| i.as_u64())
            ) {
                match content {
                    "<s>" | "<bos>" => special_tokens.bos_token_id = Some(id as u32),
                    "</s>" | "<eos>" => special_tokens.eos_token_id = Some(id as u32),
                    "<pad>" => special_tokens.pad_token_id = Some(id as u32),
                    "<unk>" => special_tokens.unk_token_id = Some(id as u32),
                    _ => {}
                }
            }
        }

        Ok(special_tokens)
    }
}
```

## Implementation Plan

### Phase 1: Test Infrastructure Development (Week 1-2)
- [ ] Create `GgufWriter` helper for test GGUF file generation
- [ ] Implement comprehensive test cases for LLaMA-3, GPT-2, BERT models
- [ ] Add test utilities for SentencePiece model generation
- [ ] Create test fixtures for various tokenizer formats

### Phase 2: Advanced Model Detection (Week 2-3)
- [ ] Implement `TensorPatternAnalyzer` with comprehensive architecture detection
- [ ] Add support for Falcon, Mistral, T5, and other popular architectures
- [ ] Create model name inference patterns for all major model families
- [ ] Add BitNet-specific pattern detection

### Phase 3: SentencePiece Integration (Week 3-4)
- [ ] Integrate `sentencepiece` crate for protobuf parsing
- [ ] Implement `SpTokenizer::from_model_proto()` constructor
- [ ] Add proper error handling for malformed SentencePiece data
- [ ] Create comprehensive SentencePiece test cases

### Phase 4: HuggingFace Tokenizer Support (Week 4-5)
- [ ] Implement BPE tokenizer parsing from tokenizer.json
- [ ] Add WordPiece tokenizer support
- [ ] Create unified special token extraction
- [ ] Add validation for tokenizer consistency

### Phase 5: Vocabulary Tokenizer Implementation (Week 5-6)
- [ ] Implement `VocabularyTokenizer` for embedded vocabularies
- [ ] Add token score and type handling
- [ ] Create fallback mechanisms for incomplete metadata
- [ ] Add legacy GGML format support

### Phase 6: Integration and Validation (Week 6-7)
- [ ] Integrate all tokenizer parsers into discovery system
- [ ] Add cross-validation with reference implementations
- [ ] Create comprehensive test suite for all supported formats
- [ ] Add performance benchmarking and optimization

## Testing Strategy

### Comprehensive Model Testing
```rust
#[test]
fn test_model_architecture_detection_comprehensive() {
    let test_cases = vec![
        ("llama3-8b.gguf", "llama"),
        ("gpt2-medium.gguf", "gpt2"),
        ("bert-base.gguf", "bert"),
        ("t5-large.gguf", "t5"),
        ("falcon-7b.gguf", "falcon"),
        ("mistral-7b.gguf", "mistral"),
        ("bitnet-b1.58-3b.gguf", "bitnet"),
    ];

    for (filename, expected_arch) in test_cases {
        let test_gguf = create_test_gguf_for_architecture(expected_arch);
        let discovery = TokenizerDiscovery::from_gguf(&test_gguf).unwrap();
        assert_eq!(discovery.model_type(), expected_arch);
    }
}

#[test]
fn test_embedded_tokenizer_parsing_formats() {
    // Test SentencePiece parsing
    let spm_gguf = create_test_gguf_with_sentencepiece();
    let discovery = TokenizerDiscovery::from_gguf(&spm_gguf).unwrap();
    let tokenizer = discovery.try_extract_embedded_tokenizer().unwrap().unwrap();

    // Verify tokenization works
    let tokens = tokenizer.tokenize("Hello 🌍 world!").unwrap();
    let reconstructed = tokenizer.detokenize(&tokens).unwrap();
    assert!(reconstructed.contains("Hello") && reconstructed.contains("world"));

    // Test HuggingFace JSON parsing
    let hf_gguf = create_test_gguf_with_hf_tokenizer();
    let discovery = TokenizerDiscovery::from_gguf(&hf_gguf).unwrap();
    let tokenizer = discovery.try_extract_embedded_tokenizer().unwrap().unwrap();

    // Verify BPE tokenization
    let tokens = tokenizer.tokenize("Hello world").unwrap();
    assert!(tokens.len() > 0);
}
```

### Error Handling and Edge Cases
```rust
#[test]
fn test_malformed_gguf_handling() {
    let malformed_cases = vec![
        create_gguf_with_invalid_sentencepiece(),
        create_gguf_with_malformed_json(),
        create_gguf_with_inconsistent_vocab_size(),
        create_gguf_with_missing_special_tokens(),
    ];

    for malformed_gguf in malformed_cases {
        let result = TokenizerDiscovery::from_gguf(&malformed_gguf);

        // Should either succeed with fallback or fail gracefully
        match result {
            Ok(discovery) => {
                // If successful, should have reasonable defaults
                assert!(discovery.vocab_size() > 0);
                assert!(!discovery.model_type().is_empty());
            }
            Err(e) => {
                // Error should be descriptive
                let error_msg = e.to_string();
                assert!(error_msg.contains("GGUF") || error_msg.contains("tokenizer"));
            }
        }
    }
}
```

### Cross-Validation Testing
```rust
#[test]
#[cfg(feature = "crossval")]
fn test_tokenizer_consistency_with_reference() {
    let test_texts = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "🌍 Unicode test with émojis and àccents.",
        "Code: fn main() { println!(\"Hello\"); }",
    ];

    let discovery = TokenizerDiscovery::from_gguf(&reference_model_path()).unwrap();
    let bitnet_tokenizer = discovery.try_extract_embedded_tokenizer().unwrap().unwrap();
    let reference_tokenizer = load_reference_tokenizer();

    for text in test_texts {
        let bitnet_tokens = bitnet_tokenizer.tokenize(text).unwrap();
        let reference_tokens = reference_tokenizer.tokenize(text).unwrap();

        // Allow for minor differences in tokenization
        assert_token_similarity(&bitnet_tokens, &reference_tokens, 0.95);
    }
}
```

## Risk Assessment

### Implementation Risks
1. **SentencePiece Dependency**: Adding protobuf parsing increases complexity
   - *Mitigation*: Use well-tested `prost` crate, comprehensive error handling
2. **GGUF Format Changes**: Format evolution may break parsing
   - *Mitigation*: Version detection, backward compatibility, graceful degradation
3. **Memory Usage**: Large embedded tokenizers may consume significant memory
   - *Mitigation*: Lazy loading, streaming parsing, memory pooling

### Compatibility Risks
1. **Model Variants**: Different implementations may use different metadata keys
   - *Mitigation*: Comprehensive fallback chains, extensive testing with real models
2. **Tokenizer Differences**: Subtle differences between tokenizer implementations
   - *Mitigation*: Cross-validation testing, reference implementation comparison

## Acceptance Criteria

### Functional Requirements
- [ ] All test placeholders replaced with comprehensive test implementations
- [ ] Model architecture detection supports 10+ major transformer families
- [ ] SentencePiece model parsing with full protobuf support
- [ ] HuggingFace tokenizer.json parsing for BPE and WordPiece
- [ ] Embedded vocabulary tokenizer with score and type handling

### Quality Requirements
- [ ] >95% accuracy in model architecture detection on diverse model set
- [ ] Tokenization consistency within 95% of reference implementations
- [ ] Graceful error handling for all malformed input cases
- [ ] Comprehensive test coverage >90% for all discovery paths
- [ ] Cross-validation passes with Microsoft BitNet C++ reference

### Performance Requirements
- [ ] Model detection completes within 100ms for typical GGUF files
- [ ] Embedded tokenizer parsing within 500ms for largest models
- [ ] Memory usage <50MB for tokenizer discovery process
- [ ] Zero-copy parsing where possible to minimize allocations

## Related Issues

- BitNet-rs #251: Production-ready inference server (depends on robust tokenizer loading)
- BitNet-rs #218: Device-aware quantization system (benefits from better model detection)
- BitNet-rs #260: Mock elimination project (uses improved test infrastructure)

## Implementation Notes

### BitNet-rs Integration
- Use existing `bitnet-models::GgufReader` as foundation
- Integrate with `bitnet-tokenizers` trait system
- Follow feature flag architecture (`--features spm` for SentencePiece support)
- Maintain compatibility with existing tokenizer discovery patterns

### Dependencies
- Add `prost` for SentencePiece protobuf parsing
- Use `serde_json` for HuggingFace tokenizer.json parsing
- Integrate `tempfile` for test infrastructure
- Leverage existing `anyhow` error handling patterns
