# [Stub Implementation] TokenizerDiscovery::try_extract_embedded_tokenizer creates BasicTokenizer instead of proper tokenizer

## Problem Description

The `TokenizerDiscovery::try_extract_embedded_tokenizer` function in `crates/bitnet-tokenizers/src/discovery.rs` contains stub implementations that create `BasicTokenizer` instances instead of parsing the actual embedded tokenizer data (JSON for HuggingFace tokenizers or SentencePiece model bytes). This prevents proper tokenizer functionality with models that have embedded tokenizers.

## Environment

- **File**: `crates/bitnet-tokenizers/src/discovery.rs`
- **Function**: `TokenizerDiscovery::try_extract_embedded_tokenizer`
- **Crate**: `bitnet-tokenizers`
- **Related Components**: GGUF metadata parsing, tokenizer creation, model loading

## Current Implementation Analysis

The function contains multiple stub implementations with explicit comments:

```rust
pub fn try_extract_embedded_tokenizer(&self) -> Result<Option<Arc<dyn Tokenizer>>> {
    debug!("Attempting to extract embedded tokenizer from GGUF metadata");

    // Check if tokenizer model is embedded as bytes
    if let Some(tokenizer_model) = self.gguf_reader.get_array_metadata("tokenizer.ggml.model") {
        debug!("Found embedded tokenizer.ggml.model ({} bytes)", tokenizer_model.len());

        // Try to create tokenizer from the embedded data
        // This is a simplified implementation - in production this would parse the model format
        if tokenizer_model.len() > 100 {
            // Sanity check for reasonable size
            let basic_tokenizer = crate::BasicTokenizer::with_config(
                self.vocab_size,
                Some(1), // BOS token
                Some(2), // EOS token
                Some(0), // PAD token
            );

            debug!("Created basic tokenizer from GGUF metadata");
            return Ok(Some(Arc::new(basic_tokenizer)));
        }
    }

    // Check for tokenizer vocab embedded in metadata
    if let Some(vocab) = self.gguf_reader.get_string_array_metadata("tokenizer.ggml.tokens")
        && vocab.len() == self.vocab_size
    {
        debug!("Found embedded vocabulary with {} tokens", vocab.len());

        // Create tokenizer with embedded vocabulary
        let basic_tokenizer = crate::BasicTokenizer::with_config(
            self.vocab_size,
            self.gguf_reader.get_u32_metadata("tokenizer.ggml.bos_token_id"),
            self.gguf_reader.get_u32_metadata("tokenizer.ggml.eos_token_id"),
            self.gguf_reader.get_u32_metadata("tokenizer.ggml.pad_token_id"),
        );

        debug!("Created tokenizer with embedded vocabulary");
        return Ok(Some(Arc::new(basic_tokenizer)));
    }

    // Check for HuggingFace tokenizer.json embedded as string
    if let Some(tokenizer_json) = self.gguf_reader.get_string_metadata("tokenizer.json") {
        debug!("Found embedded tokenizer.json ({} chars)", tokenizer_json.len());

        // In production, this would parse the JSON and create an HfTokenizer
        // For now, create a basic tokenizer with inferred parameters
        let basic_tokenizer = crate::BasicTokenizer::with_config(
            self.vocab_size,
            Some(1), // BOS token
            Some(2), // EOS token
            Some(0), // PAD token
        );

        debug!("Created tokenizer from embedded JSON metadata");
        return Ok(Some(Arc::new(basic_tokenizer)));
    }

    debug!("No embedded tokenizer found in GGUF metadata");
    Ok(None)
}
```

## Root Cause Analysis

1. **Incomplete Parsing Implementation**: Function doesn't parse actual embedded tokenizer data
2. **Basic Tokenizer Fallback**: Always creates `BasicTokenizer` instead of proper tokenizer types
3. **Missing JSON Parsing**: HuggingFace `tokenizer.json` data isn't parsed and used
4. **SentencePiece Stub**: Embedded SentencePiece model bytes aren't processed
5. **Metadata Ignorance**: Rich tokenizer metadata in GGUF is ignored

## Impact Assessment

**Severity**: High - Functionality Critical
**Affected Components**:
- Model loading with embedded tokenizers
- Text processing accuracy and compatibility
- Production model deployment
- Cross-model tokenizer compatibility

**Functional Impact**:
- **Incorrect Tokenization**: Basic tokenizer doesn't match model's expected tokenization
- **Vocabulary Mismatches**: Wrong token IDs lead to poor model performance
- **Special Token Errors**: Incorrect BOS/EOS/PAD tokens affect generation quality
- **Production Failures**: Models with embedded tokenizers fail to work properly

## Proposed Solution

### Primary Implementation: Parse Embedded Tokenizer Data

Replace stub implementations with proper parsing and tokenizer creation:

```rust
use serde_json;
use crate::{HfTokenizer, SpTokenizer, SpmTokenizer};

pub fn try_extract_embedded_tokenizer(&self) -> Result<Option<Arc<dyn Tokenizer>>> {
    debug!("Attempting to extract embedded tokenizer from GGUF metadata");

    // Priority 1: HuggingFace tokenizer.json (most complete format)
    if let Some(tokenizer_json) = self.gguf_reader.get_string_metadata("tokenizer.json") {
        debug!("Found embedded tokenizer.json ({} chars)", tokenizer_json.len());
        return self.parse_hf_tokenizer_json(&tokenizer_json);
    }

    // Priority 2: SentencePiece model bytes
    if let Some(tokenizer_model_bytes) = self.gguf_reader.get_array_metadata("tokenizer.ggml.model") {
        debug!("Found embedded tokenizer.ggml.model ({} bytes)", tokenizer_model_bytes.len());
        return self.parse_sentencepiece_model(&tokenizer_model_bytes);
    }

    // Priority 3: Vocabulary-based tokenizer (GGML format)
    if let Some(vocab) = self.gguf_reader.get_string_array_metadata("tokenizer.ggml.tokens") {
        if vocab.len() == self.vocab_size {
            debug!("Found embedded vocabulary with {} tokens", vocab.len());
            return self.create_vocab_based_tokenizer(&vocab);
        }
    }

    debug!("No embedded tokenizer found in GGUF metadata");
    Ok(None)
}

fn parse_hf_tokenizer_json(&self, tokenizer_json: &str) -> Result<Option<Arc<dyn Tokenizer>>> {
    // Parse HuggingFace tokenizer.json format
    let tokenizer_config: serde_json::Value = serde_json::from_str(tokenizer_json)
        .context("Failed to parse tokenizer.json")?;

    // Extract tokenizer type and configuration
    let tokenizer_type = tokenizer_config
        .get("model")
        .and_then(|m| m.get("type"))
        .and_then(|t| t.as_str())
        .unwrap_or("unknown");

    debug!("Detected HuggingFace tokenizer type: {}", tokenizer_type);

    match tokenizer_type {
        "BPE" => self.create_hf_bpe_tokenizer(&tokenizer_config),
        "WordPiece" => self.create_hf_wordpiece_tokenizer(&tokenizer_config),
        "Unigram" => self.create_hf_unigram_tokenizer(&tokenizer_config),
        _ => {
            warn!("Unsupported HuggingFace tokenizer type: {}", tokenizer_type);
            self.create_fallback_hf_tokenizer(&tokenizer_config)
        }
    }
}

fn create_hf_bpe_tokenizer(&self, config: &serde_json::Value) -> Result<Option<Arc<dyn Tokenizer>>> {
    // Extract BPE-specific configuration
    let vocab = self.extract_vocab_from_hf_config(config)?;
    let merges = self.extract_merges_from_hf_config(config)?;
    let special_tokens = self.extract_special_tokens_from_hf_config(config)?;

    let tokenizer = HfTokenizer::new_bpe(
        vocab,
        merges,
        special_tokens.bos_token,
        special_tokens.eos_token,
        special_tokens.pad_token,
        special_tokens.unk_token,
    )?;

    debug!("Created HuggingFace BPE tokenizer from embedded JSON");
    Ok(Some(Arc::new(tokenizer)))
}

fn parse_sentencepiece_model(&self, model_bytes: &[u8]) -> Result<Option<Arc<dyn Tokenizer>>> {
    // Parse SentencePiece model binary format
    debug!("Parsing SentencePiece model ({} bytes)", model_bytes.len());

    // Extract special token IDs from GGUF metadata
    let bos_token_id = self.gguf_reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
    let eos_token_id = self.gguf_reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
    let pad_token_id = self.gguf_reader.get_u32_metadata("tokenizer.ggml.pad_token_id");
    let unk_token_id = self.gguf_reader.get_u32_metadata("tokenizer.ggml.unk_token_id");

    // Create SentencePiece tokenizer from binary model
    let sp_tokenizer = SpTokenizer::from_bytes(
        model_bytes,
        bos_token_id,
        eos_token_id,
        pad_token_id,
        unk_token_id,
    )?;

    debug!("Created SentencePiece tokenizer from embedded model");
    Ok(Some(Arc::new(sp_tokenizer)))
}

fn create_vocab_based_tokenizer(&self, vocab: &[String]) -> Result<Option<Arc<dyn Tokenizer>>> {
    // Create tokenizer from vocabulary list and scores
    let scores = self.gguf_reader
        .get_f32_array_metadata("tokenizer.ggml.scores")
        .unwrap_or_else(|| vec![0.0; vocab.len()]);

    let token_types = self.gguf_reader
        .get_i32_array_metadata("tokenizer.ggml.token_type")
        .unwrap_or_else(|| vec![0; vocab.len()]);

    // Extract special token IDs
    let bos_token_id = self.gguf_reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
    let eos_token_id = self.gguf_reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
    let pad_token_id = self.gguf_reader.get_u32_metadata("tokenizer.ggml.pad_token_id");
    let unk_token_id = self.gguf_reader.get_u32_metadata("tokenizer.ggml.unk_token_id");

    // Determine tokenizer type from GGUF metadata
    let tokenizer_model = self.gguf_reader
        .get_string_metadata("tokenizer.ggml.model")
        .unwrap_or_else(|| "llama".to_string());

    let tokenizer = match tokenizer_model.as_str() {
        "llama" => {
            SpmTokenizer::from_vocab_and_scores(
                vocab.to_vec(),
                scores,
                bos_token_id,
                eos_token_id,
                pad_token_id,
                unk_token_id,
            )?
        }
        "gpt2" => {
            // GPT-2 style BPE tokenizer
            let merges = self.extract_merges_from_gguf_metadata()?;
            HfTokenizer::new_gpt2_style(
                vocab.to_vec(),
                merges,
                bos_token_id,
                eos_token_id,
                pad_token_id,
            )?
        }
        _ => {
            warn!("Unknown tokenizer model type: {}, using SentencePiece fallback", tokenizer_model);
            SpmTokenizer::from_vocab_and_scores(
                vocab.to_vec(),
                scores,
                bos_token_id,
                eos_token_id,
                pad_token_id,
                unk_token_id,
            )?
        }
    };

    debug!("Created {} tokenizer from embedded vocabulary", tokenizer_model);
    Ok(Some(Arc::new(tokenizer)))
}

#[derive(Debug)]
struct SpecialTokens {
    bos_token: Option<u32>,
    eos_token: Option<u32>,
    pad_token: Option<u32>,
    unk_token: Option<u32>,
}

fn extract_special_tokens_from_hf_config(&self, config: &serde_json::Value) -> Result<SpecialTokens> {
    let special_tokens = config
        .get("added_tokens")
        .or_else(|| config.get("special_tokens_map"))
        .unwrap_or(&serde_json::Value::Null);

    Ok(SpecialTokens {
        bos_token: self.extract_token_id(special_tokens, "bos_token"),
        eos_token: self.extract_token_id(special_tokens, "eos_token"),
        pad_token: self.extract_token_id(special_tokens, "pad_token"),
        unk_token: self.extract_token_id(special_tokens, "unk_token"),
    })
}

fn extract_vocab_from_hf_config(&self, config: &serde_json::Value) -> Result<HashMap<String, u32>> {
    let vocab_obj = config
        .get("model")
        .and_then(|m| m.get("vocab"))
        .context("No vocabulary found in HuggingFace config")?;

    let mut vocab = HashMap::new();
    if let Some(vocab_map) = vocab_obj.as_object() {
        for (token, id) in vocab_map {
            if let Some(id_num) = id.as_u64() {
                vocab.insert(token.clone(), id_num as u32);
            }
        }
    }

    Ok(vocab)
}

fn extract_merges_from_hf_config(&self, config: &serde_json::Value) -> Result<Vec<(String, String)>> {
    let merges_array = config
        .get("model")
        .and_then(|m| m.get("merges"))
        .and_then(|m| m.as_array())
        .context("No merges found in HuggingFace config")?;

    let mut merges = Vec::new();
    for merge in merges_array {
        if let Some(merge_str) = merge.as_str() {
            let parts: Vec<&str> = merge_str.split_whitespace().collect();
            if parts.len() == 2 {
                merges.push((parts[0].to_string(), parts[1].to_string()));
            }
        }
    }

    Ok(merges)
}
```

### Alternative Approach: Tokenizer Factory Pattern

Use factory pattern for extensible tokenizer creation:

```rust
pub struct TokenizerFactory;

impl TokenizerFactory {
    pub fn create_from_gguf_metadata(
        gguf_reader: &GgufReader,
        vocab_size: usize,
    ) -> Result<Option<Arc<dyn Tokenizer>>> {
        // Try different tokenizer extraction methods in priority order
        if let Some(tokenizer) = Self::try_hf_tokenizer_json(gguf_reader, vocab_size)? {
            return Ok(Some(tokenizer));
        }

        if let Some(tokenizer) = Self::try_sentencepiece_model(gguf_reader, vocab_size)? {
            return Ok(Some(tokenizer));
        }

        if let Some(tokenizer) = Self::try_vocab_based_tokenizer(gguf_reader, vocab_size)? {
            return Ok(Some(tokenizer));
        }

        Ok(None)
    }

    fn try_hf_tokenizer_json(
        gguf_reader: &GgufReader,
        vocab_size: usize,
    ) -> Result<Option<Arc<dyn Tokenizer>>> {
        // Implementation for HuggingFace tokenizer.json parsing
        todo!()
    }

    // ... other methods
}
```

## Implementation Plan

### Phase 1: Core Parsing Infrastructure
- [ ] Implement HuggingFace tokenizer.json parsing
- [ ] Add SentencePiece binary model parsing
- [ ] Create vocabulary-based tokenizer construction
- [ ] Add comprehensive error handling and validation

### Phase 2: Tokenizer Type Support
- [ ] Implement BPE tokenizer creation from JSON
- [ ] Add WordPiece tokenizer support
- [ ] Implement Unigram/SentencePiece tokenizer creation
- [ ] Add GPT-2 style tokenizer support

### Phase 3: Integration & Testing
- [ ] Integrate with existing tokenizer discovery pipeline
- [ ] Add comprehensive testing with real model files
- [ ] Validate tokenization accuracy against reference implementations
- [ ] Performance benchmark tokenizer creation time

### Phase 4: Fallbacks & Compatibility
- [ ] Add graceful fallbacks for unsupported formats
- [ ] Implement backward compatibility with BasicTokenizer
- [ ] Add configuration options for tokenizer selection
- [ ] Create migration path for existing models

## Testing Strategy

### Functional Testing
```rust
#[test]
fn test_hf_tokenizer_json_parsing() {
    let tokenizer_json = include_str!("test_data/gpt2_tokenizer.json");
    let discovery = create_test_discovery_with_json(tokenizer_json);

    let tokenizer = discovery.try_extract_embedded_tokenizer().unwrap().unwrap();

    // Test tokenization matches expected HuggingFace behavior
    let text = "Hello, world!";
    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(text, decoded);
}

#[test]
fn test_sentencepiece_model_parsing() {
    let sp_model_bytes = include_bytes!("test_data/llama.model");
    let discovery = create_test_discovery_with_sp_model(sp_model_bytes);

    let tokenizer = discovery.try_extract_embedded_tokenizer().unwrap().unwrap();

    // Test SentencePiece tokenization
    let text = "Hello, world!";
    let tokens = tokenizer.encode(text).unwrap();
    assert!(!tokens.is_empty());
}

#[test]
fn test_vocab_based_tokenizer() {
    let vocab = vec!["<unk>", "<s>", "</s>", "hello", "world"];
    let scores = vec![0.0, 0.0, 0.0, -1.0, -1.0];
    let discovery = create_test_discovery_with_vocab(&vocab, &scores);

    let tokenizer = discovery.try_extract_embedded_tokenizer().unwrap().unwrap();

    // Test vocabulary-based tokenization
    let tokens = tokenizer.encode("hello world").unwrap();
    assert!(tokens.contains(&3)); // "hello" token
    assert!(tokens.contains(&4)); // "world" token
}
```

### Compatibility Testing
```rust
#[test]
fn test_cross_tokenizer_compatibility() {
    // Test that tokenizers created from embedded data match reference implementations
    let test_cases = [
        ("gpt2", "test_data/gpt2_model.gguf"),
        ("llama", "test_data/llama_model.gguf"),
        ("bert", "test_data/bert_model.gguf"),
    ];

    for (model_type, model_path) in test_cases {
        let discovery = TokenizerDiscovery::from_gguf_file(model_path).unwrap();
        let extracted_tokenizer = discovery.try_extract_embedded_tokenizer().unwrap();

        if let Some(tokenizer) = extracted_tokenizer {
            validate_tokenizer_accuracy(&tokenizer, model_type);
        }
    }
}
```

## Related Issues/PRs

- Tokenizer compatibility and accuracy improvements
- GGUF metadata parsing enhancements
- Model loading reliability and validation
- Cross-platform tokenizer support

## Acceptance Criteria

- [ ] HuggingFace tokenizer.json parsing implemented and functional
- [ ] SentencePiece binary model parsing working correctly
- [ ] Vocabulary-based tokenizer creation from GGUF metadata
- [ ] Tokenization accuracy matches reference implementations
- [ ] Comprehensive error handling for malformed embedded data
- [ ] Performance impact is acceptable (<100ms for tokenizer creation)
- [ ] Backward compatibility maintained with existing BasicTokenizer fallbacks
- [ ] Support for major tokenizer types (BPE, WordPiece, SentencePiece)
- [ ] Integration tests with real model files pass
- [ ] Documentation for supported embedded tokenizer formats

## Notes

This implementation is critical for proper model compatibility, especially with models that embed their tokenizers in GGUF format. The current stub implementation significantly impacts the accuracy and usability of models with embedded tokenizers.

Priority should be given to HuggingFace tokenizer.json parsing as it's the most complete format, followed by SentencePiece model support for Llama-family models.
