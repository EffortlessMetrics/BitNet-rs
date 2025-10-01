# [Tokenizer] Implement Comprehensive Token-to-Piece Mapping in GgufTokenizer

## Problem Description

The `GgufTokenizer::token_to_piece` function in `crates/bitnet-tokenizers/src/gguf_tokenizer.rs` currently implements a simplified token-to-piece conversion that falls back to byte-level mappings and basic vocabulary lookups. This approach doesn't properly handle complex tokenization schemes used by modern language models and may produce incorrect token reconstructions.

## Environment

- **Component**: `crates/bitnet-tokenizers/src/gguf_tokenizer.rs`
- **Function**: `GgufTokenizer::token_to_piece`
- **GGUF Version Support**: Compatible with GGUF v1, v2, v3
- **Affected Models**: All GGUF-based models using complex tokenization (LLaMA, GPT variants, specialized vocabularies)

## Current Implementation Analysis

```rust
fn token_to_piece(&self, token: u32) -> Option<String> {
    if let Some(&byte) = self.id_to_byte.get(&token) {
        Some(String::from_utf8_lossy(&[byte]).to_string())
    } else if let Some(piece) = self.reverse_vocab.get(&token) {
        Some(piece.clone())
    } else if token < 256 {
        Some(String::from_utf8_lossy(&[token as u8]).to_string())
    } else {
        None
    }
}
```

**Issues Identified:**
1. **Inconsistent mapping priority**: Byte-level lookup takes precedence over vocabulary lookup
2. **Fallback to raw bytes**: Assumes tokens < 256 map to single bytes, which is incorrect for many tokenizers
3. **Missing special token handling**: No support for `<BOS>`, `<EOS>`, `<UNK>`, `<PAD>` tokens
4. **No subword handling**: Doesn't handle BPE (Byte Pair Encoding) or SentencePiece token reconstruction
5. **Incomplete vocabulary utilization**: Doesn't leverage full GGUF vocabulary metadata

## Impact Assessment

**Severity**: High
**Affected Users**: All users using GGUF models with complex tokenization schemes
**Functional Impact**:
- Incorrect text generation due to wrong token-to-text conversion
- Broken detokenization leading to garbled output
- Incompatibility with advanced tokenizers (SentencePiece, BPE)
- Failed cross-validation with reference implementations

## Root Cause Analysis

The current implementation appears to be a placeholder that handles only the most basic tokenization scenarios. Modern language models use sophisticated tokenization schemes that require:

1. **Proper vocabulary ordering**: Special tokens, regular tokens, and byte fallbacks have specific precedence
2. **Subword reconstruction**: BPE and SentencePiece tokens may need special handling
3. **Unicode handling**: Proper UTF-8 reconstruction from token sequences
4. **Special token recognition**: Model-specific special tokens with semantic meaning

## Proposed Solution

### 1. Enhanced Token-to-Piece Architecture

Implement a comprehensive tokenization system that:
- Properly prioritizes vocabulary lookup over byte fallbacks
- Handles special tokens with semantic meaning
- Supports multiple tokenization algorithms
- Maintains compatibility with GGUF vocabulary formats

### 2. Implementation Plan

```rust
impl GgufTokenizer {
    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.token_to_piece_with_type(token).map(|(piece, _)| piece)
    }

    fn token_to_piece_with_type(&self, token: u32) -> Option<(String, TokenType)> {
        // Priority 1: Special tokens (highest priority)
        if let Some(special_piece) = self.special_tokens.get(&token) {
            return Some((special_piece.clone(), TokenType::Special));
        }

        // Priority 2: Regular vocabulary tokens
        if let Some(piece) = self.reverse_vocab.get(&token) {
            let token_type = self.classify_token_type(piece);
            return Some((piece.clone(), token_type));
        }

        // Priority 3: Added tokens (user-defined tokens)
        if let Some(added_piece) = self.added_tokens.get(&token) {
            return Some((added_piece.clone(), TokenType::Added));
        }

        // Priority 4: Byte fallback (lowest priority, only for byte-level tokenizers)
        if self.is_byte_level_tokenizer() && token < 256 {
            let byte_piece = self.byte_to_piece(token as u8)?;
            return Some((byte_piece, TokenType::Byte));
        }

        // Priority 5: Unknown token handling
        if let Some(unk_token) = &self.unk_token {
            Some((unk_token.clone(), TokenType::Unknown))
        } else {
            None
        }
    }

    fn classify_token_type(&self, piece: &str) -> TokenType {
        if piece.starts_with("▁") || piece.starts_with("##") {
            TokenType::Subword
        } else if piece.chars().all(|c| c.is_ascii_graphic()) {
            TokenType::Word
        } else {
            TokenType::Other
        }
    }

    fn byte_to_piece(&self, byte: u8) -> Option<String> {
        match self.tokenizer_type {
            TokenizerType::ByteLevel => {
                // Use byte-level BPE encoding
                self.byte_decoder.get(&(byte as u32))
                    .cloned()
                    .or_else(|| Some(format!("<0x{:02X}>", byte)))
            }
            TokenizerType::SentencePiece => {
                // SentencePiece doesn't use byte fallbacks
                None
            }
            TokenizerType::WordLevel => {
                // Word-level tokenizers don't use byte fallbacks
                None
            }
        }
    }

    fn is_byte_level_tokenizer(&self) -> bool {
        matches!(self.tokenizer_type, TokenizerType::ByteLevel)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenType {
    Special,   // <BOS>, <EOS>, <PAD>, etc.
    Word,      // Complete words
    Subword,   // BPE/SentencePiece subwords
    Added,     // User-defined tokens
    Byte,      // Byte-level fallback
    Unknown,   // Fallback to UNK
    Other,     // Other token types
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerType {
    ByteLevel,     // GPT-2 style BPE
    SentencePiece, // LLaMA style
    WordLevel,     // Traditional word-based
}
```

### 3. Enhanced GGUF Vocabulary Loading

```rust
impl GgufTokenizer {
    pub fn from_gguf_reader(reader: &mut GgufReader) -> Result<Self> {
        let mut tokenizer = Self::new();

        // Load tokenizer type from metadata
        tokenizer.tokenizer_type = Self::detect_tokenizer_type(reader)?;

        // Load vocabulary with proper ordering
        tokenizer.load_vocabulary(reader)?;

        // Load special tokens
        tokenizer.load_special_tokens(reader)?;

        // Load added tokens
        tokenizer.load_added_tokens(reader)?;

        // Initialize byte decoder for byte-level tokenizers
        if tokenizer.is_byte_level_tokenizer() {
            tokenizer.initialize_byte_decoder()?;
        }

        Ok(tokenizer)
    }

    fn detect_tokenizer_type(reader: &GgufReader) -> Result<TokenizerType> {
        // Check GGUF metadata for tokenizer type hints
        if let Some(tokenizer_model) = reader.get_metadata_value("tokenizer.ggml.model")? {
            match tokenizer_model.as_str()? {
                "gpt2" | "gpt-2" => Ok(TokenizerType::ByteLevel),
                "llama" | "sentencepiece" => Ok(TokenizerType::SentencePiece),
                "bert" | "word-level" => Ok(TokenizerType::WordLevel),
                unknown => {
                    warn!("Unknown tokenizer model: {}, defaulting to SentencePiece", unknown);
                    Ok(TokenizerType::SentencePiece)
                }
            }
        } else {
            // Fallback detection based on vocabulary characteristics
            Self::detect_from_vocabulary(reader)
        }
    }

    fn detect_from_vocabulary(reader: &GgufReader) -> Result<TokenizerType> {
        let vocab_size = reader.get_metadata_value("tokenizer.ggml.tokens")?.len()?;
        let tokens = reader.get_string_array("tokenizer.ggml.tokens")?;

        // Check for SentencePiece indicators
        let sentencepiece_indicators = tokens.iter()
            .take(100)
            .filter(|token| token.starts_with("▁"))
            .count();

        if sentencepiece_indicators > 10 {
            return Ok(TokenizerType::SentencePiece);
        }

        // Check for BPE indicators
        let bpe_indicators = tokens.iter()
            .take(100)
            .filter(|token| token.starts_with("##") || token.len() == 1)
            .count();

        if bpe_indicators > 20 {
            return Ok(TokenizerType::ByteLevel);
        }

        // Default fallback
        Ok(TokenizerType::SentencePiece)
    }

    fn load_vocabulary(&mut self, reader: &GgufReader) -> Result<()> {
        let tokens = reader.get_string_array("tokenizer.ggml.tokens")?;
        let scores = reader.get_f32_array("tokenizer.ggml.scores")
            .unwrap_or_else(|_| vec![0.0; tokens.len()]);

        // Build reverse vocabulary with proper token ID assignment
        for (id, token) in tokens.iter().enumerate() {
            let token_id = id as u32;
            self.reverse_vocab.insert(token_id, token.clone());
            self.vocab.insert(token.clone(), token_id);
        }

        // Store scores for ranking/probability calculations
        self.token_scores = scores;

        Ok(())
    }

    fn load_special_tokens(&mut self, reader: &GgufReader) -> Result<()> {
        // Load standard special tokens
        if let Ok(bos_token_id) = reader.get_metadata_value("tokenizer.ggml.bos_token_id") {
            if let Some(bos_token) = self.reverse_vocab.get(&(bos_token_id.as_u32()?)) {
                self.special_tokens.insert(bos_token_id.as_u32()?, bos_token.clone());
                self.bos_token = Some(bos_token.clone());
            }
        }

        if let Ok(eos_token_id) = reader.get_metadata_value("tokenizer.ggml.eos_token_id") {
            if let Some(eos_token) = self.reverse_vocab.get(&(eos_token_id.as_u32()?)) {
                self.special_tokens.insert(eos_token_id.as_u32()?, eos_token.clone());
                self.eos_token = Some(eos_token.clone());
            }
        }

        if let Ok(unk_token_id) = reader.get_metadata_value("tokenizer.ggml.unknown_token_id") {
            if let Some(unk_token) = self.reverse_vocab.get(&(unk_token_id.as_u32()?)) {
                self.special_tokens.insert(unk_token_id.as_u32()?, unk_token.clone());
                self.unk_token = Some(unk_token.clone());
            }
        }

        if let Ok(pad_token_id) = reader.get_metadata_value("tokenizer.ggml.padding_token_id") {
            if let Some(pad_token) = self.reverse_vocab.get(&(pad_token_id.as_u32()?)) {
                self.special_tokens.insert(pad_token_id.as_u32()?, pad_token.clone());
                self.pad_token = Some(pad_token.clone());
            }
        }

        Ok(())
    }

    fn initialize_byte_decoder(&mut self) -> Result<()> {
        // Initialize GPT-2 style byte decoder
        let mut byte_decoder = HashMap::new();

        // Standard ASCII printable characters
        for byte in 33u8..=126u8 {
            byte_decoder.insert(byte as u32, String::from(byte as char));
        }

        // Extended characters for non-printable bytes
        let mut shift = 0;
        for byte in 0u8..=255u8 {
            if !byte_decoder.contains_key(&(byte as u32)) {
                let unicode_char = char::from_u32(256 + shift).unwrap();
                byte_decoder.insert(byte as u32, unicode_char.to_string());
                shift += 1;
            }
        }

        self.byte_decoder = byte_decoder;
        Ok(())
    }
}
```

## Implementation Breakdown

### Phase 1: Core Infrastructure
- [ ] Implement `TokenType` and `TokenizerType` enums
- [ ] Add comprehensive vocabulary loading from GGUF
- [ ] Implement tokenizer type detection logic
- [ ] Add unit tests for type detection

### Phase 2: Token-to-Piece Logic
- [ ] Implement priority-based token-to-piece conversion
- [ ] Add special token handling
- [ ] Implement byte-level fallback for appropriate tokenizers
- [ ] Add comprehensive error handling

### Phase 3: Advanced Features
- [ ] Implement subword reconstruction for BPE/SentencePiece
- [ ] Add support for user-defined added tokens
- [ ] Implement Unicode normalization
- [ ] Add performance optimizations

### Phase 4: Testing and Validation
- [ ] Add comprehensive unit tests for all tokenizer types
- [ ] Create integration tests with real GGUF models
- [ ] Add cross-validation against reference tokenizers
- [ ] Performance benchmarking

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentencepiece_token_to_piece() {
        let tokenizer = create_sentencepiece_tokenizer();

        // Test normal token
        assert_eq!(tokenizer.token_to_piece(100), Some("▁hello".to_string()));

        // Test special token
        assert_eq!(tokenizer.token_to_piece(1), Some("<s>".to_string()));

        // Test unknown token
        assert_eq!(tokenizer.token_to_piece(99999), Some("<unk>".to_string()));
    }

    #[test]
    fn test_bpe_token_to_piece() {
        let tokenizer = create_bpe_tokenizer();

        // Test subword token
        assert_eq!(tokenizer.token_to_piece(200), Some("##ing".to_string()));

        // Test byte fallback
        assert_eq!(tokenizer.token_to_piece(65), Some("A".to_string()));

        // Test special character
        assert_eq!(tokenizer.token_to_piece(128), Some("<0x80>".to_string()));
    }

    #[test]
    fn test_token_type_classification() {
        let tokenizer = GgufTokenizer::new();

        assert_eq!(tokenizer.classify_token_type("▁hello"), TokenType::Subword);
        assert_eq!(tokenizer.classify_token_type("##ing"), TokenType::Subword);
        assert_eq!(tokenizer.classify_token_type("hello"), TokenType::Word);
        assert_eq!(tokenizer.classify_token_type("<s>"), TokenType::Other);
    }
}
```

### Integration Tests
```rust
#[cfg(test)]
mod integration_tests {
    #[test]
    fn test_with_real_llama_model() {
        let gguf_path = "tests/data/llama-7b.gguf";
        let tokenizer = GgufTokenizer::from_file(gguf_path).unwrap();

        // Test round-trip tokenization
        let text = "Hello, world!";
        let tokens = tokenizer.encode(text).unwrap();
        let reconstructed = tokenizer.decode(&tokens).unwrap();

        assert_eq!(text, reconstructed);
    }

    #[test]
    fn test_special_token_handling() {
        let tokenizer = load_test_tokenizer();

        // Test that special tokens are properly identified
        let bos_id = tokenizer.get_bos_token_id().unwrap();
        let (piece, token_type) = tokenizer.token_to_piece_with_type(bos_id).unwrap();

        assert_eq!(token_type, TokenType::Special);
        assert!(piece.starts_with('<') && piece.ends_with('>'));
    }
}
```

### Cross-Validation Tests
```rust
#[cfg(feature = "crossval")]
mod crossval_tests {
    #[test]
    fn test_compatibility_with_huggingface_tokenizers() {
        // Compare token-to-piece conversion with HuggingFace reference
        let our_tokenizer = GgufTokenizer::from_file("model.gguf").unwrap();
        let hf_tokenizer = load_huggingface_tokenizer("model");

        for token_id in 0..1000 {
            let our_piece = our_tokenizer.token_to_piece(token_id);
            let hf_piece = hf_tokenizer.decode(&[token_id]);

            assert_eq!(our_piece, hf_piece, "Mismatch for token {}", token_id);
        }
    }
}
```

## Performance Considerations

1. **Vocabulary Lookup Optimization**: Use efficient hash maps for O(1) token lookups
2. **Memory Efficiency**: Lazy initialization of byte decoders and special token maps
3. **Caching**: Cache frequently accessed token-to-piece conversions
4. **Batch Processing**: Support efficient batch token-to-piece conversion

## Risk Assessment

**Low Risk Changes:**
- Adding new token type enums and classification
- Enhancing vocabulary loading from GGUF

**Medium Risk Changes:**
- Changing token-to-piece priority logic
- Modifying byte fallback behavior

**High Risk Changes:**
- Altering core tokenization interfaces

**Mitigation Strategies:**
- Comprehensive test coverage including edge cases
- Cross-validation with multiple reference implementations
- Gradual rollout with feature flags
- Performance regression testing

## Acceptance Criteria

- [ ] All tokenizer types (SentencePiece, BPE, word-level) supported correctly
- [ ] Special tokens handled properly with semantic meaning preserved
- [ ] Round-trip tokenization accuracy > 99.9% for standard text
- [ ] Cross-validation passes with HuggingFace tokenizers
- [ ] Performance regression < 10% compared to current implementation
- [ ] Comprehensive test coverage (>95% line coverage)
- [ ] Documentation updated with tokenizer type examples

## Related Issues/PRs

- **Related to**: GGUF format parsing improvements
- **Depends on**: Enhanced metadata reading capabilities
- **Blocks**: Advanced text generation features
- **References**: Universal tokenizer architecture standardization

## Additional Context

This enhancement is critical for ensuring accurate text generation and maintaining compatibility with the broader ecosystem of language models. The implementation should maintain backward compatibility while providing robust support for modern tokenization schemes used by state-of-the-art language models.