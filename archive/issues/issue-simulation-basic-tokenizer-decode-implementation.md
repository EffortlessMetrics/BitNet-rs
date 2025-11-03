# [SIMULATION] BasicTokenizer::decode returns placeholder text instead of actual token decoding

## Problem Description

The `BasicTokenizer::decode` method in `lib.rs` returns a simple placeholder string instead of performing actual token-to-text decoding, making the tokenizer non-functional for text generation and preventing proper integration with language models.

## Environment

**File**: `crates/bitnet-tokenizers/src/lib.rs`
**Component**: Basic Tokenizer Implementation
**Issue Type**: Simulation / Missing Core Functionality

## Root Cause Analysis

**Current Implementation:**
```rust
fn decode(&self, tokens: &[u32]) -> Result<String> {
    if tokens.is_empty() {
        return Ok(String::new());
    }

    // Simple placeholder implementation - in real tokenizer this would map back to text
    Ok(format!("Generated text from {} tokens", tokens.len()))
}
```

**Analysis:**
1. **Non-Functional Decoder**: Method returns placeholder text regardless of input tokens
2. **Missing Vocabulary Lookup**: No reverse vocabulary mapping from token IDs to text
3. **No Special Token Handling**: Special tokens (BOS, EOS, PAD, UNK) are not processed
4. **Breaks Text Generation**: Language models cannot produce readable output

## Impact Assessment

**Severity**: High
**Affected Areas**:
- Text generation functionality
- Tokenizer testing and validation
- Integration with language models
- User-facing text output

**Functional Impact**:
- Text generation produces meaningless placeholder strings
- Cannot validate tokenization round-trip accuracy
- Debugging and development severely hampered
- User experience completely broken for text generation

**Business Impact**:
- Core functionality non-operational
- Cannot demonstrate working language model capabilities
- Testing and quality assurance compromised

## Proposed Solution

### Complete Token Decoding Implementation

```rust
impl BasicTokenizer {
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        if tokens.is_empty() {
            return Ok(String::new());
        }

        let mut text_parts = Vec::new();

        for &token in tokens {
            let text_piece = self.decode_single_token(token)?;

            // Skip padding tokens in output
            if token == self.pad_token_id.unwrap_or(u32::MAX) {
                continue;
            }

            // Handle special tokens
            if token == self.bos_token_id.unwrap_or(u32::MAX) {
                continue; // BOS typically not included in output
            }

            if token == self.eos_token_id.unwrap_or(u32::MAX) {
                break; // Stop at EOS token
            }

            text_parts.push(text_piece);
        }

        // Join text pieces with proper spacing
        let decoded_text = self.join_text_pieces(&text_parts)?;
        Ok(decoded_text)
    }

    fn decode_single_token(&self, token: u32) -> Result<String> {
        // Look up token in reverse vocabulary
        if let Some(text_piece) = self.reverse_vocab.get(&token) {
            return Ok(text_piece.clone());
        }

        // Handle special tokens
        if token == self.unk_token_id.unwrap_or(u32::MAX) {
            return Ok(self.unk_token.clone().unwrap_or("<UNK>".to_string()));
        }

        if token == self.pad_token_id.unwrap_or(u32::MAX) {
            return Ok(self.pad_token.clone().unwrap_or("<PAD>".to_string()));
        }

        if token == self.bos_token_id.unwrap_or(u32::MAX) {
            return Ok(self.bos_token.clone().unwrap_or("<BOS>".to_string()));
        }

        if token == self.eos_token_id.unwrap_or(u32::MAX) {
            return Ok(self.eos_token.clone().unwrap_or("<EOS>".to_string()));
        }

        // Unknown token - return UNK representation
        warn!("Unknown token ID encountered during decoding: {}", token);
        Ok(format!("<UNK:{}>", token))
    }

    fn join_text_pieces(&self, pieces: &[String]) -> Result<String> {
        if pieces.is_empty() {
            return Ok(String::new());
        }

        // Handle different tokenization schemes
        match self.tokenization_type {
            TokenizationType::WordLevel => {
                // Word-level: join with spaces
                Ok(pieces.join(" "))
            }
            TokenizationType::SubwordBPE => {
                // BPE: handle subword merging
                self.merge_bpe_pieces(pieces)
            }
            TokenizationType::CharLevel => {
                // Character-level: direct concatenation
                Ok(pieces.concat())
            }
            TokenizationType::SentencePiece => {
                // SentencePiece: handle special prefix markers
                self.merge_sentencepiece_tokens(pieces)
            }
        }
    }

    fn merge_bpe_pieces(&self, pieces: &[String]) -> Result<String> {
        let mut result = String::new();

        for (i, piece) in pieces.iter().enumerate() {
            if piece.starts_with("##") {
                // Subword continuation - remove prefix and concatenate
                result.push_str(&piece[2..]);
            } else {
                // Start of new word
                if i > 0 {
                    result.push(' ');
                }
                result.push_str(piece);
            }
        }

        Ok(result)
    }

    fn merge_sentencepiece_tokens(&self, pieces: &[String]) -> Result<String> {
        let mut result = String::new();

        for piece in pieces {
            if piece.starts_with('▁') {
                // SentencePiece space marker
                if !result.is_empty() {
                    result.push(' ');
                }
                result.push_str(&piece[3..]); // Remove ▁ (3 bytes in UTF-8)
            } else {
                // Continuation piece
                result.push_str(piece);
            }
        }

        Ok(result)
    }

    // Helper method to build reverse vocabulary during initialization
    fn build_reverse_vocab(&mut self) -> Result<()> {
        self.reverse_vocab.clear();

        for (text, &token_id) in &self.vocab {
            self.reverse_vocab.insert(token_id, text.clone());
        }

        // Verify all special tokens are in reverse vocab
        if let Some(bos_id) = self.bos_token_id {
            if !self.reverse_vocab.contains_key(&bos_id) {
                let bos_text = self.bos_token.clone().unwrap_or("<BOS>".to_string());
                self.reverse_vocab.insert(bos_id, bos_text);
            }
        }

        // Similar for other special tokens...

        info!("Built reverse vocabulary with {} entries", self.reverse_vocab.len());
        Ok(())
    }
}

// Add new fields to BasicTokenizer struct
#[derive(Debug, Clone)]
pub struct BasicTokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>, // Add this field
    tokenization_type: TokenizationType, // Add this field

    // Special tokens
    bos_token: Option<String>,
    eos_token: Option<String>,
    unk_token: Option<String>,
    pad_token: Option<String>,

    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    unk_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
pub enum TokenizationType {
    WordLevel,
    SubwordBPE,
    CharLevel,
    SentencePiece,
}
```

## Implementation Plan

### Task 1: Core Decoding Infrastructure
- [ ] Add reverse vocabulary mapping (token ID -> text)
- [ ] Implement `decode_single_token` for individual token lookup
- [ ] Add proper special token handling (BOS, EOS, PAD, UNK)
- [ ] Build reverse vocabulary during tokenizer initialization

### Task 2: Tokenization Scheme Support
- [ ] Add `TokenizationType` enum to support different schemes
- [ ] Implement word-level token joining with spaces
- [ ] Add BPE subword merging logic
- [ ] Implement SentencePiece token reconstruction

### Task 3: Text Reconstruction Logic
- [ ] Implement proper text piece joining based on tokenization type
- [ ] Handle subword markers (##, ▁) correctly
- [ ] Add special token filtering (remove BOS/PAD, stop at EOS)
- [ ] Optimize string concatenation for large token sequences

### Task 4: Error Handling and Validation
- [ ] Add comprehensive error handling for invalid tokens
- [ ] Implement graceful degradation for unknown tokens
- [ ] Add validation for reverse vocabulary completeness
- [ ] Handle edge cases (empty input, only special tokens)

## Testing Strategy

### Basic Decoding Tests
```rust
#[test]
fn test_basic_token_decoding() {
    let mut tokenizer = BasicTokenizer::new();

    // Set up test vocabulary
    tokenizer.vocab.insert("hello".to_string(), 1);
    tokenizer.vocab.insert("world".to_string(), 2);
    tokenizer.vocab.insert("!".to_string(), 3);
    tokenizer.build_reverse_vocab().unwrap();

    let tokens = vec![1, 2, 3]; // "hello", "world", "!"
    let result = tokenizer.decode(&tokens).unwrap();

    assert_eq!(result, "hello world !");
}

#[test]
fn test_special_token_handling() {
    let mut tokenizer = BasicTokenizer::new();
    tokenizer.bos_token_id = Some(0);
    tokenizer.eos_token_id = Some(999);
    tokenizer.pad_token_id = Some(998);

    tokenizer.vocab.insert("test".to_string(), 1);
    tokenizer.build_reverse_vocab().unwrap();

    // Test with special tokens
    let tokens = vec![0, 1, 999, 998]; // BOS, "test", EOS, PAD
    let result = tokenizer.decode(&tokens).unwrap();

    // Should skip BOS and PAD, stop at EOS
    assert_eq!(result, "test");
}

#[test]
fn test_unknown_token_handling() {
    let mut tokenizer = BasicTokenizer::new();
    tokenizer.unk_token_id = Some(0);
    tokenizer.unk_token = Some("<UNK>".to_string());

    tokenizer.vocab.insert("known".to_string(), 1);
    tokenizer.build_reverse_vocab().unwrap();

    let tokens = vec![1, 999, 1]; // known, unknown, known
    let result = tokenizer.decode(&tokens).unwrap();

    assert!(result.contains("known"));
    assert!(result.contains("<UNK:999>"));
}
```

### Round-Trip Tests
```rust
#[test]
fn test_encode_decode_round_trip() {
    let mut tokenizer = BasicTokenizer::new();

    // Set up vocabulary for common words
    let test_vocab = vec!["hello", "world", "this", "is", "a", "test"];
    for (i, word) in test_vocab.iter().enumerate() {
        tokenizer.vocab.insert(word.to_string(), i as u32 + 1);
    }
    tokenizer.build_reverse_vocab().unwrap();

    let original_text = "hello world this is a test";

    // Encode then decode
    let tokens = tokenizer.encode(original_text).unwrap();
    let decoded_text = tokenizer.decode(&tokens).unwrap();

    assert_eq!(original_text, decoded_text);
}

#[test]
fn test_bpe_subword_reconstruction() {
    let mut tokenizer = BasicTokenizer::new();
    tokenizer.tokenization_type = TokenizationType::SubwordBPE;

    // Set up BPE vocabulary
    tokenizer.vocab.insert("un".to_string(), 1);
    tokenizer.vocab.insert("##happy".to_string(), 2);
    tokenizer.build_reverse_vocab().unwrap();

    let tokens = vec![1, 2]; // "un", "##happy"
    let result = tokenizer.decode(&tokens).unwrap();

    assert_eq!(result, "unhappy");
}
```

### Performance Tests
```rust
#[test]
fn test_decode_performance() {
    let mut tokenizer = create_large_test_tokenizer(10000); // 10k vocab
    let large_token_sequence: Vec<u32> = (1..1000).collect(); // 1k tokens

    let start = Instant::now();
    let result = tokenizer.decode(&large_token_sequence);
    let duration = start.elapsed();

    assert!(result.is_ok());
    assert!(duration < Duration::from_millis(100)); // Should decode quickly
}
```

## Related Issues/PRs

- Fundamental requirement for text generation functionality
- Related to tokenizer validation and testing framework
- Part of comprehensive tokenization system implementation

## Acceptance Criteria

- [ ] Token IDs are correctly mapped back to text using reverse vocabulary
- [ ] Special tokens (BOS, EOS, PAD, UNK) are handled appropriately
- [ ] Different tokenization schemes (word-level, BPE, SentencePiece) work correctly
- [ ] Round-trip tests (encode -> decode) preserve original text
- [ ] Unknown tokens are handled gracefully with clear error indication
- [ ] Performance is acceptable for typical token sequence lengths
- [ ] Comprehensive error handling for edge cases

## Risk Assessment

**Medium Risk**: Implementing proper token decoding requires understanding of tokenization schemes and careful handling of special cases.

**Mitigation Strategies**:
- Implement comprehensive test suite covering all tokenization types
- Add extensive validation for vocabulary completeness
- Provide clear error messages for debugging issues
- Implement performance benchmarks to detect regressions
- Add configuration options for different decoding behaviors
