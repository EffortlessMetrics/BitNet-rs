# [SIMULATION] Replace Simplified BasicTokenizer::encode with Production-Grade Tokenization

## Problem Description

The `BasicTokenizer::encode` function in `crates/bitnet-tokenizers/src/lib.rs` implements a naive whitespace-based tokenization scheme that generates token IDs based on word position rather than actual vocabulary lookup. This simulation approach completely bypasses proper tokenization algorithms like BPE (Byte Pair Encoding) or SentencePiece, making it incompatible with real-world models and producing incorrect token sequences.

## Environment

- **File**: `crates/bitnet-tokenizers/src/lib.rs`
- **Function**: `BasicTokenizer::encode`
- **Crate**: `bitnet-tokenizers`
- **Affected Models**: All models using BasicTokenizer (GPT-2, LLaMA fallbacks)
- **Impact**: Core tokenization functionality

## Current Implementation Issues

```rust
fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut tokens: Vec<u32> = Vec::new();

    // PROBLEM: Uses word index as token ID instead of vocabulary lookup
    for (i, _) in words.iter().enumerate() {
        let id = i as u32;  // This is completely wrong!
        if id >= self.vocab_size as u32 {
            return Err(/* error */);
        }
        tokens.push(id);
    }
}
```

## Root Cause Analysis

### Fundamental Design Issues
1. **No Vocabulary Mapping**: Uses word position instead of vocabulary lookup
2. **Naive Tokenization**: Simple whitespace splitting ignores subword tokenization
3. **Missing Algorithms**: No BPE, SentencePiece, or other modern tokenization methods
4. **Incorrect Token Generation**: Produces invalid token sequences for any real model
5. **No Unicode Handling**: Fails with non-ASCII text and complex scripts

### Incompatibility Problems
- **Model Mismatch**: Tokens don't match training vocabulary
- **Sequence Length**: Incorrect token count affects model input
- **Special Token Handling**: Improper BOS/EOS token placement
- **Cross-Model Compatibility**: Can't handle different tokenizer configurations

## Impact Assessment

- **Severity**: Critical - Core functionality completely broken
- **Affected Components**: All text processing, model inference, tokenization
- **User Impact**: Incorrect model outputs, inference failures, compatibility issues
- **Model Support**: Prevents use with any real pre-trained models

## Proposed Solution

Implement a comprehensive tokenization system supporting multiple algorithms:

### 1. Vocabulary-Based Tokenization Infrastructure
```rust
use std::collections::HashMap;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct Vocabulary {
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    special_tokens: SpecialTokens,
    vocab_size: usize,
}

#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
    pub pad_token: Option<String>,
    pub mask_token: Option<String>,
}

impl Vocabulary {
    pub fn from_tokens(tokens: Vec<String>, special_tokens: SpecialTokens) -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (id, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
            id_to_token.insert(id as u32, token.clone());
        }

        Self {
            token_to_id,
            id_to_token,
            special_tokens,
            vocab_size: tokens.len(),
        }
    }

    pub fn get_token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }
}
```

### 2. BPE Tokenization Implementation
```rust
#[derive(Debug, Clone)]
pub struct BPETokenizer {
    vocabulary: Vocabulary,
    merges: Vec<(String, String)>,
    pattern: Regex,
    cache: HashMap<String, Vec<String>>,
}

impl BPETokenizer {
    pub fn new(vocabulary: Vocabulary, merges: Vec<(String, String)>) -> Result<Self> {
        // GPT-2 style regex pattern for tokenization
        let pattern = Regex::new(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        )?;

        Ok(Self {
            vocabulary,
            merges,
            pattern,
            cache: HashMap::new(),
        })
    }

    fn apply_bpe(&mut self, word: &str) -> Vec<String> {
        if let Some(cached) = self.cache.get(word) {
            return cached.clone();
        }

        let mut word_chars: Vec<char> = word.chars().collect();
        if word_chars.is_empty() {
            return vec![];
        }

        // Start with character-level tokens
        let mut pairs = self.get_pairs(&word_chars);

        if pairs.is_empty() {
            let result = vec![word.to_string()];
            self.cache.insert(word.to_string(), result.clone());
            return result;
        }

        loop {
            let bigram = self.find_best_merge(&pairs);
            if bigram.is_none() {
                break;
            }

            let (first, second) = bigram.unwrap();
            let mut new_word = Vec::new();
            let mut i = 0;

            while i < word_chars.len() {
                if i < word_chars.len() - 1
                    && word_chars[i].to_string() == first
                    && word_chars[i + 1].to_string() == second {
                    new_word.push(format!("{}{}", first, second));
                    i += 2;
                } else {
                    new_word.push(word_chars[i].to_string());
                    i += 1;
                }
            }

            word_chars = new_word.iter().map(|s| s.chars().next().unwrap()).collect();
            if word_chars.len() == 1 {
                break;
            }

            pairs = self.get_pairs(&word_chars);
        }

        let result: Vec<String> = word_chars.iter().map(|c| c.to_string()).collect();
        self.cache.insert(word.to_string(), result.clone());
        result
    }

    fn find_best_merge(&self, pairs: &HashSet<(String, String)>) -> Option<(String, String)> {
        pairs.iter()
            .filter_map(|(first, second)| {
                self.merges.iter().position(|(a, b)| a == first && b == second)
                    .map(|rank| (rank, (first.clone(), second.clone())))
            })
            .min_by_key(|(rank, _)| *rank)
            .map(|(_, pair)| pair)
    }
}

impl TokenizerTrait for BPETokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();

        // Add BOS token if requested
        if add_bos {
            if let Some(bos_token) = &self.vocabulary.special_tokens.bos_token {
                if let Some(bos_id) = self.vocabulary.get_token_id(bos_token) {
                    tokens.push(bos_id);
                }
            }
        }

        // Tokenize the input text
        for word_match in self.pattern.find_iter(text) {
            let word = word_match.as_str();
            let bpe_tokens = self.apply_bpe(word);

            for token in bpe_tokens {
                if let Some(token_id) = self.vocabulary.get_token_id(&token) {
                    tokens.push(token_id);
                } else if let Some(unk_id) = self.vocabulary.special_tokens.unk_token
                    .as_ref()
                    .and_then(|unk| self.vocabulary.get_token_id(unk)) {
                    tokens.push(unk_id);
                } else {
                    return Err(BitNetError::Tokenization(
                        TokenizationError::UnknownToken { token: token.clone() }
                    ));
                }
            }
        }

        // Add special tokens if requested
        if add_special {
            if let Some(eos_token) = &self.vocabulary.special_tokens.eos_token {
                if let Some(eos_id) = self.vocabulary.get_token_id(eos_token) {
                    tokens.push(eos_id);
                }
            }
        }

        Ok(tokens)
    }
}
```

### 3. SentencePiece Integration
```rust
#[cfg(feature = "sentencepiece")]
pub struct SentencePieceTokenizer {
    processor: sentencepiece::SentencePieceProcessor,
    vocabulary: Vocabulary,
}

#[cfg(feature = "sentencepiece")]
impl SentencePieceTokenizer {
    pub fn from_model_file(model_path: &Path) -> Result<Self> {
        let processor = sentencepiece::SentencePieceProcessor::open(model_path)?;
        let vocabulary = Self::extract_vocabulary(&processor)?;

        Ok(Self {
            processor,
            vocabulary,
        })
    }

    fn extract_vocabulary(processor: &sentencepiece::SentencePieceProcessor) -> Result<Vocabulary> {
        let vocab_size = processor.get_vocab_size();
        let mut tokens = Vec::with_capacity(vocab_size as usize);

        for id in 0..vocab_size {
            let token = processor.id_to_piece(id)?;
            tokens.push(token);
        }

        let special_tokens = SpecialTokens {
            bos_token: processor.bos_piece().map(|s| s.to_string()),
            eos_token: processor.eos_piece().map(|s| s.to_string()),
            unk_token: processor.unk_piece().map(|s| s.to_string()),
            pad_token: processor.pad_piece().map(|s| s.to_string()),
            mask_token: None,
        };

        Ok(Vocabulary::from_tokens(tokens, special_tokens))
    }
}

#[cfg(feature = "sentencepiece")]
impl TokenizerTrait for SentencePieceTokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        let mut pieces = self.processor.encode_as_ids(text)?;

        if add_bos && !pieces.is_empty() {
            if let Some(bos_id) = self.processor.bos_id() {
                pieces.insert(0, bos_id as u32);
            }
        }

        if add_special {
            if let Some(eos_id) = self.processor.eos_id() {
                pieces.push(eos_id as u32);
            }
        }

        Ok(pieces)
    }
}
```

### 4. Enhanced BasicTokenizer
```rust
impl BasicTokenizer {
    pub fn new(vocabulary: Vocabulary) -> Self {
        Self {
            vocabulary,
            normalization: TextNormalization::default(),
        }
    }

    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();

        // Add BOS token if requested
        if add_bos {
            if let Some(bos_token) = &self.vocabulary.special_tokens.bos_token {
                if let Some(bos_id) = self.vocabulary.get_token_id(bos_token) {
                    tokens.push(bos_id);
                }
            }
        }

        // Normalize and tokenize text
        let normalized_text = self.normalization.normalize(text);
        let words = self.tokenize_words(&normalized_text);

        for word in words {
            if let Some(token_id) = self.vocabulary.get_token_id(&word) {
                tokens.push(token_id);
            } else if let Some(unk_id) = self.vocabulary.special_tokens.unk_token
                .as_ref()
                .and_then(|unk| self.vocabulary.get_token_id(unk)) {
                tokens.push(unk_id);
            } else {
                return Err(BitNetError::Tokenization(
                    TokenizationError::UnknownToken { token: word }
                ));
            }
        }

        // Add special tokens if requested
        if add_special {
            if let Some(eos_token) = &self.vocabulary.special_tokens.eos_token {
                if let Some(eos_id) = self.vocabulary.get_token_id(eos_token) {
                    tokens.push(eos_id);
                }
            }
        }

        Ok(tokens)
    }

    fn tokenize_words(&self, text: &str) -> Vec<String> {
        // Implement proper word tokenization with punctuation handling
        text.split_whitespace()
            .flat_map(|word| self.split_punctuation(word))
            .map(|s| s.to_string())
            .collect()
    }

    fn split_punctuation(&self, word: &str) -> Vec<&str> {
        // Split punctuation while preserving word boundaries
        // Implementation depends on tokenizer configuration
        vec![word] // Simplified for now
    }
}
```

## Implementation Plan

### Phase 1: Vocabulary Infrastructure
- [ ] Implement `Vocabulary` and `SpecialTokens` structures
- [ ] Add vocabulary loading from files (JSON, text)
- [ ] Create vocabulary validation and error handling
- [ ] Add vocabulary serialization/deserialization

### Phase 2: BPE Implementation
- [ ] Implement BPE merge operations and ranking
- [ ] Add BPE model loading (GPT-2 format)
- [ ] Create efficient caching system for BPE operations
- [ ] Add regex-based pre-tokenization

### Phase 3: Enhanced BasicTokenizer
- [ ] Replace naive implementation with vocabulary-based tokenization
- [ ] Add proper text normalization and preprocessing
- [ ] Implement punctuation and special character handling
- [ ] Add Unicode and multi-language support

### Phase 4: SentencePiece Integration
- [ ] Add optional SentencePiece dependency
- [ ] Implement SentencePiece tokenizer wrapper
- [ ] Add model format compatibility checking
- [ ] Create unified tokenizer interface

### Phase 5: Testing and Validation
- [ ] Add comprehensive tokenization tests
- [ ] Create cross-validation with reference implementations
- [ ] Add performance benchmarks
- [ ] Test with real model vocabularies

## Testing Strategy

### Tokenization Accuracy Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_tokenization() {
        let vocab = load_gpt2_vocabulary();
        let tokenizer = BPETokenizer::new(vocab, load_gpt2_merges()).unwrap();

        let text = "Hello, world!";
        let tokens = tokenizer.encode(text, false, false).unwrap();

        // Verify tokens match expected GPT-2 tokenization
        assert_eq!(tokens, vec![15496, 11, 995, 0]);
    }

    #[test]
    fn test_special_token_handling() {
        let tokenizer = create_test_tokenizer();

        let tokens = tokenizer.encode("Test text", true, true).unwrap();

        // Verify BOS and EOS tokens are properly added
        assert_eq!(tokens[0], BOS_TOKEN_ID);
        assert_eq!(tokens[tokens.len() - 1], EOS_TOKEN_ID);
    }

    #[test]
    fn test_unicode_handling() {
        let tokenizer = create_test_tokenizer();

        let unicode_text = "Hello 🌍 world! 中文测试";
        let tokens = tokenizer.encode(unicode_text, false, false).unwrap();

        // Verify proper Unicode tokenization
        assert!(!tokens.is_empty());
        assert!(tokens.iter().all(|&id| id < tokenizer.vocab_size() as u32));
    }
}
```

## BitNet.rs Integration Notes

### Tokenizer Discovery Integration
- Integrate with existing tokenizer discovery system
- Support automatic tokenizer selection based on model type
- Maintain compatibility with GGUF embedded tokenizers

### Model Compatibility
- Ensure tokenization matches model training data
- Support multiple vocabulary formats (GGUF, JSON, SentencePiece)
- Add vocabulary validation against model requirements

### Performance Considerations
- Optimize for inference-time tokenization speed
- Implement efficient caching for repeated tokenization
- Balance memory usage with tokenization performance

## Dependencies

```toml
[dependencies]
regex = "1.10"
unicode-normalization = "0.1"
sentencepiece = { version = "0.20", optional = true }

[features]
sentencepiece = ["dep:sentencepiece"]
```

## Acceptance Criteria

- [ ] Complete removal of naive word-index tokenization
- [ ] Vocabulary-based tokenization with proper lookup
- [ ] BPE implementation compatible with GPT-2/GPT-4 models
- [ ] SentencePiece integration for LLaMA and T5 models
- [ ] Proper special token handling (BOS, EOS, UNK, PAD)
- [ ] Unicode and multi-language text support
- [ ] Performance optimization for inference workloads
- [ ] Comprehensive test coverage with real model vocabularies
- [ ] Cross-validation with reference tokenizer implementations
- [ ] Integration with existing model loading and discovery systems

## Related Issues

- Tokenizer discovery and model compatibility
- GGUF tokenizer integration
- Model loading and validation
- Text preprocessing and normalization

## Priority

**Critical** - Core functionality that affects all text processing and model compatibility. Without proper tokenization, the system cannot work with any real pre-trained models.
