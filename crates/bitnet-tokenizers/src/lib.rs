//! Tokenization support for BitNet models

use bitnet_common::Result;
use std::path::Path;
use std::sync::Arc;

/// Tokenizer trait
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn eos_token_id(&self) -> Option<u32>;
    fn pad_token_id(&self) -> Option<u32>;
}

/// Basic tokenizer implementation
pub struct BasicTokenizer {
    vocab_size: usize,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

impl BasicTokenizer {
    pub fn new() -> Self {
        Self {
            vocab_size: 50257, // GPT-2 vocab size
            eos_token_id: Some(50256),
            pad_token_id: None,
        }
    }

    pub fn with_config(
        vocab_size: usize,
        eos_token_id: Option<u32>,
        pad_token_id: Option<u32>,
    ) -> Self {
        Self {
            vocab_size,
            eos_token_id,
            pad_token_id,
        }
    }
}

impl Default for BasicTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for BasicTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Simple word-based tokenization for testing
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens: Vec<u32> = words.iter().enumerate().map(|(i, _)| i as u32).collect();

        // Add special tokens if requested
        if add_special_tokens {
            if let Some(eos_id) = self.eos_token_id {
                tokens.push(eos_id);
            }
        }

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        if tokens.is_empty() {
            return Ok(String::new());
        }

        let mut filtered_tokens = tokens.to_vec();

        // Filter special tokens if requested
        if skip_special_tokens {
            if let Some(eos_id) = self.eos_token_id {
                filtered_tokens.retain(|&token| token != eos_id);
            }
            if let Some(pad_id) = self.pad_token_id {
                filtered_tokens.retain(|&token| token != pad_id);
            }
        }

        // Handle case where all tokens were filtered out
        if filtered_tokens.is_empty() {
            return Ok(String::new());
        }

        // Simple placeholder implementation - in real tokenizer this would map back to text
        Ok(format!(
            "Generated text from {} tokens",
            filtered_tokens.len()
        ))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }
}

/// Tokenizer builder for creating tokenizers
pub struct TokenizerBuilder;

impl TokenizerBuilder {
    /// Create tokenizer from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Arc<dyn Tokenizer>> {
        // Placeholder implementation
        tracing::debug!("Loading tokenizer from: {}", path.as_ref().display());
        Ok(Arc::new(BasicTokenizer::new()))
    }

    /// Create tokenizer from pretrained model
    pub fn from_pretrained(name: &str) -> Result<Arc<dyn Tokenizer>> {
        // Placeholder implementation
        tracing::debug!("Loading pretrained tokenizer: {}", name);

        // Return different configurations based on model name for testing
        match name {
            "gpt2" => Ok(Arc::new(BasicTokenizer::with_config(
                50257,
                Some(50256),
                None,
            ))),
            "bert" => Ok(Arc::new(BasicTokenizer::with_config(
                30522,
                Some(102),
                Some(0),
            ))),
            "tiny" => Ok(Arc::new(BasicTokenizer::with_config(
                1000,
                Some(999),
                Some(0),
            ))),
            _ => Ok(Arc::new(BasicTokenizer::new())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_basic_tokenizer_creation() {
        let tokenizer = BasicTokenizer::new();
        assert_eq!(tokenizer.vocab_size(), 50257);
        assert_eq!(tokenizer.eos_token_id(), Some(50256));
        assert_eq!(tokenizer.pad_token_id(), None);
    }

    #[test]
    fn test_basic_tokenizer_with_config() {
        let tokenizer = BasicTokenizer::with_config(1000, Some(999), Some(0));
        assert_eq!(tokenizer.vocab_size(), 1000);
        assert_eq!(tokenizer.eos_token_id(), Some(999));
        assert_eq!(tokenizer.pad_token_id(), Some(0));
    }

    #[test]
    fn test_basic_tokenizer_default() {
        let tokenizer = BasicTokenizer::default();
        assert_eq!(tokenizer.vocab_size(), 50257);
        assert_eq!(tokenizer.eos_token_id(), Some(50256));
        assert_eq!(tokenizer.pad_token_id(), None);
    }

    #[test]
    fn test_encode_empty_text() {
        let tokenizer = BasicTokenizer::new();
        let result = tokenizer.encode("", false).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_encode_simple_text() {
        let tokenizer = BasicTokenizer::new();
        let result = tokenizer.encode("hello world", false).unwrap();
        assert_eq!(result, vec![0, 1]); // Two words -> two tokens
    }

    #[test]
    fn test_encode_with_special_tokens() {
        let tokenizer = BasicTokenizer::new();
        let result = tokenizer.encode("hello world", true).unwrap();
        assert_eq!(result, vec![0, 1, 50256]); // Two words + EOS token
    }

    #[test]
    fn test_encode_without_special_tokens() {
        let tokenizer = BasicTokenizer::new();
        let result = tokenizer.encode("hello world", false).unwrap();
        assert_eq!(result, vec![0, 1]); // Two words, no EOS token
    }

    #[test]
    fn test_encode_single_word() {
        let tokenizer = BasicTokenizer::new();
        let result = tokenizer.encode("hello", false).unwrap();
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_encode_multiple_spaces() {
        let tokenizer = BasicTokenizer::new();
        let result = tokenizer.encode("hello    world", false).unwrap();
        assert_eq!(result, vec![0, 1]); // Multiple spaces should be treated as single separator
    }

    #[test]
    fn test_encode_leading_trailing_spaces() {
        let tokenizer = BasicTokenizer::new();
        let result = tokenizer.encode("  hello world  ", false).unwrap();
        assert_eq!(result, vec![0, 1]); // Leading/trailing spaces should be ignored
    }

    #[test]
    fn test_decode_empty_tokens() {
        let tokenizer = BasicTokenizer::new();
        let result = tokenizer.decode(&[], false).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_decode_simple_tokens() {
        let tokenizer = BasicTokenizer::new();
        let result = tokenizer.decode(&[0, 1, 2], false).unwrap();
        assert_eq!(result, "Generated text from 3 tokens");
    }

    #[test]
    fn test_decode_with_special_tokens() {
        let tokenizer = BasicTokenizer::new();
        let tokens = vec![0, 1, 50256]; // Include EOS token
        let result = tokenizer.decode(&tokens, false).unwrap();
        assert_eq!(result, "Generated text from 3 tokens"); // Should include special token
    }

    #[test]
    fn test_decode_skip_special_tokens() {
        let tokenizer = BasicTokenizer::new();
        let tokens = vec![0, 1, 50256]; // Include EOS token
        let result = tokenizer.decode(&tokens, true).unwrap();
        assert_eq!(result, "Generated text from 2 tokens"); // Should skip EOS token
    }

    #[test]
    fn test_decode_skip_pad_tokens() {
        let tokenizer = BasicTokenizer::with_config(1000, Some(999), Some(0));
        let tokens = vec![1, 2, 0, 3, 0]; // Include PAD tokens
        let result = tokenizer.decode(&tokens, true).unwrap();
        assert_eq!(result, "Generated text from 3 tokens"); // Should skip PAD tokens
    }

    #[test]
    fn test_decode_skip_multiple_special_tokens() {
        let tokenizer = BasicTokenizer::with_config(1000, Some(999), Some(0));
        let tokens = vec![1, 2, 0, 3, 999, 0]; // Include both PAD and EOS tokens
        let result = tokenizer.decode(&tokens, true).unwrap();
        assert_eq!(result, "Generated text from 3 tokens"); // Should skip both special tokens
    }

    #[test]
    fn test_vocab_size_consistency() {
        let tokenizer1 = BasicTokenizer::new();
        let tokenizer2 = BasicTokenizer::default();
        assert_eq!(tokenizer1.vocab_size(), tokenizer2.vocab_size());
    }

    #[test]
    fn test_special_token_ids_consistency() {
        let tokenizer = BasicTokenizer::new();
        assert_eq!(tokenizer.eos_token_id(), Some(50256));
        assert_eq!(tokenizer.pad_token_id(), None);
    }

    #[test]
    fn test_tokenizer_builder_from_file() {
        let result = TokenizerBuilder::from_file("test.json");
        assert!(result.is_ok());
        let tokenizer = result.unwrap();
        assert_eq!(tokenizer.vocab_size(), 50257);
    }

    #[test]
    fn test_tokenizer_builder_from_pretrained_gpt2() {
        let result = TokenizerBuilder::from_pretrained("gpt2");
        assert!(result.is_ok());
        let tokenizer = result.unwrap();
        assert_eq!(tokenizer.vocab_size(), 50257);
        assert_eq!(tokenizer.eos_token_id(), Some(50256));
        assert_eq!(tokenizer.pad_token_id(), None);
    }

    #[test]
    fn test_tokenizer_builder_from_pretrained_bert() {
        let result = TokenizerBuilder::from_pretrained("bert");
        assert!(result.is_ok());
        let tokenizer = result.unwrap();
        assert_eq!(tokenizer.vocab_size(), 30522);
        assert_eq!(tokenizer.eos_token_id(), Some(102));
        assert_eq!(tokenizer.pad_token_id(), Some(0));
    }

    #[test]
    fn test_tokenizer_builder_from_pretrained_tiny() {
        let result = TokenizerBuilder::from_pretrained("tiny");
        assert!(result.is_ok());
        let tokenizer = result.unwrap();
        assert_eq!(tokenizer.vocab_size(), 1000);
        assert_eq!(tokenizer.eos_token_id(), Some(999));
        assert_eq!(tokenizer.pad_token_id(), Some(0));
    }

    #[test]
    fn test_tokenizer_builder_from_pretrained_unknown() {
        let result = TokenizerBuilder::from_pretrained("unknown_model");
        assert!(result.is_ok());
        let tokenizer = result.unwrap();
        assert_eq!(tokenizer.vocab_size(), 50257); // Should default to basic config
    }

    #[test]
    fn test_encode_decode_roundtrip_consistency() {
        let tokenizer = BasicTokenizer::new();
        let original_text = "hello world test";

        // Encode without special tokens
        let tokens = tokenizer.encode(original_text, false).unwrap();
        let decoded = tokenizer.decode(&tokens, false).unwrap();

        // The decoded text should be consistent (though not identical due to placeholder implementation)
        assert!(!decoded.is_empty());
        assert!(decoded.contains("3 tokens")); // Should reflect the number of words
    }

    #[test]
    fn test_encode_decode_with_special_tokens_roundtrip() {
        let tokenizer = BasicTokenizer::new();
        let original_text = "hello world";

        // Encode with special tokens
        let tokens = tokenizer.encode(original_text, true).unwrap();
        assert_eq!(tokens.len(), 3); // 2 words + 1 EOS token

        // Decode without skipping special tokens
        let decoded_with_special = tokenizer.decode(&tokens, false).unwrap();
        assert!(decoded_with_special.contains("3 tokens"));

        // Decode skipping special tokens
        let decoded_without_special = tokenizer.decode(&tokens, true).unwrap();
        assert!(decoded_without_special.contains("2 tokens"));
    }

    #[test]
    fn test_tokenization_performance() {
        let tokenizer = BasicTokenizer::new();
        let text = "hello world this is a test of tokenization performance with multiple words";

        let start = Instant::now();
        for _ in 0..1000 {
            let _ = tokenizer.encode(text, false).unwrap();
        }
        let encode_duration = start.elapsed();

        // Encoding 1000 times should be reasonably fast (less than 100ms)
        assert!(encode_duration.as_millis() < 100);

        let tokens = tokenizer.encode(text, false).unwrap();
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = tokenizer.decode(&tokens, false).unwrap();
        }
        let decode_duration = start.elapsed();

        // Decoding 1000 times should be reasonably fast (less than 100ms)
        assert!(decode_duration.as_millis() < 100);
    }

    #[test]
    fn test_memory_usage_with_large_text() {
        let tokenizer = BasicTokenizer::new();

        // Create a large text with many words
        let words: Vec<String> = (0..10000).map(|i| format!("word{}", i)).collect();
        let large_text = words.join(" ");

        let tokens = tokenizer.encode(&large_text, false).unwrap();
        assert_eq!(tokens.len(), 10000);

        let decoded = tokenizer.decode(&tokens, false).unwrap();
        assert!(decoded.contains("10000 tokens"));
    }

    #[test]
    fn test_edge_case_unicode_text() {
        let tokenizer = BasicTokenizer::new();

        // Test with Unicode characters
        let unicode_text = "hello 世界 🌍 test";
        let tokens = tokenizer.encode(unicode_text, false).unwrap();
        assert_eq!(tokens.len(), 4); // Should split on whitespace regardless of Unicode

        let decoded = tokenizer.decode(&tokens, false).unwrap();
        assert!(decoded.contains("4 tokens"));
    }

    #[test]
    fn test_edge_case_punctuation() {
        let tokenizer = BasicTokenizer::new();

        // Test with punctuation (current implementation treats as single words)
        let punct_text = "hello, world! how are you?";
        let tokens = tokenizer.encode(punct_text, false).unwrap();
        assert_eq!(tokens.len(), 5); // Split on whitespace: ["hello,", "world!", "how", "are", "you?"]

        let decoded = tokenizer.decode(&tokens, false).unwrap();
        assert!(decoded.contains("5 tokens"));
    }

    #[test]
    fn test_edge_case_numbers() {
        let tokenizer = BasicTokenizer::new();

        let number_text = "123 456.789 -42 3.14159";
        let tokens = tokenizer.encode(number_text, false).unwrap();
        assert_eq!(tokens.len(), 4); // Each number is treated as a separate token

        let decoded = tokenizer.decode(&tokens, false).unwrap();
        assert!(decoded.contains("4 tokens"));
    }

    #[test]
    fn test_edge_case_very_long_word() {
        let tokenizer = BasicTokenizer::new();

        let long_word = "a".repeat(10000);
        let tokens = tokenizer.encode(&long_word, false).unwrap();
        assert_eq!(tokens.len(), 1); // Single very long word

        let decoded = tokenizer.decode(&tokens, false).unwrap();
        assert!(decoded.contains("1 tokens"));
    }

    #[test]
    fn test_consistency_across_multiple_calls() {
        let tokenizer = BasicTokenizer::new();
        let text = "consistent tokenization test";

        // Multiple calls should produce identical results
        let tokens1 = tokenizer.encode(text, false).unwrap();
        let tokens2 = tokenizer.encode(text, false).unwrap();
        let tokens3 = tokenizer.encode(text, false).unwrap();

        assert_eq!(tokens1, tokens2);
        assert_eq!(tokens2, tokens3);

        // Same for decoding
        let decoded1 = tokenizer.decode(&tokens1, false).unwrap();
        let decoded2 = tokenizer.decode(&tokens1, false).unwrap();
        let decoded3 = tokenizer.decode(&tokens1, false).unwrap();

        assert_eq!(decoded1, decoded2);
        assert_eq!(decoded2, decoded3);
    }

    #[test]
    fn test_different_tokenizer_configurations() {
        let tokenizer1 = BasicTokenizer::with_config(1000, Some(999), Some(0));
        let tokenizer2 = BasicTokenizer::with_config(2000, Some(1999), Some(1));

        assert_ne!(tokenizer1.vocab_size(), tokenizer2.vocab_size());
        assert_ne!(tokenizer1.eos_token_id(), tokenizer2.eos_token_id());
        assert_ne!(tokenizer1.pad_token_id(), tokenizer2.pad_token_id());

        // But they should handle the same text consistently within their own configuration
        let text = "test configuration";
        let tokens1 = tokenizer1.encode(text, false).unwrap();
        let tokens2 = tokenizer2.encode(text, false).unwrap();

        // Same number of tokens (same splitting logic)
        assert_eq!(tokens1.len(), tokens2.len());
    }

    #[test]
    fn test_special_token_filtering_edge_cases() {
        let tokenizer = BasicTokenizer::with_config(100, Some(99), Some(0));

        // Test with only special tokens
        let only_special = vec![0, 99, 0, 99];
        let decoded = tokenizer.decode(&only_special, true).unwrap();
        assert_eq!(decoded, ""); // Should be empty after filtering all special tokens

        // Test with no special tokens
        let no_special = vec![1, 2, 3, 4];
        let decoded = tokenizer.decode(&no_special, true).unwrap();
        assert!(decoded.contains("4 tokens"));
    }

    #[test]
    fn test_tokenizer_trait_object_usage() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(BasicTokenizer::new());

        let text = "trait object test";
        let tokens = tokenizer.encode(text, false).unwrap();
        let decoded = tokenizer.decode(&tokens, false).unwrap();

        assert_eq!(tokens.len(), 3);
        assert!(decoded.contains("3 tokens"));
        assert_eq!(tokenizer.vocab_size(), 50257);
    }

    #[test]
    fn test_builder_creates_trait_objects() {
        let tokenizer1 = TokenizerBuilder::from_pretrained("gpt2").unwrap();
        let tokenizer2 = TokenizerBuilder::from_file("test.json").unwrap();

        // Both should work as trait objects
        let text = "builder test";
        let tokens1 = tokenizer1.encode(text, false).unwrap();
        let tokens2 = tokenizer2.encode(text, false).unwrap();

        assert_eq!(tokens1.len(), 2);
        assert_eq!(tokens2.len(), 2);
    }

    #[test]
    fn test_linguistic_validation_basic() {
        let tokenizer = BasicTokenizer::new();

        // Test basic linguistic patterns
        let sentences = vec![
            "The quick brown fox jumps over the lazy dog.",
            "Hello, how are you today?",
            "This is a test sentence with multiple words.",
            "Short text.",
            "A",
        ];

        for sentence in sentences {
            let tokens = tokenizer.encode(sentence, false).unwrap();
            let decoded = tokenizer.decode(&tokens, false).unwrap();

            // Basic validation: should produce tokens and decode successfully
            assert!(!tokens.is_empty());
            assert!(!decoded.is_empty());
            assert!(decoded.contains("tokens"));
        }
    }

    #[test]
    fn test_linguistic_validation_multilingual() {
        let tokenizer = BasicTokenizer::new();

        // Test with different languages (basic whitespace splitting should work)
        let multilingual_tests = vec![
            ("Hello world", 2),      // English
            ("Bonjour le monde", 3), // French - 3 words
            ("Hola mundo", 2),       // Spanish
            ("Hallo Welt", 2),       // German
            ("こんにちは 世界", 2),  // Japanese (with space)
            ("你好 世界", 2),        // Chinese (with space)
        ];

        for (text, expected_tokens) in multilingual_tests {
            let tokens = tokenizer.encode(text, false).unwrap();
            let decoded = tokenizer.decode(&tokens, false).unwrap();

            // Should handle all languages consistently
            assert_eq!(tokens.len(), expected_tokens);
            assert!(decoded.contains(&format!("{} tokens", expected_tokens)));
        }
    }

    #[test]
    fn test_coverage_all_public_methods() {
        let tokenizer = BasicTokenizer::new();

        // Test all public methods to ensure coverage
        assert_eq!(tokenizer.vocab_size(), 50257);
        assert_eq!(tokenizer.eos_token_id(), Some(50256));
        assert_eq!(tokenizer.pad_token_id(), None);

        let tokens = tokenizer.encode("test", true).unwrap();
        let _decoded = tokenizer.decode(&tokens, true).unwrap();

        // Test builder methods
        let _from_file = TokenizerBuilder::from_file("test").unwrap();
        let _from_pretrained = TokenizerBuilder::from_pretrained("test").unwrap();

        // Test with_config constructor
        let _custom = BasicTokenizer::with_config(1000, Some(999), Some(0));

        // Test default implementation
        let _default = BasicTokenizer::default();
    }
    #[test]
    fn test_comprehensive_tokenizer_formats() {
        // Test different tokenizer configurations to simulate various formats
        let configs = vec![
            ("gpt2", 50257, Some(50256), None),
            ("bert", 30522, Some(102), Some(0)),
            ("tiny", 1000, Some(999), Some(0)),
        ];

        for (name, vocab_size, eos_id, pad_id) in configs {
            let tokenizer = TokenizerBuilder::from_pretrained(name).unwrap();
            assert_eq!(tokenizer.vocab_size(), vocab_size);
            assert_eq!(tokenizer.eos_token_id(), eos_id);
            assert_eq!(tokenizer.pad_token_id(), pad_id);

            // Test encoding/decoding with this configuration
            let text = "test configuration";
            let tokens = tokenizer.encode(text, true).unwrap();
            let decoded = tokenizer.decode(&tokens, false).unwrap();

            assert!(!tokens.is_empty());
            assert!(!decoded.is_empty());
        }
    }

    #[test]
    fn test_tokenizer_memory_efficiency() {
        let tokenizer = BasicTokenizer::new();

        // Test with progressively larger inputs to check memory efficiency
        let sizes = vec![10, 100, 1000, 5000];

        for size in sizes {
            let words: Vec<String> = (0..size).map(|i| format!("word{}", i)).collect();
            let text = words.join(" ");

            let start_time = std::time::Instant::now();
            let tokens = tokenizer.encode(&text, false).unwrap();
            let encode_time = start_time.elapsed();

            let start_time = std::time::Instant::now();
            let _decoded = tokenizer.decode(&tokens, false).unwrap();
            let decode_time = start_time.elapsed();

            // Verify correctness
            assert_eq!(tokens.len(), size);

            // Performance should scale reasonably (less than 1ms per 1000 tokens)
            assert!(encode_time.as_millis() < size as u128 / 1000 + 1);
            assert!(decode_time.as_millis() < size as u128 / 1000 + 1);
        }
    }

    #[test]
    fn test_special_token_handling_comprehensive() {
        let tokenizer = BasicTokenizer::with_config(1000, Some(999), Some(0));

        // Test various combinations of special tokens
        let test_cases = vec![
            (vec![1, 2, 3], false, 3),           // No special tokens
            (vec![1, 999, 3], false, 3),         // EOS token, don't skip
            (vec![1, 999, 3], true, 2),          // EOS token, skip
            (vec![0, 1, 0, 2], false, 4),        // PAD tokens, don't skip
            (vec![0, 1, 0, 2], true, 2),         // PAD tokens, skip
            (vec![0, 999, 1, 0, 999], false, 5), // Mixed special tokens, don't skip
            (vec![0, 999, 1, 0, 999], true, 1),  // Mixed special tokens, skip
            (vec![999, 0, 999, 0], false, 4),    // Only special tokens, don't skip
            (vec![999, 0, 999, 0], true, 0),     // Only special tokens, skip (empty result)
        ];

        for (tokens, skip_special, expected_count) in test_cases {
            let decoded = tokenizer.decode(&tokens, skip_special).unwrap();

            if expected_count == 0 {
                assert_eq!(decoded, "");
            } else {
                assert!(decoded.contains(&format!("{} tokens", expected_count)));
            }
        }
    }

    #[test]
    fn test_linguistic_edge_cases() {
        let tokenizer = BasicTokenizer::new();

        // Test various linguistic edge cases
        let edge_cases = vec![
            ("", 0),                        // Empty string
            ("   ", 0),                     // Only whitespace
            ("a", 1),                       // Single character
            ("a b c d e f g h i j", 10),    // Many short words
            ("word\tword\nword\rword", 4),  // Different whitespace types
            ("CamelCaseWord", 1),           // CamelCase (treated as single word)
            ("hyphen-ated-word", 1),        // Hyphenated (treated as single word)
            ("email@domain.com", 1),        // Email (treated as single word)
            ("http://example.com/path", 1), // URL (treated as single word)
            ("123.456.789", 1),             // Dotted numbers (treated as single word)
            ("word1 word2 word3", 3),       // Mixed alphanumeric
        ];

        for (text, expected_tokens) in edge_cases {
            let tokens = tokenizer.encode(text, false).unwrap();
            assert_eq!(tokens.len(), expected_tokens, "Failed for text: '{}'", text);

            if expected_tokens > 0 {
                let decoded = tokenizer.decode(&tokens, false).unwrap();
                assert!(decoded.contains(&format!("{} tokens", expected_tokens)));
            }
        }
    }

    #[test]
    fn test_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let tokenizer = Arc::new(BasicTokenizer::new());
        let mut handles = vec![];

        // Spawn multiple threads to test concurrent access
        for i in 0..10 {
            let tokenizer_clone = Arc::clone(&tokenizer);
            let handle = thread::spawn(move || {
                let text = format!("thread {} test message", i);
                let tokens = tokenizer_clone.encode(&text, false).unwrap();
                let decoded = tokenizer_clone.decode(&tokens, false).unwrap();
                (tokens.len(), decoded)
            });
            handles.push(handle);
        }

        // Collect results
        for handle in handles {
            let (token_count, decoded) = handle.join().unwrap();
            assert_eq!(token_count, 4); // "thread", "X", "test", "message"
            assert!(decoded.contains("4 tokens"));
        }
    }

    #[test]
    fn test_error_handling() {
        // Test that the tokenizer handles various inputs gracefully
        let tokenizer = BasicTokenizer::new();

        // These should all succeed (no panics)
        assert!(tokenizer.encode("", false).is_ok());
        assert!(tokenizer.encode("normal text", false).is_ok());
        assert!(tokenizer.encode("🚀🌟✨", false).is_ok());
        assert!(tokenizer.encode(&"x".repeat(100000), false).is_ok());

        assert!(tokenizer.decode(&[], false).is_ok());
        assert!(tokenizer.decode(&[0, 1, 2], false).is_ok());
        assert!(tokenizer.decode(&[u32::MAX], false).is_ok());

        // Test builder methods
        assert!(TokenizerBuilder::from_file("nonexistent.json").is_ok());
        assert!(TokenizerBuilder::from_pretrained("unknown").is_ok());
    }
}
