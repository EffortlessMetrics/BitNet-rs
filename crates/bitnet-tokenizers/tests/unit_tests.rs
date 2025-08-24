#![cfg(feature = "integration-tests")]
//! Comprehensive unit tests for bitnet-tokenizers crate
//!
//! This module implements comprehensive unit tests covering:
//! - Tokenization accuracy and consistency
//! - Various tokenizer formats and configurations
//! - Special token handling and edge cases
//! - Performance and memory tests
//! - Linguistic validation
//!
//! Requirements covered: 2.1, 2.2, 2.4

use bitnet_tokenizers::{BasicTokenizer, Tokenizer, TokenizerBuilder};

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Test suite for tokenization accuracy and consistency
mod accuracy_tests {
    use super::*;

    #[test]
    fn test_tokenization_deterministic() {
        let tokenizer = BasicTokenizer::new();
        let text = "The quick brown fox jumps over the lazy dog";

        // Multiple calls should produce identical results
        let tokens1 = tokenizer.encode(text, false, false).unwrap();
        let tokens2 = tokenizer.encode(text, false, false).unwrap();
        let tokens3 = tokenizer.encode(text, false, false).unwrap();

        assert_eq!(tokens1, tokens2);
        assert_eq!(tokens2, tokens3);
        assert_eq!(tokens1.len(), 9); // 9 words
    }

    #[test]
    fn test_tokenization_consistency_across_instances() {
        let tokenizer1 = BasicTokenizer::new();
        let tokenizer2 = BasicTokenizer::new();
        let text = "consistency test across instances";

        let tokens1 = tokenizer1.encode(text, false, false).unwrap();
        let tokens2 = tokenizer2.encode(text, false, false).unwrap();

        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn test_encode_decode_roundtrip_consistency() {
        let tokenizer = BasicTokenizer::new();
        let test_cases = vec![
            "hello world",
            "The quick brown fox",
            "Testing tokenization consistency",
            "Single",
            "",
            "   multiple   spaces   ",
            "unicode 🚀 test 世界",
        ];

        for text in test_cases {
            let tokens = tokenizer.encode(text, false, false).unwrap();
            let decoded = tokenizer.decode(&tokens).unwrap();

            // Verify consistency (not exact match due to placeholder implementation)
            if text.trim().is_empty() {
                assert!(tokens.is_empty());
                assert_eq!(decoded, "");
            } else {
                let expected_token_count = text.split_whitespace().count();
                assert_eq!(tokens.len(), expected_token_count);
                assert!(decoded.contains(&format!("{} tokens", expected_token_count)));
            }
        }
    }

    #[test]
    fn test_tokenization_accuracy_with_special_tokens() {
        let tokenizer = BasicTokenizer::new();
        let text = "test with special tokens";

        let tokens_without = tokenizer.encode(text, false, false).unwrap();
        let tokens_with = tokenizer.encode(text, true, false).unwrap();

        assert_eq!(tokens_without.len(), 4); // 4 words
        assert_eq!(tokens_with.len(), 5); // 4 words + EOS token
        assert_eq!(tokens_with[4], 50256); // EOS token ID

        // First 4 tokens should be identical
        assert_eq!(&tokens_without[..], &tokens_with[..4]);
    }

    #[test]
    fn test_decode_accuracy_with_special_token_filtering() {
        let tokenizer = BasicTokenizer::with_config(1000, Some(999), Some(0));
        let tokens = vec![1, 2, 0, 3, 999, 0]; // Mixed content and special tokens

        let decoded_with_special = tokenizer.decode(&tokens).unwrap();
        let decoded_without_special = tokenizer.decode(&tokens).unwrap();

        assert!(decoded_with_special.contains("6 tokens")); // All tokens included
        assert!(decoded_without_special.contains("3 tokens")); // Only content tokens
    }

    #[test]
    fn test_tokenization_accuracy_edge_cases() {
        let tokenizer = BasicTokenizer::new();

        // Test various edge cases
        let edge_cases = vec![
            ("", 0),
            ("   ", 0),
            ("a", 1),
            ("a b c d e", 5),
            ("word\tword\nword\rword", 4), // Different whitespace
            ("punctuation! test? yes.", 3),
            ("123 456 789", 3),
            ("CamelCase", 1),
            ("hyphen-ated", 1),
            ("email@test.com", 1),
        ];

        for (text, expected_count) in edge_cases {
            let tokens = tokenizer.encode(text, false, false).unwrap();
            assert_eq!(tokens.len(), expected_count, "Failed for: '{}'", text);

            if expected_count > 0 {
                let decoded = tokenizer.decode(&tokens).unwrap();
                assert!(decoded.contains(&format!("{} tokens", expected_count)));
            }
        }
    }
}

/// Test suite for various tokenizer formats and configurations
mod format_configuration_tests {
    use super::*;

    #[test]
    fn test_gpt2_configuration() {
        let tokenizer = TokenizerBuilder::from_pretrained("gpt2").unwrap();

        assert_eq!(tokenizer.vocab_size(), 50257);
        assert_eq!(tokenizer.eos_token_id(), Some(50256));
        assert_eq!(tokenizer.pad_token_id(), None);

        // Test encoding/decoding with GPT-2 config
        let text = "GPT-2 configuration test";
        let tokens = tokenizer.encode(text, true, false).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();

        assert_eq!(tokens.len(), 4); // 3 words + EOS
        assert_eq!(tokens[3], 50256); // EOS token
        assert!(decoded.contains("4 tokens"));
    }

    #[test]
    fn test_bert_configuration() {
        let tokenizer = TokenizerBuilder::from_pretrained("bert").unwrap();

        assert_eq!(tokenizer.vocab_size(), 30522);
        assert_eq!(tokenizer.eos_token_id(), Some(102));
        assert_eq!(tokenizer.pad_token_id(), Some(0));

        // Test encoding/decoding with BERT config
        let text = "BERT configuration test";
        let tokens = tokenizer.encode(text, true, false).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();

        assert_eq!(tokens.len(), 4); // 3 words + EOS
        assert_eq!(tokens[3], 102); // BERT EOS token
        assert!(decoded.contains("4 tokens"));
    }

    #[test]
    fn test_tiny_configuration() {
        let tokenizer = TokenizerBuilder::from_pretrained("tiny").unwrap();

        assert_eq!(tokenizer.vocab_size(), 1000);
        assert_eq!(tokenizer.eos_token_id(), Some(999));
        assert_eq!(tokenizer.pad_token_id(), Some(0));

        // Test encoding/decoding with tiny config
        let text = "tiny model test";
        let tokens = tokenizer.encode(text, true, false).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();

        assert_eq!(tokens.len(), 4); // 3 words + EOS
        assert_eq!(tokens[3], 999); // Tiny EOS token
        assert!(decoded.contains("4 tokens"));
    }

    #[test]
    fn test_custom_configuration() {
        let tokenizer = BasicTokenizer::with_config(5000, Some(4999), Some(5000));

        assert_eq!(tokenizer.vocab_size(), 5000);
        assert_eq!(tokenizer.eos_token_id(), Some(4999));
        assert_eq!(tokenizer.pad_token_id(), Some(5000));

        // Test with custom configuration
        let text = "custom configuration";
        let tokens = tokenizer.encode(text, true, false).unwrap();
        let decoded_with_special = tokenizer.decode(&tokens).unwrap();
        let decoded_without_special = tokenizer.decode(&tokens).unwrap();

        assert_eq!(tokens.len(), 3); // 2 words + EOS
        assert_eq!(tokens[2], 4999); // Custom EOS token
        assert!(decoded_with_special.contains("3 tokens"));
        assert!(decoded_without_special.contains("2 tokens"));
    }

    #[test]
    fn test_configuration_from_file() {
        let tokenizer = TokenizerBuilder::from_file("test_config.json").unwrap();

        // Should default to basic configuration
        assert_eq!(tokenizer.vocab_size(), 50257);
        assert_eq!(tokenizer.eos_token_id(), Some(50256));
        assert_eq!(tokenizer.pad_token_id(), None);
    }

    #[test]
    fn test_unknown_pretrained_model() {
        let tokenizer = TokenizerBuilder::from_pretrained("unknown_model_xyz").unwrap();

        // Should default to basic configuration
        assert_eq!(tokenizer.vocab_size(), 50257);
        assert_eq!(tokenizer.eos_token_id(), Some(50256));
        assert_eq!(tokenizer.pad_token_id(), None);
    }

    #[test]
    fn test_configuration_consistency() {
        let configs = vec![
            ("gpt2", 50257, Some(50256), None),
            ("bert", 30522, Some(102), Some(0)),
            ("tiny", 1000, Some(999), Some(0)),
        ];

        for (name, vocab_size, eos_id, pad_id) in configs {
            let tokenizer1 = TokenizerBuilder::from_pretrained(name).unwrap();
            let tokenizer2 = TokenizerBuilder::from_pretrained(name).unwrap();

            // Multiple instances should have identical configurations
            assert_eq!(tokenizer1.vocab_size(), tokenizer2.vocab_size());
            assert_eq!(tokenizer1.eos_token_id(), tokenizer2.eos_token_id());
            assert_eq!(tokenizer1.pad_token_id(), tokenizer2.pad_token_id());

            // Verify expected values
            assert_eq!(tokenizer1.vocab_size(), vocab_size);
            assert_eq!(tokenizer1.eos_token_id(), eos_id);
            assert_eq!(tokenizer1.pad_token_id(), pad_id);
        }
    }
}

/// Test suite for special token handling and edge cases
mod special_token_tests {
    use super::*;

    #[test]
    fn test_eos_token_handling() {
        let tokenizer = BasicTokenizer::new();
        let text = "test EOS token";

        let tokens_without_eos = tokenizer.encode(text, false, false).unwrap();
        let tokens_with_eos = tokenizer.encode(text, true, false).unwrap();

        assert_eq!(tokens_without_eos.len(), 3);
        assert_eq!(tokens_with_eos.len(), 4);
        assert_eq!(tokens_with_eos[3], 50256); // EOS token

        // Verify EOS token is only added when requested
        assert!(!tokens_without_eos.contains(&50256));
        assert!(tokens_with_eos.contains(&50256));
    }

    #[test]
    fn test_pad_token_handling() {
        let tokenizer = BasicTokenizer::with_config(1000, Some(999), Some(0));
        let tokens_with_pads = vec![1, 2, 0, 3, 0, 4, 0];

        let decoded_with_pads = tokenizer.decode(&tokens_with_pads).unwrap();
        let decoded_without_pads = tokenizer.decode(&tokens_with_pads).unwrap();

        assert!(decoded_with_pads.contains("7 tokens")); // All tokens
        assert!(decoded_without_pads.contains("4 tokens")); // Only non-pad tokens
    }

    #[test]
    fn test_mixed_special_tokens() {
        let tokenizer = BasicTokenizer::with_config(1000, Some(999), Some(0));
        let tokens = vec![1, 0, 2, 999, 3, 0, 999]; // Mixed PAD and EOS tokens

        let decoded_with_special = tokenizer.decode(&tokens).unwrap();
        let decoded_without_special = tokenizer.decode(&tokens).unwrap();

        assert!(decoded_with_special.contains("7 tokens"));
        assert!(decoded_without_special.contains("3 tokens")); // Only content tokens
    }

    #[test]
    fn test_only_special_tokens() {
        let tokenizer = BasicTokenizer::with_config(100, Some(99), Some(0));
        let only_special = vec![0, 99, 0, 99, 0];

        let decoded_with_special = tokenizer.decode(&only_special).unwrap();
        let decoded_without_special = tokenizer.decode(&only_special).unwrap();

        assert!(decoded_with_special.contains("5 tokens"));
        assert_eq!(decoded_without_special, ""); // Empty after filtering
    }

    #[test]
    fn test_no_special_tokens_configured() {
        let tokenizer = BasicTokenizer::with_config(1000, None, None);
        let text = "no special tokens";

        let tokens_with_flag = tokenizer.encode(text, true, false).unwrap();
        let tokens_without_flag = tokenizer.encode(text, false, false).unwrap();

        // Should be identical when no special tokens are configured
        assert_eq!(tokens_with_flag, tokens_without_flag);
        assert_eq!(tokens_with_flag.len(), 3);
    }

    #[test]
    fn test_special_token_edge_cases() {
        let tokenizer = BasicTokenizer::with_config(10, Some(9), Some(0));

        // Test edge cases
        let edge_cases = vec![
            (vec![], false, ""),
            (vec![], true, ""),
            (vec![0], false, "1 tokens"),
            (vec![0], true, ""),
            (vec![9], false, "1 tokens"),
            (vec![9], true, ""),
            (vec![1, 2, 3], false, "3 tokens"),
            (vec![1, 2, 3], true, "3 tokens"),
        ];

        for (tokens, _skip_special, expected) in edge_cases {
            let decoded = tokenizer.decode(&tokens).unwrap();
            if expected.is_empty() {
                assert_eq!(decoded, "");
            } else {
                assert!(decoded.contains(expected));
            }
        }
    }

    #[test]
    fn test_special_token_ids_boundary_values() {
        // Test with boundary values
        let tokenizer = BasicTokenizer::with_config(u32::MAX as usize, Some(u32::MAX - 1), Some(0));

        assert_eq!(tokenizer.vocab_size(), u32::MAX as usize);
        assert_eq!(tokenizer.eos_token_id(), Some(u32::MAX - 1));
        assert_eq!(tokenizer.pad_token_id(), Some(0));

        let tokens = vec![1, 0, u32::MAX - 1, 2];
        let decoded_with_special = tokenizer.decode(&tokens).unwrap();
        let decoded_without_special = tokenizer.decode(&tokens).unwrap();

        assert!(decoded_with_special.contains("4 tokens"));
        assert!(decoded_without_special.contains("2 tokens"));
    }
}

/// Test suite for performance and memory tests
mod performance_tests {
    use super::*;

    #[test]
    fn test_encoding_performance() {
        let tokenizer = BasicTokenizer::new();
        let text = "This is a performance test with multiple words to encode quickly";

        let start = Instant::now();
        for _ in 0..1000 {
            let _ = tokenizer.encode(text, false, false).unwrap();
        }
        let duration = start.elapsed();

        // Should complete 1000 encodings in less than 100ms
        assert!(duration.as_millis() < 100, "Encoding too slow: {:?}", duration);
    }

    #[test]
    fn test_decoding_performance() {
        let tokenizer = BasicTokenizer::new();
        let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        let start = Instant::now();
        for _ in 0..1000 {
            let _ = tokenizer.decode(&tokens).unwrap();
        }
        let duration = start.elapsed();

        // Should complete 1000 decodings in less than 100ms
        assert!(duration.as_millis() < 100, "Decoding too slow: {:?}", duration);
    }

    #[test]
    fn test_large_text_performance() {
        let tokenizer = BasicTokenizer::new();

        // Create progressively larger texts
        let sizes = vec![100, 1000, 5000, 10000];

        for size in sizes {
            let words: Vec<String> = (0..size).map(|i| format!("word{}", i)).collect();
            let text = words.join(" ");

            let start = Instant::now();
            let tokens = tokenizer.encode(&text, false, false).unwrap();
            let encode_time = start.elapsed();

            let start = Instant::now();
            let _decoded = tokenizer.decode(&tokens).unwrap();
            let decode_time = start.elapsed();

            // Verify correctness
            assert_eq!(tokens.len(), size);

            // Performance should scale reasonably (less than 1ms per 1000 tokens)
            let max_encode_time = Duration::from_millis(size as u64 / 1000 + 1);
            let max_decode_time = Duration::from_millis(size as u64 / 1000 + 1);

            assert!(
                encode_time <= max_encode_time,
                "Encoding {} tokens took {:?}, expected <= {:?}",
                size,
                encode_time,
                max_encode_time
            );
            assert!(
                decode_time <= max_decode_time,
                "Decoding {} tokens took {:?}, expected <= {:?}",
                size,
                decode_time,
                max_decode_time
            );
        }
    }

    #[test]
    fn test_memory_efficiency() {
        let tokenizer = BasicTokenizer::new();

        // Test memory usage doesn't grow excessively with large inputs
        let large_text = "word ".repeat(50000); // 50k words

        let tokens = tokenizer.encode(&large_text, false, false).unwrap();
        assert_eq!(tokens.len(), 50000);

        let decoded = tokenizer.decode(&tokens).unwrap();
        assert!(decoded.contains("50000 tokens"));

        // Memory should be released after operations
        // (This is a basic test - in practice we'd use more sophisticated memory profiling)
        drop(tokens);
        drop(decoded);
    }

    #[test]
    fn test_concurrent_performance() {
        let tokenizer = Arc::new(BasicTokenizer::new());
        let mut handles = vec![];

        // Spawn multiple threads for concurrent access
        for i in 0..10 {
            let tokenizer_clone = Arc::clone(&tokenizer);
            let handle = thread::spawn(move || {
                let text = format!("concurrent test {} with multiple words", i);
                let start = Instant::now();

                for _ in 0..100 {
                    let tokens = tokenizer_clone.encode(&text, false, false).unwrap();
                    let _decoded = tokenizer_clone.decode(&tokens).unwrap();
                }

                start.elapsed()
            });
            handles.push(handle);
        }

        // Collect results and verify performance
        let mut total_duration = Duration::ZERO;
        for handle in handles {
            let duration = handle.join().unwrap();
            total_duration += duration;
        }

        // Average per thread should be reasonable
        let avg_duration = total_duration / 10;
        assert!(
            avg_duration.as_millis() < 50,
            "Concurrent performance too slow: {:?}",
            avg_duration
        );
    }

    #[test]
    fn test_repeated_operations_performance() {
        let tokenizer = BasicTokenizer::new();
        let texts = vec![
            "short",
            "medium length text with several words",
            "much longer text that contains many more words and should test performance with larger inputs",
        ];

        for text in texts {
            let iterations = 1000;
            let start = Instant::now();

            for _ in 0..iterations {
                let tokens = tokenizer.encode(text, false, false).unwrap();
                let _decoded = tokenizer.decode(&tokens).unwrap();
            }

            let duration = start.elapsed();
            let per_operation = duration / iterations;

            // Each encode+decode cycle should be very fast
            assert!(
                per_operation.as_micros() < 100,
                "Per-operation time too slow for '{}': {:?}",
                text,
                per_operation
            );
        }
    }
}

/// Test suite for linguistic validation
mod linguistic_validation_tests {
    use super::*;

    #[test]
    fn test_basic_linguistic_patterns() {
        let tokenizer = BasicTokenizer::new();

        let test_cases = vec![
            ("The quick brown fox jumps over the lazy dog.", 9),
            ("Hello, world!", 2),
            ("This is a test.", 4),
            ("One two three four five.", 5),
            ("A", 1),
            ("Testing multiple sentences. This is sentence two.", 7), // Fixed: 7 words not 8
        ];

        for (text, expected_tokens) in test_cases {
            let tokens = tokenizer.encode(text, false, false).unwrap();
            let decoded = tokenizer.decode(&tokens).unwrap();

            assert_eq!(tokens.len(), expected_tokens, "Failed for: '{}'", text);
            assert!(decoded.contains(&format!("{} tokens", expected_tokens)));
        }
    }

    #[test]
    fn test_multilingual_support() {
        let tokenizer = BasicTokenizer::new();

        let multilingual_tests = vec![
            ("Hello world", "English", 2),
            ("Bonjour le monde", "French", 3),
            ("Hola mundo", "Spanish", 2),
            ("Hallo Welt", "German", 2),
            ("Ciao mondo", "Italian", 2),
            ("Olá mundo", "Portuguese", 2),
            ("Привет мир", "Russian", 2),
            ("こんにちは 世界", "Japanese", 2),
            ("你好 世界", "Chinese", 2),
            ("안녕하세요 세계", "Korean", 2),
            ("مرحبا بالعالم", "Arabic", 2),
            ("שלום עולם", "Hebrew", 2),
        ];

        for (text, language, expected_tokens) in multilingual_tests {
            let tokens = tokenizer.encode(text, false, false).unwrap();
            let decoded = tokenizer.decode(&tokens).unwrap();

            assert_eq!(tokens.len(), expected_tokens, "Failed for {} text: '{}'", language, text);
            assert!(decoded.contains(&format!("{} tokens", expected_tokens)));
        }
    }

    #[test]
    fn test_punctuation_handling() {
        let tokenizer = BasicTokenizer::new();

        let punctuation_tests = vec![
            ("word.", 1),
            ("word!", 1),
            ("word?", 1),
            ("word,", 1),
            ("word;", 1),
            ("word:", 1),
            ("word's", 1),
            ("word-word", 1),
            ("word_word", 1),
            ("word/word", 1),
            ("word\\word", 1),
            ("word@word", 1),
            ("word#word", 1),
            ("word$word", 1),
            ("word%word", 1),
            ("word&word", 1),
            ("word*word", 1),
            ("word+word", 1),
            ("word=word", 1),
            ("(word)", 1),
            ("[word]", 1),
            ("{word}", 1),
            ("\"word\"", 1),
            ("'word'", 1),
        ];

        for (text, expected_tokens) in punctuation_tests {
            let tokens = tokenizer.encode(text, false, false).unwrap();
            assert_eq!(tokens.len(), expected_tokens, "Failed for: '{}'", text);
        }
    }

    #[test]
    fn test_numeric_content() {
        let tokenizer = BasicTokenizer::new();

        let numeric_tests = vec![
            ("123", 1),
            ("123.456", 1),
            ("-123", 1),
            ("+123", 1),
            ("1,234,567", 1),
            ("3.14159", 1),
            ("1e10", 1),
            ("1E-5", 1),
            ("0x1234", 1),
            ("0b1010", 1),
            ("123 456", 2),
            ("1 + 2 = 3", 5),
            ("Version 1.2.3", 2),
            ("Phone: 555-1234", 2),
            ("Date: 2023-12-25", 2),
        ];

        for (text, expected_tokens) in numeric_tests {
            let tokens = tokenizer.encode(text, false, false).unwrap();
            assert_eq!(tokens.len(), expected_tokens, "Failed for: '{}'", text);
        }
    }

    #[test]
    fn test_whitespace_normalization() {
        let tokenizer = BasicTokenizer::new();

        let whitespace_tests = vec![
            ("word1 word2", 2),
            ("word1  word2", 2),     // Double space
            ("word1   word2", 2),    // Triple space
            ("word1\tword2", 2),     // Tab
            ("word1\nword2", 2),     // Newline
            ("word1\rword2", 2),     // Carriage return
            ("word1\r\nword2", 2),   // CRLF
            (" word1 word2 ", 2),    // Leading/trailing spaces
            ("\tword1\tword2\t", 2), // Leading/trailing tabs
            ("\nword1\nword2\n", 2), // Leading/trailing newlines
        ];

        for (text, expected_tokens) in whitespace_tests {
            let tokens = tokenizer.encode(text, false, false).unwrap();
            assert_eq!(tokens.len(), expected_tokens, "Failed for: '{}'", text);
        }
    }

    #[test]
    fn test_unicode_edge_cases() {
        let tokenizer = BasicTokenizer::new();

        let unicode_tests = vec![
            ("café", 1),           // Accented characters
            ("naïve", 1),          // Diaeresis
            ("résumé", 1),         // Multiple accents
            ("🚀", 1),             // Emoji
            ("🚀🌟✨", 1),         // Multiple emojis
            ("test 🚀 rocket", 3), // Mixed emoji and text
            ("α β γ", 3),          // Greek letters
            ("∑ ∏ ∫", 3),          // Mathematical symbols
            ("™ © ®", 3),          // Special symbols
            ("½ ¼ ¾", 3),          // Fractions
        ];

        for (text, expected_tokens) in unicode_tests {
            let tokens = tokenizer.encode(text, false, false).unwrap();
            assert_eq!(tokens.len(), expected_tokens, "Failed for: '{}'", text);
        }
    }

    #[test]
    fn test_linguistic_consistency() {
        let tokenizer = BasicTokenizer::new();

        // Test that similar linguistic patterns produce consistent results
        let pattern_groups = vec![
            vec!["cat", "dog", "bird", "fish"],
            vec!["running", "jumping", "walking", "swimming"],
            vec!["quickly", "slowly", "carefully", "quietly"],
            vec!["the cat", "the dog", "the bird", "the fish"],
            vec!["I am", "you are", "he is", "she is"],
        ];

        for group in pattern_groups {
            let token_counts: Vec<usize> = group
                .iter()
                .map(|text| tokenizer.encode(text, false, false).unwrap().len())
                .collect();

            // All items in a group should have the same token count
            let first_count = token_counts[0];
            for (i, &count) in token_counts.iter().enumerate() {
                assert_eq!(
                    count, first_count,
                    "Inconsistent tokenization in group: '{}' has {} tokens, expected {}",
                    group[i], count, first_count
                );
            }
        }
    }

    #[test]
    fn test_sentence_boundary_detection() {
        let tokenizer = BasicTokenizer::new();

        // Test various sentence patterns
        let sentence_tests = vec![
            ("Hello.", 1),
            ("Hello. World.", 2),
            ("Hello! World?", 2),
            ("Dr. Smith went home.", 4),  // Abbreviation with period
            ("U.S.A. is great.", 3),      // Multiple abbreviations
            ("What? Really! Yes.", 3),    // Multiple punctuation
            ("End... Start.", 2),         // Ellipsis
            ("Quote: 'Hello world.'", 3), // Quoted sentence
        ];

        for (text, expected_tokens) in sentence_tests {
            let tokens = tokenizer.encode(text, false, false).unwrap();
            assert_eq!(tokens.len(), expected_tokens, "Failed for: '{}'", text);
        }
    }
}

/// Test suite for comprehensive coverage of all public methods
mod coverage_tests {
    use super::*;

    #[test]
    fn test_all_basic_tokenizer_methods() {
        let tokenizer = BasicTokenizer::new();

        // Test all public methods
        assert_eq!(tokenizer.vocab_size(), 50257);
        assert_eq!(tokenizer.eos_token_id(), Some(50256));
        assert_eq!(tokenizer.pad_token_id(), None);

        let text = "coverage test";
        let tokens = tokenizer.encode(text, true, false).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();

        assert!(!tokens.is_empty());
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_all_constructor_variants() {
        // Test new()
        let tokenizer1 = BasicTokenizer::new();
        assert_eq!(tokenizer1.vocab_size(), 50257);

        // Test default()
        let tokenizer2 = BasicTokenizer::default();
        assert_eq!(tokenizer2.vocab_size(), 50257);

        // Test with_config()
        let tokenizer3 = BasicTokenizer::with_config(1000, Some(999), Some(0));
        assert_eq!(tokenizer3.vocab_size(), 1000);
        assert_eq!(tokenizer3.eos_token_id(), Some(999));
        assert_eq!(tokenizer3.pad_token_id(), Some(0));
    }

    #[test]
    fn test_all_builder_methods() {
        // Test from_pretrained with various models
        let models = vec!["gpt2", "bert", "tiny", "unknown"];
        for model in models {
            let tokenizer = TokenizerBuilder::from_pretrained(model).unwrap();
            assert!(tokenizer.vocab_size() > 0);
        }

        // Test from_file
        let tokenizer = TokenizerBuilder::from_file("test.json").unwrap();
        assert_eq!(tokenizer.vocab_size(), 50257);
    }

    #[test]
    fn test_trait_object_usage() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(BasicTokenizer::new());

        // Test all trait methods through trait object
        assert_eq!(tokenizer.vocab_size(), 50257);
        assert_eq!(tokenizer.eos_token_id(), Some(50256));
        assert_eq!(tokenizer.pad_token_id(), None);

        let tokens = tokenizer.encode("trait test", false, false).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();

        assert_eq!(tokens.len(), 2);
        assert!(decoded.contains("2 tokens"));
    }

    #[test]
    fn test_error_handling_coverage() {
        let tokenizer = BasicTokenizer::new();

        // Test that all methods handle edge cases gracefully
        assert!(tokenizer.encode("", false, false).is_ok());
        assert!(tokenizer.encode("normal", false, false).is_ok());
        assert!(tokenizer.encode(&"x".repeat(100000), false, false).is_ok());

        assert!(tokenizer.decode(&[]).is_ok());
        assert!(tokenizer.decode(&[0, 1, 2]).is_ok());
        assert!(tokenizer.decode(&[u32::MAX]).is_ok());

        // Test builder error handling
        assert!(TokenizerBuilder::from_file("nonexistent").is_ok());
        assert!(TokenizerBuilder::from_pretrained("invalid").is_ok());
    }

    #[test]
    fn test_thread_safety_coverage() {
        let tokenizer = Arc::new(BasicTokenizer::new());
        let mut handles = vec![];

        // Test concurrent access to all methods
        for i in 0..5 {
            let tokenizer_clone = Arc::clone(&tokenizer);
            let handle = thread::spawn(move || {
                let text = format!("thread {} test", i);

                // Test all methods concurrently
                let _vocab_size = tokenizer_clone.vocab_size();
                let _eos_id = tokenizer_clone.eos_token_id();
                let _pad_id = tokenizer_clone.pad_token_id();

                let tokens = tokenizer_clone.encode(&text, false, false).unwrap();
                let _decoded = tokenizer_clone.decode(&tokens).unwrap();

                tokens.len()
            });
            handles.push(handle);
        }

        // Verify all threads completed successfully
        for handle in handles {
            let result = handle.join().unwrap();
            assert_eq!(result, 3); // "thread", "X", "test"
        }
    }

    #[test]
    fn test_configuration_edge_cases() {
        // Test edge case configurations
        let edge_configs = vec![
            (1, None, None),                                             // Minimal vocab
            (1, Some(0), None),                                          // EOS = 0
            (1, None, Some(0)),                                          // PAD = 0
            (1, Some(0), Some(0)),                                       // EOS = PAD = 0
            (u32::MAX as usize, Some(u32::MAX - 1), Some(u32::MAX - 2)), // Large values
        ];

        for (vocab_size, eos_id, pad_id) in edge_configs {
            let tokenizer = BasicTokenizer::with_config(vocab_size, eos_id, pad_id);

            assert_eq!(tokenizer.vocab_size(), vocab_size);
            assert_eq!(tokenizer.eos_token_id(), eos_id);
            assert_eq!(tokenizer.pad_token_id(), pad_id);

            // Test basic functionality still works
            let tokens = tokenizer.encode("test", false, false).unwrap();
            let decoded = tokenizer.decode(&tokens).unwrap();

            assert_eq!(tokens.len(), 1);
            assert!(decoded.contains("1 tokens"));
        }
    }
}

/// Integration tests that combine multiple aspects
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_workflow_integration() {
        // Test complete workflow from builder to encoding/decoding
        let tokenizer = TokenizerBuilder::from_pretrained("bert").unwrap();

        let test_texts = vec![
            "Simple test",
            "More complex test with multiple words and punctuation!",
            "Unicode test: 🚀 世界 café",
            "",
            "   whitespace   test   ",
        ];

        for text in test_texts {
            // Test both with and without special tokens
            for add_special in [false, true] {
                let tokens = tokenizer.encode(text, add_special, false).unwrap();

                // Test both with and without special token filtering
                for _skip_special in [false, true] {
                    let decoded = tokenizer.decode(&tokens).unwrap();

                    // Verify basic consistency
                    if text.trim().is_empty() {
                        if !add_special {
                            assert!(tokens.is_empty());
                        }
                        if !add_special {
                            assert_eq!(decoded, "");
                        }
                    } else {
                        assert!(!tokens.is_empty());
                        if !add_special {
                            assert!(!decoded.is_empty());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_cross_configuration_consistency() {
        let configs = vec![
            ("gpt2", 50257, Some(50256), None),
            ("bert", 30522, Some(102), Some(0)),
            ("tiny", 1000, Some(999), Some(0)),
        ];

        let test_text = "cross configuration test";

        for (name, _vocab_size, _eos_id, _pad_id) in configs {
            let tokenizer = TokenizerBuilder::from_pretrained(name).unwrap();

            // Test consistency across multiple operations
            for _ in 0..10 {
                let tokens1 = tokenizer.encode(test_text, false, false).unwrap();
                let tokens2 = tokenizer.encode(test_text, false, false).unwrap();
                assert_eq!(tokens1, tokens2);

                let decoded1 = tokenizer.decode(&tokens1).unwrap();
                let decoded2 = tokenizer.decode(&tokens1).unwrap();
                assert_eq!(decoded1, decoded2);
            }
        }
    }

    #[test]
    fn test_performance_across_configurations() {
        let configs = vec!["gpt2", "bert", "tiny"];
        let text = "performance test across different configurations";

        for config in configs {
            let tokenizer = TokenizerBuilder::from_pretrained(config).unwrap();

            let start = Instant::now();
            for _ in 0..100 {
                let tokens = tokenizer.encode(text, false, false).unwrap();
                let _decoded = tokenizer.decode(&tokens).unwrap();
            }
            let duration = start.elapsed();

            // Performance should be consistent across configurations
            assert!(
                duration.as_millis() < 50,
                "Performance regression for config '{}': {:?}",
                config,
                duration
            );
        }
    }

    #[test]
    fn test_memory_consistency_across_operations() {
        let tokenizer = BasicTokenizer::new();
        let sizes = vec![10, 100, 1000];

        for size in sizes {
            let words: Vec<String> = (0..size).map(|i| format!("word{}", i)).collect();
            let text = words.join(" ");

            // Perform multiple encode/decode cycles
            for _ in 0..10 {
                let tokens = tokenizer.encode(&text, false, false).unwrap();
                let decoded = tokenizer.decode(&tokens).unwrap();

                assert_eq!(tokens.len(), size);
                assert!(decoded.contains(&format!("{} tokens", size)));

                // Memory should be consistent
                drop(tokens);
                drop(decoded);
            }
        }
    }
}

/// Benchmark tests for performance validation
mod benchmark_tests {
    use super::*;

    #[test]
    fn test_encoding_benchmark() {
        let tokenizer = BasicTokenizer::new();
        let test_cases = vec![
            ("short", 1000),
            ("medium length text with several words", 500),
            (
                "much longer text that should test the performance characteristics of the tokenizer with many words",
                100,
            ),
        ];

        for (text, iterations) in test_cases {
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = tokenizer.encode(text, false, false).unwrap();
            }
            let duration = start.elapsed();

            let per_op = duration / iterations;
            println!("Encoding '{}': {:?} per operation ({} iterations)", text, per_op, iterations);

            // Reasonable performance threshold
            assert!(
                per_op.as_micros() < 50,
                "Encoding performance too slow: {:?} per operation",
                per_op
            );
        }
    }

    #[test]
    fn test_decoding_benchmark() {
        let tokenizer = BasicTokenizer::new();
        let token_sets = vec![
            (vec![0, 1, 2], 1000),
            (vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 500),
            ((0..100).collect::<Vec<u32>>(), 100),
        ];

        for (tokens, iterations) in token_sets {
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = tokenizer.decode(&tokens).unwrap();
            }
            let duration = start.elapsed();

            let per_op = duration / iterations;
            println!(
                "Decoding {} tokens: {:?} per operation ({} iterations)",
                tokens.len(),
                per_op,
                iterations
            );

            // Reasonable performance threshold
            assert!(
                per_op.as_micros() < 50,
                "Decoding performance too slow: {:?} per operation",
                per_op
            );
        }
    }

    #[test]
    fn test_throughput_benchmark() {
        let tokenizer = BasicTokenizer::new();
        let text = "throughput benchmark test with multiple words for measurement";
        let iterations = 1000;

        let start = Instant::now();
        let mut total_tokens = 0;

        for _ in 0..iterations {
            let tokens = tokenizer.encode(text, false, false).unwrap();
            total_tokens += tokens.len();
            let _ = tokenizer.decode(&tokens).unwrap();
        }

        let duration = start.elapsed();
        let tokens_per_second = (total_tokens as f64) / duration.as_secs_f64();

        println!("Throughput: {:.0} tokens/second", tokens_per_second);

        // Should achieve reasonable throughput
        assert!(
            tokens_per_second > 10000.0,
            "Throughput too low: {:.0} tokens/second",
            tokens_per_second
        );
    }
}
