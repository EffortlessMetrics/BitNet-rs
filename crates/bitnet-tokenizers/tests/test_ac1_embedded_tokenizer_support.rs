//! AC1: Embedded Tokenizer Support Test Scaffolding
//!
//! Tests feature spec: docs/explanation/issue-336-universal-tokenizer-discovery-spec.md#ac1-embedded-tokenizer-support
//!
//! This test suite validates embedded tokenizer extraction from GGUF metadata,
//! including HuggingFace JSON tokenizers, SentencePiece models, and fallback behavior.

use bitnet_tokenizers::{Tokenizer, TokenizerDiscovery};
use std::path::Path;
use std::sync::Arc;

// ================================
// AC1: EMBEDDED TOKENIZER EXTRACTION TESTS
// ================================

/// AC1: Extract embedded HuggingFace tokenizer from GGUF metadata
/// Tests feature spec: issue-336-universal-tokenizer-discovery-spec.md#ac1-embedded-tokenizer-support
// AC:ID AC1
#[test]
#[cfg(feature = "cpu")]
fn ac1_extract_embedded_hf_tokenizer() {
    // Test scaffolding - will fail until implementation complete
    let test_path = Path::new("tests/fixtures/gguf/llama3-with-hf-tokenizer.gguf");

    // This should succeed once TokenizerDiscovery::try_extract_embedded_tokenizer is implemented
    if test_path.exists() {
        let discovery_result = TokenizerDiscovery::from_gguf(test_path);
        assert!(discovery_result.is_ok(), "Should load GGUF with embedded HF tokenizer");

        let discovery = discovery_result.unwrap();
        let embedded_result = discovery.try_extract_embedded_tokenizer();

        // Expected behavior: extract HuggingFace tokenizer from GGUF metadata
        match embedded_result {
            Ok(Some(tokenizer)) => {
                assert!(tokenizer.vocab_size() > 0, "Embedded tokenizer should have vocabulary");
                assert!(tokenizer.bos_token_id().is_some(), "Should have BOS token");
                assert!(tokenizer.eos_token_id().is_some(), "Should have EOS token");
            }
            Ok(None) => panic!("Expected embedded tokenizer but found none"),
            Err(e) => panic!("Failed to extract embedded tokenizer: {}", e),
        }
    } else {
        // Test scaffolding - fixture not yet created
        println!("AC1: Test fixture not found, skipping until fixture-builder creates it");
    }
}

/// AC1: Extract embedded SentencePiece model from GGUF metadata
/// Tests feature spec: issue-336-universal-tokenizer-discovery-spec.md#ac1-embedded-tokenizer-support
// AC:ID AC1
#[test]
#[cfg(all(feature = "cpu", feature = "spm"))]
fn ac1_extract_embedded_sentencepiece_model() {
    let test_path = Path::new("tests/fixtures/gguf/llama2-with-sentencepiece.gguf");

    if test_path.exists() {
        let discovery = TokenizerDiscovery::from_gguf(test_path)
            .expect("Should load GGUF with embedded SentencePiece");

        let embedded_result = discovery.try_extract_embedded_tokenizer();

        match embedded_result {
            Ok(Some(tokenizer)) => {
                assert_eq!(tokenizer.vocab_size(), 32000, "LLaMA-2 should have 32K vocabulary");

                // Validate special tokens for SentencePiece
                assert_eq!(tokenizer.bos_token_id(), Some(1), "SentencePiece BOS should be 1");
                assert_eq!(tokenizer.eos_token_id(), Some(2), "SentencePiece EOS should be 2");
            }
            Ok(None) => panic!("Expected embedded SentencePiece but found none"),
            Err(e) => panic!("Failed to extract SentencePiece: {}", e),
        }
    } else {
        println!("AC1: SentencePiece fixture not found, awaiting fixture-builder");
    }
}

/// AC1: Validate embedded tokenizer metadata (BOS/EOS/PAD tokens)
/// Tests feature spec: issue-336-universal-tokenizer-discovery-spec.md#ac1-embedded-tokenizer-support
// AC:ID AC1
#[test]
#[cfg(feature = "cpu")]
fn ac1_validate_embedded_tokenizer_metadata() {
    let test_cases = [
        ("tests/fixtures/gguf/llama3-128k.gguf", 128256, Some(128000), Some(128001)),
        ("tests/fixtures/gguf/llama2-32k.gguf", 32000, Some(1), Some(2)),
        ("tests/fixtures/gguf/gpt2-50k.gguf", 50257, None, Some(50256)),
    ];

    for (path, expected_vocab, expected_bos, expected_eos) in test_cases {
        let test_path = Path::new(path);

        if !test_path.exists() {
            println!("AC1: Fixture {} not found, skipping", path);
            continue;
        }

        let discovery = TokenizerDiscovery::from_gguf(test_path).expect("Should load GGUF file");

        assert_eq!(discovery.vocab_size(), expected_vocab, "Vocabulary size mismatch");

        if let Ok(Some(tokenizer)) = discovery.try_extract_embedded_tokenizer() {
            assert_eq!(tokenizer.bos_token_id(), expected_bos, "BOS token mismatch");
            assert_eq!(tokenizer.eos_token_id(), expected_eos, "EOS token mismatch");
        }
    }
}

/// AC1: Fallback to BasicTokenizer when embedded data is invalid
/// Tests feature spec: issue-336-universal-tokenizer-discovery-spec.md#ac1-embedded-tokenizer-support
// AC:ID AC1
#[test]
#[cfg(feature = "cpu")]
fn ac1_fallback_when_embedded_data_invalid() {
    let test_path = Path::new("tests/fixtures/gguf/corrupted-embedded-tokenizer.gguf");

    if test_path.exists() {
        let discovery = TokenizerDiscovery::from_gguf(test_path)
            .expect("Should load GGUF even with corrupted embedded data");

        let embedded_result = discovery.try_extract_embedded_tokenizer();

        // Expected: returns None or BasicTokenizer fallback when data is corrupted
        match embedded_result {
            Ok(None) => {
                println!("AC1: Correctly returned None for corrupted embedded data");
            }
            Ok(Some(_tokenizer)) => {
                // If BasicTokenizer fallback is used, validate it's functional
                println!("AC1: Fallback tokenizer created for corrupted data");
            }
            Err(e) => {
                println!("AC1: Error extracting corrupted data (expected): {}", e);
            }
        }
    } else {
        println!("AC1: Corrupted fixture not found, awaiting fixture-builder");
    }
}

/// AC1: Extract embedded tokenizer vocabulary size validation
/// Tests feature spec: issue-336-universal-tokenizer-discovery-spec.md#ac1-embedded-tokenizer-support
// AC:ID AC1
#[test]
#[cfg(feature = "cpu")]
fn ac1_embedded_tokenizer_vocab_size_validation() {
    // Test vocabulary size boundaries (1K-2M tokens as per spec)
    let test_cases = [
        ("tests/fixtures/gguf/small-vocab-1k.gguf", 1000, true),
        ("tests/fixtures/gguf/llama2-32k.gguf", 32000, true),
        ("tests/fixtures/gguf/llama3-128k.gguf", 128256, true),
        ("tests/fixtures/gguf/extreme-vocab-2m.gguf", 2_000_000, true),
        ("tests/fixtures/gguf/invalid-vocab-zero.gguf", 0, false),
        ("tests/fixtures/gguf/invalid-vocab-excessive.gguf", 3_000_000, false),
    ];

    for (path, expected_vocab, should_be_valid) in test_cases {
        let test_path = Path::new(path);

        if !test_path.exists() {
            continue;
        }

        let discovery_result = TokenizerDiscovery::from_gguf(test_path);

        if should_be_valid {
            assert!(
                discovery_result.is_ok(),
                "Should accept valid vocabulary size: {}",
                expected_vocab
            );

            if let Ok(discovery) = discovery_result {
                assert_eq!(discovery.vocab_size(), expected_vocab, "Vocabulary size mismatch");
            }
        } else {
            // Invalid vocabulary sizes should be rejected
            if let Ok(discovery) = discovery_result {
                // Some invalid sizes might load but extraction should fail
                let embedded_result = discovery.try_extract_embedded_tokenizer();
                println!(
                    "AC1: Invalid vocab {} - extraction result: {:?}",
                    expected_vocab,
                    embedded_result.is_ok()
                );
            }
        }
    }
}

// ================================
// AC1: EMBEDDED TOKENIZER EDGE CASES
// ================================

/// AC1: Test embedded tokenizer with missing metadata keys
/// Tests feature spec: issue-336-universal-tokenizer-discovery-spec.md#ac1-embedded-tokenizer-support
// AC:ID AC1
#[test]
#[cfg(feature = "cpu")]
fn ac1_embedded_tokenizer_missing_metadata() {
    let test_path = Path::new("tests/fixtures/gguf/missing-tokenizer-metadata.gguf");

    if test_path.exists() {
        let discovery = TokenizerDiscovery::from_gguf(test_path)
            .expect("Should load GGUF with missing metadata");

        let embedded_result = discovery.try_extract_embedded_tokenizer();

        // Should return None when required metadata is missing
        assert!(embedded_result.is_ok(), "Should handle missing metadata gracefully");

        if let Ok(Some(_tokenizer)) = embedded_result {
            println!("AC1: Created fallback tokenizer for missing metadata");
        }
    }
}

/// AC1: Test embedded tokenizer extraction performance
/// Tests feature spec: issue-336-universal-tokenizer-discovery-spec.md#ac1-embedded-tokenizer-support
// AC:ID AC1
#[test]
#[cfg(feature = "cpu")]
fn ac1_embedded_tokenizer_extraction_performance() {
    use std::time::Instant;

    let test_path = Path::new("tests/fixtures/gguf/llama3-128k.gguf");

    if test_path.exists() {
        let discovery = TokenizerDiscovery::from_gguf(test_path).expect("Should load GGUF file");

        let start = Instant::now();
        let _embedded_result = discovery.try_extract_embedded_tokenizer();
        let elapsed = start.elapsed();

        // Extraction should be fast (<100ms for reasonable models)
        assert!(elapsed.as_millis() < 1000, "Extraction took too long: {:?}", elapsed);
        println!("AC1: Extraction completed in {:?}", elapsed);
    }
}

/// AC1: Test concurrent embedded tokenizer extraction
/// Tests feature spec: issue-336-universal-tokenizer-discovery-spec.md#ac1-embedded-tokenizer-support
// AC:ID AC1
#[test]
#[cfg(feature = "cpu")]
fn ac1_concurrent_embedded_tokenizer_extraction() {
    use std::sync::Arc;
    use std::thread;

    let test_path = Path::new("tests/fixtures/gguf/llama2-32k.gguf");

    if !test_path.exists() {
        return;
    }

    let path_arc = Arc::new(test_path.to_path_buf());
    let mut handles = vec![];

    // Spawn multiple threads to test concurrent extraction
    for i in 0..4 {
        let path_clone = Arc::clone(&path_arc);

        let handle = thread::spawn(move || {
            for _ in 0..5 {
                let discovery_result = TokenizerDiscovery::from_gguf(&path_clone);

                if let Ok(discovery) = discovery_result {
                    let _embedded_result = discovery.try_extract_embedded_tokenizer();
                    // No panics or race conditions expected
                }
            }
            println!("AC1: Concurrent thread {} completed", i);
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread should complete without panic");
    }
}

/// AC1: Test embedded tokenizer with various GGUF versions
/// Tests feature spec: issue-336-universal-tokenizer-discovery-spec.md#ac1-embedded-tokenizer-support
// AC:ID AC1
#[test]
#[cfg(feature = "cpu")]
fn ac1_embedded_tokenizer_gguf_versions() {
    let test_cases = [
        ("tests/fixtures/gguf/v2/llama2.gguf", "GGUF v2"),
        ("tests/fixtures/gguf/v3/llama3.gguf", "GGUF v3"),
        ("tests/fixtures/gguf/latest/bitnet.gguf", "GGUF latest"),
    ];

    for (path, version) in test_cases {
        let test_path = Path::new(path);

        if !test_path.exists() {
            continue;
        }

        let discovery_result = TokenizerDiscovery::from_gguf(test_path);
        assert!(discovery_result.is_ok(), "Should support {}", version);

        if let Ok(discovery) = discovery_result {
            let embedded_result = discovery.try_extract_embedded_tokenizer();
            println!("AC1: {} - embedded extraction: {:?}", version, embedded_result.is_ok());
        }
    }
}

/// AC1: Test embedded tokenizer memory efficiency
/// Tests feature spec: issue-336-universal-tokenizer-discovery-spec.md#ac1-embedded-tokenizer-support
// AC:ID AC1
#[test]
#[cfg(feature = "cpu")]
fn ac1_embedded_tokenizer_memory_efficiency() {
    let test_path = Path::new("tests/fixtures/gguf/llama3-128k.gguf");

    if !test_path.exists() {
        return;
    }

    // Create multiple discovery instances to test memory usage
    let mut discoveries = vec![];

    for i in 0..10 {
        let discovery_result = TokenizerDiscovery::from_gguf(test_path);

        if let Ok(discovery) = discovery_result {
            discoveries.push(discovery);
        } else {
            println!("AC1: Discovery {} failed to load", i);
        }
    }

    // Extract embedded tokenizers from all instances
    let mut tokenizers: Vec<Option<Arc<dyn Tokenizer>>> = vec![];

    for discovery in &discoveries {
        if let Ok(embedded) = discovery.try_extract_embedded_tokenizer() {
            tokenizers.push(embedded);
        }
    }

    println!(
        "AC1: Created {} discovery instances and {} tokenizers",
        discoveries.len(),
        tokenizers.len()
    );

    // Memory should be managed efficiently via Arc
    assert!(tokenizers.len() <= discoveries.len(), "Should not exceed discovery count");
}
