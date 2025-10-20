//! Test for from_gguf_model_with_preference implementation
//!
//! This test validates that the from_gguf_model_with_preference function
//! correctly loads a tokenizer from BitNetModel metadata with backend preference.

#[cfg(test)]
mod tests {
    use bitnet_common::{BitNetConfig, Device};
    use bitnet_models::BitNetModel;
    use bitnet_tokenizers::{Tokenizer, TokenizerBackend, UniversalTokenizer};

    #[test]
    fn test_from_gguf_model_with_mock_preference() {
        // Create a minimal BitNet model configuration
        let mut config = BitNetConfig::default();
        config.model.vocab_size = 50257; // GPT-2 vocab size
        config.model.tokenizer.bos_id = Some(1);
        config.model.tokenizer.eos_id = Some(2);
        config.model.tokenizer.pad_id = Some(0);
        config.model.tokenizer.unk_id = Some(3);
        config.inference.add_bos = true;
        config.inference.append_eos = false;

        // Create BitNet model
        let model = BitNetModel::new(config, Device::Cpu);

        // Create tokenizer from model with Mock backend preference
        let tokenizer =
            UniversalTokenizer::from_gguf_model_with_preference(&model, TokenizerBackend::Mock)
                .expect("Should create tokenizer from model metadata");

        // Validate tokenizer properties
        assert_eq!(tokenizer.vocab_size(), 50257, "Vocab size should match model");
        assert_eq!(tokenizer.backend_type(), TokenizerBackend::Mock, "Backend should be Mock");

        // Test basic tokenization
        let text = "Hello, world!";
        let tokens = tokenizer.encode(text, true, false).expect("Should encode text successfully");

        // Should produce some tokens (mock tokenizer splits on whitespace)
        assert!(!tokens.is_empty(), "Should produce tokens");

        println!("✅ Successfully created tokenizer from BitNetModel with Mock preference");
        println!("  Vocab size: {}", tokenizer.vocab_size());
        println!("  Backend type: {:?}", tokenizer.backend_type());
        println!("  Encoded '{}' to {} tokens", text, tokens.len());
    }

    #[test]
    fn test_backend_type_detection() {
        // Create model with specific configuration
        let mut config = BitNetConfig::default();
        config.model.vocab_size = 32000;
        config.model.tokenizer.bos_id = Some(100);
        config.model.tokenizer.eos_id = Some(200);

        let model = BitNetModel::new(config, Device::Cpu);

        // Create tokenizer with Mock backend
        let tokenizer =
            UniversalTokenizer::from_gguf_model_with_preference(&model, TokenizerBackend::Mock)
                .expect("Should create tokenizer");

        // Verify backend type
        assert_eq!(tokenizer.backend_type(), TokenizerBackend::Mock, "Should detect Mock backend");

        println!("✅ Backend type detection works correctly");
    }

    #[test]
    fn test_encode_batch() {
        // Create a simple model
        let config = BitNetConfig::default();
        let model = BitNetModel::new(config, Device::Cpu);

        // Create tokenizer
        let tokenizer =
            UniversalTokenizer::from_gguf_model_with_preference(&model, TokenizerBackend::Mock)
                .expect("Should create tokenizer");

        // Test batch encoding
        let texts =
            vec!["Hello world".to_string(), "Test text".to_string(), "Another example".to_string()];

        let batch_results =
            tokenizer.encode_batch(&texts).expect("Should encode batch successfully");

        assert_eq!(batch_results.len(), 3, "Should have 3 results for 3 inputs");
        assert!(
            batch_results.iter().all(|tokens| !tokens.is_empty()),
            "All results should have tokens"
        );

        println!("✅ Batch encoding works correctly");
        println!("  Batch size: {}", batch_results.len());
        println!("  Token counts: {:?}", batch_results.iter().map(|t| t.len()).collect::<Vec<_>>());
    }

    #[test]
    fn test_config_extraction() {
        // Test that configuration is properly extracted from BitNetModel
        let mut config = BitNetConfig::default();
        config.model.vocab_size = 128256; // LLaMA-3 style
        config.model.tokenizer.bos_id = Some(128000);
        config.model.tokenizer.eos_id = Some(128001);
        config.inference.add_bos = true;
        config.inference.append_eos = true;

        let model = BitNetModel::new(config.clone(), Device::Cpu);

        let tokenizer =
            UniversalTokenizer::from_gguf_model_with_preference(&model, TokenizerBackend::Mock)
                .expect("Should create tokenizer from model config");

        // Verify vocab size is correctly extracted
        assert_eq!(
            tokenizer.vocab_size(),
            config.model.vocab_size,
            "Vocab size should match model configuration"
        );

        println!("✅ Configuration extraction works correctly");
        println!("  Vocab size: {}", tokenizer.vocab_size());
        println!("  BOS ID: {:?}", config.model.tokenizer.bos_id);
        println!("  EOS ID: {:?}", config.model.tokenizer.eos_id);
    }
}
