//! # HuggingFace Tokenizers Integration Example
//!
//! This example demonstrates how to integrate BitNet.rs with the HuggingFace tokenizers
//! crate for seamless tokenization and model compatibility.

use anyhow::Result;
use tokenizers::{Tokenizer, EncodeInput, Encoding};
use bitnet_tokenizers::{TokenizerBuilder, Tokenizer as BitNetTokenizer};
use bitnet_inference::InferenceEngine;
use std::sync::Arc;
use std::path::Path;

/// Demonstrates basic tokenizer integration
fn basic_tokenizer_example() -> Result<()> {
    println!("=== HuggingFace Tokenizers Integration Example ===\n");

    // Load a HuggingFace tokenizer
    let hf_tokenizer = load_huggingface_tokenizer("gpt2")?;
    println!("Loaded HuggingFace GPT-2 tokenizer");

    // Create BitNet tokenizer wrapper
    let bitnet_tokenizer = create_bitnet_tokenizer_wrapper(hf_tokenizer)?;
    println!("Created BitNet tokenizer wrapper");

    // Test tokenization
    let text = "Hello, world! This is a test of the tokenization system.";
    println!("Input text: \"{}\"", text);

    let tokens = bitnet_tokenizer.encode(text, true)?;
    println!("Encoded tokens: {:?}", tokens);
    println!("Number of tokens: {}", tokens.len());

    let decoded = bitnet_tokenizer.decode(&tokens, true)?;
    println!("Decoded text: \"{}\"", decoded);

    // Verify round-trip accuracy
    assert_eq!(text.trim(), decoded.trim());
    println!("✓ Round-trip tokenization successful");

    Ok(())
}

/// Load HuggingFace tokenizer from pretrained model
fn load_huggingface_tokenizer(model_name: &str) -> Result<Tokenizer> {
    // In a real implementation, this would download from HuggingFace Hub
    // For this example, we'll create a simple tokenizer

    println!("Loading tokenizer for model: {}", model_name);

    // Create a basic tokenizer (in practice, load from HuggingFace)
    let tokenizer = Tokenizer::from_pretrained(model_name, None)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    Ok(tokenizer)
}

/// Create BitNet tokenizer wrapper around HuggingFace tokenizer
fn create_bitnet_tokenizer_wrapper(hf_tokenizer: Tokenizer) -> Result<Arc<dyn BitNetTokenizer>> {
    Ok(Arc::new(HuggingFaceTokenizerWrapper::new(hf_tokenizer)))
}

/// Wrapper to make HuggingFace tokenizer compatible with BitNet interface
struct HuggingFaceTokenizerWrapper {
    tokenizer: Tokenizer,
}

impl HuggingFaceTokenizerWrapper {
    fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }
}

impl BitNetTokenizer for HuggingFaceTokenizerWrapper {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        // For backwards compatibility, use add_special for the HF tokenizer
        let encoding = self.tokenizer.encode(text, add_special)
            .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;

        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        // Always skip special tokens for consistency
        self.tokenizer.decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.tokenizer.token_to_id("<|endoftext|>")
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.tokenizer.token_to_id("<|pad|>")
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.tokenizer.id_to_token(token)
    }
}

/// Demonstrates advanced tokenization features
fn advanced_tokenization_example() -> Result<()> {
    println!("\n=== Advanced Tokenization Features ===\n");

    let hf_tokenizer = load_huggingface_tokenizer("gpt2")?;
    let bitnet_tokenizer = create_bitnet_tokenizer_wrapper(hf_tokenizer)?;

    // Test batch encoding
    let texts = vec![
        "First example text for batch processing.",
        "Second example with different length and content.",
        "Third example that's even longer and contains more complex tokenization scenarios.",
    ];

    println!("Batch encoding {} texts:", texts.len());
    for (i, text) in texts.iter().enumerate() {
        let tokens = bitnet_tokenizer.encode(text, true)?;
        println!("Text {}: {} tokens", i + 1, tokens.len());
        println!("  Tokens: {:?}", &tokens[..std::cmp::min(10, tokens.len())]);
    }

    // Test special tokens
    println!("\nSpecial tokens:");
    println!("Vocab size: {}", bitnet_tokenizer.vocab_size());
    if let Some(eos_id) = bitnet_tokenizer.eos_token_id() {
        println!("EOS token ID: {}", eos_id);
    }
    if let Some(pad_id) = bitnet_tokenizer.pad_token_id() {
        println!("PAD token ID: {}", pad_id);
    }

    // Test truncation and padding
    let long_text = "This is a very long text that will be used to test truncation and padding functionality of the tokenizer integration.".repeat(10);
    let tokens = bitnet_tokenizer.encode(&long_text, true)?;
    println!("\nLong text tokenization:");
    println!("Original length: {} characters", long_text.len());
    println!("Tokenized length: {} tokens", tokens.len());

    Ok(())
}

/// Demonstrates tokenizer integration with inference pipeline
fn inference_pipeline_example() -> Result<()> {
    println!("\n=== Tokenizer + Inference Pipeline ===\n");

    // Load tokenizer
    let hf_tokenizer = load_huggingface_tokenizer("gpt2")?;
    let tokenizer = create_bitnet_tokenizer_wrapper(hf_tokenizer)?;

    // Simulate loading a BitNet model (in practice, load from file)
    println!("Loading BitNet model...");
    let model = create_mock_bitnet_model()?;

    // Create inference engine
    let mut engine = InferenceEngine::new(
        model,
        tokenizer.clone(),
        bitnet_common::Device::Cpu,
    )?;

    // Test inference with tokenization
    let prompt = "The future of artificial intelligence is";
    println!("Prompt: \"{}\"", prompt);

    // Tokenize input
    let input_tokens = tokenizer.encode(prompt, true)?;
    println!("Input tokens: {:?}", input_tokens);

    // Run inference (simulated)
    let generated_text = engine.generate(prompt)?;
    println!("Generated text: \"{}\"", generated_text);

    // Demonstrate streaming with tokenization
    println!("\nStreaming generation:");
    let mut stream = engine.generate_stream(prompt);
    let mut full_response = String::new();

    // Simulate streaming (in practice, this would be async)
    for i in 0..5 {
        let token_text = format!(" token_{}", i);
        full_response.push_str(&token_text);
        println!("Stream chunk {}: \"{}\"", i + 1, token_text);
    }

    println!("Complete streamed response: \"{}{}\"", prompt, full_response);

    Ok(())
}

/// Create a mock BitNet model for demonstration
fn create_mock_bitnet_model() -> Result<Box<dyn bitnet_models::Model>> {
    // This would normally load a real model from disk
    Ok(Box::new(MockBitNetModel::new()))
}

/// Mock BitNet model for demonstration purposes
struct MockBitNetModel {
    config: bitnet_common::BitNetConfig,
}

impl MockBitNetModel {
    fn new() -> Self {
        Self {
            config: bitnet_common::BitNetConfig::default(),
        }
    }
}

impl bitnet_models::Model for MockBitNetModel {
    type Config = bitnet_common::BitNetConfig;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn forward(
        &self,
        input: &dyn bitnet_common::Tensor,
        cache: &mut bitnet_inference::KVCache,
    ) -> Result<Box<dyn bitnet_common::Tensor>> {
        // Mock forward pass - return random logits
        let batch_size = input.shape()[0];
        let vocab_size = 50257; // GPT-2 vocab size

        let logits = vec![0.1f32; batch_size * vocab_size];

        Ok(Box::new(MockTensor {
            data: logits,
            shape: vec![batch_size, vocab_size],
        }))
    }

    fn generate(&self, tokens: &[u32], config: &bitnet_inference::GenerationConfig) -> Result<Vec<u32>> {
        // Mock generation - return some example tokens
        Ok(vec![1234, 5678, 9012])
    }
}

/// Mock tensor for demonstration
struct MockTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl bitnet_common::Tensor for MockTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> bitnet_common::DType {
        bitnet_common::DType::F32
    }

    fn device(&self) -> &bitnet_common::Device {
        &bitnet_common::Device::Cpu
    }

    fn as_slice<T>(&self) -> Result<&[T]> {
        unsafe {
            let ptr = self.data.as_ptr() as *const T;
            let slice = std::slice::from_raw_parts(ptr, self.data.len());
            Ok(slice)
        }
    }
}

/// Demonstrates custom tokenizer creation
fn custom_tokenizer_example() -> Result<()> {
    println!("\n=== Custom Tokenizer Creation ===\n");

    // Create a custom tokenizer using HuggingFace tokenizers
    let mut tokenizer = Tokenizer::new(
        tokenizers::models::bpe::BPE::default()
    );

    // Add a simple pre-tokenizer
    tokenizer.with_pre_tokenizer(
        tokenizers::pre_tokenizers::whitespace::Whitespace::default()
    );

    // Add a decoder
    tokenizer.with_decoder(
        tokenizers::decoders::bpe::BPEDecoder::default()
    );

    println!("Created custom tokenizer");

    // Wrap in BitNet interface
    let bitnet_tokenizer = Arc::new(HuggingFaceTokenizerWrapper::new(tokenizer));

    // Test the custom tokenizer
    let test_text = "Custom tokenizer test";
    match bitnet_tokenizer.encode(test_text, false) {
        Ok(tokens) => {
            println!("Custom tokenizer encoded: {:?}", tokens);
            if let Ok(decoded) = bitnet_tokenizer.decode(&tokens, false) {
                println!("Custom tokenizer decoded: \"{}\"", decoded);
            }
        }
        Err(e) => {
            println!("Custom tokenizer encoding failed (expected for demo): {}", e);
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    // Run all examples
    basic_tokenizer_example()?;
    advanced_tokenization_example()?;
    inference_pipeline_example()?;
    custom_tokenizer_example()?;

    println!("\n=== HuggingFace Tokenizers Integration Examples Complete ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_wrapper() -> Result<()> {
        // This test would require a real tokenizer file
        // For now, we'll test the wrapper structure

        // Create a mock tokenizer for testing
        let tokenizer = Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let wrapper = HuggingFaceTokenizerWrapper::new(tokenizer);

        // Test basic interface
        assert!(wrapper.vocab_size() > 0);

        Ok(())
    }

    #[test]
    fn test_mock_model() {
        let model = MockBitNetModel::new();
        assert_eq!(model.config().model.vocab_size, 50257);
    }
}
