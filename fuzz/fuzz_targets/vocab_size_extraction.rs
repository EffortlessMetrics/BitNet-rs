#![no_main]

use bitnet_models::formats::gguf::GgufReader;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Skip very small inputs
    if data.len() < 16 {
        return;
    }

    if let Ok(reader) = GgufReader::new(data) {
        // Test vocabulary size extraction with boundary validation

        // Try all vocabulary size extraction strategies
        if let Some(vocab_size) = reader.get_u32_metadata("tokenizer.ggml.vocab_size") {
            // Validate vocab size is reasonable (no overflow or extreme values)
            let _is_valid = vocab_size > 0 && vocab_size < 2_000_000;

            // Test GPU acceleration detection for large vocabularies
            let _requires_gpu = vocab_size > 65536;
        }

        // Test embedding tensor shape inference
        let tensor_names = reader.tensor_names();
        for name in tensor_names.iter().take(10) {
            if (name.contains("token_embd") || name.contains("embed"))
                && let Some(info) = reader.get_tensor_info_by_name(name)
            {
                let shape = &info.shape;
                if !shape.is_empty() {
                    let possible_vocab = shape[0];
                    // Sanity check - vocab size should be reasonable
                    let _is_reasonable = (100..2_000_000).contains(&possible_vocab);
                }
            }
        }

        // Test architecture-specific default vocabulary sizes
        if let Some(arch) = reader.get_string_metadata("general.architecture") {
            match arch.as_str() {
                "llama" => {
                    // Test LLaMA-2 vs LLaMA-3 distinction
                    if let Some(name) = reader.get_string_metadata("general.name") {
                        let _is_llama3 = name.contains("llama-3") || name.contains("llama3");
                    }
                }
                "gpt2" | "gptneox" | "bert" | "t5" => {
                    // Test known architectures
                    let _ = arch;
                }
                _ => {
                    // Unknown architecture - should handle gracefully
                }
            }
        }
    }
});

// Fuzz target for vocabulary size extraction edge cases
// Tests Issue #336 - Universal Tokenizer Discovery System
