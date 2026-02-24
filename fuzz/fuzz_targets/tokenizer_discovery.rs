#![no_main]

use bitnet_models::formats::gguf::GgufReader;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Skip very small inputs that can't be valid GGUF files
    if data.len() < 16 {
        return;
    }

    // Try to parse as GGUF and extract tokenizer metadata
    if let Ok(reader) = GgufReader::new(data) {
        // Test vocabulary size extraction strategies
        // This should not panic even with malformed metadata

        // Strategy 1: Standard GGUF vocabulary size key
        let _ = reader.get_u32_metadata("tokenizer.ggml.vocab_size");

        // Strategy 2: Architecture-specific metadata
        if let Some(arch) = reader.get_string_metadata("general.architecture") {
            let arch_key = format!("{}.vocab_size", arch);
            let _ = reader.get_u32_metadata(&arch_key);
        }

        // Strategy 3: Alternative metadata keys
        let alt_keys = [
            "llama.vocab_size",
            "gpt2.vocab_size",
            "transformer.vocab_size",
            "model.vocab_size",
            "vocab_size",
        ];

        for key in &alt_keys {
            let _ = reader.get_u32_metadata(key);
        }

        // Strategy 4: Embedded tokenizer extraction
        let _ = reader.get_string_metadata("tokenizer.json");
        let _ = reader.get_string_array_metadata("tokenizer.ggml.tokens");
        let _ = reader.get_array_metadata("tokenizer.ggml.model");

        // Test special token extraction
        let _ = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
        let _ = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
        let _ = reader.get_u32_metadata("tokenizer.ggml.pad_token_id");
        let _ = reader.get_u32_metadata("tokenizer.ggml.unknown_token_id");

        // Test tensor-based vocabulary inference
        let tensor_names = reader.tensor_names();
        for name in tensor_names.iter().take(20) {
            // Limit iterations
            if name.contains("token_embd") || name.contains("wte") || name.contains("embed") {
                let _ = reader.get_tensor_info_by_name(name);
            }
        }

        // Test architecture detection
        let _ = reader.get_string_metadata("general.architecture");
        let _ = reader.get_string_metadata("general.name");
        let _ = reader.get_string_metadata("model.architecture");
        let _ = reader.get_string_metadata("transformer.architecture");
    }
});

// Fuzz target for tokenizer discovery GGUF metadata parsing
// Tests Issue #336 - Universal Tokenizer Discovery System
