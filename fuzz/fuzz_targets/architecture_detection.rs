#![no_main]

use bitnet_models::formats::gguf::GgufReader;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Skip very small inputs
    if data.len() < 16 {
        return;
    }

    if let Ok(reader) = GgufReader::new(data) {
        // Test architecture detection from metadata
        let _ = reader.get_string_metadata("general.architecture");
        let _ = reader.get_string_metadata("model.architecture");
        let _ = reader.get_string_metadata("transformer.architecture");

        // Test architecture detection from model name
        if let Some(name) = reader.get_string_metadata("general.name") {
            let name_lower = name.to_lowercase();

            // Test all architecture patterns
            let patterns = [
                ("bitnet", vec!["bitnet", "bitlinear"]),
                ("llama", vec!["llama"]),
                ("gpt2", vec!["gpt2", "gpt-2"]),
                ("gptneox", vec!["gpt-neo", "gptneox", "gpt-j"]),
                ("bert", vec!["bert"]),
                ("t5", vec!["t5"]),
            ];

            for (_arch, pattern_list) in patterns {
                for pattern in pattern_list {
                    let _matches = name_lower.contains(pattern);
                }
            }
        }

        // Test architecture detection from tensor patterns
        let tensor_names = reader.tensor_names();

        // BitNet patterns
        let _has_bitnet =
            tensor_names.iter().any(|name| name.contains("bitlinear") || name.contains("bitnet"));

        // LLaMA patterns
        let _has_llama = tensor_names.iter().any(|name| {
            name.contains("attn_q")
                || name.contains("attn_k")
                || name.contains("attention.wq")
                || name.contains("attention.wk")
        });

        // GPT-2 patterns (compound)
        let _has_gpt2 = tensor_names.iter().any(|name| {
            (name.contains("mlp") || name.contains("c_fc"))
                && (name.contains("attn") || name.contains("c_attn"))
        });

        // T5 patterns
        let _has_t5 = tensor_names.iter().all(|name| {
            name.contains("encoder")
                || name.contains("decoder")
                || name.contains("relative_attention_bias")
        });

        // BERT patterns
        let _has_bert = tensor_names.iter().all(|name| {
            name.contains("encoder") && name.contains("self") && name.contains("attention")
        });

        // GPT-Neo patterns
        let _has_gptneox =
            tensor_names.iter().any(|name| name.contains("gpt_neox") || name.contains("gptneox"));
    }
});

// Fuzz target for architecture detection from tensor patterns
// Tests Issue #336 - Universal Tokenizer Discovery System
