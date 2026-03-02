#![no_main]

use std::collections::HashMap;

use arbitrary::Arbitrary;
use bitnet_models::config::GgufModelConfig;
use bitnet_models::formats::gguf::GgufValue;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ConfigInput {
    num_layers: u16,
    hidden_dim: u16,
    num_heads: u16,
    vocab_size: u32,
    max_seq_len: u16,
    num_kv_heads: u16,
    intermediate_size: u16,
}

fuzz_target!(|input: ConfigInput| {
    let mut metadata: HashMap<String, GgufValue> = HashMap::new();

    // Populate with standard GGUF metadata keys.
    metadata.insert("general.architecture".into(), GgufValue::String("llama".into()));
    metadata.insert("llama.block_count".into(), GgufValue::U32(input.num_layers as u32));
    metadata.insert("llama.embedding_length".into(), GgufValue::U32(input.hidden_dim as u32));
    metadata.insert("llama.attention.head_count".into(), GgufValue::U32(input.num_heads as u32));
    metadata
        .insert("llama.attention.head_count_kv".into(), GgufValue::U32(input.num_kv_heads as u32));
    metadata.insert("general.name".into(), GgufValue::String("fuzz-model".into()));
    metadata.insert("llama.vocab_size".into(), GgufValue::U32(input.vocab_size));
    metadata.insert("llama.context_length".into(), GgufValue::U32(input.max_seq_len as u32));
    metadata
        .insert("llama.feed_forward_length".into(), GgufValue::U32(input.intermediate_size as u32));

    // Parsing must not panic on any combination.
    let config = match GgufModelConfig::from_gguf_metadata(&metadata) {
        Ok(c) => c,
        Err(_) => return,
    };

    // Validation must not panic; it should return Ok or Err.
    let valid = config.validate();

    // If validation passes, check basic invariants.
    if valid.is_ok() {
        assert!(config.vocab_size > 0, "valid config has zero vocab_size");
        assert!(config.num_layers > 0, "valid config has zero num_layers");
        assert!(config.hidden_size > 0, "valid config has zero hidden_size");
        assert!(config.num_heads > 0, "valid config has zero num_heads");

        // GQA helpers must not panic on valid configs.
        let _ = config.is_gqa();
        let _ = config.gqa_group_size();
        let _ = config.memory_estimate();
    }

    // Invalid combinations: zero dimensions should be rejected.
    if input.num_heads == 0 || input.hidden_dim == 0 || input.num_layers == 0 {
        // Parser or validator should have caught this.
        // We don't assert Err here because the parser may normalize zeros.
    }
});
