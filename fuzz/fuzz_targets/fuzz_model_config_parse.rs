#![no_main]

use arbitrary::Arbitrary;
use bitnet_models::config::GgufModelConfig;
use bitnet_models::config_detection::detect_from_hf_config;
use bitnet_models::formats::gguf::GgufValue;
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

#[derive(Arbitrary, Debug)]
struct ConfigParseInput {
    /// Key-value pairs to inject into a GGUF metadata map.
    entries: Vec<MetadataEntry>,
    /// Raw JSON string for HF config parsing.
    raw_json: Vec<u8>,
}

#[derive(Arbitrary, Debug)]
struct MetadataEntry {
    key_idx: u8,
    value: FuzzGgufValue,
}

#[derive(Arbitrary, Debug)]
enum FuzzGgufValue {
    U32(u32),
    I32(i32),
    F32Bits(u32),
    Str(Vec<u8>),
    Bool(bool),
}

const GGUF_KEYS: &[&str] = &[
    "general.architecture",
    "general.name",
    "llama.embedding_length",
    "llama.block_count",
    "llama.attention.head_count",
    "llama.attention.head_count_kv",
    "llama.feed_forward_length",
    "llama.vocab_size",
    "llama.context_length",
    "llama.rope.freq_base",
    "llama.attention.layer_norm_rms_epsilon",
    "llama.rope.scaling.type",
    "general.file_type",
    "tokenizer.ggml.model",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.bos_token_id",
];

fuzz_target!(|input: ConfigParseInput| {
    // Build a GGUF metadata map from fuzzed entries
    let mut metadata: HashMap<String, GgufValue> = HashMap::new();

    for entry in &input.entries {
        let key = GGUF_KEYS[(entry.key_idx as usize) % GGUF_KEYS.len()].to_string();
        let value = match &entry.value {
            FuzzGgufValue::U32(v) => GgufValue::U32(*v),
            FuzzGgufValue::I32(v) => GgufValue::I32(*v),
            FuzzGgufValue::F32Bits(bits) => GgufValue::F32(f32::from_bits(*bits)),
            FuzzGgufValue::Str(bytes) => {
                GgufValue::String(String::from_utf8_lossy(bytes).into_owned())
            }
            FuzzGgufValue::Bool(b) => GgufValue::Bool(*b),
        };
        metadata.insert(key, value);
    }

    // from_gguf_metadata must not panic
    let config_result = GgufModelConfig::from_gguf_metadata(&metadata);
    if let Ok(config) = config_result {
        // validate must not panic
        let _ = config.validate();
        // Debug must not panic
        let _ = format!("{config:?}");
    }

    // Test HF config.json parsing with arbitrary bytes
    if let Ok(json_str) = std::str::from_utf8(&input.raw_json) {
        // detect_from_hf_config must not panic on any valid UTF-8 string
        let _ = detect_from_hf_config(json_str);
    }
});
