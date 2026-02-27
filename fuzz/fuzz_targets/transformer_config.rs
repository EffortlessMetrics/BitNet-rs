#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::BitNetConfig;
use bitnet_models::GgufReader;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct TransformerConfigInput {
    /// Raw bytes treated as a potential GGUF stream.
    data: Vec<u8>,
    /// Config field values used to build a synthetic GGUF-like payload.
    hidden_size: u32,
    n_heads: u8,
    n_kv_heads: u8,
    n_layers: u8,
    vocab_size: u32,
    context_len: u32,
}

fuzz_target!(|input: TransformerConfigInput| {
    // --- Pass 1: parse arbitrary bytes as GGUF and extract transformer config fields ---
    if input.data.len() >= 16 {
        if let Ok(reader) = GgufReader::new(&input.data) {
            // These must never panic on any well-formed parse result.
            let _ = reader.get_string_metadata("general.architecture");
            let _ = reader.get_u32_metadata("llm.embedding_length");
            let _ = reader.get_u32_metadata("llm.attention.head_count");
            let _ = reader.get_u32_metadata("llm.attention.head_count_kv");
            let _ = reader.get_u32_metadata("llm.block_count");
            let _ = reader.get_u32_metadata("llm.context_length");
            let _ = reader.get_u32_metadata("llm.feed_forward_length");
            let _ = reader.get_u32_metadata("llm.vocab_size");
            let _ = reader.get_f32_metadata("llm.attention.layer_norm_rms_eps");
            let _ = reader.get_f32_metadata("llm.rope.freq_base");
            let _ = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
            let _ = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
            let _ = reader.get_u32_metadata("tokenizer.ggml.padding_token_id");
            let _ = reader.get_string_array_metadata("tokenizer.ggml.tokens");
            let _ = reader.validate();
        }
    }

    // --- Pass 2: construct a GGUF stream with known transformer config KVs ---
    let hidden = input.hidden_size.min(65536);
    let n_heads = input.n_heads.max(1) as u32;
    let n_kv_heads = input.n_kv_heads.max(1) as u32;
    let n_layers = input.n_layers.max(1) as u32;
    let vocab = input.vocab_size.min(1_000_000);
    let ctx = input.context_len.min(1_048_576);

    let buf = build_transformer_gguf(hidden, n_heads, n_kv_heads, n_layers, vocab, ctx);
    if let Ok(reader) = GgufReader::new(&buf) {
        let _ = reader.get_u32_metadata("llm.embedding_length");
        let _ = reader.get_u32_metadata("llm.attention.head_count");
        let _ = reader.get_u32_metadata("llm.attention.head_count_kv");
        let _ = reader.get_u32_metadata("llm.block_count");
        let _ = reader.get_u32_metadata("llm.vocab_size");
        let _ = reader.get_u32_metadata("llm.context_length");
        let _ = reader.metadata_keys();
        let _ = reader.metadata_count();
        let _ = reader.validate();
    }

    // --- Pass 3: fuzz BitNetConfig validation with derived field values ---
    let config = BitNetConfig::builder()
        .hidden_size(hidden as usize)
        .num_heads(n_heads as usize)
        .num_key_value_heads(n_kv_heads as usize)
        .num_layers(n_layers as usize)
        .vocab_size(vocab as usize)
        .max_length(ctx as usize)
        .build();
    if let Ok(cfg) = config {
        let _ = cfg.validate();
    }
});

/// Builds a minimal valid GGUF byte stream containing transformer config key-value entries.
fn build_transformer_gguf(
    hidden_size: u32,
    n_heads: u32,
    n_kv_heads: u32,
    n_layers: u32,
    vocab_size: u32,
    context_len: u32,
) -> Vec<u8> {
    const N_KV: u64 = 6;
    let mut buf = Vec::new();

    // GGUF magic + version 3
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    // tensor_count = 0, metadata_kv_count = N_KV
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&N_KV.to_le_bytes());

    write_kv_u32(&mut buf, "llm.embedding_length", hidden_size);
    write_kv_u32(&mut buf, "llm.attention.head_count", n_heads);
    write_kv_u32(&mut buf, "llm.attention.head_count_kv", n_kv_heads);
    write_kv_u32(&mut buf, "llm.block_count", n_layers);
    write_kv_u32(&mut buf, "llm.vocab_size", vocab_size);
    write_kv_u32(&mut buf, "llm.context_length", context_len);

    buf
}

/// Writes a single GGUF key-value entry of type UINT32 (type discriminant 5).
fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    // key: u64 length prefix + UTF-8 bytes
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    // value type UINT32 = 5
    buf.extend_from_slice(&5u32.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
}
