#![no_main]

use bitnet_tokenizers::{
    BasicTokenizer, Tokenizer, TokenizerConfig, universal::UniversalTokenizer,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Need at least 4 bytes for one u32 token ID.
    if data.len() < 4 || data.len() > 16 * 1024 {
        return;
    }

    // Interpret raw bytes as a u32 slice of token IDs.
    let token_ids: Vec<u32> = data[..data.len() / 4 * 4]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    if token_ids.is_empty() {
        return;
    }

    // BasicTokenizer: decode arbitrary IDs must not panic.
    let basic = BasicTokenizer::new();
    let _ = basic.decode(&token_ids);
    let _ = basic.token_to_piece(token_ids[0]);

    // UniversalTokenizer with multiple model types: decode must not panic.
    for model_type in &["gpt2", "bpe", "llama", "llama3"] {
        let config = TokenizerConfig {
            model_type: model_type.to_string(),
            vocab_size: 256,
            ..TokenizerConfig::default()
        };
        if let Ok(tok) = UniversalTokenizer::new(config) {
            let _ = tok.decode(&token_ids);
        }
    }
});
