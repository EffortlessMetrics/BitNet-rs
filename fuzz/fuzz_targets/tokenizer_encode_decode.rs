#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use bitnet_common::QuantizationType;
use bitnet_tokenizers::{
    BasicTokenizer, HfTokenizer, Tokenizer, TokenizerConfig,
    strategy::{BitNetTokenizerWrapper, Gpt2TokenizerWrapper, LlamaTokenizerWrapper},
    universal::UniversalTokenizer,
};
use libfuzzer_sys::fuzz_target;
use std::sync::Arc;

/// Structured input for exercising wrapper tokenizers.
#[derive(Arbitrary, Debug)]
struct WrapperInput {
    vocab_size: u16,
    bos_token_id: Option<u16>,
    eos_token_id: Option<u16>,
    pad_token_id: Option<u16>,
    add_bos: bool,
    add_special: bool,
}

/// Structured input for BPE construction.
#[derive(Arbitrary, Debug)]
struct BpeRoundTripInput {
    vocab: Vec<(String, f32)>,
    merges: Vec<String>,
    token_ids: Vec<u16>,
}

fuzz_target!(|data: &[u8]| {
    // --- Path 1: BasicTokenizer encode → decode round-trip ---
    if let Ok(text) = std::str::from_utf8(data) {
        let tok = BasicTokenizer::new();
        if let Ok(tokens) = tok.encode(text, false, false) {
            // decode must not panic
            let _ = tok.decode(&tokens);
        }
        // with_config paths
        if data.len() >= 2 {
            let mut u = Unstructured::new(data);
            if let Ok(wi) = WrapperInput::arbitrary(&mut u) {
                let vocab_size = (wi.vocab_size as usize).max(1);
                let bos = wi.bos_token_id.map(|v| (v as u32) % vocab_size as u32);
                let eos = wi.eos_token_id.map(|v| (v as u32) % vocab_size as u32);
                let pad = wi.pad_token_id.map(|v| (v as u32) % vocab_size as u32);
                let tok2 = BasicTokenizer::with_config(vocab_size, bos, eos, pad);
                if let Ok(tokens2) = tok2.encode(text, wi.add_bos, wi.add_special) {
                    let _ = tok2.decode(&tokens2);
                }
                // arbitrary token IDs must not panic on decode
                let arb_ids: Vec<u32> = data
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
                    .collect();
                let _ = tok2.decode(&arb_ids);
            }
        }
    }

    // --- Path 2: UniversalTokenizer (in-memory, mock backend) ---
    {
        for model_type in &["gpt2", "bpe", "llama", "llama3"] {
            let config = TokenizerConfig {
                model_type: model_type.to_string(),
                vocab_size: 256,
                ..TokenizerConfig::default()
            };
            if let Ok(tok) = UniversalTokenizer::new(config) {
                if let Ok(text) = std::str::from_utf8(data) {
                    if let Ok(tokens) = tok.encode(text, false, false) {
                        let _ = tok.decode(&tokens);
                    }
                }
                // arbitrary decode
                let ids: Vec<u32> =
                    data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let _ = tok.decode(&ids);
            }
        }
    }

    // --- Path 3: Wrapper tokenizers wrapping BasicTokenizer ---
    {
        let inner: Arc<dyn Tokenizer> = Arc::new(BasicTokenizer::new());

        // Gpt2TokenizerWrapper
        if let Ok(gpt2) = Gpt2TokenizerWrapper::new(Arc::clone(&inner)) {
            if let Ok(text) = std::str::from_utf8(data) {
                if let Ok(tokens) = gpt2.encode(text, false, false) {
                    let _ = gpt2.decode(&tokens);
                }
            }
        }

        // LlamaTokenizerWrapper (vocab_size must be >= 1)
        if let Ok(llama) = LlamaTokenizerWrapper::new(Arc::clone(&inner), 32000) {
            if let Ok(text) = std::str::from_utf8(data) {
                if let Ok(tokens) = llama.encode(text, false, false) {
                    let _ = llama.decode(&tokens);
                }
            }
        }

        // BitNetTokenizerWrapper
        for qt in &[QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
            if let Ok(bn) = BitNetTokenizerWrapper::new(Arc::clone(&inner), *qt) {
                if let Ok(text) = std::str::from_utf8(data) {
                    if let Ok(tokens) = bn.encode(text, false, false) {
                        let _ = bn.decode(&tokens);
                    }
                }
            }
        }
    }

    // --- Path 4: HfTokenizer encode → decode round-trip ---
    if data.len() >= 4 {
        let mut u = Unstructured::new(data);
        if let Ok(bpe) = BpeRoundTripInput::arbitrary(&mut u) {
            let vocab: Vec<(String, f32)> = bpe.vocab.into_iter().take(128).collect();
            let merges: Vec<String> = bpe.merges.into_iter().take(128).collect();
            let token_ids: Vec<u32> =
                bpe.token_ids.into_iter().map(|id| id as u32).collect();

            if !vocab.is_empty() {
                if let Ok(hf) = HfTokenizer::from_vocab_and_merges(&vocab, &merges) {
                    if let Ok(text) = std::str::from_utf8(data) {
                        if let Ok(tokens) = hf.encode(text, false, false) {
                            let _ = hf.decode(&tokens);
                        }
                    }
                    // decode arbitrary token ids must not panic
                    let _ = hf.decode(&token_ids);
                }
            }
        }
    }
});
