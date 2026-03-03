#![no_main]

use arbitrary::Arbitrary;
use bitnet_tokenizers::{BasicTokenizer, Tokenizer};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct TokenMapInput {
    /// Arbitrary token IDs — including values far beyond any real vocab.
    token_ids: Vec<u32>,
    /// Custom vocab size for a second tokenizer instance.
    vocab_size: u16,
    bos: Option<u32>,
    eos: Option<u32>,
    pad: Option<u32>,
}

fuzz_target!(|input: TokenMapInput| {
    if input.token_ids.is_empty() {
        return;
    }
    let ids: Vec<u32> = input.token_ids.iter().copied().take(512).collect();

    // Default BasicTokenizer — out-of-range IDs must not panic.
    let basic = BasicTokenizer::new();
    for &id in &ids {
        let _ = basic.token_to_piece(id);
    }
    let _ = basic.decode(&ids);
    let _ = basic.vocab_size();

    // Custom-configured tokenizer with arbitrary vocab size.
    let vocab = (input.vocab_size as usize).max(1);
    let custom = BasicTokenizer::with_config(vocab, input.bos, input.eos, input.pad);
    for &id in &ids {
        let _ = custom.token_to_piece(id);
    }
    let _ = custom.decode(&ids);
    let _ = custom.vocab_size();

    // Verify token_to_piece for special tokens (may or may not be within vocab).
    if let Some(bos) = input.bos {
        let _ = custom.token_to_piece(bos);
    }
    if let Some(eos) = input.eos {
        let _ = custom.token_to_piece(eos);
    }
});
