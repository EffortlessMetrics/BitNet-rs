#![no_main]

use arbitrary::Arbitrary;
use bitnet_tokenizers::{BasicTokenizer, Tokenizer};
use libfuzzer_sys::fuzz_target;

/// Structured UTF-8 input that lets the fuzzer generate valid Unicode strings
/// directly rather than filtering arbitrary byte slices.
#[derive(Arbitrary, Debug)]
struct EncodeInput {
    text: String,
    add_bos: bool,
    add_special: bool,
    /// Additional token IDs to decode (arbitrary, may be out-of-range).
    decode_ids: Vec<u32>,
}

fuzz_target!(|input: EncodeInput| {
    // Cap text length to keep individual runs bounded.
    if input.text.len() > 4096 {
        return;
    }

    let tok = BasicTokenizer::new();
    let vocab_size = tok.vocab_size();

    // Primary encode path must not panic.
    if let Ok(tokens) = tok.encode(&input.text, input.add_bos, input.add_special) {
        // Every returned token ID must be within [0, vocab_size).
        for &id in &tokens {
            assert!(
                (id as usize) < vocab_size,
                "encode returned out-of-range token id {id} for vocab_size {vocab_size}",
            );
        }
        // Round-trip decode must not panic.
        let _ = tok.decode(&tokens);
    }

    // All flag combinations must not panic.
    let _ = tok.encode(&input.text, true, true);
    let _ = tok.encode(&input.text, true, false);
    let _ = tok.encode(&input.text, false, true);
    let _ = tok.encode(&input.text, false, false);

    // Decoding arbitrary (possibly out-of-range) IDs must not panic.
    let ids: Vec<u32> = input.decode_ids.into_iter().take(512).collect();
    let _ = tok.decode(&ids);
});
