#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use bitnet_tokenizers::{BasicTokenizer, HfTokenizer, Tokenizer};
use libfuzzer_sys::fuzz_target;

/// Structured portion derived from fuzz bytes for BPE construction paths.
#[derive(Arbitrary, Debug)]
struct BpeInput {
    /// Vocabulary entries (token string, score).
    vocab: Vec<(String, f32)>,
    /// BPE merge rules, each in "token_a token_b" format.
    merges: Vec<String>,
    /// Whether to add BOS during encode.
    add_bos: bool,
    /// Whether to add special tokens during encode.
    add_special: bool,
}

fuzz_target!(|data: &[u8]| {
    // --- Path 1: BasicTokenizer encode/decode on arbitrary UTF-8 text ---
    if let Ok(text) = std::str::from_utf8(data) {
        let tok = BasicTokenizer::new();

        if let Ok(tokens) = tok.encode(text, false, false) {
            // Invariant: every token ID must be within vocab range.
            for &id in &tokens {
                assert!(
                    (id as usize) < tok.vocab_size(),
                    "token id {id} >= vocab_size {}",
                    tok.vocab_size()
                );
            }
            // Decode must not panic and must yield a string.
            let _ = tok.decode(&tokens);
        }

        // Additional flag combinations must not panic.
        let _ = tok.encode(text, true, true);
        let _ = tok.encode(text, true, false);
        let _ = tok.encode(text, false, true);
    }

    // --- Path 2: BasicTokenizer decode on arbitrary token ID sequences ---
    {
        let tok = BasicTokenizer::new();
        // Reinterpret raw bytes as little-endian u32 token IDs.
        let token_ids: Vec<u32> = data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // Must not panic regardless of what IDs are passed.
        let _ = tok.decode(&token_ids);
    }

    // --- Path 3: HfTokenizer BPE construction + encode from structured input ---
    // Use `Unstructured` to derive structured data from the same byte buffer so
    // that the fuzzer can explore both paths with a single input corpus.
    if data.len() >= 4 {
        let mut u = Unstructured::new(data);
        if let Ok(bpe) = BpeInput::arbitrary(&mut u) {
            // Cap sizes to keep individual runs bounded.
            let vocab: Vec<(String, f32)> =
                bpe.vocab.into_iter().take(256).collect();
            let merges: Vec<String> =
                bpe.merges.into_iter().take(256).collect();

            if !vocab.is_empty() {
                if let Ok(hf_tok) =
                    HfTokenizer::from_vocab_and_merges(&vocab, &merges)
                {
                    // Exercise BPE encode on the raw bytes interpreted as text.
                    if let Ok(text) = std::str::from_utf8(data) {
                        let _ = hf_tok.encode(text, bpe.add_bos, bpe.add_special);
                    }
                    // Always exercise a few fixed strings to stress merge rules.
                    let _ = hf_tok.encode("hello world", false, false);
                    let _ = hf_tok.encode("", false, false);
                    let _ = hf_tok.encode("BitNet 1.58", true, true);
                }
            }
        }
    }

    // --- Path 4: JSON config parsing ---
    // `TokenizerConfig` does not implement `serde::Deserialize`, so we parse
    // the bytes as a `serde_json::Value` and attempt to construct a BPE
    // tokenizer from any array-shaped vocab it contains.
    if let Ok(text) = std::str::from_utf8(data) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
            // If the JSON is a flat array of strings, treat it as a vocabulary.
            if let Some(arr) = json.as_array() {
                let vocab_from_json: Vec<(String, f32)> = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| (s.to_string(), 0.0_f32)))
                    .take(256)
                    .collect();
                if !vocab_from_json.is_empty() {
                    let _ = HfTokenizer::from_vocab_and_merges(&vocab_from_json, &[]);
                }
            }

            // If the JSON is an object, look for "vocab" / "merges" keys that
            // mirror a tokenizers.json layout and exercise construction.
            if let Some(obj) = json.as_object() {
                let vocab_from_obj: Vec<(String, f32)> = obj
                    .get("vocab")
                    .and_then(|v| v.as_object())
                    .map(|m| {
                        m.iter()
                            .filter_map(|(k, v)| {
                                v.as_f64().map(|score| (k.clone(), score as f32))
                            })
                            .take(256)
                            .collect()
                    })
                    .unwrap_or_default();

                let merges_from_obj: Vec<String> = obj
                    .get("merges")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .take(256)
                            .collect()
                    })
                    .unwrap_or_default();

                if !vocab_from_obj.is_empty() {
                    let _ = HfTokenizer::from_vocab_and_merges(
                        &vocab_from_obj,
                        &merges_from_obj,
                    );
                }
            }
        }
    }
});
