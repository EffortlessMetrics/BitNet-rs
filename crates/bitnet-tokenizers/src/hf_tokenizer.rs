//! Hugging Face tokenizers.json support
//!
//! This module provides support for loading and using tokenizers in the
//! Hugging Face tokenizer.json format. These tokenizers are commonly used
//! with modern transformer models and provide sophisticated tokenization
//! algorithms including WordPiece, BPE, and Unigram.

use ahash::AHashMap;
use anyhow::Result as AnyhowResult;
use bitnet_common::Result;
use std::{collections::HashMap, path::Path};

/// Wrapper for Hugging Face tokenizers
///
/// This struct wraps the `tokenizers` library Tokenizer and adapts it to
/// our `Tokenizer` trait interface. It handles special token detection and
/// management automatically.
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
}

impl HfTokenizer {
    /// Load a tokenizer from a Hugging Face tokenizer.json file
    ///
    /// This method loads the tokenizer and automatically detects special tokens
    /// like BOS (`<s>`, `<bos>`) and EOS (`</s>`, `<eos>`) from the vocabulary.
    ///
    /// # Arguments
    /// * `path` - Path to the tokenizer.json file
    ///
    /// # Returns
    /// A new HfTokenizer instance
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or parsed as a valid tokenizer
    pub fn from_file(path: &Path) -> AnyhowResult<Self> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e))?;

        // Try to discover BOS/EOS from special tokens if present
        let mut bos_id = None;
        let mut eos_id = None;

        // Get vocab and look for common special token patterns
        {
            let vocab = inner.get_vocab(true);
            for (token, id) in vocab {
                // Check for common BOS token patterns
                if token.eq_ignore_ascii_case("<s>")
                    || token.eq_ignore_ascii_case("<bos>")
                    || token.eq_ignore_ascii_case("<|startoftext|>")
                {
                    bos_id = Some(id);
                }
                // Check for common EOS token patterns
                if token.eq_ignore_ascii_case("</s>")
                    || token.eq_ignore_ascii_case("<eos>")
                    || token.eq_ignore_ascii_case("<|endoftext|>")
                {
                    eos_id = Some(id);
                }
            }
        }

        Ok(Self { inner, bos_id, eos_id })
    }

    /// Create a tokenizer directly from vocabulary and BPE merge rules
    ///
    /// This constructor is useful when models embed their tokenizer
    /// definitions (like GGUF) and we need to construct a tokenizer at
    /// runtime without an intermediate JSON file. The vocabulary vector
    /// should contain tokens in their desired id order, while `merges`
    /// should follow the standard `token1 token2` merge format.
    pub fn from_vocab_and_merges(vocab: &[(String, f32)], merges: &[String]) -> AnyhowResult<Self> {
        use tokenizers::{decoders::byte_level::ByteLevel, models::bpe::BPE};

        // Build vocabulary map preserving provided ids
        let vocab_map: HashMap<String, u32> =
            vocab.iter().enumerate().map(|(i, (tok, _))| (tok.clone(), i as u32)).collect();

        // Parse merges in "token1 token2" form into pairs
        let merges_vec: Vec<(String, String)> = merges
            .iter()
            .filter_map(|m| {
                let mut parts = m.split_whitespace();
                let a = parts.next()?.to_string();
                let b = parts.next()?.to_string();
                Some((a, b))
            })
            .collect();

        let bpe = BPE::builder()
            .vocab_and_merges(AHashMap::from_iter(vocab_map), merges_vec)
            .build()
            .map_err(|e| anyhow::anyhow!(e))?;

        let mut inner = tokenizers::Tokenizer::new(bpe);
        // Use byte-level pre-tokenizer/decoder similar to GPT-2
        // NOTE: add_prefix_space(true) is critical for BPE tokenizers to match llama.cpp behavior
        // Without this, "What" tokenizes differently than "ĠWhat" (with leading space)
        inner.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::byte_level::ByteLevel::default().add_prefix_space(true),
        ));
        inner.with_decoder(Some(ByteLevel::default()));

        Ok(Self { inner, bos_id: None, eos_id: None })
    }
}

impl super::Tokenizer for HfTokenizer {
    fn encode(&self, text: &str, add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        use tokenizers::EncodeInput;

        // The Tokenizer trait's third argument is used by prompt templates as
        // "parse embedded special tokens". Hugging Face tokenizer.json files
        // still recognize literal AddedToken specials with post-processing
        // disabled; enabling post-processing here injects template specials
        // such as BOS/EOS a second time for rendered chat prompts.
        let enc = self.inner.encode(EncodeInput::Single(text.into()), false).map_err(|e| {
            bitnet_common::BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                reason: format!("Tokenizer encode error: {}", e),
            })
        })?;

        let mut ids = enc.get_ids().to_vec();

        // Add BOS if requested and not already added
        if add_bos
            && let Some(bos) = self.bos_id
            && (ids.is_empty() || ids[0] != bos)
        {
            ids.insert(0, bos);
        }

        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, true).map_err(|e| {
            bitnet_common::BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                reason: format!("Tokenizer decode error: {}", e),
            })
        })
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.inner.id_to_token(token).map(|s| s.to_string())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        // Use the inner tokenizer's vocab to look up the token
        let vocab = self.inner.get_vocab(true);
        vocab.get(token).copied()
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos_id
    }
    fn eos_token_id(&self) -> Option<u32> {
        self.eos_id
    }
}

impl HfTokenizer {
    pub fn source_name(&self) -> &'static str {
        "hf_json"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tokenizer;

    fn fixture_tokenizer() -> AnyhowResult<HfTokenizer> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tokenizers/valid_tokenizer_a.json");
        HfTokenizer::from_file(&path)
    }

    #[test]
    fn embedded_special_prompt_does_not_get_post_processor_bos() -> AnyhowResult<()> {
        let tokenizer = fixture_tokenizer()?;

        let ids = tokenizer.encode("<s>▁t", false, true)?;

        assert_eq!(ids.first().copied(), Some(1), "embedded BOS should be parsed");
        assert_ne!(ids.get(1).copied(), Some(1), "post-processor BOS must not be injected");
        Ok(())
    }

    #[test]
    fn explicit_bos_policy_still_prepends_single_bos() -> AnyhowResult<()> {
        let tokenizer = fixture_tokenizer()?;

        let ids = tokenizer.encode("▁t", true, true)?;

        assert_eq!(ids.first().copied(), Some(1), "explicit BOS should be prepended");
        assert_ne!(ids.get(1).copied(), Some(1), "explicit BOS should not duplicate");
        Ok(())
    }
}
