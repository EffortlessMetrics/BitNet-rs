//! Hugging Face tokenizers.json support
//!
//! This module provides support for loading and using tokenizers in the
//! Hugging Face tokenizer.json format. These tokenizers are commonly used
//! with modern transformer models and provide sophisticated tokenization
//! algorithms including WordPiece, BPE, and Unigram.

use anyhow::Result as AnyhowResult;
use bitnet_common::Result;
use std::path::Path;

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
    /// like BOS (<s>, <bos>) and EOS (</s>, <eos>) from the vocabulary.
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
}

impl super::Tokenizer for HfTokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        use tokenizers::EncodeInput;

        let enc =
            self.inner.encode(EncodeInput::Single(text.into()), add_special).map_err(|e| {
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

        // Add EOS if requested
        if add_special
            && let Some(eos) = self.eos_id
            && (ids.is_empty() || ids[ids.len() - 1] != eos)
        {
            ids.push(eos);
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
