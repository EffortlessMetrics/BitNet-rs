//! Hugging Face tokenizers.json support

use anyhow::Result as AnyhowResult;
use bitnet_common::Result;
use std::path::Path;

pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
}

impl HfTokenizer {
    pub fn from_file(path: &Path) -> AnyhowResult<Self> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e))?;

        // Try to discover BOS/EOS from special tokens if present
        let mut bos_id = None;
        let mut eos_id = None;

        // Get vocab and look for special tokens
        {
            let vocab = inner.get_vocab(true);
            for (token, id) in vocab {
                if token.eq_ignore_ascii_case("<s>") || token.eq_ignore_ascii_case("<bos>") {
                    bos_id = Some(id);
                }
                if token.eq_ignore_ascii_case("</s>") || token.eq_ignore_ascii_case("<eos>") {
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
