//! SentencePiece tokenizer support

#[cfg(feature = "spm")]
use anyhow::Result as AnyhowResult;
#[cfg(feature = "spm")]
use bitnet_common::{BitNetError, ModelError, Result};
#[cfg(feature = "spm")]
use std::path::Path;

#[cfg(feature = "spm")]
pub struct SpmTokenizer {
    inner: sentencepiece::SentencePieceProcessor,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
}

#[cfg(feature = "spm")]
impl SpmTokenizer {
    pub fn from_file(path: &Path) -> AnyhowResult<Self> {
        let spp =
            sentencepiece::SentencePieceProcessor::open(path).map_err(|e| anyhow::anyhow!(e))?;
        let bos_id = spp.bos_id();
        let eos_id = spp.eos_id();
        Ok(Self { inner: spp, bos_id, eos_id })
    }
}

#[cfg(feature = "spm")]
impl super::Tokenizer for SpmTokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        let pieces = self.inner.encode(text).map_err(|e| {
            BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("Tokenizer encode error: {}", e),
            })
        })?;
        let mut ids: Vec<u32> = pieces.into_iter().map(|p| p.id).collect();

        if add_bos
            && let Some(bos) = self.bos_id
            && ids.first().copied() != Some(bos)
        {
            ids.insert(0, bos);
        }

        if add_special
            && let Some(eos) = self.eos_id
            && ids.last().copied() != Some(eos)
        {
            ids.push(eos);
        }

        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode_piece_ids(ids).map_err(|e| {
            BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("Tokenizer decode error: {}", e),
            })
        })
    }

    fn vocab_size(&self) -> usize {
        self.inner.len()
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        // Use decode_piece_ids to convert a single token ID to its piece representation
        // This is less efficient than a direct id_to_piece method but provides the same functionality
        match self.inner.decode_piece_ids(&[token]) {
            Ok(decoded) => {
                if decoded.is_empty() {
                    None
                } else {
                    Some(decoded)
                }
            }
            Err(_) => None,
        }
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos_id
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_id
    }
}

#[cfg(feature = "spm")]
impl SpmTokenizer {
    pub fn source_name(&self) -> &'static str {
        "spm"
    }
}
