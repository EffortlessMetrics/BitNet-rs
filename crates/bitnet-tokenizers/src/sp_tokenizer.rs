#[cfg(feature = "spm")]
use crate::Tokenizer;
#[cfg(feature = "spm")]
use bitnet_common::{BitNetError, Result};
#[cfg(feature = "spm")]
use sentencepiece::SentencePieceProcessor;

#[cfg(feature = "spm")]
pub struct SpTokenizer {
    sp: SentencePieceProcessor,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

#[cfg(feature = "spm")]
impl SpTokenizer {
    pub fn from_file(path: &std::path::Path) -> Result<Box<dyn Tokenizer>> {
        let sp = SentencePieceProcessor::open(path).map_err(|e| {
            BitNetError::Io(std::io::Error::other(format!(
                "Failed to load SentencePiece model: {e}"
            )))
        })?;
        Ok(Box::new(Self { sp, bos_token_id: None, eos_token_id: None }))
    }

    pub fn from_gguf_blob(
        bytes: &[u8],
        bos: Option<u32>,
        eos: Option<u32>,
    ) -> Result<Box<dyn Tokenizer>> {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new()?;
        tmp.write_all(bytes)?;
        let sp = SentencePieceProcessor::open(tmp.path()).map_err(|e| {
            BitNetError::Io(std::io::Error::other(format!(
                "Failed to load SentencePiece model from GGUF: {e}"
            )))
        })?;
        Ok(Box::new(Self { sp, bos_token_id: bos, eos_token_id: eos }))
    }
}

#[cfg(feature = "spm")]
impl Tokenizer for SpTokenizer {
    fn encode(&self, text: &str, add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        // Use encode which returns Vec<PieceWithId>
        let pieces = self
            .sp
            .encode(text)
            .map_err(|e| BitNetError::Io(std::io::Error::other(format!("encode failed: {e}"))))?;

        let mut ids: Vec<u32> = pieces.into_iter().map(|p| p.id as u32).collect();

        if add_bos {
            if let Some(b) = self.bos_token_id {
                ids.insert(0, b);
            }
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        // Use decode_piece_ids which takes &[u32] directly
        let s = self.sp.decode_piece_ids(ids).map_err(|e| {
            BitNetError::Io(std::io::Error::other(format!("decode_piece_ids failed: {e}")))
        })?;
        Ok(s)
    }

    fn vocab_size(&self) -> usize {
        self.sp.len()
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        // Decode a single token to get its piece representation
        self.sp.decode_piece_ids(&[token]).ok()
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }
}
