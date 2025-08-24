//! SentencePiece tokenizer support

#[cfg(feature = "spm")]
use anyhow::Result;
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
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut spp = sentencepiece::SentencePieceProcessor::new();
        spp.load(path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid path"))?)?;

        let bos_id = spp.bos_id().ok().map(|x| x as u32);
        let eos_id = spp.eos_id().ok().map(|x| x as u32);

        Ok(Self { inner: spp, bos_id, eos_id })
    }
}

#[cfg(feature = "spm")]
impl super::Tokenizer for SpmTokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        let mut ids = self.inner.encode(text).map_err(|e| anyhow::anyhow!(e))?;

        if add_bos {
            if let Some(bos) = self.bos_id {
                ids.insert(0, bos as i32);
            }
        }

        if add_special {
            if let Some(eos) = self.eos_id {
                ids.push(eos as i32);
            }
        }

        Ok(ids.into_iter().map(|x| x as u32).collect())
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let vec_i32: Vec<i32> = ids.iter().map(|x| *x as i32).collect();
        self.inner.decode(&vec_i32).map_err(|e| anyhow::anyhow!(e))
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size() as usize
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.inner.id_to_piece(token as i32).ok()
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
