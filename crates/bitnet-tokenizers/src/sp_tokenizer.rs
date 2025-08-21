use sentencepiece::SentencePieceProcessor;
use bitnet_common::{Result, BitNetError};
use crate::Tokenizer;

/// SentencePiece tokenizer loaded from GGUF blob
pub struct SpTokenizer {
    sp: SentencePieceProcessor,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

impl SpTokenizer {
    /// Get a piece by ID (simplified - just returns a placeholder for now)
    fn get_piece(&self, id: u32) -> Option<String> {
        // This is a placeholder - a real implementation would look up the actual piece
        // For now just return a token representation
        if id < self.sp.len() as u32 {
            Some(format!("<tok_{}>", id))
        } else {
            None
        }
    }
    
    /// Create tokenizer from GGUF embedded SentencePiece model
    pub fn from_gguf_blob(bytes: &[u8], bos: Option<u32>, eos: Option<u32>) -> Result<Box<dyn Tokenizer>> {
        use std::io::Write;
        
        // SentencePiece needs a file path, so write to temp
        let mut tmp = tempfile::NamedTempFile::new()?;
        tmp.write_all(bytes)?;
        tmp.flush()?;
        
        let sp = SentencePieceProcessor::open(tmp.path())
            .map_err(|e| BitNetError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to load SentencePiece model: {}", e))))?;
        
        Ok(Box::new(Self { 
            sp, 
            bos_token_id: bos, 
            eos_token_id: eos 
        }) as Box<dyn Tokenizer>)
    }
    
    /// Create tokenizer from a SentencePiece model file
    pub fn from_file(path: &std::path::Path) -> Result<Box<dyn Tokenizer>> {
        let sp = SentencePieceProcessor::open(path)
            .map_err(|e| BitNetError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to load SentencePiece model: {}", e))))?;
        
        // Standard BOS/EOS IDs for SentencePiece (can be overridden)
        let bos = Some(1u32);  // <s> is typically ID 1
        let eos = Some(2u32);  // </s> is typically ID 2
        
        Ok(Box::new(Self { 
            sp,
            bos_token_id: bos,
            eos_token_id: eos
        }) as Box<dyn Tokenizer>)
    }
}

impl Tokenizer for SpTokenizer {
    fn encode(&self, text: &str, add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        let pieces = self.sp
            .encode(text)
            .map_err(|e| BitNetError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Encode failed: {}", e))))?;
            
        let mut ids: Vec<u32> = pieces
            .into_iter()
            .map(|p| p.id as u32)
            .collect();
            
        if add_bos {
            if let Some(bos) = self.bos_token_id {
                ids.insert(0, bos);
            }
        }
        
        Ok(ids)
    }
    
    fn decode(&self, ids: &[u32]) -> Result<String> {
        // For now, just concatenate the pieces - this is a simplified decoder
        // A proper implementation would handle subwords correctly
        let mut result = String::new();
        for &id in ids {
            if let Some(piece) = self.get_piece(id) {
                // Remove the sentencepiece marker and append
                let clean_piece = piece.replace("▁", " ");
                result.push_str(&clean_piece);
            }
        }
        // Clean up extra spaces
        Ok(result.trim().to_string())
    }
    
    fn vocab_size(&self) -> usize {
        self.sp.len()
    }
    
    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.get_piece(token)
    }
    
    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }
}