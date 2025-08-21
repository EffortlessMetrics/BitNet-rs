use std::collections::HashMap;
use std::path::Path;
use bitnet_common::Result;
use crate::Tokenizer;

/// Tokenizer that reads vocab from GGUF files
pub struct GgufTokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

impl GgufTokenizer {
    pub fn from_gguf_file(path: &Path) -> Result<Self> {
        // Read GGUF metadata to get tokenizer info
        let metadata = read_gguf_metadata(path)?;
        
        // Extract vocabulary
        let vocab = extract_vocab(&metadata)?;
        let reverse_vocab: HashMap<u32, String> = vocab.iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
        
        // Get special tokens
        let bos_token_id = metadata.get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let eos_token_id = metadata.get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        
        Ok(Self {
            vocab,
            reverse_vocab,
            bos_token_id,
            eos_token_id,
        })
    }
}

impl Tokenizer for GgufTokenizer {
    fn encode(&self, text: &str, add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        // Simple byte-level tokenization (like GPT-2)
        let mut tokens = Vec::new();
        
        if add_bos {
            if let Some(bos) = self.bos_token_id {
                tokens.push(bos);
            }
        }
        
        // Convert text to bytes and lookup in vocab
        for byte in text.bytes() {
            let byte_str = format!("<0x{:02X}>", byte);
            if let Some(&token_id) = self.vocab.get(&byte_str) {
                tokens.push(token_id);
            } else {
                // Fallback to direct byte value if not in vocab
                tokens.push(byte as u32);
            }
        }
        
        Ok(tokens)
    }
    
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut text = String::new();
        
        for &token in tokens {
            if let Some(token_str) = self.reverse_vocab.get(&token) {
                // Handle byte tokens
                if token_str.starts_with("<0x") && token_str.ends_with(">") {
                    if let Ok(byte_val) = u8::from_str_radix(&token_str[3..5], 16) {
                        text.push(byte_val as char);
                    }
                } else {
                    text.push_str(token_str);
                }
            } else if token < 256 {
                // Direct byte value
                text.push(token as u8 as char);
            }
        }
        
        Ok(text)
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.reverse_vocab.get(&token).cloned()
    }
    
    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }
}

fn read_gguf_metadata(path: &Path) -> Result<HashMap<String, serde_json::Value>> {
    // This is a simplified version - in reality we'd use the GGUF reader
    // For now, return empty metadata
    tracing::warn!("GGUF metadata reading not yet implemented, using defaults");
    let mut metadata = HashMap::new();
    metadata.insert("tokenizer.ggml.bos_token_id".to_string(), serde_json::json!(1));
    metadata.insert("tokenizer.ggml.eos_token_id".to_string(), serde_json::json!(2));
    Ok(metadata)
}

fn extract_vocab(metadata: &HashMap<String, serde_json::Value>) -> Result<HashMap<String, u32>> {
    // Extract vocabulary from GGUF metadata
    // For now, create a simple byte-level vocab
    let mut vocab = HashMap::new();
    
    // Add byte tokens (like GPT-2)
    for i in 0..256 {
        vocab.insert(format!("<0x{:02X}>", i), i);
    }
    
    // TODO: Read actual vocab from GGUF metadata
    
    Ok(vocab)
}