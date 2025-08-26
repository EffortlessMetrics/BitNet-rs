use crate::Tokenizer;
use bitnet_common::Result;
use std::collections::HashMap;
use std::path::Path;

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
        let reverse_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        // Get special tokens
        let bos_token_id =
            metadata.get("tokenizer.ggml.bos_token_id").and_then(|v| v.as_u64()).map(|v| v as u32);
        let eos_token_id =
            metadata.get("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u64()).map(|v| v as u32);

        Ok(Self { vocab, reverse_vocab, bos_token_id, eos_token_id })
    }
}

impl Tokenizer for GgufTokenizer {
    fn encode(&self, text: &str, add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        // Tokenization based on vocab entries
        let mut tokens = Vec::new();

        if add_bos && let Some(bos) = self.bos_token_id {
            tokens.push(bos);
        }

        for piece in text.split_whitespace() {
            if let Some(&id) = self.vocab.get(piece) {
                tokens.push(id);
            } else {
                // Fallback to byte-level encoding for unknown pieces
                for b in piece.bytes() {
                    tokens.push(b as u32);
                }
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

fn read_gguf_metadata(_path: &Path) -> Result<HashMap<String, serde_json::Value>> {
    // This is a simplified version - in reality we'd use the GGUF reader
    // For now, return empty metadata
    tracing::warn!("GGUF metadata reading not yet implemented, using defaults");
    let mut metadata = HashMap::new();
    metadata.insert("tokenizer.ggml.bos_token_id".to_string(), serde_json::json!(1));
    metadata.insert("tokenizer.ggml.eos_token_id".to_string(), serde_json::json!(2));
    Ok(metadata)
}

fn extract_vocab(metadata: &HashMap<String, serde_json::Value>) -> Result<HashMap<String, u32>> {
    use bitnet_common::BitNetError;

    let tokens_val = metadata
        .get("tokenizer.ggml.tokens")
        .ok_or_else(|| BitNetError::Validation("Missing tokenizer.ggml.tokens".to_string()))?;

    let tokens_arr = tokens_val
        .as_array()
        .ok_or_else(|| BitNetError::Validation("tokenizer.ggml.tokens must be an array".to_string()))?;

    let mut vocab = HashMap::new();
    for (idx, token_val) in tokens_arr.iter().enumerate() {
        let token_str = token_val
            .as_str()
            .ok_or_else(|| BitNetError::Validation(format!("Invalid token at index {idx}")))?;
        vocab.insert(token_str.to_string(), idx as u32);
    }

    Ok(vocab)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::io::Write;
    use std::path::Path;

    const GGUF_TYPE_UINT32: u32 = 4;
    const GGUF_TYPE_STRING: u32 = 8;
    const GGUF_TYPE_ARRAY: u32 = 9;

    fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
        let key_bytes = key.as_bytes();
        buf.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(key_bytes);
        buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn write_kv_string_array(buf: &mut Vec<u8>, key: &str, values: &[&str]) {
        let key_bytes = key.as_bytes();
        buf.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(key_bytes);
        buf.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            let b = v.as_bytes();
            buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
            buf.extend_from_slice(b);
        }
    }

    fn create_test_gguf(path: &Path) {
        const ALIGN: usize = 32;
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        let kv_pos = data.len();
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&(ALIGN as u32).to_le_bytes());
        let doff_pos = data.len();
        data.extend_from_slice(&0u64.to_le_bytes());

        let mut kv_count = 0u64;
        write_kv_string_array(
            &mut data,
            "tokenizer.ggml.tokens",
            &["<bos>", "<eos>", "hello", "world"],
        );
        kv_count += 1;
        write_kv_u32(&mut data, "tokenizer.ggml.bos_token_id", 0);
        kv_count += 1;
        write_kv_u32(&mut data, "tokenizer.ggml.eos_token_id", 1);
        kv_count += 1;

        data[kv_pos..kv_pos + 8].copy_from_slice(&kv_count.to_le_bytes());
        let unpadded = data.len();
        let aligned = (unpadded + ALIGN - 1) / ALIGN * ALIGN;
        data[doff_pos..doff_pos + 8].copy_from_slice(&(aligned as u64).to_le_bytes());
        data.resize(aligned, 0);

        let mut file = fs::File::create(path).unwrap();
        file.write_all(&data).unwrap();
    }

    fn read_test_metadata(path: &Path) -> HashMap<String, serde_json::Value> {
        let data = fs::read(path).unwrap();
        let mut offset = 0usize;
        offset += 4; // magic
        offset += 4; // version
        offset += 8; // n_tensors
        let n_kv = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8; // n_kv
        offset += 4; // alignment
        offset += 8; // data_offset

        let mut map = HashMap::new();
        for _ in 0..n_kv {
            let key_len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;
            let key = String::from_utf8(data[offset..offset + key_len].to_vec()).unwrap();
            offset += key_len;
            let vtype = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            offset += 4;
            match vtype {
                GGUF_TYPE_ARRAY => {
                    let etype = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                    offset += 4;
                    let len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
                    offset += 8;
                    if etype == GGUF_TYPE_STRING {
                        let mut arr = Vec::with_capacity(len);
                        for _ in 0..len {
                            let l = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
                            offset += 8;
                            let s = String::from_utf8(data[offset..offset + l].to_vec()).unwrap();
                            offset += l;
                            arr.push(serde_json::Value::String(s));
                        }
                        map.insert(key, serde_json::Value::Array(arr));
                    } else {
                        // Skip unsupported arrays
                        offset += len * 1; // minimal skip
                    }
                }
                GGUF_TYPE_UINT32 => {
                    let val = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                    offset += 4;
                    map.insert(key, json!(val));
                }
                _ => {}
            }
        }
        map
    }

    #[test]
    fn test_extract_vocab_from_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let gguf_path = dir.path().join("test.gguf");
        create_test_gguf(&gguf_path);

        let metadata = read_test_metadata(&gguf_path);
        let vocab = extract_vocab(&metadata).unwrap();
        assert_eq!(vocab.get("hello"), Some(&2));
        assert_eq!(vocab.get("world"), Some(&3));

        let tokenizer = GgufTokenizer {
            reverse_vocab: vocab.iter().map(|(k, &v)| (v, k.clone())).collect(),
            vocab,
            bos_token_id: metadata
                .get("tokenizer.ggml.bos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
            eos_token_id: metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
        };

        let encoded = tokenizer.encode("hello world", true, false).unwrap();
        assert_eq!(encoded, vec![0, 2, 3]);

        let decoded = tokenizer.decode(&[2, 3]).unwrap();
        assert_eq!(decoded, "helloworld");
    }
}
