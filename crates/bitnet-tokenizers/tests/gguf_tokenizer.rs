#![cfg(feature = "integration-tests")]
use bitnet_tokenizers::{Tokenizer, gguf_tokenizer::GgufTokenizer};
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

fn create_dummy_gguf(path: &std::path::Path) {
    let mut tokens: Vec<String> = (0..256).map(|i| format!("<0x{:02X}>", i)).collect();
    tokens.push("<BOS>".to_string());
    tokens.push("<EOS>".to_string());
    let bos_id = 256u32;
    let eos_id = 257u32;

    let mut meta = Vec::new();
    // helper functions
    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }
    fn write_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
        write_string(buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes());
        write_string(buf, value);
    }
    fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
        write_string(buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }
    fn write_kv_tokens(buf: &mut Vec<u8>, key: &str, tokens: &[String]) {
        write_string(buf, key);
        buf.extend_from_slice(&9u32.to_le_bytes()); // array
        buf.extend_from_slice(&8u32.to_le_bytes()); // string elements
        buf.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for t in tokens {
            write_string(buf, t);
        }
    }

    write_kv_string(&mut meta, "general.architecture", "test");
    write_kv_u32(&mut meta, "tokenizer.ggml.bos_token_id", bos_id);
    write_kv_u32(&mut meta, "tokenizer.ggml.eos_token_id", eos_id);
    write_kv_tokens(&mut meta, "tokenizer.ggml.tokens", &tokens);

    let kv_count = 4u64;
    let alignment = 32u32;
    let header_size = 4 + 4 + 8 + 8 + 4 + 8;
    let data_offset = ((header_size + meta.len() + alignment as usize - 1) / alignment as usize)
        * alignment as usize;

    let mut file = Vec::new();
    file.extend_from_slice(b"GGUF");
    file.extend_from_slice(&3u32.to_le_bytes());
    file.extend_from_slice(&0u64.to_le_bytes());
    file.extend_from_slice(&kv_count.to_le_bytes());
    file.extend_from_slice(&alignment.to_le_bytes());
    file.extend_from_slice(&(data_offset as u64).to_le_bytes());
    file.extend_from_slice(&meta);
    file.resize(data_offset, 0); // pad to data_offset
    let mut f = File::create(path).unwrap();
    f.write_all(&file).unwrap();
}

#[test]
fn test_encode_decode_parity() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    create_dummy_gguf(&path);
    let tokenizer = GgufTokenizer::from_gguf_file(&path).unwrap();

    let tokens = tokenizer.encode("Hi", false, false).unwrap();
    assert_eq!(tokens, vec![0x48, 0x69]);
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, "Hi");

    let tokens_with_bos = tokenizer.encode("Hi", true, false).unwrap();
    assert_eq!(tokens_with_bos[0], 256);
    assert_eq!(&tokens_with_bos[1..], &[0x48, 0x69]);
}
