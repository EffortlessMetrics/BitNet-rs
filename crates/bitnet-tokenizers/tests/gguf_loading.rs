use bitnet_common::Result;
use bitnet_tokenizers::Tokenizer;
use bitnet_tokenizers::gguf_tokenizer::GgufTokenizer;
use tempfile::NamedTempFile;

fn write_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend(&(bytes.len() as u64).to_le_bytes());
    buf.extend(bytes);
}

fn build_test_gguf(tokens: &[&str], bos: u32, eos: u32) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend(&3u32.to_le_bytes());
    buf.extend(&0u64.to_le_bytes());
    let kv_count = 3u64;
    buf.extend(&kv_count.to_le_bytes());
    buf.extend(&32u32.to_le_bytes());
    buf.extend(&0u64.to_le_bytes()); // placeholder for data_offset

    // tokens array
    write_string(&mut buf, "tokenizer.ggml.tokens");
    buf.extend(&9u32.to_le_bytes()); // array type
    buf.extend(&8u32.to_le_bytes()); // element type string
    buf.extend(&(tokens.len() as u64).to_le_bytes());
    for t in tokens {
        write_string(&mut buf, t);
    }

    // bos token id
    write_string(&mut buf, "tokenizer.ggml.bos_token_id");
    buf.extend(&4u32.to_le_bytes());
    buf.extend(&bos.to_le_bytes());

    // eos token id
    write_string(&mut buf, "tokenizer.ggml.eos_token_id");
    buf.extend(&4u32.to_le_bytes());
    buf.extend(&eos.to_le_bytes());

    // align and set data_offset
    let data_offset = ((buf.len() + 31) / 32) * 32;
    buf.resize(data_offset, 0);
    let doff_bytes = (data_offset as u64).to_le_bytes();
    buf[28..36].copy_from_slice(&doff_bytes);

    buf
}

#[test]
fn gguf_vocab_and_special_tokens_loaded() -> Result<()> {
    let tokens = ["<bos>", "<eos>", "hello", "world"];
    let bos = 0u32;
    let eos = 1u32;
    let bytes = build_test_gguf(&tokens, bos, eos);
    let tmp = NamedTempFile::new()?;
    std::fs::write(tmp.path(), &bytes)?;

    let tokenizer = GgufTokenizer::from_gguf_file(tmp.path())?;
    assert_eq!(tokenizer.vocab_size(), tokens.len());
    assert_eq!(tokenizer.bos_token_id(), Some(bos));
    assert_eq!(tokenizer.eos_token_id(), Some(eos));
    assert_eq!(tokenizer.token_to_piece(bos).as_deref(), Some("<bos>"));
    assert_eq!(tokenizer.token_to_piece(eos).as_deref(), Some("<eos>"));
    assert_eq!(tokenizer.token_to_piece(2).as_deref(), Some("hello"));
    Ok(())
}
