#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    key: Vec<u8>,
    value_bytes: Vec<u8>,
    metadata_type_byte: u8,
}

fuzz_target!(|input: FuzzInput| {
    // Limit input size to prevent OOM
    if input.key.len() > 256 || input.value_bytes.len() > 1024 {
        return;
    }

    // Build a minimal GGUF byte stream with one arbitrary metadata entry
    let mut buf = Vec::new();
    // Magic: "GGUF"
    buf.extend_from_slice(b"GGUF");
    // version (u32 LE) = 3
    buf.extend_from_slice(&3u32.to_le_bytes());
    // tensor_count (u64) = 0
    buf.extend_from_slice(&0u64.to_le_bytes());
    // metadata_kv_count (u64) = 1
    buf.extend_from_slice(&1u64.to_le_bytes());
    // key: length-prefixed string
    let key_len = input.key.len().min(255) as u64;
    buf.extend_from_slice(&key_len.to_le_bytes());
    buf.extend_from_slice(&input.key[..key_len as usize]);
    // value type byte (cast to u32 LE as GGUF stores type as u32)
    buf.extend_from_slice(&(input.metadata_type_byte as u32).to_le_bytes());
    // value payload
    buf.extend_from_slice(&input.value_bytes);

    // The parser must never panic; returning Err is fine.
    let _ = bitnet_gguf::parse_header(&buf);
});
