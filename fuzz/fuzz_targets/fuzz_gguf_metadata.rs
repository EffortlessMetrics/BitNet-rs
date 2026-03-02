#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct MetadataInput {
    /// Number of KV pairs to encode (clamped).
    n_kv: u8,
    /// Raw key-value payload bytes.
    kv_payload: Vec<u8>,
    /// Random bytes appended after the header for extra coverage.
    trailing: Vec<u8>,
}

fuzz_target!(|input: MetadataInput| {
    // Limit total size to prevent OOM.
    if input.kv_payload.len() > 4096 || input.trailing.len() > 4096 {
        return;
    }

    let n_kv = (input.n_kv % 8) as u64; // 0-7 KV pairs

    // Build a minimal GGUF v3 byte stream with fuzzed metadata.
    let mut buf = Vec::with_capacity(24 + input.kv_payload.len() + input.trailing.len());

    // Magic: "GGUF"
    buf.extend_from_slice(b"GGUF");
    // Version (u32 LE) = 3
    buf.extend_from_slice(&3u32.to_le_bytes());
    // tensor_count (u64 LE) = 0
    buf.extend_from_slice(&0u64.to_le_bytes());
    // metadata_kv_count (u64 LE)
    buf.extend_from_slice(&n_kv.to_le_bytes());
    // Fuzzed KV payload — may be truncated, malformed, or have wrong types.
    buf.extend_from_slice(&input.kv_payload);
    buf.extend_from_slice(&input.trailing);

    // The parser must never panic on malformed input.
    let _ = bitnet_gguf::parse_header(&buf);

    // Also test with completely arbitrary bytes (no valid header).
    let _ = bitnet_gguf::parse_header(&input.kv_payload);

    // Test with various invalid magic bytes.
    if input.kv_payload.len() >= 24 {
        let _ = bitnet_gguf::parse_header(&input.kv_payload[..24]);
    }
});
