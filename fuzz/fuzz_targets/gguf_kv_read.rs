#![no_main]

use arbitrary::Arbitrary;
use bitnet_gguf::{GgufValueType, check_magic, parse_header, read_version};
use bitnet_models::GgufReader;
use libfuzzer_sys::fuzz_target;

/// Fuzz input representing one GGUF key-value entry inside a synthetic GGUF stream.
#[derive(Arbitrary, Debug)]
struct GgufKvInput {
    /// Arbitrary key bytes (may not be valid UTF-8 — the parser must handle it).
    key: Vec<u8>,
    /// Arbitrary value payload bytes.
    value_bytes: Vec<u8>,
    /// Raw value-type discriminant written into the stream.
    value_type: u8,
    /// Additional random bytes appended after the KV entry to stress the parser.
    trailing: Vec<u8>,
}

fuzz_target!(|input: GgufKvInput| {
    // Cap sizes to avoid OOM / timeout while still exercising boundary conditions.
    if input.key.len() > 512 || input.value_bytes.len() > 4096 || input.trailing.len() > 256 {
        return;
    }

    let buf = build_gguf_with_kv(&input.key, &input.value_bytes, input.value_type, &input.trailing);

    // --- Pass 1: low-level bitnet_gguf primitives ---
    // None of these must ever panic.
    let _ = check_magic(&buf);
    let _ = read_version(&buf);
    let _ = parse_header(&buf);

    // --- Pass 2: GgufValueType discriminant exhaustion ---
    let _ = GgufValueType::from_u32(input.value_type as u32);
    // Boundary and sentinel values must also be handled gracefully.
    let _ = GgufValueType::from_u32(0);
    let _ = GgufValueType::from_u32(u32::MAX);

    // --- Pass 3: GgufReader metadata access ---
    if buf.len() < 16 {
        return;
    }
    if let Ok(reader) = GgufReader::new(&buf) {
        let keys = reader.metadata_keys();
        for key in &keys {
            // Exercise every typed accessor — mismatched types must return None, not panic.
            let _ = reader.get_string_metadata(key);
            let _ = reader.get_u32_metadata(key);
            let _ = reader.get_i32_metadata(key);
            let _ = reader.get_f32_metadata(key);
            let _ = reader.get_bool_metadata(key);
            let _ = reader.get_string_array_metadata(key);
            let _ = reader.get_bin_metadata(key);
            let _ = reader.get_bin_or_u8_array(key);
        }

        // General reader invariants must hold regardless of content.
        let _ = reader.metadata_count();
        let _ = reader.tensor_count();
        let _ = reader.alignment();
        let _ = reader.version();
        let _ = reader.validate();
    }

    // --- Pass 4: directly feed raw bytes into parse_header ---
    // Try sub-slices to catch off-by-one errors at the header boundary.
    for end in [8, 16, buf.len().min(32), buf.len()] {
        let _ = parse_header(&buf[..end]);
    }
});

/// Builds a minimal GGUF byte stream containing one key-value entry constructed
/// from the fuzz inputs.
fn build_gguf_with_kv(key: &[u8], value_bytes: &[u8], value_type: u8, trailing: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();

    // GGUF magic bytes
    buf.extend_from_slice(b"GGUF");
    // Format version 3 (u32 LE)
    buf.extend_from_slice(&3u32.to_le_bytes());
    // tensor_count = 0 (u64 LE)
    buf.extend_from_slice(&0u64.to_le_bytes());
    // metadata_kv_count = 1 (u64 LE)
    buf.extend_from_slice(&1u64.to_le_bytes());

    // Key: u64 length prefix, then raw bytes (possibly non-UTF-8)
    let key_len = key.len().min(255);
    buf.extend_from_slice(&(key_len as u64).to_le_bytes());
    buf.extend_from_slice(&key[..key_len]);

    // Value type as u32 LE
    buf.extend_from_slice(&(value_type as u32).to_le_bytes());

    // Value payload
    buf.extend_from_slice(value_bytes);

    // Optional trailing garbage to exercise parser robustness
    buf.extend_from_slice(trailing);

    buf
}
