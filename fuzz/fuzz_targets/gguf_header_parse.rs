#![no_main]

use bitnet_gguf::{GgufValueType, check_magic, parse_header, read_version};
use bitnet_models::GgufReader;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Low-level bitnet_gguf header helpers must never panic on arbitrary bytes.
    let _has_magic = check_magic(data);
    let _version = read_version(data);
    let _header = parse_header(data);

    // Exercise GgufValueType discriminant for every 4-byte aligned window.
    let mut i = 0;
    while i + 4 <= data.len() {
        let discriminant = u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
        let _ = GgufValueType::from_u32(discriminant);
        i += 4;
    }

    // Higher-level GgufReader: must not panic even on malformed input.
    if data.len() < 16 {
        return;
    }
    if let Ok(reader) = GgufReader::new(data) {
        // Header field accessors must not panic.
        let _ = reader.version();
        let _ = reader.alignment();
        let _ = reader.tensor_count();
        let _ = reader.metadata_count();
        let _ = reader.metadata_kv_count();
        let _ = reader.data_offset();
        let _ = reader.file_size();

        // Validate must not panic (errors are fine).
        let _ = reader.validate();
    }
});
