#![no_main]

use bitnet_gguf::kv::parse_header as parse_kv_header;
use bitnet_gguf::{GgufValueType, check_magic, parse_header, read_version};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz all GGUF parsing entry points with arbitrary / corrupted bytes.
    // None of these should panic; errors are expected.

    // 1. Magic check.
    let _ = check_magic(data);

    // 2. Version read.
    let _ = read_version(data);

    // 3. Top-level header parse.
    let _ = parse_header(data);

    // 4. KV-layer header parse.
    let _ = parse_kv_header(data);

    // 5. GgufValueType discriminant for every 4-byte window.
    for chunk in data.windows(4) {
        let disc = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let _ = GgufValueType::from_u32(disc);
    }

    // 6. Progressively truncated slices to probe off-by-one paths.
    let max_trim = data.len().min(64);
    for trim in 1..max_trim {
        let truncated = &data[..data.len() - trim];
        let _ = parse_header(truncated);
        let _ = parse_kv_header(truncated);
    }

    // 7. Byte-shifted slices to stress alignment handling.
    for offset in 1..data.len().min(8) {
        let shifted = &data[offset..];
        let _ = check_magic(shifted);
        let _ = read_version(shifted);
        let _ = parse_header(shifted);
    }
});
