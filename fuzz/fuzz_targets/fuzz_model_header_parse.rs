#![no_main]

use bitnet_gguf::{GgufValueType, check_magic, parse_header, read_version};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // Raw arbitrary bytes — all helpers must return gracefully.
    let _ = check_magic(data);
    let _ = read_version(data);
    let _ = parse_header(data);

    // Probe GgufValueType discriminant for every 4-byte window.
    for chunk in data.windows(4) {
        let disc = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let _ = GgufValueType::from_u32(disc);
    }

    // Prefix with valid GGUF magic to exercise post-magic code paths.
    if data.len() >= 4 {
        let mut prefixed = b"GGUF".to_vec();
        prefixed.extend_from_slice(data);
        let _ = check_magic(&prefixed);
        let _ = read_version(&prefixed);
        let _ = parse_header(&prefixed);
    }

    // Inject valid magic + version 3 header to reach deeper parsing.
    {
        let mut crafted = Vec::with_capacity(20 + data.len());
        crafted.extend_from_slice(b"GGUF"); // magic
        crafted.extend_from_slice(&3u32.to_le_bytes()); // version
        crafted.extend_from_slice(data);
        let _ = parse_header(&crafted);
    }

    // Progressively truncated slices probe off-by-one paths.
    for trim in 1..data.len().min(64) {
        let _ = parse_header(&data[..data.len() - trim]);
    }
});
