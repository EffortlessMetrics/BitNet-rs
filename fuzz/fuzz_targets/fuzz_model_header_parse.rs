#![no_main]

use bitnet_gguf::{GgufValueType, check_magic, parse_header, read_version};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Header parsing must handle arbitrary bytes without panicking.
    let _ = check_magic(data);
    let _ = read_version(data);
    let _ = parse_header(data);

    // Exercise GgufValueType discriminant for every 4-byte window.
    for chunk in data.windows(4) {
        let disc = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let _ = GgufValueType::from_u32(disc);
    }

    // Parse with progressively truncated slices to probe boundary conditions.
    for trim in 1..data.len().min(64) {
        let _ = parse_header(&data[..data.len() - trim]);
    }

    // Parse with progressively extended offsets.
    for offset in 1..data.len().min(16) {
        let _ = parse_header(&data[offset..]);
    }

    // Single-byte and empty inputs must not panic.
    let _ = parse_header(&[]);
    for &b in data.iter().take(4) {
        let _ = parse_header(&[b]);
    }
});
