#![no_main]

use bitnet_gguf::{GgufValueType, check_magic, parse_header, read_version};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // check_magic must never panic.
    let _has_magic = check_magic(data);

    // read_version must never panic.
    let _version = read_version(data);

    // parse_header must never panic, regardless of input.
    let _ = parse_header(data);

    // GgufValueType::from_u32 must never panic on any u32 value derived
    // from the first 4 bytes (when present).
    if data.len() >= 4 {
        let discriminant = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let _ = GgufValueType::from_u32(discriminant);
    }
});
