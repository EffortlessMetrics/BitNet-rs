#![no_main]

use libfuzzer_sys::fuzz_target;
use safetensors::SafeTensors;

fuzz_target!(|data: &[u8]| {
    // Fuzz the SafeTensors header/metadata parser.
    // `read_metadata` parses only the 8-byte length prefix and the JSON
    // header; it does not read tensor data.  Any Err result is acceptable —
    // only panics constitute failures.
    let _ = SafeTensors::read_metadata(data);

    // Also exercise the manual header-field path used in `detect_format`:
    // read the 8-byte little-endian header-length, then attempt to parse
    // that slice as JSON.  This mirrors the exact code path in the loader.
    if data.len() >= 8 {
        let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        if header_len > 0
            && header_len < 1024 * 1024
            && let Some(header_data) = data.get(8..8 + header_len)
        {
            let _ = serde_json::from_slice::<serde_json::Value>(header_data);
        }
    }
});
