#![no_main]

use bitnet_gguf::{
    GGUF_MAGIC, GGUF_VERSION_MAX, GGUF_VERSION_MIN, GgufValueType, check_magic, parse_header,
    read_version,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // check_magic must not panic on any input.
    let is_magic = check_magic(data);
    if data.len() >= 4 {
        assert_eq!(is_magic, &data[..4] == &GGUF_MAGIC);
    } else {
        assert!(!is_magic, "check_magic true on short data");
    }

    // read_version must not panic.
    let version = read_version(data);
    if !is_magic || data.len() < 8 {
        assert!(version.is_none(), "version should be None for non-GGUF");
    }

    // parse_header must not panic.
    let result = parse_header(data);
    match result {
        Ok(info) => {
            assert!(
                (GGUF_VERSION_MIN..=GGUF_VERSION_MAX).contains(&info.version),
                "parsed version {} out of range",
                info.version
            );
            assert!(info.alignment.is_power_of_two(), "alignment not power of 2");
            // Consistency: read_version must agree.
            assert_eq!(version, Some(info.version), "version mismatch");
        }
        Err(_) => {
            // parse_header failing is fine; the data may be garbage.
        }
    }

    // GgufValueType::from_u32 must not panic for any u32.
    if data.len() >= 4 {
        let raw = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let vt = GgufValueType::from_u32(raw);
        if raw <= 12 {
            assert!(vt.is_some(), "valid discriminant {raw} returned None");
        } else {
            assert!(vt.is_none(), "invalid discriminant {raw} returned Some");
        }
    }

    // Fuzz synthetic GGUF headers: construct valid header bytes with fuzz-derived counts.
    if data.len() >= 16 {
        let tensor_count = u64::from_le_bytes(data[..8].try_into().unwrap_or([0; 8]));
        let metadata_count = u64::from_le_bytes(data[8..16].try_into().unwrap_or([0; 8]));

        for version in GGUF_VERSION_MIN..=GGUF_VERSION_MAX {
            let mut header = Vec::with_capacity(32);
            header.extend_from_slice(&GGUF_MAGIC);
            header.extend_from_slice(&version.to_le_bytes());
            header.extend_from_slice(&tensor_count.to_le_bytes());
            header.extend_from_slice(&metadata_count.to_le_bytes());
            // For v3, add alignment from fuzz data.
            if version >= 3 && data.len() >= 20 {
                let align_raw = u32::from_le_bytes(data[16..20].try_into().unwrap_or([0; 4]));
                header.extend_from_slice(&align_raw.to_le_bytes());
            }

            let parsed = parse_header(&header);
            match parsed {
                Ok(info) => {
                    assert_eq!(info.version, version);
                    assert_eq!(info.tensor_count, tensor_count);
                    assert_eq!(info.metadata_count, metadata_count);
                    assert!(
                        info.alignment.is_power_of_two(),
                        "alignment {} not power of 2",
                        info.alignment
                    );
                }
                Err(e) => {
                    // Only acceptable if the header is genuinely malformed.
                    let _ = e;
                }
            }
        }
    }
});
