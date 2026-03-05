#![no_main]
use arbitrary::Arbitrary;
use bitnet_gguf::{GGUF_VERSION_MAX, GGUF_VERSION_MIN, check_magic, parse_header, read_version};
use libfuzzer_sys::fuzz_target;

/// Structured input covering both v2 and v3 header shapes.
#[derive(Arbitrary, Debug)]
struct HeaderInput {
    /// Raw version selector — reduced to a valid version inside the target.
    version_raw: u32,
    tensor_count: u64,
    metadata_count: u64,
    /// Alignment value written into v3 headers.
    alignment_raw: u32,
}

fuzz_target!(|input: HeaderInput| {
    // Restrict to the two versions parse_header accepts (2 and 3).
    let version: u32 = if input.version_raw.is_multiple_of(2) { 2 } else { 3 };

    // --- Serialize -----------------------------------------------------------
    let mut buf: Vec<u8> = Vec::with_capacity(32);
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&version.to_le_bytes());
    buf.extend_from_slice(&input.tensor_count.to_le_bytes());
    buf.extend_from_slice(&input.metadata_count.to_le_bytes());

    // v3 appends a u32 alignment field; ensure it is non-zero.
    let alignment_field = input.alignment_raw.max(1);
    if version >= 3 {
        buf.extend_from_slice(&alignment_field.to_le_bytes());
    }

    // --- Cross-check low-level helpers before the round-trip ----------------
    assert!(check_magic(&buf), "synthesized buffer must start with valid GGUF magic");
    assert_eq!(
        read_version(&buf),
        Some(version),
        "read_version must return the version we encoded"
    );

    // --- Round-trip: parse back the header we just built --------------------
    let info = match parse_header(&buf) {
        Ok(h) => h,
        Err(e) => panic!("parse_header failed on valid synthesized header: {e}"),
    };

    // Field identity: every value we wrote must come back unchanged.
    assert_eq!(info.version, version);
    assert_eq!(info.tensor_count, input.tensor_count);
    assert_eq!(info.metadata_count, input.metadata_count);

    // --- Invariants on the parsed output ------------------------------------
    // Alignment must always be a power of two.
    assert!(
        info.alignment.is_power_of_two(),
        "parsed alignment {} must be a power of two",
        info.alignment
    );
    // Version must be within the supported range.
    assert!(
        (GGUF_VERSION_MIN..=GGUF_VERSION_MAX).contains(&info.version),
        "parsed version {} outside supported range",
        info.version
    );
    // v2 alignment always defaults to 32.
    if version == 2 {
        assert_eq!(
            info.alignment, 32,
            "v2 alignment must be the default 32, got {}",
            info.alignment
        );
    }
    // v3 with a power-of-two alignment field: parser must preserve it.
    if version >= 3 && alignment_field.is_power_of_two() {
        assert_eq!(
            info.alignment, alignment_field,
            "v3 power-of-two alignment {alignment_field} must round-trip through parse_header"
        );
    }

    // --- Prefix stress: parse_header must not panic on any leading sub-slice -
    for end in [4usize, 8, 16, 24, buf.len()] {
        if end <= buf.len() {
            let _ = parse_header(&buf[..end]);
        }
    }
});
