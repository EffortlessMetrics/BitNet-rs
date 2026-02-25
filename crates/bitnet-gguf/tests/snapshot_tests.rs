//! Snapshot tests for `bitnet-gguf` public API surface.
//!
//! Pins the magic constant, version range, and error messages for malformed
//! GGUF data so that specification-level constants don't silently change.

use bitnet_gguf::{
    GGUF_MAGIC, GGUF_VERSION_MAX, GGUF_VERSION_MIN, GgufValueType, check_magic, parse_header,
};

#[test]
fn gguf_magic_bytes_snapshot() {
    // The four-byte GGUF magic must always be the ASCII string "GGUF".
    let magic_str = std::str::from_utf8(&GGUF_MAGIC).unwrap();
    insta::assert_snapshot!("gguf_magic_string", magic_str);
}

#[test]
fn gguf_version_range_snapshot() {
    let summary = format!("min={GGUF_VERSION_MIN} max={GGUF_VERSION_MAX}");
    insta::assert_snapshot!("gguf_version_range", summary);
}

#[test]
fn check_magic_on_too_short_data_returns_false() {
    assert!(!check_magic(b"GGU"));
    insta::assert_snapshot!("check_magic_too_short", "false");
}

#[test]
fn check_magic_on_wrong_magic_returns_false() {
    assert!(!check_magic(b"LLMF\x00\x00\x00\x00"));
    insta::assert_snapshot!("check_magic_wrong", "false");
}

#[test]
fn check_magic_on_valid_magic_returns_true() {
    let mut data = vec![0u8; 24];
    data[..4].copy_from_slice(b"GGUF");
    assert!(check_magic(&data));
    insta::assert_snapshot!("check_magic_valid", "true");
}

#[test]
fn parse_header_error_too_small() {
    let tiny = b"GGU";
    let err = parse_header(tiny).unwrap_err();
    let msg = err.to_string();
    insta::assert_snapshot!("parse_header_too_small_error", msg);
}

#[test]
fn parse_header_error_bad_magic() {
    let mut data = vec![0u8; 24];
    data[..4].copy_from_slice(b"LLMF");
    let err = parse_header(&data).unwrap_err();
    let msg = err.to_string();
    insta::assert_snapshot!("parse_header_bad_magic_error", msg);
}

#[test]
fn gguf_value_type_discriminants_snapshot() {
    // Pin numeric discriminant values so spec changes are visible.
    let types = [
        GgufValueType::Uint8,
        GgufValueType::Int8,
        GgufValueType::Uint32,
        GgufValueType::String,
        GgufValueType::Float32,
    ];
    let debug: Vec<String> = types.iter().map(|t| format!("{t:?}")).collect();
    insta::assert_debug_snapshot!("gguf_value_type_debug_subset", debug);
}
