//! Wave 5 property tests for `bitnet-models`.
//!
//! These tests cover invariants not already addressed in `model_proptests.rs`:
//!
//! 1. **GGUF magic validation** – `check_magic` rejects non-GGUF headers.
//! 2. **GGUF version parsing** – `read_version` returns `None` for bad magic.
//! 3. **GGUF parse_header rejects truncated data** – anything < 24 bytes fails.
//! 4. **GGUF parse_header round-trip** – valid synthetic headers parse correctly.
//! 5. **Format detection determinism** – same path always yields same format.
//! 6. **GgufReader rejects undersized data** – data < 16 bytes always fails.

#![cfg(all(test, feature = "cpu"))]

use bitnet_gguf::{GGUF_MAGIC, check_magic, parse_header, read_version};
use bitnet_models::formats::ModelFormat;
use proptest::prelude::*;
use std::path::Path;

// ── GGUF magic validation ───────────────────────────────────────────────────

proptest! {
    /// `check_magic` returns false for any 4-byte prefix that isn't "GGUF".
    #[test]
    fn prop_check_magic_rejects_non_gguf(
        b0 in any::<u8>(),
        b1 in any::<u8>(),
        b2 in any::<u8>(),
        b3 in any::<u8>(),
    ) {
        let bytes = [b0, b1, b2, b3, 0, 0, 0, 0];
        let is_gguf = &bytes[..4] == b"GGUF";
        prop_assert_eq!(
            check_magic(&bytes),
            is_gguf,
            "check_magic({:?}) should be {} for {:?}",
            &bytes[..4], is_gguf, &bytes[..4]
        );
    }

    /// `check_magic` always returns true when the first 4 bytes are "GGUF".
    #[test]
    fn prop_check_magic_accepts_gguf_prefix(
        tail in prop::collection::vec(any::<u8>(), 0..64),
    ) {
        let mut data = b"GGUF".to_vec();
        data.extend_from_slice(&tail);
        prop_assert!(
            check_magic(&data),
            "check_magic must accept data starting with GGUF magic"
        );
    }
}

// ── GGUF version parsing ────────────────────────────────────────────────────

proptest! {
    /// `read_version` returns `None` when magic is absent.
    #[test]
    fn prop_read_version_none_without_magic(
        garbage in prop::collection::vec(any::<u8>(), 8..32),
    ) {
        // Ensure the first 4 bytes are NOT "GGUF".
        let mut data = garbage;
        if &data[..4] == b"GGUF" {
            data[0] ^= 0xFF;
        }
        prop_assert_eq!(
            read_version(&data),
            None,
            "read_version must return None for non-GGUF data"
        );
    }

    /// `read_version` returns `Some(v)` for valid magic + any version u32.
    #[test]
    fn prop_read_version_extracts_le_u32(version in any::<u32>()) {
        let mut data = b"GGUF".to_vec();
        data.extend_from_slice(&version.to_le_bytes());
        let result = read_version(&data);
        prop_assert_eq!(
            result,
            Some(version),
            "read_version should extract version {}", version
        );
    }
}

// ── GGUF parse_header rejects truncated data ────────────────────────────────

proptest! {
    /// `parse_header` always fails for data shorter than 24 bytes.
    #[test]
    fn prop_parse_header_rejects_short_data(
        data in prop::collection::vec(any::<u8>(), 0..24),
    ) {
        let result = parse_header(&data);
        prop_assert!(
            result.is_err(),
            "parse_header must reject data of length {} (< 24); got {:?}",
            data.len(), result
        );
    }
}

// ── GGUF parse_header round-trip for synthetic headers ──────────────────────

proptest! {
    /// A well-formed synthetic v2 header always parses successfully and
    /// the parsed fields match the inputs.
    #[test]
    fn prop_parse_header_roundtrip_v2(
        tensor_count in 0u64..10_000,
        metadata_count in 0u64..10_000,
    ) {
        let mut data = Vec::with_capacity(32);
        data.extend_from_slice(&GGUF_MAGIC);
        data.extend_from_slice(&2u32.to_le_bytes()); // version 2
        data.extend_from_slice(&tensor_count.to_le_bytes());
        data.extend_from_slice(&metadata_count.to_le_bytes());

        let info = parse_header(&data).expect("valid v2 header must parse");
        prop_assert_eq!(info.version, 2);
        prop_assert_eq!(info.tensor_count, tensor_count);
        prop_assert_eq!(info.metadata_count, metadata_count);
        prop_assert_eq!(info.alignment, 32, "v2 default alignment must be 32");
    }

    /// A well-formed synthetic v3 header with power-of-two alignment parses
    /// correctly and preserves the alignment field.
    #[test]
    fn prop_parse_header_roundtrip_v3(
        tensor_count in 0u64..10_000,
        metadata_count in 0u64..10_000,
        log2_align in 0u32..=12u32,
    ) {
        let alignment = 1u32 << log2_align;
        let mut data = Vec::with_capacity(32);
        data.extend_from_slice(&GGUF_MAGIC);
        data.extend_from_slice(&3u32.to_le_bytes()); // version 3
        data.extend_from_slice(&tensor_count.to_le_bytes());
        data.extend_from_slice(&metadata_count.to_le_bytes());
        data.extend_from_slice(&alignment.to_le_bytes());

        let info = parse_header(&data).expect("valid v3 header must parse");
        prop_assert_eq!(info.version, 3);
        prop_assert_eq!(info.tensor_count, tensor_count);
        prop_assert_eq!(info.metadata_count, metadata_count);
        prop_assert_eq!(info.alignment, alignment);
    }
}

// ── Format detection determinism ────────────────────────────────────────────

proptest! {
    /// `ModelFormat::detect_from_path` is deterministic: calling it twice on
    /// the same path yields the same result.
    #[test]
    fn prop_format_detection_deterministic(
        stem in "[a-zA-Z0-9_]{1,16}",
        ext in prop_oneof![Just("gguf"), Just("safetensors")],
    ) {
        let path = format!("/tmp/{}.{}", stem, ext);
        let r1 = ModelFormat::detect_from_path(Path::new(&path));
        let r2 = ModelFormat::detect_from_path(Path::new(&path));
        match (&r1, &r2) {
            (Ok(a), Ok(b)) => prop_assert_eq!(
                std::mem::discriminant(a),
                std::mem::discriminant(b),
                "format detection must be deterministic for '{}'", path
            ),
            (Err(_), Err(_)) => {} // both errors is fine
            _ => prop_assert!(false, "mismatched Ok/Err for '{}': {:?} vs {:?}", path, r1, r2),
        }
    }
}

// ── GgufReader rejects undersized data ──────────────────────────────────────

proptest! {
    /// `GgufReader::new` always fails for data shorter than 16 bytes.
    #[test]
    fn prop_gguf_reader_rejects_undersized(
        data in prop::collection::vec(any::<u8>(), 0..16),
    ) {
        let result = bitnet_models::GgufReader::new(&data);
        prop_assert!(
            result.is_err(),
            "GgufReader must reject data of length {} (< 16); got Ok",
            data.len()
        );
    }
}
