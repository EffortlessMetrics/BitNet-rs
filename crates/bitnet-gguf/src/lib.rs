//! Lightweight GGUF file-format types and header parser.
//!
//! Provides a minimal, dependency-light library for inspecting GGUF files:
//! magic validation, version detection, and header field extraction.
//!
//! The **full** GGUF parser (metadata key-value pairs, tensor data) lives in
//! `bitnet-models`; this crate focuses on the portable, zero-copy subset.
//!
//! # Example
//!
//! ```no_run
//! use bitnet_gguf::{check_magic, read_version, parse_header};
//! use std::fs;
//!
//! let data = fs::read("model.gguf").unwrap();
//! if check_magic(&data) {
//!     let version = read_version(&data).unwrap();
//!     println!("GGUF v{version}");
//!     let header = parse_header(&data).unwrap();
//!     println!("{header:?}");
//! }
//! ```

use std::path::Path;

pub mod kv;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The four-byte magic that every GGUF file starts with.
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
/// Minimum supported GGUF version.
pub const GGUF_VERSION_MIN: u32 = 2;
/// Maximum supported GGUF version (inclusive).
pub const GGUF_VERSION_MAX: u32 = 3;

// ---------------------------------------------------------------------------
// Value-type discriminant
// ---------------------------------------------------------------------------

/// Discriminant tag for GGUF metadata values.
///
/// Numeric values match those in the GGUF specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[non_exhaustive]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    /// Convert from the raw u32 discriminant in the file.
    pub const fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Lightweight value
// ---------------------------------------------------------------------------

/// A parsed GGUF metadata value.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(GgufValueType, Vec<Self>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

// ---------------------------------------------------------------------------
// Metadata KV
// ---------------------------------------------------------------------------

/// A single key-value metadata entry from a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufMetadataKv {
    pub key: String,
    pub value: GgufValue,
}

// ---------------------------------------------------------------------------
// Tensor info
// ---------------------------------------------------------------------------

/// Lightweight tensor descriptor from the GGUF tensor index.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    /// Raw GGML dtype discriminant.
    pub dtype: u32,
    /// Byte offset into the tensor-data section.
    pub offset: u64,
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// Parsed GGUF file header (magic + version + counts + v3 fields).
#[derive(Debug, Clone)]
pub struct GgufFileInfo {
    /// GGUF format version (2 or 3).
    pub version: u32,
    /// Number of tensors described in the index.
    pub tensor_count: u64,
    /// Number of metadata key-value entries.
    pub metadata_count: u64,
    /// Data alignment in bytes (defaults to 32 for v2, read from file for v3).
    pub alignment: u32,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Returns `true` if `data` starts with the GGUF magic bytes.
#[inline]
pub fn check_magic(data: &[u8]) -> bool {
    data.get(0..4) == Some(b"GGUF")
}

/// Read the GGUF version field from the first 8 bytes.
///
/// Returns `None` if the slice is too short or the magic is invalid.
pub fn read_version(data: &[u8]) -> Option<u32> {
    if data.len() < 8 || !check_magic(data) {
        return None;
    }
    Some(u32::from_le_bytes([data[4], data[5], data[6], data[7]]))
}

/// Parse only the GGUF file header without reading metadata or tensors.
///
/// Validates the magic, version, and returns a [`GgufFileInfo`].
pub fn parse_header(data: &[u8]) -> anyhow::Result<GgufFileInfo> {
    anyhow::ensure!(data.len() >= 24, "file too small for a GGUF header");
    anyhow::ensure!(check_magic(data), "invalid GGUF magic (expected 'GGUF')");

    let version = u32::from_le_bytes(data[4..8].try_into()?);
    anyhow::ensure!(
        (GGUF_VERSION_MIN..=GGUF_VERSION_MAX).contains(&version),
        "unsupported GGUF version {version} (supported: {GGUF_VERSION_MIN}–{GGUF_VERSION_MAX})"
    );

    let tensor_count = u64::from_le_bytes(data[8..16].try_into()?);
    let metadata_count = u64::from_le_bytes(data[16..24].try_into()?);

    // v3 adds alignment (u32) at byte 24.
    let alignment = if version >= 3 && data.len() >= 28 {
        let a = u32::from_le_bytes(data[24..28].try_into()?);
        if a.is_power_of_two() { a } else { 32 }
    } else {
        32
    };

    Ok(GgufFileInfo { version, tensor_count, metadata_count, alignment })
}

/// Memory-map a GGUF file and parse its header.
///
/// The mapping is released when the returned value is dropped.
pub fn open(path: &Path) -> anyhow::Result<GgufFileInfo> {
    let file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("cannot open {}: {e}", path.display()))?;
    // SAFETY: we do not mutate the mapping and the file is opened read-only.
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| anyhow::anyhow!("mmap failed for {}: {e}", path.display()))?;
    parse_header(&mmap[..])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn gguf_v2_header() -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(b"GGUF"); // magic
        d.extend_from_slice(&2u32.to_le_bytes()); // version
        d.extend_from_slice(&3u64.to_le_bytes()); // tensor_count
        d.extend_from_slice(&5u64.to_le_bytes()); // metadata_count
        d
    }

    fn gguf_v3_header() -> Vec<u8> {
        let mut d = gguf_v2_header();
        // Replace version with 3.
        d[4..8].copy_from_slice(&3u32.to_le_bytes());
        // Add alignment = 64.
        d.extend_from_slice(&64u32.to_le_bytes());
        // Add data_offset = 0 (early-v3 variant).
        d.extend_from_slice(&0u64.to_le_bytes());
        d
    }

    #[test]
    fn check_magic_valid() {
        let data = b"GGUFextra";
        assert!(check_magic(data));
    }

    #[test]
    fn check_magic_invalid() {
        assert!(!check_magic(b"GGML"));
        assert!(!check_magic(b""));
        assert!(!check_magic(b"GGU"));
    }

    #[test]
    fn read_version_v2() {
        let data = gguf_v2_header();
        assert_eq!(read_version(&data), Some(2));
    }

    #[test]
    fn parse_header_v2() {
        let data = gguf_v2_header();
        let info = parse_header(&data).unwrap();
        assert_eq!(info.version, 2);
        assert_eq!(info.tensor_count, 3);
        assert_eq!(info.metadata_count, 5);
        assert_eq!(info.alignment, 32); // default for v2
    }

    #[test]
    fn parse_header_v3_alignment() {
        let data = gguf_v3_header();
        let info = parse_header(&data).unwrap();
        assert_eq!(info.version, 3);
        assert_eq!(info.alignment, 64);
    }

    #[test]
    fn parse_header_rejects_invalid_magic() {
        let mut data = gguf_v2_header();
        data[0] = b'X'; // corrupt magic
        assert!(parse_header(&data).is_err());
    }

    #[test]
    fn parse_header_rejects_too_small() {
        let data = b"GGUF\x02\x00\x00\x00"; // only 8 bytes
        assert!(parse_header(data).is_err());
    }

    #[test]
    fn gguf_value_type_roundtrip() {
        for n in 0u32..=12 {
            let vt = GgufValueType::from_u32(n);
            assert!(vt.is_some(), "missing variant for {n}");
        }
        assert!(GgufValueType::from_u32(99).is_none());
    }

    #[test]
    fn parse_header_rejects_version_1() {
        let mut data = gguf_v2_header();
        data[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert!(parse_header(&data).is_err(), "version 1 must be rejected");
    }

    #[test]
    fn parse_header_rejects_version_0() {
        let mut data = gguf_v2_header();
        data[4..8].copy_from_slice(&0u32.to_le_bytes());
        assert!(parse_header(&data).is_err(), "version 0 must be rejected");
    }

    #[test]
    fn parse_header_rejects_version_4() {
        let mut data = gguf_v2_header();
        data[4..8].copy_from_slice(&4u32.to_le_bytes());
        assert!(parse_header(&data).is_err(), "version 4 must be rejected");
    }

    #[test]
    fn parse_header_v3_non_power_of_two_alignment_falls_back() {
        // Build a v3 header with alignment = 7 (not a power of two).
        let mut data = gguf_v2_header();
        data[4..8].copy_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&7u32.to_le_bytes()); // non-power-of-two alignment
        let info = parse_header(&data).unwrap();
        assert_eq!(info.alignment, 32, "non-power-of-two alignment must fall back to 32");
    }

    #[test]
    fn read_version_returns_none_for_too_short() {
        assert_eq!(read_version(b"GGUF\x02\x00\x00"), None); // 7 bytes, one short
        assert_eq!(read_version(b""), None);
    }

    #[test]
    fn read_version_returns_none_for_bad_magic() {
        // 8 bytes, valid length, but wrong magic
        assert_eq!(read_version(b"LLMF\x02\x00\x00\x00"), None);
    }

    // --- proptest -----------------------------------------------------------

    proptest::proptest! {
        #[test]
        fn check_magic_is_true_only_for_gguf_prefix(data in proptest::collection::vec(0u8..=255, 4..32)) {
            let is_gguf = data.starts_with(b"GGUF");
            proptest::prop_assert_eq!(check_magic(&data), is_gguf);
        }

        #[test]
        fn parse_header_never_panics_on_arbitrary_bytes(
            data in proptest::collection::vec(0u8..=255, 0..64)
        ) {
            // Must not panic, regardless of the input.
            let _ = parse_header(&data);
        }

        #[test]
        fn parse_header_rejects_wrong_magic(
            // First 4 bytes are NOT "GGUF".
            b0 in 0u8..=255u8,
            b1 in 0u8..=255u8,
            b2 in 0u8..=255u8,
            b3 in 0u8..=255u8,
            rest in proptest::collection::vec(0u8..=255, 20..40),
        ) {
            let mut data = vec![b0, b1, b2, b3];
            data.extend_from_slice(&rest);
            // If the first 4 bytes happen to be "GGUF" the predicate may pass
            // – that's fine; we only assert the invariant when magic is wrong.
            if &data[0..4] != b"GGUF" {
                proptest::prop_assert!(parse_header(&data).is_err());
            }
        }

        #[test]
        fn valid_v2_header_always_parses(
            tensor_count in 0u64..1_000_000,
            metadata_count in 0u64..1_000_000,
        ) {
            let mut d = Vec::new();
            d.extend_from_slice(b"GGUF");
            d.extend_from_slice(&2u32.to_le_bytes());
            d.extend_from_slice(&tensor_count.to_le_bytes());
            d.extend_from_slice(&metadata_count.to_le_bytes());
            let info = parse_header(&d).expect("valid v2 header must parse");
            proptest::prop_assert_eq!(info.version, 2);
            proptest::prop_assert_eq!(info.tensor_count, tensor_count);
            proptest::prop_assert_eq!(info.metadata_count, metadata_count);
            proptest::prop_assert_eq!(info.alignment, 32); // v2 default
        }
    }
}
