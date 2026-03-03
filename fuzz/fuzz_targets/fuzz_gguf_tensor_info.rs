#![no_main]

use arbitrary::Arbitrary;
use bitnet_gguf::{GgufValueType, TensorInfo, check_magic, parse_header, read_version};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct GgufTensorInfoInput {
    raw_data: Vec<u8>,
    n_dims: u8,
    dims: Vec<u32>,
    dtype: u32,
    offset: u64,
    name_bytes: Vec<u8>,
    version: u8,
    tensor_count: u8,
    metadata_count: u8,
    inject_valid_header: bool,
}

/// Build a minimal valid GGUF v2 byte stream with tensor info entries.
fn build_gguf_bytes(version: u32, tensor_count: u64, metadata_count: u64, extra: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF"); // magic
    buf.extend_from_slice(&version.to_le_bytes()); // version
    buf.extend_from_slice(&tensor_count.to_le_bytes()); // tensor_count
    buf.extend_from_slice(&metadata_count.to_le_bytes()); // metadata_count
    if version >= 3 {
        buf.extend_from_slice(&32u32.to_le_bytes()); // alignment
    }
    buf.extend_from_slice(extra);
    buf
}

fuzz_target!(|input: GgufTensorInfoInput| {
    // --- Test 1: Raw arbitrary bytes must not panic in parsers ---
    let _ = check_magic(&input.raw_data);
    let _ = read_version(&input.raw_data);
    let _ = parse_header(&input.raw_data);

    // Exercise GgufValueType discriminant for every 4-byte window
    for chunk in input.raw_data.windows(4) {
        let disc = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let _ = GgufValueType::from_u32(disc);
    }

    // --- Test 2: Progressively truncated slices ---
    for trim in 1..input.raw_data.len().min(32) {
        let _ = parse_header(&input.raw_data[..input.raw_data.len() - trim]);
    }

    // --- Test 3: Constructed valid headers ---
    if input.inject_valid_header {
        let version = (input.version as u32 % 2) + 2; // v2 or v3
        let tc = input.tensor_count as u64;
        let mc = input.metadata_count as u64;

        let buf = build_gguf_bytes(version, tc, mc, &input.raw_data);

        // Header parsing should succeed for well-formed data
        if let Ok(info) = parse_header(&buf) {
            assert_eq!(info.version, version);
            assert_eq!(info.tensor_count, tc);
            assert_eq!(info.metadata_count, mc);
            if version >= 3 {
                assert!(info.alignment.is_power_of_two());
            }
        }

        // check_magic should return true
        assert!(check_magic(&buf));

        // read_version should return the correct version
        if let Some(v) = read_version(&buf) {
            assert_eq!(v, version);
        }
    }

    // --- Test 4: TensorInfo struct construction with fuzz data ---
    let name = String::from_utf8_lossy(&input.name_bytes[..input.name_bytes.len().min(128)]);
    let n_dims = (input.n_dims as u32 % 5) + 1; // 1..5 dimensions
    let dims: Vec<u64> = input.dims.iter().take(n_dims as usize).map(|&d| d as u64).collect();

    let tensor_info = TensorInfo {
        name: name.into_owned(),
        n_dims,
        dims: dims.clone(),
        dtype: input.dtype,
        offset: input.offset,
    };

    // Invariant 1: TensorInfo fields are preserved
    assert_eq!(tensor_info.n_dims, n_dims);
    assert_eq!(tensor_info.dims.len(), dims.len());
    assert_eq!(tensor_info.dtype, input.dtype);
    assert_eq!(tensor_info.offset, input.offset);

    // Invariant 2: Debug formatting must not panic
    let debug = format!("{:?}", tensor_info);
    assert!(!debug.is_empty());

    // Invariant 3: Clone must produce identical value
    let cloned = tensor_info.clone();
    assert_eq!(cloned.name, tensor_info.name);
    assert_eq!(cloned.n_dims, tensor_info.n_dims);
    assert_eq!(cloned.dims, tensor_info.dims);
    assert_eq!(cloned.dtype, tensor_info.dtype);
    assert_eq!(cloned.offset, tensor_info.offset);

    // --- Test 5: GgufValueType exhaustive check ---
    for disc in 0..=20 {
        let vt = GgufValueType::from_u32(disc);
        if disc <= 12 {
            assert!(vt.is_some(), "valid discriminant {disc} should be Some");
        } else {
            assert!(vt.is_none(), "invalid discriminant {disc} should be None");
        }
    }

    // --- Test 6: Corrupt magic ---
    let mut corrupt = build_gguf_bytes(2, 0, 0, &[]);
    if !corrupt.is_empty() {
        corrupt[0] ^= 0xFF; // Flip first byte
        assert!(!check_magic(&corrupt), "corrupted magic should fail");
        assert!(read_version(&corrupt).is_none(), "corrupted magic version should fail");
        assert!(parse_header(&corrupt).is_err(), "corrupted magic parse should fail");
    }

    // --- Test 7: Empty and minimal slices ---
    let _ = parse_header(&[]);
    let _ = parse_header(&[b'G']);
    let _ = parse_header(b"GGUF");
    let _ = parse_header(&[0u8; 24]);
    let _ = read_version(&[]);
    let _ = check_magic(&[]);
});
