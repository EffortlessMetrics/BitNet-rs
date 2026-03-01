#![no_main]

use arbitrary::Arbitrary;
use bitnet_gguf::{GgufValueType, check_magic, parse_header, read_version};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct GgufHeaderInput {
    /// Base data that may be valid or corrupted GGUF bytes.
    data: Vec<u8>,
    /// Corruption operations to apply.
    corruptions: Vec<Corruption>,
}

#[derive(Arbitrary, Debug)]
enum Corruption {
    /// Flip a single byte at the given position.
    FlipByte { pos: u16, mask: u8 },
    /// Zero out a range of bytes.
    ZeroRange { start: u16, len: u8 },
    /// Overwrite magic bytes with garbage.
    CorruptMagic { replacement: [u8; 4] },
    /// Set version field to an extreme value.
    CorruptVersion { value: u32 },
    /// Truncate to a specific length.
    Truncate { len: u16 },
    /// Append random bytes.
    Extend { extra: Vec<u8> },
}

/// Build a minimal valid GGUF header for corruption testing.
fn build_valid_gguf_header(n_kv: u64, n_tensors: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF"); // magic
    buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
    buf.extend_from_slice(&n_tensors.to_le_bytes()); // tensor count
    buf.extend_from_slice(&n_kv.to_le_bytes()); // metadata kv count
    buf
}

fn apply_corruptions(data: &mut Vec<u8>, corruptions: &[Corruption]) {
    for corruption in corruptions.iter().take(16) {
        match corruption {
            Corruption::FlipByte { pos, mask } => {
                let idx = *pos as usize;
                if idx < data.len() {
                    data[idx] ^= mask;
                }
            }
            Corruption::ZeroRange { start, len } => {
                let s = *start as usize;
                let l = (*len as usize).min(32);
                for i in s..data.len().min(s + l) {
                    data[i] = 0;
                }
            }
            Corruption::CorruptMagic { replacement } => {
                if data.len() >= 4 {
                    data[..4].copy_from_slice(replacement);
                }
            }
            Corruption::CorruptVersion { value } => {
                if data.len() >= 8 {
                    data[4..8].copy_from_slice(&value.to_le_bytes());
                }
            }
            Corruption::Truncate { len } => {
                let l = *len as usize;
                if l < data.len() {
                    data.truncate(l);
                }
            }
            Corruption::Extend { extra } => {
                data.extend_from_slice(&extra[..extra.len().min(64)]);
            }
        }
    }
}

fuzz_target!(|input: GgufHeaderInput| {
    // --- Phase 1: Parse fully arbitrary bytes ---
    let _ = check_magic(&input.data);
    let _ = read_version(&input.data);
    let _ = parse_header(&input.data);

    // Exercise GgufValueType discriminant.
    for chunk in input.data.windows(4) {
        let disc = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let _ = GgufValueType::from_u32(disc);
    }

    // --- Phase 2: Build valid header then corrupt it ---
    let valid = build_valid_gguf_header(0, 0);

    // Invariant 1: Uncorrupted valid header passes magic check.
    assert!(check_magic(&valid), "valid GGUF header should pass magic check");

    // Invariant 2: Uncorrupted valid header has version 3.
    assert_eq!(read_version(&valid), Some(3), "valid GGUF header should have version 3");

    // Invariant 3: Uncorrupted valid header parses successfully.
    assert!(parse_header(&valid).is_ok(), "valid GGUF header should parse");

    // Now corrupt and ensure no panics.
    let mut corrupted = valid.clone();
    apply_corruptions(&mut corrupted, &input.corruptions);

    let magic_ok = check_magic(&corrupted);
    let version = read_version(&corrupted);
    let parsed = parse_header(&corrupted);

    // Invariant 4: If magic is corrupted, check_magic should fail.
    if corrupted.len() >= 4 && &corrupted[..4] != b"GGUF" {
        assert!(!magic_ok, "corrupted magic should fail check");
    }

    // Invariant 5: If version is corrupted beyond [1,3], read_version returns value or None.
    if let Some(ver) = version {
        // Version is a u32 — no constraint on range, just no panic.
        let _ = ver;
    }

    // Invariant 6: Parse result is either Ok or Err — never panic.
    let _ = parsed;

    // --- Phase 3: Progressive truncation of valid header ---
    for trim in 1..valid.len() {
        let truncated = &valid[..valid.len() - trim];
        let _ = check_magic(truncated);
        let _ = read_version(truncated);
        let _ = parse_header(truncated);
    }

    // --- Phase 4: Various tensor/kv counts ---
    for &n_kv in &[0u64, 1, 100, u64::MAX] {
        for &n_tensors in &[0u64, 1, 1000, u64::MAX] {
            let header = build_valid_gguf_header(n_kv, n_tensors);
            let _ = parse_header(&header);
        }
    }
});
