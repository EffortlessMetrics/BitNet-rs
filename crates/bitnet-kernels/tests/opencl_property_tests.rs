//! Property tests for OpenCL kernel invariants.
//!
//! Tests the ternary packing/unpacking logic that mirrors what the OpenCL
//! kernels do on the GPU, verifying round-trip correctness and encoding
//! invariants without requiring hardware.

use proptest::prelude::*;

/// Pack ternary values (-1, 0, +1) into bytes using the I2_S encoding.
///
/// Encoding: +1 → 0b01, −1 → 0b11, 0 → 0b00. Four values per byte.
fn pack_ternary(values: &[i8]) -> Vec<u8> {
    values
        .chunks(4)
        .map(|chunk| {
            let mut packed = 0u8;
            for (i, &v) in chunk.iter().enumerate() {
                let bits = match v {
                    1 => 0b01,
                    -1 => 0b11,
                    _ => 0b00,
                };
                packed |= bits << (i * 2);
            }
            packed
        })
        .collect()
}

/// Unpack bytes back into ternary values using the I2_S encoding.
fn unpack_ternary(packed: &[u8], count: usize) -> Vec<i8> {
    let mut result = Vec::with_capacity(count);
    for &byte in packed {
        for sub in 0..4 {
            if result.len() >= count {
                break;
            }
            let bits = (byte >> (sub * 2)) & 0x03;
            result.push(match bits {
                0x01 => 1,
                0x03 => -1,
                _ => 0,
            });
        }
    }
    result
}

proptest! {
    #[test]
    fn ternary_packing_roundtrips(
        values in prop::collection::vec(
            prop::sample::select(vec![-1i8, 0, 1]), 1..256
        )
    ) {
        let packed = pack_ternary(&values);
        let unpacked = unpack_ternary(&packed, values.len());
        prop_assert_eq!(&values, &unpacked);
    }

    #[test]
    fn packed_bytes_never_exceed_4_values(count in 1usize..1024) {
        let values: Vec<i8> = vec![1; count];
        let packed = pack_ternary(&values);
        prop_assert_eq!(packed.len(), (count + 3) / 4);
    }

    #[test]
    fn zero_values_pack_to_zero_bytes(count in 1usize..256) {
        let values = vec![0i8; count];
        let packed = pack_ternary(&values);
        for &b in &packed {
            prop_assert_eq!(b, 0u8, "all-zero values should pack to zero bytes");
        }
    }

    #[test]
    fn all_ones_pack_correctly(count in 1usize..256) {
        let values = vec![1i8; count];
        let packed = pack_ternary(&values);
        // Each full byte should be 0b01_01_01_01 = 0x55
        for &b in &packed[..packed.len().saturating_sub(1)] {
            if count >= 4 {
                prop_assert_eq!(b, 0x55, "all +1 should pack to 0x55");
            }
        }
    }

    #[test]
    fn all_neg_ones_pack_correctly(count in 1usize..256) {
        let values = vec![-1i8; count];
        let packed = pack_ternary(&values);
        // Each full byte should be 0b11_11_11_11 = 0xFF
        for &b in &packed[..packed.len().saturating_sub(1)] {
            if count >= 4 {
                prop_assert_eq!(b, 0xFF, "all -1 should pack to 0xFF");
            }
        }
    }

    #[test]
    fn single_value_packing(v in prop::sample::select(vec![-1i8, 0, 1])) {
        let packed = pack_ternary(&[v]);
        let expected = match v {
            1 => 0x01,
            -1 => 0x03,
            _ => 0x00,
        };
        prop_assert_eq!(packed, vec![expected]);
    }

    #[test]
    fn packing_is_little_endian_within_byte(
        a in prop::sample::select(vec![-1i8, 0, 1]),
        b in prop::sample::select(vec![-1i8, 0, 1]),
        c in prop::sample::select(vec![-1i8, 0, 1]),
        d in prop::sample::select(vec![-1i8, 0, 1]),
    ) {
        // First value occupies bits [0:1], second [2:3], third [4:5], fourth [6:7]
        let packed = pack_ternary(&[a, b, c, d]);
        let unpacked = unpack_ternary(&packed, 4);
        prop_assert_eq!(unpacked[0], a, "first value at bits [0:1]");
        prop_assert_eq!(unpacked[1], b, "second value at bits [2:3]");
        prop_assert_eq!(unpacked[2], c, "third value at bits [4:5]");
        prop_assert_eq!(unpacked[3], d, "fourth value at bits [6:7]");
    }
}

// === Deterministic unit tests for specific encoding patterns ===

#[test]
fn known_pattern_alternating() {
    // +1, -1, +1, -1 → 0b11_01_11_01 = 0xDD? Let's compute:
    // pos0: +1 = 0b01, pos1: -1 = 0b11, pos2: +1 = 0b01, pos3: -1 = 0b11
    // byte = 0b11_01_11_01 = 0xD5
    let values = vec![1i8, -1, 1, -1];
    let packed = pack_ternary(&values);
    let expected = 0b11_01_11_01u8;
    assert_eq!(packed, vec![expected], "alternating +1/-1 pattern");
}

#[test]
fn known_pattern_zero_one_negone_zero() {
    // 0, +1, -1, 0 → 0b00_11_01_00 = 0x0C + 0x04 = ...
    // pos0: 0 = 0b00 << 0 = 0x00
    // pos1: +1 = 0b01 << 2 = 0x04
    // pos2: -1 = 0b11 << 4 = 0x30
    // pos3: 0 = 0b00 << 6 = 0x00
    // byte = 0x34
    let values = vec![0i8, 1, -1, 0];
    let packed = pack_ternary(&values);
    let expected = 0x00 | 0x04 | 0x30 | 0x00;
    assert_eq!(packed, vec![expected]);
    let unpacked = unpack_ternary(&packed, 4);
    assert_eq!(unpacked, values);
}

#[test]
fn partial_last_byte_pads_with_zeros() {
    // 5 values: first 4 fill byte 0, fifth goes into byte 1 (with 3 zero-pads)
    let values = vec![1i8, 1, 1, 1, -1];
    let packed = pack_ternary(&values);
    assert_eq!(packed.len(), 2);
    assert_eq!(packed[0], 0x55); // four +1s
    assert_eq!(packed[1], 0x03); // one -1 in lowest 2 bits
    let unpacked = unpack_ternary(&packed, 5);
    assert_eq!(unpacked, values);
}
