//! Weight packing/unpacking utilities for quantized formats.
//!
//! Pack and unpack int2/int4/int8 weights into dense byte arrays.

/// Pack 4 int2 values (0..3) into one byte, LSB first.
pub fn pack_int2(values: &[u8]) -> Vec<u8> {
    let out_len = (values.len() + 3) / 4;
    let mut packed = vec![0u8; out_len];
    for (i, &v) in values.iter().enumerate() {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        packed[byte_idx] |= (v & 0x03) << bit_offset;
    }
    packed
}

/// Unpack int2 values from packed bytes. Returns exactly `count` values.
pub fn unpack_int2(packed: &[u8], count: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        if byte_idx < packed.len() {
            values.push((packed[byte_idx] >> bit_offset) & 0x03);
        } else {
            values.push(0);
        }
    }
    values
}

/// Pack 2 int4 values (0..15) into one byte (low nibble first).
pub fn pack_int4(values: &[u8]) -> Vec<u8> {
    let out_len = (values.len() + 1) / 2;
    let mut packed = vec![0u8; out_len];
    for (i, &v) in values.iter().enumerate() {
        let byte_idx = i / 2;
        if i % 2 == 0 {
            packed[byte_idx] |= v & 0x0F;
        } else {
            packed[byte_idx] |= (v & 0x0F) << 4;
        }
    }
    packed
}

/// Unpack int4 values from packed bytes. Returns exactly `count` values.
pub fn unpack_int4(packed: &[u8], count: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let byte_idx = i / 2;
        if byte_idx < packed.len() {
            if i % 2 == 0 {
                values.push(packed[byte_idx] & 0x0F);
            } else {
                values.push((packed[byte_idx] >> 4) & 0x0F);
            }
        } else {
            values.push(0);
        }
    }
    values
}

/// Pack int8 values (identity — provided for API symmetry).
pub fn pack_int8(values: &[u8]) -> Vec<u8> {
    values.to_vec()
}

/// Unpack int8 values (identity — provided for API symmetry).
pub fn unpack_int8(packed: &[u8], count: usize) -> Vec<u8> {
    let mut out = packed[..count.min(packed.len())].to_vec();
    out.resize(count, 0);
    out
}

/// Compute packed size in bytes for a given element count and bit width.
pub fn packed_size(element_count: usize, bits: u8) -> usize {
    match bits {
        2 => (element_count + 3) / 4,
        4 => (element_count + 1) / 2,
        8 => element_count,
        _ => panic!("unsupported bit width: {bits}"),
    }
}

/// Pack signed int2 values (-1, 0, +1) using the I2_S encoding (0b00=0, 0b01=+1, 0b11=-1).
pub fn pack_signed_int2(values: &[i8]) -> Vec<u8> {
    let encoded: Vec<u8> = values
        .iter()
        .map(|&v| match v {
            0 => 0b00,
            1 => 0b01,
            -1 => 0b11,
            _ => 0b00, // treat out-of-range as 0
        })
        .collect();
    pack_int2(&encoded)
}

/// Unpack signed int2 values from I2_S encoding.
pub fn unpack_signed_int2(packed: &[u8], count: usize) -> Vec<i8> {
    let raw = unpack_int2(packed, count);
    raw.iter()
        .map(|&v| match v {
            0b00 => 0,
            0b01 => 1,
            0b11 => -1,
            _ => 0, // 0b10 is unused, treat as 0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_int2() {
        let values = vec![0, 1, 2, 3, 1, 0];
        let packed = pack_int2(&values);
        let unpacked = unpack_int2(&packed, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_pack_unpack_int4() {
        let values = vec![0, 5, 10, 15, 7];
        let packed = pack_int4(&values);
        let unpacked = unpack_int4(&packed, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_pack_unpack_int8() {
        let values = vec![0, 128, 255, 42];
        let packed = pack_int8(&values);
        let unpacked = unpack_int8(&packed, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_packed_size() {
        assert_eq!(packed_size(8, 2), 2);
        assert_eq!(packed_size(8, 4), 4);
        assert_eq!(packed_size(8, 8), 8);
        assert_eq!(packed_size(7, 2), 2);
        assert_eq!(packed_size(5, 4), 3);
    }

    #[test]
    fn test_signed_int2_round_trip() {
        let values = vec![0, 1, -1, 0, 1, -1, -1, 1];
        let packed = pack_signed_int2(&values);
        let unpacked = unpack_signed_int2(&packed, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_signed_int2_encoding() {
        // Single value packing checks
        let packed = pack_signed_int2(&[1]);
        assert_eq!(packed, vec![0b01]);
        let packed = pack_signed_int2(&[-1]);
        assert_eq!(packed, vec![0b11]);
        let packed = pack_signed_int2(&[0]);
        assert_eq!(packed, vec![0b00]);
    }

    #[test]
    fn test_empty_inputs() {
        assert!(pack_int2(&[]).is_empty());
        assert!(pack_int4(&[]).is_empty());
        assert!(pack_int8(&[]).is_empty());
        assert!(unpack_int2(&[], 0).is_empty());
        assert!(unpack_int4(&[], 0).is_empty());
        assert!(unpack_int8(&[], 0).is_empty());
    }

    #[test]
    fn test_int2_single_byte() {
        let values = vec![3, 2, 1, 0];
        let packed = pack_int2(&values);
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_int2(&packed, 4);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_int4_single_byte() {
        let values = vec![0x0A, 0x05];
        let packed = pack_int4(&values);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0x5A);
        let unpacked = unpack_int4(&packed, 2);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_odd_count_int2() {
        let values = vec![1, 2, 3];
        let packed = pack_int2(&values);
        let unpacked = unpack_int2(&packed, 3);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_odd_count_int4() {
        let values = vec![7, 11, 3];
        let packed = pack_int4(&values);
        let unpacked = unpack_int4(&packed, 3);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_large_round_trip_int2() {
        let values: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
        let packed = pack_int2(&values);
        let unpacked = unpack_int2(&packed, values.len());
        assert_eq!(unpacked, values);
    }
}
