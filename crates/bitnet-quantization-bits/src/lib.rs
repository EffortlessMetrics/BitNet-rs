//! Packed 2-bit quantization codecs.
//!
//! This microcrate owns byte-level packing and unpacking for the shared 2-bit
//! layout used across I2S/TL1/TL2 paths.

/// Packs signed 2-bit values in `[-2, 1]` (clamped) into bytes.
pub fn pack_signed_2bit(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(4));

    for chunk in values.chunks(4) {
        let mut byte = 0u8;
        for (i, &val) in chunk.iter().enumerate() {
            let clamped = val.clamp(-2, 1);
            let unsigned = (clamped + 2) as u8;
            byte |= unsigned << (i * 2);
        }
        packed.push(byte);
    }

    packed
}

/// Unpacks signed 2-bit values from bytes, returning values in `[-2, 1]`.
pub fn unpack_signed_2bit(packed: &[u8], output_len: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(output_len);

    for &byte in packed {
        for i in 0..4 {
            if values.len() >= output_len {
                break;
            }
            let unsigned = (byte >> (i * 2)) & 0x3;
            values.push(unsigned as i8 - 2);
        }
    }

    values
}

/// Packs unsigned 2-bit values into bytes.
///
/// Values are masked with `0x3` and thus remain in `[0, 3]`.
pub fn pack_unsigned_2bit(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(4));

    for chunk in values.chunks(4) {
        let mut byte = 0u8;
        for (i, &val) in chunk.iter().enumerate() {
            byte |= ((val as u8) & 0x3) << (i * 2);
        }
        packed.push(byte);
    }

    packed
}

/// Unpacks unsigned 2-bit values from bytes, returning values in `[0, 3]`.
pub fn unpack_unsigned_2bit(packed: &[u8], output_len: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(output_len);

    for &byte in packed {
        for i in 0..4 {
            if values.len() >= output_len {
                break;
            }
            values.push(((byte >> (i * 2)) & 0x3) as i8);
        }
    }

    values
}

#[cfg(test)]
mod tests {
    use super::{pack_signed_2bit, pack_unsigned_2bit, unpack_signed_2bit, unpack_unsigned_2bit};

    #[test]
    fn signed_roundtrip() {
        let values = vec![-2, -1, 0, 1, -2, 1, 0];
        let packed = pack_signed_2bit(&values);
        let unpacked = unpack_signed_2bit(&packed, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn signed_pack_clamps() {
        let values = vec![-9, -2, 0, 1, 7];
        let packed = pack_signed_2bit(&values);
        let unpacked = unpack_signed_2bit(&packed, values.len());
        assert_eq!(unpacked, vec![-2, -2, 0, 1, 1]);
    }

    #[test]
    fn unsigned_roundtrip() {
        let values = vec![0, 1, 2, 3, 3, 2, 1];
        let packed = pack_unsigned_2bit(&values);
        let unpacked = unpack_unsigned_2bit(&packed, values.len());
        assert_eq!(unpacked, values);
    }
}
