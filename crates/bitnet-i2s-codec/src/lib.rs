//! Shared I2_S (2-bit ternary) encoding and decoding helpers.
//!
//! Encoding contract:
//! - `0b00 -> 0`
//! - `0b01 -> +1`
//! - `0b11 -> -1`
//! - `0b10` is reserved and decodes to `0`

/// Decode a 2-bit I2_S code to ternary integer form.
#[inline(always)]
pub const fn decode_i2s(bits: u8) -> i8 {
    match bits & 0x03 {
        0b01 => 1,
        0b11 => -1,
        _ => 0,
    }
}

/// Encode a ternary integer `{-1, 0, +1}` into an I2_S 2-bit code.
#[inline(always)]
pub const fn encode_i2s(value: i8) -> u8 {
    match value {
        1 => 0b01,
        -1 => 0b11,
        _ => 0b00,
    }
}

/// Pack four ternary values into one I2_S byte (LSB-first).
#[inline]
pub fn pack_i2s(vals: [i8; 4]) -> u8 {
    let mut byte = 0u8;
    for (i, &value) in vals.iter().enumerate() {
        byte |= encode_i2s(value) << (i * 2);
    }
    byte
}

#[cfg(test)]
mod tests {
    use super::{decode_i2s, encode_i2s, pack_i2s};

    #[test]
    fn decode_mapping_matches_i2s_contract() {
        assert_eq!(decode_i2s(0b00), 0);
        assert_eq!(decode_i2s(0b01), 1);
        assert_eq!(decode_i2s(0b10), 0);
        assert_eq!(decode_i2s(0b11), -1);
    }

    #[test]
    fn encode_mapping_matches_i2s_contract() {
        assert_eq!(encode_i2s(0), 0b00);
        assert_eq!(encode_i2s(1), 0b01);
        assert_eq!(encode_i2s(-1), 0b11);
        assert_eq!(encode_i2s(7), 0b00);
    }

    #[test]
    fn pack_and_decode_roundtrip_for_four_values() {
        let values = [1, -1, 0, 1];
        let packed = pack_i2s(values);
        for (i, expected) in values.into_iter().enumerate() {
            let decoded = decode_i2s((packed >> (i * 2)) & 0x03);
            assert_eq!(decoded, expected);
        }
    }
}
