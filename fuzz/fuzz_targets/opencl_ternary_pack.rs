#![no_main]

use libfuzzer_sys::fuzz_target;

fn pack_ternary(values: &[i8]) -> Vec<u8> {
    values
        .chunks(4)
        .map(|chunk| {
            let mut packed = 0u8;
            for (i, &v) in chunk.iter().enumerate() {
                let bits: u8 = match v {
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

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > 1024 {
        return;
    }

    // Strategy 1: Treat input as ternary values, pack and unpack
    let ternary_values: Vec<i8> = data
        .iter()
        .take(256)
        .map(|&b| match b % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect();

    let packed = pack_ternary(&ternary_values);
    let unpacked = unpack_ternary(&packed, ternary_values.len());

    assert_eq!(ternary_values.len(), unpacked.len(), "Length mismatch after round-trip");
    assert_eq!(ternary_values, unpacked, "Values changed after round-trip");

    // Strategy 2: Treat input as packed bytes, unpack them
    let unpacked_raw = unpack_ternary(data, data.len() * 4);
    // All unpacked values should be in {-1, 0, 1}
    for &v in &unpacked_raw {
        assert!(v == -1 || v == 0 || v == 1, "Unpacked value {} is not ternary", v);
    }
});
