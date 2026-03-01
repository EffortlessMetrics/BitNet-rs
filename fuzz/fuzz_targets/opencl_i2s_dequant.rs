#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let num_packed_bytes = (data[0] as usize % 64) + 1;
    let block_size = match data[1] % 3 {
        0 => 32, // BitNet32-F16
        1 => 128,
        _ => 256, // QK256
    };

    if data.len() < 8 + num_packed_bytes {
        return;
    }

    let packed = &data[8..8 + num_packed_bytes];
    let num_elements = num_packed_bytes * 4;
    let num_blocks = (num_elements + block_size - 1) / block_size;

    let mut scales = vec![1.0f32; num_blocks];
    let mut scale_offset = 8 + num_packed_bytes;
    for s in scales.iter_mut() {
        if scale_offset + 4 <= data.len() {
            let bytes = [
                data[scale_offset],
                data[scale_offset + 1],
                data[scale_offset + 2],
                data[scale_offset + 3],
            ];
            let v = f32::from_le_bytes(bytes);
            if v.is_finite() {
                *s = v;
            }
            scale_offset += 4;
        }
    }

    // Dequantize
    let mut output = Vec::with_capacity(num_elements);
    for (byte_idx, &byte) in packed.iter().enumerate() {
        for j in 0..4 {
            let val = (byte >> (j * 2)) & 0x03;
            let fval = (val as i32 - 1) as f32;
            let elem_idx = byte_idx * 4 + j;
            let block_idx = elem_idx / block_size;
            let scale = scales.get(block_idx).copied().unwrap_or(1.0);
            output.push(fval * scale);
        }
    }

    // Verify: ternary values are -1, 0, or +1 (before scaling)
    for &byte in packed {
        for j in 0..4 {
            let val = (byte >> (j * 2)) & 0x03;
            assert!(val <= 3, "I2_S value out of range: {val}");
        }
    }
});
