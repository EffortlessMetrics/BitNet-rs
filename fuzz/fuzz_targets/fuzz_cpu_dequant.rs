#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::dequant::{
    dequant_i2s_block, dequant_i2s_row, dequant_ternary, pack_ternary,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct DequantInput {
    block_size_hint: u8,
    scale_raw: [u8; 4],
    threshold_raw: [u8; 4],
    packed_data: Vec<u8>,
    float_data: Vec<u8>,
    num_blocks_hint: u8,
}

fn bytes_to_f32_single(b: &[u8; 4]) -> f32 {
    f32::from_le_bytes(*b)
}

fn bytes_to_f32_vec(data: &[u8], max_elems: usize) -> Vec<f32> {
    data.chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: DequantInput| {
    let scale = bytes_to_f32_single(&input.scale_raw);
    if !scale.is_finite() {
        return;
    }

    // --- dequant_i2s_block ---
    {
        let block_size = (input.block_size_hint as usize % 64) + 1;
        let bytes_needed = block_size.div_ceil(4);
        if input.packed_data.len() >= bytes_needed {
            match dequant_i2s_block(&input.packed_data[..bytes_needed], scale, block_size) {
                Ok(out) => {
                    assert_eq!(out.len(), block_size);
                    // Each output must be in {-scale, 0, +scale}
                    for (i, &v) in out.iter().enumerate() {
                        assert!(
                            v == 0.0 || v == scale || v == -scale,
                            "dequant_i2s_block[{i}] = {v}, expected 0/{scale}/{s}",
                            s = -scale,
                        );
                    }
                }
                Err(_) => {} // invalid input is fine
            }
        }

        // Undersized packed data should fail
        if bytes_needed > 0 && input.packed_data.len() < bytes_needed {
            assert!(dequant_i2s_block(&input.packed_data, scale, block_size).is_err());
        }
    }

    // --- dequant_ternary ---
    if !input.packed_data.is_empty() {
        let out = dequant_ternary(&input.packed_data, scale);
        assert_eq!(out.len(), input.packed_data.len() * 4);
        for (i, &v) in out.iter().enumerate() {
            assert!(v == 0.0 || v == scale || v == -scale, "dequant_ternary[{i}] = {v}",);
        }
    }

    // --- pack_ternary roundtrip ---
    {
        let floats = bytes_to_f32_vec(&input.float_data, 32);
        if !floats.is_empty() && floats.iter().all(|x| x.is_finite()) {
            let threshold = bytes_to_f32_single(&input.threshold_raw).abs();
            if threshold.is_finite() {
                let (packed, computed_scale) = pack_ternary(&floats, threshold);
                assert!(computed_scale.is_finite());
                assert!(computed_scale > 0.0);
                // Roundtrip: dequantize what we just packed
                let restored = dequant_ternary(&packed, computed_scale);
                assert!(restored.len() >= floats.len());
                // Each restored value must be ternary
                for &v in &restored {
                    assert!(
                        v == 0.0 || v == computed_scale || v == -computed_scale,
                        "roundtrip value {v} not ternary",
                    );
                }
            }
        }
    }

    // --- dequant_i2s_row ---
    if !input.packed_data.is_empty() {
        let block_size = (input.block_size_hint as usize % 32) + 1;
        let total = input.packed_data.len() * 4;
        let num_blocks = total.div_ceil(block_size);
        let num_scales = (input.num_blocks_hint as usize % 16) + 1;
        let scales = vec![scale; num_scales];

        if num_scales >= num_blocks {
            if let Ok(out) = dequant_i2s_row(&input.packed_data, &scales[..num_blocks], block_size)
            {
                assert_eq!(out.len(), total);
            }
        }
    }
});
