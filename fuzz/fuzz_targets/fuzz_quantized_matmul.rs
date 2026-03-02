#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::quantized_matmul::{i2s_matmul_blocked, i2s_matmul_f32, pack_i2s};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct QuantMatmulInput {
    m_byte: u8,
    n_byte: u8,
    k_byte: u8,
    block_size_byte: u8,
    activation_data: Vec<u8>,
    weight_data: Vec<u8>,
    scale_data: Vec<u8>,
    use_blocked: bool,
}

fn bytes_to_f32(data: &[u8], count: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    let mut out: Vec<f32> = data[..aligned]
        .chunks_exact(4)
        .take(count)
        .map(|b| {
            let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            if v.is_finite() { v } else { 0.0 }
        })
        .collect();
    out.resize(count, 0.0);
    out
}

fuzz_target!(|input: QuantMatmulInput| {
    // --- pack_i2s roundtrip: always safe to call ---
    if input.activation_data.len() >= 4 {
        let vals: [i8; 4] = [
            (input.activation_data[0] as i8) % 2,
            (input.activation_data[1] as i8) % 2,
            (input.activation_data[2] as i8) % 2,
            (input.activation_data[3] as i8) % 2,
        ];
        let _packed = pack_i2s(vals);
    }

    // Keep dimensions small to avoid OOM
    let m = (input.m_byte as usize % 4) + 1;
    let n = (input.n_byte as usize % 4) + 1;
    // k must be divisible by 4 for i2s packing
    let k_raw = (input.k_byte as usize % 8) + 1;
    let k = k_raw.div_ceil(4) * 4;
    let block_size_raw = (input.block_size_byte as usize % 8) + 1;
    let block_size = block_size_raw.div_ceil(4) * 4;

    let act_count = m * k;
    let packed_k = k.div_ceil(4);
    let weight_count = n * packed_k;
    let num_blocks_k = k.div_ceil(block_size);
    let scale_count = n * num_blocks_k;
    let out_count = m * n;

    let activations = bytes_to_f32(&input.activation_data, act_count);
    // Weights are packed bytes — just pad
    let mut weights: Vec<u8> = input.weight_data.iter().copied().take(weight_count).collect();
    weights.resize(weight_count, 0);
    let scales = bytes_to_f32(&input.scale_data, scale_count);
    let mut out = vec![0.0f32; out_count];

    if input.use_blocked {
        let _ = i2s_matmul_blocked(&activations, &weights, &scales, &mut out, m, n, k, block_size);
    } else {
        let _ = i2s_matmul_f32(&activations, &weights, &scales, &mut out, m, n, k, block_size);
    }
});
