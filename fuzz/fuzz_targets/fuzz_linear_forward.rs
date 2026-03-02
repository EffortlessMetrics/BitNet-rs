#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::linear::{LinearConfig, linear_cpu};
use libfuzzer_sys::fuzz_target;

/// Fuzz linear projection (y = x · Wᵀ + bias) with arbitrary dimensions.
#[derive(Arbitrary, Debug)]
struct LinearInput {
    /// Batch size selector.
    batch_size: u8,
    /// Input features selector.
    in_features: u8,
    /// Output features selector.
    out_features: u8,
    /// Whether to include bias.
    has_bias: bool,
    /// Raw weight/input bytes.
    raw_data: Vec<u8>,
}

fn bytes_to_f32(raw: &[u8], count: usize) -> Vec<f32> {
    let aligned = (raw.len() / 4) * 4;
    let mut out: Vec<f32> = raw[..aligned]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    out.resize(count, 0.0);
    out.truncate(count);
    out
}

fuzz_target!(|input: LinearInput| {
    // Clamp to avoid huge allocations.
    let batch = (input.batch_size as usize % 8) + 1;
    let in_f = (input.in_features as usize % 16) + 1;
    let out_f = (input.out_features as usize % 16) + 1;

    let config = match LinearConfig::new(batch, in_f, out_f) {
        Ok(c) => c.with_bias(input.has_bias),
        Err(_) => return,
    };

    let x_len = batch * in_f;
    let w_len = out_f * in_f;
    let bias_len = if input.has_bias { out_f } else { 0 };
    let total_input = x_len + w_len + bias_len;

    let all_data = bytes_to_f32(&input.raw_data, total_input);
    let x = &all_data[..x_len];
    let w = &all_data[x_len..x_len + w_len];
    let bias = if input.has_bias { Some(&all_data[x_len + w_len..] as &[f32]) } else { None };

    let mut output = vec![0.0f32; batch * out_f];
    match linear_cpu(x, w, bias, &mut output, &config) {
        Ok(()) => {
            assert_eq!(output.len(), batch * out_f);
            // Check no NaN from finite inputs.
            if all_data.iter().all(|v| v.is_finite()) {
                for (i, &v) in output.iter().enumerate() {
                    assert!(v.is_finite(), "output[{i}] is not finite: {v}");
                }
            }
        }
        Err(_) => {}
    }
});
