#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::norm_registry::{layer_norm, rms_norm};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct LayerNormInput {
    dim: u8,
    data_bytes: Vec<u8>,
    weight_bytes: Vec<u8>,
    bias_bytes: Vec<u8>,
    eps_byte: u8,
    use_bias: bool,
    use_rms: bool,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn sanitize_finite(v: &mut [f32]) {
    for x in v.iter_mut() {
        if !x.is_finite() {
            *x = 0.0;
        }
    }
}

fuzz_target!(|input: LayerNormInput| {
    let dim = (input.dim as usize % 128) + 1;

    let mut data = bytes_to_f32(&input.data_bytes, dim);
    if data.len() < dim {
        return;
    }
    sanitize_finite(&mut data);

    let mut weight = bytes_to_f32(&input.weight_bytes, dim);
    if weight.len() < dim {
        weight.resize(dim, 1.0);
    }
    sanitize_finite(&mut weight);

    // Epsilon: map byte to a reasonable range [1e-12, 1e-1].
    let eps = 1e-12_f64 + (input.eps_byte as f64 / 255.0) * (1e-1 - 1e-12);

    if input.use_rms {
        let mut rms_data = data[..dim].to_vec();
        rms_norm(&mut rms_data, &weight[..dim], eps);
        for (i, &v) in rms_data.iter().enumerate() {
            assert!(!v.is_nan(), "rms_norm produced NaN at index {i}");
        }
    } else {
        let bias = if input.use_bias {
            let mut b = bytes_to_f32(&input.bias_bytes, dim);
            if b.len() < dim {
                b.resize(dim, 0.0);
            }
            sanitize_finite(&mut b);
            Some(b)
        } else {
            None
        };

        let mut ln_data = data[..dim].to_vec();
        layer_norm(&mut ln_data, &weight[..dim], bias.as_deref(), eps);
        for (i, &v) in ln_data.iter().enumerate() {
            assert!(!v.is_nan(), "layer_norm produced NaN at index {i}");
        }
    }

    // Invariant: layer_norm of uniform input with weight=1, bias=0 → all bias values.
    let uniform_val = 3.14f32;
    let mut uniform = vec![uniform_val; dim];
    let ones = vec![1.0f32; dim];
    let zeros = vec![0.0f32; dim];
    layer_norm(&mut uniform, &ones, Some(&zeros), eps);
    for (i, &v) in uniform.iter().enumerate() {
        assert!(v.abs() < 1e-3, "layer_norm of uniform input should be ~0 at {i}, got {v}");
    }
});
