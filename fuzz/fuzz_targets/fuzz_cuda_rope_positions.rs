#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cuda::{RopeConfig, compute_sincos_table, rope_forward_cpu};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct CudaRopeInput {
    head_dim: u8,
    n_heads: u8,
    seq_len: u8,
    position_offset: u8,
    interleaved: bool,
    data: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: CudaRopeInput| {
    // head_dim must be even and >= 2.
    let head_dim = ((input.head_dim as usize % 16) + 1) * 2; // 2, 4, ..., 32
    let n_heads = (input.n_heads as usize % 4) + 1;
    let seq_len = (input.seq_len as usize % 16) + 1;
    let total = n_heads * seq_len * head_dim;

    let data = bytes_to_f32(&input.data, total);
    if data.len() < total {
        return;
    }

    // Skip non-finite inputs.
    if data[..total].iter().any(|x| !x.is_finite()) {
        return;
    }

    let config = match RopeConfig::for_shape(head_dim, n_heads, seq_len) {
        Ok(c) => c,
        Err(_) => return,
    };
    let config = config
        .with_position_offset(input.position_offset as usize % 64)
        .with_interleaved(input.interleaved);

    // Test sincos table generation — must not panic.
    let table = compute_sincos_table(&config);
    assert_eq!(table.len(), config.max_seq_len * head_dim, "sincos table length mismatch");
    for (i, &val) in table.iter().enumerate() {
        assert!(val.is_finite(), "sincos table non-finite at index {i}: {val}");
    }

    // Test forward pass.
    let mut output = vec![0.0f32; total];
    if rope_forward_cpu(&data[..total], &mut output, &config).is_ok() {
        // Invariant 1: No NaN or Inf in output.
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "RoPE output non-finite at index {i}: {val}");
        }

        // Invariant 2: Norm preservation per head/position.
        for h in 0..n_heads {
            for p in 0..seq_len {
                let start = h * seq_len * head_dim + p * head_dim;
                let in_slice = &data[start..start + head_dim];
                let out_slice = &output[start..start + head_dim];

                let norm_in: f32 = in_slice.iter().map(|x| x * x).sum();
                let norm_out: f32 = out_slice.iter().map(|x| x * x).sum();

                if norm_in > 1e-12 {
                    let ratio = norm_out / norm_in;
                    assert!(
                        (ratio - 1.0).abs() < 1e-3,
                        "Norm not preserved: in={norm_in} out={norm_out} ratio={ratio} \
                         (head={h}, pos={p})"
                    );
                }
            }
        }
    }
});
