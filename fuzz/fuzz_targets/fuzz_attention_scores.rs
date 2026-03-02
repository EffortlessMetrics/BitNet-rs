#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::attention::{AttentionConfig, AttentionKernel};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct AttentionInput {
    seq_len: u8,
    head_dim: u8,
    n_heads: u8,
    q_data: Vec<u8>,
    k_data: Vec<u8>,
    v_data: Vec<u8>,
    use_causal: bool,
    use_multi_head: bool,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: AttentionInput| {
    let seq_len = (input.seq_len as usize % 12) + 1;
    let head_dim = (input.head_dim as usize % 16) + 1;
    let n_heads = (input.n_heads as usize % 4) + 1;

    if input.use_multi_head {
        // Multi-head attention path.
        let model_dim = n_heads * head_dim;
        let total = seq_len * model_dim;

        let q = bytes_to_f32(&input.q_data, total);
        let k = bytes_to_f32(&input.k_data, total);
        let v = bytes_to_f32(&input.v_data, total);

        if q.len() < total || k.len() < total || v.len() < total {
            return;
        }
        if q[..total]
            .iter()
            .chain(k[..total].iter())
            .chain(v[..total].iter())
            .any(|x| !x.is_finite())
        {
            return;
        }

        let cfg = AttentionConfig {
            num_heads: n_heads,
            head_dim,
            seq_len,
            causal: input.use_causal,
            use_alibi: false,
            scale: None,
        };

        match AttentionKernel::multi_head_attention(&q[..total], &k[..total], &v[..total], &cfg) {
            Ok(out) => {
                // Invariant 1: Output shape matches [seq_len, n_heads * head_dim].
                assert_eq!(out.len(), total, "multi_head output shape mismatch");
                // Invariant 2: No NaN in output.
                for (i, &val) in out.iter().enumerate() {
                    assert!(!val.is_nan(), "multi_head NaN at idx {i}");
                }
            }
            Err(_) => {} // Validation errors are fine.
        }
    } else {
        // Single-head scaled dot-product path.
        let elems = seq_len * head_dim;

        let q = bytes_to_f32(&input.q_data, elems);
        let k = bytes_to_f32(&input.k_data, elems);
        let v = bytes_to_f32(&input.v_data, elems);

        if q.len() < elems || k.len() < elems || v.len() < elems {
            return;
        }
        if q[..elems]
            .iter()
            .chain(k[..elems].iter())
            .chain(v[..elems].iter())
            .any(|x| !x.is_finite())
        {
            return;
        }

        let scale = 1.0 / (head_dim as f32).sqrt();

        match AttentionKernel::scaled_dot_product(
            &q[..elems],
            &k[..elems],
            &v[..elems],
            None,
            scale,
            seq_len,
            seq_len,
            head_dim,
        ) {
            Ok(out) => {
                // Invariant 3: Output shape matches [seq_len, head_dim].
                assert_eq!(out.len(), elems, "sdp output shape mismatch");
                // Invariant 4: No NaN in output.
                for (i, &val) in out.iter().enumerate() {
                    assert!(!val.is_nan(), "sdp NaN at idx {i}");
                }
            }
            Err(_) => {}
        }
    }

    // Invariant 5: Zero-dimension configs must error, not panic.
    let bad_cfg = AttentionConfig {
        num_heads: 0,
        head_dim: 0,
        seq_len: 0,
        causal: false,
        use_alibi: false,
        scale: None,
    };
    assert!(bad_cfg.validate().is_err(), "zero-dim config should fail validation");
});
