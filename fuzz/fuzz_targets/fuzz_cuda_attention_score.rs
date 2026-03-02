#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cuda::{
    AttentionConfig, attention_cpu_fallback, multi_head_attention_cpu_fallback,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct CudaAttentionInput {
    seq_len: u8,
    head_dim: u8,
    n_heads: u8,
    q_data: Vec<u8>,
    k_data: Vec<u8>,
    v_data: Vec<u8>,
    causal: bool,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: CudaAttentionInput| {
    let seq_len = (input.seq_len as usize % 16) + 1;
    let head_dim = ((input.head_dim as usize % 16) + 1) * 2; // must be even
    let n_heads = (input.n_heads as usize % 4) + 1;
    let per_head = seq_len * head_dim;
    let total = n_heads * per_head;

    let q = bytes_to_f32(&input.q_data, total);
    let k = bytes_to_f32(&input.k_data, total);
    let v = bytes_to_f32(&input.v_data, total);

    if q.len() < total || k.len() < total || v.len() < total {
        return;
    }

    // Skip non-finite inputs.
    if q[..total].iter().chain(k[..total].iter()).chain(v[..total].iter()).any(|x| !x.is_finite()) {
        return;
    }

    // Test single-head fallback on each head.
    let config = match AttentionConfig::new(1, head_dim, seq_len, input.causal) {
        Ok(c) => c,
        Err(_) => return,
    };

    for h in 0..n_heads {
        let offset = h * per_head;
        let q_head = &q[offset..offset + per_head];
        let k_head = &k[offset..offset + per_head];
        let v_head = &v[offset..offset + per_head];

        if let Ok(out) = attention_cpu_fallback(q_head, k_head, v_head, &config) {
            // Invariant 1: Output shape matches.
            assert_eq!(out.len(), per_head, "attention output shape mismatch for head {h}");

            // Invariant 2: No NaN or Inf in output.
            for (i, &val) in out.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "attention output non-finite at index {i}: {val} (head={h})"
                );
            }
        }
    }

    // Test multi-head fallback.
    let mh_config = match AttentionConfig::new(n_heads, head_dim, seq_len, input.causal) {
        Ok(c) => c,
        Err(_) => return,
    };

    if let Ok(out) =
        multi_head_attention_cpu_fallback(&q[..total], &k[..total], &v[..total], &mh_config)
    {
        assert_eq!(out.len(), total, "multi-head attention output shape mismatch");
        for (i, &val) in out.iter().enumerate() {
            assert!(val.is_finite(), "multi-head attention output non-finite at index {i}: {val}");
        }
    }
});
