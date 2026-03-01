#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::attention::{
    attention_with_kv_cache, multi_head_attention_cpu, scaled_dot_product_attention,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct CpuAttentionInput {
    seq_q: u8,
    seq_k: u8,
    head_dim: u8,
    num_heads: u8,
    causal: bool,
    q_data: Vec<u8>,
    k_data: Vec<u8>,
    v_data: Vec<u8>,
    test_kv_cache: bool,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: CpuAttentionInput| {
    let seq_q = (input.seq_q as usize % 8) + 1;
    let seq_k = (input.seq_k as usize % 8) + 1;
    let head_dim = (input.head_dim as usize % 16) + 2;
    let num_heads = (input.num_heads as usize % 4) + 1;

    // --- scaled_dot_product_attention ---
    {
        let q_elems = seq_q * head_dim;
        let k_elems = seq_k * head_dim;
        let q = bytes_to_f32(&input.q_data, q_elems);
        let k = bytes_to_f32(&input.k_data, k_elems);
        let v = bytes_to_f32(&input.v_data, k_elems);

        if q.len() >= q_elems && k.len() >= k_elems && v.len() >= k_elems {
            let q = &q[..q_elems];
            let k = &k[..k_elems];
            let v = &v[..k_elems];

            if q.iter().chain(k.iter()).chain(v.iter()).all(|x| x.is_finite()) {
                if let Ok(out) =
                    scaled_dot_product_attention(q, k, v, seq_q, seq_k, head_dim, input.causal)
                {
                    assert_eq!(out.len(), seq_q * head_dim);
                    for (i, &val) in out.iter().enumerate() {
                        assert!(
                            val.is_finite(),
                            "sdpa non-finite at {i}: {val} (seq_q={seq_q}, seq_k={seq_k}, hd={head_dim})"
                        );
                    }
                }
            }
        }
    }

    // --- multi_head_attention_cpu ---
    {
        let seq = (seq_q.min(seq_k)).max(1);
        let total = seq * num_heads * head_dim;
        let q = bytes_to_f32(&input.q_data, total);
        let k = bytes_to_f32(&input.k_data, total);
        let v = bytes_to_f32(&input.v_data, total);

        if q.len() >= total && k.len() >= total && v.len() >= total {
            let q = &q[..total];
            let k = &k[..total];
            let v = &v[..total];

            if q.iter().chain(k.iter()).chain(v.iter()).all(|x| x.is_finite()) {
                if let Ok(out) =
                    multi_head_attention_cpu(q, k, v, num_heads, head_dim, seq, input.causal)
                {
                    assert_eq!(out.len(), total);
                    for (i, &val) in out.iter().enumerate() {
                        assert!(val.is_finite(), "mha non-finite at {i}: {val}");
                    }
                }
            }
        }
    }

    // --- attention_with_kv_cache ---
    if input.test_kv_cache {
        let q = bytes_to_f32(&input.q_data, head_dim);
        let k_new = bytes_to_f32(&input.k_data, head_dim);
        let v_new = bytes_to_f32(&input.v_data, head_dim);

        if q.len() >= head_dim && k_new.len() >= head_dim && v_new.len() >= head_dim {
            let q = &q[..head_dim];
            let k_new = &k_new[..head_dim];
            let v_new = &v_new[..head_dim];

            if q.iter().chain(k_new.iter()).chain(v_new.iter()).all(|x| x.is_finite()) {
                let mut k_cache = Vec::new();
                let mut v_cache = Vec::new();

                // Append a few entries to exercise the cache path.
                for _ in 0..seq_k.min(4) {
                    let result = attention_with_kv_cache(
                        q,
                        &mut k_cache,
                        &mut v_cache,
                        k_new,
                        v_new,
                        head_dim,
                    );
                    if let Ok(out) = result {
                        assert_eq!(out.len(), head_dim);
                        for &val in &out {
                            assert!(val.is_finite(), "kv_cache non-finite: {val}");
                        }
                    }
                }
            }
        }
    }
});
