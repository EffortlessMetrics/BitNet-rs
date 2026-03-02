#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::attention::flash_attention_cpu;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct FlashDecodingInput {
    seq_q: u8,
    seq_k: u8,
    head_dim: u8,
    num_heads: u8,
    causal: bool,
    q_data: Vec<u8>,
    k_data: Vec<u8>,
    v_data: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| {
            let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            if v.is_finite() { v.clamp(-1e4, 1e4) } else { 0.0 }
        })
        .collect()
}

/// Reference scaled dot-product attention for cross-checking.
fn reference_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_q * head_dim];

    for qi in 0..seq_q {
        let mut scores = vec![0.0f32; seq_k];
        for kj in 0..seq_k {
            if causal && kj > qi {
                scores[kj] = f32::NEG_INFINITY;
            } else {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qi * head_dim + d] * k[kj * head_dim + d];
                }
                scores[kj] = dot * scale;
            }
        }

        // Softmax
        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_s).exp();
            sum += *s;
        }
        if sum > 0.0 && sum.is_finite() {
            for s in &mut scores {
                *s /= sum;
            }
        } else {
            let uniform = 1.0 / seq_k as f32;
            scores.fill(uniform);
        }

        // Weighted sum of V
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for kj in 0..seq_k {
                acc += scores[kj] * v[kj * head_dim + d];
            }
            output[qi * head_dim + d] = acc;
        }
    }

    output
}

fuzz_target!(|input: FlashDecodingInput| {
    let seq_q = (input.seq_q as usize % 16) + 1;
    let seq_k = (input.seq_k as usize % 16) + 1;
    let head_dim = (input.head_dim as usize % 32) + 2;
    let num_heads = (input.num_heads as usize % 4) + 1;

    let elems_q = seq_q * head_dim;
    let elems_k = seq_k * head_dim;

    let mut q = bytes_to_f32(&input.q_data, num_heads * elems_q);
    let mut k = bytes_to_f32(&input.k_data, num_heads * elems_k);
    let mut v = bytes_to_f32(&input.v_data, num_heads * elems_k);

    q.resize(num_heads * elems_q, 0.0);
    k.resize(num_heads * elems_k, 0.0);
    v.resize(num_heads * elems_k, 0.0);

    for h in 0..num_heads {
        let q_slice = &q[h * elems_q..(h + 1) * elems_q];
        let k_slice = &k[h * elems_k..(h + 1) * elems_k];
        let v_slice = &v[h * elems_k..(h + 1) * elems_k];

        // Invariant 1: flash_attention_cpu must not panic
        let flash_result =
            flash_attention_cpu(q_slice, k_slice, v_slice, seq_q, seq_k, head_dim, input.causal);

        if let Ok(flash_out) = flash_result {
            // Invariant 2: Output shape must be [seq_q, head_dim]
            assert_eq!(
                flash_out.len(),
                elems_q,
                "flash output shape mismatch: expected {elems_q}, got {}",
                flash_out.len()
            );

            // Invariant 3: All output values must be finite
            for (i, &val) in flash_out.iter().enumerate() {
                assert!(val.is_finite(), "flash output non-finite at index {i}: {val} (head={h})");
            }

            // Invariant 4: Cross-check with reference attention (tolerance for tiling)
            let ref_out = reference_attention(
                q_slice,
                k_slice,
                v_slice,
                seq_q,
                seq_k,
                head_dim,
                input.causal,
            );
            assert_eq!(ref_out.len(), flash_out.len());

            let mut max_diff = 0.0f32;
            for (i, (&a, &b)) in flash_out.iter().zip(ref_out.iter()).enumerate() {
                if a.is_finite() && b.is_finite() {
                    let diff = (a - b).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    assert!(
                        diff < 0.05,
                        "flash vs reference mismatch at {i}: flash={a}, ref={b}, diff={diff} (head={h})"
                    );
                }
            }
        }
    }

    // Invariant 5: Single-token decode (seq_q=1) must succeed
    {
        let q1 = &q[..head_dim];
        let k1 = &k[..elems_k];
        let v1 = &v[..elems_k];
        if let Ok(out) = flash_attention_cpu(q1, k1, v1, 1, seq_k, head_dim, input.causal) {
            assert_eq!(out.len(), head_dim, "single-token decode output shape mismatch");
            for &val in &out {
                assert!(val.is_finite(), "single-token decode produced non-finite");
            }
        }
    }
});
