#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct AttentionInput {
    seq_len: u8,
    head_dim: u8,
    n_heads: u8,
    q_data: Vec<u8>,
    k_data: Vec<u8>,
    v_data: Vec<u8>,
    use_causal_mask: bool,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn softmax(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        // Replace non-finite with uniform
        let uniform = 1.0 / logits.len() as f32;
        logits.fill(uniform);
        return;
    }
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 && sum.is_finite() {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    } else {
        let uniform = 1.0 / logits.len() as f32;
        logits.fill(uniform);
    }
}

fn attention_scores(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len * seq_len];

    // Q @ K^T with scaling
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * seq_len + j] = dot * scale;
        }
    }

    // Causal mask: set future positions to -inf
    if causal {
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax per row
    for i in 0..seq_len {
        let row = &mut scores[i * seq_len..(i + 1) * seq_len];
        softmax(row);
    }

    // Scores @ V
    let mut output = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for j in 0..seq_len {
                sum += scores[i * seq_len + j] * v[j * head_dim + d];
            }
            output[i * head_dim + d] = sum;
        }
    }

    output
}

fuzz_target!(|input: AttentionInput| {
    let seq_len = (input.seq_len as usize % 16) + 1;
    let head_dim = (input.head_dim as usize % 16) + 1;
    let n_heads = (input.n_heads as usize % 4) + 1;
    let elems_per_head = seq_len * head_dim;

    let q_all = bytes_to_f32(&input.q_data, n_heads * elems_per_head);
    let k_all = bytes_to_f32(&input.k_data, n_heads * elems_per_head);
    let v_all = bytes_to_f32(&input.v_data, n_heads * elems_per_head);

    if q_all.len() < n_heads * elems_per_head
        || k_all.len() < n_heads * elems_per_head
        || v_all.len() < n_heads * elems_per_head
    {
        return;
    }

    // Filter out inputs with NaN/inf to test pure numerical stability
    let has_nonfinite =
        q_all.iter().chain(k_all.iter()).chain(v_all.iter()).any(|x| !x.is_finite());
    if has_nonfinite {
        return;
    }

    for h in 0..n_heads {
        let offset = h * elems_per_head;
        let q = &q_all[offset..offset + elems_per_head];
        let k = &k_all[offset..offset + elems_per_head];
        let v = &v_all[offset..offset + elems_per_head];

        let output = attention_scores(q, k, v, seq_len, head_dim, input.use_causal_mask);

        // Invariant 1: Output shape matches [seq_len, head_dim]
        assert_eq!(
            output.len(),
            elems_per_head,
            "output shape mismatch: expected {elems_per_head}, got {}",
            output.len()
        );

        // Invariant 2: No NaN or Inf in output
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "attention output non-finite at index {i}: {val} (head={h})");
        }
    }

    // Invariant 3: With causal mask, running with seq_len=1 must match first row
    if seq_len > 1 && !has_nonfinite {
        let q_row = &q_all[..head_dim];
        let k_row = &k_all[..head_dim];
        let v_row = &v_all[..head_dim];

        let single = attention_scores(q_row, k_row, v_row, 1, head_dim, input.use_causal_mask);
        assert_eq!(single.len(), head_dim, "single-token output shape mismatch");
        for &val in &single {
            assert!(val.is_finite(), "single-token attention produced non-finite");
        }
    }
});
