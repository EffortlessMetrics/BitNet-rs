#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct AttentionInput {
    seq_len: u8,
    head_dim: u8,
    num_heads: u8,
    q_data: Vec<f32>,
    k_data: Vec<f32>,
    v_data: Vec<f32>,
}

fn ref_softmax(input: &[f32]) -> Vec<f32> {
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        vec![1.0 / input.len() as f32; input.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

fn ref_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * head_dim];

    for i in 0..seq_len {
        let mut scores = vec![0.0f32; seq_len];
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[j] = dot * scale;
        }

        let weights = ref_softmax(&scores);

        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for j in 0..seq_len {
                sum += weights[j] * v[j * head_dim + d];
            }
            output[i * head_dim + d] = sum;
        }
    }
    output
}

fuzz_target!(|input: AttentionInput| {
    let seq_len = (input.seq_len as usize).clamp(1, 32);
    let head_dim = (input.head_dim as usize).clamp(2, 16) & !1; // must be even
    let num_heads = (input.num_heads as usize).clamp(1, 4);

    let total = seq_len * head_dim;

    for _head in 0..num_heads.min(256) {
        let q: Vec<f32> = input
            .q_data
            .iter()
            .copied()
            .chain(std::iter::repeat(0.0))
            .take(total)
            .map(|x| if x.is_finite() { x.clamp(-100.0, 100.0) } else { 0.0 })
            .collect();
        let k: Vec<f32> = input
            .k_data
            .iter()
            .copied()
            .chain(std::iter::repeat(0.0))
            .take(total)
            .map(|x| if x.is_finite() { x.clamp(-100.0, 100.0) } else { 0.0 })
            .collect();
        let v: Vec<f32> = input
            .v_data
            .iter()
            .copied()
            .chain(std::iter::repeat(0.0))
            .take(total)
            .map(|x| if x.is_finite() { x.clamp(-100.0, 100.0) } else { 0.0 })
            .collect();

        let output = ref_attention(&q, &k, &v, seq_len, head_dim);

        assert_eq!(output.len(), total, "output length mismatch");
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "NaN/Inf in attention output at index {} (seq_len={}, head_dim={}, heads={})",
                i,
                seq_len,
                head_dim,
                num_heads
            );
        }
    }
});
