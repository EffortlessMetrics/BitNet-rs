#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::attention_mask::{
    apply_mask, combine_masks, create_causal_mask, create_padding_mask, create_sliding_window_mask,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct AttentionMaskInput {
    seq_len: u16,
    window: u16,
    padding_lengths: Vec<u16>,
    scores_data: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: AttentionMaskInput| {
    let seq_len = (input.seq_len as usize % 128) + 1;
    let window = (input.window as usize % 128) + 1;

    // --- Causal mask ---
    let causal = create_causal_mask(seq_len);
    assert_eq!(causal.len(), seq_len * seq_len, "causal mask size mismatch");
    // Upper triangle must be -inf (blocked), diagonal and below must be 0.0 (attend).
    for row in 0..seq_len {
        for col in 0..seq_len {
            let val = causal[row * seq_len + col];
            if col <= row {
                assert_eq!(val, 0.0, "causal mask should attend at ({row},{col})");
            } else {
                assert!(
                    val.is_infinite() && val < 0.0,
                    "causal mask should block at ({row},{col})"
                );
            }
        }
    }

    // --- Sliding window mask ---
    let sliding = create_sliding_window_mask(seq_len, window);
    assert_eq!(sliding.len(), seq_len * seq_len, "sliding window mask size mismatch");
    for &val in &sliding {
        assert!(val == 0.0 || (val.is_infinite() && val < 0.0), "unexpected mask value: {val}");
    }

    // --- Padding mask ---
    if !input.padding_lengths.is_empty() {
        let batch = input.padding_lengths.len().min(16);
        let max_len = seq_len;
        let lengths: Vec<usize> =
            input.padding_lengths[..batch].iter().map(|&l| (l as usize) % (max_len + 1)).collect();
        let padding = create_padding_mask(&lengths, max_len);
        assert_eq!(padding.len(), batch * max_len, "padding mask size mismatch");
    }

    // --- combine_masks ---
    let combined = combine_masks(&causal, &sliding, seq_len);
    assert_eq!(combined.len(), seq_len * seq_len, "combined mask size mismatch");

    // --- apply_mask ---
    let total = seq_len * seq_len;
    let mut scores = bytes_to_f32(&input.scores_data, total);
    if scores.len() < total {
        scores.resize(total, 0.0);
    }
    // Filter non-finite scores to avoid spurious failures.
    for s in scores.iter_mut() {
        if !s.is_finite() {
            *s = 0.0;
        }
    }
    apply_mask(&mut scores, &causal, seq_len);
    // After applying causal mask, upper-triangle scores must be -inf.
    for row in 0..seq_len {
        for col in (row + 1)..seq_len {
            let val = scores[row * seq_len + col];
            assert!(
                val.is_infinite() && val < 0.0,
                "score at ({row},{col}) should be -inf after causal mask, got {val}"
            );
        }
    }
});
