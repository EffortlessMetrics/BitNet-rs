#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::attention_mask::{
    apply_mask, combine_masks, create_causal_mask, create_padding_mask, create_sliding_window_mask,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct AttentionMaskInput {
    seq_len: u8,
    window_size: u8,
    padding_lengths: Vec<u8>,
    scores_data: Vec<u8>,
    mask_variant: u8,
    combine_two: bool,
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
    let seq_len = (input.seq_len as usize % 32) + 1;
    let n = seq_len * seq_len;

    // --- Causal mask invariants ---
    let causal = create_causal_mask(seq_len);
    assert_eq!(causal.len(), n, "causal mask size mismatch");

    // Invariant 1: Diagonal is always 0.0 (attend to self)
    for i in 0..seq_len {
        assert_eq!(causal[i * seq_len + i], 0.0, "causal diagonal ({i},{i}) should be 0.0");
    }

    // Invariant 2: Lower triangle is 0.0, upper triangle is -inf
    for i in 0..seq_len {
        for j in 0..seq_len {
            let val = causal[i * seq_len + j];
            if j <= i {
                assert_eq!(val, 0.0, "causal ({i},{j}) should be 0.0");
            } else {
                assert!(val.is_infinite() && val < 0.0, "causal ({i},{j}) should be -inf");
            }
        }
    }

    // Invariant 3: Open position count = n*(n+1)/2
    let open = causal.iter().filter(|&&v| v == 0.0).count();
    assert_eq!(open, seq_len * (seq_len + 1) / 2);

    // --- Sliding window mask invariants ---
    let window = (input.window_size as usize % (seq_len + 2)) + 1;
    let sliding = create_sliding_window_mask(seq_len, window);
    assert_eq!(sliding.len(), n, "sliding mask size mismatch");

    // Invariant 4: Sliding window is a subset of causal mask (no future positions)
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            let val = sliding[i * seq_len + j];
            assert!(val.is_infinite() && val < 0.0, "sliding ({i},{j}) should block future");
        }
    }

    // Invariant 5: When window >= seq_len, sliding == causal
    let large_sliding = create_sliding_window_mask(seq_len, seq_len);
    assert_eq!(causal, large_sliding, "large window should equal causal");

    // --- Padding mask invariants ---
    let batch_size = (input.padding_lengths.len() % 4).max(1);
    let lengths: Vec<usize> = input
        .padding_lengths
        .iter()
        .take(batch_size)
        .map(|&l| l as usize % (seq_len + 2))
        .collect();
    let padding = create_padding_mask(&lengths, seq_len);
    assert_eq!(padding.len(), batch_size * seq_len, "padding mask size mismatch");

    // Invariant 6: Positions before length are 0.0, after are -inf
    for (b, &len) in lengths.iter().enumerate() {
        let valid = len.min(seq_len);
        for j in 0..seq_len {
            let val = padding[b * seq_len + j];
            if j < valid {
                assert_eq!(val, 0.0, "padding batch={b} pos={j} should be 0.0 (len={len})");
            } else {
                assert!(
                    val.is_infinite() && val < 0.0,
                    "padding batch={b} pos={j} should be -inf (len={len})"
                );
            }
        }
    }

    // --- apply_mask invariants ---
    let raw_scores = bytes_to_f32(&input.scores_data, n);
    if raw_scores.len() >= n {
        let mut scores = raw_scores[..n].to_vec();
        if scores.iter().all(|v| v.is_finite()) {
            let mask = match input.mask_variant % 3 {
                0 => causal.clone(),
                1 => sliding.clone(),
                _ => vec![0.0f32; n],
            };
            let original = scores.clone();
            apply_mask(&mut scores, &mask, seq_len);

            // Invariant 7: Where mask is 0.0, score is unchanged
            for i in 0..n {
                if mask[i] == 0.0 {
                    assert_eq!(scores[i], original[i], "score at {i} changed under zero mask");
                }
            }
        }
    }

    // --- combine_masks invariants ---
    if input.combine_two {
        let combined = combine_masks(&causal, &sliding, seq_len);
        assert_eq!(combined.len(), n, "combined mask size mismatch");

        // Invariant 8: Combined mask blocks whenever either input blocks
        for i in 0..n {
            if causal[i].is_infinite() || sliding[i].is_infinite() {
                assert!(
                    combined[i].is_infinite() && combined[i] < 0.0,
                    "combined at {i} should be -inf when either input blocks"
                );
            }
        }
    }

    // --- Edge case: seq_len=0 must not panic ---
    let empty = create_causal_mask(0);
    assert!(empty.is_empty());
    let empty_slide = create_sliding_window_mask(0, 5);
    assert!(empty_slide.is_empty());
});
