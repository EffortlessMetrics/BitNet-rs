#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct MaskInput {
    seq_len: u16,
    pad_positions: Vec<u16>,
    batch_size: u8,
}

fuzz_target!(|input: MaskInput| {
    // Cap seq_len to prevent OOM (seq_len^2 allocation).
    let seq_len = (input.seq_len as usize % 512) + 1;

    // build causal mask: lower-triangular with -inf above diagonal.
    let mask = build_causal_mask(seq_len);
    assert_eq!(mask.len(), seq_len * seq_len);

    // Verify causal property: mask[i][j] == -inf iff j > i.
    for i in 0..seq_len {
        for j in 0..seq_len {
            let val = mask[i * seq_len + j];
            if j > i {
                assert!(val < -1e8, "Future position ({i},{j}) should be masked");
            } else {
                assert_eq!(val, 0.0, "Past/current position ({i},{j}) should be unmasked");
            }
        }
    }

    // Build padding mask from arbitrary pad positions.
    let pad_positions: Vec<usize> =
        input.pad_positions.iter().take(64).map(|&p| p as usize % seq_len).collect();
    let pad_mask = build_padding_mask(seq_len, &pad_positions);
    assert_eq!(pad_mask.len(), seq_len);

    // Verify padded positions are masked.
    for &pos in &pad_positions {
        assert!(pad_mask[pos] < -1e8, "Padded position {pos} should be masked");
    }

    // Combined causal + padding mask.
    let combined = combine_masks(&mask, &pad_mask, seq_len);
    assert_eq!(combined.len(), seq_len * seq_len);

    // Batch dimension: multiple sequences.
    let batch = (input.batch_size as usize % 4) + 1;
    let batched: Vec<Vec<f32>> = (0..batch).map(|_| build_causal_mask(seq_len)).collect();
    assert_eq!(batched.len(), batch);
    for b in &batched {
        assert_eq!(b.len(), seq_len * seq_len);
    }

    // Edge: seq_len == 1
    let single = build_causal_mask(1);
    assert_eq!(single, vec![0.0f32]);
});

fn build_causal_mask(seq_len: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask[i * seq_len + j] = -1e9;
            }
        }
    }
    mask
}

fn build_padding_mask(seq_len: usize, pad_positions: &[usize]) -> Vec<f32> {
    let mut mask = vec![0.0f32; seq_len];
    for &pos in pad_positions {
        if pos < seq_len {
            mask[pos] = -1e9;
        }
    }
    mask
}

fn combine_masks(causal: &[f32], padding: &[f32], seq_len: usize) -> Vec<f32> {
    let mut combined = causal.to_vec();
    for i in 0..seq_len {
        for j in 0..seq_len {
            if padding[j] < -1e8 {
                combined[i * seq_len + j] = -1e9;
            }
        }
    }
    combined
}
