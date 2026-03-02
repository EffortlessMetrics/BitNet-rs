#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::attention_mask::{
    apply_mask, create_causal_mask, create_padding_mask, create_sliding_window_mask,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct MaskInput {
    seq_len: u8,
    window: u8,
    batch_size: u8,
    lengths_raw: Vec<u8>,
    scores_raw: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    data.chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: MaskInput| {
    let seq_len = (input.seq_len as usize % 16) + 1;
    let n = seq_len * seq_len;

    // --- create_causal_mask ---
    {
        let mask = create_causal_mask(seq_len);
        assert_eq!(mask.len(), n);
        // Verify lower-triangular structure
        for i in 0..seq_len {
            for j in 0..seq_len {
                let v = mask[i * seq_len + j];
                if j <= i {
                    assert_eq!(v, 0.0, "causal mask [{i},{j}] should be 0.0");
                } else {
                    assert_eq!(v, f32::NEG_INFINITY, "causal mask [{i},{j}] should be -inf");
                }
            }
        }
    }

    // --- create_sliding_window_mask ---
    {
        let window = input.window as usize % (seq_len + 2); // allow 0 and > seq_len
        let mask = create_sliding_window_mask(seq_len, window);
        assert_eq!(mask.len(), n);
        for i in 0..seq_len {
            for j in 0..seq_len {
                let v = mask[i * seq_len + j];
                assert!(v == 0.0 || v == f32::NEG_INFINITY, "unexpected mask value: {v}");
                // Future positions must be blocked
                if j > i {
                    assert_eq!(v, f32::NEG_INFINITY);
                }
            }
        }
    }

    // --- create_padding_mask ---
    {
        let batch = (input.batch_size as usize % 4) + 1;
        let lengths: Vec<usize> =
            input.lengths_raw.iter().take(batch).map(|&b| b as usize % (seq_len + 1)).collect();
        if lengths.len() >= batch {
            let mask = create_padding_mask(&lengths[..batch], seq_len);
            assert_eq!(mask.len(), batch * seq_len);
            for (b, &len) in lengths[..batch].iter().enumerate() {
                let valid = len.min(seq_len);
                for j in 0..seq_len {
                    let v = mask[b * seq_len + j];
                    if j < valid {
                        assert_eq!(v, 0.0, "padding mask [{b},{j}] should be 0.0");
                    } else {
                        assert_eq!(v, f32::NEG_INFINITY, "padding mask [{b},{j}] should be -inf");
                    }
                }
            }
        }
    }

    // --- apply_mask ---
    {
        let scores = bytes_to_f32(&input.scores_raw, n);
        if scores.len() >= n && scores[..n].iter().all(|x| x.is_finite()) {
            let mask = create_causal_mask(seq_len);
            let mut s = scores[..n].to_vec();
            apply_mask(&mut s, &mask, seq_len);
            // Masked positions should be -inf, unmasked should remain finite
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let v = s[i * seq_len + j];
                    if j > i {
                        assert_eq!(v, f32::NEG_INFINITY);
                    }
                }
            }
        }
    }
});
