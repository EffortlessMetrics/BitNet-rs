#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct MaskInput {
    /// Sequence lengths for each batch entry.
    seq_lens: Vec<u8>,
    /// Maximum sequence length override.
    max_seq_len: u8,
    /// Whether to apply causal masking.
    causal: bool,
    /// Whether to apply padding masking.
    padding: bool,
}

/// Build a causal mask: position j > i is masked (set to NEG_INFINITY).
fn build_causal_mask(seq_len: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    mask
}

/// Build a padding mask: positions beyond `actual_len` are masked.
fn build_padding_mask(actual_len: usize, max_len: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; max_len];
    for j in actual_len..max_len {
        mask[j] = f32::NEG_INFINITY;
    }
    mask
}

/// Combine causal and padding masks into a 2-D attention mask.
fn build_attention_mask(
    actual_len: usize,
    max_len: usize,
    causal: bool,
    padding: bool,
) -> Vec<f32> {
    let mut mask = vec![0.0f32; max_len * max_len];

    if causal {
        let causal_m = build_causal_mask(max_len);
        for (i, v) in causal_m.iter().enumerate() {
            if *v == f32::NEG_INFINITY {
                mask[i] = f32::NEG_INFINITY;
            }
        }
    }

    if padding {
        let pad_m = build_padding_mask(actual_len, max_len);
        for i in 0..max_len {
            for j in 0..max_len {
                if pad_m[j] == f32::NEG_INFINITY {
                    mask[i * max_len + j] = f32::NEG_INFINITY;
                }
            }
        }
    }

    mask
}

fuzz_target!(|input: MaskInput| {
    let seq_lens: Vec<usize> =
        input.seq_lens.iter().take(16).map(|&s| (s as usize % 32) + 1).collect();
    if seq_lens.is_empty() {
        return;
    }

    let max_len = ((input.max_seq_len as usize) % 32) + 1;

    for &actual_len in &seq_lens {
        let actual = actual_len.min(max_len);

        // --- Causal mask invariants ---
        let causal = build_causal_mask(max_len);
        assert_eq!(causal.len(), max_len * max_len, "causal mask size mismatch");

        // Invariant 1: Diagonal is always unmasked.
        for i in 0..max_len {
            assert_eq!(causal[i * max_len + i], 0.0, "diagonal position ({i},{i}) should be 0.0");
        }

        // Invariant 2: Lower triangle (j <= i) is unmasked.
        for i in 0..max_len {
            for j in 0..=i {
                assert_eq!(causal[i * max_len + j], 0.0, "lower-tri ({i},{j}) should be 0.0");
            }
        }

        // Invariant 3: Upper triangle (j > i) is masked.
        for i in 0..max_len {
            for j in (i + 1)..max_len {
                assert_eq!(
                    causal[i * max_len + j],
                    f32::NEG_INFINITY,
                    "upper-tri ({i},{j}) should be -inf"
                );
            }
        }

        // --- Padding mask invariants ---
        let pad = build_padding_mask(actual, max_len);
        assert_eq!(pad.len(), max_len, "padding mask size mismatch");

        // Invariant 4: Positions < actual_len are unmasked.
        for j in 0..actual {
            assert_eq!(pad[j], 0.0, "padding mask position {j} should be 0.0 (actual={actual})");
        }

        // Invariant 5: Positions >= actual_len are masked.
        for j in actual..max_len {
            assert_eq!(
                pad[j],
                f32::NEG_INFINITY,
                "padding mask position {j} should be -inf (actual={actual})"
            );
        }

        // --- Combined mask invariants ---
        let combined = build_attention_mask(actual, max_len, input.causal, input.padding);
        assert_eq!(combined.len(), max_len * max_len, "combined mask size mismatch");

        // Invariant 6: No NaN values in the mask.
        for (idx, &v) in combined.iter().enumerate() {
            assert!(!v.is_nan(), "combined mask has NaN at index {idx}");
        }

        // Invariant 7: Mask values are only 0.0 or NEG_INFINITY.
        for (idx, &v) in combined.iter().enumerate() {
            assert!(
                v == 0.0 || v == f32::NEG_INFINITY,
                "combined mask has unexpected value {v} at index {idx}"
            );
        }

        // Invariant 8: With both causal and padding, padded future positions are masked.
        if input.causal && input.padding && actual < max_len {
            for j in actual..max_len {
                for i in 0..max_len {
                    assert_eq!(
                        combined[i * max_len + j],
                        f32::NEG_INFINITY,
                        "padded column {j} in row {i} should be masked"
                    );
                }
            }
        }
    }
});
