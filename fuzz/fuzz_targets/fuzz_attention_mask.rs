#![no_main]

use arbitrary::Arbitrary;
use bitnet_inference::attention_mask::{
    MaskType, apply_padding, attended_count, build_mask, causal_mask, full_mask, mask_to_f32,
    sliding_window_mask,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct AttentionMaskInput {
    seq_len: u8,
    window_size: u8,
    actual_len: u8,
    mask_mode: u8,
}

fuzz_target!(|input: AttentionMaskInput| {
    // Clamp seq_len to avoid huge allocations (seq_len² elements).
    let seq_len = (input.seq_len as usize % 64) + 1;
    let window_size = (input.window_size as usize % 128) + 1;
    let total = seq_len * seq_len;

    // Test build_mask with all mask types.
    let mask_type = match input.mask_mode % 3 {
        0 => MaskType::Causal,
        1 => MaskType::Full,
        _ => MaskType::SlidingWindow(window_size),
    };
    let mask = build_mask(mask_type, seq_len);
    assert_eq!(mask.len(), total, "mask length mismatch");

    // Causal mask invariants.
    let causal = causal_mask(seq_len);
    assert_eq!(causal.len(), total);
    for i in 0..seq_len {
        // Diagonal must always be set (position can attend to itself).
        assert!(causal[i * seq_len + i], "causal diagonal not set at pos {i}");
        // Upper triangle must be false.
        for j in (i + 1)..seq_len {
            assert!(!causal[i * seq_len + j], "causal upper-triangle set at ({i},{j})");
        }
        // attended_count must equal i + 1 for causal mask.
        let count = attended_count(&causal, seq_len, i);
        assert_eq!(count, i + 1, "causal attended_count wrong at pos {i}");
    }

    // Full mask: every element must be true.
    let full = full_mask(seq_len);
    assert!(full.iter().all(|&b| b), "full mask has false entries");

    // Sliding window invariants.
    let sw = sliding_window_mask(seq_len, window_size);
    assert_eq!(sw.len(), total);
    for i in 0..seq_len {
        // Diagonal must be set.
        assert!(sw[i * seq_len + i], "sliding window diagonal not set at {i}");
        // Upper triangle must be false (still causal).
        for j in (i + 1)..seq_len {
            assert!(!sw[i * seq_len + j], "sliding window upper-triangle set at ({i},{j})");
        }
        // attended_count must be <= min(window_size, i+1).
        let count = attended_count(&sw, seq_len, i);
        let expected_max = window_size.min(i + 1);
        assert!(
            count <= expected_max,
            "sliding window count {count} > expected max {expected_max} at pos {i}"
        );
    }

    // apply_padding: positions beyond actual_len must be masked out.
    let actual_len = (input.actual_len as usize) % (seq_len + 1);
    let mut padded = causal.clone();
    apply_padding(&mut padded, seq_len, actual_len);
    for i in actual_len..seq_len {
        assert_eq!(
            attended_count(&padded, seq_len, i),
            0,
            "padded row {i} should have zero attended"
        );
    }

    // mask_to_f32: true → 0.0, false → -inf.
    let f32_mask = mask_to_f32(&causal);
    assert_eq!(f32_mask.len(), total);
    for (idx, (&b, &f)) in causal.iter().zip(f32_mask.iter()).enumerate() {
        if b {
            assert_eq!(f, 0.0, "mask_to_f32 true→non-zero at {idx}");
        } else {
            assert!(f.is_infinite() && f < 0.0, "mask_to_f32 false→not -inf at {idx}");
        }
    }

    // Edge: attended_count for out-of-bounds position must be 0.
    assert_eq!(attended_count(&causal, seq_len, seq_len + 100), 0);
});
