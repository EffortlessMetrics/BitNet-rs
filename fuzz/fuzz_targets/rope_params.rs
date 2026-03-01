#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, apply_rope_batch, compute_frequencies};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct RopeInput {
    head_dim_half: u8,
    max_seq_len: u8,
    base: f32,
    scaling_factor: f32,
    position: u8,
    data: Vec<f32>,
    start_pos: u8,
    seq_len: u8,
    num_heads: u8,
}

fuzz_target!(|input: RopeInput| {
    // head_dim must be even and non-zero
    let half_dim = (input.head_dim_half as usize).clamp(1, 32);
    let head_dim = half_dim * 2;
    let max_seq_len = (input.max_seq_len as usize).clamp(1, 64);

    let base = if input.base.is_finite() && input.base > 0.0 {
        input.base.clamp(1.0, 1e8)
    } else {
        10_000.0
    };
    let scaling = if input.scaling_factor.is_finite() && input.scaling_factor > 0.0 {
        input.scaling_factor.clamp(0.001, 100.0)
    } else {
        1.0
    };

    let config =
        RopeConfig::new(head_dim, max_seq_len).with_base(base).with_scaling_factor(scaling);
    let freqs = compute_frequencies(&config);

    assert_eq!(freqs.len(), max_seq_len * head_dim);

    // Single-position apply
    let position = (input.position as usize) % max_seq_len;
    let mut data: Vec<f32> = input
        .data
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .take(head_dim)
        .chain(std::iter::repeat_n(0.0f32, head_dim))
        .take(head_dim)
        .collect();

    let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    apply_rope(&mut data, position, head_dim, &freqs);

    for &v in &data {
        assert!(v.is_finite(), "apply_rope produced non-finite: {v}");
    }

    // Rotation preserves norm
    let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_before > 1e-6 {
        assert!(
            (norm_before - norm_after).abs() < 1e-3,
            "norm not preserved: {norm_before} vs {norm_after}"
        );
    }

    // Batch apply
    let num_heads = (input.num_heads as usize).clamp(1, 4);
    let seq_len = (input.seq_len as usize).clamp(1, 8);
    let start_pos = (input.start_pos as usize) % max_seq_len.saturating_sub(seq_len).max(1);

    if start_pos + seq_len > max_seq_len {
        return;
    }

    let batch_size = seq_len * num_heads * head_dim;
    let mut batch_data: Vec<f32> = (0..batch_size).map(|i| (i as f32) * 0.1 - 5.0).collect();

    apply_rope_batch(&mut batch_data, start_pos, seq_len, num_heads, head_dim, &freqs);

    for &v in &batch_data {
        assert!(v.is_finite(), "apply_rope_batch produced non-finite: {v}");
    }
});
