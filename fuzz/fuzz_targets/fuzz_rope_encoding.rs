#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, apply_rope_batch, compute_frequencies};
use libfuzzer_sys::fuzz_target;

/// Fuzz RoPE positional encoding with arbitrary head dims and positions.
#[derive(Arbitrary, Debug)]
struct RopeInput {
    /// Head dimension selector (even, clamped).
    head_dim_half: u8,
    /// Max sequence length selector.
    max_seq_len: u8,
    /// Position to encode.
    position: u8,
    /// Number of heads for batch variant.
    num_heads: u8,
    /// Base frequency selector.
    base_selector: u8,
    /// Scaling factor selector.
    scale_selector: u8,
    /// Whether to test batch variant.
    use_batch: bool,
    /// Raw data bytes for head vectors.
    raw_data: Vec<u8>,
}

fn bytes_to_f32(raw: &[u8], count: usize) -> Vec<f32> {
    let aligned = (raw.len() / 4) * 4;
    let mut out: Vec<f32> = raw[..aligned]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    out.resize(count, 0.0);
    out.truncate(count);
    out
}

fuzz_target!(|input: RopeInput| {
    // head_dim must be even and non-zero.
    let head_dim = ((input.head_dim_half as usize % 16) + 1) * 2;
    let max_seq = (input.max_seq_len as usize % 64) + 1;
    let position = input.position as usize % max_seq;

    let base = match input.base_selector % 3 {
        0 => 10_000.0,
        1 => 500_000.0,
        _ => 1_000.0,
    };
    let scale = match input.scale_selector % 3 {
        0 => 1.0,
        1 => 0.5,
        _ => 2.0,
    };

    let config = RopeConfig::new(head_dim, max_seq).with_base(base).with_scaling_factor(scale);

    let frequencies = compute_frequencies(&config);

    // Verify frequency table size.
    assert_eq!(frequencies.len(), max_seq * head_dim);

    if input.use_batch {
        let num_heads = (input.num_heads as usize % 8) + 1;
        let seq_len = 1; // single position per fuzz iteration
        let data_len = seq_len * num_heads * head_dim;
        let mut data = bytes_to_f32(&input.raw_data, data_len);
        apply_rope_batch(&mut data, position, seq_len, num_heads, head_dim, &frequencies);
        assert_eq!(data.len(), data_len);
    } else {
        let mut data = bytes_to_f32(&input.raw_data, head_dim);
        apply_rope(&mut data, position, head_dim, &frequencies);
        assert_eq!(data.len(), head_dim);
    }
});
