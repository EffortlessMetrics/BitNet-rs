#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::batch::batched_softmax;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct SoftmaxLargeInput {
    dim: u8,
    batch: u8,
    data: Vec<u8>,
    extreme_mode: u8,
    extreme_positions: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: SoftmaxLargeInput| {
    let dim = (input.dim as usize % 128) + 2;
    let batch = (input.batch as usize % 8) + 1;
    let total = batch * dim;

    let mut data = bytes_to_f32(&input.data, total);
    if data.len() < total {
        return;
    }

    // Inject extreme values based on fuzz-selected mode.
    for &pos in input.extreme_positions.iter().take(16) {
        let idx = pos as usize % total;
        match input.extreme_mode % 8 {
            0 => data[idx] = f32::MAX,
            1 => data[idx] = f32::MIN,
            2 => data[idx] = 1e38,
            3 => data[idx] = -1e38,
            4 => data[idx] = f32::MAX / 2.0,
            5 => data[idx] = f32::MIN / 2.0,
            6 => data[idx] = f32::MIN_POSITIVE,
            7 => data[idx] = -f32::MIN_POSITIVE,
            _ => {}
        }
    }

    // Filter to finite-only for the main test.
    let mut finite_data = data[..total].to_vec();
    for v in &mut finite_data {
        if !v.is_finite() {
            *v = 0.0;
        }
    }

    // batched_softmax with extreme but finite values must not produce NaN.
    if let Ok(out) = batched_softmax(&finite_data, batch, dim) {
        assert_eq!(out.len(), total);
        for (i, &v) in out.iter().enumerate() {
            assert!(!v.is_nan(), "softmax NaN at {i}");
            assert!(!v.is_infinite(), "softmax Inf at {i}");
            assert!(v >= 0.0, "softmax negative at {i}: {v}");
            assert!(v <= 1.0 + 1e-6, "softmax >1 at {i}: {v}");
        }

        // Each row must sum to approximately 1.0.
        for bi in 0..batch {
            let row = &out[bi * dim..(bi + 1) * dim];
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "softmax row {bi} sum={sum} (expected ~1.0)");
        }
    }

    // Also exercise bitnet_logits softmax for cross-check.
    let mut logits_copy = finite_data[..dim].to_vec();
    bitnet_logits::softmax_in_place(&mut logits_copy);
    for (i, &v) in logits_copy.iter().enumerate() {
        assert!(!v.is_nan(), "bitnet_logits softmax NaN at {i}");
        assert!(v >= 0.0, "bitnet_logits softmax negative at {i}: {v}");
    }
    let sum: f32 = logits_copy.iter().sum();
    if sum.is_finite() {
        assert!((sum - 1.0).abs() < 1e-3, "bitnet_logits softmax sum={sum} (expected ~1.0)");
    }

    // Invariant: softmax(x + c) == softmax(x) for any constant c (shift invariance).
    let shift = 1000.0f32;
    let shifted: Vec<f32> = finite_data[..dim].iter().map(|&v| v + shift).collect();
    // Build full batch input by repeating the first row.
    let single_batch_orig = finite_data[..dim].to_vec();
    if let (Ok(out_orig), Ok(out_shifted)) =
        (batched_softmax(&single_batch_orig, 1, dim), batched_softmax(&shifted, 1, dim))
    {
        for (i, (&a, &b)) in out_orig.iter().zip(out_shifted.iter()).enumerate() {
            if a.is_finite() && b.is_finite() {
                assert!((a - b).abs() < 1e-4, "shift invariance violated at {i}: {a} vs {b}");
            }
        }
    }

    // Invariant: uniform input → uniform output.
    let uniform = vec![42.0f32; dim];
    if let Ok(out) = batched_softmax(&uniform, 1, dim) {
        let expected = 1.0 / dim as f32;
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-5,
                "uniform softmax at {i}: expected {expected}, got {v}"
            );
        }
    }
});
