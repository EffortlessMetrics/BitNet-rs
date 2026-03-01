#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct SoftmaxInput {
    /// Raw bytes interpreted as f32 logits.
    data: Vec<u8>,
    /// Whether to inject extreme values at random positions.
    inject_max: bool,
    inject_min: bool,
    inject_nan: bool,
    inject_inf: bool,
    inject_positions: Vec<u8>,
}

fuzz_target!(|input: SoftmaxInput| {
    // Convert raw bytes to f32 slice.
    let aligned_len = (input.data.len() / 4) * 4;
    if aligned_len == 0 {
        return;
    }
    let data = &input.data[..aligned_len];
    let mut logits: Vec<f32> = data
        .chunks_exact(4)
        .take(256)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    if logits.is_empty() {
        return;
    }

    // Inject extreme values at fuzz-selected positions.
    for (i, &pos) in input.inject_positions.iter().take(16).enumerate() {
        let idx = pos as usize % logits.len();
        match i % 4 {
            0 if input.inject_max => logits[idx] = f32::MAX,
            1 if input.inject_min => logits[idx] = f32::MIN,
            2 if input.inject_nan => logits[idx] = f32::NAN,
            3 if input.inject_inf => logits[idx] = f32::INFINITY,
            _ => {}
        }
    }

    // Filter to finite values for softmax (mirrors the logits_transforms pattern).
    let finite: Vec<f32> = logits.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        return;
    }

    let mut probs = finite;
    bitnet_logits::softmax_in_place(&mut probs);

    // Invariant 1: No NaN in output.
    for (i, &p) in probs.iter().enumerate() {
        assert!(!p.is_nan(), "softmax output NaN at index {i}");
    }

    // Invariant 2: No Inf in output.
    for (i, &p) in probs.iter().enumerate() {
        assert!(p.is_finite(), "softmax output non-finite at index {i}: {p}");
    }

    // Invariant 3: All values in [0, 1].
    for (i, &p) in probs.iter().enumerate() {
        assert!(p >= 0.0, "softmax output negative at index {i}: {p}");
        assert!(p <= 1.0, "softmax output >1.0 at index {i}: {p}");
    }

    // Invariant 4: Sum ≈ 1.0 (within tolerance for float precision).
    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "softmax sum {sum} not within tolerance of 1.0 (len={})",
        probs.len()
    );

    // Also test softmax on the raw (possibly non-finite) input: must not panic.
    let mut raw = logits;
    bitnet_logits::softmax_in_place(&mut raw);
});
