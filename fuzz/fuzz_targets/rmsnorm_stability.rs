#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct RmsNormInput {
    data: Vec<f32>,
    weights: Vec<f32>,
}

fn ref_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    input.iter().zip(weight.iter()).map(|(&x, &w)| x * rms * w).collect()
}

fuzz_target!(|input: RmsNormInput| {
    if input.data.is_empty() || input.data.len() > 256 {
        return;
    }

    let n = input.data.len().min(256);

    // Sanitize inputs: replace non-finite with 0
    let data: Vec<f32> = input
        .data
        .iter()
        .take(n)
        .map(|&x| if x.is_finite() { x.clamp(-1e6, 1e6) } else { 0.0 })
        .collect();

    // Pad or trim weights to match data length
    let weights: Vec<f32> = input
        .weights
        .iter()
        .copied()
        .chain(std::iter::repeat(1.0))
        .take(n)
        .map(|x| if x.is_finite() { x.clamp(-1e6, 1e6) } else { 1.0 })
        .collect();

    let eps = 1e-5f32;
    let output = ref_rms_norm(&data, &weights, eps);

    // Verify no NaN/Inf in output
    for (i, &val) in output.iter().enumerate() {
        assert!(val.is_finite(), "NaN/Inf in rmsnorm output at index {} (input len {})", i, n);
    }

    // Verify output has finite norm
    let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(norm.is_finite(), "rmsnorm output norm is not finite: {}", norm);

    // Test near-zero input: should not produce NaN due to eps
    let near_zero: Vec<f32> = vec![1e-20; n];
    let near_zero_out = ref_rms_norm(&near_zero, &weights, eps);
    for (i, &val) in near_zero_out.iter().enumerate() {
        assert!(val.is_finite(), "near-zero input produced non-finite at index {}", i);
    }

    // Test all-zero input: should produce all zeros (0 * rms * w = 0)
    let zeros = vec![0.0f32; n];
    let zero_out = ref_rms_norm(&zeros, &weights, eps);
    for (i, &val) in zero_out.iter().enumerate() {
        assert!(
            val.abs() < 1e-10,
            "zero input should produce zero output at index {}, got {}",
            i,
            val
        );
    }

    // Test very large input: should not overflow
    let large: Vec<f32> = vec![1e6; n];
    let large_out = ref_rms_norm(&large, &weights, eps);
    for (i, &val) in large_out.iter().enumerate() {
        assert!(val.is_finite(), "large input produced non-finite at index {}", i);
    }
});
