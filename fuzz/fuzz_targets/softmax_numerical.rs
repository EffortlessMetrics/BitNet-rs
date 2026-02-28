#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct SoftmaxInput {
    data: Vec<f32>,
}

fn ref_softmax(input: &[f32]) -> Vec<f32> {
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        vec![1.0 / input.len() as f32; input.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

fuzz_target!(|input: SoftmaxInput| {
    if input.data.is_empty() || input.data.len() > 256 {
        return;
    }

    // Clamp to avoid extreme values that cause exp() overflow
    let clamped: Vec<f32> = input
        .data
        .iter()
        .take(256)
        .map(|&x| if x.is_finite() { x.clamp(-1e38, 1e38) } else { 0.0 })
        .collect();

    let output = ref_softmax(&clamped);

    // Verify no NaN in output
    for (i, &val) in output.iter().enumerate() {
        assert!(
            !val.is_nan(),
            "NaN in softmax output at index {} for input len {}",
            i,
            clamped.len()
        );
    }

    // Verify sum is approximately 1.0
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "softmax sum is {}, expected ~1.0 for input len {}",
        sum,
        clamped.len()
    );

    // Verify all values in [0, 1]
    for (i, &val) in output.iter().enumerate() {
        assert!(val >= 0.0 && val <= 1.0, "softmax[{}] = {} not in [0,1]", i, val);
    }

    // Test with extreme inputs: all same value
    if clamped.len() >= 2 {
        let uniform = vec![clamped[0]; clamped.len()];
        let uniform_out = ref_softmax(&uniform);
        let expected = 1.0 / uniform.len() as f32;
        for &val in &uniform_out {
            assert!((val - expected).abs() < 1e-5, "uniform input should give uniform output");
        }
    }

    // Test with subnormal-like inputs (near zero)
    let tiny: Vec<f32> = clamped.iter().map(|_| 1e-38f32).collect();
    let tiny_out = ref_softmax(&tiny);
    let tiny_sum: f32 = tiny_out.iter().sum();
    assert!(
        (tiny_sum - 1.0).abs() < 1e-4,
        "softmax of subnormals should still sum to 1.0, got {}",
        tiny_sum
    );
});
