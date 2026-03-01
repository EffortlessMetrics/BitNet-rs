#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct SoftmaxInput {
    rows: u8,
    cols: u8,
    data: Vec<f32>,
}

/// Numerically-stable row-wise softmax (mirrors kernel implementation).
fn softmax_row(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

fuzz_target!(|input: SoftmaxInput| {
    let rows = (input.rows as usize).clamp(1, 16);
    let cols = (input.cols as usize).clamp(1, 64);
    let total = rows * cols;

    let mut data: Vec<f32> = input
        .data
        .iter()
        .copied()
        .take(total)
        .chain(std::iter::repeat_n(0.0f32, total.saturating_sub(input.data.len())))
        .take(total)
        .collect();

    for r in 0..rows {
        let row = &mut data[r * cols..(r + 1) * cols];
        softmax_row(row);

        // All values in [0, 1]
        for &v in row.iter() {
            assert!(v >= 0.0 && v <= 1.0 + 1e-6, "softmax value out of range: {v}");
        }

        // Row sums to ~1.0
        let sum: f32 = row.iter().sum();
        if sum > 0.0 {
            assert!((sum - 1.0).abs() < 1e-5, "softmax row sum not 1.0: {sum}");
        }

        // No NaN
        for &v in row.iter() {
            assert!(!v.is_nan(), "softmax produced NaN");
        }
    }

    // Edge case: single-element rows
    let mut single = vec![42.0f32];
    softmax_row(&mut single);
    assert!((single[0] - 1.0).abs() < 1e-6, "single-element softmax should be 1.0");

    // Edge case: identical elements produce uniform distribution
    let mut uniform = vec![7.0f32; cols];
    softmax_row(&mut uniform);
    let expected = 1.0 / cols as f32;
    for &v in &uniform {
        assert!(
            (v - expected).abs() < 1e-5,
            "uniform input should give uniform output: {v} vs {expected}"
        );
    }
});
