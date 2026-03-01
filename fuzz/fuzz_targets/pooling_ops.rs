#![no_main]

//! Fuzz target: average-pooling and max-pooling with arbitrary shapes.
//!
//! Self-contained pure-Rust implementations are fuzzed here so that the
//! target compiles without requiring a real pooling crate.
//!
//! Invariants verified:
//!   1. Neither avg_pool1d nor max_pool1d ever panics for valid (non-empty)
//!      window ≤ input length.
//!   2. Output length equals `(input_len - window) / stride + 1`.
//!   3. Every avg-pool output is ≥ min(input) and ≤ max(input) (inclusive).
//!   4. Every max-pool output equals the maximum of the corresponding window.
//!   5. Window size == 1 and stride == 1 → output equals input.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    /// Raw float bytes (4 bytes per element).
    data_bytes: Vec<u8>,
    /// Pooling window size (clamped to 1..=input_len).
    window_hint: u8,
    /// Stride (clamped to 1..=window).
    stride_hint: u8,
}

/// Average pooling over a 1-D slice.
fn avg_pool1d(data: &[f32], window: usize, stride: usize) -> Vec<f32> {
    if window == 0 || stride == 0 || data.len() < window {
        return vec![];
    }
    let out_len = (data.len() - window) / stride + 1;
    (0..out_len)
        .map(|i| {
            let start = i * stride;
            let s: f32 = data[start..start + window].iter().sum();
            s / window as f32
        })
        .collect()
}

/// Max pooling over a 1-D slice.
fn max_pool1d(data: &[f32], window: usize, stride: usize) -> Vec<f32> {
    if window == 0 || stride == 0 || data.len() < window {
        return vec![];
    }
    let out_len = (data.len() - window) / stride + 1;
    (0..out_len)
        .map(|i| {
            let start = i * stride;
            data[start..start + window]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect()
}

fuzz_target!(|input: Input| {
    // Build f32 data from raw bytes.
    let aligned = (input.data_bytes.len() / 4) * 4;
    let data: Vec<f32> = input.data_bytes[..aligned]
        .chunks_exact(4)
        .take(128) // cap at 128 elements to bound runtime
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    if data.is_empty() {
        return;
    }

    // Clamp window and stride to valid ranges.
    let window = ((input.window_hint as usize % data.len()) + 1).max(1);
    let stride = ((input.stride_hint as usize % window) + 1).max(1);

    let expected_len = (data.len() - window) / stride + 1;

    // --- Invariant 1-2: avg_pool1d ---
    let avg_out = avg_pool1d(&data, window, stride);
    assert_eq!(
        avg_out.len(),
        expected_len,
        "avg_pool1d length mismatch: input_len={}, window={window}, stride={stride}",
        data.len()
    );

    // --- Invariant 3: avg output in [min, max] of input (for finite inputs) ---
    let all_finite = data.iter().all(|v| v.is_finite());
    if all_finite {
        let global_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let global_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for (i, &v) in avg_out.iter().enumerate() {
            assert!(
                v >= global_min - 1e-5 && v <= global_max + 1e-5,
                "avg_pool1d[{i}]={v} outside [{global_min}, {global_max}]"
            );
        }
    }

    // --- Invariant 1-2: max_pool1d ---
    let max_out = max_pool1d(&data, window, stride);
    assert_eq!(
        max_out.len(),
        expected_len,
        "max_pool1d length mismatch: input_len={}, window={window}, stride={stride}",
        data.len()
    );

    // --- Invariant 4: each max-pool output == actual max of window ---
    for (i, &v) in max_out.iter().enumerate() {
        let start = i * stride;
        let win_slice = &data[start..start + window];
        let actual_max = win_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // NaN-aware comparison: if actual_max is NaN, max_pool must also be NaN.
        if actual_max.is_nan() {
            assert!(v.is_nan(), "max_pool1d[{i}] should be NaN when window contains NaN");
        } else {
            assert_eq!(
                v, actual_max,
                "max_pool1d[{i}]={v} != window_max={actual_max}"
            );
        }
    }

    // --- Invariant 5: window=1, stride=1 → identity ---
    let identity_avg = avg_pool1d(&data, 1, 1);
    let identity_max = max_pool1d(&data, 1, 1);
    assert_eq!(identity_avg.len(), data.len());
    assert_eq!(identity_max.len(), data.len());
    for (i, (&a, &d)) in identity_avg.iter().zip(data.iter()).enumerate() {
        if d.is_nan() {
            assert!(a.is_nan(), "avg identity[{i}]: expected NaN, got {a}");
        } else {
            assert_eq!(a, d, "avg identity[{i}]: {a} != {d}");
        }
    }
    for (i, (&m, &d)) in identity_max.iter().zip(data.iter()).enumerate() {
        if d.is_nan() {
            assert!(m.is_nan(), "max identity[{i}]: expected NaN, got {m}");
        } else {
            assert_eq!(m, d, "max identity[{i}]: {m} != {d}");
        }
    }
});
