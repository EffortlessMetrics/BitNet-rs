//! Fuzz weight statistics analyzer with random tensor data.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct WeightStatsInput {
    name: String,
    shape: Vec<u16>,
    /// Raw bytes reinterpreted as f32 values — gives us NaN, Inf, subnormals, etc.
    raw_bytes: Vec<u8>,
}

fuzz_target!(|input: WeightStatsInput| {
    use bitnet_models::weight_stats::{TensorStats, detect_anomalies, generate_report};

    let name: String = input.name.chars().take(64).collect();
    let shape: Vec<usize> = input.shape.iter().take(6).map(|&v| v as usize).collect();

    // Reinterpret raw bytes as f32 — gives adversarial float values
    let data: Vec<f32> = input
        .raw_bytes
        .chunks_exact(4)
        .take(4096) // cap at 4K elements to limit time
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // from_f32 — must not panic on NaN/Inf/subnormal/zero data
    let stats = TensorStats::from_f32(&name, &shape, &data);

    // Derived metrics — must not panic
    let _ = stats.std_dev();
    let _ = stats.has_anomalies();
    let _ = stats.sparsity();
    let _ = stats.is_sparse();
    let _ = format!("{stats}");
    let _ = format!("{stats:?}");

    // Anomaly detection — must not panic
    let anomalies = detect_anomalies(&stats);
    for a in &anomalies {
        let _ = format!("{a}");
    }

    // Report generation with single tensor
    let report = generate_report(&[stats.clone()]);
    let _ = format!("{report:?}");
    assert!(report.tensor_count == 1);

    // Empty report
    let empty = generate_report(&[]);
    assert!(empty.tensor_count == 0);
});
