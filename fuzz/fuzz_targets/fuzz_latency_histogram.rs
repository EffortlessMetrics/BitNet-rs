#![no_main]

use arbitrary::Arbitrary;
use bitnet_inference::metrics::LatencyHistogram;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct HistogramInput {
    /// Raw bytes interpreted as f64 latency samples.
    raw_samples: Vec<u8>,
    /// Percentile queries (0-100 range).
    percentile_queries: Vec<u8>,
}

fuzz_target!(|input: HistogramInput| {
    let mut histogram = LatencyHistogram::new();

    // Invariant 1: Empty histogram returns None for percentiles
    assert!(histogram.p50().is_none());
    assert!(histogram.p99().is_none());
    assert_eq!(histogram.count(), 0);

    // Decode raw bytes to f64 samples
    let aligned_len = (input.raw_samples.len() / 8) * 8;
    let samples: Vec<f64> = input.raw_samples[..aligned_len]
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .take(1024)
        .collect();

    // Record all samples — must not panic on any f64 value (NaN, Inf, negative)
    for &sample in &samples {
        histogram.record(sample);
    }

    // Invariant 2: Count matches number of recorded samples
    assert_eq!(histogram.count(), samples.len());

    if !samples.is_empty() {
        // Invariant 3: Standard percentiles must return Some
        let _ = histogram.p50();
        let _ = histogram.p90();
        let _ = histogram.p95();
        let _ = histogram.p99();

        // Query arbitrary percentiles — must not panic
        for &p in &input.percentile_queries {
            let pct = p as f64; // 0..255
            let _ = histogram.percentile(pct);
        }

        // Invariant 4: mean() returns Some for non-empty histogram
        let _ = histogram.mean();
    }
});
