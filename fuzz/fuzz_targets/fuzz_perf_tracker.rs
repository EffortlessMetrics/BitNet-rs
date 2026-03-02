//! Fuzz perf tracker with random kernel timings and throughput values.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct PerfInput {
    prefill_ms: Option<u64>,
    decode_ms: Option<u64>,
    tokenization_encode_ms: Option<u64>,
    tokenization_decode_ms: Option<u64>,
    total_ms: u64,
    prefill_tps: Option<f64>,
    decode_tps: Option<f64>,
    e2e_tps: f64,
    total_tokens: usize,
}

fuzz_target!(|input: PerfInput| {
    use bitnet_inference_metrics_core::{ThroughputMetrics, TimingMetrics};

    // Build timing metrics — must not panic
    let timing = TimingMetrics {
        prefill_ms: input.prefill_ms,
        decode_ms: input.decode_ms,
        tokenization_encode_ms: input.tokenization_encode_ms,
        tokenization_decode_ms: input.tokenization_decode_ms,
        total_ms: input.total_ms,
    };

    // Build throughput metrics with potentially adversarial floats (NaN, Inf)
    let throughput = ThroughputMetrics {
        prefill_tokens_per_sec: input.prefill_tps,
        decode_tokens_per_sec: input.decode_tps,
        end_to_end_tokens_per_sec: input.e2e_tps,
        total_tokens: input.total_tokens,
    };

    // Serde roundtrip — must not panic
    if let Ok(json) = serde_json::to_string(&timing) {
        let _: Result<TimingMetrics, _> = serde_json::from_str(&json);
    }
    if let Ok(json) = serde_json::to_string(&throughput) {
        let _: Result<ThroughputMetrics, _> = serde_json::from_str(&json);
    }

    // Clone and equality — must not panic
    let _t2 = timing.clone();
    let _p2 = throughput.clone();
    let _ = timing == _t2;
    let _ = throughput == _p2;

    // Debug formatting — must not panic
    let _ = format!("{timing:?}");
    let _ = format!("{throughput:?}");

    // Default — must not panic
    let _ = TimingMetrics::default();
    let _ = ThroughputMetrics::default();
});
