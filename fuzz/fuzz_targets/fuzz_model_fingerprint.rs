//! Fuzz model fingerprint computation with random metadata.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct FingerprintInput {
    architecture: String,
    param_count: u64,
    num_layers: u32,
    hidden_size: u32,
    num_heads: u32,
    vocab_size: u32,
    quant_type: String,
    tags: Vec<(String, String)>,
}

fuzz_target!(|input: FingerprintInput| {
    use bitnet_models::model_fingerprint::{ModelFingerprint, identify_model, known_fingerprints};

    // Cap string lengths to avoid OOM
    let arch: String = input.architecture.chars().take(64).collect();
    let qt: String = input.quant_type.chars().take(32).collect();

    // Build via builder chain — must not panic
    let mut fp = ModelFingerprint::new(&arch)
        .with_param_count(input.param_count)
        .with_layers(input.num_layers)
        .with_hidden_size(input.hidden_size)
        .with_heads(input.num_heads)
        .with_vocab_size(input.vocab_size)
        .with_quant_type(&qt);

    for (k, v) in input.tags.iter().take(16) {
        let key: String = k.chars().take(32).collect();
        let val: String = v.chars().take(64).collect();
        fp = fp.with_tag(&key, &val);
    }

    // Derived computations — must not panic
    let _ = fp.compact_id();
    let _ = fp.estimated_weight_bytes();
    let _ = fp.is_quantized();
    let _ = fp.size_label();
    let _ = format!("{fp}");
    let _ = format!("{fp:?}");

    // Identification — must not panic
    let _ = identify_model(&fp);

    // Cross-comparison with known fingerprints
    let known = known_fingerprints();
    for k in &known {
        let _ = fp.same_architecture(k);
        let _ = fp.same_model_different_quant(k);
    }

    // Clone + equality — must not panic
    let fp2 = fp.clone();
    let _ = fp == fp2;
});
