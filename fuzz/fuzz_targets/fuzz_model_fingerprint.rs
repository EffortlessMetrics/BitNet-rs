#![no_main]

//! Fuzz model fingerprint construction and matching: exercises `ModelFingerprint`
//! builder methods, compact_id generation, size labelling, and cross-fingerprint
//! comparison to ensure no panics on arbitrary architecture strings and parameters.

use arbitrary::Arbitrary;
use bitnet_models::model_fingerprint::ModelFingerprint;
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

#[derive(Arbitrary, Debug)]
struct FingerprintPair {
    a: FingerprintInput,
    b: FingerprintInput,
}

fn build_fingerprint(input: &FingerprintInput) -> ModelFingerprint {
    let mut fp = ModelFingerprint::new(&input.architecture)
        .with_param_count(input.param_count)
        .with_layers(input.num_layers)
        .with_hidden_size(input.hidden_size)
        .with_heads(input.num_heads)
        .with_vocab_size(input.vocab_size)
        .with_quant_type(&input.quant_type);

    for (k, v) in input.tags.iter().take(16) {
        fp = fp.with_tag(k, v);
    }
    fp
}

fuzz_target!(|input: FingerprintPair| {
    let fp_a = build_fingerprint(&input.a);
    let fp_b = build_fingerprint(&input.b);

    // compact_id must not panic.
    let id_a = fp_a.compact_id();
    let id_b = fp_b.compact_id();
    let _ = (id_a, id_b);

    // Comparison methods must not panic.
    let _ = fp_a.same_architecture(&fp_b);
    let _ = fp_a.same_model_different_quant(&fp_b);

    // Derived metrics must not panic.
    let _ = fp_a.estimated_weight_bytes();
    let _ = fp_a.is_quantized();
    let _ = fp_a.size_label();
    let _ = fp_b.estimated_weight_bytes();
    let _ = fp_b.is_quantized();
    let _ = fp_b.size_label();

    // same_architecture must be reflexive.
    assert!(fp_a.same_architecture(&fp_a));

    // Debug formatting must not panic.
    let _ = format!("{:?}", fp_a);
    let _ = format!("{:?}", fp_b);
});
