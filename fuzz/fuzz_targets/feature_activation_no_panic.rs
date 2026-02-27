#![no_main]

use arbitrary::Arbitrary;
use bitnet_runtime_feature_flags_core::{
    FeatureActivation, active_features_from_activation, feature_labels_from_activation,
    feature_line_from_activation,
};
use libfuzzer_sys::fuzz_target;

/// Proxy struct that derives `Arbitrary` and maps 1:1 to `FeatureActivation`.
#[derive(Arbitrary, Debug)]
struct ActivationInput {
    cpu: bool,
    gpu: bool,
    cuda: bool,
    inference: bool,
    kernels: bool,
    tokenizers: bool,
    quantization: bool,
    cli: bool,
    server: bool,
    ffi: bool,
    python: bool,
    wasm: bool,
    crossval: bool,
    trace: bool,
    iq2s_ffi: bool,
    cpp_ffi: bool,
    fixtures: bool,
    reporting: bool,
    trend: bool,
    integration_tests: bool,
}

fuzz_target!(|input: ActivationInput| {
    let activation = FeatureActivation {
        cpu: input.cpu,
        gpu: input.gpu,
        cuda: input.cuda,
        inference: input.inference,
        kernels: input.kernels,
        tokenizers: input.tokenizers,
        quantization: input.quantization,
        cli: input.cli,
        server: input.server,
        ffi: input.ffi,
        python: input.python,
        wasm: input.wasm,
        crossval: input.crossval,
        trace: input.trace,
        iq2s_ffi: input.iq2s_ffi,
        cpp_ffi: input.cpp_ffi,
        fixtures: input.fixtures,
        reporting: input.reporting,
        trend: input.trend,
        integration_tests: input.integration_tests,
    };

    // All three conversion functions must not panic for any combination of bool flags.
    let features = active_features_from_activation(activation);
    let labels = feature_labels_from_activation(activation);
    let line = feature_line_from_activation(activation);

    // Sanity: labels and FeatureSet must agree on count.
    assert_eq!(
        features.labels().len(),
        labels.len(),
        "feature label count mismatch: FeatureSet={}, labels={}",
        features.labels().len(),
        labels.len(),
    );

    // line must be non-empty (it always contains the word "features").
    assert!(!line.is_empty(), "feature_line_from_activation returned empty string");
});
