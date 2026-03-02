#![no_main]

use arbitrary::Arbitrary;
use bitnet_models::capability_check::{ModelCapability, check_requirements, detect_capabilities};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct CapabilityInput {
    model_family: String,
    hidden_size: usize,
    num_layers: usize,
    vocab_size: usize,
    /// Extra model family variants to test.
    extra_families: Vec<String>,
}

fuzz_target!(|input: CapabilityInput| {
    // Invariant 1: detect_capabilities must never panic on arbitrary strings
    let report = detect_capabilities(&input.model_family);

    // Invariant 2: TextGeneration is always present
    assert!(report.has(ModelCapability::TextGeneration), "TextGeneration must always be present");

    // Invariant 3: model_family is preserved in report
    assert_eq!(report.model_family, input.model_family);

    // Invariant 4: can_chat and can_code must not panic
    let _ = report.can_chat();
    let _ = report.can_code();

    // Invariant 5: capability names must not panic
    let all_caps = [
        ModelCapability::TextGeneration,
        ModelCapability::ChatCompletion,
        ModelCapability::CodeGeneration,
        ModelCapability::FillInMiddle,
        ModelCapability::Embedding,
        ModelCapability::Classification,
        ModelCapability::ToolUse,
        ModelCapability::VisionInput,
        ModelCapability::AudioInput,
    ];
    for cap in &all_caps {
        let name = cap.name();
        assert!(!name.is_empty());
    }

    // Invariant 6: check_requirements must never panic with arbitrary values
    let issues = check_requirements(input.hidden_size, input.num_layers, input.vocab_size);
    // Issues list must be a valid Vec (no panic on access)
    let _ = issues.len();

    // Invariant 7: Boundary values for requirements
    let _ = check_requirements(0, 0, 0);
    let _ = check_requirements(usize::MAX, usize::MAX, usize::MAX);
    let _ = check_requirements(1, 1, 1);
    let _ = check_requirements(64, 1, 100);

    // Invariant 8: Fuzz additional model families
    for family in input.extra_families.iter().take(8) {
        let r = detect_capabilities(family);
        assert!(r.has(ModelCapability::TextGeneration));
    }
});
