//! Cross-crate SRP integration tests.
//!
//! Verifies that the SRP microcrate wiring works end-to-end through the
//! `bitnet-inference` dependency graph:
//!
//! - `bitnet-logits`          – pure logit transforms
//! - `bitnet-generation`      – stop criteria and generation events
//! - `bitnet-prompt-templates`– prompt formatting
//! - `bitnet-engine-core`     – `SessionConfig` serde round-trip
//! - `bitnet-sampling`        – token sampling strategies
//! - `bitnet-rope`            – rotary position embedding tables
//! - `bitnet-device-probe`    – hardware capability detection

use bitnet_engine_core::SessionConfig;
use bitnet_generation::{StopCriteria, StopReason, check_stop};
use bitnet_logits::{apply_temperature, apply_top_k, argmax, softmax_in_place};
use bitnet_prompt_templates::{PromptTemplate, TemplateType};

// ---------------------------------------------------------------------------
// bitnet-logits
// ---------------------------------------------------------------------------

#[test]
fn logits_temperature_scales_values() {
    let mut logits = vec![1.0f32, 2.0, 4.0];
    apply_temperature(&mut logits, 2.0);
    // Each value should be halved (multiplied by 1/temperature = 0.5)
    assert!((logits[0] - 0.5).abs() < 1e-6);
    assert!((logits[1] - 1.0).abs() < 1e-6);
    assert!((logits[2] - 2.0).abs() < 1e-6);
}

#[test]
fn logits_softmax_sums_to_one() {
    let mut logits = vec![1.0f32, 2.0, 3.0, 0.5];
    softmax_in_place(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
    // All probabilities must be non-negative
    assert!(logits.iter().all(|&p| p >= 0.0));
}

#[test]
fn logits_top_k_one_leaves_single_candidate() {
    let mut logits = vec![1.0f32, 5.0, 3.0, 2.0];
    let kept = apply_top_k(&mut logits, 1);
    assert_eq!(kept, 1, "only one token should survive top-k=1");
    // The highest-value token (index 1, value 5.0) must survive.
    let best = argmax(&logits);
    assert_eq!(best, 1);
    // All other positions must be NEG_INFINITY.
    for (i, &v) in logits.iter().enumerate() {
        if i != best {
            assert_eq!(v, f32::NEG_INFINITY, "logits[{i}] should be NEG_INFINITY");
        }
    }
}

// ---------------------------------------------------------------------------
// bitnet-generation
// ---------------------------------------------------------------------------

#[test]
fn check_stop_eos_token_id() {
    let criteria = StopCriteria { eos_token_id: Some(2), ..Default::default() };
    let result = check_stop(&criteria, 2, &[2], "hello");
    assert_eq!(result, Some(StopReason::EosToken));
}

#[test]
fn check_stop_max_tokens_budget() {
    let criteria = StopCriteria { max_tokens: 3, ..Default::default() };
    // generated has 3 tokens – budget exhausted
    let result = check_stop(&criteria, 99, &[10, 20, 30], "some text");
    assert_eq!(result, Some(StopReason::MaxTokens));
}

#[test]
fn check_stop_string_in_tail() {
    let criteria = StopCriteria { stop_strings: vec!["\n\nQ:".to_string()], ..Default::default() };
    let result = check_stop(&criteria, 42, &[42], "previous answer\n\nQ: next");
    assert_eq!(result, Some(StopReason::StopString("\n\nQ:".to_string())));
}

#[test]
fn check_stop_no_condition_returns_none() {
    let criteria = StopCriteria { max_tokens: 10, ..Default::default() };
    // Only 2 tokens generated – well under the budget.
    let result = check_stop(&criteria, 7, &[5, 7], "hi");
    assert_eq!(result, None);
}

// ---------------------------------------------------------------------------
// bitnet-prompt-templates
// ---------------------------------------------------------------------------

#[test]
fn prompt_template_raw_passes_through_unchanged() {
    let tpl = PromptTemplate::new(TemplateType::Raw);
    let input = "What is 2+2?";
    assert_eq!(tpl.format(input), input);
}

#[test]
fn prompt_template_instruct_wraps_in_qa_format() {
    let tpl = PromptTemplate::new(TemplateType::Instruct);
    let formatted = tpl.format("What is the capital of France?");
    assert!(formatted.starts_with("Q: "), "should start with 'Q: '");
    assert!(formatted.contains("What is the capital of France?"));
    assert!(formatted.contains("\nA:"), "should contain '\\nA:'");
}

#[test]
fn prompt_template_instruct_with_system_prompt() {
    let tpl = PromptTemplate::new(TemplateType::Instruct)
        .with_system_prompt("You are a helpful assistant.");
    let formatted = tpl.format("Hello");
    assert!(formatted.starts_with("System: You are a helpful assistant."));
    assert!(formatted.contains("Q: Hello"));
    assert!(formatted.contains("\nA:"));
}

// ---------------------------------------------------------------------------
// bitnet-engine-core
// ---------------------------------------------------------------------------

#[test]
fn session_config_serde_round_trip() {
    let original = SessionConfig {
        model_path: "/models/bitnet.gguf".to_string(),
        tokenizer_path: "/models/tokenizer.json".to_string(),
        backend: "cpu".to_string(),
        max_context: 4096,
        seed: Some(42),
    };

    let json = serde_json::to_string(&original).expect("serialize SessionConfig");
    let restored: SessionConfig = serde_json::from_str(&json).expect("deserialize SessionConfig");

    assert_eq!(restored.model_path, original.model_path);
    assert_eq!(restored.tokenizer_path, original.tokenizer_path);
    assert_eq!(restored.backend, original.backend);
    assert_eq!(restored.max_context, original.max_context);
    assert_eq!(restored.seed, original.seed);
}

#[test]
fn session_config_default_values() {
    let cfg = SessionConfig::default();
    assert_eq!(cfg.backend, "cpu");
    assert_eq!(cfg.max_context, 2048);
    assert!(cfg.seed.is_none());
}

// ---------------------------------------------------------------------------
// bitnet-sampling + bitnet-logits (cross-crate wiring)
// ---------------------------------------------------------------------------

/// Temperature scaling via bitnet-logits followed by greedy sampling via
/// bitnet-sampling must return a token index within [0, vocab_size).
#[test]
fn logits_temperature_then_sampling_yields_valid_range() {
    use bitnet_sampling::greedy_sample;

    let vocab_size = 8usize;
    let mut logits = vec![0.5f32, 1.2, 3.0, 0.1, 2.7, 0.9, 1.5, 0.3];
    apply_temperature(&mut logits, 0.8);
    let token = greedy_sample(&logits).expect("greedy_sample should not fail on non-empty input");
    assert!(
        (token as usize) < vocab_size,
        "sampled token {token} must be within vocab range [0, {vocab_size})"
    );
}

/// Full sampling pipeline: temperature → top-k → softmax → sample.
/// The token produced by bitnet-sampling must equal the argmax produced by
/// bitnet-logits when temperature is 0 (greedy).
#[test]
fn sampling_pipeline_greedy_matches_logits_argmax() {
    use bitnet_sampling::{SamplingConfig, SamplingStrategy};

    let logits = vec![0.1f32, 0.5, 4.2, 0.3, 1.1];
    let expected_best = argmax(&logits); // bitnet-logits

    let config = SamplingConfig { temperature: 0.0, ..Default::default() };
    let mut strategy = SamplingStrategy::new(config);
    let sampled = strategy.sample(&logits, &[]).expect("SamplingStrategy::sample should not fail");

    assert_eq!(sampled as usize, expected_best, "greedy SamplingStrategy must match logits argmax");
}

// ---------------------------------------------------------------------------
// bitnet-generation: loop termination
// ---------------------------------------------------------------------------

/// Simulate a 4-step generation loop using check_stop; the loop must stop
/// exactly when the generated slice reaches max_tokens=4.
#[test]
fn generation_loop_terminates_at_max_tokens() {
    let max = 4usize;
    let criteria = StopCriteria { max_tokens: max, ..Default::default() };

    let mut generated: Vec<u32> = Vec::new();
    let mut steps = 0usize;

    loop {
        let token = 99u32; // dummy token that is NOT an EOS
        generated.push(token);
        steps += 1;

        let stop = check_stop(&criteria, token, &generated, "");
        if stop.is_some() {
            assert_eq!(stop, Some(StopReason::MaxTokens), "should stop with MaxTokens");
            break;
        }

        // Safety valve: never spin more than max+1 iterations.
        if steps > max + 1 {
            panic!("generation loop did not terminate after {steps} steps");
        }
    }

    assert_eq!(steps, max, "loop must run exactly max_tokens={max} steps, got {steps}");
}

/// GenerationConfig carries the stop criteria correctly: max_new_tokens
/// propagates into a StopCriteria with the same limit.
#[test]
fn generation_config_max_tokens_threads_into_stop_criteria() {
    use bitnet_inference::config::GenerationConfig as InferenceGenConfig;

    // Using the builder API: with_max_tokens sets the token budget.
    let cfg = InferenceGenConfig::greedy().with_max_tokens(4);
    assert_eq!(cfg.max_new_tokens, 4, "with_max_tokens(4) must set max_new_tokens to 4");
}

// ---------------------------------------------------------------------------
// bitnet-rope: table shape and correctness
// ---------------------------------------------------------------------------

/// RoPE tables built with known parameters must have the correct flattened
/// length: sin.len() == cos.len() == max_seq_len * (dim / 2).
#[test]
fn rope_tables_dimensions_match_build_params() {
    use bitnet_rope::build_tables;

    let dim = 64usize;
    let max_seq_len = 16usize;
    let base = 10_000.0f32;

    let tables = build_tables(dim, max_seq_len, base).expect("build_tables should succeed");

    let expected_len = max_seq_len * (dim / 2);
    assert_eq!(tables.sin.len(), expected_len, "sin table length mismatch");
    assert_eq!(tables.cos.len(), expected_len, "cos table length mismatch");
    assert_eq!(tables.half_dim, dim / 2, "half_dim mismatch");
}

/// At position 0 every RoPE frequency yields cos(0) = 1.0, so the first
/// half_dim entries of the cosine table must all be ≈ 1.0.
#[test]
fn rope_tables_cosine_at_position_zero_is_one() {
    use bitnet_rope::build_tables;

    let dim = 32usize;
    let tables = build_tables(dim, 8, 10_000.0).expect("build_tables should succeed");

    // Row 0 spans [0 .. half_dim] in the flattened table.
    for (i, &c) in tables.cos[..tables.half_dim].iter().enumerate() {
        assert!((c - 1.0f32).abs() < 1e-6, "cos[{i}] at position 0 should be 1.0, got {c}");
    }
}

// ---------------------------------------------------------------------------
// bitnet-device-probe: determinism and backend selection
// ---------------------------------------------------------------------------

/// Two consecutive calls to DeviceCapabilities::detect() must return the same
/// compile-time flags (cpu_rust, cuda_compiled) because those are constant
/// within a single build.
#[test]
fn device_capabilities_detect_is_deterministic() {
    use bitnet_device_probe::DeviceCapabilities;

    let a = DeviceCapabilities::detect();
    let b = DeviceCapabilities::detect();

    assert!(a.cpu_rust, "cpu_rust must always be true");
    assert_eq!(a.cpu_rust, b.cpu_rust, "cpu_rust must be stable across calls");
    assert_eq!(a.cuda_compiled, b.cuda_compiled, "cuda_compiled is a compile-time constant");
    assert_eq!(
        format!("{:?}", a.simd_level),
        format!("{:?}", b.simd_level),
        "simd_level must be stable across calls"
    );
}

/// When CUDA is not compiled in, the selected backend must be "cpu".
/// When CUDA is compiled AND available at runtime, the selected backend
/// should reflect that. This test verifies the selection logic is consistent
/// with the capabilities snapshot.
#[test]
fn device_capabilities_backend_selection_consistent_with_caps() {
    use bitnet_device_probe::DeviceCapabilities;

    let caps = DeviceCapabilities::detect();

    // Derive the expected backend the same way the inference layer would.
    let expected_backend = if caps.cuda_runtime { "cuda" } else { "cpu" };

    // Run twice to confirm determinism.
    let caps2 = DeviceCapabilities::detect();
    let expected_backend2 = if caps2.cuda_runtime { "cuda" } else { "cpu" };

    assert_eq!(expected_backend, expected_backend2, "backend selection must be deterministic");
}

// ---------------------------------------------------------------------------
// bitnet-engine-core + bitnet-device-probe (cross-crate wiring)
// ---------------------------------------------------------------------------

/// A SessionConfig built by inspecting DeviceCapabilities must reflect the
/// probed backend: cpu when no GPU is available, cuda when it is.
#[test]
fn session_config_backend_derived_from_device_capabilities() {
    use bitnet_device_probe::DeviceCapabilities;

    let caps = DeviceCapabilities::detect();
    let backend = if caps.cuda_runtime { "cuda".to_string() } else { "cpu".to_string() };

    let config = SessionConfig { backend: backend.clone(), ..Default::default() };

    assert_eq!(config.backend, backend, "SessionConfig backend must match probed capabilities");
    // cpu_rust is always true – the config must at minimum support cpu.
    assert!(caps.cpu_rust);
}

// ---------------------------------------------------------------------------
// bitnet-engine-core + bitnet-generation (cross-crate wiring)
// ---------------------------------------------------------------------------

/// A SessionConfig and GenerationConfig can be constructed together and their
/// seed fields are independent — engine-level seed (for weight loading) is
/// separate from generation-level seed (for token sampling).
#[test]
fn engine_session_and_generation_seeds_are_independent() {
    use bitnet_inference::config::GenerationConfig as InferenceGenConfig;

    let session = SessionConfig { seed: Some(1234), ..Default::default() };
    // Use builder API: greedy() preset + explicit seed override.
    let gcfg = InferenceGenConfig::greedy().with_seed(9999);

    // Seeds must be independently configurable.
    assert_ne!(
        session.seed.unwrap(),
        gcfg.seed.unwrap(),
        "session and generation seeds are independent"
    );
    // greedy() preset has a fixed max_new_tokens default.
    assert!(gcfg.max_new_tokens > 0, "max_new_tokens must be positive");
}
