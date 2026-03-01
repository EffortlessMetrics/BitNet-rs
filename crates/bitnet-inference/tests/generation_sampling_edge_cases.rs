//! Edge-case tests for GenerationConfig, InferenceConfig, SamplingConfig,
//! and SamplingStrategy covering validation, builder chains, presets,
//! stop token management, and sampling behavior.

use bitnet_inference::config::{GenerationConfig, InferenceConfig};
use bitnet_inference::sampling::{SamplingConfig, SamplingStrategy};
use bitnet_sampling::{greedy_sample, temperature_sample};

// ── GenerationConfig defaults ────────────────────────────────────────────

#[test]
fn gen_config_default_values() {
    let cfg = GenerationConfig::default();
    assert_eq!(cfg.max_new_tokens, 100);
    assert!((cfg.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(cfg.top_k, 50);
    assert!((cfg.top_p - 0.9).abs() < f32::EPSILON);
    assert!((cfg.repetition_penalty - 1.0).abs() < f32::EPSILON);
    assert!(cfg.stop_sequences.is_empty());
    assert!(cfg.stop_token_ids.is_empty());
    assert_eq!(cfg.seed, None);
    assert!(cfg.skip_special_tokens);
    assert_eq!(cfg.eos_token_id, None);
    assert!(!cfg.add_bos);
}

// ── GenerationConfig presets ─────────────────────────────────────────────

#[test]
fn gen_config_greedy_preset() {
    let cfg = GenerationConfig::greedy();
    assert!((cfg.temperature - 0.0).abs() < f32::EPSILON);
    assert_eq!(cfg.top_k, 1);
    assert!((cfg.top_p - 1.0).abs() < f32::EPSILON);
}

#[test]
fn gen_config_creative_preset() {
    let cfg = GenerationConfig::creative();
    assert!(cfg.temperature > 0.7);
    assert!(cfg.top_k > 50);
    assert!(cfg.repetition_penalty > 1.0);
}

#[test]
fn gen_config_balanced_preset() {
    let cfg = GenerationConfig::balanced();
    assert!((cfg.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(cfg.top_k, 50);
    assert!(cfg.repetition_penalty > 1.0);
}

// ── GenerationConfig validation ──────────────────────────────────────────

#[test]
fn gen_config_validates_default() {
    assert!(GenerationConfig::default().validate().is_ok());
}

#[test]
fn gen_config_validates_greedy() {
    assert!(GenerationConfig::greedy().validate().is_ok());
}

#[test]
fn gen_config_validates_creative() {
    assert!(GenerationConfig::creative().validate().is_ok());
}

#[test]
fn gen_config_rejects_zero_max_tokens() {
    let cfg = GenerationConfig::default().with_max_tokens(0);
    assert!(cfg.validate().is_err());
}

#[test]
fn gen_config_rejects_negative_temperature() {
    let cfg = GenerationConfig::default().with_temperature(-0.1);
    assert!(cfg.validate().is_err());
}

#[test]
fn gen_config_rejects_zero_top_p() {
    let cfg = GenerationConfig::default().with_top_p(0.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn gen_config_rejects_top_p_over_one() {
    let cfg = GenerationConfig::default().with_top_p(1.1);
    assert!(cfg.validate().is_err());
}

#[test]
fn gen_config_rejects_zero_repetition_penalty() {
    let cfg = GenerationConfig::default().with_repetition_penalty(0.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn gen_config_rejects_negative_repetition_penalty() {
    let cfg = GenerationConfig::default().with_repetition_penalty(-1.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn gen_config_accepts_top_p_exactly_one() {
    let cfg = GenerationConfig::default().with_top_p(1.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn gen_config_accepts_zero_temperature() {
    let cfg = GenerationConfig::default().with_temperature(0.0);
    assert!(cfg.validate().is_ok());
}

// ── GenerationConfig builder chain ───────────────────────────────────────

#[test]
fn gen_config_builder_chain() {
    let cfg = GenerationConfig::default()
        .with_seed(42)
        .with_max_tokens(200)
        .with_temperature(0.5)
        .with_top_k(20)
        .with_top_p(0.8)
        .with_repetition_penalty(1.2)
        .with_stop_string_window(128)
        .with_skip_special_tokens(false)
        .with_add_bos(true)
        .with_eos_token_id(Some(2));

    assert_eq!(cfg.seed, Some(42));
    assert_eq!(cfg.max_new_tokens, 200);
    assert!((cfg.temperature - 0.5).abs() < f32::EPSILON);
    assert_eq!(cfg.top_k, 20);
    assert!((cfg.top_p - 0.8).abs() < f32::EPSILON);
    assert!((cfg.repetition_penalty - 1.2).abs() < f32::EPSILON);
    assert_eq!(cfg.stop_string_window, 128);
    assert!(!cfg.skip_special_tokens);
    assert!(cfg.add_bos);
    assert_eq!(cfg.eos_token_id, Some(2));
}

// ── Stop sequences ───────────────────────────────────────────────────────

#[test]
fn gen_config_stop_sequence_single() {
    let cfg = GenerationConfig::default().with_stop_sequence("<|end|>".to_string());
    assert_eq!(cfg.stop_sequences, vec!["<|end|>"]);
}

#[test]
fn gen_config_stop_sequences_multiple() {
    let cfg = GenerationConfig::default()
        .with_stop_sequences(vec!["<|end|>", "<|eot|>", "\n\n"].into_iter().map(String::from));
    assert_eq!(cfg.stop_sequences.len(), 3);
}

// ── Stop token IDs ───────────────────────────────────────────────────────

#[test]
fn gen_config_stop_token_ids() {
    let cfg = GenerationConfig::default().with_stop_token_ids(vec![2, 100257, 100265]);
    assert!(cfg.is_stop_token(2));
    assert!(cfg.is_stop_token(100257));
    assert!(cfg.is_stop_token(100265));
    assert!(!cfg.is_stop_token(0));
}

#[test]
fn gen_config_stop_token_single() {
    let cfg = GenerationConfig::default().with_stop_token_id(2);
    assert!(cfg.is_stop_token(2));
    assert!(!cfg.is_stop_token(1));
}

#[test]
fn gen_config_stop_token_rebuild_set() {
    let mut cfg = GenerationConfig::default();
    cfg.stop_token_ids = vec![10, 20, 30];
    // Set not yet built
    assert!(!cfg.is_stop_token(10));
    cfg.rebuild_stop_token_set();
    assert!(cfg.is_stop_token(10));
    assert!(cfg.is_stop_token(20));
    assert!(cfg.is_stop_token(30));
}

#[test]
fn gen_config_stop_token_empty_by_default() {
    let cfg = GenerationConfig::default();
    assert!(!cfg.is_stop_token(0));
    assert!(!cfg.is_stop_token(u32::MAX));
}

// ── GenerationConfig serialization ───────────────────────────────────────

#[test]
fn gen_config_serde_roundtrip() {
    let cfg = GenerationConfig::default()
        .with_seed(42)
        .with_max_tokens(50)
        .with_stop_sequence("stop".to_string());
    let json = serde_json::to_string(&cfg).expect("serialize");
    let cfg2: GenerationConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cfg2.seed, Some(42));
    assert_eq!(cfg2.max_new_tokens, 50);
    assert_eq!(cfg2.stop_sequences, vec!["stop"]);
}

#[test]
fn gen_config_debug_format() {
    let cfg = GenerationConfig::default();
    let dbg = format!("{:?}", cfg);
    assert!(dbg.contains("GenerationConfig"));
    assert!(dbg.contains("max_new_tokens"));
    assert!(dbg.contains("temperature"));
}

// ── InferenceConfig defaults ─────────────────────────────────────────────

#[test]
fn inf_config_default_values() {
    let cfg = InferenceConfig::default();
    assert_eq!(cfg.max_context_length, 2048);
    assert!(cfg.num_threads >= 1);
    assert_eq!(cfg.batch_size, 1);
    assert!(!cfg.mixed_precision);
    assert_eq!(cfg.memory_pool_size, 512 * 1024 * 1024);
}

// ── InferenceConfig presets ──────────────────────────────────────────────

#[test]
fn inf_config_cpu_optimized() {
    let cfg = InferenceConfig::cpu_optimized();
    assert!(cfg.num_threads >= 1);
    assert!(!cfg.mixed_precision);
    assert_eq!(cfg.batch_size, 1);
}

#[test]
fn inf_config_gpu_optimized() {
    let cfg = InferenceConfig::gpu_optimized();
    assert!(cfg.mixed_precision);
    assert!(cfg.batch_size > 1);
    assert!(cfg.memory_pool_size >= 1024 * 1024 * 1024);
}

#[test]
fn inf_config_memory_efficient() {
    let cfg = InferenceConfig::memory_efficient();
    assert!(cfg.max_context_length <= 1024);
    assert_eq!(cfg.batch_size, 1);
    assert!(cfg.memory_pool_size <= 256 * 1024 * 1024);
}

// ── InferenceConfig validation ───────────────────────────────────────────

#[test]
fn inf_config_validates_default() {
    assert!(InferenceConfig::default().validate().is_ok());
}

#[test]
fn inf_config_rejects_zero_context() {
    let cfg = InferenceConfig { max_context_length: 0, ..Default::default() };
    assert!(cfg.validate().is_err());
}

#[test]
fn inf_config_rejects_zero_threads() {
    let cfg = InferenceConfig { num_threads: 0, ..Default::default() };
    assert!(cfg.validate().is_err());
}

#[test]
fn inf_config_rejects_zero_batch_size() {
    let cfg = InferenceConfig { batch_size: 0, ..Default::default() };
    assert!(cfg.validate().is_err());
}

#[test]
fn inf_config_rejects_zero_memory_pool() {
    let cfg = InferenceConfig { memory_pool_size: 0, ..Default::default() };
    assert!(cfg.validate().is_err());
}

// ── InferenceConfig builder chain ────────────────────────────────────────

#[test]
fn inf_config_builder_chain() {
    let cfg = InferenceConfig::default()
        .with_threads(8)
        .with_batch_size(4)
        .with_mixed_precision(true)
        .with_memory_pool_size(2 * 1024 * 1024 * 1024);

    assert_eq!(cfg.num_threads, 8);
    assert_eq!(cfg.batch_size, 4);
    assert!(cfg.mixed_precision);
    assert_eq!(cfg.memory_pool_size, 2 * 1024 * 1024 * 1024);
}

// ── SamplingConfig defaults ──────────────────────────────────────────────

#[test]
fn sampling_config_defaults() {
    let cfg = SamplingConfig::default();
    assert!((cfg.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(cfg.top_k, 50);
    assert!((cfg.top_p - 0.9).abs() < f32::EPSILON);
    assert!((cfg.repetition_penalty - 1.0).abs() < f32::EPSILON);
    assert_eq!(cfg.seed, None);
}

// ── SamplingStrategy creation ────────────────────────────────────────────

#[test]
fn sampling_strategy_creation_with_seed() {
    let cfg = SamplingConfig { seed: Some(42), ..Default::default() };
    let _strategy = SamplingStrategy::new(cfg);
}

#[test]
fn sampling_strategy_creation_without_seed() {
    let cfg = SamplingConfig::default();
    let _strategy = SamplingStrategy::new(cfg);
}

// ── Greedy sampling ──────────────────────────────────────────────────────

#[test]
fn greedy_sample_picks_argmax() {
    let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
    let token = greedy_sample(&logits).unwrap();
    assert_eq!(token, 3); // index of 0.9
}

#[test]
fn greedy_sample_first_max_on_tie() {
    let logits = vec![0.5, 0.5, 0.5];
    let token = greedy_sample(&logits).unwrap();
    assert_eq!(token, 0); // first occurrence
}

#[test]
fn greedy_sample_single_element() {
    let logits = vec![1.0];
    let token = greedy_sample(&logits).unwrap();
    assert_eq!(token, 0);
}

#[test]
fn greedy_sample_negative_logits() {
    let logits = vec![-1.0, -0.5, -2.0];
    let token = greedy_sample(&logits).unwrap();
    assert_eq!(token, 1); // -0.5 is the max
}

#[test]
fn greedy_sample_empty_logits_is_err() {
    let result = greedy_sample(&[]);
    assert!(result.is_err());
}

// ── Sampling determinism ─────────────────────────────────────────────────

#[test]
fn sampling_strategy_deterministic_with_seed() {
    let make_strategy = || {
        let cfg =
            SamplingConfig { temperature: 0.7, top_k: 10, seed: Some(42), ..Default::default() };
        SamplingStrategy::new(cfg)
    };

    let logits: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();

    let mut s1 = make_strategy();
    let mut s2 = make_strategy();

    let t1 = s1.sample(&logits, &[]).unwrap();
    let t2 = s2.sample(&logits, &[]).unwrap();
    assert_eq!(t1, t2);
}

#[test]
fn sampling_strategy_reset_clears_state() {
    let cfg = SamplingConfig { seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);

    let logits: Vec<f32> = (0..50).map(|i| (i as f32 * 0.2).cos()).collect();
    let _ = strategy.sample(&logits, &[]).unwrap();
    strategy.reset();

    // After reset, should behave as if fresh (though RNG state may differ)
    let _ = strategy.sample(&logits, &[]).unwrap();
}

#[test]
fn sampling_strategy_update_config() {
    let cfg = SamplingConfig { temperature: 0.5, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);

    let new_cfg = SamplingConfig { temperature: 1.0, seed: Some(99), ..Default::default() };
    strategy.update_config(new_cfg);

    // Should still be able to sample
    let logits: Vec<f32> = vec![0.1, 0.9, 0.5];
    let _ = strategy.sample(&logits, &[]).unwrap();
}

// ── Temperature sampling ─────────────────────────────────────────────────

#[test]
fn temperature_sample_valid_output() {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let token = temperature_sample(&logits, 0.7, &mut rng).unwrap();
    assert!(token < 5);
}

#[test]
fn temperature_sample_empty_is_err() {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let result = temperature_sample(&[], 0.7, &mut rng);
    assert!(result.is_err());
}
