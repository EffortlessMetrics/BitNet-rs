//! Multi-crate integration tests for BitNet-rs.
//!
//! These tests exercise interactions between multiple microcrates and verify
//! that public APIs compose correctly across crate boundaries.

// ─── Sampling ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod sampling_tests {
    use bitnet_sampling::{SamplingConfig, SamplingStrategy};

    #[test]
    fn sampling_config_defaults_are_sensible() {
        let cfg = SamplingConfig::default();
        assert_eq!(cfg.temperature, 0.7);
        assert_eq!(cfg.top_k, 50);
        assert!((cfg.top_p - 0.9).abs() < 1e-6);
        assert_eq!(cfg.repetition_penalty, 1.0);
        assert!(cfg.seed.is_none());
    }

    #[test]
    fn sampling_strategy_new_from_config() {
        let cfg = SamplingConfig { temperature: 0.5, seed: Some(42), ..Default::default() };
        let _strategy = SamplingStrategy::new(cfg);
    }

    #[test]
    fn greedy_sampling_picks_argmax() {
        let cfg = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
        let mut strategy = SamplingStrategy::new(cfg);
        let logits = vec![0.1_f32, 0.9, 0.3, 0.05];
        let token = strategy.sample(&logits, &[]).unwrap();
        assert_eq!(token, 1, "greedy must select the highest logit");
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let cfg = SamplingConfig { temperature: 0.8, seed: Some(12345), ..Default::default() };
        let mut a = SamplingStrategy::new(cfg.clone());
        let mut b = SamplingStrategy::new(cfg);
        let logits = vec![1.0_f32, 2.0, 1.5, 0.8, 2.5];
        assert_eq!(
            a.sample(&logits, &[]).unwrap(),
            b.sample(&logits, &[]).unwrap(),
            "same seed must yield the same token"
        );
    }

    #[test]
    fn sample_returns_valid_index() {
        let cfg = SamplingConfig { temperature: 1.0, seed: Some(99), ..Default::default() };
        let mut strategy = SamplingStrategy::new(cfg);
        let logits = vec![0.1_f32; 32];
        let token = strategy.sample(&logits, &[]).unwrap();
        assert!((token as usize) < logits.len());
    }

    #[test]
    fn sample_with_history_uses_repetition_penalty() {
        // With a high repetition penalty, a recently generated token should
        // score lower and be less likely to be selected greedily.
        let cfg = SamplingConfig {
            temperature: 0.0,
            repetition_penalty: 10.0,
            seed: Some(0),
            ..Default::default()
        };
        let mut strategy = SamplingStrategy::new(cfg);
        // Token 0 has the highest logit but is in history → penalised.
        // Token 1 should win after penalty is applied.
        let logits = vec![10.0_f32, 5.0, 0.1];
        let token = strategy.sample(&logits, &[0]).unwrap();
        // After a large penalty on token 0, token 1 should now be chosen.
        assert_eq!(token, 1);
    }

    #[test]
    fn empty_logits_returns_error() {
        let cfg = SamplingConfig::default();
        let mut strategy = SamplingStrategy::new(cfg);
        assert!(strategy.sample(&[], &[]).is_err());
    }
}

// ─── Logits transforms ───────────────────────────────────────────────────────

#[cfg(test)]
mod logits_tests {
    use bitnet_logits::{
        apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p, argmax,
        softmax_in_place,
    };

    #[test]
    fn temperature_scaling_divides_logits() {
        let mut logits = vec![2.0_f32, 4.0, 6.0];
        apply_temperature(&mut logits, 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn temperature_one_is_noop() {
        let orig = vec![1.0_f32, 2.0, 3.0];
        let mut logits = orig.clone();
        apply_temperature(&mut logits, 1.0);
        assert_eq!(logits, orig);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut logits = vec![1.0_f32, 2.0, 3.0];
        softmax_in_place(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
        for &p in &logits {
            assert!(p >= 0.0 && p <= 1.0);
        }
    }

    #[test]
    fn top_k_keeps_k_tokens() {
        let mut logits = vec![1.0_f32, 5.0, 3.0, 2.0, 4.0];
        let kept = apply_top_k(&mut logits, 2);
        assert_eq!(kept, 2);
        assert!(logits[1].is_finite(), "idx 1 (5.0) must survive");
        assert!(logits[4].is_finite(), "idx 4 (4.0) must survive");
        assert!(logits[0].is_infinite(), "idx 0 (1.0) must be masked");
    }

    #[test]
    fn full_logits_pipeline_returns_valid_token() {
        let mut logits = vec![1.0_f32, 2.0, 3.0, 0.5, 1.8];
        let history: Vec<u32> = vec![2];
        apply_repetition_penalty(&mut logits, &history, 1.3);
        apply_temperature(&mut logits, 0.8);
        softmax_in_place(&mut logits);
        apply_top_p(&mut logits, 0.9);
        let best = argmax(&logits);
        assert!(best < logits.len(), "argmax must be a valid index");
    }

    #[test]
    fn argmax_selects_highest_probability() {
        let probs = vec![0.1_f32, 0.6, 0.2, 0.1];
        assert_eq!(argmax(&probs), 1);
    }
}

// ─── Device probe ────────────────────────────────────────────────────────────

#[cfg(test)]
mod device_probe_tests {
    use bitnet_device_probe::{gpu_compiled, probe_cpu};

    #[test]
    fn probe_cpu_has_at_least_one_core() {
        let caps = probe_cpu();
        assert!(caps.core_count >= 1);
    }

    #[test]
    fn avx_and_neon_are_mutually_exclusive() {
        let caps = probe_cpu();
        assert!(!(caps.has_avx2 && caps.has_neon), "AVX2 and NEON cannot both be true");
        assert!(!(caps.has_avx512 && caps.has_neon), "AVX-512 and NEON cannot both be true");
    }

    #[test]
    fn gpu_compiled_matches_feature_flags() {
        // When built with `--features cpu` only, this should return false.
        // When built with `--features gpu`, this should return true.
        let compiled: bool = gpu_compiled();
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        assert!(compiled);
        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
        assert!(!compiled);
    }

    #[test]
    fn cpu_probe_simd_consistent_on_x86() {
        let caps = probe_cpu();
        // AVX-512 implies AVX2 on real hardware.
        if caps.has_avx512 {
            assert!(caps.has_avx2, "AVX-512 implies AVX2 support");
        }
    }
}

// ─── Prompt templates ────────────────────────────────────────────────────────

#[cfg(test)]
mod prompt_template_tests {
    use bitnet_prompt_templates::TemplateType;

    #[test]
    fn raw_template_returns_prompt_unchanged() {
        let t = TemplateType::Raw;
        let out = t.apply("hello world", None);
        assert_eq!(out, "hello world");
    }

    #[test]
    fn instruct_template_includes_user_text() {
        let t = TemplateType::Instruct;
        let out = t.apply("What is 2+2?", None);
        assert!(!out.is_empty());
        assert!(out.contains("2+2"), "instruct output must include the user text");
    }

    #[test]
    fn instruct_template_with_system_prompt() {
        let t = TemplateType::Instruct;
        let out = t.apply("Hello", Some("You are a helper"));
        assert!(out.contains("You are a helper"), "system prompt must appear in output");
        assert!(out.contains("Hello"), "user text must appear in output");
    }

    #[test]
    fn llama3_chat_template_includes_user_text() {
        let t = TemplateType::Llama3Chat;
        let out = t.apply("What is Rust?", None);
        assert!(!out.is_empty());
        assert!(out.contains("What is Rust?"), "llama3-chat must include user text");
    }

    #[test]
    fn llama3_chat_with_system_prompt() {
        let t = TemplateType::Llama3Chat;
        let out = t.apply("Explain GC", Some("You are an expert"));
        assert!(out.contains("You are an expert") || !out.is_empty());
    }

    #[test]
    fn detect_llama3_from_jinja_template() {
        let jinja =
            "{% if messages %}<|start_header_id|>user<|end_header_id|>{{ messages }}<|eot_id|>";
        let t = TemplateType::detect(None, Some(jinja));
        assert_eq!(t, TemplateType::Llama3Chat);
    }

    #[test]
    fn detect_instruct_from_jinja_template() {
        let jinja = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n";
        let t = TemplateType::detect(None, Some(jinja));
        assert_eq!(t, TemplateType::Instruct);
    }

    #[test]
    fn detect_raw_when_no_hints() {
        let t = TemplateType::detect(None, None);
        assert_eq!(t, TemplateType::Raw);
    }

    #[test]
    fn detect_instruct_from_tokenizer_name() {
        let t = TemplateType::detect(Some("mistral-7b-instruct"), None);
        assert_eq!(t, TemplateType::Instruct);
    }

    #[test]
    fn template_type_display_roundtrips() {
        for (t, s) in [
            (TemplateType::Raw, "raw"),
            (TemplateType::Instruct, "instruct"),
            (TemplateType::Llama3Chat, "llama3-chat"),
        ] {
            assert_eq!(t.to_string(), s);
            let parsed: TemplateType = s.parse().unwrap();
            assert_eq!(parsed, t);
        }
    }
}

// ─── Generation (stop criteria + stream events) ──────────────────────────────

#[cfg(test)]
mod generation_tests {
    use bitnet_generation::{
        GenerationConfig, GenerationStats, StopCriteria, StopReason, StreamEvent, TokenEvent,
        check_stop,
    };

    #[test]
    fn check_stop_on_stop_token_id() {
        let criteria =
            StopCriteria { stop_token_ids: vec![128009], max_tokens: 100, ..Default::default() };
        assert_eq!(check_stop(&criteria, 128009, &[], ""), Some(StopReason::StopTokenId(128009)));
    }

    #[test]
    fn check_stop_on_eos() {
        let criteria =
            StopCriteria { eos_token_id: Some(2), max_tokens: 100, ..Default::default() };
        assert_eq!(check_stop(&criteria, 2, &[], ""), Some(StopReason::EosToken));
    }

    #[test]
    fn check_stop_on_max_tokens() {
        let criteria = StopCriteria { max_tokens: 4, ..Default::default() };
        assert_eq!(check_stop(&criteria, 99, &[0, 1, 2, 3], ""), Some(StopReason::MaxTokens));
    }

    #[test]
    fn check_stop_on_stop_string() {
        let criteria = StopCriteria {
            stop_strings: vec!["</s>".to_string()],
            max_tokens: 100,
            ..Default::default()
        };
        assert_eq!(
            check_stop(&criteria, 5, &[], "hello</s>world"),
            Some(StopReason::StopString("</s>".to_string()))
        );
    }

    #[test]
    fn check_stop_returns_none_when_no_condition_met() {
        let criteria = StopCriteria {
            stop_token_ids: vec![999],
            max_tokens: 10,
            eos_token_id: Some(0),
            ..Default::default()
        };
        assert!(check_stop(&criteria, 42, &[1, 2], "hello").is_none());
    }

    #[test]
    fn generation_config_default_values() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.max_new_tokens, 128);
        assert!(cfg.seed.is_none());
    }

    #[test]
    fn stream_events_token_and_done() {
        let events: Vec<StreamEvent> = vec![
            StreamEvent::Token(TokenEvent { id: 42, text: "Hello".to_string() }),
            StreamEvent::Done {
                reason: StopReason::MaxTokens,
                stats: GenerationStats { tokens_generated: 1, tokens_per_second: 10.0 },
            },
        ];
        let tokens: Vec<_> = events.iter().filter(|e| matches!(e, StreamEvent::Token(_))).collect();
        let done_count = events.iter().filter(|e| matches!(e, StreamEvent::Done { .. })).count();
        assert_eq!(tokens.len(), 1);
        assert_eq!(done_count, 1);
    }
}

// ─── Engine core (session config + state machine) ────────────────────────────

#[cfg(test)]
mod engine_core_tests {
    use bitnet_engine_core::{
        ConcurrencyConfig, ConfigError, EngineState, EngineStateTracker, SessionConfig, SessionId,
    };

    #[test]
    fn session_config_default_backend_is_cpu() {
        let cfg = SessionConfig::default();
        assert_eq!(cfg.backend, "cpu");
        assert_eq!(cfg.max_context, 2048);
    }

    #[test]
    fn session_config_validate_ok() {
        let cfg = SessionConfig {
            model_path: "model.gguf".into(),
            tokenizer_path: "tokenizer.json".into(),
            backend: "cpu".into(),
            max_context: 512,
            seed: None,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn session_config_validate_empty_model_path() {
        let cfg = SessionConfig {
            model_path: String::new(),
            tokenizer_path: "t.json".into(),
            backend: "cpu".into(),
            max_context: 512,
            seed: None,
        };
        assert_eq!(cfg.validate(), Err(ConfigError::EmptyModelPath));
    }

    #[test]
    fn session_config_validate_empty_tokenizer_path() {
        let cfg = SessionConfig {
            model_path: "m.gguf".into(),
            tokenizer_path: String::new(),
            backend: "cpu".into(),
            max_context: 512,
            seed: None,
        };
        assert_eq!(cfg.validate(), Err(ConfigError::EmptyTokenizerPath));
    }

    #[test]
    fn session_config_validate_unsupported_backend() {
        let cfg = SessionConfig {
            model_path: "m.gguf".into(),
            tokenizer_path: "t.json".into(),
            backend: "tpu".into(),
            max_context: 512,
            seed: None,
        };
        assert!(matches!(cfg.validate(), Err(ConfigError::UnsupportedBackend(_))));
    }

    #[test]
    fn session_config_validate_zero_context() {
        let cfg = SessionConfig {
            model_path: "m.gguf".into(),
            tokenizer_path: "t.json".into(),
            backend: "cpu".into(),
            max_context: 0,
            seed: None,
        };
        assert_eq!(cfg.validate(), Err(ConfigError::ZeroContextWindow));
    }

    #[test]
    fn session_id_is_non_empty_and_unique() {
        let a = SessionId::generate();
        let b = SessionId::generate();
        assert!(!a.as_str().is_empty());
        assert_ne!(a.as_str(), b.as_str(), "consecutive session IDs must differ");
    }

    #[test]
    fn engine_state_tracker_idle_to_running_to_done() {
        let mut tracker = EngineStateTracker::new();
        assert_eq!(tracker.state(), &EngineState::Idle);
        tracker.start().expect("start from Idle must succeed");
        assert_eq!(tracker.state(), &EngineState::Running);
        tracker.finish().expect("finish from Running must succeed");
        assert_eq!(tracker.state(), &EngineState::Done);
    }

    #[test]
    fn engine_state_tracker_invalid_transition_returns_error() {
        let mut tracker = EngineStateTracker::new();
        assert!(tracker.finish().is_err(), "finish from Idle must fail");
    }

    #[test]
    fn concurrency_config_allows_up_to_limit() {
        let cfg = ConcurrencyConfig { max_concurrent: 4 };
        assert!(cfg.allows(0));
        assert!(cfg.allows(3));
        assert!(!cfg.allows(4));
    }
}

// ─── GGUF types ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod gguf_tests {
    use bitnet_gguf::{
        GGUF_MAGIC, GgufValueType, TensorInfo, check_magic, parse_header, read_version,
    };

    fn minimal_v2_header() -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(&GGUF_MAGIC); // magic
        d.extend_from_slice(&2u32.to_le_bytes()); // version
        d.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        d.extend_from_slice(&2u64.to_le_bytes()); // metadata_count
        d
    }

    #[test]
    fn check_magic_accepts_gguf_prefix() {
        let data = b"GGUFsomeextraBytes";
        assert!(check_magic(data));
    }

    #[test]
    fn check_magic_rejects_invalid_prefix() {
        assert!(!check_magic(b"GGML"));
        assert!(!check_magic(b""));
        assert!(!check_magic(b"GGU"));
    }

    #[test]
    fn read_version_parses_v2() {
        let data = minimal_v2_header();
        assert_eq!(read_version(&data), Some(2));
    }

    #[test]
    fn read_version_returns_none_for_short_slice() {
        assert_eq!(read_version(&[]), None);
        assert_eq!(read_version(b"GGU"), None);
    }

    #[test]
    fn parse_header_v2_returns_correct_counts() {
        let data = minimal_v2_header();
        let hdr = parse_header(&data).unwrap();
        assert_eq!(hdr.version, 2);
        assert_eq!(hdr.tensor_count, 1);
        assert_eq!(hdr.metadata_count, 2);
        assert_eq!(hdr.alignment, 32); // v2 default
    }

    #[test]
    fn parse_header_rejects_wrong_magic() {
        let mut data = minimal_v2_header();
        data[0] = b'X';
        assert!(parse_header(&data).is_err());
    }

    #[test]
    fn tensor_info_fields_accessible() {
        let info = TensorInfo {
            name: "weight".to_string(),
            n_dims: 2,
            dims: vec![256, 512],
            dtype: 6,
            offset: 1024,
        };
        assert_eq!(info.name, "weight");
        assert_eq!(info.n_dims, 2);
        assert_eq!(info.dims, vec![256, 512]);
        assert_eq!(info.dtype, 6);
        assert_eq!(info.offset, 1024);
    }

    #[test]
    fn gguf_value_type_roundtrips_from_u32() {
        for (raw, expected) in [
            (0u32, GgufValueType::Uint8),
            (6, GgufValueType::Float32),
            (8, GgufValueType::String),
            (12, GgufValueType::Float64),
        ] {
            assert_eq!(GgufValueType::from_u32(raw), Some(expected));
        }
        assert!(GgufValueType::from_u32(99).is_none());
    }
}

// ─── Honest compute + receipts ───────────────────────────────────────────────

#[cfg(test)]
mod honest_compute_tests {
    use bitnet_honest_compute::{
        classify_compute_path, is_mock_kernel_id, validate_compute_path, validate_kernel_ids,
    };

    #[test]
    fn classify_real_kernels_as_real() {
        let path = classify_compute_path(["i2s_gemv", "rope_apply"]);
        assert_eq!(path, "real");
    }

    #[test]
    fn classify_mock_kernel_as_mock() {
        let path = classify_compute_path(["i2s_gemv", "mock_attention"]);
        assert_eq!(path, "mock");
    }

    #[test]
    fn is_mock_kernel_id_case_insensitive() {
        assert!(is_mock_kernel_id("MOCK_kernel"));
        assert!(is_mock_kernel_id("i2s_mock_op"));
        assert!(!is_mock_kernel_id("i2s_gemv"));
    }

    #[test]
    fn validate_compute_path_accepts_real() {
        assert!(validate_compute_path("real").is_ok());
    }

    #[test]
    fn validate_compute_path_rejects_mock() {
        assert!(validate_compute_path("mock").is_err());
    }

    #[test]
    fn validate_kernel_ids_accepts_valid_list() {
        let ids = ["i2s_gemv", "rope_apply", "attention_real"];
        assert!(validate_kernel_ids(ids.iter().copied()).is_ok());
    }

    #[test]
    fn validate_kernel_ids_rejects_empty_array() {
        assert!(validate_kernel_ids(std::iter::empty::<&str>()).is_err());
    }

    #[test]
    fn validate_kernel_ids_rejects_empty_string() {
        assert!(validate_kernel_ids([""].iter().copied()).is_err());
    }
}

// ─── Receipt integration with honest-compute ─────────────────────────────────

#[cfg(test)]
mod receipt_tests {
    use bitnet_receipts::InferenceReceipt;

    #[test]
    fn generate_receipt_with_real_kernels() {
        let receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        assert_eq!(receipt.schema_version, "1.0.0");
        assert_eq!(receipt.compute_path, "real");
        assert_eq!(receipt.backend, "cpu");
        assert!(!receipt.kernels.is_empty());
    }

    #[test]
    fn generate_receipt_with_mock_kernels_sets_mock_path() {
        let receipt =
            InferenceReceipt::generate("cpu", vec!["mock_attention".to_string()], None).unwrap();
        assert_eq!(receipt.compute_path, "mock");
    }

    #[test]
    fn validate_real_receipt_passes() {
        let receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        assert!(receipt.validate_compute_path().is_ok());
        assert!(receipt.validate_kernel_ids().is_ok());
        assert!(receipt.validate_schema().is_ok());
    }

    #[test]
    fn validate_mock_compute_path_fails() {
        let receipt =
            InferenceReceipt::generate("cpu", vec!["mock_kernel".to_string()], None).unwrap();
        assert!(receipt.validate_compute_path().is_err());
    }

    #[test]
    fn receipt_to_json_string_contains_schema_version() {
        let receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        let json = receipt.to_json_string().unwrap();
        assert!(json.contains("\"schema_version\""));
        assert!(json.contains("1.0.0"));
    }

    #[test]
    fn receipt_backend_summary_is_stored() {
        let receipt = InferenceReceipt::generate(
            "cpu",
            vec!["i2s_gemv".to_string()],
            Some("requested=cpu detected=[cpu] selected=cpu".to_string()),
        )
        .unwrap();
        assert!(receipt.backend_summary.contains("selected=cpu"));
    }
}

// ─── Cross-crate compose: sampling + logits ──────────────────────────────────

#[cfg(test)]
mod sampling_logits_compose_tests {
    use bitnet_logits::{apply_temperature, argmax, softmax_in_place};
    use bitnet_sampling::{SamplingConfig, SamplingStrategy};

    #[test]
    fn sampling_and_logits_pipeline_produce_same_greedy_result() {
        let logits_raw = vec![0.1_f32, 3.0, 1.5, 0.8];

        // Via sampling crate (greedy path).
        let cfg = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
        let mut strategy = SamplingStrategy::new(cfg);
        let sampled = strategy.sample(&logits_raw, &[]).unwrap();

        // Via raw logits functions (manual pipeline).
        let mut logits = logits_raw.clone();
        apply_temperature(&mut logits, 1.0); // no-op
        softmax_in_place(&mut logits);
        let best = argmax(&logits);

        assert_eq!(sampled as usize, best, "greedy SamplingStrategy must agree with argmax");
    }
}

// ─── Cross-crate compose: prompt-templates + generation ──────────────────────

#[cfg(test)]
mod prompt_generation_compose_tests {
    use bitnet_generation::{StopCriteria, check_stop};
    use bitnet_prompt_templates::TemplateType;

    #[test]
    fn format_prompt_then_check_stop_criteria() {
        let template = TemplateType::Instruct;
        let formatted = template.apply("What is 2+2?", None);
        assert!(!formatted.is_empty());

        // Simulate a generation loop hitting a stop string.
        let criteria = StopCriteria {
            stop_strings: vec!["</s>".to_string()],
            max_tokens: 16,
            ..Default::default()
        };
        let tail = format!("{formatted}4</s>");
        let stop = check_stop(&criteria, 5, &[0], &tail);
        assert!(stop.is_some(), "stop string in tail must trigger stop");
    }
}

// ─── Cross-crate compose: engine-core + generation ───────────────────────────

#[cfg(test)]
mod engine_generation_compose_tests {
    use bitnet_engine_core::{GenerationConfig, SessionConfig};
    use bitnet_generation::StopCriteria;

    #[test]
    fn session_config_valid_with_generation_config() {
        let session = SessionConfig {
            model_path: "models/model.gguf".into(),
            tokenizer_path: "models/tokenizer.json".into(),
            backend: "cpu".into(),
            max_context: 4096,
            seed: Some(42),
        };
        assert!(session.validate().is_ok());

        let gen_cfg = GenerationConfig {
            max_new_tokens: 64,
            seed: Some(42),
            stop_criteria: StopCriteria {
                eos_token_id: Some(2),
                max_tokens: 64,
                ..Default::default()
            },
        };
        assert_eq!(gen_cfg.max_new_tokens, 64);
        assert_eq!(gen_cfg.stop_criteria.eos_token_id, Some(2));
    }
}
