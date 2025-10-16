//! Prompt Template Auto-Detection Tests
//!
//! Tests acceptance criteria AC4 and AC7 for the CLI UX Improvements specification.
//! Verifies automatic template detection from GGUF metadata and tokenizer hints.
//!
//! # Specification References
//! - AC4: Automatic Template Detection
//! - AC7: Template Detection Tests
//! - Spec: docs/explanation/cli-ux-improvements-spec.md
//! - ADR: docs/explanation/architecture/adr-014-prompt-template-auto-detection.md

use anyhow::Result;
use bitnet_inference::TemplateType;

#[cfg(test)]
mod ac7_pattern_matching {
    use super::*;

    // AC7:llama3 - Test LLaMA-3 detection from chat_template metadata
    #[test]
    fn test_matches_llama3_pattern() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC7:llama3
        // Verify that LLaMA-3 template patterns are correctly detected

        // Test with LLaMA-3 special tokens
        let chat_template = "<|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>";

        let detected = TemplateType::detect(None, Some(chat_template));
        assert_eq!(detected, TemplateType::Llama3Chat);

        // Test with minimal LLaMA-3 pattern
        let minimal_template = "something <|start_header_id|> and <|eot_id|> here";
        let detected_minimal = TemplateType::detect(None, Some(minimal_template));
        assert_eq!(detected_minimal, TemplateType::Llama3Chat);

        Ok(())
    }

    // AC7:llama3_full - Test complete LLaMA-3 template detection
    #[test]
    fn test_llama3_full_template_pattern() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC7
        // ADR: adr-014-prompt-template-auto-detection.md
        // Verify all LLaMA-3 special tokens are recognized

        let chat_template = concat!(
            "<|begin_of_text|>",
            "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
            "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        );

        // Verify all distinctive tokens are present
        assert!(chat_template.contains("<|start_header_id|>"));
        assert!(chat_template.contains("<|end_header_id|>"));
        assert!(chat_template.contains("<|eot_id|>"));

        // Verify detection works
        let detected = TemplateType::detect(None, Some(chat_template));
        assert_eq!(detected, TemplateType::Llama3Chat);

        Ok(())
    }

    // AC7:instruct - Test instruct detection from template patterns
    #[test]
    fn test_matches_instruct_pattern() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC7:instruct
        // Verify that instruct template patterns are correctly detected

        // Test with generic Jinja loop pattern
        let chat_template1 = "{% for message in messages %}Q: {user_text}\nA:{% endfor %}";
        let detected1 = TemplateType::detect(None, Some(chat_template1));
        assert_eq!(detected1, TemplateType::Instruct);

        // Test with another Jinja loop pattern
        let chat_template2 = "### Instruction\n{% for message in messages %}{instruction}{% endfor %}\n\n### Response";
        let detected2 = TemplateType::detect(None, Some(chat_template2));
        assert_eq!(detected2, TemplateType::Instruct);

        Ok(())
    }

    // AC7:instruct_variants - Test various instruct format variants
    #[test]
    fn test_instruct_pattern_variants() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC7
        // Verify that different instruct format variants are detected

        let patterns = vec![
            "{% for message in messages %}Q: {question}\nA:{% endfor %}",
            "{% for message in messages %}### Instruction\n{instruction}\n\n### Response{% endfor %}",
            "{% for message in messages %}Human: {input}\n\nAssistant:{% endfor %}",
        ];

        // All patterns should be recognized as instruct
        for pattern in patterns {
            let detected = TemplateType::detect(None, Some(pattern));
            assert_eq!(
                detected,
                TemplateType::Instruct,
                "Failed to match instruct pattern: {}",
                pattern
            );
        }

        Ok(())
    }

    // AC7:fallback - Test fallback to raw when no patterns match
    #[test]
    fn test_no_pattern_fallback_to_raw() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC7:fallback
        // Verify that unknown templates fall back to Raw

        // Test with non-matching template
        let chat_template = "Just plain text {placeholder}";

        // Should not match LLaMA-3 or instruct patterns
        let detected = TemplateType::detect(None, Some(chat_template));
        assert_eq!(detected, TemplateType::Raw);

        // Test with completely empty/unknown pattern
        let unknown_template = "some random template without special tokens";
        let detected2 = TemplateType::detect(None, Some(unknown_template));
        assert_eq!(detected2, TemplateType::Raw);

        Ok(())
    }
}

#[cfg(test)]
mod ac4_template_detection {
    use super::*;

    // AC4:llama3 - Detects llama3-chat from chat_template metadata
    #[test]
    fn test_detect_llama3_from_chat_template() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC4:llama3
        // ADR: adr-014-prompt-template-auto-detection.md
        // Verify LLaMA-3 detection from GGUF chat_template metadata

        // Test with LLaMA-3 chat_template (GGUF metadata - priority 1)
        let jinja = "<|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>";

        let detected = TemplateType::detect(None, Some(jinja));
        assert_eq!(detected, TemplateType::Llama3Chat);

        Ok(())
    }

    // AC4:instruct - Detects instruct from tokenizer family hints
    #[test]
    fn test_detect_instruct_from_family() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC4:instruct
        // ADR: adr-014-prompt-template-auto-detection.md
        // Verify instruct detection from tokenizer family name

        // Test with mistral-instruct tokenizer name (priority 2)
        let detected1 = TemplateType::detect(Some("mistral-instruct"), None);
        assert_eq!(detected1, TemplateType::Instruct);

        // Test with generic instruct hint
        let detected2 = TemplateType::detect(Some("some-instruct-model"), None);
        assert_eq!(detected2, TemplateType::Instruct);

        // Test with mistral (also maps to instruct)
        let detected3 = TemplateType::detect(Some("mistral-7b"), None);
        assert_eq!(detected3, TemplateType::Instruct);

        Ok(())
    }

    // AC4:fallback - Falls back to raw if no hints
    #[test]
    fn test_detect_fallback_to_raw() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC4:fallback
        // ADR: adr-014-prompt-template-auto-detection.md
        // Verify fallback to Raw template when no hints available

        // Test detection with no metadata or tokenizer (priority 3 fallback)
        let detected = TemplateType::detect(None, None);
        assert_eq!(detected, TemplateType::Raw);

        // Test with unrecognized tokenizer name
        let detected2 = TemplateType::detect(Some("unknown-model"), None);
        assert_eq!(detected2, TemplateType::Raw);

        Ok(())
    }

    // AC4:priority - GGUF metadata takes priority over tokenizer hints
    #[test]
    fn test_detection_priority_gguf_over_tokenizer() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC4
        // ADR: adr-014-prompt-template-auto-detection.md
        // Verify that GGUF metadata has highest priority

        // Create conflicting hints: GGUF says LLaMA-3, tokenizer says instruct
        let jinja = "<|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>";
        let tokenizer_name = "mistral-instruct"; // Would detect Instruct

        // Expected: GGUF metadata wins (Llama3Chat, not Instruct)
        let detected = TemplateType::detect(Some(tokenizer_name), Some(jinja));
        assert_eq!(detected, TemplateType::Llama3Chat);

        Ok(())
    }

    // AC4:architecture - Architecture hints enable heuristic matching
    #[test]
    fn test_detect_from_architecture_hints() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC4
        // ADR: adr-014-prompt-template-auto-detection.md
        // Verify detection from model architecture hints

        // Test with llama3 in tokenizer name (architecture hint)
        let detected1 = TemplateType::detect(Some("llama3-8b"), None);
        assert_eq!(detected1, TemplateType::Llama3Chat);

        // Test with llama-3 variant
        let detected2 = TemplateType::detect(Some("meta-llama-3-instruct"), None);
        assert_eq!(detected2, TemplateType::Llama3Chat);

        // Test case insensitivity
        let detected3 = TemplateType::detect(Some("LLaMA3-Chat"), None);
        assert_eq!(detected3, TemplateType::Llama3Chat);

        Ok(())
    }

    // AC4:override - User override bypasses auto-detection
    #[test]
    fn test_user_override_bypasses_detection() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC4
        // ADR: adr-014-prompt-template-auto-detection.md
        // Verify that explicit --prompt-template overrides detection

        // This test verifies the logic for user override at the library level
        // CLI integration is tested separately in CLI tests

        // Simulate what CLI does: auto-detect first (with complete LLaMA-3 template)
        let jinja = "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>";
        let auto_detected = TemplateType::detect(None, Some(jinja));
        assert_eq!(auto_detected, TemplateType::Llama3Chat);

        // User explicitly specifies raw via CLI --prompt-template flag
        // The CLI would parse this and use it directly instead of auto-detection
        let user_override: TemplateType = "raw".parse()?;
        assert_eq!(user_override, TemplateType::Raw);

        // Verify user override takes precedence (simulated CLI behavior)
        let final_template = user_override; // CLI uses user's choice
        assert_eq!(final_template, TemplateType::Raw);

        Ok(())
    }
}

#[cfg(test)]
mod ac4_debug_logging {
    use super::*;

    // AC4:logging - Debug logs show detection decisions
    #[test]
    #[ignore = "implementation pending: implement debug logging for detection"]
    fn test_detection_logs_decision() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC4
        // ADR: adr-014-prompt-template-auto-detection.md
        // Verify that detection decisions are logged at debug level

        // TODO: Capture log output during detection
        // let gguf_metadata = GgufMetadata {
        //     chat_template: Some("<|start_header_id|>".into()),
        //     ..Default::default()
        // };

        // Enable debug logging
        // let _guard = setup_test_logging("debug");

        // Run detection
        // let _detected = TemplateType::detect(Some(&gguf_metadata), None);

        // Expected: Debug log contains detection rationale
        // let logs = capture_logs();
        // assert!(logs.contains("Auto-detected template: llama3-chat"));
        // assert!(logs.contains("from chat_template metadata"));

        panic!("Test not implemented: needs debug logging verification");
    }

    // AC4:logging_fallback - Fallback logs warning
    #[test]
    #[ignore = "implementation pending: implement fallback logging"]
    fn test_fallback_logs_warning() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC4
        // ADR: adr-014-prompt-template-auto-detection.md
        // Verify that fallback to Raw logs debug warning

        // TODO: Capture log output during fallback
        // let _guard = setup_test_logging("debug");

        // Run detection with no hints
        // let _detected = TemplateType::detect(None, None);

        // Expected: Debug log contains fallback warning
        // let logs = capture_logs();
        // assert!(logs.contains("No template hints found"));
        // assert!(logs.contains("using raw template"));

        panic!("Test not implemented: needs fallback logging verification");
    }
}

#[cfg(test)]
mod template_detection_integration {
    use super::*;

    // Integration: Detection affects prompt formatting
    #[test]
    #[ignore = "implementation pending: verify detection affects formatting"]
    fn test_detected_template_affects_formatting() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC4
        // Verify that auto-detected template is used for formatting

        // TODO: Detect template and apply to user input
        // let gguf_metadata = GgufMetadata {
        //     chat_template: Some("<|start_header_id|>".into()),
        //     ..Default::default()
        // };
        // let detected = TemplateType::detect(Some(&gguf_metadata), None);

        // Apply to user text
        // let formatted = detected.apply("Hello", Some("You are helpful"));

        // Expected: LLaMA-3 formatting is applied
        // assert!(formatted.starts_with("<|begin_of_text|>"));
        // assert!(formatted.contains("<|start_header_id|>system<|end_header_id|>"));
        // assert!(formatted.contains("You are helpful"));
        // assert!(formatted.contains("<|start_header_id|>user<|end_header_id|>"));
        // assert!(formatted.contains("Hello"));

        panic!("Test not implemented: needs formatting integration");
    }

    // Integration: Detection overhead is minimal
    #[test]
    #[ignore = "implementation pending: benchmark detection overhead"]
    fn test_detection_overhead_minimal() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC4
        // ADR: adr-014-prompt-template-auto-detection.md
        // Verify that detection overhead is <5ms

        // TODO: Benchmark detection time
        // let gguf_metadata = GgufMetadata {
        //     chat_template: Some("<|start_header_id|>".into()),
        //     ..Default::default()
        // };

        // let start = std::time::Instant::now();
        // let _detected = TemplateType::detect(Some(&gguf_metadata), None);
        // let elapsed = start.elapsed();

        // Expected: Detection completes in <5ms
        // assert!(elapsed.as_millis() < 5, "Detection took {}ms (target: <5ms)", elapsed.as_millis());

        panic!("Test not implemented: needs performance benchmarking");
    }

    // Integration: Detection works with real GGUF files
    #[test]
    #[ignore = "implementation pending: test with real GGUF models"]
    fn test_detection_with_real_gguf() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC4
        // Verify detection works with actual GGUF model files

        // TODO: Load real GGUF model and detect template
        // let model_path = "tests/models/llama3-tiny.gguf";
        // let model = load_gguf_model(model_path)?;
        // let metadata = extract_gguf_metadata(&model)?;

        // Expected: Template detected from real metadata
        // let detected = TemplateType::detect(Some(&metadata), None);
        // assert!(detected == TemplateType::Llama3Chat || detected == TemplateType::Raw);

        panic!("Test not implemented: needs real GGUF integration");
    }
}

#[cfg(test)]
mod tokenizer_family_hints {
    use super::*;

    // Tokenizer: LLaMA-3 family name detection
    #[test]
    #[ignore = "implementation pending: implement get_family_name for tokenizers"]
    fn test_tokenizer_llama3_family() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC4
        // ADR: adr-014-prompt-template-auto-detection.md
        // Verify LLaMA-3 detection from tokenizer family name

        // TODO: Create tokenizer with llama3 family
        // let tokenizer = MockTokenizer {
        //     family_name: Some("llama3".into()),
        //     ..Default::default()
        // };

        // Expected: Detects Llama3Chat template
        // let detected = TemplateType::detect(None, Some(&tokenizer));
        // assert_eq!(detected, TemplateType::Llama3Chat);

        panic!("Test not implemented: needs tokenizer family detection");
    }

    // Tokenizer: Mistral-instruct family name detection
    #[test]
    #[ignore = "implementation pending: implement instruct family detection"]
    fn test_tokenizer_mistral_instruct_family() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC4
        // Verify Mistral-instruct detection from family name

        // TODO: Create tokenizer with mistral-instruct family
        // let tokenizer = MockTokenizer {
        //     family_name: Some("mistral-instruct".into()),
        //     ..Default::default()
        // };

        // Expected: Detects Instruct template
        // let detected = TemplateType::detect(None, Some(&tokenizer));
        // assert_eq!(detected, TemplateType::Instruct);

        panic!("Test not implemented: needs mistral family detection");
    }

    // Tokenizer: get_family_name trait method
    #[test]
    #[ignore = "implementation pending: add get_family_name to Tokenizer trait"]
    fn test_tokenizer_get_family_name_trait() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC4
        // ADR: adr-014-prompt-template-auto-detection.md
        // Verify Tokenizer trait has get_family_name method

        // TODO: Test trait method exists
        // let tokenizer = create_test_tokenizer();

        // Expected: get_family_name returns Option<String>
        // let family = tokenizer.get_family_name();
        // assert!(family.is_none() || family.is_some());

        panic!("Test not implemented: needs Tokenizer trait extension");
    }
}
