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
    #[test]
    fn test_matches_llama3_pattern() -> Result<()> {
        let chat_template = "<|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>";
        let detected = TemplateType::detect(None, Some(chat_template));
        assert_eq!(detected, TemplateType::Llama3Chat);
        let minimal_template = "something <|start_header_id|> and <|eot_id|> here";
        let detected_minimal = TemplateType::detect(None, Some(minimal_template));
        assert_eq!(detected_minimal, TemplateType::Llama3Chat);
        Ok(())
    }
    #[test]
    fn test_llama3_full_template_pattern() -> Result<()> {
        let chat_template = concat!(
            "<|begin_of_text|>",
            "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
            "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        assert!(chat_template.contains("<|start_header_id|>"));
        assert!(chat_template.contains("<|end_header_id|>"));
        assert!(chat_template.contains("<|eot_id|>"));
        let detected = TemplateType::detect(None, Some(chat_template));
        assert_eq!(detected, TemplateType::Llama3Chat);
        Ok(())
    }
    #[test]
    fn test_matches_instruct_pattern() -> Result<()> {
        let chat_template1 = "{% for message in messages %}Q: {user_text}\nA:{% endfor %}";
        let detected1 = TemplateType::detect(None, Some(chat_template1));
        assert_eq!(detected1, TemplateType::Instruct);
        let chat_template2 = "### Instruction\n{% for message in messages %}{instruction}{% endfor %}\n\n### Response";
        let detected2 = TemplateType::detect(None, Some(chat_template2));
        assert_eq!(detected2, TemplateType::Instruct);
        Ok(())
    }
    #[test]
    fn test_instruct_pattern_variants() -> Result<()> {
        let patterns = vec![
            "{% for message in messages %}Q: {question}\nA:{% endfor %}",
            "{% for message in messages %}### Instruction\n{instruction}\n\n### Response{% endfor %}",
            "{% for message in messages %}Human: {input}\n\nAssistant:{% endfor %}",
        ];
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
    #[test]
    fn test_no_pattern_fallback_to_raw() -> Result<()> {
        let chat_template = "Just plain text {placeholder}";
        let detected = TemplateType::detect(None, Some(chat_template));
        assert_eq!(detected, TemplateType::Raw);
        let unknown_template = "some random template without special tokens";
        let detected2 = TemplateType::detect(None, Some(unknown_template));
        assert_eq!(detected2, TemplateType::Raw);
        Ok(())
    }
}
#[cfg(test)]
mod ac4_template_detection {
    use super::*;
    #[test]
    fn test_detect_llama3_from_chat_template() -> Result<()> {
        let jinja = "<|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>";
        let detected = TemplateType::detect(None, Some(jinja));
        assert_eq!(detected, TemplateType::Llama3Chat);
        Ok(())
    }
    #[test]
    fn test_detect_instruct_from_family() -> Result<()> {
        let detected1 = TemplateType::detect(Some("mistral-instruct"), None);
        assert_eq!(detected1, TemplateType::Instruct);
        let detected2 = TemplateType::detect(Some("some-instruct-model"), None);
        assert_eq!(detected2, TemplateType::Instruct);
        let detected3 = TemplateType::detect(Some("mistral-7b"), None);
        assert_eq!(detected3, TemplateType::Instruct);
        Ok(())
    }
    #[test]
    fn test_detect_fallback_to_raw() -> Result<()> {
        let detected = TemplateType::detect(None, None);
        assert_eq!(detected, TemplateType::Raw);
        let detected2 = TemplateType::detect(Some("unknown-model"), None);
        assert_eq!(detected2, TemplateType::Raw);
        Ok(())
    }
    #[test]
    fn test_detection_priority_gguf_over_tokenizer() -> Result<()> {
        let jinja = "<|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>";
        let tokenizer_name = "mistral-instruct";
        let detected = TemplateType::detect(Some(tokenizer_name), Some(jinja));
        assert_eq!(detected, TemplateType::Llama3Chat);
        Ok(())
    }
    #[test]
    fn test_detect_from_architecture_hints() -> Result<()> {
        let detected1 = TemplateType::detect(Some("llama3-8b"), None);
        assert_eq!(detected1, TemplateType::Llama3Chat);
        let detected2 = TemplateType::detect(Some("meta-llama-3-instruct"), None);
        assert_eq!(detected2, TemplateType::Llama3Chat);
        let detected3 = TemplateType::detect(Some("LLaMA3-Chat"), None);
        assert_eq!(detected3, TemplateType::Llama3Chat);
        Ok(())
    }
    #[test]
    fn test_user_override_bypasses_detection() -> Result<()> {
        let jinja = "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>";
        let auto_detected = TemplateType::detect(None, Some(jinja));
        assert_eq!(auto_detected, TemplateType::Llama3Chat);
        let user_override: TemplateType = "raw".parse()?;
        assert_eq!(user_override, TemplateType::Raw);
        let final_template = user_override;
        assert_eq!(final_template, TemplateType::Raw);
        Ok(())
    }
}
#[cfg(test)]
mod ac4_debug_logging {
    use super::*;
    #[test]
    #[ignore = "implementation pending: implement debug logging for detection"]
    fn test_detection_logs_decision() -> Result<()> {
        panic!("Test not implemented: needs debug logging verification");
    }
    #[test]
    #[ignore = "implementation pending: implement fallback logging"]
    fn test_fallback_logs_warning() -> Result<()> {
        panic!("Test not implemented: needs fallback logging verification");
    }
}
#[cfg(test)]
mod template_detection_integration {
    use super::*;
    #[test]
    fn test_detected_template_affects_formatting() -> Result<()> {
        let user_text = "What is 2+2?";
        let instruct_output = TemplateType::Instruct.apply(user_text, None);
        let raw_output = TemplateType::Raw.apply(user_text, None);
        // Instruct template wraps with Q: / A: markers, Raw returns plain text.
        assert_ne!(
            instruct_output, raw_output,
            "Instruct and Raw templates must produce different output"
        );
        assert!(instruct_output.contains(user_text), "Instruct output must contain user text");
        assert!(raw_output.contains(user_text), "Raw output must contain user text");
        Ok(())
    }
    #[test]
    fn test_detection_overhead_minimal() -> Result<()> {
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            let _ = TemplateType::detect(Some("llama3-chat"), None);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 1000,
            "10k detect calls should complete within 1s, took {}ms",
            elapsed.as_millis()
        );
        Ok(())
    }
    #[test]
    #[ignore = "implementation pending: test with real GGUF models"]
    fn test_detection_with_real_gguf() -> Result<()> {
        panic!("Test not implemented: needs real GGUF integration");
    }
}
#[cfg(test)]
mod tokenizer_family_hints {
    use super::*;
    use bitnet_tokenizers::{MockTokenizer, Tokenizer};

    #[test]
    fn test_tokenizer_llama3_family() -> Result<()> {
        let tok = MockTokenizer::with_special_tokens(&[
            ("<|eot_id|>", 128009),
            ("<|start_header_id|>", 128006),
        ]);
        assert_eq!(tok.get_family_name(), "llama3");
        Ok(())
    }

    #[test]
    fn test_tokenizer_mistral_instruct_family() -> Result<()> {
        let tok = MockTokenizer::with_special_tokens(&[("[INST]", 1001), ("[/INST]", 1002)]);
        assert_eq!(tok.get_family_name(), "mistral-instruct");
        Ok(())
    }

    #[test]
    fn test_tokenizer_get_family_name_trait() -> Result<()> {
        // Unknown tokenizer (no special tokens) returns "unknown".
        let unknown = MockTokenizer::new();
        assert_eq!(unknown.get_family_name(), "unknown");

        // LLaMA-3 check: <|eot_id|> alone is sufficient.
        let llama3 = MockTokenizer::with_special_tokens(&[("<|eot_id|>", 128009)]);
        assert_eq!(llama3.get_family_name(), "llama3");

        // Mistral: [INST] token without LLaMA-3 markers.
        let mistral = MockTokenizer::with_special_tokens(&[("[INST]", 2001)]);
        assert_eq!(mistral.get_family_name(), "mistral-instruct");
        Ok(())
    }
}
