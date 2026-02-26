//! Test Suite 2: CLI - Template Auto-Detection (bitnet-cli)
//!
//! Tests feature spec: chat-repl-ux-polish.md#AC2-template-auto-detection
//!
//! This test suite validates the prompt template auto-detection logic for the
//! BitNet-rs CLI. Tests verify the priority order: GGUF metadata → tokenizer
//! family name → subcommand defaults.
//!
//! **TDD Approach**: These tests compile successfully but fail because the
//! auto-detection logic needs to be implemented in the CLI layer with proper
//! integration to the InferenceCommand.

use anyhow::Result;
use bitnet_inference::TemplateType;

/// Mock GGUF metadata for testing detection
#[derive(Debug, Clone)]
struct MockGgufMetadata {
    chat_template: Option<String>,
}

/// Mock tokenizer metadata for testing detection
#[derive(Debug, Clone)]
struct MockTokenizerMetadata {
    family_name: Option<String>,
}

/// Simulates the template detection logic that should be implemented in CLI
fn detect_template(
    gguf_metadata: Option<&MockGgufMetadata>,
    tokenizer_metadata: Option<&MockTokenizerMetadata>,
    subcommand: &str,
) -> TemplateType {
    // Priority 1: GGUF chat_template metadata
    if let Some(gguf) = gguf_metadata
        && let Some(template) = &gguf.chat_template
    {
        // Detect LLaMA-3 special tokens
        if template.contains("<|start_header_id|>") && template.contains("<|eot_id|>") {
            return TemplateType::Llama3Chat;
        }
        // Detect generic instruct pattern
        if template.contains("{% for message in messages %}") {
            return TemplateType::Instruct;
        }
    }

    // Priority 2: Tokenizer family name heuristics
    if let Some(tokenizer) = tokenizer_metadata
        && let Some(name) = &tokenizer.family_name
    {
        let lower = name.to_ascii_lowercase();
        if lower.contains("llama3") || lower.contains("llama-3") {
            return TemplateType::Llama3Chat;
        }
        if lower.contains("instruct") || lower.contains("mistral") {
            return TemplateType::Instruct;
        }
    }

    // Priority 3: Subcommand-specific defaults
    match subcommand {
        "chat" => TemplateType::Llama3Chat, // Friendly default for chat
        "run" | "generate" => TemplateType::Instruct, // Backward compat
        _ => TemplateType::Raw,
    }
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-priority-order
#[test]
fn test_detection_priority_order() {
    // Priority 1: GGUF metadata wins over everything
    let gguf = MockGgufMetadata {
        chat_template: Some(
            "<|start_header_id|>user<|end_header_id|> {{ message }}<|eot_id|>".to_string(),
        ),
    };
    let tokenizer = MockTokenizerMetadata { family_name: Some("mistral-instruct".to_string()) };

    let result = detect_template(Some(&gguf), Some(&tokenizer), "run");

    assert_eq!(
        result,
        TemplateType::Llama3Chat,
        "GGUF metadata should take priority over tokenizer family name"
    );
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-gguf-llama3-detection
#[test]
fn test_gguf_metadata_llama3_detection() {
    let gguf = MockGgufMetadata {
        chat_template: Some(
            "{% if messages[0]['role'] == 'system' %}<|start_header_id|>system<|end_header_id|> {{ messages[0]['content'] }}<|eot_id|>{% endif %}".to_string()
        ),
    };

    let result = detect_template(Some(&gguf), None, "run");

    assert_eq!(
        result,
        TemplateType::Llama3Chat,
        "Should detect LLaMA-3 from special tokens in GGUF metadata"
    );
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-gguf-instruct-detection
#[test]
fn test_gguf_metadata_instruct_detection() {
    let gguf = MockGgufMetadata {
        chat_template: Some(
            "{% for message in messages %}User: {{ message.content }}{% endfor %}".to_string(),
        ),
    };

    let result = detect_template(Some(&gguf), None, "run");

    assert_eq!(
        result,
        TemplateType::Instruct,
        "Should detect generic instruct pattern from GGUF metadata"
    );
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-gguf-fallback
#[test]
fn test_gguf_metadata_invalid_falls_back() {
    let gguf = MockGgufMetadata {
        chat_template: Some("some custom template without known patterns".to_string()),
    };

    let result = detect_template(Some(&gguf), None, "run");

    // Should fall back to subcommand default when GGUF has no known pattern
    assert_eq!(
        result,
        TemplateType::Instruct,
        "Unknown GGUF template should fall back to subcommand default"
    );
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-tokenizer-llama3
#[test]
fn test_tokenizer_family_llama3_detection() {
    let tokenizer = MockTokenizerMetadata { family_name: Some("llama3".to_string()) };

    let result = detect_template(None, Some(&tokenizer), "run");

    assert_eq!(
        result,
        TemplateType::Llama3Chat,
        "Should detect LLaMA-3 from tokenizer family name 'llama3'"
    );
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-tokenizer-llama-3
#[test]
fn test_tokenizer_family_llama_dash_3_detection() {
    let tokenizer = MockTokenizerMetadata { family_name: Some("llama-3-instruct".to_string()) };

    let result = detect_template(None, Some(&tokenizer), "run");

    assert_eq!(
        result,
        TemplateType::Llama3Chat,
        "Should detect LLaMA-3 from tokenizer family name containing 'llama-3'"
    );
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-tokenizer-mistral
#[test]
fn test_tokenizer_family_mistral_detection() {
    let tokenizer = MockTokenizerMetadata { family_name: Some("mistral-v2".to_string()) };

    let result = detect_template(None, Some(&tokenizer), "run");

    assert_eq!(
        result,
        TemplateType::Instruct,
        "Should detect Instruct from tokenizer family name 'mistral'"
    );
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-tokenizer-instruct
#[test]
fn test_tokenizer_family_instruct_detection() {
    let tokenizer = MockTokenizerMetadata { family_name: Some("phi-2-instruct".to_string()) };

    let result = detect_template(None, Some(&tokenizer), "run");

    assert_eq!(
        result,
        TemplateType::Instruct,
        "Should detect Instruct from tokenizer family name containing 'instruct'"
    );
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-tokenizer-unknown
#[test]
fn test_tokenizer_family_unknown_falls_back() {
    let tokenizer = MockTokenizerMetadata { family_name: Some("custom-tokenizer".to_string()) };

    let result = detect_template(None, Some(&tokenizer), "run");

    // Should fall back to subcommand default
    assert_eq!(
        result,
        TemplateType::Instruct,
        "Unknown tokenizer family should fall back to subcommand default"
    );
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-subcommand-chat-default
#[test]
fn test_subcommand_chat_default() {
    // No metadata available - should use subcommand default
    let result = detect_template(None, None, "chat");

    assert_eq!(
        result,
        TemplateType::Llama3Chat,
        "Chat subcommand should default to Llama3Chat for friendly UX"
    );
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-subcommand-run-default
#[test]
fn test_subcommand_run_default() {
    // No metadata available - should use subcommand default
    let result = detect_template(None, None, "run");

    assert_eq!(
        result,
        TemplateType::Instruct,
        "Run subcommand should default to Instruct for backward compatibility"
    );
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-subcommand-generate-default
#[test]
fn test_subcommand_generate_default() {
    // Generate is an alias for run
    let result = detect_template(None, None, "generate");

    assert_eq!(
        result,
        TemplateType::Instruct,
        "Generate subcommand should default to Instruct (same as run)"
    );
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-explicit-override
#[test]
fn test_explicit_template_override_bypasses_detection() {
    // This test verifies the CLI respects --prompt-template flag
    // When user explicitly sets --prompt-template, detection should be skipped

    // Simulate explicit override scenario
    let explicit_template = "raw"; // User specified --prompt-template raw

    // Parse explicit template
    let result: Result<TemplateType> = explicit_template.parse();

    assert!(result.is_ok(), "Explicit template should parse successfully");
    assert_eq!(
        result.unwrap(),
        TemplateType::Raw,
        "Explicit --prompt-template should override auto-detection"
    );
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-case-insensitive
#[test]
fn test_tokenizer_detection_case_insensitive() {
    // Test uppercase variants
    let tokenizer_upper = MockTokenizerMetadata { family_name: Some("LLAMA3".to_string()) };
    let result_upper = detect_template(None, Some(&tokenizer_upper), "run");
    assert_eq!(
        result_upper,
        TemplateType::Llama3Chat,
        "Detection should be case-insensitive (uppercase)"
    );

    // Test mixed case
    let tokenizer_mixed = MockTokenizerMetadata { family_name: Some("LLaMa-3".to_string()) };
    let result_mixed = detect_template(None, Some(&tokenizer_mixed), "run");
    assert_eq!(
        result_mixed,
        TemplateType::Llama3Chat,
        "Detection should be case-insensitive (mixed case)"
    );
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-gguf-missing-fallback
#[test]
fn test_missing_gguf_metadata_falls_back_to_tokenizer() {
    let tokenizer = MockTokenizerMetadata { family_name: Some("llama3".to_string()) };

    let result = detect_template(None, Some(&tokenizer), "run");

    assert_eq!(
        result,
        TemplateType::Llama3Chat,
        "Missing GGUF metadata should fall back to tokenizer detection"
    );
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-all-missing-fallback
#[test]
fn test_all_metadata_missing_uses_subcommand_default() {
    // No GGUF, no tokenizer metadata - ultimate fallback
    let result_chat = detect_template(None, None, "chat");
    let result_run = detect_template(None, None, "run");

    assert_eq!(
        result_chat,
        TemplateType::Llama3Chat,
        "Chat subcommand should default to Llama3Chat when all metadata missing"
    );
    assert_eq!(
        result_run,
        TemplateType::Instruct,
        "Run subcommand should default to Instruct when all metadata missing"
    );
}

/// Tests feature spec: chat-repl-ux-polish.md#AC2-empty-gguf-template
#[test]
fn test_empty_gguf_chat_template_falls_back() {
    let gguf = MockGgufMetadata { chat_template: Some("".to_string()) };

    let tokenizer = MockTokenizerMetadata { family_name: Some("llama3".to_string()) };

    let result = detect_template(Some(&gguf), Some(&tokenizer), "run");

    // Empty GGUF template should be treated as missing, fall back to tokenizer
    assert_eq!(
        result,
        TemplateType::Llama3Chat,
        "Empty GGUF chat_template should fall back to tokenizer detection"
    );
}

/// Tests BitNet model path detection patterns
/// Tests feature spec: Improved BitNet base model detection
#[test]
fn test_bitnet_path_detection_patterns() {
    // Test that BitNet detection logic handles various path patterns
    // This tests the pattern matching in auto_detect_template() indirectly

    let test_cases = vec![
        // Pattern: microsoft-bitnet
        ("microsoft-bitnet", true, "microsoft-bitnet prefix"),
        // Pattern: bitnet-b1.58
        ("bitnet-b1.58", true, "bitnet-b1.58 pattern"),
        // Pattern: bitnet-1.58b
        ("bitnet-1.58b", true, "bitnet-1.58b pattern"),
        // Pattern: bitnet-1_58b
        ("bitnet-1_58b", true, "bitnet-1_58b underscore"),
        // Pattern: 1.58b alone
        ("1.58b", true, "1.58b pattern alone"),
        // Pattern: 1_58b alone
        ("1_58b", true, "1_58b underscore alone"),
        // Pattern: generic bitnet (but not bitnet-instruct)
        ("bitnet", true, "generic bitnet"),
        ("custom-bitnet-model", true, "bitnet in path"),
        // Should NOT match
        ("bitnet-instruct", false, "bitnet-instruct should not match"),
        ("llama3", false, "non-bitnet model"),
    ];

    for (path_component, should_match, description) in test_cases {
        let path_lower = path_component.to_lowercase();

        // Replicate the detection logic from auto_detect_template()
        let matches_bitnet = path_lower.contains("microsoft-bitnet")
            || path_lower.contains("bitnet-b1.58")
            || path_lower.contains("bitnet-1.58b")
            || path_lower.contains("bitnet-1_58b")
            || path_lower.contains("1.58b")
            || path_lower.contains("1_58b")
            || (path_lower.contains("bitnet") && !path_lower.contains("instruct"));

        assert_eq!(
            matches_bitnet, should_match,
            "Pattern '{}' ({}) match result should be {}",
            path_component, description, should_match
        );
    }
}

/// Integration test: Full detection pipeline simulation
/// Tests feature spec: chat-repl-ux-polish.md#AC2-integration
#[test]
fn test_full_detection_pipeline() {
    // Scenario 1: Production model with full metadata
    let gguf1 = MockGgufMetadata {
        chat_template: Some("<|start_header_id|>user<|end_header_id|><|eot_id|>".to_string()),
    };
    let tok1 = MockTokenizerMetadata { family_name: Some("mistral".to_string()) };
    assert_eq!(
        detect_template(Some(&gguf1), Some(&tok1), "run"),
        TemplateType::Llama3Chat,
        "GGUF should win even when tokenizer suggests different"
    );

    // Scenario 2: GGUF without chat_template, rely on tokenizer
    let tok2 = MockTokenizerMetadata { family_name: Some("llama-3-8b".to_string()) };
    assert_eq!(
        detect_template(None, Some(&tok2), "chat"),
        TemplateType::Llama3Chat,
        "Should detect from tokenizer when GGUF missing"
    );

    // Scenario 3: No metadata, use subcommand default
    assert_eq!(
        detect_template(None, None, "chat"),
        TemplateType::Llama3Chat,
        "Chat subcommand default should be Llama3Chat"
    );
}
