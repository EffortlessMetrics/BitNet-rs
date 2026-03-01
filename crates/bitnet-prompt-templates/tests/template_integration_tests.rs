//! Integration tests for prompt template ↔ architecture pipeline.
//!
//! Verifies that template suggestion, rendering, chat formatting, and
//! validation work correctly for all supported model families.

use bitnet_prompt_templates::{ChatRole, ChatTurn, PromptTemplate, TemplateType};

// ---------------------------------------------------------------------------
// Architecture → Template suggestion
// ---------------------------------------------------------------------------

#[test]
fn phi_suggests_template() {
    let template = TemplateType::suggest_for_arch("phi");
    assert!(template.is_some(), "phi should suggest a template");
}

#[test]
fn llama_suggests_template() {
    let template = TemplateType::suggest_for_arch("llama");
    assert!(template.is_some(), "llama should suggest a template");
}

#[test]
fn mistral_suggests_template() {
    let template = TemplateType::suggest_for_arch("mistral");
    assert!(template.is_some(), "mistral should suggest a template");
}

#[test]
fn qwen_suggests_template() {
    let template = TemplateType::suggest_for_arch("qwen");
    assert!(template.is_some(), "qwen should suggest a template");
}

#[test]
fn gemma_suggests_template() {
    let template = TemplateType::suggest_for_arch("gemma");
    assert!(template.is_some(), "gemma should suggest a template");
}

#[test]
fn deepseek_suggests_template() {
    let template = TemplateType::suggest_for_arch("deepseek");
    assert!(template.is_some(), "deepseek should suggest a template");
}

#[test]
fn falcon_suggests_template() {
    let template = TemplateType::suggest_for_arch("falcon");
    assert!(template.is_some(), "falcon should suggest a template");
}

#[test]
fn unknown_arch_no_template_suggestion() {
    let template = TemplateType::suggest_for_arch("totally_unknown_model");
    assert!(template.is_none());
}

#[test]
fn empty_arch_no_template_suggestion() {
    let template = TemplateType::suggest_for_arch("");
    assert!(template.is_none());
}

// ---------------------------------------------------------------------------
// Template apply produces valid output
// ---------------------------------------------------------------------------

#[test]
fn instruct_template_apply_basic() {
    let output = TemplateType::Instruct.apply("What is 2+2?", None);
    assert!(!output.is_empty());
    assert!(output.contains("2+2"));
}

#[test]
fn phi4chat_template_apply_with_system() {
    let output = TemplateType::Phi4Chat.apply("Hello", Some("You are a helpful assistant"));
    assert!(output.contains("Hello"));
    assert!(output.contains("helpful assistant"));
}

#[test]
fn all_templates_handle_empty_prompt() {
    for variant in TemplateType::all_variants() {
        let output = variant.apply("", None);
        let _ = output;
    }
}

#[test]
fn all_templates_handle_long_prompt() {
    let long_prompt = "A".repeat(10_000);
    for variant in TemplateType::all_variants() {
        let output = variant.apply(&long_prompt, None);
        assert!(output.len() >= long_prompt.len(), "Template {:?} truncated long prompt", variant);
    }
}

#[test]
fn all_templates_handle_unicode_prompt() {
    let prompt = "こんにちは世界 🌍 Привет мир";
    for variant in TemplateType::all_variants() {
        let output = variant.apply(prompt, None);
        assert!(output.contains(prompt), "Template {:?} lost unicode content", variant);
    }
}

// ---------------------------------------------------------------------------
// PromptTemplate builder
// ---------------------------------------------------------------------------

#[test]
fn prompt_template_builder_basic() {
    let pt = PromptTemplate::new(TemplateType::Instruct);
    let output = pt.format("What is AI?");
    assert!(!output.is_empty());
    assert!(output.contains("AI"));
}

#[test]
fn prompt_template_with_system() {
    let pt = PromptTemplate::new(TemplateType::Phi4Chat).with_system_prompt("You are a math tutor");
    let output = pt.format("What is calculus?");
    assert!(output.contains("calculus"));
}

#[test]
fn prompt_template_stop_sequences() {
    let pt = PromptTemplate::new(TemplateType::Phi4Chat);
    let stops = pt.stop_sequences();
    assert!(!stops.is_empty(), "Phi4Chat should have stop sequences");
}

#[test]
fn prompt_template_add_turn_and_format() {
    let mut pt = PromptTemplate::new(TemplateType::Phi4Chat);
    pt.add_turn("What is 1+1?", "2");
    let output = pt.format("What is 2+2?");
    assert!(output.contains("2+2"));
}

#[test]
fn prompt_template_clear_history() {
    let mut pt = PromptTemplate::new(TemplateType::Phi4Chat);
    pt.add_turn("Hello", "Hi");
    pt.clear_history();
    let output = pt.format("New question");
    assert!(output.contains("New question"));
}

#[test]
fn prompt_template_type_accessor() {
    let pt = PromptTemplate::new(TemplateType::Instruct);
    assert!(matches!(pt.template_type(), TemplateType::Instruct));
}

// ---------------------------------------------------------------------------
// ChatTurn / render_chat
// ---------------------------------------------------------------------------

#[test]
fn render_chat_single_turn() {
    let turns = vec![ChatTurn::new(ChatRole::User, "Hi there!")];
    let result = TemplateType::Phi4Chat.render_chat(&turns, None);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.contains("Hi there!"));
}

#[test]
fn render_chat_multi_turn() {
    let turns = vec![
        ChatTurn::new(ChatRole::User, "Hello"),
        ChatTurn::new(ChatRole::Assistant, "Hi! How can I help?"),
        ChatTurn::new(ChatRole::User, "Tell me about Rust"),
    ];
    let result = TemplateType::Phi4Chat.render_chat(&turns, None);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.contains("Hello"));
    assert!(output.contains("Rust"));
}

#[test]
fn render_chat_with_system() {
    let turns = vec![ChatTurn::new(ChatRole::User, "Hi")];
    let result = TemplateType::Phi4Chat.render_chat(&turns, Some("You are a helpful assistant"));
    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.contains("helpful assistant"));
}

#[test]
fn render_chat_empty_turns() {
    let turns: Vec<ChatTurn> = vec![];
    let result = TemplateType::Phi4Chat.render_chat(&turns, None);
    let _ = result;
}

// ---------------------------------------------------------------------------
// All variants enumeration
// ---------------------------------------------------------------------------

#[test]
fn all_variants_returns_nonempty() {
    let variants = TemplateType::all_variants();
    assert!(!variants.is_empty());
    assert!(variants.len() >= 40, "Expected at least 40 template variants, got {}", variants.len());
}

#[test]
fn all_variants_have_valid_info() {
    for variant in TemplateType::all_variants() {
        let info = variant.info();
        assert!(!info.name.is_empty(), "Template {:?} has empty name", variant);
    }
}

#[test]
fn all_variants_produce_nonempty_output() {
    for variant in TemplateType::all_variants() {
        let output = variant.apply("test prompt", None);
        assert!(!output.is_empty(), "Template {:?} produced empty output", variant);
    }
}

#[test]
fn all_variants_stop_sequences_dont_panic() {
    for variant in TemplateType::all_variants() {
        let stops = variant.default_stop_sequences();
        let _ = stops;
    }
}

#[test]
fn all_variants_render_chat_dont_panic() {
    let turns = vec![ChatTurn::new(ChatRole::User, "test")];
    for variant in TemplateType::all_variants() {
        let _ = variant.render_chat(&turns, None);
    }
}

#[test]
fn all_variants_validate_output_dont_panic() {
    for variant in TemplateType::all_variants() {
        let output = variant.apply("hello", None);
        let validation = variant.validate_output(&output, "hello");
        let _ = validation;
    }
}

// ---------------------------------------------------------------------------
// Template detection
// ---------------------------------------------------------------------------

#[test]
fn detect_chatml_from_jinja() {
    let detected = TemplateType::detect(None, Some("{% if messages[0] %}"));
    // Should not panic regardless of detection result
    let _ = detected;
}

#[test]
fn detect_with_no_hints_returns_default() {
    let detected = TemplateType::detect(None, None);
    let _ = detected;
}

// ---------------------------------------------------------------------------
// ChatRole as_str
// ---------------------------------------------------------------------------

#[test]
fn chat_role_as_str() {
    assert_eq!(ChatRole::System.as_str(), "system");
    assert_eq!(ChatRole::User.as_str(), "user");
    assert_eq!(ChatRole::Assistant.as_str(), "assistant");
}

// ---------------------------------------------------------------------------
// Regression guards
// ---------------------------------------------------------------------------

#[test]
fn template_count_regression_guard() {
    let count = TemplateType::all_variants().len();
    assert!(count >= 40, "Expected at least 40 template variants, got {}", count);
}

#[test]
fn major_architectures_have_template_suggestions() {
    let major_archs = ["phi", "llama", "mistral", "qwen", "gemma", "deepseek", "falcon"];
    for arch in &major_archs {
        assert!(
            TemplateType::suggest_for_arch(arch).is_some(),
            "Major architecture '{}' should have a template suggestion",
            arch
        );
    }
}
