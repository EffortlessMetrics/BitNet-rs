//! render_chat and validate_output tests for all TemplateType variants.
//!
//! Coverage targets:
//! - render_chat for every template in all_variants()
//! - validate_output return values (TemplateValidation struct)
//! - PromptTemplate builder pattern
//! - Edge cases: empty history, system-only, multi-turn
//! - all_variants() exhaustiveness

use bitnet_prompt_templates::{
    ChatRole, ChatTurn, PromptTemplate, TemplateType, TemplateValidation,
};

// ---------------------------------------------------------------------------
// render_chat: every template variant renders without error
// ---------------------------------------------------------------------------

#[test]
fn render_chat_all_variants_single_user_turn() {
    let history = vec![ChatTurn::new(ChatRole::User, "Hello")];
    for &t in TemplateType::all_variants() {
        let result = t.render_chat(&history, None);
        assert!(result.is_ok(), "{t:?} failed render_chat: {:?}", result.err());
        let output = result.unwrap();
        assert!(!output.is_empty(), "{t:?} produced empty output");
    }
}

#[test]
fn render_chat_all_variants_with_system() {
    let history = vec![ChatTurn::new(ChatRole::User, "Hello")];
    for &t in TemplateType::all_variants() {
        let result = t.render_chat(&history, Some("You are an assistant."));
        assert!(result.is_ok(), "{t:?} failed render_chat with system: {:?}", result.err());
    }
}

#[test]
fn render_chat_all_variants_multi_turn() {
    let history = vec![
        ChatTurn::new(ChatRole::User, "Hi"),
        ChatTurn::new(ChatRole::Assistant, "Hello!"),
        ChatTurn::new(ChatRole::User, "How are you?"),
    ];
    for &t in TemplateType::all_variants() {
        let result = t.render_chat(&history, Some("Be brief."));
        assert!(result.is_ok(), "{t:?} failed multi-turn render_chat: {:?}", result.err());
        let output = result.unwrap();
        assert!(!output.is_empty(), "{t:?} produced empty multi-turn output");
    }
}

#[test]
fn render_chat_all_variants_empty_history() {
    for &t in TemplateType::all_variants() {
        let result = t.render_chat(&[], None);
        assert!(result.is_ok(), "{t:?} failed render_chat with empty history: {:?}", result.err());
    }
}

// ---------------------------------------------------------------------------
// render_chat: output contains user text
// ---------------------------------------------------------------------------

#[test]
fn render_chat_contains_user_text() {
    let history = vec![ChatTurn::new(ChatRole::User, "UNIQUE_MARKER_42")];
    for &t in TemplateType::all_variants() {
        let result = t.render_chat(&history, None).unwrap();
        assert!(
            result.contains("UNIQUE_MARKER_42"),
            "{t:?} render_chat output missing user text: {result}"
        );
    }
}

#[test]
fn render_chat_contains_assistant_text() {
    let history = vec![
        ChatTurn::new(ChatRole::User, "question"),
        ChatTurn::new(ChatRole::Assistant, "ASSISTANT_MARKER_99"),
    ];
    for &t in TemplateType::all_variants() {
        let result = t.render_chat(&history, None).unwrap();
        // FillInMiddle doesn't render assistant turns
        if matches!(t, TemplateType::FillInMiddle) {
            continue;
        }
        assert!(
            result.contains("ASSISTANT_MARKER_99"),
            "{t:?} render_chat output missing assistant text: {result}"
        );
    }
}

// ---------------------------------------------------------------------------
// validate_output: basic behavior
// ---------------------------------------------------------------------------

#[test]
fn validate_output_valid_for_all_templates() {
    for &t in TemplateType::all_variants() {
        let output = t.apply("test input", None);
        let validation = t.validate_output(&output, "test input");
        assert!(
            validation.is_valid,
            "{t:?} validation failed for its own apply output: {:?}",
            validation.warnings
        );
        assert!(
            validation.warnings.is_empty(),
            "{t:?} unexpected warnings: {:?}",
            validation.warnings
        );
    }
}

#[test]
fn validate_output_empty_output_warns() {
    let t = TemplateType::Raw;
    let validation = t.validate_output("", "hello");
    assert!(!validation.is_valid);
    assert!(validation.warnings.iter().any(|w| w.contains("empty")));
}

#[test]
fn validate_output_missing_user_text_warns() {
    let t = TemplateType::Instruct;
    let validation = t.validate_output("some random output", "MISSING_TEXT");
    assert!(!validation.is_valid);
    assert!(validation.warnings.iter().any(|w| w.contains("user text")));
}

#[test]
fn validate_output_empty_user_text_no_missing_warning() {
    let t = TemplateType::Raw;
    let validation = t.validate_output("some output", "");
    // Empty user text should not produce "missing user text" warning
    let has_missing_warning = validation.warnings.iter().any(|w| w.contains("user text"));
    assert!(!has_missing_warning, "Should not warn about empty user text");
}

// ---------------------------------------------------------------------------
// TemplateValidation struct
// ---------------------------------------------------------------------------

#[test]
fn template_validation_debug() {
    let v = TemplateValidation { is_valid: true, warnings: vec![] };
    let dbg = format!("{v:?}");
    assert!(dbg.contains("TemplateValidation"));
    assert!(dbg.contains("true"));
}

#[test]
fn template_validation_clone() {
    let v = TemplateValidation { is_valid: false, warnings: vec!["test warning".to_string()] };
    let v2 = v.clone();
    assert_eq!(v2.is_valid, false);
    assert_eq!(v2.warnings.len(), 1);
    assert_eq!(v2.warnings[0], "test warning");
}

// ---------------------------------------------------------------------------
// TemplateInfo struct
// ---------------------------------------------------------------------------

#[test]
fn template_info_all_variants() {
    for &t in TemplateType::all_variants() {
        let info = t.info();
        assert!(!info.name.is_empty(), "{t:?} has empty info name");
        // stop_sequences may or may not be empty, that's fine
        let _ = info.adds_bos;
        let _ = info.parses_special;
    }
}

#[test]
fn template_info_debug() {
    let info = TemplateType::Phi4Chat.info();
    let dbg = format!("{info:?}");
    assert!(dbg.contains("TemplateInfo"));
}

#[test]
fn template_info_clone() {
    let info = TemplateType::Llama3Chat.info();
    let info2 = info.clone();
    assert_eq!(info.name, info2.name);
    assert_eq!(info.adds_bos, info2.adds_bos);
}

// ---------------------------------------------------------------------------
// PromptTemplate builder
// ---------------------------------------------------------------------------

#[test]
fn prompt_template_builder_basic() {
    let pt = PromptTemplate::new(TemplateType::Phi4Chat);
    assert_eq!(pt.template_type(), TemplateType::Phi4Chat);
    let output = pt.format("Hello");
    assert!(output.contains("Hello"));
}

#[test]
fn prompt_template_builder_with_system() {
    let pt = PromptTemplate::new(TemplateType::Phi4Chat).with_system_prompt("You are a pirate.");
    let output = pt.format("Ahoy!");
    assert!(output.contains("You are a pirate."));
    assert!(output.contains("Ahoy!"));
}

#[test]
fn prompt_template_add_turn_and_clear() {
    let mut pt = PromptTemplate::new(TemplateType::Instruct);
    pt.add_turn("question1", "answer1");
    pt.add_turn("question2", "answer2");
    // After clear, template still works
    pt.clear_history();
    let output = pt.format("new question");
    assert!(output.contains("new question"));
}

#[test]
fn prompt_template_stop_sequences() {
    let pt = PromptTemplate::new(TemplateType::Llama3Chat);
    let stops = pt.stop_sequences();
    // Llama3 should have stop sequences
    assert!(!stops.is_empty(), "Llama3Chat should have stop sequences");
}

#[test]
fn prompt_template_should_add_bos() {
    let pt = PromptTemplate::new(TemplateType::Raw);
    let _ = pt.should_add_bos(); // Should not panic
}

#[test]
fn prompt_template_debug_clone() {
    let pt = PromptTemplate::new(TemplateType::MistralChat).with_system_prompt("test");
    let dbg = format!("{pt:?}");
    assert!(dbg.contains("PromptTemplate"));
    let pt2 = pt.clone();
    assert_eq!(pt2.template_type(), TemplateType::MistralChat);
}

// ---------------------------------------------------------------------------
// all_variants exhaustiveness
// ---------------------------------------------------------------------------

#[test]
fn all_variants_returns_many() {
    let variants = TemplateType::all_variants();
    // We know there are 50+ variants
    assert!(variants.len() >= 50, "Expected 50+ variants, got {}", variants.len());
}

#[test]
fn all_variants_no_duplicates() {
    let variants = TemplateType::all_variants();
    let mut seen = std::collections::HashSet::new();
    for &v in variants {
        let name = v.to_string();
        assert!(seen.insert(name.clone()), "Duplicate variant: {name}");
    }
}

// ---------------------------------------------------------------------------
// suggest_for_arch coverage
// ---------------------------------------------------------------------------

#[test]
fn suggest_for_arch_known_families() {
    let cases = vec![
        ("phi", TemplateType::Phi4Chat),
        ("phi-4", TemplateType::Phi4Chat),
        ("phi-3", TemplateType::Phi3Instruct),
        ("llama", TemplateType::Llama3Chat),
        ("llama2", TemplateType::Llama2Chat),
        ("llama-3.1", TemplateType::Llama31Chat),
        ("llama-3.2", TemplateType::Llama32Chat),
        ("mistral", TemplateType::MistralChat),
        ("mistral-nemo", TemplateType::MistralNemoChat),
        ("mixtral", TemplateType::MixtralInstruct),
        ("qwen", TemplateType::QwenChat),
        ("qwen2.5", TemplateType::Qwen25Chat),
        ("gemma", TemplateType::GemmaChat),
        ("gemma2", TemplateType::Gemma2Chat),
        ("deepseek", TemplateType::DeepSeekChat),
        ("deepseek-v3", TemplateType::DeepSeekV3Chat),
        ("starcoder", TemplateType::StarCoder),
        ("falcon", TemplateType::FalconChat),
        ("falcon-2", TemplateType::Falcon2Chat),
    ];
    for (arch, expected) in cases {
        let result = TemplateType::suggest_for_arch(arch);
        assert_eq!(result, Some(expected), "suggest_for_arch({arch:?}) mismatch");
    }
}

#[test]
fn suggest_for_arch_none_for_unknown() {
    assert_eq!(TemplateType::suggest_for_arch("bert"), None);
    assert_eq!(TemplateType::suggest_for_arch("bitnet"), None);
    assert_eq!(TemplateType::suggest_for_arch("nonexistent-model"), None);
}

#[test]
fn suggest_for_arch_case_insensitive() {
    assert_eq!(TemplateType::suggest_for_arch("PHI"), Some(TemplateType::Phi4Chat));
    assert_eq!(TemplateType::suggest_for_arch("LLAMA"), Some(TemplateType::Llama3Chat));
    assert_eq!(TemplateType::suggest_for_arch("Mistral"), Some(TemplateType::MistralChat));
}

// ---------------------------------------------------------------------------
// default_stop_sequences: all templates
// ---------------------------------------------------------------------------

#[test]
fn default_stop_sequences_all_templates() {
    for &t in TemplateType::all_variants() {
        let stops = t.default_stop_sequences();
        // Just verify it doesn't panic and returns a vec
        let _ = stops;
    }
}

// ---------------------------------------------------------------------------
// should_add_bos / parse_special: all templates
// ---------------------------------------------------------------------------

#[test]
fn should_add_bos_all_templates() {
    for &t in TemplateType::all_variants() {
        let _ = t.should_add_bos(); // Must not panic
    }
}

#[test]
fn parse_special_all_templates() {
    for &t in TemplateType::all_variants() {
        let _ = t.parse_special(); // Must not panic
    }
}

// ---------------------------------------------------------------------------
// render_chat edge cases
// ---------------------------------------------------------------------------

#[test]
fn render_chat_system_role_in_history() {
    let history = vec![
        ChatTurn::new(ChatRole::System, "You are helpful."),
        ChatTurn::new(ChatRole::User, "Hello"),
    ];
    for &t in TemplateType::all_variants() {
        let result = t.render_chat(&history, None);
        assert!(result.is_ok(), "{t:?} failed with system role in history: {:?}", result.err());
    }
}

#[test]
fn render_chat_unicode_content() {
    let history = vec![ChatTurn::new(ChatRole::User, "日本語テスト 🦀 Ü ñ")];
    for &t in TemplateType::all_variants() {
        let result = t.render_chat(&history, Some("你好世界"));
        assert!(result.is_ok(), "{t:?} failed with unicode: {:?}", result.err());
        let output = result.unwrap();
        assert!(output.contains("日本語テスト"), "{t:?} lost unicode text");
    }
}

#[test]
fn render_chat_very_long_text() {
    let long_text = "x".repeat(10_000);
    let history = vec![ChatTurn::new(ChatRole::User, &long_text)];
    for &t in TemplateType::all_variants() {
        let result = t.render_chat(&history, None);
        assert!(result.is_ok(), "{t:?} failed with long text: {:?}", result.err());
    }
}

#[test]
fn render_chat_many_turns() {
    let mut history = Vec::new();
    for i in 0..20 {
        history.push(ChatTurn::new(ChatRole::User, &format!("Question {i}")));
        history.push(ChatTurn::new(ChatRole::Assistant, &format!("Answer {i}")));
    }
    history.push(ChatTurn::new(ChatRole::User, "Final question"));
    // Test just a few representative templates to keep this fast
    let templates = [
        TemplateType::Phi4Chat,
        TemplateType::Llama3Chat,
        TemplateType::MistralChat,
        TemplateType::GemmaChat,
        TemplateType::DeepSeekChat,
    ];
    for &t in &templates {
        let result = t.render_chat(&history, Some("Be concise."));
        assert!(result.is_ok(), "{t:?} failed with many turns: {:?}", result.err());
        let output = result.unwrap();
        assert!(output.contains("Final question"), "{t:?} lost final question");
    }
}

// ---------------------------------------------------------------------------
// ChatTurn and ChatRole
// ---------------------------------------------------------------------------

#[test]
fn chat_role_as_str() {
    assert_eq!(ChatRole::User.as_str(), "user");
    assert_eq!(ChatRole::Assistant.as_str(), "assistant");
    assert_eq!(ChatRole::System.as_str(), "system");
}

#[test]
fn chat_turn_new() {
    let turn = ChatTurn::new(ChatRole::User, "test");
    assert_eq!(turn.role, ChatRole::User);
    assert_eq!(turn.text, "test");
}

// ---------------------------------------------------------------------------
// apply vs render_chat consistency
// ---------------------------------------------------------------------------

#[test]
fn apply_single_turn_contains_user_text() {
    for &t in TemplateType::all_variants() {
        let output = t.apply("MARKER_APPLY", None);
        if !matches!(t, TemplateType::Raw) || !output.is_empty() {
            assert!(output.contains("MARKER_APPLY"), "{t:?} apply() missing user text: {output}");
        }
    }
}
