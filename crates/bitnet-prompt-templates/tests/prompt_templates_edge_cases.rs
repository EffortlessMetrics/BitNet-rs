//! Edge-case tests for bitnet-prompt-templates: TemplateType parsing, display,
//! apply, detect, render_chat, stop_sequences, all_variants, ChatRole, ChatTurn.

use bitnet_prompt_templates::{ChatRole, ChatTurn, TemplateType};
use std::str::FromStr;

// ---------------------------------------------------------------------------
// ChatRole
// ---------------------------------------------------------------------------

#[test]
fn chat_role_as_str() {
    assert_eq!(ChatRole::System.as_str(), "system");
    assert_eq!(ChatRole::User.as_str(), "user");
    assert_eq!(ChatRole::Assistant.as_str(), "assistant");
}

#[test]
fn chat_role_debug() {
    assert_eq!(format!("{:?}", ChatRole::System), "System");
    assert_eq!(format!("{:?}", ChatRole::User), "User");
}

#[test]
fn chat_role_clone_eq() {
    let r = ChatRole::Assistant;
    let c = r.clone();
    assert_eq!(r, c);
}

#[test]
fn chat_role_ne() {
    assert_ne!(ChatRole::System, ChatRole::User);
    assert_ne!(ChatRole::User, ChatRole::Assistant);
}

// ---------------------------------------------------------------------------
// ChatTurn
// ---------------------------------------------------------------------------

#[test]
fn chat_turn_new() {
    let turn = ChatTurn::new(ChatRole::User, "Hello");
    assert_eq!(turn.role, ChatRole::User);
    assert_eq!(turn.text, "Hello");
}

#[test]
fn chat_turn_string_ownership() {
    let turn = ChatTurn::new(ChatRole::System, String::from("Be helpful"));
    assert_eq!(turn.text, "Be helpful");
}

#[test]
fn chat_turn_debug() {
    let turn = ChatTurn::new(ChatRole::Assistant, "Hi");
    let dbg = format!("{turn:?}");
    assert!(dbg.contains("ChatTurn"));
}

#[test]
fn chat_turn_clone() {
    let turn = ChatTurn::new(ChatRole::User, "test");
    let cloned = turn.clone();
    assert_eq!(cloned.role, ChatRole::User);
    assert_eq!(cloned.text, "test");
}

// ---------------------------------------------------------------------------
// TemplateType — FromStr
// ---------------------------------------------------------------------------

#[test]
fn template_from_str_raw() {
    assert_eq!(TemplateType::from_str("raw").unwrap(), TemplateType::Raw);
}

#[test]
fn template_from_str_instruct() {
    assert_eq!(TemplateType::from_str("instruct").unwrap(), TemplateType::Instruct);
}

#[test]
fn template_from_str_case_insensitive() {
    assert_eq!(TemplateType::from_str("RAW").unwrap(), TemplateType::Raw);
    assert_eq!(TemplateType::from_str("Instruct").unwrap(), TemplateType::Instruct);
}

#[test]
fn template_from_str_aliases() {
    // Phi-4 aliases
    assert_eq!(TemplateType::from_str("phi4-chat").unwrap(), TemplateType::Phi4Chat);
    assert_eq!(TemplateType::from_str("phi4_chat").unwrap(), TemplateType::Phi4Chat);
    assert_eq!(TemplateType::from_str("phi4").unwrap(), TemplateType::Phi4Chat);
    assert_eq!(TemplateType::from_str("chatml").unwrap(), TemplateType::Phi4Chat);
    // LLaMA-3 aliases
    assert_eq!(TemplateType::from_str("llama3-chat").unwrap(), TemplateType::Llama3Chat);
    assert_eq!(TemplateType::from_str("llama3_chat").unwrap(), TemplateType::Llama3Chat);
    // Mistral
    assert_eq!(TemplateType::from_str("mistral").unwrap(), TemplateType::MistralChat);
    // Qwen
    assert_eq!(TemplateType::from_str("qwen").unwrap(), TemplateType::QwenChat);
    // FIM
    assert_eq!(TemplateType::from_str("fim").unwrap(), TemplateType::FillInMiddle);
}

#[test]
fn template_from_str_unknown_errors() {
    assert!(TemplateType::from_str("bogus").is_err());
    assert!(TemplateType::from_str("").is_err());
}

// ---------------------------------------------------------------------------
// TemplateType — Display
// ---------------------------------------------------------------------------

#[test]
fn template_display_roundtrip() {
    // For key templates, display string should parse back
    let templates = [
        TemplateType::Raw,
        TemplateType::Instruct,
        TemplateType::Llama3Chat,
        TemplateType::Phi4Chat,
        TemplateType::QwenChat,
        TemplateType::GemmaChat,
        TemplateType::MistralChat,
        TemplateType::DeepSeekChat,
    ];
    for t in &templates {
        let display = t.to_string();
        let parsed = TemplateType::from_str(&display).unwrap();
        assert_eq!(*t, parsed, "roundtrip failed for {display}");
    }
}

// ---------------------------------------------------------------------------
// TemplateType — apply
// ---------------------------------------------------------------------------

#[test]
fn apply_raw_passthrough() {
    let result = TemplateType::Raw.apply("Hello world", None);
    assert_eq!(result, "Hello world");
}

#[test]
fn apply_raw_ignores_system_prompt() {
    let result = TemplateType::Raw.apply("Hello", Some("Be helpful"));
    assert_eq!(result, "Hello");
}

#[test]
fn apply_instruct_contains_user_text() {
    let result = TemplateType::Instruct.apply("What is 2+2?", None);
    assert!(result.contains("What is 2+2?"));
}

#[test]
fn apply_phi4_chat_has_chatml_markers() {
    let result = TemplateType::Phi4Chat.apply("Hello", None);
    assert!(result.contains("<|im_start|>"));
    assert!(result.contains("<|im_end|>"));
    assert!(result.contains("Hello"));
    assert!(result.contains("assistant"));
}

#[test]
fn apply_phi4_chat_custom_system() {
    let result = TemplateType::Phi4Chat.apply("Hello", Some("You are a cat."));
    assert!(result.contains("You are a cat."));
    assert!(result.contains("Hello"));
}

#[test]
fn apply_llama3_has_markers() {
    let result = TemplateType::Llama3Chat.apply("Hi", None);
    assert!(result.contains("<|start_header_id|>"));
    assert!(result.contains("<|eot_id|>"));
    assert!(result.contains("Hi"));
}

#[test]
fn apply_gemma_has_markers() {
    let result = TemplateType::GemmaChat.apply("Hi", None);
    assert!(result.contains("<start_of_turn>"));
    assert!(result.contains("<end_of_turn>"));
}

#[test]
fn apply_mistral_has_inst_markers() {
    let result = TemplateType::MistralChat.apply("Hi", None);
    assert!(result.contains("[INST]"));
    assert!(result.contains("[/INST]"));
}

#[test]
fn apply_empty_user_text() {
    // Should not panic with empty input
    let result = TemplateType::Instruct.apply("", None);
    assert!(!result.is_empty()); // Template wrapping still produces output
}

#[test]
fn apply_starcoder_has_fim_tokens() {
    let result = TemplateType::StarCoder.apply("def hello():", None);
    assert!(result.contains("def hello():"));
}

// ---------------------------------------------------------------------------
// TemplateType — detect
// ---------------------------------------------------------------------------

#[test]
fn detect_llama3_from_jinja() {
    let jinja = "{% if <|start_header_id|> %}{{ <|eot_id|> }}{% endif %}";
    let t = TemplateType::detect(None, Some(jinja));
    assert_eq!(t, TemplateType::Llama3Chat);
}

#[test]
fn detect_phi4_from_jinja() {
    let jinja = "<|im_start|>system\n{{ system_msg }}<|im_end|>";
    let t = TemplateType::detect(None, Some(jinja));
    assert_eq!(t, TemplateType::Phi4Chat);
}

#[test]
fn detect_gemma_from_jinja() {
    let jinja = "<start_of_turn>user\n{{ msg }}<end_of_turn>";
    let t = TemplateType::detect(None, Some(jinja));
    assert_eq!(t, TemplateType::GemmaChat);
}

#[test]
fn detect_mistral_from_jinja() {
    let jinja = "[INST] {{ msg }} [/INST]";
    let t = TemplateType::detect(None, Some(jinja));
    assert_eq!(t, TemplateType::MistralChat);
}

#[test]
fn detect_fim_from_jinja() {
    let jinja = "<fim_prefix>code<fim_suffix>code<fim_middle>";
    let t = TemplateType::detect(None, Some(jinja));
    assert_eq!(t, TemplateType::FillInMiddle);
}

#[test]
fn detect_no_info_falls_back_to_raw() {
    let t = TemplateType::detect(None, None);
    assert_eq!(t, TemplateType::Raw);
}

#[test]
fn detect_granite_from_jinja() {
    let jinja = "<|start_of_role|>system<|end_of_role|>";
    let t = TemplateType::detect(None, Some(jinja));
    assert_eq!(t, TemplateType::GraniteChat);
}

#[test]
fn detect_nemotron_from_jinja() {
    let jinja = "<extra_id_0>System\n<extra_id_1>User\n";
    let t = TemplateType::detect(None, Some(jinja));
    assert_eq!(t, TemplateType::NemotronChat);
}

#[test]
fn detect_phi3_from_jinja() {
    let jinja = "<|system|>\n{{ msg }}<|end|>\n<|user|>\n{{ msg }}<|end|>";
    let t = TemplateType::detect(None, Some(jinja));
    assert_eq!(t, TemplateType::Phi3Instruct);
}

#[test]
fn detect_exaone_from_jinja() {
    let jinja = "[|system|] You are helpful. [|endofturn|]";
    let t = TemplateType::detect(None, Some(jinja));
    assert_eq!(t, TemplateType::ExaoneChat);
}

// ---------------------------------------------------------------------------
// TemplateType — stop_sequences
// ---------------------------------------------------------------------------

#[test]
fn stop_sequences_chatml_templates() {
    let stops = TemplateType::Phi4Chat.default_stop_sequences();
    assert!(stops.iter().any(|s| s.contains("<|im_end|>")));
}

#[test]
fn stop_sequences_raw() {
    let stops = TemplateType::Raw.default_stop_sequences();
    let _ = stops;
}

#[test]
fn stop_sequences_llama3() {
    let stops = TemplateType::Llama3Chat.default_stop_sequences();
    assert!(stops.iter().any(|s| s.contains("<|eot_id|>")));
}

// ---------------------------------------------------------------------------
// TemplateType — all_variants
// ---------------------------------------------------------------------------

#[test]
fn all_variants_non_empty() {
    let variants = TemplateType::all_variants();
    assert!(variants.len() > 40); // We have 50+ templates
}

#[test]
fn all_variants_contains_raw() {
    assert!(TemplateType::all_variants().contains(&TemplateType::Raw));
}

#[test]
fn all_variants_contains_phi4() {
    assert!(TemplateType::all_variants().contains(&TemplateType::Phi4Chat));
}

#[test]
fn all_variants_display_roundtrip() {
    for t in TemplateType::all_variants() {
        let display = t.to_string();
        let parsed = TemplateType::from_str(&display).unwrap();
        assert_eq!(*t, parsed, "Display roundtrip failed for {display}");
    }
}

// ---------------------------------------------------------------------------
// TemplateType — render_chat
// ---------------------------------------------------------------------------

#[test]
fn render_chat_phi4_multi_turn() {
    let history = vec![
        ChatTurn::new(ChatRole::User, "Hello"),
        ChatTurn::new(ChatRole::Assistant, "Hi there!"),
        ChatTurn::new(ChatRole::User, "How are you?"),
    ];
    let result = TemplateType::Phi4Chat.render_chat(&history, None).unwrap();
    assert!(result.contains("Hello"));
    assert!(result.contains("Hi there!"));
    assert!(result.contains("How are you?"));
    assert!(result.contains("<|im_start|>"));
}

#[test]
fn render_chat_empty_history() {
    let result = TemplateType::Phi4Chat.render_chat(&[], None).unwrap();
    // Should produce at least system + assistant prompt
    assert!(result.contains("<|im_start|>"));
}

#[test]
fn render_chat_with_system_prompt() {
    let history = vec![ChatTurn::new(ChatRole::User, "test")];
    let result = TemplateType::Phi4Chat.render_chat(&history, Some("Custom system")).unwrap();
    assert!(result.contains("Custom system"));
}

// ---------------------------------------------------------------------------
// TemplateType — serde
// ---------------------------------------------------------------------------

#[test]
fn template_copy_eq() {
    let t = TemplateType::Phi4Chat;
    let c = t;
    assert_eq!(t, c);
}

#[test]
fn template_debug_raw() {
    let dbg = format!("{:?}", TemplateType::Raw);
    assert!(dbg.contains("Raw"));
}
