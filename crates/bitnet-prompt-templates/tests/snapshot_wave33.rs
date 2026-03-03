//! Wave 33 snapshot tests for bitnet-prompt-templates.
//!
//! Covers: template rendering output for each key template type,
//! stop sequences, TemplateInfo, TemplateValidation, multi-turn,
//! PromptTemplate builder patterns.

use bitnet_prompt_templates::{ChatRole, ChatTurn, PromptTemplate, TemplateInfo, TemplateType};

// ── Raw template ────────────────────────────────────────────────────────────

#[test]
fn w33_raw_empty_prompt() {
    let out = TemplateType::Raw.apply("", None);
    insta::assert_snapshot!(out);
}

#[test]
fn w33_raw_simple_prompt() {
    let out = TemplateType::Raw.apply("Hello, world!", None);
    insta::assert_snapshot!(out);
}

#[test]
fn w33_raw_ignores_system_prompt() {
    let out = TemplateType::Raw.apply("Hello", Some("You are a bot."));
    insta::assert_snapshot!(out);
}

// ── Instruct template ───────────────────────────────────────────────────────

#[test]
fn w33_instruct_simple() {
    let out = TemplateType::Instruct.apply("What is 2+2?", None);
    insta::assert_snapshot!(out);
}

#[test]
fn w33_instruct_with_system() {
    let out = TemplateType::Instruct.apply("What is 2+2?", Some("You are a math tutor."));
    insta::assert_snapshot!(out);
}

// ── Llama3Chat template ─────────────────────────────────────────────────────

#[test]
fn w33_llama3_simple() {
    let out = TemplateType::Llama3Chat.apply("What is the capital of France?", None);
    insta::assert_snapshot!(out);
}

#[test]
fn w33_llama3_with_system() {
    let out =
        TemplateType::Llama3Chat.apply("Explain gravity.", Some("You are a science teacher."));
    insta::assert_snapshot!(out);
}

// ── Phi4Chat template ───────────────────────────────────────────────────────

#[test]
fn w33_phi4_simple() {
    let out = TemplateType::Phi4Chat.apply("Summarize this text.", None);
    insta::assert_snapshot!(out);
}

#[test]
fn w33_phi4_with_system() {
    let out = TemplateType::Phi4Chat.apply("Summarize.", Some("Be concise."));
    insta::assert_snapshot!(out);
}

// ── GemmaChat template ─────────────────────────────────────────────────────

#[test]
fn w33_gemma_simple() {
    let out = TemplateType::GemmaChat.apply("Tell me a joke.", None);
    insta::assert_snapshot!(out);
}

// ── MistralChat template ────────────────────────────────────────────────────

#[test]
fn w33_mistral_simple() {
    let out = TemplateType::MistralChat.apply("How does Rust work?", None);
    insta::assert_snapshot!(out);
}

// ── Multi-turn render_chat ──────────────────────────────────────────────────

#[test]
fn w33_llama3_render_chat_multi_turn() {
    let history = vec![
        ChatTurn::new(ChatRole::User, "Hello"),
        ChatTurn::new(ChatRole::Assistant, "Hi there!"),
        ChatTurn::new(ChatRole::User, "What is Rust?"),
    ];
    let out = TemplateType::Llama3Chat
        .render_chat(&history, Some("You are a programming assistant."))
        .unwrap();
    insta::assert_snapshot!(out);
}

#[test]
fn w33_phi4_render_chat_multi_turn() {
    let history = vec![
        ChatTurn::new(ChatRole::User, "Explain closures"),
        ChatTurn::new(ChatRole::Assistant, "Closures capture their environment."),
        ChatTurn::new(ChatRole::User, "Give an example"),
    ];
    let out =
        TemplateType::Phi4Chat.render_chat(&history, Some("You are a Rust teacher.")).unwrap();
    insta::assert_snapshot!(out);
}

#[test]
fn w33_instruct_render_chat_single() {
    let history = vec![ChatTurn::new(ChatRole::User, "What is 2+2?")];
    let out = TemplateType::Instruct.render_chat(&history, None).unwrap();
    insta::assert_snapshot!(out);
}

// ── Stop sequences ──────────────────────────────────────────────────────────

#[test]
fn w33_stop_sequences_raw() {
    let stops = TemplateType::Raw.default_stop_sequences();
    insta::assert_debug_snapshot!(stops);
}

#[test]
fn w33_stop_sequences_instruct() {
    let stops = TemplateType::Instruct.default_stop_sequences();
    insta::assert_debug_snapshot!(stops);
}

#[test]
fn w33_stop_sequences_llama3() {
    let stops = TemplateType::Llama3Chat.default_stop_sequences();
    insta::assert_debug_snapshot!(stops);
}

#[test]
fn w33_stop_sequences_phi4() {
    let stops = TemplateType::Phi4Chat.default_stop_sequences();
    insta::assert_debug_snapshot!(stops);
}

#[test]
fn w33_stop_sequences_gemma() {
    let stops = TemplateType::GemmaChat.default_stop_sequences();
    insta::assert_debug_snapshot!(stops);
}

#[test]
fn w33_stop_sequences_mistral() {
    let stops = TemplateType::MistralChat.default_stop_sequences();
    insta::assert_debug_snapshot!(stops);
}

// ── TemplateInfo ────────────────────────────────────────────────────────────

#[test]
fn w33_template_info_raw_debug() {
    let info = TemplateType::Raw.info();
    insta::assert_debug_snapshot!(info);
}

#[test]
fn w33_template_info_instruct_debug() {
    let info = TemplateType::Instruct.info();
    insta::assert_debug_snapshot!(info);
}

#[test]
fn w33_template_info_llama3_debug() {
    let info = TemplateType::Llama3Chat.info();
    insta::assert_debug_snapshot!(info);
}

// ── TemplateType Display ────────────────────────────────────────────────────

#[test]
fn w33_template_type_display_variants() {
    let variants = vec![
        TemplateType::Raw,
        TemplateType::Instruct,
        TemplateType::Llama3Chat,
        TemplateType::Phi4Chat,
        TemplateType::QwenChat,
        TemplateType::GemmaChat,
        TemplateType::MistralChat,
        TemplateType::DeepSeekChat,
    ];
    let display: Vec<String> = variants.iter().map(|t| t.to_string()).collect();
    insta::assert_debug_snapshot!(display);
}

// ── TemplateType parse round-trip ───────────────────────────────────────────

#[test]
fn w33_template_type_parse_round_trip() {
    let types = vec!["raw", "instruct", "llama3-chat", "phi4-chat", "gemma-chat", "mistral-chat"];
    let parsed: Vec<String> = types
        .iter()
        .map(|s| {
            let t: TemplateType = s.parse().unwrap();
            format!("{s} -> {t}")
        })
        .collect();
    insta::assert_debug_snapshot!(parsed);
}

#[test]
fn w33_template_type_parse_unknown_error() {
    let err = "nonexistent-template".parse::<TemplateType>().unwrap_err();
    // Just snapshot the start of the error message (it's long)
    let msg = format!("{err}");
    let truncated = if msg.len() > 100 { format!("{}...", &msg[..100]) } else { msg };
    insta::assert_snapshot!(truncated);
}

// ── PromptTemplate builder ──────────────────────────────────────────────────

#[test]
fn w33_prompt_template_builder_debug() {
    let tmpl = PromptTemplate::new(TemplateType::Llama3Chat).with_system_prompt("You are helpful.");
    insta::assert_debug_snapshot!(tmpl);
}

#[test]
fn w33_prompt_template_stop_sequences() {
    let tmpl = PromptTemplate::new(TemplateType::Llama3Chat);
    let stops = tmpl.stop_sequences();
    insta::assert_debug_snapshot!(stops);
}

#[test]
fn w33_prompt_template_should_add_bos() {
    let bos_map: Vec<(String, bool)> = vec![
        TemplateType::Raw,
        TemplateType::Instruct,
        TemplateType::Llama3Chat,
        TemplateType::Phi4Chat,
        TemplateType::GemmaChat,
        TemplateType::MistralChat,
    ]
    .into_iter()
    .map(|t| (t.to_string(), t.should_add_bos()))
    .collect();
    insta::assert_debug_snapshot!(bos_map);
}

// ── TemplateValidation ──────────────────────────────────────────────────────

#[test]
fn w33_validate_valid_output_debug() {
    let t = TemplateType::Instruct;
    let output = t.apply("What is 2+2?", None);
    let validation = t.validate_output(&output, "What is 2+2?");
    insta::assert_debug_snapshot!(validation);
}

#[test]
fn w33_validate_empty_output_debug() {
    let t = TemplateType::Instruct;
    let validation = t.validate_output("", "What is 2+2?");
    insta::assert_debug_snapshot!(validation);
}
