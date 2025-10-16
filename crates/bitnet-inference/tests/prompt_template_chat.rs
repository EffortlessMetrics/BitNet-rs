//! Test Suite 1: Library - Typed Chat Rendering (bitnet-inference)
//!
//! Tests feature spec: chat-repl-ux-polish.md#AC1-typed-chat-rendering
//!
//! This test suite validates the multi-turn chat rendering functionality
//! for the BitNet.rs prompt template system. Tests ensure proper formatting
//! of conversation history with system prompts and multiple turns.
//!
//! **TDD Approach**: These tests compile successfully but fail because the
//! `render_chat` method needs implementation enhancements for full multi-turn
//! support and proper role formatting.

use anyhow::Result;
use bitnet_inference::{ChatRole, ChatTurn, TemplateType};

/// Tests feature spec: chat-repl-ux-polish.md#AC1-llama3-chat-multi-turn
#[test]
fn test_llama3_chat_multi_turn_with_system_prompt() -> Result<()> {
    let template = TemplateType::Llama3Chat;

    // Build multi-turn conversation history
    let history = vec![
        ChatTurn::new(ChatRole::User, "What is Rust?"),
        ChatTurn::new(
            ChatRole::Assistant,
            "Rust is a systems programming language focused on safety and performance.",
        ),
        ChatTurn::new(ChatRole::User, "Can you give me an example?"),
    ];

    let system_prompt = Some("You are a helpful programming assistant.");

    // Render the full conversation
    let rendered = template.render_chat(&history, system_prompt)?;

    // Verify structure includes all expected components
    assert!(
        rendered.starts_with("<|begin_of_text|>"),
        "LLaMA-3 chat must start with begin_of_text token"
    );

    // System prompt should be present
    assert!(
        rendered.contains("<|start_header_id|>system<|end_header_id|>"),
        "System header missing"
    );
    assert!(
        rendered.contains("You are a helpful programming assistant."),
        "System prompt content missing"
    );
    assert!(rendered.contains("<|eot_id|>"), "EOT tokens missing");

    // First user turn
    assert!(rendered.contains("<|start_header_id|>user<|end_header_id|>"), "User header missing");
    assert!(rendered.contains("What is Rust?"), "First user message missing");

    // First assistant turn
    assert!(
        rendered.contains("<|start_header_id|>assistant<|end_header_id|>"),
        "Assistant header missing"
    );
    assert!(
        rendered.contains("Rust is a systems programming language"),
        "First assistant message missing"
    );

    // Second user turn
    assert!(rendered.contains("Can you give me an example?"), "Second user message missing");

    // Must end with assistant header ready for continuation
    assert!(
        rendered.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"),
        "Chat must end with assistant header ready for generation. Got ending: {:?}",
        &rendered[rendered.len().saturating_sub(100)..]
    );

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC1-llama3-chat-no-system
#[test]
fn test_llama3_chat_multi_turn_without_system_prompt() -> Result<()> {
    let template = TemplateType::Llama3Chat;

    let history = vec![
        ChatTurn::new(ChatRole::User, "Hello!"),
        ChatTurn::new(ChatRole::Assistant, "Hi there! How can I help you today?"),
        ChatTurn::new(ChatRole::User, "Tell me a joke."),
    ];

    let rendered = template.render_chat(&history, None)?;

    // Should start directly with begin_of_text, no system prompt
    assert!(rendered.starts_with("<|begin_of_text|>"), "Must start with begin_of_text");

    // Should NOT contain system header
    assert!(
        !rendered.contains("<|start_header_id|>system<|end_header_id|>"),
        "Should not have system header when no system prompt provided"
    );

    // Should contain all user/assistant turns
    assert!(rendered.contains("Hello!"), "First user turn missing");
    assert!(rendered.contains("Hi there!"), "First assistant turn missing");
    assert!(rendered.contains("Tell me a joke."), "Second user turn missing");

    // Must end ready for assistant response
    assert!(
        rendered.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"),
        "Must end with assistant header"
    );

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC1-llama3-eot-tokens
#[test]
fn test_llama3_chat_eot_token_placement() -> Result<()> {
    let template = TemplateType::Llama3Chat;

    let history = vec![
        ChatTurn::new(ChatRole::User, "First"),
        ChatTurn::new(ChatRole::Assistant, "Response"),
    ];

    let rendered = template.render_chat(&history, Some("System"))?;

    // Count EOT tokens - should have one after each complete turn
    let eot_count = rendered.matches("<|eot_id|>").count();

    // System (1) + User (1) + Assistant (1) = 3 EOT tokens
    assert_eq!(
        eot_count, 3,
        "Expected 3 EOT tokens (system, user, assistant). Found: {}",
        eot_count
    );

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC1-instruct-multi-turn
#[test]
fn test_instruct_template_multi_turn_with_system() -> Result<()> {
    let template = TemplateType::Instruct;

    let history = vec![
        ChatTurn::new(ChatRole::User, "What is 2+2?"),
        ChatTurn::new(ChatRole::Assistant, "2+2 equals 4."),
        ChatTurn::new(ChatRole::User, "What about 3+3?"),
    ];

    let system_prompt = Some("You are a math tutor.");

    let rendered = template.render_chat(&history, system_prompt)?;

    // System prompt should be at the beginning
    assert!(rendered.starts_with("System: You are a math tutor."), "System prompt should be first");

    // User turns should be formatted as "Q: {text}\n"
    assert!(rendered.contains("Q: What is 2+2?"), "First user turn missing Q: prefix");
    assert!(rendered.contains("Q: What about 3+3?"), "Second user turn missing Q: prefix");

    // Assistant turns should be formatted as "A: {text}\n"
    assert!(rendered.contains("A: 2+2 equals 4."), "Assistant turn missing A: prefix");

    // Should end with "A: " ready for response
    assert!(rendered.ends_with("A: "), "Should end with 'A: ' ready for continuation");

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC1-instruct-no-system
#[test]
fn test_instruct_template_multi_turn_without_system() -> Result<()> {
    let template = TemplateType::Instruct;

    let history = vec![
        ChatTurn::new(ChatRole::User, "Hello"),
        ChatTurn::new(ChatRole::Assistant, "Hi!"),
        ChatTurn::new(ChatRole::User, "How are you?"),
    ];

    let rendered = template.render_chat(&history, None)?;

    // Should NOT start with System:
    assert!(!rendered.starts_with("System:"), "Should not have system prompt");

    // Should contain all Q&A pairs
    assert!(rendered.contains("Q: Hello"), "First Q missing");
    assert!(rendered.contains("A: Hi!"), "First A missing");
    assert!(rendered.contains("Q: How are you?"), "Second Q missing");

    assert!(rendered.ends_with("A: "), "Should end with A: prompt");

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC1-raw-full-history-concatenation
#[test]
fn test_raw_template_concatenates_full_history() -> Result<()> {
    let template = TemplateType::Raw;

    // CRITICAL REGRESSION TEST: Raw template currently only takes the LAST user message
    // This test verifies the bug is fixed and full history is preserved
    let history = vec![
        ChatTurn::new(ChatRole::User, "First user message"),
        ChatTurn::new(ChatRole::Assistant, "First assistant response"),
        ChatTurn::new(ChatRole::User, "Second user message"),
        ChatTurn::new(ChatRole::Assistant, "Second assistant response"),
        ChatTurn::new(ChatRole::User, "Third user message"),
    ];

    let rendered = template.render_chat(&history, None)?;

    // CRITICAL: All messages should be present, not just the last user message
    assert!(
        rendered.contains("First user message"),
        "Raw template should preserve first user message. Current implementation drops it!"
    );
    assert!(
        rendered.contains("First assistant response"),
        "Raw template should preserve first assistant response"
    );
    assert!(
        rendered.contains("Second user message"),
        "Raw template should preserve second user message"
    );
    assert!(
        rendered.contains("Second assistant response"),
        "Raw template should preserve second assistant response"
    );
    assert!(
        rendered.contains("Third user message"),
        "Raw template should preserve third user message"
    );

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC1-raw-with-system
#[test]
fn test_raw_template_with_system_prompt() -> Result<()> {
    let template = TemplateType::Raw;

    let history = vec![
        ChatTurn::new(ChatRole::User, "Question 1"),
        ChatTurn::new(ChatRole::Assistant, "Answer 1"),
        ChatTurn::new(ChatRole::User, "Question 2"),
    ];

    let system_prompt = Some("This is the system context.");

    let rendered = template.render_chat(&history, system_prompt)?;

    // System prompt should be prepended
    assert!(
        rendered.starts_with("This is the system context.") || rendered.contains("system context"),
        "System prompt should be included in raw template"
    );

    // All history should still be present
    assert!(rendered.contains("Question 1"), "First user message missing");
    assert!(rendered.contains("Answer 1"), "First assistant message missing");
    assert!(rendered.contains("Question 2"), "Second user message missing");

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC1-should-add-bos
#[test]
fn test_should_add_bos_behavior() {
    // LLaMA-3 chat should NOT add BOS (it's embedded in the template)
    assert!(
        !TemplateType::Llama3Chat.should_add_bos(),
        "Llama3Chat should return false (BOS embedded via <|begin_of_text|>)"
    );

    // Instruct should add BOS
    assert!(TemplateType::Instruct.should_add_bos(), "Instruct template should add BOS token");

    // Raw should add BOS
    assert!(TemplateType::Raw.should_add_bos(), "Raw template should add BOS token");
}

/// Tests feature spec: chat-repl-ux-polish.md#AC1-empty-history
#[test]
fn test_render_chat_with_empty_history() -> Result<()> {
    let template = TemplateType::Llama3Chat;
    let history: Vec<ChatTurn> = vec![];
    let system_prompt = Some("System message");

    let rendered = template.render_chat(&history, system_prompt)?;

    // Should have begin_of_text + system + assistant header
    assert!(rendered.contains("<|begin_of_text|>"), "Should start with begin_of_text");
    assert!(rendered.contains("System message"), "Should contain system prompt");
    assert!(
        rendered.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"),
        "Should end ready for assistant response"
    );

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC1-instruct-role-filtering
#[test]
fn test_instruct_template_ignores_system_role_in_history() -> Result<()> {
    let template = TemplateType::Instruct;

    // Include a system role in history (should be ignored since system is separate param)
    let history = vec![
        ChatTurn::new(ChatRole::System, "System in history (should be ignored)"),
        ChatTurn::new(ChatRole::User, "User message"),
        ChatTurn::new(ChatRole::Assistant, "Assistant response"),
    ];

    let rendered = template.render_chat(&history, None)?;

    // System role in history should not appear as Q: or A:
    assert!(
        !rendered.contains("Q: System in history"),
        "System role in history should not be formatted as Q:"
    );
    assert!(
        !rendered.contains("A: System in history"),
        "System role in history should not be formatted as A:"
    );

    // Only user and assistant turns should be formatted
    assert!(rendered.contains("Q: User message"), "User turn should be formatted");
    assert!(rendered.contains("A: Assistant response"), "Assistant turn should be formatted");

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC1-long-conversation
#[test]
fn test_render_chat_handles_long_conversation() -> Result<()> {
    let template = TemplateType::Instruct;

    // Create a long conversation (10 turns)
    let mut history = Vec::new();
    for i in 0..10 {
        history.push(ChatTurn::new(ChatRole::User, format!("Question {}", i + 1)));
        history.push(ChatTurn::new(ChatRole::Assistant, format!("Answer {}", i + 1)));
    }

    history.push(ChatTurn::new(ChatRole::User, "Final question"));

    let rendered = template.render_chat(&history, None)?;

    // Verify all turns are present
    for i in 0..10 {
        assert!(
            rendered.contains(&format!("Question {}", i + 1)),
            "Turn {} user message missing",
            i + 1
        );
        assert!(
            rendered.contains(&format!("Answer {}", i + 1)),
            "Turn {} assistant message missing",
            i + 1
        );
    }

    assert!(rendered.contains("Final question"), "Final turn missing");
    assert!(rendered.ends_with("A: "), "Should end with A: prompt");

    Ok(())
}
