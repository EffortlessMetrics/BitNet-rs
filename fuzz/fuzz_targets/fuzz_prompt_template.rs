#![no_main]

use arbitrary::Arbitrary;
use bitnet_prompt_templates::{ChatRole, ChatTurn, TemplateType};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct PromptTemplateInput {
    /// Raw bytes for user prompt (may include non-UTF-8, null bytes, etc.).
    user_raw: Vec<u8>,
    /// Raw bytes for system prompt.
    system_raw: Vec<u8>,
    /// Additional chat turns to stress multi-turn rendering.
    extra_turns: Vec<FuzzTurn>,
    /// Template selector.
    template_byte: u8,
}

#[derive(Arbitrary, Debug)]
struct FuzzTurn {
    /// 0=User, 1=Assistant, 2=System.
    role_byte: u8,
    /// Raw content bytes.
    content: Vec<u8>,
}

fuzz_target!(|input: PromptTemplateInput| {
    let user = String::from_utf8_lossy(&input.user_raw).into_owned();
    let system_owned = String::from_utf8_lossy(&input.system_raw).into_owned();
    let system: Option<&str> = if system_owned.is_empty() { None } else { Some(&system_owned) };

    let template = match input.template_byte % 3 {
        0 => TemplateType::Raw,
        1 => TemplateType::Instruct,
        _ => TemplateType::Llama3Chat,
    };

    // apply must never panic for any input
    let result = template.apply(&user, system);
    // Result string must be valid UTF-8 (guaranteed by String)
    let _ = result.len();

    // Single-turn render_chat must never panic
    let turn = ChatTurn::new(ChatRole::User, &user);
    let _ = template.render_chat(&[turn], system);

    // Multi-turn render_chat must never panic
    let mut turns: Vec<ChatTurn> = Vec::new();
    for ft in input.extra_turns.iter().take(32) {
        let role = match ft.role_byte % 3 {
            0 => ChatRole::User,
            1 => ChatRole::Assistant,
            _ => ChatRole::System,
        };
        let content = String::from_utf8_lossy(&ft.content).into_owned();
        turns.push(ChatTurn::new(role, content));
    }
    if !turns.is_empty() {
        let _ = template.render_chat(&turns, system);
    }

    // All three templates must handle the same input without panicking
    for t in [TemplateType::Raw, TemplateType::Instruct, TemplateType::Llama3Chat] {
        let _ = t.apply(&user, system);
        let single = ChatTurn::new(ChatRole::User, &user);
        let _ = t.render_chat(&[single], system);
    }
});
