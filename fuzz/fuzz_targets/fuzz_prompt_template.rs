#![no_main]

use arbitrary::Arbitrary;
use bitnet_prompt_templates::{ChatRole, ChatTurn, TemplateType};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct PromptInput {
    user_raw: Vec<u8>,
    system_raw: Vec<u8>,
    template_idx: u8,
    /// Extra turns for multi-turn chat rendering.
    extra_turn_raw: Vec<u8>,
}

const TEMPLATES: &[TemplateType] =
    &[TemplateType::Raw, TemplateType::Instruct, TemplateType::Llama3Chat];

fuzz_target!(|input: PromptInput| {
    // Limit input size to avoid OOM.
    if input.user_raw.len() > 4096 || input.system_raw.len() > 4096 {
        return;
    }

    let user = std::str::from_utf8(&input.user_raw).unwrap_or("");
    let system_str = std::str::from_utf8(&input.system_raw).unwrap_or("");
    let system: Option<&str> = if system_str.is_empty() { None } else { Some(system_str) };
    let template = TEMPLATES[input.template_idx as usize % TEMPLATES.len()];

    // apply() must never panic and output must contain the user text
    // (unless user text is empty or template is Raw with empty input).
    let output = template.apply(user, system);

    if !user.is_empty() {
        assert!(
            output.contains(user),
            "template {:?} output does not contain user text: output={:?}, user={:?}",
            template,
            &output[..output.len().min(200)],
            &user[..user.len().min(100)],
        );
    }

    // render_chat with a single user turn must not panic.
    let turn = ChatTurn::new(ChatRole::User, user);
    if let Ok(chat_output) = template.render_chat(&[turn], system) {
        if !user.is_empty() {
            assert!(chat_output.contains(user), "render_chat output does not contain user text",);
        }
    }

    // Multi-turn: user + assistant + user must not panic.
    let extra = std::str::from_utf8(&input.extra_turn_raw).unwrap_or("ok");
    let turns = vec![
        ChatTurn::new(ChatRole::User, user),
        ChatTurn::new(ChatRole::Assistant, extra),
        ChatTurn::new(ChatRole::User, "follow-up"),
    ];
    let _ = template.render_chat(&turns, system);
});
