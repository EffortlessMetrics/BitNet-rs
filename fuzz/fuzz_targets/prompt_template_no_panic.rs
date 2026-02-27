#![no_main]

use arbitrary::Arbitrary;
use bitnet_prompt_templates::{ChatRole, ChatTurn, TemplateType};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct TemplateInput {
    /// Raw bytes used as the user prompt (non-UTF-8 sequences are skipped).
    user_raw: Vec<u8>,
    /// Raw bytes used as the optional system prompt.
    system_raw: Vec<u8>,
}

fuzz_target!(|input: TemplateInput| {
    let user = std::str::from_utf8(&input.user_raw).unwrap_or("");
    let system_str = std::str::from_utf8(&input.system_raw).unwrap_or("");
    let system: Option<&str> = if system_str.is_empty() { None } else { Some(system_str) };

    for template in [TemplateType::Raw, TemplateType::Instruct, TemplateType::Llama3Chat] {
        // `apply` must never panic for any combination of template + text.
        let _ = template.apply(user, system);

        // `render_chat` with a single user turn must never panic.
        let turn = ChatTurn::new(ChatRole::User, user);
        let _ = template.render_chat(&[turn], system);
    }
});
