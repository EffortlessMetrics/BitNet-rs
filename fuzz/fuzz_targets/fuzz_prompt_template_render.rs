#![no_main]

use arbitrary::Arbitrary;
use bitnet_prompt_templates::{ChatRole, ChatTurn, PromptTemplate, TemplateType};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct TemplateRenderInput {
    user_raw: Vec<u8>,
    system_raw: Vec<u8>,
    tokenizer_name_raw: Vec<u8>,
    jinja_raw: Vec<u8>,
    multi_turn_users: Vec<Vec<u8>>,
    multi_turn_assistants: Vec<Vec<u8>>,
}

fuzz_target!(|input: TemplateRenderInput| {
    let user = std::str::from_utf8(&input.user_raw).unwrap_or("");
    let system_str = std::str::from_utf8(&input.system_raw).unwrap_or("");
    let system: Option<&str> = if system_str.is_empty() { None } else { Some(system_str) };

    // Test detect with arbitrary tokenizer name and jinja template.
    let tok_name = std::str::from_utf8(&input.tokenizer_name_raw).ok();
    let jinja = std::str::from_utf8(&input.jinja_raw).ok();
    let detected = TemplateType::detect(tok_name, jinja);
    let _ = detected.apply(user, system);

    // Test all template variants with arbitrary text.
    for &template in TemplateType::all_variants() {
        let _ = template.apply(user, system);
        let _ = template.default_stop_sequences();
        let _ = template.validate_output(user, user);

        // Single-turn render_chat must not panic.
        let turn = ChatTurn::new(ChatRole::User, user);
        let _ = template.render_chat(&[turn], system);
    }

    // Multi-turn chat must not panic.
    let turns: Vec<ChatTurn> = input
        .multi_turn_users
        .iter()
        .zip(input.multi_turn_assistants.iter())
        .take(8)
        .flat_map(|(u, a)| {
            let u_str = std::str::from_utf8(u).unwrap_or("");
            let a_str = std::str::from_utf8(a).unwrap_or("");
            [ChatTurn::new(ChatRole::User, u_str), ChatTurn::new(ChatRole::Assistant, a_str)]
        })
        .collect();

    for &template in TemplateType::all_variants() {
        let _ = template.render_chat(&turns, system);
    }

    // PromptTemplate builder API must not panic.
    let mut pt = PromptTemplate::new(detected);
    if let Some(s) = system {
        pt = pt.with_system_prompt(s);
    }
    let _ = pt.format(user);
    let _ = pt.stop_sequences();
});
