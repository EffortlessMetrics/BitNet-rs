#![no_main]

use arbitrary::Arbitrary;
use bitnet_prompt_templates::{ChatRole, ChatTurn, PromptTemplate, TemplateType};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct PromptInput {
    /// Template variant selector.
    template_idx: u8,
    /// System prompt bytes (may be non-UTF-8).
    system_raw: Vec<u8>,
    /// Multi-turn conversation: pairs of (user, assistant) messages.
    turns: Vec<(Vec<u8>, Vec<u8>)>,
    /// Final user prompt.
    user_raw: Vec<u8>,
    /// Tokenizer name hint for detect().
    tokenizer_hint: Vec<u8>,
    /// Jinja template hint for detect().
    jinja_hint: Vec<u8>,
}

fuzz_target!(|input: PromptInput| {
    let templates = [TemplateType::Raw, TemplateType::Instruct, TemplateType::Llama3Chat];
    let template = templates[input.template_idx as usize % templates.len()];

    let user = std::str::from_utf8(&input.user_raw).unwrap_or("");
    let system_str = std::str::from_utf8(&input.system_raw).unwrap_or("");
    let system: Option<&str> = if system_str.is_empty() { None } else { Some(system_str) };

    // PromptTemplate with multi-turn history must never panic.
    let mut pt = PromptTemplate::new(template);
    if let Some(s) = system {
        pt = pt.with_system_prompt(s);
    }
    for (u, a) in input.turns.iter().take(8) {
        let u_str = std::str::from_utf8(u).unwrap_or("");
        let a_str = std::str::from_utf8(a).unwrap_or("");
        pt.add_turn(u_str, a_str);
    }
    let _ = pt.format(user);
    let _ = pt.stop_sequences();
    let _ = pt.should_add_bos();
    let _ = pt.template_type();
    pt.clear_history();
    let _ = pt.format(user);

    // render_chat with many turns must never panic.
    let turns: Vec<ChatTurn> = input
        .turns
        .iter()
        .take(8)
        .flat_map(|(u, a)| {
            let u_str = std::str::from_utf8(u).unwrap_or("hi");
            let a_str = std::str::from_utf8(a).unwrap_or("ok");
            [ChatTurn::new(ChatRole::User, u_str), ChatTurn::new(ChatRole::Assistant, a_str)]
        })
        .collect();
    let _ = template.render_chat(&turns, system);

    // detect() must never panic on arbitrary tokenizer/jinja hints.
    let tok_hint = std::str::from_utf8(&input.tokenizer_hint).ok();
    let jinja_hint = std::str::from_utf8(&input.jinja_hint).ok();
    let _ = TemplateType::detect(tok_hint, jinja_hint);

    // default_stop_sequences and parse_special must not panic.
    let _ = template.default_stop_sequences();
    let _ = template.parse_special();
});
