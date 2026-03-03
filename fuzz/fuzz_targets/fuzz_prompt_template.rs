#![no_main]

use arbitrary::Arbitrary;
use bitnet_prompt_templates::{ChatRole, ChatTurn, PromptTemplate, TemplateType};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct PromptTemplateInput {
    user_text_raw: Vec<u8>,
    system_prompt_raw: Vec<u8>,
    history_count: u8,
    history_data: Vec<Vec<u8>>,
    template_idx: u8,
    use_system_prompt: bool,
    use_history: bool,
}

fuzz_target!(|input: PromptTemplateInput| {
    let user_text = String::from_utf8_lossy(&input.user_text_raw);
    let user_text = &user_text[..user_text.len().min(512)];
    let system_prompt = String::from_utf8_lossy(&input.system_prompt_raw);
    let system_prompt = &system_prompt[..system_prompt.len().min(256)];

    let variants = TemplateType::all_variants();
    if variants.is_empty() {
        return;
    }
    let template_type = variants[input.template_idx as usize % variants.len()];

    // --- apply() must never panic ---
    let sys = if input.use_system_prompt { Some(system_prompt.as_ref()) } else { None };
    let output = template_type.apply(user_text, sys);

    // Invariant 1: Output is a valid String (non-empty when user_text is non-empty)
    if !user_text.is_empty() {
        assert!(!output.is_empty(), "apply() produced empty output for non-empty input");
    }

    // Invariant 2: User text appears in the output (templates wrap, not discard)
    if !user_text.is_empty() && user_text.len() <= 128 {
        assert!(
            output.contains(user_text),
            "user text not found in template output for {:?}",
            template_type
        );
    }

    // --- validate_output() must never panic ---
    let validation = template_type.validate_output(&output, user_text);
    let _ = validation.is_valid;

    // --- info() must never panic ---
    let info = template_type.info();
    assert!(!info.name.is_empty(), "template info name should not be empty");

    // --- default_stop_sequences() must never panic ---
    let stops = template_type.default_stop_sequences();
    let _ = stops.len();

    // --- should_add_bos() must never panic ---
    let _ = template_type.should_add_bos();

    // --- PromptTemplate with history ---
    if input.use_history {
        let mut pt = PromptTemplate::new(template_type);
        if input.use_system_prompt {
            pt = pt.with_system_prompt(system_prompt.to_string());
        }

        let n_turns = (input.history_count as usize % 4).min(input.history_data.len() / 2);
        for i in 0..n_turns {
            let user_msg = String::from_utf8_lossy(
                input.history_data.get(i * 2).map(|v| v.as_slice()).unwrap_or(b"hello"),
            );
            let asst_msg = String::from_utf8_lossy(
                input.history_data.get(i * 2 + 1).map(|v| v.as_slice()).unwrap_or(b"hi"),
            );
            pt.add_turn(
                user_msg[..user_msg.len().min(64)].to_string(),
                asst_msg[..asst_msg.len().min(64)].to_string(),
            );
        }

        // format() must never panic
        let formatted = pt.format(user_text);
        assert!(!formatted.is_empty() || user_text.is_empty());

        // stop_sequences() must never panic
        let _ = pt.stop_sequences();

        // clear_history() must never panic
        pt.clear_history();
        let after_clear = pt.format(user_text);
        let _ = after_clear.len();
    }

    // --- render_chat() must never panic ---
    let history = vec![
        ChatTurn::new(ChatRole::User, user_text.to_string()),
        ChatTurn::new(ChatRole::Assistant, "OK".to_string()),
        ChatTurn::new(ChatRole::User, "follow-up"),
    ];
    let _ = template_type.render_chat(&history, sys);

    // --- detect() must never panic on arbitrary strings ---
    let tokenizer_name = std::str::from_utf8(&input.user_text_raw).ok();
    let chat_jinja = std::str::from_utf8(&input.system_prompt_raw).ok();
    let detected = TemplateType::detect(tokenizer_name, chat_jinja);
    let _ = detected.apply(user_text, None);

    // --- suggest_for_arch() must never panic ---
    let _ = TemplateType::suggest_for_arch(user_text);
});
