//! Property-based tests for bitnet-prompt-templates.
//!
//! Verifies key invariants:
//! - `format()` always contains the user text
//! - Adding history does not erase the current user message
//! - Template types produce structurally different output
//! - Round-trip: clear_history restores empty state
//! - System prompt appears in non-Raw formatted output
//! - TemplateType::Raw is an identity transform (no system prompt)
//! - TemplateType::Instruct wraps content with Q/A formatting
//! - TemplateType display/parse round-trips correctly

use bitnet_prompt_templates::{ChatRole, ChatTurn, PromptTemplate, TemplateType};
use proptest::prelude::*;

proptest! {
    /// The formatted output always contains the original user text.
    #[test]
    fn user_text_preserved(user_text in "[a-zA-Z0-9 .,!?]{1,200}") {
        for ttype in [TemplateType::Raw, TemplateType::Instruct, TemplateType::Llama3Chat, TemplateType::Phi4Chat, TemplateType::QwenChat, TemplateType::GemmaChat, TemplateType::MistralChat, TemplateType::DeepSeekChat, TemplateType::StarCoder, TemplateType::FalconChat, TemplateType::CodeLlamaInstruct, TemplateType::CohereCommand, TemplateType::InternLMChat, TemplateType::YiChat, TemplateType::BaichuanChat, TemplateType::ChatGLMChat, TemplateType::MptInstruct, TemplateType::RwkvWorld, TemplateType::OlmoInstruct, TemplateType::FillInMiddle, TemplateType::ZephyrChat, TemplateType::VicunaChat, TemplateType::OrcaChat, TemplateType::SolarInstruct, TemplateType::AlpacaInstruct, TemplateType::CommandRPlus, TemplateType::NousHermes, TemplateType::WizardLM, TemplateType::OpenChat, TemplateType::GraniteChat, TemplateType::NemotronChat, TemplateType::SaigaChat, TemplateType::Llama2Chat, TemplateType::Gemma2Chat, TemplateType::Phi3Instruct] {
            let tmpl = PromptTemplate::new(ttype);
            let formatted = tmpl.format(&user_text);
            prop_assert!(
                formatted.contains(&user_text),
                "Template {:?} dropped user text. formatted='{}'",
                ttype,
                formatted
            );
        }
    }

    /// History does not corrupt the current user message.
    #[test]
    fn history_does_not_corrupt_current_message(
        past_user in "[a-z]{1,50}",
        past_asst in "[a-z]{1,50}",
        current in "[A-Z]{1,50}"
    ) {
        let mut tmpl = PromptTemplate::new(TemplateType::Instruct);
        tmpl.add_turn(past_user, past_asst);
        let formatted = tmpl.format(&current);
        prop_assert!(
            formatted.contains(&current),
            "Current user text missing after adding history. formatted='{}'",
            formatted
        );
    }

    /// Greedy (temperature=0) is idempotent: same input → same formatted output.
    #[test]
    fn format_is_deterministic(
        user_text in "[a-zA-Z0-9 ]{1,100}",
        system in "[a-zA-Z ]{0,50}"
    ) {
        let tmpl1 = PromptTemplate::new(TemplateType::Instruct)
            .with_system_prompt(&system);
        let tmpl2 = PromptTemplate::new(TemplateType::Instruct)
            .with_system_prompt(&system);

        let out1 = tmpl1.format(&user_text);
        let out2 = tmpl2.format(&user_text);
        prop_assert_eq!(out1, out2);
    }

    /// After clear_history, the output is the same as a fresh template.
    #[test]
    fn clear_history_resets_state(
        user in "[a-z]{1,30}",
        asst in "[a-z]{1,30}",
        query in "[A-Z]{1,30}"
    ) {
        let fresh = PromptTemplate::new(TemplateType::Instruct).format(&query);
        let mut dirty = PromptTemplate::new(TemplateType::Instruct);
        dirty.add_turn(&user, &asst);
        dirty.clear_history();
        let cleared = dirty.format(&query);
        prop_assert_eq!(fresh, cleared);
    }
}

proptest! {
    /// System prompt text appears in the Instruct-formatted output.
    #[test]
    fn prop_system_prompt_appears_in_instruct_output(
        system in "[a-zA-Z0-9]{1,40}",
        user in "[a-zA-Z0-9 ]{1,60}",
    ) {
        let out = TemplateType::Instruct.apply(&user, Some(&system));
        prop_assert!(
            out.contains(&system),
            "system prompt {system:?} missing from instruct output: {out:?}"
        );
    }

    /// TemplateType::Raw.apply() with no system prompt is the identity transform.
    #[test]
    fn prop_raw_type_apply_is_identity(user in "[a-zA-Z0-9 .,?!]{1,80}") {
        let out = TemplateType::Raw.apply(&user, None);
        prop_assert_eq!(&out, &user, "Raw.apply() must return input unchanged");
    }

    /// TemplateType::Instruct output always ends with the answer marker "\nA:".
    #[test]
    fn prop_instruct_output_ends_with_answer_marker(
        user in "[a-zA-Z0-9 .,?!]{1,80}",
    ) {
        let out = TemplateType::Instruct.apply(&user, None);
        prop_assert!(
            out.ends_with("\nA:"),
            "Instruct output must end with '\\nA:', got: {out:?}"
        );
    }

    /// TemplateType::Display round-trips through FromStr.
    #[test]
    fn prop_template_type_display_roundtrip(
        template in prop_oneof![
            Just(TemplateType::Raw),
            Just(TemplateType::Instruct),
            Just(TemplateType::Llama3Chat),
            Just(TemplateType::Phi4Chat),
            Just(TemplateType::QwenChat),
            Just(TemplateType::GemmaChat),
            Just(TemplateType::MistralChat),
            Just(TemplateType::DeepSeekChat),
            Just(TemplateType::StarCoder),
            Just(TemplateType::FalconChat),
            Just(TemplateType::CodeLlamaInstruct),
            Just(TemplateType::CohereCommand),
            Just(TemplateType::InternLMChat),
            Just(TemplateType::YiChat),
            Just(TemplateType::BaichuanChat),
            Just(TemplateType::ChatGLMChat),
            Just(TemplateType::MptInstruct),
        Just(TemplateType::RwkvWorld),
        Just(TemplateType::OlmoInstruct),
        Just(TemplateType::FillInMiddle),
        Just(TemplateType::ZephyrChat),
        Just(TemplateType::VicunaChat),
        Just(TemplateType::OrcaChat),
        Just(TemplateType::SolarInstruct),
        Just(TemplateType::AlpacaInstruct),
        Just(TemplateType::CommandRPlus),
        Just(TemplateType::NousHermes),
        Just(TemplateType::WizardLM),
        Just(TemplateType::OpenChat),
        ],
    ) {
        let s = template.to_string();
        let parsed: TemplateType = s.parse().expect("display output must be parseable");
        prop_assert_eq!(template, parsed, "display→parse round-trip failed for {:?}", s);
    }

    /// Non-empty user input always produces non-empty output for every template.
    #[test]
    fn prop_nonempty_input_produces_nonempty_output(
        template in prop_oneof![
            Just(TemplateType::Raw),
            Just(TemplateType::Instruct),
            Just(TemplateType::Llama3Chat),
            Just(TemplateType::Phi4Chat),
            Just(TemplateType::QwenChat),
            Just(TemplateType::GemmaChat),
            Just(TemplateType::MistralChat),
            Just(TemplateType::DeepSeekChat),
            Just(TemplateType::StarCoder),
            Just(TemplateType::FalconChat),
            Just(TemplateType::CodeLlamaInstruct),
            Just(TemplateType::CohereCommand),
            Just(TemplateType::InternLMChat),
            Just(TemplateType::YiChat),
            Just(TemplateType::BaichuanChat),
            Just(TemplateType::ChatGLMChat),
            Just(TemplateType::MptInstruct),
        Just(TemplateType::RwkvWorld),
        Just(TemplateType::OlmoInstruct),
        Just(TemplateType::FillInMiddle),
        Just(TemplateType::ZephyrChat),
        Just(TemplateType::VicunaChat),
        Just(TemplateType::OrcaChat),
        Just(TemplateType::SolarInstruct),
        Just(TemplateType::AlpacaInstruct),
        Just(TemplateType::CommandRPlus),
        Just(TemplateType::NousHermes),
        Just(TemplateType::WizardLM),
        Just(TemplateType::OpenChat),
        ],
        user in "[a-zA-Z0-9]{1,50}",
    ) {
        let out = template.apply(&user, None);
        prop_assert!(!out.is_empty(), "template {template:?} produced empty output");
    }
}

#[test]
fn raw_template_is_identity() {
    let tmpl = PromptTemplate::new(TemplateType::Raw);
    let text = "Hello, world!";
    // Raw template should pass through with minimal wrapping
    let out = tmpl.format(text);
    assert!(out.contains(text));
}

#[test]
fn instruct_template_adds_formatting() {
    let raw = PromptTemplate::new(TemplateType::Raw).format("Q");
    let inst = PromptTemplate::new(TemplateType::Instruct).format("Q");
    // Instruct should differ from raw
    assert_ne!(raw, inst, "Instruct template should add formatting");
}

#[test]
fn snapshot_instruct_output() {
    let tmpl = PromptTemplate::new(TemplateType::Instruct)
        .with_system_prompt("You are a helpful assistant");
    let out = tmpl.format("What is 2+2?");
    insta::assert_snapshot!("instruct_with_system", out);
}

#[test]
fn snapshot_raw_output() {
    let tmpl = PromptTemplate::new(TemplateType::Raw);
    let out = tmpl.format("Tell me a story");
    insta::assert_snapshot!("raw_passthrough", out);
}

proptest! {
    /// TemplateType::detect() never panics with arbitrary optional string inputs
    /// and always returns a valid template variant.
    #[test]
    fn prop_detect_never_panics(
        name in proptest::option::of("[a-zA-Z0-9 _-]{0,50}"),
        jinja in proptest::option::of("[a-zA-Z0-9 _<>|{}%\n]{0,100}"),
    ) {
        let t = TemplateType::detect(name.as_deref(), jinja.as_deref());
        prop_assert!(
            matches!(t, TemplateType::Raw | TemplateType::Instruct | TemplateType::Llama3Chat | TemplateType::Phi4Chat | TemplateType::QwenChat | TemplateType::GemmaChat | TemplateType::MistralChat | TemplateType::DeepSeekChat | TemplateType::StarCoder | TemplateType::FalconChat | TemplateType::CodeLlamaInstruct | TemplateType::CohereCommand | TemplateType::InternLMChat | TemplateType::YiChat | TemplateType::BaichuanChat | TemplateType::ChatGLMChat | TemplateType::MptInstruct | TemplateType::RwkvWorld | TemplateType::OlmoInstruct | TemplateType::FillInMiddle | TemplateType::ZephyrChat | TemplateType::VicunaChat),
            "detect() returned an unexpected variant"
        );
    }

    /// LLaMA-3 chat template always includes <|start_header_id|> and <|end_header_id|>
    /// in its output regardless of user or system prompt content.
    #[test]
    fn prop_llama3_contains_header_tokens(
        user in "[a-zA-Z0-9 .,!?]{1,80}",
        system in proptest::option::of("[a-zA-Z0-9 ]{1,40}"),
    ) {
        let out = TemplateType::Llama3Chat.apply(&user, system.as_deref());
        prop_assert!(
            out.contains("<|start_header_id|>"),
            "LLaMA-3 output missing <|start_header_id|>: {out:?}"
        );
        prop_assert!(
            out.contains("<|end_header_id|>"),
            "LLaMA-3 output missing <|end_header_id|>: {out:?}"
        );
    }

    /// Empty user input does not cause a panic for any template type.
    #[test]
    fn prop_empty_input_does_not_panic(
        template in prop_oneof![
            Just(TemplateType::Raw),
            Just(TemplateType::Instruct),
            Just(TemplateType::Llama3Chat),
            Just(TemplateType::Phi4Chat),
            Just(TemplateType::QwenChat),
            Just(TemplateType::GemmaChat),
            Just(TemplateType::MistralChat),
            Just(TemplateType::DeepSeekChat),
            Just(TemplateType::StarCoder),
            Just(TemplateType::FalconChat),
            Just(TemplateType::CodeLlamaInstruct),
            Just(TemplateType::CohereCommand),
            Just(TemplateType::InternLMChat),
            Just(TemplateType::YiChat),
            Just(TemplateType::BaichuanChat),
            Just(TemplateType::ChatGLMChat),
            Just(TemplateType::MptInstruct),
        Just(TemplateType::RwkvWorld),
        Just(TemplateType::OlmoInstruct),
        Just(TemplateType::FillInMiddle),
        Just(TemplateType::ZephyrChat),
        Just(TemplateType::VicunaChat),
        ],
    ) {
        let out = template.apply("", None);
        // Verify the call completed without panicking and returned a String.
        let _ = out.len();
    }

    /// Unicode characters (Latin Extended-A) are preserved in template output.
    #[test]
    fn prop_unicode_in_prompts_preserved(
        template in prop_oneof![
            Just(TemplateType::Raw),
            Just(TemplateType::Instruct),
            Just(TemplateType::Llama3Chat),
            Just(TemplateType::Phi4Chat),
            Just(TemplateType::QwenChat),
            Just(TemplateType::GemmaChat),
            Just(TemplateType::MistralChat),
            Just(TemplateType::DeepSeekChat),
            Just(TemplateType::StarCoder),
            Just(TemplateType::FalconChat),
            Just(TemplateType::CodeLlamaInstruct),
            Just(TemplateType::CohereCommand),
            Just(TemplateType::InternLMChat),
            Just(TemplateType::YiChat),
            Just(TemplateType::BaichuanChat),
            Just(TemplateType::ChatGLMChat),
            Just(TemplateType::MptInstruct),
        Just(TemplateType::RwkvWorld),
        Just(TemplateType::OlmoInstruct),
        Just(TemplateType::FillInMiddle),
        Just(TemplateType::ZephyrChat),
        Just(TemplateType::VicunaChat),
        ],
        user in "[a-z\u{00E0}-\u{00FF}]{1,30}",
    ) {
        let out = template.apply(&user, None);
        prop_assert!(
            out.contains(user.as_str()),
            "Unicode input not preserved for {template:?}: user={user:?}"
        );
    }

    /// Very long prompts (≥1 000 characters) do not cause a panic.
    #[test]
    fn prop_very_long_prompt_no_panic(
        template in prop_oneof![
            Just(TemplateType::Raw),
            Just(TemplateType::Instruct),
            Just(TemplateType::Llama3Chat),
            Just(TemplateType::Phi4Chat),
            Just(TemplateType::QwenChat),
            Just(TemplateType::GemmaChat),
            Just(TemplateType::MistralChat),
            Just(TemplateType::DeepSeekChat),
            Just(TemplateType::StarCoder),
            Just(TemplateType::FalconChat),
            Just(TemplateType::CodeLlamaInstruct),
            Just(TemplateType::CohereCommand),
            Just(TemplateType::InternLMChat),
            Just(TemplateType::YiChat),
            Just(TemplateType::BaichuanChat),
            Just(TemplateType::ChatGLMChat),
            Just(TemplateType::MptInstruct),
        Just(TemplateType::RwkvWorld),
        Just(TemplateType::OlmoInstruct),
        Just(TemplateType::FillInMiddle),
        Just(TemplateType::ZephyrChat),
        Just(TemplateType::VicunaChat),
        ],
        base in "[a-z]{5,10}",
        repeats in 200usize..=250usize,
    ) {
        let long = base.repeat(repeats); // 1 000 – 2 500 characters
        let out = template.apply(&long, None);
        prop_assert!(
            !out.is_empty(),
            "template {template:?} produced empty output for long input ({} chars)",
            long.len()
        );
    }

    /// Instruct template wraps user text with "Q: " prefix and "A:" suffix.
    #[test]
    fn prop_instruct_wraps_with_qa_markers(
        user in "[a-zA-Z0-9 .,?!]{1,80}",
    ) {
        let out = TemplateType::Instruct.apply(&user, None);
        prop_assert!(out.contains("Q: "), "Instruct output missing 'Q: ' marker");
        prop_assert!(out.contains("A:"), "Instruct output missing 'A:' marker");
    }

    /// In Instruct format the "System: " section appears before the "Q: " section.
    #[test]
    fn prop_system_header_before_user_in_instruct(
        system in "[a-zA-Z0-9]{3,20}",
        user in "[a-zA-Z0-9 ]{3,20}",
    ) {
        let out = TemplateType::Instruct.apply(&user, Some(&system));
        let sys_pos = out.find("System: ").expect("'System: ' marker must be present");
        let user_pos = out.find("Q: ").expect("'Q: ' marker must be present");
        prop_assert!(
            sys_pos < user_pos,
            "system section must precede user section (sys={sys_pos}, user={user_pos})"
        );
    }

    /// In LLaMA-3 format the system role header appears before the user role header.
    #[test]
    fn prop_system_header_before_user_in_llama3(
        system in "[a-zA-Z0-9]{3,20}",
        user in "[a-zA-Z0-9 ]{3,20}",
    ) {
        let out = TemplateType::Llama3Chat.apply(&user, Some(&system));
        let sys_pos = out
            .find("<|start_header_id|>system<|end_header_id|>")
            .expect("system role header must be present");
        let user_pos = out
            .find("<|start_header_id|>user<|end_header_id|>")
            .expect("user role header must be present");
        prop_assert!(
            sys_pos < user_pos,
            "system role header must precede user role header (sys={sys_pos}, user={user_pos})"
        );
    }

    /// render_chat with identical inputs always produces identical output (deterministic).
    #[test]
    fn prop_render_chat_is_deterministic(
        template in prop_oneof![
            Just(TemplateType::Raw),
            Just(TemplateType::Instruct),
            Just(TemplateType::Llama3Chat),
            Just(TemplateType::Phi4Chat),
            Just(TemplateType::QwenChat),
            Just(TemplateType::GemmaChat),
            Just(TemplateType::MistralChat),
            Just(TemplateType::DeepSeekChat),
            Just(TemplateType::StarCoder),
            Just(TemplateType::FalconChat),
            Just(TemplateType::CodeLlamaInstruct),
            Just(TemplateType::CohereCommand),
            Just(TemplateType::InternLMChat),
            Just(TemplateType::YiChat),
            Just(TemplateType::BaichuanChat),
            Just(TemplateType::ChatGLMChat),
            Just(TemplateType::MptInstruct),
        Just(TemplateType::RwkvWorld),
        Just(TemplateType::OlmoInstruct),
        Just(TemplateType::FillInMiddle),
        Just(TemplateType::ZephyrChat),
        Just(TemplateType::VicunaChat),
        ],
        user_text in "[a-zA-Z0-9 ]{1,60}",
    ) {
        let turns = vec![ChatTurn::new(ChatRole::User, user_text.as_str())];
        let out1 = template.render_chat(&turns, None).unwrap();
        let out2 = template.render_chat(&turns, None).unwrap();
        prop_assert_eq!(out1, out2, "render_chat must be deterministic for {:?}", template);
    }

    /// Phi4Chat template always includes <|im_start|> and <|im_end|> in its output.
    #[test]
    fn prop_phi4_contains_chatml_tokens(
        user in "[a-zA-Z0-9 .,!?]{1,80}",
        system in proptest::option::of("[a-zA-Z0-9 ]{1,40}"),
    ) {
        let out = TemplateType::Phi4Chat.apply(&user, system.as_deref());
        prop_assert!(
            out.contains("<|im_start|>"),
            "Phi4Chat output missing <|im_start|>: {out:?}"
        );
        prop_assert!(
            out.contains("<|im_end|>"),
            "Phi4Chat output missing <|im_end|>: {out:?}"
        );
    }

    /// Phi4Chat display/parse round-trips correctly.
    #[test]
    fn prop_phi4_display_roundtrip(
        _dummy in Just(()),
    ) {
        let t = TemplateType::Phi4Chat;
        let s = t.to_string();
        let parsed: TemplateType = s.parse().expect("phi4-chat must parse");
        prop_assert_eq!(t, parsed);
    }

    /// QwenChat template always includes <|im_start|> and <|im_end|> in its output.
    #[test]
    fn prop_qwen_contains_chatml_tokens(
        user in "[a-zA-Z0-9 .,!?]{1,80}",
        system in proptest::option::of("[a-zA-Z0-9 ]{1,40}"),
    ) {
        let out = TemplateType::QwenChat.apply(&user, system.as_deref());
        prop_assert!(
            out.contains("<|im_start|>"),
            "QwenChat output missing <|im_start|>: {out:?}"
        );
        prop_assert!(
            out.contains("<|im_end|>"),
            "QwenChat output missing <|im_end|>: {out:?}"
        );
    }

    /// QwenChat display/parse round-trips correctly.
    #[test]
    fn prop_qwen_display_roundtrip(
        _dummy in Just(()),
    ) {
        let t = TemplateType::QwenChat;
        let s = t.to_string();
        let parsed: TemplateType = s.parse().expect("qwen-chat must parse");
        prop_assert_eq!(t, parsed);
    }

    /// GemmaChat template always includes <start_of_turn> and <end_of_turn>.
    #[test]
    fn prop_gemma_contains_turn_tokens(
        user in "[a-zA-Z0-9 .,!?]{1,80}",
        system in proptest::option::of("[a-zA-Z0-9 ]{1,40}"),
    ) {
        let out = TemplateType::GemmaChat.apply(&user, system.as_deref());
        prop_assert!(
            out.contains("<start_of_turn>"),
            "GemmaChat output missing <start_of_turn>: {out:?}"
        );
        prop_assert!(
            out.contains("<end_of_turn>"),
            "GemmaChat output missing <end_of_turn>: {out:?}"
        );
    }

    /// GemmaChat display/parse round-trips correctly.
    #[test]
    fn prop_gemma_display_roundtrip(
        _dummy in Just(()),
    ) {
        let t = TemplateType::GemmaChat;
        let s = t.to_string();
        let parsed: TemplateType = s.parse().expect("gemma-chat must parse");
        prop_assert_eq!(t, parsed);
    }

    /// MistralChat template always includes [INST] and [/INST] in its output.
    #[test]
    fn prop_mistral_contains_inst_tokens(
        user in "[a-zA-Z0-9 .,!?]{1,80}",
        system in proptest::option::of("[a-zA-Z0-9 ]{1,40}"),
    ) {
        let out = TemplateType::MistralChat.apply(&user, system.as_deref());
        prop_assert!(
            out.contains("[INST]"),
            "MistralChat output missing [INST]: {out:?}"
        );
        prop_assert!(
            out.contains("[/INST]"),
            "MistralChat output missing [/INST]: {out:?}"
        );
    }

    /// MistralChat display/parse round-trips correctly.
    #[test]
    fn prop_mistral_display_roundtrip(
        _dummy in Just(()),
    ) {
        let t = TemplateType::MistralChat;
        let s = t.to_string();
        let parsed: TemplateType = s.parse().expect("mistral-chat must parse");
        prop_assert_eq!(t, parsed);
    }

    /// DeepSeekChat template always includes ChatML tokens.
    #[test]
    fn prop_deepseek_contains_chatml_tokens(
        user in "[a-zA-Z0-9 ]{1,100}",
        system in proptest::option::of("[a-zA-Z0-9 ]{1,50}"),
    ) {
        let out = TemplateType::DeepSeekChat.apply(&user, system.as_deref());
        prop_assert!(
            out.contains("<|im_start|>"),
            "DeepSeekChat output missing <|im_start|>: {out:?}"
        );
        prop_assert!(
            out.contains("<|im_end|>"),
            "DeepSeekChat output missing <|im_end|>: {out:?}"
        );
    }

    /// DeepSeekChat display/parse round-trips correctly.
    #[test]
    fn prop_deepseek_display_roundtrip(
        _dummy in Just(()),
    ) {
        let t = TemplateType::DeepSeekChat;
        let s = t.to_string();
        let parsed: TemplateType = s.parse().expect("deepseek-chat must parse");
        prop_assert_eq!(t, parsed);
    }

    /// StarCoder output preserves user text verbatim.
    #[test]
    fn prop_starcoder_preserves_code(
        code in "[a-zA-Z0-9_ (){}:;]{1,100}",
    ) {
        let out = TemplateType::StarCoder.apply(&code, None);
        prop_assert_eq!(out, code);
    }

    /// StarCoder display/parse round-trips correctly.
    #[test]
    fn prop_starcoder_display_roundtrip(
        _dummy in Just(()),
    ) {
        let t = TemplateType::StarCoder;
        let s = t.to_string();
        let parsed: TemplateType = s.parse().expect("starcoder must parse");
        prop_assert_eq!(t, parsed);
    }

    /// FalconChat output contains "User:" and ends with "Falcon:".
    #[test]
    fn prop_falcon_contains_role_markers(
        user in "[a-zA-Z0-9 .,!?]{1,80}",
    ) {
        let out = TemplateType::FalconChat.apply(&user, None);
        prop_assert!(
            out.contains("User:"),
            "FalconChat output missing 'User:': {out:?}"
        );
        prop_assert!(
            out.contains("Falcon:"),
            "FalconChat output missing 'Falcon:': {out:?}"
        );
    }

    /// FalconChat display/parse round-trips correctly.
    #[test]
    fn prop_falcon_display_roundtrip(
        _dummy in Just(()),
    ) {
        let t = TemplateType::FalconChat;
        let s = t.to_string();
        let parsed: TemplateType = s.parse().expect("falcon-chat must parse");
        prop_assert_eq!(t, parsed);
    }

    /// CodeLlamaInstruct output contains [INST] and [/INST].
    #[test]
    fn prop_codellama_contains_inst_tokens(
        user in "[a-zA-Z0-9 .,!?]{1,80}",
    ) {
        let out = TemplateType::CodeLlamaInstruct.apply(&user, None);
        prop_assert!(
            out.contains("[INST]"),
            "CodeLlamaInstruct output missing [INST]: {out:?}"
        );
        prop_assert!(
            out.contains("[/INST]"),
            "CodeLlamaInstruct output missing [/INST]: {out:?}"
        );
    }

    /// CodeLlamaInstruct with system prompt contains <<SYS>>.
    #[test]
    fn prop_codellama_sys_contains_markers(
        user in "[a-zA-Z0-9 ]{1,40}",
        system in "[a-zA-Z0-9 ]{1,40}",
    ) {
        let out = TemplateType::CodeLlamaInstruct.apply(&user, Some(&system));
        prop_assert!(
            out.contains("<<SYS>>"),
            "CodeLlamaInstruct with system missing <<SYS>>: {out:?}"
        );
        prop_assert!(
            out.contains("<</SYS>>"),
            "CodeLlamaInstruct with system missing <</SYS>>: {out:?}"
        );
    }

    /// CodeLlamaInstruct display/parse round-trips correctly.
    #[test]
    fn prop_codellama_display_roundtrip(
        _dummy in Just(()),
    ) {
        let t = TemplateType::CodeLlamaInstruct;
        let s = t.to_string();
        let parsed: TemplateType = s.parse().expect("codellama-instruct must parse");
        prop_assert_eq!(t, parsed);
    }

    /// CohereCommand output contains turn tokens.
    #[test]
    fn prop_cohere_contains_turn_tokens(
        user in "[a-zA-Z0-9 .,!?]{1,80}",
    ) {
        let out = TemplateType::CohereCommand.apply(&user, None);
        prop_assert!(
            out.contains("<|START_OF_TURN_TOKEN|>"),
            "CohereCommand output missing turn token: {out:?}"
        );
        prop_assert!(
            out.contains("<|END_OF_TURN_TOKEN|>"),
            "CohereCommand output missing end turn token: {out:?}"
        );
    }

    /// CohereCommand display/parse round-trips correctly.
    #[test]
    fn prop_cohere_display_roundtrip(
        _dummy in Just(()),
    ) {
        let t = TemplateType::CohereCommand;
        let s = t.to_string();
        let parsed: TemplateType = s.parse().expect("cohere-command must parse");
        prop_assert_eq!(t, parsed);
    }

    /// InternLMChat template always includes ChatML tokens.
    #[test]
    fn prop_internlm_contains_chatml_tokens(
        user in "[a-zA-Z0-9 ]{1,100}",
        system in proptest::option::of("[a-zA-Z0-9 ]{1,50}"),
    ) {
        let out = TemplateType::InternLMChat.apply(&user, system.as_deref());
        prop_assert!(
            out.contains("<|im_start|>"),
            "InternLMChat output missing <|im_start|>: {out:?}"
        );
        prop_assert!(
            out.contains("<|im_end|>"),
            "InternLMChat output missing <|im_end|>: {out:?}"
        );
    }

    /// InternLMChat display/parse round-trips correctly.
    #[test]
    fn prop_internlm_display_roundtrip(
        _dummy in Just(()),
    ) {
        let t = TemplateType::InternLMChat;
        let s = t.to_string();
        let parsed: TemplateType = s.parse().expect("internlm-chat must parse");
        prop_assert_eq!(t, parsed);
    }

    // ── Yi Chat ────────────────────────────────────────────────────────

    /// YiChat template always includes ChatML tokens.
    #[test]
    fn prop_yi_contains_chatml_tokens(
        user in "[a-zA-Z0-9 ]{1,100}",
        system in proptest::option::of("[a-zA-Z0-9 ]{1,50}"),
    ) {
        let out = TemplateType::YiChat.apply(&user, system.as_deref());
        prop_assert!(out.contains("<|im_start|>"), "YiChat missing <|im_start|>: {out:?}");
        prop_assert!(out.contains("<|im_end|>"), "YiChat missing <|im_end|>: {out:?}");
    }

    /// YiChat display/parse round-trips correctly.
    #[test]
    fn prop_yi_display_roundtrip(_dummy in Just(())) {
        let t = TemplateType::YiChat;
        let s = t.to_string();
        let parsed: TemplateType = s.parse().expect("yi-chat must parse");
        prop_assert_eq!(t, parsed);
    }

    // ── Baichuan Chat ──────────────────────────────────────────────────

    /// BaichuanChat template always includes reserved tokens.
    #[test]
    fn prop_baichuan_contains_reserved_tokens(
        user in "[a-zA-Z0-9 ]{1,100}",
    ) {
        let out = TemplateType::BaichuanChat.apply(&user, None);
        prop_assert!(out.contains("<reserved_106>"), "BaichuanChat missing <reserved_106>: {out:?}");
        prop_assert!(out.contains("<reserved_107>"), "BaichuanChat missing <reserved_107>: {out:?}");
    }

    /// BaichuanChat display/parse round-trips correctly.
    #[test]
    fn prop_baichuan_display_roundtrip(_dummy in Just(())) {
        let t = TemplateType::BaichuanChat;
        let s = t.to_string();
        let parsed: TemplateType = s.parse().expect("baichuan-chat must parse");
        prop_assert_eq!(t, parsed);
    }

    // ── ChatGLM Chat ───────────────────────────────────────────────────

    /// ChatGLMChat template always starts with [gMASK]<sop>.
    #[test]
    fn prop_chatglm_starts_with_gmask(
        user in "[a-zA-Z0-9 ]{1,100}",
    ) {
        let out = TemplateType::ChatGLMChat.apply(&user, None);
        prop_assert!(out.starts_with("[gMASK]<sop>"), "ChatGLMChat missing [gMASK]<sop>: {out:?}");
        prop_assert!(out.contains("<|user|>"), "ChatGLMChat missing <|user|>: {out:?}");
        prop_assert!(out.contains("<|assistant|>"), "ChatGLMChat missing <|assistant|>: {out:?}");
    }

    /// ChatGLMChat display/parse round-trips correctly.
    #[test]
    fn prop_chatglm_display_roundtrip(_dummy in Just(())) {
        let t = TemplateType::ChatGLMChat;
        let s = t.to_string();
        let parsed: TemplateType = s.parse().expect("chatglm-chat must parse");
        prop_assert_eq!(t, parsed);
    }

    // ── MPT Instruct ───────────────────────────────────────────────────

    /// MptInstruct template always includes ### markers.
    #[test]
    fn prop_mpt_contains_hash_markers(
        user in "[a-zA-Z0-9 ]{1,100}",
    ) {
        let out = TemplateType::MptInstruct.apply(&user, None);
        prop_assert!(out.contains("### Instruction"), "MptInstruct missing ### Instruction: {out:?}");
        prop_assert!(out.contains("### Response"), "MptInstruct missing ### Response: {out:?}");
    }

    /// MptInstruct display/parse round-trips correctly.
    #[test]
    fn prop_mpt_display_roundtrip(_dummy in Just(())) {
        let t = TemplateType::MptInstruct;
        let s = t.to_string();
        let parsed: TemplateType = s.parse().expect("mpt-instruct must parse");
        prop_assert_eq!(t, parsed);
    }
}
