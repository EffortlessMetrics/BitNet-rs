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

use bitnet_prompt_templates::{PromptTemplate, TemplateType};
use proptest::prelude::*;

proptest! {
    /// The formatted output always contains the original user text.
    #[test]
    fn user_text_preserved(user_text in "[a-zA-Z0-9 .,!?]{1,200}") {
        for ttype in [TemplateType::Raw, TemplateType::Instruct, TemplateType::Llama3Chat] {
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
