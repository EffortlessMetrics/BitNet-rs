//! Property-based tests for bitnet-prompt-templates.
//!
//! Verifies key invariants:
//! - `format()` always contains the user text
//! - Adding history does not erase the current user message
//! - Template types produce structurally different output
//! - Round-trip: clear_history restores empty state

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
