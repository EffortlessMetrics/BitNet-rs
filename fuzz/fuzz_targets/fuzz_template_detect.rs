#![no_main]

use arbitrary::Arbitrary;
use bitnet_prompt_templates::TemplateType;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct DetectInput {
    /// Raw bytes for tokenizer name.
    tokenizer_name_raw: Vec<u8>,
    /// Raw bytes for chat template jinja.
    chat_template_raw: Vec<u8>,
    /// Raw bytes for architecture name.
    arch_raw: Vec<u8>,
    /// Raw bytes for user text (for validate_output).
    user_text_raw: Vec<u8>,
}

fuzz_target!(|input: DetectInput| {
    let tokenizer_name = std::str::from_utf8(&input.tokenizer_name_raw).ok();
    let chat_template = std::str::from_utf8(&input.chat_template_raw).ok();
    let arch = std::str::from_utf8(&input.arch_raw).unwrap_or("");
    let user_text = std::str::from_utf8(&input.user_text_raw).unwrap_or("");

    // detect() must never panic on any string combination
    let detected = TemplateType::detect(tokenizer_name, chat_template);

    // The detected template must be usable
    let output = detected.apply(user_text, None);
    // Output is always a valid String (never panics)
    let _ = output.len();

    // validate_output must never panic
    let validation = detected.validate_output(&output, user_text);
    let _ = validation.is_valid;

    // suggest_for_arch must never panic on arbitrary strings
    let _ = TemplateType::suggest_for_arch(arch);

    // info() must never panic
    let info = detected.info();
    let _ = info.name;

    // default_stop_sequences must never panic
    let stops = detected.default_stop_sequences();
    let _ = stops.len();

    // should_add_bos must never panic
    let _ = detected.should_add_bos();

    // all_variants must return a non-empty list
    let variants = TemplateType::all_variants();
    assert!(!variants.is_empty());

    // Each variant's apply/info must not panic
    for &variant in variants.iter().take(8) {
        let _ = variant.apply(user_text, None);
        let _ = variant.info();
    }
});
