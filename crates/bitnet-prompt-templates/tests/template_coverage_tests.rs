//! Template-Architecture coverage tests.
//!
//! Validates that the prompt template system correctly maps to architecture
//! families and that template apply/render operations produce valid output.

use bitnet_prompt_templates::{ChatRole, ChatTurn, PromptTemplate, TemplateType};

// ============================================================
// Architecture → Template mapping coverage
// ============================================================

#[test]
fn test_non_chat_architectures_return_none() {
    for arch in &["gpt", "bert", "bitnet", "bitnet-b1.58", "unknown-model"] {
        assert!(
            TemplateType::suggest_for_arch(arch).is_none(),
            "non-chat arch '{}' should not have a template suggestion",
            arch
        );
    }
}

#[test]
fn test_template_suggestion_is_case_insensitive() {
    let test_cases = [
        ("PHI-4", "phi-4"),
        ("Llama", "llama"),
        ("QWEN", "qwen"),
        ("Gemma", "gemma"),
        ("MISTRAL", "mistral"),
    ];

    for (upper, lower) in &test_cases {
        assert_eq!(
            TemplateType::suggest_for_arch(upper),
            TemplateType::suggest_for_arch(lower),
            "case mismatch for '{}'/'{}' ",
            upper,
            lower
        );
    }
}

// ============================================================
// Template apply produces non-empty output
// ============================================================

#[test]
fn test_all_templates_produce_nonempty_apply() {
    let templates = [
        TemplateType::Raw,
        TemplateType::Instruct,
        TemplateType::Llama3Chat,
        TemplateType::Phi4Chat,
        TemplateType::QwenChat,
        TemplateType::GemmaChat,
        TemplateType::MistralChat,
        TemplateType::DeepSeekChat,
        TemplateType::FalconChat,
        TemplateType::CohereCommand,
        TemplateType::InternLMChat,
        TemplateType::YiChat,
        TemplateType::BaichuanChat,
        TemplateType::ChatGLMChat,
        TemplateType::MptInstruct,
        TemplateType::RwkvWorld,
        TemplateType::OlmoInstruct,
        TemplateType::ZephyrChat,
        TemplateType::VicunaChat,
        TemplateType::OrcaChat,
        TemplateType::SolarInstruct,
        TemplateType::AlpacaInstruct,
        TemplateType::CommandRPlus,
        TemplateType::NousHermes,
        TemplateType::WizardLM,
        TemplateType::OpenChat,
        TemplateType::GraniteChat,
        TemplateType::NemotronChat,
        TemplateType::SaigaChat,
        TemplateType::Llama2Chat,
        TemplateType::Gemma2Chat,
        TemplateType::Phi3Instruct,
        TemplateType::TinyLlamaChat,
        TemplateType::DolphinChat,
        TemplateType::ChatGptChat,
        TemplateType::MixtralInstruct,
        TemplateType::StableLMChat,
        TemplateType::BloomChat,
        TemplateType::JambaChat,
        TemplateType::PersimmonChat,
        TemplateType::XverseChat,
        TemplateType::Qwen25Chat,
        TemplateType::MistralNemoChat,
        TemplateType::ArcticInstruct,
        TemplateType::DbrxInstruct,
        TemplateType::ExaoneChat,
        TemplateType::MiniCPMChat,
        TemplateType::CodeGemma,
        TemplateType::Llama31Chat,
        TemplateType::DeepSeekV3Chat,
        TemplateType::Falcon2Chat,
        TemplateType::OLMo2Chat,
        TemplateType::Llama32Chat,
        TemplateType::CohereAya,
        TemplateType::SmolLMChat,
        TemplateType::Phi2Instruct,
    ];

    for template in &templates {
        let output = template.apply("Hello, world!", None);
        assert!(!output.is_empty(), "template {:?} produced empty output for apply()", template);
        assert!(
            output.contains("Hello, world!"),
            "template {:?} output does not contain user prompt",
            template
        );
    }
}

// ============================================================
// Template render produces valid multi-turn output
// ============================================================

#[test]
fn test_all_templates_produce_nonempty_render_chat() {
    let history = vec![
        ChatTurn::new(ChatRole::User, "What is Rust?"),
        ChatTurn::new(ChatRole::Assistant, "Rust is a systems programming language."),
        ChatTurn::new(ChatRole::User, "Tell me more."),
    ];

    let templates_with_render = [
        TemplateType::Llama3Chat,
        TemplateType::Phi4Chat,
        TemplateType::QwenChat,
        TemplateType::GemmaChat,
        TemplateType::MistralChat,
        TemplateType::DeepSeekChat,
        TemplateType::FalconChat,
        TemplateType::CohereCommand,
        TemplateType::InternLMChat,
        TemplateType::YiChat,
        TemplateType::BaichuanChat,
        TemplateType::ChatGLMChat,
        TemplateType::MptInstruct,
        TemplateType::OlmoInstruct,
        TemplateType::ZephyrChat,
        TemplateType::VicunaChat,
    ];

    for template in &templates_with_render {
        let result = template.render_chat(&history, None);
        assert!(result.is_ok(), "template {:?} render_chat failed: {:?}", template, result.err());
        let output = result.unwrap();
        assert!(
            !output.is_empty(),
            "template {:?} produced empty output for render_chat()",
            template
        );
    }
}

// ============================================================
// Template stop sequences
// ============================================================

#[test]
fn test_templates_have_stop_sequences() {
    // ChatML variants should have stop sequences
    let chatml_templates = [
        TemplateType::Phi4Chat,
        TemplateType::QwenChat,
        TemplateType::DeepSeekChat,
        TemplateType::InternLMChat,
        TemplateType::YiChat,
    ];

    for template in &chatml_templates {
        let stops = template.default_stop_sequences();
        assert!(!stops.is_empty(), "ChatML template {:?} should have stop sequences", template);
    }
}

// ============================================================
// Phi-4 specific template tests
// ============================================================

#[test]
fn test_phi4_template_uses_chatml_format() {
    let output = TemplateType::Phi4Chat.apply("What is 2+2?", None);
    assert!(output.contains("<|im_start|>"), "Phi-4 template should use <|im_start|>");
    assert!(output.contains("<|im_end|>"), "Phi-4 template should use <|im_end|>");
    assert!(output.contains("system"), "Phi-4 template should include system role");
    assert!(output.contains("What is 2+2?"), "Phi-4 template should include user prompt");
}

#[test]
fn test_phi4_template_with_custom_system() {
    let output = TemplateType::Phi4Chat.apply("Hello", Some("You are a math tutor."));
    assert!(output.contains("You are a math tutor."));
    assert!(output.contains("Hello"));
}

#[test]
fn test_phi4_multi_turn_render() {
    let history = vec![
        ChatTurn::new(ChatRole::User, "Hi"),
        ChatTurn::new(ChatRole::Assistant, "Hello!"),
        ChatTurn::new(ChatRole::User, "How are you?"),
    ];
    let output = TemplateType::Phi4Chat.render_chat(&history, None).unwrap();
    assert!(output.contains("Hi"));
    assert!(output.contains("Hello!"));
    assert!(output.contains("How are you?"));
}

// ============================================================
// Architecture family coverage matrix
// ============================================================

#[test]
fn test_major_model_families_have_templates() {
    let families = [
        ("phi-4", TemplateType::Phi4Chat),
        ("phi-3", TemplateType::Phi3Instruct),
        ("phi-2", TemplateType::Phi2Instruct),
        ("llama", TemplateType::Llama3Chat),
        ("llama-3.1", TemplateType::Llama31Chat),
        ("llama-3.2", TemplateType::Llama32Chat),
        ("llama2", TemplateType::Llama2Chat),
        ("mistral", TemplateType::MistralChat),
        ("mistral-nemo", TemplateType::MistralNemoChat),
        ("mixtral", TemplateType::MixtralInstruct),
        ("qwen", TemplateType::QwenChat),
        ("qwen2.5", TemplateType::Qwen25Chat),
        ("gemma", TemplateType::GemmaChat),
        ("gemma2", TemplateType::Gemma2Chat),
        ("deepseek", TemplateType::DeepSeekChat),
        ("deepseek-v3", TemplateType::DeepSeekV3Chat),
        ("starcoder", TemplateType::StarCoder),
        ("falcon", TemplateType::FalconChat),
        ("falcon-2", TemplateType::Falcon2Chat),
        ("command", TemplateType::CohereCommand),
        ("command-r-plus", TemplateType::CommandRPlus),
        ("aya", TemplateType::CohereAya),
        ("internlm", TemplateType::InternLMChat),
        ("yi", TemplateType::YiChat),
        ("baichuan", TemplateType::BaichuanChat),
        ("chatglm", TemplateType::ChatGLMChat),
        ("mpt", TemplateType::MptInstruct),
        ("rwkv", TemplateType::RwkvWorld),
        ("olmo", TemplateType::OlmoInstruct),
        ("olmo2", TemplateType::OLMo2Chat),
        ("zephyr", TemplateType::ZephyrChat),
        ("vicuna", TemplateType::VicunaChat),
        ("orca", TemplateType::OrcaChat),
        ("solar", TemplateType::SolarInstruct),
        ("alpaca", TemplateType::AlpacaInstruct),
        ("nous-hermes", TemplateType::NousHermes),
        ("wizardlm", TemplateType::WizardLM),
        ("openchat", TemplateType::OpenChat),
        ("granite", TemplateType::GraniteChat),
        ("nemotron", TemplateType::NemotronChat),
        ("saiga", TemplateType::SaigaChat),
        ("tinyllama", TemplateType::TinyLlamaChat),
        ("dolphin", TemplateType::DolphinChat),
        ("chatgpt", TemplateType::ChatGptChat),
        ("stablelm", TemplateType::StableLMChat),
        ("bloom", TemplateType::BloomChat),
        ("jamba", TemplateType::JambaChat),
        ("persimmon", TemplateType::PersimmonChat),
        ("xverse", TemplateType::XverseChat),
        ("arctic", TemplateType::ArcticInstruct),
        ("dbrx", TemplateType::DbrxInstruct),
        ("exaone", TemplateType::ExaoneChat),
        ("minicpm", TemplateType::MiniCPMChat),
        ("smollm", TemplateType::SmolLMChat),
        ("codellama", TemplateType::CodeLlamaInstruct),
        ("codegemma", TemplateType::CodeGemma),
    ];

    for (arch, expected) in &families {
        assert_eq!(
            TemplateType::suggest_for_arch(arch),
            Some(*expected),
            "arch '{}' should map to {:?}",
            arch,
            expected
        );
    }
}

// ============================================================
// PromptTemplate wrapper tests
// ============================================================

#[test]
fn test_prompt_template_format() {
    let pt = PromptTemplate::new(TemplateType::Phi4Chat);
    let output = pt.format("What is 2+2?");
    assert!(output.contains("What is 2+2?"));
    assert!(output.contains("<|im_start|>"));
}

#[test]
fn test_prompt_template_with_system() {
    let pt =
        PromptTemplate::new(TemplateType::Phi4Chat).with_system_prompt("You are a math tutor.");
    let output = pt.format("What is 2+2?");
    assert!(output.contains("You are a math tutor."));
}

#[test]
fn test_prompt_template_stop_sequences() {
    let pt = PromptTemplate::new(TemplateType::Phi4Chat);
    let stops = pt.stop_sequences();
    assert!(!stops.is_empty());
}

// ============================================================
// FillInMiddle and StarCoder code templates
// ============================================================

#[test]
fn test_fill_in_middle_template() {
    let output = TemplateType::FillInMiddle.apply("def hello():", None);
    assert!(!output.is_empty());
}

#[test]
fn test_starcoder_template() {
    let output = TemplateType::StarCoder.apply("# Write a Python function", None);
    assert!(!output.is_empty());
    assert!(output.contains("Write a Python function"));
}
