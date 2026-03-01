//! Template-to-architecture mapping and API consistency tests.
//!
//! Validates that:
//! - Every architecture with a template gets the right one from suggest_for_arch
//! - All templates roundtrip through Display/FromStr
//! - All templates produce non-empty output from apply()
//! - All templates have stop sequences
//! - suggest_for_arch covers all ArchitectureRegistry entries

use bitnet_common::ArchitectureRegistry;
use bitnet_prompt_templates::TemplateType;

// ────────────────────────────────────────────────────────────────
// 1. Display/FromStr roundtrip for every variant
// ────────────────────────────────────────────────────────────────

#[test]
fn all_variants_roundtrip_display_fromstr() {
    for variant in TemplateType::all_variants() {
        let display = variant.to_string();
        let parsed: TemplateType = display.parse().unwrap_or_else(|e| {
            panic!("Failed to parse '{}' back to TemplateType: {}", display, e)
        });
        assert_eq!(
            *variant, parsed,
            "Roundtrip failed for {}: displayed as '{}', parsed as {:?}",
            variant, display, parsed,
        );
    }
}

// ────────────────────────────────────────────────────────────────
// 2. All variants produce non-empty output from apply()
// ────────────────────────────────────────────────────────────────

#[test]
fn all_variants_apply_produces_output() {
    let user_text = "Hello, world!";
    for variant in TemplateType::all_variants() {
        let output = variant.apply(user_text, None);
        assert!(!output.is_empty(), "apply() for {:?} produced empty output", variant,);
        // Output should contain the user text somewhere
        assert!(output.contains(user_text), "apply() for {:?} doesn't contain user text", variant,);
    }
}

#[test]
fn all_variants_apply_with_system_prompt() {
    let user_text = "What is 2+2?";
    let system = "You are a helpful assistant.";
    for variant in TemplateType::all_variants() {
        let output = variant.apply(user_text, Some(system));
        assert!(!output.is_empty(), "apply() with system for {:?} produced empty output", variant,);
    }
}

// ────────────────────────────────────────────────────────────────
// 3. All variants have stop sequences (except Raw and FIM)
// ────────────────────────────────────────────────────────────────

#[test]
fn chat_templates_have_stop_sequences() {
    let exceptions = [TemplateType::Raw, TemplateType::FillInMiddle];
    for variant in TemplateType::all_variants() {
        if exceptions.contains(variant) {
            continue;
        }
        let stops = variant.default_stop_sequences();
        assert!(!stops.is_empty(), "{:?} should have stop sequences", variant,);
    }
}

// ────────────────────────────────────────────────────────────────
// 4. suggest_for_arch covers registry architectures
// ────────────────────────────────────────────────────────────────

#[test]
fn suggest_for_arch_covers_most_architectures() {
    let archs = ArchitectureRegistry::known_architectures();
    let mut covered = 0;
    let mut uncovered: Vec<&str> = Vec::new();

    for arch in archs {
        if TemplateType::suggest_for_arch(arch).is_some() {
            covered += 1;
        } else {
            uncovered.push(arch);
        }
    }

    // Most architectures should be covered (only gpt, bert, bitnet are expected to be None)
    let expected_uncovered = ["gpt", "bert", "bitnet", "bitnet-b1.58"];
    for arch in &uncovered {
        assert!(
            expected_uncovered.contains(arch),
            "Unexpected uncovered architecture: '{}'. Should it have a template?",
            arch,
        );
    }

    assert!(
        covered > archs.len() / 2,
        "Expected most architectures to have templates, but only {}/{} are covered",
        covered,
        archs.len(),
    );
}

// ────────────────────────────────────────────────────────────────
// 5. Specific arch-to-template mappings
// ────────────────────────────────────────────────────────────────

#[test]
fn phi_family_templates() {
    assert_eq!(TemplateType::suggest_for_arch("phi-4"), Some(TemplateType::Phi4Chat));
    assert_eq!(TemplateType::suggest_for_arch("phi-3"), Some(TemplateType::Phi3Instruct));
    assert_eq!(TemplateType::suggest_for_arch("phi-2"), Some(TemplateType::Phi2Instruct));
}

#[test]
fn llama_family_templates() {
    assert_eq!(TemplateType::suggest_for_arch("llama"), Some(TemplateType::Llama3Chat));
    assert_eq!(TemplateType::suggest_for_arch("llama2"), Some(TemplateType::Llama2Chat));
    assert_eq!(TemplateType::suggest_for_arch("llama-3.1"), Some(TemplateType::Llama31Chat));
    assert_eq!(TemplateType::suggest_for_arch("llama-3.2"), Some(TemplateType::Llama32Chat));
}

#[test]
fn mistral_family_templates() {
    assert_eq!(TemplateType::suggest_for_arch("mistral"), Some(TemplateType::MistralChat));
    assert_eq!(TemplateType::suggest_for_arch("mistral-nemo"), Some(TemplateType::MistralNemoChat));
    assert_eq!(TemplateType::suggest_for_arch("mixtral"), Some(TemplateType::MixtralInstruct));
}

#[test]
fn qwen_family_templates() {
    assert_eq!(TemplateType::suggest_for_arch("qwen"), Some(TemplateType::QwenChat));
    assert_eq!(TemplateType::suggest_for_arch("qwen2.5"), Some(TemplateType::Qwen25Chat));
}

#[test]
fn gemma_family_templates() {
    assert_eq!(TemplateType::suggest_for_arch("gemma"), Some(TemplateType::GemmaChat));
    assert_eq!(TemplateType::suggest_for_arch("gemma2"), Some(TemplateType::Gemma2Chat));
    assert_eq!(TemplateType::suggest_for_arch("codegemma"), Some(TemplateType::CodeGemma));
}

#[test]
fn deepseek_family_templates() {
    assert_eq!(TemplateType::suggest_for_arch("deepseek"), Some(TemplateType::DeepSeekChat));
    assert_eq!(TemplateType::suggest_for_arch("deepseek-v3"), Some(TemplateType::DeepSeekV3Chat));
}

#[test]
fn falcon_family_templates() {
    assert_eq!(TemplateType::suggest_for_arch("falcon"), Some(TemplateType::FalconChat));
    assert_eq!(TemplateType::suggest_for_arch("falcon-2"), Some(TemplateType::Falcon2Chat));
}

#[test]
fn olmo_family_templates() {
    assert_eq!(TemplateType::suggest_for_arch("olmo"), Some(TemplateType::OlmoInstruct));
    assert_eq!(TemplateType::suggest_for_arch("olmo2"), Some(TemplateType::OLMo2Chat));
}

#[test]
fn cohere_family_templates() {
    assert_eq!(TemplateType::suggest_for_arch("command"), Some(TemplateType::CohereCommand));
    assert_eq!(TemplateType::suggest_for_arch("command-r-plus"), Some(TemplateType::CommandRPlus));
    assert_eq!(TemplateType::suggest_for_arch("aya"), Some(TemplateType::CohereAya));
}

#[test]
fn no_template_for_base_models() {
    assert_eq!(TemplateType::suggest_for_arch("gpt"), None);
    assert_eq!(TemplateType::suggest_for_arch("bert"), None);
    assert_eq!(TemplateType::suggest_for_arch("bitnet"), None);
}

// ────────────────────────────────────────────────────────────────
// 6. all_variants count sanity check
// ────────────────────────────────────────────────────────────────

#[test]
fn all_variants_count_is_at_least_55() {
    let count = TemplateType::all_variants().len();
    assert!(count >= 55, "Expected at least 55 template variants, found {count}",);
}

#[test]
fn all_variants_are_unique() {
    let variants = TemplateType::all_variants();
    let mut seen = std::collections::HashSet::new();
    for v in variants {
        assert!(seen.insert(v.to_string()), "Duplicate variant found: {:?}", v,);
    }
}

// ────────────────────────────────────────────────────────────────
// 7. Template info consistency
// ────────────────────────────────────────────────────────────────

#[test]
fn all_variants_have_valid_info() {
    for variant in TemplateType::all_variants() {
        let info = variant.info();
        assert!(!info.name.is_empty(), "{:?} should have a non-empty name", variant,);
        // Name should match Display output
        assert_eq!(info.name, variant.to_string(), "{:?} info.name should match Display", variant,);
    }
}

// ────────────────────────────────────────────────────────────────
// 8. ChatML template family consistency
// ────────────────────────────────────────────────────────────────

#[test]
fn chatml_templates_use_im_tokens() {
    let chatml_variants = [
        TemplateType::Phi4Chat,
        TemplateType::QwenChat,
        TemplateType::DeepSeekChat,
        TemplateType::InternLMChat,
        TemplateType::YiChat,
        TemplateType::OrcaChat,
        TemplateType::NousHermes,
        TemplateType::TinyLlamaChat,
        TemplateType::DolphinChat,
        TemplateType::ChatGptChat,
        TemplateType::StableLMChat,
        TemplateType::JambaChat,
        TemplateType::Qwen25Chat,
        TemplateType::ArcticInstruct,
        TemplateType::DbrxInstruct,
        TemplateType::MiniCPMChat,
        TemplateType::DeepSeekV3Chat,
        TemplateType::Falcon2Chat,
        TemplateType::OLMo2Chat,
        TemplateType::CohereAya,
        TemplateType::SmolLMChat,
    ];

    for variant in &chatml_variants {
        let output = variant.apply("test", None);
        assert!(
            output.contains("<|im_start|>") && output.contains("<|im_end|>"),
            "{:?} should use ChatML im_start/im_end tokens, got: {}",
            variant,
            &output[..output.len().min(100)],
        );
    }
}

#[test]
fn llama_templates_use_header_tokens() {
    let output = TemplateType::Llama3Chat.apply("test", None);
    assert!(output.contains("<|start_header_id|>"), "Llama3Chat should use <|start_header_id|>",);
}

#[test]
fn gemma_templates_use_turn_tokens() {
    let output = TemplateType::GemmaChat.apply("test", None);
    assert!(output.contains("<start_of_turn>"), "GemmaChat should use <start_of_turn>",);
}

#[test]
fn mistral_templates_use_inst_tokens() {
    let output = TemplateType::MistralChat.apply("test", None);
    assert!(output.contains("[INST]"), "MistralChat should use [INST]",);
}

// ────────────────────────────────────────────────────────────────
// 9. render_chat multi-turn consistency
// ────────────────────────────────────────────────────────────────

#[test]
fn all_variants_render_chat_succeeds() {
    use bitnet_prompt_templates::{ChatRole, ChatTurn};

    let history = vec![
        ChatTurn::new(ChatRole::User, "Hello"),
        ChatTurn::new(ChatRole::Assistant, "Hi there!"),
        ChatTurn::new(ChatRole::User, "How are you?"),
    ];

    for variant in TemplateType::all_variants() {
        let result = variant.render_chat(&history, Some("You are helpful."));
        assert!(result.is_ok(), "{:?} render_chat failed: {:?}", variant, result.err(),);
        let rendered = result.unwrap();
        assert!(!rendered.is_empty(), "{:?} render_chat produced empty output", variant,);
    }
}

// ────────────────────────────────────────────────────────────────
// 10. PromptTemplate builder
// ────────────────────────────────────────────────────────────────

#[test]
fn prompt_template_builder_basic() {
    use bitnet_prompt_templates::PromptTemplate;

    let pt =
        PromptTemplate::new(TemplateType::Phi4Chat).with_system_prompt("You are a math tutor.");

    let output = pt.format("What is 2+2?");
    assert!(output.contains("What is 2+2?"));
    assert!(output.contains("math tutor"));
}

#[test]
fn prompt_template_builder_with_history() {
    use bitnet_prompt_templates::PromptTemplate;

    let mut pt =
        PromptTemplate::new(TemplateType::Llama3Chat).with_system_prompt("You are helpful.");

    pt.add_turn("Hello", "Hi there!");
    let output = pt.format("How are you?");
    assert!(output.contains("How are you?"));
}
