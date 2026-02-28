//! Snapshot tests for bitnet-prompt-templates.
//!
//! Pins the exact formatted output of each TemplateType under common scenarios:
//! - Raw passthrough
//! - Instruct with/without system prompt
//! - Llama3Chat single-turn and multi-turn
//! - Multi-turn history format

use bitnet_prompt_templates::{PromptTemplate, TemplateType};

#[test]
fn snapshot_raw_simple() {
    let tmpl = PromptTemplate::new(TemplateType::Raw);
    let out = tmpl.format("What is the capital of France?");
    insta::assert_snapshot!("raw_simple", out);
}

#[test]
fn snapshot_raw_with_system() {
    let tmpl =
        PromptTemplate::new(TemplateType::Raw).with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("raw_with_system", out);
}

#[test]
fn snapshot_raw_multi_turn() {
    let mut tmpl =
        PromptTemplate::new(TemplateType::Raw).with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("raw_multi_turn", out);
}

#[test]
fn snapshot_instruct_no_system() {
    let tmpl = PromptTemplate::new(TemplateType::Instruct);
    let out = tmpl.format("What is 2+2?");
    insta::assert_snapshot!("instruct_no_system", out);
}

#[test]
fn snapshot_instruct_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::Instruct)
        .with_system_prompt("You are a helpful assistant.");
    let out = tmpl.format("What is 2+2?");
    insta::assert_snapshot!("instruct_with_system_prompt", out);
}

#[test]
fn snapshot_llama3_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::Llama3Chat);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("llama3_single_turn", out);
}

#[test]
fn snapshot_llama3_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::Llama3Chat)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("llama3_with_system", out);
}

#[test]
fn snapshot_instruct_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::Instruct);
    tmpl.add_turn("What is Rust?", "Rust is a systems programming language.");
    tmpl.add_turn("Is it fast?", "Yes, Rust is very fast.");
    let out = tmpl.format("Can I use it for web servers?");
    insta::assert_snapshot!("instruct_multi_turn", out);
}

#[test]
fn snapshot_llama3_multi_turn() {
    let mut tmpl =
        PromptTemplate::new(TemplateType::Llama3Chat).with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("llama3_multi_turn", out);
}

#[test]
fn snapshot_phi4_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::Phi4Chat);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("phi4_single_turn", out);
}

#[test]
fn snapshot_phi4_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::Phi4Chat)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("phi4_with_system", out);
}

#[test]
fn snapshot_phi4_multi_turn() {
    let mut tmpl =
        PromptTemplate::new(TemplateType::Phi4Chat).with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("phi4_multi_turn", out);
}

#[test]
fn snapshot_qwen_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::QwenChat);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("qwen_single_turn", out);
}

#[test]
fn snapshot_qwen_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::QwenChat)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("qwen_with_system", out);
}

#[test]
fn snapshot_qwen_multi_turn() {
    let mut tmpl =
        PromptTemplate::new(TemplateType::QwenChat).with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("qwen_multi_turn", out);
}

#[test]
fn snapshot_gemma_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::GemmaChat);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("gemma_single_turn", out);
}

#[test]
fn snapshot_gemma_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::GemmaChat)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("gemma_with_system", out);
}

#[test]
fn snapshot_gemma_multi_turn() {
    let mut tmpl =
        PromptTemplate::new(TemplateType::GemmaChat).with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("gemma_multi_turn", out);
}

#[test]
fn snapshot_mistral_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::MistralChat);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("mistral_single_turn", out);
}

#[test]
fn snapshot_mistral_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::MistralChat)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("mistral_with_system", out);
}

#[test]
fn snapshot_mistral_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::MistralChat)
        .with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("mistral_multi_turn", out);
}

// ── DeepSeek Chat ──────────────────────────────────────────────────

#[test]
fn snapshot_deepseek_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::DeepSeekChat);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("deepseek_single_turn", out);
}

#[test]
fn snapshot_deepseek_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::DeepSeekChat)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("deepseek_with_system", out);
}

#[test]
fn snapshot_deepseek_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::DeepSeekChat)
        .with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("deepseek_multi_turn", out);
}

// ── StarCoder ───────────────────────────────────────────────────

#[test]
fn snapshot_starcoder_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::StarCoder);
    let out = tmpl.format("Write a fibonacci function in Python.");
    insta::assert_snapshot!("starcoder_single_turn", out);
}

#[test]
fn snapshot_starcoder_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::StarCoder)
        .with_system_prompt("You are a Python expert.");
    let out = tmpl.format("Write a sort function.");
    insta::assert_snapshot!("starcoder_with_system", out);
}

#[test]
fn snapshot_starcoder_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::StarCoder)
        .with_system_prompt("You are a code assistant.");
    tmpl.add_turn("Write hello world", "print('Hello, world!')");
    let out = tmpl.format("Now write a loop.");
    insta::assert_snapshot!("starcoder_multi_turn", out);
}

// ── Falcon Chat ──────────────────────────────────────────────────

#[test]
fn snapshot_falcon_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::FalconChat);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("falcon_single_turn", out);
}

#[test]
fn snapshot_falcon_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::FalconChat)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("falcon_with_system", out);
}

#[test]
fn snapshot_falcon_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::FalconChat)
        .with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("falcon_multi_turn", out);
}

// ── CodeLlama Instruct ─────────────────────────────────────────

#[test]
fn snapshot_codellama_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::CodeLlamaInstruct);
    let out = tmpl.format("Write a fibonacci function in Python.");
    insta::assert_snapshot!("codellama_single_turn", out);
}

#[test]
fn snapshot_codellama_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::CodeLlamaInstruct)
        .with_system_prompt("You are a Python expert.");
    let out = tmpl.format("Write a sort function.");
    insta::assert_snapshot!("codellama_with_system", out);
}

#[test]
fn snapshot_codellama_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::CodeLlamaInstruct)
        .with_system_prompt("You are a code assistant.");
    tmpl.add_turn("Write hello world", "print('Hello, world!')");
    let out = tmpl.format("Now write a loop.");
    insta::assert_snapshot!("codellama_multi_turn", out);
}

// ── Cohere Command ─────────────────────────────────────────────

#[test]
fn snapshot_cohere_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::CohereCommand);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("cohere_single_turn", out);
}

#[test]
fn snapshot_cohere_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::CohereCommand)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("cohere_with_system", out);
}

#[test]
fn snapshot_cohere_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::CohereCommand)
        .with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("cohere_multi_turn", out);
}

// ── InternLM Chat ──────────────────────────────────────────────

#[test]
fn snapshot_internlm_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::InternLMChat);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("internlm_single_turn", out);
}

#[test]
fn snapshot_internlm_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::InternLMChat)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("internlm_with_system", out);
}

#[test]
fn snapshot_internlm_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::InternLMChat)
        .with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("internlm_multi_turn", out);
}

// ── Yi Chat ────────────────────────────────────────────────────

#[test]
fn snapshot_yi_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::YiChat);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("yi_single_turn", out);
}

#[test]
fn snapshot_yi_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::YiChat)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("yi_with_system", out);
}

#[test]
fn snapshot_yi_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::YiChat)
        .with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("yi_multi_turn", out);
}

// ── Baichuan Chat ──────────────────────────────────────────────

#[test]
fn snapshot_baichuan_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::BaichuanChat);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("baichuan_single_turn", out);
}

#[test]
fn snapshot_baichuan_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::BaichuanChat)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("baichuan_with_system", out);
}

#[test]
fn snapshot_baichuan_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::BaichuanChat)
        .with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("baichuan_multi_turn", out);
}

// ── ChatGLM Chat ───────────────────────────────────────────────

#[test]
fn snapshot_chatglm_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::ChatGLMChat);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("chatglm_single_turn", out);
}

#[test]
fn snapshot_chatglm_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::ChatGLMChat)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("chatglm_with_system", out);
}

#[test]
fn snapshot_chatglm_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::ChatGLMChat)
        .with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("chatglm_multi_turn", out);
}

// ── MPT Instruct ───────────────────────────────────────────────

#[test]
fn snapshot_mpt_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::MptInstruct);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("mpt_single_turn", out);
}

#[test]
fn snapshot_mpt_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::MptInstruct)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("mpt_with_system", out);
}

#[test]
fn snapshot_mpt_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::MptInstruct)
        .with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("mpt_multi_turn", out);
}

// ── RWKV World ──────────────────────────────────────────────────

#[test]
fn test_rwkv_world_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::RwkvWorld);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("rwkv_world_single_turn", out);
}

#[test]
fn test_rwkv_world_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::RwkvWorld)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("rwkv_world_with_system", out);
}

#[test]
fn test_rwkv_world_multi_turn() {
    let mut tmpl =
        PromptTemplate::new(TemplateType::RwkvWorld).with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("rwkv_world_multi_turn", out);
}

// ── OLMo Instruct ───────────────────────────────────────────────

#[test]
fn test_olmo_instruct_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::OlmoInstruct);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("olmo_instruct_single_turn", out);
}

#[test]
fn test_olmo_instruct_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::OlmoInstruct)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("olmo_instruct_with_system", out);
}

#[test]
fn test_olmo_instruct_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::OlmoInstruct)
        .with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("olmo_instruct_multi_turn", out);
}

// ── Fill-in-the-Middle ──────────────────────────────────────────────

#[test]
fn snapshot_fim_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::FillInMiddle);
    let out = tmpl.format("def fibonacci(n):");
    insta::assert_snapshot!("fim_single_turn", out);
}

#[test]
fn snapshot_fim_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::FillInMiddle)
        .with_system_prompt("return result");
    let out = tmpl.format("def fibonacci(n):");
    insta::assert_snapshot!("fim_with_system", out);
}

#[test]
fn snapshot_fim_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::FillInMiddle)
        .with_system_prompt("return result");
    tmpl.add_turn("Write hello world", "print('Hello, world!')");
    let out = tmpl.format("def sort(arr):");
    insta::assert_snapshot!("fim_multi_turn", out);
}

// ── Zephyr Chat ─────────────────────────────────────────────────────

#[test]
fn snapshot_zephyr_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::ZephyrChat);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("zephyr_single_turn", out);
}

#[test]
fn snapshot_zephyr_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::ZephyrChat)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("zephyr_with_system", out);
}

#[test]
fn snapshot_zephyr_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::ZephyrChat)
        .with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("zephyr_multi_turn", out);
}

// ── Vicuna Chat ─────────────────────────────────────────────────────

#[test]
fn snapshot_vicuna_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::VicunaChat);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("vicuna_single_turn", out);
}

#[test]
fn snapshot_vicuna_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::VicunaChat)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("vicuna_with_system", out);
}

#[test]
fn snapshot_vicuna_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::VicunaChat)
        .with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("vicuna_multi_turn", out);
}

// ── Orca Chat ───────────────────────────────────────────────────────

#[test]
fn snapshot_orca_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::OrcaChat);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("orca_single_turn", out);
}

#[test]
fn snapshot_orca_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::OrcaChat)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("orca_with_system", out);
}

#[test]
fn snapshot_orca_multi_turn() {
    let mut tmpl =
        PromptTemplate::new(TemplateType::OrcaChat).with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("orca_multi_turn", out);
}

// ── Solar Instruct ──────────────────────────────────────────────────

#[test]
fn snapshot_solar_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::SolarInstruct);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("solar_single_turn", out);
}

#[test]
fn snapshot_solar_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::SolarInstruct)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("solar_with_system", out);
}

#[test]
fn snapshot_solar_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::SolarInstruct)
        .with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("solar_multi_turn", out);
}

// ── Alpaca Instruct ─────────────────────────────────────────────────

#[test]
fn snapshot_alpaca_single_turn() {
    let tmpl = PromptTemplate::new(TemplateType::AlpacaInstruct);
    let out = tmpl.format("Explain photosynthesis briefly.");
    insta::assert_snapshot!("alpaca_single_turn", out);
}

#[test]
fn snapshot_alpaca_with_system() {
    let tmpl = PromptTemplate::new(TemplateType::AlpacaInstruct)
        .with_system_prompt("You are a science tutor.");
    let out = tmpl.format("What is ATP?");
    insta::assert_snapshot!("alpaca_with_system", out);
}

#[test]
fn snapshot_alpaca_multi_turn() {
    let mut tmpl = PromptTemplate::new(TemplateType::AlpacaInstruct)
        .with_system_prompt("You are a Rust expert.");
    tmpl.add_turn("What is ownership?", "Ownership is Rust's memory management system.");
    let out = tmpl.format("How does borrowing work?");
    insta::assert_snapshot!("alpaca_multi_turn", out);
}
