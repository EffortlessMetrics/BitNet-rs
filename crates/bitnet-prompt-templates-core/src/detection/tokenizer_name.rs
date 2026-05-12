use crate::TemplateType;

pub(super) fn detect(name: &str) -> Option<TemplateType> {
    let lower = name.to_ascii_lowercase();
    // TinyLlama must be checked before "llama" to avoid false match
    if lower.contains("tinyllama") || lower.contains("tiny-llama") {
        tracing::debug!(
            template = "TinyLlamaChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::TinyLlamaChat);
    }
    if lower.contains("llama-3.2") || lower.contains("llama3.2") || lower.contains("llama-32") {
        tracing::debug!(
            template = "Llama32Chat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::Llama32Chat);
    }
    if lower.contains("llama-3.1") || lower.contains("llama3.1") || lower.contains("llama-31") {
        tracing::debug!(
            template = "Llama31Chat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::Llama31Chat);
    }
    if lower.contains("llama3") || lower.contains("llama-3") {
        tracing::debug!(
            template = "Llama3Chat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::Llama3Chat);
    }
    if lower.contains("llama2") || lower.contains("llama-2") {
        tracing::debug!(
            template = "Llama2Chat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::Llama2Chat);
    }
    if lower.contains("qwen2.5") || lower.contains("qwen-2.5") {
        tracing::debug!(
            template = "Qwen25Chat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::Qwen25Chat);
    }
    if lower.contains("qwen") {
        tracing::debug!(
            template = "QwenChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::QwenChat);
    }
    if lower.contains("phi2") || lower.contains("phi-2") {
        tracing::debug!(
            template = "Phi2Instruct",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::Phi2Instruct);
    }
    if lower.contains("phi3") || lower.contains("phi-3") {
        tracing::debug!(
            template = "Phi3Instruct",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::Phi3Instruct);
    }
    // Dolphin must be checked before "phi" because "dolphin" contains "phi"
    if lower.contains("dolphin") {
        tracing::debug!(
            template = "DolphinChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::DolphinChat);
    }
    if lower.contains("phi") {
        tracing::debug!(
            template = "Phi4Chat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::Phi4Chat);
    }
    if lower.contains("codegemma") || lower.contains("code-gemma") {
        tracing::debug!(
            template = "CodeGemma",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::CodeGemma);
    }
    if lower.contains("gemma2") || lower.contains("gemma-2-") {
        tracing::debug!(
            template = "Gemma2Chat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::Gemma2Chat);
    }
    if lower.contains("gemma") {
        tracing::debug!(
            template = "GemmaChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::GemmaChat);
    }
    if lower.contains("mixtral") {
        tracing::debug!(
            template = "MixtralInstruct",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::MixtralInstruct);
    }
    if lower.contains("mistral-nemo") || lower.contains("nemo") {
        tracing::debug!(
            template = "MistralNemoChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::MistralNemoChat);
    }
    if lower.contains("mistral") {
        tracing::debug!(
            template = "MistralChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::MistralChat);
    }
    if lower.contains("deepseek-v3") || lower.contains("deepseekv3") || lower.contains("deepseek3")
    {
        tracing::debug!(
            template = "DeepSeekV3Chat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::DeepSeekV3Chat);
    }
    if lower.contains("deepseek") {
        tracing::debug!(
            template = "DeepSeekChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::DeepSeekChat);
    }
    if lower.contains("fim") || lower.contains("fill-in-middle") {
        tracing::debug!(
            template = "FillInMiddle",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::FillInMiddle);
    }
    if lower.contains("starcoder") || lower.contains("star-coder") {
        tracing::debug!(
            template = "StarCoder",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::StarCoder);
    }
    if lower.contains("falcon-2") || lower.contains("falcon2") {
        tracing::debug!(
            template = "Falcon2Chat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::Falcon2Chat);
    }
    if lower.contains("falcon") {
        tracing::debug!(
            template = "FalconChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::FalconChat);
    }
    if lower.contains("codellama") || lower.contains("code-llama") {
        tracing::debug!(
            template = "CodeLlamaInstruct",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::CodeLlamaInstruct);
    }
    if lower.contains("aya") {
        tracing::debug!(
            template = "CohereAya",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::CohereAya);
    }
    if lower.contains("cohere") || lower.contains("command-r") {
        tracing::debug!(
            template = "CommandRPlus",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::CommandRPlus);
    }
    if lower.contains("internlm") {
        tracing::debug!(
            template = "InternLMChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::InternLMChat);
    }
    if lower.contains("yi-") || lower.contains("yi_") || lower == "yi" {
        tracing::debug!(
            template = "YiChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::YiChat);
    }
    if lower.contains("baichuan") {
        tracing::debug!(
            template = "BaichuanChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::BaichuanChat);
    }
    if lower.contains("chatglm") || lower.contains("glm-4") || lower.contains("glm4") {
        tracing::debug!(
            template = "ChatGLMChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::ChatGLMChat);
    }
    if lower.contains("mpt") {
        tracing::debug!(
            template = "MptInstruct",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::MptInstruct);
    }
    if lower.contains("rwkv") {
        tracing::debug!(
            template = "RwkvWorld",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::RwkvWorld);
    }
    if lower.contains("olmo-2") || lower.contains("olmo2") {
        tracing::debug!(
            template = "OLMo2Chat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::OLMo2Chat);
    }
    if lower.contains("olmo") {
        tracing::debug!(
            template = "OlmoInstruct",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::OlmoInstruct);
    }
    if lower.contains("zephyr") {
        tracing::debug!(
            template = "ZephyrChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::ZephyrChat);
    }
    if lower.contains("vicuna") || lower.contains("sharegpt") {
        tracing::debug!(
            template = "VicunaChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::VicunaChat);
    }
    if lower.contains("orca") {
        tracing::debug!(
            template = "OrcaChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::OrcaChat);
    }
    if lower.contains("solar") {
        tracing::debug!(
            template = "SolarInstruct",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::SolarInstruct);
    }
    if lower.contains("alpaca") {
        tracing::debug!(
            template = "AlpacaInstruct",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::AlpacaInstruct);
    }
    if lower.contains("nous") || lower.contains("hermes") {
        tracing::debug!(
            template = "NousHermes",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::NousHermes);
    }
    if lower.contains("wizard") {
        tracing::debug!(
            template = "WizardLM",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::WizardLM);
    }
    if lower.contains("openchat") {
        tracing::debug!(
            template = "OpenChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::OpenChat);
    }
    if lower.contains("granite") {
        tracing::debug!(
            template = "GraniteChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::GraniteChat);
    }
    if lower.contains("nemotron") {
        tracing::debug!(
            template = "NemotronChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::NemotronChat);
    }
    if lower.contains("saiga") {
        tracing::debug!(
            template = "SaigaChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::SaigaChat);
    }
    // ChatGPT/GPT-4 must be checked before generic "instruct" fallback
    // and must NOT match "gpt2" (that's the base GPT template)
    if lower.contains("chatgpt")
        || lower.contains("gpt-4")
        || (lower.contains("gpt4") && !lower.contains("gpt2"))
    {
        tracing::debug!(
            template = "ChatGptChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::ChatGptChat);
    }
    if lower.contains("stablelm") || lower.contains("stable-lm") || lower.contains("stablecode") {
        tracing::debug!(
            template = "StableLMChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::StableLMChat);
    }
    if lower.contains("bloom") || lower.contains("bloomz") {
        tracing::debug!(
            template = "BloomChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::BloomChat);
    }
    if lower.contains("jamba") {
        tracing::debug!(
            template = "JambaChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::JambaChat);
    }
    if lower.contains("persimmon") || lower.contains("adept") {
        tracing::debug!(
            template = "PersimmonChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::PersimmonChat);
    }
    if lower.contains("xverse") {
        tracing::debug!(
            template = "XverseChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::XverseChat);
    }
    if lower.contains("arctic") {
        tracing::debug!(
            template = "ArcticInstruct",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::ArcticInstruct);
    }
    if lower.contains("dbrx") {
        tracing::debug!(
            template = "DbrxInstruct",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::DbrxInstruct);
    }
    if lower.contains("exaone") {
        tracing::debug!(
            template = "ExaoneChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::ExaoneChat);
    }
    if lower.contains("minicpm") {
        tracing::debug!(
            template = "MiniCPMChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::MiniCPMChat);
    }
    if lower.contains("smollm") || lower.contains("smol-lm") {
        tracing::debug!(
            template = "SmolLMChat",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::SmolLMChat);
    }
    if lower.contains("instruct") {
        tracing::debug!(
            template = "Instruct",
            source = "tokenizer_name",
            hint = name,
            "auto-detected prompt template"
        );
        return Some(TemplateType::Instruct);
    }
    None
}
