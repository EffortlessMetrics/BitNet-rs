use crate::TemplateType;

pub(super) fn detect(jinja: &str) -> Option<TemplateType> {
    // GGUF chat_template metadata is authoritative when it exposes a known signature.
    // LLaMA-3 signature
    if jinja.contains("<|start_header_id|>") && jinja.contains("<|eot_id|>") {
        tracing::debug!(
            template = "Llama3Chat",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::Llama3Chat);
    }
    if TemplateType::looks_like_bitnet_answer_template(jinja) {
        tracing::debug!(
            template = "BitnetCppAnswer",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::BitnetCppAnswer);
    }
    // Fill-in-the-middle signature
    if jinja.contains("<fim_prefix>") {
        tracing::debug!(
            template = "FillInMiddle",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::FillInMiddle);
    }
    // Command-R+ signature (must be before CohereCommand)
    if jinja.contains("<|START_OF_TURN_TOKEN|>") {
        tracing::debug!(
            template = "CommandRPlus",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::CommandRPlus);
    }
    // Granite signature (must be before ChatML check)
    if jinja.contains("<|start_of_role|>") {
        tracing::debug!(
            template = "GraniteChat",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::GraniteChat);
    }
    // Nemotron signature (must be before ChatML check)
    if jinja.contains("<extra_id_0>") || jinja.contains("<extra_id_1>") {
        tracing::debug!(
            template = "NemotronChat",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::NemotronChat);
    }
    // Phi-3 signature (must be before ChatML check; uses <|system|>/<|end|>/<|user|>)
    if jinja.contains("<|system|>") && jinja.contains("<|end|>") && jinja.contains("<|user|>") {
        tracing::debug!(
            template = "Phi3Instruct",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::Phi3Instruct);
    }
    // EXAONE signature (must be before ChatML check)
    if jinja.contains("[|system|]") || jinja.contains("[|endofturn|]") {
        tracing::debug!(
            template = "ExaoneChat",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::ExaoneChat);
    }
    // ChatML / Phi-4 signature
    if jinja.contains("<|im_start|>") && jinja.contains("<|im_end|>") {
        tracing::debug!(
            template = "Phi4Chat",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::Phi4Chat);
    }
    // Gemma signature
    if jinja.contains("<start_of_turn>") && jinja.contains("<end_of_turn>") {
        tracing::debug!(
            template = "GemmaChat",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::GemmaChat);
    }
    // [INST] with <<SYS>>/<</SYS>> maps to Llama2Chat (CodeLlama handled by tokenizer name)
    if jinja.contains("[INST]") && jinja.contains("<<SYS>>") && jinja.contains("<</SYS>>") {
        tracing::debug!(
            template = "Llama2Chat",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::Llama2Chat);
    }
    // Mistral [INST] signature (no <<SYS>>)
    if jinja.contains("[INST]") && jinja.contains("[/INST]") {
        tracing::debug!(
            template = "MistralChat",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::MistralChat);
    }
    // Cohere Command signature
    if jinja.contains("<|START_OF_TURN_TOKEN|>") && jinja.contains("<|END_OF_TURN_TOKEN|>") {
        tracing::debug!(
            template = "CohereCommand",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::CohereCommand);
    }
    // ChatGLM/GLM-4 signature (requires [gMASK])
    if jinja.contains("[gMASK]") {
        tracing::debug!(
            template = "ChatGLMChat",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::ChatGLMChat);
    }
    // Zephyr signature (</s> delimiters with <|user|>, no [gMASK] or <|im_start|>)
    if jinja.contains("</s>")
        && jinja.contains("<|user|>")
        && !jinja.contains("[gMASK]")
        && !jinja.contains("<|im_start|>")
    {
        tracing::debug!(
            template = "ZephyrChat",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::ZephyrChat);
    }
    // OLMo instruct signature (<|user|>/<|assistant|> without [gMASK])
    if jinja.contains("<|user|>") && jinja.contains("<|assistant|>") {
        tracing::debug!(
            template = "OlmoInstruct",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::OlmoInstruct);
    }
    // Alpaca ### Instruction/### Response markers (but NOT ### User: which is Solar)
    if jinja.contains("### Instruction:")
        && jinja.contains("### Response:")
        && !jinja.contains("### User:")
    {
        tracing::debug!(
            template = "AlpacaInstruct",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::AlpacaInstruct);
    }
    // Solar ### User:/### Assistant: markers
    if jinja.contains("### User:") && jinja.contains("### Assistant:") {
        tracing::debug!(
            template = "SolarInstruct",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::SolarInstruct);
    }
    // MPT ### instruction markers
    if jinja.contains("### Instruction") && jinja.contains("### Response") {
        tracing::debug!(
            template = "MptInstruct",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::MptInstruct);
    }
    // OpenChat GPT4 Correct signature
    if jinja.contains("GPT4 Correct") {
        tracing::debug!(
            template = "OpenChat",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::OpenChat);
    }
    // WizardLM signature (USER:/ASSISTANT: with "A chat between")
    if jinja.contains("USER:") && jinja.contains("ASSISTANT:") && jinja.contains("A chat between") {
        tracing::debug!(
            template = "WizardLM",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::WizardLM);
    }
    // Vicuna/ShareGPT signature (USER:/ASSISTANT:, not Falcon's "User:")
    if jinja.contains("USER:") && jinja.contains("ASSISTANT:") {
        tracing::debug!(
            template = "VicunaChat",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::VicunaChat);
    }
    // Generic instruct template
    if jinja.contains("{% for message in messages %}") {
        tracing::debug!(
            template = "Instruct",
            source = "gguf_chat_template",
            "auto-detected prompt template"
        );
        return Some(TemplateType::Instruct);
    }
    None
}
