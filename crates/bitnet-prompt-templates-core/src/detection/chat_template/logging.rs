use crate::TemplateType;

pub(super) fn detected(template: TemplateType) {
    tracing::debug!(
        template = trace_name(template),
        source = "gguf_chat_template",
        "auto-detected prompt template"
    );
}

fn trace_name(template: TemplateType) -> &'static str {
    match template {
        TemplateType::Llama3Chat => "Llama3Chat",
        TemplateType::BitnetCppAnswer => "BitnetCppAnswer",
        TemplateType::FillInMiddle => "FillInMiddle",
        TemplateType::CommandRPlus => "CommandRPlus",
        TemplateType::GraniteChat => "GraniteChat",
        TemplateType::NemotronChat => "NemotronChat",
        TemplateType::Phi3Instruct => "Phi3Instruct",
        TemplateType::ExaoneChat => "ExaoneChat",
        TemplateType::Phi4Chat => "Phi4Chat",
        TemplateType::GemmaChat => "GemmaChat",
        TemplateType::Llama2Chat => "Llama2Chat",
        TemplateType::MistralChat => "MistralChat",
        TemplateType::CohereCommand => "CohereCommand",
        TemplateType::ChatGLMChat => "ChatGLMChat",
        TemplateType::ZephyrChat => "ZephyrChat",
        TemplateType::OlmoInstruct => "OlmoInstruct",
        TemplateType::AlpacaInstruct => "AlpacaInstruct",
        TemplateType::SolarInstruct => "SolarInstruct",
        TemplateType::MptInstruct => "MptInstruct",
        TemplateType::OpenChat => "OpenChat",
        TemplateType::WizardLM => "WizardLM",
        TemplateType::VicunaChat => "VicunaChat",
        TemplateType::Instruct => "Instruct",
        _ => "Unknown",
    }
}
