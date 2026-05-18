use crate::TemplateType;

struct SignatureRule {
    template: TemplateType,
    matches: fn(&str) -> bool,
}

pub(super) fn detect(jinja: &str) -> Option<TemplateType> {
    ORDERED_SIGNATURES.iter().find(|rule| (rule.matches)(jinja)).map(|rule| rule.template)
}

const ORDERED_SIGNATURES: &[SignatureRule] = &[
    // GGUF chat_template metadata is authoritative when it exposes a known
    // signature. Ordering is part of the contract where template families share
    // markers (for example Command-R+ before Cohere Command and Granite before
    // ChatML).
    SignatureRule { template: TemplateType::Llama3Chat, matches: is_llama3 },
    SignatureRule { template: TemplateType::BitnetCppAnswer, matches: is_bitnet_answer },
    SignatureRule { template: TemplateType::FillInMiddle, matches: is_fill_in_middle },
    SignatureRule { template: TemplateType::CommandRPlus, matches: is_command_r_plus },
    SignatureRule { template: TemplateType::GraniteChat, matches: is_granite },
    SignatureRule { template: TemplateType::NemotronChat, matches: is_nemotron },
    SignatureRule { template: TemplateType::Phi3Instruct, matches: is_phi3 },
    SignatureRule { template: TemplateType::ExaoneChat, matches: is_exaone },
    SignatureRule { template: TemplateType::Phi4Chat, matches: is_chatml_or_phi4 },
    SignatureRule { template: TemplateType::GemmaChat, matches: is_gemma },
    SignatureRule { template: TemplateType::Llama2Chat, matches: is_llama2 },
    SignatureRule { template: TemplateType::MistralChat, matches: is_mistral },
    SignatureRule { template: TemplateType::CohereCommand, matches: is_cohere_command },
    SignatureRule { template: TemplateType::ChatGLMChat, matches: is_chatglm },
    SignatureRule { template: TemplateType::ZephyrChat, matches: is_zephyr },
    SignatureRule { template: TemplateType::OlmoInstruct, matches: is_olmo },
    SignatureRule { template: TemplateType::AlpacaInstruct, matches: is_alpaca },
    SignatureRule { template: TemplateType::SolarInstruct, matches: is_solar },
    SignatureRule { template: TemplateType::MptInstruct, matches: is_mpt },
    SignatureRule { template: TemplateType::OpenChat, matches: is_openchat },
    SignatureRule { template: TemplateType::WizardLM, matches: is_wizardlm },
    SignatureRule { template: TemplateType::VicunaChat, matches: is_vicuna },
    SignatureRule { template: TemplateType::Instruct, matches: is_generic_instruct },
];

fn is_llama3(jinja: &str) -> bool {
    jinja.contains("<|start_header_id|>") && jinja.contains("<|eot_id|>")
}

fn is_bitnet_answer(jinja: &str) -> bool {
    TemplateType::looks_like_bitnet_answer_template(jinja)
}

fn is_fill_in_middle(jinja: &str) -> bool {
    jinja.contains("<fim_prefix>")
}

fn is_command_r_plus(jinja: &str) -> bool {
    jinja.contains("<|START_OF_TURN_TOKEN|>")
}

fn is_granite(jinja: &str) -> bool {
    jinja.contains("<|start_of_role|>")
}

fn is_nemotron(jinja: &str) -> bool {
    jinja.contains("<extra_id_0>") || jinja.contains("<extra_id_1>")
}

fn is_phi3(jinja: &str) -> bool {
    jinja.contains("<|system|>") && jinja.contains("<|end|>") && jinja.contains("<|user|>")
}

fn is_exaone(jinja: &str) -> bool {
    jinja.contains("[|system|]") || jinja.contains("[|endofturn|]")
}

fn is_chatml_or_phi4(jinja: &str) -> bool {
    jinja.contains("<|im_start|>") && jinja.contains("<|im_end|>")
}

fn is_gemma(jinja: &str) -> bool {
    jinja.contains("<start_of_turn>") && jinja.contains("<end_of_turn>")
}

fn is_llama2(jinja: &str) -> bool {
    jinja.contains("[INST]") && jinja.contains("<<SYS>>") && jinja.contains("<</SYS>>")
}

fn is_mistral(jinja: &str) -> bool {
    jinja.contains("[INST]") && jinja.contains("[/INST]")
}

fn is_cohere_command(jinja: &str) -> bool {
    jinja.contains("<|START_OF_TURN_TOKEN|>") && jinja.contains("<|END_OF_TURN_TOKEN|>")
}

fn is_chatglm(jinja: &str) -> bool {
    jinja.contains("[gMASK]")
}

fn is_zephyr(jinja: &str) -> bool {
    jinja.contains("</s>")
        && jinja.contains("<|user|>")
        && !jinja.contains("[gMASK]")
        && !jinja.contains("<|im_start|>")
}

fn is_olmo(jinja: &str) -> bool {
    jinja.contains("<|user|>") && jinja.contains("<|assistant|>")
}

fn is_alpaca(jinja: &str) -> bool {
    jinja.contains("### Instruction:")
        && jinja.contains("### Response:")
        && !jinja.contains("### User:")
}

fn is_solar(jinja: &str) -> bool {
    jinja.contains("### User:") && jinja.contains("### Assistant:")
}

fn is_mpt(jinja: &str) -> bool {
    jinja.contains("### Instruction") && jinja.contains("### Response")
}

fn is_openchat(jinja: &str) -> bool {
    jinja.contains("GPT4 Correct")
}

fn is_wizardlm(jinja: &str) -> bool {
    jinja.contains("USER:") && jinja.contains("ASSISTANT:") && jinja.contains("A chat between")
}

fn is_vicuna(jinja: &str) -> bool {
    jinja.contains("USER:") && jinja.contains("ASSISTANT:")
}

fn is_generic_instruct(jinja: &str) -> bool {
    jinja.contains("{% for message in messages %}")
}
