//! # Prompt Template System
//!
//! Provides chat and instruct format templates for common model families.
//! Ensures proper prompt formatting for optimal model behavior.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// Role in a chat conversation
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        }
    }
}

/// A single turn in a chat conversation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatTurn {
    pub role: ChatRole,
    pub text: String,
}

impl ChatTurn {
    pub fn new(role: ChatRole, text: impl Into<String>) -> Self {
        Self { role, text: text.into() }
    }
}

/// Supported prompt template types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TemplateType {
    /// Raw text (no formatting)
    Raw,
    /// Simple Q&A instruct format
    Instruct,
    /// LLaMA-3 chat format with special tokens
    Llama3Chat,
    /// Phi-4 ChatML format with im_start/im_end tokens
    Phi4Chat,
    /// Qwen ChatML format with im_start/im_end tokens
    QwenChat,
    /// Gemma chat format with start_of_turn/end_of_turn tokens
    GemmaChat,
    /// Mistral chat format with [INST]...[/INST] tokens
    MistralChat,
    /// DeepSeek ChatML format with im_start/im_end tokens
    DeepSeekChat,
    /// StarCoder code completion format (prefix/suffix/middle FIM tokens)
    StarCoder,
    /// Falcon chat format (User:/Assistant: roles)
    FalconChat,
    /// CodeLlama instruct format (LLaMA-style [INST] for code)
    CodeLlamaInstruct,
    /// Cohere Command format with special turn tokens
    CohereCommand,
    /// InternLM ChatML format with im_start/im_end tokens
    InternLMChat,
    /// Yi chat format (ChatML-style with im_start/im_end)
    YiChat,
    /// Baichuan chat format with custom role tokens
    BaichuanChat,
    /// ChatGLM/GLM-4 chat format with custom role markers
    ChatGLMChat,
    /// MPT instruct format (simple ### markers)
    MptInstruct,
    /// RWKV World format (User:/Assistant: roles for RWKV-5/6 models)
    RwkvWorld,
    /// OLMo instruct format (<|user|>/<|assistant|> tokens)
    OlmoInstruct,
    /// Fill-in-the-middle format for code infilling (<fim_prefix>/<fim_suffix>/<fim_middle>)
    FillInMiddle,
    /// HuggingFace Zephyr chat format (<|user|>/<|assistant|> with </s> delimiters)
    ZephyrChat,
    /// Vicuna/ShareGPT chat format (USER:/ASSISTANT: roles)
    VicunaChat,
    /// Orca ChatML format with Orca default system prompt
    OrcaChat,
    /// SOLAR instruct format (### User:/### Assistant:)
    SolarInstruct,
    /// Stanford Alpaca instruct format (### Instruction:/### Response:)
    AlpacaInstruct,
    /// Cohere Command-R+ format with START_OF_TURN_TOKEN markers
    CommandRPlus,
    /// NousResearch Hermes ChatML variant with safety-focused default system prompt
    NousHermes,
    /// WizardLM Vicuna-derived format (USER:/ASSISTANT: with descriptive preamble)
    WizardLM,
    /// OpenChat GPT4 Correct User/Assistant format with end_of_turn
    OpenChat,
    /// IBM Granite chat format with start_of_role/end_of_role tokens
    GraniteChat,
    /// NVIDIA Nemotron chat format with extra_id tokens
    NemotronChat,
    /// Russian Saiga/YandexGPT ChatML variant with Cyrillic system prompt
    SaigaChat,
    /// Meta Llama-2 chat format with [INST]<<SYS>>/<</SYS>> markers
    Llama2Chat,
    /// Google Gemma 2 chat format (same turn tokens as Gemma, version-specific detection)
    Gemma2Chat,
    /// Microsoft Phi-3 instruct format with <|system|>/<|user|>/<|assistant|>/<|end|> markers
    Phi3Instruct,
}

impl std::str::FromStr for TemplateType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "raw" => Ok(Self::Raw),
            "instruct" => Ok(Self::Instruct),
            "llama3-chat" | "llama3_chat" => Ok(Self::Llama3Chat),
            "phi4-chat" | "phi4_chat" | "phi4" | "chatml" => Ok(Self::Phi4Chat),
            "qwen-chat" | "qwen_chat" | "qwen" => Ok(Self::QwenChat),
            "gemma-chat" | "gemma_chat" | "gemma" => Ok(Self::GemmaChat),
            "mistral-chat" | "mistral_chat" | "mistral" => Ok(Self::MistralChat),
            "deepseek-chat" | "deepseek_chat" | "deepseek" => Ok(Self::DeepSeekChat),
            "starcoder" | "star_coder" | "code-completion" => Ok(Self::StarCoder),
            "falcon-chat" | "falcon_chat" | "falcon" => Ok(Self::FalconChat),
            "codellama-instruct" | "codellama_instruct" | "codellama" => {
                Ok(Self::CodeLlamaInstruct)
            }
            "cohere-command" | "cohere_command" | "cohere" | "command-r" => Ok(Self::CohereCommand),
            "internlm-chat" | "internlm_chat" | "internlm" => Ok(Self::InternLMChat),
            "yi-chat" | "yi_chat" | "yi" => Ok(Self::YiChat),
            "baichuan-chat" | "baichuan_chat" | "baichuan" => Ok(Self::BaichuanChat),
            "chatglm-chat" | "chatglm_chat" | "chatglm" | "glm-4" | "glm4" => Ok(Self::ChatGLMChat),
            "mpt-instruct" | "mpt_instruct" | "mpt" => Ok(Self::MptInstruct),
            "rwkv-world" | "rwkv_world" | "rwkv" => Ok(Self::RwkvWorld),
            "olmo-instruct" | "olmo_instruct" | "olmo" => Ok(Self::OlmoInstruct),
            "fill-in-middle" | "fim" => Ok(Self::FillInMiddle),
            "zephyr-chat" | "zephyr" => Ok(Self::ZephyrChat),
            "vicuna-chat" | "vicuna" => Ok(Self::VicunaChat),
            "orca-chat" | "orca" => Ok(Self::OrcaChat),
            "solar-instruct" | "solar" => Ok(Self::SolarInstruct),
            "alpaca-instruct" | "alpaca" => Ok(Self::AlpacaInstruct),
            "command-r-plus" | "command-r+" | "commandr" => Ok(Self::CommandRPlus),
            "nous-hermes" | "nous" | "hermes" => Ok(Self::NousHermes),
            "wizard-lm" | "wizard" | "wizardlm" => Ok(Self::WizardLM),
            "openchat" | "open-chat" => Ok(Self::OpenChat),
            "granite-chat" | "granite" => Ok(Self::GraniteChat),
            "nemotron-chat" | "nemotron" => Ok(Self::NemotronChat),
            "saiga-chat" | "saiga" => Ok(Self::SaigaChat),
            "llama2-chat" | "llama-2-chat" | "llama2" => Ok(Self::Llama2Chat),
            "gemma2-chat" | "gemma-2-chat" | "gemma2" => Ok(Self::Gemma2Chat),
            "phi3-instruct" | "phi-3-instruct" | "phi3" => Ok(Self::Phi3Instruct),
            _ => bail!(
                "Unknown template type: {}. Supported: raw, instruct, \
                 llama3-chat, phi4-chat, qwen-chat, gemma-chat, \
                 mistral-chat, deepseek-chat, starcoder, falcon-chat, \
                 codellama-instruct, cohere-command, internlm-chat, \
                 yi-chat, baichuan-chat, chatglm-chat, mpt-instruct, \
                 rwkv-world, olmo-instruct, fill-in-middle, \
                 zephyr-chat, vicuna-chat, orca-chat, solar-instruct, \
                 alpaca-instruct, command-r-plus, nous-hermes, \
                 wizard-lm, openchat, granite-chat, nemotron-chat, \
                 saiga-chat, llama2-chat, gemma2-chat, phi3-instruct",
                s
            ),
        }
    }
}

impl std::fmt::Display for TemplateType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Raw => write!(f, "raw"),
            Self::Instruct => write!(f, "instruct"),
            Self::Llama3Chat => write!(f, "llama3-chat"),
            Self::Phi4Chat => write!(f, "phi4-chat"),
            Self::QwenChat => write!(f, "qwen-chat"),
            Self::GemmaChat => write!(f, "gemma-chat"),
            Self::MistralChat => write!(f, "mistral-chat"),
            Self::DeepSeekChat => write!(f, "deepseek-chat"),
            Self::StarCoder => write!(f, "starcoder"),
            Self::FalconChat => write!(f, "falcon-chat"),
            Self::CodeLlamaInstruct => write!(f, "codellama-instruct"),
            Self::CohereCommand => write!(f, "cohere-command"),
            Self::InternLMChat => write!(f, "internlm-chat"),
            Self::YiChat => write!(f, "yi-chat"),
            Self::BaichuanChat => write!(f, "baichuan-chat"),
            Self::ChatGLMChat => write!(f, "chatglm-chat"),
            Self::MptInstruct => write!(f, "mpt-instruct"),
            Self::RwkvWorld => write!(f, "rwkv-world"),
            Self::OlmoInstruct => write!(f, "olmo-instruct"),
            Self::FillInMiddle => write!(f, "fill-in-middle"),
            Self::ZephyrChat => write!(f, "zephyr-chat"),
            Self::VicunaChat => write!(f, "vicuna-chat"),
            Self::OrcaChat => write!(f, "orca-chat"),
            Self::SolarInstruct => write!(f, "solar-instruct"),
            Self::AlpacaInstruct => write!(f, "alpaca-instruct"),
            Self::CommandRPlus => write!(f, "command-r-plus"),
            Self::NousHermes => write!(f, "nous-hermes"),
            Self::WizardLM => write!(f, "wizard-lm"),
            Self::OpenChat => write!(f, "openchat"),
            Self::GraniteChat => write!(f, "granite-chat"),
            Self::NemotronChat => write!(f, "nemotron-chat"),
            Self::SaigaChat => write!(f, "saiga-chat"),
            Self::Llama2Chat => write!(f, "llama2-chat"),
            Self::Gemma2Chat => write!(f, "gemma2-chat"),
            Self::Phi3Instruct => write!(f, "phi3-instruct"),
        }
    }
}

impl TemplateType {
    /// Detect template type from GGUF metadata and tokenizer hints.
    ///
    /// Priority order:
    /// 1. GGUF chat_template metadata (if present)
    /// 2. Tokenizer family name heuristics
    /// 3. Fallback to Raw
    pub fn detect(tokenizer_name: Option<&str>, chat_template_jinja: Option<&str>) -> Self {
        // Priority 1: GGUF chat_template metadata
        if let Some(jinja) = chat_template_jinja {
            // LLaMA-3 signature
            if jinja.contains("<|start_header_id|>") && jinja.contains("<|eot_id|>") {
                tracing::debug!(
                    template = "Llama3Chat",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::Llama3Chat;
            }
            // Fill-in-the-middle signature
            if jinja.contains("<fim_prefix>") {
                tracing::debug!(
                    template = "FillInMiddle",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::FillInMiddle;
            }
            // Command-R+ signature (must be before CohereCommand)
            if jinja.contains("<|START_OF_TURN_TOKEN|>") {
                tracing::debug!(
                    template = "CommandRPlus",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::CommandRPlus;
            }
            // Granite signature (must be before ChatML check)
            if jinja.contains("<|start_of_role|>") {
                tracing::debug!(
                    template = "GraniteChat",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::GraniteChat;
            }
            // Nemotron signature (must be before ChatML check)
            if jinja.contains("<extra_id_0>") || jinja.contains("<extra_id_1>") {
                tracing::debug!(
                    template = "NemotronChat",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::NemotronChat;
            }
            // Phi-3 signature (must be before ChatML check; uses <|system|>/<|end|>/<|user|>)
            if jinja.contains("<|system|>")
                && jinja.contains("<|end|>")
                && jinja.contains("<|user|>")
            {
                tracing::debug!(
                    template = "Phi3Instruct",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::Phi3Instruct;
            }
            // ChatML / Phi-4 signature
            if jinja.contains("<|im_start|>") && jinja.contains("<|im_end|>") {
                tracing::debug!(
                    template = "Phi4Chat",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::Phi4Chat;
            }
            // Gemma signature
            if jinja.contains("<start_of_turn>") && jinja.contains("<end_of_turn>") {
                tracing::debug!(
                    template = "GemmaChat",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::GemmaChat;
            }
            // [INST] with <<SYS>>/<</SYS>> → Llama2Chat (CodeLlama handled by tokenizer name)
            if jinja.contains("[INST]")
                && jinja.contains("<<SYS>>")
                && jinja.contains("<</SYS>>")
            {
                tracing::debug!(
                    template = "Llama2Chat",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::Llama2Chat;
            }
            // Mistral [INST] signature (no <<SYS>>)
            if jinja.contains("[INST]") && jinja.contains("[/INST]") {
                tracing::debug!(
                    template = "MistralChat",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::MistralChat;
            }
            // Cohere Command signature
            if jinja.contains("<|START_OF_TURN_TOKEN|>") && jinja.contains("<|END_OF_TURN_TOKEN|>")
            {
                tracing::debug!(
                    template = "CohereCommand",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::CohereCommand;
            }
            // ChatGLM/GLM-4 signature (requires [gMASK])
            if jinja.contains("[gMASK]") {
                tracing::debug!(
                    template = "ChatGLMChat",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::ChatGLMChat;
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
                return Self::ZephyrChat;
            }
            // OLMo instruct signature (<|user|>/<|assistant|> without [gMASK])
            if jinja.contains("<|user|>") && jinja.contains("<|assistant|>") {
                tracing::debug!(
                    template = "OlmoInstruct",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::OlmoInstruct;
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
                return Self::AlpacaInstruct;
            }
            // Solar ### User:/### Assistant: markers
            if jinja.contains("### User:") && jinja.contains("### Assistant:") {
                tracing::debug!(
                    template = "SolarInstruct",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::SolarInstruct;
            }
            // MPT ### instruction markers
            if jinja.contains("### Instruction") && jinja.contains("### Response") {
                tracing::debug!(
                    template = "MptInstruct",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::MptInstruct;
            }
            // OpenChat GPT4 Correct signature
            if jinja.contains("GPT4 Correct") {
                tracing::debug!(
                    template = "OpenChat",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::OpenChat;
            }
            // WizardLM signature (USER:/ASSISTANT: with "A chat between")
            if jinja.contains("USER:")
                && jinja.contains("ASSISTANT:")
                && jinja.contains("A chat between")
            {
                tracing::debug!(
                    template = "WizardLM",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::WizardLM;
            }
            // Vicuna/ShareGPT signature (USER:/ASSISTANT:, not Falcon's "User:")
            if jinja.contains("USER:") && jinja.contains("ASSISTANT:") {
                tracing::debug!(
                    template = "VicunaChat",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::VicunaChat;
            }
            // Generic instruct template
            if jinja.contains("{% for message in messages %}") {
                tracing::debug!(
                    template = "Instruct",
                    source = "gguf_chat_template",
                    "auto-detected prompt template"
                );
                return Self::Instruct;
            }
        }

        // Priority 2: Tokenizer family name heuristics
        if let Some(name) = tokenizer_name {
            let lower = name.to_ascii_lowercase();
            if lower.contains("llama3") || lower.contains("llama-3") {
                tracing::debug!(
                    template = "Llama3Chat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::Llama3Chat;
            }
            if lower.contains("llama2") || lower.contains("llama-2") {
                tracing::debug!(
                    template = "Llama2Chat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::Llama2Chat;
            }
            if lower.contains("qwen") {
                tracing::debug!(
                    template = "QwenChat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::QwenChat;
            }
            if lower.contains("phi3") || lower.contains("phi-3") {
                tracing::debug!(
                    template = "Phi3Instruct",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::Phi3Instruct;
            }
            if lower.contains("phi") {
                tracing::debug!(
                    template = "Phi4Chat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::Phi4Chat;
            }
            if lower.contains("gemma2") || lower.contains("gemma-2-") {
                tracing::debug!(
                    template = "Gemma2Chat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::Gemma2Chat;
            }
            if lower.contains("gemma") {
                tracing::debug!(
                    template = "GemmaChat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::GemmaChat;
            }
            if lower.contains("mistral") {
                tracing::debug!(
                    template = "MistralChat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::MistralChat;
            }
            if lower.contains("deepseek") {
                tracing::debug!(
                    template = "DeepSeekChat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::DeepSeekChat;
            }
            if lower.contains("fim") || lower.contains("fill-in-middle") {
                tracing::debug!(
                    template = "FillInMiddle",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::FillInMiddle;
            }
            if lower.contains("starcoder") || lower.contains("star-coder") {
                tracing::debug!(
                    template = "StarCoder",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::StarCoder;
            }
            if lower.contains("falcon") {
                tracing::debug!(
                    template = "FalconChat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::FalconChat;
            }
            if lower.contains("codellama") || lower.contains("code-llama") {
                tracing::debug!(
                    template = "CodeLlamaInstruct",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::CodeLlamaInstruct;
            }
            if lower.contains("cohere") || lower.contains("command-r") {
                tracing::debug!(
                    template = "CommandRPlus",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::CommandRPlus;
            }
            if lower.contains("internlm") {
                tracing::debug!(
                    template = "InternLMChat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::InternLMChat;
            }
            if lower.contains("yi-") || lower.contains("yi_") || lower == "yi" {
                tracing::debug!(
                    template = "YiChat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::YiChat;
            }
            if lower.contains("baichuan") {
                tracing::debug!(
                    template = "BaichuanChat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::BaichuanChat;
            }
            if lower.contains("chatglm") || lower.contains("glm-4") || lower.contains("glm4") {
                tracing::debug!(
                    template = "ChatGLMChat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::ChatGLMChat;
            }
            if lower.contains("mpt") {
                tracing::debug!(
                    template = "MptInstruct",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::MptInstruct;
            }
            if lower.contains("rwkv") {
                tracing::debug!(
                    template = "RwkvWorld",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::RwkvWorld;
            }
            if lower.contains("olmo") {
                tracing::debug!(
                    template = "OlmoInstruct",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::OlmoInstruct;
            }
            if lower.contains("zephyr") {
                tracing::debug!(
                    template = "ZephyrChat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::ZephyrChat;
            }
            if lower.contains("vicuna") || lower.contains("sharegpt") {
                tracing::debug!(
                    template = "VicunaChat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::VicunaChat;
            }
            if lower.contains("orca") {
                tracing::debug!(
                    template = "OrcaChat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::OrcaChat;
            }
            if lower.contains("solar") {
                tracing::debug!(
                    template = "SolarInstruct",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::SolarInstruct;
            }
            if lower.contains("alpaca") {
                tracing::debug!(
                    template = "AlpacaInstruct",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::AlpacaInstruct;
            }
            if lower.contains("nous") || lower.contains("hermes") {
                tracing::debug!(
                    template = "NousHermes",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::NousHermes;
            }
            if lower.contains("wizard") {
                tracing::debug!(
                    template = "WizardLM",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::WizardLM;
            }
            if lower.contains("openchat") {
                tracing::debug!(
                    template = "OpenChat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::OpenChat;
            }
            if lower.contains("granite") {
                tracing::debug!(
                    template = "GraniteChat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::GraniteChat;
            }
            if lower.contains("nemotron") {
                tracing::debug!(
                    template = "NemotronChat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::NemotronChat;
            }
            if lower.contains("saiga") {
                tracing::debug!(
                    template = "SaigaChat",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::SaigaChat;
            }
            if lower.contains("instruct") {
                tracing::debug!(
                    template = "Instruct",
                    source = "tokenizer_name",
                    hint = name,
                    "auto-detected prompt template"
                );
                return Self::Instruct;
            }
        }

        // Priority 3: Fallback — no recognisable signature found
        tracing::warn!(template = "Raw", "no template signature found; falling back to Raw");
        Self::Raw
    }

    /// Apply the template to a user prompt
    pub fn apply(&self, user_text: &str, system_prompt: Option<&str>) -> String {
        match self {
            Self::Raw => user_text.to_string(),
            Self::Instruct => Self::apply_instruct(user_text, system_prompt),
            Self::Llama3Chat => Self::apply_llama3_chat(user_text, system_prompt),
            Self::Phi4Chat => Self::apply_phi4_chat(user_text, system_prompt),
            Self::QwenChat => Self::apply_qwen_chat(user_text, system_prompt),
            Self::GemmaChat => Self::apply_gemma_chat(user_text, system_prompt),
            Self::MistralChat => Self::apply_mistral_chat(user_text, system_prompt),
            Self::DeepSeekChat => Self::apply_deepseek_chat(user_text, system_prompt),
            Self::StarCoder => Self::apply_starcoder(user_text, system_prompt),
            Self::FalconChat => Self::apply_falcon_chat(user_text, system_prompt),
            Self::CodeLlamaInstruct => Self::apply_codellama_instruct(user_text, system_prompt),
            Self::CohereCommand => Self::apply_cohere_command(user_text, system_prompt),
            Self::InternLMChat => Self::apply_internlm_chat(user_text, system_prompt),
            Self::YiChat => Self::apply_yi_chat(user_text, system_prompt),
            Self::BaichuanChat => Self::apply_baichuan_chat(user_text, system_prompt),
            Self::ChatGLMChat => Self::apply_chatglm_chat(user_text, system_prompt),
            Self::MptInstruct => Self::apply_mpt_instruct(user_text, system_prompt),
            Self::RwkvWorld => Self::apply_rwkv_world(user_text, system_prompt),
            Self::OlmoInstruct => Self::apply_olmo_instruct(user_text, system_prompt),
            Self::FillInMiddle => Self::apply_fill_in_middle(user_text, system_prompt),
            Self::ZephyrChat => Self::apply_zephyr_chat(user_text, system_prompt),
            Self::VicunaChat => Self::apply_vicuna_chat(user_text, system_prompt),
            Self::OrcaChat => Self::apply_orca_chat(user_text, system_prompt),
            Self::SolarInstruct => Self::apply_solar_instruct(user_text, system_prompt),
            Self::AlpacaInstruct => Self::apply_alpaca_instruct(user_text, system_prompt),
            Self::CommandRPlus => Self::apply_command_r_plus(user_text, system_prompt),
            Self::NousHermes => Self::apply_nous_hermes(user_text, system_prompt),
            Self::WizardLM => Self::apply_wizard_lm(user_text, system_prompt),
            Self::OpenChat => Self::apply_openchat(user_text, system_prompt),
            Self::GraniteChat => Self::apply_granite_chat(user_text, system_prompt),
            Self::NemotronChat => Self::apply_nemotron_chat(user_text, system_prompt),
            Self::SaigaChat => Self::apply_saiga_chat(user_text, system_prompt),
            Self::Llama2Chat => Self::apply_llama2_chat(user_text, system_prompt),
            Self::Gemma2Chat => Self::apply_gemma2_chat(user_text, system_prompt),
            Self::Phi3Instruct => Self::apply_phi3_instruct(user_text, system_prompt),
        }
    }

    /// Apply simple instruct template
    fn apply_instruct(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();

        if let Some(system) = system_prompt {
            result.push_str("System: ");
            result.push_str(system);
            result.push_str("\n\n");
        }

        result.push_str("Q: ");
        result.push_str(user_text);
        result.push_str("\nA:");

        result
    }

    /// Apply LLaMA-3 chat template with proper special tokens
    ///
    /// Format:
    /// ```text
    /// <|begin_of_text|>
    /// [<|start_header_id|>system<|end_header_id|>
    /// {system_prompt}<|eot_id|>]
    /// <|start_header_id|>user<|end_header_id|>
    /// {user_text}<|eot_id|>
    /// <|start_header_id|>assistant<|end_header_id|>
    /// ```
    fn apply_llama3_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::from("<|begin_of_text|>");

        // Add system prompt if provided
        if let Some(system) = system_prompt {
            result.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
            result.push_str(system);
            result.push_str("<|eot_id|>");
        }

        // Add user message
        result.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
        result.push_str(user_text);
        result.push_str("<|eot_id|>");

        // Start assistant response
        result.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

        result
    }

    /// Apply Phi-4 ChatML template with im_start/im_end tokens
    ///
    /// Format:
    /// ```text
    /// <|im_start|>system
    /// You are a helpful assistant.<|im_end|>
    /// <|im_start|>user
    /// {user_text}<|im_end|>
    /// <|im_start|>assistant
    /// ```
    fn apply_phi4_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();

        // Add system prompt (default if not provided)
        let system = system_prompt.unwrap_or("You are a helpful assistant.");
        result.push_str("<|im_start|>system\n");
        result.push_str(system);
        result.push_str("<|im_end|>\n");

        // Add user message
        result.push_str("<|im_start|>user\n");
        result.push_str(user_text);
        result.push_str("<|im_end|>\n");

        // Start assistant response
        result.push_str("<|im_start|>assistant\n");

        result
    }

    /// Apply Qwen ChatML format (same structure as Phi-4 ChatML)
    ///
    /// ```text
    /// <|im_start|>system
    /// You are a helpful assistant.<|im_end|>
    /// <|im_start|>user
    /// {user_text}<|im_end|>
    /// <|im_start|>assistant
    /// ```
    fn apply_qwen_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();

        let system = system_prompt.unwrap_or("You are a helpful assistant.");
        result.push_str("<|im_start|>system\n");
        result.push_str(system);
        result.push_str("<|im_end|>\n");

        result.push_str("<|im_start|>user\n");
        result.push_str(user_text);
        result.push_str("<|im_end|>\n");

        result.push_str("<|im_start|>assistant\n");

        result
    }

    /// Apply Gemma chat template with start_of_turn/end_of_turn tokens
    ///
    /// Format:
    /// ```text
    /// <start_of_turn>user
    /// {user_text}<end_of_turn>
    /// <start_of_turn>model
    /// ```
    ///
    /// Gemma doesn't have a native system role; system messages are
    /// prepended to the user message.
    fn apply_gemma_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();

        // Gemma has no system role — prepend system text to user message
        result.push_str("<start_of_turn>user\n");
        if let Some(system) = system_prompt {
            result.push_str(system);
            result.push_str("\n\n");
        }
        result.push_str(user_text);
        result.push_str("<end_of_turn>\n");

        // Start model response
        result.push_str("<start_of_turn>model\n");

        result
    }

    /// Apply Mistral chat template with [INST]...[/INST] tokens
    ///
    /// Format:
    /// ```text
    /// <s>[INST] {user_text} [/INST]
    /// ```
    /// With system prompt:
    /// ```text
    /// <s>[INST] {system_prompt}
    ///
    /// {user_text} [/INST]
    /// ```
    fn apply_mistral_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::from("<s>[INST] ");

        if let Some(system) = system_prompt {
            result.push_str(system);
            result.push_str("\n\n");
        }

        result.push_str(user_text);
        result.push_str(" [/INST]");

        result
    }

    /// Apply DeepSeek ChatML format (same structure as Qwen/Phi-4 ChatML)
    ///
    /// ```text
    /// <|im_start|>system
    /// You are a helpful assistant.<|im_end|>
    /// <|im_start|>user
    /// {user_text}<|im_end|>
    /// <|im_start|>assistant
    /// ```
    fn apply_deepseek_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();

        let system = system_prompt.unwrap_or("You are a helpful assistant.");
        result.push_str("<|im_start|>system\n");
        result.push_str(system);
        result.push_str("<|im_end|>\n");

        result.push_str("<|im_start|>user\n");
        result.push_str(user_text);
        result.push_str("<|im_end|>\n");

        result.push_str("<|im_start|>assistant\n");

        result
    }

    /// Apply StarCoder code completion format
    ///
    /// StarCoder uses a simple completion format. If a system prompt is
    /// provided it is prepended as a comment.
    fn apply_starcoder(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();

        if let Some(system) = system_prompt {
            result.push_str("# ");
            result.push_str(system);
            result.push('\n');
        }

        result.push_str(user_text);
        result
    }

    /// Apply Falcon chat template with User:/Falcon: roles
    ///
    /// Format:
    /// ```text
    /// User: {user_text}
    /// Falcon:
    /// ```
    fn apply_falcon_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();

        if let Some(system) = system_prompt {
            result.push_str("System: ");
            result.push_str(system);
            result.push_str("\n\n");
        }

        result.push_str("User: ");
        result.push_str(user_text);
        result.push_str("\nFalcon:");

        result
    }

    /// Apply CodeLlama instruct template (LLaMA-style [INST] for code)
    ///
    /// Format:
    /// ```text
    /// [INST] {user_text} [/INST]
    /// ```
    fn apply_codellama_instruct(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::from("[INST] ");

        if let Some(system) = system_prompt {
            result.push_str("<<SYS>>\n");
            result.push_str(system);
            result.push_str("\n<</SYS>>\n\n");
        }

        result.push_str(user_text);
        result.push_str(" [/INST]");

        result
    }

    /// Apply Cohere Command format with START_OF_TURN/END_OF_TURN tokens
    ///
    /// Format:
    /// ```text
    /// <|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system}<|END_OF_TURN_TOKEN|>
    /// <|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user}<|END_OF_TURN_TOKEN|>
    /// <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
    /// ```
    fn apply_cohere_command(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();

        if let Some(system) = system_prompt {
            result.push_str("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>");
            result.push_str(system);
            result.push_str("<|END_OF_TURN_TOKEN|>");
        }

        result.push_str("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>");
        result.push_str(user_text);
        result.push_str("<|END_OF_TURN_TOKEN|>");

        result.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");

        result
    }

    /// Apply InternLM ChatML format (same structure as Phi-4 ChatML)
    ///
    /// ```text
    /// <|im_start|>system
    /// You are a helpful assistant.<|im_end|>
    /// <|im_start|>user
    /// {user_text}<|im_end|>
    /// <|im_start|>assistant
    /// ```
    fn apply_internlm_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();

        let system = system_prompt.unwrap_or("You are a helpful assistant.");
        result.push_str("<|im_start|>system\n");
        result.push_str(system);
        result.push_str("<|im_end|>\n");

        result.push_str("<|im_start|>user\n");
        result.push_str(user_text);
        result.push_str("<|im_end|>\n");

        result.push_str("<|im_start|>assistant\n");

        result
    }

    /// Apply Yi chat template (ChatML format)
    fn apply_yi_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        let system = system_prompt.unwrap_or("You are a helpful assistant.");
        result.push_str("<|im_start|>system\n");
        result.push_str(system);
        result.push_str("<|im_end|>\n");
        result.push_str("<|im_start|>user\n");
        result.push_str(user_text);
        result.push_str("<|im_end|>\n");
        result.push_str("<|im_start|>assistant\n");
        result
    }

    /// Apply Baichuan chat template
    ///
    /// Format: `<reserved_106>{user}<reserved_107>`
    fn apply_baichuan_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        if let Some(sys) = system_prompt {
            result.push_str("<reserved_106>");
            result.push_str(sys);
            result.push_str("<reserved_107>");
        }
        result.push_str("<reserved_106>");
        result.push_str(user_text);
        result.push_str("<reserved_107>");
        result
    }

    /// Apply ChatGLM/GLM-4 chat template
    ///
    /// Format: `[gMASK]<sop><|system|>\n{sys}<|user|>\n{user}<|assistant|>\n`
    fn apply_chatglm_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::from("[gMASK]<sop>");
        if let Some(sys) = system_prompt {
            result.push_str("<|system|>\n");
            result.push_str(sys);
        }
        result.push_str("<|user|>\n");
        result.push_str(user_text);
        result.push_str("<|assistant|>\n");
        result
    }

    /// Apply MPT instruct template
    ///
    /// Format: `### Instruction\n{text}\n\n### Response\n`
    fn apply_mpt_instruct(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        if let Some(sys) = system_prompt {
            result.push_str("### System\n");
            result.push_str(sys);
            result.push_str("\n\n");
        }
        result.push_str("### Instruction\n");
        result.push_str(user_text);
        result.push_str("\n\n### Response\n");
        result
    }

    /// Apply RWKV World template
    ///
    /// Format: `User: {text}\n\nAssistant:`
    fn apply_rwkv_world(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        if let Some(sys) = system_prompt {
            result.push_str("User: ");
            result.push_str(sys);
            result.push_str("\n\nAssistant: OK\n\n");
        }
        result.push_str("User: ");
        result.push_str(user_text);
        result.push_str("\n\nAssistant:");
        result
    }

    /// Apply OLMo instruct template
    ///
    /// Format: `<|user|>\n{text}\n<|assistant|>\n`
    fn apply_olmo_instruct(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        if let Some(sys) = system_prompt {
            result.push_str("<|system|>\n");
            result.push_str(sys);
            result.push('\n');
        }
        result.push_str("<|user|>\n");
        result.push_str(user_text);
        result.push_str("\n<|assistant|>\n");
        result
    }

    /// Apply fill-in-the-middle template for code infilling
    ///
    /// Format: `<fim_prefix>{user_text}<fim_suffix>{suffix_context}<fim_middle>`
    fn apply_fill_in_middle(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::from("<fim_prefix>");
        result.push_str(user_text);
        result.push_str("<fim_suffix>");
        if let Some(suffix) = system_prompt {
            result.push_str(suffix);
        }
        result.push_str("<fim_middle>");
        result
    }

    /// Apply Zephyr chat template with </s> delimiters
    ///
    /// Format:
    /// ```text
    /// <|system|>
    /// You are a helpful assistant.</s>
    /// <|user|>
    /// {user_text}</s>
    /// <|assistant|>
    /// ```
    fn apply_zephyr_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        let system = system_prompt.unwrap_or("You are a helpful assistant.");
        result.push_str("<|system|>\n");
        result.push_str(system);
        result.push_str("</s>\n");
        result.push_str("<|user|>\n");
        result.push_str(user_text);
        result.push_str("</s>\n");
        result.push_str("<|assistant|>\n");
        result
    }

    /// Apply Vicuna/ShareGPT chat template
    ///
    /// Format:
    /// ```text
    /// A chat between a curious user and an artificial intelligence assistant. ...
    ///
    /// USER: {user_text}
    /// ASSISTANT:
    /// ```
    fn apply_vicuna_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        let system = system_prompt.unwrap_or(
            "A chat between a curious user and an artificial intelligence \
             assistant. The assistant gives helpful, detailed, and polite \
             answers to the user's questions.",
        );
        result.push_str(system);
        result.push_str("\n\nUSER: ");
        result.push_str(user_text);
        result.push_str("\nASSISTANT:");
        result
    }

    /// Apply Orca ChatML template (ChatML variant with Orca default system prompt)
    fn apply_orca_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        let system = system_prompt.unwrap_or(
            "You are Orca, an AI language model created by Microsoft. You are \
             a cautious assistant. You carefully follow instructions.",
        );
        result.push_str("<|im_start|>system\n");
        result.push_str(system);
        result.push_str("<|im_end|>\n");
        result.push_str("<|im_start|>user\n");
        result.push_str(user_text);
        result.push_str("<|im_end|>\n");
        result.push_str("<|im_start|>assistant\n");
        result
    }

    /// Apply SOLAR instruct template
    ///
    /// Format: `### User:\n{text}\n\n### Assistant:\n`
    fn apply_solar_instruct(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        if let Some(sys) = system_prompt {
            result.push_str("### System:\n");
            result.push_str(sys);
            result.push_str("\n\n");
        }
        result.push_str("### User:\n");
        result.push_str(user_text);
        result.push_str("\n\n### Assistant:\n");
        result
    }

    /// Apply Stanford Alpaca instruct template
    ///
    /// Format: `Below is an instruction ...\n\n### Instruction:\n{text}\n\n### Response:\n`
    fn apply_alpaca_instruct(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        let system = system_prompt.unwrap_or(
            "Below is an instruction that describes a task. Write a response \
             that appropriately completes the request.",
        );
        result.push_str(system);
        result.push_str("\n\n### Instruction:\n");
        result.push_str(user_text);
        result.push_str("\n\n### Response:\n");
        result
    }

    /// Apply Command-R+ format with START_OF_TURN_TOKEN markers
    fn apply_command_r_plus(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        let system = system_prompt.unwrap_or(
            "You are Command-R+, a large language model trained to have \
             polite, helpful, inclusive conversations with people.",
        );
        result.push_str("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>");
        result.push_str(system);
        result.push_str("<|END_OF_TURN_TOKEN|>\n");
        result.push_str("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>");
        result.push_str(user_text);
        result.push_str("<|END_OF_TURN_TOKEN|>\n");
        result.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
        result
    }

    /// Apply NousHermes ChatML variant
    fn apply_nous_hermes(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        let system =
            system_prompt.unwrap_or("You are a helpful, honest and harmless AI assistant.");
        result.push_str("<|im_start|>system\n");
        result.push_str(system);
        result.push_str("<|im_end|>\n");
        result.push_str("<|im_start|>user\n");
        result.push_str(user_text);
        result.push_str("<|im_end|>\n");
        result.push_str("<|im_start|>assistant\n");
        result
    }

    /// Apply WizardLM Vicuna-derived format
    fn apply_wizard_lm(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        let system = system_prompt.unwrap_or(
            "A chat between a curious user and an artificial intelligence \
             assistant. The assistant gives helpful, detailed, and polite \
             answers to the user's questions.",
        );
        result.push_str(system);
        result.push_str("\n\nUSER: ");
        result.push_str(user_text);
        result.push_str("\nASSISTANT: ");
        result
    }

    /// Apply OpenChat GPT4 Correct format
    fn apply_openchat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        result.push_str("GPT4 Correct User: ");
        if let Some(sys) = system_prompt {
            result.push_str(sys);
            result.push_str("\n\n");
        }
        result.push_str(user_text);
        result.push_str("<|end_of_turn|>GPT4 Correct Assistant:");
        result
    }

    /// Apply IBM Granite chat format with start_of_role/end_of_role tokens
    fn apply_granite_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        let system = system_prompt
            .unwrap_or("You are Granite, an AI language model developed by IBM.");
        result.push_str("<|start_of_role|>system<|end_of_role|>");
        result.push_str(system);
        result.push('\n');
        result.push_str("<|start_of_role|>user<|end_of_role|>");
        result.push_str(user_text);
        result.push('\n');
        result.push_str("<|start_of_role|>assistant<|end_of_role|>");
        result
    }

    /// Apply NVIDIA Nemotron chat format with extra_id tokens
    fn apply_nemotron_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        let system = system_prompt
            .unwrap_or("You are a helpful, respectful and honest assistant.");
        result.push_str("<extra_id_0>System\n");
        result.push_str(system);
        result.push('\n');
        result.push_str("<extra_id_1>User\n");
        result.push_str(user_text);
        result.push('\n');
        result.push_str("<extra_id_1>Assistant\n");
        result
    }

    /// Apply Saiga/YandexGPT ChatML variant with Russian default system prompt
    fn apply_saiga_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        let system = system_prompt.unwrap_or(
            "Ты — Сайга, русскоязычный автоматический ассистент. \
             Ты разговариваешь с людьми и помогаешь им.",
        );
        result.push_str("<|im_start|>system\n");
        result.push_str(system);
        result.push_str("<|im_end|>\n");
        result.push_str("<|im_start|>user\n");
        result.push_str(user_text);
        result.push_str("<|im_end|>\n");
        result.push_str("<|im_start|>assistant\n");
        result
    }

    /// Apply Meta Llama-2 chat format with [INST]<<SYS>>/<</SYS>> markers
    fn apply_llama2_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::from("[INST] ");
        let system = system_prompt
            .unwrap_or("You are a helpful, respectful and honest assistant.");
        result.push_str("<<SYS>>\n");
        result.push_str(system);
        result.push_str("\n<</SYS>>\n\n");
        result.push_str(user_text);
        result.push_str(" [/INST] ");
        result
    }

    /// Apply Google Gemma 2 chat format (same format as Gemma)
    fn apply_gemma2_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        // Gemma 2 uses the same format as Gemma
        Self::apply_gemma_chat(user_text, system_prompt)
    }

    /// Apply Microsoft Phi-3 instruct format
    fn apply_phi3_instruct(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();
        let system = system_prompt.unwrap_or("You are a helpful AI assistant.");
        result.push_str("<|system|>\n");
        result.push_str(system);
        result.push_str("<|end|>\n");
        result.push_str("<|user|>\n");
        result.push_str(user_text);
        result.push_str("<|end|>\n");
        result.push_str("<|assistant|>\n");
        result
    }

    pub fn default_stop_sequences(&self) -> Vec<String> {
        match self {
            Self::Raw => vec![],
            Self::Instruct => vec!["\n\nQ:".to_string(), "\n\nHuman:".to_string()],
            Self::Llama3Chat => vec!["<|eot_id|>".to_string(), "<|end_of_text|>".to_string()],
            Self::Phi4Chat => vec!["<|im_end|>".to_string(), "<|endoftext|>".to_string()],
            Self::QwenChat => {
                vec!["<|im_end|>".to_string(), "<|endoftext|>".to_string()]
            }
            Self::GemmaChat => vec!["<end_of_turn>".to_string()],
            Self::MistralChat => vec!["</s>".to_string()],
            Self::DeepSeekChat => {
                vec!["<|im_end|>".to_string(), "<|end▁of▁sentence|>".to_string()]
            }
            Self::StarCoder => {
                vec!["<|endoftext|>".to_string()]
            }
            Self::FalconChat => {
                vec!["\nUser:".to_string(), "<|endoftext|>".to_string()]
            }
            Self::CodeLlamaInstruct => {
                vec!["</s>".to_string()]
            }
            Self::CohereCommand => {
                vec!["<|END_OF_TURN_TOKEN|>".to_string()]
            }
            Self::InternLMChat => {
                vec!["<|im_end|>".to_string(), "<eoa>".to_string()]
            }
            Self::YiChat => {
                vec!["<|im_end|>".to_string(), "<|endoftext|>".to_string()]
            }
            Self::BaichuanChat => {
                vec!["</s>".to_string()]
            }
            Self::ChatGLMChat => {
                vec!["<|user|>".to_string(), "<|observation|>".to_string()]
            }
            Self::MptInstruct => {
                vec!["### Instruction".to_string(), "<|endoftext|>".to_string()]
            }
            Self::RwkvWorld => {
                vec!["\nUser:".to_string(), "\n\n".to_string()]
            }
            Self::OlmoInstruct => {
                vec!["<|endoftext|>".to_string(), "<|user|>".to_string()]
            }
            Self::FillInMiddle => {
                vec![
                    "<fim_suffix>".to_string(),
                    "<|endoftext|>".to_string(),
                    "<fim_pad>".to_string(),
                ]
            }
            Self::ZephyrChat => {
                vec!["</s>".to_string(), "<|user|>".to_string()]
            }
            Self::VicunaChat => {
                vec!["USER:".to_string(), "</s>".to_string()]
            }
            Self::OrcaChat => {
                vec!["<|im_end|>".to_string(), "<|im_start|>".to_string()]
            }
            Self::SolarInstruct => {
                vec!["### User:".to_string(), "</s>".to_string()]
            }
            Self::AlpacaInstruct => {
                vec!["### Instruction:".to_string(), "</s>".to_string()]
            }
            Self::CommandRPlus => {
                vec!["<|END_OF_TURN_TOKEN|>".to_string()]
            }
            Self::NousHermes => {
                vec!["<|im_end|>".to_string(), "<|im_start|>".to_string()]
            }
            Self::WizardLM => {
                vec!["USER:".to_string(), "</s>".to_string()]
            }
            Self::OpenChat => {
                vec!["<|end_of_turn|>".to_string()]
            }
            Self::GraniteChat => {
                vec!["<|end_of_role|>".to_string(), "<|end_of_text|>".to_string()]
            }
            Self::NemotronChat => {
                vec!["<extra_id_1>".to_string(), "</s>".to_string()]
            }
            Self::SaigaChat => {
                vec!["<|im_end|>".to_string(), "<|im_start|>".to_string()]
            }
            Self::Llama2Chat => {
                vec!["</s>".to_string()]
            }
            Self::Gemma2Chat => {
                vec!["<end_of_turn>".to_string(), "<start_of_turn>".to_string()]
            }
            Self::Phi3Instruct => {
                vec!["<|end|>".to_string(), "<|endoftext|>".to_string()]
            }
        }
    }

    /// Resolve stop sequences to token IDs using the provided tokenizer
    ///
    /// This method converts the template's default stop sequences (like "<|eot_id|>")
    /// to their corresponding token IDs for efficient stop detection during generation.
    ///
    /// Token ID-based stops are checked before string matching, making termination
    /// faster and more reliable for models with special stop tokens.
    ///
    /// # Arguments
    /// * `tokenizer` - The tokenizer to use for token ID resolution
    ///
    /// # Returns
    /// A vector of token IDs that should trigger generation stop.
    /// Returns empty if no stop sequences can be resolved or if the template has no stops.
    ///
    /// # Example
    /// ```ignore
    /// let template = TemplateType::Llama3Chat;
    /// let stop_ids = template.resolve_stop_token_ids(&tokenizer);
    /// // stop_ids might contain [128009] for <|eot_id|>
    /// ```
    pub fn resolve_stop_token_ids(&self, tokenizer: &dyn bitnet_tokenizers::Tokenizer) -> Vec<u32> {
        let stop_sequences = self.default_stop_sequences();
        let mut stop_ids = Vec::new();

        for seq in &stop_sequences {
            if let Some(id) = tokenizer.token_to_id(seq) {
                stop_ids.push(id);
            }
        }

        stop_ids
    }

    /// Check if BOS should be added for this template
    /// LLaMA-3 chat includes its own BOS token in the template
    pub fn should_add_bos(&self) -> bool {
        match self {
            Self::Raw | Self::Instruct => true,
            Self::Llama3Chat => false, // Template includes <|begin_of_text|>
            Self::Phi4Chat => false,   // ChatML uses im_start/im_end tokens
            Self::QwenChat => false,   // ChatML uses im_start/im_end tokens
            Self::GemmaChat => false,  // Uses start_of_turn/end_of_turn tokens
            Self::MistralChat => false, // Template includes <s>
            Self::DeepSeekChat => false, // ChatML uses im_start/im_end tokens
            Self::StarCoder => true,   // Simple completion, BOS helpful
            Self::FalconChat => true,  // Simple User:/Falcon: format
            Self::CodeLlamaInstruct => false, // [INST] format with own markers
            Self::CohereCommand => false, // Uses START_OF_TURN tokens
            Self::InternLMChat => false, // ChatML uses im_start/im_end tokens
            Self::YiChat => false,     // ChatML uses im_start/im_end tokens
            Self::BaichuanChat => false, // Uses reserved tokens
            Self::ChatGLMChat => false, // Uses [gMASK]<sop> tokens
            Self::MptInstruct => true, // Simple ### markers, BOS helpful
            Self::RwkvWorld => true,   // Simple User:/Assistant: format
            Self::OlmoInstruct => false, // Uses <|user|>/<|assistant|> tokens
            Self::FillInMiddle => false, // Uses <fim_prefix>/<fim_middle> tokens
            Self::ZephyrChat => false,   // Uses <|system|>/<|user|> tokens
            Self::VicunaChat => true,    // Simple USER:/ASSISTANT: format
            Self::OrcaChat => false,     // ChatML uses im_start/im_end tokens
            Self::SolarInstruct => true, // Simple ### markers, BOS helpful
            Self::AlpacaInstruct => true, // Simple ### markers, BOS helpful
            Self::CommandRPlus => true,   // Has <BOS_TOKEN> but we handle separately
            Self::NousHermes => false,    // ChatML uses im_start/im_end tokens
            Self::WizardLM => true,       // Simple USER:/ASSISTANT: format
            Self::OpenChat => true,       // Simple GPT4 Correct format
            Self::GraniteChat => false,   // Uses start_of_role tokens
            Self::NemotronChat => false,  // Uses extra_id tokens
            Self::SaigaChat => false,     // ChatML uses im_start/im_end tokens
            Self::Llama2Chat => true,     // Llama-2 benefits from BOS
            Self::Gemma2Chat => true,     // Like Gemma, benefits from BOS
            Self::Phi3Instruct => false,  // Uses <|system|>/<|user|> tokens
        }
    }

    /// Check if special tokens should be parsed during encoding
    /// LLaMA-3 chat templates contain special tokens that need to be parsed
    pub fn parse_special(&self) -> bool {
        matches!(
            self,
            Self::Llama3Chat
                | Self::Phi4Chat
                | Self::QwenChat
                | Self::GemmaChat
                | Self::MistralChat
                | Self::DeepSeekChat
                | Self::StarCoder
                | Self::CodeLlamaInstruct
                | Self::CohereCommand
                | Self::InternLMChat
                | Self::YiChat
                | Self::BaichuanChat
                | Self::ChatGLMChat
                | Self::OlmoInstruct
                | Self::FillInMiddle
                | Self::ZephyrChat
                | Self::OrcaChat
                | Self::CommandRPlus
                | Self::NousHermes
                | Self::OpenChat
                | Self::GraniteChat
                | Self::NemotronChat
                | Self::SaigaChat
                | Self::Gemma2Chat
                | Self::Phi3Instruct
        )
    }

    /// Render a chat history (system + turns) into a single prompt string.
    /// This method formats multi-turn conversations with proper role markers.
    pub fn render_chat(&self, history: &[ChatTurn], system: Option<&str>) -> Result<String> {
        use std::fmt::Write as _;
        let mut out = String::new();

        match self {
            TemplateType::Llama3Chat => {
                // LLaMA-3 chat format with special tokens
                out.push_str("<|begin_of_text|>");

                // System prompt if provided
                if let Some(sys) = system {
                    write!(out, "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>", sys)?;
                }

                // Render conversation history
                for turn in history {
                    let role = turn.role.as_str();
                    write!(
                        out,
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        role, turn.text
                    )?;
                }

                // Start assistant response
                write!(out, "<|start_header_id|>assistant<|end_header_id|>\n\n")?;
            }
            TemplateType::Phi4Chat => {
                // ChatML format with im_start/im_end tokens
                let sys = system.unwrap_or("You are a helpful assistant.");
                writeln!(out, "<|im_start|>system\n{}<|im_end|>", sys)?;

                // Render conversation history
                for turn in history {
                    let role = turn.role.as_str();
                    writeln!(out, "<|im_start|>{}\n{}<|im_end|>", role, turn.text)?;
                }

                // Start assistant response
                writeln!(out, "<|im_start|>assistant")?;
            }
            TemplateType::QwenChat => {
                // ChatML format with im_start/im_end tokens
                let sys = system.unwrap_or("You are a helpful assistant.");
                writeln!(out, "<|im_start|>system\n{}<|im_end|>", sys)?;

                for turn in history {
                    let role = turn.role.as_str();
                    writeln!(out, "<|im_start|>{}\n{}<|im_end|>", role, turn.text)?;
                }

                writeln!(out, "<|im_start|>assistant")?;
            }
            TemplateType::GemmaChat => {
                // Gemma format with start_of_turn/end_of_turn tokens
                // Gemma has no system role — prepend to first user turn
                let mut system_prepended = false;

                for turn in history {
                    let role = match turn.role {
                        ChatRole::User => "user",
                        ChatRole::Assistant => "model",
                        ChatRole::System => continue,
                    };
                    writeln!(out, "<start_of_turn>{}", role)?;
                    if role == "user" && !system_prepended {
                        if let Some(sys) = system {
                            writeln!(out, "{}\n", sys)?;
                        }
                        system_prepended = true;
                    }
                    writeln!(out, "{}<end_of_turn>", turn.text)?;
                }

                // If no user turn was seen, still emit system prompt
                if !system_prepended && let Some(sys) = system {
                    writeln!(out, "<start_of_turn>user\n{}<end_of_turn>", sys)?;
                }

                // Start model response
                writeln!(out, "<start_of_turn>model")?;
            }
            TemplateType::MistralChat => {
                // Mistral [INST]...[/INST] format
                out.push_str("<s>");

                // Render prior turns
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            write!(out, "[INST] {} [/INST]", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            write!(out, "{}</s>", turn.text)?;
                        }
                        ChatRole::System => {}
                    }
                }

                // Final user turn with optional system prefix
                if let Some(sys) = system {
                    write!(out, "[INST] {}\n\n", sys)?;
                } else {
                    write!(out, "[INST] ")?;
                }
            }
            TemplateType::DeepSeekChat => {
                // ChatML format with im_start/im_end tokens
                let sys = system.unwrap_or("You are a helpful assistant.");
                writeln!(out, "<|im_start|>system\n{}<|im_end|>", sys)?;

                for turn in history {
                    let role = turn.role.as_str();
                    writeln!(out, "<|im_start|>{}\n{}<|im_end|>", role, turn.text)?;
                }

                writeln!(out, "<|im_start|>assistant")?;
            }
            TemplateType::Instruct => {
                // Simple Q&A format
                if let Some(sys) = system {
                    writeln!(out, "System: {}\n", sys)?;
                }

                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            writeln!(out, "Q: {}", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            writeln!(out, "A: {}", turn.text)?;
                        }
                        ChatRole::System => {
                            // System messages already emitted above
                        }
                    }
                }

                // Prompt for assistant response
                write!(out, "A: ")?;
            }
            TemplateType::Raw => {
                // Minimal: concatenate system prompt and full history
                if let Some(sys) = system {
                    writeln!(out, "{}\n", sys)?;
                }

                // Concatenate all turns with double newline separators
                for (i, turn) in history.iter().enumerate() {
                    if i > 0 {
                        write!(out, "\n\n")?;
                    }
                    write!(out, "{}", turn.text)?;
                }
            }
            TemplateType::StarCoder => {
                // Code completion: system as comment, code only
                if let Some(sys) = system {
                    writeln!(out, "# {}", sys)?;
                }

                for turn in history {
                    write!(out, "{}", turn.text)?;
                }
            }
            TemplateType::FalconChat => {
                // Falcon User:/Falcon: format
                if let Some(sys) = system {
                    writeln!(out, "System: {}\n", sys)?;
                }

                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            writeln!(out, "User: {}", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            writeln!(out, "Falcon: {}", turn.text)?;
                        }
                        ChatRole::System => {}
                    }
                }

                write!(out, "Falcon:")?;
            }
            TemplateType::CodeLlamaInstruct => {
                // CodeLlama [INST]...[/INST] with <<SYS>>
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            write!(out, "[INST] ")?;
                            if let Some(sys) = system {
                                write!(out, "<<SYS>>\n{}\n<</SYS>>\n\n", sys)?;
                            }
                            write!(out, "{} [/INST]", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            write!(out, " {} ", turn.text)?;
                        }
                        ChatRole::System => {}
                    }
                }
            }
            TemplateType::CohereCommand => {
                // Cohere Command format
                if let Some(sys) = system {
                    write!(
                        out,
                        "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\
                         {}<|END_OF_TURN_TOKEN|>",
                        sys
                    )?;
                }

                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            write!(
                                out,
                                "<|START_OF_TURN_TOKEN|>\
                                 <|USER_TOKEN|>{}\
                                 <|END_OF_TURN_TOKEN|>",
                                turn.text
                            )?;
                        }
                        ChatRole::Assistant => {
                            write!(
                                out,
                                "<|START_OF_TURN_TOKEN|>\
                                 <|CHATBOT_TOKEN|>{}\
                                 <|END_OF_TURN_TOKEN|>",
                                turn.text
                            )?;
                        }
                        ChatRole::System => {}
                    }
                }

                write!(out, "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")?;
            }
            TemplateType::InternLMChat => {
                // ChatML format with im_start/im_end tokens
                let sys = system.unwrap_or("You are a helpful assistant.");
                writeln!(out, "<|im_start|>system\n{}<|im_end|>", sys)?;

                for turn in history {
                    let role = turn.role.as_str();
                    writeln!(out, "<|im_start|>{}\n{}<|im_end|>", role, turn.text)?;
                }

                writeln!(out, "<|im_start|>assistant")?;
            }
            TemplateType::YiChat => {
                // Yi ChatML format (same as Phi4/Qwen)
                let sys = system.unwrap_or("You are a helpful assistant.");
                writeln!(out, "<|im_start|>system\n{}<|im_end|>", sys)?;
                for turn in history {
                    let role = turn.role.as_str();
                    writeln!(out, "<|im_start|>{}\n{}<|im_end|>", role, turn.text)?;
                }
                writeln!(out, "<|im_start|>assistant")?;
            }
            TemplateType::BaichuanChat => {
                // Baichuan reserved token format
                if let Some(sys) = system {
                    write!(out, "<reserved_106>{}<reserved_107>", sys)?;
                }
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            write!(out, "<reserved_106>{}", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            write!(out, "<reserved_107>{}", turn.text)?;
                        }
                        ChatRole::System => {}
                    }
                }
                write!(out, "<reserved_107>")?;
            }
            TemplateType::ChatGLMChat => {
                // ChatGLM/GLM-4 format
                write!(out, "[gMASK]<sop>")?;
                if let Some(sys) = system {
                    write!(out, "<|system|>\n{}", sys)?;
                }
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            write!(out, "<|user|>\n{}", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            write!(out, "<|assistant|>\n{}", turn.text)?;
                        }
                        ChatRole::System => {}
                    }
                }
                writeln!(out, "<|assistant|>")?;
            }
            TemplateType::MptInstruct => {
                // MPT ### marker format
                if let Some(sys) = system {
                    writeln!(out, "### System\n{}\n", sys)?;
                }
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            writeln!(out, "### Instruction\n{}\n", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            writeln!(out, "### Response\n{}\n", turn.text)?;
                        }
                        ChatRole::System => {}
                    }
                }
                writeln!(out, "### Response")?;
            }
            TemplateType::RwkvWorld => {
                // RWKV World User:/Assistant: format
                if let Some(sys) = system {
                    write!(out, "User: {}\n\nAssistant: OK\n\n", sys)?;
                }
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            write!(out, "User: {}\n\n", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            write!(out, "Assistant: {}\n\n", turn.text)?;
                        }
                        ChatRole::System => {}
                    }
                }
                write!(out, "Assistant:")?;
            }
            TemplateType::OlmoInstruct => {
                // OLMo <|user|>/<|assistant|> format
                if let Some(sys) = system {
                    write!(out, "<|system|>\n{}\n", sys)?;
                }
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            write!(out, "<|user|>\n{}\n", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            write!(out, "<|assistant|>\n{}\n", turn.text)?;
                        }
                        ChatRole::System => {}
                    }
                }
                writeln!(out, "<|assistant|>")?;
            }
            TemplateType::FillInMiddle => {
                // FIM: just format the last user message as FIM, ignore assistant turns
                let last_user = history
                    .iter()
                    .rev()
                    .find(|t| t.role == ChatRole::User)
                    .map(|t| t.text.as_str())
                    .unwrap_or("");
                write!(out, "<fim_prefix>{}<fim_suffix>", last_user)?;
                if let Some(sys) = system {
                    write!(out, "{}", sys)?;
                }
                write!(out, "<fim_middle>")?;
            }
            TemplateType::ZephyrChat => {
                // Zephyr format with </s> delimiters
                let sys = system.unwrap_or("You are a helpful assistant.");
                write!(out, "<|system|>\n{}</s>\n", sys)?;
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            write!(out, "<|user|>\n{}</s>\n", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            write!(out, "<|assistant|>\n{}</s>\n", turn.text)?;
                        }
                        ChatRole::System => {}
                    }
                }
                writeln!(out, "<|assistant|>")?;
            }
            TemplateType::VicunaChat => {
                // Vicuna/ShareGPT USER:/ASSISTANT: format
                let sys = system.unwrap_or(
                    "A chat between a curious user and an artificial intelligence \
                     assistant. The assistant gives helpful, detailed, and polite \
                     answers to the user's questions.",
                );
                writeln!(out, "{}\n", sys)?;
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            writeln!(out, "USER: {}", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            writeln!(out, "ASSISTANT: {}", turn.text)?;
                        }
                        ChatRole::System => {}
                    }
                }
                write!(out, "ASSISTANT:")?;
            }
            TemplateType::OrcaChat => {
                // Orca ChatML format (same structure as Phi4Chat)
                let sys = system.unwrap_or(
                    "You are Orca, an AI language model created by Microsoft. You are \
                     a cautious assistant. You carefully follow instructions.",
                );
                writeln!(out, "<|im_start|>system\n{}<|im_end|>", sys)?;
                for turn in history {
                    let role = turn.role.as_str();
                    writeln!(out, "<|im_start|>{}\n{}<|im_end|>", role, turn.text)?;
                }
                writeln!(out, "<|im_start|>assistant")?;
            }
            TemplateType::SolarInstruct => {
                // Solar ### User:/### Assistant: format
                if let Some(sys) = system {
                    write!(out, "### System:\n{}\n\n", sys)?;
                }
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            write!(out, "### User:\n{}\n\n", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            write!(out, "### Assistant:\n{}\n\n", turn.text)?;
                        }
                        ChatRole::System => {}
                    }
                }
                writeln!(out, "### Assistant:")?;
            }
            TemplateType::AlpacaInstruct => {
                // Alpaca ### Instruction:/### Response: format
                let sys = system.unwrap_or(
                    "Below is an instruction that describes a task. Write a response \
                     that appropriately completes the request.",
                );
                writeln!(out, "{}\n", sys)?;
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            write!(out, "### Instruction:\n{}\n\n", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            write!(out, "### Response:\n{}\n\n", turn.text)?;
                        }
                        ChatRole::System => {}
                    }
                }
                writeln!(out, "### Response:")?;
            }
            TemplateType::CommandRPlus => {
                // Command-R+ format with turn tokens
                let sys = system.unwrap_or(
                    "You are Command-R+, a large language model trained to have \
                     polite, helpful, inclusive conversations with people.",
                );
                writeln!(
                    out,
                    "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\
                     {}<|END_OF_TURN_TOKEN|>",
                    sys
                )?;
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            writeln!(
                                out,
                                "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\
                                 {}<|END_OF_TURN_TOKEN|>",
                                turn.text
                            )?;
                        }
                        ChatRole::Assistant => {
                            writeln!(
                                out,
                                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\
                                 {}<|END_OF_TURN_TOKEN|>",
                                turn.text
                            )?;
                        }
                        ChatRole::System => {}
                    }
                }
                write!(out, "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")?;
            }
            TemplateType::NousHermes => {
                // NousHermes ChatML variant
                let sys = system
                    .unwrap_or("You are a helpful, honest and harmless AI assistant.");
                writeln!(out, "<|im_start|>system\n{}<|im_end|>", sys)?;
                for turn in history {
                    let role = turn.role.as_str();
                    writeln!(out, "<|im_start|>{}\n{}<|im_end|>", role, turn.text)?;
                }
                writeln!(out, "<|im_start|>assistant")?;
            }
            TemplateType::WizardLM => {
                // WizardLM USER:/ASSISTANT: format
                let sys = system.unwrap_or(
                    "A chat between a curious user and an artificial intelligence \
                     assistant. The assistant gives helpful, detailed, and polite \
                     answers to the user's questions.",
                );
                writeln!(out, "{}\n", sys)?;
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            writeln!(out, "USER: {}", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            writeln!(out, "ASSISTANT: {}", turn.text)?;
                        }
                        ChatRole::System => {}
                    }
                }
                write!(out, "ASSISTANT: ")?;
            }
            TemplateType::OpenChat=> {
                // OpenChat GPT4 Correct format
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            write!(
                                out,
                                "GPT4 Correct User: {}<|end_of_turn|>",
                                turn.text
                            )?;
                        }
                        ChatRole::Assistant => {
                            write!(
                                out,
                                "GPT4 Correct Assistant: {}<|end_of_turn|>",
                                turn.text
                            )?;
                        }
                        ChatRole::System => {}
                    }
                }
                write!(out, "GPT4 Correct Assistant:")?;
            }
            TemplateType::GraniteChat => {
                // Granite role-token format
                let sys = system
                    .unwrap_or("You are Granite, an AI language model developed by IBM.");
                writeln!(out, "<|start_of_role|>system<|end_of_role|>{}", sys)?;
                for turn in history {
                    let role = turn.role.as_str();
                    writeln!(out, "<|start_of_role|>{}<|end_of_role|>{}", role, turn.text)?;
                }
                write!(out, "<|start_of_role|>assistant<|end_of_role|>")?;
            }
            TemplateType::NemotronChat => {
                // Nemotron extra_id format
                let sys = system
                    .unwrap_or("You are a helpful, respectful and honest assistant.");
                write!(out, "<extra_id_0>System\n{}\n", sys)?;
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            write!(out, "<extra_id_1>User\n{}\n", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            write!(out, "<extra_id_1>Assistant\n{}\n", turn.text)?;
                        }
                        ChatRole::System => {}
                    }
                }
                writeln!(out, "<extra_id_1>Assistant")?;
            }
            TemplateType::SaigaChat => {
                // Saiga ChatML variant
                let sys = system.unwrap_or(
                    "Ты — Сайга, русскоязычный автоматический ассистент. \
                     Ты разговариваешь с людьми и помогаешь им.",
                );
                writeln!(out, "<|im_start|>system\n{}<|im_end|>", sys)?;
                for turn in history {
                    let role = turn.role.as_str();
                    writeln!(out, "<|im_start|>{}\n{}<|im_end|>", role, turn.text)?;
                }
                writeln!(out, "<|im_start|>assistant")?;
            }
            TemplateType::Llama2Chat => {
                // Llama-2 [INST]<<SYS>>/<</SYS>> format
                let sys = system
                    .unwrap_or("You are a helpful, respectful and honest assistant.");
                let mut first_user = true;
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            if first_user {
                                write!(
                                    out,
                                    "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST] ",
                                    sys, turn.text
                                )?;
                                first_user = false;
                            } else {
                                write!(out, "<s>[INST] {} [/INST] ", turn.text)?;
                            }
                        }
                        ChatRole::Assistant => {
                            write!(out, "{} </s>", turn.text)?;
                        }
                        ChatRole::System => {}
                    }
                }
                if first_user {
                    write!(
                        out,
                        "[INST] <<SYS>>\n{}\n<</SYS>>\n\n [/INST] ",
                        sys
                    )?;
                }
            }
            TemplateType::Gemma2Chat => {
                // Gemma 2 format (identical to GemmaChat)
                let mut system_prepended = false;

                for turn in history {
                    let role = match turn.role {
                        ChatRole::User => "user",
                        ChatRole::Assistant => "model",
                        ChatRole::System => continue,
                    };
                    writeln!(out, "<start_of_turn>{}", role)?;
                    if role == "user" && !system_prepended {
                        if let Some(sys) = system {
                            writeln!(out, "{}\n", sys)?;
                        }
                        system_prepended = true;
                    }
                    writeln!(out, "{}<end_of_turn>", turn.text)?;
                }

                if !system_prepended && let Some(sys) = system {
                    writeln!(out, "<start_of_turn>user\n{}<end_of_turn>", sys)?;
                }

                writeln!(out, "<start_of_turn>model")?;
            }
            TemplateType::Phi3Instruct => {
                // Phi-3 <|system|>/<|user|>/<|assistant|>/<|end|> format
                let sys = system.unwrap_or("You are a helpful AI assistant.");
                write!(out, "<|system|>\n{}<|end|>\n", sys)?;
                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            write!(out, "<|user|>\n{}<|end|>\n", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            write!(out, "<|assistant|>\n{}<|end|>\n", turn.text)?;
                        }
                        ChatRole::System => {}
                    }
                }
                writeln!(out, "<|assistant|>")?;
            }
        }

        Ok(out)
    }

    /// Validate that template output meets basic quality constraints.
    ///
    /// Checks:
    /// - Output is non-empty
    /// - Output contains the user text (unless Raw with empty input)
    /// - Stop sequences don't appear in the middle of the output
    pub fn validate_output(&self, output: &str, user_text: &str) -> TemplateValidation {
        let mut warnings = Vec::new();

        if output.is_empty() {
            warnings.push("Template produced empty output".to_string());
        }

        if !user_text.is_empty() && !output.contains(user_text) {
            warnings.push(format!(
                "Output does not contain user text: {:?}",
                &user_text[..user_text.len().min(50)]
            ));
        }

        // Check if any stop sequence appears beyond the template's structural usage
        let structural = self.apply("", None);
        for stop in self.default_stop_sequences() {
            let structural_count = structural.matches(&stop).count();
            let output_count = output.matches(&stop).count();
            if output_count > structural_count {
                warnings.push(format!(
                    "Stop sequence {:?} found {} extra time(s) beyond template structure",
                    stop, output_count - structural_count
                ));
            }
        }

        TemplateValidation {
            is_valid: warnings.is_empty(),
            warnings,
        }
    }

    /// Returns a human-readable summary of this template type's configuration.
    pub fn info(&self) -> TemplateInfo {
        TemplateInfo {
            name: self.to_string(),
            stop_sequences: self.default_stop_sequences(),
            adds_bos: self.should_add_bos(),
            parses_special: self.parse_special(),
        }
    }
}

/// Validation result for template output.
#[derive(Debug, Clone)]
pub struct TemplateValidation {
    /// Whether the output passes all checks.
    pub is_valid: bool,
    /// List of warnings (empty if valid).
    pub warnings: Vec<String>,
}

/// Summary information about a template type.
#[derive(Debug, Clone)]
pub struct TemplateInfo {
    /// Display name of the template.
    pub name: String,
    /// Default stop sequences.
    pub stop_sequences: Vec<String>,
    /// Whether BOS token should be added.
    pub adds_bos: bool,
    /// Whether special tokens should be parsed.
    pub parses_special: bool,
}

/// Prompt template builder with history support
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    template_type: TemplateType,
    system_prompt: Option<String>,
    conversation_history: Vec<(String, String)>,
}

impl PromptTemplate {
    /// Create a new prompt template
    pub fn new(template_type: TemplateType) -> Self {
        Self { template_type, system_prompt: None, conversation_history: Vec::new() }
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Add a turn to conversation history
    pub fn add_turn(&mut self, user: impl Into<String>, assistant: impl Into<String>) {
        self.conversation_history.push((user.into(), assistant.into()));
    }

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
    }

    /// Format a user message with full context
    pub fn format(&self, user_text: &str) -> String {
        // For now, just apply the template to the current message
        // Multi-turn history can be added later
        self.template_type.apply(user_text, self.system_prompt.as_deref())
    }

    /// Get default stop sequences for this template
    pub fn stop_sequences(&self) -> Vec<String> {
        self.template_type.default_stop_sequences()
    }

    /// Check if BOS should be added
    pub fn should_add_bos(&self) -> bool {
        self.template_type.should_add_bos()
    }

    /// Get template type
    pub fn template_type(&self) -> TemplateType {
        self.template_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi4_chat_template() {
        let template = TemplateType::Phi4Chat;

        // Without system prompt (default system prompt added)
        let result = template.apply("Hello!", None);
        assert!(result.contains("<|im_start|>system\n"));
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains("<|im_end|>"));
        assert!(result.contains("<|im_start|>user\n"));
        assert!(result.contains("Hello!"));
        assert!(result.ends_with("<|im_start|>assistant\n"));

        // With custom system prompt
        let result = template.apply("Hello!", Some("You are a math tutor."));
        assert!(result.contains("You are a math tutor."));
        assert!(!result.contains("You are a helpful assistant."));
    }

    #[test]
    fn test_render_chat_phi4() {
        let t = TemplateType::Phi4Chat;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi there!"),
            ChatTurn::new(ChatRole::User, "How are you?"),
        ];
        let s = t.render_chat(&hist, Some("You are helpful.")).unwrap();

        assert!(s.contains("<|im_start|>system\n"));
        assert!(s.contains("You are helpful."));
        assert!(s.contains("<|im_start|>user\n"));
        assert!(s.contains("Hello"));
        assert!(s.contains("<|im_start|>assistant\n"));
        assert!(s.contains("Hi there!"));
        assert!(s.contains("How are you?"));
        assert!(s.contains("<|im_end|>"));
        assert!(s.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_detect_phi4_from_jinja() {
        let t = TemplateType::detect(
            None,
            Some("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>"),
        );
        assert_eq!(t, TemplateType::Phi4Chat);
    }

    #[test]
    fn test_detect_phi4_from_name() {
        let t = TemplateType::detect(Some("phi-4-mini"), None);
        assert_eq!(t, TemplateType::Phi4Chat);
    }

    #[test]
    fn test_qwen_chat_template() {
        let template = TemplateType::QwenChat;

        let result = template.apply("Hello!", None);
        assert!(result.contains("<|im_start|>system\n"));
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains("<|im_end|>"));
        assert!(result.contains("<|im_start|>user\n"));
        assert!(result.contains("Hello!"));
        assert!(result.ends_with("<|im_start|>assistant\n"));

        let result = template.apply("Hello!", Some("You are a math tutor."));
        assert!(result.contains("You are a math tutor."));
        assert!(!result.contains("You are a helpful assistant."));
    }

    #[test]
    fn test_detect_qwen_from_name() {
        let t = TemplateType::detect(Some("qwen2-7b"), None);
        assert_eq!(t, TemplateType::QwenChat);
    }

    #[test]
    fn test_render_chat_qwen() {
        let t = TemplateType::QwenChat;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi there!"),
            ChatTurn::new(ChatRole::User, "How are you?"),
        ];
        let s = t.render_chat(&hist, Some("You are helpful.")).unwrap();

        assert!(s.contains("<|im_start|>system\n"));
        assert!(s.contains("You are helpful."));
        assert!(s.contains("<|im_start|>user\n"));
        assert!(s.contains("Hello"));
        assert!(s.contains("<|im_start|>assistant\n"));
        assert!(s.contains("Hi there!"));
        assert!(s.contains("How are you?"));
        assert!(s.contains("<|im_end|>"));
        assert!(s.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_gemma_chat_template() {
        let template = TemplateType::GemmaChat;

        // Without system prompt
        let result = template.apply("Hello!", None);
        assert!(result.contains("<start_of_turn>user\n"));
        assert!(result.contains("Hello!"));
        assert!(result.contains("<end_of_turn>"));
        assert!(result.ends_with("<start_of_turn>model\n"));

        // With system prompt (prepended to user message)
        let result = template.apply("Hello!", Some("You are a math tutor."));
        assert!(result.contains("You are a math tutor."));
        assert!(result.contains("Hello!"));
        assert!(result.contains("<start_of_turn>user\n"));
        assert!(result.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn test_render_chat_gemma() {
        let t = TemplateType::GemmaChat;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi there!"),
            ChatTurn::new(ChatRole::User, "How are you?"),
        ];
        let s = t.render_chat(&hist, Some("You are helpful.")).unwrap();

        assert!(s.contains("<start_of_turn>user\n"));
        assert!(s.contains("You are helpful."));
        assert!(s.contains("Hello"));
        assert!(s.contains("<start_of_turn>model\n"));
        assert!(s.contains("Hi there!"));
        assert!(s.contains("How are you?"));
        assert!(s.contains("<end_of_turn>"));
        assert!(s.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn test_detect_gemma_from_jinja() {
        let t = TemplateType::detect(
            None,
            Some("<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"),
        );
        assert_eq!(t, TemplateType::GemmaChat);
    }

    #[test]
    fn test_detect_gemma_from_name() {
        let t = TemplateType::detect(Some("gemma-2b"), None);
        assert_eq!(t, TemplateType::GemmaChat);
    }

    #[test]
    fn test_deepseek_chat_template() {
        let template = TemplateType::DeepSeekChat;

        let result = template.apply("Hello!", None);
        assert!(result.contains("<|im_start|>system\n"));
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains("<|im_end|>"));
        assert!(result.contains("<|im_start|>user\n"));
        assert!(result.contains("Hello!"));
        assert!(result.ends_with("<|im_start|>assistant\n"));

        let result = template.apply("Hello!", Some("You are a math tutor."));
        assert!(result.contains("You are a math tutor."));
        assert!(!result.contains("You are a helpful assistant."));
    }

    #[test]
    fn test_detect_deepseek_from_name() {
        let t = TemplateType::detect(Some("deepseek-v2-lite"), None);
        assert_eq!(t, TemplateType::DeepSeekChat);
    }

    #[test]
    fn test_render_chat_deepseek() {
        let t = TemplateType::DeepSeekChat;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi there!"),
            ChatTurn::new(ChatRole::User, "How are you?"),
        ];
        let s = t.render_chat(&hist, Some("You are helpful.")).unwrap();

        assert!(s.contains("<|im_start|>system\n"));
        assert!(s.contains("You are helpful."));
        assert!(s.contains("<|im_start|>user\n"));
        assert!(s.contains("Hello"));
        assert!(s.contains("<|im_start|>assistant\n"));
        assert!(s.contains("Hi there!"));
        assert!(s.contains("How are you?"));
        assert!(s.contains("<|im_end|>"));
        assert!(s.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn snapshot_deepseek_single_turn() {
        let result = TemplateType::DeepSeekChat.apply("What is 2+2?", None);
        insta::assert_snapshot!(result);
    }

    #[test]
    fn snapshot_deepseek_with_system() {
        let result =
            TemplateType::DeepSeekChat.apply("Explain monads", Some("You are a Haskell tutor."));
        insta::assert_snapshot!(result);
    }

    #[test]
    fn snapshot_deepseek_multi_turn() {
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi!"),
            ChatTurn::new(ChatRole::User, "How are you?"),
        ];
        let result =
            TemplateType::DeepSeekChat.render_chat(&hist, Some("You are friendly.")).unwrap();
        insta::assert_snapshot!(result);
    }

    #[test]
    fn test_starcoder_template() {
        let template = TemplateType::StarCoder;

        let result = template.apply("def hello():", None);
        assert_eq!(result, "def hello():");

        let result = template.apply("def hello():", Some("Complete this function"));
        assert!(result.starts_with("# Complete this function\n"));
        assert!(result.contains("def hello():"));
    }

    #[test]
    fn test_detect_starcoder_from_name() {
        let t = TemplateType::detect(Some("bigcode-starcoder"), None);
        assert_eq!(t, TemplateType::StarCoder);
    }

    #[test]
    fn test_raw_template() {
        let template = TemplateType::Raw;
        let result = template.apply("Hello, world!", None);
        assert_eq!(result, "Hello, world!");

        let result_with_system = template.apply("Hello, world!", Some("You are helpful"));
        assert_eq!(result_with_system, "Hello, world!");
    }

    #[test]
    fn test_instruct_template() {
        let template = TemplateType::Instruct;

        // Without system prompt
        let result = template.apply("What is 2+2?", None);
        assert_eq!(result, "Q: What is 2+2?\nA:");

        // With system prompt
        let result = template.apply("What is 2+2?", Some("You are a math tutor"));
        assert!(result.contains("System: You are a math tutor"));
        assert!(result.contains("Q: What is 2+2?"));
        assert!(result.ends_with("\nA:"));
    }

    #[test]
    fn test_llama3_chat_template() {
        let template = TemplateType::Llama3Chat;

        // Without system prompt
        let result = template.apply("Hello!", None);
        assert!(result.starts_with("<|begin_of_text|>"));
        assert!(result.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(result.contains("Hello!"));
        assert!(result.contains("<|eot_id|>"));
        assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));

        // With system prompt
        let result = template.apply("Hello!", Some("You are helpful"));
        assert!(result.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(result.contains("You are helpful"));
    }

    #[test]
    fn test_template_from_str() {
        assert_eq!("raw".parse::<TemplateType>().unwrap(), TemplateType::Raw);
        assert_eq!("instruct".parse::<TemplateType>().unwrap(), TemplateType::Instruct);
        assert_eq!("llama3-chat".parse::<TemplateType>().unwrap(), TemplateType::Llama3Chat);
        assert_eq!("llama3_chat".parse::<TemplateType>().unwrap(), TemplateType::Llama3Chat);
        assert_eq!("phi4-chat".parse::<TemplateType>().unwrap(), TemplateType::Phi4Chat);
        assert_eq!("phi4_chat".parse::<TemplateType>().unwrap(), TemplateType::Phi4Chat);
        assert_eq!("phi4".parse::<TemplateType>().unwrap(), TemplateType::Phi4Chat);
        assert_eq!("chatml".parse::<TemplateType>().unwrap(), TemplateType::Phi4Chat);
        assert_eq!("qwen-chat".parse::<TemplateType>().unwrap(), TemplateType::QwenChat);
        assert_eq!("qwen_chat".parse::<TemplateType>().unwrap(), TemplateType::QwenChat);
        assert_eq!("qwen".parse::<TemplateType>().unwrap(), TemplateType::QwenChat);
        assert_eq!("gemma-chat".parse::<TemplateType>().unwrap(), TemplateType::GemmaChat);
        assert_eq!("gemma_chat".parse::<TemplateType>().unwrap(), TemplateType::GemmaChat);
        assert_eq!("gemma".parse::<TemplateType>().unwrap(), TemplateType::GemmaChat);
        assert_eq!("mistral-chat".parse::<TemplateType>().unwrap(), TemplateType::MistralChat);
        assert_eq!("mistral_chat".parse::<TemplateType>().unwrap(), TemplateType::MistralChat);
        assert_eq!("mistral".parse::<TemplateType>().unwrap(), TemplateType::MistralChat);
        assert_eq!("deepseek-chat".parse::<TemplateType>().unwrap(), TemplateType::DeepSeekChat);
        assert_eq!("deepseek_chat".parse::<TemplateType>().unwrap(), TemplateType::DeepSeekChat);
        assert_eq!("deepseek".parse::<TemplateType>().unwrap(), TemplateType::DeepSeekChat);
        assert_eq!("starcoder".parse::<TemplateType>().unwrap(), TemplateType::StarCoder);
        assert_eq!("code-completion".parse::<TemplateType>().unwrap(), TemplateType::StarCoder);
        assert_eq!("falcon-chat".parse::<TemplateType>().unwrap(), TemplateType::FalconChat);
        assert_eq!("falcon".parse::<TemplateType>().unwrap(), TemplateType::FalconChat);
        assert_eq!(
            "codellama-instruct".parse::<TemplateType>().unwrap(),
            TemplateType::CodeLlamaInstruct
        );
        assert_eq!("codellama".parse::<TemplateType>().unwrap(), TemplateType::CodeLlamaInstruct);
        assert_eq!("cohere-command".parse::<TemplateType>().unwrap(), TemplateType::CohereCommand);
        assert_eq!("cohere".parse::<TemplateType>().unwrap(), TemplateType::CohereCommand);
        assert_eq!("command-r".parse::<TemplateType>().unwrap(), TemplateType::CohereCommand);
        assert_eq!("internlm-chat".parse::<TemplateType>().unwrap(), TemplateType::InternLMChat);
        assert_eq!("internlm".parse::<TemplateType>().unwrap(), TemplateType::InternLMChat);
        assert_eq!("yi-chat".parse::<TemplateType>().unwrap(), TemplateType::YiChat);
        assert_eq!("yi".parse::<TemplateType>().unwrap(), TemplateType::YiChat);
        assert_eq!("baichuan-chat".parse::<TemplateType>().unwrap(), TemplateType::BaichuanChat);
        assert_eq!("baichuan".parse::<TemplateType>().unwrap(), TemplateType::BaichuanChat);
        assert_eq!("chatglm-chat".parse::<TemplateType>().unwrap(), TemplateType::ChatGLMChat);
        assert_eq!("glm-4".parse::<TemplateType>().unwrap(), TemplateType::ChatGLMChat);
        assert_eq!("mpt-instruct".parse::<TemplateType>().unwrap(), TemplateType::MptInstruct);
        assert_eq!("mpt".parse::<TemplateType>().unwrap(), TemplateType::MptInstruct);
        assert_eq!("rwkv-world".parse::<TemplateType>().unwrap(), TemplateType::RwkvWorld);
        assert_eq!("rwkv".parse::<TemplateType>().unwrap(), TemplateType::RwkvWorld);
        assert_eq!("olmo-instruct".parse::<TemplateType>().unwrap(), TemplateType::OlmoInstruct);
        assert_eq!("olmo".parse::<TemplateType>().unwrap(), TemplateType::OlmoInstruct);
        assert_eq!("fill-in-middle".parse::<TemplateType>().unwrap(), TemplateType::FillInMiddle);
        assert_eq!("fim".parse::<TemplateType>().unwrap(), TemplateType::FillInMiddle);
        assert_eq!("zephyr-chat".parse::<TemplateType>().unwrap(), TemplateType::ZephyrChat);
        assert_eq!("zephyr".parse::<TemplateType>().unwrap(), TemplateType::ZephyrChat);
        assert_eq!("vicuna-chat".parse::<TemplateType>().unwrap(), TemplateType::VicunaChat);
        assert_eq!("vicuna".parse::<TemplateType>().unwrap(), TemplateType::VicunaChat);
        assert_eq!("orca-chat".parse::<TemplateType>().unwrap(), TemplateType::OrcaChat);
        assert_eq!("orca".parse::<TemplateType>().unwrap(), TemplateType::OrcaChat);
        assert_eq!("solar-instruct".parse::<TemplateType>().unwrap(), TemplateType::SolarInstruct);
        assert_eq!("solar".parse::<TemplateType>().unwrap(), TemplateType::SolarInstruct);
        assert_eq!(
            "alpaca-instruct".parse::<TemplateType>().unwrap(),
            TemplateType::AlpacaInstruct
        );
        assert_eq!("alpaca".parse::<TemplateType>().unwrap(), TemplateType::AlpacaInstruct);

        assert!("invalid".parse::<TemplateType>().is_err());
    }

    #[test]
    fn test_stop_sequences() {
        assert_eq!(TemplateType::Raw.default_stop_sequences(), Vec::<String>::new());
        assert!(!TemplateType::Instruct.default_stop_sequences().is_empty());
        assert!(!TemplateType::Llama3Chat.default_stop_sequences().is_empty());
        assert!(!TemplateType::Phi4Chat.default_stop_sequences().is_empty());
        assert!(!TemplateType::QwenChat.default_stop_sequences().is_empty());
        assert!(!TemplateType::GemmaChat.default_stop_sequences().is_empty());
        assert!(!TemplateType::MistralChat.default_stop_sequences().is_empty());
        assert!(!TemplateType::DeepSeekChat.default_stop_sequences().is_empty());
        assert!(!TemplateType::StarCoder.default_stop_sequences().is_empty());
        assert!(!TemplateType::FalconChat.default_stop_sequences().is_empty());
        assert!(!TemplateType::CodeLlamaInstruct.default_stop_sequences().is_empty());
        assert!(!TemplateType::CohereCommand.default_stop_sequences().is_empty());
        assert!(!TemplateType::InternLMChat.default_stop_sequences().is_empty());
        assert!(!TemplateType::YiChat.default_stop_sequences().is_empty());
        assert!(!TemplateType::BaichuanChat.default_stop_sequences().is_empty());
        assert!(!TemplateType::ChatGLMChat.default_stop_sequences().is_empty());
        assert!(!TemplateType::MptInstruct.default_stop_sequences().is_empty());
        assert!(!TemplateType::RwkvWorld.default_stop_sequences().is_empty());
        assert!(!TemplateType::OlmoInstruct.default_stop_sequences().is_empty());
        assert!(!TemplateType::FillInMiddle.default_stop_sequences().is_empty());
        assert!(!TemplateType::ZephyrChat.default_stop_sequences().is_empty());
        assert!(!TemplateType::VicunaChat.default_stop_sequences().is_empty());
        assert!(!TemplateType::OrcaChat.default_stop_sequences().is_empty());
        assert!(!TemplateType::SolarInstruct.default_stop_sequences().is_empty());
        assert!(!TemplateType::AlpacaInstruct.default_stop_sequences().is_empty());

        // Check llama3-chat has the expected stop tokens
        let llama3_stops = TemplateType::Llama3Chat.default_stop_sequences();
        assert!(llama3_stops.contains(&"<|eot_id|>".to_string()));

        // Check phi4-chat has the expected stop tokens
        let phi4_stops = TemplateType::Phi4Chat.default_stop_sequences();
        assert!(phi4_stops.contains(&"<|im_end|>".to_string()));

        // Check gemma-chat has the expected stop tokens
        let gemma_stops = TemplateType::GemmaChat.default_stop_sequences();
        assert!(gemma_stops.contains(&"<end_of_turn>".to_string()));

        // Check mistral-chat has the expected stop tokens
        let mistral_stops = TemplateType::MistralChat.default_stop_sequences();
        assert!(mistral_stops.contains(&"</s>".to_string()));
    }

    #[test]
    fn test_resolve_stop_token_ids() {
        // Create a mock tokenizer that can resolve special tokens
        use bitnet_tokenizers::MockTokenizer;
        let tokenizer = MockTokenizer::new();

        // Test that Raw template returns empty (no stops)
        let raw_ids = TemplateType::Raw.resolve_stop_token_ids(&tokenizer);
        assert_eq!(raw_ids, Vec::<u32>::new());

        // Test that Instruct template returns empty for mock tokenizer
        // (mock tokenizer doesn't resolve the instruct stop sequences)
        let instruct_ids = TemplateType::Instruct.resolve_stop_token_ids(&tokenizer);
        assert_eq!(instruct_ids, Vec::<u32>::new());

        // Test that LLaMA3Chat template also returns empty for mock tokenizer
        // In a real scenario with a real tokenizer that has <|eot_id|> in vocab,
        // this would return the resolved token IDs
        let llama3_ids = TemplateType::Llama3Chat.resolve_stop_token_ids(&tokenizer);
        assert_eq!(llama3_ids, Vec::<u32>::new());
    }

    #[test]
    fn test_template_glue_with_real_token_ids() {
        // This test proves the complete template glue: template → stops → token IDs
        // Given a mock tokenizer that maps <|eot_id|> → 128009 (LLaMA-3's actual EOT token ID)
        use bitnet_tokenizers::MockTokenizer;

        let tokenizer = MockTokenizer::with_special_tokens(&[
            ("<|eot_id|>", 128009),
            ("<|end_of_text|>", 128010),
        ]);

        // Test LLaMA3Chat template
        let template = TemplateType::Llama3Chat;

        // Assert: default_stop_sequences includes "<|eot_id|>"
        let stops = template.default_stop_sequences();
        assert!(stops.contains(&"<|eot_id|>".to_string()));
        assert!(stops.contains(&"<|end_of_text|>".to_string()));

        // Assert: resolve_stop_token_ids returns [128009, 128010]
        let stop_ids = template.resolve_stop_token_ids(&tokenizer);
        assert!(stop_ids.contains(&128009), "Expected 128009 for <|eot_id|>");
        assert!(stop_ids.contains(&128010), "Expected 128010 for <|end_of_text|>");

        // Assert: apply() wraps system_prompt + user in LLaMA-3 format
        let formatted = template.apply("What is 2+2?", Some("You are helpful"));
        assert!(formatted.contains("<|begin_of_text|>"));
        assert!(formatted.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(formatted.contains("You are helpful"));
        assert!(formatted.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(formatted.contains("What is 2+2?"));
        assert!(formatted.contains("<|eot_id|>"));
        assert!(formatted.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_bos_control() {
        assert!(TemplateType::Raw.should_add_bos());
        assert!(TemplateType::Instruct.should_add_bos());
        assert!(!TemplateType::Llama3Chat.should_add_bos()); // Has its own BOS
        assert!(!TemplateType::Phi4Chat.should_add_bos()); // Uses im_start/im_end
        assert!(!TemplateType::QwenChat.should_add_bos()); // Uses im_start/im_end
        assert!(!TemplateType::GemmaChat.should_add_bos()); // Uses start_of_turn
        assert!(!TemplateType::MistralChat.should_add_bos()); // Template includes <s>
        assert!(!TemplateType::DeepSeekChat.should_add_bos()); // ChatML tokens
        assert!(TemplateType::StarCoder.should_add_bos()); // Simple completion
        assert!(TemplateType::FalconChat.should_add_bos()); // User:/Falcon: format
        assert!(!TemplateType::CodeLlamaInstruct.should_add_bos()); // [INST] markers
        assert!(!TemplateType::CohereCommand.should_add_bos()); // Turn tokens
        assert!(!TemplateType::InternLMChat.should_add_bos()); // ChatML tokens
        assert!(!TemplateType::YiChat.should_add_bos()); // ChatML tokens
        assert!(!TemplateType::BaichuanChat.should_add_bos()); // Reserved tokens
        assert!(!TemplateType::ChatGLMChat.should_add_bos()); // gMASK tokens
        assert!(TemplateType::MptInstruct.should_add_bos()); // Simple markers
        assert!(TemplateType::RwkvWorld.should_add_bos()); // Simple User:/Assistant: format
        assert!(!TemplateType::OlmoInstruct.should_add_bos()); // Uses special tokens
        assert!(!TemplateType::FillInMiddle.should_add_bos()); // Uses FIM tokens
        assert!(!TemplateType::ZephyrChat.should_add_bos()); // Uses special tokens
        assert!(TemplateType::VicunaChat.should_add_bos()); // Simple USER:/ASSISTANT: format
        assert!(!TemplateType::OrcaChat.should_add_bos()); // ChatML uses im_start/im_end tokens
        assert!(TemplateType::SolarInstruct.should_add_bos()); // Simple ### markers
        assert!(TemplateType::AlpacaInstruct.should_add_bos()); // Simple ### markers
    }

    #[test]
    fn test_parse_special_control() {
        assert!(!TemplateType::Raw.parse_special());
        assert!(!TemplateType::Instruct.parse_special());
        assert!(TemplateType::Llama3Chat.parse_special()); // LLaMA-3 has special tokens
        assert!(TemplateType::Phi4Chat.parse_special()); // Phi-4 has special tokens
        assert!(TemplateType::QwenChat.parse_special()); // Qwen has special tokens
        assert!(TemplateType::GemmaChat.parse_special()); // Gemma has special tokens
        assert!(TemplateType::MistralChat.parse_special()); // Mistral has special tokens
        assert!(TemplateType::DeepSeekChat.parse_special()); // DeepSeek has special tokens
        assert!(TemplateType::StarCoder.parse_special()); // StarCoder has endoftext
        assert!(!TemplateType::FalconChat.parse_special()); // Simple text format
        assert!(TemplateType::CodeLlamaInstruct.parse_special()); // Has special tokens
        assert!(TemplateType::CohereCommand.parse_special()); // Has turn tokens
        assert!(TemplateType::InternLMChat.parse_special()); // Has im_start/im_end
        assert!(TemplateType::YiChat.parse_special()); // Has im_start/im_end
        assert!(TemplateType::BaichuanChat.parse_special()); // Has reserved tokens
        assert!(TemplateType::ChatGLMChat.parse_special()); // Has gMASK/sop tokens
        assert!(!TemplateType::MptInstruct.parse_special()); // Simple text markers
        assert!(!TemplateType::RwkvWorld.parse_special()); // Simple text format
        assert!(TemplateType::OlmoInstruct.parse_special()); // Has special tokens
        assert!(TemplateType::FillInMiddle.parse_special()); // Has FIM tokens
        assert!(TemplateType::ZephyrChat.parse_special()); // Has special tokens
        assert!(!TemplateType::VicunaChat.parse_special()); // Simple text format
        assert!(TemplateType::OrcaChat.parse_special()); // Has im_start/im_end
        assert!(!TemplateType::SolarInstruct.parse_special()); // Simple text markers
        assert!(!TemplateType::AlpacaInstruct.parse_special()); // Simple text markers
    }

    #[test]
    fn test_prompt_template_builder() {
        let template = PromptTemplate::new(TemplateType::Instruct)
            .with_system_prompt("You are a helpful assistant");

        let formatted = template.format("What is Rust?");
        assert!(formatted.contains("System: You are a helpful assistant"));
        assert!(formatted.contains("Q: What is Rust?"));

        assert!(!template.stop_sequences().is_empty());
        assert!(template.should_add_bos());
    }

    #[test]
    fn test_conversation_history() {
        let mut template = PromptTemplate::new(TemplateType::Raw);

        template.add_turn("Hello", "Hi there!");
        template.add_turn("How are you?", "I'm doing well!");

        assert_eq!(template.conversation_history.len(), 2);

        template.clear_history();
        assert_eq!(template.conversation_history.len(), 0);
    }

    #[test]
    fn test_render_chat_llama3() {
        let t = TemplateType::Llama3Chat;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi there!"),
            ChatTurn::new(ChatRole::User, "How are you?"),
        ];
        let s = t.render_chat(&hist, Some("You are helpful.")).unwrap();

        // Check for LLaMA-3 special tokens
        assert!(s.contains("<|begin_of_text|>"));
        assert!(s.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(s.contains("You are helpful."));
        assert!(s.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(s.contains("Hello"));
        assert!(s.contains("<|start_header_id|>assistant<|end_header_id|>"));
        assert!(s.contains("Hi there!"));
        assert!(s.contains("How are you?"));
        assert!(s.contains("<|eot_id|>"));

        // Should end with assistant header ready for generation
        assert!(s.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_render_chat_instruct() {
        let t = TemplateType::Instruct;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "What is 2+2?"),
            ChatTurn::new(ChatRole::Assistant, "It's 4."),
            ChatTurn::new(ChatRole::User, "What about 3+3?"),
        ];
        let s = t.render_chat(&hist, None).unwrap();

        // Check Q&A format
        assert!(s.contains("Q: What is 2+2?"));
        assert!(s.contains("A: It's 4."));
        assert!(s.contains("Q: What about 3+3?"));

        // Should end with "A: " to prompt for response
        assert!(s.ends_with("A: "));
    }

    #[test]
    fn test_render_chat_instruct_with_system() {
        let t = TemplateType::Instruct;
        let hist = vec![ChatTurn::new(ChatRole::User, "Q1")];
        let s = t.render_chat(&hist, Some("You are a math tutor")).unwrap();

        assert!(s.contains("System: You are a math tutor"));
        assert!(s.contains("Q: Q1"));
        assert!(s.ends_with("A: "));
    }

    #[test]
    fn test_render_chat_raw() {
        let t = TemplateType::Raw;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "First message"),
            ChatTurn::new(ChatRole::Assistant, "First response"),
            ChatTurn::new(ChatRole::User, "Second message"),
        ];
        let s = t.render_chat(&hist, None).unwrap();

        // Raw mode should concatenate full history with double newline separators
        assert!(s.contains("First message"));
        assert!(s.contains("First response"));
        assert!(s.contains("Second message"));
    }

    #[test]
    fn test_render_chat_raw_with_system() {
        let t = TemplateType::Raw;
        let hist = vec![ChatTurn::new(ChatRole::User, "Hello")];
        let s = t.render_chat(&hist, Some("System context")).unwrap();

        assert!(s.contains("System context"));
        assert!(s.contains("Hello"));
    }

    #[test]
    fn test_chat_role_as_str() {
        assert_eq!(ChatRole::System.as_str(), "system");
        assert_eq!(ChatRole::User.as_str(), "user");
        assert_eq!(ChatRole::Assistant.as_str(), "assistant");
    }

    #[test]
    fn test_chat_turn_new() {
        let turn = ChatTurn::new(ChatRole::User, "test message");
        assert_eq!(turn.role, ChatRole::User);
        assert_eq!(turn.text, "test message");
    }

    #[test]
    fn test_validate_output_valid() {
        let t = TemplateType::Instruct;
        let output = t.apply("Hello world", None);
        let v = t.validate_output(&output, "Hello world");
        assert!(v.is_valid, "warnings: {:?}", v.warnings);
    }

    #[test]
    fn test_validate_output_empty() {
        let t = TemplateType::Raw;
        let v = t.validate_output("", "Hello");
        assert!(!v.is_valid);
        assert!(v.warnings.iter().any(|w| w.contains("empty")));
    }

    #[test]
    fn test_validate_output_missing_user_text() {
        let t = TemplateType::Instruct;
        let v = t.validate_output("Some random output", "Hello world");
        assert!(!v.is_valid);
        assert!(v.warnings.iter().any(|w| w.contains("user text")));
    }

    #[test]
    fn test_template_info() {
        let info = TemplateType::Phi4Chat.info();
        assert_eq!(info.name, "phi4-chat");
        assert!(!info.stop_sequences.is_empty());
        assert!(!info.adds_bos);
        assert!(info.parses_special);
    }

    #[test]
    fn test_all_templates_validate_own_output() {
        for template in &[
            TemplateType::Raw, TemplateType::Instruct, TemplateType::Llama3Chat,
            TemplateType::Phi4Chat, TemplateType::QwenChat, TemplateType::GemmaChat,
            TemplateType::MistralChat, TemplateType::DeepSeekChat, TemplateType::StarCoder,
            TemplateType::FalconChat, TemplateType::CodeLlamaInstruct,
            TemplateType::CohereCommand, TemplateType::InternLMChat,
            TemplateType::YiChat, TemplateType::BaichuanChat,
            TemplateType::ChatGLMChat, TemplateType::MptInstruct,
            TemplateType::RwkvWorld, TemplateType::OlmoInstruct,
            TemplateType::FillInMiddle, TemplateType::ZephyrChat,
            TemplateType::VicunaChat,
        ] {
            let output = template.apply("Test input", None);
            let v = template.validate_output(&output, "Test input");
            assert!(
                v.is_valid,
                "Template {:?} failed self-validation: {:?}",
                template, v.warnings
            );
        }
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    fn arb_template_type() -> impl Strategy<Value = TemplateType> {
        prop_oneof![
            Just(TemplateType::Raw),
            Just(TemplateType::Instruct),
            Just(TemplateType::Llama3Chat),
            Just(TemplateType::Phi4Chat),
            Just(TemplateType::QwenChat),
            Just(TemplateType::GemmaChat),
            Just(TemplateType::MistralChat),
            Just(TemplateType::DeepSeekChat),
            Just(TemplateType::StarCoder),
            Just(TemplateType::FalconChat),
            Just(TemplateType::CodeLlamaInstruct),
            Just(TemplateType::CohereCommand),
            Just(TemplateType::InternLMChat),
            Just(TemplateType::YiChat),
            Just(TemplateType::BaichuanChat),
            Just(TemplateType::ChatGLMChat),
            Just(TemplateType::MptInstruct),
            Just(TemplateType::RwkvWorld),
            Just(TemplateType::OlmoInstruct),
            Just(TemplateType::FillInMiddle),
            Just(TemplateType::ZephyrChat),
            Just(TemplateType::VicunaChat),
        ]
    }

    // apply always returns a non-empty string containing the user text.
    proptest! {
        #[test]
        fn apply_contains_user_text(
            template in arb_template_type(),
            user_text in "[a-zA-Z0-9 .,?!]{1,80}",
        ) {
            let result = template.apply(&user_text, None);
            prop_assert!(
                !result.is_empty(),
                "apply returned empty string for template={:?}",
                template
            );
            prop_assert!(
                result.contains(&user_text),
                "output {:?} should contain user_text {:?}",
                result,
                user_text
            );
        }
    }

    // Raw template passes user text through unchanged (no system prompt).
    proptest! {
        #[test]
        fn raw_template_is_identity(user_text in "[a-zA-Z0-9 .,?!]{1,80}") {
            let result = TemplateType::Raw.apply(&user_text, None);
            prop_assert_eq!(result, user_text);
        }
    }

    // Instruct template always ends with "\nA:".
    proptest! {
        #[test]
        fn instruct_ends_with_answer_prompt(
            user_text in "[a-zA-Z0-9 .,?!]{1,80}",
            system in proptest::option::of("[a-zA-Z0-9 ]{1,40}"),
        ) {
            let result = TemplateType::Instruct.apply(&user_text, system.as_deref());
            prop_assert!(
                result.ends_with("\nA:"),
                "instruct result {:?} should end with '\\nA:'",
                result
            );
        }
    }

    // default_stop_sequences returns at least one entry for non-Raw templates.
    proptest! {
        #[test]
        fn non_raw_templates_have_stop_sequences(
            template in prop_oneof![
                Just(TemplateType::Instruct),
                Just(TemplateType::Llama3Chat),
                Just(TemplateType::Phi4Chat),
                Just(TemplateType::QwenChat),
                Just(TemplateType::GemmaChat),
                Just(TemplateType::MistralChat),
                Just(TemplateType::DeepSeekChat),
                Just(TemplateType::StarCoder),
                Just(TemplateType::FalconChat),
                Just(TemplateType::CodeLlamaInstruct),
                Just(TemplateType::CohereCommand),
                Just(TemplateType::InternLMChat),
                Just(TemplateType::YiChat),
                Just(TemplateType::BaichuanChat),
                Just(TemplateType::ChatGLMChat),
                Just(TemplateType::MptInstruct),
                Just(TemplateType::RwkvWorld),
                Just(TemplateType::OlmoInstruct),
                Just(TemplateType::FillInMiddle),
                Just(TemplateType::ZephyrChat),
                Just(TemplateType::VicunaChat),
            ],
        ) {
            let stops = template.default_stop_sequences();
            prop_assert!(
                !stops.is_empty(),
                "template={:?} should have default stop sequences",
                template
            );
        }
    }
}

#[cfg(test)]
mod detect_logging_tests {
    use super::*;
    use tracing_test::traced_test;

    /// `detect()` emits a debug log naming the chosen template when a GGUF signature matches.
    #[test]
    #[traced_test]
    fn detection_decision_is_logged() {
        let _t = TemplateType::detect(
            None,
            Some("<|start_header_id|>user<|end_header_id|>\n{u}<|eot_id|>"),
        );
        assert!(
            logs_contain("Llama3Chat") || logs_contain("auto-detected"),
            "detect() must emit a debug log for the detected template"
        );
    }

    /// `detect()` emits a warn log when no signature matches and falling back to Raw.
    #[test]
    #[traced_test]
    fn fallback_to_raw_is_warned() {
        let _t = TemplateType::detect(None, None);
        assert!(
            logs_contain("falling back to Raw") || logs_contain("Raw"),
            "detect() must emit a warn log when falling back to Raw"
        );
    }

    // ── Falcon Chat ────────────────────────────────────────────────────

    #[test]
    fn test_falcon_chat_template() {
        let template = TemplateType::FalconChat;

        let result = template.apply("Hello!", None);
        assert!(result.contains("User: Hello!"));
        assert!(result.ends_with("\nFalcon:"));

        let result = template.apply("Hello!", Some("Be concise."));
        assert!(result.contains("System: Be concise."));
        assert!(result.contains("User: Hello!"));
    }

    #[test]
    fn test_detect_falcon_from_name() {
        let t = TemplateType::detect(Some("tiiuae-falcon-7b"), None);
        assert_eq!(t, TemplateType::FalconChat);
    }

    #[test]
    fn test_render_chat_falcon() {
        let t = TemplateType::FalconChat;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi there!"),
            ChatTurn::new(ChatRole::User, "How are you?"),
        ];
        let s = t.render_chat(&hist, Some("Be helpful.")).unwrap();

        assert!(s.contains("System: Be helpful."));
        assert!(s.contains("User: Hello"));
        assert!(s.contains("Falcon: Hi there!"));
        assert!(s.contains("User: How are you?"));
        assert!(s.ends_with("Falcon:"));
    }

    // ── CodeLlama Instruct ─────────────────────────────────────────────

    #[test]
    fn test_codellama_instruct_template() {
        let template = TemplateType::CodeLlamaInstruct;

        let result = template.apply("Write a hello world", None);
        assert!(result.starts_with("[INST] "));
        assert!(result.contains("Write a hello world"));
        assert!(result.ends_with(" [/INST]"));

        let result = template.apply("Write a sort", Some("You are a Python expert."));
        assert!(result.contains("<<SYS>>"));
        assert!(result.contains("You are a Python expert."));
        assert!(result.contains("<</SYS>>"));
        assert!(result.contains("Write a sort"));
    }

    #[test]
    fn test_detect_codellama_from_name() {
        let t = TemplateType::detect(Some("codellama-7b-instruct"), None);
        assert_eq!(t, TemplateType::CodeLlamaInstruct);
    }

    // ── Cohere Command ─────────────────────────────────────────────────

    #[test]
    fn test_cohere_command_template() {
        let template = TemplateType::CohereCommand;

        let result = template.apply("Hello!", None);
        assert!(result.contains("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"));
        assert!(result.contains("Hello!"));
        assert!(result.contains("<|END_OF_TURN_TOKEN|>"));
        assert!(result.contains("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"));

        let result = template.apply("Hello!", Some("Be concise."));
        assert!(result.contains("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>"));
        assert!(result.contains("Be concise."));
    }

    #[test]
    fn test_detect_cohere_from_name() {
        // "command-r" in name now maps to CommandRPlus
        let t = TemplateType::detect(Some("cohere-command-r-plus"), None);
        assert_eq!(t, TemplateType::CommandRPlus);
    }

    #[test]
    fn test_detect_cohere_from_jinja() {
        // <|START_OF_TURN_TOKEN|> in jinja now maps to CommandRPlus
        let t = TemplateType::detect(
            None,
            Some("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user}<|END_OF_TURN_TOKEN|>"),
        );
        assert_eq!(t, TemplateType::CommandRPlus);
    }

    #[test]
    fn test_render_chat_cohere() {
        let t = TemplateType::CohereCommand;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi!"),
            ChatTurn::new(ChatRole::User, "How are you?"),
        ];
        let s = t.render_chat(&hist, Some("Be helpful.")).unwrap();

        assert!(s.contains("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Be helpful."));
        assert!(s.contains("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello"));
        assert!(s.contains("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Hi!"));
        assert!(s.ends_with("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"));
    }

    // ── InternLM Chat ──────────────────────────────────────────────────

    #[test]
    fn test_internlm_chat_template() {
        let template = TemplateType::InternLMChat;

        let result = template.apply("Hello!", None);
        assert!(result.contains("<|im_start|>system\n"));
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains("<|im_end|>"));
        assert!(result.contains("<|im_start|>user\n"));
        assert!(result.contains("Hello!"));
        assert!(result.ends_with("<|im_start|>assistant\n"));

        let result = template.apply("Hello!", Some("You are a math tutor."));
        assert!(result.contains("You are a math tutor."));
        assert!(!result.contains("You are a helpful assistant."));
    }

    #[test]
    fn test_detect_internlm_from_name() {
        let t = TemplateType::detect(Some("internlm2-chat-7b"), None);
        assert_eq!(t, TemplateType::InternLMChat);
    }

    #[test]
    fn test_render_chat_internlm() {
        let t = TemplateType::InternLMChat;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi there!"),
            ChatTurn::new(ChatRole::User, "How are you?"),
        ];
        let s = t.render_chat(&hist, Some("You are helpful.")).unwrap();

        assert!(s.contains("<|im_start|>system\n"));
        assert!(s.contains("You are helpful."));
        assert!(s.contains("<|im_start|>user\n"));
        assert!(s.contains("Hello"));
        assert!(s.contains("<|im_start|>assistant\n"));
        assert!(s.contains("Hi there!"));
        assert!(s.contains("How are you?"));
        assert!(s.contains("<|im_end|>"));
        assert!(s.ends_with("<|im_start|>assistant\n"));
    }

    // ── Yi Chat ────────────────────────────────────────────────────────

    #[test]
    fn test_yi_chat_template() {
        let template = TemplateType::YiChat;
        let result = template.apply("Hello!", None);
        assert!(result.contains("<|im_start|>system\n"));
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains("<|im_start|>user\nHello!"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_detect_yi_from_name() {
        let t = TemplateType::detect(Some("yi-34b-chat"), None);
        assert_eq!(t, TemplateType::YiChat);
    }

    #[test]
    fn test_render_chat_yi() {
        let t = TemplateType::YiChat;
        let hist =
            vec![ChatTurn::new(ChatRole::User, "Hello"), ChatTurn::new(ChatRole::Assistant, "Hi!")];
        let s = t.render_chat(&hist, Some("Be concise.")).unwrap();
        assert!(s.contains("<|im_start|>system\nBe concise.<|im_end|>"));
        assert!(s.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(s.contains("<|im_start|>assistant\nHi!<|im_end|>"));
        assert!(s.ends_with("<|im_start|>assistant\n"));
    }

    // ── Baichuan Chat ──────────────────────────────────────────────────

    #[test]
    fn test_baichuan_chat_template() {
        let template = TemplateType::BaichuanChat;
        let result = template.apply("Hello!", None);
        assert!(result.contains("<reserved_106>Hello!"));
        assert!(result.contains("<reserved_107>"));
    }

    #[test]
    fn test_detect_baichuan_from_name() {
        let t = TemplateType::detect(Some("baichuan2-13b-chat"), None);
        assert_eq!(t, TemplateType::BaichuanChat);
    }

    #[test]
    fn test_render_chat_baichuan() {
        let t = TemplateType::BaichuanChat;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi!"),
            ChatTurn::new(ChatRole::User, "Bye"),
        ];
        let s = t.render_chat(&hist, None).unwrap();
        assert!(s.contains("<reserved_106>Hello"));
        assert!(s.contains("<reserved_107>Hi!"));
        assert!(s.contains("<reserved_106>Bye"));
        assert!(s.ends_with("<reserved_107>"));
    }

    // ── ChatGLM Chat ───────────────────────────────────────────────────

    #[test]
    fn test_chatglm_chat_template() {
        let template = TemplateType::ChatGLMChat;
        let result = template.apply("Hello!", None);
        assert!(result.starts_with("[gMASK]<sop>"));
        assert!(result.contains("<|user|>\nHello!"));
        assert!(result.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_chatglm_chat_with_system() {
        let template = TemplateType::ChatGLMChat;
        let result = template.apply("Hello!", Some("Be helpful."));
        assert!(result.contains("<|system|>\nBe helpful."));
        assert!(result.contains("<|user|>\nHello!"));
    }

    #[test]
    fn test_detect_chatglm_from_name() {
        let t = TemplateType::detect(Some("chatglm3-6b"), None);
        assert_eq!(t, TemplateType::ChatGLMChat);

        let t2 = TemplateType::detect(Some("glm-4-9b"), None);
        assert_eq!(t2, TemplateType::ChatGLMChat);
    }

    #[test]
    fn test_detect_chatglm_from_jinja() {
        let t = TemplateType::detect(None, Some("[gMASK]<sop><|user|>\n{content}<|assistant|>"));
        assert_eq!(t, TemplateType::ChatGLMChat);
    }

    #[test]
    fn test_render_chat_chatglm() {
        let t = TemplateType::ChatGLMChat;
        let hist =
            vec![ChatTurn::new(ChatRole::User, "Hello"), ChatTurn::new(ChatRole::Assistant, "Hi!")];
        let s = t.render_chat(&hist, Some("System.")).unwrap();
        assert!(s.starts_with("[gMASK]<sop>"));
        assert!(s.contains("<|system|>\nSystem."));
        assert!(s.contains("<|user|>\nHello"));
        assert!(s.contains("<|assistant|>\nHi!"));
        assert!(s.ends_with("<|assistant|>\n"));
    }

    // ── MPT Instruct ───────────────────────────────────────────────────

    #[test]
    fn test_mpt_instruct_template() {
        let template = TemplateType::MptInstruct;
        let result = template.apply("Hello!", None);
        assert!(result.contains("### Instruction\nHello!"));
        assert!(result.ends_with("### Response\n"));
    }

    #[test]
    fn test_mpt_instruct_with_system() {
        let template = TemplateType::MptInstruct;
        let result = template.apply("Hello!", Some("Be concise."));
        assert!(result.contains("### System\nBe concise."));
        assert!(result.contains("### Instruction\nHello!"));
    }

    #[test]
    fn test_detect_mpt_from_name() {
        let t = TemplateType::detect(Some("mpt-7b-instruct"), None);
        assert_eq!(t, TemplateType::MptInstruct);
    }

    #[test]
    fn test_detect_mpt_from_jinja() {
        let t = TemplateType::detect(None, Some("### Instruction\n{content}\n\n### Response\n"));
        assert_eq!(t, TemplateType::MptInstruct);
    }

    #[test]
    fn test_render_chat_mpt() {
        let t = TemplateType::MptInstruct;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi!"),
            ChatTurn::new(ChatRole::User, "Bye"),
        ];
        let s = t.render_chat(&hist, Some("System.")).unwrap();
        assert!(s.contains("### System\nSystem."));
        assert!(s.contains("### Instruction\nHello"));
        assert!(s.contains("### Response\nHi!"));
        assert!(s.contains("### Instruction\nBye"));
        assert!(s.ends_with("### Response\n"));
    }

    // ── RWKV World ─────────────────────────────────────────────────────

    #[test]
    fn test_rwkv_world_template() {
        let template = TemplateType::RwkvWorld;
        let result = template.apply("Hello!", None);
        assert!(result.contains("User: Hello!"));
        assert!(result.ends_with("Assistant:"));
    }

    #[test]
    fn test_rwkv_world_with_system() {
        let template = TemplateType::RwkvWorld;
        let result = template.apply("Hello!", Some("Be concise."));
        assert!(result.contains("User: Be concise."));
        assert!(result.contains("Assistant: OK"));
        assert!(result.contains("User: Hello!"));
    }

    #[test]
    fn test_detect_rwkv_from_name() {
        let t = TemplateType::detect(Some("rwkv-5-world-3b"), None);
        assert_eq!(t, TemplateType::RwkvWorld);
    }

    #[test]
    fn test_render_chat_rwkv() {
        let t = TemplateType::RwkvWorld;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi!"),
            ChatTurn::new(ChatRole::User, "Bye"),
        ];
        let s = t.render_chat(&hist, Some("System.")).unwrap();
        assert!(s.contains("User: System."));
        assert!(s.contains("Assistant: OK"));
        assert!(s.contains("User: Hello"));
        assert!(s.contains("Assistant: Hi!"));
        assert!(s.contains("User: Bye"));
        assert!(s.ends_with("Assistant:"));
    }

    // ── OLMo Instruct ──────────────────────────────────────────────────

    #[test]
    fn test_olmo_instruct_template() {
        let template = TemplateType::OlmoInstruct;
        let result = template.apply("Hello!", None);
        assert!(result.contains("<|user|>\nHello!"));
        assert!(result.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_olmo_instruct_with_system() {
        let template = TemplateType::OlmoInstruct;
        let result = template.apply("Hello!", Some("Be concise."));
        assert!(result.contains("<|system|>\nBe concise."));
        assert!(result.contains("<|user|>\nHello!"));
    }

    #[test]
    fn test_detect_olmo_from_name() {
        let t = TemplateType::detect(Some("olmo-7b-instruct"), None);
        assert_eq!(t, TemplateType::OlmoInstruct);
    }

    #[test]
    fn test_detect_olmo_from_jinja() {
        let t = TemplateType::detect(None, Some("<|user|>\n{content}\n<|assistant|>\n"));
        assert_eq!(t, TemplateType::OlmoInstruct);
    }

    #[test]
    fn test_render_chat_olmo() {
        let t = TemplateType::OlmoInstruct;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi!"),
            ChatTurn::new(ChatRole::User, "Bye"),
        ];
        let s = t.render_chat(&hist, Some("System.")).unwrap();
        assert!(s.contains("<|system|>\nSystem."));
        assert!(s.contains("<|user|>\nHello"));
        assert!(s.contains("<|assistant|>\nHi!"));
        assert!(s.contains("<|user|>\nBye"));
        assert!(s.ends_with("<|assistant|>\n"));
    }

    // ── Detection Edge Cases ───────────────────────────────────────────

    #[test]
    fn test_detect_jinja_takes_priority_over_tokenizer_name() {
        // When both are present, jinja (GGUF chat_template) wins
        let t = TemplateType::detect(
            Some("qwen2-7b-chat"),
            Some("<|start_header_id|>user<|end_header_id|>\n{u}<|eot_id|>"),
        );
        // Jinja has LLaMA-3 signature, should override Qwen name
        assert_eq!(t, TemplateType::Llama3Chat);
    }

    #[test]
    fn test_detect_jinja_chatml_overrides_tokenizer_name() {
        let t = TemplateType::detect(
            Some("meta-llama-3-8b"),
            Some("<|im_start|>user\n{content}<|im_end|>"),
        );
        assert_eq!(t, TemplateType::Phi4Chat);
    }

    #[test]
    fn test_detect_empty_tokenizer_name_falls_back_to_raw() {
        let t = TemplateType::detect(Some(""), None);
        assert_eq!(t, TemplateType::Raw);
    }

    #[test]
    fn test_detect_none_both_falls_back_to_raw() {
        let t = TemplateType::detect(None, None);
        assert_eq!(t, TemplateType::Raw);
    }

    #[test]
    fn test_detect_empty_jinja_falls_to_tokenizer_name() {
        let t = TemplateType::detect(Some("phi-4-chat"), Some(""));
        assert_eq!(t, TemplateType::Phi4Chat);
    }

    #[test]
    fn test_detect_mixed_case_tokenizer_names() {
        assert_eq!(TemplateType::detect(Some("QWEN2-72B-CHAT"), None), TemplateType::QwenChat);
        assert_eq!(TemplateType::detect(Some("Phi-4-Mini"), None), TemplateType::Phi4Chat);
        assert_eq!(TemplateType::detect(Some("GEMMA-2-9B"), None), TemplateType::Gemma2Chat);
        assert_eq!(
            TemplateType::detect(Some("DeepSeek-V2-Lite"), None),
            TemplateType::DeepSeekChat
        );
    }

    #[test]
    fn test_detect_model_name_substrings() {
        // "instruct" in name falls back to generic Instruct
        assert_eq!(
            TemplateType::detect(Some("some-unknown-instruct-model"), None),
            TemplateType::Instruct
        );
    }

    #[test]
    fn test_detect_chatglm_jinja_variants() {
        // GLM-4 uses [gMASK] in jinja
        let t = TemplateType::detect(None, Some("[gMASK]<sop><|user|>\n{content}<|assistant|>\n"));
        assert_eq!(t, TemplateType::ChatGLMChat);
    }

    #[test]
    fn test_detect_mpt_jinja_variant() {
        let t =
            TemplateType::detect(None, Some("### Instruction\n{{ message }}\n\n### Response\n"));
        assert_eq!(t, TemplateType::MptInstruct);
    }

    #[test]
    fn test_detect_all_name_heuristics_cover_families() {
        // Ensure each family can be detected from its tokenizer name
        let cases = vec![
            ("llama3-8b", TemplateType::Llama3Chat),
            ("phi-4-mini", TemplateType::Phi4Chat),
            ("qwen2-7b", TemplateType::QwenChat),
            ("gemma-2b", TemplateType::GemmaChat),
            ("mistral-7b", TemplateType::MistralChat),
            ("deepseek-coder", TemplateType::DeepSeekChat),
            ("starcoder2-15b", TemplateType::StarCoder),
            ("falcon-40b", TemplateType::FalconChat),
            ("codellama-instruct-7b", TemplateType::CodeLlamaInstruct),
            ("cohere-command-r", TemplateType::CommandRPlus),
            ("internlm2-20b", TemplateType::InternLMChat),
            ("yi-34b-chat", TemplateType::YiChat),
            ("baichuan2-13b", TemplateType::BaichuanChat),
            ("chatglm3-6b", TemplateType::ChatGLMChat),
            ("mpt-7b-instruct", TemplateType::MptInstruct),
            ("rwkv-5-world-3b", TemplateType::RwkvWorld),
            ("olmo-7b-instruct", TemplateType::OlmoInstruct),
            ("fim-coder", TemplateType::FillInMiddle),
            ("zephyr-7b-beta", TemplateType::ZephyrChat),
            ("vicuna-13b-v1.5", TemplateType::VicunaChat),
            ("llama-2-7b-chat-hf", TemplateType::Llama2Chat),
            ("gemma-2-9b-it", TemplateType::Gemma2Chat),
            ("phi-3-mini-4k", TemplateType::Phi3Instruct),
        ];
        for (name, expected) in cases {
            let detected = TemplateType::detect(Some(name), None);
            assert_eq!(
                detected, expected,
                "Name '{}' should detect as {:?}, got {:?}",
                name, expected, detected
            );
        }
    }

    // ── Fill-in-the-Middle ──────────────────────────────────────────────

    #[test]
    fn test_fill_in_middle_template() {
        let template = TemplateType::FillInMiddle;
        let result = template.apply("def hello():", None);
        assert!(result.starts_with("<fim_prefix>"));
        assert!(result.contains("def hello():"));
        assert!(result.contains("<fim_suffix>"));
        assert!(result.ends_with("<fim_middle>"));
    }

    #[test]
    fn test_fill_in_middle_with_suffix_context() {
        let template = TemplateType::FillInMiddle;
        let result = template.apply("def hello():", Some("return 'world'"));
        assert!(result.contains("<fim_prefix>def hello():"));
        assert!(result.contains("<fim_suffix>return 'world'<fim_middle>"));
    }

    #[test]
    fn test_detect_fim_from_jinja() {
        let t = TemplateType::detect(
            None,
            Some("<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"),
        );
        assert_eq!(t, TemplateType::FillInMiddle);
    }

    #[test]
    fn test_detect_fim_from_name() {
        let t = TemplateType::detect(Some("starcoder-fim-model"), None);
        assert_eq!(t, TemplateType::FillInMiddle);
    }

    #[test]
    fn test_render_chat_fim() {
        let t = TemplateType::FillInMiddle;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "def sort():"),
            ChatTurn::new(ChatRole::Assistant, "pass"),
            ChatTurn::new(ChatRole::User, "def hello():"),
        ];
        let s = t.render_chat(&hist, Some("return 'world'")).unwrap();
        assert!(s.contains("<fim_prefix>def hello():"));
        assert!(s.contains("<fim_suffix>return 'world'<fim_middle>"));
    }

    // ── Zephyr Chat ────────────────────────────────────────────────────

    #[test]
    fn test_zephyr_chat_template() {
        let template = TemplateType::ZephyrChat;
        let result = template.apply("Hello!", None);
        assert!(result.contains("<|system|>\n"));
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains("</s>\n"));
        assert!(result.contains("<|user|>\nHello!</s>\n"));
        assert!(result.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_zephyr_chat_with_system() {
        let template = TemplateType::ZephyrChat;
        let result = template.apply("Hello!", Some("Be concise."));
        assert!(result.contains("<|system|>\nBe concise.</s>"));
        assert!(!result.contains("You are a helpful assistant."));
    }

    #[test]
    fn test_detect_zephyr_from_jinja() {
        let t = TemplateType::detect(None, Some("<|user|>\n{content}</s>\n<|assistant|>\n"));
        assert_eq!(t, TemplateType::ZephyrChat);
    }

    #[test]
    fn test_detect_zephyr_from_name() {
        let t = TemplateType::detect(Some("zephyr-7b-beta"), None);
        assert_eq!(t, TemplateType::ZephyrChat);
    }

    #[test]
    fn test_render_chat_zephyr() {
        let t = TemplateType::ZephyrChat;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi!"),
            ChatTurn::new(ChatRole::User, "Bye"),
        ];
        let s = t.render_chat(&hist, Some("Be helpful.")).unwrap();
        assert!(s.contains("<|system|>\nBe helpful.</s>"));
        assert!(s.contains("<|user|>\nHello</s>"));
        assert!(s.contains("<|assistant|>\nHi!</s>"));
        assert!(s.contains("<|user|>\nBye</s>"));
        assert!(s.ends_with("<|assistant|>\n"));
    }

    // ── Vicuna Chat ────────────────────────────────────────────────────

    #[test]
    fn test_vicuna_chat_template() {
        let template = TemplateType::VicunaChat;
        let result = template.apply("Hello!", None);
        assert!(result.contains("A chat between a curious user"));
        assert!(result.contains("USER: Hello!"));
        assert!(result.ends_with("\nASSISTANT:"));
    }

    #[test]
    fn test_vicuna_chat_with_system() {
        let template = TemplateType::VicunaChat;
        let result = template.apply("Hello!", Some("Be concise."));
        assert!(result.starts_with("Be concise."));
        assert!(!result.contains("A chat between a curious user"));
        assert!(result.contains("USER: Hello!"));
    }

    #[test]
    fn test_detect_vicuna_from_jinja() {
        let t = TemplateType::detect(None, Some("USER: {content}\nASSISTANT:"));
        assert_eq!(t, TemplateType::VicunaChat);
    }

    #[test]
    fn test_detect_vicuna_from_name() {
        let t = TemplateType::detect(Some("vicuna-13b-v1.5"), None);
        assert_eq!(t, TemplateType::VicunaChat);
    }

    #[test]
    fn test_detect_sharegpt_from_name() {
        let t = TemplateType::detect(Some("sharegpt-model"), None);
        assert_eq!(t, TemplateType::VicunaChat);
    }

    #[test]
    fn test_render_chat_vicuna() {
        let t = TemplateType::VicunaChat;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi!"),
            ChatTurn::new(ChatRole::User, "Bye"),
        ];
        let s = t.render_chat(&hist, Some("Be helpful.")).unwrap();
        assert!(s.starts_with("Be helpful.\n\n"));
        assert!(s.contains("USER: Hello"));
        assert!(s.contains("ASSISTANT: Hi!"));
        assert!(s.contains("USER: Bye"));
        assert!(s.ends_with("ASSISTANT:"));
    }

    // ── Orca Chat ──────────────────────────────────────────────────

    #[test]
    fn test_orca_chat_template() {
        let template = TemplateType::OrcaChat;
        let result = template.apply("Hello!", None);
        assert!(result.contains("<|im_start|>system\n"));
        assert!(result.contains("You are Orca"));
        assert!(result.contains("<|im_end|>"));
        assert!(result.contains("<|im_start|>user\n"));
        assert!(result.contains("Hello!"));
        assert!(result.ends_with("<|im_start|>assistant\n"));

        let result = template.apply("Hello!", Some("Custom system."));
        assert!(result.contains("Custom system."));
        assert!(!result.contains("You are Orca"));
    }

    #[test]
    fn test_detect_orca_from_name() {
        let t = TemplateType::detect(Some("orca-mini-3b"), None);
        assert_eq!(t, TemplateType::OrcaChat);
    }

    #[test]
    fn test_render_chat_orca() {
        let t = TemplateType::OrcaChat;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi!"),
            ChatTurn::new(ChatRole::User, "Bye"),
        ];
        let s = t.render_chat(&hist, Some("Be helpful.")).unwrap();
        assert!(s.contains("<|im_start|>system\nBe helpful.<|im_end|>"));
        assert!(s.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(s.contains("<|im_start|>assistant\nHi!<|im_end|>"));
        assert!(s.ends_with("<|im_start|>assistant\n"));
    }

    // ── Solar Instruct ─────────────────────────────────────────────

    #[test]
    fn test_solar_instruct_template() {
        let template = TemplateType::SolarInstruct;
        let result = template.apply("Hello!", None);
        assert!(result.contains("### User:\nHello!"));
        assert!(result.contains("### Assistant:\n"));
        assert!(!result.contains("### System:"));

        let result = template.apply("Hello!", Some("Be brief."));
        assert!(result.contains("### System:\nBe brief."));
        assert!(result.contains("### User:\nHello!"));
    }

    #[test]
    fn test_detect_solar_from_jinja() {
        let t = TemplateType::detect(None, Some("### User: {content}\n### Assistant:"));
        assert_eq!(t, TemplateType::SolarInstruct);
    }

    #[test]
    fn test_detect_solar_from_name() {
        let t = TemplateType::detect(Some("solar-10.7b"), None);
        assert_eq!(t, TemplateType::SolarInstruct);
    }

    #[test]
    fn test_render_chat_solar() {
        let t = TemplateType::SolarInstruct;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi!"),
            ChatTurn::new(ChatRole::User, "Bye"),
        ];
        let s = t.render_chat(&hist, Some("Be helpful.")).unwrap();
        assert!(s.contains("### System:\nBe helpful."));
        assert!(s.contains("### User:\nHello"));
        assert!(s.contains("### Assistant:\nHi!"));
        assert!(s.contains("### User:\nBye"));
        assert!(s.ends_with("### Assistant:\n"));
    }

    // ── Alpaca Instruct ────────────────────────────────────────────

    #[test]
    fn test_alpaca_instruct_template() {
        let template = TemplateType::AlpacaInstruct;
        let result = template.apply("Hello!", None);
        assert!(result.contains("Below is an instruction"));
        assert!(result.contains("### Instruction:\nHello!"));
        assert!(result.contains("### Response:\n"));

        let result = template.apply("Hello!", Some("Custom system."));
        assert!(result.starts_with("Custom system."));
        assert!(!result.contains("Below is an instruction"));
    }

    #[test]
    fn test_detect_alpaca_from_jinja() {
        let t = TemplateType::detect(
            None,
            Some("### Instruction: {content}\n### Response:"),
        );
        assert_eq!(t, TemplateType::AlpacaInstruct);
    }

    #[test]
    fn test_detect_alpaca_from_name() {
        let t = TemplateType::detect(Some("alpaca-7b"), None);
        assert_eq!(t, TemplateType::AlpacaInstruct);
    }

    #[test]
    fn test_render_chat_alpaca() {
        let t = TemplateType::AlpacaInstruct;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi!"),
            ChatTurn::new(ChatRole::User, "Bye"),
        ];
        let s = t.render_chat(&hist, Some("Be helpful.")).unwrap();
        assert!(s.contains("Be helpful."));
        assert!(s.contains("### Instruction:\nHello"));
        assert!(s.contains("### Response:\nHi!"));
        assert!(s.contains("### Instruction:\nBye"));
        assert!(s.ends_with("### Response:\n"));
    }

    // ── CommandRPlus tests ─────────────────────────────────────────

    #[test]
    fn test_command_r_plus_template() {
        let t = TemplateType::CommandRPlus;
        let result = t.apply("Hello!", None);
        assert!(result.contains("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>"));
        assert!(result.contains("Command-R+"));
        assert!(result.contains("<|END_OF_TURN_TOKEN|>"));
        assert!(result.contains("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello!"));
        assert!(result.ends_with("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"));

        let result = t.apply("Hello!", Some("Custom system."));
        assert!(result.contains("Custom system."));
        assert!(!result.contains("Command-R+"));
    }

    #[test]
    fn test_detect_command_r_plus_from_jinja() {
        let t = TemplateType::detect(
            None,
            Some("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user}<|END_OF_TURN_TOKEN|>"),
        );
        assert_eq!(t, TemplateType::CommandRPlus);
    }

    #[test]
    fn test_detect_command_r_plus_from_name() {
        let t = TemplateType::detect(Some("command-r-plus"), None);
        assert_eq!(t, TemplateType::CommandRPlus);
    }

    #[test]
    fn test_render_chat_command_r_plus() {
        let t = TemplateType::CommandRPlus;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi!"),
            ChatTurn::new(ChatRole::User, "Bye"),
        ];
        let s = t.render_chat(&hist, Some("Be helpful.")).unwrap();
        assert!(s.contains("Be helpful."));
        assert!(s.contains("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello"));
        assert!(s.contains("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Hi!"));
        assert!(s.ends_with("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"));
    }

    // ── NousHermes tests ───────────────────────────────────────────

    #[test]
    fn test_nous_hermes_template() {
        let t = TemplateType::NousHermes;
        let result = t.apply("Hello!", None);
        assert!(result.contains("<|im_start|>system\n"));
        assert!(result.contains("helpful, honest and harmless"));
        assert!(result.contains("<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHello!"));
        assert!(result.ends_with("<|im_start|>assistant\n"));

        let result = t.apply("Hello!", Some("Custom."));
        assert!(result.contains("Custom."));
        assert!(!result.contains("helpful, honest and harmless"));
    }

    #[test]
    fn test_detect_nous_hermes_from_name() {
        let t = TemplateType::detect(Some("nous-hermes-2"), None);
        assert_eq!(t, TemplateType::NousHermes);
    }

    #[test]
    fn test_render_chat_nous_hermes() {
        let t = TemplateType::NousHermes;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi!"),
            ChatTurn::new(ChatRole::User, "Bye"),
        ];
        let s = t.render_chat(&hist, Some("Be helpful.")).unwrap();
        assert!(s.contains("Be helpful."));
        assert!(s.contains("<|im_start|>user\nHello"));
        assert!(s.contains("<|im_start|>assistant\nHi!"));
        assert!(s.ends_with("<|im_start|>assistant\n"));
    }

    // ── WizardLM tests ─────────────────────────────────────────────

    #[test]
    fn test_wizard_lm_template() {
        let t = TemplateType::WizardLM;
        let result = t.apply("Hello!", None);
        assert!(result.contains("A chat between a curious user"));
        assert!(result.contains("USER: Hello!"));
        assert!(result.ends_with("\nASSISTANT: "));

        let result = t.apply("Hello!", Some("Custom."));
        assert!(result.contains("Custom."));
        assert!(result.contains("USER: Hello!"));
    }

    #[test]
    fn test_detect_wizard_lm_from_jinja() {
        let t = TemplateType::detect(
            None,
            Some("A chat between USER: text ASSISTANT: response"),
        );
        assert_eq!(t, TemplateType::WizardLM);
    }

    #[test]
    fn test_detect_wizard_lm_from_name() {
        let t = TemplateType::detect(Some("wizard-lm-13b"), None);
        assert_eq!(t, TemplateType::WizardLM);
    }

    #[test]
    fn test_render_chat_wizard_lm() {
        let t = TemplateType::WizardLM;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi!"),
            ChatTurn::new(ChatRole::User, "Bye"),
        ];
        let s = t.render_chat(&hist, Some("Be helpful.")).unwrap();
        assert!(s.contains("Be helpful."));
        assert!(s.contains("USER: Hello"));
        assert!(s.contains("ASSISTANT: Hi!"));
        assert!(s.ends_with("ASSISTANT: "));
    }

    // ── OpenChat tests ─────────────────────────────────────────────

    #[test]
    fn test_openchat_template() {
        let t = TemplateType::OpenChat;
        let result = t.apply("Hello!", None);
        assert!(result.starts_with("GPT4 Correct User: Hello!"));
        assert!(result.contains("<|end_of_turn|>"));
        assert!(result.ends_with("GPT4 Correct Assistant:"));

        let result = t.apply("Hello!", Some("Be nice."));
        assert!(result.contains("Be nice."));
        assert!(result.contains("Hello!"));
    }

    #[test]
    fn test_detect_openchat_from_jinja() {
        let t = TemplateType::detect(
            None,
            Some("GPT4 Correct User: {text}<|end_of_turn|>GPT4 Correct Assistant:"),
        );
        assert_eq!(t, TemplateType::OpenChat);
    }

    #[test]
    fn test_detect_openchat_from_name() {
        let t = TemplateType::detect(Some("openchat-3.5"), None);
        assert_eq!(t, TemplateType::OpenChat);
    }

    #[test]
    fn test_render_chat_openchat() {
        let t = TemplateType::OpenChat;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi!"),
            ChatTurn::new(ChatRole::User, "Bye"),
        ];
        let s = t.render_chat(&hist, None).unwrap();
        assert!(s.contains("GPT4 Correct User: Hello"));
        assert!(s.contains("GPT4 Correct Assistant: Hi!"));
        assert!(s.ends_with("GPT4 Correct Assistant:"));
    }
}
