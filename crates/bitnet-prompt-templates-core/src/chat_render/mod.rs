//! Single-responsibility chat history rendering for [`TemplateType`].
//!
//! This module owns multi-turn chat formatting so the template enum impl can
//! remain focused on public API dispatch and metadata.

mod common;

use crate::{ChatRole, ChatTurn, TemplateType};
use anyhow::Result;
use common::{render_chatml, render_qwen25_chatml};

pub(crate) fn render_chat(
    template: TemplateType,
    history: &[ChatTurn],
    system: Option<&str>,
) -> Result<String> {
    use std::fmt::Write as _;
    let mut out = String::new();

    match template {
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
        TemplateType::BitnetCppAnswer => {
            if let Some(sys) = system.filter(|sys| !sys.trim().is_empty()) {
                write!(out, "System: {}<|eot_id|>", sys)?;
            }

            for turn in history {
                match turn.role {
                    ChatRole::User => write!(out, "User: {}<|eot_id|>", turn.text)?,
                    ChatRole::Assistant => write!(out, "Assistant: {}<|eot_id|>", turn.text)?,
                    ChatRole::System => write!(out, "System: {}<|eot_id|>", turn.text)?,
                }
            }

            write!(out, "Assistant: ")?;
        }
        TemplateType::Phi4Chat => {
            // ChatML format with im_start/im_end tokens
            let sys = system.unwrap_or("You are a helpful assistant.");
            out = render_chatml(sys, history);
        }
        TemplateType::QwenChat => {
            // ChatML format with im_start/im_end tokens
            let sys = system.unwrap_or("You are a helpful assistant.");
            out = render_chatml(sys, history);
        }
        TemplateType::GemmaChat => {
            // Gemma format with start_of_turn/end_of_turn tokens
            // Gemma has no system role ΓÇö prepend to first user turn
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
            out = render_chatml(sys, history);
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
            out = render_chatml(sys, history);
        }
        TemplateType::YiChat => {
            // Yi ChatML format (same as Phi4/Qwen)
            let sys = system.unwrap_or("You are a helpful assistant.");
            out = render_chatml(sys, history);
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
            out = render_chatml(sys, history);
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
            let sys = system.unwrap_or("You are a helpful, honest and harmless AI assistant.");
            out = render_chatml(sys, history);
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
        TemplateType::OpenChat => {
            // OpenChat GPT4 Correct format
            for turn in history {
                match turn.role {
                    ChatRole::User => {
                        write!(out, "GPT4 Correct User: {}<|end_of_turn|>", turn.text)?;
                    }
                    ChatRole::Assistant => {
                        write!(out, "GPT4 Correct Assistant: {}<|end_of_turn|>", turn.text)?;
                    }
                    ChatRole::System => {}
                }
            }
            write!(out, "GPT4 Correct Assistant:")?;
        }
        TemplateType::GraniteChat => {
            // Granite role-token format
            let sys = system.unwrap_or("You are Granite, an AI language model developed by IBM.");
            writeln!(out, "<|start_of_role|>system<|end_of_role|>{}", sys)?;
            for turn in history {
                let role = turn.role.as_str();
                writeln!(out, "<|start_of_role|>{}<|end_of_role|>{}", role, turn.text)?;
            }
            write!(out, "<|start_of_role|>assistant<|end_of_role|>")?;
        }
        TemplateType::NemotronChat => {
            // Nemotron extra_id format
            let sys = system.unwrap_or("You are a helpful, respectful and honest assistant.");
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
                    "╨ó╤ï ΓÇö ╨í╨░╨╣╨│╨░, ╤Ç╤â╤ü╤ü╨║╨╛╤Å╨╖╤ï╤ç╨╜╤ï╨╣ ╨░╨▓╤é╨╛╨╝╨░╤é╨╕╤ç╨╡╤ü╨║╨╕╨╣ ╨░╤ü╤ü╨╕╤ü╤é╨╡╨╜╤é. \
                     ╨ó╤ï ╤Ç╨░╨╖╨│╨╛╨▓╨░╤Ç╨╕╨▓╨░╨╡╤ê╤î ╤ü ╨╗╤Ä╨┤╤î╨╝╨╕ ╨╕ ╨┐╨╛╨╝╨╛╨│╨░╨╡╤ê╤î ╨╕╨╝.",
                );
            out = render_chatml(sys, history);
        }
        TemplateType::Llama2Chat => {
            // Llama-2 [INST]<<SYS>>/<</SYS>> format
            let sys = system.unwrap_or("You are a helpful, respectful and honest assistant.");
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
                write!(out, "[INST] <<SYS>>\n{}\n<</SYS>>\n\n [/INST] ", sys)?;
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
        TemplateType::TinyLlamaChat => {
            let sys = system
                .unwrap_or("You are a friendly chatbot who always responds in a helpful manner.");
            out = render_chatml(sys, history);
        }
        TemplateType::DolphinChat => {
            let sys = system.unwrap_or("You are Dolphin, a helpful AI assistant.");
            out = render_chatml(sys, history);
        }
        TemplateType::ChatGptChat => {
            let sys = system.unwrap_or("You are a helpful assistant.");
            out = render_chatml(sys, history);
        }
        TemplateType::MixtralInstruct => {
            // Mixtral uses same [INST] format as Mistral
            out.push_str("<s>");
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
            if let Some(sys) = system {
                write!(out, "[INST] {}\n\n", sys)?;
            } else {
                write!(out, "[INST] ")?;
            }
        }
        TemplateType::StableLMChat => {
            let sys = system.unwrap_or("You are a helpful, respectful and honest assistant.");
            out = render_chatml(sys, history);
        }
        TemplateType::BloomChat => {
            if let Some(sys) = system {
                write!(out, "{}\n\n", sys)?;
            }
            for turn in history {
                match turn.role {
                    ChatRole::User => {
                        writeln!(out, "User: {}", turn.text)?;
                    }
                    ChatRole::Assistant => {
                        writeln!(out, "Assistant: {}", turn.text)?;
                    }
                    ChatRole::System => {}
                }
            }
            write!(out, "Assistant: ")?;
        }
        TemplateType::JambaChat => {
            let sys = system.unwrap_or("You are Jamba, a helpful AI assistant made by AI21 Labs.");
            out = render_chatml(sys, history);
        }
        TemplateType::PersimmonChat => {
            if let Some(sys) = system {
                write!(out, "{}\n\n", sys)?;
            }
            for turn in history {
                match turn.role {
                    ChatRole::User => {
                        writeln!(out, "human: {}", turn.text)?;
                    }
                    ChatRole::Assistant => {
                        writeln!(out, "adept: {}", turn.text)?;
                    }
                    ChatRole::System => {}
                }
            }
            write!(out, "adept: ")?;
        }
        TemplateType::XverseChat => {
            if let Some(sys) = system {
                write!(out, "{}\n\n", sys)?;
            }
            for turn in history {
                match turn.role {
                    ChatRole::User => {
                        write!(out, "Human: {}\n\n", turn.text)?;
                    }
                    ChatRole::Assistant => {
                        write!(out, "Assistant: {}\n\n", turn.text)?;
                    }
                    ChatRole::System => {}
                }
            }
            write!(out, "Assistant: ")?;
        }
        TemplateType::Qwen25Chat => {
            let sys = system
                .unwrap_or("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.");
            out = render_qwen25_chatml(sys, history);
        }
        TemplateType::MistralNemoChat => {
            // Mistral Nemo uses same [INST] format as Mistral
            out.push_str("<s>");
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
            if let Some(sys) = system {
                write!(out, "[INST] {}\n\n", sys)?;
            } else {
                write!(out, "[INST] ")?;
            }
        }
        TemplateType::ArcticInstruct => {
            let sys = system.unwrap_or("You are a helpful AI assistant.");
            out = render_chatml(sys, history);
        }
        TemplateType::DbrxInstruct => {
            let sys = system
                .unwrap_or("You are DBRX, created by Databricks. You are a helpful assistant.");
            out = render_chatml(sys, history);
        }
        TemplateType::ExaoneChat => {
            let sys =
                system.unwrap_or("You are EXAONE model from LG AI Research, a helpful assistant.");
            writeln!(out, "[|system|]{}[|endofturn|]", sys)?;
            for turn in history {
                match turn.role {
                    ChatRole::User => {
                        write!(out, "[|user|]{}\n[|endofturn|]\n", turn.text)?;
                    }
                    ChatRole::Assistant => {
                        write!(out, "[|assistant|]{}\n[|endofturn|]\n", turn.text)?;
                    }
                    ChatRole::System => {}
                }
            }
            write!(out, "[|assistant|]")?;
        }
        TemplateType::MiniCPMChat => {
            let sys = system.unwrap_or("You are a helpful assistant.");
            out = render_chatml(sys, history);
        }
        TemplateType::CodeGemma => {
            // CodeGemma uses same format as Gemma
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
        TemplateType::Llama31Chat => {
            // Llama 3.1 format with <|begin_of_text|> prefix
            out.push_str("<|begin_of_text|>");

            let sys = system.unwrap_or("You are a helpful, harmless, and honest AI assistant.");
            write!(out, "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>", sys)?;

            for turn in history {
                let role = turn.role.as_str();
                write!(
                    out,
                    "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                    role, turn.text
                )?;
            }

            write!(out, "<|start_header_id|>assistant<|end_header_id|>\n\n")?;
        }
        TemplateType::DeepSeekV3Chat => {
            let sys =
                system.unwrap_or("You are DeepSeek Chat, a helpful and harmless AI assistant.");
            out = render_chatml(sys, history);
        }
        TemplateType::Falcon2Chat => {
            let sys = system.unwrap_or("You are a helpful assistant.");
            out = render_chatml(sys, history);
        }
        TemplateType::OLMo2Chat => {
            let sys = system.unwrap_or("You are OLMo 2, a helpful AI assistant.");
            out = render_chatml(sys, history);
        }
        TemplateType::Llama32Chat => {
            out.push_str("<|begin_of_text|>");

            let sys = system.unwrap_or("You are a helpful, harmless, and honest AI assistant.");
            write!(out, "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>", sys)?;

            for turn in history {
                let role = turn.role.as_str();
                write!(
                    out,
                    "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                    role, turn.text
                )?;
            }

            write!(out, "<|start_header_id|>assistant<|end_header_id|>\n\n")?;
        }
        TemplateType::CohereAya => {
            let sys =
                system.unwrap_or("You are Aya, a multilingual AI assistant created by Cohere.");
            out = render_chatml(sys, history);
        }
        TemplateType::SmolLMChat => {
            let sys = system.unwrap_or("You are a helpful AI assistant.");
            out = render_chatml(sys, history);
        }
        TemplateType::Phi2Instruct => {
            if let Some(sys) = system {
                out.push_str(sys);
                out.push_str("\n\n");
            }
            for turn in history {
                match turn.role {
                    ChatRole::User => {
                        writeln!(out, "Instruct: {}", turn.text)?;
                    }
                    ChatRole::Assistant => {
                        writeln!(out, "Output: {}", turn.text)?;
                    }
                    ChatRole::System => {
                        writeln!(out, "{}", turn.text)?;
                    }
                }
            }
            out.push_str("Output: ");
        }
    }

    Ok(out)
}
