//! xtask library for shared functionality
//!
//! This library exposes modules that are shared between the xtask binary
//! and other crates (including tests).

pub mod build_helpers;
pub mod cpp_setup_auto;
pub mod crossval;
pub mod ffi;

// Prompt template argument type for cross-validation
// This mirrors the PromptTemplateArg in main.rs but is available to the library
#[cfg(feature = "inference")]
#[derive(Clone, Copy, Debug)]
pub enum PromptTemplateArg {
    /// Auto-detect from GGUF metadata or tokenizer
    Auto,
    /// Raw text (no formatting)
    Raw,
    /// Q&A instruction format
    Instruct,
    /// LLaMA-3 chat format with special tokens
    Llama3Chat,
}

#[cfg(feature = "inference")]
impl PromptTemplateArg {
    /// Convert to TemplateType
    pub fn to_template_type(&self) -> bitnet_inference::prompt_template::TemplateType {
        use bitnet_inference::prompt_template::TemplateType;
        match self {
            Self::Auto => TemplateType::Raw, // Placeholder - will add auto-detection
            Self::Raw => TemplateType::Raw,
            Self::Instruct => TemplateType::Instruct,
            Self::Llama3Chat => TemplateType::Llama3Chat,
        }
    }
}
