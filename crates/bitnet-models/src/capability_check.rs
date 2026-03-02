//! Model capability checking.
//!
//! Determines what operations a loaded model supports.

use std::collections::HashSet;

/// Capabilities a model may have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelCapability {
    TextGeneration,
    ChatCompletion,
    CodeGeneration,
    FillInMiddle,
    Embedding,
    Classification,
    ToolUse,
    VisionInput,
    AudioInput,
}

impl ModelCapability {
    pub fn name(&self) -> &'static str {
        match self {
            Self::TextGeneration => "text_generation",
            Self::ChatCompletion => "chat_completion",
            Self::CodeGeneration => "code_generation",
            Self::FillInMiddle => "fill_in_middle",
            Self::Embedding => "embedding",
            Self::Classification => "classification",
            Self::ToolUse => "tool_use",
            Self::VisionInput => "vision_input",
            Self::AudioInput => "audio_input",
        }
    }
}

/// Result of capability analysis.
#[derive(Debug, Clone)]
pub struct CapabilityReport {
    pub capabilities: HashSet<ModelCapability>,
    pub model_family: String,
    pub notes: Vec<String>,
}

impl CapabilityReport {
    pub fn has(&self, cap: ModelCapability) -> bool {
        self.capabilities.contains(&cap)
    }

    pub fn can_chat(&self) -> bool {
        self.has(ModelCapability::ChatCompletion)
    }

    pub fn can_code(&self) -> bool {
        self.has(ModelCapability::CodeGeneration) || self.has(ModelCapability::FillInMiddle)
    }
}

/// Detect capabilities from model family name.
pub fn detect_capabilities(model_family: &str) -> CapabilityReport {
    let family = model_family.to_lowercase();
    let mut caps = HashSet::new();
    let mut notes = Vec::new();

    // All LLMs can do text generation
    caps.insert(ModelCapability::TextGeneration);

    // Chat models
    let chat_families = [
        "phi", "phi2", "phi3", "phi4", "llama", "llama2", "llama3", "mistral", "mixtral", "qwen",
        "qwen2", "gemma", "gemma2", "falcon", "yi", "internlm", "deepseek", "baichuan",
    ];
    if chat_families.iter().any(|&f| family.contains(f)) {
        caps.insert(ModelCapability::ChatCompletion);
    }

    // Code models
    let code_families =
        ["codellama", "starcoder", "deepseek-coder", "phi", "qwen", "granite-code", "codegemma"];
    if code_families.iter().any(|&f| family.contains(f)) {
        caps.insert(ModelCapability::CodeGeneration);
    }

    // Fill-in-middle
    let fim_families = ["codellama", "starcoder", "deepseek-coder", "codegemma"];
    if fim_families.iter().any(|&f| family.contains(f)) {
        caps.insert(ModelCapability::FillInMiddle);
        notes.push("supports fill-in-middle with special tokens".into());
    }

    // Tool use
    if family.contains("mistral") || family.contains("llama3") || family.contains("qwen2") {
        caps.insert(ModelCapability::ToolUse);
        notes.push("may support function calling".into());
    }

    // Vision
    if family.contains("llava") || family.contains("phi3-vision") || family.contains("qwen-vl") {
        caps.insert(ModelCapability::VisionInput);
    }

    CapabilityReport { capabilities: caps, model_family: model_family.to_string(), notes }
}

/// Check if model meets minimum requirements.
pub fn check_requirements(hidden_size: usize, num_layers: usize, vocab_size: usize) -> Vec<String> {
    let mut issues = Vec::new();

    if hidden_size < 64 {
        issues.push(format!("hidden_size {hidden_size} is very small (min recommended: 64)"));
    }
    if num_layers == 0 {
        issues.push("num_layers is 0".into());
    }
    if vocab_size < 100 {
        issues.push(format!("vocab_size {vocab_size} is very small (min recommended: 100)"));
    }
    if !hidden_size.is_multiple_of(2) {
        issues.push(format!("hidden_size {hidden_size} is odd (should be even for attention)"));
    }

    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_name() {
        assert_eq!(ModelCapability::TextGeneration.name(), "text_generation");
        assert_eq!(ModelCapability::ChatCompletion.name(), "chat_completion");
    }

    #[test]
    fn test_detect_phi4() {
        let r = detect_capabilities("phi4");
        assert!(r.has(ModelCapability::TextGeneration));
        assert!(r.has(ModelCapability::ChatCompletion));
        assert!(r.has(ModelCapability::CodeGeneration));
    }

    #[test]
    fn test_detect_llama3() {
        let r = detect_capabilities("llama3");
        assert!(r.has(ModelCapability::ChatCompletion));
        assert!(r.has(ModelCapability::ToolUse));
    }

    #[test]
    fn test_detect_starcoder() {
        let r = detect_capabilities("starcoder");
        assert!(r.has(ModelCapability::CodeGeneration));
        assert!(r.has(ModelCapability::FillInMiddle));
    }

    #[test]
    fn test_can_chat() {
        let r = detect_capabilities("mistral");
        assert!(r.can_chat());
    }

    #[test]
    fn test_can_code() {
        let r = detect_capabilities("codellama");
        assert!(r.can_code());
    }

    #[test]
    fn test_unknown_model() {
        let r = detect_capabilities("unknown_model");
        assert!(r.has(ModelCapability::TextGeneration)); // always
        assert!(!r.has(ModelCapability::ChatCompletion));
    }

    #[test]
    fn test_vision_model() {
        let r = detect_capabilities("llava");
        assert!(r.has(ModelCapability::VisionInput));
    }

    #[test]
    fn test_notes() {
        let r = detect_capabilities("codellama");
        assert!(!r.notes.is_empty());
    }

    #[test]
    fn test_requirements_ok() {
        let issues = check_requirements(4096, 32, 32000);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_requirements_small_hidden() {
        let issues = check_requirements(32, 4, 1000);
        assert!(issues.iter().any(|i| i.contains("hidden_size")));
    }

    #[test]
    fn test_requirements_zero_layers() {
        let issues = check_requirements(4096, 0, 32000);
        assert!(issues.iter().any(|i| i.contains("num_layers")));
    }

    #[test]
    fn test_requirements_odd_hidden() {
        let issues = check_requirements(4097, 32, 32000);
        assert!(issues.iter().any(|i| i.contains("odd")));
    }

    #[test]
    fn test_qwen_tool_use() {
        let r = detect_capabilities("qwen2");
        assert!(r.has(ModelCapability::ToolUse));
    }
}
