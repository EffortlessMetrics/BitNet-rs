//! Backend selection for cross-validation
//!
//! This module defines the C++ backend type for identifying which reference
//! implementation is being used during cross-validation.

use std::fmt;

/// C++ backend for cross-validation
///
/// Identifies which C++ reference implementation is being used for comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CppBackend {
    /// BitNet C++ reference (bitnet.cpp)
    ///
    /// Used for BitNet models (microsoft-bitnet-b1.58-2B-4T-gguf)
    BitNet,

    /// LLaMA C++ reference (llama.cpp)
    ///
    /// Used for LLaMA models (llama-3, llama-2, SmolLM3, etc.)
    Llama,
}

impl CppBackend {
    /// Get backend name for diagnostics
    ///
    /// # Examples
    ///
    /// ```
    /// use bitnet_crossval::backend::CppBackend;
    ///
    /// assert_eq!(CppBackend::BitNet.name(), "BitNet");
    /// assert_eq!(CppBackend::Llama.name(), "LLaMA");
    /// ```
    pub fn name(&self) -> &'static str {
        match self {
            Self::BitNet => "BitNet",
            Self::Llama => "LLaMA",
        }
    }

    /// Get detailed backend name with implementation
    ///
    /// # Examples
    ///
    /// ```
    /// use bitnet_crossval::backend::CppBackend;
    ///
    /// assert_eq!(CppBackend::BitNet.full_name(), "BitNet (bitnet.cpp)");
    /// assert_eq!(CppBackend::Llama.full_name(), "LLaMA (llama.cpp)");
    /// ```
    pub fn full_name(&self) -> &'static str {
        match self {
            Self::BitNet => "BitNet (bitnet.cpp)",
            Self::Llama => "LLaMA (llama.cpp)",
        }
    }
}

impl fmt::Display for CppBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// Conversion from string representation (for interop with xtask)
impl CppBackend {
    /// Parse backend from string name
    ///
    /// # Examples
    ///
    /// ```
    /// use bitnet_crossval::backend::CppBackend;
    ///
    /// assert_eq!(CppBackend::from_name("bitnet"), Some(CppBackend::BitNet));
    /// assert_eq!(CppBackend::from_name("llama"), Some(CppBackend::Llama));
    /// assert_eq!(CppBackend::from_name("unknown"), None);
    /// ```
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "bitnet" => Some(Self::BitNet),
            "llama" => Some(Self::Llama),
            _ => None,
        }
    }

    /// Get setup command for this backend
    ///
    /// Returns the command users should run to set up the C++ reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitnet_crossval::backend::CppBackend;
    ///
    /// let cmd = CppBackend::BitNet.setup_command();
    /// assert!(cmd.contains("setup-cpp-auto"));
    /// ```
    pub fn setup_command(&self) -> &'static str {
        match self {
            Self::BitNet => "eval \"$(cargo run -p xtask -- setup-cpp-auto --bitnet --emit=sh)\"",
            Self::Llama => "eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh)\"",
        }
    }

    /// Get required library patterns for preflight checks
    ///
    /// Returns library name prefixes that must be found during build.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitnet_crossval::backend::CppBackend;
    ///
    /// assert_eq!(CppBackend::BitNet.required_libs(), &["libbitnet"]);
    /// assert_eq!(CppBackend::Llama.required_libs(), &["libllama", "libggml"]);
    /// ```
    pub fn required_libs(&self) -> &[&'static str] {
        match self {
            Self::BitNet => &["libbitnet"],
            Self::Llama => &["libllama", "libggml"],
        }
    }

    /// Auto-detect backend from model path
    ///
    /// Detection heuristics:
    /// - Path contains "bitnet" or "microsoft/bitnet" → BitNet
    /// - Path contains "llama" → Llama
    /// - Default: Llama (more common, safer fallback)
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    /// use bitnet_crossval::backend::CppBackend;
    ///
    /// let path = Path::new("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf");
    /// assert_eq!(CppBackend::from_model_path(path), CppBackend::BitNet);
    ///
    /// let path = Path::new("models/llama-3-8b-instruct.gguf");
    /// assert_eq!(CppBackend::from_model_path(path), CppBackend::Llama);
    /// ```
    pub fn from_model_path<P: AsRef<std::path::Path>>(path: P) -> Self {
        let path_str = path.as_ref().to_string_lossy().to_lowercase();

        if path_str.contains("bitnet") || path_str.contains("microsoft/bitnet") {
            Self::BitNet
        } else if path_str.contains("llama") {
            Self::Llama
        } else {
            // Conservative default: llama (more common, better tested)
            Self::Llama
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_names() {
        assert_eq!(CppBackend::BitNet.name(), "BitNet");
        assert_eq!(CppBackend::Llama.name(), "LLaMA");

        assert_eq!(CppBackend::BitNet.full_name(), "BitNet (bitnet.cpp)");
        assert_eq!(CppBackend::Llama.full_name(), "LLaMA (llama.cpp)");
    }

    #[test]
    fn test_backend_display() {
        assert_eq!(format!("{}", CppBackend::BitNet), "BitNet");
        assert_eq!(format!("{}", CppBackend::Llama), "LLaMA");
    }

    #[test]
    fn test_backend_setup_commands() {
        let bitnet_cmd = CppBackend::BitNet.setup_command();
        assert!(bitnet_cmd.contains("setup-cpp-auto"));
        assert!(bitnet_cmd.contains("--bitnet"));

        let llama_cmd = CppBackend::Llama.setup_command();
        assert!(llama_cmd.contains("setup-cpp-auto"));
    }

    #[test]
    fn test_backend_required_libs() {
        assert_eq!(CppBackend::BitNet.required_libs(), &["libbitnet"]);
        assert_eq!(CppBackend::Llama.required_libs(), &["libllama", "libggml"]);
    }

    #[test]
    fn test_from_name() {
        assert_eq!(CppBackend::from_name("bitnet"), Some(CppBackend::BitNet));
        assert_eq!(CppBackend::from_name("BITNET"), Some(CppBackend::BitNet));
        assert_eq!(CppBackend::from_name("llama"), Some(CppBackend::Llama));
        assert_eq!(CppBackend::from_name("LLAMA"), Some(CppBackend::Llama));
        assert_eq!(CppBackend::from_name("unknown"), None);
    }

    #[test]
    fn test_from_model_path_bitnet() {
        use std::path::Path;

        // Test BitNet model path patterns
        let path = Path::new("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf");
        assert_eq!(CppBackend::from_model_path(path), CppBackend::BitNet);

        let path = Path::new("models/bitnet/model.gguf");
        assert_eq!(CppBackend::from_model_path(path), CppBackend::BitNet);

        let path = Path::new("/path/to/microsoft-bitnet/model.gguf");
        assert_eq!(CppBackend::from_model_path(path), CppBackend::BitNet);

        let path = Path::new("bitnet-b1.58-large.gguf");
        assert_eq!(CppBackend::from_model_path(path), CppBackend::BitNet);
    }

    #[test]
    fn test_from_model_path_llama() {
        use std::path::Path;

        // Test LLaMA model path patterns
        let path = Path::new("models/llama-3-8b-instruct.gguf");
        assert_eq!(CppBackend::from_model_path(path), CppBackend::Llama);

        let path = Path::new("models/llama/ggml-model.gguf");
        assert_eq!(CppBackend::from_model_path(path), CppBackend::Llama);

        let path = Path::new("/path/to/llama-2-7b-chat/model.gguf");
        assert_eq!(CppBackend::from_model_path(path), CppBackend::Llama);
    }

    #[test]
    fn test_from_model_path_default() {
        use std::path::Path;

        // Test unknown model paths default to Llama
        let path = Path::new("models/unknown-model.gguf");
        assert_eq!(CppBackend::from_model_path(path), CppBackend::Llama);

        let path = Path::new("SmolLM3-3B-F16.gguf");
        assert_eq!(CppBackend::from_model_path(path), CppBackend::Llama);

        let path = Path::new("random/path/model.gguf");
        assert_eq!(CppBackend::from_model_path(path), CppBackend::Llama);
    }
}
