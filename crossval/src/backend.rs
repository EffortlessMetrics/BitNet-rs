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
}
