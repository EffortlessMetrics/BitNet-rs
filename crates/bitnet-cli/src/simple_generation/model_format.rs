use anyhow::Result;
use std::path::Path;

/// Validated model-format mode selected for a generation run.
pub(crate) enum ModelFormatMode {
    Auto,
    Gguf,
    Safetensors,
}

impl ModelFormatMode {
    pub(crate) fn parse(model_format: &str) -> Result<Self> {
        match model_format {
            "auto" => Ok(Self::Auto),
            "gguf" => Ok(Self::Gguf),
            "safetensors" => Ok(Self::Safetensors),
            other => anyhow::bail!(
                "Invalid --model-format '{}'. Supported values: auto, gguf, safetensors",
                other
            ),
        }
    }

    pub(crate) fn is_hf_directory(&self, model_path: &Path) -> bool {
        match self {
            Self::Gguf => false,
            Self::Safetensors => true,
            Self::Auto => model_path.is_dir(),
        }
    }
}
