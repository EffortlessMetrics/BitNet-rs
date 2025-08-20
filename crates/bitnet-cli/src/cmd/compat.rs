use std::path::Path;
use anyhow::Result;

/// Check GGUF file compatibility with llama.cpp
pub fn compat_check(path: &Path) -> Result<()> {
    bitnet_compat::gguf_fixer::GgufCompatibilityFixer::print_report(path)?;
    Ok(())
}

/// Fix GGUF file compatibility issues
pub fn compat_fix(input_path: &Path, output_path: &Path) -> Result<()> {
    bitnet_compat::gguf_fixer::GgufCompatibilityFixer::auto_fix(input_path, output_path)?;
    Ok(())
}