use anyhow::Result;
use std::path::Path;

/// Check GGUF file compatibility with llama.cpp
pub fn compat_check(path: &Path) -> Result<()> {
    bitnet_compat::gguf_fixer::GgufCompatibilityFixer::print_report(path)?;
    Ok(())
}

/// Export a fixed GGUF file with missing metadata inserted
pub fn compat_fix(input_path: &Path, output_path: &Path) -> Result<()> {
    bitnet_compat::gguf_fixer::GgufCompatibilityFixer::fix_and_export(input_path, output_path)?;
    Ok(())
}
