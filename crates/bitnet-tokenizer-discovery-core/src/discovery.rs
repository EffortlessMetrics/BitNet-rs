use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use tracing::debug;

#[derive(Debug, Clone)]
pub struct TokenizerDiscovery {
    model_path: PathBuf,
}

impl TokenizerDiscovery {
    #[must_use]
    pub fn new(model_path: PathBuf) -> Self {
        Self { model_path }
    }

    pub fn discover_with<F>(&self, verifier: &mut F) -> Result<PathBuf>
    where
        F: FnMut(&Path) -> Result<bool>,
    {
        debug!("Starting tokenizer auto-discovery for model: {}", self.model_path.display());

        if let Some(path) = self.check_sibling_tokenizer(verifier)? {
            debug!("Discovered sibling tokenizer: {}", path.display());
            return Ok(path);
        }

        if let Some(path) = self.check_parent_tokenizer(verifier)? {
            debug!("Discovered parent tokenizer: {}", path.display());
            return Ok(path);
        }

        Err(self.discovery_failed_error())
    }

    fn check_sibling_tokenizer<F>(&self, verifier: &mut F) -> Result<Option<PathBuf>>
    where
        F: FnMut(&Path) -> Result<bool>,
    {
        let model_dir = self.model_path.parent().unwrap_or_else(|| Path::new("."));
        let sibling_path = model_dir.join("tokenizer.json");

        debug!("Checking sibling tokenizer: {}", sibling_path.display());

        if sibling_path.exists() && sibling_path.is_file() && verifier(&sibling_path)? {
            return Ok(Some(sibling_path.canonicalize()?));
        }

        Ok(None)
    }

    fn check_parent_tokenizer<F>(&self, verifier: &mut F) -> Result<Option<PathBuf>>
    where
        F: FnMut(&Path) -> Result<bool>,
    {
        let model_dir = self.model_path.parent().unwrap_or_else(|| Path::new("."));

        if let Some(parent_dir) = model_dir.parent() {
            let parent_path = parent_dir.join("tokenizer.json");

            debug!("Checking parent tokenizer: {}", parent_path.display());

            if parent_path.exists() && parent_path.is_file() && verifier(&parent_path)? {
                return Ok(Some(parent_path.canonicalize()?));
            }
        }

        Ok(None)
    }

    fn discovery_failed_error(&self) -> anyhow::Error {
        let model_dir = self.model_path.parent().unwrap_or_else(|| Path::new("."));
        let sibling_path = model_dir.join("tokenizer.json");
        let parent_path = model_dir
            .parent()
            .map(|p| p.join("tokenizer.json"))
            .unwrap_or_else(|| PathBuf::from("N/A"));

        anyhow!(
            "Tokenizer not found for model: {}\n\
             \n\
             Tokenizer auto-discovery failed. Tried:\n\
             1. Sibling tokenizer.json: {} (not found/invalid)\n\
             2. Parent directory: {} (not found/invalid)\n\
             \n\
             Solution:\n\
             1. Download tokenizer:\n\
                cargo run -p xtask -- tokenizer --into {}\n\
             2. Provide explicit tokenizer path:\n\
                --tokenizer /path/to/tokenizer.json",
            self.model_path.display(),
            sibling_path.display(),
            parent_path.display(),
            model_dir.display(),
        )
    }
}
