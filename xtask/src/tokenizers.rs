//! LLaMA-3 tokenizer download functionality
//!
//! This module implements tokenizer downloading from HuggingFace repositories
//! with support for official (authenticated) and mirror (unauthenticated) sources.
//!
//! AC:ID llama3-tokenizer-fetching-spec.md#ac1-xtask-tokenizer-subcommand

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use reqwest::header::AUTHORIZATION;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

/// Tokenizer source enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerSource {
    /// meta-llama/Meta-Llama-3-8B (requires HF_TOKEN)
    Official,
    /// baseten/Meta-Llama-3-tokenizer (no auth)
    Mirror,
}

impl FromStr for TokenizerSource {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "official" => Ok(Self::Official),
            "mirror" => Ok(Self::Mirror),
            _ => bail!("Invalid source: {}. Use 'official' or 'mirror'", s),
        }
    }
}

impl TokenizerSource {
    /// Get HuggingFace URL for this source
    fn url(&self) -> &'static str {
        match self {
            Self::Official => {
                "https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/tokenizer.json"
            }
            Self::Mirror => {
                "https://huggingface.co/baseten/Meta-Llama-3-tokenizer/resolve/main/tokenizer.json"
            }
        }
    }

    /// Check if this source requires HF_TOKEN authentication
    fn requires_auth(&self) -> bool {
        matches!(self, Self::Official)
    }
}

/// Download LLaMA-3 tokenizer.json
///
/// AC:ID llama3-tokenizer-api-contracts.md#download-llama3_tokenizer-v1
pub fn download_llama3_tokenizer(
    target_dir: &Path,
    source: TokenizerSource,
    force: bool,
    verbose: bool,
) -> Result<PathBuf> {
    let output_path = target_dir.join("tokenizer.json");

    // Check if file already exists (idempotent behavior)
    if output_path.exists() && !force {
        if verbose {
            eprintln!("✓ Tokenizer already exists at: {}", output_path.display());
        }
        return Ok(output_path);
    }

    // Validate preconditions
    validate_preconditions(target_dir, source)?;

    // Download to temporary file (atomic operation)
    let temp_path = target_dir.join("tokenizer.json.tmp");

    // Ensure cleanup on error
    let result = (|| -> Result<()> {
        download_from_source(source, &temp_path, verbose)?;

        // Verify downloaded tokenizer
        verify_llama3_tokenizer(&temp_path)?;

        // Atomic rename
        fs::rename(&temp_path, &output_path).context("Failed to rename temporary file")?;

        Ok(())
    })();

    // Clean up temp file on error
    if result.is_err() && temp_path.exists() {
        let _ = fs::remove_file(&temp_path);
    }

    result?;

    if verbose {
        eprintln!("✓ Downloaded tokenizer to: {}", output_path.display());
    }

    Ok(output_path)
}

/// Validate preconditions before download
fn validate_preconditions(target_dir: &Path, source: TokenizerSource) -> Result<()> {
    // Check target directory exists
    if !target_dir.exists() {
        bail!("Target directory does not exist: {}", target_dir.display());
    }

    // Check target directory is writable
    if target_dir.metadata()?.permissions().readonly() {
        bail!("Target directory is not writable: {}", target_dir.display());
    }

    // Check HF_TOKEN for official source
    if source.requires_auth() && std::env::var("HF_TOKEN").is_err() {
        bail!(
            "HF_TOKEN environment variable required for official source\n\
             \n\
             Setup instructions:\n\
             1. Accept LLaMA-3 license: https://huggingface.co/meta-llama/Meta-Llama-3-8B\n\
             2. Generate token: https://huggingface.co/settings/tokens\n\
             3. Export token: export HF_TOKEN=<your-token>"
        );
    }

    Ok(())
}

/// Download from source with retry logic
///
/// AC:ID llama3-tokenizer-api-contracts.md#network-protocol
fn download_from_source(source: TokenizerSource, dest: &Path, verbose: bool) -> Result<()> {
    let max_attempts = 3;

    for attempt in 1..=max_attempts {
        match download_once(source, dest, verbose) {
            Ok(()) => return Ok(()),
            Err(e) if attempt < max_attempts && is_retryable(&e) => {
                let backoff_ms = 1000u64 * (1 << (attempt - 1)); // 1s, 2s, 4s
                if verbose {
                    eprintln!(
                        "⚠ Download attempt {}/{} failed: {}. Retrying in {}s...",
                        attempt,
                        max_attempts,
                        e,
                        backoff_ms / 1000
                    );
                }
                std::thread::sleep(Duration::from_millis(backoff_ms));
            }
            Err(e) => return Err(e),
        }
    }

    bail!("Network error downloading tokenizer (attempt {}/{})", max_attempts, max_attempts)
}

/// Single download attempt
fn download_once(source: TokenizerSource, dest: &Path, verbose: bool) -> Result<()> {
    let client = Client::builder()
        .timeout(Duration::from_secs(300)) // 5 minute timeout
        .user_agent("bitnet-xtask/0.1 (+https://github.com/microsoft/BitNet-rs)")
        .build()?;

    let url = source.url();
    let mut request = client.get(url);

    // Add auth header for official source
    if source.requires_auth()
        && let Ok(token) = std::env::var("HF_TOKEN")
    {
        request = request.header(AUTHORIZATION, format!("Bearer {}", token));
    }

    if verbose {
        eprintln!("📥 Downloading from: {}", url);
    }

    let response = request.send().context("Failed to send HTTP request")?;

    // Check status
    if response.status() == 401 || response.status() == 403 {
        bail!(
            "Authentication failed: {} {}\n\
             \n\
             Ensure HF_TOKEN is set and valid:\n\
             export HF_TOKEN=<your-token>\n\
             \n\
             Get token at: https://huggingface.co/settings/tokens",
            response.status(),
            response.status().canonical_reason().unwrap_or("Unauthorized")
        );
    }

    if !response.status().is_success() {
        bail!(
            "HTTP error: {} {}",
            response.status(),
            response.status().canonical_reason().unwrap_or("")
        );
    }

    // Download with progress
    let total_size = response.content_length();
    let mut file = fs::File::create(dest).context("Failed to create temporary file")?;

    // Helper to fsync file after write
    let fsync_file = |f: fs::File| -> Result<()> {
        #[cfg(unix)]
        {
            // Sync file data to disk
            f.sync_all().context("Failed to sync file to disk")?;
        }
        Ok(())
    };

    if verbose && is_terminal::IsTerminal::is_terminal(&std::io::stderr()) {
        if let Some(size) = total_size {
            let pb = ProgressBar::new(size);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            pb.set_message("Downloading tokenizer.json");

            let mut downloaded = 0u64;
            let mut buffer = vec![0u8; 8192];
            let mut reader = response;

            loop {
                let n = reader.read(&mut buffer).context("Failed to read response")?;
                if n == 0 {
                    break;
                }
                file.write_all(&buffer[..n]).context("Failed to write to file")?;
                downloaded += n as u64;
                pb.set_position(downloaded);
            }
            pb.finish_with_message("Download complete");

            // Sync to disk before rename
            fsync_file(file)?;
        } else {
            // No progress bar if size unknown
            let bytes = response.bytes().context("Failed to read response body")?;
            file.write_all(&bytes).context("Failed to write to file")?;

            // Sync to disk before rename
            fsync_file(file)?;
        }
    } else {
        // No progress bar
        let bytes = response.bytes().context("Failed to read response body")?;
        file.write_all(&bytes).context("Failed to write to file")?;

        // Sync to disk before rename
        fsync_file(file)?;
    }

    Ok(())
}

/// Verify LLaMA-3 tokenizer
///
/// AC:ID llama3-tokenizer-api-contracts.md#verification-contract
pub fn verify_llama3_tokenizer(path: &Path) -> Result<()> {
    // Load tokenizer using bitnet-tokenizers
    let tokenizer =
        bitnet_tokenizers::loader::load_tokenizer(path).context("Failed to load tokenizer")?;

    let vocab_size = tokenizer.vocab_size();
    let expected = 128_256;
    let tolerance = (expected as f64 * 0.01) as usize; // 1% tolerance

    if vocab_size < expected - tolerance || vocab_size > expected + tolerance {
        bail!(
            "Invalid vocab size: expected ~{}, got {} (outside ±1% tolerance)",
            expected,
            vocab_size
        );
    }

    // Additional checks: ensure it's BPE-based
    // bitnet-tokenizers already validates this during load

    Ok(())
}

/// Check if error is retryable
fn is_retryable(e: &anyhow::Error) -> bool {
    // Retry on network timeouts and connection errors
    if let Some(reqwest_err) = e.downcast_ref::<reqwest::Error>() {
        return reqwest_err.is_timeout() || reqwest_err.is_connect();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_source_parsing() {
        assert_eq!(TokenizerSource::from_str("official").unwrap(), TokenizerSource::Official);
        assert_eq!(TokenizerSource::from_str("mirror").unwrap(), TokenizerSource::Mirror);
        assert_eq!(TokenizerSource::from_str("OFFICIAL").unwrap(), TokenizerSource::Official);
        assert!(TokenizerSource::from_str("invalid").is_err());
    }

    #[test]
    fn test_tokenizer_source_urls() {
        assert!(TokenizerSource::Official.url().contains("meta-llama"));
        assert!(TokenizerSource::Mirror.url().contains("baseten"));
    }

    #[test]
    fn test_tokenizer_source_auth() {
        assert!(TokenizerSource::Official.requires_auth());
        assert!(!TokenizerSource::Mirror.requires_auth());
    }
}
