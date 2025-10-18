# LLaMA-3 Tokenizer Fetching API Contracts

## Overview

This document defines the public API contracts for the LLaMA-3 tokenizer fetching and auto-discovery feature in BitNet.rs. These contracts ensure consistent behavior across the xtask CLI, bitnet-cli auto-discovery, and CI integration workflows.

## API Stability Guarantees

- **Stability Level**: Stable (v1.0.0)
- **Semantic Versioning**: Breaking changes require major version bump
- **Backward Compatibility**: Existing `--tokenizer` flag preserved indefinitely
- **Deprecation Policy**: 2 release cycles minimum before removal

## Core API Contracts

### 1. xtask Tokenizer Subcommand

**Contract ID**: `xtask.tokenizer.v1`

**CLI Signature**:
```bash
cargo run -p xtask -- tokenizer [OPTIONS]
```

**Options**:
| Flag | Type | Default | Required | Description |
|------|------|---------|----------|-------------|
| `--into <PATH>` | PathBuf | `models` | No | Output directory for tokenizer.json |
| `--source <SOURCE>` | TokenizerSource | `mirror` | No | Source preference (official\|mirror) |
| `--force` | bool | `false` | No | Force re-download if file exists |
| `--verbose` | bool | `false` | No | Verbose output for debugging |

**TokenizerSource Enum**:
```rust
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
            _ => Err(anyhow!("Invalid source: {}. Use 'official' or 'mirror'", s)),
        }
    }
}
```

**Success Behavior**:
- Downloads `tokenizer.json` to `<into>/tokenizer.json`
- Verifies file integrity using `bitnet-tokenizers::loader::load_tokenizer`
- Validates vocab size (~128,256 for LLaMA-3)
- Prints success message: `✓ Downloaded tokenizer to: <path>`
- Exits with code 0

**Failure Modes**:

| Exit Code | Scenario | Error Message Format |
|-----------|----------|---------------------|
| `EXIT_AUTH (11)` | Missing/invalid HF_TOKEN | `Authentication failed: HF_TOKEN required for official source\n\n[Setup instructions]` |
| `EXIT_NETWORK (14)` | Network connectivity issues | `Network error downloading tokenizer (attempt X/3)\n\n[Troubleshooting]` |
| `EXIT_VERIFICATION_FAILED (15)` | Invalid tokenizer file | `Tokenizer validation failed: [reason]\n\n[Re-download guidance]` |
| `1` | Generic error | `Error: [description]\n\n[Actionable guidance]` |

**Environment Variables**:
- `HF_TOKEN`: HuggingFace authentication token (required for `--source official`)
- `HTTP_PROXY`: HTTP proxy URL (optional)
- `HTTPS_PROXY`: HTTPS proxy URL (optional)

**Network Protocol**:
```
Request:
  GET https://huggingface.co/{repo}/resolve/main/tokenizer.json
  Headers:
    Authorization: Bearer {HF_TOKEN}  (only for official source)
    User-Agent: bitnet-xtask/0.1 (+https://github.com/microsoft/BitNet-rs)

Response (success):
  Status: 200 OK
  Content-Type: application/json
  Content-Length: <size>
  Body: tokenizer.json content

Response (auth failure):
  Status: 401 Unauthorized
  Body: {"error": "Unauthorized"}

Response (not found):
  Status: 404 Not Found
  Body: {"error": "Repository not found"}
```

**File Format Contract**:
```json
{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [...],
  "normalizer": {...},
  "pre_tokenizer": {...},
  "post_processor": {...},
  "decoder": {...},
  "model": {
    "type": "BPE",
    "vocab": {...},
    "merges": [...]
  }
}
```

**Verification Contract**:
```rust
/// Verify downloaded tokenizer meets LLaMA-3 requirements
///
/// Checks:
/// 1. Valid JSON structure (model.type, vocab, merges)
/// 2. Vocab size ~128,256 (±1% tolerance for padding)
/// 3. BPE model type
/// 4. Non-empty vocab and merges
///
/// Returns: Ok(()) if valid, Err with specific validation failure
pub fn verify_llama3_tokenizer(path: &Path) -> Result<()> {
    let tokenizer = load_tokenizer(path)
        .context("Failed to load tokenizer")?;

    let vocab_size = tokenizer.vocab_size();
    let expected = 128_256;
    let tolerance = (expected as f64 * 0.01) as usize; // 1% tolerance

    if vocab_size < expected - tolerance || vocab_size > expected + tolerance {
        bail!(
            "Invalid vocab size: expected ~{}, got {} (outside ±1% tolerance)",
            expected, vocab_size
        );
    }

    Ok(())
}
```

### 2. CLI Auto-Discovery

**Contract ID**: `cli.auto_discovery.v1`

**Public Interface**:
```rust
/// Tokenizer auto-discovery for BitNet.rs CLI
///
/// Discovery order:
/// 1. GGUF embedded tokenizer (tokenizer.ggml.model metadata)
/// 2. Sibling tokenizer.json (same directory as model)
/// 3. Parent directory tokenizer.json (one level up)
///
/// Invariants:
/// - Returns first valid tokenizer found in discovery order
/// - Never returns mock/placeholder tokenizers (fail-fast)
/// - Error messages include actionable guidance
/// - Discovery attempts logged at debug level
pub struct TokenizerDiscovery {
    model_path: PathBuf,
}

impl TokenizerDiscovery {
    /// Create discovery engine from model path
    ///
    /// Preconditions:
    /// - model_path must exist and be readable
    /// - model_path must point to a file (not directory)
    ///
    /// Postconditions:
    /// - Returns valid discovery instance
    /// - No filesystem operations performed (lazy evaluation)
    pub fn new(model_path: PathBuf) -> Self {
        Self { model_path }
    }

    /// Discover tokenizer via fallback chain
    ///
    /// Preconditions:
    /// - model_path exists and is readable
    ///
    /// Postconditions (success):
    /// - Returns absolute path to valid tokenizer
    /// - Tokenizer verified via load_tokenizer
    ///
    /// Postconditions (failure):
    /// - Returns error with discovery chain details
    /// - Error message includes actionable guidance
    /// - No side effects (no file creation/modification)
    pub fn discover(&self) -> Result<PathBuf> {
        // 1. Check GGUF embedded tokenizer
        if let Some(path) = self.check_embedded_tokenizer()? {
            tracing::debug!("Discovered embedded GGUF tokenizer");
            return Ok(path);
        }

        // 2. Check sibling tokenizer.json
        if let Some(path) = self.check_sibling_tokenizer()? {
            tracing::debug!("Discovered sibling tokenizer: {}", path.display());
            return Ok(path);
        }

        // 3. Check parent directory
        if let Some(path) = self.check_parent_tokenizer()? {
            tracing::debug!("Discovered parent tokenizer: {}", path.display());
            return Ok(path);
        }

        // No tokenizer found - return actionable error
        Err(self.discovery_failed_error())
    }

    /// Check for GGUF embedded tokenizer
    ///
    /// Returns: Some(PathBuf) if embedded tokenizer found and valid, None otherwise
    fn check_embedded_tokenizer(&self) -> Result<Option<PathBuf>>;

    /// Check for sibling tokenizer.json
    ///
    /// Returns: Some(PathBuf) if sibling exists and is valid, None otherwise
    fn check_sibling_tokenizer(&self) -> Result<Option<PathBuf>>;

    /// Check for parent directory tokenizer.json
    ///
    /// Returns: Some(PathBuf) if parent tokenizer exists and is valid, None otherwise
    fn check_parent_tokenizer(&self) -> Result<Option<PathBuf>>;

    /// Generate actionable error when discovery fails
    fn discovery_failed_error(&self) -> anyhow::Error;
}
```

**Discovery Chain Contract**:

| Priority | Location | Check | Example |
|----------|----------|-------|---------|
| 1 | GGUF embedded | `tokenizer.ggml.model` metadata exists | Model contains embedded tokenizer |
| 2 | Sibling | `{model_dir}/tokenizer.json` exists | `models/model-dir/tokenizer.json` |
| 3 | Parent | `{model_parent}/tokenizer.json` exists | `models/tokenizer.json` |

**Error Message Contract**:
```rust
/// Generate discovery failed error with actionable guidance
///
/// Format:
///   Error: Tokenizer not found for model: <model_path>
///
///   Tokenizer auto-discovery failed. Tried:
///     1. GGUF embedded tokenizer: Not present
///     2. Sibling tokenizer.json: <sibling_path> (not found)
///     3. Parent directory: <parent_path> (not found)
///
///   Solution:
///     1. Download LLaMA-3 tokenizer:
///        cargo run -p xtask -- tokenizer --into <dir>
///     2. Provide explicit tokenizer path:
///        --tokenizer /path/to/tokenizer.json
///
///   For more help, see: docs/quickstart.md#tokenizer-setup
fn discovery_failed_error(&self) -> anyhow::Error {
    let model_dir = self.model_path.parent().unwrap_or(Path::new("."));
    let sibling_path = model_dir.join("tokenizer.json");
    let parent_path = model_dir.parent()
        .map(|p| p.join("tokenizer.json"))
        .unwrap_or_else(|| PathBuf::from("N/A"));

    anyhow!(
        "Tokenizer not found for model: {}\n\
         \n\
         Tokenizer auto-discovery failed. Tried:\n\
         1. GGUF embedded tokenizer: Not present\n\
         2. Sibling tokenizer.json: {} (not found)\n\
         3. Parent directory: {} (not found)\n\
         \n\
         Solution:\n\
         1. Download LLaMA-3 tokenizer:\n\
            cargo run -p xtask -- tokenizer --into {}\n\
         2. Provide explicit tokenizer path:\n\
            --tokenizer /path/to/tokenizer.json\n\
         \n\
         For more help, see: docs/quickstart.md#tokenizer-setup",
        self.model_path.display(),
        sibling_path.display(),
        parent_path.display(),
        model_dir.display()
    )
}
```

### 3. Download Implementation

**Contract ID**: `download.llama3_tokenizer.v1`

**Public Interface**:
```rust
/// LLaMA-3 tokenizer download implementation
///
/// Responsibilities:
/// - Download tokenizer.json from HuggingFace repositories
/// - Verify file integrity and vocab size
/// - Handle authentication and network errors
/// - Provide progress reporting
///
/// Thread-safety: Safe (reqwest::Client is thread-safe)
/// Async: Yes (uses tokio runtime)
pub struct TokenizerDownloader {
    source: TokenizerSource,
    client: reqwest::Client,
    progress_enabled: bool,
}

impl TokenizerDownloader {
    /// Create downloader with source preference
    ///
    /// Preconditions:
    /// - None (source validation in TokenizerSource::from_str)
    ///
    /// Postconditions:
    /// - Returns configured downloader
    /// - HTTP client initialized with default settings
    pub fn new(source: TokenizerSource) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300)) // 5 minute timeout
            .user_agent("bitnet-xtask/0.1 (+https://github.com/microsoft/BitNet-rs)")
            .build()?;

        Ok(Self {
            source,
            client,
            progress_enabled: true,
        })
    }

    /// Enable/disable progress reporting
    pub fn with_progress(mut self, enabled: bool) -> Self {
        self.progress_enabled = enabled;
        self
    }

    /// Download tokenizer.json to target directory
    ///
    /// Preconditions:
    /// - target_dir exists and is writable
    /// - HF_TOKEN set in environment (if source == Official)
    ///
    /// Postconditions (success):
    /// - tokenizer.json written to target_dir/tokenizer.json
    /// - File verified via load_tokenizer
    /// - Vocab size validated (~128,256)
    /// - Returns absolute path to tokenizer
    ///
    /// Postconditions (failure):
    /// - No files written (atomic failure)
    /// - Returns error with specific failure reason
    /// - Temporary files cleaned up
    ///
    /// Network errors: Retry up to 3 times with exponential backoff
    pub async fn download(&self, target_dir: &Path) -> Result<PathBuf> {
        // 1. Validate preconditions
        self.validate_preconditions(target_dir)?;

        // 2. Build download URL
        let url = self.get_tokenizer_url();

        // 3. Download with retries
        let temp_path = target_dir.join("tokenizer.json.tmp");
        self.download_with_retries(&url, &temp_path).await?;

        // 4. Verify downloaded file
        self.verify_tokenizer(&temp_path)?;

        // 5. Atomic rename
        let final_path = target_dir.join("tokenizer.json");
        fs::rename(&temp_path, &final_path)?;

        Ok(final_path)
    }

    /// Get HuggingFace URL for tokenizer based on source
    ///
    /// Contract:
    /// - Official: https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/tokenizer.json
    /// - Mirror: https://huggingface.co/baseten/Meta-Llama-3-tokenizer/resolve/main/tokenizer.json
    fn get_tokenizer_url(&self) -> &'static str {
        match self.source {
            TokenizerSource::Official => {
                "https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/tokenizer.json"
            }
            TokenizerSource::Mirror => {
                "https://huggingface.co/baseten/Meta-Llama-3-tokenizer/resolve/main/tokenizer.json"
            }
        }
    }

    /// Verify downloaded tokenizer
    ///
    /// Contract:
    /// - File must be valid JSON
    /// - model.type must be "BPE"
    /// - vocab and merges must be non-empty
    /// - vocab_size must be ~128,256 (±1% tolerance)
    fn verify_tokenizer(&self, path: &Path) -> Result<()> {
        verify_llama3_tokenizer(path)
    }

    /// Validate preconditions before download
    fn validate_preconditions(&self, target_dir: &Path) -> Result<()> {
        // Check target directory exists
        if !target_dir.exists() {
            bail!("Target directory does not exist: {}", target_dir.display());
        }

        // Check target directory is writable
        if target_dir.metadata()?.permissions().readonly() {
            bail!("Target directory is not writable: {}", target_dir.display());
        }

        // Check HF_TOKEN for official source
        if self.source == TokenizerSource::Official && std::env::var("HF_TOKEN").is_err() {
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

    /// Download with retry logic
    ///
    /// Retry strategy:
    /// - Max attempts: 3
    /// - Backoff: Exponential (200ms, 400ms, 800ms)
    /// - Retryable errors: Network timeouts, 5xx responses
    /// - Non-retryable errors: 401/403 (auth), 404 (not found)
    async fn download_with_retries(&self, url: &str, dest: &Path) -> Result<()> {
        let max_attempts = 3;

        for attempt in 1..=max_attempts {
            match self.download_once(url, dest).await {
                Ok(()) => return Ok(()),
                Err(e) if attempt < max_attempts && is_retryable(&e) => {
                    let backoff_ms = 200u64 * (1 << (attempt - 1)); // 200ms, 400ms, 800ms
                    tracing::warn!(
                        "Download attempt {}/{} failed: {}. Retrying in {}ms...",
                        attempt, max_attempts, e, backoff_ms
                    );
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                }
                Err(e) => return Err(e),
            }
        }

        bail!("Download failed after {} attempts", max_attempts)
    }

    /// Single download attempt
    async fn download_once(&self, url: &str, dest: &Path) -> Result<()> {
        let mut request = self.client.get(url);

        // Add auth header for official source
        if self.source == TokenizerSource::Official {
            if let Ok(token) = std::env::var("HF_TOKEN") {
                request = request.header("Authorization", format!("Bearer {}", token));
            }
        }

        let response = request.send().await?;

        // Check status
        if response.status() == 401 || response.status() == 403 {
            bail!(
                "Authentication failed: {} {}\n\
                 \n\
                 Ensure HF_TOKEN is set and valid:\n\
                 export HF_TOKEN=<your-token>",
                response.status(),
                response.status().canonical_reason().unwrap_or("Unauthorized")
            );
        }

        if !response.status().is_success() {
            bail!("HTTP error: {} {}", response.status(), response.status().canonical_reason().unwrap_or(""));
        }

        // Download with progress
        let total_size = response.content_length();
        let mut file = fs::File::create(dest)?;
        let mut downloaded = 0u64;

        if self.progress_enabled && is_terminal::IsTerminal::is_terminal(&std::io::stderr()) {
            let pb = ProgressBar::new(total_size.unwrap_or(0));
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
                    .progress_chars("#>-")
            );
            pb.set_message("Downloading tokenizer.json");

            let mut stream = response.bytes_stream();
            while let Some(chunk) = stream.try_next().await? {
                file.write_all(&chunk)?;
                downloaded += chunk.len() as u64;
                pb.set_position(downloaded);
            }
            pb.finish_with_message("Download complete");
        } else {
            // No progress bar
            let bytes = response.bytes().await?;
            file.write_all(&bytes)?;
        }

        Ok(())
    }
}

/// Check if error is retryable
fn is_retryable(e: &anyhow::Error) -> bool {
    // Retry on network timeouts and 5xx errors
    if let Some(reqwest_err) = e.downcast_ref::<reqwest::Error>() {
        return reqwest_err.is_timeout() || reqwest_err.is_connect() ||
               reqwest_err.status().map_or(false, |s| s.is_server_error());
    }
    false
}
```

## Behavioral Contracts

### 1. Idempotency

**Contract**: Multiple invocations with same parameters produce same result

**Scenarios**:
- Running `xtask tokenizer` twice with same `--into` skips download (unless `--force`)
- Auto-discovery returns same path for same model across multiple runs
- Downloaded tokenizer verified identically across runs

**Guarantees**:
```rust
// Idempotency test
#[test]
fn test_download_idempotency() {
    let dir = TempDir::new().unwrap();

    // First download
    let cmd1 = TokenizerCommand { into: dir.path().to_path_buf(), source: TokenizerSource::Mirror, force: false, verbose: false };
    let result1 = cmd1.execute().unwrap();

    // Second download (should skip)
    let cmd2 = TokenizerCommand { into: dir.path().to_path_buf(), source: TokenizerSource::Mirror, force: false, verbose: false };
    let result2 = cmd2.execute().unwrap();

    // Both return same path
    assert_eq!(result1, result2);
    assert_eq!(result1, dir.path().join("tokenizer.json"));
}
```

### 2. Atomicity

**Contract**: Download either succeeds completely or fails without side effects

**Guarantees**:
- Temporary file used during download (`tokenizer.json.tmp`)
- Atomic rename only on successful verification
- Cleanup on failure (no partial files)

**Implementation**:
```rust
pub async fn download(&self, target_dir: &Path) -> Result<PathBuf> {
    let temp_path = target_dir.join("tokenizer.json.tmp");
    let final_path = target_dir.join("tokenizer.json");

    // Download to temp file
    self.download_with_retries(&url, &temp_path).await?;

    // Verify before rename
    self.verify_tokenizer(&temp_path)?;

    // Atomic rename (POSIX guarantees atomicity)
    fs::rename(&temp_path, &final_path)?;

    Ok(final_path)
}
```

### 3. Fail-Fast

**Contract**: Errors detected early with actionable messages

**Scenarios**:
- Missing HF_TOKEN: Fail before network request
- Invalid target directory: Fail before download
- Network errors: Fail with retry guidance
- Verification errors: Fail with re-download guidance

**Example**:
```rust
// Fail-fast on missing HF_TOKEN
fn validate_preconditions(&self, target_dir: &Path) -> Result<()> {
    if self.source == TokenizerSource::Official && std::env::var("HF_TOKEN").is_err() {
        // Fail immediately with actionable guidance
        bail!("HF_TOKEN required for official source\n\n[Setup instructions]");
    }
    Ok(())
}
```

## Compatibility Contracts

### 1. Backward Compatibility

**Contract**: Existing workflows continue to work without modification

**Guarantees**:
- Explicit `--tokenizer` flag continues to work (no deprecation)
- Auto-discovery is purely additive (opt-in via omitting flag)
- No changes to tokenizer file format or loading logic
- Existing scripts and CI workflows unaffected

### 2. Forward Compatibility

**Contract**: Future enhancements maintain existing API contracts

**Extension Points**:
- Additional tokenizer sources (e.g., local mirror, S3)
- Enhanced verification (checksum validation)
- Parallel downloads for multi-file tokenizers
- Automatic cache cleanup

**Non-Breaking Changes**:
- New CLI flags with sensible defaults
- Additional discovery locations in fallback chain
- Enhanced error messages and logging

## Testing Contracts

### 1. Unit Tests

**Contract**: All public APIs covered with unit tests

**Coverage Requirements**:
- `TokenizerCommand`: Parsing, source selection, validation
- `TokenizerDiscovery`: Discovery chain, error messages
- `TokenizerDownloader`: URL generation, verification logic

### 2. Integration Tests

**Contract**: End-to-end workflows validated

**Scenarios**:
- Download from mirror (no auth)
- Download from official (with HF_TOKEN)
- Auto-discovery finds sibling tokenizer
- Auto-discovery finds parent tokenizer
- Auto-discovery fails with clear error

### 3. CI Tests

**Contract**: CI pipelines validate production workflows

**Requirements**:
- Parity smoke test with automatic tokenizer provisioning
- Both official (with secret) and mirror (fallback) sources
- Cache integration to avoid redundant downloads

## Versioning and Stability

**Semantic Versioning**:
- **Patch (0.1.x)**: Bug fixes, non-breaking enhancements
- **Minor (0.x.0)**: New features, backward-compatible changes
- **Major (x.0.0)**: Breaking changes (rare, requires migration guide)

**Deprecation Policy**:
- Deprecation announced 2 releases in advance
- Migration guide provided in documentation
- Deprecated features maintained for 2 release cycles
- Clear deprecation warnings in CLI output

**API Evolution**:
```rust
// Example: Adding new source without breaking existing code
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerSource {
    Official,  // Existing
    Mirror,    // Existing
    // New sources added without breaking existing code
    LocalMirror(PathBuf), // Added in v0.2.0
}
```

## Conclusion

These API contracts ensure consistent, reliable, and maintainable behavior for the LLaMA-3 tokenizer fetching and auto-discovery feature. By defining clear contracts for CLI interfaces, auto-discovery logic, download implementation, and behavioral guarantees, we provide a solid foundation for production-ready BitNet.rs inference workflows.

All contracts are designed with backward compatibility, fail-fast error handling, and actionable user guidance as core principles.
