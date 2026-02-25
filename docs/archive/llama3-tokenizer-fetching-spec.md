# LLaMA-3 Tokenizer Fetching and Auto-Discovery Specification

## Context

The Microsoft BitNet 2B4T GGUF model uses the LLaMA-3 tokenizer (vocab size: 128,256) but the GGUF file doesn't embed it. Users must manually provide `tokenizer.json`, which breaks the "copy-paste works" README experience and creates friction for new users attempting to run inference.

This specification defines the architecture for implementing automatic tokenizer fetching and auto-discovery to enable zero-configuration inference for LLaMA-3 models.

**Affected BitNet.rs Components:**
- `xtask`: New `tokenizer` subcommand for downloading and verification
- `bitnet-cli`: Auto-discovery logic when `--tokenizer` flag not provided
- `bitnet-tokenizers`: Tokenizer loading and validation utilities
- `.github/workflows/`: CI integration for parity smoke tests

**Neural Network Inference Pipeline Impact:**
- **Model Loading**: GGUF parsing unaffected
- **Tokenizer Discovery**: New auto-discovery phase before inference (Entry point)
- **Inference**: Seamless integration with existing generation pipeline
- **Cross-Validation**: Enhanced parity tests with automatic tokenizer provisioning

## User Stories

### Story 1: Zero-Configuration Quick Start
**As a** new BitNet.rs user
**I want** to run inference without manually downloading tokenizers
**So that** I can follow README quick-start examples without additional setup steps

**Business Value**: Reduces onboarding friction, improves first-run success rate, enables production-ready workflows

### Story 2: Reproducible CI Workflows
**As a** BitNet.rs CI maintainer
**I want** automatic tokenizer provisioning in CI pipelines
**So that** parity smoke tests pass without manual tokenizer management

**Business Value**: Robust CI/CD, predictable test environments, reduced maintenance overhead

### Story 3: Flexible Source Selection
**As a** enterprise deployment engineer
**I want** to choose between official and mirror tokenizer sources
**So that** I can comply with licensing requirements and network constraints

**Business Value**: Enterprise-ready deployment, compliance with organizational policies, air-gapped environment support

## Acceptance Criteria

### AC1: xtask tokenizer subcommand
**Given** a user needs the LLaMA-3 tokenizer
**When** they run `cargo run -p xtask -- tokenizer --into <dir>`
**Then** the system downloads `tokenizer.json` from HuggingFace repositories
**And** saves to `<dir>/tokenizer.json` with verification

**Test Tag**: `// AC1:tokenizer_subcommand`

**Validation**:
- Primary source: `meta-llama/Meta-Llama-3-8B` or `meta-llama/Meta-Llama-3-8B-Instruct` (requires `HF_TOKEN`)
- Fallback source: `baseten/Meta-Llama-3-tokenizer` (no auth required)
- `--source official|mirror` flag controls source preference
- Downloaded file verified using `bitnet-tokenizers::loader::load_tokenizer`
- LLaMA-3 vocab size validated (~128,256 tokens)
- Progress indicator displays download status
- Skips download if `tokenizer.json` already exists
- Error handling for network failures, auth errors, invalid files

**CLI Interface**:
```bash
# Download to default location (models/)
cargo run -p xtask -- tokenizer --into models/

# Prefer official source (requires HF_TOKEN)
HF_TOKEN=<token> cargo run -p xtask -- tokenizer --into models/ --source official

# Use mirror (no auth)
cargo run -p xtask -- tokenizer --into models/ --source mirror

# Verbose output for debugging
cargo run -p xtask -- tokenizer --into models/ --source mirror --verbose
```

### AC2: CLI auto-discovery
**Given** a user runs inference without `--tokenizer` flag
**When** the CLI initializes inference
**Then** the system auto-discovers tokenizer via fallback chain
**And** fails with clear error if not found

**Test Tag**: `// AC2:cli_auto_discovery`

**Discovery Order**:
1. Check GGUF for embedded tokenizer metadata (`tokenizer.ggml.model`)
2. Look for sibling `tokenizer.json` in model directory (e.g., `models/model-dir/tokenizer.json`)
3. Look for `tokenizer.json` in parent directory (e.g., `models/tokenizer.json`)
4. Fail with actionable error message

**Error Message Format**:
```
Error: Tokenizer not found for model: models/model.gguf

Tokenizer auto-discovery failed. Tried:
  1. GGUF embedded tokenizer: Not present
  2. Sibling tokenizer.json: models/model-dir/tokenizer.json (not found)
  3. Parent directory: models/tokenizer.json (not found)

Solution:
  1. Download LLaMA-3 tokenizer:
     cargo run -p xtask -- tokenizer --into models/
  2. Provide explicit tokenizer path:
     --tokenizer /path/to/tokenizer.json

For more help, see: docs/quickstart.md#tokenizer-setup
```

**Validation**:
- Explicit `--tokenizer` flag bypasses auto-discovery (backward compatibility)
- Auto-discovery logs discovery attempts at debug level
- Error messages include actionable guidance and documentation links
- No silent fallbacks to mock tokenizers (fails fast)

### AC3: CI integration
**Given** CI runs parity smoke tests
**When** the workflow provisions tokenizers
**Then** the system fetches tokenizers before tests
**And** supports both official and mirror sources

**Test Tag**: `// AC3:ci_integration`

**Workflow Integration**:
```yaml
# .github/workflows/parity-smoke.yml
steps:
  - name: Fetch LLaMA-3 tokenizer (mirror)
    run: |
      cargo run -p xtask -- tokenizer --into models/ --source mirror
    if: env.HF_TOKEN == ''

  - name: Fetch LLaMA-3 tokenizer (official)
    run: |
      cargo run -p xtask -- tokenizer --into models/ --source official
    env:
      HF_TOKEN: ${{ secrets.HF_TOKEN }}
    if: env.HF_TOKEN != ''

  - name: Run parity smoke test
    run: |
      ./scripts/parity_smoke.sh models/model.gguf models/tokenizer.json
```

**Validation**:
- CI passes with mirror source (no auth)
- CI passes with official source (when `HF_TOKEN` available)
- Tokenizer path passed to `parity_smoke.sh` script
- Download failures do not block entire CI run (graceful degradation)
- Cache integration to avoid redundant downloads

### AC4: Documentation
**Given** users need guidance on tokenizer setup
**When** they read README and quickstart docs
**Then** clear instructions for tokenizer fetching are provided

**Test Tag**: `// AC4:documentation`

**Documentation Updates**:

**README.md Quick Start Section**:
```markdown
## Quick Start

### 1. Download Model and Tokenizer

```bash
# Download Microsoft BitNet 2B4T model
cargo run -p xtask -- download-model

# Download LLaMA-3 tokenizer (required for BitNet 2B4T)
cargo run -p xtask -- tokenizer --into models/
```

### 2. Run Inference

```bash
# Auto-discovery finds tokenizer in models/
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --prompt "What is 2+2?" \
  --max-tokens 16

# Or provide explicit path
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 16
```
```

**docs/quickstart.md Tokenizer Section**:
```markdown
## Tokenizer Setup

BitNet.rs requires a tokenizer for text-to-token encoding. The CLI supports automatic tokenizer discovery.

### Automatic Discovery

When `--tokenizer` flag is not provided, BitNet.rs searches for tokenizers in this order:

1. **GGUF embedded tokenizer** (some models include embedded tokenizers)
2. **Sibling tokenizer.json** in model directory
3. **Parent directory tokenizer.json**

### Manual Download

For models requiring external tokenizers (e.g., Microsoft BitNet 2B4T with LLaMA-3 tokenizer):

```bash
# Download from mirror (no authentication)
cargo run -p xtask -- tokenizer --into models/ --source mirror

# Download from official source (requires HF_TOKEN)
HF_TOKEN=<your-token> cargo run -p xtask -- tokenizer --into models/ --source official
```

### HF_TOKEN Requirement

Official LLaMA-3 tokenizer requires accepting Meta's license agreement on HuggingFace:

1. Create HuggingFace account: https://huggingface.co/join
2. Accept LLaMA-3 license: https://huggingface.co/meta-llama/Meta-Llama-3-8B
3. Generate access token: https://huggingface.co/settings/tokens
4. Export token: `export HF_TOKEN=<your-token>`

### Troubleshooting

**Error: "Tokenizer not found"**
- Run: `cargo run -p xtask -- tokenizer --into models/`
- Or provide explicit path: `--tokenizer /path/to/tokenizer.json`

**Error: "401 Unauthorized"**
- Accept LLaMA-3 license on HuggingFace
- Verify `HF_TOKEN` is set: `echo $HF_TOKEN`
- Use mirror source: `--source mirror`

**Error: "Invalid tokenizer file"**
- Re-download: `rm models/tokenizer.json && cargo run -p xtask -- tokenizer --into models/`
- Check file integrity: `cargo run -p bitnet-cli -- inspect models/tokenizer.json`
```

## Technical Requirements

### Scope

**Affected Workspace Crates**:
- `xtask`: New `tokenizer` subcommand implementation
- `bitnet-cli`: Auto-discovery integration in `run`, `chat`, `inspect` commands
- `bitnet-tokenizers`: Tokenizer loading and validation (existing utilities)
- `.github/workflows/`: CI workflow updates for tokenizer provisioning

**Pipeline Stages**:
1. **Tokenizer Provisioning** (new): One-time download via xtask
2. **Tokenizer Discovery** (new): Auto-discovery in CLI before inference
3. **Tokenizer Loading**: Existing `load_tokenizer` infrastructure (unchanged)
4. **Inference Integration**: Seamless integration with generation pipeline (unchanged)

### Constraints

**Performance Targets**:
- Tokenizer download: <30s for 10MB file over typical broadband
- Auto-discovery latency: <50ms for filesystem checks
- Verification overhead: <100ms for tokenizer validation

**Network Requirements**:
- HTTP/HTTPS support via `reqwest` with `rustls-tls`
- Proxy support via standard env vars (`HTTP_PROXY`, `HTTPS_PROXY`)
- Retry logic for transient network failures (3 retries with exponential backoff)
- Content-Length validation before download

**Authentication**:
- `HF_TOKEN` environment variable for official sources
- Clear error messages for 401/403 (missing/invalid token)
- Mirror sources require no authentication
- Token validation before download attempts

**Production Reliability**:
- No new external dependencies beyond `reqwest`/`tokio` (already in workspace)
- Fail-fast on invalid tokenizer files (no silent corruption)
- Atomic file writes to prevent partial downloads
- Clear error messages with actionable guidance
- No mock tokenizer fallbacks (strict mode enforcement)

### Public Contracts

**xtask CLI Interface**:
```rust
/// Download LLaMA-3 tokenizer from HuggingFace
///
/// Subcommand: cargo run -p xtask -- tokenizer [OPTIONS]
#[derive(Debug, Clone)]
pub struct TokenizerCommand {
    /// Output directory for tokenizer.json
    #[arg(long, default_value = "models")]
    pub into: PathBuf,

    /// Source preference: official (requires HF_TOKEN) or mirror (no auth)
    #[arg(long, default_value = "mirror")]
    pub source: TokenizerSource,

    /// Force re-download even if file exists
    #[arg(long, default_value_t = false)]
    pub force: bool,

    /// Verbose output for debugging
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerSource {
    /// meta-llama/Meta-Llama-3-8B (requires HF_TOKEN)
    Official,
    /// baseten/Meta-Llama-3-tokenizer (no auth)
    Mirror,
}

impl TokenizerCommand {
    /// Execute tokenizer download
    pub fn execute(&self) -> Result<()>;
}
```

**Auto-Discovery API**:
```rust
/// Tokenizer auto-discovery for BitNet.rs CLI
pub struct TokenizerDiscovery {
    model_path: PathBuf,
}

impl TokenizerDiscovery {
    /// Create discovery engine from model path
    pub fn new(model_path: PathBuf) -> Self;

    /// Discover tokenizer via fallback chain
    ///
    /// Discovery order:
    /// 1. GGUF embedded tokenizer
    /// 2. Sibling tokenizer.json
    /// 3. Parent directory tokenizer.json
    ///
    /// Returns: PathBuf to discovered tokenizer or error with actionable guidance
    pub fn discover(&self) -> Result<PathBuf>;

    /// Check for GGUF embedded tokenizer
    fn check_embedded_tokenizer(&self) -> Result<Option<PathBuf>>;

    /// Check for sibling tokenizer.json
    fn check_sibling_tokenizer(&self) -> Result<Option<PathBuf>>;

    /// Check for parent directory tokenizer.json
    fn check_parent_tokenizer(&self) -> Result<Option<PathBuf>>;
}
```

**Download Implementation**:
```rust
/// LLaMA-3 tokenizer download implementation
pub struct TokenizerDownloader {
    source: TokenizerSource,
    client: reqwest::Client,
}

impl TokenizerDownloader {
    /// Create downloader with source preference
    pub fn new(source: TokenizerSource) -> Result<Self>;

    /// Download tokenizer.json to target directory
    ///
    /// Steps:
    /// 1. Validate target directory exists
    /// 2. Build HuggingFace URL based on source
    /// 3. Add Authorization header if HF_TOKEN present
    /// 4. Download with progress indicator
    /// 5. Verify file using bitnet-tokenizers::loader
    /// 6. Validate vocab size (~128,256 for LLaMA-3)
    ///
    /// Errors:
    /// - NetworkError: Connection failures, timeouts
    /// - AuthError: 401/403 (missing/invalid HF_TOKEN)
    /// - ValidationError: Invalid tokenizer file
    /// - IoError: Filesystem issues
    pub async fn download(&self, target_dir: &Path) -> Result<PathBuf>;

    /// Get HuggingFace URL for tokenizer
    fn get_tokenizer_url(&self) -> &'static str;

    /// Verify downloaded tokenizer
    fn verify_tokenizer(&self, path: &Path) -> Result<()>;
}
```

### Error Handling Strategy

**Error Categories**:

1. **Network Errors** (recoverable with retries)
   - Connection timeout: Retry with exponential backoff (3 attempts)
   - DNS resolution failure: Fail fast with network connectivity guidance
   - HTTP 5xx: Retry with exponential backoff (3 attempts)

2. **Authentication Errors** (user action required)
   - HTTP 401/403: Clear error message with HF_TOKEN setup instructions
   - Invalid token format: Validate token format before API call
   - License not accepted: Guidance to accept LLaMA-3 license on HuggingFace

3. **Validation Errors** (fail fast)
   - Invalid JSON: Parse error with file path and line number
   - Missing required fields: Specify missing field and expected structure
   - Incorrect vocab size: Display expected vs actual vocab size

4. **Filesystem Errors** (fail fast)
   - Permission denied: Check directory write permissions
   - Disk full: Display available space and required space
   - Path traversal: Sanitize paths before filesystem operations

**Error Message Templates**:

```rust
/// Network error with retry guidance
pub fn network_error(attempt: u32, max_attempts: u32) -> String {
    format!(
        "Network error downloading tokenizer (attempt {}/{})\n\
         \n\
         Troubleshooting:\n\
         1. Check internet connectivity\n\
         2. Verify proxy settings (HTTP_PROXY, HTTPS_PROXY)\n\
         3. Try mirror source: --source mirror",
        attempt, max_attempts
    )
}

/// Authentication error with setup guidance
pub fn auth_error() -> String {
    "Authentication failed: HF_TOKEN required for official source\n\
     \n\
     Setup instructions:\n\
     1. Accept LLaMA-3 license: https://huggingface.co/meta-llama/Meta-Llama-3-8B\n\
     2. Generate token: https://huggingface.co/settings/tokens\n\
     3. Export token: export HF_TOKEN=<your-token>\n\
     \n\
     Alternative: Use mirror source (no auth required)\n\
     cargo run -p xtask -- tokenizer --into models/ --source mirror".to_string()
}

/// Validation error with re-download guidance
pub fn validation_error(reason: &str) -> String {
    format!(
        "Tokenizer validation failed: {}\n\
         \n\
         Solution: Re-download tokenizer\n\
         rm models/tokenizer.json\n\
         cargo run -p xtask -- tokenizer --into models/",
        reason
    )
}
```

### Testing Approach

**Unit Tests** (`// AC1:tokenizer_subcommand`):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    // AC1: Tokenizer subcommand parsing
    #[test]
    fn test_tokenizer_command_parsing() {
        let cmd = TokenizerCommand {
            into: PathBuf::from("models/"),
            source: TokenizerSource::Mirror,
            force: false,
            verbose: false,
        };
        assert_eq!(cmd.source, TokenizerSource::Mirror);
        assert_eq!(cmd.into, PathBuf::from("models/"));
    }

    // AC1: URL generation for sources
    #[test]
    fn test_tokenizer_url_generation() {
        let official = TokenizerDownloader::new(TokenizerSource::Official)
            .unwrap()
            .get_tokenizer_url();
        assert!(official.contains("meta-llama/Meta-Llama-3"));

        let mirror = TokenizerDownloader::new(TokenizerSource::Mirror)
            .unwrap()
            .get_tokenizer_url();
        assert!(mirror.contains("baseten/Meta-Llama-3-tokenizer"));
    }

    // AC1: Verification logic
    #[test]
    fn test_tokenizer_verification() {
        let downloader = TokenizerDownloader::new(TokenizerSource::Mirror).unwrap();

        // Valid tokenizer
        let valid_path = Path::new("tests/fixtures/llama3-tokenizer.json");
        assert!(downloader.verify_tokenizer(valid_path).is_ok());

        // Invalid tokenizer
        let invalid_path = Path::new("tests/fixtures/invalid-tokenizer.json");
        assert!(downloader.verify_tokenizer(invalid_path).is_err());
    }
}
```

**Integration Tests** (`// AC2:cli_auto_discovery`):
```rust
#[test]
fn test_cli_auto_discovery_sibling() {
    // Setup: Create model and sibling tokenizer.json
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("model.gguf");
    let tokenizer_path = temp_dir.path().join("tokenizer.json");

    fs::write(&model_path, b"mock gguf").unwrap();
    fs::write(&tokenizer_path, VALID_LLAMA3_TOKENIZER).unwrap();

    // Execute: Auto-discovery
    let discovery = TokenizerDiscovery::new(model_path);
    let discovered = discovery.discover().unwrap();

    // Assert: Sibling tokenizer found
    assert_eq!(discovered, tokenizer_path);
}

#[test]
fn test_cli_auto_discovery_parent() {
    // Setup: Create model in subdirectory, tokenizer in parent
    let temp_dir = TempDir::new().unwrap();
    let model_dir = temp_dir.path().join("model-dir");
    fs::create_dir(&model_dir).unwrap();

    let model_path = model_dir.join("model.gguf");
    let tokenizer_path = temp_dir.path().join("tokenizer.json");

    fs::write(&model_path, b"mock gguf").unwrap();
    fs::write(&tokenizer_path, VALID_LLAMA3_TOKENIZER).unwrap();

    // Execute: Auto-discovery
    let discovery = TokenizerDiscovery::new(model_path);
    let discovered = discovery.discover().unwrap();

    // Assert: Parent tokenizer found
    assert_eq!(discovered, tokenizer_path);
}

#[test]
fn test_cli_auto_discovery_fails_with_clear_error() {
    // Setup: Model with no tokenizer
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("model.gguf");
    fs::write(&model_path, b"mock gguf").unwrap();

    // Execute: Auto-discovery
    let discovery = TokenizerDiscovery::new(model_path);
    let result = discovery.discover();

    // Assert: Clear error message
    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(error.to_string().contains("Tokenizer not found"));
    assert!(error.to_string().contains("cargo run -p xtask -- tokenizer"));
}
```

**CI Tests** (`// AC3:ci_integration`):
```rust
#[test]
fn test_ci_tokenizer_provisioning_mirror() {
    // Simulate CI environment (no HF_TOKEN)
    std::env::remove_var("HF_TOKEN");

    // Execute: Download from mirror
    let cmd = TokenizerCommand {
        into: PathBuf::from("target/test-tokenizers/"),
        source: TokenizerSource::Mirror,
        force: true,
        verbose: false,
    };

    let result = cmd.execute();
    assert!(result.is_ok());

    // Verify: tokenizer.json exists and is valid
    let tokenizer_path = PathBuf::from("target/test-tokenizers/tokenizer.json");
    assert!(tokenizer_path.exists());

    let tokenizer = load_tokenizer(&tokenizer_path).unwrap();
    assert_eq!(tokenizer.vocab_size(), 128256);
}

#[test]
#[ignore] // Requires HF_TOKEN secret in CI
fn test_ci_tokenizer_provisioning_official() {
    // Requires HF_TOKEN to be set
    if std::env::var("HF_TOKEN").is_err() {
        eprintln!("Skipping test: HF_TOKEN not set");
        return;
    }

    // Execute: Download from official source
    let cmd = TokenizerCommand {
        into: PathBuf::from("target/test-tokenizers/"),
        source: TokenizerSource::Official,
        force: true,
        verbose: false,
    };

    let result = cmd.execute();
    assert!(result.is_ok());
}
```

**Documentation Tests** (`// AC4:documentation`):
```rust
#[test]
fn test_readme_quickstart_commands() {
    // Verify README commands are valid
    let output = Command::new("cargo")
        .args(["run", "-p", "xtask", "--", "tokenizer", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());

    let help_text = String::from_utf8_lossy(&output.stdout);
    assert!(help_text.contains("--into"));
    assert!(help_text.contains("--source"));
}

#[test]
fn test_documentation_links_valid() {
    // Verify documentation files exist
    assert!(Path::new("docs/quickstart.md").exists());
    assert!(Path::new("README.md").exists());

    // Verify quickstart contains tokenizer section
    let quickstart = fs::read_to_string("docs/quickstart.md").unwrap();
    assert!(quickstart.contains("Tokenizer Setup"));
    assert!(quickstart.contains("cargo run -p xtask -- tokenizer"));
}
```

### Dependencies and External Integrations

**Required Dependencies** (already in workspace):
- `reqwest = { version = "0.12", features = ["blocking", "rustls-tls"] }`
- `tokio = { version = "1.48", features = ["rt"] }` (optional for async download)
- `indicatif = "0.18"` (progress bars - already in xtask)
- `anyhow = "1.0"` (error handling - already in xtask)

**HuggingFace Integration**:
- **Official Repository**: `meta-llama/Meta-Llama-3-8B` or `meta-llama/Meta-Llama-3-8B-Instruct`
  - File path: `tokenizer.json`
  - URL pattern: `https://huggingface.co/{repo}/resolve/main/tokenizer.json`
  - Authentication: Bearer token via `HF_TOKEN` header
  - License: Meta LLaMA-3 Community License Agreement (requires acceptance)

- **Mirror Repository**: `baseten/Meta-Llama-3-tokenizer`
  - File path: `tokenizer.json`
  - URL pattern: `https://huggingface.co/baseten/Meta-Llama-3-tokenizer/resolve/main/tokenizer.json`
  - Authentication: None required
  - License: Apache 2.0 / MIT (permissive)

**Existing Integration Points**:
- `bitnet-tokenizers::loader::load_tokenizer`: Verification and loading
- `xtask/src/main.rs`: CLI parsing and subcommand dispatch
- `bitnet-cli/src/main.rs`: Auto-discovery integration
- `.github/workflows/parity-smoke.yml`: CI workflow updates

### Migration Path for Existing Users

**Backward Compatibility**:
- Explicit `--tokenizer` flag continues to work (no breaking changes)
- Auto-discovery is purely additive (opt-in via omitting `--tokenizer`)
- Existing scripts and workflows unaffected

**Migration Steps**:

1. **Update README Quick Start** (non-breaking):
   ```markdown
   # Before (manual tokenizer download)
   wget https://huggingface.co/.../tokenizer.json -O models/tokenizer.json
   cargo run -p bitnet-cli -- run --model model.gguf --tokenizer models/tokenizer.json

   # After (automatic tokenizer provisioning)
   cargo run -p xtask -- tokenizer --into models/
   cargo run -p bitnet-cli -- run --model model.gguf  # Auto-discovery
   ```

2. **Update CI Workflows** (additive):
   ```yaml
   # Before (manual setup)
   - name: Setup tokenizer
     run: wget ... -O models/tokenizer.json

   # After (xtask subcommand)
   - name: Fetch tokenizer
     run: cargo run -p xtask -- tokenizer --into models/ --source mirror
   ```

3. **Deprecation Timeline** (gradual):
   - **v0.1.0**: Introduce `xtask tokenizer` and auto-discovery (opt-in)
   - **v0.2.0**: Update all documentation to use new workflow
   - **v1.0.0**: Keep explicit `--tokenizer` flag for advanced use cases (no removal)

**Communication Plan**:
- Update CHANGELOG.md with new feature announcement
- Add migration guide to docs/howto/
- Update all examples and documentation
- Announce in release notes and GitHub Discussions

## Architecture Decisions

### Decision 1: xtask Subcommand vs CLI Subcommand

**Choice**: Implement as `cargo run -p xtask -- tokenizer` (not `bitnet-cli tokenizer`)

**Rationale**:
- **One-time setup operation**: Tokenizer fetching is a setup task, not runtime inference
- **Aligns with existing patterns**: `download-model` is already in xtask
- **Keeps CLI focused**: `bitnet-cli` focuses on inference operations (`run`, `chat`, `inspect`)
- **Developer workflow consistency**: xtask is the established pattern for developer tooling

**Alternatives Considered**:
- CLI subcommand: Increases CLI surface area, conflicts with inference focus
- Shell script: Less portable, harder to maintain, no Rust ecosystem integration
- Automatic download on first run: Network dependency during inference, surprising behavior

**Trade-offs**:
- ✅ Consistent with existing architecture
- ✅ Clear separation of concerns
- ✅ No runtime network dependencies
- ❌ Requires two commands (download + infer) instead of one

### Decision 2: Mirror vs Official Default Source

**Choice**: Default to `--source mirror` (no authentication required)

**Rationale**:
- **Zero-friction experience**: New users can run commands without HF_TOKEN setup
- **CI-friendly**: Public CI environments can provision tokenizers without secrets
- **Licensing flexibility**: Mirror uses permissive Apache 2.0 / MIT license
- **Fallback availability**: Official source available via explicit flag

**Alternatives Considered**:
- Default to official: Requires license acceptance, HF_TOKEN setup friction
- Prompt user for source: Adds interactivity, breaks CI automation
- Try official then mirror: Network latency, confusing auth errors

**Trade-offs**:
- ✅ Best default for new users
- ✅ CI-friendly without secrets
- ✅ Permissive licensing
- ❌ Mirror may lag behind official releases (acceptable for stable LLaMA-3 tokenizer)

### Decision 3: Auto-Discovery vs Explicit Paths

**Choice**: Implement auto-discovery with clear error messages, preserve explicit `--tokenizer` flag

**Rationale**:
- **Zero-configuration goal**: Enables "copy-paste works" README examples
- **Backward compatibility**: Explicit flag preserves existing workflows
- **Fail-fast philosophy**: Clear errors better than silent mock fallbacks
- **Production reliability**: Predictable behavior with actionable guidance

**Alternatives Considered**:
- Require explicit path always: Breaks zero-config goal, friction for users
- Automatic download on missing: Network dependency during inference, surprising behavior
- Silent mock fallback: Hides real issues, production reliability concern

**Trade-offs**:
- ✅ Best user experience for new users
- ✅ Backward compatible
- ✅ Clear error messages
- ❌ More complex implementation (discovery chain)

## Integration Points

### xtask Integration

**File**: `xtask/src/main.rs`

**Changes**:
1. Add `Tokenizer` variant to `Cmd` enum
2. Implement `TokenizerCommand` struct with clap args
3. Add download and verification logic
4. Update help documentation

**CLI Interface**:
```rust
#[derive(Subcommand)]
enum Cmd {
    // ... existing commands ...

    /// Download LLaMA-3 tokenizer from HuggingFace
    ///
    /// Downloads tokenizer.json for LLaMA-3 models (vocab size: 128,256).
    /// Supports both official source (requires HF_TOKEN) and mirror (no auth).
    ///
    /// Examples:
    ///   cargo run -p xtask -- tokenizer --into models/ --source mirror
    ///   HF_TOKEN=<token> cargo run -p xtask -- tokenizer --into models/ --source official
    Tokenizer {
        /// Output directory for tokenizer.json
        #[arg(long, default_value = "models")]
        into: PathBuf,

        /// Source preference: official (requires HF_TOKEN) or mirror (no auth)
        #[arg(long, default_value = "mirror")]
        source: String,

        /// Force re-download even if file exists
        #[arg(long, default_value_t = false)]
        force: bool,

        /// Verbose output for debugging
        #[arg(short, long, default_value_t = false)]
        verbose: bool,
    },
}
```

### CLI Integration

**File**: `crates/bitnet-cli/src/main.rs`

**Changes**:
1. Add `TokenizerDiscovery` before inference initialization
2. Update `--tokenizer` flag to be optional
3. Add debug logging for discovery attempts
4. Update error messages with actionable guidance

**Auto-Discovery Integration**:
```rust
// In run/chat/inspect command handlers
fn resolve_tokenizer(model_path: &Path, explicit_path: Option<PathBuf>) -> Result<PathBuf> {
    // If explicit path provided, use it (backward compatibility)
    if let Some(path) = explicit_path {
        tracing::debug!("Using explicit tokenizer path: {}", path.display());
        return Ok(path);
    }

    // Auto-discovery
    tracing::debug!("Starting tokenizer auto-discovery for model: {}", model_path.display());
    let discovery = TokenizerDiscovery::new(model_path.to_path_buf());
    discovery.discover()
}
```

### CI Integration

**File**: `.github/workflows/parity-smoke.yml`

**Changes**:
1. Add tokenizer provisioning step before tests
2. Support both official (with secret) and mirror (fallback)
3. Pass tokenizer path to `parity_smoke.sh`
4. Cache tokenizer downloads

**Workflow Updates**:
```yaml
- name: Fetch LLaMA-3 tokenizer
  run: |
    if [ -n "${{ secrets.HF_TOKEN }}" ]; then
      echo "Using official source (HF_TOKEN available)"
      cargo run -p xtask -- tokenizer --into models/ --source official
    else
      echo "Using mirror source (no HF_TOKEN)"
      cargo run -p xtask -- tokenizer --into models/ --source mirror
    fi
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}

- name: Run parity smoke test
  run: |
    ./scripts/parity_smoke.sh \
      models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
      models/tokenizer.json
```

## Performance Implications

### Download Performance
- **Network Latency**: ~10-30s for 10MB tokenizer file over broadband
- **Retry Overhead**: Exponential backoff adds 200ms, 400ms, 800ms per retry
- **Verification Overhead**: <100ms for JSON parsing and vocab size check

### Auto-Discovery Performance
- **Filesystem Checks**: <50ms for 3-level discovery chain
- **GGUF Parsing**: <100ms for embedded tokenizer check (already parsed for model loading)
- **Total Overhead**: <150ms added to inference initialization (negligible)

### Caching Strategy
- **Persistent Tokenizer**: Once downloaded, no re-download unless `--force`
- **CI Caching**: GitHub Actions cache reduces redundant downloads by >95%
- **Bandwidth Efficiency**: Single 10MB download vs repeated fetches

## Success Metrics

1. **Zero-Configuration Success Rate**: >90% of new users can run inference without manual tokenizer setup
2. **CI Reliability**: Parity smoke tests pass consistently with automatic tokenizer provisioning
3. **Error Reduction**: Tokenizer-related support requests reduced by >80%
4. **Documentation Quality**: README quick-start works copy-paste for >95% of users
5. **Performance Overhead**: Auto-discovery adds <200ms to inference initialization

## Conclusion

This specification provides a production-ready solution for automatic LLaMA-3 tokenizer fetching and auto-discovery in BitNet.rs. The architecture balances user experience (zero-configuration quick start), flexibility (official vs mirror sources), and reliability (fail-fast with clear errors).

The implementation maintains backward compatibility while enabling the "copy-paste works" README experience, reducing onboarding friction and improving production readiness for BitNet.rs neural network inference workflows.

## Next Steps

**FINALIZE → spec-analyzer** for requirements validation and technical feasibility assessment.
