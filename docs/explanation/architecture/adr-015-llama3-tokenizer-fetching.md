# ADR-015: LLaMA-3 Tokenizer Automatic Fetching and Discovery

## Status
**Proposed** - Architectural Decision Record for LLaMA-3 Tokenizer Feature

## Context

The Microsoft BitNet 2B4T GGUF model uses the LLaMA-3 tokenizer (vocab size: 128,256) but does not embed it in the GGUF file. This creates significant friction for new users:

**Current State Pain Points**:
- Manual tokenizer download required before inference
- README quick-start examples fail without additional setup
- CI workflows require manual tokenizer provisioning
- No clear guidance on where to obtain compatible tokenizers
- Enterprise deployments blocked by licensing/auth requirements

**Neural Network Inference Context**:
- **Pipeline Stage**: Tokenization occurs before inference (text → tokens → model)
- **Criticality**: Inference cannot proceed without valid tokenizer
- **Frequency**: One-time setup per model/deployment
- **Performance**: Tokenizer loading is not latency-critical (<100ms acceptable)

**Existing Solutions**:
- Manual wget/curl from HuggingFace (fragile, undocumented)
- Embedded tokenizers in GGUF (not available for BitNet 2B4T)
- SentencePiece models (different format, limited compatibility)

## Decision

We will implement a **two-component automatic tokenizer provisioning system**:

### 1. xtask Tokenizer Subcommand (One-Time Provisioning)
```bash
cargo run -p xtask -- tokenizer --into models/ --source mirror|official
```

**Responsibilities**:
- Download `tokenizer.json` from HuggingFace repositories
- Verify file integrity and vocab size (~128,256)
- Support both official (requires auth) and mirror (no auth) sources
- Provide progress indication and clear error messages

**Rationale**:
- **Explicit Control**: Users choose when to download (no surprise network calls during inference)
- **Source Flexibility**: Enterprise deployments can use internal mirrors
- **Fail-Fast**: Network/auth errors detected early, not during inference
- **CI-Friendly**: Deterministic provisioning step in workflows

### 2. CLI Auto-Discovery (Zero-Configuration Usage)
```rust
// Discovery chain (first match wins):
1. GGUF embedded tokenizer (tokenizer.ggml.model)
2. Sibling tokenizer.json (same directory as model)
3. Parent directory tokenizer.json (one level up)
```

**Responsibilities**:
- Automatically locate tokenizer when `--tokenizer` flag omitted
- Fail with actionable error if not found (no mock fallbacks)
- Log discovery attempts at debug level
- Preserve explicit `--tokenizer` flag for advanced users

**Rationale**:
- **Zero-Configuration Goal**: "Copy-paste works" README experience
- **Backward Compatible**: Explicit flag still works (no breaking changes)
- **Predictable**: Well-defined discovery order, no hidden heuristics
- **Production-Ready**: Fails fast with clear guidance, no silent degradation

## Architecture Decisions

### Decision 1: xtask vs CLI Subcommand

**Choice**: Implement tokenizer fetching as `cargo run -p xtask -- tokenizer` (not `bitnet-cli tokenizer`)

**Rationale**:
- **Consistency**: Aligns with existing `download-model` xtask pattern
- **Separation of Concerns**: xtask handles one-time setup, CLI handles runtime inference
- **No Runtime Network**: Keeps CLI focused on inference operations (no surprise downloads)
- **Developer Workflow**: xtask is established pattern for developer tooling

**Alternatives Considered**:

| Alternative | Pros | Cons | Decision |
|------------|------|------|----------|
| CLI subcommand | Single binary distribution | Blurs setup vs runtime, increases CLI surface area | ❌ Rejected |
| Shell script | Portable, simple | No Rust integration, harder to maintain, platform-specific | ❌ Rejected |
| Automatic download on first run | Zero setup steps | Network dependency during inference, surprising behavior | ❌ Rejected |
| Python helper script | Cross-platform | Adds Python dependency, breaks Rust-native tooling | ❌ Rejected |

**Trade-offs**:
- ✅ Clear separation between setup (xtask) and runtime (CLI)
- ✅ Consistent with existing architecture patterns
- ✅ No runtime network dependencies
- ❌ Requires two commands (provision + infer) instead of one
- ❌ Slightly more complex initial setup

**Mitigation**: Comprehensive documentation and README quick-start examples minimize setup friction.

### Decision 2: Official vs Mirror Default Source

**Choice**: Default to `--source mirror` (baseten/Meta-Llama-3-tokenizer, no auth required)

**Rationale**:
- **Zero-Friction**: New users can run commands without HF_TOKEN setup
- **CI-Friendly**: Public CI environments work without secrets
- **Licensing**: Mirror uses permissive Apache 2.0 / MIT license
- **Fallback**: Official source available via explicit `--source official` flag

**Alternatives Considered**:

| Alternative | Pros | Cons | Decision |
|------------|------|------|----------|
| Default to official | Authoritative source | Requires license acceptance, HF_TOKEN friction | ❌ Rejected |
| Prompt user for source | Explicit choice | Breaks automation, adds interactivity | ❌ Rejected |
| Try official then mirror | Automatic fallback | Network latency, confusing auth errors | ❌ Rejected |
| No default (require flag) | Explicit control | Poor user experience, friction | ❌ Rejected |

**Source Comparison**:

| Aspect | Official (meta-llama) | Mirror (baseten) |
|--------|---------------------|------------------|
| **Authentication** | Requires HF_TOKEN | None |
| **License** | Meta LLaMA-3 Community License | Apache 2.0 / MIT |
| **Acceptance Required** | Yes (on HuggingFace) | No |
| **Update Frequency** | Immediate | May lag official |
| **Enterprise Use** | Licensing review needed | Permissive |
| **CI Integration** | Requires secrets management | Works without secrets |

**Trade-offs**:
- ✅ Best default for new users and CI
- ✅ Permissive licensing for enterprise
- ✅ No authentication friction
- ❌ Mirror may lag behind official releases (acceptable for stable LLaMA-3)
- ❌ Not "official" source (mitigated by making official source available)

**Mitigation**: Clear documentation on using official source, mirror adequacy for stable tokenizers.

### Decision 3: Auto-Discovery vs Explicit Paths Only

**Choice**: Implement auto-discovery with fail-fast error messages, preserve explicit `--tokenizer` flag

**Rationale**:
- **Zero-Configuration**: Enables "copy-paste works" README examples
- **Backward Compatible**: Explicit flag preserves existing workflows
- **Fail-Fast**: Clear errors better than silent mock fallbacks
- **Production-Ready**: Predictable discovery order, actionable guidance

**Discovery Order Justification**:

| Priority | Location | Rationale |
|----------|----------|-----------|
| 1 | GGUF embedded | Authoritative (if present), no filesystem search |
| 2 | Sibling | Most common layout, minimal directory traversal |
| 3 | Parent | Shared tokenizer for multiple models in subdirectories |

**Alternatives Considered**:

| Alternative | Pros | Cons | Decision |
|------------|------|------|----------|
| Require explicit path always | Simple, predictable | Breaks zero-config goal, poor UX | ❌ Rejected |
| Automatic download on missing | Zero setup | Network dependency during inference, surprising | ❌ Rejected |
| Silent mock fallback | Never fails | Hides real issues, production unreliable | ❌ Rejected |
| Search entire models/ tree | Finds all tokenizers | Slow, ambiguous, unpredictable | ❌ Rejected |

**Trade-offs**:
- ✅ Best user experience for new users
- ✅ Backward compatible with explicit flag
- ✅ Clear error messages with actionable guidance
- ❌ More complex implementation (discovery chain logic)
- ❌ Potential ambiguity if multiple tokenizers present

**Mitigation**: Well-defined priority order, debug logging for discovery attempts, explicit flag for advanced users.

## Performance Implications

### Neural Network Pipeline Integration

**Tokenizer Provisioning (xtask)**:
- **Network Latency**: ~10-30s for 10MB file over broadband
- **Verification**: <100ms for JSON parsing and vocab size check
- **Impact**: One-time cost, amortized across all inference runs
- **Optimization**: Cache integration in CI (>95% cache hit rate)

**Auto-Discovery (CLI)**:
- **GGUF Check**: <50ms (already parsed for model loading)
- **Filesystem Checks**: <50ms for 2 additional path lookups
- **Verification**: <100ms for tokenizer loading (already required)
- **Total Overhead**: <200ms added to inference initialization (negligible)

**Inference Pipeline**:
```
Model Load (500ms) → [Tokenizer Discovery (200ms)] → Tokenizer Load (100ms) → Inference (variable)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                      New overhead (one-time per session)
```

### Quantization Requirements

**Tokenizer Impact on Quantization**:
- **Vocab Size**: 128,256 tokens for LLaMA-3 (requires sufficient embedding layer support)
- **Token Range**: 0-128,255 (must fit in u32 token IDs)
- **I2S Quantization**: Vocab size affects embedding layer dimensions, not quantization format
- **Cross-Validation**: Tokenizer must match C++ reference for parity validation

**Performance Metrics**:
- Tokenization overhead: <5% of total inference time
- Token encoding: O(1) per token with BPE (acceptable for 128K vocab)
- Memory footprint: ~10MB for tokenizer model (negligible vs ~2GB GGUF model)

## Implementation Risk Mitigation

### Risk 1: Network Dependency During Setup

**Risk**: Tokenizer download failures block setup
**Probability**: Medium (network issues common in CI)
**Impact**: High (blocks all inference workflows)

**Mitigation Strategy**:
1. **Retry Logic**: 3 attempts with exponential backoff (200ms, 400ms, 800ms)
2. **Mirror Fallback**: Default to no-auth mirror for CI reliability
3. **Cache Integration**: GitHub Actions cache reduces redundant downloads
4. **Clear Errors**: Network failures provide actionable troubleshooting steps

**Validation**:
```bash
# CI test with network simulation
BITNET_SIMULATE_NETWORK_ERROR=1 cargo run -p xtask -- tokenizer --into models/
# Should fail with retry attempts and clear guidance
```

### Risk 2: Tokenizer Version Skew

**Risk**: Mirror tokenizer diverges from official (e.g., bug fixes, updates)
**Probability**: Low (LLaMA-3 tokenizer is stable)
**Impact**: Medium (potential inference quality issues)

**Mitigation Strategy**:
1. **Verification**: Vocab size check catches major incompatibilities
2. **Documentation**: Clear guidance on using official source when needed
3. **Monitoring**: Cross-validation tests detect inference quality issues
4. **Version Pinning**: Future enhancement to pin tokenizer versions

**Validation**:
```bash
# Download both sources and compare
cargo run -p xtask -- tokenizer --into /tmp/official --source official
cargo run -p xtask -- tokenizer --into /tmp/mirror --source mirror
diff /tmp/official/tokenizer.json /tmp/mirror/tokenizer.json
```

### Risk 3: Auth Token Security

**Risk**: HF_TOKEN leaked in logs, CI artifacts, or error messages
**Probability**: Low (standard security practices)
**Impact**: High (unauthorized access to HuggingFace account)

**Mitigation Strategy**:
1. **Never Log Token**: Token only used in HTTP headers, never logged
2. **CI Secrets**: GitHub secrets management (masked in logs)
3. **Error Messages**: Auth errors never include token value
4. **Documentation**: Clear security guidance for token management

**Implementation**:
```rust
// Safe token handling - never log token value
let token = std::env::var("HF_TOKEN")?;
let request = client.get(url)
    .header("Authorization", format!("Bearer {}", token)); // Not logged
tracing::debug!("Requesting tokenizer from {}", url); // URL only, no token
```

## Integration Points

### xtask Integration

**File**: `xtask/src/main.rs`

**Changes**:
```rust
#[derive(Subcommand)]
enum Cmd {
    // ... existing commands ...

    /// Download LLaMA-3 tokenizer from HuggingFace
    Tokenizer {
        #[arg(long, default_value = "models")]
        into: PathBuf,
        #[arg(long, default_value = "mirror")]
        source: String,
        #[arg(long, default_value_t = false)]
        force: bool,
        #[arg(short, long, default_value_t = false)]
        verbose: bool,
    },
}
```

**Dependencies** (already in workspace):
- `reqwest = { version = "0.12", features = ["blocking", "rustls-tls"] }`
- `indicatif = "0.18"` (progress bars)
- `anyhow = "1.0"` (error handling)

### CLI Integration

**File**: `crates/bitnet-cli/src/main.rs`

**Changes**:
```rust
// Make --tokenizer optional
#[arg(long)]
tokenizer: Option<PathBuf>,

// Add auto-discovery before inference
fn resolve_tokenizer(model_path: &Path, explicit_path: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = explicit_path {
        return Ok(path); // Backward compatibility
    }
    TokenizerDiscovery::new(model_path.to_path_buf()).discover()
}
```

**Dependencies**: None (uses existing `bitnet-tokenizers::loader`)

### CI Integration

**File**: `.github/workflows/parity-smoke.yml`

**Changes**:
```yaml
- name: Fetch LLaMA-3 tokenizer
  run: |
    if [ -n "${{ secrets.HF_TOKEN }}" ]; then
      cargo run -p xtask -- tokenizer --into models/ --source official
    else
      cargo run -p xtask -- tokenizer --into models/ --source mirror
    fi
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}

- name: Run parity smoke test
  run: ./scripts/parity_smoke.sh models/model.gguf models/tokenizer.json
```

**Cache Configuration**:
```yaml
- name: Cache tokenizers
  uses: actions/cache@v3
  with:
    path: models/tokenizer.json
    key: ${{ runner.os }}-llama3-tokenizer-${{ hashFiles('xtask/src/main.rs') }}
```

## Success Metrics

**Quantitative Goals**:
1. **Zero-Configuration Success Rate**: >90% of new users run inference without manual tokenizer setup
2. **CI Reliability**: Parity smoke tests pass consistently with automatic provisioning (>99% success rate)
3. **Error Reduction**: Tokenizer-related support requests reduced by >80%
4. **Performance Overhead**: Auto-discovery adds <200ms to inference initialization
5. **Documentation Quality**: README quick-start works copy-paste for >95% of users

**Qualitative Goals**:
- Clear, actionable error messages for all failure modes
- Predictable behavior across different deployment environments
- Backward compatible with existing workflows
- Enterprise-ready (licensing, auth, air-gapped support)

**Validation Plan**:
- **User Testing**: 10 new users attempt README quick-start (target: 9/10 succeed)
- **CI Monitoring**: Track parity test success rate over 30 days (target: >99%)
- **Performance Benchmarks**: Measure auto-discovery latency (target: <200ms P95)
- **Error Analysis**: Review support tickets for tokenizer issues (target: 80% reduction)

## Alternatives Rejected

### Alternative 1: Automatic Download During Inference

**Description**: CLI automatically downloads tokenizer on first inference run if missing

**Pros**:
- Single command for setup + inference
- Zero explicit provisioning steps
- "Just works" experience

**Cons**:
- Network dependency during inference (surprising behavior)
- Confusing errors if network unavailable during inference
- Breaks deterministic builds and air-gapped deployments
- No control over when network access occurs

**Decision**: ❌ Rejected - Network calls during inference violate fail-fast principle and break determinism

### Alternative 2: Bundled Tokenizer in Binary

**Description**: Embed LLaMA-3 tokenizer in bitnet-cli binary at compile time

**Pros**:
- Zero setup, zero network
- Guaranteed availability
- Simplest user experience

**Cons**:
- Increases binary size by ~10MB (significant for CLI)
- Licensing concerns (Meta LLaMA-3 license vs MIT/Apache)
- Inflexible (cannot use custom tokenizers)
- Cannot update tokenizer without recompiling

**Decision**: ❌ Rejected - Licensing issues and inflexibility outweigh convenience

### Alternative 3: Python Wrapper Script

**Description**: Provide Python script to download tokenizer using `huggingface-hub` library

**Pros**:
- Leverages existing HuggingFace tooling
- Cross-platform
- Simple implementation

**Cons**:
- Adds Python dependency (breaks Rust-native tooling)
- Inconsistent with BitNet.rs architecture
- Requires separate distribution and maintenance
- Confusing for users ("Is BitNet.rs Rust or Python?")

**Decision**: ❌ Rejected - Breaks Rust-native architecture, adds unnecessary dependency

### Alternative 4: Document Manual Download Only

**Description**: Keep current state, improve documentation for manual wget/curl

**Pros**:
- No implementation effort
- Maximum flexibility
- No new dependencies

**Cons**:
- Poor user experience (manual steps error-prone)
- Breaks "copy-paste works" goal
- CI workflows remain fragile
- No progress on zero-configuration vision

**Decision**: ❌ Rejected - Does not address core pain points, poor UX

## Conclusion

This Architecture Decision Record establishes automatic LLaMA-3 tokenizer fetching and discovery as a foundational capability for BitNet.rs production readiness. The two-component approach (explicit provisioning + automatic discovery) balances user experience, reliability, and architectural consistency.

**Key Architectural Principles**:
1. **Explicit Provisioning**: One-time `xtask tokenizer` command for setup
2. **Automatic Discovery**: Zero-configuration inference when tokenizer present
3. **Fail-Fast**: Clear errors with actionable guidance, no silent degradation
4. **Backward Compatible**: Explicit `--tokenizer` flag preserved
5. **Enterprise-Ready**: Flexible sources (official/mirror), licensing clarity

This architecture aligns with BitNet.rs values of developer experience, production reliability, and zero-configuration workflows while maintaining architectural consistency with existing patterns (xtask for setup, CLI for runtime).

**Next Steps**:
1. Implementation in phases: xtask subcommand → CLI discovery → CI integration
2. Comprehensive testing: unit, integration, CI workflows
3. Documentation updates: README, quickstart, troubleshooting
4. User validation: feedback from community on UX and error messages
