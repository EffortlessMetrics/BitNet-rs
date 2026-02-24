# BitNet-rs v0.1.0 Release Checklist

**Release Date Target:** TBD
**Release Type:** Initial MVP Release
**MSRV:** Rust 1.90.0 (Edition 2024)

---

## Pre-Release Preparation ✓ (COMPLETED)

### Phase 0: Packaging Hygiene

- [x] **Internal crates marked as `publish = false`**
  - `bitnet-tests` (tests/)
  - `bitnet-compat` (crates/bitnet-compat/)
  - `bitnet-st-tools` (crates/bitnet-st-tools/)
  - `bitnet-sys` (crates/bitnet-sys/)
  - `xtask` (xtask/)
  - `xtask-build-helper` (xtask-build-helper/)
  - `crossval` (crossval/)
  - `fuzz` (fuzz/)

- [x] **Workspace metadata added to FFI crates**
  - `bitnet-ggml-ffi`: Added authors, keywords, categories, homepage, documentation
  - `bitnet-py`: Added authors, keywords, categories, homepage, documentation

- [x] **Include patterns added to all publishable crates**
  - Root `bitnet`: src/**, Cargo.toml, build.rs, README.md, LICENSE
  - `bitnet-common`: src/**, Cargo.toml, WARN_ONCE_README.md
  - `bitnet-quantization`: src/**, Cargo.toml
  - `bitnet-models`: src/**, Cargo.toml
  - `bitnet-kernels`: src/**, Cargo.toml, build.rs, README.md, csrc/**
  - `bitnet-inference`: src/**, Cargo.toml
  - `bitnet-tokenizers`: src/**, Cargo.toml, docs/**, TEST_COVERAGE_SUMMARY.md
  - `bitnet-ggml-ffi`: src/**, Cargo.toml, build.rs, README.md, csrc/**
  - `bitnet-cli`: src/**, Cargo.toml
  - `bitnet-server`: src/**, Cargo.toml, build.rs, HEALTH_ENDPOINTS.md
  - `bitnet-st2gguf`: src/**, Cargo.toml

- [x] **Version specifications added to internal path dependencies**
  - 36 internal dependencies updated with `version = "0.1.0"`
  - All publishable crates verified

---

## Phase 1: Pre-Release Verification (Run Before Tagging)

### 1.1 Clean Build and Test Suite

```bash
# Clean workspace
cargo clean

# Build all publishable crates with CPU features
cargo build --workspace --no-default-features --features cpu --release

# Run full test suite with nextest (recommended)
cargo nextest run --workspace --no-default-features --features cpu --profile ci

# Run doctests
cargo test --doc --workspace --no-default-features --features cpu

# Check for compilation warnings
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

**Expected:** All tests pass, no clippy warnings.

---

### 1.2 Verify Package Contents

Test that all publishable crates package correctly without bloated content:

```bash
# Core libraries (Phase 1 publication priority)
cargo package -p bitnet-ggml-ffi --list --allow-dirty | wc -l  # Expect ~14 files
cargo package -p bitnet-common --list --allow-dirty | wc -l     # Expect ~14 files
cargo package -p bitnet-kernels --list --allow-dirty | wc -l    # Expect ~30 files
cargo package -p bitnet-quantization --list --allow-dirty | wc -l
cargo package -p bitnet-models --list --allow-dirty | wc -l
cargo package -p bitnet-inference --list --allow-dirty | wc -l
cargo package -p bitnet-tokenizers --list --allow-dirty | wc -l
cargo package -p bitnet --list --allow-dirty | wc -l

# Tools (Phase 2 publication)
cargo package -p bitnet-cli --list --allow-dirty | wc -l
cargo package -p bitnet-server --list --allow-dirty | wc -l
cargo package -p bitnet-st2gguf --list --allow-dirty | wc -l
```

**Expected:** File counts should be minimal (<50 files for most crates), no test fixtures, no CI artifacts.

---

### 1.3 Verify Package Sizes

```bash
# Check compressed package sizes (should be <1MB for most crates)
for pkg in bitnet-ggml-ffi bitnet-common bitnet-kernels bitnet-quantization bitnet-models bitnet-inference bitnet-tokenizers bitnet-cli bitnet-server bitnet-st2gguf bitnet; do
  echo "=== $pkg ==="
  cargo package -p $pkg --allow-dirty --no-verify 2>&1 | grep "Packaged"
done
```

**Expected:** All crates <1MB compressed (FFI crates may be larger due to C sources).

---

### 1.4 Quality Gates

```bash
# Lychee: Check documentation links
lychee --config .lychee.toml "README.md" "docs/**/*.md"

# Cargo deny: Check dependencies for security/license issues
cargo deny check advisories bans licenses sources

# Fixture integrity
( cd ci/fixtures/qk256 && sha256sum -c SHA256SUMS )

# Format check
cargo fmt --all -- --check
```

**Expected:** All links valid, no denied dependencies, fixtures verified, formatting correct.

---

### 1.5 Documentation Build

Verify that docs.rs will be able to build documentation:

```bash
# Test docs build with all features
RUSTDOCFLAGS="--cfg docsrs" cargo +nightly doc --all-features --no-deps --workspace

# Open and manually verify key crates
cargo doc --open --no-deps -p bitnet
cargo doc --open --no-deps -p bitnet-inference
cargo doc --open --no-deps -p bitnet-kernels
```

**Expected:** Documentation builds without errors, all public APIs documented.

---

## Phase 2: Release Notes Preparation

### 2.1 Update CHANGELOG.md

Move `[Unreleased]` section to `[0.1.0] - 2025-MM-DD`:

```markdown
## [0.1.0] - 2025-MM-DD

### Added
- CPU inference with SIMD optimization (AVX2/AVX-512/NEON)
- GPU inference with CUDA acceleration
- QK256 (GGML I2_S) MVP with scalar kernels
- QK256 AVX2 dequantization foundation (1.2× uplift, targeting ≥3×)
- Interactive chat and Q&A workflows with prompt templates
- Model validation and inspection tools
- Cross-validation framework against C++ reference
- GGUF fixtures & dual-flavor tests (12/12 passing)
- EnvGuard environment isolation for robust parallel test execution
- Receipt verification system (v1.0.0 schema, 25/25 tests passing)
- Strict mode runtime guards for production safety
- SafeTensors to GGUF converter with LayerNorm preservation

### Known Limitations (MVP Phase)
- QK256 performance: AVX2 foundation established (~1.2× uplift); targeting ≥3× with nibble-LUT + FMA optimizations
- Model quality: microsoft-bitnet-b1.58-2B-4T-gguf produces non-sensical output in some configurations (model quality issue, not inference bug)
- Test scaffolding: ~548 TODO/FIXME/unimplemented markers and ~70 ignored tests represent TDD-style scaffolding for planned features
- Active blockers: Issues #254, #260, #469 affect some real inference tests and cross-validation

### Dependencies
- MSRV: Rust 1.90.0 (Edition 2024)
- candle-core 0.9.1
- tokenizers 0.22.1
- cudarc 0.17.3 (GPU builds)
```

---

### 2.2 Create GitHub Release Notes

Draft release notes with:

**Title:** `v0.1.0 - Q&A-Ready MVP`

**Body:**

```markdown
# v0.1.0 - Q&A-Ready MVP

First public release of BitNet.rs, a high-performance Rust implementation of BitNet 1-bit LLM inference.

## Highlights

✅ **CPU Inference** - SIMD-optimized (AVX2/AVX-512/NEON) inference on x86_64 and ARM
✅ **GPU Inference** - CUDA acceleration with mixed precision (FP16/BF16)
✅ **QK256 Support** - GGML I2_S quantization MVP with AVX2 foundation
✅ **Interactive CLI** - Chat and Q&A workflows with prompt template auto-detection
✅ **Model Validation** - 3-stage validation (LayerNorm, projection, linguistic sanity)
✅ **Cross-Validation** - Systematic comparison with Microsoft BitNet C++ reference
✅ **Receipt Verification** - Honest compute gates (schema v1.0.0, 8 validation gates)
✅ **Production Safety** - Strict mode runtime guards and environment isolation

## Quick Start

### Installation

```bash
# From crates.io (recommended)
cargo install bitnet-cli --no-default-features --features cpu,full-cli

# From source
git clone https://github.com/microsoft/BitNet.git
cd BitNet
cargo build --release --no-default-features --features cpu,full-cli
```

### Run Inference

```bash
# Download a model (requires xtask)
cargo run -p xtask -- download-model

# Interactive chat
bitnet chat --model models/model.gguf --tokenizer models/tokenizer.json

# One-shot Q&A
bitnet run --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 32
```

## What's Working

- ✅ CPU inference with SIMD optimization
- ✅ GPU inference with CUDA acceleration
- ✅ QK256 (GGML I2_S) MVP with AVX2 foundation
- ✅ Interactive chat and Q&A workflows
- ✅ Model validation and inspection
- ✅ Cross-validation framework
- ✅ Comprehensive test infrastructure

## Known Limitations (MVP Phase)

⚠️ **QK256 Performance:** AVX2 foundation established (~1.2× uplift); targeting ≥3× with optimizations. For quick validation, limit to `--max-new-tokens 4-16`.

⚠️ **Model Quality:** Some models produce non-sensical output in certain configurations. This is a known model quality issue, not an inference engine bug.

⚠️ **Test Scaffolding:** ~548 TODO/FIXME markers and ~70 ignored tests represent TDD-style scaffolding for planned features. See [Test Status](docs/CLAUDE.md#test-status-mvp-phase) for details.

⚠️ **Active Blockers:** Issues [#254](https://github.com/microsoft/BitNet/issues/254), [#260](https://github.com/microsoft/BitNet/issues/260), [#469](https://github.com/microsoft/BitNet/issues/469) affect some real inference tests and cross-validation.

## Documentation

- 📘 [Getting Started](docs/getting-started.md)
- 📘 [Architecture Overview](docs/architecture-overview.md)
- 📘 [Development Guide](CLAUDE.md)
- 📘 [Model Validation](docs/howto/validate-models.md)
- 📘 [Performance Benchmarking](docs/performance-benchmarking.md)

## Release Artifacts

- **CLI Binaries:** Linux x64 (musl), macOS x64/ARM64, Windows x64
- **Fixture Receipts:** `ci/fixtures/qk256/SHA256SUMS`
- **GGML Commit:** `BITNET_GGML_COMMIT=<sha>` (see CI logs)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for complete list of changes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT OR Apache-2.0
```

---

## Phase 3: Git Tag and Publish

### 3.1 Create Git Tag

**⚠️ ONLY run this when ready to publish!**

```bash
# Ensure all changes are committed
git status  # Should show clean working directory

# Create annotated tag
git tag -a v0.1.0 -m "Release v0.1.0 - Q&A-Ready MVP"

# Verify tag
git show v0.1.0

# Push tag (triggers CI release workflow if configured)
git push origin v0.1.0
```

---

### 3.2 Publish Crates to crates.io

**⚠️ PUBLISH IN DEPENDENCY ORDER!**

Crates must be published in order because dependent crates will look for their dependencies on crates.io:

```bash
# Phase 1: Foundation crates (no internal dependencies)
cargo publish -p bitnet-ggml-ffi
sleep 15  # Wait for crates.io to index
cargo publish -p bitnet-common
sleep 15

# Phase 2: Core libraries (depend on Phase 1)
cargo publish -p bitnet-kernels
sleep 15
cargo publish -p bitnet-quantization
sleep 15
cargo publish -p bitnet-models
sleep 15

# Phase 3: High-level libraries (depend on Phase 1 + 2)
cargo publish -p bitnet-tokenizers
sleep 15
cargo publish -p bitnet-inference
sleep 15

# Phase 4: Root aggregate crate
cargo publish -p bitnet
sleep 15

# Phase 5: Tools (depend on all previous)
cargo publish -p bitnet-cli
sleep 15
cargo publish -p bitnet-server
sleep 15
cargo publish -p bitnet-st2gguf
```

**Notes:**
- Sleep 15-30 seconds between publishes to allow crates.io indexing
- Watch for errors; if a crate fails, fix and retry with patch version bump
- Once published, crates are **immutable** - you cannot unpublish or edit

---

### 3.3 Create GitHub Release

1. Go to https://github.com/microsoft/BitNet/releases/new
2. Select tag: `v0.1.0`
3. Release title: `v0.1.0 - Q&A-Ready MVP`
4. Paste release notes from Section 2.2
5. Attach artifacts:
   - `ci/fixtures/qk256/SHA256SUMS`
   - CLI binaries (if available from CI)
6. Check "Set as the latest release"
7. Click "Publish release"

---

## Phase 4: Post-Release Verification

### 4.1 Verify crates.io Publication

```bash
# Wait 5-10 minutes for crates.io to fully update

# Create clean test project
mkdir /tmp/bitnet-test && cd /tmp/bitnet-test
cargo init --name bitnet-test

# Add bitnet dependency
cargo add bitnet@0.1.0 --no-default-features --features cpu

# Verify it builds
cargo check
cargo build

# Verify CLI installation
cargo install bitnet-cli@0.1.0 --no-default-features --features cpu,full-cli
bitnet --version
bitnet --help
```

**Expected:** All commands succeed, no build errors.

---

### 4.2 Verify docs.rs

Check that documentation built successfully:

1. Visit https://docs.rs/bitnet/0.1.0/bitnet/
2. Verify all modules documented
3. Check feature flags documented
4. Verify examples render correctly

---

### 4.3 Announce Release

- [ ] Update repository README.md to reference v0.1.0
- [ ] Post announcement to project channels (Discord, Reddit, Twitter, etc.)
- [ ] Update project homepage (if applicable)

---

## Phase 5: Rollback Plan (If Needed)

### If a crate fails to publish:

1. **Fix the issue** in the source
2. **Bump the patch version** (e.g., `0.1.0` → `0.1.1`) for just that crate
3. **Update dependent crates** to reference the new patch version
4. **Re-publish** the fixed crate and any dependents
5. **Update release notes** to document the hotfix

### If critical bug found post-release:

1. **Create hotfix branch** from `v0.1.0` tag
2. **Fix the bug** with minimal changes
3. **Bump patch version** to `0.1.1`
4. **Re-run all verification steps** (Phase 1)
5. **Publish hotfix** (Phase 3)
6. **Create GitHub release** for v0.1.1 with hotfix notes

---

## Phase 6: Post-Release Tasks

### 6.1 Update Main Branch

```bash
# Switch back to main
git checkout main

# Merge release tag
git merge v0.1.0

# Push updated main
git push origin main
```

---

### 6.2 Prepare for Next Release

- [ ] Update version in workspace to `0.2.0-dev` (or `0.1.1-dev` for hotfixes)
- [ ] Create `[Unreleased]` section in CHANGELOG.md
- [ ] Close completed milestones on GitHub
- [ ] Create milestone for next release

---

## Checklist Summary

### Pre-Release (COMPLETED ✓)
- [x] Internal crates marked `publish = false`
- [x] Workspace metadata added to FFI crates
- [x] Include patterns added to publishable crates
- [x] Version specifications added to path dependencies

### Phase 1: Pre-Release Verification
- [ ] Clean build and test suite passes
- [ ] Package contents verified (no bloat)
- [ ] Package sizes reasonable (<1MB compressed)
- [ ] Quality gates pass (clippy, deny, lychee, fixtures)
- [ ] Documentation builds without errors

### Phase 2: Release Notes
- [ ] CHANGELOG.md updated
- [ ] GitHub release notes drafted

### Phase 3: Publish
- [ ] Git tag created and pushed
- [ ] Crates published to crates.io (in dependency order)
- [ ] GitHub release created with artifacts

### Phase 4: Post-Release Verification
- [ ] crates.io installation verified
- [ ] docs.rs documentation verified
- [ ] Release announced

### Phase 5: Rollback Plan
- [ ] Documented and ready if needed

### Phase 6: Post-Release Tasks
- [ ] Main branch updated
- [ ] Next release prepared

---

## Publication Order Reference

**Dependency order (must publish in this sequence):**

```
Layer 0 (no internal deps):
  └─ bitnet-ggml-ffi
  └─ bitnet-common

Layer 1 (depend on Layer 0):
  └─ bitnet-kernels → bitnet-common
  └─ bitnet-quantization → bitnet-common, bitnet-kernels
  └─ bitnet-models → bitnet-common, bitnet-quantization, bitnet-ggml-ffi

Layer 2 (depend on Layer 0-1):
  └─ bitnet-tokenizers → bitnet-common, bitnet-models, bitnet-quantization
  └─ bitnet-inference → bitnet-common, bitnet-models, bitnet-tokenizers, bitnet-kernels, bitnet-quantization

Layer 3 (aggregate):
  └─ bitnet → all above

Layer 4 (tools):
  └─ bitnet-cli → all above
  └─ bitnet-server → all above
  └─ bitnet-st2gguf → bitnet-common, bitnet-models
```

---

## Notes

- **Sleep between publishes:** crates.io needs time to index each crate before dependent crates can reference it
- **No dry-run for publish:** `cargo publish` has no `--dry-run` flag; use `cargo package --list` for verification
- **Immutable releases:** Once published, crates cannot be unpublished or edited (only yanked)
- **Version bumps:** If you need to re-publish a crate, you must bump the version number

---

## Contact

For release questions or issues:
- GitHub Issues: https://github.com/microsoft/BitNet/issues
- Release Manager: [Your contact info]

---

**Document Version:** 1.0
**Last Updated:** 2025-10-24
**Status:** Ready for v0.1.0 release execution
