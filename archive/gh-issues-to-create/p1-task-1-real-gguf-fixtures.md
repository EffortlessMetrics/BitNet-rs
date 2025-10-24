# Issue: Real GGUF Fixtures for QK256 Testing

## Context

BitNet.rs currently generates GGUF test fixtures in-memory during test execution, adding ~50-100ms overhead per test and reducing CI/CD stability. Following the successful merge of PR #475 (comprehensive integration with QK256 AVX2, EnvGuard isolation, receipts, and strict mode), we need persistent disk-based fixtures for stable, version-controlled testing.

**Affected Components:**
- `bitnet-models/tests/qk256_dual_flavor_tests.rs` - Currently uses in-memory fixture generation
- `ci/fixtures/` - New directory for persistent GGUF test files
- CI/CD pipelines - Will benefit from faster, more stable test execution

**Inference Pipeline Impact:**
- Model Loading stage - Fixtures validate GGUF v3 compliance and alignment
- Quantization stage - Validates QK256 and BitNet32F16 dual-flavor detection

**Performance Implications:**
- CI speedup: Eliminate ~150ms per test run (3 tests × ~50ms generation overhead)
- Determinism: Fixtures produce identical results across platforms (x86_64, ARM64, WASM)
- Storage: Minimal footprint (< 10KB total for 3 fixtures)

## User Story

As a developer, I want minimal GGUF test files for QK256/BitNet-32 testing so that CI/CD pipelines have stable, version-controlled fixtures without runtime generation overhead.

## Acceptance Criteria

AC1: Generate 3 persistent GGUF fixtures (4×256, 3×300, 2×64) and store in `ci/fixtures/qk256/` directory
AC2: Update `qk256_dual_flavor_tests.rs` to load fixtures from disk instead of generating in-memory
AC3: Validate all fixtures pass `bitnet-cli compat-check` command with zero errors
AC4: Add SHA256 checksums to `ci/fixtures/qk256/checksums.txt` for fixture integrity verification
AC5: Document fixture format, regeneration process, and alignment requirements in `ci/fixtures/qk256/README.md`
AC6: Ensure all 12/12 tests in `qk256_dual_flavor_tests.rs` pass with disk-based fixture loading
AC7: Verify fixtures load identically on x86_64, ARM64, and WASM targets (cross-platform validation)
AC8: Create regeneration script `scripts/regenerate-fixtures.sh` for reproducible fixture updates

## Technical Implementation Notes

- **Affected crates**: `bitnet-models` (test infrastructure), `ci/fixtures/` (new directory structure)
- **Pipeline stages**: Model loading, quantization flavor detection (QK256 vs BitNet32F16)
- **Performance considerations**:
  - Eliminate ~150ms CI overhead per test run
  - Fixtures < 50KB total (4×256 ≈ 2KB, 3×300 ≈ 3KB, 2×64 ≈ 1KB)
  - Deterministic results across platforms
- **Quantization requirements**:
  - QK256: 256 elements → 64 bytes packed (2-bit quantization)
  - BitNet32F16: 32 elements + F16 scale → 10 bytes per block
  - GGUF v3 alignment: 32-byte boundary for tensor data
- **Cross-validation**: Fixtures enable consistent C++/Rust parity testing via `cargo run -p xtask -- crossval`
- **Feature flags**: Uses existing `fixtures` feature - no new flags required
- **GGUF compatibility**:
  - GGUF v3 magic: `GGUF` (0x46554747)
  - Version: 3 (little-endian u32)
  - Tensor alignment: 32-byte boundary
  - Required metadata KV pairs: 8 keys (general.name, tokenizer.ggml.tokens, bitnet-b1.58.embedding_length, etc.)
- **Testing strategy**:
  - TDD with `// AC:ID` tags for each acceptance criterion
  - CPU smoke testing: `cargo test -p bitnet-models --features cpu,fixtures test_qk256_detection_by_size`
  - Alignment validation: Existing `helpers/alignment_validator.rs` tests against disk fixtures
  - CI integration: Add fixture existence check in `.github/workflows/ci.yml`
  - Strict mode validation: `BITNET_STRICT_MODE=1 cargo run -p bitnet-cli --features cpu,full-cli -- inspect --ln-stats --gate auto ci/fixtures/qk256/qk256_4x256_seed42.gguf`

**Validation Commands:**
```bash
# Generate fixtures (one-time script)
cargo run -p bitnet-models --test qk256_dual_flavor_tests --features fixtures -- test_dump_fixture_for_debug --nocapture

# Validate with bitnet-cli
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- compat-check ci/fixtures/qk256/qk256_4x256_seed42.gguf

# Verify SHA256 checksums
sha256sum -c ci/fixtures/qk256/checksums.txt

# Run all fixture-based tests
cargo test -p bitnet-models --features cpu,fixtures test_qk256_detection_by_size
cargo nextest run -p bitnet-models --features cpu,fixtures --profile ci
```

**Estimate**: 2-3 hours

---

<!-- gates:start -->
| Gate | Status | Evidence |
|------|--------|----------|
| spec | ✅ pass | Feature spec created in docs/explanation/specs/SPEC-2025-001-next-steps-priority-implementation.md (Story 1) |
| format | pending | Code formatting with cargo fmt --all --check |
| clippy | pending | Linting with cargo clippy --workspace --all-targets --no-default-features --features cpu -- -D warnings |
| tests | pending | TDD scaffolding with cargo test -p bitnet-models --features cpu,fixtures |
| build | pending | Build validation with cargo build --release --no-default-features --features cpu |
| features | pending | Feature smoke testing: cpu+fixtures combo |
| benchmarks | pending | CI speedup validation (target: 150ms reduction) |
| docs | pending | Documentation updates in ci/fixtures/qk256/README.md |
<!-- gates:end -->

<!-- hoplog:start -->
### Hop log
- Created feature spec: Story 1 in docs/explanation/specs/SPEC-2025-001-next-steps-priority-implementation.md
<!-- hoplog:end -->

<!-- decision:start -->
**State:** in-progress
**Why:** Feature spec created and validated, ready for implementation
**Next:** NEXT → implementation with TDD workflow (AC1-AC8 test scaffolding)
<!-- decision:end -->
