# PR1 Fixture Implementation - Quick Reference

## TL;DR Summary

**Status**: 🟢 Infrastructure ready - only test gating needed

**The Good News**:
1. ✅ QK256 fixture generators are **already complete and working**
2. ✅ No additional crate dependencies needed
3. ✅ No modifications to st2gguf writer required
4. ✅ Tempfile already in dev-dependencies

**What's Needed**:
1. Add `fixtures = []` feature to Cargo.toml (2 lines)
2. Add `#[cfg(feature = "fixtures")]` to 3 tests (3 additions)
3. Optional: Update CI workflows

**Implementation Time**: ~30 minutes

---

## Quick Checklist

### Phase 1: Feature Flag (5 min)
```bash
# Edit: crates/bitnet-models/Cargo.toml
[features]
fixtures = []  # ADD THIS LINE
```

### Phase 2: Gate Tests (5 min)
```bash
# Edit: crates/bitnet-models/tests/qk256_dual_flavor_tests.rs
# Add to 3 tests:
#[cfg(feature = "fixtures")]
fn test_qk256_detection_by_size() { ... }
#[cfg(feature = "fixtures")]
fn test_bitnet32_still_uses_fp_path() { ... }
#[cfg(feature = "fixtures")]
fn test_qk256_with_non_multiple_cols() { ... }
```

### Phase 3: Verify (10 min)
```bash
# Without fixtures (should skip 3 tests)
cargo test -p bitnet-models qk256_dual_flavor_tests --no-default-features --features cpu

# With fixtures (should run all 6 tests)
cargo test -p bitnet-models qk256_dual_flavor_tests --no-default-features --features cpu,fixtures

# Verify other QK256 tests still pass
cargo test -p bitnet-models qk256_ --no-default-features --features cpu
```

---

## Key Files Overview

### Already Complete ✅

| File | Status | Reason |
|------|--------|--------|
| `tests/helpers/qk256_fixtures.rs` | ✅ Ready | Generates 3 fixture types (QK256 4×256, BitNet32 2×64, QK256 3×300) |
| `tests/helpers/mod.rs` | ✅ Ready | Already re-exports fixtures |
| `tests/qk256_fixture_validation.rs` | ✅ Ready | Tests fixture generators |

### Need Modification ⚙️

| File | Changes |
|------|---------|
| `Cargo.toml` | Add `fixtures = []` feature |
| `tests/qk256_dual_flavor_tests.rs` | Add `#[cfg(feature = "fixtures")]` to 3 tests |

### Optional Updates 📋

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | Add job for fixture tests |
| `docs/test-suite.md` | Document fixtures feature |

---

## Fixture Generation Functions

All three already exist and work perfectly:

### 1. `generate_qk256_4x256(seed: u64) -> Vec<u8>`
- Single-block QK256 (4 rows × 256 cols)
- File size: ~900 bytes
- Test: `test_qk256_detection_by_size`

### 2. `generate_bitnet32_2x64(seed: u64) -> Vec<u8>`
- BitNet32F16 two-block (2 rows × 64 cols)
- File size: ~700 bytes
- Test: `test_bitnet32_still_uses_fp_path`

### 3. `generate_qk256_3x300(seed: u64) -> Vec<u8>`
- QK256 multi-block with tail (3 rows × 300 cols)
- File size: ~900 bytes
- Test: `test_qk256_with_non_multiple_cols`

---

## Dependency Status

✅ **No new dependencies needed**

| Crate | Version | Used For | Status |
|-------|---------|----------|--------|
| `tempfile` | 3.23.0 | Test file I/O | Already in dev-deps |
| `std::io` | stdlib | I/O traits | Built-in |

---

## Test Gating Strategy

### Tests that NEED gating (3 tests)
```rust
#[test]
#[cfg(feature = "fixtures")]  // ← ADD THIS
fn test_qk256_detection_by_size() { ... }
```

**Why**: Require GGUF file generation and I/O

### Tests that DON'T need gating (3 tests)
```rust
#[test]  // ← No gate needed
fn test_qk256_i2s_qk256_noscale_creation() { ... }
```

**Why**: Direct struct construction, no GGUF files

### Other test files (no changes needed)
- `qk256_error_handling.rs` - 40+ tests, all pass ✅
- `qk256_integration.rs` - 15+ tests, all pass ✅

---

## Running Tests

### Default (no fixtures)
```bash
# Skips fixture-gated tests
cargo test -p bitnet-models --no-default-features --features cpu qk256
```

### With fixtures
```bash
# Includes fixture-gated tests
cargo test -p bitnet-models --no-default-features --features cpu,fixtures qk256
```

### Specific tests
```bash
# Non-fixture tests only
cargo test test_qk256_i2s_qk256_noscale_creation
cargo test test_qk256_size_mismatch_error

# Fixture tests only (with features)
cargo test --features fixtures test_qk256_detection_by_size
```

---

## Why This Design

### Feature Flag Advantages
- ✅ Zero runtime overhead
- ✅ Opt-in testing (not required for basic build)
- ✅ Clean separation of concerns
- ✅ Follows existing codebase patterns
- ✅ Easy to extend for future fixtures

### No st2gguf Writer Changes Needed
- st2gguf is for SafeTensors → GGUF conversion (production)
- Test fixtures use simpler, inline generator (development)
- qk256_fixtures.rs is purpose-built for test data

### Zero New Dependencies
- Already have tempfile
- Byte serialization uses `to_le_bytes()` (stdlib)
- Everything else is basic Vec/I/O operations

---

## Potential Issues & Solutions

| Issue | Solution |
|-------|----------|
| Tests still marked `#[ignore]` | Will be fixed by feature gating |
| GgufReader rejects fixtures | Already solved (qk256_fixtures.rs includes all required metadata) |
| Feature not in Cargo.toml | Add in Phase 1 |
| Tests expect old fixture format | qk256_fixtures.rs is already updated |

---

## Integration Timeline

```
Phase 1: Feature Flag Setup       [5 min]
  └─ Add fixtures = [] to Cargo.toml

Phase 2: Test Gating             [5 min]
  └─ Add #[cfg(feature = "fixtures")] to 3 tests

Phase 3: Verification            [10 min]
  └─ Run tests with/without features

Phase 4: CI Update (optional)     [10 min]
  └─ Update workflows

Phase 5: Documentation (optional) [5 min]
  └─ Update README/docs

TOTAL: ~35 minutes (including verification)
```

---

## Success Criteria

- [ ] `cargo test -p bitnet-models` passes (without fixtures)
- [ ] `cargo test -p bitnet-models --features fixtures` passes (with fixtures)
- [ ] Fixture-dependent tests only run with `--features fixtures`
- [ ] All 6 tests in qk256_dual_flavor_tests.rs pass with features
- [ ] Other QK256 tests unaffected
- [ ] No new warnings or clippy issues
- [ ] Documentation updated

---

## File Locations Reference

**Fixture Generators**:
- `/crates/bitnet-models/tests/helpers/qk256_fixtures.rs` (main)
- `/crates/bitnet-models/tests/helpers/mod.rs` (exports)
- `/crates/bitnet-models/tests/qk256_fixture_validation.rs` (validation)

**Test Files**:
- `/crates/bitnet-models/tests/qk256_dual_flavor_tests.rs` (target for gating)
- `/crates/bitnet-models/tests/qk256_error_handling.rs` (no changes)
- `/crates/bitnet-models/tests/qk256_integration.rs` (no changes)

**Configuration**:
- `/crates/bitnet-models/Cargo.toml` (add feature)
- `/.github/workflows/ci.yml` (optional update)

---

## Full Documentation

For complete details, methodology, and reference material, see:
- `PR1_fixture_implementation_plan.md` (1000+ lines, comprehensive analysis)

For quick answers, refer back to this document.

